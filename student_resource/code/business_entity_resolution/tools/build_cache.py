#!/usr/bin/env python3
"""Build a normalized cache so every later experiment is fast.

Outputs (pickle):
  cache/norm_{split}_{source}.pkl   -> dict(ids, names, addrs, countries, country_codes)
  cache/gt_train.pkl                -> dict(offsets, flat)  int32 indices into S23 space
  cache/s23_ids_{split}.pkl         -> object array of S2 then S3 ids (index space)

The S23 index space is: [0, n2) = source2 rows, [n2, n2+n3) = source3 rows.

Run:
  python3 tools/build_cache.py --data-dir ../../dataset --out ../../cache
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd

from pipeline import normalize_name, normalize_address, normalize_country


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_and_normalize(path, label):
    t0 = time.time()
    log(f"loading {os.path.basename(path)} ...")
    df = pd.read_csv(path, sep='\t', dtype=str, engine='c',
                     keep_default_na=False, na_filter=False)
    log(f"  {len(df):,} rows in {time.time()-t0:.1f}s")
    t0 = time.time()
    ids = df['entity_id'].to_numpy(dtype=object)
    names = np.array([normalize_name(x) for x in df['business_name']], dtype=object)
    addrs = np.array([normalize_address(x) for x in df['business_address']], dtype=object)
    countries = df['country'].map(normalize_country).to_numpy(dtype=object)
    del df
    log(f"  normalized {label} in {time.time()-t0:.1f}s")
    return dict(ids=ids, names=names, addrs=addrs, countries=countries)


def save(obj, path):
    t0 = time.time()
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    size = os.path.getsize(path) / 1e6
    log(f"  wrote {os.path.basename(path)} ({size:.0f} MB) in {time.time()-t0:.1f}s")


def build_gt(data_dir, out_dir, s23_ids, s1_ids):
    """Ground truth as CSR int32 indices into the S23 index space.

    CRITICAL: the ground-truth file's rows are NOT in the same order as
    train_source1.tsv (verified empirically: row-aligned indexing produced
    pairs with unrelated names/cities).  Everything is therefore keyed by
    entity id and re-indexed onto *S1 row positions*, which is what the
    blockers and the scorer use.
    """
    t0 = time.time()
    gt_path = os.path.join(data_dir, 'train_ground_truth.tsv')
    log(f"loading {os.path.basename(gt_path)} ...")
    df = pd.read_csv(gt_path, sep='\t', dtype=str, engine='c',
                     keep_default_na=False, na_filter=False)
    log(f"  {len(df):,} rows in {time.time()-t0:.1f}s")
    if list(df.columns) != ['source1_entity_id', 'matched_entity_ids']:
        raise SystemExit(f"unexpected gt columns: {list(df.columns)}")

    gt_ids = df['source1_entity_id'].to_numpy(dtype=object)
    raw = df['matched_entity_ids'].to_numpy(dtype=object)

    # gt rows -> S1 row positions
    n1 = len(s1_ids)
    pos = pd.Index(s1_ids).get_indexer(gt_ids)
    n_missing = int((pos < 0).sum())
    if n_missing:
        log(f"  WARNING: {n_missing:,} gt ids not present in source1")
    log(f"  gt rows: {len(df):,}  S1 rows: {n1:,}  ids matched: {len(df)-n_missing:,}")
    if len(df) == n1:
        n_moved = int((pos != np.arange(len(df))).sum())
        log(f"  gt row order differs from source1 for {n_moved:,} rows "
            f"({100*n_moved/max(n1,1):.1f}%) — id mapping required")

    counts = np.zeros(n1, dtype=np.int64)
    flat_parts = []
    order = np.argsort(pos, kind='stable')      # emit flat in S1 row order
    for i in order:
        p = pos[i]
        if p < 0:
            continue
        s = raw[i]
        if s:
            parts = s.split(',')
            counts[p] = len(parts)
            flat_parts.append(parts)
    flat_ids = np.array([x for parts in flat_parts for x in parts], dtype=object)
    log(f"  {len(flat_ids):,} matched pairs")

    # matched ids -> S23 index
    id_to_idx = pd.Series(np.arange(len(s23_ids), dtype=np.int32),
                          index=pd.Index(s23_ids))
    t0 = time.time()
    flat_idx = id_to_idx.reindex(flat_ids).to_numpy()
    n_bad = int(np.isnan(flat_idx.astype(np.float64)).sum())
    log(f"  mapped to int32 in {time.time()-t0:.1f}s ({n_bad:,} unresolved)")

    offsets = np.zeros(n1 + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    gt = {'offsets': offsets, 'flat': flat_idx.astype(np.int32)}
    save(gt, os.path.join(out_dir, 'gt_train.pkl'))

    n_empty = int((counts == 0).sum())
    log(f"  {n_empty:,} singletons of {n1:,} ({100*n_empty/n1:.1f}%)")
    log(f"  matches/entity mean={counts.mean():.2f} max={counts.max()}")
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='../../dataset')
    ap.add_argument('--out', default='../../cache')
    ap.add_argument('--splits', nargs='+', default=['train', 'test'])
    ap.add_argument('--skip-gt', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for split in args.splits:
        ddir = os.path.join(args.data_dir, split)
        log(f"==== {split} ====")
        for src in ['source1', 'source2', 'source3']:
            out_path = os.path.join(args.out, f'norm_{split}_{src}.pkl')
            if os.path.exists(out_path):
                log(f"  {out_path} exists, skipping")
                continue
            rec = load_and_normalize(os.path.join(ddir, f'{split}_{src}.tsv'), src)
            save(rec, out_path)
            del rec

        # S23 id index space
        s23_path = os.path.join(args.out, f's23_ids_{split}.pkl')
        if not os.path.exists(s23_path):
            with open(os.path.join(args.out, f'norm_{split}_source2.pkl'), 'rb') as f:
                s2 = pickle.load(f)
            with open(os.path.join(args.out, f'norm_{split}_source3.pkl'), 'rb') as f:
                s3 = pickle.load(f)
            s23_ids = np.concatenate([s2['ids'], s3['ids']])
            save(s23_ids, s23_path)
            log(f"  S2 n={len(s2['ids']):,}  S3 n={len(s3['ids']):,}  "
                f"S23 n={len(s23_ids):,}")
            if split == 'train' and not args.skip_gt:
                with open(os.path.join(args.out, f'norm_{split}_source1.pkl'), 'rb') as f:
                    s1_ids = pickle.load(f)['ids']
                build_gt(ddir, args.out, s23_ids, s1_ids)
            del s2, s3

    log("cache build complete")


if __name__ == '__main__':
    main()
