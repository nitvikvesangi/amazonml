#!/usr/bin/env python3
"""Turn a predict run's scores.tsv into a submission at any decision rule.

`pipeline_fast.py predict --scores ...` writes every candidate pair with
probability >= score-floor.  That file is a superset of every possible
submission that run can produce, so the shipped decision layer can be re-tuned
without re-running the (hours-long) pipeline:

  * threshold           : accept pairs with prob >= t
  * per-country thr     : override t for the S1 record's own country, e.g.
                          --country-thr France=0.97 (blocking is a hard country
                          filter, so the S1 country is the pair's country)
  * conflict cleanup    : keep only the best claim per S2/S3 record (ground truth
                          is a strict partition, so extra claims are always wrong)
  * top-k per record    : at most K accepted candidates for one S1 entity

Output is a normal submission TSV, one row per S1 entity in the dataset's own
order (empty second column where nothing is accepted), ready for
utils/validate_submission.py.

Run from code/business_entity_resolution:
    python3 tools/threshold_apply.py --scores ../../output/run4/scores.tsv \
        --threshold 0.62 --conflict --out ../../output/ship/matching_results.tsv
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

R_CHUNK = 4_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_ids(path, column):
    ids = pd.read_csv(path, sep='\t', usecols=[column], dtype=str,
                      keep_default_na=False, na_filter=False)[column].to_numpy()
    return ids


def read_ids_country(path):
    """S1 ids plus their country column (same row order as the dataset)."""
    df = pd.read_csv(path, sep='\t', usecols=['entity_id', 'country'], dtype=str,
                     keep_default_na=False, na_filter=False)
    return df['entity_id'].to_numpy(), df['country'].to_numpy()


def load_scores(path, s1_index, floor=None):
    """Stream scores.tsv into int arrays (rec index, candidate id parts, prob)."""
    recs, cands, is3, probs = [], [], [], []
    n_rows = 0
    for chunk in pd.read_csv(path, sep='\t', dtype={'source1_entity_id': str,
                                                    'candidate_entity_id': str,
                                                    'prob': np.float32},
                             chunksize=R_CHUNK, keep_default_na=False,
                             na_filter=False):
        s1 = chunk['source1_entity_id'].to_numpy()
        cid = chunk['candidate_entity_id'].to_numpy()
        p = chunk['prob'].to_numpy(dtype=np.float32)
        if floor is not None:
            keep = p >= floor
            if not keep.all():
                s1, cid, p = s1[keep], cid[keep], p[keep]
        if len(p) == 0:
            continue
        rec = np.fromiter((s1_index[x] for x in s1), np.int32, len(s1))
        is_source3 = np.fromiter((x.startswith('S3-') for x in cid), bool, len(cid))
        cand = np.fromiter((int(x[3:]) for x in cid), np.int32, len(cid))
        recs.append(rec)
        cands.append(cand)
        is3.append(is_source3)
        probs.append(p)
        n_rows += len(p)
    if not recs:
        z = np.zeros(0)
        return (z.astype(np.int32), z.astype(np.int32), z.astype(bool),
                z.astype(np.float32))
    return (np.concatenate(recs), np.concatenate(cands), np.concatenate(is3),
            np.concatenate(probs))


def dedupe_pairs(rec, cand, is3, prob):
    """Drop duplicate (record, candidate) pairs, keeping the highest probability."""
    order = np.lexsort((-prob, cand, is3, rec))
    r, c, s = rec[order], cand[order], is3[order]
    same = (r[1:] == r[:-1]) & (c[1:] == c[:-1]) & (s[1:] == s[:-1])
    keep = np.ones(len(r), dtype=bool)
    keep[1:] = ~same
    return r[keep], c[keep], s[keep], prob[order][keep]


def conflict_clean(rec, cand, is3, prob):
    """Keep at most one claim per candidate id: the highest-probability one."""
    key = np.lexsort((-prob, is3, cand))
    c, s = cand[key], is3[key]
    same = (c[1:] == c[:-1]) & (s[1:] == s[:-1])
    keep = np.ones(len(c), dtype=bool)
    keep[1:] = ~same
    out = np.zeros(len(rec), dtype=bool)
    out[key[keep]] = True
    return out


def topk_mask(rec, prob, k):
    """At most k accepted candidates per record (best probability first)."""
    order = np.lexsort((-prob, rec))
    rs = rec[order]
    _, first, cnts = np.unique(rs, return_index=True, return_counts=True)
    rank = np.arange(len(rs), dtype=np.int64) - np.repeat(first, cnts)
    out = np.zeros(len(rec), dtype=bool)
    out[order[rank < k]] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True,
                    help='one or more scores.tsv files (shards are concatenated)')
    ap.add_argument('--threshold', type=float, required=True)
    ap.add_argument('--country-thr', action='append', default=[],
                    metavar='CC=T', help='per-country override, repeatable '
                    '(country values as in test_source1.tsv: US, India, France)')
    ap.add_argument('--conflict', action='store_true',
                    help='keep only the best claim per S2/S3 record')
    ap.add_argument('--topk', type=int, default=None)
    ap.add_argument('--dedupe', action='store_true', default=True)
    ap.add_argument('--source1', default='../../dataset/test/test_source1.tsv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', default=None)
    args = ap.parse_args()

    t0 = time.time()
    ids1, country1 = read_ids_country(args.source1)
    thr_map = {}
    for spec in args.country_thr:
        cc, _, val = spec.partition('=')
        thr_map[cc] = float(val)
    thr_rec = np.full(len(ids1), np.float32(args.threshold), dtype=np.float32)
    for cc, t in thr_map.items():
        mask = country1 == cc
        log(f"threshold override: {cc} -> {t:.3f} on {int(mask.sum()):,} S1 rows")
        thr_rec[mask] = np.float32(t)
    log(f"S1 rows: {len(ids1):,} (order taken from {args.source1})")
    s1_index = {x: i for i, x in enumerate(ids1.tolist())}

    recs, cands, is3s, probs = [], [], [], []
    for part in args.scores:
        r, c, s, p = load_scores(part, s1_index)
        log(f"  {part}: {len(p):,} pairs")
        recs.append(r); cands.append(c); is3s.append(s); probs.append(p)
    rec = np.concatenate(recs); cand = np.concatenate(cands)
    is3 = np.concatenate(is3s); prob = np.concatenate(probs)
    del recs, cands, is3s, probs
    log(f"scores: {len(prob):,} pairs, {int((prob >= thr_rec[rec]).sum()):,} "
        f"at/above threshold")
    if args.dedupe:
        rec, cand, is3, prob = dedupe_pairs(rec, cand, is3, prob)
        log(f"  after dedupe: {len(prob):,} pairs")

    hit = prob >= thr_rec[rec]
    if args.conflict:
        hit = hit & conflict_clean(rec, cand, is3, prob)
        log(f"  conflict cleanup keeps {int(hit.sum()):,} claims")
    if args.topk:
        hit = hit & topk_mask(rec, prob, args.topk)
        log(f"  top-{args.topk} keeps {int(hit.sum()):,} claims")

    r = rec[hit]
    c = cand[hit]
    s = is3[hit]
    order = np.lexsort((s, c, r))
    r, c, s = r[order], c[order], s[order]
    ids23 = np.array([f"S{2 + int(x)}-{int(y)}" for x, y in zip(s.tolist(), c.tolist())],
                     dtype=object)
    bounds = np.searchsorted(r, np.arange(len(ids1) + 1))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_matched = n_ids = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        lines = []
        for i in range(len(ids1)):
            b0, b1 = bounds[i], bounds[i + 1]
            if b1 > b0:
                # string-sorted exactly like `predict` writes them, so a
                # re-thresholded file is comparable byte-for-byte with predict's
                ids = sorted(ids23[b0:b1].tolist())
                n_matched += 1
                n_ids += len(ids)
                lines.append(f"{ids1[i]}\t{','.join(ids)}\n")
            else:
                lines.append(f"{ids1[i]}\t\n")
            if len(lines) >= 200_000:
                f.write(''.join(lines))
                lines = []
        f.write(''.join(lines))
    log(f"wrote {args.out}: {n_matched:,}/{len(ids1):,} rows with >=1 match, "
        f"{n_ids:,} ids, {n_ids / max(n_matched, 1):.2f} ids per matched row, "
        f"{time.time() - t0:.0f}s")
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(dict(scores=list(args.scores), threshold=args.threshold,
                           country_thr=thr_map,
                           conflict=bool(args.conflict), topk=args.topk,
                           rows=int(len(ids1)), matched_records=n_matched,
                           matched_ids=n_ids, seconds=round(time.time() - t0, 1)), f,
                      indent=1)


if __name__ == '__main__':
    main()
