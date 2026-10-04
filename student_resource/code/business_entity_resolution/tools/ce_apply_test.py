#!/usr/bin/env python3
"""Apply the validated cross-encoder rule to the test score matrix.

Reads a predict score file (one shard per invocation, superset of every
decision this model can make), selects US/India pairs in the wide zone
``0.60 <= prob < 0.99``, scores them with the trained cross-encoder and
rewrites the matrix for tools/threshold_apply.py:

  rescue: r_lo <= prob < 0.94  and ce >= c_res  -> prob := 0.945
  clean : 0.94 <= prob < z_hi  and ce <  c_cl   -> prob := 0.0

France rows are never modified (no French ground truth exists anywhere in
train, so the CE cannot be validated on France; the frozen France 0.97 rule
stays exactly as shipped).

TWO-MODE DESIGN (the 21:50 lesson: 80 min of CE scoring died in the write step
because pandas 3.0 returns read-only arrays from to_numpy — never again):

  --save-ce DIR   score every wide-zone chunk and cache (ids1, cid, prob, ce)
                  per chunk as ``DIR/part<k>.npz`` as soon as it is scored.
  --load-ce DIR   skip scoring entirely and rebuild the decisions from those
                  caches. Chunks without a cache are left unchanged, so a
                  partial cache degrades gracefully instead of failing.

The zone is always the wide one; --r-lo/--c-res/--z-hi/--c-cl are applied at
decision time, so changing the rule costs a ~5 min replay, not a rescore.

Text policy is identical to training/validation:
    A = name1 + ' | ' + addr1[:200]      B = name23 + ' | ' + addr23[:200]

Run from code/business_entity_resolution (under .venv-sem), one process per
score shard so they share the device:

    .venv-sem/bin/python3 tools/ce_apply_test.py \
        --scores ../../output/pred500k_C/scores.tsv ../../output/pred500k_D/scores.tsv \
        --r-lo 0.6 --c-res 0.7 --z-hi 0.99 --c-cl 0.2 --fp16 --batch 512 \
        --save-ce ../../output/ce_apply/cache \
        --out ../../output/ce_apply/scores_ce.tsv

then:

    python3 tools/threshold_apply.py \
        --scores ../../output/ce_apply/scores_ce.tsv \
        --threshold 0.94 --country-thr France=0.97 \
        --out ../../output/ce_apply/ship_ce_fr097.tsv
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from sem_probe import log   # noqa: E402

CHUNK = 2_000_000
BASE_THR = 0.94
WIDE_LO, WIDE_HI = 0.60, 0.99


def read_meta(path):
    df = pd.read_csv(path, sep='\t', usecols=['entity_id', 'country'], dtype=str,
                     keep_default_na=False, na_filter=False)
    return df['entity_id'].to_numpy(), df['country'].to_numpy()


def collect_texts(path, need):
    """id -> (name, address) for the wanted ids, chunked scan."""
    out = {}
    for ch in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False,
                          na_filter=False, chunksize=500_000):
        sel = ch[ch['entity_id'].isin(need)]
        for r in sel.itertuples(index=False):
            out[r.entity_id] = (r.business_name, r.business_address)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True)
    ap.add_argument('--test-dir', default='../../dataset/test')
    ap.add_argument('--model-dir', default='../../output/ce_model')
    ap.add_argument('--base-model',
                    default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    ap.add_argument('--r-lo', type=float, default=WIDE_LO)
    ap.add_argument('--c-res', type=float, required=True)
    ap.add_argument('--z-hi', type=float, default=WIDE_HI)
    ap.add_argument('--c-cl', type=float, required=True)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--fp16', action='store_true',
                    help='half precision + dynamic padding: 2.3x on MPS, '
                         'max|dCE| ~0.005 and ~1/1500 decision flips (measured)')
    ap.add_argument('--save-ce', default=None,
                    help='cache dir: score wide-zone chunks and save part<k>.npz')
    ap.add_argument('--load-ce', default=None,
                    help='cache dir: skip scoring, replay decisions from part<k>.npz')
    ap.add_argument('--out', default='../../output/ce_apply/scores_ce.tsv')
    args = ap.parse_args()
    if args.save_ce and args.load_ce:
        ap.error('--save-ce and --load-ce are exclusive')

    t0 = time.time()
    s1_path = os.path.join(args.test_dir, 'test_source1.tsv')
    ids1, country1 = read_meta(s1_path)
    s1_index = {x: i for i, x in enumerate(ids1.tolist())}
    log(f"S1 rows: {len(ids1):,}")

    # ---- pass 1: select the wide zone (US/India only), one part per chunk ----
    parts = []          # (rec_idx, cid, prob, s1_id, id23)
    for part in args.scores:
        n = 0
        for ch in pd.read_csv(part, sep='\t', dtype={'source1_entity_id': str,
                                                     'candidate_entity_id': str,
                                                     'prob': np.float32},
                              chunksize=CHUNK, keep_default_na=False,
                              na_filter=False):
            s1 = ch['source1_entity_id'].to_numpy()
            cid = ch['candidate_entity_id'].to_numpy()
            p = ch['prob'].to_numpy(np.float32)
            rec = np.fromiter((s1_index[x] for x in s1), np.int64, len(s1))
            cc = country1[rec]
            ca = (cc == 'US') | (cc == 'India')
            m = ca & (p >= WIDE_LO) & (p < WIDE_HI)
            if m.any():
                parts.append((rec[m], cid[m], p[m], s1[m], cid[m]))
            n += int(m.sum())
        log(f"  {part}: {n:,} wide-zone pairs")
    n_tot = sum(len(q[2]) for q in parts)
    n_lo = sum(int((q[2] < BASE_THR).sum()) for q in parts)
    log(f"wide-zone pairs: {n_tot:,} in {len(parts)} parts "
        f"(rescue-side {n_lo:,} / clean-side {n_tot - n_lo:,})")

    # ---- pass 2: texts for every selected pair (identical policy to training) ---
    ids1s = np.concatenate([q[3] for q in parts])
    ids23s = np.concatenate([q[4] for q in parts])
    t1 = time.time()
    need1 = set(ids1s.tolist())
    need2 = set(x for x in ids23s.tolist() if x.startswith('S2-'))
    need3 = set(x for x in ids23s.tolist() if x.startswith('S3-'))
    rows1 = collect_texts(s1_path, need1)
    rows2 = collect_texts(os.path.join(args.test_dir, 'test_source2.tsv'), need2)
    rows3 = collect_texts(os.path.join(args.test_dir, 'test_source3.tsv'), need3)
    log(f"texts: s1={len(rows1):,} s2={len(rows2):,} s3={len(rows3):,} "
        f"in {time.time()-t1:.0f}s")

    # ---- pass 2b: score (or load) each part, decide immediately ----
    tok = model = None
    if not args.load_ce:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        tok = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model, num_labels=1)
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'ce_state.pt'),
                                         map_location='cpu'))
        model.to(device)
        if args.fp16:
            model.half()
        model.eval()
        log(f"model ready on {device}{' fp16+dynamic-pad' if args.fp16 else ' fp32'}")
    if args.save_ce:
        os.makedirs(args.save_ce, exist_ok=True)

    dec = {}
    off = 0
    n_res = n_cl = n_skip = 0
    for pi, (rec_arr, cid_arr, p, s1_arr, id23_arr) in enumerate(parts):
        n = len(p)
        ce = None
        cache = None
        if args.load_ce:
            cache = os.path.join(args.load_ce, f"part{pi}.npz")
            if os.path.exists(cache):
                # allow_pickle: the id arrays in the shipped caches are object-dtype strings
                # (pandas to_numpy of a string column); the comparison below is dtype-agnostic.
                with np.load(cache, allow_pickle=True) as z:
                    if (len(z['ce']) == n and list(z['ids1']) == s1_arr.tolist()
                            and list(z['cid']) == cid_arr.tolist()):
                        ce = z['ce']
                    else:
                        log(f"part{pi}: cache does not match this selection -> skipped")
        else:
            A, B = [], []
            for a, b in zip(s1_arr.tolist(), id23_arr.tolist()):
                r1 = rows1[a]
                r2 = rows2[b] if b.startswith('S2-') else rows3[b]
                A.append(r1[0] + ' | ' + r1[1][:200])
                B.append(r2[0] + ' | ' + r2[1][:200])
            ce = np.zeros(n, np.float32)
            t2 = time.time()
            with torch.inference_mode():
                for i in range(0, n, args.batch):
                    enc = tok(A[i:i + args.batch], B[i:i + args.batch],
                              padding=(True if args.fp16 else 'max_length'),
                              truncation=True,
                              max_length=args.max_len, return_tensors='pt')
                    enc = {k: v.to(device) for k, v in enc.items()}
                    ce[i:i + args.batch] = torch.sigmoid(
                        model(**enc).logits.squeeze(-1)).float().cpu().numpy()
            log(f"  part{pi}: {n:,} pairs scored in {time.time()-t2:.0f}s "
                f"({n/max(time.time()-t2, 1e-9):,.0f} pairs/s)")
            if args.save_ce:
                cache = os.path.join(args.save_ce, f"part{pi}.npz")
                # store ids as unicode (not object) so the cache reloads without pickle
                np.savez(cache, ids1=s1_arr.astype(str), cid=cid_arr.astype(str),
                         prob=p, ce=ce)
                log(f"    cached -> {cache}")
            del A, B
        if ce is None:
            n_skip += n
            off += n
            continue
        res = (p >= args.r_lo) & (p < BASE_THR) & (ce >= args.c_res)
        cln = (p >= BASE_THR) & (p < args.z_hi) & (ce < args.c_cl)
        n_res += int(res.sum())
        n_cl += int(cln.sum())
        rl = rec_arr.tolist()
        cl_ = cid_arr.tolist()
        for i in np.flatnonzero(res):
            dec[(rl[i], cl_[i])] = 0.945
        for i in np.flatnonzero(cln):
            dec[(rl[i], cl_[i])] = 0.0
        off += n
    log(f"decisions: rescue +{n_res:,} / clean -{n_cl:,} "
        f"(map {len(dec):,}, unscored pairs left unchanged {n_skip:,})")
    if n_skip:
        log(f"!! {n_skip:,} pairs were left at their GBDT score (no CE cache for that part) - "
            f"the rewritten matrix will NOT match the shipped submission")

    # ---- pass 3: rewrite the score matrix for threshold_apply.py ----
    if os.path.dirname(args.out):
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    total = changed = 0
    with open(args.out, 'w') as f:
        f.write('source1_entity_id\tcandidate_entity_id\tprob\n')
        for part in args.scores:
            for ch in pd.read_csv(part, sep='\t', dtype={'source1_entity_id': str,
                                                         'candidate_entity_id': str,
                                                         'prob': np.float32},
                                  chunksize=CHUNK, keep_default_na=False,
                                  na_filter=False):
                s1 = ch['source1_entity_id'].tolist()
                cid = ch['candidate_entity_id'].tolist()
                p = ch['prob'].to_numpy(np.float32).copy()   # pandas 3.0: view is read-only
                rec = np.fromiter((s1_index[x] for x in s1), np.int64, len(s1))
                for i in range(len(p)):
                    v = dec.get((int(rec[i]), cid[i]))
                    if v is not None:
                        p[i] = v
                        changed += 1
                out = pd.DataFrame({'source1_entity_id': s1,
                                    'candidate_entity_id': cid, 'prob': p})
                out.to_csv(f, sep='\t', index=False, header=False,
                           float_format='%.6f')
                total += len(p)
    log(f"wrote {args.out}: {total:,} rows, {changed:,} changed, "
        f"{time.time()-t0:.0f}s total")


if __name__ == '__main__':
    main()
