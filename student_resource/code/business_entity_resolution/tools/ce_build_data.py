#!/usr/bin/env python3
"""Mine cross-encoder training pairs from a fresh eval dump.

The cross-encoder is meant to fix exactly the pairs the 41-feature GBDT cannot
separate: *hard positives* (true pairs scored just below the decision threshold)
and *hard negatives* (false pairs scored near/above it).  We mine them from a
fresh eval dump (100K train records, seed 7, scored by the shipped 500k model),
join the raw name/address strings and write JSONL for tools/ce_train.py.

Selection (defaults):
  pos_hard  label=1, prob in [0.30, 0.94)     cap 120k
  pos_acc   label=1, prob >= 0.94             sample 40k  (calibration anchor)
  neg_hard  label=0, prob >= 0.55             cap 120k
  neg_easy  label=0, prob in [0.10, 0.55)     sample 25k

Run from code/business_entity_resolution (after the fresh dump exists):
    python3 tools/ce_build_data.py \
        --dump ../../output/ce_train/sample_seed7.npz \
        --out ../../cache/ce_train.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import postproc as pp            # noqa: E402
import pipeline_fast as pf       # noqa: E402
from sem_probe import read_rows, log   # noqa: E402  (reused text reader)


def sample(idx, k, rng):
    if len(idx) <= k:
        return idx
    return idx[rng.choice(len(idx), k, replace=False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='../../output/ce_train/sample_seed7.npz')
    ap.add_argument('--dataset', default='../../dataset/train')
    ap.add_argument('--out', default='../../cache/ce_train.jsonl')
    ap.add_argument('--pos-hard-cap', type=int, default=120_000)
    ap.add_argument('--neg-hard-cap', type=int, default=120_000)
    ap.add_argument('--pos-acc', type=int, default=40_000)
    ap.add_argument('--neg-easy', type=int, default=25_000)
    ap.add_argument('--addr-max', type=int, default=200)
    args = ap.parse_args()

    d = pp.load_dump(args.dump)
    rec, cand, prob, rec_idx = d['rec'], d['cand'], d['prob'], d['rec_idx']
    tp = d['tp_flag'] == 1
    log(f"dump {args.dump}: {len(prob):,} pairs, {d['n']:,} records, "
        f"{int(tp.sum()):,} true")

    rng = np.random.default_rng(11)
    pos_hard = np.flatnonzero(tp & (prob >= 0.30) & (prob < 0.94))
    pos_acc = np.flatnonzero(tp & (prob >= 0.94))
    neg_hard = np.flatnonzero(~tp & (prob >= 0.55))
    neg_easy = np.flatnonzero(~tp & (prob >= 0.10) & (prob < 0.55))
    log(f"pool: pos_hard={len(pos_hard):,} pos_acc={len(pos_acc):,} "
        f"neg_hard={len(neg_hard):,} neg_easy={len(neg_easy):,}")
    pos_hard = sample(pos_hard, args.pos_hard_cap, rng)
    pos_acc = sample(pos_acc, args.pos_acc, rng)
    neg_hard = sample(neg_hard, args.neg_hard_cap, rng)
    neg_easy = sample(neg_easy, args.neg_easy, rng)

    take = np.concatenate([pos_hard, pos_acc, neg_hard, neg_easy])
    y = np.concatenate([np.ones(len(pos_hard) + len(pos_acc), np.int8),
                        np.zeros(len(neg_hard) + len(neg_easy), np.int8)])
    log(f"take {len(take):,} pairs (pos={int(y.sum()):,}, "
        f"neg={len(y) - int(y.sum()):,})")

    n2path = os.path.join(args.dataset, 'train_source2.tsv')
    n2 = sum(1 for _ in open(n2path, 'rb')) - 1
    need1 = set(rec_idx[rec[take]].tolist())
    is3 = cand[take] >= n2
    need2 = set(cand[take][~is3].tolist())
    need3 = set((cand[take][is3] - n2).tolist())
    log(f"strings needed: s1={len(need1):,} s2={len(need2):,} "
        f"s3={len(need3):,} (n2={n2:,})")

    t0 = time.time()
    rows1, _ = read_rows(os.path.join(args.dataset, 'train_source1.tsv'), need1)
    rows2, _ = read_rows(n2path, need2)
    rows3, _ = read_rows(os.path.join(args.dataset, 'train_source3.tsv'), need3)
    log(f"read {len(rows1) + len(rows2) + len(rows3):,} rows "
        f"in {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        for k, lab in zip(take, y):
            p1 = int(rec_idx[rec[k]])
            c = int(cand[k])
            a1 = rows1[p1]
            b1 = rows2[c] if c < n2 else rows3[c - n2]
            f.write(json.dumps(dict(
                y=int(lab), prob=round(float(prob[k]), 6), grp=p1,
                n1=a1[0], a1=a1[1][:args.addr_max],
                n2=b1[0], a2=b1[1][:args.addr_max]), ensure_ascii=False) + '\n')
    log(f"wrote {args.out} ({len(take):,} rows)")


if __name__ == '__main__':
    main()
