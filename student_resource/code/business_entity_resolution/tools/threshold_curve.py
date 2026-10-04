#!/usr/bin/env python3
"""F0.5 / precision / recall vs decision threshold for a saved model.

The competition weights precision 2x, so we report the whole curve rather than a
single cutoff: if two thresholds are within noise on validation, the higher one is
the safer bet on the leaderboard (fewer false merges, and false merges cost more).

  python3 tools/threshold_curve.py --sample 20000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pickle

import fast_data as fd
import fast_score as fs
import pipeline_fast as pf

CACHE = fd.CACHE


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--chunk-records', type=int, default=2000)
    ap.add_argument('--cand-cap', type=int, default=pf.CAND_CAP)
    ap.add_argument('--model', default='../../output/model_fast.pkl')
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--grid',
                    default='0.30,0.40,0.45,0.50,0.55,0.58,0.61,0.64,0.67,0.70,'
                            '0.75,0.80,0.85,0.90,0.95')
    ap.add_argument('--out', default=os.path.join(CACHE, 'threshold_curve.json'))
    args = ap.parse_args()

    data = fd.load_split('train', with_ids=False)
    keys = pf.load_keys('train')
    gt = pf.load_pkl(os.path.join(CACHE, 'gt_train.pkl'))
    cache = pf.data_cache(data, keys)
    with open(args.model, 'rb') as f:
        saved = pickle.load(f)
    model = saved['model']

    rec_idx = pf.spread_sample(data['n1'], args.sample, seed=args.seed)
    log(f"sample={len(rec_idx):,} records (seed {args.seed}), "
        f"saved threshold={saved['threshold']:.3f}")
    X, y, rec, cand, gt_local = pf.sample_pairs(
        data, keys, cache, rec_idx, gt, args.chunk_records, args.cand_cap,
        tag='curve ')
    nt = np.diff(gt_local['offsets']).astype(np.float64)
    tp_flag = fs.make_labels(rec, cand, gt_local, 1 << 30)
    found = np.bincount(rec[tp_flag == 1], minlength=len(rec_idx)).astype(np.float64)
    blk = float((found[nt > 0] / nt[nt > 0]).mean())
    oracle, _ = pf.macro_f05(found, nt, found)
    log(f"pairs={len(X):,} positives={int(y.sum()):,}  "
        f"blocking macro recall={blk:.4f}  oracle F0.5={oracle:.4f}")

    prob = model.predict(X)
    grid = [float(x) for x in args.grid.split(',')]
    log(f"{'thr':>6} {'macro_F0.5':>11} {'precision':>10} {'recall':>8} "
        f"{'predicted':>10} {'tp':>9} {'false_merge':>12}")
    rows = []
    for t in grid:
        macro, stats = pf.evaluate_at_threshold(rec, cand, prob, t, gt_local,
                                                len(rec_idx))
        rows.append(dict(threshold=t, macro_f05=macro, **stats))
        log(f"{t:>6.2f} {macro:>11.4f} {stats['precision']:>10.4f} "
            f"{stats['recall']:>8.4f} {stats['predicted']:>10,} {stats['tp']:>9,} "
            f"{stats['false_merge']:>12,}")
    best = max(rows, key=lambda r: r['macro_f05'])
    log(f"best on this sample: F0.5={best['macro_f05']:.4f} @ {best['threshold']:.2f}")
    with open(args.out, 'w') as f:
        json.dump(dict(sample=len(rec_idx), pairs=int(len(X)),
                       blocking_recall=blk, oracle_f05=oracle,
                       saved_threshold=saved['threshold'], rows=rows), f, indent=1)
    log(f"wrote {args.out}")


if __name__ == '__main__':
    main()
