#!/usr/bin/env python3
"""Where does the F0.5 loss come from: the features or the model?

Reports, on a sample of train records:
  * per-feature AUC (positives vs negatives among blocked candidates)
  * best achievable macro F0.5 from each single feature alone
  * the trained model's macro F0.5 and the blocking/oracle ceiling

If a single feature beats the model, the model/threshold path is broken.  If all
features are weak, the features themselves are the limit.

Run:  python3 tools/diag_features.py --sample 6000
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


def auc(y, s):
    """Fast AUC (ties handled approximately, fine for diagnostics)."""
    order = np.argsort(s, kind='stable')
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_macro_f05(rec, score, y, gt_local, n_records, thresholds=None):
    """Best macro F0.5 obtainable by thresholding one score."""
    if thresholds is None:
        thresholds = np.quantile(score, np.linspace(0.5, 0.9999, 60))
    nt = np.diff(gt_local['offsets']).astype(np.float64)
    best = (0.0, 0.0)
    for t in np.unique(thresholds):
        hit = score >= t
        r = rec[hit]
        pred = np.bincount(r, minlength=n_records).astype(np.float64)
        tp = np.bincount(r[y[hit] == 1], minlength=n_records).astype(np.float64)
        macro, _ = pf.macro_f05(pred, nt, tp)
        if macro > best[0]:
            best = (float(macro), float(t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=6000)
    ap.add_argument('--chunk-records', type=int, default=2000)
    ap.add_argument('--cand-cap', type=int, default=pf.CAND_CAP)
    ap.add_argument('--model', default='../../output/model_fast.pkl')
    ap.add_argument('--out', default=os.path.join(CACHE, 'diag_features.json'))
    args = ap.parse_args()

    data = fd.load_split('train', with_ids=False)
    keys = pf.load_keys('train')
    gt = pf.load_pkl(os.path.join(CACHE, 'gt_train.pkl'))
    cache = pf.data_cache(data, keys)
    rec_idx = pf.spread_sample(data['n1'], args.sample, seed=7)
    log(f"sample={len(rec_idx):,} records")
    X, y, rec, cand, gt_local = pf.sample_pairs(
        data, keys, cache, rec_idx, gt, args.chunk_records, args.cand_cap,
        tag='diag ')
    log(f"pairs={len(X):,} positives={int(y.sum()):,} ({100*y.mean():.2f}%)")

    nt = np.diff(gt_local['offsets']).astype(np.float64)
    tp_flag = y.astype(bool)
    found = np.bincount(rec[tp_flag], minlength=len(rec_idx)).astype(np.float64)
    blk = float((found[nt > 0] / nt[nt > 0]).mean())
    oracle, _ = pf.macro_f05(found, nt, found)
    log(f"blocking macro recall={blk:.4f}   oracle F0.5={oracle:.4f}")

    with open(args.model, 'rb') as f:
        saved = pickle.load(f)
    model = saved['model']
    prob = model.predict(X)
    t = saved['threshold']
    macro_t, stats_t = pf.evaluate_at_threshold(rec, cand, prob, t, gt_local,
                                                len(rec_idx))
    bt, bf, bs = pf.tune_threshold(rec, cand, prob, gt_local, len(rec_idx))
    log(f"model: macro F0.5 @{t:.3f}={macro_t:.4f}  best={bf:.4f} @{bt:.3f}")
    log(f"  stats @best: {json.dumps(bs)}")

    rows = []
    log(f"\n{'feature':<18} {'AUC':>7} {'bestF0.5':>9} {'thr':>7}")
    for j, name in enumerate(fs.FEATURES):
        col = X[:, j].astype(np.float64)
        a = auc(y, col)
        f05, thr = best_macro_f05(rec, col, y, gt_local, len(rec_idx))
        rows.append(dict(feature=name, auc=float(a), best_f05=float(f05),
                         threshold=float(thr)))
        log(f"{name:<18} {a:>7.4f} {f05:>9.4f} {thr:>7.3f}")
    rows.sort(key=lambda r: -r['best_f05'])

    # a hand-tuned rule on the most promising combination (name+addr token set)
    nt_set = X[:, fs.FEATURES.index('name_token_set')]
    at_set = X[:, fs.FEATURES.index('addr_token_set')]
    rule = np.maximum(nt_set, at_set)
    f05_rule, thr_rule = best_macro_f05(rec, rule, y, gt_local, len(rec_idx))
    log(f"\nrule max(name_token_set, addr_token_set): F0.5={f05_rule:.4f} "
        f"@{thr_rule:.3f}")
    with open(args.out, 'w') as f:
        json.dump(dict(sample=len(rec_idx), pairs=int(len(X)),
                       positives=int(y.sum()), blocking_recall=blk,
                       oracle_f05=oracle, model_f05_at_saved=macro_t,
                       model_best_f05=bf, model_best_threshold=bt,
                       rule_f05=f05_rule, rule_threshold=thr_rule,
                       features=rows, stats=bs), f, indent=1)
    log(f"wrote {args.out}")


if __name__ == '__main__':
    main()
