#!/usr/bin/env python3
"""Why does the model miss true pairs - are they lexically findable at all?

The oracle run showed the residual loss is recall-side: a perfect scorer on the
same candidate set reaches 0.9745 while the model reaches 0.9066, and the loss
is concentrated in *partial* clusters (some members found, some rejected).  That
gap can only be closed by better features if the rejected true pairs actually
carry lexical evidence.  If instead they look exactly like the false pairs the
model already rejects, no lexical feature can separate them and the remaining
levers are data/objective, not new columns.

Run from code/business_entity_resolution:
    python3 tools/diag_missed.py --dump ../../output/eval41a30_fixed/pairs_base41_250k.npz
"""
import argparse
import os
import sys

import numpy as np
from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast_data as fd          # noqa: E402
import postproc as pp           # noqa: E402

N2 = 5034616


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--split', default='train')
    ap.add_argument('--sample', type=int, default=0, help='0 = every pair in the dump')
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    d = pp.load_dump(args.dump, gt_split=args.split)
    rec_idx, rec, cand, prob, tp = (d['rec_idx'], d['rec'], d['cand'], d['prob'],
                                    d['tp_flag'] == 1)
    n = len(prob)
    if args.sample and args.sample < n:
        rng = np.random.default_rng(args.seed)
        keep = np.sort(rng.choice(n, size=args.sample, replace=False))
        # keep every true pair (rare) and a sample of the rest
        keep = np.union1d(keep, np.flatnonzero(tp))
        rec, cand, prob, tp = rec[keep], cand[keep], prob[keep], tp[keep]
        rec_idx = rec_idx
    print(f"{len(prob):,} pairs ({int(tp.sum()):,} true)", flush=True)

    s1 = fd.load_norm(args.split, 'source1')
    s2 = fd.load_norm(args.split, 'source2')
    s3 = fd.load_norm(args.split, 'source3')
    is3 = cand >= N2
    c_all = np.where(is3, cand - N2, cand)
    an = s1['names'][rec_idx[rec]].tolist()
    bn = np.empty(len(cand), dtype=object)
    bn[~is3] = s2['names'][c_all[~is3]]
    bn[is3] = s3['names'][c_all[is3]]
    bn = bn.tolist()
    aa = s1['addrs'][rec_idx[rec]].tolist()
    ba = np.empty(len(cand), dtype=object)
    ba[~is3] = s2['addrs'][c_all[~is3]]
    ba[is3] = s3['addrs'][c_all[is3]]
    ba = ba.tolist()
    del s1, s2, s3

    n_ts = process.cpdist(an, bn, scorer=fuzz.token_set_ratio, workers=-1)
    a_ts = process.cpdist(aa, ba, scorer=fuzz.token_set_ratio, workers=-1)
    n_r = process.cpdist(an, bn, scorer=fuzz.ratio, workers=-1)
    a_r = process.cpdist(aa, ba, scorer=fuzz.ratio, workers=-1)
    best_ts = np.maximum(n_ts, a_ts) / 100.0
    best_r = np.maximum(n_r, a_r) / 100.0
    exact = np.array([x == y for x, y in zip(an, bn)])

    bands = [('rejected   (<0.50)', prob < 0.50),
             ('borderline (0.5-0.94)', (prob >= 0.50) & (prob < 0.94)),
             ('accepted   (>=0.94)', prob >= 0.94)]
    for label, m in bands:
        for kind, sub in (('true ', m & tp), ('false', m & ~tp)):
            if not sub.any():
                continue
            b = best_ts[sub]
            print(f"  {label} {kind} n={int(sub.sum()):>9,}  "
                  f"best_token_set mean={b.mean():.3f} p10={np.percentile(b, 10):.3f} "
                  f"median={np.median(b):.3f} | share<0.5={np.mean(b < 0.5):.1%} "
                  f"| exact_name={exact[sub].mean():.2%} "
                  f"| best_ratio mean={best_r[sub].mean():.3f}")
    print("\nReading: if 'rejected true' looks as strong as 'rejected false' "
          "(similar token_set distribution), the model is already using the lexical "
          "signal and new lexical columns cannot recover them.")


if __name__ == '__main__':
    main()
