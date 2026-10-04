#!/usr/bin/env python3
"""Compare two dumped val pair sets on a shared, leak-free record subset.

The canonical eval holdout (--sample 30000, seed 42) overlaps the train
samples: a 250K-record train sample shares 3,413 of the 30K eval records, a
500K sample 6,777 (22.6%).  Headline macro is therefore inflated per model by a
*different* amount, so any model comparison must be run on the subset no model
saw in training.  This tool does that: it drops the leaked records (computed
with the same spread_sample seed the trainer uses) and prints headline vs clean
macro for each dump, plus each dump's oracle on the clean subset.

Run from code/business_entity_resolution:
    python3 tools/compare_dumps.py \
        --dump ../../output/eval41a30_fixed/pairs_base41_250k.npz:champion \
        --dump ../../output/eval41a30_fixed/pairs_base41_500k.npz:model500k \
        --clean-train 250000 --clean-train 500000
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import pipeline_fast as pf          # noqa: E402
import postproc as pp               # noqa: E402

N_TOTAL = 2_206_821      # train S1 records the train/eval samples are drawn from


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', action='append', required=True,
                    help='path[:label], repeat for each dumped model')
    ap.add_argument('--clean-train', type=int, action='append', default=[],
                    help='train sample size, repeat for every model compared; the '
                         'union of the sampled record sets is dropped from EVERY '
                         'dump (samples of different sizes are not nested)')
    ap.add_argument('--thresholds', default='0.90,0.92,0.93,0.94,0.95,0.97')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    grid = [float(x) for x in args.thresholds.split(',')]

    leaked = None
    for n_train in args.clean_train:
        if n_train <= 0:
            continue
        s = pf.spread_sample(N_TOTAL, n_train, seed=0)
        leaked = s if leaked is None else np.union1d(leaked, s)
    if leaked is not None:
        print(f"masking {len(leaked):,} train-sampled records "
              f"({', '.join(f'{n:,}' for n in args.clean_train)}) from every dump",
              flush=True)
    rep = {}
    for spec in args.dump:
        path, _, label = spec.partition(':')
        label = label or os.path.basename(path)
        d = pp.load_dump(path)
        thr = d['threshold'] if d['threshold'] is not None else 0.94
        clean = (~np.isin(d['rec_idx'], leaked) if leaked is not None else None)
        n_clean = int(clean.sum()) if clean is not None else d['n']
        hm, _ = pp.score(d, d['prob'] >= thr, None)
        cm, cs = pp.score(d, d['prob'] >= thr, clean)
        best_m, best_t = max((pp.score(d, d['prob'] >= t, clean)[0], t)
                             for t in grid)
        om, _ = pp.score(d, d['tp_flag'] == 1, clean)
        print(f"{label}: thr={thr:.3f} headline={hm:.4f} | clean n={n_clean:,} "
              f"macro={cm:.4f} (P={cs['precision']:.4f} R={cs['recall']:.4f} "
              f"pred={cs['predicted']:,}) | clean best on grid: thr={best_t:.3f} "
              f"-> {best_m:.4f} | clean ORACLE={om:.4f}", flush=True)
        rep[label] = dict(threshold=thr, headline=hm, clean=cm, clean_P=cs['precision'],
                          clean_R=cs['recall'], clean_best_thr=best_t,
                          clean_best=best_m, clean_oracle=om, n_clean=n_clean)
    if len(rep) == 2:
        (la, a), (lb, b) = list(rep.items())
        print(f"DELTA {lb} - {la}: own-threshold {b['clean'] - a['clean']:+.4f} | "
              f"best-on-grid {b['clean_best'] - a['clean_best']:+.4f} | "
              f"oracle {b['clean_oracle'] - a['clean_oracle']:+.4f}", flush=True)
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rep, f, indent=1)


if __name__ == '__main__':
    main()
