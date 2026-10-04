#!/usr/bin/env python3
"""Per-country sanity check of a prediction file.

France never appears in training, so the main risk of a threshold tuned on train is
that it behaves differently there.  This reports, per country: how many S1 records,
how many have >=1 predicted match, the mean predicted matches per record, and the
distribution of prediction scores — a healthy run should look broadly similar across
countries rather than collapsing to all-empty or all-matched for one of them.

  python3 tools/country_stats.py --matching ../../output/matching_results.tsv --split test
"""
import argparse
import os
import sys
import pickle
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

CACHE = os.environ.get('ER_CACHE', '../../cache')
COUNTRIES = {0: 'US', 1: 'India', 2: 'France', -1: 'unknown'}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matching', default='../../output/matching_results.tsv')
    ap.add_argument('--split', default='test')
    ap.add_argument('--cache', default=CACHE)
    args = ap.parse_args()

    with open(os.path.join(args.cache, f'norm_{args.split}_source1.pkl'), 'rb') as f:
        s1 = pickle.load(f)
    c1 = np.zeros(len(s1['countries']), dtype=np.int8)
    for code, name in COUNTRIES.items():
        if code < 0:
            continue
        c1[np.asarray(s1['countries'], dtype=object) == name] = code
    id_to_row = {eid: i for i, eid in enumerate(s1['ids'])}

    n_pred = np.zeros(len(s1['ids']), dtype=np.int32)
    seen = 0
    with open(args.matching, encoding='utf-8') as f:
        f.readline()
        for line in f:
            s1_id, _, rest = line.partition('\t')
            row = id_to_row.get(s1_id)
            if row is None:
                continue
            seen += 1
            rest = rest.strip()
            if rest:
                n_pred[row] = rest.count(',') + 1
    log(f"rows seen: {seen:,} of {len(s1['ids']):,}")

    log(f"{'country':<9} {'records':>10} {'with_match':>11} {'%':>6} "
        f"{'mean/rec':>9} {'mean/rec|matched':>17} {'max':>5}")
    for code in (0, 1, 2, -1):
        m = c1 == code
        if not m.any():
            continue
        pred = n_pred[m]
        matched = pred > 0
        mean_matched = float(pred[matched].mean()) if matched.any() else 0.0
        log(f"{COUNTRIES[code]:<9} {int(m.sum()):>10,} {int(matched.sum()):>11,} "
            f"{100*matched.mean():>6.1f} "
            f"{pred.mean():>9.2f} {mean_matched:>17.2f} "
            f"{int(pred.max()):>5}")
    log(f"overall: mean {n_pred.mean():.2f} predicted matches/record, "
        f"{100*(n_pred > 0).mean():.1f}% non-empty "
        f"(training truth: 3.46 mean, 94.4% non-empty)")


if __name__ == '__main__':
    main()
