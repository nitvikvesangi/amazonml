#!/usr/bin/env python3
"""How many *true* pairs are cross-country?

The blocker filters candidate pairs to same-country only (`_pool_pairs`), which
is a huge precision win and lets the per-record cap be spent on the right pool.
This tool measures what that filter costs in recall: count ground-truth pairs
whose S1 country differs from the S2/S3 country.

Run from code/business_entity_resolution:
    python3 tools/country_pairs.py --split train
"""
import argparse
import os
import pickle

import numpy as np

CACHE = os.environ.get('ER_CACHE', '../../cache')
COUNTRY = {0: 'US', 1: 'India', 2: 'France', -1: 'unknown'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--examples', type=int, default=5)
    args = ap.parse_args()

    with open(os.path.join(args.cache, f'gt_{args.split}.pkl'), 'rb') as f:
        gt = pickle.load(f)
    with open(os.path.join(args.cache, f'country_{args.split}.pkl'), 'rb') as f:
        c1, c23 = pickle.load(f)

    off, flat = gt['offsets'], gt['flat']
    cnt = np.diff(off)
    rec = np.repeat(np.arange(len(cnt), dtype=np.int64), cnt)
    cc1 = c1[rec].astype(np.int16)
    cc23 = c23[flat].astype(np.int16)
    same = cc1 == cc23
    n_cross = int((~same).sum())
    print(f"{args.split}: {len(flat):,} true pairs | same-country "
          f"{100.0 * same.mean():.4f}% | cross-country {n_cross:,}")
    if n_cross:
        keys = np.stack([cc1[~same], cc23[~same]], 1)
        uniq, cts = np.unique(keys, axis=0, return_counts=True)
        for (x, y), n in sorted(zip(uniq.tolist(), cts.tolist()),
                                key=lambda t: -t[1])[:12]:
            print(f"   {COUNTRY.get(x, '?'):<7} -> {COUNTRY.get(y, '?'):<7} {n:,}")
        print("   first mismatching s1 rows:", np.flatnonzero(~same)[:args.examples].tolist())
    # also: how many true pairs have an unknown (-1) country on either side
    unknown = (cc1 < 0) | (cc23 < 0)
    print(f"   pairs with an unknown country: {int(unknown.sum()):,}")


if __name__ == '__main__':
    main()
