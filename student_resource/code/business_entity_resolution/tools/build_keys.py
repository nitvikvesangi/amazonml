#!/usr/bin/env python3
"""Build (and cache) blocking keys for a split.

  python3 tools/build_keys.py --split test

Writes cache/keys_{split}.pkl: per strategy, the S1/S23 CSR key arrays, key df,
postings (key_start + post_rec) and the vocabulary.  Needs the normalized cache
(see tools/build_cache.py).  ~2 minutes per split, then every experiment reuses it.
"""
import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from fast_data import load_norm
from fast_keys import build_split_keys, STRATEGIES

CACHE = os.environ.get('ER_CACHE', '../../cache')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--cache', default=CACHE)
    ap.add_argument('--max-df-ratio', type=float, default=0.01)
    args = ap.parse_args()

    out_path = os.path.join(args.cache, f'keys_{args.split}.pkl')
    if os.path.exists(out_path):
        log(f"{out_path} already exists ({os.path.getsize(out_path)/1e6:.0f} MB)")
        return

    s1 = load_norm(args.split, 'source1', args.cache)
    s2 = load_norm(args.split, 'source2', args.cache)
    s3 = load_norm(args.split, 'source3', args.cache)
    data = dict(s1=dict(names=s1['names'], addrs=s1['addrs']),
                s23=dict(names=np.concatenate([s2['names'], s3['names']]),
                         addrs=np.concatenate([s2['addrs'], s3['addrs']])))
    del s1, s2, s3
    log(f"split={args.split}: n1={len(data['s1']['names']):,} "
        f"n23={len(data['s23']['names']):,}")
    build_split_keys(data, out_path, strategies=STRATEGIES,
                     max_df_ratio=args.max_df_ratio)
    log(f"done: {out_path} ({os.path.getsize(out_path)/1e6:.0f} MB)")


if __name__ == '__main__':
    main()
