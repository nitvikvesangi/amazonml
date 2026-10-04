#!/usr/bin/env python3
"""Fast smoke test: key building + budgeted blocking on small slices."""
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import fast_block as fb
from fast_keys import build_strategy, STRATEGIES

CACHE = os.environ.get('ER_CACHE', '../../cache')


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    n1 = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    n23 = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
    s1 = load_pkl(os.path.join(CACHE, 'norm_train_source1.pkl'))
    s2 = load_pkl(os.path.join(CACHE, 'norm_train_source2.pkl'))
    s3 = load_pkl(os.path.join(CACHE, 'norm_train_source3.pkl'))

    data1 = dict(names=s1['names'][:n1], addrs=s1['addrs'][:n1])
    data23 = dict(names=np.concatenate([s2['names'][:n23 // 2], s3['names'][:n23 // 2]]),
                  addrs=np.concatenate([s2['addrs'][:n23 // 2], s3['addrs'][:n23 // 2]]))

    keys = {}
    for st in STRATEGIES:
        t0 = time.time()
        keys[st] = build_strategy(data1, data23, st, max_df_ratio=0.02)
        print(f"  {st} built in {time.time()-t0:.1f}s")

    cfg = {'top_k': 200, 'pool': {
        'ntok': {'budget': 300, 'max_df': 20000},
        'npre': {'budget': 150, 'max_df': 2000},
        'anum': {'budget': 50, 'max_df': 100000},
        'aword': {'budget': 150, 'max_df': 10000},
    }}
    t0 = time.time()
    res = fb.block_split(keys, n1, n23, cfg, chunk_records=20000)
    dt = time.time() - t0
    counts = np.diff(res['off'])
    print(f"\nblock_split: {dt:.1f}s  pairs={res['n_pairs']:,} "
          f"({res['n_pairs']/n1:.1f}/rec)  raw_expansion={res['n_raw_expansion']:,} "
          f"({res['n_raw_expansion']/n1:.1f}/rec)")
    print(f"  records with candidates: {res['n_with_cand']:,}/{n1:,}")
    print(f"  shared-count distribution: "
          f"mean={res['shared'].mean():.2f} max={res['shared'].max()} "
          f">=2: {(res['shared'] >= 2).mean()*100:.1f}%")
    print(f"  name-pool bits>0: {(res['nbits'] > 0).sum():,}  "
          f"addr-pool bits>0: {(res['abits'] > 0).sum():,}")
    print(f"  candidate index range: {res['cand'].min()} .. {res['cand'].max()} "
          f"(n23={n23})")
    assert res['cand'].max() < n23
    assert counts.sum() == res['n_pairs']
    print("\nSMOKE OK")


if __name__ == '__main__':
    main()
