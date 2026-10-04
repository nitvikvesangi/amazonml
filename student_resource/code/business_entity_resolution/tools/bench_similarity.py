#!/usr/bin/env python3
"""Benchmark rapidfuzz vectorized throughput on real data.

Answers: how many (S1, S2/S3) pairs can we score per second, per metric, with
one process and with all cores?

Run:  python3 tools/bench_similarity.py --data-dir ../../dataset/train --rows 300000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from pipeline import normalize_name, normalize_address


def timed(label, fn, n_pairs):
    t0 = time.time()
    out = fn()
    dt = time.time() - t0
    rate = n_pairs / dt
    print(f"  {label:<34} {dt:7.2f}s  {rate/1e6:7.2f} M pairs/s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default='../../dataset/train')
    ap.add_argument('--rows', type=int, default=300000)
    ap.add_argument('--pairs', type=int, default=1000000)
    ap.add_argument('--workers', type=int, default=-1)
    args = ap.parse_args()

    n = args.rows
    print(f"Loading {n:,} rows of train_source2 ...")
    t0 = time.time()
    df = pd.read_csv(os.path.join(args.data_dir, 'train_source2.tsv'), sep='\t',
                     dtype=str, engine='c', keep_default_na=False, na_filter=False,
                     nrows=n)
    print(f"  loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    names = [normalize_name(x) for x in df['business_name']]
    print(f"  normalize_name():   {time.time()-t0:.1f}s "
          f"({n/(time.time()-t0)/1000:.0f}K rec/s)")

    t0 = time.time()
    addrs = [normalize_address(x) for x in df['business_address']]
    print(f"  normalize_address():{time.time()-t0:.1f}s "
          f"({n/(time.time()-t0)/1000:.0f}K rec/s)")

    # Build a pair list: names[i] vs names[(i*7+3) % n]  (same length strings
    # distribution as real candidate pairs)
    P = args.pairs
    idx1 = np.arange(P, dtype=np.int32) % n
    idx2 = (idx1 * 7 + 3) % n
    a_names = [names[i] for i in idx1]
    b_names = [names[i] for i in idx2]
    a_addrs = [addrs[i] for i in idx1]
    b_addrs = [addrs[i] for i in idx2]
    print(f"\nBenchmarking {P:,} pairs "
          f"(avg name len {np.mean([len(s) for s in a_names]):.0f} chars)\n")

    w = args.workers
    timed("JaroWinkler.similarity (name)", lambda: process.cpdist(
        a_names, b_names, scorer=JaroWinkler.similarity, workers=w), P)
    timed("fuzz.ratio (name)", lambda: process.cpdist(
        a_names, b_names, scorer=fuzz.ratio, workers=w), P)
    timed("fuzz.partial_ratio (name)", lambda: process.cpdist(
        a_names, b_names, scorer=fuzz.partial_ratio, workers=w), P)
    timed("fuzz.token_sort_ratio (name)", lambda: process.cpdist(
        a_names, b_names, scorer=fuzz.token_sort_ratio, workers=w), P)
    timed("fuzz.token_set_ratio (name)", lambda: process.cpdist(
        a_names, b_names, scorer=fuzz.token_set_ratio, workers=w), P)
    timed("JaroWinkler.similarity (addr)", lambda: process.cpdist(
        a_addrs, b_addrs, scorer=JaroWinkler.similarity, workers=w), P)
    timed("fuzz.token_set_ratio (addr)", lambda: process.cpdist(
        a_addrs, b_addrs, scorer=fuzz.token_set_ratio, workers=w), P)

    print("\nRaw python loop for comparison (10k pairs):")
    sub = 10000
    t0 = time.time()
    for i in range(sub):
        JaroWinkler.similarity(a_names[i], b_names[i])
        fuzz.ratio(a_names[i], b_names[i])
        fuzz.token_sort_ratio(a_names[i], b_names[i])
        fuzz.token_set_ratio(a_names[i], b_names[i])
        JaroWinkler.similarity(a_addrs[i], b_addrs[i])
        fuzz.token_set_ratio(a_addrs[i], b_addrs[i])
    dt = time.time() - t0
    print(f"  6 metrics, scalar loop:  {dt:.2f}s for {sub:,} pairs "
          f"= {sub*6/dt/1e6:.2f} M metric-calls/s "
          f"(-> {6*1.7e8/ (sub*6/dt) /3600:.1f} h for 170M pairs)")


if __name__ == '__main__':
    main()
