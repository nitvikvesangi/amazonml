#!/usr/bin/env python3
"""Shrink candidate_pairs.tsv while guaranteeing matches stay a subset.

The full blocking audit is ~200 candidates per S1 record (~5 GB for the test
split).  If the submission zip has a size limit, this keeps only the first N
candidates per record AND every predicted match (so the validator's
"matches must be a subset of candidates" check still passes), which is all the
file is used for.

  python3 tools/shrink_candidates.py \
      --matching ../../output/matching_results.tsv \
      --candidate ../../output/candidate_pairs.tsv \
      --out ../../output/candidate_pairs_100.tsv --keep 100
"""
import argparse
import os
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matching', default='../../output/matching_results.tsv')
    ap.add_argument('--candidate', default='../../output/candidate_pairs.tsv')
    ap.add_argument('--out', default='../../output/candidate_pairs_small.tsv')
    ap.add_argument('--keep', type=int, default=100)
    args = ap.parse_args()

    # matches per S1 id (small file, ~30 MB)
    matched = {}
    with open(args.matching, encoding='utf-8') as f:
        header = f.readline().rstrip('\n')
        for line in f:
            s1, _, rest = line.partition('\t')
            rest = rest.rstrip('\n')
            matched[s1] = set(rest.split(',')) if rest else set()
    log(f"loaded {len(matched):,} matched rows from {args.matching}")

    n_rows = n_kept = n_added = 0
    t0 = time.time()
    with open(args.candidate, encoding='utf-8') as fin, \
            open(args.out, 'w', encoding='utf-8') as fout:
        fout.write(header + '\n')
        for line in fin:
            s1, tab, rest = line.partition('\t')
            if not tab:
                continue
            ids = rest.rstrip('\n').split(',') if rest.strip() else []
            keep = ids[:args.keep]
            need = matched.get(s1, set())
            if need:
                have = set(keep)
                missing = [m for m in ids[args.keep:] if m in need and m not in have]
                if missing:
                    keep = keep + missing
                    n_added += len(missing)
            n_rows += 1
            n_kept += len(keep)
            fout.write(f"{s1}\t{','.join(keep)}\n")
            if n_rows % 500_000 == 0:
                log(f"  {n_rows:,} rows ({time.time()-t0:.0f}s)")
    log(f"{args.out}: {n_rows:,} rows, {n_kept:,} candidate ids "
        f"({n_kept/max(n_rows,1):.1f}/record), {n_added:,} matches re-added")
    log(f"  size {os.path.getsize(args.out)/1e9:.2f} GB "
        f"(was {os.path.getsize(args.candidate)/1e9:.2f} GB)")


if __name__ == '__main__':
    main()
