#!/usr/bin/env python3
"""Streaming output checker (bounded memory) for the two submission files.

`utils/validate_submission.py` is the official gate and is used for
matching_results.tsv, but it builds a set of every candidate id per row, which on
the full ~343 M-id candidate file needs far more than 16 GB.  This checker applies
the same rules with O(number of S1 ids) memory:

  * every test S1 entity appears exactly once in each file (empty allowed)
  * no duplicate S1 rows
  * no S1- ids inside the match/candidate lists
  * only S2-/S3- prefixed ids
  * no repeated id inside one list
  * matching_results.tsv matches are a subset of candidate_pairs.tsv
  * rows are in the same order in both files (so the subset check can stream)

  python3 tools/check_outputs.py --matching ../../output/matching_results.tsv \
      --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
"""
import argparse
import os
import sys
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_required_ids(path):
    with open(path, encoding='utf-8') as f:
        next(f, None)
        return {line.split('\t', 1)[0].strip() for line in f if line.strip()}


def stream_pairs(path, col):
    """Yield (s1_id, [ids]) for a results TSV, after checking the header."""
    with open(path, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        if header != ['source1_entity_id', col]:
            raise SystemExit(f"{path}: bad header {header}")
        for line in f:
            s1, tab, rest = line.partition('\t')
            if not tab:
                continue
            rest = rest.rstrip('\n')
            yield s1, (rest.split(',') if rest else [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matching', default='../../output/matching_results.tsv')
    ap.add_argument('--candidate', default='../../output/candidate_pairs.tsv')
    ap.add_argument('--test-dir', default='../../dataset/test')
    ap.add_argument('--sample-rows', type=int, default=0,
                    help='only check the first N rows (0 = all)')
    args = ap.parse_args()

    required = read_required_ids(os.path.join(args.test_dir, 'test_source1.tsv'))
    log(f"required S1 entities: {len(required):,}")
    errors = []

    # --- pass 1: matching file ---
    seen = set()
    n_rows = n_empty = n_ids = 0
    dup_rows = self_rows = bad_prefix = intra = 0
    t0 = time.time()
    matches = {}
    for s1, ids in stream_pairs(args.matching, 'matched_entity_ids'):
        n_rows += 1
        if s1 in seen:
            dup_rows += 1
        seen.add(s1)
        if not ids:
            n_empty += 1
        if len(ids) != len(set(ids)):
            intra += 1
        for mid in ids:
            if mid.startswith('S1-'):
                self_rows += 1
            elif not mid.startswith(('S2-', 'S3-')):
                bad_prefix += 1
        n_ids += len(ids)
        matches[s1] = set(ids)
        if args.sample_rows and n_rows >= args.sample_rows:
            break
    log(f"matching: {n_rows:,} rows, {n_empty:,} empty, {n_ids:,} ids "
        f"({time.time()-t0:.0f}s)")
    missing = len(required - seen)
    extra = len(seen - required)

    # --- pass 2: candidate file, streamed in the same order ---
    t0 = time.time()
    c_rows = c_empty = c_ids = 0
    not_subset = 0
    offenders = []
    with open(args.candidate, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        if header != ['source1_entity_id', 'candidate_entity_ids']:
            raise SystemExit(f"candidate: bad header {header}")
        for line in f:
            s1, tab, rest = line.partition('\t')
            if not tab:
                continue
            rest = rest.rstrip('\n')
            ids = rest.split(',') if rest else []
            c_rows += 1
            if not ids:
                c_empty += 1
            c_ids += len(ids)
            m = matches.get(s1)
            if m:
                missing_ids = m - set(ids)
                if missing_ids:
                    not_subset += 1
                    if len(offenders) < 5:
                        offenders.append(s1)
            del matches[s1]
            if args.sample_rows and c_rows >= args.sample_rows:
                break
    log(f"candidates: {c_rows:,} rows, {c_empty:,} empty, {c_ids:,} ids "
        f"({time.time()-t0:.0f}s)")

    if dup_rows:
        errors.append(f"{dup_rows:,} duplicate S1 rows in matching")
    if intra:
        errors.append(f"{intra:,} rows with repeated ids inside a match list")
    if self_rows:
        errors.append(f"{self_rows:,} S1- ids inside match lists")
    if bad_prefix:
        errors.append(f"{bad_prefix:,} ids without S2-/S3- prefix")
    if missing:
        errors.append(f"{missing:,} required S1 entities missing from matching"
                      + (" (expected when --sample-rows is used)" if args.sample_rows else ""))
    if extra:
        errors.append(f"{extra:,} matching rows use unknown S1 ids")
    if not_subset:
        errors.append(f"{not_subset:,} rows have matches not present in candidates, "
                      f"e.g. {offenders}")
    if c_rows != n_rows:
        errors.append(f"row count differs: matching {n_rows:,} vs candidates {c_rows:,}")

    print()
    if errors:
        print(f"FAIL — {len(errors)} issue(s):")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
        return 1
    print("PASS — rows, ids, prefixes and match⊆candidate all consistent.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
