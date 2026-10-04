#!/usr/bin/env python3
"""Merge a prefix + contiguous predict shards into one submission file.

`pipeline_fast.py predict --start-records A --max-records B` writes only rows
A..B-1, so a full run can be split across processes (1.7 M records in ~3x less
wall time) and stitched back in record order here.  The merge also *checks* the
result against the dataset's own S1 id order, which is the cheapest way to catch
an off-by-one range before wasting a submission:

  * every row present exactly once, in the dataset's order
  * no empty/duplicate/malformed id lists

Run from code/business_entity_resolution:
    python3 tools/merge_shards.py \
        --parts ../../output/run3/matching_results.tsv \
                ../../output/run3_shard1/matching_results.tsv ... \
        --out ../../output/run3/matching_results.tsv \
        --expected ../../dataset/test/test_source1.tsv
"""
import argparse
import os
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def copy_parts(parts, out, skip_headers):
    n = 0
    with open(out, 'w', encoding='utf-8') as f:
        for i, part in enumerate(parts):
            with open(part, encoding='utf-8') as g:
                if skip_headers and i > 0:
                    first = g.readline()
                    if not first.startswith('source1_entity_id'):
                        raise SystemExit(f"{part}: missing header, refusing to merge")
                for line in g:
                    f.write(line)
                    n += 1
    return n


def check(path, expected_path):
    expect = []
    with open(expected_path, encoding='utf-8') as f:
        next(f, None)
        for line in f:
            expect.append(line.split('\t', 1)[0].strip())
    n = matched = ids = 0
    problems = []
    with open(path, encoding='utf-8') as f:
        header = next(f, None)
        for line in f:
            n += 1
            s1, _, rest = line.rstrip('\n').partition('\t')
            if n <= len(expect) and s1 != expect[n - 1]:
                problems.append(f"row {n}: {s1} != expected {expect[n-1]}")
            if rest:
                lst = rest.split(',')
                if len(set(lst)) != len(lst):
                    problems.append(f"row {n}: duplicate id inside the row")
                ids += len(lst)
                matched += 1
    return dict(rows=n, expected_rows=len(expect), header=bool(header),
                matched_rows=matched, ids=ids, problems=problems[:10],
                n_problems=len(problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', required=True,
                    help='prefix first, then shards, in record order')
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected', default=None,
                    help='dataset source1 tsv to check row order/count against')
    ap.add_argument('--append-candidates', action='store_true',
                    help='also merge candidate_pairs.tsv next to each match file')
    args = ap.parse_args()

    had_header = args.parts[0].endswith('.tsv') and os.path.exists(args.parts[0])
    n = copy_parts(args.parts, args.out, skip_headers=had_header)
    log(f"merged {len(args.parts)} parts -> {args.out}: {n:,} rows")
    if args.append_candidates:
        cparts = [os.path.join(os.path.dirname(p), 'candidate_pairs.tsv')
                  for p in args.parts]
        cout = os.path.join(os.path.dirname(args.out), 'candidate_pairs.tsv')
        missing = [p for p in cparts if not os.path.exists(p)]
        if missing:
            log(f"candidates: skipped, missing {missing[:2]}")
        else:
            cn = copy_parts(cparts, cout, skip_headers=True)
            log(f"merged candidates -> {cout}: {cn:,} rows")
    if args.expected:
        rep = check(args.out, args.expected)
        log(f"check: rows={rep['rows']:,} (expected {rep['expected_rows']:,}), "
            f"rows with ids={rep['matched_rows']:,}, ids={rep['ids']:,}, "
            f"problems={rep['n_problems']}")
        for p in rep['problems']:
            log(f"  !! {p}")
        if rep['rows'] != rep['expected_rows'] or rep['n_problems']:
            raise SystemExit("merged file FAILED the self-check")
        log("self-check PASS")


if __name__ == '__main__':
    main()
