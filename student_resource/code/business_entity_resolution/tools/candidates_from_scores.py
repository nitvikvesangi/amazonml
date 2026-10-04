#!/usr/bin/env python3
"""Rebuild `candidate_pairs.tsv` (submission format) from the GBDT score matrix.

`pipeline_fast.py predict` writes the *raw* blocking output next to the score matrix
(cap-200 / alpha-30 -> ~198 candidates per record, 343 M pairs, 4.4 GB on the test
split). The organisers' upload field caps a submission package at 1024 MB, and no
compression gets 4.4 GB of entity-id TSV under that (deflate ~1.6 GB, LZMA ~1.5 GB),
so the package ships the candidate set that the shipped model actually *used*: every
pair the scorer saw (`prob >= --score-floor`, default 0.10 = the predict default),
grouped back into one `source1_entity_id<TAB>candidate_entity_ids` row per Source-1
record, in the dataset's record order.

Every match in `matching_results.tsv` is a scored pair by construction, so
*matches subset-of candidates* still holds (checked with tools/check_outputs.py).
The full raw blocking output is reproduced by step 2 of README section 3.

    python3 src/candidates_from_scores.py \
        --scores ../../output/pred500k_C/scores.tsv ../../output/pred500k_D/scores.tsv \
        --out ../../output/candidate_pairs_scored.tsv \
        --expected ../../dataset/test/test_source1.tsv
"""
import argparse
import os
import time


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected', default='../../dataset/test/test_source1.tsv')
    ap.add_argument('--min-prob', type=float, default=0.0,
                    help='only keep pairs at/above this probability (predict wrote prob >= 0.10)')
    args = ap.parse_args()

    order = []
    with open(args.expected, encoding='utf-8') as f:
        next(f, None)
        for line in f:
            order.append(line.split('\t', 1)[0].strip())
    index = {s1: i for i, s1 in enumerate(order)}
    log(f"dataset records: {len(order):,}")

    cands = {}                      # s1 index -> [candidate ids] (deduped, in score order)
    n_rows = n_ids = n_dup = n_bad_prefix = n_empty_rows = 0
    last_idx = -1
    t0 = time.time()
    for part in args.scores:
        n_part = 0
        with open(part, encoding='utf-8') as f:
            header = f.readline().rstrip('\n').split('\t')
            if header[:2] != ['source1_entity_id', 'candidate_entity_id']:
                raise SystemExit(f"{part}: unexpected header {header}")
            for line in f:
                s1, cid, prob = line.rstrip('\n').split('\t')
                if args.min_prob and float(prob) < args.min_prob:
                    continue
                i = index.get(s1)
                if i is None:
                    raise SystemExit(f"{part}: unknown S1 id {s1}")
                if i < last_idx:
                    raise SystemExit(f"{part}: rows are not grouped by record "
                                     f"({s1} after index {last_idx})")
                last_idx = i
                if not cid.startswith(('S2-', 'S3-')):
                    n_bad_prefix += 1
                lst = cands.setdefault(i, [])
                if cid in lst:
                    n_dup += 1
                else:
                    lst.append(cid)
                n_rows += 1
                n_ids += 1
                n_part += 1
        last_idx = -1               # each shard is a contiguous, ordered range
        log(f"  {part}: {n_part:,} scored pairs")

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for i, s1 in enumerate(order):
            lst = cands.get(i)
            if not lst:
                n_empty_rows += 1
                f.write(f'{s1}\t\n')
            else:
                f.write(f'{s1}\t{",".join(lst)}\n')
    size = os.path.getsize(args.out)
    log(f"wrote {args.out}: {len(order):,} rows, {n_ids:,} scored pairs, {n_empty_rows:,} empty rows, "
        f"{size/1e6:.1f} MB in {time.time()-t0:.0f}s")
    log(f"  duplicate candidate ids inside a row: {n_dup:,}; non-S2/S3 ids: {n_bad_prefix:,}")
    if n_bad_prefix:
        raise SystemExit("id prefix check failed")


if __name__ == '__main__':
    main()
