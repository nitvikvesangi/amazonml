#!/usr/bin/env python3
"""Diff two submission TSVs and report added/removed claims per country.

Usage (from code/business_entity_resolution):
    python3 tools/ship_diff.py --a ../../output/ship500k_fr097.tsv \
        --b ../../output/ce_apply/ship_ce_fr097.tsv \
        --source1 ../../dataset/test/test_source1.tsv
"""
import argparse

import pandas as pd


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--source1', default='../../dataset/test/test_source1.tsv')
    args = ap.parse_args()

    a = pd.read_csv(args.a, sep='\t', dtype=str, keep_default_na=False,
                    na_filter=False)
    b = pd.read_csv(args.b, sep='\t', dtype=str, keep_default_na=False,
                    na_filter=False)
    assert list(a['source1_entity_id']) == list(b['source1_entity_id']), \
        'row order differs between ships'
    cty = pd.read_csv(args.source1, sep='\t', usecols=['entity_id', 'country'],
                      dtype=str, keep_default_na=False, na_filter=False)
    cmap = dict(zip(cty['entity_id'], cty['country']))

    per = {}
    n_add = n_rem = n_rows_changed = 0
    for i, (ida, sa, sb) in enumerate(zip(a['source1_entity_id'],
                                          a['matched_entity_ids'],
                                          b['matched_entity_ids'])):
        sa = set(sa.split(',')) if sa else set()
        sb = set(sb.split(',')) if sb else set()
        add = sb - sa
        rem = sa - sb
        n_add += len(add)
        n_rem += len(rem)
        if sa != sb:
            n_rows_changed += 1
        if add or rem:
            c = cmap[ida]
            d = per.setdefault(c, [0, 0, 0])
            d[0] += len(add)
            d[1] += len(rem)
            d[2] += 1
        if i and i % 500_000 == 0:
            log(f"  diffed {i:,} rows...")
    log(f"added {n_add:,} claims | removed {n_rem:,} claims | "
        f"net {n_add - n_rem:+,} | rows changed {n_rows_changed:,}")
    for c in sorted(per):
        d = per[c]
        log(f"  {c:<7} +{d[0]:,} -{d[1]:,} (net {d[0]-d[1]:+,}) on {d[2]:,} rows")


if __name__ == '__main__':
    main()
