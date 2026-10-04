#!/usr/bin/env python3
"""Debug why blocking recall is low: inspect concrete records end to end."""
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import fast_block as fb
from fast_data import load_norm, ensure_country_codes

CACHE = os.environ.get('ER_CACHE', '../../cache')


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def main():
    keys = load_pkl(os.path.join(CACHE, 'keys_train.pkl'))
    gt = load_pkl(os.path.join(CACHE, 'gt_train.pkl'))
    s1 = load_norm('train', 'source1')
    s2 = load_norm('train', 'source2')
    s3 = load_norm('train', 'source3')
    names23 = np.concatenate([s2['names'], s3['names']])
    addrs23 = np.concatenate([s2['addrs'], s3['addrs']])
    n2 = len(s2['names'])
    n23 = len(names23)
    c1, c23 = ensure_country_codes('train')

    off, flat = gt['offsets'], gt['flat']
    n1 = int(keys['ntok']['n1'])

    # pick records that have >=1 true match
    rng = np.random.default_rng(0)
    picks = []
    for _ in range(200):
        r = int(rng.integers(0, n1))
        if off[r + 1] > off[r]:
            picks.append(r)
        if len(picks) == 3:
            break

    print("=== per-record inspection ===")
    for r in picks:
        truth = flat[off[r]:off[r + 1]]
        print(f"\nS1[{r}] {s1['names'][r]!r} | {s1['addrs'][r]!r} | "
              f"country_code={c1[r]} ({s1['countries'][r]})")
        print(f"  true matches: {len(truth)}")
        for t in truth[:3]:
            src = 'S2' if t < n2 else 'S3'
            print(f"   - {src}[{t}] {names23[t]!r} | {addrs23[t]!r} | "
                  f"code={c23[t]} ({s2['countries'][t] if t < n2 else s3['countries'][t - n2]})")
            # shared keys in the full key structures
            for st in ('ntok', 'npre', 'anum', 'aword'):
                a = set(keys[st]['s1_flat'][keys[st]['s1_off'][r]:keys[st]['s1_off'][r + 1]].tolist())
                b = set(keys[st]['s23_flat'][keys[st]['s23_off'][t]:keys[st]['s23_off'][t + 1]].tolist())
                shared = a & b
                if st == 'ntok':
                    uniq = keys[st]['uniques']
                    print(f"     {st}: a={[uniq[i] for i in a]} "
                          f"b={[uniq[i] for i in b]} shared={[uniq[i] for i in shared]}")

    # now run blocking on just these 3 records and see if matches survive
    print("\n=== blocking on the picked records ===")
    picks = sorted(picks)          # key slicing needs ascending records
    ks = {}
    for st, idx in keys.items():
        new = dict(idx)
        new['s1_flat'] = idx['s1_flat'][idx['s1_off'][picks[0]]:idx['s1_off'][picks[-1] + 1]]
        new['s1_off'] = idx['s1_off'][picks[0]:picks[-1] + 2] - idx['s1_off'][picks[0]]
        ks[st] = new
    cfg = {'top_k': None, 'pool': {
        'ntok': {'budget': 300, 'max_df': 20000},
        'npre': {'budget': 150, 'max_df': 2000},
        'anum': {'budget': 50, 'max_df': 100000},
        'aword': {'budget': 150, 'max_df': 10000}}}
    span = picks[-1] - picks[0] + 1
    res = fb.block_split(ks, span, n23, cfg, chunk_records=span,
                         country1=c1[picks[0]:picks[-1] + 1], country23=c23)
    counts = np.diff(res['off'])
    for i, r in enumerate(picks):
        local = r - picks[0]
        cands = res['cand'][res['off'][local]:res['off'][local + 1]]
        truth = set(flat[off[r]:off[r + 1]].tolist())
        hit = truth & set(cands.tolist())
        print(f"  rec {r}: {len(cands)} candidates, truth={len(truth)}, "
              f"hits={len(hit)}; shared counts head="
              f"{sorted(res['shared'][res['off'][local]:res['off'][local + 1]].tolist())[-5:]}")


if __name__ == '__main__':
    main()
