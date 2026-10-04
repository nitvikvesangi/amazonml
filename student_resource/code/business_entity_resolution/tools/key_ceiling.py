"""Which keys catch true pairs, and what do the misses look like?

Works directly on TRUE PAIRS (no block expansion, so it is cheap and safe to run
while a prediction job is using the machine).

For each sampled S1 record with ground truth we ask, per true pair:
  - does it share any key in each of the 4 production families?
  - would a NEW key family catch it?
and finally we print real examples of pairs that every family misses, so the
next key idea is driven by data rather than guessing.

Run: python3 tools/key_ceiling.py --sample 40000
"""
import argparse
import collections
import os
import pickle
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

from fast_keys import extract_keys, STRATEGIES

CACHE = os.environ.get('ER_CACHE', '../../cache')

LEGAL = {'inc', 'ltd', 'llc', 'pvt', 'private', 'limited', 'corp', 'corporation',
         'co', 'company', 'llp', 'plc', 'gmbh', 'sarl', 'sa', 'sas', 'pte',
         'and', 'the', 'of', '&'}
SUFFIXES = ('ing', 'ers', 'er', 'es', 'ed', 's')

SOUNDEX_MAP = {**{c: '1' for c in 'BFPV'}, **{c: '2' for c in 'CGJKQSXZ'},
               **{c: '3' for c in 'DT'}, **{c: '4' for c in 'L'},
               **{c: '5' for c in 'MN'}, **{c: '6' for c in 'R'}}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def soundex(tok):
    tok = re.sub(r'[^a-z]', '', tok.lower())
    if not tok:
        return ''
    first = tok[0]
    rest = ''.join(SOUNDEX_MAP.get(c, '') for c in tok[1:])
    rest = re.sub(r'(.)\1+', r'\1', rest).replace('0', '')
    return (first + rest + '000')[:4]


def new_families(name, addr):
    """Candidate NEW key families -> set of keys (prototypes)."""
    nt = [t for t in name.split() if len(t) >= 2]
    aw = [t for t in addr.split() if len(t) >= 3 and not t.isdigit()]
    core = [t for t in nt if t not in LEGAL]
    out = {
        # legal-suffix-stripped name tokens (recovers "X Inc" vs "X Pvt Ltd")
        'ncore': set(core),
        # tolerant stem of name tokens (plurals / gerunds)
        'nstem': {re.sub(r'(ing|ers|es|ed|s)$', '', t) for t in core if len(t) >= 5},
        # phonetic name tokens (typos, transliteration: "fone" ~ "phone")
        'nphon': {soundex(t) for t in core if len(t) >= 4},
        # zip / pin code (5-6 digits) -- very discriminative when present
        'zip': set(re.findall(r'\b\d{5,6}\b', addr)),
        # any long digit run in the address (house number + zip fused)
        'anum6': set(re.findall(r'\d{6,}', addr)),
        # first 6 chars of the *first* address token (street-name prefix)
        'astreet': {aw[0][:6]} if aw and not aw[0].isdigit() else set(),
        # name tokens that contain digits (7-Eleven, 7eleven)
        'ndigit': {t for t in nt if re.search(r'\d', t)},
        # last 4 chars of long name tokens (suffix match: "consultants"/"consulting")
        'nsuf': {t[-4:] for t in core if len(t) >= 7},
    }
    out['zip'] |= set(re.findall(r'\b\d{5}\b', addr))
    return out


def production_families(names, addrs):
    out = {}
    for st in STRATEGIES:
        ex = extract_keys(names, addrs, st)
        d = collections.defaultdict(set)
        if len(ex):
            idx = ex.index.to_numpy()
            vals = ex.to_numpy(dtype=object)
            for i, v in zip(idx, vals):
                d[int(i)].add(v)
        out[st] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=40000)
    ap.add_argument('--split', default='train')
    ap.add_argument('--examples', type=int, default=15)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    log('loading normalized text + ground truth')
    s1 = pickle.load(open(os.path.join(CACHE, f'norm_{args.split}_source1.pkl'), 'rb'))
    s2 = pickle.load(open(os.path.join(CACHE, f'norm_{args.split}_source2.pkl'), 'rb'))
    s3 = pickle.load(open(os.path.join(CACHE, f'norm_{args.split}_source3.pkl'), 'rb'))
    gt = pickle.load(open(os.path.join(CACHE, f'gt_{args.split}.pkl'), 'rb'))
    n2 = len(s2['names'])
    n23 = n2 + len(s3['names'])
    names23 = np.concatenate([s2['names'], s3['names']])
    addrs23 = np.concatenate([s2['addrs'], s3['addrs']])
    del s2, s3

    n1 = len(s1['names'])
    rng = np.random.default_rng(0)
    rec_idx = np.sort(rng.choice(n1, size=min(args.sample, n1), replace=False))
    off, flat = gt['offsets'], gt['flat']
    truth = [flat[off[r]:off[r + 1]] for r in rec_idx]
    keep = [i for i, t in enumerate(truth) if len(t)]
    rec_idx = rec_idx[keep]
    truth = [truth[i] for i in keep]
    n_pairs = sum(len(t) for t in truth)
    log(f'sample={len(rec_idx):,} records, {n_pairs:,} true pairs')

    # keys for the sampled S1 records, and only for the S23 records they touch
    touched = np.unique(np.concatenate(truth))
    log(f'extracting keys for {len(touched):,} touched S23 records')
    prod1 = production_families(s1['names'][rec_idx], s1['addrs'][rec_idx])
    prod23 = production_families(names23[touched], addrs23[touched])
    pos = {int(v): i for i, v in enumerate(touched)}
    log('extracting prototype new families')
    new1 = collections.defaultdict(dict)
    new23 = collections.defaultdict(dict)
    for i in range(len(rec_idx)):
        new1[i] = new_families(s1['names'][rec_idx[i]], s1['addrs'][rec_idx[i]])
    for j, v in enumerate(touched):
        new23[j] = new_families(names23[v], addrs23[v])

    cov = collections.Counter()
    newcov = collections.Counter()
    union_cov = collections.Counter()
    misses = []
    nt_hist = collections.Counter()
    for i, t in enumerate(truth):
        f1n = new1[i]
        for rec in t:
            j = pos[int(rec)]
            hit = set()
            for st in STRATEGIES:
                if prod1[st].get(i) and (prod1[st][i] & prod23[st].get(j, set())):
                    hit.add(st)
            nhit = set()
            for k, v in f1n.items():
                if v and (v & new23[j].get(k, set())):
                    nhit.add(k)
            if hit:
                for st in hit:
                    cov[st] += 1
                union_cov['+'.join(sorted(hit))] += 1
            for k in nhit:
                newcov[k] += 1
            if not hit:
                if nhit:
                    union_cov['NEW:' + '+'.join(sorted(nhit))] += 1
                    misses.append((s1['names'][rec_idx[i]], s1['addrs'][rec_idx[i]],
                                   names23[rec], addrs23[rec], sorted(nhit)))
                else:
                    union_cov['NEITHER'] += 1
                    misses.append((s1['names'][rec_idx[i]], s1['addrs'][rec_idx[i]],
                                   names23[rec], addrs23[rec], []))

    print('\n=== per-family coverage of true pairs (production) ===')
    for st in STRATEGIES:
        print(f'  {st:<7} {cov[st]:>9,d}  {100*cov[st]/n_pairs:6.2f}%')
    print('\n=== NEW family coverage, among pairs ALL production families miss '
          '(%-d such pairs) ===' % union_cov['NEITHER'])
    miss_n = union_cov['NEITHER']
    for k, v in newcov.most_common():
        print(f'  {k:<8} {v:>9,d}  {100*v/max(1,miss_n):6.2f}% of misses   '
              f'{100*v/n_pairs:6.2f}% of all')
    print('\n=== combined production coverage ===')
    hit_any = n_pairs - union_cov['NEITHER']
    print(f'  any production family : {hit_any:,d} / {n_pairs:,d} = '
          f'{100*hit_any/n_pairs:.3f}%   (structural recall ceiling, unlimited cap)')
    rescue = sum(v for k, v in union_cov.items() if k.startswith('NEW:'))
    print(f'  rescued by a new family : {rescue:,d} = {100*rescue/n_pairs:.3f}%')
    print(f'  still missed by everything: {union_cov["NEITHER"]:,d} '
          f'= {100*union_cov["NEITHER"]/n_pairs:.3f}%')

    print('\n=== sample of TRUE PAIRS with no shared production key ===')
    shown = 0
    for nm1, ad1, nm2, ad2, nh in misses:
        if shown >= args.examples:
            break
        print(f'  S1 : {nm1!r} | {ad1!r}')
        print(f'  S23: {nm2!r} | {ad2!r}')
        print(f'       new-family hits: {nh}')
        shown += 1

    if args.out:
        import json
        json.dump({'n_pairs': n_pairs, 'production': dict(cov),
                   'new_on_misses': dict(newcov), 'neither': int(union_cov['NEITHER']),
                   'union': {k: int(v) for k, v in union_cov.items()}},
                  open(args.out, 'w'), indent=1)
        log(f'wrote {args.out}')


if __name__ == '__main__':
    main()
