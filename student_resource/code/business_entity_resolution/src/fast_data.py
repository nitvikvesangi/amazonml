#!/usr/bin/env python3
"""Shared loader for the normalized cache.

Index space convention (used everywhere in the fast pipeline):
    S23 index i < n2  -> source2 record i
    S23 index i >= n2 -> source3 record i - n2
Ground truth, blockers and predictions all speak this int32 index space, so no
string ids are ever compared during the heavy stages.
"""
import os
import pickle

import numpy as np

CACHE = os.environ.get('ER_CACHE', '../../cache')

# Clean and consistent across all three sources (verified: only these appear).
COUNTRY_MAP = {'US': 0, 'India': 1, 'France': 2}


def load_norm(split, src, cache_dir=CACHE):
    with open(os.path.join(cache_dir, f'norm_{split}_{src}.pkl'), 'rb') as f:
        return pickle.load(f)


def _codes(values):
    out = np.full(len(values), -1, dtype=np.int8)
    for name, code in COUNTRY_MAP.items():
        out[np.asarray(values, dtype=object) == name] = code
    return out


def ensure_country_codes(split, cache_dir=CACHE):
    """Small cached (c1, c23) int8 country codes, built once from the norm cache."""
    path = os.path.join(cache_dir, f'country_{split}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    s1 = load_norm(split, 'source1', cache_dir)
    s2 = load_norm(split, 'source2', cache_dir)
    s3 = load_norm(split, 'source3', cache_dir)
    out = (_codes(s1['countries']),
           np.concatenate([_codes(s2['countries']), _codes(s3['countries'])]))
    del s1, s2, s3
    with open(path, 'wb') as f:
        pickle.dump(out, f)
    return out


def load_split(split, cache_dir=CACHE, with_ids=True):
    """Return dict(s1, s23, n1, n2, n23, ids...)."""
    s1 = load_norm(split, 'source1', cache_dir)
    s2 = load_norm(split, 'source2', cache_dir)
    s3 = load_norm(split, 'source3', cache_dir)
    out = {
        's1': dict(names=s1['names'], addrs=s1['addrs'],
                   countries=_codes(s1['countries'])),
        's23': dict(names=np.concatenate([s2['names'], s3['names']]),
                    addrs=np.concatenate([s2['addrs'], s3['addrs']]),
                    countries=np.concatenate([_codes(s2['countries']),
                                              _codes(s3['countries'])])),
        'n1': len(s1['names']),
        'n2': len(s2['names']),
        'n23': len(s2['names']) + len(s3['names']),
    }
    if with_ids:
        out['ids1'] = s1['ids']
        out['ids23'] = np.concatenate([s2['ids'], s3['ids']])
    del s1, s2, s3
    return out
