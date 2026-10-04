#!/usr/bin/env python3
"""Fast blocking keys — CSR int32 arrays, never per-record Python loops.

Each record's blocking keys for one strategy are stored as:
    flat     int32 array of key ids, grouped by record (record-major order)
    offsets  int64 array, length n_records+1  (offsets[i]:offsets[i+1] = i's keys)

Strategies
----------
  ntok   significant name tokens          (exact token overlap)
  npre   3-char prefixes of name tokens   (typo tolerance: netw0rk ~ network)
  anum   2+ digit numbers in the address  (DBA / trade-name matches)
  aword  significant address words        (city, locality, street name)

The S1 side is mapped into the S23 vocabulary with ``pd.Index.get_indexer``;
keys absent from S23 are dropped (they cannot match anything).

Cached to ``keys_{split}.pkl`` so experiments never rebuild them.
"""
import os
import pickle
import time

import numpy as np
import pandas as pd

# Reuse the exact stopword sets / semantics of the original pipeline so results
# stay comparable with the v1/v2 numbers.
from pipeline import BLOCK_STOPWORDS, ADDR_STOPWORDS

STRATEGIES = ('ntok', 'npre', 'anum', 'aword')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# key extraction (all pandas / C-level)
# ---------------------------------------------------------------------------

def _name_token_series(names):
    """Exploded significant name tokens, indexed by record position."""
    s = pd.Series(names, dtype=object)
    ex = s.str.split().explode()
    ex = ex[ex.str.len() >= 2]
    ex = ex[~ex.isin(BLOCK_STOPWORDS)]
    return ex


def _addr_word_series(addrs):
    s = pd.Series(addrs, dtype=object)
    ex = s.str.split().explode()
    ex = ex[ex.str.len() >= 3]
    ex = ex[~ex.isin(ADDR_STOPWORDS)]
    ex = ex[~ex.str.isdigit()]
    return ex


def extract_keys(names, addrs, strategy):
    """Return an exploded Series (index = record idx, value = key string)."""
    if strategy == 'ntok':
        return _name_token_series(names)
    if strategy == 'npre':
        ex = _name_token_series(names)
        return ex[ex.str.len() >= 5].str.slice(0, 3)
    if strategy == 'anum':
        s = pd.Series(addrs, dtype=object)
        ex = s.str.findall(r'\d{2,}').explode()
        return ex.dropna()
    if strategy == 'aword':
        return _addr_word_series(addrs)
    raise ValueError(strategy)


def _csr_from_exploded(ex, n_records):
    """Codes + offsets from an exploded Series whose index holds record ids."""
    if len(ex) == 0:
        return (np.zeros(0, dtype=np.int32),
                np.zeros(n_records + 1, dtype=np.int64),
                np.zeros(0, dtype=object))
    rec = ex.index.to_numpy(dtype=np.int64)
    codes, uniques = pd.factorize(ex.to_numpy(dtype=object))
    flat = codes.astype(np.int32)
    counts = np.bincount(rec, minlength=n_records)
    offsets = np.zeros(n_records + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    return flat, offsets, uniques


# ---------------------------------------------------------------------------
# building one strategy for a whole split
# ---------------------------------------------------------------------------

def build_strategy(s1, s23, strategy, max_df_ratio=0.01):
    """Build keys for both sides; S1 is mapped into the S23 vocabulary.

    Returns dict with s1_flat/s1_off/s23_flat/s23_off/df/n_keys.
    """
    t0 = time.time()
    n1 = len(s1['names'])
    n23 = len(s23['names'])

    ex23 = extract_keys(s23['names'], s23['addrs'], strategy)
    flat23, off23, uniques = _csr_from_exploded(ex23, n23)
    n_keys = len(uniques)
    log(f"    {strategy}: S23 {len(flat23):,} keys, {n_keys:,} distinct "
        f"({time.time()-t0:.1f}s)")
    del ex23

    # df over S23 (used for rare-key selection + df cap)
    df = np.bincount(flat23, minlength=n_keys).astype(np.int32)
    max_df = max(int(n23 * max_df_ratio), 50)
    n_drop = int((df > max_df).sum())
    log(f"      df: max={df.max():,} median={np.median(df):.0f} "
        f"keys>cap({max_df:,})={n_drop:,}")

    t0 = time.time()
    ex1 = extract_keys(s1['names'], s1['addrs'], strategy)
    if len(ex1) == 0:
        flat1 = np.zeros(0, dtype=np.int32)
        off1 = np.zeros(n1 + 1, dtype=np.int64)
    else:
        rec1 = ex1.index.to_numpy(dtype=np.int64)
        codes1 = pd.Index(uniques).get_indexer(ex1.to_numpy(dtype=object))
        keep = codes1 >= 0
        rec1 = rec1[keep]
        codes1 = codes1[keep]
        flat1 = codes1.astype(np.int32)
        counts1 = np.bincount(rec1, minlength=n1)
        off1 = np.zeros(n1 + 1, dtype=np.int64)
        np.cumsum(counts1, out=off1[1:])
        log(f"      S1 {len(flat1):,} keys "
            f"({100*len(flat1)/max(len(ex1),1):.1f}% in S23 vocab, "
            f"{time.time()-t0:.1f}s)")
    del ex1

    # postings, sorted by key: key_start[k]:key_start[k+1] -> records
    t0 = time.time()
    order = np.argsort(flat23, kind='stable')
    post_rec = np.repeat(np.arange(n23, dtype=np.int32), np.diff(off23))[order]
    counts_per_key = np.bincount(flat23, minlength=n_keys)
    key_start = np.zeros(n_keys + 1, dtype=np.int64)
    np.cumsum(counts_per_key, out=key_start[1:])
    del order, counts_per_key
    log(f"      postings built in {time.time()-t0:.1f}s")

    return dict(strategy=strategy, n1=n1, n23=n23, n_keys=n_keys,
                s1_flat=flat1, s1_off=off1, s23_flat=flat23, s23_off=off23,
                df=df, key_start=key_start, post_rec=post_rec,
                max_df=max_df, uniques=uniques)


def build_split_keys(data, out_path, strategies=STRATEGIES, max_df_ratio=0.01):
    """Build (or load) keys for all strategies of one split."""
    if os.path.exists(out_path):
        log(f"loading {os.path.basename(out_path)}")
        with open(out_path, 'rb') as f:
            return pickle.load(f)
    out = {}
    for st in strategies:
        log(f"  strategy {st}")
        out[st] = build_strategy(data['s1'], data['s23'], st, max_df_ratio)
    with open(out_path, 'wb') as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  wrote {os.path.basename(out_path)} "
        f"({os.path.getsize(out_path)/1e6:.0f} MB)")
    return out
