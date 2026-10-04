#!/usr/bin/env python3
"""Vectorized pair features — 46 of them, computed chunk-wise.

The legacy pipeline called a Python function per pair (~40 µs/pair).  Here the
same string metrics run through ``rapidfuzz.process.cpdist`` over whole chunks
(25–38 M pairs/s for the cheap metrics, 2–5 M/s for the token-set family, using
all cores), so we can afford both richer features and far more candidates.

Columns are assembled through a dict keyed by feature name and then written in
``FEATURES`` order, so adding a feature can never silently shift another one.

Feature list
------------
name:   jw, ratio, partial, token_sort, token_set, wratio, partial_token_set,
        len_ratio, jaccard, shared_keys, coverage_a, coverage_b, exact
address:jw, ratio, partial, token_sort, token_set,
        len_ratio, jaccard, shared_keys, coverage_a, coverage_b, exact
blocking: shared_keys_total, name_shared, addr_shared (pool-linked flags),
        pool_ntok, pool_npre, pool_anum, pool_aword
other:  is_source3
"""
import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

# The validated 41.  Measured at 60K records / cap 200 / alpha 30: 0.9031 val macro
# F0.5, vs 0.9005 with EXTRA_FEATURES added (same sample, same val split): the
# extra columns raise capacity enough to overfit earlier (best_iter 218 -> 161),
# so they are opt-in via `--extra-features` and must be re-validated at whatever
# training size is in play before they ship.
BASE_FEATURES = [
    # name string metrics
    'name_jw', 'name_ratio', 'name_partial', 'name_token_sort', 'name_token_set',
    'name_wratio', 'name_partial_token_set', 'name_len_ratio',
    # address string metrics
    'addr_jw', 'addr_ratio', 'addr_partial', 'addr_token_sort', 'addr_token_set',
    'addr_len_ratio',
    # blocking-derived overlap
    'shared_keys', 'name_shared', 'addr_shared', 'name_shared_n', 'addr_shared_a',
    'name_jaccard', 'addr_jaccard',
    'name_cov_a', 'name_cov_b', 'addr_cov_a', 'addr_cov_b',
    # pool membership
    'pool_ntok', 'pool_npre', 'pool_anum', 'pool_aword',
    # misc
    'is_source3', 'exact_name', 'exact_addr',
    # rarity-weighted overlap + shape / missingness
    'shared_wsum', 'name_len_diff', 'addr_len_diff', 'name_missing', 'addr_missing',
    # candidate context (chunk-local: how this pair compares with the record's
    # other candidates, and how contested the S2/S3 record is)
    'sim_rank', 'sim_gap_best', 'rec_best_sim', 'claim_pop',
]

# High-gain but overfit-prone: agreement between the two fields, average rarity
# of shared keys, and the ambiguity of the record's own best candidate.
EXTRA_FEATURES = [
    'sim_min_ts', 'sim_max_ts', 'wsum_per_shared', 'rec_second_sim', 'rec_top_gap',
]

# Default view: everything.  `set_features` lets the pipeline pin the run to the
# validated subset without touching the feature code (models store their own list).
FEATURES = list(BASE_FEATURES) + list(EXTRA_FEATURES)
N_FEATURES = len(FEATURES)
IDX = {name: i for i, name in enumerate(FEATURES)}


def set_features(names):
    """Pin the active feature set (order matters: it defines the matrix columns)."""
    global FEATURES, N_FEATURES, IDX
    FEATURES = list(names)
    N_FEATURES = len(FEATURES)
    IDX = {name: i for i, name in enumerate(FEATURES)}


def _cp(a, b, scorer, workers):
    """rapidfuzz pairwise (elementwise) scores for two equal-length lists."""
    return process.cpdist(a, b, scorer=scorer, workers=workers)


def compute_features(names1, addrs1, names23, addrs23, rec, cand, ctx,
                     chunk=2_000_000, workers=-1, progress=None):
    """Feature matrix for (rec, cand) pairs.

    ``names1``/``addrs1`` MUST be exactly the rows for this chunk (they are
    indexed by the local ``rec`` ids); ``names23``/``addrs23`` are the full S2+S3
    arrays indexed by the global ``cand`` ids.

    ctx keys: shared, shared_n, shared_a, nbits, abits (blocking arrays aligned
    with rec/cand), cnt_ntok_a, cnt_ntok_b, cnt_aword_a, cnt_aword_b (per-record
    key counts), n2 (source2 size).  Missing optional keys degrade gracefully.
    """
    n = len(rec)
    F = np.empty((n, N_FEATURES), dtype=np.float32)
    shared = _col(ctx.get('shared'), n, np.float32)
    shared_n = _col(ctx.get('shared_n'), n, np.float32)
    shared_a = _col(ctx.get('shared_a'), n, np.float32)
    nbits = _col(ctx.get('nbits'), n, np.uint8)
    abits = _col(ctx.get('abits'), n, np.uint8)
    have_counts = 'cnt_ntok_a' in ctx and 'cnt_aword_a' in ctx

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        r, c = rec[s:e], cand[s:e]
        an = names1[r].tolist()
        bn = names23[c].tolist()
        aa = addrs1[r].tolist()
        ba = addrs23[c].tolist()
        m = e - s

        col = {}
        col['name_jw'] = _cp(an, bn, JaroWinkler.similarity, workers)
        col['name_ratio'] = _cp(an, bn, fuzz.ratio, workers) / 100.0
        col['name_partial'] = _cp(an, bn, fuzz.partial_ratio, workers) / 100.0
        col['name_token_sort'] = _cp(an, bn, fuzz.token_sort_ratio, workers) / 100.0
        col['name_token_set'] = _cp(an, bn, fuzz.token_set_ratio, workers) / 100.0
        col['name_wratio'] = _cp(an, bn, fuzz.WRatio, workers) / 100.0
        col['name_partial_token_set'] = _cp(
            an, bn, fuzz.partial_token_set_ratio, workers) / 100.0
        col['addr_jw'] = _cp(aa, ba, JaroWinkler.similarity, workers)
        col['addr_ratio'] = _cp(aa, ba, fuzz.ratio, workers) / 100.0
        col['addr_partial'] = _cp(aa, ba, fuzz.partial_ratio, workers) / 100.0
        col['addr_token_sort'] = _cp(aa, ba, fuzz.token_sort_ratio, workers) / 100.0
        col['addr_token_set'] = _cp(aa, ba, fuzz.token_set_ratio, workers) / 100.0

        ln1 = np.fromiter((len(x) for x in an), np.int32, m)
        ln2 = np.fromiter((len(x) for x in bn), np.int32, m)
        la1 = np.fromiter((len(x) for x in aa), np.int32, m)
        la2 = np.fromiter((len(x) for x in ba), np.int32, m)
        col['name_len_ratio'] = np.minimum(ln1, ln2) / np.maximum(np.maximum(ln1, ln2), 1)
        col['addr_len_ratio'] = np.minimum(la1, la2) / np.maximum(np.maximum(la1, la2), 1)

        sn = shared_n[s:e]
        sa = shared_a[s:e]
        col['shared_keys'] = shared[s:e]
        col['name_shared'] = (nbits[s:e] > 0)
        col['addr_shared'] = (abits[s:e] > 0)
        col['name_shared_n'] = sn
        col['addr_shared_a'] = sa
        col['pool_ntok'] = (nbits[s:e] & 1) > 0
        col['pool_npre'] = (nbits[s:e] & 2) > 0
        col['pool_anum'] = (abits[s:e] & 1) > 0
        col['pool_aword'] = (abits[s:e] & 2) > 0

        if have_counts:
            na = ctx['cnt_ntok_a'][r].astype(np.float32)
            nb = ctx['cnt_ntok_b'][c].astype(np.float32)
            wa = ctx['cnt_aword_a'][r].astype(np.float32)
            wb = ctx['cnt_aword_b'][c].astype(np.float32)
            col['name_jaccard'] = sn / np.maximum(na + nb - sn, 1.0)
            col['addr_jaccard'] = sa / np.maximum(wa + wb - sa, 1.0)
            col['name_cov_a'] = sn / np.maximum(na, 1.0)
            col['name_cov_b'] = sn / np.maximum(nb, 1.0)
            col['addr_cov_a'] = sa / np.maximum(wa, 1.0)
            col['addr_cov_b'] = sa / np.maximum(wb, 1.0)
        else:
            for k in ('name_jaccard', 'addr_jaccard', 'name_cov_a', 'name_cov_b',
                      'addr_cov_a', 'addr_cov_b'):
                col[k] = np.zeros(m, dtype=np.float32)

        col['is_source3'] = (c >= ctx.get('n2', 0))
        col['exact_name'] = np.asarray(an, dtype=object) == np.asarray(bn, dtype=object)
        col['exact_addr'] = np.asarray(aa, dtype=object) == np.asarray(ba, dtype=object)

        # rarity-weighted shared evidence (already computed by the blocker for
        # the cap ranking) + shape / missingness of the two records
        wsum = _col(ctx.get('wsum'), n, np.float32)[s:e]
        col['shared_wsum'] = wsum
        col['sim_min_ts'] = np.minimum(col['name_token_set'], col['addr_token_set'])
        col['sim_max_ts'] = np.maximum(col['name_token_set'], col['addr_token_set'])
        col['wsum_per_shared'] = wsum / np.maximum(shared[s:e], 1.0)
        col['name_len_diff'] = (np.abs(ln1 - ln2)
                                / np.maximum(np.maximum(ln1, ln2), 1))
        col['addr_len_diff'] = (np.abs(la1 - la2)
                                / np.maximum(np.maximum(la1, la2), 1))
        col['name_missing'] = np.minimum(ln1, ln2) == 0
        col['addr_missing'] = np.minimum(la1, la2) == 0

        for j, name in enumerate(FEATURES):
            if name in col:
                F[s:e, j] = col[name]
        if progress:
            progress(f"features {e:,}/{n:,}", e / n)
    _context_features(F, rec, cand)
    return F, rec, cand


def _context_features(F, rec, cand):
    """Fill the candidate-context columns in place (chunk-local).

    For every pair we know the *whole candidate list of its record*, and (within
    the chunk) how many other S1 entities claim the same S2/S3 record.  A pair
    whose similarity ranks below its siblings is far likelier to be a false merge
    under this precision-weighted metric, and a heavily contested S2/S3 record is
    likelier to be an unmatched (noise) record.
    """
    n = len(rec)
    if n == 0:
        return F
    sim = np.maximum(F[:, IDX['name_ratio']], F[:, IDX['addr_ratio']])
    order = np.lexsort((-sim, rec))          # record-major, best similarity first
    rec_s, sim_s = rec[order], sim[order]
    _, first, cnts = np.unique(rec_s, return_index=True, return_counts=True)
    sizes = np.repeat(cnts.astype(np.int64), cnts)
    rank = np.arange(n, dtype=np.int64) - np.repeat(first, cnts)
    best = np.repeat(np.maximum.reduceat(sim_s, first), cnts)
    # second-best similarity of the record (in sorted order rank 1 is next); a
    # single-candidate record gets its own best, so the top gap is 0 = ambiguous
    second = np.where(cnts >= 2, sim_s[np.minimum(first + 1, len(sim_s) - 1)],
                      sim_s[first])
    second_rep = np.repeat(second, cnts)

    rank_norm = np.empty(n, dtype=np.float32)
    rank_norm[order] = (rank / np.maximum(sizes - 1, 1)).astype(np.float32)
    best_out = np.empty(n, dtype=np.float32)
    best_out[order] = best.astype(np.float32)

    pop = np.bincount(cand).astype(np.float32)
    F[:, IDX['sim_rank']] = rank_norm
    F[:, IDX['sim_gap_best']] = best_out - sim
    F[:, IDX['rec_best_sim']] = best_out
    F[:, IDX['claim_pop']] = np.log1p(pop[cand])
    if 'rec_second_sim' in IDX:          # absent when the validated subset is pinned
        F[:, IDX['rec_second_sim']] = second_rep.astype(np.float32)
        F[:, IDX['rec_top_gap']] = (best_out - second_rep).astype(np.float32)
    return F


def select_features(X, names):
    """Re-order/subset a feature matrix to a saved model's feature list.

    Lets a model trained before a new feature was added keep predicting: the
    matrix gains columns, the model's view of it does not change.
    """
    if list(names) == FEATURES:
        return X
    return X[:, [IDX[name] for name in names]]


def _col(arr, n, dtype):
    if arr is None:
        return np.zeros(n, dtype=dtype)
    out = np.asarray(arr, dtype=dtype)
    if len(out) != n:
        raise ValueError(f"context array has length {len(out)}, expected {n}")
    return out


def length_caches(names, addrs):
    """int32 char-length arrays (faster than recomputing per chunk)."""
    ln = np.fromiter((len(x) for x in names), dtype=np.int32, count=len(names))
    la = np.fromiter((len(x) for x in addrs), dtype=np.int32, count=len(addrs))
    return ln, la


def make_labels(rec, cand, gt, n23, chunk=20_000_000):
    """1 if the candidate is a true match of the record (searchsorted lookup)."""
    off, flat = gt['offsets'], gt['flat']
    counts = np.diff(off)
    truth_rec = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
    truth_comp = np.sort(truth_rec * n23 + flat.astype(np.int64))
    y = np.zeros(len(rec), dtype=np.int8)
    if len(truth_comp) == 0:
        return y
    for s in range(0, len(rec), chunk):
        e = min(s + chunk, len(rec))
        comp = rec[s:e].astype(np.int64) * n23 + cand[s:e]
        pos = np.searchsorted(truth_comp, comp)
        pos = np.minimum(pos, len(truth_comp) - 1)
        y[s:e] = (truth_comp[pos] == comp)
    return y
