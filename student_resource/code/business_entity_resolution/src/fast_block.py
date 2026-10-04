#!/usr/bin/env python3
"""Vectorized budgeted blocking with rarity-weighted, per-pool ranking.

Why it is built this way
------------------------
1. *Budgets*: for each record the keys are sorted by document frequency and kept
   rarest-first until a pair budget is spent.  Rarity is what separates a true
   match (a distinctive token, a street number) from noise, and the budget bounds
   the expansion cost.  Measured on train: 98.6% of distinct name tokens have
   df<=100 but they cover only 13.7% of token occurrences, so the tail must be
   capped or each record drowns in common-word candidates.

2. *Rarity-weighted ranking*: counting shared keys alone is useless when 50000
   records share "services".  Each key contributes ``log1p(n23/df)`` instead, so
   a link through a near-unique key outranks a link through a common one.

3. *Independent pools with per-pool top-K*: name-token, name-prefix,
   address-number and address-word pools each contribute their own best K.
   Ranking globally lets the noisiest pool take every slot (measured: recall at
   200 candidates collapsed to 6.9%); per-pool quotas keep DBA/trade-name
   matches reachable through the address pools.

Country is a hard filter (a business lives in one country; true matches are
always same-country).
"""
import time

import numpy as np

NAME_POOLS = ('ntok', 'npre')
ADDR_POOLS = ('anum', 'aword')
POOLS = NAME_POOLS + ADDR_POOLS

DEFAULT_POOL_CFG = {
    'ntok': {'budget': 200, 'max_df': 3000, 'k': 60},
    'npre': {'budget': 100, 'max_df': 300, 'k': 30},
    'anum': {'budget': 60, 'max_df': 1000, 'k': 35},
    'aword': {'budget': 120, 'max_df': 300, 'k': 35},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def subset_keys(keys, rec_idx):
    """Restrict the S1 side of every strategy to the given record indices."""
    out = {}
    for st, idx in keys.items():
        off, flat = idx['s1_off'], idx['s1_flat']
        starts = off[rec_idx]
        counts = off[rec_idx + 1] - off[rec_idx]
        total = int(counts.sum())
        new_off = np.zeros(len(rec_idx) + 1, dtype=np.int64)
        np.cumsum(counts, out=new_off[1:])
        if total:
            ends = np.repeat(np.cumsum(counts), counts)
            s = ends - np.repeat(counts, counts)
            pos = np.arange(total, dtype=np.int64) - s + np.repeat(starts, counts)
            new_flat = flat[pos]
        else:
            new_flat = np.zeros(0, dtype=np.int32)
        new = dict(idx)
        new['s1_flat'] = new_flat
        new['s1_off'] = new_off
        new['n1'] = len(rec_idx)
        out[st] = new
    return out


def per_record_topk_mask(rec, score, k):
    """Boolean mask keeping the top-k pairs per record, ranked by score desc."""
    if k is None or len(rec) == 0:
        return np.ones(len(rec), dtype=bool)
    order = np.lexsort((-score, rec))
    rec_s = rec[order]
    _, first, cnts = np.unique(rec_s, return_index=True, return_counts=True)
    rank = np.arange(len(rec_s), dtype=np.int64) - np.repeat(first, cnts)
    mask = np.zeros(len(rec), dtype=bool)
    mask[order[rank < k]] = True
    return mask


def cap_result(res, n1, n23, cap, score=None):
    """Per-record candidate cap; rebuilds the CSR.

    ``score`` overrides the default ranking key (summed rarity weight).  Passing
    a score that blends rarity with a cheap similarity keeps true matches that
    would otherwise sit below the cap (measured: 70% of predicted matches ranked
    below #100 by rarity weight alone).
    """
    if cap is None:
        return res
    cnt = np.diff(res['off'])
    rec = np.repeat(np.arange(n1, dtype=np.int32), cnt)
    if score is None:
        score = res['wsum']
    keep = per_record_topk_mask(rec, score, cap)
    idx = np.flatnonzero(keep)
    rec_k = rec[idx]
    new_off = np.zeros(n1 + 1, dtype=np.int64)
    np.cumsum(np.bincount(rec_k, minlength=n1), out=new_off[1:])
    out = dict(res)
    out.update(cand=res['cand'][idx], off=new_off, shared=res['shared'][idx],
               wsum=res['wsum'][idx], nbits=res['nbits'][idx],
               abits=res['abits'][idx], n_pairs=int(len(idx)))
    # every per-pair array must be subset together, or lengths silently diverge
    for k in ('shared_n', 'shared_a'):
        if k in res:
            out[k] = res[k][idx]
    return out


def key_weights(df, n23):
    """log1p(n23/df): how much rarer than average this key is (float32)."""
    return np.log1p(n23 / np.maximum(df.astype(np.float64), 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# key selection
# ---------------------------------------------------------------------------

def select_keys(idx, r0, r1, budget, max_df):
    """Rarest-first key selection for records [r0, r1) of one strategy.

    Returns (rec int32, key int32, df int32) of the kept key occurrences.
    """
    off = idx['s1_off']
    lo, hi = int(off[r0]), int(off[r1])
    flat = idx['s1_flat'][lo:hi]
    if len(flat) == 0:
        z = np.zeros(0, dtype=np.int32)
        return z, z, z

    counts = off[r0 + 1:r1 + 1] - off[r0:r1]
    rec = np.repeat(np.arange(r0, r1, dtype=np.int32), counts)
    dfk = idx['df'][flat]

    valid = (dfk > 0) & (dfk <= max_df)
    if not valid.all():
        rec, flat, dfk = rec[valid], flat[valid], dfk[valid]
    if len(flat) == 0:
        z = np.zeros(0, dtype=np.int32)
        return z, z, z

    order = np.lexsort((dfk, rec))            # by record, then df ascending
    rec_s, flat_s, df_s = rec[order], flat[order], dfk[order]

    # spend the budget rarest-first
    csum = np.cumsum(df_s, dtype=np.int64)
    uniq_rec, first_pos = np.unique(rec_s, return_index=True)
    base = np.where(first_pos > 0, csum[first_pos - 1], 0)
    rec_pos = np.searchsorted(uniq_rec, rec_s)
    before = csum - df_s - base[rec_pos]
    keep = before < budget
    return rec_s[keep], flat_s[keep], df_s[keep]


def gather_pairs(idx, rec, key, w, max_out=None):
    """Expand (record, key) rows into (record, candidate, weight) triples."""
    if len(key) == 0:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32)
    ks = idx['key_start'][key]
    counts = idx['key_start'][key + 1] - ks
    total = int(counts.sum())
    if max_out is not None and total > max_out:
        csum = np.cumsum(counts)
        keep = np.concatenate([[True], csum[:-1] < max_out])
        rec, key, counts, w = rec[keep], key[keep], counts[keep], w[keep]
        total = int(counts.sum())
    if total == 0:
        z = np.zeros(0, dtype=np.int32)
        return z, z, np.zeros(0, dtype=np.float32)
    ends = np.repeat(np.cumsum(counts), counts)
    starts = ends - np.repeat(counts, counts)
    pos = np.arange(total, dtype=np.int64) - starts + np.repeat(ks, counts)
    return np.repeat(rec, counts), idx['post_rec'][pos], np.repeat(w[key], counts)


# ---------------------------------------------------------------------------
# per-pool -> unique pairs with count + weight
# ---------------------------------------------------------------------------

def _unique_stats(rec, cand, w_occ, n23):
    """Unique (rec, cand) pairs with shared-key count and summed rarity weight."""
    if len(rec) == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32),
                np.zeros(0, dtype=np.float32))
    comp = rec.astype(np.int64) * n23 + cand
    order = np.argsort(comp, kind='stable')
    comp_s = comp[order]
    w_s = w_occ[order]
    first = np.concatenate([[True], comp_s[1:] != comp_s[:-1]])
    idxs = np.flatnonzero(first)
    up = comp_s[idxs]
    cnt = np.diff(np.append(idxs, len(comp_s))).astype(np.int32)
    wsum = np.add.reduceat(w_s.astype(np.float64), idxs).astype(np.float32)
    return up, cnt, wsum


def _pool_pairs(idx, r0, r1, cfg, n23, weights, country1=None, country23=None):
    """Unique pairs + count + rarity weight for one pool over [r0, r1)."""
    rec, key, _ = select_keys(idx, r0, r1, cfg['budget'], cfg['max_df'])
    pr, pc, pw = gather_pairs(idx, rec, key, weights)
    n_raw = len(pc)
    if country1 is not None and n_raw:
        keep = country23[pc] == country1[pr]
        if not keep.all():
            pr, pc, pw = pr[keep], pc[keep], pw[keep]
    up, cnt, wsum = _unique_stats(pr, pc, pw, n23)
    return up, cnt, wsum, n_raw


def _pool_topk(up, cnt, wsum, n23, k):
    """Keep the k best pairs per record: rerarity weight desc, then count desc."""
    if k is None or len(up) <= k:
        return None
    rec = (up // n23).astype(np.int32)
    order = np.lexsort((-cnt, -wsum, rec))
    rec_s = rec[order]
    _, first_idx, cnts = np.unique(rec_s, return_index=True, return_counts=True)
    rank = np.arange(len(rec_s), dtype=np.int64) - np.repeat(first_idx, cnts)
    return order[rank < k]


def _union_stats(parts, n23):
    """Union per-pool (up, cnt, wsum): sum counts/weights, OR the pool bits."""
    parts = [p for p in parts if len(p[0]) > 0]
    if not parts:
        z = np.zeros(0, dtype=np.int64)
        return z, np.zeros(0, np.int32), np.zeros(0, np.float32), np.zeros(0, np.uint8)
    if len(parts) == 1:
        up, cnt, wsum = parts[0]
        return up, cnt, wsum, np.full(len(up), 1, dtype=np.uint8)
    allp = np.concatenate([p[0] for p in parts])
    allc = np.concatenate([p[1] for p in parts])
    allw = np.concatenate([p[2] for p in parts])
    pool_of = np.concatenate([np.full(len(p[0]), j, dtype=np.uint8)
                              for j, p in enumerate(parts)])
    order = np.argsort(allp, kind='stable')
    allp, allc, allw, pool_of = allp[order], allc[order], allw[order], pool_of[order]
    first = np.concatenate([[True], allp[1:] != allp[:-1]])
    idxs = np.flatnonzero(first)
    up = allp[idxs]
    cnt = np.add.reduceat(allc.astype(np.int64), idxs).astype(np.int32)
    wsum = np.add.reduceat(allw.astype(np.float64), idxs).astype(np.float32)
    bits = np.zeros(len(up), dtype=np.uint8)
    for j in range(len(parts)):
        hit = np.add.reduceat((pool_of == j).astype(np.uint8), idxs) > 0
        bits |= (hit.astype(np.uint8) << j)
    return up, cnt, wsum, bits


# ---------------------------------------------------------------------------
# full pass
# ---------------------------------------------------------------------------

def block_split(keys, n1, n23, config, chunk_records=40000, progress=None,
                country1=None, country23=None):
    """Block every S1 record; returns CSR candidates + per-pair stats.

    config: {'pool': {name: {budget,max_df,k}}, 'pool_order': tuple}
    Result keys: cand, off (CSR), shared, wsum, nbits, abits (+ counters).
    """
    pool_cfg = config['pool']
    pools = config.get('pool_order', POOLS)
    cand_parts, cnt_parts, w_parts, nbits_parts, abits_parts = [], [], [], [], []
    cn_parts, ca_parts = [], []      # distinct shared keys per group (name / address)
    total_pairs = raw_pairs = 0
    sel_stats = {}
    if progress:
        progress(f"blocking {n1:,} records", 0.0)

    for r0 in range(0, n1, chunk_records):
        r1 = min(r0 + chunk_records, n1)
        nsel = []
        ares = []
        for st in pools:
            idx = keys[st]
            if 'weights' not in idx:
                idx['weights'] = key_weights(idx['df'], n23)
            cfg = pool_cfg[st]
            up, cnt, wsum, n_raw = _pool_pairs(idx, r0, r1, cfg, n23,
                                               idx['weights'], country1, country23)
            raw_pairs += n_raw
            sel_stats.setdefault(st, []).append((len(up), n_raw))
            sel = _pool_topk(up, cnt, wsum, n23, cfg.get('k'))
            if sel is not None:
                up, cnt, wsum = up[sel], cnt[sel], wsum[sel]
            (nsel if st in NAME_POOLS else ares).append((up, cnt, wsum))
        up_n, cnt_n, w_n, bits_n = _union_stats(nsel, n23)
        up_a, cnt_a, w_a, bits_a = _union_stats(ares, n23)
        up, cnt, wsum, bn, ba, cn, ca = _union_two(up_n, cnt_n, w_n, bits_n,
                                                  up_a, cnt_a, w_a, bits_a)
        total_pairs += len(up)
        cand_parts.append(up)
        cnt_parts.append(cnt)
        w_parts.append(wsum)
        nbits_parts.append(bn)
        abits_parts.append(ba)
        cn_parts.append(cn)
        ca_parts.append(ca)
        if progress:
            progress(f"blocking {r1:,}/{n1:,}", r1 / n1)

    cand64 = np.concatenate(cand_parts) if cand_parts else np.zeros(0, np.int64)
    counts = np.concatenate(cnt_parts) if cnt_parts else np.zeros(0, np.int32)
    wsum = np.concatenate(w_parts) if w_parts else np.zeros(0, np.float32)
    nbits = np.concatenate(nbits_parts) if nbits_parts else np.zeros(0, np.uint8)
    abits = np.concatenate(abits_parts) if abits_parts else np.zeros(0, np.uint8)
    shared_n = np.concatenate(cn_parts) if cn_parts else np.zeros(0, np.int32)
    shared_a = np.concatenate(ca_parts) if ca_parts else np.zeros(0, np.int32)
    rec = (cand64 // n23).astype(np.int32)
    cand = (cand64 % n23).astype(np.int32)

    counts_per_rec = np.bincount(rec, minlength=n1)
    off = np.zeros(n1 + 1, dtype=np.int64)
    np.cumsum(counts_per_rec, out=off[1:])

    return dict(cand=cand, off=off, shared=counts, wsum=wsum, nbits=nbits,
                abits=abits, shared_n=shared_n, shared_a=shared_a,
                n_pairs=int(len(cand)),
                n_raw_expansion=int(raw_pairs),
                n_unique_before_cap=int(total_pairs),
                n_with_cand=int((counts_per_rec > 0).sum()),
                per_pool={k: (sum(a for a, _ in v), sum(b for _, b in v))
                          for k, v in sel_stats.items()})


def _union_two(up_n, cnt_n, w_n, bits_n, up_a, cnt_a, w_a, bits_a):
    """Union the name-group and address-group results.

    Returns (up, cnt, wsum, name_bits, addr_bits, name_shared_keys,
    addr_shared_keys): the per-group shared-key counts are kept so downstream
    features can tell a name overlap from an address overlap.
    """
    if len(up_n) == 0:
        return (up_a, cnt_a, w_a, np.zeros(len(up_a), np.uint8), bits_a,
                np.zeros(len(up_a), np.int32), cnt_a)
    if len(up_a) == 0:
        return (up_n, cnt_n, w_n, bits_n, np.zeros(len(up_n), np.uint8),
                cnt_n, np.zeros(len(up_n), np.int32))
    allp = np.concatenate([up_n, up_a])
    allc = np.concatenate([cnt_n, cnt_a])
    allw = np.concatenate([w_n, w_a])
    src_n = np.concatenate([np.ones(len(up_n), np.uint8),
                            np.zeros(len(up_a), np.uint8)])
    order = np.argsort(allp, kind='stable')
    allp, allc, allw, src_n = allp[order], allc[order], allw[order], src_n[order]
    first = np.concatenate([[True], allp[1:] != allp[:-1]])
    idxs = np.flatnonzero(first)
    up = allp[idxs]
    cnt = np.add.reduceat(allc.astype(np.int64), idxs).astype(np.int32)
    wsum = np.add.reduceat(allw.astype(np.float64), idxs).astype(np.float32)
    cn = np.add.reduceat((allc * src_n).astype(np.int64), idxs).astype(np.int32)
    ca = np.add.reduceat((allc * (1 - src_n)).astype(np.int64), idxs).astype(np.int32)
    is_n = np.add.reduceat(src_n, idxs) > 0
    is_a = np.add.reduceat((1 - src_n).astype(np.uint8), idxs) > 0
    bn = np.zeros(len(up), dtype=np.uint8)
    ba = np.zeros(len(up), dtype=np.uint8)
    if is_n.any():
        pos = np.minimum(np.searchsorted(up_n, up), len(up_n) - 1)
        bn[is_n] = bits_n[pos[is_n]]
    if is_a.any():
        pos = np.minimum(np.searchsorted(up_a, up), len(up_a) - 1)
        ba[is_a] = bits_a[pos[is_a]]
    return up, cnt, wsum, bn, ba, cn, ca
