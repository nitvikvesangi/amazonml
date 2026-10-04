#!/usr/bin/env python3
"""Recall vs candidate budget: what does each candidate actually buy us?

For a given key-selection policy, generate *all* candidates (no per-pool cap),
then measure per-record blocking recall when we keep only the best N per record
under two rankings:

  global  top-N by summed rarity weight across all pools
  quota   per-pool quotas (40/15/20/25% of N) so no pool starves the others

Output picks the policy + N that maximizes recall per candidate.

Run:  python3 tools/curve_blocking.py --sample 20000
"""
import copy
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

import fast_block as fb
from fast_data import ensure_country_codes

CACHE = os.environ.get('ER_CACHE', '../../cache')

POLICIES = {
    'tight': {'ntok': (200, 3000), 'npre': (100, 300),
              'anum': (60, 1000), 'aword': (120, 300)},
    'medium': {'ntok': (400, 10000), 'npre': (200, 2000),
               'anum': (100, 2000), 'aword': (250, 2000)},
    'generous': {'ntok': (1000, 30000), 'npre': (400, 5000),
                 'anum': (200, 5000), 'aword': (500, 5000)},
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def per_record_topk(rec, score, k):
    """Boolean mask of the top-k pairs per record ranked by score (desc)."""
    order = np.lexsort((-score, rec))
    rec_s = rec[order]
    _, first, cnts = np.unique(rec_s, return_index=True, return_counts=True)
    rank = np.arange(len(rec_s), dtype=np.int64) - np.repeat(first, cnts)
    mask = np.zeros(len(rec), dtype=bool)
    mask[order[rank < k]] = True
    return mask


def quota_mask(rec, wsum, bits, n_budget, ratios):
    """Per-pool quotas; pools that come up short leave slots unused."""
    sel = np.zeros(len(rec), dtype=bool)
    for j, r in enumerate(ratios):
        k = max(int(round(n_budget * r)), 1)
        in_pool = (bits >> j) & 1 > 0
        if not in_pool.any():
            continue
        sub = in_pool.copy()
        m = per_record_topk(rec[in_pool], wsum[in_pool], k)
        idx = np.flatnonzero(in_pool)
        sel[idx[m]] = True
    return sel


def recall_of(rec, cand, truth_list):
    n = len(truth_list)
    order = np.argsort(rec, kind='stable')
    rec_s, cand_s = rec[order], cand[order]
    bounds = np.searchsorted(rec_s, np.arange(n + 1))
    per = np.zeros(n)
    has = np.zeros(n, bool)
    tp = tot = 0
    for i, truth in enumerate(truth_list):
        tot += len(truth)
        if len(truth) == 0:
            continue
        has[i] = True
        c = cand_s[bounds[i]:bounds[i + 1]]
        if len(c):
            hit = np.intersect1d(truth, c).size
            tp += hit
            per[i] = hit / len(truth)
    return (per[has].mean() if has.any() else 0.0, tp / tot if tot else 0.0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--chunk', type=int, default=4000)
    ap.add_argument('--policies', default='tight,medium,generous')
    ap.add_argument('--ns', default='25,50,100,150,200,300,400,600')
    ap.add_argument('--out', default=os.path.join(CACHE, 'curve_blocking.json'))
    args = ap.parse_args()

    keys = load_pkl(os.path.join(CACHE, 'keys_train.pkl'))
    n23 = int(keys['ntok']['n23'])
    n1_full = int(keys['ntok']['n1'])
    rng = np.random.default_rng(0)
    rec_idx = np.sort(rng.choice(n1_full, size=args.sample, replace=False)).astype(np.int64)
    gt = load_pkl(os.path.join(CACHE, 'gt_train.pkl'))
    truth_list = [gt['flat'][gt['offsets'][r]:gt['offsets'][r + 1]] for r in rec_idx]
    c1, c23 = ensure_country_codes('train')
    country1, country23 = c1[rec_idx], c23

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from tune_blocking import subset_keys
    ks = subset_keys(keys, rec_idx)
    del keys
    n = len(rec_idx)
    log(f"sample={n:,}  n23={n23:,}  "
        f"with_gt={sum(1 for t in truth_list if len(t)):,}")

    Ns = [int(x) for x in args.ns.split(',')]
    ratios = (0.40, 0.15, 0.20, 0.25)
    out = {'sample': n, 'n23': n23, 'policies': {}}

    for pname in args.policies.split(','):
        pool = {st: {'budget': b, 'max_df': mdf, 'k': None}
                for st, (b, mdf) in POLICIES[pname].items()}
        cfg = {'pool': pool}
        t0 = time.time()
        res = fb.block_split(ks, n, n23, cfg, chunk_records=args.chunk,
                             country1=country1, country23=country23)
        dt = time.time() - t0
        counts_per_rec = np.diff(res['off'])
        rec = np.repeat(np.arange(n, dtype=np.int32), counts_per_rec)
        cand = res['cand']
        wsum, cnt = res['wsum'], res['shared']
        bits = (res['nbits'].astype(np.int32) | (res['abits'].astype(np.int32) << 2))
        log(f"\n=== policy {pname}: cand/rec={len(cand)/n:.1f} "
            f"raw/rec={res['n_raw_expansion']/n:.1f} block_s={dt:.1f} ===")
        log(f"{'N':>5} {'global_macro':>13} {'quota_macro':>12} "
            f"{'global_micro':>13} {'kept_glob':>10} {'kept_quota':>11}")
        rows = []
        for N in Ns:
            if N > max(counts_per_rec.max(), 1) * 2:
                pass
            m_g = per_record_topk(rec, wsum, N)
            mg, mug = recall_of(rec[m_g], cand[m_g], truth_list)
            m_q = quota_mask(rec, wsum, bits, N, ratios)
            mq, muq = recall_of(rec[m_q], cand[m_q], truth_list)
            rows.append(dict(N=N, global_macro=float(mg), global_micro=float(mug),
                             quota_macro=float(mq), quota_micro=float(muq),
                             kept_global=float(m_g.sum()) / n,
                             kept_quota=float(m_q.sum()) / n))
            log(f"{N:>5} {mg:>13.4f} {mq:>12.4f} {mug:>13.4f} "
                f"{m_g.sum()/n:>10.1f} {m_q.sum()/n:>11.1f}")
        out['policies'][pname] = dict(cand_per_rec=float(len(cand) / n),
                                      raw_per_rec=res['n_raw_expansion'] / n,
                                      block_s=dt, rows=rows)
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=1)
    log(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
