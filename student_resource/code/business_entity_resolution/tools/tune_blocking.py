#!/usr/bin/env python3
"""Measure blocking recall vs candidate cost for the rarity-weighted blocker.

per-record recall is macro-averaged over records that have >=1 true match, which
is exactly what F0.5 averages over — so it is the honest ceiling for the score.

Run:
  python3 tools/tune_blocking.py --sample 30000 --sweep all
  python3 tools/tune_blocking.py --sample 30000 --sweep ablation
"""
import argparse
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
from fast_keys import build_split_keys, STRATEGIES

CACHE = os.environ.get('ER_CACHE', '../../cache')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


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


def gt_for_records(gt, rec_idx):
    off, flat = gt['offsets'], gt['flat']
    return [flat[off[r]:off[r + 1]] for r in rec_idx]


def recall_of(rec, cand, truth_list):
    """(macro per-record recall, micro recall) for candidate pairs."""
    n = len(truth_list)
    order = np.argsort(rec, kind='stable')
    rec_s, cand_s = rec[order], cand[order]
    bounds = np.searchsorted(rec_s, np.arange(n + 1))
    per = np.zeros(n)
    has_gt = np.zeros(n, bool)
    tp = tot = 0
    for i, truth in enumerate(truth_list):
        tot += len(truth)
        if len(truth) == 0:
            continue
        has_gt[i] = True
        c = cand_s[bounds[i]:bounds[i + 1]]
        if len(c) == 0:
            continue
        hit = np.intersect1d(truth, c).size
        tp += hit
        per[i] = hit / len(truth)
    return (per[has_gt].mean() if has_gt.any() else 0.0,
            tp / tot if tot else 0.0)


def run_config(ks, n23, truth_list, cfg, country=None):
    t0 = time.time()
    c1, c23 = country if country else (None, None)
    res = fb.block_split(ks, len(truth_list), n23, cfg, chunk_records=20000,
                         country1=c1, country23=c23)
    dt = time.time() - t0
    counts_per_rec = np.diff(res['off'])
    rec = np.repeat(np.arange(len(truth_list), dtype=np.int32), counts_per_rec)
    macro, micro = recall_of(rec, res['cand'], truth_list)
    return dict(recall_macro=float(macro), recall_micro=float(micro),
                pairs_per_rec=float(counts_per_rec.sum()) / len(truth_list),
                raw_per_rec=res['n_raw_expansion'] / len(truth_list),
                block_s=dt, per_pool={k: list(v) for k, v in res['per_pool'].items()})


def cfg_of(base, **over):
    cfg = copy.deepcopy(base)
    for k, v in over.items():
        pool, field = k.split('_', 1)
        cfg['pool'][pool][field] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=30000)
    ap.add_argument('--sweep', default='all')
    ap.add_argument('--split', default='train')
    ap.add_argument('--out', default=os.path.join(CACHE, 'tune_blocking.json'))
    ap.add_argument('--from-keys', action='store_true')
    args = ap.parse_args()

    keys_path = os.path.join(CACHE, f'keys_{args.split}.pkl')
    if os.path.exists(keys_path):
        log(f"loading cached keys {keys_path}")
        keys = load_pkl(keys_path)
    else:
        log("building keys (cached on first run)")
        s1 = load_pkl(os.path.join(CACHE, f'norm_{args.split}_source1.pkl'))
        s2 = load_pkl(os.path.join(CACHE, f'norm_{args.split}_source2.pkl'))
        s3 = load_pkl(os.path.join(CACHE, f'norm_{args.split}_source3.pkl'))
        data = dict(s1=s1, s23=dict(names=np.concatenate([s2['names'], s3['names']]),
                                    addrs=np.concatenate([s2['addrs'], s3['addrs']])))
        del s2, s3
        keys = build_split_keys(data, keys_path, strategies=STRATEGIES)
        del data, s1

    n23 = int(keys['ntok']['n23'])
    if args.split != 'train':
        raise SystemExit("recall sweep needs the train split")
    n1_full = int(keys['ntok']['n1'])
    rng = np.random.default_rng(0)
    rec_idx = np.sort(rng.choice(n1_full, size=min(args.sample, n1_full),
                                 replace=False)).astype(np.int64)
    gt = load_pkl(os.path.join(CACHE, f'gt_{args.split}.pkl'))
    truth_list = gt_for_records(gt, rec_idx)
    n_nonempty = sum(1 for t in truth_list if len(t))
    log(f"n23={n23:,}  sample={len(rec_idx):,}  with_gt={n_nonempty:,} "
        f"({100*n_nonempty/len(rec_idx):.1f}%)")

    ks = subset_keys(keys, rec_idx)
    del keys
    c1_full, c23_full = ensure_country_codes(args.split)
    country = (c1_full[rec_idx], c23_full)

    base = {'pool': copy.deepcopy(fb.DEFAULT_POOL_CFG)}
    configs = [('BASE (budgets 200/100/60/120, k 60/30/35/35)', base)]
    if args.sweep in ('all', 'k'):
        for k in (30, 60, 120, 300):
            configs.append((f'ntok k={k}', cfg_of(base, ntok_k=k)))
        for k in (15, 30, 60):
            configs.append((f'npre k={k}', cfg_of(base, npre_k=k)))
        for k in (15, 35, 70):
            configs.append((f'anum k={k}', cfg_of(base, anum_k=k)))
        for k in (15, 35, 70):
            configs.append((f'aword k={k}', cfg_of(base, aword_k=k)))
    if args.sweep in ('all', 'df'):
        for mdf in (300, 1000, 3000, 10000):
            configs.append((f'ntok df<={mdf}', cfg_of(base, ntok_max_df=mdf)))
        for mdf in (100, 300, 1000):
            configs.append((f'npre df<={mdf}', cfg_of(base, npre_max_df=mdf)))
        for mdf in (300, 1000, 3000):
            configs.append((f'anum df<={mdf}', cfg_of(base, anum_max_df=mdf)))
        for mdf in (100, 300, 1000):
            configs.append((f'aword df<={mdf}', cfg_of(base, aword_max_df=mdf)))
    if args.sweep in ('all', 'budget'):
        for b in (60, 200, 600):
            configs.append((f'ntok budget={b}', cfg_of(base, ntok_budget=b)))

    results = []
    log(f"{'config':<44} {'mac_recall':>10} {'mic_recall':>10} "
        f"{'cand/rec':>9} {'raw/rec':>9} {'blk_s':>6}")
    for name, cfg in configs:
        r = run_config(ks, n23, truth_list, cfg, country=country)
        r['name'] = name
        r['cfg'] = cfg
        results.append(r)
        log(f"{name:<44} {r['recall_macro']:>10.4f} {r['recall_micro']:>10.4f} "
            f"{r['pairs_per_rec']:>9.1f} {r['raw_per_rec']:>9.1f} {r['block_s']:>6.1f}")

    if args.sweep in ('all', 'ablation'):
        log("\n=== pool ablation (each pool alone, k=400) ===")
        for only in [('ntok',), ('npre',), ('anum',), ('aword',),
                     ('ntok', 'npre'), ('anum', 'aword')]:
            cfg = copy.deepcopy(base)
            cfg['pool_order'] = only
            for st in only:
                cfg['pool'][st]['k'] = 400
            r = run_config(ks, n23, truth_list, cfg, country=country)
            r['name'] = 'pools:' + '+'.join(only)
            r['cfg'] = cfg
            results.append(r)
            log(f"{r['name']:<44} {r['recall_macro']:>10.4f} "
                f"{r['recall_micro']:>10.4f} {r['pairs_per_rec']:>9.1f} "
                f"{r['raw_per_rec']:>9.1f} {r['block_s']:>6.1f}")

    with open(args.out, 'w') as f:
        json.dump({'sample': len(rec_idx), 'n23': n23, 'n_nonempty': n_nonempty,
                   'results': results}, f, indent=1)
    log(f"wrote {args.out}")


if __name__ == '__main__':
    main()
