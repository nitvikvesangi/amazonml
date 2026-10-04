#!/usr/bin/env python3
"""Does a similarity-blended cap beat a pure-rarity cap at the same N?

The 200-candidate cap is ranked by summed rarity weight.  A shrink experiment
showed ~70% of actually-predicted matches sit *below* #100 under that ranking, so
the cap may be discarding true matches that a cheap similarity would have kept.

This measures per-record blocking recall at a fixed cap for
    score = wsum + alpha * max(name_ratio, addr_ratio)
with alpha = 0 (current) .. pure similarity.

Run:  python3 tools/measure_cap.py --sample 20000 --cap 200
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

import fast_block as fb
import fast_data as fd
import pipeline_fast as pf
from tune_blocking import subset_keys, gt_for_records, recall_of

CACHE = fd.CACHE


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=20000)
    ap.add_argument('--cap', type=int, default=200)
    ap.add_argument('--alphas', default='0,0.5,1,3,10,1000')
    ap.add_argument('--chunk-records', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=os.path.join(CACHE, 'measure_cap.json'))
    args = ap.parse_args()

    data = fd.load_split('train', with_ids=False)
    keys = pf.load_keys('train')
    gt = pf.load_pkl(os.path.join(CACHE, 'gt_train.pkl'))
    rec_idx = pf.spread_sample(data['n1'], args.sample, seed=args.seed)
    truth = gt_for_records(gt, rec_idx)
    n = len(rec_idx)
    log(f"sample={n:,} records  cap={args.cap}  alphas={args.alphas}")

    ks = subset_keys(keys, rec_idx)
    del keys
    cfg = {'pool': {st: dict(pf.POOL_CFG[st]) for st in fb.POOLS}}
    for st in cfg['pool']:
        cfg['pool'][st]['k'] = None
    t0 = time.time()
    res = fb.block_split(ks, n, data['n23'], cfg, chunk_records=args.chunk_records,
                         country1=data['s1']['countries'][rec_idx],
                         country23=data['s23']['countries'])
    log(f"uncapped: {res['n_pairs']:,} pairs ({res['n_pairs']/n:.1f}/rec), "
        f"raw expansion {res['n_raw_expansion']/n:.0f}/rec, {time.time()-t0:.0f}s")

    t0 = time.time()
    px = dict(names1=data['s1']['names'][rec_idx], names23=data['s23']['names'],
              addrs1=data['s1']['addrs'][rec_idx], addrs23=data['s23']['addrs'])
    sim = pf.cap_score(res, n, px, args.cap, 1.0) - res['wsum']
    log(f"similarity computed in {time.time()-t0:.0f}s "
        f"(mean {float(sim.mean()):.3f}, p90 {float(np.percentile(sim, 90)):.3f})")

    alphas = [float(x) for x in args.alphas.split(',')]
    wsum = res['wsum']
    rows = []
    log(f"\n{'alpha':>8} {'macro':>8} {'micro':>8} {'cand/rec':>9} {'kept':>9}")
    for a in alphas:
        score = None if a == 0 else (wsum + np.float32(a) * sim)
        capped = fb.cap_result(res, n, data['n23'], args.cap, score=score)
        cnt = np.diff(capped['off'])
        rec = np.repeat(np.arange(n, dtype=np.int32), cnt)
        macro, micro = recall_of(rec, capped['cand'], truth)
        rows.append(dict(alpha=a, recall_macro=float(macro), recall_micro=float(micro),
                         cand_per_rec=float(cnt.sum()) / n))
        log(f"{a:>8} {macro:>8.4f} {micro:>8.4f} {cnt.sum()/n:>9.1f} "
            f"{int((cnt > 0).sum()):>9,}")
    best = max(rows, key=lambda r: r['recall_macro'])
    log(f"\nbest alpha on this sample: {best['alpha']} "
        f"-> macro recall {best['recall_macro']:.4f}")
    with open(args.out, 'w') as f:
        json.dump(dict(sample=n, cap=args.cap, rows=rows), f, indent=1)
    log(f"wrote {args.out}")


if __name__ == '__main__':
    main()
