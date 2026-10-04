#!/usr/bin/env python3
"""Apply the trained cross-encoder to the canonical holdout and sweep rules.

CE scores are computed for every holdout pair with prob >= --lo (rescue band +
near-threshold cleaning candidates), then three rule families are evaluated with
the exact per-record macro-F0.5 scorer (postproc.score):

  base   : prob >= 0.94                       (frozen US/India rule)
  rescue : base | (prob >= r_lo & ce >= c)    (add true pairs the GBDT missed)
  clean  : base & ~(prob < z_hi & ce < c)     (drop false accepts near the edge)

All deltas are on the *union-clean* record subset: the 500k model-train mask and
the CE's own 100k seed-7 training sample removed.  VERDICT GO means the best
clean delta >= +0.006, the bar for spending a test-side rebuild.

Run from code/business_entity_resolution (under .venv-sem, torch+MPS):
    .venv-sem/bin/python3 tools/ce_holdout_eval.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import postproc as pp            # noqa: E402
import pipeline_fast as pf       # noqa: E402
from sem_probe import read_rows, log   # noqa: E402

N_TRAIN = 2_206_821
BASE_THR = 0.94
BAR = 0.006


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump',
                    default='../../output/eval41a30_fixed/pairs_base41_500k.npz')
    ap.add_argument('--dataset', default='../../dataset/train')
    ap.add_argument('--model-dir', default='../../output/ce_model')
    ap.add_argument('--base-model',
                    default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    ap.add_argument('--lo', type=float, default=0.30)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--out', default='../../cache/ce_holdout.npz')
    ap.add_argument('--report', default='../../output/ce_holdout_eval.json')
    args = ap.parse_args()

    d = pp.load_dump(args.dump)
    rec, cand, prob, rec_idx = d['rec'], d['cand'], d['prob'], d['rec_idx']
    n = d['n']
    tp = d['tp_flag'] == 1
    sel = np.flatnonzero(prob >= args.lo)
    log(f"holdout {os.path.basename(args.dump)}: {len(prob):,} pairs, "
        f"{n:,} records | CE-scoring {len(sel):,} (prob>={args.lo}, "
        f"{int(tp[sel].sum()):,} true)")

    n2path = os.path.join(args.dataset, 'train_source2.tsv')
    n2 = sum(1 for _ in open(n2path, 'rb')) - 1
    need1 = set(rec_idx[rec[sel]].tolist())
    is3 = cand[sel] >= n2
    need2 = set(cand[sel][~is3].tolist())
    need3 = set((cand[sel][is3] - n2).tolist())
    t0 = time.time()
    rows1, _ = read_rows(os.path.join(args.dataset, 'train_source1.tsv'), need1)
    rows2, _ = read_rows(n2path, need2)
    rows3, _ = read_rows(os.path.join(args.dataset, 'train_source3.tsv'), need3)
    log(f"strings s1={len(need1):,} s2={len(need2):,} s3={len(need3):,} "
        f"read in {time.time()-t0:.0f}s")

    A, B = [], []
    for k in sel:
        p1 = rows1[int(rec_idx[rec[k]])]
        c = int(cand[k])
        q = rows2[c] if c < n2 else rows3[c - n2]
        A.append(p1[0] + ' | ' + p1[1][:200])
        B.append(q[0] + ' | ' + q[1][:200])

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'ce_state.pt'),
                                     map_location='cpu'))
    model.to(device).eval()
    ce = np.zeros(len(sel), np.float32)
    t1 = time.time()
    with torch.inference_mode():
        for i in range(0, len(sel), args.batch):
            enc = tok(A[i:i + args.batch], B[i:i + args.batch],
                      padding='max_length', truncation=True,
                      max_length=args.max_len, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            ce[i:i + args.batch] = torch.sigmoid(
                model(**enc).logits.squeeze(-1)).float().cpu().numpy()
            if i and i % (args.batch * 20) == 0:
                log(f"  ce {i:,}/{len(sel):,} "
                    f"({i/max(time.time()-t1, 1e-9):,.0f} pairs/s)")
    np.savez(args.out, sel=sel, ce=ce)
    log(f"CE scored {len(sel):,} pairs in {time.time()-t1:.0f}s -> {args.out}")
    ce_p = np.full(len(prob), -1.0, np.float32)
    ce_p[sel] = ce

    leaked = np.union1d(pf.spread_sample(N_TRAIN, 500_000, seed=0),
                        pf.spread_sample(N_TRAIN, 100_000, seed=7))
    keep = ~np.isin(rec_idx, leaked)
    log(f"union-clean subset: {int(keep.sum()):,}/{n:,} records")

    base_hit = prob >= BASE_THR
    base_c, base_st = pp.score(d, base_hit, keep)
    base_h, _ = pp.score(d, base_hit)
    log(f"baseline prob>={BASE_THR}: clean={base_c:.4f} headline={base_h:.4f} "
        f"P={base_st['precision']:.4f} R={base_st['recall']:.4f}")
    for t in (0.90, 0.93, 0.95, 0.97):
        mc, st = pp.score(d, prob >= t, keep)
        log(f"  ref thr {t:.2f}: clean={mc:.4f} "
            f"P={st['precision']:.4f} R={st['recall']:.4f}")

    def rescue_hit(r_lo, c):
        return base_hit | ((prob >= r_lo) & (ce_p >= c))

    def clean_hit(z, c):
        return base_hit & ~((prob < z) & (ce_p >= 0) & (ce_p < c))

    res = []
    for r_lo in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90):
        for c in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98):
            hit = rescue_hit(r_lo, c)
            mc, st = pp.score(d, hit, keep)
            m, _ = pp.score(d, hit)
            res.append(dict(fam='rescue', delta=mc - base_c, mc=mc, head=m,
                            r_lo=r_lo, c=c, n=int(hit.sum() - base_hit.sum()),
                            p=st['precision'], r=st['recall']))
    for z in (0.95, 0.96, 0.97, 0.98, 0.99):
        for c in (0.05, 0.10, 0.20, 0.30, 0.50):
            hit = clean_hit(z, c)
            mc, st = pp.score(d, hit, keep)
            m, _ = pp.score(d, hit)
            res.append(dict(fam='clean', delta=mc - base_c, mc=mc, head=m,
                            z=z, c=c, n=int(hit.sum() - base_hit.sum()),
                            p=st['precision'], r=st['recall']))
    res.sort(key=lambda x: -x['delta'])
    log("top rules (dClean | head | fam | r_lo/z | c | pairs | P | R):")
    for x in res[:15]:
        tag = f"r_lo={x.get('r_lo')}" if x['fam'] == 'rescue' else f"z={x.get('z')}"
        log(f"  {x['delta']:+.4f} | {x['head']:.4f} | {x['fam']:6s} {tag} "
            f"c={x['c']} | {x['n']:+,} | P={x['p']:.4f} R={x['r']:.4f}")

    best_r = [x for x in res if x['fam'] == 'rescue'][:3]
    best_c = [x for x in res if x['fam'] == 'clean'][:3]
    combos = []
    for a in best_r:
        for b in best_c:
            hit = rescue_hit(a['r_lo'], a['c']) & \
                ~((prob < b['z']) & (ce_p >= 0) & (ce_p < b['c']))
            mc, st = pp.score(d, hit, keep)
            m, _ = pp.score(d, hit)
            combos.append(dict(fam='combo', delta=mc - base_c, mc=mc, head=m,
                               r_lo=a['r_lo'], c_res=a['c'], z=b['z'],
                               c_cl=b['c'], p=st['precision'], r=st['recall']))
    combos.sort(key=lambda x: -x['delta'])
    for x in combos[:5]:
        log(f"  combo {x['delta']:+.4f} | {x['head']:.4f} | rescue r_lo={x['r_lo']} "
            f"c={x['c_res']} + clean z={x['z']} c={x['c_cl']} | "
            f"P={x['p']:.4f} R={x['r']:.4f}")

    best = max(combos + res, key=lambda x: x['delta'])
    verdict = 'GO' if best['delta'] >= BAR else 'NO-GO'
    log(f"VERDICT {verdict}: best clean delta {best['delta']:+.4f} "
        f"(bar +{BAR:.3f})")

    per_c = {}
    if best['fam'] == 'combo':
        hit = rescue_hit(best['r_lo'], best['c_res']) & \
            ~((prob < best['z']) & (ce_p >= 0) & (ce_p < best['c_cl']))
    elif best['fam'] == 'rescue':
        hit = rescue_hit(best['r_lo'], best['c'])
    else:
        hit = clean_hit(best['z'], best['c'])
    for code, name in ((0, 'US'), (1, 'India')):
        km = keep & (d['ccodes'] == code)
        mc, st = pp.score(d, hit, km)
        mc0, _ = pp.score(d, base_hit, km)
        per_c[name] = dict(delta=mc - mc0, mc=mc, base=mc0)
        log(f"  per-country {name}: base {mc0:.4f} -> {mc:.4f} "
            f"({mc - mc0:+.4f})")

    with open(args.report, 'w') as f:
        json.dump(dict(dump=args.dump, baseline_clean=base_c,
                       baseline_head=base_h, baseline_stats=base_st,
                       scanned=len(sel), verdict=verdict, best=best,
                       top=res[:20], combos=combos[:10], per_country=per_c),
                  f, indent=1)
    log(f"wrote {args.report}")


if __name__ == '__main__':
    main()
