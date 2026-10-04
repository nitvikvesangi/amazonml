#!/usr/bin/env python3
"""Semantic rescue probe: can multilingual embeddings separate the band the model cannot?

The measured dead zone: pairs with model prob in [0.5, 0.94) hold 8,629 true and
4,511 false pairs, and their *lexical* features are identical (token-set 0.961 vs
0.958) - that is why every threshold/rule/feature tweak failed there.  This tool
asks a different question: does a pretrained multilingual sentence embedding see
a difference?  If yes, a rescue rule

    accept pair  <=>  (prob >= 0.94)  or  (prob >= t_lo  and  cos(name1, name23) >= c)

can add true links without the false flood, lifting macro F0.5 on the holdout -
and it applies directly to the saved test scores.tsv, no retrain, no re-predict.

Strings are read from the raw TSVs (row order == the split's id space) and only
the ids the pairs need, so the run stays memory-light next to other jobs.

Run from code/business_entity_resolution:
    python3 tools/sem_probe.py --dump ../../output/eval41a30_fixed/pairs_base41_250k.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
import postproc as pp            # noqa: E402
import pipeline_fast as pf       # noqa: E402

MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def auc(y, s):
    """Rank AUC with tie handling (no sklearn dependency)."""
    y = np.asarray(y, bool)
    s = np.asarray(s, float)
    order = np.argsort(s, kind='stable')
    r = np.empty(len(s), float)
    r[order] = np.arange(1, len(s) + 1)
    # average ranks over ties
    s_s = s[order]
    i = 0
    while i < len(s_s):
        j = i
        while j + 1 < len(s_s) and s_s[j + 1] == s_s[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2
        i = j + 1
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def read_rows(path, want_pos, n_cols=3, chunk=400_000):
    """Rows at the given 0-based positions -> {pos: (name, addr)}.

    Streaming in chunks, so only the needed rows are materialised (the test
    source2/source3 TSVs are ~10 M rows).
    """
    out = {}
    pos0 = 0
    for ch in pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False,
                          na_filter=False, chunksize=chunk):
        n = len(ch)
        if want_pos:
            rel = [p - pos0 for p in want_pos if pos0 <= p < pos0 + n]
            # tuples (position, row offset) to fetch
            for p in rel:
                row = ch.iloc[p]
                out[pos0 + p] = (row['business_name'], row.get('business_address', ''))
        pos0 += n
        if want_pos and pos0 > max(want_pos):
            break
    return out, pos0


def read_pos_ids(path, want_id_col, chunk=500_000):
    """entity_id per row position (to map ids -> positions cheaply if needed)."""
    ids = []
    for ch in pd.read_csv(path, sep='\t', usecols=[want_id_col], dtype=str,
                          keep_default_na=False, na_filter=False, chunksize=chunk):
        ids.extend(ch[want_id_col].tolist())
    return ids


def embed(texts, batch=512):
    from sentence_transformers import SentenceTransformer
    import torch
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    m = SentenceTransformer(MODEL, device=dev)
    t0 = time.time()
    E = m.encode(texts, batch_size=batch, show_progress_bar=False,
                 normalize_embeddings=True, convert_to_numpy=True)
    dt = time.time() - t0
    log(f"embedded {len(texts):,} strings in {dt:.1f}s "
        f"({len(texts)/max(dt,1e-9):,.0f} strings/s, {dev})")
    return E.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', default='../../output/eval41a30_fixed/pairs_base41_250k.npz')
    ap.add_argument('--dataset', default='../../dataset/train')
    ap.add_argument('--lo', type=float, default=0.50, help='lower edge of the band')
    ap.add_argument('--hi', type=float, default=0.94, help='model threshold (upper edge)')
    ap.add_argument('--acc-sample', type=int, default=30000,
                    help='accepted pairs sampled for reference distributions')
    ap.add_argument('--out', default='../../cache/sem_probe.json')
    args = ap.parse_args()

    d = pp.load_dump(args.dump)
    rec, cand, prob = d['rec'], d['cand'], d['prob']
    rec_idx, tp = d['rec_idx'], (d['tp_flag'] == 1)
    n = d['n']
    log(f"dump {args.dump}: {len(prob):,} pairs, {n:,} records, "
        f"{(prob >= args.hi).sum():,} accepted, threshold={d['threshold']}")

    band = (prob >= args.lo) & (prob < args.hi)
    acc = prob >= args.hi
    rng = np.random.default_rng(7)
    acc_sel = np.flatnonzero(acc)
    if len(acc_sel) > args.acc_sample:
        acc_sel = acc_sel[rng.choice(len(acc_sel), args.acc_sample, replace=False)]
    take = np.flatnonzero(band)
    take = np.union1d(take, acc_sel).astype(np.int64)
    log(f"band [{args.lo},{args.hi}): {int(band.sum()):,} pairs "
        f"({int(tp[band].sum()):,} true) | reference accepted sample: {len(acc_sel):,}")

    # positions -> strings.  split order: source1 positions == rec_idx space;
    # source2 positions == cand space below n2; source3 positions == cand - n2.
    need1 = set(rec_idx[rec[take]].tolist())
    n2path = os.path.join(args.dataset, 'train_source2.tsv')
    n2 = sum(1 for _ in open(n2path, 'rb')) - 1      # fast row count
    is3 = cand[take] >= n2
    need2 = set(cand[take][~is3].tolist())
    need3 = set((cand[take][is3] - n2).tolist())
    log(f"strings needed: source1={len(need1):,} source2={len(need2):,} "
        f"source3={len(need3):,} (n2={n2:,})")

    t0 = time.time()
    rows1, _ = read_rows(os.path.join(args.dataset, 'train_source1.tsv'), need1)
    rows2, _ = read_rows(n2path, need2)
    rows3, _ = read_rows(os.path.join(args.dataset, 'train_source3.tsv'), need3)
    log(f"read {len(rows1)+len(rows2)+len(rows3):,} rows in {time.time()-t0:.0f}s")

    def name_of(r_local, c):
        return rows1[rec_idx[rec[r_local]]][0], (rows2[c][0] if c < n2 else rows3[c - n2][0])

    def addr_of(r_local, c):
        return rows1[rec_idx[rec[r_local]]][1], (rows2[c][1] if c < n2 else rows3[c - n2][1])

    # build the unique string list once for both sides
    n1s, n23s, a1s, a23s = [], [], [], []
    for k in take:
        a, b = name_of(k, cand[k])
        c, e = addr_of(k, cand[k])
        n1s.append(a); n23s.append(b); a1s.append(c[:200]); a23s.append(e[:200])
    uniq = {}
    for lst in (n1s, n23s, a1s, a23s):
        for s in lst:
            uniq.setdefault(s, None)
    keys = list(uniq)
    E = embed(keys)
    idx = {s: i for i, s in enumerate(keys)}
    cos_n = np.array([float(E[idx[a]] @ E[idx[b]]) for a, b in zip(n1s, n23s)])
    cos_a = np.array([float(E[idx[a]] @ E[idx[b]]) for a, b in zip(a1s, a23s)])
    log(f"cos name: true-median={np.median(cos_n[tp[take]]):.3f} "
        f"false-median={np.median(cos_n[~tp[take]]):.3f} | "
        f"addr: {np.median(cos_a[tp[take]]):.3f} vs {np.median(cos_a[~tp[take]]):.3f}")

    band_rows = np.isin(take, np.flatnonzero(band))
    for tag, sel in (('band', band_rows), ('accepted', ~band_rows)):
        y = tp[take[sel]]
        log(f"  {tag:<9} n={len(y):,} true={int(y.sum()):,}  "
            f"AUC cos_name={auc(y, cos_n[sel]):.4f}  cos_addr={auc(y, cos_a[sel]):.4f}  "
            f"AUC max(name,addr)={auc(y, np.maximum(cos_n, cos_a)[sel]):.4f}")

    # rescue rule sweep on the exact metric, headline and own-leak-masked
    leaked = pf.spread_sample(2_206_821, 250_000, seed=0)
    clean = ~np.isin(rec_idx, leaked)
    base_h, _ = pp.score(d, prob >= args.hi, None)
    base_c, _ = pp.score(d, prob >= args.hi, clean)
    log(f"baseline (prob>={args.hi}): headline={base_h:.4f} clean={base_c:.4f}")

    cos_p = np.zeros(len(prob), np.float32)
    cos_p[take] = np.maximum(cos_n, cos_a)
    best = []
    for t_lo in (0.05, 0.20, 0.30, 0.50):
        for c in np.round(np.arange(0.50, 0.991, 0.02), 2):
            hit = (prob >= args.hi) | ((prob >= t_lo) & (cos_p >= c))
            m, st = pp.score(d, hit, None)
            mc, _ = pp.score(d, hit, clean)
            added = int(hit.sum() - (prob >= args.hi).sum())
            best.append((mc, m, float(t_lo), float(c), added,
                         float(st['precision']), float(st['recall'])))
    best.sort(reverse=True)
    log("top rescue rules (clean delta | headline | t_lo | cos>=c | added pairs | P | R):")
    for mc, m, t_lo, c, added, p_, r_ in best[:8]:
        log(f"  clean {mc:.4f} ({mc - base_c:+.4f}) | headline {m:.4f} ({m - base_h:+.4f}) "
            f"| t_lo={t_lo:.2f} cos>={c:.2f} | +{added:,} pairs | P={p_:.4f} R={r_:.4f}")

    with open(args.out, 'w') as f:
        json.dump(dict(dump=args.dump, lo=args.lo, hi=args.hi,
                       baseline_headline=base_h, baseline_clean=base_c,
                       auc_band_name=auc(tp[take[band_rows]], cos_n[band_rows]),
                       auc_band_addr=auc(tp[take[band_rows]], cos_a[band_rows]),
                       best=best[:20]), f, indent=1)
    log(f"wrote {args.out}")


if __name__ == '__main__':
    main()
