#!/usr/bin/env python3
"""Offline last-mile policy tuning on a dumped val pair set.

Every model change costs a full predict (hours) to reach the leaderboard, so the
decision layer is tuned separately, on a *dumped* probability matrix (produced by
`pipeline_fast.py eval --dump-pairs`).  Everything here is exact: the same
per-record F0.5 + singleton rule as the competition metric.

What it answers
---------------
  * plain       : baseline macro F0.5 at the model's own threshold, the best
                  threshold (sanity check against the eval report - must agree),
                  and the best top-K-per-record rule
  * conflict    : ground truth is a strict partition (each S2/S3 record is in at
                  most one cluster), so keep only the best-probability claim per
                  candidate - drops false merges no model can justify
  * per-country : optimal threshold *per country* - the upper bound of what a
                  country-aware threshold could buy
  * rate-match  : transfer one country's optimal threshold to another by matching
                  the share of records with >=1 accepted claim.  That is the only
                  label-free way to calibrate France (no French ground truth), and
                  it is validated here on India, where labels do exist.

Run from code/business_entity_resolution:
    python3 tools/postproc.py --dump ../../output/eval41a30_fixed/pairs.npz --conflict
    python3 tools/postproc.py --dump ... --per-country --rate-match India:US
"""
import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import fast_data as fd          # noqa: E402
import fast_score as fs         # noqa: E402
import pipeline_fast as pf      # noqa: E402

CACHE = os.environ.get('ER_CACHE', '../../cache')


def log(msg):
    print(msg, flush=True)


def load_dump(path, cache=CACHE, gt_split='train'):
    z = np.load(path)
    rec = z['rec'].astype(np.int64)
    cand = z['cand'].astype(np.int64)
    prob = z['prob'].astype(np.float32)
    rec_idx = z['rec_idx'].astype(np.int64)
    with open(os.path.join(cache, f'gt_{gt_split}.pkl'), 'rb') as f:
        gt = pickle.load(f)
    gt_local = pf.subset_csr(gt['offsets'], gt['flat'], rec_idx)
    return dict(rec=rec, cand=cand, prob=prob, rec_idx=rec_idx, n=len(rec_idx),
                ccodes=(z['ccodes'].astype(np.int8) if 'ccodes' in z else None),
                threshold=(float(z['threshold']) if 'threshold' in z else None),
                gt=gt_local, tp_flag=fs.make_labels(rec, cand, gt_local, 1 << 30))


def score(d, hit, keep=None):
    """Macro F0.5 (+pair stats) for a hit mask, optionally over a record subset."""
    nt = np.diff(d['gt']['offsets']).astype(np.float64)
    r = d['rec'][hit]
    pred = np.bincount(r, minlength=d['n']).astype(np.float64)
    tp = np.bincount(r[d['tp_flag'][hit] == 1], minlength=d['n']).astype(np.float64)
    if keep is not None:
        pred, nt, tp = pred[keep], nt[keep], tp[keep]
    macro, br = pf.macro_f05(pred, nt, tp)
    return macro, dict(precision=float(tp.sum() / max(pred.sum(), 1)),
                       recall=float(tp.sum() / max(nt.sum(), 1)),
                       predicted=int(pred.sum()), true=int(nt.sum()),
                       tp=int(tp.sum()), **br)


def record_f(d, hit):
    """Per-record F0.5 array for a hit mask (same buckets as macro_f05)."""
    nt = np.diff(d['gt']['offsets']).astype(np.float64)
    r = d['rec'][hit]
    pred = np.bincount(r, minlength=d['n']).astype(np.float64)
    tp = np.bincount(r[d['tp_flag'][hit] == 1], minlength=d['n']).astype(np.float64)
    both_empty = (nt == 0) & (pred == 0)
    false_merge = (nt == 0) & (pred > 0)
    empty_pred = (nt > 0) & (pred == 0)
    precision = np.where(pred > 0, tp / np.maximum(pred, 1), 0.0)
    recall = np.where(nt > 0, tp / np.maximum(nt, 1), 0.0)
    denom = 0.25 * precision + recall
    f = np.where(denom > 0, 1.25 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    f = np.where(both_empty, 1.0, np.where(false_merge, 0.0,
                 np.where(empty_pred, 0.0, f)))
    return f, nt, pred


def record_max(d):
    """Per-record best pair probability (the only label-free confidence signal)."""
    m = np.zeros(d['n'], dtype=np.float32)
    np.maximum.at(m, d['rec'], d['prob'])
    return m


def policy_sweep(d, base, hi_grid, lo_grid, mid_grid):
    """Record-conditional accept rule, swept on the dump.

    A record whose best candidate is very confident (max >= h_hi) may afford to
    accept weaker candidates (>= t_lo); a record with no confident candidate
    (max < h_mid) is more likely unclustered noise, so predicting *nothing* is
    often worth more than a lone 0.95 claim.  Both directions are pure
    post-processing on the probabilities, so they cost no predict.
    """
    rmax = record_max(d)
    pm = rmax[d['rec']]
    best = []
    for h_hi in hi_grid:
        for h_mid in mid_grid:
            for t_lo in lo_grid:
                hit = np.where(pm >= h_hi, d['prob'] >= t_lo,
                               np.where(pm >= h_mid, d['prob'] >= base, False))
                macro, stats = score(d, hit)
                best.append((macro, h_hi, h_mid, t_lo, stats))
    best.sort(key=lambda x: -x[0])
    return best


def sweep(d, thresholds, keep=None, make_hit=None):
    """Best (threshold, macro, stats) over a grid; make_hit may post-process."""
    best = (None, -1.0, {})
    for t in thresholds:
        hit = (d['prob'] >= t) if make_hit is None else make_hit(t)
        macro, stats = score(d, hit, keep)
        if macro > best[1]:
            best = (float(t), macro, stats)
    return best


def conflict_clean(d, hit):
    """Keep at most one claim per candidate: the highest-probability one."""
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return hit
    order = np.lexsort((-d['prob'][idx], d['cand'][idx]))
    cs = d['cand'][idx][order]
    first = np.ones(len(cs), dtype=bool)
    first[1:] = cs[1:] != cs[:-1]
    out = np.zeros(len(hit), dtype=bool)
    out[idx[order[first]]] = True
    return out


def topk_mask(d, hit, k):
    """At most k accepted candidates per record (best probability first)."""
    idx = np.flatnonzero(hit)
    if len(idx) == 0:
        return hit
    order = np.lexsort((-d['prob'][idx], d['rec'][idx]))
    rs = d['rec'][idx][order]
    _, first, cnts = np.unique(rs, return_index=True, return_counts=True)
    rank = np.arange(len(rs), dtype=np.int64) - np.repeat(first, cnts)
    out = np.zeros(len(hit), dtype=bool)
    out[idx[order[rank < k]]] = True
    return out


def sweep_mix(d, thresholds, shares):
    """Best threshold for a target country *mix* (weighted macro F0.5).

    The shipping threshold is tuned on the val country mix but applied to the
    test mix, which is a different blend (test is much more India-heavy), and
    the metric is a macro average - so the per-country numbers have to be
    weighted before comparing thresholds.
    """
    total = sum(shares.values())
    keeps = {c: (d['ccodes'] == c) for c in shares}
    best = (None, -1.0, {})
    for t in thresholds:
        hit = d['prob'] >= t
        mixed = 0.0
        per = {}
        for code, w in shares.items():
            macro, _ = score(d, hit, keeps[code])
            per[code] = macro
            mixed += (w / total) * macro
        if mixed > best[1]:
            best = (float(t), mixed, per)
    return best


def rescue_mask(d, t, t2):
    """Accept pairs above t, plus each record's *best* candidate down to t2.

    F0.5 gives a record 0.0 when a true cluster is missed entirely and 1.0 when
    a true singleton is correctly left empty, so rescuing the top candidate of an
    otherwise-empty record trades those two buckets against each other - worth
    measuring, since `missed_all` is 2.5 % of records and `both_empty` 4.7 %.
    """
    hit = d['prob'] >= t
    idx = np.flatnonzero(~hit)
    if len(idx) == 0:
        return hit
    r, p = d['rec'][idx], d['prob'][idx]
    order = np.lexsort((-p, r))
    rs = r[order]
    first = np.ones(len(rs), dtype=bool)
    first[1:] = rs[1:] != rs[:-1]
    cand_idx = idx[order[first]]
    out = hit.copy()
    out[cand_idx[p[order[first]] >= t2]] = True
    return out


def share_with_claim(d, hit, keep=None):
    """Share of (subset) records with >=1 accepted candidate: the label-free knob."""
    rec_hit = d['rec'][hit]
    if keep is not None:
        rec_hit = rec_hit[keep[rec_hit]]
        denom = int(keep.sum())
    else:
        denom = d['n']
    return float(np.unique(rec_hit).size / max(denom, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True)
    ap.add_argument('--ensemble', default=None,
                    help='second dump over the SAME records: measures the gain '
                         'from averaging two models before paying for a predict')
    ap.add_argument('--split', default='train')
    ap.add_argument('--thresholds', default='0.30,0.90,0.005',
                    help='lo,hi,step grid for the threshold sweep')
    ap.add_argument('--oracle', action='store_true',
                    help='ceiling of the candidate set + loss decomposition')
    ap.add_argument('--policy', action='store_true',
                    help='sweep record-conditional accept rules (max-prob gated)')
    ap.add_argument('--conflict', action='store_true')
    ap.add_argument('--topk', default=None, help='comma list of K values')
    ap.add_argument('--per-country', action='store_true')
    ap.add_argument('--rate-match', default=None, help='FROM:TO, e.g. India:US')
    ap.add_argument('--rescue', action='store_true',
                    help='2-D sweep: threshold t plus a lower top-1 rescue threshold')
    ap.add_argument('--per-source', action='store_true',
                    help='2-D sweep: separate thresholds for S2 and S3 candidates')
    ap.add_argument('--n2', type=int, default=5034616,
                    help='source2 size in the split (S2 ids < n2 < S3 ids)')
    ap.add_argument('--mix', default=None,
                    help='country mix to weight the macro by, e.g. US:0.383,India:0.467')
    args = ap.parse_args()

    lo, hi, step = [float(x) for x in args.thresholds.split(',')]
    thresholds = np.round(np.arange(lo, hi + 1e-9, step), 5)
    d = load_dump(args.dump, gt_split=args.split)
    log(f"dump {args.dump}: {len(d['prob']):,} pairs, {d['n']:,} records, "
        f"{int((d['prob'] >= 0.5).sum()):,} pairs >= 0.5")
    if args.ensemble:
        e = load_dump(args.ensemble, gt_split=args.split)
        same = (len(e['prob']) == len(d['prob']) and np.array_equal(e['rec'], d['rec'])
                and np.array_equal(e['cand'], d['cand']))
        if not same:
            log("  ensemble: WARNING pair sets differ, restricting to the shared pairs")
            key_d = d['rec'].astype(np.int64) * (1 << 31) + d['cand']
            key_e = e['rec'].astype(np.int64) * (1 << 31) + e['cand']
            common, ia, ib = np.intersect1d(key_d, key_e, return_indices=True)
            order = np.argsort(ia)
            ia, ib = ia[order], ib[order]
            for k in ('rec', 'cand', 'prob'):
                d[k] = d[k][ia]
                e[k] = e[k][ib]
            d['n'] = len(np.unique(d['rec']))
            d['tp_flag'] = d['tp_flag'][ia]
        m1 = sweep(d, thresholds)[1]
        d['prob'] = ((d['prob'].astype(np.float64)
                      + e['prob'].astype(np.float64)) / 2).astype(np.float32)
        et, em, es = sweep(d, thresholds)
        log(f"  ensemble of 2 models: {et:.3f} -> {em:.4f}  ({em - m1:+.4f} vs "
            f"best single {m1:.4f})  P={es['precision']:.4f} R={es['recall']:.4f}")
        return

    if d['threshold'] is not None:
        macro, stats = score(d, d['prob'] >= d['threshold'])
        log(f"  model threshold {d['threshold']:.3f}: macro F0.5={macro:.4f}  "
            f"P={stats['precision']:.4f} R={stats['recall']:.4f}")

    bt, bm, bs = sweep(d, thresholds)
    log(f"  best plain thr       : {bt:.3f} -> {bm:.4f}  "
        f"P={bs['precision']:.4f} R={bs['recall']:.4f} predicted={bs['predicted']:,} "
        f"partial={bs['partial']:,} missed={bs['missed_all']:,} "
        f"false_merge={bs['false_merge']:,}")
    best_plain = bm

    if args.oracle:
        # What could a *perfect* scorer have done with the candidate set the
        # blocker+cap actually produced?  This is the pipeline's true ceiling:
        # every extra point the model needs must come from inside this gap.
        nt = np.diff(d['gt']['offsets']).astype(np.int64)
        present = np.bincount(d['rec'][d['tp_flag'] == 1], minlength=d['n'])
        has_true = nt > 0
        full = has_true & (present >= nt)
        part = has_true & (present > 0) & (present < nt)
        zero = has_true & (present == 0)
        om, ost = score(d, d['tp_flag'] == 1)
        log(f"  candidate set: {int((d['tp_flag'] == 1).sum()):,}/{int(nt.sum()):,} true pairs "
            f"present ({100.0 * int((d['tp_flag'] == 1).sum()) / max(int(nt.sum()), 1):.3f}%) | "
            f"records: all-present={int(full.sum()):,} partial={int(part.sum()):,} "
            f"none={int(zero.sum()):,} true-empty={int((~has_true).sum()):,}")
        log(f"  ORACLE macro (perfect scorer, same candidates): {om:.4f}  "
            f"P={ost['precision']:.4f} R={ost['recall']:.4f}")
        for label, hit in (('model', d['prob'] >= d['threshold']),
                           ('oracle', d['tp_flag'] == 1)):
            f, nt_, pred_ = record_f(d, hit)
            loss = 1.0 - f
            both = (nt_ == 0) & (pred_ == 0)
            fm = (nt_ == 0) & (pred_ > 0)
            miss = (nt_ > 0) & (pred_ == 0)
            pt = ~(both | fm | miss)
            log(f"    {label:<6} macro={f.mean():.4f}  loss/N from false_merge="
                f"{loss[fm].sum() / d['n']:.4f} missed_all={loss[miss].sum() / d['n']:.4f} "
                f"partial={loss[pt].sum() / d['n']:.4f}  "
                f"(fm={int(fm.sum()):,} miss={int(miss.sum()):,} part={int(pt.sum()):,})")
            if label == 'model' and miss.any():
                q = np.percentile(nt_[miss], [50, 90, 99])
                log(f"      missed records, |true cluster| median={q[0]:.0f} "
                    f"p90={q[1]:.0f} p99={q[2]:.0f} max={int(nt_[miss].max())}")
        f, _, _ = record_f(d, d['prob'] >= d['threshold'])
        log(f"  headroom to oracle: {om - f.mean():+.4f} (oracle is "
            f"{100.0 * om:.2f}% vs model {100.0 * f.mean():.2f}%)")

    if args.policy:
        def grid(spec):
            return np.round(np.arange(*[float(x) for x in spec.split(',')]), 5)
        hi_grid = np.r_[grid('0.50,0.99,0.06'), 1.01]
        lo_grid = grid('0.05,0.95,0.05')
        mid_grid = grid('0.05,0.96,0.05')
        res = policy_sweep(d, bt, hi_grid, lo_grid, mid_grid)
        log(f"  conditional policy sweep ({len(res)} rules, base={bt:.3f}):")
        for macro, h_hi, h_mid, t_lo, st in res[:6]:
            log(f"    max>={h_hi:.2f} -> accept>={t_lo:.2f} | max>={h_mid:.2f} -> "
                f"base | else empty : {macro:.4f} ({macro - best_plain:+.4f})  "
                f"P={st['precision']:.4f} R={st['recall']:.4f} miss={st['missed_all']:,} "
                f"fm={st['false_merge']:,}")

    if args.conflict:
        ct, cm, cs = sweep(d, thresholds,
                           make_hit=lambda t: conflict_clean(d, d['prob'] >= t))
        log(f"  + conflict cleanup   : {ct:.3f} -> {cm:.4f}  ({cm - best_plain:+.4f})  "
            f"P={cs['precision']:.4f} R={cs['recall']:.4f} "
            f"predicted={cs['predicted']:,} false_merge={cs['false_merge']:,}")

    if args.topk:
        for k in [int(x) for x in args.topk.split(',')]:
            kt, km, _ = sweep(d, thresholds,
                              make_hit=lambda t, k=k: topk_mask(d, d['prob'] >= t, k))
            log(f"  top-{k} per record     : {kt:.3f} -> {km:.4f}  ({km - best_plain:+.4f})")

    if args.per_country and d['ccodes'] is not None:
        log("  per-country (optimal threshold per country = calibration upper bound):")
        for name, code in sorted(fd.COUNTRY_MAP.items(), key=lambda kv: kv[1]):
            keep = d['ccodes'] == code
            if not keep.any():
                continue
            nt, nm, _ = sweep(d, thresholds, keep=keep)
            at_global = score(d, d['prob'] >= bt, keep)[0]
            log(f"    {name:<7} n={int(keep.sum()):,} best thr={nt:.3f} macro={nm:.4f} "
                f"({nm - at_global:+.4f} vs global thr {bt:.3f} = {at_global:.4f})")

    if args.rescue:
        grid = np.round(np.arange(lo, hi + 1e-9, 0.02), 4)
        best = (None, None, -1.0, {})
        for t in grid:
            for t2 in grid[grid < t]:
                macro, stats = score(d, rescue_mask(d, t, t2))
                if macro > best[2]:
                    best = (float(t), float(t2), macro, stats)
        log(f"  top-1 rescue         : t={best[0]:.2f} rescue>={best[1]:.2f} -> "
            f"{best[2]:.4f}  ({best[2] - best_plain:+.4f} vs plain thr {bt:.3f})  "
            f"missed={best[3].get('missed_all', 0):,} "
            f"both_empty={best[3].get('both_empty', 0):,}")

    if args.per_source:
        is3 = d['cand'] >= args.n2
        grid = thresholds[::4]
        best = (None, None, -1.0, {})
        for t2 in grid:
            for t3 in grid:
                hit = np.where(is3, d['prob'] >= t3, d['prob'] >= t2)
                macro, stats = score(d, hit)
                if macro > best[2]:
                    best = (float(t2), float(t3), macro, stats)
        log(f"  per-source thresholds : S2={best[0]:.3f} S3={best[1]:.3f} -> "
            f"{best[2]:.4f}  ({best[2] - best_plain:+.4f} vs single thr {bt:.3f})  "
            f"predicted={best[3].get('predicted', 0):,}")

    if args.mix and d['ccodes'] is not None:
        shares = {}
        for part in args.mix.split(','):
            name, w = part.split(':')
            shares[fd.COUNTRY_MAP[name]] = float(w)
        mt, mm, mper = sweep_mix(d, thresholds, shares)
        pretty = ", ".join(f"{k}={v:.4f}" for k, v in mper.items())
        log(f"  mix-weighted threshold: {mt:.3f} -> {mm:.4f}  ({pretty})  "
            f"vs global-best thr {bt:.3f} weighted "
            f"{sweep_mix(d, [bt], shares)[1]:.4f}")

    if args.rate_match and d['ccodes'] is not None:
        from_name, to_name = args.rate_match.split(':')
        c_from = fd.COUNTRY_MAP[from_name]
        c_to = fd.COUNTRY_MAP[to_name]
        keep_from, keep_to = d['ccodes'] == c_from, d['ccodes'] == c_to
        ft, fm, _ = sweep(d, thresholds, keep=keep_from)
        target = share_with_claim(d, d['prob'] >= ft, keep_from)
        best_t, best_gap = ft, np.inf
        for t in thresholds:
            gap = abs(share_with_claim(d, d['prob'] >= t, keep_to) - target)
            if gap < best_gap:
                best_t, best_gap = float(t), gap
        transfer = score(d, d['prob'] >= best_t, keep_to)[0]
        own_t, own_macro, _ = sweep(d, thresholds, keep=keep_to)
        log(f"  rate-match {from_name}->{to_name}: {from_name} thr={ft:.3f} "
            f"macro={fm:.4f} share={target:.3f} -> {to_name} thr={best_t:.3f} "
            f"macro={transfer:.4f} | own best thr={own_t:.3f} macro={own_macro:.4f} "
            f"| gap {transfer - own_macro:+.4f}")


if __name__ == '__main__':
    main()
