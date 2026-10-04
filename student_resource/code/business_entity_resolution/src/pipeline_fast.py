#!/usr/bin/env python3
"""Fast end-to-end entity resolution pipeline (vectorized).

Modes
-----
  train     block + score a sample of train records, fit LightGBM, tune the F0.5
            threshold on a record-level holdout, save model + threshold
  eval      score a (fresh) sample of train records with a saved model and report
            macro F0.5 / precision / recall — the honest generalisation check
  predict   block + score every test record, apply the threshold, and stream out
            matching_results.tsv + candidate_pairs.tsv

The whole path runs in record chunks, so peak memory stays ~1-2 GB: block a
chunk, vectorize its pair features (rapidfuzz cdist on all cores), predict, then
either accumulate training rows or stream output rows.  There is no per-pair
Python loop anywhere.

Key-selection policy (measured on train, macro per-record blocking recall vs
candidates kept per record):
    N=100 0.838 | N=150 0.849 | N=200 0.856 | N=400 0.875 | N=600 0.885
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

import fast_block as fb
import fast_data as fd
import fast_score as fs

CACHE = fd.CACHE

POOL_CFG = {
    'ntok': {'budget': 1000, 'max_df': 30000},
    'npre': {'budget': 400, 'max_df': 5000},
    'anum': {'budget': 200, 'max_df': 5000},
    'aword': {'budget': 500, 'max_df': 5000},
}
CAND_CAP = 200

# 0 = rank the per-record cap purely by summed rarity weight.  >0 blends in a
# cheap similarity (max of name/address fuzz.ratio) so true matches that rarity
# ranking pushes below the cap survive.  Measure before changing (tools/measure_cap.py).
SIM_CAP_ALPHA = 0.0

# Trained unweighted on purpose: the F0.5 threshold is swept afterwards, and a
# large scale_pos_weight makes val logloss peak at iteration 1 (measured), which
# kills early stopping.  Threshold tuning handles the calibration instead.
LGB_PARAMS = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 50,
    'verbose': -1,
    'n_jobs': -1,
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def load_keys(split, cache_dir=CACHE):
    path = os.path.join(cache_dir, f'keys_{split}.pkl')
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: build it with tools/tune_blocking.py")
    return load_pkl(path)


def subset_csr(off, flat, rec_idx):
    """CSR restricted to rec_idx, re-indexed to local positions 0..len-1."""
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
        new_flat = np.zeros(0, dtype=flat.dtype)
    return {'offsets': new_off, 'flat': new_flat}


def spread_sample(n_total, n_want, seed=0):
    """Deterministic spread sample (avoids the order/country bias of head())."""
    rng = np.random.default_rng(seed)
    n_want = min(n_want, n_total)
    return np.sort(rng.choice(n_total, size=n_want, replace=False)).astype(np.int64)


def cap_score(res, n, px, cand_cap, alpha):
    """Ranking key for the per-record cap: rarity weight + alpha * cheap similarity.

    Rapidfuzz on the *uncapped* unique pairs is affordable (≈2 metric calls per
    unique pair ≈ 4 min over the whole test split) and keeps true matches that
    plain rarity ranking pushes below the cap.
    """
    if alpha <= 0:
        return None
    from rapidfuzz import fuzz, process
    cnt = np.diff(res['off'])
    rec = np.repeat(np.arange(n, dtype=np.int32), cnt)
    cand = res['cand']
    nr = process.cpdist(px['names1'][rec], px['names23'][cand],
                        scorer=fuzz.ratio, workers=-1) / 100.0
    ar = process.cpdist(px['addrs1'][rec], px['addrs23'][cand],
                        scorer=fuzz.ratio, workers=-1) / 100.0
    sim = np.maximum(nr, ar).astype(np.float32)
    return res['wsum'] + np.float32(alpha) * sim


def block_chunk(keys, data, rec_idx, cand_cap, sim_alpha=None):
    """Block a set of absolute S1 record indices -> capped candidate CSR."""
    n = len(rec_idx)
    rec_idx = np.asarray(rec_idx, dtype=np.int64)
    alpha = SIM_CAP_ALPHA if sim_alpha is None else sim_alpha
    ks = fb.subset_keys(keys, rec_idx)
    cfg = {'pool': {st: dict(POOL_CFG[st]) for st in fb.POOLS}}
    for st in cfg['pool']:
        cfg['pool'][st]['k'] = None          # one global cap below, by rarity
    res = fb.block_split(ks, n, data['n23'], cfg, chunk_records=n,
                         country1=data['s1']['countries'][rec_idx],
                         country23=data['s23']['countries'])
    score = None
    if alpha > 0:
        score = cap_score(res, n, dict(names1=data['s1']['names'][rec_idx],
                                       names23=data['s23']['names'],
                                       addrs1=data['s1']['addrs'][rec_idx],
                                       addrs23=data['s23']['addrs']), cand_cap,
                          alpha)
    res = fb.cap_result(res, n, data['n23'], cand_cap, score=score)
    res['_keys'] = ks
    return res


def features_for(res, data, cache, s1_names, s1_addrs, chunk=2_000_000):
    """Vectorized features for one blocked chunk; returns (F, rec, cand).

    ``s1_names``/``s1_addrs`` MUST be exactly this chunk's S1 rows (same length
    and order as the chunk's records): pair features are indexed by *local*
    record ids, so passing the whole split here silently compares every pair
    against the wrong record — which shows up as AUC~0.5 for all string metrics.
    """
    n_chunk = len(s1_names)
    cnt = np.diff(res['off'])
    rec = np.repeat(np.arange(n_chunk, dtype=np.int32), cnt)
    ks = res.pop('_keys')
    ctx = dict(shared=res['shared'], nbits=res['nbits'], abits=res['abits'],
               shared_n=res.get('shared_n'), shared_a=res.get('shared_a'),
               wsum=res.get('wsum'), n2=data['n2'],
               cnt_ntok_a=np.diff(ks['ntok']['s1_off']).astype(np.int32),
               cnt_ntok_b=cache['cnt_ntok_b'],
               cnt_aword_a=np.diff(ks['aword']['s1_off']).astype(np.int32),
               cnt_aword_b=cache['cnt_aword_b'])
    return fs.compute_features(s1_names, s1_addrs,
                               cache['names23'], cache['addrs23'],
                               rec, res['cand'], ctx, chunk=chunk)


def data_cache(data, keys):
    return dict(names1=data['s1']['names'], addrs1=data['s1']['addrs'],
                names23=data['s23']['names'], addrs23=data['s23']['addrs'],
                cnt_ntok_b=np.diff(keys['ntok']['s23_off']).astype(np.int32),
                cnt_aword_b=np.diff(keys['aword']['s23_off']).astype(np.int32))


def macro_f05(pred_count, true_count, tp_count):
    """Per-record F0.5 with the scorer's singleton rule, macro-averaged."""
    both_empty = (true_count == 0) & (pred_count == 0)
    false_merge = (true_count == 0) & (pred_count > 0)
    empty_pred = (true_count > 0) & (pred_count == 0)
    precision = np.where(pred_count > 0, tp_count / np.maximum(pred_count, 1), 0.0)
    recall = np.where(true_count > 0, tp_count / np.maximum(true_count, 1), 0.0)
    denom = 0.25 * precision + recall
    f = np.where(denom > 0, 1.25 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    scores = np.where(both_empty, 1.0, np.where(false_merge, 0.0,
                      np.where(empty_pred, 0.0, f)))
    return float(scores.mean()), dict(
        both_empty=int(both_empty.sum()), false_merge=int(false_merge.sum()),
        missed_all=int(empty_pred.sum()), partial=int((~both_empty & ~false_merge
                                                       & ~empty_pred).sum()))


def evaluate_at_threshold(rec, cand, prob, threshold, gt_local, n_records):
    """Macro F0.5 + precision/recall at a fixed threshold."""
    nt = np.diff(gt_local['offsets']).astype(np.float64)
    tp_flag = fs.make_labels(rec, cand, gt_local, 1 << 30)
    hit = prob >= threshold
    r = rec[hit]
    pred = np.bincount(r, minlength=n_records).astype(np.float64)
    tp = np.bincount(r[tp_flag[hit] == 1], minlength=n_records).astype(np.float64)
    macro, breakdown = macro_f05(pred, nt, tp)
    stats = dict(precision=float(tp.sum() / max(pred.sum(), 1)),
                 recall=float(tp.sum() / max(nt.sum(), 1)),
                 predicted=int(pred.sum()), true=int(nt.sum()),
                 tp=int(tp.sum()), **breakdown)
    return macro, stats


def tune_threshold(rec, cand, prob, gt_local, n_records, thresholds=None):
    """Sweep thresholds for macro F0.5 (the competition metric)."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 0.995, 0.01), 4)
    tp_flag = fs.make_labels(rec, cand, gt_local, 1 << 30)
    nt = np.diff(gt_local['offsets']).astype(np.float64)
    best = (0.5, -1.0, {})
    for t in thresholds:
        hit = prob >= t
        r = rec[hit]
        pred = np.bincount(r, minlength=n_records).astype(np.float64)
        tp = np.bincount(r[tp_flag[hit] == 1], minlength=n_records).astype(np.float64)
        macro, breakdown = macro_f05(pred, nt, tp)
        if macro > best[1]:
            best = (float(t), macro, dict(
                precision=float(tp.sum() / max(pred.sum(), 1)),
                recall=float(tp.sum() / max(nt.sum(), 1)),
                predicted=int(pred.sum()), true=int(nt.sum()), tp=int(tp.sum()),
                **breakdown))
    return best


def f05_by_country(rec, cand, prob, threshold, gt_local, n_records, ccodes):
    """Macro F0.5 restricted to each country present in the sample.

    The competition metric is already a macro average over S1 records, so
    splitting by the S1 record's country is the honest way to see whether an
    unseen country (test has France, train does not) scores worse.
    """
    nt = np.diff(gt_local['offsets']).astype(np.float64)
    tp_flag = fs.make_labels(rec, cand, gt_local, 1 << 30)
    hit = prob >= threshold
    r = rec[hit]
    pred = np.bincount(r, minlength=n_records).astype(np.float64)
    tp = np.bincount(r[tp_flag[hit] == 1], minlength=n_records).astype(np.float64)
    out = {}
    for name, code in sorted(fd.COUNTRY_MAP.items(), key=lambda kv: kv[1]):
        keep = ccodes == code
        if not keep.any():
            continue
        macro, breakdown = macro_f05(pred[keep], nt[keep], tp[keep])
        out[name] = dict(records=int(keep.sum()), f05=macro,
                         precision=float(tp[keep].sum() / max(pred[keep].sum(), 1)),
                         recall=float(tp[keep].sum() / max(nt[keep].sum(), 1)),
                         pred=int(pred[keep].sum()), true=int(nt[keep].sum()),
                         tp=int(tp[keep].sum()), **breakdown)
    return out


def country_pool(data, only_country):
    """Index pool for --only-country (comma list of US/India/France), else None."""
    if not only_country:
        return None
    m = {k.lower(): v for k, v in fd.COUNTRY_MAP.items()}
    codes = [m[w.strip().lower()] for w in only_country.split(',') if w.strip()]
    return np.flatnonzero(np.isin(data['s1']['countries'], codes))


def run_lengths(rec):
    """Group sizes for a rank objective: rows are contiguous per record.

    Records are only present if they have at least one candidate row, so this is
    a run-length encoding of the (sorted) record column rather than a bincount -
    a zero-size group would break LightGBM.
    """
    if len(rec) == 0:
        return np.zeros(0, dtype=np.int32)
    breaks = np.r_[True, rec[1:] != rec[:-1]]
    return np.diff(np.flatnonzero(np.r_[breaks, True])).astype(np.int32)


def sample_records(data, n_want, seed, only_country=None, tag=''):
    """spread_sample over the whole split or over a country-restricted pool."""
    pool = country_pool(data, only_country)
    if pool is None:
        return spread_sample(data['n1'], n_want, seed=seed)
    n_pool = min(n_want, len(pool))
    rec_idx = pool[spread_sample(len(pool), n_pool, seed=seed)]
    log(f"{tag}country pool '{only_country}': {len(pool):,} records available, "
        f"sampled {n_pool:,}")
    return rec_idx


def sample_pairs(data, keys, cache, rec_idx, gt, chunk_records, cand_cap,
                 progress=log, tag='', neg_rate=1.0, is_val=None):
    """Block + feature a set of records -> (X, y, rec, cand) in local space.

    rec is local (0..len(rec_idx)-1); labels come from the subset CSR so nothing
    has to be re-blocked later.

    neg_rate < 1 keeps every positive and validation negative but only that
    fraction of the *fit* negatives, per chunk, so a 5x bigger training sample
    fits in memory without touching the features (measured: features are built at
    ~1.2 M pairs/min, so pair count, not row count, is the real cost).
    """
    gt_local = subset_csr(gt['offsets'], gt['flat'], rec_idx)
    n = len(rec_idx)
    X_parts, y_parts, rec_parts, cand_parts = [], [], [], []
    t0 = time.time()
    n_pairs = 0
    n_raw = 0
    rng = np.random.default_rng(1234) if neg_rate < 1.0 else None
    for c0 in range(0, n, chunk_records):
        c1 = min(c0 + chunk_records, n)
        sub = rec_idx[c0:c1]
        res = block_chunk(keys, data, sub, cand_cap)
        if len(res['cand']) == 0:
            continue
        sub_idx = rec_idx[c0:c1]
        F, rec, cand = features_for(res, data, cache,
                                    cache['names1'][sub_idx],
                                    cache['addrs1'][sub_idx])
        rec_g = rec + c0
        y = fs.make_labels(rec_g, cand, gt_local, data['n23'])
        n_raw += len(y)
        if rng is not None:
            keep = (y == 1) | (rng.random(len(y)) < neg_rate)
            if is_val is not None:
                keep |= is_val[rec_g]          # keep every validation pair
            if keep.sum() < len(keep):
                F, rec_g, cand, y = F[keep], rec_g[keep], cand[keep], y[keep]
                if len(y) == 0:
                    continue
        X_parts.append(F)
        y_parts.append(y.astype(np.int8))
        rec_parts.append(rec_g)
        cand_parts.append(cand)
        n_pairs += len(rec_g)
        if progress and (c1 % (chunk_records * 5) == 0 or c1 == n):
            el = time.time() - t0
            progress(f"{tag}{c1:,}/{n:,} records, {n_pairs:,} pairs, "
                     f"{el:.0f}s (ETA {(n-c1)*el/max(c1,1)/60:.1f}min)")
    X = np.concatenate(X_parts) if X_parts else np.zeros((0, fs.N_FEATURES), np.float32)
    y = np.concatenate(y_parts) if y_parts else np.zeros(0, np.int8)
    rec_g = np.concatenate(rec_parts) if rec_parts else np.zeros(0, np.int32)
    cand_g = np.concatenate(cand_parts) if cand_parts else np.zeros(0, np.int32)
    if neg_rate < 1.0 and progress:
        progress(f"{tag}kept {len(y):,}/{n_raw:,} pairs "
                 f"(pos {int(y.sum()):,} = {100 * y.mean():.2f}%)")
    return X, y, rec_g, cand_g, gt_local


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------

def run_train(args):
    import lightgbm as lgb
    data = fd.load_split('train', with_ids=False)
    keys = load_keys('train')
    gt = load_pkl(os.path.join(CACHE, 'gt_train.pkl'))
    cache = data_cache(data, keys)

    rec_idx = sample_records(data, args.sample, 0, args.only_country, tag='train: ')
    n_sample = len(rec_idx)
    rng = np.random.default_rng(1)
    perm = rng.permutation(n_sample)
    n_val = max(int(n_sample * args.val_frac), 1000)
    is_val = np.zeros(n_sample, dtype=bool)
    is_val[np.sort(perm[:n_val])] = True
    log(f"train: sample={n_sample:,} records (fit={n_sample-n_val:,}, "
        f"val={n_val:,})  n23={data['n23']:,}  cand_cap={args.cand_cap}")

    X, y, rec_g, cand_g, gt_local = sample_pairs(
        data, keys, cache, rec_idx, gt, args.chunk_records, args.cand_cap,
        tag='train ', neg_rate=args.neg_rate, is_val=is_val)
    log(f"training matrix {X.shape} positives={int(y.sum()):,} "
        f"({100*y.mean():.2f}%)" +
        (f"  neg_rate={args.neg_rate} (fit negatives subsampled)"
         if args.neg_rate < 1.0 else ""))

    is_fit = ~is_val[rec_g]
    pos = int(y[is_fit].sum())
    neg = int(is_fit.sum() - pos)
    params = dict(LGB_PARAMS)
    rank = args.objective == 'lambdarank'
    if rank:
        # the task is "which of this record's candidates belong to its cluster",
        # so a rank objective on per-record groups matches the metric better than
        # a global binary logloss - but it gives up cross-record calibration, so
        # the threshold sweep (which is per-record anyway) has to re-find the cut
        params['objective'] = 'lambdarank'
        params['metric'] = 'ndcg'
        params['ndcg_eval_at'] = [3]
        params['label_gain'] = [0, 1]
        params.pop('scale_pos_weight', None)
    params['num_leaves'] = args.num_leaves
    params['learning_rate'] = args.learning_rate
    if args.scale_pos_weight:
        params['scale_pos_weight'] = args.scale_pos_weight
    log(f"  fit rows={int(is_fit.sum()):,} pos={pos:,} neg={neg:,} "
        f"scale_pos_weight={params.get('scale_pos_weight', 1.0):.1f}")

    fit_rec = rec_g[is_fit]
    train_data = lgb.Dataset(X[is_fit], label=y[is_fit], feature_name=fs.FEATURES,
                             group=(run_lengths(fit_rec) if rank else None))
    val_rows = np.flatnonzero(~is_fit)
    keep_val = val_rows[:min(len(val_rows), 500_000)]
    val_data = lgb.Dataset(X[keep_val], label=y[keep_val], reference=train_data,
                           group=(run_lengths(rec_g[keep_val]) if rank else None))
    model = lgb.train(params, train_data, num_boost_round=args.rounds,
                      valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(args.early_stopping,
                                                    verbose=False),
                                 lgb.log_evaluation(period=50)])
    log(f"  best_iteration={model.best_iteration}")
    imp = sorted(zip(fs.FEATURES, model.feature_importance('gain')),
                 key=lambda kv: -kv[1])
    log("  importance (gain): " + ", ".join(f"{k}={v:.0f}" for k, v in imp[:10]))

    prob = model.predict(X)
    # remap validation records into their own 0..n_val-1 index space
    val_idx = np.flatnonzero(is_val)
    remap = np.full(n_sample, -1, dtype=np.int32)
    remap[val_idx] = np.arange(len(val_idx), dtype=np.int32)
    vmask = is_val[rec_g]
    rec_val = remap[rec_g[vmask]]
    gt_val = subset_csr(gt_local['offsets'], gt_local['flat'], val_idx)
    t_thr, f_thr, s_thr = tune_threshold(rec_val, cand_g[vmask], prob[vmask],
                                         gt_val, len(val_idx))
    log(f"  val macro F0.5={f_thr:.4f} at threshold={t_thr:.3f}")
    log(f"  val stats: {json.dumps(s_thr)}")
    f_all = evaluate_at_threshold(rec_val, cand_g[vmask], prob[vmask], t_thr,
                                 gt_val, len(val_idx))[0]

    # ceiling check: perfect scoring on the *blocked* candidates.  Splits the
    # remaining loss into "blocker missed it" vs "model failed to pick it".
    tp_flag = fs.make_labels(rec_val, cand_g[vmask], gt_val, 1 << 30)
    n_true = np.diff(gt_val['offsets']).astype(np.float64)
    found = np.bincount(rec_val[tp_flag == 1], minlength=len(val_idx)).astype(np.float64)
    blk_recall = float((found[n_true > 0] / n_true[n_true > 0]).mean()) if (n_true > 0).any() else 0.0
    oracle, _ = macro_f05(found, n_true, found)
    log(f"  blocking macro recall on val={blk_recall:.4f}  "
        f"oracle F0.5 (perfect scoring)={oracle:.4f}")

    model_path = args.model or os.path.join(args.out_dir, 'model_fast.pkl')
    os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'threshold': t_thr, 'features': fs.FEATURES,
                     'pool_cfg': POOL_CFG, 'cand_cap': args.cand_cap,
                     'params': params, 'val_f05': f_thr, 'val_stats': s_thr,
                     'sample_records': int(n_sample),
                     'fit_records': int(n_sample - n_val),
                     'pairs': int(len(X)), 'best_iteration': model.best_iteration,
                     'trained_at': time.strftime('%Y-%m-%d %H:%M:%S')}, f)
    log(f"saved {model_path}")
    rep = dict(val_f05=f_thr, val_f05_recheck=f_all, threshold=t_thr, stats=s_thr,
               blocking_recall_val=blk_recall, oracle_f05=oracle,
               pairs=int(len(X)), fit_rows=int(is_fit.sum()), pos=pos, neg=neg,
               best_iteration=model.best_iteration,
               importance={k: float(v) for k, v in imp},
               pool_cfg=POOL_CFG, cand_cap=args.cand_cap,
               features=fs.FEATURES)
    with open(os.path.join(args.out_dir, 'train_report.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    return model_path


def run_eval(args):
    data = fd.load_split('train', with_ids=False)
    keys = load_keys('train')
    gt = load_pkl(os.path.join(CACHE, 'gt_train.pkl'))
    cache = data_cache(data, keys)
    saved = load_pkl(args.model or os.path.join(args.out_dir, 'model_fast.pkl'))
    model, t = saved['model'], saved['threshold']
    # the model's own feature list defines the matrix columns, so a 46-feature
    # model is scored on 46 columns without the caller having to remember a flag
    fs.set_features(saved.get('features', fs.FEATURES))

    # a fresh index range (different seed) so this is an independent holdout
    rec_idx = sample_records(data, args.sample, args.seed, args.only_country,
                             tag='eval: ')
    log(f"eval: {len(rec_idx):,} fresh records, model threshold={t:.3f}")
    X, y, rec_g, cand_g, gt_local = sample_pairs(
        data, keys, cache, rec_idx, gt, args.chunk_records, args.cand_cap,
        tag='eval ')
    prob = model.predict(fs.select_features(X, saved.get('features', fs.FEATURES)))
    if args.dump_pairs:
        # every val pair's probability, so post-processing (thresholds, conflict
        # cleanup, per-country rules) can be evaluated offline without refitting
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_pairs)), exist_ok=True)
        np.savez(args.dump_pairs, rec=rec_g, cand=cand_g,
                 prob=prob.astype(np.float32), rec_idx=rec_idx.astype(np.int64),
                 ccodes=data['s1']['countries'][rec_idx].astype(np.int8),
                 threshold=np.float32(t))
        log(f"  dumped {len(prob):,} pairs ({int((prob >= t).sum()):,} above "
            f"{t:.3f}) -> {args.dump_pairs}")
    macro, stats = evaluate_at_threshold(rec_g, cand_g, prob, t, gt_local,
                                         len(rec_idx))
    bt, bf, bs = tune_threshold(rec_g, cand_g, prob, gt_local, len(rec_idx))
    log(f"  macro F0.5 @{t:.3f} = {macro:.4f}   (best {bf:.4f} @ {bt:.3f})")
    log(f"  stats @{t:.3f}: {json.dumps(stats)}")
    by_country = f05_by_country(rec_g, cand_g, prob, t, gt_local, len(rec_idx),
                                data['s1']['countries'][rec_idx])
    for name, d in by_country.items():
        log(f"    {name:<7} n={d['records']:,}  macro F0.5={d['f05']:.4f}  "
            f"P={d['precision']:.3f} R={d['recall']:.3f}  "
            f"partial={d['partial']:,} missed={d['missed_all']:,} "
            f"false_merge={d['false_merge']:,}")
    log(f"  pairs={len(X):,}  blocking ceiling check:")
    out = dict(eval_records=len(rec_idx), threshold=t, f05=macro,
               best_threshold=bt, best_f05=bf, stats=stats, pairs=int(len(X)),
               by_country=by_country)
    with open(os.path.join(args.out_dir, 'eval_report.json'), 'w') as f:
        json.dump(out, f, indent=1)
    return macro


def run_predict(args):
    data = fd.load_split('test', with_ids=True)
    keys = load_keys('test')
    cache = data_cache(data, keys)
    saved = load_pkl(args.model or os.path.join(args.out_dir, 'model_fast.pkl'))
    model, t = saved['model'], saved['threshold']
    fs.set_features(saved.get('features', fs.FEATURES))
    model_features = fs.FEATURES
    n1 = data['n1']
    limit = args.max_records or n1
    start = args.start_records or 0
    n_rows = limit - start
    log(f"predict: records {start:,}..{limit:,} ({n_rows:,}), threshold={t:.3f}, "
        f"model val_f05={saved.get('val_f05'):.4f}, "
        f"model features={len(model_features)}/{fs.N_FEATURES}")

    match_path = os.path.join(args.out_dir, 'matching_results.tsv')
    cand_path = os.path.join(args.out_dir, 'candidate_pairs.tsv')
    scores_path = args.scores or os.path.join(args.out_dir, 'scores.tsv')
    ids1, ids23 = data['ids1'], data['ids23']
    t0 = time.time()
    n_pairs = n_matched = n_match_ids = n_score_rows = 0
    with open(match_path, 'w', encoding='utf-8') as fm, \
            open(cand_path, 'w', encoding='utf-8') as fc, \
            open(scores_path, 'w', encoding='utf-8') as fsout:
        fm.write('source1_entity_id\tmatched_entity_ids\n')
        fc.write('source1_entity_id\tcandidate_entity_ids\n')
        fsout.write('source1_entity_id\tcandidate_entity_id\tprob\n')
        log(f"  scores: pairs with prob >= {args.score_floor:.3f} -> {scores_path}")
        for r0 in range(start, limit, args.chunk_records):
            r1 = min(r0 + args.chunk_records, limit)
            res = block_chunk(keys, data, np.arange(r0, r1), args.cand_cap)
            F, rec, cand = features_for(res, data, cache,
                                        data['s1']['names'][r0:r1],
                                        data['s1']['addrs'][r0:r1])
            prob = model.predict(fs.select_features(F, model_features))
            n_pairs += len(cand)
            if args.score_floor is not None:
                sm = prob >= args.score_floor
                if sm.any():
                    rows = zip((rec[sm] + r0).tolist(), cand[sm].tolist(),
                               prob[sm].tolist())
                    # 6 decimals: at 4, a pair a hair below the threshold could
                    # round up to it and reappear when re-thresholding
                    fsout.write(''.join(f"{ids1[a]}\t{ids23[b]}\t{p:.6f}\n"
                                        for a, b, p in rows))
                    n_score_rows += int(sm.sum())
            sel = prob >= t
            m_rec, m_cand = rec[sel], cand[sel]
            mb = np.searchsorted(m_rec, np.arange(r1 - r0 + 1))
            cb = np.searchsorted(rec, np.arange(r1 - r0 + 1))
            m_lines, c_lines = [], []
            for i in range(r1 - r0):
                mc = m_cand[mb[i]:mb[i + 1]]
                if len(mc):
                    ids = sorted(ids23[mc].tolist())
                    n_matched += 1
                    n_match_ids += len(ids)
                    m_lines.append(f"{ids1[r0+i]}\t{','.join(ids)}\n")
                else:
                    m_lines.append(f"{ids1[r0+i]}\t\n")
                cc = cand[cb[i]:cb[i + 1]]
                c_lines.append(f"{ids1[r0+i]}\t{','.join(ids23[cc].tolist())}\n")
            fm.write(''.join(m_lines))
            fc.write(''.join(c_lines))
            if (r1 % (args.chunk_records * 10) == 0) or r1 == limit:
                el = time.time() - t0
                done = r1 - start
                log(f"  {done:,}/{n_rows:,} records of this range, "
                    f"{n_pairs:,} scored pairs, {el:.0f}s "
                    f"(ETA {(n_rows-done)*el/max(done,1)/60:.1f}min)")
    log(f"wrote {match_path}, {cand_path} and {scores_path}")
    log(f"  {n_matched:,}/{n_rows:,} records have >=1 match; "
        f"{n_match_ids:,} matched ids; {n_pairs:,} candidate pairs "
        f"({n_pairs/max(n_rows,1):.1f}/record); {n_score_rows:,} scored rows")
    with open(os.path.join(args.out_dir, 'predict_report.json'), 'w') as f:
        json.dump(dict(records=n_rows, record_start=start, record_end=limit,
                       threshold=t, matched_records=n_matched,
                       matched_ids=n_match_ids, candidates=n_pairs,
                       cand_per_record=n_pairs / max(n_rows, 1),
                       score_rows=n_score_rows,
                       model_val_f05=saved.get('val_f05'),
                       seconds=time.time() - t0), f, indent=1)
    return match_path, cand_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['train', 'eval', 'predict'])
    ap.add_argument('--sample', type=int, default=40000)
    ap.add_argument('--seed', type=int, default=42,
                    help='eval: sample seed; 42 = the canonical holdout')
    ap.add_argument('--chunk-records', type=int, default=4000)
    ap.add_argument('--cand-cap', type=int, default=CAND_CAP)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--only-country', default=None,
                    help='comma list (US,India,France) to restrict train/eval samples')
    ap.add_argument('--rounds', type=int, default=800)
    ap.add_argument('--early-stopping', type=int, default=50)
    ap.add_argument('--scale-pos-weight', type=float, default=0.0,
                    help='0 = train unweighted (threshold handles calibration)')
    ap.add_argument('--num-leaves', type=int, default=63)
    ap.add_argument('--learning-rate', type=float, default=0.05)
    ap.add_argument('--sim-cap-alpha', type=float, default=SIM_CAP_ALPHA,
                    help='blend cheap similarity into the candidate-cap ranking')
    ap.add_argument('--model', default=None)
    ap.add_argument('--out-dir', default='../../output')
    ap.add_argument('--max-records', type=int, default=None)
    ap.add_argument('--start-records', type=int, default=0,
                    help='predict: first S1 row of this shard (see tools/merge_shards.py)')
    ap.add_argument('--scores', default=None,
                    help='per-pair probability TSV (predict); default out-dir/scores.tsv')
    ap.add_argument('--neg-rate', type=float, default=1.0,
                    help='train: fraction of FIT negatives to keep (val keeps all)')
    ap.add_argument('--objective', default='binary', choices=['binary', 'lambdarank'],
                    help='lambdarank ranks each record(candidate group) by NDCG; it '
                         'fits the per-record metric but gives up global calibration')
    ap.add_argument('--extra-features', action='store_true',
                    help='add fast_score.EXTRA_FEATURES (measured worse at 60K: '
                         '0.9005 vs 0.9031 - re-validate before shipping)')
    ap.add_argument('--score-floor', type=float, default=0.10,
                    help='only write pairs at/above this probability to the scores TSV')
    ap.add_argument('--dump-pairs', default=None,
                    help='eval: .npz of every val pair (rec/cand/prob/rec_idx/ccodes)')
    args = ap.parse_args()
    globals()['SIM_CAP_ALPHA'] = args.sim_cap_alpha
    if not args.extra_features:
        fs.set_features(fs.BASE_FEATURES)
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    if args.mode == 'train':
        run_train(args)
    elif args.mode == 'eval':
        run_eval(args)
    else:
        run_predict(args)
    log(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
