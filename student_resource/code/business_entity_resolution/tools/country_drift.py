#!/usr/bin/env python3
"""Per-country drift check — does the model treat France like US/India?

Train has no France records at all, and test is ~15% France, so the whole
"unseen country" risk has to be measured without labels.  What we *can* measure
on the test split is the shape of the model's own output per country:

  * pairs per record and the mean/quantiles of the record's best-candidate
    probability (a country the model finds "harder" has a lower top-1 curve)
  * the share of records whose best candidate clears the shipping threshold
  * the mean of the top string-similarity features on candidate pairs

If France's curves sit far below US/India on the same split, the global
threshold is mis-calibrated for France (fix: per-country threshold), and the
size of the gap says how much is at stake.  On the train split the same numbers
can be printed next to the real macro F0.5 per country, so the link between the
curve and the metric is calibrated rather than guessed.

Run from code/business_entity_resolution:
    python3 tools/country_drift.py --split test --per-country 6000 \
        --model ../../output/model41a30/model_fast.pkl --sim-cap-alpha 30
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
import fast_data as fd          # noqa: E402
import fast_score as fs         # noqa: E402
import pipeline_fast as pf      # noqa: E402

CACHE = fd.CACHE
NAMES = {v: k for k, v in fd.COUNTRY_MAP.items()}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stats_for(data, keys, cache, model, features, rec_idx, cap, alpha,
              chunk_records=4000):
    """Block+score one country's records; return curve stats and pair features."""
    per_rec = []
    feat_mean, feat_n = {}, 0
    watch = ['name_wratio', 'name_token_set', 'addr_token_set', 'addr_token_sort',
             'name_ratio', 'addr_ratio', 'shared_wsum', 'claim_pop', 'sim_rank']
    t0 = time.time()
    n_pairs = 0
    for r0 in range(0, len(rec_idx), chunk_records):
        sub = rec_idx[r0:r0 + chunk_records]
        res = pf.block_chunk(keys, data, sub, cap, sim_alpha=alpha)
        if len(res['cand']) == 0:
            per_rec.append(np.zeros(len(sub), np.float32))
            continue
        F, rec, cand = pf.features_for(res, data, cache,
                                      data['s1']['names'][sub],
                                      data['s1']['addrs'][sub])
        prob = model.predict(fs.select_features(F, features))
        n_pairs += len(prob)
        mx = np.zeros(len(sub), np.float32)
        np.maximum.at(mx, rec, prob)
        per_rec.append(mx)
        for name in watch:
            feat_mean[name] = feat_mean.get(name, 0.0) + float(
                F[:, fs.IDX[name]].sum())
        feat_n += len(prob)
    best = np.concatenate(per_rec) if per_rec else np.zeros(0, np.float32)
    out = dict(records=int(len(best)), pairs=int(n_pairs),
               pairs_per_record=float(n_pairs / max(len(best), 1)),
               seconds=round(time.time() - t0, 1),
               best_prob=dict(mean=float(best.mean()),
                              p10=float(np.quantile(best, 0.10)),
                              p25=float(np.quantile(best, 0.25)),
                              p50=float(np.quantile(best, 0.50)),
                              p75=float(np.quantile(best, 0.75)),
                              p90=float(np.quantile(best, 0.90))),
               feat_mean={k: v / max(feat_n, 1) for k, v in feat_mean.items()})
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test')
    ap.add_argument('--per-country', type=int, default=6000)
    ap.add_argument('--cand-cap', type=int, default=pf.CAND_CAP)
    ap.add_argument('--sim-cap-alpha', type=float, default=pf.SIM_CAP_ALPHA)
    ap.add_argument('--model', default='../../output/model41a30/model_fast.pkl')
    ap.add_argument('--threshold', type=float, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    data = fd.load_split(args.split, with_ids=False)
    keys = pf.load_keys(args.split)
    cache = pf.data_cache(data, keys)
    with open(args.model, 'rb') as f:
        saved = pickle.load(f)
    model, features = saved['model'], saved.get('features', fs.FEATURES)
    thr = args.threshold if args.threshold is not None else float(saved['threshold'])
    log(f"{args.split}: model={os.path.basename(args.model)} "
        f"threshold={thr:.3f} cap={args.cand_cap} alpha={args.sim_cap_alpha} "
        f"per_country={args.per_country:,}")

    rep = dict(split=args.split, model=os.path.abspath(args.model), threshold=thr,
               cand_cap=args.cand_cap, sim_cap_alpha=args.sim_cap_alpha, by_country={})
    for name, code in sorted(fd.COUNTRY_MAP.items(), key=lambda kv: kv[1]):
        pool = np.flatnonzero(data['s1']['countries'] == code)
        if len(pool) == 0:
            continue
        take = pool[pf.spread_sample(len(pool), args.per_country, seed=7)]
        st, best = stats_for(data, keys, cache, model, features, take,
                             args.cand_cap, args.sim_cap_alpha)
        above = float((best >= thr).mean())
        st['share_best_ge_threshold'] = above
        # thresholds that would put this country's best-prob curve on the same
        # quantile footing as the others (rank-preserving calibration idea)
        st['best_prob_at_quantile'] = {q: float(np.quantile(best, q))
                                       for q in (0.25, 0.50, 0.75)}
        rep['by_country'][name] = st
        log(f"  {name:<7} n={st['records']:,} pairs/rec={st['pairs_per_record']:.1f} "
            f"best_p mean={st['best_prob']['mean']:.4f} "
            f"p25={st['best_prob']['p25']:.4f} p50={st['best_prob']['p50']:.4f} "
            f"p75={st['best_prob']['p75']:.4f} "
            f"share>={thr:.2f}={above:.3f}  ({st['seconds']:.0f}s)")
        log("          pair-feature means: " + " ".join(
            f"{k}={v:.3f}" for k, v in st['feat_mean'].items()))

    out = args.out or os.path.join(CACHE, f'country_drift_{args.split}.json')
    with open(out, 'w') as f:
        json.dump(rep, f, indent=1)
    log(f"wrote {out}")


if __name__ == '__main__':
    main()
