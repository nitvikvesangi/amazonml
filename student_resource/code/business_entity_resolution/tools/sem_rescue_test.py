#!/usr/bin/env python3
"""Apply an embedding-cosine rescue rule to the saved test scores -> matching_results.tsv.

Rule (validated on the canonical holdout by tools/sem_probe.py first):

    accept pair  <=>  prob >= thr[country]
                      or  (lo <= prob < hi  and  cos(name1, name23) >= c
                           and country in --countries)

France is deliberately excluded from the rescue unless listed: its scores are
miscalibrated the other way (raising France 0.94 -> 0.97 gained +0.004 on the
LB, the only correction the leaderboard had to make), so it keeps its own
threshold instead of taking a US/India-trained rule.

Two streaming passes over the scores; only band rows get embeddings, and only
their cosine is kept, so memory stays flat.

    python3 tools/sem_rescue_test.py \
        --scores ../../output/final41_shardC/scores.tsv ../../output/final41_shardD/scores.tsv \
        --lo 0.30 --hi 0.94 --cos 0.85 --countries US,India \
        --out ../../output/final/matching_results_sem.tsv \
        --report ../../output/final/matching_results_sem_report.json
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
R_CHUNK = 3_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores', nargs='+', required=True)
    ap.add_argument('--lo', type=float, default=0.30)
    ap.add_argument('--hi', type=float, default=0.94)
    ap.add_argument('--cos', type=float, default=0.85)
    ap.add_argument('--thr', type=float, default=0.94)
    ap.add_argument('--country-thr', action='append', default=[])
    ap.add_argument('--countries', default='US,India',
                    help='countries the rescue rule is allowed to touch')
    ap.add_argument('--dataset', default='../../dataset/test')
    ap.add_argument('--out', required=True)
    ap.add_argument('--report', default=None)
    args = ap.parse_args()

    rescue_cc = args.countries.split(',')
    thr_map = {}
    for spec in args.country_thr:
        cc, _, val = spec.partition('=')
        thr_map[cc] = float(val)
    log(f"rule: prob>=thr[country] | ({args.lo}<=prob<{args.hi} & cos>={args.cos} "
        f"& country in {rescue_cc}); overrides {thr_map}")

    t0 = time.time()
    s1 = pd.read_csv(os.path.join(args.dataset, 'test_source1.tsv'), sep='\t',
                     dtype=str, keep_default_na=False, na_filter=False)
    ids1 = s1['entity_id'].to_numpy()
    country1 = s1['country'].to_numpy()
    name1 = s1['business_name'].to_numpy()
    thr_rec = np.full(len(ids1), np.float32(args.thr), dtype=np.float32)
    for cc, t in thr_map.items():
        thr_rec[country1 == cc] = np.float32(t)
    s1_index = {x: i for i, x in enumerate(ids1.tolist())}
    in_rescue_cc = np.isin(country1, rescue_cc)
    log(f"S1 rows {len(ids1):,} | threshold overrides {thr_map} | "
        f"rescue countries {rescue_cc} ({int(in_rescue_cc.sum()):,} rows)")

    def chunk_iter():
        for part in args.scores:
            for ch in pd.read_csv(part, sep='\t',
                                  dtype={'source1_entity_id': str,
                                         'candidate_entity_id': str,
                                         'prob': np.float32},
                                  chunksize=R_CHUNK, keep_default_na=False,
                                  na_filter=False):
                yield part, ch

    # pass 1: band rows in the rescue countries
    band_r, band_c = [], []
    n_rows = 0
    for part, ch in chunk_iter():
        p = ch['prob'].to_numpy(dtype=np.float32)
        ri = np.fromiter((s1_index[x] for x in ch['source1_entity_id']),
                         np.int64, len(ch))
        sel = (p >= args.lo) & (p < args.hi) & in_rescue_cc[ri]
        n_rows += len(ch)
        if sel.any():
            band_r.append(ch['source1_entity_id'].to_numpy()[sel])
            band_c.append(ch['candidate_entity_id'].to_numpy()[sel])
    band_r = np.concatenate(band_r) if band_r else np.zeros(0, dtype=object)
    band_c = np.concatenate(band_c) if band_c else np.zeros(0, dtype=object)
    log(f"rows scanned {n_rows:,}; band pairs to check {len(band_r):,} "
        f"({time.time()-t0:.0f}s)")

    # names for the band's S2/S3 ids
    need23 = set(band_c.tolist())
    name23 = {}
    for src in (2, 3):
        path = os.path.join(args.dataset, f'test_source{src}.tsv')
        got = 0
        for ch in pd.read_csv(path, sep='\t', usecols=['entity_id', 'business_name'],
                              dtype=str, keep_default_na=False, na_filter=False,
                              chunksize=1_000_000):
            hit = ch[ch['entity_id'].isin(need23)]
            if len(hit):
                name23.update(zip(hit['entity_id'].tolist(),
                                  hit['business_name'].tolist()))
                got += len(hit)
            if got >= len(need23):
                break
        log(f"  source{src}: {got:,}/{len(need23):,} band candidates have names")
    del need23

    name1_map = dict(zip(ids1.tolist(), name1.tolist()))

    from sentence_transformers import SentenceTransformer
    import torch
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = SentenceTransformer(MODEL, device=dev)
    cos = np.zeros(len(band_r), np.float32)
    CH = 150_000
    t_emb = time.time()
    for i in range(0, len(band_r), CH):
        j = min(i + CH, len(band_r))
        a = [name1_map[x] for x in band_r[i:j].tolist()]
        b = [name23.get(x, '') for x in band_c[i:j].tolist()]
        uniq = list(dict.fromkeys([s for s in a + b if s]))
        E = model.encode(uniq, batch_size=512, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True
                         ).astype(np.float32)
        idx = {s: k for k, s in enumerate(uniq)}
        c = np.zeros(j - i, np.float32)
        ok = np.array([s != '' for s in b])
        if ok.any():
            ea = E[[idx[s] for s in np.array(a)[ok]]]
            eb = E[[idx[s] for s in np.array(b)[ok]]]
            c[ok] = np.einsum('ij,ij->i', ea, eb)
        cos[i:j] = c
        if (i // CH) % 10 == 0:
            log(f"  embedded {j:,}/{len(band_r):,} ({(j)/max(time.time()-t_emb,1e-9):,.0f}/s)")
    log(f"embeddings done in {time.time()-t_emb:.0f}s; cos quantiles "
        f"{np.round(np.quantile(cos, [0.1, 0.5, 0.9]), 3)}, "
        f"rescue accepts {int((cos >= args.cos).sum()):,}/{len(cos):,}")

    # sorted key array for a vectorized membership test in pass 2
    res_keys = band_r[cos >= args.cos].astype(object)
    res_keys = np.char.add(np.char.add(res_keys.astype(str), '\t'),
                           band_c[cos >= args.cos].astype(str))
    res_keys = np.sort(res_keys)

    # pass 2: accept and collect
    acc = {}
    for part, ch in chunk_iter():
        p = ch['prob'].to_numpy(dtype=np.float32)
        r = ch['source1_entity_id'].to_numpy()
        c = ch['candidate_entity_id'].to_numpy()
        ri = np.fromiter((s1_index[x] for x in r), np.int64, len(ch))
        keep = p >= thr_rec[ri]
        cand_mask = (~keep) & (p >= args.lo) & (p < args.hi) & in_rescue_cc[ri]
        if cand_mask.any() and len(res_keys):
            keys = np.char.add(np.char.add(r[cand_mask].astype(str), '\t'),
                               c[cand_mask].astype(str))
            pos = np.searchsorted(res_keys, keys)
            pos = np.clip(pos, 0, max(len(res_keys) - 1, 0))
            hit = res_keys[pos] == keys
            idx = np.flatnonzero(cand_mask)[hit]
            keep[idx] = True
        for k in np.flatnonzero(keep):
            acc.setdefault(int(ri[k]), []).append(c[k])
        log(f"  {part}: pass 2 chunk done, accepted so far "
            f"{sum(len(v) for v in acc.values()):,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n_matched = n_ids = 0
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        lines = []
        for i, x in enumerate(ids1.tolist()):
            v = acc.get(i)
            if v:
                v.sort()
                n_matched += 1
                n_ids += len(v)
                lines.append(f"{x}\t{','.join(v)}\n")
            else:
                lines.append(f"{x}\t\n")
            if len(lines) >= 200_000:
                f.write(''.join(lines))
                lines = []
        f.write(''.join(lines))
    log(f"wrote {args.out}: {n_matched:,}/{len(ids1):,} rows matched, {n_ids:,} ids, "
        f"{n_ids/max(n_matched,1):.2f} ids/row, total {time.time()-t0:.0f}s")
    if args.report:
        with open(args.report, 'w') as f:
            json.dump(dict(scores=list(args.scores), lo=args.lo, hi=args.hi,
                           cos=args.cos, thr=args.thr, country_thr=thr_map,
                           countries=rescue_cc, band_pairs=int(len(band_r)),
                           rescue_accepts=int((cos >= args.cos).sum()),
                           matched_records=n_matched, matched_ids=n_ids,
                           seconds=round(time.time() - t0, 1)), f, indent=1)


if __name__ == '__main__':
    main()
