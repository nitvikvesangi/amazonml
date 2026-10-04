# Amazon ML Challenge 2026 — Business Entity Resolution

Public leaderboard: **macro F0.5 = 0.914** (Round-2 submission, validated against the organisers' official validator).
Scale: **1.73 M** Source-1 entities, **9.97 M** records across three noisy sources, **3.4 × 10⁸** candidate pairs considered.

---

## The task

Business identity records arrive from three independent sources as partial, noisy fragments — no shared key, no
reliable address format, names mangled by abbreviations, transliterations, typos and DBA names. For every Source-1
record we must return the list of Source-2/Source-3 records that refer to the same real-world business.

Scoring is **macro F0.5** (β = 0.5), so precision matters roughly twice as much as recall: a wrong ID costs more than
a missing one. A Source-1 entity may legitimately have **zero** matches (116,537 of the 1.73 M test entities do), so
the empty row is a prediction, not an error.

Wrinkles that shaped the design:

| Wrinkle | Consequence |
| --- | --- |
| Test set adds **France**, absent from training | country treated as an open string label; its own frozen threshold |
| Tabs also separate the ID-list column | strict TSV writing, no pandas `to_csv` shortcuts |
| 10 M records × 3 sources | blocking + sharded scoring must stream, never materialise all pairs |
| EL (LightGBM 4.7) × (torch 2.14) won't co-install | a small CPU-only side venv for the cross-encoder stage |

## Approach

```
   raw TSV ──▶ normalisation ──▶ rarity-weighted budgeted blocking ──▶ 41-feature GBDT ──▶ cross-encoder rescue ──▶ thresholds ──▶ matching_results.tsv
                                 (4 pools, ≤200 cand/record)          (LightGBM, 500K train)   (US + India only)        (frozen)
```

1. **Normalisation.** Unicode fold (`unidecode`), case-fold, strip punctuation and legal suffixes (`pvt`, `ltd`, `inc`,
   `llp`, …), expand common address tokens, extract numeric atoms (house numbers, PINs, postcodes) and word/prefix
   tokens.
2. **Rarity-weighted budgeted blocking.** Four inverted-index pools — name tokens, name prefixes, address numbers,
   address words — each with a document-frequency cap and its own candidate budget (1000 / 400 / 200 / 500 per
   Source-1 record). A record's finite candidate budget is filled in the order that rarity and an α-30
   lexical similarity blend rank highest, capping every Source-1 record at **200 candidates**. This keeps recall high
   while producing a scoring set two orders of magnitude smaller than the raw cross product.
3. **41-feature pair model.** Lexical similarity (token-set, Jaro-Winkler, partial ratios), token overlap on
   name/address, numeric-atom agreement, blocking-pool provenance, document-frequency statistics, missing-field
   flags. Trained with **LightGBM** (`num_leaves` 63, `learning_rate` 0.05, ≤800 rounds, early stopping 50) on a
   500K-record sample with a 12% negative rate.
4. **Cross-encoder rescue pass** (US and India only — France's frozen threshold already cleared it). A
   `paraphrase-multilingual-MiniLM-L12-v2` cross-encoder, six frozen layers, 10.8 M trainable parameters, trained on
   152,159 mined pairs (val AUC 0.9426) over `name | address[:200]` per side. Rescue: `0.60 ≤ p < 0.94 and ce ≥ 0.70`
   → accept at 0.945. Clean: `0.94 ≤ p < 0.99 and ce < 0.20` → drop. Net **+155,063** entity matches.
5. **Frozen thresholds.** GBDT score **≥ 0.940** (US, India), **≥ 0.970** (France); every ID below the bar is dropped
   rather than emitted, which is what the F0.5 metric rewards.

### Evidence

| Measurement | Value |
| --- | --- |
| Clean 30K holdout, GBDT only | 0.9100 macro F0.5 |
| Same holdout **+ cross-encoder** | **0.9327** |
| Holdout oracle (best possible with our candidates) | 0.9745 |
| Public LB ladder (6 of 7 submissions used) | 0.879 → 0.877 → 0.875 → 0.883 → 0.889 → **0.914** |
| Test-set per-country match rate | US 0.942 · India 0.923 · France 0.938 |

## Repository layout

```
README.md                              ← this file
LICENSE
student_resource/
  README.md                            ← organisers' problem statement (verbatim copy)
  Documentation_template.md            ← the approach document written for Round 2
  code/business_entity_resolution/
    README.md                          ← authoritative run instructions
    requirements.txt                   ← pinned: pandas 3.0.5, numpy 2.5.2, lightgbm 4.7.0, torch 2.14.0, …
    src/                               ← core pipeline: pipeline_fast.py, fast_block/keys/data/score.py
    tools/                             ← training, evaluation, diagnostics, validation, packaging scripts
```

The campaign's working notes, the verification reports, the organiser dataset and every trained artifact stay outside
git: they are either reproducible, derivable, or hundreds of megabytes. The submission archive described below carries
the trained artifacts and the produced outputs.

## Reproducing

The dataset is organiser-provided and **not redistributed here**. Place it as `dataset/{train,test}/…` inside
`student_resource/` and follow
[`code/business_entity_resolution/README.md`](student_resource/code/business_entity_resolution/README.md), which
documents the full recipe — feature build, blocking, training, prediction, cross-encoder pass, thresholding.

Fast replay of the *shipped* result (≈1 minute on the frozen artifacts, no training):

```bash
# score the shipped candidate subset through the cross-encoder
.venv-sem/bin/python3 tools/ce_apply_test.py \
  --scores artifacts/gbdt_scores/pred500k_C.tsv artifacts/gbdt_scores/pred500k_D.tsv \
  --load-ce artifacts/ce_scores_cache --c-res 0.70 --c-cl 0.20

# apply the frozen thresholds  →  byte-identical matching_results.tsv
python3 tools/threshold_apply.py --threshold 0.94 --country-thr France=0.97
```

`matching_results.tsv` produced this way hashes to
`b0bb7de26cde0102cd501a77d21e9ade68eabc72aa1b95b2def51b673acc915d` — the exact file that scored 0.914.

### Submitted archive

The Round-2 package (342 MB, 83 members) contains `output/matching_results.tsv`, `output/candidate_pairs.tsv`, the full
`code/` tree, the approach document and a `MANIFEST.json` with sha256 hashes for every member. It was checked with the
organisers' validator in package mode:

> `PASS — no blocking issues found. Safe to submit.`

Because the upload field caps a package at 1024 MB, the shipped `candidate_pairs.tsv` is the *scored* candidate subset
the shipped scorer evaluated (14,408,403 pairs) rather than the 343 M-pair raw blocking output; the raw set is one
bundled command away and the choice is disclosed in the manifest, the code README and the approach document.

## License

MIT — see [LICENSE](LICENSE). Third-party components keep their own licences — LightGBM (MIT),
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0) — both inside the challenge's
"MIT/Apache-2.0, ≤8 B parameters" rule. No external data and no hosted APIs were used.
