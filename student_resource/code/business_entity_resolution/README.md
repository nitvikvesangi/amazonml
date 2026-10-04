# Business Entity Resolution — Amazon ML Challenge 2026

Self-contained, runnable copy of the pipeline behind our best leaderboard submission
(**macro F0.5 = 0.914**). It regenerates both files in `output/` from the provided
`dataset/` alone: a rarity-weighted blocking stage → a 41-feature LightGBM classifier →
a fine-tuned cross-encoder that re-decides only the ambiguous US/India pairs → a frozen
threshold rule.

Everything below runs from **this folder** (`code/business_entity_resolution/`), and all
source lives in `src/` (library modules and CLI drivers side by side).

```
src/pipeline_fast.py      entry point: train / eval / predict — streams both output TSVs
src/fast_data.py          cache loader, country codes, int32 S23 index space
src/fast_keys.py          blocking-key builder (CSR int32 + inverted index + document freqs)
src/fast_block.py         budgeted, rarity-weighted blocker (4 independent pools, top-200 cap)
src/fast_score.py         41 vectorized pair features (rapidfuzz cpdist + cheap numpy)
src/pipeline.py           legacy v1/v2 pipeline (reference only — not part of the recipe)
src/build_cache.py        dataset TSVs → ../../cache (parsed rows, ground truth, index space)
src/build_keys.py         cache → blocking keys for one split
src/merge_shards.py       stitch predict shards back into one submission file (+ self-check)
src/candidates_from_scores.py  score matrix → the scored candidate_pairs.tsv that ships
src/check_outputs.py      bounded-memory audit: row counts, id rules, matches ⊆ candidates
src/threshold_apply.py    score matrix → submission TSV under an explicit threshold rule
src/ce_build_data.py      mine hard positive/negative pairs for the cross-encoder
src/ce_train.py           fine-tune the multilingual cross-encoder (MPS/CUDA/CPU)
src/ce_apply_test.py      score the ambiguous test pairs (cached, replayable) — the CE post-pass
src/ce_holdout_eval.py    measure the CE rule on the canonical holdout
src/ce_bench.py           cross-encoder throughput / precision benchmark
src/campaign_scripts/     the exact orchestration wrappers used during the campaign
artifacts/                reference artefacts of the shipped run (see "Fast verification")
requirements.txt          pinned dependencies for the reference environment
```

## 1. Environment

Reference machine: Apple M5 Pro, 10 cores, 16 GB RAM, macOS; **Python 3.14.7**.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # CPU pipeline; the CE steps add torch/transformers
python3 -c "import pandas, numpy, rapidfuzz, lightgbm; print('cpu pipeline ready')"
```

The blocking/GBDT stages need only the first four packages; `torch` + `transformers` are
needed for the cross-encoder steps (they run on Apple-silicon MPS, CUDA or CPU). On Apple
silicon the reference run used an arm64 PyPI build; on Linux/Windows pick the torch build
matching your CUDA/CPU setup.

## 2. Data layout

Extract the archive anywhere and place the provided dataset one level **above** this
folder, i.e. next to `code/` and `output/`:

```
<archive root>/
├── dataset/{train,test}/{train,test}_source{1,2,3}.tsv, train_ground_truth.tsv   (provided)
├── output/                      # matching_results.tsv + candidate_pairs.tsv (this submission)
└── code/business_entity_resolution/    ← run everything from here
```

Every command below uses the same `../../dataset` / `../../output` paths, and
`../../utils/validate_submission.py` is the organisers' validator (copy it next to
`dataset/` if it is not already there). Long runs are worth detaching —
`python3 src/detach.py logs/run.log <command…>` survives a closed shell on macOS.

## 3. Reproduce the submission end to end

```bash
mkdir -p logs

# (0) one-time caches: parsed rows, ground truth, normalised S1/S2/S3 strings (~5 min)
python3 src/build_cache.py --data-dir ../../dataset --out ../../cache
python3 src/build_keys.py --split train          # blocking keys + document frequencies (~90 s)
python3 src/build_keys.py --split test

# (1) GBDT on 500 k training records, α-30 candidates, 12 % negative rate (~45 min)
python3 src/pipeline_fast.py train --sample 500000 --sim-cap-alpha 30 --neg-rate 0.12 \
        --cand-cap 200 --chunk-records 4000 --out-dir ../../output/model_base41_500k

# (2) block + score + predict the whole test split as two halves (~2 h wall-clock)
for R in "0 866272 C" "866272 1732544 D"; do set -- $R
  python3 src/pipeline_fast.py predict --chunk-records 4000 --cand-cap 200 --sim-cap-alpha 30 \
      --model ../../output/model_base41_500k/model_fast.pkl \
      --start-records $1 --max-records $2 --out-dir ../../output/pred500k_$3 \
      --scores ../../output/pred500k_$3/scores.tsv &
done; wait

# (3) candidate set. The raw blocking output is the two halves merged — 343,192,595 pairs /
#     4.4 GB at cap 200 with alpha 30. The file that *ships* is the subset the shipped scorer
#     actually evaluated (prob >= 0.10): 14,408,403 pairs / 208 MB — the organisers' upload field
#     caps a package at 1024 MB and no compressor fits the raw set (see §5).
python3 src/merge_shards.py --parts ../../output/pred500k_C/candidate_pairs.tsv \
        ../../output/pred500k_D/candidate_pairs.tsv --out ../../output/candidate_pairs_raw.tsv
python3 src/candidates_from_scores.py --scores ../../output/pred500k_C/scores.tsv \
        ../../output/pred500k_D/scores.tsv --out ../../output/candidate_pairs.tsv \
        --expected ../../dataset/test/test_source1.tsv

# (4) cross-encoder training data: score 100 k training records (seed 7), then mine pairs
python3 src/pipeline_fast.py eval --sample 100000 --seed 7 --sim-cap-alpha 30 --chunk-records 4000 \
        --cand-cap 200 --model ../../output/model_base41_500k/model_fast.pkl \
        --dump-pairs ../../output/ce_train/sample_seed7.npz --out-dir ../../output/ce_train
python3 src/ce_build_data.py --dump ../../output/ce_train/sample_seed7.npz \
        --out ../../cache/ce_train.jsonl                 # 152,159 pairs (70.6 k pos / 81.6 k neg)
python3 src/ce_train.py --data ../../cache/ce_train.jsonl --out ../../output/ce_model \
        --freeze-layers 6 --batch 64 --epochs 2 --lr 2e-5 --seed 1234 --timebox 3300
                                                         # ~36 min on MPS, best val AUC 0.9426

# (5) CE-score the ambiguous US/India pairs and rewrite the score matrix (~27 min fp16 on MPS)
python3 src/ce_apply_test.py --scores ../../output/pred500k_C/scores.tsv \
        ../../output/pred500k_D/scores.tsv --model-dir ../../output/ce_model \
        --c-res 0.70 --c-cl 0.20 --fp16 --save-ce ../../output/ce_apply/cache \
        --out ../../output/ce_apply/scores_ce.tsv

# (6) the frozen decision rule → the submitted file
python3 src/threshold_apply.py --scores ../../output/ce_apply/scores_ce.tsv \
        --threshold 0.94 --country-thr France=0.97 --out ../../output/matching_results.tsv

# (7) gates: organisers' validator + the bounded-memory superset/format audit
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv \
        --test-dir ../../dataset/test
python3 src/check_outputs.py --matching ../../output/matching_results.tsv \
        --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```

The **decision rule** (identical in every stage, frozen before the final upload): a pair is
kept when its probability ≥ **0.940** (US, India) / ≥ **0.970** (France), where the
probability is the GBDT score *after* the cross-encoder post-pass, which only touches
US/India:

* **rescue** — `0.60 ≤ p < 0.94` and `CE ≥ 0.70` → `p := 0.945`
* **clean**  — `0.94 ≤ p < 0.99` and `CE < 0.20` → `p := 0.0`

## 4. Fast verification instead of a full re-run

The reference artefacts of the shipped run are bundled so the submitted file can be
reproduced in minutes on a laptop (no GPU, no re-training), and so every intermediate
number in `Documentation_template.md` can be checked:

| artefact | what it is |
|---|---|
| `artifacts/model_base41_500k/model_fast.pkl` | the shipped GBDT (LightGBM 4.7.0, 41 features, 4 MB) + its `train_report.json` |
| `artifacts/gbdt_scores/pred500k_C.tsv`, `pred500k_D.tsv` | the GBDT score matrix of the shipped run (the two predict halves, pairs above the 0.10 score floor) — the exact input of the CE stage |
| `artifacts/ce_train/ce_train.jsonl` | the 152,159 mined CE training pairs (rebuildable with `src/ce_build_data.py`) |
| `artifacts/ce_scores_cache/part0-7.npz` | CE probability for each of the 2,277,278 ambiguous test pairs (chunk cache written by `src/ce_apply_test.py --save-ce`) |
| `artifacts/reports/` | validator PASS output, the full `matches ⊆ candidates` audit over all 343,192,595 candidate ids, and the threshold/CE/per-country diff reports of the shipped run |

```bash
# replay the CE decisions and the frozen rule with the bundled artefacts: ~1 min, CPU only,
# no GPU, no re-training, no re-prediction (needs only the provided dataset/ for the id texts)
python3 src/ce_apply_test.py --scores artifacts/gbdt_scores/pred500k_C.tsv \
        artifacts/gbdt_scores/pred500k_D.tsv --load-ce artifacts/ce_scores_cache \
        --c-res 0.70 --c-cl 0.20 --out ../../output/ce_apply/scores_ce.tsv
python3 src/threshold_apply.py --scores ../../output/ce_apply/scores_ce.tsv \
        --threshold 0.94 --country-thr France=0.97 --out ../../output/matching_results.tsv
```

This replay was verified against the shipped run: the replayed score matrix is byte-identical
to the shipped `scores_ce.tsv` and the final file reproduces
`output/matching_results.tsv` (and therefore the leaderboard upload) exactly —
sha256 `b0bb7de26cde0102cd501a77d21e9ade68eabc72aa1b95b2def51b673acc915d`. Sizes, row
counts and sha256 of both shipped output files — and of every file in `src/` and `artifacts/` —
are recorded in `MANIFEST.json` at the archive root. After step 2 you can also point
`--scores` at your own `../../output/pred500k_C/scores.tsv` + `pred500k_D/scores.tsv`.

## 5. Notes

* **Why `output/candidate_pairs.tsv` is the scored subset.** The organisers cap a submission
  upload at **1024 MB**; the raw cap-200 / α-30 blocking output is 343,192,595 pairs (4.4 GB) and
  no compressor fits it — measured: deflate ≈ 1.6 GB and LZMA ≈ 3.0× on a 100 MB slice → ≈ 1.5 GB.
  The shipped file is every pair the shipped scorer evaluated (`prob ≥ 0.10`: 14,408,403 pairs,
  208 MB, 7,962 records with no scored candidate, ≈ 8.3 candidates per record), which contains
  *every* match in `matching_results.tsv`. It is regenerated by step 3 above; both files pass the
  organisers' validator in package mode (`--matching` + `--candidate`: **PASS**,
  `artifacts/reports/verify_package.txt`).
* **Determinism.** Blocking is deterministic; LightGBM and the cross-encoder are seeded
  (`--seed 42` for training samples, `--seed 1234` for the CE), but thread counts and MPS
  kernels can shift the last decimals. Compare against the sha256 in `MANIFEST.json` when
  you need byte equality; small score differences (≪ the 0.70/0.20 CE bands) do not change
  the decisions.
* **Cost.** Full re-run on the 10-core reference laptop: ≈ 45 min (GBDT training) + ≈ 2 h
  (whole-test prediction, two halves sharing the cores) + ≈ 10 min (CE mining) + ≈ 36 min
  (CE training) + ≈ 27 min (CE application). Peak RSS ≈ 7.5 GB during prediction;
  `src/pipeline_fast.py predict` runs in record chunks and streams both TSVs.
* **`src/campaign_scripts/`** are the shell wrappers actually used during the campaign
  (gate checks, shard orchestration, memory guards). They assume the original development
  checkout layout (drivers in `tools/`, live PIDs, `logs/*.log` state) and are included for
  provenance — the reproducible path is §3/§4 above.
* **Fair play & licences.** Only the provided training/test data is used; there is no
  external database, API, geocoder or identity lookup anywhere. Two public models are
  involved, both inside the *MIT / Apache-2.0, ≤ 8 B parameters* rule: LightGBM (MIT,
  ~0.5 M parameters) and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (Apache-2.0, 117 M parameters, first 6 layers frozen → 10.8 M trainable).

## 6. Critical invariants (breaking these silently ruins the score)

1. **Ground truth is keyed by entity id**, never by row order — `train_ground_truth.tsv` is
   in a different order from `train_source1.tsv` (100 % of rows).
2. **Pair features are indexed by chunk-local record ids**, so the S1 name/address arrays
   passed into the feature builder must be exactly that chunk's rows.
3. **Rank candidates by rarity weight** (`sum log1p(N/df)` over shared keys), not by
   shared-key count: count-ranking collapses recall to 0.069 at a 200-cap vs 0.856.
4. **Blend similarity into the cap ranking** (`--sim-cap-alpha 30`) — training and
   prediction must use the same α, otherwise the model is tuned on candidates the
   predictor never produces (blocking recall 0.856 → 0.940).
5. **Never threshold the GBDT score before the CE pass on US/India**, and never apply the
   CE rule to France (no French labels exist anywhere in training; its threshold comes from
   a label-free acceptance-rate analysis confirmed by a leaderboard probe).
