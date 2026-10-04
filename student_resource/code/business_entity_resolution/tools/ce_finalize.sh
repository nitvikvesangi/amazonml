#!/usr/bin/env bash
# Finalize the cross-encoder ship file end to end (run from anywhere).
#   scores_ce.tsv --threshold_apply--> ship_ce_fr097.tsv
#   -> official validator -> per-country shape -> diff vs 0.889 ship file.
# Heavy steps: validator ~2-3 min, ship_diff ~1-2 min. Total ~8 min.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/opt/homebrew/bin/python3
OUT=../../output
CE=$OUT/ce_apply/ship_ce_fr097.tsv
OLD=$OUT/ship500k_fr097.tsv
DIR=$OUT/ce_apply

echo "[finalize] $(date '+%H:%M:%S') threshold_apply (0.94 + France 0.97)"
$PY tools/threshold_apply.py \
    --scores "$DIR/scores_ce.tsv" \
    --threshold 0.94 --country-thr France=0.97 \
    --out "$CE" --report "$DIR/threshold_report.txt"

echo "[finalize] $(date '+%H:%M:%S') official validator"
$PY ../../utils/validate_submission.py \
    --matching "$CE" --test-dir ../../dataset/test | tee "$DIR/verify.txt"

echo "[finalize] $(date '+%H:%M:%S') per-country shape (new vs old)"
$PY tools/ship_table.py "$CE" "$OLD" | tee "$DIR/ship_table.txt"

echo "[finalize] $(date '+%H:%M:%S') diff vs 0.889 ship file"
$PY tools/ship_diff.py --a "$OLD" --b "$CE" | tee "$DIR/ship_diff.txt"

echo "[finalize] $(date '+%H:%M:%S') rows + size"
wc -l "$CE" | tee "$DIR/rows.txt"
ls -l "$CE" | tee "$DIR/size.txt"

echo "[finalize] $(date '+%H:%M:%S') DONE — inspect verify.txt / ship_diff.txt"
