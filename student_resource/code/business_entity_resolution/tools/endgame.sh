#!/usr/bin/env bash
# Endgame, memory-safe: run the shards ONE AT A TIME, merge, validate, probes.
#
#   tools/endgame.sh --shards C,D [--resume 18760] [--out ../../output/final]
#
# The second shard is expected to be parked (`kill -STOP`) so only one heavy
# process ever runs; this script resumes it as soon as the first reports, which
# keeps a 16 GB Mac usable (two parallel predicts + a training grew the swap to
# 39 GB and tripped macOS's out-of-application-memory force quit).
#
# Completion signal = each shard's predict_report.json (written last).
# Everything is appended to logs/endgame.log.
set -uo pipefail
cd "$(dirname "$0")/.."                       # code/business_entity_resolution
BASE=../..
LOG=logs/endgame.log
mkdir -p "$BASE/output/final"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

SHARDS="C,D"; OUT="../../output/final"; RESUME_PID=""
while [ $# -gt 0 ]; do
  case "$1" in
    --shards) SHARDS=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --resume) RESUME_PID=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
IFS=',' read -ra S <<< "$SHARDS"
PARTS=(); REPORTS=()
for s in "${S[@]}"; do
  PARTS+=("../../output/final41_shard$s/matching_results.tsv")
  REPORTS+=("../../output/final41_shard$s/predict_report.json")
done

log "waiting for shard ${S[0]} (one job at a time, by design)"
while [ ! -f "${REPORTS[0]}" ]; do sleep 60; done
log "shard ${S[0]} reported"

if [ -n "$RESUME_PID" ]; then
  if kill -0 "$RESUME_PID" 2>/dev/null; then
    kill -CONT "$RESUME_PID" && log "resumed parked shard pid $RESUME_PID"
  else
    log "WARNING: pid $RESUME_PID is gone; that shard must be re-run by hand"
  fi
fi
for i in "${!S[@]}"; do
  [ "$i" -eq 0 ] && continue
  log "waiting for shard ${S[$i]}"
  while [ ! -f "${REPORTS[$i]}" ]; do sleep 60; done
done
log "all shards reported; merging"

/opt/homebrew/bin/python3 tools/merge_shards.py --parts "${PARTS[@]}" \
    --out "$OUT/matching_results.tsv" --append-candidates \
    --expected ../../dataset/test/test_source1.tsv >>"$LOG" 2>&1
if [ $? -ne 0 ]; then
  log "MERGE FAILED - stopping, nothing may be uploaded from a failed merge"
  exit 1
fi

log "official validator (this is the upload gate)"
/opt/homebrew/bin/python3 "$BASE/utils/validate_submission.py" \
    --matching "$OUT/matching_results.tsv" --test-dir "$BASE/dataset/test" >>"$LOG" 2>&1
log "validator exit=$?"

SCORES=()
for s in "${S[@]}"; do SCORES+=("../../output/final41_shard$s/scores.tsv"); done
for t in 0.93 0.97; do
  tag=$(echo "$t" | tr -d '.')
  log "threshold probe $t -> matching_results_t${tag}.tsv"
  /opt/homebrew/bin/python3 tools/threshold_apply.py --scores "${SCORES[@]}" \
      --threshold "$t" --out "$OUT/matching_results_t${tag}.tsv" >>"$LOG" 2>&1
done

log "rows: $(wc -l < "$OUT/matching_results.tsv")  <- output/final/matching_results.tsv is the upload"
log "done"
