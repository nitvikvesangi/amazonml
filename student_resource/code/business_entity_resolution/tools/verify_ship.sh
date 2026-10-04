#!/bin/bash
# Validate every possible upload candidate, once, at memory-quiet moments.
#
# Phase 0: the emergency floor (champion scores, France 0.97) — validated now.
# Phase 1: ship500k_fr097.tsv, once post500k.sh has exited AND the 1M training
#          is no longer running (validation is stdlib-only but not free).
# Phase 2: ship1m_fr097.tsv, if chain1m.sh adopts the 1M model and builds it.
#
#   python3 tools/detach.py logs/verify_ship.log bash tools/verify_ship.sh
set -u
cd "$(dirname "$0")/.."
LOG=logs/verify_ship.log
PY=/opt/homebrew/bin/python3
D=../../output

validate () {   # $1 = tsv, $2 = tag
  local tsv="$1" tag="$2" rep="$D/${2}_verify.txt"
  local t0; t0=$(date '+%H:%M:%S')
  nice -n 20 $PY ../../utils/validate_submission.py \
      --matching "$tsv" --test-dir ../../dataset/test > "$rep" 2>&1
  local rc=$?
  {
    echo "[$t0] validate $tag"
    echo "  file: $tsv  rows=$(wc -l < "$tsv" | tr -d ' ')"
    tail -n 20 "$rep" | sed 's/^/  | /'
    if [ $rc -eq 0 ]; then
      echo "VALIDATOR PASS ($tag) — full report: $rep"
    else
      echo "VALIDATOR FAIL rc=$rc ($tag) — full report: $rep"
    fi
  } >> "$LOG"
}

{
  echo "=================================================================="
  echo "[$(date '+%H:%M:%S')] verify_ship start (floor -> 500K -> 1M)"
  echo "swap: $(sysctl -n vm.swapusage)"

  # ---- phase 0: emergency floor -------------------------------------------
  if [ -f "$D/final/matching_results_fr097.tsv" ] && [ ! -f "$D/floor_fr097_verify.txt" ]; then
    echo "[$(date '+%H:%M:%S')] phase 0: emergency floor validation"
    validate "$D/final/matching_results_fr097.tsv" floor_fr097
  fi

  # ---- phase 1: 500K ship --------------------------------------------------
  echo "[$(date '+%H:%M:%S')] phase 1: waiting for ship500k_fr097.tsv + post500k exit"
  for _ in $(seq 1 600); do      # cap 5 h
    if [ -f "$D/ship500k_fr097.tsv" ] && ! pgrep -f 'post500k' >/dev/null 2>&1; then break; fi
    sleep 30
  done
  if [ -f "$D/ship500k_fr097.tsv" ]; then
    echo "[$(date '+%H:%M:%S')] ship500k_fr097.tsv present; waiting for 1M training to end first"
    for _ in $(seq 1 480); do    # cap 4 h; triggers below fire much earlier
      if [ -f "$D/model_base41_1m/model_fast.pkl" ] || ! pgrep -f 'pipeline_fast.py train' >/dev/null 2>&1; then break; fi
      sleep 30
    done
    validate "$D/ship500k_fr097.tsv" ship500k_fr097
  else
    echo "ship500k_fr097.tsv never appeared — nothing to validate in phase 1"
  fi

  # ---- phase 2: 1M ship ----------------------------------------------------
  echo "[$(date '+%H:%M:%S')] phase 2: waiting for ship1m_fr097.tsv (cap 5 h)"
  for _ in $(seq 1 600); do
    if [ -f "$D/ship1m_fr097.tsv" ] && ! pgrep -f 'chain1m' >/dev/null 2>&1; then break; fi
    sleep 30
  done
  if [ -f "$D/ship1m_fr097.tsv" ]; then
    sleep 30
    validate "$D/ship1m_fr097.tsv" ship1m_fr097
  else
    echo "no ship1m_fr097.tsv (1M not adopted) — ship500k_fr097.tsv stands"
  fi
  echo "[$(date '+%H:%M:%S')] verify_ship done; swap: $(sysctl -n vm.swapusage)"
} >> "$LOG" 2>&1
