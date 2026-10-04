#!/usr/bin/env bash
# mem_guard: if macOS swap climbs back to the level that tripped the
# out-of-application-memory force quit, park the *low-value* job and say so.
# It never touches the prediction shards - those are the deliverable.
#
#   tools/mem_guard.sh [limit_mb] [pid ...]
#
# Defaults: park the LambdaRank training if swap used passes 30 GB.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/mem_guard.log
LIMIT=${1:-30000}
shift || true
PIDS=${*:-19678}

sw() { sysctl -n vm.swapusage | awk -F'used = ' '{split($2, a, "M"); print int(a[1])}'; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "guarding: limit ${LIMIT}MB, pids '${PIDS}' (swap now $(sw)MB)"
while :; do
  used=$(sw)
  if [ "${used:-0}" -gt "$LIMIT" ]; then
    for p in $PIDS; do
      if kill -0 "$p" 2>/dev/null; then
        kill -STOP "$p" && log "swap ${used}MB > ${LIMIT}MB -> parked $p (resume: kill -CONT $p)"
      fi
    done
    log "guard done; the predictions were left alone"
    exit 0
  fi
  sleep 60
done
