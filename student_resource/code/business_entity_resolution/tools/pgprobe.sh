#!/bin/bash
# Probe: why does `pgrep -f 'pipeline_fast.py predict'` return success inside
# the chained waiters even after the predicts exited?  Its own argv is just
# "bash tools/pgprobe.sh", so it cannot self-match.
set -u
cd "$(dirname "$0")/.."
LOG=logs/pgprobe.log
{
  echo "[$(date '+%H:%M:%S')] argv: $0 $*"
  echo "PATH=$PATH"
  echo "which -a pgrep:"; which -a pgrep
  echo "ps -p \$PPID:"; ps -o pid,command -p $PPID
  echo "--- /usr/bin/pgrep -f:"
  /usr/bin/pgrep -f 'pipeline_fast.py predict'; echo "rc=$?"
  echo "--- PATH pgrep -f:"
  pgrep -f 'pipeline_fast.py predict'; echo "rc=$?"
  echo "--- PATH pgrep -fl (list any matches):"
  pgrep -fl 'pipeline_fast.py predict'; echo "rc=$?"
  echo "--- pgrep versions:"
  /usr/bin/pgrep -v 2>&1 | head -2
  command -v pgrep
  pgrep -v 2>&1 | head -2
  echo "--- full ps lines mentioning pipeline_fast / detach / post500k / chain1m:"
  ps -eo pid,ppid,stat,command | grep -Ei 'pipeline_fast|detach|post500k|chain1m|verify_ship' | grep -v grep
  echo "[$(date '+%H:%M:%S')] probe done"
} >> "$LOG" 2>&1
