#!/bin/bash
# Gate for the 500K-record model: the moment training exits, eval it on the
# canonical holdout dump and compare with the 250K champion on the union-clean
# subset (both models' leaked eval records removed).
#
# Ship rule (decided in advance): re-predict the full test only if the clean
# delta is >= +0.003 at own thresholds or on the best grid threshold.
#
#   python3 tools/detach.py logs/gate500k.log tools/gate500k.sh
set -u
cd "$(dirname "$0")/.."
LOG=logs/gate500k.log
PY=/opt/homebrew/bin/python3
PID=30191

{
  echo "[$(date '+%H:%M:%S')] waiting for the 500K training (pid $PID)"
  for _ in $(seq 1 480); do          # up to 4 h
    kill -0 "$PID" 2>/dev/null || break
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] training gone (or wait expired)"
  tail -6 logs/train500k_base41.log
  if [ ! -f ../../output/model_base41_500k/model_fast.pkl ]; then
    echo "no model saved -> gate aborts (nothing to compare)"
    exit 1
  fi
  echo "[$(date '+%H:%M:%S')] eval 500K model on the canonical holdout (seed 42, 30K)"
  nice -n 10 $PY -u src/pipeline_fast.py eval --sample 30000 \
      --sim-cap-alpha 30 --chunk-records 4000 --cand-cap 200 \
      --model ../../output/model_base41_500k/model_fast.pkl \
      --dump-pairs ../../output/eval41a30_fixed/pairs_base41_500k.npz \
      --out-dir ../../output/eval_base41_500k
  echo "[$(date '+%H:%M:%S')] eval rc=$?"
  echo "[$(date '+%H:%M:%S')] clean-subset comparison vs champion (union of both train samples masked)"
  nice -n 10 $PY -u tools/compare_dumps.py \
      --dump ../../output/eval41a30_fixed/pairs_base41_250k.npz:champion250k \
      --dump ../../output/eval41a30_fixed/pairs_base41_500k.npz:model500k \
      --clean-train 250000 --clean-train 500000 \
      --out ../../output/eval41a30_fixed/compare_500k.json
  echo "[$(date '+%H:%M:%S')] GATE verdict: 500K ships only if DELTA >= +0.003"
  echo "[$(date '+%H:%M:%S')] swap: $(sysctl -n vm.swapusage)"
} >> "$LOG" 2>&1
