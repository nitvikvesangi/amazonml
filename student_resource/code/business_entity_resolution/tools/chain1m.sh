#!/bin/bash
# Last data-size step: 1M-record model.
#
# The learning curve is still climbing (60k->250k gave +0.0035, 250k->500k
# +0.0052 on the union-clean holdout), so one more doubling is the only lever
# left with measured headroom.  Sequence, all serialized so the 16 GB box never
# runs two heavy jobs:
#
#   1. wait for the 500K predicts to exit
#   2. train 1M records (neg-rate 0.06, val-frac 0.08 to bound memory)
#   3. eval on the canonical holdout + compare against 250K/500K on the
#      union-clean subset (masks 250k, 500k, 1M)
#   4. if 1M beats 500K by >= +0.002 -> predict both halves + build variants
#
#   python3 tools/detach.py logs/chain1m.log tools/chain1m.sh
set -u
cd "$(dirname "$0")/.."
LOG=logs/chain1m.log
PY=/opt/homebrew/bin/python3
D=../../output
ROOT=$PWD

{
  echo "[$(date '+%H:%M:%S')] waiting for the 500K predicts to exit"
  for _ in $(seq 1 400); do
    pgrep -f 'pipeline_fast.py predict' >/dev/null || break
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] predicts gone; 60 s grace"
  sleep 60

  echo "[$(date '+%H:%M:%S')] launching 1M training (this is the long pole)"
  nice -n 10 $PY tools/detach.py logs/train1m_base41.log nice -n 10 $PY -u \
      src/pipeline_fast.py train --sample 1000000 --neg-rate 0.06 --val-frac 0.08 \
      --sim-cap-alpha 30 --out-dir $D/model_base41_1m
  sleep 30
  TPID=$(pgrep -f 'pipeline_fast.py train' | head -1)
  echo "[$(date '+%H:%M:%S')] 1M train pid=$TPID; guarding memory (park if swap > 26 GB)"
  nice -n 10 $PY tools/detach.py logs/mem_guard_1m.log bash tools/mem_guard.sh 26000 "${TPID:-0}"
  for _ in $(seq 1 480); do          # up to 4 h
    kill -0 "${TPID:-0}" 2>/dev/null || break
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] 1M training exited (or wait expired)"
  tail -8 logs/train1m_base41.log
  if [ ! -f $D/model_base41_1m/model_fast.pkl ]; then
    echo "no 1M model saved -> chain stops; the 500K ship stands"
    exit 0
  fi

  echo "[$(date '+%H:%M:%S')] eval 1M model on the canonical holdout"
  nice -n 10 $PY -u src/pipeline_fast.py eval --sample 30000 \
      --sim-cap-alpha 30 --chunk-records 4000 --cand-cap 200 \
      --model $D/model_base41_1m/model_fast.pkl \
      --dump-pairs $D/eval41a30_fixed/pairs_base41_1m.npz \
      --out-dir $D/eval_base41_1m
  echo "[$(date '+%H:%M:%S')] eval rc=$?"

  echo "[$(date '+%H:%M:%S')] union-clean comparison (masks 250k/500k/1M)"
  nice -n 10 $PY tools/compare_dumps.py \
      --dump $D/eval41a30_fixed/pairs_base41_500k.npz:model500k \
      --dump $D/eval41a30_fixed/pairs_base41_1m.npz:model1m \
      --clean-train 250000 --clean-train 500000 --clean-train 1000000 \
      --out $D/eval41a30_fixed/compare_1m.json

  line=$(grep -m1 'DELTA model1m' $LOG | tail -1)
  own=$(printf '%s' "$line" | sed -n 's/.*own-threshold \([+-][0-9.]*\).*/\1/p')
  best=$(printf '%s' "$line" | sed -n 's/.*best-on-grid \([+-][0-9.]*\).*/\1/p')
  echo "parsed: own=$own best=$best"
  ok=$($PY -c "print(1 if max(float('${own:-0}'), float('${best:-0}')) >= 0.002 else 0)")
  if [ "$ok" != "1" ]; then
    echo "[$(date '+%H:%M:%S')] 1M below +0.002 vs 500K -> no 1M predict; ship the 500K file"
    exit 0
  fi

  echo "[$(date '+%H:%M:%S')] 1M passed -> predicting both halves"
  nice -n 10 $PY tools/detach.py logs/pred1m_C.log nice -n 10 $PY -u \
      src/pipeline_fast.py predict --chunk-records 4000 --cand-cap 200 \
      --sim-cap-alpha 30 --model $D/model_base41_1m/model_fast.pkl \
      --start-records 0 --max-records 866272 \
      --out-dir $D/pred1m_C --scores $D/pred1m_C/scores.tsv
  nice -n 10 $PY tools/detach.py logs/pred1m_D.log nice -n 10 $PY -u \
      src/pipeline_fast.py predict --chunk-records 4000 --cand-cap 200 \
      --sim-cap-alpha 30 --model $D/model_base41_1m/model_fast.pkl \
      --start-records 866272 --max-records 1732544 \
      --out-dir $D/pred1m_D --scores $D/pred1m_D/scores.tsv
  for _ in $(seq 1 420); do
    pgrep -f 'pipeline_fast.py predict' >/dev/null || break
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] 1M predicts done; building variants"
  nice -n 10 $PY tools/threshold_apply.py --scores $D/pred1m_C/scores.tsv $D/pred1m_D/scores.tsv \
      --threshold 0.94 --country-thr France=0.97 \
      --out $D/ship1m_fr097.tsv --report $D/ship1m_fr097_report.json
  nice -n 10 $PY tools/threshold_apply.py --scores $D/pred1m_C/scores.tsv $D/pred1m_D/scores.tsv \
      --threshold 0.94 --out $D/ship1m_t094.tsv --report $D/ship1m_t094_report.json
  echo "[$(date '+%H:%M:%S')] 1M chain complete - final ship decision is the agent's"
  echo "[$(date '+%H:%M:%S')] swap: $(sysctl -n vm.swapusage)"
} >> "$LOG" 2>&1
