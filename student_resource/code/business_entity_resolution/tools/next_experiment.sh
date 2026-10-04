#!/bin/bash
# Chain the post-prediction work so the machine never idles.
#
#   1. wait for both predict shards to exit, then for the endgame merge to land
#   2. diagnose the true pairs the model rejects (are they lexically findable?)
#   3. resume the parked LambdaRank training (rank objective = the metric itself)
#   4. eval it on the canonical fixed holdout and compare with the champion
#
# The champion's honest holdout number is 0.9048 (the headline 0.9066 includes
# +0.0018 of holdout leakage: the 250K train sample shares 3,413 of the 30K eval
# records).  Any new model must clear 0.9048 on the CLEAN subset to count.
#
# Deliberately does NOT start a predict: that is a 3 h decision made afterwards.
#
#   python3 tools/detach.py logs/next_experiment.log tools/next_experiment.sh
set -u
cd "$(dirname "$0")/.."
LOG=logs/next_experiment.log
PY=/opt/homebrew/bin/python3
RANK_PID=19678

{
  echo "[$(date '+%H:%M:%S')] waiting for the predict shards to exit"
  while pgrep -f 'pipeline_fast.py predict' >/dev/null; do sleep 20; done
  echo "[$(date '+%H:%M:%S')] predicts gone; 60 s grace so swap can drain"
  sleep 60
  for _ in $(seq 1 60); do            # endgame merge + validator + probes
    [ -f ../../output/final/matching_results.tsv ] && break
    sleep 20
  done
  echo "[$(date '+%H:%M:%S')] merged submission present: " \
       "$(ls -la ../../output/final/matching_results.tsv 2>/dev/null | awk '{print $5" bytes"}')"
  echo "[$(date '+%H:%M:%S')] swap: $(sysctl -n vm.swapusage)"

  echo "[$(date '+%H:%M:%S')] (1/2) diagnostic: are the missed true pairs lexical?"
  nice -n 10 $PY -u tools/diag_missed.py \
      --dump ../../output/eval41a30_fixed/pairs_base41_250k.npz --sample 1500000

  echo "[$(date '+%H:%M:%S')] (2/2) resuming parked LambdaRank training (pid $RANK_PID)"
  if kill -CONT "$RANK_PID" 2>/dev/null; then
    # bounded wait: mem_guard parks the trainer again if swap blows past 30 GB,
    # in which case waiting forever would stall this chain
    for _ in $(seq 1 300); do
      kill -0 "$RANK_PID" 2>/dev/null || break
      sleep 30
    done
    echo "[$(date '+%H:%M:%S')] rank training exited (or wait expired)"
    tail -3 logs/train_lambdarank.log
    echo "[$(date '+%H:%M:%S')] eval rank model on the canonical holdout"
    nice -n 10 $PY -u src/pipeline_fast.py eval --sample 30000 \
        --sim-cap-alpha 30 --chunk-records 4000 --cand-cap 200 \
        --model ../../output/model_rank41_250k/model_fast.pkl \
        --dump-pairs ../../output/eval41a30_fixed/pairs_rank41_250k.npz \
        --out-dir ../../output/eval_rank41_250k
    echo "[$(date '+%H:%M:%S')] rank eval rc=$?  (champion clean holdout: 0.9048)"
    echo "[$(date '+%H:%M:%S')] postproc on the new dump:"
    nice -n 10 $PY -u tools/postproc.py \
        --dump ../../output/eval41a30_fixed/pairs_rank41_250k.npz --oracle \
        --thresholds 0.30,0.995,0.005 | tail -14
  else
    echo "[$(date '+%H:%M:%S')] pid $RANK_PID is gone - rank training lost, skipping"
  fi
} >> "$LOG" 2>&1
