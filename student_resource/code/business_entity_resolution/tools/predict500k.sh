#!/bin/bash
# If the 500K gate clears (clean delta >= +0.002), predict the whole test with
# the 500K model — two halves in parallel — and build the candidate ship files
# from its saved scores, so the final upload decision is a 27 s re-threshold.
#
#   python3 tools/detach.py logs/predict500k_chain.log tools/predict500k.sh
set -u
cd "$(dirname "$0")/.."
LOG=logs/predict500k_chain.log
PY=/opt/homebrew/bin/python3
D=../../output

{
  echo "[$(date '+%H:%M:%S')] waiting for the gate verdict (logs/gate500k.log)"
  for _ in $(seq 1 240); do          # up to 2 h; match the delta line, NOT the
    grep -q 'own-threshold' logs/gate500k.log && break   # verdict line that also says DELTA
    sleep 30
  done
  line=$(grep -m1 'own-threshold' logs/gate500k.log)
  echo "[$(date '+%H:%M:%S')] gate line: ${line:-<none>}"
  own=$(printf '%s' "$line" | sed -n 's/.*own-threshold \([+-][0-9.]*\).*/\1/p')
  best=$(printf '%s' "$line" | sed -n 's/.*best-on-grid \([+-][0-9.]*\).*/\1/p')
  echo "parsed: own=$own best=$best"
  ok=$($PY -c "print(1 if max(float('${own:-0}'), float('${best:-0}')) >= 0.002 else 0)")
  if [ "$ok" != "1" ]; then
    echo "[$(date '+%H:%M:%S')] gate below +0.002 - no 500K predict, the champion ships"
    exit 0
  fi
  echo "[$(date '+%H:%M:%S')] gate passed - predicting the whole test with the 500K model"
  nice -n 10 $PY tools/detach.py logs/pred500k_C.log nice -n 10 $PY -u \
      src/pipeline_fast.py predict --chunk-records 4000 --cand-cap 200 \
      --sim-cap-alpha 30 --model $D/model_base41_500k/model_fast.pkl \
      --start-records 0 --max-records 866272 \
      --out-dir $D/pred500k_C --scores $D/pred500k_C/scores.tsv
  nice -n 10 $PY tools/detach.py logs/pred500k_D.log nice -n 10 $PY -u \
      src/pipeline_fast.py predict --chunk-records 4000 --cand-cap 200 \
      --sim-cap-alpha 30 --model $D/model_base41_500k/model_fast.pkl \
      --start-records 866272 --max-records 1732544 \
      --out-dir $D/pred500k_D --scores $D/pred500k_D/scores.tsv
  for _ in $(seq 1 420); do          # up to 3.5 h for both halves
    pgrep -f 'pipeline_fast.py predict' >/dev/null || break
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] predicts done; building ship variants from the 500K scores"
  nice -n 10 $PY tools/threshold_apply.py \
      --scores $D/pred500k_C/scores.tsv $D/pred500k_D/scores.tsv \
      --threshold 0.94 --country-thr France=0.97 \
      --out $D/ship500k_fr097.tsv --report $D/ship500k_fr097_report.json
  nice -n 10 $PY tools/threshold_apply.py \
      --scores $D/pred500k_C/scores.tsv $D/pred500k_D/scores.tsv \
      --threshold 0.97 --out $D/ship500k_t097.tsv --report $D/ship500k_t097_report.json
  nice -n 10 $PY tools/threshold_apply.py \
      --scores $D/pred500k_C/scores.tsv $D/pred500k_D/scores.tsv \
      --threshold 0.94 --out $D/ship500k_t094.tsv --report $D/ship500k_t094_report.json
  echo "[$(date '+%H:%M:%S')] France sanity on the 500K scores (champion at fr097 was 3.47 ids/row, 860,374 ids):"
  $PY - <<'EOF'
import pandas as pd
s1 = pd.read_csv('../../dataset/test/test_source1.tsv', sep='\t', usecols=['entity_id', 'country'])
fr = set(s1.loc[s1.country == 'France', 'entity_id'])
for tag in ('ship500k_fr097', 'ship500k_t097', 'ship500k_t094'):
    m = pd.read_csv(f'../../output/{tag}.tsv', sep='\t', dtype=str)
    sub = m[m.source1_entity_id.isin(fr)]['matched_entity_ids'].fillna('')
    n = sub.map(lambda s: 0 if s == '' else s.count(',') + 1)
    print(f"  {tag}: France matched={int((n>0).sum()):,} ids={int(n.sum()):,} "
          f"ids/row={n.sum()/max(int((n>0).sum()),1):.2f}")
EOF
  echo "[$(date '+%H:%M:%S')] chain done - final ship decision is the agent's (read this log)"
  echo "[$(date '+%H:%M:%S')] swap: $(sysctl -n vm.swapusage)"
} >> "$LOG" 2>&1
