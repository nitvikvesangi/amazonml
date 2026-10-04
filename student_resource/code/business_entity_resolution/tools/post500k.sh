#!/bin/bash
# After the two 500K predict shards exit: build the ship variants from their
# scores (LB-validated rule: France 0.97, US+India 0.94) and print the France
# sanity numbers.  Light job - safe next to the 1M training that chains behind it.
#
#   python3 tools/detach.py logs/post500k.log tools/post500k.sh
set -u
cd "$(dirname "$0")/.."
LOG=logs/post500k.log
PY=/opt/homebrew/bin/python3
D=../../output

{
  echo "[$(date '+%H:%M:%S')] waiting for the 500K predict shards"
  for _ in $(seq 1 400); do
    pgrep -f 'pipeline_fast.py predict' >/dev/null || break
    sleep 30
  done
  echo "[$(date '+%H:%M:%S')] predicts gone"
  ls -la $D/pred500k_C/scores.tsv $D/pred500k_D/scores.tsv 2>&1

  echo "[$(date '+%H:%M:%S')] building variants (France 0.97 / all 0.97 / all 0.94)"
  nice -n 10 $PY tools/threshold_apply.py --scores $D/pred500k_C/scores.tsv $D/pred500k_D/scores.tsv \
      --threshold 0.94 --country-thr France=0.97 \
      --out $D/ship500k_fr097.tsv --report $D/ship500k_fr097_report.json
  nice -n 10 $PY tools/threshold_apply.py --scores $D/pred500k_C/scores.tsv $D/pred500k_D/scores.tsv \
      --threshold 0.97 --out $D/ship500k_t097.tsv --report $D/ship500k_t097_report.json
  nice -n 10 $PY tools/threshold_apply.py --scores $D/pred500k_C/scores.tsv $D/pred500k_D/scores.tsv \
      --threshold 0.94 --out $D/ship500k_t094.tsv --report $D/ship500k_t094_report.json

  echo "[$(date '+%H:%M:%S')] own-leak-clean holdout numbers for the 500K model:"
  nice -n 10 $PY tools/compare_dumps.py \
      --dump ../../output/eval41a30_fixed/pairs_base41_250k.npz:champion250k \
      --dump ../../output/eval41a30_fixed/pairs_base41_500k.npz:model500k \
      --clean-train 250000 --clean-train 500000

  echo "[$(date '+%H:%M:%S')] per-country shape of the 500K variants vs the champion ship:"
  $PY - <<'EOF'
import pandas as pd
s1 = pd.read_csv('../../dataset/test/test_source1.tsv', sep='\t', usecols=['entity_id', 'country'])
for tag, path in (('champion fr097', '../../output/final/matching_results_fr097.tsv'),
                  ('500k fr097', '../../output/ship500k_fr097.tsv'),
                  ('500k all0.94', '../../output/ship500k_t094.tsv'),
                  ('500k all0.97', '../../output/ship500k_t097.tsv')):
    try:
        m = pd.read_csv(path, sep='\t', dtype=str)
    except FileNotFoundError:
        print(f'  {tag}: MISSING'); continue
    df = s1.merge(pd.DataFrame({'entity_id': m['source1_entity_id'],
                                'ids': m['matched_entity_ids'].fillna('')}), on='entity_id')
    n = df['ids'].map(lambda s: 0 if s == '' else s.count(',') + 1)
    df = df.assign(n=n)
    g = df.groupby('country').agg(rows=('n', 'size'), matched=('n', lambda s: (s > 0).sum()),
                                  ids=('n', 'sum'))
    g['mr'] = (g.matched / g.rows).round(3)
    g['ipr'] = (g.ids / g.matched).round(2)
    tot = f"total ids {int(g.ids.sum()):,}"
    print(f'  {tag}: ' + ' | '.join(f"{c}: mr={r.mr} ipr={r.ipr}" for c, r in g.iterrows()) + f'  ({tot})')
EOF
  echo "[$(date '+%H:%M:%S')] post500k done"
} >> "$LOG" 2>&1
