#!/bin/bash
# One-shot status digest for the 27 Sep endgame.  Cheap to run any time:
#   bash tools/status.sh
set -u
cd "$(dirname "$0")/.."
D=../../output

echo "== $(date '+%a %d %b %H:%M:%S %Z') =="
echo "swap: $(sysctl -n vm.swapusage)"
echo "disk: $(df -h /System/Volumes/Data | awk 'NR==2 {print $4 " free"}')"
echo "----------------------------------------------------------------"
echo "== processes =="
pgrep -fl 'pipeline_fast|post500k|chain1m|verify_ship|mem_guard' | sed 's/^/  /' || echo "  (none)"
echo "----------------------------------------------------------------"
echo "== 500K predict shards =="
for s in C D; do
  f="logs/pred500k_$s.log"
  if [ -f "$f" ]; then
    last=$(grep 'records of this range' "$f" | tail -1)
    out="$D/pred500k_$s/matching_results.tsv"
    sz=$(du -sh "$D/pred500k_$s" 2>/dev/null | cut -f1)
    mt=$(stat -f '%Sm' "$out" 2>/dev/null || echo '-')
    if [ -z "$last" ]; then
      echo "  shard $s: no progress line (see pipeline quirk) — out $sz, last write $mt"
    else
      echo "  shard $s: $last"
    fi
  fi
done
echo "----------------------------------------------------------------"
echo "== chained jobs (last line each) =="
for f in logs/post500k.log logs/chain1m.log logs/train1m_base41.log logs/verify_ship.log logs/mem_guard_1m.log; do
  [ -f "$f" ] && echo "  $(basename "$f"): $(tail -1 "$f")"
done
echo "----------------------------------------------------------------"
echo "== ship / candidate files =="
for f in "$D/ship500k_fr097.tsv" "$D/ship500k_t094.tsv" "$D/ship500k_t097.tsv" \
         "$D/ship1m_fr097.tsv" "$D/ship1m_t094.tsv" \
         "$D/final/matching_results_fr097.tsv" "$D/final/matching_results.tsv" \
         "$D/final/matching_results.tsv"; do
  if [ -f "$f" ]; then
    printf "  %-42s %6s  rows=%-9s  %s\n" "$(basename "$f")" \
      "$(du -h "$f" | cut -f1)" "$(wc -l < "$f" | tr -d ' ')" "$(stat -f '%Sm' "$f")"
  fi
done
echo "----------------------------------------------------------------"
echo "== upload decision state =="
if grep -q '1M passed' logs/chain1m.log 2>/dev/null; then
  echo "  1M GATE PASSED — 1M predicts running/done; final file = ship1m_fr097.tsv"
elif grep -q 'below +0.002' logs/chain1m.log 2>/dev/null; then
  echo "  1M GATE FAILED — final file = ship500k_fr097.tsv"
elif grep -q 'no 1M model saved' logs/chain1m.log 2>/dev/null; then
  echo "  1M TRAIN DIED — final file = ship500k_fr097.tsv"
else
  echo "  1M gate verdict not in yet (waiting; check logs/chain1m.log)"
fi
if [ -f logs/verify_ship.log ]; then
  v=$(grep -c 'VALIDATOR PASS' logs/verify_ship.log 2>/dev/null || true)
  echo "  verify_ship: $(tail -1 logs/verify_ship.log)"
fi
