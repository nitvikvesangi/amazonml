#!/usr/bin/env bash
# Deploy this pipeline to a fresh Ubuntu EC2 box and run jobs there.
#
#   tools/aws_deploy.sh <public-ip> <key.pem> probe
#   tools/aws_deploy.sh <public-ip> <key.pem> setup          # venv + pip deps
#   tools/aws_deploy.sh <public-ip> <key.pem> stage1         # code + train-side data (~3.5 GB)
#   tools/aws_deploy.sh <public-ip> <key.pem> stage2         # test-side data (~3.5 GB)
#   tools/aws_deploy.sh <public-ip> <key.pem> run <tag> <pipeline args...>
#   tools/aws_deploy.sh <public-ip> <key.pem> status
#   tools/aws_deploy.sh <public-ip> <key.pem> tail <tag> [lines]
#   tools/aws_deploy.sh <public-ip> <key.pem> fetch <remote-rel-path> [local-rel-path]
#   tools/aws_deploy.sh <public-ip> <key.pem> kill <pid>
#
# The box mirrors the Mac layout (~/mlchallenge/{code,dataset,cache,output}), so
# every command in the runbook works unchanged after step 4 of HANDOFF.md.
#
#   stage1 is enough to start training; stage2 adds what `predict` needs.
#   Everything is rsync, so a dropped connection can just be re-run.
set -euo pipefail

IP=${1:?usage: aws_deploy.sh <public-ip> <key.pem> <mode> [args...]}
PEM=${2:?usage: aws_deploy.sh <public-ip> <key.pem> <mode> [args...]}
MODE=${3:?usage: aws_deploy.sh <public-ip> <key.pem> <mode> [args...]}
shift 3 || true

AWS_USER=${AWS_USER:-ubuntu}
SSH_OPTS=(-i "$PEM" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30
          -o ServerAliveCountMax=6)
REMOTE="$AWS_USER@$IP"
R_BASE="/home/$AWS_USER/mlchallenge"
PY="/home/$AWS_USER/venv/bin/python"
L_BASE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)   # student_resource
CODE_DIR="$L_BASE/code/business_entity_resolution"
RSYNC_OPTS=(-az --partial --info=progress2 -e "ssh ${SSH_OPTS[*]}")

say() { printf '\n=== %s ===\n' "$*"; }

case "$MODE" in
probe)
  say "remote probe"
  ssh "${SSH_OPTS[@]}" "$REMOTE" 'uname -srm; nproc; free -g | head -2; df -h / | tail -1; python3 -V'
  ;;

setup)
  say "apt + venv + pip (needs sudo on the box)"
  ssh "${SSH_OPTS[@]}" "$REMOTE" 'set -e
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-venv python3-pip rsync tmux htop
    python3 -m venv ~/venv
    ~/venv/bin/pip -q install --upgrade pip'
  say "pip deps"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "$PY -m pip -q install pandas numpy rapidfuzz lightgbm scikit-learn unidecode && $PY -c 'import pandas, numpy, rapidfuzz, lightgbm, sklearn, unidecode; print(\"deps ok\", pandas.__version__, numpy.__version__, rapidfuzz.__version__, lightgbm.__version__)'"
  ;;

stage1)
  say "mkdirs"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "mkdir -p $R_BASE/code $R_BASE/dataset/train $R_BASE/cache $R_BASE/output"
  say "code"
  rsync "${RSYNC_OPTS[@]}" --exclude '__pycache__' --exclude 'logs/*.log' \
        "$CODE_DIR/" "$REMOTE:$R_BASE/code/business_entity_resolution/"
  say "train raw data"
  rsync "${RSYNC_OPTS[@]}" "$L_BASE/dataset/train/" "$REMOTE:$R_BASE/dataset/train/"
  say "gt + train-side cache (norm + keys)"
  rsync "${RSYNC_OPTS[@]}" \
        "$L_BASE/cache/gt_train.pkl" "$L_BASE/cache/country_train.pkl" \
        "$L_BASE/cache/norm_train_source1.pkl" "$L_BASE/cache/norm_train_source2.pkl" \
        "$L_BASE/cache/norm_train_source3.pkl" "$L_BASE/cache/keys_train.pkl" \
        "$L_BASE/cache/tune_blocking.json" "$REMOTE:$R_BASE/cache/"
  say "stage1 done — training can start"
  ;;

stage2)
  say "test raw data"
  rsync "${RSYNC_OPTS[@]}" "$L_BASE/dataset/test/" "$REMOTE:$R_BASE/dataset/test/"
  say "test-side cache (norm + keys)"
  rsync "${RSYNC_OPTS[@]}" \
        "$L_BASE/cache/norm_test_source1.pkl" "$L_BASE/cache/norm_test_source2.pkl" \
        "$L_BASE/cache/norm_test_source3.pkl" "$L_BASE/cache/keys_test.pkl" \
        "$REMOTE:$R_BASE/cache/"
  say "stage2 done — predict can start"
  ;;

run)
  TAG=${1:?run needs a tag}; shift || true
  ARGS=${*:?run needs pipeline args, e.g. train --sample 400000 ...}
  say "launching '$TAG' detached"
  ssh "${SSH_OPTS[@]}" "$REMOTE" "cd $R_BASE/code/business_entity_resolution && mkdir -p logs && \
    setsid nohup $PY -u src/pipeline_fast.py $ARGS > logs/$TAG.log 2>&1 < /dev/null & \
    sleep 2; echo \"launched: pid \$! -> logs/$TAG.log\"; tail -n 3 logs/$TAG.log || true"
  ;;

status)
  ssh "${SSH_OPTS[@]}" "$REMOTE" 'uptime; echo; ps -eo pid,etime,pcpu,pmem:8,args --sort=-pcpu | grep -E "pipeline_fast" | grep -v grep | cut -c1-160; echo; free -g | head -2; df -h / | tail -1; echo; for f in ~/mlchallenge/code/business_entity_resolution/logs/*.log; do echo "--- $f"; tail -n 3 "$f"; done'
  ;;

tail)
  TAG=${1:?tail needs a tag}; N=${2:-10}
  ssh "${SSH_OPTS[@]}" "$REMOTE" "tail -n $N $R_BASE/code/business_entity_resolution/logs/$TAG.log"
  ;;

fetch)
  REL=${1:?fetch needs a path relative to ~/mlchallenge}
  DEST=${2:-$L_BASE/remote/$(basename "$REL")}
  mkdir -p "$(dirname "$DEST")"
  say "fetching $REL"
  rsync "${RSYNC_OPTS[@]}" "$REMOTE:$R_BASE/$REL" "$DEST"
  say "saved to $DEST"
  ;;

kill)
  PID=${1:?kill needs a pid}
  ssh "${SSH_OPTS[@]}" "$REMOTE" "kill $PID && echo killed $PID"
  ;;

*)
  echo "unknown mode: $MODE" >&2; exit 2 ;;
esac
