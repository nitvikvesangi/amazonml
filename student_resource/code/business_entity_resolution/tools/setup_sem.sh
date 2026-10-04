#!/bin/bash
# Semantic-layer prototype toolchain: venv (system site packages so pandas/numpy
# from brew are visible), torch + sentence-transformers, and a pre-fetched
# multilingual MiniLM so the measurement run needs no network.
#
#   python3 tools/detach.py logs/sem_setup.log tools/setup_sem.sh
set -u
cd "$(dirname "$0")/.."
LOG=logs/sem_setup.log
PY=/opt/homebrew/bin/python3
VENV=.venv-sem

{
  echo "[$(date '+%H:%M:%S')] setup start (python: $($PY -V 2>&1))"
  if [ ! -x "$VENV/bin/python" ]; then
    $PY -m venv --system-site-packages "$VENV" || { echo "venv failed"; exit 1; }
  fi
  "$VENV/bin/python" -m pip install --upgrade pip 2>&1 | tail -2
  echo "[$(date '+%H:%M:%S')] installing torch + sentence-transformers (this is the big download)"
  "$VENV/bin/python" -m pip install torch sentence-transformers 2>&1 | tail -6
  rc=$?
  echo "[$(date '+%H:%M:%S')] pip rc=$rc"
  "$VENV/bin/python" - <<'EOF'
import torch, sentence_transformers, numpy, pandas  # noqa
print('imports ok: torch', torch.__version__, '| st',
      sentence_transformers.__version__, '| mps', torch.backends.mps.is_available())
from sentence_transformers import SentenceTransformer
m = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
e = m.encode(['ram marketing private limited',
              'rama maarketinga priyaveta limited',
              'shree lakshmi enterprises'],
             batch_size=8, show_progress_bar=False)
print('model ok, dim', e.shape, 'cos[0,1]=%.3f cos[0,2]=%.3f' % (
    float((e[0] @ e[1]) / ((e[0] ** 2).sum() ** 0.5 * (e[1] ** 2).sum() ** 0.5)),
    float((e[0] @ e[2]) / ((e[0] ** 2).sum() ** 0.5 * (e[2] ** 2).sum() ** 0.5))))
EOF
  echo "[$(date '+%H:%M:%S')] setup done rc=$?"
} >> "$LOG" 2>&1
