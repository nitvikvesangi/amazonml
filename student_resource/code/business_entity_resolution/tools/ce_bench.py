#!/usr/bin/env python3
"""Throughput micro-benchmark for the CE on MPS: padding policy, batch, dtype.

Text is built exactly like training/validation/apply:
    name + ' | ' + address[:200]

    .venv-sem/bin/python3 tools/ce_bench.py --n 1500
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'src'))
from sem_probe import log   # noqa: E402

MODEL_DIR = '../../output/ce_model'
BASE_MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
TRAIN_JSONL = '../../cache/ce_train.jsonl'


def load_pairs(n):
    A, B = [], []
    with open(TRAIN_JSONL) as f:
        for line in f:
            if len(A) >= n:
                break
            r = json.loads(line)
            A.append(r['n1'] + ' | ' + r['a1'][:200])
            B.append(r['n2'] + ' | ' + r['a2'][:200])
    return A, B


def score(tok, model, A, B, batch, max_len, dynamic, device):
    out = np.zeros(len(A), np.float32)
    t = time.time()
    with torch.inference_mode():
        for i in range(0, len(A), batch):
            enc = tok(A[i:i + batch], B[i:i + batch],
                      padding=(True if dynamic else 'max_length'),
                      truncation=True, max_length=max_len, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            out[i:i + batch] = torch.sigmoid(
                model(**enc).logits.squeeze(-1)).float().cpu().numpy()
    return len(A) / max(time.time() - t, 1e-9), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=1500)
    ap.add_argument('--model-dir', default=MODEL_DIR)
    args = ap.parse_args()

    A, B = load_pairs(args.n)
    log(f"bench pairs: {len(A):,} (avg chars "
        f"{np.mean([len(a) for a in A]):.0f} / {np.mean([len(b) for b in B]):.0f})")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=1)
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'ce_state.pt'),
                                     map_location='cpu'))
    model.to(device).eval()

    cfgs = [
        ('baseline pad=maxlen b256 fp32', 256, 128, False, False),
        ('dynamic b256 fp32           ', 256, 128, True, False),
        ('dynamic b512 fp32           ', 512, 128, True, False),
        ('dynamic b512 fp16           ', 512, 128, True, True),
    ]
    base_out = None
    for name, batch, ml, dyn, half in cfgs:
        model.half() if half else model.float()
        try:
            rate, out = score(tok, model, A, B, batch, ml, dyn, device)
            extra = ''
            if half and base_out is not None:
                d = np.abs(out - base_out)
                flips = int(((base_out >= 0.7) != (out >= 0.7)).sum()) + \
                        int(((base_out < 0.2) != (out < 0.2)).sum())
                extra = (f"  | vs fp32: max|d|={d.max():.4f} mean={d.mean():.5f} "
                         f"decision flips={flips}/{len(out)}")
            if not dyn and not half:
                base_out = out
            log(f"  {name}: {rate:,.0f} pairs/s{extra}")
        except Exception as exc:                        # noqa: BLE001
            log(f"  {name}: FAILED ({type(exc).__name__}: {exc})")


if __name__ == '__main__':
    main()
