#!/usr/bin/env python3
"""Fine-tune a multilingual cross-encoder on the mined hard pairs.

Task: the 41-feature GBDT cannot separate a band of pairs (true/false overlap in
score space).  A cross-encoder reads the raw strings and can, in principle, see
the token-level evidence the summary features miss.  Training data comes from
tools/ce_build_data.py (hard positives/negatives near the decision threshold +
calibration anchors), split by source1 record so validation is honest.

MPS device, fp32, manual loop (no Trainer so a timeboxed kill always leaves the
best checkpoint on disk).  Saves the *best-by-val-AUC* state dict.

Run from code/business_entity_resolution:
    python3 tools/ce_train.py --data ../../cache/ce_train.jsonl \
        --out ../../output/ce_model --timebox 3300
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sem_probe import auc, log   # noqa: E402

MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def texts(rows):
    return ([r['n1'] + ' | ' + r['a1'] for r in rows],
            [r['n2'] + ' | ' + r['a2'] for r in rows])


def valid_of(grp, mod=20):
    """5% of records (stable hash) -> validation."""
    return zlib.crc32(str(grp).encode()) % mod == 0


def pr_grid(y, s, grid):
    out = []
    for t in grid:
        h = s >= t
        tp = int((h & (y == 1)).sum())
        fp = int((h & (y == 0)).sum())
        fn = int((~h & (y == 1)).sum())
        out.append(dict(thr=round(float(t), 3), p=round(tp / max(tp + fp, 1), 4),
                        r=round(tp / max(tp + fn, 1), 4), n=int(h.sum())))
    return out


@torch.inference_mode()
def evaluate(model, tok, rows, device, max_len, batch=256, cap=20_000):
    model.eval()
    if len(rows) > cap:
        idx = np.random.default_rng(3).choice(len(rows), cap, replace=False)
        rows = [rows[i] for i in idx]
    scores = []
    for i in range(0, len(rows), batch):
        A, B = texts(rows[i:i + batch])
        enc = tok(A, B, padding='max_length', truncation=True,
                  max_length=max_len, return_tensors='pt')
        enc = {k: v.to(device) for k, v in enc.items()}
        logits = model(**enc).logits.squeeze(-1)
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
    s = np.concatenate(scores) if scores else np.zeros(0, np.float32)
    y = np.array([r['y'] for r in rows], np.int8)
    return y, s.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='../../cache/ce_train.jsonl')
    ap.add_argument('--out', default='../../output/ce_model')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=96)
    ap.add_argument('--max-len', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup-frac', type=float, default=0.08)
    ap.add_argument('--timebox', type=int, default=3600)
    ap.add_argument('--eval-every', type=int, default=300)
    ap.add_argument('--export-every', type=int, default=900)
    ap.add_argument('--freeze-layers', type=int, default=0,
                    help='freeze embeddings + first N encoder layers')
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

    rows = load_rows(args.data)
    train_rows = [r for r in rows if not valid_of(r['grp'])]
    val_rows = [r for r in rows if valid_of(r['grp'])]
    npos = sum(r['y'] for r in train_rows)
    log(f"data {args.data}: {len(rows):,} rows -> train {len(train_rows):,} "
        f"(pos {npos:,}) / val {len(val_rows):,} | device={device}")

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        # MPS SDPA has no dropout kernel, so train with the eager attention
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=1, attn_implementation='eager')
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=1)
    if args.freeze_layers:
        inner = getattr(model, 'bert', None)
        if inner is None:
            inner = getattr(model, 'model', None)
        if inner is None or not hasattr(inner, 'encoder'):
            log('freeze: no BERT-style inner module found - skipping')
        else:
            for p in inner.embeddings.parameters():
                p.requires_grad = False
            for i, layer in enumerate(inner.encoder.layer):
                if i < args.freeze_layers:
                    for p in layer.parameters():
                        p.requires_grad = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.to(device)
    log(f"model {os.path.basename(args.model.rstrip('/'))} loaded "
        f"({sum(p.numel() for p in model.parameters())/1e6:.1f}M params, "
        f"{n_train/1e6:.1f}M trainable, freeze={args.freeze_layers})")

    steps_per_epoch = max(1, len(train_rows) // args.batch)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(10, int(total_steps * args.warmup_frac))

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup))

    os.makedirs(args.out, exist_ok=True)
    best = dict(auc=-1.0, step=0, epoch=0)
    t0 = time.time()
    run_loss, run_n, run_t = 0.0, 0, time.time()
    gstep = 0
    stop = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_rows))
        for i in range(0, len(order) - args.batch + 1, args.batch):
            if time.time() - t0 > args.timebox:
                log(f"timebox {args.timebox}s hit at step {gstep}")
                stop = True
                break
            batch = [train_rows[j] for j in order[i:i + args.batch]]
            A, B = texts(batch)
            enc = tok(A, B, padding='max_length', truncation=True,
                      max_length=args.max_len, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            yy = torch.tensor([r['y'] for r in batch], dtype=torch.float32,
                              device=device)
            logits = model(**enc).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, yy)
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            run_loss += float(loss.detach())
            run_n += 1
            gstep += 1

            if gstep in (1, 5, 20) or gstep % 100 == 0:
                dt = time.time() - run_t
                log(f"  step {gstep} ep{epoch} loss={run_loss/max(run_n,1):.4f} "
                    f"({args.batch*max(run_n,1)/max(dt,1e-9):.0f} pairs/s) "
                    f"lr={sched.get_last_lr()[0]:.2e}")
                run_loss, run_n, run_t = 0.0, 0, time.time()

            if gstep % args.eval_every == 0 or gstep == total_steps:
                y, s = evaluate(model, tok, val_rows, device, args.max_len)
                a = auc(y, s)
                log(f"  eval step {gstep}: val n={len(y):,} AUC={a:.4f} | "
                    + ' '.join(f"t={d['thr']}:P={d['p']:.3f},R={d['r']:.3f}"
                               for d in pr_grid(y, s, (0.5, 0.9, 0.95, 0.98))))
                if a > best['auc']:
                    best = dict(auc=float(a), step=gstep, epoch=epoch)
                    torch.save({k: v.detach().cpu()
                                for k, v in model.state_dict().items()},
                               os.path.join(args.out, 'ce_state.pt'))
                    tok.save_pretrained(args.out)
                    log(f"  saved best (AUC {a:.4f}) -> {args.out}/ce_state.pt")
        if stop:
            break

    rep = dict(model=args.model, rows=len(rows), train=len(train_rows),
               val=len(val_rows), epochs=args.epochs, batch=args.batch,
               max_len=args.max_len, lr=args.lr, device=device,
               steps=gstep, seconds=round(time.time() - t0, 1), best=best,
               finished=not stop)
    with open(os.path.join(args.out, 'train_report.json'), 'w') as f:
        json.dump(rep, f, indent=1)
    log(f"done: {gstep} steps in {(time.time()-t0)/60:.1f} min, best AUC "
        f"{best['auc']:.4f} @ step {best['step']} -> {args.out}")


if __name__ == '__main__':
    main()
