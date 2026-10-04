"""Check for cross-split leakage between train and test.

Top-of-leaderboard macro-F0.5 (0.987) implies near-exact cluster recovery, which
is suspiciously high for a noisy ER task.  The cheapest way to get there is if
test records/businesses are shared with, or derivable from, the training set.
This script measures every cheap leak we can think of, in streaming passes so
peak RAM stays low.

Usage: python3 tools/leak_check.py [--dataset ../../dataset]
"""
import argparse
import hashlib
import os
import re
import sys


def h(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big")


def num(entity_id):
    """S2-166376419 -> 166376419 (int), None if not parseable."""
    m = re.search(r"(\d+)\s*$", entity_id)
    return int(m.group(1)) if m else None


NORM_RE = re.compile(r"[^a-z0-9]+")


def norm(s):
    return NORM_RE.sub(" ", s.lower()).strip()


def stream(path, cols=(0, 1, 2)):
    with open(path, "r", encoding="utf-8", newline="") as fh:
        fh.readline()
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            yield tuple(parts[c] if c < len(parts) else "" for c in cols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../../dataset")
    args = ap.parse_args()
    d = args.dataset

    train = {s: os.path.join(d, "train", "train_source%d.tsv" % s) for s in (1, 2, 3)}
    test = {s: os.path.join(d, "test", "test_source%d.tsv" % s) for s in (1, 2, 3)}
    gt = os.path.join(d, "train", "train_ground_truth.tsv")

    for s in (1, 2, 3):
        print("checking source %d ..." % s, flush=True)
        t_ids = set()
        t_nameaddr = set()
        for eid, name, addr in stream(train[s]):
            n = num(eid)
            if n is not None:
                t_ids.add(n)
            if s == 1:
                t_nameaddr.add(h(norm(name) + "|" + norm(addr)))
        print("  train S%d: %d ids, %d name+addr" % (s, len(t_ids), len(t_nameaddr) if s == 1 else 0))

        n_rows = id_hit = na_hit = 0
        samples = []
        for eid, name, addr in stream(test[s]):
            n_rows += 1
            n = num(eid)
            if n is not None and n in t_ids:
                id_hit += 1
                if len(samples) < 5:
                    samples.append((eid, name, addr))
            if s == 1 and h(norm(name) + "|" + norm(addr)) in t_nameaddr:
                na_hit += 1
        print("  test S%d rows=%d  ID OVERLAP=%d  NAME+ADDR OVERLAP=%d"
              % (s, n_rows, id_hit, na_hit))
        for x in samples:
            print("    sample shared id:", x)
        del t_ids, t_nameaddr

    # Direct id transfer check: does any test S1 id appear in train gt?
    print("\nchecking test S1 <-> train ground truth id transfer ...", flush=True)
    gt_ids = set()
    with open(gt, "r", encoding="utf-8", newline="") as fh:
        fh.readline()
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            a, _, b = line.partition("\t")
            n = num(a)
            if n is not None:
                gt_ids.add(n)
    hit = 0
    with open(test[1], "r", encoding="utf-8", newline="") as fh:
        fh.readline()
        for line in fh:
            n = num(line.split("\t", 1)[0])
            if n is not None and n in gt_ids:
                hit += 1
    print("  test S1 ids present in train gt keys: %d" % hit)


if __name__ == "__main__":
    main()
