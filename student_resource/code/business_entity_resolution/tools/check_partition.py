"""Is the ground truth a PARTITION?  i.e. does every S2/S3 record belong to at
most one S1 entity?

If yes, we get a very strong precision constraint at inference time: when two
different S1 entities both claim the same S2/S3 record, at most one of them can
be right.  Under a precision-heavy macro-F0.5 that is a huge lever.

Run:  python3 tools/check_partition.py
"""
import os
import sys
import collections

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "..", "..", "dataset", "train")


def main():
    gt = pd.read_csv(
        os.path.join(DATA, "train_ground_truth.tsv"),
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )
    print("gt rows:", len(gt), "cols:", list(gt.columns))

    s1_ids = gt.iloc[:, 0].to_numpy()
    lists = gt.iloc[:, 1].to_numpy()

    n_empty = 0
    claim = collections.Counter()
    n_pairs = 0
    per_s1 = np.zeros(len(gt), dtype=np.int32)
    for i, s in enumerate(lists):
        s = s.strip()
        if not s:
            n_empty += 1
            continue
        ids = s.split(",")
        per_s1[i] = len(ids)
        n_pairs += len(ids)
        for x in ids:
            claim[x] += 1

    print("S1 rows               :", len(gt))
    print("singletons (empty gt) :", n_empty, "(%.2f%%)" % (100.0 * n_empty / len(gt)))
    print("true pairs            :", n_pairs)
    print("distinct S2/S3 ids    :", len(claim))
    print("pairs per S1: mean %.3f  max %d" % (n_pairs / max(1, len(gt) - n_empty), per_s1.max()))

    counts = np.array(list(claim.values()))
    print("\n--- claims per S2/S3 record ---")
    for k in range(1, 6):
        print("  claimed by %d S1 entity(ies): %d" % (k, int((counts == k).sum())))
    print("  claimed by >=2            :", int((counts >= 2).sum()),
          "(%.4f%%)" % (100.0 * (counts >= 2).sum() / max(1, len(counts))))
    print("  claimed by >=3            :", int((counts >= 3).sum()))
    print("  MAX claims                :", int(counts.max()) if len(counts) else 0)

    # how much of S2 / S3 is covered at all
    for src, fn in (("S2", "train_source2.tsv"), ("S3", "train_source3.tsv")):
        ids = pd.read_csv(os.path.join(DATA, fn), sep="\t", dtype=str,
                          keep_default_na=False, usecols=["entity_id"]).iloc[:, 0]
        tot = len(ids)
        hit = sum(1 for x in ids if x in claim)
        print("\n%s records: %d   in a true pair: %d (%.2f%%)   unmatched: %d (%.2f%%)"
              % (src, tot, hit, 100.0 * hit / tot, tot - hit, 100.0 * (tot - hit) / tot))

    # distribution of the S1 fan-out (how many clusters an S1 belongs to) - should be 1 by construction
    print("\n--- does an S1 entity appear once? ---")
    print("  duplicate S1 rows:", int(len(s1_ids) - len(set(s1_ids))))


if __name__ == "__main__":
    main()
