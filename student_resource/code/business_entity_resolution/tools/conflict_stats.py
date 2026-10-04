"""Measure structural conflicts in a submitted matching_results.tsv.

The train ground truth is a strict PARTITION: every S2/S3 record is claimed by
exactly one S1 entity (verified: 7,638,365 pairs, 0 conflicts).  So whenever our
final matches let two different S1 entities claim the same S2/S3 record, at
least one of those claims is a *guaranteed false merge* -- and macro-F0.5 is
precision-heavy, so each false merge is expensive.

Usage:
  python3 tools/conflict_stats.py --matching ../../output/matching_results.tsv
  python3 tools/conflict_stats.py --matching ../../output/matching_results.tsv \
      --gt ../../dataset/train/train_ground_truth.tsv      # train only
"""
import argparse
import collections
import os


def _open(path):
    return open(path, "r", encoding="utf-8", newline="")


def read_matching(path):
    """yield (s1_id, [ids])."""
    with _open(path) as fh:
        header = fh.readline()
        if not header:
            return
        start = 1
        if "source1" not in header and "entity" not in header:
            yield header.rstrip("\n").split("\t")[0], []
            start = 0
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            s1 = parts[0]
            ids = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            yield s1, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--top", type=int, default=8, help="show the worst records")
    ap.add_argument("--dump-conflicts", default=None,
                    help="write a TSV of rec_id<TAB>claimant1,claimant2,... (capped)")
    args = ap.parse_args()

    n_rows = n_empty = n_ids = 0
    claimants = collections.defaultdict(list)  # s23 id -> [s1 id, ...]
    for s1, ids in read_matching(args.matching):
        n_rows += 1
        if not ids:
            n_empty += 1
            continue
        n_ids += len(ids)
        for x in ids:
            claimants[x].append(s1)

    print("rows                 : %d" % n_rows)
    print("empty rows           : %d (%.2f%%)" % (n_empty, 100.0 * n_empty / max(1, n_rows)))
    print("matched ids (total)  : %d" % n_ids)
    print("distinct S2/S3 ids   : %d" % len(claimants))

    counts = collections.Counter(len(v) for v in claimants.values())
    print("\n--- how many S1 entities claim each S2/S3 record ---")
    for k in sorted(counts):
        if k <= 5 or k in (counts):
            print("  claimed by %-3d : %10d" % (k, counts[k]))
    multi = sum(c for k, c in counts.items() if k >= 2)
    extra = sum((k - 1) * c for k, c in counts.items() if k >= 2)
    print("  claimed by >=2 : %d (%.3f%% of distinct ids)" %
          (multi, 100.0 * multi / max(1, len(claimants))))
    print("  MAX claims     : %d" % max(counts) if counts else 0)

    # every extra claim beyond the best one is a guaranteed-invalid claim:
    # at most one S1 in each conflict group can own the record.
    # Upper bound on false merges = number of claims on multi-claimed records,
    # minus the ones that happen to be the true owner.
    print("\n--- false-merge opportunity ---")
    print("  claims sitting on contested records : %d" % sum(
        k * c for k, c in counts.items() if k >= 2))
    print("  claims that MUST be dropped         : >= %d" % extra)
    print("  (each dropped claim removes a false merge, but may cost recall)")

    if args.gt:
        gt_owner = {}
        with _open(args.gt) as fh:
            fh.readline()
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                a, _, b = line.partition("\t")
                if not b:
                    continue
                for x in b.split(","):
                    gt_owner[x] = a
        n_true_present = n_true_missing = n_true_dup = 0
        for rec, who in claimants.items():
            if len(who) < 2:
                continue
            true = gt_owner.get(rec)
            if true is None:
                n_true_missing += 1
            elif true in who:
                n_true_present += 1
            else:
                n_true_dup += 1
        print("\n--- among contested records (train only) ---")
        print("  true owner is one of the claimants  : %d" % n_true_present)
        print("  record is a true single (no owner)  : %d" % n_true_missing)
        print("  true owner NOT among claimants      : %d" % n_true_dup)
        print("  => keeping the best claim keeps the truth in %.1f%% of cases"
              % (100.0 * n_true_present / max(1, n_true_present + n_true_missing + n_true_dup)))

    if args.dump_conflicts:
        with open(args.dump_conflicts, "w", encoding="utf-8") as out:
            out.write("record_id\tclaimants\n")
            n = 0
            for rec, who in claimants.items():
                if len(who) >= 2:
                    out.write("%s\t%s\n" % (rec, ",".join(who)))
                    n += 1
                    if n >= 2_000_000:
                        break
        print("\nwrote conflicts -> %s (%d rows)" % (args.dump_conflicts, n))


if __name__ == "__main__":
    main()
