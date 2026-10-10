#!/usr/bin/env python3
"""Human reliability ceiling for rho_H on the 1,200-item GoEmotions subset (split-half, Spearman-Brown).

For each of N_SPLITS random splits, the annotators of every item are randomly permuted and split into
two halves (the first floor(n/2) annotators vs. the rest). Each half gives a soft label per label and
an item-level mean binary entropy; the split-half reliability is the Spearman correlation of the two
entropies across items. The ceiling is the Spearman-Brown correction of the mean split-half
correlation, r_SB = 2 r / (1 + r).

Annotator rows are read from the raw GoEmotions files goemotions_{1,2,3}.csv in file order (every
annotator row counts, including rows with no label selected).

Usage:
  python3 analysis/human_reliability_ceiling.py --raw_dir data/goemotions \
      --reference data/goe_reference_1200.npz --out output/human_reliability_ceiling.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

N_SPLITS = 1000
SEED = 20260925
EPS = 1e-12


def mean_binary_entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).mean(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--reference", required=True, help="goe_reference_1200.npz (defines items and their order)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    emotions = [l.strip() for l in open(raw / "emotions.txt") if l.strip()]
    ref = np.load(args.reference, allow_pickle=True)
    ids = [str(x) for x in ref["ids"]]
    wanted = set(ids)
    ann = {}
    for i in (1, 2, 3):
        with open(raw / f"goemotions_{i}.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["id"] in wanted:
                    ann.setdefault(row["id"], []).append([1 if row[k] == "1" else 0 for k in emotions])
    A = [np.array(ann[x], dtype=float) for x in ids]
    dev = float(np.abs(np.stack([a.mean(0) for a in A]) - ref["p_human"]).max())
    print(f"items: {len(A)}; max |p_human(raw) - p_human(reference)| = {dev:.2e}")

    rng = np.random.RandomState(SEED)
    rs = []
    for _ in range(N_SPLITS):
        h1, h2 = [], []
        for a in A:
            n = len(a)
            perm = rng.permutation(n)
            k = n // 2
            h1.append(a[perm[:k]].mean(0))
            h2.append(a[perm[k:]].mean(0))
        rs.append(spearmanr(mean_binary_entropy(np.array(h1)), mean_binary_entropy(np.array(h2))).correlation)
    r = float(np.mean(rs))
    out = {
        "split_half_rho_mean": r,
        "split_half_rho_std": float(np.std(rs, ddof=1)),
        "spearman_brown_ceiling": 2 * r / (1 + r),
        "n_splits": N_SPLITS,
        "seed": SEED,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
