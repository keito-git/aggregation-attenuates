#!/usr/bin/env python3
"""Build goe_test_prepared.json from the raw GoEmotions release (no LLM involved).

For every test example with at least three annotators, the per-label human soft label is the
fraction of annotators that selected the label (all annotator rows in goemotions_{1,2,3}.csv are
counted, including rows with no label selected). H_human is the mean binary entropy over the
28 labels (bits/label). Entropy strata use the quartile boundaries of H_human over the filtered
test set (numpy.quantile, linear interpolation):
  stratum 1: H <= Q25, 2: Q25 < H <= Q50, 3: Q50 < H <= Q75, 4: H > Q75
(Q25 == Q50 on this data, so stratum 2 is empty.)

Usage:
  python3 data/prepare_goemotions.py --raw_dir data/goemotions --out data/goe_test_prepared.json
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

MIN_ANNOTATORS = 3


def binary_entropy(p):
    out = 0.0
    for q in (p, 1.0 - p):
        if q > 0.0:
            out -= q * np.log2(q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", required=True,
                    help="directory with emotions.txt, test.tsv and goemotions_{1,2,3}.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    raw = Path(args.raw_dir)

    emotions = [l.strip() for l in open(raw / "emotions.txt") if l.strip()]
    assert len(emotions) == 28

    # raw annotations: id -> list of 28-dim 0/1 tuples (one per annotator row), and id -> text
    annotations, texts = {}, {}
    for i in (1, 2, 3):
        with open(raw / f"goemotions_{i}.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ex = row["id"]
                annotations.setdefault(ex, []).append(tuple(1 if row[k] == "1" else 0 for k in emotions))
                texts[ex] = row["text"]

    # test split
    rows, n_excluded = [], 0
    with open(raw / "test.tsv", newline="", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            text, ex_id = parts[0], parts[2]
            raters = annotations.get(ex_id)
            if raters is None or len(raters) < MIN_ANNOTATORS:
                n_excluded += 1
                continue
            arr = np.array(raters, dtype=float)
            p_human = arr.sum(axis=0) / len(raters)
            H = float(np.mean([binary_entropy(pk) for pk in p_human]))
            rows.append((ex_id, texts.get(ex_id, text), len(raters), p_human, H))

    Hs = np.array([r[4] for r in rows])
    q25, q50, q75 = (float(x) for x in np.quantile(Hs, [0.25, 0.5, 0.75]))

    def stratum(H):
        if H <= q25:
            return 1
        if H <= q50:
            return 2
        if H <= q75:
            return 3
        return 4

    records = [{
        "id": ex_id,
        "text": text,
        "n_annotators": n_a,
        "p_human": [round(float(v), 6) for v in p_human],
        "H_human": round(H, 6),
        "quartile": stratum(H),
    } for ex_id, text, n_a, p_human, H in rows]

    out = {
        "emotions": emotions,
        "boundaries": {"Q25": q25, "Q50": q50, "Q75": q75},
        "n_records": len(records),
        "n_excluded_lt3": n_excluded,
        "quartile_counts": {q: sum(1 for r in records if r["quartile"] == q) for q in (1, 2, 3, 4)},
        "records": records,
    }
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(outpath, "w"), ensure_ascii=False)
    print(f"n_records={len(records)} n_excluded_lt3={n_excluded}")
    print(f"boundaries Q25={q25} Q50={q50} Q75={q75}")
    print(f"quartile_counts={out['quartile_counts']}")
    print(f"saved: {outpath}")


if __name__ == "__main__":
    main()
