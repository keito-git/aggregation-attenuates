#!/usr/bin/env python3
"""Build the GoEmotions reference file used by the analysis scripts.

The evaluation subset is the 1,200 items whose ids are stored in ref/C_sc_probs.npz (400 items per
non-empty human-entropy stratum). For these items, in that order, this script writes
  ids       (1200,)     GoEmotions example ids
  p_human   (1200, 28)  human soft labels (float32)
  H_human   (1200,)     mean binary entropy of p_human (bits/label)
  quartile  (1200,)     human-entropy stratum (1, 3 or 4)
taken from goe_test_prepared.json (see data/prepare_goemotions.py).

Usage:
  python3 analysis/build_goe_reference.py --data data/goe_test_prepared.json \
      --subset ref/C_sc_probs.npz --out data/goe_reference_1200.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="goe_test_prepared.json")
    ap.add_argument("--subset", required=True, help="npz with the 1,200 subset ids (key 'ids')")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.load(open(args.data))
    rec = {str(r["id"]): r for r in data["records"]}
    ids = [str(x) for x in np.load(args.subset, allow_pickle=True)["ids"]]
    missing = [x for x in ids if x not in rec]
    if missing:
        raise ValueError(f"{len(missing)} subset ids are not in {args.data}")

    p_human = np.array([rec[x]["p_human"] for x in ids], dtype=np.float32)
    H_human = np.array([rec[x]["H_human"] for x in ids], dtype=np.float64)
    quartile = np.array([rec[x]["quartile"] for x in ids], dtype=np.int64)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=np.array(ids), p_human=p_human, H_human=H_human, quartile=quartile)
    print(f"wrote {args.out}: {len(ids)} items, strata {dict(zip(*np.unique(quartile, return_counts=True)))}")


if __name__ == "__main__":
    main()
