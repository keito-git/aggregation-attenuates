"""
Inventory of the generated result files under RESULTS_DIR/{p1,p3} (read-only; nothing is modified).

For every planned family x method x seed x dataset cell it reports whether the file exists, and for each existing
file it loads the npz and checks (do not trust file size):
  - row count == expected item count (GoEmotions 1,200 frozen subset; LeWiDi-MP = len(MP_test.json))
  - column count (28 for GoEmotions, 2 for LeWiDi-MP), no NaN, values in [0, 1]
  - ids unique and identical (as a set) to the reference item set
  - p_human (if stored) equal to the frozen reference human soft labels
  - values on the expected grid (multiples of 1/K for K=5 listing-based methods)
  - share of all-zero rows (no emotion parsed by any sample/agent; parse-failure proxy for listing outputs)
  - mean number of labels per item (sum of p), compared within family across methods
Environment variables: RESULTS_DIR [results], GOE_REF [data/goe_reference_1200.npz],
MP_DATA [data/MP_test.json], OUT_DIR [output].
Output: printed table + OUT_DIR/inventory.json

Usage: python3 analysis/inventory_results.py
"""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
REF = Path(os.environ.get("GOE_REF", ROOT / "data/goe_reference_1200.npz"))
MP = Path(os.environ.get("MP_DATA", ROOT / "data/MP_test.json"))
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "output"))
SEEDS = (43, 44, 45)

ref = np.load(REF, allow_pickle=True)
REF_IDS = [str(x) for x in ref["ids"]]
REF_PH = dict(zip(REF_IDS, ref["p_human"].astype(np.float64)))
N_MP = len(json.load(open(MP)))

# planned cells: (dataset, family, method, seeds, relpath pattern, K for grid check)
PLAN = []
for fam in ("qwen", "mistral", "yi"):
    for meth, name in (("plain", "plain_k5"), ("debate", "debateF_k5"), ("moa", "moaF_k5")):
        PLAN.append(("goe", fam, meth, SEEDS, f"p1/C_goe_{fam}_{name}_v2_s{{s}}.npz", 5))
        PLAN.append(("lwd", fam, meth, SEEDS, f"p1/C_mp_{fam}_{name}_v2_s{{s}}.npz", 5))
for meth, name, seeds, K in (("sc", "sc43", SEEDS, 43), ("plainbig", "plain_k43", SEEDS, 43), ("cot", "cot_k5", SEEDS, 5),
                             ("listingone", "plain_k1", (43,), 1), ("yesnoone", "yesno_k1", (43,), 1),
                             ("yesnobig", "yesno_k43", (43,), 43), ("hou", "hou", (43,), None),
                             ("ccd", "ccd", (43,), None)):
    PLAN.append(("goe", "qwen", meth, seeds, f"p1/C_goe_qwen_{name}_v2_s{{s}}.npz", K))
for fam in ("mistral", "yi"):
    PLAN.append(("goe", fam, "cot", SEEDS, f"p1/C_goe_{fam}_cot_k5_v2_s{{s}}.npz", 5))
for op, name in (("debate", "debateF"), ("moa", "moaF")):
    for L in (1, 2, 4):
        PLAN.append(("goe", "qwen", f"{op}L{L}", SEEDS, f"p3/C_goe_qwen_{name}_L{L}_k5_v2_s{{s}}.npz", 5))


def check(path, ds, K):
    z = np.load(path, allow_pickle=True)
    keys = z.files
    pkeys = [k for k in keys if k.startswith("p") and k != "p_human"]
    ids = [str(x) for x in z["ids"]]
    out = {"keys": keys, "n_rows": None, "problems": []}
    exp_n, exp_c = (1200, 28) if ds == "goe" else (N_MP, 2)
    for pk in pkeys:
        p = z[pk].astype(np.float64)
        out[f"shape_{pk}"] = list(p.shape)
        if p.shape != (exp_n, exp_c):
            out["problems"].append(f"{pk} shape {p.shape} != {(exp_n, exp_c)}")
        if np.isnan(p).any():
            out["problems"].append(f"{pk} has NaN")
        if (p < -1e-9).any() or (p > 1 + 1e-9).any():
            out["problems"].append(f"{pk} outside [0,1]")
    p = z[pkeys[0]].astype(np.float64)
    out["n_rows"] = int(p.shape[0])
    if len(set(ids)) != len(ids):
        out["problems"].append("duplicate ids")
    if ds == "goe":
        if set(ids) != set(REF_IDS):
            out["problems"].append(f"id set differs from the 1,200 reference items ({len(set(ids) & set(REF_IDS))} overlap)")
        if "p_human" in keys:
            ph = z["p_human"].astype(np.float64)
            dev = max(float(np.abs(ph[j] - REF_PH[x]).max()) for j, x in enumerate(ids) if x in REF_PH)
            out["p_human_max_abs_dev_vs_ref"] = dev
            if dev > 1e-5:
                out["problems"].append(f"p_human deviates from reference by {dev:.2g}")
        out["allzero_rows"] = int((p.sum(1) == 0).sum())
        out["labels_per_item"] = float(p.sum(1).mean())
    else:
        out["rowsum_dev_from_1_max"] = float(np.abs(p.sum(1) - 1).max())
    if K:
        off = np.abs(p * K - np.round(p * K)).max()
        out["grid_1_over_K_max_dev"] = float(off)
        if off > 1e-4:
            out["problems"].append(f"values not on 1/{K} grid (max dev {off:.3g})")
    return out


rows, missing = [], []
for ds, fam, meth, seeds, pat, K in PLAN:
    for s in seeds:
        f = RES / pat.format(s=s)
        cell = {"dataset": ds, "family": fam, "method": meth, "seed": s, "file": str(f.relative_to(RES))}
        if not f.exists():
            cell["status"] = "MISSING"
            missing.append(cell)
        else:
            cell.update(check(f, ds, K))
            cell["status"] = "OK" if not cell["problems"] else "PROBLEM"
        rows.append(cell)

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "inventory.json"
OUT.write_text(json.dumps({"n_mp_items": N_MP, "cells": rows}, indent=1))
print(f"LeWiDi-MP items in MP_test.json: {N_MP}")
print(f"{'ds':4s} {'family':8s} {'method':11s} {'seed':4s} {'status':8s} {'rows':>5s} {'allzero':>7s} {'lab/item':>8s}  problems")
for c in rows:
    print(f"{c['dataset']:4s} {c['family']:8s} {c['method']:11s} {c['seed']:4d} {c['status']:8s} "
          f"{str(c.get('n_rows', '-')):>5s} {str(c.get('allzero_rows', '-')):>7s} "
          f"{(format(c['labels_per_item'], '.3f') if 'labels_per_item' in c else '-'):>8s}  {'; '.join(c.get('problems', []))}")
n_ok = sum(c["status"] == "OK" for c in rows)
print(f"\n{n_ok} OK, {sum(c['status'] == 'PROBLEM' for c in rows)} PROBLEM, {len(missing)} MISSING of {len(rows)} planned files")
print(f"wrote {OUT}")
