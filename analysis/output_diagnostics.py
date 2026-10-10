"""
Plausibility checks of the generated GoEmotions outputs. DIAGNOSTIC ONLY: nothing here enters the paper
(paper numbers come only from final_analysis.py).

Checks:
  - Mistral and Yi, per method and seed: rhoH, H_mod, labels per item, all-zero rows (no emotion parsed by
    any sample or agent), share of items whose largest soft label equals 1. Failure modes such as a missing
    or duplicated BOS token or a broken chat format show up as many all-zero rows or a collapsed H_mod.
  - Dose-response: debate with L=1 must equal CoT K=5 exactly (same prompt and per-sample seeds); MoA with
    L=1 is an independent CoT-style K=5 average with different seeds; summaries for L=2 and L=4.
  - Optional (OLD_DIR set): comparison with earlier outputs C_goe_<family>_<method>_s<seed>.npz in OLD_DIR
    (mean |dp|, Pearson r of labels per item and of item entropy).

Environment variables: RESULTS_DIR [results], GOE_REF [data/goe_reference_1200.npz], OUT_DIR [output],
OLD_DIR [unset].
Output: OUT_DIR/output_diagnostics.json + console.
Usage: python3 analysis/output_diagnostics.py
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr

ROOT = Path(__file__).resolve().parent.parent
RES = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
REF = np.load(Path(os.environ.get("GOE_REF", ROOT / "data/goe_reference_1200.npz")), allow_pickle=True)
OUT_DIR = Path(os.environ.get("OUT_DIR", ROOT / "output"))
OLD = Path(os.environ["OLD_DIR"]) if os.environ.get("OLD_DIR") else None
REF_I = {str(x): i for i, x in enumerate(REF["ids"])}
PH = REF["p_human"].astype(np.float64)
EPS = 1e-12
SEEDS = (43, 44, 45)


def bent(p):
    p = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def load_aligned(f):
    z = np.load(f, allow_pickle=True)
    ids = [str(x) for x in z["ids"]]
    p = z["p"].astype(np.float64)
    out = np.full(PH.shape, np.nan)
    for j, x in enumerate(ids):
        if x in REF_I:
            out[REF_I[x]] = p[j]
    return out


def summ(p):
    ok = ~np.isnan(p).any(1)
    p, ph = p[ok], PH[ok]
    return {"n": int(ok.sum()), "rhoH": float(spearmanr(bent(p).mean(1), bent(ph).mean(1)).correlation),
            "Hmod": float(bent(p).mean()), "labels_per_item": float(p.sum(1).mean()),
            "allzero_rows": int((p.sum(1) == 0).sum()), "max_p_eq_1_share": float((p.max(1) == 1).mean())}


OUTD = {"families": {}, "p3": {}}
for fam in ("mistral", "yi"):
    OUTD["families"][fam] = {}
    for meth, name in (("plain", "plain_k5"), ("debate", "debateF_k5"), ("moa", "moaF_k5"), ("cot", "cot_k5")):
        for s in SEEDS:
            f = RES / f"p1/C_goe_{fam}_{name}_v2_s{s}.npz"
            if not f.exists():
                continue
            pn = load_aligned(f)
            row = {"new": summ(pn)}
            fo = OLD / f"C_goe_{fam}_{name}_s{s}.npz" if OLD else None
            if fo is not None and fo.exists():
                po = load_aligned(fo)
                row["old"] = summ(po)
                ok = ~np.isnan(po).any(1) & ~np.isnan(pn).any(1)
                row["mean_abs_dp"] = float(np.abs(pn[ok] - po[ok]).mean())
                row["r_labels_per_item"] = float(pearsonr(pn[ok].sum(1), po[ok].sum(1))[0])
                row["r_item_entropy"] = float(pearsonr(bent(pn[ok]).mean(1), bent(po[ok]).mean(1))[0])
            OUTD["families"][fam][f"{meth}_s{s}"] = row
for s in SEEDS:
    fc = RES / f"p1/C_goe_qwen_cot_k5_v2_s{s}.npz"
    fd, fm = RES / f"p3/C_goe_qwen_debateF_L1_k5_v2_s{s}.npz", RES / f"p3/C_goe_qwen_moaF_L1_k5_v2_s{s}.npz"
    if not (fc.exists() and fd.exists() and fm.exists()):
        continue
    cot, d1, m1 = load_aligned(fc), load_aligned(fd), load_aligned(fm)
    OUTD["p3"][f"s{s}"] = {"debateL1_equals_cotK5": bool(np.array_equal(cot, d1)),
                           "moaL1_vs_cotK5_mean_abs_dp": float(np.abs(cot - m1).mean()),
                           "moaL1": summ(m1), "cotK5": summ(cot)}
    for op, name in (("debate", "debateF"), ("moa", "moaF")):
        for L in (2, 4):
            f = RES / f"p3/C_goe_qwen_{name}_L{L}_k5_v2_s{s}.npz"
            if f.exists():
                OUTD["p3"][f"s{s}"][f"{op}L{L}"] = summ(load_aligned(f))

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "output_diagnostics.json").write_text(json.dumps(OUTD, indent=1))
for fam, rows in OUTD["families"].items():
    for k, v in rows.items():
        n, o = v["new"], v.get("old", {})
        print(f"{fam:7s} {k:10s} rhoH {n['rhoH']:+.3f} Hmod {n['Hmod']:.3f} lab {n['labels_per_item']:.2f} "
              f"zero {n['allzero_rows']:3d}" + (f" | old rhoH {o['rhoH']:+.3f} Hmod {o['Hmod']:.3f} "
                                               f"|dp| {v['mean_abs_dp']:.3f}" if o else ""))
for s, v in OUTD["p3"].items():
    print("p3", s, "debateL1==cotK5:", v["debateL1_equals_cotK5"], "moaL1 |dp| vs cot:", round(v["moaL1_vs_cotK5_mean_abs_dp"], 4),
          {k: (round(x["rhoH"], 3), round(x["Hmod"], 3), x["allzero_rows"]) for k, x in v.items() if isinstance(x, dict)})
print(f"wrote {OUT_DIR / 'output_diagnostics.json'}")
