"""
Final analysis for "Aggregation Attenuates the Human Disagreement Signal in Compute-Matched Multi-Agent
LLM Soft-Label Prediction". Every number reported in the manuscript is written by this script to
numbers.tex (LaTeX macros) and final_numbers.json.

Inputs (environment variables; defaults relative to the repository root):
  RESULTS_DIR  directory with p1/ (main conditions) and p3/ (dose-response) npz outputs   [results]
  GOE_REF      GoEmotions reference npz: ids, p_human, quartile (analysis/build_goe_reference.py)
                                                                              [data/goe_reference_1200.npz]
  CEILING_JSON human reliability ceiling (analysis/human_reliability_ceiling.py)
                                                                       [output/human_reliability_ceiling.json]
  OUT_DIR      output directory                                                             [output]
Outputs: OUT_DIR/final_numbers.json, numbers.tex, tab_softce_sensitivity.tex, tab_ci_full.tex,
         MISSING.md (conditions whose outputs are not available).

Metrics:
  JSD      per-label binary JSD (bits), averaged over labels, then over items
  rhoH     Spearman( mean binary entropy of model p , mean binary entropy of human p ) across items
  softCE   per-label binary cross-entropy (bits) against human p, model p clipped at SOFTCE_EPS = 1e-9
           (listing methods emit exact zeros, so soft-CE depends on this constant; the sensitivity
           over EPS is reported)
  macroF1  0.5 threshold on model and human p, macro over labels
  AURC     risk = per-item JSD, confidence = negative mean binary entropy of model p
  catJSD   JSD between L1-normalised 28-dim vectors (GoEmotions only)
  Hmod     mean per-item binary entropy of model p (direct spread of the output distribution)

Inference: item-level paired bootstrap, B = 5000, seed 20260929.
For a condition with several seeds, the estimate is the mean over seeds and each bootstrap replicate
resamples items once and averages the statistic over seeds (seeds are pooled, items are the unit).
Two-sided bootstrap p = 2 min(P(d<=0), P(d>=0)), floored at 1/(B+1). Holm correction within each
pre-declared family below. Two Holm adjustments are reported: over the contrasts whose outputs are
available (p_holm), and over the full pre-declared family with every missing contrast entering with
p = 1 (p_holm_decl, conservative).

Usage: python3 analysis/final_analysis.py
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

EPS = 1e-12
SOFTCE_EPS = 1e-9
SOFTCE_EPS_GRID = (1e-4, 1e-6, 1e-9, 1e-12)
B_BOOT = 5000
SEED = 20260929
SEEDS = (43, 44, 45)

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("OUT_DIR", ROOT / "output"))
P1 = Path(os.environ.get("RESULTS_DIR", ROOT / "results")) / "p1"
REF = Path(os.environ.get("GOE_REF", ROOT / "data/goe_reference_1200.npz"))
CEIL = Path(os.environ.get("CEILING_JSON", OUT / "human_reliability_ceiling.json"))

# ---------------------------------------------------------------- conditions
# key -> (file pattern with {s}, seeds, npz key)
GOE = "C_goe_{model}_{name}_v2_s{s}.npz"
COND = {}
for model in ("qwen", "mistral", "yi"):
    for method, name in (("plain", "plain_k5"), ("debate", "debateF_k5"), ("moa", "moaF_k5")):
        COND[("goe", model, method)] = (GOE.format(model=model, name=name, s="{s}"), SEEDS, "p")
COND.update({
    ("goe", "qwen", "sc"): ("C_goe_qwen_sc43_v2_s{s}.npz", SEEDS, "p"),
    ("goe", "qwen", "plainbig"): ("C_goe_qwen_plain_k43_v2_s{s}.npz", SEEDS, "p"),
    ("goe", "qwen", "cot"): ("C_goe_qwen_cot_k5_v2_s{s}.npz", SEEDS, "p"),
    ("goe", "mistral", "cot"): ("C_goe_mistral_cot_k5_v2_s{s}.npz", SEEDS, "p"),
    ("goe", "yi", "cot"): ("C_goe_yi_cot_k5_v2_s{s}.npz", SEEDS, "p"),
    ("goe", "qwen", "listingone"): ("C_goe_qwen_plain_k1_v2_s{s}.npz", (43,), "p"),
    ("goe", "qwen", "yesnoone"): ("C_goe_qwen_yesno_k1_v2_s{s}.npz", (43,), "p"),
    ("goe", "qwen", "yesnobig"): ("C_goe_qwen_yesno_k43_v2_s{s}.npz", (43,), "p"),
    ("goe", "qwen", "hou"): ("C_goe_qwen_hou_v2_s{s}.npz", (43,), "p_hou"),
    ("goe", "qwen", "ccd"): ("C_goe_qwen_ccd_v2_s{s}.npz", (43,), "p_ccd"),
    ("goe", "qwen", "zeroshot"): ("C_goe_qwen_ccd_v2_s{s}.npz", (43,), "p_base"),
})
for method, name in (("plain", "plain_k5"), ("debate", "debateF_k5"), ("moa", "moaF_k5")):
    for model in ("qwen", "mistral", "yi"):  # LeWiDi-MP
        COND[("lwd", model, method)] = (f"C_mp_{model}_{name}_v2_s{{s}}.npz", SEEDS, "p")

# dose-response: total interaction rounds L; L=3 is the main condition (p1/), L=1,2,4 are in p3/
DOSE_L = (1, 2, 3, 4)
for op, name in (("debate", "debateF"), ("moa", "moaF")):
    for L in (1, 2, 4):
        COND[("goe", "qwen", f"{op}L{L}")] = (f"../p3/C_goe_qwen_{name}_L{L}_k5_v2_s{{s}}.npz", SEEDS, "p")

# ------------------------------------------------- pre-declared families (Holm)
FAMILY_RHO = [  # primary conclusion: independent averaging vs interaction-based aggregation, on rhoH
    (("goe", "qwen", "sc"), ("goe", "qwen", "debate")),
    (("goe", "qwen", "sc"), ("goe", "qwen", "moa")),
    (("goe", "qwen", "plain"), ("goe", "qwen", "debate")),
    (("goe", "qwen", "plain"), ("goe", "qwen", "moa")),
    (("goe", "qwen", "cot"), ("goe", "qwen", "debate")),
    (("goe", "qwen", "cot"), ("goe", "qwen", "moa")),
    (("goe", "mistral", "plain"), ("goe", "mistral", "debate")),
    (("goe", "mistral", "plain"), ("goe", "mistral", "moa")),
    (("goe", "yi", "plain"), ("goe", "yi", "debate")),
    (("goe", "yi", "plain"), ("goe", "yi", "moa")),
    (("lwd", "qwen", "plain"), ("lwd", "qwen", "debate")),
    (("lwd", "qwen", "plain"), ("lwd", "qwen", "moa")),
    (("goe", "mistral", "cot"), ("goe", "mistral", "debate")),
    (("goe", "mistral", "cot"), ("goe", "mistral", "moa")),
    (("goe", "yi", "cot"), ("goe", "yi", "debate")),
    (("goe", "yi", "cot"), ("goe", "yi", "moa")),
    (("lwd", "mistral", "plain"), ("lwd", "mistral", "debate")),
    (("lwd", "mistral", "plain"), ("lwd", "mistral", "moa")),
    (("lwd", "yi", "plain"), ("lwd", "yi", "debate")),
    (("lwd", "yi", "plain"), ("lwd", "yi", "moa")),
]
FAMILY_JSD = FAMILY_RHO + [  # distribution fit; context generation vs zero-shot and vs independent sampling
    (("goe", "qwen", "sc"), ("goe", "qwen", "plainbig")),
    (("goe", "qwen", "hou"), ("goe", "qwen", "zeroshot")),
    (("goe", "qwen", "ccd"), ("goe", "qwen", "zeroshot")),
    (("goe", "qwen", "plainbig"), ("goe", "qwen", "hou")),
]
SECONDARY_RHO = [(("goe", "qwen", "sc"), ("goe", "qwen", "plainbig"))]  # reported, not in the family


# ------------------------------------------------------------------ metrics
def clip(p):
    return np.clip(p.astype(np.float64), EPS, 1 - EPS)


def bent(p):
    p = clip(p)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def jsd_items(pm, ph):
    pm, ph = clip(pm), clip(ph)
    m = 0.5 * (pm + ph)

    def kl(a, b):
        return a * np.log2(a / b) + (1 - a) * np.log2((1 - a) / (1 - b))
    return (0.5 * kl(pm, m) + 0.5 * kl(ph, m)).mean(axis=1)


def softce_items(pm, ph, eps=SOFTCE_EPS):
    pm = np.clip(pm.astype(np.float64), eps, 1 - eps)
    return -(ph * np.log2(pm) + (1 - ph) * np.log2(1 - pm)).mean(axis=1)


def macro_f1(pm, ph):
    pred, gold = pm >= 0.5, ph >= 0.5
    f = []
    for k in range(pm.shape[1]):
        tp = (pred[:, k] & gold[:, k]).sum()
        fp = (pred[:, k] & ~gold[:, k]).sum()
        fn = (~pred[:, k] & gold[:, k]).sum()
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f))


def aurc(loss, conf):
    order = np.argsort(-conf)
    l = loss[order]
    return float((np.cumsum(l) / np.arange(1, len(l) + 1)).mean())


def cat_jsd_items(p, q, eps=1e-9):
    def norm(a):
        s = a.sum(axis=1, keepdims=True)
        return a / np.where(s < eps, 1.0, s)
    P, Q = norm(p.astype(np.float64)), norm(q.astype(np.float64))
    M = 0.5 * (P + Q)

    def kl(a, b):
        a, b = np.clip(a, eps, 1), np.clip(b, eps, 1)
        return (a * np.log2(a / b)).sum(axis=1)
    return 0.5 * kl(P, M) + 0.5 * kl(Q, M)


# -------------------------------------------------------------------- data
ref = np.load(REF, allow_pickle=True)
REF_IDS = [str(x) for x in ref["ids"]]
REF_I = {x: i for i, x in enumerate(REF_IDS)}
HUMAN = {"goe": ref["p_human"].astype(np.float64)}
QUART = ref["quartile"]


def load(key):
    """Return (list of per-seed p aligned to the dataset's item order, human p) or None if missing."""
    pat, seeds, pkey = COND[key]
    ds = key[0]
    ps, ph = [], None
    for s in seeds:
        f = P1 / pat.format(s=s)
        if not f.exists():
            return None
        z = np.load(f, allow_pickle=True)
        ids = [str(x) for x in z["ids"]]
        p = z[pkey].astype(np.float64)
        if np.isnan(p).any():
            raise ValueError(f"{f.name}: NaN in {pkey}")
        if ds == "goe":
            out = np.full(HUMAN["goe"].shape, np.nan)
            for j, x in enumerate(ids):
                out[REF_I[x]] = p[j]
            if np.isnan(out).any():
                raise ValueError(f"{f.name}: does not cover the 1,200 reference items")
            ps.append(out)
        else:  # LeWiDi: item order is the file's own; all seeds must agree
            if ph is None:
                ph, lwd_ids = z["p_human"].astype(np.float64), ids
            elif ids != lwd_ids:
                raise ValueError(f"{f.name}: item order differs across seeds")
            ps.append(p)
    if ds == "goe":
        ph = HUMAN["goe"]
    return ps, ph


DATA, MISSING = {}, []
for k in COND:
    d = load(k)
    if d is None:
        MISSING.append(k)
    else:
        DATA[k] = d


def metrics(p, ph, ds):
    je = jsd_items(p, ph)
    m = {"JSD": float(je.mean()),
         "rhoH": float(spearmanr(bent(p).mean(1), bent(ph).mean(1)).correlation),
         "softCE": float(softce_items(p, ph).mean()),
         "macroF1": macro_f1(p, ph),
         "AURC": aurc(je, -bent(p).mean(1)),
         "Hmod": float(bent(p).mean())}
    if ds == "goe":
        m["catJSD"] = float(cat_jsd_items(p, ph).mean())
    return m


RES = {"conditions": {}, "contrasts_rhoH": {}, "contrasts_JSD": {}, "missing": [list(k) for k in MISSING]}
for k, (ps, ph) in DATA.items():
    per = [metrics(p, ph, k[0]) for p in ps]
    agg = {m: float(np.mean([x[m] for x in per])) for m in per[0]}
    sd = {m: float(np.std([x[m] for x in per], ddof=1)) if len(per) > 1 else None for m in per[0]}
    RES["conditions"]["/".join(k)] = {"n_seeds": len(per), "per_seed": per, "mean": agg, "sd": sd,
                                      "n_items": int(len(ph))}
RES["human_Hmod_goe"] = float(bent(HUMAN["goe"]).mean())

# soft-CE sensitivity to the clipping constant (GoEmotions, Qwen, main-table methods, seed means)
MAIN7 = ("sc", "plainbig", "moa", "debate", "zeroshot", "hou", "ccd")
sens = {}
for eps in SOFTCE_EPS_GRID:
    row = {}
    for meth in MAIN7:
        k = ("goe", "qwen", meth)
        if k in DATA:
            ps, ph = DATA[k]
            row[meth] = float(np.mean([softce_items(p, ph, eps).mean() for p in ps]))
    sens[f"{eps:g}"] = row
RES["softce_eps_sensitivity"] = sens
if all(len(r) == len(MAIN7) for r in sens.values()):
    RES["softce_eps_rank_stable"] = {
        "sc_best_at_all_eps": all(min(r, key=r.get) == "sc" for r in sens.values()),
        "moa_worst_at_all_eps": all(max(r, key=r.get) == "moa" for r in sens.values()),
        "full_ranking_identical": len({tuple(sorted(r, key=r.get)) for r in sens.values()}) == 1}


# --------------------------------------------------------------- bootstrap
def boot_contrast(a, b, stat):
    """Paired item bootstrap of mean_s stat(A_s) - mean_s stat(B_s)."""
    psa, ph = DATA[a]
    psb, _ = DATA[b]
    n = len(ph)
    Hh = bent(ph).mean(1)
    if stat == "rhoH":
        HA = [bent(p).mean(1) for p in psa]
        HB = [bent(p).mean(1) for p in psb]

        def f(idx):
            ra = np.mean([spearmanr(h[idx], Hh[idx]).correlation for h in HA])
            rb = np.mean([spearmanr(h[idx], Hh[idx]).correlation for h in HB])
            return ra - rb
    else:
        JA = np.mean([jsd_items(p, ph) for p in psa], axis=0)
        JB = np.mean([jsd_items(p, ph) for p in psb], axis=0)
        d = JA - JB

        def f(idx):
            return d[idx].mean()
    obs = f(np.arange(n))
    rng = np.random.default_rng(SEED)
    bs = np.array([f(rng.integers(0, n, n)) for _ in range(B_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / (B_BOOT + 1))
    return {"diff": float(obs), "ci95": [float(lo), float(hi)], "p_boot": float(p), "B": B_BOOT,
            "seeds_a": len(psa), "seeds_b": len(psb)}


def holm(results, keys, declared=False):
    """Holm step-down. declared=False: over the contrasts available now (field p_holm).
    declared=True: over the full declared family, missing contrasts entering with p = 1 (field p_holm_decl);
    since missing p = 1 sort last, this equals multiplying with the declared family size m (conservative)."""
    avail = [k for k in keys if k in results]
    order = sorted(avail, key=lambda k: results[k]["p_boot"])
    m, run = (len(keys) if declared else len(avail)), 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (m - r) * results[k]["p_boot"]))
        results[k]["p_holm_decl" if declared else "p_holm"] = float(run)
        results[k]["holm_family_size_declared" if declared else "holm_family_size_available"] = m


def cname(a, b):
    return "/".join(a) + "  minus  " + "/".join(b)


for fam, stat, dest in ((FAMILY_RHO, "rhoH", "contrasts_rhoH"), (FAMILY_JSD, "JSD", "contrasts_JSD"),
                        (SECONDARY_RHO, "rhoH", "contrasts_rhoH")):
    for a, b in fam:
        if a in DATA and b in DATA and cname(a, b) not in RES[dest]:
            RES[dest][cname(a, b)] = boot_contrast(a, b, stat)
holm(RES["contrasts_rhoH"], [cname(a, b) for a, b in FAMILY_RHO])
holm(RES["contrasts_JSD"], [cname(a, b) for a, b in FAMILY_JSD])
holm(RES["contrasts_rhoH"], [cname(a, b) for a, b in FAMILY_RHO], declared=True)
holm(RES["contrasts_JSD"], [cname(a, b) for a, b in FAMILY_JSD], declared=True)
RES["holm_family_sizes"] = {
    "rhoH": {"declared": len(FAMILY_RHO), "available": sum(cname(a, b) in RES["contrasts_rhoH"] for a, b in FAMILY_RHO)},
    "JSD": {"declared": len(FAMILY_JSD), "available": sum(cname(a, b) in RES["contrasts_JSD"] for a, b in FAMILY_JSD)}}
RES["holm_family_complete"] = {
    "rhoH": all(cname(a, b) in RES["contrasts_rhoH"] for a, b in FAMILY_RHO),
    "JSD": all(cname(a, b) in RES["contrasts_JSD"] for a, b in FAMILY_JSD)}

# ------------------------------------------------------- 2x2 factorial (output format x K)
fac_keys = {"listing_K1": ("goe", "qwen", "listingone"), "listing_K43": ("goe", "qwen", "plainbig"),
            "yesno_K1": ("goe", "qwen", "yesnoone"), "yesno_K43": ("goe", "qwen", "yesnobig")}
if all(k in DATA for k in fac_keys.values()):
    ph = HUMAN["goe"]
    cell = {c: DATA[k][0][0] for c, k in fac_keys.items()}  # seed 43 in every cell
    fac = {"cells": {c: {"binJSD": float(jsd_items(p, ph).mean()), "catJSD": float(cat_jsd_items(p, ph).mean())}
                     for c, p in cell.items()}}
    for name in ("binJSD", "catJSD"):
        T = {c: fac["cells"][c][name] for c in cell}
        fmt = 0.5 * ((T["yesno_K1"] - T["listing_K1"]) + (T["yesno_K43"] - T["listing_K43"]))
        kef = 0.5 * ((T["listing_K1"] - T["listing_K43"]) + (T["yesno_K1"] - T["yesno_K43"]))
        inter = (T["yesno_K1"] - T["listing_K1"]) - (T["yesno_K43"] - T["listing_K43"])
        fac[name] = {"format_effect": fmt, "K_effect": kef, "interaction": inter,
                     "format_share_pct": 100 * abs(fmt) / (abs(fmt) + abs(kef))}
    # temperature calibration of Yes/No K=43, T chosen on the first half by soft-CE
    pyn = cell["yesno_K43"]
    k = len(pyn) // 2
    best_T, best = 1.0, np.inf
    for T in np.linspace(0.3, 3.0, 28):
        pc = clip(pyn[:k])
        pcal = pc ** (1 / T) / (pc ** (1 / T) + (1 - pc) ** (1 / T))
        ce = softce_items(pcal, ph[:k]).mean()
        if ce < best:
            best, best_T = ce, T
    pc = clip(pyn)
    pcal = pc ** (1 / best_T) / (pc ** (1 / best_T) + (1 - pc) ** (1 / best_T))
    fac["yesno_K43_calibrated"] = {"T": float(best_T), "binJSD": float(jsd_items(pcal, ph).mean()),
                                   "catJSD": float(cat_jsd_items(pcal, ph).mean())}
    RES["factorial"] = fac

# ------------------------------------------------- item-level by human-entropy stratum
RES["by_quartile_goe_qwen"] = {}
for meth in ("sc", "plainbig", "plain", "cot", "debate", "moa", "zeroshot", "hou", "ccd"):
    k = ("goe", "qwen", meth)
    if k in DATA:
        ps, ph = DATA[k]
        je = np.mean([jsd_items(p, ph) for p in ps], axis=0)
        hm = np.mean([bent(p).mean(1) for p in ps], axis=0)
        RES["by_quartile_goe_qwen"][meth] = {
            int(q): {"JSD": float(je[QUART == q].mean()), "Hmod": float(hm[QUART == q].mean())}
            for q in np.unique(QUART)}
RES["by_quartile_goe_human_Hmod"] = {int(q): float(bent(HUMAN["goe"][QUART == q]).mean())
                                     for q in np.unique(QUART)}
RES["quartile_n"] = {int(q): int((QUART == q).sum()) for q in np.unique(QUART)}

# ------------------------------------------------ human reliability ceiling
ceil = json.loads(CEIL.read_text())
RES["human_ceiling"] = {"r_half": ceil["split_half_rho_mean"], "r_half_sd": ceil["split_half_rho_std"],
                        "r_SB": ceil["spearman_brown_ceiling"], "n_splits": ceil["n_splits"]}
if ("goe", "qwen", "sc") in DATA:
    RES["human_ceiling"]["sc_share_pct"] = 100 * RES["conditions"]["goe/qwen/sc"]["mean"]["rhoH"] / ceil["spearman_brown_ceiling"]

# ------------------------------------------------ per-condition rhoH CI (forest plot, "significantly positive")
def boot_rho_single(k):
    ps, ph = DATA[k]
    Hh = bent(ph).mean(1)
    Hs = [bent(p).mean(1) for p in ps]
    n = len(ph)

    def f(idx):
        return float(np.mean([spearmanr(h[idx], Hh[idx]).correlation for h in Hs]))
    rng = np.random.default_rng(SEED)
    bs = np.array([f(rng.integers(0, n, n)) for _ in range(B_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"rhoH": f(np.arange(n)), "ci95": [float(lo), float(hi)]}


RES["rhoH_ci"] = {}
for k in DATA:
    if k[2] in ("listingone", "yesnoone"):  # K=1: rhoH undefined (outputs are 0/1)
        continue
    RES["rhoH_ci"]["/".join(k)] = boot_rho_single(k)

# ------------------------------------------------ dose-response (Holm over the 2 operators)
# Pre-declared test: slope of the seed-mean rhoH on L (least squares over L = 1..4), item-level
# bootstrap; H1 = slope < 0 is tested two-sided.
def dose_levels(op):
    return [("goe", "qwen", f"{op}L{L}") if L != 3 else ("goe", "qwen", op) for L in DOSE_L]


RES["dose"] = {}
for op in ("debate", "moa"):
    keys = dose_levels(op)
    if not all(k in DATA for k in keys):
        continue
    ph = HUMAN["goe"]
    Hh = bent(ph).mean(1)
    Hs = [[bent(p).mean(1) for p in DATA[k][0]] for k in keys]
    Ls = np.array(DOSE_L, dtype=float)

    def slope(idx):
        r = [np.mean([spearmanr(h[idx], Hh[idx]).correlation for h in lev]) for lev in Hs]
        return float(np.polyfit(Ls, r, 1)[0]), r
    n = len(ph)
    obs, r_obs = slope(np.arange(n))
    rng = np.random.default_rng(SEED)
    bs = np.array([slope(rng.integers(0, n, n))[0] for _ in range(B_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pb = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / (B_BOOT + 1))
    hm = [float(np.mean([bent(p).mean() for p in DATA[k][0]])) for k in keys]  # descriptive, not tested
    RES["dose"][op] = {"L": list(DOSE_L), "rhoH_by_L": r_obs, "slope": obs, "ci95": [float(lo), float(hi)],
                       "p_boot": float(pb), "monotone_decreasing": bool(all(np.diff(r_obs) < 0)),
                       "Hmod_by_L": hm, "n_seeds_by_L": [len(DATA[k][0]) for k in keys]}
if RES["dose"]:
    holm(RES["dose"], list(RES["dose"]))

# ------------------------------------------------------------- numbers.tex
CAP = {"goe": "Goe", "lwd": "Lwd", "qwen": "Qwen", "mistral": "Mistral", "yi": "Yi",
       "plain": "Plain", "debate": "Debate", "moa": "Moa", "sc": "Sc", "plainbig": "PlainBig",
       "cot": "Cot", "listingone": "ListingOne", "yesnoone": "YesNoOne", "yesnobig": "YesNoBig",
       "hou": "Hou", "ccd": "Ccd", "zeroshot": "ZeroShot",
       "debateL1": "DebateLOne", "debateL2": "DebateLTwo", "debateL4": "DebateLFour",
       "moaL1": "MoaLOne", "moaL2": "MoaLTwo", "moaL4": "MoaLFour"}
MET = {"JSD": ("Jsd", 3), "rhoH": ("RhoH", 3), "softCE": ("SoftCe", 3), "macroF1": ("MacroFone", 3),
       "AURC": ("Aurc", 3), "Hmod": ("Hmod", 3), "catJSD": ("CatJsd", 3)}


def fmt(x, nd, sign=False):
    x = 0.0 if round(x, nd) == 0 else x  # no "-0.000"
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace("-", "$-$") if sign or x < 0 else s


def fmt_p(p):
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


lines = ["% AUTO-GENERATED by revision_2026-09/final_analysis.py -- do not edit by hand"]


def mac(name, val):
    # \xspace: keep the space after the macro when a word follows (requires \usepackage{xspace})
    lines.append(f"\\newcommand{{\\{name}}}{{{val}\\xspace}}")


for k, v in RES["conditions"].items():
    base = "".join(CAP[x] for x in k.split("/"))
    for met, (mn, nd) in MET.items():
        if met in v["mean"]:
            mac(mn + base, fmt(v["mean"][met], nd, sign=(met == "rhoH")))
            if v["sd"][met] is not None:
                mac(mn + "Sd" + base, fmt(v["sd"][met], 3 if met == "rhoH" else 4))
    if v["n_seeds"] > 1:
        signs = [x["rhoH"] for x in v["per_seed"]]
        mac("RhoHNegSeeds" + base, str(sum(s < 0 for s in signs)))
for dest, tag in (("contrasts_rhoH", "DRhoH"), ("contrasts_JSD", "DJsd")):
    for k, v in RES[dest].items():
        a, b = [x.strip().split("/") for x in k.split("minus")]
        base = "".join(CAP[x] for x in a) + "Vs" + "".join(CAP[x] for x in b[2:])
        nd = 3
        mac(tag + base, fmt(v["diff"], nd, sign=True))
        mac(tag + base + "Lo", fmt(v["ci95"][0], nd, sign=True))
        mac(tag + base + "Hi", fmt(v["ci95"][1], nd, sign=True))
        if "p_holm" in v:
            mac(tag + base + "PHolm", fmt_p(v["p_holm"]))
        if "p_holm_decl" in v:
            mac(tag + base + "PHolmDecl", fmt_p(v["p_holm_decl"]))
        mac(tag + base + "P", fmt_p(v["p_boot"]))
if "factorial" in RES:
    f = RES["factorial"]
    for c, v in f["cells"].items():
        cc = c.replace("_K1", "One").replace("_K43", "Big").replace("listing", "Listing").replace("yesno", "YesNo")
        mac("FacBinJsd" + cc, fmt(v["binJSD"], 3))
        mac("FacCatJsd" + cc, fmt(v["catJSD"], 3))
    for name, tag in (("binJSD", "Bin"), ("catJSD", "Cat")):
        mac(f"Fac{tag}FormatEffect", fmt(f[name]["format_effect"], 3, sign=True))
        mac(f"Fac{tag}KEffect", fmt(f[name]["K_effect"], 3, sign=True))
        mac(f"Fac{tag}FormatSharePct", f"{f[name]['format_share_pct']:.1f}")
    mac("FacCalT", f"{f['yesno_K43_calibrated']['T']:.2f}")
    mac("FacCalBinJsd", fmt(f["yesno_K43_calibrated"]["binJSD"], 3))
    mac("FacCalCatJsd", fmt(f["yesno_K43_calibrated"]["catJSD"], 3))
for k, v in RES["rhoH_ci"].items():
    base = "".join(CAP[x] for x in k.split("/"))
    mac("RhoH" + base + "Lo", fmt(v["ci95"][0], 3, sign=True))
    mac("RhoH" + base + "Hi", fmt(v["ci95"][1], 3, sign=True))
for op, v in RES["dose"].items():
    mac(f"DoseSlope{CAP[op]}", fmt(v["slope"], 3, sign=True))
    mac(f"DoseSlope{CAP[op]}Lo", fmt(v["ci95"][0], 3, sign=True))
    mac(f"DoseSlope{CAP[op]}Hi", fmt(v["ci95"][1], 3, sign=True))
    mac(f"DoseSlope{CAP[op]}PHolm", fmt_p(v["p_holm"]))
    mac(f"DoseSlope{CAP[op]}P", fmt_p(v["p_boot"]))
    mac(f"DoseMonotone{CAP[op]}", "yes" if v["monotone_decreasing"] else "no")
for fam_name, tag in (("rhoH", "Rho"), ("JSD", "Jsd")):
    mac(f"HolmFam{tag}Declared", str(RES["holm_family_sizes"][fam_name]["declared"]))
    mac(f"HolmFam{tag}Available", str(RES["holm_family_sizes"][fam_name]["available"]))
hc = RES["human_ceiling"]
mac("HumanCeilingRSB", f"{hc['r_SB']:.3f}")
mac("HumanCeilingRHalf", f"{hc['r_half']:.3f}")
mac("HumanCeilingNSplits", f"{hc['n_splits']:,}".replace(",", "{,}"))
if "sc_share_pct" in hc:
    mac("ScShareOfCeilingPct", f"{hc['sc_share_pct']:.0f}")
mac("HumanHmodGoe", fmt(RES["human_Hmod_goe"], 3))
mac("NBoot", f"{B_BOOT:,}".replace(",", "{,}"))
mac("SoftCeEpsExp", "9")  # SOFTCE_EPS = 10^-9
if "softce_eps_sensitivity" in RES and ("goe", "qwen", "sc") in DATA:
    v = [r["sc"] for r in RES["softce_eps_sensitivity"].values()]
    mac("SoftCeScEpsMin", fmt(min(v), 3)); mac("SoftCeScEpsMax", fmt(max(v), 3))
for q, v in RES["by_quartile_goe_qwen"].items():
    for qq, vv in v.items():
        mac(f"QJsd{CAP[q]}Q{'ABCD'[qq - 1]}", fmt(vv["JSD"], 3))

# --------------------------------------------------------- generated table bodies
LABEL = {"sc": "SC", "plainbig": "plain-listing ($K=43$)", "plain": "plain-listing ($K=5$)", "cot": "CoT ($K=5$)",
         "moa": "MoA", "debate": "debate", "zeroshot": "zero-shot", "hou": "clarification", "ccd": "CCD"}
DSL = {"goe": "GoEmotions", "lwd": "LeWiDi-MP"}
MDL = {"qwen": "Qwen3-32B", "mistral": "Mistral-24B", "yi": "Yi-1.5-34B"}
TABLES = {}
if "softce_eps_sensitivity" in RES:
    eps_keys = list(RES["softce_eps_sensitivity"])
    rows = [r"\begin{tabular}{l" + "c" * len(eps_keys) + "}", r"\toprule",
            " & " + " & ".join(f"$\\epsilon=10^{{{int(round(np.log10(float(e))))}}}$" for e in eps_keys) + r" \\",
            r"\midrule"]
    for meth in MAIN7:
        vals = [RES["softce_eps_sensitivity"][e].get(meth) for e in eps_keys]
        if None in vals:
            continue
        best = [min(RES["softce_eps_sensitivity"][e].values()) for e in eps_keys]
        cells = [(f"\\textbf{{{v:.3f}}}" if abs(v - b) < 1e-12 else f"{v:.3f}") for v, b in zip(vals, best)]
        rows.append(LABEL[meth] + " & " + " & ".join(cells) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    TABLES["tab_softce_sensitivity.tex"] = rows


def cname_label(k):
    a, b = [x.strip().split("/") for x in k.split("minus")]
    return f"{DSL[a[0]]} & {MDL[a[1]]} & {LABEL[a[2]]} $-$ {LABEL[b[2]]}"


rows = [r"\begin{tabular}{lllccccc}", r"\toprule",
        r"Data & Model & Contrast & Metric & Difference & 95\% CI & $p_{\mathrm{Holm}}$ & $p_{\mathrm{Holm}}^{\mathrm{decl}}$ \\",
        r"\midrule"]
for dest, met in (("contrasts_rhoH", r"$\rho_H$"), ("contrasts_JSD", "JSD")):
    for k, v in RES[dest].items():
        ph = fmt_p(v["p_holm"]) if "p_holm" in v else "--"
        phd = fmt_p(v["p_holm_decl"]) if "p_holm_decl" in v else "--"
        rows.append(f"{cname_label(k)} & {met} & {fmt(v['diff'], 3, sign=True)} & "
                    f"[{fmt(v['ci95'][0], 3, sign=True)}, {fmt(v['ci95'][1], 3, sign=True)}] & {ph} & {phd} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
TABLES["tab_ci_full.tex"] = rows

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "final_numbers.json").write_text(json.dumps(RES, indent=1))
(OUT / "numbers.tex").write_text("\n".join(lines) + "\n")
for name, rows in TABLES.items():
    (OUT / name).write_text("% AUTO-GENERATED by final_analysis.py\n" + "\n".join(rows) + "\n")
(OUT / "MISSING.md").write_text(
    "# Conditions not yet available\n\n" + "\n".join(f"- {'/'.join(k)}  ({COND[k][0]})" for k in MISSING) + "\n"
    + f"\nHolm families complete: {RES['holm_family_complete']}\n")

# ---------------------------------------------------------------- console
print(f"available {len(DATA)}/{len(COND)} conditions; missing: {['/'.join(k) for k in MISSING]}")
print(f"{'condition':24s} {'seeds':>5s} {'rhoH':>7s} {'JSD':>6s} {'softCE':>7s} {'mF1':>6s} {'Hmod':>6s}")
for k, v in RES["conditions"].items():
    m = v["mean"]
    print(f"{k:24s} {v['n_seeds']:5d} {m['rhoH']:+7.3f} {m['JSD']:6.3f} {m['softCE']:7.3f} {m['macroF1']:6.3f} {m['Hmod']:6.3f}")
for dest in ("contrasts_rhoH", "contrasts_JSD"):
    print(f"\n{dest} (Holm family complete: {RES['holm_family_complete']})")
    for k, v in RES[dest].items():
        print(f"  {k:62s} {v['diff']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] "
              f"p={v['p_boot']:.4f} pHolm={v.get('p_holm', float('nan')):.4f} "
              f"pHolmDecl={v.get('p_holm_decl', float('nan')):.4f}")
print("\nsoft-CE eps sensitivity:", json.dumps(RES.get("softce_eps_sensitivity"), indent=None))
print("rank stability:", RES.get("softce_eps_rank_stable"))
print("Holm family sizes:", RES["holm_family_sizes"])
for op, v in RES["dose"].items():
    print(f"dose {op}: rhoH by L={[round(x, 3) for x in v['rhoH_by_L']]} Hmod by L={[round(x, 3) for x in v['Hmod_by_L']]} "
          f"slope={v['slope']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}] p={v['p_boot']:.4f} pHolm={v['p_holm']:.4f} "
          f"monotone={v['monotone_decreasing']}")
print(f"\nwrote {OUT}/numbers.tex ({len(lines) - 1} macros)")
