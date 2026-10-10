"""
Result figures, drawn only from OUT_DIR/final_numbers.json (written by final_analysis.py).
No value is typed by hand and there are no fallbacks: a missing key raises an error.

Figures (single-column elsarticle, text width ~ 6.3 in):
  rho_h_forest.pdf  rhoH of the seven compute-matched methods with 95% item-bootstrap CIs
  quartile_jsd.pdf  JSD by human-entropy stratum for the same methods, (a) listing and (b) Yes/No formats
Usage: python3 analysis/make_figures.py [figure_dir]   (default: OUT_DIR/figures)
"""
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 9,
})

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = Path(os.environ.get("OUT_DIR", ROOT / "output"))
R = json.loads((RES_DIR / "final_numbers.json").read_text())
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else RES_DIR / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# method key -> (label, group); order = the order of Table 1
METHODS = [("sc", "SC", "indep"), ("plainbig", "plain-listing", "indep"), ("moa", "MoA", "agg"),
           ("debate", "debate", "agg"), ("zeroshot", "zero-shot", "ctx"), ("hou", "clarification", "ctx"),
           ("ccd", "CCD", "ctx")]
COLOR = {"indep": "#1f77b4", "agg": "#d62728", "ctx": "#7f7f7f"}
GROUP_LABEL = {"indep": "independent averaging", "agg": "interaction-based aggregation",
               "ctx": "context generation / zero-shot"}

# ---------------------------------------------------------------- forest plot
fig = plt.figure(figsize=(6.3, 2.6))
ax = fig.add_axes([0.2, 0.2, 0.77, 0.75])
for i, (k, lab, g) in enumerate(METHODS):
    v = R["rhoH_ci"][f"goe/qwen/{k}"]
    y = len(METHODS) - 1 - i
    ax.errorbar(v["rhoH"], y, xerr=[[v["rhoH"] - v["ci95"][0]], [v["ci95"][1] - v["rhoH"]]],
                fmt="o", color=COLOR[g], ms=4.5, capsize=2.5, lw=1.2)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(range(len(METHODS)))
ax.set_yticklabels([lab for _, lab, _ in METHODS][::-1])
ax.set_xlabel(r"$\rho_H$ (Spearman correlation with human-label entropy)")
handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[g], label=GROUP_LABEL[g]) for g in ("indep", "agg", "ctx")]
ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
fig.savefig(OUT / "rho_h_forest.pdf")
plt.close(fig)

# ------------------------------------------------------------- quartile JSD
strata = sorted(R["by_quartile_goe_qwen"]["sc"], key=int)
names = {"1": "lower half", "3": "3rd quartile", "4": "4th quartile"}
fig = plt.figure(figsize=(6.3, 2.9))
axes = [fig.add_axes([0.09, 0.2, 0.37, 0.6]), fig.add_axes([0.60, 0.2, 0.37, 0.6])]
panels = [("(a) listing format", ["sc", "plainbig", "moa", "debate"]),
          ("(b) Yes/No format", ["zeroshot", "hou", "ccd"])]
style = {"sc": "-o", "plainbig": "--o", "moa": "-s", "debate": "--s", "zeroshot": "-^", "hou": "--^", "ccd": ":^"}
lab = {k: l for k, l, _ in METHODS}
grp = {k: g for k, _, g in METHODS}
for ax, (title, ks) in zip(axes, panels):
    for k in ks:
        ys = [R["by_quartile_goe_qwen"][k][q]["JSD"] for q in strata]
        ax.plot(range(len(strata)), ys, style[k], color=COLOR[grp[k]], ms=4, lw=1.2, label=lab[k])
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([names[q] for q in strata])
    ax.set_xlim(-0.25, len(strata) - 0.75)
    ax.set_xlabel(title + ": human-label entropy stratum")
    ax.set_ylabel("JSD")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))
fig.savefig(OUT / "quartile_jsd.pdf")
plt.close(fig)
print(f"wrote {OUT}/rho_h_forest.pdf, quartile_jsd.pdf")
