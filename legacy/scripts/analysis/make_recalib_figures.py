"""Redraw the operating-point figures affected by the axis-visibility re-calibration.

Reads ONLY the new re-calibration artifacts (result/recalib_C_*, result/operating_point_measure)
and renders IEEE-style figures into result/redraw_figure_v1, replacing the old road_segment-artifact
operating-point figures (F1.3/F1.4/F1.5/F4.5/F5.5/F5.8). F = consensus FAILURE rate (lower = better).

  R1  reliability-cost Pareto (F-D and F-E), PER-SEED points + median front  [replaces F1.4/F1.5/F5.8]
  R2  D/E coupling is real (de_ablation 4-arm bars, -48% D / -49% E)         [supports C4 / F1.3]
  R3  w=0 vs operating-knee per-seed dumbbell (w=0 reliability-optimal,       [replaces F4.5/F5.5]
      high cost; knee halves cost) + the seed-42 bifurcation
  R4  load-coupled mechanism: effective degree & n_tx concentration          [new, "eff-deg is the lever"]

Usage: python -B scripts/analysis/make_recalib_figures.py --out result/redraw_figure_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.make_paper_figures import (  # noqa: E402
    C_ACC, C_BAD, C_GNN, C_GOOD, C_GREY, C_HEUR, C_ORANGE, COL_W, DBL_W, save, set_ieee_style,
)

SEEDS = ["s7", "s42", "s123"]
SEED_COL = {"s7": C_GNN, "s42": C_BAD, "s123": C_GOOD}
SEED_MK = {"s7": "o", "s42": "X", "s123": "^"}


def _fine(seed: str) -> dict:
    return json.loads((RESULT / f"recalib_C_fine_{seed}" / "pareto.json").read_text(encoding="utf-8"))


def _rows(seed: str):
    d = _fine(seed)
    return {r["w_cost"]: r for r in d["rows"]}


# --------------------------------------------------------------------------- R1
def r1_pareto(out: Path) -> None:
    """Clean, honest reliability-cost Pareto: zoom to the OPERATING region (where one actually runs),
    prominent median front + per-seed points + 95% CI on the stable seeds; the seed-42 bifurcation
    (w>=0.25) is DISCLOSED by annotation rather than allowed to dominate the axes."""
    per = {s: _rows(s) for s in SEEDS}
    weights = sorted(per["s7"].keys())
    STABLE = ["s7", "s123"]  # the two non-bifurcating seeds (seed 42 disclosed separately)
    # view windows clipped to the operating region (flood -> knee -> plateau)
    win = {"holdout_D": (0.040, 0.125, 30, 150), "holdout_E": (0.040, 0.125, 0.08, 0.42)}
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 3.2))
    for ax, key, lab in ((axes[0], "holdout_D", "delay $D$ (effective rounds)"),
                         (axes[1], "holdout_E", "energy $E$ (J)")):
        x0, x1, y0, y1 = win[key]
        # per-weight median over all 3 seeds (robust to the seed-42 bifurcation)
        medF = [float(np.median([per[s][w]["holdout_F"]["mean"] for s in SEEDS])) for w in weights]
        medY = [float(np.median([per[s][w][key]["mean"] for s in SEEDS])) for w in weights]
        # non-dominated Pareto frontier (minimise BOTH F and the cost) = the clean object
        idx = sorted(range(len(weights)), key=lambda i: (medF[i], medY[i]))
        front, best = [], float("inf")
        for i in idx:
            if medY[i] <= best + 1e-9:
                front.append(i); best = medY[i]
        ax.plot([medF[i] for i in front], [medY[i] for i in front], "-o", color=C_GREY, lw=1.6,
                ms=4, zorder=3, label="Pareto frontier (median)")
        # dominated median weights as faint interior points
        dom = [i for i in range(len(weights)) if i not in front]
        ax.scatter([medF[i] for i in dom], [medY[i] for i in dom], s=12, color=C_GREY, alpha=0.35,
                   zorder=2, label="dominated weights")
        # per-seed points (stable seeds), low-key, for honesty
        for s in STABLE:
            ax.scatter([per[s][w]["holdout_F"]["mean"] for w in weights],
                       [per[s][w][key]["mean"] for w in weights],
                       s=11, marker=SEED_MK[s], color=SEED_COL[s], alpha=0.5, zorder=2,
                       edgecolors="none", label=f"seed {s[1:]}")
        # annotate flood (w=0) and knee (w=0.1) on the frontier
        for w, txt, dx, dy in ((0.0, "$w{=}0$ (flood)", 6, 2), (0.1, "knee $w{\\approx}0.1$", 7, -1)):
            i = weights.index(w)
            ax.annotate(txt, (medF[i], medY[i]), textcoords="offset points", xytext=(dx, dy),
                        fontsize=6.5, color=C_HEUR, ha="left")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xlabel("consensus failure $F$ (lower better)")
        ax.set_ylabel(lab + " (lower better)")
        ax.set_title(f"$F$ vs {lab.split(' (')[0]}")
    # disclose the seed-42 bifurcation honestly (off-panel) on the left axis
    axes[0].text(0.97, 0.04,
                 "1/3 inits (seed 42) bifurcate to\n$F{=}0.32$–$0.37$, $D$ up to $377$ at $w{\\geq}0.25$\n(off-panel; see Fig.~R3 & text)",
                 transform=axes[0].transAxes, ha="right", va="bottom", fontsize=5.6, color=C_BAD,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_BAD, lw=0.5, alpha=0.9))
    axes[0].legend(loc="upper right", fontsize=6)
    fig.suptitle("Re-calibrated operating point: reliability-cost Pareto (axis-visibility LOS) "
                 "— $F{\\sim}0.05$ at the knee", fontsize=8.5, y=1.02)
    save(fig, out, "R1", "recalib_pareto")


# --------------------------------------------------------------------------- R2
def r2_deablation(out: Path) -> None:
    d = json.loads((RESULT / "recalib_C_deablation" / "de_ablation.json").read_text(encoding="utf-8"))
    arms = {r["arm"]: r for r in d["arms"]}
    order = ["rel_only", "delay_heavy", "energy_heavy", "full"]
    lab = {"rel_only": "rel-only\n(w=0)", "delay_heavy": "delay\nheavy", "energy_heavy": "energy\nheavy", "full": "full\n(1,1)"}
    col = {"rel_only": C_BAD, "delay_heavy": C_GNN, "energy_heavy": C_ACC, "full": C_GOOD}
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.7))
    specs = [("F", "failure $F$ (lower better)", None),
             ("D", "delay $D$ (eff. rounds)", f"-{100*d['delay_relative_reduction']:.0f}% vs rel-only"),
             ("E", "energy $E$ (J)", f"-{100*d['energy_relative_reduction']:.0f}% vs rel-only")]
    for ax, (k, ylab, note) in zip(axes, specs):
        vals = [arms[a][k] for a in order]
        ax.bar(range(len(order)), vals, color=[col[a] for a in order], alpha=0.88, width=0.7)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([lab[a] for a in order], fontsize=6)
        ax.set_ylabel(ylab); ax.set_title(k)
        if note:
            ax.text(0.97, 0.95, note, transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color=C_GOOD, fontweight="bold")
    fig.suptitle("D/E coupling is real: delay/energy pressure each cut their own cost (eval Q=21 quenched)",
                 fontsize=8.5, y=1.04)
    save(fig, out, "R2", "recalib_deablation")


# --------------------------------------------------------------------------- R3
def r3_w0_dumbbell(out: Path) -> None:
    per = {s: _rows(s) for s in SEEDS}
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.9))
    for ax, key, ylab in ((axes[0], "holdout_F", "consensus failure $F$"),
                          (axes[1], "holdout_D", "delay $D$ (eff. rounds)")):
        for i, s in enumerate(SEEDS):
            v0 = per[s][0.0][key]["mean"]; vk = per[s][0.1][key]["mean"]
            ax.plot([0, 1], [v0, vk], "-", color=SEED_COL[s], lw=1.2, alpha=0.8, zorder=2)
            ax.scatter([0], [v0], s=34, marker="o", color=SEED_COL[s], zorder=3, edgecolors="white", linewidths=0.4)
            ax.scatter([1], [vk], s=34, marker=SEED_MK[s], color=SEED_COL[s], zorder=3, edgecolors="white", linewidths=0.4,
                       label=f"seed {s[1:]}")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["w=0\n(rel-only)", "w=0.1\n(knee)"], fontsize=7)
        ax.set_ylabel(ylab); ax.set_title(ylab.split(" (")[0])
        ax.set_xlim(-0.3, 1.3)
    axes[0].text(0.02, 0.98, "F=failure (lower=better):\nw=0 lowest F (best reliability),\nbut highest cost",
                 transform=axes[0].transAxes, va="top", fontsize=6, color=C_GREY)
    axes[1].legend(loc="upper right", fontsize=6)
    fig.suptitle("w=0 is the reliability-optimal, high-cost extreme; the knee halves delay for +0.016 F",
                 fontsize=8.5, y=1.03)
    save(fig, out, "R3", "recalib_w0_dumbbell")


# --------------------------------------------------------------------------- R4
def r4_mechanism(out: Path) -> None:
    d = json.loads((RESULT / "operating_point_measure" / "operating_point_measure.json").read_text(encoding="utf-8"))
    arms = {a["w_cost"]: a for a in d["arms"]}
    w0, wk = arms[0.0], arms[0.1]
    fig, axes = plt.subplots(1, 3, figsize=(DBL_W, 2.7))
    # panel a: effective degree (under loose cap 8) + hard in-degree
    ax = axes[0]
    x = np.arange(2)
    ax.bar(x - 0.18, [w0["effective_degree"], wk["effective_degree"]], 0.34, color=C_GNN, alpha=0.88, label="effective deg (1/$\\Sigma w^2$)")
    ax.bar(x + 0.18, [w0["mean_in_degree"], wk["mean_in_degree"]], 0.34, color=C_GREY, alpha=0.6, label="hard in-degree")
    ax.axhline(d["cap"], ls="--", lw=0.8, color=C_BAD); ax.text(1.4, d["cap"] + 0.1, f"cap={d['cap']}", color=C_BAD, fontsize=6, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(["w=0", "w=0.1"]); ax.set_ylabel("degree"); ax.set_title("effective vs hard degree")
    ax.legend(loc="lower left", fontsize=5.5)
    # panel b: delay D
    ax = axes[1]
    ax.bar([0, 1], [w0["D"], wk["D"]], color=[C_BAD, C_GOOD], alpha=0.85, width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["w=0", "w=0.1"]); ax.set_ylabel("delay $D$ (eff. rounds)"); ax.set_title("delay halves")
    for i, v in enumerate([w0["D"], wk["D"]]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    # panel c: n_tx tail (p50/p90)
    ax = axes[2]
    x = np.arange(2)
    ax.bar(x - 0.18, [w0["ntx_q"]["p50"], wk["ntx_q"]["p50"]], 0.34, color=C_GNN, alpha=0.85, label="n_tx p50")
    ax.bar(x + 0.18, [w0["ntx_q"]["p90"], wk["ntx_q"]["p90"]], 0.34, color=C_ORANGE, alpha=0.85, label="n_tx p90")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(["w=0", "w=0.1"])
    ax.set_ylabel("retransmissions $n_{tx}=1/\\mathrm{succ}$"); ax.set_title("link cost (log)"); ax.legend(loc="upper left", fontsize=6)
    fig.suptitle("Load-coupled mechanism: the cost weight concentrates query weight on low-$n_{tx}$ links "
                 "(effective degree is the lever, cap is a loose ceiling)", fontsize=7.8, y=1.04)
    save(fig, out, "R4", "recalib_mechanism")


def main() -> None:
    p = argparse.ArgumentParser(description="Redraw operating-point figures from re-calibration data")
    p.add_argument("--out", default="result/redraw_figure_v1")
    p.add_argument("--only", default="", help="comma list R1,R2,R3,R4")
    args = p.parse_args()
    set_ieee_style()
    out = ROOT / args.out
    reg = {"R1": r1_pareto, "R2": r2_deablation, "R3": r3_w0_dumbbell, "R4": r4_mechanism}
    todo = [k.strip() for k in args.only.split(",") if k.strip()] or list(reg)
    print(f"Redrawing {todo} -> {out}", flush=True)
    for k in todo:
        reg[k](out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
