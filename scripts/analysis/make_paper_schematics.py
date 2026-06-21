"""IEEE-style schematic diagrams (block diagrams, no data) for the paper.

Reference style: serif type, restrained fills, thin dark borders, clean arrows. Produces:
  F0.1  method pipeline (snapshot -> ... -> evaluator -> coupled loss -> one .backward())
  F2.1  emission feedback loop (carried P(correct) re-grounds the recurrent state)
  F3.7  selection vs. unification (the two routes to serve the envelope)

Usage:  python -B scripts/analysis/make_paper_schematics.py [--out result/paper_figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from scripts.analysis.make_paper_figures import DBL_W, COL_W, save, set_ieee_style  # noqa: E402

# restrained, print-friendly fills
F_DATA = "#eceff1"     # data / observation blocks
F_LEARN = "#d6e4f0"    # learned / differentiable blocks
F_EVAL = "#dceede"     # analytic evaluator
F_LOSS = "#f6e0d6"     # loss
F_HL = "#fff3cd"       # highlight (emission)
EDGE = "#37474f"
GRAD = "#b71c1c"


def _box(ax, xy, w, h, text, fc=F_DATA, fs=7.5, ec=EDGE, lw=1.0, style="round,pad=0.02,rounding_size=0.06"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, color="#102027")


def _arrow(ax, p0, p1, color=EDGE, lw=1.1, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, lw=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}", zorder=1, linestyle=ls))


# =========================================================================== F0.1
def f0_1_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(DBL_W, 3.0))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")
    boxes = [
        ("Vehicle\nsnapshot", F_DATA),
        ("Candidate\ngraph", F_DATA),
        ("GNN edge\nscorer", F_LEARN),
        ("Hard top-$k$\nconstructor\n(straight-thru)", F_LEARN),
        ("Differentiable\nAvalanche\nevaluator $\\rightarrow$\nC / D / E", F_EVAL),
        ("Coupled loss\n$\\mathcal{L}(F,D,E)$", F_LOSS),
    ]
    w, h, gap, y = 14.0, 15, 2.4, 24
    xs = []
    for i, (txt, fc) in enumerate(boxes):
        x = 2 + i * (w + gap); xs.append(x)
        _box(ax, (x, y), w, h, txt, fc=fc, fs=6.9)
        if i:
            _arrow(ax, (xs[i - 1] + w, y + h / 2), (x, y + h / 2))
    # forward label (top)
    ax.text(2, y + h + 3.2, "forward", fontsize=7, style="italic", color=EDGE)
    _arrow(ax, (xs[0], y + h + 2.4), (xs[-1] + w, y + h + 2.4), color=EDGE, lw=0.8, style="-|>")
    # gradient feedback: clean 3-segment loop BELOW the boxes (loss -> scorer)
    xl = xs[5] + w / 2; xg = xs[2] + w / 2; yb = 12
    _arrow(ax, (xl, y), (xl, yb), color=GRAD, lw=1.3, style="-")
    _arrow(ax, (xl, yb), (xg, yb), color=GRAD, lw=1.3, style="-")
    _arrow(ax, (xg, yb), (xg, y), color=GRAD, lw=1.3, style="-|>")
    ax.text((xg + xl) / 2, yb - 4.2, "single $\\nabla$ through one  .backward()   —   no labels",
            ha="center", fontsize=7.6, color=GRAD, fontweight="bold")
    ax.set_title("End-to-end differentiable topology constructor", fontsize=9, fontweight="bold")
    save(fig, out, "F0.1", "pipeline_schematic")


# =========================================================================== F2.1
def f2_1_emission(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(COL_W * 1.55, 2.9))
    ax.set_xlim(0, 100); ax.set_ylim(0, 60); ax.axis("off")
    # frame t blocks
    _box(ax, (4, 38), 26, 13, "graph-GRU\n(recurrent scorer)", fc=F_LEARN)
    _box(ax, (40, 38), 26, 13, "recurrent state\n$h_t$", fc=F_DATA)
    _box(ax, (74, 38), 22, 13, "edge scores\n$\\rightarrow$ topology", fc=F_LEARN)
    _box(ax, (40, 8), 26, 13, "carried reliability\n$\\hat P(\\mathrm{correct})$  (detached)", fc=F_HL)
    _box(ax, (4, 8), 26, 13, "node feature\n(6th channel)", fc=F_HL)
    # forward chain
    _arrow(ax, (30, 44.5), (40, 44.5))
    _arrow(ax, (66, 44.5), (74, 44.5))
    # state -> next-frame input (recurrence) top arc
    _arrow(ax, (53, 51), (17, 51), color=EDGE, lw=1.0, rad=0.4)
    ax.text(35, 57, "recurrence ($t\\!\\rightarrow\\!t{+}1$)", ha="center", fontsize=6.8, color=EDGE)
    # emission loop: scores/topology -> P(correct) -> node feature -> GRU
    _arrow(ax, (85, 38), (66, 14.5), color=GRAD, lw=1.2, rad=-0.25)
    _arrow(ax, (40, 14.5), (30, 14.5), color=GRAD, lw=1.2)
    _arrow(ax, (17, 21), (17, 38), color=GRAD, lw=1.2)
    ax.text(50, 2.0, "emission: re-grounds $h_t$ in a calibrated [0,1] observation each frame\n"
                     "$\\Rightarrow$ bounds the gate input $\\Rightarrow$ gradient survives $\\Rightarrow$ escapes collapse",
            ha="center", fontsize=6.8, color=GRAD)
    ax.set_title("Emission-grounded recurrence stabilization", fontsize=9, fontweight="bold")
    save(fig, out, "F2.1", "emission_loop_schematic")


# =========================================================================== F3.7
def f3_7_routes(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(DBL_W, 2.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
    # shared substrate
    _box(ax, (34, 40), 32, 8, "Advantage map (§1.3)\nseed-banded operating envelope", fc=F_EVAL, fs=7)
    # left route: selection / gating
    _box(ax, (4, 20), 40, 13, "SELECTION  —  runtime gating (§3.1)\n"
                              "estimate context (density, SINR proxy)\n$\\rightarrow$ route GNN vs. heuristic per frame", fc=F_LEARN, fs=6.8)
    # right route: unification / generalist
    _box(ax, (56, 20), 40, 13, "UNIFICATION  —  governed generalist (§3.2)\n"
                               "one model over all cells +\ngradient governance (GradNorm)", fc=F_LEARN, fs=6.8)
    _arrow(ax, (44, 40), (24, 33), rad=0.1)
    _arrow(ax, (56, 40), (76, 33), rad=-0.1)
    # trade-off captions
    _box(ax, (4, 3), 40, 12, "+ zero deploy-time training\n+ interpretable routing\n− needs expert/heuristic set\n− only as good as estimable context", fc=F_DATA, fs=6.3)
    _box(ax, (56, 3), 40, 12, "+ single artifact, no routing\n+ interpolates off-grid\n− offline governance step\n− hides per-cell rationale", fc=F_DATA, fs=6.3)
    _arrow(ax, (24, 20), (24, 15), lw=0.8); _arrow(ax, (76, 20), (76, 15), lw=0.8)
    ax.set_title("Two routes to serve the operating envelope", fontsize=9, fontweight="bold")
    save(fig, out, "F3.7", "selection_vs_unification")


SCHEMS = {"F0.1": f0_1_pipeline, "F2.1": f2_1_emission, "F3.7": f3_7_routes}


def main() -> None:
    p = argparse.ArgumentParser(description="IEEE-style schematics")
    p.add_argument("--only", default="")
    p.add_argument("--out", default="result/paper_figures")
    args = p.parse_args()
    set_ieee_style()
    out = ROOT / args.out
    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(SCHEMS)
    for fid in want:
        SCHEMS[fid](out)
    print(f"done: {len(want)} schematics -> {out}")


if __name__ == "__main__":
    main()
