"""Publication figures for the envelope characterization (P0 seed bands + D1 floor lattice).

Fig 1 — GNN-advantage envelope with seed error bands: per-density panels of gap (mean +- seed sigma)
        vs the 0.010 advantage threshold, colored by robustness class (ROBUST / FRAGILE / PARITY /
        FLOOR_LIMITED). This is the "advantage region with error bands" headline figure.
Fig 2 — Floor-anchored feasibility: protocol floor (log scale) vs initial-confidence regime, with the
        F<=0.01 target line, showing the feasibility crossover at ic~0.90.

Usage: python -B scripts/analysis/plot_envelope.py --out result/envelope_figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

# class -> (color, label) for the legend
_CLASS_COLOR = {
    ("GNN_ADVANTAGE", True): ("#2e7d32", "GNN advantage (robust)"),
    ("GNN_ADVANTAGE", False): ("#ed6c02", "GNN advantage (fragile, gap-2σ<0.010)"),
    ("HEURISTIC_PARITY", None): ("#757575", "heuristic parity"),
    ("FLOOR_LIMITED", None): ("#1565c0", "floor-limited (graceful parity)"),
}
_PROF_SHORT = {"toy": "toy", "near_target_synthetic": "near", "hard_low_confidence": "hard"}
_IC = {"very_hard_low_confidence": 0.25, "hard_low_confidence": 0.40, "toy": 0.50,
       "near_target_synthetic": 0.65, "high_reliability_synthetic": 0.90}


def _color(cell):
    key = (cell["cell_class"], cell.get("label_robust"))
    if key in _CLASS_COLOR:
        return _CLASS_COLOR[key][0]
    # GNN_DEFICIT / TRAIN_DIVERGED fallbacks
    return "#b71c1c"


def fig_envelope(map_json: dict, out: Path) -> None:
    cells = map_json["cells"]
    densities = sorted({c["density"] for c in cells})
    fig, axes = plt.subplots(1, len(densities), figsize=(5.2 * len(densities), 4.6))
    if len(densities) == 1:
        axes = [axes]
    for ax, dens in zip(axes, densities):
        sub = sorted([c for c in cells if c["density"] == dens],
                     key=lambda c: (_PROF_SHORT.get(c["profile"], c["profile"]), c["coupling_db"]))
        xs = list(range(len(sub)))
        gaps = [c["gap_mean"] for c in sub]
        errs = [c["gap_std"] for c in sub]
        cols = [_color(c) for c in sub]
        for x, g, e, col in zip(xs, gaps, errs, cols):
            ax.errorbar(x, g, yerr=e, fmt="o", color=col, ecolor=col, elinewidth=2,
                        capsize=4, markersize=7, zorder=3)
        ax.axhline(0.010, ls="--", color="#444", lw=1.0, zorder=1)
        ax.text(len(sub) - 0.5, 0.010, " gap=0.010\n advantage thr.", va="bottom", ha="right",
                fontsize=8, color="#444")
        ax.axhline(0.0, ls="-", color="#ccc", lw=0.8, zorder=0)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{_PROF_SHORT.get(c['profile'], c['profile'])}\n{int(c['coupling_db'])}dB"
                            for c in sub], fontsize=8)
        ax.set_title(f"density {int(dens)} veh/km²", fontsize=11, fontweight="bold")
        ax.set_xlabel("profile / interference coupling", fontsize=9)
        ax.grid(axis="y", ls=":", alpha=0.4)
        ax.margins(x=0.06)
    axes[0].set_ylabel("GNN advantage   gap = bestH − F_gnn   (mean ± seed σ, 3 seeds)", fontsize=9)
    # shared legend
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=l)
               for (c, l) in _CLASS_COLOR.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("GNN-advantage envelope with seed error bands "
                 "(quenched Q=21, paper env, N=600)\n"
                 "robust advantage = 9 sparse cells; density-200 boundary fragile; density-300 floor-limited",
                 fontsize=12, y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "envelope_gap_bands.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "envelope_gap_bands.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_floor_feasibility(floors_by_ic: dict, out: Path) -> None:
    ics = sorted(floors_by_ic)
    vals = [floors_by_ic[ic] for ic in ics]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(ics, vals, "-o", color="#5e35b1", markersize=8, lw=2, zorder=3)
    for ic, v in zip(ics, vals):
        feasible = v <= 0.01
        ax.annotate(f"{v:.4f}", (ic, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8,
                    color=("#2e7d32" if feasible else "#b71c1c"))
    ax.axhline(0.01, ls="--", color="#444", lw=1.2)
    ax.text(0.255, 0.0115, "F ≤ 0.01 target", fontsize=9, color="#444", va="bottom")
    ax.fill_between([0.2, 0.95], 1e-3, 0.01, color="#2e7d32", alpha=0.07)
    ax.fill_between([0.2, 0.95], 0.01, 1.2, color="#b71c1c", alpha=0.05)
    ax.set_yscale("log")
    ax.set_xlim(0.2, 0.95)
    ax.set_ylim(1e-3, 1.2)
    ax.set_xlabel("initial-confidence regime  (node_initial_correct, ic)", fontsize=10)
    ax.set_ylabel("protocol floor  (perfect-link F, degree budget 4)", fontsize=10)
    ax.set_title("Floor-anchored feasibility: high reliability is gated by ic\n"
                 "F ≤ 0.01 becomes feasible only at ic ≈ 0.90 (floor 0.0048)", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out / "floor_feasibility.png", dpi=160, bbox_inches="tight")
    fig.savefig(out / "floor_feasibility.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="result/advantage_map/advantage_map.json")
    p.add_argument("--floor-table", default="result/protocol_floor_table/floor_table.json")
    p.add_argument("--out", default="result/envelope_figures")
    args = p.parse_args()

    map_json = json.loads((ROOT / args.map).read_text(encoding="utf-8"))
    floor_rows = json.loads((ROOT / args.floor_table).read_text(encoding="utf-8"))["rows"]
    floor_by_profile = {r["profile"]: r["floors"]["small_realistic (k5 a3 b5 r20)"]
                        for r in floor_rows if r["degree"] == 4}
    floors_by_ic = {_IC[prof]: f for prof, f in floor_by_profile.items() if prof in _IC}

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    fig_envelope(map_json, out)
    fig_floor_feasibility(floors_by_ic, out)
    print(f"wrote {out / 'envelope_gap_bands.png'}")
    print(f"wrote {out / 'floor_feasibility.png'}")
    print("floors_by_ic:", {k: round(v, 4) for k, v in sorted(floors_by_ic.items())})


if __name__ == "__main__":
    main()
