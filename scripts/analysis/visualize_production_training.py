"""Publication-quality visualization for production training results.

Reads a production_training JSON (produced by run_production_training.py with
per-run ``trajectory`` series) and renders a 2x3 matplotlib figure that tells the
convergence story: loss / reliability / correctness trajectories at the focus
scale, the gradient trace, the gradient scale-law across N, and the final
reliability vs the from-scratch baseline.

Usage:
    python -B scripts/analysis/visualize_production_training.py \
        --json .agent/tmp/production_training_seed42.json \
        --out reports/training/figures/production_training_seed42.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
_REM = "#1f77b4"   # remediated (blue)
_BASE = "#d62728"  # baseline (red)


def _runs_by_key(payload: Mapping[str, Any]) -> dict[tuple[int, bool], dict[str, Any]]:
    return {(int(r["node_count"]), bool(r["remediated"])): r for r in payload["runs"]}


def _traj(run: Mapping[str, Any], key: str) -> tuple[list[int], list[float]]:
    t = run["trajectory"]
    return t["step"], t[key]


def render(payload: Mapping[str, Any], focus: int, out_path: Path) -> Path:
    by_key = _runs_by_key(payload)
    rem_scales = sorted({n for (n, r) in by_key if r})
    rem_focus = by_key.get((focus, True))
    base_focus = by_key.get((focus, False))
    verdict = payload.get("verdict_block", {})
    ready = bool(verdict.get("ready_for_production_training", False))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    seed = payload.get("seed", "?")
    fig.suptitle(
        f"V2X Topology-GNN — Production Training (focus N={focus}, seed={seed})    "
        f"verdict: {'READY' if ready else 'NOT READY'}",
        fontsize=15, fontweight="bold",
    )

    def _trajectory_panel(ax, key, title, ylabel, logy=False):
        if rem_focus is not None:
            xs, ys = _traj(rem_focus, key)
            ax.plot(xs, ys, color=_REM, lw=2.2, label="remediated (P0+P1+P2)")
        if base_focus is not None:
            xs, ys = _traj(base_focus, key)
            ax.plot(xs, ys, color=_BASE, lw=2.0, ls="--", label="baseline (off)")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("optimizer step")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # (0,0) loss
    _trajectory_panel(axes[0, 0], "total_loss", "Raw total loss", "loss")
    # (0,1) failure (log)
    _trajectory_panel(axes[0, 1], "F_mean", "Failure rate F (lower=better)", "F", logy=True)
    # (0,2) correctness
    _trajectory_panel(axes[0, 2], "C_avalanche_node_mean", "Consensus correctness C", "C")
    # (1,0) gradient norm (log)
    _trajectory_panel(axes[1, 0], "gradient_norm_total", "Gradient norm (training signal)", "‖grad‖", logy=True)

    # (1,1) gradient scale-law: grad0 vs N
    ax = axes[1, 1]
    rem_n = rem_scales
    rem_g0 = [by_key[(n, True)]["grad_norm_initial"] for n in rem_n]
    ax.plot(rem_n, rem_g0, "o-", color=_REM, lw=2.2, label="remediated grad₀")
    base_pts = sorted([(n, by_key[(n, False)]["grad_norm_initial"]) for (n, r) in by_key if not r])
    if base_pts:
        ax.plot([p[0] for p in base_pts], [p[1] for p in base_pts], "s--", color=_BASE, lw=2.0, label="baseline grad₀")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Gradient scale-law: initial ‖grad‖ vs N", fontsize=12)
    ax.set_xlabel("node count N (log)")
    ax.set_ylabel("initial ‖grad‖ (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # (1,2) final reliability bar
    ax = axes[1, 2]
    labels, finals, colors = [], [], []
    if rem_focus is not None:
        labels.append("remediated"); finals.append(rem_focus["F_final"]); colors.append(_REM)
    if base_focus is not None:
        labels.append("baseline"); finals.append(base_focus["F_final"]); colors.append(_BASE)
    bars = ax.bar(labels, finals, color=colors, alpha=0.85)
    ax.axhline(0.01, color="green", ls=":", lw=1.5, label="reliability target 0.01")
    ax.set_title(f"Final failure rate @ N={focus}", fontsize=12)
    ax.set_ylabel("F_final")
    for b, v in zip(bars, finals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize production training results")
    parser.add_argument("--json", default=".agent/tmp/production_training_seed42.json")
    parser.add_argument("--out", default="reports/training/figures/production_training_seed42.png")
    parser.add_argument("--focus-node-count", type=int, default=None)
    args = parser.parse_args()

    payload = json.loads((ROOT / args.json).read_text(encoding="utf-8"))
    focus = args.focus_node_count or int(payload.get("verdict_block", {}).get("focus_node_count", 10000))
    out = render(payload, focus, ROOT / args.out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
