"""Aggregate per-seed re-calibration Pareto fronts into a clean multi-seed front + figure.

Each run_pareto_frontier seed run already averages F/D/E over held-out SCENES (within-seed CI).
This combines several SEED runs (init variance) into a per-weight front with a cross-seed mean and
95% CI, so the headline front is not a single noisy init. Renders F-vs-D and F-vs-E fronts.

Usage:
  python -B scripts/analysis/aggregate_recalib_pareto.py \
    --runs recalib_C_fine_s7,recalib_C_fine_s42,recalib_C_fine_s123 --label C --out result/recalib_C_front
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402


def _ci95(vals):
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    n = a.size
    m = float(a.mean()) if n else float("nan")
    if n < 2:
        return {"mean": m, "ci": 0.0, "n": n, "vals": a.tolist()}
    half = float(stats.t.ppf(0.975, n - 1) * a.std(ddof=1) / np.sqrt(n))
    return {"mean": m, "ci": half, "n": n, "vals": a.tolist()}


def pareto_front(points):
    idx = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    front, best_y = [], float("inf")
    for i in idx:
        if points[i][1] <= best_y + 1e-12:
            front.append(i)
            best_y = points[i][1]
    return front


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate multi-seed re-calibration Pareto front")
    p.add_argument("--runs", required=True, help="comma-separated result/<run> dirs (per seed)")
    p.add_argument("--label", default="C")
    p.add_argument("--out", default="result/recalib_front")
    args = p.parse_args()

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    per_run = []
    for r in runs:
        path = ROOT / "result" / r / "pareto.json"
        if not path.exists():
            print(f"WARN missing {path}", flush=True)
            continue
        per_run.append(json.load(open(path)))
    if not per_run:
        raise SystemExit("no seed runs found")

    weights = [row["w_cost"] for row in per_run[0]["rows"]]
    agg = []
    for w in weights:
        Fs, Ds, Es = [], [], []
        for run in per_run:
            row = next((x for x in run["rows"] if x["w_cost"] == w), None)
            if row is None:
                continue
            Fs.append(row["holdout_F"]["mean"]); Ds.append(row["holdout_D"]["mean"]); Es.append(row["holdout_E"]["mean"])
        agg.append({"w_cost": w, "F": _ci95(Fs), "D": _ci95(Ds), "E": _ci95(Es)})

    print(f"=== Multi-seed front [{args.label}] over {len(per_run)} seeds ===", flush=True)
    print(f"{'w':>6} | {'F (mean+-CI)':>20} | {'D (mean+-CI)':>20} | {'E (mean+-CI)':>20}", flush=True)
    for a in agg:
        print(f"{a['w_cost']:>6} | {a['F']['mean']:>8.4f} +- {a['F']['ci']:<7.4f} | "
              f"{a['D']['mean']:>8.2f} +- {a['D']['ci']:<7.2f} | "
              f"{a['E']['mean']:>8.3e} +- {a['E']['ci']:<8.1e}", flush=True)

    # monotonicity diagnostics on the cross-seed means
    Fm = [a["F"]["mean"] for a in agg]
    Dm = [a["D"]["mean"] for a in agg]
    Em = [a["E"]["mean"] for a in agg]
    # sort by weight ascending (already) -> D should be non-increasing, F non-decreasing for a clean trade-off
    d_mono = all(Dm[i + 1] <= Dm[i] + max(0.05 * Dm[i], 1.0) for i in range(len(Dm) - 1))
    f_mono = all(Fm[i + 1] >= Fm[i] - max(0.05 * abs(Fm[i]), 0.01) for i in range(len(Fm) - 1))
    d_range = (max(Dm) / max(min(Dm), 1e-9))
    print(f"\nD non-increasing w/ w (tol): {d_mono} | F non-decreasing w/ w (tol): {f_mono} | "
          f"D dynamic range: {d_range:.1f}x", flush=True)

    out_dir = ROOT / args.out
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "front.json").write_text(json.dumps(
        {"label": args.label, "runs": runs, "n_seeds": len(per_run), "front": agg,
         "D_monotone": bool(d_mono), "F_monotone": bool(f_mono), "D_dynamic_range_x": float(d_range)},
        indent=2, sort_keys=True), encoding="utf-8")

    # figure: F vs D and F vs E, cross-seed error bars, Pareto front line
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.suptitle(f"Re-calibrated operating point [{args.label}] — coupled reliability-cost Pareto "
                 f"(axis-visibility LOS, {len(per_run)} seeds, mean +-95% CI)", fontsize=12, fontweight="bold")
    cmap = plt.get_cmap("viridis")
    wlist = [a["w_cost"] for a in agg]
    wn = lambda w: (w - min(wlist)) / (max(wlist) - min(wlist) + 1e-9)
    for ax, key, lab in ((axes[0], "D", "delay D (effective rounds)"), (axes[1], "E", "energy E (J)")):
        pts = [(a["F"]["mean"], a[key]["mean"]) for a in agg]
        fr = pareto_front(pts)
        ax.plot([pts[i][0] for i in fr], [pts[i][1] for i in fr], "-", color="0.6", lw=1.2, zorder=1)
        for a in agg:
            x, y = a["F"]["mean"], a[key]["mean"]
            ax.errorbar(x, y, xerr=a["F"]["ci"], yerr=a[key]["ci"], fmt="o", ms=8,
                        color=cmap(wn(a["w_cost"])), ecolor="gray", elinewidth=1, capsize=3, zorder=5)
            ax.annotate(f"w={a['w_cost']:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("failure F (lower=better)"); ax.set_ylabel(lab + " (lower=better)")
        ax.set_title(f"F vs {lab}", fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "figures" / "pareto_front.png", dpi=130)
    plt.close(fig)
    print(f"\nwrote {out_dir / 'front.json'} and figures/pareto_front.png", flush=True)


if __name__ == "__main__":
    main()
