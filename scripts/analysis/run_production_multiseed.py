"""Multi-seed production training validation + aggregate visualization.

Runs the remediated production pipeline (and the remediation-off baseline) at the
production scale across several seeds, aggregates the final reliability and the
training signal, and emits an overall production-readiness verdict plus an
aggregate figure. This is the multi-seed validation recommended after the single
-seed seed=42 run.

Usage:
    python -B scripts/analysis/run_production_multiseed.py \
        --seeds 7,42,123 --node-count 10000 --max-steps 100
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.run_production_training import (  # noqa: E402
    _run_one,
    _verdict,
    load_training_smoke_config,
)

_REM = "#1f77b4"
_BASE = "#d62728"


def _agg(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def render(results: list[dict], node_count: int, target: float, out_path: Path) -> Path:
    seeds = [r["seed"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    overall_ready = all(r["verdict"]["ready_for_production_training"] for r in results)
    fig.suptitle(
        f"Multi-seed production training (N={node_count}, seeds={seeds})    "
        f"overall: {'ALL READY' if overall_ready else 'NOT ALL READY'}",
        fontsize=14, fontweight="bold",
    )

    # Left: F trajectory per seed (remediated), log y.
    ax = axes[0]
    for r in results:
        t = r["remediated"]["trajectory"]
        ax.plot(t["step"], t["F_mean"], lw=1.8, label=f"seed {r['seed']}")
    ax.axhline(target, color="green", ls=":", lw=1.5, label=f"target {target}")
    ax.set_yscale("log")
    ax.set_title("Remediated failure rate F vs step (per seed)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("F (log)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Right: final F per seed, remediated vs baseline (grouped bars).
    ax = axes[1]
    x = range(len(seeds))
    w = 0.38
    rem_F = [r["remediated"]["F_final"] for r in results]
    base_F = [r["baseline"]["F_final"] for r in results]
    ax.bar([i - w / 2 for i in x], rem_F, width=w, color=_REM, label="remediated", alpha=0.85)
    ax.bar([i + w / 2 for i in x], base_F, width=w, color=_BASE, label="baseline", alpha=0.85)
    ax.axhline(target, color="green", ls=":", lw=1.5, label=f"target {target}")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed {s}" for s in seeds])
    ax.set_title(f"Final failure rate F @ N={node_count}")
    ax.set_ylabel("F_final")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed production training")
    parser.add_argument("--config", default="configs/production_training_v1.yaml")
    parser.add_argument("--seeds", default="7,42,123")
    parser.add_argument("--node-count", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--reliability-target", type=float, default=0.01)
    parser.add_argument("--optimizer", default=None, help="override optimizer for ALL runs (sgd|adam)")
    parser.add_argument("--learning-rate", type=float, default=None, help="override lr for ALL runs")
    # P0-1 currency: Q=1 = mean-field surrogate (banned reliability conclusion); Q>=21 quenched.
    parser.add_argument("--quench", type=int, default=None, help="training quenched_quadrature override")
    parser.add_argument("--eval-quench", type=int, default=None, help="headline-eval quadrature override")
    parser.add_argument(
        "--run-name",
        default=None,
        help="if set, write multiseed.json + figures/multiseed.png under result/<run-name>/",
    )
    parser.add_argument("--json-out", default=".agent/tmp/production_training_multiseed.json")
    parser.add_argument("--fig-out", default="reports/training/figures/production_training_multiseed.png")
    args = parser.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    if args.optimizer is not None:
        base["optimizer"] = str(args.optimizer)
    if args.learning_rate is not None:
        base["learning_rate"] = float(args.learning_rate)
    if args.quench is not None:
        base["quenched_quadrature"] = int(args.quench)
    if args.eval_quench is not None:
        base["eval_quenched_quadrature"] = int(args.eval_quench)
    eval_q = int(base.get("eval_quenched_quadrature", base.get("quenched_quadrature", 1)))
    currency = "mean_field_surrogate" if eval_q <= 1 else f"quenched_Q{eval_q}"
    print(f"[currency] headline F_final at Q={eval_q} -> {currency}", flush=True)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    target = float(args.reliability_target)

    results: list[dict] = []
    for seed in seeds:
        cfg = dict(base)
        cfg["seed"] = seed
        rem = _run_one(cfg, args.node_count, args.max_steps, remediated=True)
        base_run = _run_one(cfg, args.node_count, args.max_steps, remediated=False)
        verdict = _verdict([rem, base_run], focus_node_count=args.node_count, reliability_target=target)
        results.append({"seed": seed, "remediated": rem, "baseline": base_run, "verdict": verdict})
        print(
            f"seed {seed:4d}: remediated F={rem['F_final']:.4f} C={rem['C_final']:.4f} grad0={rem['grad_norm_initial']:.1f} "
            f"ready={verdict['ready_for_production_training']}",
            flush=True,
        )

    rem_F = [r["remediated"]["F_final"] for r in results]
    rem_C = [r["remediated"]["C_final"] for r in results]
    rem_g0 = [r["remediated"]["grad_norm_initial"] for r in results]
    overall_ready = all(r["verdict"]["ready_for_production_training"] for r in results)
    summary = {
        "config": args.config,
        "node_count": args.node_count,
        "max_steps": args.max_steps,
        "seeds": seeds,
        "reliability_target": target,
        "evaluator_currency": currency,
        "eval_quench": eval_q,
        "train_quench": int(base.get("quenched_quadrature", 1)),
        "remediated_F_final": _agg(rem_F),
        "remediated_C_final": _agg(rem_C),
        "remediated_grad0": _agg(rem_g0),
        "per_seed_ready": {r["seed"]: r["verdict"]["ready_for_production_training"] for r in results},
        "overall_ready_for_production_training": overall_ready,
        "results": results,
    }
    if args.run_name:
        out_dir = ROOT / "result" / args.run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "multiseed.json"
        fig_path = out_dir / "figures" / "multiseed.png"
    else:
        json_path = ROOT / args.json_out
        json_path.parent.mkdir(parents=True, exist_ok=True)
        fig_path = ROOT / args.fig_out
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    fig = render(results, args.node_count, target, fig_path)

    fF = summary["remediated_F_final"]
    print(
        f"\nAGGREGATE @ N={args.node_count}: F_final={fF['mean']:.4f}±{fF['std']:.4f} "
        f"(min {fF['min']:.4f}, max {fF['max']:.4f})"
    )
    print(f"overall_ready_for_production_training: {overall_ready}")
    print(f"figure: {fig}")


if __name__ == "__main__":
    main()
