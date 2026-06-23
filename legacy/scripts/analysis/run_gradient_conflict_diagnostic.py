"""Phase 0 — gradient-conflict & leverage diagnostic.

Design contract: docs/GRADIENT_GOVERNANCE_DESIGN.md (Phase 0). Trains the `full` arm
(w_R=w_D=w_E=1) at the Track-A operating point and logs, every step, the per-task
gradient norms ||dL_R||, ||dL_D||, ||dL_E|| w.r.t. the constructor parameters and the
pairwise cosines cos(D,E), cos(D,R), cos(R,E), alongside F/C/D/E and the gate. It then
emits the two Phase-0 verdicts:

  P0-G1  (for D-S, stability):  is the seed-123 `full`-arm divergence caused by gradient
          CONFLICT (a negative inter-task cosine coinciding with F blow-up)? If yes,
          PCGrad/GradNorm is the right fix. If F diverges with no negative cosine, the
          cause is LR/scale, not conflict.
  P0-G2  (for D-C, controllability):  does D have any topology LEVERAGE? D-diag-A if
          ||dL_D|| >= 0.10*max(||dL_R||,||dL_E||) for >=30% of steps (suppressed leverage,
          governance may help); D-diag-B otherwise (~zero leverage, governance cannot make
          D controllable -> structural delay model needed).

Pure path: this is a diagnostic; it does not change the default training path. The
optimizer step here reuses the three per-task gradient vectors (their sum == the normal
combined gradient == effective_backward_loss.backward()), so logging is free of an extra
backward pass and exactly mirrors the gradient Track G1 will govern.

Outputs result/<run-name>/: conflict.json + figures/cos_and_norms.png + RESULT.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.run_de_ablation import _forward  # noqa: E402
from src.losses import compute_coupled_loss, compute_task_gradient_vectors  # noqa: E402
from src.topology import TopologyConstructionLayer  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _initial_preferences,
    _loss_config,
    _make_environment,
    _make_model,
    _normalized_config,
    load_training_smoke_config,
)

TASKS = ("R", "D", "E")


def _metrics_detached(ev) -> dict[str, float]:
    # Same C/F/D/E as run_de_ablation._metrics, but detached: here the forward graph is
    # kept alive for the per-task gradients, so float() on a grad tensor would warn.
    return {
        "C": float(ev["C_avalanche_node_mean"].mean().detach()),
        "F": float(ev["F_avalanche_node_mean"].mean().detach()),
        "D": float(ev["D_avalanche_rounds_mean"].mean().detach()),
        "E": float(ev["E_consensus_node_mean"].mean().detach()),
    }


# --------------------------------------------------------------------------- #
# Pure, testable verdict helpers
# --------------------------------------------------------------------------- #
def _cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    norm_u = float(u.norm())
    norm_v = float(v.norm())
    if norm_u < eps or norm_v < eps:
        return 0.0
    return float(torch.dot(u, v) / (u.norm() * v.norm()))


def _leverage_fraction(ratios: Sequence[float], thresh: float = 0.10) -> float:
    """Fraction of steps where ||dL_D|| >= thresh * max(||dL_R||, ||dL_E||)."""
    vals = [r for r in ratios if r == r]  # drop NaN
    if not vals:
        return 0.0
    return sum(1 for r in vals if r >= thresh) / len(vals)


def _leverage_verdict(frac: float, frac_a: float = 0.30) -> str:
    return "D-diag-A" if frac >= frac_a else "D-diag-B"


def _diverged(final_failure: float, band: float = 0.10) -> bool:
    # NaN (failed finite check) counts as diverged.
    return (final_failure != final_failure) or (final_failure > band)


def _conflict_confirmed(diverged: bool, min_cosine: float) -> bool:
    """A diverged arm whose per-task gradients went into conflict (negative cosine)."""
    return bool(diverged and min_cosine < 0.0)


# --------------------------------------------------------------------------- #
# Per-seed run
# --------------------------------------------------------------------------- #
def _run_seed(seed: int, args) -> dict:
    config = dict(load_training_smoke_config(str(ROOT / args.config)))
    config["vehicle_count"] = int(args.node_count)
    config["seed"] = int(seed)
    cfg = _normalized_config(config)
    candidate, features = _make_environment(cfg)
    ic, iw = _initial_preferences(cfg, candidate.num_nodes)
    env = {"cfg": cfg, "candidate": candidate, "features": features, "ic": ic, "iw": iw}

    budget = None if cfg["max_out_degree"] is None else int(cfg["max_out_degree"])
    topology_layer = TopologyConstructionLayer(
        max_out_degree=budget, support_mode=str(cfg["support_mode"]), temperature=1.0,
        topk_backend=str(cfg["topk_backend"]),
        gradient_mode=str(cfg.get("gradient_mode", "selected_row_softmax")),
        straight_through_temperature=cfg.get("straight_through_temperature", None),
    )
    fixed_caps = torch.full((candidate.num_nodes,), budget, dtype=torch.long) if budget else None

    loss_cfg = dict(_loss_config(cfg))
    loss_cfg["reliability_failure_target"] = float(args.reliability_target)
    loss_cfg["reliability_tail_failure_target"] = float(args.reliability_target)
    loss_cfg["use_reliability_gate"] = True
    loss_cfg["weight_reliability"] = 1.0  # the `full` arm: all three objectives active
    loss_cfg["weight_delay"] = 1.0
    loss_cfg["weight_energy"] = 1.0

    torch.manual_seed(int(seed))
    model = _make_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))

    steps: list[dict] = []
    for step in range(int(args.max_steps)):
        opt.zero_grad(set_to_none=True)
        ev = _forward(model, env, topology_layer, fixed_caps)
        lo = compute_coupled_loss(ev, loss_cfg)
        scale = lo["scale_backward_multiplier"]
        # The ACTUAL training gradients: weighted, scale-invariant per-task losses.
        task_losses = {
            "R": lo["weighted_L_R"] * scale,
            "D": lo["weighted_L_D"] * scale,
            "E": lo["weighted_L_E"] * scale,
        }
        vecs = compute_task_gradient_vectors(task_losses, params)
        norms = {t: float(vecs[t].norm()) for t in TASKS}
        denom = max(norms["R"], norms["E"]) if max(norms["R"], norms["E"]) > 0 else float("nan")
        ratio_d = norms["D"] / denom if denom == denom and denom > 0 else float("nan")
        m = _metrics_detached(ev)
        steps.append({
            "step": step,
            "F": m["F"], "C": m["C"], "D": m["D"], "E": m["E"],
            "gate": bool(lo["reliability_loss_should_weaken"]),
            "norm_R": norms["R"], "norm_D": norms["D"], "norm_E": norms["E"],
            "ratio_D": ratio_d,
            "cos_DE": _cosine(vecs["D"], vecs["E"]),
            "cos_DR": _cosine(vecs["D"], vecs["R"]),
            "cos_RE": _cosine(vecs["R"], vecs["E"]),
        })
        # Optimizer step: sum of per-task grads == the normal combined gradient
        # (== effective_backward_loss.backward()). Reuses the vectors -> no extra pass.
        summed = vecs["R"] + vecs["D"] + vecs["E"]
        if not bool(torch.isfinite(summed).all()):
            steps[-1]["nonfinite_grad"] = True
            break
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = summed[offset:offset + n].view_as(p).clone()
            offset += n
        opt.step()

    final_f = steps[-1]["F"] if steps else float("nan")
    ratios = [s["ratio_D"] for s in steps]
    frac_lev = _leverage_fraction(ratios, args.leverage_threshold)
    min_cos = min(
        [min(s["cos_DE"], s["cos_DR"], s["cos_RE"]) for s in steps],
        default=0.0,
    )
    diverged = _diverged(final_f, args.divergence_band)
    neg_cos_frac = (
        sum(1 for s in steps if min(s["cos_DE"], s["cos_DR"], s["cos_RE"]) < 0.0) / len(steps)
        if steps else 0.0
    )
    return {
        "seed": int(seed),
        "n_steps": len(steps),
        "final_F": final_f,
        "max_F": max((s["F"] for s in steps), default=float("nan")),
        "diverged": bool(diverged),
        "frac_D_leverage": frac_lev,
        "leverage_verdict": _leverage_verdict(frac_lev, args.leverage_frac_a),
        "min_cosine": min_cos,
        "neg_cosine_fraction": neg_cos_frac,
        "conflict_confirmed": _conflict_confirmed(diverged, min_cos),
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 0 gradient-conflict & leverage diagnostic")
    parser.add_argument("--config", default="configs/operating_point_v1.yaml")
    parser.add_argument("--node-count", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--reliability-target", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 123])
    parser.add_argument("--leverage-threshold", type=float, default=0.10)
    parser.add_argument("--leverage-frac-a", type=float, default=0.30)
    parser.add_argument("--divergence-band", type=float, default=0.10)
    parser.add_argument("--run-name", default="gradient_conflict_v1")
    args = parser.parse_args()

    print(f"Phase 0 gradient-conflict diagnostic (N={args.node_count}, steps={args.max_steps}, "
          f"rel target={args.reliability_target}, seeds={args.seeds})")
    per_seed = []
    for seed in args.seeds:
        rec = _run_seed(seed, args)
        per_seed.append(rec)
        print(f"  seed {seed:>4}: final_F={rec['final_F']:.4f} diverged={rec['diverged']} | "
              f"frac_D_leverage={rec['frac_D_leverage']:.2f} -> {rec['leverage_verdict']} | "
              f"min_cos={rec['min_cosine']:+.3f} neg_cos_frac={rec['neg_cosine_fraction']:.2f} "
              f"conflict={rec['conflict_confirmed']}")

    diverged_seeds = [r["seed"] for r in per_seed if r["diverged"]]
    conflict_seeds = [r["seed"] for r in per_seed if r["conflict_confirmed"]]
    # P0-G1: among diverged arms, is conflict (negative cosine) the mechanism?
    if not diverged_seeds:
        p0g1 = "NO DIVERGENCE OBSERVED (stability defect did not reproduce at this budget)"
        p0g1_pass = None
    elif conflict_seeds:
        p0g1 = (f"CONFLICT CONFIRMED on diverged seed(s) {conflict_seeds}: negative inter-task "
                f"cosine coincides with F blow-up -> PCGrad/GradNorm is the right D-S fix (Track G1).")
        p0g1_pass = True
    else:
        p0g1 = (f"DIVERGENCE WITHOUT CONFLICT on seed(s) {diverged_seeds}: no negative cosine -> "
                f"cause is LR/scale, not gradient conflict; PCGrad not expected to help (G1 off-ramp).")
        p0g1_pass = False

    # P0-G2: aggregate D-leverage verdict (D-diag-A only if a majority of seeds show leverage).
    a_seeds = [r["seed"] for r in per_seed if r["leverage_verdict"] == "D-diag-A"]
    overall_leverage = "D-diag-A" if len(a_seeds) * 2 >= len(per_seed) and per_seed else "D-diag-B"
    if overall_leverage == "D-diag-A":
        p0g2 = (f"D-diag-A: D carries suppressed gradient leverage on seed(s) {a_seeds} "
                f"-> Track G2 (governed D-control) is viable; measure if PCGrad/GradNorm frees D.")
    else:
        p0g2 = ("D-diag-B: ||dL_D|| is ~negligible vs ||dL_R||/||dL_E|| -> the topology has ~no "
                "gradient lever on D. Gradient governance CANNOT make D controllable; Track G2 "
                "off-ramps to a structural delay model (hop-count / path-length). 'Coupled C/E' stands.")

    summary = {
        "config": args.config, "node_count": args.node_count, "max_steps": args.max_steps,
        "reliability_target": args.reliability_target, "seeds": list(args.seeds),
        "leverage_threshold": args.leverage_threshold, "leverage_frac_a": args.leverage_frac_a,
        "divergence_band": args.divergence_band,
        "per_seed": [{k: v for k, v in r.items() if k != "steps"} for r in per_seed],
        "P0_G1_conflict": {"diverged_seeds": diverged_seeds, "conflict_seeds": conflict_seeds,
                           "pass": p0g1_pass, "verdict": p0g1},
        "P0_G2_leverage": {"D_diag_A_seeds": a_seeds, "overall": overall_leverage, "verdict": p0g2},
    }

    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    # full traces in a sibling file (kept out of the summary to keep it readable)
    (out_dir / "conflict_traces.json").write_text(
        json.dumps({str(r["seed"]): r["steps"] for r in per_seed}, indent=2), encoding="utf-8")
    (out_dir / "conflict.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _render(per_seed, out_dir / "figures" / "cos_and_norms.png")
    _write_result_md(summary, out_dir / "RESULT.md")

    print(f"\nP0-G1 (stability cause): {p0g1}")
    print(f"P0-G2 (D leverage):      {p0g2}")


def _render(per_seed: list[dict], out_path: Path) -> None:
    n = len(per_seed)
    fig, axes = plt.subplots(n, 3, figsize=(15, 3.6 * max(n, 1)), squeeze=False)
    for row, rec in enumerate(per_seed):
        steps = rec["steps"]
        xs = [s["step"] for s in steps]
        ax_f, ax_n, ax_c = axes[row]
        ax_f.plot(xs, [s["F"] for s in steps], color="#d62728")
        ax_f.axhline(0.10, color="grey", ls=":", lw=1, label="band edge 0.10")
        ax_f.set_title(f"seed {rec['seed']}  F (diverged={rec['diverged']})")
        ax_f.set_yscale("log"); ax_f.grid(True, alpha=0.3); ax_f.legend(fontsize=7)
        ax_n.plot(xs, [s["norm_R"] for s in steps], label="||dL_R||", color="#1f77b4")
        ax_n.plot(xs, [s["norm_D"] for s in steps], label="||dL_D||", color="#2ca02c")
        ax_n.plot(xs, [s["norm_E"] for s in steps], label="||dL_E||", color="#9467bd")
        ax_n.set_title(f"grad norms  (frac_D_leverage={rec['frac_D_leverage']:.2f} -> {rec['leverage_verdict']})")
        ax_n.set_yscale("log"); ax_n.grid(True, alpha=0.3); ax_n.legend(fontsize=7)
        ax_c.plot(xs, [s["cos_DE"] for s in steps], label="cos(D,E)", color="#ff7f0e")
        ax_c.plot(xs, [s["cos_DR"] for s in steps], label="cos(D,R)", color="#8c564b")
        ax_c.plot(xs, [s["cos_RE"] for s in steps], label="cos(R,E)", color="#17becf")
        ax_c.axhline(0.0, color="black", lw=0.8)
        ax_c.set_title(f"task cosines (min={rec['min_cosine']:+.2f}, conflict={rec['conflict_confirmed']})")
        ax_c.set_ylim(-1.05, 1.05); ax_c.grid(True, alpha=0.3); ax_c.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _write_result_md(summary: dict, out_path: Path) -> None:
    rows = "\n".join(
        f"| {r['seed']} | {r['final_F']:.4f} | {r['diverged']} | {r['frac_D_leverage']:.2f} | "
        f"{r['leverage_verdict']} | {r['min_cosine']:+.3f} | {r['neg_cosine_fraction']:.2f} | "
        f"{r['conflict_confirmed']} |"
        for r in summary["per_seed"]
    )
    text = f"""# Phase 0 — Gradient-Conflict & Leverage Diagnostic: Result

Design contract: `docs/GRADIENT_GOVERNANCE_DESIGN.md` (Phase 0). Harness:
`scripts/analysis/run_gradient_conflict_diagnostic.py` (`make gradient-conflict-diagnostic`).
Trains the `full` arm (w_R=w_D=w_E=1) at `{summary['config']}` (N={summary['node_count']},
{summary['max_steps']} steps, rel target {summary['reliability_target']}), logging per-task
gradient norms and pairwise cosines every step.

## P0-G1 (D-S stability cause)

{summary['P0_G1_conflict']['verdict']}

## P0-G2 (D-C leverage)

{summary['P0_G2_leverage']['verdict']}

## Per-seed table

| seed | final_F | diverged | frac_D_leverage | leverage | min_cosine | neg_cos_frac | conflict |
|---|---:|:--:|---:|---|---:|---:|:--:|
{rows}

`frac_D_leverage` = fraction of steps with ||dL_D|| >= {summary['leverage_threshold']}*max(||dL_R||,||dL_E||);
D-diag-A iff >= {summary['leverage_frac_a']}. `conflict` = diverged AND some inter-task cosine < 0.

Artifacts: `conflict.json` (summary), `conflict_traces.json` (full per-step traces),
`figures/cos_and_norms.png` (F / grad-norms / cosines per seed).
"""
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
