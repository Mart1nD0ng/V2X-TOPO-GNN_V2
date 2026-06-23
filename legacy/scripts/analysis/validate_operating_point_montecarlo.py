"""Operating-point Monte-Carlo validation: is the HEADLINE reliability optimistic?

The N-sweep validation (validate_closed_form_montecarlo.py) showed the mean-field closed
form is optimistic about correctness C at low initial confidence, with the gap shrinking
as N grows. This script nails the question at the ACTUAL headline operating point: the real
V2X candidate graph at N (density 200 veh/km^2, 20 dB load coupling, hard_low_confidence
ic=0.40), the TRAINED planner's selected topology, and the EXACT per-edge link_success the
bridge computes (finite-blocklength + load-aware interference). It runs the closed-form
consensus the headline uses (extracted from the bridge's avalanche_details) and a faithful
Monte-Carlo of the same protocol, and reports how far the headline F is from the simulated F.

Relative comparisons (GNN vs heuristics, D/E ablation) use the same evaluator on both sides
and are unaffected by any bias; this only checks the ABSOLUTE reliability claim.

Outputs result/<run-name>/: op_validation.json + figures/op_validation.png + RESULT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.generalization_common import model_score  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer  # noqa: E402
from scripts.analysis.validate_closed_form_montecarlo import monte_carlo  # noqa: E402
from src.evaluation import evaluate_v2x_graph_consensus  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _avalanche_config,
    _evaluator_delay_config,
    _evaluator_energy_config,
    _evaluator_physical_config,
    _make_model,
    load_training_smoke_config,
)


def _select_topology(score_or_model, env, layer, caps):
    f = env["features"]
    score = score_or_model if isinstance(score_or_model, torch.Tensor) else model_score(score_or_model, env)
    return layer(num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
                 edge_score=score, per_node_budget=caps)


def _evaluate_details(topo, env, quenched_quadrature=None):
    f = env["features"]
    sel = topo.selected_candidate_index
    av_cfg = dict(_avalanche_config(env["cfg"]))
    if quenched_quadrature is not None:
        av_cfg["quenched_quadrature"] = int(quenched_quadrature)
    return evaluate_v2x_graph_consensus(
        **topo.as_evaluation_kwargs(),
        distance_m=f["distance_m"].index_select(0, sel), los_flag=f["los_flag"].index_select(0, sel),
        node_initial_correct=env["ic"], node_initial_wrong=env["iw"],
        physical_config=_evaluator_physical_config(env["cfg"]),
        avalanche_config=av_cfg,
        energy_config=_evaluator_energy_config(env["cfg"]),
        delay_config=_evaluator_delay_config(env["cfg"]),
        return_details=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Operating-point Monte-Carlo validation of the headline reliability")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--model", default="result/planner_paperenv/planner.pt",
                   help="trained planner checkpoint; falls back to nearest-k if missing")
    p.add_argument("--node-counts", default="500,1000,2000")
    p.add_argument("--scene-seed", type=int, default=7)
    p.add_argument("--trials", type=int, default=1200)
    p.add_argument("--quenched-quadrature", type=int, default=21,
                   help="SSMC quenched disorder copies for the headline closed form; >=21 is converged on "
                        "the skewed load-coupled support (1 = legacy mean-field, the ~40x-optimistic baseline)")
    p.add_argument("--run-name", default="operating_point_validation_v1")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    node_counts = [int(x) for x in str(args.node_counts).split(",") if x.strip()]
    rng = np.random.default_rng(int(args.scene_seed))

    model = None
    model_path = ROOT / args.model
    if model_path.exists():
        probe = _scene_env(base_config, node_counts[0], args.scene_seed)
        model = _make_model(probe["cfg"])
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        topo_src = f"trained planner ({args.model})"
    else:
        topo_src = "nearest-k heuristic (no checkpoint found)"
    print(f"Operating-point validation [{args.config}] topology={topo_src}; N={node_counts}; "
          f"{args.trials} MC trials/scene", flush=True)

    rows = []
    for n in node_counts:
        env = _scene_env(base_config, n, args.scene_seed)
        layer, caps = _topology_layer(env["cfg"], env["candidate"].num_nodes)
        if model is not None:
            with torch.no_grad():
                topo = _select_topology(model, env, layer, caps)
        else:
            topo = _select_topology(-env["features"]["distance_m"].to(dtype=torch.float64), env, layer, caps)
        ev = _evaluate_details(topo, env, quenched_quadrature=args.quenched_quadrature)
        ev_mf = _evaluate_details(topo, env, quenched_quadrature=1)  # legacy mean-field baseline
        av = ev["avalanche_details"]
        support = av["query_support"]
        cf_C = av["node_p_correct_decision"].detach().numpy()
        cf_F = (1.0 - av["node_p_correct_decision"]).detach().numpy()
        cf_D = av["node_expected_rounds"].detach().numpy()
        mf_F = (1.0 - ev_mf["avalanche_details"]["node_p_correct_decision"]).detach().numpy()
        link_s = ev["channel_diagnostics"]["link_success"].detach().to(torch.float64).numpy()
        rw = (support.normalized_query_weight.detach().to(torch.float64).numpy()) * link_s
        src = support.src_index.detach().numpy()
        dst = support.dst_index.detach().numpy()
        ic = env["ic"].detach().to(torch.float64).numpy()
        iw = env["iw"].detach().to(torch.float64).numpy()
        ava = _avalanche_config(env["cfg"])
        mc_C, mc_F, mc_D = monte_carlo(
            n, src, dst, rw, ic, iw,
            k=int(ava["k"]), alpha=int(ava["alpha"]), beta=int(ava["beta"]), rounds=int(ava["rounds"]),
            trials=args.trials, rng=rng)
        row = {
            "num_nodes": n,
            "quenched_quadrature": int(args.quenched_quadrature),
            "headline_F_closed_form": float(cf_F.mean()),     # SSMC quenched (converged Q)
            "meanfield_F_closed_form": float(mf_F.mean()),    # legacy Q=1 baseline
            "simulated_F_montecarlo": float(mc_F.mean()),
            "F_optimism_abs": float(mc_F.mean() - cf_F.mean()),
            "F_optimism_ratio": float(mc_F.mean() / max(cf_F.mean(), 1e-12)),
            "meanfield_optimism_abs": float(mc_F.mean() - mf_F.mean()),
            "meanfield_optimism_ratio": float(mc_F.mean() / max(mf_F.mean(), 1e-12)),
            "cf_C_mean": float(cf_C.mean()), "mc_C_mean": float(mc_C.mean()),
            "C_abs_err_mean": float(np.abs(cf_C - mc_C).mean()),
            "C_abs_err_max": float(np.abs(cf_C - mc_C).max()),
            "cf_D_mean": float(cf_D.mean()), "mc_D_mean": float(mc_D.mean()),
            "D_rel_err_mean": float((np.abs(cf_D - mc_D) / np.clip(mc_D, 1e-9, None)).mean()),
            "link_success_mean": float(link_s.mean()), "link_success_min": float(link_s.min()),
            "edge_count": int(src.size),
        }
        rows.append(row)
        print(f"  N={n:>5}: link mean={row['link_success_mean']:.3f}; "
              f"F(mean-field Q=1)={row['meanfield_F_closed_form']:.4f} ({row['meanfield_optimism_ratio']:.1f}x) -> "
              f"F(quenched Q={args.quenched_quadrature})={row['headline_F_closed_form']:.4f} "
              f"(opt +{row['F_optimism_abs']:.4f}, {row['F_optimism_ratio']:.2f}x)  "
              f"vs simulated F(MC)={row['simulated_F_montecarlo']:.4f}; D rel-err {row['D_rel_err_mean']*100:.1f}%",
              flush=True)

    summary = {"config": args.config, "topology_source": topo_src, "node_counts": node_counts,
               "trials": args.trials, "scene_seed": args.scene_seed, "rows": rows}
    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "op_validation.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _render(rows, out_dir / "figures" / "op_validation.png")
    _write_result(summary, out_dir / "RESULT.md")
    print(f"wrote {out_dir}")


def _render(rows, out_path: Path) -> None:
    ns = [r["num_nodes"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.suptitle("Operating-point validation: headline (closed-form) vs simulated (Monte-Carlo) reliability",
                 fontsize=12, fontweight="bold")
    ax = axes[0]
    x = range(len(ns)); wd = 0.38
    ax.bar([i - wd / 2 for i in x], [r["headline_F_closed_form"] for r in rows], width=wd,
           color="#1f77b4", label="headline F (closed-form)", alpha=0.85)
    ax.bar([i + wd / 2 for i in x], [r["simulated_F_montecarlo"] for r in rows], width=wd,
           color="#d62728", label="simulated F (Monte-Carlo)", alpha=0.85)
    ax.set_xticks(list(x)); ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("num nodes N"); ax.set_ylabel("failure F (lower=better)")
    ax.set_title("absolute reliability: closed-form vs simulated", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    ax = axes[1]
    ax.plot(ns, [r["F_optimism_abs"] for r in rows], "-o", color="#9467bd", label="F optimism (MC − closed)")
    ax.plot(ns, [r["C_abs_err_mean"] for r in rows], "-s", color="#2ca02c", label="C MAE")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("num nodes N"); ax.set_ylabel("closed-form optimism")
    ax.set_title("optimism vs N (should shrink toward 0)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_result(summary: dict, out_path: Path) -> None:
    lines = [
        "# Operating-Point Monte-Carlo Validation — Is the Headline Reliability Optimistic?",
        "",
        f"Config `{summary['config']}`, topology = {summary['topology_source']}, "
        f"{summary['trials']} MC trials/scene. Real candidate graph (density 200 veh/km^2, 20 dB "
        "coupling, hard_low_confidence ic=0.40), exact bridge link_success (finite-blocklength + "
        "load coupling). Harness: `scripts/analysis/validate_operating_point_montecarlo.py`.",
        "",
        "| N | headline F (closed-form) | simulated F (MC) | optimism (abs) | optimism (ratio) | D rel-err |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary["rows"]:
        lines.append(f"| {r['num_nodes']} | {r['headline_F_closed_form']:.4f} | "
                     f"{r['simulated_F_montecarlo']:.4f} | +{r['F_optimism_abs']:.4f} | "
                     f"{r['F_optimism_ratio']:.2f}x | {r['D_rel_err_mean']*100:.1f}% |")
    last = summary["rows"][-1]
    lines += [
        "",
        f"At the headline scale N={last['num_nodes']}: closed-form F = {last['headline_F_closed_form']:.4f}, "
        f"simulated F = {last['simulated_F_montecarlo']:.4f} — the headline understates failure by "
        f"{last['F_optimism_abs']:.4f} absolute ({last['F_optimism_ratio']:.2f}x). Delay D matches within "
        f"{last['D_rel_err_mean']*100:.1f}%.",
        "",
        "Interpretation: report absolute reliability as the *simulated* value (or a band closed-form→MC); "
        "the closed-form is the differentiable training surrogate. Relative claims (GNN vs heuristics, D/E "
        "ablation) are unaffected — both arms use the same surrogate.",
        "",
        "Artifacts: `op_validation.json`, `figures/op_validation.png`.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
