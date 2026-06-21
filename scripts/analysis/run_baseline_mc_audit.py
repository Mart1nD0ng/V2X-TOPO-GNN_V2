"""Monte-Carlo ground-truth audit of the baseline comparison.

The headline F-table is computed with the quenched CLOSED FORM (Q=21). A reviewer asks: does the
ranking survive the actual sampled consensus (Monte-Carlo), or is it a closed-form artifact? This
re-evaluates a representative set of methods on one held-out scene under BOTH the closed form and
Monte-Carlo, and checks the ordering agrees.

Methods audited: proposed (hierarchical), mean-field-trained proposed, best-channel-k heuristic,
no-planning flood, informed greedy. All through the SAME constructor; MC uses the SAME per-edge
link success x query weight as the closed form (validate_effective_degree pattern).

Usage:
  python -B scripts/analysis/run_baseline_mc_audit.py --config configs/operating_point_v1.yaml \
    --node-count 600 --scene-seed 1007 --trials 1500
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

from scripts.analysis.run_baseline_comparison import VanillaGNNScorer, _heuristic_scores, _model_score  # noqa: E402
from scripts.analysis.run_baseline_comparison_v2 import _flood_layer, _oracle_score, _train  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer  # noqa: E402
from scripts.analysis.validate_closed_form_montecarlo import monte_carlo  # noqa: E402
from scripts.analysis.validate_operating_point_montecarlo import _evaluate_details  # noqa: E402
from src.training.training_smoke import _avalanche_config, load_training_smoke_config  # noqa: E402

torch.set_default_dtype(torch.float64)


def _topo_from_score(score, env, layer, caps):
    f = env["features"]
    return layer(num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
                 edge_score=score.reshape(-1), per_node_budget=caps)


def _closed_and_mc(topo, env, eval_q, rng, trials):
    det_q = _evaluate_details(topo, env, quenched_quadrature=eval_q)
    F_closed = float((1.0 - det_q["avalanche_details"]["node_p_correct_decision"]).mean())
    sup = det_q["avalanche_details"]["query_support"]
    link = det_q["channel_diagnostics"]["link_success"].detach().to(torch.float64).numpy()
    rw = sup.normalized_query_weight.detach().to(torch.float64).numpy() * link
    ava = _avalanche_config(env["cfg"])
    _, mc, _ = monte_carlo(env["candidate"].num_nodes, sup.src_index.numpy(), sup.dst_index.numpy(), rw,
                           env["ic"].numpy(), env["iw"].numpy(), k=int(ava["k"]), alpha=int(ava["alpha"]),
                           beta=int(ava["beta"]), rounds=int(ava["rounds"]), trials=int(trials), rng=rng)
    return F_closed, float(np.asarray(mc).mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Monte-Carlo audit of the baseline comparison")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--scene-seed", type=int, default=1007)
    p.add_argument("--max-steps", type=int, default=140)
    p.add_argument("--trials", type=int, default=1500)
    p.add_argument("--run-name", default="baseline_mc_audit")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    base_mf = dict(base); base_mf["quenched_quadrature"] = 1
    eval_q = int(base.get("eval_quenched_quadrature", base.get("quenched_quadrature", 21)))
    rng = np.random.default_rng(7)

    train_env = _scene_env(base, args.node_count, args.train_seed)
    train_env_mf = _scene_env(base_mf, args.node_count, args.train_seed)
    cfg = train_env["cfg"]
    layer, caps = _topology_layer(cfg, train_env["candidate"].num_nodes)
    layer_mf, caps_mf = _topology_layer(train_env_mf["cfg"], train_env_mf["candidate"].num_nodes)

    print(f"MC audit [{args.config}] N={args.node_count} eval_Q={eval_q} trials={args.trials}", flush=True)
    prop = _train("proposed", cfg, train_env, layer, caps, args.max_steps, args.train_seed, 64)
    propmf = _train("proposed", train_env_mf["cfg"], train_env_mf, layer_mf, caps_mf, args.max_steps, args.train_seed, 64)

    env = _scene_env(base, args.node_count, args.scene_seed)
    el, ec = _topology_layer(env["cfg"], env["candidate"].num_nodes)
    fl, fc = _flood_layer(env["cfg"], env["candidate"].num_nodes)

    methods = {}
    with torch.no_grad():
        methods["proposed (hierarchical)"] = _topo_from_score(_model_score(prop, env), env, el, ec)
        methods["proposed (mean-field-trained)"] = _topo_from_score(_model_score(propmf, env), env, el, ec)
        methods["best-channel-k"] = _topo_from_score(_heuristic_scores(env)["best-channel-k"], env, el, ec)
        methods["informed greedy"] = _topo_from_score(_oracle_score(env), env, el, ec)
        methods["no-planning (flood)"] = _topo_from_score(
            torch.zeros(env["features"]["src_index"].numel(), dtype=torch.float64), env, fl, fc)

    rows = []
    for name, topo in methods.items():
        Fc, Fmc = _closed_and_mc(topo, env, eval_q, rng, args.trials)
        rows.append({"method": name, "F_closed_Q{}".format(eval_q): Fc, "F_montecarlo": Fmc,
                     "mc_over_closed": Fmc / max(Fc, 1e-9)})
        print(f"  {name:32s}: F_closed={Fc:.4f}  F_MC={Fmc:.4f}  (MC/closed={Fmc/max(Fc,1e-9):.2f})", flush=True)

    rank_closed = [r["method"] for r in sorted(rows, key=lambda r: r[f"F_closed_Q{eval_q}"])]
    rank_mc = [r["method"] for r in sorted(rows, key=lambda r: r["F_montecarlo"])]
    ranking_agrees = rank_closed == rank_mc
    proposed_best_mc = rank_mc[0].startswith("proposed (hierarchical")
    summary = {"config": args.config, "eval_q": eval_q, "trials": args.trials, "scene_seed": args.scene_seed,
               "rows": rows, "ranking_closed": rank_closed, "ranking_mc": rank_mc,
               "ranking_agrees": bool(ranking_agrees), "proposed_best_under_MC": bool(proposed_best_mc)}
    out_dir = ROOT / "result" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "mc_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nclosed-form ranking == MC ranking: {ranking_agrees}", flush=True)
    print(f"ranking (MC, best->worst): {rank_mc}", flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
