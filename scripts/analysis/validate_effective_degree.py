"""#4 evidence harness — the effective-degree (query-spread) lever and why it needs the quenched closure.

Three reproducible parts at the operating point (geo graph, load-coupled links), evaluated honestly
against the Monte-Carlo:

  A. CAP sweep (nearest-k): failure F is FLAT in the out-degree cap, and the effective degree is pinned
     at ~1 by the concentrated row-softmax. So the fixed soft cap K is NOT the lever (the user's concern,
     resolved by validation).
  B. TEMPERATURE sweep (cap fixed): spreading the query weight (effective degree 1 -> ~3) drops F_MC by
     ~0.2 -- but the MEAN-FIELD evaluator is BLIND to it (F flat), while the SSMC quenched closed form
     SEES it (F drops). This is why mean-field training never learns to spread.
  C. TRAIN mean-field vs quenched: a planner trained on the mean-field stays concentrated; trained on the
     quenched closed form it AUTONOMOUSLY spreads (higher effective degree) and reaches lower true failure.

Usage: python scripts/analysis/validate_effective_degree.py  (or `make effective-degree`).
Outputs result/<run-name>/effective_degree.json + RESULT.md (gitignored).
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

from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer, train_one, _loss_config  # noqa: E402
from scripts.analysis.validate_operating_point_montecarlo import _select_topology, _evaluate_details  # noqa: E402
from scripts.analysis.validate_closed_form_montecarlo import monte_carlo  # noqa: E402
from src.topology.construction import TopologyConstructionLayer  # noqa: E402
from src.training.gradient_governance import GradientGovernanceConfig  # noqa: E402
from src.training.training_smoke import load_training_smoke_config, _avalanche_config  # noqa: E402

torch.set_default_dtype(torch.float64)


def _mc_F(env, sup, link, rng, trials):
    rw = sup.normalized_query_weight.detach().to(torch.float64).numpy() * link
    ava = _avalanche_config(env["cfg"])
    _, mc, _ = monte_carlo(env["candidate"].num_nodes, sup.src_index.numpy(), sup.dst_index.numpy(), rw,
                           env["ic"].numpy(), env["iw"].numpy(), k=int(ava["k"]), alpha=int(ava["alpha"]),
                           beta=int(ava["beta"]), rounds=int(ava["rounds"]), trials=trials, rng=rng)
    return float(np.asarray(mc).mean())


def _eval(env, topo, rng, quench, trials):
    av_mf = _evaluate_details(topo, env, quenched_quadrature=1)["avalanche_details"]
    av_q = _evaluate_details(topo, env, quenched_quadrature=quench)["avalanche_details"]
    sup = av_q["query_support"]
    link = _evaluate_details(topo, env, quenched_quadrature=1)["channel_diagnostics"]["link_success"].detach().to(torch.float64).numpy()
    return {
        "eff_deg": float(sup.effective_unique_peer_degree.mean()),
        "F_meanfield": float((1.0 - av_mf["node_p_correct_decision"]).mean()),
        "F_quenched": float((1.0 - av_q["node_p_correct_decision"]).mean()),
        "F_MC": _mc_F(env, sup, link, rng, trials),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="#4 effective-degree lever evidence (cap / temperature / training)")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--train-node-count", type=int, default=400)
    p.add_argument("--quench", type=int, default=21)
    p.add_argument("--trials", type=int, default=1200)
    p.add_argument("--train-quench", type=int, default=11)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--run-name", default="effective_degree_v1")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    base["node_density_per_km2"] = 200.0
    rng = np.random.default_rng(7)
    out: dict = {"config": args.config, "quench": args.quench}

    env = _scene_env({**base, "max_out_degree": 8}, args.node_count, 7)
    score = -env["features"]["distance_m"].to(torch.float64)
    f = env["features"]; n = env["candidate"].num_nodes

    print("A. CAP sweep (nearest-k): F flat in cap, effective degree pinned ~1")
    out["cap_sweep"] = []
    for cap in (2, 4, 6, 8):
        layer = TopologyConstructionLayer(max_out_degree=cap, support_mode="topk", topk_backend="segmented_fast")
        topo = layer(num_nodes=n, src_index=f["src_index"], dst_index=f["dst_index"], edge_score=score,
                     per_node_budget=torch.full((n,), cap, dtype=torch.long))
        r = _eval(env, topo, rng, args.quench, args.trials); r["cap"] = cap
        out["cap_sweep"].append(r)
        print(f"   cap={cap}: eff_deg={r['eff_deg']:.2f}  F_MC={r['F_MC']:.3f}")

    print("\nB. TEMPERATURE sweep (cap=8): spread is a huge lever; mean-field BLIND, quenched SEES it")
    out["temperature_sweep"] = []
    for tau in (1.0, 4.0, 8.0, 20.0):
        layer = TopologyConstructionLayer(max_out_degree=8, support_mode="topk",
                                          row_softmax_temperature=tau, topk_backend="segmented_fast")
        topo = layer(num_nodes=n, src_index=f["src_index"], dst_index=f["dst_index"], edge_score=score,
                     per_node_budget=torch.full((n,), 8, dtype=torch.long))
        r = _eval(env, topo, rng, args.quench, args.trials); r["tau"] = tau
        out["temperature_sweep"].append(r)
        print(f"   tau={tau:>4.1f}: eff_deg={r['eff_deg']:.2f}  F_meanfield={r['F_meanfield']:.3f} (flat)  "
              f"F_quenched={r['F_quenched']:.3f}  F_MC={r['F_MC']:.3f}")

    print("\nC. TRAIN mean-field vs quenched -> learned effective degree + true reliability")
    gov = GradientGovernanceConfig.from_name("none")
    out["training"] = []
    for q_train, label in ((1, "mean-field"), (args.train_quench, "quenched")):
        cfg = {**base, "quenched_quadrature": int(q_train)}
        tr_env = _scene_env(cfg, args.train_node_count, 7)
        layer, caps = _topology_layer(tr_env["cfg"], tr_env["candidate"].num_nodes)
        bl = dict(_loss_config(tr_env["cfg"])); bl["reliability_failure_target"] = 0.02
        bl["reliability_tail_failure_target"] = 0.02; bl["use_reliability_gate"] = True
        torch.manual_seed(7)
        model, _ = train_one(tr_env["cfg"], tr_env, layer, caps, bl, 0.0, gov, args.steps, 7)
        ev_env = _scene_env(cfg, args.train_node_count, 107)  # held-out
        lay2, caps2 = _topology_layer(ev_env["cfg"], ev_env["candidate"].num_nodes)
        topo = _select_topology(model, ev_env, lay2, caps2)
        r = _eval(ev_env, topo, rng, args.quench, args.trials); r["trained_on"] = label
        out["training"].append(r)
        print(f"   trained on {label:>10}: learned eff_deg={r['eff_deg']:.2f}  F_quenched={r['F_quenched']:.3f}  F_MC={r['F_MC']:.3f}")

    out_dir = ROOT / "result" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "effective_degree.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_dir}")


if __name__ == "__main__":
    main()
