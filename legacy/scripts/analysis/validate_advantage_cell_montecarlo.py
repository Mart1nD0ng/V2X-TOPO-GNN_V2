"""T2c — Monte-Carlo ground-truth band on advantage-map CELLS (does the GNN gap survive MC?).

The advantage region is measured in the QUENCHED surrogate currency. This wrapper checks, per cell,
that the GNN-beats-heuristic GAP is real under a faithful Monte-Carlo of the same consensus protocol
(not a surrogate artifact). For each (density, profile, coupling) it: builds the cell env, trains a
short screening GNN, deploys the GNN topology AND the best-heuristic topology, and runs MC on both —
reporting F_quenched and F_MC for each, and the gap under both currencies.

Reuses the existing MC core (validate_closed_form_montecarlo.monte_carlo) + detail extraction
(validate_operating_point_montecarlo._evaluate_details). Target the robust-advantage d100 cells.

Usage:
  python -B scripts/analysis/validate_advantage_cell_montecarlo.py --cells 100:hard_low_confidence:20,\
100:toy:20 --trials 1000 --out result/advantage_montecarlo
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

from scripts.analysis.generalization_common import build_topology_layer, caps_for, env_from_snapshot, train_model  # noqa: E402
from scripts.analysis.run_envelope_generalization import _cell_cfg  # noqa: E402
from scripts.analysis.validate_closed_form_montecarlo import monte_carlo  # noqa: E402
from scripts.analysis.validate_operating_point_montecarlo import _evaluate_details, _select_topology  # noqa: E402
from src.training.training_smoke import _avalanche_config, _normalized_config, load_training_smoke_config  # noqa: E402
from src.v2x_env.profiles import density_matched_vehicle_config  # noqa: E402
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot  # noqa: E402


def _mc_of_topology(topo, env, eval_q: int, trials: int, rng) -> dict:
    """Quenched F (eval Q) and Monte-Carlo F for one deployed topology."""
    ev = _evaluate_details(topo, env, quenched_quadrature=eval_q)
    av = ev["avalanche_details"]
    support = av["query_support"]
    cf_F = float((1.0 - av["node_p_correct_decision"]).mean())
    link_s = ev["channel_diagnostics"]["link_success"].detach().to(torch.float64).numpy()
    rw = support.normalized_query_weight.detach().to(torch.float64).numpy() * link_s
    src = support.src_index.detach().numpy()
    dst = support.dst_index.detach().numpy()
    ic = env["ic"].detach().to(torch.float64).numpy()
    iw = env["iw"].detach().to(torch.float64).numpy()
    ava = _avalanche_config(env["cfg"])
    _mc_C, mc_F, _mc_D = monte_carlo(
        env["candidate"].num_nodes, src, dst, rw, ic, iw,
        k=int(ava["k"]), alpha=int(ava["alpha"]), beta=int(ava["beta"]), rounds=int(ava["rounds"]),
        trials=trials, rng=rng)
    return {"F_quenched": cf_F, "F_mc": float(mc_F.mean()),
            "optimism_abs": float(mc_F.mean() - cf_F), "link_success_mean": float(link_s.mean())}


def main() -> None:
    p = argparse.ArgumentParser(description="Monte-Carlo band on advantage-map cells (T2c)")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--cells", default="100:hard_low_confidence:20,100:toy:20,100:near_target_synthetic:20",
                   help="comma list of density:profile:coupling")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--trials", type=int, default=1000)
    p.add_argument("--scene-seed", type=int, default=42)
    p.add_argument("--out", default="result/advantage_montecarlo")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    rng = np.random.default_rng(int(args.scene_seed))
    cells = []
    for tok in args.cells.split(","):
        d, pr, k = tok.split(":")
        cells.append((float(d), pr, float(k)))

    print(f"Advantage-cell MC band: {len(cells)} cells, {args.trials} trials, N={args.node_count}", flush=True)
    rows = []
    for (density, profile, coupling) in cells:
        cfg = _normalized_config(_cell_cfg(base, density, profile, coupling, args.node_count))
        snap = generate_vehicle_snapshot(density_matched_vehicle_config(args.node_count, density, seed=args.scene_seed))
        env = env_from_snapshot(snap, cfg, interference_coupling_db=coupling)
        env["cfg"] = cfg  # _evaluate_details needs it
        layer = build_topology_layer(cfg)
        caps = caps_for(env, cfg)
        eval_q = int(cfg.get("eval_quenched_quadrature", cfg.get("quenched_quadrature", 21)))

        # best heuristic by quenched F (channel/success/sinr/nearest)
        ef = env["features"]["edge_features"].to(dtype=torch.float64)
        dist = env["features"]["distance_m"].to(dtype=torch.float64)
        cfg_q = dict(cfg); cfg_q["quenched_quadrature"] = eval_q
        best = None
        for name, sc in {"channel": ef[:, 2], "success": ef[:, 3], "sinr": ef[:, 4], "nearest": -dist}.items():
            topo_h = _select_topology(sc.reshape(-1), env, layer, caps)
            f = float((1.0 - _evaluate_details(topo_h, env, quenched_quadrature=eval_q)
                       ["avalanche_details"]["node_p_correct_decision"]).mean())
            if best is None or f < best[1]:
                best = (name, f, topo_h)
        best_name, _bestfq, best_topo = best

        # short screening GNN topology
        model = train_model(cfg, [env], layer, int(args.steps), model_seed=42)
        with torch.no_grad():
            gnn_topo = _select_topology(model, env, layer, caps)

        gnn_mc = _mc_of_topology(gnn_topo, env, eval_q, args.trials, rng)
        heur_mc = _mc_of_topology(best_topo, env, eval_q, args.trials, rng)
        gap_q = heur_mc["F_quenched"] - gnn_mc["F_quenched"]
        gap_mc = heur_mc["F_mc"] - gnn_mc["F_mc"]
        row = {"density": density, "profile": profile, "coupling_db": coupling,
               "best_heuristic": best_name, "gnn": gnn_mc, "heuristic": heur_mc,
               "gap_quenched": gap_q, "gap_mc": gap_mc, "gap_survives_mc": bool(gap_mc > 0)}
        rows.append(row)
        print(f"  d{int(density)}/{profile}/c{int(coupling)} | GNN F_q={gnn_mc['F_quenched']:.4f} "
              f"F_mc={gnn_mc['F_mc']:.4f} | bestH({best_name}) F_q={heur_mc['F_quenched']:.4f} "
              f"F_mc={heur_mc['F_mc']:.4f} | gap_q={gap_q:+.4f} gap_mc={gap_mc:+.4f} "
              f"-> {'SURVIVES' if gap_mc > 0 else 'GONE'}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "advantage_mc.json").write_text(
        json.dumps({"trials": args.trials, "node_count": args.node_count, "cells": rows}, indent=2, sort_keys=True),
        encoding="utf-8")
    n_surv = sum(1 for r in rows if r["gap_survives_mc"])
    print(f"\n{n_surv}/{len(rows)} cells: GNN advantage gap SURVIVES Monte-Carlo ground truth")
    print(f"wrote {out_dir / 'advantage_mc.json'}")


if __name__ == "__main__":
    main()
