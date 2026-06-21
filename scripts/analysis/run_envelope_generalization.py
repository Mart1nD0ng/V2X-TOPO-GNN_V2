"""D3 — mixture-generalist + leave-cell-out validation (roadmap item 2.4).

The claim under test: ONE unconditioned model, trained as a mixture across the (density x coupling x
profile) envelope, spans the measured advantage region — matching per-cell screening experts in-grid,
generalizing to a HELD-OUT density, and interpolating to off-grid cells it never trained on.

Arms (all on paper_environment_v1, quenched eval Q=21, N=600, the D3-fix-1 per-env evaluator coupling):
  M_all   : train on all 27 grid cells (round-robin); eval in-grid (retention vs P0 screening expert)
            + off-grid {density 150,250} x {coupling 5,15} (interpolation gap vs heuristics).
  LOCO    : leave-one-DENSITY-out (3 folds) — train on 2 densities (18 cells), eval the held-out
            density (9 cells). This is the honest generalization test (extrapolation to an unseen
            density), not a 1-knob-step interpolation.

Metric: gap = bestH - F_model (both at eval Q). retention = gap_model / gap_expert, reported ONLY where
the P0 expert gap is materially positive (>0.02), i.e. the robust-advantage cells — elsewhere the
denominator is seed-noise-scale and retention is undefined (we report raw gap instead).

Usage: python -B scripts/analysis/run_envelope_generalization.py --out result/mixture_generalization
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.generalization_common import (  # noqa: E402
    build_topology_layer, env_from_snapshot, evaluate, model_score, train_model,
)
from src.training.training_smoke import _normalized_config, load_training_smoke_config  # noqa: E402
from src.v2x_env.profiles import density_matched_vehicle_config  # noqa: E402
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot  # noqa: E402

_PROFILES = ["toy", "near_target_synthetic", "hard_low_confidence"]
_RETENTION_MIN_EXPERT_GAP = 0.02  # only score retention where the expert gap is materially positive


def _cell_cfg(base: dict, density: float, profile: str, coupling: float, n: int) -> dict:
    cfg = dict(base)
    cfg.update({"vehicle_count": int(n), "vehicle_profile": "density_matched",
                "node_density_per_km2": float(density), "training_profile": profile,
                "max_steps": 0, "init_mode": "xavier", "seed": 42})
    cfg["physical"] = {**dict(cfg.get("physical", {})),
                       "interference_density_coupling_db": float(coupling), "interference_reference_load": 1.0}
    cfg["candidate_graph"] = {**dict(cfg.get("candidate_graph", {})),
                              "interference_density_coupling_db": float(coupling),
                              "interference_reference_degree": 8.0}
    return cfg


def _build_env(base: dict, density: float, profile: str, coupling: float, n: int):
    cfg = _normalized_config(_cell_cfg(base, density, profile, coupling, n))
    snap = generate_vehicle_snapshot(density_matched_vehicle_config(n, density, seed=42))
    env = env_from_snapshot(snap, cfg, interference_coupling_db=coupling)
    return cfg, env


def _best_heuristic_F(env, topo, cfg_eval) -> float:
    ef = env["features"]["edge_features"].to(dtype=torch.float64)
    dist = env["features"]["distance_m"].to(dtype=torch.float64)
    best = None
    for sc in (ef[:, 2], ef[:, 3], ef[:, 4], -dist):
        with torch.no_grad():
            f = float(evaluate(sc.reshape(-1), env, topo, cfg_eval)["F_avalanche_node_mean"].mean())
        best = f if best is None else min(best, f)
    return best


def _model_F(model, env, topo, cfg_eval) -> float:
    with torch.no_grad():
        return float(evaluate(model_score(model, env), env, topo, cfg_eval)["F_avalanche_node_mean"].mean())


def _eval_cell(model, env, topo, cfg_eval, expert_F_gnn, floor):
    bestH = _best_heuristic_F(env, topo, cfg_eval)
    f_model = _model_F(model, env, topo, cfg_eval)
    gap_model = bestH - f_model
    out = {"bestH": bestH, "F_model": f_model, "gap_model": gap_model,
           "headroom": bestH - floor, "floor": floor}
    if expert_F_gnn is not None:
        gap_expert = bestH - expert_F_gnn
        out["F_expert_P0"] = expert_F_gnn
        out["gap_expert"] = gap_expert
        out["retention"] = (gap_model / gap_expert) if gap_expert > _RETENTION_MIN_EXPERT_GAP else None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Mixture-generalist + leave-one-density-out validation")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--map", default="result/advantage_map/advantage_map.json")
    p.add_argument("--floor-table", default="result/protocol_floor_table/floor_table.json")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--steps-per-cell", type=int, default=20, help="round-robin steps per training cell")
    p.add_argument("--out", default="result/mixture_generalization")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    map_json = json.loads((ROOT / args.map).read_text(encoding="utf-8"))
    p0 = {(c["density"], c["profile"], c["coupling_db"]): c for c in map_json["cells"]}
    floor_rows = json.loads((ROOT / args.floor_table).read_text(encoding="utf-8"))["rows"]
    floor_by_profile = {r["profile"]: r["floors"]["small_realistic (k5 a3 b5 r20)"]
                        for r in floor_rows if r["degree"] == 4}

    grid_d, grid_c = [100.0, 200.0, 300.0], [0.0, 10.0, 20.0]
    grid = list(itertools.product(grid_d, _PROFILES, grid_c))
    offgrid = list(itertools.product([150.0, 250.0], _PROFILES, [5.0, 15.0]))

    print(f"Building {len(grid)} grid + {len(offgrid)} off-grid envs (N={args.node_count})...", flush=True)
    envs = {}
    for (d, pr, k) in grid + offgrid:
        cfg, env = _build_env(base, d, pr, k, args.node_count)
        envs[(d, pr, k)] = env
    cfg_ref = _normalized_config(_cell_cfg(base, 200.0, "hard_low_confidence", 10.0, args.node_count))
    eval_q = int(cfg_ref.get("eval_quenched_quadrature", cfg_ref.get("quenched_quadrature", 21)))
    cfg_eval = dict(cfg_ref); cfg_eval["quenched_quadrature"] = eval_q
    topo = build_topology_layer(cfg_ref)

    def floor_of(pr): return float(floor_by_profile[pr])
    def expert_of(key): return p0[key]["F_gnn_mean"] if key in p0 else None

    results = {"currency": f"quenched eval Q={eval_q}", "node_count": args.node_count,
               "steps_per_cell": args.steps_per_cell, "arms": {}}

    # ---- Arm M_all: train on all 27 grid cells, eval in-grid + off-grid ----
    grid_train_envs = [envs[k] for k in grid]
    steps_all = args.steps_per_cell * len(grid)
    print(f"\n[M_all] training mixture on {len(grid)} grid cells, {steps_all} steps...", flush=True)
    m_all = train_model(cfg_ref, grid_train_envs, topo, steps_all, model_seed=42)
    in_grid, off_grid = [], []
    for (d, pr, k) in grid:
        m = _eval_cell(m_all, envs[(d, pr, k)], topo, cfg_eval, expert_of((d, pr, k)), floor_of(pr))
        m.update({"density": d, "profile": pr, "coupling_db": k}); in_grid.append(m)
    for (d, pr, k) in offgrid:
        m = _eval_cell(m_all, envs[(d, pr, k)], topo, cfg_eval, None, floor_of(pr))
        m.update({"density": d, "profile": pr, "coupling_db": k}); off_grid.append(m)
    results["arms"]["M_all"] = {"steps": steps_all, "in_grid": in_grid, "off_grid": off_grid}

    rets = [c["retention"] for c in in_grid if c.get("retention") is not None]
    print(f"[M_all] retention on materially-advantaged cells (n={len(rets)}): "
          f"mean={sum(rets)/len(rets):.3f}  min={min(rets):.3f}  max={max(rets):.3f}", flush=True)
    pos_off = sum(1 for c in off_grid if c["gap_model"] > 0.005)
    print(f"[M_all] off-grid cells with positive interpolation gap (>0.005): {pos_off}/{len(off_grid)}", flush=True)

    # ---- Arm LOCO: leave-one-DENSITY-out (3 folds) ----
    loco = []
    for hold_d in grid_d:
        train_keys = [k for k in grid if k[0] != hold_d]
        test_keys = [k for k in grid if k[0] == hold_d]
        steps = args.steps_per_cell * len(train_keys)
        print(f"\n[LOCO hold d={int(hold_d)}] train on {len(train_keys)} cells ({steps} steps), "
              f"eval {len(test_keys)} held-out...", flush=True)
        m_loco = train_model(cfg_ref, [envs[k] for k in train_keys], topo, steps, model_seed=42)
        cells = []
        for (d, pr, k) in test_keys:
            m = _eval_cell(m_loco, envs[(d, pr, k)], topo, cfg_eval, expert_of((d, pr, k)), floor_of(pr))
            m.update({"density": d, "profile": pr, "coupling_db": k}); cells.append(m)
        frets = [c["retention"] for c in cells if c.get("retention") is not None]
        summ = {"held_out_density": hold_d, "steps": steps, "cells": cells,
                "retention_mean": (sum(frets) / len(frets)) if frets else None,
                "retention_n": len(frets)}
        loco.append(summ)
        rtxt = f"{summ['retention_mean']:.3f}" if summ["retention_mean"] is not None else "n/a (no material-gap cells)"
        print(f"[LOCO hold d={int(hold_d)}] held-out retention (n={len(frets)}): {rtxt}", flush=True)
    results["arms"]["LOCO_leave_density_out"] = loco

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "envelope_generalization.json").write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_dir / 'envelope_generalization.json'}")


if __name__ == "__main__":
    main()
