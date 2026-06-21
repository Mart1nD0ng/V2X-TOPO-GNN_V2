"""Advantage-region map (Roadmap Phase 1.1): WHERE does learned topology beat heuristics?

For each cell of the (density x training-profile x interference-coupling) grid, on the standard
paper environment (tr37885 + geometric LOS + xavier + quenched currency), record:

  floor    — the perfect-link protocol floor at the deployed degree budget (from the floor table);
  bestH    — best of the extended heuristic set (channel / success / sinr / nearest / random), each
             scored through the SAME constructor + config-physics evaluator at eval Q;
  F_gnn    — a short-budget xavier-trained GNN (the screening proxy for the learnable side);
  headroom = bestH - floor   (how much a better topology COULD win at this cell);
  gap      = bestH - F_gnn   (how much the screening GNN DOES win).

Cell classification (v1 SCREENING thresholds, pre-registered for fine-stage candidacy — these are
NOT claims; claims come from the streaming fine stage with paired statistics):
  FLOOR_LIMITED      headroom < 0.015   (nothing to win — graceful-parity region)
  GNN_ADVANTAGE      gap > 0.010 and headroom > 0.020   (fine-stage candidate)
  HEURISTIC_PARITY   |gap| <= 0.010
  GNN_DEFICIT        gap < -0.010       (training recipe suspect at this cell)

Usage:
  python scripts/analysis/run_advantage_map.py --densities 100,200,300 \
      --profiles toy,near_target_synthetic,hard_low_confidence --couplings 0,10,20 \
      --node-count 600 --steps 120
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_v2x_graph_consensus  # noqa: E402
from src.topology import TopologyConstructionLayer  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _avalanche_config,
    _evaluator_physical_config,
    _initial_preferences,
    _make_environment,
    _normalized_config,
    _run_training_phase,
    load_training_smoke_config,
)


def _cell_config(base: dict, density: float, profile: str, coupling: float, n: int, steps: int,
                 seed: int = 42) -> dict:
    cfg = dict(base)
    cfg.update({
        "vehicle_count": int(n),
        "vehicle_profile": "density_matched",
        "node_density_per_km2": float(density),
        "training_profile": profile,
        "max_steps": int(steps),
        "init_mode": "xavier",
        "seed": int(seed),
    })
    physical = dict(cfg.get("physical", {}))
    physical["interference_density_coupling_db"] = float(coupling)
    physical["interference_reference_load"] = 1.0
    cfg["physical"] = physical
    cand = dict(cfg.get("candidate_graph", {}))
    cand["interference_density_coupling_db"] = float(coupling)  # feature side matches evaluator side
    cand["interference_reference_degree"] = 8.0
    cfg["candidate_graph"] = cand
    return cfg


def _evaluate_score(score, *, cfg, candidate, features, layer, caps, ic, iw) -> float:
    with torch.no_grad():
        topo = layer(num_nodes=candidate.num_nodes, src_index=features["src_index"],
                     dst_index=features["dst_index"], edge_score=score, per_node_budget=caps)
        sel = topo.selected_candidate_index
        ev = evaluate_v2x_graph_consensus(
            **topo.as_evaluation_kwargs(),
            distance_m=features["distance_m"].index_select(0, sel),
            los_flag=features["los_flag"].index_select(0, sel),
            node_initial_correct=ic, node_initial_wrong=iw,
            physical_config=_evaluator_physical_config(cfg),
            avalanche_config=_avalanche_config(cfg, eval_mode=True),
            energy_config={"packet_duration_s": 0.001},
        )
        return float(ev["F_avalanche_node_mean"])


def _classify(headroom: float, gap: float) -> str:
    if headroom < 0.015:
        return "FLOOR_LIMITED"
    if gap > 0.010 and headroom > 0.020:
        return "GNN_ADVANTAGE"
    if gap < -0.010:
        return "GNN_DEFICIT"
    return "HEURISTIC_PARITY"


def _agg(vals: list[float]) -> tuple[float, float]:
    """(mean, sample-std) over the finite values; std=0 for a single seed; NaN if all non-finite."""
    finite = [v for v in vals if v == v]
    if not finite:
        return float("nan"), float("nan")
    m = sum(finite) / len(finite)
    if len(finite) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in finite) / (len(finite) - 1)
    return m, var ** 0.5


def _train_cell(cfg_raw: dict):
    """Train the screening GNN with a divergence-retry ladder. Both init recipes have known
    explosion cells (deterministic: NaN on paper env @N=2000; xavier: NaN at low-coupling cells),
    so a screening harness must degrade gracefully: retry at lower lr with grad clipping, and if
    the cell still diverges, record it honestly (TRAIN_DIVERGED) — recipe fragility per cell is
    itself Phase-2.2 data. Heuristics and the floor are computed regardless."""
    base_lr = float(cfg_raw.get("learning_rate", 0.005))
    ladder = [
        {"learning_rate": base_lr, "gradient_clip_norm": None},
        {"learning_rate": base_lr * 0.4, "gradient_clip_norm": 1.0},
        {"learning_rate": base_lr * 0.15, "gradient_clip_norm": 1.0},
    ]
    last_err = None
    for attempt, overrides in enumerate(ladder):
        cfg_try = dict(cfg_raw)
        cfg_try["learning_rate"] = overrides["learning_rate"]
        if overrides["gradient_clip_norm"] is not None:
            cfg_try["gradient_clip_norm"] = overrides["gradient_clip_norm"]
        try:
            report, _model = _run_training_phase(cfg_try)
            return float(report["final_metric_snapshot"]["F_avalanche_node_mean"]), attempt, None
        except (ValueError, FloatingPointError) as err:  # non-finite scores / numerics
            last_err = f"{type(err).__name__}: {err}"
    return float("nan"), len(ladder), last_err


def _score_one_seed(base: dict, density: float, profile: str, coupling: float, n: int, steps: int,
                    seed: int, floor: float) -> dict:
    """Train the screening GNN + score the heuristic set for ONE (cell, seed) draw. Varying the seed
    re-draws the vehicle layout AND the xavier init together, so the spread across seeds is the honest
    variance of the LABEL (gap = bestH - F_gnn), not just init noise."""
    cfg_raw = _cell_config(base, density, profile, coupling, n, steps, seed)
    f_gnn, train_attempt, train_error = _train_cell(cfg_raw)
    cfg = _normalized_config(cfg_raw)
    candidate, features = _make_environment(cfg)
    budget = None if cfg["max_out_degree"] is None else int(cfg["max_out_degree"])
    layer = TopologyConstructionLayer(max_out_degree=budget, support_mode=str(cfg["support_mode"]),
                                      temperature=1.0, topk_backend=str(cfg["topk_backend"]))
    caps = torch.full((candidate.num_nodes,), budget, dtype=torch.long) if budget else None
    ic, iw = _initial_preferences(cfg, candidate.num_nodes, features.get("node_xy"))
    dist = features["distance_m"].to(dtype=torch.float64)
    ef = features["edge_features"].to(dtype=torch.float64)
    heuristics = {
        "best_channel_k": ef[:, 2], "best_success_k": ef[:, 3], "best_sinr_k": ef[:, 4],
        "nearest_k": -dist,
    }
    h_scores = {}
    for name, score in heuristics.items():
        h_scores[name] = _evaluate_score(score.reshape(-1), cfg=cfg, candidate=candidate,
                                         features=features, layer=layer, caps=caps, ic=ic, iw=iw)
    torch.manual_seed(42)  # random_k baseline kept on a FIXED seed so it is a stable comparator
    h_scores["random_k"] = _evaluate_score(torch.randn(dist.numel(), dtype=torch.float64),
                                           cfg=cfg, candidate=candidate, features=features,
                                           layer=layer, caps=caps, ic=ic, iw=iw)
    best_h_name = min(h_scores, key=h_scores.get)
    best_h = h_scores[best_h_name]
    headroom = best_h - floor
    gap = best_h - f_gnn
    verdict = "TRAIN_DIVERGED" if f_gnn != f_gnn else _classify(headroom, gap)
    return {"seed": int(seed), "F_gnn": f_gnn, "best_heuristic": best_h, "best_heuristic_name": best_h_name,
            "headroom": headroom, "gap": gap, "cell_class": verdict,
            "train_retry_attempt": train_attempt, "train_error": train_error, "heuristics": h_scores}


def main() -> None:
    p = argparse.ArgumentParser(description="Advantage-region map (floor / heuristics / short GNN per cell)")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--floor-table", default="result/protocol_floor_table/floor_table.json")
    p.add_argument("--densities", default="100,200,300")
    p.add_argument("--profiles", default="toy,near_target_synthetic,hard_low_confidence")
    p.add_argument("--couplings", default="0,10,20")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seeds", default="42",
                   help="comma seeds; >1 re-draws layout+init per seed and reports gap mean/std (P0)")
    p.add_argument("--run-name", default="advantage_map_v1")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    densities = [float(x) for x in args.densities.split(",") if x.strip()]
    profiles = [x.strip() for x in args.profiles.split(",") if x.strip()]
    couplings = [float(x) for x in args.couplings.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    floor_rows = json.loads((ROOT / args.floor_table).read_text(encoding="utf-8"))["rows"]
    floors = {r["profile"]: r["floors"]["small_realistic (k5 a3 b5 r20)"]
              for r in floor_rows if r["degree"] == 4}

    cells = list(itertools.product(densities, profiles, couplings))
    print(f"Advantage map: {len(cells)} cells x {len(seeds)} seeds (N={args.node_count}, {args.steps} "
          f"steps, paper env, quenched train Q11 / eval Q21), seeds={seeds}", flush=True)
    results = []
    t_total = time.perf_counter()
    for density, profile, coupling in cells:
        floor = float(floors[profile])
        t_cell = time.perf_counter()
        per_seed = [_score_one_seed(base, density, profile, coupling, args.node_count, args.steps, s, floor)
                    for s in seeds]
        elapsed = time.perf_counter() - t_cell
        f_mean, f_std = _agg([r["F_gnn"] for r in per_seed])
        gap_mean, gap_std = _agg([r["gap"] for r in per_seed])
        head_mean, head_std = _agg([r["headroom"] for r in per_seed])
        bh_mean, bh_std = _agg([r["best_heuristic"] for r in per_seed])
        diverged = [r["seed"] for r in per_seed if r["F_gnn"] != r["F_gnn"]]
        verdict = "TRAIN_DIVERGED" if f_mean != f_mean else _classify(head_mean, gap_mean)
        # P0 gate: an advantage label is ROBUST only if the lower 2-sigma bound clears the thresholds.
        label_robust = None
        if verdict == "GNN_ADVANTAGE":
            label_robust = bool(gap_mean - 2.0 * gap_std > 0.010 and head_mean - 2.0 * head_std > 0.020)
        results.append({"density": density, "profile": profile, "coupling_db": coupling, "floor": floor,
                        "F_gnn_mean": f_mean, "F_gnn_std": f_std,
                        "best_heuristic_mean": bh_mean, "best_heuristic_std": bh_std,
                        "headroom_mean": head_mean, "headroom_std": head_std,
                        "gap_mean": gap_mean, "gap_std": gap_std,
                        "cell_class": verdict, "label_robust": label_robust,
                        "diverged_seeds": diverged, "elapsed_s": elapsed, "per_seed": per_seed})
        rob = "" if label_robust is None else (" ROBUST" if label_robust else " *FRAGILE*")
        print(f"CELL d={density:>5} {profile:>24} c={coupling:>4}dB | floor={floor:.4f} "
              f"bestH={bh_mean:.4f}+-{bh_std:.4f} gnn={f_mean:.4f}+-{f_std:.4f} | "
              f"headroom={head_mean:+.4f}+-{head_std:.4f} gap={gap_mean:+.4f}+-{gap_std:.4f} "
              f"-> {verdict}{rob} [{elapsed:.0f}s]"
              + (f" diverged@{diverged}" if diverged else ""), flush=True)
    total_elapsed = time.perf_counter() - t_total

    out_dir = ROOT / "result" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"config": args.config, "node_count": args.node_count, "steps": args.steps,
               "seeds": seeds, "total_elapsed_s": total_elapsed,
               "evaluator_currency": "quenched (train Q=11 / eval Q=21)",
               "screening_thresholds": {"floor_limited_headroom": 0.015, "advantage_gap": 0.010,
                                        "advantage_headroom": 0.020},
               "cells": results}
    (out_dir / "advantage_map.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    adv = [c for c in results if c["cell_class"] == "GNN_ADVANTAGE"]
    fragile = [c for c in adv if c["label_robust"] is False]
    print(f"\nADVANTAGE CELLS: {len(adv)} ({len(fragile)} FRAGILE under 2-sigma seed band)")
    for c in adv:
        rob = "ROBUST" if c["label_robust"] else "*FRAGILE*"
        print(f"  d={c['density']} {c['profile']} c={c['coupling_db']}dB  "
              f"gap={c['gap_mean']:+.4f}+-{c['gap_std']:.4f} headroom={c['headroom_mean']:+.4f} {rob}")
    print(f"total wall-clock: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"wrote {out_dir / 'advantage_map.json'}")


if __name__ == "__main__":
    main()
