"""Re-calibrate the operating point under the CORRECT axis-visibility LOS (audit C-1 fix).

Context: operating_point_v1 was tuned to be "hard / load-coupled" while the candidate graph
still used the buggy `road_segment` LOS rule (mislabels ~30% of collinear cross-block edges as
NLOS -> a globally pessimistic channel). Fixing LOS to `axis_visibility` makes the channel much
better, so link_success -> ~1, the retransmission lever n_tx = 1/link_success collapses, and the
coupled C/D/E Pareto / w-sweet-spot story evaporates.

The load-coupled mechanism lives on the EVALUATOR side (v2x_consensus_bridge, line ~777): the
interference floor on edges into a node rises with that node's SELECTED in-load
(base + coupling_db * log10(in_degree / reference_load)). So selecting MORE in-edges raises
interference -> lowers link_success -> raises n_tx. This creates an OPTIMAL in-degree: too sparse =
poor consensus support (high F); too dense = interference collapses the links (high F AND high D/E).
The cost terms (D/E) regularise the planner toward that optimum, so an over-flaring rel-only (w=0)
planner is dominated. That is the C/D/E story we must restore WITHOUT the LOS artifact.

This script (PHASE 1, no training) sweeps the SELECTED degree (nearest-k) at a grid of
(base interference floor, density) to locate the U-shaped F-vs-degree curve: we want a profile
whose minimum F is in a learnable-hard band (~0.03-0.10) with a clear U (over-degree strictly
worse on F), so the load-coupled optimum exists and rel-only flooding is dominated. Feature-side
coupling stays 0 (historical); only the evaluator interference is hardened. Tx/MCS stay standard.

Usage:
  python -B scripts/analysis/recalibrate_operating_point.py --node-count 600 --seed 7
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

from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from src.topology import TopologyConstructionLayer  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _initial_preferences,
    _make_environment,
    _normalized_config,
    load_training_smoke_config,
)


def harden_cfg(base_config: dict, *, interference_dbm: float, coupling_db: float, density: float,
               tx_dbm: float, mcs_db: float, reference_load: float) -> dict:
    """Return a cfg with the EVALUATOR hardened (physical block) under axis_visibility LOS. Feature
    channel SINR uses the same base interference (no feature coupling -> the historical config)."""
    cfg = dict(base_config)
    cfg["node_density_per_km2"] = float(density)

    physical = dict(cfg.get("physical", {}))
    physical.update({
        "tx_power_dbm": float(tx_dbm),
        "mcs_threshold_db": float(mcs_db),
        "interference_proxy_dbm": float(interference_dbm),
        "interference_density_coupling_db": float(coupling_db),
        "interference_reference_load": float(reference_load),
    })
    cfg["physical"] = physical

    channel = dict(cfg.get("channel", {}))
    channel.update({
        "tx_power_dbm": float(tx_dbm),
        "mcs_threshold_db": float(mcs_db),
        "transition_width_db": float(physical.get("transition_width_db", 3.0)),
        "interference_proxy_dbm": float(interference_dbm),
    })
    cfg["channel"] = channel

    candidate = dict(cfg.get("candidate_graph", {}))
    candidate["los_model"] = "axis_visibility"  # feature coupling stays at its default (0)
    cfg["candidate_graph"] = candidate
    return cfg


def _layer(num_nodes: int, budget: int, cfg: dict):
    layer = TopologyConstructionLayer(
        max_out_degree=int(budget), support_mode="topk", temperature=1.0,
        topk_backend=str(cfg["topk_backend"]),
        gradient_mode=str(cfg.get("gradient_mode", "straight_through_full_candidate")),
        straight_through_temperature=cfg.get("straight_through_temperature", None),
    )
    caps = torch.full((num_nodes,), int(budget), dtype=torch.long)
    return layer, caps


def _quant(arr) -> dict:
    a = np.asarray(arr, dtype=float)
    if a.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}
    return {"p10": float(np.quantile(a, 0.1)), "p50": float(np.quantile(a, 0.5)),
            "p90": float(np.quantile(a, 0.9)), "mean": float(a.mean())}


def degree_sweep(base_config: dict, *, interference_dbm: float, coupling_db: float, density: float,
                 node_count: int, seed: int, tx_dbm: float, mcs_db: float, reference_load: float,
                 degrees: list[int]) -> dict:
    cfg = _normalized_config(harden_cfg(base_config, interference_dbm=interference_dbm,
                                        coupling_db=coupling_db, density=density, tx_dbm=tx_dbm,
                                        mcs_db=mcs_db, reference_load=reference_load))
    cfg["vehicle_count"] = int(node_count)
    cfg["seed"] = int(seed)
    candidate, features = _make_environment(cfg)
    ic, iw = _initial_preferences(cfg, candidate.num_nodes, features.get("node_xy"))
    env = {"cfg": cfg, "candidate": candidate, "features": features, "ic": ic, "iw": iw}

    dist = features["distance_m"].to(dtype=torch.float64)
    by_deg = {}
    for d in degrees:
        layer, caps = _layer(candidate.num_nodes, d, cfg)
        with torch.no_grad():
            by_deg[d] = _metrics(_forward(-dist.reshape(-1), env, layer, caps, eval_mode=True))

    fvals = {d: by_deg[d]["F"] for d in degrees}
    best_deg = min(fvals, key=fvals.get)
    f_min = fvals[best_deg]
    f_max_deg = max(degrees)
    # U-shape: a degree higher than the F-optimum exists AND is strictly worse on F (load penalty).
    over = [d for d in degrees if d > best_deg]
    u_shaped = bool(over) and all(fvals[d] > f_min + 1e-3 for d in over)
    f_over_penalty = (fvals[f_max_deg] - f_min) if f_max_deg > best_deg else 0.0
    # cost lever: D at the densest degree relative to the F-optimum degree.
    d_lever = by_deg[f_max_deg]["D"] / max(by_deg[best_deg]["D"], 1e-9)
    learnable = 0.02 <= f_min <= 0.15
    return {
        "interference_dbm": interference_dbm, "coupling_db": coupling_db, "density": density,
        "reference_load": reference_load, "tx_dbm": tx_dbm, "mcs_db": mcs_db,
        "node_count": node_count, "seed": seed, "edge_count": int(candidate.edge_count),
        "los_fraction": float(np.mean(candidate.los_flag)),
        "success_q": _quant(candidate.success_probability),
        "by_degree": {str(d): by_deg[d] for d in degrees},
        "F_by_degree": {str(d): fvals[d] for d in degrees},
        "best_degree_F": int(best_deg), "F_min": float(f_min),
        "F_over_penalty": float(f_over_penalty), "D_lever_maxdeg_vs_best": float(d_lever),
        "u_shaped": u_shaped, "learnable_hard": bool(learnable),
        "candidate": bool(u_shaped and learnable),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Phase-1 degree-sweep hardness probe (operating-point re-calibration)")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--interferences", default="-82,-80,-78,-76,-74")
    p.add_argument("--couplings", default="20")
    p.add_argument("--densities", default="200,300")
    p.add_argument("--reference-load", type=float, default=1.0)
    p.add_argument("--degrees", default="1,2,3,4,6,8")
    p.add_argument("--tx-dbm", type=float, default=23.0)
    p.add_argument("--mcs-db", type=float, default=8.0)
    p.add_argument("--out", default="result/operating_point_recalibration")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    interferences = [float(x) for x in args.interferences.split(",") if x.strip()]
    couplings = [float(x) for x in args.couplings.split(",") if x.strip()]
    densities = [float(x) for x in args.densities.split(",") if x.strip()]
    degrees = [int(x) for x in args.degrees.split(",") if x.strip()]

    print(f"Phase-1 degree-sweep probe [{args.config}] N={args.node_count} seed={args.seed} "
          f"tx={args.tx_dbm} mcs={args.mcs_db} ref_load={args.reference_load} (axis_visibility, feat coupling=0)",
          flush=True)
    head = f"{'I(dBm)':>7} {'cpl':>4} {'dens':>5} | {'succ.p50':>8} | " + " ".join(f"F@{d:<4}" for d in degrees) + \
           f" | {'bestD':>5} {'Fmin':>6} {'Uovr':>5} {'Dlev':>5} | flags"
    print(head, flush=True)

    rows = []
    for dens in densities:
        for cpl in couplings:
            for itf in interferences:
                r = degree_sweep(base_config, interference_dbm=itf, coupling_db=cpl, density=dens,
                                 node_count=args.node_count, seed=args.seed, tx_dbm=args.tx_dbm,
                                 mcs_db=args.mcs_db, reference_load=args.reference_load, degrees=degrees)
                rows.append(r)
                fcells = " ".join(f"{r['F_by_degree'][str(d)]:>5.3f}" for d in degrees)
                flags = []
                if r["learnable_hard"]:
                    flags.append("LEARNABLE")
                if r["u_shaped"]:
                    flags.append("U")
                if r["candidate"]:
                    flags.append("**CAND**")
                print(f"{itf:>7.0f} {cpl:>4.0f} {dens:>5.0f} | {r['success_q']['p50']:>8.3f} | {fcells} | "
                      f"{r['best_degree_F']:>5d} {r['F_min']:>6.3f} {r['F_over_penalty']:>5.2f} "
                      f"{r['D_lever_maxdeg_vs_best']:>5.1f} | {' '.join(flags)}", flush=True)

    cands = [r for r in rows if r["candidate"]]
    ranked = sorted(cands, key=lambda r: (r["F_over_penalty"] * min(r["D_lever_maxdeg_vs_best"], 50.0)), reverse=True)
    print(f"\n{len(cands)} candidate profile(s) (U-shaped AND learnable-hard):", flush=True)
    for r in ranked[:8]:
        print(f"  I={r['interference_dbm']:.0f}dBm cpl={r['coupling_db']:.0f} dens={r['density']:.0f}: "
              f"Fmin={r['F_min']:.3f}@deg{r['best_degree_F']} over_pen={r['F_over_penalty']:.2f} "
              f"Dlever={r['D_lever_maxdeg_vs_best']:.1f}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phase1_degree_sweep.json").write_text(
        json.dumps({"config": args.config, "node_count": args.node_count, "seed": args.seed,
                    "degrees": degrees, "rows": rows, "ranked_candidates": ranked},
                   indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_dir / 'phase1_degree_sweep.json'}", flush=True)


if __name__ == "__main__":
    main()
