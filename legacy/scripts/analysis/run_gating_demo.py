"""D2 — domain-aware deployment gating (table + forward-only routing demo).

The seed-banded advantage map (P0) IS a deployment gating table: from estimable per-frame context
(local density from beacon positions, interference proxy from mean edge SINR) the system selects the
GNN-vs-heuristic policy and flags floor-limited regimes. This script:

  (A) builds the gating table from result/advantage_map (density x coupling -> predicted
      class + recommended policy + expected F/gap bands, aggregated over the 3 confidence profiles),
  (B) runs a FORWARD-ONLY demo (no training): stream mobility frames at several deployment scenarios,
      estimate density (N / bbox-area) and an interference proxy (mean edge SINR) per frame, look up
      the gate, and VALIDATE by running the best heuristic forward and checking its realized F lands
      in the gate's predicted heuristic-F band (confirming the estimated context indexes the right
      cell). The GNN-advantage MAGNITUDE is cited from the map, not re-run.

Honesty: the gate is 2-D (density x coupling, both estimable). The 3rd axis (confidence profile, ic/iw)
is NOT deployment-observable (ground-truth-referenced) and is logged as a CONFIGURED assumption.

Usage: python -B scripts/analysis/run_gating_demo.py --out result/gating_demo
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

from scripts.analysis.generalization_common import build_topology_layer, env_from_snapshot, evaluate  # noqa: E402
from src.training.training_smoke import _normalized_config, load_training_smoke_config  # noqa: E402
from src.v2x_env.profiles import density_matched_vehicle_config  # noqa: E402
from src.v2x_env.vehicle_snapshot import advance_vehicle_snapshot, generate_vehicle_snapshot  # noqa: E402


def _cell_cfg(base: dict, density: float, profile: str, coupling: float, n: int) -> dict:
    """Same coupling plumbing as run_advantage_map._cell_config (both feature + evaluator side), no training."""
    cfg = dict(base)
    cfg.update({"vehicle_count": int(n), "vehicle_profile": "density_matched",
                "node_density_per_km2": float(density), "training_profile": profile,
                "max_steps": 0, "init_mode": "xavier", "seed": 42})
    physical = dict(cfg.get("physical", {}))
    physical["interference_density_coupling_db"] = float(coupling)
    physical["interference_reference_load"] = 1.0
    cfg["physical"] = physical
    cand = dict(cfg.get("candidate_graph", {}))
    cand["interference_density_coupling_db"] = float(coupling)
    cand["interference_reference_degree"] = 8.0
    cfg["candidate_graph"] = cand
    return cfg


def _policy(profs: list[dict]) -> tuple[str, str]:
    """Recommend a policy from the 3 per-profile cells at a (density, coupling) gate."""
    robust = sum(1 for c in profs if c["cell_class"] == "GNN_ADVANTAGE" and c.get("label_robust"))
    adv = sum(1 for c in profs if c["cell_class"] == "GNN_ADVANTAGE")
    floor = sum(1 for c in profs if c["cell_class"] == "FLOOR_LIMITED")
    if robust >= 2:
        return "USE_GNN", "robust GNN advantage - deploy the learned planner"
    if adv >= 2:
        return "USE_GNN_MARGINAL", "marginal/fragile GNN advantage - GNN preferred, low margin (gap within ~2*sigma_seed)"
    if floor >= 2:
        return "HEURISTIC_OK", "floor-limited - heuristic suffices (graceful parity), GNN optional"
    return "GNN_DEFAULT", "heuristic parity - GNN by default, no loss"


def build_gating_table(map_json: dict) -> dict:
    cells = map_json["cells"]
    densities = sorted({c["density"] for c in cells})
    couplings = sorted({c["coupling_db"] for c in cells})
    entries = []
    for d in densities:
        for k in couplings:
            profs = [c for c in cells if c["density"] == d and c["coupling_db"] == k]
            if not profs:
                continue
            decision, why = _policy(profs)
            fg = [c["F_gnn_mean"] for c in profs]; fgs = [c["F_gnn_std"] for c in profs]
            bh = [c["best_heuristic_mean"] for c in profs]; bhs = [c["best_heuristic_std"] for c in profs]
            gp = [c["gap_mean"] for c in profs]
            entries.append({
                "density": d, "coupling_db": k, "decision": decision, "rationale": why,
                "classes_by_profile": {c["profile"]: c["cell_class"] for c in profs},
                "expected_gnn_F": [min(f - s for f, s in zip(fg, fgs)), max(f + s for f, s in zip(fg, fgs))],
                "expected_heuristic_F": [min(b - s for b, s in zip(bh, bhs)), max(b + s for b, s in zip(bh, bhs))],
                "expected_gap": [min(gp), max(gp)],
            })
    return {"densities": densities, "couplings": couplings,
            "note": "gate is 2-D (density x coupling, both deploy-estimable); profile (ic/iw) is a "
                    "CONFIGURED assumption, not measured; bands aggregate the 3 confidence profiles",
            "entries": entries}


def _lookup(table: dict, density: float, coupling: float) -> dict:
    dens = min(table["densities"], key=lambda x: abs(x - density))
    coup = min(table["couplings"], key=lambda x: abs(x - coupling))
    for e in table["entries"]:
        if e["density"] == dens and e["coupling_db"] == coup:
            return e
    raise KeyError((dens, coup))


def _best_heuristic_F(env, topo, cfg) -> float:
    ef = env["features"]["edge_features"].to(dtype=torch.float64)
    dist = env["features"]["distance_m"].to(dtype=torch.float64)
    cands = {"channel": ef[:, 2], "success": ef[:, 3], "sinr": ef[:, 4], "nearest": -dist}
    best = None
    for sc in cands.values():
        with torch.no_grad():
            f = float(evaluate(sc.reshape(-1), env, topo, cfg)["F_avalanche_node_mean"].mean())
        best = f if best is None else min(best, f)
    return best


def _estimate_density(env) -> float:
    nf = env["features"]["node_features"].detach().cpu().numpy()
    x, y = nf[:, 0] * 600.0, nf[:, 1] * 600.0          # de-normalize positions (m)
    area_km2 = max(((x.max() - x.min()) * (y.max() - y.min())) / 1e6, 1e-6)
    return float(nf.shape[0]) / area_km2


def _mean_sinr_db(env) -> float:
    return float(env["features"]["edge_features"][:, 4].mean().item()) * 40.0   # de-normalize sinr_db/40


def main() -> None:
    p = argparse.ArgumentParser(description="Domain-aware gating table + forward-only routing demo")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--map", default="result/advantage_map/advantage_map.json")
    p.add_argument("--assumed-profile", default="hard_low_confidence",
                   help="CONFIGURED confidence assumption (NOT deploy-observable)")
    p.add_argument("--scenarios", default="100:20,200:20,300:20,100:0,300:0",
                   help="comma list of density:coupling deployment scenarios")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--out", default="result/gating_demo")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    map_json = json.loads((ROOT / args.map).read_text(encoding="utf-8"))
    table = build_gating_table(map_json)

    scenarios = []
    for tok in args.scenarios.split(","):
        d, k = tok.split(":")
        scenarios.append((float(d), float(k)))

    print(f"GATING DEMO - assumed profile (configured, NOT measured) = {args.assumed_profile}", flush=True)
    log = []
    for density, coupling in scenarios:
        cfg = _normalized_config(_cell_cfg(base, density, args.assumed_profile, coupling, args.node_count))
        topo = build_topology_layer(cfg)
        snap0 = generate_vehicle_snapshot(density_matched_vehicle_config(args.node_count, density, seed=42))
        print(f"\nSCENARIO density={int(density)} veh/km2, coupling={int(coupling)} dB", flush=True)
        for fr in range(int(args.frames)):
            snap = advance_vehicle_snapshot(snap0, fr * float(args.dt)) if fr else snap0
            env = env_from_snapshot(snap, cfg, interference_coupling_db=coupling)
            est_d = _estimate_density(env)
            sinr = _mean_sinr_db(env)
            gate = _lookup(table, est_d, coupling)
            realized_h = _best_heuristic_F(env, topo, cfg)
            lo, hi = gate["expected_heuristic_F"]
            in_band = bool(lo - 1e-6 <= realized_h <= hi + 1e-6)
            rec = {"scenario": {"density": density, "coupling_db": coupling}, "frame": fr,
                   "estimated_density": est_d, "mean_sinr_db": sinr,
                   "gate_cell": {"density": gate["density"], "coupling_db": gate["coupling_db"]},
                   "decision": gate["decision"], "rationale": gate["rationale"],
                   "expected_heuristic_F": [lo, hi], "realized_heuristic_F": realized_h,
                   "heuristic_F_in_band": in_band, "expected_gnn_F": gate["expected_gnn_F"],
                   "expected_gap": gate["expected_gap"]}
            log.append(rec)
            print(f"  frame {fr}: est_density={est_d:6.1f} (true {int(density)}), meanSINR={sinr:6.2f}dB -> "
                  f"gate(d{int(gate['density'])},c{int(gate['coupling_db'])}) = {gate['decision']:18s} | "
                  f"heurF={realized_h:.4f} in band[{lo:.4f},{hi:.4f}]={in_band} | "
                  f"expected gap {gate['expected_gap'][0]:+.4f}..{gate['expected_gap'][1]:+.4f}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gating_table.json").write_text(json.dumps(table, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "gating_demo_log.json").write_text(
        json.dumps({"assumed_profile": args.assumed_profile, "frames": log}, indent=2, sort_keys=True),
        encoding="utf-8")
    n_in = sum(1 for r in log if r["heuristic_F_in_band"])
    print(f"\nrouting log: {len(log)} frames, heuristic-F in predicted band: {n_in}/{len(log)}")
    print(f"wrote {out_dir / 'gating_table.json'}")
    print(f"wrote {out_dir / 'gating_demo_log.json'}")


if __name__ == "__main__":
    main()
