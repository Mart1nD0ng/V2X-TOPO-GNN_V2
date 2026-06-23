"""Environment forensics (P0-2): quantify WHY the current sim yields temporal/architecture nulls.

Read-only. Computes the four diagnostics the audit (S1.2 / S3.3) relies on, as reproducible artifacts:

  1. LOS-artifact stats        -> the same-road-segment LOS rule mislabels physically-visible edges.
       nlos_but_collinear_same_roadline_fraction : NLOS edges that are actually on the SAME road line
                                                    (clear line-of-sight down the street, wrongly NLOS).
       nlos_under_30m_fraction                   : very short NLOS edges (near-intersection cross-road).
       rsu_always_los_fraction                   : edges touching an RSU forced LOS regardless of geometry.
  2. Channel distribution      -> success_probability / sinr_db / distance quantiles (the operating regime).
  3. Frame Jaccard             -> candidate edge-set overlap across consecutive mobility frames (redundancy).
  4. Markov-observability      -> predict frame t from frame t-1 via the closed-form linear advance and
                                  measure the residual; ~0 (off wrap boundaries) proves the next frame is a
                                  deterministic function of the current observation, so memory has no edge.

Usage: python scripts/analysis/run_environment_forensics.py --node-count 400 --density 200 --frames 12 --dt 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v2x_env.candidate_graph import build_candidate_graph  # noqa: E402
from src.v2x_env.channel_model import ChannelConfig  # noqa: E402
from src.v2x_env.mobility import MobilityConfig, MobilityStream  # noqa: E402
from src.v2x_env.profiles import density_matched_vehicle_config  # noqa: E402
from src.v2x_env.vehicle_snapshot import advance_vehicle_snapshot, generate_vehicle_snapshot  # noqa: E402

_CAND = {"radius_m": 230.0, "max_candidates_per_node": 8, "cell_size_m": 230.0}
_CH = ChannelConfig(tx_power_dbm=23.0, mcs_threshold_db=8.0, transition_width_db=3.0)


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(np.min(values)), "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)), "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)), "mean": float(np.mean(values)),
    }


def _roadline_membership(snapshot) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-node nearest vertical/horizontal road-line index + on-line mask (within road_half_width)."""
    grid = snapshot["grid"]
    x = np.asarray(snapshot["x"], dtype=float)
    y = np.asarray(snapshot["y"], dtype=float)
    tol = float(grid.config.road_half_width_m) + 1e-6
    dx = np.abs(x[:, None] - grid.x_roads_m[None, :])
    dy = np.abs(y[:, None] - grid.y_roads_m[None, :])
    vline_idx = dx.argmin(axis=1)
    hline_idx = dy.argmin(axis=1)
    on_v = dx.min(axis=1) <= tol
    on_h = dy.min(axis=1) <= tol
    return vline_idx, on_v, hline_idx, on_h


def _los_artifacts(snapshot) -> dict:
    graph = build_candidate_graph(snapshot, _CH, _CAND)
    src, dst = graph.source, graph.target
    los = np.asarray(graph.los_flag, dtype=bool)
    dist = np.asarray(graph.distance_m, dtype=float)
    nlos = ~los
    node_types = np.asarray(snapshot["node_type"], dtype=object)
    rsu = (node_types[src] == "rsu") | (node_types[dst] == "rsu")
    vline_idx, on_v, hline_idx, on_h = _roadline_membership(snapshot)
    collinear = ((on_v[src] & on_v[dst] & (vline_idx[src] == vline_idx[dst]))
                 | (on_h[src] & on_h[dst] & (hline_idx[src] == hline_idx[dst])))
    n_nlos = int(nlos.sum())
    return {
        "edge_count": int(src.size),
        "los_fraction": float(los.mean()) if src.size else 0.0,
        "nlos_fraction": float(nlos.mean()) if src.size else 0.0,
        "nlos_but_collinear_same_roadline_fraction": float((nlos & collinear).sum() / max(n_nlos, 1)),
        "nlos_under_30m_fraction": float((nlos & (dist < 30.0)).sum() / max(src.size, 1)),
        "rsu_always_los_fraction": float(rsu.mean()) if src.size else 0.0,
        "los_max_distance_m": float(dist[los].max()) if los.any() else 0.0,
        "success_probability": _quantiles(np.asarray(graph.success_probability, dtype=float)),
        "sinr_db": _quantiles(np.asarray(graph.sinr_db, dtype=float)),
        "distance_m": _quantiles(dist),
    }


def _edge_set(snapshot) -> set:
    graph = build_candidate_graph(snapshot, _CH, _CAND)
    return set(zip(graph.source.tolist(), graph.target.tolist()))


def _frame_forensics(base, dt: float, frames: int) -> dict:
    stream = MobilityStream(base, MobilityConfig(dt_s=dt, num_frames=frames))
    snaps = [stream.frame_at(t) for t in range(len(stream))]
    edge_sets = [_edge_set(s) for s in snaps]
    jaccards = []
    for a, b in zip(edge_sets, edge_sets[1:]):
        union = len(a | b)
        jaccards.append(len(a & b) / union if union else 1.0)
    # Markov-observability: predict frame t from frame t-1 via the closed-form linear advance.
    bounds = base["bounds"]
    span_x = float(bounds["max_x"]) - float(bounds["min_x"])
    span_y = float(bounds["max_y"]) - float(bounds["min_y"])
    errors, errors_off_wrap = [], []
    for t in range(1, len(snaps)):
        pred = advance_vehicle_snapshot(snaps[t - 1], dt)
        ex = np.abs(np.asarray(pred["x"]) - np.asarray(snaps[t]["x"]))
        ey = np.abs(np.asarray(pred["y"]) - np.asarray(snaps[t]["y"]))
        err = np.sqrt(ex ** 2 + ey ** 2)
        # wrap-boundary nodes produce a large apparent error (mod jump) that is NOT model-relevant.
        off_wrap = (ex < 0.5 * span_x) & (ey < 0.5 * span_y)
        errors.append(float(err.max()))
        if off_wrap.any():
            errors_off_wrap.append(float(err[off_wrap].max()))
    return {
        "num_frames": len(snaps),
        "candidate_edge_jaccard_consecutive": _quantiles(np.asarray(jaccards, dtype=float)),
        "markov_prediction_error_max": float(max(errors)) if errors else 0.0,
        "markov_prediction_error_max_off_wrap": float(max(errors_off_wrap)) if errors_off_wrap else 0.0,
        "markov_fully_observed": bool((max(errors_off_wrap) if errors_off_wrap else 0.0) < 1e-6),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Environment forensics (LOS artifacts, channel, Jaccard, Markov)")
    p.add_argument("--node-count", type=int, default=400)
    p.add_argument("--density", type=float, default=200.0)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--run-name", default="environment_forensics_v1")
    args = p.parse_args()

    base = generate_vehicle_snapshot(
        density_matched_vehicle_config(int(args.node_count), float(args.density), seed=int(args.seed))
    )
    report = {
        "node_count": int(args.node_count), "density_per_km2": float(args.density),
        "seed": int(args.seed), "dt_s": float(args.dt),
        "los_artifacts": _los_artifacts(base),
        "frame_forensics": _frame_forensics(base, float(args.dt), int(args.frames)),
    }
    out_dir = ROOT / "result" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "environment_forensics.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    la = report["los_artifacts"]
    ff = report["frame_forensics"]
    print(f"=== Environment forensics (N={args.node_count}, {args.density} veh/km^2, seed {args.seed}) ===")
    print(f"LOS fraction                              : {la['los_fraction']:.3f}")
    print(f"NLOS-but-collinear-same-roadline fraction : {la['nlos_but_collinear_same_roadline_fraction']:.3f}"
          f"   <- physically-visible edges wrongly NLOS (12 dB penalty artifact)")
    print(f"NLOS under 30 m fraction                  : {la['nlos_under_30m_fraction']:.3f}")
    print(f"RSU-always-LOS fraction                   : {la['rsu_always_los_fraction']:.3f}")
    print(f"success_probability p50 / mean            : {la['success_probability']['p50']:.4f} / {la['success_probability']['mean']:.4f}")
    print(f"candidate-edge Jaccard (consecutive) p50  : {ff['candidate_edge_jaccard_consecutive']['p50']:.3f}")
    print(f"Markov prediction error (off-wrap) max    : {ff['markov_prediction_error_max_off_wrap']:.3e}  "
          f"-> markov_fully_observed={ff['markov_fully_observed']}")
    print(f"\nwrote {out_dir / 'environment_forensics.json'}")


if __name__ == "__main__":
    main()
