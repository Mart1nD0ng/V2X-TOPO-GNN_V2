from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.evaluation import evaluate_v2x_graph_consensus
from src.topology import TopologyConstructionLayer
from src.v2x_env.candidate_graph import build_candidate_graph
from src.v2x_env.channel_model import ChannelConfig
from src.v2x_env.profiles import (
    PRODUCTION_ACTIVE_DEGREE,
    PRODUCTION_CANDIDATE_CAP,
    PRODUCTION_CANDIDATE_RADIUS_M,
    density_matched_vehicle_config,
)
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_OUT = REPO_ROOT / ".agent/tmp/intermediate_reliability_band_probe.json"
DEFAULT_MD_OUT = REPO_ROOT / ".agent/tmp/intermediate_reliability_band_probe.md"


def _as_list(values: Sequence[int | float] | int | float) -> list[int | float]:
    if isinstance(values, (int, float)):
        return [values]
    return list(values)


def _valid_avalanche(k: int, alpha: int, beta: int, rounds: int) -> bool:
    return k > 0 and 0 < alpha <= k and 2 * alpha > k and beta > 0 and rounds > 0 and beta <= rounds


def _topology_for_density(node_count: int, density: float, seed: int) -> tuple[Any, Any]:
    snapshot = generate_vehicle_snapshot(density_matched_vehicle_config(node_count, density, seed=seed))
    candidate = build_candidate_graph(
        snapshot,
        ChannelConfig(tx_power_dbm=23.0, mcs_threshold_db=8.0, transition_width_db=3.0),
        {
            "radius_m": PRODUCTION_CANDIDATE_RADIUS_M,
            "max_candidates_per_node": PRODUCTION_CANDIDATE_CAP,
            "cell_size_m": PRODUCTION_CANDIDATE_RADIUS_M,
        },
    )
    edge_score = torch.as_tensor(candidate.channel_score, dtype=torch.float64)
    topology = TopologyConstructionLayer(
        max_out_degree=PRODUCTION_ACTIVE_DEGREE,
        support_mode="topk",
        topk_backend="segmented_fast",
    )(
        num_nodes=candidate.num_nodes,
        src_index=torch.as_tensor(candidate.source, dtype=torch.long),
        dst_index=torch.as_tensor(candidate.target, dtype=torch.long),
        edge_score=edge_score,
    )
    return candidate, topology


def _row_score(row: Mapping[str, Any], target_min: float, target_max: float) -> float:
    target_center = 0.5 * (target_min + target_max)
    return abs(float(row["active_failure_node_fraction"]) - target_center) + 0.25 * abs(float(row["F"]) - target_center)


def run_intermediate_reliability_band_probe(
    *,
    node_count: int = 500,
    seed: int = 7,
    densities: Sequence[float] = (50.0, 100.0, 150.0, 200.0, 300.0, 450.0, 600.0),
    initial_correct_values: Sequence[float] = (0.25, 0.35, 0.45, 0.55),
    initial_wrong_values: Sequence[float] = (0.10, 0.20, 0.30),
    k_values: Sequence[int] = (3, 5, 7),
    alpha_values: Sequence[int] = (2, 3, 4, 5),
    beta_values: Sequence[int] = (2, 3, 5),
    rounds_values: Sequence[int] = (8, 12, 20),
    active_failure_min: float = 0.1,
    active_failure_max: float = 0.6,
    physical_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if node_count <= 0:
        raise ValueError("node_count must be positive")
    if not 0.0 <= active_failure_min < active_failure_max <= 1.0:
        raise ValueError("active failure band must be ordered in [0, 1]")

    physical = {
        "tx_power_dbm": 23.0,
        "mcs_threshold_db": 8.0,
        "transition_width_db": 3.0,
        "interference_proxy_dbm": -82.0,
        "interference_density_coupling_db": 10.0,
        "interference_reference_load": 1.0,
        "finite_blocklength_reliability": True,
        "payload_bits": 100.0,
        "resource_block_count": 4.0,
        "subcarrier_spacing_hz": 15_000.0,
        "single_hop_delay_s": 0.001,
    }
    if physical_config:
        physical.update(dict(physical_config))

    rows: list[dict[str, Any]] = []
    invalid_avalanche_configs = 0
    invalid_initial_configs = 0
    topology_cache: dict[float, tuple[Any, Any]] = {}
    for density in densities:
        candidate, topology = topology_cache.setdefault(float(density), _topology_for_density(node_count, float(density), seed))
        selected = topology.selected_candidate_index
        distance = torch.as_tensor(candidate.distance_m, dtype=torch.float64).index_select(0, selected)
        los = torch.as_tensor(candidate.los_flag, dtype=torch.float64).index_select(0, selected)
        for initial_correct in initial_correct_values:
            for initial_wrong in initial_wrong_values:
                if initial_correct < 0.0 or initial_wrong < 0.0 or initial_correct + initial_wrong > 1.0:
                    invalid_initial_configs += 1
                    continue
                ic = torch.full((candidate.num_nodes,), float(initial_correct), dtype=torch.float64)
                iw = torch.full((candidate.num_nodes,), float(initial_wrong), dtype=torch.float64)
                for k in k_values:
                    for alpha in alpha_values:
                        for beta in beta_values:
                            for rounds in rounds_values:
                                k_i, alpha_i, beta_i, rounds_i = int(k), int(alpha), int(beta), int(rounds)
                                if not _valid_avalanche(k_i, alpha_i, beta_i, rounds_i):
                                    invalid_avalanche_configs += 1
                                    continue
                                out = evaluate_v2x_graph_consensus(
                                    **topology.as_evaluation_kwargs(),
                                    distance_m=distance,
                                    los_flag=los,
                                    node_initial_correct=ic,
                                    node_initial_wrong=iw,
                                    physical_config=physical,
                                    avalanche_config={
                                        "k": k_i,
                                        "alpha": alpha_i,
                                        "beta": beta_i,
                                        "rounds": rounds_i,
                                        "eps": 1e-6,
                                        "query_support_backend": "fused_fast",
                                        "diagnostics_mode": "lite",
                                    },
                                    energy_config={},
                                )
                                failure = out["node_failure_probability"].detach().to(dtype=torch.float64)
                                active_fraction = float(
                                    ((failure >= active_failure_min) & (failure <= active_failure_max))
                                    .to(dtype=torch.float64)
                                    .mean()
                                    .cpu()
                                )
                                row = {
                                    "density": float(density),
                                    "initial_correct": float(initial_correct),
                                    "initial_wrong": float(initial_wrong),
                                    "k": k_i,
                                    "alpha": alpha_i,
                                    "beta": beta_i,
                                    "rounds": rounds_i,
                                    "C": float(out["C_avalanche_node_mean"].detach().cpu()),
                                    "F": float(out["F_avalanche_node_mean"].detach().cpu()),
                                    "D": float(out["D_avalanche_rounds_mean"].detach().cpu()),
                                    "E": float(out["E_consensus_node_mean"].detach().cpu()),
                                    "active_failure_node_fraction": active_fraction,
                                    "failure_p10": float(torch.quantile(failure, 0.10).cpu()),
                                    "failure_p50": float(torch.quantile(failure, 0.50).cpu()),
                                    "failure_p90": float(torch.quantile(failure, 0.90).cpu()),
                                }
                                row["target_band_hit"] = bool(active_failure_min <= active_fraction <= active_failure_max)
                                rows.append(row)

    target_hits = [row for row in rows if row["target_band_hit"]]
    target_hits.sort(key=lambda row: _row_score(row, active_failure_min, active_failure_max))
    rows.sort(key=lambda row: _row_score(row, active_failure_min, active_failure_max))
    return {
        "probe_name": "intermediate_reliability_band_probe",
        "node_count": int(node_count),
        "seed": int(seed),
        "active_failure_band": [float(active_failure_min), float(active_failure_max)],
        "scan_sizes": {
            "densities": len(list(densities)),
            "initial_correct_values": len(list(initial_correct_values)),
            "initial_wrong_values": len(list(initial_wrong_values)),
            "k_values": len(list(k_values)),
            "alpha_values": len(list(alpha_values)),
            "beta_values": len(list(beta_values)),
            "rounds_values": len(list(rounds_values)),
        },
        "invalid_avalanche_configs": int(invalid_avalanche_configs),
        "invalid_initial_configs": int(invalid_initial_configs),
        "row_count": len(rows),
        "target_hit_count": len(target_hits),
        "best_profiles": target_hits[:20],
        "closest_profiles": rows[:20],
        "found_intermediate_profile": bool(target_hits),
        "physical_config": physical,
    }


def write_intermediate_reliability_band_probe(report: Mapping[str, Any], json_out: str | Path, md_out: str | Path) -> None:
    json_path = Path(json_out)
    md_path = Path(md_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Intermediate Reliability Band Probe",
        "",
        f"- found_intermediate_profile: `{report['found_intermediate_profile']}`",
        f"- target_hit_count: `{report['target_hit_count']}`",
        f"- row_count: `{report['row_count']}`",
        "",
        "| density | ic | iw | k | alpha | beta | rounds | C | F | active_failure_node_fraction | D | E |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("best_profiles") or report.get("closest_profiles", []):
        lines.append(
            f"| {row['density']:.0f} | {row['initial_correct']:.2f} | {row['initial_wrong']:.2f} | "
            f"{row['k']} | {row['alpha']} | {row['beta']} | {row['rounds']} | "
            f"{row['C']:.4f} | {row['F']:.4f} | {row['active_failure_node_fraction']:.3f} | "
            f"{row['D']:.3f} | {row['E']:.3e} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
