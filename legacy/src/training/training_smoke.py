"""Tiny deterministic optimizer smoke for the sparse V2X topology pipeline.

This module is intentionally not a general trainer. It builds one deterministic
environment instance, runs a few optimizer steps, and reports gradient/contract
diagnostics for the existing scorer -> constructor -> evaluator -> loss chain.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from src.evaluation import evaluate_v2x_graph_consensus
from src.losses import compute_coupled_loss
from src.models import (
    HierarchicalGNNScorer,
    apply_dropedge,
    bridge_logit_regularizer,
    budget_target_loss,
    role_balance_loss,
    sector_entropy_loss,
)
from src.models.structural_diagnostics import compute_cap_diagnostics
from src.topology import TopologyConstructionLayer
from src.training.temporal_state import TemporalStateConfig, temporal_state_dims
from src.v2x_env.candidate_graph import build_candidate_graph
from src.v2x_env.channel_model import ChannelConfig
from src.v2x_env.profiles import density_matched_vehicle_config, production_like_density_v0_vehicle_config
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot


AVALANCHE_PROFILES: dict[str, dict[str, int | float]] = {
    "toy": {"k": 1, "alpha": 1, "beta": 2, "rounds": 3, "eps": 0.0},
    "small_realistic": {"k": 5, "alpha": 3, "beta": 5, "rounds": 20, "eps": 1e-6},
}

TRAINING_PROFILES: dict[str, dict[str, float]] = {
    "toy": {"initial_correct": 0.50, "initial_wrong": 0.25},
    "near_target_synthetic": {"initial_correct": 0.65, "initial_wrong": 0.15},
    "high_reliability_synthetic": {"initial_correct": 0.90, "initial_wrong": 0.02},
    # Curriculum/harder operating point (docs/CURRICULUM_TRAINING_DESIGN.md):
    # low initial confidence -> large realized reliability gap to close.
    "hard_low_confidence": {"initial_correct": 0.40, "initial_wrong": 0.25},
    "very_hard_low_confidence": {"initial_correct": 0.25, "initial_wrong": 0.30},
}


def load_training_smoke_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("training smoke config must contain a mapping")
    return data


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    return value if isinstance(value, Mapping) else {}


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    constructor = _section(config, "constructor")
    targets = _section(config, "targets")
    aux = _section(config, "auxiliary_weights")
    diagnostics = _section(config, "diagnostics")
    physical = dict(_section(config, "physical"))
    energy = dict(_section(config, "energy"))
    delay = dict(_section(config, "delay"))
    support_mode = str(config.get("support_mode", config.get("phase_0_support_mode", "all")))
    if support_mode not in {"all", "topk"}:
        raise ValueError("support_mode must be 'all' or 'topk'")
    if support_mode == "all":
        default_cap = constructor.get("phase_0_max_out_degree", None)
    else:
        default_cap = constructor.get("phase_1_max_out_degree", 4)
    topk_backend = str(config.get("topk_backend", constructor.get("topk_backend", "legacy")))
    if topk_backend not in {"legacy", "segmented_fast"}:
        raise ValueError("topk_backend must be 'legacy' or 'segmented_fast'")
    avalanche_profile = str(config.get("avalanche_profile", "toy"))
    if avalanche_profile not in AVALANCHE_PROFILES:
        raise ValueError("avalanche_profile must be 'toy' or 'small_realistic'")
    training_profile = str(config.get("training_profile", "toy"))
    if training_profile not in TRAINING_PROFILES:
        raise ValueError(
            "training_profile must be one of: " + ", ".join(sorted(TRAINING_PROFILES))
        )
    return {
        "seed": int(config.get("seed", 7)),
        "vehicle_count": int(config.get("vehicle_count", 100)),
        "support_mode": support_mode,
        "topk_backend": topk_backend,
        "max_out_degree": config.get("max_out_degree", default_cap),
        "max_steps": int(config.get("max_steps", 3)),
        "learning_rate": float(config.get("learning_rate", 1e-2)),
        "hidden_dim": int(config.get("hidden_dim", 16)),
        "message_layers": int(config.get("message_layers", 1)),
        "init_mode": str(config.get("init_mode", "deterministic")),
        "use_structural_score_bias": bool(config.get("use_structural_score_bias", True)),
        "sector_bias_weight": float(config.get("sector_bias_weight", 0.1)),
        "role_bias_weight": float(config.get("role_bias_weight", 0.1)),
        "bridge_bias_weight": float(config.get("bridge_bias_weight", 0.1)),
        "score_output_gain": float(config.get("score_output_gain", 1.0)),
        "train_budget_head": bool(config.get("train_budget_head", config.get("phase_0_train_budget_head", False))),
        "budget_strategy": str(config.get("budget_strategy", config.get("phase_1_budget_strategy", "fixed"))),
        "budget_cap_mode": str(constructor.get("budget_cap_mode", config.get("budget_cap_mode", "round"))),
        "budget_min_cap": int(constructor.get("budget_min_cap", config.get("budget_min_cap", 1))),
        "budget_max_cap": int(constructor.get("budget_max_cap", config.get("budget_max_cap", 4))),
        "budget_target": float(config.get("budget_target", 2.5)),
        "reliability_failure_target": float(
            config.get("reliability_failure_target", targets.get("reliability_failure_target", 1e-2))
        ),
        "delay_target_rounds": float(config.get("delay_target_rounds", targets.get("delay_target_rounds", 1.0))),
        "energy_target_j": float(config.get("energy_target_j", targets.get("energy_target_j", 1e-4))),
        "budget_target_weight": float(config.get("budget_target_weight", aux.get("budget_target_weight", 0.0))),
        "sector_entropy_weight": float(config.get("sector_entropy_weight", aux.get("sector_entropy_weight", 0.0))),
        "role_balance_weight": float(config.get("role_balance_weight", aux.get("role_balance_weight", 0.0))),
        "bridge_regularizer_weight": float(
            config.get("bridge_regularizer_weight", aux.get("bridge_regularizer_weight", 0.0))
        ),
        "training_profile": training_profile,
        "avalanche_profile": avalanche_profile,
        # #1/#4: number of SSMC quenched-disorder copies for the TRAINING evaluator. 1 = mean-field
        # (legacy, blind to the query-spread/effective-degree lever); >=21 = quenched closure that
        # rewards spreading, so the planner learns per-node autonomous effective degree.
        "quenched_quadrature": int(config.get("quenched_quadrature", 1)),
        # P0-1 currency unification: the EVALUATION (headline F) quadrature can differ from the
        # training one, so we can train at Q=11 (speed) but report the headline F at Q=21 (the
        # converged quenched currency). Defaults to quenched_quadrature -> single-Q (byte-identical).
        "eval_quenched_quadrature": int(
            config.get("eval_quenched_quadrature", config.get("quenched_quadrature", 1))
        ),
        # #5 spatially-structured initial confidence: when enabled, ic_i varies with distance to an
        # event anchor (near the event -> better initial info), with the spatial MEAN anchored to the
        # profile scalar so the baseline reliability is unchanged. Empty/disabled -> uniform (legacy).
        "spatial_ic": dict(_section(config, "spatial_ic")),
        # #3 consensus reliability as a carried temporal state feature (opt-in extra node channels).
        "temporal_state": dict(_section(config, "temporal_state")),
        # Axis A structural-encoder upgrades (docs/MODEL_ARCHITECTURE_DESIGN.md). All OFF by default
        # -> byte-identical to the legacy mean-pool MLP-MPNN scorer.
        "attention_heads": int(config.get("attention_heads", 0)),
        "attention_negative_slope": float(config.get("attention_negative_slope", 0.2)),
        "gcnii_alpha": float(config.get("gcnii_alpha", 0.0)),
        "gcnii_lambda": float(config.get("gcnii_lambda", 1.0)),
        "jk_mode": str(config.get("jk_mode", "last")),
        "channel_recalibration": str(config.get("channel_recalibration", "none")),
        "se_reduction": int(config.get("se_reduction", 4)),
        "dropedge_prob": float(config.get("dropedge_prob", 0.0)),
        "scale_invariant_backward": bool(config.get("scale_invariant_backward", False)),
        "scale_reference_node_count": float(config.get("scale_reference_node_count", 100.0)),
        "gradient_mode": str(config.get("gradient_mode", "selected_row_softmax")),
        "straight_through_temperature": config.get("straight_through_temperature", None),
        "optimizer": str(config.get("optimizer", "sgd")),
        "vehicle_profile": str(config.get("vehicle_profile", "fixed_grid")),
        "node_density_per_km2": config.get("node_density_per_km2", None),
        "learnable_score_gain": bool(config.get("learnable_score_gain", False)),
        "score_standardization": bool(config.get("score_standardization", False)),
        "gradient_clip_norm": config.get("gradient_clip_norm", None),
        "report_pre_clip_grad_norm": bool(config.get("report_pre_clip_grad_norm", diagnostics.get("report_pre_clip_grad_norm", True))),
        "report_post_clip_grad_norm": bool(config.get("report_post_clip_grad_norm", diagnostics.get("report_post_clip_grad_norm", True))),
        "curriculum_phase_steps": int(config.get("curriculum_phase_steps", 2)),
        "physical_config": physical,
        "energy_config": energy,
        "delay_config": delay,
        # P1-1 standard-environment plumbing: optional `channel:` and `candidate_graph:` config
        # sections flow into _make_environment (candidate graph + feature channel) AND, for the
        # pathloss-model keys, into the evaluator physical config (one switch flips BOTH sides).
        # Empty/absent sections -> the historical hardcoded defaults (byte-identical).
        "channel_config": dict(_section(config, "channel")),
        "candidate_config": dict(_section(config, "candidate_graph")),
    }


def _vehicle_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(config["seed"]),
        "vehicle_count": int(config["vehicle_count"]),
        "grid": {
            "block_length_m": 120.0,
            "block_width_m": 120.0,
            "road_count_x": 5,
            "road_count_y": 5,
            "lanes_per_direction": 1,
            "lane_width_m": 3.5,
        },
        "speed": {"mean_mps": 12.0, "std_mps": 2.0, "min_mps": 0.0, "max_mps": 25.0},
    }


def _coarse_region_id(snapshot: Mapping[str, Any]) -> tuple[np.ndarray, int]:
    x = np.asarray(snapshot["x"], dtype=float)
    y = np.asarray(snapshot["y"], dtype=float)
    bounds = snapshot["bounds"]
    x_span = max(float(bounds["max_x"]) - float(bounds["min_x"]), 1.0)
    y_span = max(float(bounds["max_y"]) - float(bounds["min_y"]), 1.0)
    x_bin = np.clip(np.floor((x - float(bounds["min_x"])) / x_span * 4.0), 0, 3).astype(int)
    y_bin = np.clip(np.floor((y - float(bounds["min_y"])) / y_span * 4.0), 0, 3).astype(int)
    return x_bin * 4 + y_bin, 16


def _edge_sector_id(snapshot: Mapping[str, Any], source: np.ndarray, target: np.ndarray, num_sectors: int) -> np.ndarray:
    x = np.asarray(snapshot["x"], dtype=float)
    y = np.asarray(snapshot["y"], dtype=float)
    angles = np.arctan2(y[target] - y[source], x[target] - x[source])
    wrapped = (angles + 2.0 * np.pi) % (2.0 * np.pi)
    return np.floor(wrapped / (2.0 * np.pi / float(num_sectors))).astype(int) % num_sectors


def _build_feature_tensors(snapshot: Mapping[str, Any], candidate: Any) -> dict[str, torch.Tensor | int]:
    x = np.asarray(snapshot["x"], dtype=float)
    y = np.asarray(snapshot["y"], dtype=float)
    speed = np.asarray(snapshot["speed_mps"], dtype=float)
    heading = np.asarray(snapshot["heading"], dtype=float)
    region_id_np, num_regions = _coarse_region_id(snapshot)
    edge_sector_np = _edge_sector_id(snapshot, candidate.source, candidate.target, 8)
    edge_cross_np = region_id_np[candidate.source] != region_id_np[candidate.target]
    node_features = torch.as_tensor(
        np.stack(
            [
                x / 600.0,
                y / 600.0,
                speed / 30.0,
                np.sin(np.deg2rad(heading)),
                np.cos(np.deg2rad(heading)),
            ],
            axis=1,
        ),
        dtype=torch.float64,
    )
    edge_features = torch.as_tensor(
        np.stack(
            [
                candidate.distance_m / 250.0,
                candidate.los_flag.astype(float),
                candidate.channel_score,
                candidate.success_probability,
                candidate.sinr_db / 40.0,
            ],
            axis=1,
        ),
        dtype=torch.float64,
    )
    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "src_index": torch.as_tensor(candidate.source, dtype=torch.long),
        "dst_index": torch.as_tensor(candidate.target, dtype=torch.long),
        "distance_m": torch.as_tensor(candidate.distance_m, dtype=torch.float64),
        "los_flag": torch.as_tensor(candidate.los_flag, dtype=torch.float64),
        "node_xy": torch.as_tensor(np.stack([x, y], axis=1), dtype=torch.float64),
        "region_id": torch.as_tensor(region_id_np, dtype=torch.long),
        "num_regions": int(num_regions),
        "edge_sector_id": torch.as_tensor(edge_sector_np, dtype=torch.long),
        "edge_is_cross_region": torch.as_tensor(edge_cross_np, dtype=torch.bool),
    }


def _loss_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "reliability_failure_target": float(config["reliability_failure_target"]),
        "reliability_tail_failure_target": float(config["reliability_failure_target"]),
        "reliability_tau": 0.5,
        "delay_target_rounds": float(config["delay_target_rounds"]),
        "delay_p90_target_rounds": float(config["delay_target_rounds"]) + 1.0,
        "delay_tau": 1.0,
        "energy_target_j": float(config["energy_target_j"]),
        "energy_p90_target_j": float(config["energy_target_j"]) * 2.0,
        "energy_tau": max(float(config["energy_target_j"]) * 10.0, 1e-4),
        "reliability_tail_mode": "max",
        "use_reliability_gate": True,
        "scale_invariant_backward": bool(config.get("scale_invariant_backward", False)),
        "scale_reference_node_count": float(config.get("scale_reference_node_count", 100.0)),
    }


def _avalanche_config(config: Mapping[str, Any], *, eval_mode: bool = False) -> dict[str, Any]:
    profile = dict(AVALANCHE_PROFILES[str(config["avalanche_profile"])])
    profile["reliability_failure_target"] = float(config["reliability_failure_target"])
    profile["reliability_boundary_factor"] = 10.0
    # #1/#4: SSMC quenched-disorder closed form. 1 = mean-field (legacy, blind to the query-spread
    # lever); >=21 = converged quenched closure (sees it -> training learns autonomous effective degree).
    # P0-1: in eval_mode the (possibly higher) eval_quenched_quadrature is used so the headline F is
    # reported at the converged currency even when training runs at a cheaper Q.
    if eval_mode:
        profile["quenched_quadrature"] = int(
            config.get("eval_quenched_quadrature", config.get("quenched_quadrature", 1))
        )
    else:
        profile["quenched_quadrature"] = int(config.get("quenched_quadrature", 1))
    return profile


def _evaluator_physical_config(config: Mapping[str, Any]) -> dict[str, Any]:
    physical = {
        "tx_power_dbm": 23.0,
        "mcs_threshold_db": 8.0,
        "transition_width_db": 3.0,
        "interference_proxy_dbm": -82.0,
    }
    # P0-1 one-switch consistency: the pathloss-model keys from the `channel:` section (which the
    # candidate graph consumes) also feed the evaluator, so setting channel.pathloss_model=tr37885
    # flips BOTH sides at once. Explicit `physical:` keys still override. Absent -> byte-identical.
    channel = config.get("channel_config", {})
    if isinstance(channel, Mapping):
        for key in ("pathloss_model", "scenario", "nlosv_extra_db", "nlosv_extra_std_db", "carrier_frequency_ghz", "nlos_penalty_db"):
            if key in channel:
                physical[key] = channel[key]
    extra = config.get("physical_config", {})
    if isinstance(extra, Mapping):
        physical.update(extra)
    return physical


def _evaluator_energy_config(config: Mapping[str, Any]) -> dict[str, Any]:
    extra = config.get("energy_config", {})
    energy = dict(extra) if isinstance(extra, Mapping) else {}
    physical = config.get("physical_config", {})
    finite_blocklength = isinstance(physical, Mapping) and bool(physical.get("finite_blocklength_reliability", False))
    if not finite_blocklength and "packet_duration_s" not in energy:
        energy["packet_duration_s"] = 0.001
    return energy


def _evaluator_delay_config(config: Mapping[str, Any]) -> dict[str, Any]:
    extra = config.get("delay_config", {})
    return dict(extra) if isinstance(extra, Mapping) else {}


def _initial_preferences(
    config: Mapping[str, Any], num_nodes: int, node_xy: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-node initial preferences. Uniform from the profile by default; SPATIALLY STRUCTURED
    (#5) when ``config['spatial_ic'].enabled`` and node positions are provided: nodes near an event
    anchor get higher initial confidence (better local info), nodes far away stay undecided.

    The spatial modulation is MEAN-PRESERVING by construction (``ic`` is the profile scalar plus a
    zero-mean spatial term), so the baseline reliability is unchanged and any change is attributable
    to spatial STRUCTURE, not to a shifted average confidence. ``iw`` stays uniform (far nodes lack
    info -> they are undecided, not wrong). The amplitude is bounded into ``[eps, 1 - iw]`` so no
    clamping is needed. Empty/disabled config or missing positions -> legacy uniform (byte-identical).
    """
    profile = TRAINING_PROFILES[str(config["training_profile"])]
    ic_mean = float(profile["initial_correct"])
    iw_mean = float(profile["initial_wrong"])
    spatial = config.get("spatial_ic", {})
    enabled = isinstance(spatial, Mapping) and bool(spatial.get("enabled", False))
    if not enabled or node_xy is None or num_nodes <= 0:
        return (
            torch.full((num_nodes,), ic_mean, dtype=torch.float64),
            torch.full((num_nodes,), iw_mean, dtype=torch.float64),
        )
    xy = node_xy.to(dtype=torch.float64).reshape(num_nodes, 2)
    lo = xy.amin(dim=0)
    span = (xy.amax(dim=0) - lo).clamp_min(1.0)
    anchor = spatial.get("anchor_xy_frac", (0.15, 0.15))
    event = lo + xy.new_tensor([float(anchor[0]), float(anchor[1])]) * span
    diag = torch.sqrt((span * span).sum()).clamp_min(1.0)
    decay = float(spatial.get("decay_length_frac", 0.35)) * float(diag)
    proximity = torch.exp(-torch.sqrt(((xy - event) ** 2).sum(dim=1)) / max(decay, 1e-6))
    centered = proximity - proximity.mean()  # zero-mean spatial term -> mean-preserving
    denom = centered.abs().amax().clamp_min(1e-12)
    headroom = max(min(ic_mean - 1e-3, 1.0 - iw_mean - ic_mean), 0.0)
    amplitude = float(spatial.get("contrast", 1.0)) * float(spatial.get("spread", 0.8)) * headroom
    correct = (ic_mean + amplitude * (centered / denom)).clamp(1e-6, 1.0 - iw_mean)
    wrong = torch.full((num_nodes,), iw_mean, dtype=torch.float64)
    return correct, wrong


def _structural_auxiliary_loss(
    scorer_output: Mapping[str, Any],
    config: Mapping[str, Any],
    edge_is_cross_region: torch.Tensor,
) -> torch.Tensor:
    edge_score = scorer_output["edge_score"]
    if not isinstance(edge_score, torch.Tensor):
        raise TypeError("scorer output must include edge_score tensor")
    aux_terms: list[torch.Tensor] = []
    if (
        config["budget_strategy"] == "auxiliary"
        and bool(config["train_budget_head"])
        and float(config["budget_target_weight"]) > 0.0
    ):
        aux_terms.append(
            edge_score.new_tensor(float(config["budget_target_weight"]))
            * budget_target_loss(scorer_output["node_budget_expected"], float(config["budget_target"]))
        )
    if float(config["sector_entropy_weight"]) > 0.0:
        aux_terms.append(
            edge_score.new_tensor(float(config["sector_entropy_weight"]))
            * sector_entropy_loss(scorer_output["sector_preference_logits"], mode="discourage_entropy")
        )
    if float(config["role_balance_weight"]) > 0.0:
        role_logits = scorer_output["node_role_logits"]
        if isinstance(role_logits, torch.Tensor) and role_logits.ndim == 2 and role_logits.shape[1] > 0:
            target = role_logits.new_full((role_logits.shape[1],), 1.0 / float(role_logits.shape[1]))
            aux_terms.append(
                edge_score.new_tensor(float(config["role_balance_weight"]))
                * role_balance_loss(role_logits, target)
            )
    if float(config["bridge_regularizer_weight"]) > 0.0:
        aux_terms.append(
            edge_score.new_tensor(float(config["bridge_regularizer_weight"]))
            * bridge_logit_regularizer(scorer_output["region_bridge_logits"], edge_is_cross_region)
        )
    if not aux_terms:
        return edge_score.new_tensor(0.0)
    return torch.stack(aux_terms).sum()


def _grad_report(model: torch.nn.Module) -> dict[str, float | bool]:
    total_sq = 0.0
    parameter_grad_max = 0.0
    edge_path_grad_max = 0.0
    sector_grad_max = 0.0
    role_grad_max = 0.0
    bridge_grad_max = 0.0
    budget_grad_max = 0.0
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        finite = finite and bool(torch.all(torch.isfinite(grad)).cpu())
        grad_double = grad.to(dtype=torch.float64)
        grad_sq = float(torch.sum(grad_double * grad_double).cpu())
        total_sq += grad_sq
        grad_max = float(torch.max(torch.abs(grad_double)).cpu()) if grad_double.numel() else 0.0
        parameter_grad_max = max(parameter_grad_max, grad_max)
        if name.startswith(("node_encoder", "edge_encoder", "message_blocks", "edge_score_head")):
            edge_path_grad_max = max(edge_path_grad_max, grad_max)
        if name.startswith("sector_head"):
            sector_grad_max = max(sector_grad_max, grad_max)
        if name.startswith("role_head"):
            role_grad_max = max(role_grad_max, grad_max)
        if name.startswith("region_bridge_head"):
            bridge_grad_max = max(bridge_grad_max, grad_max)
        if name.startswith("budget_head"):
            budget_grad_max = max(budget_grad_max, grad_max)
    structural_grad_max = max(sector_grad_max, role_grad_max, bridge_grad_max)
    norm = total_sq ** 0.5
    return {
        "gradient_norm_total": norm,
        "gradient_norm_model": norm,
        "parameter_grad_max": parameter_grad_max,
        "edge_path_grad_max": edge_path_grad_max,
        "sector_grad_max": sector_grad_max,
        "role_grad_max": role_grad_max,
        "bridge_grad_max": bridge_grad_max,
        "structural_grad_max": structural_grad_max,
        "budget_grad_max": budget_grad_max,
        "gradients_finite": finite,
    }


def _tensor_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        return float(tensor.mean().cpu()) if tensor.ndim else float(tensor.cpu())
    if isinstance(value, bool):
        return float(value)
    return float(value)


def _finite_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    finite = 0
    nonfinite = 0
    for record in records:
        for value in record.values():
            if isinstance(value, bool) or isinstance(value, str):
                continue
            if isinstance(value, (int, float)):
                if np.isfinite(float(value)):
                    finite += 1
                else:
                    nonfinite += 1
    return finite, nonfinite


def _trend_report(step_records: list[dict[str, Any]]) -> dict[str, Any]:
    total_losses = [float(item["total_loss"]) for item in step_records]
    initial = total_losses[0]
    final = total_losses[-1]
    best_step = int(min(range(len(total_losses)), key=lambda idx: total_losses[idx]))
    nonincreasing = sum(
        1 for idx in range(1, len(total_losses)) if total_losses[idx] <= total_losses[idx - 1] + 1e-12
    )
    finite_count, nonfinite_count = _finite_counts(step_records)
    return {
        "loss_delta": final - initial,
        "loss_relative_delta": (final - initial) / max(abs(initial), 1e-12),
        "loss_nonincreasing_step_count": int(nonincreasing),
        "best_step": best_step,
        "gradient_norm_total_per_step": [float(item["gradient_norm_total"]) for item in step_records],
        "gradient_norm_model_per_step": [float(item["gradient_norm_model"]) for item in step_records],
        "max_grad_per_step": [float(item["edge_score_grad_or_parameter_grad_max"]) for item in step_records],
        "finite_metric_count": int(finite_count),
        "nonfinite_metric_count": int(nonfinite_count),
        "C_mean_delta": float(step_records[-1]["C_avalanche_node_mean"]) - float(step_records[0]["C_avalanche_node_mean"]),
        "F_mean_delta": float(step_records[-1]["F_mean"]) - float(step_records[0]["F_mean"]),
        "D_mean_delta": float(step_records[-1]["D_avalanche_rounds_mean"]) - float(step_records[0]["D_avalanche_rounds_mean"]),
        "E_mean_delta": float(step_records[-1]["E_consensus_node_mean"]) - float(step_records[0]["E_consensus_node_mean"]),
    }


def _parameter_change(model: torch.nn.Module, initial_parameters: Mapping[str, torch.Tensor]) -> tuple[float, float]:
    max_change = 0.0
    sum_sq = 0.0
    for name, parameter in model.named_parameters():
        delta = parameter.detach() - initial_parameters[name]
        max_change = max(max_change, float(torch.max(torch.abs(delta)).cpu()))
        delta_double = delta.to(dtype=torch.float64)
        sum_sq += float(torch.sum(delta_double * delta_double).cpu())
    return max_change, sum_sq ** 0.5


def _make_environment(config: Mapping[str, Any]) -> tuple[Any, dict[str, torch.Tensor | int]]:
    # Historical hardcoded channel defaults; an optional `channel:` config section overrides them
    # (e.g. pathloss_model: tr37885 for the standard paper environment). Absent -> byte-identical.
    channel_overrides = config.get("channel_config", {}) or {}
    channel_config = ChannelConfig.from_mapping(
        {"tx_power_dbm": 23.0, "mcs_threshold_db": 8.0, "transition_width_db": 3.0, **channel_overrides}
    )
    vehicle_profile = str(config.get("vehicle_profile", "fixed_grid"))
    if vehicle_profile == "production_like_density_v0":
        # Realistic spatial density (300 veh/km^2): spreads N nodes over a real
        # area so the candidate graph stays sparse and reliability is marginal /
        # topology-sensitive (F ~ 0.06-0.12), instead of the fixed ~600x600m grid
        # that makes large N hyper-dense and saturates consensus to F=0.
        vehicle_cfg = production_like_density_v0_vehicle_config(
            int(config["vehicle_count"]), seed=int(config["seed"])
        )
        snapshot = generate_vehicle_snapshot(vehicle_cfg)
    elif vehicle_profile == "fixed_grid":
        snapshot = generate_vehicle_snapshot(_vehicle_config(config))
    elif vehicle_profile == "density_matched":
        # Operating-point profile: spread N nodes to a requested spatial density
        # (node_density_per_km2). Used to land the non-trivial, load-coupled operating
        # point from Track A (docs/COUPLING_AND_OPERATING_POINT_DESIGN.md).
        density = config.get("node_density_per_km2", None)
        if density is None:
            raise ValueError("vehicle_profile 'density_matched' requires node_density_per_km2")
        vehicle_cfg = density_matched_vehicle_config(
            int(config["vehicle_count"]), float(density), seed=int(config["seed"])
        )
        snapshot = generate_vehicle_snapshot(vehicle_cfg)
    else:
        raise ValueError(
            "vehicle_profile must be 'fixed_grid', 'production_like_density_v0', or 'density_matched'"
        )
    candidate_overrides = config.get("candidate_config", {}) or {}
    candidate = build_candidate_graph(
        snapshot,
        channel_config,
        {"radius_m": 230.0, "max_candidates_per_node": 8, "cell_size_m": 230.0, **candidate_overrides},
    )
    if candidate.edge_count <= 0:
        raise ValueError("training smoke candidate graph has no edges")
    return candidate, _build_feature_tensors(snapshot, candidate)


_PER_EDGE_FEATURE_KEYS = (
    "edge_features",
    "src_index",
    "dst_index",
    "distance_m",
    "los_flag",
    "edge_sector_id",
    "edge_is_cross_region",
)


def _dropedge_features(
    features: Mapping[str, Any], drop_prob: float, generator: torch.Generator | None
) -> dict[str, Any]:
    """A4 DropEdge (train-only): return a per-step view of ``features`` with a random subset of
    candidate edges dropped. ``drop_prob<=0`` returns the features unchanged (byte-identical). Every
    per-edge tensor is subset by the SAME keep mask so the scorer, constructor, and evaluator stay
    aligned; per-node tensors pass through untouched (docs/MODEL_ARCHITECTURE_DESIGN.md A4)."""
    src = features["src_index"]
    if drop_prob <= 0.0 or int(src.numel()) == 0:
        return dict(features)
    _, _, _, keep = apply_dropedge(
        src, features["dst_index"], features["edge_features"], drop_prob, generator=generator
    )
    out = dict(features)
    for key in _PER_EDGE_FEATURE_KEYS:
        out[key] = features[key][keep]
    return out


def _make_model(config: Mapping[str, Any]) -> HierarchicalGNNScorer:
    # #3: opt-in carried temporal-state channels widen the node feature input (5 -> 5 + dims).
    node_dim = 5 + temporal_state_dims(TemporalStateConfig.from_mapping(config.get("temporal_state")))
    return HierarchicalGNNScorer(
        node_dim,
        5,
        hidden_dim=int(config["hidden_dim"]),
        message_layers=int(config["message_layers"]),
        init_mode=str(config["init_mode"]),
        use_structural_score_bias=bool(config["use_structural_score_bias"]),
        sector_bias_weight=float(config["sector_bias_weight"]),
        role_bias_weight=float(config["role_bias_weight"]),
        bridge_bias_weight=float(config["bridge_bias_weight"]),
        score_output_gain=float(config.get("score_output_gain", 1.0)),
        learnable_score_gain=bool(config.get("learnable_score_gain", False)),
        score_standardization=bool(config.get("score_standardization", False)),
        attention_heads=int(config.get("attention_heads", 0)),
        attention_negative_slope=float(config.get("attention_negative_slope", 0.2)),
        gcnii_alpha=float(config.get("gcnii_alpha", 0.0)),
        gcnii_lambda=float(config.get("gcnii_lambda", 1.0)),
        jk_mode=str(config.get("jk_mode", "last")),
        channel_recalibration=str(config.get("channel_recalibration", "none")),
        se_reduction=int(config.get("se_reduction", 4)),
    ).double()


def _evaluate_model_snapshot(
    *,
    model: HierarchicalGNNScorer,
    topology_layer: TopologyConstructionLayer,
    candidate: Any,
    features: Mapping[str, torch.Tensor | int],
    config: Mapping[str, Any],
    fixed_caps: torch.Tensor | None,
    initial_correct: torch.Tensor,
    initial_wrong: torch.Tensor,
) -> dict[str, Any]:
    with torch.no_grad():
        scorer_output = model(
            num_nodes=candidate.num_nodes,
            src_index=features["src_index"],
            dst_index=features["dst_index"],
            node_features=features["node_features"],
            edge_features=features["edge_features"],
            region_id=features["region_id"],
            num_regions=features["num_regions"],
            edge_sector_id=features["edge_sector_id"],
            edge_is_cross_region=features["edge_is_cross_region"],
            use_structural_score_bias=bool(config["use_structural_score_bias"]),
            sector_bias_weight=float(config["sector_bias_weight"]),
            role_bias_weight=float(config["role_bias_weight"]),
            bridge_bias_weight=float(config["bridge_bias_weight"]),
        )
        topology = topology_layer(
            num_nodes=candidate.num_nodes,
            src_index=features["src_index"],
            dst_index=features["dst_index"],
            edge_score=scorer_output["edge_score"],
            per_node_budget=fixed_caps,
        )
        selected = topology.selected_candidate_index
        evaluator_output = evaluate_v2x_graph_consensus(
            **topology.as_evaluation_kwargs(),
            distance_m=features["distance_m"].index_select(0, selected),
            los_flag=features["los_flag"].index_select(0, selected),
            node_initial_correct=initial_correct,
            node_initial_wrong=initial_wrong,
            physical_config=_evaluator_physical_config(config),
            avalanche_config=_avalanche_config(config, eval_mode=True),
            energy_config=_evaluator_energy_config(config),
            delay_config=_evaluator_delay_config(config),
        )
        loss_output = compute_coupled_loss(evaluator_output, _loss_config(config))
        structural_aux = _structural_auxiliary_loss(
            scorer_output,
            config,
            features["edge_is_cross_region"],
        )
        primary_loss = loss_output["total_loss"]
        total_loss = primary_loss + structural_aux
        return {
            "total_loss": _tensor_float(total_loss),
            "L_primary": _tensor_float(primary_loss),
            "L_R": _tensor_float(loss_output["L_R"]),
            "L_D": _tensor_float(loss_output["L_D"]),
            "L_E": _tensor_float(loss_output["L_E"]),
            "L_aux_structural": _tensor_float(structural_aux),
            "C_avalanche_node_mean": _tensor_float(evaluator_output["C_avalanche_node_mean"]),
            "F_avalanche_node_mean": _tensor_float(evaluator_output["F_avalanche_node_mean"]),
            "F_tail": _tensor_float(loss_output["F_tail"]),
            "F_avalanche_node_p90": _tensor_float(evaluator_output["F_avalanche_node_p90"]),
            "D_avalanche_rounds_mean": _tensor_float(evaluator_output["D_avalanche_rounds_mean"]),
            "D_avalanche_rounds_p90": _tensor_float(evaluator_output["D_avalanche_rounds_p90"]),
            "E_consensus_node_mean": _tensor_float(evaluator_output["E_consensus_node_mean"]),
            "E_consensus_node_p90": _tensor_float(evaluator_output["E_consensus_node_p90"]),
            "gradient_coverage_fraction": _tensor_float(topology.diagnostics["gradient_coverage_fraction"]),
            "active_edge_count": _tensor_float(topology.diagnostics["active_edge_count"]),
            "selected_candidate_index": [int(value) for value in selected.detach().cpu().tolist()],
        }


def _run_training_phase(
    config: Mapping[str, Any],
    *,
    model: HierarchicalGNNScorer | None = None,
) -> tuple[dict[str, Any], HierarchicalGNNScorer]:
    cfg = _normalized_config(config)
    if cfg["max_steps"] <= 0:
        raise ValueError("max_steps must be positive")
    if cfg["learning_rate"] <= 0.0:
        raise ValueError("learning_rate must be positive")
    if cfg["budget_strategy"] not in {"fixed", "auxiliary"}:
        raise ValueError("budget_strategy must be 'fixed' or 'auxiliary'")
    clip_value = cfg["gradient_clip_norm"]
    gradient_clip_norm = None if clip_value is None else float(clip_value)
    if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when provided")

    torch.manual_seed(int(cfg["seed"]))
    candidate, features = _make_environment(cfg)
    if model is None:
        model = _make_model(cfg)
    optimizer_name = str(cfg.get("optimizer", "sgd")).lower()
    if optimizer_name == "adam":
        # Adam is robust to the absolute gradient magnitude, which the P2 score
        # gain inflates; it normalizes per-parameter so the now-strong, scale-
        # invariant signal converges stably without hand-tuned per-scale LRs.
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=float(cfg["learning_rate"]))
    else:
        raise ValueError("optimizer must be 'sgd' or 'adam'")
    initial_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    max_out_degree_value = cfg["max_out_degree"]
    max_out_degree = None if max_out_degree_value is None else int(max_out_degree_value)
    topology_layer = TopologyConstructionLayer(
        max_out_degree=max_out_degree,
        support_mode=str(cfg["support_mode"]),
        temperature=1.0,
        topk_backend=str(cfg["topk_backend"]),
        gradient_mode=str(cfg.get("gradient_mode", "selected_row_softmax")),
        straight_through_temperature=cfg.get("straight_through_temperature", None),
    )
    fixed_caps: torch.Tensor | None = None
    cap_diagnostics: dict[str, torch.Tensor] | None = None
    if cfg["support_mode"] == "topk" and max_out_degree is not None:
        fixed_caps = torch.full((candidate.num_nodes,), max_out_degree, dtype=torch.long)
        cap_diagnostics = compute_cap_diagnostics(
            fixed_caps,
            min_cap=int(cfg["budget_min_cap"]),
            max_cap=max(int(cfg["budget_max_cap"]), max_out_degree),
        )

    initial_correct, initial_wrong = _initial_preferences(cfg, candidate.num_nodes, features.get("node_xy"))
    initial_metric_snapshot = _evaluate_model_snapshot(
        model=model,
        topology_layer=topology_layer,
        candidate=candidate,
        features=features,
        config=cfg,
        fixed_caps=fixed_caps,
        initial_correct=initial_correct,
        initial_wrong=initial_wrong,
    )
    step_records: list[dict[str, Any]] = []
    loss_finite_all = True
    gradients_finite_all = True
    optimizer_steps_completed = 0
    # A4 DropEdge (train-only): a per-step random subset of candidate edges. drop_prob<=0 -> off /
    # byte-identical. Seeded so a given (seed, dropedge_prob) run is reproducible.
    dropedge_prob = float(cfg.get("dropedge_prob", 0.0))
    dropedge_generator = (
        torch.Generator().manual_seed(int(cfg["seed"]) + 104729) if dropedge_prob > 0.0 else None
    )
    for step in range(int(cfg["max_steps"])):
        optimizer.zero_grad(set_to_none=True)
        step_features = _dropedge_features(features, dropedge_prob, dropedge_generator)
        scorer_output = model(
            num_nodes=candidate.num_nodes,
            src_index=step_features["src_index"],
            dst_index=step_features["dst_index"],
            node_features=step_features["node_features"],
            edge_features=step_features["edge_features"],
            region_id=step_features["region_id"],
            num_regions=step_features["num_regions"],
            edge_sector_id=step_features["edge_sector_id"],
            edge_is_cross_region=step_features["edge_is_cross_region"],
            use_structural_score_bias=bool(cfg["use_structural_score_bias"]),
            sector_bias_weight=float(cfg["sector_bias_weight"]),
            role_bias_weight=float(cfg["role_bias_weight"]),
            bridge_bias_weight=float(cfg["bridge_bias_weight"]),
        )
        edge_score = scorer_output["edge_score"]
        if isinstance(edge_score, torch.Tensor):
            edge_score.retain_grad()
        topology = topology_layer(
            num_nodes=candidate.num_nodes,
            src_index=step_features["src_index"],
            dst_index=step_features["dst_index"],
            edge_score=edge_score,
            per_node_budget=fixed_caps,
        )
        selected = topology.selected_candidate_index
        evaluator_output = evaluate_v2x_graph_consensus(
            **topology.as_evaluation_kwargs(),
            distance_m=step_features["distance_m"].index_select(0, selected),
            los_flag=step_features["los_flag"].index_select(0, selected),
            node_initial_correct=initial_correct,
            node_initial_wrong=initial_wrong,
            physical_config=_evaluator_physical_config(cfg),
            avalanche_config=_avalanche_config(cfg),
            energy_config=_evaluator_energy_config(cfg),
            delay_config=_evaluator_delay_config(cfg),
        )
        loss_output = compute_coupled_loss(evaluator_output, _loss_config(cfg))
        structural_aux = _structural_auxiliary_loss(scorer_output, cfg, step_features["edge_is_cross_region"])
        primary_loss = loss_output["total_loss"]
        total_loss = primary_loss + structural_aux  # RAW loss, reported unchanged
        # P1: backward on the scale-invariant effective loss when enabled. Falls
        # back to the raw total loss when scale_invariant_backward is off.
        backward_primary = loss_output.get("effective_backward_loss", primary_loss)
        backward_loss = backward_primary + structural_aux
        loss_is_finite = bool(torch.isfinite(total_loss.detach()).cpu()) and bool(
            torch.isfinite(backward_loss.detach()).cpu()
        )
        loss_finite_all = loss_finite_all and loss_is_finite
        backward_loss.backward()

        pre_clip = _grad_report(model)
        edge_score_grad_max = 0.0
        if isinstance(edge_score, torch.Tensor) and edge_score.grad is not None:
            edge_score_grad_max = float(torch.max(torch.abs(edge_score.grad.detach())).cpu())
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
        post_clip = _grad_report(model)
        grad_info = post_clip
        gradients_finite_all = gradients_finite_all and bool(grad_info["gradients_finite"])
        optimizer.step()
        optimizer_steps_completed += 1
        record: dict[str, Any] = {
            "step": step,
            "total_loss": _tensor_float(total_loss),
            "L_primary": _tensor_float(primary_loss),
            "L_R": _tensor_float(loss_output["L_R"]),
            "L_D": _tensor_float(loss_output["L_D"]),
            "L_E": _tensor_float(loss_output["L_E"]),
            "L_aux_structural": _tensor_float(structural_aux),
            "effective_backward_loss": _tensor_float(backward_loss),
            "scale_backward_multiplier": _tensor_float(loss_output.get("scale_backward_multiplier", 1.0)),
            "F_mean": _tensor_float(loss_output["F_mean"]),
            "F_tail": _tensor_float(loss_output["F_tail"]),
            "C_avalanche_node_mean": _tensor_float(evaluator_output["C_avalanche_node_mean"]),
            "D_avalanche_rounds_mean": _tensor_float(evaluator_output["D_avalanche_rounds_mean"]),
            "E_consensus_node_mean": _tensor_float(evaluator_output["E_consensus_node_mean"]),
            "reliability_loss_should_weaken": bool(loss_output["reliability_loss_should_weaken"]),
            "gradient_norm_total": float(grad_info["gradient_norm_total"]),
            "gradient_norm_model": float(grad_info["gradient_norm_model"]),
            "pre_clip_grad_norm": float(pre_clip["gradient_norm_total"]),
            "post_clip_grad_norm": float(post_clip["gradient_norm_total"]),
            "gradient_clip_norm": 0.0 if gradient_clip_norm is None else float(gradient_clip_norm),
            "edge_score_grad_or_parameter_grad_max": max(edge_score_grad_max, float(grad_info["parameter_grad_max"])),
            "gradient_coverage_fraction": _tensor_float(topology.diagnostics["gradient_coverage_fraction"]),
            "active_edge_count": int(_tensor_float(topology.diagnostics["active_edge_count"])),
            "selected_fraction": _tensor_float(topology.diagnostics["selected_fraction"]),
            "support_mode": cfg["support_mode"],
            "topk_backend": cfg["topk_backend"],
            "budget_grad_max": float(grad_info["budget_grad_max"]),
            "sector_grad_max": float(grad_info["sector_grad_max"]),
            "role_grad_max": float(grad_info["role_grad_max"]),
            "bridge_grad_max": float(grad_info["bridge_grad_max"]),
            "structural_grad_max": float(grad_info["structural_grad_max"]),
            "edge_path_grad_max": float(grad_info["edge_path_grad_max"]),
            "any_nonfinite": not (loss_is_finite and bool(grad_info["gradients_finite"])),
        }
        if not bool(cfg["report_pre_clip_grad_norm"]):
            record.pop("pre_clip_grad_norm")
        if not bool(cfg["report_post_clip_grad_norm"]):
            record.pop("post_clip_grad_norm")
        if cap_diagnostics is not None:
            record.update(
                {
                    "cap_min": _tensor_float(cap_diagnostics["cap_min"]),
                    "cap_mean": _tensor_float(cap_diagnostics["cap_mean"]),
                    "cap_max": _tensor_float(cap_diagnostics["cap_max"]),
                    "cap_histogram": [float(v) for v in cap_diagnostics["cap_histogram"].detach().cpu().tolist()],
                }
            )
        step_records.append(record)

    final_metric_snapshot = _evaluate_model_snapshot(
        model=model,
        topology_layer=topology_layer,
        candidate=candidate,
        features=features,
        config=cfg,
        fixed_caps=fixed_caps,
        initial_correct=initial_correct,
        initial_wrong=initial_wrong,
    )
    parameter_change_max, parameter_change_l2 = _parameter_change(model, initial_parameters)
    total_losses = [float(item["total_loss"]) for item in step_records]
    final_record = step_records[-1]
    report: dict[str, Any] = {
        "config": cfg,
        "training_profile": cfg["training_profile"],
        "avalanche_profile": cfg["avalanche_profile"],
        "node_count": int(candidate.num_nodes),
        "candidate_edge_count": int(candidate.edge_count),
        "steps": step_records,
        "initial_metric_snapshot": initial_metric_snapshot,
        "final_metric_snapshot": final_metric_snapshot,
        "initial_total_loss": total_losses[0],
        "final_total_loss": total_losses[-1],
        "min_total_loss": min(total_losses),
        "loss_finite_all_steps": bool(loss_finite_all),
        "gradients_finite_all_steps": bool(gradients_finite_all),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "parameter_change_max": float(parameter_change_max),
        "parameter_change_l2": float(parameter_change_l2),
        "parameters_changed": bool(parameter_change_max > 0.0),
        "contract_ok": bool(loss_finite_all and gradients_finite_all and optimizer_steps_completed == int(cfg["max_steps"])),
        "support_mode": cfg["support_mode"],
        "topk_backend": cfg["topk_backend"],
        "gradient_coverage_fraction_final": float(final_record["gradient_coverage_fraction"]),
        "budget_grad_max_final": float(final_record["budget_grad_max"]),
        "sector_grad_max_final": float(final_record["sector_grad_max"]),
        "role_grad_max_final": float(final_record["role_grad_max"]),
        "bridge_grad_max_final": float(final_record["bridge_grad_max"]),
        "structural_grad_max_final": float(final_record["structural_grad_max"]),
        "edge_path_grad_max_final": float(final_record["edge_path_grad_max"]),
        "gradient_clip_applied": bool(gradient_clip_norm is not None),
    }
    report.update(_trend_report(step_records))
    return report, model


def run_tiny_training_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    report, _model = _run_training_phase(config)
    return report


def run_curriculum_training_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    cfg = _normalized_config(config)
    phase_steps = max(1, int(cfg["curriculum_phase_steps"]))
    constructor = _section(config, "constructor")
    phase0_budget_strategy = str(config.get("phase_0_budget_strategy", "fixed"))
    phase0_train_budget_head = bool(config.get("phase_0_train_budget_head", False))
    phase1_budget_strategy = str(config.get("phase_1_budget_strategy", "fixed"))
    phase1_train_budget_head = bool(
        config.get("phase_1_train_budget_head", phase1_budget_strategy == "auxiliary")
    )
    phase1_cap_source = constructor.get("phase_1_max_out_degree", config.get("max_out_degree", 4))
    phase1_max_out_degree = int(phase1_cap_source or 4)

    phase0_config = dict(config)
    phase0_config.update(
        {
            "support_mode": "all",
            "max_out_degree": None,
            "max_steps": phase_steps,
            "budget_strategy": phase0_budget_strategy,
            "train_budget_head": phase0_train_budget_head,
        }
    )
    phase0_report, model = _run_training_phase(phase0_config)
    phase1_config = dict(config)
    phase1_config.update(
        {
            "support_mode": "topk",
            "max_out_degree": phase1_max_out_degree,
            "max_steps": phase_steps,
            "budget_strategy": phase1_budget_strategy,
            "train_budget_head": phase1_train_budget_head,
        }
    )
    phase1_report, _model = _run_training_phase(phase1_config, model=model)
    phase0_final = phase0_report["steps"][-1]
    phase1_initial = phase1_report["steps"][0]
    phase1_final = phase1_report["steps"][-1]
    phase0_aux_loss_final = float(phase0_final["L_aux_structural"])
    phase1_aux_loss_final = float(phase1_final["L_aux_structural"])
    phase0_budget_grad = float(phase0_report["budget_grad_max_final"])
    phase1_budget_grad = float(phase1_report["budget_grad_max_final"])
    return {
        "phase_0": phase0_report,
        "phase_1": phase1_report,
        "carried_model_parameters": True,
        "phase_0_budget_strategy": phase0_report["config"]["budget_strategy"],
        "phase_1_budget_strategy": phase1_report["config"]["budget_strategy"],
        "phase_0_train_budget_head": bool(phase0_report["config"]["train_budget_head"]),
        "phase_1_train_budget_head": bool(phase1_report["config"]["train_budget_head"]),
        "phase_0_aux_loss_final": phase0_aux_loss_final,
        "phase_1_aux_loss_final": phase1_aux_loss_final,
        "phase_0_budget_grad_max_final": phase0_budget_grad,
        "phase_1_budget_grad_max_final": phase1_budget_grad,
        "phase_1_budget_auxiliary_active": bool(
            phase1_report["config"]["budget_strategy"] == "auxiliary"
            and bool(phase1_report["config"]["train_budget_head"])
            and phase1_aux_loss_final > 0.0
            and phase1_budget_grad > 0.0
        ),
        "phase_0_final_loss": phase0_report["final_total_loss"],
        "phase_1_initial_loss": phase1_report["initial_total_loss"],
        "phase_1_final_loss": phase1_report["final_total_loss"],
        "support_switch_active_edge_count_change": int(phase1_initial["active_edge_count"]) - int(phase0_final["active_edge_count"]),
        "gradient_coverage_fraction_change": float(phase1_final["gradient_coverage_fraction"])
        - float(phase0_final["gradient_coverage_fraction"]),
        "finite_gradients_both_phases": bool(
            phase0_report["gradients_finite_all_steps"] and phase1_report["gradients_finite_all_steps"]
        ),
        "parameters_changed_both_phases": bool(phase0_report["parameters_changed"] and phase1_report["parameters_changed"]),
        "contract_ok": bool(phase0_report["contract_ok"] and phase1_report["contract_ok"]),
    }
