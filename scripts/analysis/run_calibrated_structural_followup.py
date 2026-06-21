from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.training_smoke import (  # noqa: E402
    _run_training_phase,
    load_training_smoke_config,
    run_tiny_training_smoke,
)


SUCCESS_STATUS = "success"
FAILED_STATUS = "failed"
DEFAULT_TOLERANCE = 1.0e-4
LOW_STRUCTURAL_WEIGHTS = {"sector": 0.01, "role": 0.01, "bridge": 0.01}
ZERO_STRUCTURAL_WEIGHTS = {"sector": 0.0, "role": 0.0, "bridge": 0.0}
HIGH_UNSAFE_WEIGHTS = {"sector": 0.1, "role": 0.1, "bridge": 0.1}

DEFAULT_FOLLOWUP: dict[str, Any] = {
    "policy": {
        "all_mode_weights": {
            "sector_bias_weight": 0.01,
            "role_bias_weight": 0.01,
            "bridge_bias_weight": 0.01,
        },
        "topk_weights": {
            "sector_bias_weight": 0.0,
            "role_bias_weight": 0.0,
            "bridge_bias_weight": 0.0,
        },
        "unsafe_reference_weights": {
            "sector_bias_weight": 0.1,
            "role_bias_weight": 0.1,
            "bridge_bias_weight": 0.1,
        },
        "tolerance": DEFAULT_TOLERANCE,
    },
    "smoke": {
        "seeds": [7],
        "max_steps": 2,
        "node_count": 100,
        "avalanche_profile": "toy",
        "cases": [
            "all_zero_structural",
            "all_low_structural",
            "topk_zero_structural",
            "topk_high_structural_unsafe_reference",
            "curriculum_calibrated",
        ],
        "agent_check_safe": True,
    },
    "short": {
        "seeds": [7, 13],
        "max_steps": 3,
        "node_count": 100,
        "avalanche_profile": "toy",
        "cases": [
            "all_zero_structural",
            "all_low_structural",
            "topk_zero_structural",
            "topk_high_structural_unsafe_reference",
            "curriculum_calibrated",
        ],
        "agent_check_safe": False,
    },
}

CASE_ROLES = {
    "all_zero_structural": "all_mode_baseline",
    "all_low_structural": "safe_all_candidate",
    "topk_zero_structural": "safe_topk_default",
    "topk_high_structural_unsafe_reference": "unsafe_reference",
    "curriculum_calibrated": "curriculum_calibrated",
}
VALID_CASES = set(CASE_ROLES)
DEFAULT_OUTPUT_STEM = ".agent/tmp/calibrated_structural_followup"


def _parse_seeds(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def default_output_paths(tier: str) -> tuple[Path, Path]:
    if tier not in {"smoke", "short"}:
        raise ValueError("tier must be 'smoke' or 'short'")
    stem = f"{DEFAULT_OUTPUT_STEM}_{tier}"
    return ROOT / f"{stem}.json", ROOT / f"{stem}.md"


def _small_config(config: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(config)
    updated["hidden_dim"] = min(int(updated.get("hidden_dim", 16)), 16)
    updated["message_layers"] = min(int(updated.get("message_layers", 1)), 1)
    updated["init_mode"] = str(updated.get("init_mode", "deterministic"))
    return updated


def load_calibrated_followup_manifest(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError("controlled ablation manifest must be a mapping")
    section = data.get("calibrated_structural_followup", {})
    if section is None:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError("calibrated_structural_followup must be a mapping")
    merged = {
        "policy": dict(DEFAULT_FOLLOWUP["policy"]),
        "smoke": dict(DEFAULT_FOLLOWUP["smoke"]),
        "short": dict(DEFAULT_FOLLOWUP["short"]),
    }
    for key in ("policy", "smoke", "short"):
        raw = section.get(key, {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"calibrated_structural_followup.{key} must be a mapping")
        if key == "policy":
            policy = dict(merged["policy"])
            for policy_key, default_value in DEFAULT_FOLLOWUP["policy"].items():
                if isinstance(default_value, Mapping):
                    policy[policy_key] = dict(default_value)
            for raw_key, raw_value in raw.items():
                if isinstance(raw_value, Mapping) and isinstance(policy.get(raw_key), Mapping):
                    nested = dict(policy[raw_key])
                    nested.update(dict(raw_value))
                    policy[raw_key] = nested
                else:
                    policy[raw_key] = raw_value
            merged[key] = policy
        else:
            plan = dict(merged[key])
            plan.update(dict(raw))
            plan["seeds"] = [int(seed) for seed in plan.get("seeds", [])]
            plan["max_steps"] = int(plan["max_steps"])
            plan["node_count"] = int(plan["node_count"])
            plan["avalanche_profile"] = str(plan.get("avalanche_profile", "toy"))
            plan["cases"] = [str(case) for case in plan.get("cases", [])]
            unknown = sorted(set(plan["cases"]) - VALID_CASES)
            if unknown:
                raise ValueError(f"unknown calibrated follow-up cases: {unknown}")
            plan["agent_check_safe"] = bool(plan.get("agent_check_safe", False))
            merged[key] = plan
    return merged


def _plan_for_tier(
    manifest: Mapping[str, Any],
    tier: str,
    *,
    seeds: Iterable[int] | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    if tier not in {"smoke", "short"}:
        raise ValueError("tier must be 'smoke' or 'short'")
    plan = dict(manifest[tier])
    if seeds is not None:
        plan["seeds"] = [int(seed) for seed in seeds]
    if max_steps_override is not None:
        plan["max_steps"] = int(max_steps_override)
    return plan


def _weights_from_policy(policy: Mapping[str, Any], key: str) -> dict[str, float]:
    raw = policy.get(key, {})
    if not isinstance(raw, Mapping):
        raise ValueError(f"policy.{key} must be a mapping")
    return {
        "sector": float(raw.get("sector_bias_weight", 0.0)),
        "role": float(raw.get("role_bias_weight", 0.0)),
        "bridge": float(raw.get("bridge_bias_weight", 0.0)),
    }


def _weights_json(weights: Mapping[str, float]) -> dict[str, float]:
    return {
        "sector_bias_weight": float(weights["sector"]),
        "role_bias_weight": float(weights["role"]),
        "bridge_bias_weight": float(weights["bridge"]),
    }


def _base_case_config(
    base_config: Mapping[str, Any],
    *,
    seed: int,
    node_count: int,
    max_steps: int,
    avalanche_profile: str,
    support_mode: str,
    max_out_degree: int | None,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    config = dict(base_config)
    config.update(
        {
            "seed": int(seed),
            "vehicle_count": int(node_count),
            "max_steps": int(max_steps),
            "training_profile": "toy",
            "avalanche_profile": str(avalanche_profile),
            "support_mode": support_mode,
            "max_out_degree": None if max_out_degree is None else int(max_out_degree),
            "init_mode": "deterministic",
            "use_structural_score_bias": True,
            "sector_bias_weight": float(weights["sector"]),
            "role_bias_weight": float(weights["role"]),
            "bridge_bias_weight": float(weights["bridge"]),
            "budget_strategy": "auxiliary",
            "train_budget_head": True,
        }
    )
    config.setdefault("budget_target_weight", 0.05)
    config.setdefault("sector_entropy_weight", 0.01)
    config.setdefault("role_balance_weight", 0.01)
    config.setdefault("bridge_regularizer_weight", 0.01)
    return config


def case_config(
    base_config: Mapping[str, Any],
    *,
    case_name: str,
    seed: int,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    node_count = int(plan["node_count"])
    max_steps = int(plan["max_steps"])
    avalanche_profile = str(plan["avalanche_profile"])
    all_weights = _weights_from_policy(policy, "all_mode_weights")
    topk_weights = _weights_from_policy(policy, "topk_weights")
    unsafe_weights = _weights_from_policy(policy, "unsafe_reference_weights")
    if case_name == "all_zero_structural":
        return _base_case_config(
            base_config,
            seed=seed,
            node_count=node_count,
            max_steps=max_steps,
            avalanche_profile=avalanche_profile,
            support_mode="all",
            max_out_degree=None,
            weights=ZERO_STRUCTURAL_WEIGHTS,
        )
    if case_name == "all_low_structural":
        return _base_case_config(
            base_config,
            seed=seed,
            node_count=node_count,
            max_steps=max_steps,
            avalanche_profile=avalanche_profile,
            support_mode="all",
            max_out_degree=None,
            weights=all_weights,
        )
    if case_name == "topk_zero_structural":
        return _base_case_config(
            base_config,
            seed=seed,
            node_count=node_count,
            max_steps=max_steps,
            avalanche_profile=avalanche_profile,
            support_mode="topk",
            max_out_degree=2,
            weights=topk_weights,
        )
    if case_name == "topk_high_structural_unsafe_reference":
        return _base_case_config(
            base_config,
            seed=seed,
            node_count=node_count,
            max_steps=max_steps,
            avalanche_profile=avalanche_profile,
            support_mode="topk",
            max_out_degree=2,
            weights=unsafe_weights,
        )
    raise ValueError(f"case_config does not apply to curriculum case: {case_name}")


def _run_summary_from_training_report(
    *,
    case_name: str,
    seed: int,
    case_role: str,
    runtime_s: float,
    report: Mapping[str, Any],
    policy_status: str,
    unsafe_reference: bool = False,
) -> dict[str, Any]:
    first = report["steps"][0]
    final = report["steps"][-1]
    cfg = report["config"]
    return {
        "case_name": case_name,
        "case_role": case_role,
        "seed": int(seed),
        "case_status": SUCCESS_STATUS,
        "case_error": None,
        "runtime_s": float(runtime_s),
        "support_mode": str(report["support_mode"]),
        "avalanche_profile": str(report["avalanche_profile"]),
        "max_steps": int(report["optimizer_steps_completed"]),
        "unsafe_reference": bool(unsafe_reference),
        "structural_policy_status": policy_status,
        "structural_weights": {
            "sector_bias_weight": float(cfg["sector_bias_weight"]),
            "role_bias_weight": float(cfg["role_bias_weight"]),
            "bridge_bias_weight": float(cfg["bridge_bias_weight"]),
        },
        "initial_total_loss": float(report["initial_total_loss"]),
        "final_total_loss": float(report["final_total_loss"]),
        "initial_primary_loss": float(first["L_primary"]),
        "final_primary_loss": float(final["L_primary"]),
        "L_aux_structural_final": float(final["L_aux_structural"]),
        "C_initial": float(first["C_avalanche_node_mean"]),
        "C_final": float(final["C_avalanche_node_mean"]),
        "F_initial": float(first["F_mean"]),
        "F_final": float(final["F_mean"]),
        "D_initial": float(first["D_avalanche_rounds_mean"]),
        "D_final": float(final["D_avalanche_rounds_mean"]),
        "E_initial": float(first["E_consensus_node_mean"]),
        "E_final": float(final["E_consensus_node_mean"]),
        "active_edge_count_initial": int(first["active_edge_count"]),
        "active_edge_count_final": int(final["active_edge_count"]),
        "gradient_coverage_initial": float(first["gradient_coverage_fraction"]),
        "gradient_coverage_final": float(final["gradient_coverage_fraction"]),
        "losses_finite": bool(report["loss_finite_all_steps"]),
        "gradients_finite": bool(report["gradients_finite_all_steps"]),
        "parameter_change_max": float(report["parameter_change_max"]),
        "edge_path_grad_max_final": float(report["edge_path_grad_max_final"]),
        "structural_grad_max_final": float(report["structural_grad_max_final"]),
        "budget_grad_max_final": float(report["budget_grad_max_final"]),
    }


def _failed_summary(
    *,
    case_name: str,
    seed: int,
    case_role: str,
    runtime_s: float,
    error: BaseException,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "case_role": case_role,
        "seed": int(seed),
        "case_status": FAILED_STATUS,
        "case_error": f"{type(error).__name__}: {error}",
        "runtime_s": float(runtime_s),
        "support_mode": str(config.get("support_mode", "")) if config else "",
        "avalanche_profile": str(config.get("avalanche_profile", "")) if config else "",
        "max_steps": int(config.get("max_steps", 0)) if config else 0,
        "losses_finite": False,
        "gradients_finite": False,
        "parameter_change_max": 0.0,
    }


def _run_single_case(
    *,
    base_config: Mapping[str, Any],
    case_name: str,
    seed: int,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    cfg = case_config(base_config, case_name=case_name, seed=seed, plan=plan, policy=policy)
    start = perf_counter()
    try:
        raw = run_tiny_training_smoke(cfg)
        return _run_summary_from_training_report(
            case_name=case_name,
            seed=seed,
            case_role=CASE_ROLES[case_name],
            runtime_s=perf_counter() - start,
            report=raw,
            policy_status=str(policy.get("policy_status", "disable_structural_bias_in_topk")),
            unsafe_reference=case_name == "topk_high_structural_unsafe_reference",
        )
    except Exception as exc:  # noqa: BLE001 - per-case errors are report data.
        return _failed_summary(
            case_name=case_name,
            seed=seed,
            case_role=CASE_ROLES[case_name],
            runtime_s=perf_counter() - start,
            error=exc,
            config=cfg,
        )


def _run_curriculum_pair(
    *,
    base_config: Mapping[str, Any],
    seed: int,
    plan: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    node_count = int(plan["node_count"])
    max_steps = int(plan["max_steps"])
    avalanche_profile = str(plan["avalanche_profile"])
    phase_steps = max(1, max_steps)
    topk_weights = _weights_from_policy(policy, "topk_weights")
    all_weights = _weights_from_policy(policy, "all_mode_weights")

    def phase_config(weights: Mapping[str, float], *, support_mode: str, max_out_degree: int | None) -> dict[str, Any]:
        return _base_case_config(
            base_config,
            seed=seed,
            node_count=node_count,
            max_steps=phase_steps,
            avalanche_profile=avalanche_profile,
            support_mode=support_mode,
            max_out_degree=max_out_degree,
            weights=weights,
        )

    start = perf_counter()
    try:
        baseline_phase0, baseline_model = _run_training_phase(
            phase_config(ZERO_STRUCTURAL_WEIGHTS, support_mode="all", max_out_degree=None)
        )
        baseline_phase1, _baseline_model = _run_training_phase(
            phase_config(topk_weights, support_mode="topk", max_out_degree=2),
            model=baseline_model,
        )
        baseline = _curriculum_summary(
            case_name="curriculum_zero_baseline",
            seed=seed,
            runtime_s=perf_counter() - start,
            phase0=baseline_phase0,
            phase1=baseline_phase1,
            policy=policy,
            case_role="curriculum_baseline",
        )
    except Exception as exc:  # noqa: BLE001
        baseline = _failed_summary(
            case_name="curriculum_zero_baseline",
            seed=seed,
            case_role="curriculum_baseline",
            runtime_s=perf_counter() - start,
            error=exc,
        )

    start = perf_counter()
    try:
        calibrated_phase0, calibrated_model = _run_training_phase(
            phase_config(all_weights, support_mode="all", max_out_degree=None)
        )
        calibrated_phase1, _calibrated_model = _run_training_phase(
            phase_config(topk_weights, support_mode="topk", max_out_degree=2),
            model=calibrated_model,
        )
        calibrated = _curriculum_summary(
            case_name="curriculum_calibrated",
            seed=seed,
            runtime_s=perf_counter() - start,
            phase0=calibrated_phase0,
            phase1=calibrated_phase1,
            policy=policy,
            case_role=CASE_ROLES["curriculum_calibrated"],
        )
    except Exception as exc:  # noqa: BLE001
        calibrated = _failed_summary(
            case_name="curriculum_calibrated",
            seed=seed,
            case_role=CASE_ROLES["curriculum_calibrated"],
            runtime_s=perf_counter() - start,
            error=exc,
        )
    return baseline, calibrated


def _curriculum_summary(
    *,
    case_name: str,
    seed: int,
    runtime_s: float,
    phase0: Mapping[str, Any],
    phase1: Mapping[str, Any],
    policy: Mapping[str, Any],
    case_role: str,
) -> dict[str, Any]:
    phase0_final = phase0["steps"][-1]
    phase1_final = phase1["steps"][-1]
    return {
        "case_name": case_name,
        "case_role": case_role,
        "seed": int(seed),
        "case_status": SUCCESS_STATUS,
        "case_error": None,
        "runtime_s": float(runtime_s),
        "support_mode": "all_to_topk",
        "avalanche_profile": str(phase1["avalanche_profile"]),
        "max_steps": int(phase0["optimizer_steps_completed"]) + int(phase1["optimizer_steps_completed"]),
        "unsafe_reference": False,
        "structural_policy_status": str(policy.get("policy_status", "disable_structural_bias_in_topk")),
        "phase_0_structural_weights": {
            "sector_bias_weight": float(phase0["config"]["sector_bias_weight"]),
            "role_bias_weight": float(phase0["config"]["role_bias_weight"]),
            "bridge_bias_weight": float(phase0["config"]["bridge_bias_weight"]),
        },
        "phase_1_structural_weights": {
            "sector_bias_weight": float(phase1["config"]["sector_bias_weight"]),
            "role_bias_weight": float(phase1["config"]["role_bias_weight"]),
            "bridge_bias_weight": float(phase1["config"]["bridge_bias_weight"]),
        },
        "structural_weights": {
            "phase_0": {
                "sector_bias_weight": float(phase0["config"]["sector_bias_weight"]),
                "role_bias_weight": float(phase0["config"]["role_bias_weight"]),
                "bridge_bias_weight": float(phase0["config"]["bridge_bias_weight"]),
            },
            "phase_1": {
                "sector_bias_weight": float(phase1["config"]["sector_bias_weight"]),
                "role_bias_weight": float(phase1["config"]["role_bias_weight"]),
                "bridge_bias_weight": float(phase1["config"]["bridge_bias_weight"]),
            },
        },
        "initial_total_loss": float(phase0["initial_total_loss"]),
        "final_total_loss": float(phase1["final_total_loss"]),
        "initial_primary_loss": float(phase0["steps"][0]["L_primary"]),
        "final_primary_loss": float(phase1_final["L_primary"]),
        "L_aux_structural_final": float(phase1_final["L_aux_structural"]),
        "C_initial": float(phase0["steps"][0]["C_avalanche_node_mean"]),
        "C_final": float(phase1_final["C_avalanche_node_mean"]),
        "F_initial": float(phase0["steps"][0]["F_mean"]),
        "F_final": float(phase1_final["F_mean"]),
        "D_initial": float(phase0["steps"][0]["D_avalanche_rounds_mean"]),
        "D_final": float(phase1_final["D_avalanche_rounds_mean"]),
        "E_initial": float(phase0["steps"][0]["E_consensus_node_mean"]),
        "E_final": float(phase1_final["E_consensus_node_mean"]),
        "active_edge_count_initial": int(phase0["steps"][0]["active_edge_count"]),
        "active_edge_count_final": int(phase1_final["active_edge_count"]),
        "phase_0_active_edge_count_final": int(phase0_final["active_edge_count"]),
        "phase_1_active_edge_count_final": int(phase1_final["active_edge_count"]),
        "gradient_coverage_initial": float(phase0["steps"][0]["gradient_coverage_fraction"]),
        "gradient_coverage_final": float(phase1_final["gradient_coverage_fraction"]),
        "phase_0_gradient_coverage_final": float(phase0_final["gradient_coverage_fraction"]),
        "phase_1_gradient_coverage_final": float(phase1_final["gradient_coverage_fraction"]),
        "losses_finite": bool(phase0["loss_finite_all_steps"] and phase1["loss_finite_all_steps"]),
        "gradients_finite": bool(phase0["gradients_finite_all_steps"] and phase1["gradients_finite_all_steps"]),
        "parameter_change_max": max(float(phase0["parameter_change_max"]), float(phase1["parameter_change_max"])),
        "edge_path_grad_max_final": float(phase1["edge_path_grad_max_final"]),
        "structural_grad_max_final": float(phase1["structural_grad_max_final"]),
        "budget_grad_max_final": float(phase1["budget_grad_max_final"]),
    }


def _delta(ablation: Mapping[str, Any], baseline: Mapping[str, Any], key: str) -> float | None:
    if ablation.get("case_status") != SUCCESS_STATUS or baseline.get("case_status") != SUCCESS_STATUS:
        return None
    return float(ablation[key]) - float(baseline[key])


def _non_worse(comparison: Mapping[str, Any], *, tolerance: float) -> bool:
    return (
        comparison.get("primary_loss_delta") is not None
        and float(comparison["primary_loss_delta"]) <= tolerance
        and float(comparison["F_delta"]) <= tolerance
        and float(comparison["C_delta"]) >= -tolerance
        and float(comparison["D_delta"]) <= tolerance
        and float(comparison["E_delta"]) <= tolerance
    )


def _support_changed(comparison: Mapping[str, Any]) -> bool:
    active_delta = comparison.get("active_edge_count_delta")
    coverage_delta = comparison.get("gradient_coverage_delta")
    return (active_delta not in (None, 0)) or (
        coverage_delta is not None and abs(float(coverage_delta)) > DEFAULT_TOLERANCE
    )


def _comparison(
    *,
    name: str,
    baseline: Mapping[str, Any],
    ablation: Mapping[str, Any],
    tolerance: float,
    category_hint: str,
) -> dict[str, Any]:
    comparison = {
        "comparison_name": name,
        "seed": int(ablation.get("seed", baseline.get("seed", 0))),
        "baseline_case": baseline.get("case_name", "unknown"),
        "ablation_case": ablation.get("case_name", "unknown"),
        "category_hint": category_hint,
        "primary_loss_delta": _delta(ablation, baseline, "final_primary_loss"),
        "total_loss_delta": _delta(ablation, baseline, "final_total_loss"),
        "F_delta": _delta(ablation, baseline, "F_final"),
        "C_delta": _delta(ablation, baseline, "C_final"),
        "D_delta": _delta(ablation, baseline, "D_final"),
        "E_delta": _delta(ablation, baseline, "E_final"),
        "active_edge_count_delta": _delta(ablation, baseline, "active_edge_count_final"),
        "gradient_coverage_delta": _delta(ablation, baseline, "gradient_coverage_final"),
        "baseline_status": baseline.get("case_status", FAILED_STATUS),
        "ablation_status": ablation.get("case_status", FAILED_STATUS),
    }
    if baseline.get("case_status") != SUCCESS_STATUS or ablation.get("case_status") != SUCCESS_STATUS:
        interpretation = "failed_case"
        claim_allowed = False
        recommendation = "blocked_due_to_failure"
        note = "At least one compared case failed; no claim is allowed."
    elif category_hint == "unsafe_reference":
        changed = _support_changed(comparison)
        worsened = not _non_worse(comparison, tolerance=tolerance)
        interpretation = "unsafe_reference" if (changed or worsened) else "unsafe_reference_not_harmful_in_smoke"
        claim_allowed = False
        recommendation = "do_not_promote"
        note = "High-weight top-k structural bias is an unsafe reference and is never promoted."
    elif category_hint == "safe_topk_default":
        interpretation = "safe_topk_default"
        claim_allowed = True
        recommendation = "default_deployable_topk_policy"
        note = "Top-k structural weights remain zero by calibrated policy."
    elif category_hint == "safe_all_candidate":
        non_worse = _non_worse(comparison, tolerance=tolerance)
        interpretation = "safe_all_candidate" if non_worse else "not_safe_all_candidate"
        claim_allowed = bool(non_worse)
        recommendation = "candidate_for_deeper_study" if non_worse else "needs_longer_run_or_recalibration"
        note = (
            "Low all-mode structural weights are non-worse versus all-zero in this seed."
            if non_worse
            else "Low all-mode structural weights worsened at least one primary C/F/D/E metric."
        )
    elif category_hint == "curriculum_calibrated":
        non_worse = _non_worse(comparison, tolerance=tolerance)
        interpretation = "curriculum_calibrated" if non_worse else "curriculum_not_recommended"
        claim_allowed = bool(non_worse)
        recommendation = "candidate_for_deeper_study" if non_worse else "keep_as_diagnostic_only"
        note = (
            "Calibrated curriculum did not worsen primary metrics relative to zero-structural curriculum."
            if non_worse
            else "Calibrated curriculum worsened at least one primary metric relative to zero-structural curriculum."
        )
    else:
        interpretation = "inconclusive"
        claim_allowed = False
        recommendation = "needs_review"
        note = "No calibrated interpretation rule matched this comparison."
    comparison.update(
        {
            "interpretation_category": interpretation,
            "primary_benefit_claim_allowed": bool(claim_allowed),
            "claim_gate_passed": bool(claim_allowed),
            "recommendation": recommendation,
            "support_changed_confounder": _support_changed(comparison),
            "interpretation_note": note,
        }
    )
    return comparison


def _mean(values: list[float]) -> float | None:
    return sum(values) / float(len(values)) if values else None


def _std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    mean = _mean(values)
    assert mean is not None
    return (sum((value - mean) ** 2 for value in values) / float(len(values))) ** 0.5


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": _mean(values),
        "std": _std(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def _sign(value: float, *, tolerance: float = DEFAULT_TOLERANCE) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _sign_consistency_fraction(values: list[float]) -> float | None:
    if not values:
        return None
    signs = [_sign(value) for value in values]
    return max(signs.count(-1), signs.count(0), signs.count(1)) / float(len(signs))


def _aggregate_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in sorted({str(item["comparison_name"]) for item in comparisons}):
        items = [item for item in comparisons if str(item["comparison_name"]) == name]
        successful_items = [
            item
            for item in items
            if item.get("baseline_status") == SUCCESS_STATUS and item.get("ablation_status") == SUCCESS_STATUS
        ]
        failed_items = [item for item in items if item not in successful_items]
        metric_values: dict[str, list[float]] = {}
        for metric in (
            "primary_loss_delta",
            "F_delta",
            "C_delta",
            "D_delta",
            "E_delta",
            "total_loss_delta",
        ):
            metric_values[metric] = [
                float(item[metric]) for item in successful_items if item.get(metric) is not None
            ]
        recommendations = {str(item.get("recommendation", "")) for item in items}
        categories = {str(item["interpretation_category"]) for item in items}
        claim_gate_pass_count = sum(1 for item in items if bool(item["claim_gate_passed"]))
        seed_count = len(items)
        summary: dict[str, Any] = {
            "seed_count": seed_count,
            "successful_seed_count": len(successful_items),
            "failed_seed_count": len(failed_items),
            "claim_gate_pass_count": claim_gate_pass_count,
            "all_claim_gates_passed": bool(items) and all(bool(item["claim_gate_passed"]) for item in items),
            "claim_gate_pass_fraction": (claim_gate_pass_count / float(seed_count)) if seed_count else 0.0,
            "recommendation_consistency": (
                "consistent" if seed_count > 0 and len(recommendations) <= 1 and len(categories) <= 1 else "mixed"
            ),
            "interpretation_categories": sorted(categories),
            "recommendations": sorted(recommendations),
        }
        for metric, values in metric_values.items():
            stats = _stats(values)
            summary[f"{metric}_mean"] = stats["mean"]
            summary[f"{metric}_std"] = stats["std"]
            summary[f"{metric}_min"] = stats["min"]
            summary[f"{metric}_max"] = stats["max"]
        for metric in ("primary_loss_delta", "F_delta", "C_delta"):
            summary[f"{metric}_sign_consistency_fraction"] = _sign_consistency_fraction(metric_values[metric])
        output[name] = {
            **summary,
        }
    return output


def _comparison_all_claims_passed(aggregate: Mapping[str, Any], name: str) -> bool:
    item = aggregate.get(name, {})
    return bool(item.get("successful_seed_count", 0)) and bool(item.get("all_claim_gates_passed", False))


def _high_weight_topk_stably_unsafe(aggregate: Mapping[str, Any]) -> bool:
    high = aggregate.get("topk_high_vs_topk_zero", {})
    if int(high.get("successful_seed_count", 0)) <= 0:
        return False
    categories = set(high.get("interpretation_categories", []))
    return (
        int(high.get("claim_gate_pass_count", 0)) == 0
        and bool(categories)
        and categories <= {"unsafe_reference"}
    )


def _readiness_summary(
    *,
    tier: str,
    failed_case_count: int,
    failed_seed_count: int,
    successful_seed_count: int,
    aggregate: Mapping[str, Any],
    all_low_ok: bool,
    curriculum_ok: bool,
    high_unsafe: bool,
) -> tuple[str, list[str], str, bool, list[str], str]:
    reasons: list[str] = []
    blocking: list[str] = []
    if failed_case_count:
        reasons.append("failed_seed_or_case_present")
        blocking.append("failed_seed_or_case_present")
    if not all_low_ok:
        reasons.append("all_low_structural_not_stable_nonworse")
        blocking.append("all_low_structural_not_stable_nonworse")
    if not curriculum_ok:
        reasons.append("calibrated_curriculum_not_stable_nonworse")
        blocking.append("calibrated_curriculum_not_stable_nonworse")
    if not high_unsafe:
        reasons.append("high_weight_topk_requires_reaudit")
        blocking.append("high_weight_topk_requires_reaudit")
    high = aggregate.get("topk_high_vs_topk_zero", {})
    if int(high.get("successful_seed_count", 0)) <= 0:
        reasons.append("no_successful_high_weight_topk_reference")
        blocking.append("no_successful_high_weight_topk_reference")
    if tier == "smoke":
        blocking.append("requires_short_tier_evidence")
    if successful_seed_count < 2:
        blocking.append("insufficient_successful_seed_count")
    if failed_seed_count:
        blocking.append("failed_seed_present")

    if not high_unsafe:
        return "blocked", reasons, "rerun_structural_weight_audit", False, blocking, "blocked_until_reaudit"
    if failed_case_count:
        return (
            "conditionally_ready",
            reasons,
            "inspect_failed_short_seed_then_rerun",
            False,
            blocking,
            "inspect_failed_short_seed_then_rerun",
        )
    selected_allowed = (
        tier != "smoke"
        and all_low_ok
        and curriculum_ok
        and high_unsafe
        and failed_seed_count == 0
        and successful_seed_count >= 2
    )
    if selected_allowed:
        return (
            "ready_for_selected_training_smoke",
            ["short_tier_calibrated_policy_nonworse"],
            "M16_selected_training_smoke_with_calibrated_policy",
            True,
            [],
            "M16_selected_training_smoke_with_calibrated_policy",
        )
    if all_low_ok and curriculum_ok:
        return (
            "conditionally_ready",
            reasons or ["short_tier_evidence_required_for_M16"],
            "run_calibrated_structural_short_tier",
            False,
            blocking,
            "run_calibrated_structural_short_tier",
        )
    return (
        "conditionally_ready",
        reasons,
        "extend_short_or_recalibrate",
        False,
        blocking,
        "extend_short_or_recalibrate",
    )


def _summary(
    *,
    tier: str,
    plan: Mapping[str, Any],
    runs: list[dict[str, Any]],
    baseline_runs: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    successful = [run for run in runs if run.get("case_status") == SUCCESS_STATUS]
    failed = [run for run in runs if run.get("case_status") != SUCCESS_STATUS]
    all_report_runs = runs + baseline_runs
    failed_seeds = {
        int(run["seed"])
        for run in all_report_runs
        if run.get("case_status") != SUCCESS_STATUS
    }
    planned_seeds = {int(seed) for seed in plan["seeds"]}
    successful_seed_count = len(planned_seeds - failed_seeds)
    failed_seed_count = len(failed_seeds)
    aggregate = _aggregate_comparisons(comparisons)
    all_low_ok = _comparison_all_claims_passed(aggregate, "all_low_vs_all_zero")
    curriculum_ok = _comparison_all_claims_passed(aggregate, "curriculum_calibrated_vs_zero_baseline")
    high_unsafe = _high_weight_topk_stably_unsafe(aggregate)
    unsafe_not_promoted = all(
        item.get("comparison_name") != "topk_high_vs_topk_zero" or not bool(item.get("claim_gate_passed"))
        for item in comparisons
    )
    topk_default_ok = any(run["case_name"] == "topk_zero_structural" and run["case_status"] == SUCCESS_STATUS for run in runs)
    (
        readiness,
        readiness_reasons,
        next_stage,
        selected_allowed,
        selected_blocking,
        selected_next_stage,
    ) = _readiness_summary(
        tier=tier,
        failed_case_count=len(failed),
        failed_seed_count=failed_seed_count,
        successful_seed_count=successful_seed_count,
        aggregate=aggregate,
        all_low_ok=all_low_ok,
        curriculum_ok=curriculum_ok,
        high_unsafe=high_unsafe,
    )
    if failed:
        recommendation_status = "blocked"
    elif all_low_ok and topk_default_ok:
        recommendation_status = "calibrated_policy_supported"
    elif topk_default_ok:
        recommendation_status = "topk_zero_default_only"
    else:
        recommendation_status = "inconclusive"
    all_weights = _weights_from_policy(policy, "all_mode_weights") if all_low_ok else ZERO_STRUCTURAL_WEIGHTS
    return {
        "tier": tier,
        "evidence_tier": tier,
        "readiness_basis_tier": tier,
        "seeds": list(plan["seeds"]),
        "seed_count": len(plan["seeds"]),
        "readiness_seed_count": len(plan["seeds"]),
        "readiness_successful_seed_count": successful_seed_count,
        "node_count": int(plan["node_count"]),
        "max_steps": int(plan["max_steps"]),
        "avalanche_profile": str(plan["avalanche_profile"]),
        "cases_planned": list(plan["cases"]),
        "agent_check_safe": bool(plan["agent_check_safe"]),
        "case_count": len(runs),
        "baseline_case_count": len(baseline_runs),
        "successful_case_count": len(successful),
        "failed_case_count": len(failed),
        "failed_seed_count": failed_seed_count,
        "cases_by_status": {
            SUCCESS_STATUS: len(successful),
            FAILED_STATUS: len(failed),
        },
        "comparison_summary": aggregate,
        "recommendation_status": recommendation_status,
        "calibrated_policy_readiness": readiness,
        "calibrated_policy_reasons": readiness_reasons,
        "all_low_structural_stable_nonworse": bool(all_low_ok),
        "calibrated_curriculum_stable_nonworse": bool(curriculum_ok),
        "topk_high_reference_stably_unsafe": bool(high_unsafe),
        "topk_structural_bias_default": "disabled",
        "recommended_next_stage": next_stage,
        "selected_training_smoke_allowed": bool(selected_allowed),
        "selected_training_smoke_blocking_reasons": selected_blocking,
        "selected_training_smoke_recommended_next_stage": selected_next_stage,
        "recommended_followup_policy": {
            "all_mode_weights": _weights_json(all_weights),
            "topk_weights": _weights_json(_weights_from_policy(policy, "topk_weights")),
            "topk_policy": "disable_structural_bias_in_topk",
            "curriculum_calibrated_recommended": bool(curriculum_ok),
            "unsafe_reference_not_promoted": bool(unsafe_not_promoted),
        },
        "recommended_cases_for_followup": (
            ["all_low_structural"] if all_low_ok else []
        )
        + (["curriculum_calibrated"] if curriculum_ok else []),
        "unsafe_reference_cases": ["topk_high_structural_unsafe_reference"],
        "claim_boundary": [
            "all_low_structural is promoted only if primary C/F/D/E metrics are non-worse across seeds.",
            "topk_high_structural_unsafe_reference is never promoted.",
            "topk_zero_structural remains the default deployable top-k policy until a safe nonzero top-k setting is found.",
            "curriculum_calibrated requires non-worse primary metrics versus zero-structural curriculum.",
        ],
    }


def run_calibrated_structural_followup(
    base_config: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    tier: str = "smoke",
    seeds: Iterable[int] | None = None,
    max_steps_override: int | None = None,
) -> dict[str, Any]:
    plan = _plan_for_tier(manifest, tier, seeds=seeds, max_steps_override=max_steps_override)
    policy = dict(manifest["policy"])
    policy.setdefault("policy_status", "disable_structural_bias_in_topk")
    policy.setdefault("tolerance", DEFAULT_TOLERANCE)
    tolerance = float(policy.get("tolerance", DEFAULT_TOLERANCE))
    runs: list[dict[str, Any]] = []
    baseline_runs: list[dict[str, Any]] = []
    for seed in plan["seeds"]:
        for case_name in plan["cases"]:
            if case_name == "curriculum_calibrated":
                baseline, calibrated = _run_curriculum_pair(
                    base_config=base_config,
                    seed=int(seed),
                    plan=plan,
                    policy=policy,
                )
                baseline_runs.append(baseline)
                runs.append(calibrated)
            else:
                runs.append(
                    _run_single_case(
                        base_config=base_config,
                        case_name=case_name,
                        seed=int(seed),
                        plan=plan,
                        policy=policy,
                    )
                )
    by_seed_case = {(int(run["seed"]), str(run["case_name"])): run for run in runs}
    baseline_by_seed = {(int(run["seed"]), str(run["case_name"])): run for run in baseline_runs}
    comparisons: list[dict[str, Any]] = []
    for seed in plan["seeds"]:
        all_zero = by_seed_case.get((int(seed), "all_zero_structural"))
        all_low = by_seed_case.get((int(seed), "all_low_structural"))
        topk_zero = by_seed_case.get((int(seed), "topk_zero_structural"))
        topk_high = by_seed_case.get((int(seed), "topk_high_structural_unsafe_reference"))
        curriculum_zero = baseline_by_seed.get((int(seed), "curriculum_zero_baseline"))
        curriculum = by_seed_case.get((int(seed), "curriculum_calibrated"))
        if all_zero is not None and all_low is not None:
            comparisons.append(
                _comparison(
                    name="all_low_vs_all_zero",
                    baseline=all_zero,
                    ablation=all_low,
                    tolerance=tolerance,
                    category_hint="safe_all_candidate",
                )
            )
        if topk_zero is not None:
            comparisons.append(
                _comparison(
                    name="topk_zero_default",
                    baseline=topk_zero,
                    ablation=topk_zero,
                    tolerance=tolerance,
                    category_hint="safe_topk_default",
                )
            )
        if topk_zero is not None and topk_high is not None:
            comparisons.append(
                _comparison(
                    name="topk_high_vs_topk_zero",
                    baseline=topk_zero,
                    ablation=topk_high,
                    tolerance=tolerance,
                    category_hint="unsafe_reference",
                )
            )
        if curriculum_zero is not None and curriculum is not None:
            comparisons.append(
                _comparison(
                    name="curriculum_calibrated_vs_zero_baseline",
                    baseline=curriculum_zero,
                    ablation=curriculum,
                    tolerance=tolerance,
                    category_hint="curriculum_calibrated",
                )
            )
    summary = _summary(
        tier=tier,
        plan=plan,
        runs=runs,
        baseline_runs=baseline_runs,
        comparisons=comparisons,
        policy=policy,
    )
    return {
        "audit_scope": "M15 calibrated structural follow-up",
        "structural_weight_policy": {
            "all_mode_candidate_weights": _weights_json(_weights_from_policy(policy, "all_mode_weights")),
            "topk_weights": _weights_json(_weights_from_policy(policy, "topk_weights")),
            "unsafe_reference_weights": _weights_json(_weights_from_policy(policy, "unsafe_reference_weights")),
            "policy_status": str(policy.get("policy_status", "disable_structural_bias_in_topk")),
            "tolerance": tolerance,
        },
        **summary,
        "runs": runs,
        "baseline_runs": baseline_runs,
        "comparisons": comparisons,
        "contract_notes": [
            "M15 is a targeted selected evidence harness, not production training.",
            "The calibrated top-k policy keeps structural score-bias weights at zero.",
            "High-weight top-k structural bias is included only as an unsafe reference.",
            "No broad model-quality claim is made from these tiny deterministic runs.",
        ],
    }


def write_followup_json(report: Mapping[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.6e}"
    except (TypeError, ValueError):
        return str(value)


def write_followup_markdown(report: Mapping[str, Any], path: str | Path) -> None:
    policy = report["structural_weight_policy"]
    lines = [
        "# Calibrated Structural Follow-Up",
        "",
        "This M15 report tests the M14.2 calibrated structural policy in tiny deterministic selected runs.",
        "",
        "## Decision Summary",
        "",
        f"- Recommendation status: `{report['recommendation_status']}`",
        f"- Evidence tier: `{report.get('evidence_tier', report.get('tier', 'unknown'))}`",
        f"- Readiness basis tier: `{report.get('readiness_basis_tier', report.get('tier', 'unknown'))}`",
        f"- Calibrated policy readiness: `{report.get('calibrated_policy_readiness', 'unknown')}`",
        f"- Recommended next stage: `{report.get('recommended_next_stage', 'unknown')}`",
        f"- Selected training smoke allowed: `{report.get('selected_training_smoke_allowed', False)}`",
        f"- Selected training smoke blocking reasons: `{report.get('selected_training_smoke_blocking_reasons', [])}`",
        f"- Selected training smoke next stage: `{report.get('selected_training_smoke_recommended_next_stage', '')}`",
        f"- Recommended follow-up policy: `{report['recommended_followup_policy']}`",
        f"- Follow-up cases: `{', '.join(report['recommended_cases_for_followup']) or 'none'}`",
        f"- Unsafe references: `{', '.join(report['unsafe_reference_cases']) or 'none'}`",
        f"- Policy status: `{policy['policy_status']}`",
        f"- Agent-check safe: `{report['agent_check_safe']}`",
        f"- All-mode low structural stable non-worse: `{report.get('all_low_structural_stable_nonworse', False)}`",
        f"- Calibrated curriculum stable non-worse: `{report.get('calibrated_curriculum_stable_nonworse', False)}`",
        f"- High-weight top-k stably unsafe: `{report.get('topk_high_reference_stably_unsafe', False)}`",
        "",
        "## Structural Weight Policy",
        "",
        f"- all-mode candidate weights: `{policy['all_mode_candidate_weights']}`",
        f"- top-k weights: `{policy['topk_weights']}`",
        f"- unsafe reference weights: `{policy['unsafe_reference_weights']}`",
        "",
        "## Aggregate Comparison Summary",
        "",
        "| comparison | seeds | success | failed | claim pass fraction | recommendation consistency | primary mean/std/min/max | F mean/std/min/max | C mean/std/min/max | sign consistency primary/F/C |",
        "|---|---:|---:|---:|---:|---|---|---|---|---|",
    ]
    for name, item in sorted(report.get("comparison_summary", {}).items()):
        primary = (
            f"{_fmt(item.get('primary_loss_delta_mean'))}/"
            f"{_fmt(item.get('primary_loss_delta_std'))}/"
            f"{_fmt(item.get('primary_loss_delta_min'))}/"
            f"{_fmt(item.get('primary_loss_delta_max'))}"
        )
        failure = (
            f"{_fmt(item.get('F_delta_mean'))}/"
            f"{_fmt(item.get('F_delta_std'))}/"
            f"{_fmt(item.get('F_delta_min'))}/"
            f"{_fmt(item.get('F_delta_max'))}"
        )
        correct = (
            f"{_fmt(item.get('C_delta_mean'))}/"
            f"{_fmt(item.get('C_delta_std'))}/"
            f"{_fmt(item.get('C_delta_min'))}/"
            f"{_fmt(item.get('C_delta_max'))}"
        )
        sign = (
            f"{_fmt(item.get('primary_loss_delta_sign_consistency_fraction'))}/"
            f"{_fmt(item.get('F_delta_sign_consistency_fraction'))}/"
            f"{_fmt(item.get('C_delta_sign_consistency_fraction'))}"
        )
        lines.append(
            "| "
            f"{name} | {item.get('seed_count', 0)} | {item.get('successful_seed_count', 0)} | "
            f"{item.get('failed_seed_count', 0)} | {_fmt(item.get('claim_gate_pass_fraction'))} | "
            f"{item.get('recommendation_consistency', '')} | {primary} | {failure} | {correct} | {sign} |"
        )
    lines.extend(
        [
            "",
            "## Case Table",
            "",
            "| case | role | seed | status | support | primary final | F final | C final | D final | E final | active edges | coverage | weights |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for run in report["runs"]:
        weights = str(run.get("structural_weights", {})).replace("|", "/")
        if run["case_status"] == SUCCESS_STATUS:
            lines.append(
                "| "
                f"{run['case_name']} | {run['case_role']} | {run['seed']} | {run['case_status']} | "
                f"{run['support_mode']} | {run['final_primary_loss']:.6f} | {run['F_final']:.6e} | "
                f"{run['C_final']:.6e} | {run['D_final']:.6f} | {run['E_final']:.6e} | "
                f"{run['active_edge_count_final']} | {run['gradient_coverage_final']:.3f} | {weights} |"
            )
        else:
            lines.append(
                "| "
                f"{run['case_name']} | {run['case_role']} | {run['seed']} | {run['case_status']} | "
                f"{run.get('support_mode', '')} |  |  |  |  |  |  |  | {weights} |"
            )
    lines.extend(
        [
            "",
            "## Comparison Table",
            "",
            "| comparison | seed | category | claim gate | recommendation | primary delta | F delta | C delta | D delta | E delta | active delta | coverage delta | note |",
            "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in report["comparisons"]:
        note = str(item["interpretation_note"]).replace("|", "/")
        lines.append(
            "| "
            f"{item['comparison_name']} | {item['seed']} | {item['interpretation_category']} | "
            f"{item['claim_gate_passed']} | {item['recommendation']} | {_fmt(item['primary_loss_delta'])} | "
            f"{_fmt(item['F_delta'])} | {_fmt(item['C_delta'])} | {_fmt(item['D_delta'])} | "
            f"{_fmt(item['E_delta'])} | {_fmt(item['active_edge_count_delta'])} | "
            f"{_fmt(item['gradient_coverage_delta'])} | {note} |"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
        ]
    )
    for item in report["claim_boundary"]:
        lines.append(f"- {item}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_report(path: str | Path) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _print_summary(report: Mapping[str, Any]) -> None:
    print(
        "calibrated_structural_followup "
        f"tier={report['tier']} "
        f"evidence_tier={report.get('evidence_tier', report.get('tier', 'unknown'))} "
        f"readiness_basis={report.get('readiness_basis_tier', report.get('tier', 'unknown'))} "
        f"status={report['recommendation_status']} "
        f"readiness={report.get('calibrated_policy_readiness', 'unknown')} "
        f"next={report.get('recommended_next_stage', 'unknown')} "
        f"selected_allowed={report.get('selected_training_smoke_allowed', False)} "
        f"selected_blockers={','.join(report.get('selected_training_smoke_blocking_reasons', [])) or 'none'} "
        f"followup={','.join(report['recommended_cases_for_followup']) or 'none'} "
        f"all_weights={report['structural_weight_policy']['all_mode_candidate_weights']} "
        f"topk_weights={report['structural_weight_policy']['topk_weights']} "
        f"unsafe={','.join(report['unsafe_reference_cases']) or 'none'} "
        f"all_low_stable={report.get('all_low_structural_stable_nonworse', False)} "
        f"curriculum_stable={report.get('calibrated_curriculum_stable_nonworse', False)} "
        f"high_topk_stably_unsafe={report.get('topk_high_reference_stably_unsafe', False)}"
    )
    for item in report["comparisons"]:
        print(
            "calibrated_comparison "
            f"name={item['comparison_name']} "
            f"seed={item['seed']} "
            f"category={item['interpretation_category']} "
            f"claim={item['claim_gate_passed']} "
            f"primary_delta={item['primary_loss_delta']} "
            f"F_delta={item['F_delta']} "
            f"C_delta={item['C_delta']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run calibrated structural selected follow-up evidence.")
    parser.add_argument("--tier", choices=("smoke", "short"), default="smoke")
    parser.add_argument("--config", default="configs/training_smoke.yaml")
    parser.add_argument("--manifest", default="configs/controlled_ablation.yaml")
    parser.add_argument("--max-steps-override", type=int)
    parser.add_argument("--seeds")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json-in", default=None)
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--md-out", default=None)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        if args.json_in is None:
            raise SystemExit("--summary-only requires --json-in")
        report = _load_report(args.json_in)
    else:
        config = _small_config(load_training_smoke_config(ROOT / args.config))
        manifest = load_calibrated_followup_manifest(ROOT / args.manifest)
        report = run_calibrated_structural_followup(
            config,
            manifest=manifest,
            tier=args.tier,
            seeds=_parse_seeds(args.seeds),
            max_steps_override=args.max_steps_override,
        )
    if not args.no_write:
        default_json, default_md = default_output_paths(str(report.get("tier", args.tier)))
        json_path = Path(args.json_out) if args.json_out is not None else default_json
        md_path = Path(args.md_out) if args.md_out is not None else default_md
        if not json_path.is_absolute():
            json_path = ROOT / json_path
        if not md_path.is_absolute():
            md_path = ROOT / md_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        write_followup_json(report, json_path)
        write_followup_markdown(report, md_path)
    _print_summary(report)


if __name__ == "__main__":
    main()
