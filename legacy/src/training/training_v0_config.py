"""M32 formal training v0 configuration validation.

This module is report-only. It freezes and validates the canonical v0
configuration against the M31 pure edge-score benchmark package. It does not
run training, load datasets, create checkpoints, or choose models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/training_v0.yaml"
DEFAULT_BENCHMARK_JSON = ROOT / ".agent/tmp/pure_edge_benchmark_package.json"
DEFAULT_JSON_OUT = ROOT / ".agent/tmp/training_v0_config_validation.json"
DEFAULT_MD_OUT = ROOT / ".agent/tmp/training_v0_config_validation.md"

REQUIRED_METRICS = {
    "C_avalanche_node_mean",
    "F_avalanche_node_mean",
    "D_avalanche_rounds_mean",
    "E_consensus_node_mean",
    "active_edge_count",
    "gradient_coverage",
    "loss_total",
    "L_R",
    "L_D",
    "L_E",
}

REQUIRED_CLAIM_BOUNDARIES = {
    "no_model_quality_claim",
    "no_topk_superiority_claim_from_loss_only",
    "all_to_topk_support_change_confounded",
    "no_production_training_claim",
}

ALLOWED_BUDGET_HEAD_MODES = {
    "diagnostic_or_disabled",
    "diagnostic",
    "disabled",
}

FORBIDDEN_CONFIG_KEY_TOKENS = (
    "checkpoint",
    "dataset_loader",
    "scheduler",
    "model_selection",
)

FORBIDDEN_EXPLICIT_VALUES = {
    "hard_cap_trainable",
    "checkpointing",
    "dataset_loader",
    "scheduler",
    "model_selection",
    "production_training",
}


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value


def load_training_v0_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = _resolve_path(path)
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("training v0 config must contain a YAML mapping")
    return data


def load_training_v0_validation(path: str | Path) -> dict[str, Any]:
    report_path = _resolve_path(path)
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("training v0 validation report must contain a JSON object")
    return data


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_float(value: Any, default: float = 1.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _all_zero_weights(model_policy: Mapping[str, Any]) -> bool:
    return (
        _as_float(model_policy.get("sector_bias_weight")) == 0.0
        and _as_float(model_policy.get("role_bias_weight")) == 0.0
        and _as_float(model_policy.get("bridge_bias_weight")) == 0.0
    )


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "enabled", "allowed"}
    return bool(value)


def _recursive_forbidden_config_reasons(value: Any, *, path: str = "config") -> list[str]:
    reasons: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            key_lower = key_text.lower()
            child_path = f"{path}.{key_text}"
            if any(token in key_lower for token in FORBIDDEN_CONFIG_KEY_TOKENS):
                reasons.append(f"forbidden_config_field:{child_path}")
            if key_lower == "production_training" and _truthy(child):
                reasons.append(f"production_training_field_enabled:{child_path}")
            reasons.extend(_recursive_forbidden_config_reasons(child, path=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reasons.extend(_recursive_forbidden_config_reasons(child, path=f"{path}[{index}]"))
    elif isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in FORBIDDEN_EXPLICIT_VALUES:
            reasons.append(f"forbidden_config_value:{path}={normalized}")
    return reasons


def _validate_config(config: Mapping[str, Any], benchmark: Mapping[str, Any] | None) -> tuple[list[str], list[str]]:
    blockers: list[str] = []
    cautions: list[str] = []

    if str(config.get("training_v0_name", "")) != "pure_edge_training_v0":
        blockers.append("training_v0_name_must_be_pure_edge_training_v0")

    benchmark_source = str(config.get("benchmark_source", ""))
    benchmark_name = str((benchmark or {}).get("benchmark_name", benchmark_source))
    if benchmark_source != "pure_edge_score_benchmark_v1":
        blockers.append("benchmark_source_must_be_pure_edge_score_benchmark_v1")
    if benchmark and benchmark_name and benchmark_name != benchmark_source:
        blockers.append("benchmark_source_mismatch")

    model_policy = _mapping(config.get("model_policy"))
    if model_policy.get("model_family") != "pure_edge_score":
        blockers.append("model_family_must_be_pure_edge_score")
    if _truthy(model_policy.get("structural_bias_enabled", True)):
        blockers.append("structural_bias_enabled_must_be_false")
    if not _all_zero_weights(model_policy):
        blockers.append("structural_bias_weights_must_be_zero")
    if _truthy(model_policy.get("structural_heads_trainable", True)):
        blockers.append("structural_heads_trainable_must_be_false")
    if str(model_policy.get("budget_head_mode", "")) not in ALLOWED_BUDGET_HEAD_MODES:
        blockers.append("budget_head_mode_must_be_diagnostic_or_disabled")

    phase_policy = _mapping(config.get("phase_policy"))
    phase_0 = _mapping(phase_policy.get("phase_0"))
    phase_1 = _mapping(phase_policy.get("phase_1"))
    if phase_0.get("support_mode") != "all":
        blockers.append("phase_0_support_mode_must_be_all")
    if phase_1.get("support_mode") != "topk":
        blockers.append("phase_1_support_mode_must_be_topk")
    if not _truthy(phase_1.get("topk_structural_bias_disabled", False)):
        blockers.append("phase_1_topk_structural_bias_must_be_disabled")
    if not _truthy(phase_1.get("carry_parameters_from_phase_0", False)):
        blockers.append("phase_1_must_carry_parameters_from_phase_0")

    scale_policy = _mapping(config.get("scale_policy"))
    node_counts = scale_policy.get("node_counts", [])
    if not isinstance(node_counts, list) or int(scale_policy.get("default_node_count", -1) or -1) not in node_counts:
        blockers.append("scale_policy_default_node_count_must_be_listed")

    seed_policy = _mapping(config.get("seed_policy"))
    seeds = seed_policy.get("seeds", [])
    if not isinstance(seeds, list) or len(seeds) < int(seed_policy.get("min_successful_seed_count", 0) or 0):
        blockers.append("seed_policy_requires_enough_configured_seeds")

    avalanche_policy = _mapping(config.get("avalanche_policy"))
    profiles = avalanche_policy.get("profiles", [])
    if not isinstance(profiles, list) or avalanche_policy.get("default_profile") not in profiles:
        blockers.append("avalanche_default_profile_must_be_listed")

    loss_policy = _mapping(config.get("loss_policy"))
    if _truthy(loss_policy.get("direct_link_physics_loss_allowed", True)):
        blockers.append("direct_link_physics_loss_must_be_forbidden")
    for key in ("sinr_loss_allowed", "bler_loss_allowed", "harq_loss_allowed", "coverage_loss_allowed"):
        if _truthy(loss_policy.get(key, False)):
            blockers.append(f"{key}_must_be_false")
    if str(loss_policy.get("tail_mode", "")) != "max":
        blockers.append("loss_policy_tail_mode_must_be_max")

    reporting_policy = _mapping(config.get("reporting_policy"))
    metrics = set(str(item) for item in reporting_policy.get("required_metrics", []) if isinstance(item, str))
    missing_metrics = sorted(REQUIRED_METRICS - metrics)
    if missing_metrics:
        blockers.append("missing_required_metrics:" + ",".join(missing_metrics))

    claim_boundaries = set(
        str(item) for item in reporting_policy.get("required_claim_boundaries", []) if isinstance(item, str)
    )
    missing_claims = sorted(REQUIRED_CLAIM_BOUNDARIES - claim_boundaries)
    if missing_claims:
        blockers.append("missing_required_claim_boundaries:" + ",".join(missing_claims))

    contract = _mapping(config.get("contract"))
    required_contract_flags = {
        "train_eval_deploy_same_graph_rule",
        "no_structural_bias_in_baseline",
        "no_dense_NxN",
        "no_sampling_or_monte_carlo",
        "no_pbft",
        "no_direct_link_reliability_loss",
    }
    missing_contract_flags = sorted(flag for flag in required_contract_flags if not _truthy(contract.get(flag, False)))
    if missing_contract_flags:
        blockers.append("missing_required_contract_flags:" + ",".join(missing_contract_flags))

    forbidden_reasons = _recursive_forbidden_config_reasons(config)
    if forbidden_reasons:
        blockers.extend(forbidden_reasons)

    if "small_realistic" in profiles and _truthy(avalanche_policy.get("realistic_profile_explicit_only", False)):
        cautions.append("small_realistic_profile_is_explicit_only")
    if scale_policy.get("large_node_counts_explicit_only"):
        cautions.append("large_node_counts_are_explicit_only")
    cautions.append("training_v0_config_freeze_is_report_only")

    return blockers, cautions


def _readiness_blockers(
    benchmark_path: Path,
    benchmark: Mapping[str, Any] | None,
    load_error: str | None,
) -> list[str]:
    blockers: list[str] = []
    if benchmark is None:
        blockers.append("missing_benchmark_artifact" if load_error is None else f"invalid_benchmark_artifact:{load_error}")
        return blockers
    if not benchmark.get("ready_for_M32_formal_training_config", False):
        blockers.append("benchmark_not_ready_for_M32_formal_training_config")
    if str(benchmark.get("pure_edge_benchmark_package_status", "")) == "blocked":
        blockers.append("benchmark_package_status_blocked")
    return blockers


def build_training_v0_config_validation_report(
    *,
    config: Mapping[str, Any],
    benchmark: Mapping[str, Any] | None,
    config_path: str | Path = DEFAULT_CONFIG,
    benchmark_json: str | Path = DEFAULT_BENCHMARK_JSON,
    benchmark_load_error: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    benchmark_path = _resolve_path(benchmark_json)
    readiness_blockers = _readiness_blockers(benchmark_path, benchmark, benchmark_load_error)
    config_blockers, cautions = _validate_config(config, benchmark)

    effective_readiness_blockers = [] if force else readiness_blockers
    if force and readiness_blockers:
        cautions.append("forced_benchmark_readiness_override:" + ",".join(readiness_blockers))

    blocking_reasons = [*effective_readiness_blockers, *config_blockers]
    if effective_readiness_blockers:
        status = "blocked"
        next_stage = "regenerate_benchmark_package"
    elif config_blockers:
        status = "invalid_training_v0_config"
        next_stage = "fix_training_v0_config"
    else:
        status = "valid_training_v0_config"
        next_stage = "M33_formal_training_v0_smoke"

    ready_for_m33 = status == "valid_training_v0_config"
    benchmark_name = str((benchmark or {}).get("benchmark_name", config.get("benchmark_source", "")))

    reporting_policy = _mapping(config.get("reporting_policy"))
    metrics = list(reporting_policy.get("required_metrics", []) if isinstance(reporting_policy.get("required_metrics", []), list) else [])
    claim_boundaries = list(
        reporting_policy.get("required_claim_boundaries", [])
        if isinstance(reporting_policy.get("required_claim_boundaries", []), list)
        else []
    )

    return {
        "audit_scope": "M32 formal training v0 config freeze",
        "training_v0_config_status": status,
        "ready_for_M33_formal_training_v0_smoke": ready_for_m33,
        "readiness_source_artifact": str(benchmark_path),
        "source_ready_for_M32": bool((benchmark or {}).get("ready_for_M32_formal_training_config", False)),
        "forced": bool(force),
        "config_path": str(_resolve_path(config_path)),
        "benchmark_name": benchmark_name,
        "training_v0_name": str(config.get("training_v0_name", "")),
        "benchmark_source": str(config.get("benchmark_source", "")),
        "phase_0_policy": dict(_mapping(_mapping(config.get("phase_policy")).get("phase_0"))),
        "phase_1_policy": dict(_mapping(_mapping(config.get("phase_policy")).get("phase_1"))),
        "seed_policy": dict(_mapping(config.get("seed_policy"))),
        "scale_policy": dict(_mapping(config.get("scale_policy"))),
        "avalanche_policy": dict(_mapping(config.get("avalanche_policy"))),
        "loss_policy": dict(_mapping(config.get("loss_policy"))),
        "model_policy": dict(_mapping(config.get("model_policy"))),
        "claim_policy": {
            "required_claim_boundaries": claim_boundaries,
            "no_model_quality_claim": "no_model_quality_claim" in claim_boundaries,
            "no_topk_superiority_claim_from_loss_only": "no_topk_superiority_claim_from_loss_only"
            in claim_boundaries,
            "all_to_topk_support_change_confounded": "all_to_topk_support_change_confounded" in claim_boundaries,
            "no_production_training_claim": "no_production_training_claim" in claim_boundaries,
        },
        "reporting_policy": {
            "required_metrics": metrics,
            "required_claim_boundaries": claim_boundaries,
            "missing_required_metrics": sorted(REQUIRED_METRICS - set(str(item) for item in metrics)),
            "missing_required_claim_boundaries": sorted(REQUIRED_CLAIM_BOUNDARIES - set(str(item) for item in claim_boundaries)),
        },
        "contract": dict(_mapping(config.get("contract"))),
        "validation_checks": {
            "benchmark_ready": not readiness_blockers,
            "structural_bias_disabled": not _truthy(_mapping(config.get("model_policy")).get("structural_bias_enabled", True)),
            "structural_weights_zero": _all_zero_weights(_mapping(config.get("model_policy"))),
            "structural_heads_trainable_false": not _truthy(
                _mapping(config.get("model_policy")).get("structural_heads_trainable", True)
            ),
            "phase_0_all": _mapping(_mapping(config.get("phase_policy")).get("phase_0")).get("support_mode") == "all",
            "phase_1_topk": _mapping(_mapping(config.get("phase_policy")).get("phase_1")).get("support_mode") == "topk",
            "phase_1_topk_structural_bias_disabled": _truthy(
                _mapping(_mapping(config.get("phase_policy")).get("phase_1")).get(
                    "topk_structural_bias_disabled",
                    False,
                )
            ),
            "direct_link_physics_loss_forbidden": not _truthy(
                _mapping(config.get("loss_policy")).get("direct_link_physics_loss_allowed", True)
            ),
            "required_metrics_present": not (REQUIRED_METRICS - set(str(item) for item in metrics)),
            "required_claim_boundaries_present": not (
                REQUIRED_CLAIM_BOUNDARIES - set(str(item) for item in claim_boundaries)
            ),
        },
        "blocking_reasons": blocking_reasons,
        "caution_reasons": cautions,
        "recommended_next_stage": next_stage,
    }


def validate_training_v0_config(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    benchmark_json: str | Path = DEFAULT_BENCHMARK_JSON,
    force: bool = False,
) -> dict[str, Any]:
    resolved_config = _resolve_path(config_path)
    resolved_benchmark = _resolve_path(benchmark_json)
    config = load_training_v0_config(resolved_config)
    benchmark: dict[str, Any] | None = None
    load_error: str | None = None
    if resolved_benchmark.exists():
        try:
            benchmark = _load_json(resolved_benchmark)
        except (json.JSONDecodeError, ValueError) as exc:
            load_error = str(exc)
    return build_training_v0_config_validation_report(
        config=config,
        benchmark=benchmark,
        config_path=resolved_config,
        benchmark_json=resolved_benchmark,
        benchmark_load_error=load_error,
        force=force,
    )


def write_training_v0_validation_json(report: Mapping[str, Any], path: str | Path = DEFAULT_JSON_OUT) -> None:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def write_training_v0_validation_markdown(report: Mapping[str, Any], path: str | Path = DEFAULT_MD_OUT) -> None:
    output_path = _resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_training_v0_validation_markdown(report), encoding="utf-8")


def render_training_v0_validation_markdown(report: Mapping[str, Any]) -> str:
    blockers = report.get("blocking_reasons", [])
    cautions = report.get("caution_reasons", [])
    phase_0 = _mapping(report.get("phase_0_policy"))
    phase_1 = _mapping(report.get("phase_1_policy"))
    seed_policy = _mapping(report.get("seed_policy"))
    scale_policy = _mapping(report.get("scale_policy"))
    avalanche_policy = _mapping(report.get("avalanche_policy"))
    claim_policy = _mapping(report.get("claim_policy"))
    metrics = _mapping(report.get("reporting_policy")).get("required_metrics", [])

    lines = [
        "# Training v0 Config Validation",
        "",
        "## Decision Summary",
        "",
        f"- Status: `{report.get('training_v0_config_status')}`",
        f"- Ready for M33 formal training v0 smoke: `{report.get('ready_for_M33_formal_training_v0_smoke')}`",
        f"- Recommended next stage: `{report.get('recommended_next_stage')}`",
        f"- Readiness source: `{report.get('readiness_source_artifact')}`",
        f"- Benchmark: `{report.get('benchmark_name')}`",
        "",
        "## Phase Policy",
        "",
        "| Phase | Support mode | Max steps | Key guard |",
        "| --- | --- | ---: | --- |",
        f"| phase_0 | `{phase_0.get('support_mode')}` | {phase_0.get('max_steps')} | `{phase_0.get('name')}` |",
        "| phase_1 | "
        f"`{phase_1.get('support_mode')}` | {phase_1.get('max_steps')} | "
        f"topk structural bias disabled=`{phase_1.get('topk_structural_bias_disabled')}` |",
        "",
        "## Seed, Scale, Profile Policy",
        "",
        f"- Seeds: `{seed_policy.get('seeds')}`; minimum successful seeds: `{seed_policy.get('min_successful_seed_count')}`",
        f"- Node counts: `{scale_policy.get('node_counts')}`; default: `{scale_policy.get('default_node_count')}`",
        f"- Explicit-only large node counts: `{scale_policy.get('large_node_counts_explicit_only')}`",
        f"- Avalanche profiles: `{avalanche_policy.get('profiles')}`; default: `{avalanche_policy.get('default_profile')}`",
        "",
        "## Claim Boundaries",
        "",
        f"- No model quality claim: `{claim_policy.get('no_model_quality_claim')}`",
        f"- No top-k superiority claim from loss only: `{claim_policy.get('no_topk_superiority_claim_from_loss_only')}`",
        f"- All-to-topk support change is confounded: `{claim_policy.get('all_to_topk_support_change_confounded')}`",
        f"- No production training claim: `{claim_policy.get('no_production_training_claim')}`",
        "",
        "## Required Metrics",
        "",
        ", ".join(f"`{metric}`" for metric in metrics),
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {reason}" for reason in blockers or ["none"])
    lines.extend(["", "## Cautions", ""])
    lines.extend(f"- {reason}" for reason in cautions or ["none"])
    lines.extend(
        [
            "",
            "## Contract Notes",
            "",
            "- M32 freezes a formal v0 configuration; it does not run training.",
            "- Structural bias is disabled and structural heads are diagnostic or disabled.",
            "- Direct link/SINR/BLER/HARQ/coverage losses remain forbidden.",
            "- Production training, checkpointing, dataset loading, scheduling, and model selection are outside this contract.",
            "",
        ]
    )
    return "\n".join(lines)
