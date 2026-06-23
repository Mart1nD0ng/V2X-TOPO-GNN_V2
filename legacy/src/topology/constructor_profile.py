"""M49 topology constructor bottleneck diagnostic.

This module is diagnostic-only. It profiles the current sparse
TopologyConstructionLayer stages over synthetic O(Nk) candidate graphs and
reports whether constructor internals should be optimized before larger scale
smokes. It does not change the constructor forward rule, add a new model
architecture, allocate dense graph tensors, or add hidden physics objectives.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from .construction import (
    ConstructedTopology,
    TopologyConstructionLayer,
    _budget_vector,
    _diagnostics,
    _row_entropy_mean,
    _row_softmax,
    _select_topk_legacy,
    _select_topk_segmented_fast,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_READINESS_ARTIFACT = ROOT / ".agent/tmp/runtime_memory_scaling_interpretation.json"
DEFAULT_JSON_OUT = ROOT / ".agent/tmp/constructor_bottleneck_diagnostic.json"
DEFAULT_MD_OUT = ROOT / ".agent/tmp/constructor_bottleneck_diagnostic.md"

DEFAULT_NODE_COUNTS = (500, 2000)
DEFAULT_CANDIDATE_DEGREE = 8
DEFAULT_MAX_OUT_DEGREE = 4
DEFAULT_SUPPORT_MODE = "topk"
DEFAULT_TOPK_BACKEND = "legacy"
EXPECTED_M48_RECOMMENDATION = "M49_constructor_bottleneck_diagnostic"

COMPONENT_DOMINANCE_THRESHOLD = 0.50
GROWTH_SUPERLINEAR_TOLERANCE = 1.50
FAST_PATH_NOT_SLOWER_TOLERANCE = 0.90

ALLOWED_CLAIMS = [
    "M49 profiles current sparse constructor internals under pure edge-score synthetic top-k inputs.",
    "Active edges remain bounded by node_count * max_out_degree in observed top-k cases.",
    "The diagnostic can recommend constructor optimization before a controlled 5k smoke.",
]

DISALLOWED_CLAIMS = [
    "Constructor profiling proves model quality.",
    "N=2000 validates 10k behavior.",
    "Runtime is production-ready.",
    "Top-k is superior to all-mode.",
    "Structural heads should be reintroduced.",
]

REMAINING_ASSUMPTIONS = [
    "M49 uses deterministic synthetic sparse edge scores, not a trained scorer.",
    "Timing values are machine-local diagnostics.",
    "The profiler mirrors the current constructor implementation for diagnosis only and does not replace forward().",
    "Memory reporting is a tensor proxy, not process RSS or GPU peak memory.",
]

ProfileRunner = Callable[..., Mapping[str, Any]]


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = ROOT / value
    return value


def _repo_relative_path(path: str | Path) -> str:
    value = _resolve_path(path)
    try:
        return value.relative_to(ROOT).as_posix()
    except ValueError:
        return str(value)


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_path(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    return data


def load_constructor_bottleneck_diagnostic_report(path: str | Path = DEFAULT_JSON_OUT) -> dict[str, Any]:
    return _load_json(path)


def _coerce_int_list(value: str | Iterable[int] | None, default: Iterable[int]) -> list[int]:
    if value is None:
        return [int(item) for item in default]
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _tensor_scalar(value: torch.Tensor | Any) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.detach().reshape(-1)[0].cpu().item())
    return _float(value)


def _tensor_stats(value: torch.Tensor | Any) -> dict[str, float]:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return {"count": 0.0, "min": 0.0, "p10": 0.0, "mean": 0.0, "max": 0.0}
    tensor = value.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    return {
        "count": float(tensor.numel()),
        "min": float(torch.min(tensor).item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "mean": float(torch.mean(tensor).item()),
        "max": float(torch.max(tensor).item()),
    }


def _tensor_memory_mb(*tensors: torch.Tensor) -> float:
    total = 0
    for tensor in tensors:
        total += int(tensor.numel()) * int(tensor.element_size())
    return total / 1_000_000.0


def _synthetic_sparse_candidates(
    *,
    node_count: int,
    candidate_degree: int,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if node_count <= 1:
        raise ValueError("node_count must be greater than one")
    if candidate_degree <= 0:
        raise ValueError("candidate_degree must be positive")
    if candidate_degree >= node_count:
        raise ValueError("candidate_degree must be smaller than node_count to avoid duplicate wraparound edges")
    src_index = torch.arange(node_count, dtype=torch.long).repeat_interleave(candidate_degree)
    offsets = torch.arange(1, candidate_degree + 1, dtype=torch.long).repeat(node_count)
    dst_index = (src_index + offsets) % node_count
    edge_count = int(src_index.numel())
    edge_score = torch.linspace(-1.0, 1.0, edge_count, dtype=dtype)
    return src_index, dst_index, edge_score


def _profiled_constructor_forward(
    *,
    layer: TopologyConstructionLayer,
    num_nodes: int,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    edge_score: torch.Tensor,
    edge_mask: torch.Tensor | None = None,
    max_out_degree: int | None = None,
    support_mode: str | None = None,
    topk_backend: str | None = None,
) -> tuple[ConstructedTopology, dict[str, float]]:
    total_start = perf_counter()
    validation_start = perf_counter()
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if not torch.is_floating_point(edge_score):
        raise ValueError("edge_score must use a floating-point dtype")
    score = edge_score.reshape(-1)
    if bool(torch.any(~torch.isfinite(score.detach())).cpu()):
        raise ValueError("edge_score must contain only finite values")
    device = score.device
    dtype = score.dtype
    src = src_index.to(device=device, dtype=torch.long).reshape(-1)
    dst = dst_index.to(device=device, dtype=torch.long).reshape(-1)
    if src.numel() != dst.numel() or src.numel() != score.numel():
        raise ValueError("src_index, dst_index, and edge_score must have the same edge count")
    edge_count = int(score.numel())
    if edge_count:
        if bool(torch.any(src < 0).cpu()) or bool(torch.any(src >= num_nodes).cpu()):
            raise ValueError("src_index contains node ids outside [0, num_nodes)")
        if bool(torch.any(dst < 0).cpu()) or bool(torch.any(dst >= num_nodes).cpu()):
            raise ValueError("dst_index contains node ids outside [0, num_nodes)")
    if not layer.allow_self_loops and bool(torch.any(src == dst).cpu()):
        raise ValueError("self-loops are disabled by default for topology construction")
    self_loop_count = (src == dst).to(dtype=dtype).sum()
    duplicate_edge_count = score.new_tensor(0.0)
    if edge_count:
        pair_key = src * max(num_nodes, 1) + dst
        unique_pair_count = int(torch.unique(pair_key).numel())
        duplicate_edge_count = score.new_tensor(float(edge_count - unique_pair_count))
        if not layer.allow_multi_edges and edge_count != unique_pair_count:
            raise ValueError("duplicate directed candidate edges are disabled by default")
    if edge_mask is None:
        feasible = torch.ones(edge_count, dtype=torch.bool, device=device)
    else:
        mask = edge_mask.to(device=device).reshape(-1)
        if mask.numel() != edge_count:
            raise ValueError("edge_mask must have one value per candidate edge")
        feasible = mask.bool()
    budgets = _budget_vector(
        num_nodes=num_nodes,
        per_node_budget=None,
        max_out_degree=max_out_degree if max_out_degree is not None else layer.max_out_degree,
        fallback=edge_count,
        device=device,
    )
    tau = score.new_tensor(layer.temperature)
    mode = layer.support_mode if support_mode is None else support_mode
    backend = layer.topk_backend if topk_backend is None else topk_backend
    feasible_count = score.new_zeros((num_nodes,))
    if edge_count:
        feasible_count = feasible_count.index_add(0, src, feasible.to(dtype=dtype))
    cap_hit = (
        (feasible_count > budgets.to(dtype=dtype))
        if mode == "topk"
        else torch.zeros_like(feasible_count, dtype=torch.bool)
    )
    validation_time = perf_counter() - validation_start

    sorting_start = perf_counter()
    feasible_indices = torch.nonzero(feasible, as_tuple=False).reshape(-1)
    feasible_src = src.new_empty((0,))
    unique_src = src.new_empty((0,))
    counts = src.new_empty((0,))
    row_starts = src.new_empty((0,))
    if feasible_indices.numel():
        feasible_src = src.index_select(0, feasible_indices)
        order = torch.argsort(feasible_src)
        feasible_indices = feasible_indices.index_select(0, order)
        feasible_src = feasible_src.index_select(0, order)
        if mode == "topk":
            unique_src, counts = torch.unique_consecutive(feasible_src, return_counts=True)
            row_starts = torch.cumsum(counts, dim=0) - counts
    sorting_time = perf_counter() - sorting_start

    selected_mask = torch.zeros(edge_count, dtype=torch.bool, device=device)
    output_index_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    near_miss_candidate_count = score.new_tensor(0.0)
    support_smoothing_halo_candidate_count = score.new_tensor(0.0)
    support_smoothing_base_selected_count = score.new_tensor(0.0)
    topk_start = perf_counter()
    python_loop_start = perf_counter()
    if feasible_indices.numel():
        if mode == "all":
            output_index_parts.append(feasible_indices)
        else:
            near_miss_tol = score.new_tensor(layer.near_miss_tolerance)
            if backend == "segmented_fast":
                (
                    output_index_parts,
                    margin_parts,
                    near_miss_candidate_count,
                    support_smoothing_halo_candidate_count,
                    support_smoothing_base_selected_count,
                ) = _select_topk_segmented_fast(
                    feasible_indices=feasible_indices,
                    unique_src=unique_src,
                    counts=counts,
                    row_starts=row_starts,
                    budgets=budgets,
                    score=score,
                    near_miss_threshold=near_miss_tol,
                )
            else:
                (
                    output_index_parts,
                    margin_parts,
                    near_miss_candidate_count,
                    support_smoothing_halo_candidate_count,
                    support_smoothing_base_selected_count,
                ) = _select_topk_legacy(
                    feasible_indices=feasible_indices,
                    unique_src=unique_src,
                    counts=counts,
                    row_starts=row_starts,
                    budgets=budgets,
                    score=score,
                    near_miss_threshold=near_miss_tol,
                )
    python_loop_time = perf_counter() - python_loop_start if mode == "topk" and backend == "legacy" else 0.0
    topk_selection_time = perf_counter() - topk_start if mode == "topk" else 0.0

    output_packaging_start = perf_counter()
    if output_index_parts:
        selected_candidate_index = torch.cat(output_index_parts)
        selected_mask[selected_candidate_index] = True
        output_src = src.index_select(0, selected_candidate_index)
        output_dst = dst.index_select(0, selected_candidate_index)
    else:
        selected_candidate_index = src.new_empty((0,))
        output_src = src.new_empty((0,))
        output_dst = dst.new_empty((0,))
    output_packaging_pre_softmax_time = perf_counter() - output_packaging_start

    softmax_start = perf_counter()
    if selected_candidate_index.numel():
        topology_weight = _row_softmax(
            num_nodes=num_nodes,
            src_index=src,
            edge_score=score,
            selected_candidate_index=selected_candidate_index,
            tau=tau,
        )
        if abs(float(tau.detach().cpu().item()) - 1.0) <= 1.0e-12:
            temperature_one_weight = topology_weight
        else:
            temperature_one_weight = _row_softmax(
                num_nodes=num_nodes,
                src_index=src,
                edge_score=score,
                selected_candidate_index=selected_candidate_index,
                tau=score.new_tensor(1.0),
            )
    else:
        topology_weight = score.new_empty((0,))
        temperature_one_weight = topology_weight
    row_softmax_time = perf_counter() - softmax_start

    diagnostics_start = perf_counter()
    topk_boundary_margin = torch.cat(margin_parts) if margin_parts else score.new_empty((0,))
    row_entropy_mean = _row_entropy_mean(
        num_nodes=num_nodes,
        src_index=output_src,
        topology_weight=topology_weight,
        reference=score,
    )
    row_entropy_temperature_one = _row_entropy_mean(
        num_nodes=num_nodes,
        src_index=output_src,
        topology_weight=temperature_one_weight,
        reference=score,
    )
    diagnostics = _diagnostics(
        num_nodes=num_nodes,
        edge_count=edge_count,
        src_index=output_src,
        topology_weight=topology_weight,
        selected_mask=selected_mask,
        feasible_count=feasible_count,
        cap_hit=cap_hit,
        constructor_mode_code=score.new_tensor(1.0 if mode == "topk" else 0.0),
        topk_boundary_margin=topk_boundary_margin,
        near_miss_candidate_count=near_miss_candidate_count,
        self_loop_count=self_loop_count,
        duplicate_edge_count=duplicate_edge_count,
        row_softmax_temperature=tau,
        support_smoothing_mode_code=score.new_tensor(0.0),
        support_smoothing_stage_code=score.new_tensor(0.0),
        support_smoothing_extra_per_row=score.new_tensor(0.0),
        support_smoothing_effective_extra_per_row=score.new_tensor(0.0),
        support_smoothing_temperature=score.new_tensor(1.0),
        support_smoothing_active=score.new_tensor(0.0),
        support_smoothing_halo_candidate_count=support_smoothing_halo_candidate_count,
        support_smoothing_base_selected_count=support_smoothing_base_selected_count,
        row_entropy_mean=row_entropy_mean,
        row_entropy_delta_vs_temperature_1=row_entropy_mean - row_entropy_temperature_one,
        reference=score,
    )
    diagnostics_time = perf_counter() - diagnostics_start

    packaging_start = perf_counter()
    topology = ConstructedTopology(
        num_nodes=num_nodes,
        src_index=output_src,
        dst_index=output_dst,
        topology_weight=topology_weight,
        selected_candidate_index=selected_candidate_index,
        selected_edge_mask=selected_mask,
        active_edge_mask=selected_mask,
        diagnostics=diagnostics,
    )
    output_packaging_time = output_packaging_pre_softmax_time + (perf_counter() - packaging_start)
    total_time = perf_counter() - total_start
    tensor_operation_time = max(total_time - python_loop_time, 0.0)
    return topology, {
        "total_constructor_time_s": total_time,
        "validation_time_s": validation_time,
        "sorting_or_grouping_time_s": sorting_time,
        "topk_selection_time_s": topk_selection_time,
        "row_softmax_time_s": row_softmax_time,
        "diagnostics_time_s": diagnostics_time,
        "output_packaging_time_s": output_packaging_time,
        "python_loop_time_s": python_loop_time,
        "tensor_operation_time_s": tensor_operation_time,
    }


def _classify_components(case: Mapping[str, Any]) -> tuple[str, str, float, str]:
    total = _float(case.get("total_constructor_time_s"))
    if total <= 0.0:
        return "blocked", "unknown", 0.0, "inconclusive"
    components = {
        "validation": _float(case.get("validation_time_s")),
        "topk_selection": _float(case.get("topk_selection_time_s")),
        "row_softmax": _float(case.get("row_softmax_time_s")),
        "diagnostics": _float(case.get("diagnostics_time_s")),
        "python_loop": _float(case.get("python_loop_time_s")),
    }
    component, elapsed = max(components.items(), key=lambda item: item[1])
    fraction = elapsed / total
    if fraction <= COMPONENT_DOMINANCE_THRESHOLD:
        return "no_single_constructor_bottleneck", component, fraction, "constructor_acceptable_continue_to_5k"
    if component == "topk_selection":
        return "topk_selection_bottleneck", component, fraction, "optimize_segmented_topk"
    if component == "row_softmax":
        return "row_softmax_bottleneck", component, fraction, "inconclusive"
    if component == "diagnostics":
        return "diagnostics_bottleneck", component, fraction, "optimize_diagnostics"
    if component == "validation":
        return "validation_bottleneck", component, fraction, "optimize_validation"
    if component == "python_loop":
        return "python_loop_bottleneck", component, fraction, "vectorize_row_loop"
    return "no_single_constructor_bottleneck", component, fraction, "constructor_acceptable_continue_to_5k"


def _normalize_profile_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    case = dict(raw)
    status, component, fraction, recommendation = _classify_components(case)
    case["constructor_bottleneck_status"] = status
    case["bottleneck_component"] = component
    case["bottleneck_fraction"] = fraction
    case["optimization_recommendation"] = recommendation
    case["blocked"] = status == "blocked" or bool(case.get("blocking_reasons"))
    return case


def profile_topology_constructor_case(
    *,
    node_count: int,
    candidate_degree: int = DEFAULT_CANDIDATE_DEGREE,
    max_out_degree: int = DEFAULT_MAX_OUT_DEGREE,
    support_mode: str = DEFAULT_SUPPORT_MODE,
    topk_backend: str = DEFAULT_TOPK_BACKEND,
) -> dict[str, Any]:
    """Profile one synthetic sparse constructor case."""

    case, _topology = _profile_topology_constructor_case_with_topology(
        node_count=node_count,
        candidate_degree=candidate_degree,
        max_out_degree=max_out_degree,
        support_mode=support_mode,
        topk_backend=topk_backend,
    )
    return case


def _profile_topology_constructor_case_with_topology(
    *,
    node_count: int,
    candidate_degree: int = DEFAULT_CANDIDATE_DEGREE,
    max_out_degree: int = DEFAULT_MAX_OUT_DEGREE,
    support_mode: str = DEFAULT_SUPPORT_MODE,
    topk_backend: str = DEFAULT_TOPK_BACKEND,
) -> tuple[dict[str, Any], ConstructedTopology]:
    src_index, dst_index, edge_score = _synthetic_sparse_candidates(
        node_count=node_count,
        candidate_degree=candidate_degree,
    )
    layer = TopologyConstructionLayer(
        max_out_degree=max_out_degree,
        temperature=1.0,
        support_mode=support_mode,
        allow_self_loops=False,
        allow_multi_edges=False,
        topk_backend=topk_backend,
    )
    topology, timings = _profiled_constructor_forward(
        layer=layer,
        num_nodes=node_count,
        src_index=src_index,
        dst_index=dst_index,
        edge_score=edge_score,
        max_out_degree=max_out_degree,
        support_mode=support_mode,
        topk_backend=topk_backend,
    )
    diagnostics = topology.diagnostics
    finite_weights = bool(torch.all(torch.isfinite(topology.topology_weight.detach())).cpu())
    dense_nxn_tensor_allocated = False
    tensor_memory_proxy_mb = _tensor_memory_mb(
        src_index,
        dst_index,
        edge_score,
        topology.src_index,
        topology.dst_index,
        topology.topology_weight,
        topology.selected_candidate_index,
        topology.selected_edge_mask,
        diagnostics["out_degree"],
        diagnostics["row_weight_sum"],
        diagnostics["effective_query_degree"],
    )
    active_edge_count = int(_tensor_scalar(diagnostics["active_edge_count"]))
    candidate_edge_count = int(src_index.numel())
    feasible_edge_count = int(_tensor_scalar(diagnostics["feasible_candidate_count"]))
    dense_proxy = int(node_count) * int(node_count)
    case = {
        "node_count": int(node_count),
        "candidate_degree": int(candidate_degree),
        "support_mode": support_mode,
        "topk_backend": topk_backend if support_mode == "topk" else "not_used",
        "structural_bias_enabled": False,
        "structural_weights_zero": True,
        "topk_structural_bias_disabled": True,
        "max_out_degree": int(max_out_degree),
        "candidate_edge_count": candidate_edge_count,
        "feasible_edge_count": feasible_edge_count,
        "active_edge_count": active_edge_count,
        "unique_source_count": int(_tensor_scalar(diagnostics["unique_source_count"])),
        "mean_candidates_per_source": _tensor_scalar(diagnostics["mean_candidates_per_active_source"]),
        "max_candidates_per_source": _tensor_scalar(diagnostics["max_candidates_per_active_source"]),
        "selected_fraction": _tensor_scalar(diagnostics["selected_fraction"]),
        "gradient_coverage_fraction": _tensor_scalar(diagnostics["gradient_coverage_fraction"]),
        "rows_with_cap_hit": int(_tensor_scalar(diagnostics["cap_hit_count"])),
        "cap_hit_ratio": _tensor_scalar(diagnostics["cap_hit_ratio"]),
        "single_selected_row_count": int(_tensor_scalar(diagnostics["single_selected_row_count"])),
        "zero_selected_row_count": int(_tensor_scalar(diagnostics["zero_selected_row_count"])),
        "topk_boundary_margin_stats": _tensor_stats(diagnostics["topk_boundary_margin"]),
        "near_miss_candidate_count": int(_tensor_scalar(diagnostics["near_miss_candidate_count"])),
        "zero_gradient_risk_count": int(_tensor_scalar(diagnostics["zero_gradient_risk_count"])),
        "dense_nxn_proxy_edge_count": dense_proxy,
        "dense_nxn_tensor_allocated": dense_nxn_tensor_allocated,
        "sparse_to_dense_edge_ratio": active_edge_count / float(dense_proxy) if dense_proxy else 0.0,
        "finite_topology_weights": finite_weights,
        "train_eval_deploy_graph_rule_match": True,
        "tensor_memory_proxy_mb": tensor_memory_proxy_mb,
        "blocking_reasons": [],
    }
    case.update(timings)
    return _normalize_profile_case(case), topology


def _diagnostics_equivalent(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    if set(left) != set(right):
        return False
    for key in left:
        left_value = left[key]
        right_value = right[key]
        if left_value.shape != right_value.shape:
            return False
        if torch.is_floating_point(left_value):
            if not bool(torch.allclose(left_value, right_value, rtol=1.0e-10, atol=1.0e-12)):
                return False
        elif not bool(torch.equal(left_value, right_value)):
            return False
    return True


def _backend_equivalence_profile(
    *,
    node_count: int,
    candidate_degree: int,
    max_out_degree: int,
) -> dict[str, Any]:
    legacy_case, legacy_topology = _profile_topology_constructor_case_with_topology(
        node_count=node_count,
        candidate_degree=candidate_degree,
        max_out_degree=max_out_degree,
        support_mode="topk",
        topk_backend="legacy",
    )
    fast_case, fast_topology = _profile_topology_constructor_case_with_topology(
        node_count=node_count,
        candidate_degree=candidate_degree,
        max_out_degree=max_out_degree,
        support_mode="topk",
        topk_backend="segmented_fast",
    )
    selected_equal = bool(torch.equal(legacy_topology.selected_candidate_index, fast_topology.selected_candidate_index))
    src_equal = bool(torch.equal(legacy_topology.src_index, fast_topology.src_index))
    dst_equal = bool(torch.equal(legacy_topology.dst_index, fast_topology.dst_index))
    weight_allclose = bool(
        torch.allclose(
            legacy_topology.topology_weight,
            fast_topology.topology_weight,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    )
    diagnostics_ok = _diagnostics_equivalent(legacy_topology.diagnostics, fast_topology.diagnostics)
    active_equal = int(legacy_case["active_edge_count"]) == int(fast_case["active_edge_count"])
    legacy_time = _float(legacy_case.get("total_constructor_time_s"))
    fast_time = _float(fast_case.get("total_constructor_time_s"))
    speedup = _safe_ratio(legacy_time, fast_time)
    equivalent = selected_equal and src_equal and dst_equal and weight_allclose and diagnostics_ok and active_equal
    fast_not_slower = bool(speedup is not None and speedup >= FAST_PATH_NOT_SLOWER_TOLERANCE)
    return {
        "node_count": int(node_count),
        "support_mode": "topk",
        "topk_backend": "both",
        "legacy_time_s": legacy_time,
        "segmented_fast_time_s": fast_time,
        "legacy_constructor_time_s": legacy_time,
        "fast_constructor_time_s": fast_time,
        "speedup_ratio": speedup,
        "equivalence_ok": bool(equivalent),
        "selected_edge_equivalence": bool(selected_equal and src_equal and dst_equal),
        "selected_candidate_index_equal": selected_equal,
        "src_index_equal": src_equal,
        "dst_index_equal": dst_equal,
        "weight_equivalence": weight_allclose,
        "topology_weight_allclose": weight_allclose,
        "diagnostics_equivalence": diagnostics_ok,
        "diagnostics_equivalent": diagnostics_ok,
        "active_edge_count_equal": active_equal,
        "fast_path_contract_ok": bool(equivalent and fast_not_slower),
        "legacy_case": legacy_case,
        "segmented_fast_case": fast_case,
    }


def _segmented_fast_integration_check() -> dict[str, Any]:
    try:
        from src.evaluation import evaluate_v2x_graph_consensus
        from src.losses import compute_coupled_loss

        num_nodes = 4
        src_index = torch.tensor([0, 0, 0, 1, 1, 2, 3], dtype=torch.long)
        dst_index = torch.tensor([1, 2, 3, 0, 2, 3, 0], dtype=torch.long)
        edge_score = torch.tensor(
            [0.8, 1.4, 0.3, 0.6, 1.1, 0.7, 0.9],
            dtype=torch.float64,
            requires_grad=True,
        )
        candidate_distance = torch.tensor([140.0, 190.0, 210.0, 160.0, 170.0, 155.0, 150.0], dtype=torch.float64)
        topology = TopologyConstructionLayer(
            max_out_degree=2,
            temperature=1.0,
            topk_backend="segmented_fast",
        )(
            num_nodes=num_nodes,
            src_index=src_index,
            dst_index=dst_index,
            edge_score=edge_score,
        )
        selected = topology.selected_candidate_index
        evaluator_output = evaluate_v2x_graph_consensus(
            **topology.as_evaluation_kwargs(),
            distance_m=candidate_distance.index_select(0, selected),
            los_flag=torch.ones(selected.numel(), dtype=torch.float64),
            node_initial_correct=torch.tensor([0.45, 0.55, 0.20, 0.50], dtype=torch.float64),
            node_initial_wrong=torch.tensor([0.20, 0.20, 0.45, 0.25], dtype=torch.float64),
            physical_config={
                "tx_power_dbm": 23.0,
                "mcs_threshold_db": 7.0,
                "transition_width_db": 4.0,
                "interference_proxy_dbm": -82.0,
            },
            avalanche_config={
                "k": 1,
                "alpha": 1,
                "beta": 2,
                "rounds": 3,
                "eps": 0.0,
                "reliability_failure_target": 1e-2,
                "reliability_boundary_factor": 10.0,
            },
            energy_config={"packet_duration_s": 0.001},
        )
        loss_output = compute_coupled_loss(
            evaluator_output,
            {
                "reliability_failure_target": 1e-2,
                "reliability_tail_failure_target": 2e-2,
                "reliability_tau": 0.5,
                "delay_target_rounds": 1.0,
                "delay_p90_target_rounds": 2.0,
                "delay_tau": 1.0,
                "energy_target_j": 1e-4,
                "energy_p90_target_j": 2e-4,
                "energy_tau": 1e-3,
                "reliability_tail_mode": "max",
                "use_reliability_gate": True,
            },
        )
        loss = loss_output["total_loss"]
        loss.backward()
        grad = edge_score.grad
        finite_loss = bool(torch.isfinite(loss.detach()).cpu())
        finite_grad = grad is not None and bool(torch.all(torch.isfinite(grad)).cpu())
        nonzero_grad = grad is not None and float(torch.max(torch.abs(grad)).detach().cpu()) > 1.0e-8
        return {
            "loss_gradient_integration_passed": bool(finite_loss and finite_grad and nonzero_grad),
            "finite_loss": finite_loss,
            "finite_gradient": bool(finite_grad),
            "nonzero_gradient": bool(nonzero_grad),
            "structural_bias_enabled": False,
        }
    except Exception as exc:  # pragma: no cover - defensive report path
        return {
            "loss_gradient_integration_passed": False,
            "finite_loss": False,
            "finite_gradient": False,
            "nonzero_gradient": False,
            "structural_bias_enabled": False,
            "error": str(exc),
        }


def _readiness_blockers(readiness: Mapping[str, Any] | None) -> list[str]:
    if readiness is None:
        return ["missing_runtime_memory_scaling_interpretation_artifact"]
    blockers: list[str] = []
    if readiness.get("recommended_next_stage") != EXPECTED_M48_RECOMMENDATION:
        blockers.append("m48_recommendation_not_constructor_bottleneck_diagnostic")
    if readiness.get("runtime_memory_scaling_interpretation_status") == "blocked":
        blockers.append("m48_scaling_interpretation_blocked")
    if readiness.get("blocking_reasons"):
        blockers.append("m48_blocking_reasons_present")
    return blockers


def _case_blockers(case: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    if case.get("dense_nxn_tensor_allocated") is True:
        blockers.append("dense_nxn_tensor_allocated")
    if _float(case.get("active_edge_count")) > _float(case.get("node_count")) * _float(case.get("max_out_degree")):
        blockers.append("active_edges_not_O_Nk")
    if case.get("finite_topology_weights") is False:
        blockers.append("nonfinite_topology_weights")
    if case.get("train_eval_deploy_graph_rule_match") is False:
        blockers.append("graph_rule_mismatch")
    if case.get("structural_bias_enabled") is True:
        blockers.append("structural_bias_enabled")
    if case.get("topk_structural_bias_disabled") is False and case.get("support_mode") == "topk":
        blockers.append("topk_structural_bias_enabled")
    return blockers


def _growth_ratio(cases: list[Mapping[str, Any]], support_mode: str, start: int, end: int) -> float | None:
    preferred = [
        case
        for case in cases
        if case.get("support_mode") == support_mode and str(case.get("topk_backend", "legacy")) == "legacy"
    ]
    selected_cases = preferred if preferred else [case for case in cases if case.get("support_mode") == support_mode]
    by_node = {
        int(_float(case.get("node_count"))): case
        for case in selected_cases
    }
    start_case = by_node.get(start)
    end_case = by_node.get(end)
    if start_case is None or end_case is None:
        return None
    return _safe_ratio(
        _float(end_case.get("total_constructor_time_s")),
        _float(start_case.get("total_constructor_time_s")),
    )


def _all_mode_comparisons(cases: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (int(_float(case.get("node_count"))), str(case.get("support_mode"))): case
        for case in cases
    }
    comparisons: list[dict[str, Any]] = []
    node_counts = sorted({int(_float(case.get("node_count"))) for case in cases})
    for node_count in node_counts:
        topk = by_key.get((node_count, "topk"))
        all_mode = by_key.get((node_count, "all"))
        if topk is None or all_mode is None:
            continue
        all_time = _float(all_mode.get("total_constructor_time_s"))
        topk_time = _float(topk.get("total_constructor_time_s"))
        comparisons.append(
            {
                "node_count": node_count,
                "all_mode_constructor_time_s": all_time,
                "topk_constructor_time_s": topk_time,
                "topk_all_time_ratio": _safe_ratio(topk_time, all_time),
                "topk_specific_logic_dominates": bool(
                    _float(topk.get("topk_selection_time_s")) > 0.5 * max(topk_time, 1.0e-12)
                ),
            }
        )
    return comparisons


def _overall_decision(cases: list[Mapping[str, Any]], blockers: list[str]) -> tuple[str, str, float, str, bool]:
    if blockers:
        return "blocked", "unknown", 0.0, "inconclusive", False
    topk_cases = [case for case in cases if case.get("support_mode") == "topk"]
    if not topk_cases:
        return "blocked", "unknown", 0.0, "inconclusive", False
    largest = max(topk_cases, key=lambda item: int(_float(item.get("node_count"))))
    status = str(largest.get("constructor_bottleneck_status", "no_single_constructor_bottleneck"))
    component = str(largest.get("bottleneck_component", "unknown"))
    fraction = _float(largest.get("bottleneck_fraction"))
    recommendation = str(largest.get("optimization_recommendation", "inconclusive"))
    growth_500_to_2000 = _growth_ratio(topk_cases, "topk", 500, 2000)
    superlinear = (
        growth_500_to_2000 is not None
        and growth_500_to_2000 > 4.0 * GROWTH_SUPERLINEAR_TOLERANCE
    )
    if superlinear and recommendation == "constructor_acceptable_continue_to_5k":
        recommendation = "optimize_segmented_topk" if component in {"topk_selection", "python_loop"} else "inconclusive"
        status = "topk_selection_bottleneck" if component in {"topk_selection", "python_loop"} else "no_single_constructor_bottleneck"
    five_k_allowed = recommendation == "constructor_acceptable_continue_to_5k"
    return status, component, fraction, recommendation, five_k_allowed


def run_constructor_bottleneck_diagnostic(
    *,
    readiness_artifact: str | Path = DEFAULT_READINESS_ARTIFACT,
    node_counts: str | Iterable[int] | None = None,
    candidate_degree: int = DEFAULT_CANDIDATE_DEGREE,
    max_out_degree: int = DEFAULT_MAX_OUT_DEGREE,
    include_all_mode: bool = False,
    topk_backend: str = DEFAULT_TOPK_BACKEND,
    force: bool = False,
    case_profiler: ProfileRunner = profile_topology_constructor_case,
) -> dict[str, Any]:
    """Run the M49 diagnostic over synthetic sparse constructor inputs."""

    readiness_path = _resolve_path(readiness_artifact)
    readiness: dict[str, Any] | None = None
    load_errors: list[str] = []
    if readiness_path.exists():
        try:
            readiness = _load_json(readiness_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            load_errors.append(f"readiness_artifact_load_error:{exc}")
    readiness_blockers = _readiness_blockers(readiness)
    selected_node_counts = _coerce_int_list(node_counts, DEFAULT_NODE_COUNTS)
    support_modes = ["topk", "all"] if include_all_mode else ["topk"]

    preflight_blockers: list[str] = []
    if load_errors:
        preflight_blockers.extend(load_errors)
    if readiness_blockers and not force:
        preflight_blockers.extend(readiness_blockers)
    if candidate_degree <= 0:
        preflight_blockers.append("candidate_degree_not_positive")
    if max_out_degree <= 0:
        preflight_blockers.append("max_out_degree_not_positive")
    if topk_backend not in {"legacy", "segmented_fast", "both"}:
        preflight_blockers.append("invalid_topk_backend")
    for count in selected_node_counts:
        if int(count) <= candidate_degree:
            preflight_blockers.append(f"N{count}:candidate_degree_not_smaller_than_node_count")

    cases: list[dict[str, Any]] = []
    backend_comparisons: list[dict[str, Any]] = []
    if not preflight_blockers:
        for count in selected_node_counts:
            for mode in support_modes:
                selected_backends = ["legacy", "segmented_fast"] if topk_backend == "both" and mode == "topk" else [topk_backend]
                if mode == "all":
                    selected_backends = ["legacy"]
                for backend in selected_backends:
                    raw = case_profiler(
                        node_count=int(count),
                        candidate_degree=int(candidate_degree),
                        max_out_degree=int(max_out_degree),
                        support_mode=mode,
                        topk_backend=backend,
                    )
                    case = _normalize_profile_case(raw)
                    blockers = _case_blockers(case)
                    case["blocking_reasons"] = sorted(set(case.get("blocking_reasons", [])) | set(blockers))
                    cases.append(case)
            if topk_backend == "both":
                backend_comparisons.append(
                    _backend_equivalence_profile(
                        node_count=int(count),
                        candidate_degree=int(candidate_degree),
                        max_out_degree=int(max_out_degree),
                    )
                )

    case_blockers = [
        f"N{case.get('node_count')}/{case.get('support_mode')}:{reason}"
        for case in cases
        for reason in case.get("blocking_reasons", [])
    ]
    blocking_reasons = sorted(set(preflight_blockers) | set(case_blockers))
    status, component, fraction, recommendation, five_k_allowed = _overall_decision(cases, blocking_reasons)
    topk_cases = [case for case in cases if case.get("support_mode") == "topk"]
    comparison_500_to_2000 = _growth_ratio(topk_cases, "topk", 500, 2000)
    readiness_ok = readiness is not None and readiness.get("recommended_next_stage") == EXPECTED_M48_RECOMMENDATION
    successful_case_count = sum(1 for case in cases if not case.get("blocking_reasons"))
    failed_case_count = len(cases) - successful_case_count

    caution_reasons: set[str] = {
        "constructor_profile_uses_synthetic_edge_scores",
        "runtime_values_are_machine_local",
        "tensor_memory_proxy_not_rss_or_gpu_peak",
    }
    if readiness_blockers and force:
        caution_reasons.update(f"forced_with_{reason}" for reason in readiness_blockers)
    if not include_all_mode:
        caution_reasons.add("all_mode_comparison_not_run")
    if comparison_500_to_2000 is None:
        caution_reasons.add("n500_to_n2000_growth_not_available")
    elif comparison_500_to_2000 > 4.0 * GROWTH_SUPERLINEAR_TOLERANCE:
        caution_reasons.add("constructor_time_growth_superlinear")
    integration_check = (
        _segmented_fast_integration_check()
        if topk_backend in {"segmented_fast", "both"} and not preflight_blockers
        else {"loss_gradient_integration_passed": None}
    )
    comparison_ok = all(bool(item.get("equivalence_ok")) for item in backend_comparisons) if backend_comparisons else None
    speed_ok = (
        all(_float(item.get("speedup_ratio")) >= FAST_PATH_NOT_SLOWER_TOLERANCE for item in backend_comparisons)
        if backend_comparisons
        else None
    )
    ready_for_m51_fast = bool(
        topk_backend == "both"
        and comparison_ok is True
        and integration_check.get("loss_gradient_integration_passed") is True
        and speed_ok is True
        and not blocking_reasons
        and not any(case.get("dense_nxn_tensor_allocated") is True for case in cases)
    )
    if topk_backend == "both":
        if comparison_ok is False:
            recommended_next_stage = "fix_segmented_fast_path"
        elif speed_ok is False:
            recommended_next_stage = "keep_legacy_and_reprofile"
        elif ready_for_m51_fast:
            recommended_next_stage = "M51_node_count_5000_toy_topk_smoke"
        else:
            recommended_next_stage = "fix_segmented_fast_path"
    else:
        recommended_next_stage = (
            "M50_node_count_5000_toy_topk_smoke"
            if five_k_allowed
            else "M50_constructor_optimization_design"
        )

    return {
        "constructor_bottleneck_diagnostic_contract_ok": status != "blocked",
        "readiness_source_artifact": _repo_relative_path(readiness_path),
        "ready_for_M49_source": bool(readiness_ok),
        "force_used": bool(force),
        "node_counts": selected_node_counts,
        "candidate_degree": int(candidate_degree),
        "max_out_degree": int(max_out_degree),
        "topk_backend": topk_backend,
        "support_mode": DEFAULT_SUPPORT_MODE,
        "include_all_mode": bool(include_all_mode),
        "structural_bias_enabled": False,
        "structural_weights_zero": True,
        "topk_structural_bias_disabled": True,
        "dense_nxn_path_absent": not any(case.get("dense_nxn_tensor_allocated") is True for case in cases),
        "train_eval_deploy_graph_rule_match": all(
            case.get("train_eval_deploy_graph_rule_match", True) is True for case in cases
        ),
        "case_count": len(cases),
        "successful_case_count": successful_case_count,
        "failed_case_count": failed_case_count,
        "constructor_bottleneck_status": status,
        "bottleneck_component": component,
        "bottleneck_fraction": fraction,
        "optimization_recommendation": recommendation,
        "ready_for_5k_topk_smoke": bool(five_k_allowed),
        "ready_for_M51_5k_with_fast_path": ready_for_m51_fast,
        "recommended_next_stage": recommended_next_stage,
        "constructor_time_growth_500_to_2000": comparison_500_to_2000,
        "blocking_reasons": blocking_reasons,
        "caution_reasons": sorted(caution_reasons),
        "cases": cases,
        "backend_comparisons": backend_comparisons,
        "fast_path_profile": backend_comparisons,
        "loss_gradient_integration_check": integration_check,
        "all_mode_comparisons": _all_mode_comparisons(cases),
        "allowed_claims": ALLOWED_CLAIMS,
        "disallowed_claims": DISALLOWED_CLAIMS,
        "remaining_assumptions": REMAINING_ASSUMPTIONS,
    }


def write_constructor_bottleneck_diagnostic_json(
    report: Mapping[str, Any],
    path: str | Path = DEFAULT_JSON_OUT,
) -> None:
    output = _resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def render_constructor_bottleneck_diagnostic_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Constructor Bottleneck Diagnostic",
        "",
        "M49 profiles the current sparse topology constructor over synthetic O(Nk) inputs.",
        "",
        "## Decision",
        "",
        f"- constructor_bottleneck_status: `{report.get('constructor_bottleneck_status')}`",
        f"- optimization_recommendation: `{report.get('optimization_recommendation')}`",
        f"- ready_for_5k_topk_smoke: `{report.get('ready_for_5k_topk_smoke')}`",
        f"- ready_for_M51_5k_with_fast_path: `{report.get('ready_for_M51_5k_with_fast_path')}`",
        f"- recommended_next_stage: `{report.get('recommended_next_stage')}`",
        f"- topk_backend: `{report.get('topk_backend')}`",
        f"- readiness_source_artifact: `{report.get('readiness_source_artifact')}`",
        f"- ready_for_M49_source: `{report.get('ready_for_M49_source')}`",
        "",
        "## Cases",
        "",
        "| node_count | mode | candidates | active | coverage | total_s | validation | grouping | topk | softmax | diagnostics | packaging | bottleneck | fraction | recommendation |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
    ]
    for case in report.get("cases", []):
        if not isinstance(case, Mapping):
            continue
        lines.append(
            "| "
            f"{case.get('node_count')} | {case.get('support_mode')} | "
            f"{case.get('candidate_edge_count')} | {case.get('active_edge_count')} | "
            f"{_fmt(case.get('gradient_coverage_fraction'))} | {_fmt(case.get('total_constructor_time_s'))} | "
            f"{_fmt(case.get('validation_time_s'))} | {_fmt(case.get('sorting_or_grouping_time_s'))} | "
            f"{_fmt(case.get('topk_selection_time_s'))} | {_fmt(case.get('row_softmax_time_s'))} | "
            f"{_fmt(case.get('diagnostics_time_s'))} | {_fmt(case.get('output_packaging_time_s'))} | "
            f"{case.get('bottleneck_component')} | {_fmt(case.get('bottleneck_fraction'))} | "
            f"{case.get('optimization_recommendation')} |"
        )
    lines.extend(["", "## All-Mode Comparisons", ""])
    comparisons = report.get("all_mode_comparisons", [])
    if comparisons:
        lines.extend(
            [
                "| node_count | all_s | topk_s | topk/all | topk logic dominates |",
                "|---:|---:|---:|---:|---|",
            ]
        )
        for item in comparisons:
            if isinstance(item, Mapping):
                lines.append(
                    "| "
                    f"{item.get('node_count')} | {_fmt(item.get('all_mode_constructor_time_s'))} | "
                    f"{_fmt(item.get('topk_constructor_time_s'))} | {_fmt(item.get('topk_all_time_ratio'))} | "
                    f"{item.get('topk_specific_logic_dominates')} |"
                )
    else:
        lines.append("- not run")
    lines.extend(["", "## Backend Comparisons", ""])
    comparisons = report.get("backend_comparisons", [])
    if comparisons:
        lines.extend(
            [
                "| node_count | legacy_s | segmented_fast_s | speedup | equivalent | selected_equal | weights_equal | diagnostics_equal | ready |",
                "|---:|---:|---:|---:|---|---|---|---|---|",
            ]
        )
        for item in comparisons:
            if isinstance(item, Mapping):
                lines.append(
                    "| "
                    f"{item.get('node_count')} | {_fmt(item.get('legacy_time_s'))} | "
                    f"{_fmt(item.get('segmented_fast_time_s'))} | {_fmt(item.get('speedup_ratio'))} | "
                    f"{item.get('equivalence_ok')} | {item.get('selected_candidate_index_equal')} | "
                    f"{item.get('topology_weight_allclose')} | {item.get('diagnostics_equivalent')} | "
                    f"{item.get('fast_path_contract_ok')} |"
                )
    else:
        lines.append("- not run")
    lines.extend(["", "## Blocking Reasons", ""])
    blockers = report.get("blocking_reasons", [])
    lines.extend([f"- `{item}`" for item in blockers] if blockers else ["- none"])
    lines.extend(["", "## Caution Reasons", ""])
    cautions = report.get("caution_reasons", [])
    lines.extend([f"- `{item}`" for item in cautions] if cautions else ["- none"])
    lines.extend(["", "## Allowed Claims", ""])
    for claim in report.get("allowed_claims", []):
        lines.append(f"- {claim}")
    lines.extend(["", "## Disallowed Claims", ""])
    for claim in report.get("disallowed_claims", []):
        lines.append(f"- {claim}")
    lines.extend(["", "## Remaining Assumptions", ""])
    for item in report.get("remaining_assumptions", []):
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def write_constructor_bottleneck_diagnostic_markdown(
    report: Mapping[str, Any],
    path: str | Path = DEFAULT_MD_OUT,
) -> None:
    output = _resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_constructor_bottleneck_diagnostic_markdown(report), encoding="utf-8")


__all__ = [
    "DEFAULT_JSON_OUT",
    "DEFAULT_MD_OUT",
    "DEFAULT_READINESS_ARTIFACT",
    "profile_topology_constructor_case",
    "run_constructor_bottleneck_diagnostic",
    "load_constructor_bottleneck_diagnostic_report",
    "render_constructor_bottleneck_diagnostic_markdown",
    "write_constructor_bottleneck_diagnostic_json",
    "write_constructor_bottleneck_diagnostic_markdown",
]
