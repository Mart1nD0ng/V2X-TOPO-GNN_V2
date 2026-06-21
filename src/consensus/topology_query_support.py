from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .avalanche_closed_form import evaluate_avalanche_closed_form

PROBABILITY_TOLERANCE = 1e-6
QUERY_SUPPORT_BACKENDS = {"legacy", "fused_fast"}
DIAGNOSTICS_MODES = {"full", "lite", "off"}


@dataclass(frozen=True)
class QuerySupport:
    p_correct_query: torch.Tensor
    p_wrong_query: torch.Tensor
    p_link_response: torch.Tensor
    p_link_no_response: torch.Tensor
    p_neutral_query: torch.Tensor
    p_no_support_query: torch.Tensor
    p_no_response: torch.Tensor
    p_response: torch.Tensor
    effective_query_degree: torch.Tensor
    effective_unique_peer_degree: torch.Tensor
    normalized_query_weight: torch.Tensor
    src_index: torch.Tensor
    dst_index: torch.Tensor
    diagnostics: Mapping[str, object]


def _validate_query_support_backend(value: str) -> str:
    if value not in QUERY_SUPPORT_BACKENDS:
        allowed = ", ".join(sorted(QUERY_SUPPORT_BACKENDS))
        raise ValueError(f"query_support_backend must be one of: {allowed}")
    return value


def _validate_diagnostics_mode(value: str) -> str:
    if value not in DIAGNOSTICS_MODES:
        allowed = ", ".join(sorted(DIAGNOSTICS_MODES))
        raise ValueError(f"diagnostics_mode must be one of: {allowed}")
    return value


def _require_probability(name: str, value: torch.Tensor) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must use a floating-point dtype")
    if bool(torch.any(~torch.isfinite(value.detach())).cpu()):
        raise ValueError(f"{name} must contain only finite values")
    if bool(torch.any(value.detach() < -PROBABILITY_TOLERANCE).cpu()) or bool(
        torch.any(value.detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()
    ):
        raise ValueError(f"{name} must be in [0, 1]")


def _require_weight(name: str, value: torch.Tensor) -> None:
    if not torch.is_floating_point(value):
        raise ValueError(f"{name} must use a floating-point dtype")
    if bool(torch.any(~torch.isfinite(value.detach())).cpu()):
        raise ValueError(f"{name} must contain only finite values")
    if bool(torch.any(value.detach() < 0.0).cpu()):
        raise ValueError(f"{name} must be nonnegative")


def _bounded_probability(name: str, value: torch.Tensor) -> torch.Tensor:
    detached = value.detach()
    if bool(torch.any(detached < -PROBABILITY_TOLERANCE).cpu()) or bool(
        torch.any(detached > 1.0 + PROBABILITY_TOLERANCE).cpu()
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return torch.clamp(value, 0.0, 1.0)


def _as_probability_vector(name: str, value: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if value.ndim != 1 or value.numel() != num_nodes:
        raise ValueError(f"{name} must be a length num_nodes tensor")
    _require_probability(name, value)
    return value


def _edge_indices(
    *,
    num_nodes: int,
    edge_index: torch.Tensor | None,
    src_index: torch.Tensor | None,
    dst_index: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_index is not None:
        if src_index is not None or dst_index is not None:
            raise ValueError("provide either edge_index or src_index/dst_index, not both")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edge_count]")
        src = edge_index[0].to(device=device, dtype=torch.long)
        dst = edge_index[1].to(device=device, dtype=torch.long)
    else:
        if src_index is None or dst_index is None:
            raise ValueError("src_index and dst_index are required when edge_index is not provided")
        src = src_index.to(device=device, dtype=torch.long).reshape(-1)
        dst = dst_index.to(device=device, dtype=torch.long).reshape(-1)

    if src.numel() != dst.numel():
        raise ValueError("src_index and dst_index must have the same edge count")
    if src.numel() == 0:
        return src, dst
    if bool(torch.any(src < 0).cpu()) or bool(torch.any(src >= num_nodes).cpu()):
        raise ValueError("src_index contains node ids outside [0, num_nodes)")
    if bool(torch.any(dst < 0).cpu()) or bool(torch.any(dst >= num_nodes).cpu()):
        raise ValueError("dst_index contains node ids outside [0, num_nodes)")
    return src, dst


def _edge_values(
    *,
    edge_count: int,
    reference: torch.Tensor,
    query_weight: torch.Tensor | None,
    topology_weight: torch.Tensor | None,
    link_success: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_weight is not None and topology_weight is not None:
        raise ValueError("provide query_weight or topology_weight, not both")
    raw_weight = query_weight if query_weight is not None else topology_weight
    if raw_weight is None:
        weight = torch.ones(edge_count, dtype=reference.dtype, device=reference.device)
    else:
        weight = raw_weight.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    if weight.numel() != edge_count:
        raise ValueError("query/topology weight must have one value per edge")
    _require_weight("query_weight", weight)

    success = link_success.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    if success.numel() != edge_count:
        raise ValueError("link_success must have one value per edge")
    _require_probability("link_success", success)
    return weight, success


def _edge_pair_groups(
    src: torch.Tensor,
    dst: torch.Tensor,
    *,
    allow_multi_edges: bool,
) -> tuple[list[int], list[list[int]], int]:
    pair_to_group: dict[tuple[int, int], int] = {}
    group_sources: list[int] = []
    group_edge_indices: list[list[int]] = []
    duplicate_count = 0
    for edge_idx, (src_value, dst_value) in enumerate(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist())):
        key = (int(src_value), int(dst_value))
        existing = pair_to_group.get(key)
        if existing is None:
            pair_to_group[key] = len(group_sources)
            group_sources.append(int(src_value))
            group_edge_indices.append([edge_idx])
        else:
            duplicate_count += 1
            group_edge_indices[existing].append(edge_idx)
    if duplicate_count and not allow_multi_edges:
        raise ValueError("duplicate directed edges are disabled by default for peer query support")
    return group_sources, group_edge_indices, duplicate_count


def _sparse_duplicate_edge_count(src: torch.Tensor, dst: torch.Tensor, *, num_nodes: int) -> int:
    if src.numel() == 0:
        return 0
    pair_key = src * int(num_nodes) + dst
    unique_key = torch.unique(pair_key)
    return int(src.numel() - unique_key.numel())


def _unique_peer_diagnostics(
    *,
    num_nodes: int,
    reference: torch.Tensor,
    group_sources: list[int],
    group_edge_indices: list[list[int]],
    weight: torch.Tensor,
    normalized: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    base = reference.new_zeros((num_nodes,))
    if not group_sources:
        return base, base, base
    unique_src = torch.as_tensor(group_sources, dtype=torch.long, device=reference.device)
    one_per_peer = torch.ones(len(group_sources), dtype=reference.dtype, device=reference.device)
    unique_out_degree = base.index_add(0, unique_src, one_per_peer)

    unique_weight_values: list[torch.Tensor] = []
    unique_query_values: list[torch.Tensor] = []
    for indices in group_edge_indices:
        index = torch.as_tensor(indices, dtype=torch.long, device=reference.device)
        unique_weight_values.append(weight.index_select(0, index).sum())
        unique_query_values.append(normalized.index_select(0, index).sum())
    unique_weight = torch.stack(unique_weight_values)
    unique_query = torch.stack(unique_query_values)
    positive_unique_out_degree = base.index_add(0, unique_src, (unique_weight > 0.0).to(dtype=reference.dtype))
    unique_square_sum = base.index_add(0, unique_src, unique_query * unique_query)
    effective_unique_degree = torch.where(unique_square_sum > 0.0, 1.0 / unique_square_sum, unique_square_sum)
    return unique_out_degree, positive_unique_out_degree, effective_unique_degree


def _unique_peer_diagnostics_fused(
    *,
    num_nodes: int,
    reference: torch.Tensor,
    src: torch.Tensor,
    dst: torch.Tensor,
    weight: torch.Tensor,
    normalized: torch.Tensor,
    outgoing_count: torch.Tensor,
    positive_outgoing_count: torch.Tensor,
    effective_degree: torch.Tensor,
    allow_multi_edges: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if src.numel() == 0:
        base = reference.new_zeros((num_nodes,))
        return base, base, base
    if not allow_multi_edges:
        return outgoing_count, positive_outgoing_count, effective_degree

    pair_key = src * int(num_nodes) + dst
    unique_key, inverse = torch.unique(pair_key, sorted=False, return_inverse=True)
    unique_src = torch.div(unique_key, int(num_nodes), rounding_mode="floor").to(dtype=torch.long)
    base = reference.new_zeros((num_nodes,))
    unique_count = int(unique_key.numel())
    one_per_peer = torch.ones(unique_count, dtype=reference.dtype, device=reference.device)
    unique_out_degree = base.index_add(0, unique_src, one_per_peer)
    unique_weight = reference.new_zeros((unique_count,)).index_add(0, inverse, weight)
    unique_query = reference.new_zeros((unique_count,)).index_add(0, inverse, normalized)
    positive_unique_out_degree = base.index_add(0, unique_src, (unique_weight > 0.0).to(dtype=reference.dtype))
    unique_square_sum = base.index_add(0, unique_src, unique_query * unique_query)
    effective_unique_degree = torch.where(unique_square_sum > 0.0, 1.0 / unique_square_sum, unique_square_sum)
    return unique_out_degree, positive_unique_out_degree, effective_unique_degree


def compute_topology_query_support(
    *,
    num_nodes: int,
    link_success: torch.Tensor,
    node_correct_preference: torch.Tensor,
    node_wrong_preference: torch.Tensor,
    edge_index: torch.Tensor | None = None,
    src_index: torch.Tensor | None = None,
    dst_index: torch.Tensor | None = None,
    query_weight: torch.Tensor | None = None,
    topology_weight: torch.Tensor | None = None,
    allow_self_loops: bool = False,
    allow_multi_edges: bool = False,
    query_support_backend: str = "legacy",
    diagnostics_mode: str = "full",
) -> QuerySupport:
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    query_support_backend = _validate_query_support_backend(query_support_backend)
    diagnostics_mode = _validate_diagnostics_mode(diagnostics_mode)
    reference = node_correct_preference
    if not isinstance(reference, torch.Tensor):
        raise TypeError("node_correct_preference must be a torch.Tensor")
    if not isinstance(node_wrong_preference, torch.Tensor):
        raise TypeError("node_wrong_preference must be a torch.Tensor")
    if not isinstance(link_success, torch.Tensor):
        raise TypeError("link_success must be a torch.Tensor")

    correct_pref = _as_probability_vector(
        "node_correct_preference",
        node_correct_preference.to(dtype=reference.dtype, device=reference.device),
        num_nodes,
    )
    wrong_pref = _as_probability_vector(
        "node_wrong_preference",
        node_wrong_preference.to(dtype=reference.dtype, device=reference.device),
        num_nodes,
    )
    if bool(torch.any((correct_pref + wrong_pref).detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()):
        raise ValueError("node_correct_preference + node_wrong_preference must be <= 1")

    src, dst = _edge_indices(
        num_nodes=num_nodes,
        edge_index=edge_index,
        src_index=src_index,
        dst_index=dst_index,
        device=reference.device,
    )
    self_loop_mask = src == dst
    self_loop_count = self_loop_mask.to(dtype=reference.dtype).sum()
    if not allow_self_loops and bool(torch.any(self_loop_mask).cpu()):
        raise ValueError("self-loops are disabled by default for peer query support")
    group_sources: list[int] = []
    group_edge_indices: list[list[int]] = []
    if query_support_backend == "legacy" and diagnostics_mode == "full":
        group_sources, group_edge_indices, duplicate_edge_count = _edge_pair_groups(
            src,
            dst,
            allow_multi_edges=allow_multi_edges,
        )
    else:
        duplicate_edge_count = _sparse_duplicate_edge_count(src, dst, num_nodes=num_nodes)
        if duplicate_edge_count and not allow_multi_edges:
            raise ValueError("duplicate directed edges are disabled by default for peer query support")
    edge_count = int(src.numel())
    weight, success = _edge_values(
        edge_count=edge_count,
        reference=reference,
        query_weight=query_weight,
        topology_weight=topology_weight,
        link_success=link_success,
    )

    base = reference.new_zeros((num_nodes,))
    outgoing_count = base.index_add(0, src, torch.ones(edge_count, dtype=reference.dtype, device=reference.device))
    positive_outgoing_count = base.index_add(0, src, (weight > 0.0).to(dtype=reference.dtype))
    row_weight_sum = base.index_add(0, src, weight)
    row_weight = row_weight_sum[src] if edge_count else weight
    positive_row = row_weight > 0.0
    normalized = torch.where(positive_row, weight / torch.clamp(row_weight, min=torch.finfo(reference.dtype).tiny), weight.new_zeros(()))

    response_terms = normalized * success
    correct_terms = response_terms * correct_pref[dst]
    wrong_terms = response_terms * wrong_pref[dst]
    p_link_response = _bounded_probability("p_link_response", base.index_add(0, src, response_terms))
    p_correct_query = base.index_add(0, src, correct_terms)
    p_wrong_query = base.index_add(0, src, wrong_terms)
    support_sum = p_correct_query + p_wrong_query
    if bool(torch.any(support_sum.detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()):
        raise ValueError("p_correct_query + p_wrong_query exceeds one")
    p_correct_query = _bounded_probability("p_correct_query", p_correct_query)
    p_wrong_query = _bounded_probability("p_wrong_query", p_wrong_query)
    support_sum = p_correct_query + p_wrong_query
    p_link_no_response = _bounded_probability("p_link_no_response", 1.0 - p_link_response)
    p_neutral_query = _bounded_probability("p_neutral_query", p_link_response - support_sum)
    p_no_support_query = _bounded_probability("p_no_support_query", 1.0 - support_sum)

    q_square_sum = base.index_add(0, src, normalized * normalized)
    effective_degree = torch.where(q_square_sum > 0.0, 1.0 / q_square_sum, q_square_sum)
    if query_support_backend == "legacy" and diagnostics_mode == "full":
        unique_out_degree, positive_unique_out_degree, effective_unique_degree = _unique_peer_diagnostics(
            num_nodes=num_nodes,
            reference=reference,
            group_sources=group_sources,
            group_edge_indices=group_edge_indices,
            weight=weight,
            normalized=normalized,
        )
        unique_peer_diagnostics_available = True
        effective_unique_peer_degree_is_approximate = False
    else:
        unique_out_degree, positive_unique_out_degree, effective_unique_degree = _unique_peer_diagnostics_fused(
            num_nodes=num_nodes,
            reference=reference,
            src=src,
            dst=dst,
            weight=weight,
            normalized=normalized,
            outgoing_count=outgoing_count,
            positive_outgoing_count=positive_outgoing_count,
            effective_degree=effective_degree,
            allow_multi_edges=allow_multi_edges,
        )
        unique_peer_diagnostics_available = diagnostics_mode != "off"
        effective_unique_peer_degree_is_approximate = False
    isolated = outgoing_count == 0.0
    row_weight_zero = row_weight_sum <= 0.0
    empty = reference.new_tensor(0.0)
    diagnostics = {
        "out_degree": outgoing_count,
        "positive_out_degree": positive_outgoing_count,
        "row_weight_sum": row_weight_sum,
        "self_loop_count": self_loop_count,
        "duplicate_edge_count": reference.new_tensor(float(duplicate_edge_count)),
        "query_support_backend": query_support_backend,
        "diagnostics_mode": diagnostics_mode,
        "full_query_support_diagnostics": diagnostics_mode == "full",
        "lite_query_support_diagnostics": diagnostics_mode == "lite",
        "query_support_diagnostics_off": diagnostics_mode == "off",
        "unique_peer_diagnostics_available": unique_peer_diagnostics_available,
        "effective_unique_peer_degree_is_approximate": effective_unique_peer_degree_is_approximate,
        "unique_out_degree": unique_out_degree,
        "positive_unique_out_degree": positive_unique_out_degree,
        "mean_response_probability": p_link_response.mean() if num_nodes else empty,
        "min_response_probability": p_link_response.min() if num_nodes else empty,
        "max_response_probability": p_link_response.max() if num_nodes else empty,
        "isolated_node_count": isolated.to(dtype=reference.dtype).sum(),
        "row_weight_zero_count": row_weight_zero.to(dtype=reference.dtype).sum(),
        "mean_effective_query_degree": effective_degree.mean() if num_nodes else empty,
        "mean_effective_unique_peer_degree": effective_unique_degree.mean() if num_nodes else empty,
    }
    if diagnostics_mode == "full":
        diagnostics.update(
            {
                "min_unique_out_degree": unique_out_degree.min() if num_nodes else empty,
                "p10_unique_out_degree": torch.quantile(unique_out_degree, 0.10) if num_nodes else empty,
                "mean_unique_out_degree": unique_out_degree.mean() if num_nodes else empty,
                "min_effective_query_degree": effective_degree.min() if num_nodes else empty,
                "p10_effective_query_degree": torch.quantile(effective_degree, 0.10) if num_nodes else empty,
                "min_effective_unique_peer_degree": effective_unique_degree.min() if num_nodes else empty,
                "p10_effective_unique_peer_degree": torch.quantile(effective_unique_degree, 0.10) if num_nodes else empty,
            }
        )
    else:
        diagnostics.update(
            {
                "min_unique_out_degree": empty,
                "p10_unique_out_degree": empty,
                "mean_unique_out_degree": unique_out_degree.mean() if num_nodes and diagnostics_mode == "lite" else empty,
                "min_effective_query_degree": empty,
                "p10_effective_query_degree": empty,
                "min_effective_unique_peer_degree": empty,
                "p10_effective_unique_peer_degree": empty,
            }
        )
    return QuerySupport(
        p_correct_query=p_correct_query,
        p_wrong_query=p_wrong_query,
        p_link_response=p_link_response,
        p_link_no_response=p_link_no_response,
        p_neutral_query=p_neutral_query,
        p_no_support_query=p_no_support_query,
        p_no_response=p_no_support_query,
        p_response=p_link_response,
        effective_query_degree=effective_degree,
        effective_unique_peer_degree=effective_unique_degree,
        normalized_query_weight=normalized,
        src_index=src,
        dst_index=dst,
        diagnostics=diagnostics,
    )


def evaluate_topology_avalanche_static(
    *,
    num_nodes: int,
    link_success: torch.Tensor,
    node_correct_preference: torch.Tensor,
    node_wrong_preference: torch.Tensor,
    k: int,
    alpha: int,
    beta: int,
    rounds: int,
    edge_index: torch.Tensor | None = None,
    src_index: torch.Tensor | None = None,
    dst_index: torch.Tensor | None = None,
    query_weight: torch.Tensor | None = None,
    topology_weight: torch.Tensor | None = None,
    allow_self_loops: bool = False,
    allow_multi_edges: bool = False,
    query_support_backend: str = "legacy",
    diagnostics_mode: str = "full",
    initial_correct_preference: torch.Tensor | float = 1.0,
    eps: float = 1e-6,
    temperature: float = 1.0,
) -> dict[str, object]:
    support = compute_topology_query_support(
        num_nodes=num_nodes,
        edge_index=edge_index,
        src_index=src_index,
        dst_index=dst_index,
        query_weight=query_weight,
        topology_weight=topology_weight,
        link_success=link_success,
        node_correct_preference=node_correct_preference,
        node_wrong_preference=node_wrong_preference,
        allow_self_loops=allow_self_loops,
        allow_multi_edges=allow_multi_edges,
        query_support_backend=query_support_backend,
        diagnostics_mode=diagnostics_mode,
    )
    avalanche = evaluate_avalanche_closed_form(
        support.p_correct_query,
        support.p_wrong_query,
        k=k,
        alpha=alpha,
        beta=beta,
        rounds=rounds,
        initial_correct_preference=initial_correct_preference,
        eps=eps,
        temperature=temperature,
    )
    correct = avalanche["p_correct_decision"]
    wrong = avalanche["p_wrong_decision"]
    undecided = avalanche["p_undecided"]
    expected = avalanche["expected_rounds"]
    out_degree = support.diagnostics["out_degree"]
    positive_out_degree = support.diagnostics["positive_out_degree"]
    unique_out_degree = support.diagnostics["unique_out_degree"]
    positive_unique_out_degree = support.diagnostics["positive_unique_out_degree"]
    effective_degree = support.effective_query_degree
    effective_unique_degree = support.effective_unique_peer_degree
    metric_dtype = correct.dtype
    graph_metrics = {
        "node_mean_correct_decision": correct.mean() if num_nodes else correct.new_tensor(0.0),
        "node_min_correct_decision": correct.min() if num_nodes else correct.new_tensor(0.0),
        "node_10pct_correct_decision": torch.quantile(correct, 0.10) if num_nodes else correct.new_tensor(0.0),
        "node_mean_wrong_decision": wrong.mean() if num_nodes else wrong.new_tensor(0.0),
        "undecided_mean": undecided.mean() if num_nodes else undecided.new_tensor(0.0),
        "expected_rounds_mean": expected.mean() if num_nodes else expected.new_tensor(0.0),
        "isolated_node_count": support.diagnostics["isolated_node_count"],
        "out_degree_lt_k_count": (out_degree < float(k)).to(dtype=metric_dtype).sum(),
        "positive_out_degree_lt_k_count": (positive_out_degree < float(k)).to(dtype=metric_dtype).sum(),
        "unique_out_degree_lt_k_count": (unique_out_degree < float(k)).to(dtype=metric_dtype).sum(),
        "positive_unique_out_degree_lt_k_count": (positive_unique_out_degree < float(k)).to(dtype=metric_dtype).sum(),
        "effective_degree_lt_k_count": (effective_degree < float(k)).to(dtype=metric_dtype).sum(),
        "effective_unique_peer_degree_lt_k_count": (effective_unique_degree < float(k)).to(dtype=metric_dtype).sum(),
        "min_effective_query_degree": support.diagnostics["min_effective_query_degree"],
        "p10_effective_query_degree": support.diagnostics["p10_effective_query_degree"],
        "min_effective_unique_peer_degree": support.diagnostics["min_effective_unique_peer_degree"],
        "p10_effective_unique_peer_degree": support.diagnostics["p10_effective_unique_peer_degree"],
    }
    return {
        "query_support": support,
        "avalanche": avalanche,
        "node_p_correct_decision": correct,
        "node_p_wrong_decision": wrong,
        "node_p_undecided": undecided,
        "node_expected_rounds": expected,
        "graph_metrics": graph_metrics,
    }
