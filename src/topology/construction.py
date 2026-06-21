from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

VALID_TOPK_BACKENDS = {"legacy", "segmented_fast"}
VALID_SUPPORT_SMOOTHING_MODES = {"none", "deterministic_topk_halo"}
VALID_SUPPORT_SMOOTHING_STAGES = {"hard_topk", "expanded_sparse_support"}
# P0 remediation: gradient pathway for the topology weights.
#   "selected_row_softmax" -> legacy backward; gradient flows only through the
#       row-softmax over the hard-selected support. Excluded candidates receive
#       exactly zero gradient (the R10/R11/R12 "constructor Jacobian bottleneck").
#   "straight_through_full_candidate" -> Mode A straight-through estimator. The
#       forward value is byte-identical to the hard top-k row-softmax, but the
#       backward gradient flows through a soft row-softmax taken over the full
#       feasible candidate set, so excluded candidates get a non-zero gradient
#       and membership can be learned. This honours the AGENTS.md invariant
#       "A straight-through surrogate may be used only for backward gradients".
VALID_GRADIENT_MODES = {"selected_row_softmax", "straight_through_full_candidate"}


@dataclass(frozen=True)
class ConstructedTopology:
    num_nodes: int
    src_index: torch.Tensor
    dst_index: torch.Tensor
    topology_weight: torch.Tensor
    selected_candidate_index: torch.Tensor
    selected_edge_mask: torch.Tensor
    active_edge_mask: torch.Tensor
    diagnostics: Mapping[str, torch.Tensor]

    def as_evaluation_kwargs(self) -> dict[str, int | torch.Tensor]:
        return {
            "num_nodes": self.num_nodes,
            "src_index": self.src_index,
            "dst_index": self.dst_index,
            "topology_weight": self.topology_weight,
        }


class TopologyConstructionLayer(torch.nn.Module):
    """Deterministic sparse candidate-score to query-weight constructor."""

    def __init__(
        self,
        *,
        max_out_degree: int | None = 4,
        row_softmax_temperature: float = 1.0,
        temperature: float | None = None,
        support_mode: str = "topk",
        allow_self_loops: bool = False,
        allow_multi_edges: bool = False,
        near_miss_tolerance: float = 0.05,
        topk_backend: str = "legacy",
        support_smoothing_mode: str = "none",
        support_smoothing_extra_per_row: int = 0,
        support_smoothing_temperature: float = 1.0,
        support_smoothing_stage: str = "hard_topk",
        gradient_mode: str = "selected_row_softmax",
        straight_through_temperature: float | None = None,
    ) -> None:
        super().__init__()
        if max_out_degree is not None and max_out_degree < 0:
            raise ValueError("max_out_degree must be nonnegative or None")
        if str(gradient_mode) not in VALID_GRADIENT_MODES:
            raise ValueError("gradient_mode must be 'selected_row_softmax' or 'straight_through_full_candidate'")
        if straight_through_temperature is not None and float(straight_through_temperature) <= 0.0:
            raise ValueError("straight_through_temperature must be positive when provided")
        temperature_value = _resolve_row_softmax_temperature(
            row_softmax_temperature=row_softmax_temperature,
            temperature=temperature,
            default=1.0,
        )
        if support_mode not in {"topk", "all"}:
            raise ValueError("support_mode must be 'topk' or 'all'")
        if near_miss_tolerance < 0.0:
            raise ValueError("near_miss_tolerance must be nonnegative")
        if topk_backend not in VALID_TOPK_BACKENDS:
            raise ValueError("topk_backend must be 'legacy' or 'segmented_fast'")
        smoothing_mode = str(support_smoothing_mode)
        if smoothing_mode not in VALID_SUPPORT_SMOOTHING_MODES:
            raise ValueError("support_smoothing_mode must be 'none' or 'deterministic_topk_halo'")
        if int(support_smoothing_extra_per_row) != support_smoothing_extra_per_row or support_smoothing_extra_per_row < 0:
            raise ValueError("support_smoothing_extra_per_row must be a nonnegative integer")
        smoothing_temperature = float(support_smoothing_temperature)
        if smoothing_temperature <= 0.0:
            raise ValueError("support_smoothing_temperature must be positive")
        smoothing_stage = str(support_smoothing_stage)
        if smoothing_stage not in VALID_SUPPORT_SMOOTHING_STAGES:
            raise ValueError("support_smoothing_stage must be 'hard_topk' or 'expanded_sparse_support'")
        self.max_out_degree = max_out_degree
        self.row_softmax_temperature = float(temperature_value)
        self.temperature = self.row_softmax_temperature
        self.support_mode = support_mode
        self.allow_self_loops = bool(allow_self_loops)
        self.allow_multi_edges = bool(allow_multi_edges)
        self.near_miss_tolerance = float(near_miss_tolerance)
        self.topk_backend = topk_backend
        self.support_smoothing_mode = smoothing_mode
        self.support_smoothing_extra_per_row = int(support_smoothing_extra_per_row)
        self.support_smoothing_temperature = smoothing_temperature
        self.support_smoothing_stage = smoothing_stage
        self.gradient_mode = str(gradient_mode)
        self.straight_through_temperature = (
            None if straight_through_temperature is None else float(straight_through_temperature)
        )

    def forward(
        self,
        *,
        num_nodes: int,
        src_index: torch.Tensor,
        dst_index: torch.Tensor,
        edge_score: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        per_node_budget: torch.Tensor | None = None,
        max_out_degree: int | None = None,
        row_softmax_temperature: float | None = None,
        temperature: float | None = None,
        support_mode: str | None = None,
        allow_self_loops: bool | None = None,
        allow_multi_edges: bool | None = None,
        near_miss_tolerance: float | None = None,
        topk_backend: str | None = None,
        support_smoothing_mode: str | None = None,
        support_smoothing_extra_per_row: int | None = None,
        support_smoothing_temperature: float | None = None,
        support_smoothing_stage: str | None = None,
        gradient_mode: str | None = None,
        straight_through_temperature: float | None = None,
    ) -> ConstructedTopology:
        if num_nodes < 0:
            raise ValueError("num_nodes must be nonnegative")
        if not isinstance(edge_score, torch.Tensor):
            raise TypeError("edge_score must be a torch.Tensor")
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

        self_loop_enabled = self.allow_self_loops if allow_self_loops is None else bool(allow_self_loops)
        multi_edge_enabled = self.allow_multi_edges if allow_multi_edges is None else bool(allow_multi_edges)
        self_loop_count = (src == dst).to(dtype=dtype).sum()
        if not self_loop_enabled and bool(torch.any(src == dst).cpu()):
            raise ValueError("self-loops are disabled by default for topology construction")
        duplicate_edge_count = score.new_tensor(0.0)
        if edge_count:
            pair_key = src * max(num_nodes, 1) + dst
            unique_pair_count = int(torch.unique(pair_key).numel())
            duplicate_edge_count = score.new_tensor(float(edge_count - unique_pair_count))
            if not multi_edge_enabled and edge_count != unique_pair_count:
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
            per_node_budget=per_node_budget,
            max_out_degree=max_out_degree if max_out_degree is not None else self.max_out_degree,
            fallback=edge_count,
            device=device,
        )
        tau_value = _resolve_row_softmax_temperature(
            row_softmax_temperature=row_softmax_temperature,
            temperature=temperature,
            default=self.row_softmax_temperature,
        )
        mode = self.support_mode if support_mode is None else str(support_mode)
        if mode not in {"topk", "all"}:
            raise ValueError("support_mode must be 'topk' or 'all'")
        near_miss_value = self.near_miss_tolerance if near_miss_tolerance is None else float(near_miss_tolerance)
        if near_miss_value < 0.0:
            raise ValueError("near_miss_tolerance must be nonnegative")
        near_miss_threshold = score.new_tensor(near_miss_value)
        backend = self.topk_backend if topk_backend is None else str(topk_backend)
        if backend not in VALID_TOPK_BACKENDS:
            raise ValueError("topk_backend must be 'legacy' or 'segmented_fast'")
        smoothing_mode = self.support_smoothing_mode if support_smoothing_mode is None else str(support_smoothing_mode)
        if smoothing_mode not in VALID_SUPPORT_SMOOTHING_MODES:
            raise ValueError("support_smoothing_mode must be 'none' or 'deterministic_topk_halo'")
        smoothing_extra_value = (
            self.support_smoothing_extra_per_row
            if support_smoothing_extra_per_row is None
            else support_smoothing_extra_per_row
        )
        if int(smoothing_extra_value) != smoothing_extra_value or smoothing_extra_value < 0:
            raise ValueError("support_smoothing_extra_per_row must be a nonnegative integer")
        smoothing_extra = int(smoothing_extra_value)
        smoothing_temperature = (
            self.support_smoothing_temperature
            if support_smoothing_temperature is None
            else float(support_smoothing_temperature)
        )
        if smoothing_temperature <= 0.0:
            raise ValueError("support_smoothing_temperature must be positive")
        smoothing_stage = self.support_smoothing_stage if support_smoothing_stage is None else str(support_smoothing_stage)
        if smoothing_stage not in VALID_SUPPORT_SMOOTHING_STAGES:
            raise ValueError("support_smoothing_stage must be 'hard_topk' or 'expanded_sparse_support'")
        smoothing_active = bool(
            mode == "topk"
            and smoothing_mode == "deterministic_topk_halo"
            and smoothing_stage == "expanded_sparse_support"
            and smoothing_extra > 0
        )
        effective_smoothing_extra = smoothing_extra if smoothing_active else 0
        effective_tau = score.new_tensor(float(smoothing_temperature if smoothing_active else tau_value))
        resolved_gradient_mode = self.gradient_mode if gradient_mode is None else str(gradient_mode)
        if resolved_gradient_mode not in VALID_GRADIENT_MODES:
            raise ValueError("gradient_mode must be 'selected_row_softmax' or 'straight_through_full_candidate'")
        # Straight-through is only meaningful when a hard top-k cap removes candidates.
        straight_through_active = bool(
            resolved_gradient_mode == "straight_through_full_candidate" and mode == "topk"
        )
        st_temperature_value = (
            self.straight_through_temperature
            if straight_through_temperature is None
            else float(straight_through_temperature)
        )
        if st_temperature_value is None:
            st_temperature_value = float(effective_tau.detach().cpu().item())
        if st_temperature_value <= 0.0:
            raise ValueError("straight_through_temperature must be positive")
        st_tau = score.new_tensor(float(st_temperature_value))

        feasible_count = score.new_zeros((num_nodes,))
        if edge_count:
            feasible_count = feasible_count.index_add(0, src, feasible.to(dtype=dtype))
        cap_hit = (feasible_count > budgets.to(dtype=dtype)) if mode == "topk" else torch.zeros_like(feasible_count, dtype=torch.bool)
        selected_mask = torch.zeros(edge_count, dtype=torch.bool, device=device)
        output_index_parts: list[torch.Tensor] = []
        margin_parts: list[torch.Tensor] = []
        near_miss_candidate_count = score.new_tensor(0.0)
        support_smoothing_halo_candidate_count = score.new_tensor(0.0)
        support_smoothing_base_selected_count = score.new_tensor(0.0)

        feasible_indices = torch.nonzero(feasible, as_tuple=False).reshape(-1)
        if feasible_indices.numel():
            feasible_src = src.index_select(0, feasible_indices)
            order = torch.argsort(feasible_src, stable=True)
            feasible_indices = feasible_indices.index_select(0, order)
            feasible_src = feasible_src.index_select(0, order)
            if mode == "all":
                output_index_parts.append(feasible_indices)
            else:
                unique_src, counts = torch.unique_consecutive(feasible_src, return_counts=True)
                row_starts = torch.cumsum(counts, dim=0) - counts
                if backend == "legacy":
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
                        near_miss_threshold=near_miss_threshold,
                        support_smoothing_extra_per_row=effective_smoothing_extra,
                    )
                else:
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
                        near_miss_threshold=near_miss_threshold,
                        support_smoothing_extra_per_row=effective_smoothing_extra,
                    )

        if output_index_parts:
            selected_candidate_index = torch.cat(output_index_parts)
            selected_mask[selected_candidate_index] = True
            output_src = src.index_select(0, selected_candidate_index)
            output_dst = dst.index_select(0, selected_candidate_index)
            hard_weight = _row_softmax(
                num_nodes=num_nodes,
                src_index=src,
                edge_score=score,
                selected_candidate_index=selected_candidate_index,
                tau=effective_tau,
            )
            if straight_through_active:
                # Mode A: forward value == hard_weight (deployment-faithful), but
                # the backward gradient flows through a soft row-softmax over the
                # whole feasible candidate set, so excluded candidates receive a
                # non-zero gradient through the shared row denominator.
                soft_weight = _row_softmax_full_candidate_at_selected(
                    num_nodes=num_nodes,
                    src_index=src,
                    edge_score=score,
                    feasible_indices=feasible_indices,
                    selected_candidate_index=selected_candidate_index,
                    tau=st_tau,
                )
                topology_weight = soft_weight + (hard_weight - soft_weight).detach()
            else:
                topology_weight = hard_weight
            effective_tau_value = float(effective_tau.detach().cpu().item())
            if abs(effective_tau_value - 1.0) <= 1.0e-12:
                temperature_one_weight = hard_weight
            else:
                temperature_one_weight = _row_softmax(
                    num_nodes=num_nodes,
                    src_index=src,
                    edge_score=score,
                    selected_candidate_index=selected_candidate_index,
                    tau=score.new_tensor(1.0),
                )
        else:
            output_src = src.new_empty((0,))
            output_dst = dst.new_empty((0,))
            hard_weight = score.new_empty((0,))
            topology_weight = hard_weight
            temperature_one_weight = hard_weight
            selected_candidate_index = src.new_empty((0,))
        topk_boundary_margin = torch.cat(margin_parts) if margin_parts else score.new_empty((0,))
        row_entropy_mean = _row_entropy_mean(
            num_nodes=num_nodes,
            src_index=output_src,
            topology_weight=hard_weight,
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
            row_softmax_temperature=score.new_tensor(float(tau_value)),
            support_smoothing_mode_code=score.new_tensor(1.0 if smoothing_mode == "deterministic_topk_halo" else 0.0),
            support_smoothing_stage_code=score.new_tensor(1.0 if smoothing_stage == "expanded_sparse_support" else 0.0),
            support_smoothing_extra_per_row=score.new_tensor(float(smoothing_extra)),
            support_smoothing_effective_extra_per_row=score.new_tensor(float(effective_smoothing_extra)),
            support_smoothing_temperature=score.new_tensor(float(smoothing_temperature)),
            support_smoothing_active=score.new_tensor(1.0 if smoothing_active else 0.0),
            support_smoothing_halo_candidate_count=support_smoothing_halo_candidate_count,
            support_smoothing_base_selected_count=support_smoothing_base_selected_count,
            row_entropy_mean=row_entropy_mean,
            row_entropy_delta_vs_temperature_1=row_entropy_mean - row_entropy_temperature_one,
            reference=score,
        )
        diagnostics = {
            **diagnostics,
            "gradient_mode_straight_through": score.new_tensor(1.0 if straight_through_active else 0.0),
            "straight_through_temperature": st_tau.detach() if straight_through_active else score.new_tensor(0.0),
        }
        return ConstructedTopology(
            num_nodes=num_nodes,
            src_index=output_src,
            dst_index=output_dst,
            topology_weight=topology_weight,
            selected_candidate_index=selected_candidate_index,
            selected_edge_mask=selected_mask,
            active_edge_mask=selected_mask,
            diagnostics=diagnostics,
        )


def _resolve_row_softmax_temperature(
    *,
    row_softmax_temperature: float | None,
    temperature: float | None,
    default: float,
) -> float:
    if row_softmax_temperature is not None and temperature is not None:
        row_value = float(row_softmax_temperature)
        alias_value = float(temperature)
        if abs(row_value - float(default)) <= 1.0e-12:
            value = alias_value
        elif abs(row_value - alias_value) > 1.0e-12:
            raise ValueError("row_softmax_temperature and temperature aliases disagree")
        else:
            value = row_value
    elif row_softmax_temperature is not None:
        value = float(row_softmax_temperature)
    elif temperature is not None:
        value = float(temperature)
    else:
        value = float(default)
    if value <= 0.0:
        raise ValueError("row_softmax_temperature must be positive")
    return value


def _select_topk_legacy(
    *,
    feasible_indices: torch.Tensor,
    unique_src: torch.Tensor,
    counts: torch.Tensor,
    row_starts: torch.Tensor,
    budgets: torch.Tensor,
    score: torch.Tensor,
    near_miss_threshold: torch.Tensor,
    support_smoothing_extra_per_row: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = score.dtype
    output_index_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    near_miss_candidate_count = score.new_tensor(0.0)
    halo_candidate_count = score.new_tensor(0.0)
    base_selected_count = score.new_tensor(0.0)
    for row_pos in range(int(unique_src.numel())):
        source = int(unique_src[row_pos].item())
        row_start = int(row_starts[row_pos].item())
        row_count = int(counts[row_pos].item())
        row_indices = feasible_indices.narrow(0, row_start, row_count)
        budget = int(budgets[source].item())
        if budget > 0:
            base_count = min(row_count, budget)
            base_selected_count = base_selected_count + score.new_tensor(float(base_count))
            if row_count > budget:
                original_row_indices = row_indices
                row_scores = score.index_select(0, row_indices)
                ranked = torch.argsort(row_scores, descending=True, stable=True)
                selected_count = min(row_count, budget + int(support_smoothing_extra_per_row))
                row_indices = row_indices.index_select(0, ranked[:selected_count])
                halo_candidate_count = halo_candidate_count + score.new_tensor(float(max(selected_count - budget, 0)))
                boundary_score = row_scores.index_select(0, ranked[budget - 1 : budget])
                best_unselected = row_scores.index_select(0, ranked[budget : budget + 1])
                margin_parts.append(boundary_score - best_unselected)
                unselected_scores = score.index_select(0, original_row_indices.index_select(0, ranked[budget:]))
                near_miss_candidate_count = near_miss_candidate_count + (
                    (boundary_score - unselected_scores) <= near_miss_threshold
                ).to(dtype=dtype).sum()
            output_index_parts.append(row_indices)
    return output_index_parts, margin_parts, near_miss_candidate_count, halo_candidate_count, base_selected_count


def _select_topk_segmented_fast(
    *,
    feasible_indices: torch.Tensor,
    unique_src: torch.Tensor,
    counts: torch.Tensor,
    row_starts: torch.Tensor,
    budgets: torch.Tensor,
    score: torch.Tensor,
    near_miss_threshold: torch.Tensor,
    support_smoothing_extra_per_row: int = 0,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    if feasible_indices.numel() == 0 or unique_src.numel() == 0:
        empty = score.new_tensor(0.0)
        return [], [], empty, empty, empty
    row_budget = budgets.index_select(0, unique_src).to(dtype=torch.long)
    base_selected_count = torch.minimum(counts, torch.clamp(row_budget, min=0))
    expanded_budget = torch.where(
        row_budget > 0,
        row_budget + int(support_smoothing_extra_per_row),
        row_budget,
    )
    selected_count = torch.minimum(counts, torch.clamp(expanded_budget, min=0))
    max_selected = int(selected_count.max().item()) if selected_count.numel() else 0
    if max_selected <= 0:
        empty = score.new_tensor(0.0)
        return [], [], empty, empty, base_selected_count.to(dtype=score.dtype).sum()

    row_count = int(unique_src.numel())
    max_candidates = int(counts.max().item())
    row_id = torch.repeat_interleave(torch.arange(row_count, device=score.device, dtype=torch.long), counts)
    column = torch.arange(feasible_indices.numel(), device=score.device, dtype=torch.long) - torch.repeat_interleave(
        row_starts,
        counts,
    )
    index_matrix = torch.full(
        (row_count, max_candidates),
        -1,
        dtype=torch.long,
        device=score.device,
    )
    score_matrix = score.new_full((row_count, max_candidates), -torch.inf)
    index_matrix[row_id, column] = feasible_indices
    score_matrix[row_id, column] = score.index_select(0, feasible_indices)

    ranked_column = torch.argsort(score_matrix, dim=1, descending=True, stable=True)
    ranked_index = torch.gather(index_matrix, 1, ranked_column)
    needs_cap = (counts > row_budget) & (row_budget > 0)
    original_index = index_matrix[:, :max_selected]
    ranked_index = ranked_index[:, :max_selected]
    selected_matrix = torch.where(needs_cap.unsqueeze(1), ranked_index, original_index)
    position = torch.arange(max_selected, device=score.device, dtype=torch.long).unsqueeze(0)
    valid = position < selected_count.unsqueeze(1)
    selected_candidate_index = selected_matrix[valid]

    margin_parts: list[torch.Tensor] = []
    near_miss_candidate_count = score.new_tensor(0.0)
    if bool(torch.any(needs_cap).cpu()):
        capped_rows = torch.nonzero(needs_cap, as_tuple=False).reshape(-1)
        capped_budget = row_budget.index_select(0, capped_rows)
        capped_counts = counts.index_select(0, capped_rows)
        ranked_score = torch.gather(score_matrix, 1, ranked_column)
        capped_scores = ranked_score.index_select(0, capped_rows)
        boundary = capped_scores.gather(1, (capped_budget - 1).reshape(-1, 1)).reshape(-1)
        best_unselected = capped_scores.gather(1, capped_budget.reshape(-1, 1)).reshape(-1)
        margin_parts.append(boundary - best_unselected)
        candidate_position = torch.arange(max_candidates, device=score.device, dtype=torch.long).unsqueeze(0)
        unselected = (candidate_position >= capped_budget.reshape(-1, 1)) & (
            candidate_position < capped_counts.reshape(-1, 1)
        )
        near_miss = (boundary.reshape(-1, 1) - capped_scores) <= near_miss_threshold
        near_miss_candidate_count = (near_miss & unselected).to(dtype=score.dtype).sum()
    halo_candidate_count = torch.clamp(selected_count - base_selected_count, min=0).to(dtype=score.dtype).sum()

    return [
        selected_candidate_index
    ], margin_parts, near_miss_candidate_count, halo_candidate_count, base_selected_count.to(dtype=score.dtype).sum()


def _budget_vector(
    *,
    num_nodes: int,
    per_node_budget: torch.Tensor | None,
    max_out_degree: int | None,
    fallback: int,
    device: torch.device,
) -> torch.Tensor:
    if per_node_budget is not None:
        if torch.is_floating_point(per_node_budget) or torch.is_complex(per_node_budget) or per_node_budget.dtype == torch.bool:
            raise TypeError("per_node_budget must use an integer dtype")
        budget = per_node_budget.to(device=device, dtype=torch.long).reshape(-1)
        if budget.numel() != num_nodes:
            raise ValueError("per_node_budget must contain num_nodes values")
        if bool(torch.any(budget < 0).cpu()):
            raise ValueError("per_node_budget must be nonnegative")
        return budget
    cap = fallback if max_out_degree is None else int(max_out_degree)
    if cap < 0:
        raise ValueError("max_out_degree must be nonnegative or None")
    return torch.full((num_nodes,), cap, dtype=torch.long, device=device)


def _row_softmax(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    edge_score: torch.Tensor,
    selected_candidate_index: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    if selected_candidate_index.numel() == 0:
        return edge_score.new_empty((0,))
    selected_src = src_index.index_select(0, selected_candidate_index)
    selected_score = edge_score.index_select(0, selected_candidate_index)
    row_max = edge_score.new_full((num_nodes,), -torch.inf)
    row_max.scatter_reduce_(0, selected_src, selected_score.detach(), reduce="amax", include_self=True)
    shifted = (selected_score - row_max.index_select(0, selected_src)) / tau
    exp_score = torch.exp(shifted)
    row_sum = edge_score.new_zeros((num_nodes,)).index_add(0, selected_src, exp_score)
    return exp_score / torch.clamp(row_sum.index_select(0, selected_src), min=torch.finfo(edge_score.dtype).tiny)


def _row_softmax_full_candidate_at_selected(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    edge_score: torch.Tensor,
    feasible_indices: torch.Tensor,
    selected_candidate_index: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Soft row-softmax over the full feasible candidate set, evaluated only at
    the hard-selected positions.

    This is the backward surrogate for the Mode A straight-through estimator.
    Because the per-row normalizing denominator (``row_sum``) is accumulated over
    *all* feasible candidates of each source row, the gradient of a selected
    edge's weight flows to the scores of the *excluded* candidates in the same
    row as well. That is exactly the membership-learning signal the legacy
    ``selected_row_softmax`` path lacks (excluded candidates get zero gradient).
    The returned tensor has one entry per selected edge, aligned with
    ``selected_candidate_index``.
    """
    if selected_candidate_index.numel() == 0:
        return edge_score.new_empty((0,))
    feasible_src = src_index.index_select(0, feasible_indices)
    feasible_score = edge_score.index_select(0, feasible_indices)
    row_max = edge_score.new_full((num_nodes,), -torch.inf)
    row_max.scatter_reduce_(0, feasible_src, feasible_score.detach(), reduce="amax", include_self=True)
    feasible_shifted = (feasible_score - row_max.index_select(0, feasible_src)) / tau
    feasible_exp = torch.exp(feasible_shifted)
    row_sum = edge_score.new_zeros((num_nodes,)).index_add(0, feasible_src, feasible_exp)
    selected_src = src_index.index_select(0, selected_candidate_index)
    selected_score = edge_score.index_select(0, selected_candidate_index)
    selected_shifted = (selected_score - row_max.index_select(0, selected_src)) / tau
    selected_exp = torch.exp(selected_shifted)
    return selected_exp / torch.clamp(
        row_sum.index_select(0, selected_src), min=torch.finfo(edge_score.dtype).tiny
    )


def _row_entropy_mean(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    topology_weight: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    active_count = int(topology_weight.numel())
    if num_nodes <= 0 or active_count == 0:
        return reference.new_tensor(0.0)
    base = reference.new_zeros((num_nodes,))
    safe_weight = torch.clamp(topology_weight, min=torch.finfo(reference.dtype).tiny)
    entropy_terms = -topology_weight * torch.log(safe_weight)
    row_entropy = base.index_add(0, src_index, entropy_terms)
    active_degree = base.index_add(
        0,
        src_index,
        torch.ones(active_count, dtype=reference.dtype, device=reference.device),
    )
    active_rows = active_degree > 0.0
    if not bool(torch.any(active_rows).cpu()):
        return reference.new_tensor(0.0)
    return row_entropy[active_rows].mean()


def _diagnostics(
    *,
    num_nodes: int,
    edge_count: int,
    src_index: torch.Tensor,
    topology_weight: torch.Tensor,
    selected_mask: torch.Tensor,
    feasible_count: torch.Tensor,
    cap_hit: torch.Tensor,
    constructor_mode_code: torch.Tensor,
    topk_boundary_margin: torch.Tensor,
    near_miss_candidate_count: torch.Tensor,
    self_loop_count: torch.Tensor,
    duplicate_edge_count: torch.Tensor,
    row_softmax_temperature: torch.Tensor,
    support_smoothing_mode_code: torch.Tensor,
    support_smoothing_stage_code: torch.Tensor,
    support_smoothing_extra_per_row: torch.Tensor,
    support_smoothing_effective_extra_per_row: torch.Tensor,
    support_smoothing_temperature: torch.Tensor,
    support_smoothing_active: torch.Tensor,
    support_smoothing_halo_candidate_count: torch.Tensor,
    support_smoothing_base_selected_count: torch.Tensor,
    row_entropy_mean: torch.Tensor,
    row_entropy_delta_vs_temperature_1: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor]:
    active_count = int(topology_weight.numel())
    base = reference.new_zeros((num_nodes,))
    if active_count:
        active_degree = base.index_add(0, src_index, torch.ones(active_count, dtype=reference.dtype, device=reference.device))
        row_weight_sum = base.index_add(0, src_index, topology_weight)
        square_sum = base.index_add(0, src_index, topology_weight * topology_weight)
    else:
        active_degree = base
        row_weight_sum = base
        square_sum = base
    effective_degree = torch.where(square_sum > 0.0, 1.0 / square_sum, square_sum)
    row_weight_zero = row_weight_sum <= 0.0
    single_selected = active_degree == 1.0
    multi_selected = active_degree > 1.0
    zero_selected = active_degree == 0.0
    selected_count = selected_mask.to(dtype=reference.dtype).sum()
    feasible_total = feasible_count.sum()
    unselected_count = torch.clamp(feasible_total - selected_count, min=0.0)
    gradient_risk_rows = single_selected | cap_hit
    active_source = feasible_count > 0.0
    unique_source_count = active_source.to(dtype=reference.dtype).sum()
    active_source_candidates = feasible_count[active_source]
    selected_fraction = selected_count / torch.clamp(reference.new_tensor(float(max(edge_count, 1))), min=1.0)
    gradient_coverage = selected_count / torch.clamp(feasible_total, min=1.0)
    empty = reference.new_tensor(0.0)
    return {
        "constructor_mode_code": constructor_mode_code,
        "row_softmax_temperature": row_softmax_temperature,
        "support_smoothing_mode_code": support_smoothing_mode_code,
        "support_smoothing_stage_code": support_smoothing_stage_code,
        "support_smoothing_extra_per_row": support_smoothing_extra_per_row,
        "support_smoothing_effective_extra_per_row": support_smoothing_effective_extra_per_row,
        "support_smoothing_temperature": support_smoothing_temperature,
        "support_smoothing_active": support_smoothing_active,
        "support_smoothing_affects_support": reference.new_tensor(
            1.0 if float(support_smoothing_halo_candidate_count.detach().cpu().item()) > 0.0 else 0.0
        ),
        "support_smoothing_base_selected_count": support_smoothing_base_selected_count,
        "support_smoothing_halo_candidate_count": support_smoothing_halo_candidate_count,
        "support_smoothing_sparse_ratio": selected_count / torch.clamp(feasible_total, min=1.0),
        "temperature_affects_weights": reference.new_tensor(
            1.0 if abs(float(row_softmax_temperature.detach().cpu().item()) - 1.0) > 1.0e-12 else 0.0
        ),
        "temperature_affects_support": reference.new_tensor(0.0),
        "edge_count": reference.new_tensor(float(edge_count)),
        "candidate_edge_count": reference.new_tensor(float(edge_count)),
        "feasible_candidate_count": feasible_total,
        "active_edge_count": reference.new_tensor(float(active_count)),
        "unique_source_count": unique_source_count,
        "mean_candidates_per_active_source": active_source_candidates.mean() if active_source_candidates.numel() else empty,
        "max_candidates_per_active_source": active_source_candidates.max() if active_source_candidates.numel() else empty,
        "selected_fraction": selected_fraction,
        "gradient_coverage_fraction": gradient_coverage,
        "mean_out_degree": active_degree.mean() if num_nodes else empty,
        "max_out_degree": active_degree.max() if num_nodes else empty,
        "out_degree": active_degree,
        "feasible_out_degree": feasible_count,
        "row_weight_zero_count": row_weight_zero.to(dtype=reference.dtype).sum(),
        "cap_hit_count": cap_hit.to(dtype=reference.dtype).sum(),
        "topk_cap_active_count": cap_hit.to(dtype=reference.dtype).sum(),
        "cap_hit_ratio": cap_hit.to(dtype=reference.dtype).mean() if num_nodes else empty,
        "min_row_weight_sum": row_weight_sum.min() if num_nodes else empty,
        "max_row_weight_sum": row_weight_sum.max() if num_nodes else empty,
        "row_weight_sum": row_weight_sum,
        "row_entropy_mean": row_entropy_mean,
        "row_entropy_delta_vs_temperature_1": row_entropy_delta_vs_temperature_1,
        "min_topology_weight": topology_weight.min() if active_count else empty,
        "max_topology_weight": topology_weight.max() if active_count else empty,
        "effective_query_degree": effective_degree,
        "min_effective_query_degree": effective_degree.min() if num_nodes else empty,
        "mean_effective_query_degree": effective_degree.mean() if num_nodes else empty,
        "selected_candidate_count": selected_count,
        "unselected_candidate_count": unselected_count,
        "single_selected_row_count": single_selected.to(dtype=reference.dtype).sum(),
        "multi_selected_row_count": multi_selected.to(dtype=reference.dtype).sum(),
        "zero_selected_row_count": zero_selected.to(dtype=reference.dtype).sum(),
        "topk_boundary_margin": topk_boundary_margin,
        "min_topk_boundary_margin": topk_boundary_margin.min() if topk_boundary_margin.numel() else empty,
        "p10_topk_boundary_margin": torch.quantile(topk_boundary_margin, 0.10) if topk_boundary_margin.numel() else empty,
        "near_miss_candidate_count": near_miss_candidate_count,
        "zero_gradient_risk_count": gradient_risk_rows.to(dtype=reference.dtype).sum(),
        "self_loop_count": self_loop_count,
        "duplicate_edge_count": duplicate_edge_count,
    }
