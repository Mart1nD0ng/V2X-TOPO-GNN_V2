"""Differentiable building blocks for the temporal (horizon) training objective.

The horizon objective rewards quantities a per-frame objective cannot express:
  - anticipation: evaluate the topology chosen at frame t on a SUB-FRAME a fraction
    of dt later (vehicles advanced) -> does the chosen topology hold up as the scene
    moves? `edge_distances` recomputes per-edge distances at the sub-frame positions
    for the already-selected edges (node identity is stable), so the same evaluator
    yields F_transition, differentiable through the topology weights.
  - temporal smoothness: `churn_proxy` is a DIFFERENTIABLE surrogate for re-planning
    cost (mean L1 change in per-node incoming load), complementing the non-
    differentiable set-based `topology_churn` reporting metric.

The horizon aggregation itself (summing per-frame F + anticipation + churn over a
window) lives in the training loop, which owns the evaluator and the stream.
"""

from __future__ import annotations

import torch


def edge_distances(pos_x: torch.Tensor, pos_y: torch.Tensor, src_index: torch.Tensor, dst_index: torch.Tensor) -> torch.Tensor:
    """Euclidean distance (meters) per directed edge from node positions.

    Used to re-evaluate an already-chosen topology on advanced (sub-frame) positions
    for the anticipation term.
    """
    src = src_index.reshape(-1).to(dtype=torch.long)
    dst = dst_index.reshape(-1).to(dtype=torch.long)
    dx = pos_x.index_select(0, dst) - pos_x.index_select(0, src)
    dy = pos_y.index_select(0, dst) - pos_y.index_select(0, src)
    return torch.sqrt(dx * dx + dy * dy + torch.finfo(pos_x.dtype).tiny)


def node_in_load(num_nodes: int, dst_index: torch.Tensor, topology_weight: torch.Tensor) -> torch.Tensor:
    """Per-node incoming topology weight (differentiable load proxy)."""
    base = topology_weight.new_zeros((int(num_nodes),))
    return base.index_add(0, dst_index.reshape(-1).to(dtype=torch.long), topology_weight.reshape(-1))


def churn_proxy(in_load_prev: torch.Tensor, in_load_curr: torch.Tensor) -> torch.Tensor:
    """Differentiable re-planning-cost surrogate: mean L1 change in per-node in-load.

    Operates on the stable node space (in-load per node), so it is comparable across
    frames whose candidate edge sets differ. Returns a non-negative scalar; 0 when
    the per-node load is unchanged.
    """
    if in_load_prev.numel() != in_load_curr.numel():
        raise ValueError("in-load vectors must have equal length (stable node space)")
    return (in_load_curr - in_load_prev).abs().mean()
