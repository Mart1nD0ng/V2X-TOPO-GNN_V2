"""Temporal topology metrics for mobility / sequence training.

Across mobility frames the candidate graph changes (vehicles move), so the
candidate-index space is NOT comparable between frames. Topology stability must be
measured on the directed NODE-PAIR set ``(src_node, dst_node)`` — node identity is
stable across frames (fixed population). ``support_overlap`` is the Jaccard overlap
of two frames' selected node-pair sets; ``topology_churn = 1 - support_overlap`` is
the fraction of the support that was re-planned between frames (a proxy for control/
signalling re-association cost). These are reporting/diagnostic metrics (set-based,
not differentiable); a differentiable churn surrogate is a separate concern for any
loss-coupled (temporal) phase.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _edge_keys(src_index: Any, dst_index: Any, num_nodes: int) -> np.ndarray:
    src = np.asarray(src_index).reshape(-1).astype(np.int64)
    dst = np.asarray(dst_index).reshape(-1).astype(np.int64)
    if src.size != dst.size:
        raise ValueError("src_index and dst_index must have equal length")
    if num_nodes <= 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(src * np.int64(num_nodes) + dst)


def support_overlap(
    src_a: Any, dst_a: Any, src_b: Any, dst_b: Any, num_nodes: int
) -> float:
    """Jaccard overlap of two selected topologies' directed node-pair sets.

    Returns 1.0 when both supports are empty (defined as fully overlapping).
    """
    a = _edge_keys(src_a, dst_a, num_nodes)
    b = _edge_keys(src_b, dst_b, num_nodes)
    if a.size == 0 and b.size == 0:
        return 1.0
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = np.union1d(a, b).size
    return float(intersection) / float(union) if union else 1.0


def topology_churn(
    src_a: Any, dst_a: Any, src_b: Any, dst_b: Any, num_nodes: int
) -> float:
    """Fraction of the support re-planned between two frames: 1 - support_overlap."""
    return 1.0 - support_overlap(src_a, dst_a, src_b, dst_b, num_nodes)


def mean_sequence_churn(selected_pairs: list[tuple[Any, Any]], num_nodes: int) -> float:
    """Mean consecutive-frame churn over a sequence of selected topologies.

    Args:
        selected_pairs: ordered list of ``(src_index, dst_index)`` per frame.
        num_nodes: node count (stable across frames).
    Returns 0.0 for sequences shorter than two frames (no transition to score).
    """
    if len(selected_pairs) < 2:
        return 0.0
    churns = [
        topology_churn(
            selected_pairs[t - 1][0], selected_pairs[t - 1][1],
            selected_pairs[t][0], selected_pairs[t][1], num_nodes,
        )
        for t in range(1, len(selected_pairs))
    ]
    return float(np.mean(churns))
