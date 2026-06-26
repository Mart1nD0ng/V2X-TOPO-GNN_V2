"""Exact per-edge ``k``-subset inclusion probabilities on a candidate graph (spec §4).

Given per-edge query log-weights ``s_ij`` (``a_ij = exp(s_ij)``) the distinct-peer
``k``-subset policy selects exactly ``k`` peers per source with the elementary-symmetric
law ``P(S) ∝ prod_{j∈S} a_ij`` (spec §4, Eq. 16). The per-edge inclusion marginal is

    pi_ij = a_ij e_{k-1}(a_{i,-j}) / e_k(a_i),     sum_j pi_ij = k                 (Eq. 18-19)

This module computes ``pi`` for every edge of a :class:`RadiusGraph` using the approved
degree-bucketed layout (total padded cells ``<= 2E``, no ``N x N``, no degree cap) and the
exact log-domain elementary-symmetric routine from ``src.mainline.symmetric_polynomials``.
It is the canonical-path bridge from a query policy's log-weights to the ``pi`` consumed by
the round physics (load / interference) and the quorum DP.

This is the **diagonal-kernel special case** of the CDQ ``k``-DPP query law (spec §9.4);
G4 generalises it to a low-rank determinantal kernel with the same interface.
"""

from __future__ import annotations

import torch

from src.mainline.global_evaluator import BucketedPadding, build_bucketed_padding
from src.mainline.symmetric_polynomials import edge_inclusion_probability

__all__ = ["edge_inclusion_probabilities"]


def edge_inclusion_probabilities(
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    num_nodes: int,
    log_weights: torch.Tensor,
    k: int,
    *,
    padding: BucketedPadding | None = None,
) -> torch.Tensor:
    """Per-edge inclusion probability ``pi_ij`` for the ``k``-subset policy (Eq. 18).

    Args:
        src_index, dst_index: directed candidate edges ``i -> j`` (``G_comm``).
        num_nodes: ``N``.
        log_weights: ``[E]`` per-edge query log-weights ``s_ij`` (policy output).
        k: subset size ``k_poll``.
        padding: optional precomputed degree-bucketed layout (reused across rounds).

    Returns:
        ``[E]`` inclusion probabilities; ``sum_{j: i->j} pi_ij = k`` for every source.

    Raises:
        ValueError: if any source has out-degree ``< k`` (apply the §7.2
            candidate-shortage protocol upstream -- no duplication padding, constraint #4).
    """
    if log_weights.ndim != 1 or log_weights.shape[0] != int(src_index.numel()):
        raise ValueError("log_weights must be [E] matching the edge count")
    if padding is None:
        padding = build_bucketed_padding(src_index, dst_index, num_nodes)
    if bool(torch.any(padding.out_degree < k).cpu()):
        raise ValueError(
            "a source has out-degree < k; apply the §7.2 candidate-shortage protocol "
            "(k_i = min(k, |N_i|) or RSU fallback) upstream -- never pad by duplication"
        )
    pi = torch.zeros_like(log_weights)
    neg = log_weights.new_full((), float("-inf"))
    for bucket in padding.buckets:
        lw = log_weights[bucket.slot_edge]                     # [m, w] (slot_edge 0 where invalid)
        lw = torch.where(bucket.slot_mask, lw, neg)
        inc = edge_inclusion_probability(lw, k, mask=bucket.slot_mask)  # [m, w]
        se = bucket.slot_edge[bucket.slot_mask]                # real edge ids
        pi[se] = inc[bucket.slot_mask]
    return pi
