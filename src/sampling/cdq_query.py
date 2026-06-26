"""Wire the CDQ k-DPP query (G4) + determinantal quorum (G5) into the canonical path (G9 infra).

This is where the core model-math finally drives the headline evaluator: a CDQ query policy
emits a per-edge quality ``q_ij > 0`` and diversity embedding ``b_ij in R^r`` (the kernel root
``B``); the canonical episode then uses

* :func:`cdq_edge_inclusion` -- per-source ``k``-DPP inclusion marginals ``pi_ij`` (G4) for the
  receiver load / interference;
* :func:`cdq_bucketed_quorum` -- the exact determinantal heterogeneous quorum ``(h^+,h^-,h^0)``
  (G5) with ``p^+_ij = ell_ij u_j``, ``p^-_ij = ell_ij v_j``.

Both run on the degree-bucketed layout (total padded cells ``<= 2E``, no ``N x N``, no degree
cap). Padded slots are excluded EXACTLY by zeroing their kernel rows (a zero ``B`` row
contributes nothing to ``L = B B^T`` and gets inclusion 0), so ``sum_j pi_ij = k`` over the
real candidates and the masked candidates never enter the quorum. ``r >= k`` is required;
``r`` need NOT exceed the degree (the low-rank dual handles ``w > r``).

Diagonal special case (the wiring's correctness anchor). With orthonormal per-source diversity
rows the kernel is ``diag(q)`` and the CDQ inclusion/quorum reduce EXACTLY to the §4 ESP
inclusion and §5 ``quorum_dp`` -- so a :class:`DiagonalCDQPolicy` built from an ESP policy
reproduces the ESP canonical episode bit-for-bit (the consistency test).
"""

from __future__ import annotations

import torch

from src.mainline.global_evaluator import BucketedPadding, _edge_rank, build_bucketed_padding

from .determinantal_quorum import determinantal_quorum_decision
from .dpp_query import kdpp_inclusion, low_rank_kernel

__all__ = ["cdq_edge_inclusion", "cdq_bucketed_quorum", "DiagonalCDQPolicy"]


def _bucket_kernel(quality: torch.Tensor, diversity: torch.Tensor, bucket):
    """Per-bucket masked ``(quality_b [m,w], diversity_b [m,w,r])`` with padded rows zeroed."""
    qse = quality[bucket.slot_edge]                 # [m, w]
    dse = diversity[bucket.slot_edge]               # [m, w, r]
    mask = bucket.slot_mask                          # [m, w]
    qb = torch.where(mask, qse, torch.zeros_like(qse))
    db = torch.where(mask.unsqueeze(-1), dse, torch.zeros_like(dse))
    return qb, db, mask


def cdq_edge_inclusion(
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    num_nodes: int,
    quality: torch.Tensor,
    diversity: torch.Tensor,
    k: int,
    *,
    padding: BucketedPadding | None = None,
) -> torch.Tensor:
    """Per-edge ``k``-DPP inclusion ``pi_ij`` for the CDQ kernel ``L = B B^T`` (G4). ``[E]``.

    ``sum_{j: i->j} pi_ij = k`` for every source. Masked/padded slots get exactly 0.
    """
    if quality.ndim != 1 or quality.shape[0] != int(src_index.numel()):
        raise ValueError("quality must be [E] matching the edge count")
    if diversity.ndim != 2 or diversity.shape[0] != quality.shape[0]:
        raise ValueError("diversity must be [E, r] matching quality")
    if padding is None:
        padding = build_bucketed_padding(src_index, dst_index, num_nodes)
    if bool(torch.any(padding.out_degree < k).cpu()):
        raise ValueError("a source has out-degree < k; apply the §7.2 shortage protocol upstream")
    pi = torch.zeros_like(quality)
    for bucket in padding.buckets:
        qb, db, mask = _bucket_kernel(quality, diversity, bucket)
        inc = kdpp_inclusion(qb, db, k)             # [m, w] (Σ over w = k; padded rows -> 0)
        pi[bucket.slot_edge[mask]] = inc[mask]
    return pi


def cdq_bucketed_quorum(
    padding: BucketedPadding,
    quality: torch.Tensor,
    diversity: torch.Tensor,
    ell_edge: torch.Tensor,        # [E, Q]
    pref_c: torch.Tensor,          # [N, Q]  u_j
    pref_w: torch.Tensor,          # [N, Q]  v_j
    k: int,
    alpha: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One round of CDQ determinantal quorum decisions for all sources -> ``(h+,h-,h0)`` [N,Q]."""
    N, Q = pref_c.shape
    r = diversity.shape[-1]
    h_plus = pref_c.new_zeros((N, Q))
    h_minus = pref_c.new_zeros((N, Q))
    h_zero = pref_c.new_zeros((N, Q))
    for bucket in padding.buckets:
        m, w = bucket.node_ids.numel(), bucket.width
        qb, db, mask = _bucket_kernel(quality, diversity, bucket)        # [m,w], [m,w,r]
        B_b = low_rank_kernel(qb, db)                                    # [m, w, r]
        ell_b = ell_edge[bucket.slot_edge]                              # [m, w, Q]
        dst_b = bucket.dst_slot                                          # [m, w]
        u_b = pref_c[dst_b]                                              # [m, w, Q]
        v_b = pref_w[dst_b]
        mk = mask.unsqueeze(-1)
        p_plus = torch.where(mk, ell_b * u_b, torch.zeros_like(ell_b))   # [m, w, Q]
        p_minus = torch.where(mk, ell_b * v_b, torch.zeros_like(ell_b))
        # batch over scenarios Q: kernel is scenario-independent, response probs are per-Q
        B_q = B_b.unsqueeze(1).expand(m, Q, w, r)                        # [m, Q, w, r]
        pp_q = p_plus.permute(0, 2, 1)                                   # [m, Q, w]
        pm_q = p_minus.permute(0, 2, 1)
        dec = determinantal_quorum_decision(B_q, pp_q, pm_q, k, alpha)   # h.* [m, Q]
        h_plus = h_plus.index_copy(0, bucket.node_ids, dec.h_plus)
        h_minus = h_minus.index_copy(0, bucket.node_ids, dec.h_minus)
        h_zero = h_zero.index_copy(0, bucket.node_ids, dec.h_zero)
    return h_plus, h_minus, h_zero


class DiagonalCDQPolicy:
    """A CDQ policy whose kernel is exactly diagonal ``diag(exp(s_ij))`` -- so the CDQ canonical
    path reproduces the ESP episode of the wrapped log-weight policy bit-for-bit.

    Diversity rows are per-source one-hots (rank within the source), requiring ``r >= max
    out-degree``. Used to validate the CDQ wiring and as the "ESP-as-CDQ" baseline.
    """

    query_law = "cdq"
    name = "diagonal_cdq"

    def __init__(self, base_log_weight_policy, r: int):
        self.base = base_log_weight_policy
        self.r = int(r)

    def kernel(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        log_w = self.base.log_weights(graph)
        quality = torch.exp(log_w)                                       # a_ij = exp(s_ij)
        deg = torch.bincount(graph.src_index, minlength=graph.num_nodes)
        if int(deg.max()) > self.r:
            raise ValueError(f"DiagonalCDQPolicy needs r >= max out-degree ({int(deg.max())}); got r={self.r}")
        rank = _edge_rank(graph.src_index, deg, graph.num_nodes)         # [E] within-source rank
        diversity = torch.nn.functional.one_hot(rank, self.r).to(dtype=quality.dtype)  # [E, r]
        return quality, diversity
