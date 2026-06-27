"""Wire the CDQ 2.0 kernel (k-DPP inclusion + determinantal quorum) into the canonical path.

The CDQ 2.0 analogue of ``src/sampling/cdq_query.py`` (the old low-rank ``L = BB^T`` CDQ): a CDQ 2.0
query policy emits a per-edge quality ``a_ij > 0``, a diversity embedding ``z_ij in R^r``, and a
diversity strength ``eta >= 0``; the canonical episode then uses

* :func:`cdq2_edge_inclusion` -- per-source ``k``-DPP inclusion marginals ``pi_ij`` for the
  receiver load / interference;
* :func:`cdq2_bucketed_quorum` -- the exact CDQ 2.0 heterogeneous quorum ``(h^+, h^-, h^0)`` with
  ``p^+_ij = ell_ij u_j``, ``p^-_ij = ell_ij v_j``.

Both run on the degree-bucketed layout (total padded cells ``<= 2E``, no ``N x N``). Padded slots
are excluded EXACTLY via the bucket ``slot_mask`` threaded into the CDQ 2.0 kernel (which zeroes
the padded kernel rows -> a zero eigenvalue, contributing nothing to ``e_k``, and a zero inclusion)
-- so ``sum_j pi_ij = k`` over the REAL candidates and the masked candidates never enter the
quorum. This is the EXACT padded-exclusion (not the old ``sqrt(quality.clamp_min(eps))`` ~eps
approximation).

ESP anchor. Because ``eta = 0 => L = D = diag(a)`` EXACTLY (for ANY diversity ``Z``), a
:class:`CDQ2Policy` built from an ESP log-weight policy with ``eta = 0`` reproduces the ESP
canonical episode bit-for-bit -- the wiring's correctness anchor.
"""

from __future__ import annotations

import torch

from src.mainline.global_evaluator import BucketedPadding, _edge_rank, build_bucketed_padding

from .cdq2_kernel import cdq2_inclusion
from .cdq2_quorum import cdq2_quorum_decision

__all__ = ["cdq2_edge_inclusion", "cdq2_bucketed_quorum", "CDQ2Policy"]


def _bucket_kernel(quality: torch.Tensor, diversity: torch.Tensor, bucket):
    """Per-bucket masked ``(quality_b [m,w], diversity_b [m,w,r], mask [m,w])`` with padded rows zeroed."""
    qse = quality[bucket.slot_edge]                 # [m, w]
    dse = diversity[bucket.slot_edge]               # [m, w, r]
    mask = bucket.slot_mask                          # [m, w]
    qb = torch.where(mask, qse, torch.zeros_like(qse))
    db = torch.where(mask.unsqueeze(-1), dse, torch.zeros_like(dse))
    return qb, db, mask


def cdq2_edge_inclusion(
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    num_nodes: int,
    quality: torch.Tensor,
    diversity: torch.Tensor,
    eta,
    k: int,
    *,
    padding: BucketedPadding | None = None,
) -> torch.Tensor:
    """Per-edge ``k``-DPP inclusion ``pi_ij`` for the CDQ 2.0 kernel ``L = D^{1/2}(I+eta ZZ^T)D^{1/2}``.

    ``[E]``; ``sum_{j: i->j} pi_ij = k`` for every source. Masked/padded slots get exactly 0.
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
        inc = cdq2_inclusion(qb, db, eta, k, mask=mask)             # [m, w] (padded rows -> 0)
        pi[bucket.slot_edge[mask]] = inc[mask]
    return pi


def cdq2_bucketed_quorum(
    padding: BucketedPadding,
    quality: torch.Tensor,
    diversity: torch.Tensor,
    eta,
    ell_edge: torch.Tensor,        # [E, Q]
    pref_c: torch.Tensor,          # [N, Q]  u_j
    pref_w: torch.Tensor,          # [N, Q]  v_j
    k: int,
    alpha: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One round of CDQ 2.0 determinantal quorum decisions for all sources -> ``(h+,h-,h0)`` [N,Q]."""
    N, Q = pref_c.shape
    r = diversity.shape[-1]
    h_plus = pref_c.new_zeros((N, Q))
    h_minus = pref_c.new_zeros((N, Q))
    h_zero = pref_c.new_zeros((N, Q))
    for bucket in padding.buckets:
        m, w = bucket.node_ids.numel(), bucket.width
        qb, db, mask = _bucket_kernel(quality, diversity, bucket)        # [m,w], [m,w,r], [m,w]
        ell_b = ell_edge[bucket.slot_edge]                              # [m, w, Q]
        dst_b = bucket.dst_slot                                          # [m, w]
        u_b = pref_c[dst_b]                                              # [m, w, Q]
        v_b = pref_w[dst_b]
        mk = mask.unsqueeze(-1)
        p_plus = torch.where(mk, ell_b * u_b, torch.zeros_like(ell_b)).permute(0, 2, 1)   # [m,Q,w]
        p_minus = torch.where(mk, ell_b * v_b, torch.zeros_like(ell_b)).permute(0, 2, 1)
        a_q = qb.unsqueeze(1).expand(m, Q, w)                            # [m, Q, w]
        Z_q = db.unsqueeze(1).expand(m, Q, w, r)                         # [m, Q, w, r]
        mask_q = mask.unsqueeze(1).expand(m, Q, w)                       # [m, Q, w]
        dec = cdq2_quorum_decision(a_q, Z_q, eta, p_plus, p_minus, k, alpha, mask=mask_q)  # h.* [m,Q]
        h_plus = h_plus.index_copy(0, bucket.node_ids, dec.h_plus)
        h_minus = h_minus.index_copy(0, bucket.node_ids, dec.h_minus)
        h_zero = h_zero.index_copy(0, bucket.node_ids, dec.h_zero)
    return h_plus, h_minus, h_zero


class CDQ2Policy:
    """A CDQ 2.0 query policy: quality ``a_ij = exp(s_ij)`` from a wrapped ESP log-weight policy,
    a diversity embedding ``z_ij``, and a strength ``eta``. ``query_law = "cdq2"``.

    With ``eta = 0`` the kernel is EXACTLY ``diag(a)`` (for any diversity), so the CDQ 2.0 canonical
    path reproduces the wrapped policy's ESP episode bit-for-bit. ``diversity`` may be a tensor
    ``[E, r]``, a callable ``graph -> [E, r]``, or ``None`` (default per-source one-hot rank, which
    needs ``r >= max out-degree`` and is meaningful only as the ``eta = 0`` anchor; pass real
    embeddings for ``eta > 0``).
    """

    query_law = "cdq2"
    name = "cdq2"

    def __init__(self, base_log_weight_policy, r: int, eta=0.0, *, diversity=None):
        self.base = base_log_weight_policy
        self.r = int(r)
        self.eta = eta
        self._diversity = diversity

    def kernel(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        log_w = self.base.log_weights(graph)
        quality = torch.exp(log_w)                                       # a_ij = exp(s_ij)
        if self._diversity is not None:
            div = self._diversity(graph) if callable(self._diversity) else self._diversity
            return quality, div.to(dtype=quality.dtype)
        deg = torch.bincount(graph.src_index, minlength=graph.num_nodes)
        if int(deg.max()) > self.r:
            raise ValueError(f"default one-hot diversity needs r >= max out-degree ({int(deg.max())}); got r={self.r}")
        rank = _edge_rank(graph.src_index, deg, graph.num_nodes)         # [E] within-source rank
        diversity = torch.nn.functional.one_hot(rank, self.r).to(dtype=quality.dtype)
        return quality, diversity
