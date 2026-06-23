"""Exact heterogeneous quorum probabilities via a differentiable DP (spec §5).

In shared scenario ``r`` and round ``t`` each candidate neighbour ``j`` of source
``i`` returns *correct*, *wrong* or *no-response* with edge probabilities

    p^+_{ij} = ell_{ij} u_{j},   p^-_{ij} = ell_{ij} v_{j},   p^0_{ij} = 1 - p^+ - p^-   (Eqs. 22-24)

The query policy selects *exactly* ``k`` distinct peers with the weighted
elementary-symmetric law of §4 (weights ``a_{ij} = exp(s_{ij})``).  The joint
generating function over (#selected, #correct, #wrong) is

    Psi_i(z, x, y) = prod_j [ 1 + z a_{ij} ( p^0_{ij} + p^+_{ij} x + p^-_{ij} y ) ]   (Eq. 25)

and the probability of obtaining ``m`` correct, ``n`` wrong among exactly ``k``
selected peers is the normalised coefficient

    P_i(m, n) = [z^k x^m y^n] Psi_i / e_k(a_i).                                       (Eq. 26)

This module extracts that coefficient with the exact division-free DP of Eq. 30,

    C_{j,q,m,n} = C_{j-1,q,m,n}
                + a_j p^0_j C_{j-1,q-1,m,n}
                + a_j p^+_j C_{j-1,q-1,m-1,n}
                + a_j p^-_j C_{j-1,q-1,m,n-1},

keeping only ``q <= k`` and ``m + n <= q``.  Per source the cost is
``O(|N_i| k^3)`` and over the graph ``O(E k^3)`` (Eq. 31).  Crucially this is an
*exact* heterogeneous quorum over distinct peers -- it does **not** use the iid
exchangeable-vote closure (a per-draw Beta integral over a single realised support)
that spec §5 explicitly replaces.

The quorum decisions follow (with a strict majority ``2 alpha > k`` so the correct
and wrong quorums are mutually exclusive, Eqs. 27-29):

    h^+_i = sum_{m>=alpha} P_i(m, n),   h^-_i = sum_{n>=alpha} P_i(m, n),   h^0_i = 1 - h^+ - h^-.

Everything is differentiable w.r.t. the logits ``s_{ij}`` and the edge response
probabilities, with no Monte-Carlo sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

import torch

__all__ = [
    "QuorumDecision",
    "quorum_response_distribution",
    "quorum_decision_probabilities",
    "bruteforce_quorum_distribution",
]


def _validate_quorum_params(k: int, alpha: int) -> None:
    if not isinstance(k, int) or not isinstance(alpha, int):
        raise TypeError("k and alpha must be ints")
    if k < 1:
        raise ValueError("k must be >= 1")
    if alpha < 1 or alpha > k:
        raise ValueError("alpha must satisfy 1 <= alpha <= k")
    if 2 * alpha <= k:
        raise ValueError("alpha must be a strict majority: 2*alpha > k (so + and - quorums are exclusive)")


def _prepare_inputs(
    log_weights: torch.Tensor,
    p_correct: torch.Tensor,
    p_wrong: torch.Tensor,
    mask: torch.Tensor | None,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(a_norm, p0, p_plus, p_minus)`` as ``[B, n]`` tensors.

    ``a`` is normalised by the per-source max logit (``P_i(m,n)`` is scale-invariant
    in ``a``) so the linear-domain DP never overflows for GNN logits.  Masked-out
    candidates get weight ``0`` -> they contribute nothing (exact, like §4 padding).
    """
    if log_weights.ndim != 2:
        raise ValueError("log_weights must be [B, n]")
    if p_correct.shape != log_weights.shape or p_wrong.shape != log_weights.shape:
        raise ValueError("p_correct / p_wrong must match log_weights shape")
    dtype, device = log_weights.dtype, log_weights.device
    if mask is None:
        mask = torch.ones_like(log_weights, dtype=torch.bool)
    else:
        mask = mask.to(dtype=torch.bool, device=device)

    # validate edge response probabilities
    for name, val in (("p_correct", p_correct), ("p_wrong", p_wrong)):
        d = val.detach()
        if bool(torch.any(d < -tol).cpu()) or bool(torch.any(d > 1.0 + tol).cpu()):
            raise ValueError(f"{name} must be in [0, 1]")
    if bool(torch.any((p_correct + p_wrong).detach() > 1.0 + tol).cpu()):
        raise ValueError("p_correct + p_wrong must be <= 1 (p0 = 1 - p+ - p-)")

    # per-source max for scale-stable normalisation; masked entries excluded
    neg = log_weights.new_full((), float("-inf"))
    masked_logits = torch.where(mask, log_weights, neg)
    row_max = masked_logits.max(dim=-1, keepdim=True).values
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    a = torch.exp(log_weights - row_max)
    a = torch.where(mask, a, torch.zeros_like(a))

    p_plus = torch.clamp(p_correct, 0.0, 1.0)
    p_minus = torch.clamp(p_wrong, 0.0, 1.0)
    p0 = torch.clamp(1.0 - p_plus - p_minus, 0.0, 1.0)
    # zero out masked edges entirely (defensive; a=0 already kills them)
    p_plus = torch.where(mask, p_plus, torch.zeros_like(p_plus))
    p_minus = torch.where(mask, p_minus, torch.zeros_like(p_minus))
    p0 = torch.where(mask, p0, torch.zeros_like(p0))
    return a, p0, p_plus, p_minus


def _shift_up(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Shift ``t`` toward higher index by 1 along ``dim``, zero-filling index 0."""
    size = t.size(dim)
    zeros = torch.zeros_like(t.narrow(dim, 0, 1))
    return torch.cat([zeros, t.narrow(dim, 0, size - 1)], dim=dim)


def _quorum_dp_coefficients(
    a: torch.Tensor,
    p0: torch.Tensor,
    p_plus: torch.Tensor,
    p_minus: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Run the Eq. 30 DP and return the ``z^k`` coefficient slice ``C[:, k, m, n]``.

    ``C`` has shape ``[B, k+1, k+1, k+1]`` indexed ``[q, m, n]``.  Returns
    ``[B, k+1, k+1]`` (the ``q = k`` slice), whose ``(m, n)`` entry is the
    unnormalised coefficient ``[z^k x^m y^n] Psi`` evaluated on the normalised
    weights.
    """
    B, n = a.shape
    K1 = k + 1
    C = a.new_zeros((B, K1, K1, K1))
    C[:, 0, 0, 0] = 1.0
    for j in range(n):
        aj = a[:, j].reshape(B, 1, 1, 1)
        p0j = p0[:, j].reshape(B, 1, 1, 1)
        ppj = p_plus[:, j].reshape(B, 1, 1, 1)
        pmj = p_minus[:, j].reshape(B, 1, 1, 1)
        Cq = _shift_up(C, 1)  # C[q-1, m, n]
        term0 = p0j * Cq
        term_plus = ppj * _shift_up(Cq, 2)  # C[q-1, m-1, n]
        term_minus = pmj * _shift_up(Cq, 3)  # C[q-1, m, n-1]
        C = C + aj * (term0 + term_plus + term_minus)
    return C[:, k, :, :]


@dataclass(frozen=True)
class QuorumDecision:
    h_plus: torch.Tensor  # [B] P(correct quorum)
    h_minus: torch.Tensor  # [B] P(wrong quorum)
    h_zero: torch.Tensor  # [B] P(no quorum this round)
    response_distribution: torch.Tensor  # [B, k+1, k+1] P(m correct, n wrong)


def quorum_response_distribution(
    log_weights: torch.Tensor,
    p_correct: torch.Tensor,
    p_wrong: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
    tol: float = 1e-6,
) -> torch.Tensor:
    """``P_i(m, n)`` for all ``0 <= m, n`` with ``m + n <= k`` (Eq. 26).

    Returns ``[B, k+1, k+1]``; entry ``[b, m, n]`` is the probability of ``m``
    correct and ``n`` wrong responses among exactly ``k`` selected distinct peers.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive int")
    a, p0, p_plus, p_minus = _prepare_inputs(log_weights, p_correct, p_wrong, mask, tol)
    coeff = _quorum_dp_coefficients(a, p0, p_plus, p_minus, k)  # [B, k+1, k+1]
    z = coeff.sum(dim=(1, 2), keepdim=True)  # e_k(a_norm)
    if bool(torch.any(z.detach() <= 0).cpu()):
        raise ValueError("e_k(a) is zero for some source: fewer than k valid candidates (apply §7.2 upstream)")
    return coeff / z


def quorum_decision_probabilities(
    log_weights: torch.Tensor,
    p_correct: torch.Tensor,
    p_wrong: torch.Tensor,
    k: int,
    alpha: int,
    *,
    mask: torch.Tensor | None = None,
    tol: float = 1e-6,
) -> QuorumDecision:
    """Correct / wrong / no-quorum probabilities ``(h^+, h^-, h^0)`` (Eqs. 27-29)."""
    _validate_quorum_params(k, alpha)
    P = quorum_response_distribution(log_weights, p_correct, p_wrong, k, mask=mask, tol=tol)
    h_plus = P[:, alpha:, :].sum(dim=(1, 2))
    h_minus = P[:, :, alpha:].sum(dim=(1, 2))
    h_zero = torch.clamp(1.0 - h_plus - h_minus, 0.0, 1.0)
    return QuorumDecision(h_plus=h_plus, h_minus=h_minus, h_zero=h_zero, response_distribution=P)


def bruteforce_quorum_distribution(
    weights: Sequence[float] | torch.Tensor,
    p_correct: Sequence[float] | torch.Tensor,
    p_wrong: Sequence[float] | torch.Tensor,
    k: int,
    alpha: int,
) -> dict:
    """Reference Eq. 26 by explicit enumeration (single source, for tests).

    Enumerates all ``C(n, k)`` distinct-peer subsets and, for each, all ``3^k``
    correct/wrong/no-response assignments.  Returns the full ``P(m, n)`` dict plus
    ``h_plus`` / ``h_minus``.
    """
    a = [float(x) for x in (weights.tolist() if isinstance(weights, torch.Tensor) else weights)]
    pp = [float(x) for x in (p_correct.tolist() if isinstance(p_correct, torch.Tensor) else p_correct)]
    pm = [float(x) for x in (p_wrong.tolist() if isinstance(p_wrong, torch.Tensor) else p_wrong)]
    p0 = [1.0 - pp[j] - pm[j] for j in range(len(a))]
    n = len(a)
    coeff: dict[tuple[int, int], float] = {}
    ek = 0.0
    for subset in combinations(range(n), k):
        w_subset = 1.0
        for j in subset:
            w_subset *= a[j]
        ek += w_subset
        for assign in product((0, 1, 2), repeat=k):  # 0=none, 1=correct, 2=wrong
            w = w_subset
            m_corr = 0
            n_wrong = 0
            for slot, j in enumerate(subset):
                kind = assign[slot]
                if kind == 0:
                    w *= p0[j]
                elif kind == 1:
                    w *= pp[j]
                    m_corr += 1
                else:
                    w *= pm[j]
                    n_wrong += 1
            coeff[(m_corr, n_wrong)] = coeff.get((m_corr, n_wrong), 0.0) + w
    dist = {mn: c / ek for mn, c in coeff.items()}
    h_plus = sum(p for (m_corr, _), p in dist.items() if m_corr >= alpha)
    h_minus = sum(p for (_, n_wrong), p in dist.items() if n_wrong >= alpha)
    return {"distribution": dist, "e_k": ek, "h_plus": h_plus, "h_minus": h_minus}
