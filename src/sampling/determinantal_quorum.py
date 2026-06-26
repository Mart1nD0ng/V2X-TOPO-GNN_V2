"""CDQ exact heterogeneous quorum law (spec §9.5 -- G5).

Under the correlation-aware ``k``-DPP query (``L = B B^T``, spec §9.4) each polled peer ``j``
returns correct/wrong/no-response with edge probabilities ``p^+_{ij}, p^-_{ij}, p^0_{ij}``.
With ``g_{ij}(x,y) = p^0 + p^+ x + p^- y`` and ``D_g = diag(g_{ij})`` the joint generating
function over (#selected, #correct, #wrong) is (principal-minor expansion, spec §9.5)

    det(I + z L D_g) = sum_S z^{|S|} det(L_S) prod_{j in S} g_{ij}(x,y),

so the exact probability of ``m`` correct and ``n`` wrong among the ``k`` selected peers is

    P_i(m,n) = [z^k x^m y^n] det(I + z L D_g) / e_k(lambda(L)).                     (spec §9.5)

This is the heterogeneous correct/wrong/no-response quorum under the CDQ query -- the exact
generalisation of the §5 elementary-symmetric ``quorum_dp`` (which it recovers when the
kernel is diagonal).

Implementation (exact, low-rank, differentiable -- NO sampling/straight-through). By the
matrix-determinant lemma ``det(I_d + z B B^T D_g) = det(I_r + z B^T D_g B)``, the cost falls
on the small rank ``r``. The ``[z^k]`` coefficient is

    [z^k] det(I_r + z M(x,y)) = sum_{|T|=k} det(M_T(x,y)) = e_k(lambda(M(x,y))),

where ``M(x,y) = B^T D_g B = C0 + C+ x + C- y`` is ``r x r`` with entries linear in
``(x,y)`` (``C. = B^T diag(p.) B``). This ``[z^k]`` is a bivariate polynomial of degree
``<= k`` in each of ``x,y``; we evaluate it on a ``(k+1) x (k+1)`` grid -- each grid point is
just ``e_k`` of a NUMERIC ``r x r`` matrix (the §9.4 principal-minor normaliser) -- and
recover its coefficients by the inverse-Vandermonde (exact 2-D polynomial interpolation, the
only error being float64 Vandermonde conditioning, kept ``< 1e-10`` for small ``k``). The
normaliser ``e_k(lambda(L)) = sum_{m,n} numerator(m,n)`` (the generating function at
``x=y=1``, where ``g=1``), so ``sum_{m,n} P_i(m,n) = 1`` by construction.

Complexity: ``O((k+1)^2 C(r,k) k^3)`` per source -- a constant in ``r,k``, hence ``O(N)``
across sources (no ``d x d`` kernel). Reduces EXACTLY to ``quorum_dp`` for a diagonal kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, product

import torch

from .dpp_query import _ek_principal_minors, low_rank_kernel

__all__ = [
    "QuorumDecisionCDQ",
    "determinantal_quorum_distribution",
    "determinantal_quorum_decision",
    "bruteforce_determinantal_quorum",
]


@dataclass(frozen=True)
class QuorumDecisionCDQ:
    h_plus: torch.Tensor          # [...] P(correct quorum)
    h_minus: torch.Tensor         # [...] P(wrong quorum)
    h_zero: torch.Tensor          # [...] P(no quorum)
    distribution: torch.Tensor    # [..., k+1, k+1] P(m correct, n wrong)


@lru_cache(maxsize=None)
def _vandermonde_inverse(k: int, dtype_str: str) -> torch.Tensor:
    """Inverse Vandermonde for grid points ``0..k`` (cached). ``[k+1, k+1]``."""
    dtype = getattr(torch, dtype_str)
    t = torch.arange(k + 1, dtype=dtype)
    V = torch.stack([t ** m for m in range(k + 1)], dim=1)   # V[a, m] = t_a^m
    return torch.linalg.inv(V)


def _validate(k: int, alpha: int) -> None:
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive int")
    if not isinstance(alpha, int) or alpha < 1 or alpha > k:
        raise ValueError("alpha must satisfy 1 <= alpha <= k")
    if 2 * alpha <= k:
        raise ValueError("alpha must be a strict majority: 2*alpha > k (so + / - quorums exclusive)")


def determinantal_quorum_distribution(
    B: torch.Tensor,
    p_correct: torch.Tensor,
    p_wrong: torch.Tensor,
    k: int,
    *,
    tol: float = 1e-6,
) -> torch.Tensor:
    """``P_i(m,n)`` for ``0 <= m,n``, ``m+n <= k`` (spec §9.5). Returns ``[..., k+1, k+1]``.

    Args:
        B: ``[..., d, r]`` kernel root (``L = B B^T``); ``k <= min(d, r)``.
        p_correct, p_wrong: ``[..., d]`` per-candidate ``p^+_{ij}, p^-_{ij}`` (``p^0=1-p^+-p^-``).
    """
    if B.ndim < 2:
        raise ValueError("B must be [..., d, r]")
    d, r = B.shape[-2], B.shape[-1]
    if k < 1 or k > min(d, r):
        raise ValueError(f"k={k} must satisfy 1 <= k <= min(d, r) = {min(d, r)}")
    if p_correct.shape != B.shape[:-1] or p_wrong.shape != B.shape[:-1]:
        raise ValueError("p_correct / p_wrong must be [..., d] matching B[..., d, r]")
    pp = p_correct.clamp(0.0, 1.0)
    pm = p_wrong.clamp(0.0, 1.0)
    if bool(torch.any((p_correct.detach() < -tol) | (p_wrong.detach() < -tol)).cpu()):
        raise ValueError("p_correct / p_wrong must be >= 0")
    if bool(torch.any((pp + pm).detach() > 1.0 + tol).cpu()):
        raise ValueError("p_correct + p_wrong must be <= 1")
    p0 = (1.0 - pp - pm).clamp(0.0, 1.0)

    # C. = B^T diag(p.) B  (r x r), so M(x,y) = C0 + C+ x + C- y
    def _BtDB(pvec: torch.Tensor) -> torch.Tensor:
        return B.transpose(-1, -2) @ (pvec.unsqueeze(-1) * B)
    C0, Cp, Cm = _BtDB(p0), _BtDB(pp), _BtDB(pm)

    # evaluate D[a,b] = [z^k] det(I_r + z M(x_a, y_b)) = e_k(lambda(M(x_a,y_b))) on the grid
    grid = torch.arange(k + 1, dtype=B.dtype, device=B.device)
    batch = B.shape[:-2]
    D = B.new_zeros((*batch, k + 1, k + 1))
    for a in range(k + 1):
        for b in range(k + 1):
            M = C0 + Cp * grid[a] + Cm * grid[b]           # [..., r, r]
            D[..., a, b] = _ek_principal_minors(M, k)
    # 2-D inverse-Vandermonde interpolation: N = Vinv D Vinv^T  -> numerator coeffs [m, n]
    Vinv = _vandermonde_inverse(k, str(B.dtype).split(".")[-1]).to(device=B.device)
    N = Vinv @ D @ Vinv.transpose(-1, -2)                  # [..., k+1, k+1]
    # zero the unreachable m+n>k coefficients (exactly 0 in theory; numerical dust here)
    mask = (torch.arange(k + 1).unsqueeze(1) + torch.arange(k + 1).unsqueeze(0)) <= k
    N = N * mask.to(dtype=B.dtype, device=B.device)
    ek = N.sum(dim=(-1, -2), keepdim=True)                 # e_k(lambda(L)) = sum of coeffs
    if bool(torch.any(ek.detach() <= 0).cpu()):
        raise ValueError("e_k(lambda(L)) <= 0: fewer than k usable candidates (apply §7.2 upstream)")
    return (N / ek).clamp_min(0.0)


def determinantal_quorum_decision(
    B: torch.Tensor,
    p_correct: torch.Tensor,
    p_wrong: torch.Tensor,
    k: int,
    alpha: int,
    *,
    tol: float = 1e-6,
) -> QuorumDecisionCDQ:
    """Correct / wrong / no-quorum probabilities ``(h^+, h^-, h^0)`` under the CDQ law."""
    _validate(k, alpha)
    P = determinantal_quorum_distribution(B, p_correct, p_wrong, k, tol=tol)
    h_plus = P[..., alpha:, :].sum(dim=(-1, -2))
    h_minus = P[..., :, alpha:].sum(dim=(-1, -2))
    h_zero = torch.clamp(1.0 - h_plus - h_minus, 0.0, 1.0)
    return QuorumDecisionCDQ(h_plus=h_plus, h_minus=h_minus, h_zero=h_zero, distribution=P)


def bruteforce_determinantal_quorum(
    B: torch.Tensor,
    p_correct,
    p_wrong,
    k: int,
    alpha: int,
) -> dict:
    """Reference ``P(m,n)`` by explicit enumeration (single source ``B`` ``[d, r]``, tests).

    Enumerates all ``C(d,k)`` ``k``-DPP subsets weighted by ``det(L_S)`` and, within each, all
    ``3^k`` correct/wrong/no-response assignments. ``O(C(d,k) 3^k)`` -- tiny configs only.
    """
    if B.ndim != 2:
        raise ValueError("bruteforce expects a single source B [d, r]")
    d = B.shape[0]
    pp = [float(x) for x in (p_correct.tolist() if torch.is_tensor(p_correct) else p_correct)]
    pm = [float(x) for x in (p_wrong.tolist() if torch.is_tensor(p_wrong) else p_wrong)]
    p0 = [1.0 - pp[j] - pm[j] for j in range(d)]
    coeff: dict[tuple[int, int], float] = {}
    ek = 0.0
    for S in combinations(range(d), k):
        Bs = B[list(S)]
        det = max(float(torch.det(Bs @ Bs.transpose(-1, -2))), 0.0)
        ek += det
        for assign in product((0, 1, 2), repeat=k):        # 0=none, 1=correct, 2=wrong
            w = det
            m = n = 0
            for slot, j in enumerate(S):
                a = assign[slot]
                if a == 0:
                    w *= p0[j]
                elif a == 1:
                    w *= pp[j]; m += 1
                else:
                    w *= pm[j]; n += 1
            coeff[(m, n)] = coeff.get((m, n), 0.0) + w
    dist = {mn: c / ek for mn, c in coeff.items()}
    h_plus = sum(p for (m, _), p in dist.items() if m >= alpha)
    h_minus = sum(p for (_, n), p in dist.items() if n >= alpha)
    return {"distribution": dist, "e_k": ek, "h_plus": h_plus, "h_minus": h_minus}
