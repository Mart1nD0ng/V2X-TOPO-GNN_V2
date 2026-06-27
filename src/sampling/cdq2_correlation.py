"""CDQ 2.0 differentiable pairwise-correlation training objective (spec §11).

The overlapping environment supplies a deployment-observable evidence-error correlation matrix
``R_{jl}`` (``src/environment/overlapping_evidence.py::overlapping_pairwise_correlation_matrix``).
The CDQ 2.0 query should learn to AVOID co-selecting correlated peers; the differentiable cost is

    L_corr = sum_i sum_{j<l} pi^(2)_{i,jl} R_{jl}  =  E_{S~kDPP}[ sum_{j<l in S} R_{jl} ]      (spec §11)

i.e. the expected within-selected-set total correlation, where ``pi^(2)_{jl} = P(j,l in S)`` is the
CDQ 2.0 ``k``-DPP PAIRWISE inclusion. Minimising it pushes the diversity head ``(Z, eta)`` to spread
the poll across uncorrelated peers -- the lever a marginal-only (ESP / region-aware) policy lacks
(Mechanism Contract C3).

**First-order differentiability (the gradient-gate requirement).** ``pi^(2)`` must be once
differentiable -- the training loss backprops through it exactly once; computing it as a SECOND
derivative of ``e_k`` (autograd-of-autograd) would make that a third-order op, fragile at degenerate
eigenvalues. We instead use the Schur-complement quotient identity (a determinant identity, not an
autograd second derivative):

    sum_{S superset of T, |S|=k} det(L_S) = det(L_T) . e_{k-|T|}(lambda(L / L_T)),

where ``L / L_T`` is the Schur complement of the principal block ``L_T``. Hence

    pi^(2)_{jl} = det(L_{{j,l}}) . e_{k-2}(lambda(L / L_{{j,l}})) / e_k(lambda(L)),
    pi_j       = L_{jj}        . e_{k-1}(lambda(L / L_{{j}}))   / e_k(lambda(L)),       (diagonal)

each computed with ``eigvalsh`` + the cancellation-free positive elementary-symmetric recursion of
``cdq2_kernel`` (smooth first-order gradient: ``e_{k-t}`` is a symmetric function of the Schur
eigenvalues, so its eigenvalue-backward has no ``1/(lambda_i-lambda_j)`` gap denominators). NO
``float()`` / ``.item()`` / ``.detach()`` on the differentiable path (spec §11 prohibition).

Exactness boundary: exact ``pi^(2)`` / ``L_corr`` under the CDQ 2.0 kernel for ``2 <= k <= d``;
validated against brute-force subset enumeration (< 1e-10) and finite-difference gradients
(rel-err < 1e-4). Per source ``O(d^2 (d-2)^3)`` (a Schur eigvalsh per pair on the ``d x d`` kernel,
never ``N x N``) -- polynomial in the bounded candidate degree, linear in the number of polling
sources; the constant-factor reduction to ``O(d^3)`` all-pairs is a scale-phase optimisation.
"""

from __future__ import annotations

from itertools import combinations

import torch

from .cdq2_kernel import _as_eta, _elem_sym_positive, cdq2_unit_normalize

__all__ = [
    "cdq2_pairwise_inclusion",
    "cdq2_correlation_cost",
    "bruteforce_cdq2_pairwise_inclusion",
]


def _build_L(a: torch.Tensor, Zn: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """``L = D^{1/2}(I + eta Z Z^T) D^{1/2}`` (single source ``[d, d]``, differentiable)."""
    d = a.shape[-1]
    root = a.clamp_min(0.0).sqrt()
    G = torch.eye(d, dtype=a.dtype, device=a.device) + e * (Zn @ Zn.transpose(-1, -2))
    L = root.unsqueeze(-1) * G * root.unsqueeze(-2)
    return 0.5 * (L + L.transpose(-1, -2))


def _conditioned_numerator(L: torch.Tensor, T: list[int], k: int) -> torch.Tensor:
    """``det(L_T) . e_{k-|T|}(lambda(L / L_T))`` = ``sum_{S superset of T, |S|=k} det(L_S)``.

    First-order differentiable: 2x2/1x1 inverse + eigvalsh of the SPD Schur complement +
    positive ESP recursion.
    """
    d = L.shape[-1]
    t = len(T)
    rest = [i for i in range(d) if i not in T]
    Tt = torch.tensor(T, dtype=torch.long, device=L.device)
    Rr = torch.tensor(rest, dtype=torch.long, device=L.device)
    L_TT = L.index_select(0, Tt).index_select(1, Tt)                 # [t, t]
    det_T = torch.det(L_TT)
    if k - t == 0:
        return det_T
    L_RR = L.index_select(0, Rr).index_select(1, Rr)                 # [d-t, d-t]
    L_RT = L.index_select(0, Rr).index_select(1, Tt)                 # [d-t, t]
    schur = L_RR - L_RT @ torch.linalg.solve(L_TT, L_RT.transpose(-1, -2))
    schur = 0.5 * (schur + schur.transpose(-1, -2))
    lam = torch.linalg.eigvalsh(schur).clamp_min(0.0)
    return det_T * _elem_sym_positive(lam, k - t)


def cdq2_pairwise_inclusion(a: torch.Tensor, Z: torch.Tensor, eta, k: int) -> torch.Tensor:
    """``Pi`` with ``Pi[j,l] = P(j,l in S)`` (off-diagonal) and ``Pi[j,j] = P(j in S) = pi_j``
    (diagonal) for the CDQ 2.0 ``k``-DPP (single source ``a`` ``[d]``). Differentiable (first order).
    """
    if a.ndim != 1:
        raise ValueError("cdq2_pairwise_inclusion expects a single source (a is [d])")
    d = a.shape[0]
    if k < 1 or k > d:
        raise ValueError(f"k={k} must satisfy 1 <= k <= d = {d}")
    if bool(torch.any(a.detach() <= 0).cpu()):
        raise ValueError("quality a must be > 0")
    Zn = cdq2_unit_normalize(Z)
    e = _as_eta(eta, a)
    # Factor out the mean quality: L scales by 1/s, det(L_T)/e_{k-t}/e_k all scale homogeneously,
    # so EVERY pi^(2)=numerator/e_k ratio is exactly scale-invariant — overflow-safe at large a.
    s = a.mean().clamp_min(torch.finfo(a.dtype).tiny)
    L = _build_L(a / s, Zn, e)
    lam = torch.linalg.eigvalsh(L).clamp_min(0.0)
    ek = _elem_sym_positive(lam, k)                                  # e_k(lambda(L)) (scaled domain)
    Pi = a.new_zeros((d, d))
    for j in range(d):
        Pi = Pi + _diag_set(Pi, j, _conditioned_numerator(L, [j], k) / ek)
    if k >= 2:
        for j, l in combinations(range(d), 2):
            val = _conditioned_numerator(L, [j, l], k) / ek
            Pi = Pi + _offdiag_set(Pi, j, l, val)
    return Pi


def _diag_set(Pi: torch.Tensor, j: int, val: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(Pi)
    out[j, j] = val
    return out


def _offdiag_set(Pi: torch.Tensor, j: int, l: int, val: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(Pi)
    out[j, l] = val
    out[l, j] = val
    return out


def cdq2_correlation_cost(a: torch.Tensor, Z: torch.Tensor, eta, R: torch.Tensor, k: int) -> torch.Tensor:
    """``L_corr = sum_{j<l} pi^(2)_{jl} R_{jl}`` for one source (spec §11). Differentiable scalar.

    ``R`` is the ``[d, d]`` symmetric evidence-error correlation (diagonal ignored). Equals the
    expected within-selected-set total correlation ``E_{S~kDPP}[ sum_{j<l in S} R_{jl} ]``.
    """
    if R.shape != (a.shape[0], a.shape[0]):
        raise ValueError("R must be [d, d] matching a")
    Pi = cdq2_pairwise_inclusion(a, Z, eta, k)
    Rs = 0.5 * (R + R.transpose(-1, -2))
    # sum_{j<l} = 1/2 sum_{j!=l}; the diagonal of Pi (single inclusion) is excluded by R's zero diag
    off = Rs - torch.diag(torch.diagonal(Rs))
    return 0.5 * (Pi * off).sum()


# --------------------------------------------------------------------------------------------
# brute-force reference (single source, tests only)
# --------------------------------------------------------------------------------------------

def bruteforce_cdq2_pairwise_inclusion(a: torch.Tensor, Z: torch.Tensor, eta, k: int) -> torch.Tensor:
    """``Pi[j,l] = sum_{S∋j,l} det(L_S) / e_k`` (diag = single inclusion) by enumeration (tests-only)."""
    d = a.shape[0]
    Zn = cdq2_unit_normalize(Z)
    e = _as_eta(eta, a)
    dets = {}
    ek = 0.0
    for S in combinations(range(d), k):
        idx = torch.tensor(S)
        Zs = Zn[idx]
        det = float(a[idx].prod() * torch.det(torch.eye(len(S), dtype=a.dtype) + e * Zs @ Zs.T))
        dets[S] = max(det, 0.0)
        ek += dets[S]
    Pi = a.new_zeros((d, d))
    for S, det in dets.items():
        for j in S:
            Pi[j, j] += det
        for j, l in combinations(S, 2):
            Pi[j, l] += det
            Pi[l, j] += det
    return Pi / ek
