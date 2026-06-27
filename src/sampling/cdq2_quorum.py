"""CDQ 2.0 heterogeneous determinantal quorum ``P(m,n)`` (spec §10).

Under the CDQ 2.0 query (``L = D^{1/2}(I + eta Z Z^T) D^{1/2}``, ``src/sampling/cdq2_kernel.py``)
each polled peer ``j`` returns correct / wrong / no-response with edge probabilities
``p^+_j, p^-_j, p^0_j``. With ``g_j(x,y) = p^0_j + p^+_j x + p^-_j y`` and ``G = diag(g_j)`` the
joint generating function over (#selected, #correct, #wrong) is (principal-minor expansion)

    det(I + z L G) = sum_S z^{|S|} det(L_S) prod_{j in S} g_j(x,y),

so the exact probability of ``m`` correct and ``n`` wrong among the ``k`` selected peers is

    P(m,n) = [z^k x^m y^n] det(I + z L G) / e_k(lambda(L)).                          (spec §10)

This is the CDQ-2.0 generalisation of the §5 elementary-symmetric quorum; it reduces EXACTLY to
the ESP quorum when ``eta = 0`` (``L = diag(a)``).

Implementation (exact, differentiable, no ``N x N``). ``[z^k] det(I + z L G)`` evaluated at a
numeric ``(x_a, y_b)`` is ``e_k(lambda(L G))`` with ``L G`` similar to ``(I + eta Z Z^T) diag(c)``,
``c_j = a_j g_j(x_a, y_b) >= 0`` -- i.e. EXACTLY the CDQ 2.0 normaliser evaluated at the deformed
quality ``c`` (the stable eigenvalue route ``_cdq2_log_ek_from_c``). We evaluate it on the
``(k+1) x (k+1)`` integer grid and recover the ``P(m,n)`` coefficients by 2-D inverse-Vandermonde
(the same exact polynomial-interpolation step as the ``L = BB^T`` quorum). Because the grid values
span orders of magnitude (largest at ``x = y = k``), the grid is assembled in the log domain and a
single common offset is removed before the linear interpolation -- ``P(m,n) = N(m,n) / sum N`` is
invariant to that common scale, so this is exact and overflow-free.

The normaliser ``e_k(lambda(L)) = sum_{m,n} numerator(m,n)`` (the generating function at
``x = y = 1``, ``g = 1``), so ``sum_{m,n} P(m,n) = 1`` by construction.

Exactness boundary: exact ``P(m,n)`` under the CDQ 2.0 kernel; validated against the explicit
subset x assignment enumeration AND the principal-minor reference on ``B_tilde = D^{1/2}[I|sqrt(eta)Z]``
(< 1e-10), and against the ESP quorum at ``eta = 0``. Per source ``O((k+1)^2 (d^3 + d^2 r))`` --
the eigenvalue route's ``d^3`` is on the ``d x d`` per-source kernel (bounded candidate degree),
never an ``N x N`` matrix.
"""

from __future__ import annotations

import torch

from .cdq2_kernel import _as_eta, _cdq2_log_ek_from_c, cdq2_unit_normalize
from .determinantal_quorum import QuorumDecisionCDQ, _validate, _vandermonde_inverse

__all__ = [
    "cdq2_quorum_distribution",
    "cdq2_quorum_decision",
]


def cdq2_quorum_distribution(
    a: torch.Tensor,
    Z: torch.Tensor,
    eta,
    p_correct: torch.Tensor,
    p_wrong: torch.Tensor,
    k: int,
    *,
    tol: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``P_i(m,n)`` for ``0 <= m, n``, ``m + n <= k`` (spec §10). Returns ``[..., k+1, k+1]``.

    Args:
        a: ``[..., d]`` per-candidate quality ``a_j = exp(s_j) > 0``.
        Z: ``[..., d, r]`` diversity embedding (unit-normalised internally).
        eta: diversity strength ``>= 0`` (scalar or ``[...]``).
        p_correct, p_wrong: ``[..., d]`` per-candidate ``p^+_j, p^-_j`` (``p^0 = 1 - p^+ - p^-``).
        k: subset size; ``1 <= k <= d``.
    """
    if a.ndim < 1:
        raise ValueError("a must be [..., d]")
    d = a.shape[-1]
    if k < 1 or k > d:
        raise ValueError(f"k={k} must satisfy 1 <= k <= d = {d}")
    if p_correct.shape != a.shape or p_wrong.shape != a.shape:
        raise ValueError("p_correct / p_wrong must be [..., d] matching a")
    bad = (a.detach() <= 0) & mask if mask is not None else (a.detach() <= 0)
    if bool(torch.any(bad).cpu()):
        raise ValueError("quality a must be > 0 on every real (unmasked) candidate")
    pp = p_correct.clamp(0.0, 1.0)
    pm = p_wrong.clamp(0.0, 1.0)
    if bool(torch.any((p_correct.detach() < -tol) | (p_wrong.detach() < -tol)).cpu()):
        raise ValueError("p_correct / p_wrong must be >= 0")
    if bool(torch.any((pp + pm).detach() > 1.0 + tol).cpu()):
        raise ValueError("p_correct + p_wrong must be <= 1")
    p0 = (1.0 - pp - pm).clamp(0.0, 1.0)

    Zn = cdq2_unit_normalize(Z)
    e = _as_eta(eta, a)
    batch = a.shape[:-1]

    # logD[..., g_x, g_y] = log e_k(lambda(L G)) with g_j = p0 + pp*x + pm*y, c_j = a_j g_j
    grid = torch.arange(k + 1, dtype=a.dtype, device=a.device)
    log_layers = []
    for xa in range(k + 1):
        row = []
        for yb in range(k + 1):
            g = p0 + pp * grid[xa] + pm * grid[yb]            # [..., d] >= 0
            c = a * g                                          # [..., d] >= 0
            row.append(_cdq2_log_ek_from_c(c, Zn, e, k, mask=mask))   # [...]
        log_layers.append(torch.stack(row, dim=-1))            # [..., k+1]
    logD = torch.stack(log_layers, dim=-2)                     # [..., k+1, k+1]

    # remove a single common offset (invariant under P = N / sum N) so D stays O(1)
    finite = torch.isfinite(logD)
    neg_inf = logD.new_full((), float("-inf"))
    M = torch.where(finite, logD, neg_inf).amax(dim=(-1, -2), keepdim=True)
    D = torch.exp(logD - M)                                    # [..., k+1, k+1], 0 where logD=-inf

    Vinv = _vandermonde_inverse(k, str(a.dtype).split(".")[-1]).to(device=a.device)
    N = Vinv @ D @ Vinv.transpose(-1, -2)                      # [..., k+1, k+1]
    tri = (torch.arange(k + 1).unsqueeze(1) + torch.arange(k + 1).unsqueeze(0)) <= k
    N = N * tri.to(dtype=a.dtype, device=a.device)
    ek = N.sum(dim=(-1, -2), keepdim=True)
    if bool(torch.any(ek.detach() <= 0).cpu()):
        raise ValueError("e_k(lambda(L)) <= 0: fewer than k usable candidates (apply §7.2 upstream)")
    return (N / ek).clamp_min(0.0)


def cdq2_quorum_decision(
    a: torch.Tensor,
    Z: torch.Tensor,
    eta,
    p_correct: torch.Tensor,
    p_wrong: torch.Tensor,
    k: int,
    alpha: int,
    *,
    tol: float = 1e-6,
    mask: torch.Tensor | None = None,
) -> QuorumDecisionCDQ:
    """Correct / wrong / no-quorum probabilities ``(h^+, h^-, h^0)`` under the CDQ 2.0 quorum."""
    _validate(k, alpha)
    P = cdq2_quorum_distribution(a, Z, eta, p_correct, p_wrong, k, tol=tol, mask=mask)
    h_plus = P[..., alpha:, :].sum(dim=(-1, -2))
    h_minus = P[..., :, alpha:].sum(dim=(-1, -2))
    h_zero = torch.clamp(1.0 - h_plus - h_minus, 0.0, 1.0)
    return QuorumDecisionCDQ(h_plus=h_plus, h_minus=h_minus, h_zero=h_zero, distribution=P)
