"""Participation-weighted macrostate consensus quantities ``C_r, W_r, U_r`` (spec §3).

After polling epoch ``r`` the macrostate is, over the exogenous participation measure ``ω``
(``src.metrics.participation``),

    C_r = Σ_i ω_i 1{i correctly finalized}      (correct decided mass)
    W_r = Σ_i ω_i 1{i wrongly  finalized}       (wrong   decided mass)
    U_r = 1 − C_r − W_r                          (undecided mass)

These live in ``[0,1]`` and DO NOT mechanically couple to ``N`` through a node union: under
uniform ``ω`` they are population *fractions*, so replicating the population leaves them
unchanged (the explicit anti-``1-(1-p)^N`` property — forbidden shortcut #1).

``final_state`` encodes each node's finalization as an integer in ``{-1, 0, +1}``
(``+1`` correct, ``-1`` wrong, ``0`` undecided); it may carry any leading batch/time dims.
A retained fixed-``N`` strict-disagreement audit (spec §4) is also provided.
"""

from __future__ import annotations

import torch

__all__ = ["macrostate_occupancy", "strict_disagreement",
           "pairwise_disagreement", "region_disagreement"]


def macrostate_occupancy(final_state: torch.Tensor, omega: torch.Tensor):
    """Return ``(C, W, U)`` masses for a finalization tensor over the participation measure.

    Args:
        final_state: ``[..., N]`` integer tensor in ``{-1, 0, +1}`` (trailing dim = nodes).
        omega: ``[N]`` participation weights (``ω_i ≥ 0``, ``Σ ω_i = 1``).

    Returns:
        ``(C, W, U)`` each of shape ``final_state.shape[:-1]``.
    """
    if omega.ndim != 1:
        raise ValueError("omega must be 1-D [N]")
    if final_state.shape[-1] != omega.shape[0]:
        raise ValueError("final_state trailing dim must equal N = omega length")
    w = omega.to(dtype=torch.float64 if not omega.is_floating_point() else omega.dtype)
    correct = (final_state == 1).to(w.dtype)
    wrong = (final_state == -1).to(w.dtype)
    C = correct @ w
    W = wrong @ w
    U = (1.0 - C - W).clamp_min(0.0)
    return C, W, U


def pairwise_disagreement(C: torch.Tensor, W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Continuous disagreement diagnostic ``D_pair = 2 C W / ((C+W)^2 + eps)`` (spec §3).

    The probability that two participants drawn (by mass) from the DECIDED population disagree.
    Zero when all decided agree (one of ``C, W`` is 0), maximal (→ 1) at ``C = W``. A
    diagnostic/auxiliary signal — it does NOT replace the basin outcome.
    """
    return 2.0 * C * W / ((C + W) ** 2 + eps)


def region_disagreement(final_state: torch.Tensor, omega: torch.Tensor,
                        region_of: torch.Tensor, num_regions: int | None = None,
                        eps: float = 1e-12) -> torch.Tensor:
    """Regional disagreement ``D_region = Σ_g Ω_g [(C_g − C)^2 + (W_g − W)^2]`` (spec §3).

    ``Ω_g = Σ_{i∈g} ω_i`` is region ``g``'s participation mass and ``C_g, W_g`` its
    within-region decided fractions; the global ``C, W`` are the participation-weighted means.
    Measures how heterogeneously opinion is distributed across regions (a weak-cut / handoff
    diagnostic). ``final_state``: ``[..., N]``; returns shape ``final_state.shape[:-1]``.
    """
    if omega.ndim != 1 or region_of.ndim != 1:
        raise ValueError("omega and region_of must be 1-D [N]")
    N = omega.shape[0]
    if final_state.shape[-1] != N or region_of.shape[0] != N:
        raise ValueError("final_state trailing dim, omega and region_of must all be N")
    G = int(region_of.max()) + 1 if num_regions is None else int(num_regions)
    w = omega.to(torch.float64 if not omega.is_floating_point() else omega.dtype)
    C, W, _ = macrostate_occupancy(final_state, w)                    # [...] global
    correct = (final_state == 1).to(w.dtype)                          # [..., N]
    wrong = (final_state == -1).to(w.dtype)
    batch_shape = final_state.shape[:-1]
    out = torch.zeros(batch_shape, dtype=w.dtype, device=final_state.device)
    for g in range(G):
        m = region_of == g                                           # [N]
        Omega_g = w[m].sum()
        if float(Omega_g) <= 0:
            continue
        wg = w[m]
        Cg = (correct[..., m] * wg).sum(dim=-1) / Omega_g            # within-region fractions
        Wg = (wrong[..., m] * wg).sum(dim=-1) / Omega_g
        out = out + Omega_g * ((Cg - C) ** 2 + (Wg - W) ** 2)
    return out


def strict_disagreement(final_state: torch.Tensor) -> torch.Tensor:
    """Fixed-``N`` strict safety audit ``F_strict``: ``∃ i,j`` decided with ``Y_i ≠ Y_j`` (spec §4).

    Returns a boolean tensor of shape ``final_state.shape[:-1]`` — ``True`` iff at least one
    correctly-finalized AND one wrongly-finalized node coexist. Reported separately from the
    cross-scale basin outcome (it is NOT the headline metric).
    """
    any_correct = (final_state == 1).any(dim=-1)
    any_wrong = (final_state == -1).any(dim=-1)
    return any_correct & any_wrong
