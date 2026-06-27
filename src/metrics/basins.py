"""Macrostate consensus basins (spec §4).

Given the basin masses ``ρ_f`` (decisive correct/wrong) and ``ρ_s`` (split) from a
:class:`~src.config.service_profile.ConsensusServiceProfile`,

    B_C = {C_r ≥ ρ_f},   B_W = {W_r ≥ ρ_f},   B_S = {C_r ≥ ρ_s and W_r ≥ ρ_s}.

The profile invariant ``ρ_s > 1 − ρ_f`` (with ``ρ_f > ½``, ``ρ_s < ½``) guarantees the three
basins are pairwise DISJOINT: no ``(C, W)`` with ``C + W ≤ 1`` lies in two basins at once, so
the first-hitting outcome is unambiguous (no within-epoch tie-break needed).
"""

from __future__ import annotations

import torch

__all__ = ["basin_membership", "basin_label", "basins_disjoint",
           "CORRECT", "WRONG", "SPLIT", "NONE"]

CORRECT, WRONG, SPLIT, NONE = "correct", "wrong", "split", "none"


def basin_membership(C: torch.Tensor, W: torch.Tensor, profile):
    """Return ``(in_correct, in_wrong, in_split)`` boolean tensors (spec §4)."""
    rho_f = profile.correct_basin_mass
    rho_s = profile.split_basin_mass
    in_c = C >= rho_f
    in_w = W >= rho_f
    in_s = (C >= rho_s) & (W >= rho_s)
    return in_c, in_w, in_s


def basin_label(C: torch.Tensor, W: torch.Tensor, profile) -> str:
    """Classify a single ``(C, W)`` point into ``correct | wrong | split | none`` (spec §4).

    Disjointness (profile invariant) makes this well-defined; if a degenerate profile ever
    produced overlap, this raises rather than silently picking one.
    """
    rho_f = profile.correct_basin_mass
    rho_s = profile.split_basin_mass
    c = float(C)
    w = float(W)
    hits = []
    if c >= rho_f:
        hits.append(CORRECT)
    if w >= rho_f:
        hits.append(WRONG)
    if c >= rho_s and w >= rho_s:
        hits.append(SPLIT)
    if len(hits) > 1:
        raise ValueError(
            f"basins overlap at (C={c}, W={w}) under rho_f={rho_f}, rho_s={rho_s}; "
            "profile invariant rho_s > 1 - rho_f violated")
    return hits[0] if hits else NONE


def basins_disjoint(profile, *, n_grid: int = 201) -> bool:
    """Verify the three basins are pairwise disjoint over the feasible simplex ``C + W ≤ 1``.

    Exact algebraically (the invariant ``ρ_f + ρ_s > 1`` implies disjointness); this grid
    check is a cheap empirical confirmation used by the gate.
    """
    rho_f = profile.correct_basin_mass
    rho_s = profile.split_basin_mass
    grid = torch.linspace(0.0, 1.0, n_grid, dtype=torch.float64)
    for c in grid.tolist():
        for w in grid.tolist():
            if c + w > 1.0 + 1e-9:
                continue
            n = (c >= rho_f) + (w >= rho_f) + (c >= rho_s and w >= rho_s)
            if n > 1:
                return False
    return True
