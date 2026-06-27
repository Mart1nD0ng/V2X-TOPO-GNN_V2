"""Basin first-hitting outcome and mutually-exclusive run probabilities (spec §4).

The run outcome is the FIRST basin entered along the macrostate trajectory ``(C_r, W_r)``,
``r = 0 … R_d``; if no basin is entered by the deadline ``R_d`` the run is a *deadline* miss:

    τ_C = inf{r : (C_r,W_r) ∈ B_C},   τ_W, τ_S analogously,
    P_C        = P(τ_C < min(τ_W, τ_S, R_d+1))
    F_wrong    = P(τ_W < min(τ_C, τ_S, R_d+1))
    F_split    = P(τ_S < min(τ_C, τ_W, R_d+1))
    F_deadline = P(min(τ_C, τ_W, τ_S) > R_d)

Because the basins are disjoint per epoch (profile invariant ``ρ_s > 1 − ρ_f``), at the
first-hit epoch exactly one basin is entered, so the four outcomes partition the sample
space and their probabilities sum to 1 with no tie-break.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .basins import CORRECT, WRONG, SPLIT, NONE, basin_label

__all__ = ["FirstHitting", "first_hitting_outcome", "basin_outcome_probabilities",
           "basin_first_hitting_batched", "confirm_time_stats"]

DEADLINE = "deadline"


@dataclass(frozen=True)
class FirstHitting:
    outcome: str   # "correct" | "wrong" | "split" | "deadline"
    tau: int       # first-hit epoch r; R_d + 1 if deadline (no basin reached)


def first_hitting_outcome(C_path: torch.Tensor, W_path: torch.Tensor, profile) -> FirstHitting:
    """First basin hit along one ``(C_path, W_path)`` trajectory (spec §4).

    Args:
        C_path, W_path: ``[R+1]`` macrostate masses at epochs ``r = 0 … R`` (``r=0`` initial).
        profile: supplies ``ρ_f, ρ_s`` and the deadline ``R_d = max_poll_epochs``.

    Returns:
        :class:`FirstHitting`. Epochs beyond ``R_d`` are not examined (deadline already missed).
    """
    if C_path.shape != W_path.shape or C_path.ndim != 1:
        raise ValueError("C_path and W_path must be matching 1-D [R+1] tensors")
    R_d = profile.max_poll_epochs
    horizon = min(C_path.numel() - 1, R_d)
    for r in range(horizon + 1):
        label = basin_label(C_path[r], W_path[r], profile)
        if label != NONE:
            return FirstHitting(outcome=label, tau=r)
    return FirstHitting(outcome=DEADLINE, tau=R_d + 1)


def basin_first_hitting_batched(C_paths: torch.Tensor, W_paths: torch.Tensor, profile):
    """Vectorised per-trial first-hitting over a ``[T, R+1]`` batch (spec §4).

    Returns ``(outcome_code, tau)`` integer tensors of shape ``[T]`` with codes
    ``0=correct, 1=wrong, 2=split, 3=deadline`` and ``tau`` the first-hit epoch (``R_d+1`` on
    deadline). Basins are disjoint per epoch, so ``argmax`` of the (single-True-per-epoch) hit
    mask gives the unambiguous first-hit. ``O(T·R)`` tensor ops, no Python loop.
    """
    if C_paths.shape != W_paths.shape or C_paths.ndim != 2:
        raise ValueError("C_paths and W_paths must be matching 2-D [T, R+1] tensors")
    rho_f = profile.correct_basin_mass
    rho_s = profile.split_basin_mass
    R_d = profile.max_poll_epochs
    H = min(C_paths.shape[1] - 1, R_d)
    C = C_paths[:, : H + 1]
    W = W_paths[:, : H + 1]
    T = C.shape[0]
    in_c = C >= rho_f
    in_w = W >= rho_f
    in_s = (C >= rho_s) & (W >= rho_s)
    hit = in_c | in_w | in_s                         # [T, H+1] (disjoint ⇒ ≤1 basin per epoch)
    hit_any = hit.any(dim=1)                          # [T]
    first_r = torch.argmax(hit.to(torch.int64), dim=1)  # first True epoch (0 if none; guarded)
    ar = torch.arange(T, device=C.device)
    code = torch.full((T,), 3, dtype=torch.int64, device=C.device)  # default deadline
    code = torch.where(hit_any & in_c[ar, first_r], torch.zeros_like(code), code)
    code = torch.where(hit_any & in_w[ar, first_r], torch.ones_like(code), code)
    code = torch.where(hit_any & in_s[ar, first_r], torch.full_like(code, 2), code)
    tau = torch.where(hit_any, first_r, torch.full_like(first_r, R_d + 1))
    return code, tau


def confirm_time_stats(outcome_code: torch.Tensor, tau: torch.Tensor, profile) -> dict:
    """Confirmation-time statistics over CORRECT runs (spec §5.3, §6; P0-A delay fix).

    The confirmation time of a correct run is ``T_confirm = τ_C · Δ_poll`` (the first-hit epoch
    times the fixed poll window). This sidesteps the legacy survival-sum off-by-one (P0-A):
    a run that finalises on the FIRST epoch has ``τ=1`` ⇒ ``T_confirm = Δ_poll`` (one window,
    never 0). Returns the mean and the upper-tail ``CVaR_q(T_confirm | O=C)`` — the objective
    of spec §6 — plus the survival-sum cross-check ``E[τ] = Σ_{r≥0} P(τ>r)``.

    Args:
        outcome_code: ``[T]`` int (0=correct,1=wrong,2=split,3=deadline) from
            :func:`basin_first_hitting_batched`.
        tau: ``[T]`` first-hit epochs (same source).
        profile: supplies ``Δ_poll`` (``poll_window_s``) and ``q`` (``latency_quantile``).

    Returns dict with ``mean_confirm_s``, ``cvar_confirm_s``, ``num_correct``,
    ``mean_tau`` and ``survival_sum_tau`` (must equal ``mean_tau``).
    """
    dpoll = profile.poll_window_s
    q = profile.latency_quantile
    correct = outcome_code == 0
    tau_c = tau[correct].to(torch.float64)
    n = int(tau_c.numel())
    if n == 0:
        return {"mean_confirm_s": float("nan"), "cvar_confirm_s": float("nan"),
                "num_correct": 0, "mean_tau": float("nan"), "survival_sum_tau": float("nan")}
    mean_tau = float(tau_c.mean())
    # survival-sum identity E[τ] = Σ_{r=0}^{∞} P(τ > r); finite since τ ≤ R_d+1
    rmax = int(tau_c.max())
    survival = sum(float((tau_c > r).to(torch.float64).mean()) for r in range(rmax + 1))
    t_confirm = tau_c * dpoll
    var = torch.quantile(t_confirm, q)
    tail = t_confirm[t_confirm >= var]
    cvar = float(tail.mean()) if tail.numel() else float(var)
    return {
        "mean_confirm_s": float(t_confirm.mean()),
        "cvar_confirm_s": cvar,
        "num_correct": n,
        "mean_tau": mean_tau,
        "survival_sum_tau": survival,
    }


def basin_outcome_probabilities(C_paths: torch.Tensor, W_paths: torch.Tensor, profile) -> dict:
    """Mutually-exclusive outcome probabilities over a batch of trajectories (spec §4).

    Args:
        C_paths, W_paths: ``[T, R+1]`` macrostate trajectories for ``T`` independent runs
            (e.g. dynamic-MC trials, or exact joint-chain enumerated paths to be weighted).
        profile: supplies ``ρ_f, ρ_s, R_d``.

    Returns:
        dict with ``P_correct, F_wrong, F_split, F_deadline`` (summing to 1), integer
        ``counts``, ``tau_correct_mean`` (mean first-hit epoch of correct runs, ``nan`` if
        none) and the per-trial ``outcome_code`` / ``tau`` tensors — the basis for
        ``T_confirm`` and its CVaR (spec §6).
    """
    code, tau = basin_first_hitting_batched(C_paths, W_paths, profile)
    T = int(code.numel())
    inv_T = 1.0 / T if T else 0.0
    counts = {CORRECT: int((code == 0).sum()), WRONG: int((code == 1).sum()),
              SPLIT: int((code == 2).sum()), DEADLINE: int((code == 3).sum())}
    tau_c = tau[code == 0]
    tau_mean = float(tau_c.to(torch.float64).mean()) if tau_c.numel() else float("nan")
    return {
        "P_correct": counts[CORRECT] * inv_T,
        "F_wrong": counts[WRONG] * inv_T,
        "F_split": counts[SPLIT] * inv_T,
        "F_deadline": counts[DEADLINE] * inv_T,
        "counts": counts,
        "tau_correct_mean": tau_mean,
        "outcome_code": code,
        "tau": tau,
        "num_trials": T,
    }
