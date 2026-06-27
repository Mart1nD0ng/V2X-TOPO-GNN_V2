"""Differentiable analytic macrostate basin-hazard surrogate (spec §12 Level-2; plan §6).

A differentiable training proxy for the participation-weighted basin first-hitting outcome
probabilities ``(P_correct, F_wrong, F_split, F_deadline)`` and the conditional confirmation
latency CVaR, built from the analytic episode's per-node per-epoch finalization marginals
``c_traj[r,i,q] = P(node i finalized CORRECT by epoch r | scenario Z=q)`` and ``w_traj``.

Design (design-panel synthesis, run ``wf_f37372a9-061`` — top two proposals both 27/30):

* **Gaussian-CLT occupancy.** Given ``Z=q`` the nodes are independent (spec §3.1), so
  ``C_r = sum_i omega_i 1{i correct}`` is a weighted sum of independent Bernoullis with
  ``mu_C = sum omega_i c_i``, ``sigma_C^2 = sum omega_i^2 c_i(1-c_i)``; the basin occupancy
  ``g_C(r,q) = P(C_r >= rho_f | q) ≈ Phi((mu_C - rho_f)/(beta_T sigma_C))`` — ``O(N)`` per
  ``(r,q)``, no ``N x N`` DP. ``beta_T`` is a temperature (anneal -> hard threshold).
* **Increment-share first-passage.** Cumulative occupancies are made monotone (``cummax``);
  the per-epoch *new* mass entering each basin is split by its marginal increment share, with a
  survival cap so the four outcomes **sum to 1 exactly** (telescoping) — honest first-hitting
  (which basin FIRST), not terminal membership.
* **Negative-correlation split.** ``C`` and ``W`` are negatively correlated given ``Z``
  (``C+W<=1``); ``g_S`` uses a first-order bivariate-normal (Mehler) correction
  ``g_C^s g_W^s + rho_CW phi(z_C) phi(z_W)`` (``rho_CW<=0`` ⇒ suppresses split), clamped.
* **CVaR.** Rockafellar form on the conditional first-hit-epoch law of correct runs.

Exactness boundary: EXACT only in the shared-latent model AND the variance→0 / temperature→0
(well-separated) limit. Elsewhere it is the labelled Level-2 approximation (CLT tail error
``O(1/sqrt(N_eff))``, Gaussian-closure split, soft-CVaR temperature). The dynamic MC owns the
headline magnitudes + real CVaR; this is the differentiable ranking/gradient proxy.
"""

from __future__ import annotations

import math

import torch

__all__ = ["basin_surrogate_outcomes", "confirm_cvar_surrogate"]

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def _normal_pdf(x: torch.Tensor) -> torch.Tensor:
    return _INV_SQRT_2PI * torch.exp(-0.5 * x * x)


def _increment(a: torch.Tensor) -> torch.Tensor:
    """Per-epoch non-negative increment of a monotone cumulative ``[H+1, Q]`` occupancy."""
    prev = torch.cat([torch.zeros_like(a[:1]), a[:-1]], dim=0)
    return (a - prev).clamp_min(0.0)


def basin_surrogate_outcomes(
    c_traj: torch.Tensor,
    w_traj: torch.Tensor,
    omega_part: torch.Tensor,
    omega_scn: torch.Tensor,
    profile,
    *,
    beta_T: float = 1.0,
    sigma_floor: float = 1e-4,
    eps: float = 1e-9,
    validate: bool = True,
    return_occupancy: bool = False,
) -> dict:
    """Differentiable basin first-hitting outcome probabilities from the marginal trajectories.

    Args:
        c_traj, w_traj: ``[R+1, N, Q]`` per-node per-epoch correct/wrong finalization marginals.
            PRECONDITION (guaranteed by the analytic episode, since a node finalizes correct XOR
            wrong XOR stays undecided): ``c_i(r,q) + w_i(r,q) <= 1`` per node. This makes the
            macrostate basins genuinely DISJOINT (``mu_C + mu_W <= 1`` ⇒ ``C`` and ``W`` cannot
            both reach ``rho_f > 1/2``), so the increment-share survival is well-formed.
        omega_part: ``[N]`` participation measure (``>=0``, sums to 1).
        omega_scn: ``[Q]`` shared-latent scenario weights (sums to 1) — REQUIRED for the four
            outcomes to sum to 1 (the per-scenario partition is exact; the mixture inherits it).
        profile: supplies ``rho_f, rho_s, R_d (max_poll_epochs)``.
        beta_T: occupancy temperature (1 = true CLT; larger = softer).
        validate: cheap guards on the two measures summing to 1 (set False on a hot path if the
            caller already guarantees it).

    Returns:
        dict with scalar ``P_correct, F_wrong, F_split, F_deadline`` (summing to 1), the per-epoch
        per-scenario correct first-passage mass ``p_correct_rq`` ``[H+1, Q]`` (for the CVaR), and —
        if ``return_occupancy`` — the cumulative occupancies ``g_correct/g_wrong/g_split`` ``[H+1, Q]``.
    """
    if c_traj.shape != w_traj.shape or c_traj.ndim != 3:
        raise ValueError("c_traj and w_traj must be matching [R+1, N, Q] tensors")
    if validate:
        if abs(float(omega_part.sum()) - 1.0) > 1e-6:
            raise ValueError("omega_part (participation) must sum to 1")
        if abs(float(omega_scn.sum()) - 1.0) > 1e-6:
            raise ValueError("omega_scn (scenario weights) must sum to 1 for the outcomes to sum to 1")
    rho_f = profile.correct_basin_mass
    rho_s = profile.split_basin_mass
    R_d = profile.max_poll_epochs
    Rp1, N, Q = c_traj.shape
    H = min(Rp1 - 1, R_d)
    c = c_traj[: H + 1].clamp(0.0, 1.0)                      # [H+1, N, Q]
    w = w_traj[: H + 1].clamp(0.0, 1.0)
    wp = omega_part.reshape(1, N, 1).to(c.dtype)
    wp2 = (omega_part.reshape(1, N, 1) ** 2).to(c.dtype)

    mu_C = (wp * c).sum(dim=1)                               # [H+1, Q]
    mu_W = (wp * w).sum(dim=1)
    s2_C = (wp2 * c * (1.0 - c)).sum(dim=1)
    s2_W = (wp2 * w * (1.0 - w)).sum(dim=1)
    cov = -(wp2 * c * w).sum(dim=1)                          # Cov(C,W|q) = -sum omega^2 c w
    sig_C = (s2_C + sigma_floor * sigma_floor).sqrt() * beta_T
    sig_W = (s2_W + sigma_floor * sigma_floor).sqrt() * beta_T
    # continuity correction: C_r is a DISCRETE weighted count, so P(C>=rho) ~ Phi((mu-(rho-cc))/sig)
    # with cc = half a tick (half the largest participation weight) — tightens the CLT vs the exact
    # weighted Poisson-binomial near the threshold.
    cc = 0.5 * float(omega_part.max())

    # occupancies (Gaussian-CLT tail, continuity-corrected)
    g_C = _normal_cdf((mu_C - rho_f + cc) / sig_C)
    g_W = _normal_cdf((mu_W - rho_f + cc) / sig_W)
    z_Cs = (mu_C - rho_s + cc) / sig_C
    z_Ws = (mu_W - rho_s + cc) / sig_W
    g_Cs = _normal_cdf(z_Cs)
    g_Ws = _normal_cdf(z_Ws)
    rho_CW = (cov / (sig_C * sig_W + eps)).clamp(-1.0, 1.0)
    g_S = (g_Cs * g_Ws + rho_CW * _normal_pdf(z_Cs) * _normal_pdf(z_Ws))
    g_S = g_S.clamp(min=0.0)
    g_S = torch.minimum(g_S, torch.minimum(g_Cs, g_Ws))     # split <= each marginal

    # cumulative-monotone occupancies (the CLT tail can wobble slightly via sigma)
    a_C = torch.cummax(g_C, dim=0).values
    a_W = torch.cummax(g_W, dim=0).values
    a_S = torch.cummax(g_S, dim=0).values
    A = torch.cummax((a_C + a_W + a_S).clamp(0.0, 1.0), dim=0).values   # any-basin-by-r (monotone)

    inc_C, inc_W, inc_S = _increment(a_C), _increment(a_W), _increment(a_S)
    dA = _increment(A)
    # split the new any-basin mass dA among basins by their increment share. Use clamp_min (NOT +eps)
    # so that whenever the increments are non-trivial the three shares sum to dA EXACTLY (the four
    # outcomes then telescope to 1); where all increments are 0, dA is 0 too, so the shares are 0.
    denom = (inc_C + inc_W + inc_S).clamp_min(eps)
    p_C = dA * inc_C / denom                                # [H+1, Q] first-passage mass into each basin
    p_W = dA * inc_W / denom
    p_S = dA * inc_S / denom

    P_correct_q = p_C.sum(dim=0)                            # [Q]
    F_wrong_q = p_W.sum(dim=0)
    F_split_q = p_S.sum(dim=0)
    F_deadline_q = (1.0 - A[-1]).clamp(0.0, 1.0)

    om = omega_scn.to(c.dtype)
    out = {
        "P_correct": (om * P_correct_q).sum(),
        "F_wrong": (om * F_wrong_q).sum(),
        "F_split": (om * F_split_q).sum(),
        "F_deadline": (om * F_deadline_q).sum(),
        "p_correct_rq": p_C,
    }
    if return_occupancy:
        out["g_correct"] = a_C
        out["g_wrong"] = a_W
        out["g_split"] = a_S
    return out


def confirm_cvar_surrogate(p_correct_rq: torch.Tensor, omega_scn: torch.Tensor, profile,
                           *, eps: float = 1e-12) -> dict:
    """Differentiable ``CVaR_q(T_confirm | O=C)`` + mean confirm time (spec §6; Rockafellar form).

    Args:
        p_correct_rq: ``[H+1, Q]`` correct first-passage mass per epoch/scenario
            (from :func:`basin_surrogate_outcomes`).
        omega_scn: ``[Q]`` scenario weights.
        profile: supplies ``Delta_poll (poll_window_s)`` and ``q (latency_quantile)``.

    Returns ``mean_confirm_s`` and ``cvar_confirm_s`` (differentiable scalars). ``T_confirm =
    tau_C * Delta_poll`` (a hit at epoch ``r`` -> ``r`` windows; spec §5.3 / P0-A no off-by-one).
    """
    dpoll = profile.poll_window_s
    q = profile.latency_quantile
    Hp1, Q = p_correct_rq.shape
    om = omega_scn.to(p_correct_rq.dtype).reshape(1, Q)
    P_C_r = (om * p_correct_rq).sum(dim=1)                  # [H+1] correct mass at epoch r (mixed)
    total = P_C_r.sum().clamp_min(eps)
    pi_C = P_C_r / total                                   # conditional pmf of tau_C | O=C
    r_idx = torch.arange(Hp1, dtype=p_correct_rq.dtype, device=p_correct_rq.device)
    t = r_idx * dpoll                                      # T_confirm at each epoch
    mean_confirm = (pi_C * t).sum()
    # Rockafellar CVaR: VaR z* = the q-quantile of t under pi_C (detached; envelope theorem),
    # CVaR = z* + 1/(1-q) E[(t - z*)_+].
    cdf = torch.cumsum(pi_C, dim=0)
    var_idx = int(torch.searchsorted(cdf.detach(), torch.tensor(q, dtype=cdf.dtype)).clamp(max=Hp1 - 1))
    z = (var_idx * dpoll)
    cvar = z + (1.0 / (1.0 - q)) * (pi_C * (t - z).clamp_min(0.0)).sum()
    return {"mean_confirm_s": mean_confirm, "cvar_confirm_s": cvar}
