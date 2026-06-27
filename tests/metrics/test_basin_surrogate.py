"""G-CONSTRAINED-OBJECTIVE: differentiable analytic macrostate basin-hazard surrogate (spec §12 L2).

A differentiable Level-2 training proxy for the participation-weighted basin first-hitting
outcome probabilities, built from the per-node per-epoch finalization marginals (the analytic
episode's c/w trajectories). Design (design-panel synthesis, run wf_f37372a9-061):
  * Gaussian-CLT per-epoch occupancy g_C(r,q)=Phi((mu_C-rho_f)/sigma_C) etc. (O(N), no N^2 DP);
  * increment-share FIRST-PASSAGE over epochs -> the 4 outcomes sum to 1 EXACTLY (telescoping);
  * bivariate-normal (negative-correlation) Mehler correction for the split.
The independent dynamic MC is the Level-3 judge; this surrogate need only be faithful + smooth.
Validated against an independent Poisson-binomial exact occupancy at small N.
"""

import math

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.metrics.basin_surrogate import basin_surrogate_outcomes, confirm_cvar_surrogate

PROFILE = ConsensusServiceProfile.urban_default()  # rho_f=0.6, rho_s=0.45, R_d=20


def _traj(c_final, w_final, N, Q=1, R=6):
    """A ramping trajectory: per-node marginals rise linearly from 0 to the given finals."""
    c = torch.zeros(R + 1, N, Q, dtype=torch.float64)
    w = torch.zeros(R + 1, N, Q, dtype=torch.float64)
    ramp = torch.linspace(0, 1, R + 1, dtype=torch.float64).reshape(R + 1, 1, 1)
    c = ramp * torch.as_tensor(c_final, dtype=torch.float64).reshape(1, N, 1)
    w = ramp * torch.as_tensor(w_final, dtype=torch.float64).reshape(1, N, 1)
    return c, w


def _uniform(N):
    return torch.full((N,), 1.0 / N, dtype=torch.float64)


def test_outcomes_sum_to_one_exactly():
    N = 10
    c_final = torch.rand(N, generator=torch.Generator().manual_seed(1), dtype=torch.float64) * 0.7
    w_final = torch.rand(N, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.3
    c, w = _traj(c_final, w_final, N)
    out = basin_surrogate_outcomes(c, w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    total = out["P_correct"] + out["F_wrong"] + out["F_split"] + out["F_deadline"]
    assert abs(float(total) - 1.0) < 1e-9
    for k in ("P_correct", "F_wrong", "F_split", "F_deadline"):
        assert -1e-12 <= float(out[k]) <= 1.0 + 1e-9


def test_all_correct_limit():
    N = 12
    c, w = _traj(torch.ones(N), torch.zeros(N), N)
    out = basin_surrogate_outcomes(c, w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    assert float(out["P_correct"]) > 0.98
    assert float(out["F_wrong"]) < 0.02 and float(out["F_deadline"]) < 0.02


def test_all_wrong_limit():
    N = 12
    c, w = _traj(torch.zeros(N), torch.ones(N), N)
    out = basin_surrogate_outcomes(c, w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    assert float(out["F_wrong"]) > 0.98


def test_tiny_mass_is_deadline():
    N = 12
    c, w = _traj(torch.full((N,), 0.02), torch.full((N,), 0.02), N)
    out = basin_surrogate_outcomes(c, w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    assert float(out["F_deadline"]) > 0.9


def test_balanced_opposing_groups_produce_split():
    N = 12
    cf = torch.zeros(N); wf = torch.zeros(N)
    cf[: N // 2] = 1.0          # half finalize correct
    wf[N // 2:] = 1.0           # half finalize wrong
    c, w = _traj(cf, wf, N)
    out = basin_surrogate_outcomes(c, w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    assert float(out["F_split"]) > 0.3   # C~0.5, W~0.5 -> both >= rho_s=0.45 -> split


def test_F_wrong_monotone_in_wrong_mass():
    N = 12
    lo_c, lo_w = _traj(torch.full((N,), 0.2), torch.full((N,), 0.5), N)
    hi_c, hi_w = _traj(torch.full((N,), 0.2), torch.full((N,), 0.8), N)
    lo = basin_surrogate_outcomes(lo_c, lo_w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    hi = basin_surrogate_outcomes(hi_c, hi_w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    assert float(hi["F_wrong"]) > float(lo["F_wrong"])


def test_differentiable_in_marginals():
    N = 10
    c, w = _traj(torch.full((N,), 0.55), torch.full((N,), 0.2), N)
    c = c.clone().requires_grad_(True)
    out = basin_surrogate_outcomes(c, w, _uniform(N), torch.ones(1, dtype=torch.float64), PROFILE)
    # increasing correct mass should raise P_correct -> a usable gradient
    out["P_correct"].backward()
    assert c.grad is not None and bool(torch.isfinite(c.grad).all())
    assert float(c.grad.abs().sum()) > 0


def _poisson_binomial_ge(probs, omega, thresh, *, n_grid=2000):
    """Independent reference: exact P(sum_i omega_i Bern(probs_i) >= thresh) via a tick DP."""
    K = n_grid
    ticks = torch.round(omega * K).to(torch.int64).clamp_min(0)
    M = int(ticks.sum())
    pmf = torch.zeros(M + 1, dtype=torch.float64)
    pmf[0] = 1.0
    for p, m in zip(probs.tolist(), ticks.tolist()):
        if m == 0:
            continue
        shifted = torch.zeros_like(pmf)
        shifted[m:] = pmf[:-m] if m <= M else pmf[:0]
        pmf = pmf * (1.0 - p) + shifted * p
    cut = int(math.ceil(thresh * M))
    return float(pmf[cut:].sum())


def test_clt_occupancy_matches_poisson_binomial_reference():
    """The Gaussian-CLT terminal occupancy P(C_R>=rho_f) must agree with the INDEPENDENT exact
    weighted Poisson-binomial CDF (validates the CLT approximation at a moderate N)."""
    N = 40
    g = torch.Generator().manual_seed(5)
    cf = (0.4 + 0.4 * torch.rand(N, generator=g, dtype=torch.float64))   # spread around the threshold
    c, w = _traj(cf, torch.zeros(N), N)
    omega = _uniform(N)
    out = basin_surrogate_outcomes(c, w, omega, torch.ones(1, dtype=torch.float64), PROFILE,
                                   return_occupancy=True)
    g_clt = float(out["g_correct"][-1, 0])   # terminal correct occupancy P(C_R>=rho_f)
    g_exact = _poisson_binomial_ge(cf, omega, PROFILE.correct_basin_mass)
    assert abs(g_clt - g_exact) < 0.05       # CLT vs exact Poisson-binomial within 5%


def test_confirm_cvar_is_finite_and_tail_geq_mean():
    N = 12
    c, w = _traj(torch.full((N,), 0.9), torch.full((N,), 0.05), N, R=10)
    prof = PROFILE.replace(max_poll_epochs=10, latency_quantile=0.9, poll_window_ms=10.0)
    out = basin_surrogate_outcomes(c, w, _uniform(N), torch.ones(1, dtype=torch.float64), prof)
    cv = confirm_cvar_surrogate(out["p_correct_rq"], torch.ones(1, dtype=torch.float64), prof)
    assert math.isfinite(cv["mean_confirm_s"]) and math.isfinite(cv["cvar_confirm_s"])
    assert cv["cvar_confirm_s"] >= cv["mean_confirm_s"] - 1e-9
    assert cv["mean_confirm_s"] > 0
