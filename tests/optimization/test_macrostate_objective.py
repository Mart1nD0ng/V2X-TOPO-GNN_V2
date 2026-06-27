"""G-CONSTRAINED-OBJECTIVE: reliability-constrained macrostate objective + primal-dual (spec §6).

    min CVaR_q(T_confirm | O=C) + lambda_E E
    s.t. F_wrong <= eps_w, F_split <= eps_s, F_deadline <= eps_d

via the augmented Lagrangian with dual ascent on (mu_w, mu_s, mu_d). The reliability terms are
the participation-weighted MACROSTATE basin probabilities (the differentiable surrogate; the MC
is the headline judge) — NOT the legacy node-union F. Selection-bias-robust reporting
(conditional-correct latency, deadline-capped unconditional latency, energy per attempted /
successful instance) is included.
"""

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.optimization.macrostate_objective import (
    MacrostateDuals,
    macrostate_lagrangian,
    macrostate_metrics,
)

PROFILE = ConsensusServiceProfile.urban_default().replace(
    max_poll_epochs=8, poll_window_ms=10.0,
    max_wrong_basin_probability=1e-2, max_split_basin_probability=1e-2,
    max_deadline_miss_probability=5e-2)


def _traj(c_final, w_final, N, Q=1, R=8):
    ramp = torch.linspace(0, 1, R + 1, dtype=torch.float64).reshape(R + 1, 1, 1)
    c = ramp * torch.as_tensor(c_final, dtype=torch.float64).reshape(1, N, 1)
    w = ramp * torch.as_tensor(w_final, dtype=torch.float64).reshape(1, N, 1)
    return c, w


def _omega(N):
    return torch.full((N,), 1.0 / N, dtype=torch.float64)


def test_metrics_have_macrostate_and_selection_bias_fields():
    N = 12
    c, w = _traj(torch.full((N,), 0.85), torch.full((N,), 0.05), N)
    energy = torch.tensor([0.5], dtype=torch.float64)        # [Q]
    m = macrostate_metrics(c, w, _omega(N), torch.ones(1, dtype=torch.float64), energy, PROFILE)
    for k in ("P_correct", "F_wrong", "F_split", "F_deadline", "cvar_confirm", "mean_confirm",
              "energy", "energy_per_attempt", "energy_per_success", "deadline_capped_latency"):
        assert k in m
    # the four basin outcomes still partition probability
    total = m["P_correct"] + m["F_wrong"] + m["F_split"] + m["F_deadline"]
    assert abs(float(total) - 1.0) < 1e-9
    # energy per success >= energy per attempt (you only succeed on a fraction of attempts)
    assert float(m["energy_per_success"]) >= float(m["energy_per_attempt"]) - 1e-9


def test_lagrangian_is_differentiable():
    N = 10
    c, w = _traj(torch.full((N,), 0.6), torch.full((N,), 0.2), N)
    c = c.clone().requires_grad_(True)
    energy = torch.tensor([0.3], dtype=torch.float64)
    m = macrostate_metrics(c, w, _omega(N), torch.ones(1, dtype=torch.float64), energy, PROFILE)
    duals = MacrostateDuals()
    loss = macrostate_lagrangian(m, duals, PROFILE)
    loss.backward()
    assert c.grad is not None and bool(torch.isfinite(c.grad).all())
    assert float(c.grad.abs().sum()) > 0


def test_dual_ascent_responds_to_violation():
    # a wrong-leaning instance violates F_wrong <= eps_w -> mu_w must increase; a satisfied
    # constraint pushes mu down (floored at 0).
    N = 12
    c, w = _traj(torch.full((N,), 0.1), torch.full((N,), 0.9), N)   # mostly wrong -> F_wrong high
    energy = torch.tensor([0.4], dtype=torch.float64)
    m = macrostate_metrics(c, w, _omega(N), torch.ones(1, dtype=torch.float64), energy, PROFILE)
    assert float(m["F_wrong"]) > PROFILE.max_wrong_basin_probability   # genuinely infeasible
    duals = MacrostateDuals(mu_w=1.0, mu_s=1.0, mu_d=1.0)
    duals.update(m, PROFILE, eta_mu=5.0)
    assert duals.mu_w > 1.0                       # violated -> dual rises
    # a comfortably-satisfied split constraint drives mu_s toward 0 (floored)
    duals2 = MacrostateDuals(mu_w=0.0, mu_s=0.01, mu_d=0.0)
    cc, ww = _traj(torch.full((N,), 0.95), torch.full((N,), 0.01), N)
    m2 = macrostate_metrics(cc, ww, _omega(N), torch.ones(1, dtype=torch.float64), energy, PROFILE)
    duals2.update(m2, PROFILE, eta_mu=5.0)
    assert duals2.mu_s == 0.0                      # satisfied -> floored at 0


def test_feasibility_flag_excludes_infeasible_policy():
    from src.optimization.macrostate_objective import is_feasible
    N = 12
    c_good, w_good = _traj(torch.full((N,), 0.95), torch.full((N,), 0.01), N)
    c_bad, w_bad = _traj(torch.full((N,), 0.1), torch.full((N,), 0.9), N)
    mg = macrostate_metrics(c_good, w_good, _omega(N), torch.ones(1, dtype=torch.float64),
                            torch.tensor([0.3], dtype=torch.float64), PROFILE)
    mb = macrostate_metrics(c_bad, w_bad, _omega(N), torch.ones(1, dtype=torch.float64),
                            torch.tensor([0.3], dtype=torch.float64), PROFILE)
    assert is_feasible(mg, PROFILE) and not is_feasible(mb, PROFILE)
