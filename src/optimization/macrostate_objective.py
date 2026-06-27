"""Reliability-constrained macrostate objective + primal-dual (spec §6; plan §6 — G-CONSTRAINED-OBJECTIVE).

Replaces the legacy node-union F/D/E objective (``primal_dual.py``) with the participation-weighted
MACROSTATE basin objective:

    min_theta  CVaR_q(T_confirm | O=C) + lambda_E E
    s.t.       F_wrong <= eps_w,  F_split <= eps_s,  F_deadline <= eps_d

Reliability is a HARD constraint (constraint #4 — never traded for latency/energy): the duals
``(mu_w, mu_s, mu_d)`` ascend on the constraint slacks (spec §6). The reliability terms are the
differentiable macrostate basin SURROGATE (``src.metrics.basin_surrogate``) — the analytic Level-2
training proxy; the dynamic MC is the headline judge (spec §12). Selection-bias-robust reporting is
included so an infeasible policy can be filtered before any latency/energy comparison (plan §6):

* ``cvar_confirm`` / ``mean_confirm`` — latency CONDITIONAL on a correct outcome;
* ``deadline_capped_latency`` — UNconditional latency (non-correct runs charged the deadline);
* ``energy_per_attempt`` — E[energy] per attempted instance;
* ``energy_per_success`` — E[energy] / P_correct (per successful instance).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.metrics.basin_surrogate import basin_surrogate_outcomes, confirm_cvar_surrogate

__all__ = ["MacrostateDuals", "macrostate_metrics", "macrostate_lagrangian", "is_feasible",
           "train_macrostate"]


def _to_float(x) -> float:
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


@dataclass
class MacrostateDuals:
    """Dual variables for the three reliability constraints (spec §6)."""

    mu_w: float = 1.0     # F_wrong
    mu_s: float = 1.0     # F_split
    mu_d: float = 1.0     # F_deadline

    def update(self, m: dict, profile, eta_mu: float) -> None:
        """Dual ascent ``mu_x <- [mu_x + eta_mu (F_x - eps_x)]_+`` (spec §6)."""
        self.mu_w = max(0.0, self.mu_w + eta_mu * (_to_float(m["F_wrong"]) - profile.max_wrong_basin_probability))
        self.mu_s = max(0.0, self.mu_s + eta_mu * (_to_float(m["F_split"]) - profile.max_split_basin_probability))
        self.mu_d = max(0.0, self.mu_d + eta_mu * (_to_float(m["F_deadline"]) - profile.max_deadline_miss_probability))


def macrostate_metrics(
    c_traj: torch.Tensor,
    w_traj: torch.Tensor,
    omega_part: torch.Tensor,
    omega_scn: torch.Tensor,
    energy: torch.Tensor,
    profile,
    *,
    beta_T: float = 1.0,
    eps: float = 1e-12,
) -> dict:
    """Differentiable macrostate objective metrics from the per-node per-epoch marginals.

    Args:
        c_traj, w_traj: ``[R+1, N, Q]`` correct/wrong finalization marginals (analytic episode).
        omega_part: ``[N]`` participation measure; omega_scn: ``[Q]`` scenario weights.
        energy: ``[Q]`` per-scenario network energy (from the episode).
        profile: the :class:`ConsensusServiceProfile` (basin thresholds, budgets, Delta_poll, q).

    Returns a dict of differentiable scalars: the four basin outcomes, ``cvar_confirm`` /
    ``mean_confirm`` (conditional-correct latency), ``deadline_capped_latency`` (unconditional),
    ``energy`` (per attempt), ``energy_per_attempt``, ``energy_per_success``.
    """
    out = basin_surrogate_outcomes(c_traj, w_traj, omega_part, omega_scn, profile, beta_T=beta_T)
    cvar = confirm_cvar_surrogate(out["p_correct_rq"], omega_scn, profile)
    om = omega_scn.to(energy.dtype)
    E_network = (om * energy).sum()                                  # per attempted instance
    P_correct = out["P_correct"]

    # deadline-capped UNconditional latency: correct runs at their tau_C, all others at the deadline.
    dpoll = profile.poll_window_s
    R_d = profile.max_poll_epochs
    p_c_r = (om.reshape(1, -1) * out["p_correct_rq"]).sum(dim=1)     # [H+1] correct mass at epoch r
    Hp1 = p_c_r.shape[0]
    r_idx = torch.arange(Hp1, dtype=p_c_r.dtype, device=p_c_r.device)
    correct_time = (p_c_r * r_idx * dpoll).sum()
    deadline_capped = correct_time + (1.0 - P_correct) * (R_d * dpoll)

    return {
        "P_correct": P_correct,
        "F_wrong": out["F_wrong"],
        "F_split": out["F_split"],
        "F_deadline": out["F_deadline"],
        "cvar_confirm": cvar["cvar_confirm_s"],
        "mean_confirm": cvar["mean_confirm_s"],
        "deadline_capped_latency": deadline_capped,
        "energy": E_network,
        "energy_per_attempt": E_network,
        "energy_per_success": E_network / P_correct.clamp_min(eps),
    }


def macrostate_lagrangian(m: dict, duals: MacrostateDuals, profile, *, lambda_E: float = 0.0) -> torch.Tensor:
    """Augmented-Lagrangian training loss (spec §6).

    ``L = CVaR_q(T_confirm|O=C) + lambda_E E + mu_w(F_wrong-eps_w) + mu_s(F_split-eps_s)
         + mu_d(F_deadline-eps_d)``. Reliability is a hard constraint enforced by the duals, never a
    hand-weighted Pareto axis (constraint #4).
    """
    return (m["cvar_confirm"] + lambda_E * m["energy"]
            + duals.mu_w * (m["F_wrong"] - profile.max_wrong_basin_probability)
            + duals.mu_s * (m["F_split"] - profile.max_split_basin_probability)
            + duals.mu_d * (m["F_deadline"] - profile.max_deadline_miss_probability))


def train_macrostate(
    model,
    train_instances: list[tuple],          # list of (scene, evidence)
    protocol_cfg,
    phy_cfg,
    profile,
    *,
    participation_fn=None,                  # scene -> omega [N]; default uniform
    steps: int = 40,
    lr: float = 5e-3,
    eta_mu: float = 5.0,
    lambda_E: float = 0.0,
    beta_T: float = 1.0,
    link_override: float | None = None,
    duals: "MacrostateDuals | None" = None,
) -> dict:
    """Primal-dual training of the query topology on the MACROSTATE constrained objective (spec §6).

    Each step: run the analytic canonical episode (the differentiable surrogate path), form the
    macrostate metrics + augmented Lagrangian, descend the model params, then ascend the duals on
    the constraint slacks. The reliability terms are the participation-weighted basin surrogate
    (``basin_surrogate``); the independent dynamic MC is the headline judge. ``link_override``
    isolates the topology lever for a fast demonstration (the full-physics headline training is
    Phase 7 / G-ESP-BASELINE).

    Returns ``{model, duals, history}`` (history records loss / the four basins / CVaR / the duals).
    """
    import torch as _torch

    from src.metrics.participation import uniform_participation
    from src.models import ESDGNNQueryPolicy
    from src.environment.canonical_episode import run_consensus_episode

    if participation_fn is None:
        def participation_fn(sc):
            return uniform_participation(sc.num_nodes, dtype=_torch.float64, device=sc.positions.device)

    opt = _torch.optim.Adam(model.parameters(), lr=lr)
    duals = duals or MacrostateDuals()
    history = {k: [] for k in ("loss", "P_correct", "F_wrong", "F_split", "F_deadline",
                               "cvar_confirm", "mu_w", "mu_s", "mu_d")}
    n = len(train_instances)
    for step in range(steps):
        scene, ev = train_instances[step % n]
        omega = participation_fn(scene)
        policy = ESDGNNQueryPolicy(model, scene)
        opt.zero_grad()
        res = run_consensus_episode(scene, ev, policy, protocol_cfg, phy_cfg,
                                    return_trajectory=True, link_override=link_override)
        m = macrostate_metrics(res.c_trajectory, res.w_trajectory, omega, res.scenario_weight,
                               res.energy, profile, beta_T=beta_T)
        loss = macrostate_lagrangian(m, duals, profile, lambda_E=lambda_E)
        loss.backward()
        opt.step()
        duals.update(m, profile, eta_mu)
        history["loss"].append(_to_float(loss))
        for key in ("P_correct", "F_wrong", "F_split", "F_deadline", "cvar_confirm"):
            history[key].append(_to_float(m[key]))
        history["mu_w"].append(duals.mu_w)
        history["mu_s"].append(duals.mu_s)
        history["mu_d"].append(duals.mu_d)
    return {"model": model, "duals": duals, "history": history}


def is_feasible(m: dict, profile, *, tol: float = 0.0) -> bool:
    """True iff all three hard reliability constraints hold (plan §6 — infeasible policies are
    excluded BEFORE any latency/energy comparison; reliability is never traded away).

    Works on whichever metric dict it is given: pass the SURROGATE ``macrostate_metrics`` for the
    training-time feasibility check, or the dynamic-MC basin outcomes (``basin_F_wrong`` etc.,
    remapped to these keys) for the HEADLINE feasibility filter — the MC is the judge of record
    (spec §12). The surrogate is faithful here because the ``c_i+w_i<=1`` precondition keeps the
    basins disjoint, so its ``F_deadline`` is not floored at 0 for genuinely-infeasible policies.
    """
    return (_to_float(m["F_wrong"]) <= profile.max_wrong_basin_probability + tol
            and _to_float(m["F_split"]) <= profile.max_split_basin_probability + tol
            and _to_float(m["F_deadline"]) <= profile.max_deadline_miss_probability + tol)
