"""Primal-dual reliability-constrained training of the ESD-GNN (spec §4.5 -- G9b).

Optimises the query topology to minimise expected tail-confirmation latency + energy SUBJECT
to the hard reliability constraints (spec §4.4), via the augmented Lagrangian with dual ascent:

    L_theta = ET + lambda_E E
              + mu_s (F_disagree - eps_s) + mu_v (F_wrong - eps_v) + mu_d (F_deadline - eps_d)
    mu_r   <- [ mu_r + eta_mu (F_r - eps_r) ]_+                                    (spec §4.5)

Reliability is a HARD constraint (non-negotiable #1): fixed hand-weights may NOT replace the
dual ascent on ``mu``; only points satisfying the constraints are admissible. The latency
surrogate ``ET`` (differentiable expected completion time ``sum_t round_duration(t)
P(not all-correct after t)``) is the trainable analytic proxy for ``CVaR_q(T_all)``; the
headline ``CVaR`` is reported by the independent dynamic MC (G11), not this surrogate.

The analytic canonical episode (differentiable) drives training; the MC is the judge.
Training and deployment use the SAME ESD-GNN forward (constraint #3); the model sees only
observable features (constraint #10). All metrics come from the single canonical evaluator
(constraint #6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.environment.canonical_episode import ProtocolConfig, run_consensus_episode
from src.environment.round_physics import RoundPhysicsConfig

__all__ = [
    "ReliabilityThresholds",
    "DualState",
    "episode_metrics",
    "lagrangian",
    "train_esd_gnn",
]


@dataclass(frozen=True)
class ReliabilityThresholds:
    eps_s: float = 1e-4          # F_disagree budget
    eps_v: float = 1e-3          # F_wrong budget
    eps_d: float = 1e-2          # F_deadline budget
    lambda_E: float = 0.0        # energy weight (0 = latency/reliability headline)
    deadline_round: int | None = None   # round for F_deadline (default: full horizon)


def _to_float(x) -> float:
    """Detach-safe scalar cast (the duals/history are report-only, off the training graph)."""
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


@dataclass
class DualState:
    mu_s: float = 1.0
    mu_v: float = 1.0
    mu_d: float = 1.0

    def update(self, m: dict, thr: ReliabilityThresholds, eta_mu: float) -> None:
        self.mu_s = max(0.0, self.mu_s + eta_mu * (_to_float(m["F_disagree"]) - thr.eps_s))
        self.mu_v = max(0.0, self.mu_v + eta_mu * (_to_float(m["F_wrong"]) - thr.eps_v))
        self.mu_d = max(0.0, self.mu_d + eta_mu * (_to_float(m["F_deadline"]) - thr.eps_d))


def episode_metrics(res, eligible_mask: torch.Tensor | None, *, deadline_round: int | None = None,
                    eps: float = 1e-12) -> dict:
    """Differentiable metrics from a canonical-episode result (needs ``return_trajectory=True``).

    Returns ``F_disagree``, ``F_wrong`` (from the episode), ``F_deadline`` = ``1 - P(all
    eligible decided correct by the deadline round)``, ``ET`` (expected completion time), and
    ``energy`` -- all differentiable tensors.
    """
    if res.c_trajectory is None:
        raise ValueError("episode_metrics needs return_trajectory=True")
    omega = res.scenario_weight                      # [Q]
    c = res.c_trajectory                             # [R+1, N, Q]
    N, Q = res.c_ir.shape
    elig = (torch.ones(N, dtype=torch.bool, device=c.device) if eligible_mask is None
            else eligible_mask.to(device=c.device, dtype=torch.bool))
    c_e = c[:, elig, :].clamp(eps, 1.0)              # [R+1, |H|, Q]
    log_prod = torch.log(c_e).sum(dim=1)             # [R+1, Q]  log prod_i c_ir(t)
    S_all_t = torch.exp(log_prod)                    # [R+1, Q]  P(all correct by t | r)
    S_correct_t = (omega.unsqueeze(0) * S_all_t).sum(dim=1)   # [R+1]  mixture
    R = c.shape[0] - 1
    d_round = R if deadline_round is None else min(deadline_round, R)
    F_deadline = (1.0 - S_correct_t[d_round]).clamp(0.0, 1.0)
    # expected completion time: E[T] = sum_t round_duration(t) * P(not all-correct after t)
    rd = res.round_duration                          # [R, Q]
    not_done = (1.0 - S_all_t[1:]).clamp(0.0, 1.0)    # [R, Q] P(not complete after round t)
    ET = (omega.unsqueeze(0) * (rd * not_done)).sum()
    energy = (omega * res.energy).sum()
    return {"F_disagree": res.F_disagree, "F_wrong": res.F_wrong, "F_deadline": F_deadline,
            "ET": ET, "energy": energy, "S_allcorrect": res.S_allcorrect}


def lagrangian(m: dict, duals: DualState, thr: ReliabilityThresholds) -> torch.Tensor:
    """Augmented-Lagrangian training loss (spec §4.5)."""
    return (m["ET"] + thr.lambda_E * m["energy"]
            + duals.mu_s * (m["F_disagree"] - thr.eps_s)
            + duals.mu_v * (m["F_wrong"] - thr.eps_v)
            + duals.mu_d * (m["F_deadline"] - thr.eps_d))


def train_esd_gnn(
    model,
    train_instances: list[tuple],          # list of (scene, evidence)
    protocol_cfg: ProtocolConfig,
    phy_cfg: RoundPhysicsConfig,
    thr: ReliabilityThresholds,
    *,
    steps: int = 60,
    lr: float = 5e-3,
    eta_mu: float = 5.0,
    link_override: float | None = None,
    seed: int = 0,
    duals: DualState | None = None,
) -> dict:
    """Primal-dual training of an :class:`ESDGNN` over a set of (scene, evidence) instances.

    Each step: pick an instance round-robin, run the analytic CDQ episode, compute the
    Lagrangian, descend on the model params, then ascend the duals on the constraint slacks.
    Returns a history dict (loss, F_*, duals) and the trained model (mutated in place).
    """
    from src.models import ESDGNNQueryPolicy

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    duals = duals or DualState()
    history = {"loss": [], "F_disagree": [], "F_wrong": [], "F_deadline": [], "ET": [],
               "mu_v": [], "S_allcorrect": []}
    n = len(train_instances)
    for step in range(steps):
        scene, ev = train_instances[step % n]
        policy = ESDGNNQueryPolicy(model, scene)
        opt.zero_grad()
        res = run_consensus_episode(scene, ev, policy, protocol_cfg, phy_cfg,
                                    return_trajectory=True, link_override=link_override)
        m = episode_metrics(res, None, deadline_round=thr.deadline_round)
        loss = lagrangian(m, duals, thr)
        loss.backward()
        opt.step()
        duals.update(m, thr, eta_mu)                 # dual ascent on the constraint slacks
        history["loss"].append(_to_float(loss))
        for key in ("F_disagree", "F_wrong", "F_deadline", "ET", "S_allcorrect"):
            history[key].append(_to_float(m[key]))
        history["mu_v"].append(duals.mu_v)
    return {"model": model, "duals": duals, "history": history}
