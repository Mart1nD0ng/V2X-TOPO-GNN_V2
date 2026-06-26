"""Direct per-scene topology optimizer -- the optimization CEILING (Phase 7, stop-condition #2).

To justify building the ESD-GNN we must first show the query topology is a real lever: a
DIRECT per-scene optimizer of the edge query-weights (the spec plan §10 "direct per-scene
edge-logit optimizer") must beat the heuristic policies on the reliability objective, and the
advantage must survive the independent dynamic MC (spec plan §10 criterion). If it does not,
loop stop-condition #2 fires: there is no topology lever, so no GNN can help -- STOP + report.

This optimizer holds a free per-edge log-weight vector ``s_ij`` (the ESP query law, which the
dynamic MC supports) and descends on the analytic episode's safety+validity objective
``F_disagree + F_wrong`` by Adam. It is an UPPER BOUND per scene -- it uses the scene's
evidence through the objective, so it is NOT a deployable policy (constraint #10); the ESD-GNN
must approach this ceiling from observable features alone. The CDQ kernel oracle (the richer
diversity lever) and a CDQ dynamic MC are a follow-up; the edge-logit oracle is the spec's
designated Phase-7 ceiling and is MC-confirmable with the existing ESP sampler.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.environment.canonical_episode import ProtocolConfig, run_consensus_episode
from src.environment.candidate_graph import build_candidate_graph
from src.environment.round_physics import RoundPhysicsConfig
from src.sampling import DistanceQueryPolicy, UniformQueryPolicy
from src.validation import run_dynamic_mc

__all__ = ["LearnedLogWeightPolicy", "optimize_logweight_topology", "oracle_vs_heuristics"]


class LearnedLogWeightPolicy:
    """ESP query policy with a free per-edge log-weight vector (the optimizer's variable)."""

    query_law = "esp"
    name = "oracle_logweight"

    def __init__(self, log_weights: torch.Tensor):
        self.log_weights_param = log_weights

    def log_weights(self, graph) -> torch.Tensor:
        if self.log_weights_param.shape[0] != graph.num_edges:
            raise ValueError("log_weights size must match the candidate graph edge count")
        return self.log_weights_param


@dataclass(frozen=True)
class OracleResult:
    policy: LearnedLogWeightPolicy
    history: list[float]
    final_objective: float


def optimize_logweight_topology(
    scene,
    evidence,
    protocol_cfg: ProtocolConfig,
    phy_cfg: RoundPhysicsConfig,
    *,
    steps: int = 120,
    lr: float = 0.2,
    link_override: float | None = None,
    weight_clip: float = 8.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
) -> OracleResult:
    """Per-scene Adam descent on ``F_disagree + F_wrong`` over the edge log-weights."""
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    E = gc.num_edges
    log_w = torch.zeros(E, dtype=dtype, requires_grad=True)   # init = uniform policy
    opt = torch.optim.Adam([log_w], lr=lr)
    policy = LearnedLogWeightPolicy(log_w)
    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        res = run_consensus_episode(scene, evidence, policy, protocol_cfg, phy_cfg,
                                    return_trajectory=False, link_override=link_override)
        loss = res.F_disagree + res.F_wrong
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_w.clamp_(-weight_clip, weight_clip)           # keep weights bounded (no degenerate blow-up)
        history.append(float(loss))
    return OracleResult(policy=LearnedLogWeightPolicy(log_w.detach()),
                        history=history, final_objective=history[-1])


def _analytic_objective(scene, evidence, policy, pcfg, phy, link_override) -> dict:
    res = run_consensus_episode(scene, evidence, policy, pcfg, phy,
                                return_trajectory=False, link_override=link_override)
    return {"F_disagree": float(res.F_disagree), "F_wrong": float(res.F_wrong),
            "S_allcorrect": float(res.S_allcorrect),
            "objective": float(res.F_disagree + res.F_wrong)}


def oracle_vs_heuristics(
    scene,
    evidence,
    protocol_cfg: ProtocolConfig,
    phy_cfg: RoundPhysicsConfig,
    *,
    steps: int = 120,
    lr: float = 0.2,
    link_override: float | None = None,
    mc_trials: int = 0,
    mc_seed: int = 0,
    seed: int = 0,
) -> dict:
    """Compare the per-scene edge-logit oracle to the uniform / distance heuristics.

    Returns analytic objectives for each policy and (if ``mc_trials > 0``) dynamic-MC
    ``F_wrong`` for the best heuristic vs the oracle (the spec-required MC confirmation that
    the topology advantage survives the independent simulation).
    """
    heuristics = {"uniform": UniformQueryPolicy(), "distance": DistanceQueryPolicy(beta_per_m=0.03)}
    analytic = {name: _analytic_objective(scene, evidence, pol, protocol_cfg, phy_cfg, link_override)
                for name, pol in heuristics.items()}
    oracle = optimize_logweight_topology(scene, evidence, protocol_cfg, phy_cfg,
                                         steps=steps, lr=lr, link_override=link_override, seed=seed)
    analytic["oracle"] = _analytic_objective(scene, evidence, oracle.policy, protocol_cfg, phy_cfg, link_override)
    best_heur = min(heuristics, key=lambda n: analytic[n]["objective"])
    out = {"analytic": analytic, "best_heuristic": best_heur,
           "analytic_gain": analytic[best_heur]["objective"] - analytic["oracle"]["objective"]}
    if mc_trials > 0:
        mc = {}
        for name, pol in [("oracle", oracle.policy), (best_heur, heuristics[best_heur])]:
            r = run_dynamic_mc(scene, evidence, pol, protocol_cfg, phy_cfg, num_trials=mc_trials,
                               generator=torch.Generator().manual_seed(mc_seed), link_override=link_override)
            mc[name] = {"F_wrong": r.F_wrong, "F_wrong_ci": r.F_wrong_ci,
                        "F_disagree": r.F_disagree, "S_allcorrect": r.S_allcorrect}
        out["mc"] = mc
        out["mc_gain"] = mc[best_heur]["F_wrong"] - mc["oracle"]["F_wrong"]
    return out
