"""Baseline + oracle calibration for ESP scale validation (esp_performance_scale_v2, G-ESP-BASELINE-ORACLE).

A scale claim is meaningful only against strong, capability-matched baselines AND a problem-headroom
oracle (workflow §5). This module gives a single ``make_baseline`` factory for the deployable policies
and a ``direct_edge_logit_oracle`` that estimates per-scene topology headroom by directly optimising free
per-edge logits against the differentiable analytic macro objective.

The oracle is NOT deployable: it optimises the analytic macrostate objective for a SPECIFIC scene+evidence
(it sees the scene's correctness structure through the objective), so it upper-bounds what any ESP query
law could achieve on that scene. If the oracle does not beat the heuristics, the scene/profile has little
topology-learning headroom and a GNN that only matches heuristics is not a failure (workflow §5.3).
"""

from __future__ import annotations

import torch

from src.environment.canonical_episode import run_consensus_episode
from src.environment.candidate_graph import build_candidate_graph
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.optimization.macrostate_objective import macrostate_metrics
from src.policies.heuristics import LinkQualityPolicy, LoadBalancedPolicy, RegionBridgePolicy
from src.sampling.baseline_policies import DistanceQueryPolicy, UniformQueryPolicy

__all__ = ["DEPLOYABLE_BASELINES", "make_baseline", "direct_edge_logit_oracle"]

# deployable baseline kinds (the GNN policies -- expert / shared -- are built by the caller from a model)
DEPLOYABLE_BASELINES = ("uniform_esp", "distance", "link_quality", "load_balanced", "region_bridge")


def make_baseline(kind: str, scene, *, distance_beta: float = 0.04):
    """A deployable ESP baseline policy bound to ``scene`` (heuristics that need scene structure)."""
    if kind == "uniform_esp":
        return UniformQueryPolicy()
    if kind == "distance":
        return DistanceQueryPolicy(beta_per_m=distance_beta)
    if kind == "link_quality":
        return LinkQualityPolicy()
    if kind == "load_balanced":
        return LoadBalancedPolicy(scene)
    if kind == "region_bridge":
        return RegionBridgePolicy(scene)
    raise ValueError(f"unknown baseline kind {kind!r}")


class _FreeLogitPolicy:
    """ESP policy whose per-edge log-weights are FREE parameters (the topology oracle's variables).

    The episode rebuilds ``G_comm`` deterministically from the same positions, so the edge order matches
    the logit vector built from the same ``build_candidate_graph``."""

    name = "direct_edge_logit_optimizer"

    def __init__(self, logits: torch.Tensor):
        self.logits = logits

    def log_weights(self, graph) -> torch.Tensor:
        if graph.num_edges != self.logits.shape[0]:                       # defensive: order/size must match
            raise ValueError("edge count mismatch between oracle logits and the episode graph")
        return self.logits


def direct_edge_logit_oracle(scene, ev, profile, proto, phy, *, steps: int = 60, lr: float = 0.1,
                             link_override=None, beta_T: float = 1.0,
                             feas_weight: float = 50.0) -> dict:
    """Per-scene topology headroom oracle: directly optimise free per-edge logits to MAXIMISE the analytic
    ``macro_P_correct`` (with a soft wrong/split feasibility penalty), then return the optimised analytic
    macro block. Upper-bounds any ESP query law on this scene (workflow §5.3). Analytic (screening) only.
    """
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    # init from the distance heuristic so the optimiser starts at a sensible operating point
    logits = (-0.04 * gc.distance).clone().to(torch.float64).requires_grad_(True)
    omega = uniform_participation(scene.num_nodes, dtype=torch.float64, device=scene.positions.device)
    opt = torch.optim.Adam([logits], lr=lr)
    policy = _FreeLogitPolicy(logits)
    last = None
    for _ in range(steps):
        opt.zero_grad()
        res = run_consensus_episode(scene, ev, policy, proto, phy, return_trajectory=True,
                                    link_override=link_override)
        m = macrostate_metrics(res.c_trajectory, res.w_trajectory, omega, res.scenario_weight,
                               res.energy, profile, beta_T=beta_T)
        # maximise P_correct; softly penalise wrong/split ABOVE the budget (reliability stays a constraint)
        over_w = torch.clamp(m["F_wrong"] - profile.max_wrong_basin_probability, min=0.0)
        over_s = torch.clamp(m["F_split"] - profile.max_split_basin_probability, min=0.0)
        loss = -m["P_correct"] + feas_weight * (over_w + over_s)
        loss.backward()
        opt.step()
        last = m
    return schema.macro_block(float(last["P_correct"].detach()), float(last["F_wrong"].detach()),
                              float(last["F_split"].detach()), float(last["F_deadline"].detach()))
