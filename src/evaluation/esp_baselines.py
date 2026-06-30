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

__all__ = ["DEPLOYABLE_BASELINES", "make_baseline", "direct_edge_logit_oracle",
           "train_mc_edge_logit_oracle", "free_logit_policy"]

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


def free_logit_policy(logits: torch.Tensor) -> _FreeLogitPolicy:
    """Wrap a free per-edge logit vector as an ESP query policy (for evaluation under the dynamic MC)."""
    return _FreeLogitPolicy(logits)


def train_mc_edge_logit_oracle(scene, ev, profile, proto, phy, *, steps: int = 150, train_trials: int = 150,
                               init: str = "distance", lr: float = 0.1, base_seed: int = 0,
                               distance_beta: float = 0.04, rand_seed: int = 0, rand_scale: float = 0.5):
    """MC-JUDGED per-scene topology headroom oracle (Campaign-A Lever 1, the gate for section-13.1).

    Unlike ``direct_edge_logit_oracle`` (which optimises the peer-BLIND analytic macro, EV1/EV2), this
    optimises FREE per-edge logits directly against the DYNAMIC-MC basin reward via the score-function
    gradient (``run_dynamic_mc(reinforce=True)`` -> per-(trial,node) ``log pi`` + per-node correct reward),
    i.e. the same objective the judge measures. It upper-bounds what ANY diagonal (per-edge) ESP law -- a
    trained GNN included -- can achieve on this scene under the judge: if even these free per-scene logits
    cannot CI-separately beat the distance heuristic, the regime has no diagonal topology headroom and
    parity is the honest ceiling (workflow §5.3); if they do, superiority is a training/capacity problem.

    Performance is UN-GATED (reward = per-node correct finalisation, matching A0); reliability (F_wrong /
    F_split) is read off the eval separately. ``init='distance'`` starts AT the distance operating point
    (REINFORCE can only improve from there); ``init='random'`` is the independent control that distinguishes
    a true no-headroom attractor from an optimiser artefact. Returns ``{logits, history, num_edges, init}``.
    """
    from src.validation import run_dynamic_mc
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    if init == "distance":
        logits = (-distance_beta * gc.distance).clone().to(torch.float64)
    elif init == "random":
        g = torch.Generator().manual_seed(int(rand_seed))
        logits = rand_scale * torch.randn(int(gc.num_edges), generator=g, dtype=torch.float64)
    else:
        raise ValueError(f"unknown init {init!r}")
    logits = logits.requires_grad_(True)
    omega = uniform_participation(scene.num_nodes, dtype=torch.float64, device=scene.positions.device)
    opt = torch.optim.Adam([logits], lr=lr)
    policy = _FreeLogitPolicy(logits)
    history = []
    for step in range(steps):
        opt.zero_grad()
        res = run_dynamic_mc(scene, ev, policy, proto, phy, num_trials=train_trials,
                             generator=torch.Generator().manual_seed(base_seed + step),
                             service_profile=profile, participation=omega, reinforce=True)
        R = res.reinforce_correct                                   # [T, N] per-node correct
        logp = res.reinforce_logp                                  # [T, N] differentiable
        advantage = (R - R.mean(dim=0, keepdim=True)).detach()     # per-node baseline (variance reduction)
        loss = -(omega.unsqueeze(0) * advantage * logp).sum(dim=1).mean()
        loss.backward()
        opt.step()
        history.append(float(R.mean()))                            # in-sample per-node correct mass (proxy)
    return {"logits": logits.detach(), "history": history, "num_edges": int(gc.num_edges), "init": init}
