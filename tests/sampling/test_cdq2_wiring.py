"""S14: CDQ 2.0 wired into the canonical path (the prerequisite for the Phase-10 factorial).

Acceptance: the bucketed CDQ 2.0 inclusion / quorum match the single-source CDQ 2.0 math (with
EXACT padded-slot exclusion); at eta=0 the bucketed inclusion equals the ESP inclusion and a
CDQ2Policy reproduces the ESP canonical episode bit-for-bit (the wiring anchor); a real (eta>0)
CDQ2 episode runs and is differentiable; and a CDQ2Policy runs in the dynamic MC with the basin
outcomes summing to 1.
"""

import math
from itertools import combinations

import pytest
import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
    run_consensus_episode,
)
from src.environment.candidate_graph import build_candidate_graph
from src.mainline.global_evaluator import build_bucketed_padding
from src.mainline.symmetric_polynomials import enumerate_subset_distribution
from src.metrics.participation import uniform_participation
from src.sampling import DistanceQueryPolicy
from src.sampling.cdq2_kernel import cdq2_enumerate_distribution, cdq2_inclusion
from src.sampling.cdq2_wiring import CDQ2Policy, cdq2_bucketed_quorum, cdq2_edge_inclusion
from src.sampling.esp_query import edge_inclusion_probabilities
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)


def _scene(per=4, seed=0):
    return build_manhattan_scene(2, 2, per, block_m=130.0, comm_radius=70.0, int_radius=110.0,
                                 generator=torch.Generator().manual_seed(seed))


def _maxdeg(gc, N):
    return int(torch.bincount(gc.src_index, minlength=N).max())


def _rand_kernel(gc, r, seed):
    g = torch.Generator().manual_seed(seed)
    quality = torch.rand(gc.num_edges, generator=g, dtype=torch.float64) * 2 + 0.2
    diversity = torch.randn(gc.num_edges, r, generator=g, dtype=torch.float64)
    return quality, diversity


# ---------------------------------------------------- bucketed inclusion vs per-source + sum=k
def test_cdq2_edge_inclusion_matches_per_source_and_sums_to_k():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    quality, diversity = _rand_kernel(gc, r=4, seed=1)
    eta = 1.3
    pi = cdq2_edge_inclusion(gc.src_index, gc.dst_index, N, quality, diversity, eta, k)
    for i in range(N):
        em = (gc.src_index == i).nonzero().reshape(-1)
        if em.numel() == 0:
            continue
        ref = cdq2_inclusion(quality[em], diversity[em], eta, k)     # real edges, no padding
        assert torch.allclose(pi[em], ref, atol=1e-9)
        assert abs(float(pi[em].sum()) - k) < 1e-9


def test_cdq2_edge_inclusion_padded_exclusion_exact():
    """Sources of different degree share a bucket (padded to the bucket width). The padded slots
    must contribute EXACTLY nothing: real-edge inclusions sum to k and match the per-source value."""
    scene = _scene(per=5, seed=4)
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    deg = torch.bincount(gc.src_index, minlength=N)
    assert int(deg.max()) > int(deg[deg > 0].min())             # degree skew => real padding exists
    k = min(3, int(deg[deg > 0].min()))
    quality, diversity = _rand_kernel(gc, r=4, seed=2)
    pi = cdq2_edge_inclusion(gc.src_index, gc.dst_index, N, quality, diversity, 0.9, k)
    for i in range(N):
        em = (gc.src_index == i).nonzero().reshape(-1)
        if em.numel() == 0:
            continue
        assert abs(float(pi[em].sum()) - k) < 1e-9
        assert torch.allclose(pi[em], cdq2_inclusion(quality[em], diversity[em], 0.9, k), atol=1e-9)


def test_cdq2_edge_inclusion_eta_zero_equals_esp():
    """eta=0 => L=diag(a) => the CDQ 2.0 inclusion equals the ESP inclusion EXACTLY (for any Z)."""
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    quality, diversity = _rand_kernel(gc, r=4, seed=7)
    pi_cdq2 = cdq2_edge_inclusion(gc.src_index, gc.dst_index, N, quality, diversity, 0.0, k)
    pi_esp = edge_inclusion_probabilities(gc.src_index, gc.dst_index, N, torch.log(quality), k)
    assert torch.allclose(pi_cdq2, pi_esp, atol=1e-9)


# ---------------------------------------------------- bucketed quorum smoke + per-source
def test_cdq2_bucketed_quorum_runs_and_normalises():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    alpha = 2
    quality, diversity = _rand_kernel(gc, r=4, seed=2)
    padding = build_bucketed_padding(gc.src_index, gc.dst_index, N)
    Q = 2
    g = torch.Generator().manual_seed(3)
    ell = torch.rand(gc.num_edges, Q, generator=g, dtype=torch.float64) * 0.5 + 0.4
    pref_c = torch.rand(N, Q, generator=g, dtype=torch.float64)
    pref_w = 1 - pref_c
    hp, hm, hz = cdq2_bucketed_quorum(padding, quality, diversity, 1.0, ell, pref_c, pref_w, k, alpha)
    assert torch.allclose(hp + hm + hz, torch.ones(N, Q, dtype=torch.float64), atol=1e-9)
    assert bool((hp >= -1e-12).all()) and bool((hm >= -1e-12).all())


# ---------------------------------------------------- eta=0 CDQ2 episode == ESP episode (anchor)
def test_cdq2_episode_eta_zero_equals_esp_episode():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    maxdeg = _maxdeg(gc, N)
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    pcfg = ProtocolConfig(k=k, alpha=2, beta=3, r_max=12)
    base = DistanceQueryPolicy(beta_per_m=0.03)
    esp = run_consensus_episode(scene, ev, base, pcfg, PHY, return_trajectory=False)
    cdq2 = run_consensus_episode(scene, ev, CDQ2Policy(base, r=maxdeg, eta=0.0), pcfg, PHY,
                                 return_trajectory=False)
    assert cdq2.mechanism_trace["query_law"] == "cdq2"
    assert abs(float(esp.S_allcorrect) - float(cdq2.S_allcorrect)) < 1e-9
    assert abs(float(esp.F_wrong) - float(cdq2.F_wrong)) < 1e-9
    assert abs(float(esp.F_disagree) - float(cdq2.F_disagree)) < 1e-9
    assert torch.allclose(esp.c_ir, cdq2.c_ir, atol=1e-9)


def test_cdq2_episode_eta_positive_runs_and_differentiable():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    r = 4
    g = torch.Generator().manual_seed(5)
    raw_q = torch.rand(gc.num_edges, generator=g, dtype=torch.float64).requires_grad_(True)
    div = torch.randn(gc.num_edges, r, generator=g, dtype=torch.float64).requires_grad_(True)

    class LearnCDQ2:
        query_law = "cdq2"
        name = "learn_cdq2"
        eta = 1.5

        def kernel(self, graph):
            return torch.nn.functional.softplus(raw_q) + 0.1, div

    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    res = run_consensus_episode(scene, ev, LearnCDQ2(), ProtocolConfig(k=k, alpha=2, beta=3, r_max=10),
                                PHY, return_trajectory=False)
    assert res.mechanism_trace["query_law"] == "cdq2"
    res.F_wrong.backward()
    assert raw_q.grad is not None and bool(torch.isfinite(raw_q.grad).all())
    assert div.grad is not None and bool(torch.isfinite(div.grad).all())
    assert float(raw_q.grad.abs().sum()) > 0


def test_cdq2_episode_ideal_link_backprop_finite():
    """Regression (wiring audit): under link_override=1.0 (ideal-link ablation) ell=1 => p0=0 on
    every real candidate, hitting the c=0 sqrt-gradient corner. The episode must backprop FINITE
    gradients (no NaN)."""
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    r = 4
    g = torch.Generator().manual_seed(6)
    raw_q = torch.rand(gc.num_edges, generator=g, dtype=torch.float64).requires_grad_(True)
    div = torch.randn(gc.num_edges, r, generator=g, dtype=torch.float64).requires_grad_(True)

    class LearnCDQ2:
        query_law = "cdq2"
        name = "learn_cdq2"
        eta = 1.2

        def kernel(self, graph):
            return torch.nn.functional.softplus(raw_q) + 0.1, div

    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    res = run_consensus_episode(scene, ev, LearnCDQ2(), ProtocolConfig(k=k, alpha=2, beta=3, r_max=8),
                                PHY, return_trajectory=False, link_override=1.0)
    res.F_wrong.backward()
    assert raw_q.grad is not None and bool(torch.isfinite(raw_q.grad).all())
    assert div.grad is not None and bool(torch.isfinite(div.grad).all())


# ---------------------------------------------------- dynamic MC: sampler law + basin sum-to-1
def test_cdq2_mc_sampler_law_eta_zero_equals_esp():
    """The MC sampler enumerates the exact CDQ 2.0 k-DPP law; at eta=0 it equals the ESP subset law."""
    g = torch.Generator().manual_seed(8)
    q = 0.3 + 1.5 * torch.rand(6, generator=g, dtype=torch.float64)
    Z = torch.randn(6, 3, generator=g, dtype=torch.float64)
    k = 3
    cdq2_dist = cdq2_enumerate_distribution(q, Z, 0.0, k)
    esp_dist = enumerate_subset_distribution(torch.log(q), k)
    for S, p in esp_dist.items():
        assert abs(cdq2_dist[S] - p) < 1e-10


def test_cdq2_policy_runs_in_dynamic_mc_basins_sum_to_one():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    maxdeg = _maxdeg(gc, N)
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.8)
    pcfg = ProtocolConfig(k=k, alpha=2, beta=3, r_max=8)
    prof = ConsensusServiceProfile.urban_default().replace(k=k, alpha=2, beta=3, max_poll_epochs=8)
    omega = uniform_participation(N)
    mc = run_dynamic_mc(scene, ev, CDQ2Policy(base_log_weight_policy=DistanceQueryPolicy(beta_per_m=0.03),
                                              r=maxdeg, eta=0.0),
                        pcfg, PHY, num_trials=200, generator=torch.Generator().manual_seed(0),
                        link_override=0.9, service_profile=prof, participation=omega)
    total = (float(mc.basin_P_correct) + float(mc.basin_F_wrong)
             + float(mc.basin_F_split) + float(mc.basin_F_deadline))
    assert abs(total - 1.0) < 1e-9
    assert 0.0 <= float(mc.S_allcorrect) <= 1.0
