"""G9 infra -- CDQ wired into the canonical path (spec §9; G4/G5 enter the headline).

Acceptance: the bucketed CDQ inclusion / quorum match the single-source CDQ math; the
diagonal-kernel CDQ reproduces the ESP canonical episode bit-for-bit (the wiring is correct);
a real (non-diagonal) CDQ episode runs and is differentiable in the kernel.
"""

import torch

from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
    run_consensus_episode,
)
from src.environment.candidate_graph import build_candidate_graph
from src.mainline.global_evaluator import build_bucketed_padding
from src.sampling import DistanceQueryPolicy, UniformQueryPolicy
from src.sampling.cdq_query import DiagonalCDQPolicy, cdq_bucketed_quorum, cdq_edge_inclusion
from src.sampling.dpp_query import kdpp_inclusion, low_rank_kernel
from src.sampling.determinantal_quorum import determinantal_quorum_decision

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)


def _scene(per=4, seed=0):
    return build_manhattan_scene(2, 2, per, block_m=130.0, comm_radius=70.0, int_radius=110.0,
                                 generator=torch.Generator().manual_seed(seed))


def _maxdeg(gc, N):
    return int(torch.bincount(gc.src_index, minlength=N).max())


def _rand_kernel_on_graph(gc, r, seed):
    g = torch.Generator().manual_seed(seed)
    quality = torch.rand(gc.num_edges, generator=g, dtype=torch.float64) * 2 + 0.2
    diversity = torch.randn(gc.num_edges, r, generator=g, dtype=torch.float64)
    return quality, diversity


# ------------------------------------------------ bucketed inclusion vs per-source
def test_cdq_edge_inclusion_matches_per_source_and_sums_to_k():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    quality, diversity = _rand_kernel_on_graph(gc, r=4, seed=1)
    pi = cdq_edge_inclusion(gc.src_index, gc.dst_index, N, quality, diversity, k)
    # per-source reference
    for i in range(N):
        em = (gc.src_index == i).nonzero().reshape(-1)
        if em.numel() == 0:
            continue
        ref = kdpp_inclusion(quality[em], diversity[em], k)
        assert torch.allclose(pi[em], ref, atol=1e-10)
        assert abs(float(pi[em].sum()) - k) < 1e-9


# ------------------------------------------------ bucketed quorum vs per-source
def test_cdq_bucketed_quorum_matches_per_source():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    alpha = 2
    quality, diversity = _rand_kernel_on_graph(gc, r=4, seed=2)
    padding = build_bucketed_padding(gc.src_index, gc.dst_index, N)
    Q = 2
    g = torch.Generator().manual_seed(3)
    ell = torch.rand(gc.num_edges, Q, generator=g, dtype=torch.float64) * 0.5 + 0.4
    pref_c = torch.rand(N, Q, generator=g, dtype=torch.float64)
    pref_w = 1 - pref_c
    h_plus, h_minus, h_zero = cdq_bucketed_quorum(padding, quality, diversity, ell, pref_c, pref_w, k, alpha)
    # per-source reference for a few sources/scenarios
    for i in range(N):
        em = (gc.src_index == i).nonzero().reshape(-1)
        if em.numel() == 0:
            continue
        dst = gc.dst_index[em]
        B = low_rank_kernel(quality[em], diversity[em])          # this source's kernel root
        for q in range(Q):
            pp = ell[em, q] * pref_c[dst, q]
            pm = ell[em, q] * pref_w[dst, q]
            dec = determinantal_quorum_decision(B, pp, pm, k, alpha)
            assert abs(float(h_plus[i, q]) - float(dec.h_plus)) < 1e-9
            assert abs(float(h_minus[i, q]) - float(dec.h_minus)) < 1e-9


# ------------------------------------------------ diagonal CDQ == ESP (the anchor)
def test_diagonal_cdq_episode_equals_esp_episode():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    maxdeg = _maxdeg(gc, N)
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    pcfg = ProtocolConfig(k=k, alpha=2, beta=3, r_max=12)
    base = DistanceQueryPolicy(beta_per_m=0.03)
    esp = run_consensus_episode(scene, ev, base, pcfg, PHY, return_trajectory=False)
    cdq = run_consensus_episode(scene, ev, DiagonalCDQPolicy(base, r=maxdeg), pcfg, PHY,
                                return_trajectory=False)
    assert cdq.mechanism_trace["query_law"] == "cdq"
    assert abs(float(esp.S_allcorrect) - float(cdq.S_allcorrect)) < 1e-9
    assert abs(float(esp.F_wrong) - float(cdq.F_wrong)) < 1e-9
    assert abs(float(esp.F_disagree) - float(cdq.F_disagree)) < 1e-9
    assert torch.allclose(esp.c_ir, cdq.c_ir, atol=1e-9)


def test_real_cdq_episode_runs_and_is_differentiable():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    r = 4
    g = torch.Generator().manual_seed(5)
    raw_q = (torch.rand(gc.num_edges, generator=g, dtype=torch.float64)).requires_grad_(True)
    div = torch.randn(gc.num_edges, r, generator=g, dtype=torch.float64).requires_grad_(True)

    class LearnCDQ:
        query_law = "cdq"
        name = "learn_cdq"

        def kernel(self, graph):
            return torch.nn.functional.softplus(raw_q) + 0.1, div

    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    res = run_consensus_episode(scene, ev, LearnCDQ(), ProtocolConfig(k=k, alpha=2, beta=3, r_max=10),
                                PHY, return_trajectory=False)
    assert res.mechanism_trace["query_law"] == "cdq"
    res.F_wrong.backward()
    assert raw_q.grad is not None and bool(torch.isfinite(raw_q.grad).all())
    assert div.grad is not None and bool(torch.isfinite(div.grad).all())
    # the query topology actually affects validity (non-trivial gradient)
    assert float(raw_q.grad.abs().sum()) > 0
