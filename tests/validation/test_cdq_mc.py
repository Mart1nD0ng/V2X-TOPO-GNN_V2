"""G6+CDQ -- the dynamic MC extended to the CDQ k-DPP query law (spec §8, constraint #14).

Acceptance: the CDQ subset sampler draws exactly-k distinct peers from the true k-DPP; the
diagonal-kernel CDQ MC reproduces the (validated) ESP MC; the CDQ MC agrees with the analytic
CDQ episode where the mean-field is exact; it runs for a real ESD-GNN policy and is reproducible.
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
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.sampling import DiagonalCDQPolicy, DistanceQueryPolicy
from src.validation import run_dynamic_mc
from src.validation.dynamic_mc import _CDQSubsetSampler

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)


def _scene(per=4, seed=0):
    return build_manhattan_scene(2, 2, per, block_m=130.0, comm_radius=75.0, int_radius=115.0,
                                 generator=torch.Generator().manual_seed(seed))


def _maxdeg(gc, N):
    return int(torch.bincount(gc.src_index, minlength=N).max())


def test_cdq_sampler_draws_exactly_k_distinct():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    from src.mainline.global_evaluator import build_source_padding
    pad = build_source_padding(gc.src_index, gc.dst_index, N)
    g = torch.Generator().manual_seed(0)
    quality = torch.rand(gc.num_edges, generator=g, dtype=torch.float64) + 0.2
    diversity = torch.randn(gc.num_edges, 4, generator=g, dtype=torch.float64)
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    sampler = _CDQSubsetSampler(quality, diversity, pad.slot_edge, pad.slot_mask, k)
    chosen = sampler.sample(50, torch.Generator().manual_seed(1))      # [50, N, nmax]
    # exactly k chosen per (trial, node), all within valid slots
    assert torch.all(chosen.sum(dim=-1) == k)
    assert torch.all(chosen <= pad.slot_mask.unsqueeze(0))


def test_diagonal_cdq_mc_matches_esp_mc():
    """The CDQ k-DPP sampler reproduces the validated ESP ancestral sampler on a diagonal kernel."""
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    N = scene.num_nodes
    maxdeg = _maxdeg(gc, N)
    k = min(3, int(torch.bincount(gc.src_index, minlength=N).min()))
    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    pcfg = ProtocolConfig(k=k, alpha=2, beta=3, r_max=10)
    base = DistanceQueryPolicy(beta_per_m=0.03)
    esp = run_dynamic_mc(scene, ev, base, pcfg, PHY, num_trials=8000,
                         generator=torch.Generator().manual_seed(7), link_override=1.0)
    cdq = run_dynamic_mc(scene, ev, DiagonalCDQPolicy(base, r=maxdeg), pcfg, PHY, num_trials=8000,
                         generator=torch.Generator().manual_seed(7), link_override=1.0)
    # same subset law -> stats agree within MC sampling error (CIs overlap)
    assert cdq.F_wrong_ci[0] <= esp.F_wrong <= cdq.F_wrong_ci[1]
    assert abs(cdq.S_allcorrect - esp.S_allcorrect) < 0.03


def test_cdq_mc_agrees_with_analytic_on_all_correct():
    scene = _scene()
    ev = build_scenario("all_correct", scene)
    pcfg = ProtocolConfig(k=min(3, 2), alpha=2, beta=3, r_max=10)
    model = ESDGNN(ESDGNNConfig(hidden_dim=16, r=4, n_enc=2, n_refine=1, k=pcfg.k)).double()
    torch.manual_seed(0)
    pol = ESDGNNQueryPolicy(model, scene)
    an = run_consensus_episode(scene, ev, pol, pcfg, PHY, return_trajectory=False, link_override=1.0)
    mc = run_dynamic_mc(scene, ev, pol, pcfg, PHY, num_trials=3000,
                        generator=torch.Generator().manual_seed(2), link_override=1.0)
    assert float(an.S_allcorrect) > 0.99 and mc.S_allcorrect > 0.99   # perfect evidence -> agree


def test_cdq_mc_runs_for_esd_gnn_and_is_reproducible():
    scene = _scene()
    pcfg = ProtocolConfig(k=min(3, 2), alpha=2, beta=3, r_max=8)
    torch.manual_seed(0)
    model = ESDGNN(ESDGNNConfig(hidden_dim=16, r=4, n_enc=2, n_refine=1, k=pcfg.k)).double()
    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    a = run_dynamic_mc(scene, ev, ESDGNNQueryPolicy(model, scene), pcfg, PHY, num_trials=2000,
                       generator=torch.Generator().manual_seed(5), link_override=1.0)
    b = run_dynamic_mc(scene, ev, ESDGNNQueryPolicy(model, scene), pcfg, PHY, num_trials=2000,
                       generator=torch.Generator().manual_seed(5), link_override=1.0)
    assert a.F_wrong == b.F_wrong and a.S_allcorrect == b.S_allcorrect   # reproducible
    assert 0.0 <= a.F_wrong <= 1.0
