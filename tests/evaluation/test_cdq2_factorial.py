"""G-CDQ2-EVALUATION (S15): the fair CDQ 2.0 vs ESP factorial harness + its HONEST invariants.

This test validates the harness and the structural claims that MUST hold (truth-independent
observable diversity, matched-marginal control premise, basins sum to 1, eta=0 == ESP, CRN/MC as
judge). It deliberately does NOT assert an F_wrong reduction -- the rigorous MC evidence
(docs/gate_evidence/macrostate/cdq2_factorial_results.json) shows CDQ 2.0's benefit is a SCOPED
P_correct gain via faster quorum, with F_wrong NOT reduced; that boundary is documented, not faked.
"""

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
)
from src.environment.candidate_graph import build_candidate_graph
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.evaluation import (
    FactorialResult,
    esp_vs_cdq2_cell,
    observable_group_diversity,
    run_factorial_cell,
    wilson_ci,
)
from src.metrics.participation import uniform_participation
from src.sampling import DistanceQueryPolicy

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)


def _scene(seed=0):
    return build_manhattan_scene(2, 2, 3, block_m=120.0, comm_radius=100.0, int_radius=150.0,
                                 generator=torch.Generator().manual_seed(seed))


def _setup(scene, arm):
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    deg = torch.bincount(gc.src_index, minlength=scene.num_nodes)
    k = min(3, int(deg[deg > 0].min()))
    proto = ProtocolConfig(k=k, alpha=2, beta=3, r_max=10)
    prof = ConsensusServiceProfile.urban_default().replace(k=k, alpha=2, beta=3, max_poll_epochs=10)
    omega = uniform_participation(scene.num_nodes)
    model = build_overlapping_scenario(scene, arm, base_node_err=0.35, corr_strength=0.3,
                                       n_sensor=3, n_map=3)
    return gc, k, proto, prof, omega, model


def test_observable_diversity_is_truth_independent():
    """C2: the diversity embedding uses ONLY observable group labels -- two models with the SAME
    labels but DIFFERENT error probabilities (different truth distribution) give the SAME Z."""
    sc = _scene()
    gc = build_candidate_graph(sc.positions, sc.comm_radius)
    lo = build_overlapping_scenario(sc, "matched_marginal_low", base_node_err=0.2, corr_strength=0.15)
    hi = build_overlapping_scenario(sc, "matched_marginal_high", base_node_err=0.2, corr_strength=0.15)
    # same spatial bands => same sensor/map labels, but different p_sensor/p_node (different truth)
    assert torch.equal(lo.sensor_of, hi.sensor_of) and torch.equal(lo.map_of, hi.map_of)
    z_lo = observable_group_diversity(lo)(gc)
    z_hi = observable_group_diversity(hi)(gc)
    assert torch.equal(z_lo, z_hi)                     # embedding independent of the truth distribution
    assert z_lo.shape == (gc.num_edges, 6)             # sensor(3) + map(3) one-hots


def test_matched_marginal_pair_has_identical_marginal():
    """The control premise: matched_marginal_low / _high share q_i (only covariance differs)."""
    sc = _scene()
    lo = build_overlapping_scenario(sc, "matched_marginal_low", base_node_err=0.3, corr_strength=0.2)
    hi = build_overlapping_scenario(sc, "matched_marginal_high", base_node_err=0.3, corr_strength=0.2)
    assert torch.allclose(lo.correct_observation_prob(), hi.correct_observation_prob(), atol=1e-12)


def test_wilson_ci_sanity():
    lo, hi = wilson_ci(0.5, 1000)
    assert lo < 0.5 < hi and (hi - lo) < 0.1
    lo2, hi2 = wilson_ci(0.5, 100000)
    assert (hi2 - lo2) < (hi - lo)                     # tighter with more trials


def test_basins_sum_to_one_and_result_shape():
    sc = _scene()
    gc, k, proto, prof, omega, model = _setup(sc, "matched_marginal_high")
    res = run_factorial_cell(sc, model, DistanceQueryPolicy(beta_per_m=0.05), proto, PHY, prof, omega,
                             trials=400, seeds=[0, 1], link_override=0.85)
    assert isinstance(res, FactorialResult)
    assert abs(res.basins_sum() - 1.0) < 1e-9
    lo, hi = res.ci("P_correct")
    assert lo <= res.P_correct <= hi
    assert res.n_pool == 800


def test_eta_zero_cell_matches_esp_in_mc():
    """eta=0 => CDQ2 kernel == diag(a) == ESP, so the MC basin outcomes agree within sampling noise
    (the analytic episode is bit-for-bit; the MC samplers differ in RNG path only)."""
    sc = _scene()
    gc, k, proto, prof, omega, model = _setup(sc, "matched_marginal_high")
    base = DistanceQueryPolicy(beta_per_m=0.05)
    div = observable_group_diversity(model)
    esp, cdq2_0 = esp_vs_cdq2_cell(sc, model, base, proto, PHY, prof, omega, eta=0.0, diversity=div,
                                   r=6, trials=1500, seeds=[0, 1, 2], link_override=0.85)
    assert abs(esp.P_correct - cdq2_0.P_correct) < 0.04   # same law => agree within MC noise
    assert abs(esp.F_wrong - cdq2_0.F_wrong) < 0.04


def test_harness_runs_esp_vs_cdq2_and_reports_all_basins():
    """Smoke: the harness produces a fair ESP-vs-CDQ2 cell with all four basins (the honest report
    surface). No F_wrong-reduction is asserted -- the documented finding is a SCOPED P_correct gain
    via the deadline channel, with F_wrong not reduced (see cdq2_factorial_results.json)."""
    sc = _scene()
    gc, k, proto, prof, omega, model = _setup(sc, "matched_marginal_high")
    esp, cdq2 = esp_vs_cdq2_cell(sc, model, DistanceQueryPolicy(beta_per_m=0.05), proto, PHY, prof,
                                 omega, eta=8.0, diversity=observable_group_diversity(model), r=6,
                                 trials=800, seeds=[0, 1], link_override=0.85)
    for res in (esp, cdq2):
        assert abs(res.basins_sum() - 1.0) < 1e-9
        assert all(0.0 <= getattr(res, b) <= 1.0 for b in ("P_correct", "F_wrong", "F_split", "F_deadline"))
