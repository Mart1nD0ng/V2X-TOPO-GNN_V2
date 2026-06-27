"""G-CORRELATED-ENV: scene-level overlapping scenario matrix + Mechanism Contract C1/C4 (spec §7).

The scenario builder produces, on a real scene, a correlation STRENGTH SWEEP and a
MATCHED-MARGINAL pair (identical marginal correctness, different covariance) — the causal
controls that isolate the diversity mechanism — and the overlapping model drops into the
canonical episode + dynamic MC unchanged (duck-typed).
"""

import torch

from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene, run_consensus_episode
from src.environment.overlapping_evidence import (
    OVERLAPPING_SCENARIOS,
    build_overlapping_scenario,
    overlapping_pairwise_correlation_matrix,
)
from src.sampling import UniformQueryPolicy
from src.validation import run_dynamic_mc


def _scene(seed=0):
    return build_manhattan_scene(3, 3, 3, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                                 generator=torch.Generator().manual_seed(seed))


def _mean_abs_offdiag_corr(model):
    R = overlapping_pairwise_correlation_matrix(model)
    N = R.shape[0]
    off = R[~torch.eye(N, dtype=torch.bool)]
    return float(off.abs().mean())


def test_scenario_matrix_builds_all():
    scene = _scene()
    # matched-marginal scenarios need corr_strength <= base_node_err (a shared bit cannot carry
    # more error than the node's total marginal error)
    for name in OVERLAPPING_SCENARIOS:
        m = build_overlapping_scenario(scene, name, base_node_err=0.35, corr_strength=0.15)
        assert m.num_nodes == scene.num_nodes


def test_matched_marginal_pair_same_marginal_different_covariance():
    """C1/C4 causal control: matched_marginal_low vs _high have IDENTICAL q_i, different corr."""
    scene = _scene()
    lo = build_overlapping_scenario(scene, "matched_marginal_low", base_node_err=0.3, corr_strength=0.2)
    hi = build_overlapping_scenario(scene, "matched_marginal_high", base_node_err=0.3, corr_strength=0.2)
    assert torch.allclose(lo.correct_observation_prob(), hi.correct_observation_prob(), atol=1e-12)
    assert _mean_abs_offdiag_corr(hi) > _mean_abs_offdiag_corr(lo) + 0.01
    assert _mean_abs_offdiag_corr(lo) < 1e-9          # the low arm is the zero-structure control


def test_correlation_strength_sweep_is_monotone():
    """C1 strength sweep: in the operating regime (shared-bit error below the de-correlating
    max-entropy turnover near p=0.5) rising corr_strength raises the design correlation."""
    scene = _scene()
    prev = -1.0
    for cs in (0.0, 0.05, 0.1, 0.15):
        m = build_overlapping_scenario(scene, "overlapping_sensor_source", base_node_err=0.1,
                                       corr_strength=cs)
        c = _mean_abs_offdiag_corr(m)
        assert c >= prev - 1e-9
        prev = c
    assert prev > 0.0                                  # non-trivial correlation at the top


def test_overlapping_breaks_exchangeability():
    """Same-road pairs are NOT exchangeable: their correlation varies with whether they ALSO
    share a sensor/map (the lever the region-block model lacked)."""
    scene = _scene()
    m = build_overlapping_scenario(scene, "overlapping_sensor_source", base_node_err=0.1,
                                   corr_strength=0.3)
    road = m.road_of
    # find two same-road pairs that differ in sensor co-membership
    R = overlapping_pairwise_correlation_matrix(m)
    same_road = (road.unsqueeze(0) == road.unsqueeze(1))
    same_sensor = (m.sensor_of.unsqueeze(0) == m.sensor_of.unsqueeze(1))
    eye = torch.eye(m.num_nodes, dtype=torch.bool)
    a = R[same_road & same_sensor & ~eye]
    b = R[same_road & ~same_sensor & ~eye]
    assert a.numel() > 0 and b.numel() > 0
    assert float(a.mean()) > float(b.mean()) + 1e-6    # extra shared cause -> higher correlation


def test_correlated_evidence_sentinel_accurate_for_sensor_only_correlation():
    """Constraint #13 regression: the mechanism-activation sentinel must report correlated_evidence
    even when the correlation lives ONLY in the sensor/map common causes (p_road = 0)."""
    scene = _scene()
    ev = build_overlapping_scenario(scene, "overlapping_sensor_source", base_node_err=0.1,
                                    corr_strength=0.2)
    assert bool((ev.p_road == 0).all())            # no ROAD correlation in this scenario
    assert ev.has_correlated_evidence()            # but there IS correlation (sensor/map)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=8)
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    res = run_consensus_episode(scene, ev, UniformQueryPolicy(), proto, phy,
                                return_trajectory=False, link_override=0.9)
    assert res.mechanism_trace["correlated_evidence"] is True
    # an iid scenario (zero structure) must report False
    iid = build_overlapping_scenario(scene, "iid", base_node_err=0.1)
    res2 = run_consensus_episode(scene, iid, UniformQueryPolicy(), proto, phy,
                                 return_trajectory=False, link_override=0.9)
    assert res2.mechanism_trace["correlated_evidence"] is False


def test_overlapping_model_runs_in_canonical_episode_and_mc():
    scene = _scene()
    ev = build_overlapping_scenario(scene, "overlapping_sensor_source", base_node_err=0.1,
                                    corr_strength=0.2)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=8)
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    res = run_consensus_episode(scene, ev, UniformQueryPolicy(), proto, phy,
                                return_trajectory=True, link_override=0.9)
    assert bool(torch.isfinite(res.F_wrong))
    mc = run_dynamic_mc(scene, ev, UniformQueryPolicy(), proto, phy, num_trials=200,
                        generator=torch.Generator().manual_seed(0), link_override=0.9)
    assert 0.0 <= mc.S_allcorrect <= 1.0
