"""G-CORRELATED-ENV: overlapping common-cause evidence model (spec §7).

    O_i = Y* ⊕ B_road(i) ⊕ B_sensor(i) ⊕ B_map(i) ⊕ E_i

Each node belongs to a road / sensor / map group; the shared group bits make co-group peers'
evidence errors POSITIVELY correlated, and — crucially — the group memberships OVERLAP (two
nodes can share a sensor but not a road), so same-road peers are NO LONGER exchangeable. This
fixes the prior round's D18 near-exchangeable region-block obstacle and gives determinantal
diversity a lever. The headline check is the MATCHED-MARGINAL control: identical marginal
correctness q_i, different covariance — so a marginal-only (region-aware ESP) policy cannot
distinguish two scenarios that a diversity-aware policy can.
"""

import math

import torch

import pytest

from src.environment.overlapping_evidence import (
    OverlappingEvidenceModel,
    matched_marginal_shared,
    overlapping_pairwise_correlation,
    overlapping_pairwise_correlation_matrix,
)


def _model(N=6, p_road=0.0, p_sensor=0.0, p_map=0.0, p_node=0.1,
           road=None, sensor=None, map_=None):
    road = road if road is not None else torch.zeros(N, dtype=torch.long)
    sensor = sensor if sensor is not None else torch.zeros(N, dtype=torch.long)
    map_ = map_ if map_ is not None else torch.zeros(N, dtype=torch.long)
    return OverlappingEvidenceModel(
        road_of=road, sensor_of=sensor, map_of=map_,
        p_road=torch.tensor([p_road], dtype=torch.float64),
        p_sensor=torch.tensor([p_sensor], dtype=torch.float64),
        p_map=torch.tensor([p_map], dtype=torch.float64),
        p_node=torch.full((N,), float(p_node), dtype=torch.float64))


def test_marginal_closed_form():
    # q_i = (1 + prod_k (1-2 p_k)) / 2  over road/sensor/map/node bits
    m = _model(N=3, p_road=0.2, p_sensor=0.1, p_map=0.0, p_node=0.15)
    q = m.correct_observation_prob()
    mu = (1 - 2 * 0.2) * (1 - 2 * 0.1) * (1 - 2 * 0.0) * (1 - 2 * 0.15)
    assert torch.allclose(q, torch.full((3,), (1 + mu) / 2, dtype=torch.float64), atol=1e-12)


def test_zero_structure_control_has_zero_correlation():
    # all group probs 0 -> only independent node errors -> zero pairwise correlation
    m = _model(N=4, p_road=0.0, p_sensor=0.0, p_map=0.0, p_node=0.2)
    assert abs(overlapping_pairwise_correlation(m, 0, 1)) < 1e-12


def test_shared_sensor_gives_positive_correlation_matching_theory():
    # two nodes sharing a sensor group (p_sensor>0), different road/map -> positive corr
    N = 2
    m = OverlappingEvidenceModel(
        road_of=torch.tensor([0, 1]), sensor_of=torch.tensor([0, 0]), map_of=torch.tensor([0, 1]),
        p_road=torch.tensor([0.1, 0.1], dtype=torch.float64),
        p_sensor=torch.tensor([0.25], dtype=torch.float64),
        p_map=torch.tensor([0.05, 0.05], dtype=torch.float64),
        p_node=torch.tensor([0.1, 0.1], dtype=torch.float64))
    theo = overlapping_pairwise_correlation(m, 0, 1)
    assert theo > 0.0
    # empirical correlation of the correctness indicators matches theory
    ev = m.sample(200_000, generator=torch.Generator().manual_seed(0))
    c = ev.correct.to(torch.float64)
    ci, cj = c[:, 0], c[:, 1]
    emp = float(((ci * cj).mean() - ci.mean() * cj.mean())
                / (ci.std(unbiased=False) * cj.std(unbiased=False)))
    assert abs(emp - theo) < 0.01


def test_more_shared_groups_higher_correlation():
    # share sensor only vs share sensor+map -> the latter is more correlated
    share1 = OverlappingEvidenceModel(
        road_of=torch.tensor([0, 1]), sensor_of=torch.tensor([0, 0]), map_of=torch.tensor([0, 1]),
        p_road=torch.tensor([0.1, 0.1], dtype=torch.float64),
        p_sensor=torch.tensor([0.2], dtype=torch.float64),
        p_map=torch.tensor([0.2, 0.2], dtype=torch.float64),
        p_node=torch.tensor([0.1, 0.1], dtype=torch.float64))
    share2 = OverlappingEvidenceModel(
        road_of=torch.tensor([0, 1]), sensor_of=torch.tensor([0, 0]), map_of=torch.tensor([0, 0]),
        p_road=torch.tensor([0.1, 0.1], dtype=torch.float64),
        p_sensor=torch.tensor([0.2], dtype=torch.float64),
        p_map=torch.tensor([0.2], dtype=torch.float64),
        p_node=torch.tensor([0.1, 0.1], dtype=torch.float64))
    assert overlapping_pairwise_correlation(share2, 0, 1) > overlapping_pairwise_correlation(share1, 0, 1)


def test_matched_marginal_different_covariance():
    """THE mechanism-identifiability control: SAME marginal q_i, DIFFERENT correlation.

    Move error mass from the independent node bit into a shared sensor bit while preserving
    prod(1-2p) (hence q_i) — the marginal is identical but co-sensor peers become correlated.
    """
    N = 2
    road = torch.tensor([0, 1]); sensor = torch.tensor([0, 0]); map_ = torch.tensor([0, 1])
    target_p_node = 0.3
    # low-corr: all error in the node bit, sensor clean
    lo = OverlappingEvidenceModel(
        road_of=road, sensor_of=sensor, map_of=map_,
        p_road=torch.tensor([0.0, 0.0], dtype=torch.float64),
        p_sensor=torch.tensor([0.0], dtype=torch.float64),
        p_map=torch.tensor([0.0, 0.0], dtype=torch.float64),
        p_node=torch.full((N,), target_p_node, dtype=torch.float64))
    # high-corr: split the SAME marginal between a shared sensor bit and a smaller node bit
    p_sensor, p_node_new = matched_marginal_shared(target_p_node, p_shared=0.2)
    hi = OverlappingEvidenceModel(
        road_of=road, sensor_of=sensor, map_of=map_,
        p_road=torch.tensor([0.0, 0.0], dtype=torch.float64),
        p_sensor=torch.tensor([p_sensor], dtype=torch.float64),
        p_map=torch.tensor([0.0, 0.0], dtype=torch.float64),
        p_node=torch.full((N,), p_node_new, dtype=torch.float64))
    # marginals identical
    assert torch.allclose(lo.correct_observation_prob(), hi.correct_observation_prob(), atol=1e-12)
    # but covariance differs: lo independent (0), hi positively correlated
    assert abs(overlapping_pairwise_correlation(lo, 0, 1)) < 1e-12
    assert overlapping_pairwise_correlation(hi, 0, 1) > 0.05


def test_truth_and_observable_proxy_are_separate():
    # sample() returns the TRUTH-derived correctness AND the group bits separately; the group
    # MEMBERSHIPS (road/sensor/map ids) are the deployment-observable proxy, never Y*.
    m = OverlappingEvidenceModel(
        road_of=torch.zeros(3, dtype=torch.long), sensor_of=torch.tensor([0, 0, 1]),
        map_of=torch.zeros(3, dtype=torch.long),
        p_road=torch.tensor([0.0], dtype=torch.float64),
        p_sensor=torch.tensor([0.2, 0.2], dtype=torch.float64),
        p_map=torch.tensor([0.0], dtype=torch.float64),
        p_node=torch.full((3,), 0.1, dtype=torch.float64))
    ev = m.sample(16, generator=torch.Generator().manual_seed(1))
    assert ev.correct.shape == (16, 3) and ev.correct.dtype == torch.bool
    # the observable proxy is the membership vector, not the sampled truth
    assert m.sensor_of.tolist() == [0, 0, 1]
    assert not torch.is_floating_point(m.sensor_of)  # an exogenous label


def test_correlation_robust_at_half_probability_shared_bit():
    """A pure-noise shared common cause (p=0.5) must NOT produce NaN, and the scalar and matrix
    forms must AGREE (regression for the audit's correlation-math finding)."""
    m = OverlappingEvidenceModel(
        road_of=torch.tensor([0, 1]), sensor_of=torch.tensor([0, 0]), map_of=torch.tensor([0, 1]),
        p_road=torch.tensor([0.0, 0.0], dtype=torch.float64),
        p_sensor=torch.tensor([0.5], dtype=torch.float64),   # pure-noise SHARED sensor bit
        p_map=torch.tensor([0.0, 0.0], dtype=torch.float64),
        p_node=torch.tensor([0.0, 0.0], dtype=torch.float64))
    scal = overlapping_pairwise_correlation(m, 0, 1)
    mat = float(overlapping_pairwise_correlation_matrix(m)[0, 1])
    assert math.isfinite(scal) and math.isfinite(mat)
    # both nodes are C = 1 - B_sensor (same shared coin) -> perfectly correlated
    assert abs(scal - 1.0) < 1e-9 and abs(mat - 1.0) < 1e-9


def test_scalar_and_matrix_correlation_agree_generic():
    m = OverlappingEvidenceModel(
        road_of=torch.tensor([0, 0, 1]), sensor_of=torch.tensor([0, 1, 1]), map_of=torch.tensor([0, 0, 1]),
        p_road=torch.tensor([0.1, 0.15], dtype=torch.float64),
        p_sensor=torch.tensor([0.2, 0.25], dtype=torch.float64),
        p_map=torch.tensor([0.05, 0.1], dtype=torch.float64),
        p_node=torch.tensor([0.1, 0.12, 0.08], dtype=torch.float64))
    R = overlapping_pairwise_correlation_matrix(m)
    for i in range(3):
        for j in range(3):
            assert abs(float(R[i, j]) - overlapping_pairwise_correlation(m, i, j)) < 1e-12


def test_matched_marginal_validates_inputs():
    with pytest.raises(ValueError, match="must be in"):
        matched_marginal_shared(1.5, 0.2)          # p_node_target out of [0,1]
    with pytest.raises(ValueError, match="pure-noise|0.5"):
        matched_marginal_shared(0.3, 0.5)          # pure-noise shared bit
    with pytest.raises(ValueError, match="infeasible"):
        matched_marginal_shared(0.1, 0.3)          # |s_target| > |s_shared| (p_shared>p_target<0.5)


def test_analytic_scenarios_reproduce_marginal_and_correlation():
    # the shared-latent decomposition (omega, init_cp) must reproduce the marginal q_i exactly
    m = OverlappingEvidenceModel(
        road_of=torch.tensor([0, 0]), sensor_of=torch.tensor([0, 1]), map_of=torch.tensor([0, 0]),
        p_road=torch.tensor([0.15], dtype=torch.float64),
        p_sensor=torch.tensor([0.2, 0.2], dtype=torch.float64),
        p_map=torch.tensor([0.1], dtype=torch.float64),
        p_node=torch.tensor([0.1, 0.1], dtype=torch.float64))
    omega, init_cp = m.analytic_scenarios()
    q_decomp = (omega.unsqueeze(0) * init_cp).sum(dim=1)     # E_Z[ P(correct|Z) ] = q_i
    assert torch.allclose(q_decomp, m.correct_observation_prob(), atol=1e-12)
    assert abs(float(omega.sum()) - 1.0) < 1e-12
