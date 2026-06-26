"""G2 -- named correlated-evidence scenarios (spec §6.3)."""

import pytest
import torch

from src.environment import build_manhattan_scene, build_scenario
from src.environment.evidence_model import pairwise_correlation_theory


def _scene():
    return build_manhattan_scene(4, 4, 6, generator=torch.Generator().manual_seed(0))


def test_all_correct_control():
    m = build_scenario("all_correct", _scene())
    q = m.correct_observation_prob()
    assert torch.allclose(q, torch.ones_like(q), atol=1e-12)  # everyone starts correct


def test_iid_zero_correlation():
    scene = _scene()
    m = build_scenario("iid", scene, base_node_err=0.1)
    q = m.correct_observation_prob()
    assert torch.allclose(q, torch.full_like(q, 0.9), atol=1e-12)
    # all pairwise correlations zero (no shared region bias)
    same_region = (scene.region_of == scene.region_of[0]).nonzero().reshape(-1).tolist()
    if len(same_region) >= 2:
        i, j = same_region[0], same_region[1]
        assert pairwise_correlation_theory(m, i, j) == 0.0


def test_one_biased_region():
    scene = _scene()
    m = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    q = m.correct_observation_prob()
    biased = (scene.region_of == 0).nonzero().reshape(-1).tolist()
    clean = (scene.region_of != 0).nonzero().reshape(-1).tolist()
    assert max(float(q[i]) for i in biased) < 0.3      # biased region mostly wrong
    assert min(float(q[i]) for i in clean) > 0.7       # clean regions mostly correct
    # within the biased region, observations are positively correlated (shared B_g)
    if len(biased) >= 2:
        assert pairwise_correlation_theory(m, biased[0], biased[1]) > 0.05
    # cross-region correlation is zero
    assert pairwise_correlation_theory(m, biased[0], clean[0]) == 0.0


def test_two_opposing_regions():
    scene = _scene()
    m = build_scenario("two_opposing_regions", scene, base_node_err=0.1, region_bias=0.85)
    q = m.correct_observation_prob()
    # there exist both mostly-correct and mostly-wrong nodes (opposite opinion clusters)
    assert float(q.max()) > 0.7
    assert float(q.min()) < 0.3
    n_correct = int((q > 0.5).sum())
    n_wrong = int((q < 0.5).sum())
    assert n_correct > 0 and n_wrong > 0


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        build_scenario("weak_cut", _scene())  # geometric scenario, not an evidence one
