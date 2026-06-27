"""G-CANONICAL-CLOSURE / Phase 1: ConsensusServiceProfile contract.

The single configuration object that fixes the protocol, participation rule, basin
thresholds, reliability budgets and deadline (spec §6, plan §2). Tests pin the
spec §4 basin-disjointness invariant (ρ_s > 1−ρ_f), the strict-majority quorum,
and a deterministic config hash for the checkpoint manifest (plan §2 "profile hash
into checkpoint manifest").
"""

import math

import pytest

from src.config.service_profile import ConsensusServiceProfile


def test_default_profile_is_spec_valid():
    p = ConsensusServiceProfile.urban_default()
    # spec §4: ρ_f > 1/2, ρ_s < 1/2, and the disjointness condition ρ_s > 1 − ρ_f
    assert p.correct_basin_mass > 0.5
    assert p.split_basin_mass < 0.5
    assert p.split_basin_mass > 1.0 - p.correct_basin_mass
    # strict majority quorum 2α > k
    assert 2 * p.alpha > p.k
    assert 0.0 < p.latency_quantile < 1.0
    for eps in (p.max_wrong_basin_probability, p.max_split_basin_probability,
                p.max_deadline_miss_probability):
        assert 0.0 < eps < 1.0


def test_basin_overlap_rejected():
    # ρ_s = 1 − ρ_f would let a state be in both the split and a decisive basin
    with pytest.raises(ValueError, match="disjoint|basin"):
        ConsensusServiceProfile.urban_default().replace(
            correct_basin_mass=0.6, split_basin_mass=0.4)  # 0.4 == 1-0.6, not strictly greater


def test_correct_basin_must_be_majority():
    with pytest.raises(ValueError, match="correct_basin_mass|rho_f|> 0.5"):
        ConsensusServiceProfile.urban_default().replace(correct_basin_mass=0.5)


def test_split_basin_must_be_below_half():
    with pytest.raises(ValueError, match="split_basin_mass|rho_s|< 0.5"):
        ConsensusServiceProfile.urban_default().replace(
            correct_basin_mass=0.6, split_basin_mass=0.5)


def test_non_strict_majority_quorum_rejected():
    with pytest.raises(ValueError, match="majority|2.*alpha"):
        ConsensusServiceProfile.urban_default().replace(k=4, alpha=2)  # 2*2 == 4, not > 4


def test_config_hash_is_deterministic_and_sensitive():
    a = ConsensusServiceProfile.urban_default()
    b = ConsensusServiceProfile.urban_default()
    assert a.config_hash() == b.config_hash()
    # changing any field changes the hash (manifest must detect train/eval drift)
    assert a.config_hash() != a.replace(k=5, alpha=3).config_hash()
    assert a.config_hash() != a.replace(poll_window_ms=20.0).config_hash()
    assert a.config_hash() != a.replace(participation_weight_rule="application").config_hash()


def test_deadline_to_epochs_floor():
    p = ConsensusServiceProfile.urban_default().replace(poll_window_ms=10.0)
    # R_d = floor(T_d / Δ_poll)  (spec §5.3)
    assert p.epochs_for_deadline(105.0) == 10
    assert p.epochs_for_deadline(100.0) == 10
    assert p.epochs_for_deadline(99.9) == 9
    q = ConsensusServiceProfile.from_deadline(deadline_ms=200.0, poll_window_ms=10.0)
    assert q.max_poll_epochs == 20


def test_participation_rule_validated():
    with pytest.raises(ValueError, match="participation"):
        ConsensusServiceProfile.urban_default().replace(participation_weight_rule="policy")


def test_profile_is_frozen():
    p = ConsensusServiceProfile.urban_default()
    with pytest.raises(Exception):
        p.k = 7  # frozen dataclass — config is immutable once built
