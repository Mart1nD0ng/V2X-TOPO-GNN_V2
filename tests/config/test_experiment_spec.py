"""G-CANONICAL-CLOSURE: ExperimentSpec train==eval compatibility (plan §4).

The headline must train and evaluate under IDENTICAL physics/protocol/profile/query-law; the
only permitted differences are explicitly-registered OOD axes (plan §11). The ideal/full-link
mismatch — the historical bug — is blocked unconditionally (ideal link is a separate ablation,
never an OOD axis). Config fingerprints are deterministic hashes carried in the checkpoint
manifest (Mechanism Contract C5).
"""

import pytest

from src.config.experiment_spec import (
    ExperimentSpec,
    IncompatibleExperimentError,
    build_experiment_spec,
    check_train_eval_compatible,
)
from src.config.service_profile import ConsensusServiceProfile
from src.environment.canonical_episode import ProtocolConfig
from src.environment.round_physics import RoundPhysicsConfig


def _spec(*, phy=None, proto=None, profile=None, evidence="iid:p=0.1",
          scene="manhattan:4x4x4", query_law="esp", full_physics=True, ood=()):
    return build_experiment_spec(
        protocol_cfg=proto or ProtocolConfig(),
        service_profile=profile or ConsensusServiceProfile.urban_default(),
        phy_cfg=phy or RoundPhysicsConfig(),
        evidence_descriptor=evidence,
        scene_descriptor=scene,
        query_law=query_law,
        full_physics=full_physics,
        allowed_ood_axes=ood,
    )


def test_config_hashes_deterministic():
    a, b = _spec(), _spec()
    assert a == b and a.config_hash() == b.config_hash()
    assert all(isinstance(getattr(a, h), str) and len(getattr(a, h)) == 64
               for h in ("protocol_hash", "service_profile_hash", "physics_hash",
                         "evidence_hash", "scene_distribution_hash"))


def test_identical_specs_are_compatible():
    check_train_eval_compatible(_spec(), _spec())  # must not raise


def test_physics_mismatch_blocks_headline():
    train = _spec(phy=RoundPhysicsConfig(tx_power_dbm=23.0))
    evalu = _spec(phy=RoundPhysicsConfig(tx_power_dbm=30.0))
    assert train.physics_hash != evalu.physics_hash
    with pytest.raises(IncompatibleExperimentError, match="physics"):
        check_train_eval_compatible(train, evalu)


def test_physics_mismatch_allowed_only_with_registered_ood_axis():
    train = _spec(phy=RoundPhysicsConfig(noise_dbm=-95.0))
    evalu = _spec(phy=RoundPhysicsConfig(noise_dbm=-90.0), ood=("interference",))
    check_train_eval_compatible(train, evalu)  # registered OOD axis -> allowed


def test_ideal_full_mismatch_always_blocked_even_with_ood():
    """The ideal/full-link mismatch is NEVER an OOD axis — it is the historical headline bug."""
    train = _spec(full_physics=True)
    evalu = _spec(full_physics=False, ood=("physics", "interference", "node_count"))
    with pytest.raises(IncompatibleExperimentError, match="ideal|full_physics|link"):
        check_train_eval_compatible(train, evalu)


def test_protocol_mismatch_blocks_unless_registered():
    train = _spec(proto=ProtocolConfig(k=4, alpha=3, beta=5, r_max=20))
    evalu = _spec(proto=ProtocolConfig(k=5, alpha=3, beta=5, r_max=20))
    with pytest.raises(IncompatibleExperimentError, match="protocol"):
        check_train_eval_compatible(train, evalu)
    check_train_eval_compatible(train, _spec(proto=ProtocolConfig(k=5, alpha=3, beta=5, r_max=20),
                                             ood=("protocol",)))


def test_query_law_mismatch_always_blocks():
    # query_law is NOT an OOD axis (a checkpoint is one policy family: ESP xor CDQ), so a
    # train/eval query-law mismatch ALWAYS blocks and "query_law" is not registerable.
    with pytest.raises(IncompatibleExperimentError, match="query_law"):
        check_train_eval_compatible(_spec(query_law="esp"), _spec(query_law="cdq"))
    with pytest.raises(ValueError, match="unknown OOD axis"):
        _spec(query_law="cdq", ood=("query_law",))


def test_scene_ood_does_not_excuse_physics():
    """Registering node_count must NOT silently let the physics differ too."""
    train = _spec(scene="manhattan:4x4x4", phy=RoundPhysicsConfig(tx_power_dbm=23.0))
    evalu = _spec(scene="manhattan:8x8x8", phy=RoundPhysicsConfig(tx_power_dbm=30.0),
                  ood=("node_count",))
    with pytest.raises(IncompatibleExperimentError, match="physics"):
        check_train_eval_compatible(train, evalu)
    # scene-only difference under node_count is fine
    check_train_eval_compatible(_spec(scene="manhattan:4x4x4"),
                                _spec(scene="manhattan:8x8x8", ood=("node_count",)))


def test_unknown_ood_axis_rejected():
    with pytest.raises(ValueError, match="unknown OOD axis|ood"):
        _spec(ood=("teleportation",))
