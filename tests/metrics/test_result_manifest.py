"""G-RESULT-MANIFEST (Guarded-CDQ2 round, Phase 2): hash & provenance enforcement.

Every result JSON must be reproducible and hash-bound (spec §7.4 + plan §3): physics / service-profile /
evidence / scene-distribution / protocol / policy / checkpoint hashes, a provenance id (git commit or
manifest id), the query family, the tracked model seeds, and complete macro outcomes. Train/eval physics
mismatch must fail-fast unless the differing axis is a registered OOD axis (constraint #9).
"""

import pytest

from src.config.experiment_spec import ExperimentSpec, IncompatibleExperimentError
from src.metrics import manifest as mf
from src.metrics import schema


def _spec(**over) -> ExperimentSpec:
    base = dict(protocol_hash="ph", service_profile_hash="sph", physics_hash="phys",
                evidence_hash="ev", scene_distribution_hash="sc", query_law="esp",
                full_physics=True, allowed_ood_axes=())
    base.update(over)
    return ExperimentSpec(**base)


def _record(manifest):
    return schema.build_result_record(policy="ESP", query_family="ESP",
                                      macro=schema.macro_block(0.62, 0.20, 0.03, 0.15),
                                      hashes=manifest)


# ---------------------------------------------------------------- manifest construction
def test_build_manifest_carries_every_required_hash():
    m = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt",
                          model_seeds=[0, 1, 2, 3, 4], git_commit="abc1234")
    for key in mf.REQUIRED_HASH_KEYS:
        assert m[key], key
    assert m["model_seeds"] == [0, 1, 2, 3, 4]
    assert m["query_law"] == "esp"
    assert m["git_commit"] == "abc1234"
    assert m["experiment_config_hash"] == _spec().config_hash()


def test_validate_manifest_accepts_complete_record():
    m = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt",
                          model_seeds=[0, 1, 2, 3, 4], git_commit="abc1234")
    rec = _record(m)
    mf.validate_manifest(rec)                 # must not raise
    schema.validate_result(rec)               # also a clean headline record


# ---------------------------------------------------------------- missing-field fail-fast
@pytest.mark.parametrize("drop", ["physics_hash", "service_profile_hash", "evidence_hash",
                                  "scene_distribution_hash", "policy_hash", "checkpoint_hash"])
def test_missing_required_hash_fails(drop):
    m = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt",
                          model_seeds=[0], git_commit="abc1234")
    del m[drop]
    with pytest.raises(schema.MetricSchemaError):
        mf.validate_manifest(_record(m))


def test_empty_hash_value_fails():
    m = mf.build_manifest(_spec(), policy_hash="", checkpoint_hash="ckpt",
                          model_seeds=[0], git_commit="abc1234")
    with pytest.raises(schema.MetricSchemaError):
        mf.validate_manifest(_record(m))


def test_missing_provenance_id_fails():
    m = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt", model_seeds=[0])
    # neither git_commit nor manifest_id
    with pytest.raises(schema.MetricSchemaError):
        mf.validate_manifest(_record(m))
    m2 = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt", model_seeds=[0],
                           manifest_id="GS2-run-001")
    mf.validate_manifest(_record(m2))         # manifest_id alone is sufficient provenance


# ---------------------------------------------------------------- model-seed tracking
def test_untracked_model_seed_fails():
    m = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt",
                          model_seeds=[], git_commit="abc")
    with pytest.raises(schema.MetricSchemaError):
        mf.validate_manifest(_record(m), require_seeds=True)


def test_duplicate_model_seed_fails():
    m = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt",
                          model_seeds=[0, 0, 1], git_commit="abc")
    with pytest.raises(schema.MetricSchemaError):
        mf.validate_manifest(_record(m))


def test_min_seeds_enforced_for_headline():
    m = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt",
                          model_seeds=[0, 1], git_commit="abc")
    with pytest.raises(schema.MetricSchemaError):
        mf.validate_manifest(_record(m), min_seeds=5)   # single-seed headline forbidden
    m5 = mf.build_manifest(_spec(), policy_hash="pol", checkpoint_hash="ckpt",
                           model_seeds=[0, 1, 2, 3, 4], git_commit="abc")
    mf.validate_manifest(_record(m5), min_seeds=5)


# ---------------------------------------------------------------- train/eval consistency
def test_train_eval_physics_mismatch_fails_unless_ood():
    train = _spec(physics_hash="A")
    ev = _spec(physics_hash="B")              # different physics, no registered axis
    m = mf.build_manifest(ev, policy_hash="pol", checkpoint_hash="ckpt", model_seeds=[0], git_commit="x")
    rec = _record(m)
    with pytest.raises(IncompatibleExperimentError):
        mf.assert_train_eval_consistent(rec, train, ev)
    # registering the physics/interference axis lets it through
    ev_ood = _spec(physics_hash="B", allowed_ood_axes=("interference",))
    m2 = mf.build_manifest(ev_ood, policy_hash="pol", checkpoint_hash="ckpt", model_seeds=[0], git_commit="x")
    mf.assert_train_eval_consistent(_record(m2), train, ev_ood)


def test_ideal_full_link_mismatch_always_fails():
    train = _spec(full_physics=True)
    ev = _spec(full_physics=False, allowed_ood_axes=("physics", "interference"))
    m = mf.build_manifest(ev, policy_hash="pol", checkpoint_hash="ckpt", model_seeds=[0], git_commit="x")
    with pytest.raises(IncompatibleExperimentError):
        mf.assert_train_eval_consistent(_record(m), train, ev)   # ideal/full never an OOD axis


def test_build_manifest_from_real_experiment_spec():
    """End-to-end: a manifest built from a real ExperimentSpec (real config hashes) validates."""
    from src.config.experiment_spec import build_experiment_spec
    from src.config.service_profile import ConsensusServiceProfile
    from src.environment import ProtocolConfig, RoundPhysicsConfig

    prof = ConsensusServiceProfile.urban_default()
    proto = ProtocolConfig(k=prof.k, alpha=prof.alpha, beta=prof.beta, r_max=prof.max_poll_epochs)
    phy = RoundPhysicsConfig(poll_window_s=prof.poll_window_s)
    spec = build_experiment_spec(protocol_cfg=proto, service_profile=prof, phy_cfg=phy,
                                 evidence_descriptor="iid:p=0.1", scene_descriptor={"grid": [3, 3, 3]},
                                 query_law="esp", full_physics=True)
    m = mf.build_manifest(spec, policy_hash="uniform-esp", checkpoint_hash="uniform-esp",
                          model_seeds=[0, 1, 2, 3, 4], git_commit="deadbee")
    rec = _record(m)
    mf.validate_manifest(rec, min_seeds=5)
    assert len(m["physics_hash"]) == 64        # real sha-256


def test_record_hashes_must_match_eval_spec():
    """A result cannot claim a spec it did not run under: recorded hashes must match the eval spec."""
    ev = _spec(physics_hash="real")
    m = mf.build_manifest(ev, policy_hash="pol", checkpoint_hash="ckpt", model_seeds=[0], git_commit="x")
    m["physics_hash"] = "tampered"
    with pytest.raises(schema.MetricSchemaError):
        mf.assert_train_eval_consistent(_record(m), ev, ev)
