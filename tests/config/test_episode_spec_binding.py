"""G-CANONICAL-CLOSURE: profile<->protocol<->physics single-source binding + unused-config.

The ConsensusServiceProfile is the single source of (k, alpha, beta, R_d, Delta_poll); the
ProtocolConfig and RoundPhysicsConfig used by an episode MUST agree with it, else the config
is internally inconsistent (a silent train/eval hazard). Every config field must also enter
its fingerprint (no field silently dropped from reproducibility / "unused config").
"""

import dataclasses

import pytest

from src.config.experiment_spec import build_episode_experiment_spec
from src.config.service_profile import ConsensusServiceProfile
from src.environment.canonical_episode import ProtocolConfig
from src.environment.round_physics import RoundPhysicsConfig


def _consistent():
    profile = ConsensusServiceProfile.urban_default()  # k=4,alpha=3,beta=5,R_d=20,Δ=10ms
    proto = ProtocolConfig(k=profile.k, alpha=profile.alpha, beta=profile.beta,
                           r_max=profile.max_poll_epochs)
    phy = RoundPhysicsConfig(poll_window_s=profile.poll_window_s)
    return profile, proto, phy


def test_default_physics_window_matches_default_profile_delta_poll():
    profile = ConsensusServiceProfile.urban_default()
    phy = RoundPhysicsConfig()
    assert abs(phy.poll_window_s - profile.poll_window_s) < 1e-12


def test_consistent_binding_builds_spec():
    profile, proto, phy = _consistent()
    spec = build_episode_experiment_spec(
        protocol_cfg=proto, service_profile=profile, phy_cfg=phy,
        evidence_descriptor="iid:p=0.1", scene_descriptor="manhattan:4x4x4",
        query_law="esp", link_override=None)
    assert spec.full_physics is True
    assert spec.service_profile_hash == profile.config_hash()


def test_protocol_disagreeing_with_profile_is_rejected():
    profile, _, phy = _consistent()
    bad = ProtocolConfig(k=profile.k + 1, alpha=profile.alpha, beta=profile.beta,
                         r_max=profile.max_poll_epochs)  # k disagrees with the profile
    with pytest.raises(ValueError, match="k|profile|protocol"):
        build_episode_experiment_spec(
            protocol_cfg=bad, service_profile=profile, phy_cfg=phy,
            evidence_descriptor="iid", scene_descriptor="m", query_law="esp", link_override=None)


def test_physics_window_disagreeing_with_profile_is_rejected():
    profile, proto, _ = _consistent()
    bad = RoundPhysicsConfig(poll_window_s=profile.poll_window_s * 2)  # Δ_poll disagrees
    with pytest.raises(ValueError, match="poll_window|Delta_poll|profile"):
        build_episode_experiment_spec(
            protocol_cfg=proto, service_profile=profile, phy_cfg=bad,
            evidence_descriptor="iid", scene_descriptor="m", query_law="esp", link_override=None)


def test_link_override_makes_spec_non_full_physics():
    profile, proto, phy = _consistent()
    spec = build_episode_experiment_spec(
        protocol_cfg=proto, service_profile=profile, phy_cfg=phy,
        evidence_descriptor="iid", scene_descriptor="m", query_law="esp", link_override=0.9)
    assert spec.full_physics is False


def test_unknown_config_field_rejected_by_frozen_dataclasses():
    # typo'd / unused config keys cannot be silently set (no silent misconfiguration)
    with pytest.raises(TypeError):
        ProtocolConfig(kk=4)
    with pytest.raises(TypeError):
        RoundPhysicsConfig(transmit_power=23)
    with pytest.raises(TypeError):
        ConsensusServiceProfile(poll_window_millis=10)


def test_every_physics_field_enters_the_fingerprint():
    """No RoundPhysicsConfig field may be silently dropped from physics_hash (unused-in-hash) —
    INCLUDING the nested pathloss dataclass (every sub-field)."""
    base = RoundPhysicsConfig()
    h0 = base.config_hash()
    perturb = {
        "fc_ghz": 6.0, "tx_power_dbm": 25.0, "noise_dbm": -90.0, "subchannels": 6.0,
        "slots_per_window": 25.0, "request_blocklength": 70.0, "response_blocklength": 700.0,
        "request_bits": 50.0, "response_bits": 320.0, "max_harq_attempts": 3,
        "harq_combining": "ir", "fading": "none", "use_shadow_fading": False,
        "slot_time_s": 2e-3, "request_slots": 2.0, "response_slots": 2.0, "service_rate": 10.0,
        "poll_window_s": 0.03, "los_d0_m": 60.0,
    }
    # every FLAT field is covered by this perturbation map (pathloss handled separately below)
    flat_fields = {f.name for f in dataclasses.fields(base)} - {"pathloss"}
    assert flat_fields == set(perturb), f"flat fields not covered: {flat_fields ^ set(perturb)}"
    for name, val in perturb.items():
        h = dataclasses.replace(base, **{name: val}).config_hash()
        assert h != h0, f"changing {name} did not change physics_hash (field dropped from fingerprint)"

    # the NESTED pathloss must ALSO enter the fingerprint — perturb every sub-field
    pl = base.pathloss
    for f in dataclasses.fields(pl):
        cur = getattr(pl, f.name)
        new = tuple(c + 1.0 for c in cur) if isinstance(cur, tuple) else cur + 1.0
        perturbed = dataclasses.replace(base, pathloss=dataclasses.replace(pl, **{f.name: new}))
        assert perturbed.config_hash() != h0, \
            f"changing pathloss.{f.name} did not change physics_hash (nested field dropped)"
