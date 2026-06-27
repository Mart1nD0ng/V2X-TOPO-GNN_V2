"""G-CANONICAL-CLOSURE: mandatory mechanism trace + headline guards (plan §4).

The single canonical episode must record that EVERY mandatory mechanism is on the live path
(plan §4 JSON), and the headline run must be full physics (a bare ideal-link override is
quarantined to an explicit ablation, constraint #9).
"""

import pytest
import torch

from src.config.experiment_spec import (
    MANDATORY_MECHANISMS,
    assert_canonical_mechanisms,
    assert_headline_grounded,
    build_experiment_spec,
)
from src.config.service_profile import ConsensusServiceProfile
from src.environment.canonical_episode import ProtocolConfig, run_consensus_episode
from src.environment.evidence_model import EvidenceModel
from src.environment.round_physics import RoundPhysicsConfig
from src.environment.urban_scene import ManhattanScene
from src.sampling import UniformQueryPolicy


def _tiny():
    pos = torch.tensor([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0], [90.0, 0.0]], dtype=torch.float64)
    reg = torch.tensor([0, 0, 0, 0])
    scene = ManhattanScene(positions=pos, region_of=reg,
                           segment_endpoints=torch.zeros((1, 2, 2), dtype=torch.float64),
                           comm_radius=70.0, int_radius=110.0, block_m=100.0, grid=(2, 1))
    ev = EvidenceModel(region_of=reg, p_region=torch.zeros(1, dtype=torch.float64),
                       p_node=torch.full((4,), 0.1, dtype=torch.float64))
    return scene, ev


def test_headline_trace_has_all_mandatory_mechanisms():
    scene, ev = _tiny()
    res = run_consensus_episode(scene, ev, UniformQueryPolicy(),
                                ProtocolConfig(k=2, alpha=2, beta=2, r_max=4),
                                RoundPhysicsConfig(), return_trajectory=False)
    tr = res.mechanism_trace
    for flag in MANDATORY_MECHANISMS:
        assert flag in tr, f"trace missing mandatory mechanism flag {flag!r}"
    # full-physics headline: every mandatory boolean mechanism is active
    assert_canonical_mechanisms(tr)            # must not raise
    assert tr["full_physics"] is True
    assert tr["parallel_unicast"] is True
    assert tr["source_destination_accounting"] is True
    assert tr["collision_self_exclusion"] is True
    assert tr["poll_window_ms"] > 0


def test_disabled_mechanism_fails_canonical_assertion():
    scene, ev = _tiny()
    res = run_consensus_episode(scene, ev, UniformQueryPolicy(),
                                ProtocolConfig(k=2, alpha=2, beta=2, r_max=4),
                                RoundPhysicsConfig(), return_trajectory=False,
                                disable_collision=True)
    with pytest.raises(ValueError, match="collision|mechanism"):
        assert_canonical_mechanisms(res.mechanism_trace)


def test_ideal_link_override_is_not_headline_grounded():
    scene, ev = _tiny()
    res = run_consensus_episode(scene, ev, UniformQueryPolicy(),
                                ProtocolConfig(k=2, alpha=2, beta=2, r_max=4),
                                RoundPhysicsConfig(), return_trajectory=False,
                                link_override=0.9)
    assert res.mechanism_trace["full_physics"] is False
    # an ideal-link trace must FAIL the headline-grounding assertion (it is an ablation only)
    with pytest.raises(ValueError, match="full physics|ideal|link"):
        assert_canonical_mechanisms(res.mechanism_trace)


def test_assert_headline_grounded_rejects_ideal_spec():
    spec_full = build_experiment_spec(
        protocol_cfg=ProtocolConfig(), service_profile=ConsensusServiceProfile.urban_default(),
        phy_cfg=RoundPhysicsConfig(), evidence_descriptor="iid", scene_descriptor="m",
        query_law="esp", full_physics=True)
    spec_ideal = build_experiment_spec(
        protocol_cfg=ProtocolConfig(), service_profile=ConsensusServiceProfile.urban_default(),
        phy_cfg=RoundPhysicsConfig(), evidence_descriptor="iid", scene_descriptor="m",
        query_law="esp", full_physics=False)
    assert_headline_grounded(spec_full)            # OK
    with pytest.raises(ValueError, match="full physics|ideal|headline"):
        assert_headline_grounded(spec_ideal)
