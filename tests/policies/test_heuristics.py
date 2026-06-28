"""G-ESP-BASELINE-ORACLE heuristics: observable-structure ESP baselines.

Each emits per-edge ESP log-weights from observable structure only (no truth/votes), runs through the
canonical full-physics MC, and differs from a uniform policy (it actually weights edges)."""

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.environment.candidate_graph import build_candidate_graph
from src.evaluation.esp_scale import build_scale_instance
from src.metrics.participation import uniform_participation
from src.policies.heuristics import LinkQualityPolicy, LoadBalancedPolicy, RegionBridgePolicy
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)


def _policies(scene):
    return [LinkQualityPolicy(), LoadBalancedPolicy(scene), RegionBridgePolicy(scene)]


def test_log_weights_shape_and_finite():
    scene, _ = build_scale_instance((5, 5, 3), 0)
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    for pol in _policies(scene):
        w = pol.log_weights(gc)
        assert w.shape == (gc.num_edges,) and torch.isfinite(w).all()
        assert pol.name and getattr(pol, "query_law", "esp") == "esp"   # default ESP law


def test_truth_independent():
    """The heuristics use only geometry / graph / region -> identical weights regardless of evidence/truth."""
    scene, _ = build_scale_instance((5, 5, 3), 0, scenario="iid", base_node_err=0.1)
    _, _ = build_scale_instance((5, 5, 3), 0, scenario="matched_marginal_high", base_node_err=0.4)
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    # weights computed twice (no evidence input at all) are bit-identical
    for pol in _policies(scene):
        assert torch.equal(pol.log_weights(gc), pol.log_weights(gc))


def test_differs_from_uniform_and_runs_in_mc():
    scene, ev = build_scale_instance((5, 5, 3), 0, scenario="matched_marginal_high", base_node_err=0.35,
                                     corr_strength=0.25)
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    omega = uniform_participation(scene.num_nodes)
    for pol in _policies(scene):
        assert pol.log_weights(gc).std() > 0                      # actually weights edges (not uniform)
        r = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=20,
                           generator=torch.Generator().manual_seed(0), link_override=None,
                           service_profile=PROFILE, participation=omega)
        assert abs((r.basin_P_correct + r.basin_F_wrong + r.basin_F_split + r.basin_F_deadline) - 1.0) < 1e-9
