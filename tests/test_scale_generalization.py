"""G-SCALE-GENERALIZATION (S16): near-linear scaling (no N x N) + OOD-axis enforcement.

The deterministic structural facts (bounded degree, total padded cells <= 2E, fixed
protocol/profile/physics hashes across N, the OOD allow/block matrix) are asserted directly; the
precise runtime log-log slope is in docs/gate_evidence/macrostate/scale_generalization_results.json
(here only a loose sub-quadratic check, robust to machine noise).
"""

import time

import torch
import torch.nn.functional as F

from src.config.experiment_spec import (
    IncompatibleExperimentError,
    build_experiment_spec,
    check_train_eval_compatible,
)
from src.config.service_profile import ConsensusServiceProfile
from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    run_consensus_episode,
)
from src.environment.candidate_graph import build_candidate_graph
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.mainline.global_evaluator import build_bucketed_padding
from src.sampling import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
COMM, BLOCK = 95.0, 120.0


def _scene(gx, gy, v=3):
    return build_manhattan_scene(gx, gy, v, block_m=BLOCK, comm_radius=COMM, int_radius=150.0,
                                 generator=torch.Generator().manual_seed(0))


def _spec(gx, gy, *, query_law="esp", proto=PROTO, full_physics=False, allowed=()):
    return build_experiment_spec(protocol_cfg=proto, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor="iid:p=0.2",
                                 scene_descriptor={"builder": "manhattan", "gx": gx, "gy": gy, "v": 3},
                                 query_law=query_law, full_physics=full_physics, allowed_ood_axes=allowed)


def test_bounded_degree_and_cells_le_2E_across_scale():
    """Fixed density => maxdeg BOUNDED (constant) and total padded cells <= 2E at every N -- the
    structural no-N x N guarantee; E grows ~linearly with N."""
    maxdegs, ratios = [], []
    prevN = prevE = None
    for (gx, gy) in [(5, 5), (8, 8), (14, 14), (22, 22)]:
        sc = _scene(gx, gy)
        N = sc.num_nodes
        gc = build_candidate_graph(sc.positions, sc.comm_radius)
        E = gc.num_edges
        pad = build_bucketed_padding(gc.src_index, gc.dst_index, N)
        assert pad.total_cells <= 2 * E
        maxdegs.append(int(torch.bincount(gc.src_index, minlength=N).max()))
        if prevN is not None:
            ratios.append((E / prevE) / (N / prevN))         # E/N ratio ~ constant => E ~ linear in N
        prevN, prevE = N, E
    assert len(set(maxdegs)) == 1                             # degree bounded (constant) across scale
    assert all(0.8 < rr < 1.25 for rr in ratios)             # E grows ~linearly with N


def test_runtime_subquadratic():
    """Episode runtime grows SUB-QUADRATICALLY in N (an N x N path would be ~N^2). Loose margin."""
    def episode_time(gx, gy):
        sc = _scene(gx, gy)
        ev = build_overlapping_scenario(sc, "iid", base_node_err=0.2)
        base = DistanceQueryPolicy(beta_per_m=0.04)
        t0 = time.perf_counter()
        run_consensus_episode(sc, ev, base, PROTO, PHY, return_trajectory=False, link_override=0.9)
        return sc.num_nodes, time.perf_counter() - t0
    n1, t1 = episode_time(6, 6)
    n2, t2 = episode_time(18, 18)
    # t2/t1 must be far below the quadratic bound (n2/n1)^2
    assert t2 / max(t1, 1e-6) < (n2 / n1) ** 1.6


def test_fixed_protocol_profile_physics_hash_across_scale():
    """Scaling N changes ONLY scene_distribution_hash; protocol/service-profile/physics are fixed."""
    specs = [_spec(gx, gy) for (gx, gy) in [(5, 5), (8, 8), (14, 14)]]
    assert len({s.protocol_hash for s in specs}) == 1
    assert len({s.service_profile_hash for s in specs}) == 1
    assert len({s.physics_hash for s in specs}) == 1
    assert len({s.scene_distribution_hash for s in specs}) == len(specs)     # node_count varies


def test_ood_enforcement_matrix():
    train = _spec(5, 5)
    # node_count registered -> allowed
    check_train_eval_compatible(train, _spec(22, 22, allowed=("node_count",)))
    # node_count NOT registered -> blocked
    import pytest
    with pytest.raises(IncompatibleExperimentError):
        check_train_eval_compatible(train, _spec(22, 22))
    # protocol mismatch (non-OOD axis) -> blocked even with node_count registered
    with pytest.raises(IncompatibleExperimentError):
        check_train_eval_compatible(train, _spec(22, 22, proto=ProtocolConfig(k=3, alpha=2, beta=4, r_max=6),
                                                 allowed=("node_count",)))
    # ideal/full-link mismatch -> ALWAYS blocked (constraint #9), even with axes registered
    with pytest.raises(IncompatibleExperimentError):
        check_train_eval_compatible(train, _spec(5, 5, full_physics=True, allowed=("node_count", "physics")))


def test_basin_headline_at_scale_via_mc():
    """The headline (macrostate basin first-hitting) works at a larger scale via the INDEPENDENT
    dynamic MC: four basins sum to 1, P_correct in (0,1) -- the participation-weighted basin
    judge generalises across N (not a node-union metric)."""
    from src.metrics.participation import uniform_participation
    from src.validation import run_dynamic_mc
    sc = _scene(8, 8)                                          # N=336
    ev = build_overlapping_scenario(sc, "matched_marginal_high", base_node_err=0.35, corr_strength=0.3)
    mc = run_dynamic_mc(sc, ev, DistanceQueryPolicy(beta_per_m=0.04), PROTO, PHY, num_trials=300,
                        generator=torch.Generator().manual_seed(0), link_override=0.9,
                        service_profile=PROFILE, participation=uniform_participation(sc.num_nodes))
    total = (mc.basin_P_correct + mc.basin_F_wrong + mc.basin_F_split + mc.basin_F_deadline)
    assert abs(total - 1.0) < 1e-9
    assert 0.0 < mc.basin_P_correct < 1.0                      # non-trivial basins (correlated regime)


def test_cdq2_eta_zero_equals_esp_at_scale():
    """The CDQ 2.0 == ESP (eta=0) identity holds at a larger scale (not just dev size)."""
    sc = _scene(10, 10)
    ev = build_overlapping_scenario(sc, "iid", base_node_err=0.2)
    base = DistanceQueryPolicy(beta_per_m=0.04)
    n = int(ev.sensor_of.max()) + 1
    div = lambda g: F.one_hot(ev.sensor_of[g.dst_index], n).to(torch.float64)
    esp = run_consensus_episode(sc, ev, base, PROTO, PHY, return_trajectory=False, link_override=0.9)
    cdq2 = run_consensus_episode(sc, ev, CDQ2Policy(base, r=n, eta=0.0, diversity=div), PROTO, PHY,
                                 return_trajectory=False, link_override=0.9)
    assert abs(float(esp.F_wrong) - float(cdq2.F_wrong)) < 1e-9
    assert torch.allclose(esp.c_ir, cdq2.c_ir, atol=1e-9)
