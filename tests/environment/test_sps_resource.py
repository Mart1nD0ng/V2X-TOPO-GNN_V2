"""G-NDH-SPS-PERSISTENCE -- sensing-based SPS resource persistence + same-resource collision physics.

Acceptance (engineering plan §4 + spec §3):
  * same-resource G_int neighbours RAISE collision; different-resource neighbours do NOT;
  * a single active same-bucket transmitter has collision EXACTLY 0 (self-exclusion preserved);
  * SPS off (kappa=0 or no bucket) reproduces the memoryless physics byte-for-byte;
  * config_hash changes when resource_collision_kappa changes (train==eval binding, constraint #4);
  * the sensing surrogate is deterministic from the seed; tau_res>0 reduces same-bucket conflict vs random;
  * no future/truth leak (assignment uses only geometry + sensed occupancy + noise);
  * SPS enters the CANONICAL dynamic-MC path (not just a unit test) and raises collision-driven failure.
"""

import math

import pytest
import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig, build_candidate_graph
from src.environment.nonuniform_urban_scene import build_nonuniform_urban_scene
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.environment.round_physics import round_physics, _sps_same_resource_collision
from src.environment.sps_resource import (
    assign_sps_buckets,
    same_resource_conflict_degree,
)
from src.environment.interference_graph import build_interference_graph
from src.metrics.participation import vehicle_only_participation
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.validation import run_dynamic_mc

GEN = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
# Non-degenerate base physics: S_phys = subchannels*slots_per_window = 400 (the campaign regime).
# SPS reservation pool S_sps = sps_n_buckets <= S_phys is the congestion knob (smaller -> more conflict).
PHY_SPS = RoundPhysicsConfig(subchannels=10, slots_per_window=40, resource_collision_kappa=0.5)
PHY_OFF = RoundPhysicsConfig(subchannels=10, slots_per_window=40)  # kappa=0 -> memoryless


def _edge_index(graph, i, j):
    for e, (s, d) in enumerate(zip(graph.src_index.tolist(), graph.dst_index.tolist())):
        if s == i and d == j:
            return e
    raise AssertionError(f"edge {i}->{j} not in graph")


# --------------------------------------------------------------- bucket assignment surrogate
def test_assign_buckets_range_and_reproducible():
    pos = torch.rand(40, 2, generator=GEN(0), dtype=torch.float64) * 200
    b0 = assign_sps_buckets(pos, 160.0, 100, generator=GEN(1))
    b1 = assign_sps_buckets(pos, 160.0, 100, generator=GEN(1))
    assert b0.shape == (40,) and b0.dtype == torch.long
    assert int(b0.min()) >= 0 and int(b0.max()) < 100
    assert torch.equal(b0, b1)                              # deterministic from seed


def test_sensing_reduces_same_bucket_conflict_vs_random():
    """tau_res>0 (sensing-avoidance) yields FEWER same-bucket interference neighbours than tau_res=0."""
    pos = torch.rand(80, 2, generator=GEN(4), dtype=torch.float64) * 250
    b_sense = assign_sps_buckets(pos, 160.0, 40, tau_res=6.0, sensing_noise_std=0.0, generator=GEN(2))
    b_rand = assign_sps_buckets(pos, 160.0, 40, tau_res=0.0, sensing_noise_std=0.0, generator=GEN(2))
    deg_sense = same_resource_conflict_degree(b_sense, pos, 160.0).mean()
    deg_rand = same_resource_conflict_degree(b_rand, pos, 160.0).mean()
    assert float(deg_sense) <= float(deg_rand) + 1e-9      # avoidance never increases conflict


# --------------------------------------------------------------- same-resource collision physics
def _three_node_setup(bucket_vals):
    """j=0 receiver; 1,2 near 0. Edge 1->0; node 2 is a G_int contender near 0. Returns
    (p_col_request on edge 1->0). bucket_vals sets nodes' persistent buckets."""
    pos = torch.tensor([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]], dtype=torch.float64)
    gc = build_candidate_graph(pos, 100.0)
    gi = build_interference_graph(pos, 160.0)
    e10 = _edge_index(gc, 1, 0)
    pi = torch.ones(gc.num_edges, dtype=torch.float64)
    active = torch.ones((3, 1), dtype=torch.float64)       # all active/transmitting
    bucket = torch.tensor(bucket_vals, dtype=torch.long)
    r = round_physics(gc, gi, pi, active, PHY_SPS, resource_bucket=bucket)
    return float(r.p_collision_request[e10, 0])


def test_same_resource_raises_collision_different_does_not():
    same = _three_node_setup([0, 5, 5])                    # nodes 1 and 2 share bucket 5
    diff = _three_node_setup([0, 5, 7])                    # nodes 1 and 2 different buckets
    assert same > 0.05                                     # same-bucket contender -> real collision risk
    assert diff < 1e-9                                     # different bucket -> no persistent collision


def test_single_transmitter_self_excluded_zero_collision():
    """One active same-bucket transmitter (node 2 inactive) -> collision EXACTLY 0 (self-exclusion)."""
    pos = torch.tensor([[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]], dtype=torch.float64)
    gc = build_candidate_graph(pos, 100.0)
    gi = build_interference_graph(pos, 160.0)
    e10 = _edge_index(gc, 1, 0)
    pi = torch.ones(gc.num_edges, dtype=torch.float64)
    active = torch.tensor([[1.0], [1.0], [0.0]], dtype=torch.float64)   # node 2 NOT transmitting
    bucket = torch.tensor([0, 5, 5], dtype=torch.long)
    r = round_physics(gc, gi, pi, active, PHY_SPS, resource_bucket=bucket)
    assert float(r.p_collision_request[e10, 0]) < 1e-12


def _bruteforce_sps(gi, tx, bucket, kappa, resource_node, receiver_node, self_node):
    """Naive per-edge reference for _sps_same_resource_collision (locks BOTH phases' index roles)."""
    import collections
    contend = collections.defaultdict(list)
    for s, d in zip(gi.src_index.tolist(), gi.dst_index.tolist()):
        contend[d].append(s)
    out = []
    for e in range(receiver_node.shape[0]):
        r = int(bucket[resource_node[e]]); recv = int(receiver_node[e]); slf = int(self_node[e])
        L = sum(float(tx[u, 0]) for u in contend[recv] if int(bucket[u]) == r) - float(tx[slf, 0])
        out.append(1.0 - math.exp(-kappa * max(L, 0.0)))
    return torch.tensor(out, dtype=tx.dtype).unsqueeze(-1)


def test_response_phase_collision_index_roles_correct():
    """Directly lock the RESPONSE-phase roles (resource=dst, receiver=src, self=dst) against a
    brute-force reference -- a transposed response index would be caught here (not just end-to-end)."""
    pos = torch.rand(12, 2, generator=GEN(0), dtype=torch.float64) * 180
    gc = build_candidate_graph(pos, 90.0)
    gi = build_interference_graph(pos, 160.0)
    tx = torch.rand((12, 1), generator=GEN(1), dtype=torch.float64)
    bucket = torch.randint(0, 5, (12,), generator=GEN(2))
    for roles in [dict(resource_node=gc.src_index, receiver_node=gc.dst_index, self_node=gc.src_index),
                  dict(resource_node=gc.dst_index, receiver_node=gc.src_index, self_node=gc.dst_index)]:
        got = _sps_same_resource_collision(gi, tx, bucket, 0.5, N=12, **roles)
        ref = _bruteforce_sps(gi, tx, bucket, 0.5, **roles)
        assert torch.allclose(got, ref, atol=1e-12)


def test_single_responder_self_excluded_zero():
    """Response self-exclusion: a lone same-bucket responder near the requester -> p=0 exactly."""
    pos = torch.tensor([[0.0, 0.0], [30.0, 0.0]], dtype=torch.float64)   # only nodes 0,1
    gi = build_interference_graph(pos, 160.0)
    tx = torch.tensor([[1.0], [0.0]], dtype=torch.float64)               # only node 0 responds
    bucket = torch.tensor([5, 5], dtype=torch.long)
    p = _sps_same_resource_collision(gi, tx, bucket, 0.5, resource_node=torch.tensor([0]),
                                     receiver_node=torch.tensor([1]), self_node=torch.tensor([0]), N=2)
    assert float(p[0, 0]) < 1e-12


def test_mechanism_config_hash_binds_sps_params():
    """The NDH-mechanism hash captures the bucket-STRUCTURE params the physics config_hash omits."""
    a = build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=100, sps_tau_res=4.0, generator=GEN(1))
    b = build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=100, sps_tau_res=1.0, generator=GEN(1))
    c = build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=100, sps_tau_res=4.0, generator=GEN(1))
    assert a.mechanism_config_hash != b.mechanism_config_hash   # different tau_res -> different structure hash
    assert a.mechanism_config_hash == c.mechanism_config_hash   # same params -> same hash


def test_sps_off_reproduces_memoryless():
    """kappa=0 OR no bucket -> byte-identical to the memoryless physics."""
    pos = torch.rand(30, 2, generator=GEN(3), dtype=torch.float64) * 200
    gc = build_candidate_graph(pos, 90.0)
    gi = build_interference_graph(pos, 160.0)
    pi = torch.ones(gc.num_edges, dtype=torch.float64)
    active = torch.rand((30, 2), generator=GEN(3), dtype=torch.float64)
    bucket = assign_sps_buckets(pos, 160.0, 100, generator=GEN(5))
    base = round_physics(gc, gi, pi, active, PHY_OFF)                       # memoryless, no bucket
    off_kappa = round_physics(gc, gi, pi, active, PHY_OFF, resource_bucket=bucket)  # kappa=0 -> off
    assert torch.allclose(base.p_collision_request, off_kappa.p_collision_request)
    assert torch.allclose(base.ell_poll, off_kappa.ell_poll)


def test_config_hash_binds_kappa():
    assert PHY_SPS.config_hash() != PHY_OFF.config_hash()  # SPS-on physics is a distinct train==eval regime


def test_mechanism_trace_sentinel_flips_with_sps():
    """Plan §4 acceptance / Contract C5: the canonical mechanism_trace proves SPS executed on the path
    (sps_persistence / resource_conflict_graph True; mode2_collision correctly False under SPS)."""
    from src.environment.canonical_episode import run_consensus_episode
    from src.sampling.baseline_policies import UniformQueryPolicy
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=4)
    sc_on = build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=100, sps_tau_res=4.0,
                                         generator=GEN(7))
    ev_on = build_overlapping_scenario(sc_on, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    tr_on = run_consensus_episode(sc_on, ev_on, UniformQueryPolicy(), proto, PHY_SPS,
                                  return_trajectory=False).mechanism_trace
    assert tr_on["sps_persistence"] is True and tr_on["resource_conflict_graph"] is True
    assert tr_on["mode2_collision"] is False           # persistent model active -> memoryless flag OFF

    sc_off = build_nonuniform_urban_scene(5, 5, 3, enable_sps=False, generator=GEN(7))
    ev_off = build_overlapping_scenario(sc_off, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    tr_off = run_consensus_episode(sc_off, ev_off, UniformQueryPolicy(), proto, PHY_OFF,
                                   return_trajectory=False).mechanism_trace
    assert tr_off["sps_persistence"] is False and tr_off["mode2_collision"] is True


# --------------------------------------------------------------- canonical path + MC effect
def _run_mc(sc, phy, prof, seed=9, trials=200):
    ev = build_overlapping_scenario(sc, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
    return run_dynamic_mc(sc, ev, DistanceQueryPolicy(beta_per_m=0.04), proto, phy, num_trials=trials,
                          generator=GEN(seed), service_profile=prof, participation=vehicle_only_participation(sc))


def test_sps_enters_dynamic_mc_and_changes_outcome_stress():
    """SPS on the canonical dynamic-MC path is ACTIVE: a congested reservation pool (S_sps=40, poor
    sensing) with S_phys=400 drives persistent collision -> Pc collapses vs SPS off on the same scene."""
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=40, sps_tau_res=1.0,
                                      generator=GEN(7))
    assert sc.resource_bucket is not None and sc.resource_bucket.shape[0] == sc.num_nodes
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    r_on = _run_mc(sc, PHY_SPS, prof)
    r_off = _run_mc(sc, PHY_OFF, prof)
    for r in (r_on, r_off):
        assert abs(r.basin_P_correct + r.basin_F_wrong + r.basin_F_split + r.basin_F_deadline - 1.0) < 1e-6
    assert r_on.basin_P_correct < r_off.basin_P_correct - 0.05   # congested pool + poor sensing: SPS worsens


def test_sps_good_sensing_non_degenerate():
    """R4/mechanism: with a healthy reservation pool + good sensing (S_sps=100, tau_res=4), SPS is
    NON-degenerate -- it removes the memoryless collision floor rather than collapsing consensus."""
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=100, sps_tau_res=4.0,
                                      generator=GEN(7))
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    r_on = _run_mc(sc, PHY_SPS, prof)
    r_off = _run_mc(sc, PHY_OFF, prof)
    assert r_on.basin_P_correct > 0.15                          # NOT degenerate/collapsed
    assert r_on.basin_P_correct >= r_off.basin_P_correct - 1e-9  # good sensing does not worsen vs memoryless


def test_pool_consistency_guard_rejects_oversized_pool():
    """S_sps must be <= physics S_phys (reservations are a subset of the instantaneous pool); an
    oversized reservation pool raises on the canonical MC path (not silently run)."""
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=500, sps_tau_res=1.0,
                                      generator=GEN(7))                       # 500 > S_phys=400
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    with pytest.raises(ValueError, match="resource_pool"):
        _run_mc(sc, PHY_SPS, prof)


def test_round_physics_rejects_float_bucket():
    pos = torch.rand(10, 2, generator=GEN(0), dtype=torch.float64) * 100
    gc = build_candidate_graph(pos, 90.0)
    gi = build_interference_graph(pos, 160.0)
    pi = torch.ones(gc.num_edges, dtype=torch.float64)
    active = torch.ones((10, 1), dtype=torch.float64)
    with pytest.raises(ValueError, match="integer dtype"):
        round_physics(gc, gi, pi, active, PHY_SPS,
                      resource_bucket=torch.zeros(10, dtype=torch.float64))
    with pytest.raises(ValueError, match="one entry per node"):
        round_physics(gc, gi, pi, active, PHY_SPS, resource_bucket=torch.zeros(9, dtype=torch.long))
