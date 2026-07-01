"""G-NDH-SCENE-RSU-HOTSPOT -- nonuniform urban scene + sparse intersection hotspots + capped RSU.

Acceptance (engineering plan §3 + NDH_PARAMETER_REGISTRY §3/§6):
  * drop-in interface: the 7 frozen scene fields + num_nodes/num_regions properties, correct
    shapes/dtypes; a control config recovers a uniform-grid, fully-contained scene;
  * HARD CAPS enforced: p_intersection_rsu <= 0.5, #RSU <= max_rsu_fraction*N, #hotspots <= 10%
    of intersections, hotspot vehicles <= 30% of vehicles, hotspots non-overlapping (>= 2*radius);
  * RSU is responder/witness only: vehicle_only_participation gives omega_RSU = 0 and sums to 1;
  * road graph connected; scene reproducible from the generator seed;
  * the NDH scene flows through the CANONICAL path (build_overlapping_scenario -> run_dynamic_mc),
    i.e. the mechanism is on the mainline, not just in tests (constraint #5).
"""

import itertools

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig, build_candidate_graph
from src.environment.nonuniform_urban_scene import (
    NonuniformUrbanScene,
    build_nonuniform_urban_scene,
)
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.metrics.participation import vehicle_only_participation
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.validation import run_dynamic_mc

GEN = lambda s: torch.Generator().manual_seed(s)  # noqa: E731


# --------------------------------------------------------------- interface / drop-in
def test_interface_and_shapes_control_grid():
    sc = build_nonuniform_urban_scene(5, 5, 3, generator=GEN(0))  # all NDH knobs default off-structure
    assert isinstance(sc, NonuniformUrbanScene)
    N, G = sc.num_nodes, sc.num_regions
    assert sc.positions.shape == (N, 2) and sc.positions.dtype == torch.float64
    assert sc.region_of.shape == (N,) and sc.region_of.dtype == torch.long
    assert sc.segment_endpoints.shape == (G, 2, 2)
    assert isinstance(sc.comm_radius, float) and isinstance(sc.int_radius, float)
    assert sc.int_radius >= sc.comm_radius > 0
    # NDH extensions present
    assert sc.node_type.shape == (N,) and sc.node_type.dtype == torch.long
    assert sc.hotspot_score.shape == (N,)
    # dense region coverage (evidence model requires it)
    assert int(sc.region_of.min()) == 0 and int(sc.region_of.max()) == G - 1
    # control config = pure vehicles, no RSU, no hotspots
    assert int((sc.node_type == 1).sum()) == 0
    assert float(sc.hotspot_score.max()) == 0.0


def test_control_grid_vehicles_contained():
    """With logstd=0, jitter=0 the control scene must contain vehicles within lane_jitter of
    their segment line (the evidence-correlation locality invariant)."""
    sc = build_nonuniform_urban_scene(4, 4, 5, lane_jitter_m=3.0, generator=GEN(2))
    for s in range(sc.num_regions):
        a, b = sc.segment_endpoints[s, 0], sc.segment_endpoints[s, 1]
        seg = b - a
        L2 = float((seg * seg).sum())
        idx = (sc.region_of == s).nonzero().reshape(-1)
        for i in idx.tolist():
            if int(sc.node_type[i]) != 0:
                continue  # RSU may sit at roadside offset; checked separately
            rel = sc.positions[i] - a
            t = float((rel * seg).sum()) / L2
            perp = rel - t * seg
            assert -1e-9 <= t <= 1 + 1e-9
            assert float((perp * perp).sum() ** 0.5) <= 3.0 + 1e-6


def test_reproducible_from_seed():
    a = build_nonuniform_urban_scene(5, 5, 3, block_length_logstd=0.2, intersection_jitter_m=20.0,
                                     enable_hotspots=True, enable_rsu=True, generator=GEN(7))
    b = build_nonuniform_urban_scene(5, 5, 3, block_length_logstd=0.2, intersection_jitter_m=20.0,
                                     enable_hotspots=True, enable_rsu=True, generator=GEN(7))
    assert torch.equal(a.positions, b.positions)
    assert torch.equal(a.region_of, b.region_of)
    assert torch.equal(a.node_type, b.node_type)


# --------------------------------------------------------------- hard caps
def test_rsu_density_capped():
    sc = build_nonuniform_urban_scene(6, 6, 3, enable_rsu=True, p_intersection_rsu=0.5,
                                      p_hotspot_rsu_boost=0.5, max_rsu_fraction=0.10,
                                      enable_hotspots=True, generator=GEN(1))
    n_rsu = int((sc.node_type == 1).sum())
    assert n_rsu <= 0.10 * sc.num_nodes + 1e-9
    assert n_rsu >= 1  # at p=0.5 on a 36-intersection grid we expect some RSU


def test_p_intersection_rsu_hard_cap_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_nonuniform_urban_scene(5, 5, 3, enable_rsu=True, p_intersection_rsu=0.6,
                                     generator=GEN(0))


def test_hotspot_count_and_vehicle_fraction_capped():
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_hotspots=True, num_hotspots=3,
                                      hotspot_vehicle_fraction=0.3, hotspot_radius_m=50.0,
                                      generator=GEN(3))
    n_int = sc.intersection_xy.shape[0]
    n_hot = sc.hotspot_intersections.numel()
    assert n_hot <= max(1, int(0.10 * n_int))          # <= 10% of intersections
    # the cap bounds the INJECTED queue mass (the mechanism), not incidental grid density
    assert sc.params["num_hotspot_vehicles"] <= 0.30 * sc.num_vehicles + 1e-9


def test_coarse_grid_large_radius_no_crash():
    """Registry-legal coarse grid + large radius must NOT crash on the fraction cap (the cap
    bounds injected queue mass, which is always satisfiable)."""
    for seed in range(6):
        sc = build_nonuniform_urban_scene(2, 2, 3, enable_hotspots=True, num_hotspots=1,
                                          hotspot_radius_m=80.0, hotspot_vehicle_fraction=0.3,
                                          generator=GEN(seed))
        assert sc.params["num_hotspot_vehicles"] <= 0.30 * sc.num_vehicles + 1e-9


def test_queue_length_drives_extent():
    """queue_length_m (not hotspot_radius_m) sets the along-road queue extent."""
    def max_queue_reach(qlen):
        sc = build_nonuniform_urban_scene(6, 6, 2, block_m=200.0, enable_hotspots=True,
                                          num_hotspots=1, hotspot_radius_m=50.0, queue_length_m=qlen,
                                          hotspot_vehicle_fraction=0.3, generator=GEN(1))
        n_q = sc.params["num_hotspot_vehicles"]
        assert n_q >= 1
        h = sc.intersection_xy[sc.hotspot_intersections[0]]
        queued = sc.positions[sc.num_vehicles - n_q:sc.num_vehicles]   # the appended queued vehicles
        return float((queued - h).norm(dim=1).max())     # farthest queued vehicle from the hotspot
    far_long = max_queue_reach(150.0)
    far_short = max_queue_reach(40.0)
    assert far_long > far_short + 5.0                    # longer queue reaches measurably farther


def test_hotspots_non_overlapping():
    sc = build_nonuniform_urban_scene(6, 6, 3, enable_hotspots=True, num_hotspots=3,
                                      hotspot_radius_m=50.0, generator=GEN(5))
    hs = sc.intersection_xy[sc.hotspot_intersections]
    for i, j in itertools.combinations(range(hs.shape[0]), 2):
        d = float((hs[i] - hs[j]).norm())
        assert d >= 2 * 50.0 - 1e-6


# --------------------------------------------------------------- omega_RSU = 0
def test_vehicle_only_participation_excludes_rsu():
    sc = build_nonuniform_urban_scene(6, 6, 3, enable_rsu=True, p_intersection_rsu=0.4,
                                      enable_hotspots=True, generator=GEN(4))
    omega = vehicle_only_participation(sc)
    assert abs(float(omega.sum()) - 1.0) < 1e-9
    is_rsu = (sc.node_type == 1)
    assert float(omega[is_rsu].abs().sum()) == 0.0       # omega_RSU = 0 exactly
    veh = omega[sc.node_type == 0]
    assert torch.allclose(veh, veh[0].expand_as(veh))     # uniform over vehicles
    assert int((sc.node_type == 1).sum()) >= 1            # the test actually has RSU


# --------------------------------------------------------------- road graph connectivity
def test_road_graph_connected_under_road_dropout():
    sc = build_nonuniform_urban_scene(5, 5, 3, road_presence_probability=0.7, generator=GEN(9))
    n_int = sc.intersection_xy.shape[0]
    parent = list(range(n_int))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in sc.segment_intersections.tolist():
        ra, rb = find(e[0]), find(e[1])
        parent[ra] = rb
    roots = {find(i) for i in range(n_int)}
    assert len(roots) == 1                                # all intersections connected


# --------------------------------------------------------------- canonical path (constraint #5)
def test_flows_through_canonical_dynamic_mc():
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_hotspots=True, enable_rsu=True,
                                      p_intersection_rsu=0.3, generator=GEN(11))
    ev = build_overlapping_scenario(sc, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    omega = vehicle_only_participation(sc)
    r = run_dynamic_mc(sc, ev, DistanceQueryPolicy(beta_per_m=0.04), proto, phy, num_trials=20,
                       generator=GEN(11), service_profile=prof, participation=omega)
    # basin masses are well-formed probabilities that sum to ~1 over the four macrostates
    tot = r.basin_P_correct + r.basin_F_wrong + r.basin_F_split + r.basin_F_deadline
    assert 0.0 <= r.basin_P_correct <= 1.0
    assert abs(tot - 1.0) < 1e-6


def test_rsu_presence_does_not_change_vehicle_evidence():
    """C4 causal-control guard: enabling RSU must NOT move the vehicle-vehicle evidence structure.
    Same seed + same hotspots => identical vehicle positions; vehicle sensor/map bands must be
    invariant to RSU presence (RSU geometry no longer leaks into the matched-marginal lever)."""
    kw = dict(enable_hotspots=True, num_hotspots=2, hotspot_vehicle_fraction=0.2)
    sc0 = build_nonuniform_urban_scene(6, 6, 3, enable_rsu=False, generator=GEN(22), **kw)
    sc1 = build_nonuniform_urban_scene(6, 6, 3, enable_rsu=True, p_intersection_rsu=0.5,
                                       p_hotspot_rsu_boost=0.5, generator=GEN(22), **kw)
    nv = sc0.num_vehicles
    assert sc1.num_vehicles == nv and int(sc1.num_rsu) >= 1
    assert torch.equal(sc0.positions[:nv], sc1.positions[:nv])     # vehicle positions identical
    ev0 = build_overlapping_scenario(sc0, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    ev1 = build_overlapping_scenario(sc1, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    # vehicle sensor/map band assignments must be byte-identical (RSU did not move band edges)
    assert torch.equal(ev0.sensor_of[:nv], ev1.sensor_of[:nv])
    assert torch.equal(ev0.map_of[:nv], ev1.map_of[:nv])


def test_rsu_nodes_are_ordinary_graph_nodes():
    """RSU-ness lives ONLY in node_type/participation -- the graph builder treats RSU as plain
    nodes (no special handling), so comm edges to/from RSU exist normally."""
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_rsu=True, p_intersection_rsu=0.5,
                                      generator=GEN(6))
    g = build_candidate_graph(sc.positions, sc.comm_radius)
    assert g.num_nodes == sc.num_nodes
    assert g.num_edges > 0
