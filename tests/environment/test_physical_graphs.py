"""G2/G3-prep -- urban geometry + the two physical graphs G_comm, G_int (spec §7.1).

Acceptance:
  * radius graphs match brute force, have no self-loops / no degree cap;
  * G_comm ⊆ G_int and G_int adds non-intended interferers (the spec §7.1 correction);
  * interference aggregated over G_int includes transmitters that do NOT poll the receiver
    (the mechanism the legacy destination-keyed aggregation drops);
  * Manhattan regions contain their vehicles; edge count is ~O(N) at fixed density.
"""

import itertools

import torch

from src.environment import (
    aggregate_over_graph,
    build_candidate_graph,
    build_interference_graph,
    build_manhattan_scene,
    build_radius_graph,
    edge_set,
    non_intended_interferers,
    received_interference_mw,
)


def _bruteforce_edges(pos, r):
    N = pos.shape[0]
    s = set()
    for i, j in itertools.permutations(range(N), 2):
        d = float(((pos[i] - pos[j]) ** 2).sum() ** 0.5)
        if d <= r:
            s.add((i, j))
    return s


def test_radius_graph_matches_bruteforce_no_selfloops():
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(40, 2, generator=g, dtype=torch.float64) * 300
    graph = build_radius_graph(pos, 70.0)
    es = edge_set(graph)
    assert es == _bruteforce_edges(pos, 70.0)
    assert all(i != j for i, j in es)                       # no self-loops
    # symmetric relation: i->j iff j->i
    assert all((j, i) in es for (i, j) in es)
    # distances are correct and within radius
    assert float(graph.distance.max()) <= 70.0 + 1e-9


def test_no_degree_cap_dense_cluster():
    # one tight cluster of 30 within radius -> every node degree 29 (no cap)
    cluster = torch.zeros(30, 2, dtype=torch.float64)
    cluster[:, 0] = torch.linspace(0, 10, 30)  # all within radius 50
    graph = build_radius_graph(cluster, 50.0)
    deg = graph.out_degree()
    assert int(deg.max()) == 29 and int(deg.min()) == 29   # full neighbourhood, uncapped


def test_comm_subset_of_int_and_non_intended_interferers():
    # j at origin; partner p within comm; external interferer t beyond comm but within int
    pos = torch.tensor([[0.0, 0.0],     # 0 = j (receiver)
                        [50.0, 0.0],    # 1 = p (within comm 80)
                        [120.0, 0.0]],  # 2 = t (beyond comm 80, within int 160)
                       dtype=torch.float64)
    comm = build_candidate_graph(pos, 80.0)
    inter = build_interference_graph(pos, 160.0)
    es_comm, es_int = edge_set(comm), edge_set(inter)
    assert es_comm <= es_int                                # G_comm ⊆ G_int
    # t<->j present in G_int, absent in G_comm
    assert (2, 0) in es_int and (2, 0) not in es_comm
    assert (0, 2) in es_int and (0, 2) not in es_comm
    ni = non_intended_interferers(comm, inter)
    assert (2, 0) in ni and (2, 0) not in es_comm


def test_interference_includes_non_intended_transmitters():
    """Interference at j over G_int must exceed interference over only its comm-edges:
    the external transmitter t (not polling j) still raises j's floor (spec §7.1)."""
    pos = torch.tensor([[0.0, 0.0], [50.0, 0.0], [120.0, 0.0]], dtype=torch.float64)
    comm = build_candidate_graph(pos, 80.0)
    inter = build_interference_graph(pos, 160.0)
    tx_activity = torch.ones(3, dtype=torch.float64)
    # synthetic per-edge received power = 1/d^2 (monotone proxy; real power is in round_physics)
    rx_int = 1.0 / (inter.distance ** 2)
    I_int = received_interference_mw(inter, rx_int, tx_activity, subchannels=4.0)
    # interference over only the comm edges into each node (the legacy, lossy aggregation)
    rx_comm = 1.0 / (comm.distance ** 2)
    I_comm_only = (1.0 / 4.0) * aggregate_over_graph(comm, tx_activity[comm.src_index] * rx_comm)
    # at receiver j (node 0) the full G_int floor is strictly higher (t contributes)
    assert float(I_int[0]) > float(I_comm_only[0]) + 1e-12


def test_manhattan_regions_contain_vehicles():
    g = torch.Generator().manual_seed(1)
    scene = build_manhattan_scene(3, 3, 6, lane_jitter_m=3.0, generator=g)
    assert scene.num_nodes == scene.num_regions * 6
    assert int(scene.region_of.min()) == 0
    assert int(scene.region_of.max()) == scene.num_regions - 1
    # every vehicle lies within lane_jitter of its segment line and within its extent
    for s in range(scene.num_regions):
        a = scene.segment_endpoints[s, 0]
        b = scene.segment_endpoints[s, 1]
        seg = b - a
        L2 = float((seg * seg).sum())
        idx = (scene.region_of == s).nonzero().reshape(-1)
        for i in idx.tolist():
            rel = scene.positions[i] - a
            t = float((rel * seg).sum()) / L2          # projection fraction
            perp = rel - t * seg
            perp_d = float((perp * perp).sum() ** 0.5)
            assert -1e-9 <= t <= 1 + 1e-9              # within segment extent
            assert perp_d <= 3.0 + 1e-6                # within lane jitter


def test_edge_count_near_linear_fixed_density():
    """At fixed spatial density the edge count grows ~O(N) (no N^2). Build a scene that
    doubles in node count at constant per-segment density and check E/N stays bounded."""
    ratios = []
    for gx in (3, 5, 7):                                # grids grow -> N grows ~quadratically in gx
        scene = build_manhattan_scene(gx, gx, 8, generator=torch.Generator().manual_seed(gx))
        graph = build_candidate_graph(scene.positions, scene.comm_radius)
        ratios.append(graph.num_edges / scene.num_nodes)
    # average degree (E/N) stays bounded and roughly constant across scales
    assert max(ratios) < 60.0
    assert max(ratios) / min(ratios) < 2.0


def test_build_scales_to_several_thousand_nodes():
    # cell-list build must handle a few thousand nodes quickly (no N x N allocation)
    scene = build_manhattan_scene(10, 10, 20, generator=torch.Generator().manual_seed(7))
    assert scene.num_nodes >= 3000
    graph = build_candidate_graph(scene.positions, scene.comm_radius)
    assert graph.num_edges > 0
    assert int(graph.out_degree().max()) >= 1
