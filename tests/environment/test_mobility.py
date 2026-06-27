"""G12 -- temporal robustness via vehicle mobility / topology churn (spec §9.7, §G12).

Acceptance: drift moves vehicles along their segments (region-preserving), genuinely churns the
candidate graph, and keeps vehicles on their roads; and the MEMORYLESS trained ESD-GNN(ESP)
re-adapts to a drifted scene -- it stays reliable and still beats the uniform baseline under drift
(temporal generalization without a hidden state). The contractive temporal-memory model is a
deferred extension (only added if it earns a benefit).
"""

import torch

from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
    drift_scene,
    run_consensus_episode,
)
from src.environment.candidate_graph import build_candidate_graph
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.optimization import ReliabilityThresholds, train_esd_gnn
from src.sampling import UniformQueryPolicy
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=12)


def _scene(seed=0):
    return build_manhattan_scene(4, 4, 4, block_m=110.0, comm_radius=95.0, int_radius=140.0,
                                 generator=torch.Generator().manual_seed(seed))


def test_drift_preserves_regions_keeps_vehicles_on_segments_and_churns_graph():
    scene = _scene()
    drifted = drift_scene(scene, step_frac=0.25, generator=torch.Generator().manual_seed(1))
    # regions, endpoints, radii, N unchanged
    assert torch.equal(drifted.region_of, scene.region_of)
    assert drifted.num_nodes == scene.num_nodes
    assert drifted.positions.shape == scene.positions.shape
    # positions actually moved
    assert not torch.allclose(drifted.positions, scene.positions)
    # vehicles stay near their segment line (within lane jitter) -> region still valid
    g = scene.region_of
    start = scene.segment_endpoints[g, 0, :]
    end = scene.segment_endpoints[g, 1, :]
    seg = end - start
    seg_len = seg.norm(dim=1).clamp_min(1e-9)
    dirn = seg / seg_len.unsqueeze(1)
    rel = drifted.positions - start
    perp_dist = (rel - (rel * dirn).sum(dim=1, keepdim=True) * dirn).norm(dim=1)
    assert float(perp_dist.max()) < 4.0                  # within lane jitter (3 m) + epsilon
    # candidate graph genuinely churned (edge set changed)
    g0 = build_candidate_graph(scene.positions, scene.comm_radius)
    g1 = build_candidate_graph(drifted.positions, drifted.comm_radius)
    s0 = set(zip(g0.src_index.tolist(), g0.dst_index.tolist()))
    s1 = set(zip(g1.src_index.tolist(), g1.dst_index.tolist()))
    assert s0 != s1                                       # topology churn occurred


def test_trained_policy_readapts_and_stays_reliable_under_drift():
    train = [(s := _scene(i), build_scenario("one_biased_region", s, base_node_err=0.05,
                                             region_bias=0.92)) for i in range(2)]
    torch.manual_seed(0)
    model = ESDGNN(ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3, use_cdq=False)).double()
    train_esd_gnn(model, train, PROTO, PHY, ReliabilityThresholds(), steps=35, lr=0.01,
                  eta_mu=8.0, link_override=1.0)
    # a held-out scene + its drifted version (topology the policy never trained on)
    base = _scene(99)
    ev = build_scenario("one_biased_region", base, base_node_err=0.05, region_bias=0.92)
    drifted = drift_scene(base, step_frac=0.3, churn_frac=0.1,
                          generator=torch.Generator().manual_seed(7))

    def f_wrong(scene, policy):
        return float(run_consensus_episode(scene, ev, policy, PROTO, PHY,
                                           return_trajectory=False, link_override=1.0).F_wrong.detach())

    # the SAME memoryless model, features recomputed on each topology (train==deploy policy, #3)
    gnn_base = f_wrong(base, ESDGNNQueryPolicy(model, base))
    gnn_drift = f_wrong(drifted, ESDGNNQueryPolicy(model, drifted))
    unif_drift = f_wrong(drifted, UniformQueryPolicy())
    # reliability holds under drift (no catastrophic degradation) and still beats uniform
    assert gnn_drift < unif_drift                          # re-adapts: beats the baseline on the drifted topology
    assert gnn_drift < 0.1                                 # stays reliable under churn
    assert gnn_drift <= gnn_base + 0.05                    # graceful: close to the undrifted reliability


def test_drift_is_reproducible():
    scene = _scene()
    a = drift_scene(scene, step_frac=0.2, generator=torch.Generator().manual_seed(3))
    b = drift_scene(scene, step_frac=0.2, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a.positions, b.positions)
