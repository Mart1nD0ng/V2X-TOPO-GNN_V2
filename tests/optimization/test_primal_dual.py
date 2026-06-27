"""G9b -- primal-dual reliability-constrained training (spec §4.5).

Acceptance: the augmented-Lagrangian metrics/loss are differentiable; dual ascent moves the
right way; training the ESD-GNN reduces the reliability failure toward the topology-oracle
ceiling and generalises to held-out scenes (analytic). MC confirmation of the trained CDQ
model needs the CDQ dynamic MC (next slice) and is deferred to G11.
"""

import copy

import torch

from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
    run_consensus_episode,
)
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.optimization import (
    DualState,
    ReliabilityThresholds,
    episode_metrics,
    lagrangian,
    train_esd_gnn,
)
from src.optimization.topology_oracle import optimize_logweight_topology

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=12)


def _instances(n, per=3):
    out = []
    for s in range(n):
        sc = build_manhattan_scene(3, 3, per, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                                   generator=torch.Generator().manual_seed(s))
        ev = build_scenario("one_biased_region", sc, base_node_err=0.05, region_bias=0.95)
        out.append((sc, ev))
    return out


def _model(seed=0):
    torch.manual_seed(seed)
    return ESDGNN(ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3)).double()


def _f_wrong(model, scene, ev):
    res = run_consensus_episode(scene, ev, ESDGNNQueryPolicy(model, scene), PROTO, PHY,
                                return_trajectory=False, link_override=1.0)
    return float(res.F_wrong)


def test_metrics_and_lagrangian_differentiable():
    inst = _instances(1)
    model = _model()
    res = run_consensus_episode(inst[0][0], inst[0][1], ESDGNNQueryPolicy(model, inst[0][0]),
                                PROTO, PHY, return_trajectory=True, link_override=1.0)
    m = episode_metrics(res, None)
    assert 0.0 <= float(m["F_deadline"].detach()) <= 1.0
    assert float(m["ET"].detach()) > 0.0
    loss = lagrangian(m, DualState(), ReliabilityThresholds())
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_dual_update_directions():
    thr = ReliabilityThresholds(eps_s=1e-4, eps_v=1e-3, eps_d=1e-2)
    d = DualState(mu_s=1.0, mu_v=1.0, mu_d=1.0)
    # F above budget -> mu grows
    d.update({"F_disagree": 0.1, "F_wrong": 0.1, "F_deadline": 0.1}, thr, eta_mu=5.0)
    assert d.mu_v > 1.0 and d.mu_s > 1.0 and d.mu_d > 1.0
    # F well below budget -> mu shrinks but stays >= 0
    d2 = DualState(mu_s=0.01, mu_v=0.01, mu_d=0.01)
    d2.update({"F_disagree": 0.0, "F_wrong": 0.0, "F_deadline": 0.0}, thr, eta_mu=5.0)
    assert d2.mu_v >= 0.0 and d2.mu_v < 0.01


def test_training_reduces_failure_toward_oracle():
    inst = _instances(1)
    scene, ev = inst[0]
    model = _model()
    f_before = _f_wrong(model, scene, ev)
    out = train_esd_gnn(model, inst, PROTO, PHY, ReliabilityThresholds(),
                        steps=40, lr=0.01, eta_mu=8.0, link_override=1.0)
    f_after = _f_wrong(model, scene, ev)
    # oracle ceiling on the same scene
    orc = optimize_logweight_topology(scene, ev, PROTO, PHY, steps=60, lr=0.3, link_override=1.0)
    f_oracle = float(run_consensus_episode(scene, ev, orc.policy, PROTO, PHY,
                                           return_trajectory=False, link_override=1.0).F_wrong)
    assert f_after < f_before * 0.5                  # training substantially reduces failure
    assert f_after < f_before                         # monotone improvement overall
    assert f_after <= f_oracle + 0.02                 # approaches the oracle ceiling
    assert out["history"]["mu_v"][-1] >= out["history"]["mu_v"][0]   # dual active


def test_training_generalises_to_heldout_scene():
    train = _instances(2)
    heldout = _instances(4)[3]                        # a different scene seed
    model = _model()
    f_before = _f_wrong(model, *heldout)
    train_esd_gnn(model, train, PROTO, PHY, ReliabilityThresholds(),
                  steps=40, lr=0.01, eta_mu=8.0, link_override=1.0)
    f_after = _f_wrong(model, *heldout)
    assert f_after < f_before                         # generalises to an unseen scene


def test_training_reproducible():
    inst = _instances(2)
    m1 = _model(seed=0)
    m2 = _model(seed=0)
    h1 = train_esd_gnn(m1, inst, PROTO, PHY, ReliabilityThresholds(),
                       steps=15, lr=0.01, eta_mu=8.0, link_override=1.0)["history"]
    h2 = train_esd_gnn(m2, inst, PROTO, PHY, ReliabilityThresholds(),
                       steps=15, lr=0.01, eta_mu=8.0, link_override=1.0)["history"]
    assert abs(h1["F_wrong"][-1] - h2["F_wrong"][-1]) < 1e-9
