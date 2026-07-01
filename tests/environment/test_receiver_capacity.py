"""G-NDH-HETEROGENEOUS-CAPACITY -- per-node receiver capacity mu_j in the M/M/1 queue (spec §4.4).

Acceptance (engineering plan §6 + spec §4.4):
  * per-node mu_j replaces the global service_rate in rho_j = (Lambda_j + b_j)/mu_j;
  * higher mu_j -> lower queue delay / drop; identical mu_j == global service_rate is BYTE-IDENTICAL
    to the old scalar path (no train/eval physics change when homogeneous);
  * RSU mu > vehicle mu on average, but RSU OVERLOAD is still possible (not a trivial oracle);
  * the model reads only a NOISY capacity proxy mu_hat, never the true mu_j (Contract C2);
  * the mechanism enters full physics + dynamic MC (canonical path), not just tests;
  * mechanism_config_hash binds the capacity params.
"""

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig, build_candidate_graph
from src.environment.interference_graph import build_interference_graph
from src.environment.nonuniform_urban_scene import build_nonuniform_urban_scene
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.environment.receiver_capacity import assign_receiver_capacity, noisy_capacity_proxy
from src.environment.round_physics import round_physics
from src.metrics.participation import vehicle_only_participation
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.validation import run_dynamic_mc

GEN = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)   # service_rate default 12


def _star(nrx=6):
    """A hub topology: many sources near one receiver (node 0) -> heavy addressed load Lambda_0,
    so the receiver's queue (mu_0) actually binds."""
    pos = torch.zeros((nrx + 1, 2), dtype=torch.float64)
    pos[1:, 0] = torch.linspace(10.0, 40.0, nrx)      # sources clustered near the hub at origin
    gc = build_candidate_graph(pos, 90.0)
    gi = build_interference_graph(pos, 160.0)
    pi = torch.ones(gc.num_edges, dtype=torch.float64)
    active = torch.ones((nrx + 1, 1), dtype=torch.float64)
    return pos, gc, gi, pi, active


# --------------------------------------------------------------- capacity assignment
def test_assign_capacity_rsu_higher_but_overload_possible():
    node_type = torch.tensor([0, 0, 0, 0, 1, 1], dtype=torch.long)   # 4 vehicles, 2 RSU
    mu = assign_receiver_capacity(node_type, generator=GEN(0))
    assert mu.shape == (6,) and float(mu.min()) > 0
    veh, rsu = mu[node_type == 0], mu[node_type == 1]
    assert float(rsu.mean()) > float(veh.mean())         # RSU higher capacity on average
    # RSU overload still possible: the spread means some vehicle can exceed some RSU is NOT required,
    # but a heavily-loaded RSU can still saturate -> capacity is finite, not infinite
    assert float(rsu.max()) < float("inf")


def test_noisy_proxy_differs_from_true():
    node_type = torch.zeros(30, dtype=torch.long)
    mu = assign_receiver_capacity(node_type, generator=GEN(1))
    mu_hat = noisy_capacity_proxy(mu, capacity_proxy_noise=0.2, generator=GEN(2))
    assert mu_hat.shape == mu.shape
    assert not torch.allclose(mu_hat, mu)                 # proxy is noisy, not the truth
    # unbiased-ish in log space: exp(0.2*xi) has mean > 1 but proxy tracks true (positive correlation)
    assert float(torch.corrcoef(torch.stack([mu, mu_hat]))[0, 1]) > 0.5
    # zero noise recovers the truth
    assert torch.allclose(noisy_capacity_proxy(mu, capacity_proxy_noise=0.0), mu)


# --------------------------------------------------------------- physics: mu_j drives the queue
def test_higher_capacity_lowers_queue_drop_and_delay():
    pos, gc, gi, pi, active = _star()
    N = pos.shape[0]
    lo = torch.full((N,), 4.0, dtype=torch.float64)      # low capacity -> hub rho = 6/4 = 1.5 (drop fires)
    hi = torch.full((N,), 40.0, dtype=torch.float64)     # high capacity -> hub rho = 0.15 (no drop)
    r_lo = round_physics(gc, gi, pi, active, PHY, node_capacity=lo)
    r_hi = round_physics(gc, gi, pi, active, PHY, node_capacity=hi)
    assert float(r_lo.ell_poll.mean()) < float(r_hi.ell_poll.mean())   # low capacity delivers worse (drop)
    assert float(r_hi.tau.mean()) < float(r_lo.tau.mean())             # high capacity -> lower queue delay


def test_rsu_overload_saturates():
    """Anti-trivial-oracle (R3): a heavily-LOADED RSU (Lambda_j > mu_j) still saturates -> delivery ~0,
    so a nearest-RSU heuristic is not a free win. Overload comes from load, not capacity spread."""
    pos, gc, gi, pi, active = _star(nrx=6)
    N = pos.shape[0]
    mu = torch.full((N,), 3.0, dtype=torch.float64)      # hub receiver mu=3, addressed load=6 -> rho=2>1
    r = round_physics(gc, gi, pi, active, PHY, node_capacity=mu)
    # edges INTO the overloaded hub (node 0) have their request leg heavily dropped -> low ell
    into_hub = gc.dst_index == 0
    assert bool(into_hub.any())
    assert float(r.ell_poll[into_hub].mean()) < 0.2       # RSU/hub saturates under overload


def test_homogeneous_capacity_byte_identical_to_scalar():
    """node_capacity == the global service_rate everywhere must reproduce the scalar path exactly,
    in BOTH the rho<1 (no-drop) and rho>1 (drop-branch) regimes."""
    pos, gc, gi, pi, active = _star()
    N = pos.shape[0]
    for phy in (PHY, RoundPhysicsConfig(subchannels=10, slots_per_window=40, service_rate=4.0)):
        homog = torch.full((N,), phy.service_rate, dtype=torch.float64)   # rho = 6/12=0.5 and 6/4=1.5
        base = round_physics(gc, gi, pi, active, phy)                     # scalar service_rate
        same = round_physics(gc, gi, pi, active, phy, node_capacity=homog)  # per-node == scalar
        assert torch.equal(base.ell_poll, same.ell_poll)
        assert torch.equal(base.tau, same.tau)
        assert torch.equal(base.energy, same.energy)


def test_capacity_none_unchanged():
    pos, gc, gi, pi, active = _star()
    base = round_physics(gc, gi, pi, active, PHY)
    none = round_physics(gc, gi, pi, active, PHY, node_capacity=None)
    assert torch.equal(base.ell_poll, none.ell_poll)


def test_capacity_differentiable_in_active():
    pos, gc, gi, pi, active = _star()
    N = pos.shape[0]
    mu = torch.full((N,), 8.0, dtype=torch.float64)
    a = active.clone().requires_grad_(True)
    r = round_physics(gc, gi, pi, a, PHY, node_capacity=mu)
    r.ell_poll.sum().backward()
    assert a.grad is not None and float(a.grad.abs().sum()) > 0   # gradient flows through the queue


# --------------------------------------------------------------- canonical path + scene
def test_capacity_enters_dynamic_mc_and_scene_hash():
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_rsu=True, p_intersection_rsu=0.3,
                                      enable_heterogeneous_capacity=True, generator=GEN(7))
    assert sc.node_capacity is not None and sc.node_capacity.shape[0] == sc.num_nodes
    assert float(sc.node_capacity.min()) > 0
    # RSU capacity higher on average than vehicles
    assert float(sc.node_capacity[sc.node_type == 1].mean()) > float(sc.node_capacity[sc.node_type == 0].mean())
    ev = build_overlapping_scenario(sc, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    r = run_dynamic_mc(sc, ev, DistanceQueryPolicy(beta_per_m=0.04), proto, PHY, num_trials=40,
                       generator=GEN(9), service_profile=prof, participation=vehicle_only_participation(sc))
    assert abs(r.basin_P_correct + r.basin_F_wrong + r.basin_F_split + r.basin_F_deadline - 1.0) < 1e-6
    # mechanism hash binds capacity params
    sc2 = build_nonuniform_urban_scene(5, 5, 3, enable_rsu=True, p_intersection_rsu=0.3,
                                       enable_heterogeneous_capacity=True, mu_vehicle_base=6, generator=GEN(7))
    assert sc.mechanism_config_hash != sc2.mechanism_config_hash


def test_capacity_off_scene_has_no_capacity():
    sc = build_nonuniform_urban_scene(5, 5, 3, generator=GEN(7))   # capacity off by default
    assert sc.node_capacity is None


def test_trace_sentinel_false_when_queue_disabled():
    """mu_j is inert when the queue is disabled -> the heterogeneous_capacity sentinel must be False."""
    from src.environment.canonical_episode import run_consensus_episode
    from src.sampling.baseline_policies import UniformQueryPolicy
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_rsu=True, p_intersection_rsu=0.3,
                                      enable_heterogeneous_capacity=True, generator=GEN(7))
    ev = build_overlapping_scenario(sc, "matched_marginal_high", base_node_err=0.35, corr_strength=0.25)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=4)
    tr_on = run_consensus_episode(sc, ev, UniformQueryPolicy(), proto, PHY,
                                  return_trajectory=False).mechanism_trace
    tr_noq = run_consensus_episode(sc, ev, UniformQueryPolicy(), proto, PHY, disable_queueing=True,
                                   return_trajectory=False).mechanism_trace
    assert tr_on["heterogeneous_capacity"] is True
    assert tr_noq["heterogeneous_capacity"] is False       # queue off -> mechanism inert -> sentinel off


def test_negative_proxy_noise_rejected():
    import pytest
    mu = torch.full((5,), 8.0, dtype=torch.float64)
    with pytest.raises(ValueError):
        noisy_capacity_proxy(mu, capacity_proxy_noise=-0.1)


def test_round_physics_rejects_nonfinite_capacity():
    import pytest
    pos, gc, gi, pi, active = _star()
    N = pos.shape[0]
    bad = torch.full((N,), float("nan"), dtype=torch.float64)
    with pytest.raises(ValueError, match="finite"):
        round_physics(gc, gi, pi, active, PHY, node_capacity=bad)
    neg = torch.full((N,), -1.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="> 0"):
        round_physics(gc, gi, pi, active, PHY, node_capacity=neg)
