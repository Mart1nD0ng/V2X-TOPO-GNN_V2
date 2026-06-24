"""G3 -- round-coupled full physics (spec §7.2-§7.4) + ESP inclusion (spec §4).

Acceptance: each mechanism is active and has the correct causal direction; request and
response are distinct; interference uses G_int (cross-destination); the round duration is
load/quality-coupled (no tau_proxy); near-linear, differentiable.

Tests run at a *feasible* operating point (the dense saturated regime where every node
polls at once is the Phase-5/G8 feasibility study, not a unit-test fixture).
"""

import torch

from src.environment import (
    build_candidate_graph,
    build_interference_graph,
    build_manhattan_scene,
)
from src.environment.round_physics import RoundPhysicsConfig, round_physics
from src.sampling import UniformQueryPolicy
from src.sampling.esp_query import edge_inclusion_probabilities

CFG = RoundPhysicsConfig(subchannels=5, slots_per_window=20)


def _scene_graphs(gx=4, gy=4, per=4, block=120.0, cr=70.0, ir=110.0, seed=0):
    scene = build_manhattan_scene(gx, gy, per, block_m=block, comm_radius=cr, int_radius=ir,
                                  generator=torch.Generator().manual_seed(seed))
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    gi = build_interference_graph(scene.positions, scene.int_radius)
    return scene, gc, gi


def _pi(gc, k):
    lw = UniformQueryPolicy().log_weights(gc)
    return edge_inclusion_probabilities(gc.src_index, gc.dst_index, gc.num_nodes, lw, k)


def _k(gc, want=4):
    return min(want, int(gc.out_degree().min()))


# --------------------------------------------------------------- ESP inclusion
def test_esp_inclusion_sums_to_k():
    scene, gc, gi = _scene_graphs()
    k = _k(gc)
    assert k >= 1
    pi = edge_inclusion_probabilities(gc.src_index, gc.dst_index, gc.num_nodes,
                                      UniformQueryPolicy().log_weights(gc), k)
    per_src = torch.zeros(gc.num_nodes, dtype=pi.dtype).index_add(0, gc.src_index, pi)
    active_src = per_src[gc.out_degree() > 0]
    assert torch.allclose(active_src, torch.full_like(active_src, float(k)), atol=1e-9)
    assert bool((pi >= -1e-12).all()) and bool((pi <= 1 + 1e-9).all())


def test_esp_inclusion_uniform_is_k_over_degree():
    scene, gc, gi = _scene_graphs()
    k = _k(gc)
    pi = edge_inclusion_probabilities(gc.src_index, gc.dst_index, gc.num_nodes,
                                      UniformQueryPolicy().log_weights(gc), k)
    expected = float(k) / gc.out_degree()[gc.src_index].to(pi.dtype)
    assert torch.allclose(pi, expected, atol=1e-9)


# --------------------------------------------------------------- round physics
def test_round_physics_basic_shapes_and_ranges():
    scene, gc, gi = _scene_graphs()
    k = _k(gc)
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 2), 0.2, dtype=torch.float64)
    res = round_physics(gc, gi, pi, active, CFG)
    assert res.ell_poll.shape == (gc.num_edges, 2)
    assert res.tau.shape == (scene.num_nodes, 2)
    assert bool((res.ell_poll >= -1e-9).all()) and bool((res.ell_poll <= 1 + 1e-9).all())
    assert bool((res.tau > 0).all()) and bool((res.energy >= 0).all())
    assert float(res.ell_poll.mean()) > 0.3  # feasible operating point


def test_sinr_mechanisms_lower_poll_success_at_moderate_load():
    """At moderate activity (response feedback second-order) each SINR-side impairment
    can only reduce ell -> disabling it raises ell (causal direction, spec §7.3)."""
    scene, gc, gi = _scene_graphs(per=4)
    k = _k(gc)
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 1), 0.15, dtype=torch.float64)
    full = round_physics(gc, gi, pi, active, CFG).ell_poll
    for flag in ("disable_interference", "disable_collision", "disable_half_duplex"):
        off = round_physics(gc, gi, pi, active, CFG, **{flag: True}).ell_poll
        assert bool((off >= full - 1e-9).all()), f"{flag}: disabling did not raise ell"
        assert float(off.mean()) > float(full.mean()) + 1e-4, f"{flag}: no measurable effect"


def test_queueing_drops_and_delays_under_overload():
    """Queueing only bites under overload: its DROP lowers ell and its DELAY lengthens tau
    (both vanish when disabled). Uses high activity so receiver load > service rate."""
    scene, gc, gi = _scene_graphs(gx=4, gy=4, per=8, block=120.0, cr=70.0, ir=110.0)
    k = _k(gc)
    pi = _pi(gc, k)
    active = torch.ones(scene.num_nodes, 1, dtype=torch.float64)  # overload
    cfg = RoundPhysicsConfig(subchannels=8, slots_per_window=30, service_rate=12.0)
    full = round_physics(gc, gi, pi, active, cfg)
    noq = round_physics(gc, gi, pi, active, cfg, disable_queueing=True)
    assert float(noq.ell_poll.mean()) > float(full.ell_poll.mean()) + 1e-3   # drop removed
    assert float(noq.tau.mean()) < float(full.tau.mean()) - 1e-6             # delay removed


def test_cross_destination_interference_uses_int_graph():
    """Interference over G_int (incl. non-intended transmitters) lowers ell vs aggregating
    over only G_comm -- the spec §7.1 correction is active in the canonical physics."""
    scene, gc, gi = _scene_graphs(per=5)
    k = _k(gc)
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 1), 0.3, dtype=torch.float64)
    ell_int = round_physics(gc, gi, pi, active, CFG).ell_poll      # true G_int
    ell_comm = round_physics(gc, gc, pi, active, CFG).ell_poll      # interference over G_comm only
    assert float(ell_int.mean()) < float(ell_comm.mean())


def test_request_response_are_distinct():
    """Request and response use different blocklengths / interference -> not interchangeable."""
    scene, gc, gi = _scene_graphs(per=4)
    k = _k(gc)
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 1), 0.2, dtype=torch.float64)
    res = round_physics(gc, gi, pi, active, CFG)
    assert not torch.allclose(res.gamma_request, res.gamma_response)
    cfg2 = RoundPhysicsConfig(subchannels=5, slots_per_window=20,
                              response_blocklength=CFG.response_blocklength * 2)
    res2 = round_physics(gc, gi, pi, active, cfg2)
    assert not torch.allclose(res.ell_poll, res2.ell_poll)


def test_round_duration_is_load_coupled_not_proxy():
    """tau must vary with the active mass (load) -- proving no fixed tau_proxy."""
    scene, gc, gi = _scene_graphs(gx=4, gy=4, per=8, block=120.0, cr=70.0, ir=110.0)
    k = _k(gc)
    pi = _pi(gc, k)
    cfg = RoundPhysicsConfig(subchannels=8, slots_per_window=30, service_rate=12.0)
    low = round_physics(gc, gi, pi, torch.full((scene.num_nodes, 1), 0.2, dtype=torch.float64), cfg).tau
    high = round_physics(gc, gi, pi, torch.full((scene.num_nodes, 1), 1.0, dtype=torch.float64), cfg).tau
    assert float(high.mean()) > float(low.mean()) + 1e-9
    assert float(high.std()) > 0


def test_differentiable_in_inclusion():
    scene, gc, gi = _scene_graphs()
    k = _k(gc)
    pi = _pi(gc, k).clone().requires_grad_(True)
    active = torch.full((scene.num_nodes, 1), 0.2, dtype=torch.float64)
    res = round_physics(gc, gi, pi, active, CFG)
    res.ell_poll.sum().backward()
    assert pi.grad is not None and bool(torch.isfinite(pi.grad).all())


def test_scales_to_hundreds_of_nodes():
    scene, gc, gi = _scene_graphs(gx=6, gy=6, per=8)   # several hundred nodes
    k = _k(gc)
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 3), 0.2, dtype=torch.float64)
    res = round_physics(gc, gi, pi, active, CFG)
    assert res.ell_poll.shape == (gc.num_edges, 3)
    assert bool(torch.isfinite(res.ell_poll).all())
