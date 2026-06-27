"""G-P0-PHYSICS P0-E: unified poll-window ell(Delta_poll) with a timeout factor (spec §5.2).

The polling epoch is a FIXED window Delta_poll. A poll succeeds only if the request+response
round-trip both decode AND complete within the window. We model the window as a HARQ
round-trip budget: after the M/M/1 queue wait consumes part of Delta_poll, the remaining time
allows ``M_win = (Delta_poll/slot - queue_slots)/rt_slots`` HARQ round-trips (capped at the
configured max, floored at 0); the leg decode probabilities are evaluated at that (soft,
differentiable) budget. Limits: Delta_poll -> large recovers the no-timeout ell; Delta_poll
-> 0 gives ell -> 0; a heavier queue shortens the budget and lowers ell.
"""

import torch

from src.environment.candidate_graph import build_candidate_graph
from src.environment.interference_graph import build_interference_graph
from src.environment.round_physics import RoundPhysicsConfig, round_physics
from src.environment.urban_scene import build_manhattan_scene
from src.sampling import UniformQueryPolicy
from src.sampling.esp_query import edge_inclusion_probabilities


def _scene_graphs(gx=4, gy=4, per=4, block=120.0, cr=70.0, ir=110.0, seed=0):
    scene = build_manhattan_scene(gx, gy, per, block_m=block, comm_radius=cr, int_radius=ir,
                                  generator=torch.Generator().manual_seed(seed))
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    gi = build_interference_graph(scene.positions, scene.int_radius)
    return scene, gc, gi


def _pi(gc, k):
    lw = UniformQueryPolicy().log_weights(gc)
    return edge_inclusion_probabilities(gc.src_index, gc.dst_index, gc.num_nodes, lw, k)


def test_large_window_recovers_no_timeout_ell():
    """A poll window large enough to fit the full HARQ budget leaves ell at its no-timeout value."""
    scene, gc, gi = _scene_graphs()
    k = min(4, int(gc.out_degree().min()))
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 1), 0.2, dtype=torch.float64)
    # request_slots+response_slots = 2 slots/round-trip; max_harq=2 -> needs >=4 slots; 1000ms is plenty
    big = RoundPhysicsConfig(subchannels=5, slots_per_window=20, poll_window_s=1.0)
    huge = RoundPhysicsConfig(subchannels=5, slots_per_window=20, poll_window_s=100.0)
    ell_big = round_physics(gc, gi, pi, active, big).ell_poll
    ell_huge = round_physics(gc, gi, pi, active, huge).ell_poll
    assert torch.allclose(ell_big, ell_huge, atol=1e-9)   # saturated: more time doesn't help


def test_tiny_window_drives_ell_to_zero():
    scene, gc, gi = _scene_graphs()
    k = min(4, int(gc.out_degree().min()))
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 1), 0.2, dtype=torch.float64)
    tiny = RoundPhysicsConfig(subchannels=5, slots_per_window=20, poll_window_s=1e-7)  # << 1 slot
    ell = round_physics(gc, gi, pi, active, tiny).ell_poll
    assert float(ell.max()) < 1e-6   # ~no round-trip fits -> timeout (both legs ~ budget -> 0)


def test_ell_is_monotone_increasing_in_poll_window():
    scene, gc, gi = _scene_graphs()
    k = min(4, int(gc.out_degree().min()))
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 1), 0.2, dtype=torch.float64)
    means = []
    for w in (1e-3, 2e-3, 4e-3, 1e-2, 1e-1):
        cfg = RoundPhysicsConfig(subchannels=5, slots_per_window=20, poll_window_s=w)
        means.append(float(round_physics(gc, gi, pi, active, cfg).ell_poll.mean()))
    for a, b in zip(means, means[1:]):
        assert b >= a - 1e-9   # longer window never lowers success


def test_heavier_queue_shortens_budget_and_lowers_ell():
    """A receiver-overloading load lengthens the M/M/1 wait, consuming the poll window and
    lowering ell beyond the plain queue-drop effect (the window-budget coupling, spec §5.2)."""
    scene, gc, gi = _scene_graphs(gx=4, gy=4, per=8, block=120.0, cr=70.0, ir=110.0)
    k = min(2, int(gc.out_degree().min()))
    pi = _pi(gc, k)
    active = torch.ones(scene.num_nodes, 1, dtype=torch.float64)
    # moderate window so the queue wait materially eats the budget; low service rate => long wait
    busy = RoundPhysicsConfig(subchannels=8, slots_per_window=30, poll_window_s=6e-3, service_rate=3.0)
    full = round_physics(gc, gi, pi, active, busy).ell_poll
    noq = round_physics(gc, gi, pi, active, busy, disable_queueing=True).ell_poll
    assert float(noq.mean()) > float(full.mean()) + 1e-4   # queue (delay+drop) lowers ell


def test_timeout_factor_is_differentiable_in_inclusion():
    scene, gc, gi = _scene_graphs()
    k = min(4, int(gc.out_degree().min()))
    pi = _pi(gc, k).clone().requires_grad_(True)
    active = torch.full((scene.num_nodes, 1), 0.4, dtype=torch.float64)
    cfg = RoundPhysicsConfig(subchannels=5, slots_per_window=20, poll_window_s=4e-3, service_rate=4.0)
    ell = round_physics(gc, gi, pi, active, cfg).ell_poll
    ell.sum().backward()
    assert pi.grad is not None and bool(torch.isfinite(pi.grad).all())
    assert float(pi.grad.abs().sum()) > 0   # the window/queue budget couples ell to pi


def test_ell_in_unit_interval_under_window():
    scene, gc, gi = _scene_graphs()
    k = min(4, int(gc.out_degree().min()))
    pi = _pi(gc, k)
    active = torch.full((scene.num_nodes, 2), 0.5, dtype=torch.float64)
    cfg = RoundPhysicsConfig(subchannels=5, slots_per_window=20, poll_window_s=3e-3)
    ell = round_physics(gc, gi, pi, active, cfg).ell_poll
    assert bool((ell >= -1e-12).all()) and bool((ell <= 1.0 + 1e-9).all())
