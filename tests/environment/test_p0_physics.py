"""G-P0-PHYSICS: P0-C source/destination ownership + P0-D collision self-exclusion.

Spec §5.4 (source/destination responsibility) and §5.5 (collision excludes the desired
transmission). A hand-built collinear 3-node reference A—B—C (only adjacent nodes in range)
makes the attribution unambiguous:

* with only A active, A polls B:
    - request TX energy is charged to the SOURCE A (not the destination B) — P0-C;
    - response TX energy is charged to the responder = DESTINATION B — P0-C;
    - the single active poll suffers ZERO collision — P0-D (single transmission ⇒ p_col=0);
    - source request activity A_A^req = k·u_A — P0-B/§5.4.
"""

import torch

from src.environment.candidate_graph import (
    build_candidate_graph,
    scatter_destination,
    scatter_source,
)
from src.environment.interference_graph import build_interference_graph
from src.environment.round_physics import RoundPhysicsConfig, round_physics
from src.sampling.esp_query import edge_inclusion_probabilities
from src.sampling import UniformQueryPolicy

CFG = RoundPhysicsConfig(subchannels=5, slots_per_window=20)


def _line_graphs(comm=12.0, intr=12.0):
    pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]], dtype=torch.float64)
    gc = build_candidate_graph(pos, comm)
    gi = build_interference_graph(pos, intr)
    return pos, gc, gi


def _pi(gc, k):
    lw = UniformQueryPolicy().log_weights(gc)
    return edge_inclusion_probabilities(gc.src_index, gc.dst_index, gc.num_nodes, lw, k)


# ---------------------------------------------------------- explicit scatter helpers
def test_scatter_source_and_destination_are_distinct_and_correct():
    _, gc, _ = _line_graphs()
    val = torch.ones(gc.num_edges, 1, dtype=torch.float64)
    out_src = scatter_source(gc, val, gc.num_nodes)     # out-degree per node
    out_dst = scatter_destination(gc, val, gc.num_nodes)  # in-degree per node
    assert torch.allclose(out_src.squeeze(-1), gc.out_degree().to(torch.float64))
    assert torch.allclose(out_dst.squeeze(-1), gc.in_degree().to(torch.float64))
    # on a symmetric graph degrees match, so use an asymmetric per-edge value to separate them
    asym = gc.src_index.to(torch.float64).unsqueeze(-1)   # edge value = its source id
    s = scatter_source(gc, asym, gc.num_nodes).squeeze(-1)
    # node i's scatter_source sum = (its out-degree) * i
    assert torch.allclose(s, gc.out_degree().to(torch.float64) * torch.arange(gc.num_nodes, dtype=torch.float64))


# ---------------------------------------------------------- P0-D collision self-exclusion
def test_single_active_poll_has_zero_collision():
    _, gc, gi = _line_graphs()
    k = 1
    pi = _pi(gc, k)
    active = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)  # only A polls
    res = round_physics(gc, gi, pi, active, CFG)
    # A polls exactly one peer (B); no other active transmitter ⇒ collision must be exactly 0
    # locate edge A->B
    ab = ((gc.src_index == 0) & (gc.dst_index == 1)).nonzero().item()
    assert float(res.p_collision_request[ab]) == 0.0
    # and the realised poll success on A->B is not suppressed by a phantom self-collision
    assert float(res.ell_poll[ab]) > 0.0


def test_collision_returns_when_a_second_transmitter_contends():
    _, gc, gi = _line_graphs()
    k = 1
    pi = _pi(gc, k)
    # B and C both active and both within int-range of the shared receiver region ⇒ contention
    active = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float64)  # A and C both poll B
    res = round_physics(gc, gi, pi, active, CFG)
    ab = ((gc.src_index == 0) & (gc.dst_index == 1)).nonzero().item()
    # now A->B sees C as a co-channel contender near B ⇒ strictly positive collision
    assert float(res.p_collision_request[ab]) > 0.0


# ---------------------------------------------------------- P0-C source/destination energy
def test_request_energy_charged_to_source_response_to_destination():
    _, gc, gi = _line_graphs()
    k = 1
    pi = _pi(gc, k)
    active = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)  # only A active, polls B
    res = round_physics(gc, gi, pi, active, CFG)
    er = res.energy_request.squeeze(-1)
    ep = res.energy_response.squeeze(-1)
    # request TX energy belongs to the SOURCE/poller A (node 0), not the destination B
    assert float(er[0]) > 0.0
    assert float(er[1]) == 0.0 and float(er[2]) == 0.0
    # response TX energy belongs to the responder = DESTINATION B (node 1)
    assert float(ep[1]) > 0.0
    assert float(ep[0]) == 0.0 and float(ep[2]) == 0.0
    # total energy is the sum of the two legs
    assert torch.allclose(res.energy, res.energy_request + res.energy_response)


def test_source_request_activity_is_k_times_active():
    _, gc, gi = _line_graphs()
    k = 1
    pi = _pi(gc, k)
    active = torch.tensor([[0.7], [0.4], [0.0]], dtype=torch.float64)
    res = round_physics(gc, gi, pi, active, CFG)
    # A_i^req = sum_j a_ij = k * u_i   (spec §5.4); only active sources with >=k peers
    expected = active.squeeze(-1) * float(k)
    # nodes 0,1 have out-degree >= k=1
    assert torch.allclose(res.source_activity.squeeze(-1)[:2], expected[:2], atol=1e-9)


def test_tau_attributed_to_poller_source():
    # the poller's epoch-completion time is a SOURCE quantity: an active poller has tau>0,
    # an isolated inactive non-polled node accrues no polling time of its own.
    _, gc, gi = _line_graphs()
    k = 1
    pi = _pi(gc, k)
    active = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float64)
    res = round_physics(gc, gi, pi, active, CFG)
    assert float(res.tau[0]) > 0.0     # A is the active poller
