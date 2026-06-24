"""Interference propagation graph ``G_int`` (spec §7.1 -- the cross-destination fix).

The spec's key physics correction: interference at a receiver ``j`` must include **every
active transmitter within interference range of ``j``**, regardless of that transmitter's
*intended* destination. Aggregating interference over only the communication edges into
``j`` (the legacy ``aggregate_interference`` keyed on ``G_comm`` ``dst``) misses a
transmission ``t -> m`` (``m ≠ j``) that still raises the interference floor at ``j``.

``G_int`` is therefore a separate radius graph at ``int_radius ≥ comm_radius``. Every
communication edge is also an interference edge (``G_comm ⊆ G_int``), but ``G_int`` adds
the *non-intended* interferer pairs ``(t, j)`` with ``t`` near ``j`` yet not polling ``j``.
:func:`non_intended_interferers` extracts exactly that set -- the mechanism the headline
must use and the legacy path drops.

Built with the same cell-list core as ``G_comm`` -- ``O(N+E)`` at fixed density, no
``N x N`` tensor, no degree cap.
"""

from __future__ import annotations

import torch

from .candidate_graph import RadiusGraph, aggregate_over_graph, build_radius_graph, edge_set

__all__ = [
    "build_interference_graph",
    "non_intended_interferers",
    "received_interference_mw",
]


def build_interference_graph(positions: torch.Tensor, int_radius: float,
                             *, cell_size: float | None = None) -> RadiusGraph:
    """The interference propagation graph ``G_int`` at ``int_radius`` (spec §7.1).

    ``int_radius`` should be ``>= comm_radius`` so that ``G_comm ⊆ G_int`` (every intended
    link also propagates interference); the extra edges are the non-intended interferers.
    """
    return build_radius_graph(positions, int_radius, cell_size=cell_size)


def non_intended_interferers(comm_graph: RadiusGraph, int_graph: RadiusGraph) -> set[tuple[int, int]]:
    """Edges ``(t, j) ∈ G_int \\ G_comm`` -- interferers that do NOT intend to poll ``j``.

    A non-empty result is the sentinel that the interference graph carries strictly more
    information than the candidate graph (spec §7.1); the headline interference floor must
    include these (they are dropped by destination-keyed aggregation over ``G_comm``).
    """
    return edge_set(int_graph) - edge_set(comm_graph)


def received_interference_mw(
    int_graph: RadiusGraph,
    edge_rx_power_mw: torch.Tensor,   # [E_int] received power t->j on each interference edge
    tx_activity: torch.Tensor,        # [N] expected transmit activity per node (in [0,1]+)
    subchannels: float,
) -> torch.Tensor:
    """Expected co-channel interference power at each receiver ``[N]`` over ``G_int``.

    ``I_j = (1/S) sum_{t->j in G_int} a_t · rx_power(t->j)`` where ``a_t`` is transmitter
    ``t``'s activity and each contender collides on the same sub-channel with probability
    ``1/S`` (Mode-2 random selection). Summed over **all** transmitters near ``j``, not
    just those polling ``j`` -- this is the spec §7.1 correction. Differentiable; ``O(E)``.

    The per-edge received power is supplied by the round-physics layer (geometry/path-loss
    is owned there, spec §7.3); this function owns only the ``G_int`` aggregation.
    """
    if subchannels < 1.0:
        raise ValueError("subchannels must be >= 1")
    if edge_rx_power_mw.shape[0] != int_graph.num_edges:
        raise ValueError("edge_rx_power_mw must have one entry per interference edge")
    a_src = tx_activity[int_graph.src_index]
    contrib = (1.0 / float(subchannels)) * a_src * edge_rx_power_mw
    return aggregate_over_graph(int_graph, contrib)
