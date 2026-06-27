"""Physical radius graphs via cell lists -- no degree cap, no ``N x N`` tensor (spec §7.1).

The communication candidate graph ``G_comm`` holds the *intended* query links: a directed
edge ``i -> j`` exists iff ``j`` is within ``comm_radius`` of ``i`` (``i`` may poll ``j``).
It is built with spatial hashing (cell lists) so the cost is ``O(N + E)`` at fixed spatial
density -- **no** ``N x N`` distance matrix is ever formed, and there is **no** fixed
per-node degree cap and **no** top-k truncation (non-negotiable constraint #4, #11). The
full physical neighbourhood is kept; sparsity/hub-avoidance must emerge from physical cost
downstream, never from a cap.

The same core (:func:`build_radius_graph`) builds the interference graph ``G_int`` at a
larger radius (see ``interference_graph.py``); ``G_comm ⊆ G_int`` when ``r_int ≥ r_comm``.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass

import torch

__all__ = [
    "RadiusGraph",
    "build_radius_graph",
    "build_candidate_graph",
    "aggregate_over_graph",
    "scatter_source",
    "scatter_destination",
    "edge_set",
]


@dataclass(frozen=True)
class RadiusGraph:
    """Directed radius graph ``i -> j`` for all ``j`` within ``radius`` of ``i``."""

    src_index: torch.Tensor   # [E] long
    dst_index: torch.Tensor   # [E] long
    distance: torch.Tensor    # [E] float, differentiable in positions
    num_nodes: int
    radius: float

    @property
    def num_edges(self) -> int:
        return int(self.src_index.numel())

    def out_degree(self) -> torch.Tensor:
        return torch.bincount(self.src_index, minlength=self.num_nodes)

    def in_degree(self) -> torch.Tensor:
        return torch.bincount(self.dst_index, minlength=self.num_nodes)


def build_radius_graph(
    positions: torch.Tensor,
    radius: float,
    *,
    cell_size: float | None = None,
) -> RadiusGraph:
    """Directed radius graph via cell lists -- ``O(N+E)`` at fixed density, no ``N x N``.

    Args:
        positions: ``[N, D]`` node coordinates (D = 2 or 3).
        radius: connection radius (same units as positions).
        cell_size: cell side (defaults to ``radius``; must be ``>= radius`` so the
            ``3^D`` neighbour cells fully cover the radius).

    Returns:
        :class:`RadiusGraph` with edges sorted by ``(src, dst)`` (deterministic). No
        self-loops; both ``i->j`` and ``j->i`` are present (the relation is symmetric).
    """
    if positions.ndim != 2:
        raise ValueError("positions must be [N, D]")
    N, D = positions.shape
    r = float(radius)
    if r <= 0:
        raise ValueError("radius must be positive")
    cs = r if cell_size is None else float(cell_size)
    if cs < r:
        raise ValueError("cell_size must be >= radius so neighbour cells cover the radius")

    pos_np = positions.detach().cpu().numpy()
    cells = (pos_np // cs).astype(int)
    cell_map: dict[tuple, list[int]] = defaultdict(list)
    for idx in range(N):
        cell_map[tuple(cells[idx])].append(idx)
    offsets = list(itertools.product((-1, 0, 1), repeat=D))
    r2 = r * r
    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(N):
        ci = tuple(cells[i])
        pi = pos_np[i]
        seen: set[int] = set()
        for off in offsets:
            nc = tuple(ci[d] + off[d] for d in range(D))
            for j in cell_map.get(nc, ()):  # only nearby cells -> O(1) per node at fixed density
                if j == i or j in seen:
                    continue
                seen.add(j)
                diff = pos_np[j] - pi
                if float((diff * diff).sum()) <= r2:
                    src_list.append(i)
                    dst_list.append(j)
    device = positions.device
    if not src_list:
        empty = torch.zeros((0,), dtype=torch.long, device=device)
        return RadiusGraph(empty, empty, positions.new_zeros((0,)), N, r)
    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    order = torch.argsort(src * N + dst)  # deterministic (src, dst) order
    src, dst = src[order], dst[order]
    diff = positions[src] - positions[dst]
    dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # differentiable in positions
    return RadiusGraph(src_index=src, dst_index=dst, distance=dist, num_nodes=N, radius=r)


def build_candidate_graph(positions: torch.Tensor, comm_radius: float,
                          *, cell_size: float | None = None) -> RadiusGraph:
    """The communication candidate graph ``G_comm`` (intended query links, spec §7.1)."""
    return build_radius_graph(positions, comm_radius, cell_size=cell_size)


def scatter_source(graph: RadiusGraph, edge_value: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Scatter-add a per-edge value to its SOURCE: ``out[i] = sum_{i->* } value`` (P0-C).

    The explicit *source-ownership* aggregation (spec §5.4): poller epoch completion,
    request TX energy and source request activity ``A_i^req = sum_j a_ij`` are charged to the
    transmitting source ``i``. Differentiable in ``edge_value``; ``O(E)``; no ``N x N``.
    """
    if edge_value.shape[0] != graph.num_edges:
        raise ValueError("edge_value must have one entry per edge")
    out = edge_value.new_zeros((num_nodes, *edge_value.shape[1:]))
    return out.index_add(0, graph.src_index, edge_value)


def scatter_destination(graph: RadiusGraph, edge_value: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Scatter-add a per-edge value to its DESTINATION: ``out[j] = sum_{*->j} value`` (P0-C).

    The explicit *destination-ownership* aggregation (spec §5.4): receiver addressed load
    ``Lambda_j``, receiver congestion / queueing and response TX energy are charged to the
    receiving / responding destination ``j``. Differentiable; ``O(E)``; no ``N x N``.
    """
    if edge_value.shape[0] != graph.num_edges:
        raise ValueError("edge_value must have one entry per edge")
    out = edge_value.new_zeros((num_nodes, *edge_value.shape[1:]))
    return out.index_add(0, graph.dst_index, edge_value)


def aggregate_over_graph(graph: RadiusGraph, edge_value: torch.Tensor) -> torch.Tensor:
    """Scatter-add a per-edge value to its destination: ``out[j] = sum_{i->j} value``.

    Thin wrapper over :func:`scatter_destination` kept for back-compatibility; new code
    should call the explicitly-named source/destination helpers (P0-C).
    """
    return scatter_destination(graph, edge_value, graph.num_nodes)


def edge_set(graph: RadiusGraph) -> set[tuple[int, int]]:
    """Edges as a Python set of ``(src, dst)`` tuples (tests / set algebra, small N)."""
    return set(zip(graph.src_index.tolist(), graph.dst_index.tolist()))
