"""Manhattan urban V2X scene: street geometry + region (road-segment) structure (spec §6).

Generates vehicle positions clustered along the road segments of a Manhattan grid, with
each vehicle's **region** = the road segment it occupies. Spatial clustering by segment is
what creates the local-evidence correlation studied by the evidence model
(``evidence_model.py``): vehicles on the same segment are physically close (high link
quality) *and* share the same occlusion / sensor region (correlated observations) -- the
core "local delivery quality vs global evidence diversity" tension (spec §9.1).

The scene owns only geometry + region assignment; the two physical graphs are built by
``candidate_graph.build_candidate_graph`` (``G_comm``) and
``interference_graph.build_interference_graph`` (``G_int``), and the per-region/per-node
error probabilities are attached by the evidence model. ``comm_radius`` / ``int_radius``
are carried on the scene as recommended physical radii for those builders.

No ``N x N`` structure; ``N = num_segments · vehicles_per_segment``; positions and region
ids are deterministic given the ``torch.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["ManhattanScene", "build_manhattan_scene"]


@dataclass(frozen=True)
class ManhattanScene:
    positions: torch.Tensor        # [N, 2] vehicle coordinates (metres)
    region_of: torch.Tensor        # [N] long, road-segment id g(i) in {0..G-1}
    segment_endpoints: torch.Tensor  # [G, 2, 2] (start_xy, end_xy) per segment
    comm_radius: float
    int_radius: float
    block_m: float
    grid: tuple[int, int]          # (gx, gy) intersections per axis

    @property
    def num_nodes(self) -> int:
        return int(self.region_of.numel())

    @property
    def num_regions(self) -> int:
        return int(self.segment_endpoints.shape[0])


def _segments(gx: int, gy: int, block_m: float) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Horizontal + vertical road segments between adjacent intersections of a gx*gy grid."""
    segs: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for iy in range(gy):
        for ix in range(gx - 1):
            segs.append(((ix * block_m, iy * block_m), ((ix + 1) * block_m, iy * block_m)))
    for ix in range(gx):
        for iy in range(gy - 1):
            segs.append(((ix * block_m, iy * block_m), (ix * block_m, (iy + 1) * block_m)))
    return segs


def build_manhattan_scene(
    gx: int,
    gy: int,
    vehicles_per_segment: int,
    *,
    block_m: float = 100.0,
    lane_jitter_m: float = 3.0,
    comm_radius: float = 80.0,
    int_radius: float = 160.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> ManhattanScene:
    """Build a Manhattan scene with ``vehicles_per_segment`` vehicles on each road segment.

    Vehicles are placed at evenly-spaced positions along each segment (avoiding the exact
    intersections) with a small lateral lane offset ``U(-lane_jitter_m, +lane_jitter_m)``,
    so every vehicle stays within ``lane_jitter_m`` of its segment line -- a property the
    region-containment test relies on. Deterministic given ``generator``.
    """
    if gx < 1 or gy < 1 or (gx - 1) * gy + gx * (gy - 1) < 1:
        raise ValueError("grid too small: need at least one road segment")
    if vehicles_per_segment < 1:
        raise ValueError("vehicles_per_segment must be >= 1")
    if int_radius < comm_radius:
        raise ValueError("int_radius must be >= comm_radius")

    segs = _segments(gx, gy, block_m)
    G = len(segs)
    n = vehicles_per_segment
    pos = torch.empty((G * n, 2), dtype=dtype)
    region = torch.empty((G * n,), dtype=torch.long)
    endpoints = torch.tensor(segs, dtype=dtype)  # [G, 2, 2]

    # fractional positions along each segment: (k+0.5)/n in (0,1), avoids intersections
    frac = (torch.arange(n, dtype=dtype) + 0.5) / n  # [n]
    for s, (a, b) in enumerate(segs):
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        length = (dx * dx + dy * dy) ** 0.5
        # unit normal for lateral lane offset
        nx, ny = (-dy / length, dx / length)
        jitter = (torch.rand(n, generator=generator, dtype=dtype) * 2 - 1) * lane_jitter_m  # [n]
        x = ax + frac * dx + jitter * nx
        y = ay + frac * dy + jitter * ny
        rows = slice(s * n, (s + 1) * n)
        pos[rows, 0] = x
        pos[rows, 1] = y
        region[rows] = s

    return ManhattanScene(
        positions=pos,
        region_of=region,
        segment_endpoints=endpoints,
        comm_radius=comm_radius,
        int_radius=int_radius,
        block_m=block_m,
        grid=(gx, gy),
    )
