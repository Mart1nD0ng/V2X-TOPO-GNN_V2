"""Temporal scene drift -- vehicle mobility for G12 robustness (spec §9.7, §G12).

Vehicles drive ALONG their road segments: each vehicle's along-segment coordinate takes a random
step (with fresh lateral lane jitter), so positions -- and therefore the radius candidate graph
``G_comm`` / interference graph ``G_int`` -- change between time steps (TOPOLOGY CHURN), while the
region membership ``g(i)`` is preserved (vehicles stay on their road), keeping the evidence model's
region structure valid. ``churn_frac`` optionally relocates a fraction of vehicles to a random
segment (hand-offs / sudden weak-cut). Deterministic given ``generator``; no ``N x N`` structure.

The deployed ESD-GNN is a MEMORYLESS function of observable structure, so it re-adapts to a drifted
scene by recomputing features -- this module lets us verify that reliability holds under drift. The
contractive temporal-MEMORY model (§9.7 GRU/SSM) is a deferred extension, added only if it earns a
measured benefit (plan §15); it is NOT part of the verified static headline.
"""

from __future__ import annotations

from dataclasses import replace

import torch

from src.environment.urban_scene import ManhattanScene

__all__ = ["drift_scene"]


def drift_scene(
    scene: ManhattanScene,
    *,
    step_frac: float = 0.1,
    lane_jitter_m: float = 3.0,
    churn_frac: float = 0.0,
    generator: torch.Generator | None = None,
) -> ManhattanScene:
    """Return a new scene with vehicles moved along their segments (region-preserving churn).

    ``step_frac``: stddev of the along-segment step as a fraction of segment length.
    ``churn_frac``: fraction of vehicles relocated to a uniformly random along-position of their
    own segment (a larger jump -- models a vehicle that moved a lot in one step). Regions, segment
    endpoints and radii are unchanged, so the evidence/region model stays valid.
    """
    dtype = scene.positions.dtype
    N = scene.num_nodes
    g = scene.region_of
    start = scene.segment_endpoints[:, 0, :]                 # [G, 2]
    end = scene.segment_endpoints[:, 1, :]                   # [G, 2]
    seg_vec = end - start                                     # [G, 2]
    seg_len = seg_vec.norm(dim=1).clamp_min(1e-9)            # [G]
    dirn = seg_vec / seg_len.unsqueeze(1)                     # [G, 2] unit along-direction

    rel = scene.positions - start[g]                          # [N, 2]
    t = (rel * dirn[g]).sum(dim=1) / seg_len[g]               # [N] current along-fraction
    dt = torch.randn(N, generator=generator, dtype=dtype) * step_frac
    t_new = (t + dt).clamp(0.02, 0.98)

    if churn_frac > 0.0:
        u = torch.rand(N, generator=generator, dtype=dtype)
        jump = u < churn_frac
        t_jump = torch.rand(N, generator=generator, dtype=dtype).clamp(0.02, 0.98)
        t_new = torch.where(jump, t_jump, t_new)

    along = start[g] + t_new.unsqueeze(1) * seg_vec[g]        # [N, 2] on the segment line
    perp = torch.stack([-dirn[g][:, 1], dirn[g][:, 0]], dim=1)   # [N, 2] unit perpendicular
    lat = (torch.rand(N, generator=generator, dtype=dtype) * 2 - 1) * lane_jitter_m
    positions = along + lat.unsqueeze(1) * perp

    return replace(scene, positions=positions)
