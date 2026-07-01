"""Sensing-based SPS resource persistence surrogate (G-NDH-SPS-PERSISTENCE, spec §3).

Semi-persistent scheduling (SPS) gives each node a *persistent* Mode-2 resource bucket; nodes that
picked the SAME bucket and sit in each other's interference neighbourhood collide **repeatedly** —
a non-local, history-correlated collision structure that distance cannot explain (``d_ij↓ ⇏ resource
conflict↓``). This module assigns the buckets with a lightweight SENSING-based surrogate and exposes
the deployable proxies a model / heuristic may read.

Phase-1 buckets are STATIC per scene: the SPS reselection interval (~1 s) is much longer than one
consensus episode (~60-200 ms at Δ_poll=10 ms), so within a single decision the buckets are
effectively frozen. The static assignment is therefore a faithful surrogate for the persistence that
matters (same-bucket neighbours colliding round after round), and it needs no mutable state threaded
through the dynamic-MC judge. The collision physics itself lives in ``round_physics`` (gated by
``RoundPhysicsConfig.resource_collision_kappa`` + a supplied ``resource_bucket``); this module only
produces the static bucket field and its observable proxies. Temporal reselection (using
``keep_probability``) is deferred to Phase 2.

Deployability (Contract C2): the assignment uses only observable geometry + a *sensed* occupancy
surrogate (with noise). No future resource id, no future collision/delivery outcome, no MC truth
enters — the exact forbidden set of spec §3.5.
"""

from __future__ import annotations

import torch

from .candidate_graph import scatter_destination
from .interference_graph import build_interference_graph

__all__ = ["assign_sps_buckets", "same_resource_conflict_degree", "sensed_channel_busy_ratio",
           "assert_sps_pool_consistent"]


def assert_sps_pool_consistent(scene, phy_cfg) -> None:
    """Guard: the SPS reservation pool must fit inside the physics Mode-2 resource pool.

    TWO DISTINCT pools (registry §2, kept separate on purpose):
      * ``S_phys = subchannels * slots_per_window`` (``RoundPhysicsConfig.resource_pool``) is the
        INSTANTANEOUS per-window pool that drives the memoryless ``1/S`` collision and the SINR
        interference floor.
      * ``sps_n_buckets`` (``S_sps``) is the number of distinct PERSISTENT SPS reservations. Because an
        SPS reservation occupies one subchannel on a *periodic* slot pattern, there are fewer distinct
        reservations than instantaneous (subchannel x slot) resources: ``S_sps <= S_phys``. A smaller
        ``S_sps`` is the congestion knob that makes same-reservation neighbours collide persistently
        (the non-distance structure); the registry sweep {40,60,100} sits at/below the S_phys default.
    So the invariant is ``1 <= sps_n_buckets <= S_phys`` (NOT equality). A pool that EXCEEDS the physical
    pool is unphysical (more reservations than resources) and rejected. No-op unless SPS is active
    (bucket present + kappa > 0). Called from both canonical entry points so it holds train==eval.
    """
    if getattr(scene, "resource_bucket", None) is None or phy_cfg.resource_collision_kappa <= 0.0:
        return
    params = getattr(scene, "params", None) or {}
    n_b = params.get("sps_n_buckets")
    S = round(float(phy_cfg.resource_pool))
    if n_b is not None and not (1 <= int(n_b) <= S):
        raise ValueError(
            f"SPS reservation pool sps_n_buckets={int(n_b)} must satisfy 1 <= S_sps <= physics "
            f"resource_pool S_phys = subchannels*slots_per_window = {S} (registry §2; reservations "
            f"are a subset of the instantaneous pool)")


def assign_sps_buckets(
    positions: torch.Tensor,
    int_radius: float,
    n_buckets: int,
    *,
    tau_res: float = 4.0,
    sensing_noise_std: float = 0.1,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Static sensing-based SPS bucket assignment ``r_u in {0..n_buckets-1}`` (spec §3.3).

    Nodes select resources one at a time (random order) to AVOID sensed occupancy: a node ``v`` senses
    how many of its already-assigned interference neighbours occupy each bucket (plus Gaussian sensing
    noise) and samples a bucket with ``P(r) ∝ exp(-tau_res * sensed_occ_r)``. Larger ``tau_res`` ⇒
    sharper avoidance (fewer same-bucket neighbours); ``tau_res=0`` ⇒ uniform (random) buckets — the
    zero-structure control. Deterministic given ``generator``. This is the sensing surrogate of spec
    §3.3, NOT a full 3GPP sensing window (Q4: lightweight, no ns-3).

    Args:
        positions: ``[N, 2]`` node coordinates.
        int_radius: interference radius (same as the scene's ``int_radius``); defines who contends.
        n_buckets: resource-pool size ``S`` (registry §2: {40, 60, 100}; should match the physics
            ``resource_pool`` for a consistent collision pool).
        tau_res: sensing-selection temperature (registry §2 default 4.0).
        sensing_noise_std: std of the additive Gaussian sensing noise (registry §2 default 0.1).

    Returns:
        ``[N]`` long bucket ids, ``requires_grad=False`` (buckets are constant structure).
    """
    if n_buckets < 1:
        raise ValueError("n_buckets (resource pool S) must be >= 1")
    N = positions.shape[0]
    gi = build_interference_graph(positions, int_radius)
    contenders: list[list[int]] = [[] for _ in range(N)]   # contenders[j] = interference nbrs of j
    for s_, d_ in zip(gi.src_index.tolist(), gi.dst_index.tolist()):
        contenders[d_].append(s_)
    bucket = torch.full((N,), -1, dtype=torch.long)
    order = torch.randperm(N, generator=generator).tolist()
    for v in order:
        occ = torch.zeros(n_buckets, dtype=dtype)
        for u in contenders[v]:
            bu = int(bucket[u])
            if bu >= 0:
                occ[bu] += 1.0
        if sensing_noise_std > 0:
            occ = occ + sensing_noise_std * torch.randn(n_buckets, generator=generator, dtype=dtype)
        p = torch.softmax(-float(tau_res) * occ, dim=0)
        bucket[v] = int(torch.multinomial(p, 1, generator=generator))
    return bucket.detach()


def same_resource_conflict_degree(
    bucket: torch.Tensor, positions: torch.Tensor, int_radius: float,
    *, dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Deployable proxy (spec §3.5): ``[N]`` count of same-bucket interference neighbours per node.

    ``deg_j = |{u in N_int(j): r_u = r_j}|`` — the persistent same-resource contention each node sees.
    Observable (buckets + geometry), no truth/future. This is the feature the GNN and the
    resource-aware heuristics read to reason about SPS conflict.
    """
    N = positions.shape[0]
    gi = build_interference_graph(positions, int_radius)
    same = (bucket[gi.src_index] == bucket[gi.dst_index]).to(dtype)     # [E_int]
    return scatter_destination(gi, same, N)


def sensed_channel_busy_ratio(
    positions: torch.Tensor, int_radius: float, *, dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Deployable proxy (spec §3.2): ``[N]`` sensed channel-busy ratio ≈ interference-neighbour count.

    A geometry-only surrogate for CBR (how crowded each node's interference neighbourhood is);
    observable, no truth/future. Distinct from the same-resource conflict degree (which needs buckets).
    """
    N = positions.shape[0]
    gi = build_interference_graph(positions, int_radius)
    return scatter_destination(gi, torch.ones(gi.num_edges, dtype=dtype), N)
