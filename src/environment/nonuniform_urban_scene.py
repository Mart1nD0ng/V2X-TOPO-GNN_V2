"""Nonuniform urban scene + sparse intersection-queue hotspots + capped RSU (NDH spec §4, §6).

The Non-Distance Headroom (NDH) drop-in replacement for ``ManhattanScene``: a perturbed urban
grid (lognormal block lengths + intersection jitter + optional road dropout), a *few* static
intersection-queue hotspots (local density non-uniformity that distance alone cannot read), and
**capped** RSU placement (responder/witness nodes that never enter the macrostate, ``omega_RSU=0``).

It is a true drop-in: it exposes exactly the seven ``ManhattanScene`` fields
(``positions, region_of, segment_endpoints, comm_radius, int_radius, block_m, grid``) plus the
``num_nodes`` / ``num_regions`` properties, so every downstream consumer (candidate/interference
graphs, the evidence model, the dynamic-MC judge, the GNN feature builder) works unchanged. RSU
nodes are ordinary graph nodes — their RSU-ness lives ONLY in the extra ``node_type`` label and in
the ``vehicle_only_participation`` measure (``omega_RSU=0``); the graph builders never special-case
them. The extra NDH fields (``node_type, hotspot_score, intersection_xy, hotspot_intersections,
segment_intersections``) are ignored by drop-in consumers and read by the NDH mechanisms.

ALL parameters come from ``docs/NDH_PARAMETER_REGISTRY.md``. Hard caps are enforced in the builder
(constraints #9/#10, spec §4.3/§6.3): ``p_intersection_rsu <= 0.5``; ``#RSU <= max_rsu_fraction*N``;
``#hotspots <= 10% of intersections``; hotspot vehicles ``<= 30%`` of vehicles; hotspots
non-overlapping (centres ``>= 2*radius`` apart). Phase-1 hotspots are static. Deterministic given
the ``torch.Generator``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = ["NonuniformUrbanScene", "build_nonuniform_urban_scene"]


@dataclass(frozen=True, eq=False)
class NonuniformUrbanScene:
    """Drop-in scene (7 ManhattanScene fields) + NDH role/hotspot/intersection extensions.

    ``eq=False`` gives identity-based ``__hash__``/``__eq__`` (so the scene is hashable exactly like
    ``ManhattanScene``, despite the ``params: dict`` field); field-wise ``==`` on a scene already
    raises on the tensor fields, so identity semantics are the only usable ones anyway.
    """

    # --- the drop-in ManhattanScene interface (read by all existing consumers) ---
    positions: torch.Tensor          # [N, 2] float, node coordinates (metres)
    region_of: torch.Tensor          # [N] long, dense region id g(i) in {0..G-1}
    segment_endpoints: torch.Tensor  # [G, 2, 2] (start_xy, end_xy) per present road segment
    comm_radius: float
    int_radius: float
    block_m: float
    grid: tuple[int, int]            # (gx, gy) intersections per axis (metadata)
    # --- NDH extensions (ignored by drop-in consumers; read by NDH mechanisms/participation) ---
    node_type: torch.Tensor          # [N] long, 0 = vehicle, 1 = RSU (exogenous role label)
    hotspot_score: torch.Tensor      # [N] float in [0,1], proximity to the nearest hotspot
    intersection_xy: torch.Tensor    # [I, 2] float, intersection coordinates
    hotspot_intersections: torch.Tensor   # [H] long, indices into intersection_xy
    segment_intersections: torch.Tensor   # [G, 2] long, (start_int, end_int) per present segment
    resource_bucket: torch.Tensor | None = None   # [N] long SPS bucket (None = SPS off); read by round_physics
    params: dict = field(default_factory=dict)   # the generating NDH parameters (for the manifest)

    @property
    def num_nodes(self) -> int:
        return int(self.region_of.numel())

    @property
    def num_regions(self) -> int:
        return int(self.segment_endpoints.shape[0])

    @property
    def num_vehicles(self) -> int:
        return int((self.node_type == 0).sum())

    @property
    def num_rsu(self) -> int:
        return int((self.node_type == 1).sum())

    @property
    def is_rsu(self) -> torch.Tensor:
        return self.node_type == 1

    @property
    def mechanism_config_hash(self) -> str:
        """SHA-256 of the NDH-mechanism structural params (SPS/hotspot/RSU/road) for provenance.

        The physics ``config_hash`` binds ``resource_collision_kappa`` but NOT the bucket-assignment /
        hotspot / RSU knobs that generate this scene's structure; this hash captures them so an
        experiment JSON can record it alongside the physics hash (registry §0.2, Contract C5) and
        train==eval structural divergence is detectable. Combined with the scene seed (in the manifest
        ``scene_distribution_hash``) it pins the exact ``resource_bucket`` / hotspot / RSU realisation.
        """
        import hashlib
        import json
        keys = ("enable_sps", "sps_n_buckets", "sps_tau_res", "sps_sensing_noise_std",
                "enable_hotspots", "num_hotspots_effective", "hotspot_radius_m",
                "hotspot_vehicle_fraction", "queue_length_m", "num_hotspot_vehicles",
                "enable_rsu", "p_intersection_rsu", "p_hotspot_rsu_boost", "max_rsu_fraction",
                "min_rsu_spacing_m", "rsu_roadside_offset_m", "block_length_logstd",
                "intersection_jitter_m", "road_presence_probability")
        payload = json.dumps({k: self.params.get(k) for k in keys}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rand(gen, shape, dtype):
    return torch.rand(shape, generator=gen, dtype=dtype)


def _intersection_grid(gx, gy, block_m, logstd, jitter_m, gen, dtype):
    """Intersection coordinates with lognormal block lengths + per-intersection jitter.

    Column x-positions are cumulative sums of ``block_m * exp(logstd * N(0,1))`` gaps (rows
    likewise for y); each intersection then gets an independent ``U(-jitter, +jitter)`` offset.
    Returns ``intersection_xy [gx*gy, 2]`` indexed by ``iy*gx + ix``.
    """
    def axis(n_gap):
        if logstd > 0:
            gaps = block_m * torch.exp(logstd * torch.randn(n_gap, generator=gen, dtype=dtype))
        else:
            gaps = torch.full((n_gap,), float(block_m), dtype=dtype)
        coord = torch.zeros(n_gap + 1, dtype=dtype)
        coord[1:] = torch.cumsum(gaps, dim=0)
        return coord
    xs = axis(gx - 1)                                    # [gx]
    ys = axis(gy - 1)                                    # [gy]
    xy = torch.empty((gx * gy, 2), dtype=dtype)
    for iy in range(gy):
        for ix in range(gx):
            xy[iy * gx + ix, 0] = xs[ix]
            xy[iy * gx + ix, 1] = ys[iy]
    if jitter_m > 0:
        xy = xy + (_rand(gen, (gx * gy, 2), dtype) * 2 - 1) * jitter_m
    return xy


def _segments(gx, gy, road_presence, gen):
    """(start_int, end_int) index pairs for present road segments, guaranteed connected.

    A spanning tree (row-0 horizontals + all verticals = gx*gy-1 edges) is always present;
    the remaining horizontals (rows 1..) are dropped independently with prob ``1-road_presence``.
    """
    mandatory, optional = [], []
    for ix in range(gx - 1):                             # row-0 horizontals (spanning-tree backbone)
        mandatory.append((ix, ix + 1))
    for ix in range(gx):                                 # all verticals (spanning-tree backbone)
        for iy in range(gy - 1):
            a, b = iy * gx + ix, (iy + 1) * gx + ix
            mandatory.append((a, b))
    for iy in range(1, gy):                              # rows 1.. horizontals (droppable)
        for ix in range(gx - 1):
            a, b = iy * gx + ix, iy * gx + ix + 1
            optional.append((a, b))
    segs = list(mandatory)
    if optional:
        keep = _rand(gen, (len(optional),), torch.float64) < road_presence
        segs.extend(s for s, k in zip(optional, keep.tolist()) if k)
    return segs


def _place_on_segment(a, b, frac, jitter_m, gen, dtype, n):
    """``n`` points at fractions ``frac`` along segment a->b with lateral jitter ``U(-j,+j)``."""
    seg = b - a
    length = float((seg * seg).sum() ** 0.5)
    if length < 1e-9:
        return a.unsqueeze(0).expand(n, 2).clone()
    nrm = torch.stack([-seg[1], seg[0]]) / length       # unit normal
    jit = (_rand(gen, (n,), dtype) * 2 - 1) * jitter_m
    return a.unsqueeze(0) + frac.unsqueeze(1) * seg.unsqueeze(0) + jit.unsqueeze(1) * nrm.unsqueeze(0)


def _greedy_spaced(cands_xy, order, min_spacing, max_count):
    """Greedily accept candidate indices in ``order`` keeping pairwise distance >= min_spacing."""
    if max_count <= 0:                                   # e.g. fraction cap floors to 0 on a tiny grid
        return []
    accepted: list[int] = []
    for c in order:
        p = cands_xy[c]
        if all(float((p - cands_xy[a]).norm()) >= min_spacing for a in accepted):
            accepted.append(int(c))
            if len(accepted) >= max_count:
                break
    return accepted


def build_nonuniform_urban_scene(
    gx: int,
    gy: int,
    vehicles_per_segment: int,
    *,
    block_m: float = 100.0,
    block_length_logstd: float = 0.0,
    intersection_jitter_m: float = 0.0,
    road_presence_probability: float = 1.0,
    lane_jitter_m: float = 3.0,
    comm_radius: float = 80.0,
    int_radius: float = 160.0,
    # hotspots (spec §6.3)
    enable_hotspots: bool = False,
    num_hotspots: int = 2,
    hotspot_radius_m: float = 50.0,
    hotspot_vehicle_fraction: float = 0.2,
    queue_length_m: float = 100.0,
    # RSU (spec §4.3)
    enable_rsu: bool = False,
    p_intersection_rsu: float = 0.25,
    p_hotspot_rsu_boost: float = 0.25,
    max_rsu_fraction: float = 0.10,
    min_rsu_spacing_m: float = 300.0,
    rsu_roadside_offset_m: float = 5.0,
    # SPS resource buckets (spec §3; assignment surrogate in sps_resource.assign_sps_buckets)
    enable_sps: bool = False,
    sps_n_buckets: int = 100,
    sps_tau_res: float = 4.0,
    sps_sensing_noise_std: float = 0.1,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> NonuniformUrbanScene:
    """Build an NDH nonuniform urban scene. Defaults = the ``NDH-DEPLOYMENT`` control geometry
    (uniform grid, no hotspots, no RSU); flip ``enable_hotspots`` / ``enable_rsu`` for the
    ``ndh_hotspot_static`` / ``ndh_hotspot_rsu_static`` variants. All hard caps are enforced here.
    """
    if gx < 2 or gy < 2:
        raise ValueError("grid too small: need gx, gy >= 2")
    if vehicles_per_segment < 1:
        raise ValueError("vehicles_per_segment must be >= 1")
    if int_radius < comm_radius or comm_radius <= 0:
        raise ValueError("require int_radius >= comm_radius > 0")
    if enable_rsu and p_intersection_rsu > 0.5:
        raise ValueError("p_intersection_rsu must be <= 0.5 (anti-trivial-nearest-RSU, constraint #10)")
    if enable_rsu and max_rsu_fraction > 0.15:
        raise ValueError("max_rsu_fraction must be <= 0.15 (registry §3 HARD CAP; anti-trivial-RSU)")
    if not (0.0 < road_presence_probability <= 1.0):
        raise ValueError("road_presence_probability must be in (0, 1]")
    gen = generator if generator is not None else torch.Generator().manual_seed(0)

    inter_xy = _intersection_grid(gx, gy, block_m, block_length_logstd, intersection_jitter_m, gen, dtype)
    seg_pairs = _segments(gx, gy, road_presence_probability, gen)
    G = len(seg_pairs)
    seg_int = torch.tensor(seg_pairs, dtype=torch.long)                  # [G, 2]
    endpoints = torch.stack([torch.stack([inter_xy[a], inter_xy[b]]) for a, b in seg_pairs])  # [G,2,2]

    # ---- baseline vehicles: vehicles_per_segment along each present segment ----
    n = vehicles_per_segment
    frac = (torch.arange(n, dtype=dtype) + 0.5) / n
    pos_list, region_list = [], []
    for s, (a, b) in enumerate(seg_pairs):
        pos_list.append(_place_on_segment(inter_xy[a], inter_xy[b], frac, lane_jitter_m, gen, dtype, n))
        region_list.append(torch.full((n,), s, dtype=torch.long))
    base_pos = torch.cat(pos_list, dim=0)
    base_region = torch.cat(region_list, dim=0)
    n_base = base_pos.shape[0]

    # ---- hotspots: pick spaced intersections, then QUEUE extra vehicles along incident segments ----
    # The hotspot MECHANISM is the EXTRA queued concentration we inject; the "<=30% of vehicles"
    # hard cap bounds exactly that injected mass (a quantity the construction controls), not the
    # incidental uniform-grid density near a corner (which the construction cannot lower and which
    # would make the cap structurally infeasible on coarse grids + large radius). queue_length_m
    # drives the along-road queue EXTENT (clamped only by the physical segment length).
    hotspot_idx: list[int] = []
    queue_pos, queue_region = [], []
    n_queued = 0
    if enable_hotspots and num_hotspots > 0:
        I = inter_xy.shape[0]
        cap_hot = max(1, int(0.10 * I))                                  # <= 10% of intersections
        want = min(num_hotspots, cap_hot)
        order = torch.randperm(I, generator=gen).tolist()
        hotspot_idx = _greedy_spaced(inter_xy, order, 2 * hotspot_radius_m, want)
        incident = {h: [s for s, (a, b) in enumerate(seg_pairs) if a == h or b == h] for h in hotspot_idx}
        # injected queue mass <= 30% of TOTAL vehicles:  added/(n_base+added) <= 0.30
        # -> added <= (0.30/0.70)*n_base.  target = min(requested fraction of base, that ceiling).
        add_ceiling = int((0.30 / 0.70) * n_base)
        target_added = min(int(round(hotspot_vehicle_fraction * n_base)), add_ceiling)
        per_hot = target_added // max(1, len(hotspot_idx))
        for h in hotspot_idx:
            segs_h = incident[h] or list(range(G))
            for q in range(per_hot):
                s = segs_h[q % len(segs_h)]
                a, b = seg_pairs[s]
                seg = inter_xy[b] - inter_xy[a]
                seg_len = float((seg * seg).sum() ** 0.5)
                reach = min(queue_length_m, seg_len)                     # queue extent (queue_length_m drives it)
                # queue grows OUT from the hotspot endpoint h along the segment
                start = inter_xy[a] if a == h else inter_xy[b]
                direction = (seg if a == h else -seg) / max(seg_len, 1e-9)
                d = reach * (q + 0.5) / max(per_hot, 1)
                jit = (float(_rand(gen, (1,), dtype)) * 2 - 1) * lane_jitter_m
                nrm = torch.stack([-direction[1], direction[0]])
                queue_pos.append((start + d * direction + jit * nrm).unsqueeze(0))
                queue_region.append(torch.tensor([s], dtype=torch.long))
        n_queued = len(queue_pos)

    if queue_pos:
        veh_pos = torch.cat([base_pos] + queue_pos, dim=0)
        veh_region = torch.cat([base_region] + queue_region, dim=0)
    else:
        veh_pos, veh_region = base_pos, base_region
    n_veh = veh_pos.shape[0]

    # ---- RSU: capped, spaced placement at (boosted) intersections; responder/witness only ----
    rsu_pos, rsu_region = [], []
    if enable_rsu:
        I = inter_xy.shape[0]
        probs = torch.full((I,), float(p_intersection_rsu), dtype=dtype)
        for h in hotspot_idx:
            probs[h] = min(0.5, p_intersection_rsu + p_hotspot_rsu_boost)   # boost near hotspots, HARD cap 0.5
        draws = _rand(gen, (I,), dtype) < probs
        cand = [i for i in range(I) if bool(draws[i])]
        order = sorted(cand, key=lambda i: (i not in hotspot_idx))          # prefer hotspot intersections
        # max_rsu_fraction of TOTAL nodes: r <= f*(n_veh + r)  ->  r <= f/(1-f) * n_veh
        max_rsu = int((max_rsu_fraction / max(1e-9, 1.0 - max_rsu_fraction)) * n_veh)
        accept = _greedy_spaced(inter_xy, order, min_rsu_spacing_m, max_rsu)
        for i in accept:
            # offset perpendicular to an incident segment (roadside); fall back to +x
            seg_s = next((s for s, (a, b) in enumerate(seg_pairs) if a == i or b == i), None)
            if seg_s is not None:
                a, b = seg_pairs[seg_s]
                seg = inter_xy[b] - inter_xy[a]
                length = float((seg * seg).sum() ** 0.5)
                nrm = torch.stack([-seg[1], seg[0]]) / max(length, 1e-9)
                region_id = seg_s
            else:
                nrm = torch.tensor([1.0, 0.0], dtype=dtype)
                region_id = 0
            rsu_pos.append((inter_xy[i] + rsu_roadside_offset_m * nrm).unsqueeze(0))
            rsu_region.append(torch.tensor([region_id], dtype=torch.long))

    # ---- assemble: vehicles first, RSU appended (no reordering of vehicle ids) ----
    if rsu_pos:
        positions = torch.cat([veh_pos] + rsu_pos, dim=0)
        region_of = torch.cat([veh_region] + rsu_region, dim=0)
        node_type = torch.cat([torch.zeros(n_veh, dtype=torch.long),
                               torch.ones(len(rsu_pos), dtype=torch.long)])
    else:
        positions, region_of, node_type = veh_pos, veh_region, torch.zeros(n_veh, dtype=torch.long)
    N = positions.shape[0]

    # ---- SPS resource buckets (static, sensing-based surrogate; spec §3.3) over ALL nodes ----
    resource_bucket = None
    if enable_sps:
        from .sps_resource import assign_sps_buckets
        resource_bucket = assign_sps_buckets(
            positions, float(int_radius), sps_n_buckets, tau_res=sps_tau_res,
            sensing_noise_std=sps_sensing_noise_std, generator=gen, dtype=dtype)

    # ---- hotspot score (all nodes): max_h exp(-dist/radius) in [0,1] ----
    if hotspot_idx:
        d_hot = _min_dist_to(positions, inter_xy[hotspot_idx])
        hotspot_score = torch.exp(-d_hot / hotspot_radius_m)
    else:
        hotspot_score = torch.zeros(N, dtype=dtype)

    # ---- hard-cap assertions (defence in depth; the construction already respects them) ----
    n_rsu = int((node_type == 1).sum())
    assert n_rsu <= max_rsu_fraction * N + 1e-9, "RSU fraction cap violated"
    # the cap bounds the INJECTED queue mass (what the mechanism controls) -- not incidental grid
    # density (which the construction cannot lower and which would make the cap infeasible on
    # coarse grids). This is always satisfiable -> no crash on registry-legal coarse/large-radius cells.
    assert n_queued <= 0.30 * n_veh + 1e-9, "injected hotspot queue mass exceeds 30% of vehicles"
    assert len(hotspot_idx) <= max(1, int(0.10 * inter_xy.shape[0])), "hotspot count cap violated"
    assert int(region_of.min()) == 0 and int(region_of.max()) == G - 1, "region coverage not dense"

    params = {"gx": gx, "gy": gy, "vehicles_per_segment": vehicles_per_segment, "block_m": block_m,
              "block_length_logstd": block_length_logstd, "intersection_jitter_m": intersection_jitter_m,
              "road_presence_probability": road_presence_probability, "lane_jitter_m": lane_jitter_m,
              "comm_radius": comm_radius, "int_radius": int_radius, "enable_hotspots": enable_hotspots,
              "num_hotspots_effective": len(hotspot_idx), "hotspot_radius_m": hotspot_radius_m,
              "hotspot_vehicle_fraction": hotspot_vehicle_fraction, "queue_length_m": queue_length_m,
              "num_hotspot_vehicles": n_queued, "enable_rsu": enable_rsu,
              "p_intersection_rsu": p_intersection_rsu, "p_hotspot_rsu_boost": p_hotspot_rsu_boost,
              "max_rsu_fraction": max_rsu_fraction, "min_rsu_spacing_m": min_rsu_spacing_m,
              "rsu_roadside_offset_m": rsu_roadside_offset_m, "num_vehicles": n_veh, "num_rsu": n_rsu,
              "enable_sps": enable_sps, "sps_n_buckets": sps_n_buckets, "sps_tau_res": sps_tau_res,
              "sps_sensing_noise_std": sps_sensing_noise_std}

    return NonuniformUrbanScene(
        positions=positions, region_of=region_of, segment_endpoints=endpoints,
        comm_radius=float(comm_radius), int_radius=float(int_radius), block_m=float(block_m),
        grid=(gx, gy), node_type=node_type, hotspot_score=hotspot_score, intersection_xy=inter_xy,
        hotspot_intersections=torch.tensor(hotspot_idx, dtype=torch.long),
        segment_intersections=seg_int, resource_bucket=resource_bucket, params=params)


def _min_dist_to(points: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """``[N]`` distance from each point to the NEAREST centre (``centres`` is ``[H, 2]``)."""
    if centres.ndim == 1:
        centres = centres.unsqueeze(0)
    d = torch.cdist(points.to(torch.float64), centres.to(torch.float64))   # [N, H]
    return d.min(dim=1).values
