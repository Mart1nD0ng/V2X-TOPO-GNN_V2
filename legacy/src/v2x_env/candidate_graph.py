from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np

from .channel_model import ChannelConfig, density_coupled_interference_dbm, edge_success_probability


@dataclass(frozen=True)
class CandidateGraphConfig:
    radius_m: float = 180.0
    max_candidates_per_node: int = 12
    cell_size_m: float | None = None
    prefer_los: bool = True
    # Density-aware interference (realism): when > 0, a receiver's interference floor
    # rises with its local feasible degree, so denser deployments are congested
    # (not trivially easy). 0.0 preserves the legacy fixed-proxy behaviour.
    interference_density_coupling_db: float = 0.0
    interference_reference_degree: float = 8.0
    # P1-1.4: also emit a no-load (fixed-proxy interference) success column so the model FEATURE side
    # can be made consistent with the evaluator's in-load definition. Default False -> byte-identical.
    emit_noload_success: bool = False
    # P1-1.1 LOS model. "road_segment" (default, byte-identical) = the same-road-segment-block rule
    # in _same_road_segment, which mislabels ~30% of physically-visible collinear cross-block edges
    # as NLOS and forces every RSU edge LOS. "axis_visibility" = a geometric rule: LOS iff the two
    # nodes share a straight road line (same x-road or same y-road) OR are both within
    # los_intersection_radius_m of a shared intersection; RSUs get the SAME geometric test (no longer
    # unconditionally LOS). Requires snapshot['grid'] (the UrbanGrid).
    los_model: str = "road_segment"
    los_roadline_tol_m: float | None = None        # None -> grid.config.road_half_width_m
    los_intersection_radius_m: float = 30.0        # cross-road LOS when both near a shared intersection
    # When True, emit LOS-artifact forensics into diagnostics (nlos_but_collinear_same_roadline_fraction,
    # nlos_under_30m_fraction). Default False keeps the diagnostics dict byte-identical for legacy callers.
    emit_los_forensics: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "CandidateGraphConfig":
        data = data or {}
        radius = float(data.get("radius_m", data.get("candidate_radius_m", cls.radius_m)))
        raw_cell = data.get("cell_size_m", None)
        raw_tol = data.get("los_roadline_tol_m", None)
        return cls(
            radius_m=radius,
            max_candidates_per_node=int(data.get("max_candidates_per_node", cls.max_candidates_per_node)),
            cell_size_m=float(raw_cell) if raw_cell is not None else radius,
            prefer_los=bool(data.get("prefer_los", cls.prefer_los)),
            interference_density_coupling_db=float(
                data.get("interference_density_coupling_db", cls.interference_density_coupling_db)
            ),
            interference_reference_degree=float(
                data.get("interference_reference_degree", cls.interference_reference_degree)
            ),
            emit_noload_success=bool(data.get("emit_noload_success", cls.emit_noload_success)),
            los_model=str(data.get("los_model", cls.los_model)),
            los_roadline_tol_m=float(raw_tol) if raw_tol is not None else cls.los_roadline_tol_m,
            los_intersection_radius_m=float(
                data.get("los_intersection_radius_m", cls.los_intersection_radius_m)
            ),
            emit_los_forensics=bool(data.get("emit_los_forensics", cls.emit_los_forensics)),
        )


@dataclass(frozen=True)
class CandidateGraph:
    num_nodes: int
    source: np.ndarray
    target: np.ndarray
    distance_m: np.ndarray
    los_flag: np.ndarray
    success_probability: np.ndarray
    channel_score: np.ndarray
    sinr_db: np.ndarray
    diagnostics: dict[str, float | int]
    # P1-1.4 interference consistency (opt-in): the no-load (fixed-proxy) success, so a model FEATURE
    # can be reconciled with the evaluator's selected-in-load success definition (the feature side
    # currently uses feasibility-degree interference, a DIFFERENT load definition than the evaluator).
    # None unless CandidateGraphConfig.emit_noload_success is set -> default callers byte-identical.
    noload_success_probability: np.ndarray | None = None

    @property
    def edge_count(self) -> int:
        return int(self.source.size)

    def outgoing_counts(self) -> np.ndarray:
        if self.num_nodes == 0:
            return np.array([], dtype=int)
        return np.bincount(self.source, minlength=self.num_nodes)


def _cell_key(x_value: float, y_value: float, cell_size_m: float) -> tuple[int, int]:
    return (int(np.floor(x_value / cell_size_m)), int(np.floor(y_value / cell_size_m)))


def _same_road_segment(snapshot: Mapping[str, object], a_idx: int, b_idx: int) -> bool:
    segment_ids = np.asarray(snapshot["road_segment_id"])
    node_types = np.asarray(snapshot["node_type"], dtype=object)
    if node_types[a_idx] == "rsu" or node_types[b_idx] == "rsu":
        return True
    return int(segment_ids[a_idx]) >= 0 and int(segment_ids[a_idx]) == int(segment_ids[b_idx])


def _roadline_membership(
    x: np.ndarray, y: np.ndarray, grid, tol: float
) -> dict[str, np.ndarray]:
    """Per-node nearest road-line indices + on-line masks + distance-to-nearest-intersection.

    O(N*R) (R = number of roads, small): computed ONCE before the neighbour loop so the per-pair
    axis-visibility predicate is O(1). A node is 'on' a vertical (horizontal) road line when its
    perpendicular distance to the nearest x-road (y-road) is within ``tol`` (= road_half_width by
    default, the maximum lane offset, so generated vehicles are exactly on-line)."""
    x_roads = np.asarray(grid.x_roads_m, dtype=float)
    y_roads = np.asarray(grid.y_roads_m, dtype=float)
    dx = np.abs(x[:, None] - x_roads[None, :])
    dy = np.abs(y[:, None] - y_roads[None, :])
    vline_idx = dx.argmin(axis=1)
    hline_idx = dy.argmin(axis=1)
    on_v = dx.min(axis=1) <= tol
    on_h = dy.min(axis=1) <= tol
    # Nearest grid intersection for each node and the node's distance to it (for the cross-road
    # near-intersection LOS clause).
    int_x = x_roads[vline_idx]
    int_y = y_roads[hline_idx]
    dist_to_int = np.sqrt((x - int_x) ** 2 + (y - int_y) ** 2)
    return {
        "vline_idx": vline_idx, "hline_idx": hline_idx,
        "on_v": on_v, "on_h": on_h, "dist_to_int": dist_to_int,
    }


def _axis_visibility_los(a_idx: int, b_idx: int, rl: dict[str, np.ndarray], intersection_radius_m: float) -> bool:
    """Geometric (axis-aligned) line-of-sight: LOS iff the two nodes share a straight road line OR
    are both within ``intersection_radius_m`` of a shared intersection. RSUs (which sit exactly on an
    intersection -> on both a vertical and a horizontal line) are subject to the SAME test, so they are
    LOS to anything on either of their crossing lines but not unconditionally to everything."""
    vidx, hidx = rl["vline_idx"], rl["hline_idx"]
    on_v, on_h, dint = rl["on_v"], rl["on_h"], rl["dist_to_int"]
    # same vertical road line (collinear down one street)
    if on_v[a_idx] and on_v[b_idx] and vidx[a_idx] == vidx[b_idx]:
        return True
    # same horizontal road line
    if on_h[a_idx] and on_h[b_idx] and hidx[a_idx] == hidx[b_idx]:
        return True
    # cross-road pair both near the SAME open intersection (straight line passes through the crossing)
    if (
        vidx[a_idx] == vidx[b_idx]
        and hidx[a_idx] == hidx[b_idx]
        and dint[a_idx] <= intersection_radius_m
        and dint[b_idx] <= intersection_radius_m
    ):
        return True
    return False


def build_candidate_graph(
    snapshot: Mapping[str, object],
    channel_config: ChannelConfig | Mapping[str, object] | None = None,
    candidate_config: CandidateGraphConfig | Mapping[str, object] | None = None,
) -> CandidateGraph:
    start_time = perf_counter()
    cfg = candidate_config if isinstance(candidate_config, CandidateGraphConfig) else CandidateGraphConfig.from_mapping(candidate_config)
    ch_cfg = channel_config if isinstance(channel_config, ChannelConfig) else ChannelConfig.from_mapping(channel_config)
    if cfg.radius_m <= 0:
        raise ValueError("candidate radius must be positive")
    if cfg.max_candidates_per_node <= 0:
        raise ValueError("max_candidates_per_node must be positive")
    if cfg.los_model not in {"road_segment", "axis_visibility"}:
        raise ValueError("los_model must be 'road_segment' or 'axis_visibility'")

    x = np.asarray(snapshot["x"], dtype=float)
    y = np.asarray(snapshot["y"], dtype=float)
    count = int(x.size)
    # P1-1.1: geometric LOS / LOS-forensics need the UrbanGrid road lines. Only the default
    # 'road_segment' path touches NEITHER, so byte-identity for hand-built (grid-less) snapshots holds.
    roadline = None
    if cfg.los_model == "axis_visibility" or cfg.emit_los_forensics:
        grid = snapshot.get("grid")
        if grid is None:
            raise ValueError(
                "los_model='axis_visibility' / emit_los_forensics require snapshot['grid'] "
                "(an UrbanGrid with x_roads_m / y_roads_m)"
            )
        tol = float(cfg.los_roadline_tol_m) if cfg.los_roadline_tol_m is not None else float(grid.config.road_half_width_m)
        tol = tol + 1e-6  # inclusive of the exact max lane offset against float wrap
        roadline = _roadline_membership(x, y, grid, tol)
    cell_size = float(cfg.cell_size_m or cfg.radius_m)
    bins: dict[tuple[int, int], list[int]] = defaultdict(list)
    for node_idx, (x_value, y_value) in enumerate(zip(x, y)):
        bins[_cell_key(float(x_value), float(y_value), cell_size)].append(node_idx)
    occupancies = [len(bucket) for bucket in bins.values()]

    src_values: list[int] = []
    dst_values: list[int] = []
    distance_values: list[float] = []
    los_values: list[bool] = []
    pair_checks = 0
    cap_hit_count = 0

    feasible_degree = np.zeros(count, dtype=float)  # local density proxy (neighbours within radius)
    neighbor_span = int(np.ceil(cfg.radius_m / cell_size))
    for node_idx, (x_value, y_value) in enumerate(zip(x, y)):
        base_cell = _cell_key(float(x_value), float(y_value), cell_size)
        local: list[tuple[float, float, int, bool]] = []
        for dx_cell in range(-neighbor_span, neighbor_span + 1):
            for dy_cell in range(-neighbor_span, neighbor_span + 1):
                bucket = bins.get((base_cell[0] + dx_cell, base_cell[1] + dy_cell), ())
                for other_idx in bucket:
                    if other_idx == node_idx:
                        continue
                    pair_checks += 1
                    dx = float(x[other_idx] - x_value)
                    dy = float(y[other_idx] - y_value)
                    distance = float((dx * dx + dy * dy) ** 0.5)
                    if distance <= cfg.radius_m:
                        if cfg.los_model == "axis_visibility":
                            los = _axis_visibility_los(node_idx, other_idx, roadline, cfg.los_intersection_radius_m)
                        else:
                            los = _same_road_segment(snapshot, node_idx, other_idx)
                        local.append((distance, 0.0 if los else 1.0, other_idx, los))
        feasible_degree[node_idx] = float(len(local))
        if not local:
            continue
        if len(local) > cfg.max_candidates_per_node:
            cap_hit_count += 1
        local.sort(key=lambda item: (item[1], item[0]) if cfg.prefer_los else (item[0], item[1]))
        for distance, _los_rank, other_idx, los in local[: cfg.max_candidates_per_node]:
            src_values.append(node_idx)
            dst_values.append(other_idx)
            distance_values.append(distance)
            los_values.append(los)

    source = np.asarray(src_values, dtype=int)
    target = np.asarray(dst_values, dtype=int)
    distance_m = np.asarray(distance_values, dtype=float)
    los_flag = np.asarray(los_values, dtype=bool)
    if distance_m.size:
        if cfg.interference_density_coupling_db > 0.0:
            interference = density_coupled_interference_dbm(
                ch_cfg.interference_proxy_dbm, feasible_degree[target],
                cfg.interference_reference_degree, cfg.interference_density_coupling_db,
            )
        else:
            interference = None
        success, sinr_values = edge_success_probability(distance_m, los_flag, ch_cfg, interference)
        score = success - 0.05 * (distance_m / cfg.radius_m) + np.where(los_flag, 0.02, 0.0)
        if cfg.emit_noload_success:
            # No-load success = fixed-proxy interference (interference=None), regardless of coupling.
            noload_success, _ = edge_success_probability(distance_m, los_flag, ch_cfg, None)
        else:
            noload_success = None
    else:
        success = np.asarray([], dtype=float)
        sinr_values = np.asarray([], dtype=float)
        score = np.asarray([], dtype=float)
        noload_success = np.asarray([], dtype=float) if cfg.emit_noload_success else None
    elapsed_ms = (perf_counter() - start_time) * 1000.0
    diagnostics = {
        "max_cell_occupancy": int(max(occupancies)) if occupancies else 0,
        "mean_cell_occupancy": float(np.mean(occupancies)) if occupancies else 0.0,
        "candidate_pair_checks_before_radius": int(pair_checks),
        "candidate_pair_checks_per_node": float(pair_checks / max(count, 1)),
        "cap_hit_count": int(cap_hit_count),
        "cap_hit_ratio": float(cap_hit_count / max(count, 1)),
        "build_wall_time_ms": float(elapsed_ms),
    }
    # P1-1.1 LOS forensics (opt-in -> default diagnostics dict stays byte-identical). Quantifies how
    # many NLOS edges are actually collinear-same-roadline (the same-segment-rule artifact).
    if cfg.emit_los_forensics and roadline is not None and source.size:
        nlos = ~los_flag
        vidx, hidx = roadline["vline_idx"], roadline["hline_idx"]
        on_v, on_h = roadline["on_v"], roadline["on_h"]
        collinear = (
            (on_v[source] & on_v[target] & (vidx[source] == vidx[target]))
            | (on_h[source] & on_h[target] & (hidx[source] == hidx[target]))
        )
        n_nlos = int(nlos.sum())
        diagnostics["nlos_but_collinear_same_roadline_fraction"] = float((nlos & collinear).sum() / max(n_nlos, 1))
        diagnostics["nlos_under_30m_fraction"] = float((nlos & (distance_m < 30.0)).sum() / max(source.size, 1))
    return CandidateGraph(
        num_nodes=count,
        source=source,
        target=target,
        distance_m=distance_m,
        los_flag=los_flag,
        success_probability=success,
        channel_score=score,
        sinr_db=sinr_values,
        diagnostics=diagnostics,
        noload_success_probability=noload_success,
    )
