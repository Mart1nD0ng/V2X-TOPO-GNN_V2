from __future__ import annotations

from typing import Mapping

import numpy as np

from .urban_grid import UrbanGrid, make_urban_grid, road_segment_id_for_position


def _vehicle_count(config: Mapping[str, object], grid: UrbanGrid) -> int:
    if "vehicle_count" in config:
        return int(config["vehicle_count"])
    if "explicit_vehicle_count" in config:
        return int(config["explicit_vehicle_count"])
    density = float(config.get("vehicle_density_per_km2", 500.0))
    area_km2 = max(grid.config.width_m * grid.config.height_m / 1_000_000.0, 0.001)
    return max(1, int(round(density * area_km2)))


def _resolved_speed_config(config: Mapping[str, object]) -> dict[str, float]:
    """The speed distribution config with defaults resolved. Stored INTO the snapshot so that
    stateful mobility (P1-1.3 churn births) draws injected vehicles from the SAME distribution
    as the initial fleet — previously births silently fell back to the defaults (13.0/2.5)
    because the snapshot only carried the per-vehicle 'speed_mps' array, not the config."""
    speed_cfg = config.get("speed", {})
    if not isinstance(speed_cfg, Mapping):
        speed_cfg = {}
    return {
        "mean_mps": float(speed_cfg.get("mean_mps", 13.0)),
        "std_mps": float(speed_cfg.get("std_mps", 2.5)),
        "min_mps": float(speed_cfg.get("min_mps", 0.0)),
        "max_mps": float(speed_cfg.get("max_mps", 25.0)),
    }


def _speed_values(config: Mapping[str, object], count: int, rng: np.random.Generator) -> np.ndarray:
    resolved = _resolved_speed_config(config)
    return np.clip(
        rng.normal(resolved["mean_mps"], resolved["std_mps"], count),
        resolved["min_mps"], resolved["max_mps"],
    )


def _lane_offsets(lane_ids: np.ndarray, lanes_per_direction: int, lane_width_m: float) -> np.ndarray:
    center = lanes_per_direction - 0.5
    return (lane_ids.astype(float) - center) * lane_width_m


def _rsu_positions(config: Mapping[str, object], grid: UrbanGrid) -> np.ndarray:
    rsu_cfg = config.get("rsu", {})
    if not isinstance(rsu_cfg, Mapping) or not bool(rsu_cfg.get("enabled", False)):
        return np.empty((0, 2), dtype=float)
    stride = max(1, int(rsu_cfg.get("every_n_intersections", 2)))
    positions: list[tuple[float, float]] = []
    for ix, x in enumerate(grid.x_roads_m):
        for iy, y in enumerate(grid.y_roads_m):
            if ix % stride == 0 and iy % stride == 0:
                positions.append((float(x), float(y)))
    return np.array(positions, dtype=float) if positions else np.empty((0, 2), dtype=float)


def generate_vehicle_snapshot(config: Mapping[str, object]) -> dict[str, object]:
    grid = make_urban_grid(config.get("grid", {}))
    seed = int(config.get("seed", 0))
    rng = np.random.default_rng(seed)
    vehicle_count = _vehicle_count(config, grid)
    lane_choices = 2 * grid.config.lanes_per_direction

    orientation_is_vertical = rng.random(vehicle_count) < 0.5
    road_indices = np.empty(vehicle_count, dtype=int)
    lane_ids = rng.integers(0, lane_choices, size=vehicle_count)
    x = np.empty(vehicle_count, dtype=float)
    y = np.empty(vehicle_count, dtype=float)
    heading = np.empty(vehicle_count, dtype=float)
    road_segment_id = np.empty(vehicle_count, dtype=int)

    vertical_positions = np.flatnonzero(orientation_is_vertical)
    horizontal_positions = np.flatnonzero(~orientation_is_vertical)
    lane_offsets = _lane_offsets(lane_ids, grid.config.lanes_per_direction, grid.config.lane_width_m)

    if vertical_positions.size:
        road_indices[vertical_positions] = rng.integers(0, grid.config.road_count_x, size=vertical_positions.size)
        y[vertical_positions] = rng.uniform(0.0, grid.config.height_m, size=vertical_positions.size)
        x[vertical_positions] = grid.x_roads_m[road_indices[vertical_positions]] + lane_offsets[vertical_positions]
        northbound = lane_ids[vertical_positions] < grid.config.lanes_per_direction
        heading[vertical_positions] = np.where(northbound, 90.0, 270.0)
        for out_idx in vertical_positions:
            road_segment_id[out_idx] = road_segment_id_for_position(
                "vertical", int(road_indices[out_idx]), float(y[out_idx]), grid
            )

    if horizontal_positions.size:
        road_indices[horizontal_positions] = rng.integers(0, grid.config.road_count_y, size=horizontal_positions.size)
        x[horizontal_positions] = rng.uniform(0.0, grid.config.width_m, size=horizontal_positions.size)
        y[horizontal_positions] = grid.y_roads_m[road_indices[horizontal_positions]] + lane_offsets[horizontal_positions]
        eastbound = lane_ids[horizontal_positions] < grid.config.lanes_per_direction
        heading[horizontal_positions] = np.where(eastbound, 0.0, 180.0)
        for out_idx in horizontal_positions:
            road_segment_id[out_idx] = road_segment_id_for_position(
                "horizontal", int(road_indices[out_idx]), float(x[out_idx]), grid
            )

    speed = _speed_values(config, vehicle_count, rng)
    node_type = np.full(vehicle_count, "vehicle", dtype=object)

    rsu_xy = _rsu_positions(config, grid)
    rsu_count = int(rsu_xy.shape[0])
    if rsu_count:
        x = np.concatenate([x, rsu_xy[:, 0]])
        y = np.concatenate([y, rsu_xy[:, 1]])
        heading = np.concatenate([heading, np.zeros(rsu_count, dtype=float)])
        speed = np.concatenate([speed, np.zeros(rsu_count, dtype=float)])
        road_segment_id = np.concatenate([road_segment_id, np.full(rsu_count, -1, dtype=int)])
        lane_ids = np.concatenate([lane_ids, np.full(rsu_count, -1, dtype=int)])
        node_type = np.concatenate([node_type, np.full(rsu_count, "rsu", dtype=object)])

    node_count = int(x.shape[0])
    return {
        "node_id": np.arange(node_count, dtype=int),
        "x": x,
        "y": y,
        "heading": heading,
        "speed_mps": speed,
        "road_segment_id": road_segment_id,
        "lane_id": lane_ids,
        "node_type": node_type,
        "bounds": grid.bounds,
        "grid": grid,
        "seed": seed,
        "time_s": 0.0,
        # Resolved speed config so churn births (step_vehicle_snapshot) draw from the SAME
        # distribution as the initial fleet (defect fix: births used hardcoded 13.0/2.5 before).
        "speed": _resolved_speed_config(config),
    }


def advance_vehicle_snapshot(snapshot: Mapping[str, object], dt_s: float) -> dict[str, object]:
    """Advance a snapshot by ``dt_s`` seconds along each node's heading (mobility).

    Vehicles move at their speed along ``heading`` (0=E, 90=N, 180=W, 270=S) and
    wrap toroidally within the grid bounds, which keeps them on their road line and
    inside the area. RSUs (speed 0) stay put. Returns a NEW snapshot dict with the
    same structure/road metadata but advanced positions and ``time_s += dt_s``;
    ``build_candidate_graph`` recomputes distances/LOS from the new positions.
    """
    out = dict(snapshot)
    x = np.asarray(snapshot["x"], dtype=float).copy()
    y = np.asarray(snapshot["y"], dtype=float).copy()
    heading = np.asarray(snapshot["heading"], dtype=float)
    speed = np.asarray(snapshot["speed_mps"], dtype=float)
    bounds = snapshot["bounds"]
    min_x, max_x = float(bounds["min_x"]), float(bounds["max_x"])
    min_y, max_y = float(bounds["min_y"]), float(bounds["max_y"])
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    rad = np.deg2rad(heading)
    x = x + speed * float(dt_s) * np.cos(rad)
    y = y + speed * float(dt_s) * np.sin(rad)
    out["x"] = min_x + np.mod(x - min_x, width)
    out["y"] = min_y + np.mod(y - min_y, height)
    out["time_s"] = float(snapshot.get("time_s", 0.0)) + float(dt_s)
    return out


def _spawn_boundary_vehicles(grid, count, rng, start_id, speed_cfg):
    """Create ``count`` new vehicles entering at grid boundaries (P1-1.3 churn injection).

    Reuses the lane/heading/road-segment conventions of generate_vehicle_snapshot: vehicles are placed
    on a road line at the entry edge, heading inward. Returns a dict of per-vehicle arrays (length count)
    with fresh monotonically-increasing node_id starting at ``start_id``."""
    from .urban_grid import road_segment_id_for_position

    n = int(count)
    if n <= 0:
        return None
    bounds = grid.bounds
    half = float(grid.config.road_half_width_m)
    lane_choices = 2 * grid.config.lanes_per_direction
    lane_ids = rng.integers(0, lane_choices, size=n)
    lane_off = _lane_offsets(lane_ids, grid.config.lanes_per_direction, grid.config.lane_width_m)
    vertical = rng.random(n) < 0.5
    x = np.empty(n, dtype=float)
    y = np.empty(n, dtype=float)
    heading = np.empty(n, dtype=float)
    seg = np.empty(n, dtype=int)
    mean = float(speed_cfg.get("mean_mps", 13.0)); std = float(speed_cfg.get("std_mps", 2.5))
    lo = float(speed_cfg.get("min_mps", 0.0)); hi = float(speed_cfg.get("max_mps", 25.0))
    speed = np.clip(rng.normal(mean, std, n), lo, hi)
    for i in range(n):
        if vertical[i]:
            road = int(rng.integers(0, grid.config.road_count_x))
            northbound = lane_ids[i] < grid.config.lanes_per_direction
            x[i] = grid.x_roads_m[road] + lane_off[i]
            y[i] = float(bounds["min_y"]) + half if northbound else float(bounds["max_y"]) - half
            heading[i] = 90.0 if northbound else 270.0
            seg[i] = road_segment_id_for_position("vertical", road, float(y[i]), grid)
        else:
            road = int(rng.integers(0, grid.config.road_count_y))
            eastbound = lane_ids[i] < grid.config.lanes_per_direction
            y[i] = grid.y_roads_m[road] + lane_off[i]
            x[i] = float(bounds["min_x"]) + half if eastbound else float(bounds["max_x"]) - half
            heading[i] = 0.0 if eastbound else 180.0
            seg[i] = road_segment_id_for_position("horizontal", road, float(x[i]), grid)
    return {
        "node_id": np.arange(start_id, start_id + n, dtype=int),
        "x": x, "y": y, "heading": heading, "speed_mps": speed,
        "road_segment_id": seg, "lane_id": lane_ids,
        "node_type": np.full(n, "vehicle", dtype=object),
    }


def step_vehicle_snapshot(
    snapshot,
    dt_s,
    *,
    rng,
    turn_probs=(1.0, 0.0, 0.0),
    boundary_mode="toroidal",
    churn_rate_per_frame=0.0,
    turn_zone_m=8.0,
):
    """Incremental one-step mobility update with optional intersection turning + boundary churn (P1-1.3).

    Distinct from advance_vehicle_snapshot (the pure closed-form k*dt primitive, kept byte-identical):
    this is STATEFUL — frame k+1 is derived from frame k. Behaviour:
      * straight motion + recomputed road_segment_id;
      * turning: a vehicle within ``turn_zone_m`` of its nearest intersection turns left/right with
        prob turn_probs[1]/turn_probs[2] (snaps onto the intersection, rotates heading +/-90 deg);
      * boundary_mode='toroidal' wraps (legacy); 'absorb_inject' REMOVES vehicles that exit the bounds
        (computed pre-wrap) and INJECTS Poisson(churn_rate_per_frame) new boundary vehicles (real churn);
      * node_id is preserved for survivors and assigned fresh (max+1...) for births; RSUs are immobile
        and never churned.
    With turn_probs=(1,0,0), boundary_mode='toroidal', churn_rate_per_frame=0 this reproduces the linear
    toroidal advance (modulo road_segment_id refresh, which advance_vehicle_snapshot omits).
    """
    grid = snapshot.get("grid")
    if grid is None:
        raise ValueError("step_vehicle_snapshot requires snapshot['grid'] (an UrbanGrid)")
    from .urban_grid import road_segment_id_for_position

    node_id = np.asarray(snapshot["node_id"], dtype=int).copy()
    x = np.asarray(snapshot["x"], dtype=float).copy()
    y = np.asarray(snapshot["y"], dtype=float).copy()
    heading = np.asarray(snapshot["heading"], dtype=float).copy()
    speed = np.asarray(snapshot["speed_mps"], dtype=float).copy()
    seg = np.asarray(snapshot["road_segment_id"], dtype=int).copy()
    lane = np.asarray(snapshot["lane_id"], dtype=int).copy()
    ntype = np.asarray(snapshot["node_type"], dtype=object).copy()
    bounds = snapshot["bounds"]
    min_x, max_x = float(bounds["min_x"]), float(bounds["max_x"])
    min_y, max_y = float(bounds["min_y"]), float(bounds["max_y"])
    width = max(max_x - min_x, 1e-6); height = max(max_y - min_y, 1e-6)
    is_vehicle = ntype == "vehicle"

    rad = np.deg2rad(heading)
    raw_x = x + speed * float(dt_s) * np.cos(rad)
    raw_y = y + speed * float(dt_s) * np.sin(rad)

    turn_l, turn_r = float(turn_probs[1]), float(turn_probs[2])
    if (turn_l + turn_r) > 0.0:
        x_roads = np.asarray(grid.x_roads_m, dtype=float)
        y_roads = np.asarray(grid.y_roads_m, dtype=float)
        for i in np.flatnonzero(is_vehicle):
            ix = int(np.abs(raw_x[i] - x_roads).argmin())
            iy = int(np.abs(raw_y[i] - y_roads).argmin())
            cx, cy = x_roads[ix], y_roads[iy]
            if abs(raw_x[i] - cx) <= turn_zone_m and abs(raw_y[i] - cy) <= turn_zone_m:
                draw = rng.random()
                if draw < turn_l or draw < turn_l + turn_r:
                    delta = 90.0 if draw < turn_l else -90.0
                    heading[i] = (heading[i] + delta) % 360.0
                    raw_x[i], raw_y[i] = cx, cy  # snap onto the intersection, then continue next step
    x, y = raw_x, raw_y

    # recompute road_segment_id from the (post-move) orientation so LOS uses the current segment.
    for i in np.flatnonzero(is_vehicle):
        h = heading[i] % 360.0
        if abs(h - 90.0) < 1e-6 or abs(h - 270.0) < 1e-6:  # vertical travel
            road = int(np.abs(x[i] - np.asarray(grid.x_roads_m)).argmin())
            seg[i] = road_segment_id_for_position("vertical", road, float(y[i]), grid)
        else:  # horizontal travel
            road = int(np.abs(y[i] - np.asarray(grid.y_roads_m)).argmin())
            seg[i] = road_segment_id_for_position("horizontal", road, float(x[i]), grid)

    deaths = 0
    if boundary_mode == "absorb_inject":
        exited = is_vehicle & ((x < min_x) | (x > max_x) | (y < min_y) | (y > max_y))
        keep = ~exited
        deaths = int(exited.sum())
        node_id, x, y, heading, speed, seg, lane, ntype = (
            node_id[keep], x[keep], y[keep], heading[keep], speed[keep], seg[keep], lane[keep], ntype[keep]
        )
    else:  # toroidal wrap (legacy)
        x = np.where(is_vehicle, min_x + np.mod(x - min_x, width), x)
        y = np.where(is_vehicle, min_y + np.mod(y - min_y, height), y)

    births = 0
    if boundary_mode == "absorb_inject" and churn_rate_per_frame > 0.0:
        births = int(rng.poisson(float(churn_rate_per_frame)))
        spawned = _spawn_boundary_vehicles(grid, births, rng, int(node_id.max()) + 1 if node_id.size else 0,
                                           snapshot.get("speed", {}))
        if spawned is not None:
            node_id = np.concatenate([node_id, spawned["node_id"]])
            x = np.concatenate([x, spawned["x"]]); y = np.concatenate([y, spawned["y"]])
            heading = np.concatenate([heading, spawned["heading"]])
            speed = np.concatenate([speed, spawned["speed_mps"]])
            seg = np.concatenate([seg, spawned["road_segment_id"]])
            lane = np.concatenate([lane, spawned["lane_id"]])
            ntype = np.concatenate([ntype, spawned["node_type"]])

    out = dict(snapshot)
    out.update({
        "node_id": node_id, "x": x, "y": y, "heading": heading, "speed_mps": speed,
        "road_segment_id": seg, "lane_id": lane, "node_type": ntype,
        "time_s": float(snapshot.get("time_s", 0.0)) + float(dt_s),
        "births": births, "deaths": deaths,
    })
    return out
