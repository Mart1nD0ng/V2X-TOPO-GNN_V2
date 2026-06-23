from __future__ import annotations

import math
from typing import Any


PRODUCTION_NODE_DENSITY_PER_KM2 = 300.0
PRODUCTION_TARGET_ROAD_SPACING_M = 180.0
PRODUCTION_CANDIDATE_RADIUS_M = 230.0
PRODUCTION_CANDIDATE_CAP = 8
PRODUCTION_ACTIVE_DEGREE = 4
PRODUCTION_INITIAL_CORRECT = 0.50
PRODUCTION_INITIAL_WRONG = 0.25


def production_like_density_v0_vehicle_config(node_count: int, *, seed: int) -> dict[str, Any]:
    """Production-density urban-grid snapshot config.

    This profile keeps spatial density explicit as node count changes, so the
    sparse candidate graph remains an O(Nk) deployment proxy instead of a fixed
    toy canvas that becomes hyper-dense at large N.
    """
    target_area_m2 = float(node_count) / PRODUCTION_NODE_DENSITY_PER_KM2 * 1_000_000.0
    side_m = math.sqrt(target_area_m2)
    road_count = max(3, int(round(side_m / PRODUCTION_TARGET_ROAD_SPACING_M)) + 1)
    block_m = side_m / max(road_count - 1, 1)
    return {
        "seed": int(seed),
        "vehicle_count": int(node_count),
        "grid": {
            "block_length_m": float(block_m),
            "block_width_m": float(block_m),
            "road_count_x": int(road_count),
            "road_count_y": int(road_count),
            "lanes_per_direction": 1,
            "lane_width_m": 3.5,
        },
        "speed": {"mean_mps": 12.0, "std_mps": 2.0, "min_mps": 0.0, "max_mps": 25.0},
        "profile_contract": {
            "target_node_density_per_km2": PRODUCTION_NODE_DENSITY_PER_KM2,
            "target_area_m2": target_area_m2,
            "target_road_spacing_m": PRODUCTION_TARGET_ROAD_SPACING_M,
            "rsu_policy": "rsu_count_zero_for_v0",
            "candidate_radius_m": PRODUCTION_CANDIDATE_RADIUS_M,
            "candidate_cap": PRODUCTION_CANDIDATE_CAP,
            "active_topk_degree": PRODUCTION_ACTIVE_DEGREE,
        },
    }


def density_matched_vehicle_config(node_count: int, density_per_km2: float, *, seed: int) -> dict[str, Any]:
    """Urban-grid snapshot config sized to a requested vehicle density."""
    if density_per_km2 <= 0.0:
        raise ValueError("density_per_km2 must be positive")
    target_area_m2 = float(node_count) / float(density_per_km2) * 1_000_000.0
    side_m = math.sqrt(target_area_m2)
    road_count = max(3, int(round(side_m / PRODUCTION_TARGET_ROAD_SPACING_M)) + 1)
    block_m = side_m / max(road_count - 1, 1)
    return {
        "seed": int(seed),
        "vehicle_count": int(node_count),
        "grid": {
            "block_length_m": float(block_m),
            "block_width_m": float(block_m),
            "road_count_x": int(road_count),
            "road_count_y": int(road_count),
            "lanes_per_direction": 1,
            "lane_width_m": 3.5,
        },
        "speed": {"mean_mps": 12.0, "std_mps": 2.0, "min_mps": 0.0, "max_mps": 25.0},
        "profile_contract": {
            "target_node_density_per_km2": float(density_per_km2),
            "target_area_m2": target_area_m2,
            "target_road_spacing_m": PRODUCTION_TARGET_ROAD_SPACING_M,
            "candidate_radius_m": PRODUCTION_CANDIDATE_RADIUS_M,
            "candidate_cap": PRODUCTION_CANDIDATE_CAP,
            "active_topk_degree": PRODUCTION_ACTIVE_DEGREE,
        },
    }
