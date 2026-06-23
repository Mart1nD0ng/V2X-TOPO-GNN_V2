from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class UrbanGridConfig:
    block_length_m: float = 200.0
    block_width_m: float = 200.0
    road_count_x: int = 5
    road_count_y: int = 5
    lanes_per_direction: int = 1
    lane_width_m: float = 3.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "UrbanGridConfig":
        data = data or {}
        return cls(
            block_length_m=float(data.get("block_length_m", cls.block_length_m)),
            block_width_m=float(data.get("block_width_m", cls.block_width_m)),
            road_count_x=int(data.get("road_count_x", cls.road_count_x)),
            road_count_y=int(data.get("road_count_y", cls.road_count_y)),
            lanes_per_direction=int(data.get("lanes_per_direction", cls.lanes_per_direction)),
            lane_width_m=float(data.get("lane_width_m", cls.lane_width_m)),
        )

    @property
    def width_m(self) -> float:
        return self.block_length_m * max(self.road_count_x - 1, 1)

    @property
    def height_m(self) -> float:
        return self.block_width_m * max(self.road_count_y - 1, 1)

    @property
    def road_half_width_m(self) -> float:
        return self.lanes_per_direction * self.lane_width_m

    @property
    def bounds(self) -> dict[str, float]:
        margin = self.road_half_width_m
        return {
            "min_x": -margin,
            "max_x": self.width_m + margin,
            "min_y": -margin,
            "max_y": self.height_m + margin,
        }


@dataclass(frozen=True)
class UrbanGrid:
    config: UrbanGridConfig
    x_roads_m: np.ndarray
    y_roads_m: np.ndarray
    intersections_xy: np.ndarray

    @property
    def bounds(self) -> dict[str, float]:
        return self.config.bounds


def make_urban_grid(config: UrbanGridConfig | Mapping[str, object] | None = None) -> UrbanGrid:
    grid_config = config if isinstance(config, UrbanGridConfig) else UrbanGridConfig.from_mapping(config)
    if grid_config.road_count_x < 2 or grid_config.road_count_y < 2:
        raise ValueError("urban grid requires at least two roads in each direction")
    if grid_config.lanes_per_direction < 1:
        raise ValueError("lanes_per_direction must be positive")

    x_roads = np.arange(grid_config.road_count_x, dtype=float) * grid_config.block_length_m
    y_roads = np.arange(grid_config.road_count_y, dtype=float) * grid_config.block_width_m
    intersections = np.array([(x, y) for x in x_roads for y in y_roads], dtype=float)
    return UrbanGrid(config=grid_config, x_roads_m=x_roads, y_roads_m=y_roads, intersections_xy=intersections)


def road_segment_id_for_position(
    orientation: str,
    road_index: int,
    along_m: float,
    grid: UrbanGrid,
) -> int:
    if orientation == "vertical":
        block = int(np.clip(along_m // grid.config.block_width_m, 0, grid.config.road_count_y - 2))
        return road_index * (grid.config.road_count_y - 1) + block
    if orientation == "horizontal":
        block = int(np.clip(along_m // grid.config.block_length_m, 0, grid.config.road_count_x - 2))
        offset = grid.config.road_count_x * (grid.config.road_count_y - 1)
        return offset + road_index * (grid.config.road_count_x - 1) + block
    raise ValueError(f"unknown road orientation: {orientation}")
