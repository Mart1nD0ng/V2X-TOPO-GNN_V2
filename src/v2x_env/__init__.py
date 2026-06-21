"""Deterministic V2X environment feasibility utilities."""

from .urban_grid import UrbanGridConfig, make_urban_grid
from .vehicle_snapshot import advance_vehicle_snapshot, generate_vehicle_snapshot
from .candidate_graph import CandidateGraph, CandidateGraphConfig, build_candidate_graph
from .channel_model import ChannelConfig
from .feasibility import evaluate_environment_config
from .mobility import MobilityConfig, MobilityStream
from .profiles import (
    PRODUCTION_ACTIVE_DEGREE,
    PRODUCTION_CANDIDATE_CAP,
    PRODUCTION_CANDIDATE_RADIUS_M,
    PRODUCTION_INITIAL_CORRECT,
    PRODUCTION_INITIAL_WRONG,
    PRODUCTION_NODE_DENSITY_PER_KM2,
    PRODUCTION_TARGET_ROAD_SPACING_M,
    density_matched_vehicle_config,
    production_like_density_v0_vehicle_config,
)

__all__ = [
    "UrbanGridConfig",
    "make_urban_grid",
    "generate_vehicle_snapshot",
    "advance_vehicle_snapshot",
    "CandidateGraph",
    "CandidateGraphConfig",
    "build_candidate_graph",
    "ChannelConfig",
    "evaluate_environment_config",
    "MobilityConfig",
    "MobilityStream",
    "PRODUCTION_NODE_DENSITY_PER_KM2",
    "PRODUCTION_TARGET_ROAD_SPACING_M",
    "PRODUCTION_CANDIDATE_RADIUS_M",
    "PRODUCTION_CANDIDATE_CAP",
    "PRODUCTION_ACTIVE_DEGREE",
    "PRODUCTION_INITIAL_CORRECT",
    "PRODUCTION_INITIAL_WRONG",
    "production_like_density_v0_vehicle_config",
    "density_matched_vehicle_config",
]
