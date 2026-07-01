"""Correlated-evidence urban V2X environment (engineering plan §2, spec §6).

Modules:

* ``evidence_model`` -- the region/node shared-error observation model
  ``O_i = Y* ⊕ B_{g(i)} ⊕ E_i`` (spec §6.2). Produces (a) per-instance node
  observations for the dynamic MC and (b) the shared-latent scenario decomposition
  ``(omega_r, init_correct_pref[i,r])`` for the analytic evaluator, with exact
  pairwise-correlation theory for validation (G2).

Geometry (`urban_scene`), candidate/interference graphs, and the round-coupled
canonical episode are added in later slices (Phase 3-4, G2/G3).
"""

from .candidate_graph import (
    RadiusGraph,
    aggregate_over_graph,
    build_candidate_graph,
    build_radius_graph,
    edge_set,
)
from .evidence_model import (
    EvidenceModel,
    EvidenceSample,
    pairwise_correlation_theory,
)
from .interference_graph import (
    build_interference_graph,
    non_intended_interferers,
    received_interference_mw,
)
from .canonical_episode import EpisodeResult, ProtocolConfig, run_consensus_episode
from .mobility import drift_scene
from .round_physics import RoundPhysicsConfig, RoundPhysicsResult, edge_geometry, round_physics
from .scenarios import EVIDENCE_SCENARIOS, GEOMETRIC_SCENARIOS, build_scenario
from .urban_scene import ManhattanScene, build_manhattan_scene
from .nonuniform_urban_scene import NonuniformUrbanScene, build_nonuniform_urban_scene
from .sps_resource import assign_sps_buckets, same_resource_conflict_degree, sensed_channel_busy_ratio
from .csi_aging import stale_csi_edge_features, csi_uncertainty_db
from .receiver_capacity import assign_receiver_capacity, noisy_capacity_proxy

__all__ = [
    "EVIDENCE_SCENARIOS",
    "GEOMETRIC_SCENARIOS",
    "build_scenario",
    "EpisodeResult",
    "ProtocolConfig",
    "run_consensus_episode",
    "RoundPhysicsConfig",
    "RoundPhysicsResult",
    "round_physics",
    "edge_geometry",
    "EvidenceModel",
    "EvidenceSample",
    "pairwise_correlation_theory",
    "RadiusGraph",
    "build_radius_graph",
    "build_candidate_graph",
    "aggregate_over_graph",
    "edge_set",
    "build_interference_graph",
    "non_intended_interferers",
    "received_interference_mw",
    "ManhattanScene",
    "build_manhattan_scene",
    "NonuniformUrbanScene",
    "build_nonuniform_urban_scene",
    "assign_sps_buckets",
    "same_resource_conflict_degree",
    "sensed_channel_busy_ratio",
    "stale_csi_edge_features",
    "csi_uncertainty_db",
    "assign_receiver_capacity",
    "noisy_capacity_proxy",
    "drift_scene",
]
