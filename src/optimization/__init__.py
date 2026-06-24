"""Optimization layer (engineering plan §2, Phase 7).

* ``topology_oracle`` -- the direct per-scene topology optimizer (spec plan §10 "direct
  per-scene edge-logit optimizer") used to establish the topology-optimization CEILING and
  decide loop stop-condition #2 (does optimizing the query topology beat the heuristics?).
  It is a per-scene UPPER BOUND, not a deployable policy: it descends on the analytic
  episode objective, which encodes the scene's evidence; the deployable ESD-GNN (G9) must
  approach this ceiling from observable features alone (constraint #10).

Primal-dual reliability-constrained training of the ESD-GNN is added at G9.
"""

from .primal_dual import (
    DualState,
    ReliabilityThresholds,
    episode_metrics,
    lagrangian,
    train_esd_gnn,
)
from .topology_oracle import (
    LearnedLogWeightPolicy,
    optimize_logweight_topology,
    oracle_vs_heuristics,
)

__all__ = [
    "LearnedLogWeightPolicy",
    "optimize_logweight_topology",
    "oracle_vs_heuristics",
    "ReliabilityThresholds",
    "DualState",
    "episode_metrics",
    "lagrangian",
    "train_esd_gnn",
]
