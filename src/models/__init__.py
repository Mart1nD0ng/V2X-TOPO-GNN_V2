"""ESD-GNN models (engineering plan §2, spec §9).

* ``esd_gnn`` -- the Effective-Sampling-Dynamics GNN: a multi-graph, multi-scale encoder
  (G_comm / G_int / region supergraph) over OBSERVABLE structural features (degrees,
  distances, region structure -- NO ground truth, NO peer votes; constraint #10) producing
  per-edge quality ``q`` and diversity ``b`` for the CDQ k-DPP query (G4) + determinantal
  quorum (G5). Scene/scale-agnostic (no scene-specific ids) so a single model transfers
  across N=100..10000. Wrapped as a ``query_law="cdq"`` policy for the canonical episode.

Primal-dual reliability-constrained training (spec §4.5) and the mechanism ablations are
added in the G9 training / ablation slices.
"""

from .esd_gnn import (
    ESDGNN,
    ESDGNNConfig,
    ESDGNNQueryPolicy,
    build_scene_features,
)

__all__ = ["ESDGNN", "ESDGNNConfig", "ESDGNNQueryPolicy", "build_scene_features"]
