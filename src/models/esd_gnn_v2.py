"""ESDGNNStaticV2 -- the ESD-GNN encoder on the expanded NDH feature schema (G-NDH-FEATURE-SCHEMA).

Phase-1 minimal upgrade: the SAME multi-graph encoder + dynamics refinement + CDQ/ESP heads as
``ESDGNN`` (source-side / destination-side / interference / region-pooling / load-refinement
channels, spec §9.3-§9.6), only with the input dimensions of the expanded ``SceneFeaturesV2``
schema (``NODE_FEATURE_NAMES`` / ``EDGE_FEATURE_NAMES``). Off-mechanism feature columns are zeroed
by the builder (and flagged in the availability mask), so a plain scene reduces to the base
structural behaviour. No legacy GRU / temporal state (that is the Phase-2 branch).

The model consumes only the DEPLOYABLE proxy features (spec §7.2 / Contract C2) — the builder never
puts the true ``mu_j`` or any truth/future signal into the tensor.
"""

from __future__ import annotations

import torch

from .esd_gnn import ESDGNN, ESDGNNConfig
from .scene_features_v2 import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    SceneFeaturesV2,
    build_scene_features_v2,
)

__all__ = ["ESDGNNStaticV2", "ESDGNNStaticV2QueryPolicy"]


class ESDGNNStaticV2(ESDGNN):
    """``ESDGNN`` sized for the expanded ``SceneFeaturesV2`` node/edge schema."""

    def __init__(self, cfg: ESDGNNConfig):
        super().__init__(cfg, node_dim=len(NODE_FEATURE_NAMES), edge_dim=len(EDGE_FEATURE_NAMES))

    def forward(self, feats: SceneFeaturesV2) -> tuple[torch.Tensor, torch.Tensor]:
        # SceneFeaturesV2 exposes the same node_feat/edge_feat/gc/gi/region_of/num_regions the base
        # ESDGNN forward reads, so the encoder runs unchanged on the wider feature schema.
        return super().forward(feats)


class ESDGNNStaticV2QueryPolicy:
    """Wrap an :class:`ESDGNNStaticV2` + scene as a query policy on the V2 feature schema.

    Mirrors ``ESDGNNQueryPolicy`` but builds ``SceneFeaturesV2`` (the expanded deployable features).
    ``query_law`` follows ``cfg.use_cdq`` (CDQ kernel vs ESP diagonal log-weights).
    """

    name = "esd_gnn_v2"

    def __init__(self, model: ESDGNNStaticV2, scene, phy_cfg, *,
                 features: SceneFeaturesV2 | None = None, generator: torch.Generator | None = None,
                 **feature_kwargs):
        self.model = model
        self.scene = scene
        self.query_law = "cdq" if model.cfg.use_cdq else "esp"
        self.features = features if features is not None else build_scene_features_v2(
            scene, phy_cfg, generator=generator, **feature_kwargs)

    def kernel(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(self.features)

    def log_weights(self, graph) -> torch.Tensor:
        quality, _ = self.model(self.features)
        return torch.log(quality)
