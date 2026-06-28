"""Strong observable-structure ESP heuristics (esp_performance_scale_v2, G-ESP-BASELINE-ORACLE).

A learned-topology scale claim is only meaningful against STRONG, capability-matched baselines
(workflow §5). These are deployable ESP policies that emit per-edge log-weights ``s_ij`` from
DEPLOYMENT-OBSERVABLE structure only (geometry / candidate-graph degree / region labels) — never truth,
votes, or simulator-only latent state (workflow §5.4). Each exposes the ``log_weights(graph) -> [E]``
QueryPolicy interface and the default ESP query law.

- ``LinkQualityPolicy``  — prefer high-LOS (better-link) peers (a link-quality signal beyond raw distance).
- ``LoadBalancedPolicy`` — prefer low-incoming-load receivers (avoid congesting popular nodes).
- ``RegionBridgePolicy`` — prefer cross-region (bridge) peers that spread consensus across the map.
"""

from __future__ import annotations

import torch

from src.environment.round_physics import _los_probability

__all__ = ["LinkQualityPolicy", "LoadBalancedPolicy", "RegionBridgePolicy"]


class LinkQualityPolicy:
    """ESP log-weights from the observable line-of-sight probability (prefer better links)."""

    name = "link_quality"

    def __init__(self, *, los_ref_m: float = 50.0, scale: float = 4.0):
        self.los_ref_m = float(los_ref_m)
        self.scale = float(scale)

    def log_weights(self, graph) -> torch.Tensor:
        los = _los_probability(graph.distance, self.los_ref_m)             # [E] in (0,1]
        return self.scale * los.to(torch.float64)


class LoadBalancedPolicy:
    """ESP log-weights that DOWN-weight high-incoming-load receivers (observable candidate in-degree).

    Bound to a scene at construction (like ``ESDGNNQueryPolicy``); the episode passes its own G_comm,
    which shares the scene's positions so the in-degree is consistent."""

    name = "load_balanced"

    def __init__(self, scene, *, scale: float = 1.0):
        self.scene = scene
        self.scale = float(scale)

    def log_weights(self, graph) -> torch.Tensor:
        n = graph.num_nodes
        in_deg = torch.bincount(graph.dst_index, minlength=n).to(torch.float64)
        return -self.scale * torch.log1p(in_deg[graph.dst_index])         # prefer low-load destinations


class RegionBridgePolicy:
    """ESP log-weights that prefer CROSS-region (bridge) peers (observable road-region labels), to spread
    consensus across the map faster; a small same-region floor keeps in-region peers eligible."""

    name = "region_bridge"

    def __init__(self, scene, *, cross_bonus: float = 2.0, same_floor: float = 0.5):
        self.scene = scene
        self.cross_bonus = float(cross_bonus)
        self.same_floor = float(same_floor)

    def log_weights(self, graph) -> torch.Tensor:
        region = self.scene.region_of
        same = (region[graph.src_index] == region[graph.dst_index]).to(torch.float64)
        return self.cross_bonus * (1.0 - same) + self.same_floor * same
