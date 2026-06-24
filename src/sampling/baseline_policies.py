"""Baseline query policies (plan §10.1) -- log-weights from observable features only.

A query policy maps the candidate graph + deployment-observable per-edge features to a
per-edge log-weight ``s_ij`` (the ESP/CDQ query weight). It may use geometry, distance,
historical link quality, credibility and region -- but **never** the ground truth ``Y*`` or
any peer's current vote/preference (spec §6.3, constraint #10). The same policy object is
used identically in the analytic evaluator and the dynamic MC (constraint #3).

These are the heuristic baselines the learned ESD-GNN must beat (plan §10.1):

* :class:`UniformQueryPolicy`     -- equal weights ⇒ ``pi_ij = k / deg_i`` (distinct-peer).
* :class:`DistanceQueryPolicy`    -- prefer closer / stronger links (``s = -beta·d``).

The ESP-product and learned policies plug into the same interface (G4/G9).
"""

from __future__ import annotations

from typing import Protocol

import torch

from src.environment.candidate_graph import RadiusGraph

__all__ = ["QueryPolicy", "UniformQueryPolicy", "DistanceQueryPolicy"]


class QueryPolicy(Protocol):
    """Produces per-edge query log-weights from observable features (no truth/vote)."""

    name: str

    def log_weights(self, graph: RadiusGraph) -> torch.Tensor:
        """Return ``[E]`` log-weights ``s_ij`` for the candidate graph's edges."""
        ...


class UniformQueryPolicy:
    """Uniform distinct-peer policy: equal weight on every candidate edge."""

    name = "uniform"

    def log_weights(self, graph: RadiusGraph) -> torch.Tensor:
        return torch.zeros(graph.num_edges, dtype=graph.distance.dtype, device=graph.distance.device)


class DistanceQueryPolicy:
    """Distance-decaying policy: ``s_ij = -beta · distance_ij`` (prefers nearer peers).

    A canonical "exploit local link quality" heuristic -- closer peers have higher SINR but
    are also more likely to share the same region/occlusion (the local-quality-vs-diversity
    tension, spec §9.1). Uses only the observable edge distance.
    """

    name = "distance"

    def __init__(self, beta_per_m: float = 0.02):
        self.beta_per_m = float(beta_per_m)

    def log_weights(self, graph: RadiusGraph) -> torch.Tensor:
        return -self.beta_per_m * graph.distance
