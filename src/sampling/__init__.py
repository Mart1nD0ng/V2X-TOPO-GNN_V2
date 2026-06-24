"""Query-sampling layer (engineering plan §2).

* ``esp_query`` -- exact weighted distinct-peer ``k``-subset inclusion probabilities via
  elementary symmetric polynomials (spec §4); the diagonal-kernel special case of the CDQ
  ``k``-DPP query law (spec §9.4) added in G4. Reuses the approved
  ``src.mainline.symmetric_polynomials`` / degree-bucketed layout assets.
* ``baseline_policies`` -- query policies producing per-edge log-weights from
  deployment-observable features only (no ground truth / no peer vote, constraint #10):
  uniform, distance/SINR, used as the canonical baselines (plan §10) until the ESD-GNN.
"""

from .baseline_policies import (
    DistanceQueryPolicy,
    QueryPolicy,
    UniformQueryPolicy,
)
from .esp_query import edge_inclusion_probabilities

__all__ = [
    "edge_inclusion_probabilities",
    "QueryPolicy",
    "UniformQueryPolicy",
    "DistanceQueryPolicy",
]
