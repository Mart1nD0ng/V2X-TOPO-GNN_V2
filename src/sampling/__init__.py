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
from .dpp_query import (
    diagonal_diversity,
    enumerate_kdpp_distribution,
    kdpp_elementary_symmetric,
    kdpp_inclusion,
    kdpp_log_normalizer,
    kdpp_normalizer,
    kdpp_sample,
    kdpp_subset_log_prob,
    low_rank_kernel,
)
from .cdq_query import DiagonalCDQPolicy, cdq_bucketed_quorum, cdq_edge_inclusion
from .determinantal_quorum import (
    QuorumDecisionCDQ,
    bruteforce_determinantal_quorum,
    determinantal_quorum_decision,
    determinantal_quorum_distribution,
)
from .esp_query import edge_inclusion_probabilities

__all__ = [
    "edge_inclusion_probabilities",
    "QuorumDecisionCDQ",
    "determinantal_quorum_distribution",
    "determinantal_quorum_decision",
    "bruteforce_determinantal_quorum",
    "cdq_edge_inclusion",
    "cdq_bucketed_quorum",
    "DiagonalCDQPolicy",
    "QueryPolicy",
    "UniformQueryPolicy",
    "DistanceQueryPolicy",
    "low_rank_kernel",
    "kdpp_elementary_symmetric",
    "kdpp_normalizer",
    "kdpp_log_normalizer",
    "kdpp_inclusion",
    "kdpp_subset_log_prob",
    "enumerate_kdpp_distribution",
    "kdpp_sample",
    "diagonal_diversity",
]
