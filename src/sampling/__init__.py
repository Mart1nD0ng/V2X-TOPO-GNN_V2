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
from .cdq2_kernel import (
    bruteforce_cdq2_normalizer,
    cdq2_enumerate_distribution,
    cdq2_inclusion,
    cdq2_k2_diversity_factor,
    cdq2_log_normalizer,
    cdq2_normalizer,
    cdq2_sample,
    cdq2_subset_log_prob,
    cdq2_subset_logdet,
    cdq2_unit_normalize,
)
from .cdq2_quorum import cdq2_quorum_decision, cdq2_quorum_distribution
from .cdq2_correlation import (
    bruteforce_cdq2_pairwise_inclusion,
    cdq2_correlation_cost,
    cdq2_pairwise_inclusion,
)
from .cdq2_wiring import CDQ2Policy, cdq2_bucketed_quorum, cdq2_edge_inclusion
from .effective_dynamics import (
    cross_region_response_mass,
    effective_sample_size,
    progress_drift,
    region_response_kernel,
    region_spectral_gap,
    response_conditioned_marginal,
)
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
    "response_conditioned_marginal",
    "progress_drift",
    "effective_sample_size",
    "cross_region_response_mass",
    "region_response_kernel",
    "region_spectral_gap",
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
    "cdq2_unit_normalize",
    "cdq2_k2_diversity_factor",
    "cdq2_subset_logdet",
    "cdq2_log_normalizer",
    "cdq2_normalizer",
    "cdq2_subset_log_prob",
    "cdq2_inclusion",
    "cdq2_enumerate_distribution",
    "bruteforce_cdq2_normalizer",
    "cdq2_sample",
    "cdq2_quorum_distribution",
    "cdq2_quorum_decision",
    "cdq2_pairwise_inclusion",
    "cdq2_correlation_cost",
    "bruteforce_cdq2_pairwise_inclusion",
    "CDQ2Policy",
    "cdq2_edge_inclusion",
    "cdq2_bucketed_quorum",
]
