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

from .evidence_model import (
    EvidenceModel,
    EvidenceSample,
    pairwise_correlation_theory,
)

__all__ = [
    "EvidenceModel",
    "EvidenceSample",
    "pairwise_correlation_theory",
]
