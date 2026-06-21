"""Sparse model skeletons for V2X topology scoring.

Diagnostic modules (structural_signal_audit, structural_redesign_review,
scorer_parameterization_redesign, scorer_dynamic_range_diagnostic) were moved to
``archive/diagnostics/src/models/`` — see ``docs/MAINLINE.md``.
"""

from .hierarchical_gnn import HierarchicalGNNScorer, apply_dropedge
from .temporal_scorer import TemporalGNNScorer
from .structural_diagnostics import compute_cap_diagnostics, compute_structural_diagnostics
from .structural_objectives import (
    bridge_logit_regularizer,
    budget_entropy_loss,
    budget_target_loss,
    role_balance_loss,
    sector_entropy_loss,
)

__all__ = [
    "HierarchicalGNNScorer",
    "TemporalGNNScorer",
    "apply_dropedge",
    "bridge_logit_regularizer",
    "budget_entropy_loss",
    "budget_target_loss",
    "compute_cap_diagnostics",
    "compute_structural_diagnostics",
    "role_balance_loss",
    "sector_entropy_loss",
]
