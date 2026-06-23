"""Closed-form Snowball/Avalanche consensus evaluators.

The query_support_optimization_design diagnostic module was moved to
``archive/diagnostics/src/consensus/`` — see ``docs/MAINLINE.md``.
"""

from .avalanche_closed_form import (
    avalanche_state_count,
    evaluate_avalanche_closed_form,
    quorum_success_probability,
)
from .graph_coupled_avalanche import evaluate_graph_coupled_avalanche
from .topology_query_support import (
    DIAGNOSTICS_MODES,
    QUERY_SUPPORT_BACKENDS,
    QuerySupport,
    compute_topology_query_support,
    evaluate_topology_avalanche_static,
)

__all__ = [
    "QuerySupport",
    "DIAGNOSTICS_MODES",
    "QUERY_SUPPORT_BACKENDS",
    "avalanche_state_count",
    "compute_topology_query_support",
    "evaluate_avalanche_closed_form",
    "evaluate_graph_coupled_avalanche",
    "evaluate_topology_avalanche_static",
    "quorum_success_probability",
]
