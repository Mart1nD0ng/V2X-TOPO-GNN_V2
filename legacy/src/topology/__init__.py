"""Sparse topology construction utilities.

Diagnostic design/report modules (constructor_optimization_design,
row_softmax_temperature_design, row_softmax_temperature_constructor_report,
support_smoothing_constructor_report) were moved to ``archive/diagnostics/src/topology/``
— see ``docs/MAINLINE.md``.
"""

from .construction import ConstructedTopology, TopologyConstructionLayer
from .budget_adapter import expected_budget_to_cap
from .constructor_profile import profile_topology_constructor_case, run_constructor_bottleneck_diagnostic

__all__ = [
    "ConstructedTopology",
    "TopologyConstructionLayer",
    "expected_budget_to_cap",
    "profile_topology_constructor_case",
    "run_constructor_bottleneck_diagnostic",
]
