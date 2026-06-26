"""Independent validation layer (engineering plan §2, spec §8).

* ``dynamic_mc`` -- the independent dynamic Monte-Carlo judge (spec §8.1 level 3): a
  round-by-round FORWARD simulation that samples evidence, query subsets, per-poll
  request/response outcomes and uses the **actual** sampled peer states, advancing the
  **true** binary-Snowball counters. It NEVER samples from the analytic terminal marginals
  (constraint #8); it shares only the system definition (physics model + query policy) with
  the analytic episode. Its role is to calibrate absolute safety/deadline and validate the
  analytic ranking/gradient (spec §8.3); a systematic analytic-vs-MC disagreement is a
  spec stop condition (#3), not something to paper over.

The small-``N`` exact joint chain (spec §8.1 level 2) and rare-event estimators (level 4)
are added in follow-up slices.
"""

from .dynamic_mc import DynamicMCResult, run_dynamic_mc
from .feasibility import (
    FeasibilityFloors,
    binomial_tail,
    is_feasible,
    network_floors,
    scan_feasibility,
    wellmixed_terminal,
)

__all__ = [
    "DynamicMCResult",
    "run_dynamic_mc",
    "FeasibilityFloors",
    "binomial_tail",
    "wellmixed_terminal",
    "network_floors",
    "is_feasible",
    "scan_feasibility",
]
