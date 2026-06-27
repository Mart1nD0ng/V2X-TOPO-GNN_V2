"""Hazard-weighted service profiles (Guarded-CDQ2 round, spec §5 / plan §7).

An application's choice between ESP / CDQ2 / Guarded-CDQ2 is NOT one-size-fits-all: it depends on the
hazard COSTS of each outcome. A safety-first application pays a huge cost for a wrong/split consensus
(so it should stay on ESP); a deadline-critical application pays a huge cost for a missed deadline (so
it should accept CDQ2's liveness gain where the reliability guard permits). A :class:`HazardProfile`
encodes those costs, and :mod:`src.evaluation.hazard_utility` turns them into a net-benefit
``B_CDQ`` and a feasibility-gated policy selection.

The costs are the relative HAZARD of moving each basin/objective by one unit vs ESP; reliability
(wrong/split) is penalised only on an INCREASE (``[ΔF]_+``) -- it is a hard constraint, never a tradable
benefit (constraint #5). Deadline / tail-latency / energy are optimisation targets within the feasible
set (constraint #6).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HazardProfile", "STANDARD_PROFILES"]


@dataclass(frozen=True)
class HazardProfile:
    """Hazard costs for the net-benefit objective ``B`` (spec §5.2).

    ``c_w``/``c_s``: cost per unit INCREASE in the wrong/split basin (vs ESP) -- the validity hazard.
    ``c_d``: value per unit REDUCTION in the deadline-miss basin -- the liveness benefit.
    ``c_T``: value per unit reduction in tail latency (``D_q`` / CVaR). ``c_E``: value per unit energy
    saved. ``eps_w``/``eps_s``: the hard wrong/split UCB budgets a candidate policy must satisfy to be
    eligible at all (the feasibility gate, constraint #4/#5).
    """

    name: str
    c_w: float
    c_s: float
    c_d: float
    c_T: float = 0.0
    c_E: float = 0.0
    eps_w: float = 0.05
    eps_s: float = 0.05
    expected_policy: str = ""        # the spec's qualitative expectation (documentation / acceptance)


# spec §5.3 suggested profiles. Costs are relative magnitudes (very-high >> high >> medium >> low). The
# reliability budget eps is part of the profile: a safety-first / fail-safe application sets a STRICT eps
# (so any risk-adding policy is ineligible -> ESP), while a deadline-critical application tolerates a
# looser eps (so the reliability guard can enable diversity for the liveness gain).
STANDARD_PROFILES = (
    HazardProfile("safety_first", c_w=1000.0, c_s=1000.0, c_d=10.0, c_T=1.0, c_E=0.1,
                  eps_w=1e-3, eps_s=1e-3, expected_policy="ESP"),
    HazardProfile("balanced", c_w=100.0, c_s=100.0, c_d=100.0, c_T=10.0, c_E=1.0,
                  eps_w=0.08, eps_s=0.08, expected_policy="Guarded-CDQ2"),
    HazardProfile("deadline_critical", c_w=50.0, c_s=100.0, c_d=1000.0, c_T=100.0, c_E=1.0,
                  eps_w=0.10, eps_s=0.10, expected_policy="CDQ2 or Guarded-CDQ2"),
    HazardProfile("fail_safe_available", c_w=1000.0, c_s=1000.0, c_d=1.0, c_T=1.0, c_E=0.1,
                  eps_w=1e-3, eps_s=1e-3, expected_policy="ESP"),
    HazardProfile("energy_constrained", c_w=100.0, c_s=100.0, c_d=10.0, c_T=10.0, c_E=100.0,
                  eps_w=0.08, eps_s=0.08, expected_policy="Guarded-CDQ2 only if energy improves"),
)
