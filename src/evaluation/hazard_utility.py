"""Hazard-weighted net benefit + feasibility-gated policy selection (spec §5.2 / plan §7.3).

Given a policy's macrostate outcomes vs ESP and a :class:`HazardProfile`, compute

    B = c_d (F_d^ESP - F_d^policy) + c_T (D_q^ESP - D_q^policy) + c_E (E^ESP - E^policy)
        - c_w [F_w^policy - F_w^ESP]_+ - c_s [F_s^policy - F_s^ESP]_+

(ESP is the baseline, B=0). A policy is ELIGIBLE only if its wrong/split UCB are within the profile's
budgets (constraint #4/#5); among eligible policies the one with the largest B is selected, with ESP
(the default, constraint #3) chosen whenever no alternative beats it. The deadline/tail/energy terms are
optimisation targets within the feasible set (constraint #6) -- a deadline gain is a LIVENESS benefit,
never counted as a reliability improvement (constraint #1).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config.hazard_profile import HazardProfile

__all__ = ["PolicyOutcome", "hazard_benefit", "is_eligible", "select_policy", "SelectionResult"]


@dataclass(frozen=True)
class PolicyOutcome:
    """A policy's dynamic-MC outcomes for the hazard objective (read from a result record, not recomputed)."""

    name: str
    F_wrong: float
    F_split: float
    F_deadline: float
    F_wrong_ucb: float
    F_split_ucb: float
    D_q: float = 0.0           # tail latency (CVaR / D_99), seconds
    energy: float = 0.0        # mean per-trial network energy

    @classmethod
    def from_macro(cls, name: str, macro: dict, *, D_q: float = 0.0, energy: float = 0.0) -> "PolicyOutcome":
        return cls(name=name, F_wrong=macro["macro_F_wrong"], F_split=macro["macro_F_split"],
                   F_deadline=macro["macro_F_deadline"],
                   F_wrong_ucb=macro.get("macro_F_wrong_ci", (0.0, macro["macro_F_wrong"]))[1],
                   F_split_ucb=macro.get("macro_F_split_ci", (0.0, macro["macro_F_split"]))[1],
                   D_q=D_q, energy=energy)


def hazard_benefit(esp: PolicyOutcome, policy: PolicyOutcome, profile: HazardProfile,
                   *, eps: float = 1e-9) -> float:
    """Net benefit ``B`` of ``policy`` relative to ``esp`` under ``profile`` (spec §5.2). ESP vs ESP = 0.

    The deadline term is on the basin probability scale [0,1]; tail latency and energy are NORMALISED to
    FRACTIONAL changes relative to ESP (``ΔD_q/D_q^ESP``, ``ΔE/E^ESP``) so all four optimisation terms live
    on a comparable 0..1 scale and the cost weights are unit-free relative hazards (otherwise an absolute
    energy in joules would swamp a probability)."""
    d_deadline = esp.F_deadline - policy.F_deadline                       # >0 => fewer deadline misses
    d_tail = (esp.D_q - policy.D_q) / max(abs(esp.D_q), eps)              # fractional tail-latency reduction
    d_energy = (esp.energy - policy.energy) / max(abs(esp.energy), eps)   # fractional energy saving
    inc_wrong = max(0.0, policy.F_wrong - esp.F_wrong)       # validity hazard (penalised only on increase)
    inc_split = max(0.0, policy.F_split - esp.F_split)
    return (profile.c_d * d_deadline + profile.c_T * d_tail + profile.c_E * d_energy
            - profile.c_w * inc_wrong - profile.c_s * inc_split)


def is_eligible(policy: PolicyOutcome, profile: HazardProfile) -> bool:
    """A policy is eligible only if its wrong AND split UCB are within the profile budgets (the hard
    feasibility gate -- reliability is never traded for liveness, constraint #4/#5)."""
    return policy.F_wrong_ucb <= profile.eps_w and policy.F_split_ucb <= profile.eps_s


@dataclass(frozen=True)
class SelectionResult:
    selected: str
    benefit: float                       # B of the selected policy vs ESP (0 if ESP)
    eligible: tuple[str, ...]
    benefits: dict                       # name -> B for every candidate (eligible or not)
    esp_eligible: bool


def select_policy(esp: PolicyOutcome, candidates, profile: HazardProfile) -> SelectionResult:
    """Select the eligible policy with the largest hazard benefit; ESP is the default fallback.

    ``candidates`` are the non-ESP alternatives (e.g. CDQ2, Guarded-CDQ2). ESP is always considered (its
    B is 0 by definition). If ESP itself is ineligible (its own wrong/split UCB exceed the budget -- the
    whole scene is reliability-infeasible) the selection still returns ESP as the least-unsafe default but
    flags ``esp_eligible=False`` so the caller can report the scene as infeasible.
    """
    benefits = {esp.name: 0.0}
    for c in candidates:
        benefits[c.name] = hazard_benefit(esp, c, profile)
    eligible = tuple([esp.name] * is_eligible(esp, profile)
                     + [c.name for c in candidates if is_eligible(c, profile)])

    # among eligible alternatives, the best positive-benefit one beats ESP; else ESP (default).
    best_name, best_B = esp.name, 0.0
    for c in candidates:
        if is_eligible(c, profile) and benefits[c.name] > best_B:
            best_name, best_B = c.name, benefits[c.name]
    return SelectionResult(selected=best_name, benefit=best_B, eligible=eligible, benefits=benefits,
                           esp_eligible=is_eligible(esp, profile))
