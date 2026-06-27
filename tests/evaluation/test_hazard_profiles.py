"""G-HAZARD-PROFILES (Guarded-CDQ2 round, Phase 6): hazard-weighted policy selection.

Locks the contract: the net benefit B is a feasibility-gated function of the per-policy macrostate
outcomes vs ESP, and policy selection changes RATIONALLY with the hazard weights (safety-first -> ESP;
deadline-critical -> CDQ2/Guarded-CDQ2; etc., spec §5.5). Uses the GS5 enable-regime numbers as fixtures.
"""

import pytest

from src.config.hazard_profile import STANDARD_PROFILES, HazardProfile
from src.evaluation.hazard_utility import PolicyOutcome, hazard_benefit, is_eligible, select_policy

# GS5 enable regime (err=0.20, corr=0.10), dynamic-MC outcomes:
ESP = PolicyOutcome("ESP", F_wrong=0.023, F_split=0.0, F_deadline=0.037,
                    F_wrong_ucb=0.047, F_split_ucb=0.020, D_q=0.10, energy=1.0)
CDQ2 = PolicyOutcome("CDQ2", F_wrong=0.027, F_split=0.0, F_deadline=0.017,
                     F_wrong_ucb=0.052, F_split_ucb=0.020, D_q=0.07, energy=1.15)


# ---------------------------------------------------------------- benefit + eligibility
def test_esp_benefit_is_zero():
    for prof in STANDARD_PROFILES:
        assert hazard_benefit(ESP, ESP, prof) == 0.0


def test_deadline_benefit_positive_under_high_c_d():
    dl = next(p for p in STANDARD_PROFILES if p.name == "deadline_critical")
    # deadline gain 0.020 * c_d(1000)=20 minus wrong increase 0.004 * c_w(50)=0.2 -> strongly positive
    assert hazard_benefit(ESP, CDQ2, dl) > 10


def test_wrong_penalty_dominates_under_safety_first():
    sf = next(p for p in STANDARD_PROFILES if p.name == "safety_first")
    # wrong increase 0.004 * c_w(1000)=4 dominates deadline gain 0.020 * c_d(10)=0.2 -> negative
    assert hazard_benefit(ESP, CDQ2, sf) < 0


def test_eligibility_gate():
    strict = HazardProfile("x", c_w=1, c_s=1, c_d=1, eps_w=1e-3, eps_s=1e-3)
    loose = HazardProfile("y", c_w=1, c_s=1, c_d=1, eps_w=0.10, eps_s=0.10)
    assert not is_eligible(CDQ2, strict)        # Fw_ucb 0.052 > 1e-3
    assert is_eligible(CDQ2, loose)             # 0.052 <= 0.10


# ---------------------------------------------------------------- rational selection (spec §5.5)
def test_safety_first_selects_esp():
    sf = next(p for p in STANDARD_PROFILES if p.name == "safety_first")
    r = select_policy(ESP, [CDQ2], sf)
    assert r.selected == "ESP" and r.benefit == 0.0      # CDQ2 ineligible (strict eps) + penalised

def test_deadline_critical_selects_cdq2():
    dl = next(p for p in STANDARD_PROFILES if p.name == "deadline_critical")
    r = select_policy(ESP, [CDQ2], dl)
    assert r.selected == "CDQ2" and r.benefit > 0        # eligible (loose eps) + huge deadline value


def test_fail_safe_selects_esp_even_if_eligible():
    # make CDQ2 eligible but give a low deadline value -> the wrong penalty still wins
    fs = HazardProfile("fail_safe", c_w=1000.0, c_s=1000.0, c_d=1.0, eps_w=0.10, eps_s=0.10)
    r = select_policy(ESP, [CDQ2], fs)
    assert r.selected == "ESP"


def test_guarded_cdq2_is_robust_choice():
    """Guarded-CDQ2 disables to ESP under a strict eps (so it's always eligible) and equals CDQ2 under a
    loose eps -- modelled here as two outcomes; selection picks the eligible one with the best B."""
    sf = next(p for p in STANDARD_PROFILES if p.name == "safety_first")
    dl = next(p for p in STANDARD_PROFILES if p.name == "deadline_critical")
    guarded_off = PolicyOutcome(**{**ESP.__dict__, "name": "Guarded-CDQ2"})    # outcome == ESP
    guarded_on = CDQ2                                                                          # == CDQ2 outcome
    assert select_policy(ESP, [guarded_off], sf).selected in ("ESP", "Guarded-CDQ2")          # B=0 either way
    assert select_policy(ESP, [guarded_on], dl).selected == "CDQ2"                             # captures gain


def test_energy_term_penalises_higher_energy():
    # an energy-constrained profile: CDQ2 uses MORE energy (1.15 vs 1.0) -> energy term is negative
    en = next(p for p in STANDARD_PROFILES if p.name == "energy_constrained")
    b = hazard_benefit(ESP, CDQ2, en)
    # energy penalty c_E(100)*0.15 = 15 dominates the deadline gain c_d(10)*0.02=0.2 -> negative
    assert b < 0


def test_all_five_profiles_present():
    names = {p.name for p in STANDARD_PROFILES}
    assert names == {"safety_first", "balanced", "deadline_critical", "fail_safe_available",
                     "energy_constrained"}
