"""G-ETA-RISK-LIVENESS (Guarded-CDQ2 round, Phase 4): the eta-risk-liveness curve harness.

Locks the contract at tiny scale: eta=0 is exactly ESP, the sweep produces valid macro blocks at
every eta, and the mass-shift classifier names HOW probability mass moves (deadline->correct vs
deadline->wrong vs split->correct vs none) -- always surfacing a wrong-risk increase (constraint #12).
"""

import pytest

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation import eta_curve as ec
from src.metrics import schema

TINY = (5, 5, 3)          # 120 nodes
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)


# ---------------------------------------------------------------- mass-shift classifier (pure)
def _macro(P, Fw, Fs, Fd):
    return schema.macro_block(P, Fw, Fs, Fd)


def test_classify_deadline_to_correct():
    esp = _macro(0.80, 0.02, 0.03, 0.15)
    cdq2 = _macro(0.90, 0.02, 0.03, 0.05)         # Fd down, Pc up, no wrong increase
    ms = ec.classify_mass_shift(esp, cdq2)
    assert ms.label == "deadline->correct" and not ms.wrong_increased
    assert ms.d_F_deadline < 0 and ms.d_P_correct > 0


def test_classify_deadline_to_wrong_flags_validity_cost():
    esp = _macro(0.80, 0.02, 0.03, 0.15)
    cdq2 = _macro(0.83, 0.09, 0.03, 0.05)         # Fd down BUT Fw up -> validity cost
    ms = ec.classify_mass_shift(esp, cdq2)
    assert ms.wrong_increased                      # constraint #12: never hidden
    assert ms.label in ("mixed", "deadline->wrong")
    assert ms.d_F_wrong > 0


def test_classify_none_when_static():
    esp = _macro(0.80, 0.02, 0.03, 0.15)
    ms = ec.classify_mass_shift(esp, esp)
    assert ms.label == "none" and not ms.wrong_increased


def test_classify_split_to_correct():
    esp = _macro(0.70, 0.02, 0.18, 0.10)
    cdq2 = _macro(0.85, 0.02, 0.03, 0.10)         # Fs down, Pc up, deadline unchanged
    ms = ec.classify_mass_shift(esp, cdq2)
    assert ms.label == "split->correct"


def test_eta_grid_spec():
    assert ec.ETA_GRID[0] == 0.0 and ec.ETA_GRID[-1] == 16.0 and len(ec.ETA_GRID) == 8


# ---------------------------------------------------------------- sweep (tiny, real MC)
def test_eta_sweep_eta0_is_esp_and_blocks_valid():
    """eta=0 must be exactly ESP (same as the bare distance policy); every eta yields a valid macro
    block. Fixed-link keeps it fast; 2 etas keep the test bounded."""
    sweep = ec.eta_sweep(TINY, [0], scenario="matched_marginal_high", base_node_err=0.35,
                         corr_strength=0.3, profile=PROFILE, proto=PROTO, phy=PHY, trials=40,
                         link_override=0.85, eta_grid=(0.0, 8.0))
    assert set(sweep) == {0.0, 8.0}
    for eta, macro in sweep.items():
        schema.validate_macro_block(macro)
        assert "macro_F_wrong_ci" in macro

    # eta=0 == bare ESP under identical CRN (same scene seed) -> basins match closely
    from src.evaluation.esp_scale import build_scale_instance, policy_factory, evaluate_macro
    esp = evaluate_macro(TINY, [0], policy_factory("distance", distance_beta=0.04), PROFILE, PROTO,
                         PHY, trials=40, scenario="matched_marginal_high", base_node_err=0.35,
                         link_override=0.85)
    assert abs(sweep[0.0]["macro_P_correct"] - esp.macro["macro_P_correct"]) < 1e-9
