"""G-GUARDED-CDQ2 (Guarded-CDQ2 round, Phase 5): the reliability/deadline guard.

Locks the guard contract: ESP is the default; CDQ2 diversity is enabled ONLY with reliability slack
AND deadline pressure (spec §4). The hard guard never trades reliability (constraint #4/#5); the soft
guard is its differentiable relaxation; a guarded policy that disables falls back to ESP exactly.
"""

import pytest
import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.policies.guarded_cdq2 import (
    GuardConfig, GuardedCDQ2Policy, guard_factor, hard_guard_eta, reliability_slack, soft_guard_eta,
)

CFG = GuardConfig(eps_w=0.05, eps_s=0.05, delta_w=0.01, delta_s=0.01, delta_d=0.05,
                  T_w=0.01, T_s=0.01, T_d=0.01)


# ---------------------------------------------------------------- guard math (pure)
def test_reliability_slack():
    m_w, m_s = reliability_slack(0.02, 0.03, 0.05, 0.05)
    assert m_w == pytest.approx(0.03) and m_s == pytest.approx(0.02)


def test_hard_guard_enables_with_slack_disables_without():
    # ample slack (Fw,Fs well below eps) -> keep eta_raw
    assert hard_guard_eta(8.0, Fw_ucb=0.02, Fs_ucb=0.02, cfg=CFG) == 8.0
    # wrong UCB too close to eps (slack < delta_w) -> fall back to ESP
    assert hard_guard_eta(8.0, Fw_ucb=0.045, Fs_ucb=0.02, cfg=CFG) == 0.0
    # split UCB over budget -> fall back to ESP
    assert hard_guard_eta(8.0, Fw_ucb=0.02, Fs_ucb=0.06, cfg=CFG) == 0.0


def test_soft_guard_factor_bounds_and_limits():
    # high slack + high deadline pressure -> G ~ 1
    g_hi = guard_factor(Fw_ucb=0.0, Fs_ucb=0.0, p_d=0.5, cfg=CFG)
    assert 0.0 <= g_hi <= 1.0 and g_hi > 0.95
    # reliability slack gone (Fw above eps) -> G ~ 0 regardless of pressure
    g_unsafe = guard_factor(Fw_ucb=0.10, Fs_ucb=0.0, p_d=0.5, cfg=CFG)
    assert g_unsafe < 0.05
    # no deadline pressure -> G ~ 0 (don't risk validity for a comfortably-met deadline)
    g_nopressure = guard_factor(Fw_ucb=0.0, Fs_ucb=0.0, p_d=0.0, cfg=CFG)
    assert g_nopressure < 0.05
    assert soft_guard_eta(8.0, Fw_ucb=0.0, Fs_ucb=0.0, p_d=0.5, cfg=CFG) == pytest.approx(8.0 * g_hi)


def test_guard_config_from_profile():
    prof = ConsensusServiceProfile.urban_default()
    cfg = GuardConfig.from_profile(prof, delta_frac=0.2)
    assert cfg.eps_w == prof.max_wrong_basin_probability
    assert cfg.delta_w == pytest.approx(0.2 * prof.max_wrong_basin_probability)


# ---------------------------------------------------------------- guarded policy wiring
def _scene_and_div():
    from src.evaluation.esp_scale import build_scale_instance
    from src.evaluation.eta_curve import cdq2_diversity_for
    scene, ev = build_scale_instance((5, 5, 3), 0, scenario="matched_marginal_high", base_node_err=0.3)
    div, r = cdq2_diversity_for(ev, use_sensor=True, use_map=False)
    return scene, ev, div, r


def test_guarded_policy_disables_to_esp():
    from src.sampling.baseline_policies import DistanceQueryPolicy
    _, _, div, r = _scene_and_div()
    base = DistanceQueryPolicy(beta_per_m=0.04)
    # wrong UCB far above eps -> guard disables -> policy IS ESP
    g = GuardedCDQ2Policy(base, r=r, eta_raw=8.0, diversity=div, cfg=CFG, mode="hard",
                          Fw_ucb=0.5, Fs_ucb=0.0, p_d=0.5)
    assert not g.guard_active and g.eta_eff == 0.0
    assert g.query_law == "esp"
    assert g._inner is base


def test_guarded_policy_enables_cdq2_with_slack():
    from src.sampling.baseline_policies import DistanceQueryPolicy
    _, _, div, r = _scene_and_div()
    base = DistanceQueryPolicy(beta_per_m=0.04)
    g = GuardedCDQ2Policy(base, r=r, eta_raw=8.0, diversity=div, cfg=CFG, mode="hard",
                          Fw_ucb=0.0, Fs_ucb=0.0, p_d=0.5)
    assert g.guard_active and g.eta_eff == 8.0
    assert g.query_law == "cdq2"
    assert float(getattr(g, "eta")) == 8.0          # proxies to the inner CDQ2Policy


def test_guarded_disabled_matches_esp_in_mc():
    """A disabled guard must reproduce ESP outcomes bit-for-bit under common random numbers."""
    from src.metrics.participation import uniform_participation
    from src.sampling.baseline_policies import DistanceQueryPolicy
    from src.validation import run_dynamic_mc
    scene, ev, div, r = _scene_and_div()
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    omega = uniform_participation(scene.num_nodes)
    base = DistanceQueryPolicy(beta_per_m=0.04)
    # fresh generator (same seed) per run -> common random numbers (NOT a shared, state-advancing object)
    args = dict(num_trials=40, link_override=0.85, service_profile=prof, participation=omega)
    g_off = GuardedCDQ2Policy(base, r=r, eta_raw=8.0, diversity=div, cfg=CFG, mode="hard",
                              Fw_ucb=0.9, Fs_ucb=0.0, p_d=0.5)          # disabled
    esp = run_dynamic_mc(scene, ev, base, proto, phy,
                         generator=torch.Generator().manual_seed(0), **args)
    guarded = run_dynamic_mc(scene, ev, g_off, proto, phy,
                             generator=torch.Generator().manual_seed(0), **args)
    assert guarded.basin_P_correct == pytest.approx(esp.basin_P_correct, abs=1e-12)
    assert guarded.basin_F_wrong == pytest.approx(esp.basin_F_wrong, abs=1e-12)
