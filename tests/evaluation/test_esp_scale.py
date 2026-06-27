"""G-ESP-PERFORMANCE-SCALE (Guarded-CDQ2 round, Phase 3): the performance-scale harness.

Locks the harness contract at TINY scale (fast): trained ESP/ESD-GNN checkpoints, dynamic-MC basin
outcomes (the judge -- not runtime), scale-regret / feasibility-retention, and the fixed-protocol vs
fixed-service-profile calibration. The heavy multi-scale headline run lives in the evidence script.
"""

import math

import pytest
import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation import esp_scale as es
from src.metrics import schema

TINY = (5, 5, 3)          # 120 nodes -- the smallest grid the canonical episode accepts (degree >= k)
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=5)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=5)


# ---------------------------------------------------------------- scaling + calibration (pure)
def test_grid_for_target_N_monotone():
    g_small = es.grid_for_target_N(100)
    g_big = es.grid_for_target_N(10000)
    n = lambda g: ((g[0] - 1) * g[1] + g[0] * (g[1] - 1)) * g[2]
    assert n(g_small) < n(g_big)


def test_calibrated_profile_modes():
    base = PROFILE
    assert es.calibrated_profile(base, 5000, mode="fixed_protocol") is base
    cal = es.calibrated_profile(base, 1920, mode="fixed_service_profile", base_N=120)
    assert cal.max_poll_epochs > base.max_poll_epochs        # sqrt(1920/120)=4x -> R_d scales up
    cal_same = es.calibrated_profile(base, 120, mode="fixed_service_profile", base_N=120)
    assert cal_same.max_poll_epochs == base.max_poll_epochs   # at base_N, unchanged
    with pytest.raises(ValueError):
        es.calibrated_profile(base, 100, mode="bogus")


def test_scale_regret_and_retention_math():
    assert es.scale_regret(0.30, 0.25) == pytest.approx(0.05)
    # shared matches expert -> normalized regret 0; shared == heuristic -> 1
    assert es.normalized_scale_regret(0.25, 0.25, 0.40) == pytest.approx(0.0, abs=1e-6)
    assert es.normalized_scale_regret(0.40, 0.25, 0.40) == pytest.approx(1.0, abs=1e-6)
    assert math.isnan(es.normalized_scale_regret(math.inf, 0.25, 0.40))   # infeasible -> nan
    assert es.feasibility_retention(5, 5) == pytest.approx(1.0, abs=1e-6)
    assert es.feasibility_retention(3, 5) == pytest.approx(0.6, abs=1e-6)


def test_feasible_ucb_and_cost_gate():
    feas = schema.macro_block(0.95, 0.0, 0.0, 0.05,
                              ci={"macro_F_wrong": (0.0, 0.0005), "macro_F_split": (0.0, 0.0008)})
    infeas = schema.macro_block(0.7, 0.2, 0.05, 0.05,
                                ci={"macro_F_wrong": (0.18, 0.22), "macro_F_split": (0.04, 0.06)})
    assert es.feasible_ucb(feas, PROFILE)
    assert not es.feasible_ucb(infeas, PROFILE)
    assert es.headline_cost(feas, PROFILE) == pytest.approx(0.05)
    assert math.isinf(es.headline_cost(infeas, PROFILE))   # infeasible excluded
    # point feasibility: zero observed wrong/split is point-feasible even when the UCB is loose
    loose = schema.macro_block(0.9, 0.0, 0.0, 0.1,
                               ci={"macro_F_wrong": (0.0, 0.02), "macro_F_split": (0.0, 0.02)})
    assert es.feasible_point(loose, PROFILE) and not es.feasible_ucb(loose, PROFILE)
    assert es.headline_cost(loose, PROFILE) == pytest.approx(0.1)             # point gate passes
    assert math.isinf(es.headline_cost(loose, PROFILE, ucb=True))            # UCB gate fails (loose)


# ---------------------------------------------------------------- checkpoint hashing
def test_checkpoint_hash_deterministic_and_sensitive():
    from src.models import ESDGNN, ESDGNNConfig
    torch.manual_seed(0)
    m = ESDGNN(ESDGNNConfig(hidden_dim=8, r=4, n_enc=1, n_refine=0, k=3, use_cdq=False)).double()
    h1 = es.checkpoint_hash(m)
    assert h1 == es.checkpoint_hash(m)                       # deterministic
    assert len(h1) == 64
    with torch.no_grad():
        next(m.parameters()).add_(1.0)
    assert es.checkpoint_hash(m) != h1                       # sensitive to params


def test_policy_factory_kinds():
    from src.models import ESDGNN, ESDGNNConfig
    sc, _ = es.build_scale_instance(TINY, 0)
    assert es.policy_factory("uniform_esp")(sc).name
    assert es.policy_factory("distance")(sc).name
    m = ESDGNN(ESDGNNConfig(hidden_dim=8, n_enc=1, n_refine=0, k=3, use_cdq=False)).double()
    pol = es.policy_factory("esd_gnn", model=m)(sc)
    assert pol.query_law == "esp"                            # use_cdq=False -> ESP law
    with pytest.raises(ValueError):
        es.policy_factory("esd_gnn")                         # needs a model
    with pytest.raises(ValueError):
        es.policy_factory("nope")


# ---------------------------------------------------------------- end-to-end (tiny, real MC)
def test_train_checkpoint_and_evaluate_basins():
    """Train a tiny ESP/ESD-GNN checkpoint on the full physical chain, then evaluate REAL dynamic-MC
    basin outcomes -- they must form a valid macro block (sum to 1) with CIs on every outcome."""
    ck = es.train_esp_checkpoint([TINY], seed=0, profile=PROFILE, proto=PROTO, phy=PHY,
                                 steps=3, scenes_per_grid=1, hidden_dim=8, link_override=None)
    assert len(ck["checkpoint_hash"]) == 64 and ck["model_seed"] == 0
    pol = es.policy_factory("esd_gnn", model=ck["model"])
    ev = es.evaluate_macro(TINY, [0, 1], pol, PROFILE, PROTO, PHY, trials=60, link_override=None)
    schema.validate_macro_block(ev.macro)                    # sums to 1, namespaced, CIs present
    assert ev.n_pool == 120
    assert "macro_F_wrong_ci" in ev.macro and "macro_P_correct_ci" in ev.macro
    assert 0.0 <= ev.P_correct <= 1.0


def test_different_seeds_give_different_checkpoints():
    c0 = es.train_esp_checkpoint([TINY], seed=0, profile=PROFILE, proto=PROTO, phy=PHY,
                                 steps=2, scenes_per_grid=1, hidden_dim=8)
    c1 = es.train_esp_checkpoint([TINY], seed=1, profile=PROFILE, proto=PROTO, phy=PHY,
                                 steps=2, scenes_per_grid=1, hidden_dim=8)
    assert c0["checkpoint_hash"] != c1["checkpoint_hash"]    # distinct model seeds -> distinct nets
