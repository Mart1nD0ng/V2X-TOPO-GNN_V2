"""G-ESP-TRAINING-BUDGET (esp_performance_scale_v2): the training-budget curve harness.

Pure-logic tests for the budget aggregation + selection; one tiny full-physics integration test that the
canonical training loop produces a held-out validation curve (GNN vs distance) at the budget points.
"""

import pytest

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation import esp_training as et
from src.evaluation.esp_training import _Curve
from src.metrics import schema

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)


def _curve(seed, vals):
    """vals: {budget: (gnn_Pc, dist_Pc)} -> a synthetic _Curve."""
    curve = {b: {"gnn": schema.macro_block(g, 0.0, 0.0, 1 - g),
                 "distance": schema.macro_block(d, 0.0, 0.0, 1 - d)} for b, (g, d) in vals.items()}
    return _Curve(seed, f"ck{seed}", curve, [1.0, 0.5], [0.1, 0.2])


# ---------------------------------------------------------------- budget aggregation (pure)
def test_budget_improvement_detects_improvement():
    # GNN improves 0.5 -> 0.8 across budget, beats distance (0.6) at full
    curves = [_curve(s, {5: (0.50, 0.60), 20: (0.70, 0.60), 40: (0.80, 0.60)}) for s in range(3)]
    r = et.budget_improvement(curves)
    assert r["improves_over_pilot"] and r["beats_distance_at_full"]
    assert r["budget_points"] == [5, 20, 40] and r["pilot"] == 5 and r["full"] == 40


def test_budget_improvement_detects_plateau_and_no_beat():
    # GNN flat at 0.55, never beats distance 0.60 -> negative result (interpretation §13.3)
    curves = [_curve(s, {5: (0.55, 0.60), 20: (0.555, 0.60), 40: (0.55, 0.60)}) for s in range(3)]
    r = et.budget_improvement(curves)
    assert not r["improves_over_pilot"] and r["plateaued"] and not r["beats_distance_at_full"]


def test_select_best_budget_prefers_higher_Pc_then_cheaper():
    curves = [_curve(s, {5: (0.5, 0.6), 20: (0.8, 0.6), 40: (0.8, 0.6)}) for s in range(2)]
    assert et.select_best_budget(curves) == 20            # tie at 0.8 -> the cheaper budget


# ---------------------------------------------------------------- integration (tiny, full physics)
def test_train_with_curve_produces_validation_curve():
    c = et.train_with_curve([(5, 5, 3)], seed=0, profile=PROFILE, proto=PROTO, phy=PHY, total_steps=3,
                            budget_points=[1, 3], val_grids=[(5, 5, 3)], val_seeds=[0],
                            scenario="iid", base_node_err=0.2, hidden_dim=8, link_override=None)
    assert set(c.curve) == {1, 3} and len(c.loss_history) == 3
    for b in (1, 3):
        schema.validate_macro_block(c.curve[b]["gnn"])
        schema.validate_macro_block(c.curve[b]["distance"])
    assert len(c.checkpoint_hash) == 64
