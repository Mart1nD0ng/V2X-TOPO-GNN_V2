"""Campaign-A infra: MC-faithful checkpoint train/save/load + multi-seed aggregation."""
import os

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.mc_faithful_campaign import (aggregate_seed_macros, checkpoint_policy_factory,
                                                 load_checkpoint, paired_seed_separation,
                                                 reliability_status, save_checkpoint,
                                                 seed_level_bootstrap_ci, train_mc_faithful, ungated_cost)
from src.metrics import schema

_PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
_PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
_PROF = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)


def test_checkpoint_round_trip_preserves_hash_and_policy(tmp_path):
    torch.manual_seed(0)
    ckpt = train_mc_faithful(0, (5, 5, 3), profile=_PROF, proto=_PROTO, phy=_PHY,
                             scenario="matched_marginal_high", base_node_err=0.35, corr_strength=0.25,
                             steps=2, trials=20, hidden_dim=8)
    p = os.path.join(tmp_path, "ckpt.pt")
    save_checkpoint(ckpt, p)
    loaded = load_checkpoint(p)                                  # hash-verified inside
    assert loaded.checkpoint_hash == ckpt.checkpoint_hash
    assert loaded.train_meta["corr_strength"] == 0.25
    assert len(ckpt.history["mc_P_correct"]) == 2
    # the loaded model reproduces the trained model's quality law on a scene
    from src.evaluation.esp_scale import build_scale_instance
    scene, _ = build_scale_instance((5, 5, 3), 0, scenario="matched_marginal_high", base_node_err=0.35,
                                    corr_strength=0.25)
    pol_a = checkpoint_policy_factory(ckpt)(scene)
    pol_b = checkpoint_policy_factory(loaded)(scene)
    qa = pol_a.log_weights(scene) if hasattr(pol_a, "log_weights") else None
    if qa is not None:
        assert torch.allclose(qa, pol_b.log_weights(scene))


def test_snapshot_is_trajectory_preserving():
    """A step-S snapshot taken DURING a longer run must equal a standalone step-S run (same init/seed):
    snapshots don't perturb the optimizer, so one 150-step run yields the budget axis for free (A2)."""
    from src.models import ESDGNN
    from src.evaluation.esp_scale import _esp_config, build_scale_instance
    from src.optimization.mc_reinforce import train_esp_reinforce
    inst = [build_scale_instance((5, 5, 3), 1000, scenario="matched_marginal_high", base_node_err=0.35,
                                 corr_strength=0.25)]
    torch.manual_seed(7); long = ESDGNN(_esp_config(8, _PROF.k)).double()
    torch.manual_seed(7); short = ESDGNN(_esp_config(8, _PROF.k)).double()
    res_long = train_esp_reinforce(long, inst, _PROTO, _PHY, _PROF, steps=4, trials=20, lr=1e-2,
                                   base_seed=50, snapshot_steps=(2,))
    res_short = train_esp_reinforce(short, inst, _PROTO, _PHY, _PROF, steps=2, trials=20, lr=1e-2,
                                    base_seed=50)
    snap2 = res_long["snapshots"][2]
    for k, v in short.state_dict().items():
        assert torch.allclose(snap2[k], v), k                      # step-2 snapshot == standalone step-2 run


def test_ungated_cost_and_reliability_are_separate():
    feasible = schema.macro_block(0.42, 0.0, 0.0, 0.58,
                                  ci={"macro_F_wrong": (0.0, 0.0005), "macro_F_split": (0.0, 0.0005)})
    infeasible = schema.macro_block(0.42, 0.0525, 0.0, 0.5275,
                                    ci={"macro_F_wrong": (0.04, 0.07), "macro_F_split": (0.0, 0.001)})
    # ungated cost ignores feasibility entirely (the pre-registered rule): both give J = 1 - 0.42
    assert abs(ungated_cost(feasible) - 0.58) < 1e-9 and abs(ungated_cost(infeasible) - 0.58) < 1e-9
    assert reliability_status(infeasible, _PROF)["feasible_wrong_ucb"] is False   # F_wrong UCB 0.07 > 1e-3
    assert reliability_status(feasible, _PROF)["feasible_wrong_ucb"] is True


def test_seed_level_separation():
    # trained clearly above ref at every seed -> separated, trained_better
    sep = paired_seed_separation([0.41, 0.42, 0.40, 0.43, 0.41], [0.36, 0.37, 0.36, 0.37, 0.36])
    assert sep["separated"] and sep["trained_better"] and sep["mean_diff"] > 0.03
    # overlapping (sign flips across seeds) -> NOT separated
    noisy = paired_seed_separation([0.41, 0.34, 0.45, 0.33, 0.44], [0.40, 0.39, 0.38, 0.41, 0.37])
    assert not noisy["separated"]
    boot = seed_level_bootstrap_ci([0.415, 0.405, 0.41, 0.40, 0.42])
    assert 0.40 < boot["mean"] < 0.42 and boot["ci"][0] < boot["mean"] < boot["ci"][1]


def test_aggregate_seed_macros_pools_mean_and_ci():
    macros = [schema.macro_block(0.40, 0.02, 0.03, 0.55), schema.macro_block(0.44, 0.02, 0.03, 0.51),
              schema.macro_block(0.42, 0.02, 0.03, 0.53)]
    agg = aggregate_seed_macros(macros, n_pool_per_seed=400)
    assert abs(agg["macro_P_correct"] - 0.42) < 1e-9
    assert agg["n_seeds"] == 3
    lo, hi = agg["macro_P_correct_ci"]
    assert lo < 0.42 < hi and (hi - lo) < 0.06            # CI tighter at 1200 pooled than at 400
    assert not schema.forbidden_keys_in(agg)
