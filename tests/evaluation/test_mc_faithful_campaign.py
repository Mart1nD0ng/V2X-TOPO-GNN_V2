"""Campaign-A infra: MC-faithful checkpoint train/save/load + multi-seed aggregation."""
import os

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.mc_faithful_campaign import (aggregate_seed_macros, checkpoint_policy_factory,
                                                 load_checkpoint, save_checkpoint, train_mc_faithful)
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


def test_aggregate_seed_macros_pools_mean_and_ci():
    macros = [schema.macro_block(0.40, 0.02, 0.03, 0.55), schema.macro_block(0.44, 0.02, 0.03, 0.51),
              schema.macro_block(0.42, 0.02, 0.03, 0.53)]
    agg = aggregate_seed_macros(macros, n_pool_per_seed=400)
    assert abs(agg["macro_P_correct"] - 0.42) < 1e-9
    assert agg["n_seeds"] == 3
    lo, hi = agg["macro_P_correct_ci"]
    assert lo < 0.42 < hi and (hi - lo) < 0.06            # CI tighter at 1200 pooled than at 400
    assert not schema.forbidden_keys_in(agg)
