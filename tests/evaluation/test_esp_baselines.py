"""G-ESP-BASELINE-ORACLE: baseline factory + the direct edge-logit topology-headroom oracle."""

import pytest

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.esp_baselines import DEPLOYABLE_BASELINES, direct_edge_logit_oracle, make_baseline
from src.evaluation.esp_scale import build_scale_instance
from src.evaluation.esp_training import validate_macro_analytic
from src.metrics import schema

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)


def test_make_baseline_kinds():
    scene, _ = build_scale_instance((5, 5, 3), 0)
    for kind in DEPLOYABLE_BASELINES:
        pol = make_baseline(kind, scene)
        assert pol.name and getattr(pol, "query_law", "esp") == "esp"
    with pytest.raises(ValueError):
        make_baseline("nope", scene)


def test_oracle_upper_bounds_distance():
    """The direct edge-logit oracle (inits from distance, optimises analytic P_correct) must end at
    least as good as the distance heuristic -- it estimates the per-scene topology headroom."""
    scene, ev = build_scale_instance((5, 5, 3), 0, scenario="matched_marginal_high", base_node_err=0.35,
                                     corr_strength=0.25)
    dist = validate_macro_analytic(lambda sc: make_baseline("distance", sc), [(scene, ev)], PROFILE,
                                   PROTO, PHY, link_override=None)
    oracle = direct_edge_logit_oracle(scene, ev, PROFILE, PROTO, PHY, steps=25, lr=0.1, link_override=None)
    schema.validate_macro_block(oracle)
    # oracle >= distance (it started there and ascended); allow tiny numerical slack
    assert oracle["macro_P_correct"] >= dist["macro_P_correct"] - 1e-3
