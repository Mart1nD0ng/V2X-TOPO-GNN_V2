"""G11 -- per-trial energy + CVaR tail-latency from the dynamic MC (spec §4.5/§7.4).

Acceptance: the MC reports a positive mean per-trial energy and an upper-tail CVaR for latency and
energy that dominates the mean (a tail is never below the average); the CVaR helper is exact on a
known input; metrics are reproducible.
"""

import torch

from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
)
from src.sampling import DistanceQueryPolicy
from src.validation import run_dynamic_mc
from src.validation.dynamic_mc import _cvar_upper

PHY = RoundPhysicsConfig(subchannels=8, slots_per_window=30)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=12)


def _scene(seed=0):
    return build_manhattan_scene(3, 3, 4, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                                 generator=torch.Generator().manual_seed(seed))


def test_cvar_upper_matches_hand_value():
    x = torch.tensor([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=torch.float64)
    # worst-10% tail (level 0.9): quantile_0.9 = 9.1 -> only value >= 9.1 is 10 -> mean 10
    assert abs(_cvar_upper(x, 0.9) - 10.0) < 1e-9
    # level 0.5 -> values >= median(=5.5) are 6..10 -> mean 8
    assert abs(_cvar_upper(x, 0.5) - 8.0) < 1e-9
    assert _cvar_upper(x, 0.9) >= float(x.mean())            # tail dominates the mean


def test_mc_reports_positive_energy_and_tail_dominant_cvar():
    scene = _scene()
    ev = build_scenario("one_biased_region", scene, base_node_err=0.05, region_bias=0.9)
    pol = DistanceQueryPolicy(beta_per_m=0.03)
    r = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=1500,
                       generator=torch.Generator().manual_seed(1))      # FULL physics (no override)
    assert r.mean_energy > 0.0 and r.mean_energy == r.mean_energy        # finite, positive
    assert r.energy_cvar >= r.mean_energy                                 # tail >= mean
    assert r.latency_cvar >= 0.0 and r.latency_cvar == r.latency_cvar
    assert r.cvar_level == 0.9


def test_energy_and_latency_reproducible():
    scene = _scene()
    ev = build_scenario("one_biased_region", scene, base_node_err=0.05, region_bias=0.9)
    pol = DistanceQueryPolicy(beta_per_m=0.03)
    a = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=800,
                       generator=torch.Generator().manual_seed(3))
    b = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=800,
                       generator=torch.Generator().manual_seed(3))
    assert a.mean_energy == b.mean_energy and a.latency_cvar == b.latency_cvar
