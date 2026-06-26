"""Phase 7 -- topology-optimization ceiling (stop-condition #2).

Acceptance (spec plan §10): a direct per-scene edge-logit optimizer must SIGNIFICANTLY beat
the heuristic policies on the reliability objective, and the advantage must survive the
independent dynamic MC. Demonstrated on a correlated (one-biased-region) scenario where the
peer-selection lever genuinely exists. (Perfect link isolates the topology/evidence lever.)
"""

import torch

from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene, build_scenario
from src.optimization.topology_oracle import optimize_logweight_topology, oracle_vs_heuristics

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=14)


def _scene():
    return build_manhattan_scene(3, 3, 3, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                                 generator=torch.Generator().manual_seed(0))


def test_oracle_beats_heuristics_and_survives_mc():
    """The viability gate: oracle >> heuristics analytically AND under the independent MC."""
    scene = _scene()
    ev = build_scenario("one_biased_region", scene, base_node_err=0.05, region_bias=0.95)
    out = oracle_vs_heuristics(scene, ev, PROTO, PHY, steps=70, lr=0.3,
                               link_override=1.0, mc_trials=3000, seed=0)
    a = out["analytic"]
    best = out["best_heuristic"]
    # significant analytic gain: oracle objective at most half the best heuristic's
    assert a["oracle"]["objective"] < 0.5 * a[best]["objective"]
    assert out["analytic_gain"] > 0.02
    # MC confirmation: oracle F_wrong significantly below the best heuristic, CIs separated
    mc = out["mc"]
    assert mc["oracle"]["F_wrong"] < mc[best]["F_wrong"] - 0.02
    assert mc["oracle"]["F_wrong_ci"][1] < mc[best]["F_wrong_ci"][0]   # non-overlapping CIs
    assert out["mc_gain"] > 0.02                                       # real, significant lever


def test_distance_heuristic_worse_than_uniform_under_correlation():
    """The §9.1 tension: concentrating polls on nearby (same-region, redundant/biased) peers
    is WORSE than uniform under correlated evidence -- the reason diversity matters."""
    scene = _scene()
    ev = build_scenario("one_biased_region", scene, base_node_err=0.05, region_bias=0.95)
    out = oracle_vs_heuristics(scene, ev, PROTO, PHY, steps=1, lr=0.1, link_override=1.0, mc_trials=0)
    a = out["analytic"]
    assert a["distance"]["F_wrong"] > a["uniform"]["F_wrong"]


def test_oracle_loss_decreases():
    scene = _scene()
    ev = build_scenario("one_biased_region", scene, base_node_err=0.05, region_bias=0.95)
    res = optimize_logweight_topology(scene, ev, PROTO, PHY, steps=40, lr=0.3, link_override=1.0)
    assert res.history[-1] < res.history[0]            # the topology objective is being reduced
    assert res.final_objective < 0.5 * res.history[0]  # substantially
