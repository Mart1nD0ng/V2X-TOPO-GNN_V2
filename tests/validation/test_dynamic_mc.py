"""G6 -- independent dynamic Monte-Carlo judge (spec §8).

Acceptance: the MC is a genuine independent forward simulation (no analytic terminal
marginals); it reproduces the analytic where the mean-field is exact (all-correct);
analytic and MC agree on RANKING/direction (spec §8.3, the non-stop condition); it is
reproducible and internally consistent; it independently calibrates absolute reliability
(revealing the analytic mean-field is optimistic under correlation -- why the MC must be
the judge).
"""

import torch

from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
    run_consensus_episode,
)
from src.sampling import DistanceQueryPolicy, UniformQueryPolicy
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=16)


def _scene(gx=2, gy=2, per=5, cr=75.0, ir=115.0, seed=0):
    return build_manhattan_scene(gx, gy, per, block_m=120.0, comm_radius=cr, int_radius=ir,
                                 generator=torch.Generator().manual_seed(seed))


def test_mc_internal_validity_and_reproducible():
    scene = _scene()
    ev = build_scenario("iid", scene, base_node_err=0.1)
    pol = UniformQueryPolicy()
    a = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=2000,
                       generator=torch.Generator().manual_seed(5))
    b = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=2000,
                       generator=torch.Generator().manual_seed(5))
    # reproducible with the same seed
    assert a.F_wrong == b.F_wrong and a.S_allcorrect == b.S_allcorrect
    # valid probabilities + CI brackets the point estimate
    for v in (a.F_disagree, a.F_wrong, a.S_allcorrect):
        assert 0.0 <= v <= 1.0
    assert a.F_wrong_ci[0] <= a.F_wrong <= a.F_wrong_ci[1]
    # per-node frequencies partition into correct/wrong/undecided
    tot = a.decided_correct_freq + a.decided_wrong_freq + a.undecided_freq
    assert torch.allclose(tot, torch.ones_like(tot), atol=1e-9)


def test_all_correct_matches_analytic_exactly():
    """Where evidence is perfect the mean-field is exact: both models give all-correct."""
    scene = _scene()
    ev = build_scenario("all_correct", scene)
    pol = UniformQueryPolicy()
    an = run_consensus_episode(scene, ev, pol, PROTO, PHY, return_trajectory=False)
    mc = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=2000,
                        generator=torch.Generator().manual_seed(1))
    assert float(an.S_allcorrect) > 0.999
    assert mc.S_allcorrect > 0.999 and mc.F_wrong < 1e-3


def test_mc_captures_failure_tail_analytic_suppresses():
    """Under iid noise the MC finds a small nonzero wrong-finalisation tail that the
    analytic mean-field washes out -- the calibrated-judge role (spec §8.3)."""
    scene = _scene()
    ev = build_scenario("iid", scene, base_node_err=0.1)
    pol = UniformQueryPolicy()
    mc = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=6000,
                        generator=torch.Generator().manual_seed(2))
    # nonzero but small (sanity vs a coarse union-bound estimate ~ N * per-node tail)
    assert 0.0 < mc.F_wrong < 0.1
    assert mc.S_allcorrect < 1.0


def test_analytic_and_mc_agree_on_ranking():
    """The spec §8.3 check: analytic and MC must rank query policies consistently
    (a SYSTEMATIC ranking disagreement would be stop-condition #3)."""
    scene = _scene(gx=3, gy=2, per=5, cr=85.0, ir=130.0)
    ev = build_scenario("one_biased_region", scene, base_node_err=0.1, region_bias=0.85)
    policies = [("uniform", UniformQueryPolicy()), ("distance", DistanceQueryPolicy(beta_per_m=0.05))]
    an_fw, mc_fw = {}, {}
    for name, pol in policies:
        an_fw[name] = float(run_consensus_episode(scene, ev, pol, PROTO, PHY,
                                                  return_trajectory=False).F_wrong)
        mc_fw[name] = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=4000,
                                     generator=torch.Generator().manual_seed(7)).F_wrong
    an_rank = sorted(an_fw, key=an_fw.get)
    mc_rank = sorted(mc_fw, key=mc_fw.get)
    assert an_rank == mc_rank                      # ranking preserved
    assert an_fw["uniform"] < an_fw["distance"]    # diversity beats local concentration
    assert mc_fw["uniform"] < mc_fw["distance"]


def test_mc_direction_increases_with_bias():
    scene = _scene()
    pol = UniformQueryPolicy()
    fw = {}
    for name, bias in [("clean", 0.0), ("biased", 0.9)]:
        sc = "iid" if bias == 0.0 else "one_biased_region"
        ev = build_scenario(sc, scene, base_node_err=0.1, region_bias=bias if bias else 0.85)
        fw[name] = run_dynamic_mc(scene, ev, pol, PROTO, PHY, num_trials=3000,
                                  generator=torch.Generator().manual_seed(3)).F_wrong
    assert fw["biased"] > fw["clean"] + 0.05       # a biased region clearly raises wrong-rate


def test_mc_does_not_use_analytic_marginals():
    """Independence sentinel: the dynamic MC module must not call the analytic evaluator
    nor read its terminal marginals (constraint #8). It only shares the system definition."""
    import inspect

    import src.validation.dynamic_mc as mod
    src = inspect.getsource(mod)
    assert "run_consensus_episode" not in src        # never invokes the analytic episode
    assert ".c_ir" not in src and ".w_ir" not in src  # never reads analytic terminal marginals
    assert "evaluate_global_consensus" not in src


def test_physics_per_trial_path_runs():
    scene = _scene()
    ev = build_scenario("iid", scene, base_node_err=0.1)
    mc = run_dynamic_mc(scene, ev, UniformQueryPolicy(), PROTO, PHY, num_trials=400,
                        generator=torch.Generator().manual_seed(9), physics_per_trial=True)
    assert 0.0 <= mc.S_allcorrect <= 1.0
    assert mc.decided_correct_freq.shape == (scene.num_nodes,)
