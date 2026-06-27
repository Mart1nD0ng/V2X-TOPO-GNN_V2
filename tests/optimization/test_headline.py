"""G11 -- the headline comparison harness (paired CRN + paired bootstrap + Bonferroni).

The statistics are validated on deterministic synthetic per-scene metric vectors (no training /
no MC, so these are fast and exact); the paired-CRN evaluation against the real MC is exercised
by the integration test at the bottom (small, link-ideal).
"""

import torch

from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
)
from src.optimization import (
    PolicyScores,
    compare_to_reference,
    evaluate_policies_paired,
)
from src.sampling import DistanceQueryPolicy, UniformQueryPolicy


def _scores(name, fw):
    s = PolicyScores(name)
    s.F_wrong = list(fw)
    return s


def test_paired_diff_detects_a_consistent_improvement():
    # policy 'good' is better than 'ref' on every scene by ~0.1 -> CI must exclude 0 and be < 0
    ref = _scores("ref", [0.30, 0.32, 0.28, 0.31, 0.29, 0.33, 0.27, 0.30])
    good = _scores("good", [0.20, 0.22, 0.18, 0.21, 0.19, 0.23, 0.17, 0.20])
    cmps = compare_to_reference({"ref": ref, "good": good}, "ref", metric="F_wrong", n_boot=2000)
    c = cmps[0]
    assert c.mean_diff < 0 and c.significant and c.better
    assert c.ci[1] < 0


def test_paired_diff_reports_no_significant_difference_when_equal():
    # 'tie' differs from ref by zero-mean noise -> CI must straddle 0 (NOT significant)
    g = torch.Generator().manual_seed(0)
    base = [0.30 + 0.02 * float(torch.randn((), generator=g)) for _ in range(12)]
    noise = [0.01 * float(torch.randn((), generator=g)) for _ in range(12)]
    ref = _scores("ref", base)
    tie = _scores("tie", [b + n for b, n in zip(base, noise)])
    c = compare_to_reference({"ref": ref, "tie": tie}, "ref", n_boot=3000)[0]
    assert not c.significant and c.ci[0] < 0 < c.ci[1]


def test_bonferroni_widens_intervals_with_more_comparisons():
    ref = _scores("ref", [0.30, 0.31, 0.29, 0.30, 0.32, 0.28])
    a = _scores("a", [0.27, 0.28, 0.26, 0.27, 0.29, 0.25])
    b = _scores("b", [0.28, 0.29, 0.27, 0.28, 0.30, 0.26])
    c = _scores("c", [0.26, 0.27, 0.25, 0.26, 0.28, 0.24])
    one = compare_to_reference({"ref": ref, "a": a}, "ref", alpha=0.05, n_boot=4000)[0]
    many = compare_to_reference({"ref": ref, "a": a, "b": b, "c": c}, "ref", alpha=0.05,
                                n_boot=4000)
    a_many = next(x for x in many if x.name == "a")
    w_one = one.ci[1] - one.ci[0]
    w_many = a_many.ci[1] - a_many.ci[0]
    assert w_many > w_one        # more comparisons -> wider (more conservative) CI


def test_paired_crn_uses_shared_seed_and_is_reproducible():
    scene = build_manhattan_scene(3, 3, 3, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                                  generator=torch.Generator().manual_seed(0))
    ev = build_scenario("one_biased_region", scene, base_node_err=0.05, region_bias=0.9)
    pcfg = ProtocolConfig(k=3, alpha=2, beta=3, r_max=10)
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    specs = [(scene, ev, 0), (scene, ev, 1)]

    def make_policies(sc):
        return {"uniform": UniformQueryPolicy(), "distance": DistanceQueryPolicy(beta_per_m=0.03)}

    # this test exercises the CRN/comparison mechanics, not physics -> ideal link, explicitly
    # opted in as an ablation (the headline default is now full physics; constraint #9).
    a = evaluate_policies_paired(specs, make_policies, pcfg, phy, num_trials=600,
                                 link_override=1.0, allow_ideal_ablation=True)
    b = evaluate_policies_paired(specs, make_policies, pcfg, phy, num_trials=600,
                                 link_override=1.0, allow_ideal_ablation=True)
    assert a["uniform"].F_wrong == b["uniform"].F_wrong          # reproducible (seeded CRN)
    assert len(a["uniform"].F_wrong) == 2 and len(a["distance"].F_wrong) == 2
    # both policies scored on the SAME two scenes -> aligned, comparable vectors
    cmp = compare_to_reference(a, "uniform", metric="F_wrong", n_boot=1000)
    assert cmp[0].name == "distance" and cmp[0].reference == "uniform"
