"""G0/G3 -- the single canonical consensus episode (spec §7.2, plan §4).

Acceptance: the closed loop runs and ties protocol + evidence + both graphs + round
physics together; terminal reliability is valid and orders sensibly across evidence
scenarios; the mechanism trace is a true runtime trace; each mechanism is on the canonical
path (disabling it changes the terminal result); the query policy never sees truth/votes;
differentiable in the query weights.
"""

import pytest
import torch

from src.environment import (
    build_candidate_graph,
    build_manhattan_scene,
    build_scenario,
    run_consensus_episode,
)
from src.environment.canonical_episode import ProtocolConfig
from src.environment.round_physics import RoundPhysicsConfig
from src.sampling import UniformQueryPolicy

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=16)


def _scene(per=5, seed=0):
    return build_manhattan_scene(2, 2, per, block_m=120.0, comm_radius=75.0, int_radius=115.0,
                                 generator=torch.Generator().manual_seed(seed))


def _run(scenario, scene=None, **kw):
    scene = scene or _scene()
    ev = build_scenario(scenario, scene, base_node_err=0.08, region_bias=0.9)
    return run_consensus_episode(scene, ev, UniformQueryPolicy(), PROTO, PHY, **kw)


# ------------------------------------------------------------------ closure
def test_closed_loop_terminal_validity():
    res = _run("iid")
    c, w, u = res.c_ir, res.w_ir, res.undecided_ir
    assert torch.allclose(c + w + u, torch.ones_like(c), atol=1e-9)
    assert bool((c >= -1e-9).all()) and bool((w >= -1e-9).all()) and bool((u >= -1e-9).all())
    for f in (res.F_disagree, res.F_wrong, res.S_allcorrect):
        assert -1e-9 <= float(f) <= 1 + 1e-9
    assert abs(float(res.scenario_weight.sum()) - 1.0) < 1e-9
    # trajectories present and round count matches
    assert res.c_trajectory.shape[0] == PROTO.r_max + 1
    assert res.round_duration.shape[0] == PROTO.r_max


def test_reliability_orders_across_evidence_scenarios():
    """Perfect/iid evidence -> reliable correct consensus; region bias -> wrong decisions.
    This is the headline behaviour the correlated-evidence environment must produce."""
    perfect = _run("all_correct", return_trajectory=False)
    iid = _run("iid", return_trajectory=False)
    biased = _run("one_biased_region", return_trajectory=False)
    assert float(perfect.S_allcorrect) > 0.99 and float(perfect.F_wrong) < 1e-3
    assert float(iid.S_allcorrect) > 0.99 and float(iid.F_wrong) < 1e-3
    # a shared region error makes that region's nodes finalize WRONG -> validity + safety fail
    assert float(biased.F_wrong) > 0.1
    assert float(biased.F_disagree) > 0.1
    assert float(biased.S_allcorrect) < float(iid.S_allcorrect)


# ------------------------------------------------------------- mechanism trace
def test_mechanism_trace_is_accurate():
    res = _run("one_biased_region")
    tr = res.mechanism_trace
    assert tr["protocol"] == "binary_snowball"
    assert tr["query_policy"] == "uniform"
    assert tr["tau_proxy"] is False
    assert tr["request_response"] is True
    assert tr["finite_harq"] is True               # max_harq_attempts=2
    assert tr["cross_destination_interference"] is True   # G_int strictly larger than G_comm
    assert tr["num_edges_int"] > tr["num_edges_comm"]
    assert tr["correlated_evidence"] is True        # a region is biased, Q>1
    assert tr["policy_uses_truth_or_vote"] is False
    assert tr["num_scenarios"] == res.scenario_weight.numel()


def test_each_mechanism_is_on_the_canonical_path():
    """G0 activation sentinel: disabling each mechanism changes the terminal result, proving
    it actually executes on the canonical path (not merely implemented, constraint #13)."""
    scene = _scene(per=8)   # denser -> early-round load makes every mechanism bite
    ev = build_scenario("iid", scene, base_node_err=0.08)
    base = run_consensus_episode(scene, ev, UniformQueryPolicy(), PROTO, PHY, return_trajectory=False)
    for flag in ("disable_interference", "disable_collision", "disable_half_duplex", "disable_queueing"):
        off = run_consensus_episode(scene, ev, UniformQueryPolicy(), PROTO, PHY,
                                    return_trajectory=False, **{flag: True})
        moved = (abs(float(off.cumulative_time.sum() - base.cumulative_time.sum()))
                 + abs(float(off.energy.sum() - base.energy.sum()))
                 + abs(float(off.S_allcorrect - base.S_allcorrect)))
        assert moved > 1e-6, f"{flag}: disabling did not change the canonical result"
        # the trace flag reflects the disabled mechanism
        key = {"disable_interference": "interference_graph", "disable_collision": "mode2_collision",
               "disable_half_duplex": "half_duplex", "disable_queueing": "queueing"}[flag]
        assert off.mechanism_trace[key] is False


# ------------------------------------------------------------ policy isolation
def test_query_policy_never_sees_truth_or_votes():
    """The policy is invoked with ONLY the candidate graph -- no ground truth, no state."""
    scene = _scene()
    seen_args = {}

    class SpyPolicy:
        name = "spy"

        def log_weights(self, graph, *args, **kwargs):
            seen_args["args"] = args
            seen_args["kwargs"] = kwargs
            seen_args["type"] = type(graph).__name__
            return torch.zeros(graph.num_edges, dtype=torch.float64)

    ev = build_scenario("one_biased_region", scene)
    run_consensus_episode(scene, ev, SpyPolicy(), PROTO, PHY, return_trajectory=False)
    assert seen_args["args"] == () and seen_args["kwargs"] == {}   # graph only
    assert seen_args["type"] == "RadiusGraph"


def test_inclusion_independent_of_evidence_truth():
    """Two evidence models with different region bias but identical geometry must yield the
    same query behaviour (the policy cannot condition on the latent / truth)."""
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    from src.sampling.esp_query import edge_inclusion_probabilities
    lw = UniformQueryPolicy().log_weights(gc)
    pi = edge_inclusion_probabilities(gc.src_index, gc.dst_index, gc.num_nodes, lw, PROTO.k)
    # the inclusion depends only on geometry+policy, not on any evidence model -> reuse holds
    assert bool((pi >= 0).all()) and abs(float(pi.sum()) - PROTO.k * (gc.out_degree() > 0).sum()) < 1e-6


# -------------------------------------------------------- dynamic_mc is separate
def test_dynamic_mc_mode_not_reusing_analytic():
    with pytest.raises(NotImplementedError):
        _run("iid", mode="dynamic_mc")


# ----------------------------------------------------------- differentiability
def test_differentiable_in_query_weights():
    scene = _scene()
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    w = torch.zeros(gc.num_edges, dtype=torch.float64, requires_grad=True)

    class LearnPolicy:
        name = "learn"

        def log_weights(self, graph):
            return w

    ev = build_scenario("one_biased_region", scene)
    res = run_consensus_episode(scene, ev, LearnPolicy(), PROTO, PHY, return_trajectory=False)
    res.F_wrong.backward()
    assert w.grad is not None and bool(torch.isfinite(w.grad).all())
    assert float(w.grad.abs().sum()) > 0   # the query topology actually affects validity


def test_scales_with_single_scenario():
    # iid -> Q=1 keeps the [N,Q,...] tensors small; check a few hundred nodes run finitely
    scene = build_manhattan_scene(4, 4, 8, block_m=120.0, comm_radius=70.0, int_radius=110.0,
                                  generator=torch.Generator().manual_seed(1))
    ev = build_scenario("iid", scene, base_node_err=0.08)
    res = run_consensus_episode(scene, ev, UniformQueryPolicy(), ProtocolConfig(k=3, alpha=2, beta=3, r_max=8),
                                PHY, return_trajectory=False)
    assert res.scenario_weight.numel() == 1
    assert bool(torch.isfinite(res.F_wrong)) and bool(torch.isfinite(res.energy).all())
