"""G10 -- large-scale near-linear complexity (spec §11, constraints #4/#11, stop-condition #4).

Acceptance (deterministic, structural -- not wall-clock, to stay CI-robust):
* average degree stays BOUNDED as N grows (the radius graph is local) -> E = O(N), so NO degree
  cap or candidate truncation is needed (constraint #4) and there is no N x N blow-up (#11);
* edge count grows ~linearly with N (not quadratically);
* the full canonical dynamic MC runs at ~10^3 nodes and returns valid statistics (viability;
  an N x N dense path would OOM / be intractable here).

The wall-clock near-linearity (per-trial-per-edge cost ~constant 100->10000 nodes) is recorded in
``scripts/analysis/scaling_benchmark.py`` -> ``result/scaling/scaling.json``.
"""

import torch

from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene, build_scenario
from src.environment.candidate_graph import build_candidate_graph
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.validation import run_dynamic_mc


def _scene(gx):
    return build_manhattan_scene(gx, gx, 4, block_m=100.0, comm_radius=80.0, int_radius=160.0,
                                 generator=torch.Generator().manual_seed(0))


def test_degree_is_bounded_as_N_grows():
    """E/N must NOT grow with N -- a local radius graph has geometry-bounded degree (no cap)."""
    sizes = [(_scene(gx)) for gx in (4, 8, 12)]
    stats = []
    for sc in sizes:
        gc = build_candidate_graph(sc.positions, sc.comm_radius)
        stats.append((sc.num_nodes, gc.num_edges, gc.num_edges / sc.num_nodes))
    Ns = [s[0] for s in stats]
    degs = [s[2] for s in stats]
    assert Ns[-1] > 8 * Ns[0]                 # genuinely spanning ~10x in N
    assert max(degs) < 25.0                    # degree stays small (local)
    assert degs[-1] < 1.5 * degs[0]            # bounded: does NOT grow with N


def test_edge_count_grows_linearly_not_quadratically():
    small, large = _scene(4), _scene(12)
    Es = build_candidate_graph(small.positions, small.comm_radius).num_edges
    El = build_candidate_graph(large.positions, large.comm_radius).num_edges
    ratio_N = large.num_nodes / small.num_nodes
    ratio_E = El / Es
    assert 0.6 * ratio_N < ratio_E < 1.6 * ratio_N        # E ~ linear in N
    assert ratio_E < 0.2 * ratio_N ** 2                   # decisively sub-quadratic


def test_canonical_mc_runs_at_thousand_nodes():
    sc = _scene(12)                                        # N ~ 1056
    assert sc.num_nodes > 900
    ev = build_scenario("one_biased_region", sc, base_node_err=0.05, region_bias=0.9)
    pcfg = ProtocolConfig(k=3, alpha=2, beta=3, r_max=12)
    phy = RoundPhysicsConfig(subchannels=12, slots_per_window=50)
    torch.manual_seed(0)
    model = ESDGNN(ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3)).double()
    res = run_dynamic_mc(sc, ev, ESDGNNQueryPolicy(model, sc), pcfg, phy, num_trials=40,
                         generator=torch.Generator().manual_seed(1), link_override=1.0)
    assert 0.0 <= res.F_wrong <= 1.0 and 0.0 <= res.S_allcorrect <= 1.0
    assert res.F_disagree_ci[0] <= res.F_disagree <= res.F_disagree_ci[1]
