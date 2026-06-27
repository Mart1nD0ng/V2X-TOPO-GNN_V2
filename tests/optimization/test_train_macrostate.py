"""G-CONSTRAINED-OBJECTIVE: end-to-end primal-dual training on the macrostate objective (spec §6).

Demonstrates the reliability-constrained objective working end to end: training the ESD-GNN on
the analytic macrostate surrogate (a) reduces the dominant reliability failure / raises
P_correct, (b) ascends the dual of the violated constraint, (c) does NOT trade away the other
reliability terms (hard constraint, #4), (d) generalises to a held-out scene, and (e) is
confirmed by the INDEPENDENT dynamic MC (the headline judge, S4 basin outcomes).

Scenario: two opposing regions + a TIGHT deadline + an imperfect (fixed) link, so the dominant
macrostate failure is the deadline basin (slow / failed consensus) and the topology lever
(polling agreeing peers -> faster quorum) genuinely reduces it. link_override isolates the
topology lever for a fast demo; full-physics headline training is Phase 7 / G-ESP-BASELINE.
"""

import copy

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import (
    ProtocolConfig,
    RoundPhysicsConfig,
    build_manhattan_scene,
    build_scenario,
    run_consensus_episode,
)
from src.metrics.participation import uniform_participation
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.optimization.macrostate_objective import macrostate_metrics, train_macrostate
from src.validation import run_dynamic_mc

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
R_D = 8
LO = 0.6
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=R_D)
PROFILE = ConsensusServiceProfile.urban_default().replace(
    k=3, alpha=2, beta=3, max_poll_epochs=R_D,
    max_wrong_basin_probability=2e-2, max_split_basin_probability=2e-2,
    max_deadline_miss_probability=5e-2)


def _inst(s):
    sc = build_manhattan_scene(3, 3, 3, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                               generator=torch.Generator().manual_seed(s))
    return sc, build_scenario("two_opposing_regions", sc, base_node_err=0.05, region_bias=0.9)


def _part(sc):
    return uniform_participation(sc.num_nodes, dtype=torch.float64)


def _surrogate_pcorrect(model, sc, ev):
    res = run_consensus_episode(sc, ev, ESDGNNQueryPolicy(model, sc), PROTO, PHY,
                                return_trajectory=True, link_override=LO)
    m = macrostate_metrics(res.c_trajectory, res.w_trajectory, _part(sc), res.scenario_weight,
                           res.energy, PROFILE)
    return m


def test_train_macrostate_constrained_objective_end_to_end():
    held = _inst(7)
    torch.manual_seed(0)
    model = ESDGNN(ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3)).double()
    untrained = copy.deepcopy(model)

    pre_held = _surrogate_pcorrect(untrained, *held)
    assert float(pre_held["F_deadline"].detach()) > 0.5   # the scenario is genuinely hard (failure to reduce)

    out = train_macrostate(model, [_inst(0), _inst(1), _inst(2)], PROTO, PHY, PROFILE,
                           participation_fn=_part, steps=35, lr=8e-3, eta_mu=10.0, link_override=LO)
    h = out["history"]

    # (a) training raises the surrogate P_correct (reduces the total reliability failure)
    assert h["P_correct"][-1] > h["P_correct"][0] + 0.01
    # (b) the violated-constraint dual ascends (F_deadline >> eps_d -> mu_d grows)
    assert h["mu_d"][-1] > h["mu_d"][0]
    # (c) reliability is a HARD constraint: F_wrong is not traded UP for the latency/deadline gain
    assert h["F_wrong"][-1] <= h["F_wrong"][0] + 1e-6
    # (d) generalises: held-out surrogate P_correct higher after training than the untrained model
    post_held = _surrogate_pcorrect(model, *held)
    assert float(post_held["P_correct"].detach()) > float(pre_held["P_correct"].detach()) + 0.005

    # (e) INDEPENDENT dynamic-MC confirmation on the held-out scene: trained basin P_correct up
    sc, ev = held
    mc_un = run_dynamic_mc(sc, ev, ESDGNNQueryPolicy(untrained, sc), PROTO, PHY, num_trials=1200,
                           generator=torch.Generator().manual_seed(3), link_override=LO,
                           service_profile=PROFILE, participation=_part(sc))
    mc_tr = run_dynamic_mc(sc, ev, ESDGNNQueryPolicy(model, sc), PROTO, PHY, num_trials=1200,
                           generator=torch.Generator().manual_seed(3), link_override=LO,
                           service_profile=PROFILE, participation=_part(sc))
    # the MC (judge) agrees with the surrogate direction: more correct-basin first-hitting
    assert mc_tr.basin_P_correct > mc_un.basin_P_correct
    # and the MC's four basin outcomes still partition probability
    for mc in (mc_un, mc_tr):
        total = mc.basin_P_correct + mc.basin_F_wrong + mc.basin_F_split + mc.basin_F_deadline
        assert abs(total - 1.0) < 1e-9


def test_train_macrostate_history_shape():
    # lightweight: a 2-step run returns aligned history vectors and mutates the model in place
    torch.manual_seed(1)
    model = ESDGNN(ESDGNNConfig(hidden_dim=16, r=4, n_enc=2, n_refine=1, k=3)).double()
    res = train_macrostate(model, [_inst(0)], PROTO, PHY, PROFILE, participation_fn=_part,
                           steps=2, link_override=LO)
    h = res["history"]
    assert len(h["loss"]) == 2 and len(h["mu_d"]) == 2 and len(h["P_correct"]) == 2
    assert res["model"] is model and res["duals"].mu_d >= 0.0
