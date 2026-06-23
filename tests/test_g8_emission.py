"""G8 (spec §3.8, Eqs. 58-59): global-risk emission + temporal feedback + stop-gradient.

Checks:
  1. Eq. 58 risk contribution ``r_ir = -log(max(c,eps)) >= 0`` and its posterior average.
  2. Per-scenario identity ``-log S_r = sum_{i in H} r_ir`` to machine precision, and that it
     reconstructs the G1 ``S_C`` exactly (ties the emission to the H1 loss).
  3. Eq. 59 emission is bounded in ``[0,1]`` and carries NO gradient (stop-gradient): a
     discriminative test against a no-detach control.
  4. The emission is monotonically aligned with the global ``F`` (higher network risk ->
     higher emission).
  5. The temporal scalar-feedback model runs, is end-to-end differentiable through the live
     path, and the emission channel genuinely influences the GNN.
  6. THE MECHANISM ABLATION: feeding the SAME bounded emission into a recurrence, the
     non-contractive (expansive) cell's hidden-state norm diverges while gated/contractive
     cells stay bounded -- FALSIFYING "a bounded scalar auto-constrains all hidden state".
"""

from __future__ import annotations

import math

import torch

from src.mainline.emission import (
    EmissionConfig,
    ScalarEmissionRecurrentModel,
    global_risk_contribution,
    hidden_state_boundedness_ablation,
    neg_log_S_r,
    risk_emission,
)
from src.mainline.global_evaluator import evaluate_global_consensus
from src.mainline.model import OperatingPointConfig, model_operating_point
from src.mainline.topology import build_candidate_graph

torch.manual_seed(0)
DT = torch.float64
EPS = 1e-6
RMAX = -math.log(EPS)


def _complete_digraph(n):
    """Fully-connected directed candidate graph (out-degree n-1 >= k)."""
    src = torch.tensor([i for i in range(n) for j in range(n) if i != j])
    dst = torch.tensor([j for i in range(n) for j in range(n) if i != j])
    return src, dst


def test_risk_contribution_formula():
    N, Q = 6, 3
    c = (torch.rand(N, Q, dtype=DT) * 0.8 + 0.1)
    rho = torch.softmax(torch.randn(Q, dtype=DT), 0)
    r_ir, node_risk = global_risk_contribution(c, rho, eps=EPS)
    assert torch.allclose(r_ir, -torch.log(c.clamp_min(EPS)), atol=1e-15)
    assert bool((r_ir >= 0).all())  # risk non-negative (c in [0,1])
    assert torch.allclose(node_risk, (rho.reshape(1, -1) * r_ir).sum(1), atol=1e-15)


def test_neg_log_S_r_identity_and_reconstructs_SC():
    # toy graph -> real consensus, then verify the §3.8 decomposition + tie to G1 S_C
    n, k = 5, 3
    src, dst = _complete_digraph(n)
    s = torch.randn(src.numel(), dtype=DT)
    ell = torch.full((src.numel(),), 0.85, dtype=DT)
    omega = torch.tensor([0.5, 0.3, 0.2], dtype=DT)
    res = evaluate_global_consensus(
        num_nodes=n, src_index=src, dst_index=dst, log_query_weight=s.unsqueeze(-1).expand(-1, 3),
        link_reliability=ell.unsqueeze(-1).expand(-1, 3), scenario_weight=omega, k=k, alpha=2,
        beta=2, rounds=6, initial_correct_preference=0.7,
    )
    nlsr = neg_log_S_r(res.c_ir, eps=EPS)  # [Q] = sum_i r_ir
    # direct per-scenario product (c away from eps -> floor inactive)
    direct = -torch.log(res.c_ir.clamp_min(EPS)).sum(0)
    assert float((nlsr - direct).abs().max()) < 1e-12
    # reconstruct S_C = sum_r omega_r exp(-(-log S_r)) and match G1 exactly
    S_C_recon = (omega * torch.exp(-nlsr)).sum()
    assert float((S_C_recon - res.S_C).abs()) < 1e-12


def test_emission_bounded_and_stop_gradient():
    N, Q = 7, 2
    rho = torch.softmax(torch.randn(Q, dtype=DT), 0)
    c = torch.rand(N, Q, dtype=DT).clamp(0.05, 0.95).requires_grad_(True)

    e = risk_emission(c, rho, eps=EPS, r_max=RMAX)
    assert bool((e >= 0).all() and (e <= 1).all())
    assert e.requires_grad is False  # detached: emission is a feature, not a grad path

    # bound holds over the FULL c range incl. the eps-floor tail (c->0) and c->1, not just a
    # narrow mid-range (so an un-clipped >1 emission would be caught here)
    c_wide = torch.linspace(0.0, 1.0, 128, dtype=DT).reshape(-1, 1)
    c_wide = torch.cat([torch.full((1, 1), EPS / 2, dtype=DT), c_wide, torch.full((1, 1), 1 - 1e-9, dtype=DT)])
    e_wide = risk_emission(c_wide, torch.ones(1, dtype=DT), eps=EPS, r_max=RMAX)
    assert bool((e_wide >= 0).all() and (e_wide <= 1.0 + 1e-12).all())

    # discriminative stop-gradient: emission contributes ZERO gradient to c ...
    loss = e.sum() + 0.0 * c.sum()  # +0*c so the output requires grad (autograd.grad legal)
    g = torch.autograd.grad(loss, c)[0]
    assert float(g.abs().sum()) == 0.0
    # ... whereas WITHOUT the stop-gradient, the same path carries a real gradient
    _, node_risk = global_risk_contribution(c, rho, eps=EPS)
    e_nodetach = (node_risk / RMAX).clamp(0.0, 1.0)
    g2 = torch.autograd.grad(e_nodetach.sum(), c)[0]
    assert float(g2.abs().sum()) > 0.0


def test_default_rmax_keeps_unit_interval_without_clip():
    # default r_max = -log(eps) is the theoretical max risk, so the pre-clip emission is
    # already in [0,1] -- the clip is a safety net, not the source of the bound.
    N, Q = 50, 4
    c = torch.rand(N, Q, dtype=DT)  # full range incl. near 0
    rho = torch.softmax(torch.randn(Q, dtype=DT), 0)
    _, node_risk = global_risk_contribution(c, rho, eps=EPS)
    pre_clip = node_risk / RMAX
    assert float(pre_clip.max()) <= 1.0 + 1e-9 and float(pre_clip.min()) >= -1e-9


def test_emission_aggregates_to_global_risk():
    # EXACT alignment with the global F (via the G1 loss -log S_C): for Q=1 the aggregate of
    # the (pre-clip) emission equals the H1 loss exactly, whatever shape it takes in
    # reliability (F is U-shaped, the §9.4/D7 finding -- so we assert the *identity*, not a
    # spurious monotonicity that the U-shape rightly violates).
    n, k = 5, 3
    src, dst = _complete_digraph(n)
    s = torch.zeros(src.numel(), dtype=DT)
    omega = torch.ones(1, dtype=DT)
    for ell_val in [0.55, 0.65, 0.75, 0.85, 0.95]:
        ell = torch.full((src.numel(),), ell_val, dtype=DT)
        res = evaluate_global_consensus(
            num_nodes=n, src_index=src, dst_index=dst, log_query_weight=s.unsqueeze(-1),
            link_reliability=ell.unsqueeze(-1), scenario_weight=omega, k=k, alpha=2, beta=2,
            rounds=8, initial_correct_preference=0.7,
        )
        _, node_risk = global_risk_contribution(res.c_ir, res.scenario_posterior, eps=EPS)
        # sum_i node_risk_i = sum_i -log c_i = -log S_r = -log S_C = loss_F  (Q=1)
        assert float((node_risk.sum() - res.loss_F).abs()) < 1e-9, ell_val
        # the bounded emission is the same quantity / r_max (clip inactive here)
        e = risk_emission(res.c_ir, res.scenario_posterior, eps=EPS, r_max=RMAX)
        assert float((RMAX * e.sum() - res.loss_F).abs()) < 1e-9, ell_val


def _graph_and_feats(n=8, seed=11):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(n, 2, generator=gen, dtype=DT) * 60.0
    g = build_candidate_graph(pos, 80.0)
    src, dst = g.src_index, g.dst_index
    outdeg = torch.bincount(src, minlength=n).to(DT)
    indeg = torch.bincount(dst, minlength=n).to(DT)
    nf = torch.stack([outdeg / outdeg.max(), indeg / indeg.clamp_min(1).max(), torch.ones(n, dtype=DT)], 1)
    ef = (g.distance / 80.0).unsqueeze(-1)
    return g, nf, ef


def test_temporal_model_runs_and_is_differentiable():
    g, nf, ef = _graph_and_feats()
    cfg = OperatingPointConfig(rounds=6)
    model = ScalarEmissionRecurrentModel(static_node_dim=3, edge_dim=1, hidden=16, layers=2).double()
    lam = torch.tensor([0.34, 0.33, 0.33], dtype=DT)
    res = model(g, nf, ef, lam, cfg, frames=4)
    assert len(res["ops"]) == 4 and len(res["emissions"]) == 4
    assert all(bool((e >= 0).all() and (e <= 1).all()) for e in res["emissions"])
    assert all(not e.requires_grad for e in res["emissions"])  # stop-gradient every frame
    last = res["ops"][-1]
    loss = last["F"] + 10.0 * last["D"] + last["E"]
    loss.backward()
    gn = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    assert math.isfinite(gn) and gn > 0


def test_emission_channel_influences_gnn():
    # robust wiring check: a hand-set nonzero emission channel changes the operating point
    g, nf, ef = _graph_and_feats()
    cfg = OperatingPointConfig(rounds=6)
    model = ScalarEmissionRecurrentModel(static_node_dim=3, edge_dim=1, hidden=16, layers=2).double()
    lam = torch.tensor([0.4, 0.3, 0.3], dtype=DT)
    with torch.no_grad():
        nf0 = torch.cat([nf, torch.zeros(g.num_nodes, 1, dtype=DT)], dim=-1)
        nf1 = torch.cat([nf, 0.5 * torch.ones(g.num_nodes, 1, dtype=DT)], dim=-1)
        o0 = model_operating_point(model.gnn, g, nf0, ef, lam, cfg)
        o1 = model_operating_point(model.gnn, g, nf1, ef, lam, cfg)
    assert abs(float(o0["F"]) - float(o1["F"])) > 1e-9 or abs(float(o0["E"]) - float(o1["E"])) > 1e-9


def test_emission_matches_eq59_per_node():
    # discriminative per-node check: Eq.59 defines e_i as NODE i's contribution; the temporal
    # model feeds it as a per-node channel, so a sum-preserving scramble (right total, wrong
    # attribution) must NOT pass.  On a heterogeneous graph c_ir varies per node.
    n, k = 6, 3
    src, dst = _complete_digraph(n)
    gen = torch.Generator().manual_seed(7)
    s = torch.randn(src.numel(), generator=gen, dtype=DT)
    ell = torch.rand(src.numel(), generator=gen, dtype=DT) * 0.4 + 0.55
    omega = torch.tensor([0.6, 0.4], dtype=DT)
    res = evaluate_global_consensus(
        num_nodes=n, src_index=src, dst_index=dst, log_query_weight=s.unsqueeze(-1).expand(-1, 2),
        link_reliability=ell.unsqueeze(-1).expand(-1, 2), scenario_weight=omega, k=k, alpha=2,
        beta=2, rounds=6, initial_correct_preference=0.7,
    )
    e = risk_emission(res.c_ir, res.scenario_posterior, eps=EPS, r_max=RMAX)
    expected = (global_risk_contribution(res.c_ir, res.scenario_posterior, eps=EPS)[1].detach() / RMAX).clamp(0, 1)
    assert torch.allclose(e, expected, atol=1e-12)
    assert float(e.std()) > 1e-6  # genuinely per-node; a constant emission would fail allclose
    # a sum-preserving scramble (all risk on node 0) does NOT reproduce the Eq.59 per-node field
    scrambled = torch.zeros_like(e)
    scrambled[0] = e.sum()
    assert not torch.allclose(scrambled.clamp(0, 1), expected, atol=1e-9)


def test_ablation_norms_are_genuine():
    # independent reconstruction of the linear-cell trajectories: the ablation's reported norms
    # must equal a from-scratch recurrence (same seed/proj), so a hard-coded growth_ratio dict
    # (input-ignoring stub) cannot pass.
    N, hidden, seed, T = 8, 16, 1, 20
    emis = [torch.rand(N, generator=torch.Generator().manual_seed(100 + t), dtype=DT) for t in range(T)]
    ab = hidden_state_boundedness_ablation(emis, hidden_dim=hidden, seed=seed)
    # proj is the FIRST randn(1, hidden) drawn from Generator(seed), exactly as the function does
    proj = torch.randn(1, hidden, generator=torch.Generator().manual_seed(seed), dtype=DT)
    for kind, rho in [("expansive", 1.3), ("contractive", 0.5)]:
        H = torch.ones(N, hidden, dtype=DT)
        norms = []
        for e in emis:
            H = rho * H + e.reshape(-1, 1) * proj
            norms.append(float(H.norm(dim=1).max()))
        assert torch.allclose(torch.tensor(norms, dtype=DT), torch.tensor(ab[kind]["norms"], dtype=DT), atol=1e-9)
        assert abs(ab[kind]["growth_ratio"] - norms[-1] / (norms[0] + 1e-12)) < 1e-9


def test_hidden_state_boundedness_ablation_falsifies_claim():
    # core G8 mechanism experiment: SAME bounded emission, different recurrence cells.
    g, nf, ef = _graph_and_feats()
    cfg = OperatingPointConfig(rounds=6)
    model = ScalarEmissionRecurrentModel(static_node_dim=3, edge_dim=1, hidden=16, layers=2).double()
    lam = torch.tensor([0.34, 0.33, 0.33], dtype=DT)
    res = model(g, nf, ef, lam, cfg, frames=6)
    emis = [e.detach() for e in res["emissions"]]
    # extend the bounded sequence to a long horizon (premise is only e in [0,1])
    emis_long = (emis * 6)[:30]
    ab = hidden_state_boundedness_ablation(emis_long, hidden_dim=16, seed=1)
    g_exp = ab["expansive"]["growth_ratio"]
    g_gru = ab["gru"]["growth_ratio"]
    g_con = ab["contractive"]["growth_ratio"]
    # bounded input does NOT bound the expansive state (claim falsified) ...
    assert g_exp > 10.0, g_exp
    # ... while the gated/contractive recurrences stay bounded (state bound comes from the
    # cell, not from the bounded emission)
    assert g_gru < 3.0 and g_con < 3.0, (g_gru, g_con)
    assert g_exp > 5.0 * max(g_gru, g_con)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G8 tests passed.")
