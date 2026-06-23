"""G8 (spec §3.8 / Eqs. 58-59): global-risk emission + stop-gradient + bounded-scalar claim.

Acceptance: the legacy per-node confidence emission is replaced by the node's contribution to
the GLOBAL risk ``r_ir = -log c_ir`` (Eq. 58), the next-frame feature is the scenario-averaged,
stop-gradient, clipped emission ``e_i = clip(sg[sum_r rho_r r_ir]/r_max, 0, 1)`` (Eq. 59), and
the bounded-scalar claim is settled by a MECHANISM EXPERIMENT.

Verified (exact / discriminative):
  - per-scenario identity ``-log S_r = sum_{i in H} r_ir`` to machine precision, and it
    reconstructs the G1 ``S_C`` exactly (the emission is literally a summand of the H1 loss);
  - the emission is in ``[0,1]`` by construction and the stop-gradient cuts the backward risk
    path (zero gradient vs a nonzero no-detach control);
  - for Q=1 the aggregate (pre-clip) emission equals the global-risk loss ``-log S_C`` exactly.

Mechanism experiment (the §3.8 requirement to "verify OR falsify, then correct"):
  feeding the SAME bounded emission into a recurrence, a non-contractive cell's hidden-state
  norm DIVERGES while gated/contractive cells stay bounded.  => the claim "a bounded scalar
  auto-constrains all hidden state" is FALSIFIED; boundedness is a property of the recurrence,
  not of the bounded input.  The mainline therefore (a) never asserts the auto-constraint and
  (b) uses a bounded scalar feedback channel (faithful, narrow) while attributing any
  hidden-state bound to the cell.
"""

from __future__ import annotations

import math
import sys

import torch

from _common import GateResult, main_single, run_pytest  # type: ignore

from src.mainline.emission import (  # noqa: E402
    EmissionConfig, ScalarEmissionRecurrentModel, global_risk_contribution,
    hidden_state_boundedness_ablation, neg_log_S_r, risk_emission,
)
from src.mainline.global_evaluator import evaluate_global_consensus  # noqa: E402
from src.mainline.model import OperatingPointConfig, model_operating_point  # noqa: E402
from src.mainline.topology import build_candidate_graph  # noqa: E402

DT = torch.float64
EPS = 1e-6
RMAX = -math.log(EPS)


def _complete_digraph(n):
    src = torch.tensor([i for i in range(n) for j in range(n) if i != j])
    dst = torch.tensor([j for i in range(n) for j in range(n) if i != j])
    return src, dst


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


def run() -> GateResult:
    evidence: dict = {}

    # --- exact identity + S_C reconstruction (ties emission to the G1/H1 loss) ----------
    n, k = 5, 3
    src, dst = _complete_digraph(n)
    s = torch.randn(src.numel(), dtype=DT)
    omega3 = torch.tensor([0.5, 0.3, 0.2], dtype=DT)
    res3 = evaluate_global_consensus(
        num_nodes=n, src_index=src, dst_index=dst, log_query_weight=s.unsqueeze(-1).expand(-1, 3),
        link_reliability=torch.full((src.numel(), 3), 0.85, dtype=DT), scenario_weight=omega3,
        k=k, alpha=2, beta=2, rounds=6, initial_correct_preference=0.7,
    )
    nlsr = neg_log_S_r(res3.c_ir, eps=EPS)
    direct = -torch.log(res3.c_ir.clamp_min(EPS)).sum(0)
    identity_err = float((nlsr - direct).abs().max())
    sc_recon_err = float(((omega3 * torch.exp(-nlsr)).sum() - res3.S_C).abs())
    evidence["identity -logS_r = sum_i r_ir (max err)"] = f"{identity_err:.2e}"
    evidence["reconstructs G1 S_C (err)"] = f"{sc_recon_err:.2e}"

    # --- bounded + stop-gradient (discriminative) ---------------------------------------
    rho = torch.softmax(torch.randn(2, dtype=DT), 0)
    c = torch.rand(7, 2, dtype=DT).clamp(0.05, 0.95).requires_grad_(True)
    e = risk_emission(c, rho, eps=EPS, r_max=RMAX)
    bounded = bool((e >= 0).all() and (e <= 1).all()) and (e.requires_grad is False)
    sg_grad = float(torch.autograd.grad(e.sum() + 0.0 * c.sum(), c)[0].abs().sum())
    _, node_risk_c = global_risk_contribution(c, rho, eps=EPS)
    ctrl_grad = float(torch.autograd.grad((node_risk_c / RMAX).clamp(0, 1).sum(), c)[0].abs().sum())
    evidence["emission in [0,1] & detached"] = str(bounded)
    evidence["stop-grad: grad via emission / control"] = f"{sg_grad:.1e} / {ctrl_grad:.2e}"

    # --- exact aggregate alignment with global-risk loss (Q=1), via the CANDIDATE emission --
    # NB: this exercises risk_emission itself (not just the helper), so a constant/degenerate
    # emission that ignores c_ir flips the gate's OWN inline verdict to FAIL.
    align_errs, cand_align_errs = [], []
    for ell_val in [0.55, 0.7, 0.85, 0.95]:
        r1 = evaluate_global_consensus(
            num_nodes=n, src_index=src, dst_index=dst, log_query_weight=torch.zeros(src.numel(), 1, dtype=DT),
            link_reliability=torch.full((src.numel(), 1), ell_val, dtype=DT), scenario_weight=torch.ones(1, dtype=DT),
            k=k, alpha=2, beta=2, rounds=8, initial_correct_preference=0.7,
        )
        _, nr = global_risk_contribution(r1.c_ir, r1.scenario_posterior, eps=EPS)
        align_errs.append(float((nr.sum() - r1.loss_F).abs()))
        e_cand = risk_emission(r1.c_ir, r1.scenario_posterior, eps=EPS, r_max=RMAX)
        cand_align_errs.append(float((RMAX * e_cand.sum() - r1.loss_F).abs()))
    evidence["aggregate emission = -log S_C (helper / candidate)"] = f"{max(align_errs):.2e} / {max(cand_align_errs):.2e}"

    # --- temporal model: runs, differentiable, channel wired ----------------------------
    g, nf, ef = _graph_and_feats()
    cfg = OperatingPointConfig(rounds=6)
    model = ScalarEmissionRecurrentModel(static_node_dim=3, edge_dim=1, hidden=16, layers=2).double()
    lam = torch.tensor([0.34, 0.33, 0.33], dtype=DT)
    tr = model(g, nf, ef, lam, cfg, frames=6)
    last = tr["ops"][-1]
    (last["F"] + 10.0 * last["D"] + last["E"]).backward()
    grad_norm = sum(float(p.grad.norm()) for p in model.parameters() if p.grad is not None)
    emis_bounded = all(bool((e >= 0).all() and (e <= 1).all()) and not e.requires_grad for e in tr["emissions"])
    with torch.no_grad():
        o0 = model_operating_point(model.gnn, g, torch.cat([nf, torch.zeros(g.num_nodes, 1, dtype=DT)], -1), ef, lam, cfg)
        o1 = model_operating_point(model.gnn, g, torch.cat([nf, 0.5 * torch.ones(g.num_nodes, 1, dtype=DT)], -1), ef, lam, cfg)
    channel_effect = abs(float(o0["F"]) - float(o1["F"])) + abs(float(o0["E"]) - float(o1["E"]))
    evidence["temporal grad_norm / emission bounded all frames"] = f"{grad_norm:.2e} / {emis_bounded}"
    evidence["emission channel effect on op-point"] = f"{channel_effect:.2e}"

    # --- candidate emission is the genuine PER-NODE Eq.59 (not a sum-preserving scramble) --
    # On an asymmetric consensus (random per-edge query weights + reliability) c_ir varies per
    # node; pin risk_emission element-wise to (sum_r rho_r r_ir)/r_max clipped, and require it
    # to actually vary (a constant or node-0-scramble emission has wrong per-node values ->
    # fails, even if its aggregate sum matches).
    gen_a = torch.Generator().manual_seed(7)
    s_a = torch.randn(src.numel(), generator=gen_a, dtype=DT)
    ell_a = torch.rand(src.numel(), generator=gen_a, dtype=DT) * 0.4 + 0.55
    res_asym = evaluate_global_consensus(
        num_nodes=n, src_index=src, dst_index=dst, log_query_weight=s_a.unsqueeze(-1).expand(-1, 2),
        link_reliability=ell_a.unsqueeze(-1).expand(-1, 2), scenario_weight=torch.tensor([0.6, 0.4], dtype=DT),
        k=k, alpha=2, beta=2, rounds=6, initial_correct_preference=0.7,
    )
    e_cand = risk_emission(res_asym.c_ir, res_asym.scenario_posterior, eps=EPS, r_max=RMAX)
    expected = (global_risk_contribution(res_asym.c_ir, res_asym.scenario_posterior, eps=EPS)[1].detach() / RMAX).clamp(0, 1)
    per_node_err = float((e_cand - expected).abs().max())
    per_node_std = float(e_cand.std())
    # [0,1] bound over the FULL c range incl. the eps-floor tail and c->1 (catches a no-clip
    # 2x cheat that stays <1 only on narrow mid-range inputs)
    c_wide = torch.linspace(0.0, 1.0, 64, dtype=DT).unsqueeze(-1)
    c_wide = torch.cat([torch.full((1, 1), EPS / 2, dtype=DT), c_wide, torch.full((1, 1), 1 - 1e-9, dtype=DT)])
    e_wide = risk_emission(c_wide, torch.ones(1, dtype=DT), eps=EPS, r_max=RMAX)
    bound_ok = bool((e_wide >= 0).all() and (e_wide <= 1.0 + 1e-12).all())
    evidence["candidate Eq.59 per-node err / std / bound"] = f"{per_node_err:.2e} / {per_node_std:.3f} / {bound_ok}"

    # --- MECHANISM ABLATION: bounded emission, vary recurrence cell ----------------------
    emis = [e.detach() for e in tr["emissions"]]
    emis_long = (emis * 6)[:30]  # extend bounded sequence; premise is only e in [0,1]
    ab = hidden_state_boundedness_ablation(emis_long, hidden_dim=16, seed=1)
    g_exp, g_gru, g_con = ab["expansive"]["growth_ratio"], ab["gru"]["growth_ratio"], ab["contractive"]["growth_ratio"]
    evidence["||H||-growth gru / contractive / expansive (30 frames)"] = f"{g_gru:.2f} / {g_con:.2f} / {g_exp:.2f}"

    # independent verification of the ablation (do NOT trust the self-reported scalar):
    # (a) growth_ratio must equal norms[-1]/norms[0]; (b) reconstruct the linear cells'
    # trajectories from scratch (same seed/proj) and match norms elementwise.  A hard-coded /
    # input-ignoring stub fails both (its norms don't match the genuine recurrence).
    recompute_ok = all(abs(ab[k]["growth_ratio"] - ab[k]["norms"][-1] / (ab[k]["norms"][0] + 1e-12)) < 1e-9
                       for k in ("gru", "contractive", "expansive"))
    proj_chk = torch.randn(1, 16, generator=torch.Generator().manual_seed(1), dtype=DT)  # 1st draw, as the fn does
    recon_ok = True
    for kind, rho_s in [("expansive", 1.3), ("contractive", 0.5)]:
        H = torch.ones(int(emis_long[0].numel()), 16, dtype=DT)
        rn = []
        for e in emis_long:
            H = rho_s * H + e.reshape(-1, 1) * proj_chk
            rn.append(float(H.norm(dim=1).max()))
        got = ab[kind]["norms"]
        if len(got) != len(rn) or not torch.allclose(
                torch.tensor(rn, dtype=DT), torch.tensor(got, dtype=DT), atol=1e-9):
            recon_ok = False  # fabricated / input-ignoring norms -> clean FAIL, not a crash
    evidence["ablation self-consistent / independent-reconstruction"] = f"{recompute_ok} / {recon_ok}"

    # --- verdicts -----------------------------------------------------------------------
    exact_ok = identity_err < 1e-12 and sc_recon_err < 1e-12 and max(align_errs) < 1e-9
    # emission_ok exercises the CANDIDATE risk_emission: aggregate tie + per-node Eq.59 + bound
    emission_ok = (max(cand_align_errs) < 1e-9 and per_node_err < 1e-12 and per_node_std > 1e-6 and bound_ok)
    sg_ok = bounded and sg_grad == 0.0 and ctrl_grad > 0.0
    temporal_ok = math.isfinite(grad_norm) and grad_norm > 0 and emis_bounded and channel_effect > 1e-9
    # the claim is FALSIFIED: bounded emission diverges the expansive state but not the
    # gated/contractive state -> boundedness is a recurrence property, not an input property.
    # (verified independently from norms, so a fabricated growth_ratio cannot pass.)
    claim_falsified = (g_exp > 10.0 and g_gru < 3.0 and g_con < 3.0 and g_exp > 5.0 * max(g_gru, g_con)
                       and recompute_ok and recon_ok)
    evidence["bounded-scalar claim"] = "FALSIFIED (state bound is a recurrence property, not the input's)"

    tests_ok, tail = run_pytest("tests/test_g8_emission.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    passed = bool(exact_ok and emission_ok and sg_ok and temporal_ok and claim_falsified and tests_ok)
    return GateResult(
        gate="G8",
        title="global-risk emission + stop-gradient (one-summand-of -log S_C; bounded-scalar claim falsified)",
        passed=passed,
        evidence=evidence,
        notes="Eq.58 r_ir=-log c_ir, Eq.59 e=clip(sg[sum_r rho_r r_ir]/r_max,0,1); emission is an exact "
              "summand of the G1 loss; stop-gradient cuts the backward risk path; mechanism ablation "
              "FALSIFIES the auto-constraint claim (expansive recurrence diverges under the same bounded "
              "emission) -- the mainline uses bounded scalar feedback and never asserts it bounds hidden state.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
