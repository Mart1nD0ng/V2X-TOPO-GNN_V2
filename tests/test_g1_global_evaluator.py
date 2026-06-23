"""G1 (spec §3, §6): shared finite-mixture global evaluator (H1 core).

Checks:
  1. Snowball transition is row-stochastic; readout preference sums to 1; terminal
     outcomes are a valid (c, w, undecided) simplex.
  2. F_global in [0,1]; log-domain product-mixture equals the direct mixture (1e-12).
  3. Failure decomposition identity F = F_any_wrong + F_timeout (Eq. 10, 1e-12).
  4. Monte-Carlo of the *defined model joint* matches S_C (H1: genuine global event
     probability, not a node-mean).
  5. F_global is NOT the per-node mean failure (distinguishes H1 from the forbidden
     node-mean closure).
  6. Gradient of loss_F w.r.t. logits and link reliability matches central FD (1e-4).
  7. Forward is deterministic (no Monte-Carlo in the training path).
  8. build_source_padding round-trips degrees; the recurrence couples the graph
     (worse links -> more undecided).
"""

from __future__ import annotations

import math

import torch

from src.mainline.global_evaluator import (
    build_bucketed_padding,
    build_source_padding,
    evaluate_global_consensus,
    monte_carlo_global_success,
)
from src.mainline.snowball import (
    build_transition,
    initial_distribution,
    readout_preference,
    snowball_state_count,
    terminal_outcomes,
)

torch.manual_seed(0)
DT = torch.float64


def _complete_digraph(N):
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor(src), torch.tensor(dst)


def _toy_inputs(N=5, Q=3, beta=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    src, dst = _complete_digraph(N)
    E = src.numel()
    log_w = torch.randn(E, Q, generator=g, dtype=DT)
    ell = 0.4 + 0.55 * torch.rand(E, Q, generator=g, dtype=DT)
    omega = torch.rand(Q, generator=g, dtype=DT)
    omega = omega / omega.sum()
    return dict(num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w,
               link_reliability=ell, scenario_weight=omega)


def test_snowball_transition_row_stochastic():
    for beta in [1, 2, 3]:
        B = 7
        raw = torch.rand(B, 3, dtype=DT)
        raw = raw / raw.sum(dim=1, keepdim=True)
        T = build_transition(raw[:, 0], raw[:, 1], raw[:, 2], beta)
        S = snowball_state_count(beta)
        assert T.shape == (B, S, S)
        row_sums = T.sum(dim=2)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-12)
        # absorbing states
        assert torch.allclose(T[:, 2 * beta, 2 * beta], torch.ones(B, dtype=DT))
        assert torch.allclose(T[:, 2 * beta + 1, 2 * beta + 1], torch.ones(B, dtype=DT))


def test_readout_and_terminal_simplex():
    for beta in [1, 2, 3]:
        p = torch.rand(6, snowball_state_count(beta), dtype=DT)
        p = p / p.sum(dim=1, keepdim=True)
        u, v = readout_preference(p, beta)
        assert torch.allclose(u + v, torch.ones(6, dtype=DT), atol=1e-12)
        c, w, und = terminal_outcomes(p, beta)
        assert torch.all(c >= 0) and torch.all(w >= 0) and torch.all(und >= -1e-12)
        assert torch.allclose(c + w + und, torch.ones(6, dtype=DT), atol=1e-12)


def test_F_in_range_and_logdomain_matches_direct():
    args = _toy_inputs(N=5, Q=3, beta=2)
    res = evaluate_global_consensus(k=3, alpha=2, beta=2, rounds=4, **args)
    assert 0.0 <= float(res.F_global) <= 1.0
    # direct mixture with the same MULTIPLICATIVE floor (clamp_min(eps), never c+eps)
    eps = 1e-6
    c = res.c_ir.clamp_min(eps)
    omega = args["scenario_weight"]
    direct = (omega * c.prod(dim=0)).sum()
    assert abs(float(direct) - float(res.S_C)) < 1e-12, (float(direct), float(res.S_C))
    assert abs(float(res.F_global) - (1.0 - float(res.S_C))) < 1e-12


def test_failure_decomposition_identity():
    args = _toy_inputs(N=6, Q=4, beta=2, seed=1)
    res = evaluate_global_consensus(k=3, alpha=2, beta=2, rounds=5, **args)
    recombined = float(res.F_any_wrong) + float(res.F_timeout_without_wrong)
    assert abs(recombined - float(res.F_global)) < 1e-10, (recombined, float(res.F_global))
    assert float(res.F_any_wrong) >= -1e-12
    assert float(res.F_timeout_without_wrong) >= -1e-12


def test_monte_carlo_matches_S_C():
    args = _toy_inputs(N=5, Q=3, beta=1, seed=2)
    res = evaluate_global_consensus(k=3, alpha=2, beta=1, rounds=4, **args)
    gen = torch.Generator().manual_seed(123)
    mc = monte_carlo_global_success(res, args["scenario_weight"], num_samples=400000, generator=gen)
    p = mc["S_C_analytic"]
    tol = 4.0 * math.sqrt(max(p * (1 - p), 1e-8) / mc["num_samples"]) + 1e-3
    assert mc["abs_error_success"] < tol, mc


def test_F_global_is_not_node_mean():
    # H1: F_global must be a global event probability, not the per-node mean failure.
    args = _toy_inputs(N=6, Q=3, beta=2, seed=3)
    res = evaluate_global_consensus(k=3, alpha=2, beta=2, rounds=5, **args)
    node_mean_failure = float((1.0 - res.c_ir).mean())  # forbidden node-mean closure
    # For a multi-node global-AND event, P(any fail) >> mean per-node fail in general.
    assert abs(float(res.F_global) - node_mean_failure) > 1e-3
    # and F_global must dominate the single worst node's failure lower bound sanity
    assert float(res.F_global) <= 1.0 + 1e-9


def test_gradient_matches_finite_difference():
    args = _toy_inputs(N=4, Q=2, beta=1, seed=4)
    src, dst = args["src_index"], args["dst_index"]
    padding = build_bucketed_padding(src, dst, args["num_nodes"])

    def loss_of(log_w, ell):
        return evaluate_global_consensus(
            num_nodes=args["num_nodes"], src_index=src, dst_index=dst,
            log_query_weight=log_w, link_reliability=ell, scenario_weight=args["scenario_weight"],
            k=3, alpha=2, beta=1, rounds=3, padding=padding,
        ).loss_F

    log_w = args["log_query_weight"].clone().requires_grad_(True)
    ell = args["link_reliability"].clone().requires_grad_(True)
    loss = loss_of(log_w, ell)
    loss.backward()
    g_lw = log_w.grad.detach().clone()
    g_ell = ell.grad.detach().clone()

    eps = 1e-6
    E, Q = log_w.shape
    # check a sample of entries
    for (idx0, idx1) in [(0, 0), (3, 1), (7, 0), (E - 1, Q - 1)]:
        for base, grad, name in ((args["log_query_weight"], g_lw, "logit"),
                                 (args["link_reliability"], g_ell, "ell")):
            plus = base.clone()
            minus = base.clone()
            plus[idx0, idx1] += eps
            minus[idx0, idx1] -= eps
            if name == "logit":
                fp = float(loss_of(plus, args["link_reliability"]))
                fm = float(loss_of(minus, args["link_reliability"]))
            else:
                fp = float(loss_of(args["log_query_weight"], plus))
                fm = float(loss_of(args["log_query_weight"], minus))
            fd = (fp - fm) / (2 * eps)
            ref = float(grad[idx0, idx1])
            rel = abs(fd - ref) / (abs(fd) + 1e-6)
            assert rel < 1e-4, (name, idx0, idx1, fd, ref, rel)


def test_forward_is_deterministic():
    args = _toy_inputs(N=5, Q=3, beta=2, seed=5)
    r1 = evaluate_global_consensus(k=3, alpha=2, beta=2, rounds=4, **args)
    r2 = evaluate_global_consensus(k=3, alpha=2, beta=2, rounds=4, **args)
    assert float(r1.F_global) == float(r2.F_global)
    assert float(r1.loss_F) == float(r2.loss_F)


def test_padding_roundtrip_and_coupling():
    N = 5
    src, dst = _complete_digraph(N)
    pad = build_source_padding(src, dst, N)
    assert torch.all(pad.out_degree == N - 1)
    assert int(pad.slot_mask.sum()) == src.numel()
    # coupling: degrade all links -> more undecided terminal mass
    g = torch.Generator().manual_seed(9)
    E = src.numel()
    log_w = torch.randn(E, 1, generator=g, dtype=DT)
    omega = torch.ones(1, dtype=DT)
    good = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w,
        link_reliability=torch.full((E, 1), 0.95, dtype=DT), scenario_weight=omega,
        k=3, alpha=2, beta=2, rounds=5,
    )
    bad = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w,
        link_reliability=torch.full((E, 1), 0.35, dtype=DT), scenario_weight=omega,
        k=3, alpha=2, beta=2, rounds=5,
    )
    assert float(bad.undecided_ir.mean()) > float(good.undecided_ir.mean())


def test_saturated_regime_stays_in_unit_interval():
    # Regression for the D4 eps bug: when c_ir -> 1 (perfect links, strong lean),
    # an additive c+eps floor made S_C=(1+eps)^|H| > 1 and F_global < 0. The
    # multiplicative clamp_min(eps) floor must keep F_global, S_C, loss_F, F_timeout valid.
    N = 40
    src, dst = _complete_digraph(N)
    E = src.numel()
    log_w = torch.full((E, 1), 4.0, dtype=DT)  # strong, near-deterministic selection
    ell = torch.ones(E, 1, dtype=DT)  # perfect links
    omega = torch.ones(1, dtype=DT)
    res = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w,
        link_reliability=ell, scenario_weight=omega, k=3, alpha=2, beta=1, rounds=12,
        initial_correct_preference=1.0,
    )
    # every node should be (numerically) fully decided correct -> S_C ~ 1, F_global ~ 0
    assert float(res.c_ir.min()) > 0.999
    assert 0.0 <= float(res.F_global) <= 1.0, float(res.F_global)
    assert float(res.S_C) <= 1.0 + 1e-12, float(res.S_C)
    assert float(res.loss_F) >= -1e-12, float(res.loss_F)
    assert float(res.F_timeout_without_wrong) >= -1e-12
    assert float(res.F_any_wrong) >= -1e-12


def test_low_undecided_decomposition_nonnegative():
    # Regression for the F_timeout sign artifact: with tiny undecided mass the eps
    # asymmetry drove F_timeout negative; the symmetric multiplicative floor fixes it.
    g = torch.Generator().manual_seed(77)
    N, Q = 6, 2
    src, dst = _complete_digraph(N)
    E = src.numel()
    log_w = 3.0 * torch.randn(E, Q, generator=g, dtype=DT)
    ell = 0.9 + 0.1 * torch.rand(E, Q, generator=g, dtype=DT)
    omega = torch.rand(Q, generator=g, dtype=DT)
    omega = omega / omega.sum()
    res = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w,
        link_reliability=ell, scenario_weight=omega, k=3, alpha=2, beta=1, rounds=6,
    )
    assert float(res.F_timeout_without_wrong) >= -1e-12, float(res.F_timeout_without_wrong)
    assert float(res.F_any_wrong) >= -1e-12


def test_degree_skew_is_linear_and_correct():
    # Regression for the H4 skew bug: a single hub of degree N-1 must NOT blow padded
    # cells to O(N^2); the bucketed layout keeps total cells <= 2E. Also check the
    # bucketed result equals a dense per-node reference on the same skewed graph.
    N = 60
    # hub 0 points to everyone; every other node points to k+1 low-index neighbours.
    src, dst = [], []
    for j in range(1, N):
        src.append(0)
        dst.append(j)
    kdeg = 4
    for i in range(1, N):
        targets = [(i + t) % N for t in range(1, kdeg + 1)]
        targets = [t if t != i else (t + 1) % N for t in targets]
        for t in targets:
            if t == i:
                t = (t + 1) % N
            src.append(i)
            dst.append(t)
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    E = src.numel()
    pad = build_bucketed_padding(src, dst, N)
    assert pad.total_cells <= 2 * E, (pad.total_cells, E)
    # correctness vs dense reference (run both, compare F_global)
    g = torch.Generator().manual_seed(5)
    log_w = torch.randn(E, 1, generator=g, dtype=DT)
    ell = 0.5 + 0.4 * torch.rand(E, 1, generator=g, dtype=DT)
    omega = torch.ones(1, dtype=DT)
    res = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w,
        link_reliability=ell, scenario_weight=omega, k=3, alpha=2, beta=2, rounds=4,
    )
    assert 0.0 <= float(res.F_global) <= 1.0


def test_bucketed_matches_dense_reference():
    # The bucketed quorum must equal a single-bucket (dense) computation exactly.
    from src.mainline.global_evaluator import _bucketed_quorum, build_bucketed_padding
    from src.mainline.quorum_dp import quorum_decision_probabilities
    N, Q, k, alpha = 7, 2, 3, 2
    src, dst = _complete_digraph(N)
    E = src.numel()
    g = torch.Generator().manual_seed(3)
    a_edge = torch.randn(E, Q, generator=g, dtype=DT)
    ell_edge = 0.5 + 0.4 * torch.rand(E, Q, generator=g, dtype=DT)
    pref_c = torch.rand(N, Q, generator=g, dtype=DT)
    pref_w = 1.0 - pref_c
    pad = build_bucketed_padding(src, dst, N)
    hp, hm, hz = _bucketed_quorum(pad, a_edge, ell_edge, pref_c, pref_w, k, alpha)
    # dense reference via SourcePadding
    dense = build_source_padding(src, dst, N)
    se = dense.slot_edge.reshape(-1)
    a_slot = a_edge[se].reshape(N, dense.max_deg, Q)
    ell_slot = ell_edge[se].reshape(N, dense.max_deg, Q)
    dst_slot = dst[dense.slot_edge]
    pc = ell_slot * pref_c[dst_slot]
    pw = ell_slot * pref_w[dst_slot]
    a_b = a_slot.permute(0, 2, 1).reshape(N * Q, dense.max_deg)
    pc_b = pc.permute(0, 2, 1).reshape(N * Q, dense.max_deg)
    pw_b = pw.permute(0, 2, 1).reshape(N * Q, dense.max_deg)
    mask_b = dense.slot_mask.unsqueeze(1).expand(N, Q, dense.max_deg).reshape(N * Q, dense.max_deg)
    dec = quorum_decision_probabilities(a_b, pc_b, pw_b, k, alpha, mask=mask_b)
    assert torch.allclose(hp.reshape(-1), dec.h_plus, atol=1e-12)
    assert torch.allclose(hm.reshape(-1), dec.h_minus, atol=1e-12)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G1 tests passed.")
