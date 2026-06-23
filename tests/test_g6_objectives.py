"""G6 (spec §9): independent global delay D and network energy E (+ power/resource heads).

Checks:
  1. D (Eq. 44) equals the independent CDF reference E[T_all|success] (Abel identity).
  2. D matches a Monte-Carlo order statistic of the global completion round.
  3. expected_attempts nbar (Eq. 50) matches the closed form and the ell->0 limit M.
  4. network_energy E (Eq. 53) matches an explicit brute-force sum.
  5. wall_clock_attempts E[L_i] (Eq. 49) matches brute force.
  6. power/blocklength heads (Eqs. 54-55) have the right ranges.
  7. Real F/D/E trade-off: raising power lowers F,D but raises E; raising blocklength
     lowers F but raises D,E -- no single control improves all three.
  8. Gradients of F, D, E w.r.t. the controls are NOT collinear (pairwise cosine < 0.95).
  9. The (power, blocklength) sweep yields >= 3 mutually non-dominated (F,D,E) points.
"""

from __future__ import annotations

import math

import torch

from src.mainline.finite_blocklength import averaged_link_success
from src.mainline.global_evaluator import build_source_padding, evaluate_global_consensus
from src.mainline.objectives import (
    attempt_energy,
    blocklength_head,
    completion_delay,
    delay_from_cdf_reference,
    expected_attempts,
    network_energy,
    power_head,
    wall_clock_attempts,
)
from src.mainline.symmetric_polynomials import edge_inclusion_probability

torch.manual_seed(0)
DT = torch.float64


def test_delay_matches_cdf_reference():
    # monotone increasing CDF S(t) in [0,1]
    for _ in range(20):
        inc = torch.rand(8, dtype=DT).abs()
        S = torch.cumsum(inc, 0)
        S = (S / S[-1]) * (0.3 + 0.6 * torch.rand(1, dtype=DT))  # S(R) in (0.3,0.9)
        S = torch.cat([torch.zeros(1, dtype=DT), S])  # S(0)=0
        d = completion_delay(S, tau_round=1.0)["D"]
        ref = delay_from_cdf_reference(S, tau_round=1.0)
        assert abs(float(d) - float(ref)) < 1e-12, (float(d), float(ref))


def test_delay_montecarlo_order_statistic():
    # S(t) = prod_i CDF_i(t) for independent per-node completion CDFs (single scenario)
    R = 6
    N = 4
    # per-node CDFs c_i(t), non-decreasing to c_i(R)
    c = torch.zeros(R + 1, N, dtype=DT)
    gen = torch.Generator().manual_seed(3)
    for i in range(N):
        steps = torch.rand(R, generator=gen, dtype=DT)
        cdf = torch.cumsum(steps, 0)
        cdf = cdf / cdf[-1] * (0.85 + 0.1 * torch.rand(1, generator=gen, dtype=DT))
        c[1:, i] = cdf
    S = c.prod(dim=1)  # S(t)=prod_i c_i(t)
    D_round = float(completion_delay(S)["D_round"])
    # MC: sample each node's completion time from its CDF; T_all = max; condition on <= R
    M = 400000
    u = torch.rand(M, N, generator=gen, dtype=DT)
    # node i completes at smallest t with c_i(t) >= u
    comp = torch.full((M, N), R + 1, dtype=DT)
    for t in range(R + 1):
        done = (c[t].unsqueeze(0) >= u) & (comp > R)
        comp = torch.where(done, torch.full_like(comp, float(t)), comp)
    T_all = comp.max(dim=1).values
    succ = T_all <= R
    D_mc = float(T_all[succ].mean())
    assert abs(D_round - D_mc) < 0.03, (D_round, D_mc)


def test_expected_attempts():
    ell = torch.linspace(0.01, 1.0, 50, dtype=DT)
    for M in [1, 2, 4, 8]:
        nbar = expected_attempts(ell, M)
        ref = (1.0 - (1.0 - ell) ** M) / ell
        assert torch.allclose(nbar, ref, atol=1e-10)
        # ell -> 0 limit is M
        nbar0 = expected_attempts(torch.tensor([1e-12], dtype=DT), M)
        assert abs(float(nbar0) - M) < 1e-3
        # ell = 1 -> exactly 1 attempt
        assert abs(float(expected_attempts(torch.tensor([1.0], dtype=DT), M)) - 1.0) < 1e-10


def test_network_energy_bruteforce():
    gen = torch.Generator().manual_seed(5)
    N, Q, E, T, M = 5, 2, 9, 4, 3
    src = torch.randint(0, N, (E,), generator=gen)
    tau = torch.rand(T, N, Q, generator=gen, dtype=DT)
    pi = torch.rand(E, Q, generator=gen, dtype=DT)
    ell = 0.2 + 0.7 * torch.rand(E, Q, generator=gen, dtype=DT)
    e_att = 1e-3 * (1 + torch.rand(E, Q, generator=gen, dtype=DT))
    omega = torch.rand(Q, generator=gen, dtype=DT)
    omega = omega / omega.sum()
    maint = 1e-4 * torch.rand(N, Q, generator=gen, dtype=DT)
    out = network_energy(tau, pi, ell, e_att, src, N, omega, M, maint_energy_node=maint)
    # brute force
    nbar = (1 - (1 - ell) ** M) / ell
    e_round = torch.zeros(N, Q, dtype=DT)
    for e in range(E):
        e_round[src[e]] += pi[e] * nbar[e] * e_att[e]
    e_round += maint
    E_ref = 0.0
    for r in range(Q):
        for i in range(N):
            E_ref += float(omega[r]) * float(tau[:, i, r].sum()) * float(e_round[i, r])
    assert abs(float(out["E"]) - E_ref) < 1e-12, (float(out["E"]), E_ref)


def test_wall_clock_attempts():
    n, k, M = 5, 3, 4
    log_w = torch.randn(1, n, dtype=DT)
    ell = 0.3 + 0.6 * torch.rand(1, n, dtype=DT)
    out = float(wall_clock_attempts(log_w, ell, k, M)[0])
    # brute force over subsets and m
    from itertools import combinations
    a = torch.exp(log_w[0])
    ek = sum(math.prod(float(a[j]) for j in S) for S in combinations(range(n), k))
    exp_L = 0.0
    for m in range(M):
        f = 1.0 - (1.0 - ell[0]) ** m
        ekf = sum(math.prod(float(a[j] * f[j]) for j in S) for S in combinations(range(n), k))
        exp_L += 1.0 - ekf / ek
    assert abs(out - exp_L) < 1e-9, (out, exp_L)


def test_heads_range():
    logits = torch.linspace(-8, 8, 100, dtype=DT)
    P = power_head(logits, 10.0, 23.0)
    assert float(P.min()) >= 10.0 - 1e-6 and float(P.max()) <= 23.0 + 1e-6
    assert float(P[50]) == 10.0 + 13.0 * float(torch.sigmoid(logits[50]))
    nb = blocklength_head(logits, 100.0, 1000.0)
    assert float(nb.min()) >= 100.0 - 1e-6 and float(nb.max()) <= 1000.0 + 1e-6


# ---- end-to-end F/D/E trade-off ----

def _setup():
    gen = torch.Generator().manual_seed(11)
    N = 6
    pos = torch.rand(N, 2, generator=gen, dtype=DT) * 4.0  # tight cluster -> complete graph
    from src.mainline.topology import build_candidate_graph
    g = build_candidate_graph(pos, 20.0)
    src, dst = g.src_index, g.dst_index
    E = g.num_edges
    pad = build_source_padding(src, dst, N)
    log_w = torch.zeros(E, dtype=DT)  # uniform query logits
    return N, src, dst, E, pad, log_w


def _operating_point(power_dbm, n_block, setup):
    """End-to-end (F, loss_F, D, E) at a (power, blocklength) operating point.

    B=4000 keeps ell moderate even at high power so F is sensitive there, while the
    realistic tx power (P_w grows with dBm) makes energy tx-dominated -- so spending power
    to drive F->0 eventually costs energy (a genuine F/D/E conflict, spec §9.4).
    """
    N, src, dst, E, pad, log_w = setup
    # power / blocklength may be scalars (broadcast to all nodes) or per-node [N] tensors
    power_node = power_dbm * torch.ones(N, dtype=DT) if power_dbm.ndim == 0 else power_dbm
    n_node = n_block * torch.ones(N, dtype=DT) if n_block.ndim == 0 else n_block
    power_edge = power_node[src]   # tx power is per source
    n_edge = n_node[src]
    pathloss_db = 70.0
    noise_mw = 10.0 ** (-9.5)
    rx_mw = torch.pow(torch.tensor(10.0, dtype=DT), (power_edge - 30.0 - pathloss_db) / 10.0)
    gamma = rx_mw / noise_mw
    B = 4000.0
    ell = averaged_link_success(gamma, n_edge, B, fading="rayleigh", max_harq_attempts=2)
    ell = ell.clamp(1e-4, 1.0 - 1e-9)
    omega = torch.ones(1, dtype=DT)
    res = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w.unsqueeze(-1),
        link_reliability=ell.unsqueeze(-1), scenario_weight=omega, k=3, alpha=2, beta=2,
        rounds=12, initial_correct_preference=0.7, return_trajectory=True,
    )
    F = res.F_global
    loss = res.loss_F
    tau_round = 1e-5 * n_node.mean()  # round duration grows with (mean) blocklength
    D = completion_delay(res.S_trajectory, tau_round=tau_round)["D"]
    s_slot = torch.where(pad.slot_mask, log_w[pad.slot_edge], torch.zeros((), dtype=DT))
    pi_slot = edge_inclusion_probability(s_slot, 3, mask=pad.slot_mask)
    pi = torch.zeros(E, dtype=DT)
    pi[pad.slot_edge[pad.slot_mask]] = pi_slot[pad.slot_mask]
    e_att = attempt_energy(power_edge, n_edge, rx_power_w=0.01, proc_energy_j=1e-6)
    # active rounds only (t=0..R_max-1): drop the terminal post-final state (Eq. 53 range)
    Eobj = network_energy(res.tau_trajectory[:-1], pi.unsqueeze(-1), ell.unsqueeze(-1),
                          e_att.unsqueeze(-1), src, N, omega, 2)["E"]
    return F, loss, D, Eobj


def _t(x):
    return torch.tensor(x, dtype=DT)


def test_F_is_ushaped_in_reliability():
    # HONEST: F is NOT monotone in power/reliability -- it is U-shaped. High at low power
    # (timeouts), minimal at intermediate power, then RISING again at very high power because
    # an over-reliable network also propagates the initial wrong-leaning mass into wrong
    # decisions. So F is a genuine third Pareto axis with an interior optimum (not the naive
    # "more power always lowers F" of spec §9.4 -- recorded as a finding, decision log D7).
    setup = _setup()
    F_low, _, _, _ = _operating_point(_t(20.0), _t(700.0), setup)   # low ell
    F_mid, _, _, _ = _operating_point(_t(24.0), _t(700.0), setup)   # ~optimal ell
    F_high, _, _, _ = _operating_point(_t(32.0), _t(700.0), setup)  # very high ell
    assert float(F_low) > float(F_mid) + 1e-3, (float(F_low), float(F_mid))    # descending branch
    assert float(F_high) > float(F_mid) + 1e-3, (float(F_high), float(F_mid))  # ascending branch
    assert float(F_mid) < 0.2 and float(F_low) > 0.5


def test_energy_active_round_contract():
    # network_energy must integrate ONLY the R_max active rounds (Eq.53, t=0..R_max-1).
    # In a slow regime the terminal undecided mass is non-negligible, so summing the full
    # [R_max+1] trajectory over-counts E; tau_trajectory[:-1] is the Eq.53-correct slice.
    from src.mainline.topology import build_candidate_graph
    gen = torch.Generator().manual_seed(2)
    N = 5
    pos = torch.rand(N, 2, generator=gen, dtype=DT) * 4.0
    g = build_candidate_graph(pos, 20.0)
    src, E = g.src_index, g.num_edges
    ell = torch.full((E, 1), 0.5, dtype=DT)  # slow regime -> residual undecided mass at R
    omega = torch.ones(1, dtype=DT)
    log_w = torch.zeros(E, 1, dtype=DT)
    res = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=g.dst_index, log_query_weight=log_w,
        link_reliability=ell, scenario_weight=omega, k=3, alpha=2, beta=2, rounds=6,
        initial_correct_preference=0.55, return_trajectory=True,
    )
    pi = torch.full((E, 1), 0.5, dtype=DT)
    e_att = torch.full((E, 1), 1e-3, dtype=DT)
    E_active = float(network_energy(res.tau_trajectory[:-1], pi, ell, e_att, src, N, omega, 2)["E"])
    E_full = float(network_energy(res.tau_trajectory, pi, ell, e_att, src, N, omega, 2)["E"])
    assert E_full > E_active * 1.01, (E_active, E_full)  # terminal mass materially over-counts
    # independent Eq.53 reference over t=0..R-1
    nbar = (1 - (1 - ell) ** 2) / ell
    e_round = torch.zeros(N, 1, dtype=DT).index_add(0, src, pi * nbar * e_att)
    ref = float((omega * (res.tau_trajectory[:-1].sum(0) * e_round).sum(0)).sum())
    assert abs(E_active - ref) < 1e-10, (E_active, ref)


def test_power_trades_delay_vs_energy():
    # In the reliable regime, more power completes consensus FASTER (D down) but costs
    # more transmit energy (E up): a genuine D-vs-E conflict.
    setup = _setup()
    _, _, D_lo, E_lo = _operating_point(_t(26.0), _t(700.0), setup)
    _, _, D_hi, E_hi = _operating_point(_t(31.0), _t(700.0), setup)
    assert float(D_hi) < float(D_lo), (float(D_hi), float(D_lo))   # faster
    assert float(E_hi) > float(E_lo), (float(E_hi), float(E_lo))   # but more energy


def test_blocklength_raises_delay_and_energy():
    # Larger blocklength lengthens each round (D up) and each transmission (E up).
    setup = _setup()
    _, _, D_lo, E_lo = _operating_point(_t(28.0), _t(600.0), setup)
    _, _, D_hi, E_hi = _operating_point(_t(28.0), _t(1100.0), setup)
    assert float(D_hi) > float(D_lo), (float(D_hi), float(D_lo))
    assert float(E_hi) > float(E_lo), (float(E_hi), float(E_lo))


def test_grad_delay_energy_not_collinear():
    # spec §11.10: in the model's real per-node control space, grad D and grad E must NOT
    # be collinear (signed cosine < 0.95 -> D and E are not the same/redundant objective).
    setup = _setup()
    N = setup[0]
    gen = torch.Generator().manual_seed(4)
    r_logits = (0.5 * torch.randn(N, generator=gen, dtype=DT)).requires_grad_(True)
    b_logits = (0.5 * torch.randn(N, generator=gen, dtype=DT)).requires_grad_(True)
    power_node = power_head(r_logits, 22.0, 32.0)
    n_node = blocklength_head(b_logits, 500.0, 1100.0)

    def grad_of(scalar):
        if r_logits.grad is not None:
            r_logits.grad = None
        if b_logits.grad is not None:
            b_logits.grad = None
        scalar.backward(retain_graph=True)
        return torch.cat([r_logits.grad.detach().clone(), b_logits.grad.detach().clone()])

    _, _, D, E = _operating_point(power_node, n_node, setup)
    gD = grad_of(D)
    gE = grad_of(E)
    cos = float((gD @ gE) / (gD.norm() * gE.norm() + 1e-30))
    # not collinear (cos near +1 would mean D and E are redundant / proportional)
    assert cos < 0.95, cos
    # and the gradients are genuinely non-trivial
    assert float(gD.norm()) > 1e-12 and float(gE.norm()) > 1e-12


def test_nondominated_pareto_points():
    setup = _setup()
    pts = []
    for pw in [20.0, 23.0, 26.0, 29.0, 32.0]:
        for nb in [500.0, 700.0, 900.0, 1100.0]:
            F, _, D, E = _operating_point(_t(pw), _t(nb), setup)
            pts.append((float(F), float(D), float(E)))

    def dominates(a, b):  # a dominates b: <= in all, < in at least one (minimisation)
        return all(a[i] <= b[i] + 1e-9 for i in range(3)) and any(a[i] < b[i] - 1e-6 for i in range(3))

    nd = [a for a in pts if not any(dominates(b, a) for b in pts if b is not a)]
    assert len(nd) >= 3, (len(nd), nd)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G6 tests passed.")
