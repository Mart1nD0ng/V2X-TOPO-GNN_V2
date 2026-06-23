"""G6 (spec §9 / module 3.6): independent global delay D and network energy E.

Acceptance: D (order statistic, Eq. 44) and E (Eqs. 50-53) are independently implemented
and validated; power and blocklength heads exist; a real F/D/E trade-off is confirmed by
a Pareto experiment (>= 3 non-dominated points) with non-collinear grad D / grad E.
"""

from __future__ import annotations

import sys

import torch

from _common import GateResult, main_single, run_pytest  # type: ignore

from src.mainline.finite_blocklength import averaged_link_success  # noqa: E402
from src.mainline.global_evaluator import build_source_padding, evaluate_global_consensus  # noqa: E402
from src.mainline.objectives import (  # noqa: E402
    attempt_energy, blocklength_head, completion_delay, delay_from_cdf_reference,
    expected_attempts, network_energy, power_head,
)
from src.mainline.symmetric_polynomials import edge_inclusion_probability  # noqa: E402
from src.mainline.topology import build_candidate_graph  # noqa: E402

DT = torch.float64


def _setup():
    gen = torch.Generator().manual_seed(11)
    N = 6
    pos = torch.rand(N, 2, generator=gen, dtype=DT) * 4.0
    g = build_candidate_graph(pos, 20.0)
    src, dst, E = g.src_index, g.dst_index, g.num_edges
    pad = build_source_padding(src, dst, N)
    log_w = torch.zeros(E, dtype=DT)
    s_slot = torch.where(pad.slot_mask, log_w[pad.slot_edge], torch.zeros((), dtype=DT))
    pi = torch.zeros(E, dtype=DT)
    pi[pad.slot_edge[pad.slot_mask]] = edge_inclusion_probability(s_slot, 3, mask=pad.slot_mask)[pad.slot_mask]
    return N, src, dst, E, log_w, pi


def _op(power_node, n_node, setup):
    N, src, dst, E, log_w, pi = setup
    pe, ne = power_node[src], n_node[src]
    gamma = torch.pow(_t(10.0), (pe - 100.0) / 10.0) / (10.0 ** -9.5)
    ell = averaged_link_success(gamma, ne, 4000.0, fading="rayleigh", max_harq_attempts=2).clamp(1e-4, 1 - 1e-9)
    omega = torch.ones(1, dtype=DT)
    res = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=log_w.unsqueeze(-1),
        link_reliability=ell.unsqueeze(-1), scenario_weight=omega, k=3, alpha=2, beta=2,
        rounds=12, initial_correct_preference=0.7, return_trajectory=True,
    )
    D = completion_delay(res.S_trajectory, tau_round=1e-5 * n_node.mean())["D"]
    e_att = attempt_energy(pe, ne, rx_power_w=0.01, proc_energy_j=1e-6)
    Eo = network_energy(res.tau_trajectory[:-1], pi.unsqueeze(-1), ell.unsqueeze(-1), e_att.unsqueeze(-1), src, N, omega, 2)["E"]
    return res.F_global, D, Eo


def _t(x):
    return torch.tensor(x, dtype=DT)


def _scalar_op(power, n, setup):
    N = setup[0]
    return _op(_t(power) * torch.ones(N, dtype=DT), _t(n) * torch.ones(N, dtype=DT), setup)


def run() -> GateResult:
    evidence: dict = {}

    # 1. D (Eq. 44) vs the independent CDF reference (Abel identity).
    gen = torch.Generator().manual_seed(1)
    max_d_err = 0.0
    for _ in range(50):
        inc = torch.rand(8, generator=gen, dtype=DT).abs()
        S = torch.cumsum(inc, 0)
        S = torch.cat([torch.zeros(1, dtype=DT), S / S[-1] * 0.7])
        d = float(completion_delay(S)["D"])
        ref = float(delay_from_cdf_reference(S))
        max_d_err = max(max_d_err, abs(d - ref))
    evidence["D_vs_cdf_reference"] = f"{max_d_err:.2e}"

    # 2. E (Eq. 53) vs brute-force sum.
    N, Q, E, T, M = 5, 2, 9, 4, 3
    src = torch.randint(0, N, (E,), generator=gen)
    tau = torch.rand(T, N, Q, generator=gen, dtype=DT)
    pi = torch.rand(E, Q, generator=gen, dtype=DT)
    ell = 0.2 + 0.7 * torch.rand(E, Q, generator=gen, dtype=DT)
    e_att = 1e-3 * (1 + torch.rand(E, Q, generator=gen, dtype=DT))
    omega = torch.rand(Q, generator=gen, dtype=DT)
    omega = omega / omega.sum()
    Eobj = float(network_energy(tau, pi, ell, e_att, src, N, omega, M)["E"])
    nbar = (1 - (1 - ell) ** M) / ell
    e_round = torch.zeros(N, Q, dtype=DT).index_add(0, src, pi * nbar * e_att)
    E_ref = float((omega * (tau.sum(0) * e_round).sum(0)).sum())
    evidence["E_vs_bruteforce"] = f"{abs(Eobj - E_ref):.2e}"
    nbar_limit = abs(float(expected_attempts(_t([1e-12]), M)) - M)
    evidence["nbar_ell->0_limit_err"] = f"{nbar_limit:.2e}"
    formulas_ok = max_d_err < 1e-12 and abs(Eobj - E_ref) < 1e-12 and nbar_limit < 1e-3

    # 3. Heads present with correct ranges.
    P = power_head(torch.linspace(-6, 6, 20, dtype=DT), 22.0, 32.0)
    nb = blocklength_head(torch.linspace(-6, 6, 20, dtype=DT), 500.0, 1100.0)
    heads_ok = float(P.min()) >= 22 - 1e-6 and float(P.max()) <= 32 + 1e-6 and \
        float(nb.min()) >= 500 - 1e-6 and float(nb.max()) <= 1100 + 1e-6
    evidence["power_head_range / blocklength_head_range"] = "[22,32] dBm / [500,1100] uses"

    # 4. F/D/E trade-off: F is U-shaped in reliability (interior optimum, NOT monotone --
    #    honest correction to spec §9.4); D-vs-E conflict via power; blocklength raises D,E.
    setup = _setup()
    F_lo, _, _ = _scalar_op(20.0, 700.0, setup)   # low ell -> high F (timeouts)
    F_mid, _, _ = _scalar_op(24.0, 700.0, setup)  # ~optimal ell -> low F
    F_high, _, _ = _scalar_op(32.0, 700.0, setup)  # very high ell -> F rises again
    _, D_p_lo, E_p_lo = _scalar_op(26.0, 700.0, setup)
    _, D_p_hi, E_p_hi = _scalar_op(31.0, 700.0, setup)
    _, D_n_lo, E_n_lo = _scalar_op(28.0, 600.0, setup)
    _, D_n_hi, E_n_hi = _scalar_op(28.0, 1100.0, setup)
    evidence["F U-shape (P=20/24/32)"] = f"{float(F_lo):.3f} / {float(F_mid):.3f} / {float(F_high):.3f}"
    evidence["power: D conflict E (P=26->31)"] = f"D {float(D_p_lo):.2e}->{float(D_p_hi):.2e}, E {float(E_p_lo):.2e}->{float(E_p_hi):.2e}"
    evidence["blocklength: D,E (n=600->1100)"] = f"D {float(D_n_lo):.2e}->{float(D_n_hi):.2e}, E {float(E_n_lo):.2e}->{float(E_n_hi):.2e}"
    tradeoff_ok = (
        float(F_lo) > float(F_mid) and float(F_high) > float(F_mid)  # F U-shaped (interior optimum)
        and float(F_lo) > 0.5 and float(F_mid) < 0.2                 # F genuinely sensitive
        and float(D_p_hi) < float(D_p_lo) and float(E_p_hi) > float(E_p_lo)  # power: D down, E up
        and float(D_n_hi) > float(D_n_lo) and float(E_n_hi) > float(E_n_lo)  # blocklength: D up, E up
    )

    # 5. Non-collinear grad D / grad E in the per-node control space (spec §11.10).
    gen2 = torch.Generator().manual_seed(4)
    r_logits = (0.5 * torch.randn(setup[0], generator=gen2, dtype=DT)).requires_grad_(True)
    b_logits = (0.5 * torch.randn(setup[0], generator=gen2, dtype=DT)).requires_grad_(True)
    _, D, Eo = _op(power_head(r_logits, 22.0, 32.0), blocklength_head(b_logits, 500.0, 1100.0), setup)

    def grad_of(scalar):
        r_logits.grad = None
        b_logits.grad = None
        scalar.backward(retain_graph=True)
        return torch.cat([r_logits.grad.clone(), b_logits.grad.clone()])

    gD, gE = grad_of(D), grad_of(Eo)
    cos = float((gD @ gE) / (gD.norm() * gE.norm() + 1e-30))
    evidence["cos(gradD, gradE)"] = f"{cos:+.3f}"
    grad_ok = cos < 0.95 and float(gD.norm()) > 1e-12 and float(gE.norm()) > 1e-12

    # 6. >= 3 non-dominated (F,D,E) points over a (power, blocklength) grid.
    pts = []
    for pw in [20.0, 23.0, 26.0, 29.0, 32.0]:
        for nn in [500.0, 700.0, 900.0, 1100.0]:
            F, D, Eo = _scalar_op(pw, nn, setup)
            pts.append((float(F), float(D), float(Eo)))

    def dominates(a, b):
        return all(a[i] <= b[i] + 1e-9 for i in range(3)) and any(a[i] < b[i] - 1e-6 for i in range(3))

    nd = [a for a in pts if not any(dominates(b, a) for b in pts if b is not a)]
    evidence["nondominated_points (of 20)"] = f"{len(nd)}"
    pareto_ok = len(nd) >= 3

    tests_ok, tail = run_pytest("tests/test_g6_objectives.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    return GateResult(
        gate="G6",
        title="independent global delay D and network energy E (+ power/resource heads)",
        passed=bool(formulas_ok and heads_ok and tradeoff_ok and grad_ok and pareto_ok and tests_ok),
        evidence=evidence,
        notes="D = order statistic (Eq.44); E = total joules (Eq.53); real F/D/E Pareto, non-collinear gradD/gradE.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
