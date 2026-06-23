"""G1 (spec §3, §6 / module 3.3): shared finite-mixture global evaluator (H1 core).

Acceptance: F_global is computed by the log-domain product-mixture, lies in [0,1],
matches an explicit brute-force enumeration of the model's global joint event, and
passes gradient checks -- establishing F_global as a strict global event probability
(not a per-node mean).
"""

from __future__ import annotations

import sys

import torch

from _common import GateResult, grep_repo, main_single, run_pytest  # type: ignore

from src.mainline.global_evaluator import (  # noqa: E402
    build_bucketed_padding,
    evaluate_global_consensus,
    monte_carlo_global_success,
)


def _toy(N, Q, seed):
    g = torch.Generator().manual_seed(seed)
    src, dst = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                src.append(i)
                dst.append(j)
    src = torch.tensor(src)
    dst = torch.tensor(dst)
    E = src.numel()
    log_w = torch.randn(E, Q, generator=g, dtype=torch.float64)
    ell = 0.4 + 0.55 * torch.rand(E, Q, generator=g, dtype=torch.float64)
    omega = torch.rand(Q, generator=g, dtype=torch.float64)
    omega = omega / omega.sum()
    return N, src, dst, log_w, ell, omega


def _exact_global_success(c_ir, w_ir, omega):
    """Brute force P(all nodes = C) by enumerating the full model joint table.

    For each scenario r, materialise the joint over (Y_1..Y_N) in {C,W,U}^N as the
    iterated outer product of per-node [c, w, u], verify it sums to 1, then sum the
    all-C region.  Mix over omega.  This is independent of the closed-form S_C.
    """
    N, Q = c_ir.shape
    u_ir = (1.0 - c_ir - w_ir).clamp_min(0.0)
    total = 0.0
    joint_mass_ok = True
    for r in range(Q):
        joint = torch.tensor([1.0], dtype=torch.float64)
        for i in range(N):
            node = torch.stack([c_ir[i, r], w_ir[i, r], u_ir[i, r]])
            joint = torch.outer(joint, node).reshape(-1)
        joint = joint.reshape([3] * N)
        if abs(float(joint.sum()) - 1.0) > 1e-9:
            joint_mass_ok = False
        all_correct = joint[tuple(0 for _ in range(N))]  # every node in state C (index 0)
        total += float(omega[r]) * float(all_correct)
    return total, joint_mass_ok


def run() -> GateResult:
    evidence: dict = {}

    # 1. F in [0,1] across several configs.
    in_range = True
    for (N, Q, beta, k, alpha, rounds, seed) in [
        (5, 3, 1, 3, 2, 4, 0), (5, 3, 2, 3, 2, 5, 1), (6, 4, 2, 4, 3, 4, 2),
    ]:
        Nn, src, dst, log_w, ell, omega = _toy(N, Q, seed)
        res = evaluate_global_consensus(
            num_nodes=Nn, src_index=src, dst_index=dst, log_query_weight=log_w,
            link_reliability=ell, scenario_weight=omega, k=k, alpha=alpha, beta=beta, rounds=rounds,
        )
        in_range = in_range and (0.0 <= float(res.F_global) <= 1.0)
    evidence["F_global_in_[0,1]"] = in_range

    # 2. Brute-force enumeration of the model joint vs closed-form S_C.
    Nn, src, dst, log_w, ell, omega = _toy(5, 3, 7)
    res = evaluate_global_consensus(
        num_nodes=Nn, src_index=src, dst_index=dst, log_query_weight=log_w,
        link_reliability=ell, scenario_weight=omega, k=3, alpha=2, beta=2, rounds=5,
    )
    brute, mass_ok = _exact_global_success(res.c_ir, res.w_ir, omega)
    # brute uses raw c (no eps); compare with the unregularised closed form
    direct = float((omega * res.c_ir.prod(dim=0)).sum())
    err_brute = abs(brute - direct)
    evidence["bruteforce_joint_vs_closed_form"] = f"{err_brute:.2e}"
    evidence["joint_sums_to_one"] = mass_ok

    # 3. log-domain matches direct mixture (multiplicative clamp_min(eps) floor).
    eps = 1e-6
    direct_eps = float((omega * res.c_ir.clamp_min(eps).prod(dim=0)).sum())
    err_log = abs(direct_eps - float(res.S_C))
    evidence["logdomain_vs_direct"] = f"{err_log:.2e}"

    # 3b. Saturated regime (c_ir -> 1): F_global, S_C, loss_F, F_timeout stay valid
    #     (regression for the additive-eps [0,1] escape bug, decision log D4).
    Ns, ss, ds = 40, [], []
    for i in range(Ns):
        for j in range(Ns):
            if i != j:
                ss.append(i)
                ds.append(j)
    ss = torch.tensor(ss)
    ds = torch.tensor(ds)
    Es = ss.numel()
    sat = evaluate_global_consensus(
        num_nodes=Ns, src_index=ss, dst_index=ds,
        log_query_weight=torch.full((Es, 1), 4.0, dtype=torch.float64),
        link_reliability=torch.ones(Es, 1, dtype=torch.float64),
        scenario_weight=torch.ones(1, dtype=torch.float64),
        k=3, alpha=2, beta=1, rounds=12, initial_correct_preference=1.0,
    )
    sat_ok = (
        0.0 <= float(sat.F_global) <= 1.0 and float(sat.S_C) <= 1.0 + 1e-12
        and float(sat.loss_F) >= -1e-12 and float(sat.F_timeout_without_wrong) >= -1e-12
        and float(sat.c_ir.min()) > 0.999
    )
    evidence["saturated_regime_F_global"] = f"{float(sat.F_global):.3e} (c_min={float(sat.c_ir.min()):.4f})"
    evidence["saturated_in_unit_interval"] = sat_ok

    # 3c. Degree skew stays near-linear: a hub of out-degree N-1 must keep padded
    #     cells <= 2E (no N x N blow-up under the bucketed layout, H4).
    hub_src, hub_dst = [], []
    Nh = 80
    for j in range(1, Nh):
        hub_src.append(0)
        hub_dst.append(j)
    for i in range(1, Nh):
        for t in range(1, 5):
            tgt = (i + t) % Nh
            if tgt == i:
                tgt = (tgt + 1) % Nh
            hub_src.append(i)
            hub_dst.append(tgt)
    hub_src = torch.tensor(hub_src)
    hub_dst = torch.tensor(hub_dst)
    Eh = hub_src.numel()
    pad = build_bucketed_padding(hub_src, hub_dst, Nh)
    skew_ok = pad.total_cells <= 2 * Eh
    evidence["degree_skew_cells_vs_2E"] = f"{pad.total_cells} <= {2 * Eh}"

    # 4. failure decomposition identity (Eq. 10).
    err_decomp = abs((float(res.F_any_wrong) + float(res.F_timeout_without_wrong)) - float(res.F_global))
    evidence["decomposition_identity_error"] = f"{err_decomp:.2e}"

    # 5. Monte-Carlo of the model joint matches S_C (H1: real global event prob).
    gen = torch.Generator().manual_seed(2024)
    mc = monte_carlo_global_success(res, omega, num_samples=400000, generator=gen)
    evidence["montecarlo_vs_S_C"] = f"{mc['abs_error_success']:.2e}"
    import math
    p = mc["S_C_analytic"]
    mc_tol = 4.0 * math.sqrt(max(p * (1 - p), 1e-8) / mc["num_samples"]) + 2e-3

    # 6. H1 distinction: F_global is NOT the per-node mean failure.
    node_mean = float((1.0 - res.c_ir).mean())
    evidence["F_global_vs_node_mean_gap"] = f"{abs(float(res.F_global) - node_mean):.3f}"
    not_node_mean = abs(float(res.F_global) - node_mean) > 1e-2

    # 7. No legacy mean-field / node-mean import in the mainline evaluator.
    #    Target the LEGACY consensus PACKAGE import + mean-field symbols -- NOT the substring
    #    "consensus" in legitimate mainline names like evaluate_global_consensus.
    legacy_hits = grep_repo(
        r"from\s+\.\.consensus\b|from\s+src\.consensus\b|import\s+src\.consensus\b|C_avalanche_node_mean|node_mean_correct",
        globs=("src/mainline/*.py",))
    evidence["legacy_meanfield_imports"] = f"{len(legacy_hits)} hits"
    grep_ok = len(legacy_hits) == 0

    numeric_ok = (
        in_range and err_brute < 1e-10 and mass_ok and err_log < 1e-11
        and err_decomp < 1e-10 and mc["abs_error_success"] < mc_tol and not_node_mean and grep_ok
        and sat_ok and skew_ok
    )

    # 8. Full unit-test suite (snowball, gradient FD, determinism, coupling).
    tests_ok, tail = run_pytest("tests/test_g1_global_evaluator.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    return GateResult(
        gate="G1",
        title="shared finite-mixture global evaluator (log-domain product-mixture F)",
        passed=bool(numeric_ok and tests_ok),
        evidence=evidence,
        notes="F_global = strict global event prob of the shared-mixture joint (H1); not node-mean.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
