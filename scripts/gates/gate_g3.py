"""G3 (spec §5 / module 3.2): exact heterogeneous quorum DP.

Acceptance: three-way generating-function DP matches brute force, runtime scales
~O(E) in edges and ~O(k^3) in quorum size, gradients pass finite-difference, and
there is no iid-beta-tail residue in the mainline path.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch

from _common import GateResult, grep_repo, main_single, run_pytest  # type: ignore

from src.mainline.quorum_dp import (  # noqa: E402
    _quorum_dp_coefficients,
    bruteforce_quorum_distribution,
    quorum_response_distribution,
)


def _time_dp(B: int, n: int, k: int, repeats: int = 5) -> float:
    torch.manual_seed(0)
    a = torch.rand(B, n, dtype=torch.float64) + 0.05
    raw = torch.rand(B, 3, n, dtype=torch.float64)
    raw = raw / raw.sum(dim=1, keepdim=True)
    p0, pp, pm = raw[:, 0], raw[:, 1], raw[:, 2]
    # warmup
    _quorum_dp_coefficients(a, p0, pp, pm, k)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        _quorum_dp_coefficients(a, p0, pp, pm, k)
        best = min(best, time.perf_counter() - t0)
    return best


def _fit_exponent(xs: list[float], ys: list[float]) -> float:
    lx = np.log(np.asarray(xs))
    ly = np.log(np.asarray(ys))
    slope, _ = np.polyfit(lx, ly, 1)
    return float(slope)


def run() -> GateResult:
    torch.manual_seed(3)
    evidence: dict = {}

    # 1. Brute-force exactness on fixed instances.
    max_dist_err = 0.0
    max_h_err = 0.0
    for n, k, alpha in [(5, 3, 2), (6, 4, 3), (7, 5, 3), (8, 5, 3)]:
        log_w = torch.randn(n, dtype=torch.float64)
        raw = torch.rand(3, n, dtype=torch.float64)
        raw = raw / raw.sum(dim=0, keepdim=True)
        pc, pw = raw[0], raw[1]
        P = quorum_response_distribution(log_w.unsqueeze(0), pc.unsqueeze(0), pw.unsqueeze(0), k)[0]
        ref = bruteforce_quorum_distribution(torch.exp(log_w), pc, pw, k, alpha)
        for (m, nn), p in ref["distribution"].items():
            max_dist_err = max(max_dist_err, abs(float(P[m, nn]) - p))
        h_plus = float(P[alpha:, :].sum())
        max_h_err = max(max_h_err, abs(h_plus - ref["h_plus"]))
    evidence["max_distribution_vs_bruteforce"] = f"{max_dist_err:.2e}"
    evidence["max_h_plus_vs_bruteforce"] = f"{max_h_err:.2e}"
    exact_ok = max_dist_err < 1e-10 and max_h_err < 1e-10

    # 2. Complexity: ~linear in E (= B*n) at fixed k (the H4-critical scale axis),
    #    and bounded by ~cubic in k (the permitted small constant factor).
    #    Measured in the compute-dominated regime so launch overhead is amortised.
    n_fixed, k_fixed = 16, 5
    Bs = [2000, 4000, 8000, 16000, 32000]
    Es = [B * n_fixed for B in Bs]
    e_times = [_time_dp(B, n_fixed, k_fixed) for B in Bs]
    e_exp = _fit_exponent(Es, e_times)
    evidence["edge_scaling_exponent (target ~1.0)"] = f"{e_exp:.2f}"

    ks = [3, 4, 5, 6, 7, 8, 10]
    k_times = [_time_dp(4000, n_fixed, k) for k in ks]
    k_exp = _fit_exponent([k + 1 for k in ks], k_times)
    evidence["k_scaling_exponent (<= cubic)"] = f"{k_exp:.2f}"
    # Analytic structure: DP table is [B, k+1, k+1, k+1] -> O(k^3); no N x N tensor.
    evidence["dp_table_shape"] = f"[B, {k_fixed + 1}, {k_fixed + 1}, {k_fixed + 1}] (analytic O(k^3))"
    # H4: near-linear in E; k-cost provably <= cubic (upper bound, small constant).
    complexity_ok = 0.8 <= e_exp <= 1.25 and k_exp <= 3.6

    # 3. No iid-beta-tail closure in the mainline path (grep for legacy code identifiers).
    beta_pattern = r"betainc|_RegularizedBetaInc|betabinomial_upper_tail|scipy\.special\.beta"
    mainline_hits = list(grep_repo(beta_pattern, globs=("src/mainline/*.py",)))
    evidence["beta_tail_residue_in_mainline"] = f"{len(mainline_hits)} hits"
    if mainline_hits:
        evidence["beta_tail_hits"] = mainline_hits[:5]
    grep_ok = len(mainline_hits) == 0

    # 4. Full unit-test suite.
    tests_ok, tail = run_pytest("tests/test_g3_quorum_dp.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    return GateResult(
        gate="G3",
        title="exact heterogeneous quorum DP (three-way generating function)",
        passed=bool(exact_ok and complexity_ok and grep_ok and tests_ok),
        evidence=evidence,
        notes="O(E k^3) differentiable DP; correct/wrong/no-response; no beta-tail.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
