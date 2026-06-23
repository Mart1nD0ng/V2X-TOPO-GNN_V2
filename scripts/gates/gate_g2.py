"""G2 (spec §4 / module 3.1): weighted distinct-peer k-subset policy.

Acceptance: k-subset distribution is the exact elementary-symmetric-polynomial
law; normalisation Σp=1, small-n brute-force agreement, and a code-level "same
sampler" assertion all pass.
"""

from __future__ import annotations

import math
import sys

from _common import GateResult, main_single, run_pytest  # type: ignore
import torch

from src.mainline.symmetric_polynomials import (  # noqa: E402
    edge_inclusion_probability,
    enumerate_subset_distribution,
    verify_sampler_matches_distribution,
)


def run() -> GateResult:
    torch.manual_seed(7)
    evidence: dict = {}

    # 1. Numeric exactness demonstration on a fixed instance.
    max_norm_err = 0.0
    max_incl_err = 0.0
    max_sampler_err = 0.0
    max_sum_pi_err = 0.0
    for n, k in [(5, 2), (6, 3), (7, 4), (8, 3)]:
        log_w = torch.randn(n, dtype=torch.float64)
        dist = enumerate_subset_distribution(log_w, k)
        max_norm_err = max(max_norm_err, abs(sum(dist.values()) - 1.0))
        pi = edge_inclusion_probability(log_w, k)
        max_sum_pi_err = max(max_sum_pi_err, abs(float(pi.sum()) - k))
        for j in range(n):
            marg = sum(p for combo, p in dist.items() if j in combo)
            max_incl_err = max(max_incl_err, abs(float(pi[j]) - marg))
        diag = verify_sampler_matches_distribution(log_w, k, atol=1e-9)
        max_sampler_err = max(max_sampler_err, diag["max_abs_error"])

    evidence["max_normalisation_error (Σp-1)"] = f"{max_norm_err:.2e}"
    evidence["max_inclusion_vs_bruteforce"] = f"{max_incl_err:.2e}"
    evidence["max_|Σπ - k|"] = f"{max_sum_pi_err:.2e}"
    evidence["max_sampler_vs_distribution"] = f"{max_sampler_err:.2e}"

    numeric_ok = (
        max_norm_err < 1e-10
        and max_incl_err < 1e-10
        and max_sum_pi_err < 1e-9
        and max_sampler_err < 1e-9
    )

    # 2. Full unit-test suite (gradient FD, MC convergence, uniform degeneration).
    tests_ok, tail = run_pytest("tests/test_g2_symmetric_polynomials.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    return GateResult(
        gate="G2",
        title="weighted distinct-peer k-subset policy (elementary symmetric polynomial)",
        passed=bool(numeric_ok and tests_ok),
        evidence=evidence,
        notes="exact Eq.16 distribution; shared ancestral sampler; no hard top-k.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
