"""G2 (spec §4): exact weighted distinct-peer k-subset policy.

Checks:
  1. Normalisation: sum over all k-subsets of Eq. 16 equals 1 (< 1e-10).
  2. e_k matches brute-force subset enumeration (< 1e-10 relative).
  3. Inclusion probability pi_{ij} matches brute-force, and sum_j pi = k (Eq. 19).
  4. The ancestral sampler reproduces Eq. 16 exactly (code-level "same sampler").
  5. Monte-Carlo sample frequencies converge to the analytic subset distribution.
  6. Gradient of log e_k matches central finite differences (< 1e-4 relative).
"""

from __future__ import annotations

import math
from itertools import combinations

import torch

from src.mainline.symmetric_polynomials import (
    edge_inclusion_probability,
    elementary_symmetric,
    enumerate_subset_distribution,
    log_elementary_symmetric,
    sample_k_subset,
    subset_log_probability,
    verify_sampler_matches_distribution,
)

torch.manual_seed(0)
DT = torch.float64


def _brute_force_e(weights, k):
    n = weights.numel()
    total = 0.0
    for combo in combinations(range(n), k):
        prod = 1.0
        for j in combo:
            prod *= float(weights[j])
        total += prod
    return total


def test_normalisation_sums_to_one():
    for n, k in [(5, 2), (6, 3), (7, 4), (8, 1)]:
        log_w = torch.randn(n, dtype=DT)
        dist = enumerate_subset_distribution(log_w, k)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-10, (n, k, total)


def test_ek_matches_brute_force():
    for n, k in [(4, 2), (6, 3), (7, 5)]:
        w = torch.rand(n, dtype=DT) + 0.05
        e = elementary_symmetric(w, k)
        for m in range(k + 1):
            ref = _brute_force_e(w, m)
            assert math.isclose(float(e[m]), ref, rel_tol=1e-10, abs_tol=1e-12), (n, k, m)


def test_inclusion_probability_matches_bruteforce_and_sums_to_k():
    for n, k in [(5, 2), (6, 3), (7, 4)]:
        log_w = torch.randn(n, dtype=DT)
        pi = edge_inclusion_probability(log_w, k)
        # Eq. 19
        assert abs(float(pi.sum()) - k) < 1e-10, (n, k, float(pi.sum()))
        # brute-force marginal: P(j in S) = sum_{S: j in S} P(S)
        dist = enumerate_subset_distribution(log_w, k)
        for j in range(n):
            marg = sum(p for combo, p in dist.items() if j in combo)
            assert abs(float(pi[j]) - marg) < 1e-10, (n, k, j, float(pi[j]), marg)


def test_subset_log_probability_consistent():
    n, k = 6, 3
    log_w = torch.randn(n, dtype=DT)
    dist = enumerate_subset_distribution(log_w, k)
    for combo, p in dist.items():
        lp = subset_log_probability(log_w, combo, k)
        assert abs(float(torch.exp(lp)) - p) < 1e-12, (combo, float(torch.exp(lp)), p)


def test_sampler_matches_distribution_exactly():
    # The code-level "same sampler" assertion required by §4.1 / G2.
    for n, k in [(4, 2), (5, 2), (6, 3), (7, 4)]:
        log_w = torch.randn(n, dtype=DT)
        diag = verify_sampler_matches_distribution(log_w, k, atol=1e-10)
        assert diag["max_abs_error"] < 1e-10, (n, k, diag)
        assert abs(diag["sampler_total_mass"] - 1.0) < 1e-9, diag


def test_sampler_with_mask_and_shortage():
    # masked padding is exact: a node with n=5 valid but padded to 8
    log_w = torch.randn(8, dtype=DT)
    mask = torch.tensor([True, True, True, True, True, False, False, False])
    diag = verify_sampler_matches_distribution(log_w, 3, mask=mask)
    assert diag["max_abs_error"] < 1e-10
    # shortage must raise (spec §7.2 handled upstream, never pad by duplication)
    short_mask = torch.tensor([True, True, False, False, False, False, False, False])
    try:
        sample_k_subset(log_w.unsqueeze(0), 3, mask=short_mask.unsqueeze(0))
    except ValueError as exc:
        assert "candidate-shortage" in str(exc)
    else:
        raise AssertionError("expected ValueError on candidate shortage")


def test_monte_carlo_frequencies_converge():
    n, k = 5, 2
    log_w = torch.randn(n, dtype=DT)
    dist = enumerate_subset_distribution(log_w, k)
    gen = torch.Generator().manual_seed(1234)
    num = 40000
    batch = log_w.unsqueeze(0).expand(num, n).contiguous()
    sel = sample_k_subset(batch, k, generator=gen)
    counts: dict[tuple[int, ...], int] = {}
    sel_idx = [tuple(torch.nonzero(row).flatten().tolist()) for row in sel]
    for combo in sel_idx:
        assert len(combo) == k
        counts[combo] = counts.get(combo, 0) + 1
    for combo, p in dist.items():
        freq = counts.get(combo, 0) / num
        # 4-sigma binomial tolerance
        tol = 4.0 * math.sqrt(max(p * (1 - p), 1e-6) / num) + 1e-3
        assert abs(freq - p) < tol, (combo, freq, p, tol)


def test_gradient_matches_finite_difference():
    n, k = 6, 3
    log_w = torch.randn(n, dtype=DT, requires_grad=True)
    log_ek = log_elementary_symmetric(log_w, k)[k]
    log_ek.backward()
    grad = log_w.grad.detach().clone()
    eps = 1e-6
    for j in range(n):
        plus = log_w.detach().clone()
        minus = log_w.detach().clone()
        plus[j] += eps
        minus[j] -= eps
        fp = float(log_elementary_symmetric(plus, k)[k])
        fm = float(log_elementary_symmetric(minus, k)[k])
        fd = (fp - fm) / (2 * eps)
        rel = abs(fd - float(grad[j])) / (abs(fd) + 1e-8)
        assert rel < 1e-4, (j, fd, float(grad[j]), rel)


def test_uniform_weights_give_uniform_subsets():
    # spec §4.1: all a_ij equal -> uniform distinct-peer sampling
    n, k = 6, 3
    log_w = torch.zeros(n, dtype=DT)
    dist = enumerate_subset_distribution(log_w, k)
    expected = 1.0 / math.comb(n, k)
    for p in dist.values():
        assert abs(p - expected) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G2 tests passed.")
