"""G4 -- CDQ low-rank k-DPP query layer exactness (spec §9.4).

Acceptance: normaliser / subset law / inclusion marginals match brute force (<1e-10); the
exact sampler reproduces the law; the diagonal kernel collapses EXACTLY to the §4 ESP
product policy (CDQ is a strict generalisation); differentiable; k<=r enforced.
"""

from itertools import combinations

import pytest
import torch

from src.mainline.symmetric_polynomials import (
    edge_inclusion_probability,
    elementary_symmetric,
)
from src.sampling.dpp_query import (
    diagonal_diversity,
    enumerate_kdpp_distribution,
    kdpp_inclusion,
    kdpp_normalizer,
    kdpp_sample,
    kdpp_subset_log_prob,
    low_rank_kernel,
)


def _rand_kernel(d=6, r=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    quality = torch.rand(d, generator=g, dtype=torch.float64) * 2 + 0.1
    diversity = torch.randn(d, r, generator=g, dtype=torch.float64)
    return quality, diversity, low_rank_kernel(quality, diversity)


# ----------------------------------------------------------------- normaliser
def test_normalizer_matches_bruteforce():
    for k in (1, 2, 3):
        q, b, B = _rand_kernel(d=6, r=4, seed=k)
        ek_newton = float(kdpp_normalizer(B, k))
        _, ek_brute = enumerate_kdpp_distribution(B, k)
        assert abs(ek_newton - ek_brute) < 1e-9 * max(1.0, ek_brute)


def test_subset_distribution_normalises_and_matches_bruteforce():
    k = 2
    q, b, B = _rand_kernel(d=5, r=3, seed=7)
    dist, _ = enumerate_kdpp_distribution(B, k)
    assert abs(sum(dist.values()) - 1.0) < 1e-12
    for S in combinations(range(5), k):
        p = float(torch.exp(kdpp_subset_log_prob(B, S, k)))  # float64 throughout
        assert abs(p - dist[S]) < 1e-10


# ----------------------------------------------------------------- inclusion
def test_inclusion_matches_bruteforce_and_sums_to_k():
    k = 3
    q, b, B = _rand_kernel(d=7, r=5, seed=3)
    pi = kdpp_inclusion(q, b, k)
    assert abs(float(pi.sum()) - k) < 1e-9          # sum of inclusion marginals = k
    dist, _ = enumerate_kdpp_distribution(B, k)
    brute = torch.zeros(7, dtype=torch.float64)
    for S, p in dist.items():
        for j in S:
            brute[j] += p
    assert torch.allclose(pi, brute, atol=1e-9)


# ------------------------------------------------------ diagonal -> ESP (key)
def test_diagonal_kernel_recovers_esp_product_policy():
    d, k = 6, 2
    g = torch.Generator().manual_seed(11)
    quality = torch.rand(d, generator=g, dtype=torch.float64) * 3 + 0.2
    B = low_rank_kernel(quality, diagonal_diversity(d))     # L = diag(quality)
    # normaliser == elementary symmetric e_k of the quality weights
    e_esp = elementary_symmetric(quality, k)[k]
    assert abs(float(kdpp_normalizer(B, k)) - float(e_esp)) < 1e-10
    # inclusion == ESP inclusion with weights = quality
    pi_dpp = kdpp_inclusion(quality, diagonal_diversity(d), k)
    pi_esp = edge_inclusion_probability(torch.log(quality), k)
    assert torch.allclose(pi_dpp, pi_esp, atol=1e-10)
    # subset prob == product of quality / e_k
    dist, _ = enumerate_kdpp_distribution(B, k)
    for S in combinations(range(d), k):
        prod = float(quality[list(S)].prod())
        assert abs(dist[S] - prod / float(e_esp)) < 1e-10


def test_diversity_reduces_coselection_of_similar_peers():
    """The whole point of CDQ: two near-identical peers are less likely to be co-selected
    than two orthogonal peers of the same quality (ESP cannot express this)."""
    k = 2
    q = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    # peers 0,1 nearly identical direction; peer 2 orthogonal
    div = torch.tensor([[1.0, 0.0], [0.999, 0.0447], [0.0, 1.0]], dtype=torch.float64)
    B = low_rank_kernel(q, div)
    dist, _ = enumerate_kdpp_distribution(B, k)
    assert dist[(0, 1)] < dist[(0, 2)]              # similar pair suppressed
    assert dist[(0, 1)] < dist[(1, 2)]


# ----------------------------------------------------------------- sampler
def test_exact_sampler_matches_distribution():
    k = 2
    q, b, B = _rand_kernel(d=5, r=3, seed=21)
    dist, _ = enumerate_kdpp_distribution(B, k)
    gen = torch.Generator().manual_seed(123)
    counts: dict[tuple, int] = {}
    N = 40000
    for _ in range(N):
        S = tuple(kdpp_sample(B, k, generator=gen))
        counts[S] = counts.get(S, 0) + 1
    for S, p in dist.items():
        emp = counts.get(S, 0) / N
        assert abs(emp - p) < 0.02, f"subset {S}: emp={emp:.4f} exact={p:.4f}"


def test_sampler_always_returns_k_distinct():
    q, b, B = _rand_kernel(d=6, r=4, seed=5)
    gen = torch.Generator().manual_seed(1)
    for _ in range(200):
        S = kdpp_sample(B, 3, generator=gen)
        assert len(S) == 3 and len(set(S)) == 3


# ----------------------------------------------------------------- autograd
def test_differentiable_in_quality_and_diversity():
    d, r, k = 6, 4, 2
    g = torch.Generator().manual_seed(9)
    quality = (torch.rand(d, generator=g, dtype=torch.float64) * 2 + 0.1).requires_grad_(True)
    diversity = torch.randn(d, r, generator=g, dtype=torch.float64).requires_grad_(True)
    pi = kdpp_inclusion(quality, diversity, k)
    pi.sum().backward()
    assert quality.grad is not None and bool(torch.isfinite(quality.grad).all())
    assert diversity.grad is not None and bool(torch.isfinite(diversity.grad).all())
    # log-normaliser is also differentiable
    from src.sampling.dpp_query import kdpp_log_normalizer
    q2 = quality.detach().clone().requires_grad_(True)
    B = low_rank_kernel(q2, diversity.detach())
    kdpp_log_normalizer(B, k).backward()
    assert bool(torch.isfinite(q2.grad).all())


def test_k_greater_than_rank_raises():
    q, b, B = _rand_kernel(d=5, r=2, seed=0)
    with pytest.raises(ValueError):
        kdpp_normalizer(B, 3)        # k=3 > r=2
