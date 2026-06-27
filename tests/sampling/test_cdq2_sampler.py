"""G-CDQ2-MATH (S12d): the exact k-DPP sampler for the CDQ 2.0 kernel (spec §9).

L = D^{1/2}(I + eta ZZ^T) D^{1/2} is a full-rank d x d SPD kernel, so the standard
Kulesza-Taskar k-DPP eigen-sampler applies directly to its d x d eigendecomposition (a
per-source matrix, never N x N). This sampler is used ONLY by the dynamic Monte-Carlo judge
(no gradient).

Validation: it is the standard EXACT k-DPP sampler, so its analytic law is P(S) = det(L_S)/e_k
by construction; we check (a) it always returns exactly k distinct candidates, (b) its empirical
subset frequencies converge to the exact cdq2_enumerate_distribution, (c) its empirical inclusion
frequencies converge to the exact cdq2_inclusion, (d) at eta=0 it reproduces the ESP sampler's
law, and (e) the diversity correction (eta>0) demonstrably avoids co-selecting similar peers.
"""

import math

import pytest
import torch

from src.sampling import cdq2_kernel as cdq2
from src.mainline.symmetric_polynomials import edge_inclusion_probability

sample = pytest.importorskip("src.sampling.cdq2_kernel").__dict__.get("cdq2_sample")


def _rand(d, r, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = 0.3 + 1.5 * torch.rand(d, generator=g, dtype=torch.float64)
    Z = torch.randn(d, r, generator=g, dtype=torch.float64)
    return a, Z


def test_sample_returns_k_distinct_in_range():
    a, Z = _rand(7, 3, seed=1)
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        S = cdq2.cdq2_sample(a, Z, 1.0, 3, generator=gen)
        assert len(S) == 3
        assert len(set(S)) == 3
        assert all(0 <= j < 7 for j in S)


def test_sample_k_too_large_raises():
    a, Z = _rand(4, 3, seed=2)
    with pytest.raises(ValueError):
        cdq2.cdq2_sample(a, Z, 1.0, 5)            # k > d


def test_empirical_subset_distribution_converges():
    a, Z = _rand(4, 3, seed=3)
    eta, k = 1.2, 2
    exact = cdq2.cdq2_enumerate_distribution(a, Z, eta, k)
    gen = torch.Generator().manual_seed(7)
    N = 40000
    counts = {S: 0 for S in exact}
    for _ in range(N):
        S = tuple(cdq2.cdq2_sample(a, Z, eta, k, generator=gen))
        counts[S] += 1
    for S, p in exact.items():
        assert abs(counts[S] / N - p) < 0.02, (S, counts[S] / N, p)


def test_empirical_inclusion_converges_to_exact():
    a, Z = _rand(6, 3, seed=4)
    eta, k = 0.9, 3
    exact_pi = cdq2.cdq2_inclusion(a, Z, eta, k)
    gen = torch.Generator().manual_seed(11)
    N = 30000
    inc = torch.zeros(6, dtype=torch.float64)
    for _ in range(N):
        for j in cdq2.cdq2_sample(a, Z, eta, k, generator=gen):
            inc[j] += 1
    inc /= N
    assert torch.allclose(inc, exact_pi, atol=0.02)
    assert abs(float(inc.sum()) - k) < 0.05


def test_eta_zero_matches_esp_sampler_law():
    """eta=0 => L=diag(a) => the k-DPP sampler reduces to the ESP elementary-symmetric subset
    sampler; its empirical inclusion matches the exact ESP inclusion."""
    a, Z = _rand(6, 4, seed=5)
    k = 3
    esp_pi = edge_inclusion_probability(torch.log(a), k)
    gen = torch.Generator().manual_seed(13)
    N = 30000
    inc = torch.zeros(6, dtype=torch.float64)
    for _ in range(N):
        for j in cdq2.cdq2_sample(a, Z, 0.0, k, generator=gen):
            inc[j] += 1
    inc /= N
    assert torch.allclose(inc, esp_pi, atol=0.02)


def test_sampler_huge_magnitude_no_overflow():
    """Robustness (sampler review): at uniformly HUGE quality magnitude (a ~ 1e200, where an
    unscaled L would overflow eigh to inf/NaN) the mean scale-out keeps the sampler finite; it
    must still return exactly k distinct valid candidates and be reproducible (law is
    scale-invariant). Moderate spread so every candidate stays representable (no shortage)."""
    a = 1e200 * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float64)
    Z = torch.randn(6, 3, generator=torch.Generator().manual_seed(21), dtype=torch.float64)
    g1 = torch.Generator().manual_seed(2)
    g2 = torch.Generator().manual_seed(2)
    for _ in range(20):
        S1 = cdq2.cdq2_sample(a, Z, 1.5, 3, generator=g1)
        S2 = cdq2.cdq2_sample(a, Z, 1.5, 3, generator=g2)
        assert len(S1) == 3 and len(set(S1)) == 3
        assert all(0 <= j < 6 for j in S1)
        assert S1 == S2                                   # reproducible under the same generator
    # the scale-invariant law: huge-magnitude inclusion == unit-scale inclusion
    pi_huge = cdq2.cdq2_inclusion(a, Z, 1.5, 3)
    pi_unit = cdq2.cdq2_inclusion(a / a.mean(), Z, 1.5, 3)
    assert torch.allclose(pi_huge, pi_unit, atol=1e-9)


def test_diversity_sampler_avoids_similar_peers():
    """eta>0 lowers the probability of co-selecting two SIMILAR (collinear) peers vs ESP.

    Two collinear clusters {0,1} (e1) and {2,3} (e2 orthogonal). Uniform quality => ESP is
    uniform over the 6 pairs (P(same-cluster pair)=1/3); the diversity correction must push the
    sampler toward the 4 cross-cluster pairs. Checked analytically AND empirically.
    """
    a = torch.ones(4, dtype=torch.float64)
    Z = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    k = 2
    # analytic: P(same-cluster) drops below 1/3 as eta rises
    d0 = cdq2.cdq2_enumerate_distribution(a, Z, 0.0, k)
    d1 = cdq2.cdq2_enumerate_distribution(a, Z, 3.0, k)
    same0 = d0[(0, 1)] + d0[(2, 3)]
    same1 = d1[(0, 1)] + d1[(2, 3)]
    assert abs(same0 - 1.0 / 3.0) < 1e-9
    assert same1 < same0 - 0.1
    # empirical: the sampler reproduces the reduced same-cluster rate
    gen = torch.Generator().manual_seed(17)
    N = 20000
    same = 0
    for _ in range(N):
        S = tuple(cdq2.cdq2_sample(a, Z, 3.0, k, generator=gen))
        if S in ((0, 1), (2, 3)):
            same += 1
    assert abs(same / N - same1) < 0.02
