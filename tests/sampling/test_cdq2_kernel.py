"""G-CDQ2-MATH: the CDQ 2.0 kernel ``L = D^{1/2}(I + eta ZZ^T) D^{1/2}`` (spec §9).

CDQ 2.0 fixes the prior round's low-rank ``L = BB^T`` CDQ, which only recovers ESP at FULL
rank (r = d). The new kernel keeps a full-rank diagonal quality ``D = diag(a_j)`` and adds a
rank-``r`` diversity correction ``eta ZZ^T`` (``Z`` unit-normalised rows), so

    eta = 0  =>  L = D  =>  P_CDQ = P_ESP        EXACTLY, at ANY rank r.

This is the central identifiability property (the model starts as ESP and learns diversity only
if the environment rewards it). The math contract this slice must satisfy:

* ESP exact degeneracy (normaliser / subset prob / inclusion) at eta = 0, for any Z, any r;
* the quality/diversity separation  det(L_S) = (prod_{j in S} a_j) det(I + eta Z_S Z_S^T);
* the k=2 closed form  det(I + eta Z_S Z_S^T) = (1+eta)^2 - eta^2 (z_j . z_l)^2;
* exact e_k(lambda(L)), subset probability, inclusion marginals -- all vs brute force < 1e-10;
* gradient (autograd) vs finite-difference rel-err < 1e-4 on a, Z, eta;
* near-linear in d (no dense d x d / N x N kernel): the algorithm only ever forms r x r and
  d-vectors -- exercised by a large-d run that still matches the homogeneity invariant.
"""

import math
from itertools import combinations

import pytest
import torch

from src.mainline.symmetric_polynomials import (
    edge_inclusion_probability,
    elementary_symmetric,
    subset_log_probability,
)

cdq2 = pytest.importorskip("src.sampling.cdq2_kernel")


def _rand(d, r, *, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    a = (0.2 + 1.5 * torch.rand(d, generator=g, dtype=torch.float64)) * scale
    Z = torch.randn(d, r, generator=g, dtype=torch.float64)
    return a, Z


# --------------------------------------------------------------------------------------------
# ESP exact degeneracy at eta = 0 (the identifiability anchor)
# --------------------------------------------------------------------------------------------

def test_eta_zero_normalizer_is_esp_ek():
    a, Z = _rand(7, 3, seed=1)
    for k in (1, 2, 3):
        esp_ek = elementary_symmetric(a, k)[k]
        cdq_ek = cdq2.cdq2_normalizer(a, Z, 0.0, k)
        assert torch.allclose(cdq_ek, esp_ek, rtol=1e-12, atol=1e-12)


def test_eta_zero_subset_prob_is_esp():
    a, Z = _rand(6, 4, seed=2)
    k = 3
    log_a = torch.log(a)
    for S in combinations(range(6), k):
        esp_lp = subset_log_probability(log_a, list(S), k)
        cdq_lp = cdq2.cdq2_subset_log_prob(a, Z, 0.0, list(S), k)
        assert torch.allclose(cdq_lp, esp_lp, atol=1e-10)


def test_eta_zero_inclusion_is_esp():
    a, Z = _rand(8, 5, seed=3)
    k = 3
    esp_pi = edge_inclusion_probability(torch.log(a), k)
    cdq_pi = cdq2.cdq2_inclusion(a, Z, 0.0, k)
    assert torch.allclose(cdq_pi, esp_pi, atol=1e-10)
    assert abs(float(cdq_pi.sum()) - k) < 1e-9


# --------------------------------------------------------------------------------------------
# quality / diversity separation + k=2 closed form
# --------------------------------------------------------------------------------------------

def test_separation_identity_matches_explicit_LS():
    """det(L_S) = (prod a_j) det(I + eta Z_S Z_S^T), checked against the explicit |S|x|S| L_S."""
    a, Z = _rand(6, 4, seed=4)
    eta = 0.7
    Zn = cdq2.cdq2_unit_normalize(Z)
    k = 3
    for S in combinations(range(6), k):
        idx = torch.tensor(S)
        D_half = torch.diag(torch.sqrt(a[idx]))
        Zs = Zn[idx]
        L_S = D_half @ (torch.eye(k, dtype=torch.float64) + eta * Zs @ Zs.T) @ D_half
        explicit = torch.logdet(L_S)
        formula = cdq2.cdq2_subset_logdet(a, Z, eta, list(S))
        assert torch.allclose(formula, explicit, atol=1e-10)


def test_k2_closed_form():
    eta = 0.9
    g = torch.Generator().manual_seed(5)
    for _ in range(20):
        zj = torch.randn(4, generator=g, dtype=torch.float64)
        zl = torch.randn(4, generator=g, dtype=torch.float64)
        zj = zj / zj.norm()
        zl = zl / zl.norm()
        rho = float(zj @ zl)
        closed = (1 + eta) ** 2 - eta ** 2 * rho ** 2
        Zs = torch.stack([zj, zl])
        explicit = float(torch.det(torch.eye(2, dtype=torch.float64) + eta * Zs @ Zs.T))
        assert abs(closed - explicit) < 1e-12
        # the helper re-applies the spec's z/(||z||+eps) normalisation, a ~1e-12 perturbation
        assert abs(float(cdq2.cdq2_k2_diversity_factor(zj, zl, eta)) - closed) < 1e-9


def test_more_similar_lowers_joint_weight():
    """The diversity mechanism: more-similar peers => smaller det(I+eta Z_S Z_S^T) => lower
    joint-selection weight (this is what ESP's product law cannot express)."""
    eta = 1.0
    base = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    prev = None
    for rho in (0.0, 0.3, 0.6, 0.9):
        other = torch.tensor([rho, math.sqrt(max(1 - rho ** 2, 0.0)), 0.0, 0.0], dtype=torch.float64)
        f = float(cdq2.cdq2_k2_diversity_factor(base, other, eta))
        if prev is not None:
            assert f < prev - 1e-9         # strictly decreasing in similarity
        prev = f


# --------------------------------------------------------------------------------------------
# exact normaliser / subset prob / inclusion vs brute force (< 1e-10)
# --------------------------------------------------------------------------------------------

def test_normalizer_matches_bruteforce():
    a, Z = _rand(9, 4, seed=6)
    for eta in (0.25, 1.0, 3.0):
        for k in (1, 2, 3, 4):
            bf = cdq2.bruteforce_cdq2_normalizer(a, Z, eta, k)
            fast = cdq2.cdq2_normalizer(a, Z, eta, k)
            assert abs(float(fast) - float(bf)) <= 1e-10 * (1 + abs(float(bf)))


def test_normalizer_matches_bruteforce_low_rank_r_lt_k():
    """Embedding rank r < k: det(I + eta Z_S Z_S^T) is rank-deficient (eigenvalues 1 outside
    the r-dim span) but still exact -- the r x r Sylvester / z-series path must handle it."""
    a, Z = _rand(8, 2, seed=14)            # r = 2 < k
    for eta in (0.5, 2.0):
        for k in (3, 4):
            bf = cdq2.bruteforce_cdq2_normalizer(a, Z, eta, k)
            fast = cdq2.cdq2_normalizer(a, Z, eta, k)
            assert abs(float(fast) - float(bf)) <= 1e-10 * (1 + abs(float(bf)))
    # and eta=0 still collapses to ESP at r < k
    assert torch.allclose(cdq2.cdq2_normalizer(a, Z, 0.0, 4), elementary_symmetric(a, 4)[4],
                          rtol=1e-12, atol=1e-12)


def test_subset_distribution_matches_bruteforce():
    a, Z = _rand(7, 3, seed=7)
    eta, k = 1.3, 3
    dist = cdq2.cdq2_enumerate_distribution(a, Z, eta, k)   # {S: P(S)}
    assert abs(sum(dist.values()) - 1.0) < 1e-10
    log_norm = cdq2.cdq2_log_normalizer(a, Z, eta, k)
    for S, p in dist.items():
        lp = cdq2.cdq2_subset_log_prob(a, Z, eta, list(S), k)
        # consistency: P(S) = det(L_S)/e_k recovered from the fast normaliser too
        ld = cdq2.cdq2_subset_logdet(a, Z, eta, list(S))
        assert abs(float(torch.exp(ld - log_norm)) - p) < 1e-10
        assert abs(float(torch.exp(lp)) - p) < 1e-10


def test_inclusion_matches_bruteforce_and_sums_to_k():
    a, Z = _rand(8, 4, seed=8)
    eta, k = 0.8, 3
    dist = cdq2.cdq2_enumerate_distribution(a, Z, eta, k)
    bf_pi = torch.zeros(8, dtype=torch.float64)
    for S, p in dist.items():
        for j in S:
            bf_pi[j] += p
    fast_pi = cdq2.cdq2_inclusion(a, Z, eta, k)
    assert torch.allclose(fast_pi, bf_pi, atol=1e-9)
    assert abs(float(fast_pi.sum()) - k) < 1e-9
    assert bool((fast_pi >= -1e-12).all()) and bool((fast_pi <= 1 + 1e-9).all())


# --------------------------------------------------------------------------------------------
# gradients (autograd vs finite difference)
# --------------------------------------------------------------------------------------------

def _finite_diff(f, x, eps=1e-6):
    g = torch.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.numel()):
        orig = float(flat[i])
        flat[i] = orig + eps
        fp = float(f())
        flat[i] = orig - eps
        fm = float(f())
        flat[i] = orig
        g.reshape(-1)[i] = (fp - fm) / (2 * eps)
    return g


def test_log_normalizer_gradient_wrt_a():
    a, Z = _rand(6, 3, seed=9)
    eta, k = 1.1, 3
    a = a.clone().requires_grad_(True)
    val = cdq2.cdq2_log_normalizer(a, Z, eta, k)
    val.backward()
    ana = a.grad.detach().clone()
    with torch.no_grad():
        num = _finite_diff(lambda: cdq2.cdq2_log_normalizer(a.detach(), Z, eta, k), a.detach())
    rel = (ana - num).norm() / (num.norm() + 1e-30)
    assert float(rel) < 1e-4


def test_log_normalizer_gradient_wrt_Z_and_eta():
    a, Z = _rand(6, 3, seed=10)
    k = 3
    Zv = Z.clone().requires_grad_(True)
    eta = torch.tensor(0.9, dtype=torch.float64, requires_grad=True)
    val = cdq2.cdq2_log_normalizer(a, Zv, eta, k)
    val.backward()
    gZ = Zv.grad.detach().clone()
    geta = float(eta.grad)
    with torch.no_grad():
        numZ = _finite_diff(lambda: cdq2.cdq2_log_normalizer(a, Zv.detach(), float(eta), k), Zv.detach())

        def f_eta():
            return cdq2.cdq2_log_normalizer(a, Z, float(eta_scalar[0]), k)
        eta_scalar = [float(eta)]
        e0 = 1e-6
        eta_scalar[0] = float(eta) + e0
        fp = float(cdq2.cdq2_log_normalizer(a, Z, eta_scalar[0], k))
        eta_scalar[0] = float(eta) - e0
        fm = float(cdq2.cdq2_log_normalizer(a, Z, eta_scalar[0], k))
        num_eta = (fp - fm) / (2 * e0)
    relZ = (gZ - numZ).norm() / (numZ.norm() + 1e-30)
    assert float(relZ) < 1e-4
    assert abs(geta - num_eta) / (abs(num_eta) + 1e-30) < 1e-4


# --------------------------------------------------------------------------------------------
# batching, scale stability, near-linear (no dense d x d)
# --------------------------------------------------------------------------------------------

def test_batched_normalizer_matches_per_source():
    g = torch.Generator().manual_seed(11)
    a = 0.3 + torch.rand(5, 6, generator=g, dtype=torch.float64)
    Z = torch.randn(5, 6, 3, generator=g, dtype=torch.float64)
    eta, k = 0.7, 3
    batched = cdq2.cdq2_normalizer(a, Z, eta, k)
    for b in range(5):
        single = cdq2.cdq2_normalizer(a[b], Z[b], eta, k)
        assert torch.allclose(batched[b], single, rtol=1e-10, atol=1e-12)


def test_homogeneity_degree_k_in_quality():
    """e_k(lambda(L)) is degree-k homogeneous in a (scaling a by s scales L by s). Used as a
    large-d correctness invariant where brute force is infeasible (near-linear path only)."""
    a, Z = _rand(400, 4, seed=12, scale=1.0)
    eta, k = 0.9, 3
    s = 2.5
    base = cdq2.cdq2_log_normalizer(a, Z, eta, k)
    scaled = cdq2.cdq2_log_normalizer(s * a, Z, eta, k)
    assert math.isfinite(float(base)) and math.isfinite(float(scaled))
    assert abs(float(scaled - base) - k * math.log(s)) < 1e-8


def test_scale_stability_large_quality():
    a, Z = _rand(10, 3, seed=13, scale=50.0)   # large a would overflow a naive product
    eta, k = 1.0, 3
    log_norm = cdq2.cdq2_log_normalizer(a, Z, eta, k)
    assert math.isfinite(float(log_norm))
    bf = cdq2.bruteforce_cdq2_normalizer(a, Z, eta, k)
    assert abs(float(log_norm) - math.log(float(bf))) < 1e-8


# --------------------------------------------------------------------------------------------
# adversarial-audit regressions (wf_c696e8e5-3d6): wide dynamic range / collinear / eta=0 grad
# / heterogeneous inclusion / inclusion double-backprop -- all 5 confirmed findings
# --------------------------------------------------------------------------------------------

def test_normalizer_wide_dynamic_range_no_cancellation():
    """Finding #1/#3 (HIGH): wide-dynamic-range a made the z-series e_k cancel to a negative/NaN
    value. The eigenvalue route is cancellation-free and stays exact + finite."""
    g = torch.Generator().manual_seed(99)
    Z = torch.randn(6, 3, generator=g, dtype=torch.float64)
    for a in (torch.tensor([1e-8, 1e8, 1.0, 5.0, 1e-3, 1e5], dtype=torch.float64),
              torch.tensor([1e6, 1e-3, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)):
        for eta in (0.9, 2.0):
            for k in (2, 3, 4):
                ln = cdq2.cdq2_log_normalizer(a, Z, eta, k)
                bf = cdq2.bruteforce_cdq2_normalizer(a, Z, eta, k)
                assert math.isfinite(float(ln))
                assert abs(float(ln) - math.log(float(bf))) < 1e-9


def test_inclusion_heterogeneous_quality_nonnegative():
    """Finding #3 (HIGH): heterogeneous quality drove inclusion marginals NEGATIVE. They must
    match the brute-force k-DPP marginals and stay in [0,1]."""
    a = torch.tensor([1e6, 1e-3, 1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    Z = torch.randn(6, 3, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    eta, k = 0.9, 3
    dist = cdq2.cdq2_enumerate_distribution(a, Z, eta, k)
    bf_pi = torch.zeros(6, dtype=torch.float64)
    for S, p in dist.items():
        for j in S:
            bf_pi[j] += p
    pi = cdq2.cdq2_inclusion(a, Z, eta, k)
    assert bool((pi >= -1e-12).all()) and bool((pi <= 1 + 1e-9).all())
    assert torch.allclose(pi, bf_pi, atol=1e-9)
    assert abs(float(pi.sum()) - k) < 1e-9


def test_normalizer_collinear_rows_large_eta_finite_and_exact():
    """Finding #2 (HIGH): near-collinear diversity rows with large eta gave NaN via log(neg).
    e_k(lambda(L)) is provably positive (L SPD) -- it must stay finite and match brute force."""
    a = torch.ones(6, dtype=torch.float64)
    base = torch.tensor([1.0, 0.0], dtype=torch.float64)
    noise = 1e-9 * torch.randn(6, 2, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    Z = base.expand(6, 2) + noise               # 6 near-identical rows
    for eta in (1e3, 1e6, 1e9):
        for k in (2, 3, 4):
            ln = cdq2.cdq2_log_normalizer(a, Z, eta, k)
            bf = cdq2.bruteforce_cdq2_normalizer(a, Z, eta, k)
            assert math.isfinite(float(ln))
            assert abs(float(ln) - math.log(float(bf))) < 1e-7


def test_eta_gradient_at_exactly_zero():
    """Finding #4 (MED): at exactly eta=0 the fast path detached eta from autograd (backward
    raised / eta.grad was None). The identifiability anchor needs a LIVE eta gradient at 0."""
    a, Z = _rand(7, 3, seed=1)
    k = 3
    eta = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    val = cdq2.cdq2_log_normalizer(a, Z, eta, k)
    assert val.requires_grad and val.grad_fn is not None
    val.backward()                                              # must NOT raise
    assert eta.grad is not None
    # one-sided finite difference of d log e_k / d eta at 0
    e0 = 1e-7
    fp = float(cdq2.cdq2_log_normalizer(a, Z, e0, k))
    f0 = float(cdq2.cdq2_log_normalizer(a, Z, 0.0, k))
    fd = (fp - f0) / e0
    assert abs(float(eta.grad) - fd) / (abs(fd) + 1e-30) < 1e-4
    # also as the SOLE grad leaf via cdq2_inclusion
    eta2 = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    pi = cdq2.cdq2_inclusion(a, Z, eta2, k)
    geta = torch.autograd.grad(pi.sum(), eta2, allow_unused=True)[0]
    assert geta is not None


def test_inclusion_loss_backprops_to_logits():
    """Finding #5 (HIGH): backprop of an inclusion-based loss to the GNN logits s (a=exp(s)) was
    silently wrong (~0.8 rel-err) because the ESP factor's first-order grad had no grad_fn. The
    eigenvalue ESP recursion (and the _LogAddExp double-backward fix) make it correct."""
    for eta in (0.0, 0.9):
        s = torch.randn(6, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
        s = s.clone().requires_grad_(True)
        Z = torch.randn(6, 3, generator=torch.Generator().manual_seed(5), dtype=torch.float64)
        k = 3
        pi = cdq2.cdq2_inclusion(torch.exp(s), Z, eta, k)
        loss = (pi ** 2).sum()
        (ana,) = torch.autograd.grad(loss, s)
        with torch.no_grad():
            def f():
                return float((cdq2.cdq2_inclusion(torch.exp(s.detach()), Z, eta, k) ** 2).sum())
            num = _finite_diff(f, s.detach(), eps=1e-6)
        rel = (ana - num).norm() / (num.norm() + 1e-30)
        assert float(rel) < 1e-4, f"eta={eta} rel-err {float(rel)}"
