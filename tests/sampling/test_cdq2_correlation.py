"""G-CDQ2-GRADIENT (S13): the differentiable pairwise-correlation training objective (spec §11).

    L_corr = sum_i sum_{j<l} pi^(2)_{i,jl} R_{jl}

pi^(2)_{i,jl} = P(j,l in S_i) is the CDQ 2.0 k-DPP PAIRWISE inclusion; R_{jl} is the evidence-error
correlation (deployment-observable proxy, from the overlapping environment). The objective is the
EXPECTED within-selected-set total correlation E_{S~kDPP}[ sum_{j<l in S} R_{jl} ] -- minimising it
trains the diversity head (Z, eta) to avoid co-selecting correlated peers.

pi^(2) must be FIRST-ORDER differentiable (the training loss backprops once through it; a second-order
autograd of e_k would make that a fragile third-order op at degenerate eigenvalues). We use the Schur-
complement quotient identity sum_{S>=T} det(L_S) = det(L_T) e_{k-|T|}(lambda(L/L_T)), so

    pi^(2)_{jl} = det(L_{{j,l}}) e_{k-2}(lambda(L/L_{{j,l}})) / e_k(lambda(L)),

all via eigvalsh + the cancellation-free positive ESP recursion (smooth first-order gradient). Spec §11
bans float()/.item()/detached scores on the training path.
"""

import math
from itertools import combinations

import pytest
import torch

from src.sampling import cdq2_kernel as cdq2

corr = pytest.importorskip("src.sampling.cdq2_correlation")


def _rand(d, r, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = 0.3 + 1.5 * torch.rand(d, generator=g, dtype=torch.float64)
    Z = torch.randn(d, r, generator=g, dtype=torch.float64)
    return a, Z


def _bf_pairwise(a, Z, eta, k):
    """Brute-force Pi^(2)[j,l] = sum_{S∋j,l} det(L_S) / e_k (and diag = single inclusion)."""
    d = a.shape[0]
    Zn = cdq2.cdq2_unit_normalize(Z)
    e = torch.as_tensor(eta, dtype=torch.float64)
    ek = float(cdq2.bruteforce_cdq2_normalizer(a, Z, eta, k))
    Pi = torch.zeros(d, d, dtype=torch.float64)
    for S in combinations(range(d), k):
        idx = torch.tensor(S)
        Zs = Zn[idx]
        det = float(a[idx].prod() * torch.det(torch.eye(len(S), dtype=torch.float64) + e * Zs @ Zs.T))
        for j in S:
            Pi[j, j] += det
        for j, l in combinations(S, 2):
            Pi[j, l] += det
            Pi[l, j] += det
    return Pi / ek


# --------------------------------------------------------------------------------------------
# pairwise inclusion: exactness + structural identities
# --------------------------------------------------------------------------------------------

def test_pairwise_inclusion_matches_bruteforce():
    for seed, (d, r) in enumerate([(6, 3), (7, 4), (8, 2)]):       # last is r < k
        a, Z = _rand(d, r, seed=seed)
        for eta in (0.4, 1.5):
            for k in (2, 3, 4):
                Pi = corr.cdq2_pairwise_inclusion(a, Z, eta, k)
                bf = _bf_pairwise(a, Z, eta, k)
                assert torch.allclose(Pi, bf, atol=1e-9), (d, r, eta, k)


def test_pairwise_consistency_identities():
    a, Z = _rand(7, 3, seed=10)
    eta, k = 0.9, 3
    Pi = corr.cdq2_pairwise_inclusion(a, Z, eta, k)
    pi1 = cdq2.cdq2_inclusion(a, Z, eta, k)
    # diagonal == single inclusion
    assert torch.allclose(torch.diagonal(Pi), pi1, atol=1e-9)
    # sum_{l != j} pi^(2)_{jl} = (k-1) pi_j  (given j in S, k-1 other members)
    offdiag_rowsum = Pi.sum(dim=1) - torch.diagonal(Pi)
    assert torch.allclose(offdiag_rowsum, (k - 1) * pi1, atol=1e-9)
    # sum_{j<l} pi^(2)_{jl} = C(k,2)
    total = (Pi.sum() - torch.diagonal(Pi).sum()) / 2
    assert abs(float(total) - k * (k - 1) / 2) < 1e-9


def test_eta_zero_pairwise_is_esp():
    a, Z = _rand(7, 4, seed=11)
    k = 3
    Pi = corr.cdq2_pairwise_inclusion(a, Z, 0.0, k)
    bf = _bf_pairwise(a, Z, 0.0, k)       # at eta=0 this is the ESP pairwise inclusion
    assert torch.allclose(Pi, bf, atol=1e-9)


# --------------------------------------------------------------------------------------------
# correlation cost: exactness, gradient, no forbidden ops
# --------------------------------------------------------------------------------------------

def _sym_R(d, *, seed):
    g = torch.Generator().manual_seed(seed)
    M = torch.randn(d, d, generator=g, dtype=torch.float64)
    R = 0.5 * (M + M.T)
    R.fill_diagonal_(0.0)
    return R


def test_correlation_cost_matches_bruteforce():
    a, Z = _rand(6, 3, seed=12)
    R = _sym_R(6, seed=99)
    eta, k = 1.2, 3
    cost = corr.cdq2_correlation_cost(a, Z, eta, R, k)
    Pi = _bf_pairwise(a, Z, eta, k)
    bf = sum(float(R[j, l]) * float(Pi[j, l]) for j, l in combinations(range(6), 2))
    assert abs(float(cost) - bf) < 1e-9


def test_correlation_cost_gradient_vs_finite_diff():
    a, Z = _rand(6, 3, seed=13)
    R = _sym_R(6, seed=7)
    k = 3
    a = a.clone().requires_grad_(True)
    Zv = Z.clone().requires_grad_(True)
    eta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    cost = corr.cdq2_correlation_cost(a, Zv, eta, R, k)
    cost.backward()
    ga, gZ, ge = a.grad.clone(), Zv.grad.clone(), float(eta.grad)

    def f(av, Zv_, ev):
        return float(corr.cdq2_correlation_cost(av, Zv_, ev, R, k))

    eps = 1e-6
    # a
    na = torch.zeros_like(a)
    for j in range(6):
        ap = a.detach().clone(); ap[j] += eps
        am = a.detach().clone(); am[j] -= eps
        na[j] = (f(ap, Z, 0.8) - f(am, Z, 0.8)) / (2 * eps)
    # Z
    nZ = torch.zeros_like(Z)
    for j in range(6):
        for c in range(3):
            zp = Z.clone(); zp[j, c] += eps
            zm = Z.clone(); zm[j, c] -= eps
            nZ[j, c] = (f(a.detach(), zp, 0.8) - f(a.detach(), zm, 0.8)) / (2 * eps)
    ne = (f(a.detach(), Z, 0.8 + eps) - f(a.detach(), Z, 0.8 - eps)) / (2 * eps)
    assert float((ga - na).norm() / (na.norm() + 1e-30)) < 1e-4
    assert float((gZ - nZ).norm() / (nZ.norm() + 1e-30)) < 1e-4
    assert abs(ge - ne) / (abs(ne) + 1e-30) < 1e-4


def test_no_forbidden_ops_on_training_path():
    """Spec §11: no float() / .item() / detached score on the DIFFERENTIABLE path. Inspect the
    actual training-path functions (not the tests-only brute-force reference)."""
    import inspect
    from src.sampling import cdq2_correlation as mod
    for fn in (mod.cdq2_pairwise_inclusion, mod.cdq2_correlation_cost,
               mod._conditioned_numerator, mod._build_L):
        src = inspect.getsource(fn)
        for line in src.splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            assert ".item(" not in line, (fn.__name__, line)
            assert "float(" not in line, (fn.__name__, line)
            if ".detach(" in line:                          # only allowed in a bool(...) guard
                assert "bool(" in line, (fn.__name__, line)


# --------------------------------------------------------------------------------------------
# C3 contract: the correlation gradient REACHES the CDQ parameters and DISCRIMINATES structure
# --------------------------------------------------------------------------------------------

def test_matched_marginal_gradient_discrimination():
    """C3 (Mechanism Identifiability): with IDENTICAL marginals (same q_i) but DIFFERENT covariance
    R, the correlation objective produces DIFFERENT gradients to the CDQ diversity params -- i.e.
    the correlation structure is reachable by / identifiable from the gradient (the lever a
    marginal-only policy lacks). BOTH arms carry NON-ZERO correlation (else the test degenerates to
    a presence-vs-absence check that a structure-blind gradient would also pass)."""
    from src.environment.overlapping_evidence import (
        OverlappingEvidenceModel,
        matched_marginal_shared,
        overlapping_pairwise_correlation_matrix,
    )

    d, r, k = 8, 4, 3
    target_p_node = 0.3
    # TWO sensor groups => HETEROGENEOUS (block) correlation: within-group pairs correlated,
    # cross-group not. (A uniform R would make the cost constant — sum_{j<l} pi^(2) = C(k,2) — so
    # diversity has no lever against EXCHANGEABLE correlation; the heterogeneity is the D18 fix.)
    sensor = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)

    def model_R(p_shared):
        p_sensor, p_node = matched_marginal_shared(target_p_node, p_shared)
        m = OverlappingEvidenceModel(
            road_of=torch.zeros(d, dtype=torch.long),
            sensor_of=sensor,
            map_of=torch.zeros(d, dtype=torch.long),
            p_road=torch.tensor([0.0], dtype=torch.float64),
            p_sensor=torch.tensor([p_sensor, p_sensor], dtype=torch.float64),
            p_map=torch.tensor([0.0], dtype=torch.float64),
            p_node=torch.full((d,), p_node, dtype=torch.float64))
        R = overlapping_pairwise_correlation_matrix(m).clone()
        R.fill_diagonal_(0.0)
        return m.correct_observation_prob(), R

    q_lo, R_lo = model_R(0.10)
    q_hi, R_hi = model_R(0.20)
    assert torch.allclose(q_lo, q_hi, atol=1e-12)            # IDENTICAL marginals
    assert float(R_lo.norm()) > 1e-3 and float(R_hi.norm()) > 1e-3   # BOTH non-zero structure
    assert float((R_hi - R_lo).abs().max()) > 1e-3          # genuinely different covariance
    # diversity embeddings ALIGNED with the sensor groups (the realistic learned correlate): within
    # a group the rows are similar, so raising eta avoids co-selecting the correlated peers.
    g = torch.Generator().manual_seed(5)
    base = torch.zeros(d, r, dtype=torch.float64)
    base[:4, 0] = 1.0
    base[4:, 1] = 1.0
    Z = base + 0.1 * torch.randn(d, r, generator=g, dtype=torch.float64)
    a = 0.5 + 0.5 * torch.rand(d, generator=g, dtype=torch.float64)

    def grad_eta(R):
        eta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        Zv = Z.clone().requires_grad_(True)
        cost = corr.cdq2_correlation_cost(a, Zv, eta, R, k)
        cost.backward()
        return float(eta.grad), Zv.grad.clone()

    ge_lo, gZ_lo = grad_eta(R_lo)
    ge_hi, gZ_hi = grad_eta(R_hi)
    # identical marginals => any gradient difference is attributable PURELY to the covariance
    assert abs(ge_hi - ge_lo) > 1e-4
    assert float((gZ_hi - gZ_lo).norm()) > 1e-4


def test_correlation_cost_huge_quality_overflow_safe():
    """Robustness (audit complexity lens): the mean scale-out keeps pi^(2)/L_corr finite at huge
    quality magnitude (where an un-scaled L would overflow eigh); the cost is scale-invariant."""
    R = _sym_R(6, seed=3)
    a, Z = _rand(6, 3, seed=4)
    k = 3
    cost_huge = corr.cdq2_correlation_cost(1e200 * a, Z, 0.9, R, k)
    cost_unit = corr.cdq2_correlation_cost(a, Z, 0.9, R, k)
    assert math.isfinite(float(cost_huge))
    assert abs(float(cost_huge) - float(cost_unit)) < 1e-9      # scale-invariant
