"""G-CDQ2-MATH (S12c): the heterogeneous determinantal quorum P(m,n) under the CDQ 2.0 kernel
(spec §10).

With g_j(x,y) = p^0_j + p^+_j x + p^-_j y and G = diag(g_j),

    P(m,n) = [z^k x^m y^n] det(I + z L G) / e_k(lambda(L)),   L = D^{1/2}(I+eta ZZ^T)D^{1/2}.

We evaluate [z^k] det(I + z L G) = e_k(lambda(L G)) = e_k(lambda((I+eta ZZ^T) diag(a g))) on the
(x,y) integer grid via the STABLE eigenvalue route of the kernel module, then recover the P(m,n)
coefficients by 2-D inverse-Vandermonde -- exactly mirroring the diagonal-reduction quorum for the
old L = BB^T kernel, but with the CDQ 2.0 evaluator.

Independent references (FREE): L = B_tilde B_tilde^T with B_tilde = D^{1/2}[I | sqrt(eta) Z], so the
existing low-rank quorum bruteforce_determinantal_quorum(B_tilde,..) (subset x 3^k enumeration) and
determinantal_quorum_distribution(B_tilde,..) (principal-minor grid) BOTH give the exact CDQ 2.0
quorum -- two independent checks of the new evaluator.
"""

import math

import pytest
import torch

from src.sampling.cdq2_kernel import cdq2_unit_normalize
from src.sampling.determinantal_quorum import (
    bruteforce_determinantal_quorum,
    determinantal_quorum_distribution,
)
from src.sampling.dpp_query import diagonal_diversity, low_rank_kernel

cdq2q = pytest.importorskip("src.sampling.cdq2_quorum")


def _inputs(d=6, r=3, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = 0.3 + 1.2 * torch.rand(d, generator=g, dtype=torch.float64)
    Z = torch.randn(d, r, generator=g, dtype=torch.float64)
    pc = 0.5 * torch.rand(d, generator=g, dtype=torch.float64)
    pw = 0.3 * torch.rand(d, generator=g, dtype=torch.float64)   # pc+pw <= 0.8 < 1
    return a, Z, pc, pw


def _btilde(a, Z, eta):
    """L = B_tilde B_tilde^T = D^{1/2}(I + eta Zn Zn^T) D^{1/2}."""
    Zn = cdq2_unit_normalize(Z)
    left = torch.diag(torch.sqrt(a))
    right = torch.sqrt(a * eta).unsqueeze(-1) * Zn
    return torch.cat([left, right], dim=-1)                       # [d, d+r]


def test_quorum_sums_to_one():
    a, Z, pc, pw = _inputs(seed=1)
    for eta in (0.0, 0.7, 2.5):
        for k in (2, 3, 4):
            P = cdq2q.cdq2_quorum_distribution(a, Z, eta, pc, pw, k)
            assert P.shape == (k + 1, k + 1)
            assert abs(float(P.sum()) - 1.0) < 1e-10
            assert bool((P >= -1e-12).all())


def test_quorum_matches_bruteforce():
    a, Z, pc, pw = _inputs(seed=2)
    for eta in (0.5, 2.0):
        for k in (2, 3):
            P = cdq2q.cdq2_quorum_distribution(a, Z, eta, pc, pw, k)
            bf = bruteforce_determinantal_quorum(_btilde(a, Z, eta), pc, pw, k, alpha=max(1, (k // 2) + 1))
            for (m, n), p in bf["distribution"].items():
                assert abs(float(P[m, n]) - p) < 1e-10


def test_quorum_matches_principal_minor_reference():
    """Independent check vs the existing low-rank principal-minor quorum on B_tilde."""
    a, Z, pc, pw = _inputs(seed=3)
    eta, k = 1.3, 3
    P = cdq2q.cdq2_quorum_distribution(a, Z, eta, pc, pw, k)
    P_ref = determinantal_quorum_distribution(_btilde(a, Z, eta), pc, pw, k)
    assert torch.allclose(P, P_ref, atol=1e-10)


def test_eta_zero_recovers_esp_quorum():
    """eta=0 => L=diag(a) => the CDQ 2.0 quorum collapses EXACTLY to the ESP (diagonal-kernel)
    determinantal quorum."""
    a, Z, pc, pw = _inputs(seed=4)
    k = 3
    P = cdq2q.cdq2_quorum_distribution(a, Z, 0.0, pc, pw, k)
    B_diag = low_rank_kernel(a, diagonal_diversity(a.shape[0]))
    P_esp = determinantal_quorum_distribution(B_diag, pc, pw, k)
    assert torch.allclose(P, P_esp, atol=1e-10)


def test_quorum_decision_consistency():
    a, Z, pc, pw = _inputs(seed=5)
    eta, k, alpha = 1.0, 4, 3                                     # 2*alpha=6 > k=4
    dec = cdq2q.cdq2_quorum_decision(a, Z, eta, pc, pw, k, alpha)
    P = dec.distribution
    h_plus = float(P[alpha:, :].sum())
    h_minus = float(P[:, alpha:].sum())
    assert abs(float(dec.h_plus) - h_plus) < 1e-12
    assert abs(float(dec.h_minus) - h_minus) < 1e-12
    assert abs(float(dec.h_plus + dec.h_minus + dec.h_zero) - 1.0) < 1e-10


def test_quorum_batched_matches_per_source():
    g = torch.Generator().manual_seed(6)
    a = 0.3 + torch.rand(4, 6, generator=g, dtype=torch.float64)
    Z = torch.randn(4, 6, 3, generator=g, dtype=torch.float64)
    pc = 0.4 * torch.rand(4, 6, generator=g, dtype=torch.float64)
    pw = 0.3 * torch.rand(4, 6, generator=g, dtype=torch.float64)
    eta, k = 0.9, 3
    P = cdq2q.cdq2_quorum_distribution(a, Z, eta, pc, pw, k)
    assert P.shape == (4, k + 1, k + 1)
    for b in range(4):
        Pb = cdq2q.cdq2_quorum_distribution(a[b], Z[b], eta, pc[b], pw[b], k)
        assert torch.allclose(P[b], Pb, atol=1e-10)


def test_quorum_differentiable():
    a, Z, pc, pw = _inputs(seed=7)
    eta = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    a = a.clone().requires_grad_(True)
    Zv = Z.clone().requires_grad_(True)
    pcv = pc.clone().requires_grad_(True)
    k, alpha = 3, 2
    dec = cdq2q.cdq2_quorum_decision(a, Zv, eta, pcv, pw, k, alpha)
    dec.h_plus.backward()
    for t in (a, Zv, eta, pcv):
        assert t.grad is not None and torch.isfinite(t.grad).all()
    assert float(eta.grad.abs()) > 0          # diversity strength genuinely moves the quorum


def test_quorum_gradient_finite_when_p0_zero():
    """Regression (wiring audit): a real candidate with p^+ + p^- = 1 (=> p0 = 0, e.g. an ideal
    link ell=1) makes the deformed quality c = a*p0 hit exactly 0 at the (0,0) grid corner. The
    forward must stay finite AND the backward must NOT NaN (sqrt(0) has an infinite derivative)."""
    a = torch.tensor([1.0, 1.5, 0.8], dtype=torch.float64, requires_grad=True)
    Z = torch.randn(3, 3, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    eta, k = 1.0, 2
    pc = torch.tensor([0.6, 0.3, 0.5], dtype=torch.float64)
    pw = torch.tensor([0.4, 0.7, 0.5], dtype=torch.float64)      # pc+pw = 1 for every candidate
    P = cdq2q.cdq2_quorum_distribution(a, Z, eta, pc, pw, k)
    total = P.sum()
    assert math.isfinite(float(total.detach())) and abs(float(total.detach()) - 1.0) < 1e-9
    total.backward()                                             # must not raise / NaN
    assert bool(torch.isfinite(a.grad).all())


def test_diversity_changes_quorum():
    """The diversity mechanism must move P(m,n): with UNIFORM quality (selection is purely
    diversity-driven) and two orthogonal collinear clusters of OPPOSITE correctness, eta>0 spreads
    the poll across clusters (ESP cannot) and shifts the correct/wrong balance vs eta=0."""
    a = torch.ones(6, dtype=torch.float64)                       # quality uniform -> only diversity
    e1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    e2 = torch.tensor([0.0, 1.0], dtype=torch.float64)
    Z = torch.stack([e1, e1, e1, e2, e2, e2])                    # 2 orthogonal collinear clusters
    pc = torch.tensor([0.8, 0.8, 0.8, 0.05, 0.05, 0.05], dtype=torch.float64)   # A correct, B wrong
    pw = torch.tensor([0.05, 0.05, 0.05, 0.8, 0.8, 0.8], dtype=torch.float64)
    k, alpha = 3, 2
    P0 = cdq2q.cdq2_quorum_distribution(a, Z, 0.0, pc, pw, k)
    P1 = cdq2q.cdq2_quorum_distribution(a, Z, 3.0, pc, pw, k)
    assert float((P0 - P1).abs().max()) > 1e-3
    assert abs(float(P1.sum()) - 1.0) < 1e-10
    # spreading across the opposing clusters lowers the chance of a (one-sided) correct quorum
    h_plus_0 = float(P0[alpha:, :].sum())
    h_plus_1 = float(P1[alpha:, :].sum())
    assert abs(h_plus_0 - h_plus_1) > 1e-3
