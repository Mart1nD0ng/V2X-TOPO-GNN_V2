"""G5 -- CDQ exact heterogeneous quorum law (spec §9.5).

Acceptance: P_i(m,n) and (h+,h-,h0) match brute-force subset x ternary enumeration (<1e-10);
the diagonal kernel recovers the §5 quorum_dp exactly; gradient relative error <1e-4;
differentiable; reachable-support and normalisation hold.
"""

import pytest
import torch

from src.mainline.quorum_dp import quorum_decision_probabilities, quorum_response_distribution
from src.sampling.determinantal_quorum import (
    bruteforce_determinantal_quorum,
    determinantal_quorum_decision,
    determinantal_quorum_distribution,
)
from src.sampling.dpp_query import diagonal_diversity, low_rank_kernel


def _rand_source(d=5, r=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    quality = torch.rand(d, generator=g, dtype=torch.float64) * 2 + 0.1
    diversity = torch.randn(d, r, generator=g, dtype=torch.float64)
    B = low_rank_kernel(quality, diversity)
    pp = torch.rand(d, generator=g, dtype=torch.float64) * 0.5
    pm = torch.rand(d, generator=g, dtype=torch.float64) * 0.4
    # ensure pp+pm <= 1
    s = (pp + pm).clamp_min(1e-9)
    scale = torch.clamp(0.9 / s, max=1.0)
    return B, pp * scale, pm * scale


# ----------------------------------------------------------- exactness vs brute
def test_distribution_matches_bruteforce():
    for k in (2, 3):
        B, pp, pm = _rand_source(d=6, r=4, seed=k)
        P = determinantal_quorum_distribution(B, pp, pm, k)
        assert abs(float(P.sum()) - 1.0) < 1e-10           # normalised
        ref = bruteforce_determinantal_quorum(B, pp, pm, k, alpha=k)["distribution"]
        for (m, n), p in ref.items():
            assert abs(float(P[m, n]) - p) < 1e-10, f"P({m},{n}): {float(P[m,n])} vs {p}"


def test_decision_matches_bruteforce():
    k, alpha = 3, 2
    B, pp, pm = _rand_source(d=6, r=5, seed=4)
    dec = determinantal_quorum_decision(B, pp, pm, k, alpha)
    ref = bruteforce_determinantal_quorum(B, pp, pm, k, alpha)
    assert abs(float(dec.h_plus) - ref["h_plus"]) < 1e-10
    assert abs(float(dec.h_minus) - ref["h_minus"]) < 1e-10
    assert abs(float(dec.h_plus + dec.h_minus + dec.h_zero) - 1.0) < 1e-12


def test_unreachable_support_is_zero():
    k = 3
    B, pp, pm = _rand_source(d=5, r=4, seed=2)
    P = determinantal_quorum_distribution(B, pp, pm, k)
    for m in range(k + 1):
        for n in range(k + 1):
            if m + n > k:
                assert abs(float(P[m, n])) < 1e-10        # can't have >k responses


# ------------------------------------------------- diagonal recovers quorum_dp
def test_diagonal_kernel_recovers_quorum_dp():
    """Diagonal kernel (L = diag(q)) must reproduce the §5 elementary-symmetric quorum_dp."""
    d, k, alpha = 6, 3, 2
    g = torch.Generator().manual_seed(13)
    quality = torch.rand(d, generator=g, dtype=torch.float64) * 2 + 0.2
    B = low_rank_kernel(quality, diagonal_diversity(d))     # L = diag(quality)
    pp = torch.rand(d, generator=g, dtype=torch.float64) * 0.4
    pm = torch.rand(d, generator=g, dtype=torch.float64) * 0.3

    P_cdq = determinantal_quorum_distribution(B, pp, pm, k)
    # quorum_dp uses log-weights a_j = quality (the diagonal kernel entries)
    log_w = torch.log(quality).unsqueeze(0)                 # [1, d]
    P_dp = quorum_response_distribution(log_w, pp.unsqueeze(0), pm.unsqueeze(0), k)[0]  # [k+1,k+1]
    assert torch.allclose(P_cdq, P_dp, atol=1e-10)

    dec_cdq = determinantal_quorum_decision(B, pp, pm, k, alpha)
    dec_dp = quorum_decision_probabilities(log_w, pp.unsqueeze(0), pm.unsqueeze(0), k, alpha)
    assert abs(float(dec_cdq.h_plus) - float(dec_dp.h_plus[0])) < 1e-10
    assert abs(float(dec_cdq.h_minus) - float(dec_dp.h_minus[0])) < 1e-10


# ----------------------------------------------------------------- gradients
def test_gradient_relative_error_below_1e_4():
    """Autograd gradient of h_plus w.r.t. p_correct matches finite differences (<1e-4 rel)."""
    k, alpha = 3, 2
    B, pp, pm = _rand_source(d=5, r=4, seed=6)
    pp = pp.clone().requires_grad_(True)
    dec = determinantal_quorum_decision(B, pp, pm, k, alpha)
    dec.h_plus.backward()
    ana = pp.grad.clone()
    # central finite differences
    eps = 1e-6
    fd = torch.zeros_like(pp)
    pp0 = pp.detach()
    for j in range(pp0.numel()):
        d_ = torch.zeros_like(pp0); d_[j] = eps
        hp_plus = float(determinantal_quorum_decision(B, pp0 + d_, pm, k, alpha).h_plus)
        hp_minus = float(determinantal_quorum_decision(B, pp0 - d_, pm, k, alpha).h_plus)
        fd[j] = (hp_plus - hp_minus) / (2 * eps)
    rel = float((ana - fd).norm() / fd.norm().clamp_min(1e-12))
    assert rel < 1e-4, f"gradient relative error {rel:.2e}"


def test_batched_and_differentiable():
    d, r, k, alpha = 5, 4, 2, 2
    g = torch.Generator().manual_seed(8)
    quality = (torch.rand(3, d, generator=g, dtype=torch.float64) * 2 + 0.1).requires_grad_(True)
    diversity = torch.randn(3, d, r, generator=g, dtype=torch.float64)
    pp = torch.rand(3, d, generator=g, dtype=torch.float64) * 0.4
    pm = torch.rand(3, d, generator=g, dtype=torch.float64) * 0.3
    B = low_rank_kernel(quality, diversity)
    dec = determinantal_quorum_decision(B, pp, pm, k, alpha)
    assert dec.h_plus.shape == (3,)
    assert torch.allclose(dec.h_plus + dec.h_minus + dec.h_zero, torch.ones(3, dtype=torch.float64), atol=1e-12)
    dec.h_plus.sum().backward()
    assert quality.grad is not None and bool(torch.isfinite(quality.grad).all())


def test_strict_majority_enforced():
    B, pp, pm = _rand_source(d=5, r=4, seed=0)
    with pytest.raises(ValueError):
        determinantal_quorum_decision(B, pp, pm, 4, 2)     # 2*2 = 4 not > 4
