"""G-ESP-MC-FAITHFUL-TRAINING: the differentiable batched ESP subset log-probability (REINFORCE core)."""

import torch

from src.mainline.symmetric_polynomials import subset_log_probability
from src.optimization.mc_reinforce import batched_subset_log_prob


def test_matches_unbatched_reference():
    torch.manual_seed(0)
    n, k = 6, 3
    log_w = torch.randn(4, n, dtype=torch.float64)            # 4 sources
    chosen = torch.zeros(4, n, dtype=torch.float64)
    subsets = [[0, 1, 2], [1, 3, 5], [0, 2, 4], [2, 3, 4]]
    for b, s in enumerate(subsets):
        chosen[b, s] = 1.0
    got = batched_subset_log_prob(log_w, chosen, k)
    for b, s in enumerate(subsets):
        ref = subset_log_probability(log_w[b], s, k)
        assert torch.allclose(got[b], ref, atol=1e-9), (b, got[b].item(), ref.item())


def test_respects_mask():
    torch.manual_seed(1)
    n, k = 7, 2
    log_w = torch.randn(n, dtype=torch.float64)
    mask = torch.tensor([True, True, True, True, True, False, False])   # last 2 padded
    chosen = torch.zeros(n, dtype=torch.float64); chosen[[0, 3]] = 1.0
    got = batched_subset_log_prob(log_w, chosen, k, mask=mask)
    ref = subset_log_probability(log_w[:5], [0, 3], k)        # normaliser over valid candidates only
    assert torch.allclose(got, ref, atol=1e-9)


def test_differentiable_and_normalised():
    torch.manual_seed(2)
    n, k = 5, 2
    log_w = torch.randn(n, dtype=torch.float64, requires_grad=True)
    # probabilities over ALL k-subsets must sum to 1 (so exp(log_prob) is a valid law)
    import itertools
    total = 0.0
    for combo in itertools.combinations(range(n), k):
        ch = torch.zeros(n, dtype=torch.float64); ch[list(combo)] = 1.0
        total = total + batched_subset_log_prob(log_w, ch, k).exp()
    assert torch.allclose(total, torch.tensor(1.0, dtype=torch.float64), atol=1e-9)
    # gradient flows to the logits
    ch = torch.zeros(n, dtype=torch.float64); ch[[0, 1]] = 1.0
    batched_subset_log_prob(log_w, ch, k).backward()
    assert log_w.grad is not None and torch.isfinite(log_w.grad).all() and log_w.grad.abs().sum() > 0
