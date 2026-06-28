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


def _tiny_scene():
    from src.evaluation.esp_scale import build_scale_instance
    return build_scale_instance((5, 5, 3), 0, scenario="matched_marginal_high", base_node_err=0.35,
                                corr_strength=0.25)


def test_reinforce_mode_does_not_change_the_judge():
    """The gated reinforce path must be numerically IDENTICAL to the default judge (same basins) under
    the same CRN -- it only ADDS the log-pi accumulation."""
    import torch
    from src.config.service_profile import ConsensusServiceProfile
    from src.environment import ProtocolConfig, RoundPhysicsConfig
    from src.metrics.participation import uniform_participation
    from src.sampling.baseline_policies import DistanceQueryPolicy
    from src.validation import run_dynamic_mc
    scene, ev = _tiny_scene()
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    om = uniform_participation(scene.num_nodes)
    kw = dict(num_trials=30, link_override=None, service_profile=prof, participation=om)
    base = run_dynamic_mc(scene, ev, DistanceQueryPolicy(beta_per_m=0.04), proto, phy,
                          generator=torch.Generator().manual_seed(0), **kw)
    rein = run_dynamic_mc(scene, ev, DistanceQueryPolicy(beta_per_m=0.04), proto, phy,
                          generator=torch.Generator().manual_seed(0), reinforce=True, **kw)
    assert rein.basin_P_correct == base.basin_P_correct          # judge outcome unchanged
    assert rein.basin_F_wrong == base.basin_F_wrong
    assert base.reinforce_logp is None and rein.reinforce_logp is not None


def test_reinforce_returns_differentiable_logp_and_a_step_trains():
    import torch
    from src.config.service_profile import ConsensusServiceProfile
    from src.environment import ProtocolConfig, RoundPhysicsConfig
    from src.evaluation.esp_scale import _esp_config
    from src.models import ESDGNN
    from src.optimization.mc_reinforce import train_esp_reinforce
    scene, ev = _tiny_scene()
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
    prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
    torch.manual_seed(0)
    model = ESDGNN(_esp_config(8, prof.k)).double()
    p0 = next(model.parameters()).detach().clone()
    out = train_esp_reinforce(model, [(scene, ev)], proto, phy, prof, steps=2, trials=30, lr=5e-3)
    assert len(out["history"]["mc_P_correct"]) == 2
    assert all(0.0 <= v <= 1.0 for v in out["history"]["mc_P_correct"])
    assert not torch.equal(next(model.parameters()).detach(), p0)   # REINFORCE moved the model


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
