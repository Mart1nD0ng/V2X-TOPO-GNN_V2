"""MC-faithful ESP training via the score-function (REINFORCE) gradient (esp_performance_scale_v2,
G-ESP-MC-FAITHFUL-TRAINING).

The EV1/EV2 finding: the analytic mean-field training surrogate is blind to the peer-selection effects the
dynamic-MC judge rewards (the GNN budget curve is flat; MC heuristic spread 0.075-0.085 vs analytic <=0.002).
To close that gap, train the GNN on a gradient that reflects what the MC judge measures. The ESP k-subset
sampler defines a differentiable law

    log pi(S_i) = sum_{j in S_i} s_ij - log e_k(exp(s_i))     (Eq. 16),

so for a trial outcome reward R (e.g. correct-basin first-hit) the policy gradient is

    grad E[R] = E[ (R - b) * sum_{i,t} grad log pi(S_{i,t}) ]   (REINFORCE, b a baseline),

which trains the GNN edge logits to maximise the MC basin objective WITHOUT a mean-field relaxation.

This module provides the differentiable building block ``batched_subset_log_prob`` (vectorised over all
(node, epoch, trial)) used to accumulate ``sum log pi`` along a judge rollout. The rollout itself is
collected by the canonical dynamic-MC path (extended with a gated log-pi accumulation), so the judge code
is reused verbatim -- no snowball duplication.
"""

from __future__ import annotations

import torch

from src.mainline.symmetric_polynomials import log_elementary_symmetric

__all__ = ["batched_subset_log_prob", "train_esp_reinforce"]

_NEG = -1e30


def batched_subset_log_prob(log_weights: torch.Tensor, chosen: torch.Tensor, k: int,
                            mask: torch.Tensor | None = None) -> torch.Tensor:
    """Vectorised ESP k-subset log-probability ``log pi(S)`` (Eq. 16), differentiable in ``log_weights``.

    Args:
        log_weights: ``[..., n]`` per-candidate ESP log-weights (the GNN's ``log quality``).
        chosen: ``[..., n]`` 0/1 indicator of the sampled k-subset (exactly ``k`` ones over valid slots).
        k: subset size.
        mask: ``[..., n]`` bool of valid candidates (masked slots contribute neither to the selected sum
            nor to the normaliser). Defaults to all-valid.

    Returns ``[...]`` = ``sum_{j in S} s_j - log e_k(exp(s))`` per source. Matches the unbatched
    ``symmetric_polynomials.subset_log_probability`` to numerical precision.
    """
    if mask is None:
        mask = torch.ones_like(log_weights, dtype=torch.bool)
    lw = torch.where(mask, log_weights, torch.full_like(log_weights, _NEG))
    ch = chosen.to(log_weights.dtype) * mask.to(log_weights.dtype)
    selected = (ch * lw).sum(dim=-1)                                  # sum_{j in S} s_j
    log_ek = log_elementary_symmetric(lw, k)[..., k]                  # log e_k over valid candidates
    return selected - log_ek


def train_esp_reinforce(model, instances, proto, phy, profile, *, steps: int, trials: int = 100,
                        lr: float = 5e-3, participation_fn=None, link_override=None, base_seed: int = 0,
                        snapshot_steps: tuple[int, ...] = ()):
    """Train an ESP/ESD-GNN by the MC-faithful score-function gradient (G-ESP-MC-FAITHFUL-TRAINING).

    Each step rolls out ``trials`` full-physics MC trials of the GNN policy (``reinforce=True``), takes the
    per-trial reward ``R`` = correct-basin first-hit and the differentiable per-trial ``sum log pi``, and
    descends ``-mean((R - b) * sum_log_pi)`` with a per-batch mean baseline ``b`` (variance reduction).
    Returns ``{model, history, snapshots}`` with the per-step training MC ``macro_P_correct`` (= ``R.mean()``)
    so the gap-closing (MC improves where analytic training was flat) is visible.

    ``snapshot_steps``: budgets at which to snapshot ``state_dict`` along the SAME persistent-Adam trajectory
    (cloned, never mutated -> the snapshot is trajectory-preserving; a step-150 run with snapshots at 40/80
    yields the identical step-150 weights as a plain step-150 run + free intermediate budgets for A2)."""
    import torch as _torch

    from src.metrics.participation import uniform_participation
    from src.models import ESDGNNQueryPolicy
    from src.validation import run_dynamic_mc

    if participation_fn is None:
        def participation_fn(sc):
            return uniform_participation(sc.num_nodes, dtype=_torch.float64, device=sc.positions.device)

    opt = _torch.optim.Adam(model.parameters(), lr=lr)
    history = {"loss": [], "mc_P_correct": [], "correct_mass": []}
    snapshots: dict[int, dict] = {}
    snap_set = set(int(s) for s in snapshot_steps)
    n = len(instances)
    for step in range(steps):
        scene, ev = instances[step % n]
        omega = participation_fn(scene).to(_torch.float64)
        opt.zero_grad()
        res = run_dynamic_mc(scene, ev, ESDGNNQueryPolicy(model, scene), proto, phy, num_trials=trials,
                             generator=_torch.Generator().manual_seed(base_seed + step),
                             link_override=link_override, service_profile=profile, participation=omega,
                             reinforce=True)
        R = res.reinforce_correct                                     # [T, N] per-node correct in {0,1}
        logp = res.reinforce_logp                                    # [T, N] differentiable
        # PER-NODE advantage (baseline = each node's mean reward over trials) -> low-variance credit;
        # participation-weighted to align with the participation-weighted macrostate objective.
        baseline = R.mean(dim=0, keepdim=True)                       # [1, N]
        advantage = (R - baseline).detach()                          # [T, N]
        loss = -(omega.unsqueeze(0) * advantage * logp).sum(dim=1).mean()   # mean over trials
        loss.backward()
        opt.step()
        # training progress: participation-weighted correct MASS per trial (smooth) + the basin P_correct
        mass = (omega.unsqueeze(0) * R).sum(dim=1)                   # [T]
        history["loss"].append(float(loss.detach()))
        history["correct_mass"].append(float(mass.mean()))
        history["mc_P_correct"].append(float((mass >= profile.correct_basin_mass).to(_torch.float64).mean()))
        if (step + 1) in snap_set:
            snapshots[step + 1] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return {"model": model, "history": history, "snapshots": snapshots}
