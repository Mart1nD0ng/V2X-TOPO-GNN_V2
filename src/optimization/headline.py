"""G11 -- the reliability-constrained superiority headline harness (spec §10-§11, constraint #14).

The single place that compares query policies fairly:

* **one canonical evaluator** -- every policy is scored by ``run_dynamic_mc`` (the independent
  judge, constraint #6/#14); nothing here re-implements the dynamics.
* **paired common random numbers (CRN)** -- for a given scene seed, *every* policy is run with the
  SAME MC generator seed, so the region-bit / channel randomness is shared and the policy
  difference is isolated (variance reduction for the paired comparison).
* **paired bootstrap + Bonferroni** -- the headline statistic is the per-scene paired difference
  ``metric(policy) - metric(reference)`` bootstrapped over scene seeds; CIs are widened by the
  number of simultaneous comparisons (Bonferroni) so "significant" means the corrected CI excludes 0.

No SciPy (its import is noisy here); the bootstrap is self-contained and seeded (reproducible).
Lower ``F_wrong`` / ``F_disagree`` / ``latency`` is better, so a NEGATIVE paired difference means
the policy beats the reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.validation import run_dynamic_mc

__all__ = ["PolicyScores", "evaluate_policies_paired", "PairedComparison", "compare_to_reference"]


@dataclass
class PolicyScores:
    """Per-scene metric vectors for one policy (index = scene index, aligned across policies)."""

    name: str
    F_wrong: list[float] = field(default_factory=list)
    F_disagree: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)
    latency_cvar: list[float] = field(default_factory=list)
    energy: list[float] = field(default_factory=list)
    finished_fraction: list[float] = field(default_factory=list)

    def metric(self, key: str) -> list[float]:
        return getattr(self, key)


def evaluate_policies_paired(
    scene_specs,
    make_policies,
    protocol_cfg,
    physics_cfg,
    *,
    num_trials: int,
    link_override: float | None = None,
    allow_ideal_ablation: bool = False,
    crn_base_seed: int = 100_000,
    verbose: bool = False,
) -> dict[str, PolicyScores]:
    """Score every policy on every scene under paired CRN.

    ``scene_specs``: iterable of ``(scene, evidence, scene_seed)``. ``make_policies``: callable
    ``scene -> dict[name, policy]`` (rebuilt per scene so trained models attach to the scene's
    features). All policies on a given scene share generator seed ``crn_base_seed + scene_seed``.

    The headline default is FULL physics (``link_override=None``, constraint #7). Using an ideal
    ``link_override`` quarantines the run as an explicit ablation: it must be opted into with
    ``allow_ideal_ablation=True`` so a bare ideal link can never silently become a headline.
    """
    if link_override is not None and not allow_ideal_ablation:
        raise ValueError(
            "evaluate_policies_paired received an ideal link_override but allow_ideal_ablation is "
            "False: a bare ideal link cannot be a headline comparison (constraint #9). Pass "
            "allow_ideal_ablation=True to run it explicitly as an ablation, or use link_override=None "
            "for the full-physics headline.")
    scores: dict[str, PolicyScores] = {}
    for si, (scene, evidence, scene_seed) in enumerate(scene_specs):
        policies = make_policies(scene)
        for name, policy in policies.items():
            gen = torch.Generator().manual_seed(crn_base_seed + int(scene_seed))   # CRN
            r = run_dynamic_mc(scene, evidence, policy, protocol_cfg, physics_cfg,
                               num_trials=num_trials, generator=gen, link_override=link_override)
            s = scores.setdefault(name, PolicyScores(name))
            s.F_wrong.append(r.F_wrong)
            s.F_disagree.append(r.F_disagree)
            s.latency.append(r.mean_finalisation_time)
            s.latency_cvar.append(r.latency_cvar)
            s.energy.append(r.mean_energy)
            s.finished_fraction.append(r.finished_fraction)
        if verbose:
            print(f"  scene {si + 1}/{len(scene_specs)} (seed {scene_seed}) scored", flush=True)
    return scores


def _paired_bootstrap(diffs: list[float], *, n_boot: int, alpha: float, seed: int):
    """Mean paired difference + a ``1-alpha`` percentile-bootstrap CI (resampling scenes)."""
    d = torch.tensor(diffs, dtype=torch.float64)
    n = d.numel()
    mean = float(d.mean())
    if n < 2:
        return mean, float("-inf"), float("inf")
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=gen)
    boot = d[idx].mean(dim=1)
    lo = float(torch.quantile(boot, alpha / 2))
    hi = float(torch.quantile(boot, 1 - alpha / 2))
    return mean, lo, hi


@dataclass
class PairedComparison:
    name: str
    reference: str
    metric: str
    mean_diff: float           # metric(policy) - metric(reference); negative = policy better
    ci: tuple[float, float]    # Bonferroni-corrected bootstrap CI of the mean paired diff
    significant: bool          # corrected CI excludes 0
    better: bool               # significant AND mean_diff < 0 (lower-is-better metric)


def compare_to_reference(
    scores: dict[str, PolicyScores],
    reference: str,
    *,
    metric: str = "F_wrong",
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 7,
) -> list[PairedComparison]:
    """Paired comparison of every other policy against ``reference`` on ``metric``.

    Bonferroni: with ``m`` comparisons each CI is at level ``1 - alpha/m``. Lower metric is better,
    so ``better`` = the corrected CI lies entirely below 0.
    """
    ref = scores[reference].metric(metric)
    others = [name for name in scores if name != reference]
    m = max(1, len(others))
    alpha_corr = alpha / m
    out: list[PairedComparison] = []
    for j, name in enumerate(others):
        vals = scores[name].metric(metric)
        diffs = [a - b for a, b in zip(vals, ref)]
        mean, lo, hi = _paired_bootstrap(diffs, n_boot=n_boot, alpha=alpha_corr, seed=seed + j)
        sig = (lo > 0.0) or (hi < 0.0)
        out.append(PairedComparison(name, reference, metric, mean, (lo, hi), sig, sig and mean < 0))
    return out
