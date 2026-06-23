"""Track G1 — gradient governance integration layer.

Design contract: docs/GRADIENT_GOVERNANCE_DESIGN.md (Track G1). The FIRST authorized
wiring of PCGrad / GradNorm into the training backward step (docs/LOSS_DESIGN.md kept
`pcgrad.py`/`gradnorm.py` as "utilities only, no optimizer integration"; AGENTS.md now
allows default-path changes that are opt-in, unit-tested, and training-validated).

`coupled_backward` consumes a `compute_coupled_loss` output and populates `.grad`:

- **governance all-off  ->  byte-identical** to ``effective_backward_loss.backward()``
  (the literal same call), so every existing run/test is unaffected.
- **pcgrad on**  ->  per-task gradients of the *weighted* L_R/L_D/L_E are projected
  pairwise to remove conflict (`pcgrad_project`), then summed and scattered into `.grad`.
  Static arm weights are preserved (so the de_ablation w_D x10 probe still means w_D x10).
- **gradnorm on**  ->  per-task gradients of the *unweighted* L_R/L_D/L_E are re-weighted
  by `GradNormBalancer` (training-rate matching). This OVERRIDES static arm weights, so a
  gradnorm/both run is a governed-production config, not a fixed-weight ablation.

This is gradient PLUMBING, not a new objective: it only recombines the existing C/D/E
gradients. No link/SINR/BLER/HARQ/coverage term is introduced; it lives in `src/training/`
and consumes only `compute_coupled_loss` outputs, so `verify_no_link_reliability_loss.py`
(which scans `src/losses/`) is unaffected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from src.losses import GradNormBalancer, compute_task_gradient_vectors, pcgrad_project

_TASKS = ("R", "D", "E")


@dataclass(frozen=True)
class GradientGovernanceConfig:
    pcgrad: bool = False
    gradnorm: bool = False
    gradnorm_alpha: float = 0.5
    gradnorm_lr: float = 0.025
    log_conflict: bool = False

    @property
    def active(self) -> bool:
        return self.pcgrad or self.gradnorm

    @classmethod
    def from_name(cls, name: str | None, **overrides) -> "GradientGovernanceConfig":
        key = (name or "none").lower()
        table = {
            "none": dict(),
            "pcgrad": dict(pcgrad=True),
            "gradnorm": dict(gradnorm=True),
            "both": dict(pcgrad=True, gradnorm=True),
        }
        if key not in table:
            raise ValueError(f"unknown governance mode {name!r}; expected one of {sorted(table)}")
        return cls(**{**table[key], **overrides})


def make_balancer(governance: GradientGovernanceConfig) -> GradNormBalancer | None:
    """A fresh (stateful) balancer for a gradnorm run, or None. Create one PER training
    run/arm — the balancer accumulates the initial-loss baseline across steps."""
    if not governance.gradnorm:
        return None
    return GradNormBalancer(tasks=_TASKS, alpha=governance.gradnorm_alpha, lr=governance.gradnorm_lr)


def _cosine(u: torch.Tensor, v: torch.Tensor, eps: float = 1e-12) -> float:
    norm_u = float(u.norm())
    norm_v = float(v.norm())
    if norm_u < eps or norm_v < eps:
        return 0.0
    return float(torch.dot(u, v) / (u.norm() * v.norm()))


def coupled_backward(
    loss_out: Mapping[str, Any],
    parameters: Iterable[torch.Tensor],
    governance: GradientGovernanceConfig,
    balancer: GradNormBalancer | None = None,
) -> dict[str, Any]:
    """Populate ``.grad`` on ``parameters`` from a ``compute_coupled_loss`` output.

    Returns a (possibly empty) diagnostics dict; when ``governance.log_conflict`` it
    carries per-task gradient norms, pairwise cosines, and the weights used.
    """
    params = [p for p in parameters if p.requires_grad]

    if not governance.active:
        # Unchanged production path — the literal same backward, byte-identical.
        loss_out["effective_backward_loss"].backward()
        return {}

    scale = loss_out["scale_backward_multiplier"]
    if governance.gradnorm:
        if balancer is None:
            raise ValueError("gradnorm requires a GradNormBalancer (use make_balancer)")
        # GradNorm balances the RAW per-task gradients, then sets dynamic weights.
        raw_scaled = {t: loss_out[f"L_{t}"] * scale for t in _TASKS}
        vectors = compute_task_gradient_vectors(raw_scaled, params)
        norms = {t: vectors[t].norm() for t in _TASKS}
        new_weights = balancer.update({t: loss_out[f"L_{t}"].detach() for t in _TASKS}, norms)
        task_grads = {t: float(new_weights[t]) * vectors[t] for t in _TASKS}
        weights_used = {t: float(new_weights[t]) for t in _TASKS}
    else:
        # PCGrad-only: keep the static (arm) weights; de-conflict the WEIGHTED gradients.
        weighted_scaled = {t: loss_out[f"weighted_L_{t}"] * scale for t in _TASKS}
        task_grads = compute_task_gradient_vectors(weighted_scaled, params)
        weights_used = {t: float(loss_out["weights"][t]) for t in _TASKS}

    if governance.pcgrad:
        projected = pcgrad_project([task_grads["R"], task_grads["D"], task_grads["E"]])
        merged = projected[0] + projected[1] + projected[2]
    else:
        merged = task_grads["R"] + task_grads["D"] + task_grads["E"]

    offset = 0
    for p in params:
        n = p.numel()
        p.grad = merged[offset:offset + n].view_as(p).clone()
        offset += n

    if not governance.log_conflict:
        return {}
    return {
        "task_norms": {t: float(task_grads[t].norm()) for t in _TASKS},
        "cos_DE": _cosine(task_grads["D"], task_grads["E"]),
        "cos_DR": _cosine(task_grads["D"], task_grads["R"]),
        "cos_RE": _cosine(task_grads["R"], task_grads["E"]),
        "weights": weights_used,
    }
