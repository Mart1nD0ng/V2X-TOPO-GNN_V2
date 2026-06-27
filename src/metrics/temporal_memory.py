"""Temporal memory for the macrostate / CDQ 2.0 round (Phase 12 / G-TEMPORAL).

A CAUSAL, DIFFERENTIABLE, OBSERVABLE-DRIVEN memory state that accumulates an observable per-epoch
signal across a sequence of consensus episodes, so the query policy can adapt to PERSISTENT
structure (effective-sampling drift, region correlation, load, weak cuts) instead of re-discovering
it every epoch. The headline stays the macrostate basin first-hitting (the legacy global-risk
emission does NOT enter here); these are auxiliary signals feeding the CDQ 2.0 query, not the
outcome.

Core primitive -- the causal exponential moving average over an observable signal ``x_t``:

    m_0 = x_0,   m_t = (1 - rho) m_{t-1} + rho x_t,   rho in (0, 1].

Properties (the temporal Mechanism Contract):

* **C2 / no-future-leak (CAUSAL).** ``m_t`` depends ONLY on ``x_{<=t}`` -- a future observation
  ``x_{t+1..}`` has EXACTLY zero gradient into ``m_t``. The driving signal ``x_t`` must itself be a
  deployment-observable proxy (poll outcomes / response cohesion / measured load), never ``Y*`` or
  future evidence -- enforced by the caller; the primitive only guarantees the temporal causality.
* **C3 / gradient-reachable.** ``m`` is a smooth (linear-recurrence) function of ``x`` and ``rho``,
  so a correlation/diversity loss on the memory reaches the CDQ 2.0 parameters; no ``float()`` /
  ``.item()`` / detach on the path.
* **C5 / mainline consistency.** ``rho = 1`` => ``m_t = x_t`` (memory OFF, current-epoch only) =>
  the policy reduces to the static mainline.
* **Scoping.** Under PERSISTENCE (``x_t`` stationary + observation noise) the EMA denoises toward
  the true signal; under iid-in-time (``x_t`` reshuffled each epoch) it converges to the marginal
  mean and carries no per-epoch information -- so a memory-driven policy can only help when the
  structure actually persists (the matched-marginal-in-time control isolates this).
"""

from __future__ import annotations

import torch

__all__ = [
    "causal_ema",
    "no_memory",
    "effective_memory_horizon",
    "estimate_quality",
]


def causal_ema(x: torch.Tensor, rho) -> torch.Tensor:
    """Causal exponential moving average over the leading (time) axis.

    Args:
        x: ``[T, ...]`` observable per-epoch signal (T epochs).
        rho: decay in ``(0, 1]`` (scalar or 0-d tensor). ``rho = 1`` is memory-off (``m = x``);
            small ``rho`` is long memory.

    Returns:
        ``m`` ``[T, ...]`` with ``m_t = (1-rho) m_{t-1} + rho x_t`` (``m_0 = x_0``). Differentiable
        in ``x`` and ``rho``; ``m_t`` is independent of ``x_{>t}`` (causal -- no future leak).
    """
    if x.ndim < 1:
        raise ValueError("x must have a leading time axis [T, ...]")
    r = torch.as_tensor(rho, dtype=x.dtype, device=x.device)
    if bool(((r <= 0) | (r > 1)).any().cpu()):
        raise ValueError("rho must be in (0, 1]")
    T = x.shape[0]
    out = [x[0]]
    for t in range(1, T):
        out.append((1.0 - r) * out[-1] + r * x[t])
    return torch.stack(out, dim=0)


def no_memory(x: torch.Tensor) -> torch.Tensor:
    """The memory-OFF baseline: use only the current epoch's observation (``rho = 1``)."""
    return x


def effective_memory_horizon(rho) -> float:
    """Effective memory horizon ``1/rho`` epochs (for reporting / picking rho)."""
    r = float(rho)
    if not (0.0 < r <= 1.0):
        raise ValueError("rho must be in (0, 1]")
    return 1.0 / r


def estimate_quality(m: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Quality of the memory estimate ``m`` against a (held-out, evaluation-only) ``target``:
    ``1 - ||m - target||^2 / (||target - mean(target)||^2 + eps)`` (a coefficient-of-determination /
    skill score in ``(-inf, 1]``; 1 = perfect, 0 = no better than the target mean).

    This is an EVALUATION diagnostic of how well the memory tracks the true (persistent) structure;
    it is NOT used on the query/training path (it needs the target, which is evaluation-side).
    """
    sse = ((m - target) ** 2).sum()
    sst = ((target - target.mean()) ** 2).sum() + eps
    return 1.0 - sse / sst
