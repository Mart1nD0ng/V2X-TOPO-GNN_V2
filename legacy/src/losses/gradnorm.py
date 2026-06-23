from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch


def compute_task_gradient_norms(
    losses: Mapping[str, torch.Tensor],
    parameters: Iterable[torch.Tensor],
    eps: float = 1e-12,
) -> dict[str, torch.Tensor]:
    params = [parameter for parameter in parameters if parameter.requires_grad]
    result: dict[str, torch.Tensor] = {}
    for name, loss_value in losses.items():
        grads = torch.autograd.grad(loss_value, params, retain_graph=True, allow_unused=True)
        pieces = [grad.reshape(-1) for grad in grads if grad is not None]
        if pieces:
            vector = torch.cat(pieces)
            result[name] = torch.sqrt(torch.sum(vector * vector) + vector.new_tensor(eps))
        else:
            result[name] = loss_value.new_tensor(0.0)
    return result


def compute_task_gradient_vectors(
    losses: Mapping[str, torch.Tensor],
    parameters: Iterable[torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-task gradient as a single flat vector w.r.t. ``parameters``.

    Unlike :func:`compute_task_gradient_norms` (which returns only the L2 norm), this
    returns the concatenated gradient vector, so callers can compute inter-task cosine
    similarity (the Phase-0 conflict diagnostic) or run PCGrad. ``allow_unused``
    gradients are **zero-filled per parameter** so every task vector shares an identical
    layout — required for cosine similarity and gradient projection to be meaningful.
    The L2 norm of the returned vector matches ``compute_task_gradient_norms`` up to its
    internal ``eps``.
    """
    params = [parameter for parameter in parameters if parameter.requires_grad]
    result: dict[str, torch.Tensor] = {}
    for name, loss_value in losses.items():
        grads = torch.autograd.grad(loss_value, params, retain_graph=True, allow_unused=True)
        pieces = [
            (grad if grad is not None else torch.zeros_like(param)).reshape(-1)
            for grad, param in zip(grads, params)
        ]
        result[name] = torch.cat(pieces) if pieces else loss_value.new_zeros(1)
    return result


@dataclass
class GradNormBalancer:
    tasks: Sequence[str] = ("R", "D", "E")
    alpha: float = 0.5
    lr: float = 0.025
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.alpha < 0.0:
            raise ValueError("alpha must be nonnegative")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        self.weights = torch.ones(len(self.tasks), dtype=torch.float64)
        self.initial_losses: torch.Tensor | None = None

    def update(
        self,
        losses: Mapping[str, torch.Tensor | float],
        gradient_norms: Mapping[str, torch.Tensor | float],
    ) -> dict[str, torch.Tensor]:
        loss_tensor = torch.stack([torch.as_tensor(losses[name], dtype=torch.float64) for name in self.tasks])
        norm_tensor = torch.stack([torch.as_tensor(gradient_norms[name], dtype=torch.float64) for name in self.tasks])
        if self.initial_losses is None:
            self.initial_losses = torch.clamp(loss_tensor.detach(), min=self.eps)
        rates = torch.clamp(loss_tensor.detach(), min=self.eps) / self.initial_losses
        inverse_rate = rates / torch.clamp(rates.mean(), min=self.eps)
        target_norm = torch.clamp(norm_tensor.detach().mean(), min=self.eps) * torch.pow(inverse_rate, self.alpha)
        relative_norm = norm_tensor.detach() / torch.clamp(target_norm, min=self.eps)
        updated = self.weights * torch.exp(-self.lr * (relative_norm - 1.0))
        updated = torch.clamp(updated, min=self.eps)
        self.weights = updated * (len(self.tasks) / torch.clamp(updated.sum(), min=self.eps))
        return {name: self.weights[idx].clone() for idx, name in enumerate(self.tasks)}
