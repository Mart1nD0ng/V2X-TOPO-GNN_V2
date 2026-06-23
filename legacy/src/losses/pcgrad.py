from __future__ import annotations

from collections.abc import Sequence

import torch


def pcgrad_project(gradients: Sequence[torch.Tensor], eps: float = 1e-12) -> list[torch.Tensor]:
    if not gradients:
        return []
    originals = [gradient.reshape(-1) for gradient in gradients]
    projected = [gradient.clone() for gradient in originals]
    for i, current in enumerate(projected):
        for j, other in enumerate(originals):
            if i == j:
                continue
            dot = torch.dot(current, other)
            if bool((dot < 0.0).detach().cpu()):
                denom = torch.dot(other, other) + current.new_tensor(eps)
                current = current - dot / denom * other
        projected[i] = current
    return [value.reshape_as(template) for value, template in zip(projected, gradients)]


def merge_pcgrad(gradients: Sequence[torch.Tensor], reduction: str = "mean") -> torch.Tensor:
    projected = pcgrad_project(gradients)
    if not projected:
        return torch.empty(0)
    stacked = torch.stack([gradient.reshape(-1) for gradient in projected])
    if reduction == "mean":
        return stacked.mean(dim=0)
    if reduction == "sum":
        return stacked.sum(dim=0)
    raise ValueError("reduction must be 'mean' or 'sum'")
