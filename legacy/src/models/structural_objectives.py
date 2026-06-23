"""Differentiable structural auxiliary objectives for diagnostics.

These utilities are not the coupled C/D/E objective. They exist to make future
curriculum experiments explicit when structural heads are not trained by the
main evaluator path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _finite_float_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating point tensor")
    if bool(torch.any(~torch.isfinite(value.detach())).cpu()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    entropy = -probabilities * torch.log(torch.clamp(probabilities, min=torch.finfo(probabilities.dtype).tiny))
    return entropy.sum(dim=-1)


def budget_entropy_loss(node_budget_logits: torch.Tensor, *, mode: str = "encourage_entropy") -> torch.Tensor:
    """Auxiliary budget entropy objective.

    Minimizing `encourage_entropy` increases entropy; minimizing
    `discourage_entropy` decreases entropy.
    """

    logits = _finite_float_tensor("node_budget_logits", node_budget_logits)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("node_budget_logits must have shape [N, bins]")
    entropy = _entropy_from_logits(logits).mean()
    if mode == "encourage_entropy":
        return -entropy
    if mode == "discourage_entropy":
        return entropy
    raise ValueError("mode must be 'encourage_entropy' or 'discourage_entropy'")


def budget_target_loss(node_budget_expected: torch.Tensor, target_budget: torch.Tensor | float) -> torch.Tensor:
    """Auxiliary expected-budget target objective."""

    expected = _finite_float_tensor("node_budget_expected", node_budget_expected).reshape(-1)
    target = target_budget
    if not torch.is_tensor(target):
        target = expected.new_tensor(float(target))
    target_tensor = _finite_float_tensor("target_budget", target).to(dtype=expected.dtype, device=expected.device)
    if target_tensor.ndim == 0:
        target_tensor = target_tensor.expand_as(expected)
    else:
        target_tensor = target_tensor.reshape(-1)
        if target_tensor.shape != expected.shape:
            raise ValueError("target_budget must be scalar or match node_budget_expected shape")
    return F.mse_loss(expected, target_tensor)


def sector_entropy_loss(sector_preference_logits: torch.Tensor, *, mode: str = "encourage_entropy") -> torch.Tensor:
    """Auxiliary sector preference entropy objective."""

    logits = _finite_float_tensor("sector_preference_logits", sector_preference_logits)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("sector_preference_logits must have shape [N, sectors]")
    entropy = _entropy_from_logits(logits).mean()
    if mode == "encourage_entropy":
        return -entropy
    if mode == "discourage_entropy":
        return entropy
    raise ValueError("mode must be 'encourage_entropy' or 'discourage_entropy'")


def role_balance_loss(node_role_logits: torch.Tensor, target_role_distribution: torch.Tensor) -> torch.Tensor:
    """Auxiliary role-distribution matching objective."""

    logits = _finite_float_tensor("node_role_logits", node_role_logits)
    if logits.ndim != 2 or logits.shape[1] == 0:
        raise ValueError("node_role_logits must have shape [N, roles]")
    target = _finite_float_tensor("target_role_distribution", target_role_distribution).to(
        dtype=logits.dtype,
        device=logits.device,
    )
    target = target.reshape(-1)
    if target.numel() != logits.shape[1]:
        raise ValueError("target_role_distribution must contain one value per role")
    target = target / torch.clamp(target.sum(), min=torch.finfo(logits.dtype).tiny)
    observed = torch.softmax(logits, dim=1).mean(dim=0)
    return F.mse_loss(observed, target)


def bridge_logit_regularizer(region_bridge_logits: torch.Tensor, edge_is_cross_region: torch.Tensor) -> torch.Tensor:
    """Auxiliary sparse bridge-logit regularizer for cross-region candidates."""

    logits = _finite_float_tensor("region_bridge_logits", region_bridge_logits).reshape(-1)
    if not torch.is_tensor(edge_is_cross_region):
        raise TypeError("edge_is_cross_region must be a torch.Tensor")
    if edge_is_cross_region.dtype != torch.bool:
        raise TypeError("edge_is_cross_region must use bool dtype")
    mask = edge_is_cross_region.to(device=logits.device).reshape(-1)
    if mask.shape != logits.shape:
        raise ValueError("edge_is_cross_region must have one value per bridge logit")
    if logits.numel() == 0:
        return logits.new_tensor(0.0)
    parts: list[torch.Tensor] = []
    if bool(torch.any(mask)):
        parts.append(F.softplus(-logits[mask]).mean())
    if bool(torch.any(~mask)):
        parts.append(F.softplus(logits[~mask]).mean())
    if not parts:
        return logits.new_tensor(0.0)
    return torch.stack(parts).mean()
