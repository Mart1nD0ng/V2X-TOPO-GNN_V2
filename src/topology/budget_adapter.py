"""Deterministic adapter from soft budget expectations to integer caps."""

from __future__ import annotations

import torch


def expected_budget_to_cap(
    node_budget_expected: torch.Tensor,
    *,
    min_cap: int,
    max_cap: int,
    mode: str = "round",
    detach: bool = True,
) -> torch.Tensor:
    """Map soft expected budget values to integer per-node caps.

    Integer caps are not differentiable. The default detaches the soft input
    before rounding so callers do not mistake this adapter for a gradient path.
    """

    if not torch.is_tensor(node_budget_expected):
        raise TypeError("node_budget_expected must be a torch.Tensor")
    if not torch.is_floating_point(node_budget_expected):
        raise TypeError("node_budget_expected must be a floating point tensor")
    if not detach:
        raise NotImplementedError("integer budget caps are nondifferentiable; detach=False is not supported")
    if min_cap < 0:
        raise ValueError("min_cap must be nonnegative")
    if max_cap < min_cap:
        raise ValueError("max_cap must be greater than or equal to min_cap")
    if mode not in {"round", "floor", "ceil"}:
        raise ValueError("mode must be one of: round, floor, ceil")
    values = node_budget_expected.detach().reshape(-1)
    if bool(torch.any(~torch.isfinite(values)).cpu()):
        raise ValueError("node_budget_expected must contain only finite values")
    if mode == "round":
        caps = torch.round(values)
    elif mode == "floor":
        caps = torch.floor(values)
    else:
        caps = torch.ceil(values)
    caps = torch.clamp(caps, min=float(min_cap), max=float(max_cap))
    return caps.to(dtype=torch.long, device=node_budget_expected.device)
