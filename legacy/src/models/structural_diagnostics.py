"""Diagnostics for sparse structural GNN head outputs."""

from __future__ import annotations

from typing import Any, Mapping

import torch


def _reference(output: Mapping[str, Any]) -> torch.Tensor:
    for value in output.values():
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
            return value
    return torch.tensor(0.0, dtype=torch.float64)


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def _quantile(values: torch.Tensor, q: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.quantile(values.reshape(-1), q) if values.numel() else _zero(reference)


def _entropy_from_logits(logits: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or logits.ndim != 2 or logits.shape[1] == 0:
        return _zero(reference)
    probabilities = torch.softmax(logits, dim=1)
    entropy = -probabilities * torch.log(torch.clamp(probabilities, min=torch.finfo(probabilities.dtype).tiny))
    return entropy.sum(dim=1).mean()


def _argmax_counts(logits: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0 or logits.ndim != 2 or logits.shape[1] == 0:
        return reference.new_empty((0,))
    argmax = torch.argmax(logits.detach(), dim=1)
    return torch.bincount(argmax, minlength=int(logits.shape[1])).to(dtype=reference.dtype, device=reference.device)


def _tensor(output: Mapping[str, Any], key: str, reference: torch.Tensor) -> torch.Tensor:
    value = output.get(key)
    if isinstance(value, torch.Tensor):
        return value.to(dtype=reference.dtype, device=reference.device)
    return reference.new_empty((0,))


def compute_structural_diagnostics(
    scorer_output: Mapping[str, Any],
    *,
    edge_is_cross_region: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize structural head outputs without creating dense graph tensors."""

    reference = _reference(scorer_output)
    budget_expected = _tensor(scorer_output, "node_budget_expected", reference).reshape(-1)
    budget_logits = _tensor(scorer_output, "node_budget_logits", reference)
    role_logits = _tensor(scorer_output, "node_role_logits", reference)
    sector_logits = _tensor(scorer_output, "sector_preference_logits", reference)
    bridge_logits = _tensor(scorer_output, "region_bridge_logits", reference).reshape(-1)
    structural_bias = _tensor(scorer_output, "structural_bias", reference).reshape(-1)
    sector_bias = _tensor(scorer_output, "sector_bias", reference).reshape(-1)
    role_bias = _tensor(scorer_output, "role_bias", reference).reshape(-1)
    bridge_bias = _tensor(scorer_output, "bridge_bias", reference).reshape(-1)

    if edge_is_cross_region is not None:
        if not torch.is_tensor(edge_is_cross_region):
            raise TypeError("edge_is_cross_region must be a torch.Tensor")
        if edge_is_cross_region.dtype != torch.bool:
            raise TypeError("edge_is_cross_region must use bool dtype")
        cross_mask = edge_is_cross_region.to(device=reference.device).reshape(-1)
        if cross_mask.numel() != bridge_logits.numel():
            raise ValueError("edge_is_cross_region must have one value per region bridge logit")
    else:
        cross_mask = torch.zeros(bridge_logits.shape, dtype=torch.bool, device=reference.device)
    same_mask = ~cross_mask if cross_mask.numel() == bridge_logits.numel() else torch.zeros_like(cross_mask)

    return {
        "node_budget_expected_mean": budget_expected.mean() if budget_expected.numel() else _zero(reference),
        "node_budget_expected_p10": _quantile(budget_expected, 0.10, reference),
        "node_budget_expected_p90": _quantile(budget_expected, 0.90, reference),
        "node_budget_entropy_mean": _entropy_from_logits(budget_logits, reference),
        "role_entropy_mean": _entropy_from_logits(role_logits, reference),
        "role_argmax_counts": _argmax_counts(role_logits, reference),
        "sector_entropy_mean": _entropy_from_logits(sector_logits, reference),
        "sector_argmax_counts": _argmax_counts(sector_logits, reference),
        "region_bridge_logit_mean": bridge_logits.mean() if bridge_logits.numel() else _zero(reference),
        "region_bridge_logit_p10": _quantile(bridge_logits, 0.10, reference),
        "region_bridge_logit_p90": _quantile(bridge_logits, 0.90, reference),
        "cross_region_bridge_logit_mean": bridge_logits[cross_mask].mean()
        if bridge_logits.numel() and bool(torch.any(cross_mask))
        else _zero(reference),
        "same_region_bridge_logit_mean": bridge_logits[same_mask].mean()
        if bridge_logits.numel() and bool(torch.any(same_mask))
        else _zero(reference),
        "structural_bias_abs_mean": structural_bias.abs().mean() if structural_bias.numel() else _zero(reference),
        "structural_bias_abs_max": structural_bias.abs().max() if structural_bias.numel() else _zero(reference),
        "sector_bias_abs_mean": sector_bias.abs().mean() if sector_bias.numel() else _zero(reference),
        "sector_bias_abs_max": sector_bias.abs().max() if sector_bias.numel() else _zero(reference),
        "role_bias_abs_mean": role_bias.abs().mean() if role_bias.numel() else _zero(reference),
        "role_bias_abs_max": role_bias.abs().max() if role_bias.numel() else _zero(reference),
        "bridge_bias_abs_mean": bridge_bias.abs().mean() if bridge_bias.numel() else _zero(reference),
        "bridge_bias_abs_max": bridge_bias.abs().max() if bridge_bias.numel() else _zero(reference),
    }


def compute_cap_diagnostics(
    caps: torch.Tensor,
    *,
    min_cap: int,
    max_cap: int,
    previous_caps: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Summarize deterministic integer constructor caps."""

    if not torch.is_tensor(caps):
        raise TypeError("caps must be a torch.Tensor")
    if torch.is_floating_point(caps) or torch.is_complex(caps) or caps.dtype == torch.bool:
        raise TypeError("caps must use an integer dtype")
    if min_cap < 0:
        raise ValueError("min_cap must be nonnegative")
    if max_cap < min_cap:
        raise ValueError("max_cap must be greater than or equal to min_cap")
    cap_values = caps.reshape(-1).to(dtype=torch.long)
    reference = cap_values.to(dtype=torch.float64)
    if cap_values.numel() == 0:
        histogram = reference.new_zeros((max_cap - min_cap + 1,))
        return {
            "cap_min": reference.new_tensor(0.0),
            "cap_mean": reference.new_tensor(0.0),
            "cap_max": reference.new_tensor(0.0),
            "cap_histogram": histogram,
            "cap_change_count": reference.new_tensor(0.0),
            "cap_saturation_low_count": reference.new_tensor(0.0),
            "cap_saturation_high_count": reference.new_tensor(0.0),
        }
    if bool(torch.any(cap_values < min_cap).cpu()) or bool(torch.any(cap_values > max_cap).cpu()):
        raise ValueError("caps must already be clipped to [min_cap, max_cap]")
    shifted = cap_values - int(min_cap)
    histogram = torch.bincount(shifted, minlength=max_cap - min_cap + 1).to(dtype=reference.dtype, device=caps.device)
    if previous_caps is not None:
        if not torch.is_tensor(previous_caps):
            raise TypeError("previous_caps must be a torch.Tensor")
        if torch.is_floating_point(previous_caps) or torch.is_complex(previous_caps) or previous_caps.dtype == torch.bool:
            raise TypeError("previous_caps must use an integer dtype")
        previous = previous_caps.to(device=caps.device, dtype=torch.long).reshape(-1)
        if previous.shape != cap_values.shape:
            raise ValueError("previous_caps must have the same shape as caps")
        change_count = (previous != cap_values).to(dtype=reference.dtype).sum()
    else:
        change_count = reference.new_tensor(0.0)
    return {
        "cap_min": cap_values.min().to(dtype=reference.dtype),
        "cap_mean": cap_values.to(dtype=reference.dtype).mean(),
        "cap_max": cap_values.max().to(dtype=reference.dtype),
        "cap_histogram": histogram,
        "cap_change_count": change_count,
        "cap_saturation_low_count": (cap_values == min_cap).to(dtype=reference.dtype).sum(),
        "cap_saturation_high_count": (cap_values == max_cap).to(dtype=reference.dtype).sum(),
    }
