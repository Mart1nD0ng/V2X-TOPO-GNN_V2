"""#3 consensus reliability as a CARRIED TEMPORAL STATE feature.

Consensus is fast vs vehicle motion (k=5, 20 rounds finish in ~ms; a 12 m/s vehicle moves ~cm), so the
per-snapshot consensus reliability — now faithful via the SSMC quenched closed form (#1) — is a slow
state variable that evolves frame-to-frame only as mobility slowly reshapes the topology/channel. This
module carries that state across frames as an OPT-IN extra node-feature channel the planner reads:

  * Layer 0 (``carry_reliability``): the previous frame's per-node consensus reliability P(correct).
  * Layer 2 (``carry_sensitivity``): the per-node confidence-sensitivity dF/dic — the operational form of
    the "is reliability decoupled from node trustworthiness here?" question (small => topology has taken
    control; large => ic is still a hard floor). The caller computes it (finite-diff / autograd).
  * Layer 1 (``carry_blend`` > 0): feed the carried reliability back as a prior on next-frame ic; the
    caller detaches each frame to bound BPTT. Off by default — the carried value rides the closed form,
    so feed it back only once #1 is trusted.

Everything is OPT-IN: with all flags off, ``temporal_state_dims`` is 0 and ``attach_temporal_state`` is a
no-op, so the model and features are byte-identical to the legacy static-per-snapshot path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class TemporalStateConfig:
    carry_reliability: bool = False  # Layer 0: carried per-node consensus reliability P(correct)
    carry_sensitivity: bool = False  # Layer 2: carried per-node dF/dic confidence sensitivity
    carry_blend: float = 0.0         # Layer 1: blend carried reliability into next-frame ic prior (0 = read-only)
    frame0_reliability: float = 0.5  # neutral carried value before any frame has been evaluated

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "TemporalStateConfig":
        data = data or {}
        cfg = cls(
            carry_reliability=bool(data.get("carry_reliability", cls.carry_reliability)),
            carry_sensitivity=bool(data.get("carry_sensitivity", cls.carry_sensitivity)),
            carry_blend=float(data.get("carry_blend", cls.carry_blend)),
            frame0_reliability=float(data.get("frame0_reliability", cls.frame0_reliability)),
        )
        if not 0.0 <= cfg.carry_blend <= 1.0:
            raise ValueError("carry_blend must be in [0, 1]")
        if not 0.0 <= cfg.frame0_reliability <= 1.0:
            raise ValueError("frame0_reliability must be in [0, 1]")
        return cfg


def temporal_state_dims(config: TemporalStateConfig) -> int:
    """Number of extra node-feature channels the carried state adds (0, 1, or 2)."""
    return int(bool(config.carry_reliability)) + int(bool(config.carry_sensitivity))


def node_reliability_state(evaluator_output: Mapping[str, Any]) -> torch.Tensor:
    """Per-node consensus reliability P(correct decision) in [0, 1], DETACHED — it is a carried state
    from a *previous* frame, a constant input to the current frame, not a within-frame gradient path."""
    correct = evaluator_output["node_p_correct_decision"]
    if not isinstance(correct, torch.Tensor):
        correct = torch.as_tensor(correct)
    return torch.clamp(correct.detach().reshape(-1), 0.0, 1.0)


def attach_temporal_state(
    node_features: torch.Tensor,
    config: TemporalStateConfig,
    *,
    reliability: torch.Tensor | None = None,
    sensitivity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append the carried-state channels to ``node_features`` ([N, F] -> [N, F + temporal_state_dims]).
    ``None`` (frame 0, before any evaluation) uses neutral defaults. No-op when all flags are off."""
    if temporal_state_dims(config) == 0:
        return node_features
    if node_features.ndim != 2:
        raise ValueError("node_features must have shape [N, F]")
    n = int(node_features.shape[0])
    columns = [node_features]
    if config.carry_reliability:
        if reliability is None:
            column = node_features.new_full((n, 1), float(config.frame0_reliability))
        else:
            column = reliability.detach().to(node_features).reshape(-1)
            if column.numel() != n:
                raise ValueError("reliability must have num_nodes values")
            column = torch.clamp(column, 0.0, 1.0).reshape(n, 1)
        columns.append(column)
    if config.carry_sensitivity:
        if sensitivity is None:
            column = node_features.new_zeros((n, 1))
        else:
            sens = sensitivity.detach().to(node_features).reshape(-1)
            if sens.numel() == n:
                column = sens.reshape(n, 1)
            elif sens.numel() == 1:
                column = node_features.new_full((n, 1), float(sens))
            else:
                raise ValueError("sensitivity must have 1 or num_nodes values")
        columns.append(column)
    return torch.cat(columns, dim=1)


def remap_carried_by_node_id(
    prev_values: torch.Tensor,
    prev_node_ids,
    new_node_ids,
    *,
    fill: float | torch.Tensor,
) -> torch.Tensor:
    """Re-index per-node carried state across a churn frame transition (Roadmap Phase 0.2).

    Under churn (boundary_mode='absorb_inject') the population changes between frames and POSITIONAL
    alignment breaks: deaths compact the arrays and births append. The stable key is ``node_id``
    (survivors keep theirs; births get fresh ids). This gathers ``prev_values`` (per-node rows or a
    [N, D] state matrix from the PREVIOUS frame, ordered by ``prev_node_ids``) into the NEW frame's
    node order: survivors copy their row, births get ``fill`` (a scalar — e.g. frame0_reliability —
    or a [D] template row, e.g. a GRU init-state row), deaths are dropped.

    No-churn streams have identical id arrays -> this is an exact identity gather (byte-identical),
    so harnesses can call it unconditionally.
    """
    if prev_values.ndim not in (1, 2):
        raise ValueError("prev_values must be [N] or [N, D]")
    prev_ids = [int(v) for v in (prev_node_ids.tolist() if hasattr(prev_node_ids, "tolist") else prev_node_ids)]
    new_ids = [int(v) for v in (new_node_ids.tolist() if hasattr(new_node_ids, "tolist") else new_node_ids)]
    if prev_values.shape[0] != len(prev_ids):
        raise ValueError("prev_values rows must match prev_node_ids length")
    index_of = {nid: row for row, nid in enumerate(prev_ids)}
    if prev_values.ndim == 1:
        out = prev_values.new_empty((len(new_ids),))
    else:
        out = prev_values.new_empty((len(new_ids), prev_values.shape[1]))
    if isinstance(fill, torch.Tensor):
        fill_row = fill.to(prev_values).reshape(-1)
        if prev_values.ndim == 2 and fill_row.numel() != prev_values.shape[1]:
            raise ValueError("tensor fill must match the state row width")
    else:
        fill_row = None
    for row, nid in enumerate(new_ids):
        src = index_of.get(nid)
        if src is not None:
            out[row] = prev_values[src]
        elif fill_row is not None:
            out[row] = fill_row if prev_values.ndim == 2 else fill_row[0]
        else:
            out[row] = float(fill)
    return out


def blended_initial_correct(
    base_ic: torch.Tensor, carried_reliability: torch.Tensor | None, config: TemporalStateConfig
) -> torch.Tensor:
    """Layer 1: blend the previous frame's reliability into this frame's initial-correct prior:
    ``ic' = (1-blend)*ic + blend*reliability``. Detached so back-prop does not cross frames (bounded
    BPTT). ``blend == 0`` (default) returns ``base_ic`` unchanged (read-only carried state)."""
    if config.carry_blend <= 0.0 or carried_reliability is None:
        return base_ic
    rel = torch.clamp(carried_reliability.detach().to(base_ic).reshape(-1), 0.0, 1.0)
    if rel.numel() != base_ic.numel():
        raise ValueError("carried_reliability must match base_ic shape")
    return (1.0 - config.carry_blend) * base_ic + config.carry_blend * rel
