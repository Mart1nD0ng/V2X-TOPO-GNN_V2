"""Single-node Snowball/Avalanche finite-horizon state machine (spec §6).

Each node maintains a conviction state ``p_ir(t) in Delta^{2*beta+2}`` (Eq. 32).
``beta`` is the consecutive-success decision threshold.  The ``2*beta+2`` states are

    [0 .. beta-1]        leaning CORRECT with that many consecutive correct quorums
    [beta .. 2*beta-1]   leaning WRONG with that many consecutive wrong quorums
    2*beta               absorbed: decided CORRECT
    2*beta+1             absorbed: decided WRONG

Given the per-round quorum outcome probabilities ``(h^+, h^-, h^0)`` (correct
quorum / wrong quorum / no quorum, from the §5 quorum DP) the one-step transition
``T(h^+, h^-, h^0)`` is:

    leaning-correct(c): +correct quorum  -> c+1 (absorb CORRECT at c+1==beta)
                        +wrong   quorum  -> flip to wrong side (count 1; absorb if beta==1)
                        +no      quorum  -> reset to neutral-correct (state 0)
    leaning-wrong(c):   symmetric
    absorbing states stay put.

The node's *current preference* (what colour it answers when polled, feeding the
neighbour response probabilities ``p^+ = ell*u``, ``p^- = ell*v`` of Eqs. 22-24) is
read out as the total mass leaning/decided correct vs wrong; these sum to 1.

This is a clean mainline reimplementation (NOT a legacy import) of the Snowball
automaton; it carries no beta-tail / mean-field closure.  Everything is a
differentiable stochastic matrix applied round by round (time-inhomogeneous, since
the ``h`` change as neighbours evolve), with no Monte-Carlo sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

__all__ = [
    "SnowballLayout",
    "snowball_state_count",
    "snowball_layout",
    "build_transition",
    "initial_distribution",
    "readout_preference",
    "terminal_outcomes",
]


def snowball_state_count(beta: int) -> int:
    if not isinstance(beta, int) or beta < 1:
        raise ValueError("beta must be a positive int")
    return 2 * beta + 2


@dataclass(frozen=True)
class SnowballLayout:
    beta: int
    state_count: int
    correct_lean: tuple[int, ...]
    wrong_lean: tuple[int, ...]
    correct_abs: int
    wrong_abs: int
    # transition target table, materialised as (flat_index, kind) lists where
    # kind in {"+","-","0","1"} selects h_plus / h_minus / h_zero / one.
    transitions: tuple[tuple[int, str], ...]


@lru_cache(maxsize=None)
def snowball_layout(beta: int) -> SnowballLayout:
    if not isinstance(beta, int) or beta < 1:
        raise ValueError("beta must be a positive int")
    S = snowball_state_count(beta)
    correct_abs = 2 * beta
    wrong_abs = 2 * beta + 1
    correct_lean = tuple(range(0, beta))
    wrong_lean = tuple(range(beta, 2 * beta))

    trans: list[tuple[int, str]] = []
    for count in range(beta):
        plus_state = count
        minus_state = beta + count
        plus_success = correct_abs if (count + 1 >= beta) else (count + 1)
        minus_success = wrong_abs if (count + 1 >= beta) else (beta + count + 1)
        switch_to_minus = wrong_abs if beta == 1 else (beta + 1)
        switch_to_plus = correct_abs if beta == 1 else 1
        # leaning-correct row
        trans.append((plus_state * S + plus_success, "+"))
        trans.append((plus_state * S + switch_to_minus, "-"))
        trans.append((plus_state * S + 0, "0"))
        # leaning-wrong row
        trans.append((minus_state * S + minus_success, "-"))
        trans.append((minus_state * S + switch_to_plus, "+"))
        trans.append((minus_state * S + beta, "0"))
    # absorbing rows
    trans.append((correct_abs * S + correct_abs, "1"))
    trans.append((wrong_abs * S + wrong_abs, "1"))
    return SnowballLayout(
        beta=beta,
        state_count=S,
        correct_lean=correct_lean,
        wrong_lean=wrong_lean,
        correct_abs=correct_abs,
        wrong_abs=wrong_abs,
        transitions=tuple(trans),
    )


def build_transition(
    h_plus: torch.Tensor,
    h_minus: torch.Tensor,
    h_zero: torch.Tensor,
    beta: int,
) -> torch.Tensor:
    """Differentiable batched transition matrices ``[B, S, S]`` (Eq. 32).

    Each ``h`` is ``[B]`` with ``h_plus + h_minus + h_zero = 1``.  Row-stochastic by
    construction (each transient row places exactly ``h_plus + h_minus + h_zero``;
    absorbing rows place ``1``).
    """
    if not (h_plus.shape == h_minus.shape == h_zero.shape):
        raise ValueError("h_plus, h_minus, h_zero must share shape [B]")
    if h_plus.ndim != 1:
        raise ValueError("h tensors must be 1-D [B]")
    layout = snowball_layout(beta)
    S = layout.state_count
    B = h_plus.shape[0]
    ones = torch.ones_like(h_plus)
    value_map = {"+": h_plus, "-": h_minus, "0": h_zero, "1": ones}

    cols = torch.tensor([idx for idx, _ in layout.transitions], dtype=torch.long, device=h_plus.device)
    vals = torch.stack([value_map[kind] for _, kind in layout.transitions], dim=1)  # [B, num_trans]
    flat = h_plus.new_zeros((B, S * S))
    flat = flat.scatter_add(1, cols.unsqueeze(0).expand(B, -1), vals)
    return flat.reshape(B, S, S)


def initial_distribution(
    initial_correct_preference: torch.Tensor | float,
    beta: int,
    batch: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Initial state ``[B, S]``: mass ``init`` at neutral-correct (state 0), ``1-init``
    at neutral-wrong (state beta)."""
    layout = snowball_layout(beta)
    S = layout.state_count
    if isinstance(initial_correct_preference, torch.Tensor):
        init = initial_correct_preference.to(dtype=dtype, device=device).reshape(-1)
        if init.numel() == 1:
            init = init.expand(batch)
        if init.numel() != batch:
            raise ValueError("initial_correct_preference must be scalar or length B")
        device = init.device
    else:
        init = torch.full((batch,), float(initial_correct_preference), dtype=dtype, device=device)
    if bool(torch.any((init < -1e-9) | (init > 1 + 1e-9)).cpu()):
        raise ValueError("initial_correct_preference must be in [0, 1]")
    init = init.clamp(0.0, 1.0)
    p0 = init.new_zeros((batch, S))
    p0[:, 0] = init
    p0[:, beta] = 1.0 - init
    return p0


def readout_preference(p: torch.Tensor, beta: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Current correct/wrong preference ``(u, v)`` from state ``p[..., S]``; ``u+v=1``."""
    layout = snowball_layout(beta)
    pref_correct = p[..., list(layout.correct_lean)].sum(dim=-1) + p[..., layout.correct_abs]
    pref_wrong = p[..., list(layout.wrong_lean)].sum(dim=-1) + p[..., layout.wrong_abs]
    return pref_correct, pref_wrong


def terminal_outcomes(p: torch.Tensor, beta: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Terminal ``(c, w, undecided)`` = P(decided correct), P(decided wrong), P(transient)."""
    layout = snowball_layout(beta)
    c = p[..., layout.correct_abs]
    w = p[..., layout.wrong_abs]
    undecided = torch.clamp(1.0 - c - w, 0.0, 1.0)
    return c, w, undecided
