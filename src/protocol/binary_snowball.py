"""True binary Snowball finite-horizon confidence process (spec §3.2).

This is the protocol-semantics correction mandated by the technical spec §3.1 and
recorded in ``docs/PROTOCOL_SEMANTICS.md``: the preferred colour follows the
*accumulated confidence* ``d = d⁺ − d⁻`` and is **sticky** against a single opposite
quorum, unlike the legacy ``src/mainline/snowball.py`` Snowflake streak automaton
(which flips preference on any single opposite quorum and is retained only as a named
baseline).

Per-node state ``X = (d, pref, last, c, decision)`` over a finite horizon ``R_max``
(spec §3.2):

    d        in {-R_max .. R_max}   confidence difference d⁺ − d⁻
    pref     in {+, -}              argmax-confidence colour (ties keep current)
    last     in {+, -, ⊥}           colour of the most recent successful quorum
    c        in {0 .. β-1}          consecutive same-colour quorum streak length
    decision in {U, +, -}           U undecided; +/- absorbing finalized colour

One-round update given the ternary quorum outcome ``o ∈ {+,-,0}`` (probabilities
``h⁺,h⁻,h⁰`` from the §5 quorum DP):

    o = 0 :  last←⊥, c←0                          (streak broken; d, pref PERSIST)
    o = + :  d←d+1; pref←sign(d) (ties keep);     (confidence accumulation)
             if last=+ then c←c+1 else last←+,c←1;
             if c≥β then decide(pref)
    o = - :  symmetric (d←d-1)

The state set is enumerated by BFS over the reachable states (finite because ``d`` is
clamped to ``[-R_max,R_max]`` and over exactly ``R_max`` rounds the clamp is never
exercised, so it introduces no distortion -- see ``docs/PROTOCOL_SEMANTICS.md`` §2).

Crucially the per-round transition is applied as a **sparse scatter** (each undecided
state has exactly three out-transitions, each absorbing state one self-loop), so the
cost is ``O(B · S)`` per round and **no** ``[B, S, S]`` dense matrix is ever
materialised -- this keeps the true-Snowball state, whose ``S`` is ``O(R_max·β)`` rather
than the Snowflake ``2β+2``, tractable at the target ``N`` (spec stop condition #5).
Everything is differentiable w.r.t. ``(h⁺,h⁻,h⁰)`` and the state distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

__all__ = [
    "SnowballLayout",
    "snowball_layout",
    "reachable_states",
    "apply_round",
    "transition_matrix",
    "initial_distribution",
    "readout_preference",
    "terminal_outcomes",
    "simulate_trajectory",
]

# Outcome / kind codes.
PLUS, MINUS, ZERO = 1, -1, 0
KIND_HPLUS, KIND_HMINUS, KIND_HZERO, KIND_ONE = 0, 1, 2, 3

# State tuple conventions:
#   undecided: ("U", d, pref, last, c)   pref,last in {+1,-1}, last 0 = ⊥
#   decided:   ("D", sign)               sign in {+1,-1}
_MAX_STATES = 2_000_000  # defensive cap on R_max·β blow-up


def _step(state: tuple, outcome: int, beta: int, r_max: int) -> tuple:
    """Deterministic one-round state update for quorum ``outcome`` (spec §3.3)."""
    if state[0] == "D":
        return state  # absorbing
    _, d, pref, last, c = state
    if outcome == ZERO:
        return ("U", d, pref, 0, 0)  # streak broken; confidence persists
    nd = d + outcome
    if nd > r_max:
        nd = r_max
    elif nd < -r_max:
        nd = -r_max
    if nd > 0:
        npref = 1
    elif nd < 0:
        npref = -1
    else:
        npref = pref  # tie keeps current preference
    nc = c + 1 if last == outcome else 1
    if nc >= beta:
        return ("D", npref)
    return ("U", nd, npref, outcome, nc)


@dataclass(frozen=True)
class SnowballLayout:
    beta: int
    r_max: int
    state_count: int
    states: tuple[tuple, ...]            # index -> state tuple
    decided_plus: int
    decided_minus: int
    initial_plus: int                    # ("U",0,+1,0,0)
    initial_minus: int                   # ("U",0,-1,0,0)
    pref_plus_idx: tuple[int, ...]       # states answering "+" when polled (incl decided+)
    pref_minus_idx: tuple[int, ...]
    # sparse transition template, grouped by kind: kind_from[k], kind_to[k] are
    # equal-length index lists (CPU tensors) of the transitions driven by that kind.
    kind_from: tuple[torch.Tensor, ...]
    kind_to: tuple[torch.Tensor, ...]


@lru_cache(maxsize=None)
def snowball_layout(beta: int, r_max: int) -> SnowballLayout:
    """Enumerate the reachable true-Snowball states and build the sparse transition.

    BFS from the two initial states ``(d=0, pref=±, last=⊥, c=0)``; the reachable set
    is finite because ``d`` is clamped to ``[-r_max, r_max]`` and ``c < β``.
    """
    if not isinstance(beta, int) or beta < 1:
        raise ValueError("beta must be a positive int")
    if not isinstance(r_max, int) or r_max < 1:
        raise ValueError("r_max must be a positive int")

    init_plus = ("U", 0, 1, 0, 0)
    init_minus = ("U", 0, -1, 0, 0)
    index: dict[tuple, int] = {}
    order: list[tuple] = []

    def intern(s: tuple) -> int:
        i = index.get(s)
        if i is None:
            i = len(order)
            if i >= _MAX_STATES:
                raise ValueError(
                    f"true-Snowball state count exceeds {_MAX_STATES} for "
                    f"beta={beta}, r_max={r_max} (spec stop condition #5)"
                )
            index[s] = i
            order.append(s)
        return i

    intern(init_plus)
    intern(init_minus)
    frontier = [init_plus, init_minus]
    while frontier:
        nxt: list[tuple] = []
        for s in frontier:
            for o in (PLUS, MINUS, ZERO):
                t = _step(s, o, beta, r_max)
                if t not in index:
                    intern(t)
                    nxt.append(t)
        frontier = nxt

    states = tuple(order)
    S = len(states)

    kind_from: list[list[int]] = [[], [], [], []]
    kind_to: list[list[int]] = [[], [], [], []]
    for i, s in enumerate(states):
        if s[0] == "D":
            kind_from[KIND_ONE].append(i)
            kind_to[KIND_ONE].append(i)  # absorbing self-loop
            continue
        for o, kind in ((PLUS, KIND_HPLUS), (MINUS, KIND_HMINUS), (ZERO, KIND_HZERO)):
            j = index[_step(s, o, beta, r_max)]
            kind_from[kind].append(i)
            kind_to[kind].append(j)

    decided_plus = index[("D", 1)]
    decided_minus = index[("D", -1)]
    pref_plus_idx = tuple(
        i for i, s in enumerate(states)
        if (s[0] == "U" and s[2] == 1) or (s[0] == "D" and s[1] == 1)
    )
    pref_minus_idx = tuple(
        i for i, s in enumerate(states)
        if (s[0] == "U" and s[2] == -1) or (s[0] == "D" and s[1] == -1)
    )

    return SnowballLayout(
        beta=beta,
        r_max=r_max,
        state_count=S,
        states=states,
        decided_plus=decided_plus,
        decided_minus=decided_minus,
        initial_plus=index[init_plus],
        initial_minus=index[init_minus],
        pref_plus_idx=pref_plus_idx,
        pref_minus_idx=pref_minus_idx,
        kind_from=tuple(torch.tensor(f, dtype=torch.long) for f in kind_from),
        kind_to=tuple(torch.tensor(t, dtype=torch.long) for t in kind_to),
    )


def reachable_states(beta: int, r_max: int) -> tuple[tuple, ...]:
    """Ordered tuple of reachable state tuples (for tests / inspection)."""
    return snowball_layout(beta, r_max).states


def _kind_index(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    return t.to(device=device)


def apply_round(
    p: torch.Tensor,
    h_plus: torch.Tensor,
    h_minus: torch.Tensor,
    h_zero: torch.Tensor,
    layout: SnowballLayout,
) -> torch.Tensor:
    """One differentiable round, applied as a sparse scatter (no ``[...,S,S]``).

    Args:
        p: state distribution ``[..., S]`` (sums to 1 along the last dim).
        h_plus, h_minus, h_zero: ``[...]`` per-node/scenario quorum outcome
            probabilities (each broadcastable to ``p[..., 0]``); sum to 1.
        layout: the :class:`SnowballLayout` for ``(beta, r_max)``.

    Returns:
        ``[..., S]`` next-round state distribution.
    """
    if p.shape[-1] != layout.state_count:
        raise ValueError(f"p last dim {p.shape[-1]} != state_count {layout.state_count}")
    device = p.device
    p_next = torch.zeros_like(p)
    for kind, h in ((KIND_HPLUS, h_plus), (KIND_HMINUS, h_minus), (KIND_HZERO, h_zero)):
        fr = layout.kind_from[kind]
        if fr.numel() == 0:
            continue
        fr = _kind_index(fr, device)
        to = _kind_index(layout.kind_to[kind], device)
        contrib = p.index_select(-1, fr) * h.unsqueeze(-1)
        p_next = p_next.index_add(-1, to, contrib)
    fr1 = layout.kind_from[KIND_ONE]
    if fr1.numel() > 0:
        fr1 = _kind_index(fr1, device)
        to1 = _kind_index(layout.kind_to[KIND_ONE], device)
        p_next = p_next.index_add(-1, to1, p.index_select(-1, fr1))
    return p_next


def transition_matrix(
    h_plus: torch.Tensor,
    h_minus: torch.Tensor,
    h_zero: torch.Tensor,
    layout: SnowballLayout,
) -> torch.Tensor:
    """Dense ``[..., S, S]`` row-stochastic transition (small-``S`` tests only).

    Provided for verifying row-stochasticity directly; the production path uses the
    sparse :func:`apply_round` and never materialises this matrix.
    """
    shape = h_plus.shape
    S = layout.state_count
    T = torch.zeros((*shape, S, S), dtype=h_plus.dtype, device=h_plus.device)
    vals = {KIND_HPLUS: h_plus, KIND_HMINUS: h_minus, KIND_HZERO: h_zero}
    for kind, h in vals.items():
        for f, t in zip(layout.kind_from[kind].tolist(), layout.kind_to[kind].tolist()):
            T[..., f, t] = T[..., f, t] + h
    for f, t in zip(layout.kind_from[KIND_ONE].tolist(), layout.kind_to[KIND_ONE].tolist()):
        T[..., f, t] = T[..., f, t] + torch.ones_like(h_plus)
    return T


def initial_distribution(
    initial_correct_preference: torch.Tensor | float,
    layout: SnowballLayout,
    batch: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Initial state ``[B, S]``: mass ``init`` at ``(d=0,pref=+)``, ``1-init`` at
    ``(d=0,pref=-)`` (spec §3.2; ``init`` = per-node correct-observation probability)."""
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
    p0[:, layout.initial_plus] = init
    p0[:, layout.initial_minus] = 1.0 - init
    return p0


def readout_preference(p: torch.Tensor, layout: SnowballLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Current correct/wrong answer probabilities ``(u, v)`` when the node is polled.

    A node answers its current ``pref`` (decided nodes answer their decided colour);
    ``u + v = 1``.
    """
    device = p.device
    pp = torch.tensor(layout.pref_plus_idx, dtype=torch.long, device=device)
    u = p.index_select(-1, pp).sum(dim=-1)
    v = 1.0 - u
    return u, v


def terminal_outcomes(p: torch.Tensor, layout: SnowballLayout) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Terminal ``(c, w, undecided)`` = P(decided +), P(decided -), P(still undecided)."""
    c = p[..., layout.decided_plus]
    w = p[..., layout.decided_minus]
    undecided = torch.clamp(1.0 - c - w, 0.0, 1.0)
    return c, w, undecided


def simulate_trajectory(
    outcomes: list[int],
    beta: int,
    r_max: int,
    *,
    initial_pref: int = 1,
) -> list[tuple]:
    """Deterministic state path for an explicit quorum-outcome sequence (for tests).

    Returns the list of states ``[s_0, s_1, ..., s_T]`` where ``s_0`` is the initial
    state and ``s_{t+1} = step(s_t, outcomes[t])``. ``outcomes[t] ∈ {+1,-1,0}``.
    """
    if initial_pref not in (1, -1):
        raise ValueError("initial_pref must be +1 or -1")
    s = ("U", 0, initial_pref, 0, 0)
    path = [s]
    for o in outcomes:
        if o not in (PLUS, MINUS, ZERO):
            raise ValueError("outcomes must be in {+1,-1,0}")
        s = _step(s, o, beta, r_max)
        path.append(s)
    return path
