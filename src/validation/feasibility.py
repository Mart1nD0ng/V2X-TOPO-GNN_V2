"""Perfect-link protocol feasibility floors (spec §3.3, plan Phase 5 -- G8, the viability gate).

Before training any model, the protocol parameters ``(k, alpha, beta, R_max)`` must be able to
meet the reliability budget under the BEST case: perfect links (``ell = 1``) and IDEAL
representative sampling (each node polls ``k`` peers representative of the whole network).
This is the well-mixed Snowball limit. If even this floor fails at the target ``N`` and the
safety budget, no topology/GNN can rescue it -- the loop must STOP and report (spec §3.3:
"不得让 GNN 补救不可行协议"; loop stop-condition #1).

Well-mixed recursion (the floor model). Under representative sampling all honest nodes are
exchangeable, so a single per-node Snowball state distribution ``p(t)`` suffices. A node polls
``k`` peers, each preferring the correct colour with the population fraction ``u(t)`` (perfect
link ⇒ every poll returns the peer's colour), so the ternary quorum is a binomial tail

    h^+(t) = P(Binom(k, u) >= alpha),   h^-(t) = P(Binom(k, 1-u) >= alpha),   h^0 = 1-h^+-h^-,

and ``p(t+1) = p(t) · T(h^+,h^-,h^0)`` with the canonical ``binary_snowball`` transition
(self-consistent ``u(t) = readout_correct(p(t))``). This is exact in the well-mixed
``N -> inf`` limit (sampling-with-replacement); it is validated against the independent
dynamic MC in a complete-graph well-mixed setting at measurable failure rates.

Network floors over ``N`` exchangeable (mean-field-independent) honest nodes, from the per-node
terminal ``(c, w, u)`` = P(decided correct / wrong / undecided by ``R_max``):

    F_wrong^floor    = 1 - (1-w)^N
    F_disagree^floor = 1 - [ (1-w)^N + (1-c)^N - u^N ]            (inclusion-exclusion)
    F_deadline^floor = 1 - c^N                                    (P(not all correct by R_max))

Feasible iff ``F_disagree<=eps_s/10``, ``F_wrong<=eps_v/10``, ``F_deadline<=eps_d/10``
(spec §3.3). Computed in the log domain so the ``~1e-7`` per-node tails the floors require are
representable. Exactness boundary: this is the well-mixed mean-field floor (the idealised best
case), NOT the canonical local-topology headline evaluator; the deep tail (``~1e-7``) rests on
the recursion (MC-validated at measurable rates) + the classic exponential-in-``beta`` Snowball
safety scaling -- direct MC of ``1e-7`` needs rare-event methods (spec §8.1 lvl 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from src.protocol.binary_snowball import (
    apply_round,
    initial_distribution,
    readout_preference,
    snowball_layout,
    terminal_outcomes,
)

__all__ = [
    "FeasibilityFloors",
    "binomial_tail",
    "wellmixed_terminal",
    "network_floors",
    "is_feasible",
    "scan_feasibility",
]


def binomial_tail(k: int, alpha: int, u: torch.Tensor) -> torch.Tensor:
    """``P(Binom(k, u) >= alpha)`` (differentiable in ``u``)."""
    out = torch.zeros_like(u)
    for m in range(alpha, k + 1):
        out = out + math.comb(k, m) * u ** m * (1 - u) ** (k - m)
    return out


def wellmixed_terminal(init_correct: float, k: int, alpha: int, beta: int, r_max: int,
                       *, dtype: torch.dtype = torch.float64) -> tuple[float, float, float]:
    """Per-node terminal ``(c, w, undecided)`` under the well-mixed perfect-link recursion."""
    if not (0.0 <= init_correct <= 1.0):
        raise ValueError("init_correct must be in [0, 1]")
    if not (1 <= alpha <= k) or 2 * alpha <= k:
        # strict majority required so {>=alpha correct} and {>=alpha wrong} are exclusive
        # (h^0 = 1 - h^+ - h^- only valid then); otherwise the update is non-stochastic.
        raise ValueError("alpha must be a strict majority of k (1 <= alpha <= k and 2*alpha > k)")
    layout = snowball_layout(beta, r_max)
    p = initial_distribution(torch.tensor([init_correct], dtype=dtype), layout, 1, dtype=dtype)
    for _ in range(r_max):
        u, v = readout_preference(p, layout)                 # [1]
        h_plus = binomial_tail(k, alpha, u)
        h_minus = binomial_tail(k, alpha, v)
        h_zero = (1.0 - h_plus - h_minus).clamp_min(0.0)
        p = apply_round(p, h_plus, h_minus, h_zero, layout)
    c, w, undec = terminal_outcomes(p, layout)
    return float(c), float(w), float(undec)


@dataclass(frozen=True)
class FeasibilityFloors:
    F_disagree: float
    F_wrong: float
    F_deadline: float
    c: float       # per-node P(decided correct by R_max)
    w: float       # per-node P(decided wrong)
    undecided: float


def _one_minus_pow(x: float, N: int) -> float:
    """``1 - (1-x)^N`` stable for tiny ``x`` (log domain)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return -math.expm1(N * math.log1p(-x))


def network_floors(c: float, w: float, undec: float, N: int) -> FeasibilityFloors:
    """Network reliability floors over ``N`` exchangeable honest nodes (mean-field independent)."""
    F_wrong = _one_minus_pow(w, N)
    # F_disagree = 1 - [P(no wrong) + P(no correct decided) - P(none decided)]
    #            = [1 - (1-w)^N] - (1-c)^N + undec^N   (log-domain on the dominant 1-(1-w)^N
    #              term to avoid catastrophic cancellation of two near-1 quantities).
    no_correct = math.exp(N * math.log1p(-c)) if c < 1 else 0.0
    none_decided = math.exp(N * math.log(undec)) if undec > 0 else 0.0
    F_disagree = min(1.0, max(0.0, _one_minus_pow(w, N) - no_correct + none_decided))
    F_deadline = _one_minus_pow(1.0 - c, N)   # 1 - c^N  (P(not all decided correct by R_max))
    return FeasibilityFloors(F_disagree=F_disagree, F_wrong=F_wrong, F_deadline=F_deadline,
                             c=c, w=w, undecided=undec)


def is_feasible(floors: FeasibilityFloors, *, eps_s: float, eps_v: float, eps_d: float) -> bool:
    """Feasible iff the floors meet ``eps_./10`` (spec §3.3)."""
    return (floors.F_disagree <= eps_s / 10.0
            and floors.F_wrong <= eps_v / 10.0
            and floors.F_deadline <= eps_d / 10.0)


def scan_feasibility(
    init_correct: float,
    N: int,
    *,
    eps_s: float = 1e-4,
    eps_v: float = 1e-3,
    eps_d: float = 1e-2,
    k_values=(3, 5, 7),
    alpha_frac: float = 0.7,
    beta_values=tuple(range(2, 41)),
    r_max_factor: int = 4,
) -> dict:
    """Find the minimum-``beta`` feasible ``(k, alpha, beta, R_max)`` at scale ``N``.

    ``alpha = ceil(alpha_frac * k)`` clamped to a strict majority ``2*alpha > k``; ``R_max =
    r_max_factor * beta`` (enough rounds to finalize). Returns the feasible parameter set (min
    beta per k) and the floors achieved, or ``feasible=False`` if none in the grid works.
    """
    feasible_params = []
    for k in k_values:
        alpha = max((k // 2) + 1, math.ceil(alpha_frac * k))      # strict majority
        if 2 * alpha <= k:
            alpha = k // 2 + 1
        for beta in beta_values:
            r_max = r_max_factor * beta
            c, w, undec = wellmixed_terminal(init_correct, k, alpha, beta, r_max)
            floors = network_floors(c, w, undec, N)
            if is_feasible(floors, eps_s=eps_s, eps_v=eps_v, eps_d=eps_d):
                feasible_params.append({
                    "k": k, "alpha": alpha, "beta": beta, "r_max": r_max,
                    "F_disagree": floors.F_disagree, "F_wrong": floors.F_wrong,
                    "F_deadline": floors.F_deadline, "c": floors.c, "w": floors.w,
                })
                break  # min beta for this k
    return {
        "init_correct": init_correct, "N": N,
        "thresholds": {"eps_s": eps_s, "eps_v": eps_v, "eps_d": eps_d},
        "feasible": len(feasible_params) > 0,
        "feasible_params": feasible_params,
    }
