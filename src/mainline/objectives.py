"""Independent global delay (D) and network energy (E) objectives (spec §9, G6).

D and E are computed from genuinely different statistics of the §6 recurrence, so they
are NOT proportional (the old ``D ∝ E`` is gone):

* ``D`` -- global completion delay -- is the order statistic of the LAST eligible node to
  decide correctly (Eqs. 42-46).  From ``S(t) = sum_r omega_r prod_{i in H} c_ir(t)`` (the
  probability ALL eligible nodes are correct by round ``t``, which is the CDF of the global
  completion round ``T_all``),

      D_round = E[T_all | T_all <= R_max] = sum_{t=0}^{R-1} (S(R) - S(t)) / S(R)          (Eq. 44)
      D = tau_round * D_round                                                              (Eq. 45)
      D_cap = tau_round * sum_{t=0}^{R-1} (1 - S(t))   (deadline-penalised diagnostic)      (Eq. 46)

* ``E`` -- network total energy -- is the SUM over all nodes, rounds, scenarios and
  retransmission attempts (linearity of expectation, NOT a global max), Eqs. 50-53:

      nbar_ij = (1 - (1-ell_ij)^M) / ell_ij                                                (Eq. 50)
      e_attempt_ij = E_tx + E_rx + E_proc                                                   (Eq. 51)
      e_round_i = sum_j pi_ij nbar_ij e_attempt_ij + E_maint_i                              (Eq. 52)
      E = sum_r omega_r sum_t sum_i tau_ir(t) e_round_ir                                    (Eq. 53)

To create a real F/D/E Pareto conflict the model gets two extra control heads (§9.4):

      P_i = P_min + (P_max - P_min) sigmoid(r_i)                                            (Eq. 54)
      n_i = n_min + (n_max - n_min) sigmoid(b_i)                                            (Eq. 55)

Higher power and larger blocklength both raise link reliability ``ell``; higher power
speeds completion (D down) while larger blocklength lengthens each round (D up), and both
raise transmit energy (E up).  NOTE: F is NOT monotone in reliability for this Snowball
quorum -- it is U-shaped, bottoming at an intermediate ``ell`` and rising again at very high
reliability (better links also propagate WRONG votes from the initial wrong-leaning mass,
so an over-reliable network locks in more wrong decisions).  So no single control optimises
all three objectives, and F itself has an interior optimum -- a genuine three-way conflict,
not the naive "more reliability is always better" of spec §9.4 (see REFACTOR_PROGRESS D7).
The §9.2 wall-clock query expansion ``E[L_i]`` (Eq. 49) is also provided.
"""

from __future__ import annotations

import torch

from .symmetric_polynomials import log_elementary_symmetric

__all__ = [
    "power_head",
    "blocklength_head",
    "expected_attempts",
    "attempt_energy",
    "completion_delay",
    "network_energy",
    "wall_clock_attempts",
    "delay_from_cdf_reference",
]


def power_head(logits: torch.Tensor, p_min_dbm: float, p_max_dbm: float) -> torch.Tensor:
    """Per-node transmit power ``P_i = P_min + (P_max-P_min) sigmoid(r_i)`` [dBm] (Eq. 54)."""
    if p_max_dbm < p_min_dbm:
        raise ValueError("p_max_dbm must be >= p_min_dbm")
    return p_min_dbm + (p_max_dbm - p_min_dbm) * torch.sigmoid(logits)


def blocklength_head(logits: torch.Tensor, n_min: float, n_max: float) -> torch.Tensor:
    """Per-node blocklength ``n_i = n_min + (n_max-n_min) sigmoid(b_i)`` [ch. uses] (Eq. 55)."""
    if n_max < n_min:
        raise ValueError("n_max must be >= n_min")
    return n_min + (n_max - n_min) * torch.sigmoid(logits)


def expected_attempts(ell: torch.Tensor, max_attempts: int) -> torch.Tensor:
    """Truncated-geometric expected number of attempts ``nbar = (1-(1-ell)^M)/ell`` (Eq. 50).

    Differentiable; the ``ell -> 0`` limit is ``M`` (handled without a 0/0).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    one_minus = (1.0 - ell).clamp(0.0, 1.0)
    num = 1.0 - one_minus ** max_attempts
    safe = ell.clamp_min(1e-12)
    nbar = num / safe
    # ell -> 0 limit: (1-(1-ell)^M)/ell -> M
    return torch.where(ell > 1e-9, nbar, torch.full_like(ell, float(max_attempts)))


def attempt_energy(
    power_dbm: torch.Tensor,
    blocklength: torch.Tensor,
    *,
    symbol_time_s: float = 1e-5,
    rx_power_w: float = 0.1,
    proc_energy_j: float = 1e-4,
) -> torch.Tensor:
    """Energy of one request-response attempt ``E_tx + E_rx + E_proc`` [J] (Eq. 51).

    ``E_tx = P_lin_W * t_tx``, ``E_rx = rx_power_w * t_tx`` with transmission time
    ``t_tx = blocklength * symbol_time_s`` (longer blocklength -> more energy AND time),
    plus a fixed processing term.  Differentiable in power and blocklength.
    """
    p_w = torch.pow(torch.as_tensor(10.0, dtype=power_dbm.dtype, device=power_dbm.device),
                    (power_dbm - 30.0) / 10.0)  # dBm -> W
    t_tx = blocklength * symbol_time_s
    return p_w * t_tx + rx_power_w * t_tx + proc_energy_j


def completion_delay(S_trajectory: torch.Tensor, tau_round: float = 1.0) -> dict:
    """Global completion delay from the ``S(t)`` trajectory (Eqs. 44-46).

    Args:
        S_trajectory: ``[R+1]`` with ``S(0)..S(R)`` (the CDF of the global completion round).
        tau_round: per-round wall-clock duration.

    Returns ``D_round`` (Eq. 44), ``D`` (Eq. 45), ``D_cap`` (Eq. 46).
    """
    if S_trajectory.ndim != 1 or S_trajectory.numel() < 2:
        raise ValueError("S_trajectory must be 1-D with >= 2 entries (t=0..R)")
    S_R = S_trajectory[-1].clamp_min(1e-12)
    D_round = (S_R - S_trajectory[:-1]).sum() / S_R
    D = tau_round * D_round
    D_cap = tau_round * (1.0 - S_trajectory[:-1]).sum()
    return {"D_round": D_round, "D": D, "D_cap": D_cap}


def delay_from_cdf_reference(S_trajectory: torch.Tensor, tau_round: float = 1.0) -> torch.Tensor:
    """Independent reference for ``D`` via ``E[T_all | success] = sum_t t (S(t)-S(t-1)) / S(R)``.

    Equals :func:`completion_delay`'s ``D`` by Abel summation -- used to validate Eq. 44.
    """
    S = S_trajectory
    R = S.numel() - 1
    S_prev = torch.cat([S.new_zeros(1), S[:-1]])  # S(t-1), S(-1)=0
    pmf = S - S_prev  # P(T_all = t)
    t = torch.arange(R + 1, dtype=S.dtype, device=S.device)
    return tau_round * (t * pmf).sum() / S[-1].clamp_min(1e-12)


def network_energy(
    tau_trajectory: torch.Tensor,   # [R_active, N, Q] transient prob per ACTIVE round
    pi_edge: torch.Tensor,          # [E] or [E, Q] inclusion probability
    ell_edge: torch.Tensor,         # [E] or [E, Q] link reliability
    attempt_energy_edge: torch.Tensor,  # [E] or [E, Q] per-attempt energy
    src_index: torch.Tensor,
    num_nodes: int,
    scenario_weight: torch.Tensor,  # [Q]
    max_attempts: int,
    *,
    maint_energy_node: torch.Tensor | float = 0.0,  # [N] or [N, Q] per-round maintenance
) -> dict:
    """Network total expected energy ``E`` (Eqs. 50-53).  Returns ``E`` and per-node energy.

    CONTRACT: ``tau_trajectory`` carries one transient-state snapshot per ENERGY-CONSUMING
    round, i.e. ``t = 0 .. R_max-1`` (Eq. 53 range).  Round ``r`` (r=1..R_max) is paid by
    nodes active *entering* it, ``tau(r-1)``, so when using the §6 evaluator's
    ``[R_max+1]``-length trajectory you must pass ``result.tau_trajectory[:-1]`` (the
    terminal post-final state ``tau(R_max)`` has no subsequent round and consumes nothing).
    This mirrors :func:`completion_delay` truncating ``S_trajectory[:-1]``.
    """
    Q = int(scenario_weight.numel())
    omega = scenario_weight.reshape(Q)

    def _to_eq(x):
        x = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=tau_trajectory.dtype)
        if x.ndim == 1:
            return x.unsqueeze(-1).expand(x.shape[0], Q)
        return x

    pi = _to_eq(pi_edge)               # [E, Q]
    ell = _to_eq(ell_edge)             # [E, Q]
    e_att = _to_eq(attempt_energy_edge)  # [E, Q]
    nbar = expected_attempts(ell, max_attempts)  # [E, Q]
    g = pi * nbar * e_att              # [E, Q] per-edge support energy
    # e_round_i = sum_{j} pi_ij nbar_ij e_attempt_ij + maint
    e_round = tau_trajectory.new_zeros((num_nodes, Q)).index_add(0, src_index, g)  # [N, Q]
    if isinstance(maint_energy_node, torch.Tensor):
        maint = maint_energy_node if maint_energy_node.ndim == 2 else maint_energy_node.unsqueeze(-1).expand(num_nodes, Q)
        e_round = e_round + maint
    elif maint_energy_node:
        e_round = e_round + float(maint_energy_node)
    # E = sum_r omega_r sum_i (sum_t tau_ir(t)) e_round_ir
    tau_sum = tau_trajectory.sum(dim=0)  # [N, Q] expected active-round count
    per_scenario = (tau_sum * e_round).sum(dim=0)  # [Q]
    E = (omega * per_scenario).sum()
    return {"E": E, "e_round_node": e_round, "active_rounds_node": tau_sum}


def wall_clock_attempts(
    log_weights: torch.Tensor,  # [B, n] query logits per source
    ell: torch.Tensor,          # [B, n] single-attempt link success per candidate
    k: int,
    max_attempts: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected attempts of the slowest selected query ``E[L_i]`` per source (Eq. 49).

    ``E[L_i] = sum_{m=0}^{M-1} [1 - e_k(a_i ⊙ f_i(m)) / e_k(a_i)]`` with
    ``f_ij(m) = 1-(1-ell_ij)^m`` (Eq. 47) and the §4 weighted k-subset normaliser.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    log_ek_a = log_elementary_symmetric(log_weights, k, mask=mask)[..., k]  # [B]
    one_minus = (1.0 - ell).clamp(0.0, 1.0)
    out = torch.zeros(log_weights.shape[:-1], dtype=log_weights.dtype, device=log_weights.device)
    for m in range(max_attempts):
        f_m = 1.0 - one_minus ** m  # f_ij(m); f(0)=0
        # a ⊙ f(m) -> log weight = log_w + log f(m); f(m)=0 -> -inf (drops that candidate)
        log_f = torch.log(f_m.clamp_min(1e-300))
        log_ek_af = log_elementary_symmetric(log_weights + log_f, k, mask=mask)[..., k]
        ratio = torch.exp(log_ek_af - log_ek_a)  # P(L_i <= m)  (Eq. 48)
        out = out + (1.0 - ratio)
    return out
