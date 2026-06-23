"""Shared finite-mixture global consensus evaluator (spec §3, §6 -- the H1 core).

This is the mainline spine of Eq. 56:

    (a, ell, omega)  ->  graph-coupled Snowball recurrence (Eq. 32)
                     ->  per-scenario terminal marginals c_ir, w_ir, u_ir
                     ->  log-domain product-mixture S_C, F_global (Eqs. 6-13)

A network-shared discrete latent scenario ``Z in {1..Q}`` with weights ``omega_r``
(spec §3.1) couples the nodes: *given* ``Z=r`` the terminal node states are a
conditional product (Eq. 5), and

    S_C       = sum_r omega_r prod_{i in H} c_ir                                   (Eq. 6)
    F_global  = 1 - S_C = -expm1(logsumexp_r(log omega_r + sum_i log c_ir))        (Eqs. 7,11,12)
    L_F       = -log S_C                                                           (Eq. 13)

This ``F_global`` is the EXACT global event probability ``P(exists i: Y_i != C)`` of
the defined shared-mixture conditional-product joint -- a genuine global event
probability, NOT a per-node marginal mean (H1).  Cross-node correlation is carried
by the shared latent ``Z``; the within-scenario marginals ``c_ir`` come from the
§6 graph-coupled finite-horizon recurrence, whose per-round quorum probabilities use
the exact heterogeneous quorum DP of §5 (no iid beta-tail, no degree cap).

Per the spec §3.4 exactness boundary, this is exact *under the proposed model*; it is
NOT claimed exact for the unrestricted Avalanche process.  Fidelity to a richer
ground-truth simulation is a reported diagnostic (see ``monte_carlo_global_success``),
not an exactness claim.

Complexity.  The per-round quorum work is evaluated with a *degree-bucketed* layout:
sources are grouped into geometric out-degree buckets and each bucket is padded only
to its own max degree, so total padded cells are ``<= 2E`` regardless of degree skew
(a single high-degree hub no longer forces every row to pay ``max_deg``).  Overall
cost is ``O(Q R_max (E k^3 + N beta^2))`` -- near-linear in N and E with NO ``N x N``
dense tensor, holding *unconditionally* (not only under bounded max degree).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .quorum_dp import quorum_decision_probabilities
from .snowball import (
    build_transition,
    initial_distribution,
    readout_preference,
    snowball_state_count,
    terminal_outcomes,
)

__all__ = [
    "SourcePadding",
    "DegreeBucket",
    "BucketedPadding",
    "GlobalConsensusResult",
    "build_source_padding",
    "build_bucketed_padding",
    "evaluate_global_consensus",
    "simulate_model_joint",
    "monte_carlo_global_success",
]


@dataclass(frozen=True)
class SourcePadding:
    """Ragged out-edge -> dense ``[N, max_deg]`` layout (utility / diagnostics)."""

    slot_edge: torch.Tensor  # [N, max_deg] edge id for each (source, slot); 0 where invalid
    slot_mask: torch.Tensor  # [N, max_deg] bool, True where the slot is a real edge
    out_degree: torch.Tensor  # [N]
    max_deg: int


def _edge_rank(src: torch.Tensor, deg: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Position of each edge within its source's out-edge group (``rank[e] in [0,deg-1]``)."""
    E = int(src.numel())
    device = src.device
    order = torch.argsort(src, stable=True)
    sorted_src = src[order]
    offset = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    offset[1:] = torch.cumsum(deg, 0)
    rank = torch.empty(E, dtype=torch.long, device=device)
    rank[order] = torch.arange(E, device=device) - offset[sorted_src]
    return rank


def build_source_padding(src_index: torch.Tensor, dst_index: torch.Tensor, num_nodes: int) -> SourcePadding:
    """Dense ``[N, max_deg]`` grouping of directed edges by source (no degree cap, H2).

    Kept as a utility/diagnostic; the evaluator uses :func:`build_bucketed_padding`
    to avoid the ``N x max_deg`` allocation under degree skew.
    """
    if src_index.ndim != 1 or dst_index.ndim != 1 or src_index.numel() != dst_index.numel():
        raise ValueError("src_index and dst_index must be 1-D with equal length")
    device = src_index.device
    E = int(src_index.numel())
    src = src_index.to(torch.long)
    deg = torch.bincount(src, minlength=num_nodes)
    max_deg = int(deg.max().item()) if E > 0 else 0
    slot_edge = torch.zeros((num_nodes, max_deg), dtype=torch.long, device=device)
    slot_mask = torch.zeros((num_nodes, max_deg), dtype=torch.bool, device=device)
    if E > 0:
        rank = _edge_rank(src, deg, num_nodes)
        slot_edge[src, rank] = torch.arange(E, device=device)
        slot_mask[src, rank] = True
    return SourcePadding(slot_edge=slot_edge, slot_mask=slot_mask, out_degree=deg, max_deg=max_deg)


@dataclass(frozen=True)
class DegreeBucket:
    node_ids: torch.Tensor  # [m] source node ids in this bucket
    slot_edge: torch.Tensor  # [m, w] edge id per slot (0 where invalid)
    slot_mask: torch.Tensor  # [m, w] bool
    dst_slot: torch.Tensor  # [m, w] dst node id per slot (dst[slot_edge])
    width: int


@dataclass(frozen=True)
class BucketedPadding:
    """Degree-bucketed out-edge layout: total padded cells ``<= 2E`` for any degree
    distribution.  Built once per graph and reused every round."""

    buckets: tuple[DegreeBucket, ...]
    out_degree: torch.Tensor  # [N]
    total_cells: int


def build_bucketed_padding(src_index: torch.Tensor, dst_index: torch.Tensor, num_nodes: int) -> BucketedPadding:
    """Group sources into geometric out-degree buckets (width = bucket max degree).

    A node of degree ``d`` lands in bucket ``ceil(log2(d))``, padded to the bucket's
    max degree, so per-node padding waste is ``< 2x`` and total cells ``<= 2E``.  No
    fixed degree cap (H2): a hub of degree ``Theta(N)`` occupies its own bucket and
    pays only its own ``Theta(N)`` cells, never forcing other rows to ``max_deg``.
    """
    if src_index.ndim != 1 or dst_index.ndim != 1 or src_index.numel() != dst_index.numel():
        raise ValueError("src_index and dst_index must be 1-D with equal length")
    device = src_index.device
    E = int(src_index.numel())
    src = src_index.to(torch.long)
    dst = dst_index.to(torch.long)
    deg = torch.bincount(src, minlength=num_nodes)
    if E == 0:
        return BucketedPadding(buckets=tuple(), out_degree=deg, total_cells=0)

    rank = _edge_rank(src, deg, num_nodes)

    # bucket index per node: b = ceil(log2(deg)) = (deg-1).bit_length(), via integer shifts.
    node_bucket = torch.zeros(num_nodes, dtype=torch.long, device=device)
    x = (deg - 1).clamp_min(0)
    iters = max(1, int(deg.max().item()).bit_length() + 1)
    for _ in range(iters):
        node_bucket += (x > 0).long()
        x = torch.div(x, 2, rounding_mode="floor")

    buckets: list[DegreeBucket] = []
    total_cells = 0
    present = torch.unique(node_bucket[deg > 0])
    edge_bucket = node_bucket[src]  # [E]
    for b in present.tolist():
        node_ids = torch.nonzero((node_bucket == b) & (deg > 0), as_tuple=False).reshape(-1)
        m = int(node_ids.numel())
        w = int(deg[node_ids].max().item())
        node_to_local = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        node_to_local[node_ids] = torch.arange(m, device=device)
        emask = edge_bucket == b
        e_ids = torch.nonzero(emask, as_tuple=False).reshape(-1)
        rows = node_to_local[src[e_ids]]
        cols = rank[e_ids]
        slot_edge = torch.zeros((m, w), dtype=torch.long, device=device)
        slot_mask = torch.zeros((m, w), dtype=torch.bool, device=device)
        slot_edge[rows, cols] = e_ids
        slot_mask[rows, cols] = True
        dst_slot = dst[slot_edge]
        buckets.append(DegreeBucket(node_ids=node_ids, slot_edge=slot_edge, slot_mask=slot_mask,
                                    dst_slot=dst_slot, width=w))
        total_cells += m * w
    return BucketedPadding(buckets=tuple(buckets), out_degree=deg, total_cells=total_cells)


@dataclass(frozen=True)
class GlobalConsensusResult:
    F_global: torch.Tensor  # scalar, in [0,1]
    S_C: torch.Tensor  # scalar P(all eligible nodes decide correct), in (0,1]
    log_S_C: torch.Tensor  # scalar, <= 0
    loss_F: torch.Tensor  # scalar = -log S_C, >= 0
    c_ir: torch.Tensor  # [N, Q] terminal P(correct | Z=r)
    w_ir: torch.Tensor  # [N, Q] terminal P(wrong | Z=r)
    undecided_ir: torch.Tensor  # [N, Q] terminal P(undecided | Z=r)
    F_any_wrong: torch.Tensor  # Eq. 8
    F_timeout_without_wrong: torch.Tensor  # Eq. 9
    scenario_posterior: torch.Tensor  # [Q] rho_r (Eq. 14)
    # Optional per-round trajectories (populated when return_trajectory=True); used by the
    # §9 delay / energy objectives (G6).  Shape [rounds+1, N, Q] for c/w/tau; [rounds+1] for S.
    c_trajectory: torch.Tensor | None = None      # c_ir(t) = P(decided correct by round t)
    w_trajectory: torch.Tensor | None = None      # w_ir(t)
    tau_trajectory: torch.Tensor | None = None     # tau_ir(t) = transient (undecided) mass
    S_trajectory: torch.Tensor | None = None       # S(t) = sum_r omega_r prod_{i in H} c_ir(t)


def _broadcast_edge(value: torch.Tensor, E: int, Q: int, name: str) -> torch.Tensor:
    v = value
    if v.ndim == 1:
        if v.numel() != E:
            raise ValueError(f"{name} must have one value per edge")
        v = v.unsqueeze(-1).expand(E, Q)
    elif v.ndim == 2:
        if v.shape != (E, Q):
            raise ValueError(f"{name} must be [E] or [E, Q]")
    else:
        raise ValueError(f"{name} must be [E] or [E, Q]")
    return v


def _bucketed_quorum(
    padding: BucketedPadding,
    a_edge: torch.Tensor,  # [E, Q]
    ell_edge: torch.Tensor,  # [E, Q]
    pref_c: torch.Tensor,  # [N, Q]
    pref_w: torch.Tensor,  # [N, Q]
    k: int,
    alpha: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One round of quorum decisions for all sources, bucketed by degree.

    Returns ``(h_plus, h_minus, h_zero)`` each ``[N, Q]``.
    """
    N, Q = pref_c.shape
    h_plus = pref_c.new_zeros((N, Q))
    h_minus = pref_c.new_zeros((N, Q))
    h_zero = pref_c.new_zeros((N, Q))
    for bucket in padding.buckets:
        m, w = bucket.node_ids.numel(), bucket.width
        se = bucket.slot_edge.reshape(-1)  # [m*w]
        a_b = a_edge[se].reshape(m, w, Q)
        ell_b = ell_edge[se].reshape(m, w, Q)
        dst_b = bucket.dst_slot  # [m, w]
        pc_b = ell_b * pref_c[dst_b]  # [m, w, Q]
        pw_b = ell_b * pref_w[dst_b]
        a_bb = a_b.permute(0, 2, 1).reshape(m * Q, w)
        pc_bb = pc_b.permute(0, 2, 1).reshape(m * Q, w)
        pw_bb = pw_b.permute(0, 2, 1).reshape(m * Q, w)
        mask_bb = bucket.slot_mask.unsqueeze(1).expand(m, Q, w).reshape(m * Q, w)
        dec = quorum_decision_probabilities(a_bb, pc_bb, pw_bb, k, alpha, mask=mask_bb)
        h_plus = h_plus.index_copy(0, bucket.node_ids, dec.h_plus.reshape(m, Q))
        h_minus = h_minus.index_copy(0, bucket.node_ids, dec.h_minus.reshape(m, Q))
        h_zero = h_zero.index_copy(0, bucket.node_ids, dec.h_zero.reshape(m, Q))
    return h_plus, h_minus, h_zero


def evaluate_global_consensus(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    log_query_weight: torch.Tensor,  # [E] or [E, Q]
    link_reliability: torch.Tensor,  # [E] or [E, Q]  ell_ij in [0,1]
    scenario_weight: torch.Tensor,  # [Q] omega_r, sums to 1
    k: int,
    alpha: int,
    beta: int,
    rounds: int,
    initial_correct_preference: torch.Tensor | float = 0.5,
    eligible_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
    padding: BucketedPadding | None = None,
    return_trajectory: bool = False,
) -> GlobalConsensusResult:
    """Run the graph-coupled recurrence and compute the global mixture failure.

    Args:
        num_nodes: N.
        src_index, dst_index: directed candidate edges ``i -> j`` (``i`` polls ``j``).
        log_query_weight: GNN logits ``s_{ij}`` (``a = exp(s)``), per edge / scenario.
        link_reliability: ``ell_{ij}`` per edge / scenario (from §8; given here).
        scenario_weight: ``omega`` over the Q shared latent scenarios.
        k, alpha, beta, rounds: quorum size, majority, conviction threshold, R_max.
        initial_correct_preference: initial lean (scalar, [N], or [N, Q]).
        eligible_mask: ``[N]`` bool for the honest eligible set ``H`` (default all).

    Raises:
        ValueError: if any source has out-degree < k (apply the §7.2 candidate-shortage
            protocol -- k_i=min(k,|N_i|) or RSU fallback -- upstream; G4).
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    device = log_query_weight.device
    Q = int(scenario_weight.numel())
    E = int(src_index.numel())

    omega = scenario_weight.reshape(Q)
    if bool(torch.any(omega.detach() < -1e-9).cpu()) or abs(float(omega.sum()) - 1.0) > 1e-5:
        raise ValueError("scenario_weight must be nonnegative and sum to 1")

    if eligible_mask is None:
        eligible_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
    eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)

    if padding is None:
        padding = build_bucketed_padding(src_index, dst_index, num_nodes)

    # §7.2: every participating source (all nodes poll) must form a full quorum.
    if bool(torch.any(padding.out_degree < k).cpu()):
        raise ValueError(
            "a source has out-degree < k; apply the §7.2 candidate-shortage protocol "
            "(k_i=min(k,|N_i|) or RSU fallback) upstream (G4) before G1"
        )

    a_edge = _broadcast_edge(log_query_weight, E, Q, "log_query_weight")  # [E, Q]
    ell_edge = _broadcast_edge(link_reliability, E, Q, "link_reliability")  # [E, Q]
    if bool(torch.any((ell_edge.detach() < -1e-6) | (ell_edge.detach() > 1 + 1e-6)).cpu()):
        raise ValueError("link_reliability must be in [0, 1]")
    ell_edge = ell_edge.clamp(0.0, 1.0)

    dtype = a_edge.dtype
    S = snowball_state_count(beta)
    p0 = initial_distribution(initial_correct_preference, beta, num_nodes * Q, dtype=dtype, device=device)
    p = p0.reshape(num_nodes, Q, S)

    c_traj: list[torch.Tensor] = []
    w_traj: list[torch.Tensor] = []
    if return_trajectory:
        c0, w0, _ = terminal_outcomes(p, beta)  # state at t=0
        c_traj.append(c0)
        w_traj.append(w0)
    for _ in range(rounds):
        pref_c, pref_w = readout_preference(p, beta)  # [N, Q] each (sum to 1)
        h_plus, h_minus, h_zero = _bucketed_quorum(padding, a_edge, ell_edge, pref_c, pref_w, k, alpha)
        T = build_transition(h_plus.reshape(-1), h_minus.reshape(-1), h_zero.reshape(-1), beta)
        p = torch.bmm(p.reshape(num_nodes * Q, 1, S), T).reshape(num_nodes, Q, S)
        if return_trajectory:
            ct, wt, _ = terminal_outcomes(p, beta)
            c_traj.append(ct)
            w_traj.append(wt)

    c_ir, w_ir, undecided_ir = terminal_outcomes(p, beta)  # [N, Q]

    # log-domain product-mixture over eligible set H (Eqs. 11-13).
    # MULTIPLICATIVE floor clamp_min(eps): keeps every factor <= 1 so S_C <= 1 exactly
    # (an ADDITIVE c+eps would make (1+eps)^|H| > 1 -> F_global < 0 at c->1; see D4).
    elig = eligible_mask
    log_c = torch.log(c_ir[elig].clamp_min(eps))  # [|H|, Q]
    log_S_r = log_c.sum(dim=0)  # [Q]
    log_omega = torch.log(omega.clamp_min(torch.finfo(omega.dtype).tiny))
    log_S_C = torch.logsumexp(log_omega + log_S_r, dim=0).clamp_max(0.0)  # scalar, <= 0
    F_global = -torch.expm1(log_S_C)
    loss_F = -log_S_C

    rho = torch.softmax(log_omega + log_S_r, dim=0)  # [Q] scenario posterior (Eq. 14)

    # exact failure decomposition (Eqs. 8-10), same multiplicative floor on both legs so
    # prod(c) <= prod(1-w) termwise -> F_timeout >= 0.
    log_nw = torch.log((1.0 - w_ir[elig]).clamp_min(eps))
    S_no_wrong = torch.exp(torch.logsumexp(log_omega + log_nw.sum(dim=0), dim=0).clamp_max(0.0))
    S_C_lin = torch.exp(log_S_C)
    F_any_wrong = 1.0 - S_no_wrong
    F_timeout = (S_no_wrong - S_C_lin).clamp_min(0.0)

    c_traj_t = w_traj_t = tau_traj_t = S_traj_t = None
    if return_trajectory:
        c_traj_t = torch.stack(c_traj, dim=0)  # [rounds+1, N, Q]
        w_traj_t = torch.stack(w_traj, dim=0)
        tau_traj_t = (1.0 - c_traj_t - w_traj_t).clamp_min(0.0)
        # S(t) = sum_r omega_r prod_{i in H} c_ir(t), log-domain (Eqs. 42-43)
        log_c_t = torch.log(c_traj_t[:, elig, :].clamp_min(eps))  # [T, |H|, Q]
        log_S_r_t = log_c_t.sum(dim=1)  # [T, Q]
        S_traj_t = torch.exp(torch.logsumexp(log_omega + log_S_r_t, dim=1).clamp_max(0.0))  # [T]

    return GlobalConsensusResult(
        F_global=F_global,
        S_C=S_C_lin,
        log_S_C=log_S_C,
        loss_F=loss_F,
        c_ir=c_ir,
        w_ir=w_ir,
        undecided_ir=undecided_ir,
        F_any_wrong=F_any_wrong,
        F_timeout_without_wrong=F_timeout,
        scenario_posterior=rho,
        c_trajectory=c_traj_t,
        w_trajectory=w_traj_t,
        tau_trajectory=tau_traj_t,
        S_trajectory=S_traj_t,
    )


def simulate_model_joint(
    c_ir: torch.Tensor,
    w_ir: torch.Tensor,
    scenario_weight: torch.Tensor,
    *,
    num_samples: int,
    eligible_mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> dict:
    """Monte-Carlo the *defined model joint*: draw Z~omega, then each node C/W/U
    independently from ``(c_ir, w_ir, 1-c-w)``; return empirical global-success and
    any-wrong frequencies.  Validates that ``S_C`` is a genuine global event
    probability (H1), not a node-mean.
    """
    N, Q = c_ir.shape
    device = c_ir.device
    if eligible_mask is None:
        eligible_mask = torch.ones(N, dtype=torch.bool, device=device)
    elig = eligible_mask.to(device=device, dtype=torch.bool)
    omega = scenario_weight.reshape(Q)
    z = torch.multinomial(omega, num_samples, replacement=True, generator=generator)  # [S]
    c_sel = c_ir[:, z].transpose(0, 1)  # [S, N]
    w_sel = w_ir[:, z].transpose(0, 1)
    u = torch.rand(num_samples, N, generator=generator, device=device, dtype=c_ir.dtype)
    is_correct = u < c_sel
    is_wrong = (u >= c_sel) & (u < c_sel + w_sel)
    elig_b = elig.unsqueeze(0)
    all_correct = (is_correct | ~elig_b).all(dim=1)  # ignore non-eligible
    any_wrong = (is_wrong & elig_b).any(dim=1)
    return {
        "global_success_freq": float(all_correct.to(torch.float64).mean()),
        "any_wrong_freq": float(any_wrong.to(torch.float64).mean()),
        "num_samples": num_samples,
    }


def monte_carlo_global_success(result: GlobalConsensusResult, scenario_weight: torch.Tensor, *,
                               num_samples: int = 200000, eligible_mask: torch.Tensor | None = None,
                               generator: torch.Generator | None = None) -> dict:
    """Convenience: MC the model joint from a result and compare to ``S_C``."""
    mc = simulate_model_joint(
        result.c_ir, result.w_ir, scenario_weight,
        num_samples=num_samples, eligible_mask=eligible_mask, generator=generator,
    )
    mc["S_C_analytic"] = float(result.S_C)
    mc["F_global_analytic"] = float(result.F_global)
    mc["abs_error_success"] = abs(mc["global_success_freq"] - mc["S_C_analytic"])
    return mc
