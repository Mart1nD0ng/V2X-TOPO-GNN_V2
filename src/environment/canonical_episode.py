"""The single canonical consensus evaluator (spec §7.2, plan §4 -- G0/G3).

``run_consensus_episode`` is the ONE entry point every headline / baseline / ablation /
figure must call (non-negotiable constraint #6; plan §17 prohibition #16). It runs the
spec §7.2 per-round closed loop

    X_t -> tau_t -> Pi_theta -> Lambda_t -> gamma_t, ell_t -> (h+,h-,h0) -> X_{t+1}

tying together the true binary Snowball protocol (``src.protocol.binary_snowball``), the
correlated-evidence shared-latent decomposition (``evidence_model``), the two physical
graphs ``G_comm`` / ``G_int`` and the round-coupled physics (``round_physics``), and the
exact heterogeneous quorum DP (``src.mainline.quorum_dp`` via the bucketed layout). There
is **no fixed ``tau_proxy``** (constraint #7) and the same query policy is used end to end
(constraint #3); the policy sees only observable features, never ``Y*`` or peer votes
(constraint #10 -- the policy weights are scenario-independent ``[E]``).

Reliability (spec §4) under the shared-latent conditional-independence model (given the
region-bit scenario ``Z=r`` the terminal node states are independent):

    F_disagree = 1 - sum_r omega_r [ prod_i(1-c_ir) + prod_i(1-w_ir) - prod_i u_ir ]   (§4.1)
    F_wrong    = 1 - sum_r omega_r prod_i (1 - w_ir)                                    (§4.2)
    S_allcorrect = sum_r omega_r prod_i c_ir

over the eligible honest set. Exactness boundary: this analytic episode is the
differentiable surrogate; it is exact only under the shared-latent model, and the
independent dynamic MC (G6, ``mode='dynamic_mc'``) is the final judge (spec §8.3).

A runtime ``mechanism_trace`` (G0) records which mechanisms actually executed; the
mechanism-activation sentinels (each ``disable_*`` flag changes the terminal ``F``) prove
the mechanisms are on the canonical path, not merely implemented (constraint #13).

Complexity: ``O(R (E_comm + E_int) Q + R N Q S)`` -- near-linear in ``N, E`` with no
``N x N`` tensor (constraint #11); ``S = O(R_max beta)``; ``Q = 2^G`` scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.mainline.global_evaluator import (
    BucketedPadding,
    _bucketed_quorum,
    build_bucketed_padding,
)
from src.protocol.binary_snowball import (
    apply_round,
    initial_distribution,
    readout_preference,
    snowball_layout,
    terminal_outcomes,
)

from .candidate_graph import build_candidate_graph
from .evidence_model import EvidenceModel
from .interference_graph import build_interference_graph
from .round_physics import EdgeGeometry, RoundPhysicsConfig, edge_geometry, round_physics
from .urban_scene import ManhattanScene

__all__ = ["ProtocolConfig", "EpisodeResult", "run_consensus_episode"]


@dataclass(frozen=True)
class ProtocolConfig:
    """Snowball quorum parameters (spec §3.3); calibrated in Phase 5 (G8)."""

    k: int = 4               # query subset size
    alpha: int = 3           # quorum majority (2*alpha > k so + / - are exclusive)
    beta: int = 5            # consecutive-quorum decision threshold
    r_max: int = 20          # finite horizon (rounds)

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not (1 <= self.alpha <= self.k):
            raise ValueError("alpha must satisfy 1 <= alpha <= k")
        if 2 * self.alpha <= self.k:
            raise ValueError("alpha must be a strict majority: 2*alpha > k")
        if self.beta < 1:
            raise ValueError("beta must be >= 1")
        if self.r_max < 1:
            raise ValueError("r_max must be >= 1")

    def config_hash(self) -> str:
        """Deterministic SHA-256 of the protocol parameters (ExperimentSpec, plan §4)."""
        import hashlib
        import json
        payload = json.dumps({"k": self.k, "alpha": self.alpha, "beta": self.beta,
                              "r_max": self.r_max}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EpisodeResult:
    F_disagree: torch.Tensor       # scalar, agreement-safety failure (spec §4.1)
    F_wrong: torch.Tensor          # scalar, validity failure (spec §4.2)
    S_allcorrect: torch.Tensor     # scalar, P(all eligible decide correct)
    c_ir: torch.Tensor             # [N, Q] terminal P(decided correct | Z=r)
    w_ir: torch.Tensor             # [N, Q] terminal P(decided wrong | Z=r)
    undecided_ir: torch.Tensor     # [N, Q]
    scenario_weight: torch.Tensor  # [Q] omega_r
    energy: torch.Tensor           # [Q] total network energy
    cumulative_time: torch.Tensor  # [Q] total wall-clock (sum of round durations)
    round_duration: torch.Tensor   # [R, Q] per-round network duration (max active tau)
    mechanism_trace: dict
    c_trajectory: torch.Tensor | None = None   # [R+1, N, Q]
    w_trajectory: torch.Tensor | None = None   # [R+1, N, Q]
    tau_trajectory: torch.Tensor | None = None  # [R, N, Q]


def _mixture_reliability(c_ir, w_ir, u_ir, omega, elig, eps: float = 1e-12):
    """Shared-latent mixture safety/validity (spec §4.1-§4.2), log-domain over H."""
    c_e = c_ir[elig].clamp(eps, 1.0)
    w_e = w_ir[elig].clamp(0.0, 1.0 - eps)
    u_e = u_ir[elig].clamp(eps, 1.0)
    logA = torch.log((1.0 - c_e).clamp_min(eps)).sum(dim=0)  # [Q] P(no node decides +)
    logB = torch.log((1.0 - w_e).clamp_min(eps)).sum(dim=0)  # [Q] P(no node decides -)
    logC = torch.log(u_e).sum(dim=0)                          # [Q] P(no node decides)
    logAll = torch.log(c_e).sum(dim=0)                        # [Q] P(all correct)
    agree_r = (torch.exp(logA) + torch.exp(logB) - torch.exp(logC)).clamp(0.0, 1.0)
    no_wrong_r = torch.exp(logB)
    allcorrect_r = torch.exp(logAll)
    F_disagree = (1.0 - (omega * agree_r).sum()).clamp(0.0, 1.0)
    F_wrong = (1.0 - (omega * no_wrong_r).sum()).clamp(0.0, 1.0)
    S_allcorrect = (omega * allcorrect_r).sum().clamp(0.0, 1.0)
    return F_disagree, F_wrong, S_allcorrect


def run_consensus_episode(
    scene: ManhattanScene,
    evidence: EvidenceModel,
    query_policy,
    protocol_cfg: ProtocolConfig,
    phy_cfg: RoundPhysicsConfig,
    *,
    mode: str = "analytic",
    eligible_mask: torch.Tensor | None = None,
    return_trajectory: bool = True,
    return_trace: bool = True,
    disable_interference: bool = False,
    disable_collision: bool = False,
    disable_half_duplex: bool = False,
    disable_queueing: bool = False,
    link_override: float | None = None,
    max_scenarios: int = 1 << 16,
) -> EpisodeResult:
    """Run one consensus episode through the canonical round-coupled loop (spec §7.2).

    Args:
        scene: the urban scene (geometry + regions).
        evidence: the correlated-evidence model (must share ``scene``'s region structure).
        query_policy: object with ``log_weights(graph)->[E]`` and ``name`` (no truth/vote).
        protocol_cfg: Snowball ``(k, alpha, beta, r_max)``.
        phy_cfg: fixed headline physical resources.
        mode: ``"analytic"`` (this function) or ``"dynamic_mc"`` (G6, separate module).
        disable_*: ablation switches forwarded to the round physics (G3 causal tests / G0
            mechanism-activation sentinels).

    Returns:
        :class:`EpisodeResult` with terminal reliability, energy, wall-clock, trajectories
        and the runtime mechanism trace.
    """
    if mode == "dynamic_mc":
        raise NotImplementedError(
            "dynamic_mc mode is the independent G6 judge (src/validation/dynamic_mc.py); "
            "it samples subsets/fading/request-response per trial and must NOT reuse the "
            "analytic terminal marginals (spec §8). Use mode='analytic' here."
        )
    if mode != "analytic":
        raise ValueError("mode must be 'analytic' or 'dynamic_mc'")
    if evidence.num_nodes != scene.num_nodes:
        raise ValueError("evidence and scene must have the same N")

    k, alpha, beta, r_max = protocol_cfg.k, protocol_cfg.alpha, protocol_cfg.beta, protocol_cfg.r_max
    N = scene.num_nodes
    device = scene.positions.device

    # ---- graphs (G_comm, G_int) and round-invariant geometry ----
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    gi = build_interference_graph(scene.positions, scene.int_radius)
    geom_c = edge_geometry(gc, phy_cfg)
    geom_i = edge_geometry(gi, phy_cfg)
    resource_bucket = getattr(scene, "resource_bucket", None)   # NDH-SPS static bucket (None = SPS off)
    if resource_bucket is not None:
        from .sps_resource import assert_sps_pool_consistent
        assert_sps_pool_consistent(scene, phy_cfg)              # SPS bucket pool == physics resource_pool
    padding = build_bucketed_padding(gc.src_index, gc.dst_index, N)
    if bool(torch.any(padding.out_degree < k).cpu()):
        raise ValueError(
            "a source has G_comm out-degree < k; apply the §7.2 candidate-shortage "
            "protocol (k_i=min(k,|N_i|) or RSU fallback) before the episode (constraint #4)"
        )

    # ---- shared-latent scenario decomposition (the analytic Z) ----
    omega, init_cp = evidence.analytic_scenarios(max_scenarios=max_scenarios)  # [Q], [N, Q]
    omega = omega.to(device)
    init_cp = init_cp.to(device)
    Q = int(omega.numel())

    # ---- query policy -> scenario-independent inclusion pi + quorum query law ----
    # Two query laws share one interface (constraint #3, same policy train+deploy): the ESP
    # product law (diagonal kernel) and the CDQ low-rank k-DPP (G4/G5). The policy declares
    # which via ``query_law``; both feed the SAME canonical physics + protocol.
    query_law = getattr(query_policy, "query_law", "esp")
    if query_law == "cdq":
        from src.sampling.cdq_query import cdq_bucketed_quorum, cdq_edge_inclusion
        quality, diversity = query_policy.kernel(gc)             # [E], [E, r]  (no truth/vote)
        quality = quality.to(device)
        diversity = diversity.to(device)
        pi = cdq_edge_inclusion(gc.src_index, gc.dst_index, N, quality, diversity, k, padding=padding)
    elif query_law == "cdq2":
        from src.sampling.cdq2_wiring import cdq2_bucketed_quorum, cdq2_edge_inclusion
        quality, diversity = query_policy.kernel(gc)             # [E], [E, r]  (no truth/vote)
        quality = quality.to(device)
        diversity = diversity.to(device)
        eta = getattr(query_policy, "eta", 0.0)                  # CDQ 2.0 diversity strength (eta=0 => ESP)
        pi = cdq2_edge_inclusion(gc.src_index, gc.dst_index, N, quality, diversity, eta, k, padding=padding)
    else:
        from src.sampling.esp_query import edge_inclusion_probabilities
        log_weights = query_policy.log_weights(gc).to(device)    # [E]  (no truth/vote)
        pi = edge_inclusion_probabilities(gc.src_index, gc.dst_index, N, log_weights, k, padding=padding)
        a_edge = log_weights.unsqueeze(-1).expand(gc.num_edges, Q)   # [E, Q] quorum query weights

    if eligible_mask is None:
        eligible_mask = torch.ones(N, dtype=torch.bool, device=device)
    elig = eligible_mask.to(device=device, dtype=torch.bool)

    # ---- protocol state init [N, Q, S] ----
    layout = snowball_layout(beta, r_max)
    S = layout.state_count
    p = initial_distribution(init_cp.reshape(-1), layout, N * Q, dtype=init_cp.dtype, device=device)
    p = p.reshape(N, Q, S)

    energy = torch.zeros(Q, dtype=init_cp.dtype, device=device)
    cumulative_time = torch.zeros(Q, dtype=init_cp.dtype, device=device)
    round_durations: list[torch.Tensor] = []
    c_traj: list[torch.Tensor] = []
    w_traj: list[torch.Tensor] = []
    tau_traj: list[torch.Tensor] = []
    if return_trajectory:
        c0, w0, _ = terminal_outcomes(p, layout)
        c_traj.append(c0)
        w_traj.append(w0)

    abl = dict(disable_interference=disable_interference, disable_collision=disable_collision,
               disable_half_duplex=disable_half_duplex, disable_queueing=disable_queueing,
               link_override=link_override)

    # ---- the spec §7.2 per-round closed loop ----
    for _ in range(r_max):
        c_t, w_t, undec = terminal_outcomes(p, layout)              # [N, Q]
        u, v = readout_preference(p, layout)                        # [N, Q] answer when polled
        phys = round_physics(gc, gi, pi, undec, phy_cfg,
                             geom_comm=geom_c, geom_int=geom_i, resource_bucket=resource_bucket, **abl)
        if query_law == "cdq":
            h_plus, h_minus, h_zero = cdq_bucketed_quorum(
                padding, quality, diversity, phys.ell_poll, u, v, k, alpha)
        elif query_law == "cdq2":
            h_plus, h_minus, h_zero = cdq2_bucketed_quorum(
                padding, quality, diversity, eta, phys.ell_poll, u, v, k, alpha)
        else:
            h_plus, h_minus, h_zero = _bucketed_quorum(padding, a_edge, phys.ell_poll, u, v, k, alpha)
        p = apply_round(p, h_plus, h_minus, h_zero, layout)
        energy = energy + phys.energy.sum(dim=0)                    # [Q]
        rd = phys.tau.max(dim=0).values                            # [Q] slowest node sets round
        cumulative_time = cumulative_time + rd
        round_durations.append(rd)
        if return_trajectory:
            ct, wt, _ = terminal_outcomes(p, layout)
            c_traj.append(ct)
            w_traj.append(wt)
            tau_traj.append(phys.tau)

    c_ir, w_ir, undecided_ir = terminal_outcomes(p, layout)
    F_disagree, F_wrong, S_allcorrect = _mixture_reliability(c_ir, w_ir, undecided_ir, omega, elig)

    trace = {}
    if return_trace:
        # accurate correlated-evidence sentinel (constraint #13): use the model's own predicate so
        # correlation living in sensor/map common causes (overlapping model) is not missed by a
        # road-only p_region check.
        _hce = getattr(evidence, "has_correlated_evidence", None)
        any_region_bias = (bool(_hce()) if callable(_hce)
                           else bool((evidence.p_region > 0).any().cpu()))
        # The physical chain is BYPASSED under an ideal link_override (round_physics early-returns
        # a constant ell). So the physical-mechanism flags must be reported HONESTLY: they are on
        # the live path only when the full chain actually ran (constraint #9 — no flag may claim a
        # mechanism executed when it did not).
        full_phys = link_override is None
        # NDH-SPS runtime sentinel (plan §4 acceptance / Contract C5): proves the persistent
        # same-resource collision actually executed on the canonical path (not just in a test).
        sps_active = (resource_bucket is not None and phy_cfg.resource_collision_kappa > 0.0
                      and not disable_collision and full_phys)
        trace = {
            "protocol": "binary_snowball",
            "query_policy": getattr(query_policy, "name", type(query_policy).__name__),
            "query_law": query_law,
            "mode": mode,
            "num_nodes": N,
            "num_regions": evidence.num_regions,
            "num_scenarios": Q,
            "num_edges_comm": gc.num_edges,
            "num_edges_int": gi.num_edges,
            "cross_destination_interference": (gi.num_edges > gc.num_edges
                                               and not disable_interference and full_phys),
            "interference_graph": (not disable_interference) and full_phys,
            # memoryless 1/S collision runs only when SPS is NOT active; else the persistent model does
            "mode2_collision": (not disable_collision) and full_phys and not sps_active,
            "sps_persistence": sps_active,             # NDH-SPS persistent same-resource collision (spec §3.4)
            "resource_conflict_graph": sps_active,     # same-bucket G_int contention drives collision
            "half_duplex": (not disable_half_duplex) and full_phys,
            "queueing": (not disable_queueing) and full_phys,
            "request_response": True,
            "finite_harq": (phy_cfg.max_harq_attempts > 1) and full_phys,
            "fbl_dispersion": full_phys,
            # ---- plan §4 mandatory mechanism flags (the canonical path has them all) ----
            "parallel_unicast": True,                  # k parallel unicast request-response polls (§5.1)
            "poll_window_ms": phy_cfg.poll_window_s * 1e3,   # Δ_poll (P0-E)
            "source_destination_accounting": full_phys,  # explicit scatter_source/destination (P0-C)
            "collision_self_exclusion": full_phys,     # L_{j,-ij} = L_j - a_ij (P0-D)
            "dynamic_transient_load": full_phys,       # load follows the evolving active mass (§7.3.1)
            "correlated_evidence": any_region_bias and Q > 1,
            "tau_proxy": False,
            "policy_uses_truth_or_vote": False,
            "link_override": link_override,            # None on the headline path (full PHY)
            "full_physics": link_override is None,
            "k": k, "alpha": alpha, "beta": beta, "r_max": r_max,
            "snowball_states": S,
        }

    return EpisodeResult(
        F_disagree=F_disagree, F_wrong=F_wrong, S_allcorrect=S_allcorrect,
        c_ir=c_ir, w_ir=w_ir, undecided_ir=undecided_ir,
        scenario_weight=omega, energy=energy, cumulative_time=cumulative_time,
        round_duration=torch.stack(round_durations, dim=0) if round_durations else torch.zeros(0, Q),
        mechanism_trace=trace,
        c_trajectory=torch.stack(c_traj, dim=0) if return_trajectory else None,
        w_trajectory=torch.stack(w_traj, dim=0) if return_trajectory else None,
        tau_trajectory=torch.stack(tau_traj, dim=0) if (return_trajectory and tau_traj) else None,
    )
