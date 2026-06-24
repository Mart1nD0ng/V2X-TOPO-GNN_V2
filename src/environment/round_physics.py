"""Round-coupled full physical chain (spec §7.2-§7.4 -- the canonical per-round physics).

One round of the closed loop ``X_t -> tau_t -> Pi -> Lambda_t -> gamma_t, ell_t``
(spec §7.2), driven by the current protocol state (via the per-node *active* mass) and the
query policy (via per-edge inclusion probabilities ``pi_ij``). Every step of spec §7.3 is
present and request/response are kept physically distinct (spec §7.4):

    ell_poll_ij = (1-p_col_req_j)(1-p_HD_req_j)(1-eps_req_ij)
                · (1-p_col_resp_i)(1-p_HD_resp_i)(1-eps_resp_ji)
                · (1-p_queue_drop_j)                                            (Eq. 41 + §7.3.8)

The two phases use DIFFERENT transmitter/receiver/interference sets:

* **request phase** -- every still-active source transmits its query; the receiver ``j``
  of edge ``i->j`` suffers ambient interference from *all* request transmitters near ``j``
  on ``G_int`` (not just those polling ``j`` -- the spec §7.1 cross-destination fix);
* **response phase** -- a polled peer ``j`` that received the request answers; the original
  source ``i`` (now the receiver of ``j->i``) suffers interference from *all* responders
  near ``i`` on ``G_int``. Response activity is the receiver load weighted by request
  success, so it is computed only *after* the request phase (no circular dependency).

The round duration ``tau_i`` (spec §5.4) is derived from the physics -- base request+
response slots scaled by the expected HARQ attempt count plus an M/M/1 queueing delay --
and is therefore load- and SINR-dependent. There is **no fixed ``tau_proxy``** anywhere
(non-negotiable constraint #7). Transmit power, blocklength, HARQ profile and sub-channel
pool are fixed headline resources (spec §7.5); only the query topology varies.

Everything is batched over a trailing dimension ``B`` (the analytic shared-latent scenarios
``Q``; the policy ``pi`` is scenario-independent -- it never sees the latent ``Z`` / truth,
constraint #10). All operations are ``O(E_comm + E_int)`` scatter/gather over the two
graphs -- no ``N x N`` tensor (constraint #11). Differentiable end-to-end in ``pi`` and the
geometry; the FBL/HARQ/path-loss primitives are the approved reusable assets from
``src.mainline.finite_blocklength`` (spec §2.1).

Exactness boundary. This is a *mean-field-per-scenario* analytic surrogate: loads,
interference and link successes are expectations over the active mass, not a sampled
realisation. It is the differentiable training/screening model; the independent dynamic MC
(G6) -- which samples subsets, fading and request/response per trial -- is the final judge
(spec §8.3). The ablation flags (`disable_*`) remove a mechanism for the G3 causal tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from src.mainline.finite_blocklength import PathLoss3GPP, averaged_link_success

from .candidate_graph import RadiusGraph

__all__ = ["RoundPhysicsConfig", "RoundPhysicsResult", "edge_geometry", "round_physics"]


@dataclass(frozen=True)
class RoundPhysicsConfig:
    """Fixed headline physical resources (spec §7.5). Only query topology varies."""

    fc_ghz: float = 5.9
    tx_power_dbm: float = 23.0
    noise_dbm: float = -95.0
    subchannels: float = 5.0            # frequency sub-channels per slot
    slots_per_window: float = 20.0      # time slots in the Mode-2 resource-selection window
    # blocklength (complex channel uses) and information bits, request vs response
    request_blocklength: float = 60.0
    response_blocklength: float = 600.0
    request_bits: float = 48.0
    response_bits: float = 300.0
    # HARQ / fading
    max_harq_attempts: int = 2
    harq_combining: str = "chase"
    fading: str = "rayleigh"
    use_shadow_fading: bool = True
    # timing / queueing
    slot_time_s: float = 1e-3          # one sidelink slot
    request_slots: float = 1.0          # base slots for the request phase
    response_slots: float = 1.0         # base slots for the response phase
    service_rate: float = 12.0          # M/M/1 receiver service rate (requests/round)
    # LOS proxy
    los_d0_m: float = 50.0
    pathloss: PathLoss3GPP = field(default_factory=PathLoss3GPP)

    def __post_init__(self) -> None:
        if self.subchannels < 1.0:
            raise ValueError("subchannels must be >= 1")
        if self.slots_per_window < 1.0:
            raise ValueError("slots_per_window must be >= 1")
        if self.max_harq_attempts < 1:
            raise ValueError("max_harq_attempts must be >= 1")
        if self.service_rate <= 0:
            raise ValueError("service_rate must be > 0")
        if self.harq_combining not in ("chase", "ir"):
            raise ValueError("harq_combining must be 'chase' or 'ir'")
        if self.fading not in ("rayleigh", "none"):
            raise ValueError("fading must be 'rayleigh' or 'none'")

    @property
    def resource_pool(self) -> float:
        """Mode-2 (subchannel x slot) resources in the selection window = collision pool."""
        return float(self.subchannels) * float(self.slots_per_window)

    @property
    def noise_mw(self) -> float:
        return 10.0 ** (self.noise_dbm / 10.0)

    @property
    def tx_power_mw(self) -> float:
        return 10.0 ** (self.tx_power_dbm / 10.0)


@dataclass(frozen=True)
class EdgeGeometry:
    """Geometry-derived, round-invariant per-edge quantities (computed once, reused)."""

    rx_power_mw: torch.Tensor   # [E] received power on the edge (linear mW)
    los_prob: torch.Tensor      # [E] LOS probability
    shadow_std_db: torch.Tensor  # [E] log-normal shadow std


@dataclass(frozen=True)
class RoundPhysicsResult:
    ell_poll: torch.Tensor       # [E_comm, B] per-edge poll success this round (Eq. 41)
    tau: torch.Tensor            # [N, B] per-node round duration (spec §5.4; no tau_proxy)
    energy: torch.Tensor         # [N, B] per-node energy spent this round
    load_request: torch.Tensor   # [N, B] request contenders near each receiver (G_int)
    load_response: torch.Tensor  # [N, B] response contenders near each receiver (G_int)
    receiver_load: torch.Tensor  # [N, B] expected valid requests arriving (Lambda, Eq. 33)
    gamma_request: torch.Tensor  # [E_comm, B]
    gamma_response: torch.Tensor  # [E_comm, B]
    succ_request: torch.Tensor   # [E_comm, B]
    succ_response: torch.Tensor  # [E_comm, B]


def _los_probability(distance: torch.Tensor, d0: float) -> torch.Tensor:
    return torch.clamp(d0 / distance.clamp_min(1.0), 0.0, 1.0)


def _rx_power_mw(distance: torch.Tensor, los: torch.Tensor, cfg: RoundPhysicsConfig) -> torch.Tensor:
    pl = cfg.pathloss
    d = distance.clamp_min(1.0)
    log_d = torch.log10(d)
    log_fc = math.log10(cfg.fc_ghz)
    pl_los = pl.los[0] + pl.los[1] * log_d + pl.los[2] * log_fc
    pl_nlos = pl.nlos[0] + pl.nlos[1] * log_d + pl.nlos[2] * log_fc
    pl_non = torch.maximum(pl_nlos, pl_los + pl.nlosv_extra_db)
    losc = los.clamp(0.0, 1.0)
    pl_db = losc * pl_los + (1.0 - losc) * pl_non
    rx_dbm = cfg.tx_power_dbm - pl_db
    return torch.pow(torch.as_tensor(10.0, dtype=rx_dbm.dtype, device=rx_dbm.device), rx_dbm / 10.0)


def edge_geometry(graph: RadiusGraph, cfg: RoundPhysicsConfig) -> EdgeGeometry:
    """Per-edge received power / LOS / shadow std (round-invariant; geometry is fixed)."""
    los = _los_probability(graph.distance, cfg.los_d0_m)
    rx = _rx_power_mw(graph.distance, los, cfg)
    if cfg.use_shadow_fading:
        shadow = los * cfg.pathloss.shadow_std_los_db + (1 - los) * cfg.pathloss.shadow_std_nlos_db
    else:
        shadow = torch.zeros_like(los)
    return EdgeGeometry(rx_power_mw=rx, los_prob=los, shadow_std_db=shadow)


def _scatter_dst(graph: RadiusGraph, edge_val_BE: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Scatter-add a ``[E, B]`` per-edge value to its destination -> ``[N, B]``."""
    B = edge_val_BE.shape[-1]
    out = edge_val_BE.new_zeros((num_nodes, B))
    return out.index_add(0, graph.dst_index, edge_val_BE)


def _expected_attempts_chase(
    gamma_BE: torch.Tensor, n: float, bits: float, cfg: RoundPhysicsConfig,
    shadow_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(succ_M, E[attempts])`` under <= M chase-combining HARQ attempts.

    ``succ_M`` = success with up to M attempts; ``E[attempts] = 1 + sum_{m=1}^{M-1}
    P(not decoded after m attempts)`` = ``1 + sum_{m=1}^{M-1} (1 - succ_m)`` -- the
    chase-consistent attempt count (the round ends when a poll decodes or hits M). Both
    fading- and shadow-averaged; differentiable; no Monte-Carlo.
    """
    M = cfg.max_harq_attempts
    succ_by_m = []
    for m in range(1, M + 1):
        s = averaged_link_success(
            gamma_BE, n, bits, max_harq_attempts=m, harq_combining=cfg.harq_combining,
            shadow_std_db=shadow_std, fading=cfg.fading,
        )
        succ_by_m.append(s)
    succ_M = succ_by_m[-1]
    attempts = torch.ones_like(succ_M)
    for m in range(1, M):  # m = 1 .. M-1
        attempts = attempts + (1.0 - succ_by_m[m - 1])
    return succ_M, attempts


def round_physics(
    graph_comm: RadiusGraph,
    graph_int: RadiusGraph,
    inclusion_prob: torch.Tensor,   # [E_comm] pi_ij (scenario-independent query policy)
    active: torch.Tensor,           # [N, B] P(node still polling) per scenario
    cfg: RoundPhysicsConfig,
    *,
    geom_comm: EdgeGeometry | None = None,
    geom_int: EdgeGeometry | None = None,
    disable_interference: bool = False,
    disable_collision: bool = False,
    disable_half_duplex: bool = False,
    disable_queueing: bool = False,
    link_override: float | None = None,
) -> RoundPhysicsResult:
    """One round of the full physical chain (spec §7.3, steps 1-9 + tau/energy 11-12).

    Args:
        graph_comm: communication candidate graph ``G_comm`` (intended polls).
        graph_int: interference graph ``G_int`` (``int_radius >= comm_radius``).
        inclusion_prob: ``[E_comm]`` per-edge inclusion probability ``pi_ij`` (from the
            query policy; sums to ``k`` per source). Scenario-independent (constraint #10).
        active: ``[N, B]`` per-node active (undecided) mass per scenario (the protocol
            state ``X_t`` enters only through this -- spec §7.3.1).
        cfg: fixed headline physical resources.
        geom_comm / geom_int: precomputed :class:`EdgeGeometry` (else computed here).
        disable_*: ablation switches removing a single mechanism (G3 causal tests).

    Returns:
        :class:`RoundPhysicsResult` with the per-edge poll success ``ell_poll`` (fed to the
        quorum DP), the per-node round duration ``tau`` and energy, and the load /
        interference / SINR / per-phase success diagnostics.
    """
    if active.ndim != 2:
        raise ValueError("active must be [N, B]")
    N, Bn = active.shape
    if N != graph_comm.num_nodes or N != graph_int.num_nodes:
        raise ValueError("graph node counts must match active's N")
    if inclusion_prob.shape[0] != graph_comm.num_edges:
        raise ValueError("inclusion_prob must have one entry per G_comm edge")

    # Ideal / fixed-link mode (spec §3.3 perfect-link feasibility floor; protocol-isolation
    # validation, G6). Bypasses the physical chain: every poll succeeds with a FIXED
    # probability and the round takes the base request+response slots. This is NOT the
    # headline path (constraint #7) -- the canonical episode records it in the trace and the
    # headline asserts link_override is None.
    if link_override is not None:
        if not (0.0 <= float(link_override) <= 1.0):
            raise ValueError("link_override must be a probability in [0, 1]")
        E = graph_comm.num_edges
        ell = active.new_full((E, Bn), float(link_override))
        base = cfg.slot_time_s * (cfg.request_slots + cfg.response_slots)
        tau = active.new_full((N, Bn), base)
        e_unit = cfg.tx_power_mw * base
        energy = active * e_unit
        z_n = active.new_zeros((N, Bn))
        z_e = active.new_zeros((E, Bn))
        return RoundPhysicsResult(
            ell_poll=ell, tau=tau, energy=energy,
            load_request=z_n, load_response=z_n, receiver_load=z_n,
            gamma_request=z_e, gamma_response=z_e, succ_request=ell, succ_response=ell,
        )

    gc = geom_comm or edge_geometry(graph_comm, cfg)
    gi = geom_int or edge_geometry(graph_int, cfg)
    # Co-channel interference and Mode-2 collision share the (subchannel x slot) resource
    # pool S_eff = subchannels * slots_per_window (two transmissions interfere/collide only
    # when they reuse the SAME resource, prob ~1/S_eff). Half-duplex is the separate
    # time-domain duty cycle over W = slots_per_window (a node cannot TX and RX in one slot).
    S = cfg.resource_pool
    W = float(cfg.slots_per_window)
    noise = cfg.noise_mw
    src_c, dst_c = graph_comm.src_index, graph_comm.dst_index
    src_i, dst_i = graph_int.src_index, graph_int.dst_index
    pi = inclusion_prob.unsqueeze(-1)                       # [E_comm, 1]
    rx_c = gc.rx_power_mw.unsqueeze(-1)                     # [E_comm, 1]
    rx_i = gi.rx_power_mw.unsqueeze(-1)                     # [E_int, 1]
    shadow_c = gc.shadow_std_db.unsqueeze(-1)              # [E_comm, 1]

    # ---- §7.3.1-2: active mass -> request transmit activity (each active source polls) ----
    req_tx = active                                        # [N, B]

    # Receiver load Lambda_j = sum_{i->j in G_comm} active_i pi_ij (Eq. 33): the requests
    # ADDRESSED to j (pi-weighted, over G_comm). This drives the receiver's M/M/1 SERVICE
    # queue -- distinct from the G_int co-channel contender mass below, which drives
    # interference/collision (a transmission to some other m still contends on the channel
    # near j but never enters j's service queue).
    recv_load = _scatter_dst(graph_comm, req_tx[src_c] * pi, N)                    # [N, B] Lambda

    # ---- §7.3.4-5: request-phase interference + collision load over G_int ----
    req_tx_src_i = req_tx[src_i]                           # [E_int, B]
    if disable_interference:
        I_req_node = torch.zeros_like(active)
    else:
        I_req_node = _scatter_dst(graph_int, (1.0 / S) * req_tx_src_i * rx_i, N)  # [N, B]
    load_req_node = _scatter_dst(graph_int, req_tx_src_i, N)                       # [N, B] contenders

    I_req_at_j = I_req_node[dst_c]                         # [E_comm, B]
    own_req = (1.0 / S) * req_tx[src_c] * rx_c             # [E_comm, B] desired not self-interf
    interf_req = (I_req_at_j - own_req).clamp_min(0.0)
    gamma_req = rx_c / (noise + interf_req)                # [E_comm, B] (Eq. 34, request)

    # ---- §7.3.7: request FBL/HARQ ----
    succ_req, attempts_req = _expected_attempts_chase(
        gamma_req, cfg.request_blocklength, cfg.request_bits, cfg, shadow_c)

    # ---- §7.3.5-6: request collision + half-duplex (receiver j) ----
    load_req_at_j = load_req_node[dst_c]                   # [E_comm, B]
    if disable_collision:
        p_col_req = torch.zeros_like(gamma_req)
    else:
        base = 1.0 - 1.0 / S
        p_col_req = 1.0 - torch.pow(torch.as_tensor(base, dtype=active.dtype, device=active.device),
                                    load_req_at_j.clamp_min(0.0))
    if disable_half_duplex:
        p_hd_req = torch.zeros_like(gamma_req)
    else:
        # receiver j busy transmitting its OWN request cannot receive i's request
        # (duty cycle = own transmissions spread over the W-slot window)
        p_hd_req = (req_tx[dst_c] / W).clamp(0.0, 1.0)

    # ---- §7.3.8: queueing at the receiver (M/M/1), driven by the ADDRESSED load Lambda_j ----
    rho_node = recv_load / cfg.service_rate               # [N, B] = Lambda_j / mu
    rho_j = rho_node[dst_c]                               # [E_comm, B] at the receiver j
    if disable_queueing:
        p_queue_drop = torch.zeros_like(gamma_req)
        queue_delay_node = torch.zeros_like(active)
    else:
        p_queue_drop = (1.0 - 1.0 / rho_j.clamp_min(1e-12)).clamp(0.0, 1.0)  # frac dropped if rho>1
        queue_delay_node = cfg.slot_time_s * (rho_node / (1.0 - rho_node).clamp_min(1e-3)).clamp(0.0, 50.0)

    # ---- §7.4: request-leg delivery (FBL AND collision AND half-duplex AND not dropped) ----
    ell_request_leg = (succ_req * (1.0 - p_col_req) * (1.0 - p_hd_req) * (1.0 - p_queue_drop))

    # ---- §7.3 response activity: nodes answer only requests they actually RECEIVED ----
    # responders weight Lambda by the full request-leg delivery (better request delivery ->
    # more responders -> more response-phase congestion: the real hub-overload feedback,
    # spec §9.1).
    response_tx = _scatter_dst(graph_comm, req_tx[src_c] * pi * ell_request_leg, N)  # [N, B]

    # ---- response-phase interference + collision over G_int (different tx/rx set) ----
    resp_tx_src_i = response_tx[src_i]                     # [E_int, B]
    if disable_interference:
        I_resp_node = torch.zeros_like(active)
    else:
        I_resp_node = _scatter_dst(graph_int, (1.0 / S) * resp_tx_src_i * rx_i, N)
    load_resp_node = _scatter_dst(graph_int, resp_tx_src_i, N)

    # response j->i is received at i = src_c; responder is j = dst_c (signal rx_c, symmetric d)
    I_resp_at_i = I_resp_node[src_c]                       # [E_comm, B]
    own_resp = (1.0 / S) * response_tx[dst_c] * rx_c
    interf_resp = (I_resp_at_i - own_resp).clamp_min(0.0)
    gamma_resp = rx_c / (noise + interf_resp)              # [E_comm, B] (Eq. 34, response)
    succ_resp, attempts_resp = _expected_attempts_chase(
        gamma_resp, cfg.response_blocklength, cfg.response_bits, cfg, shadow_c)

    load_resp_at_i = load_resp_node[src_c]
    if disable_collision:
        p_col_resp = torch.zeros_like(gamma_resp)
    else:
        base = 1.0 - 1.0 / S
        p_col_resp = 1.0 - torch.pow(torch.as_tensor(base, dtype=active.dtype, device=active.device),
                                     load_resp_at_i.clamp_min(0.0))
    if disable_half_duplex:
        p_hd_resp = torch.zeros_like(gamma_resp)
    else:
        p_hd_resp = (response_tx[src_c] / W).clamp(0.0, 1.0)  # i busy answering others

    # ---- §7.4 / Eq. 41: full poll success = request leg AND response leg ----
    ell_response_leg = succ_resp * (1.0 - p_col_resp) * (1.0 - p_hd_resp)
    ell_poll = (ell_request_leg * ell_response_leg).clamp(0.0, 1.0)

    # ---- §5.4 / §7.3.12: round duration tau (load- and quality-dependent; NO tau_proxy) ----
    # per-leg service time: request leg occupies request_slots per attempt, response leg
    # response_slots per attempt (the two legs are physically distinct, spec §7.4).
    per_edge_time = pi * cfg.slot_time_s * (cfg.request_slots * attempts_req
                                            + cfg.response_slots * attempts_resp)  # [E_comm, B]
    sum_pi = _scatter_dst(graph_comm, pi.expand(-1, Bn), N).clamp_min(1e-9)       # [N,B] ~ k
    tau = _scatter_dst(graph_comm, per_edge_time, N) / sum_pi + queue_delay_node  # [N, B]

    # ---- energy: per-node tx energy over its request + response transmissions (with HARQ) ----
    p_mw = cfg.tx_power_mw
    e_req_unit = p_mw * cfg.slot_time_s * cfg.request_slots * (cfg.request_blocklength / cfg.response_blocklength)
    e_resp_unit = p_mw * cfg.slot_time_s * cfg.response_slots
    energy_req = req_tx * e_req_unit * (_scatter_dst(graph_comm, pi * attempts_req, N) / sum_pi)
    energy_resp = response_tx * e_resp_unit * (
        _scatter_dst(graph_comm, pi * attempts_resp, N) / sum_pi)
    energy = energy_req + energy_resp                     # [N, B]

    return RoundPhysicsResult(
        ell_poll=ell_poll, tau=tau, energy=energy,
        load_request=load_req_node, load_response=load_resp_node, receiver_load=recv_load,
        gamma_request=gamma_req, gamma_response=gamma_resp,
        succ_request=succ_req, succ_response=succ_resp,
    )
