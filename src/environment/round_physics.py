"""Round-coupled full physical chain (spec §7.2-§7.4 -- the canonical per-round physics).

One round of the closed loop ``X_t -> tau_t -> Pi -> Lambda_t -> gamma_t, ell_t``
(spec §7.2), driven by the current protocol state (via the per-node *active* mass) and the
query policy (via per-edge inclusion probabilities ``pi_ij``). Every step of spec §7.3 is
present and request/response are kept physically distinct (spec §7.4):

    ell_poll_ij(Delta_poll)
                = succ_req(M_win) (1-p_col_req_j)(1-p_HD_req_j)(1-p_queue_drop_j)
                · succ_resp(M_win)(1-p_col_resp_i)(1-p_HD_resp_i)               (Eq. 41 + §5.2 + §7.3.8)

where succ_req/resp(M_win) is the FBL/HARQ decode probability within the HARQ round-trip
budget ``M_win = clamp((Delta_poll/slot - queue_slots_j)/(req_slots+resp_slots), 0, M)`` that
fits in the FIXED polling window ``Delta_poll`` after the M/M/1 wait (P0-E poll-window
timeout, spec §5.1-5.2). ``Delta_poll -> inf`` recovers the no-timeout ``ell``; ``-> 0`` gives 0.

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

from .candidate_graph import RadiusGraph, scatter_destination, scatter_source

__all__ = ["RoundPhysicsConfig", "RoundPhysicsResult", "edge_geometry", "round_physics",
           "harq_success_at_budget", "harq_success_within_window_discrete"]


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
    # NDH-SPS (G-NDH-SPS-PERSISTENCE, spec §3.4): when > 0 AND a per-node resource_bucket is supplied
    # to round_physics, the memoryless 1/S Mode-2 collision is REPLACED by persistent same-resource
    # contention -- p_col = 1 - exp(-kappa*(L_{j,r}-a_i)_+) over same-bucket G_int neighbours. 0 = off
    # (unchanged memoryless physics). In cfg so config_hash binds it (train==eval physics, constraint #4).
    resource_collision_kappa: float = 0.0
    poll_window_s: float = 0.01         # Delta_poll: the fixed polling-epoch window (matches the
    #                                     ConsensusServiceProfile default 10 ms; spec §5.1-5.2)
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
        if self.resource_collision_kappa < 0:
            raise ValueError("resource_collision_kappa must be >= 0 (0 = SPS off)")
        if self.poll_window_s <= 0:
            raise ValueError("poll_window_s (Delta_poll) must be > 0")
        if self.harq_combining not in ("chase", "ir"):
            raise ValueError("harq_combining must be 'chase' or 'ir'")
        if self.fading not in ("rayleigh", "none"):
            raise ValueError("fading must be 'rayleigh' or 'none'")

    def config_hash(self) -> str:
        """Deterministic SHA-256 of ALL fixed PHY resources (ExperimentSpec physics_hash, plan §4).

        This is the train==eval physics fingerprint: any change to power/blocklength/HARQ/
        sub-channels/timing/path-loss alters it, so a checkpoint trained under one physics cannot
        be silently evaluated under another (the historical ideal/full mismatch guard).
        """
        import hashlib
        import json
        from dataclasses import asdict
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
    tau: torch.Tensor            # [N, B] per-poller (SOURCE) round duration (spec §5.4; no tau_proxy)
    energy: torch.Tensor         # [N, B] per-node total energy = request(source) + response(dest)
    energy_request: torch.Tensor  # [N, B] request TX energy charged to the SOURCE/poller (§5.4)
    energy_response: torch.Tensor  # [N, B] response TX energy charged to the responder/DESTINATION (§5.4)
    source_activity: torch.Tensor  # [N, B] A_i^req = sum_j a_ij = k u_i (source request activity, §5.4)
    load_request: torch.Tensor   # [N, B] request contenders near each receiver (G_int)
    load_response: torch.Tensor  # [N, B] response contenders near each receiver (G_int)
    receiver_load: torch.Tensor  # [N, B] expected valid requests arriving (Lambda, Eq. 33)
    p_collision_request: torch.Tensor  # [E_comm, B] per-edge request collision (self-excluded, §5.5)
    p_collision_response: torch.Tensor  # [E_comm, B] per-edge response collision (self-excluded, §5.5)
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


def _expected_attempts_chase(
    gamma_BE: torch.Tensor, n: float, bits: float, cfg: RoundPhysicsConfig,
    shadow_std: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Return ``(succ_M, E[attempts], succ_by_m)`` under <= M chase-combining HARQ attempts.

    ``succ_by_m[m-1]`` = success with up to ``m`` attempts (``m = 1..M``); ``succ_M`` is the
    last; ``E[attempts] = 1 + sum_{m=1}^{M-1} (1 - succ_m)`` -- the chase-consistent attempt
    count. Both fading- and shadow-averaged; differentiable; no Monte-Carlo. ``succ_by_m`` is
    reused by the poll-window budget (P0-E) to evaluate success at a fractional attempt budget.
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
    return succ_M, attempts, succ_by_m


def harq_success_within_window_discrete(
    succ_by_m: list[torch.Tensor], queue_delay_s: torch.Tensor, rt_time_s: float,
    poll_window_s: float, *, n_quad: int = 64,
) -> torch.Tensor:
    """INDEPENDENT discrete completion-time reference for the poll-window success (P0-F).

    The analytic ``_harq_success_at_budget`` soft-interpolates the decode probability at the
    MEAN-queue budget ``M_win = (Delta_poll - E[W])/rt``. This reference instead integrates the
    DISCRETE (floor) attempt-fitting over the random M/M/1 sojourn ``W ~ Exp(mean=queue_delay)``:

        ell_window = E_W[ S( floor((Delta_poll - W) / rt_time) ) ],   S(0)=0, S(m)=succ_by_m[m-1].

    It is a genuinely different computation (samples the queue-wait distribution + uses the
    discrete floor, not the mean + a smooth ramp), so agreement validates the analytic surrogate
    (spec §12: the MC/discrete model is the judge). Exponential inverse-CDF midpoint quadrature.
    """
    M = len(succ_by_m)
    S = torch.stack([torch.zeros_like(succ_by_m[0])] + succ_by_m, dim=0)   # [M+1, ...] S(0..M)
    qd = queue_delay_s.clamp_min(0.0)                                      # mean sojourn (0 => W=0)
    out = torch.zeros_like(succ_by_m[0])
    for i in range(n_quad):
        u = (i + 0.5) / n_quad
        W = -qd * math.log(1.0 - u)                                       # Exp(mean=qd) inverse-CDF
        budget = (poll_window_s - W) / rt_time_s
        m_idx = torch.floor(budget).clamp(0.0, float(M)).to(torch.long)   # DISCRETE floor
        s_w = torch.gather(S, 0, m_idx.unsqueeze(0)).squeeze(0)
        out = out + s_w / n_quad
    return out


def harq_success_at_budget(succ_by_m: list[torch.Tensor], budget: torch.Tensor) -> torch.Tensor:
    """Public alias of the analytic soft-budget HARQ success (P0-E); see :func:`_harq_success_at_budget`."""
    return _harq_success_at_budget(succ_by_m, budget)


def _harq_success_at_budget(succ_by_m: list[torch.Tensor], budget: torch.Tensor) -> torch.Tensor:
    """Decode probability within a SOFT, fractional HARQ-attempt budget (spec §5.2, P0-E).

    ``succ_by_m[m-1]`` is the decode probability with up to ``m`` attempts (``m=1..M``), with
    the convention ``S(0)=0`` (no attempts -> no decode). For a real ``budget`` in ``[0, M]``
    we linearly interpolate ``S`` between the bracketing integers, so the result is continuous
    and differentiable in ``budget`` (hence in the queue delay -> ``pi``). ``budget>=M``
    saturates to ``succ_M`` (the no-timeout value); ``budget=0`` gives 0.
    """
    M = len(succ_by_m)
    S = torch.stack([torch.zeros_like(succ_by_m[0])] + succ_by_m, dim=0)   # [M+1, ...] S(0..M)
    b = budget.clamp(0.0, float(M))
    m_lo = torch.floor(b).clamp(0.0, float(M - 1))
    frac = b - m_lo                                                        # in [0, 1], diff in b
    idx_lo = m_lo.to(torch.long).unsqueeze(0)
    idx_hi = (m_lo.to(torch.long) + 1).clamp(max=M).unsqueeze(0)
    S_lo = torch.gather(S, 0, idx_lo).squeeze(0)
    S_hi = torch.gather(S, 0, idx_hi).squeeze(0)
    return (1.0 - frac) * S_lo + frac * S_hi


def _sps_same_resource_collision(
    graph_int: RadiusGraph, tx: torch.Tensor, bucket: torch.Tensor, kappa: float,
    resource_node: torch.Tensor, receiver_node: torch.Tensor, self_node: torch.Tensor, N: int,
) -> torch.Tensor:
    """Persistent same-resource (SPS) collision per comm edge (NDH spec §3.4).

    Replaces the memoryless ``1/S`` Mode-2 collision. A transmission on edge ``(i->j)`` uses the
    *persistent* bucket ``r = bucket[resource_node]`` (the source's bucket for the request phase, the
    responder's for the response phase) and collides only with SAME-bucket contenders in the
    interference neighbourhood of the receiver:

        L_{recv, r} = sum_{u in N_int(recv)} tx_u * 1{bucket_u = r},
        p_col = 1 - exp(-kappa * (L_{recv, r} - tx_self)_+).

    The desired transmission excludes itself (``tx_self``), so a single active same-bucket transmitter
    has collision probability EXACTLY 0 (constraint #7, mirrors the memoryless self-exclusion). Static
    buckets are a faithful Phase-1 surrogate: the SPS reselection interval (~1 s) >> a consensus episode
    (~60-200 ms), so buckets are effectively frozen within one decision. Differentiable in ``tx``
    (buckets are constant / no grad); ``O(S_used * E_int)`` with ``S_used`` = distinct buckets present.

    Phase-1 deviation from spec §3.4: the spec weights each same-bucket contender by its resource age
    ``a_u(t)``; Phase 1 (static buckets, no temporal reselection) uses the **active transmit mass**
    ``tx_u`` as the contention weight and self-excludes ``tx_self`` (the resource-age weighting is
    deferred to the temporal Phase 2). ``kappa`` (``resource_collision_kappa``) absorbs the units.
    """
    Bp = tx.shape[1]
    E = receiver_node.shape[0]
    L = torch.zeros((E, Bp), dtype=tx.dtype, device=tx.device)
    edge_bucket = bucket[resource_node]                    # [E] resource bucket governing each edge
    for b in torch.unique(bucket).tolist():
        txb = tx * (bucket == b).to(tx.dtype).unsqueeze(-1)             # [N, B] tx of bucket-b nodes
        Lb = scatter_destination(graph_int, txb[graph_int.src_index], N)  # [N, B] same-bucket load at each node
        sel = edge_bucket == b
        if bool(sel.any()):
            L[sel] = Lb[receiver_node][sel]
    L_excl = (L - tx[self_node]).clamp_min(0.0)            # self-exclude the desired transmission
    return 1.0 - torch.exp(-float(kappa) * L_excl)


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
    resource_bucket: torch.Tensor | None = None,
    node_capacity: torch.Tensor | None = None,
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
        # request TX energy -> source (poller); response TX energy -> responded destination.
        pi_ov = inclusion_prob.unsqueeze(-1)                                  # [E_comm, 1]
        e_req_unit = cfg.tx_power_mw * cfg.slot_time_s * cfg.request_slots
        e_resp_unit = cfg.tx_power_mw * cfg.slot_time_s * cfg.response_slots
        a_req_edge = active[graph_comm.src_index] * pi_ov                     # [E_comm, B]
        energy_req = e_req_unit * scatter_source(graph_comm, a_req_edge, N)   # [N, B]
        energy_resp = e_resp_unit * scatter_destination(
            graph_comm, a_req_edge * float(link_override), N)                 # responders answer delivered reqs
        energy = energy_req + energy_resp
        source_activity = scatter_source(graph_comm, a_req_edge, N)           # A_i = k u_i
        z_n = active.new_zeros((N, Bn))
        z_e = active.new_zeros((E, Bn))
        return RoundPhysicsResult(
            ell_poll=ell, tau=tau, energy=energy,
            energy_request=energy_req, energy_response=energy_resp,
            source_activity=source_activity,
            load_request=z_n, load_response=z_n, receiver_load=z_n,
            p_collision_request=z_e, p_collision_response=z_e,
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
    # NDH-SPS: persistent same-resource collision replaces the memoryless 1/S when a per-node bucket
    # is supplied AND kappa > 0 (spec §3.4). SINR interference is unchanged -- only COLLISION persists.
    sps_on = resource_bucket is not None and cfg.resource_collision_kappa > 0.0
    if sps_on:
        if resource_bucket.ndim != 1 or resource_bucket.shape[0] != N:
            raise ValueError("resource_bucket must be a 1-D tensor with one entry per node (N)")
        if resource_bucket.dtype not in (torch.long, torch.int64, torch.int32):
            raise ValueError("resource_bucket must be an integer dtype (bucket ids)")
        bucket = resource_bucket.to(device=active.device)
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
    a_req_edge = req_tx[src_c] * pi                        # [E_comm, B] a_ij^req = u_i pi_ij (§5.4)
    recv_load = scatter_destination(graph_comm, a_req_edge, N)                     # [N, B] Lambda_j (§5.4)
    source_activity = scatter_source(graph_comm, a_req_edge, N)                    # [N, B] A_i = k u_i (§5.4)

    # ---- §7.3.4-5: request-phase interference + collision load over G_int ----
    req_tx_src_i = req_tx[src_i]                           # [E_int, B]
    if disable_interference:
        I_req_node = torch.zeros_like(active)
    else:
        I_req_node = scatter_destination(graph_int, (1.0 / S) * req_tx_src_i * rx_i, N)  # [N, B]
    load_req_node = scatter_destination(graph_int, req_tx_src_i, N)                # [N, B] L_j^req contenders

    I_req_at_j = I_req_node[dst_c]                         # [E_comm, B]
    own_req = (1.0 / S) * req_tx[src_c] * rx_c             # [E_comm, B] desired not self-interf
    interf_req = (I_req_at_j - own_req).clamp_min(0.0)
    gamma_req = rx_c / (noise + interf_req)                # [E_comm, B] (Eq. 34, request)

    # ---- §7.3.7: request FBL/HARQ (success at each attempt budget m=1..M) ----
    succ_req, attempts_req, succ_by_m_req = _expected_attempts_chase(
        gamma_req, cfg.request_blocklength, cfg.request_bits, cfg, shadow_c)

    # ---- §7.3.5-6 / §5.5: request collision (self-excluded) + half-duplex (receiver j) ----
    # P0-D: the desired transmission i->j must NOT count itself as a contender, so the collision
    # contender mass is L_{j,-ij} = L_j^req - a_ij^req (here the source's own presence req_tx[i]).
    # A single active transmission then has collision probability EXACTLY 0 (constraint #7).
    load_req_at_j = load_req_node[dst_c]                   # [E_comm, B] L_j^req
    load_req_excl = (load_req_at_j - req_tx[src_c]).clamp_min(0.0)  # L_{j,-ij}^req (self-excluded)
    if disable_collision:
        p_col_req = torch.zeros_like(gamma_req)
    elif sps_on:
        # request bucket = the SOURCE i's persistent bucket; contenders near receiver j (spec §3.4)
        p_col_req = _sps_same_resource_collision(
            graph_int, req_tx, bucket, cfg.resource_collision_kappa,
            resource_node=src_c, receiver_node=dst_c, self_node=src_c, N=N)
    else:
        base = 1.0 - 1.0 / S
        p_col_req = 1.0 - torch.pow(torch.as_tensor(base, dtype=active.dtype, device=active.device),
                                    load_req_excl)
    if disable_half_duplex:
        p_hd_req = torch.zeros_like(gamma_req)
    else:
        # receiver j busy transmitting its OWN request cannot receive i's request
        # (duty cycle = own transmissions spread over the W-slot window)
        p_hd_req = (req_tx[dst_c] / W).clamp(0.0, 1.0)

    # ---- §7.3.8: queueing at the receiver (M/M/1), driven by the ADDRESSED load Lambda_j ----
    # NDH heterogeneous capacity (spec §4.4): per-node service rate mu_j replaces the global scalar.
    # node_capacity=None recovers the scalar service_rate EXACTLY (homogeneous byte-identity).
    if node_capacity is not None:
        if node_capacity.ndim != 1 or node_capacity.shape[0] != N:
            raise ValueError("node_capacity must be a 1-D [N] tensor of per-node service rates")
        if not torch.isfinite(node_capacity).all() or bool((node_capacity <= 0).any()):
            raise ValueError("node_capacity must be finite and > 0 (per-node service rates)")
        mu = node_capacity.to(device=active.device, dtype=active.dtype).unsqueeze(-1)  # [N, 1]
        rho_node = recv_load / mu.clamp_min(1e-9)         # [N, B] = Lambda_j / mu_j
    else:
        rho_node = recv_load / cfg.service_rate           # [N, B] = Lambda_j / mu (global)
    rho_j = rho_node[dst_c]                               # [E_comm, B] at the receiver j
    if disable_queueing:
        p_queue_drop = torch.zeros_like(gamma_req)
        queue_delay_node = torch.zeros_like(active)
    else:
        p_queue_drop = (1.0 - 1.0 / rho_j.clamp_min(1e-12)).clamp(0.0, 1.0)  # frac dropped if rho>1
        queue_delay_node = cfg.slot_time_s * (rho_node / (1.0 - rho_node).clamp_min(1e-3)).clamp(0.0, 50.0)

    # ---- §5.2 poll-window budget: HARQ round-trips that fit in Delta_poll after the queue wait ----
    # The polling epoch is a FIXED window Delta_poll. After the M/M/1 wait at the receiver j, the
    # remaining time admits ``M_win`` HARQ round-trips (each = request+response slots); the leg
    # decode probabilities are evaluated at that soft, differentiable budget (P0-E). M_win -> M
    # recovers the no-timeout ell; M_win -> 0 forces ell -> 0. Differentiable in the queue delay
    # (hence in pi); the queue delay is the receiver j = dst_c's wait.
    W_poll_slots = cfg.poll_window_s / cfg.slot_time_s
    rt_slots = cfg.request_slots + cfg.response_slots
    queue_slots_at_j = queue_delay_node[dst_c] / cfg.slot_time_s              # [E_comm, B]
    M_win = ((W_poll_slots - queue_slots_at_j) / rt_slots).clamp(0.0, float(cfg.max_harq_attempts))
    succ_req_win = _harq_success_at_budget(succ_by_m_req, M_win)              # [E_comm, B]

    # ---- §7.4: request-leg delivery (FBL-within-window AND collision AND half-duplex AND not dropped) ----
    ell_request_leg = (succ_req_win * (1.0 - p_col_req) * (1.0 - p_hd_req) * (1.0 - p_queue_drop))

    # ---- §7.3 response activity: nodes answer only requests they actually RECEIVED ----
    # responders weight Lambda by the full request-leg delivery (better request delivery ->
    # more responders -> more response-phase congestion: the real hub-overload feedback,
    # spec §9.1).
    a_resp_edge = req_tx[src_c] * pi * ell_request_leg    # [E_comm, B] response triggered by delivered req
    response_tx = scatter_destination(graph_comm, a_resp_edge, N)  # [N, B] responder j's response activity

    # ---- response-phase interference + collision over G_int (different tx/rx set) ----
    resp_tx_src_i = response_tx[src_i]                     # [E_int, B]
    if disable_interference:
        I_resp_node = torch.zeros_like(active)
    else:
        I_resp_node = scatter_destination(graph_int, (1.0 / S) * resp_tx_src_i * rx_i, N)
    load_resp_node = scatter_destination(graph_int, resp_tx_src_i, N)

    # response j->i is received at i = src_c; responder is j = dst_c (signal rx_c, symmetric d)
    I_resp_at_i = I_resp_node[src_c]                       # [E_comm, B]
    own_resp = (1.0 / S) * response_tx[dst_c] * rx_c
    interf_resp = (I_resp_at_i - own_resp).clamp_min(0.0)
    gamma_resp = rx_c / (noise + interf_resp)              # [E_comm, B] (Eq. 34, response)
    succ_resp, attempts_resp, succ_by_m_resp = _expected_attempts_chase(
        gamma_resp, cfg.response_blocklength, cfg.response_bits, cfg, shadow_c)
    # same poll-window round-trip budget gates the response decode (P0-E, §5.2)
    succ_resp_win = _harq_success_at_budget(succ_by_m_resp, M_win)         # [E_comm, B]

    # P0-D response self-exclusion: the desired responder j must not count its own response as a
    # contender near i, so L_{i,-ji}^resp = L_i^resp - a_ji^resp (the responder's own presence).
    load_resp_at_i = load_resp_node[src_c]
    load_resp_excl = (load_resp_at_i - response_tx[dst_c]).clamp_min(0.0)
    if disable_collision:
        p_col_resp = torch.zeros_like(gamma_resp)
    elif sps_on:
        # response bucket = the RESPONDER j's persistent bucket; contenders near receiver i (spec §3.4)
        p_col_resp = _sps_same_resource_collision(
            graph_int, response_tx, bucket, cfg.resource_collision_kappa,
            resource_node=dst_c, receiver_node=src_c, self_node=dst_c, N=N)
    else:
        base = 1.0 - 1.0 / S
        p_col_resp = 1.0 - torch.pow(torch.as_tensor(base, dtype=active.dtype, device=active.device),
                                     load_resp_excl)
    if disable_half_duplex:
        p_hd_resp = torch.zeros_like(gamma_resp)
    else:
        p_hd_resp = (response_tx[src_c] / W).clamp(0.0, 1.0)  # i busy answering others

    # ---- §7.4 / Eq. 41 / §5.2: full poll success = request leg AND response leg, within Delta_poll ----
    ell_response_leg = succ_resp_win * (1.0 - p_col_resp) * (1.0 - p_hd_resp)
    ell_poll = (ell_request_leg * ell_response_leg).clamp(0.0, 1.0)

    # ---- §5.4 / §7.3.12: poller (SOURCE) round duration tau (load/quality-coupled; NO tau_proxy) ----
    # P0-C: epoch completion belongs to the polling SOURCE i. The per-edge service time of poll
    # i->j is aggregated to the source i (pi-weighted mean over i's k parallel polls), plus the
    # queue delay i incurs WAITING on its polled peers' receiver queues (a destination quantity
    # gathered back to the source).
    per_edge_time = pi * cfg.slot_time_s * (cfg.request_slots * attempts_req
                                            + cfg.response_slots * attempts_resp)  # [E_comm, B]
    sum_pi_src = scatter_source(graph_comm, pi.expand(-1, Bn), N).clamp_min(1e-9)  # [N,B] ~ k
    queue_delay_at_source = scatter_source(graph_comm, pi * queue_delay_node[dst_c], N) / sum_pi_src
    tau = scatter_source(graph_comm, per_edge_time, N) / sum_pi_src + queue_delay_at_source  # [N, B]

    # ---- §5.4 energy: request TX -> SOURCE/poller, response TX -> responder/DESTINATION ----
    p_mw = cfg.tx_power_mw
    e_req_unit = p_mw * cfg.slot_time_s * cfg.request_slots * (cfg.request_blocklength / cfg.response_blocklength)
    e_resp_unit = p_mw * cfg.slot_time_s * cfg.response_slots
    # request energy = (per-poll attempts) summed over i's outgoing polls, charged to source i.
    energy_request = e_req_unit * scatter_source(graph_comm, a_req_edge * attempts_req, N)   # [N, B]
    # response energy = (per-poll attempts) of delivered requests, charged to responder j (dest).
    energy_response = e_resp_unit * scatter_destination(graph_comm, a_resp_edge * attempts_resp, N)  # [N, B]
    energy = energy_request + energy_response              # [N, B]

    return RoundPhysicsResult(
        ell_poll=ell_poll, tau=tau, energy=energy,
        energy_request=energy_request, energy_response=energy_response,
        source_activity=source_activity,
        load_request=load_req_node, load_response=load_resp_node, receiver_load=recv_load,
        p_collision_request=p_col_req, p_collision_response=p_col_resp,
        gamma_request=gamma_req, gamma_response=gamma_resp,
        succ_request=succ_req, succ_response=succ_resp,
    )
