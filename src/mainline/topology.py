"""Physics-constrained adaptive topology (spec §7, §3.4 -- the H2 core).

The candidate graph is produced ONLY by physical reachability (communication radius,
geometry), built with spatial hashing / cell lists so that ``E = O(N)`` in expectation
at fixed density and radius -- with NO ``N x N`` dense tensor and, crucially, NO fixed
per-node degree cap and NO top-k truncation (H2).  Every physically-reachable candidate
edge is kept on the differentiable path.

Sparsity and hub avoidance are NOT imposed by a cap; they EMERGE from physical cost.
The receiver-load chain (spec §7.1, Eqs. 33-34) is

    Lambda_jr(t) = sum_{i: i->j} tau_ir(t) pi_ij                                      (Eq. 33)
    Lambda  ->  interference / Mode-2 collision / half-duplex / queueing  ->  gamma
            ->  ell  ->  (F, D, E)                                                    (Eq. 34)

so concentrating many high-weight queries onto one hub raises that receiver's load,
which raises its co-channel interference, Mode-2 collision and queueing utilisation,
which lowers the SINR and link reliability of every poll that touches the hub, which
raises the global failure ``F``.  The model is therefore penalised for hub overload by
a real physical mechanism, not by an artificial degree limit.

This module composes with the §4 inclusion probabilities (``pi``) and the §8
finite-blocklength link model (``ell``); it computes the load, interference and costs in
between.  All per-edge quantities are differentiable; the discrete candidate-edge set is
fixed geometry (no gradient through edge existence, by design).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from .finite_blocklength import (
    PathLoss3GPP,
    averaged_link_success,
    mode2_collision_probability,
    poll_success,
)

__all__ = [
    "CandidateGraph",
    "build_candidate_graph",
    "los_probability",
    "receiver_load",
    "aggregate_interference",
    "link_sinr_with_interference",
    "mode2_collision_from_load",
    "queueing_utilisation",
    "half_duplex_probability",
    "LoadCoupledLinkConfig",
    "load_coupled_link_reliability",
]


@dataclass(frozen=True)
class CandidateGraph:
    src_index: torch.Tensor  # [E]
    dst_index: torch.Tensor  # [E]
    distance: torch.Tensor   # [E] differentiable in positions
    num_nodes: int

    @property
    def num_edges(self) -> int:
        return int(self.src_index.numel())


def build_candidate_graph(
    positions: torch.Tensor,
    comm_radius: float,
    *,
    cell_size: float | None = None,
) -> CandidateGraph:
    """Radius candidate graph via spatial hashing -- ``E = O(N)``, no degree cap (H2).

    Directed edges ``i -> j`` for every distinct ``j`` within ``comm_radius`` of ``i``.
    Cell lists restrict each node's distance checks to its own and adjacent cells, so the
    cost is ``O(N)`` at fixed density (no ``N x N`` matrix is ever formed).  NO top-k, NO
    fixed neighbour count -- the full physical neighbourhood is kept.

    Args:
        positions: ``[N, D]`` node coordinates (D = 2 or 3).
        comm_radius: communication radius (same units as positions).
        cell_size: cell side (defaults to ``comm_radius``; must be >= comm_radius so the
            3^D neighbour cells fully cover the radius).
    """
    if positions.ndim != 2:
        raise ValueError("positions must be [N, D]")
    N, D = positions.shape
    r = float(comm_radius)
    if r <= 0:
        raise ValueError("comm_radius must be positive")
    cs = r if cell_size is None else float(cell_size)
    if cs < r:
        raise ValueError("cell_size must be >= comm_radius so neighbour cells cover the radius")
    pos_np = positions.detach().cpu().numpy()
    cells = (pos_np // cs).astype(int)
    cell_map: dict[tuple, list[int]] = defaultdict(list)
    for idx in range(N):
        cell_map[tuple(cells[idx])].append(idx)
    offsets = list(itertools.product((-1, 0, 1), repeat=D))
    r2 = r * r
    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(N):
        ci = tuple(cells[i])
        pi = pos_np[i]
        seen: set[int] = set()
        for off in offsets:
            nc = tuple(ci[d] + off[d] for d in range(D))
            for j in cell_map.get(nc, ()):  # only nearby cells -> O(1) per node at fixed density
                if j == i or j in seen:
                    continue
                seen.add(j)
                diff = pos_np[j] - pi
                if float((diff * diff).sum()) <= r2:
                    src_list.append(i)
                    dst_list.append(j)
    device = positions.device
    src = torch.tensor(src_list, dtype=torch.long, device=device)
    dst = torch.tensor(dst_list, dtype=torch.long, device=device)
    if src.numel() == 0:
        dist = positions.new_zeros((0,))
    else:
        diff = positions[src] - positions[dst]
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-12)  # differentiable in positions
    return CandidateGraph(src_index=src, dst_index=dst, distance=dist, num_nodes=N)


def los_probability(distance: torch.Tensor, *, d0: float = 50.0) -> torch.Tensor:
    """Smooth LOS probability decaying with distance (TR 37.885-style ``min(1, d0/d)``-ish).

    A differentiable, monotone-decreasing LOS proxy in [0, 1]; the exact 3GPP piecewise
    form can be substituted without changing the topology mechanics.
    """
    d = distance.clamp_min(1.0)
    return torch.clamp(d0 / d, 0.0, 1.0)


def _scatter_add(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = values.new_zeros((num_nodes,))
    return out.index_add(0, index, values)


def receiver_load(
    inclusion_prob: torch.Tensor,  # [E] pi_ij (from §4 / G2)
    transient_prob: torch.Tensor,  # [N] tau_i(t) (from §6 / G1 recurrence)
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Receiver load ``Lambda_j = sum_{i->j} tau_i pi_ij`` (Eq. 33).  Returns ``[N]``.

    The expected number of selected queries arriving at receiver ``j`` per round, summed
    over still-active sources.  ``O(E)`` scatter-add, differentiable in ``pi`` and ``tau``.
    """
    w = transient_prob[src_index] * inclusion_prob
    return _scatter_add(w, dst_index, num_nodes)


def aggregate_interference(
    rx_power_mw: torch.Tensor,     # [E] received power i->j (linear mW)
    weight: torch.Tensor,          # [E] tau_i pi_ij (expected activity on the edge)
    dst_index: torch.Tensor,
    num_nodes: int,
    subchannels: float,
) -> torch.Tensor:
    """Expected co-channel interference power at each receiver ``[N]`` (linear mW).

    ``I_j = (1/S) sum_{i->j} (tau_i pi_ij) rx_power_ij`` -- each contending transmitter
    collides on the same sub-channel with probability ``1/S`` (Mode-2 random selection),
    contributing its received power.  Higher load -> higher interference floor.
    """
    if subchannels < 1.0:
        raise ValueError("subchannels must be >= 1")
    contrib = (1.0 / float(subchannels)) * weight * rx_power_mw
    return _scatter_add(contrib, dst_index, num_nodes)


def link_sinr_with_interference(
    rx_power_mw: torch.Tensor,     # [E]
    weight: torch.Tensor,          # [E] tau_i pi_ij
    interference_mw: torch.Tensor,  # [N] from aggregate_interference
    dst_index: torch.Tensor,
    subchannels: float,
    noise_mw: float,
) -> torch.Tensor:
    """Per-edge linear SINR ``gamma_ij`` with load-driven co-channel interference (Eq. 34).

    ``gamma_ij = rx_ij / (noise + I_j - own_ij)`` where the edge's own contribution to
    ``I_j`` is removed (the desired signal is not its own interference).  Differentiable.
    """
    own = (1.0 / float(subchannels)) * weight * rx_power_mw
    interf = (interference_mw[dst_index] - own).clamp_min(0.0)
    return rx_power_mw / (noise_mw + interf)


def mode2_collision_from_load(load: torch.Tensor, subchannels: float) -> torch.Tensor:
    """Mode-2 collision probability from receiver load ``1 - (1 - 1/S)^Lambda`` (Eq. spec §7.1).

    Uses the continuous load ``Lambda`` as the expected number of contenders (the
    legacy form used the integer concurrent-tx count); differentiable in ``Lambda``.
    """
    if subchannels < 1.0:
        raise ValueError("subchannels must be >= 1")
    base = torch.as_tensor(1.0 - 1.0 / float(subchannels), dtype=load.dtype, device=load.device)
    return 1.0 - torch.pow(base, load.clamp_min(0.0))


def queueing_utilisation(load: torch.Tensor, service_rate: float) -> torch.Tensor:
    """M/M/1-style utilisation ``rho = Lambda / mu`` and a bounded queueing loss factor.

    Returns ``rho`` (can exceed 1 under overload).  A receiver whose load approaches /
    exceeds its service rate ``mu`` saturates -- the basis for queueing delay (D) and
    overload loss; feeds G6.  Differentiable in ``Lambda``.
    """
    if service_rate <= 0:
        raise ValueError("service_rate must be positive")
    return load / float(service_rate)


def half_duplex_probability(tx_activity: torch.Tensor, slots_per_round: float) -> torch.Tensor:
    """Half-duplex blocking probability = transmit duty cycle, in [0, 1].

    A node transmitting (polling) cannot receive in the same slot.  ``tx_activity`` is the
    node's expected number of own transmissions per round; spread over ``slots_per_round``
    transmission opportunities the probability it is transmitting in any given response
    slot is ``min(1, tx_activity / slots_per_round)``.  Differentiable, increasing in
    activity, bounded -- a busier node blocks more incoming polls (a real half-duplex
    cost), but it is a *duty cycle*, never the raw activity count.
    """
    if slots_per_round <= 0:
        raise ValueError("slots_per_round must be positive")
    return (tx_activity / float(slots_per_round)).clamp(0.0, 1.0)


@dataclass(frozen=True)
class LoadCoupledLinkConfig:
    """Parameters for the load-coupled physical link chain (no degree cap anywhere)."""

    fc_ghz: float = 5.9
    tx_power_dbm: float = 23.0
    noise_dbm: float = -95.0
    subchannels: float = 5.0
    slots_per_round: float = 10.0  # transmission opportunities per round (half-duplex duty)
    service_rate: float = 8.0
    response_blocklength: float = 600.0  # PSSCH complex channel uses (from BlocklengthSpec)
    request_blocklength: float = 60.0    # PSCCH/SCI complex channel uses
    response_bits: float = 300.0
    request_bits: float = 48.0
    max_harq_attempts: int = 2
    harq_combining: str = "chase"
    fading: str = "rayleigh"
    use_shadow_fading: bool = True
    pathloss: PathLoss3GPP = field(default_factory=PathLoss3GPP)


def _rx_power_mw(distance: torch.Tensor, los: torch.Tensor, cfg: LoadCoupledLinkConfig) -> torch.Tensor:
    import math
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


def load_coupled_link_reliability(
    graph: CandidateGraph,
    inclusion_prob: torch.Tensor,   # [E] pi_ij
    transient_prob: torch.Tensor,   # [N] tau_i
    los_prob: torch.Tensor,         # [E] LOS probability per edge
    cfg: LoadCoupledLinkConfig,
    *,
    disable_physical_cost: bool = False,
) -> dict:
    """Full physical chain ``pi, tau, geometry -> Lambda -> interference/collision -> gamma -> ell``.

    Returns per-edge poll reliability ``ell_poll`` (Eq. 41) and the intermediate load /
    interference / collision tensors.  ``disable_physical_cost=True`` is the ABLATION that
    removes interference + collision + half-duplex (keeps only the isolated FBL link), used
    to prove the hub-overload suppression is the *mechanism*, not a cap.
    """
    import math
    src, dst, N = graph.src_index, graph.dst_index, graph.num_nodes
    rx_mw = _rx_power_mw(graph.distance, los_prob, cfg)  # [E]
    weight = transient_prob[src] * inclusion_prob        # [E] tau_i pi_ij
    load = receiver_load(inclusion_prob, transient_prob, src, dst, N)  # [N]

    noise_mw = 10.0 ** (cfg.noise_dbm / 10.0)
    if disable_physical_cost:
        gamma = rx_mw / noise_mw
        p_col = torch.zeros_like(rx_mw)
        p_hd = torch.zeros_like(rx_mw)
    else:
        interf = aggregate_interference(rx_mw, weight, dst, N, cfg.subchannels)  # [N]
        gamma = link_sinr_with_interference(rx_mw, weight, interf, dst, cfg.subchannels, noise_mw)
        p_col = mode2_collision_from_load(load[dst], cfg.subchannels)  # [E] at the receiver
        # half-duplex: receiver j busy transmitting its own queries ~ its own out-activity
        tx_activity = _scatter_add(weight, src, N)  # expected own-tx per node
        p_hd = half_duplex_probability(tx_activity[dst], cfg.slots_per_round)

    shadow_std = (los_prob.clamp(0, 1) * cfg.pathloss.shadow_std_los_db
                  + (1 - los_prob.clamp(0, 1)) * cfg.pathloss.shadow_std_nlos_db
                  if cfg.use_shadow_fading and not disable_physical_cost
                  else torch.zeros_like(los_prob))
    succ_resp = averaged_link_success(
        gamma, cfg.response_blocklength, cfg.response_bits, max_harq_attempts=cfg.max_harq_attempts,
        harq_combining=cfg.harq_combining, shadow_std_db=shadow_std, fading=cfg.fading,
    )
    succ_req = averaged_link_success(
        gamma, cfg.request_blocklength, cfg.request_bits, max_harq_attempts=cfg.max_harq_attempts,
        harq_combining=cfg.harq_combining, shadow_std_db=shadow_std, fading=cfg.fading,
    )
    ell_poll = poll_success(
        p_collision=p_col, p_half_duplex=p_hd,
        eps_request=1.0 - succ_req, eps_response=1.0 - succ_resp,
    )
    return {
        "ell_poll": ell_poll,
        "load": load,
        "gamma": gamma,
        "p_collision": p_col,
        "p_half_duplex": p_hd,
        "rx_power_mw": rx_mw,
    }
