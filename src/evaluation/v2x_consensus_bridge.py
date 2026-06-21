from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from src.consensus import evaluate_graph_coupled_avalanche
from src.consensus.avalanche_closed_form import PROBABILITY_TOLERANCE
from src.v2x_env.channel_spec import LOGISTIC_PROBIT, resolve_pathloss_spec


@dataclass(frozen=True)
class V2XPhysicalConfig:
    carrier_frequency_ghz: float = 5.9
    tx_power_dbm: float = 23.0
    noise_dbm: float = -95.0
    interference_proxy_dbm: float = -78.0
    nlos_penalty_db: float = 12.0
    mcs_threshold_db: float = 8.0
    transition_width_db: float = 2.0
    finite_blocklength_reliability: bool = False
    payload_bits: float = 100.0
    resource_block_count: float = 12.0
    subcarrier_spacing_hz: float = 15_000.0
    single_hop_delay_s: float = 0.001
    # Load-aware interference (D/E enabler): when > 0, each receiver's interference
    # floor rises with the in-load that the SELECTED topology imposes on it
    # (sum of incoming topology_weight), differentiable through topology_weight, so
    # congestion lowers link success -> more avalanche rounds (delay) and energy.
    # 0.0 keeps the fixed-proxy behaviour (all existing behaviour unchanged).
    interference_density_coupling_db: float = 0.0
    interference_reference_load: float = 1.0
    # P0-1 single-source: the evaluator now honours pathloss_model the same way the candidate
    # graph does (src/v2x_env/channel_spec.py is the shared coefficient source). "legacy" (default)
    # is byte-identical to the historical inline single-slope formula; "tr37885" selects the
    # 3GPP TR 37.885 V2V models (LOS/NLOS soft-mix + shadow-variance BLER broadening), so one
    # config switch flips BOTH the candidate-graph feature and the evaluator. A parity contract
    # test (tests/evaluation/test_channel_parity.py) pins numpy == torch.
    pathloss_model: str = "legacy"
    scenario: str = "urban"
    nlosv_extra_db: float = 6.0
    nlosv_extra_std_db: float = 4.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "V2XPhysicalConfig":
        data = data or {}
        return cls(
            carrier_frequency_ghz=float(data.get("carrier_frequency_ghz", cls.carrier_frequency_ghz)),
            tx_power_dbm=float(data.get("tx_power_dbm", cls.tx_power_dbm)),
            noise_dbm=float(data.get("noise_dbm", cls.noise_dbm)),
            interference_proxy_dbm=float(data.get("interference_proxy_dbm", cls.interference_proxy_dbm)),
            nlos_penalty_db=float(data.get("nlos_penalty_db", cls.nlos_penalty_db)),
            mcs_threshold_db=float(data.get("mcs_threshold_db", cls.mcs_threshold_db)),
            transition_width_db=float(data.get("transition_width_db", cls.transition_width_db)),
            finite_blocklength_reliability=bool(
                data.get("finite_blocklength_reliability", cls.finite_blocklength_reliability)
            ),
            payload_bits=float(data.get("payload_bits", cls.payload_bits)),
            resource_block_count=float(data.get("resource_block_count", cls.resource_block_count)),
            subcarrier_spacing_hz=float(data.get("subcarrier_spacing_hz", cls.subcarrier_spacing_hz)),
            single_hop_delay_s=float(data.get("single_hop_delay_s", cls.single_hop_delay_s)),
            interference_density_coupling_db=float(
                data.get("interference_density_coupling_db", cls.interference_density_coupling_db)
            ),
            interference_reference_load=float(
                data.get("interference_reference_load", cls.interference_reference_load)
            ),
            pathloss_model=str(data.get("pathloss_model", cls.pathloss_model)),
            scenario=str(data.get("scenario", cls.scenario)),
            nlosv_extra_db=float(data.get("nlosv_extra_db", cls.nlosv_extra_db)),
            nlosv_extra_std_db=float(data.get("nlosv_extra_std_db", cls.nlosv_extra_std_db)),
        )


@dataclass(frozen=True)
class AvalancheBridgeConfig:
    k: int = 1
    alpha: int = 1
    beta: int = 2
    rounds: int = 5
    eps: float = 1e-6
    temperature: float = 1.0
    # SSMC quenched-disorder closed form (src/consensus/graph_coupled_avalanche.py): number of
    # persistent Gauss-Hermite disorder copies per node. 1 = mean-field (byte-identical legacy);
    # >1 captures the quenched neighbourhood persistence the mean-field erases (the 40x optimism).
    quenched_quadrature: int = 1
    # MACRO graph-level reliability (constraint c): when set, also report F_network_tail =
    # P[network fraction-correct < tau], the whole-graph joint failure tail (not a per-node average).
    network_tail_tau: float | None = None
    network_tail_var_floor: float = 1e-6
    allow_self_loops: bool = False
    allow_multi_edges: bool = False
    query_support_backend: str = "legacy"
    diagnostics_mode: str = "full"
    return_history: bool = False
    reliability_failure_target: float = 1e-5
    reliability_boundary_factor: float = 10.0
    reliability_soft_tail_tau: float = 0.01

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "AvalancheBridgeConfig":
        data = data or {}
        return cls(
            k=int(data.get("k", cls.k)),
            alpha=int(data.get("alpha", cls.alpha)),
            beta=int(data.get("beta", cls.beta)),
            rounds=int(data.get("rounds", cls.rounds)),
            eps=float(data.get("eps", cls.eps)),
            temperature=float(data.get("temperature", cls.temperature)),
            quenched_quadrature=int(data.get("quenched_quadrature", cls.quenched_quadrature)),
            network_tail_tau=(
                None if data.get("network_tail_tau", cls.network_tail_tau) is None
                else float(data.get("network_tail_tau"))
            ),
            network_tail_var_floor=float(data.get("network_tail_var_floor", cls.network_tail_var_floor)),
            allow_self_loops=bool(data.get("allow_self_loops", cls.allow_self_loops)),
            allow_multi_edges=bool(data.get("allow_multi_edges", cls.allow_multi_edges)),
            query_support_backend=str(data.get("query_support_backend", cls.query_support_backend)),
            diagnostics_mode=str(data.get("diagnostics_mode", cls.diagnostics_mode)),
            return_history=bool(data.get("return_history", cls.return_history)),
            reliability_failure_target=float(data.get("reliability_failure_target", cls.reliability_failure_target)),
            reliability_boundary_factor=float(data.get("reliability_boundary_factor", cls.reliability_boundary_factor)),
            reliability_soft_tail_tau=float(data.get("reliability_soft_tail_tau", cls.reliability_soft_tail_tau)),
        )


@dataclass(frozen=True)
class EnergyProxyConfig:
    packet_duration_s: float | None = None
    rx_power_watt: float = 0.3
    processing_power_watt: float = 0.05
    tx_power_watt: float | None = None
    # B-fix-2 (Track B, docs/COUPLING_AND_OPERATING_POINT_DESIGN.md): retransmission-aware
    # energy. When enabled, per-edge query energy is scaled by the expected number of
    # transmissions n_tx = 1/link_success (a differentiable ARQ proxy), so energy depends
    # on link quality -- which the selected topology controls via receiver in-load -- and
    # is no longer merely delay x a constant. Opt-in; default off is byte-identical to the
    # legacy proxy.
    retransmission_aware: bool = False
    retransmission_success_floor: float = 1e-3
    # Phase 3.1: HARQ retry limit for the energy proxy (mirror of delay.structural_ntx_cap so D and
    # E stay on the SAME physical n_tx when both caps are set). None (default) = uncapped legacy.
    retransmission_ntx_cap: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "EnergyProxyConfig":
        data = data or {}
        packet_value = data.get("packet_duration_s", cls.packet_duration_s)
        tx_value = data.get("tx_power_watt", cls.tx_power_watt)
        ntx_cap = data.get("retransmission_ntx_cap", cls.retransmission_ntx_cap)
        return cls(
            packet_duration_s=None if packet_value is None else float(packet_value),
            rx_power_watt=float(data.get("rx_power_watt", cls.rx_power_watt)),
            processing_power_watt=float(data.get("processing_power_watt", cls.processing_power_watt)),
            tx_power_watt=None if tx_value is None else float(tx_value),
            retransmission_aware=bool(data.get("retransmission_aware", cls.retransmission_aware)),
            retransmission_success_floor=float(
                data.get("retransmission_success_floor", cls.retransmission_success_floor)
            ),
            retransmission_ntx_cap=None if ntx_cap is None else float(ntx_cap),
        )


@dataclass(frozen=True)
class DelayProxyConfig:
    single_hop_delay_s: float | None = None  # default -> physical.single_hop_delay_s
    # Structural delay (docs/STRUCTURAL_DELAY_MODEL_DESIGN.md, D-fix-A): when enabled, the
    # per-hop time is scaled by expected ARQ transmissions n_tx = 1/link_success (the same
    # finite-blocklength link reliability the energy proxy uses), so the delay the loss
    # optimizes (effective rounds = expected_rounds x weighted_mean(n_tx)) becomes
    # topology-controllable. Unlike energy this does NOT multiply by k (the k queries of a
    # round happen in parallel -> round latency is the query-mass-weighted MEAN per-hop
    # delay, not a sum). Opt-in; default off is byte-identical to the legacy round-count D.
    structural_delay: bool = False
    structural_success_floor: float = 1e-3
    # P1-2 (D redesign): how the per-round latency reduces over a node's parallel query set.
    #   "mean" (default, byte-identical) = query-mass-weighted MEAN per-hop delay. This UNDER-states the
    #          round latency: parallel queries finish when the SLOWEST returns, not the average, so D is
    #          pinned near the protocol round floor and barely responds to topology (report defect F8).
    #   "max"  = differentiable soft-max (segment logsumexp) over the per-edge n_tx hop delays -> the
    #          SLOWEST query controls the round latency, giving D a real topology lever (pick faster links
    #          for the worst query). Temperature controls sharpness; lower = closer to the true max.
    # Only active together with structural_delay=True; default "mean" keeps the legacy D byte-identical.
    structural_delay_reduce: str = "mean"
    structural_delay_softmax_temperature: float = 0.25
    # Phase 3.1 (ms calibration): HARQ retransmission CAP. Real NR sidelink HARQ retries 3-5 times
    # then drops; the uncapped ARQ proxy n_tx = 1/success (up to 1/floor = 1000) inflates D far past
    # any physical latency budget. When set (e.g. 4.0), n_tx = min(1/success, cap), so D_seconds is
    # interpretable in milliseconds against a V2X budget. None (default) = uncapped legacy proxy.
    structural_ntx_cap: float | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "DelayProxyConfig":
        data = data or {}
        single_hop = data.get("single_hop_delay_s", cls.single_hop_delay_s)
        reduce_mode = str(data.get("structural_delay_reduce", cls.structural_delay_reduce))
        if reduce_mode not in {"mean", "max"}:
            raise ValueError("structural_delay_reduce must be 'mean' or 'max'")
        ntx_cap = data.get("structural_ntx_cap", cls.structural_ntx_cap)
        return cls(
            single_hop_delay_s=None if single_hop is None else float(single_hop),
            structural_delay=bool(data.get("structural_delay", cls.structural_delay)),
            structural_success_floor=float(
                data.get("structural_success_floor", cls.structural_success_floor)
            ),
            structural_delay_reduce=reduce_mode,
            structural_delay_softmax_temperature=float(
                data.get("structural_delay_softmax_temperature", cls.structural_delay_softmax_temperature)
            ),
            structural_ntx_cap=None if ntx_cap is None else float(ntx_cap),
        )


def _physical_config(config: V2XPhysicalConfig | Mapping[str, object] | None) -> V2XPhysicalConfig:
    if isinstance(config, V2XPhysicalConfig):
        return config
    return V2XPhysicalConfig.from_mapping(config)


def _avalanche_config(config: AvalancheBridgeConfig | Mapping[str, object] | None) -> AvalancheBridgeConfig:
    if isinstance(config, AvalancheBridgeConfig):
        return config
    return AvalancheBridgeConfig.from_mapping(config)


def _energy_config(config: EnergyProxyConfig | Mapping[str, object] | None) -> EnergyProxyConfig:
    if isinstance(config, EnergyProxyConfig):
        return config
    return EnergyProxyConfig.from_mapping(config)


def _delay_config(config: DelayProxyConfig | Mapping[str, object] | None) -> DelayProxyConfig:
    if isinstance(config, DelayProxyConfig):
        return config
    return DelayProxyConfig.from_mapping(config)


def _scalar_like(value: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, dtype=reference.dtype, device=reference.device)


def _dbm_to_mw(value_dbm: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = value_dbm if isinstance(value_dbm, torch.Tensor) else _scalar_like(float(value_dbm), reference)
    return torch.pow(_scalar_like(10.0, reference), value / _scalar_like(10.0, reference))


def _dbm_to_watt(value_dbm: torch.Tensor | float, reference: torch.Tensor) -> torch.Tensor:
    value = value_dbm if isinstance(value_dbm, torch.Tensor) else _scalar_like(float(value_dbm), reference)
    return torch.pow(_scalar_like(10.0, reference), (value - _scalar_like(30.0, reference)) / _scalar_like(10.0, reference))


def _q_function(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.special.erfc(value / torch.sqrt(value.new_tensor(2.0)))


def _require_floating_vector(name: str, value: torch.Tensor, expected_count: int, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    if result.numel() != expected_count:
        raise ValueError(f"{name} must have one value per edge")
    if not torch.is_floating_point(result):
        raise ValueError(f"{name} must use a floating-point dtype")
    if bool(torch.any(~torch.isfinite(result.detach())).cpu()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _require_node_probability(name: str, value: torch.Tensor, num_nodes: int, reference: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    result = value.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    if result.numel() != num_nodes:
        raise ValueError(f"{name} must contain num_nodes values")
    if bool(torch.any(~torch.isfinite(result.detach())).cpu()):
        raise ValueError(f"{name} must contain only finite values")
    if bool(torch.any(result.detach() < -PROBABILITY_TOLERANCE).cpu()) or bool(
        torch.any(result.detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()
    ):
        raise ValueError(f"{name} must be in [0, 1]")
    return torch.clamp(result, 0.0, 1.0)


def _validate_edge_inputs(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    topology_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative")
    if not isinstance(src_index, torch.Tensor) or not isinstance(dst_index, torch.Tensor):
        raise TypeError("src_index and dst_index must be torch.Tensor values")
    device = topology_weight.device
    src = src_index.to(device=device, dtype=torch.long).reshape(-1)
    dst = dst_index.to(device=device, dtype=torch.long).reshape(-1)
    if src.numel() != dst.numel():
        raise ValueError("src_index and dst_index must have the same edge count")
    if src.numel():
        if bool(torch.any(src < 0).cpu()) or bool(torch.any(src >= num_nodes).cpu()):
            raise ValueError("src_index contains node ids outside [0, num_nodes)")
        if bool(torch.any(dst < 0).cpu()) or bool(torch.any(dst >= num_nodes).cpu()):
            raise ValueError("dst_index contains node ids outside [0, num_nodes)")
    if not torch.is_floating_point(topology_weight):
        raise ValueError("topology_weight must use a floating-point dtype")
    weight = topology_weight.reshape(-1)
    if weight.numel() != src.numel():
        raise ValueError("topology_weight must have one value per edge")
    if bool(torch.any(~torch.isfinite(weight.detach())).cpu()):
        raise ValueError("topology_weight must contain only finite values")
    if bool(torch.any(weight.detach() < 0.0).cpu()):
        raise ValueError("topology_weight must be nonnegative")
    return src, dst, weight


def _channel_proxy(
    *,
    distance_m: torch.Tensor,
    los_flag: torch.Tensor,
    physical: V2XPhysicalConfig,
    link_success: torch.Tensor | None,
    interference_dbm: torch.Tensor | None = None,
    shadow_offset_db: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    reference = distance_m
    distance = distance_m.reshape(-1)
    if bool(torch.any(~torch.isfinite(distance.detach())).cpu()):
        raise ValueError("distance_m must contain only finite values")
    if bool(torch.any(distance.detach() < 0.0).cpu()):
        raise ValueError("distance_m must be nonnegative")
    los = los_flag.to(dtype=reference.dtype, device=reference.device).reshape(-1)
    if los.numel() != distance.numel():
        raise ValueError("los_flag must have one value per edge")
    if bool(torch.any(~torch.isfinite(los.detach())).cpu()):
        raise ValueError("los_flag must contain only finite values")

    safe_distance = torch.clamp(distance, min=1.0)
    frequency = _scalar_like(physical.carrier_frequency_ghz, reference)
    log_d = torch.log10(safe_distance)
    log_fc = torch.log10(frequency)
    # P0-1 single source: coefficients come from src/v2x_env/channel_spec.py, shared with the
    # numpy candidate-graph path. pathloss_model="legacy" reproduces the historical inline
    # single-slope formula byte-for-byte (nlos_mode="flat_penalty", shadow_var=0).
    spec = resolve_pathloss_spec(
        physical.pathloss_model,
        scenario=physical.scenario,
        nlos_penalty_db=physical.nlos_penalty_db,
        nlosv_extra_db=physical.nlosv_extra_db,
        nlosv_extra_std_db=physical.nlosv_extra_std_db,
    )
    pl_los = (
        _scalar_like(spec.los_intercept_db, reference)
        + _scalar_like(spec.los_log_distance_slope, reference) * log_d
        + _scalar_like(spec.los_log_frequency_slope, reference) * log_fc
    )
    if spec.nlos_mode == "flat_penalty":
        path_loss_db = pl_los + (_scalar_like(1.0, reference) - los) * _scalar_like(spec.nlos_flat_penalty_db, reference)
        # Legacy: no shadow folding -> the BLER transition is the raw width (exact, byte-identical).
        eff_transition = _scalar_like(physical.transition_width_db, reference)
    else:
        pl_nlos = (
            _scalar_like(spec.nlos_intercept_db, reference)
            + _scalar_like(spec.nlos_log_distance_slope, reference) * log_d
            + _scalar_like(spec.nlos_log_frequency_slope, reference) * log_fc
        )
        pl_non = torch.maximum(pl_nlos, pl_los + _scalar_like(spec.nlosv_extra_db, reference))
        path_loss_db = los * pl_los + (_scalar_like(1.0, reference) - los) * pl_non
        shadow_var = los * _scalar_like(spec.shadow_var_los_db2, reference) + (
            _scalar_like(1.0, reference) - los
        ) * _scalar_like(spec.shadow_var_nlos_db2, reference)
        # tr37885: shadow fading broadens the BLER transition (closed-form, no sampling).
        eff_transition = torch.sqrt(
            _scalar_like(physical.transition_width_db, reference) ** 2
            + shadow_var / _scalar_like(LOGISTIC_PROBIT, reference) ** 2
        )
    # P1-1.2 hidden AR(1) shadow fading: a per-edge additive dB offset that rides the
    # differentiable SINR path (NOT the link_success override). Default None -> byte-identical.
    if shadow_offset_db is not None:
        offset = shadow_offset_db.to(dtype=reference.dtype, device=reference.device).reshape(-1)
        if offset.numel() != distance.numel():
            raise ValueError("shadow_offset_db must have one value per edge")
        if bool(torch.any(~torch.isfinite(offset.detach())).cpu()):
            raise ValueError("shadow_offset_db must contain only finite values")
        path_loss_db = path_loss_db + offset
    received_power_dbm = _scalar_like(physical.tx_power_dbm, reference) - path_loss_db
    signal_mw = _dbm_to_mw(received_power_dbm, reference)
    if interference_dbm is None:
        interference_mw = _dbm_to_mw(physical.interference_proxy_dbm, reference)
    else:
        interference_mw = _dbm_to_mw(interference_dbm.to(dtype=reference.dtype, device=reference.device), reference)
    impairment_mw = _dbm_to_mw(physical.noise_dbm, reference) + interference_mw
    sinr_linear = torch.clamp(signal_mw / impairment_mw, min=torch.finfo(reference.dtype).tiny)
    sinr_db = _scalar_like(10.0, reference) * torch.log10(sinr_linear)
    computed_success = torch.sigmoid(
        (sinr_db - _scalar_like(physical.mcs_threshold_db, reference))
        / torch.clamp(eff_transition, min=1e-6)
    )
    packet_error = torch.clamp(1.0 - computed_success, 0.0, 1.0)
    blocklength = _scalar_like(
        physical.resource_block_count * physical.subcarrier_spacing_hz * physical.single_hop_delay_s,
        reference,
    )
    blocklength = torch.clamp(blocklength, min=1.0)
    if physical.finite_blocklength_reliability:
        capacity_nats = torch.log1p(sinr_linear)
        payload_nats = _scalar_like(physical.payload_bits, reference) * torch.log(_scalar_like(2.0, reference))
        remainder_nats = 0.5 * torch.log(blocklength)
        q_argument = (
            blocklength * capacity_nats
            - payload_nats
            + remainder_nats
        ) / torch.sqrt(blocklength)
        packet_error = torch.clamp(_q_function(q_argument), 0.0, 1.0)
        computed_success = torch.clamp(1.0 - packet_error, 0.0, 1.0)
    if link_success is None:
        success = computed_success
    else:
        success = link_success.to(dtype=reference.dtype, device=reference.device).reshape(-1)
        if success.numel() != distance.numel():
            raise ValueError("link_success must have one value per edge")
        if bool(torch.any(~torch.isfinite(success.detach())).cpu()):
            raise ValueError("link_success must contain only finite values")
        if bool(torch.any(success.detach() < -PROBABILITY_TOLERANCE).cpu()) or bool(
            torch.any(success.detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()
        ):
            raise ValueError("link_success must be in [0, 1]")
        success = torch.clamp(success, 0.0, 1.0)
    return {
        "path_loss_db": path_loss_db,
        "received_power_dbm": received_power_dbm,
        "sinr_db": sinr_db,
        "link_success": success,
        "computed_link_success": computed_success,
        "packet_error_probability": packet_error,
        "finite_blocklength_blocklength": blocklength.expand_as(distance),
    }


def _empty_metric(reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(0.0)


def _quantile_or_zero(values: torch.Tensor, q: float, reference: torch.Tensor) -> torch.Tensor:
    return torch.quantile(values, q) if values.numel() else _empty_metric(reference)


def _fraction(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.numel() == 0:
        return _empty_metric(reference)
    return mask.to(dtype=reference.dtype).mean()


def _classify_failure(value: torch.Tensor, *, target: float, factor: float) -> str:
    failure_value = float(value.detach().cpu())
    lower = target / factor
    upper = target * factor
    if failure_value > upper:
        return "below_target"
    if failure_value < lower:
        return "above_target_high_reliability"
    return "near_target_boundary"


def _failure_metrics(
    *,
    correct: torch.Tensor,
    wrong: torch.Tensor,
    undecided: torch.Tensor,
    expected_rounds: torch.Tensor,
    node_energy: torch.Tensor,
    avalanche: AvalancheBridgeConfig,
    reference: torch.Tensor,
) -> dict[str, torch.Tensor | str | bool]:
    failure = torch.clamp(wrong + undecided, 0.0, 1.0)
    eps_value = max(float(avalanche.eps), 1e-12)
    nines = -torch.log10(failure + _scalar_like(eps_value, reference))
    mean_failure = failure.mean() if failure.numel() else _empty_metric(reference)
    p90_failure = _quantile_or_zero(failure, 0.90, reference)
    max_failure = failure.max() if failure.numel() else _empty_metric(reference)
    target = float(avalanche.reliability_failure_target)
    factor = float(avalanche.reliability_boundary_factor)
    if target <= 0.0:
        raise ValueError("reliability_failure_target must be positive")
    if factor <= 1.0:
        raise ValueError("reliability_boundary_factor must be greater than one")
    soft_tail_tau = float(avalanche.reliability_soft_tail_tau)
    if soft_tail_tau <= 0.0:
        raise ValueError("reliability_soft_tail_tau must be positive")
    tau = _scalar_like(soft_tail_tau, reference)
    if failure.numel():
        soft_tail_failure = tau * (torch.logsumexp(failure / tau, dim=0) - torch.log(_scalar_like(float(failure.numel()), reference)))
    else:
        soft_tail_failure = _empty_metric(reference)
    mean_regime = _classify_failure(mean_failure, target=target, factor=factor)
    tail_regime = _classify_failure(max_failure, target=target, factor=factor)
    max_regime = tail_regime
    mean_should_weaken = mean_regime == "above_target_high_reliability"
    tail_should_weaken = tail_regime == "above_target_high_reliability"
    global_should_weaken = mean_should_weaken and tail_should_weaken
    target_band = (failure >= target / factor) & (failure <= target * factor)
    wide_failure_band = (failure >= target / (factor * factor)) & (failure <= target * factor * factor)
    timeout_threshold = max(float(avalanche.rounds) * 0.95, 0.0)
    return {
        "node_failure_probability": failure,
        "node_reliability_nines": nines,
        "F_avalanche_node_mean": mean_failure,
        "F_avalanche_node_p90": p90_failure,
        "F_avalanche_node_max": max_failure,
        "F_avalanche_node_softmax_tail": soft_tail_failure,
        "reliability_nines_mean": nines.mean() if nines.numel() else _empty_metric(reference),
        "reliability_nines_p10": _quantile_or_zero(nines, 0.10, reference),
        "reliability_nines_min": nines.min() if nines.numel() else _empty_metric(reference),
        "reliability_nines_soft_tail": -torch.log10(soft_tail_failure + _scalar_like(eps_value, reference)),
        "mean_reliability_regime": mean_regime,
        "reliability_regime": mean_regime,
        "reliability_status": mean_regime,
        "tail_reliability_regime": tail_regime,
        "max_reliability_regime": max_regime,
        "tail_below_target": tail_regime == "below_target",
        "tail_near_boundary": tail_regime == "near_target_boundary",
        "tail_above_target": tail_regime == "above_target_high_reliability",
        "mean_reliability_loss_should_weaken": mean_should_weaken,
        "tail_reliability_loss_should_weaken": tail_should_weaken,
        "global_reliability_loss_should_weaken": global_should_weaken,
        "reliability_loss_should_weaken": global_should_weaken,
        "optimize_delay_energy_next": global_should_weaken,
        "c_flat_high_fraction": _fraction(correct > 0.999, reference),
        "c_flat_low_fraction": _fraction(correct < 1e-3, reference),
        "c_mid_band_fraction": _fraction((correct > 0.05) & (correct < 0.95), reference),
        "d_timeout_fraction": _fraction((undecided > 0.95) | (expected_rounds >= timeout_threshold), reference),
        "energy_nonzero_fraction": _fraction(node_energy > 0.0, reference),
        "failure_mid_band_fraction": _fraction(wide_failure_band, reference),
        "reliability_target_band_fraction": _fraction(target_band, reference),
    }


def _energy_proxy(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    normalized_query_weight: torch.Tensor,
    node_expected_rounds: torch.Tensor,
    physical: V2XPhysicalConfig,
    avalanche: AvalancheBridgeConfig,
    energy: EnergyProxyConfig,
    link_success: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    reference = node_expected_rounds
    tx_power_watt = (
        _scalar_like(energy.tx_power_watt, reference)
        if energy.tx_power_watt is not None
        else _dbm_to_watt(physical.tx_power_dbm, reference)
    )
    packet_duration_s = (
        float(energy.packet_duration_s)
        if energy.packet_duration_s is not None
        else float(physical.single_hop_delay_s)
    )
    per_attempt_energy = _scalar_like(packet_duration_s, reference) * (
        tx_power_watt + _scalar_like(energy.rx_power_watt, reference) + _scalar_like(energy.processing_power_watt, reference)
    )
    # B-fix-2: retransmission-aware per-edge energy. n_tx = 1/link_success (ARQ proxy),
    # differentiable through link_success -> SINR -> receiver in-load -> topology_weight,
    # giving E a lever independent of the avalanche round count. Default off (link_success
    # ignored) -> per-edge energy is the constant per-attempt energy (legacy, byte-identical).
    retransmission_active = bool(energy.retransmission_aware and link_success is not None)
    if retransmission_active:
        success = link_success.to(dtype=reference.dtype, device=reference.device).reshape(-1)
        if success.numel() != normalized_query_weight.numel():
            raise ValueError("link_success must align with normalized_query_weight (one value per edge)")
        floor = max(float(energy.retransmission_success_floor), float(torch.finfo(reference.dtype).tiny))
        n_tx = 1.0 / torch.clamp(success, min=floor)
        if energy.retransmission_ntx_cap is not None:
            n_tx = torch.clamp(n_tx, max=float(energy.retransmission_ntx_cap))  # HARQ retry limit
        per_edge_query_energy = per_attempt_energy * n_tx
    else:
        per_edge_query_energy = per_attempt_energy.expand_as(normalized_query_weight)
    edge_energy = normalized_query_weight * per_edge_query_energy
    per_round_energy = reference.new_zeros((num_nodes,)).index_add(0, src_index, edge_energy)
    per_round_energy = per_round_energy * float(avalanche.k)
    node_energy = node_expected_rounds * per_round_energy
    return {
        "per_edge_query_energy_joule": per_edge_query_energy,
        "node_expected_query_energy_per_round": per_round_energy,
        "node_consensus_energy": node_energy,
        "node_consensus_energy_j": node_energy,
        "E_consensus_node_mean": node_energy.mean() if num_nodes else _empty_metric(reference),
        "E_consensus_node_p90": _quantile_or_zero(node_energy, 0.90, reference),
        "E_consensus_total": node_energy.sum(),
        "energy_retransmission_aware": reference.new_tensor(1.0 if retransmission_active else 0.0),
    }


def _delay_proxy(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    normalized_query_weight: torch.Tensor,
    node_expected_rounds: torch.Tensor,
    physical: V2XPhysicalConfig,
    delay: DelayProxyConfig,
    link_success: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    # D-fix-A (docs/STRUCTURAL_DELAY_MODEL_DESIGN.md). Mirror of _energy_proxy but for time,
    # and WITHOUT the x k factor (parallel queries -> the round latency is the query-mass-
    # weighted MEAN per-hop delay, not a sum). When structural_delay is on, per-hop time is
    # scaled by n_tx = 1/link_success so the effective rounds depend on link quality, which
    # the selected topology controls. Default off -> per_round_delay is a constant per hop,
    # so node_effective_rounds == node_expected_rounds (byte-identical to the legacy D).
    reference = node_expected_rounds
    single_hop_delay_s = (
        float(delay.single_hop_delay_s)
        if delay.single_hop_delay_s is not None
        else float(physical.single_hop_delay_s)
    )
    base = _scalar_like(single_hop_delay_s, reference)
    structural_active = bool(delay.structural_delay and link_success is not None)
    if structural_active:
        success = link_success.to(dtype=reference.dtype, device=reference.device).reshape(-1)
        if success.numel() != normalized_query_weight.numel():
            raise ValueError("link_success must align with normalized_query_weight (one value per edge)")
        floor = max(float(delay.structural_success_floor), float(torch.finfo(reference.dtype).tiny))
        n_tx = 1.0 / torch.clamp(success, min=floor)
        if delay.structural_ntx_cap is not None:
            n_tx = torch.clamp(n_tx, max=float(delay.structural_ntx_cap))  # HARQ retry limit
        per_edge_hop_delay = base * n_tx
    else:
        per_edge_hop_delay = base.expand_as(normalized_query_weight)
    edge_delay = normalized_query_weight * per_edge_hop_delay
    reduce_mode = delay.structural_delay_reduce if structural_active else "mean"
    if reduce_mode == "max":
        # P1-2: round latency = the SLOWEST parallel query (differentiable soft-max). Query-mass-weighted
        # so non-queried edges are down-weighted, with a detached global shift for exp stability. The
        # weighted MEAN is the T->inf limit of this soft-max, so D_max >= D_mean always.
        temperature = max(float(delay.structural_delay_softmax_temperature), 1e-6)
        if per_edge_hop_delay.numel():
            shift = per_edge_hop_delay.detach().max()
        else:
            shift = _scalar_like(0.0, reference)
        softmax_w = normalized_query_weight * torch.exp((per_edge_hop_delay - shift) / temperature)
        num = reference.new_zeros((num_nodes,)).index_add(0, src_index, softmax_w * per_edge_hop_delay)
        den = reference.new_zeros((num_nodes,)).index_add(0, src_index, softmax_w)
        node_per_round_delay = num / torch.clamp(den, min=torch.finfo(reference.dtype).tiny)
    else:
        # PARALLEL round latency: query-mass-weighted MEAN per-hop delay (NO x k, unlike energy).
        node_per_round_delay = reference.new_zeros((num_nodes,)).index_add(0, src_index, edge_delay)
    node_structural_delay_seconds = node_expected_rounds * node_per_round_delay
    # Effective rounds = latency / single-hop time = expected_rounds x reduce(n_tx).
    node_effective_rounds = node_structural_delay_seconds / base
    return {
        "node_effective_rounds": node_effective_rounds,
        "node_structural_delay_seconds": node_structural_delay_seconds,
        "delay_structural_active": reference.new_tensor(1.0 if structural_active else 0.0),
        "delay_structural_reduce_mode": reduce_mode,
    }


def _diagnostics(
    *,
    avalanche_result: Mapping[str, Any],
    energy_result: Mapping[str, torch.Tensor],
    channel_result: Mapping[str, torch.Tensor],
    num_nodes: int,
    reference: torch.Tensor,
    failure_metrics: Mapping[str, torch.Tensor | str | bool],
) -> dict[str, torch.Tensor | str | bool]:
    correct = avalanche_result["node_p_correct_decision"]
    expected = avalanche_result["node_expected_rounds"]
    energy = energy_result["node_consensus_energy"]
    undecided = avalanche_result["node_p_undecided"]
    saturated_correct_count = (correct > 0.999).to(dtype=reference.dtype).sum()
    undecided_timeout_count = (undecided > 0.999).to(dtype=reference.dtype).sum()
    if "h_plus_history" in avalanche_result:
        h_plus_history = avalanche_result["h_plus_history"]
        h_min = h_plus_history.min() if h_plus_history.numel() else _empty_metric(reference)
        h_max = h_plus_history.max() if h_plus_history.numel() else _empty_metric(reference)
    else:
        h_min = _empty_metric(reference)
        h_max = _empty_metric(reference)
    flat_count = saturated_correct_count + undecided_timeout_count
    flat_region_warning = bool(num_nodes > 0 and float((flat_count / float(num_nodes)).detach().cpu()) >= 0.8)
    link_success = channel_result["link_success"]
    sinr_values = channel_result["sinr_db"]
    return {
        "h_plus_min": h_min,
        "h_plus_max": h_max,
        "node_C_min": correct.min() if num_nodes else _empty_metric(reference),
        "node_C_max": correct.max() if num_nodes else _empty_metric(reference),
        "expected_rounds_min": expected.min() if num_nodes else _empty_metric(reference),
        "expected_rounds_max": expected.max() if num_nodes else _empty_metric(reference),
        "energy_min": energy.min() if num_nodes else _empty_metric(reference),
        "energy_max": energy.max() if num_nodes else _empty_metric(reference),
        "saturated_correct_count": saturated_correct_count,
        "undecided_timeout_count": undecided_timeout_count,
        "flat_region_warning": flat_region_warning,
        "link_success_min": link_success.min() if link_success.numel() else _empty_metric(reference),
        "link_success_mean": link_success.mean() if link_success.numel() else _empty_metric(reference),
        "link_success_max": link_success.max() if link_success.numel() else _empty_metric(reference),
        "sinr_min_db": sinr_values.min() if sinr_values.numel() else _empty_metric(reference),
        "sinr_mean_db": sinr_values.mean() if sinr_values.numel() else _empty_metric(reference),
        "sinr_max_db": sinr_values.max() if sinr_values.numel() else _empty_metric(reference),
        "mean_reliability_regime": failure_metrics["mean_reliability_regime"],
        "reliability_regime": failure_metrics["reliability_regime"],
        "reliability_status": failure_metrics["reliability_status"],
        "tail_reliability_regime": failure_metrics["tail_reliability_regime"],
        "max_reliability_regime": failure_metrics["max_reliability_regime"],
        "tail_below_target": failure_metrics["tail_below_target"],
        "tail_near_boundary": failure_metrics["tail_near_boundary"],
        "tail_above_target": failure_metrics["tail_above_target"],
        "mean_reliability_loss_should_weaken": failure_metrics["mean_reliability_loss_should_weaken"],
        "tail_reliability_loss_should_weaken": failure_metrics["tail_reliability_loss_should_weaken"],
        "global_reliability_loss_should_weaken": failure_metrics["global_reliability_loss_should_weaken"],
        "reliability_loss_should_weaken": failure_metrics["reliability_loss_should_weaken"],
        "optimize_delay_energy_next": failure_metrics["optimize_delay_energy_next"],
        "c_flat_high_fraction": failure_metrics["c_flat_high_fraction"],
        "c_flat_low_fraction": failure_metrics["c_flat_low_fraction"],
        "c_mid_band_fraction": failure_metrics["c_mid_band_fraction"],
        "d_timeout_fraction": failure_metrics["d_timeout_fraction"],
        "energy_nonzero_fraction": failure_metrics["energy_nonzero_fraction"],
        "failure_mid_band_fraction": failure_metrics["failure_mid_band_fraction"],
        "reliability_target_band_fraction": failure_metrics["reliability_target_band_fraction"],
    }


def evaluate_v2x_graph_consensus(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    topology_weight: torch.Tensor,
    distance_m: torch.Tensor,
    los_flag: torch.Tensor,
    node_initial_correct: torch.Tensor,
    node_initial_wrong: torch.Tensor,
    physical_config: V2XPhysicalConfig | Mapping[str, object] | None = None,
    avalanche_config: AvalancheBridgeConfig | Mapping[str, object] | None = None,
    energy_config: EnergyProxyConfig | Mapping[str, object] | None = None,
    delay_config: DelayProxyConfig | Mapping[str, object] | None = None,
    link_success: torch.Tensor | None = None,
    shadow_offset_db: torch.Tensor | None = None,
    return_details: bool = False,
) -> dict[str, Any]:
    """Evaluate sparse V2X topology metrics C, D, and E without creating an objective.

    ``shadow_offset_db`` (P1-1.2, default None -> byte-identical) is an optional per-edge additive
    path-loss dB offset (e.g. a carried AR(1) log-normal shadow draw). It rides the differentiable
    SINR path and is MODEL-UNOBSERVABLE (it enters only here, never the node/edge features).
    """

    physical = _physical_config(physical_config)
    avalanche = _avalanche_config(avalanche_config)
    energy = _energy_config(energy_config)
    delay = _delay_config(delay_config)

    src, dst, weight = _validate_edge_inputs(
        num_nodes=num_nodes,
        src_index=src_index,
        dst_index=dst_index,
        topology_weight=topology_weight,
    )
    edge_count = int(src.numel())
    distance = _require_floating_vector("distance_m", distance_m, edge_count, weight)
    los = los_flag.to(dtype=weight.dtype, device=weight.device).reshape(-1)
    if los.numel() != edge_count:
        raise ValueError("los_flag must have one value per edge")
    initial_correct = _require_node_probability("node_initial_correct", node_initial_correct, num_nodes, weight)
    initial_wrong = _require_node_probability("node_initial_wrong", node_initial_wrong, num_nodes, weight)
    if bool(torch.any((initial_correct + initial_wrong).detach() > 1.0 + PROBABILITY_TOLERANCE).cpu()):
        raise ValueError("node_initial_correct + node_initial_wrong must be <= 1")

    # D/E enabler: load-aware interference from the receiver in-load that the SELECTED
    # topology imposes (differentiable through topology_weight). Only when enabled and
    # no explicit link_success override is given.
    edge_interference = None
    if physical.interference_density_coupling_db > 0.0 and link_success is None and edge_count:
        in_load = weight.new_zeros((num_nodes,)).index_add(0, dst, weight)
        reference_load = max(float(physical.interference_reference_load), 1e-9)
        load_ratio = torch.clamp(in_load.index_select(0, dst) / reference_load, min=1.0)
        edge_interference = (
            _scalar_like(physical.interference_proxy_dbm, weight)
            + _scalar_like(physical.interference_density_coupling_db, weight) * torch.log10(load_ratio)
        )

    edge_shadow = None
    if shadow_offset_db is not None and edge_count:
        edge_shadow = _require_floating_vector("shadow_offset_db", shadow_offset_db, edge_count, weight)

    channel = _channel_proxy(
        distance_m=distance,
        los_flag=los,
        physical=physical,
        link_success=link_success,
        interference_dbm=edge_interference,
        shadow_offset_db=edge_shadow,
    )
    avalanche_result = evaluate_graph_coupled_avalanche(
        num_nodes=num_nodes,
        src_index=src,
        dst_index=dst,
        topology_weight=weight,
        link_success=channel["link_success"],
        initial_correct_preference=initial_correct,
        initial_wrong_preference=initial_wrong,
        k=avalanche.k,
        alpha=avalanche.alpha,
        beta=avalanche.beta,
        rounds=avalanche.rounds,
        eps=avalanche.eps,
        temperature=avalanche.temperature,
        allow_self_loops=avalanche.allow_self_loops,
        allow_multi_edges=avalanche.allow_multi_edges,
        query_support_backend=avalanche.query_support_backend,
        diagnostics_mode=avalanche.diagnostics_mode,
        quenched_quadrature=avalanche.quenched_quadrature,
        network_tail_tau=avalanche.network_tail_tau,
        network_tail_var_floor=avalanche.network_tail_var_floor,
        # the quenched closure (Q>1) carries Q copies and does not emit per-round history;
        # request history only in the mean-field (Q=1) path.
        return_history=(return_details or avalanche.return_history) and int(avalanche.quenched_quadrature) <= 1,
    )

    expected_rounds = avalanche_result["node_expected_rounds"]
    node_delay_seconds = expected_rounds * _scalar_like(float(physical.single_hop_delay_s), weight)
    support = avalanche_result["query_support"]
    energy_result = _energy_proxy(
        num_nodes=num_nodes,
        src_index=support.src_index,
        normalized_query_weight=support.normalized_query_weight,
        node_expected_rounds=expected_rounds,
        physical=physical,
        avalanche=avalanche,
        energy=energy,
        link_success=channel["link_success"],
    )
    delay_result = _delay_proxy(
        num_nodes=num_nodes,
        src_index=support.src_index,
        normalized_query_weight=support.normalized_query_weight,
        node_expected_rounds=expected_rounds,
        physical=physical,
        delay=delay,
        link_success=channel["link_success"],
    )
    structural_delay_on = bool(delay.structural_delay and channel["link_success"] is not None)
    node_effective_rounds = delay_result["node_effective_rounds"]
    # The D the loss reads: structural effective-rounds when on, legacy round count when off.
    d_rounds_for_loss = node_effective_rounds if structural_delay_on else expected_rounds
    d_seconds_for_loss = (
        delay_result["node_structural_delay_seconds"] if structural_delay_on else node_delay_seconds
    )
    correct = avalanche_result["node_p_correct_decision"]
    wrong = avalanche_result["node_p_wrong_decision"]
    undecided = avalanche_result["node_p_undecided"]
    failure_result = _failure_metrics(
        correct=correct,
        wrong=wrong,
        undecided=undecided,
        expected_rounds=expected_rounds,
        node_energy=energy_result["node_consensus_energy"],
        avalanche=avalanche,
        reference=weight,
    )
    result: dict[str, Any] = {
        "node_p_correct_decision": correct,
        "node_p_wrong_decision": wrong,
        "node_p_undecided": undecided,
        "node_expected_rounds": expected_rounds,
        "C_avalanche_node_mean": correct.mean() if num_nodes else _empty_metric(weight),
        "C_avalanche_node_min": correct.min() if num_nodes else _empty_metric(weight),
        "C_avalanche_node_p10": _quantile_or_zero(correct, 0.10, weight),
        "wrong_decision_node_mean": wrong.mean() if num_nodes else _empty_metric(weight),
        "undecided_node_mean": undecided.mean() if num_nodes else _empty_metric(weight),
        "D_avalanche_rounds_mean": d_rounds_for_loss.mean() if num_nodes else _empty_metric(weight),
        "D_avalanche_rounds_p90": _quantile_or_zero(d_rounds_for_loss, 0.90, weight),
        "D_avalanche_seconds_mean": d_seconds_for_loss.mean() if num_nodes else _empty_metric(weight),
        "D_avalanche_seconds_p90": _quantile_or_zero(d_seconds_for_loss, 0.90, weight),
        "node_delay_seconds": d_seconds_for_loss,
        # Always-on diagnostics so the RESULT can compare the two D definitions:
        "D_protocol_rounds_mean": expected_rounds.mean() if num_nodes else _empty_metric(weight),
        "D_structural_rounds_mean": node_effective_rounds.mean() if num_nodes else _empty_metric(weight),
        "D_structural_active": delay_result["delay_structural_active"],
        "D_structural_reduce_mode": delay_result["delay_structural_reduce_mode"],
        "C_avalanche": correct.mean() if num_nodes else _empty_metric(weight),
        "D_avalanche": d_rounds_for_loss.mean() if num_nodes else _empty_metric(weight),
        "E_avalanche": energy_result["E_consensus_node_mean"],
        "graph_metrics": avalanche_result["graph_metrics"],
        "query_support": support,
        "diagnostics": _diagnostics(
            avalanche_result=avalanche_result,
            energy_result=energy_result,
            channel_result=channel,
            num_nodes=num_nodes,
            reference=weight,
            failure_metrics=failure_result,
        ),
    }
    result.update(failure_result)
    result.update(energy_result)
    if "F_network_tail" in avalanche_result:
        result["F_network_tail"] = avalanche_result["F_network_tail"]
    if return_details:
        result["channel_diagnostics"] = channel
        result["avalanche_details"] = avalanche_result
    return result
