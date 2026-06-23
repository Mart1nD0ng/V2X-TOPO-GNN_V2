from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .channel_spec import resolve_pathloss_spec


@dataclass(frozen=True)
class ChannelConfig:
    carrier_frequency_ghz: float = 5.9
    bandwidth_mhz: float = 20.0
    tx_power_dbm: float = 23.0
    noise_dbm: float = -95.0
    interference_proxy_dbm: float = -78.0
    nlos_penalty_db: float = 12.0
    mcs_threshold_db: float = 8.0
    transition_width_db: float = 2.0
    # #2: opt-in 3GPP TR 37.885-grounded NR V2X sidelink path loss. "legacy" = the single-slope proxy
    # above (default, byte-identical); "tr37885" = the 37.885 V2V models (LOS/NLOS/NLOSv soft-mixed,
    # shadowing folded as BLER-transition broadening). See src/v2x_env/nr_v2x_sidelink.py.
    pathloss_model: str = "legacy"
    scenario: str = "urban"          # "highway" | "urban" (used when pathloss_model="tr37885")
    nlosv_extra_db: float = 6.0
    subchannels: float = 5.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ChannelConfig":
        data = data or {}
        return cls(
            carrier_frequency_ghz=float(data.get("carrier_frequency_ghz", cls.carrier_frequency_ghz)),
            bandwidth_mhz=float(data.get("bandwidth_mhz", cls.bandwidth_mhz)),
            tx_power_dbm=float(data.get("tx_power_dbm", cls.tx_power_dbm)),
            noise_dbm=float(data.get("noise_dbm", cls.noise_dbm)),
            interference_proxy_dbm=float(data.get("interference_proxy_dbm", cls.interference_proxy_dbm)),
            nlos_penalty_db=float(data.get("nlos_penalty_db", cls.nlos_penalty_db)),
            mcs_threshold_db=float(data.get("mcs_threshold_db", cls.mcs_threshold_db)),
            transition_width_db=float(data.get("transition_width_db", cls.transition_width_db)),
            pathloss_model=str(data.get("pathloss_model", cls.pathloss_model)),
            scenario=str(data.get("scenario", cls.scenario)),
            nlosv_extra_db=float(data.get("nlosv_extra_db", cls.nlosv_extra_db)),
            subchannels=float(data.get("subchannels", cls.subchannels)),
        )


def path_loss_db(distance_m: float | np.ndarray, los_flag: bool | np.ndarray, config: ChannelConfig | Mapping[str, object]) -> np.ndarray:
    cfg = config if isinstance(config, ChannelConfig) else ChannelConfig.from_mapping(config)
    # Coefficients come from the single source (src/v2x_env/channel_spec.py) so the numpy
    # candidate-graph path and the torch evaluator path cannot drift.
    spec = resolve_pathloss_spec("legacy", nlos_penalty_db=cfg.nlos_penalty_db)
    distance = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    base = (
        spec.los_intercept_db
        + spec.los_log_distance_slope * np.log10(distance)
        + spec.los_log_frequency_slope * np.log10(cfg.carrier_frequency_ghz)
    )
    los = np.asarray(los_flag, dtype=bool)
    return base + np.where(los, 0.0, spec.nlos_flat_penalty_db)


def received_power_dbm(tx_power_dbm: float, path_loss_value_db: float | np.ndarray) -> np.ndarray:
    return float(tx_power_dbm) - np.asarray(path_loss_value_db, dtype=float)


def _dbm_to_mw(value_dbm: float | np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(value_dbm, dtype=float) / 10.0)


def sinr_db(
    received_power_value_dbm: float | np.ndarray,
    noise_dbm: float,
    interference_proxy_dbm: float,
) -> np.ndarray:
    signal_mw = _dbm_to_mw(received_power_value_dbm)
    impairment_mw = _dbm_to_mw(noise_dbm) + _dbm_to_mw(interference_proxy_dbm)
    return 10.0 * np.log10(np.maximum(signal_mw / impairment_mw, 1e-12))


def packet_success_probability(
    sinr_value_db: float | np.ndarray,
    mcs_index: int | None = None,
    threshold: float | None = None,
    transition_width_db: float = 2.0,
) -> np.ndarray:
    if threshold is None:
        threshold = 4.0 + 1.5 * float(mcs_index if mcs_index is not None else 2)
    scaled = (np.asarray(sinr_value_db, dtype=float) - float(threshold)) / max(float(transition_width_db), 1e-6)
    return 1.0 / (1.0 + np.exp(-scaled))


def density_coupled_interference_dbm(
    base_interference_dbm: float,
    local_degree: float | np.ndarray,
    reference_degree: float,
    coupling_db: float,
) -> np.ndarray:
    """Interference floor that rises with local concurrent-transmitter density.

    Realistic V2X interference grows with how many neighbours share the medium. A
    receiver whose local feasible degree is ``local_degree`` sees an interference
    floor ``base + coupling_db * log10(max(local_degree / reference_degree, 1))``.
    With ``coupling_db = 0`` this reduces to the fixed proxy (legacy behaviour).
    """
    ratio = np.maximum(np.asarray(local_degree, dtype=float) / max(float(reference_degree), 1e-9), 1.0)
    return float(base_interference_dbm) + float(coupling_db) * np.log10(ratio)


def edge_success_probability(
    distance_m: np.ndarray,
    los_flag: np.ndarray,
    config: ChannelConfig,
    interference_dbm: float | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if getattr(config, "pathloss_model", "legacy") == "tr37885":
        # #2: 3GPP TR 37.885-grounded NR V2X sidelink path loss + shadowing + BLER.
        from .nr_v2x_sidelink import NRV2XSidelinkConfig, link_success
        nr = NRV2XSidelinkConfig(
            scenario=config.scenario, carrier_frequency_ghz=config.carrier_frequency_ghz,
            tx_power_dbm=config.tx_power_dbm, noise_dbm=config.noise_dbm,
            interference_dbm=config.interference_proxy_dbm,
            mcs_sinr_threshold_db=config.mcs_threshold_db, bler_transition_db=config.transition_width_db,
            nlosv_extra_db=config.nlosv_extra_db, subchannels=config.subchannels,
        )
        return link_success(distance_m, np.asarray(los_flag, dtype=float), nr, interference_dbm=interference_dbm)
    path_values = path_loss_db(distance_m, los_flag, config)
    rx_power = received_power_dbm(config.tx_power_dbm, path_values)
    interference = config.interference_proxy_dbm if interference_dbm is None else interference_dbm
    sinr_values = sinr_db(rx_power, config.noise_dbm, interference)
    success = packet_success_probability(
        sinr_values,
        threshold=config.mcs_threshold_db,
        transition_width_db=config.transition_width_db,
    )
    return success, sinr_values
