"""3GPP TR 37.885-grounded NR V2X sidelink (PC5) PHY abstraction.

Implements the V2V path-loss models of 3GPP TR 37.885 "Study on evaluation methodology of new V2X use
cases" (Highway and Urban scenarios, LOS / NLOSv / NLOS), with:
  * shadow fading folded as a CLOSED-FORM VARIANCE (it broadens the BLER-vs-SINR transition rather than
    being a sampled draw), so the resulting link reliability stays smooth/differentiable -- no Monte-Carlo;
  * an NR-sidelink BLER-vs-SINR mapping (logistic on effective SINR, MCS-grounded threshold);
  * a mode-2 (sensing-based semi-persistent scheduling) RESOURCE-COLLISION abstraction: with N concurrent
    transmitters over a pool of S subchannels the half-duplex/collision loss is 1-(1-1/S)^(N-1), which is
    the channel through which the SELECTED topology's receiver in-load couples into link reliability
    (the load-coupling correlation #1 consumes).

HONEST SCOPE: this is a 37.885-GROUNDED ABSTRACTION (path loss + shadowing variance + SINR + BLER +
mode-2 collision), NOT a PC5 link-level simulator (no PSCCH/PSSCH/DMRS, no per-slot sensing window or
resource-reservation modelling). The per-slot resource grid and sensing belong to an NS-3 co-simulation,
which this module is designed to be DRIVEN by: every entry point accepts externally-supplied distances,
LOS state, and/or interference (from SUMO/NS-3 traces) instead of the built-in proxies.

References: 3GPP TR 37.885 v15.3.0 Table 6.2.1-1 (V2V path loss), 6.2.1-2/-3 (LOS probability, shadowing).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .channel_spec import LOGISTIC_PROBIT as _LOGISTIC_PROBIT
from .channel_spec import resolve_pathloss_spec

# _LOGISTIC_PROBIT (pi/sqrt(3) ~= 1.814) is now sourced from channel_spec so the shadow->BLER-transition
# broadening factor is shared with the torch evaluator (a N(0, sigma^2) shadow broadens a logistic BLER of
# scale s to sqrt(s^2 + (sigma/LOGISTIC_PROBIT)^2)).


@dataclass(frozen=True)
class NRV2XSidelinkConfig:
    scenario: str = "urban"            # "highway" | "urban" (TR 37.885 6.1.x)
    carrier_frequency_ghz: float = 5.9
    tx_power_dbm: float = 23.0
    noise_dbm: float = -95.0
    interference_dbm: float = -95.0    # external/background interference floor (mode-2 collision added separately)
    mcs_sinr_threshold_db: float = 8.0  # NR MCS operating SINR (BLER=0.5 point)
    bler_transition_db: float = 2.0     # logistic BLER-vs-SINR transition width (steepness)
    nlosv_extra_db: float = 6.0         # TR 37.885 NLOSv additional vehicle-blockage loss (mean)
    nlosv_extra_std_db: float = 4.5     # NLOSv blockage std (folded into shadow variance)
    subchannels: float = 5.0            # mode-2 resource pool size S (sub-channels)
    half_duplex: bool = True            # a transmitting node cannot receive in the same slot (mode-2)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "NRV2XSidelinkConfig":
        data = data or {}
        cfg = cls(
            scenario=str(data.get("scenario", cls.scenario)),
            carrier_frequency_ghz=float(data.get("carrier_frequency_ghz", cls.carrier_frequency_ghz)),
            tx_power_dbm=float(data.get("tx_power_dbm", cls.tx_power_dbm)),
            noise_dbm=float(data.get("noise_dbm", cls.noise_dbm)),
            interference_dbm=float(data.get("interference_dbm", cls.interference_dbm)),
            mcs_sinr_threshold_db=float(data.get("mcs_sinr_threshold_db", cls.mcs_sinr_threshold_db)),
            bler_transition_db=float(data.get("bler_transition_db", cls.bler_transition_db)),
            nlosv_extra_db=float(data.get("nlosv_extra_db", cls.nlosv_extra_db)),
            nlosv_extra_std_db=float(data.get("nlosv_extra_std_db", cls.nlosv_extra_std_db)),
            subchannels=float(data.get("subchannels", cls.subchannels)),
            half_duplex=bool(data.get("half_duplex", cls.half_duplex)),
        )
        if cfg.scenario not in {"highway", "urban"}:
            raise ValueError("scenario must be 'highway' or 'urban'")
        if cfg.bler_transition_db <= 0.0:
            raise ValueError("bler_transition_db must be positive")
        if cfg.subchannels < 1.0:
            raise ValueError("subchannels must be >= 1")
        return cfg


def path_loss_and_shadow_var(
    distance_m: np.ndarray, los_prob: np.ndarray, config: NRV2XSidelinkConfig
) -> tuple[np.ndarray, np.ndarray]:
    """TR 37.885 V2V path loss [dB] and shadow-fading VARIANCE [dB^2], soft-mixed over LOS/non-LOS by
    ``los_prob`` in [0, 1]. fc in GHz, d in m. (Highway is LOS-dominated; the non-LOS branch there is
    LOS + NLOSv vehicle blockage. Urban non-LOS blends the NLOS building model with NLOSv.)"""
    d = np.maximum(np.asarray(distance_m, dtype=float), 1.0)
    p_los = np.clip(np.asarray(los_prob, dtype=float), 0.0, 1.0)
    log_d = np.log10(d)
    log_fc = np.log10(config.carrier_frequency_ghz)
    # Coefficients from the single source (channel_spec.py); shared with the torch evaluator.
    spec = resolve_pathloss_spec(
        "tr37885",
        scenario=config.scenario,
        nlosv_extra_db=config.nlosv_extra_db,
        nlosv_extra_std_db=config.nlosv_extra_std_db,
    )
    pl_los = spec.los_intercept_db + spec.los_log_distance_slope * log_d + spec.los_log_frequency_slope * log_fc
    pl_nlos = spec.nlos_intercept_db + spec.nlos_log_distance_slope * log_d + spec.nlos_log_frequency_slope * log_fc
    pl_non = np.maximum(pl_nlos, pl_los + spec.nlosv_extra_db)  # non-LOS = worse of building/vehicle block
    pl = p_los * pl_los + (1.0 - p_los) * pl_non
    shadow_var = p_los * spec.shadow_var_los_db2 + (1.0 - p_los) * spec.shadow_var_nlos_db2
    return pl, shadow_var


def mode2_collision_probability(concurrent_tx: np.ndarray, subchannels: float) -> np.ndarray:
    """Mode-2 (sensing-based SPS) resource-collision probability at a receiver with ``concurrent_tx``
    contending transmitters over ``subchannels`` resources: 1 - (1 - 1/S)^(N-1). Differentiable in N."""
    n = np.maximum(np.asarray(concurrent_tx, dtype=float), 0.0)
    s = max(float(subchannels), 1.0)
    others = np.maximum(n - 1.0, 0.0)
    return 1.0 - np.power(1.0 - 1.0 / s, others)


def link_success(
    distance_m: np.ndarray,
    los_prob: np.ndarray,
    config: NRV2XSidelinkConfig,
    *,
    interference_dbm: np.ndarray | float | None = None,
    concurrent_tx: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-link success probability and effective SINR [dB]. Combines TR 37.885 path loss, shadow fading
    folded as BLER-transition broadening (closed-form, no sampling), the NR BLER-vs-SINR logistic, and the
    mode-2 collision loss. ``interference_dbm`` / ``concurrent_tx`` may be supplied externally (NS-3/SUMO)."""
    pl, shadow_var = path_loss_and_shadow_var(distance_m, los_prob, config)
    rx_dbm = config.tx_power_dbm - pl
    interference = config.interference_dbm if interference_dbm is None else interference_dbm
    signal_mw = np.power(10.0, rx_dbm / 10.0)
    impair_mw = np.power(10.0, config.noise_dbm / 10.0) + np.power(10.0, np.asarray(interference, dtype=float) / 10.0)
    sinr = 10.0 * np.log10(np.maximum(signal_mw / impair_mw, 1e-12))
    # shadow fading broadens the BLER transition: integrate logistic(scale=t) over N(0, shadow_var)
    eff_transition = np.sqrt(config.bler_transition_db ** 2 + shadow_var / (_LOGISTIC_PROBIT ** 2))
    bler_success = 1.0 / (1.0 + np.exp(-(sinr - config.mcs_sinr_threshold_db) / eff_transition))
    if concurrent_tx is not None:
        bler_success = bler_success * (1.0 - mode2_collision_probability(concurrent_tx, config.subchannels))
    return np.clip(bler_success, 0.0, 1.0), sinr
