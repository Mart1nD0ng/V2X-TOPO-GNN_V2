"""CSI aging: stale channel-state features for the model (G-NDH-CSI-AGING, spec §5).

Deployment reality: the link quality a node can OBSERVE at decision time is a *stale* CSI estimate
`\\hat{\\gamma}_{ij}(t - a)`, aged by `a` (CSI age) and corrupted by measurement + shadow-decorrelation
noise. The TRUE physics still uses the CURRENT channel `\\gamma_{ij}(t)` (computed by
`round_physics`/`edge_geometry` from the current geometry) — CSI aging changes ONLY the model's
*feature*, never the physics link, so there is NO train/eval physics mismatch (constraint #4, spec §5.1).

This lets distance / stale-link-quality stop being optimal where the channel has decorrelated: the age
and uncertainty become informative structure a robust policy must read. Phase 1 is STATIC (no GRU):
the stale estimate is the geometry-true link SINR perturbed by an age-dependent Gaussian whose std is

    sigma_gamma(a) = sqrt( csi_noise_std_db^2 + shadow_ar_std_db^2 * (1 - exp(-2 a / tau_decorr)) ),

an AR(1) shadow-decorrelation model: at `a = 0` the only error is the fixed measurement noise
`csi_noise_std_db`; as `a -> inf` the shadow term fully decorrelates to `shadow_ar_std_db`. All features
are DEPLOYABLE (spec §5.2): geometry + age + observation noise only — no `Y*`, no future CSI, no MC
outcome (Contract C2).
"""

from __future__ import annotations

import math

import torch

from .candidate_graph import build_candidate_graph
from .round_physics import edge_geometry

__all__ = ["stale_csi_edge_features", "csi_uncertainty_db"]


def csi_uncertainty_db(
    csi_age_ms: float, *, csi_noise_std_db: float = 1.0, shadow_ar_std_db: float = 4.0,
    shadow_decorrelation_s: float = 3.0,
) -> float:
    """Age-driven CSI uncertainty (dB): AR(1) shadow decorrelation + fixed measurement noise (spec §5.2).

    `sigma(a) = sqrt(noise^2 + shadow^2 * (1 - exp(-2 a / tau)))`. Monotone non-decreasing in age;
    `= csi_noise_std_db` at `a = 0`; asymptotes to `sqrt(noise^2 + shadow^2)`.
    """
    a_s = max(0.0, csi_age_ms) / 1000.0
    decorr = 1.0 - math.exp(-2.0 * a_s / max(shadow_decorrelation_s, 1e-9))   # in [0, 1)
    return math.sqrt(csi_noise_std_db ** 2 + (shadow_ar_std_db ** 2) * decorr)


def stale_csi_edge_features(
    scene, phy_cfg, *,
    csi_age_ms: float = 100.0,
    csi_noise_std_db: float = 1.0,
    shadow_ar_std_db: float = 4.0,
    shadow_decorrelation_s: float = 3.0,
    delivery_threshold_db: float = 3.0,
    delivery_scale_db: float = 3.0,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict:
    """Per-edge stale-CSI DEPLOYABLE features on `G_comm` (spec §5.2). Deterministic given `generator`.

    Returns a dict of `[E_comm]` tensors:
      * `stale_sinr_db`             — aged link SINR estimate `= true_sinr_db + N(0, sigma(a))`;
      * `stale_delivery`            — deployable delivery-prob proxy `sigmoid((stale_sinr - thr)/scale)`;
      * `csi_age_ms`                — the CSI age (broadcast), an observable feature;
      * `csi_uncertainty`           — `sigma(a)` in dB (broadcast), the age-driven error std;
      * `stale_vs_distance_residual`— `stale_sinr_db - distance_only_sinr_db` (how far the stale link
                                       deviates from what pure distance/LOS pathloss predicts).

    The TRUE link (`edge_geometry(...).rx_power_mw`) is untouched — this is a feature-only mechanism.
    """
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    geom = edge_geometry(gc, phy_cfg)
    E = gc.num_edges
    noise_mw = phy_cfg.noise_mw
    true_sinr_db = 10.0 * torch.log10(geom.rx_power_mw / noise_mw)          # [E] geometry-true link SINR

    sigma = csi_uncertainty_db(csi_age_ms, csi_noise_std_db=csi_noise_std_db,
                               shadow_ar_std_db=shadow_ar_std_db,
                               shadow_decorrelation_s=shadow_decorrelation_s)
    draw = torch.randn(E, generator=generator, dtype=dtype) if sigma > 0 else torch.zeros(E, dtype=dtype)
    stale_sinr_db = true_sinr_db + sigma * draw

    # distance-only predicted SINR: LOS path-loss at the edge distance, no shadow, no NLOS. Pure geometry.
    pl = phy_cfg.pathloss
    d = gc.distance.clamp_min(1.0).to(dtype)
    pl_los_db = pl.los[0] + pl.los[1] * torch.log10(d) + pl.los[2] * math.log10(phy_cfg.fc_ghz)
    dist_rx_dbm = phy_cfg.tx_power_dbm - pl_los_db
    dist_sinr_db = dist_rx_dbm - 10.0 * math.log10(noise_mw)
    stale_vs_distance_residual = stale_sinr_db - dist_sinr_db

    stale_delivery = torch.sigmoid((stale_sinr_db - delivery_threshold_db) / max(delivery_scale_db, 1e-9))
    return {
        "stale_sinr_db": stale_sinr_db,
        "stale_delivery": stale_delivery,
        "csi_age_ms": torch.full((E,), float(csi_age_ms), dtype=dtype),
        "csi_uncertainty": torch.full((E,), float(sigma), dtype=dtype),
        "stale_vs_distance_residual": stale_vs_distance_residual,
    }
