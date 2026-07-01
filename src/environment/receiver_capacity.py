"""Heterogeneous receiver capacity mu_j (G-NDH-HETEROGENEOUS-CAPACITY, spec §4.4).

The current physics uses a single GLOBAL M/M/1 service rate, so "nearest peer" == "fastest responder"
and distance need not distinguish responder capability. This mechanism gives each node its own service
rate ``mu_j`` (polls served per window), drawn log-normally, with RSUs stronger on average than
vehicles. The receiver queue then becomes ``rho_j = (Lambda_j + b_j) / mu_j`` (round_physics), so a
slightly-farther high-``mu`` peer can beat a near low-``mu`` peer — a non-distance structure.

Truth/observability split (Contract C2): the TRUE ``mu_j`` enters the physics (the queue); the model
and every heuristic may read ONLY a NOISY proxy ``mu_hat_j = mu_j * exp(sigma_obs * xi)`` (spec §4.5),
never the true ``mu_j``. RSU capacity is higher on average but FINITE and log-normally spread, so a
heavily-loaded RSU can still saturate (RSU overload possible; no trivial nearest-RSU oracle, risk R3).
"""

from __future__ import annotations

import torch

__all__ = ["assign_receiver_capacity", "noisy_capacity_proxy"]


def assign_receiver_capacity(
    node_type: torch.Tensor,
    *,
    mu_vehicle_base: float = 8.0,
    vehicle_capacity_logstd: float = 0.5,
    rsu_capacity_multiplier: float = 5.0,
    rsu_capacity_logstd: float = 0.1,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Per-node true service rate ``mu_j`` (polls/window), log-normal (spec §4.4). Registry §4 defaults.

    Vehicles: ``mu_j = mu_vehicle_base * exp(sigma_veh * eps)``. RSU:
    ``mu_j = rsu_capacity_multiplier * mu_vehicle_base * exp(sigma_rsu * eps)`` (higher mean, tighter
    spread). ``node_type``: 0 = vehicle, 1 = RSU. Deterministic given ``generator``; always positive.
    """
    if mu_vehicle_base <= 0 or rsu_capacity_multiplier <= 0:
        raise ValueError("mu_vehicle_base and rsu_capacity_multiplier must be > 0")
    node_type = node_type.to(torch.long)
    N = int(node_type.numel())
    eps = torch.randn(N, generator=generator, dtype=dtype)
    is_rsu = node_type == 1
    base = torch.where(is_rsu, torch.as_tensor(rsu_capacity_multiplier * mu_vehicle_base, dtype=dtype),
                       torch.as_tensor(float(mu_vehicle_base), dtype=dtype))
    logstd = torch.where(is_rsu, torch.as_tensor(float(rsu_capacity_logstd), dtype=dtype),
                         torch.as_tensor(float(vehicle_capacity_logstd), dtype=dtype))
    return (base * torch.exp(logstd * eps)).to(dtype)


def noisy_capacity_proxy(
    mu: torch.Tensor, capacity_proxy_noise: float = 0.2,
    *, generator: torch.Generator | None = None,
) -> torch.Tensor:
    """DEPLOYABLE noisy capacity estimate ``mu_hat_j = mu_j * exp(sigma_obs * xi)`` (spec §4.5).

    The ONLY capacity signal a model / heuristic may read (the true ``mu_j`` is simulator-side truth,
    Contract C2). ``capacity_proxy_noise = 0`` returns the true ``mu`` (the noiseless observability
    limit). Deterministic given ``generator``.

    Bias note: this estimator is **log-unbiased** but **level-biased upward** —
    ``E[mu_hat] = mu * exp(sigma_obs^2 / 2)`` (≈ +2% at the default σ=0.2, +13% at σ=0.5). The intended
    deployable FEATURE (spec §4.5) is therefore ``capacity_proxy_log = log(mu_hat)`` (log-space,
    where the estimate is unbiased); the feature-schema gate consumes the log form.
    """
    if capacity_proxy_noise < 0:
        raise ValueError("capacity_proxy_noise (sigma_obs) must be >= 0")
    if capacity_proxy_noise == 0:
        return mu.clone()
    xi = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
    return mu * torch.exp(capacity_proxy_noise * xi)
