"""Exogenous normalized participation measure ``ω`` (spec §2).

Replaces the hard *eligible set* of the legacy rounds. ``ω`` says how much each node's
final decision matters to the macrostate consensus event, with

    ω_i ≥ 0,    Σ_i ω_i = 1.

Hard contracts (spec §2):

1. ``ω`` is given **exogenously** by the application/scene — these functions take only the
   scene and application-level config (an event location, roles), never a query policy or
   model; there is structurally no way for the policy to enter (constraint #3 "participation
   measure 必须外生、归一化、不可被 policy 修改").
2. ``ω`` is **not** computed from simulator-only truth (no ``Y*``, no peer votes). The
   application kernel uses observable geometry (distance to the application's event location)
   and exogenous role labels only.
3. The SAME rule is used in train / validation / deployment (it lives in the
   ``ConsensusServiceProfile``).
4. ``ω`` carries no gradient (``requires_grad=False``): it is a fixed measure, never an
   optimization target.

Two admissible rules (spec §2):

* **uniform** — ``ω_i = 1/N`` (the network-wide instance);
* **application** — ``ω_i = K_app(d_i, role_i) / Σ_j K_app(d_j, role_j)`` (event-relevant soft
  scope). The headline reports BOTH (uniform headline + application sensitivity), so neither
  rule can hide a scope definition.
"""

from __future__ import annotations

import torch

__all__ = ["uniform_participation", "application_participation", "participation_measure",
           "vehicle_only_participation"]


def uniform_participation(
    num_nodes: int, *, dtype: torch.dtype = torch.float64, device=None
) -> torch.Tensor:
    """``ω_i = 1/N`` — the network-wide uniform instance (spec §2)."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be >= 1")
    w = torch.full((num_nodes,), 1.0 / num_nodes, dtype=dtype, device=device)
    return w.detach()


def vehicle_only_participation(
    scene, *, dtype: torch.dtype = torch.float64, device=None
) -> torch.Tensor:
    """``ω_i = 1/N_veh`` for vehicles, ``0`` for RSU/responder-only nodes (NDH spec §4.2).

    RSU nodes are responders/witnesses, NOT macrostate consensus participants, so they carry
    ``ω_RSU = 0`` (NDH non-degradable constraint #9). Reads the EXOGENOUS role label
    ``scene.node_type`` (0 = vehicle, 1 = RSU) — an admissible exogenous role input (spec §2,
    rule 2); never ``Y*``, votes, or future information, so the exogeneity contract holds. The
    measure still sums to 1 (over the vehicles) and carries no gradient.
    """
    node_type = getattr(scene, "node_type", None)
    if node_type is None:
        raise ValueError(
            "vehicle_only_participation requires scene.node_type (exogenous role labels: "
            "0=vehicle, 1=RSU)")
    is_vehicle = node_type == 0
    n_veh = int(is_vehicle.sum())
    if n_veh < 1:
        raise ValueError("no vehicle nodes -> degenerate participation")
    w = torch.zeros(int(node_type.numel()), dtype=dtype, device=device)
    w[is_vehicle.to(w.device)] = 1.0 / n_veh
    return w.detach()


def application_participation(
    scene,
    *,
    event_xy: torch.Tensor,
    length_scale: float = 50.0,
    roles: torch.Tensor | None = None,
    role_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Application-weighted soft scope ``ω_i ∝ K_app(d_i, role_i)`` (spec §2).

    ``K_app(d_i) = exp(−d_i / length_scale)`` with ``d_i = ‖pos_i − event_xy‖`` the OBSERVABLE
    distance to the application's event location (e.g. a hazard), optionally scaled by an
    exogenous per-node role factor. Vehicles nearer the event participate more. No truth /
    vote / future information enters (constraint #2). Deterministic given the scene + event.

    Args:
        scene: object exposing ``positions`` ``[N, 2]``.
        event_xy: ``[2]`` application event location (exogenous app input, not ``Y*``).
        length_scale: decay length (m) of the soft scope; smaller ⇒ tighter scope.
        roles: optional ``[N]`` long role ids; ``role_weights[roles]`` scales ``K_app``.
        role_weights: optional ``[num_roles]`` non-negative role multipliers.

    Returns:
        ``[N]`` participation measure, non-negative, summing to 1, ``requires_grad=False``.
    """
    pos = scene.positions
    event = torch.as_tensor(event_xy, dtype=pos.dtype, device=pos.device).reshape(2)
    if length_scale <= 0:
        raise ValueError("length_scale must be > 0")
    with torch.no_grad():
        d = (pos - event).norm(dim=1)                       # [N] observable geometry
        k_app = torch.exp(-d / length_scale)                # [N] soft scope kernel
        if roles is not None and role_weights is not None:
            rw = torch.as_tensor(role_weights, dtype=pos.dtype, device=pos.device)
            if bool((rw < 0).any()):
                raise ValueError("role_weights must be non-negative")
            k_app = k_app * rw[roles.to(pos.device)]
        total = k_app.sum()
        if not bool(total > 0):
            raise ValueError("application kernel summed to 0 (degenerate scope)")
        w = k_app / total
    return w.detach()


def participation_measure(scene, rule: str, **kwargs) -> torch.Tensor:
    """Dispatch to the participation rule named in the service profile (spec §2).

    Deliberately takes only ``(scene, rule, **app-config)`` — never a policy/model/evidence —
    so the exogeneity contract is structural (constraint #3).
    """
    if rule == "uniform":
        dtype = kwargs.get("dtype", scene.positions.dtype)
        device = kwargs.get("device", scene.positions.device)
        return uniform_participation(scene.num_nodes, dtype=dtype, device=device)
    if rule == "application":
        return application_participation(scene, **kwargs)
    if rule == "vehicle_uniform":
        dtype = kwargs.get("dtype", scene.positions.dtype)
        device = kwargs.get("device", scene.positions.device)
        return vehicle_only_participation(scene, dtype=dtype, device=device)
    raise ValueError(
        f"unknown participation rule {rule!r}; expected one of "
        "('uniform', 'application', 'vehicle_uniform')")
