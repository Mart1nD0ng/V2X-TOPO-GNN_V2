"""``ExperimentSpec`` — the train/eval reproducibility & compatibility fingerprint (plan §4).

Captures the full configuration of a headline run as deterministic hashes so that train and
eval cannot silently diverge (Mechanism Identifiability Contract C5; plan §4 "train/eval
consistency"). The compatibility check enforces:

* **physics**, **protocol**, **service profile** and **query law** must MATCH between train and
  eval, unless the differing axis is explicitly registered in ``allowed_ood_axes`` (plan §11
  OOD experiments register exactly one axis at a time);
* the **ideal/full-link** distinction (``full_physics``) must ALWAYS match — the historical
  ideal-trained-but-full-physics-evaluated bug is NEVER an OOD axis (ideal link is a separate,
  explicitly-flagged ablation, spec §12 / constraint #9).

``evidence_descriptor`` / ``scene_descriptor`` are caller-supplied generation descriptors (the
scenario name + parameters, the scene-size distribution) — hashing a descriptor, not a single
sampled instance, so the spec pins the *distribution*, not one draw.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

__all__ = [
    "ExperimentSpec",
    "IncompatibleExperimentError",
    "build_experiment_spec",
    "build_episode_experiment_spec",
    "check_train_eval_compatible",
    "assert_canonical_mechanisms",
    "assert_headline_grounded",
    "MANDATORY_MECHANISMS",
    "OOD_AXES",
]

# plan §4 mandatory mechanism flags the canonical episode trace must carry.
MANDATORY_MECHANISMS = (
    "parallel_unicast", "poll_window_ms", "source_destination_accounting",
    "collision_self_exclusion", "request_response", "dynamic_transient_load",
    "interference_graph", "mode2_collision", "half_duplex", "queueing", "finite_harq",
    "fbl_dispersion", "full_physics",
)
# runtime-toggleable mechanisms that must be ON for a canonical/headline episode.
_MUST_BE_TRUE = (
    "parallel_unicast", "source_destination_accounting", "collision_self_exclusion",
    "request_response", "dynamic_transient_load", "interference_graph", "mode2_collision",
    "half_duplex", "queueing", "finite_harq", "fbl_dispersion",
)


def assert_canonical_mechanisms(trace: dict, *, require_full_physics: bool = True) -> None:
    """Assert a mechanism trace is a fully-wired canonical (headline) episode (plan §4).

    Raises :class:`ValueError` if any mandatory flag is missing, any toggleable mechanism is
    off, a ``tau_proxy`` is used, the policy reads truth/votes, ``poll_window_ms <= 0``, or
    (when ``require_full_physics``) the episode ran on an ideal ``link_override`` instead of the
    full physical chain (constraint #9 — ideal link is an ablation, not the headline).
    """
    missing = [m for m in MANDATORY_MECHANISMS if m not in trace]
    if missing:
        raise ValueError(f"canonical mechanism trace is missing mandatory flag(s): {missing}")
    # A bare ideal link_override BYPASSES the physical chain, so such a trace can NEVER be a
    # canonical episode — reject it unconditionally (independent of require_full_physics), so the
    # check cannot be satisfied by a bypassed episode whose flags are nominally True (constraint #9).
    if trace.get("link_override", None) is not None:
        raise ValueError(
            "episode ran on an ideal link_override (the physical chain was bypassed): NOT a "
            "canonical episode; the ideal link is a separate explicit ablation (constraint #9).")
    if require_full_physics and not trace.get("full_physics", False):
        raise ValueError(
            "episode is NOT full physics: not headline-grounded; the ideal link is a separate "
            "explicit ablation (constraint #9).")
    if trace.get("poll_window_ms", 0) <= 0:
        raise ValueError("poll_window_ms (Delta_poll) must be > 0 on the canonical path")
    if trace.get("tau_proxy", False):
        raise ValueError("tau_proxy must be False on the canonical path (constraint #7)")
    if trace.get("policy_uses_truth_or_vote", False):
        raise ValueError("the query policy must not read truth/votes (constraint #10)")
    off = [m for m in _MUST_BE_TRUE if not trace.get(m, False)]
    if off:
        raise ValueError(f"mandatory mechanism(s) disabled / off the canonical path: {off}")


def assert_headline_grounded(spec: "ExperimentSpec") -> None:
    """Assert an :class:`ExperimentSpec` is a valid HEADLINE spec — i.e. full physics (plan §4).

    A bare ideal-link spec (``full_physics=False``) is an ablation and must never be a headline.
    """
    if not spec.full_physics:
        raise ValueError(
            "ExperimentSpec is not headline-grounded: full_physics is False (ideal link). The "
            "headline must run the full physical chain; ideal link is a separate ablation.")

# OOD axis -> the spec field it is allowed to vary. EXACTLY the plan §12/Phase-11 axis
# catalogue (node count, density, road geometry, interference, evidence covariance,
# sensor-source composition, protocol, mobility). NOTE: service_profile and query_law are
# DELIBERATELY NOT OOD axes — a checkpoint is never trained under one service target / query
# law and evaluated under another (that would compare different constraints / a different
# policy family), so those mismatches always block the headline.
_AXIS_TO_FIELD: dict[str, str] = {
    "protocol": "protocol_hash",
    "physics": "physics_hash",
    "interference": "physics_hash",
    "evidence_covariance": "evidence_hash",
    "sensor_source": "evidence_hash",
    "node_count": "scene_distribution_hash",
    "density": "scene_distribution_hash",
    "road_geometry": "scene_distribution_hash",
    "mobility": "scene_distribution_hash",
}
OOD_AXES = tuple(_AXIS_TO_FIELD)

_COMPARED_FIELDS = ("protocol_hash", "service_profile_hash", "physics_hash",
                    "evidence_hash", "scene_distribution_hash", "query_law")


class IncompatibleExperimentError(RuntimeError):
    """Raised when a train spec and an eval spec differ on a non-registered axis."""


def _sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExperimentSpec:
    protocol_hash: str
    service_profile_hash: str
    physics_hash: str
    evidence_hash: str
    scene_distribution_hash: str
    query_law: str
    full_physics: bool
    allowed_ood_axes: tuple[str, ...] = ()

    def config_hash(self) -> str:
        """Single fingerprint of the whole spec (goes in the checkpoint manifest)."""
        return _sha({
            "protocol_hash": self.protocol_hash,
            "service_profile_hash": self.service_profile_hash,
            "physics_hash": self.physics_hash,
            "evidence_hash": self.evidence_hash,
            "scene_distribution_hash": self.scene_distribution_hash,
            "query_law": self.query_law,
            "full_physics": self.full_physics,
            "allowed_ood_axes": sorted(self.allowed_ood_axes),
        })


def build_experiment_spec(
    *,
    protocol_cfg,
    service_profile,
    phy_cfg,
    evidence_descriptor,
    scene_descriptor,
    query_law: str,
    full_physics: bool,
    allowed_ood_axes: tuple[str, ...] = (),
) -> ExperimentSpec:
    """Build an :class:`ExperimentSpec` from the live configs (plan §4).

    ``evidence_descriptor`` / ``scene_descriptor`` may be any JSON-able value describing the
    GENERATIVE config (e.g. ``"iid:p=0.1"`` or a dict). ``allowed_ood_axes`` must be a subset of
    :data:`OOD_AXES` (an unknown axis is a configuration error, raised here).
    """
    for ax in allowed_ood_axes:
        if ax not in _AXIS_TO_FIELD:
            raise ValueError(f"unknown OOD axis {ax!r}; allowed axes (ood) are {OOD_AXES}")
    if query_law not in ("esp", "cdq", "cdq2"):
        raise ValueError("query_law must be 'esp', 'cdq', or 'cdq2'")
    return ExperimentSpec(
        protocol_hash=protocol_cfg.config_hash(),
        service_profile_hash=service_profile.config_hash(),
        physics_hash=phy_cfg.config_hash(),
        evidence_hash=_sha(evidence_descriptor),
        scene_distribution_hash=_sha(scene_descriptor),
        query_law=query_law,
        full_physics=bool(full_physics),
        allowed_ood_axes=tuple(allowed_ood_axes),
    )


def build_episode_experiment_spec(
    *,
    protocol_cfg,
    service_profile,
    phy_cfg,
    evidence_descriptor,
    scene_descriptor,
    query_law: str,
    link_override,
    allowed_ood_axes: tuple[str, ...] = (),
) -> ExperimentSpec:
    """Build an :class:`ExperimentSpec` for a concrete episode, binding the configs as a single
    consistent source (plan §4 "single source").

    The :class:`ConsensusServiceProfile` is the source of truth for ``(k, alpha, beta, R_d,
    Delta_poll)``; the ``ProtocolConfig`` and ``RoundPhysicsConfig`` actually used by the episode
    MUST agree with it (else the configuration is internally inconsistent — a silent train/eval
    hazard). ``full_physics`` is derived from ``link_override`` (``None`` => full physics).
    """
    if (protocol_cfg.k, protocol_cfg.alpha, protocol_cfg.beta) != (
            service_profile.k, service_profile.alpha, service_profile.beta):
        raise ValueError(
            f"ProtocolConfig (k={protocol_cfg.k}, alpha={protocol_cfg.alpha}, beta={protocol_cfg.beta}) "
            f"disagrees with the service profile (k={service_profile.k}, alpha={service_profile.alpha}, "
            f"beta={service_profile.beta}); the profile is the single source of the polling epoch.")
    if protocol_cfg.r_max != service_profile.max_poll_epochs:
        raise ValueError(
            f"ProtocolConfig.r_max ({protocol_cfg.r_max}) disagrees with the profile's "
            f"max_poll_epochs / R_d ({service_profile.max_poll_epochs}).")
    if abs(phy_cfg.poll_window_s - service_profile.poll_window_s) > 1e-12:
        raise ValueError(
            f"RoundPhysicsConfig.poll_window_s ({phy_cfg.poll_window_s}) disagrees with the "
            f"profile's Delta_poll ({service_profile.poll_window_s}); they must be identical.")
    return build_experiment_spec(
        protocol_cfg=protocol_cfg, service_profile=service_profile, phy_cfg=phy_cfg,
        evidence_descriptor=evidence_descriptor, scene_descriptor=scene_descriptor,
        query_law=query_law, full_physics=(link_override is None),
        allowed_ood_axes=allowed_ood_axes)


def check_train_eval_compatible(train: ExperimentSpec, evaluation: ExperimentSpec) -> None:
    """Raise :class:`IncompatibleExperimentError` if eval is not a valid comparison to train.

    The eval run declares which axes it intentionally varies via ``evaluation.allowed_ood_axes``
    (plan §11: change ONE main axis at a time). Every other fingerprint must match. The
    ideal/full-link distinction is checked FIRST and unconditionally (constraint #9).
    """
    # (1) the ideal/full-link guard — never an OOD axis (the historical headline bug).
    if train.full_physics != evaluation.full_physics:
        raise IncompatibleExperimentError(
            "ideal/full-link mismatch: full_physics differs between train "
            f"({train.full_physics}) and eval ({evaluation.full_physics}). A bare link_override "
            "(ideal link) is a separate explicit ablation, NEVER an OOD axis (constraint #9).")

    allowed_fields = {_AXIS_TO_FIELD[ax] for ax in evaluation.allowed_ood_axes}
    for field in _COMPARED_FIELDS:
        if getattr(train, field) != getattr(evaluation, field) and field not in allowed_fields:
            raise IncompatibleExperimentError(
                f"{field} differs between train and eval but no registered OOD axis permits it; "
                f"either use identical config or register the corresponding axis in "
                f"allowed_ood_axes (axes->field map: {_AXIS_TO_FIELD}).")
