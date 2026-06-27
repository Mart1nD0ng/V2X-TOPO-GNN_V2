"""``ConsensusServiceProfile`` — the single source of protocol / scope / threshold config.

Spec basis: ``MACROSTATE_CONSENSUS_AND_CDQ2_TECHNICAL_SPEC.md`` §4-§6, plan §2. This object
fixes, in ONE place (constraint: "协议、范围、阈值和 deadline 只有一套定义"):

* the **polling epoch** ``k`` parallel unicast request-response polls, quorum ``(α, β)``,
  and the poll window ``Δ_poll`` / max epochs ``R_d``;
* the **participation rule** (uniform | application) for the exogenous measure ``ω``
  (``src.metrics.participation``);
* the **macrostate basin thresholds** ``ρ_f`` (correct/wrong decisive mass) and ``ρ_s``
  (split mass), which MUST satisfy the spec §4 disjointness condition ``ρ_s > 1 − ρ_f`` so
  the correct/wrong basins never overlap the split basin;
* the **reliability budgets** ``ε_w, ε_s, ε_d`` (hard constraints, spec §6) and the
  latency quantile ``q`` / energy budget of the constrained objective.

A deterministic :meth:`config_hash` goes into the checkpoint manifest so train/eval profile
drift is detectable (plan §2; Mechanism Contract C5). The dataclass is frozen — config is
immutable once built; use :meth:`replace` for a validated variant.

All thresholds are *recommended defaults flagged for user override* (see
``docs/MACROSTATE_PROGRESS.md`` "Adopted defaults"): the spec gives valid ranges, not exact
constants, so the user overrides here without touching any math/code.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace as _dc_replace

__all__ = ["ConsensusServiceProfile"]

_PARTICIPATION_RULES = ("uniform", "application")


@dataclass(frozen=True)
class ConsensusServiceProfile:
    """Unified consensus service configuration (spec §6). Frozen + validated."""

    # ---- participation scope (spec §2) ----
    participation_weight_rule: str = "uniform"   # "uniform" | "application"

    # ---- polling epoch (spec §5) ----
    poll_window_ms: float = 10.0     # Δ_poll
    k: int = 4                       # parallel unicast polls per epoch
    alpha: int = 3                   # quorum majority (2α > k ⇒ correct/wrong exclusive)
    beta: int = 5                    # consecutive-quorum streak to finalize
    max_poll_epochs: int = 20        # R_d = ⌊T_d / Δ_poll⌋

    # ---- macrostate basins (spec §4) ----
    correct_basin_mass: float = 0.60   # ρ_f  (> 1/2)
    split_basin_mass: float = 0.45     # ρ_s  (1 − ρ_f < ρ_s < 1/2)

    # ---- reliability budgets (hard constraints, spec §6) ----
    max_wrong_basin_probability: float = 1e-3     # ε_w
    max_split_basin_probability: float = 1e-3     # ε_s
    max_deadline_miss_probability: float = 1e-2   # ε_d

    # ---- objective (spec §6) ----
    latency_quantile: float = 0.95     # q for CVaR_q(T_confirm | O=C)
    energy_budget: float = math.inf    # report-only soft budget unless set finite

    def __post_init__(self) -> None:
        if self.participation_weight_rule not in _PARTICIPATION_RULES:
            raise ValueError(
                f"participation_weight_rule must be one of {_PARTICIPATION_RULES}, "
                f"got {self.participation_weight_rule!r}")
        if self.poll_window_ms <= 0:
            raise ValueError("poll_window_ms must be > 0")
        if self.k < 1:
            raise ValueError("k must be >= 1")
        if not (1 <= self.alpha <= self.k):
            raise ValueError("alpha must satisfy 1 <= alpha <= k")
        if 2 * self.alpha <= self.k:
            raise ValueError("alpha must be a strict majority of the quorum: 2*alpha > k")
        if self.beta < 1:
            raise ValueError("beta must be >= 1")
        if self.max_poll_epochs < 1:
            raise ValueError("max_poll_epochs must be >= 1")

        rho_f, rho_s = self.correct_basin_mass, self.split_basin_mass
        if not (rho_f > 0.5):
            raise ValueError("correct_basin_mass (rho_f) must be > 0.5 (decisive majority)")
        if not (rho_s < 0.5):
            raise ValueError("split_basin_mass (rho_s) must be < 0.5")
        # spec §4 disjointness: rho_s > 1 − rho_f  ⇒ correct/wrong & split basins never overlap
        if not (rho_s > 1.0 - rho_f):
            raise ValueError(
                "split_basin_mass (rho_s) must be > 1 - correct_basin_mass (rho_f) so the "
                "correct/wrong and split basins are disjoint (spec §4)")

        if not (0.0 < self.latency_quantile < 1.0):
            raise ValueError("latency_quantile must be in (0, 1)")
        for name in ("max_wrong_basin_probability", "max_split_basin_probability",
                     "max_deadline_miss_probability"):
            v = getattr(self, name)
            if not (0.0 < v < 1.0):
                raise ValueError(f"{name} must be in (0, 1)")
        if not (self.energy_budget > 0):
            raise ValueError("energy_budget must be > 0 (use math.inf for report-only)")

    # ---- convenience constructors ----

    @classmethod
    def urban_default(cls) -> "ConsensusServiceProfile":
        """The headline urban-V2X profile (recommended, override-flagged defaults)."""
        return cls()

    @classmethod
    def from_deadline(
        cls, *, deadline_ms: float, poll_window_ms: float = 10.0, **kwargs
    ) -> "ConsensusServiceProfile":
        """Build a profile whose ``max_poll_epochs = ⌊deadline_ms / poll_window_ms⌋`` (spec §5.3)."""
        r_d = int(math.floor(deadline_ms / poll_window_ms))
        return cls(poll_window_ms=poll_window_ms, max_poll_epochs=max(1, r_d), **kwargs)

    def replace(self, **changes) -> "ConsensusServiceProfile":
        """Return a validated copy with ``changes`` applied (frozen dataclass)."""
        return _dc_replace(self, **changes)

    # ---- derived quantities ----

    def epochs_for_deadline(self, deadline_ms: float) -> int:
        """``R_d = ⌊T_d / Δ_poll⌋`` (spec §5.3)."""
        return int(math.floor(deadline_ms / self.poll_window_ms))

    @property
    def poll_window_s(self) -> float:
        return self.poll_window_ms * 1e-3

    # ---- manifest hash (Mechanism Contract C5; plan §2) ----

    def to_dict(self) -> dict:
        d = asdict(self)
        # represent inf reproducibly across json round-trips
        if math.isinf(d["energy_budget"]):
            d["energy_budget"] = "inf"
        return d

    def config_hash(self) -> str:
        """Deterministic SHA-256 over all fields — the profile fingerprint for the manifest."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
