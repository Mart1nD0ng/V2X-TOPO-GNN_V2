"""Continuous-mobility frame stream over a fixed vehicle population.

A :class:`MobilityStream` turns a single base snapshot into a reproducible sequence
of frames advanced in time. Frame ``k`` is the base snapshot advanced by ``k * dt``
seconds (linear motion along heading + toroidal wrap, RSUs fixed). Because linear
motion is composable under the toroidal wrap, ``frame_at(k)`` is computed directly
from the base (no incremental accumulation drift) and is fully deterministic.

Default (linear_toroidal, no turning, no churn): the population is FIXED and frame_at(k) is the
closed-form base advance (byte-identical to the original) — node identity is positional and stable.

P1-1.3 OPT-IN modes (config flags, default off):
  * intersection_markov turning and/or absorb_inject boundary churn switch the stream to STATEFUL
    incremental stepping (turning is path-dependent and churn accumulates, so frame_at(k) is no longer
    f(base, k*dt); frames are replayed-and-cached 0..k from a single seeded RNG).
  * NODE-IDENTITY CONTRACT under churn: ``node_id`` is the stable identity (NOT the positional index,
    which shifts when deaths compact the arrays). Survivors keep their node_id; births get fresh ids
    (max+1...) appended at the END. Downstream per-node recurrent/carried state MUST be re-indexed by
    node_id between frames (gather survivors, init births, drop deaths) — positional alignment is only
    valid in the no-churn default. Each frame carries ``births``/``deaths`` counts for diagnostics.
  * sumo mode delegates frame_at/len to an injected trace_source (no hard traci dependency by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Mapping

import numpy as np

from .vehicle_snapshot import advance_vehicle_snapshot, step_vehicle_snapshot

_REQUIRED_SNAPSHOT_KEYS = ("x", "y", "heading", "speed_mps", "bounds", "node_type")
# Extra keys the STATEFUL (turning/churn) path needs; the legacy closed-form path does not.
_STATEFUL_SNAPSHOT_KEYS = ("grid", "node_id", "road_segment_id", "lane_id")
_VALID_MOBILITY_MODES = ("linear_toroidal", "intersection_markov", "sumo")
_VALID_BOUNDARY_MODES = ("toroidal", "absorb_inject")


@dataclass(frozen=True)
class MobilityConfig:
    """Configuration for a continuous-mobility frame stream.

    Attributes:
        dt_s: time step between consecutive frames, seconds (> 0).
        num_frames: number of frames in the stream, including frame 0 (>= 1).
    P1-1.3 OPT-IN realism (all default to the legacy no-op so byte-identical-when-off holds):
        mobility_mode: "linear_toroidal" (default) | "intersection_markov" | "sumo".
        turn_probs: (straight, left, right) turn probabilities at intersections; default (1,0,0) = no turns.
        boundary_mode: "toroidal" (default wrap) | "absorb_inject" (out-of-bounds removal + boundary births).
        churn_rate_per_frame: Poisson mean births/frame at boundaries (absorb_inject only); 0 = no churn.
        turn_zone_m / stream_seed: turning trigger radius and RNG seed for the stochastic path.
    """

    dt_s: float = 2.0
    num_frames: int = 25
    mobility_mode: str = "linear_toroidal"
    turn_probs: tuple = (1.0, 0.0, 0.0)
    boundary_mode: str = "toroidal"
    churn_rate_per_frame: float = 0.0
    turn_zone_m: float = 8.0
    stream_seed: int = 0

    def __post_init__(self) -> None:
        if not self.dt_s > 0.0:
            raise ValueError("dt_s must be positive")
        if int(self.num_frames) != self.num_frames or self.num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        if self.mobility_mode not in _VALID_MOBILITY_MODES:
            raise ValueError(f"mobility_mode must be one of {_VALID_MOBILITY_MODES}")
        if self.boundary_mode not in _VALID_BOUNDARY_MODES:
            raise ValueError(f"boundary_mode must be one of {_VALID_BOUNDARY_MODES}")
        tp = tuple(float(v) for v in self.turn_probs)
        if len(tp) != 3 or any(v < 0.0 for v in tp) or abs(sum(tp) - 1.0) > 1e-6:
            raise ValueError("turn_probs must be 3 nonnegative values summing to 1 (straight,left,right)")
        if self.churn_rate_per_frame < 0.0:
            raise ValueError("churn_rate_per_frame must be nonnegative")

    @property
    def is_stateful(self) -> bool:
        """Whether the stream needs incremental stepping (turning / churn / non-toroidal / sumo)."""
        return (
            self.mobility_mode != "linear_toroidal"
            or (float(self.turn_probs[1]) + float(self.turn_probs[2])) > 0.0
            or self.boundary_mode == "absorb_inject"
            or self.churn_rate_per_frame > 0.0
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "MobilityConfig":
        data = data or {}
        tp = data.get("turn_probs", (1.0, 0.0, 0.0))
        return cls(
            dt_s=float(data.get("dt_s", cls.dt_s)),
            num_frames=int(data.get("num_frames", cls.num_frames)),
            mobility_mode=str(data.get("mobility_mode", cls.mobility_mode)),
            turn_probs=tuple(float(v) for v in tp),
            boundary_mode=str(data.get("boundary_mode", cls.boundary_mode)),
            churn_rate_per_frame=float(data.get("churn_rate_per_frame", cls.churn_rate_per_frame)),
            turn_zone_m=float(data.get("turn_zone_m", cls.turn_zone_m)),
            stream_seed=int(data.get("stream_seed", cls.stream_seed)),
        )


class MobilityStream:
    """Reproducible sequence of mobility frames over a fixed vehicle population.

    Args:
        base_snapshot: a snapshot produced by ``generate_vehicle_snapshot`` (t=0).
        config: a :class:`MobilityConfig` or a mapping accepted by
            :meth:`MobilityConfig.from_mapping`.
    """

    def __init__(
        self,
        base_snapshot: Mapping[str, object],
        config: MobilityConfig | Mapping[str, object] | None = None,
        *,
        trace_source: object | None = None,
    ) -> None:
        missing = [k for k in _REQUIRED_SNAPSHOT_KEYS if k not in base_snapshot]
        if missing:
            raise ValueError(f"base_snapshot is missing required keys: {missing}")
        self._base = base_snapshot
        self.config = config if isinstance(config, MobilityConfig) else MobilityConfig.from_mapping(config)
        self._population = int(len(base_snapshot["x"]))
        # P1-1.3: SUMO seam — when mobility_mode='sumo', delegate frame_at/num_frames to an injected
        # trace source with the canonical snapshot schema (no hard traci dependency on the default path).
        self._trace_source = trace_source
        if self.config.mobility_mode == "sumo" and trace_source is None:
            raise NotImplementedError(
                "mobility_mode='sumo' requires an injected trace_source (e.g. a SUMO/NS-3 adapter "
                "exposing frame_at(k) -> snapshot dict and __len__)"
            )
        # Stateful (turning/churn) path needs grid + per-node identity; the legacy path does not.
        if self.config.is_stateful and self.config.mobility_mode != "sumo":
            missing_stateful = [k for k in _STATEFUL_SNAPSHOT_KEYS if k not in base_snapshot]
            if missing_stateful:
                raise ValueError(
                    f"stateful mobility (turning/churn) needs base_snapshot keys {missing_stateful}; "
                    "build the base with generate_vehicle_snapshot"
                )
        self._frames: list[dict] | None = None
        self._rng = None

    @property
    def population(self) -> int:
        """Fixed number of nodes (vehicles + RSUs) present in every frame."""
        return self._population

    def __len__(self) -> int:
        return int(self.config.num_frames)

    def time_at(self, index: int) -> float:
        """Wall-clock time (seconds) of frame ``index``."""
        self._check_index(index)
        return float(index) * float(self.config.dt_s)

    def times(self) -> list[float]:
        """Times (seconds) of all frames in the stream."""
        return [self.time_at(k) for k in range(len(self))]

    def _ensure_frames(self, up_to: int) -> None:
        """Build & cache stateful frames 0..up_to incrementally (turning/churn are non-composable, so
        random access requires replay from 0; a single seeded RNG drives the realised sequence)."""
        if self._frames is None:
            self._frames = [dict(self._base)]
            self._rng = np.random.default_rng(int(self.config.stream_seed))
        while len(self._frames) <= up_to:
            self._frames.append(
                step_vehicle_snapshot(
                    self._frames[-1], self.config.dt_s, rng=self._rng,
                    turn_probs=self.config.turn_probs, boundary_mode=self.config.boundary_mode,
                    churn_rate_per_frame=self.config.churn_rate_per_frame, turn_zone_m=self.config.turn_zone_m,
                )
            )

    def frame_at(self, index: int) -> dict[str, object]:
        """Snapshot at frame ``index``. Legacy (linear+toroidal, no turn/churn): closed-form base advance.
        Stateful (turning/churn): incremental replay from frame 0 (cached). SUMO: delegated to the source."""
        self._check_index(index)
        if self.config.mobility_mode == "sumo":
            return self._trace_source.frame_at(index)
        if not self.config.is_stateful:
            if index == 0:
                return dict(self._base)
            return advance_vehicle_snapshot(self._base, self.time_at(index))
        self._ensure_frames(index)
        return self._frames[index]

    def __iter__(self) -> Iterator[dict[str, object]]:
        for index in range(len(self)):
            yield self.frame_at(index)

    def _check_index(self, index: int) -> None:
        if int(index) != index or index < 0 or index >= len(self):
            raise IndexError(f"frame index {index} out of range [0, {len(self)})")
