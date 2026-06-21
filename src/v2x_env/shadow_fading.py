"""Per-link AR(1) log-normal shadow fading as a HIDDEN, carried temporal state (P1-1.2).

The current sim has NO time-correlated hidden channel state: given (geometry, LOS) the link is fully
determined, so the environment is a fully-observable Markov process and a recurrent model has no
theoretical edge (run_environment_forensics.py measures markov-prediction error ~= 0). This module
adds the first real hidden temporal state: a per-link log-normal shadow offset (dB) that evolves as a
first-order autoregressive process across frames,

    s_t = rho * s_{t-1} + sqrt(1 - rho^2) * eps,    eps ~ N(0, std_db^2),  rho = exp(-dt / decorr_time),

so it is a stationary AR(1) with std `std_db` and lag-1 autocorrelation `rho`. Properties:
  * DETERMINISTIC given a seed (the draws are reproducible) -> a "carried state variable", not a
    Monte-Carlo sample inside training (the eps sequence is drawn by a dedicated RNG in the harness);
  * MODEL-UNOBSERVABLE -> it enters ONLY the evaluator channel (evaluate_v2x_graph_consensus's
    shadow_offset_db -> path loss -> SINR), never the node/edge features the scorer reads;
  * DIFFERENTIABLE-safe -> it is a fixed additive dB offset per edge per frame, so gradients still
    flow through SINR (it does NOT take the link_success-override branch).

Keying: node identity is stable across frames (fixed population), but the candidate EDGE set changes
per frame, so the AR(1) state is keyed by the undirected node PAIR (min(i,j), max(i,j)); a pair that
reappears after an absence continues from its last value advanced by the elapsed frames.
"""
from __future__ import annotations

import numpy as np


def advance_shadow_ar1(prev_db: np.ndarray, eps_db: np.ndarray, rho: float) -> np.ndarray:
    """One AR(1) step: s_t = rho*s_{t-1} + sqrt(1-rho^2)*eps. Pure (no RNG); caller supplies eps."""
    rho = float(np.clip(rho, 0.0, 1.0))
    return rho * np.asarray(prev_db, dtype=float) + np.sqrt(max(1.0 - rho * rho, 0.0)) * np.asarray(eps_db, dtype=float)


class ShadowField:
    """Carried per-node-pair AR(1) shadow state over a mobility frame stream.

    Usage (in the harness frame loop, frames consumed in increasing order):
        field = ShadowField(std_db=4.0, decorrelation_time_s=8.0, dt_s=2.0, seed=scene_seed)
        for t in range(num_frames):
            offset = field.frame_offsets(t, candidate.source, candidate.target)  # per candidate edge, dB
            ... evaluate_v2x_graph_consensus(..., shadow_offset_db=torch.as_tensor(offset)[selected])
    With std_db <= 0 the field is a no-op (all-zero offsets) -> byte-identical to no shadow.
    """

    def __init__(self, std_db: float, decorrelation_time_s: float, dt_s: float, seed: int = 0) -> None:
        self.std_db = float(std_db)
        self.dt_s = float(dt_s)
        self.decorrelation_time_s = float(decorrelation_time_s)
        # rho = exp(-dt/T) per frame; T<=0 -> rho=0 (independent draws); T->inf -> rho->1 (frozen).
        self.rho = float(np.exp(-self.dt_s / self.decorrelation_time_s)) if self.decorrelation_time_s > 0 else 0.0
        # Counter-based randomness: eps is a pure function of (seed, frame, pair) — see _pair_eps.
        self._seed = int(seed) & 0x7FFFFFFF
        self._state: dict[tuple[int, int], float] = {}
        self._last_frame: dict[tuple[int, int], int] = {}

    @property
    def enabled(self) -> bool:
        return self.std_db > 0.0

    def _pair_eps(self, frame_index: int, key: tuple[int, int]) -> float:
        """The AR(1) innovation for ``key`` at ``frame_index`` — a PURE function of
        (seed, frame, pair), independent of edge-array order, duplicates, or which other pairs
        exist (defect fix: a shared sequential RNG made the realised shadow an artifact of the
        edge-array construction). Counter-based: a fresh PCG64 keyed by the tuple per draw —
        O(E) generator inits per frame, fine at experiment scale (vectorise if N >= 1e4 hurts)."""
        rng = np.random.default_rng((self._seed, int(frame_index), int(key[0]), int(key[1])))
        return float(rng.normal(0.0, self.std_db))

    def frame_offsets(self, frame_index: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
        """Per-edge shadow offset (dB) for the given candidate edges at frame ``frame_index``.

        Advances every UNDIRECTED pair present in this frame by its elapsed-frame AR(1) step (new
        pairs start from a stationary draw); both directions of a pair share one value, computed
        once. Frames MUST be consumed in increasing order for the recursion to be meaningful.
        Returns zeros when the field is disabled (std_db <= 0)."""
        src = np.asarray(src, dtype=int)
        dst = np.asarray(dst, dtype=int)
        if not self.enabled or src.size == 0:
            return np.zeros(src.size, dtype=float)
        offsets = np.empty(src.size, dtype=float)
        for e in range(src.size):
            i, j = int(src[e]), int(dst[e])
            key = (i, j) if i <= j else (j, i)
            if self._last_frame.get(key) != int(frame_index):
                eps = self._pair_eps(frame_index, key)
                if key not in self._state:
                    value = eps  # stationary initial draw ~ N(0, std^2)
                else:
                    gap = max(int(frame_index) - self._last_frame[key], 1)
                    rho_gap = self.rho ** gap
                    value = rho_gap * self._state[key] + np.sqrt(max(1.0 - rho_gap * rho_gap, 0.0)) * eps
                self._state[key] = value
                self._last_frame[key] = int(frame_index)
            offsets[e] = self._state[key]
        return offsets
