"""Temporal correlated-evidence sequence (Phase 12 / G-TEMPORAL).

A sequence of ``T`` overlapping-evidence epochs in which exactly one sensor band is the ACTIVE
correlated common cause; the active band follows a persistence process (a 2-state-style Markov
schedule):

* ``persistence = 1`` -> the SAME band is correlated every epoch (full persistence);
* ``persistence = 0`` -> a fresh uniform-random band each epoch (iid-in-time, the control);
* in between -> the band stays with probability ``persistence``, else re-draws.

**Matched-marginal-in-time (the §C1/C4 control).** Whichever band is active, its nodes split the
SAME marginal error between the shared (sensor) bit and the node bit via
:func:`matched_marginal_shared`, and non-active-band nodes carry that error in the node bit -- so
EVERY node has the SAME marginal correctness ``q_i`` in EVERY epoch, regardless of which band is
active. Only the COVARIANCE (which band is correlated, and whether it persists) changes. A
marginal-only policy is therefore blind to the temporal structure; only a covariance/diversity-aware
policy that REMEMBERS the persistent band can exploit it.

**Observable proxy (C2, no truth leak).** :meth:`observable_proxy` returns a per-band NOISY
MEASUREMENT of which band is currently cohesive (the deployment-observable correlation signal --
e.g. measured within-band response agreement), never ``Y*`` / the sampled bits. The held-out
:meth:`true_active_onehot` is EVALUATION-ONLY (for scoring how well a memory tracks the structure).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .overlapping_evidence import OverlappingEvidenceModel, matched_marginal_shared

__all__ = ["TemporalCorrelationSequence"]


@dataclass(frozen=True)
class TemporalCorrelationSequence:
    scene: object
    T: int
    persistence: float
    base_node_err: float = 0.35
    corr_strength: float = 0.3
    n_sensor: int = 3
    n_map: int = 3
    obs_noise: float = 0.3
    seed: int = 0

    def __post_init__(self):
        if not (0.0 <= self.persistence <= 1.0):
            raise ValueError("persistence must be in [0, 1]")
        if self.T < 1:
            raise ValueError("T must be >= 1")
        g = torch.Generator().manual_seed(self.seed)
        N = self.scene.num_nodes
        pos = self.scene.positions

        def band(coord, k):
            lo, hi = float(coord.min()), float(coord.max())
            if hi <= lo:
                return torch.zeros(N, dtype=torch.long)
            edges = torch.linspace(lo, hi, k + 1, dtype=coord.dtype)[1:-1].contiguous()
            return torch.bucketize(coord.contiguous(), edges).long().clamp(0, k - 1)

        object.__setattr__(self, "_sensor_of", band(pos[:, 0], self.n_sensor))
        object.__setattr__(self, "_map_of", band(pos[:, 1], self.n_map))
        # active-band schedule (Markov persistence)
        active = [int(torch.randint(self.n_sensor, (1,), generator=g))]
        for _ in range(1, self.T):
            stay = bool(torch.rand((), generator=g) < self.persistence)
            active.append(active[-1] if stay else int(torch.randint(self.n_sensor, (1,), generator=g)))
        object.__setattr__(self, "_active", active)
        # observation noise for the per-band proxy (precomputed, deterministic given seed)
        object.__setattr__(self, "_noise", torch.randn(self.T, self.n_sensor, generator=g,
                                                        dtype=torch.float64) * self.obs_noise)

    # ---- per-epoch evidence model (matched marginal across epochs) ----
    def model(self, t: int) -> OverlappingEvidenceModel:
        N = self.scene.num_nodes
        b = self._active[t]
        p_sensor, p_node_active = matched_marginal_shared(self.base_node_err, self.corr_strength)
        p_sens = torch.zeros(self.n_sensor, dtype=torch.float64)
        p_sens[b] = p_sensor
        in_active = self._sensor_of == b
        p_node = torch.where(in_active, torch.full((N,), p_node_active, dtype=torch.float64),
                             torch.full((N,), float(self.base_node_err), dtype=torch.float64))
        return OverlappingEvidenceModel(
            road_of=torch.zeros(N, dtype=torch.long), sensor_of=self._sensor_of, map_of=self._map_of,
            p_road=torch.tensor([0.0], dtype=torch.float64), p_sensor=p_sens,
            p_map=torch.zeros(self.n_map, dtype=torch.float64), p_node=p_node)

    # ---- observable (C2) per-band correlation proxy at epoch t (truth-free measurement) ----
    def observable_proxy(self, t: int) -> torch.Tensor:
        x = torch.zeros(self.n_sensor, dtype=torch.float64)
        x[self._active[t]] = 1.0
        return (x + self._noise[t]).clamp_min(0.0)            # measured cohesion + noise, >= 0

    def proxy_sequence(self) -> torch.Tensor:
        return torch.stack([self.observable_proxy(t) for t in range(self.T)], dim=0)   # [T, n_sensor]

    # ---- evaluation-only ground truth (NEVER on the query path) ----
    def true_active_onehot(self) -> torch.Tensor:
        oh = torch.zeros(self.T, self.n_sensor, dtype=torch.float64)
        for t in range(self.T):
            oh[t, self._active[t]] = 1.0
        return oh

    @property
    def sensor_of(self) -> torch.Tensor:
        return self._sensor_of

    @property
    def map_of(self) -> torch.Tensor:
        return self._map_of
