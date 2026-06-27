"""Fair CDQ 2.0 vs ESP mechanism-benefit factorial (Phase 10 / G-CDQ2-EVALUATION).

A reusable harness for the matched-marginal factorial judged by the INDEPENDENT dynamic-MC basin
outcomes (the headline judge -- NOT the analytic node-union F). ESP and CDQ 2.0 share the SAME
quality (the wrapped base policy), so ``ESP == CDQ2(eta=0)`` exactly and the ONLY difference is the
CDQ 2.0 diversity correction ``(eta, observable Z)`` -- the isolation the round requires.

Honest result (recorded in docs/gate_evidence/macrostate/, S15). CDQ 2.0 yields a GENUINE,
CI-separated, matched-marginal-controlled, correlation-SCOPED **P_correct** benefit (neutral in the
exchangeable/iid arm, a win only when covariance is present -- the plan's "mechanism effective and
scoped" rule). BUT the benefit flows through FASTER QUORUM (``F_deadline`` down), NOT the intended
correlation-avoidance: ``F_wrong`` is NOT reduced (it rises slightly). In a majority-correct regime,
diverse polling raises exposure to the minority correlated-wrong clusters, so it does not lower the
wrong-consensus basin. This harness reports ALL four basins so that boundary is never hidden.

The diversity embedding is built from DEPLOYMENT-OBSERVABLE group labels only (sensor / map / road
of the destination peer) -- never ``Y*`` / votes / sampled truth (Mechanism Contract C2).
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.sampling.cdq2_wiring import CDQ2Policy
from src.validation import run_dynamic_mc

__all__ = [
    "observable_group_diversity",
    "wilson_ci",
    "FactorialResult",
    "run_factorial_cell",
    "esp_vs_cdq2_cell",
]


def observable_group_diversity(model, *, use_sensor: bool = True, use_map: bool = True,
                               use_road: bool = False):
    """Closure ``graph -> [E, r]`` diversity embedding from DEPLOYMENT-OBSERVABLE group labels of
    the destination peer (one-hot of sensor / map / road group ids), concatenated.

    Uses ONLY the exogenous group labels -- never ``Y*``, votes, or sampled correctness (C2). Two
    models with the SAME group labels but DIFFERENT error probabilities produce the SAME embedding
    (the embedding is truth-distribution-independent). Co-group peers get parallel one-hots
    (``z_j . z_l > 0``), so the CDQ 2.0 kernel down-weights co-selecting them.
    """
    blocks = []
    if use_sensor:
        blocks.append((model.sensor_of, int(model.sensor_of.max()) + 1))
    if use_map:
        blocks.append((model.map_of, int(model.map_of.max()) + 1))
    if use_road:
        blocks.append((model.road_of, int(model.road_of.max()) + 1))
    if not blocks:
        raise ValueError("at least one observable group (sensor/map/road) must be used")

    def f(graph):
        dst = graph.dst_index
        return torch.cat([F.one_hot(g[dst], n) for g, n in blocks], dim=-1).to(torch.float64)

    return f


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a basin frequency ``p`` over ``n`` pooled trials."""
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


@dataclass(frozen=True)
class FactorialResult:
    """Pooled basin outcomes for one factorial cell (mean over CRN seeds)."""

    P_correct: float
    F_wrong: float
    F_split: float
    F_deadline: float
    n_pool: int

    def ci(self, field: str) -> tuple[float, float]:
        return wilson_ci(getattr(self, field), self.n_pool)

    def basins_sum(self) -> float:
        return self.P_correct + self.F_wrong + self.F_split + self.F_deadline

    def to_macro_block(self) -> dict:
        """Namespaced macrostate headline block (``macro_*``; macrostate_v2) with Wilson CIs."""
        from src.metrics import schema
        ci = {"macro_P_correct": self.ci("P_correct"), "macro_F_wrong": self.ci("F_wrong"),
              "macro_F_split": self.ci("F_split"), "macro_F_deadline": self.ci("F_deadline")}
        return schema.macro_block(self.P_correct, self.F_wrong, self.F_split, self.F_deadline, ci=ci)

    def to_result_record(self, *, policy: str, query_family: str, hashes=None) -> dict:
        """Full §7.4 result record (version + macro block + manifest hashes if provided)."""
        from src.metrics import schema
        return schema.build_result_record(policy=policy, query_family=query_family,
                                          macro=self.to_macro_block(),
                                          runtime={"runtime_n_pool": self.n_pool}, hashes=hashes)


def run_factorial_cell(scene, model, policy, proto, phy, profile, omega, *,
                       trials: int, seeds, link_override) -> FactorialResult:
    """Run the INDEPENDENT dynamic MC over CRN seeds; return pooled basin outcomes.

    The MC samples evidence + the k-DPP query subset + Bernoulli(ell) responses and reads peers'
    ACTUAL colours -- it does NOT sample the analytic terminal marginals (constraint #10).
    """
    rows = [run_dynamic_mc(scene, model, policy, proto, phy, num_trials=trials,
                           generator=torch.Generator().manual_seed(int(s)),
                           link_override=link_override, service_profile=profile, participation=omega)
            for s in seeds]
    mean = lambda key: statistics.mean([getattr(r, "basin_" + key) for r in rows])
    return FactorialResult(mean("P_correct"), mean("F_wrong"), mean("F_split"), mean("F_deadline"),
                           trials * len(list(seeds)))


def esp_vs_cdq2_cell(scene, model, base_policy, proto, phy, profile, omega, *, eta, diversity, r,
                     trials: int, seeds, link_override) -> tuple[FactorialResult, FactorialResult]:
    """One factorial cell: ESP (the bare ``base_policy``) vs CDQ 2.0 (same quality + diversity ``eta``).

    Returns ``(esp_result, cdq2_result)``. ESP == CDQ2(eta=0) by construction, so the difference is
    purely the diversity mechanism (a fair comparison; no quality advantage to either).
    """
    esp = run_factorial_cell(scene, model, base_policy, proto, phy, profile, omega,
                             trials=trials, seeds=seeds, link_override=link_override)
    cdq2_policy = CDQ2Policy(base_policy, r=r, eta=eta, diversity=diversity)
    cdq2 = run_factorial_cell(scene, model, cdq2_policy, proto, phy, profile, omega,
                              trials=trials, seeds=seeds, link_override=link_override)
    return esp, cdq2
