"""Overlapping common-cause evidence model (spec §7 — G-CORRELATED-ENV).

Generalises the single-region model (``evidence_model.EvidenceModel``) to OVERLAPPING shared
common causes:

    O_i = Y* ⊕ B_road(i) ⊕ B_sensor(i) ⊕ B_map(i) ⊕ E_i,
    B_road[g] ~ Bern(p_road[g]),  B_sensor, B_map similarly,  E_i ~ Bern(p_node[i]).

Each node carries three EXOGENOUS, deployment-observable group labels (road segment, sensor
source, map source). Co-group peers share a common-cause bit, so their evidence errors are
positively correlated; because the three groupings overlap independently, two same-road peers
are NO LONGER exchangeable (they may or may not also share a sensor/map). This is the lever the
prior round lacked (D18 near-exchangeable region-block): a region-aware ESP policy sees only the
marginal correctness ``q_i``, while a diversity-aware (CDQ) policy can avoid peers that share
MULTIPLE common causes.

Closed forms (XOR of independent bits ⇒ product of ``±1`` signs ``σ = 1-2·bit``):

    μ_i  = ∏_k (1 - 2 p_k)   over node i's road/sensor/map/node bits;     q_i = (1 + μ_i)/2
    Cov(C_i, C_j) = μ_i μ_j (1/ρ_sh² − 1)/4,   ρ_sh = ∏_{groups SHARED by i,j}(1 - 2 p)
    Corr(C_i, C_j) = μ_i μ_j (1/ρ_sh² − 1) / √((1-μ_i²)(1-μ_j²))

(shared bits cancel in ``E[σ_i σ_j]`` since ``(1-2b)²=1``; node bits are never shared ⇒ no
shared groups gives ``ρ_sh=1`` ⇒ zero correlation). **Matched-marginal control**: ``q_i`` depends
only on ``μ_i = ∏(1-2p)``, so moving error mass between a shared group bit and the independent
node bit while preserving ``μ_i`` keeps the marginal IDENTICAL but changes ``ρ_sh`` (hence the
covariance) — the exact control that isolates the correlation/diversity mechanism (spec §C1).

Implements the duck-typed evidence interface (``correct_observation_prob``, ``sample``,
``analytic_scenarios``, ``num_nodes``/``num_regions``/``region_of``/``p_region``) so it drops into
the canonical episode and the dynamic MC unchanged. Truth (``Y*``, ``C_i``, the sampled bits) is
kept strictly separate from the observable group labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .evidence_model import EvidenceSample

__all__ = ["OverlappingEvidenceModel", "overlapping_pairwise_correlation",
           "overlapping_pairwise_correlation_matrix", "matched_marginal_shared",
           "build_overlapping_scenario", "OVERLAPPING_SCENARIOS"]

OVERLAPPING_SCENARIOS = ("iid", "single_road", "overlapping_sensor_source",
                         "matched_marginal_low", "matched_marginal_high")


def _vp(name, p):
    if bool(torch.any((p.detach() < -1e-9) | (p.detach() > 1 + 1e-9)).cpu()):
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class OverlappingEvidenceModel:
    road_of: torch.Tensor       # [N] long
    sensor_of: torch.Tensor     # [N] long
    map_of: torch.Tensor        # [N] long
    p_road: torch.Tensor        # [G_road]
    p_sensor: torch.Tensor      # [G_sensor]
    p_map: torch.Tensor         # [G_map]
    p_node: torch.Tensor        # [N]

    def __post_init__(self) -> None:
        N = self.p_node.numel()
        for nm, g, p in (("road", self.road_of, self.p_road), ("sensor", self.sensor_of, self.p_sensor),
                         ("map", self.map_of, self.p_map)):
            if g.ndim != 1 or g.numel() != N:
                raise ValueError(f"{nm}_of must be 1-D length N")
            if int(g.max()) >= p.numel() or int(g.min()) < 0:
                raise ValueError(f"{nm}_of indices must be in [0, len(p_{nm})-1]")
            _vp(f"p_{nm}", p)
        _vp("p_node", self.p_node)

    # ---- duck-typed evidence interface (spatial region == road group) ----
    @property
    def region_of(self) -> torch.Tensor:
        return self.road_of

    @property
    def p_region(self) -> torch.Tensor:
        return self.p_road

    @property
    def num_nodes(self) -> int:
        return int(self.p_node.numel())

    @property
    def num_regions(self) -> int:
        return int(self.p_road.numel())

    def has_correlated_evidence(self, *, tol: float = 1e-12) -> bool:
        """True iff ANY common cause (road OR sensor OR map) is active — so the trace's
        ``correlated_evidence`` sentinel (constraint #13) is accurate even when the correlation
        lives in the sensor/map groups (not just the road/region), e.g. ``matched_marginal_high``.
        """
        return bool(((self.p_road > tol).any() or (self.p_sensor > tol).any()
                     or (self.p_map > tol).any()).cpu())

    def _mu(self) -> torch.Tensor:
        """``μ_i = ∏_k (1-2p_k)`` over node i's road/sensor/map/node bits -> ``[N]``."""
        s = (1 - 2 * self.p_road[self.road_of]) * (1 - 2 * self.p_sensor[self.sensor_of]) \
            * (1 - 2 * self.p_map[self.map_of]) * (1 - 2 * self.p_node)
        return s

    def correct_observation_prob(self) -> torch.Tensor:
        """Marginal correctness ``q_i = (1 + μ_i)/2`` -> ``[N]``."""
        return (1 + self._mu()) / 2

    # ---- sampler (dynamic MC) ----
    def sample(self, num_samples: int, *, generator: torch.Generator | None = None,
               device: torch.device | None = None) -> EvidenceSample:
        device = device or self.p_node.device
        dt = self.p_node.dtype
        N = self.num_nodes

        def bits(p, idx):
            u = torch.rand(num_samples, p.numel(), generator=generator, device=device, dtype=dt)
            return (u < p.to(device).unsqueeze(0))[:, idx.to(device)]     # [S, N]

        b_road = bits(self.p_road, self.road_of)
        b_sensor = bits(self.p_sensor, self.sensor_of)
        b_map = bits(self.p_map, self.map_of)
        un = torch.rand(num_samples, N, generator=generator, device=device, dtype=dt)
        node_err = un < self.p_node.to(device).unsqueeze(0)               # [S, N]
        flip = b_road ^ b_sensor ^ b_map ^ node_err                      # XOR of all error bits
        correct = ~flip
        group_bits = torch.stack([b_road, b_sensor, b_map], dim=-1)      # [S, N, 3] (truth-derived)
        return EvidenceSample(correct=correct, region_bits=group_bits, node_err=node_err,
                              init_pref_correct=correct.to(dt))

    # ---- analytic shared-latent decomposition ----
    def analytic_scenarios(self, *, max_scenarios: int = 1 << 16, tol: float = 1e-12
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact decomposition ``(omega [Q], init_correct_pref [N, Q])`` over the joint config of
        all ACTIVE common-cause bits (road ∪ sensor ∪ map; ``tol<p<1-tol``). Given ``Z=q`` the
        nodes are conditionally independent, so the canonical analytic episode uses this as ``Z``.
        """
        dt = self.p_node.dtype
        device = self.p_node.device
        # enumerate active group bits across all three group types
        groups = []  # (probs_tensor, group_idx_of_node, global_group_id)
        for p, idx in ((self.p_road, self.road_of), (self.p_sensor, self.sensor_of), (self.p_map, self.map_of)):
            for g in range(p.numel()):
                if tol < float(p[g]) < 1 - tol:
                    groups.append((float(p[g]), idx, g))
        Ga = len(groups)
        Q = 1 << Ga
        if Q > max_scenarios:
            raise ValueError(f"2^{Ga}={Q} active common-cause scenarios exceeds max_scenarios="
                             f"{max_scenarios}; use the dynamic MC / reduced set (spec §7 boundary)")
        # deterministic base group-XOR per node from degenerate (p>=1-tol) bits
        base_xor = torch.zeros(self.num_nodes, dtype=torch.long, device=device)
        for p, idx in ((self.p_road, self.road_of), (self.p_sensor, self.sensor_of), (self.p_map, self.map_of)):
            for g in range(p.numel()):
                if float(p[g]) >= 1 - tol:
                    base_xor = base_xor ^ (idx == g).long()
        omega = torch.ones(Q, dtype=dt, device=device)
        gx = base_xor.unsqueeze(1).repeat(1, Q)                          # [N, Q] group-XOR per scenario
        for a, (pa, idx, _) in enumerate(groups):
            bit = ((torch.arange(Q, device=device) >> a) & 1)            # [Q] this group's bit
            omega = omega * torch.where(bit.bool(), torch.tensor(pa, dtype=dt, device=device),
                                        torch.tensor(1 - pa, dtype=dt, device=device))
            member = (idx.to(device) == groups[a][2]).long().unsqueeze(1)  # [N,1] node in this group?
            gx = gx ^ (member * bit.unsqueeze(0))                        # toggle group-XOR where member & bit
        omega = omega / omega.sum()
        gxf = (gx & 1).to(dt)                                            # [N, Q] parity of shared error
        pi = self.p_node.unsqueeze(1)
        init_cp = (1 - gxf) * (1 - pi) + gxf * pi                       # P(correct|Z): E_i must match group-XOR
        return omega, init_cp


def overlapping_pairwise_correlation(model: OverlappingEvidenceModel, i: int, j: int) -> float:
    """Exact ``Corr(C_i, C_j)`` (correctness indicators) under the overlapping model.

    Uses the DIRECT cross-moment ``E[σ_i σ_j] = (∏_{unshared groups} s_i s_j)·s_node_i s_node_j``
    (a SHARED group contributes 1 because ``(1-2b)²=1``), avoiding the ``μ/ρ_sh`` division that is
    singular when a shared common cause is pure noise (``p=0.5`` ⇒ sign 0). Correct everywhere.
    """
    if i == j:
        return 1.0
    mu = model._mu()
    mu_i, mu_j = float(mu[i]), float(mu[j])
    var_i, var_j = 1.0 - mu_i * mu_i, 1.0 - mu_j * mu_j
    if var_i <= 0 or var_j <= 0:
        return 0.0
    e_ss = (1 - 2 * float(model.p_node[i])) * (1 - 2 * float(model.p_node[j]))  # node bits (never shared)
    for p, idx in ((model.p_road, model.road_of), (model.p_sensor, model.sensor_of),
                   (model.p_map, model.map_of)):
        if int(idx[i]) != int(idx[j]):                                  # unshared -> both signs
            e_ss *= (1 - 2 * float(p[int(idx[i])])) * (1 - 2 * float(p[int(idx[j])]))
        # shared group contributes a factor of 1 (the common bit cancels)
    cov = e_ss - mu_i * mu_j
    return cov / (var_i * var_j) ** 0.5


def overlapping_pairwise_correlation_matrix(model: OverlappingEvidenceModel, *, eps: float = 1e-18
                                            ) -> torch.Tensor:
    """Full ``[N, N]`` correlation matrix (vectorised; for ESS / diversity diagnostics).

    Same direct cross-moment as the scalar (NaN-free at ``p=0.5`` shared bits): ``E[σσ]`` keeps a
    factor of 1 for shared groups and ``s_i s_j`` for unshared ones — no ``1/ρ_sh²`` division.
    """
    mu = model._mu()                                                    # [N]
    s_node = 1 - 2 * model.p_node
    e_ss = s_node.unsqueeze(0) * s_node.unsqueeze(1)                    # [N,N] node bits (unshared)
    for p, idx in ((model.p_road, model.road_of), (model.p_sensor, model.sensor_of),
                   (model.p_map, model.map_of)):
        same = idx.unsqueeze(0) == idx.unsqueeze(1)
        s = (1 - 2 * p[idx])                                            # [N]
        e_ss = e_ss * torch.where(same, torch.ones_like(e_ss), s.unsqueeze(0) * s.unsqueeze(1))
    cov = e_ss - mu.unsqueeze(0) * mu.unsqueeze(1)
    var = (1 - mu * mu).clamp_min(eps)
    corr = cov / (var.unsqueeze(0) * var.unsqueeze(1)).sqrt()
    corr.fill_diagonal_(1.0)
    return corr


def build_overlapping_scenario(
    scene,
    name: str,
    *,
    base_node_err: float = 0.1,
    corr_strength: float = 0.25,
    n_sensor: int = 3,
    n_map: int = 3,
) -> OverlappingEvidenceModel:
    """Build an :class:`OverlappingEvidenceModel` on ``scene`` (spec §7 scenario matrix).

    Road groups = the scene's spatial road segments; sensor / map groups are CROSSCUTTING SPATIAL
    BANDS (sensor = x-band, map = y-band) that cut across road segments, so a same-road pair may or
    may not also share a sensor/map — the overlap that breaks exchangeability. The matched-marginal
    scenarios require ``corr_strength <= base_node_err`` (a shared bit cannot carry more error than
    the node's total marginal error). The marginal correctness is held FIXED across the correlation
    by the matched-marginal split (``matched_marginal_shared``), so ``iid`` /
    ``matched_marginal_low`` / ``matched_marginal_high`` share the SAME ``q_i`` with rising
    covariance — the spec §C1 causal control. Scenarios:

    * ``iid`` — group probs 0; only independent node error (zero correlation control).
    * ``single_road`` — only road group 0 is biased (legacy single-region analogue).
    * ``overlapping_sensor_source`` — crosscutting sensor+map common causes active.
    * ``matched_marginal_low`` / ``matched_marginal_high`` — IDENTICAL marginal, low/high shared
      covariance (the matched-marginal pair).
    """
    N = scene.num_nodes
    dt = torch.float64
    road = scene.region_of.clone()
    Gr = int(road.max()) + 1

    # Crosscutting spatial bands must be invariant to RSU geometry: RSU (responder/witness)
    # roadside positions can extend the coordinate extrema and silently shift the band edges,
    # which would perturb the VEHICLE-VEHICLE covariance -- the exact quantity the matched-marginal
    # control isolates (spec §C1). So derive the band edges from VEHICLE positions only (the
    # exogenous node_type role label, 0=vehicle), then bucketize ALL nodes against those fixed
    # edges. With no node_type (the ManhattanScene path) every node is a vehicle -> identical to
    # the previous all-node binning (behaviour-preserving).
    node_type = getattr(scene, "node_type", None)
    veh_mask = (node_type == 0) if node_type is not None else None

    def _band(coord: torch.Tensor, k: int) -> torch.Tensor:
        cv = coord if veh_mask is None else coord[veh_mask]
        lo, hi = float(cv.min()), float(cv.max())
        if hi <= lo:
            return torch.zeros(N, dtype=torch.long)
        edges = torch.linspace(lo, hi, k + 1, dtype=coord.dtype)[1:-1].contiguous()
        return torch.bucketize(coord.contiguous(), edges).long().clamp(0, k - 1)

    # crosscutting SPATIAL bands: sensor = x-band, map = y-band (cut across road segments)
    sensor = _band(scene.positions[:, 0], n_sensor)
    map_ = _band(scene.positions[:, 1], n_map)
    zeros_r = torch.zeros(Gr, dtype=dt)
    zeros_s = torch.zeros(n_sensor, dtype=dt)
    zeros_m = torch.zeros(n_map, dtype=dt)

    def model(p_road, p_sensor, p_map, p_node):
        return OverlappingEvidenceModel(
            road_of=road, sensor_of=sensor, map_of=map_,
            p_road=p_road, p_sensor=p_sensor, p_map=p_map,
            p_node=torch.full((N,), float(p_node), dtype=dt))

    if name == "iid":
        return model(zeros_r.clone(), zeros_s.clone(), zeros_m.clone(), base_node_err)
    if name == "single_road":
        pr = zeros_r.clone(); pr[0] = corr_strength
        return model(pr, zeros_s.clone(), zeros_m.clone(), base_node_err)
    if name == "overlapping_sensor_source":
        return model(zeros_r.clone(), torch.full((n_sensor,), corr_strength, dtype=dt),
                     torch.full((n_map,), corr_strength, dtype=dt), base_node_err)
    if name in ("matched_marginal_low", "matched_marginal_high"):
        if name == "matched_marginal_low":
            # all error in the independent node bit (zero covariance), marginal set by base_node_err
            return model(zeros_r.clone(), zeros_s.clone(), zeros_m.clone(), base_node_err)
        # same marginal, but split into a shared sensor bit (positive covariance)
        p_sensor, p_node_new = matched_marginal_shared(base_node_err, p_shared=corr_strength)
        return model(zeros_r.clone(), torch.full((n_sensor,), p_sensor, dtype=dt),
                     zeros_m.clone(), p_node_new)
    raise ValueError(f"unknown overlapping scenario {name!r}; expected one of {OVERLAPPING_SCENARIOS}")


def matched_marginal_shared(p_node_target: float, p_shared: float) -> tuple[float, float]:
    """Split a node's marginal error between a SHARED group bit and the independent node bit so the
    MARGINAL is preserved (spec §C1 matched-marginal control).

    Returns ``(p_shared, p_node_new)`` with ``(1-2 p_shared)(1-2 p_node_new) = (1-2 p_node_target)``,
    i.e. the same ``μ`` (hence the same ``q``) but now co-group peers are correlated via the shared bit.

    Feasibility is EXACTLY ``|1-2 p_node_target| <= |1-2 p_shared|`` (so the residual node sign stays a
    valid probability). For the common case ``p_shared, p_node_target ∈ [0, 1/2)`` this is just
    ``p_shared <= p_node_target`` ("a shared bit cannot carry more error than the node's marginal");
    a systematically-biased shared sensor (``p_shared > 1/2``) is also admissible when it satisfies
    the sign-magnitude condition. Raises if the inputs are out of ``[0, 1]`` or the split is infeasible.
    """
    if not (0.0 <= p_node_target <= 1.0) or not (0.0 <= p_shared <= 1.0):
        raise ValueError("p_node_target and p_shared must be in [0, 1]")
    s_target = 1 - 2 * p_node_target
    s_shared = 1 - 2 * p_shared
    if abs(s_shared) < 1e-12:
        raise ValueError("p_shared too close to 0.5 (a pure-noise shared bit carries no information)")
    s_node_new = s_target / s_shared
    if not (-1.0 <= s_node_new <= 1.0):
        raise ValueError(
            f"infeasible matched-marginal split: |1-2*p_node_target| ({abs(s_target):.3f}) must be "
            f"<= |1-2*p_shared| ({abs(s_shared):.3f}) (for p<1/2: p_shared <= p_node_target)")
    return p_shared, (1 - s_node_new) / 2
