"""Region/node shared-error evidence model (spec §6.2).

Each consensus instance has binary ground truth ``Y* ∈ {+1,-1}``. Node ``i`` belongs to
region ``g(i)`` (road segment / intersection / visibility region). Its raw observation is

    O_i = Y* ⊕ B_{g(i)} ⊕ E_i,        B_g ~ Bern(p_g),   E_i ~ Bern(p_i)         (spec §6.2)

where ``B_g`` is a **region-level shared error** (occlusion, common sensor/map source)
and ``E_i`` an independent node error. The protocol works on colours; correctness is the
labelling relative to ``Y*`` (``+`` = aligned with ``Y*``). The correctness indicator

    C_i = 1[ O_i = Y* ] = 1[ B_{g(i)} ⊕ E_i = 0 ] = 1[ B_{g(i)} = E_i ]

is what drives the initial preference (a node starts with ``pref = O_i``). Because all
nodes in a region share ``B_g``, their observations are **correlated within a region and
independent across regions** — exactly the structure needed to study sampling
representativeness and evidence diversity (spec §5, §6.3). The shared region bits are the
analytic evaluator's shared latent ``Z`` (spec §3.1): *given* the region-bit vector the
node observations are conditionally independent, so

    omega_r            = prod_g p_g^{b_g^{(r)}} (1-p_g)^{1-b_g^{(r)}}              (scenario weight)
    init_correct_pref  = P(C_i = 1 | Z=r) = (1-p_i) if b_{g(i)}^{(r)}=0 else p_i

This module gives both the per-instance sampler (for the dynamic MC, G6) and the exact
scenario decomposition + pairwise-correlation theory (for G2 validation). The query
policy never sees ``Y*`` or any vote — only the region id / geometry / credibility
(spec §6.3); enforcement lives in the model interface (Phase 9), here we merely keep
truth (``Y*``, ``C_i``) and the observable region structure in separate return fields.

Exactness boundary. The analytic scenario decomposition enumerates the ``2^G`` region-bit
configurations; it is exact but only tractable for a modest number of regions ``G``.
Large-``N`` deployments with many regions use the analytic path on a reduced dominant
-scenario set while the **dynamic MC samples region bits directly** (the final judge,
spec §8). ``analytic_scenarios`` raises above ``max_scenarios`` rather than silently
truncating (plan §17 prohibition #15).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["EvidenceModel", "EvidenceSample", "pairwise_correlation_theory"]


@dataclass(frozen=True)
class EvidenceSample:
    """One batch of sampled instances (truth and observables kept separate)."""

    correct: torch.Tensor       # [S, N] bool  C_i  (TRUTH-derived; not policy-visible)
    region_bits: torch.Tensor   # [S, G] bool  B_g  (TRUTH-derived)
    node_err: torch.Tensor      # [S, N] bool  E_i  (TRUTH-derived)
    init_pref_correct: torch.Tensor  # [S, N] float in {0,1}: 1 if node starts on '+'

    @property
    def num_samples(self) -> int:
        return int(self.correct.shape[0])


def _validate_probs(name: str, p: torch.Tensor) -> None:
    if bool(torch.any((p.detach() < -1e-9) | (p.detach() > 1 + 1e-9)).cpu()):
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class EvidenceModel:
    """Region/node shared-error evidence model.

    Attributes:
        region_of: ``[N]`` long, region id ``g(i) in {0..G-1}`` for each node.
        p_region: ``[G]`` region-level error probabilities ``p_g``.
        p_node: ``[N]`` node-level independent error probabilities ``p_i``.
    """

    region_of: torch.Tensor
    p_region: torch.Tensor
    p_node: torch.Tensor

    def __post_init__(self) -> None:
        if self.region_of.ndim != 1 or self.p_node.ndim != 1 or self.p_region.ndim != 1:
            raise ValueError("region_of, p_node, p_region must be 1-D")
        if self.region_of.numel() != self.p_node.numel():
            raise ValueError("region_of and p_node must have length N")
        if int(self.region_of.max()) >= self.p_region.numel() or int(self.region_of.min()) < 0:
            raise ValueError("region_of indices must be in [0, G-1]")
        _validate_probs("p_region", self.p_region)
        _validate_probs("p_node", self.p_node)

    @property
    def num_nodes(self) -> int:
        return int(self.region_of.numel())

    @property
    def num_regions(self) -> int:
        return int(self.p_region.numel())

    # ---------------------------------------------------------------- marginals
    def correct_observation_prob(self) -> torch.Tensor:
        """Marginal ``q_i = P(C_i = 1) = (1-p_g)(1-p_i) + p_g p_i`` -> ``[N]``."""
        pg = self.p_region[self.region_of]
        pi = self.p_node
        return (1 - pg) * (1 - pi) + pg * pi

    # ----------------------------------------------------------- sampler (MC)
    def sample(self, num_samples: int, *, generator: torch.Generator | None = None,
               device: torch.device | None = None) -> EvidenceSample:
        """Draw ``num_samples`` independent instances (for the dynamic MC, G6).

        ``Y*`` is taken as ``+1`` (the correctness frame is ``Y*``-symmetric; the dynamic
        MC may relabel colours per instance without changing any correctness statistic).
        """
        device = device or self.region_of.device
        G, N = self.num_regions, self.num_nodes
        pg = self.p_region.to(device)
        pi = self.p_node.to(device)
        ub = torch.rand(num_samples, G, generator=generator, device=device, dtype=pg.dtype)
        un = torch.rand(num_samples, N, generator=generator, device=device, dtype=pi.dtype)
        region_bits = ub < pg.unsqueeze(0)           # [S, G]
        node_err = un < pi.unsqueeze(0)              # [S, N]
        b_node = region_bits[:, self.region_of.to(device)]  # [S, N]
        correct = b_node == node_err                 # C_i = 1[B_g = E_i]
        init_pref_correct = correct.to(pi.dtype)     # pref = '+' iff observation correct
        return EvidenceSample(
            correct=correct,
            region_bits=region_bits,
            node_err=node_err,
            init_pref_correct=init_pref_correct,
        )

    # ------------------------------------------------- analytic scenario decomp
    def analytic_scenarios(self, *, max_scenarios: int = 1 << 16, tol: float = 1e-12
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exact shared-latent decomposition ``(omega [Q], init_correct_pref [N, Q])``.

        Only the **non-degenerate** regions (``tol < p_g < 1-tol``) are enumerated; a region
        with ``p_g = 0`` always has ``B_g = 0`` and one with ``p_g = 1`` always ``B_g = 1``,
        so they need no enumeration. Hence ``Q = 2^{#active regions}`` -- e.g. an iid scene
        (all ``p_g = 0``) is a single scenario, a one-biased-region scene is two. Raises if
        ``Q`` exceeds ``max_scenarios`` (no silent truncation; use the dynamic MC / a reduced
        scenario set for many simultaneously-biased regions, spec §6 boundary).
        """
        G, N = self.num_regions, self.num_nodes
        pg = self.p_region
        pi = self.p_node
        dtype = pi.dtype
        device = pi.device
        active = [g for g in range(G) if tol < float(pg[g]) < 1.0 - tol]
        Ga = len(active)
        Q = 1 << Ga
        if Q > max_scenarios:
            raise ValueError(
                f"2^{Ga} = {Q} active-region scenarios exceeds max_scenarios={max_scenarios}; "
                f"use the dynamic MC or a reduced dominant-scenario set (spec §6 boundary)"
            )
        # deterministic base bits for degenerate regions
        base_bit = (pg >= 1.0 - tol).to(dtype)                       # [G]
        bits = base_bit.unsqueeze(0).repeat(Q, 1)                    # [Q, G]
        if Ga > 0:
            idx = torch.arange(Q, device=device)
            sub = ((idx.unsqueeze(1) >> torch.arange(Ga, device=device).unsqueeze(0)) & 1).to(dtype)  # [Q,Ga]
            active_idx = torch.tensor(active, device=device, dtype=torch.long)
            bits[:, active_idx] = sub
        # omega_r = prod over active regions of pg^b (1-pg)^(1-b)  (degenerate -> factor 1)
        if Ga > 0:
            pga = pg[active_idx]                                     # [Ga]
            ba = bits[:, active_idx]                                 # [Q, Ga]
            log_w = ba * torch.log(pga.clamp_min(1e-300)) + (1 - ba) * torch.log((1 - pga).clamp_min(1e-300))
            omega = torch.exp(log_w.sum(dim=1))
            omega = omega / omega.sum()
        else:
            omega = torch.ones(1, dtype=dtype, device=device)
        b_node = bits[:, self.region_of].transpose(0, 1)            # [N, Q]
        init_cp = (1 - b_node) * (1 - pi).unsqueeze(1) + b_node * pi.unsqueeze(1)  # [N, Q]
        return omega, init_cp


def pairwise_correlation_theory(model: EvidenceModel, i: int, j: int) -> float:
    """Exact Pearson correlation of correctness indicators ``Corr(C_i, C_j)`` (spec §6.2).

    Same region: positive correlation via the shared ``B_g``. Different regions: ``0``.
    """
    if i == j:
        return 1.0
    gi = int(model.region_of[i])
    gj = int(model.region_of[j])
    pi = float(model.p_node[i])
    pj = float(model.p_node[j])
    q = model.correct_observation_prob()
    qi, qj = float(q[i]), float(q[j])
    var_i = qi * (1 - qi)
    var_j = qj * (1 - qj)
    if var_i <= 0 or var_j <= 0:
        return 0.0
    if gi != gj:
        return 0.0  # independent region bits -> Cov = 0
    pg = float(model.p_region[gi])
    e_cc = (1 - pg) * (1 - pi) * (1 - pj) + pg * pi * pj
    cov = e_cc - qi * qj
    return cov / (var_i * var_j) ** 0.5
