"""Effective-sampling-dynamics diagnostics (spec §5 -- G7, the project's namesake layer).

These are the interpretability + auxiliary-training signals (spec §5.8) computed from the
canonical round quantities; they DO NOT replace the headline safety/deadline metrics
(spec §5.7). All differentiable, all ``O(E)`` (or ``O(sum_i deg_i^2)`` for the per-source ESS
correlation) -- no ``N x N`` tensor.

* **response-conditioned marginal** (spec §5.2): ``pi~_ij = pi_ij ell_ij / sum_m pi_im ell_im``
  -- what Avalanche actually samples (planned ``pi`` thinned by physical delivery ``ell``).
* **progress / drift** (spec §5.4-§5.5): ``g_i = h^+ + h^-`` (a quorum that updates the
  protocol), ``Delta_i = h^+ - h^-`` (correct-direction drift), and the per-physical-time rates
  ``nu^prog = g/tau``, ``nu^drift = Delta/tau``. High ``g`` with ``Delta ~ 0`` = fast but
  conflicting evidence -- response rate alone is insufficient.
* **effective sample size** (spec §5.6): ``k_eff,i = 1 / (w_i^T R_i w_i)`` with ``w_i`` the
  normalised successful-response weights and ``R_i`` the candidate evidence-correlation matrix
  (``R_jj = 1``). Redundant (correlated) peers -> low ``k_eff``; diverse peers -> high.
* **mixing / weak cut** (spec §5.7): the region response kernel ``P^resp_ij = pi~_ij``
  aggregated to a region supergraph -> cross-region response mass + spectral gap (additive
  reversibilization). Weak cut -> little cross-region mass / small gap.
* **receiver load** ``Lambda`` (spec §5.8): re-exported from the round physics.
"""

from __future__ import annotations

import torch

from src.environment.evidence_model import EvidenceModel, pairwise_correlation_theory
from src.mainline.topology import receiver_load  # noqa: F401  (re-export Lambda_j = sum tau_i pi_ij)

__all__ = [
    "response_conditioned_marginal",
    "progress_drift",
    "effective_sample_size",
    "cross_region_response_mass",
    "region_response_kernel",
    "region_spectral_gap",
    "receiver_load",
]


def _scatter_src(values: torch.Tensor, src_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    out = values.new_zeros((num_nodes,))
    return out.index_add(0, src_index, values)


def response_conditioned_marginal(
    src_index: torch.Tensor, num_nodes: int, pi: torch.Tensor, ell: torch.Tensor,
    *, eps: float = 1e-12,
) -> torch.Tensor:
    """``pi~_ij = pi_ij ell_ij / (sum_m pi_im ell_im + eps)`` (spec §5.2). ``[E]``."""
    r = pi * ell
    denom = _scatter_src(r, src_index, num_nodes)
    return r / (denom[src_index] + eps)


def progress_drift(h_plus: torch.Tensor, h_minus: torch.Tensor, tau: torch.Tensor,
                   *, eps: float = 1e-12) -> dict[str, torch.Tensor]:
    """Progress ``g``, drift ``Delta`` and their per-physical-time rates (spec §5.4-§5.5)."""
    g = h_plus + h_minus
    drift = h_plus - h_minus
    taus = tau.clamp_min(eps)
    return {"progress": g, "drift": drift, "nu_prog": g / taus, "nu_drift": drift / taus}


def effective_sample_size(
    src_index: torch.Tensor, dst_index: torch.Tensor, num_nodes: int,
    response_weight: torch.Tensor, evidence: EvidenceModel, *, eps: float = 1e-12,
) -> torch.Tensor:
    """Per-node effective sample size ``k_eff_i = 1/(w_i^T R_i w_i)`` (spec §5.6). ``[N]``.

    ``w_i`` = ``response_weight`` (e.g. ``pi~``) normalised over source ``i``'s candidates;
    ``R_i`` = the candidates' correctness-correlation matrix (``pairwise_correlation_theory``).
    Per-source ``deg_i x deg_i`` matrices -- no ``N x N``.
    """
    dtype = response_weight.dtype
    k_eff = torch.zeros(num_nodes, dtype=dtype)
    src = src_index.tolist()
    dst = dst_index.tolist()
    by_src: dict[int, list[int]] = {}
    for e, s in enumerate(src):
        by_src.setdefault(s, []).append(e)
    for i, edges in by_src.items():
        J = [dst[e] for e in edges]
        w = response_weight[edges]
        wsum = float(w.sum())
        if wsum <= eps:
            continue
        w = (w / wsum).to(dtype)
        d = len(J)
        R = torch.eye(d, dtype=dtype)
        for a in range(d):
            for b in range(a + 1, d):
                rho = pairwise_correlation_theory(evidence, J[a], J[b])
                R[a, b] = R[b, a] = rho
        quad = float(w @ R @ w)
        k_eff[i] = 1.0 / max(quad, eps)
    return k_eff


def region_response_kernel(
    src_index: torch.Tensor, dst_index: torch.Tensor, region_of: torch.Tensor,
    pi_tilde: torch.Tensor, num_regions: int, *, node_mass: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Row-stochastic region response kernel ``P^resp`` aggregated to the region supergraph.

    ``P[g,h] = sum_{i in g, i->j, j in h} mass_i pi~_ij`` row-normalised. ``[G, G]``.
    """
    rg = region_of[src_index]
    rh = region_of[dst_index]
    mass = (node_mass[src_index] if node_mass is not None else torch.ones_like(pi_tilde)) * pi_tilde
    P = pi_tilde.new_zeros((num_regions, num_regions))
    idx = rg * num_regions + rh
    P = P.reshape(-1).index_add(0, idx, mass).reshape(num_regions, num_regions)
    P = P / (P.sum(dim=1, keepdim=True) + eps)
    return P


def cross_region_response_mass(
    src_index: torch.Tensor, dst_index: torch.Tensor, region_of: torch.Tensor,
    pi_tilde: torch.Tensor, *, node_mass: torch.Tensor | None = None, eps: float = 1e-12,
) -> torch.Tensor:
    """Fraction of response mass that crosses regions (spec §5.7). Scalar; low under a weak cut."""
    mass = (node_mass[src_index] if node_mass is not None else torch.ones_like(pi_tilde)) * pi_tilde
    cross = mass[region_of[src_index] != region_of[dst_index]].sum()
    total = mass.sum()
    return cross / (total + eps)


def region_spectral_gap(P: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Spectral gap ``1 - |lambda_2|`` of the additive reversibilization ``(P + P^T)/2`` (spec §5.7).

    Larger gap = better mixing across regions; a weak cut yields a near-zero gap.
    """
    M = 0.5 * (P + P.transpose(-1, -2))
    evals = torch.linalg.eigvalsh(M)               # ascending, real (symmetric)
    sorted_abs = torch.sort(evals.abs(), descending=True).values
    lam2 = sorted_abs[1] if sorted_abs.numel() > 1 else sorted_abs.new_zeros(())
    return (1.0 - lam2).clamp_min(0.0)
