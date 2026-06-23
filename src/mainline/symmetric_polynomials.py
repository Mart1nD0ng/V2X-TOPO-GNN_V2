"""Exact weighted distinct-peer k-subset policy (spec §4, Eqs. 15-21).

The GNN emits a logit ``s_{ij}`` per physically-reachable candidate edge and the
unnormalised query weight is ``a_{ij} = exp(s_{ij}) > 0`` (Eq. 15).  Each polling
round selects *exactly* ``k_poll`` distinct neighbours from the candidate set
``N_i`` with the elementary-symmetric-polynomial distribution

    P(S_i = S) = (prod_{j in S} a_{ij}) / e_k(a_i),      |S| = k        (Eq. 16)

where ``e_k`` is the k-th elementary symmetric polynomial (Eq. 17).  There is no
hard top-k support, no fixed degree cap, and every physical candidate edge stays
on the differentiable path.

This module provides, all in a numerically-stable log domain:

* :func:`log_elementary_symmetric` / :func:`elementary_symmetric` -- the e-table.
* :func:`subset_log_probability` -- Eq. 16 for an explicit subset.
* :func:`edge_inclusion_probability` -- the marginal pi_{ij} (Eq. 18), with the
  exact ``sum_j pi_{ij} = k`` identity (Eq. 19).
* :func:`sample_k_subset` -- the *single* exact ancestral sampler used for both
  training and deployment (spec §4.1: "训练与部署使用同一随机查询分布").
* :func:`verify_sampler_matches_distribution` -- the code-level assertion that the
  sampler reproduces Eq. 16 exactly.

Representation.  Per-source candidate weights are passed as a padded
``[B, n_max]`` tensor with a boolean ``mask`` marking valid candidates.  Padded
entries carry weight ``0`` (``log_weight = -inf``), which contributes nothing to
any elementary symmetric polynomial -- so padding is mathematically exact, not an
approximation.  The physically-sparse candidate construction (no ``N x N`` dense
tensor, Eq. spec §7) is the responsibility of the graph-construction layer (G4);
here ``B`` is the number of polling sources and ``n_max`` the padded degree.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from typing import Sequence

import torch

NEG_INF = float("-inf")


class _LogAddExp(torch.autograd.Function):
    """``logaddexp`` with a well-defined (zero) gradient when both inputs are ``-inf``.

    ``torch.logaddexp(-inf, -inf)`` is ``-inf`` in the forward pass but yields a
    ``0/0 = nan`` gradient, which poisons the elementary-symmetric DP whenever an
    unreachable / masked order is combined with another ``-inf``.  Here the softmax
    backward weights are forced to ``0`` wherever the total mass is ``0``.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m = torch.maximum(a, b)
        m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        ea = torch.exp(a - m_safe)
        eb = torch.exp(b - m_safe)
        s = ea + eb
        out = m_safe + torch.log(s)
        wa = torch.where(s > 0, ea / s, torch.zeros_like(s))
        wb = torch.where(s > 0, eb / s, torch.zeros_like(s))
        ctx.save_for_backward(wa, wb)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        wa, wb = ctx.saved_tensors
        return grad_out * wa, grad_out * wb


def _logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _LogAddExp.apply(a, b)

__all__ = [
    "SubsetPolicy",
    "edge_inclusion_log_probability",
    "edge_inclusion_probability",
    "elementary_symmetric",
    "enumerate_subset_distribution",
    "log_elementary_symmetric",
    "sample_k_subset",
    "subset_log_probability",
    "verify_sampler_matches_distribution",
]


def _check_k(k: int) -> int:
    if not isinstance(k, int):
        raise TypeError("k must be an int")
    if k < 0:
        raise ValueError("k must be nonnegative")
    return k


def _prepare_log_weights(
    log_weights: torch.Tensor,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(log_w, mask)`` with masked-out entries forced to ``-inf``."""
    if not isinstance(log_weights, torch.Tensor):
        raise TypeError("log_weights must be a torch.Tensor")
    if not torch.is_floating_point(log_weights):
        raise ValueError("log_weights must use a floating-point dtype")
    if log_weights.ndim < 1:
        raise ValueError("log_weights must have a trailing candidate dimension")
    if mask is None:
        mask = torch.ones_like(log_weights, dtype=torch.bool)
    else:
        if mask.shape != log_weights.shape:
            raise ValueError("mask must match log_weights shape")
        mask = mask.to(dtype=torch.bool, device=log_weights.device)
    neg = log_weights.new_full((), NEG_INF)
    log_w = torch.where(mask, log_weights, neg)
    return log_w, mask


def log_elementary_symmetric(
    log_weights: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Log elementary symmetric polynomials ``log e_0 .. log e_k`` (Eq. 17).

    Args:
        log_weights: ``[..., n]`` log candidate weights ``s_{ij}`` (``a = exp(s)``).
        k: highest polynomial order to return.
        mask: optional ``[..., n]`` boolean mask of valid candidates.

    Returns:
        ``[..., k+1]`` tensor; entry ``m`` is ``log e_m``.  ``e_0 = 1`` so the
        first column is ``0``.  Differentiable w.r.t. ``log_weights``.

    The recurrence is the standard add-one-element convolution
    ``e_m <- e_m + a_j * e_{m-1}`` evaluated in the log domain via ``logaddexp``,
    which is stable for the large/small ``a_{ij} = exp(s_{ij})`` produced by a GNN.
    """
    k = _check_k(k)
    log_w, _ = _prepare_log_weights(log_weights, mask)
    batch = log_w.shape[:-1]
    n = log_w.shape[-1]
    le = log_w.new_full((*batch, k + 1), NEG_INF)
    le[..., 0] = 0.0  # log e_0 = log 1
    if k == 0:
        return le
    neg_col = log_w.new_full((*batch, 1), NEG_INF)
    for j in range(n):
        lw = log_w[..., j : j + 1]  # [..., 1]
        # shifted[..., m] = le[..., m-1]; shifted[..., 0] = -inf
        shifted = torch.cat([neg_col, le[..., :k]], dim=-1)
        le = _logaddexp(le, lw + shifted)
    return le


def elementary_symmetric(
    weights: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Elementary symmetric polynomials ``e_0 .. e_k`` in the linear domain.

    Convenience wrapper around :func:`log_elementary_symmetric` for callers that
    hold ``a_{ij}`` directly (rather than logits).  Prefer the log-domain API for
    GNN logits.
    """
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if bool(torch.any(weights.detach() < 0).cpu()):
        raise ValueError("weights must be nonnegative")
    log_w = torch.log(weights.clamp_min(0))
    return torch.exp(log_elementary_symmetric(log_w, k, mask=mask))


def subset_log_probability(
    log_weights: torch.Tensor,
    subset: Sequence[int] | torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """``log P(S_i = subset)`` under Eq. 16 for a single (unbatched) source.

    Args:
        log_weights: ``[n]`` log weights for one source.
        subset: indices of the chosen candidates; must have length ``k``.
        k: subset size ``k_poll``.
    """
    k = _check_k(k)
    if log_weights.ndim != 1:
        raise ValueError("subset_log_probability expects a single source ([n])")
    idx = torch.as_tensor(list(subset), dtype=torch.long, device=log_weights.device)
    if idx.numel() != k:
        raise ValueError("subset must contain exactly k indices")
    if torch.unique(idx).numel() != idx.numel():
        raise ValueError("subset must contain distinct candidates")
    log_w, m = _prepare_log_weights(log_weights, mask)
    if bool((~m[idx]).any().cpu()):
        raise ValueError("subset references a masked-out candidate")
    log_ek = log_elementary_symmetric(log_w, k)[..., k]
    return log_w[idx].sum() - log_ek


def _log_prefix_suffix(
    log_w: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cumulative log e-tables.

    Returns ``(prefix, suffix)`` each of shape ``[..., n+1, k+1]`` where
    ``prefix[..., t, :]`` is the log e-table of candidates ``[0, t)`` and
    ``suffix[..., t, :]`` is the log e-table of candidates ``[t, n)``.
    """
    batch = log_w.shape[:-1]
    n = log_w.shape[-1]
    neg_col = log_w.new_full((*batch, 1), NEG_INF)
    base = log_w.new_full((*batch, k + 1), NEG_INF)
    base[..., 0] = 0.0

    prefix = [base]
    cur = base
    for j in range(n):
        lw = log_w[..., j : j + 1]
        shifted = torch.cat([neg_col, cur[..., :k]], dim=-1)
        cur = _logaddexp(cur, lw + shifted)
        prefix.append(cur)

    suffix_rev = [base]
    cur = base
    for j in range(n - 1, -1, -1):
        lw = log_w[..., j : j + 1]
        shifted = torch.cat([neg_col, cur[..., :k]], dim=-1)
        cur = _logaddexp(cur, lw + shifted)
        suffix_rev.append(cur)
    suffix = list(reversed(suffix_rev))

    prefix_t = torch.stack(prefix, dim=-2)
    suffix_t = torch.stack(suffix, dim=-2)
    return prefix_t, suffix_t


def _log_conv(a: torch.Tensor, b: torch.Tensor, k: int) -> torch.Tensor:
    """Log e-table of the disjoint union of two candidate sets.

    ``c[..., m] = logsumexp_{p+q=m} a[..., p] + b[..., q]`` for ``m in [0, k]``.
    """
    # a, b: [..., k+1]
    out = []
    for m in range(k + 1):
        terms = [a[..., p] + b[..., m - p] for p in range(m + 1)]
        out.append(reduce(_logaddexp, terms))
    return torch.stack(out, dim=-1)


def edge_inclusion_log_probability(
    log_weights: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-edge log inclusion probability ``log pi_{ij}`` (Eq. 18).

        pi_{ij} = a_{ij} * e_{k-1}(a_{i,-j}) / e_k(a_i)

    Returns ``[..., n]``; masked-out entries are ``-inf`` (``pi = 0``).  Satisfies
    ``sum_j pi_{ij} = k`` (Eq. 19) up to floating point.
    """
    k = _check_k(k)
    log_w, m = _prepare_log_weights(log_weights, mask)
    n = log_w.shape[-1]
    if k == 0:
        return log_w.new_full(log_w.shape, NEG_INF)
    prefix, suffix = _log_prefix_suffix(log_w, k)
    log_ek = suffix[..., 0, k]  # e_k over all candidates
    # leave-one-out e_{k-1}(a_{-j}) = conv(prefix[j], suffix[j+1])[k-1]
    cols = []
    for j in range(n):
        eloo = _log_conv(prefix[..., j, :], suffix[..., j + 1, :], k)
        cols.append(log_w[..., j] + eloo[..., k - 1] - log_ek[...])
    log_pi = torch.stack(cols, dim=-1)
    log_pi = torch.where(m, log_pi, log_w.new_full((), NEG_INF))
    return log_pi


def edge_inclusion_probability(
    log_weights: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Linear-domain per-edge inclusion probability ``pi_{ij}`` (Eq. 18)."""
    return torch.exp(edge_inclusion_log_probability(log_weights, k, mask=mask))


def _log_suffix_only(log_w: torch.Tensor, k: int) -> torch.Tensor:
    """``suffix[..., t, :]`` = log e-table of candidates ``[t, n)``; shape ``[..., n+1, k+1]``."""
    _, suffix = _log_prefix_suffix(log_w, k)
    return suffix


def sample_k_subset(
    log_weights: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Exact ancestral sampler for Eq. 16 -- the *single shared* query sampler.

    Used identically in training and deployment (spec §4.1).  Processes candidates
    left to right; with ``r`` peers still to choose from the suffix ``[j, n)`` the
    inclusion probability of candidate ``j`` is

        P(include j) = a_j e_{r-1}(suffix_{j+1}) / e_r(suffix_j),
        e_r(suffix_j) = e_r(suffix_{j+1}) + a_j e_{r-1}(suffix_{j+1}),

    which is exactly the conditional of Eq. 16.  No Gumbel-top-k / Plackett-Luce
    surrogate is used (spec §4.1 prohibition).

    Args:
        log_weights: ``[B, n]`` log weights (a single source is ``[1, n]``).
        k: ``k_poll``.
        mask: optional ``[B, n]`` validity mask.
        generator: optional ``torch.Generator`` for reproducibility.

    Returns:
        Boolean ``[B, n]`` selection mask with exactly ``k`` True per row.

    Raises:
        ValueError: if any row has fewer than ``k`` valid candidates.  The
            candidate-shortage protocol (spec §7.2: ``k_i = min(k, |N_i|)`` or RSU
            fallback) must be applied upstream; copying a neighbour to pad ``k`` is
            forbidden.
    """
    k = _check_k(k)
    if log_weights.ndim != 2:
        raise ValueError("sample_k_subset expects a [B, n] tensor")
    log_w, m = _prepare_log_weights(log_weights, mask)
    B, n = log_w.shape
    valid_count = m.sum(dim=-1)
    if bool((valid_count < k).any().cpu()):
        raise ValueError(
            "a source has fewer than k valid candidates; apply the §7.2 "
            "candidate-shortage protocol upstream (do not pad by duplication)"
        )
    chosen = torch.zeros((B, n), dtype=torch.bool, device=log_w.device)
    if k == 0:
        return chosen
    suffix = _log_suffix_only(log_w, k).detach()  # [B, n+1, k+1]
    r = torch.full((B,), k, dtype=torch.long, device=log_w.device)
    rows = torch.arange(B, device=log_w.device)
    for j in range(n):
        suf_next = suffix[:, j + 1, :]  # [B, k+1], e-table of (j, n)
        r_clamped = r.clamp(min=1)
        log_e_r = suf_next[rows, r_clamped]
        log_e_rm1 = suf_next[rows, r_clamped - 1]
        lw = log_w[:, j]
        log_num = lw + log_e_rm1  # a_j e_{r-1}(suffix_{j+1})
        log_den = torch.logaddexp(log_e_r, log_num)  # e_r(suffix_j)
        incl = torch.exp(log_num - log_den)
        active = (r > 0) & m[:, j]
        incl = torch.where(active, incl, incl.new_zeros(()))
        u = torch.rand((B,), generator=generator, device=log_w.device, dtype=incl.dtype)
        take = (u < incl) & active
        chosen[:, j] = take
        r = r - take.long()
    if bool((r != 0).any().cpu()):  # defensive: feasibility guaranteed above
        raise RuntimeError("ancestral sampler failed to select exactly k candidates")
    return chosen


def enumerate_subset_distribution(
    log_weights: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
) -> dict[tuple[int, ...], float]:
    """Brute-force Eq. 16 over all ``C(n, k)`` subsets (single source, for tests).

    Returns a dict mapping each sorted candidate tuple to its probability.
    """
    if log_weights.ndim != 1:
        raise ValueError("enumerate_subset_distribution expects a single source ([n])")
    log_w, m = _prepare_log_weights(log_weights, mask)
    valid = [int(i) for i in range(log_w.shape[-1]) if bool(m[i])]
    log_ek = log_elementary_symmetric(log_w, k)[..., k]
    out: dict[tuple[int, ...], float] = {}
    for combo in combinations(valid, k):
        idx = torch.as_tensor(combo, dtype=torch.long, device=log_w.device)
        lp = log_w[idx].sum() - log_ek
        out[tuple(int(c) for c in combo)] = float(torch.exp(lp))
    return out


def verify_sampler_matches_distribution(
    log_weights: torch.Tensor,
    k: int,
    *,
    mask: torch.Tensor | None = None,
    atol: float = 1e-10,
) -> dict[str, float]:
    """Assert the ancestral sampler reproduces Eq. 16 exactly (single source).

    Computes the sampler's induced subset probability *analytically* by walking the
    same conditional chain :func:`sample_k_subset` uses (not by Monte-Carlo), and
    compares it to :func:`enumerate_subset_distribution`.  This is the code-level
    "same sampler" assertion required by §4.1 / G2.

    Returns a diagnostics dict and raises ``AssertionError`` on mismatch.
    """
    if log_weights.ndim != 1:
        raise ValueError("verify_sampler_matches_distribution expects [n]")
    log_w, m = _prepare_log_weights(log_weights, mask)
    n = log_w.shape[-1]
    suffix = _log_suffix_only(log_w, k)  # [n+1, k+1], suffix[t] = e-table of [t, n)
    target = enumerate_subset_distribution(log_weights, k, mask=mask)

    def chain_log_prob(combo: tuple[int, ...]) -> torch.Tensor:
        """Log probability the ancestral chain (sample_k_subset) assigns to ``combo``.

        Accumulated entirely in the log domain using the *same* conditionals as the
        sampler: log p_in = log_num - log_den, log(1 - p_in) = log_e_r - log_den,
        both numerically stable (no catastrophic cancellation).
        """
        chosen = set(combo)
        r = k
        log_acc = log_w.new_zeros(())
        for j in range(n):
            if r == 0:
                break
            suf_next = suffix[j + 1]
            log_e_r = suf_next[r]
            log_e_rm1 = suf_next[r - 1]
            if bool(m[j]):
                log_num = log_w[j] + log_e_rm1
                log_den = _logaddexp(log_e_r, log_num)
            else:  # masked candidate: inclusion impossible
                log_num = log_w.new_full((), NEG_INF)
                log_den = log_e_r
            if j in chosen:
                log_acc = log_acc + (log_num - log_den)
                r -= 1
            else:
                log_acc = log_acc + (log_e_r - log_den)
        return log_acc

    max_abs = 0.0
    total = 0.0
    for combo, p in target.items():
        sp = float(torch.exp(chain_log_prob(combo)))
        total += sp
        max_abs = max(max_abs, abs(sp - p))
    if max_abs > atol:
        raise AssertionError(
            f"ancestral sampler does not match Eq. 16: max |delta| = {max_abs:.3e} > {atol:.1e}"
        )
    return {"max_abs_error": max_abs, "sampler_total_mass": total, "num_subsets": float(len(target))}


@dataclass(frozen=True)
class SubsetPolicy:
    """Bundled weighted distinct-peer query policy for a batch of sources.

    Attributes:
        log_weights: ``[B, n]`` log candidate weights ``s_{ij}``.
        k: ``k_poll``.
        mask: ``[B, n]`` validity mask.
    """

    log_weights: torch.Tensor
    k: int
    mask: torch.Tensor | None = None

    def log_e_table(self) -> torch.Tensor:
        return log_elementary_symmetric(self.log_weights, self.k, mask=self.mask)

    def inclusion_probability(self) -> torch.Tensor:
        return edge_inclusion_probability(self.log_weights, self.k, mask=self.mask)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        return sample_k_subset(self.log_weights, self.k, mask=self.mask, generator=generator)
