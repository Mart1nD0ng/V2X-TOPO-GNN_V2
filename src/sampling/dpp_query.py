"""CDQ -- correlation-aware low-rank ``k``-DPP query layer (spec §9.4 -- G4).

Each candidate peer ``j`` of source ``i`` gets a quality scalar ``q_ij > 0`` and a diversity
embedding ``b_ij in R^r`` (``r >= k``). The query kernel is the PSD low-rank matrix

    L_i = B_i B_i^T,    B_i[j, :] = sqrt(q_ij) b_ij                                   (spec §9.4)

and the distinct-peer ``k``-subset is drawn from the ``k``-DPP

    P_i(S) = det(L_{i,S}) / e_k(lambda(L_i)),   |S| = k                              (Eq. 9.4)

Row norms encode quality, row angles encode similarity: similar peers shrink the determinant
(less likely to be co-selected), complementary peers span more volume (more likely) -- this is
the evidence-diversity mechanism the ESP product law lacks. When the kernel is DIAGONAL
(``L_i = diag(a_i)``, i.e. orthonormal diversity rows) the law collapses exactly to the §4
elementary-symmetric product policy (``det(L_{i,S}) = prod_{j in S} a_{ij}``), so CDQ is a
strict generalisation of the current ESP query (validated in the tests).

Key exact + differentiable identities used (no eigendecomposition on the gradient path):

* ``e_k(lambda(L_i)) = e_k(lambda(M_i))``, ``M_i = B_i^T B_i`` (``r x r``), via Newton's
  identities from the power sums ``tr(M_i^t)`` (zeros of ``L`` outside the rank don't
  affect ``e_k`` for ``k <= r``).
* **Inclusion marginal** ``pi_ij = q_ij * d log e_k / d q_ij`` -- because every principal
  minor ``det(L_{i,S})`` is degree-1 homogeneous in ``q_ij`` for ``j in S``, so
  ``sum_{S in i, j in S} det(L_S) = q_ij * d e_k / d q_ij`` and ``pi_ij = that / e_k``. This
  reproduces the ESP inclusion ``a_j e_{k-1}(a_{-j}) / e_k`` in the diagonal case and gives
  ``sum_j pi_ij = k``. Obtained by autograd of ``log e_k`` -- a single backward pass yields
  all inclusion marginals, differentiable end-to-end.

Complexity: ``O(d r^2)`` per source for ``M`` and ``O(r^3)`` for the ``k`` matrix powers --
near-linear in the candidate degree (no ``d x d`` kernel materialised; ``r`` small constant).
The exact ``k``-DPP sampler (eigendecomposition of the ``r x r`` dual ``M``) is used only by
the dynamic MC (no gradient needed there).

Exactness boundary: exact ``k``-DPP probabilities/marginals under the defined low-rank kernel;
``k <= r`` required. Validated against brute-force subset enumeration (< 1e-10).
"""

from __future__ import annotations

from itertools import combinations

import torch

__all__ = [
    "low_rank_kernel",
    "kdpp_elementary_symmetric",
    "kdpp_log_normalizer",
    "kdpp_normalizer",
    "kdpp_inclusion",
    "kdpp_subset_log_prob",
    "enumerate_kdpp_distribution",
    "kdpp_sample",
    "diagonal_diversity",
]


def low_rank_kernel(quality: torch.Tensor, diversity: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """``B[..., j, :] = sqrt(q_j) b_j`` from quality ``[..., d]`` and diversity ``[..., d, r]``.

    ``quality`` is clamped to ``[eps, inf)`` before the square root so a zero/near-zero
    quality (e.g. a masked/padded candidate) gives a vanishing kernel row with a
    WELL-DEFINED (zero) gradient -- ``sqrt`` at exactly 0 has an infinite derivative that
    would NaN-poison a batched backward; ``clamp_min(eps)`` saturates there so the gradient
    is 0 (it does not flow through the clamp for inputs below ``eps``).
    """
    if diversity.shape[:-1] != quality.shape:
        raise ValueError("diversity must be [..., d, r] matching quality [..., d]")
    if bool(torch.any(quality.detach() < 0).cpu()):
        raise ValueError("quality must be nonnegative")
    return torch.sqrt(quality.clamp_min(eps)).unsqueeze(-1) * diversity


def _gram(B: torch.Tensor) -> torch.Tensor:
    return B.transpose(-1, -2) @ B               # M = B^T B, [..., r, r]


def kdpp_elementary_symmetric(M: torch.Tensor, k: int) -> torch.Tensor:
    """``e_0 .. e_k`` of the eigenvalues of ``M`` (``[..., r, r]``) via Newton's identities.

    Returns ``[..., k+1]``. Differentiable (uses only matrix powers + traces, no eigh).
    """
    if k < 0:
        raise ValueError("k must be >= 0")
    r = M.shape[-1]
    if k > r:
        raise ValueError(f"k={k} must be <= r={r} for a rank-r kernel")
    batch = M.shape[:-2]
    # power sums p_t = tr(M^t), t = 1..k
    powers: list[torch.Tensor] = []
    Mp = M
    for t in range(1, k + 1):
        if t > 1:
            Mp = Mp @ M
        powers.append(torch.diagonal(Mp, dim1=-2, dim2=-1).sum(dim=-1))   # [...]
    e = [M.new_ones(batch)]                       # e_0 = 1
    for m in range(1, k + 1):
        acc = M.new_zeros(batch)
        for i in range(1, m + 1):
            sign = 1.0 if (i - 1) % 2 == 0 else -1.0
            acc = acc + sign * e[m - i] * powers[i - 1]
        e.append(acc / m)
    return torch.stack(e, dim=-1)                 # [..., k+1]


def _check_k(B: torch.Tensor, k: int) -> tuple[int, int]:
    """Validate ``0 <= k <= min(d, r)`` (a k-DPP needs >= k distinct candidates and k <= rank)."""
    d, r = B.shape[-2], B.shape[-1]
    mn = min(d, r)
    if k < 0 or k > mn:
        raise ValueError(f"k={k} must satisfy 0 <= k <= min(d, r) = {mn} (d={d}, r={r})")
    return d, r


def _ek_principal_minors(M: torch.Tensor, k: int) -> torch.Tensor:
    """``e_k(lambda(M)) = sum_{|T|=k} det(M_T)`` (sum of k x k principal minors). ``[...]``."""
    r = M.shape[-1]
    total = M.new_zeros(M.shape[:-2])
    for T in combinations(range(r), k):
        idx = torch.tensor(T, dtype=torch.long, device=M.device)
        sub = M.index_select(-2, idx).index_select(-1, idx)        # [..., k, k]
        total = total + torch.det(sub)
    return total


def kdpp_normalizer(B: torch.Tensor, k: int) -> torch.Tensor:
    """``e_k(lambda(L))`` with ``L = B B^T`` (the ``k``-DPP normaliser). ``[...]``.

    Exact identity ``e_k(lambda(M)) = sum_{|T|=k} det(M_T)`` (principal minors of
    ``M = B^T B``) -- float64-accurate (no Newton accumulation), differentiable. For very
    large quality use :func:`kdpp_log_normalizer` (this linear form can overflow).
    """
    _check_k(B, k)
    M = _gram(B)
    if k == 0:
        return M.new_ones(M.shape[:-2])
    return _ek_principal_minors(M, k)


def kdpp_log_normalizer(B: torch.Tensor, k: int) -> torch.Tensor:
    """``log e_k(lambda(L))`` computed in a SCALE-STABLE way (no overflow for large quality).

    Factors the mean eigenvalue ``s = tr(M)/r`` out of ``M`` (``e_k`` is degree-``k``
    homogeneous: ``e_k(M) = s^k e_k(M/s)``), so ``log e_k = k log s + log e_k(M/s)`` with the
    scaled minors ``O(1)``. Exact (the scaling is an exact reparameterisation, so the autograd
    gradient is the true ``d log e_k / d.``), and differentiable.
    """
    _check_k(B, k)
    M = _gram(B)
    if k == 0:
        return M.new_zeros(M.shape[:-2])
    r = M.shape[-1]
    s = (torch.diagonal(M, dim1=-2, dim2=-1).sum(dim=-1) / r).clamp_min(torch.finfo(M.dtype).tiny)
    M_s = M / s.unsqueeze(-1).unsqueeze(-1)
    ek_s = _ek_principal_minors(M_s, k)
    return k * torch.log(s) + torch.log(ek_s)


def kdpp_inclusion(quality: torch.Tensor, diversity: torch.Tensor, k: int) -> torch.Tensor:
    """Per-candidate inclusion marginal ``pi_j = P(j in S)`` for the ``k``-DPP -> ``[..., d]``.

    Computed exactly as ``pi_j = q_j * d log e_k / d q_j`` via autograd. ``sum_j pi_j = k``.
    Differentiable w.r.t. ``quality`` and ``diversity`` when they require grad.
    """
    if k == 0:
        z = torch.zeros_like(quality)
        return z if (quality.requires_grad or diversity.requires_grad) else z.detach()
    need_graph = quality.requires_grad or diversity.requires_grad
    # enable_grad so the internal autograd works even under an ambient torch.no_grad()
    # (the value-only inference path) -- requires_grad_ alone is a no-op when grad is off.
    with torch.enable_grad():
        q = quality if quality.requires_grad else quality.detach().requires_grad_(True)
        B = low_rank_kernel(q, diversity)
        log_ek = kdpp_log_normalizer(B, k)
        grad = torch.autograd.grad(log_ek.sum(), q, create_graph=need_graph)[0]
        pi = q * grad
    return pi if need_graph else pi.detach()


def kdpp_subset_log_prob(B: torch.Tensor, subset, k: int) -> torch.Tensor:
    """``log P(S) = log det(L_S) - log e_k`` for an explicit subset (single source ``[d, r]``)."""
    if B.ndim != 2:
        raise ValueError("kdpp_subset_log_prob expects a single source B [d, r]")
    idx = torch.as_tensor(list(subset), dtype=torch.long, device=B.device)
    if idx.numel() != k:
        raise ValueError("subset must have exactly k indices")
    Bs = B[idx]                                   # [k, r]
    Ls = Bs @ Bs.transpose(-1, -2)                # [k, k]
    sign, logabsdet = torch.linalg.slogdet(Ls)
    log_ek = kdpp_log_normalizer(B, k)            # scale-stable (no overflow at large quality)
    return logabsdet - log_ek


def enumerate_kdpp_distribution(B: torch.Tensor, k: int) -> tuple[dict[tuple[int, ...], float], float]:
    """Brute-force ``P(S) = det(L_S)/e_k`` over all ``C(d,k)`` subsets (single source, tests)."""
    if B.ndim != 2:
        raise ValueError("enumerate_kdpp_distribution expects a single source B [d, r]")
    d = B.shape[0]
    dets: dict[tuple[int, ...], float] = {}
    ek = 0.0
    for S in combinations(range(d), k):
        Bs = B[list(S)]
        det = float(torch.det(Bs @ Bs.transpose(-1, -2)))
        det = max(det, 0.0)
        dets[S] = det
        ek += det
    return {S: v / ek for S, v in dets.items()}, ek


def diagonal_diversity(d: int, *, dtype: torch.dtype = torch.float64,
                       device: torch.device | None = None) -> torch.Tensor:
    """Orthonormal diversity rows ``b_j = e_j`` (``r = d``) -> the kernel is ``diag(q)`` and the
    ``k``-DPP collapses to the ESP product policy (the diagonal special case, spec §9.4)."""
    return torch.eye(d, dtype=dtype, device=device)


def kdpp_sample(B: torch.Tensor, k: int, *, generator: torch.Generator | None = None) -> list[int]:
    """Exact ``k``-DPP sampler (Kulesza-Taskar) via the low-rank dual (single source ``[d, r]``).

    Eigendecomposes the ``r x r`` dual ``M = B^T B``; selects ``k`` eigenvectors with the
    elementary-symmetric ``k``-DPP rule (weights ``lambda``), then samples an elementary DPP
    from the selected eigenvectors. No gradient (sampling is for the dynamic MC).
    """
    if B.ndim != 2:
        raise ValueError("kdpp_sample expects a single source B [d, r]")
    d, r = B.shape
    if k > r:
        raise ValueError("k must be <= r")
    M = (B.transpose(-1, -2) @ B).detach()
    lam, V = torch.linalg.eigh(M)                 # ascending; lam [r], V [r, r]
    lam = lam.clamp_min(0.0)
    pos = lam > 1e-12
    lam_p = lam[pos]
    Vp = V[:, pos]
    m = int(lam_p.numel())
    if k > m:
        raise ValueError("k exceeds the number of positive eigenvalues (rank-deficient kernel)")
    # eigenvectors of L (nonzero spectrum): U = B Vp / sqrt(lam_p), [d, m] orthonormal columns
    U = (B @ Vp) / torch.sqrt(lam_p).unsqueeze(0)

    # --- step 1: select k of the m eigenvectors with the k-DPP elementary-symmetric rule ---
    etab = B.new_zeros((k + 1, m + 1))
    etab[0, :] = 1.0
    for i in range(1, k + 1):
        for j in range(1, m + 1):
            etab[i, j] = etab[i, j - 1] + lam_p[j - 1] * etab[i - 1, j - 1]
    selected: list[int] = []
    i = k
    for j in range(m, 0, -1):
        if i == 0:
            break
        denom = etab[i, j]
        if float(denom) <= 0:
            continue
        marg = lam_p[j - 1] * etab[i - 1, j - 1] / denom
        u = torch.rand((), generator=generator, device=B.device, dtype=B.dtype)
        if bool(u < marg):
            selected.append(j - 1)
            i -= 1

    # --- step 2: sample an elementary DPP from the selected eigenvectors ---
    Vsel = U[:, selected].clone()                 # [d, k]
    Y: list[int] = []
    while Vsel.shape[1] > 0:
        probs = (Vsel ** 2).sum(dim=1)
        probs = probs / probs.sum()
        item = int(torch.multinomial(probs, 1, generator=generator))
        Y.append(item)
        col = int(torch.argmax(Vsel[item].abs()))
        vj = Vsel[:, col].clone()
        Vsel = torch.cat([Vsel[:, :col], Vsel[:, col + 1:]], dim=1)
        if Vsel.shape[1] > 0:
            Vsel = Vsel - torch.outer(vj, Vsel[item, :] / vj[item])
            Vsel = torch.linalg.qr(Vsel).Q
    return sorted(Y)
