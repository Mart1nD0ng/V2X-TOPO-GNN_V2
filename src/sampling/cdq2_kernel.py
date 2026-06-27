"""CDQ 2.0 -- the diagonal-plus-low-rank query kernel that EXACTLY contains ESP (spec §9).

The prior round's CDQ used a pure low-rank kernel ``L = B B^T`` (``src/sampling/dpp_query.py``),
which recovers the §4 ESP product policy only when the kernel is full rank (orthonormal rows,
``r = d``) -- defeating the low-rank near-linear cost. CDQ 2.0 instead keeps a FULL-RANK
diagonal quality and adds a rank-``r`` diversity correction:

    D = diag(a_1, .., a_d),  a_j = exp(s_j) > 0,   Z = [z_bar_1; ..; z_bar_d]  (unit rows)

    L = D^{1/2} (I + eta Z Z^T) D^{1/2},     eta >= 0.                              (spec §9)

Properties (all exercised in tests/sampling/test_cdq2_kernel.py):

* **ESP exact degeneracy.** ``eta = 0  =>  L = D  =>  P_CDQ = P_ESP`` EXACTLY, for ANY ``Z``
  and ANY rank ``r``. The model is initialised at ``eta ~ 0`` and learns diversity only if the
  environment rewards it (the identifiability anchor). Crucially ``eta`` keeps a LIVE gradient
  even at exactly ``eta = 0`` (dL/d_eta = D^{1/2} ZZ^T D^{1/2} != 0), so the model can move off
  ESP.
* **Quality / diversity separation.**
  ``det(L_S) = (prod_{j in S} a_j) det(I_{|S|} + eta Z_S Z_S^T)`` -- ESP quality times a
  diversity correction that depends only on the geometry of the selected rows.
* **k=2 closed form.** For unit ``z_j, z_l``:
  ``det(I + eta Z_S Z_S^T) = (1+eta)^2 - eta^2 (z_j . z_l)^2`` -- more-similar peers get a
  smaller joint-selection weight (the lever ESP's product law lacks).

Exact + differentiable normaliser ``e_k(lambda(L))`` -- the UNCONDITIONALLY STABLE eigenvalue
route. We considered the matrix-determinant-lemma ``z``-series (spec §10):

    det(I + z L) = prod_j(1 + z c_j) . det(I_r + eta z Z^T diag(c_j/(1+z c_j)) Z),

extracting ``[z^k]`` in ``R[z]/(z^{k+1})``. It is asymptotically cheaper per source (``O(d r^2 k)``)
and EXACT in exact arithmetic, but the ``(I + z D_c)^{-1}`` geometric series makes the diversity
factor's coefficients ALTERNATE in sign, so assembling the (provably positive) ``e_k`` from that
product suffers catastrophic float64 cancellation at wide quality dynamic range or large ``eta``
(e.g. realistic GNN logits in ``[-7, 7]`` give ``~1e-6`` errors and, at extreme spread, a negative
``e_k`` whose ``log`` is NaN -- an adversarial audit confirmed this end-to-end). No scale factor
cures DIFFERENTIAL cancellation across terms spanning many orders of magnitude.

So we compute ``e_k(lambda(L))`` from the EIGENVALUES of the per-source SPD kernel:

    L_sym = D_c^{1/2} (I + eta Z Z^T) D_c^{1/2}   (c_j = a_j; SPD, eig(L_sym) = eig(L))
    e_k(lambda(L)) = e_k( eigvalsh(L_sym) )       via a cancellation-free linear ESP recursion.

This is:

* **Stable** -- ``e_k`` of positive eigenvalues is a sum of products of positives (NO subtraction);
  ``eigvalsh`` of an SPD matrix is backward-stable. Correct at wide dynamic range, near-collinear
  rows, and large ``eta`` (all audit reproducers now pass < 1e-10).
* **Smoothly differentiable in a, Z, eta everywhere** -- ``e_k`` is a SYMMETRIC function of the
  eigenvalues, i.e. a polynomial in the matrix entries, so the eigenvalue backward
  ``U diag(de_k/d_lambda) U^T`` carries NO ``1/(lambda_i - lambda_j)`` eigen-gap denominators and
  stays finite even at repeated eigenvalues (collinear diversity rows).
* **No global N x N** -- ``L_sym`` is the ``d_i x d_i`` per-source kernel (``d_i`` = candidate
  degree, bounded by the communication radius), never an ``N x N`` matrix. Per-source cost
  ``O(d_i^3 + d_i^2 r)``; for bounded degree ``d_i <= D_max`` this is ``O(D_max^2 . E)`` --
  near-linear in the candidate-edge count ``E`` (the determinant-lemma route would shave the
  ``d^3`` to ``d``, but only in the well-conditioned regime, so stability wins here).

``e_k(lambda(L))`` is degree-``k`` homogeneous in ``a`` (scaling ``a`` scales ``L_sym`` linearly),
giving the inclusion identity ``pi_j = a_j d log e_k / d a_j`` (the diversity factor is
``a``-independent, so the ESP homogeneity argument carries over verbatim) and ``sum_j pi_j = k``.

Exactness boundary: exact ``k``-DPP normaliser / subset probability / inclusion under the CDQ 2.0
kernel for ``0 <= k <= d``; validated against brute-force subset enumeration (< 1e-10, incl. wide
dynamic range / collinear rows / large eta) and finite-difference gradients (rel-err < 1e-4). The
heterogeneous determinantal quorum ``P(m,n)`` under this kernel (spec §10) and the exact ``k``-DPP
sampler are the next sub-slices.
"""

from __future__ import annotations

from itertools import combinations

import torch

__all__ = [
    "cdq2_unit_normalize",
    "cdq2_k2_diversity_factor",
    "cdq2_subset_logdet",
    "cdq2_log_normalizer",
    "cdq2_normalizer",
    "cdq2_subset_log_prob",
    "cdq2_inclusion",
    "cdq2_sample",
    "cdq2_enumerate_distribution",
    "bruteforce_cdq2_normalizer",
]


def cdq2_unit_normalize(Z: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Unit-normalise the diversity rows ``z_bar_j = z_j / (||z_j|| + eps)`` (spec §9).

    The ``+ eps`` (rather than ``clamp``) keeps a well-defined zero-limit gradient for a
    vanishing embedding row and matches the spec's ``z / (||z|| + eps)`` form exactly.
    """
    if Z.shape[-1] < 1:
        raise ValueError("diversity Z must have a trailing embedding dimension r >= 1")
    norm = torch.linalg.vector_norm(Z, dim=-1, keepdim=True)
    return Z / (norm + eps)


def _as_eta(eta, ref: torch.Tensor) -> torch.Tensor:
    e = torch.as_tensor(eta, dtype=ref.dtype, device=ref.device)
    if bool(torch.any(e.detach() < 0).cpu()):
        raise ValueError("eta must be >= 0")
    return e


def cdq2_k2_diversity_factor(z_j: torch.Tensor, z_l: torch.Tensor, eta) -> torch.Tensor:
    """``det(I + eta Z_S Z_S^T) = (1+eta)^2 - eta^2 (z_j . z_l)^2`` for unit rows ``z_j, z_l``
    (spec §9, the ``k=2`` closed form). Inputs are unit-normalised internally."""
    zj = z_j / (torch.linalg.vector_norm(z_j) + 1e-12)
    zl = z_l / (torch.linalg.vector_norm(z_l) + 1e-12)
    e = _as_eta(eta, zj)
    rho = (zj * zl).sum()
    return (1 + e) ** 2 - e ** 2 * rho ** 2


def cdq2_subset_logdet(a: torch.Tensor, Z: torch.Tensor, eta, subset) -> torch.Tensor:
    """``log det(L_S) = sum_{j in S} log a_j + log det(I_{|S|} + eta Z_S Z_S^T)`` (single source).

    Uses the Sylvester identity ``det(I_{|S|} + eta Z_S Z_S^T) = det(I_r + eta Z_S^T Z_S)`` so
    the determinant is taken on the smaller of the two (``min(|S|, r)``).
    """
    if a.ndim != 1:
        raise ValueError("cdq2_subset_logdet expects a single source (a is [d])")
    idx = torch.as_tensor(list(subset), dtype=torch.long, device=a.device)
    if torch.unique(idx).numel() != idx.numel():
        raise ValueError("subset must contain distinct candidates")
    Zn = cdq2_unit_normalize(Z)
    e = _as_eta(eta, a)
    Zs = Zn.index_select(0, idx)                                  # [|S|, r]
    s_card, r = Zs.shape
    if s_card <= r:
        gram = Zs @ Zs.transpose(-1, -2)                         # [|S|, |S|]
        eye = torch.eye(s_card, dtype=a.dtype, device=a.device)
    else:
        gram = Zs.transpose(-1, -2) @ Zs                         # [r, r]
        eye = torch.eye(r, dtype=a.dtype, device=a.device)
    sign, logabsdet = torch.linalg.slogdet(eye + e * gram)
    return torch.log(a.index_select(0, idx)).sum() + logabsdet


def _elem_sym_positive(lams: torch.Tensor, k: int) -> torch.Tensor:
    """``e_k`` of nonnegative values ``lams`` (``[..., d]``) via the add-one recursion in the
    LINEAR domain -> ``[...]``.

    ``e_m <- e_m + lam_j e_{m-1}`` (m descending) over all values. Every operation is a positive
    add/multiply, so there is NO cancellation (the inputs here are eigenvalues of an SPD matrix,
    hence >= 0) and it is double-differentiable (plain mul/add, unlike the log-domain ``logaddexp``
    routine whose hand-written backward is only first-order).
    """
    batch = lams.shape[:-1]
    d = lams.shape[-1]
    e = [lams.new_ones(batch)] + [lams.new_zeros(batch) for _ in range(k)]
    for j in range(d):
        lj = lams[..., j]
        for m in range(min(k, j + 1), 0, -1):
            e[m] = e[m] + lj * e[m - 1]
    return e[k]


def _cdq2_log_ek_from_c(c: torch.Tensor, Zn: torch.Tensor, e: torch.Tensor, k: int,
                        mask: torch.Tensor | None = None) -> torch.Tensor:
    """``log e_k(lambda(L_sym))``, ``L_sym = D_c^{1/2}(I + eta Z Z^T) D_c^{1/2}`` (``c`` nonneg,
    ``Zn`` unit-normalised, ``e`` = eta tensor). Returns ``[...]``.

    Eigenvalue route (unconditionally stable, smooth gradient, no N x N). Factors the mean of
    ``c`` out first (``L_sym(c/s) = L_sym(c)/s`` exactly, so ``e_k`` scales by ``s^k``) purely to
    avoid float64 OVERFLOW for large GNN logits -- there is no cancellation to cure.

    ``mask`` (``[..., d]`` bool) marks REAL candidates; padded slots get an EXACTLY-zero kernel
    row (``root = 0`` via ``where``, so the ``sqrt(0)`` infinite local derivative is blocked) and
    the scale ``s`` is the mean over real slots only (so a padded quality has a zero gradient path
    and contributes nothing). This makes the bucketed exclusion EXACT (zero eigenvalue, not an
    eps-clamp) with finite gradients.
    """
    if k == 0:
        return c.new_zeros(c.shape[:-1])
    d = c.shape[-1]
    tiny = torch.finfo(c.dtype).tiny
    if mask is None:
        s = c.mean(dim=-1).clamp_min(tiny)
    else:
        m = mask.to(dtype=c.dtype)
        s = ((c * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)).clamp_min(tiny)
    cs = (c / s.unsqueeze(-1)).clamp_min(0.0)
    # A candidate contributes iff it is real (unmasked) AND has nonzero deformed quality. A zero cs
    # -- a padded slot (a=0) OR a real slot whose c = a*g hits 0 when g=0 (e.g. p0 = 1-ell = 0 at an
    # ideal link ell=1) -- must give an EXACTLY-zero kernel row with a FINITE backward: feeding 0 to
    # sqrt has an infinite derivative (0.5/sqrt(0)) that would NaN-poison the gradient. So we sqrt a
    # sanitized input (zeros -> 1) and re-zero via the boolean factor, which is exact in the forward
    # (zero eigenvalue) and finite (zero) in the backward at the contributing/non-contributing edge.
    pos = (cs > 0) if mask is None else (mask & (cs > 0))
    cs_safe = torch.where(pos, cs, torch.ones_like(cs))
    root = cs_safe.sqrt() * pos.to(dtype=c.dtype)               # [..., d]
    eye = torch.eye(d, dtype=c.dtype, device=c.device)
    # G = I + eta Z Z^T  (per-source d x d; never N x N), L_sym = diag(root) G diag(root)
    G = eye + e.unsqueeze(-1).unsqueeze(-1) * (Zn @ Zn.transpose(-1, -2))
    L_sym = root.unsqueeze(-1) * G * root.unsqueeze(-2)          # [..., d, d]
    L_sym = 0.5 * (L_sym + L_sym.transpose(-1, -2))             # symmetrise (float round-off)
    lam = torch.linalg.eigvalsh(L_sym).clamp_min(0.0)            # [..., d], ascending, >= 0
    ek_scaled = _elem_sym_positive(lam, k)
    # clamp_min(tiny) before log: when EVERY candidate is excluded at a grid point (all c=0, e.g.
    # an ideal link ell=1 -> p0=0 for all), e_k = 0 and log(0) = -inf with an upstream 0 gradient
    # gives 0*inf = NaN. The clamp keeps log finite with a saturated (0) gradient there; for any
    # real quorum e_k is O(1) >> tiny so the clamp is a no-op (exact).
    return k * torch.log(s) + torch.log(ek_scaled.clamp_min(tiny))


def cdq2_log_normalizer(a: torch.Tensor, Z: torch.Tensor, eta, k: int,
                        mask: torch.Tensor | None = None) -> torch.Tensor:
    """``log e_k(lambda(L))`` for ``L = D^{1/2}(I + eta Z Z^T) D^{1/2}``. ``[...]``.

    Differentiable w.r.t. ``a, Z, eta`` everywhere (incl. exactly ``eta = 0``). ``mask`` (``[..., d]``
    bool) excludes padded candidates EXACTLY (bucketed wiring); real slots must still be ``> 0``.
    """
    if k < 0 or k > a.shape[-1]:
        raise ValueError(f"k={k} must satisfy 0 <= k <= d = {a.shape[-1]}")
    bad = (a.detach() <= 0) & mask if mask is not None else (a.detach() <= 0)
    if bool(torch.any(bad).cpu()):
        raise ValueError("quality a must be > 0 on every real (unmasked) candidate")
    if k == 0:
        return a.new_zeros(a.shape[:-1])
    e = _as_eta(eta, a)
    Zn = cdq2_unit_normalize(Z)
    return _cdq2_log_ek_from_c(a, Zn, e, k, mask=mask)


def cdq2_normalizer(a: torch.Tensor, Z: torch.Tensor, eta, k: int) -> torch.Tensor:
    """``e_k(lambda(L))`` in the linear domain (``exp`` of :func:`cdq2_log_normalizer`)."""
    if k == 0:
        return a.new_ones(a.shape[:-1])
    return torch.exp(cdq2_log_normalizer(a, Z, eta, k))


def cdq2_subset_log_prob(a: torch.Tensor, Z: torch.Tensor, eta, subset, k: int) -> torch.Tensor:
    """``log P(S) = log det(L_S) - log e_k(lambda(L))`` for an explicit ``k``-subset (single source)."""
    idx = list(subset)
    if len(idx) != k:
        raise ValueError("subset must have exactly k indices")
    return cdq2_subset_logdet(a, Z, eta, idx) - cdq2_log_normalizer(a, Z, eta, k)


def cdq2_inclusion(a: torch.Tensor, Z: torch.Tensor, eta, k: int,
                   mask: torch.Tensor | None = None) -> torch.Tensor:
    """Per-candidate ``k``-DPP inclusion marginal ``pi_j = P(j in S) = a_j d log e_k / d a_j``.

    Exact via autograd (the diversity factor is ``a``-independent, so the ESP homogeneity
    identity holds verbatim). ``sum_j pi_j = k`` over the real candidates. Differentiable when
    ``a/Z/eta`` require grad. ``mask`` (``[..., d]`` bool) excludes padded candidates (their
    ``pi`` is exactly 0); the inclusion of the real candidates is computed over the real set only.
    """
    if k == 0:
        z = torch.zeros_like(a)
        return z if (a.requires_grad or Z.requires_grad) else z.detach()
    need_graph = a.requires_grad or Z.requires_grad or (torch.is_tensor(eta) and eta.requires_grad)
    with torch.enable_grad():
        aa = a if a.requires_grad else a.detach().requires_grad_(True)
        log_ek = cdq2_log_normalizer(aa, Z, eta, k, mask=mask)
        grad = torch.autograd.grad(log_ek.sum(), aa, create_graph=need_graph)[0]
        pi = aa * grad
    if mask is not None:
        pi = torch.where(mask, pi, torch.zeros_like(pi))
    return pi if need_graph else pi.detach()


def cdq2_sample(a: torch.Tensor, Z: torch.Tensor, eta, k: int, *,
                generator: torch.Generator | None = None) -> list[int]:
    """Exact ``k``-DPP sampler (Kulesza-Taskar) for the CDQ 2.0 kernel (single source ``a`` ``[d]``).

    ``L = D^{1/2}(I + eta Z Z^T) D^{1/2}`` is a full-rank ``d x d`` SPD kernel, so we
    eigendecompose it directly (``eigvalsh`` -> eigenvectors via ``eigh``) and run the standard
    two-step ``k``-DPP eigen-sampler: (1) pick ``k`` of the ``d`` eigenvectors with the
    elementary-symmetric ``k``-DPP rule (weights = eigenvalues), (2) sample an elementary DPP from
    the selected eigenvectors. The induced law is exactly ``P(S) = det(L_S) / e_k(lambda(L))``.

    No gradient (sampling is for the dynamic MC judge). ``L`` is the ``d x d`` per-source kernel
    (``d`` = candidate degree, bounded by the comm radius) -- never an ``N x N`` matrix. At
    ``eta = 0`` it reduces to the ESP elementary-symmetric ``k``-subset sampler.
    """
    if a.ndim != 1:
        raise ValueError("cdq2_sample expects a single source (a is [d])")
    d = a.shape[0]
    if k < 0 or k > d:
        raise ValueError(f"k={k} must satisfy 0 <= k <= d = {d}")
    if k == 0:
        return []
    if bool(torch.any(a.detach() <= 0).cpu()):
        raise ValueError("quality a must be > 0")
    Zn = cdq2_unit_normalize(Z).detach()
    e = _as_eta(eta, a).detach()
    # Factor out the mean quality before forming L for eigh: scaling c by 1/s scales L (and its
    # eigenvalues) by 1/s, but the step-1 selection uses scale-invariant eigenvalue RATIOS and
    # step 2 uses the (scale-invariant) eigenvectors -- so the k-DPP law is IDENTICAL, while the
    # eigh stays overflow-free and the pos = lam>1e-12 threshold becomes a meaningful relative one.
    c = a.detach().clamp_min(0.0)
    s = c.mean().clamp_min(torch.finfo(a.dtype).tiny)
    c = c / s
    root = c.sqrt()
    eye = torch.eye(d, dtype=a.dtype, device=a.device)
    G = eye + e * (Zn @ Zn.transpose(-1, -2))                # I + eta Z Z^T
    L = root.unsqueeze(-1) * G * root.unsqueeze(-2)          # D_c^{1/2} G D_c^{1/2}  [d, d]
    L = 0.5 * (L + L.transpose(-1, -2))
    lam, V = torch.linalg.eigh(L)                            # ascending; lam [d], V [d, d]
    lam = lam.clamp_min(0.0)
    pos = lam > 1e-12
    lam_p = lam[pos]
    Vp = V[:, pos]
    m = int(lam_p.numel())
    if k > m:
        raise ValueError("k exceeds the number of positive eigenvalues (apply the §7.2 shortage protocol)")

    # --- step 1: select k of the m eigenvectors (elementary-symmetric k-DPP rule) ---
    etab = L.new_zeros((k + 1, m + 1))
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
        u = torch.rand((), generator=generator, device=L.device, dtype=L.dtype)
        if bool(u < marg):
            selected.append(j - 1)
            i -= 1

    # --- step 2: sample an elementary DPP from the selected eigenvectors ---
    Vsel = Vp[:, selected].clone()                          # [d, k]
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


# --------------------------------------------------------------------------------------------
# brute-force references (single source, tests only)
# --------------------------------------------------------------------------------------------

def bruteforce_cdq2_normalizer(a: torch.Tensor, Z: torch.Tensor, eta, k: int) -> torch.Tensor:
    """``e_k(lambda(L)) = sum_{|S|=k} (prod_{j in S} a_j) det(I_{|S|} + eta Z_S Z_S^T)`` by
    explicit enumeration over all ``C(d, k)`` subsets (single source ``a`` ``[d]``)."""
    if a.ndim != 1:
        raise ValueError("bruteforce expects a single source (a is [d])")
    d = a.shape[0]
    Zn = cdq2_unit_normalize(Z)
    e = _as_eta(eta, a)
    total = a.new_zeros(())
    for S in combinations(range(d), k):
        idx = torch.tensor(S)
        Zs = Zn[idx]
        eye = torch.eye(len(S), dtype=a.dtype, device=a.device)
        det_div = torch.det(eye + e * Zs @ Zs.transpose(-1, -2))
        total = total + a[idx].prod() * det_div
    return total


def cdq2_enumerate_distribution(a: torch.Tensor, Z: torch.Tensor, eta, k: int) -> dict:
    """Brute-force ``P(S) = det(L_S) / e_k`` over all ``C(d, k)`` subsets (single source, tests)."""
    if a.ndim != 1:
        raise ValueError("cdq2_enumerate_distribution expects a single source (a is [d])")
    d = a.shape[0]
    Zn = cdq2_unit_normalize(Z)
    e = _as_eta(eta, a)
    dets: dict[tuple[int, ...], float] = {}
    ek = 0.0
    for S in combinations(range(d), k):
        idx = torch.tensor(S)
        Zs = Zn[idx]
        eye = torch.eye(len(S), dtype=a.dtype, device=a.device)
        det = float(a[idx].prod() * torch.det(eye + e * Zs @ Zs.transpose(-1, -2)))
        det = max(det, 0.0)
        dets[tuple(int(j) for j in S)] = det
        ek += det
    return {S: v / ek for S, v in dets.items()}
