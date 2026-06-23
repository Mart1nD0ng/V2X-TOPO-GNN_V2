"""Rigorous finite-blocklength (FBL) link-delivery model (spec §8, H3 core).

Replaces the legacy logistic BLER-vs-SINR sigmoid (``nr_v2x_sidelink.link_success``),
which has NO channel dispersion and NO blocklength -- the "missing-dispersion
short-blocklength proxy" the spec §0 forbids.  Here the per-link delivery uses the
Polyanskiy-Poor-Verdu normal approximation with EXPLICIT channel dispersion ``V(gamma)``
over real complex channel uses:

    C(gamma) = log2(1 + gamma)                                                      (Eq. 35)
    V(gamma) = (1 - (1+gamma)^-2) (log2 e)^2                                         (Eq. 36)
    eps_FBL(gamma, n, B) = Q( (n C(gamma) - B + 0.5 log2 n) / sqrt(n V(gamma)) )     (Eq. 37)
    ell_FBL = 1 - eps_FBL                                                            (Eq. 38)

``gamma`` is the linear SINR, ``n`` the number of complex channel uses (spec §8.2,
Eq. 39, with explicit DMRS / SCI / guard / reserved-RE deductions), and ``B`` the
payload+CRC+header bits.  ``V(gamma)`` is NEVER absorbed into ``sqrt(n)`` (spec §8.1).

A valid poll needs request AND response, with collision, half-duplex and the two PHY
decoding errors modelled SEPARATELY (Eq. 41):

    ell_poll = (1 - p_col)(1 - p_HD)(1 - eps_req)(1 - eps_resp).

Quasi-static fading is averaged by deterministic quadrature (Eq. 40), and finite HARQ
is modelled explicitly (Chase / incremental-redundancy combining over <= M attempts).

Per spec §8.4 this is a "finite-blocklength normal-approximation link-delivery model
with explicit channel dispersion" -- NOT claimed as exact NR BLER.  The headline
configuration uses only the 3GPP TR 37.885-grounded path (no idealized channel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

LOG2E = math.log2(math.e)  # 1.4426950408889634
LOG2E_SQ = LOG2E * LOG2E

__all__ = [
    "gaussian_q",
    "channel_capacity",
    "channel_dispersion",
    "fbl_error",
    "fbl_link_success",
    "BlocklengthSpec",
    "payload_bits",
    "mode2_collision_probability",
    "harq_residual_error",
    "poll_success",
    "fading_average_success",
    "PathLoss3GPP",
    "sinr_linear_from_geometry",
    "HeadlineLinkConfig",
]


def _as_tensor(x, ref: torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def gaussian_q(x: torch.Tensor) -> torch.Tensor:
    """Gaussian tail ``Q(x) = 0.5 erfc(x / sqrt 2)`` (differentiable)."""
    return 0.5 * torch.erfc(x / math.sqrt(2.0))


def channel_capacity(gamma: torch.Tensor) -> torch.Tensor:
    """Shannon capacity ``C(gamma) = log2(1 + gamma)`` [bits / channel use] (Eq. 35)."""
    return torch.log2(1.0 + gamma)


def channel_dispersion(gamma: torch.Tensor) -> torch.Tensor:
    """Channel dispersion ``V(gamma) = (1 - (1+gamma)^-2)(log2 e)^2`` [bits^2 / use] (Eq. 36).

    Explicit and never folded into ``sqrt(n)`` (spec §8.1).  ``V(0)=0``;
    ``V -> (log2 e)^2`` as ``gamma -> inf``; monotone increasing.
    """
    inv = 1.0 / (1.0 + gamma)
    return (1.0 - inv * inv) * LOG2E_SQ


def fbl_error(
    gamma: torch.Tensor,
    n: torch.Tensor | float,
    B: torch.Tensor | float,
    *,
    gamma_floor: float = 1e-9,
) -> torch.Tensor:
    """Finite-blocklength decoding error ``eps_FBL(gamma, n, B)`` (Eq. 37).

    Args:
        gamma: linear SINR (>= 0).
        n: complex channel uses (>= 1).
        B: information bits (payload + CRC + header).

    All units are traceable: numerator ``n C - B + 0.5 log2 n`` and denominator
    ``sqrt(n V)`` are both in bits, so the Q argument is dimensionless.
    """
    g = gamma if isinstance(gamma, torch.Tensor) else torch.as_tensor(gamma, dtype=torch.float64)
    g = g.clamp_min(gamma_floor)
    n_t = _as_tensor(n, g)
    B_t = _as_tensor(B, g)
    if bool(torch.any(n_t.detach() < 1.0).cpu()):
        raise ValueError("blocklength n must be >= 1 complex channel use")
    C = channel_capacity(g)
    V = channel_dispersion(g)
    num = n_t * C - B_t + 0.5 * torch.log2(n_t)
    den = torch.sqrt(n_t * V).clamp_min(torch.finfo(g.dtype).tiny)
    return gaussian_q(num / den)


def fbl_link_success(gamma, n, B, *, gamma_floor: float = 1e-9) -> torch.Tensor:
    """Per-link delivery probability ``ell_FBL = 1 - eps_FBL`` (Eq. 38)."""
    return 1.0 - fbl_error(gamma, n, B, gamma_floor=gamma_floor)


@dataclass(frozen=True)
class BlocklengthSpec:
    """Complex channel-use accounting at RESOURCE-ELEMENT granularity (spec §8.2, Eq. 39).

    The PSSCH data REs left after deducting, in physical NR-sidelink units:
      * AGC + guard symbols (carry no data) -> whole symbols removed;
      * PSCCH/SCI control region -> ``pscch_sci_symbols`` symbols over ``pscch_prbs`` PRBs;
      * DMRS -> comb-mapped within ``dmrs_symbols`` symbols, so a fraction
        ``(1 - dmrs_comb_density)`` of those symbols' REs still carry DATA (TS 38.211
        §8.4.1.1.2 config type-1 = comb-2 => ``dmrs_comb_density = 0.5``);
      * explicit reserved / pilot REs.

    Every deduction is explicit and traceable in :meth:`breakdown`.  Modelling DMRS at
    RE granularity (not whole symbols) avoids a ~28% pessimistic understatement of ``n``.
    """

    num_rb: float = 10.0
    sc_per_rb: int = 12
    sym_per_slot: int = 14
    agc_symbols: float = 1.0          # automatic gain control (no data)
    guard_symbols: float = 1.0        # guard / Tx-Rx turnaround (no data)
    pscch_sci_symbols: float = 2.0    # PSCCH carrying SCI (control region), 2 symbols
    pscch_prbs: float | None = None   # PSCCH sub-band PRBs (default: full band = num_rb)
    dmrs_symbols: float = 4.0         # PSSCH DMRS symbols (comb-mapped)
    dmrs_comb_density: float = 0.5    # fraction of a DMRS symbol's REs used for DMRS (comb-2)
    reserved_re: float = 0.0          # extra reserved resource elements
    pilot_re: float = 0.0             # additional pilot/reference REs

    def _pscch_prbs(self) -> float:
        return self.num_rb if self.pscch_prbs is None else min(float(self.pscch_prbs), self.num_rb)

    def data_symbols(self) -> float:
        """Number of FULLY-data PSSCH symbols (excludes AGC/guard/PSCCH/DMRS symbols)."""
        d = self.sym_per_slot - self.agc_symbols - self.guard_symbols - self.pscch_sci_symbols - self.dmrs_symbols
        if d < 0:
            raise ValueError("no data symbols left after deductions; check BlocklengthSpec")
        return float(d)

    def dmrs_surviving_data_re(self) -> float:
        """Data REs that survive inside DMRS symbols (comb mapping leaves data subcarriers)."""
        return self.dmrs_symbols * self.num_rb * self.sc_per_rb * (1.0 - self.dmrs_comb_density)

    def pscch_surviving_data_re(self) -> float:
        """Data REs in PSCCH symbols outside the PSCCH sub-band (full-band PSCCH -> 0)."""
        return self.pscch_sci_symbols * (self.num_rb - self._pscch_prbs()) * self.sc_per_rb

    def channel_uses(self) -> float:
        re_per_symbol = self.num_rb * self.sc_per_rb
        full_data_re = self.data_symbols() * re_per_symbol
        n = (full_data_re + self.dmrs_surviving_data_re() + self.pscch_surviving_data_re()
             - self.reserved_re - self.pilot_re)
        if n < 1.0:
            raise ValueError("blocklength < 1 channel use; check BlocklengthSpec")
        return float(n)

    def breakdown(self) -> dict:
        re_per_symbol = self.num_rb * self.sc_per_rb
        return {
            "num_rb": self.num_rb,
            "sc_per_rb": self.sc_per_rb,
            "sym_per_slot": self.sym_per_slot,
            "fully_data_symbols": self.data_symbols(),
            "fully_data_re": self.data_symbols() * re_per_symbol,
            "dmrs_symbols": self.dmrs_symbols,
            "dmrs_surviving_data_re": self.dmrs_surviving_data_re(),
            "pscch_surviving_data_re": self.pscch_surviving_data_re(),
            "agc_guard_re_removed": (self.agc_symbols + self.guard_symbols) * re_per_symbol,
            "pscch_sci_re_removed": self.pscch_sci_symbols * self._pscch_prbs() * self.sc_per_rb,
            "reserved_re": self.reserved_re,
            "pilot_re": self.pilot_re,
            "channel_uses": self.channel_uses(),
        }


def payload_bits(payload_bytes: float, *, crc_bits: float = 24.0, header_bits: float = 0.0) -> float:
    """Information bits ``B`` = payload + CRC + protocol header (spec §8.2)."""
    return float(payload_bytes) * 8.0 + float(crc_bits) + float(header_bits)


def mode2_collision_probability(concurrent_tx: torch.Tensor, subchannels: float) -> torch.Tensor:
    """Mode-2 SPS resource-collision probability ``1 - (1 - 1/S)^(N-1)`` (differentiable in N)."""
    if subchannels < 1.0:
        raise ValueError("subchannels must be >= 1")
    others = (concurrent_tx - 1.0).clamp_min(0.0)
    return 1.0 - torch.pow(torch.as_tensor(1.0 - 1.0 / float(subchannels), dtype=others.dtype, device=others.device), others)


def harq_residual_error(
    gamma: torch.Tensor,
    n: torch.Tensor | float,
    B: torch.Tensor | float,
    max_attempts: int,
    *,
    combining: str = "chase",
    gamma_floor: float = 1e-9,
) -> torch.Tensor:
    """Residual decoding error after <= ``max_attempts`` finite HARQ transmissions.

    - ``chase``: maximal-ratio combining of identical copies -> effective SNR ``M*gamma``.
    - ``ir``: incremental redundancy -> accumulated channel uses ``M*n`` (lower rate).

    Both are monotone non-increasing in ``max_attempts`` and modelled separately from
    collision / half-duplex (spec §8.3).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    g = gamma if isinstance(gamma, torch.Tensor) else torch.as_tensor(gamma, dtype=torch.float64)
    # Running minimum over m = 1..M guarantees monotone non-increasing residual by
    # construction (the decoder keeps the best combined attempt).  The plain
    # eps_FBL(M*gamma) / eps_FBL(M*n) is already monotone for any physical payload
    # (B >= CRC width); the cummin only changes the sub-CRC corner (B < ~1 bit), which
    # is outside the reachable domain (spec §8.2 fixes B = payload + CRC + header).
    residual = None
    for m in range(1, max_attempts + 1):
        if combining == "chase":
            e = fbl_error(float(m) * g, n, B, gamma_floor=gamma_floor)
        elif combining == "ir":
            n_t = _as_tensor(n, g)
            e = fbl_error(g, float(m) * n_t, B, gamma_floor=gamma_floor)
        else:
            raise ValueError("combining must be 'chase' or 'ir'")
        residual = e if residual is None else torch.minimum(residual, e)
    return residual


def poll_success(
    *,
    p_collision: torch.Tensor,
    p_half_duplex: torch.Tensor,
    eps_request: torch.Tensor,
    eps_response: torch.Tensor,
) -> torch.Tensor:
    """Effective poll success ``(1-p_col)(1-p_HD)(1-eps_req)(1-eps_resp)`` (Eq. 41).

    Collision, half-duplex and the two PHY decoding errors are kept as separate factors
    (spec §8.3): a valid poll requires the request (i->j) and response (j->i) to both be
    delivered, neither node to be half-duplex-blocked, and no resource collision.
    """
    return (1.0 - p_collision) * (1.0 - p_half_duplex) * (1.0 - eps_request) * (1.0 - eps_response)


# Gauss-Legendre nodes/weights on [0,1] (precomputed; no Date/random).  Used with the
# inverse-CDF substitution u = 1 - e^{-x}, x = -ln(1-u) so that for X ~ Exp(1)
#   E[g(X)] = int_0^inf g(x) e^{-x} dx = int_0^1 g(-ln(1-u)) du.
# Gauss-Legendre on the bounded interval converges spectrally for the smooth Q-function
# integrand and -- unlike Gauss-Laguerre -- has no high-order weight overflow.
def _gauss_legendre_unit(num_points: int) -> tuple[list[float], list[float]]:
    import numpy as np
    x, w = np.polynomial.legendre.leggauss(num_points)  # nodes/weights on [-1, 1]
    u = (x + 1.0) / 2.0
    wu = w / 2.0
    return u.tolist(), wu.tolist()


def _auto_num_quad(n: torch.Tensor | float) -> int:
    """Pick a Gauss-Legendre node count that resolves the FBL transition.

    The FBL success-vs-|H|^2 transition narrows as ``~1/sqrt(n)`` (the eps slope scales
    with ``sqrt(n V)``), so the node count must grow as ``~sqrt(n)`` to keep the
    quadrature error below ~1e-4 across the headline SINR / blocklength envelope.
    """
    n_max = float(n.max()) if isinstance(n, torch.Tensor) else float(n)
    return int(min(1024, max(96, math.ceil(3.0 * math.sqrt(max(n_max, 1.0))))))


def fading_average_success(
    gamma_mean: torch.Tensor,
    n: torch.Tensor | float,
    B: torch.Tensor | float,
    *,
    fading: str = "rayleigh",
    num_quad: int | None = None,
    gamma_floor: float = 1e-9,
) -> torch.Tensor:
    """Quasi-static-fading-averaged delivery ``E_H[1 - eps_FBL(gamma_mean |H|^2, n, B)]`` (Eq. 40).

    ``rayleigh``: ``|H|^2 ~ Exp(1)`` (mean 1).  Computed by the inverse-CDF substitution
    ``x = -ln(1-u)`` and Gauss-Legendre quadrature on ``u in [0,1]``:
    ``E[g(gamma_mean |H|^2)] = sum_i w_i g(gamma_mean * (-ln(1-u_i)))``.  Deterministic and
    differentiable in ``gamma_mean`` -- no Monte-Carlo.  ``none`` returns the unfaded value.

    ``num_quad`` defaults to an adaptive count (``~3 sqrt(n)``, clamped to [96, 1024]) so
    the sharp large-blocklength transition stays resolved (96 fixed points under-resolve
    it at ``n >~ 3000`` -- the wideband headline regime).
    """
    g = gamma_mean if isinstance(gamma_mean, torch.Tensor) else torch.as_tensor(gamma_mean, dtype=torch.float64)
    if fading == "none":
        return fbl_link_success(g, n, B, gamma_floor=gamma_floor)
    if fading != "rayleigh":
        raise ValueError("fading must be 'rayleigh' or 'none'")
    q = _auto_num_quad(n) if num_quad is None else num_quad
    u_nodes, u_weights = _gauss_legendre_unit(q)
    acc = torch.zeros_like(g)
    for ui, wi in zip(u_nodes, u_weights):
        xi = -math.log(1.0 - ui)  # inverse CDF of Exp(1)
        acc = acc + wi * fbl_link_success(g * xi, n, B, gamma_floor=gamma_floor)
    return acc.clamp(0.0, 1.0)


# --- 3GPP TR 37.885 path-loss grounding (mainline-owned constants; no legacy import) ---
@dataclass(frozen=True)
class PathLoss3GPP:
    """TR 37.885 V2V path-loss coefficients ``(intercept, log10 d slope, log10 fc slope)``
    and LOS/non-LOS shadow std [dB].  Urban defaults (Table 6.2.1-1)."""

    los: tuple[float, float, float] = (38.77, 16.7, 18.2)
    nlos: tuple[float, float, float] = (36.85, 30.0, 18.9)
    nlosv_extra_db: float = 6.0
    shadow_std_los_db: float = 3.0
    shadow_std_nlos_db: float = 4.0


def sinr_linear_from_geometry(
    distance_m: torch.Tensor,
    los_prob: torch.Tensor,
    *,
    tx_power_dbm: float,
    noise_dbm: float,
    interference_dbm: torch.Tensor | float,
    fc_ghz: float,
    pathloss: PathLoss3GPP | None = None,
) -> torch.Tensor:
    """Linear SINR ``gamma`` from TR 37.885 geometry (the headline 3GPP-grounded path).

    Soft-mixes LOS / non-LOS path loss by ``los_prob``; non-LOS is the worse of the NLOS
    building model and ``LOS + NLOSv`` vehicle blockage.  Differentiable in distance and
    interference.  This is the only physically-grounded ``gamma`` source for the headline
    (no idealized / threshold channel).
    """
    pl = pathloss or PathLoss3GPP()
    d = distance_m.clamp_min(1.0)
    log_d = torch.log10(d)
    log_fc = math.log10(fc_ghz)
    pl_los = pl.los[0] + pl.los[1] * log_d + pl.los[2] * log_fc
    pl_nlos = pl.nlos[0] + pl.nlos[1] * log_d + pl.nlos[2] * log_fc
    pl_non = torch.maximum(pl_nlos, pl_los + pl.nlosv_extra_db)
    los = los_prob.clamp(0.0, 1.0)
    pl_db = los * pl_los + (1.0 - los) * pl_non
    rx_dbm = tx_power_dbm - pl_db
    signal_mw = torch.pow(torch.as_tensor(10.0, dtype=rx_dbm.dtype, device=rx_dbm.device), rx_dbm / 10.0)
    interf = _as_tensor(interference_dbm, rx_dbm)
    noise_mw = 10.0 ** (noise_dbm / 10.0)
    impair_mw = noise_mw + torch.pow(torch.as_tensor(10.0, dtype=rx_dbm.dtype, device=rx_dbm.device), interf / 10.0)
    return (signal_mw / impair_mw).clamp_min(1e-12)


def _normal_quadrature(num_points: int) -> tuple[list[float], list[float]]:
    """Standard-normal quadrature ``E_{Z~N(0,1)}[g(Z)] = sum_i w_i g(z_i)``.

    Uses the probit substitution ``z = Phi^-1(u)`` and Gauss-Legendre on ``u in [0,1]``
    (same robust, overflow-free, sharp-transition-friendly scheme as the Rayleigh fade
    average -- unlike Gauss-Hermite, the node count can grow without weight overflow).
    """
    import numpy as np
    import scipy.special
    u, wu = _gauss_legendre_unit(num_points)
    z = scipy.special.ndtri(np.asarray(u))  # inverse normal CDF, finite at interior nodes
    return z.tolist(), wu


def averaged_link_success(
    gamma_base: torch.Tensor,
    n: torch.Tensor | float,
    B: torch.Tensor | float,
    *,
    max_harq_attempts: int = 1,
    harq_combining: str = "chase",
    shadow_std_db: torch.Tensor | float = 0.0,
    fading: str = "rayleigh",
    num_shadow_quad: int = 9,
    num_fade_quad: int | None = None,
    gamma_floor: float = 1e-9,
) -> torch.Tensor:
    """Shadow- and small-scale-fading-averaged HARQ delivery probability.

    Computes ``E_shadow E_|H|^2 [ 1 - eps_HARQ(gamma_base * 10^(shadow/10) * |H|^2, n, B) ]``:
      * log-normal shadow ``shadow ~ N(0, shadow_std_db^2)`` by Gauss-Hermite quadrature
        (makes the TR 37.885 shadow std a live part of the link, spec §8.3 / Table 6.2.1-1);
      * Rayleigh small-scale ``|H|^2 ~ Exp(1)`` by the inverse-CDF Gauss-Legendre rule;
      * finite HARQ (Chase / IR) inside the expectation.
    Deterministic and differentiable -- no Monte-Carlo.
    """
    g = gamma_base if isinstance(gamma_base, torch.Tensor) else torch.as_tensor(gamma_base, dtype=torch.float64)
    sstd = _as_tensor(shadow_std_db, g)
    # Shadow quadrature: probit + Gauss-Legendre (weights sum to 1).  When small-scale
    # fading is OFF the shadow integrand is a sharp step at large n, so the node count must
    # adapt like the fade quadrature; with Rayleigh smoothing a small fixed count suffices.
    shadow_active = bool((sstd.abs() > 0).any().cpu())
    if shadow_active:
        n_shadow = _auto_num_quad(n) if fading == "none" else num_shadow_quad
        z_nodes, z_weights = _normal_quadrature(n_shadow)
    else:
        z_nodes, z_weights = [0.0], [1.0]
    q = _auto_num_quad(n) if num_fade_quad is None else num_fade_quad
    if fading == "rayleigh":
        u_nodes, u_weights = _gauss_legendre_unit(q)
        fade_x = [(-math.log(1.0 - ui), wi) for ui, wi in zip(u_nodes, u_weights)]
    elif fading == "none":
        fade_x = [(1.0, 1.0)]
    else:
        raise ValueError("fading must be 'rayleigh' or 'none'")
    ten = torch.as_tensor(10.0, dtype=g.dtype, device=g.device)
    acc = torch.zeros_like(g)
    for zi, zw in zip(z_nodes, z_weights):
        shadow_lin = torch.pow(ten, (sstd * zi) / 10.0)
        inner = torch.zeros_like(g)
        for xj, wj in fade_x:
            gamma_eff = g * shadow_lin * xj
            eps = harq_residual_error(gamma_eff, n, B, max_harq_attempts,
                                      combining=harq_combining, gamma_floor=gamma_floor)
            inner = inner + wj * (1.0 - eps)
        acc = acc + zw * inner
    return acc.clamp(0.0, 1.0)


@dataclass(frozen=True)
class HeadlineLinkConfig:
    """Headline link configuration: 3GPP TR 37.885 geometry + FBL delivery, no idealized
    channel (spec §8.4).  ``assert_headline_grounded`` enforces the G5 invariant, and
    ``compute_link_reliability`` is the designated headline ``ell`` producer (geometry ->
    SINR -> shadow/fading-averaged FBL -> HARQ -> poll) that wires into the §6 evaluator's
    ``link_reliability`` input.
    """

    scenario: str = "urban"
    fc_ghz: float = 5.9
    tx_power_dbm: float = 23.0
    noise_dbm: float = -95.0
    subchannels: float = 5.0
    half_duplex: bool = True
    max_harq_attempts: int = 2
    harq_combining: str = "chase"
    fading: str = "rayleigh"
    use_shadow_fading: bool = True
    use_finite_blocklength: bool = True   # headline MUST be True
    idealized_channel: bool = False        # headline MUST be False (ablation-only)
    request_bits: float = 48.0             # SCI-2 + CRC (request direction)
    blocklength: BlocklengthSpec = field(default_factory=BlocklengthSpec)
    pathloss: PathLoss3GPP = field(default_factory=PathLoss3GPP)

    def assert_headline_grounded(self) -> None:
        if self.idealized_channel:
            raise AssertionError("headline config must not use an idealized channel (spec §8.4)")
        if not self.use_finite_blocklength:
            raise AssertionError("headline config must use the finite-blocklength path (H3)")
        if self.scenario not in ("urban", "highway"):
            raise AssertionError("headline scenario must be a TR 37.885 scenario")

    def compute_link_reliability(
        self,
        distance_m: torch.Tensor,
        los_prob: torch.Tensor,
        *,
        interference_dbm: torch.Tensor | float,
        response_bits: torch.Tensor | float,
        concurrent_tx: torch.Tensor | float | None = None,
        p_half_duplex: torch.Tensor | float = 0.0,
        request_blocklength: float | None = None,
    ) -> torch.Tensor:
        """Headline per-poll delivery ``ell_poll`` from geometry (spec §8.3-§8.4, Eq. 41).

        Wires the full 3GPP-grounded FBL chain: TR 37.885 SINR -> shadow + Rayleigh +
        finite-HARQ averaged request (SCI) and response (data) PHY errors -> mode-2
        collision -> half-duplex.  This is the value that feeds ``evaluate_global_consensus
        (link_reliability=...)`` on the headline path -- NO logistic-BLER proxy, NO
        idealized channel.  Differentiable end-to-end.
        """
        self.assert_headline_grounded()
        gamma = sinr_linear_from_geometry(
            distance_m, los_prob, tx_power_dbm=self.tx_power_dbm, noise_dbm=self.noise_dbm,
            interference_dbm=interference_dbm, fc_ghz=self.fc_ghz, pathloss=self.pathloss,
        )
        los = los_prob.clamp(0.0, 1.0)
        shadow_std = (los * self.pathloss.shadow_std_los_db + (1.0 - los) * self.pathloss.shadow_std_nlos_db
                      if self.use_shadow_fading else torch.zeros_like(los))
        n_resp = self.blocklength.channel_uses()
        n_req = request_blocklength if request_blocklength is not None else max(1.0, n_resp * 0.1)
        succ_resp = averaged_link_success(
            gamma, n_resp, response_bits, max_harq_attempts=self.max_harq_attempts,
            harq_combining=self.harq_combining, shadow_std_db=shadow_std, fading=self.fading,
        )
        succ_req = averaged_link_success(
            gamma, n_req, self.request_bits, max_harq_attempts=self.max_harq_attempts,
            harq_combining=self.harq_combining, shadow_std_db=shadow_std, fading=self.fading,
        )
        if concurrent_tx is None:
            p_col = torch.zeros_like(gamma)
        else:
            ctx = _as_tensor(concurrent_tx, gamma)
            p_col = mode2_collision_probability(ctx, self.subchannels)
        p_hd = _as_tensor(p_half_duplex, gamma)
        return poll_success(
            p_collision=p_col, p_half_duplex=p_hd,
            eps_request=1.0 - succ_req, eps_response=1.0 - succ_resp,
        )
