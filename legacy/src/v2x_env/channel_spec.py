"""Single source of truth for the V2X channel path-loss / shadow-fading coefficients.

Historically the channel physics was implemented TWICE with independently-typed magic
numbers: once in numpy (``channel_model.py`` / ``nr_v2x_sidelink.py``, the candidate-graph
feature path) and once in torch (``v2x_consensus_bridge._channel_proxy``, the differentiable
evaluator). The two could silently drift, and only the numpy side honoured
``pathloss_model="tr37885"`` (P0-1 / report defect F7).

This module holds the coefficients ONCE as plain Python floats and a backend-agnostic
:func:`resolve_pathloss_spec`. Both the numpy and the torch implementations import the SAME
:class:`ResolvedPathLossSpec` and apply it with their own array library (np.* / torch.*),
so:
  * "change the channel in one place" = edit this file; all three call sites update;
  * the ``tr37885`` switch flips BOTH the candidate graph and the evaluator (both read
    ``pathloss_model`` and call ``resolve_pathloss_spec``);
  * a parity contract test (tests/evaluation/test_channel_parity.py) pins numpy==torch.

It contains NO array math itself (no numpy / torch import) so it cannot couple the two
backends; it only describes the formula. The applied formula in both backends is:

    pl_los = los_intercept_db + los_log_distance_slope*log10(d) + los_log_frequency_slope*log10(fc)
    if nlos_mode == "flat_penalty":           # legacy single-slope
        pl = pl_los + (1 - los)*nlos_flat_penalty_db
        shadow_var = 0
    else:                                      # tr37885 LOS/NLOS soft-mix
        pl_nlos = nlos_intercept_db + nlos_log_distance_slope*log10(d) + nlos_log_frequency_slope*log10(fc)
        pl_non  = maximum(pl_nlos, pl_los + nlosv_extra_db)        # worse of building / vehicle block
        pl = los*pl_los + (1 - los)*pl_non
        shadow_var = los*shadow_var_los_db2 + (1 - los)*shadow_var_nlos_db2

and the BLER-vs-SINR transition is broadened by the shadow variance
(``eff_transition = sqrt(transition_width^2 + shadow_var / LOGISTIC_PROBIT^2)``); for the
legacy model ``shadow_var = 0`` so the transition is unchanged (byte-identical).

References: 3GPP TR 37.885 v15.3.0 Table 6.2.1-1 (V2V path loss), 6.2.1-2/-3 (LOS prob, shadowing).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# logistic<->probit factor pi/sqrt(3): a N(0, sigma^2) shadow integrated against a logistic
# BLER of scale s broadens to sqrt(s^2 + (sigma/LOGISTIC_PROBIT)^2). (Mirrors nr_v2x_sidelink.)
LOGISTIC_PROBIT: float = float(math.pi / math.sqrt(3.0))

# --- legacy single-slope proxy (3GPP UMi street-canyon LOS + flat NLOS penalty) ---
LEGACY_INTERCEPT_DB: float = 32.4
LEGACY_LOG_DISTANCE_SLOPE: float = 21.0
LEGACY_LOG_FREQUENCY_SLOPE: float = 20.0
LEGACY_NLOS_PENALTY_DB_DEFAULT: float = 12.0

# --- TR 37.885 V2V path loss (urban / highway) ---
TR37885_URBAN_LOS = (38.77, 16.7, 18.2)     # (intercept, log10(d) slope, log10(fc) slope), sigma=3
TR37885_URBAN_NLOS = (36.85, 30.0, 18.9)    # urban NLOS (building), sigma=4
TR37885_HIGHWAY_LOS = (32.4, 20.0, 20.0)    # highway LOS, sigma=3
TR37885_SHADOW_STD_LOS_DB: float = 3.0
TR37885_SHADOW_STD_NLOS_URBAN_DB: float = 4.0
TR37885_SHADOW_STD_NLOS_HIGHWAY_DB: float = 3.0
# TR 37.885 NLOSv vehicle-blockage extra loss std (folded into the non-LOS shadow variance).
TR37885_NLOSV_EXTRA_STD_DB_DEFAULT: float = 4.5

VALID_PATHLOSS_MODELS = ("legacy", "tr37885")
VALID_SCENARIOS = ("urban", "highway")


@dataclass(frozen=True)
class ResolvedPathLossSpec:
    """Backend-agnostic description of one path-loss model (floats only, no array math)."""

    los_intercept_db: float
    los_log_distance_slope: float
    los_log_frequency_slope: float
    # "flat_penalty" => legacy: pl = pl_los + (1-los)*nlos_flat_penalty_db, shadow_var=0.
    # "coeffs"       => tr37885: separate NLOS coeffs + NLOSv max + LOS/NLOS shadow-var mix.
    nlos_mode: str
    nlos_intercept_db: float = 0.0
    nlos_log_distance_slope: float = 0.0
    nlos_log_frequency_slope: float = 0.0
    nlos_flat_penalty_db: float = 0.0
    nlosv_extra_db: float = 0.0
    shadow_var_los_db2: float = 0.0
    shadow_var_nlos_db2: float = 0.0


def resolve_pathloss_spec(
    pathloss_model: str = "legacy",
    *,
    scenario: str = "urban",
    nlos_penalty_db: float = LEGACY_NLOS_PENALTY_DB_DEFAULT,
    nlosv_extra_db: float = 6.0,
    nlosv_extra_std_db: float = TR37885_NLOSV_EXTRA_STD_DB_DEFAULT,
) -> ResolvedPathLossSpec:
    """Resolve the coefficient set for ``pathloss_model`` (``"legacy"`` or ``"tr37885"``).

    ``nlos_penalty_db`` is the legacy flat NLOS penalty (default 12 dB). ``scenario`` /
    ``nlosv_extra_db`` / ``nlosv_extra_std_db`` only matter for ``tr37885``. The returned spec is
    applied identically by the numpy and torch backends so they stay byte-for-byte consistent.
    """
    model = str(pathloss_model)
    if model not in VALID_PATHLOSS_MODELS:
        raise ValueError(f"pathloss_model must be one of {VALID_PATHLOSS_MODELS}, got {model!r}")
    if model == "legacy":
        return ResolvedPathLossSpec(
            los_intercept_db=LEGACY_INTERCEPT_DB,
            los_log_distance_slope=LEGACY_LOG_DISTANCE_SLOPE,
            los_log_frequency_slope=LEGACY_LOG_FREQUENCY_SLOPE,
            nlos_mode="flat_penalty",
            nlos_flat_penalty_db=float(nlos_penalty_db),
            shadow_var_los_db2=0.0,
            shadow_var_nlos_db2=0.0,
        )
    sc = str(scenario)
    if sc not in VALID_SCENARIOS:
        raise ValueError(f"scenario must be one of {VALID_SCENARIOS}, got {sc!r}")
    los = TR37885_HIGHWAY_LOS if sc == "highway" else TR37885_URBAN_LOS
    if sc == "highway":
        nlos = TR37885_HIGHWAY_LOS  # highway non-LOS = LOS + NLOSv (no separate building NLOS)
        var_non = TR37885_SHADOW_STD_NLOS_HIGHWAY_DB ** 2 + float(nlosv_extra_std_db) ** 2
    else:
        nlos = TR37885_URBAN_NLOS
        var_non = TR37885_SHADOW_STD_NLOS_URBAN_DB ** 2 + float(nlosv_extra_std_db) ** 2
    return ResolvedPathLossSpec(
        los_intercept_db=los[0],
        los_log_distance_slope=los[1],
        los_log_frequency_slope=los[2],
        nlos_mode="coeffs",
        nlos_intercept_db=nlos[0],
        nlos_log_distance_slope=nlos[1],
        nlos_log_frequency_slope=nlos[2],
        nlosv_extra_db=float(nlosv_extra_db),
        shadow_var_los_db2=float(TR37885_SHADOW_STD_LOS_DB ** 2),
        shadow_var_nlos_db2=float(var_non),
    )
