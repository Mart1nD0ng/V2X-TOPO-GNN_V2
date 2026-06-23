"""G5 (spec §8 / module 3.5): rigorous finite-blocklength link-delivery (H3 core).

Acceptance: V(gamma) matches the known closed form, the headline path is the 3GPP
TR 37.885-grounded finite-blocklength model (explicit dispersion, traceable units, no
idealized channel and no legacy logistic-BLER proxy), and the end-to-end link
properties (collision / half-duplex / request / response / HARQ) are modelled
separately.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import scipy.special
import torch

from _common import GateResult, grep_repo, main_single, run_pytest  # type: ignore

from src.mainline.finite_blocklength import (  # noqa: E402
    BlocklengthSpec,
    HeadlineLinkConfig,
    channel_dispersion,
    fading_average_success,
    fbl_error,
    fbl_link_success,
    harq_residual_error,
    poll_success,
)

LOG2E = math.log2(math.e)


def run() -> GateResult:
    evidence: dict = {}

    # 1. V(gamma) vs the algebraically-independent closed form (Eq. 36).
    gamma = torch.linspace(0.0, 60.0, 600, dtype=torch.float64)
    V = channel_dispersion(gamma)
    V_alt = gamma * (gamma + 2.0) / (1.0 + gamma) ** 2 * (LOG2E ** 2)
    v_err = float((V - V_alt).abs().max())
    v0 = float(channel_dispersion(torch.tensor(0.0, dtype=torch.float64)))
    vinf = float(channel_dispersion(torch.tensor(1e8, dtype=torch.float64)))
    evidence["V(gamma)_vs_closed_form_max_err"] = f"{v_err:.2e}"
    evidence["V(0), V(inf)/(log2e)^2"] = f"{v0:.2e}, {vinf / LOG2E ** 2:.6f}"
    v_ok = v_err < 1e-12 and abs(v0) < 1e-15 and abs(vinf / LOG2E ** 2 - 1.0) < 1e-6

    # 2. eps_FBL vs independent scipy reference.
    rng = np.random.default_rng(0)
    max_eps_err = 0.0
    for _ in range(200):
        g = float(rng.uniform(0.1, 40.0))
        n = float(rng.uniform(50.0, 3000.0))
        B = float(rng.uniform(50.0, n * math.log2(1 + g) * 1.3))
        et = float(fbl_error(torch.tensor(g, dtype=torch.float64), n, B))
        C = math.log2(1 + g)
        Vv = (1 - 1 / (1 + g) ** 2) * LOG2E ** 2
        arg = (n * C - B + 0.5 * math.log2(n)) / math.sqrt(n * Vv)
        er = 0.5 * scipy.special.erfc(arg / math.sqrt(2.0))
        max_eps_err = max(max_eps_err, abs(et - er))
    evidence["eps_FBL_vs_scipy_max_err"] = f"{max_eps_err:.2e}"
    eps_ok = max_eps_err < 1e-12

    # 3. Dispersion genuinely enters eps (NOT absorbed into sqrt(n)) -- spec §8.1.
    g0 = torch.tensor(3.0, dtype=torch.float64)
    n0, B0 = 200.0, 200.0 * math.log2(4.0) * 0.9
    eps_real = float(fbl_error(g0, n0, B0))
    arg_const = (n0 * math.log2(4.0) - B0 + 0.5 * math.log2(n0)) / math.sqrt(n0 * LOG2E ** 2)
    eps_const = float(0.5 * scipy.special.erfc(arg_const / math.sqrt(2.0)))
    evidence["dispersion_effect_gap (real vs constant V)"] = f"{abs(eps_real - eps_const):.3e}"
    absorbed_ok = abs(eps_real - eps_const) > 1e-3

    # 4. Blocklength accounting traceable (Eq. 39), RE-level (comb-2 DMRS).
    spec = BlocklengthSpec(num_rb=10, agc_symbols=1, guard_symbols=1, pscch_sci_symbols=3,
                           dmrs_symbols=4, dmrs_comb_density=0.5)
    bd = spec.breakdown()
    n_uses = spec.channel_uses()
    # 5 full data symbols (600 RE) + 4 DMRS symbols * 50% data (240 RE) = 840
    block_ok = bd["fully_data_symbols"] == 5.0 and n_uses == 840.0 and bd["dmrs_surviving_data_re"] == 240.0
    evidence["blocklength_n (10 RB, comb-2 DMRS)"] = f"{n_uses:.0f} channel uses (5 data sym + 240 DMRS-survivor RE)"

    # 5. End-to-end properties modelled separately (Eq. 41) + finite HARQ.
    pc = torch.tensor(0.1, dtype=torch.float64)
    phd = torch.tensor(0.1, dtype=torch.float64)
    er = torch.tensor(0.2, dtype=torch.float64)
    es = torch.tensor(0.15, dtype=torch.float64)
    pll = float(poll_success(p_collision=pc, p_half_duplex=phd, eps_request=er, eps_response=es))
    pll_ref = float((1 - pc) * (1 - phd) * (1 - er) * (1 - es))
    # marginal SNR (gamma=1 -> C=1 bit/use, B/n=1.25 > C): a single attempt fails, finite
    # HARQ combining recovers it -- a visibly meaningful HARQ residual sequence.
    harq = [float(harq_residual_error(torch.tensor(1.0, dtype=torch.float64), 200.0, 250.0, m)) for m in [1, 2, 3]]
    poll_ok = abs(pll - pll_ref) < 1e-12 and harq[0] > harq[1] > harq[2]
    evidence["poll_composition_err"] = f"{abs(pll - pll_ref):.1e}"
    evidence["harq_residual (M=1,2,3)"] = f"{harq[0]:.2e} -> {harq[1]:.2e} -> {harq[2]:.2e}"

    # 6. Headline config is 3GPP-grounded (no idealized channel).
    headline_ok = True
    try:
        HeadlineLinkConfig().assert_headline_grounded()
        for bad in (HeadlineLinkConfig(idealized_channel=True), HeadlineLinkConfig(use_finite_blocklength=False)):
            try:
                bad.assert_headline_grounded()
                headline_ok = False
            except AssertionError:
                pass
    except AssertionError:
        headline_ok = False
    evidence["headline_grounded_and_rejects_idealized"] = headline_ok

    # 7. Mainline link path uses the FBL/dispersion path, NOT a legacy logistic-BLER proxy.
    logistic_hits = grep_repo(
        r"mcs_sinr_threshold|bler_transition|1\.0 ?/ ?\(1\.0 ?\+ ?(np|torch)?\.?exp\(-\(sinr",
        globs=("src/mainline/*.py",),
    )
    has_dispersion = grep_repo(r"channel_dispersion|erfc", globs=("src/mainline/finite_blocklength.py",))
    evidence["legacy_logistic_bler_in_mainline"] = f"{len(logistic_hits)} hits"
    evidence["fbl_dispersion_present"] = f"{len(has_dispersion)} refs"
    grep_ok = len(logistic_hits) == 0 and len(has_dispersion) > 0

    # 8. Fading average matches Monte-Carlo (small AND large blocklength via adaptive quad).
    gm = torch.tensor(5.0, dtype=torch.float64)
    analytic = float(fading_average_success(gm, 200.0, 250.0))
    gen = torch.Generator().manual_seed(11)
    h2 = -torch.log(torch.rand(600000, generator=gen, dtype=torch.float64))
    mc = float(fbl_link_success(gm * h2, 200.0, 250.0).mean())
    # large-n sharp-transition regime (the headline wideband corner the blocker exposed)
    gm2 = torch.tensor(30.0, dtype=torch.float64)
    n2, B2 = 25920.0, 0.4 * 25920.0 * math.log2(31.0)
    analytic2 = float(fading_average_success(gm2, n2, B2))
    h2b = -torch.log(torch.rand(4000000, generator=gen, dtype=torch.float64))
    mc2 = float(fbl_link_success(gm2 * h2b, n2, B2).mean())
    evidence["fading_vs_MC (n=200 / n=25920)"] = f"{abs(analytic - mc):.2e} / {abs(analytic2 - mc2):.2e}"
    fade_ok = abs(analytic - mc) < 3e-3 and abs(analytic2 - mc2) < 1e-3

    # 9. Headline FBL chain produces a valid end-to-end ell (the wiring point for G1).
    cfg = HeadlineLinkConfig()
    d = torch.tensor([20.0, 80.0, 200.0], dtype=torch.float64)
    los = torch.ones(3, dtype=torch.float64)
    ell = cfg.compute_link_reliability(d, los, interference_dbm=-95.0, response_bits=300.0,
                                       concurrent_tx=torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
    wire_ok = bool(torch.all(ell >= 0) and torch.all(ell <= 1) and float(ell[0]) >= float(ell[2]))
    evidence["headline_compute_link_reliability (d=20/80/200m)"] = f"{float(ell[0]):.3f}/{float(ell[1]):.3f}/{float(ell[2]):.3f}"

    # 10. HONEST scope: the FBL chain is the designated mainline ell producer, but the
    #     legacy logistic-BLER ell producer still exists in the (un-migrated) production
    #     pipeline. Report it transparently; its removal/migration is tracked for G10 (H5).
    legacy_logistic_eval = grep_repo(r"torch\.sigmoid\(\(.*sinr|mcs_sinr_threshold|bler_transition",
                                     globs=("src/evaluation/*.py", "src/v2x_env/*.py"))
    evidence["legacy_logistic_in_production (tracked for G10)"] = f"{len(legacy_logistic_eval)} site(s) -- NOT yet migrated"

    tests_ok, tail = run_pytest("tests/test_g5_finite_blocklength.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    return GateResult(
        gate="G5",
        title="rigorous finite-blocklength link-delivery (explicit channel dispersion)",
        passed=bool(v_ok and eps_ok and absorbed_ok and block_ok and poll_ok and headline_ok and grep_ok and fade_ok and wire_ok and tests_ok),
        evidence=evidence,
        notes="PPV normal approximation with explicit V(gamma); 3GPP-grounded headline; no logistic proxy.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
