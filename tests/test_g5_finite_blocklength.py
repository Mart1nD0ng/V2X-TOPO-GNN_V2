"""G5 (spec §8): rigorous finite-blocklength link-delivery model (H3 core).

Checks:
  1. V(gamma) matches an algebraically-independent closed form + known limits (Eq. 36).
  2. eps_FBL matches an independent scipy reference; units traceable (Eq. 37).
  3. V(gamma) genuinely enters eps (NOT absorbed into sqrt(n)) -- spec §8.1.
  4. Monotonicity (eps down in gamma, down in n below capacity, up in B) and the
     n->inf asymptotics (eps->0 if R<C, eps->1 if R>C).
  5. Gradient of eps w.r.t. gamma / n / B matches central finite differences.
  6. Blocklength accounting in physical RE units with explicit deductions (Eq. 39).
  7. Finite HARQ residual error decreases monotonically in attempts (Eqs. §8.3).
  8. Poll composition = product of separate collision/HD/req/resp factors (Eq. 41).
  9. Fading average matches Monte-Carlo over |H|^2; mode-2 collision matches formula.
 10. Headline config is 3GPP-grounded (no idealized channel); SINR sane vs distance.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.special
import torch

from src.mainline.finite_blocklength import (
    BlocklengthSpec,
    HeadlineLinkConfig,
    channel_capacity,
    channel_dispersion,
    fading_average_success,
    fbl_error,
    fbl_link_success,
    gaussian_q,
    harq_residual_error,
    mode2_collision_probability,
    payload_bits,
    poll_success,
    sinr_linear_from_geometry,
)

torch.manual_seed(0)
DT = torch.float64
LOG2E = math.log2(math.e)


def test_dispersion_matches_independent_form_and_limits():
    gamma = torch.linspace(0.0, 50.0, 200, dtype=DT)
    V = channel_dispersion(gamma)
    # algebraically-independent form: V = gamma(gamma+2)/(1+gamma)^2 * (log2 e)^2
    V_alt = gamma * (gamma + 2.0) / (1.0 + gamma) ** 2 * (LOG2E ** 2)
    assert torch.allclose(V, V_alt, atol=1e-12)
    # limits
    assert abs(float(channel_dispersion(torch.tensor(0.0, dtype=DT)))) < 1e-15  # V(0)=0
    assert float(channel_dispersion(torch.tensor(1e6, dtype=DT))) < LOG2E ** 2
    assert float(channel_dispersion(torch.tensor(1e6, dtype=DT))) > 0.999 * LOG2E ** 2
    # monotone increasing
    assert torch.all(V[1:] - V[:-1] >= -1e-12)


def test_fbl_error_matches_scipy_reference():
    rng = np.random.default_rng(1)
    max_err = 0.0
    for _ in range(50):
        gamma = float(rng.uniform(0.2, 30.0))
        n = float(rng.uniform(50.0, 2000.0))
        B = float(rng.uniform(50.0, n * math.log2(1 + gamma) * 1.2))
        eps_torch = float(fbl_error(torch.tensor(gamma, dtype=DT), n, B))
        # independent numpy/scipy reference
        C = math.log2(1.0 + gamma)
        V = (1.0 - 1.0 / (1.0 + gamma) ** 2) * (LOG2E ** 2)
        arg = (n * C - B + 0.5 * math.log2(n)) / math.sqrt(n * V)
        eps_ref = 0.5 * scipy.special.erfc(arg / math.sqrt(2.0))
        max_err = max(max_err, abs(eps_torch - eps_ref))
    assert max_err < 1e-12, max_err


def test_gaussian_q_reference():
    x = torch.linspace(-4, 4, 50, dtype=DT)
    q = gaussian_q(x)
    ref = 0.5 * scipy.special.erfc(x.numpy() / math.sqrt(2.0))
    assert np.allclose(q.numpy(), ref, atol=1e-12)


def test_dispersion_not_absorbed():
    # spec §8.1: V(gamma) must genuinely enter eps. Replacing V(gamma) with the
    # asymptotic constant (log2 e)^2 must change eps materially at moderate gamma.
    gamma = torch.tensor(3.0, dtype=DT)
    n, B = 200.0, 200.0 * math.log2(4.0) * 0.9
    eps_real = float(fbl_error(gamma, n, B))
    # constant-dispersion variant (the forbidden absorption)
    C = math.log2(1.0 + 3.0)
    V_const = LOG2E ** 2
    arg = (n * C - B + 0.5 * math.log2(n)) / math.sqrt(n * V_const)
    eps_const = float(0.5 * scipy.special.erfc(arg / math.sqrt(2.0)))
    assert abs(eps_real - eps_const) > 1e-3, (eps_real, eps_const)


def test_monotonicity():
    n, B = 300.0, 200.0
    g = torch.linspace(0.5, 10.0, 50, dtype=DT)
    eps_g = fbl_error(g, n, B)
    assert torch.all(eps_g[1:] - eps_g[:-1] <= 1e-9)  # decreasing in gamma
    # below capacity (R < C): increasing n lowers eps
    gamma = torch.tensor(4.0, dtype=DT)  # C = log2(5) ~ 2.32
    R = 1.5
    ns = torch.tensor([100.0, 200.0, 400.0, 800.0], dtype=DT)
    eps_n = torch.stack([fbl_error(gamma, float(nn), R * float(nn)) for nn in ns])
    assert torch.all(eps_n[1:] - eps_n[:-1] <= 1e-9)
    # increasing B raises eps
    Bs = torch.tensor([100.0, 200.0, 300.0], dtype=DT)
    eps_B = torch.stack([fbl_error(gamma, 300.0, float(bb)) for bb in Bs])
    assert torch.all(eps_B[1:] - eps_B[:-1] >= -1e-9)


def test_asymptotics_capacity():
    gamma = torch.tensor(3.0, dtype=DT)  # C = log2(4) = 2 exactly
    # rate below capacity -> eps -> 0
    eps_below = float(fbl_error(gamma, 5000.0, 1.5 * 5000.0))
    assert eps_below < 1e-3, eps_below
    # rate above capacity -> eps -> 1
    eps_above = float(fbl_error(gamma, 5000.0, 2.5 * 5000.0))
    assert eps_above > 0.999, eps_above


def test_gradient_matches_finite_difference():
    gamma = torch.tensor(3.5, dtype=DT, requires_grad=True)
    n_t = torch.tensor(250.0, dtype=DT, requires_grad=True)
    B_t = torch.tensor(300.0, dtype=DT, requires_grad=True)
    eps = fbl_error(gamma, n_t, B_t)
    eps.backward()
    g_gamma, g_n, g_B = float(gamma.grad), float(n_t.grad), float(B_t.grad)
    h = 1e-6
    for val, grad, name in ((3.5, g_gamma, "gamma"), (250.0, g_n, "n"), (300.0, g_B, "B")):
        def f(x):
            args = {"gamma": torch.tensor(3.5, dtype=DT), "n": 250.0, "B": 300.0}
            args[name] = torch.tensor(x, dtype=DT) if name == "gamma" else x
            return float(fbl_error(args["gamma"], args["n"], args["B"]))
        fd = (f(val + h) - f(val - h)) / (2 * h)
        rel = abs(fd - grad) / (abs(fd) + 1e-9)
        assert rel < 1e-4, (name, fd, grad, rel)


def test_blocklength_accounting():
    # RE-level accounting: DMRS comb-2 leaves 50% data REs in its symbols (not whole-symbol).
    spec = BlocklengthSpec(num_rb=10, sc_per_rb=12, sym_per_slot=14,
                           agc_symbols=1, guard_symbols=1, pscch_sci_symbols=3, dmrs_symbols=4,
                           dmrs_comb_density=0.5)
    # fully-data symbols = 14 - 1 - 1 - 3 - 4 = 5
    assert spec.data_symbols() == 5.0
    # n = 5*120 (full data) + 4*120*0.5 (DMRS-symbol survivors) + 0 (full-band PSCCH) = 840
    assert spec.channel_uses() == 840.0
    bd = spec.breakdown()
    assert bd["dmrs_surviving_data_re"] == 240.0
    assert bd["channel_uses"] == 840.0
    # sub-band PSCCH leaves data in the rest of the band
    spec_sub = BlocklengthSpec(num_rb=10, pscch_sci_symbols=2, pscch_prbs=4, dmrs_symbols=4)
    # data symbols = 14-1-1-2-4 = 6 -> 720; dmrs survivors 240; pscch survivors 2*(10-4)*12=144
    assert spec_sub.channel_uses() == 720.0 + 240.0 + 144.0
    # with reserved REs (default pscch=2 full band: 6*120 + 240 = 960)
    spec2 = BlocklengthSpec(num_rb=10, reserved_re=50.0)
    assert spec2.channel_uses() == 960.0 - 50.0
    # over-deduction raises
    try:
        BlocklengthSpec(num_rb=10, dmrs_symbols=20).data_symbols()
    except ValueError:
        pass
    else:
        raise AssertionError("over-deduction should raise")
    # B accounting
    assert payload_bits(40, crc_bits=24, header_bits=16) == 40 * 8 + 24 + 16


def test_fading_average_adaptive_large_n():
    # Regression for the under-resolved-quadrature blocker: at large blocklength the FBL
    # transition vs |H|^2 is sharp; the adaptive default num_quad must keep error < 1e-3.
    gm = torch.tensor(30.0, dtype=DT)
    n, B = 25920.0, 0.4 * 25920.0 * math.log2(1 + 30.0)
    analytic = float(fading_average_success(gm, n, B))  # adaptive default
    gen = torch.Generator().manual_seed(31)
    h2 = -torch.log(torch.rand(4000000, generator=gen, dtype=DT))
    mc = float(fbl_link_success(gm * h2, n, B).mean())
    assert abs(analytic - mc) < 1e-3, (analytic, mc)


def test_shadow_average_sharp_transition():
    # Regression: with fading='none' the log-normal shadow integrand is a sharp step at
    # large n; the adaptive probit+Gauss-Legendre shadow quadrature must stay accurate
    # (the prior fixed 9-node Gauss-Hermite gave ~0.13 error here).
    from src.mainline.finite_blocklength import averaged_link_success
    gm = torch.tensor(30.0, dtype=DT)
    n, B, std = 25920.0, 0.4 * 25920.0 * math.log2(31.0), 4.0
    for fad in ("none", "rayleigh"):
        ana = float(averaged_link_success(gm, n, B, shadow_std_db=std, fading=fad, max_harq_attempts=1))
        gen = torch.Generator().manual_seed(5)
        z = torch.randn(4000000, generator=gen, dtype=DT) * std
        shadow = torch.pow(torch.tensor(10.0, dtype=DT), z / 10.0)
        if fad == "rayleigh":
            h2 = -torch.log(torch.rand(4000000, generator=gen, dtype=DT))
            geff = gm * shadow * h2
        else:
            geff = gm * shadow
        mc = float(fbl_link_success(geff, n, B).mean())
        assert abs(ana - mc) < 1e-3, (fad, ana, mc)
    # shadow std==0 reduces to the unfaded value exactly
    no_shadow = float(averaged_link_success(gm, 200.0, 250.0, shadow_std_db=0.0, fading="none", max_harq_attempts=1))
    ref = float(fbl_link_success(gm, 200.0, 250.0))
    assert abs(no_shadow - ref) < 1e-12


def test_compute_link_reliability_chain():
    from src.mainline.finite_blocklength import HeadlineLinkConfig
    cfg = HeadlineLinkConfig()
    d = torch.tensor([20.0, 60.0, 150.0], dtype=DT)
    los = torch.ones(3, dtype=DT)
    ell = cfg.compute_link_reliability(
        d, los, interference_dbm=-95.0, response_bits=300.0, concurrent_tx=torch.tensor([3.0, 3.0, 3.0], dtype=DT),
    )
    assert torch.all(ell >= 0) and torch.all(ell <= 1)
    # closer links are more reliable
    assert float(ell[0]) >= float(ell[1]) >= float(ell[2]) - 1e-9
    # differentiable end-to-end w.r.t. distance
    d2 = d.clone().requires_grad_(True)
    ell2 = cfg.compute_link_reliability(d2, los, interference_dbm=-95.0, response_bits=300.0)
    ell2.sum().backward()
    assert torch.isfinite(d2.grad).all()
    # shadow fields are LIVE: turning shadow off changes the result
    cfg_ns = HeadlineLinkConfig(use_shadow_fading=False)
    ell_ns = cfg_ns.compute_link_reliability(d, los, interference_dbm=-95.0, response_bits=300.0)
    assert abs(float(ell[0]) - float(ell_ns[0])) > 1e-9 or abs(float(ell[1]) - float(ell_ns[1])) > 1e-9


def test_harq_monotone():
    gamma = torch.tensor(2.0, dtype=DT)
    n, B = 200.0, 250.0
    eps = [float(harq_residual_error(gamma, n, B, m, combining="chase")) for m in [1, 2, 3, 4]]
    assert all(eps[i + 1] <= eps[i] + 1e-12 for i in range(len(eps) - 1))
    assert eps[3] < eps[0]
    eps_ir = [float(harq_residual_error(gamma, n, B, m, combining="ir")) for m in [1, 2, 3]]
    assert all(eps_ir[i + 1] <= eps_ir[i] + 1e-12 for i in range(len(eps_ir) - 1))


def test_poll_composition():
    pc = torch.tensor([0.1, 0.0, 0.3], dtype=DT)
    phd = torch.tensor([0.05, 0.2, 0.0], dtype=DT)
    er = torch.tensor([0.2, 0.1, 0.4], dtype=DT)
    es = torch.tensor([0.3, 0.05, 0.1], dtype=DT)
    out = poll_success(p_collision=pc, p_half_duplex=phd, eps_request=er, eps_response=es)
    ref = (1 - pc) * (1 - phd) * (1 - er) * (1 - es)
    assert torch.allclose(out, ref, atol=1e-15)
    assert torch.all(out >= 0) and torch.all(out <= 1)
    # each channel matters: increasing any one factor lowers success
    base = float(poll_success(p_collision=torch.tensor(0.1, dtype=DT), p_half_duplex=torch.tensor(0.1, dtype=DT),
                              eps_request=torch.tensor(0.1, dtype=DT), eps_response=torch.tensor(0.1, dtype=DT)))
    worse = float(poll_success(p_collision=torch.tensor(0.5, dtype=DT), p_half_duplex=torch.tensor(0.1, dtype=DT),
                               eps_request=torch.tensor(0.1, dtype=DT), eps_response=torch.tensor(0.1, dtype=DT)))
    assert worse < base


def test_mode2_collision_formula():
    S = 5.0
    for N in [1, 2, 5, 10]:
        out = float(mode2_collision_probability(torch.tensor(float(N), dtype=DT), S))
        ref = 1.0 - (1.0 - 1.0 / S) ** max(N - 1, 0)
        assert abs(out - ref) < 1e-12, (N, out, ref)


def test_fading_average_matches_montecarlo():
    gamma_mean = torch.tensor(5.0, dtype=DT)
    n, B = 200.0, 250.0
    analytic = float(fading_average_success(gamma_mean, n, B, fading="rayleigh", num_quad=96))
    # Monte-Carlo over |H|^2 ~ Exp(1)
    gen = torch.Generator().manual_seed(7)
    h2 = -torch.log(torch.rand(400000, generator=gen, dtype=DT))  # Exp(1)
    mc = float(fbl_link_success(gamma_mean * h2, n, B).mean())
    assert abs(analytic - mc) < 5e-3, (analytic, mc)


def test_headline_config_grounded():
    cfg = HeadlineLinkConfig()
    cfg.assert_headline_grounded()  # default must pass
    bad = HeadlineLinkConfig(idealized_channel=True)
    try:
        bad.assert_headline_grounded()
    except AssertionError:
        pass
    else:
        raise AssertionError("idealized channel must be rejected for headline")
    bad2 = HeadlineLinkConfig(use_finite_blocklength=False)
    try:
        bad2.assert_headline_grounded()
    except AssertionError:
        pass
    else:
        raise AssertionError("non-FBL must be rejected for headline")


def test_sinr_decreases_with_distance():
    d = torch.tensor([10.0, 50.0, 100.0, 300.0], dtype=DT)
    los = torch.ones(4, dtype=DT)
    gamma = sinr_linear_from_geometry(
        d, los, tx_power_dbm=23.0, noise_dbm=-95.0, interference_dbm=-95.0, fc_ghz=5.9,
    )
    assert torch.all(gamma[1:] - gamma[:-1] <= 0)  # SINR falls with distance
    assert torch.all(gamma > 0)
    # nearby LOS link should have a healthy SINR (> 0 dB)
    assert float(gamma[0]) > 1.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G5 tests passed.")
