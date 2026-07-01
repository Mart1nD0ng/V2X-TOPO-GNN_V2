"""G-NDH-CSI-AGING -- stale CSI edge features (spec §5).

Acceptance (engineering plan §5 + spec §5.1-5.2):
  * physics uses CURRENT CSI; the MODEL sees only STALE CSI -> no train/eval physics change;
  * csi_age_ms=0 (+ zero noise) recovers the current-CSI feature EXACTLY;
  * larger csi_age_ms increases the feature error AND the reported csi_uncertainty (monotone);
  * csi_uncertainty is age-driven (AR(1) shadow decorrelation + fixed measurement noise);
  * deterministic from the generator; no truth/future leak (features are geometry + age + noise only);
  * stale_vs_distance_residual and stale_delivery are well-formed deployable proxies.
"""

import math

import torch

from src.environment import RoundPhysicsConfig, build_candidate_graph
from src.environment.csi_aging import stale_csi_edge_features, csi_uncertainty_db
from src.environment.round_physics import edge_geometry
from src.environment.urban_scene import build_manhattan_scene

GEN = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)


def _scene(seed=0):
    return build_manhattan_scene(5, 5, 3, generator=GEN(seed))


def _true_snr_db(scene):
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    geom = edge_geometry(gc, PHY)
    return 10.0 * torch.log10(geom.rx_power_mw / PHY.noise_mw)


# --------------------------------------------------------------- age=0 recovers current CSI
def test_age_zero_zero_noise_recovers_current_csi():
    sc = _scene()
    feats = stale_csi_edge_features(sc, PHY, csi_age_ms=0.0, csi_noise_std_db=0.0,
                                    shadow_ar_std_db=4.0, generator=GEN(1))
    assert torch.allclose(feats["stale_sinr_db"], _true_snr_db(sc), atol=1e-9)
    assert float(feats["csi_uncertainty"].max()) < 1e-9      # no age, no noise -> zero uncertainty
    assert torch.allclose(feats["csi_age_ms"], torch.zeros_like(feats["csi_age_ms"]))


def test_uncertainty_monotone_in_age():
    ages = [0.0, 50.0, 100.0, 200.0, 500.0]
    uncs = [csi_uncertainty_db(a, csi_noise_std_db=1.0, shadow_ar_std_db=4.0,
                               shadow_decorrelation_s=3.0) for a in ages]
    assert all(uncs[i] <= uncs[i + 1] + 1e-12 for i in range(len(uncs) - 1))   # non-decreasing in age
    assert uncs[0] == 1.0                                     # age 0 -> only measurement noise
    assert uncs[-1] > uncs[0]                                 # deep staleness -> larger uncertainty
    # asymptote: age >> decorrelation -> sqrt(noise^2 + shadow^2)
    assert uncs[-1] <= math.sqrt(1.0 ** 2 + 4.0 ** 2) + 1e-9


def test_larger_age_increases_feature_error():
    sc = _scene()
    true = _true_snr_db(sc)
    err_small = (stale_csi_edge_features(sc, PHY, csi_age_ms=50.0, csi_noise_std_db=1.0,
                 shadow_ar_std_db=4.0, generator=GEN(3))["stale_sinr_db"] - true).abs().mean()
    err_large = (stale_csi_edge_features(sc, PHY, csi_age_ms=500.0, csi_noise_std_db=1.0,
                 shadow_ar_std_db=4.0, generator=GEN(3))["stale_sinr_db"] - true).abs().mean()
    assert float(err_large) > float(err_small)               # staler CSI is noisier vs the true link


def test_reproducible_and_shapes():
    sc = _scene()
    a = stale_csi_edge_features(sc, PHY, csi_age_ms=100.0, generator=GEN(5))
    b = stale_csi_edge_features(sc, PHY, csi_age_ms=100.0, generator=GEN(5))
    gc = build_candidate_graph(sc.positions, sc.comm_radius)
    for k in ("stale_sinr_db", "stale_delivery", "csi_age_ms", "csi_uncertainty",
              "stale_vs_distance_residual"):
        assert a[k].shape == (gc.num_edges,)
        assert torch.equal(a[k], b[k])                        # deterministic from seed
    assert float(a["stale_delivery"].min()) >= 0.0 and float(a["stale_delivery"].max()) <= 1.0


def test_no_truth_or_future_leak_in_source():
    """The CSI-aging module must not import evidence/protocol/validation/MC (no truth/future)."""
    import inspect

    import src.environment.csi_aging as m
    src = inspect.getsource(m)
    for forbidden in ("evidence_model", "overlapping_evidence", "run_dynamic_mc", "validation",
                      "EvidenceModel", "Y_star", "y_star", "basin_", "correct"):
        assert forbidden not in src, f"csi_aging leaks forbidden dependency/token {forbidden!r}"


def test_physics_uses_current_csi_not_stale():
    """Sanity: CSI aging is a FEATURE-only mechanism -- it produces model features and does NOT alter
    the physics link. edge_geometry (the physics link source) is independent of the stale features."""
    sc = _scene()
    g0 = edge_geometry(build_candidate_graph(sc.positions, sc.comm_radius), PHY).rx_power_mw.clone()
    _ = stale_csi_edge_features(sc, PHY, csi_age_ms=500.0, csi_noise_std_db=4.0, generator=GEN(7))
    g1 = edge_geometry(build_candidate_graph(sc.positions, sc.comm_radius), PHY).rx_power_mw
    assert torch.equal(g0, g1)                                # physics link untouched by CSI aging
