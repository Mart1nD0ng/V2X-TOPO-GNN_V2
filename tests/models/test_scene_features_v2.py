"""G-NDH-FEATURE-SCHEMA -- SceneFeaturesV2 + ESDGNNStaticV2 (spec §7).

LEAK-CRITICAL acceptance (engineering plan §7 + Contract C2):
  * NO truth/future leak: the model reads only DEPLOYABLE proxies -- the NOISY capacity proxy
    log(mu_hat), never the true mu_j; stale (not current) CSI; no Y*/vote/MC outcome;
  * the true scene.node_capacity is NOT equal to / recoverable from any feature column;
  * base structural features reproduced (first 4 node / 3 edge cols == build_scene_features);
  * feature-availability mask correct: a mechanism that is OFF -> its columns are 0 AND masked 0;
  * O(E) (edge_feat has exactly gc.num_edges rows); shared builder (baselines can call it);
  * ESDGNNStaticV2 forward runs on every NDH regime and returns a positive per-edge quality.
"""

import inspect

import torch

from src.environment import RoundPhysicsConfig
from src.environment.nonuniform_urban_scene import build_nonuniform_urban_scene
from src.environment.receiver_capacity import noisy_capacity_proxy
from src.environment.urban_scene import build_manhattan_scene
from src.models.esd_gnn import build_scene_features, ESDGNNConfig
from src.models.esd_gnn_v2 import ESDGNNStaticV2
from src.models.scene_features_v2 import (
    EDGE_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    build_scene_features_v2,
)

GEN = lambda s: torch.Generator().manual_seed(s)  # noqa: E731
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)


def _full_scene(seed=7):
    """A scene with ALL four mechanisms on."""
    return build_nonuniform_urban_scene(
        5, 5, 3, enable_rsu=True, p_intersection_rsu=0.3, enable_hotspots=True,
        enable_sps=True, sps_n_buckets=100, enable_heterogeneous_capacity=True, generator=GEN(seed))


# --------------------------------------------------------------- LEAK: no true mu / no truth
def test_no_true_capacity_leak():
    sc = _full_scene()
    feats = build_scene_features_v2(sc, PHY, capacity_proxy_noise=0.2, generator=GEN(1))
    true_mu = sc.node_capacity
    ci = NODE_FEATURE_NAMES.index("capacity_proxy_log")
    cap_col = feats.node_feat[:, ci]
    # the capacity feature is log of a NOISY proxy -> NOT equal to log(true mu)
    assert not torch.allclose(cap_col, torch.log(true_mu), atol=1e-6)
    # no node feature column equals the raw true mu (truth is not a feature)
    for j in range(feats.node_feat.shape[1]):
        assert not torch.allclose(feats.node_feat[:, j], true_mu, atol=1e-9)
    # the feature must match log() of an INDEPENDENT noisy draw's distribution, not the truth:
    # recomputing the proxy with the SAME generator seed reproduces the feature (deterministic proxy)
    mu_hat = noisy_capacity_proxy(true_mu, 0.2, generator=GEN(1))
    # (feature builder uses its own generator; we only assert the feature is a NOISED mu, i.e. differs
    #  from truth in the same multiplicative-noise family -> positive correlation, not identity)
    assert float(torch.corrcoef(torch.stack([cap_col, torch.log(true_mu)]))[0, 1]) > 0.5


def test_no_true_capacity_leak_edge_columns():
    """The EDGE capacity columns must also use the NOISY proxy, never true mu[dst]."""
    sc = _full_scene()
    f1 = build_scene_features_v2(sc, PHY, capacity_proxy_noise=0.2, generator=GEN(1))
    f2 = build_scene_features_v2(sc, PHY, capacity_proxy_noise=0.2, generator=GEN(2))
    gc = f1.gc
    true_mu = sc.node_capacity
    rc = EDGE_FEATURE_NAMES.index("receiver_capacity_proxy")
    qr = EDGE_FEATURE_NAMES.index("predicted_receiver_queue_ratio")
    # receiver_capacity_proxy != log(true mu[dst]); queue ratio != in_deg/true_mu[dst]
    assert not torch.allclose(f1.edge_feat[:, rc], torch.log(true_mu[gc.dst_index].clamp_min(1e-6)), atol=1e-6)
    # noise perturbs BOTH edge capacity columns (they are noisy estimates, not truth)
    assert not torch.allclose(f1.edge_feat[:, rc], f2.edge_feat[:, rc])
    assert not torch.allclose(f1.edge_feat[:, qr], f2.edge_feat[:, qr])


def test_csi_features_are_stale_not_current():
    """CSI columns must be the STALE/aged estimate; age=0+zero-noise -> geometry-mean SINR (deployable,
    same info class as distance), larger noise/age -> larger deviation (never the current channel)."""
    sc = _full_scene()
    from src.environment.round_physics import edge_geometry
    gc = build_scene_features_v2(sc, PHY, generator=GEN(1)).gc
    geom_sinr_db = 10.0 * torch.log10(edge_geometry(gc, PHY).rx_power_mw / PHY.noise_mw) / 30.0
    si = EDGE_FEATURE_NAMES.index("stale_sinr_db_norm")
    clean = build_scene_features_v2(sc, PHY, csi_age_ms=0.0, csi_noise_std_db=0.0, shadow_ar_std_db=4.0,
                                    generator=GEN(1))
    assert torch.allclose(clean.edge_feat[:, si], geom_sinr_db, atol=1e-9)   # age0+noise0 = geometry mean
    noisy = build_scene_features_v2(sc, PHY, csi_age_ms=500.0, csi_noise_std_db=4.0, generator=GEN(1))
    assert (noisy.edge_feat[:, si] - geom_sinr_db).abs().mean() > (clean.edge_feat[:, si] - geom_sinr_db).abs().mean()


def test_availability_gating_asymmetry():
    """A NonuniformUrbanScene with RSU/hotspots OFF still HAS node_type/hotspot_score fields (all
    vehicle / all zero) -> those columns must still be masked OFF (gate on enable-flag, not presence)."""
    sc = build_nonuniform_urban_scene(5, 5, 3, enable_rsu=False, enable_hotspots=False,
                                      enable_sps=True, sps_n_buckets=100, generator=GEN(3))
    f = build_scene_features_v2(sc, PHY, generator=GEN(3))
    for name in ("node_type_rsu", "node_type_vehicle", "hotspot_score", "intersection_distance"):
        j = NODE_FEATURE_NAMES.index(name)
        assert float(f.node_mask[j]) == 0.0                  # RSU/hotspot off -> masked off despite field present
    # SPS is on -> its column is masked ON
    assert float(f.node_mask[NODE_FEATURE_NAMES.index("same_resource_conflict_degree")]) == 1.0


def test_source_imports_no_truth_modules():
    import src.models.scene_features_v2 as m
    src = inspect.getsource(m)
    for forbidden in ("evidence_model", "overlapping_evidence", "run_dynamic_mc", "EvidenceModel",
                      "basin_", "y_star", "Y_star", ".correct", "vote"):
        assert forbidden not in src, f"scene_features_v2 leaks forbidden token {forbidden!r}"


def test_true_capacity_only_in_physics_not_features():
    """scene.node_capacity feeds physics; the feature builder must read the NOISY proxy of it."""
    sc = _full_scene()
    f1 = build_scene_features_v2(sc, PHY, capacity_proxy_noise=0.2, generator=GEN(1))
    f2 = build_scene_features_v2(sc, PHY, capacity_proxy_noise=0.2, generator=GEN(2))
    ci = NODE_FEATURE_NAMES.index("capacity_proxy_log")
    # different proxy noise draws -> different capacity feature (it is a noisy estimate, not the truth)
    assert not torch.allclose(f1.node_feat[:, ci], f2.node_feat[:, ci])


# --------------------------------------------------------------- base features reproduced
def test_base_features_reproduced_and_ndh_masked_off():
    sc = build_manhattan_scene(5, 5, 3, generator=GEN(0))   # plain scene: no NDH fields
    cfg = ESDGNNConfig(use_cdq=False)
    base = build_scene_features(sc, cfg)
    v2 = build_scene_features_v2(sc, PHY, generator=GEN(0))
    # first 4 node / first 3 edge columns are the base structural features (same values)
    assert torch.allclose(v2.node_feat[:, :4], base.node_feat, atol=1e-9)
    assert torch.allclose(v2.edge_feat[:, :3], base.edge_feat, atol=1e-9)
    # NDH mechanism columns are masked OFF (mask 0) and zeroed for a plain scene
    for name in ("node_type_rsu", "capacity_proxy_log", "same_resource_conflict_degree", "hotspot_score"):
        j = NODE_FEATURE_NAMES.index(name)
        assert float(v2.node_mask[j]) == 0.0
        assert float(v2.node_feat[:, j].abs().max()) == 0.0


def test_mask_flips_on_when_mechanism_enabled():
    off = build_scene_features_v2(build_nonuniform_urban_scene(5, 5, 3, generator=GEN(3)), PHY, generator=GEN(3))
    on = build_scene_features_v2(_full_scene(3), PHY, generator=GEN(3))
    for name in ("capacity_proxy_log", "same_resource_conflict_degree", "node_type_rsu", "hotspot_score"):
        j = NODE_FEATURE_NAMES.index(name)
        assert float(off.node_mask[j]) == 0.0 and float(on.node_mask[j]) == 1.0
    # capacity edge features
    for name in ("receiver_capacity_proxy", "same_resource_bucket", "edge_to_rsu"):
        j = EDGE_FEATURE_NAMES.index(name)
        assert float(on.edge_mask[j]) == 1.0


# --------------------------------------------------------------- shape / O(E) / determinism
def test_shapes_and_determinism():
    sc = _full_scene()
    a = build_scene_features_v2(sc, PHY, generator=GEN(5))
    b = build_scene_features_v2(sc, PHY, generator=GEN(5))
    N, E = sc.num_nodes, a.gc.num_edges
    assert a.node_feat.shape == (N, len(NODE_FEATURE_NAMES))
    assert a.edge_feat.shape == (E, len(EDGE_FEATURE_NAMES))
    assert a.node_mask.shape == (len(NODE_FEATURE_NAMES),)
    assert a.edge_mask.shape == (len(EDGE_FEATURE_NAMES),)
    assert torch.equal(a.node_feat, b.node_feat) and torch.equal(a.edge_feat, b.edge_feat)
    assert torch.isfinite(a.node_feat).all() and torch.isfinite(a.edge_feat).all()


# --------------------------------------------------------------- model forward on all regimes
def test_esdgnn_static_v2_forward_all_regimes():
    cfg = ESDGNNConfig(hidden_dim=16, use_cdq=False)
    model = ESDGNNStaticV2(cfg).double()
    scenes = [
        build_manhattan_scene(5, 5, 3, generator=GEN(0)),
        build_nonuniform_urban_scene(5, 5, 3, enable_sps=True, sps_n_buckets=100, generator=GEN(1)),
        build_nonuniform_urban_scene(5, 5, 3, enable_rsu=True, enable_heterogeneous_capacity=True, generator=GEN(2)),
        _full_scene(3),
    ]
    for sc in scenes:
        feats = build_scene_features_v2(sc, PHY, generator=GEN(9))
        quality, diversity = model(feats)
        assert quality.shape == (feats.gc.num_edges,)
        assert float(quality.min().detach()) > 0             # quality floor keeps it positive
        assert torch.isfinite(quality).all()


def test_v2_model_differentiable():
    cfg = ESDGNNConfig(hidden_dim=16, use_cdq=False)
    model = ESDGNNStaticV2(cfg).double()
    feats = build_scene_features_v2(_full_scene(3), PHY, generator=GEN(9))
    quality, _ = model(feats)
    quality.sum().backward()
    g = next(model.parameters()).grad
    assert g is not None and float(g.abs().sum()) > 0
