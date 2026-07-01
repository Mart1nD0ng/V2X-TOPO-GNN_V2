"""SceneFeaturesV2 -- expanded DEPLOYABLE feature schema for the NDH benchmark (spec §7.2).

Wires every NDH mechanism's OBSERVABLE proxy into a single node/edge feature tensor for
``ESDGNNStaticV2`` (and the strong heuristics), with a per-feature AVAILABILITY MASK: a mechanism
that is off for a scene has its feature columns zeroed and masked. The base structural columns
(log-degrees, region size, distance/comm_radius, LOS, same-region) are preserved as the leading
columns so ``build_scene_features`` is a strict prefix (old behaviour reproduced when all mechanisms
are off).

LEAK CONTRACT (Contract C2, non-negotiable): every feature here is a DEPLOYABLE proxy —
  * capacity: ``capacity_proxy_log = log(noisy_capacity_proxy(mu_j))`` — the NOISY estimate μ̂, NEVER
    the true ``mu_j`` (which is simulator-side truth used only by the physics queue);
  * CSI: STALE SINR/delivery + age + uncertainty — never the current channel;
  * SPS: same-bucket conflict degree / same-bucket indicator — observable resource structure;
  * geometry/role: degrees, density, node type, hotspot score, intersection distance.
No truth label, no peer preference readout, no future resource/CSI, no MC outcome enters (spec
§3.5/§4.5/§5.2). This module imports ONLY geometry/mechanism proxy helpers — never the evidence
model, the dynamic MC, or any truth source.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.environment.candidate_graph import build_candidate_graph
from src.environment.csi_aging import stale_csi_edge_features
from src.environment.interference_graph import build_interference_graph
from src.environment.receiver_capacity import noisy_capacity_proxy
from src.environment.round_physics import _los_probability
from src.environment.sps_resource import same_resource_conflict_degree, sensed_channel_busy_ratio

__all__ = ["SceneFeaturesV2", "build_scene_features_v2", "NODE_FEATURE_NAMES", "EDGE_FEATURE_NAMES"]

# Ordered feature schema. Leading base columns match build_scene_features exactly. Each mechanism
# block is available iff the scene exposes its field (node_type / node_capacity / resource_bucket /
# hotspot_score); CSI is geometry-derivable and available whenever enable_csi (default True).
#
# Phase-1 OMISSIONS (spec §7.2 lists them; they need multi-frame HISTORY / EMA state that a single
# static episode does not carry -> deferred to the Phase-2 Temporal-ESDGNN branch, NOT Phase-1):
#   resource_age_norm, resource_busy_ratio (SPS reselection age/occupancy history — static buckets
#   have no age trajectory), ack_success_ema, queue_delay_ema (per-node ACK/queue exponential moving
#   averages across rounds). The static Phase-1 proxies (same_resource_conflict_degree, sensed_cbr,
#   predicted_receiver_queue_ratio) stand in for these where an instantaneous surrogate exists.
#
# The AVAILABILITY MASK (node_mask/edge_mask) is per-column metadata meaning "mechanism ENABLED for
# this regime" (not "column is non-degenerate"). It is consumed by the heuristic/provenance layer,
# NOT by the ESDGNN encoder: off-mechanism columns are already zeroed, and the Phase-1 contract fixes
# the mechanism set between train and eval (train==eval), so the encoder need not distinguish
# masked-off zero from measured zero. The strong heuristics (G-NDH-BASELINE-ENVELOPE) consume the
# SAME builder (named columns) so they see exactly the GNN's observable proxies.
NODE_FEATURE_NAMES = [
    "log_out_deg", "log_in_deg", "log_int_deg", "log_region_size",   # base structural (always)
    "local_density", "sensed_cbr",                                    # geometry proxies (always)
    "node_type_vehicle", "node_type_rsu",                             # RSU/role block
    "capacity_proxy_log", "capacity_uncertainty",                     # capacity block (NOISY proxy)
    "same_resource_conflict_degree",                                  # SPS block
    "hotspot_score", "intersection_distance",                        # hotspot block
]
EDGE_FEATURE_NAMES = [
    "distance_norm", "los", "same_region",                            # base structural (always)
    "stale_sinr_db_norm", "stale_link_delivery", "csi_age_norm",      # CSI block
    "csi_uncertainty_norm", "stale_vs_distance_residual_norm",        # CSI block
    "same_resource_bucket", "resource_conflict_count",                # SPS block
    "edge_to_rsu",                                                    # RSU/role block
    "receiver_capacity_proxy", "predicted_receiver_queue_ratio",     # capacity block (NOISY proxy)
    "edge_crosses_hotspot",                                          # hotspot block
]


@dataclass(frozen=True, eq=False)
class SceneFeaturesV2:
    node_feat: torch.Tensor        # [N, Fn] expanded observable node features
    edge_feat: torch.Tensor        # [E, Fe] expanded observable edge features (G_comm)
    gc: object                     # RadiusGraph G_comm
    gi: object                     # RadiusGraph G_int
    region_of: torch.Tensor        # [N]
    num_regions: int
    node_mask: torch.Tensor        # [Fn] per-feature availability (1 = mechanism on, 0 = off)
    edge_mask: torch.Tensor        # [Fe] per-feature availability


def _saturating(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Scene-INVARIANT bounded density normalizer ``1 - exp(-x/scale)`` in [0,1) with a FIXED scale
    (transferable across scenes/scales; a per-scene max normalizer would break cross-scene comparability)."""
    return 1.0 - torch.exp(-x / scale)


def build_scene_features_v2(
    scene, phy_cfg, *,
    capacity_proxy_noise: float = 0.2,
    enable_csi: bool = True,
    csi_age_ms: float = 100.0,
    csi_noise_std_db: float = 1.0,
    shadow_ar_std_db: float = 4.0,
    shadow_decorrelation_s: float = 3.0,
    generator: torch.Generator | None = None,
) -> SceneFeaturesV2:
    """Build the expanded deployable feature tensors + availability mask. Deterministic given
    ``generator`` (used only for the NOISY capacity + stale-CSI observation draws)."""
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    gi = build_interference_graph(scene.positions, scene.int_radius)
    N, E = scene.num_nodes, gc.num_edges
    dtype = scene.positions.dtype
    region_of = scene.region_of
    G = scene.num_regions
    if generator is None:
        generator = torch.Generator().manual_seed(0)

    # ---- availability: mechanism ENABLED, not mere field presence. A NonuniformUrbanScene always
    #      carries node_type / hotspot_score (all-vehicle / all-zero when the mechanism is off), so
    #      gate role/hotspot on the scene's enable-flags; capacity/SPS fields are None unless enabled.
    node_type = getattr(scene, "node_type", None)
    node_capacity = getattr(scene, "node_capacity", None)
    resource_bucket = getattr(scene, "resource_bucket", None)
    hotspot_score = getattr(scene, "hotspot_score", None)
    intersection_xy = getattr(scene, "intersection_xy", None)
    params = getattr(scene, "params", None) or {}
    avail = {
        "rsu": bool(params.get("enable_rsu", False)) and node_type is not None,
        "capacity": node_capacity is not None,                       # None unless enabled
        "sps": resource_bucket is not None,                          # None unless enabled
        "hotspot": (bool(params.get("enable_hotspots", False))
                    and hotspot_score is not None and intersection_xy is not None),
        "csi": bool(enable_csi),
    }

    # ---- base + geometry node features (always available) ----
    out_deg = torch.bincount(gc.src_index, minlength=N).to(dtype)
    in_deg = torch.bincount(gc.dst_index, minlength=N).to(dtype)
    int_deg = torch.bincount(gi.src_index, minlength=N).to(dtype)
    region_size = torch.bincount(region_of, minlength=G).to(dtype)[region_of]
    # scene-invariant (fixed-scale) bounded density proxies -> transferable across scenes/scales
    local_density = _saturating(out_deg, 8.0)                          # comm-neighbour density in [0,1)
    sensed_cbr = _saturating(sensed_channel_busy_ratio(scene.positions, scene.int_radius, dtype=dtype), 15.0)

    ncols: dict[str, torch.Tensor] = {
        "log_out_deg": torch.log1p(out_deg), "log_in_deg": torch.log1p(in_deg),
        "log_int_deg": torch.log1p(int_deg), "log_region_size": torch.log1p(region_size),
        "local_density": local_density, "sensed_cbr": sensed_cbr,
    }
    nmask = {k: 1.0 for k in ncols}                                     # base/geometry always on

    z = torch.zeros(N, dtype=dtype)
    # role block
    ncols["node_type_vehicle"] = (node_type == 0).to(dtype) if avail["rsu"] else z.clone()
    ncols["node_type_rsu"] = (node_type == 1).to(dtype) if avail["rsu"] else z.clone()
    # capacity block: NOISY proxy only (never true mu)
    if avail["capacity"]:
        mu_hat = noisy_capacity_proxy(node_capacity, capacity_proxy_noise, generator=generator)
        cap_log = torch.log(mu_hat.clamp_min(1e-6))
        ncols["capacity_proxy_log"] = cap_log
        ncols["capacity_uncertainty"] = torch.full((N,), float(capacity_proxy_noise), dtype=dtype)
    else:
        mu_hat = None
        ncols["capacity_proxy_log"] = z.clone()
        ncols["capacity_uncertainty"] = z.clone()
    # SPS block
    ncols["same_resource_conflict_degree"] = (
        torch.log1p(same_resource_conflict_degree(resource_bucket, scene.positions, scene.int_radius, dtype=dtype))
        if avail["sps"] else z.clone())
    # hotspot block
    ncols["hotspot_score"] = hotspot_score.to(dtype) if avail["hotspot"] else z.clone()
    if avail["hotspot"]:
        # nearest intersection to a node on segment s IS one of s's two endpoints -> O(N), no [N, I] cdist
        ep = scene.segment_endpoints[region_of].to(dtype)               # [N, 2, 2] (start_xy, end_xy)
        d0 = (scene.positions.to(dtype) - ep[:, 0]).norm(dim=1)
        d1 = (scene.positions.to(dtype) - ep[:, 1]).norm(dim=1)
        ncols["intersection_distance"] = torch.minimum(d0, d1) / float(scene.block_m)
    else:
        ncols["intersection_distance"] = z.clone()
    for name, block in (("node_type_vehicle", "rsu"), ("node_type_rsu", "rsu"),
                        ("capacity_proxy_log", "capacity"), ("capacity_uncertainty", "capacity"),
                        ("same_resource_conflict_degree", "sps"),
                        ("hotspot_score", "hotspot"), ("intersection_distance", "hotspot")):
        nmask[name] = 1.0 if avail[block] else 0.0

    node_feat = torch.stack([ncols[n] for n in NODE_FEATURE_NAMES], dim=1)      # [N, Fn]
    node_mask = torch.tensor([nmask[n] for n in NODE_FEATURE_NAMES], dtype=dtype)

    # ---- base edge features (always) ----
    los = _los_probability(gc.distance, 50.0)
    same_region = (region_of[gc.src_index] == region_of[gc.dst_index]).to(dtype)
    ze = torch.zeros(E, dtype=dtype)
    ecols: dict[str, torch.Tensor] = {
        "distance_norm": gc.distance / scene.comm_radius, "los": los, "same_region": same_region,
    }
    emask = {"distance_norm": 1.0, "los": 1.0, "same_region": 1.0}

    # CSI block (stale; deployable)
    if avail["csi"]:
        cf = stale_csi_edge_features(scene, phy_cfg, csi_age_ms=csi_age_ms, csi_noise_std_db=csi_noise_std_db,
                                     shadow_ar_std_db=shadow_ar_std_db, shadow_decorrelation_s=shadow_decorrelation_s,
                                     generator=generator, dtype=dtype)
        ecols["stale_sinr_db_norm"] = cf["stale_sinr_db"] / 30.0
        ecols["stale_link_delivery"] = cf["stale_delivery"]
        ecols["csi_age_norm"] = cf["csi_age_ms"] / 500.0
        ecols["csi_uncertainty_norm"] = cf["csi_uncertainty"] / 10.0
        ecols["stale_vs_distance_residual_norm"] = cf["stale_vs_distance_residual"] / 30.0
    else:
        for n in ("stale_sinr_db_norm", "stale_link_delivery", "csi_age_norm",
                  "csi_uncertainty_norm", "stale_vs_distance_residual_norm"):
            ecols[n] = ze.clone()

    # SPS edge block
    if avail["sps"]:
        ecols["same_resource_bucket"] = (resource_bucket[gc.src_index] == resource_bucket[gc.dst_index]).to(dtype)
        conf = same_resource_conflict_degree(resource_bucket, scene.positions, scene.int_radius, dtype=dtype)
        ecols["resource_conflict_count"] = torch.log1p(conf[gc.dst_index])
    else:
        ecols["same_resource_bucket"] = ze.clone()
        ecols["resource_conflict_count"] = ze.clone()

    # role block (edge -> RSU)
    ecols["edge_to_rsu"] = (node_type[gc.dst_index] == 1).to(dtype) if avail["rsu"] else ze.clone()

    # capacity edge block: NOISY proxy of the receiver + a predicted queue ratio from observables
    if avail["capacity"]:
        ecols["receiver_capacity_proxy"] = torch.log(mu_hat[gc.dst_index].clamp_min(1e-6))
        pred_rho = in_deg[gc.dst_index] / mu_hat[gc.dst_index].clamp_min(1e-6)
        ecols["predicted_receiver_queue_ratio"] = pred_rho.clamp(0.0, 10.0)
    else:
        ecols["receiver_capacity_proxy"] = ze.clone()
        ecols["predicted_receiver_queue_ratio"] = ze.clone()

    # hotspot edge block
    if avail["hotspot"]:
        hs = hotspot_score.to(dtype)
        ecols["edge_crosses_hotspot"] = torch.maximum(hs[gc.src_index], hs[gc.dst_index])
    else:
        ecols["edge_crosses_hotspot"] = ze.clone()

    for name, block in (("stale_sinr_db_norm", "csi"), ("stale_link_delivery", "csi"),
                        ("csi_age_norm", "csi"), ("csi_uncertainty_norm", "csi"),
                        ("stale_vs_distance_residual_norm", "csi"),
                        ("same_resource_bucket", "sps"), ("resource_conflict_count", "sps"),
                        ("edge_to_rsu", "rsu"), ("receiver_capacity_proxy", "capacity"),
                        ("predicted_receiver_queue_ratio", "capacity"), ("edge_crosses_hotspot", "hotspot")):
        emask[name] = 1.0 if avail[block] else 0.0

    edge_feat = torch.stack([ecols[n] for n in EDGE_FEATURE_NAMES], dim=1)      # [E, Fe]
    edge_mask = torch.tensor([emask[n] for n in EDGE_FEATURE_NAMES], dtype=dtype)

    return SceneFeaturesV2(node_feat=node_feat, edge_feat=edge_feat, gc=gc, gi=gi,
                           region_of=region_of, num_regions=G, node_mask=node_mask, edge_mask=edge_mask)
