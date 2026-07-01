"""Strong deployable heuristics on the NDH proxy schema + best-heuristic envelope (G-NDH-BASELINE-ENVELOPE).

A GNN "win" is credible only against strong, capability-matched heuristics that read the SAME observable
proxies the GNN reads (spec §11, constraint #13/#19). Each heuristic here is a diagonal ESP query policy
whose per-edge ``log_weights`` are a FIXED linear combination of ``build_scene_features_v2`` columns — all
DEPLOYABLE proxies (noisy capacity ``log μ̂``, stale CSI, SPS same-bucket conflict, RSU role, local
density), never truth (no ``Y*``/vote/MC/true ``μ``/current CSI). The coefficients are hand-set design
choices in NORMALISED feature units (they are NOT tuned on any result — constraint: no post-hoc tuning).

``best_heuristic_envelope`` runs every heuristic through the canonical dynamic-MC judge and reports the
per-scene winner among the heuristics whose wrong-basin risk is MATCHED to distance (reliability is a hard
constraint; the envelope is the strongest reliability-feasible heuristic, the bar a GNN must clear).
"""

from __future__ import annotations

import torch

from src.environment.candidate_graph import build_candidate_graph
from src.models.scene_features_v2 import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES, build_scene_features_v2
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.validation import run_dynamic_mc

__all__ = ["NDH_HEURISTICS", "make_ndh_baseline", "best_heuristic_envelope"]

# Each heuristic: (distance_beta or None, {edge_col: coef}, {node_col_at_dst: coef}). Coefficients are in
# normalised feature units; distance_beta multiplies the RAW-metre distance (matching DistanceQueryPolicy).
_HEURISTIC_SPEC = {
    "distance":              (0.04, {}, {}),                                   # special-cased to DistanceQueryPolicy
    "stale_link_quality":    (None, {"stale_sinr_db_norm": 4.0}, {}),
    "capacity_aware":        (None, {"receiver_capacity_proxy": 2.0}, {}),
    "resource_aware":        (None, {"resource_conflict_count": -2.0, "same_resource_bucket": -2.0}, {}),
    "distance_plus_capacity": (0.04, {"receiver_capacity_proxy": 2.0}, {}),
    "distance_plus_resource": (0.04, {"resource_conflict_count": -2.0}, {}),
    "distance_plus_csi_age":  (0.04, {"stale_link_delivery": 2.0}, {}),        # age is scene-level -> use stale delivery
    "load_balanced":         (None, {"predicted_receiver_queue_ratio": -2.0}, {}),
    "rsu_nearest":           (0.04, {"edge_to_rsu": 3.0}, {}),
    "rsu_capacity_aware":    (None, {"edge_to_rsu": 2.0, "receiver_capacity_proxy": 2.0}, {}),
    "local_density_aware":   (None, {}, {"local_density": -3.0}),
}
NDH_HEURISTICS = list(_HEURISTIC_SPEC)


class _ProxyLinearPolicy:
    """Diagonal ESP policy: ``log_weights = -beta*distance + Σ c_e·edge_col + Σ c_n·node_col[dst]`` over the
    deployable ``SceneFeaturesV2`` columns. Edge order matches the episode's ``G_comm`` (same positions)."""

    def __init__(self, feats, name: str, *, distance_beta=None, edge_w=None, node_dst_w=None):
        gc = feats.gc
        logits = torch.zeros(gc.num_edges, dtype=feats.edge_feat.dtype)
        if distance_beta is not None:
            logits = logits - float(distance_beta) * gc.distance.to(logits.dtype)
        for col, c in (edge_w or {}).items():
            logits = logits + float(c) * feats.edge_feat[:, EDGE_FEATURE_NAMES.index(col)]
        for col, c in (node_dst_w or {}).items():
            logits = logits + float(c) * feats.node_feat[gc.dst_index, NODE_FEATURE_NAMES.index(col)]
        self.logits = logits
        self._E = int(gc.num_edges)
        self.name = name

    def log_weights(self, graph) -> torch.Tensor:
        if graph.num_edges != self._E:
            raise ValueError("edge count mismatch between heuristic proxy logits and the episode graph")
        return self.logits


def make_ndh_baseline(kind: str, scene, phy_cfg, *, distance_beta: float = 0.04,
                      generator: torch.Generator | None = None, **feature_kwargs):
    """Return a deployable NDH heuristic ESP policy bound to ``scene``. Reads only observable proxies."""
    if kind not in _HEURISTIC_SPEC:
        raise ValueError(f"unknown NDH heuristic {kind!r}; expected one of {NDH_HEURISTICS}")
    if kind == "distance":
        return DistanceQueryPolicy(beta_per_m=distance_beta)
    beta, edge_w, node_w = _HEURISTIC_SPEC[kind]
    if beta is not None:
        beta = distance_beta                                    # honour the caller's distance beta for combos
    feats = build_scene_features_v2(scene, phy_cfg, generator=generator, **feature_kwargs)
    return _ProxyLinearPolicy(feats, kind, distance_beta=beta, edge_w=edge_w, node_dst_w=node_w)


def _mc(scene, ev, policy, profile, proto, phy, omega, trials, gen_seed):
    r = run_dynamic_mc(scene, ev, policy, proto, phy, num_trials=trials,
                       generator=torch.Generator().manual_seed(gen_seed), service_profile=profile,
                       participation=omega)
    return {"Pc": r.basin_P_correct, "Fw": r.basin_F_wrong, "Fs": r.basin_F_split, "Fd": r.basin_F_deadline}


def best_heuristic_envelope(scene, ev, profile, proto, phy, *, trials: int = 1500,
                            generator: torch.Generator | None = None, base_seed: int = 0,
                            distance_beta: float = 0.04, delta_w: float = 0.005, delta_s: float = 0.001,
                            **feature_kwargs) -> dict:
    """Evaluate every heuristic through the canonical dynamic-MC judge and return the per-scene winner
    among the reliability-feasible ones (F_wrong matched to distance within ``delta_w``). CRN: all
    heuristics share the same MC generator seed (common evidence draw). Returns per-heuristic macro +
    the admitted set + the winner (max P_correct among admitted).
    """
    from src.metrics.participation import vehicle_only_participation

    omega = (vehicle_only_participation(scene) if getattr(scene, "node_type", None) is not None
             else None)
    gen_seed = base_seed + 12345
    per = {}
    for kind in NDH_HEURISTICS:
        pol = make_ndh_baseline(kind, scene, phy, distance_beta=distance_beta,
                                generator=torch.Generator().manual_seed(base_seed), **feature_kwargs)
        per[kind] = _mc(scene, ev, pol, profile, proto, phy, omega, trials, gen_seed)
    dist = per["distance"]
    # reliability-feasible = wrong/split matched to distance (reliability is a hard constraint)
    admitted = [h for h in NDH_HEURISTICS
                if per[h]["Fw"] <= dist["Fw"] + delta_w and per[h]["Fs"] <= dist["Fs"] + delta_s]
    if "distance" not in admitted:
        admitted.append("distance")                             # distance is the reference, always admitted
    winner = max(admitted, key=lambda h: per[h]["Pc"])
    return {"per_heuristic": per, "admitted": admitted, "winner": winner,
            "winner_metrics": per[winner], "delta_w": delta_w, "delta_s": delta_s}
