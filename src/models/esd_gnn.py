"""ESD-GNN -- Effective-Sampling-Dynamics multi-graph encoder + CDQ heads (spec §9.3-§9.6).

A scene/scale-agnostic GNN that maps the OBSERVABLE structure of a V2X scene to a CDQ k-DPP
query kernel ``(quality q_ij > 0, diversity b_ij in R^r)`` per candidate edge. It encodes the
two physical graphs and the region supergraph, with the per-layer aggregations of spec §9.3:

* source-side candidate-competition aggregation (over a source's out-edges in ``G_comm``);
* destination-side incoming-load aggregation (over a node's in-edges in ``G_comm``);
* interference-neighbourhood aggregation (over ``G_int``) -- the resource/contention channel;
* vehicle<->region pooling (mean over a region, broadcast back) -- the region supergraph /
  evidence-correlation channel that lets a finite-depth model see weak cuts and region-level
  evidence imbalance (a plain two-layer candidate-graph mean is insufficient, spec §9.3);
* residual + LayerNorm.

After ``n_enc`` encoder layers (and ``n_refine`` dynamics-in-the-loop refinements that feed the
current kernel's inclusion-derived receiver load ``Lambda`` back as a node feature, spec §9.6),
edge heads emit ``q`` (softplus + floor, > 0) and ``b`` (R^r). All features are STRUCTURAL and
observable (log-degrees, region size, distance, LOS, same-region) -- they do NOT depend on the
evidence-bias realisation, ``Y*`` or any peer vote (constraint #10), and carry no scene-specific
ids, so one model transfers across scenes and scales. Everything is ``O(E)`` sparse scatter (no
``N x N``); differentiable end to end. Train and deploy use the SAME forward (constraint #3).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.environment.candidate_graph import RadiusGraph, build_candidate_graph
from src.environment.interference_graph import build_interference_graph
from src.environment.round_physics import _los_probability

__all__ = ["ESDGNNConfig", "ESDGNN", "ESDGNNQueryPolicy", "build_scene_features"]


@dataclass(frozen=True)
class ESDGNNConfig:
    hidden_dim: int = 32
    r: int = 4                 # diversity embedding rank (must satisfy r >= k)
    n_enc: int = 3             # multi-graph encoder layers
    n_refine: int = 2          # dynamics-in-the-loop refinements (load feedback, spec §9.6)
    quality_floor: float = 0.05
    k: int = 3                 # query subset size (for the refinement inclusion feedback)


@dataclass(frozen=True)
class SceneFeatures:
    node_feat: torch.Tensor        # [N, Fn] observable structural node features
    edge_feat: torch.Tensor        # [E, Fe] observable edge features (G_comm)
    gc: RadiusGraph
    gi: RadiusGraph
    region_of: torch.Tensor        # [N]
    num_regions: int


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    H = values.shape[-1]
    out = values.new_zeros((n, H))
    out = out.index_add(0, index, values)
    cnt = values.new_zeros((n, 1)).index_add(0, index, torch.ones_like(values[:, :1]))
    return out / cnt.clamp_min(1.0)


def build_scene_features(scene, cfg: ESDGNNConfig, *, comm_radius_norm: float | None = None) -> SceneFeatures:
    """Observable structural features for a scene (no truth/vote/ids; transferable)."""
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    gi = build_interference_graph(scene.positions, scene.int_radius)
    N = scene.num_nodes
    dtype = scene.positions.dtype
    out_deg = torch.bincount(gc.src_index, minlength=N).to(dtype)
    in_deg = torch.bincount(gc.dst_index, minlength=N).to(dtype)
    int_deg = torch.bincount(gi.src_index, minlength=N).to(dtype)
    region_of = scene.region_of
    G = scene.num_regions
    region_size = torch.bincount(region_of, minlength=G).to(dtype)[region_of]
    node_feat = torch.stack([
        torch.log1p(out_deg), torch.log1p(in_deg), torch.log1p(int_deg),
        torch.log1p(region_size),
    ], dim=1)                                                  # [N, 4]
    cr = comm_radius_norm if comm_radius_norm is not None else scene.comm_radius
    los = _los_probability(gc.distance, 50.0)
    same_region = (region_of[gc.src_index] == region_of[gc.dst_index]).to(dtype)
    edge_feat = torch.stack([gc.distance / cr, los, same_region], dim=1)   # [E, 3]
    return SceneFeatures(node_feat=node_feat, edge_feat=edge_feat, gc=gc, gi=gi,
                         region_of=region_of, num_regions=G)


class _MultiGraphLayer(nn.Module):
    """One spec §9.3 layer: comm source/dest + interference + region pooling, residual + norm."""

    def __init__(self, h: int, edge_dim: int):
        super().__init__()
        self.msg_src = nn.Linear(h + edge_dim, h)      # candidate -> source (competition)
        self.msg_dst = nn.Linear(h + edge_dim, h)      # poller -> destination (incoming load)
        self.update = nn.Linear(4 * h, h)              # [self, src-agg, dst-agg, region] (+int folded)
        self.msg_int = nn.Linear(h, h)                 # interference neighbour -> node
        self.norm = nn.LayerNorm(h)

    def forward(self, h, ef, gc, gi, region_of, num_regions):
        s, d = gc.src_index, gc.dst_index
        m_src = _scatter_mean(F.relu(self.msg_src(torch.cat([h[d], ef], dim=-1))), s, h.shape[0])
        m_dst = _scatter_mean(F.relu(self.msg_dst(torch.cat([h[s], ef], dim=-1))), d, h.shape[0])
        m_int = _scatter_mean(F.relu(self.msg_int(h[gi.src_index])), gi.dst_index, h.shape[0]) \
            if gi.num_edges > 0 else torch.zeros_like(h)
        reg = _scatter_mean(h, region_of, num_regions)[region_of]            # region pool + broadcast
        upd = self.update(torch.cat([h, m_src + m_int, m_dst, reg], dim=-1))
        return self.norm(h + F.relu(upd))


class ESDGNN(nn.Module):
    """Multi-graph encoder + dynamics refinement + CDQ (quality, diversity) heads."""

    def __init__(self, cfg: ESDGNNConfig, node_dim: int = 4, edge_dim: int = 3):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim
        self.embed = nn.Linear(node_dim, h)
        self.enc = nn.ModuleList([_MultiGraphLayer(h, edge_dim) for _ in range(cfg.n_enc)])
        # refinement layers consume an extra scalar node feature (inclusion-derived load Lambda)
        self.refine = nn.ModuleList([_MultiGraphLayer(h, edge_dim) for _ in range(cfg.n_refine)])
        self.load_proj = nn.Linear(1, h)
        self.q_head = nn.Linear(2 * h + edge_dim, 1)
        self.b_head = nn.Linear(2 * h + edge_dim, cfg.r)

    def _edge_readout(self, h, ef, gc):
        z = torch.cat([h[gc.src_index], h[gc.dst_index], ef], dim=-1)
        quality = F.softplus(self.q_head(z).squeeze(-1)) + self.cfg.quality_floor      # [E] > 0
        diversity = self.b_head(z)                                                      # [E, r]
        return quality, diversity

    def forward(self, feats: SceneFeatures) -> tuple[torch.Tensor, torch.Tensor]:
        gc, gi, ef = feats.gc, feats.gi, feats.edge_feat
        h = F.relu(self.embed(feats.node_feat))
        for layer in self.enc:
            h = layer(h, ef, gc, gi, feats.region_of, feats.num_regions)
        quality, diversity = self._edge_readout(h, ef, gc)
        # ---- dynamics-in-the-loop refinement (spec §9.6): feed the current kernel's inclusion
        #      -derived receiver load Lambda back as a node feature, then re-encode ----
        for layer in self.refine:
            lam = self._receiver_load(feats, quality, diversity)                        # [N]
            h = h + self.load_proj(torch.log1p(lam).unsqueeze(-1))
            h = layer(h, ef, gc, gi, feats.region_of, feats.num_regions)
            quality, diversity = self._edge_readout(h, ef, gc)
        return quality, diversity

    def _receiver_load(self, feats: SceneFeatures, quality, diversity) -> torch.Tensor:
        """Inclusion-derived receiver load Lambda_j = sum_{i->j} pi_ij (analytic feedback, §9.6)."""
        from src.sampling.cdq_query import cdq_edge_inclusion
        gc = feats.gc
        N = feats.node_feat.shape[0]
        pi = cdq_edge_inclusion(gc.src_index, gc.dst_index, N, quality, diversity, self.cfg.k)
        return pi.new_zeros(N).index_add(0, gc.dst_index, pi)            # [N]


class ESDGNNQueryPolicy:
    """Wrap an :class:`ESDGNN` + a scene as a CDQ query policy for the canonical episode."""

    query_law = "cdq"
    name = "esd_gnn"

    def __init__(self, model: ESDGNN, scene, *, features: SceneFeatures | None = None):
        self.model = model
        self.scene = scene
        self.features = features if features is not None else build_scene_features(scene, model.cfg)

    def kernel(self, graph) -> tuple[torch.Tensor, torch.Tensor]:
        # the episode passes its own G_comm; it matches the cached features' gc (same positions)
        return self.model(self.features)
