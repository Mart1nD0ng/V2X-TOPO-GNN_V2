"""Recurrent (temporal) V2X edge scorer.

`TemporalGNNScorer` adds per-node temporal memory on top of the existing
`HierarchicalGNNScorer` encoder: each frame is encoded to per-node embeddings, a per-node
recurrent cell carries the slow consensus-reliability state across frames, and an edge head
scores candidate edges from the temporal node states. Node identity must be stable across
frames (fixed population) for the per-node recurrence to be valid.

Temporal cells (S2, docs/MODEL_ARCHITECTURE_DESIGN.md):
  * ``temporal_cell="gru"`` (default) — the legacy per-node GRUCell. The state update is
    TOPOLOGY-BLIND (``new_state = cell(node_embedding, state)``): the recurrence sees only the
    node's own encoded embedding, not its neighbours' carried state.
  * ``temporal_cell="graph_gru"`` (S2/B1) — a DCGRU-style graph-coupled cell whose gates also
    see a graph-diffused (neighbour-mean) version of the carried state, so the slow reliability
    state evolves THROUGH the constructed topology, mirroring how consensus reliability spreads.

Historical note (corrected, see docs/S2_B1_ABLATION_RESULTS.md): the per-frame ceiling probe
(docs/TEMPORAL_MODEL_DESIGN.md) concluded temporal could not beat the memoryless model under the
broken mean-field evaluator. An interim "4-arm stream" experiment under the corrected quenched
evaluator suggested a ~12% temporal win, but that result has NO reproducible artifact (no script /
json / figure survives) and was SUPERSEDED by the capacity-matched S2/B1 ablation (result/s2_ablation*),
which found recurrence NEUTRAL-to-NEGATIVE on F at 10/20 dB and REVERSED-but-noisy at 30 dB. The
question "does temporal memory help reliability" is therefore OPEN, not settled — consistent with the
environment being a fully-observable, deterministic-transition, per-frame-objective process where
memory has no theoretical edge (the realism gaps that would create hidden temporal state are P1-1).
graph_gru / carried-state remain opt-in and OFF by default for exactly this reason. The per-frame
hard-forward topology construction and the avalanche evaluator remain unchanged and are applied per
frame by the caller.
"""

from __future__ import annotations

import math

import torch

from .hierarchical_gnn import HierarchicalGNNScorer, _as_index_tensor, _make_mlp


def _graph_diffuse(
    state: torch.Tensor, src_index: torch.Tensor, dst_index: torch.Tensor, num_nodes: int
) -> torch.Tensor:
    """Parameter-free mean diffusion of the carried per-node state over the current topology:
    each receiver (dst) averages the state of its in-neighbours (src). This is the structural
    coupling the graph-coupled GRU consumes so the slow reliability state evolves THROUGH the
    constructed topology (S2/B1, docs/MODEL_ARCHITECTURE_DESIGN.md)."""
    if src_index.numel() == 0:
        return torch.zeros_like(state)
    aggregated = state.new_zeros((num_nodes, state.shape[1]))
    aggregated.index_add_(0, dst_index, state.index_select(0, src_index))
    counts = state.new_zeros((num_nodes,))
    counts.index_add_(0, dst_index, torch.ones_like(dst_index, dtype=state.dtype))
    return aggregated / counts.clamp_min(1.0).unsqueeze(1)


class GraphCoupledGRUCell(torch.nn.Module):
    """DCGRU-style graph-coupled GRU cell (Li et al., DCRNN, ICLR 2018; S2/B1).

    A standard GRU's gates see only the node's own carried state ``h``. Here every gate also sees a
    GRAPH-DIFFUSED version ``h_graph`` (the mean of the node's in-neighbours' states over the current
    topology), so the slow consensus-reliability state is updated by neighbour diffusion — matching how
    consensus reliability actually spreads over the constructed graph. ``h_graph`` is parameter-free; the
    extra learning is in the wider (3*hidden) gate transforms.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.update_gate = torch.nn.Linear(3 * hidden_dim, hidden_dim)
        self.reset_gate = torch.nn.Linear(3 * hidden_dim, hidden_dim)
        self.candidate = torch.nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, state: torch.Tensor, state_graph: torch.Tensor) -> torch.Tensor:
        joined = torch.cat([x, state, state_graph], dim=1)
        z = torch.sigmoid(self.update_gate(joined))
        r = torch.sigmoid(self.reset_gate(joined))
        candidate_in = torch.cat([x, r * state, r * state_graph], dim=1)
        n = torch.tanh(self.candidate(candidate_in))
        return (1.0 - z) * n + z * state


class TemporalGNNScorer(torch.nn.Module):
    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        *,
        hidden_dim: int = 64,
        message_layers: int = 2,
        mlp_layers: int = 2,
        init_mode: str = "xavier",
        use_region_context: bool = True,
        learnable_score_gain: bool = True,
        score_output_gain: float = 10.0,
        score_standardization: bool = True,
        attention_heads: int = 0,
        attention_negative_slope: float = 0.2,
        gcnii_alpha: float = 0.0,
        gcnii_lambda: float = 1.0,
        jk_mode: str = "last",
        channel_recalibration: str = "none",
        se_reduction: int = 4,
        temporal_cell: str = "gru",
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if str(temporal_cell) not in {"gru", "graph_gru"}:
            raise ValueError("temporal_cell must be 'gru' or 'graph_gru'")
        self.hidden_dim = int(hidden_dim)
        self.temporal_cell = str(temporal_cell)
        self.learnable_score_gain = bool(learnable_score_gain)
        self.score_standardization = bool(score_standardization)
        self.score_output_gain = float(score_output_gain)
        # Per-frame node/edge encoder (structural-bias and auxiliary heads off; only
        # encode_nodes is used). Its Linear layers are initialised in its __init__.
        self.encoder = HierarchicalGNNScorer(
            node_feature_dim,
            edge_feature_dim,
            hidden_dim=hidden_dim,
            message_layers=message_layers,
            mlp_layers=mlp_layers,
            init_mode=init_mode,
            use_region_context=use_region_context,
            use_structural_score_bias=False,
            enable_budget_head=False,
            enable_region_bridge_head=False,
            enable_sector_head=False,
            enable_role_head=False,
            attention_heads=attention_heads,
            attention_negative_slope=attention_negative_slope,
            gcnii_alpha=gcnii_alpha,
            gcnii_lambda=gcnii_lambda,
            jk_mode=jk_mode,
            channel_recalibration=channel_recalibration,
            se_reduction=se_reduction,
        )
        # GRU cell and edge head use PyTorch's default (reasonable-scale, seedable)
        # init — NOT the deterministic linspace, which would make the recurrent path's
        # initial gradient vanish (the same dynamic-range failure mode as the legacy
        # scorer). Reproducibility is provided by seeding the RNG before construction.
        # Default "gru" = the legacy per-node GRUCell (topology-blind). Opt-in "graph_gru" = the
        # DCGRU-style graph-coupled cell (S2/B1); created only when selected so the "gru" path is
        # byte-identical. Both use PyTorch default (seedable) init, NOT the deterministic ramp.
        if self.temporal_cell == "graph_gru":
            self.cell = None
            self.graph_cell: GraphCoupledGRUCell | None = GraphCoupledGRUCell(hidden_dim)
        else:
            self.cell = torch.nn.GRUCell(hidden_dim, hidden_dim)
            self.graph_cell = None
        self.edge_head = _make_mlp(hidden_dim * 3, hidden_dim, 1, mlp_layers)
        if self.learnable_score_gain:
            self.score_log_gain = torch.nn.Parameter(torch.tensor(math.log(float(score_output_gain)), dtype=torch.float32))
        else:
            self.register_parameter("score_log_gain", None)

    def init_state(self, num_nodes: int, *, dtype: torch.dtype | None = None, device: torch.device | None = None) -> torch.Tensor:
        ref = next(self.parameters())
        return torch.zeros(
            int(num_nodes), self.hidden_dim,
            dtype=dtype if dtype is not None else ref.dtype,
            device=device if device is not None else ref.device,
        )

    def forward(
        self,
        *,
        num_nodes: int,
        src_index: torch.Tensor,
        dst_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        state: torch.Tensor,
        region_id: torch.Tensor | None = None,
        num_regions: int | None = None,
        zero_graph_state: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score edges for one frame and advance the per-node temporal state.

        Returns ``(edge_score [E], new_state [N, hidden_dim])``.

        ``zero_graph_state`` (graph_gru only) forces the diffused neighbour-state input to zero — a
        PARAMETER-IDENTICAL "plain recurrence" control that ablates the B1 graph coupling while keeping
        the exact same weights, for the capacity-matched S2 ablation.
        """
        node_embedding, edge_embedding = self.encoder.encode_nodes(
            num_nodes=num_nodes, src_index=src_index, dst_index=dst_index,
            node_features=node_features, edge_features=edge_features,
            region_id=region_id, num_regions=num_regions,
        )
        if state.shape != (num_nodes, self.hidden_dim):
            raise ValueError(f"state must have shape ({num_nodes}, {self.hidden_dim}); got {tuple(state.shape)}")
        state = state.to(dtype=node_embedding.dtype, device=node_embedding.device)
        device = node_embedding.device
        src = _as_index_tensor("src_index", src_index, device=device)
        dst = _as_index_tensor("dst_index", dst_index, device=device)
        # Advance the per-node temporal state. "graph_gru" diffuses the state over the current
        # topology so it updates through neighbours (S2/B1); "gru" is the legacy topology-blind cell.
        if self.temporal_cell == "graph_gru":
            state_graph = (
                torch.zeros_like(state) if zero_graph_state
                else _graph_diffuse(state, src, dst, num_nodes)
            )
            new_state = self.graph_cell(node_embedding, state, state_graph)
        else:
            new_state = self.cell(node_embedding, state)
        if src.numel() == 0:
            edge_score = edge_embedding.new_zeros((0,))
            return edge_score, new_state
        score_input = torch.cat([new_state[src], new_state[dst], edge_embedding], dim=1)
        pre = self.edge_head(score_input).squeeze(-1)
        if self.score_standardization and pre.numel() > 1:
            pre = (pre - pre.mean()) / torch.clamp(pre.std(unbiased=False), min=pre.new_tensor(1.0e-6))
        if self.learnable_score_gain and self.score_log_gain is not None:
            gain = torch.exp(self.score_log_gain).to(dtype=pre.dtype)
        else:
            gain = pre.new_tensor(self.score_output_gain)
        return pre * gain, new_state
