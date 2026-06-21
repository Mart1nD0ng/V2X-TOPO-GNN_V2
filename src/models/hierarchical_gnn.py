"""Sparse hierarchical GNN scorer skeleton for V2X candidate edges.

M6 intentionally stops at candidate edge scoring. It does not construct dense
node-pair tensors, sample edges, or run any training loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int = 2) -> torch.nn.Sequential:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    modules: list[torch.nn.Module] = []
    if layers == 1:
        modules.append(torch.nn.Linear(input_dim, output_dim))
    else:
        modules.append(torch.nn.Linear(input_dim, hidden_dim))
        modules.append(torch.nn.ReLU())
        for _ in range(layers - 2):
            modules.append(torch.nn.Linear(hidden_dim, hidden_dim))
            modules.append(torch.nn.ReLU())
        modules.append(torch.nn.Linear(hidden_dim, output_dim))
    return torch.nn.Sequential(*modules)


def _initialize_module(module: torch.nn.Module, init_mode: str) -> None:
    if init_mode not in {"xavier", "kaiming", "deterministic"}:
        raise ValueError("init_mode must be one of: xavier, kaiming, deterministic")
    for submodule in module.modules():
        if isinstance(submodule, torch.nn.Linear):
            with torch.no_grad():
                if init_mode == "deterministic" and submodule.weight.numel() > 0:
                    values = torch.linspace(
                        -0.1,
                        0.1,
                        steps=submodule.weight.numel(),
                        dtype=submodule.weight.dtype,
                        device=submodule.weight.device,
                    ).reshape_as(submodule.weight)
                    submodule.weight.copy_(values)
                elif init_mode == "xavier":
                    torch.nn.init.xavier_uniform_(submodule.weight)
                elif init_mode == "kaiming":
                    torch.nn.init.kaiming_uniform_(submodule.weight, nonlinearity="relu")
                if submodule.bias is not None:
                    submodule.bias.zero_()


def _as_float_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating point tensor")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return value


def _as_index_tensor(name: str, value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor")
    if torch.is_floating_point(value) or torch.is_complex(value) or value.dtype == torch.bool:
        raise TypeError(f"{name} must use an integer dtype")
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1-D tensor")
    if value.device != device:
        value = value.to(device)
    return value.to(dtype=torch.long)


def _segment_softmax(logits: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Numerically-stable softmax over rows sharing the same ``index`` (per-segment).

    ``logits`` is ``[E, K]`` (K attention heads), ``index`` is ``[E]`` with values in
    ``[0, num_segments)``. Returns ``[E, K]`` weights that sum to 1 within each segment.
    Used by the GATv2 attention aggregation to normalise attention over each receiver's
    incoming edges (A1, see docs/MODEL_ARCHITECTURE_DESIGN.md).
    """
    if logits.numel() == 0:
        return logits
    heads = logits.shape[1]
    expanded_index = index.unsqueeze(1).expand(-1, heads)
    seg_max = logits.new_full((num_segments, heads), float("-inf"))
    seg_max.scatter_reduce_(0, expanded_index, logits.detach(), reduce="amax", include_self=True)
    shifted = logits - seg_max.index_select(0, index)
    exp_logits = torch.exp(shifted)
    seg_sum = logits.new_zeros((num_segments, heads)).index_add(0, index, exp_logits)
    denom = seg_sum.index_select(0, index).clamp_min(torch.finfo(logits.dtype).tiny)
    return exp_logits / denom


def apply_dropedge(
    src_index: torch.Tensor,
    dst_index: torch.Tensor,
    edge_features: torch.Tensor,
    drop_prob: float,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DropEdge (Rong et al., ICLR 2020): randomly drop a fraction of candidate edges.

    Train-time augmentation / over-smoothing regulariser for the message-passing input.
    Returns kept ``(src_index, dst_index, edge_features, keep_mask)``. ``drop_prob<=0`` is a
    no-op (byte-identical), so it is safe to leave wired with the default off. Never drops
    every edge. This is a graph-level op (NOT in the model) so the model stays deployment
    faithful / deterministic (A4, see docs/MODEL_ARCHITECTURE_DESIGN.md).
    """
    num_edges = int(src_index.numel())
    if drop_prob <= 0.0 or num_edges == 0:
        keep = torch.ones(num_edges, dtype=torch.bool, device=src_index.device)
        return src_index, dst_index, edge_features, keep
    if not 0.0 < drop_prob < 1.0:
        raise ValueError("drop_prob must be in [0, 1)")
    probs = torch.rand(num_edges, generator=generator, device=src_index.device, dtype=torch.float64)
    keep = probs >= drop_prob
    if not bool(keep.any()):
        keep[int(torch.argmax(probs))] = True
    return src_index[keep], dst_index[keep], edge_features[keep], keep


class ECAChannelRecalibration(torch.nn.Module):
    """ECA-Net (Wang et al., CVPR 2020) channel recalibration for node embeddings (A5).

    Squeeze: mean over nodes -> per-channel descriptor. Excite: a parameter-light 1-D
    convolution over channels (local cross-channel interaction) -> sigmoid gate. Scale:
    multiply each node's channels by the gate. No SE bottleneck FC (better for small H).
    """

    def __init__(self, hidden_dim: int, kernel_size: int | None = None) -> None:
        super().__init__()
        if kernel_size is None:
            approx = int(abs((math.log2(max(hidden_dim, 2)) / 2.0) + 0.5))
            kernel_size = approx if approx % 2 == 1 else approx + 1
            kernel_size = max(kernel_size, 3)
        if kernel_size % 2 == 0:
            raise ValueError("ECA kernel_size must be odd")
        self.conv = torch.nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        if node_embedding.shape[0] == 0:
            return node_embedding
        descriptor = node_embedding.mean(dim=0).reshape(1, 1, -1)  # [1, 1, H]
        gate = torch.sigmoid(self.conv(descriptor)).reshape(1, -1)  # [1, H]
        return node_embedding * gate


class SEChannelRecalibration(torch.nn.Module):
    """Squeeze-and-Excitation (Hu et al., CVPR 2018) channel recalibration for node embeddings (A5)."""

    def __init__(self, hidden_dim: int, reduction: int = 4) -> None:
        super().__init__()
        bottleneck = max(hidden_dim // max(int(reduction), 1), 1)
        self.fc1 = torch.nn.Linear(hidden_dim, bottleneck)
        self.fc2 = torch.nn.Linear(bottleneck, hidden_dim)

    def forward(self, node_embedding: torch.Tensor) -> torch.Tensor:
        if node_embedding.shape[0] == 0:
            return node_embedding
        descriptor = node_embedding.mean(dim=0, keepdim=True)  # [1, H]
        gate = torch.sigmoid(self.fc2(torch.relu(self.fc1(descriptor))))  # [1, H]
        return node_embedding * gate


class FiLMConditioner(torch.nn.Module):
    """FiLM (Perez et al., AAAI 2018) regime conditioning (S3, docs/MODEL_ARCHITECTURE_DESIGN.md).

    Maps a per-graph REGIME vector z (e.g. [density, SINR operating point, load coupling, scenario one-hot])
    to per-channel affine parameters (gamma, beta) that scale-and-shift the node embeddings at each layer:
    ``FiLM(h) = gamma(z) * h + beta(z)``. This lets ONE planner SPECIALISE its behaviour per operating
    regime (sparse topology under heavy interference, denser under clean SINR) instead of averaging over
    regimes — the correct realisation of domain adaptation for measurable shift axes (condition, don't
    adversarially invert). ``zero_init_output`` makes it start at identity (gamma=1, beta=0) for stability.
    """

    def __init__(self, regime_dim: int, hidden_dim: int, num_layers: int, mlp_layers: int = 2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.generators = torch.nn.ModuleList(
            [_make_mlp(regime_dim, hidden_dim, 2 * hidden_dim, mlp_layers) for _ in range(int(num_layers))]
        )

    def zero_init_output(self) -> None:
        with torch.no_grad():
            for generator in self.generators:
                last = [m for m in generator if isinstance(m, torch.nn.Linear)][-1]
                last.weight.zero_()
                if last.bias is not None:
                    last.bias.zero_()

    def modulate(self, node_embedding: torch.Tensor, regime: torch.Tensor, layer_idx: int) -> torch.Tensor:
        out = self.generators[layer_idx](regime.reshape(-1))  # regime [regime_dim] -> [2*hidden]
        gamma = 1.0 + out[: self.hidden_dim]
        beta = out[self.hidden_dim:]
        return gamma * node_embedding + beta


@dataclass(frozen=True)
class HierarchicalGNNConfig:
    node_feature_dim: int
    edge_feature_dim: int
    hidden_dim: int = 32
    message_layers: int = 1
    mlp_layers: int = 2
    region_feature_dim: int = 0
    use_region_context: bool = True
    init_mode: str = "xavier"
    enable_budget_head: bool = True
    enable_region_bridge_head: bool = True
    enable_sector_head: bool = True
    enable_role_head: bool = True
    num_budget_bins: int = 5
    num_sectors: int = 8
    num_roles: int = 3
    use_structural_score_bias: bool = True
    bridge_bias_weight: float = 0.1
    sector_bias_weight: float = 0.1
    role_bias_weight: float = 0.1
    score_output_gain: float = 1.0
    # P2 remediation: dynamic-range fixes for the edge scorer.
    #   learnable_score_gain -> replace the fixed external score_scale band-aid
    #       with an internal learnable log-gain so the model can grow its own
    #       output spread relative to the top-k boundary margin.
    #   score_standardization -> standardize edge_score (zero-mean/unit-std
    #       across the candidate population) before the gain, so the score
    #       SPREAD relative to the top-k margin does not collapse as mean-pooled
    #       message passing over-smooths the node embeddings. This is an affine
    #       positive transform, so it preserves the top-k ranking (deployment
    #       faithful) and only re-scales the row-softmax sharpness / gradient.
    learnable_score_gain: bool = False
    score_standardization: bool = False
    # Axis A structural-encoder upgrades (docs/MODEL_ARCHITECTURE_DESIGN.md). All OFF by
    # default -> byte-identical to the legacy mean-pool MLP-MPNN.
    #   attention_heads      A1: >0 enables GATv2 dynamic-attention aggregation (0 = mean pool).
    #   gcnii_alpha          A2: >0 enables GCNII initial-residual + identity mapping (0 = legacy residual).
    #   gcnii_lambda         A2: beta_l = log(gcnii_lambda / l + 1) identity-mapping decay.
    #   jk_mode              A3: "last" (legacy) | "concat" | "max" jumping-knowledge fusion.
    #   channel_recalibration A5: "none" (legacy) | "eca" | "se" node-channel recalibration.
    attention_heads: int = 0
    attention_negative_slope: float = 0.2
    gcnii_alpha: float = 0.0
    gcnii_lambda: float = 1.0
    jk_mode: str = "last"
    channel_recalibration: str = "none"
    se_reduction: int = 4
    # S3 FiLM regime conditioning: regime_dim>0 enables per-layer FiLM modulation by a per-graph regime
    # vector (density/SINR/load/scenario). 0 = OFF (byte-identical legacy).
    regime_dim: int = 0


class SparseMessagePassingBlock(torch.nn.Module):
    """One deterministic sparse message passing block using index_add.

    Legacy path (attention off, gcnii off): mean-aggregate messages by receiver + an
    update MLP with a previous-layer residual. Opt-in A1 (GATv2 dynamic attention over each
    receiver's incoming edges) and A2 (GCNII initial-residual + identity mapping) replace the
    aggregation / residual respectively. Both off -> byte-identical to the legacy block.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_layers: int = 2,
        *,
        attention_heads: int = 0,
        attention_negative_slope: float = 0.2,
        gcnii_alpha: float = 0.0,
        gcnii_beta: float = 0.0,
    ) -> None:
        super().__init__()
        self.message_mlp = _make_mlp(hidden_dim * 3, hidden_dim, hidden_dim, mlp_layers)
        self.update_mlp = _make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, mlp_layers)
        self.attention_heads = int(attention_heads)
        self.attention_negative_slope = float(attention_negative_slope)
        self.gcnii_alpha = float(gcnii_alpha)
        self.gcnii_beta = float(gcnii_beta)
        if self.attention_heads > 0:
            # GATv2: linear transform W INSIDE the nonlinearity, attention vector a OUTSIDE
            # -> per-(src,dst,edge) dynamic attention (query-conditioned ranking).
            self.att_transform = torch.nn.Linear(hidden_dim * 3, hidden_dim)
            self.att_score = torch.nn.Linear(hidden_dim, self.attention_heads, bias=False)
        else:
            self.att_transform = None
            self.att_score = None

    def forward(
        self,
        node_embedding: torch.Tensor,
        edge_embedding: torch.Tensor,
        src_index: torch.Tensor,
        dst_index: torch.Tensor,
        *,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = node_embedding.shape[0]
        hidden_dim = node_embedding.shape[1]
        if src_index.numel() == 0:
            aggregated = node_embedding.new_zeros((num_nodes, hidden_dim))
        else:
            message_input = torch.cat(
                [node_embedding[src_index], node_embedding[dst_index], edge_embedding],
                dim=1,
            )
            messages = self.message_mlp(message_input)
            if self.attention_heads > 0:
                # A1 GATv2: a^T LeakyReLU(W [h_src ; h_dst ; e]) -> per-edge per-head logit,
                # softmaxed over each receiver's (dst) incoming edges, then weights the messages.
                logits = self.att_score(
                    torch.nn.functional.leaky_relu(
                        self.att_transform(message_input), self.attention_negative_slope
                    )
                )  # [E, heads]
                alpha = _segment_softmax(logits, dst_index, num_nodes)  # [E, heads]
                weighted = (alpha.unsqueeze(-1) * messages.unsqueeze(1)).mean(dim=1)  # [E, H]
                aggregated = node_embedding.new_zeros((num_nodes, hidden_dim))
                aggregated.index_add_(0, dst_index, weighted)
            else:
                aggregated = node_embedding.new_zeros((num_nodes, hidden_dim))
                aggregated.index_add_(0, dst_index, messages)
                counts = node_embedding.new_zeros((num_nodes,))
                counts.index_add_(0, dst_index, torch.ones_like(dst_index, dtype=node_embedding.dtype))
                aggregated = aggregated / counts.clamp_min(1.0).unsqueeze(1)
        if self.gcnii_alpha > 0.0 and h0 is not None:
            # A2 GCNII: propagate over the SELF-LOOP-augmented neighbourhood (GCNII uses \hat P
            # WITH self-loops, so each node keeps its own current representation), inject the
            # initial residual a*H0, then apply the identity-mapping (1-b)I + b*W. Keeps both the
            # node's own layer-l state and its encoder H0 alive through depth -> bounded, stable.
            propagated = 0.5 * (node_embedding + aggregated)
            smoothed = (1.0 - self.gcnii_alpha) * propagated + self.gcnii_alpha * h0
            transformed = self.update_mlp(torch.cat([node_embedding, smoothed], dim=1))
            return (1.0 - self.gcnii_beta) * smoothed + self.gcnii_beta * transformed, aggregated
        updated = self.update_mlp(torch.cat([node_embedding, aggregated], dim=1))
        return node_embedding + updated, aggregated


class HierarchicalGNNScorer(torch.nn.Module):
    """Sparse V2X candidate-edge scorer.

    The module applies identical scoring semantics regardless of train,
    validation, or deployment caller state. It contains no dropout and no graph
    construction branches; topology support selection remains the responsibility
    of :class:`TopologyConstructionLayer`.
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        *,
        hidden_dim: int = 32,
        message_layers: int = 1,
        mlp_layers: int = 2,
        region_feature_dim: int = 0,
        use_region_context: bool = True,
        init_mode: str = "xavier",
        enable_budget_head: bool = True,
        enable_region_bridge_head: bool = True,
        enable_sector_head: bool = True,
        enable_role_head: bool = True,
        num_budget_bins: int = 5,
        num_sectors: int = 8,
        num_roles: int = 3,
        use_structural_score_bias: bool = True,
        bridge_bias_weight: float = 0.1,
        sector_bias_weight: float = 0.1,
        role_bias_weight: float = 0.1,
        score_output_gain: float = 1.0,
        learnable_score_gain: bool = False,
        score_standardization: bool = False,
        attention_heads: int = 0,
        attention_negative_slope: float = 0.2,
        gcnii_alpha: float = 0.0,
        gcnii_lambda: float = 1.0,
        jk_mode: str = "last",
        channel_recalibration: str = "none",
        se_reduction: int = 4,
        regime_dim: int = 0,
    ) -> None:
        super().__init__()
        if node_feature_dim <= 0:
            raise ValueError("node_feature_dim must be positive")
        if edge_feature_dim <= 0:
            raise ValueError("edge_feature_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if message_layers < 0:
            raise ValueError("message_layers must be nonnegative")
        if region_feature_dim < 0:
            raise ValueError("region_feature_dim must be nonnegative")
        if init_mode not in {"xavier", "kaiming", "deterministic"}:
            raise ValueError("init_mode must be one of: xavier, kaiming, deterministic")
        if num_budget_bins <= 0:
            raise ValueError("num_budget_bins must be positive")
        if num_sectors <= 0:
            raise ValueError("num_sectors must be positive")
        if num_roles <= 0:
            raise ValueError("num_roles must be positive")
        if score_output_gain <= 0.0:
            raise ValueError("score_output_gain must be positive")
        if int(attention_heads) < 0:
            raise ValueError("attention_heads must be nonnegative")
        if float(gcnii_alpha) < 0.0 or float(gcnii_alpha) > 1.0:
            raise ValueError("gcnii_alpha must be in [0, 1]")
        if float(gcnii_lambda) <= 0.0:
            raise ValueError("gcnii_lambda must be positive")
        if str(jk_mode) not in {"last", "concat", "max"}:
            raise ValueError("jk_mode must be one of: last, concat, max")
        if str(channel_recalibration) not in {"none", "eca", "se"}:
            raise ValueError("channel_recalibration must be one of: none, eca, se")
        if int(se_reduction) <= 0:
            raise ValueError("se_reduction must be positive")
        if int(regime_dim) < 0:
            raise ValueError("regime_dim must be nonnegative")

        self.config = HierarchicalGNNConfig(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            message_layers=message_layers,
            mlp_layers=mlp_layers,
            region_feature_dim=region_feature_dim,
            use_region_context=use_region_context,
            init_mode=init_mode,
            enable_budget_head=bool(enable_budget_head),
            enable_region_bridge_head=bool(enable_region_bridge_head),
            enable_sector_head=bool(enable_sector_head),
            enable_role_head=bool(enable_role_head),
            num_budget_bins=int(num_budget_bins),
            num_sectors=int(num_sectors),
            num_roles=int(num_roles),
            use_structural_score_bias=bool(use_structural_score_bias),
            bridge_bias_weight=float(bridge_bias_weight),
            sector_bias_weight=float(sector_bias_weight),
            role_bias_weight=float(role_bias_weight),
            score_output_gain=float(score_output_gain),
            learnable_score_gain=bool(learnable_score_gain),
            score_standardization=bool(score_standardization),
            attention_heads=int(attention_heads),
            attention_negative_slope=float(attention_negative_slope),
            gcnii_alpha=float(gcnii_alpha),
            gcnii_lambda=float(gcnii_lambda),
            jk_mode=str(jk_mode),
            channel_recalibration=str(channel_recalibration),
            se_reduction=int(se_reduction),
            regime_dim=int(regime_dim),
        )
        self.score_output_gain = float(score_output_gain)
        self.learnable_score_gain = bool(learnable_score_gain)
        self.score_standardization = bool(score_standardization)
        self.jk_mode = str(jk_mode)
        self.node_encoder = _make_mlp(node_feature_dim, hidden_dim, hidden_dim, mlp_layers)
        self.edge_encoder = _make_mlp(edge_feature_dim, hidden_dim, hidden_dim, mlp_layers)
        # A2 GCNII identity-mapping decay beta_l = log(lambda / l + 1) is per-layer (1-based).
        self.message_blocks = torch.nn.ModuleList(
            [
                SparseMessagePassingBlock(
                    hidden_dim,
                    mlp_layers,
                    attention_heads=int(attention_heads),
                    attention_negative_slope=float(attention_negative_slope),
                    gcnii_alpha=float(gcnii_alpha),
                    gcnii_beta=(math.log(float(gcnii_lambda) / float(layer_index) + 1.0) if float(gcnii_alpha) > 0.0 else 0.0),
                )
                for layer_index in range(1, message_layers + 1)
            ]
        )
        # A3 JK "concat" needs a projection back to hidden_dim (only when >0 message layers).
        self.jk_proj = (
            _make_mlp(hidden_dim * message_layers, hidden_dim, hidden_dim, mlp_layers)
            if (str(jk_mode) == "concat" and message_layers > 0)
            else None
        )
        # A5 node-channel recalibration (applied before the edge-score / aux heads).
        if str(channel_recalibration) == "eca":
            self.channel_recal: torch.nn.Module | None = ECAChannelRecalibration(hidden_dim)
        elif str(channel_recalibration) == "se":
            self.channel_recal = SEChannelRecalibration(hidden_dim, reduction=int(se_reduction))
        else:
            self.channel_recal = None
        # S3 FiLM: one (gamma,beta) generator per modulated layer (encoder output + each message block).
        self.film = (
            FiLMConditioner(int(regime_dim), hidden_dim, message_layers + 1, mlp_layers)
            if int(regime_dim) > 0
            else None
        )
        self.region_feature_encoder = (
            _make_mlp(region_feature_dim, hidden_dim, hidden_dim, mlp_layers)
            if region_feature_dim > 0
            else None
        )
        self.region_update = _make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, mlp_layers)
        self.edge_score_head = _make_mlp(hidden_dim * 3, hidden_dim, 1, mlp_layers)
        self.budget_head = (
            _make_mlp(hidden_dim, hidden_dim, num_budget_bins, mlp_layers)
            if enable_budget_head
            else None
        )
        self.role_head = _make_mlp(hidden_dim, hidden_dim, num_roles, mlp_layers) if enable_role_head else None
        self.sector_head = (
            _make_mlp(hidden_dim, hidden_dim, num_sectors, mlp_layers) if enable_sector_head else None
        )
        self.region_bridge_head = (
            _make_mlp(hidden_dim * 2, hidden_dim, 1, mlp_layers) if enable_region_bridge_head else None
        )
        _initialize_module(self, init_mode)
        # Created after _initialize_module so the deterministic Linear ramp does
        # not touch it. exp(log_gain) keeps the gain strictly positive.
        if self.learnable_score_gain:
            self.score_log_gain = torch.nn.Parameter(
                torch.tensor(math.log(float(score_output_gain)), dtype=torch.float32)
            )
        else:
            self.register_parameter("score_log_gain", None)
        # Created after _initialize_module so the xavier/deterministic ramp does not overwrite the
        # identity (gamma=1, beta=0) start that keeps FiLM a stable near-no-op at init.
        if self.film is not None:
            self.film.zero_init_output()

    def _apply_film(self, node_embedding: torch.Tensor, regime_features: torch.Tensor | None, layer_idx: int) -> torch.Tensor:
        """S3 FiLM per-layer regime modulation (no-op / byte-identical when disabled or no regime given)."""
        if self.film is None or regime_features is None:
            return node_embedding
        return self.film.modulate(node_embedding, regime_features, layer_idx)

    def _apply_jk(self, layer_outputs: list[torch.Tensor], fallback: torch.Tensor) -> torch.Tensor:
        """A3 Jumping-Knowledge fusion of per-layer node embeddings. "last" / empty -> the last
        layer (byte-identical to legacy); "max" -> elementwise max; "concat" -> learned projection."""
        if self.jk_mode == "last" or not layer_outputs:
            return fallback
        if self.jk_mode == "max":
            return torch.stack(layer_outputs, dim=0).amax(dim=0)
        if self.jk_mode == "concat" and self.jk_proj is not None:
            return self.jk_proj(torch.cat(layer_outputs, dim=1))
        return fallback

    def _apply_channel_recal(self, node_embedding: torch.Tensor) -> torch.Tensor:
        """A5 node-channel recalibration (no-op / byte-identical when disabled)."""
        if self.channel_recal is None:
            return node_embedding
        return self.channel_recal(node_embedding)

    def compute_node_budget_logits(self, node_embedding: torch.Tensor) -> torch.Tensor:
        """Return soft node budget logits for future budget mapping."""

        if self.budget_head is None:
            raise NotImplementedError("budget head is disabled")
        return self.budget_head(node_embedding)

    def compute_region_bridge_logits(
        self,
        region_context: torch.Tensor,
        edge_src_region: torch.Tensor,
        edge_dst_region: torch.Tensor,
    ) -> torch.Tensor:
        """Return sparse per-edge region-pair bridge logits."""

        if self.region_bridge_head is None:
            raise NotImplementedError("region bridge head is disabled")
        pair_context = torch.cat(
            [
                region_context.index_select(0, edge_src_region),
                region_context.index_select(0, edge_dst_region),
            ],
            dim=1,
        )
        return self.region_bridge_head(pair_context).squeeze(-1)

    def compute_sector_preference_logits(self, node_embedding: torch.Tensor) -> torch.Tensor:
        """Return node-sector preference logits for sparse edge score bias."""

        if self.sector_head is None:
            raise NotImplementedError("sector head is disabled")
        return self.sector_head(node_embedding)

    def encode_nodes(
        self,
        *,
        num_nodes: int,
        src_index: torch.Tensor,
        dst_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        region_id: torch.Tensor | None = None,
        num_regions: int | None = None,
        regime_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-node and per-edge embeddings (encoder + message passing + region context).

        This is the same encoder path ``forward`` uses, exposed so temporal/recurrent
        models can add their own cross-time aggregation and edge head on top of the
        per-frame node embeddings. Returns ``(node_embedding [N, H], edge_embedding
        [E, H])``. Identity of node indices is the caller's responsibility (stable
        across frames for recurrent use).
        """
        node_features = _as_float_tensor("node_features", node_features)
        edge_features = _as_float_tensor("edge_features", edge_features)
        device = node_features.device
        src_index = _as_index_tensor("src_index", src_index, device=device)
        dst_index = _as_index_tensor("dst_index", dst_index, device=device)
        node_embedding = self.node_encoder(node_features)
        edge_embedding = self.edge_encoder(edge_features)
        node_embedding = self._apply_film(node_embedding, regime_features, 0)  # S3
        h0 = node_embedding  # A2 GCNII initial residual reference
        layer_outputs: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.message_blocks, start=1):
            node_embedding, _ = block(node_embedding, edge_embedding, src_index, dst_index, h0=h0)
            node_embedding = self._apply_film(node_embedding, regime_features, layer_index)  # S3
            layer_outputs.append(node_embedding)
        node_embedding = self._apply_jk(layer_outputs, node_embedding)  # A3
        if region_id is not None and self.config.use_region_context:
            region_id = _as_index_tensor("region_id", region_id, device=device)
            region_count = (
                int(num_regions) if num_regions is not None
                else (int(region_id.max().detach().item()) + 1 if num_nodes > 0 else 0)
            )
            region_sum = node_embedding.new_zeros((region_count, node_embedding.shape[1]))
            region_sum.index_add_(0, region_id, node_embedding)
            region_counts = node_embedding.new_zeros((region_count,))
            region_counts.index_add_(0, region_id, torch.ones_like(region_id, dtype=node_embedding.dtype))
            region_context = region_sum / region_counts.clamp_min(1.0).unsqueeze(1)
            node_embedding = self.region_update(torch.cat([node_embedding, region_context[region_id]], dim=1))
        node_embedding = self._apply_channel_recal(node_embedding)  # A5
        return node_embedding, edge_embedding

    def forward(
        self,
        *,
        num_nodes: int,
        src_index: torch.Tensor,
        dst_index: torch.Tensor,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        region_id: torch.Tensor | None = None,
        region_features: torch.Tensor | None = None,
        num_regions: int | None = None,
        edge_sector_id: torch.Tensor | None = None,
        edge_is_cross_region: torch.Tensor | None = None,
        regime_features: torch.Tensor | None = None,
        use_structural_score_bias: bool | None = None,
        bridge_bias_weight: float | None = None,
        sector_bias_weight: float | None = None,
        role_bias_weight: float | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if num_nodes < 0:
            raise ValueError("num_nodes must be nonnegative")
        node_features = _as_float_tensor("node_features", node_features)
        edge_features = _as_float_tensor("edge_features", edge_features)
        if node_features.ndim != 2:
            raise ValueError("node_features must have shape [N, F_node]")
        if edge_features.ndim != 2:
            raise ValueError("edge_features must have shape [E, F_edge]")
        if node_features.shape[0] != num_nodes:
            raise ValueError("node_features first dimension must equal num_nodes")
        if node_features.shape[1] != self.config.node_feature_dim:
            raise ValueError("node_features second dimension does not match model configuration")
        if edge_features.shape[1] != self.config.edge_feature_dim:
            raise ValueError("edge_features second dimension does not match model configuration")

        device = node_features.device
        src_index = _as_index_tensor("src_index", src_index, device=device)
        dst_index = _as_index_tensor("dst_index", dst_index, device=device)
        if src_index.shape != dst_index.shape:
            raise ValueError("src_index and dst_index must have matching shape")
        if edge_features.shape[0] != src_index.numel():
            raise ValueError("edge_features first dimension must equal edge count")
        if src_index.numel() > 0:
            if torch.any(src_index < 0) or torch.any(src_index >= num_nodes):
                raise ValueError("src_index contains out-of-range node ids")
            if torch.any(dst_index < 0) or torch.any(dst_index >= num_nodes):
                raise ValueError("dst_index contains out-of-range node ids")

        node_embedding = self.node_encoder(node_features)
        edge_embedding = self.edge_encoder(edge_features)
        node_embedding = self._apply_film(node_embedding, regime_features, 0)  # S3
        h0 = node_embedding  # A2 GCNII initial residual reference
        last_aggregated = node_embedding.new_zeros(node_embedding.shape)
        layer_outputs: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.message_blocks, start=1):
            node_embedding, last_aggregated = block(node_embedding, edge_embedding, src_index, dst_index, h0=h0)
            node_embedding = self._apply_film(node_embedding, regime_features, layer_index)  # S3
            layer_outputs.append(node_embedding)
        node_embedding = self._apply_jk(layer_outputs, node_embedding)  # A3

        region_count_value = 0
        region_context: torch.Tensor | None = None
        edge_src_region: torch.Tensor | None = None
        edge_dst_region: torch.Tensor | None = None
        if region_id is not None:
            region_id = _as_index_tensor("region_id", region_id, device=device)
            if region_id.shape != (num_nodes,):
                raise ValueError("region_id must have shape [num_nodes]")
            if torch.any(region_id < 0):
                raise ValueError("region_id must be nonnegative")
            if num_regions is not None:
                region_count = int(num_regions)
                if region_count < 0:
                    raise ValueError("num_regions must be nonnegative")
            else:
                region_count = int(region_id.max().detach().item()) + 1 if num_nodes > 0 else 0
            if num_nodes > 0 and torch.any(region_id >= region_count):
                raise ValueError("region_id must be less than num_regions")
            region_count_value = region_count
            region_sum = node_embedding.new_zeros((region_count, node_embedding.shape[1]))
            region_sum.index_add_(0, region_id, node_embedding)
            region_counts = node_embedding.new_zeros((region_count,))
            region_counts.index_add_(0, region_id, torch.ones_like(region_id, dtype=node_embedding.dtype))
            region_context = region_sum / region_counts.clamp_min(1.0).unsqueeze(1)
            if region_features is not None:
                if self.region_feature_encoder is None:
                    raise ValueError("region_feature_dim must be positive to consume region_features")
                region_features = _as_float_tensor("region_features", region_features)
                if region_features.ndim != 2:
                    raise ValueError("region_features must have shape [R, F_region]")
                if region_features.shape[0] != region_count:
                    raise ValueError("region_features first dimension must match region count")
                if region_features.shape[1] != self.config.region_feature_dim:
                    raise ValueError("region_features second dimension does not match model configuration")
                region_context = region_context + self.region_feature_encoder(region_features.to(device=device))
            if self.config.use_region_context:
                node_embedding = self.region_update(
                    torch.cat([node_embedding, region_context[region_id]], dim=1)
                )
            if src_index.numel() > 0:
                edge_src_region = region_id.index_select(0, src_index)
                edge_dst_region = region_id.index_select(0, dst_index)
        elif region_features is not None:
            raise ValueError("region_features require region_id")

        node_embedding = self._apply_channel_recal(node_embedding)  # A5 (before edge + aux heads)
        if src_index.numel() == 0:
            edge_score_base = edge_features.new_zeros((0,))
        else:
            score_input = torch.cat(
                [node_embedding[src_index], node_embedding[dst_index], edge_embedding],
                dim=1,
            )
            edge_score_base = self.edge_score_head(score_input).squeeze(-1)

        node_budget_logits = (
            self.compute_node_budget_logits(node_embedding)
            if self.budget_head is not None
            else edge_features.new_empty((num_nodes, 0))
        )
        if node_budget_logits.numel() > 0:
            budget_values = torch.arange(
                self.config.num_budget_bins,
                dtype=node_budget_logits.dtype,
                device=node_budget_logits.device,
            )
            node_budget_expected = torch.softmax(node_budget_logits, dim=1) @ budget_values
        else:
            node_budget_expected = edge_features.new_empty((0,))
        node_role_logits = (
            self.role_head(node_embedding) if self.role_head is not None else edge_features.new_empty((num_nodes, 0))
        )
        sector_preference_logits = (
            self.compute_sector_preference_logits(node_embedding)
            if self.sector_head is not None
            else edge_features.new_empty((num_nodes, 0))
        )
        if (
            self.region_bridge_head is not None
            and region_context is not None
            and edge_src_region is not None
            and edge_dst_region is not None
            and src_index.numel() > 0
        ):
            region_bridge_logits = self.compute_region_bridge_logits(
                region_context,
                edge_src_region,
                edge_dst_region,
            )
            region_bridge_pair_index = torch.stack([edge_src_region, edge_dst_region], dim=0)
        else:
            region_bridge_logits = edge_features.new_empty((0,))
            region_bridge_pair_index = src_index.new_empty((2, 0))

        if edge_sector_id is not None:
            edge_sector = _as_index_tensor("edge_sector_id", edge_sector_id, device=device)
            if edge_sector.shape != src_index.shape:
                raise ValueError("edge_sector_id must have one value per edge")
            if torch.any(edge_sector < 0) or torch.any(edge_sector >= self.config.num_sectors):
                raise ValueError("edge_sector_id contains values outside configured sectors")
        else:
            edge_sector = None

        if edge_is_cross_region is not None:
            if not torch.is_tensor(edge_is_cross_region):
                raise TypeError("edge_is_cross_region must be a torch.Tensor")
            if edge_is_cross_region.dtype != torch.bool:
                raise TypeError("edge_is_cross_region must use bool dtype")
            cross_region_mask = edge_is_cross_region.to(device=device).reshape(-1)
            if cross_region_mask.shape != src_index.shape:
                raise ValueError("edge_is_cross_region must have one value per edge")
        elif edge_src_region is not None and edge_dst_region is not None:
            cross_region_mask = edge_src_region != edge_dst_region
        else:
            cross_region_mask = src_index.new_zeros(src_index.shape, dtype=torch.bool)

        bias_enabled = self.config.use_structural_score_bias if use_structural_score_bias is None else bool(use_structural_score_bias)
        sector_bias = edge_score_base.new_zeros(edge_score_base.shape)
        role_bias = edge_score_base.new_zeros(edge_score_base.shape)
        bridge_bias = edge_score_base.new_zeros(edge_score_base.shape)
        sector_weight = self.config.sector_bias_weight if sector_bias_weight is None else float(sector_bias_weight)
        role_weight = self.config.role_bias_weight if role_bias_weight is None else float(role_bias_weight)
        bridge_weight = self.config.bridge_bias_weight if bridge_bias_weight is None else float(bridge_bias_weight)
        if bias_enabled and edge_score_base.numel() > 0:
            if edge_sector is not None and sector_preference_logits.numel() > 0:
                source_sector_logits = sector_preference_logits.index_select(0, src_index)
                sector_bias = source_sector_logits.gather(1, edge_sector.unsqueeze(1)).squeeze(1)
            if node_role_logits.numel() > 0 and self.config.num_roles > 1:
                support_role_logits = torch.logsumexp(node_role_logits[:, 1:], dim=1) - edge_score_base.new_tensor(
                    float(max(self.config.num_roles - 1, 1))
                ).log()
                role_bias = support_role_logits.index_select(0, dst_index)
            if region_bridge_logits.numel() == edge_score_base.numel():
                bridge_bias = region_bridge_logits * cross_region_mask.to(dtype=edge_score_base.dtype)
        structural_bias = (
            edge_score_base.new_tensor(sector_weight) * sector_bias
            + edge_score_base.new_tensor(role_weight) * role_bias
            + edge_score_base.new_tensor(bridge_weight) * bridge_bias
        )
        edge_score_pre_gain = edge_score_base + structural_bias
        # P2: standardize the score spread across the candidate population before
        # the gain so over-smoothed embeddings cannot collapse score DIFFERENCES.
        # (x - mean)/std is affine with positive scale, so the top-k ranking and
        # therefore the hard forward topology are unchanged.
        if self.score_standardization and edge_score_pre_gain.numel() > 1:
            score_mean = edge_score_pre_gain.mean()
            score_std = edge_score_pre_gain.std(unbiased=False)
            edge_score_pre_gain = (edge_score_pre_gain - score_mean) / torch.clamp(
                score_std, min=edge_score_pre_gain.new_tensor(1.0e-6)
            )
        if self.learnable_score_gain and self.score_log_gain is not None:
            gain = torch.exp(self.score_log_gain).to(dtype=edge_score_pre_gain.dtype)
        else:
            gain = edge_score_pre_gain.new_tensor(self.score_output_gain)
        edge_score = edge_score_pre_gain * gain
        effective_score_gain_value = float(gain.detach().cpu().item())

        incoming_count = node_embedding.new_zeros((num_nodes,))
        outgoing_count = node_embedding.new_zeros((num_nodes,))
        if dst_index.numel() > 0:
            incoming_count.index_add_(0, dst_index, torch.ones_like(dst_index, dtype=node_embedding.dtype))
        if src_index.numel() > 0:
            outgoing_count.index_add_(0, src_index, torch.ones_like(src_index, dtype=node_embedding.dtype))
        zero_message_node_count = (incoming_count <= 0).sum().to(dtype=node_embedding.dtype)
        zero_outgoing_candidate_node_count = (outgoing_count <= 0).sum().to(dtype=node_embedding.dtype)
        isolated_candidate_node_count = ((incoming_count <= 0) & (outgoing_count <= 0)).sum().to(dtype=node_embedding.dtype)
        node_embedding_norm = node_embedding.norm(dim=1) if num_nodes > 0 else edge_score.new_empty((0,))
        edge_embedding_norm = edge_embedding.norm(dim=1) if edge_embedding.numel() > 0 else edge_score.new_empty((0,))
        if edge_score.numel() == 0:
            edge_score_min = edge_score.new_tensor(0.0)
            edge_score_mean = edge_score.new_tensor(0.0)
            edge_score_max = edge_score.new_tensor(0.0)
            edge_score_std = edge_score.new_tensor(0.0)
            edge_score_p10 = edge_score.new_tensor(0.0)
            edge_score_p90 = edge_score.new_tensor(0.0)
        else:
            edge_score_min = edge_score.min()
            edge_score_mean = edge_score.mean()
            edge_score_max = edge_score.max()
            edge_score_std = edge_score.std(unbiased=False)
            edge_score_p10 = torch.quantile(edge_score, 0.10)
            edge_score_p90 = torch.quantile(edge_score, 0.90)
        diagnostics = {
            "edge_score_min": edge_score_min,
            "edge_score_mean": edge_score_mean,
            "edge_score_max": edge_score_max,
            "edge_score_std": edge_score_std,
            "edge_score_p10": edge_score_p10,
            "edge_score_p90": edge_score_p90,
            "edge_score_entropy_by_source_mean": _edge_score_entropy_by_source_mean(
                num_nodes=num_nodes,
                src_index=src_index,
                edge_score=edge_score,
                outgoing_count=outgoing_count,
            ),
            "node_embedding_norm_mean": node_embedding_norm.mean() if node_embedding_norm.numel() else edge_score.new_tensor(0.0),
            "node_embedding_norm_p90": torch.quantile(node_embedding_norm, 0.90) if node_embedding_norm.numel() else edge_score.new_tensor(0.0),
            "edge_embedding_norm_mean": edge_embedding_norm.mean() if edge_embedding_norm.numel() else edge_score.new_tensor(0.0),
            "edge_embedding_norm_p90": torch.quantile(edge_embedding_norm, 0.90) if edge_embedding_norm.numel() else edge_score.new_tensor(0.0),
            "message_count": edge_score.new_tensor(float(src_index.numel())),
            "region_count": edge_score.new_tensor(float(region_count_value)),
            "incoming_message_count_mean": incoming_count.mean() if num_nodes > 0 else edge_score.new_tensor(0.0),
            "incoming_message_count_max": incoming_count.max() if num_nodes > 0 else edge_score.new_tensor(0.0),
            "zero_message_node_count": zero_message_node_count,
            "zero_incoming_node_count": zero_message_node_count,
            "zero_outgoing_candidate_node_count": zero_outgoing_candidate_node_count,
            "isolated_candidate_node_count": isolated_candidate_node_count,
            "isolated_node_count": isolated_candidate_node_count,
            "structural_bias_mean": structural_bias.mean() if structural_bias.numel() else edge_score.new_tensor(0.0),
            "structural_bias_abs_max": structural_bias.abs().max() if structural_bias.numel() else edge_score.new_tensor(0.0),
            "last_message_norm_mean": last_aggregated.norm(dim=1).mean() if num_nodes > 0 else edge_score.new_tensor(0.0),
            "score_output_gain": edge_score.new_tensor(effective_score_gain_value),
            "score_output_gain_applied": edge_score.new_tensor(1.0 if abs(effective_score_gain_value - 1.0) > 1.0e-12 else 0.0),
            "learnable_score_gain": edge_score.new_tensor(1.0 if self.learnable_score_gain else 0.0),
            "score_standardization": edge_score.new_tensor(1.0 if self.score_standardization else 0.0),
        }
        return {
            "edge_score": edge_score,
            "edge_score_base": edge_score_base,
            "edge_score_pre_gain": edge_score_pre_gain,
            "structural_bias": structural_bias,
            "node_budget_logits": node_budget_logits,
            "node_budget_expected": node_budget_expected,
            "node_role_logits": node_role_logits,
            "region_bridge_logits": region_bridge_logits,
            "region_bridge_pair_index": region_bridge_pair_index,
            "sector_preference_logits": sector_preference_logits,
            "sector_bias": sector_bias,
            "role_bias": role_bias,
            "bridge_bias": bridge_bias,
            "diagnostics": diagnostics,
        }


def _edge_score_entropy_by_source_mean(
    *,
    num_nodes: int,
    src_index: torch.Tensor,
    edge_score: torch.Tensor,
    outgoing_count: torch.Tensor,
) -> torch.Tensor:
    if edge_score.numel() == 0 or num_nodes == 0:
        return edge_score.new_tensor(0.0)
    row_max = edge_score.new_full((num_nodes,), -torch.inf)
    row_max.scatter_reduce_(0, src_index, edge_score.detach(), reduce="amax", include_self=True)
    shifted = edge_score - row_max.index_select(0, src_index)
    exp_score = torch.exp(shifted)
    row_sum = edge_score.new_zeros((num_nodes,)).index_add(0, src_index, exp_score)
    probability = exp_score / torch.clamp(row_sum.index_select(0, src_index), min=torch.finfo(edge_score.dtype).tiny)
    edge_entropy = -probability * torch.log(torch.clamp(probability, min=torch.finfo(edge_score.dtype).tiny))
    row_entropy = edge_score.new_zeros((num_nodes,)).index_add(0, src_index, edge_entropy)
    active_source = outgoing_count > 0.0
    return row_entropy[active_source].mean() if bool(torch.any(active_source)) else edge_score.new_tensor(0.0)
