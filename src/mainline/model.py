"""Preference-conditioned topology GNN and the augmented-Chebyshev objective (spec §3.7, §10).

A single FiLM-conditioned message-passing GNN maps the candidate graph and a preference
vector ``lambda = (lambda_F, lambda_D, lambda_E)`` on the simplex to per-edge query logits
``s_{ij}`` and per-node power / blocklength logits (the §9.4 control heads).  Because the
preference is a model INPUT (FiLM modulation, Eq. spec §10), ONE trained checkpoint covers
a whole family of Pareto operating points: sweeping ``lambda`` traces out the F/D/E front.

Training uses the preference-conditioned augmented Chebyshev scalarisation (Eq. 57)

    L_lambda = max_m lambda_m (z_m - z_m*) / s_m  +  rho * sum_m lambda_m (z_m - z_m*) / s_m

with ``m in {F, D, E}``, utopia point ``z*`` and per-objective scales ``s``.

The forward pipeline is the full Eq. 56 spine:
    GNN(graph, lambda) -> (s, P, n) -> pi (G2) / ell (G4-G5) -> F,D (G1) , E (G6).
Everything is differentiable end to end; no degree cap / top-k, no beta-tail, no idealized
channel (the H-constraints are inherited from the gated mainline modules this composes).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .finite_blocklength import PathLoss3GPP, averaged_link_success
from .global_evaluator import build_bucketed_padding, evaluate_global_consensus
from .objectives import attempt_energy, blocklength_head, completion_delay, network_energy, power_head
from .symmetric_polynomials import edge_inclusion_probability
from .topology import CandidateGraph, los_probability, mode2_collision_from_load, receiver_load

__all__ = [
    "FiLM",
    "PreferenceConditionedTopologyGNN",
    "OperatingPointConfig",
    "model_operating_point",
    "evaluate_controls",
    "augmented_chebyshev",
    "pareto_indices",
    "sample_simplex",
]


def _scatter_mean(values: torch.Tensor, index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Mean of ``values`` (``[E, H]``) grouped by ``index`` into ``[num_nodes, H]``."""
    H = values.shape[-1]
    out = values.new_zeros((num_nodes, H))
    out = out.index_add(0, index, values)
    count = values.new_zeros((num_nodes,)).index_add(0, index, torch.ones_like(index, dtype=values.dtype))
    return out / count.clamp_min(1.0).unsqueeze(-1)


class FiLM(nn.Module):
    """Feature-wise linear modulation: preference ``lambda`` -> per-feature (scale, shift)."""

    def __init__(self, pref_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(pref_dim, hidden), nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.hidden = hidden

    def forward(self, h: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(lam)
        scale, shift = gamma_beta[: self.hidden], gamma_beta[self.hidden:]
        # 1+scale centres the per-feature gain at 1.0 (FiLM convention); not exact identity
        # since the FiLM MLP has biases.
        return (1.0 + scale) * h + shift


class _GNNLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, pref_dim: int):
        super().__init__()
        self.film = FiLM(pref_dim, hidden)
        self.msg = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.upd = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, h, src, dst, edge_feat, lam, num_nodes):
        h = self.film(h, lam)
        m = self.msg(torch.cat([h[src], h[dst], edge_feat], dim=-1))  # message j(dst)->i(src)
        agg = _scatter_mean(m, src, num_nodes)
        return h + self.upd(agg)  # residual


class PreferenceConditionedTopologyGNN(nn.Module):
    """FiLM-conditioned GNN producing query / power / blocklength logits (spec §3.7)."""

    def __init__(self, node_dim: int, edge_dim: int, *, hidden: int = 32, layers: int = 2, pref_dim: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(node_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([_GNNLayer(hidden, edge_dim, pref_dim) for _ in range(layers)])
        self.query_head = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.power_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.block_head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, node_feat, edge_feat, src, dst, lam, num_nodes):
        h = self.encoder(node_feat)
        for layer in self.layers:
            h = layer(h, src, dst, edge_feat, lam, num_nodes)
        edge_in = torch.cat([h[src], h[dst], edge_feat], dim=-1)
        query_logit = self.query_head(edge_in).squeeze(-1)  # [E]
        power_logit = self.power_head(h).squeeze(-1)        # [N]
        block_logit = self.block_head(h).squeeze(-1)        # [N]
        return query_logit, power_logit, block_logit


@dataclass(frozen=True)
class OperatingPointConfig:
    """Physics / consensus parameters for mapping model outputs to (F, D, E)."""

    k: int = 3
    alpha: int = 2
    beta: int = 2
    rounds: int = 10
    initial_correct_preference: float = 0.7
    p_min_dbm: float = 18.0
    p_max_dbm: float = 32.0
    n_min: float = 400.0
    n_max: float = 1100.0
    payload_bits: float = 4000.0
    fc_ghz: float = 5.9          # carrier frequency for the TR 37.885 geometry-grounded SINR
    noise_dbm: float = -95.0
    pathloss: PathLoss3GPP = field(default_factory=PathLoss3GPP)
    subchannels: float = 5.0
    max_harq_attempts: int = 2
    tau_proxy: float = 0.5  # representative transient prob for load-driven collision
    symbol_time_s: float = 1e-5
    rx_power_w: float = 0.01
    proc_energy_j: float = 1e-6


def _bucketed_inclusion_probability(padding, s_edge: torch.Tensor, k: int, num_edges: int) -> torch.Tensor:
    """Per-source inclusion probabilities ``pi`` (G2 / Eq. 18) via the degree-bucketed layout.

    Uses the same ``BucketedPadding`` as the consensus path (total padded cells ``<= 2E``),
    so the inclusion-probability head is ``O(E)`` with NO dense ``[N, max_deg]`` allocation --
    it cannot become ``O(N^2)`` under degree skew (a single hub of degree ``Theta(N)``), unlike
    the previous ``build_source_padding`` layout (D4 consistency, H2/H4-safe end to end).
    """
    pi = s_edge.new_zeros(num_edges)
    for bucket in padding.buckets:
        se = bucket.slot_edge  # [m, w]
        s_b = torch.where(bucket.slot_mask, s_edge[se], torch.zeros((), dtype=s_edge.dtype))
        pi_b = edge_inclusion_probability(s_b, k, mask=bucket.slot_mask)
        pi = pi.index_copy(0, se[bucket.slot_mask].reshape(-1), pi_b[bucket.slot_mask])
    return pi


def evaluate_controls(
    graph: CandidateGraph,
    s_edge: torch.Tensor,       # [E] query logits
    power_node: torch.Tensor,   # [N] dBm
    n_node: torch.Tensor,       # [N] blocklength (channel uses)
    cfg: OperatingPointConfig,
    padding=None,
) -> dict:
    """Map raw controls ``(s, P, n)`` through the SAME physics+consensus pipeline (pi -> ell ->
    F, D, E) the trained model uses.  Baselines and the model differ ONLY in how the controls
    are produced, so a baseline comparison through this function is fair (identical physics,
    G2/G4/G5/G1/G6).  Differentiable."""
    N, src, dst, E = graph.num_nodes, graph.src_index, graph.dst_index, graph.num_edges
    if padding is None:
        padding = build_bucketed_padding(src, dst, N)  # O(E) layout reused by pi + consensus
    dtype = s_edge.dtype

    # inclusion probabilities pi from the query logits (G2), degree-bucketed (no N x max_deg)
    pi = _bucketed_inclusion_probability(padding, s_edge, cfg.k, E)

    # link reliability: FBL on the source's power/blocklength, x mode-2 collision from load (G4/G5)
    pe, ne = power_node[src], n_node[src]
    # geometry-grounded per-edge SINR (TR 37.885): gamma depends on BOTH the source power
    # head AND the receiver distance, wiring the Eq. 34/56 geometry leg into the spine.
    import math
    pl = cfg.pathloss
    d = graph.distance.clamp_min(1.0)
    log_d = torch.log10(d)
    log_fc = math.log10(cfg.fc_ghz)
    pl_los = pl.los[0] + pl.los[1] * log_d + pl.los[2] * log_fc
    pl_nlos = pl.nlos[0] + pl.nlos[1] * log_d + pl.nlos[2] * log_fc
    pl_non = torch.maximum(pl_nlos, pl_los + pl.nlosv_extra_db)
    losp = los_probability(d)
    pl_db = losp * pl_los + (1.0 - losp) * pl_non
    rx_dbm = pe - pl_db  # per-edge received power [dBm] = source power - distance path loss
    gamma = torch.pow(torch.tensor(10.0, dtype=dtype), (rx_dbm - cfg.noise_dbm) / 10.0)
    ell_fbl = averaged_link_success(gamma, ne, cfg.payload_bits, fading="rayleigh",
                                    max_harq_attempts=cfg.max_harq_attempts)
    tau_proxy = torch.full((N,), cfg.tau_proxy, dtype=dtype)
    load = receiver_load(pi, tau_proxy, src, dst, N)            # Lambda_j (Eq. 33)
    p_col = mode2_collision_from_load(load[dst], cfg.subchannels)  # per-edge collision
    ell = (ell_fbl * (1.0 - p_col)).clamp(1e-4, 1.0 - 1e-9)

    omega = torch.ones(1, dtype=dtype)
    res = evaluate_global_consensus(
        num_nodes=N, src_index=src, dst_index=dst, log_query_weight=s_edge.unsqueeze(-1),
        link_reliability=ell.unsqueeze(-1), scenario_weight=omega, k=cfg.k, alpha=cfg.alpha,
        beta=cfg.beta, rounds=cfg.rounds, initial_correct_preference=cfg.initial_correct_preference,
        return_trajectory=True, padding=padding,  # reuse the same O(E) bucketed layout
    )
    F = res.F_global
    D = completion_delay(res.S_trajectory, tau_round=cfg.symbol_time_s * n_node.mean())["D"]
    e_att = attempt_energy(pe, ne, symbol_time_s=cfg.symbol_time_s,
                           rx_power_w=cfg.rx_power_w, proc_energy_j=cfg.proc_energy_j)
    Eobj = network_energy(res.tau_trajectory[:-1], pi.unsqueeze(-1), ell.unsqueeze(-1),
                          e_att.unsqueeze(-1), src, N, omega, cfg.max_harq_attempts)["E"]
    # c_ir / scenario_posterior are exposed (additively) so the G8 global-risk emission
    # (Eq. 58-59) can be computed from the SAME consensus pass; F_global is the raw scalar.
    return {"F": F, "D": D, "E": Eobj, "power_node": power_node, "n_node": n_node, "pi": pi,
            "ell": ell, "c_ir": res.c_ir, "scenario_posterior": res.scenario_posterior,
            "F_global": res.F_global}


def model_operating_point(
    model: PreferenceConditionedTopologyGNN,
    graph: CandidateGraph,
    node_feat: torch.Tensor,
    edge_feat: torch.Tensor,
    lam: torch.Tensor,
    cfg: OperatingPointConfig,
    padding=None,
) -> dict:
    """Full Eq. 56 forward: model + preference -> (F, D, E).  Differentiable.

    The trained GNN produces the controls ``(s, P, n)``; :func:`evaluate_controls` maps them
    through the shared physics/consensus pipeline.  Baselines reuse the SAME pipeline with
    non-learned controls (G11)."""
    N, src, dst = graph.num_nodes, graph.src_index, graph.dst_index
    if padding is None:
        padding = build_bucketed_padding(src, dst, N)
    s_edge, p_logit, b_logit = model(node_feat, edge_feat, src, dst, lam, N)
    power_node = power_head(p_logit, cfg.p_min_dbm, cfg.p_max_dbm)   # [N] dBm (Eq. 54)
    n_node = blocklength_head(b_logit, cfg.n_min, cfg.n_max)        # [N] (Eq. 55)
    return evaluate_controls(graph, s_edge, power_node, n_node, cfg, padding)


def augmented_chebyshev(
    z: torch.Tensor, lam: torch.Tensor, z_star: torch.Tensor, scales: torch.Tensor, rho: float = 0.05
) -> torch.Tensor:
    """Augmented Chebyshev scalarisation ``max_m t_m + rho sum_m t_m`` (Eq. 57)."""
    t = lam * (z - z_star) / scales
    return t.max() + rho * t.sum()


def pareto_indices(points: list[tuple], tol: float = 1e-12) -> list[int]:
    """Indices of mutually non-dominated points (minimisation in every coordinate).

    Uses a SINGLE symmetric tolerance for both the weak-dominance and strict-improvement
    tests: an asymmetric pair (e.g. +1e-9 / -1e-6) leaves a 'dead zone' in which a genuine
    dominator is missed, OVER-reporting non-dominated points -- the dangerous direction for
    an acceptance gate.  ``tol`` should be tiny for already-scaled objectives.
    """
    def dominates(a, b):
        return all(a[i] <= b[i] + tol for i in range(len(a))) and any(a[i] < b[i] - tol for i in range(len(a)))

    nd = []
    for i, a in enumerate(points):
        if not any(dominates(points[j], a) for j in range(len(points)) if j != i):
            nd.append(i)
    return nd


def sample_simplex(num: int, dim: int = 3, generator: torch.Generator | None = None,
                   dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Sample ``num`` preference vectors uniformly on the simplex (Dirichlet(1))."""
    g = -torch.log(torch.rand(num, dim, generator=generator, dtype=dtype).clamp_min(1e-12))
    return g / g.sum(dim=1, keepdim=True)


def directional_steering(points: list[tuple], lambdas: list) -> int:
    """Count objectives ``m`` whose minimiser over the sweep emphasises preference ``m``.

    For each objective, find the swept operating point with the smallest value; the
    preference that produced it should have ``lambda_m`` as (one of) its largest
    component(s).  Returns the count in ``{0..len}`` -- a robust steering metric (vs a
    single pairwise comparison, which the F U-shape can make noisy).
    """
    import numpy as np
    A = np.array(points)
    L = np.array(lambdas)
    count = 0
    for m in range(A.shape[1]):
        j = int(np.argmin(A[:, m]))
        if L[j, m] >= L[j].max() - 1e-9:
            count += 1
    return count


def train_preference_model(
    model: torch.nn.Module,
    forward_fn,
    *,
    steps: int,
    lr: float = 5e-3,
    refresh: int = 100,
    rho: float = 0.05,
    blind: bool = False,
    seed: int = 0,
) -> torch.nn.Module:
    """Train ``model`` with the augmented-Chebyshev objective over sampled preferences.

    ``forward_fn(lambda) -> {"F","D","E"}`` runs the full operating-point pipeline.  The
    utopia ``z*`` and scales are RE-ESTIMATED every ``refresh`` steps from the current
    model -- a stale (untrained) ``z*``/``scales`` mis-calibrates the scalarisation and
    breaks directional steering, so periodic refresh is essential.  ``blind=True`` trains
    with a constant preference (the discriminative ablation that must NOT steer).
    """
    gen = torch.Generator().manual_seed(seed)
    sample0 = forward_fn(torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=torch.float64))
    dtype = sample0["F"].dtype

    def estimate():
        with torch.no_grad():
            rows = [[float(o["F"]), float(o["D"]), float(o["E"])]
                    for o in (forward_fn(lam) for lam in sample_simplex(24, generator=gen, dtype=dtype))]
        Z = torch.tensor(rows, dtype=dtype)
        return Z.min(0).values - 1e-6, (Z.max(0).values - Z.min(0).values).clamp_min(1e-6)

    z_star, scales = estimate()
    blind_lam = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for t in range(steps):
        if refresh and t > 0 and t % refresh == 0:
            z_star, scales = estimate()
        lam = blind_lam if blind else sample_simplex(1, generator=gen, dtype=dtype)[0]
        o = forward_fn(lam)
        loss = augmented_chebyshev(torch.stack([o["F"], o["D"], o["E"]]), lam, z_star, scales, rho)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model
