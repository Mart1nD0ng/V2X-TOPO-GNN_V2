"""Shared primitives for generalization harnesses (multi-snapshot / mobility /
density). Builds envs from snapshots, trains one model across train envs, and
evaluates learned vs heuristic on test envs — all on the same constructor +
evaluator used in production.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import evaluate_v2x_graph_consensus  # noqa: E402
from src.losses import compute_coupled_loss  # noqa: E402
from src.losses.gradnorm import GradNormBalancer, compute_task_gradient_vectors  # noqa: E402
from src.losses.pcgrad import pcgrad_project  # noqa: E402
from src.topology import TopologyConstructionLayer  # noqa: E402
from src.v2x_env.candidate_graph import build_candidate_graph  # noqa: E402
from src.v2x_env.channel_model import ChannelConfig  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _avalanche_config,
    _build_feature_tensors,
    _evaluator_energy_config,
    _evaluator_physical_config,
    _initial_preferences,
    _loss_config,
    _make_model,
)

# production candidate-graph geometry (matches _make_environment / production_density_v0)
CANDIDATE_RADIUS_M = 230.0
CANDIDATE_CAP = 8
_CHANNEL = ChannelConfig(tx_power_dbm=23.0, mcs_threshold_db=8.0, transition_width_db=3.0)


def env_from_snapshot(
    snapshot,
    cfg: dict,
    *,
    label: Any = None,
    channel: ChannelConfig | None = None,
    interference_coupling_db: float | None = None,
) -> dict[str, Any]:
    physical = _evaluator_physical_config(cfg)
    coupling = (
        float(physical.get("interference_density_coupling_db", 0.0))
        if interference_coupling_db is None
        else float(interference_coupling_db)
    )
    # P1-1 standard-environment plumbing: optional `channel:` / `candidate_graph:` config sections
    # (cfg["channel_config"] / cfg["candidate_config"] after _normalized_config) override the
    # hardcoded production defaults, mirroring training_smoke._make_environment. Absent -> byte-identical.
    channel_overrides = cfg.get("channel_config", {}) or {}
    effective_channel = channel or (
        ChannelConfig.from_mapping(
            {"tx_power_dbm": 23.0, "mcs_threshold_db": 8.0, "transition_width_db": 3.0, **channel_overrides}
        )
        if channel_overrides else _CHANNEL
    )
    candidate_overrides = cfg.get("candidate_config", {}) or {}
    candidate = build_candidate_graph(
        snapshot, effective_channel,
        {"radius_m": CANDIDATE_RADIUS_M, "max_candidates_per_node": CANDIDATE_CAP,
         "cell_size_m": CANDIDATE_RADIUS_M,
         "interference_density_coupling_db": coupling,
         "interference_reference_degree": float(CANDIDATE_CAP),
         **candidate_overrides},
    )
    features = _build_feature_tensors(snapshot, candidate)
    ic, iw = _initial_preferences(cfg, candidate.num_nodes)
    # D3-fix-1: bake this env's evaluator physical_config (with ITS coupling) so a mixture trained
    # across mixed-coupling cells evaluates each env on matched feature/evaluator physics. With no
    # per-env override (interference_coupling_db=None) this equals _evaluator_physical_config(cfg)
    # exactly, so evaluate() stays byte-identical for every existing single-cfg caller.
    env_physical = dict(_evaluator_physical_config(cfg))
    env_physical["interference_density_coupling_db"] = coupling
    return {"label": label, "candidate": candidate, "features": features, "ic": ic, "iw": iw,
            "physical_config": env_physical}


def build_topology_layer(cfg: dict) -> TopologyConstructionLayer:
    budget = None if cfg["max_out_degree"] is None else int(cfg["max_out_degree"])
    layer = TopologyConstructionLayer(
        max_out_degree=budget, support_mode=str(cfg["support_mode"]), temperature=1.0,
        topk_backend=str(cfg["topk_backend"]),
        gradient_mode=str(cfg.get("gradient_mode", "selected_row_softmax")),
        straight_through_temperature=cfg.get("straight_through_temperature", None),
    )
    return layer


def caps_for(env, cfg: dict):
    budget = None if cfg["max_out_degree"] is None else int(cfg["max_out_degree"])
    return torch.full((env["candidate"].num_nodes,), budget, dtype=torch.long) if budget else None


def model_score(model, env) -> torch.Tensor:
    f = env["features"]
    return model(
        num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
        node_features=f["node_features"], edge_features=f["edge_features"],
        region_id=f["region_id"], num_regions=f["num_regions"],
        edge_sector_id=f["edge_sector_id"], edge_is_cross_region=f["edge_is_cross_region"],
        use_structural_score_bias=False,
    )["edge_score"]


def evaluate(score_or_model, env, topology_layer, cfg) -> dict[str, Any]:
    f = env["features"]
    score = score_or_model if isinstance(score_or_model, torch.Tensor) else model_score(score_or_model, env)
    topo = topology_layer(
        num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
        edge_score=score, per_node_budget=caps_for(env, cfg),
    )
    sel = topo.selected_candidate_index
    return evaluate_v2x_graph_consensus(
        **topo.as_evaluation_kwargs(),
        distance_m=f["distance_m"].index_select(0, sel), los_flag=f["los_flag"].index_select(0, sel),
        node_initial_correct=env["ic"], node_initial_wrong=env["iw"],
        physical_config=env.get("physical_config") or _evaluator_physical_config(cfg),
        avalanche_config=_avalanche_config(cfg),
        energy_config=_evaluator_energy_config(cfg),
    )


def metrics(ev) -> dict[str, float]:
    return {
        "C": float(ev["C_avalanche_node_mean"].mean()), "F": float(ev["F_avalanche_node_mean"].mean()),
        "D": float(ev["D_avalanche_rounds_mean"].mean()), "E": float(ev["E_consensus_node_mean"].mean()),
    }


def best_heuristic_F(env, topology_layer, cfg) -> float:
    f = env["features"]
    dist = f["distance_m"].to(dtype=torch.float64)
    chan = f["edge_features"].to(dtype=torch.float64)[:, 2]
    best = None
    for score in (-dist, chan):
        with torch.no_grad():
            fv = float(evaluate(score.reshape(-1), env, topology_layer, cfg)["F_avalanche_node_mean"].mean())
        best = fv if best is None else min(best, fv)
    return best


def train_model(cfg: dict, train_envs, topology_layer, max_steps: int, *, model_seed: int = 42):
    torch.manual_seed(int(model_seed))
    model = _make_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_cfg = _loss_config(cfg)
    for step in range(int(max_steps)):
        env = train_envs[step % len(train_envs)]
        opt.zero_grad(set_to_none=True)
        ev = evaluate(model_score(model, env), env, topology_layer, cfg)
        lo = compute_coupled_loss(ev, loss_cfg)
        (lo.get("effective_backward_loss", lo["total_loss"])).backward()
        opt.step()
    return model


def learned_F(model, env, topology_layer, cfg) -> tuple[float, float]:
    with torch.no_grad():
        m = metrics(evaluate(model, env, topology_layer, cfg))
    return m["F"], m["C"]


# ---- T1: per-group gradient governance (fix cross-density negative transfer) ----

def _group_gradient_vector(model, params, cells, topology_layer, cfg, loss_cfg) -> torch.Tensor:
    """Flat gradient vector of the mean coupled loss over a list of envs (one density group), one
    backward over that group only (bounds live graphs). allow_unused -> zero-fill for stable layout."""
    total = None
    for env in cells:
        lo = compute_coupled_loss(evaluate(model_score(model, env), env, topology_layer, cfg), loss_cfg)
        l = lo.get("effective_backward_loss", lo["total_loss"])
        total = l if total is None else total + l
    total = total / len(cells)
    grads = torch.autograd.grad(total, params, retain_graph=False, allow_unused=True)
    vec = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads, params)])
    return vec.detach(), float(total.detach())


def density_gradient_conflict(cfg: dict, envs_by_group: dict, topology_layer, *, model_seed: int = 42) -> dict:
    """GO/NO-GO diagnostic: per-group gradient L2 norm + pairwise cosine at the shared init. Strongly
    NEGATIVE cosines -> directional conflict (use PCGrad); near-zero cosines but imbalanced norms ->
    magnitude starvation (use GradNorm)."""
    torch.manual_seed(int(model_seed))
    model = _make_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    loss_cfg = _loss_config(cfg)
    keys = sorted(envs_by_group)
    vecs, norms = {}, {}
    for k in keys:
        v, _ = _group_gradient_vector(model, params, envs_by_group[k], topology_layer, cfg, loss_cfg)
        vecs[k] = v
        norms[k] = float(v.norm())
    cosines = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            denom = (vecs[a].norm() * vecs[b].norm()).clamp_min(1e-12)
            cosines[f"{a}|{b}"] = float(torch.dot(vecs[a], vecs[b]) / denom)
    return {"group_grad_norms": norms, "pairwise_cosines": cosines}


def train_model_governed(cfg: dict, envs_by_group: dict, topology_layer, max_steps: int, *,
                         model_seed: int = 42, mode: str = "pcgrad"):
    """Mixture training with per-GROUP gradient governance. Each step draws ONE env from every group
    (round-robin within group), computes one detached gradient vector per group, and merges them either
    by PCGrad projection (de-conflict opposing directions) or GradNorm reweighting (rebalance
    magnitudes), then applies the merged gradient. Bounds live graphs to one group at a time."""
    if mode not in ("pcgrad", "gradnorm"):
        raise ValueError("mode must be 'pcgrad' or 'gradnorm'")
    torch.manual_seed(int(model_seed))
    model = _make_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_cfg = _loss_config(cfg)
    groups = sorted(envs_by_group)
    balancer = GradNormBalancer(tasks=[str(g) for g in groups]) if mode == "gradnorm" else None
    for step in range(int(max_steps)):
        vecs, losses, norms = [], {}, {}
        for g in groups:
            cells = envs_by_group[g]
            env = cells[step % len(cells)]
            vec, lval = _group_gradient_vector(model, params, [env], topology_layer, cfg, loss_cfg)
            vecs.append(vec)
            losses[str(g)] = lval
            norms[str(g)] = float(vec.norm())
        if mode == "pcgrad":
            projected = pcgrad_project(vecs)
            merged = torch.stack([p.reshape(-1) for p in projected]).mean(dim=0)
        else:
            w = balancer.update(losses, norms)
            merged = sum(float(w[str(g)]) * vecs[i] for i, g in enumerate(groups))
        opt.zero_grad(set_to_none=True)
        offset = 0
        for p in params:
            n = p.numel()
            p.grad = merged[offset:offset + n].view_as(p).to(p.dtype).clone()
            offset += n
        opt.step()
    return model
