"""ESP/ESD-GNN training-budget calibration (esp_performance_scale_v2, G-ESP-TRAINING-BUDGET).

Answers the pivotal question the prior round left open: *does longer full-physics training materially
improve the ESP/ESD-GNN macrostate performance — and does it beat the distance heuristic, or is the
architecture not yet competitive?* (workflow §4 / interpretation §13).

`train_with_curve` runs the **canonical** training loop (same `run_consensus_episode` + `macrostate_metrics`
+ `macrostate_lagrangian` + `MacrostateDuals` as `train_macrostate`) with a **persistent optimizer** and a
**held-out validation hook** at each budget checkpoint — so the budget curve is a continuous trajectory,
not a sequence of optimizer restarts. Validation uses the differentiable analytic macro on held-out scenes
(screening only; the dynamic MC is the final judge — workflow §1.3). Training is full physics
(`link_override=None`); train==eval physics (constraint #3).
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.environment.canonical_episode import run_consensus_episode
from src.evaluation.esp_scale import _esp_config, build_scale_instance, checkpoint_hash
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.models import ESDGNN, ESDGNNQueryPolicy
from src.optimization.macrostate_objective import (
    MacrostateDuals, macrostate_lagrangian, macrostate_metrics,
)
from src.sampling.baseline_policies import DistanceQueryPolicy

__all__ = [
    "validate_macro_analytic", "train_with_curve", "budget_improvement", "select_best_budget",
]


def _to_f(x):
    return float(x.detach()) if isinstance(x, torch.Tensor) else float(x)


def validate_macro_analytic(policy_fn, val_instances, profile, proto, phy, *, link_override=None,
                            beta_T: float = 1.0) -> dict:
    """Held-out **analytic** macro (screening): mean over ``val_instances`` of the canonical-episode
    macrostate outcomes for ``policy_fn(scene)``. Returns a namespaced macro block (no CIs — analytic)."""
    Ps, Fws, Fss, Fds = [], [], [], []
    for scene, ev in val_instances:
        omega = uniform_participation(scene.num_nodes, dtype=torch.float64, device=scene.positions.device)
        res = run_consensus_episode(scene, ev, policy_fn(scene), proto, phy, return_trajectory=True,
                                    link_override=link_override)
        m = macrostate_metrics(res.c_trajectory, res.w_trajectory, omega, res.scenario_weight,
                               res.energy, profile, beta_T=beta_T)
        Ps.append(_to_f(m["P_correct"])); Fws.append(_to_f(m["F_wrong"]))
        Fss.append(_to_f(m["F_split"])); Fds.append(_to_f(m["F_deadline"]))
    mean = statistics.mean
    return schema.macro_block(mean(Ps), mean(Fws), mean(Fss), mean(Fds))


@dataclass(frozen=True)
class _Curve:
    model_seed: int
    checkpoint_hash: str
    curve: dict           # budget_step -> {"gnn": macro_block, "distance": macro_block}
    loss_history: list
    train_P_history: list


def train_with_curve(train_grids, *, seed: int, profile: ConsensusServiceProfile, proto: ProtocolConfig,
                     phy: RoundPhysicsConfig, total_steps: int, budget_points, val_grids, val_seeds,
                     scenario: str = "iid", base_node_err: float = 0.2, corr_strength: float = 0.3,
                     hidden_dim: int = 16, lr: float = 5e-3, eta_mu: float = 5.0, beta_T: float = 1.0,
                     scenes_per_grid: int = 2, scene_seed0: int = 1000, link_override=None,
                     distance_beta: float = 0.04) -> _Curve:
    """Train ONE ESP/ESD-GNN model seed for ``total_steps`` (full physics), recording the held-out
    validation macro (GNN vs the distance heuristic) at every step in ``budget_points``."""
    torch.manual_seed(int(seed))
    model = ESDGNN(_esp_config(hidden_dim, profile.k)).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    duals = MacrostateDuals()

    inst = lambda g, sd: build_scale_instance(g, sd, scenario=scenario, base_node_err=base_node_err,
                                              corr_strength=corr_strength)
    train = [inst(g, scene_seed0 + i) for g in train_grids for i in range(scenes_per_grid)]
    val = [inst(g, 7000 + s) for g in val_grids for s in val_seeds]    # held-out (disjoint seed range)

    budget = set(int(b) for b in budget_points)
    curve, loss_hist, trainP_hist = {}, [], []
    n = len(train)
    for step in range(1, total_steps + 1):
        scene, ev = train[(step - 1) % n]
        omega = uniform_participation(scene.num_nodes, dtype=torch.float64, device=scene.positions.device)
        opt.zero_grad()
        res = run_consensus_episode(scene, ev, ESDGNNQueryPolicy(model, scene), proto, phy,
                                    return_trajectory=True, link_override=link_override)
        m = macrostate_metrics(res.c_trajectory, res.w_trajectory, omega, res.scenario_weight,
                               res.energy, profile, beta_T=beta_T)
        loss = macrostate_lagrangian(m, duals, profile)
        loss.backward(); opt.step(); duals.update(m, profile, eta_mu)
        loss_hist.append(_to_f(loss)); trainP_hist.append(_to_f(m["P_correct"]))
        if step in budget:
            gnn = validate_macro_analytic(lambda sc: ESDGNNQueryPolicy(model, sc), val, profile, proto,
                                          phy, link_override=link_override, beta_T=beta_T)
            dist = validate_macro_analytic(lambda sc: DistanceQueryPolicy(beta_per_m=distance_beta), val,
                                           profile, proto, phy, link_override=link_override, beta_T=beta_T)
            curve[step] = {"gnn": gnn, "distance": dist}
    return _Curve(int(seed), checkpoint_hash(model), curve, loss_hist, trainP_hist)


def budget_improvement(curves, *, plateau_tol: float = 0.01) -> dict:
    """Aggregate the budget question over model seeds (workflow §4.5). Returns whether the FULL budget
    improves the validation `macro_P_correct` over PILOT, whether it plateaus, and the per-budget means."""
    pts = sorted({b for c in curves for b in c.curve})
    if not pts:
        return {"budget_points": [], "improves_over_pilot": False, "plateaued": False}
    def mean_at(b, who):
        return statistics.mean([c.curve[b][who]["macro_P_correct"] for c in curves if b in c.curve])
    gnn_means = {b: mean_at(b, "gnn") for b in pts}
    dist_means = {b: mean_at(b, "distance") for b in pts}
    pilot, full = pts[0], pts[-1]
    medium = pts[len(pts) // 2]
    return {
        "budget_points": pts,
        "gnn_P_correct_by_budget": gnn_means,
        "distance_P_correct_by_budget": dist_means,
        "improves_over_pilot": gnn_means[full] > gnn_means[pilot] + plateau_tol,
        "plateaued": abs(gnn_means[full] - gnn_means[medium]) <= plateau_tol,
        "beats_distance_at_full": gnn_means[full] > dist_means[full] + plateau_tol,
        "pilot": pilot, "medium": medium, "full": full,
    }


def select_best_budget(curves) -> int:
    """The budget step with the best mean held-out `macro_P_correct` (the checkpoint family to carry
    forward). Ties resolve to the SMALLER budget (cheaper)."""
    pts = sorted({b for c in curves for b in c.curve})
    return max(pts, key=lambda b: (statistics.mean([c.curve[b]["gnn"]["macro_P_correct"]
                                                    for c in curves if b in c.curve]), -b))
