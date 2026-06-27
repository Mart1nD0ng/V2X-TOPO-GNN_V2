"""ESP/ESD-GNN performance-scale validation harness (Guarded-CDQ2 round, G-ESP-PERFORMANCE-SCALE).

Validates that a TRAINED ESP/ESD-GNN checkpoint preserves **real macrostate-basin performance** (judged
by the independent dynamic MC -- NOT runtime, NOT a node-union surrogate) across node counts, against a
scale-specific expert, the uniform-ESP and distance/link-quality heuristics. Reports scale-regret and
feasibility-retention under both a fixed protocol (exposes degradation) and a fixed service profile
(protocol calibrated by a pre-registered rule to hold the service target).

ESP/ESD-GNN here is the reliability-first policy: ``ESDGNN(use_cdq=False)`` -- a diagonal (no-diversity)
query law that LEARNS the per-edge quality ``s_ij`` from observable structure. Guarded-CDQ2 is a later
extension, disabled at this phase (constraint #3: ESP is the default).

Everything is namespace-clean (``macrostate_v2``) and hash-bound (the manifest enforces provenance +
>= 5 model seeds for a headline). Training uses the differentiable analytic episode; the dynamic MC is
the judge of record (constraint #10).
"""

from __future__ import annotations

import hashlib
import io
import math
import statistics
from dataclasses import dataclass

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.evaluation.cdq2_factorial import wilson_ci
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.optimization.macrostate_objective import train_macrostate
from src.sampling.baseline_policies import DistanceQueryPolicy, UniformQueryPolicy
from src.validation import run_dynamic_mc

__all__ = [
    "grid_for_target_N", "build_scale_instance", "checkpoint_hash", "train_esp_checkpoint",
    "policy_factory", "ScaleEval", "evaluate_macro", "feasible_ucb", "headline_cost",
    "scale_regret", "normalized_scale_regret", "feasibility_retention", "calibrated_profile",
]

COMM, BLOCK, INT = 95.0, 120.0, 150.0


# ------------------------------------------------------------------ scene scaling
# A Manhattan grid (gx, gy) with v vehicles/segment has N = ((gx-1)*gy + gx*(gy-1)) * v nodes. These
# grids span the spec's N=100..10000 with FIXED density (bounded degree -> no N x N).
_GRID_TABLE = [
    (100, (5, 5, 3)),      # 120
    (300, (8, 8, 3)),      # 336
    (600, (11, 11, 3)),    # 660
    (1000, (13, 13, 4)),   # 1248
    (3000, (23, 23, 3)),   # 3036
    (10000, (41, 41, 3)),  # 9840
]


def grid_for_target_N(target_N: int) -> tuple[int, int, int]:
    """The (gx, gy, v) Manhattan grid whose node count is closest to ``target_N`` (fixed density)."""
    return min(_GRID_TABLE, key=lambda kv: abs(kv[0] - target_N))[1]


def build_scale_instance(grid: tuple[int, int, int], seed: int, *, scenario: str = "iid",
                         base_node_err: float = 0.2, corr_strength: float = 0.3):
    """Build a ``(scene, evidence)`` instance at a grid + seed (deterministic). ``scenario`` selects the
    evidence covariance family (iid / region-block / overlapping common-cause)."""
    gx, gy, v = grid
    sc = build_manhattan_scene(gx, gy, v, block_m=BLOCK, comm_radius=COMM, int_radius=INT,
                               generator=torch.Generator().manual_seed(int(seed)))
    ev = build_overlapping_scenario(sc, scenario, base_node_err=base_node_err,
                                    corr_strength=corr_strength)
    return sc, ev


# ------------------------------------------------------------------ checkpoint hashing
def checkpoint_hash(model: torch.nn.Module) -> str:
    """Deterministic sha-256 of a model's ``state_dict`` (the checkpoint fingerprint for the manifest)."""
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _esp_config(hidden_dim: int, k: int) -> ESDGNNConfig:
    # use_cdq=False -> the diagonal ESP query law (reliability-first; no diversity head in the kernel).
    return ESDGNNConfig(hidden_dim=hidden_dim, r=max(4, k), n_enc=2, n_refine=1, k=k, use_cdq=False)


def train_esp_checkpoint(train_grids, *, seed: int, profile: ConsensusServiceProfile,
                         proto: ProtocolConfig, phy: RoundPhysicsConfig, scenario: str = "iid",
                         base_node_err: float = 0.2, steps: int = 40, scenes_per_grid: int = 2,
                         scene_seed0: int = 1000, hidden_dim: int = 16, lr: float = 5e-3,
                         link_override: float | None = None) -> dict:
    """Train ONE ESP/ESD-GNN checkpoint (a single model seed) on scenes drawn from ``train_grids``.

    ``link_override=None`` => the model is trained on the FULL physical chain (constraint: do NOT train
    on ideal links and evaluate on full physics). Returns ``{model, checkpoint_hash, model_seed,
    train_grids, history}``.
    """
    torch.manual_seed(int(seed))
    model = ESDGNN(_esp_config(hidden_dim, profile.k)).double()
    instances = [build_scale_instance(g, scene_seed0 + i, scenario=scenario, base_node_err=base_node_err)
                 for g in train_grids for i in range(scenes_per_grid)]
    res = train_macrostate(model, instances, proto, phy, profile, steps=steps, lr=lr,
                           link_override=link_override)
    return {"model": model, "checkpoint_hash": checkpoint_hash(model), "model_seed": int(seed),
            "train_grids": list(train_grids), "history": res["history"]}


# ------------------------------------------------------------------ policy factories (scale-agnostic)
def policy_factory(kind: str, *, model: ESDGNN | None = None, distance_beta: float = 0.04):
    """Return a ``scene -> policy`` factory. Kinds: ``esd_gnn`` (shared or expert checkpoint, needs
    ``model``), ``uniform_esp`` (all-equal ESP weights), ``distance`` (link-quality heuristic)."""
    if kind == "esd_gnn":
        if model is None:
            raise ValueError("esd_gnn policy needs a trained model")
        return lambda scene: ESDGNNQueryPolicy(model, scene)
    if kind == "uniform_esp":
        return lambda scene: UniformQueryPolicy()
    if kind == "distance":
        return lambda scene: DistanceQueryPolicy(beta_per_m=distance_beta)
    raise ValueError(f"unknown policy kind {kind!r}")


# ------------------------------------------------------------------ evaluation (dynamic MC, pooled, UCB)
@dataclass(frozen=True)
class ScaleEval:
    """Pooled macrostate basin outcomes for one (policy, scale) cell, judged by the dynamic MC."""

    N: int
    macro: dict          # namespaced macro block with *_ci (Wilson) on every outcome
    n_pool: int

    @property
    def P_correct(self) -> float:
        return self.macro["macro_P_correct"]


def evaluate_macro(grid, scene_seeds, policy_fn, profile, proto, phy, *, trials: int,
                   scenario: str = "iid", base_node_err: float = 0.2,
                   link_override: float | None = None) -> ScaleEval:
    """Dynamic-MC basin outcomes for ``policy_fn`` at ``grid``, pooled over ``scene_seeds`` (CRN).

    Each scene seed builds an independent scene+evidence; the MC samples evidence + the k-DPP subset +
    Bernoulli(ell) responses and reads peers' ACTUAL colours (it does NOT sample analytic terminal
    marginals -- constraint #10). Returns pooled basin outcomes with a Wilson CI on every outcome (the
    upper CI is the rare-failure UCB, spec §6.7).
    """
    rows = []
    Ns = []
    for s in scene_seeds:
        scene, ev = build_scale_instance(grid, s, scenario=scenario, base_node_err=base_node_err)
        Ns.append(scene.num_nodes)
        omega = uniform_participation(scene.num_nodes)
        pol = policy_fn(scene)
        r = run_dynamic_mc(scene, ev, pol, proto, phy, num_trials=trials,
                           generator=torch.Generator().manual_seed(int(s)), link_override=link_override,
                           service_profile=profile, participation=omega)
        rows.append(r)
    n_pool = trials * len(list(scene_seeds))

    def pooled(attr):
        return statistics.mean([getattr(r, attr) for r in rows])

    P, Fw, Fs, Fd = (pooled("basin_P_correct"), pooled("basin_F_wrong"),
                     pooled("basin_F_split"), pooled("basin_F_deadline"))
    ci = {"macro_P_correct": wilson_ci(P, n_pool), "macro_F_wrong": wilson_ci(Fw, n_pool),
          "macro_F_split": wilson_ci(Fs, n_pool), "macro_F_deadline": wilson_ci(Fd, n_pool)}
    macro = schema.macro_block(P, Fw, Fs, Fd, ci=ci)
    return ScaleEval(N=int(statistics.mode(Ns)), macro=macro, n_pool=n_pool)


def feasible_point(macro: dict, profile: ConsensusServiceProfile) -> bool:
    """Point-estimate feasibility: the OBSERVED F_wrong AND F_split <= the budget. Used for the
    feasibility-retention comparison at a bounded trial budget (the UCB at eps=1e-3 needs ~3800
    zero-failure trials per spec §6.7, run as a separate certification pass where affordable)."""
    return (macro["macro_F_wrong"] <= profile.max_wrong_basin_probability
            and macro["macro_F_split"] <= profile.max_split_basin_probability)


def feasible_ucb(macro: dict, profile: ConsensusServiceProfile) -> bool:
    """ESP reliability feasibility: the UPPER Wilson CI of F_wrong AND F_split must be <= the budget
    (constraint #5: wrong/split are HARD constraints; the UCB is the conservative rare-failure bound).
    Certifying eps=1e-3 requires ~3/eps zero-failure trials (spec §6.7)."""
    fw_ucb = macro.get("macro_F_wrong_ci", (0.0, 1.0))[1]
    fs_ucb = macro.get("macro_F_split_ci", (0.0, 1.0))[1]
    return (fw_ucb <= profile.max_wrong_basin_probability
            and fs_ucb <= profile.max_split_basin_probability)


def headline_cost(macro: dict, profile: ConsensusServiceProfile, *, ucb: bool = False) -> float:
    """Feasibility-gated performance cost J = 1 - P_correct (lower is better); +inf if INFEASIBLE
    (an infeasible policy is excluded before any performance comparison, constraint #5/#6). ``ucb``
    selects the conservative UCB feasibility gate; default is the point-estimate gate."""
    feas = feasible_ucb(macro, profile) if ucb else feasible_point(macro, profile)
    return (1.0 - macro["macro_P_correct"]) if feas else math.inf


# ------------------------------------------------------------------ scale regret / retention (spec §6.5)
def scale_regret(cost_shared: float, cost_expert: float) -> float:
    """``Regret_scale(N) = J_shared(N) - J_expert(N)`` (>= 0 when the expert is at least as good)."""
    return cost_shared - cost_expert


def normalized_scale_regret(cost_shared: float, cost_expert: float, cost_heuristic: float,
                            eps: float = 1e-9) -> float:
    """``(J_shared - J_expert) / (J_heuristic - J_expert + eps)`` -- regret as a fraction of the
    heuristic gap (0 = matches the expert, 1 = no better than the heuristic). ``inf`` costs -> ``nan``."""
    if not all(math.isfinite(c) for c in (cost_shared, cost_expert, cost_heuristic)):
        return float("nan")
    return (cost_shared - cost_expert) / (cost_heuristic - cost_expert + eps)


def feasibility_retention(shared_feasible: int, expert_feasible: int, eps: float = 1e-9) -> float:
    """``Pr(shared feasible) / Pr(expert feasible)`` over the evaluated scales (spec §6.5)."""
    return shared_feasible / (expert_feasible + eps)


# ------------------------------------------------------------------ protocol calibration
def calibrated_profile(base: ConsensusServiceProfile, target_N: int, *, mode: str,
                       base_N: int = 120) -> ConsensusServiceProfile:
    """Return the service profile to evaluate at ``target_N`` under one of two pre-registered modes.

    - ``"fixed_protocol"``: the SAME profile at every N (deadline budget constant -> EXPOSES the
      degradation as the scene's consensus diameter grows with N).
    - ``"fixed_service_profile"``: scale the deadline budget ``R_d = max_poll_epochs`` with the grid's
      consensus diameter (~ sqrt(N) for a 2D Manhattan grid) so the service TARGET is held across scales
      (the deployable rule). ``R_d(N) = round(R_d0 * sqrt(N / base_N))``, clamped to >= R_d0.
    """
    if mode == "fixed_protocol":
        return base
    if mode == "fixed_service_profile":
        factor = math.sqrt(max(target_N, base_N) / base_N)
        r_d = max(base.max_poll_epochs, round(base.max_poll_epochs * factor))
        return base.replace(max_poll_epochs=r_d)
    raise ValueError(f"unknown calibration mode {mode!r}")
