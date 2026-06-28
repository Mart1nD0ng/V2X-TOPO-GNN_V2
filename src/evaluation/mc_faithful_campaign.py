"""MC-faithful ESP/ESD-GNN training campaign (esp_performance_scale_v2, Campaign A / publication-grade).

EV4/EV5 established that per-node-credit MC-faithful REINFORCE (src/optimization/mc_reinforce.py) closes the
EV2 training-signal gap: it trains the GNN on a gradient that reflects what the dynamic-MC judge measures,
where the analytic surrogate was flat. This module turns that into a reusable campaign harness:

  * ``train_mc_faithful`` trains ONE model seed by ``train_esp_reinforce`` and returns a checkpointable result.
  * ``save_checkpoint`` / ``load_checkpoint`` persist the trained ``state_dict`` + config + hash, so EXPENSIVE
    training compute is done ONCE and re-used across every downstream eval (CI-separation, scale sweep, OOD,
    rare-event) -- EV5 wastefully retrained for every eval.
  * ``aggregate_seed_macros`` pools per-seed macro blocks into a >=5-seed headline (mean + pooled Wilson CI).

The eval itself reuses the canonical ``esp_scale.evaluate_macro`` (dynamic-MC basin first-hitting, full
physics) so the judge code is shared verbatim. Everything is namespace-clean (macrostate_v2) + hash-bound.

train==eval distribution: ``corr_strength`` is threaded through both training (``build_scale_instance``) and
eval (``evaluate_macro``, extended here) so the held-out eval scenes share the training family exactly
(constraint: full-physics train==eval; no hidden train/eval distribution shift).
"""
from __future__ import annotations

import io
import os
import statistics
from dataclasses import dataclass

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.cdq2_factorial import wilson_ci
from src.evaluation.esp_scale import _esp_config, build_scale_instance, checkpoint_hash
from src.metrics import schema
from src.models import ESDGNN, ESDGNNQueryPolicy
from src.optimization.mc_reinforce import train_esp_reinforce

__all__ = [
    "MCFaithfulCheckpoint", "train_mc_faithful", "save_checkpoint", "load_checkpoint",
    "checkpoint_policy_factory", "aggregate_seed_macros",
]


@dataclass
class MCFaithfulCheckpoint:
    """A trained MC-faithful ESP/ESD-GNN checkpoint + its provenance."""

    model: ESDGNN
    checkpoint_hash: str
    model_seed: int
    hidden_dim: int
    k: int
    train_meta: dict          # regime/grid/steps/trials/lr/scene seeds -- the full training descriptor
    history: dict             # per-step {loss, mc_P_correct, correct_mass}


def train_mc_faithful(seed: int, grid: tuple[int, int, int], *, profile: ConsensusServiceProfile,
                      proto: ProtocolConfig, phy: RoundPhysicsConfig, scenario: str, base_node_err: float,
                      corr_strength: float, steps: int, trials: int = 100, n_train_scenes: int = 2,
                      lr: float = 1e-2, hidden_dim: int = 16, train_scene_seed0: int = 1000,
                      base_seed: int = 0, link_override: float | None = None) -> MCFaithfulCheckpoint:
    """Train ONE MC-faithful ESP/ESD-GNN checkpoint (model seed ``seed``).

    ``link_override=None`` => trained on the FULL physical chain (constraint: do NOT train on ideal links and
    evaluate on full physics). The ``base_seed`` decorrelates per-step MC generators across model seeds.
    """
    torch.manual_seed(int(seed))
    model = ESDGNN(_esp_config(hidden_dim, profile.k)).double()
    instances = [build_scale_instance(grid, train_scene_seed0 + i, scenario=scenario,
                                      base_node_err=base_node_err, corr_strength=corr_strength)
                 for i in range(n_train_scenes)]
    res = train_esp_reinforce(model, instances, proto, phy, profile, steps=steps, trials=trials, lr=lr,
                              link_override=link_override, base_seed=base_seed)
    meta = {"grid": list(grid), "scenario": scenario, "base_node_err": base_node_err,
            "corr_strength": corr_strength, "steps": steps, "trials": trials, "lr": lr,
            "n_train_scenes": n_train_scenes, "train_scene_seed0": train_scene_seed0,
            "base_seed": base_seed, "hidden_dim": hidden_dim, "link_override": link_override}
    return MCFaithfulCheckpoint(model=model, checkpoint_hash=checkpoint_hash(model), model_seed=int(seed),
                                hidden_dim=hidden_dim, k=profile.k, train_meta=meta, history=res["history"])


def save_checkpoint(ckpt: MCFaithfulCheckpoint, path: str) -> str:
    """Persist a checkpoint (state_dict + config + hash + train meta) to ``path``. Returns ``path``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in ckpt.model.state_dict().items()},
                "checkpoint_hash": ckpt.checkpoint_hash, "model_seed": ckpt.model_seed,
                "hidden_dim": ckpt.hidden_dim, "k": ckpt.k, "train_meta": ckpt.train_meta,
                "history": ckpt.history}, path)
    return path


def load_checkpoint(path: str) -> MCFaithfulCheckpoint:
    """Load a checkpoint saved by ``save_checkpoint`` and verify its hash (provenance integrity)."""
    blob = torch.load(path, weights_only=False)
    model = ESDGNN(_esp_config(blob["hidden_dim"], blob["k"])).double()
    model.load_state_dict(blob["state_dict"])
    h = checkpoint_hash(model)
    if h != blob["checkpoint_hash"]:
        raise ValueError(f"checkpoint hash mismatch at {path}: {h} != {blob['checkpoint_hash']}")
    return MCFaithfulCheckpoint(model=model, checkpoint_hash=h, model_seed=blob["model_seed"],
                                hidden_dim=blob["hidden_dim"], k=blob["k"],
                                train_meta=blob["train_meta"], history=blob["history"])


def checkpoint_policy_factory(ckpt: MCFaithfulCheckpoint):
    """A ``scene -> ESDGNNQueryPolicy`` factory bound to a trained checkpoint (for ``evaluate_macro``)."""
    return lambda scene: ESDGNNQueryPolicy(ckpt.model, scene)


def aggregate_seed_macros(macros: list[dict], n_pool_per_seed: int) -> dict:
    """Pool per-seed macro blocks into a multi-seed headline: the across-seed MEAN of each outcome with a
    Wilson CI at the total pooled trial count (``len(macros) * n_pool_per_seed``). Used for the >=5-seed
    headline (constraint: do not publish single-seed headline results)."""
    if not macros:
        raise ValueError("no macros to aggregate")
    n_total = n_pool_per_seed * len(macros)
    out = {}
    for key in ("macro_P_correct", "macro_F_wrong", "macro_F_split", "macro_F_deadline"):
        mean = statistics.mean(m[key] for m in macros)
        out[key] = mean
        out[f"{key}_ci"] = wilson_ci(mean, n_total)
    block = schema.macro_block(out["macro_P_correct"], out["macro_F_wrong"], out["macro_F_split"],
                               out["macro_F_deadline"],
                               ci={k: out[f"{k}_ci"] for k in
                                   ("macro_P_correct", "macro_F_wrong", "macro_F_split", "macro_F_deadline")})
    block["n_seeds"] = len(macros)
    block["per_seed_P_correct"] = [m["macro_P_correct"] for m in macros]
    block["across_seed_sd_P_correct"] = (statistics.stdev([m["macro_P_correct"] for m in macros])
                                         if len(macros) > 1 else 0.0)
    return block
