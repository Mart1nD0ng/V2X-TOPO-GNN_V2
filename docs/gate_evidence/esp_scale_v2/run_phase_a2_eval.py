"""Campaign A, Phase A2 -- G-ESP-TRAINING-BUDGET under the MC-faithful trainer.

Evaluates each model seed's budget snapshots {0(init), 40, 80, 150} (from run_phase_a1_train.py, along ONE
trajectory) under the dynamic-MC judge, and shows the held-out budget curve RISES -- in DIRECT contrast to
the EV1 FLAT analytic-surrogate budget curve (0.422 at every budget). Budget 0 = the untrained ESDGNN init
(reproduced deterministically per seed). A0 pre-registration: un-gated J, seed-level bootstrap per budget.

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a2_eval.py [--smoke]
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.config.experiment_spec import build_experiment_spec
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.esp_scale import _esp_config, evaluate_macro
from src.evaluation.mc_faithful_campaign import (checkpoint_policy_factory, load_checkpoint,
                                                 seed_level_bootstrap_ci, ungated_cost)
from src.metrics import manifest as mf
from src.metrics import schema
from src.models import ESDGNN, ESDGNNQueryPolicy

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
TRAIN_SUMMARY = os.path.join(HERE, "phase_a1_train_summary.json")
OUT = os.path.join(HERE, "phase_a2_budget_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR, HIDDEN = (5, 5, 3), "matched_marginal_high", 0.35, 0.25, 16
EVAL_SCENES = [0, 1] if SMOKE else [0, 1]
TRIALS = 20 if SMOKE else 250
BUDGETS = [0, 40, 80, 150]
EV1_FLAT_ANALYTIC = 0.422        # EV1 analytic-surrogate budget curve was FLAT at this value (contrast ONLY)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(policy_fn):
    return evaluate_macro(GRID, EVAL_SCENES, policy_fn, PROFILE, PROTO, PHY, trials=TRIALS, scenario=SCEN,
                          base_node_err=BE, corr_strength=CORR, link_override=None).macro


def _init_policy_factory(seed):
    torch.manual_seed(int(seed))                              # reproduce the train_mc_faithful init exactly
    model = ESDGNN(_esp_config(HIDDEN, PROFILE.k)).double()
    return lambda scene: ESDGNNQueryPolicy(model, scene)


def main():
    t0 = time.perf_counter()
    if not os.path.exists(TRAIN_SUMMARY):
        raise SystemExit(f"missing {TRAIN_SUMMARY}; run run_phase_a1_train.py first")
    tsum = json.load(open(TRAIN_SUMMARY))
    git = mf.current_git_commit() or "uncommitted"
    spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}",
                                 scene_descriptor={"gx": GRID[0], "gy": GRID[1], "v": GRID[2]},
                                 query_law="esp", full_physics=True)
    seeds = sorted(int(s) for s in tsum["checkpoints"])
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "phase_a2_budget", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "node_count": 120, "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "eval_scene_seeds": EVAL_SCENES, "trials_per_cell": TRIALS, "budgets": BUDGETS,
           "model_seeds": seeds, "ev1_flat_analytic_contrast": EV1_FLAT_ANALYTIC,
           "pre_registration": "A0: ungated J; seed-level bootstrap per budget; EV1 0.422 cited as CONTRAST only",
           "budget_curve": {}}

    for b in BUDGETS:
        per_seed = {}
        pcs = []
        for s in seeds:
            if b == 0:
                m = _eval(_init_policy_factory(s))
                chash = "init"
            else:
                path = os.path.join(HERE, tsum["checkpoints"][str(s)]["budget_checkpoints"][str(b)]["path"])
                ckpt = load_checkpoint(path)
                m = _eval(checkpoint_policy_factory(ckpt))
                chash = ckpt.checkpoint_hash
            per_seed[str(s)] = {"macro_P_correct": m["macro_P_correct"], "ungated_cost": ungated_cost(m),
                                "checkpoint_hash": chash}
            pcs.append(m["macro_P_correct"])
        boot = seed_level_bootstrap_ci(pcs)
        out["budget_curve"][str(b)] = {"seed_mean_P_correct": boot["mean"], "seed_bootstrap_ci": boot["ci"],
                                       "across_seed_sd": boot["sd"], "per_seed_P_correct": pcs,
                                       "per_seed": per_seed}
        log(f"budget {b}: seed-mean Pc={boot['mean']:.3f} ci={[round(x,3) for x in boot['ci']]} "
            f"per-seed={[round(x,3) for x in pcs]}")

    curve = [out["budget_curve"][str(b)]["seed_mean_P_correct"] for b in BUDGETS]
    out["monotone_nondecreasing"] = all(curve[i + 1] >= curve[i] - 0.005 for i in range(len(curve) - 1))
    out["rises_from_init"] = curve[-1] - curve[0]
    out["beats_ev1_flat"] = curve[-1] > EV1_FLAT_ANALYTIC - 0.05   # MC-faithful trains where analytic was flat
    man = mf.build_manifest(spec, policy_hash="esd_gnn_mc_faithful_budget", checkpoint_hash="a2-budget",
                            model_seeds=seeds, git_commit=git, manifest_id="A2-budget")
    out["manifest"] = man
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (result saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; budget curve {[round(c,3) for c in curve]} "
        f"rises_from_init={out['rises_from_init']:+.3f} monotone={out['monotone_nondecreasing']} "
        f"(EV1 analytic was FLAT at {EV1_FLAT_ANALYTIC})")


if __name__ == "__main__":
    main()
