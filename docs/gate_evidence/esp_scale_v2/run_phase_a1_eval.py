"""Campaign A, Phase A1 -- EVAL (headline CI-separation). Loads the 5 step-150 MC-faithful checkpoints
(run_phase_a1_train.py) and evaluates each under the dynamic-MC judge (full physics) at N=120, pooled over
held-out scene seeds, vs distance + uniform references. Applies the A0 pre-registration:
  * performance = UN-GATED J = 1 - macro_P_correct (mc_faithful_campaign.ungated_cost);
  * headline CI = seed-level bootstrap (each model seed = one observation); pooled-binomial is diagnostic;
  * reliability (F_wrong/F_split UCB vs eps) reported SEPARATELY (never in the performance comparison).
Verdict: is the 5-seed trained macro_P_correct CI-separated above uniform (conservatively, trained
seed-level lower bound > uniform MC upper bound) and how close to distance (0.427)?

Run (after training): PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a1_eval.py [--smoke]
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.config.experiment_spec import build_experiment_spec
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.esp_scale import evaluate_macro, policy_factory
from src.evaluation.mc_faithful_campaign import (aggregate_seed_macros, checkpoint_policy_factory,
                                                 load_checkpoint, paired_seed_separation,
                                                 reliability_status, seed_level_bootstrap_ci, ungated_cost)
from src.metrics import manifest as mf
from src.metrics import schema

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
TRAIN_SUMMARY = os.path.join(HERE, "phase_a1_train_summary.json")
OUT = os.path.join(HERE, "phase_a1_eval_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
EVAL_SCENES = [0, 1] if SMOKE else [0, 1, 2]
TRAINED_TRIALS = 20 if SMOKE else 300        # per held-out scene, per model seed
REF_TRIALS = 30 if SMOKE else 1000           # refs evaluated once -> bump trials to tighten their MC CI


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(policy_fn, trials):
    return evaluate_macro(GRID, EVAL_SCENES, policy_fn, PROFILE, PROTO, PHY, trials=trials, scenario=SCEN,
                          base_node_err=BE, corr_strength=CORR, link_override=None).macro


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
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "phase_a1_eval_headline", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "node_count": 120, "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "eval_scene_seeds": EVAL_SCENES, "trained_trials": TRAINED_TRIALS, "ref_trials": REF_TRIALS,
           "pre_registration": "A0: ungated J=1-P_correct; seed-level bootstrap headline; reliability separate",
           "reference": {}, "per_seed": {}}

    # references (evaluated once, higher trials for tighter MC CI), CRN-shared scenes with the trained evals
    for kind in ("distance", "uniform_esp"):
        m = _eval(policy_factory(kind), REF_TRIALS)
        out["reference"][kind] = {"macro": m, "ungated_cost": ungated_cost(m),
                                  "reliability": reliability_status(m, PROFILE)}
        log(f"ref {kind}: Pc={m['macro_P_correct']:.3f} ci={m['macro_P_correct_ci']}")

    seeds = sorted(int(s) for s in tsum["checkpoints"])
    trained_macros, trained_pc = [], []
    for s in seeds:
        path = os.path.join(HERE, tsum["checkpoints"][str(s)]["path"])
        ckpt = load_checkpoint(path)                                    # hash-verified
        m = _eval(checkpoint_policy_factory(ckpt), TRAINED_TRIALS)
        trained_macros.append(m)
        trained_pc.append(m["macro_P_correct"])
        out["per_seed"][str(s)] = {"checkpoint_hash": ckpt.checkpoint_hash, "macro": m,
                                   "ungated_cost": ungated_cost(m), "reliability": reliability_status(m, PROFILE)}
        log(f"seed {s}: trained Pc={m['macro_P_correct']:.3f} ({ckpt.checkpoint_hash[:12]})")

    # ---- A0 headline statistics ----
    boot = seed_level_bootstrap_ci(trained_pc)
    uni = out["reference"]["uniform_esp"]["macro"]
    dist = out["reference"]["distance"]["macro"]
    sep_vs_uniform = paired_seed_separation(trained_pc, [uni["macro_P_correct"]] * len(trained_pc))
    out["headline"] = {
        "trained_seed_mean_P_correct": boot["mean"], "trained_seed_bootstrap_ci": boot["ci"],
        "trained_across_seed_sd": boot["sd"], "n_model_seeds": len(seeds),
        "uniform_P_correct": uni["macro_P_correct"], "uniform_ci": uni["macro_P_correct_ci"],
        "distance_P_correct": dist["macro_P_correct"], "distance_ci": dist["macro_P_correct_ci"],
        # conservative separation: trained seed-level lower bound > uniform MC upper bound
        "separated_above_uniform_conservative": boot["ci"][0] > uni["macro_P_correct_ci"][1],
        "paired_diff_vs_uniform": sep_vs_uniform,
        "gap_to_distance": dist["macro_P_correct"] - boot["mean"],
        "frac_of_uniform_to_distance_gap_closed":
            (boot["mean"] - uni["macro_P_correct"]) / (dist["macro_P_correct"] - uni["macro_P_correct"]),
        "diagnostic_pooled": aggregate_seed_macros(trained_macros, TRAINED_TRIALS * len(EVAL_SCENES)),
    }
    man = mf.build_manifest(spec, policy_hash="esd_gnn_mc_faithful", checkpoint_hash="a1-5seed",
                            model_seeds=seeds, git_commit=git, manifest_id="A1-headline")
    out["manifest"] = man
    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    h = out["headline"]
    log(f"DONE {out['runtime_total_s']}s; trained {h['trained_seed_mean_P_correct']:.3f} "
        f"ci={[round(x,3) for x in h['trained_seed_bootstrap_ci']]} vs uniform {h['uniform_P_correct']:.3f} "
        f"vs distance {h['distance_P_correct']:.3f}; sep_above_uniform="
        f"{h['separated_above_uniform_conservative']}; gap_to_distance={h['gap_to_distance']:+.3f}")


if __name__ == "__main__":
    main()
