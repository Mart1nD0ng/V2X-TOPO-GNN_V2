"""G-ESP-MC-FAITHFUL-TRAINING proof-of-concept (workflow §4/§13): does MC-faithful REINFORCE close the
training-signal gap? Trains an ESP/ESD-GNN by the score-function gradient on the MC basin, then compares
the trained model's HELD-OUT dynamic-MC macro_P_correct to its OWN init and to the distance/uniform
heuristics (the judge). If trained > init and approaches/beats distance, the gap is closing (where the
analytic-surrogate budget curve was flat). Bounded/compute-limited.
Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_reinforce_poc.py [--smoke]
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
from src.evaluation.esp_scale import _esp_config, build_scale_instance, evaluate_macro, policy_factory
from src.metrics import manifest as mf
from src.metrics import schema
from src.models import ESDGNN
from src.optimization.mc_reinforce import train_esp_reinforce

SMOKE = "--smoke" in sys.argv
CONFIRM = "--confirm" in sys.argv          # longer, multi-seed confirmation (CI-separate + reach distance?)
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "reinforce_confirm_results.json" if CONFIRM else "reinforce_poc_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
SEEDS_MODEL = [0] if SMOKE else ([0, 1] if CONFIRM else [0])    # confirm: 2 model seeds
STEPS = 3 if SMOKE else (80 if CONFIRM else 40)                 # confirm: longer training (reach distance?)
TRAIN_TRIALS = 20 if SMOKE else 100
LR = 1e-2                                   # per-node credit is low-variance -> a larger step is safe
EVAL_TRIALS = 20 if SMOKE else 200         # tighter held-out CIs to resolve the delta
EVAL_SEEDS = [0] if SMOKE else [0, 1]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(policy_fn):
    return evaluate_macro(GRID, EVAL_SEEDS, policy_fn, PROFILE, PROTO, PHY, trials=EVAL_TRIALS,
                          scenario=SCEN, base_node_err=BE, link_override=None).macro


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor=f"{SCEN}:p={BE}",
                                 scene_descriptor={"gx": GRID[0], "gy": GRID[1], "v": GRID[2]},
                                 query_law="esp", full_physics=True)
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "reinforce_poc", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "scale_protocol": "fixed_protocol", "node_count": 120, "scenario": SCEN, "base_node_err": BE,
           "steps": STEPS, "train_trials": TRAIN_TRIALS, "eval_trials": EVAL_TRIALS,
           "model_seeds": SEEDS_MODEL, "eval_scene_seeds": EVAL_SEEDS,
           "note": ("MC-faithful REINFORCE on the basin reward. The analytic-surrogate budget curve was "
                    "FLAT here (=distance); this tests whether REINFORCE lifts the held-out MC macro_P_correct "
                    "above the GNN's init toward/past distance (gap closing)."),
           "reference": {}, "per_seed": {}}

    # held-out heuristic references (the judge)
    for kind in ("distance", "uniform_esp"):
        out["reference"][kind] = _eval(policy_factory(kind))
        log(f"reference {kind}: Pc={out['reference'][kind]['macro_P_correct']:.3f}")

    from src.models import ESDGNNQueryPolicy
    train_inst = [build_scale_instance(GRID, 1000 + i, scenario=SCEN, base_node_err=BE, corr_strength=CORR)
                  for i in range(2)]
    for s in SEEDS_MODEL:
        torch.manual_seed(s)
        model = ESDGNN(_esp_config(16, PROFILE.k)).double()
        init_macro = _eval(lambda sc: ESDGNNQueryPolicy(model, sc))
        log(f"seed {s}: GNN init held-out Pc={init_macro['macro_P_correct']:.3f}; REINFORCE {STEPS} steps ...")
        ts = time.perf_counter()
        res = train_esp_reinforce(model, train_inst, PROTO, PHY, PROFILE, steps=STEPS,
                                  trials=TRAIN_TRIALS, lr=LR, base_seed=100 * (s + 1))
        trained_macro = _eval(lambda sc: ESDGNNQueryPolicy(model, sc))
        out["per_seed"][str(s)] = {
            "init": init_macro, "trained": trained_macro,
            "train_mc_P_correct_curve": res["history"]["mc_P_correct"],
            "delta_held_out_P_correct": trained_macro["macro_P_correct"] - init_macro["macro_P_correct"]}
        log(f"  seed {s}: trained held-out Pc={trained_macro['macro_P_correct']:.3f} "
            f"(delta {out['per_seed'][str(s)]['delta_held_out_P_correct']:+.3f}) ({time.perf_counter()-ts:.0f}s)")
        json.dump(out, open(OUT, "w"), indent=2)

    deltas = [out["per_seed"][str(s)]["delta_held_out_P_correct"] for s in SEEDS_MODEL]
    out["mean_delta_P_correct"] = sum(deltas) / len(deltas)
    out["gap_closing"] = out["mean_delta_P_correct"] > 0.01
    man = mf.build_manifest(spec, policy_hash="esd_gnn_reinforce", checkpoint_hash="reinforce-poc",
                            model_seeds=SEEDS_MODEL, git_commit=git, manifest_id="EV4-reinforce-poc")
    out["manifest"] = man
    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"DONE {out['runtime_total_s']}s; mean delta Pc={out['mean_delta_P_correct']:+.3f} "
        f"gap_closing={out['gap_closing']}; distance ref={out['reference']['distance']['macro_P_correct']:.3f}")


if __name__ == "__main__":
    main()
