"""Campaign A, Phase A3b -- the SCALE SWEEP (G-ESP-FIXED-PROTOCOL-SCALE + G-ESP-FIXED-SERVICE-SCALE).

Evaluates the N=120-trained SHARED MC-faithful checkpoints (A1, 5 seeds) across the scale ladder under BOTH
calibration modes vs distance, uniform, and the scale-specific EXPERTS (A3a), judged by the dynamic MC.
Performance = UN-GATED J = 1 - macro_P_correct (A0); reliability (F_wrong/F_split UCB) reported SEPARATELY;
scale_regret(N) = J_shared - J_expert where an expert exists. Pre-registered compute-limited taper (every
reduced cell is labeled; N=9840 is an APPROXIMATION BOUND, never a validated N=10000 claim -- constraint #8).
Resumable: a completed N-block in the output JSON is skipped.

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a3b_scale.py [--smoke]
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
from src.evaluation.esp_scale import (calibrated_profile, evaluate_macro, grid_for_target_N, policy_factory)
from src.evaluation.mc_faithful_campaign import (checkpoint_policy_factory, load_checkpoint,
                                                 reliability_status, seed_level_bootstrap_ci, ungated_cost)
from src.metrics import manifest as mf
from src.metrics import schema

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
A1_SUMMARY = os.path.join(HERE, "phase_a1_train_summary.json")
A3A_SUMMARY = os.path.join(HERE, "phase_a3a_experts_summary.json")
OUT = os.path.join(HERE, "phase_a3b_scale_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
BASE_PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
SCEN, BE, CORR = "matched_marginal_high", 0.35, 0.25
SCENES = [0, 1]
BOTH = ["fixed_protocol", "fixed_service_profile"]
FP = ["fixed_protocol"]
# (target_N, trials/scene, shared_seeds, modes, has_expert, policies) -- the pre-registered taper (A3b plan)
LADDER = [
    (120, 10, 2, FP, False, ("shared", "distance", "uniform_esp")),
    (336, 8, 2, BOTH, True, ("shared", "distance", "uniform_esp", "expert")),
] if SMOKE else [
    (120, 200, 5, FP, False, ("shared", "distance", "uniform_esp")),           # modes identical at N=120
    (336, 150, 5, BOTH, True, ("shared", "distance", "uniform_esp", "expert")),  # HEADLINE scale
    (660, 100, 3, BOTH, True, ("shared", "distance", "uniform_esp", "expert")),  # compute-limited
    (1248, 40, 3, BOTH, False, ("shared", "distance", "uniform_esp")),           # compute-limited
    (3036, 16, 2, FP, False, ("shared", "distance", "uniform_esp")),             # compute-limited, FP only
    (9840, 8, 2, FP, False, ("shared", "distance")),                             # APPROXIMATION BOUND, FP only
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(grid, policy_fn, prof, trials):
    return evaluate_macro(grid, SCENES, policy_fn, prof, PROTO, PHY, trials=trials, scenario=SCEN,
                          base_node_err=BE, corr_strength=CORR, link_override=None).macro


def main():
    t0 = time.perf_counter()
    a1 = json.load(open(A1_SUMMARY))
    experts = json.load(open(A3A_SUMMARY))["experts"] if os.path.exists(A3A_SUMMARY) else {}
    git = mf.current_git_commit() or "uncommitted"
    if os.path.exists(OUT):
        out = json.load(open(OUT))                                   # resume
    else:
        out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
               "experiment": "phase_a3b_scale", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
               "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR, "scene_seeds": SCENES,
               "calibration_rule": "fixed_service R_d(N)=round(6*sqrt(N/120)) (pre-registered)",
               "pre_registration": "A0: ungated J; seed-level bootstrap; reliability separate; taper labeled",
               "cells": {}}
    all_seeds = sorted(int(s) for s in a1["checkpoints"])

    for target_N, trials, n_seeds, modes, has_expert, policies in LADDER:
        key_N = str(target_N)
        if key_N in out["cells"]:
            log(f"N={target_N}: already done, skip")
            continue
        grid = grid_for_target_N(target_N)
        seeds = all_seeds[:n_seeds]
        compute_limited = (n_seeds < 5) or (modes == FP and target_N >= 336) or (target_N >= 3036)
        block = {"grid": list(grid), "modes": {}, "shared_model_seeds": seeds,
                 "compute_limited": compute_limited,
                 "approximation_bound": target_N >= 9840, "trials_per_scene": trials}
        ts = time.perf_counter()
        for mode in modes:
            prof = calibrated_profile(BASE_PROFILE, target_N, mode=mode)
            cell = {"R_d": prof.max_poll_epochs, "policies": {}}
            # shared ESP (5/3/2 seeds) -> seed-level bootstrap
            if "shared" in policies:
                pcs, rel = [], []
                for s in seeds:
                    ck = load_checkpoint(os.path.join(HERE, a1["checkpoints"][str(s)]["path"]))
                    m = _eval(grid, checkpoint_policy_factory(ck), prof, trials)
                    pcs.append(m["macro_P_correct"]); rel.append(reliability_status(m, prof))
                boot = seed_level_bootstrap_ci(pcs)
                cell["policies"]["shared_esp"] = {
                    "seed_mean_P_correct": boot["mean"], "seed_bootstrap_ci": boot["ci"],
                    "across_seed_sd": boot["sd"], "per_seed_P_correct": pcs,
                    "ungated_cost": 1.0 - boot["mean"], "n_seeds": len(seeds),
                    "reliability_status_seed0": rel[0]}
            for kind in ("distance", "uniform_esp"):
                if kind in policies:
                    m = _eval(grid, policy_factory(kind), prof, trials)
                    cell["policies"][kind] = {"macro_P_correct": m["macro_P_correct"],
                                              "ungated_cost": ungated_cost(m),
                                              "reliability_status": reliability_status(m, prof)}
            if "expert" in policies and has_expert and mode == "fixed_service_profile" and key_N in experts:
                ek = load_checkpoint(os.path.join(HERE, experts[key_N]["path"]))
                m = _eval(grid, checkpoint_policy_factory(ek), prof, trials)
                cell["policies"]["expert"] = {"macro_P_correct": m["macro_P_correct"],
                                              "ungated_cost": ungated_cost(m),
                                              "checkpoint_hash": ek.checkpoint_hash,
                                              "reliability_status": reliability_status(m, prof)}
                cell["scale_regret_shared_vs_expert"] = (cell["policies"]["shared_esp"]["ungated_cost"]
                                                         - cell["policies"]["expert"]["ungated_cost"])
            block["modes"][mode] = cell
        out["cells"][key_N] = block
        json.dump(out, open(OUT, "w"), indent=2)                     # save per-N (resumable)
        sh = block["modes"][modes[0]]["policies"].get("shared_esp", {})
        log(f"N={target_N} (grid {grid}) modes={modes}: shared Pc={sh.get('seed_mean_P_correct', float('nan')):.3f} "
            f"({time.perf_counter()-ts:.0f}s)")

    spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=BASE_PROFILE, phy_cfg=PHY,
                                 evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}", scene_descriptor={"ladder": "120-9840"},
                                 query_law="esp", full_physics=True)
    out["manifest"] = mf.build_manifest(spec, policy_hash="esd_gnn_mc_faithful_scale",
                                        checkpoint_hash="a3b-scale", model_seeds=all_seeds, git_commit=git,
                                        manifest_id="A3b-scale")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (result saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; {len(out['cells'])} scales in {OUT}")


if __name__ == "__main__":
    main()
