"""Campaign A, Phase A4 -- G-ESP-OOD-GENERALIZATION (declared OOD ablation, one axis at a time).

Evaluates the 5 N=120 shared MC-faithful checkpoints (trained on mm_high(0.35,0.25)) on regimes they were
NOT trained on, vs distance + uniform on the SAME regime, under the dynamic-MC judge. ONE distribution axis
is varied per cell with the other axes pinned to the training values (spec 8.2; the reviewer-caught bug was
that passing scenario='iid' alone silently changes base_err+corr to build defaults -> we hold them).

NOTE (feasibility fix): the plan's cells corr=0.45 and base_err=0.20 are INFEASIBLE for matched_marginal
(constraint |1-2*p_node| <= |1-2*p_pair| => roughly corr <= base_err). Replaced with the feasible-boundary
one-axis equivalents (corr 0.25->0.35; base_err 0.35->0.45 harder, 0.35->0.30 easier), documented here.

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a4_ood.py [--smoke]
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
from src.evaluation.mc_faithful_campaign import (checkpoint_policy_factory, load_checkpoint,
                                                 paired_seed_separation, reliability_status,
                                                 seed_level_bootstrap_ci, ungated_cost)
from src.metrics import manifest as mf
from src.metrics import schema

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
A1_SUMMARY = os.path.join(HERE, "phase_a1_train_summary.json")
OUT = os.path.join(HERE, "phase_a4_ood_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID = (5, 5, 3)
SCENES = [0, 1]
TRIALS = 20 if SMOKE else 150
TRAIN_REGIME = {"scenario": "matched_marginal_high", "base_node_err": 0.35, "corr_strength": 0.25}
# in-distribution baseline = A1 (recorded there); OOD cells vary ONE axis, others pinned to training values
OOD_CELLS = [
    {"id": "ood_covariance_iid", "scenario": "iid", "base_node_err": 0.35, "corr_strength": 0.25,
     "axis": "covariance-family (mm_high -> iid); base_err+corr held"},
    {"id": "ood_corr_harder", "scenario": "matched_marginal_high", "base_node_err": 0.35, "corr_strength": 0.35,
     "axis": "correlation harder (corr 0.25 -> 0.35); base_err held"},
    {"id": "ood_baseerr_harder", "scenario": "matched_marginal_high", "base_node_err": 0.45, "corr_strength": 0.25,
     "axis": "base-error harder (0.35 -> 0.45); corr held"},
    {"id": "ood_baseerr_easier", "scenario": "matched_marginal_high", "base_node_err": 0.30, "corr_strength": 0.25,
     "axis": "base-error easier (0.35 -> 0.30); corr held"},
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(policy_fn, scen, be, corr):
    return evaluate_macro(GRID, SCENES, policy_fn, PROFILE, PROTO, PHY, trials=TRIALS, scenario=scen,
                          base_node_err=be, corr_strength=corr, link_override=None).macro


def main():
    t0 = time.perf_counter()
    a1 = json.load(open(A1_SUMMARY))
    git = mf.current_git_commit() or "uncommitted"
    seeds = sorted(int(s) for s in a1["checkpoints"])
    ckpts = {s: load_checkpoint(os.path.join(HERE, a1["checkpoints"][str(s)]["path"])) for s in seeds}
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "phase_a4_ood", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "node_count": 120, "ablation": "OUT-OF-DISTRIBUTION (declared); one axis varied per cell",
           "train_regime": TRAIN_REGIME, "model_seeds": seeds, "scene_seeds": SCENES, "trials_per_scene": TRIALS,
           "feasibility_note": "plan cells corr=0.45/base_err=0.20 infeasible for matched_marginal -> feasible one-axis equivalents used",
           "cells": {}}
    for cell in OOD_CELLS:
        scen, be, corr = cell["scenario"], cell["base_node_err"], cell["corr_strength"]
        pcs = [_eval(checkpoint_policy_factory(ckpts[s]), scen, be, corr)["macro_P_correct"] for s in seeds]
        boot = seed_level_bootstrap_ci(pcs)
        dist = _eval(policy_factory("distance"), scen, be, corr)
        uni = _eval(policy_factory("uniform_esp"), scen, be, corr)
        sep_uni = paired_seed_separation(pcs, [uni["macro_P_correct"]] * len(pcs))
        out["cells"][cell["id"]] = {
            "axis": cell["axis"], "scenario": scen, "base_node_err": be, "corr_strength": corr,
            "trained_seed_mean_P_correct": boot["mean"], "trained_seed_bootstrap_ci": boot["ci"],
            "trained_across_seed_sd": boot["sd"], "trained_per_seed_P_correct": pcs,
            "distance_P_correct": dist["macro_P_correct"], "distance_ci": dist["macro_P_correct_ci"],
            "uniform_P_correct": uni["macro_P_correct"], "uniform_ci": uni["macro_P_correct_ci"],
            "trained_vs_uniform_paired": sep_uni, "gap_to_distance": dist["macro_P_correct"] - boot["mean"],
            "trained_ungated_cost": 1.0 - boot["mean"], "reliability_status_distance": reliability_status(dist, PROFILE)}
        log(f"{cell['id']}: trained={boot['mean']:.3f} {[round(x,3) for x in boot['ci']]} vs dist={dist['macro_P_correct']:.3f} "
            f"vs uni={uni['macro_P_correct']:.3f}; sep_uni={sep_uni['separated']}")

    spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor="OOD-ablation", scene_descriptor={"gx": 5, "gy": 5, "v": 3},
                                 query_law="esp", full_physics=True)
    out["manifest"] = mf.build_manifest(spec, policy_hash="esd_gnn_mc_faithful_ood", checkpoint_hash="a4-ood",
                                        model_seeds=seeds, git_commit=git, manifest_id="A4-ood")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (result saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; {len(out['cells'])} OOD cells")


if __name__ == "__main__":
    main()
