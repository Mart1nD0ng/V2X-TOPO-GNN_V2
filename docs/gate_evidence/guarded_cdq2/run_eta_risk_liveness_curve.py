"""G-ETA-RISK-LIVENESS evidence (Guarded-CDQ2 round, Phase 4).

Sweeps CDQ2 diversity strength eta in {0,0.25,0.5,1,2,4,8,16} over five evidence-covariance families
x {fixed-link ablation, full-physics headline}, judged by the independent dynamic-MC macrostate basin
first-hitting. For each cell it classifies HOW probability mass moves vs ESP (eta=0): deadline->correct
(benign liveness), deadline->wrong / split-up (validity cost -> must be guarded), split->correct, mixed,
or none. Wrong-risk increases are ALWAYS surfaced (constraint #12). Namespace-clean + hash-bound.

Run: PYTHONPATH=. python docs/gate_evidence/guarded_cdq2/run_eta_risk_liveness_curve.py [--smoke]
"""
from __future__ import annotations

import dataclasses
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
from src.evaluation import esp_scale as es
from src.evaluation import eta_curve as ec
from src.metrics import manifest as mf
from src.metrics import schema

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "eta_risk_liveness_results.json")

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID = (5, 5, 3)            # 120 nodes (the eta curve is a mechanism characterization, not a scale claim)

# (env label, scenario, base_node_err, corr_strength)
ENVS = [
    ("iid", "iid", 0.20, 0.0),                                   # no covariance -> control (no lever)
    ("matched_marginal_low", "matched_marginal_low", 0.35, 0.3),
    ("matched_marginal_high", "matched_marginal_high", 0.35, 0.3),
    ("overlapping", "overlapping_sensor_source", 0.35, 0.3),
    ("split_risk", "overlapping_sensor_source", 0.42, 0.45),     # high-error proxy for balanced split-risk
]
ETA = (0.0, 8.0) if SMOKE else ec.ETA_GRID
SCENE_SEEDS = [0] if SMOKE else [0, 1]
# (physics label, link_override, trials, env subset)
if SMOKE:
    PHYSICS = [("fixed_link", 0.85, 8, ["matched_marginal_high"])]
else:
    PHYSICS = [
        ("fixed_link", 0.85, 80, [e[0] for e in ENVS]),                       # full sweep (shape)
        ("full_physics", None, 60, ["matched_marginal_high", "overlapping"]),  # headline subset
    ]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _spec(scenario, full_physics):
    sc = {"builder": "manhattan", "gx": GRID[0], "gy": GRID[1], "v": GRID[2]}
    return build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor=f"{scenario}", scene_descriptor=sc,
                                 query_law="cdq2", full_physics=full_physics,
                                 allowed_ood_axes=("evidence_covariance",))


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    out = {"metric_namespace_version": "macrostate_v2", "experiment": "eta_risk_liveness",
           "smoke": SMOKE, "git_commit": git, "eta_grid": list(ETA), "scene_seeds": SCENE_SEEDS,
           "N": 120, "diversity": "observable sensor one-hot (C2)", "quality": "distance(0.04)",
           "note": ("eta=0 is exactly ESP. Headline = full-physics macrostate basin first-hitting. "
                    "wrong/split are HARD constraints; a wrong-risk increase is the validity cost that "
                    "Guarded-CDQ2 must gate. The gate identifies HOW mass moves, not whether CDQ2 wins."),
           "cells": {}}

    def flush():
        json.dump(out, open(OUT, "w"), indent=2)

    env_map = {e[0]: e for e in ENVS}
    for phys_label, link, trials, env_subset in PHYSICS:
        for env_label in env_subset:
            _, scenario, base_err, corr = env_map[env_label]
            key = f"{phys_label}/{env_label}"
            log(f"sweep {key} (eta={list(ETA)}, {trials} trials x {len(SCENE_SEEDS)} scenes) ...")
            ts = time.perf_counter()
            sweep = ec.eta_sweep(GRID, SCENE_SEEDS, scenario=scenario, base_node_err=base_err,
                                 corr_strength=corr, profile=PROFILE, proto=PROTO, phy=PHY,
                                 trials=trials, link_override=link, eta_grid=ETA)
            esp_macro = sweep[0.0]
            man = mf.build_manifest(_spec(scenario, link is None), policy_hash="cdq2_distance_sensorZ",
                                    checkpoint_hash="fixed-quality+observable-Z", model_seeds=SCENE_SEEDS,
                                    git_commit=git, manifest_id="GS4-eta-curve")
            etas = {}
            for eta, macro in sweep.items():
                shift = ec.classify_mass_shift(esp_macro, macro)
                etas[f"{eta:g}"] = {"macro": macro, "mass_shift": dataclasses.asdict(shift)}
            # dominant shift = the classification at the largest eta (strongest diversity)
            dominant = ec.classify_mass_shift(esp_macro, sweep[max(ETA)])
            out["cells"][key] = {
                "physics": phys_label, "env": env_label, "scenario": scenario,
                "base_node_err": base_err, "hashes": man, "etas": etas,
                "dominant_shift_at_max_eta": dataclasses.asdict(dominant),
                "wrong_increased_any": any(etas[k]["mass_shift"]["wrong_increased"] for k in etas)}
            log(f"  {key}: ESP Pc={esp_macro['macro_P_correct']:.3f}; "
                f"max-eta shift={dominant.label} (dPc={dominant.d_P_correct:+.3f} "
                f"dFw={dominant.d_F_wrong:+.3f} dFd={dominant.d_F_deadline:+.3f}) ({time.perf_counter()-ts:.0f}s)")
            flush()

    # validate every macro block + ban-list scan
    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    for key, cell in out["cells"].items():
        for eta, e in cell["etas"].items():
            schema.validate_macro_block(e["macro"])
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    flush()
    log(f"DONE in {out['runtime_total_s']}s -> {OUT}")


if __name__ == "__main__":
    main()
