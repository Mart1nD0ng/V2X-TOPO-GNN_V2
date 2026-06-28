"""Campaign A, Phase A3a -- scale-specific EXPERTS (the scale_regret denominator).

Trains a MC-faithful ESP/ESD-GNN expert AT each affordable larger scale, UNDER that scale's
pre-registered fixed-service deadline R_d(N)=round(6*sqrt(N/120)). These experts are the J_expert term in
scale_regret(N) = J_shared(N) - J_expert(N): a shared checkpoint trained at N=120 is compared against a
policy that got to TRAIN at N. Bounded: 1 seed x 80 steps each (EXPLICITLY compute-limited vs the >=5-seed
headline). N=120 expert = the A1 5-seed checkpoints (free; reused in A3b).

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a3a_experts.py [--smoke]
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.esp_scale import calibrated_profile
from src.evaluation.mc_faithful_campaign import load_checkpoint, save_checkpoint, train_mc_faithful
from src.metrics import manifest as mf

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(HERE, "checkpoints")
SUMMARY = os.path.join(HERE, "phase_a3a_experts_summary.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
BASE_PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
SCEN, BE, CORR, HIDDEN = "matched_marginal_high", 0.35, 0.25, 16
STEPS = 3 if SMOKE else 80
TRAIN_TRIALS = 20 if SMOKE else 100
# (grid, target_N): the affordable larger scales to train experts at (N=120 expert = A1 checkpoints, free)
EXPERT_SCALES = [((8, 8, 3), 336)] if SMOKE else [((8, 8, 3), 336), ((11, 11, 3), 660)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    summary = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
               "phase": "A3a_experts", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
               "compute_limited": "1 seed x 80 steps per expert (vs >=5-seed headline) -- workflow 9.1",
               "calibration": "fixed_service R_d(N)=round(6*sqrt(N/120)) (pre-registered)", "experts": {}}
    for grid, target_N in EXPERT_SCALES:
        prof = calibrated_profile(BASE_PROFILE, target_N, mode="fixed_service_profile")
        path = os.path.join(CKPT_DIR, f"expert_N{target_N}_seed0_steps{STEPS}.pt")
        if os.path.exists(path):
            try:
                c = load_checkpoint(path)
                summary["experts"][str(target_N)] = {"path": os.path.relpath(path, HERE),
                                                      "checkpoint_hash": c.checkpoint_hash, "grid": list(grid),
                                                      "R_d": prof.max_poll_epochs, "resumed": True}
                log(f"N={target_N}: existing expert, skip ({c.checkpoint_hash[:12]})")
                json.dump(summary, open(SUMMARY, "w"), indent=2)
                continue
            except Exception as e:  # noqa: BLE001
                log(f"N={target_N}: existing expert invalid ({e}); retraining")
        ts = time.perf_counter()
        ckpt = train_mc_faithful(0, grid, profile=prof, proto=PROTO, phy=PHY, scenario=SCEN,
                                 base_node_err=BE, corr_strength=CORR, steps=STEPS, trials=TRAIN_TRIALS,
                                 hidden_dim=HIDDEN, base_seed=777)
        save_checkpoint(ckpt, path)
        curve = ckpt.history["mc_P_correct"]
        summary["experts"][str(target_N)] = {"path": os.path.relpath(path, HERE),
                                              "checkpoint_hash": ckpt.checkpoint_hash, "grid": list(grid),
                                              "R_d": prof.max_poll_epochs,
                                              "train_mc_P_correct": [round(curve[0], 3), round(curve[-1], 3)],
                                              "resumed": False}
        json.dump(summary, open(SUMMARY, "w"), indent=2)
        log(f"N={target_N} (grid {grid}, R_d={prof.max_poll_epochs}): expert trained {STEPS} steps, "
            f"{ckpt.checkpoint_hash[:12]} (train {curve[0]:.2f}->{curve[-1]:.2f}) ({time.perf_counter()-ts:.0f}s)")
    summary["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    log(f"DONE {summary['runtime_total_s']}s; {len(summary['experts'])} experts in {CKPT_DIR}")


if __name__ == "__main__":
    main()
