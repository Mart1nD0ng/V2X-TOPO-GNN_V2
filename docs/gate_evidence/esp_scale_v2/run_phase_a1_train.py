"""Campaign A, Phase A1+A2 -- TRAIN: 5 MC-faithful ESP/ESD-GNN checkpoints with budget snapshots.

Trains each model seed by per-node-credit MC-faithful REINFORCE (EV4/EV5) on mm_high(0.35,0.25) R_d=6
N=120 for STEPS steps, saving state_dicts at SNAPSHOT_STEPS along the SAME trajectory (trajectory-preserving),
so ONE run yields BOTH the A1 headline checkpoints (step 150) AND the A2 budget axis (steps 40/80). Each
seed's checkpoints are saved as soon as it finishes (resumable: a hash-valid step-150 checkpoint is skipped).
Eval is separate (run_phase_a1_eval.py / run_phase_a2_eval.py), so training compute is done once and reused
across CI-separation + budget + scale + OOD.

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a1_train.py [--smoke]
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

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.mc_faithful_campaign import (load_checkpoint, materialize_snapshot, save_checkpoint,
                                                 train_mc_faithful)
from src.metrics import manifest as mf

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(HERE, "checkpoints")
SUMMARY = os.path.join(HERE, "phase_a1_train_summary.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
STEPS = 3 if SMOKE else 150
SNAPSHOT_STEPS = (1, 2) if SMOKE else (40, 80)      # A2 budget axis along the same trajectory
TRAIN_TRIALS = 20 if SMOKE else 100
HIDDEN = 16


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    summary = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
               "phase": "A1_train", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
               "regime": {"grid": list(GRID), "node_count": 120, "scenario": SCEN, "base_node_err": BE,
                          "corr_strength": CORR}, "steps": STEPS, "train_trials": TRAIN_TRIALS,
               "hidden_dim": HIDDEN, "model_seeds": SEEDS, "checkpoints": {}}
    for s in SEEDS:
        path = os.path.join(CKPT_DIR, f"mcf_seed{s}_steps{STEPS}.pt")
        if os.path.exists(path):
            try:
                c = load_checkpoint(path)
                summary["checkpoints"][str(s)] = {"path": os.path.relpath(path, HERE),
                                                  "checkpoint_hash": c.checkpoint_hash,
                                                  "final_train_mc_P_correct": c.history["mc_P_correct"][-1],
                                                  "resumed": True}
                log(f"seed {s}: existing valid checkpoint, skip ({c.checkpoint_hash[:12]})")
                json.dump(summary, open(SUMMARY, "w"), indent=2)
                continue
            except Exception as e:  # noqa: BLE001 -- corrupt/partial checkpoint, retrain
                log(f"seed {s}: existing checkpoint invalid ({e}); retraining")
        ts = time.perf_counter()
        ckpt = train_mc_faithful(s, GRID, profile=PROFILE, proto=PROTO, phy=PHY, scenario=SCEN,
                                 base_node_err=BE, corr_strength=CORR, steps=STEPS, trials=TRAIN_TRIALS,
                                 hidden_dim=HIDDEN, base_seed=100 * (s + 1), snapshot_steps=SNAPSHOT_STEPS)
        save_checkpoint(ckpt, path)
        curve = ckpt.history["mc_P_correct"]
        budgets = {STEPS: {"path": os.path.relpath(path, HERE), "checkpoint_hash": ckpt.checkpoint_hash}}
        for b in SNAPSHOT_STEPS:                         # A2 budget axis: save each snapshot as a checkpoint
            snap = materialize_snapshot(ckpt, b)
            bpath = os.path.join(CKPT_DIR, f"mcf_seed{s}_steps{b}.pt")
            save_checkpoint(snap, bpath)
            budgets[b] = {"path": os.path.relpath(bpath, HERE), "checkpoint_hash": snap.checkpoint_hash}
        summary["checkpoints"][str(s)] = {"path": os.path.relpath(path, HERE),
                                          "checkpoint_hash": ckpt.checkpoint_hash,
                                          "final_train_mc_P_correct": curve[-1],
                                          "train_mc_P_correct_first": curve[0],
                                          "budget_checkpoints": {str(k): v for k, v in budgets.items()},
                                          "resumed": False}
        json.dump(summary, open(SUMMARY, "w"), indent=2)
        log(f"seed {s}: trained {STEPS} steps (+snap {SNAPSHOT_STEPS}), saved {ckpt.checkpoint_hash[:12]} "
            f"(train basin {curve[0]:.2f}->{curve[-1]:.2f}) ({time.perf_counter()-ts:.0f}s)")
    summary["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    log(f"DONE {summary['runtime_total_s']}s; {len(summary['checkpoints'])} checkpoints in {CKPT_DIR}")


if __name__ == "__main__":
    main()
