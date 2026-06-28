"""G-ESP-TRAINING-BUDGET evidence (esp_performance_scale_v2, workflow §4).

Trains ESP/ESD-GNN full-physics budget curves (pilot/medium/full) and records the held-out validation
macro (GNN vs the distance heuristic) at each budget checkpoint, to answer: does longer training
materially improve macrostate performance, and does the GNN beat the distance heuristic (learning
headroom) or only match it (§13.2)? Namespace-clean + hash-bound. N=120, single-scale + iid this
pass (BOUNDED / compute-limited: 3 model seeds; the workflow's >=5 seeds + mixed N{100,300,1000} +
structured regimes are follow-up passes, labeled here).
Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_training_budget.py [--smoke]
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
from src.evaluation import esp_training as et
from src.metrics import manifest as mf
from src.metrics import schema

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "training_budget_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID = (5, 5, 3)
# a HARD regime with room to learn: covariance (correlated structure) + deadline pressure -> analytic
# Pc < 1 (iid @N=120 is near-ceiling, no measurable headroom -- workflow §5.3). corr <= base_err.
SCENARIO, BASE_ERR, CORR = "matched_marginal_high", 0.35, 0.25
SEEDS = [0] if SMOKE else [0, 1, 2]
TOTAL_STEPS = 3 if SMOKE else 50
BUDGETS = [1, 3] if SMOKE else [5, 15, 30, 50]
VAL_SEEDS = [0] if SMOKE else [0, 1, 2, 3]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor=f"{SCENARIO}:p={BASE_ERR}",
                                 scene_descriptor={"gx": GRID[0], "gy": GRID[1], "v": GRID[2]},
                                 query_law="esp", full_physics=True)
    man = mf.build_manifest(spec, policy_hash="esd_gnn_esp_training", checkpoint_hash="budget-curve",
                            model_seeds=SEEDS, git_commit=git, manifest_id="EV1-training-budget")
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "training_budget", "smoke": SMOKE, "query_family": "ESP",
           "scale_protocol": "fixed_protocol", "node_count": 120, "training_node_counts": [120],
           "scenario": SCENARIO, "base_node_err": BASE_ERR, "total_steps": TOTAL_STEPS,
           "budget_points": BUDGETS, "model_seeds": SEEDS, "manifest": man,
           "compute_limited_note": ("BOUNDED: 3 model seeds (headline target >=5), single training scale "
                                    "N=120, iid regime, full physics. Validation macro is the ANALYTIC "
                                    "screen (workflow §1.3); the dynamic MC is the final judge (later gates). "
                                    "Mixed-scale N{100,300,1000} + structured regimes are follow-up passes."),
           "per_seed": {}}

    curves = []
    for s in SEEDS:
        log(f"training seed {s}: {TOTAL_STEPS} full-physics steps @N=120, budgets={BUDGETS} ...")
        ts = time.perf_counter()
        c = et.train_with_curve([GRID], seed=s, profile=PROFILE, proto=PROTO, phy=PHY,
                                total_steps=TOTAL_STEPS, budget_points=BUDGETS, val_grids=[GRID],
                                val_seeds=VAL_SEEDS, scenario=SCENARIO, base_node_err=BASE_ERR,
                                corr_strength=CORR, hidden_dim=16, link_override=None)
        curves.append(c)
        out["per_seed"][str(s)] = {
            "checkpoint_hash": c.checkpoint_hash,
            "curve": {str(b): {"gnn": c.curve[b]["gnn"], "distance": c.curve[b]["distance"]}
                      for b in c.curve},
            "final_loss": c.loss_history[-1], "loss_history": c.loss_history}
        last = c.curve[BUDGETS[-1]]
        log(f"  seed {s}: GNN Pc@full={last['gnn']['macro_P_correct']:.3f} "
            f"distance Pc={last['distance']['macro_P_correct']:.3f} ({time.perf_counter()-ts:.0f}s)")
        json.dump(out, open(OUT, "w"), indent=2)

    out["budget_summary"] = et.budget_improvement(curves)
    out["best_budget"] = et.select_best_budget(curves)
    # interpretation (workflow §13): does the GNN beat / match / lose to distance after full training?
    bs = out["budget_summary"]
    if bs["beats_distance_at_full"]:
        interp = "§13.1/§13.2 in-regime: GNN BEATS distance after full training -> learning headroom exists."
    elif abs(bs["gnn_P_correct_by_budget"][bs["full"]]
             - bs["distance_P_correct_by_budget"][bs["full"]]) <= 0.02:
        interp = "§13.2: GNN MATCHES distance (stable learned constructor, not superior in iid)."
    else:
        interp = "§13.3: GNN LOSES to distance in iid -> low topology-learning headroom in this regime."
    out["interpretation"] = interp
    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"DONE {out['runtime_total_s']}s; improves_over_pilot={bs['improves_over_pilot']} "
        f"beats_distance={bs['beats_distance_at_full']} best_budget={out['best_budget']}; {interp}")


if __name__ == "__main__":
    main()
