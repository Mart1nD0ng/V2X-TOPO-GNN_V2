"""G-ESP-BASELINE-ORACLE headroom (the JUDGE level): does the independent dynamic MC see ESP
peer-selection headroom that the analytic TRAINING surrogate is blind to?

The training surrogate (analytic macrostate basin) is ~peer-insensitive across regimes (heuristic spread
~0; GNN training curve flat). This script measures the SAME 5 heuristics under the dynamic-MC judge.
- If MC spread ~0 too  -> the macro basin is genuinely peer-invariant (environment-dominated); a GNN that
  matches heuristics is expected, not a failure (workflow §13.2 / §5.3).
- If MC spread is meaningful -> the judge rewards peer selection the analytic surrogate cannot see -> a
  TRAINING-SIGNAL gap (the GNN can't learn what the judge wants from this surrogate) -> §13.3.
Bounded/compute-limited. Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_headroom_mc.py [--smoke]
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.config.experiment_spec import build_experiment_spec
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.cdq2_factorial import wilson_ci
from src.evaluation.esp_baselines import DEPLOYABLE_BASELINES, make_baseline
from src.evaluation.esp_scale import build_scale_instance
from src.metrics import manifest as mf
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.validation import run_dynamic_mc

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "headroom_mc_results.json")
SCENE_SEEDS = [0] if SMOKE else [0, 1]
TRIALS = 20 if SMOKE else 100
# (label, grid, scenario, base_err, corr, R_d)
REGIMES = [("iid_easy", (5, 5, 3), "iid", 0.20, 0.30, 6),
           ("mm_high_R6", (5, 5, 3), "matched_marginal_high", 0.35, 0.25, 6)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "headroom_mc", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "scene_seeds": SCENE_SEEDS, "dynamic_mc_trials": TRIALS, "node_count": 120,
           "note": ("dynamic-MC (the judge) spread of 5 observable ESP heuristics. ~0 spread => macro "
                    "basin is peer-invariant (env-dominated, §13.2); meaningful spread => the judge sees "
                    "headroom the analytic TRAINING surrogate is blind to (training-signal gap, §13.3)."),
           "regimes": {}}
    for label, grid, scen, be, corr, rd in REGIMES:
        phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
        proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=rd)
        prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=rd)
        insts = [(build_scale_instance(grid, s, scenario=scen, base_node_err=be, corr_strength=corr), s)
                 for s in SCENE_SEEDS]
        log(f"=== {label} (MC {TRIALS}x{len(SCENE_SEEDS)} scenes) ===")
        cell = {}
        for kind in DEPLOYABLE_BASELINES:
            rows = []
            for (scene, ev), s in insts:
                omega = uniform_participation(scene.num_nodes)
                rows.append(run_dynamic_mc(scene, ev, make_baseline(kind, scene), proto, phy,
                                           num_trials=TRIALS, generator=torch.Generator().manual_seed(s),
                                           link_override=None, service_profile=prof, participation=omega))
            n = TRIALS * len(insts)
            P = statistics.mean(r.basin_P_correct for r in rows)
            Fw = statistics.mean(r.basin_F_wrong for r in rows)
            Fs = statistics.mean(r.basin_F_split for r in rows)
            Fd = statistics.mean(r.basin_F_deadline for r in rows)
            cell[kind] = schema.macro_block(P, Fw, Fs, Fd, ci={
                "macro_P_correct": wilson_ci(P, n), "macro_F_wrong": wilson_ci(Fw, n),
                "macro_F_split": wilson_ci(Fs, n), "macro_F_deadline": wilson_ci(Fd, n)})
            log(f"  {kind:16s} Pc={P:.3f} Fd={Fd:.3f} Fw={Fw:.3f}")
        pcs = [cell[k]["macro_P_correct"] for k in DEPLOYABLE_BASELINES]
        spec = build_experiment_spec(protocol_cfg=proto, service_profile=prof, phy_cfg=phy,
                                     evidence_descriptor=f"{scen}:p={be}",
                                     scene_descriptor={"gx": grid[0], "gy": grid[1], "v": grid[2]},
                                     query_law="esp", full_physics=True)
        out["regimes"][label] = {
            "policies": cell, "mc_spread_P_correct": max(pcs) - min(pcs),
            "best": max(DEPLOYABLE_BASELINES, key=lambda k: cell[k]["macro_P_correct"]),
            "worst": min(DEPLOYABLE_BASELINES, key=lambda k: cell[k]["macro_P_correct"]),
            "manifest": mf.build_manifest(spec, policy_hash="esp_heuristics", checkpoint_hash="none",
                                          model_seeds=SCENE_SEEDS, git_commit=git,
                                          manifest_id="EV2-headroom-mc")}
        log(f"  -> MC spread(Pc) = {out['regimes'][label]['mc_spread_P_correct']:.3f}")
        json.dump(out, open(OUT, "w"), indent=2)
    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"DONE {out['runtime_total_s']}s -> {OUT}")


if __name__ == "__main__":
    main()
