"""G-GUARDED-CDQ2 calibration: is there an operating point with reliability SLACK (low F_wrong) AND an
eta deadline LEVER (eta reduces F_deadline)? Scans (base_node_err, corr_strength) in mm_high at R_d=14,
full physics, reporting ESP F_wrong (point + UCB) and the fixed-eta deadline gain. If a cell has
F_wrong_UCB <= 0.05 AND a positive deadline gain, the guard has a meaningful enable regime; if none does,
the liveness lever and the wrong-risk are coupled (an honest finding bearing on the guard's value).
Run: PYTHONPATH=. python docs/gate_evidence/guarded_cdq2/run_guard_calibration.py
"""
from __future__ import annotations

import json
import os
import statistics
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.cdq2_factorial import wilson_ci
from src.evaluation.esp_scale import build_scale_instance
from src.evaluation.eta_curve import cdq2_diversity_for
from src.metrics.participation import uniform_participation
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy
from src.validation import run_dynamic_mc

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "guard_calibration_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
GRID, R_D, ETA, SEEDS, TRIALS = (5, 5, 3), 14, 8.0, [0, 1], 60
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=R_D)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=R_D)
# (base_node_err, corr_strength) with corr <= base_err (matched-marginal feasibility). Span low->high
# covariance to see whether a low-F_wrong + deadline-lever window exists.
CELLS = [(0.15, 0.05), (0.20, 0.05), (0.20, 0.10), (0.25, 0.10), (0.25, 0.20), (0.30, 0.25)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _run(scene_evs, make_pol):
    rows = []
    for (scene, ev, s) in scene_evs:
        omega = uniform_participation(scene.num_nodes)
        rows.append(run_dynamic_mc(scene, ev, make_pol(scene, ev), PROTO, PHY, num_trials=TRIALS,
                                   generator=torch.Generator().manual_seed(int(s)), link_override=None,
                                   service_profile=PROFILE, participation=omega))
    n = TRIALS * len(scene_evs)
    mean = lambda a: statistics.mean([getattr(r, a) for r in rows])
    return {"P_correct": mean("basin_P_correct"), "F_wrong": mean("basin_F_wrong"),
            "F_split": mean("basin_F_split"), "F_deadline": mean("basin_F_deadline"),
            "F_wrong_ucb": wilson_ci(mean("basin_F_wrong"), n)[1]}


def main():
    t0 = time.perf_counter()
    out = {"experiment": "guard_calibration", "R_d": R_D, "eta": ETA, "seeds": SEEDS, "trials": TRIALS,
           "cells": [], "feasible_lever_cells": []}
    for base_err, corr in CELLS:
        scene_evs = []
        for s in SEEDS:
            scene, ev = build_scale_instance(GRID, s, scenario="matched_marginal_high",
                                             base_node_err=base_err, corr_strength=corr)
            scene_evs.append((scene, ev, s))
        div_r = {s: cdq2_diversity_for(ev, use_sensor=True, use_map=False) for (_, ev, s) in scene_evs}

        def esp_pol(sc, ev):
            return DistanceQueryPolicy(beta_per_m=0.04)

        def cdq2_pol(sc, ev):
            for (sc2, ev2, s2) in scene_evs:
                if sc2 is sc:
                    return CDQ2Policy(DistanceQueryPolicy(beta_per_m=0.04), r=div_r[s2][1], eta=ETA,
                                      diversity=div_r[s2][0])
            return DistanceQueryPolicy(beta_per_m=0.04)

        esp = _run(scene_evs, esp_pol)
        cdq2 = _run(scene_evs, cdq2_pol)
        gain = esp["F_deadline"] - cdq2["F_deadline"]
        feasible = esp["F_wrong"] <= 0.03         # point F_wrong (UCB certification needs more trials)
        lever = gain > 0.0
        cell = {"base_node_err": base_err, "corr_strength": corr, "esp": esp, "cdq2_eta8": cdq2,
                "deadline_gain": gain, "esp_F_wrong_ucb": esp["F_wrong_ucb"],
                "reliability_feasible_eps05": feasible, "has_deadline_lever": lever,
                "wrong_increase": cdq2["F_wrong"] - esp["F_wrong"]}
        out["cells"].append(cell)
        if feasible and lever:
            out["feasible_lever_cells"].append({"base_node_err": base_err, "corr_strength": corr,
                                                "deadline_gain": gain, "esp_F_wrong_ucb": esp["F_wrong_ucb"]})
        log(f"  err={base_err} corr={corr}: ESP Fw={esp['F_wrong']:.3f}(UCB {esp['F_wrong_ucb']:.3f}) "
            f"Fd={esp['F_deadline']:.3f}; CDQ2 Fw={cdq2['F_wrong']:.3f} Fd={cdq2['F_deadline']:.3f}; "
            f"gain={gain:+.3f} feasible@.05={feasible} lever={lever}")
        json.dump(out, open(OUT, "w"), indent=2)

    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"DONE in {out['runtime_total_s']}s; feasible+lever cells: {len(out['feasible_lever_cells'])} -> {OUT}")


if __name__ == "__main__":
    main()
