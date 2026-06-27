"""G-ESP-PERFORMANCE-SCALE large-N extension: a REAL full-physics dynamic-MC datapoint at N~=3000.

Closes the spec acceptance-#2 gap (dynamic-MC performance at N=3000) with a focused contrast: a trained
shared ESP/ESD-GNN checkpoint under fixed protocol (R_d=6) vs the fixed-service-profile R_d(N)~sqrt(N)
rule. N~=10000 stays a documented statistical approximation (full rare-event MC infeasible). Merges into
esp_performance_scale_results.json under contrast['N3036'].
Run: PYTHONPATH=. python docs/gate_evidence/guarded_cdq2/run_esp_scale_largeN.py
"""
from __future__ import annotations

import json
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation import esp_scale as es

HERE = os.path.dirname(__file__)
MAIN = os.path.join(HERE, "esp_performance_scale_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, TRIALS, TARGET_N = (23, 23, 3), 20, 3036     # 23x23x3 = 3036 nodes


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.perf_counter()
    log("retraining shared ESP/ESD-GNN @N=120 seed 0 (5 steps) for the large-N contrast ...")
    ck = es.train_esp_checkpoint([(5, 5, 3)], seed=0, profile=PROFILE, proto=PROTO, phy=PHY,
                                 scenario="iid", base_node_err=0.2, steps=5, scenes_per_grid=2,
                                 hidden_dim=12)
    pol = es.policy_factory("esd_gnn", model=ck["model"])
    row = {}
    for mode in ("fixed_protocol", "fixed_service_profile"):
        prof = es.calibrated_profile(PROFILE, TARGET_N, mode=mode, base_N=120)
        proto = PROTO if prof.max_poll_epochs == PROTO.r_max else \
            ProtocolConfig(k=PROTO.k, alpha=PROTO.alpha, beta=PROTO.beta, r_max=prof.max_poll_epochs)
        log(f"  N=3036 {mode} (R_d={prof.max_poll_epochs}) MC {TRIALS} trials ...")
        te = time.perf_counter()
        ev = es.evaluate_macro(GRID, [0], pol, prof, proto, PHY, trials=TRIALS, scenario="iid",
                               base_node_err=0.2)
        row[mode] = {"R_d": prof.max_poll_epochs, "macro": ev.macro, "n_pool": ev.n_pool,
                     "feasible_point": es.feasible_point(ev.macro, prof)}
        log(f"    -> Pc={ev.macro['macro_P_correct']:.3f} Fd={ev.macro['macro_F_deadline']:.3f} "
            f"Fw={ev.macro['macro_F_wrong']:.3f} ({time.perf_counter()-te:.0f}s)")

    out = json.load(open(MAIN))
    out.setdefault("contrast", {})["N3036"] = row
    out["largeN_note"] = (f"N=3036 real full-physics dynamic MC ({TRIALS} trials, shared seed 0): "
                          "fixed protocol vs fixed-service-profile R_d(N). N>=9840 remains a documented "
                          "approximation (full rare-event MC infeasible).")
    json.dump(out, open(MAIN, "w"), indent=2)
    log(f"DONE in {time.perf_counter()-t0:.0f}s; merged contrast['N3036'] into {os.path.basename(MAIN)}")


if __name__ == "__main__":
    main()
