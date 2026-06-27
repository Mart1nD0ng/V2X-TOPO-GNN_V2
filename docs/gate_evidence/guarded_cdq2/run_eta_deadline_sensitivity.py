"""G-ETA-RISK-LIVENESS confirmation: is the full-physics absence of an eta liveness benefit real, or
an artifact of a too-tight deadline (R_d=6)? Re-runs full-physics mm_high at a TIGHT (R_d=6) and a
LOOSE (R_d=14, S15-like) deadline. If the loose deadline reveals an eta deadline benefit, the benefit
exists in a feasible window; if neither does, the full-physics absence is robust (bears on stop #3).
Also a quick mechanistic check: does CDQ2(eta) select physically MORE DISTANT peers (worse links)?
Run: PYTHONPATH=. python docs/gate_evidence/guarded_cdq2/run_eta_deadline_sensitivity.py
"""
from __future__ import annotations

import json
import os
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation import eta_curve as ec
from src.evaluation.esp_scale import build_scale_instance

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "eta_deadline_sensitivity_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
GRID = (5, 5, 3)
ETAS = (0.0, 1.0, 2.0, 4.0, 8.0)
SEEDS = [0, 1]
TRIALS = 100


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def selected_distance_check():
    """Mean physical distance of the CDQ2-selected k-subset vs ESP, on one mm_high scene (analytic
    inclusion-weighted distance). If CDQ2(eta) > ESP, diversity does pick more distant (worse-link) peers."""
    from src.environment.candidate_graph import build_candidate_graph
    from src.sampling.baseline_policies import DistanceQueryPolicy
    from src.sampling.cdq2_wiring import cdq2_edge_inclusion
    from src.sampling.esp_query import edge_inclusion_probabilities
    scene, ev = build_scale_instance(GRID, 0, scenario="matched_marginal_high", base_node_err=0.35,
                                     corr_strength=0.3)
    gc = build_candidate_graph(scene.positions, scene.comm_radius)
    base = DistanceQueryPolicy(beta_per_m=0.04)
    s = base.log_weights(gc)
    N, k = scene.num_nodes, 3
    pi_esp = edge_inclusion_probabilities(gc.src_index, gc.dst_index, N, s, k)
    div, r = ec.cdq2_diversity_for(ev, use_sensor=True, use_map=False)
    out = {}
    for eta in (0.0, 4.0, 8.0):
        if eta == 0.0:
            pi = pi_esp
        else:
            Z = div(gc)
            pi = cdq2_edge_inclusion(gc.src_index, gc.dst_index, N, torch.exp(s), Z, eta, k)
        # inclusion-weighted mean selected distance (normalised by k per source -> just weight by pi)
        out[eta] = float((pi * gc.distance).sum() / pi.sum())
    return out


def main():
    t0 = time.perf_counter()
    out = {"metric_namespace_version": "macrostate_v2", "experiment": "eta_deadline_sensitivity",
           "env": "matched_marginal_high (full physics)", "etas": list(ETAS), "seeds": SEEDS,
           "trials": TRIALS, "deadlines": {}}
    log("mechanistic check: mean selected-peer distance (ESP vs CDQ2 eta) ...")
    out["selected_mean_distance_by_eta"] = selected_distance_check()
    log(f"  selected distance: {out['selected_mean_distance_by_eta']}")

    for r_d in (6, 14):
        prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=r_d)
        proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=r_d)
        log(f"full-physics mm_high sweep, R_d={r_d} ...")
        ts = time.perf_counter()
        sweep = ec.eta_sweep(GRID, SEEDS, scenario="matched_marginal_high", base_node_err=0.35,
                             corr_strength=0.3, profile=prof, proto=proto, phy=PHY, trials=TRIALS,
                             link_override=None, eta_grid=ETAS)
        fd = {f"{e:g}": sweep[e]["macro_F_deadline"] for e in ETAS}
        pc = {f"{e:g}": sweep[e]["macro_P_correct"] for e in ETAS}
        best_eta = min(ETAS, key=lambda e: sweep[e]["macro_F_deadline"])
        out["deadlines"][f"R_d={r_d}"] = {
            "F_deadline_by_eta": fd, "P_correct_by_eta": pc,
            "min_F_deadline_eta": best_eta,
            "deadline_benefit_vs_ESP": sweep[0.0]["macro_F_deadline"] - sweep[best_eta]["macro_F_deadline"],
            "macro_blocks": {f"{e:g}": sweep[e] for e in ETAS}}
        log(f"  R_d={r_d}: min F_deadline at eta={best_eta:g}; benefit vs ESP="
            f"{out['deadlines'][f'R_d={r_d}']['deadline_benefit_vs_ESP']:+.3f} ({time.perf_counter()-ts:.0f}s)")
        json.dump(out, open(OUT, "w"), indent=2)

    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"DONE in {out['runtime_total_s']}s -> {OUT}")


if __name__ == "__main__":
    main()
