"""G-CDQ2-EVALUATION reproducible factorial (Phase 10).

Fair factorial with the MATCHED-MARGINAL control judged by the independent dynamic-MC basin
outcomes (the headline judge). Policies share the SAME quality (distance-based), so the ONLY
difference is the CDQ 2.0 diversity correction (eta, observable sensor-group Z); ESP == CDQ2(eta=0).
Fixed-link ablation isolates the diversity/topology lever (the diversity-FAVORABLE case: no link
noise to mask it); a full-physics spot confirms the headline direction.

Run:  PYTHONPATH=. python docs/gate_evidence/macrostate/run_cdq2_factorial.py
Writes docs/gate_evidence/macrostate/cdq2_factorial_results.json.
"""
import json
import math
import os
import statistics

# Single-thread BLAS/OpenMP BEFORE importing torch/numpy (mirrors conftest.py): the full-physics
# FBL Gauss-Legendre quadrature otherwise triggers a torch+MKL libiomp5md double-init abort.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

from src.environment import build_manhattan_scene, ProtocolConfig, RoundPhysicsConfig
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.environment.candidate_graph import build_candidate_graph
from src.config.service_profile import ConsensusServiceProfile
from src.metrics.participation import uniform_participation
from src.sampling import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy
from src.validation import run_dynamic_mc

N_SENSOR, N_MAP = 3, 3
SEEDS = [0, 1, 2, 3, 4, 5]
TRIALS = 3000
RMAX = 14
# iid (no structure) / matched-marginal pair (same q_i, rising cov — the control) /
# overlapping_sensor_source (structure ON: crosscutting sensor+map common causes — the plan's
# primary "structure on" environment, richer heterogeneous correlation).
ARMS = ("iid", "matched_marginal_low", "matched_marginal_high", "overlapping_sensor_source")
ETA = 8.0


def arm_div(arm, model):
    """Observable-group diversity matched to the arm's ACTIVE correlated groups (oracle upper bound;
    deployment-observable labels only, no truth). matched_marginal_high correlates the SENSOR group
    only; overlapping_sensor_source correlates sensor AND map."""
    def f(graph):
        dst = graph.dst_index
        s = F.one_hot(model.sensor_of[dst], N_SENSOR)
        if arm == "overlapping_sensor_source":
            m = F.one_hot(model.map_of[dst], N_MAP)
            return torch.cat([s, m], dim=-1).to(torch.float64)
        return s.to(torch.float64)
    return f


def _cell(sc, model, policy, k, link, profile, omega):
    proto = ProtocolConfig(k=k, alpha=2, beta=3, r_max=RMAX)
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    rows = [run_dynamic_mc(sc, model, policy, proto, phy, num_trials=TRIALS,
                           generator=torch.Generator().manual_seed(s), link_override=link,
                           service_profile=profile, participation=omega) for s in SEEDS]
    keys = ("P_correct", "F_wrong", "F_split", "F_deadline")
    vals = {k_: statistics.mean([getattr(m, "basin_" + k_) for m in rows]) for k_ in keys}
    return vals


def _wilson(p, n, z=1.96):
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p), 0.0) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def main():
    sc = build_manhattan_scene(3, 3, 3, block_m=120.0, comm_radius=100.0, int_radius=150.0,
                               generator=torch.Generator().manual_seed(0))
    gc = build_candidate_graph(sc.positions, sc.comm_radius)
    deg = torch.bincount(gc.src_index, minlength=sc.num_nodes)
    k = min(3, int(deg[deg > 0].min()))
    prof = ConsensusServiceProfile.urban_default().replace(k=k, alpha=2, beta=3, max_poll_epochs=RMAX)
    omega = uniform_participation(sc.num_nodes)
    base = DistanceQueryPolicy(beta_per_m=0.05)
    n_pool = TRIALS * len(SEEDS)

    results = {"config": {"N": sc.num_nodes, "k": k, "seeds": SEEDS, "trials_per_seed": TRIALS,
                          "rmax": RMAX, "eta": ETA, "base_node_err": 0.38, "corr_strength": 0.33,
                          "quality": "distance(beta_per_m=0.05)", "diversity": "observable sensor one-hot",
                          "n_pool": n_pool}, "cells": {}}
    for link_name, link in (("fixed_link_0.85", 0.85), ("full_physics", None)):
        seeds = SEEDS if link is not None else SEEDS[:3]   # full physics: fewer seeds (slower)
        if link is None:
            globals()["SEEDS"] = seeds
        for arm in ARMS:
            model = build_overlapping_scenario(sc, arm, base_node_err=0.38, corr_strength=0.33,
                                               n_sensor=N_SENSOR, n_map=N_MAP)
            r = N_SENSOR + N_MAP if arm == "overlapping_sensor_source" else N_SENSOR
            esp = _cell(sc, model, base, k, link, prof, omega)
            cdq2 = _cell(sc, model, CDQ2Policy(base, r=r, eta=ETA, diversity=arm_div(arm, model)),
                         k, link, prof, omega)
            np_ = TRIALS * len(seeds)
            cell = {
                "ESP": esp, "CDQ2": cdq2,
                "delta": {kk: round(cdq2[kk] - esp[kk], 5) for kk in esp},
                "P_correct_CI_ESP": [round(x, 4) for x in _wilson(esp["P_correct"], np_)],
                "P_correct_CI_CDQ2": [round(x, 4) for x in _wilson(cdq2["P_correct"], np_)],
                "F_wrong_CI_ESP": [round(x, 4) for x in _wilson(esp["F_wrong"], np_)],
                "F_wrong_CI_CDQ2": [round(x, 4) for x in _wilson(cdq2["F_wrong"], np_)],
            }
            results["cells"][f"{link_name}/{arm}"] = cell
            print(f"[{link_name}/{arm}] ESP P={esp['P_correct']:.4f} Fw={esp['F_wrong']:.4f} | "
                  f"CDQ2 P={cdq2['P_correct']:.4f} Fw={cdq2['F_wrong']:.4f} | "
                  f"ΔP={cdq2['P_correct']-esp['P_correct']:+.4f} ΔFw={cdq2['F_wrong']-esp['F_wrong']:+.4f}")

    out = os.path.join(os.path.dirname(__file__), "cdq2_factorial_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
