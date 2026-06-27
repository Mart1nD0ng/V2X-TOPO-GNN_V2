"""G-SCALE-GENERALIZATION evidence (Phase 11): near-linear scaling + OOD-axis enforcement.

(A) SCALING: the canonical episode (ESP + CDQ 2.0) is near-linear in N with BOUNDED degree (fixed
density), no N x N. Times N=100..~10000 with a FIXED protocol / service-profile / physics, fits the
log-log runtime slope, and confirms maxdeg constant + total padded cells <= 2E.

(B) OOD: scaling N is the registered ``node_count`` axis (only scene_distribution_hash changes; the
protocol / service-profile / physics hashes are IDENTICAL across N). The enforcement matrix:
node_count allowed (registered) -> OK; a protocol / service-profile mismatch -> BLOCKED; an
ideal/full-link mismatch -> ALWAYS BLOCKED (constraint #9).

Run:  PYTHONPATH=. python docs/gate_evidence/macrostate/run_scale_generalization.py
Writes docs/gate_evidence/macrostate/scale_generalization_results.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import math
import statistics
import time

import torch
import torch.nn.functional as F

from src.config.experiment_spec import (
    IncompatibleExperimentError,
    build_experiment_spec,
    check_train_eval_compatible,
)
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene, run_consensus_episode
from src.environment.candidate_graph import build_candidate_graph
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.mainline.global_evaluator import build_bucketed_padding
from src.sampling import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy

PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRIDS = [(5, 5, 3), (8, 8, 3), (14, 14, 3), (22, 22, 3), (32, 32, 3), (41, 41, 3)]
COMM, BLOCK = 95.0, 120.0


def sensor_div(model):
    n = int(model.sensor_of.max()) + 1
    return (lambda graph: F.one_hot(model.sensor_of[graph.dst_index], n).to(torch.float64)), n


def _spec(gx, gy, v, *, query_law, full_physics=False, allowed=()):
    scene_desc = {"builder": "manhattan", "gx": gx, "gy": gy, "v": v, "comm": COMM, "block": BLOCK}
    return build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor="iid:p=0.2", scene_descriptor=scene_desc,
                                 query_law=query_law, full_physics=full_physics, allowed_ood_axes=allowed)


def main():
    out = {"config": {"k": 3, "r_max": 6, "comm_radius": COMM, "block_m": BLOCK,
                      "quality": "distance(0.04)", "eta": 4.0}, "scaling": [], "ood": {}}
    specs = {}
    for (gx, gy, v) in GRIDS:
        sc = build_manhattan_scene(gx, gy, v, block_m=BLOCK, comm_radius=COMM, int_radius=150.0,
                                   generator=torch.Generator().manual_seed(0))
        N = sc.num_nodes
        gc = build_candidate_graph(sc.positions, sc.comm_radius)
        E = gc.num_edges
        deg = torch.bincount(gc.src_index, minlength=N)
        pad = build_bucketed_padding(gc.src_index, gc.dst_index, N)
        ev = build_overlapping_scenario(sc, "iid", base_node_err=0.2)
        base = DistanceQueryPolicy(beta_per_m=0.04)
        div, r = sensor_div(ev)
        cdq2 = CDQ2Policy(base, r=r, eta=4.0, diversity=div)
        t = {}
        for name, pol in (("ESP", base), ("CDQ2", cdq2)):
            t0 = time.perf_counter()
            run_consensus_episode(sc, ev, pol, PROTO, PHY, return_trajectory=False, link_override=0.9)
            t[name] = time.perf_counter() - t0
        specs[N] = _spec(gx, gy, v, query_law="esp")
        out["scaling"].append({"N": N, "E": E, "maxdeg": int(deg.max()),
                               "total_cells": pad.total_cells, "total_cells_le_2E": pad.total_cells <= 2 * E,
                               "t_ESP": round(t["ESP"], 4), "t_CDQ2": round(t["CDQ2"], 4)})
        print(f"N={N:5d} E={E:6d} maxdeg={int(deg.max())} cells={pad.total_cells} (<=2E {pad.total_cells<=2*E})"
              f"  ESP={t['ESP']:.3f}s CDQ2={t['CDQ2']:.3f}s")

    Ns = [s["N"] for s in out["scaling"]]

    def slope(ys):
        lx = [math.log(x) for x in Ns]; ly = [math.log(y) for y in ys]
        mx, my = statistics.mean(lx), statistics.mean(ly)
        return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)

    out["runtime_slope_loglog"] = {"ESP": round(slope([s["t_ESP"] for s in out["scaling"]]), 3),
                                   "CDQ2": round(slope([s["t_CDQ2"] for s in out["scaling"]]), 3)}
    out["maxdeg_constant"] = len(set(s["maxdeg"] for s in out["scaling"])) == 1
    out["all_total_cells_le_2E"] = all(s["total_cells_le_2E"] for s in out["scaling"])

    # ---- (B) OOD enforcement matrix: scaling N == the registered node_count axis ----
    small, large = Ns[0], Ns[-1]
    s_small, s_large = specs[small], specs[large]
    out["ood"]["fixed_hashes_across_N"] = {
        "protocol_hash_constant": len(set(s.protocol_hash for s in specs.values())) == 1,
        "service_profile_hash_constant": len(set(s.service_profile_hash for s in specs.values())) == 1,
        "physics_hash_constant": len(set(s.physics_hash for s in specs.values())) == 1,
        "scene_distribution_hash_varies": len(set(s.scene_distribution_hash for s in specs.values())) == len(specs),
    }
    # node_count registered -> allowed
    ok = True
    try:
        check_train_eval_compatible(s_small, _spec(*GRIDS[-1], query_law="esp", allowed=("node_count",)))
    except IncompatibleExperimentError:
        ok = False
    # not registered -> blocked
    blocked = False
    try:
        check_train_eval_compatible(s_small, s_large)
    except IncompatibleExperimentError:
        blocked = True
    # protocol mismatch (non-OOD) -> blocked even if node_count registered
    proto_blocked = False
    alt_proto = ProtocolConfig(k=3, alpha=2, beta=4, r_max=6)   # beta differs
    s_alt = build_experiment_spec(protocol_cfg=alt_proto, service_profile=PROFILE, phy_cfg=PHY,
                                  evidence_descriptor="iid:p=0.2",
                                  scene_descriptor={"builder": "manhattan", "gx": 5, "gy": 5, "v": 3},
                                  query_law="esp", full_physics=False, allowed_ood_axes=("node_count",))
    try:
        check_train_eval_compatible(s_small, s_alt)
    except IncompatibleExperimentError:
        proto_blocked = True
    # ideal/full-link mismatch -> ALWAYS blocked
    ideal_blocked = False
    s_full = _spec(*GRIDS[0], query_law="esp", full_physics=True, allowed=("node_count", "physics"))
    try:
        check_train_eval_compatible(s_small, s_full)   # s_small is full_physics=False
    except IncompatibleExperimentError:
        ideal_blocked = True
    out["ood"]["node_count_registered_allowed"] = ok
    out["ood"]["unregistered_node_count_blocked"] = blocked
    out["ood"]["protocol_mismatch_blocked"] = proto_blocked
    out["ood"]["ideal_full_link_always_blocked"] = ideal_blocked

    print(f"\nruntime slope (log-log): {out['runtime_slope_loglog']}")
    print(f"maxdeg constant: {out['maxdeg_constant']}  total_cells<=2E all: {out['all_total_cells_le_2E']}")
    print(f"OOD matrix: {out['ood']}")

    path = os.path.join(os.path.dirname(__file__), "scale_generalization_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
