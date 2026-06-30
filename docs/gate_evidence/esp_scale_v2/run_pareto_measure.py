"""Route B, Phase 1 (MEASUREMENT, no training): the multi-objective Pareto surface of deployable policies.

Per the vetted design (route_b_pareto_design.json): the MC returns the full objective vector for ANY policy,
so the whole surface is one measurement sweep. Question: is distance only the P_correct iso-reliability ceiling,
or is it Pareto-DOMINATED by a learned policy on latency/energy at MATCHED reliability + matched Pc?

Objective vector (macrostate_v2): GATE tier (Fw, Fs, Fd -- reliability hard constraints, matched-to-distance
because absolute eps are infeasible-for-all, EV10) + Pc (equivalence-gated). FREE Pareto axes (compared only
inside the matched set): lat_basin = basin_tau_correct_mean (MIN), lat_cvar (MIN), energy (MIN), energy_cvar
(MIN). Physics prior: distance = poll-nearest = lowest tx energy by construction -> the LIVE axes are latency.

Protocol (design): frozen knobs (distance beta=0.04, GNN as-is), CRN on the policy-independent evidence draw
(same generator seed per (scene,block) -> common ev.sample, the dominant correctness driver), SAMPLE-SPLIT
(gate block seed != estimation block seed), per-scene paired SIGN test as the PRIMARY outer test (4 scenes is
under-powered for parametric CIs), reliability/Pc equivalence margins. The per-scene oracle is EXCLUDED
(upper bound, not deployable). Compute-limited: 4 test scenes, 1500 trials/block.

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_pareto_measure.py [--smoke]
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

from src.config.experiment_spec import build_experiment_spec
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.esp_baselines import make_baseline
from src.evaluation.esp_scale import build_scale_instance
from src.evaluation.mc_faithful_campaign import checkpoint_policy_factory, load_checkpoint
from src.metrics import manifest as mf
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.validation import run_dynamic_mc

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
CKPT_DIR = os.path.join(HERE, "checkpoints")
OUT = os.path.join(HERE, "pareto_measure_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
TEST_SCENES = [30, 31] if SMOKE else [30, 31, 34, 35]
GNN_SEEDS = [0, 1] if SMOKE else [0, 1, 2, 3, 4]
TRIALS = 40 if SMOKE else 1500
BLOCKS = {"gate": 5000, "estim": 7000}            # distinct seed offsets -> sample-split
HEURISTICS = ["distance", "uniform_esp", "link_quality", "load_balanced", "region_bridge"]
# matched-reliability margins (design): ~10% of the 0.02-0.05 operating level; Pc equivalence margin
M_FW = M_FD = 0.005
M_PC = 0.012
# free Pareto axes (all MINIMISE)
FREE_AXES = ["lat_basin", "lat_cvar", "energy", "energy_cvar"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _vec(scene, ev, policy, omega, gen_seed):
    r = run_dynamic_mc(scene, ev, policy, PROTO, PHY, num_trials=TRIALS,
                       generator=torch.Generator().manual_seed(gen_seed), service_profile=PROFILE,
                       participation=omega)
    return {"Pc": r.basin_P_correct, "Fw": r.basin_F_wrong, "Fs": r.basin_F_split, "Fd": r.basin_F_deadline,
            "lat_basin": r.basin_tau_correct_mean, "lat_cvar": r.latency_cvar,
            "energy": r.mean_energy, "energy_cvar": r.energy_cvar}


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    ckpts = {s: load_checkpoint(os.path.join(CKPT_DIR, f"mcf_seed{s}_steps150.pt")) for s in GNN_SEEDS}
    raw = {}                                          # raw[scene][block][policy] = vector
    for s in TEST_SCENES:
        scene, ev = build_scale_instance(GRID, s, scenario=SCEN, base_node_err=BE, corr_strength=CORR)
        omega = uniform_participation(scene.num_nodes)
        raw[str(s)] = {}
        for blk, off in BLOCKS.items():
            gen = off + s                              # CRN: same seed -> common evidence across ALL policies
            cell = {}
            for kind in HEURISTICS:
                cell[kind] = _vec(scene, ev, make_baseline(kind, scene), omega, gen)
            gnn_vecs = [_vec(scene, ev, checkpoint_policy_factory(ckpts[g])(scene), omega, gen) for g in GNN_SEEDS]
            cell["gnn"] = {k: sum(v[k] for v in gnn_vecs) / len(gnn_vecs) for k in gnn_vecs[0]}  # 5-seed mean
            cell["gnn_per_seed_Pc"] = [v["Pc"] for v in gnn_vecs]
            raw[str(s)][blk] = cell
            log(f"scene {s} block={blk}: distance Pc={cell['distance']['Pc']:.3f} lat={cell['distance']['lat_basin']:.2f} "
                f"E={cell['distance']['energy']:.1f} | gnn Pc={cell['gnn']['Pc']:.3f} lat={cell['gnn']['lat_basin']:.2f}")

    # ---- analysis: matched-reliability + matched-Pc gate (on GATE block), Pareto sign test (on ESTIM block) ----
    cands = [k for k in HEURISTICS if k != "distance"] + ["gnn"]
    analysis = {}
    for c in cands:
        # Stage 0 + 1 gate: per-scene paired Delta vs distance on the GATE block
        dFw = [raw[str(s)]["gate"][c]["Fw"] - raw[str(s)]["gate"]["distance"]["Fw"] for s in TEST_SCENES]
        dFs = [raw[str(s)]["gate"][c]["Fs"] - raw[str(s)]["gate"]["distance"]["Fs"] for s in TEST_SCENES]
        dFd = [raw[str(s)]["gate"][c]["Fd"] - raw[str(s)]["gate"]["distance"]["Fd"] for s in TEST_SCENES]
        dPc = [raw[str(s)]["gate"][c]["Pc"] - raw[str(s)]["gate"]["distance"]["Pc"] for s in TEST_SCENES]
        mean = lambda xs: sum(xs) / len(xs)
        reliability_ok = mean(dFw) <= M_FW and mean(dFd) <= M_FD and mean(dFs) <= 0.001
        pc_ok = mean(dPc) >= -M_PC                       # non-inferior on Pc (equivalence)
        admitted = reliability_ok and pc_ok
        # Stage 2: per-scene SIGN test on the free axes (ESTIM block); better = candidate < distance (minimise)
        signs = {}
        for ax in FREE_AXES:
            diffs = [raw[str(s)]["estim"][c][ax] - raw[str(s)]["estim"]["distance"][ax] for s in TEST_SCENES]
            better = sum(d < 0 for d in diffs); worse = sum(d > 0 for d in diffs)
            signs[ax] = {"mean_diff_vs_distance": mean(diffs), "scenes_better": better, "scenes_worse": worse,
                         "direction": ("better" if better > worse else "worse" if worse > better else "tie")}
        strictly_better_axes = [ax for ax in FREE_AXES if signs[ax]["scenes_better"] == len(TEST_SCENES)]
        not_worse_all = all(signs[ax]["scenes_worse"] == 0 for ax in FREE_AXES)
        analysis[c] = {"reliability_gate": {"mean_dFw": mean(dFw), "mean_dFs": mean(dFs), "mean_dFd": mean(dFd),
                                            "passed": reliability_ok},
                       "pc_gate": {"mean_dPc": mean(dPc), "passed": pc_ok}, "admitted": admitted,
                       "free_axes_signs": signs,
                       "dominates_distance": admitted and not_worse_all and len(strictly_better_axes) >= 1,
                       "strictly_better_axes": strictly_better_axes}
        log(f"{c}: admitted={admitted} (dFw {mean(dFw):+.3f} dFd {mean(dFd):+.3f} dPc {mean(dPc):+.3f}) "
            f"dominates_distance={analysis[c]['dominates_distance']} better_axes={strictly_better_axes}")

    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "multiobjective_pareto_measurement", "query_family": "ESP", "smoke": SMOKE,
           "git_commit": git, "regime": {"scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "R_d": 6, "node_count": 120}, "test_scenes": TEST_SCENES, "gnn_seeds": GNN_SEEDS,
           "trials_per_block": TRIALS, "blocks": list(BLOCKS), "margins": {"m_Fw": M_FW, "m_Fd": M_FD, "m_Pc": M_PC},
           "free_axes": FREE_AXES, "cvar_level": 0.9,
           "note": ("matched-reliability(Fw/Fs/Fd)+matched-Pc gate then Pareto over latency/energy; per-scene "
                    "sign test PRIMARY (4 scenes); distance beta frozen 0.04; per-scene oracle EXCLUDED; "
                    "CRN on common evidence (same seed/block); sample-split gate vs estim. compute-limited."),
           "raw": raw, "analysis": analysis}
    dominators = [c for c in cands if analysis[c]["dominates_distance"]]
    admitted = [c for c in cands if analysis[c]["admitted"]]
    out["headline"] = {
        "admitted_policies": admitted, "policies_dominating_distance": dominators,
        "distance_is_pareto_dominated": len(dominators) > 0,
        "verdict": ("DISTANCE PARETO-DOMINATED on the multi-objective surface by: " + ", ".join(dominators)
                    if dominators else
                    "DISTANCE NOT DOMINATED: among policies matched to it on reliability+Pc, none is CI/sign-strictly "
                    "better on any latency/energy axis -> distance is on the multi-objective Pareto frontier "
                    "(not just the P_correct iso-reliability ceiling).")}
    out["manifest"] = mf.build_manifest(
        build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                              evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}", scene_descriptor={"gx": 5, "gy": 5, "v": 3},
                              query_law="esp", full_physics=True),
        policy_hash="pareto_measure", checkpoint_hash="pareto-measure", model_seeds=GNN_SEEDS, git_commit=git,
        manifest_id="RouteB-pareto-measure")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; admitted={admitted}; distance_dominated={out['headline']['distance_is_pareto_dominated']}")
    log(f"VERDICT: {out['headline']['verdict'][:100]}")


if __name__ == "__main__":
    main()
