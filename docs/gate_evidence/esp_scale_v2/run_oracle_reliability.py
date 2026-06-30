"""Campaign A, Lever 1 follow-up (a): RELIABILITY-FEASIBLE oracle headroom frontier.

The un-gated MC oracle (run_oracle_probe.py) beats distance by +0.099, but at ~2.5x its F_wrong (the gain is
mostly reduced deadline misses converted ~3:1 into correct vs wrong). Question: how much of that +0.10 headroom
SURVIVES once F_wrong is charged? Re-train the free-edge oracle with a reward ``correct - lambda * wrong`` for a
sweep of lambda, and trace the (P_correct, F_wrong) frontier vs distance. The reliability-feasible headroom is
the P_correct gain over distance at the lambda where the oracle's F_wrong is pulled down to distance's level.

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_oracle_reliability.py [--smoke]
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
from src.evaluation.cdq2_factorial import wilson_ci
from src.evaluation.esp_baselines import free_logit_policy, train_mc_edge_logit_oracle
from src.evaluation.esp_scale import build_scale_instance
from src.metrics import manifest as mf
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.validation import run_dynamic_mc

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "oracle_reliability_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
SCENE_SEEDS = [0] if SMOKE else [0, 1]
LAMBDAS = [0.0, 3.0] if SMOKE else [0.0, 2.0, 5.0, 12.0]   # span un-gated Fw~0.05 down toward distance Fw~0.02
TRAIN_STEPS = 4 if SMOKE else 100
TRAIN_TRIALS = 20 if SMOKE else 120
EVAL_TRIALS = 40 if SMOKE else 1500


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(scene, ev, policy, omega, gen_seed):
    r = run_dynamic_mc(scene, ev, policy, PROTO, PHY, num_trials=EVAL_TRIALS,
                       generator=torch.Generator().manual_seed(gen_seed), service_profile=PROFILE,
                       participation=omega)
    return {"Pc": r.basin_P_correct, "Fw": r.basin_F_wrong, "Fs": r.basin_F_split, "Fd": r.basin_F_deadline}


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}",
                                 scene_descriptor={"gx": GRID[0], "gy": GRID[1], "v": GRID[2]},
                                 query_law="esp", full_physics=True)
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "mc_oracle_reliability_frontier", "query_family": "ESP", "smoke": SMOKE,
           "git_commit": git, "node_count": 120, "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "scene_seeds": SCENE_SEEDS, "wrong_penalties": LAMBDAS, "train_steps": TRAIN_STEPS,
           "train_trials": TRAIN_TRIALS, "eval_trials": EVAL_TRIALS,
           "note": "free-edge oracle with reward=correct-lambda*wrong; traces reliability-feasible headroom vs distance.",
           "per_scene": {}}

    for s in SCENE_SEEDS:
        scene, ev = build_scale_instance(GRID, s, scenario=SCEN, base_node_err=BE, corr_strength=CORR)
        omega = uniform_participation(scene.num_nodes)
        gen_seed = 9000 + s
        dist = _eval(scene, ev, DistanceQueryPolicy(beta_per_m=0.04), omega, gen_seed)
        frontier = {}
        for lam in LAMBDAS:
            ts = time.perf_counter()
            tr = train_mc_edge_logit_oracle(scene, ev, PROFILE, PROTO, PHY, steps=TRAIN_STEPS,
                                            train_trials=TRAIN_TRIALS, init="distance", base_seed=1000 * s,
                                            wrong_penalty=lam)
            m = _eval(scene, ev, free_logit_policy(tr["logits"]), omega, gen_seed)
            frontier[str(lam)] = {**m, "gap_vs_distance": m["Pc"] - dist["Pc"],
                                  "Fw_minus_distance": m["Fw"] - dist["Fw"], "train_s": round(time.perf_counter() - ts, 0)}
            log(f"scene {s} lambda={lam}: oracle Pc={m['Pc']:.3f} Fw={m['Fw']:.3f} Fd={m['Fd']:.3f} "
                f"(gap {m['Pc']-dist['Pc']:+.3f}, dFw {m['Fw']-dist['Fw']:+.3f})")
        out["per_scene"][str(s)] = {"distance": {**dist, "Fw_ucb": wilson_ci(dist["Fw"], EVAL_TRIALS)[1]},
                                    "frontier": frontier}
        json.dump(out, open(OUT, "w"), indent=2)

    # iso-reliability headroom: per scene, the gap at the SMALLEST lambda whose oracle Fw <= distance Fw
    iso = {}
    for s in SCENE_SEEDS:
        c = out["per_scene"][str(s)]
        dfw = c["distance"]["Fw"]
        feasible = [(lam, c["frontier"][str(lam)]) for lam in LAMBDAS if c["frontier"][str(lam)]["Fw"] <= dfw + 1e-9]
        if feasible:
            lam, f = min(feasible, key=lambda kv: kv[0])      # smallest lambda that already meets distance Fw
            iso[str(s)] = {"lambda": lam, "oracle_Pc": f["Pc"], "oracle_Fw": f["Fw"], "gap_at_matched_reliability": f["gap_vs_distance"]}
        else:
            iso[str(s)] = {"lambda": None, "note": "no swept lambda pulled F_wrong down to distance's level; need higher lambda",
                           "best_gap_with_lowest_Fw": min(c["frontier"].values(), key=lambda v: v["Fw"])["gap_vs_distance"]}
    gaps_iso = [iso[str(s)].get("gap_at_matched_reliability") for s in SCENE_SEEDS if iso[str(s)].get("gap_at_matched_reliability") is not None]
    out["headline"] = {
        "iso_reliability_per_scene": iso,
        "mean_gap_at_matched_reliability": (sum(gaps_iso) / len(gaps_iso)) if gaps_iso else None,
        "ungated_mean_gap_for_reference": 0.099,
        "verdict": None}
    h = out["headline"]
    g = h["mean_gap_at_matched_reliability"]
    if g is None:
        h["verdict"] = "FRONTIER_INCOMPLETE -> no swept lambda reached distance's F_wrong; rerun with higher lambda to close the frontier"
    elif g > 0.02:
        h["verdict"] = f"RELIABILITY-FEASIBLE HEADROOM SURVIVES (~{g:+.3f} at matched F_wrong) -> the win is NOT just bought by reliability; a learned policy beating distance at equal reliability is a legitimate target"
    else:
        h["verdict"] = f"HEADROOM MOSTLY RELIABILITY-BOUGHT (~{g:+.3f} at matched F_wrong) -> most of the un-gated +0.10 is paid for with F_wrong; the legitimate (iso-reliability) target is small"
    out["manifest"] = mf.build_manifest(spec, policy_hash="mc_oracle_reliability", checkpoint_hash="oracle-reliab",
                                        model_seeds=SCENE_SEEDS, git_commit=git, manifest_id="Lever1a-reliability")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; iso-reliability mean gap {g}; VERDICT: {h['verdict'][:70]}")


if __name__ == "__main__":
    main()
