"""Route A: does iso-reliability headroom over distance appear in a LESS deadline-dominated regime?

EV12 found the iso-reliability headroom ~0 at R_d=6 (deadline-dominated, F_deadline ~0.55) -- the un-gated
oracle win was bought entirely by trading F_wrong for fewer deadline misses. Hypothesis (from the mechanism):
where the deadline is NOT the binding constraint (larger R_d), 'decide faster' may buy correctness WITHOUT the
F_wrong trade, opening a non-zero iso-reliability headroom. This re-runs the EV12 reliability-frontier machine
(free-edge oracle, reward = correct - lambda*wrong) across R_d in {10, 14} (R_d=6 already in EV12), per scene,
and reports the gap-vs-distance at the lambda where the oracle's F_wrong is pulled to distance's level.

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_oracle_reliability_rd.py [--smoke]
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
OUT = os.path.join(HERE, "oracle_reliability_rd_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
BASE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
SCENE_SEEDS = [0] if SMOKE else [0, 1]
RD_LIST = [10] if SMOKE else [10, 14]                       # R_d=6 already measured in EV12 (iso ~0)
LAMBDAS = [0.0, 2.0] if SMOKE else [0.0, 2.0, 5.0]
TRAIN_STEPS = 4 if SMOKE else 80
TRAIN_TRIALS = 20 if SMOKE else 100
EVAL_TRIALS = 40 if SMOKE else 1500


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(scene, ev, policy, omega, prof, gen_seed):
    r = run_dynamic_mc(scene, ev, policy, PROTO, PHY, num_trials=EVAL_TRIALS,
                       generator=torch.Generator().manual_seed(gen_seed), service_profile=prof,
                       participation=omega)
    return {"Pc": r.basin_P_correct, "Fw": r.basin_F_wrong, "Fs": r.basin_F_split, "Fd": r.basin_F_deadline,
            "Tconfirm": r.basin_tau_correct_mean, "energy": r.mean_energy}


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "oracle_reliability_vs_Rd", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "node_count": 120, "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "scene_seeds": SCENE_SEEDS, "R_d_list": RD_LIST, "wrong_penalties": LAMBDAS,
           "train_steps": TRAIN_STEPS, "eval_trials": EVAL_TRIALS,
           "note": "iso-reliability headroom over distance vs deadline budget R_d (EV12 found ~0 at R_d=6).",
           "per_Rd": {}}
    for rd in RD_LIST:
        prof = BASE.replace(max_poll_epochs=rd)
        per_scene = {}
        for s in SCENE_SEEDS:
            scene, ev = build_scale_instance(GRID, s, scenario=SCEN, base_node_err=BE, corr_strength=CORR)
            omega = uniform_participation(scene.num_nodes)
            gen = 9000 + s
            dist = _eval(scene, ev, DistanceQueryPolicy(beta_per_m=0.04), omega, prof, gen)
            frontier = {}
            for lam in LAMBDAS:
                ts = time.perf_counter()
                tr = train_mc_edge_logit_oracle(scene, ev, prof, PROTO, PHY, steps=TRAIN_STEPS,
                                                train_trials=TRAIN_TRIALS, init="distance", base_seed=1000 * s,
                                                wrong_penalty=lam)
                m = _eval(scene, ev, free_logit_policy(tr["logits"]), omega, prof, gen)
                frontier[str(lam)] = {**m, "gap_vs_distance": m["Pc"] - dist["Pc"],
                                      "Fw_minus_distance": m["Fw"] - dist["Fw"]}
                log(f"R_d={rd} scene {s} lambda={lam}: oracle Pc={m['Pc']:.3f} Fw={m['Fw']:.3f} Fd={m['Fd']:.3f} "
                    f"(gap {m['Pc']-dist['Pc']:+.3f}, dFw {m['Fw']-dist['Fw']:+.3f}) ({time.perf_counter()-ts:.0f}s)")
            # iso-reliability: smallest lambda whose oracle Fw <= distance Fw
            feas = [(lam, frontier[str(lam)]) for lam in LAMBDAS if frontier[str(lam)]["Fw"] <= dist["Fw"] + 1e-9]
            iso = None
            if feas:
                lam, f = min(feas, key=lambda kv: kv[0]); iso = {"lambda": lam, "gap": f["gap_vs_distance"], "oracle_Fw": f["Fw"]}
            per_scene[str(s)] = {"distance": {**dist, "Fw_ucb": wilson_ci(dist["Fw"], EVAL_TRIALS)[1]},
                                 "frontier": frontier, "iso_reliability": iso}
            json.dump(out_with(out, rd, per_scene), open(OUT, "w"), indent=2)
        gaps = [per_scene[str(s)]["iso_reliability"]["gap"] for s in SCENE_SEEDS
                if per_scene[str(s)]["iso_reliability"]]
        out["per_Rd"][str(rd)] = {"per_scene": per_scene,
                                  "mean_iso_reliability_gap": (sum(gaps) / len(gaps)) if gaps else None,
                                  "distance_mean_Fd": sum(per_scene[str(s)]["distance"]["Fd"] for s in SCENE_SEEDS) / len(SCENE_SEEDS)}
        g = out["per_Rd"][str(rd)]["mean_iso_reliability_gap"]
        log(f"R_d={rd}: distance mean F_deadline={out['per_Rd'][str(rd)]['distance_mean_Fd']:.3f}; "
            f"iso-reliability mean gap={g}")
        json.dump(out, open(OUT, "w"), indent=2)

    out["headline"] = {"iso_reliability_gap_by_Rd": {str(rd): out["per_Rd"][str(rd)]["mean_iso_reliability_gap"] for rd in RD_LIST},
                       "Rd6_reference_from_EV12": -0.0067,
                       "verdict": None}
    h = out["headline"]
    best = max((v for v in h["iso_reliability_gap_by_Rd"].values() if v is not None), default=None)
    if best is None:
        h["verdict"] = "INCONCLUSIVE -- no swept lambda reached matched F_wrong at any R_d; widen lambda grid"
    elif best > 0.02:
        h["verdict"] = f"ISO-RELIABILITY HEADROOM APPEARS at larger R_d (best +{best:.3f}) -> a legitimate superiority target exists in a less deadline-dominated regime; pursue it (EV13 proved the GNN can learn it)"
    else:
        h["verdict"] = f"NO iso-reliability headroom even at larger R_d (best {best:+.3f}) -> distance is the per-edge reliability-feasible optimum across the deadline-budget range; parity is robust"
    out["manifest"] = mf.build_manifest(
        build_experiment_spec(protocol_cfg=PROTO, service_profile=BASE, phy_cfg=PHY,
                              evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}", scene_descriptor={"gx": 5, "gy": 5, "v": 3},
                              query_law="esp", full_physics=True),
        policy_hash="mc_oracle_reliability_rd", checkpoint_hash="oracle-rd", model_seeds=SCENE_SEEDS,
        git_commit=git, manifest_id="RouteA-Rd-sweep")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; iso-reliability by R_d: {h['iso_reliability_gap_by_Rd']}; VERDICT: {h['verdict'][:70]}")


def out_with(out, rd, per_scene):  # incremental save helper (partial per_Rd)
    out["per_Rd"][str(rd)] = {"per_scene": per_scene}
    return out


if __name__ == "__main__":
    main()
