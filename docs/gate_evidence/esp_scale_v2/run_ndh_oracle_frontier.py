"""G-NDH-ORACLE-FRONTIER -- the decisive oracle-first gate for the NDH benchmark.

Question: do the NON-DISTANCE physics mechanisms (SPS persistent collision, heterogeneous receiver
capacity, RSU/hotspot geometry) open MATCHED-RELIABILITY headroom over distance AND over the best
deployable heuristic -- the precondition for training a GNN at all (plan §9/§16.3)?

Note (scope): CSI aging is FEATURE-only (physics uses the current channel), so a per-edge oracle that
optimises against the MC PHYSICS cannot extract CSI headroom -- CSI is excluded from the oracle frontier
(it can only make the deployable model NOISIER, never raise the physics optimum). The physics-changing
mechanisms tested are capacity and SPS (each isolated) plus the combined band.

Per regime x scene: eval distance + best_heuristic_envelope (reliability-feasible winner) + free-edge
oracle (wrong_penalty=0, un-gated UPPER BOUND) + wrong-penalised oracle over a lambda sweep. The
ISO-RELIABILITY frontier = at the smallest lambda where the oracle's F_wrong <= distance's F_wrong, the
oracle P_correct gap vs distance AND vs the heuristic envelope. Headroom exists only if that iso-
reliability gap is clearly > 0 (the per-scene oracle is an UPPER BOUND -> if even IT has no matched-
reliability gap, no diagonal ESP policy / GNN can, and the honest conclusion is parity, plan §16.3).

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_ndh_oracle_frontier.py [--smoke]
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
from src.environment.nonuniform_urban_scene import build_nonuniform_urban_scene
from src.environment.overlapping_evidence import build_overlapping_scenario
from src.evaluation.cdq2_factorial import wilson_ci
from src.evaluation.esp_baselines import free_logit_policy, train_mc_edge_logit_oracle
from src.evaluation.ndh_baselines import best_heuristic_envelope
from src.metrics import manifest as mf
from src.metrics import schema
from src.metrics.participation import vehicle_only_participation
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.validation import run_dynamic_mc

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "ndh_oracle_frontier_results.json")
SCEN, BE, CORR = "matched_marginal_high", 0.35, 0.25
GRID = (5, 5, 3)
SCENE_SEEDS = [30] if SMOKE else [30, 31]
LAMBDAS = [0.0, 5.0] if SMOKE else [0.0, 5.0]     # un-gated UPPER BOUND + one penalised; extend if a frontier emerges
ORACLE_STEPS = 6 if SMOKE else 40
ORACLE_TRIALS = 40 if SMOKE else 60
EVAL_TRIALS = 60 if SMOKE else 1000
ENVELOPE_TRIALS = 60 if SMOKE else 500            # envelope only needs to pick the winner
DELTA_W = 0.005

# physics-changing NDH regimes (CSI excluded -- feature-only). Each: scene kwargs, physics kwargs, R_d.
REGIMES = {
    "capacity_rd20": {"scene": dict(enable_rsu=True, p_intersection_rsu=0.3, enable_hotspots=True,
                                    enable_heterogeneous_capacity=True), "kappa": 0.0, "rd": 20},
    "sps_rd10": {"scene": dict(enable_sps=True, sps_n_buckets=40), "kappa": 0.5, "rd": 10},
    # combined_rd20 deferred to a second pass if a single-mechanism frontier emerges (compute-limited)
}
if SMOKE:
    REGIMES = {"capacity_rd20": REGIMES["capacity_rd20"]}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _eval(scene, ev, policy, omega, prof, proto, phy, gen_seed):
    r = run_dynamic_mc(scene, ev, policy, proto, phy, num_trials=EVAL_TRIALS,
                       generator=torch.Generator().manual_seed(gen_seed), service_profile=prof,
                       participation=omega)
    return {"Pc": r.basin_P_correct, "Fw": r.basin_F_wrong, "Fs": r.basin_F_split, "Fd": r.basin_F_deadline,
            "Fw_ucb": wilson_ci(r.basin_F_wrong, EVAL_TRIALS)[1]}


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "ndh_oracle_frontier", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "node_count": 120, "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "scene_seeds": SCENE_SEEDS, "wrong_penalties": LAMBDAS, "eval_trials": EVAL_TRIALS,
           "oracle_steps": ORACLE_STEPS, "delta_w": DELTA_W,
           "note": ("does any physics-changing NDH mechanism (SPS/capacity/hotspot) open matched-reliability "
                    "headroom over distance AND the heuristic envelope? CSI is feature-only -> excluded "
                    "(oracle optimises physics). Per-scene oracle = UPPER BOUND, excluded from deployable claims."),
           "per_regime": {}}

    for rname, rspec in REGIMES.items():
        prof = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=rspec["rd"])
        proto = ProtocolConfig(k=3, alpha=2, beta=3, r_max=rspec["rd"])
        phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40, resource_collision_kappa=rspec["kappa"])
        per_scene = {}
        for s in SCENE_SEEDS:
            scene, ev = _build(rspec, s)
            omega = vehicle_only_participation(scene)
            gen = 9000 + s
            dist = _eval(scene, ev, DistanceQueryPolicy(beta_per_m=0.04), omega, prof, proto, phy, gen)
            # heuristic envelope (reliability-feasible winner) -- the deployable bar
            env = best_heuristic_envelope(scene, ev, prof, proto, phy, trials=ENVELOPE_TRIALS, base_seed=s,
                                          distance_beta=0.04, delta_w=DELTA_W)
            envm = env["winner_metrics"]
            log(f"{rname} scene {s}: distance Pc={dist['Pc']:.3f} Fw={dist['Fw']:.3f} | "
                f"envelope winner={env['winner']} Pc={envm['Pc']:.3f} Fw={envm['Fw']:.3f}")
            frontier = {}
            for lam in LAMBDAS:
                ts = time.perf_counter()
                tr = train_mc_edge_logit_oracle(scene, ev, prof, proto, phy, steps=ORACLE_STEPS,
                                                train_trials=ORACLE_TRIALS, init="distance", base_seed=1000 * s,
                                                wrong_penalty=lam)
                m = _eval(scene, ev, free_logit_policy(tr["logits"]), omega, prof, proto, phy, gen)
                frontier[str(lam)] = {**m, "gap_vs_distance": m["Pc"] - dist["Pc"],
                                      "gap_vs_envelope": m["Pc"] - envm["Pc"], "Fw_minus_distance": m["Fw"] - dist["Fw"]}
                log(f"  {rname} s{s} lam={lam}: oracle Pc={m['Pc']:.3f} Fw={m['Fw']:.3f} "
                    f"(gap_dist {m['Pc']-dist['Pc']:+.3f} gap_env {m['Pc']-envm['Pc']:+.3f} dFw {m['Fw']-dist['Fw']:+.3f}) "
                    f"({time.perf_counter()-ts:.0f}s)")
                per_scene[str(s)] = {"distance": dist, "envelope": {"winner": env["winner"], **envm},
                                     "frontier": frontier}
                out["per_regime"][rname] = {"per_scene": per_scene}
                json.dump(out, open(OUT, "w"), indent=2)          # save-before-assert, incremental
            # iso-reliability: smallest lambda whose oracle Fw <= distance Fw
            feas = [(lam, frontier[str(lam)]) for lam in LAMBDAS if frontier[str(lam)]["Fw"] <= dist["Fw"] + 1e-9]
            iso = None
            if feas:
                lam, f = min(feas, key=lambda kv: kv[0])
                iso = {"lambda": lam, "gap_vs_distance": f["gap_vs_distance"],
                       "gap_vs_envelope": f["gap_vs_envelope"], "oracle_Fw": f["Fw"]}
            per_scene[str(s)]["iso_reliability"] = iso
        # regime summary
        isos_d = [per_scene[str(s)]["iso_reliability"]["gap_vs_distance"] for s in SCENE_SEEDS
                  if per_scene[str(s)].get("iso_reliability")]
        isos_e = [per_scene[str(s)]["iso_reliability"]["gap_vs_envelope"] for s in SCENE_SEEDS
                  if per_scene[str(s)].get("iso_reliability")]
        out["per_regime"][rname] = {
            "per_scene": per_scene,
            "mean_iso_gap_vs_distance": (sum(isos_d) / len(isos_d)) if isos_d else None,
            "mean_iso_gap_vs_envelope": (sum(isos_e) / len(isos_e)) if isos_e else None}
        g = out["per_regime"][rname]
        log(f"{rname}: iso-reliability mean gap vs distance={g['mean_iso_gap_vs_distance']} "
            f"vs envelope={g['mean_iso_gap_vs_envelope']}")
        json.dump(out, open(OUT, "w"), indent=2)

    # ---- headline decision ----
    def _regime_headroom(g):
        gd, ge = g.get("mean_iso_gap_vs_distance"), g.get("mean_iso_gap_vs_envelope")
        return (gd is not None and ge is not None and gd > 0.02 and ge > 0.02)
    winners = [r for r, g in out["per_regime"].items() if _regime_headroom(g)]
    out["headline"] = {
        "regimes": list(REGIMES),
        "iso_gap_vs_distance": {r: out["per_regime"][r].get("mean_iso_gap_vs_distance") for r in REGIMES},
        "iso_gap_vs_envelope": {r: out["per_regime"][r].get("mean_iso_gap_vs_envelope") for r in REGIMES},
        "regimes_with_matched_reliability_headroom": winners,
        "train_gnn": len(winners) > 0,
        "verdict": ("MATCHED-RELIABILITY HEADROOM in: " + ", ".join(winners) +
                    " -> proceed to G-NDH-STATIC-ESDGNN-V2 (the oracle upper-bounds a gnn-reachable win)"
                    if winners else
                    "NO matched-reliability headroom in any physics-changing NDH regime -> the mechanisms do "
                    "NOT create legal superiority over distance/best-heuristic for a diagonal ESP policy; the "
                    "honest conclusion is PARITY (as in the old-physics EV12-15). STOP: do NOT train the GNN.")}
    out["manifest"] = mf.build_manifest(
        build_experiment_spec(protocol_cfg=ProtocolConfig(k=3, alpha=2, beta=3, r_max=20),
                              service_profile=ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3),
                              phy_cfg=RoundPhysicsConfig(subchannels=10, slots_per_window=40),
                              evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}", scene_descriptor={"gx": 5, "gy": 5, "v": 3},
                              query_law="esp", full_physics=True),
        policy_hash="ndh_oracle_frontier", checkpoint_hash="ndh-oracle", model_seeds=SCENE_SEEDS,
        git_commit=git, manifest_id="NDH-oracle-frontier")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; train_gnn={out['headline']['train_gnn']}; VERDICT: {out['headline']['verdict'][:90]}")


def _build(rspec, seed):
    scene = build_nonuniform_urban_scene(*GRID, generator=torch.Generator().manual_seed(int(seed)), **rspec["scene"])
    ev = build_overlapping_scenario(scene, SCEN, base_node_err=BE, corr_strength=CORR)
    return scene, ev


if __name__ == "__main__":
    main()
