"""G-GUARDED-CDQ2 evidence (Guarded-CDQ2 round, Phase 5).

Demonstrates the deployable Guarded-CDQ2 policy (spec §4): CDQ2 diversity is enabled ONLY when there
is reliability slack AND deadline pressure; otherwise it falls back to ESP (the default, constraint #3).
Two regimes (full physics, judged by the independent dynamic MC):

  * FEASIBLE  -- mm_high, low node error, feasible-but-stressed deadline (R_d=14): ESP has reliability
    slack + deadline pressure -> the guard ENABLES eta -> a deadline/liveness gain, wrong/split UCB still
    within budget.
  * SAFETY-CRITICAL -- mm_high, high node error: ESP is near/over the wrong budget (no slack) -> the
    guard DISABLES eta (-> ESP). Here FIXED-eta would RAISE the wrong basin and VIOLATE the constraint,
    which is exactly why the guard is needed.

Arms: ESP / fixed-eta / hard-guard / soft-guard / oracle-guard. The guard reads CALIBRATED UCB slack
(an ESP pre-pass for hard/soft; the CDQ2 counterfactual for the oracle). Reported at TWO service targets
(strict eps=1e-3 and moderate eps=0.05) to show the guard adapts to the target -- NOT a tuned pass-knob
(constraint: thresholds are service targets, never lowered to force a pass).
Run: PYTHONPATH=. python docs/gate_evidence/guarded_cdq2/run_guarded_cdq2.py [--smoke]
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
from src.evaluation.esp_scale import build_scale_instance
from src.evaluation.eta_curve import cdq2_diversity_for
from src.metrics import manifest as mf
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.policies.guarded_cdq2 import GuardConfig, hard_guard_eta, soft_guard_eta
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy
from src.validation import run_dynamic_mc

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "guarded_cdq2_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
GRID = (5, 5, 3)
ETA_RAW = 8.0
R_D = 14
SEEDS = [0] if SMOKE else [0, 1, 2]
TRIALS = 40 if SMOKE else 100
# (regime label, base_node_err, corr_strength) -- CALIBRATED operating points (run_guard_calibration):
# enable = moderate covariance + low error + stressed deadline (reliability slack AND an eta deadline
# lever); safety_critical = high covariance/error (no slack; fixed-eta RAISES F_wrong). corr <= err.
REGIMES = [("enable", 0.20, 0.10), ("safety_critical", 0.30, 0.25)]
# SERVICE-TARGET sweep (NOT a pass-knob): the default eps=1e-3 is unachievable in these correlated-error
# regimes -> the guard correctly defaults to ESP everywhere; looser targets show the guard ENABLE eta
# only where ESP has slack, capturing the deadline gain while keeping F_wrong UCB <= eps.
TARGETS = {"strict_eps_1e-3": 1e-3, "eps_0.05": 0.05, "eps_0.10": 0.10}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _profile(eps):
    return ConsensusServiceProfile.urban_default().replace(
        k=3, alpha=2, beta=3, max_poll_epochs=R_D,
        max_wrong_basin_probability=eps, max_split_basin_probability=eps)


def _proto():
    return ProtocolConfig(k=3, alpha=2, beta=3, r_max=R_D)


def _pooled(rows, n_pool):
    mean = lambda a: statistics.mean([getattr(r, a) for r in rows])
    P, Fw, Fs, Fd = (mean("basin_P_correct"), mean("basin_F_wrong"),
                     mean("basin_F_split"), mean("basin_F_deadline"))
    ci = {"macro_P_correct": wilson_ci(P, n_pool), "macro_F_wrong": wilson_ci(Fw, n_pool),
          "macro_F_split": wilson_ci(Fs, n_pool), "macro_F_deadline": wilson_ci(Fd, n_pool)}
    return schema.macro_block(P, Fw, Fs, Fd, ci=ci)


def _run(scene_evs, policy_fn, profile, proto):
    rows = []
    for (scene, ev, s) in scene_evs:
        omega = uniform_participation(scene.num_nodes)
        rows.append(run_dynamic_mc(scene, ev, policy_fn(scene, ev), proto, PHY, num_trials=TRIALS,
                                   generator=torch.Generator().manual_seed(int(s)), link_override=None,
                                   service_profile=profile, participation=omega))
    return _pooled(rows, TRIALS * len(scene_evs))


def _feasible_ucb(macro, eps):
    return macro["macro_F_wrong_ci"][1] <= eps and macro["macro_F_split_ci"][1] <= eps


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    proto = _proto()
    out = {"metric_namespace_version": "macrostate_v2", "experiment": "guarded_cdq2", "smoke": SMOKE,
           "git_commit": git, "eta_raw": ETA_RAW, "R_d": R_D, "seeds": SEEDS, "trials": TRIALS,
           "note": ("ESP is the default; CDQ2 diversity (eta>0) enabled only with reliability slack AND "
                    "deadline pressure (spec §4). eps is the SERVICE TARGET (reported at two targets), "
                    "never lowered to force a pass. Headline judge = independent dynamic MC, full physics."),
           "regimes": {}}

    for regime, base_err, corr in REGIMES:
        log(f"=== regime {regime} (base_err={base_err}, corr={corr}, R_d={R_D}) ===")
        scene_evs = []
        for s in SEEDS:
            scene, ev = build_scale_instance(GRID, s, scenario="matched_marginal_high",
                                             base_node_err=base_err, corr_strength=corr)
            scene_evs.append((scene, ev, s))
        div_r = {}
        for (scene, ev, s) in scene_evs:
            div_r[s] = cdq2_diversity_for(ev, use_sensor=True, use_map=False)

        # the two MC anchors: ESP (eta=0) and fixed CDQ2 (eta=ETA_RAW). All hard/oracle arms reuse these.
        prof_ref = _profile(0.05)        # MC outcomes don't depend on eps; use any profile for the run
        esp = _run(scene_evs, lambda sc, ev: DistanceQueryPolicy(beta_per_m=0.04), prof_ref, proto)
        log(f"  ESP: Pc={esp['macro_P_correct']:.3f} Fw={esp['macro_F_wrong']:.3f} "
            f"Fd={esp['macro_F_deadline']:.3f} (Fw_UCB={esp['macro_F_wrong_ci'][1]:.3f})")
        cdq2 = _run(scene_evs, lambda sc, ev: CDQ2Policy(DistanceQueryPolicy(beta_per_m=0.04),
                                                         r=div_r[scene_evs[0][2]][1], eta=ETA_RAW,
                                                         diversity=div_r_lookup(div_r, scene_evs, sc)),
                    prof_ref, proto)
        log(f"  fixed-eta({ETA_RAW}): Pc={cdq2['macro_P_correct']:.3f} Fw={cdq2['macro_F_wrong']:.3f} "
            f"Fd={cdq2['macro_F_deadline']:.3f} (Fw_UCB={cdq2['macro_F_wrong_ci'][1]:.3f})")

        # guard inputs: ESP UCBs (deployable) and CDQ2 UCBs (oracle counterfactual); p_d = ESP deadline UCB
        esp_fw_ucb, esp_fs_ucb = esp["macro_F_wrong_ci"][1], esp["macro_F_split_ci"][1]
        p_d = esp["macro_F_deadline_ci"][1]
        cdq2_fw_ucb, cdq2_fs_ucb = cdq2["macro_F_wrong_ci"][1], cdq2["macro_F_split_ci"][1]

        regime_out = {"base_node_err": base_err, "ESP": esp, "fixed_eta": cdq2,
                      "esp_Fw_UCB": esp_fw_ucb, "cdq2_Fw_UCB": cdq2_fw_ucb, "deadline_pressure": p_d,
                      "targets": {}}

        # MC cache keyed by rounded eta so repeated guard arms across service targets reuse runs.
        mc_cache = {0.0: esp, ETA_RAW: cdq2}

        def arm(eta_eff):
            if eta_eff <= 1e-9:
                return esp, "ESP(disabled)"
            if abs(eta_eff - ETA_RAW) < 1e-9:
                return cdq2, f"CDQ2(eta={ETA_RAW})"
            key = round(eta_eff, 2)
            if key not in mc_cache:
                mc_cache[key] = _run(scene_evs, lambda sc, ev: CDQ2Policy(
                    DistanceQueryPolicy(beta_per_m=0.04), r=div_r[scene_evs[0][2]][1], eta=eta_eff,
                    diversity=div_r_lookup(div_r, scene_evs, sc)), prof_ref, proto)
            return mc_cache[key], f"CDQ2(eta={eta_eff:.2f})"

        for tname, eps in TARGETS.items():
            cfg = GuardConfig.from_profile(_profile(eps), delta_frac=0.2, delta_d=0.05, T_d=0.02)
            eta_hard = hard_guard_eta(ETA_RAW, Fw_ucb=esp_fw_ucb, Fs_ucb=esp_fs_ucb, cfg=cfg)
            eta_soft = soft_guard_eta(ETA_RAW, Fw_ucb=esp_fw_ucb, Fs_ucb=esp_fs_ucb, p_d=p_d, cfg=cfg)
            eta_oracle = hard_guard_eta(ETA_RAW, Fw_ucb=cdq2_fw_ucb, Fs_ucb=cdq2_fs_ucb, cfg=cfg)

            hard_m, hard_lbl = arm(eta_hard)
            soft_m, soft_lbl = arm(eta_soft)
            orac_m, orac_lbl = arm(eta_oracle)
            spec = build_experiment_spec(protocol_cfg=proto, service_profile=_profile(eps), phy_cfg=PHY,
                                         evidence_descriptor=f"mm_high:p={base_err}",
                                         scene_descriptor={"gx": 5, "gy": 5, "v": 3}, query_law="cdq2",
                                         full_physics=True, allowed_ood_axes=("evidence_covariance",))
            man = mf.build_manifest(spec, policy_hash="guarded_cdq2", checkpoint_hash="fixed-quality+Z",
                                    model_seeds=SEEDS, git_commit=git, manifest_id="GS5-guarded")
            regime_out["targets"][tname] = {
                "eps": eps, "hashes": man,
                "eta_hard": eta_hard, "eta_soft": eta_soft, "eta_oracle": eta_oracle,
                "hard_guard": {"label": hard_lbl, "macro": hard_m, "feasible_ucb": _feasible_ucb(hard_m, eps),
                               "guard_active": eta_hard > 1e-9},
                "soft_guard": {"label": soft_lbl, "macro": soft_m, "feasible_ucb": _feasible_ucb(soft_m, eps),
                               "guard_active": eta_soft > 1e-9},
                "oracle_guard": {"label": orac_lbl, "macro": orac_m, "feasible_ucb": _feasible_ucb(orac_m, eps),
                                 "guard_active": eta_oracle > 1e-9},
                "esp_feasible_ucb": _feasible_ucb(esp, eps), "fixed_eta_feasible_ucb": _feasible_ucb(cdq2, eps),
                "fixed_eta_deadline_gain_vs_esp": esp["macro_F_deadline"] - cdq2["macro_F_deadline"]}
            log(f"  [{tname}] eta_hard={eta_hard:g} eta_soft={eta_soft:.2f} | "
                f"ESP feas={_feasible_ucb(esp, eps)} fixed-eta feas={_feasible_ucb(cdq2, eps)} "
                f"hard feas={_feasible_ucb(hard_m, eps)}")
        out["regimes"][regime] = regime_out
        json.dump(out, open(OUT, "w"), indent=2)

    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"DONE in {out['runtime_total_s']}s -> {OUT}")


def div_r_lookup(div_r, scene_evs, scene):
    # map a scene object back to its (div, r); scenes are distinct per seed -> match by identity
    for (sc, ev, s) in scene_evs:
        if sc is scene:
            return div_r[s][0]
    return div_r[scene_evs[0][2]][0]


if __name__ == "__main__":
    main()
