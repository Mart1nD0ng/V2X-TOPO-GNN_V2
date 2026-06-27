"""G-HAZARD-PROFILES evidence (Guarded-CDQ2 round, Phase 6 / spec §5).

For each hazard-weighted service profile, selects ESP / CDQ2 / Guarded-CDQ2 by the feasibility-gated
net benefit B (spec §5.2). Policy outcomes (basins + energy + tail latency) come from the CANONICAL
dynamic MC (run_dynamic_mc -- NOT a separate evaluator, constraint #13). Guarded-CDQ2's outcome per
profile is the guard's decision at that profile's eps (enable -> CDQ2 outcome, disable -> ESP). Shows
policy selection changes RATIONALLY with the hazard weights AND respects scene feasibility.
Run: PYTHONPATH=. python docs/gate_evidence/guarded_cdq2/run_hazard_profiles.py [--smoke]
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
from src.config.hazard_profile import STANDARD_PROFILES
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.cdq2_factorial import wilson_ci
from src.evaluation.esp_scale import build_scale_instance
from src.evaluation.eta_curve import cdq2_diversity_for
from src.evaluation.hazard_utility import PolicyOutcome, select_policy
from src.metrics import manifest as mf
from src.metrics import schema
from src.metrics.participation import uniform_participation
from src.policies.guarded_cdq2 import GuardConfig, hard_guard_eta
from src.sampling.baseline_policies import DistanceQueryPolicy
from src.sampling.cdq2_wiring import CDQ2Policy
from src.validation import run_dynamic_mc

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "hazard_profiles_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
GRID, R_D, ETA = (5, 5, 3), 14, 8.0
SEEDS = [0] if SMOKE else [0, 1, 2]
TRIALS = 40 if SMOKE else 80
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=R_D)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=R_D)
REGIMES = [("enable", 0.20, 0.10), ("safety_critical", 0.30, 0.25)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _measure(scene_evs, make_pol):
    """Canonical dynamic MC -> pooled basins (+CIs) + mean energy + tail-latency CVaR."""
    rows = []
    for (scene, ev, s) in scene_evs:
        omega = uniform_participation(scene.num_nodes)
        rows.append(run_dynamic_mc(scene, ev, make_pol(scene, ev), PROTO, PHY, num_trials=TRIALS,
                                   generator=torch.Generator().manual_seed(int(s)), link_override=None,
                                   service_profile=PROFILE, participation=omega))
    n = TRIALS * len(scene_evs)
    mean = lambda a: statistics.mean([getattr(r, a) for r in rows])
    P, Fw, Fs, Fd = (mean("basin_P_correct"), mean("basin_F_wrong"),
                     mean("basin_F_split"), mean("basin_F_deadline"))
    macro = schema.macro_block(P, Fw, Fs, Fd, ci={
        "macro_P_correct": wilson_ci(P, n), "macro_F_wrong": wilson_ci(Fw, n),
        "macro_F_split": wilson_ci(Fs, n), "macro_F_deadline": wilson_ci(Fd, n)})
    return macro, mean("mean_energy"), mean("latency_cvar")


def _outcome(name, macro, energy, tail):
    return PolicyOutcome.from_macro(name, macro, D_q=tail, energy=energy)


def main():
    t0 = time.perf_counter()
    git = mf.current_git_commit() or "uncommitted"
    out = {"metric_namespace_version": "macrostate_v2", "experiment": "hazard_profiles", "smoke": SMOKE,
           "git_commit": git, "eta": ETA, "R_d": R_D, "seeds": SEEDS, "trials": TRIALS,
           "note": ("policy selection per hazard-weighted service profile via the feasibility-gated net "
                    "benefit B (spec §5.2). ESP is the default; Guarded-CDQ2's outcome per profile is the "
                    "guard's decision at that profile's eps. Dynamic-MC judged, full physics."),
           "regimes": {}, "acceptance": {}}

    for regime, base_err, corr in REGIMES:
        log(f"=== regime {regime} (err={base_err}, corr={corr}) ===")
        scene_evs = []
        for s in SEEDS:
            scene, ev = build_scale_instance(GRID, s, scenario="matched_marginal_high",
                                             base_node_err=base_err, corr_strength=corr)
            scene_evs.append((scene, ev, s))
        div_r = {s: cdq2_diversity_for(ev, use_sensor=True, use_map=False) for (_, ev, s) in scene_evs}

        def cdq2_pol(sc, ev):
            for (sc2, ev2, s2) in scene_evs:
                if sc2 is sc:
                    return CDQ2Policy(DistanceQueryPolicy(beta_per_m=0.04), r=div_r[s2][1], eta=ETA,
                                      diversity=div_r[s2][0])
            return DistanceQueryPolicy(beta_per_m=0.04)

        esp_macro, esp_E, esp_T = _measure(scene_evs, lambda sc, ev: DistanceQueryPolicy(beta_per_m=0.04))
        cdq2_macro, cdq2_E, cdq2_T = _measure(scene_evs, cdq2_pol)
        esp = _outcome("ESP", esp_macro, esp_E, esp_T)
        cdq2 = _outcome("CDQ2", cdq2_macro, cdq2_E, cdq2_T)
        log(f"  ESP  Fw={esp.F_wrong:.3f}(UCB {esp.F_wrong_ucb:.3f}) Fd={esp.F_deadline:.3f} "
            f"E={esp.energy:.3g} D_q={esp.D_q:.3g}")
        log(f"  CDQ2 Fw={cdq2.F_wrong:.3f}(UCB {cdq2.F_wrong_ucb:.3f}) Fd={cdq2.F_deadline:.3f} "
            f"E={cdq2.energy:.3g} D_q={cdq2.D_q:.3g}")

        reg_out = {"base_node_err": base_err, "ESP": esp_macro, "CDQ2": cdq2_macro,
                   "energy": {"ESP": esp.energy, "CDQ2": cdq2.energy},
                   "tail_latency": {"ESP": esp.D_q, "CDQ2": cdq2.D_q}, "profiles": {}}
        spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                     evidence_descriptor=f"mm_high:p={base_err}",
                                     scene_descriptor={"gx": 5, "gy": 5, "v": 3}, query_law="cdq2",
                                     full_physics=True, allowed_ood_axes=("evidence_covariance",))
        man = mf.build_manifest(spec, policy_hash="hazard_selection", checkpoint_hash="fixed-quality+Z",
                                model_seeds=SEEDS, git_commit=git, manifest_id="GS6-hazard")
        reg_out["hashes"] = man

        for prof in STANDARD_PROFILES:
            # Guarded-CDQ2's outcome at this profile's eps: enable -> CDQ2, disable -> ESP.
            cfg = GuardConfig.from_profile(prof_to_service(prof), delta_frac=0.2)
            eta_g = hard_guard_eta(ETA, Fw_ucb=esp.F_wrong_ucb, Fs_ucb=esp.F_split_ucb, cfg=cfg)
            guard_on = eta_g > 1e-9
            guarded = PolicyOutcome(**{**(cdq2 if guard_on else esp).__dict__, "name": "Guarded-CDQ2"})
            # candidates ordered so Guarded-CDQ2 wins ties (the deployable choice)
            r = select_policy(esp, [guarded, cdq2], prof)
            reg_out["profiles"][prof.name] = {
                "selected": r.selected, "benefit": r.benefit, "eligible": list(r.eligible),
                "benefits": r.benefits, "esp_eligible": r.esp_eligible,
                "guard_enabled": guard_on, "eps_w": prof.eps_w, "expected": prof.expected_policy}
            log(f"    {prof.name:20s} -> {r.selected:13s} (B={r.benefit:+.3f}) "
                f"[guard_on={guard_on}] expected={prof.expected_policy}")
        out["regimes"][regime] = reg_out
        json.dump(out, open(OUT, "w"), indent=2)

    # acceptance: in the ENABLE regime, selection should vary rationally; safety-critical -> all ESP.
    en = out["regimes"]["enable"]["profiles"]
    out["acceptance"] = {
        "safety_first_is_esp": en["safety_first"]["selected"] == "ESP",
        "deadline_critical_uses_cdq2": en["deadline_critical"]["selected"] in ("CDQ2", "Guarded-CDQ2"),
        "balanced_is_guarded": en["balanced"]["selected"] == "Guarded-CDQ2",
        "fail_safe_is_esp": en["fail_safe_available"]["selected"] == "ESP",
        "safety_critical_all_esp": all(p["selected"] == "ESP"
                                       for p in out["regimes"]["safety_critical"]["profiles"].values())}
    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    log(f"DONE in {out['runtime_total_s']}s; acceptance={out['acceptance']} -> {OUT}")


def prof_to_service(prof):
    return ConsensusServiceProfile.urban_default().replace(
        k=3, alpha=2, beta=3, max_poll_epochs=R_D,
        max_wrong_basin_probability=prof.eps_w, max_split_basin_probability=prof.eps_s)


if __name__ == "__main__":
    main()
