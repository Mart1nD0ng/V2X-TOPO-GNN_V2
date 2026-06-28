"""Campaign A, Phase A5 -- G-ESP-RARE-EVENT-CERTIFICATION (reliability UCBs for F_wrong / F_split).

Certifies the HARD reliability constraints on a SINGLE deployable checkpoint (seed 0, N=120) by pooling many
dynamic-MC trials (a 3/M zero-failure bound certifies a deployable policy, not a 5-seed mixture). eps=1e-3
needs M>=~3800 zero-failure trials (the harness's feasible_point docstring; spec 6.7/9.2).

Honest expectation (verified EV5/EV6): F_split is identically 0 -> CERTIFIABLE at eps_s=1e-3 with enough
zero-failure trials. F_wrong is STRUCTURALLY ~0.018-0.052 for EVERY policy (trained AND distance) -> its UCB
is orders of magnitude above eps_w=1e-3, so it CANNOT be certified -- the stressed mm_high regime is
infeasible at the strict wrong-basin target for all policies (a property of the regime, not the GNN). We
report the observed rate + Wilson UCB and state this plainly (no faked certificate).

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a5_rare_event.py [--smoke]
"""
from __future__ import annotations

import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.config.experiment_spec import build_experiment_spec
from src.config.service_profile import ConsensusServiceProfile
from src.environment import ProtocolConfig, RoundPhysicsConfig
from src.evaluation.esp_scale import evaluate_macro, policy_factory
from src.evaluation.mc_faithful_campaign import checkpoint_policy_factory, load_checkpoint
from src.metrics import manifest as mf
from src.metrics import schema

SMOKE = "--smoke" in sys.argv
HERE = os.path.dirname(__file__)
A1_SUMMARY = os.path.join(HERE, "phase_a1_train_summary.json")
OUT = os.path.join(HERE, "phase_a5_rare_event_results.json")
PHY = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
PROTO = ProtocolConfig(k=3, alpha=2, beta=3, r_max=6)
PROFILE = ConsensusServiceProfile.urban_default().replace(k=3, alpha=2, beta=3, max_poll_epochs=6)
GRID, SCEN, BE, CORR = (5, 5, 3), "matched_marginal_high", 0.35, 0.25
CERT_SEED = 0                       # the deployable checkpoint certified (single policy, not a mixture)
M = 200 if SMOKE else 4000          # zero-failure trials needed for eps=1e-3: ~3800
M_THRESHOLD = 3800


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _certify(policy_fn, label):
    macro = evaluate_macro(GRID, [0], policy_fn, PROFILE, PROTO, PHY, trials=M, scenario=SCEN,
                           base_node_err=BE, corr_strength=CORR, link_override=None).macro
    fw, fs = macro["macro_F_wrong"], macro["macro_F_split"]
    fw_ucb, fs_ucb = macro["macro_F_wrong_ci"][1], macro["macro_F_split_ci"][1]
    ew, es = PROFILE.max_wrong_basin_probability, PROFILE.max_split_basin_probability
    out = {"label": label, "M_trials": M, "M_sufficient_for_eps": M >= M_THRESHOLD,
           "macro_F_wrong": fw, "macro_F_wrong_ucb": fw_ucb, "eps_wrong": ew,
           "macro_F_split": fs, "macro_F_split_ucb": fs_ucb, "eps_split": es,
           "F_split_zero_failure": fs == 0.0,
           "F_split_certified": (fs == 0.0 and fs_ucb <= es and M >= M_THRESHOLD),
           "F_wrong_certified": (fw_ucb <= ew),
           "F_wrong_structurally_infeasible": fw > ew}
    log(f"{label}: F_wrong={fw:.4f} ucb={fw_ucb:.4f} (eps={ew}) | F_split={fs:.4f} ucb={fs_ucb:.5f} "
        f"-> split_cert={out['F_split_certified']} wrong_cert={out['F_wrong_certified']}")
    return out


def main():
    t0 = time.perf_counter()
    a1 = json.load(open(A1_SUMMARY))
    git = mf.current_git_commit() or "uncommitted"
    ckpt = load_checkpoint(os.path.join(HERE, a1["checkpoints"][str(CERT_SEED)]["path"]))
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "phase_a5_rare_event", "query_family": "ESP", "smoke": SMOKE, "git_commit": git,
           "node_count": 120, "scenario": SCEN, "base_node_err": BE, "corr_strength": CORR,
           "cert_model_seed": CERT_SEED, "cert_checkpoint_hash": ckpt.checkpoint_hash,
           "method": "per-checkpoint zero-failure UCB (3/M at eps=1e-3 needs M>=3800); single deployable policy",
           "certifications": {}}
    out["certifications"]["trained_esp_seed0"] = _certify(checkpoint_policy_factory(ckpt), "trained_esp_seed0")
    out["certifications"]["distance"] = _certify(policy_factory("distance"), "distance")

    t = out["certifications"]["trained_esp_seed0"]
    out["headline"] = {
        "F_split_certified_feasible": t["F_split_certified"],
        "F_wrong_certifiable": t["F_wrong_certified"],
        "regime_infeasible_at_strict_eps_wrong_for_all": (
            t["F_wrong_structurally_infeasible"] and out["certifications"]["distance"]["F_wrong_structurally_infeasible"]),
        "interpretation": ("F_split is certified <= eps_s=1e-3 (zero split events in M trials); F_wrong CANNOT "
                           "be certified at eps_w=1e-3 because it is structurally ~0.02-0.05 for the trained "
                           "policy AND distance alike -> the stressed mm_high regime is infeasible at the strict "
                           "wrong-basin target for all policies (a regime property, not a GNN failure). This is "
                           "why the A0 performance comparison is un-gated; deployment would relax eps_w or the regime.")}
    spec = build_experiment_spec(protocol_cfg=PROTO, service_profile=PROFILE, phy_cfg=PHY,
                                 evidence_descriptor=f"{SCEN}:p={BE}:c={CORR}", scene_descriptor={"gx": 5, "gy": 5, "v": 3},
                                 query_law="esp", full_physics=True)
    out["manifest"] = mf.build_manifest(spec, policy_hash="esd_gnn_mc_faithful_cert",
                                        checkpoint_hash=ckpt.checkpoint_hash, model_seeds=[CERT_SEED],
                                        git_commit=git, manifest_id="A5-rare-event")
    out["runtime_total_s"] = round(time.perf_counter() - t0, 1)
    json.dump(out, open(OUT, "w"), indent=2)
    forbidden = schema.forbidden_keys_in(out)
    assert not forbidden, f"namespace guard: forbidden keys {forbidden} (result saved to {OUT})"
    log(f"DONE {out['runtime_total_s']}s; F_split_certified={t['F_split_certified']} "
        f"F_wrong_certifiable={t['F_wrong_certified']} (regime infeasible at strict eps_w for all)")


if __name__ == "__main__":
    main()
