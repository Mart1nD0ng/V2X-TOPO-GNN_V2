"""Campaign A, Phase A6 -- G-ESP-SCALE-SYNTHESIS (the section-13 verdict + figures).

READS the committed result JSONs ONLY (A1 headline, A2 budget, A3b scale, A4 OOD, A5 rare-event) and emits a
synthesis JSON + figures. It does NOT run any MC or recompute any metric (constraint: figure/synthesis scripts
read result JSONs only). The verdict logic (workflow section 13):
  * 13.1 SUPERIORITY: shared-ESP (trained@N=120) CI-separately ABOVE distance at one+ scales (esp. large N /
    deadline collapse);
  * 13.2 PARITY: shared tracks distance (distance point within the shared seed-level CI) across scales AND
    CI-separately beats uniform -- a stable learned constructor matching the strongest heuristic;
  * 13.3 LOSES: shared below distance/uniform.
Per (N, mode): shared_lo > dist -> 'shared_above'; shared_hi < dist -> 'shared_below'; else 'parity'
(distance is a point estimate with its own MC uncertainty -- noted, not over-claimed).

Run: PYTHONPATH=. python docs/gate_evidence/esp_scale_v2/run_phase_a6_synthesis.py
"""
from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.metrics import schema

HERE = os.path.dirname(__file__)
FIGDIR = os.path.join(HERE, "figures")


def _load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _classify(shared_ci, dist_pc):
    lo, hi = shared_ci
    if lo > dist_pc:
        return "shared_above"
    if hi < dist_pc:
        return "shared_below"
    return "parity"


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    a1 = _load("phase_a1_eval_results.json")
    a2 = _load("phase_a2_budget_results.json")
    a3b = _load("phase_a3b_scale_results.json")
    a4 = _load("phase_a4_ood_results.json")
    a5 = _load("phase_a5_rare_event_results.json")
    out = {"metric_namespace_version": "macrostate_v2", "experiment_family": "esp_performance_scale_v2",
           "experiment": "phase_a6_synthesis", "query_family": "ESP",
           "reads_only": ["phase_a1_eval_results.json", "phase_a2_budget_results.json",
                          "phase_a3b_scale_results.json", "phase_a4_ood_results.json",
                          "phase_a5_rare_event_results.json"],
           "inputs_present": {k: v is not None for k, v in
                              [("a1", a1), ("a2", a2), ("a3b", a3b), ("a4", a4), ("a5", a5)]}}

    # ---- N=120 headline (A1)
    if a1:
        h = a1["headline"]
        out["headline_N120"] = {"trained": h["trained_seed_mean_P_correct"],
                                "trained_ci": h["trained_seed_bootstrap_ci"],
                                "uniform": h["uniform_P_correct"], "distance": h["distance_P_correct"],
                                "separated_above_uniform": h["separated_above_uniform_conservative"],
                                "gap_to_distance": h["gap_to_distance"]}

    # ---- budget curve (A2)
    if a2:
        out["budget_curve"] = {str(b): a2["budget_curve"][str(b)]["seed_mean_P_correct"]
                               for b in a2["budgets"]}
        out["budget_rises_vs_ev1_flat"] = a2["beats_ev1_flat"] and a2["monotone_nondecreasing"]

    # ---- scale sweep (A3b) -> the central verdict
    scale_rows = []
    if a3b:
        for N in sorted(a3b["cells"], key=int):
            blk = a3b["cells"][N]
            for mode, cell in blk["modes"].items():
                p = cell["policies"]
                if "shared_esp" not in p:
                    continue
                sh = p["shared_esp"]
                dist_pc = p.get("distance", {}).get("macro_P_correct")
                uni_pc = p.get("uniform_esp", {}).get("macro_P_correct")
                # a cell is DEGENERATE if ALL policies collapse to ~0 (total deadline) -> no differentiation;
                # in this ladder only N=1248 grid(13,13,4), the lone v=4 (high-density) grid, where the MAC
                # saturates and F_deadline->1 for shared, distance AND uniform alike (even at fixed_service
                # R_d=19) -- a density/MAC property, NOT a GNN failure. Excluded from the parity verdict.
                degenerate = (sh["seed_mean_P_correct"] < 0.05 and (dist_pc is None or dist_pc < 0.05)
                              and (uni_pc is None or uni_pc < 0.05))
                row = {"N": int(N), "mode": mode, "R_d": cell["R_d"],
                       "shared": sh["seed_mean_P_correct"], "shared_ci": sh["seed_bootstrap_ci"],
                       "n_seeds": sh.get("n_seeds"), "distance": dist_pc, "uniform": uni_pc,
                       "expert": p.get("expert", {}).get("macro_P_correct"),
                       "scale_regret_shared_vs_expert": cell.get("scale_regret_shared_vs_expert"),
                       "compute_limited": blk["compute_limited"], "approximation_bound": blk["approximation_bound"],
                       "degenerate_all_deadline": degenerate,
                       "vs_distance": None if degenerate else (_classify(sh["seed_bootstrap_ci"], dist_pc) if dist_pc is not None else None),
                       "shared_beats_uniform": (not degenerate and uni_pc is not None and sh["seed_bootstrap_ci"][0] > uni_pc)}
                scale_rows.append(row)
        out["scale_rows"] = scale_rows
        out["degenerate_cells"] = sorted({r["N"] for r in scale_rows if r["degenerate_all_deadline"]})
        out["degenerate_note"] = ("N=1248 is the lone v=4 (high-density) grid; ALL policies F_deadline->1 "
                                  "(MAC saturation), so it is excluded from the parity verdict -- the "
                                  "controlled-density scale story is the v=3 ladder 120/336/660/3036(/9840 bound).")
        # aggregate over the NON-approximation, NON-degenerate cells (exclude N=9840 bound + N=1248 collapse)
        solid = [r for r in scale_rows if not r["approximation_bound"] and not r["degenerate_all_deadline"]
                 and r["vs_distance"]]
        above = sum(r["vs_distance"] == "shared_above" for r in solid)
        parity = sum(r["vs_distance"] == "parity" for r in solid)
        below = sum(r["vs_distance"] == "shared_below" for r in solid)
        beats_uni = sum(bool(r["shared_beats_uniform"]) for r in solid)
        out["scale_summary"] = {"n_cells": len(solid), "shared_above_distance": above, "parity": parity,
                                "shared_below_distance": below, "shared_beats_uniform": beats_uni}

    # ---- OOD (A4)
    if a4:
        out["ood"] = {cid: {"trained": c["trained_seed_mean_P_correct"], "distance": c["distance_P_correct"],
                            "uniform": c["uniform_P_correct"], "beats_uniform": c["trained_vs_uniform_paired"]["trained_better"],
                            "axis": c["axis"]} for cid, c in a4["cells"].items()}

    # ---- rare-event (A5)
    if a5:
        out["rare_event"] = a5["headline"]

    # ---- overall section-13 verdict
    verdict = "insufficient_data"
    if a3b and out.get("scale_summary"):
        s = out["scale_summary"]
        above, parity, below, n = (s["shared_above_distance"], s["parity"], s["shared_below_distance"],
                                   max(1, s["n_cells"]))
        beats_uni_majority = s["shared_beats_uniform"] >= (n + 1) // 2
        if below > above + parity:
            # shared falls behind distance at most scales
            verdict = "13.3_shared_below_distance_at_scale"
        elif above > below + parity:
            # shared CI-separately ABOVE distance at a clear majority of scales
            verdict = "13.1_superiority_shared_above_distance_at_most_scales"
        elif beats_uni_majority:
            # mixed above/below around distance (= tracks it) AND consistently beats uniform
            verdict = "13.2_parity_stable_constructor_matches_distance_beats_uniform"
        else:
            verdict = "13.3_shared_does_not_beat_uniform_at_scale"
    out["section_13_verdict"] = verdict
    out["verdict_note"] = ("distance is a point estimate (its MC CI not stored in A3b); 'parity' = distance "
                           "point within the shared seed-level bootstrap CI. N=9840 excluded from the verdict "
                           "(approximation bound, 8 trials). N>=660 cells are compute-limited (3/2 seeds).")

    # ---- figures (read-only) ----
    figs = []
    if a2:
        budgets = [int(b) for b in a2["budgets"]]
        ys = [a2["budget_curve"][str(b)]["seed_mean_P_correct"] for b in budgets]
        los = [a2["budget_curve"][str(b)]["seed_bootstrap_ci"][0] for b in budgets]
        his = [a2["budget_curve"][str(b)]["seed_bootstrap_ci"][1] for b in budgets]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(budgets, ys, "o-", label="MC-faithful (held-out MC)")
        ax.fill_between(budgets, los, his, alpha=0.2)
        ax.axhline(a2["ev1_flat_analytic_contrast"], ls="--", color="gray",
                   label=f"EV1 analytic surrogate (FLAT {a2['ev1_flat_analytic_contrast']})")
        ax.set_xlabel("training budget (REINFORCE steps)"); ax.set_ylabel("macro_P_correct (held-out MC)")
        ax.set_title("A2: budget curve RISES under MC-faithful training (N=120)"); ax.legend(fontsize=8)
        f = os.path.join(FIGDIR, "a2_budget_curve.png"); fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)
        figs.append(os.path.relpath(f, HERE))
    if scale_rows:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for mode, mk in [("fixed_protocol", "o-"), ("fixed_service_profile", "s--")]:
            rows = [r for r in scale_rows if r["mode"] == mode and not r["degenerate_all_deadline"]]
            if not rows:
                continue
            Ns = [r["N"] for r in rows]
            ax.plot(Ns, [r["shared"] for r in rows], mk, label=f"shared-ESP ({mode})")
            ax.plot(Ns, [r["distance"] for r in rows], mk, alpha=0.5, label=f"distance ({mode})")
        ax.set_xscale("log"); ax.set_xlabel("N (nodes, log)"); ax.set_ylabel("macro_P_correct (held-out MC)")
        ax.set_title("A3b: scale sweep -- shared-ESP (trained@120) vs distance"); ax.legend(fontsize=7)
        f = os.path.join(FIGDIR, "a3b_scale_curve.png"); fig.tight_layout(); fig.savefig(f, dpi=120); plt.close(fig)
        figs.append(os.path.relpath(f, HERE))
    out["figures"] = figs

    assert not schema.forbidden_keys_in(out), schema.forbidden_keys_in(out)
    json.dump(out, open(os.path.join(HERE, "phase_a6_synthesis.json"), "w"), indent=2)
    print(f"DONE: verdict={verdict}; scale_summary={out.get('scale_summary')}; figures={figs}")


if __name__ == "__main__":
    main()
