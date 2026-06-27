"""G-FINAL-SYNTHESIS figures (plan §8 task 3).

Produces the round's figures STRICTLY by READING the committed result JSONs -- it never recomputes a
metric through a separate evaluator (constraint #13). Each input is checked with the namespace
figure-guard (``assert_no_legacy_metrics``) so a legacy/surrogate metric can never reach a figure.
Always writes ``figures_data.json`` (plot-ready series); also renders PNGs if matplotlib is available.

Figures: (1) eta risk-liveness curves, (2) Guarded-CDQ2 wrong-vs-deadline frontier, (3) hazard-profile
policy map, (4) ESP scale curve. Run: PYTHONPATH=. python docs/gate_evidence/guarded_cdq2/make_figures.py
"""
from __future__ import annotations

import json
import os

from src.metrics import schema

HERE = os.path.dirname(__file__)


def _read(name):
    with open(os.path.join(HERE, name)) as f:
        d = json.load(f)
    schema.assert_no_legacy_metrics(d)        # figure guard: no legacy/surrogate metric may enter a figure
    return d


def eta_curve_series():
    d = _read("eta_risk_liveness_results.json")
    etas = [float(x) for x in d["eta_grid"]]
    out = {}
    for key, cell in d["cells"].items():
        out[key] = {"eta": etas,
                    "macro_F_deadline": [cell["etas"][f"{e:g}"]["macro"]["macro_F_deadline"] for e in etas],
                    "macro_F_wrong": [cell["etas"][f"{e:g}"]["macro"]["macro_F_wrong"] for e in etas],
                    "macro_P_correct": [cell["etas"][f"{e:g}"]["macro"]["macro_P_correct"] for e in etas]}
    # the deadline-window sensitivity (the headline of GS4), if present
    try:
        s = _read("eta_deadline_sensitivity_results.json")
        out["_sensitivity"] = {rd: {"eta": [float(x) for x in s["etas"]],
                                    "macro_F_deadline": [v["macro_F_deadline"]
                                                         for v in cell["macro_blocks"].values()]}
                               for rd, cell in s["deadlines"].items()}
    except FileNotFoundError:
        pass
    return out


def guarded_frontier_series():
    d = _read("guarded_cdq2_results.json")
    out = {}
    for rk, reg in d["regimes"].items():
        pts = {"ESP": reg["ESP"], "CDQ2": reg["fixed_eta"]}      # GS5 stores the fixed-eta arm as "fixed_eta"
        out[rk] = {name: {"macro_F_wrong": m["macro_F_wrong"], "macro_F_deadline": m["macro_F_deadline"]}
                   for name, m in pts.items()}
    return out


def hazard_policy_map():
    d = _read("hazard_profiles_results.json")
    out = {}
    for rk, reg in d["regimes"].items():
        out[rk] = {pname: p["selected"] for pname, p in reg["profiles"].items()}
    return out


def esp_scale_series():
    d = _read("esp_performance_scale_results.json")
    out = {"shared_P_correct": {}, "contrast": {}}
    for label, cell in d.get("tiers", {}).items():
        rec = cell.get("esd_gnn_shared", {}).get("record", {})
        if rec:
            out["shared_P_correct"][label] = rec["macro"]["macro_P_correct"]
    for label, row in d.get("contrast", {}).items():
        out["contrast"][label] = {mode: {"R_d": r["R_d"], "macro_P_correct": r["macro"]["macro_P_correct"],
                                         "macro_F_deadline": r["macro"]["macro_F_deadline"]}
                                  for mode, r in row.items()}
    return out


def main():
    data = {"eta_curve": eta_curve_series(), "guarded_frontier": guarded_frontier_series(),
            "hazard_policy_map": hazard_policy_map(), "esp_scale": esp_scale_series()}
    assert not schema.forbidden_keys_in(data), schema.forbidden_keys_in(data)
    out_path = os.path.join(HERE, "figures_data.json")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {out_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                       # pragma: no cover - matplotlib optional
        print(f"matplotlib unavailable ({e}); plot-ready data written, PNGs skipped.")
        return

    # (1) eta risk-liveness: F_deadline vs eta per env (full sweep) + the R_d sensitivity
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, s in data["eta_curve"].items():
        if key.startswith("_"):
            continue
        ax.plot(s["eta"], s["macro_F_deadline"], marker="o", label=key)
    ax.set_xlabel("eta"); ax.set_ylabel("macro_F_deadline"); ax.set_xscale("symlog")
    ax.set_title("eta risk-liveness (deadline basin)"); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_eta_curve.png"), dpi=120); plt.close(fig)

    # (4) ESP scale: shared-checkpoint P_correct across scales
    sc = data["esp_scale"]["shared_P_correct"]
    if sc:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(list(sc), list(sc.values()))
        ax.set_ylabel("macro_P_correct (shared ckpt)"); ax.set_ylim(0, 1)
        ax.set_title("ESP/ESD-GNN scale-transfer"); fig.tight_layout()
        fig.savefig(os.path.join(HERE, "fig_esp_scale.png"), dpi=120); plt.close(fig)
    print("rendered fig_eta_curve.png, fig_esp_scale.png")


if __name__ == "__main__":
    main()
