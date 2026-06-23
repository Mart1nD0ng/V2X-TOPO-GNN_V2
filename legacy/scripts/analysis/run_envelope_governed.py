"""T1 — gradient-governed mixture (fix the D3 cross-density negative transfer).

D3 found a naive 3-density mixture under-fits each density (in-grid retention d100=0.34) while the
2-density LOCO models hit 0.85-0.99. The cause is negative transfer ACROSS the density axis. This
script: (Step 0) runs the GO/NO-GO gradient-conflict diagnostic, then (Step 1-2) trains the all-cells
mixture with per-DENSITY gradient governance (PCGrad de-confliction or GradNorm reweighting) and
re-measures in-grid retention by density vs the naive M_all and the LOCO ceiling.

Usage:
  python -B scripts/analysis/run_envelope_governed.py --governance pcgrad --out result/mixture_governed
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.generalization_common import (  # noqa: E402
    build_topology_layer, density_gradient_conflict, train_model_governed,
)
from scripts.analysis.run_envelope_generalization import (  # noqa: E402
    _PROFILES, _build_env, _cell_cfg, _eval_cell,
)
from src.training.training_smoke import _normalized_config, load_training_smoke_config  # noqa: E402


def _by_density(grid_keys, envs):
    out: dict = {}
    for (d, pr, k) in grid_keys:
        out.setdefault(d, []).append(envs[(d, pr, k)])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Gradient-governed mixture generalist (T1)")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--map", default="result/advantage_map/advantage_map.json")
    p.add_argument("--floor-table", default="result/protocol_floor_table/floor_table.json")
    p.add_argument("--naive", default="result/mixture_generalization/envelope_generalization.json",
                   help="D3 naive-mixture result for the side-by-side comparison")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--steps-per-cell", type=int, default=40)
    p.add_argument("--governance", choices=["pcgrad", "gradnorm"], default="pcgrad")
    p.add_argument("--diagnose-only", action="store_true")
    p.add_argument("--out", default="result/mixture_governed")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    map_json = json.loads((ROOT / args.map).read_text(encoding="utf-8"))
    p0 = {(c["density"], c["profile"], c["coupling_db"]): c for c in map_json["cells"]}
    floor_rows = json.loads((ROOT / args.floor_table).read_text(encoding="utf-8"))["rows"]
    floor_by_profile = {r["profile"]: r["floors"]["small_realistic (k5 a3 b5 r20)"]
                        for r in floor_rows if r["degree"] == 4}

    grid = list(itertools.product([100.0, 200.0, 300.0], _PROFILES, [0.0, 10.0, 20.0]))
    offgrid = list(itertools.product([150.0, 250.0], _PROFILES, [5.0, 15.0]))
    print(f"Building {len(grid)} grid + {len(offgrid)} off-grid envs (N={args.node_count})...", flush=True)
    envs = {}
    for (d, pr, k) in grid + offgrid:
        _, env = _build_env(base, d, pr, k, args.node_count)
        envs[(d, pr, k)] = env
    cfg_ref = _normalized_config(_cell_cfg(base, 200.0, "hard_low_confidence", 10.0, args.node_count))
    eval_q = int(cfg_ref.get("eval_quenched_quadrature", cfg_ref.get("quenched_quadrature", 21)))
    cfg_eval = dict(cfg_ref); cfg_eval["quenched_quadrature"] = eval_q
    topo = build_topology_layer(cfg_ref)
    envs_by_density = _by_density(grid, envs)

    def floor_of(pr): return float(floor_by_profile[pr])
    def expert_of(key): return p0[key]["F_gnn_mean"] if key in p0 else None

    # --- Step 0: GO/NO-GO gradient-conflict diagnostic ---
    print("\n[Step 0] per-density gradient conflict diagnostic (shared init)...", flush=True)
    diag = density_gradient_conflict(cfg_ref, envs_by_density, topo, model_seed=42)
    print("  group_grad_norms:", {f"d{int(k)}": round(v, 5) for k, v in diag["group_grad_norms"].items()})
    print("  pairwise_cosines:", {k: round(v, 4) for k, v in diag["pairwise_cosines"].items()})
    min_cos = min(diag["pairwise_cosines"].values())
    norms = list(diag["group_grad_norms"].values())
    norm_ratio = (max(norms) / max(min(norms), 1e-12))
    if min_cos < -0.05:
        recommend = "pcgrad (directional conflict: min cosine %.3f < 0)" % min_cos
    elif norm_ratio > 3.0:
        recommend = "gradnorm (magnitude imbalance: norm ratio %.1fx)" % norm_ratio
    else:
        recommend = "either (weak conflict; min cosine %.3f, norm ratio %.1fx)" % (min_cos, norm_ratio)
    print(f"  -> diagnostic recommends: {recommend}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.diagnose_only:
        (out_dir / "diagnostic.json").write_text(
            json.dumps({"diagnostic": diag, "recommendation": recommend}, indent=2, sort_keys=True),
            encoding="utf-8")
        print(f"\nwrote {out_dir / 'diagnostic.json'} (diagnose-only)")
        return

    # --- Step 1-2: train governed M_all, eval in-grid + off-grid ---
    steps = args.steps_per_cell * len(envs_by_density[100.0])  # per-cell exposure matched to naive
    print(f"\n[Step 1] training governed mixture (mode={args.governance}, {steps} steps = "
          f"{args.steps_per_cell}/cell)...", flush=True)
    model = train_model_governed(cfg_ref, envs_by_density, topo, steps, model_seed=42, mode=args.governance)

    in_grid, off_grid = [], []
    for (d, pr, k) in grid:
        m = _eval_cell(model, envs[(d, pr, k)], topo, cfg_eval, expert_of((d, pr, k)), floor_of(pr))
        m.update({"density": d, "profile": pr, "coupling_db": k}); in_grid.append(m)
    for (d, pr, k) in offgrid:
        m = _eval_cell(model, envs[(d, pr, k)], topo, cfg_eval, None, floor_of(pr))
        m.update({"density": d, "profile": pr, "coupling_db": k}); off_grid.append(m)

    # --- comparison: governed vs naive M_all vs LOCO ceiling, by density ---
    naive = json.loads((ROOT / args.naive).read_text(encoding="utf-8"))
    naive_ig = naive["arms"]["M_all"]["in_grid"]
    loco = {int(f["held_out_density"]): f["retention_mean"] for f in naive["arms"]["LOCO_leave_density_out"]}

    def ret_by_density(rows, dens):
        r = [c["retention"] for c in rows if c["density"] == dens and c.get("retention") is not None]
        return sum(r) / len(r) if r else None

    print("\n[Step 2] in-grid retention by density (governed vs naive vs LOCO ceiling):", flush=True)
    comparison = {}
    for dens in (100.0, 200.0):
        gov = ret_by_density(in_grid, dens)
        nai = ret_by_density(naive_ig, dens)
        ceil = loco.get(int(dens))
        comparison[f"d{int(dens)}"] = {"governed": gov, "naive": nai, "loco_ceiling": ceil}
        gs = f"{gov:.3f}" if gov is not None else "n/a"
        ns = f"{nai:.3f}" if nai is not None else "n/a"
        cs = f"{ceil:.3f}" if ceil is not None else "n/a"
        print(f"  d{int(dens)}: governed={gs}  naive={ns}  LOCO_ceiling={cs}", flush=True)
    pos_off = sum(1 for c in off_grid if c["gap_model"] > 0.005)
    print(f"  off-grid positive interpolation gap (>0.005): {pos_off}/{len(off_grid)}", flush=True)

    payload = {"currency": f"quenched eval Q={eval_q}", "node_count": args.node_count,
               "governance": args.governance, "steps": steps, "diagnostic": diag,
               "recommendation": recommend, "in_grid": in_grid, "off_grid": off_grid,
               "comparison_by_density": comparison}
    (out_dir / "envelope_governed.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_dir / 'envelope_governed.json'}")


if __name__ == "__main__":
    main()
