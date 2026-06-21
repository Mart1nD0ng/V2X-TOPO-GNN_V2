"""F4.1 — currency faithfulness: the same deployed topology under mean-field (Q=1), quenched (Q=21),
and Monte-Carlo ground truth. Quantifies (a) mean-field optimism and (b) quenched fidelity to MC.

For each cell: build env, train a short screening GNN, deploy its topology, and evaluate F under all
three currencies. Reports per-cell F and the ratios
  mean-field optimism = F_mc / F_meanfield   (how many x MC failure exceeds the mean-field prediction)
  quenched fidelity    = F_mc / F_quenched   (~1.x means quenched tracks MC)

Reuses the advantage-cell MC machinery. Eval-only after a short train, so it is cheap.

Usage:
  python -B scripts/analysis/validate_currency_faithfulness.py \
    --cells 100:hard_low_confidence:20,200:hard_low_confidence:10,300:toy:0 \
    --trials 2000 --out result/currency_faithfulness
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.generalization_common import build_topology_layer, caps_for, env_from_snapshot, train_model  # noqa: E402
from scripts.analysis.run_envelope_generalization import _cell_cfg  # noqa: E402
from scripts.analysis.validate_closed_form_montecarlo import monte_carlo  # noqa: E402
from scripts.analysis.validate_operating_point_montecarlo import _evaluate_details, _select_topology  # noqa: E402
from src.training.training_smoke import _avalanche_config, _normalized_config, load_training_smoke_config  # noqa: E402
from src.v2x_env.profiles import density_matched_vehicle_config  # noqa: E402
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot  # noqa: E402


def _F_at(topo, env, q: int) -> float:
    ev = _evaluate_details(topo, env, quenched_quadrature=q)
    return float((1.0 - ev["avalanche_details"]["node_p_correct_decision"]).mean())


def _F_mc(topo, env, eval_q: int, trials: int, rng) -> float:
    ev = _evaluate_details(topo, env, quenched_quadrature=eval_q)
    av = ev["avalanche_details"]; support = av["query_support"]
    link_s = ev["channel_diagnostics"]["link_success"].detach().to(torch.float64).numpy()
    rw = support.normalized_query_weight.detach().to(torch.float64).numpy() * link_s
    src = support.src_index.detach().numpy(); dst = support.dst_index.detach().numpy()
    ic = env["ic"].detach().to(torch.float64).numpy(); iw = env["iw"].detach().to(torch.float64).numpy()
    ava = _avalanche_config(env["cfg"])
    _c, mc_F, _d = monte_carlo(env["candidate"].num_nodes, src, dst, rw, ic, iw,
                               k=int(ava["k"]), alpha=int(ava["alpha"]), beta=int(ava["beta"]),
                               rounds=int(ava["rounds"]), trials=trials, rng=rng)
    return float(mc_F.mean())


def main() -> None:
    p = argparse.ArgumentParser(description="Currency faithfulness (F4.1)")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--cells", default="100:hard_low_confidence:20,200:hard_low_confidence:10,300:toy:0")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--trials", type=int, default=2000)
    p.add_argument("--scene-seed", type=int, default=42)
    p.add_argument("--out", default="result/currency_faithfulness")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    rng = np.random.default_rng(int(args.scene_seed))
    cells = [tuple(t.split(":")) for t in args.cells.split(",")]
    print(f"Currency faithfulness: {len(cells)} cells, {args.trials} MC trials", flush=True)
    rows = []
    for (d, pr, k) in cells:
        density, coupling = float(d), float(k)
        cfg = _normalized_config(_cell_cfg(base, density, pr, coupling, args.node_count))
        snap = generate_vehicle_snapshot(density_matched_vehicle_config(args.node_count, density, seed=args.scene_seed))
        env = env_from_snapshot(snap, cfg, interference_coupling_db=coupling); env["cfg"] = cfg
        layer = build_topology_layer(cfg); caps = caps_for(env, cfg)
        eval_q = int(cfg.get("eval_quenched_quadrature", cfg.get("quenched_quadrature", 21)))
        model = train_model(cfg, [env], layer, int(args.steps), model_seed=42)
        with torch.no_grad():
            topo = _select_topology(model, env, layer, caps)
        f_mf = _F_at(topo, env, 1)
        f_q = _F_at(topo, env, eval_q)
        f_mc = _F_mc(topo, env, eval_q, args.trials, rng)
        # guard: below this, mean-field has effectively underflowed to 0 (qualitatively blind, ratio meaningless)
        mf_floor = 1e-5
        mf_opt = (f_mc / f_mf) if f_mf > mf_floor else None
        row = {"density": density, "profile": pr, "coupling_db": coupling,
               "F_meanfield": f_mf, "F_quenched": f_q, "F_mc": f_mc,
               "meanfield_optimism_x": mf_opt,
               "meanfield_blind": bool(f_mf <= mf_floor),
               "quenched_fidelity_x": (f_mc / f_q) if f_q > 0 else None}
        rows.append(row)
        opt_s = f"{mf_opt:.1f}x optimistic" if mf_opt is not None else "predicts ~0 (blind)"
        print(f"  d{int(density)}/{pr}/c{int(coupling)} | F_mf={f_mf:.5f} F_q={f_q:.4f} F_mc={f_mc:.4f}"
              f" | mean-field {opt_s} | quenched {row['quenched_fidelity_x']:.2f}x of MC", flush=True)

    mf = [r["meanfield_optimism_x"] for r in rows if r["meanfield_optimism_x"]]
    qf = [r["quenched_fidelity_x"] for r in rows if r["quenched_fidelity_x"]]
    n_blind = sum(1 for r in rows if r.get("meanfield_blind"))
    out_dir = ROOT / args.out; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "currency_faithfulness.json").write_text(
        json.dumps({"trials": args.trials, "node_count": args.node_count, "cells": rows,
                    "meanfield_optimism_range": [min(mf), max(mf)] if mf else None,
                    "meanfield_blind_cells": n_blind,
                    "quenched_fidelity_range": [min(qf), max(qf)] if qf else None}, indent=2, sort_keys=True),
        encoding="utf-8")
    mfs = f"{min(mf):.0f}-{max(mf):.0f}x" + (f" (+{n_blind} blind)" if n_blind else "") if mf else f"all {n_blind} blind"
    print(f"\nmean-field optimism: {mfs} | quenched fidelity: {min(qf):.2f}-{max(qf):.2f}x of MC")
    print(f"wrote {out_dir / 'currency_faithfulness.json'}")


if __name__ == "__main__":
    main()
