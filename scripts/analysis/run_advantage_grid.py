"""Cross-domain advantage grid: WHERE (over density x interference-coupling) does the proposed
learned constructor beat the best heuristic, on reliability F AND on cost (delay D)?

For each (density, coupling) cell on the re-calibrated operating-point physics (axis-visibility LOS,
-80 dBm floor, cap 8), train the proposed hierarchical GNN (reliability objective, a few seeds) and
the heuristic set through the SAME constructor + evaluator on held-out scenes. Record:
  F-advantage   = best_heuristic_F - proposed_F      (>0  => proposed more reliable)
  cost-advantage = best_heuristic_D / proposed_D     (>1  => proposed lower delay)

Renders two heatmaps (density rows x coupling cols). The cost-advantage map is the robust headline
(the F edge is modest under MC; cost is large and currency-robust, per the MC audit).

Usage:
  python -B scripts/analysis/run_advantage_grid.py --densities 100,200,300 --couplings 10,20,30 \
    --node-count 600 --steps 120 --seeds 7,42 --run-name advantage_grid_v1
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.run_baseline_comparison import _heuristic_scores, _model_score  # noqa: E402
from scripts.analysis.run_baseline_comparison_v2 import _train  # noqa: E402
from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer  # noqa: E402
from src.training.training_smoke import load_training_smoke_config  # noqa: E402


def _cell_base(base, density, coupling):
    cb = dict(base)
    cb["node_density_per_km2"] = float(density)
    phys = dict(cb.get("physical", {}))
    phys["interference_density_coupling_db"] = float(coupling)
    cb["physical"] = phys
    return cb


def _eval_score_over(score_fn, eval_envs, eval_layers):
    Fs, Ds = [], []
    for env, (el, ec) in zip(eval_envs, eval_layers):
        with torch.no_grad():
            m = _metrics(_forward(score_fn(env).reshape(-1), env, el, ec, eval_mode=True))
        Fs.append(m["F"]); Ds.append(m["D"])
    return float(np.mean(Fs)), float(np.mean(Ds))


def run_cell(base, density, coupling, node_count, steps, seeds, train_seed, eval_scenes):
    cb = _cell_base(base, density, coupling)
    eval_seeds = [train_seed + 1000 + i for i in range(eval_scenes)]
    train_env = _scene_env(cb, node_count, train_seed)
    cfg = train_env["cfg"]
    layer, caps = _topology_layer(cfg, train_env["candidate"].num_nodes)
    eval_envs = [_scene_env(cb, node_count, s) for s in eval_seeds]
    eval_layers = [_topology_layer(e["cfg"], e["candidate"].num_nodes) for e in eval_envs]

    # proposed: median over seeds of held-out mean F / D
    pf, pd = [], []
    for s in seeds:
        model = _train("proposed", cfg, train_env, layer, caps, steps, s, 64)
        Fs, Ds = [], []
        for env, (el, ec) in zip(eval_envs, eval_layers):
            with torch.no_grad():
                m = _metrics(_forward(_model_score(model, env), env, el, ec, eval_mode=True))
            Fs.append(m["F"]); Ds.append(m["D"])
        pf.append(float(np.mean(Fs))); pd.append(float(np.mean(Ds)))
    prop_F, prop_D = float(np.median(pf)), float(np.median(pd))

    # best heuristic by F (and its D)
    hres = {}
    for label in _heuristic_scores(eval_envs[0]).keys():
        f, d = _eval_score_over(lambda e, _l=label: _heuristic_scores(e)[_l], eval_envs, eval_layers)
        hres[label] = (f, d)
    best_h = min(hres, key=lambda k: hres[k][0])
    bh_F, bh_D = hres[best_h]
    return {"density": density, "coupling_db": coupling, "proposed_F": prop_F, "proposed_D": prop_D,
            "best_heuristic": best_h, "best_heuristic_F": bh_F, "best_heuristic_D": bh_D,
            "F_advantage": bh_F - prop_F, "cost_advantage_x": bh_D / max(prop_D, 1e-9)}


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-domain advantage grid (density x coupling)")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--densities", default="100,200,300")
    p.add_argument("--couplings", default="10,20,30")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--seeds", default="7,42")
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--eval-scenes", type=int, default=4)
    p.add_argument("--run-name", default="advantage_grid_v1")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    densities = [float(x) for x in args.densities.split(",") if x.strip()]
    couplings = [float(x) for x in args.couplings.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    print(f"Advantage grid [{args.config}] densities={densities} couplings={couplings} "
          f"N={args.node_count} steps={args.steps} seeds={seeds}", flush=True)

    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    # resumable per-cell cache: a killed run (OOM / session) re-runs only the MISSING cells.
    cache_path = out_dir / "cells.jsonl"
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cc = json.loads(line)
                done[(float(cc["density"]), float(cc["coupling_db"]))] = cc

    cells = []
    for d in densities:
        for c in couplings:
            key = (float(d), float(c))
            if key in done:
                r = done[key]
                print(f"  [cached] d={d:.0f} c={c:.0f}dB: F-adv={r['F_advantage']:+.4f} "
                      f"cost-adv={r['cost_advantage_x']:.1f}x", flush=True)
            else:
                r = run_cell(base, d, c, args.node_count, args.steps, seeds, args.train_seed, args.eval_scenes)
                with cache_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(r, sort_keys=True) + "\n"); fh.flush()
                import gc
                gc.collect()
                print(f"  d={d:.0f} c={c:.0f}dB: proposed F={r['proposed_F']:.4f} D={r['proposed_D']:.1f} | "
                      f"bestH({r['best_heuristic']}) F={r['best_heuristic_F']:.4f} D={r['best_heuristic_D']:.1f} | "
                      f"F-adv={r['F_advantage']:+.4f} cost-adv={r['cost_advantage_x']:.1f}x", flush=True)
            cells.append(r)
    (out_dir / "advantage_grid.json").write_text(json.dumps(
        {"config": args.config, "densities": densities, "couplings": couplings, "seeds": seeds,
         "node_count": args.node_count, "steps": args.steps, "cells": cells}, indent=2, sort_keys=True),
        encoding="utf-8")

    # heatmaps: rows=density (top=high), cols=coupling
    def grid_of(key):
        G = np.full((len(densities), len(couplings)), np.nan)
        for r in cells:
            i = densities.index(r["density"]); j = couplings.index(r["coupling_db"])
            G[i, j] = r[key]
        return G
    Fadv = grid_of("F_advantage"); Cadv = grid_of("cost_advantage_x")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, G, title, cmap, fmt in (
        (axes[0], Fadv, "Reliability advantage\n($F_{bestH} - F_{proposed}$, >0 = proposed better)", "RdBu", "{:+.3f}"),
        (axes[1], Cadv, "Cost advantage\n($D_{bestH}/D_{proposed}$, >1 = proposed lower delay)", "viridis", "{:.1f}x")):
        im = ax.imshow(G, aspect="auto", origin="lower", cmap=cmap,
                       vmin=(-abs(np.nanmax(np.abs(Fadv))) if G is Fadv else 1.0),
                       vmax=(abs(np.nanmax(np.abs(Fadv))) if G is Fadv else np.nanmax(Cadv)))
        ax.set_xticks(range(len(couplings))); ax.set_xticklabels([f"{int(c)}" for c in couplings])
        ax.set_yticks(range(len(densities))); ax.set_yticklabels([f"{int(d)}" for d in densities])
        ax.set_xlabel("interference coupling (dB)"); ax.set_ylabel("density (veh/km$^2$)")
        ax.set_title(title, fontsize=9)
        for i in range(len(densities)):
            for j in range(len(couplings)):
                if not np.isnan(G[i, j]):
                    ax.text(j, i, fmt.format(G[i, j]), ha="center", va="center", fontsize=8,
                            color=("black"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Cross-domain advantage of the proposed constructor vs the best heuristic "
                 "(axis-visibility operating-point physics)", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_dir / "figures" / "advantage_grid.png", dpi=140); plt.close(fig)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
