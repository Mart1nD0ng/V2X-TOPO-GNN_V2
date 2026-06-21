"""Cost-controllability sub-experiment: only LEARNED methods can trade reliability for cost.

On the operating point (where D/E have live levers), sweep the cost weight w for the learned methods
(proposed hierarchical GNN, vanilla GraphSAGE) -> each traces an (F, D) front. The heuristics and the
flood baseline have NO cost knob -> each is a single point. The figure makes the contribution visual:
the learned constructor moves along a reliability-cost frontier; hand-rules cannot.

All methods routed through the SAME constructor + analytic evaluator on the SAME held-out scenes.
F = failure (lower better), D = effective delay rounds (lower better), eval at the quenched currency.

Usage:
  python -B scripts/analysis/run_baseline_cost_controllability.py --config configs/operating_point_v1.yaml \
    --node-count 600 --weights 0,0.1,0.5,2 --seeds 7,42,123 --run-name baseline_cost_controllability
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

from scripts.analysis.run_baseline_comparison import VanillaGNNScorer, _heuristic_scores, _model_score  # noqa: E402
from scripts.analysis.run_baseline_comparison_v2 import _flood_layer, _oracle_score  # noqa: E402
from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer, ci95  # noqa: E402
from src.losses import compute_coupled_loss  # noqa: E402
from src.training.training_smoke import _loss_config, _make_model, load_training_smoke_config  # noqa: E402

C_PROP = "#0072B2"; C_VAN = "#009E73"; C_HEUR = "#D55E00"; C_FLOOD = "#999999"


def _train_cost(kind, cfg, env, layer, caps, max_steps, seed, w, hidden, rel_target):
    torch.manual_seed(int(seed))
    model = _make_model(cfg) if kind == "proposed" else VanillaGNNScorer(5, 5, hidden=hidden, layers=2).double()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_cfg = dict(_loss_config(cfg))
    loss_cfg["weight_reliability"] = 1.0
    loss_cfg["weight_delay"] = float(w); loss_cfg["weight_energy"] = float(w)
    loss_cfg["reliability_failure_target"] = float(rel_target)
    loss_cfg["reliability_tail_failure_target"] = float(rel_target)
    for _ in range(int(max_steps)):
        opt.zero_grad(set_to_none=True)
        ev = _forward(_model_score(model, env), env, layer, caps)
        lo = compute_coupled_loss(ev, loss_cfg)
        (lo.get("effective_backward_loss", lo["total_loss"])).backward()
        opt.step()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Cost-controllability: learned fronts vs heuristic points")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--max-steps", type=int, default=140)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--seeds", default="7,42,123")
    p.add_argument("--weights", default="0,0.1,0.5,2")
    p.add_argument("--eval-scenes", type=int, default=4)
    p.add_argument("--reliability-target", type=float, default=0.02)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--run-name", default="baseline_cost_controllability")
    args = p.parse_args()

    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    weights = [float(w) for w in args.weights.split(",") if w.strip()]
    eval_seeds = [args.train_seed + 1000 + i for i in range(int(args.eval_scenes))]

    train_env = _scene_env(base, args.node_count, args.train_seed)
    cfg = train_env["cfg"]
    layer, caps = _topology_layer(cfg, train_env["candidate"].num_nodes)
    eval_envs = [_scene_env(base, args.node_count, s) for s in eval_seeds]
    eval_layers = [_topology_layer(e["cfg"], e["candidate"].num_nodes) for e in eval_envs]
    eval_floods = [_flood_layer(e["cfg"], e["candidate"].num_nodes) for e in eval_envs]
    print(f"Cost-controllability [{args.config}] N={args.node_count} weights={weights} seeds={seeds}", flush=True)

    def eval_model(model):
        Fs, Ds = [], []
        for env, (el, ec) in zip(eval_envs, eval_layers):
            with torch.no_grad():
                m = _metrics(_forward(_model_score(model, env), env, el, ec, eval_mode=True))
            Fs.append(m["F"]); Ds.append(m["D"])
        return float(np.mean(Fs)), float(np.mean(Ds))

    learned = {}
    for label, kind, col in (("proposed (hierarchical)", "proposed", C_PROP),
                             ("vanilla GraphSAGE", "vanilla", C_VAN)):
        front = []
        for w in weights:
            Fs, Ds = [], []
            for s in seeds:
                model = _train_cost(kind, cfg, train_env, layer, caps, args.max_steps, s, w,
                                    args.hidden_dim, args.reliability_target)
                f, d = eval_model(model)
                Fs.append(f); Ds.append(d)
            front.append({"w": w, "F": float(np.median(Fs)), "D": float(np.median(Ds)),
                          "F_all": Fs, "D_all": Ds})
            print(f"  {label} w={w}: F(med)={front[-1]['F']:.4f} D(med)={front[-1]['D']:.1f}", flush=True)
        learned[label] = {"color": col, "front": front}

    # heuristics + flood: single points (no cost knob)
    points = {}
    def eval_score(score_fn, flood=False):
        Fs, Ds = [], []
        for env, (el, ec), (fl, fc) in zip(eval_envs, eval_layers, eval_floods):
            sl, sc = (fl, fc) if flood else (el, ec)
            s = (torch.zeros(env["features"]["src_index"].numel(), dtype=torch.float64) if flood
                 else score_fn(env).reshape(-1))
            with torch.no_grad():
                m = _metrics(_forward(s, env, sl, sc, eval_mode=True))
            Fs.append(m["F"]); Ds.append(m["D"])
        return float(np.mean(Fs)), float(np.mean(Ds))

    points["no-planning (flood)"] = eval_score(None, flood=True)
    for label in _heuristic_scores(eval_envs[0]).keys():
        points[label] = eval_score(lambda e, _l=label: _heuristic_scores(e)[_l])
    points["informed greedy"] = eval_score(_oracle_score)
    for k, (f, d) in points.items():
        print(f"  [point] {k}: F={f:.4f} D={d:.1f}", flush=True)

    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "cost_controllability.json").write_text(json.dumps(
        {"config": args.config, "weights": weights, "seeds": seeds, "eval_scenes": eval_seeds,
         "learned": learned, "points": {k: {"F": v[0], "D": v[1]} for k, v in points.items()}},
        indent=2, sort_keys=True), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, d in learned.items():
        fs = [p["F"] for p in d["front"]]; ds = [p["D"] for p in d["front"]]
        ax.plot(fs, ds, "-o", color=d["color"], lw=1.6, ms=6, label=f"{label} (sweep $w$)", zorder=4)
        for p in d["front"]:
            ax.annotate(f"w={p['w']:g}", (p["F"], p["D"]), textcoords="offset points", xytext=(4, 4), fontsize=7)
    for k, (f, dd) in points.items():
        col = C_FLOOD if "flood" in k else ("#7f0000" if "greedy" in k else C_HEUR)
        mk = "s" if "flood" in k else ("*" if "greedy" in k else "x")
        ax.scatter([f], [dd], marker=mk, s=90 if mk == "*" else 60, color=col, zorder=5,
                   label=k if ("flood" in k or "greedy" in k) else None)
        ax.annotate(k.replace("-k", ""), (f, dd), textcoords="offset points", xytext=(5, -8), fontsize=6.5, color=col)
    ax.set_xlabel("consensus failure $F$ (lower better)")
    ax.set_ylabel("delay $D$ (effective rounds, lower better)")
    ax.set_title("Cost controllability: learned methods trace a reliability-cost frontier;\n"
                 "heuristics & flood are single points (no cost knob)", fontsize=10)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out_dir / "figures" / "cost_controllability.png", dpi=140); plt.close(fig)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
