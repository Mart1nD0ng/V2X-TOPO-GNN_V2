"""Group C — classic graph-algorithm baselines, evaluated through the SAME constructor + evaluator.

Adds degree-capped greedy and MST-backbone+augmentation (src/v2x_env/baselines.py) to the baseline
ladder. Each algorithm returns an EDGE SET; we reproduce it exactly through the production constructor
by scoring its selected edges high and passing a PER-NODE budget equal to that node's out-degree in the
algorithm's topology (so the row-softmax weighting is consistent with every other method). kNN-by-
distance / kNN-by-channel are listed too but equal nearest-k / best-channel-k of the heuristic group.

Zero training -> distribution is over the held-out scenes (same eval seeds as run_baseline_comparison_v2).
Outputs result/<run-name>/graph_algos.json (merge into the main baseline table).

Usage:
  python -B scripts/analysis/run_baseline_graph_algos.py --config configs/operating_point_v1.yaml \
    --node-count 1500 --train-seed 7 --eval-scenes 6 --run-name baseline_graph_op
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

from scripts.analysis.run_baseline_comparison_v2 import _candidate_cap  # noqa: E402
from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, ci95  # noqa: E402
from src.topology import TopologyConstructionLayer  # noqa: E402
from src.v2x_env.baselines import (  # noqa: E402
    degree_capped_greedy,
    knn_by_channel_score,
    knn_by_distance,
    mst_backbone_plus_augmentation,
)


def _cap_layer(cfg, num_nodes, cap):
    layer = TopologyConstructionLayer(
        max_out_degree=int(cap), support_mode="topk", temperature=1.0,
        topk_backend=str(cfg["topk_backend"]),
        gradient_mode=str(cfg.get("gradient_mode", "straight_through_full_candidate")),
        straight_through_temperature=cfg.get("straight_through_temperature", None))
    return layer


def _score_budget_from_topology(topo, env, cand_cap):
    """Map an algorithm Topology (an UNDIRECTED backbone) to a per-edge score (+1 selected / -inf else)
    and a per-node out-degree budget, reproduced through the constructor. The backbone is used
    BIDIRECTIONALLY (each undirected link becomes both i->j and j->i query edges where the candidate
    graph contains them) so each node gets its full structural degree as out-queries -- a fair query
    budget, not the half-degree a single-direction mapping would give."""
    cand = env["candidate"]
    src = np.asarray(cand.source, dtype=int); dst = np.asarray(cand.target, dtype=int)
    pair2idx = {(int(s), int(t)): i for i, (s, t) in enumerate(zip(src, dst))}
    sel = set()
    for s, t in zip(topo.source, topo.target):
        for a, b in ((int(s), int(t)), (int(t), int(s))):     # both directions
            if (a, b) in pair2idx:
                sel.add(pair2idx[(a, b)])
    sel = sorted(sel)
    n_edges = int(env["features"]["src_index"].numel())
    score = torch.full((n_edges,), -1e9, dtype=torch.float64)
    budget = torch.zeros(cand.num_nodes, dtype=torch.long)
    if sel:
        score[torch.tensor(sel, dtype=torch.long)] = 1.0
        for i in sel:
            budget[src[i]] += 1
    budget = budget.clamp(max=int(cand_cap))
    return score, budget


def main() -> None:
    p = argparse.ArgumentParser(description="Group-C graph-algorithm baselines")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--node-count", type=int, default=1500)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--eval-scenes", type=int, default=6)
    p.add_argument("--run-name", default="baseline_graph")
    args = p.parse_args()

    from src.training.training_smoke import load_training_smoke_config
    base = dict(load_training_smoke_config(str(ROOT / args.config)))
    eval_seeds = [args.train_seed + 1000 + i for i in range(int(args.eval_scenes))]
    eval_envs = [_scene_env(base, args.node_count, s) for s in eval_seeds]
    cap = _candidate_cap(eval_envs[0]["cfg"])
    print(f"Graph-algo baselines [{args.config}] N={args.node_count} cand_cap={cap} eval={eval_seeds}", flush=True)

    # only the genuinely NEW classic graph algorithms (kNN-distance / kNN-channel == nearest-k /
    # best-channel-k already in the heuristic group, so they are omitted here).
    algos = {
        "degree-capped greedy": lambda g: degree_capped_greedy(g, max_degree=cap),
        "MST backbone + augment": lambda g: mst_backbone_plus_augmentation(g, max_degree=cap, augment_k=2),
    }

    rows = []
    for label, fn in algos.items():
        Fs, Ds, Es, degs = [], [], [], []
        for env in eval_envs:
            topo = fn(env["candidate"])
            score, budget = _score_budget_from_topology(topo, env, cap)
            layer = _cap_layer(env["cfg"], env["candidate"].num_nodes, cap)
            with torch.no_grad():
                m = _metrics(_forward(score, env, layer, budget, eval_mode=True))
            Fs.append(m["F"]); Ds.append(m["D"]); Es.append(m["E"])
            degs.append(float(np.mean(budget.numpy())))
        rows.append({"method": label, "group": "graph-algorithm",
                     "F": ci95(Fs), "D": ci95(Ds), "E": ci95(Es), "mean_out_degree": float(np.mean(degs))})
        print(f"  {label:32s}: F={ci95(Fs)['mean']:.4f}±{ci95(Fs)['ci_halfwidth']:.4f}  "
              f"D={ci95(Ds)['mean']:7.1f}  E={ci95(Es)['mean']:.3e}  mean_deg={np.mean(degs):.2f}", flush=True)

    out_dir = ROOT / "result" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "graph_algos.json").write_text(json.dumps(
        {"config": args.config, "node_count": args.node_count, "eval_scenes": eval_seeds,
         "candidate_cap": cap, "rows": rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote {out_dir / 'graph_algos.json'}", flush=True)


if __name__ == "__main__":
    main()
