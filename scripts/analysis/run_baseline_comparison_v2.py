"""Comprehensive baseline comparison (v2) — paper-grade table with multi-seed 95% CIs.

Extends run_baseline_comparison.py into the full ladder of competitors, every one routed through
the SAME hard-top-k constructor + SAME analytic Avalanche evaluator on the SAME held-out scenes
(only the edge-scoring/selection differs), so the gap is attributable to the topology decision alone.

Method ladder (group : method):
  A no-planning : flood            (uniform weights over ALL candidate peers, budget = candidate cap)
  B heuristics  : nearest-k, best-channel-k, best-success-k, best-sinr-k, random-k
  D learned     : D1 vanilla GraphSAGE (architecture ablation, end-to-end)
                  D3 mean-field-trained proposed GNN (currency ablation: train Q=1, eval Q=21)
  *  proposed   : hierarchical GNN + end-to-end coupled loss (train Q=11, eval Q=21)
  E reference   : informed greedy (link-success x peer-confidence) — a strong evaluator-aligned
                  reference ("how good can edge choice get"); NOT exhaustive, labelled honestly.

Reliability-objective training (w_D=w_E=0) so the headline table is a clean F comparison; the
D/E cost-controllability story is a separate sub-experiment (run_baseline_cost_controllability.py).
F is the consensus FAILURE rate (lower = better), reported at the quenched eval currency.

Venues: paper_environment_v1 (cap 4 < candidate cap -> true edge SELECTION) and operating_point_v1
(cap 8 = candidate cap -> weight ALLOCATION over the same peers). Run both.

Usage:
  python -B scripts/analysis/run_baseline_comparison_v2.py --config configs/paper_environment_v1.yaml \
    --node-count 1500 --model-seeds 7,42,123,2024,99 --eval-scenes 6 --run-name baseline_v2_env
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
from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer, ci95  # noqa: E402
from src.losses import compute_coupled_loss  # noqa: E402
from src.topology import TopologyConstructionLayer  # noqa: E402
from src.training.training_smoke import _loss_config, _make_model, load_training_smoke_config  # noqa: E402

# group colours for the figure
GROUP_COL = {"no-planning": "#999999", "heuristic": "#D55E00", "learned-baseline": "#009E73",
             "proposed": "#0072B2", "reference": "#7f0000"}


def _candidate_cap(cfg) -> int:
    cand = cfg.get("candidate_config", {}) or {}
    return int(cand.get("max_candidates_per_node", 8) or 8)


def _flood_layer(cfg, num_nodes):
    cap = _candidate_cap(cfg)
    layer = TopologyConstructionLayer(
        max_out_degree=cap, support_mode="topk", temperature=1.0,
        topk_backend=str(cfg["topk_backend"]),
        gradient_mode=str(cfg.get("gradient_mode", "straight_through_full_candidate")),
        straight_through_temperature=cfg.get("straight_through_temperature", None))
    return layer, torch.full((num_nodes,), cap, dtype=torch.long)


def _oracle_score(env) -> torch.Tensor:
    """Informed greedy reference: prefer high-link-success edges toward high-initial-confidence peers
    (link_success_ij * ic_j) — an evaluator-aligned strong heuristic (not exhaustive)."""
    f = env["features"]
    ef = f["edge_features"].to(dtype=torch.float64)
    succ = ef[:, 3] if ef.shape[1] > 3 else torch.ones(ef.shape[0], dtype=torch.float64)
    ic = env["ic"].to(dtype=torch.float64).reshape(-1)
    peer = ic.index_select(0, f["dst_index"])
    return (succ * peer).reshape(-1)


def _train(kind: str, cfg, env, layer, caps, max_steps: int, seed: int, hidden: int):
    """Reliability-objective training (w_D=w_E=0). kind in {proposed, vanilla} (mean-field is
    'proposed' trained on a Q=1 env)."""
    torch.manual_seed(int(seed))
    model = _make_model(cfg) if kind == "proposed" else VanillaGNNScorer(5, 5, hidden=hidden, layers=2).double()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_cfg = dict(_loss_config(cfg))
    loss_cfg["weight_reliability"] = 1.0
    loss_cfg["weight_delay"] = 0.0
    loss_cfg["weight_energy"] = 0.0
    for _ in range(int(max_steps)):
        opt.zero_grad(set_to_none=True)
        ev = _forward(_model_score(model, env), env, layer, caps)
        lo = compute_coupled_loss(ev, loss_cfg)
        (lo.get("effective_backward_loss", lo["total_loss"])).backward()
        opt.step()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Comprehensive baseline comparison v2 (±95% CI)")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--node-count", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=140)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--model-seeds", default="7,42,123,2024,99")
    p.add_argument("--eval-scenes", type=int, default=6)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--run-name", default="baseline_v2")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    base_mf = dict(base_config); base_mf["quenched_quadrature"] = 1   # mean-field training currency
    model_seeds = [int(s) for s in str(args.model_seeds).split(",") if s.strip()]
    eval_seeds = [args.train_seed + 1000 + i for i in range(int(args.eval_scenes))]

    train_env = _scene_env(base_config, args.node_count, args.train_seed)
    train_env_mf = _scene_env(base_mf, args.node_count, args.train_seed)
    cfg = train_env["cfg"]
    layer, caps = _topology_layer(cfg, train_env["candidate"].num_nodes)
    layer_mf, caps_mf = _topology_layer(train_env_mf["cfg"], train_env_mf["candidate"].num_nodes)
    eval_envs = [_scene_env(base_config, args.node_count, s) for s in eval_seeds]
    eval_layers = [_topology_layer(e["cfg"], e["candidate"].num_nodes) for e in eval_envs]
    eval_floods = [_flood_layer(e["cfg"], e["candidate"].num_nodes) for e in eval_envs]
    cand_cap = _candidate_cap(cfg)
    venue = "selection (cap<candidate)" if int(cfg["max_out_degree"]) < cand_cap else "weighting (cap=candidate)"
    print(f"Baseline v2 [{args.config}] N={args.node_count} cap={cfg['max_out_degree']} cand_cap={cand_cap} "
          f"-> {venue}; seeds={model_seeds} eval={eval_seeds}", flush=True)

    methods: dict[str, dict] = {}

    def eval_score_over_scenes(score_fn):
        Fs, Ds, Es = [], [], []
        for env, (el, ec) in zip(eval_envs, eval_layers):
            with torch.no_grad():
                m = _metrics(_forward(score_fn(env).reshape(-1), env, el, ec, eval_mode=True))
            Fs.append(m["F"]); Ds.append(m["D"]); Es.append(m["E"])
        return Fs, Ds, Es

    # A no-planning: flood (uniform weights over all candidate peers)
    Fs, Ds, Es = [], [], []
    for env, (fl, fc) in zip(eval_envs, eval_floods):
        with torch.no_grad():
            m = _metrics(_forward(torch.zeros(env["features"]["src_index"].numel(), dtype=torch.float64),
                                  env, fl, fc, eval_mode=True))
        Fs.append(m["F"]); Ds.append(m["D"]); Es.append(m["E"])
    methods["no-planning (flood)"] = {"group": "no-planning", "F": Fs, "D": Ds, "E": Es}
    print(f"  no-planning (flood): F={ci95(Fs)['mean']:.4f}±{ci95(Fs)['ci_halfwidth']:.4f}", flush=True)

    # B heuristics
    for label in _heuristic_scores(eval_envs[0]).keys():
        Fs, Ds, Es = eval_score_over_scenes(lambda e, _l=label: _heuristic_scores(e)[_l])
        methods[label] = {"group": "heuristic", "F": Fs, "D": Ds, "E": Es}
        print(f"  {label}: F={ci95(Fs)['mean']:.4f}±{ci95(Fs)['ci_halfwidth']:.4f}", flush=True)

    # E reference: informed greedy
    Fs, Ds, Es = eval_score_over_scenes(_oracle_score)
    methods["informed greedy (reference)"] = {"group": "reference", "F": Fs, "D": Ds, "E": Es}
    print(f"  informed greedy (reference): F={ci95(Fs)['mean']:.4f}±{ci95(Fs)['ci_halfwidth']:.4f}", flush=True)

    # learned: proposed (Q=11), vanilla GraphSAGE, mean-field proposed (Q=1 train) — per model seed
    learned = [
        ("GNN (proposed, hierarchical)", "proposed", train_env, layer, caps, "proposed"),
        ("GNN (vanilla GraphSAGE)", "vanilla", train_env, layer, caps, "learned-baseline"),
        ("GNN (proposed, mean-field-trained)", "proposed", train_env_mf, layer_mf, caps_mf, "learned-baseline"),
    ]
    for label, kind, tenv, tlayer, tcaps, group in learned:
        Fs, Ds, Es = [], [], []
        for ms in model_seeds:
            model = _train(kind, tenv["cfg"], tenv, tlayer, tcaps, args.max_steps, ms, args.hidden_dim)
            for env, (el, ec) in zip(eval_envs, eval_layers):
                with torch.no_grad():
                    m = _metrics(_forward(_model_score(model, env), env, el, ec, eval_mode=True))
                Fs.append(m["F"]); Ds.append(m["D"]); Es.append(m["E"])
        methods[label] = {"group": group, "F": Fs, "D": Ds, "E": Es}
        print(f"  {label}: F={ci95(Fs)['mean']:.4f}±{ci95(Fs)['ci_halfwidth']:.4f} (n={len(Fs)})", flush=True)

    rows = [{"method": k, "group": v["group"], "F": ci95(v["F"]), "D": ci95(v["D"]), "E": ci95(v["E"]),
             "F_samples": v["F"]} for k, v in methods.items()]
    rows.sort(key=lambda r: r["F"]["mean"])
    proposed = next(r for r in rows if r["method"].startswith("GNN (proposed, hierarchical"))
    others = [r for r in rows if r is not proposed and r["group"] != "reference"]
    best_other = min(others, key=lambda r: r["F"]["mean"])
    sep = proposed["F"]["hi"] < best_other["F"]["lo"]
    ref = next((r for r in rows if r["group"] == "reference"), None)
    summary = {
        "config": args.config, "venue": venue, "node_count": args.node_count, "cap": cfg["max_out_degree"],
        "candidate_cap": cand_cap, "model_seeds": model_seeds, "eval_scenes": eval_seeds,
        "rows": [{k: v for k, v in r.items() if k != "F_samples"} for r in rows],
        "proposed_F": proposed["F"]["mean"], "best_other": best_other["method"],
        "best_other_F": best_other["F"]["mean"],
        "proposed_vs_best_other_ratio": best_other["F"]["mean"] / max(proposed["F"]["mean"], 1e-12),
        "proposed_CI_below_best_other": bool(sep),
        "regret_vs_reference": (proposed["F"]["mean"] - ref["F"]["mean"]) if ref else None,
    }
    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _render(rows, summary, out_dir / "figures" / "baseline_F.png")
    print(f"\nproposed F={summary['proposed_F']:.4f}; best non-proposed ({best_other['method']}) "
          f"F={summary['best_other_F']:.4f}; ratio={summary['proposed_vs_best_other_ratio']:.2f}x; "
          f"CIs disjoint={sep}; regret vs reference={summary['regret_vs_reference']}", flush=True)
    print(f"wrote {out_dir}", flush=True)


def _render(rows, summary, out_path: Path) -> None:
    labels = [r["method"].replace("GNN ", "").replace(" (", "\n(") for r in rows]
    means = [r["F"]["mean"] for r in rows]; errs = [r["F"]["ci_halfwidth"] for r in rows]
    cols = [GROUP_COL.get(r["group"], "#666") for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(rows)), 5.0))
    x = range(len(rows))
    ax.bar(x, means, yerr=errs, color=cols, alpha=0.88, capsize=4, ecolor="black")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=7, rotation=18, ha="right")
    ax.set_ylabel("consensus failure $F$ (lower = better)")
    ax.set_title(f"Topology methods at the operating point — F mean ±95% CI [{summary['venue']}]\n"
                 f"(grey=no-plan, orange=heuristic, green=learned baseline, blue=proposed, dark-red=reference)",
                 fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
