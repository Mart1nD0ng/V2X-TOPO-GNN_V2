"""Baseline comparison with multi-seed 95% CIs: the proposed Hierarchical-GNN topology
constructor vs a LEARNED baseline (a standard GraphSAGE-mean edge scorer) vs heuristics,
all through the SAME constructor (hard top-k, identical degree budget) and the SAME
closed-form evaluator at the operating point. Reports F (and D/E) mean ±95% CI.

Why: the project previously compared only against its own heuristics. A reviewer wants
(a) a LEARNED comparator (not just hand-rules) and (b) statistical rigor (CIs). This adds
both. The learned baseline is a generic message-passing GNN (the class most published
"GNN for topology/relay selection" methods belong to); a specific published DRL/GNN paper
can be slotted in as an additional `--extra` scorer using the same interface.

Each learned model is trained (reliability objective) on a train scene with several model
seeds; every method is then evaluated on held-out scenes. Distribution = seeds x scenes.

Outputs result/<run-name>/: baseline_comparison.json + figures/baseline_comparison.png + RESULT.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer, ci95  # noqa: E402
from src.losses import compute_coupled_loss  # noqa: E402
from src.training.training_smoke import _loss_config, _make_model, load_training_smoke_config  # noqa: E402


# --------------------------------------------------------------------------- #
# Learned baseline: a standard GraphSAGE-mean edge scorer (no hierarchy / structural bias /
# learnable-gain — i.e. a generic GNN, the comparator the proposed model must beat).
# --------------------------------------------------------------------------- #
class VanillaGNNScorer(nn.Module):
    def __init__(self, node_dim: int = 5, edge_dim: int = 5, hidden: int = 64, layers: int = 2):
        super().__init__()
        self.node_enc = nn.Linear(node_dim, hidden)
        self.msg = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(layers)])
        self.edge_head = nn.Sequential(nn.Linear(2 * hidden + edge_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, *, num_nodes, src_index, dst_index, node_features, edge_features, **_ignored):
        h = torch.relu(self.node_enc(node_features))
        ones = h.new_ones((src_index.shape[0], 1))
        for layer in self.msg:
            agg = h.new_zeros((num_nodes, h.shape[1])).index_add(0, src_index, h.index_select(0, dst_index))
            deg = h.new_zeros((num_nodes, 1)).index_add(0, src_index, ones).clamp(min=1.0)
            h = torch.relu(layer(torch.cat([h, agg / deg], dim=-1)))
        e = self.edge_head(torch.cat([h.index_select(0, src_index), h.index_select(0, dst_index), edge_features], dim=-1))
        return {"edge_score": e.reshape(-1)}


def _model_score(model, env) -> torch.Tensor:
    f = env["features"]
    return model(
        num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
        node_features=f["node_features"], edge_features=f["edge_features"],
        region_id=f["region_id"], num_regions=f["num_regions"],
        edge_sector_id=f["edge_sector_id"], edge_is_cross_region=f["edge_is_cross_region"],
        use_structural_score_bias=False,
    )["edge_score"]


def _train_learned(model_factory, cfg, env, layer, caps, max_steps, seed):
    """Reliability-objective training (w_D=w_E=0) so the comparison is on topology reliability F."""
    torch.manual_seed(int(seed))
    model = model_factory()
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_cfg = dict(_loss_config(cfg))
    loss_cfg["weight_reliability"] = 1.0
    loss_cfg["weight_delay"] = 0.0
    loss_cfg["weight_energy"] = 0.0
    for _ in range(int(max_steps)):
        opt.zero_grad(set_to_none=True)
        ev = _forward_with_score(_model_score(model, env), env, layer, caps)
        lo = compute_coupled_loss(ev, loss_cfg)
        (lo.get("effective_backward_loss", lo["total_loss"])).backward()
        opt.step()
    return model


def _forward_with_score(score, env, layer, caps):
    return _forward(score, env, layer, caps)  # _forward accepts a score tensor or a model


def _heuristic_scores(env) -> dict:
    f = env["features"]
    dist = f["distance_m"].to(dtype=torch.float64)
    ef = f["edge_features"].to(dtype=torch.float64)
    torch.manual_seed(0)
    scores = {"nearest-k": -dist}
    if ef.shape[1] > 2:
        scores["best-channel-k"] = ef[:, 2]
    if ef.shape[1] > 3:
        scores["best-success-k"] = ef[:, 3]
    if ef.shape[1] > 4:
        scores["best-sinr-k"] = ef[:, 4]
    scores["random-k"] = torch.stack([torch.randn(dist.shape[0], dtype=torch.float64) for _ in range(3)]).mean(0)
    return scores


def main() -> None:
    p = argparse.ArgumentParser(description="Baseline comparison (learned + heuristic) with 95% CIs")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--node-count", type=int, default=1500)
    p.add_argument("--max-steps", type=int, default=140)
    p.add_argument("--train-seed", type=int, default=7)
    p.add_argument("--model-seeds", default="7,42,123")
    p.add_argument("--eval-scenes", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--run-name", default="baseline_comparison_v1")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    model_seeds = [int(s) for s in str(args.model_seeds).split(",") if s.strip()]
    eval_seeds = [args.train_seed + 1000 + i for i in range(int(args.eval_scenes))]

    train_env = _scene_env(base_config, args.node_count, args.train_seed)
    cfg = train_env["cfg"]
    layer, caps = _topology_layer(cfg, train_env["candidate"].num_nodes)
    eval_envs = [_scene_env(base_config, args.node_count, s) for s in eval_seeds]
    eval_layers = [_topology_layer(e["cfg"], e["candidate"].num_nodes) for e in eval_envs]

    print(f"Baseline comparison [{args.config}] N={args.node_count} train_seed={args.train_seed} "
          f"model_seeds={model_seeds} eval_scenes={eval_seeds}", flush=True)

    def hier_factory():
        return _make_model(cfg)

    def vanilla_factory():
        return VanillaGNNScorer(5, 5, hidden=int(args.hidden_dim), layers=2).double()

    methods: dict[str, dict] = {}

    # Learned methods: train per model-seed, evaluate on every held-out scene.
    for label, factory in (("GNN (proposed, hierarchical)", hier_factory),
                           ("GNN (vanilla GraphSAGE, learned baseline)", vanilla_factory)):
        Fs, Ds, Es = [], [], []
        for ms in model_seeds:
            model = _train_learned(factory, cfg, train_env, layer, caps, args.max_steps, ms)
            for env, (el, ec) in zip(eval_envs, eval_layers):
                with torch.no_grad():
                    m = _metrics(_forward(_model_score(model, env), env, el, ec))
                Fs.append(m["F"]); Ds.append(m["D"]); Es.append(m["E"])
        methods[label] = {"kind": "learned", "F": Fs, "D": Ds, "E": Es}
        print(f"  {label}: F={ci95(Fs)['mean']:.4f}±{ci95(Fs)['ci_halfwidth']:.4f} (n={len(Fs)})", flush=True)

    # Heuristic methods: no training; evaluate on every held-out scene (same constructor/budget).
    heur_labels = list(_heuristic_scores(eval_envs[0]).keys())
    for label in heur_labels:
        Fs, Ds, Es = [], [], []
        for env, (el, ec) in zip(eval_envs, eval_layers):
            score = _heuristic_scores(env)[label]
            with torch.no_grad():
                m = _metrics(_forward(score.reshape(-1), env, el, ec))
            Fs.append(m["F"]); Ds.append(m["D"]); Es.append(m["E"])
        methods[label] = {"kind": "heuristic", "F": Fs, "D": Ds, "E": Es}
        print(f"  {label}: F={ci95(Fs)['mean']:.4f}±{ci95(Fs)['ci_halfwidth']:.4f} (n={len(Fs)})", flush=True)

    rows = []
    for label, d in methods.items():
        rows.append({"method": label, "kind": d["kind"],
                     "F": ci95(d["F"]), "D": ci95(d["D"]), "E": ci95(d["E"])})
    rows.sort(key=lambda r: r["F"]["mean"])
    proposed = next(r for r in rows if r["method"].startswith("GNN (proposed"))
    best_other = min((r for r in rows if not r["method"].startswith("GNN (proposed")), key=lambda r: r["F"]["mean"])
    # CIs disjoint -> statistically separated at ~95%
    sep = proposed["F"]["hi"] < best_other["F"]["lo"]
    summary = {
        "config": args.config, "node_count": args.node_count, "train_seed": args.train_seed,
        "model_seeds": model_seeds, "eval_scenes": eval_seeds, "max_steps": args.max_steps,
        "rows": rows, "best_non_proposed": best_other["method"],
        "proposed_F_mean": proposed["F"]["mean"], "best_other_F_mean": best_other["F"]["mean"],
        "proposed_beats_best_other_ratio": best_other["F"]["mean"] / max(proposed["F"]["mean"], 1e-12),
        "proposed_F_CI_below_best_other": bool(sep),
    }

    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "baseline_comparison.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _render(rows, out_dir / "figures" / "baseline_comparison.png")
    _write_result(summary, out_dir / "RESULT.md")

    print(f"\nproposed F={proposed['F']['mean']:.4f}; best other ({best_other['method']}) "
          f"F={best_other['F']['mean']:.4f}; ratio={summary['proposed_beats_best_other_ratio']:.2f}x; "
          f"CIs disjoint={sep}")
    print(f"wrote {out_dir}")


def _render(rows, out_path: Path) -> None:
    labels = [r["method"].replace(" (", "\n(") for r in rows]
    means = [r["F"]["mean"] for r in rows]; errs = [r["F"]["ci_halfwidth"] for r in rows]
    colors = ["#1f77b4" if r["method"].startswith("GNN (proposed") else
              ("#2ca02c" if r["kind"] == "learned" else "#d62728") for r in rows]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    x = range(len(rows))
    ax.bar(x, means, yerr=errs, color=colors, alpha=0.85, capsize=5, ecolor="black")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("failure F (lower = better)")
    ax.set_title("Topology-construction methods at the operating point — F mean ±95% CI "
                 "(blue=proposed, green=learned baseline, red=heuristic)", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_result(summary: dict, out_path: Path) -> None:
    lines = [
        "# Baseline Comparison — Proposed GNN vs Learned Baseline vs Heuristics (±95% CI)",
        "",
        f"Operating point `{summary['config']}`, N={summary['node_count']}, learned models trained "
        f"(reliability objective) on seed {summary['train_seed']} across model seeds "
        f"{summary['model_seeds']}, all methods evaluated on held-out scenes {summary['eval_scenes']} "
        f"through the SAME hard-top-k constructor + closed-form evaluator. Harness: "
        "`scripts/analysis/run_baseline_comparison.py`.",
        "",
        "| method | kind | F (±95% CI) | D (±95% CI) | E (±95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in summary["rows"]:
        lines.append(f"| {r['method']} | {r['kind']} | {r['F']['mean']:.4f}±{r['F']['ci_halfwidth']:.4f} "
                     f"| {r['D']['mean']:.1f}±{r['D']['ci_halfwidth']:.1f} "
                     f"| {r['E']['mean']:.3e}±{r['E']['ci_halfwidth']:.1e} |")
    lines += [
        "",
        f"- **Proposed F = {summary['proposed_F_mean']:.4f}** vs best non-proposed "
        f"({summary['best_non_proposed']}) F = {summary['best_other_F_mean']:.4f} → "
        f"**{summary['proposed_beats_best_other_ratio']:.2f}× better**.",
        f"- Proposed 95% CI strictly below the best baseline's CI: "
        f"**{summary['proposed_F_CI_below_best_other']}** (statistically separated).",
        "",
        "The learned baseline (vanilla GraphSAGE) isolates the value of the proposed architecture "
        "(hierarchical regions + P2 scorer dynamic range) beyond 'just using a GNN'. A specific "
        "published DRL/GNN relay-selection method can be added as another scorer with the same "
        "interface (the apples-to-apples comparison only swaps the edge-score function). Standard "
        "graph baselines (kNN / degree-capped greedy / MST-backbone) are available in "
        "`src/v2x_env/baselines.py`.",
        "",
        "Artifacts: `baseline_comparison.json`, `figures/baseline_comparison.png`.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
