"""Comprehensive production-training report: train ONE production model and emit the full data +
figure set for the paper.

Trains a model at the operating point with per-step logging, then renders:
  1. training_curves.png   — loss components (L_total, weighted L_R/L_D/L_E) + metrics (F/C/D/E) +
                             gradient norm, all vs step.
  2. physical_topology.png — the deployed topology in the PHYSICAL scene (vehicles at (x,y), coloured
                             by per-node reliability F, selected links weighted by query weight, faint
                             candidate links underneath).
  3. logical_topology.png  — the same topology as an ABSTRACT graph (networkx layout, nodes coloured by
                             reliability and sized by out-degree).
  4. pareto.png / pareto_3d.png — the coupled C/D/E Pareto front (F-D, F-E, and 3D F-D-E) over a cost-
                             weight sweep, with held-out 95% CIs. (D ∝ E at the operating point, so the
                             front is a curve — shown in all three views.)
  5. de_ci.png             — D/E held-out distribution: rel_only (w=0) vs optimized (w>0), mean ±95% CI.
  6. distributions.png     — per-node reliability histogram, out-degree distribution, per-edge
                             link-success distribution.
  7. scalability.png       — F/C/D/E of the ONE trained model evaluated across a node-count ladder.

Saves report.json (all timeseries + final metrics + the deployed topology) and RESULT.md.

Usage:
  python -B scripts/analysis/run_production_report.py --config configs/paper_environment_v1.yaml \
      --train-n 2000 --viz-n 80 --max-steps 200 --out result/production_report_paperenv
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
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import networkx as nx  # noqa: E402

from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer, ci95, pareto_front, train_one  # noqa: E402
from scripts.analysis.validate_operating_point_montecarlo import _evaluate_details, _select_topology  # noqa: E402
from src.losses import compute_coupled_loss  # noqa: E402
from src.training.gradient_governance import GradientGovernanceConfig  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _loss_config, _make_model, load_training_smoke_config,
)

_RDYLGN = "RdYlGn_r"


def train_main(cfg, env, layer, caps, loss_cfg, max_steps, seed):
    """Train the production model, logging per step: loss components + C/D/E/F metrics + grad norm."""
    torch.manual_seed(int(seed))
    model = _make_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    traj = []
    for step in range(int(max_steps)):
        opt.zero_grad(set_to_none=True)
        ev = _forward(model, env, layer, caps)
        lo = compute_coupled_loss(ev, loss_cfg)
        loss = lo.get("effective_backward_loss", lo["total_loss"])
        loss.backward()
        gnorm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model.parameters() if p.grad is not None)))
        m = _metrics(ev)
        traj.append({"step": step, "F": m["F"], "C": m["C"], "D": m["D"], "E": m["E"],
                     "L_total": float(lo["total_loss"]),
                     "wL_R": float(lo["weighted_L_R"]), "wL_D": float(lo["weighted_L_D"]),
                     "wL_E": float(lo["weighted_L_E"]), "grad_norm": gnorm})
        opt.step()
    return model, traj


def deploy_scene(model, base_config, viz_n, seed, eval_q):
    """Select the topology on a small physical scene and pull everything needed to render it."""
    env = _scene_env(base_config, viz_n, seed)
    env["cfg"]["quenched_quadrature"] = eval_q
    layer, caps = _topology_layer(env["cfg"], env["candidate"].num_nodes)
    with torch.no_grad():
        topo = _select_topology(model, env, layer, caps)
        ev = _evaluate_details(topo, env, quenched_quadrature=eval_q)
    nf = env["features"]["node_features"].detach().cpu().numpy()
    node_F = (1.0 - ev["avalanche_details"]["node_p_correct_decision"]).detach().cpu().numpy()
    link_s = ev["channel_diagnostics"]["link_success"].detach().to(torch.float64).cpu().numpy()
    return {
        "px": nf[:, 0] * 600.0, "py": nf[:, 1] * 600.0, "num_nodes": int(env["candidate"].num_nodes),
        "cand_src": env["features"]["src_index"].cpu().numpy(), "cand_dst": env["features"]["dst_index"].cpu().numpy(),
        "sel_src": topo.src_index.cpu().numpy(), "sel_dst": topo.dst_index.cpu().numpy(),
        "sel_w": topo.topology_weight.detach().cpu().numpy(), "node_F": node_F, "link_success": link_s,
        "metrics": _metrics(ev),
    }


# --------------------------------------------------------------------------- figures
def _segments(px, py, src, dst):
    return [[(px[int(s)], py[int(s)]), (px[int(d)], py[int(d)])] for s, d in zip(src, dst)]


def fig_training(traj, out):
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    s = [t["step"] for t in traj]
    ax[0, 0].plot(s, [t["L_total"] for t in traj], label="L_total", color="#111")
    ax[0, 0].plot(s, [t["wL_R"] for t in traj], label="weighted L_R (reliability)", color="#d62728")
    ax[0, 0].plot(s, [t["wL_D"] for t in traj], label="weighted L_D (delay)", color="#1f77b4")
    ax[0, 0].plot(s, [t["wL_E"] for t in traj], label="weighted L_E (energy)", color="#2ca02c")
    ax[0, 0].set_title("coupled loss components"); ax[0, 0].set_xlabel("step"); ax[0, 0].set_ylabel("loss")
    ax[0, 0].legend(fontsize=8); ax[0, 0].grid(alpha=0.3)
    ax[0, 1].plot(s, [t["F"] for t in traj], label="failure F", color="#d62728")
    ax[0, 1].plot(s, [t["C"] for t in traj], label="correct C", color="#2ca02c")
    ax[0, 1].set_title("reliability metrics"); ax[0, 1].set_xlabel("step"); ax[0, 1].legend(fontsize=8)
    ax[0, 1].grid(alpha=0.3)
    ax[1, 0].plot(s, [t["D"] for t in traj], label="delay D (eff-rounds)", color="#1f77b4")
    axb = ax[1, 0].twinx()
    axb.plot(s, [t["E"] for t in traj], label="energy E (J)", color="#2ca02c")
    ax[1, 0].set_title("delay & energy"); ax[1, 0].set_xlabel("step")
    ax[1, 0].set_ylabel("D", color="#1f77b4"); axb.set_ylabel("E (J)", color="#2ca02c"); ax[1, 0].grid(alpha=0.3)
    ax[1, 1].plot(s, [t["grad_norm"] for t in traj], color="#9467bd")
    ax[1, 1].set_title("gradient norm"); ax[1, 1].set_xlabel("step"); ax[1, 1].set_yscale("log"); ax[1, 1].grid(alpha=0.3)
    fig.suptitle("Production training trajectory", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out, dpi=140); plt.close(fig)


def fig_physical(dep, out):
    fig, ax = plt.subplots(figsize=(9, 8))
    norm = Normalize(vmin=0.0, vmax=max(0.02, float(np.quantile(dep["node_F"], 0.95))))
    ax.add_collection(LineCollection(_segments(dep["px"], dep["py"], dep["cand_src"], dep["cand_dst"]),
                                     colors="#cfd8dc", linewidths=0.3, alpha=0.3))
    w = dep["sel_w"]; wn = w / (w.max() + 1e-12)
    ax.add_collection(LineCollection(_segments(dep["px"], dep["py"], dep["sel_src"], dep["sel_dst"]),
                                     colors="#263238", linewidths=(0.4 + 2.6 * wn), alpha=0.8))
    sc = ax.scatter(dep["px"], dep["py"], s=40, c=dep["node_F"], cmap=_RDYLGN, norm=norm,
                    edgecolors="#37474f", linewidths=0.4, zorder=3)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("per-node failure F")
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.2)
    m = dep["metrics"]
    ax.set_title(f"Deployed optimal topology in the physical scene (N={dep['num_nodes']})\n"
                 f"{len(dep['sel_src'])} links / {len(dep['cand_src'])} candidates | "
                 f"F={m['F']:.4f} C={m['C']:.4f} D={m['D']:.2f} E={m['E']:.2e}", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_logical(dep, out):
    G = nx.DiGraph()
    G.add_nodes_from(range(dep["num_nodes"]))
    for s, d, w in zip(dep["sel_src"], dep["sel_dst"], dep["sel_w"]):
        G.add_edge(int(s), int(d), weight=float(w))
    pos = nx.spring_layout(G, seed=7, k=1.6 / np.sqrt(max(dep["num_nodes"], 1)), iterations=80)
    outdeg = np.array([G.out_degree(n) for n in range(dep["num_nodes"])], dtype=float)
    fig, ax = plt.subplots(figsize=(9, 8))
    norm = Normalize(vmin=0.0, vmax=max(0.02, float(np.quantile(dep["node_F"], 0.95))))
    ew = np.array([G[u][v]["weight"] for u, v in G.edges()]) if G.number_of_edges() else np.array([])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#607d8b", alpha=0.5,
                           width=(0.4 + 2.4 * ew / (ew.max() + 1e-12)) if ew.size else 1.0,
                           arrowsize=6, arrowstyle="-|>")
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, node_color=dep["node_F"], cmap=_RDYLGN, vmin=norm.vmin,
                                   vmax=norm.vmax, node_size=(30 + 40 * outdeg), edgecolors="#37474f", linewidths=0.4)
    fig.colorbar(nodes, ax=ax, fraction=0.046, pad=0.04).set_label("per-node failure F")
    ax.set_title(f"Optimal topology as a logical graph (N={dep['num_nodes']}, node size ∝ out-degree)", fontsize=11)
    ax.axis("off"); fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


def fig_distributions(dep, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    ax[0].hist(dep["node_F"], bins=30, color="#d62728", alpha=0.8)
    ax[0].set_title("per-node reliability (failure F)"); ax[0].set_xlabel("F"); ax[0].set_ylabel("nodes"); ax[0].grid(alpha=0.3)
    outdeg = np.bincount(dep["sel_src"], minlength=dep["num_nodes"])
    ax[1].hist(outdeg, bins=range(0, int(outdeg.max()) + 2), color="#1f77b4", alpha=0.8, align="left")
    ax[1].set_title("out-degree distribution (deployed)"); ax[1].set_xlabel("out-degree"); ax[1].set_ylabel("nodes"); ax[1].grid(alpha=0.3)
    ax[2].hist(dep["link_success"], bins=30, color="#2ca02c", alpha=0.8)
    ax[2].set_title("per-link success probability"); ax[2].set_xlabel("link_success"); ax[2].set_ylabel("links"); ax[2].grid(alpha=0.3)
    fig.suptitle("Deployed-topology distributions", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=140); plt.close(fig)


def fig_pareto(rows, out2d, out3d):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    cmap = plt.get_cmap("viridis"); ws = [r["w_cost"] for r in rows]
    for ax, key, lab in ((axes[0], "holdout_D", "delay D"), (axes[1], "holdout_E", "energy E (J)")):
        pts = [(r["holdout_F"]["mean"], r[key]["mean"]) for r in rows]
        fr = pareto_front(pts)
        ax.plot([pts[i][0] for i in fr], [pts[i][1] for i in fr], "--", color="#888", lw=1, zorder=1)
        for r in rows:
            x = r["holdout_F"]["mean"]; y = r[key]["mean"]
            ax.errorbar(x, y, xerr=r["holdout_F"]["ci_halfwidth"], yerr=r[key]["ci_halfwidth"], fmt="o", ms=7,
                        color=cmap((r["w_cost"] - min(ws)) / (max(ws) - min(ws) + 1e-9)), ecolor="gray",
                        elinewidth=1, capsize=3, zorder=5)
            ax.annotate(f"w={r['w_cost']:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xlabel("failure F"); ax.set_ylabel(lab); ax.set_title(f"F vs {lab}"); ax.grid(alpha=0.3)
    fig.suptitle("Coupled C/D/E Pareto front (held-out, ±95% CI)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out2d, dpi=140); plt.close(fig)
    # 3D F-D-E
    fig = plt.figure(figsize=(8, 6.6)); ax = fig.add_subplot(111, projection="3d")
    F = [r["holdout_F"]["mean"] for r in rows]; D = [r["holdout_D"]["mean"] for r in rows]
    E = [r["holdout_E"]["mean"] for r in rows]
    ax.plot(F, D, E, "-", color="#888", lw=1)
    sc = ax.scatter(F, D, E, c=ws, cmap="viridis", s=60, depthshade=False)
    for f, d, e, w in zip(F, D, E, ws):
        ax.text(f, d, e, f" w={w:g}", fontsize=7)
    ax.set_xlabel("F"); ax.set_ylabel("D"); ax.set_zlabel("E (J)")
    fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.1).set_label("cost weight w (=w_D=w_E)")
    ax.set_title("Pareto front in C/D/E space\n(a curve, since D ∝ E at the operating point)", fontsize=11)
    fig.tight_layout(); fig.savefig(out3d, dpi=140); plt.close(fig)


def fig_de_ci(rel, opt, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, key, lab in ((axes[0], "D", "delay D"), (axes[1], "E", "energy E (J)")):
        for xi, (name, r, col) in enumerate([("rel_only", rel, "#d62728"), (f"optimized w={opt['w_cost']:g}", opt, "#1f77b4")]):
            st = r["holdout_" + key]
            ax.bar(xi, st["mean"], yerr=st["ci_halfwidth"], width=0.55, color=col, alpha=0.8, capsize=6, ecolor="black")
            ax.scatter([xi] * len(r["holdout_per_scene"]), [m[key] for m in r["holdout_per_scene"]], color="black", s=14, zorder=5, alpha=0.7)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["rel_only", f"optimized w={opt['w_cost']:g}"]); ax.set_ylabel(lab); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("D/E across held-out scenes: rel_only vs optimized (mean ±95% CI)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(out, dpi=140); plt.close(fig)


def fig_scalability(scal, out):
    ns = [r["n"] for r in scal]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].plot(ns, [r["F"] for r in scal], "-o", color="#d62728", label="F")
    ax[0].plot(ns, [r["C"] for r in scal], "-o", color="#2ca02c", label="C")
    ax[0].set_xscale("log"); ax[0].set_xlabel("node count N"); ax[0].set_title("reliability vs scale"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ns, [r["D"] for r in scal], "-o", color="#1f77b4", label="D")
    axb = ax[1].twinx(); axb.plot(ns, [r["E"] for r in scal], "-s", color="#2ca02c", label="E")
    ax[1].set_xscale("log"); ax[1].set_xlabel("node count N"); ax[1].set_ylabel("D", color="#1f77b4"); axb.set_ylabel("E (J)", color="#2ca02c")
    ax[1].set_title("delay & energy vs scale"); ax[1].grid(alpha=0.3)
    fig.suptitle("Single trained model across the node-count ladder", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(out, dpi=140); plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Comprehensive production-training report")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--train-n", type=int, default=2000)
    p.add_argument("--viz-n", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--weights", default="0,0.5,1,2,5,10")
    p.add_argument("--pareto-n", type=int, default=600, help="N for the pareto sweep (cheaper than train-n)")
    p.add_argument("--pareto-steps", type=int, default=140)
    p.add_argument("--ci-scenes", type=int, default=6)
    p.add_argument("--scalability-ns", default="100,300,1000,3000,10000")
    p.add_argument("--reliability-target", type=float, default=0.02)
    p.add_argument("--out", default="result/production_report_paperenv")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    out_dir = ROOT / args.out
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    eval_q = int(base_config.get("eval_quenched_quadrature", base_config.get("quenched_quadrature", 21)))

    # ---- 1. main production training (full logging) ----
    train_env = _scene_env(base_config, args.train_n, args.seed)
    layer, caps = _topology_layer(train_env["cfg"], train_env["candidate"].num_nodes)
    loss_cfg = dict(_loss_config(train_env["cfg"]))
    loss_cfg["reliability_failure_target"] = float(args.reliability_target)
    loss_cfg["reliability_tail_failure_target"] = float(args.reliability_target)
    print(f"[1/4] training production model N={args.train_n}, {args.max_steps} steps...", flush=True)
    model, traj = train_main(train_env["cfg"], train_env, layer, caps, loss_cfg, args.max_steps, args.seed)
    torch.save(model.state_dict(), out_dir / "model.pt")
    print(f"      final: F={traj[-1]['F']:.4f} C={traj[-1]['C']:.4f} D={traj[-1]['D']:.2f} E={traj[-1]['E']:.2e}", flush=True)

    # ---- 2. deploy on a small physical scene (topology figures) ----
    print(f"[2/4] deploying topology on a physical scene N={args.viz_n}...", flush=True)
    dep = deploy_scene(model, base_config, args.viz_n, args.seed, eval_q)

    # ---- 3. Pareto sweep (cost-weight) ----
    print(f"[3/4] Pareto sweep weights={args.weights} N={args.pareto_n}...", flush=True)
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    governance = GradientGovernanceConfig.from_name("none")
    pareto_env = _scene_env(base_config, args.pareto_n, args.seed)
    pl, pcaps = _topology_layer(pareto_env["cfg"], pareto_env["candidate"].num_nodes)
    base_loss = dict(_loss_config(pareto_env["cfg"]))
    base_loss["reliability_failure_target"] = float(args.reliability_target)
    base_loss["reliability_tail_failure_target"] = float(args.reliability_target)
    scene_seeds = [args.seed + 1000 + i for i in range(int(args.ci_scenes))]
    rows = []
    for w in weights:
        mw, _tr = train_one(pareto_env["cfg"], pareto_env, pl, pcaps, base_loss, w, governance, args.pareto_steps, args.seed)
        hold = []
        for s in scene_seeds:
            e = _scene_env(base_config, args.pareto_n, s)
            l2, c2 = _topology_layer(e["cfg"], e["candidate"].num_nodes)
            with torch.no_grad():
                hold.append(_metrics(_forward(mw, e, l2, c2, eval_mode=True)))
        rows.append({"w_cost": w, "holdout_F": ci95([m["F"] for m in hold]),
                     "holdout_D": ci95([m["D"] for m in hold]), "holdout_E": ci95([m["E"] for m in hold]),
                     "holdout_per_scene": hold})
        print(f"      w={w:>5}: F={rows[-1]['holdout_F']['mean']:.4f} D={rows[-1]['holdout_D']['mean']:.2f} "
              f"E={rows[-1]['holdout_E']['mean']:.2e}", flush=True)
    rel = next((r for r in rows if r["w_cost"] == 0.0), rows[0])
    opt = max(rows, key=lambda r: r["w_cost"])

    # ---- 4. scalability of the main model ----
    print(f"[4/4] scalability across N={args.scalability_ns}...", flush=True)
    scal = []
    for n in [int(x) for x in args.scalability_ns.split(",") if x.strip()]:
        e = _scene_env(base_config, n, args.seed)
        l2, c2 = _topology_layer(e["cfg"], e["candidate"].num_nodes)
        with torch.no_grad():
            m = _metrics(_forward(model, e, l2, c2, eval_mode=True))
        scal.append({"n": n, **m})
        print(f"      N={n:>6}: F={m['F']:.4f} C={m['C']:.4f} D={m['D']:.2f} E={m['E']:.2e}", flush=True)

    # ---- figures ----
    fg = out_dir / "figures"
    fig_training(traj, fg / "training_curves.png")
    fig_physical(dep, fg / "physical_topology.png")
    fig_logical(dep, fg / "logical_topology.png")
    fig_distributions(dep, fg / "distributions.png")
    fig_pareto(rows, fg / "pareto.png", fg / "pareto_3d.png")
    fig_de_ci(rel, opt, fg / "de_ci.png")
    fig_scalability(scal, fg / "scalability.png")

    # ---- data ----
    report = {
        "config": args.config, "currency": f"quenched eval Q={eval_q}", "train_n": args.train_n,
        "max_steps": args.max_steps, "seed": args.seed, "reliability_target": args.reliability_target,
        "final_metrics": {k: traj[-1][k] for k in ("F", "C", "D", "E")},
        "trajectory": traj,
        "deployed_topology": {"num_nodes": dep["num_nodes"], "n_links": int(dep["sel_src"].size),
                              "src": dep["sel_src"].tolist(), "dst": dep["sel_dst"].tolist(),
                              "weight": dep["sel_w"].tolist(), "node_F": dep["node_F"].tolist(),
                              "metrics": dep["metrics"]},
        "pareto": {"weights": weights, "rows": [{k: v for k, v in r.items()} for r in rows],
                   "D_drop_opt_vs_rel": rel["holdout_D"]["mean"] - opt["holdout_D"]["mean"],
                   "E_drop_opt_vs_rel": rel["holdout_E"]["mean"] - opt["holdout_E"]["mean"]},
        "scalability": scal,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    figs = sorted(x.name for x in fg.glob("*.png"))
    print(f"\nwrote {out_dir / 'report.json'} + {len(figs)} figures: {figs}")


if __name__ == "__main__":
    main()
