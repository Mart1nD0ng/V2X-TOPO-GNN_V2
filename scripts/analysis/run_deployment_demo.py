"""End-to-end deployment demonstrator.

Loads a TRAINED planner (run_deployment_demo --model result/planner_paperenv/planner.pt),
replays continuous vehicle motion in a simulated space, and at EVERY frame runs the
frozen model forward (GNN scorer -> hard top-k constructor) to RE-PLAN the comm
topology in real time as the geometry changes — no retraining at deploy. Renders an
animated GIF of the physical space (vehicles coloured by reliability F, deployed
topology edges width proportional to consensus query weight) plus live C/F/D/E.

If --model is omitted it trains a planner inline first (convenience; for a real demo
train once with train_planner.py and load the checkpoint).

Usage:
    python -B scripts/analysis/run_deployment_demo.py --model result/planner_paperenv/planner.pt \
        --frames 30 --dt 1.5 --out result/deployment_demo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

if __name__ == "__main__":  # force the headless backend ONLY when run as the GIF script,
    matplotlib.use("Agg")    # not when imported (e.g. by the interactive GUI, which needs a window)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402

from scripts.analysis.generalization_common import (  # noqa: E402
    build_topology_layer,
    caps_for,
    env_from_snapshot,
    metrics,
    model_score,
    train_model,
)
from scripts.analysis.train_planner import base_vehicle_config  # noqa: E402
from src.evaluation import evaluate_v2x_graph_consensus  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _avalanche_config,
    _evaluator_delay_config,
    _evaluator_energy_config,
    _evaluator_physical_config,
    _make_model,
    _normalized_config,
    load_training_smoke_config,
)
from src.v2x_env.vehicle_snapshot import advance_vehicle_snapshot, generate_vehicle_snapshot  # noqa: E402


def _forward_frame(model, env, topology_layer, cfg) -> dict[str, Any]:
    """One frozen inference: score -> hard top-k topology -> evaluate. Returns the
    geometry + deployed topology + per-node reliability + scalar metrics for rendering."""
    f = env["features"]
    with torch.no_grad():
        score = model_score(model, env)
        topo = topology_layer(
            num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
            edge_score=score, per_node_budget=caps_for(env, cfg),
        )
        sel = topo.selected_candidate_index
        ev = evaluate_v2x_graph_consensus(
            **topo.as_evaluation_kwargs(),
            distance_m=f["distance_m"].index_select(0, sel), los_flag=f["los_flag"].index_select(0, sel),
            node_initial_correct=env["ic"], node_initial_wrong=env["iw"],
            physical_config=_evaluator_physical_config(cfg), avalanche_config=_avalanche_config(cfg),
            energy_config=_evaluator_energy_config(cfg), delay_config=_evaluator_delay_config(cfg),
        )
    nf = f["node_features"].detach().cpu().numpy()
    return {
        "px": (nf[:, 0] * 600.0), "py": (nf[:, 1] * 600.0),
        "cand_src": f["src_index"].cpu().numpy(), "cand_dst": f["dst_index"].cpu().numpy(),
        "sel_src": topo.src_index.cpu().numpy(), "sel_dst": topo.dst_index.cpu().numpy(),
        "sel_w": topo.topology_weight.detach().cpu().numpy(),
        "node_F": ev["node_failure_probability"].detach().cpu().numpy(),
        "metrics": metrics(ev),
    }


def _segments(px, py, src, dst):
    return [[(px[int(s)], py[int(s)]), (px[int(d)], py[int(d)])] for s, d in zip(src, dst)]


def simulate(model, cfg, base, topology_layer, num_frames: int, dt: float) -> list[dict]:
    frames = []
    for i in range(int(num_frames)):
        snap = advance_vehicle_snapshot(base, i * float(dt))
        env = env_from_snapshot(snap, cfg, label=i * float(dt))
        fr = _forward_frame(model, env, topology_layer, cfg)
        fr["t"] = i * float(dt)
        frames.append(fr)
        m = fr["metrics"]
        print(f"  frame {i:>3d}  t={fr['t']:6.1f}s  F={m['F']:.4f} C={m['C']:.4f} "
              f"D={m['D']:.2f} E={m['E']:.3e}  edges={len(fr['sel_src'])}", flush=True)
    return frames


def render_gif(frames: list[dict], cfg: dict, out_dir: Path, fps: int) -> Path:
    allx = np.concatenate([f["px"] for f in frames]); ally = np.concatenate([f["py"] for f in frames])
    mx = 0.05 * (allx.max() - allx.min() + 1.0); my = 0.05 * (ally.max() - ally.min() + 1.0)
    xlim = (allx.min() - mx, allx.max() + mx); ylim = (ally.min() - my, ally.max() + my)
    fmax = max(0.02, float(np.quantile(np.concatenate([f["node_F"] for f in frames]), 0.95)))
    norm = Normalize(vmin=0.0, vmax=fmax)
    Fseries = [f["metrics"]["F"] for f in frames]; Cseries = [f["metrics"]["C"] for f in frames]
    ts = [f["t"] for f in frames]

    fig, (axp, axm) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [2.1, 1]})
    sm = ScalarMappable(norm=norm, cmap="RdYlGn_r"); sm.set_array([])
    cb = fig.colorbar(sm, ax=axp, fraction=0.046, pad=0.04); cb.set_label("node failure probability F")
    profile = cfg.get("vehicle_profile")

    def update(i: int):
        fr = frames[i]; m = fr["metrics"]
        axp.clear()
        cand = _segments(fr["px"], fr["py"], fr["cand_src"], fr["cand_dst"])
        axp.add_collection(LineCollection(cand, colors="#cfd8dc", linewidths=0.3, alpha=0.3))
        seg = _segments(fr["px"], fr["py"], fr["sel_src"], fr["sel_dst"])
        if seg:
            w = fr["sel_w"]; wn = w / (w.max() + 1e-12)
            axp.add_collection(LineCollection(seg, colors="#263238", linewidths=(0.4 + 2.6 * wn), alpha=0.8))
        axp.scatter(fr["px"], fr["py"], s=22, c=fr["node_F"], cmap="RdYlGn_r", norm=norm,
                    zorder=3, edgecolors="#37474f", linewidths=0.3)
        axp.set_xlim(*xlim); axp.set_ylim(*ylim); axp.set_aspect("equal", adjustable="box")
        axp.set_title(f"Deployed topology re-planned in real time  (t = {fr['t']:.1f} s)\n"
                      f"{len(fr['sel_src'])} links over {len(fr['cand_src'])} feasible  |  "
                      f"N={len(fr['px'])}  profile={profile}", fontsize=11)
        axp.set_xlabel("x (m)"); axp.set_ylabel("y (m)"); axp.grid(True, alpha=0.2)

        axm.clear()
        axm.plot(ts[: i + 1], Fseries[: i + 1], "-o", color="#d62728", ms=3, label="failure F")
        axm.plot(ts[: i + 1], Cseries[: i + 1], "-o", color="#2ca02c", ms=3, label="correct C")
        axm.set_xlim(ts[0], ts[-1] + 1e-6); axm.set_ylim(0.0, 1.02)
        axm.axhline(0.0, color="grey", lw=0.5)
        axm.set_title("Live consensus metrics", fontsize=11)
        axm.set_xlabel("simulation time (s)"); axm.grid(True, alpha=0.3); axm.legend(loc="center right", fontsize=9)
        axm.text(0.03, 0.04,
                 f"F = {m['F']:.4f}\nC = {m['C']:.4f}\nD = {m['D']:.2f} eff-rounds\nE = {m['E']:.3e} J",
                 transform=axm.transAxes, va="bottom", ha="left", fontsize=10,
                 bbox=dict(boxstyle="round", fc="white", ec="#90a4ae", alpha=0.9))
        return []

    fig.suptitle("V2X end-to-end deployment — trained GNN planner, frozen inference per frame",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    anim = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(fps, 1)))
    gif = out_dir / "deployment_demo.gif"
    anim.save(gif, writer=PillowWriter(fps=int(fps)))
    plt.close(fig)
    return gif


def main() -> None:
    p = argparse.ArgumentParser(description="End-to-end deployment demonstrator (animated)")
    p.add_argument("--model", default=None, help="path to planner.pt (else trains inline)")
    p.add_argument("--config", default="configs/production_training_v1.yaml")
    p.add_argument("--node-count", type=int, default=120)
    p.add_argument("--scene-seed", type=int, default=42)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--dt", type=float, default=1.5)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=120, help="inline-train steps when --model is omitted")
    p.add_argument("--reliability-target", type=float, default=None)
    p.add_argument("--out", default="result/deployment_demo")
    args = p.parse_args()

    config_path, node_count, scene_seed = args.config, int(args.node_count), int(args.scene_seed)
    # If a checkpoint + meta exist, inherit the trained planner's config/scale for an exact match.
    meta = None
    if args.model:
        meta_path = Path(args.model).resolve().parent / "planner_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            config_path = meta.get("config", config_path); node_count = int(meta.get("node_count", node_count))
            scene_seed = int(meta.get("scene_seed", scene_seed))

    config = dict(load_training_smoke_config(str(ROOT / config_path)))
    config["vehicle_count"] = node_count
    if args.reliability_target is not None:
        config["reliability_failure_target"] = float(args.reliability_target)
        config["reliability_tail_failure_target"] = float(args.reliability_target)
    cfg = _normalized_config(config)
    topology_layer = build_topology_layer(cfg)
    base = generate_vehicle_snapshot(base_vehicle_config(cfg, node_count, scene_seed))

    if args.model:
        model = _make_model(cfg)
        model.load_state_dict(torch.load(Path(args.model).resolve(), map_location="cpu"))
        model.eval()
        print(f"loaded trained planner: {args.model}  (config={config_path}, N={node_count})")
    else:
        print("no --model given; training a planner inline (use train_planner.py for a real demo)...")
        train_times = [0.0, args.dt * args.frames * 0.33, args.dt * args.frames * 0.66]
        train_envs = [env_from_snapshot(advance_vehicle_snapshot(base, t), cfg, label=t) for t in train_times]
        model = train_model(cfg, train_envs, topology_layer, int(args.max_steps))

    print(f"replaying {args.frames} frames at dt={args.dt}s (frozen inference per frame)...")
    frames = simulate(model, cfg, base, topology_layer, int(args.frames), float(args.dt))

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "metrics_timeseries.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh); wr.writerow(["t_s", "F", "C", "D_eff_rounds", "E_joule", "deployed_edges"])
        for fr in frames:
            m = fr["metrics"]; wr.writerow([fr["t"], m["F"], m["C"], m["D"], m["E"], len(fr["sel_src"])])
    gif = render_gif(frames, cfg, out_dir, int(args.fps))
    print(f"\nanimation: {gif}")
    print(f"metrics:   {out_dir / 'metrics_timeseries.csv'}")


if __name__ == "__main__":
    main()
