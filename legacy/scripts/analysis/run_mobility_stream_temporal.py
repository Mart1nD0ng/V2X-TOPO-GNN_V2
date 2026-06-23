"""Phase 2: temporal (recurrent) vs memoryless under a HORIZON objective, with
load-aware D/E. The bundled landing of temporal objective + recurrent model +
evaluator-side load-aware interference.

Both models optimise the same horizon objective over a mobility stream:
    L = w_rel * F_t  +  w_ant * F_transition(t -> t+δ)  +  w_churn * churn_proxy(t-1, t)
where F is consensus failure under LOAD-AWARE interference (D/E enabler), F_transition
re-evaluates the chosen topology on an advanced sub-frame (anticipation), and
churn_proxy penalises per-node load change (re-planning cost). The memoryless model
is trained per frame; the temporal model with truncated BPTT (it has per-node memory).

G2 (does memory help under the temporal objective?):
  G2a anticipation : temporal F_transition < memoryless on held-out future frames
  G2b churn        : temporal churn < memoryless at comparable reliability
  G2c sustained    : temporal worst-frame F <= memoryless worst-frame F
Per-frame F is expected ~equal (memoryless is already at the per-frame floor).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.generalization_common import build_topology_layer, caps_for, env_from_snapshot  # noqa: E402
from src.evaluation import evaluate_v2x_graph_consensus  # noqa: E402
from src.losses.horizon_objective import churn_proxy, edge_distances, node_in_load  # noqa: E402
from src.models import HierarchicalGNNScorer, TemporalGNNScorer  # noqa: E402
from src.topology.temporal_metrics import mean_sequence_churn  # noqa: E402
from src.training.training_smoke import _avalanche_config, _normalized_config, load_training_smoke_config  # noqa: E402
from src.v2x_env.mobility import MobilityConfig, MobilityStream  # noqa: E402
from src.v2x_env.profiles import production_like_density_v0_vehicle_config  # noqa: E402
from src.v2x_env.vehicle_snapshot import advance_vehicle_snapshot, generate_vehicle_snapshot  # noqa: E402

_PHYS = {"tx_power_dbm": 23.0, "mcs_threshold_db": 8.0, "transition_width_db": 3.0, "interference_proxy_dbm": -82.0}
_REM, _BASE = "#1f77b4", "#d62728"


def _phys(coupling):
    return {**_PHYS, "interference_density_coupling_db": float(coupling), "interference_reference_load": 1.0}


def _score(model, env, state):
    """Return (edge_score, new_state). state is None for the memoryless model."""
    f = env["features"]
    if isinstance(model, TemporalGNNScorer):
        return model(num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
                     node_features=f["node_features"], edge_features=f["edge_features"],
                     region_id=f["region_id"], num_regions=f["num_regions"], state=state)
    out = model(num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
                node_features=f["node_features"], edge_features=f["edge_features"],
                region_id=f["region_id"], num_regions=f["num_regions"],
                edge_sector_id=f["edge_sector_id"], edge_is_cross_region=f["edge_is_cross_region"],
                use_structural_score_bias=False)
    return out["edge_score"], None


def _topology(score, env, topology_layer, cfg):
    f = env["features"]
    return topology_layer(num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
                          edge_score=score, per_node_budget=caps_for(env, cfg))


def _eval_F(topo, env, cfg, coupling, *, distance_override=None):
    f = env["features"]
    sel = topo.selected_candidate_index
    distance = f["distance_m"].index_select(0, sel) if distance_override is None else distance_override
    # P1-1.2: optional per-candidate-edge hidden AR(1) shadow offset, index-selected to the chosen
    # topology so it follows the same edges as distance/los. None -> byte-identical.
    shadow = env.get("shadow_offset_db")
    shadow_sel = shadow.index_select(0, sel) if shadow is not None else None
    ev = evaluate_v2x_graph_consensus(
        num_nodes=env["candidate"].num_nodes, src_index=topo.src_index, dst_index=topo.dst_index,
        topology_weight=topo.topology_weight, distance_m=distance,
        los_flag=f["los_flag"].index_select(0, sel),
        node_initial_correct=env["ic"], node_initial_wrong=env["iw"],
        physical_config=_phys(coupling), avalanche_config=_avalanche_config(cfg),
        energy_config={"packet_duration_s": 0.001}, shadow_offset_db=shadow_sel)
    return ev


def _anticipation_F(topo, snapshot, dt_frac, env, cfg, coupling):
    """Re-evaluate the chosen topology on a sub-frame advanced by dt_frac (anticipation)."""
    sub = advance_vehicle_snapshot(snapshot, dt_frac)
    px = torch.as_tensor(sub["x"], dtype=torch.float64); py = torch.as_tensor(sub["y"], dtype=torch.float64)
    sub_dist = edge_distances(px, py, topo.src_index, topo.dst_index)
    ev = _eval_F(topo, env, cfg, coupling, distance_override=sub_dist)
    return ev["F_avalanche_node_mean"].mean()


def _selected_pairs(topo):
    return topo.src_index.detach().cpu().numpy(), topo.dst_index.detach().cpu().numpy()


def _train(model, frames, train_idx, topology_layer, cfg, *, coupling, epochs, window, dt_frac,
           w_rel, w_ant, w_churn, temporal):
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    n = frames[train_idx[0]][1]["candidate"].num_nodes
    for _epoch in range(epochs):
        state = model.init_state(n) if temporal else None
        prev_in_load = None
        opt.zero_grad(set_to_none=True)
        window_loss = frames[train_idx[0]][1]["features"]["distance_m"].new_zeros(())
        steps_in_window = 0
        for j, t in enumerate(train_idx):
            snapshot, env = frames[t]
            score, state = _score(model, env, state)
            topo = _topology(score, env, topology_layer, cfg)
            ev = _eval_F(topo, env, cfg, coupling)
            f_t = ev["F_avalanche_node_mean"].mean()
            ant = _anticipation_F(topo, snapshot, dt_frac, env, cfg, coupling)
            in_load = node_in_load(n, topo.dst_index, topo.topology_weight)
            churn = churn_proxy(prev_in_load, in_load) if prev_in_load is not None else f_t.new_zeros(())
            window_loss = window_loss + (w_rel * f_t + w_ant * ant + w_churn * churn)
            steps_in_window += 1
            prev_in_load = in_load.detach()
            flush = (not temporal) or steps_in_window >= window or j == len(train_idx) - 1
            if flush:
                (window_loss / steps_in_window).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                window_loss = window_loss.new_zeros(()); steps_in_window = 0
                if temporal:
                    state = state.detach()
    return model


def _evaluate_horizon(model, frames, idx, topology_layer, cfg, coupling, dt_frac, *, temporal):
    n = frames[idx[0]][1]["candidate"].num_nodes
    state = model.init_state(n) if temporal else None
    rows, pairs = [], []
    with torch.no_grad():
        # warm the recurrent state over preceding frames so eval reflects online use
        for t in range(idx[0]):
            _, state = _score(model, frames[t][1], state) if temporal else (None, None)
        for t in idx:
            snapshot, env = frames[t]
            score, state = _score(model, env, state)
            topo = _topology(score, env, topology_layer, cfg)
            f_t = float(_eval_F(topo, env, cfg, coupling)["F_avalanche_node_mean"].mean())
            ant = float(_anticipation_F(topo, snapshot, dt_frac, env, cfg, coupling))
            rows.append({"frame": t, "F": f_t, "F_transition": ant})
            pairs.append(_selected_pairs(topo))
    churn = mean_sequence_churn(pairs, n)
    return rows, churn


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 2 temporal vs memoryless horizon training")
    p.add_argument("--config", default="configs/production_training_v1.yaml")
    p.add_argument("--node-count", type=int, default=800)
    p.add_argument("--scene-seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--num-frames", type=int, default=14)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--coupling-db", type=float, default=10.0)
    p.add_argument("--w-rel", type=float, default=1.0)
    p.add_argument("--w-ant", type=float, default=1.0)
    p.add_argument("--w-churn", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--run-name", default="mobility_temporal_v1")
    args = p.parse_args()

    config = dict(load_training_smoke_config(str(ROOT / args.config)))
    config["vehicle_count"] = int(args.node_count)
    cfg = _normalized_config(config)
    dt_frac = float(args.dt) * 0.5

    base = generate_vehicle_snapshot(production_like_density_v0_vehicle_config(int(args.node_count), seed=int(args.scene_seed)))
    stream = MobilityStream(base, MobilityConfig(dt_s=args.dt, num_frames=args.num_frames))
    n_train = max(1, min(len(stream) - 1, round(args.train_fraction * len(stream))))
    train_idx = list(range(n_train)); test_idx = list(range(n_train, len(stream)))
    print(f"Scene {args.scene_seed}, N={args.node_count}, {len(stream)} frames; train {train_idx}, held-out {test_idx}; "
          f"coupling={args.coupling_db}dB, horizon w=(rel {args.w_rel}, ant {args.w_ant}, churn {args.w_churn})", flush=True)

    topology_layer = build_topology_layer(cfg)
    print("Building frame environments (load-aware interference)...", flush=True)
    frames = {}
    for t in range(len(stream)):
        snap = stream.frame_at(t)
        env = env_from_snapshot(snap, cfg, label=stream.time_at(t), interference_coupling_db=args.coupling_db)
        frames[t] = (snap, env)

    common = dict(coupling=args.coupling_db, epochs=args.epochs, window=args.window, dt_frac=dt_frac,
                  w_rel=args.w_rel, w_ant=args.w_ant, w_churn=args.w_churn)
    torch.manual_seed(int(args.model_seed))
    memoryless = HierarchicalGNNScorer(5, 5, hidden_dim=args.hidden_dim, message_layers=2, init_mode="xavier",
                                       use_structural_score_bias=False, learnable_score_gain=True,
                                       score_output_gain=10.0, score_standardization=True).double()
    print("Training memoryless (horizon objective, per-frame)...", flush=True)
    _train(memoryless, frames, train_idx, topology_layer, cfg, temporal=False, **common)

    torch.manual_seed(int(args.model_seed))
    temporal = TemporalGNNScorer(5, 5, hidden_dim=args.hidden_dim, message_layers=2,
                                 learnable_score_gain=True, score_output_gain=10.0, score_standardization=True).double()
    print("Training temporal (horizon objective, truncated BPTT)...", flush=True)
    _train(temporal, frames, train_idx, topology_layer, cfg, temporal=True, **common)

    mem_rows, mem_churn = _evaluate_horizon(memoryless, frames, test_idx, topology_layer, cfg, args.coupling_db, dt_frac, temporal=False)
    tmp_rows, tmp_churn = _evaluate_horizon(temporal, frames, test_idx, topology_layer, cfg, args.coupling_db, dt_frac, temporal=True)
    mean = statistics.fmean
    mem = {"F": mean([r["F"] for r in mem_rows]), "Ft": mean([r["F_transition"] for r in mem_rows]),
           "worstF": max(r["F"] for r in mem_rows), "churn": mem_churn}
    tmp = {"F": mean([r["F"] for r in tmp_rows]), "Ft": mean([r["F_transition"] for r in tmp_rows]),
           "worstF": max(r["F"] for r in tmp_rows), "churn": tmp_churn}
    print(f"  memoryless: F={mem['F']:.4f}  F_transition={mem['Ft']:.4f}  worstF={mem['worstF']:.4f}  churn={mem['churn']:.4f}")
    print(f"  temporal  : F={tmp['F']:.4f}  F_transition={tmp['Ft']:.4f}  worstF={tmp['worstF']:.4f}  churn={tmp['churn']:.4f}")

    g2a = tmp["Ft"] < mem["Ft"] - 1e-4
    g2b = tmp["churn"] < mem["churn"] - 1e-3 and tmp["F"] <= mem["F"] * 1.10 + 1e-9
    g2c = tmp["worstF"] <= mem["worstF"] - 1e-4
    no_perframe_regression = tmp["F"] <= mem["F"] * 1.10 + 1e-9
    passed = (g2a or g2b or g2c) and no_perframe_regression
    verdict = ("TEMPORAL HELPS under the horizon objective (G2 pass)" if passed
               else "TEMPORAL DOES NOT HELP (G2 fail) -> memory adds nothing even temporally; keep memoryless")
    summary = {
        "config": args.config, "node_count": args.node_count, "scene_seed": args.scene_seed,
        "dt_s": args.dt, "num_frames": args.num_frames, "n_train": n_train, "epochs": args.epochs,
        "window": args.window, "coupling_db": args.coupling_db,
        "weights": {"rel": args.w_rel, "ant": args.w_ant, "churn": args.w_churn},
        "memoryless": mem, "temporal": tmp, "memoryless_rows": mem_rows, "temporal_rows": tmp_rows,
        "G2a_anticipation": bool(g2a), "G2b_churn_at_matched_reliability": bool(g2b),
        "G2c_sustained_worst_frame": bool(g2c), "no_per_frame_regression": bool(no_perframe_regression),
        "verdict": verdict,
    }
    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "mobility_temporal.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _render(summary, out_dir / "figures" / "mobility_temporal.png")
    print(f"\nG2a(anticipation)={g2a}  G2b(churn)={g2b}  G2c(sustained)={g2c}  no-perframe-regression={no_perframe_regression}")
    print(f"VERDICT: {verdict}")


def _render(summary, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.suptitle(f"Phase 2: temporal vs memoryless (horizon objective, load-aware D/E)\n{summary['verdict']}",
                 fontsize=11, fontweight="bold")
    metrics = ["F", "Ft", "worstF", "churn"]
    labels = ["per-frame F", "F_transition\n(anticipation)", "worst-frame F", "churn"]
    x = range(len(metrics)); w = 0.38
    ax.bar([i - w/2 for i in x], [summary["memoryless"][k] for k in metrics], width=w, color=_BASE, label="memoryless", alpha=0.85)
    ax.bar([i + w/2 for i in x], [summary["temporal"][k] for k in metrics], width=w, color=_REM, label="temporal (GRU)", alpha=0.85)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("value (lower=better)"); ax.set_title("Held-out future frames")
    ax.legend(fontsize=9); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
