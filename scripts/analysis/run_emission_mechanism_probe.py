"""T3c — WHY does the carried-reliability emission stabilize the recurrence? (C3 mechanism probe)

The filter arm (graph_gru + emission) eliminates the pure-memory collapse (39x sigma_init reduction).
This probe instruments the recurrent path during training to expose the mechanism, comparing the
`full` arm (graph_gru, NO emission -> collapses on init 7) vs `filter` (graph_gru + emission -> stable)
on the same cell/seeds. Per epoch it records:

  * gate-grad-norm : L2 norm of the GraphCoupledGRUCell gate weight gradients (update+reset+candidate),
                     pre-clip -> the gradient-scale signature (vanish/explode of the recurrent path).
  * state_norm     : mean carried hidden-state norm  -> does the recurrent state drift/blow up?
  * state_graph_norm, x_norm, joined_std : the gate INPUT distribution (the `joined` cat at
                     temporal_scorer.py:73) -> does emission keep the recurrent input bounded?

Hypothesis: WITHOUT emission the carried state drifts and the recurrent gradients diverge on bad inits
(driving the collapse); the emission (a calibrated [0,1] observation re-grounding the state each frame)
keeps both bounded, so the gates learn a robust update that generalizes across inits.

Usage: python -B scripts/analysis/run_emission_mechanism_probe.py --out result/emission_probe_paperenv
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.generalization_common import build_topology_layer, env_from_snapshot  # noqa: E402
from scripts.analysis.run_mobility_stream_temporal import _anticipation_F, _eval_F, _topology  # noqa: E402
from scripts.analysis.run_s2_temporal_ablation import (  # noqa: E402
    ARMS, _EMISSION_FILL, _carry_to_frame, _evaluate as s2_evaluate, _ids,
    _make_model as s2_make_model, _score,
)
from src.losses.horizon_objective import churn_proxy, node_in_load  # noqa: E402
from src.training.temporal_state import node_reliability_state  # noqa: E402
from src.training.training_smoke import _normalized_config, load_training_smoke_config  # noqa: E402
from src.v2x_env.mobility import MobilityConfig, MobilityStream  # noqa: E402
from src.v2x_env.profiles import density_matched_vehicle_config  # noqa: E402
from src.v2x_env.shadow_fading import ShadowField  # noqa: E402
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot  # noqa: E402


def _probe_train(model, spec, frames, train_idx, topology_layer, cfg, *, coupling, epochs, window, dt_frac):
    reset = bool(spec["reset"])
    emission = bool(spec.get("emission"))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    hidden = model.init_state(1).shape[1]
    zero_row = torch.zeros(hidden, dtype=torch.float64)
    captured: list = []

    def hook(_module, args):  # forward_pre_hook on graph_cell: (x, state, state_graph)
        x, state, state_graph = args
        joined = torch.cat([x, state, state_graph], dim=1)
        captured.append((float(x.norm(dim=1).mean()), float(state.norm(dim=1).mean()),
                         float(state_graph.norm(dim=1).mean()), float(joined.std())))
    handle = model.graph_cell.register_forward_pre_hook(hook)

    per_epoch = []
    gates = ("update_gate", "reset_gate", "candidate")
    for epoch in range(epochs):
        captured.clear()
        grad_norms = []
        state = model.init_state(frames[train_idx[0]][1]["candidate"].num_nodes)
        carried_rel = None
        prev_in_load = None
        prev_ids = _ids(frames[train_idx[0]][0])
        opt.zero_grad(set_to_none=True)
        window_loss = frames[train_idx[0]][1]["features"]["distance_m"].new_zeros(())
        steps = 0
        for j, t in enumerate(train_idx):
            snapshot, env = frames[t]
            n_t = env["candidate"].num_nodes
            cur_ids = _ids(snapshot)
            if state is not None and (j > 0 or state.shape[0] != n_t):
                state = _carry_to_frame(state, prev_ids, cur_ids, zero_row)
            carried_rel = _carry_to_frame(carried_rel, prev_ids, cur_ids, _EMISSION_FILL)
            prev_in_load = _carry_to_frame(prev_in_load, prev_ids, cur_ids, 0.0)
            if reset:
                state = model.init_state(n_t)
            score, state = _score(model, env, state, spec, carried_rel=carried_rel)
            topo = _topology(score, env, topology_layer, cfg)
            ev = _eval_F(topo, env, cfg, coupling)
            f_t = ev["F_avalanche_node_mean"].mean()
            ant = _anticipation_F(topo, snapshot, dt_frac, env, cfg, coupling)
            in_load = node_in_load(n_t, topo.dst_index, topo.topology_weight)
            churn = churn_proxy(prev_in_load, in_load) if prev_in_load is not None else f_t.new_zeros(())
            window_loss = window_loss + (f_t + ant + 0.5 * churn)
            steps += 1
            prev_in_load = in_load.detach()
            if emission:
                carried_rel = node_reliability_state(ev)
            prev_ids = cur_ids
            if steps >= window or j == len(train_idx) - 1:
                (window_loss / steps).backward()
                gn = sum(float(getattr(model.graph_cell, g).weight.grad.norm())
                         for g in gates if getattr(model.graph_cell, g).weight.grad is not None)
                grad_norms.append(gn)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                window_loss = window_loss.new_zeros(())
                steps = 0
                state = state.detach()
        cap = captured[:]
        per_epoch.append({
            "epoch": epoch,
            "gate_grad_norm": st.fmean(grad_norms) if grad_norms else float("nan"),
            "x_norm": st.fmean([c[0] for c in cap]) if cap else float("nan"),
            "state_norm": st.fmean([c[1] for c in cap]) if cap else float("nan"),
            "state_graph_norm": st.fmean([c[2] for c in cap]) if cap else float("nan"),
            "joined_std": st.fmean([c[3] for c in cap]) if cap else float("nan"),
        })
    handle.remove()
    return per_epoch


def _build_frames(cfg, density, profile, coupling, n, num_frames, dt, churn, shadow_std, shadow_decorr, scene_seed):
    base = generate_vehicle_snapshot(density_matched_vehicle_config(int(n), float(density), seed=scene_seed))
    mk = dict(dt_s=dt, num_frames=num_frames)
    if churn > 0:
        mk.update(boundary_mode="absorb_inject", churn_rate_per_frame=churn, stream_seed=scene_seed)
    stream = MobilityStream(base, MobilityConfig(**mk))
    topo = build_topology_layer(cfg)
    frames = {t: (stream.frame_at(t),
                  env_from_snapshot(stream.frame_at(t), cfg, label=stream.time_at(t),
                                    interference_coupling_db=coupling))
              for t in range(len(stream))}
    if shadow_std > 0:
        field = ShadowField(shadow_std, shadow_decorr, dt, seed=scene_seed)
        for t in range(len(stream)):
            _s, env = frames[t]
            env["shadow_offset_db"] = torch.as_tensor(
                field.frame_offsets(t, env["candidate"].source, env["candidate"].target), dtype=torch.float64)
    return frames, topo, len(stream)


def main() -> None:
    p = argparse.ArgumentParser(description="Emission-stabilization mechanism probe (T3c)")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--arms", default="full,filter")
    p.add_argument("--init-seeds", default="7,42,123")
    p.add_argument("--scene-seed", type=int, default=7)
    p.add_argument("--density", type=float, default=100.0)
    p.add_argument("--training-profile", default="hard_low_confidence")
    p.add_argument("--coupling-db", type=float, default=20.0)
    p.add_argument("--node-count", type=int, default=400)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=18)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--shadow-std-db", type=float, default=4.0)
    p.add_argument("--shadow-decorr-s", type=float, default=8.0)
    p.add_argument("--churn-rate", type=float, default=2.0)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--out", default="result/emission_probe_paperenv")
    args = p.parse_args()

    config = dict(load_training_smoke_config(str(ROOT / args.config)))
    config["vehicle_count"] = int(args.node_count)
    config["training_profile"] = str(args.training_profile)
    cfg = _normalized_config(config)
    dt_frac = float(args.dt) * 0.5
    frames, topo, nstream = _build_frames(cfg, args.density, args.training_profile, args.coupling_db,
                                          args.node_count, args.num_frames, args.dt, args.churn_rate,
                                          args.shadow_std_db, args.shadow_decorr_s, args.scene_seed)
    n_train = max(1, min(nstream - 1, round(0.6 * nstream)))
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, nstream))

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    init_seeds = [int(s) for s in args.init_seeds.split(",") if s.strip()]
    print(f"Emission mechanism probe: arms={arms} x init_seeds={init_seeds}, cell d{int(args.density)}/"
          f"{args.training_profile}/c{int(args.coupling_db)} (shadow {args.shadow_std_db}dB, churn {args.churn_rate})",
          flush=True)

    runs = []
    for arm in arms:
        for iseed in init_seeds:
            cfg["quenched_quadrature"] = 11
            model = s2_make_model(arm, int(args.hidden_dim), iseed)
            traj = _probe_train(model, ARMS[arm], frames, train_idx, topo, cfg,
                                coupling=args.coupling_db, epochs=args.epochs, window=args.window, dt_frac=dt_frac)
            cfg["quenched_quadrature"] = 21
            rows, _churn = s2_evaluate(model, ARMS[arm], frames, test_idx, topo, cfg, args.coupling_db, dt_frac)
            finalF = st.fmean([r["F"] for r in rows])
            worstF = max(r["F"] for r in rows)
            last = traj[-1]
            runs.append({"arm": arm, "init_seed": iseed, "final_F": finalF, "worst_F": worstF, "trajectory": traj})
            print(f"  arm={arm:16s} init={iseed:>3} | final_F={finalF:.4f} worst_F={worstF:.4f} | "
                  f"end: gate_grad={last['gate_grad_norm']:.3f} state_norm={last['state_norm']:.3f} "
                  f"joined_std={last['joined_std']:.3f}", flush=True)

    # arm-level summary: mean/worst final F + end-of-train state_norm & gate_grad (collapse signature)
    summary = {}
    for arm in arms:
        ar = [r for r in runs if r["arm"] == arm]
        summary[arm] = {
            "final_F_mean": st.fmean([r["final_F"] for r in ar]),
            "worst_F_max": max(r["worst_F"] for r in ar),
            "state_norm_end_max": max(r["trajectory"][-1]["state_norm"] for r in ar),
            "gate_grad_end_mean": st.fmean([r["trajectory"][-1]["gate_grad_norm"] for r in ar]),
        }
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "emission_probe.json").write_text(
        json.dumps({"cell": {"density": args.density, "profile": args.training_profile, "coupling_db": args.coupling_db},
                    "arms": arms, "init_seeds": init_seeds, "summary": summary, "runs": runs},
                   indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== arm summary (collapse signature) ===")
    for arm in arms:
        s = summary[arm]
        print(f"  {arm:16s}: final_F_mean={s['final_F_mean']:.4f} worst_F_max={s['worst_F_max']:.4f} | "
              f"state_norm_end_max={s['state_norm_end_max']:.3f} gate_grad_end_mean={s['gate_grad_end_mean']:.3f}")
    print(f"wrote {out_dir / 'emission_probe.json'}")


if __name__ == "__main__":
    main()
