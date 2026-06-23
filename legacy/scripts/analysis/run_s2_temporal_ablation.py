"""S2/B1 capacity-matched temporal ablation — does the graph-coupled GRU genuinely help, free of the
capacity/init confound? Trains each arm under a temporal/stream (horizon) objective over a mobility frame
stream with the CORRECTED quenched evaluator (#1), and evaluates held-out future frames.

The rigorous core is THREE regimes of the IDENTICAL graph_gru model (same architecture, same parameter
count, same xavier init) — only the temporal information flow differs:
  * full      : carry state across frames + B1 graph diffusion ON.
  * no_graph  : carry state, diffusion forced to zero (parameter-identical PLAIN recurrence control).
  * no_memory : state reset every frame (parameter-identical MEMORYLESS control).
So any difference among {full, no_graph, no_memory} is PURELY temporal/graph information, not capacity or
init. Two extra context arms (capacity-UNMATCHED, reported separately): the natural memoryless
`static` scorer and the legacy `gru` TemporalGNNScorer.

Comparisons that validate B1:
  full vs no_memory  -> does temporal MEMORY help?            (identical params)
  full vs no_graph   -> does the B1 GRAPH COUPLING help,      (identical params)
                        beyond plain recurrence?

Usage (one arm/seed -> RESULT_JSON last line): python scripts/analysis/run_s2_temporal_ablation.py --arm full --seed 7
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

from scripts.analysis.generalization_common import build_topology_layer, caps_for, env_from_snapshot  # noqa: E402
from scripts.analysis.run_mobility_stream_temporal import _anticipation_F, _eval_F, _selected_pairs, _topology  # noqa: E402
from src.losses.horizon_objective import churn_proxy, node_in_load  # noqa: E402
from src.models import HierarchicalGNNScorer, TemporalGNNScorer  # noqa: E402
from src.topology.temporal_metrics import mean_sequence_churn  # noqa: E402
from src.training.temporal_state import node_reliability_state, remap_carried_by_node_id  # noqa: E402
from src.training.training_smoke import _normalized_config, load_training_smoke_config  # noqa: E402
from src.v2x_env.mobility import MobilityConfig, MobilityStream  # noqa: E402
from src.v2x_env.profiles import production_like_density_v0_vehicle_config  # noqa: E402
from src.v2x_env.shadow_fading import ShadowField  # noqa: E402
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot  # noqa: E402

# kind: static (memoryless scorer) | temporal (TemporalGNNScorer). For temporal arms: cell, whether to
# zero the B1 diffusion, whether to reset state each frame. full/no_graph/no_memory share graph_gru.
# "filter" (Roadmap Phase 1.2, the FILTERING model): graph_gru + the carried-reliability EMISSION as a
# 6th node feature — the previous frame's realized per-node P(correct), computed THROUGH the (shadowed)
# evaluator, i.e. a true observation of the hidden channel state. filter-vs-full isolates the emission;
# filter-vs-no_memory isolates emission+memory. (Input-layer delta: +1 input column, a few hundred of
# 141k params — report n_params.)
ARMS = {
    "static": dict(kind="static"),
    "gru": dict(kind="temporal", cell="gru", zero_graph=False, reset=False),
    "full": dict(kind="temporal", cell="graph_gru", zero_graph=False, reset=False),
    "no_graph": dict(kind="temporal", cell="graph_gru", zero_graph=True, reset=False),
    "no_memory": dict(kind="temporal", cell="graph_gru", zero_graph=False, reset=True),
    "filter": dict(kind="temporal", cell="graph_gru", zero_graph=False, reset=False, emission=True),
    # C3 mechanism isolation (T3b): complete the emission x graph-coupling 2x2.
    #   filter_nomem    = emission + state RESET  -> isolates "carried observation alone" from recurrence.
    #   no_graph_filter = emission + graph diffusion OFF -> "does emission rescue plain recurrence?".
    "filter_nomem": dict(kind="temporal", cell="graph_gru", zero_graph=False, reset=True, emission=True),
    "no_graph_filter": dict(kind="temporal", cell="graph_gru", zero_graph=True, reset=False, emission=True),
}
CAPACITY_MATCHED = {"full", "no_graph", "no_memory", "filter", "filter_nomem", "no_graph_filter"}
_EMISSION_FILL = 0.5  # neutral carried reliability before any frame has been evaluated


def _make_model(arm: str, hidden: int, seed: int):
    spec = ARMS[arm]
    torch.manual_seed(int(seed))
    node_dim = 6 if spec.get("emission") else 5  # filter arm reads the carried-reliability emission
    if spec["kind"] == "static":
        return HierarchicalGNNScorer(
            5, 5, hidden_dim=hidden, message_layers=2, init_mode="xavier",
            use_structural_score_bias=False, enable_budget_head=False, enable_region_bridge_head=False,
            enable_sector_head=False, enable_role_head=False, learnable_score_gain=True,
            score_output_gain=10.0, score_standardization=True,
        ).double()
    return TemporalGNNScorer(
        node_dim, 5, hidden_dim=hidden, message_layers=2, temporal_cell=spec["cell"], init_mode="xavier",
        learnable_score_gain=True, score_output_gain=10.0, score_standardization=True,
    ).double()


def _score(model, env, state, spec, carried_rel=None):
    f = env["features"]
    nf = f["node_features"]
    if spec.get("emission"):
        n = nf.shape[0]
        if carried_rel is None:
            column = nf.new_full((n, 1), _EMISSION_FILL)
        else:
            column = torch.clamp(carried_rel.detach().to(nf).reshape(-1, 1), 0.0, 1.0)
        nf = torch.cat([nf, column], dim=1)
    if spec["kind"] == "static":
        out = model(
            num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
            node_features=nf, edge_features=f["edge_features"],
            region_id=f["region_id"], num_regions=f["num_regions"],
            edge_sector_id=f["edge_sector_id"], edge_is_cross_region=f["edge_is_cross_region"],
            use_structural_score_bias=False,
        )
        return out["edge_score"], None
    return model(
        num_nodes=env["candidate"].num_nodes, src_index=f["src_index"], dst_index=f["dst_index"],
        node_features=nf, edge_features=f["edge_features"],
        region_id=f["region_id"], num_regions=f["num_regions"],
        state=state, zero_graph_state=spec["zero_graph"],
    )


def _ids(snapshot):
    return snapshot["node_id"]


def _carry_to_frame(value, prev_ids, cur_ids, fill):
    """node_id-keyed carry of per-node values/state across a (possibly churned) frame boundary.
    Identity when the populations match (fixed-population streams are byte-identical)."""
    if value is None:
        return None
    return remap_carried_by_node_id(value, prev_ids, cur_ids, fill=fill)


def _train(model, spec, frames, train_idx, topology_layer, cfg, *, coupling, epochs, window, dt_frac,
           w_rel, w_ant, w_churn, lr):
    temporal = spec["kind"] == "temporal"
    reset = temporal and spec["reset"]
    emission = bool(spec.get("emission"))
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    hidden = model.init_state(1).shape[1] if temporal else 0
    zero_row = torch.zeros(hidden, dtype=torch.float64) if temporal else None
    for _epoch in range(epochs):
        n0 = frames[train_idx[0]][1]["candidate"].num_nodes
        state = model.init_state(n0) if temporal else None
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
            # churn-safe: re-key all per-node carried quantities by node_id (identity when no churn)
            if temporal and state is not None and (j > 0 or state.shape[0] != n_t):
                state = _carry_to_frame(state, prev_ids, cur_ids, zero_row)
            carried_rel = _carry_to_frame(carried_rel, prev_ids, cur_ids, _EMISSION_FILL)
            prev_in_load = _carry_to_frame(prev_in_load, prev_ids, cur_ids, 0.0)
            if reset:
                state = model.init_state(n_t)  # memoryless control: fresh state each frame
            score, state = _score(model, env, state, spec, carried_rel=carried_rel)
            topo = _topology(score, env, topology_layer, cfg)
            ev = _eval_F(topo, env, cfg, coupling)
            f_t = ev["F_avalanche_node_mean"].mean()
            ant = _anticipation_F(topo, snapshot, dt_frac, env, cfg, coupling)
            in_load = node_in_load(n_t, topo.dst_index, topo.topology_weight)
            churn = churn_proxy(prev_in_load, in_load) if prev_in_load is not None else f_t.new_zeros(())
            window_loss = window_loss + (w_rel * f_t + w_ant * ant + w_churn * churn)
            steps += 1
            prev_in_load = in_load.detach()
            if emission:
                carried_rel = node_reliability_state(ev)  # the hidden-state observation for frame t+1
            prev_ids = cur_ids
            flush = (not temporal) or reset or steps >= window or j == len(train_idx) - 1
            if flush:
                (window_loss / steps).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                window_loss = window_loss.new_zeros(())
                steps = 0
                if temporal and not reset:
                    state = state.detach()
    return model


def _evaluate(model, spec, frames, idx, topology_layer, cfg, coupling, dt_frac):
    temporal = spec["kind"] == "temporal"
    reset = temporal and spec["reset"]
    emission = bool(spec.get("emission"))
    hidden = model.init_state(1).shape[1] if temporal else 0
    zero_row = torch.zeros(hidden, dtype=torch.float64) if temporal else None
    n0 = frames[0][1]["candidate"].num_nodes
    state = model.init_state(n0) if temporal else None
    carried_rel = None
    prev_ids = _ids(frames[0][0])
    rows, pairs = [], []
    with torch.no_grad():
        if temporal and not reset:  # warm the carried state over preceding frames (online use)
            for t in range(idx[0]):
                snapshot, env = frames[t]
                cur_ids = _ids(snapshot)
                state = _carry_to_frame(state, prev_ids, cur_ids, zero_row)
                carried_rel = _carry_to_frame(carried_rel, prev_ids, cur_ids, _EMISSION_FILL)
                score, state = _score(model, env, state, spec, carried_rel=carried_rel)
                if emission:  # the filter arm needs the realized emission during warm-up too
                    topo = _topology(score, env, topology_layer, cfg)
                    carried_rel = node_reliability_state(_eval_F(topo, env, cfg, coupling))
                prev_ids = cur_ids
        for t in idx:
            snapshot, env = frames[t]
            n_t = env["candidate"].num_nodes
            cur_ids = _ids(snapshot)
            if temporal and state is not None:
                state = _carry_to_frame(state, prev_ids, cur_ids, zero_row)
            carried_rel = _carry_to_frame(carried_rel, prev_ids, cur_ids, _EMISSION_FILL)
            if reset:
                state = model.init_state(n_t)
            score, state = _score(model, env, state, spec, carried_rel=carried_rel)
            topo = _topology(score, env, topology_layer, cfg)
            ev = _eval_F(topo, env, cfg, coupling)
            f_t = float(ev["F_avalanche_node_mean"].mean())
            ant = float(_anticipation_F(topo, snapshot, dt_frac, env, cfg, coupling))
            if emission:
                carried_rel = node_reliability_state(ev)
            rows.append({"frame": t, "F": f_t, "F_transition": ant})
            pairs.append(_selected_pairs(topo))
            prev_ids = cur_ids
    return rows, mean_sequence_churn(pairs, max(int(f[1]["candidate"].num_nodes) for f in frames.values()))


def _heuristic_stream(frames, idx, topology_layer, cfg, coupling, kind: str) -> float:
    """Stream heuristics with FAIR information sets (Phase 1.4): evaluated on the same held-out
    frames, same constructor, same (shadowed) evaluator. 'channel' ranks by the channel feature.
    'carried' is the carried-reliability heuristic — it gets the SAME emission the filter model gets
    (its own previous frame's realized P(correct), node_id-remapped under churn), ranking edges by
    the receiver's carried reliability with a small channel tie-break."""
    carried = None
    prev_ids = _ids(frames[0][0])
    fs = []
    with torch.no_grad():
        for t in range(idx[0] + len(idx)):
            snapshot, env = frames[t]
            cur_ids = _ids(snapshot)
            carried = _carry_to_frame(carried, prev_ids, cur_ids, _EMISSION_FILL)
            f = env["features"]
            if kind == "channel":
                score = f["edge_features"][:, 2].to(torch.float64)
            else:  # carried
                if carried is None:
                    carried = f["node_features"].new_full((env["candidate"].num_nodes,), _EMISSION_FILL)
                score = carried[f["dst_index"]] + 0.01 * f["edge_features"][:, 2].to(torch.float64)
            topo = _topology(score.reshape(-1), env, topology_layer, cfg)
            ev = _eval_F(topo, env, cfg, coupling)
            if kind == "carried":
                carried = node_reliability_state(ev)
            if t in idx:
                fs.append(float(ev["F_avalanche_node_mean"].mean()))
            prev_ids = cur_ids
    return statistics.fmean(fs)


def main() -> None:
    p = argparse.ArgumentParser(description="S2/B1 capacity-matched temporal ablation")
    p.add_argument("--config", default="configs/production_training_v1.yaml")
    p.add_argument("--arm", default="full", choices=sorted(ARMS))
    p.add_argument("--seed", type=int, default=7)
    # P0-2 protocol: separate the SCENE seed (which layout is drawn) from the INIT seed (model init),
    # so reported variance can be decomposed into sigma_scene vs sigma_init. Default None -> both fall
    # back to --seed (byte-identical to the legacy single-seed runs; result/s2_ablation* reproduce).
    p.add_argument("--scene-seed", type=int, default=None)
    p.add_argument("--init-seed", type=int, default=None)
    p.add_argument("--node-count", type=int, default=400)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--dt", type=float, default=2.0)
    p.add_argument("--train-fraction", type=float, default=0.6)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--coupling-db", type=float, default=10.0)
    p.add_argument("--w-rel", type=float, default=1.0)
    p.add_argument("--w-ant", type=float, default=1.0)
    p.add_argument("--w-churn", type=float, default=0.5)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--train-quench", type=int, default=11)
    p.add_argument("--eval-quench", type=int, default=21)
    # P1-1.2: opt-in hidden AR(1) shadow fading (the first real hidden temporal state). std_db<=0 -> OFF
    # (byte-identical). When ON, the shadow is a model-UNOBSERVABLE carried channel state, so a temporal
    # model that tracks it can in principle help -> this is the substrate that makes the temporal
    # question live (vs the fully-observable default). Seeded by the SCENE seed so both arms see the
    # SAME channel realisation (the comparison stays fair).
    p.add_argument("--shadow-std-db", type=float, default=0.0)
    p.add_argument("--shadow-decorr-s", type=float, default=8.0)
    # P1-1.3 / Phase 0.2: opt-in birth/death churn (absorb_inject boundary). 0 -> OFF (byte-identical
    # fixed-population stream). Per-node carried state (GRU, emission, in-load) is re-keyed by node_id
    # across frames, so recurrence survives births/deaths.
    p.add_argument("--churn-rate", type=float, default=0.0)
    # Phase 1.3 fine stage: run the stream at an advantage-map CELL. Default None -> the legacy
    # production density profile / the config's training profile (byte-identical).
    p.add_argument("--density", type=float, default=None, help="veh/km^2 (density_matched profile)")
    p.add_argument("--training-profile", default=None, help="override the config's ic profile")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    spec = ARMS[args.arm]
    scene_seed = int(args.scene_seed) if args.scene_seed is not None else int(args.seed)
    init_seed = int(args.init_seed) if args.init_seed is not None else int(args.seed)
    config = dict(load_training_smoke_config(str(ROOT / args.config)))
    config["vehicle_count"] = int(args.node_count)
    if args.training_profile is not None:
        config["training_profile"] = str(args.training_profile)
    cfg = _normalized_config(config)
    dt_frac = float(args.dt) * 0.5

    if args.density is not None:
        from src.v2x_env.profiles import density_matched_vehicle_config
        base = generate_vehicle_snapshot(
            density_matched_vehicle_config(int(args.node_count), float(args.density), seed=scene_seed)
        )
    else:
        base = generate_vehicle_snapshot(production_like_density_v0_vehicle_config(int(args.node_count), seed=scene_seed))
    mobility_kwargs = dict(dt_s=args.dt, num_frames=args.num_frames)
    if float(args.churn_rate) > 0.0:
        mobility_kwargs.update(boundary_mode="absorb_inject", churn_rate_per_frame=float(args.churn_rate),
                               stream_seed=scene_seed)
    stream = MobilityStream(base, MobilityConfig(**mobility_kwargs))
    n_train = max(1, min(len(stream) - 1, round(args.train_fraction * len(stream))))
    train_idx = list(range(n_train))
    test_idx = list(range(n_train, len(stream)))
    topology_layer = build_topology_layer(cfg)
    frames = {}
    for t in range(len(stream)):
        snap = stream.frame_at(t)
        frames[t] = (snap, env_from_snapshot(snap, cfg, label=stream.time_at(t), interference_coupling_db=args.coupling_db))
    # P1-1.2: precompute the hidden AR(1) shadow ONCE in frame order (seeded by the scene seed), then
    # stash per-frame per-candidate-edge offsets into the env so they are constant across epochs and
    # IDENTICAL across arms (each arm is a separate process rebuilding the same scene -> same draws).
    if float(args.shadow_std_db) > 0.0:
        field = ShadowField(float(args.shadow_std_db), float(args.shadow_decorr_s), float(args.dt), seed=scene_seed)
        for t in range(len(stream)):
            _snap, env = frames[t]
            off = field.frame_offsets(t, env["candidate"].source, env["candidate"].target)
            env["shadow_offset_db"] = torch.as_tensor(off, dtype=torch.float64)

    model = _make_model(args.arm, int(args.hidden_dim), init_seed)
    n_params = int(sum(p.numel() for p in model.parameters()))
    common = dict(coupling=args.coupling_db, epochs=args.epochs, window=args.window, dt_frac=dt_frac,
                  w_rel=args.w_rel, w_ant=args.w_ant, w_churn=args.w_churn, lr=float(cfg["learning_rate"]))

    cfg["quenched_quadrature"] = int(args.train_quench)  # train under the (faster) quenched evaluator
    _train(model, spec, frames, train_idx, topology_layer, cfg, **common)

    cfg["quenched_quadrature"] = int(args.eval_quench)   # evaluate at converged quench
    rows, churn = _evaluate(model, spec, frames, test_idx, topology_layer, cfg, args.coupling_db, dt_frac)
    # Phase 1.4 fair-information-set baselines on the SAME held-out frames / shadow realisation.
    heur_channel_F = _heuristic_stream(frames, test_idx, topology_layer, cfg, args.coupling_db, "channel")
    heur_carried_F = _heuristic_stream(frames, test_idx, topology_layer, cfg, args.coupling_db, "carried")
    mean = statistics.fmean
    result = {
        "arm": args.arm, "seed": int(args.seed), "scene_seed": scene_seed, "init_seed": init_seed,
        "n_params": n_params, "shadow_std_db": float(args.shadow_std_db),
        "shadow_decorr_s": float(args.shadow_decorr_s), "churn_rate": float(args.churn_rate),
        "heur_channel_F": heur_channel_F, "heur_carried_F": heur_carried_F,
        "capacity_matched": bool(args.arm in CAPACITY_MATCHED),
        "F": mean([r["F"] for r in rows]),
        "F_transition": mean([r["F_transition"] for r in rows]),
        "worstF": max(r["F"] for r in rows),
        "churn": float(churn),
        "n_train": n_train, "n_test": len(test_idx),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print("RESULT_JSON " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
