"""Train a variable-N V2X topology-planner GNN and SAVE a deployable checkpoint.

The scorer is size-invariant (shared message-passing weights), so ONE checkpoint runs
at any node count. This trainer makes that explicit: it trains across a SPEC of node
counts (a range like "100-10000" -> a log-spaced ladder, or a list "100,500,2000", or a
single "2000"), sampling a different N each step so the model learns an N-invariant
scoring function. The P1 scale-invariant backward loss makes each N's gradient
contribution comparable, so mixed-N training is well-posed. It also trains across
mobility frames (vehicles advanced by dt) so the planner generalises to the continuous
motion the demonstrator replays.

Writes <out>/planner.pt (state_dict) + <out>/planner_meta.json. The meta embeds the FULL
normalized config (so the model is rebuilt exactly via _make_model(meta["cfg"]) without
depending on the YAML file), the node-count ladder + per-N trained F/C, and a label for
the operating point (so the GUI can switch between checkpoints).

Usage:
    python -B scripts/analysis/train_planner.py --config configs/production_training_v1.yaml \
        --node-counts 100-10000 --label production --out result/planner_production
    python -B scripts/analysis/train_planner.py --config configs/operating_point_v1.yaml \
        --node-counts 100-10000 --label operating_point --out result/planner_paperenv
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.generalization_common import (  # noqa: E402
    build_topology_layer,
    env_from_snapshot,
    evaluate,
    learned_F,
    model_score,
)
from src.losses import compute_coupled_loss  # noqa: E402
from src.training.training_smoke import (  # noqa: E402
    _loss_config,
    _make_model,
    _normalized_config,
    load_training_smoke_config,
)
from src.v2x_env.profiles import (  # noqa: E402
    density_matched_vehicle_config,
    production_like_density_v0_vehicle_config,
)
from src.v2x_env.vehicle_snapshot import advance_vehicle_snapshot, generate_vehicle_snapshot  # noqa: E402


def parse_node_counts(spec: str, ladder_points: int) -> list[int]:
    """Parse a node-count spec into a sorted list of node counts.

    "100-10000" -> log-spaced ladder of `ladder_points` values (inclusive of both ends);
    "100,500,2000" -> exactly those; "2000" -> single value.
    """
    spec = str(spec).strip()
    if "," in spec:
        vals = [int(x) for x in spec.split(",") if x.strip()]
    elif "-" in spec and not spec.startswith("-"):
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        if lo <= 0 or hi <= 0:
            raise ValueError("node counts must be positive")
        if lo > hi:
            lo, hi = hi, lo
        if lo == hi:
            vals = [lo]
        else:
            pts = max(int(ladder_points), 2)
            vals = [
                int(round(math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / (pts - 1))))
                for i in range(pts)
            ]
    else:
        vals = [int(spec)]
    out = sorted({v for v in vals if v > 0})
    if not out:
        raise ValueError(f"no valid node counts parsed from spec {spec!r}")
    return out


def base_vehicle_config(cfg: dict, node_count: int, seed: int) -> dict:
    """Build an initial vehicle layout per the config's density profile (holds density
    constant as N changes, so local graph statistics stay N-invariant)."""
    profile = str(cfg.get("vehicle_profile", "production_like_density_v0"))
    if profile == "density_matched":
        density = float(cfg.get("node_density_per_km2") or 200.0)
        return density_matched_vehicle_config(node_count, density, seed=seed)
    return production_like_density_v0_vehicle_config(node_count, seed=seed)


def build_env_pool(cfg: dict, node_counts: list[int], train_times: list[float], scene_seed: int) -> list:
    """One env per (node_count, mobility_time). Different layout seed per N."""
    pool = []
    for i, n in enumerate(node_counts):
        base = generate_vehicle_snapshot(base_vehicle_config(cfg, int(n), scene_seed + 17 * i))
        for t in train_times:
            snap = advance_vehicle_snapshot(base, float(t)) if t else base
            pool.append((int(n), float(t), env_from_snapshot(snap, cfg, label=(int(n), float(t)))))
        del base
    gc.collect()
    return pool


def train_multi_n(cfg: dict, pool: list, topology_layer, max_steps: int, *, model_seed: int):
    """Mixed-N training: each step trains on a shuffled env from the pool (P1 scale-invariant
    backward makes the per-N gradient magnitudes comparable). gc bounds peak memory at large N."""
    torch.manual_seed(int(model_seed))
    model = _make_model(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    loss_cfg = _loss_config(cfg)
    rng = random.Random(int(model_seed))
    order = list(range(len(pool)))
    for step in range(int(max_steps)):
        if step % len(pool) == 0:
            rng.shuffle(order)
        _, _, env = pool[order[step % len(pool)]]
        opt.zero_grad(set_to_none=True)
        ev = evaluate(model_score(model, env), env, topology_layer, cfg)
        lo = compute_coupled_loss(ev, loss_cfg)
        (lo.get("effective_backward_loss", lo["total_loss"])).backward()
        opt.step()
        if (step + 1) % 16 == 0:
            gc.collect()
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="Train and save a variable-N deployable topology planner")
    p.add_argument("--config", default="configs/production_training_v1.yaml")
    p.add_argument("--node-counts", default="100-10000",
                   help="range 'lo-hi' (log-spaced ladder), list 'a,b,c', or single 'n'")
    p.add_argument("--ladder-points", type=int, default=6, help="ladder size for a range spec")
    p.add_argument("--label", default=None, help="operating-point label for the GUI (default: config stem)")
    p.add_argument("--scene-seed", type=int, default=42)
    p.add_argument("--model-seed", type=int, default=42)
    p.add_argument("--dt", type=float, default=3.0)
    p.add_argument("--train-times", default="0,6", help="comma seconds of mobility frames to train across")
    p.add_argument("--max-steps", type=int, default=240)
    p.add_argument("--reliability-target", type=float, default=None, help="override config reliability target")
    p.add_argument("--out", default="result/planner")
    args = p.parse_args()

    config = dict(load_training_smoke_config(str(ROOT / args.config)))
    if args.reliability_target is not None:
        config["reliability_failure_target"] = float(args.reliability_target)
        config["reliability_tail_failure_target"] = float(args.reliability_target)
    # vehicle_count is set per-step by the pool; seed a nominal value for normalization.
    config.setdefault("vehicle_count", 2000)
    cfg = _normalized_config(config)

    node_counts = parse_node_counts(args.node_counts, args.ladder_points)
    train_times = [float(t) for t in str(args.train_times).split(",") if t.strip()] or [0.0]
    label = args.label or Path(args.config).stem

    print(f"Train variable-N planner [{label}]: spec={args.node_counts} -> N ladder {node_counts}, "
          f"profile={cfg.get('vehicle_profile')}, frames@{train_times}s, {args.max_steps} steps...", flush=True)

    topology_layer = build_topology_layer(cfg)
    pool = build_env_pool(cfg, node_counts, train_times, int(args.scene_seed))
    model = train_multi_n(cfg, pool, topology_layer, int(args.max_steps), model_seed=int(args.model_seed))

    # Scalability table: trained F/C per ladder node count (at the base frame t=0).
    scalability = []
    for n, t, env in pool:
        if t != 0.0:
            continue
        f, c = learned_F(model, env, topology_layer, cfg)
        scalability.append({"node_count": int(n), "F": float(f), "C": float(c)})
    print("  trained F/C across N: " + ", ".join(f"N={r['node_count']} F={r['F']:.4f}/C={r['C']:.4f}"
                                                  for r in scalability), flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "planner.pt")
    meta = {
        "label": label,
        "config": args.config,
        "node_count_spec": args.node_counts,
        "node_counts": node_counts,
        "node_min": int(min(node_counts)),
        "node_max": int(max(node_counts)),
        "train_times": train_times,
        "max_steps": int(args.max_steps),
        "scene_seed": int(args.scene_seed),
        "model_seed": int(args.model_seed),
        "dt": float(args.dt),
        "vehicle_profile": cfg.get("vehicle_profile"),
        "node_density_per_km2": cfg.get("node_density_per_km2"),
        "scalability_F_C": scalability,
        "cfg": cfg,  # full normalized config -> self-describing checkpoint
    }
    (out_dir / "planner_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"saved checkpoint: {out_dir / 'planner.pt'}")
    print(f"saved meta:       {out_dir / 'planner_meta.json'}")


if __name__ == "__main__":
    main()
