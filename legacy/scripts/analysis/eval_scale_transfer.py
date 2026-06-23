"""D4 — scale-generalization transfer regret (Roadmap Envelope-Paper deliverable 4).

Upgrades the claim "a single checkpoint transfers N=100->10000 without breaking" to a measured
TRANSFER REGRET against a from-scratch N=10000 expert. Both checkpoints are evaluated IDENTICALLY at
N=10000 on the SAME scene layouts at the headline eval currency (quenched Q=21), so the comparison is
apples-to-apples and currency-honest:

    regret = F_planner(N=10000) - F_expert(N=10000)

F is failure (lower is better), so regret > 0 means the N-randomized planner pays a cost vs a
scale-specific expert ("transfers but underperforms"); regret <= ~band means "generalizes across
scale". Reported with a +-std band over several scene seeds.

Usage:
    python -B scripts/analysis/eval_scale_transfer.py \
        --planner result/planner_paperenv --expert result/scale_anchor_n10000 \
        --node-count 10000 --scene-seeds 42,7,123 --out result/scale_transfer
"""
from __future__ import annotations

import argparse
import json
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
    metrics,
    model_score,
)
from scripts.analysis.train_planner import base_vehicle_config  # noqa: E402
from src.training.training_smoke import _make_model  # noqa: E402
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot  # noqa: E402


def _agg(vals: list[float]) -> tuple[float, float]:
    m = sum(vals) / len(vals)
    if len(vals) == 1:
        return m, 0.0
    return m, (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def _load_model(ckpt_dir: Path, cfg: dict):
    model = _make_model(cfg)
    sd = torch.load(ckpt_dir / "planner.pt", map_location="cpu")
    model.load_state_dict(sd)
    return model.double().eval()


def _eval_F_at(model, cfg_eval: dict, node_count: int, scene_seed: int, topo) -> float:
    base = generate_vehicle_snapshot(base_vehicle_config(cfg_eval, int(node_count), int(scene_seed)))
    env = env_from_snapshot(base, cfg_eval, label=(int(node_count), int(scene_seed)))
    with torch.no_grad():
        return float(metrics(evaluate(model_score(model, env), env, topo, cfg_eval))["F"])


def main() -> None:
    p = argparse.ArgumentParser(description="Scale-generalization transfer regret (N-ladder vs from-scratch expert)")
    p.add_argument("--planner", default="result/planner_paperenv",
                   help="N-ladder domain-randomized checkpoint (the 'transfers' arm)")
    p.add_argument("--expert", default="result/scale_anchor_n10000",
                   help="from-scratch single-N expert (the anchor)")
    p.add_argument("--node-count", type=int, default=10000)
    p.add_argument("--scene-seeds", default="42,7,123")
    p.add_argument("--out", default="result/scale_transfer")
    args = p.parse_args()

    planner_dir = ROOT / args.planner
    expert_dir = ROOT / args.expert
    meta = json.loads((planner_dir / "planner_meta.json").read_text(encoding="utf-8"))
    cfg = dict(meta["cfg"])  # the planner's full normalized config (both checkpoints share this architecture)
    # Headline eval currency: quenched eval Q (=21), not the train Q (=11) the planner.pt scalability used.
    cfg_eval = dict(cfg)
    eval_q = int(cfg.get("eval_quenched_quadrature", cfg.get("quenched_quadrature", 21)))
    cfg_eval["quenched_quadrature"] = eval_q

    seeds = [int(x) for x in args.scene_seeds.split(",") if x.strip()]
    topo = build_topology_layer(cfg_eval)
    n = int(args.node_count)
    print(f"Scale transfer @N={n}, eval Q={eval_q}, scene seeds={seeds}", flush=True)

    planner = _load_model(planner_dir, cfg)
    expert = _load_model(expert_dir, cfg)

    rows = []
    for s in seeds:
        fp = _eval_F_at(planner, cfg_eval, n, s, topo)
        fe = _eval_F_at(expert, cfg_eval, n, s, topo)
        rows.append({"scene_seed": s, "F_planner": fp, "F_expert": fe, "regret": fp - fe})
        print(f"  seed={s:>4}  F_planner={fp:.4f}  F_expert={fe:.4f}  regret={fp - fe:+.4f}", flush=True)

    fp_m, fp_s = _agg([r["F_planner"] for r in rows])
    fe_m, fe_s = _agg([r["F_expert"] for r in rows])
    rg_m, rg_s = _agg([r["regret"] for r in rows])
    verdict = ("generalizes across scale (regret within band of 0)" if rg_m - rg_s <= 0.0
               else "transfers without divergence but underperforms a scale-specific expert")
    summary = {
        "node_count": n, "eval_quenched_quadrature": eval_q, "scene_seeds": seeds,
        "planner_dir": args.planner, "expert_dir": args.expert,
        "F_planner_mean": fp_m, "F_planner_std": fp_s,
        "F_expert_mean": fe_m, "F_expert_std": fe_s,
        "transfer_regret_mean": rg_m, "transfer_regret_std": rg_s,
        "verdict": verdict, "currency": "quenched eval Q=%d" % eval_q, "per_seed": rows,
    }
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scale_transfer.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nF_planner={fp_m:.4f}+-{fp_s:.4f}  F_expert={fe_m:.4f}+-{fe_s:.4f}  "
          f"regret={rg_m:+.4f}+-{rg_s:.4f}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {out_dir / 'scale_transfer.json'}")


if __name__ == "__main__":
    main()
