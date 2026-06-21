"""F4.5 — w=0 reliability-only multi-seed band on operating_point_v1.

The production Pareto sweep trains one model per cost weight (single init). This converts the
"rel-only (w=0) blows up cost AND degrades reliability" observation into an init-seed BAND, by
re-training w=0 (and a w>0 reference) across several init seeds and evaluating held-out F/D/E.

Confirms whether the reliability-degradation of rel-only is robust to initialization, or an artifact.

Usage:
  python -B scripts/analysis/run_w0_seed_band.py --seeds 7,42,123,2024,99 --ref-w 5 \
    --out result/w0_seed_band
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

from scripts.analysis.run_de_ablation import _forward, _metrics  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer, ci95, train_one  # noqa: E402
from src.training.gradient_governance import GradientGovernanceConfig  # noqa: E402
from src.training.training_smoke import _loss_config, load_training_smoke_config  # noqa: E402


def _holdout(model, base_config, n, scene_seeds, eval_q):
    out = []
    for s in scene_seeds:
        e = _scene_env(base_config, n, s)
        l2, c2 = _topology_layer(e["cfg"], e["candidate"].num_nodes)
        with torch.no_grad():
            out.append(_metrics(_forward(model, e, l2, c2, eval_mode=True)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="w=0 multi-seed band (F4.5)")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--seeds", default="7,42,123,2024,99")
    p.add_argument("--ref-w", type=float, default=5.0, help="w>0 reference arm")
    p.add_argument("--train-n", type=int, default=600)
    p.add_argument("--steps", type=int, default=140)
    p.add_argument("--ci-scenes", type=int, default=6)
    p.add_argument("--reliability-target", type=float, default=0.02)
    p.add_argument("--out", default="result/w0_seed_band")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    eval_q = int(base_config.get("eval_quenched_quadrature", base_config.get("quenched_quadrature", 21)))
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    governance = GradientGovernanceConfig.from_name("none")
    arms = {"w0_rel_only": 0.0, f"w{args.ref_w:g}_optimized": float(args.ref_w)}

    print(f"w=0 seed band on {args.config}: seeds={seeds}, ref w={args.ref_w}", flush=True)
    results = {}
    for arm, w in arms.items():
        per_seed = []
        for seed in seeds:
            env = _scene_env(base_config, args.train_n, seed)
            layer, caps = _topology_layer(env["cfg"], env["candidate"].num_nodes)
            loss_cfg = dict(_loss_config(env["cfg"]))
            loss_cfg["reliability_failure_target"] = float(args.reliability_target)
            loss_cfg["reliability_tail_failure_target"] = float(args.reliability_target)
            model, _tr = train_one(env["cfg"], env, layer, caps, loss_cfg, w, governance, args.steps, seed)
            scene_seeds = [seed + 1000 + i for i in range(int(args.ci_scenes))]
            hold = _holdout(model, base_config, args.train_n, scene_seeds, eval_q)
            rec = {"seed": seed, "F": float(np.mean([m["F"] for m in hold])),
                   "D": float(np.mean([m["D"] for m in hold])), "E": float(np.mean([m["E"] for m in hold]))}
            per_seed.append(rec)
            print(f"  {arm} seed={seed}: F={rec['F']:.4f} D={rec['D']:.2f} E={rec['E']:.3e}", flush=True)
        F = [r["F"] for r in per_seed]; D = [r["D"] for r in per_seed]; E = [r["E"] for r in per_seed]
        results[arm] = {"w_cost": w, "per_seed": per_seed,
                        "F": ci95(F), "D": ci95(D), "E": ci95(E)}

    rel = results["w0_rel_only"]; opt = results[f"w{args.ref_w:g}_optimized"]
    summary = {
        "currency": f"quenched eval Q={eval_q}", "config": args.config, "seeds": seeds,
        "arms": results,
        "F_degradation_rel_vs_opt": {"mean": rel["F"]["mean"] - opt["F"]["mean"],
                                     "rel_band": [rel["F"]["mean"] - rel["F"]["ci_halfwidth"], rel["F"]["mean"] + rel["F"]["ci_halfwidth"]],
                                     "opt_band": [opt["F"]["mean"] - opt["F"]["ci_halfwidth"], opt["F"]["mean"] + opt["F"]["ci_halfwidth"]],
                                     "robust_separated": bool(rel["F"]["mean"] - rel["F"]["ci_halfwidth"] > opt["F"]["mean"] + opt["F"]["ci_halfwidth"])},
        "D_blowup_x": rel["D"]["mean"] / opt["D"]["mean"] if opt["D"]["mean"] else None,
        "E_blowup_x": rel["E"]["mean"] / opt["E"]["mean"] if opt["E"]["mean"] else None,
    }
    out_dir = ROOT / args.out; out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "w0_seed_band.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    sep = summary["F_degradation_rel_vs_opt"]["robust_separated"]
    print(f"\nrel-only F={rel['F']['mean']:.4f}+-{rel['F']['ci_halfwidth']:.4f} vs "
          f"optimized F={opt['F']['mean']:.4f}+-{opt['F']['ci_halfwidth']:.4f} | "
          f"bands separated (rel WORSE robustly): {sep}")
    print(f"D blow-up {summary['D_blowup_x']:.1f}x, E blow-up {summary['E_blowup_x']:.1f}x")
    print(f"wrote {out_dir / 'w0_seed_band.json'}")


if __name__ == "__main__":
    main()
