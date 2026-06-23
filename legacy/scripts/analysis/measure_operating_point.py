"""Characterise the re-calibrated operating point: train a planner at the flood extreme (w=0) and
at the operating knee (w=0.1), deploy on held-out scenes, and report reliability/cost together with
the LEARNED effective degree, mean selected in-degree, and link-success / n_tx distribution.

This produces the numbers the re-calibration report needs:
  * effective degree (1/sum w^2 unique-peer) under the loose cap-8 ceiling -> shows it lands ~3-4
    (adaptive), not pinned at the cap;
  * n_tx = 1/link_success stays > 1 at the optimum -> the retransmission lever is genuinely live
    (not escaped to near-perfect links).

Usage: python -B scripts/analysis/measure_operating_point.py --config configs/operating_point_v1.yaml
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
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer, train_one  # noqa: E402
from scripts.analysis.validate_operating_point_montecarlo import _evaluate_details, _select_topology  # noqa: E402
from src.training.gradient_governance import GradientGovernanceConfig  # noqa: E402
from src.training.training_smoke import _loss_config, load_training_smoke_config  # noqa: E402


def _q(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {"p10": 0.0, "p50": 0.0, "p90": 0.0, "mean": 0.0}
    return {"p10": float(np.quantile(a, .1)), "p50": float(np.quantile(a, .5)),
            "p90": float(np.quantile(a, .9)), "mean": float(a.mean())}


def _deploy(model, base_config, n, scene_seeds, eval_q):
    Fs, Ds, Es, effd, indeg, links = [], [], [], [], [], []
    for s in scene_seeds:
        env = _scene_env(base_config, n, s)
        layer, caps = _topology_layer(env["cfg"], env["candidate"].num_nodes)
        with torch.no_grad():
            m = _metrics(_forward(model, env, layer, caps, eval_mode=True))
        Fs.append(m["F"]); Ds.append(m["D"]); Es.append(m["E"])
        topo = _select_topology(model, env, layer, caps)
        det = _evaluate_details(topo, env, quenched_quadrature=eval_q)
        sup = det["avalanche_details"]["query_support"]
        effd.append(float(sup.effective_unique_peer_degree.mean()))
        # mean selected in-degree (peers per node actually wired in)
        ndst = int(topo.dst_index.reshape(-1).numel())
        indeg.append(ndst / float(env["candidate"].num_nodes))
        links.append(det["channel_diagnostics"]["link_success"].detach().to(torch.float64).numpy())
    link_all = np.concatenate(links) if links else np.asarray([])
    ntx_all = 1.0 / np.clip(link_all, 1e-3, 1.0)
    return {
        "F": float(np.mean(Fs)), "D": float(np.mean(Ds)), "E": float(np.mean(Es)),
        "effective_degree": float(np.mean(effd)), "mean_in_degree": float(np.mean(indeg)),
        "link_success_q": _q(link_all), "ntx_q": _q(ntx_all),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Re-calibrated operating-point characterisation")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--node-count", type=int, default=600)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--weights", default="0,0.1")
    p.add_argument("--reliability-target", type=float, default=0.02)
    p.add_argument("--out", default="result/operating_point_measure")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    eval_q = int(base_config.get("eval_quenched_quadrature", base_config.get("quenched_quadrature", 21)))
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    governance = GradientGovernanceConfig.from_name("pcgrad")
    scene_seeds = [args.seed + 1000 + i for i in range(4)]
    print(f"Operating-point measure [{args.config}] N={args.node_count} steps={args.steps} eval Q={eval_q} "
          f"cap={base_config.get('max_out_degree')}", flush=True)

    env = _scene_env(base_config, args.node_count, args.seed)
    layer, caps = _topology_layer(env["cfg"], env["candidate"].num_nodes)
    base_loss = dict(_loss_config(env["cfg"]))
    base_loss["reliability_failure_target"] = float(args.reliability_target)
    base_loss["reliability_tail_failure_target"] = float(args.reliability_target)
    base_loss["use_reliability_gate"] = True

    out = {"config": args.config, "node_count": args.node_count, "eval_quench": eval_q,
           "cap": base_config.get("max_out_degree"), "arms": []}
    for w in weights:
        model, _ = train_one(env["cfg"], env, layer, caps, base_loss, w, governance, args.steps, args.seed)
        d = _deploy(model, base_config, args.node_count, scene_seeds, eval_q)
        d["w_cost"] = w
        out["arms"].append(d)
        print(f"  w={w:>4}: F={d['F']:.4f} D={d['D']:7.2f} E={d['E']:.3e} | eff_deg={d['effective_degree']:.2f} "
              f"mean_in_deg={d['mean_in_degree']:.2f} | link.p50={d['link_success_q']['p50']:.3f} "
              f"ntx.p50={d['ntx_q']['p50']:.2f} ntx.p90={d['ntx_q']['p90']:.2f}", flush=True)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "operating_point_measure.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_dir / 'operating_point_measure.json'}", flush=True)


if __name__ == "__main__":
    main()
