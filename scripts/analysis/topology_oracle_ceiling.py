"""Phase-7 topology-optimization ceiling study (stop-condition #2) -> result manifest.

Reproducible: directly optimizes the per-scene edge-logit topology and compares to the
uniform / distance heuristics, analytically and under the independent dynamic MC, on a
correlated (one-biased-region) scenario. Writes result/topology_oracle_ceiling/ceiling.json.

Run:  python scripts/analysis/topology_oracle_ceiling.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene, build_scenario  # noqa: E402
from src.optimization.topology_oracle import oracle_vs_heuristics  # noqa: E402


def main() -> None:
    phy = RoundPhysicsConfig(subchannels=10, slots_per_window=40)
    pcfg = ProtocolConfig(k=3, alpha=2, beta=3, r_max=14)
    runs = []
    for seed in range(3):                                  # multiple scene seeds
        scene = build_manhattan_scene(3, 3, 4, block_m=120.0, comm_radius=95.0, int_radius=140.0,
                                      generator=torch.Generator().manual_seed(seed))
        ev = build_scenario("one_biased_region", scene, base_node_err=0.05, region_bias=0.95)
        out = oracle_vs_heuristics(scene, ev, pcfg, phy, steps=80, lr=0.3,
                                   link_override=1.0, mc_trials=5000, seed=seed, mc_seed=100 + seed)
        runs.append({"seed": seed, "N": scene.num_nodes, **out})
    manifest = {
        "study": "topology_oracle_ceiling",
        "stop_condition": "#2 (direct topology oracle must beat heuristics)",
        "protocol": {"k": pcfg.k, "alpha": pcfg.alpha, "beta": pcfg.beta, "r_max": pcfg.r_max},
        "scenario": "one_biased_region (perfect link, isolates the peer-selection lever)",
        "runs": runs,
        "verdict": "PASS -- significant, MC-confirmed topology lever; ESD-GNN (G9) justified"
        if all(r.get("mc_gain", 0) > 0.02 for r in runs) else "FAIL -- stop-condition #2",
    }
    out_dir = ROOT / "result" / "topology_oracle_ceiling"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ceiling.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"verdict": manifest["verdict"],
                      "mc_gains": [round(r.get("mc_gain", float("nan")), 4) for r in runs],
                      "analytic_gains": [round(r["analytic_gain"], 4) for r in runs]}, indent=2))


if __name__ == "__main__":
    main()
