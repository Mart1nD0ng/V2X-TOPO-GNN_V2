"""G10 -- large-scale complexity manifest (spec §11, constraints #4/#11, stop-condition #4).

Builds the FULL canonical pipeline (candidate+interference graphs, ESD-GNN forward, dynamic MC)
at N from ~100 to ~10000 and records wall-time + structural near-linearity evidence:

* average degree stays BOUNDED as N grows (E = O(N) -- local radius, no degree cap);
* graph build, GNN forward, and per-trial MC cost all grow near-linearly in E (= O(N));
* the N~10000 run completes in-memory (no N x N dense tensor -- it would OOM otherwise).

Run: ``python -m scripts.analysis.scaling_benchmark`` -> ``result/scaling/scaling.json``.
This is the reproducible large-scale viability evidence; the headline numbers (G11) come from
the dynamic MC, which is the only tractable evaluator at this scale (the analytic scenario
enumeration is 2^G with G ~ thousands of regions).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.environment import ProtocolConfig, RoundPhysicsConfig, build_manhattan_scene, build_scenario
from src.environment.candidate_graph import build_candidate_graph
from src.environment.interference_graph import build_interference_graph
from src.models import ESDGNN, ESDGNNConfig, ESDGNNQueryPolicy
from src.validation import run_dynamic_mc

GRIDS = (4, 8, 11, 19, 35)          # -> N ~= 96, 448, 880, 2736, 9520
TRIALS = 100


def run() -> dict:
    phy = RoundPhysicsConfig(subchannels=12, slots_per_window=50)
    pcfg = ProtocolConfig(k=3, alpha=2, beta=3, r_max=12)
    torch.manual_seed(0)
    model = ESDGNN(ESDGNNConfig(hidden_dim=24, r=4, n_enc=3, n_refine=2, k=3)).double()
    rows = []
    # warm up (first GNN call pays one-time graph/JIT cost we don't want to attribute to N=96)
    _warm = build_manhattan_scene(3, 3, 2, comm_radius=80.0, generator=torch.Generator().manual_seed(0))
    ESDGNNQueryPolicy(model, _warm).kernel(None)
    for gx in GRIDS:
        sc = build_manhattan_scene(gx, gx, 4, block_m=100.0, comm_radius=80.0, int_radius=160.0,
                                   generator=torch.Generator().manual_seed(0))
        N = sc.num_nodes
        t0 = time.time()
        gc = build_candidate_graph(sc.positions, sc.comm_radius)
        build_interference_graph(sc.positions, sc.int_radius)
        build_s = time.time() - t0
        E = int(gc.num_edges)
        ev = build_scenario("one_biased_region", sc, base_node_err=0.05, region_bias=0.9)
        pol = ESDGNNQueryPolicy(model, sc)
        t0 = time.time(); pol.kernel(None); gnn_s = time.time() - t0
        t0 = time.time()
        run_dynamic_mc(sc, ev, pol, pcfg, phy, num_trials=TRIALS,
                       generator=torch.Generator().manual_seed(1), link_override=1.0)
        mc_s = time.time() - t0
        rows.append({"N": N, "E": E, "avg_deg": E / N, "build_s": build_s, "gnn_s": gnn_s,
                     "mc_s": mc_s, "us_per_trial_per_edge": mc_s / TRIALS / E * 1e6})
        print(f"N={N:>6} E={E:>7} deg={E/N:>5.1f} build={build_s:>6.2f}s gnn={gnn_s:>6.2f}s "
              f"mc{TRIALS}={mc_s:>7.2f}s  {mc_s/TRIALS/E*1e6:>6.2f} us/trial/edge")
    return {"trials": TRIALS, "rows": rows}


if __name__ == "__main__":
    out = run()
    p = Path("result/scaling"); p.mkdir(parents=True, exist_ok=True)
    (p / "scaling.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {p / 'scaling.json'}")
