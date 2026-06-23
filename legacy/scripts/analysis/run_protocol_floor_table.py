"""Protocol floor table (Roadmap Phase 0.3): the perfect-link reliability lower bound per profile.

The re-measurement discovered that the trained planner's F=0.0642 at the production profile equals
the avalanche protocol's PERFECT-LINK floor (0.0639): with all selected links at success~1, F is a
deterministic function of (ic, iw, k, alpha, beta, rounds) alone — scene- and topology-independent.
A reliability target below that floor is infeasible BY PROTOCOL, and a cell whose best heuristic
already sits on the floor has NO topology headroom.

The floor additionally depends on the DEGREE BUDGET (the #4 effective-degree lever: with degree 4
and k=5 queries the per-round query diversity is constrained, raising the floor; the production
constructor deploys max_out_degree=4). This script tabulates the floor over the
(training profile x avalanche variant x degree budget) grid in the quenched currency (Q=21) and
prints which cells admit F<=target.

Closure-fidelity anchor: the quenched closure's semantics are pinned by the EXACT joint-enumeration
reference (tests/consensus/test_graph_coupled_vs_exact_reference.py) and the MC validation chain
(result/closed_form_validation_v1, docs/CLOSED_FORM_FIDELITY.md) — a hand-rolled MC variant is NOT
authoritative against it and is intentionally not included here.

Usage: python scripts/analysis/run_protocol_floor_table.py --target 0.01
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

from src.evaluation import evaluate_v2x_graph_consensus  # noqa: E402
from src.training.training_smoke import TRAINING_PROFILES  # noqa: E402

# avalanche profile variants: the current production protocol + targeted variants that trade
# rounds/quorum for reliability (candidates for re-anchoring the target).
AVALANCHE_VARIANTS = {
    "small_realistic (k5 a3 b5 r20)": dict(k=5, alpha=3, beta=5, rounds=20),
    "deeper_quorum (k5 a3 b8 r32)": dict(k=5, alpha=3, beta=8, rounds=32),
    "wider_query (k7 a4 b5 r20)": dict(k=7, alpha=4, beta=5, rounds=20),
    "wider+deeper (k7 a4 b8 r32)": dict(k=7, alpha=4, beta=8, rounds=32),
}


def _perfect_ring(n: int = 200, degree: int = 8):
    src = torch.arange(n).repeat_interleave(degree)
    dst = torch.stack([(torch.arange(n) + j + 1) % n for j in range(degree)], dim=1).reshape(-1)
    return src, dst


def floor_quenched(ic: float, iw: float, proto: dict, quench: int = 21, n: int = 200, degree: int = 8) -> float:
    src, dst = _perfect_ring(n, degree)
    ev = evaluate_v2x_graph_consensus(
        num_nodes=n, src_index=src, dst_index=dst,
        topology_weight=torch.ones(src.numel(), dtype=torch.float64),
        distance_m=torch.full((src.numel(),), 10.0, dtype=torch.float64),
        los_flag=torch.ones(src.numel(), dtype=torch.float64),
        node_initial_correct=torch.full((n,), float(ic), dtype=torch.float64),
        node_initial_wrong=torch.full((n,), float(iw), dtype=torch.float64),
        physical_config={"tx_power_dbm": 23.0, "mcs_threshold_db": 8.0, "transition_width_db": 3.0,
                         "interference_proxy_dbm": -95.0},
        avalanche_config={**proto, "eps": 1e-6, "quenched_quadrature": int(quench)},
    )
    return float(ev["F_avalanche_node_mean"])


def main() -> None:
    p = argparse.ArgumentParser(description="Protocol floor table (perfect-link reliability bound)")
    p.add_argument("--target", type=float, default=0.01)
    p.add_argument("--quench", type=int, default=21)
    p.add_argument("--degrees", default="4,8", help="degree budgets (the deployed constructor uses 4)")
    p.add_argument("--run-name", default="protocol_floor_table")
    args = p.parse_args()
    degrees = [int(d) for d in args.degrees.split(",") if d.strip()]

    rows = []
    print(f"Protocol floor table (quenched Q={args.quench}; perfect links; target {args.target})")
    for deg in degrees:
        print(f"\n--- degree budget {deg} (production constructor deploys max_out_degree=4) ---")
        print(f"{'training profile':>28} {'ic/iw':>11} | " + " | ".join(f"{k.split(' ')[0]:>22}" for k in AVALANCHE_VARIANTS))
        for pname, prof in TRAINING_PROFILES.items():
            ic, iw = float(prof["initial_correct"]), float(prof["initial_wrong"])
            floors = {vname: floor_quenched(ic, iw, proto, args.quench, degree=deg)
                      for vname, proto in AVALANCHE_VARIANTS.items()}
            rows.append({"profile": pname, "ic": ic, "iw": iw, "degree": deg, "floors": floors,
                         "feasible_targets": {v: f <= args.target for v, f in floors.items()}})
            marks = " | ".join(f"{f:>18.4f}{'  ok' if f <= args.target else ' :no'}" for f in floors.values())
            print(f"{pname:>28} {ic:.2f}/{iw:.2f} | {marks}")

    out_dir = ROOT / "result" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"target": args.target, "quench": args.quench, "evaluator_currency": f"quenched_Q{args.quench}",
               "rows": rows,
               "note": ("floor = perfect-link closure F (semantics pinned by the exact-enumeration reference "
                        "and the MC validation chain, docs/CLOSED_FORM_FIDELITY.md); a target below the floor "
                        "is infeasible BY PROTOCOL at that (profile, protocol, degree budget)")}
    (out_dir / "floor_table.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {out_dir / 'floor_table.json'}")


if __name__ == "__main__":
    main()
