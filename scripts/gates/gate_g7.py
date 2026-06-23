"""G7 (spec §3.7 / module 3.7): preference-conditioned Pareto model.

Acceptance: a SINGLE trained checkpoint, swept over the preference simplex
``lambda = (lambda_F, lambda_D, lambda_E)``, yields multiple (>= 3) mutually non-dominated
(F, D, E) operating points, with each preference STEERING its objective (the minimiser of
objective m over the sweep emphasises lambda_m).  Discriminative by construction: a
lambda-BLIND ablation (constant preference) collapses the front and loses steering, proving
the conditioning is genuine rather than scatter (a non-dominated count alone is not
discriminative).  Robust across two seeds.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from _common import GateResult, main_single, run_pytest  # type: ignore

from src.mainline.model import (  # noqa: E402
    OperatingPointConfig, PreferenceConditionedTopologyGNN, directional_steering,
    model_operating_point, pareto_indices, train_preference_model,
)
from src.mainline.topology import build_candidate_graph  # noqa: E402

DT = torch.float64
SCALE, RADIUS = 60.0, 80.0  # realistic ~20-80 m link distances (geometry-grounded SINR)
CFG = OperatingPointConfig(rounds=10, payload_bits=8000.0, p_min_dbm=18.0, p_max_dbm=32.0, subchannels=5.0)
SWEEP = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
         [.34, .33, .33], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]


def _graph_and_feats(N, seed):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(N, 2, generator=gen, dtype=DT) * SCALE
    g = build_candidate_graph(pos, RADIUS)
    src, dst = g.src_index, g.dst_index
    outdeg = torch.bincount(src, minlength=N).to(DT)
    indeg = torch.bincount(dst, minlength=N).to(DT)
    nf = torch.stack([outdeg / outdeg.max(), indeg / indeg.clamp_min(1).max(), torch.ones(N, dtype=DT)], dim=1)
    ef = (g.distance / RADIUS).unsqueeze(-1)
    return g, nf, ef


def _checkpoint(seed, steps, blind=False):
    g, nf, ef = _graph_and_feats(8, seed)
    torch.manual_seed(seed)
    model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=32, layers=2).double()

    def fwd(lam):
        return model_operating_point(model, g, nf, ef, lam, CFG)

    train_preference_model(model, fwd, steps=steps, refresh=100, blind=blind, seed=seed)
    pts = []
    with torch.no_grad():
        for lam in SWEEP:
            o = fwd(torch.tensor(lam, dtype=DT))
            pts.append((float(o["F"]), float(o["D"]), float(o["E"])))
    A = np.array(pts)
    rel = [float(np.ptp(A[:, i]) / (np.mean(A[:, i]) + 1e-12)) for i in range(3)]
    return pts, rel, len(pareto_indices(pts)), directional_steering(pts, SWEEP)


def run() -> GateResult:
    evidence: dict = {}

    pts, rel, nd, steer = _checkpoint(seed=11, steps=600)
    evidence["non_dominated (of 10)"] = f"{nd}"
    evidence["relative_spread F/D/E"] = f"{rel[0]:.2f} / {rel[1]:.2f} / {rel[2]:.2f}"
    evidence["argmin_at_vertex_steering (of 3)"] = f"{steer}"
    evidence["F-pref/D-pref/E-pref objective"] = f"F={pts[0][0]:.3f} D={pts[1][1]:.3e} E={pts[2][2]:.3e}"

    # discriminative lambda-blind ablation: collapsed front, lost steering
    _, brel, _, bsteer = _checkpoint(seed=11, steps=600, blind=True)
    evidence["lambda-blind max-spread / steering"] = f"{max(brel):.3f} / {bsteer}"

    # cross-seed robustness
    _, rel2, nd2, steer2 = _checkpoint(seed=23, steps=600)
    evidence["second_checkpoint nd / steering"] = f"{nd2} / {steer2}"

    substantive = sum(r > 0.05 for r in rel) >= 2          # trained front is real (>=2 axes move >5%)
    blind_collapsed = max(brel) < 0.05 and max(brel) < 0.3 * max(rel)  # blind front collapses
    pareto_ok = (
        nd >= 3 and nd2 >= 3 and steer == 3 and steer2 == 3
        and substantive and blind_collapsed
    )

    tests_ok, tail = run_pytest("tests/test_g7_model.py")
    evidence["pytest"] = "passed" if tests_ok else f"FAILED\n{tail}"

    return GateResult(
        gate="G7",
        title="preference-conditioned Pareto model (one checkpoint sweeps the F/D/E front)",
        passed=bool(pareto_ok and tests_ok),
        evidence=evidence,
        notes="FiLM GNN + augmented Chebyshev (Eq.57) with periodic z*/scale refresh; lambda steers F/D/E "
              "(3/3 argmin-at-vertex); lambda-blind ablation collapses; geometry-grounded gamma.",
    )


if __name__ == "__main__":
    sys.exit(main_single(run))
