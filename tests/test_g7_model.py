"""G7 (spec §3.7, §10): preference-conditioned GNN + augmented-Chebyshev Pareto model.

Checks:
  1. FiLM, augmented_chebyshev (Eq. 57), pareto_indices, sample_simplex are correct.
  2. The GNN forward produces query/power/blocklength logits of the right shapes and is
     differentiable end-to-end through the full (F, D, E) pipeline.
  3. The preference lambda genuinely affects the outputs (FiLM conditioning).
  4. After a short training, sweeping lambda from ONE checkpoint yields >= 3 mutually
     non-dominated (F, D, E) operating points with correct directional steering.
"""

from __future__ import annotations

import math

import torch

from src.mainline.model import (
    FiLM,
    OperatingPointConfig,
    PreferenceConditionedTopologyGNN,
    augmented_chebyshev,
    directional_steering,
    model_operating_point,
    pareto_indices,
    sample_simplex,
    train_preference_model,
)
from src.mainline.topology import build_candidate_graph

torch.manual_seed(0)
DT = torch.float64


# realistic ~20-80 m link distances so the geometry-grounded SINR (and thus F) is sensitive
SCALE, RADIUS = 60.0, 80.0


def _graph_and_feats(N=8, seed=11):
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(N, 2, generator=gen, dtype=DT) * SCALE
    g = build_candidate_graph(pos, RADIUS)
    src, dst = g.src_index, g.dst_index
    outdeg = torch.bincount(src, minlength=N).to(DT)
    indeg = torch.bincount(dst, minlength=N).to(DT)
    nf = torch.stack([outdeg / outdeg.max(), indeg / indeg.clamp_min(1).max(), torch.ones(N, dtype=DT)], dim=1)
    ef = (g.distance / RADIUS).unsqueeze(-1)
    return g, nf, ef


def test_augmented_chebyshev_formula():
    z = torch.tensor([0.5, 0.1, 0.3], dtype=DT)
    lam = torch.tensor([0.6, 0.3, 0.1], dtype=DT)
    z_star = torch.tensor([0.0, 0.0, 0.0], dtype=DT)
    s = torch.tensor([1.0, 0.2, 0.5], dtype=DT)
    rho = 0.05
    out = float(augmented_chebyshev(z, lam, z_star, s, rho))
    t = lam * (z - z_star) / s
    ref = float(t.max() + rho * t.sum())
    assert abs(out - ref) < 1e-12


def test_pareto_indices():
    pts = [(1.0, 1.0), (2.0, 0.0), (0.0, 2.0), (3.0, 3.0)]
    nd = pareto_indices(pts)
    assert set(nd) == {0, 1, 2}  # (3,3) is dominated by (1,1)
    # dead-zone regression: a point dominating by a tiny (but real) margin IS detected
    # (the old asymmetric +1e-9 / -1e-6 tolerance missed sub-1e-6 dominators -> over-reported)
    near = [(0.5, 0.064, 0.44), (0.5 + 5e-7, 0.064 + 5e-7, 0.44 + 5e-7)]
    assert pareto_indices(near) == [0]  # pt0 strictly dominates pt1


def test_sample_simplex():
    s = sample_simplex(1000, dim=3, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(s.sum(dim=1), torch.ones(1000, dtype=DT), atol=1e-12)
    assert torch.all(s >= 0)


def test_film_conditioning():
    film = FiLM(pref_dim=3, hidden=8).double()
    h = torch.randn(4, 8, dtype=DT)
    out_a = film(h, torch.tensor([1.0, 0.0, 0.0], dtype=DT))
    out_b = film(h, torch.tensor([0.0, 0.0, 1.0], dtype=DT))
    assert not torch.allclose(out_a, out_b)  # preference modulates features


def test_model_forward_shapes_and_grad():
    g, nf, ef = _graph_and_feats(N=6)
    model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=16, layers=2).double()
    lam = torch.tensor([0.4, 0.3, 0.3], dtype=DT)
    q, p, b = model(nf, ef, g.src_index, g.dst_index, lam, g.num_nodes)
    assert q.shape == (g.num_edges,) and p.shape == (g.num_nodes,) and b.shape == (g.num_nodes,)
    # end-to-end differentiable through the full (F,D,E) pipeline
    cfg = OperatingPointConfig(rounds=6)
    out = model_operating_point(model, g, nf, ef, lam, cfg)
    loss = out["F"] + 10.0 * out["D"] + out["E"]
    loss.backward()
    grad_norm = sum(float(prm.grad.norm()) for prm in model.parameters() if prm.grad is not None)
    assert math.isfinite(grad_norm) and grad_norm > 0


def test_lambda_affects_operating_point():
    g, nf, ef = _graph_and_feats(N=6)
    model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=16, layers=2).double()
    cfg = OperatingPointConfig(rounds=6)
    with torch.no_grad():
        oa = model_operating_point(model, g, nf, ef, torch.tensor([1.0, 0.0, 0.0], dtype=DT), cfg)
        ob = model_operating_point(model, g, nf, ef, torch.tensor([0.0, 0.0, 1.0], dtype=DT), cfg)
    assert abs(float(oa["E"]) - float(ob["E"])) > 1e-9 or abs(float(oa["F"]) - float(ob["F"])) > 1e-9


_SWEEP = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [.5, 0, .5], [0, .5, .5], [.34, .33, .33]]


def _train_and_sweep(seed, steps, blind=False):
    g, nf, ef = _graph_and_feats(N=8, seed=seed)
    cfg = OperatingPointConfig(rounds=8, payload_bits=8000.0, p_min_dbm=18.0, p_max_dbm=32.0, subchannels=5.0)
    torch.manual_seed(seed)
    model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=32, layers=2).double()

    def fwd(lam):
        return model_operating_point(model, g, nf, ef, lam, cfg)

    train_preference_model(model, fwd, steps=steps, refresh=100, blind=blind, seed=seed)
    import numpy as np
    pts = []
    with torch.no_grad():
        for lam in _SWEEP:
            o = fwd(torch.tensor(lam, dtype=DT))
            pts.append((float(o["F"]), float(o["D"]), float(o["E"])))
    A = np.array(pts)
    rel = [float(np.ptp(A[:, i]) / (np.mean(A[:, i]) + 1e-12)) for i in range(3)]
    return pts, rel


def test_single_checkpoint_sweeps_pareto_front():
    # one checkpoint, periodic-refresh Chebyshev training, sweep lambda -> Pareto front
    pts, rel = _train_and_sweep(seed=11, steps=400)
    assert len(pareto_indices(pts)) >= 3, pts
    # robust argmin-at-vertex steering: minimiser of each objective emphasises its preference
    assert directional_steering(pts, _SWEEP) >= 2, (directional_steering(pts, _SWEEP), pts)
    # substantive front (not a collapsed point)
    assert max(rel) > 0.05, rel


def test_lambda_blind_ablation_collapses():
    # DISCRIMINATIVE control: a lambda-blind checkpoint must collapse the front (lose spread
    # and steering), proving the conditioned model genuinely USES the preference input.
    _, rel = _train_and_sweep(seed=11, steps=400, blind=False)
    _, brel = _train_and_sweep(seed=11, steps=400, blind=True)
    assert max(brel) < 0.5 * max(rel), (max(brel), max(rel))  # blind front much narrower


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G7 tests passed.")
