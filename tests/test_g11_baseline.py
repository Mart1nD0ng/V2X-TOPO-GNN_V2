"""G11 (spec §4 ultimate): baseline-comparison machinery + discriminativeness.

Fast checks of the comparison harness and a tiny end-to-end sanity (the full multi-seed
significance study is the gate's job, ``scripts/gates/gate_g11.py``):
  1. ``set_coverage`` / ``_dominates`` are correct Pareto-dominance operators.
  2. ``_hypervolume_mc`` rewards a front closer to the ideal point.
  3. ``paired_significance`` is direction-aware and discriminative.
  4. END-TO-END sanity: a (briefly) trained model Pareto-dominates an untrained control and
     wins hypervolume -- the discriminative core of the gate (an untrained model must lose).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
from baseline_comparison import (  # noqa: E402
    PreferenceConditionedTopologyGNN, _dominates, _hypervolume_mc, _normaliser,
    make_scenarios, model_sweep_points, paired_significance, set_coverage, train_model,
)


def test_dominance_and_coverage():
    assert _dominates((0.1, 0.2, 0.3), (0.2, 0.3, 0.4))           # strictly better in all
    assert not _dominates((0.1, 0.5, 0.3), (0.2, 0.3, 0.4))       # worse in one -> no dominance
    A = [(0.1, 0.1, 0.1)]
    B = [(0.5, 0.5, 0.5), (0.6, 0.4, 0.5)]
    assert set_coverage(A, B) == 1.0                              # A dominates all of B
    assert set_coverage(B, A) == 0.0                              # B dominates none of A


def test_hypervolume_rewards_better_front():
    near = [(0.1, 0.1, 0.1)]
    far = [(0.8, 0.8, 0.8)]
    lo, rng = _normaliser(near + far)
    assert _hypervolume_mc(near, lo, rng) > _hypervolume_mc(far, lo, rng)


def test_paired_significance_direction():
    better = [1.0, 1.0, 1.0, 1.0, 1.0]
    worse = [2.0, 2.0, 2.0, 2.0, 2.0]
    # lower-is-better: model=better should win
    s = paired_significance(better, worse)
    assert s["win_rate"] == 1.0 and s["wilcoxon_p_one_sided"] < 0.05
    # higher-is-better: model=worse should NOT win
    s2 = paired_significance(worse, better, higher_better=True)
    assert s2["win_rate"] == 1.0  # worse > ... no; check direction explicitly
    s3 = paired_significance(better, worse, higher_better=True)
    assert s3["win_rate"] == 0.0  # better(=1) is NOT > worse(=2) when higher is better


def test_trained_beats_untrained_end_to_end():
    train = make_scenarios(range(100, 104))
    test = make_scenarios(range(200, 203))
    model = train_model(train, steps=150, seed=0)
    torch.manual_seed(999)
    untrained = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=32, layers=2).double()
    cmo, com, hv_win = [], [], 0
    for sc in test:
        mp = model_sweep_points(model, sc)
        up = model_sweep_points(untrained, sc)
        cmo.append(set_coverage(mp, up))
        com.append(set_coverage(up, mp))
        lo, rng = _normaliser(mp + up)
        hv_win += _hypervolume_mc(mp, lo, rng) > _hypervolume_mc(up, lo, rng)
    assert np.mean(cmo) > np.mean(com)        # trained dominates more of untrained than vice versa
    assert np.mean(com) < 0.1                 # untrained dominates ~none of the trained front
    assert hv_win == len(test)                # trained wins hypervolume on every scenario


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
