"""G9 (spec H4 / §11.8): end-to-end near-linear complexity profiling.

Fast checks of the profiling machinery and the structural guarantees:
  1. ``profile_scaling`` returns consistent arrays; edge count is monotone in N and the
     degree-bucketed layout never exceeds ``2E`` (O(E) memory, no N x N).
  2. ``E`` scales ~linearly with ``N`` at fixed density.
  3. The fit helpers are DISCRIMINATIVE: ``fit_exponent`` recovers slope ~1 for linear and
     ~2 for quadratic synthetic data, and ``fit_linear_vs_quadratic`` flags a genuine
     quadratic (large positive quad contribution, R2_quad >> R2_lin) while passing linear
     data (small/negative quad contribution).
  4. The measured end-to-end runtime is not super-linear over a modest range.

The full multi-size runtime/memory acceptance (and the figure) live in ``scripts/gates/gate_g9.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import math

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "analysis"))
from profile_scaling import (  # noqa: E402
    _node_features, fit_exponent, fit_linear_vs_quadratic, measure_peak_numels,
    peak_tensor_numel, profile_scaling,
)

DT = torch.float64


def test_profile_arrays_and_bucket_memory_bound():
    res = profile_scaling([100, 200, 400], reps=1)
    n = len(res["N"])
    assert n == 3
    for key in ("E", "t_total", "t_build", "t_gnn", "t_consensus", "total_cells"):
        assert len(res[key]) == n
    assert res["E"][0] < res["E"][1] < res["E"][2]  # edges grow with N
    # degree-bucketed layout is O(E): total padded cells <= 2E for every size (no N x N)
    for i in range(n):
        assert res["total_cells"][i] <= 2 * res["E"][i]


def test_edges_linear_in_N():
    res = profile_scaling([200, 400, 800, 1600], reps=1)
    exp = fit_exponent(res["N"], res["E"])
    assert 0.85 <= exp <= 1.30, exp  # near-linear (slight boundary-effect super-linearity)


def test_fit_exponent_recovers_known_slopes():
    x = np.array([1, 2, 4, 8, 16], float)
    assert abs(fit_exponent(x, 3.0 * x) - 1.0) < 1e-9          # linear -> slope 1
    assert abs(fit_exponent(x, 2.0 * x ** 2) - 2.0) < 1e-9     # quadratic -> slope 2


def test_fit_linear_vs_quadratic_is_discriminative():
    x = np.linspace(1, 10, 8)
    # linear (with intercept/overhead): quadratic term not needed
    lin = fit_linear_vs_quadratic(x, 5.0 + 2.0 * x)
    assert lin["r2_linear"] > 0.999 and lin["quad_contrib_ratio"] < 0.1
    # genuine quadratic: large positive quad contribution, quad fit clearly better
    quad = fit_linear_vs_quadratic(x, 0.5 * x ** 2)
    assert quad["quad_contrib_ratio"] > 0.5 and quad["r2_quadratic"] > quad["r2_linear"] + 0.01


def test_end_to_end_not_superlinear():
    res = profile_scaling([200, 400, 800, 1600], reps=2)
    exp = fit_exponent(res["E"], res["t_total"])
    assert exp < 1.5, exp  # definitively not quadratic (overhead-dominated -> typically < 1)


def test_peak_numel_guard_is_discriminative():
    # ANTI-DEGRADATION §7: a genuinely quadratic (N×N) implementation MUST fail the gate.
    # The honest mainline's largest tensor is the O(E k^3) quorum cube -> scales ~linearly in E.
    honest = measure_peak_numels([1600, 3200])
    honest_exp = fit_exponent(honest["E"], honest["max_numel"])
    assert honest_exp < 1.30, (honest_exp, honest["max_numel"])  # gate's numel_linear (<1.4) PASSES

    # inject a real N×N broadcast all-pairs op (a grep-evading idiom) into the profiled forward
    from src.mainline.model import (
        OperatingPointConfig, PreferenceConditionedTopologyGNN, model_operating_point,
    )
    from src.mainline.topology import build_candidate_graph
    model = PreferenceConditionedTopologyGNN(3, 1, hidden=16, layers=2).double()
    cfg = OperatingPointConfig(rounds=6)
    lam = torch.tensor([0.34, 0.33, 0.33], dtype=DT)
    Es, nums = [], []
    for N in [1600, 3200]:
        gen = torch.Generator().manual_seed(N)
        pos = torch.rand(N, 2, generator=gen, dtype=DT) * math.sqrt(N / 0.0025)
        g = build_candidate_graph(pos, 80.0)
        nf, ef = _node_features(g, N)

        def fwd():
            d = pos.unsqueeze(0) - pos.unsqueeze(1)   # [N, N, 2] all-pairs (regex-evading)
            _ = (d * d).sum(-1)                       # [N, N]
            with torch.no_grad():
                return model_operating_point(model, g, nf, ef, lam, cfg)
        nums.append(peak_tensor_numel(fwd))
        Es.append(g.num_edges)
    injected_exp = fit_exponent(Es, nums)
    assert injected_exp > 1.50, (injected_exp, nums)  # N×N detected -> gate's numel_linear FAILS


def test_pi_head_is_bucketed_O_E():
    # the inclusion-probability head must use the O(E) bucketed layout, not a dense [N,max_deg]
    # (so peak tensor stays O(E) even under degree skew); largest tensor scales ~linearly in E.
    nm = measure_peak_numels([800, 1600, 3200])
    assert fit_exponent(nm["E"], nm["max_numel"]) < 1.30, nm["max_numel"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} G9 tests passed.")
