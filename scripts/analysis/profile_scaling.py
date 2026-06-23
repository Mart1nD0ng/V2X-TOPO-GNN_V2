"""End-to-end complexity profiling for the unified mainline (spec H4 / §11.8 -- G9).

Measures wall-clock runtime (and a memory proxy) of the FULL Eq. 56 forward
``build_candidate_graph -> GNN(s,P,n) -> pi (G2) -> ell (G4 collision x G5 FBL) ->
consensus recurrence (G1) -> D/E objectives (G6)`` as the network grows, holding the
spatial *density* fixed (deployment area ``∝ N``) so that the candidate-edge count
``E = O(N)`` -- the honest near-linear regime (a fixed area would make ``E = O(N^2)`` by
construction, which is not what H4 is about).

H4 requires the overall cost to be APPROXIMATELY LINEAR in ``N`` and ``E`` (small constant
factors ``k_poll``, ``k^3`` are allowed).  The danger to rule out is SUPER-linear (quadratic)
blow-up -- e.g. a hidden ``N x N`` tensor or an all-pairs interference sum.  We therefore
report, from reproducible measurements:

  * the log-log power-law exponent of each stage and the end-to-end total vs ``E`` (and ``E``
    vs ``N``) -- quadratic would give exponent ~2, linear ~1;
  * a linear-vs-quadratic fit of ``t`` vs ``E`` -- a genuine quadratic has a *positive*
    leading coefficient comparable to the observed time; overhead-dominated linear data has
    a negligible / negative one;
  * the degree-bucketed ``total_cells <= 2E`` (the D4 structural guarantee of ``O(E)`` memory
    with no ``N x N`` allocation), and an empirical peak-RSS growth curve.

Run standalone:  ``python scripts/analysis/profile_scaling.py``
"""

from __future__ import annotations

import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mainline.global_evaluator import build_bucketed_padding, evaluate_global_consensus  # noqa: E402
from src.mainline.model import (  # noqa: E402
    OperatingPointConfig, PreferenceConditionedTopologyGNN, model_operating_point,
)
from src.mainline.topology import build_candidate_graph  # noqa: E402

DT = torch.float64


def _median_time(fn, reps: int) -> float:
    fn()  # warm-up (page faults / lazy init) so the first rep doesn't bias the median upward
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(sorted(ts)[len(ts) // 2])


def peak_tensor_numel(fn) -> int:
    """Largest single-tensor element count materialised during ``fn()`` (a TorchDispatch hook).

    Deterministic detector of any ``N x N`` (or larger) intermediate -- cdist, broadcast
    ``a[:,None]-a[None]``, matmul ``x @ x.T``, einsum, tuple-``zeros((N,N))`` -- regardless of
    how it is written (the static grep misses several of these idioms).  The honest mainline's
    largest tensor is the quorum-DP cube ``[m*Q,(k+1)^3]`` = ``O(E k^3)``, so this grows
    LINEARLY in E; a genuine ``N x N`` grows as ``N^2`` and is caught by the exponent.
    """
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_map

    peak = {"n": 0}

    class _Tracker(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))

            def rec(x):
                if isinstance(x, torch.Tensor):
                    peak["n"] = max(peak["n"], int(x.numel()))
                return x
            tree_map(rec, out)
            return out

    with _Tracker():
        fn()
    return peak["n"]


def measure_peak_numels(
    Ns, *, density: float = 0.0025, radius: float = 80.0, rounds: int = 8,
    hidden: int = 32, layers: int = 2,
) -> dict:
    """Largest materialised tensor of the full end-to-end forward at each size in ``Ns``.

    Returns ``{"N", "E", "max_numel"}``.  ``max_numel ~ E`` (exponent ~1) for the honest
    O(E) mainline; a hidden ``N x N`` makes it scale as ``N^2`` (exponent ~2).
    """
    model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=hidden, layers=layers).double()
    cfg = OperatingPointConfig(rounds=rounds)
    lam = torch.tensor([0.34, 0.33, 0.33], dtype=DT)
    out: dict = {"N": [], "E": [], "max_numel": []}
    for N in Ns:
        gen = torch.Generator().manual_seed(int(N))
        side = math.sqrt(N / density)
        pos = torch.rand(int(N), 2, generator=gen, dtype=DT) * side
        g = build_candidate_graph(pos, radius)
        nf, ef = _node_features(g, int(N))
        with torch.no_grad():
            mx = peak_tensor_numel(lambda: model_operating_point(model, g, nf, ef, lam, cfg))
        out["N"].append(int(N))
        out["E"].append(int(g.num_edges))
        out["max_numel"].append(int(mx))
    return out


def local_exponent(x, y, k: int = 3) -> float:
    """Log-log slope over only the top ``k`` (largest) points -- a top-of-range exponent that
    is NOT flattened by the fixed small-N overhead (which fools the full-range slope)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return fit_exponent(x[-k:], y[-k:])


def _node_features(g, N):
    src, dst = g.src_index, g.dst_index
    outdeg = torch.bincount(src, minlength=N).to(DT)
    indeg = torch.bincount(dst, minlength=N).to(DT)
    nf = torch.stack([outdeg / outdeg.clamp_min(1).max(),
                      indeg / indeg.clamp_min(1).max(),
                      torch.ones(N, dtype=DT)], dim=1)
    ef = (g.distance / 80.0).unsqueeze(-1)
    return nf, ef


def profile_scaling(
    Ns,
    *,
    density: float = 0.0025,   # nodes per unit area -> side = sqrt(N/density), so E = O(N)
    radius: float = 80.0,
    rounds: int = 8,
    hidden: int = 32,
    layers: int = 2,
    reps: int = 3,
    measure_memory: bool = True,
) -> dict:
    """Profile the end-to-end forward and its stages over a list of network sizes ``Ns``."""
    try:
        import psutil
        proc = psutil.Process()
    except Exception:
        proc = None
    model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=hidden, layers=layers).double()
    cfg = OperatingPointConfig(rounds=rounds)
    lam = torch.tensor([0.34, 0.33, 0.33], dtype=DT)

    res: dict = {k: [] for k in ("N", "E", "t_total", "t_build", "t_gnn", "t_consensus", "rss_mb", "total_cells")}
    gc.collect()
    rss0 = (proc.memory_info().rss / 1e6) if proc else 0.0
    for N in Ns:
        gen = torch.Generator().manual_seed(int(N))
        side = math.sqrt(N / density)
        pos = torch.rand(int(N), 2, generator=gen, dtype=DT) * side

        t_build = _median_time(lambda: build_candidate_graph(pos, radius), reps)
        g = build_candidate_graph(pos, radius)
        src, dst, E = g.src_index, g.dst_index, g.num_edges
        nf, ef = _node_features(g, int(N))
        pad = build_bucketed_padding(src, dst, int(N))

        with torch.no_grad():
            t_gnn = _median_time(lambda: model(nf, ef, src, dst, lam, int(N)), reps)
            s_edge, _, _ = model(nf, ef, src, dst, lam, int(N))
            ell = torch.full((E,), 0.6, dtype=DT)
            omega = torch.ones(1, dtype=DT)

            def _cons():
                return evaluate_global_consensus(
                    num_nodes=int(N), src_index=src, dst_index=dst, log_query_weight=s_edge.unsqueeze(-1),
                    link_reliability=ell.unsqueeze(-1), scenario_weight=omega, k=cfg.k, alpha=cfg.alpha,
                    beta=cfg.beta, rounds=rounds, initial_correct_preference=cfg.initial_correct_preference,
                    return_trajectory=True, padding=pad)
            t_cons = _median_time(_cons, reps)
            t_total = _median_time(lambda: model_operating_point(model, g, nf, ef, lam, cfg), reps)

        rss = (proc.memory_info().rss / 1e6 - rss0) if proc else 0.0
        for key, val in (("N", int(N)), ("E", int(E)), ("t_total", t_total), ("t_build", t_build),
                         ("t_gnn", t_gnn), ("t_consensus", t_cons), ("rss_mb", rss),
                         ("total_cells", int(pad.total_cells))):
            res[key].append(val)
    return res


def fit_exponent(x, y) -> float:
    """Log-log power-law exponent (slope of log y vs log x)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def fit_linear_vs_quadratic(x, y) -> dict:
    """Fit ``t`` vs ``x`` with a linear and a quadratic model; return R^2 and the quadratic
    leading coefficient (its contribution at ``x_max`` relative to the observed max is the
    'is-it-really-quadratic' diagnostic)."""
    x, y = np.asarray(x, float), np.asarray(y, float)

    def r2(coef):
        yh = np.poly1d(coef)(x)
        return float(1 - ((y - yh) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-30))
    cl = np.polyfit(x, y, 1)
    cq = np.polyfit(x, y, 2)
    quad_lead = float(cq[0])
    quad_contrib_ratio = float(quad_lead * x.max() ** 2 / (abs(y).max() + 1e-30))
    return {"r2_linear": r2(cl), "r2_quadratic": r2(cq),
            "quad_lead_coef": quad_lead, "quad_contrib_ratio": quad_contrib_ratio}


def make_figure(res: dict, path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = np.array(res["N"], float)
    E = np.array(res["E"], float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) runtime vs E, log-log, with linear reference
    ax = axes[0]
    for key, lbl in (("t_total", "end-to-end"), ("t_build", "graph build"),
                     ("t_gnn", "GNN"), ("t_consensus", "consensus")):
        ax.loglog(E, np.array(res[key], float) * 1e3, "o-", label=f"{lbl} (exp={fit_exponent(E, res[key]):.2f})")
    ref = (res["t_total"][0] * 1e3) * (E / E[0])  # slope-1 reference
    ax.loglog(E, ref, "k--", alpha=0.5, label="linear (slope 1)")
    ax.set_xlabel("candidate edges E"); ax.set_ylabel("time (ms)")
    ax.set_title("end-to-end runtime vs E (near-linear)"); ax.legend(fontsize=8)
    # (b) E and memory vs N
    ax = axes[1]
    ax.loglog(N, E, "s-", label=f"E (exp={fit_exponent(N, E):.2f})")
    if any(res["rss_mb"]):
        ax.loglog(N, np.array(res["rss_mb"], float).clip(min=1e-3), "^-",
                  label=f"peak RSS MB (exp={fit_exponent(N, np.array(res['rss_mb']).clip(min=1e-3)):.2f})")
    ax.loglog(N, 2 * E, "k:", alpha=0.5, label="2E bucket bound")
    ax.loglog(N, np.array(res["total_cells"], float), "x-", alpha=0.7, label="bucket cells (<=2E)")
    ax.set_xlabel("nodes N"); ax.set_ylabel("count / MB")
    ax.set_title("edges & memory vs N (linear, <= 2E)"); ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


if __name__ == "__main__":
    Ns = [200, 400, 800, 1600, 3200]
    res = profile_scaling(Ns)
    for i, N in enumerate(res["N"]):
        print(f"N={N:5d} E={res['E'][i]:7d} total={res['t_total'][i]*1e3:8.1f}ms "
              f"build={res['t_build'][i]*1e3:7.1f} gnn={res['t_gnn'][i]*1e3:6.1f} "
              f"cons={res['t_consensus'][i]*1e3:7.1f} rss+={res['rss_mb'][i]:6.1f}MB cells={res['total_cells'][i]}")
    print("exponents: t_total~E=%.3f  E~N=%.3f  build~E=%.3f  gnn~E=%.3f  cons~E=%.3f" % (
        fit_exponent(res["E"], res["t_total"]), fit_exponent(res["N"], res["E"]),
        fit_exponent(res["E"], res["t_build"]), fit_exponent(res["E"], res["t_gnn"]),
        fit_exponent(res["E"], res["t_consensus"])))
    print("t~E linear/quad:", fit_linear_vs_quadratic(res["E"], res["t_total"]))
    fig = make_figure(res, ROOT / "docs" / "gate_evidence" / "g9_scaling.png")
    print("figure:", fig)
