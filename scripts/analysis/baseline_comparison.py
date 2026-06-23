"""Baseline comparison for the unified mainline (spec G11 -- the ultimate gate).

Trains ONE preference-conditioned topology GNN (src/mainline) across a set of TRAINING road
scenarios and evaluates it on HELD-OUT scenarios against honest, strong, reproducible
baselines.  Every method is scored through the SAME physics/consensus pipeline
(:func:`evaluate_controls`: pi -> ell -> F, D, E); methods differ ONLY in how the controls
``(query logits s, per-node power P, per-node blocklength n)`` are produced, so the comparison
is fair (identical G2/G4/G5/G1/G6 physics, no idealisation, no degree cap).

Baselines (non-learned; the fixed-degree ranking of the old paper is FORBIDDEN, spec §12):
  * ``uniform``  : uniform query (s=0) + a grid of constant (P, n).
  * ``distance`` : prefer near peers (s = -d) + constant (P, n) grid.
  * ``degree``   : prefer high-in-degree peers + constant (P, n) grid.
  * ``random``   : random query logits (several seeds) + constant (P, n) grid.
  * ``best-fixed``: the per-(scenario, preference) ORACLE over the whole non-learned family
    above -- the strongest honest baseline (it gets to pick its best constant policy for each
    scenario AND preference, an advantage the learned model is NOT given).
  * ``lambda-blind``: a GNN trained ignoring the preference (isolates the value of preference
    conditioning); ``untrained``: random-init model (sanity -- must lose).

Primary metric: per held-out scenario, normalise (F, D, E) to [0,1] across all methods, then for
each preference ``lambda`` score the (augmented-Chebyshev) scalarisation; lower is better.  The
model uses its lambda-conditioned point; a fixed-policy baseline uses the BEST family member for
that lambda.  Significance: paired Wilcoxon across held-out scenarios.  Hypervolume (MC) is a
secondary aggregate.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mainline.model import (  # noqa: E402
    OperatingPointConfig, PreferenceConditionedTopologyGNN, augmented_chebyshev,
    evaluate_controls, model_operating_point, sample_simplex,
)
from src.mainline.topology import build_candidate_graph  # noqa: E402

DT = torch.float64
CFG = OperatingPointConfig(rounds=8, payload_bits=8000.0, p_min_dbm=18.0, p_max_dbm=32.0, subchannels=5.0)
# DENSE deployment: SCALE < RADIUS so all vehicles are in radio range -> the candidate graph is
# (near-)complete and the optimisation lever is the POLLING topology (which k peers to query, via
# the G2 k-subset query distribution) + per-node power/blocklength, NOT which links physically
# exist.  (At SCALE >> RADIUS the candidate graph is sparse and some nodes fall below the quorum
# size k, needing the §7.2 candidate-shortage protocol; that regime is out of scope here -- see
# the dense-deployment caveat in the G11 notes.  On a complete graph the 'degree' query heuristic
# coincides with 'uniform', acknowledged in the baseline set.)
SCALE, RADIUS = 60.0, 80.0
RHO = 0.05
# QUERY preferences at which "best achievement" is scored (the lambda we ask each front for)
LAMBDAS = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
           [.34, .33, .33], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]]


def simplex_grid(res: int) -> list:
    """Deterministic triangular simplex grid (``(res+1)(res+2)/2`` preference vectors)."""
    return [[i / res, j / res, (res - i - j) / res]
            for i in range(res + 1) for j in range(res + 1 - i)]


# the model's continuum advantage: ONE checkpoint produces a point for ANY preference.  Sweep a
# DENSE simplex grid (~136 points) so its learned front is budget-matched to the ~150-point
# hand-tuned baseline grid -- a fair front-vs-front comparison.
MODEL_SWEEP = simplex_grid(15)


# --------------------------------------------------------------------------------------------
# scenarios
# --------------------------------------------------------------------------------------------
@dataclass
class Scenario:
    graph: object
    nf: torch.Tensor
    ef: torch.Tensor
    N: int
    seed: int


def make_scenario(seed: int, N: int) -> Scenario:
    gen = torch.Generator().manual_seed(seed)
    pos = torch.rand(N, 2, generator=gen, dtype=DT) * SCALE
    g = build_candidate_graph(pos, RADIUS)
    if int(torch.bincount(g.src_index, minlength=N).min()) < CFG.k:   # always satisfied when complete
        pos = torch.rand(N, 2, generator=gen, dtype=DT) * (SCALE * 0.7)
        g = build_candidate_graph(pos, RADIUS)
    src, dst = g.src_index, g.dst_index
    outdeg = torch.bincount(src, minlength=N).to(DT)
    indeg = torch.bincount(dst, minlength=N).to(DT)
    nf = torch.stack([outdeg / outdeg.clamp_min(1).max(), indeg / indeg.clamp_min(1).max(),
                      torch.ones(N, dtype=DT)], dim=1)
    ef = (g.distance / RADIUS).unsqueeze(-1)
    return Scenario(graph=g, nf=nf, ef=ef, N=N, seed=seed)


def make_scenarios(seeds, n_choices=(8, 10, 12)) -> list:
    out = []
    for i, s in enumerate(seeds):
        N = n_choices[i % len(n_choices)]
        out.append(make_scenario(s, N))
    return out


# --------------------------------------------------------------------------------------------
# training (multi-graph, preference-conditioned augmented Chebyshev with periodic z*/scale refresh)
# --------------------------------------------------------------------------------------------
def train_model(train_scenarios, *, steps=900, lr=5e-3, refresh=75, blind=False, seed=0):
    torch.manual_seed(seed)
    model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=32, layers=2).double()
    gen = torch.Generator().manual_seed(seed + 1)
    blind_lam = torch.tensor([1 / 3, 1 / 3, 1 / 3], dtype=DT)

    def estimate():
        rows = []
        with torch.no_grad():
            for sc in train_scenarios:
                for lam in sample_simplex(4, generator=gen, dtype=DT):
                    o = model_operating_point(model, sc.graph, sc.nf, sc.ef, lam, CFG)
                    rows.append([float(o["F"]), float(o["D"]), float(o["E"])])
        Z = torch.tensor(rows, dtype=DT)
        return Z.min(0).values - 1e-6, (Z.max(0).values - Z.min(0).values).clamp_min(1e-6)

    z_star, scales = estimate()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = torch.randperm(len(train_scenarios), generator=gen)
    for t in range(steps):
        if refresh and t > 0 and t % refresh == 0:
            z_star, scales = estimate()
        sc = train_scenarios[int(order[t % len(order)])]
        lam = blind_lam if blind else sample_simplex(1, generator=gen, dtype=DT)[0]
        o = model_operating_point(model, sc.graph, sc.nf, sc.ef, lam, CFG)
        loss = augmented_chebyshev(torch.stack([o["F"], o["D"], o["E"]]), lam, z_star, scales, RHO)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


# --------------------------------------------------------------------------------------------
# control policies (non-learned baselines), all scored via evaluate_controls
# --------------------------------------------------------------------------------------------
def _query_logits(name: str, sc: Scenario, gen=None) -> torch.Tensor:
    g = sc.graph
    d = g.distance
    if name == "uniform":
        return torch.zeros(g.num_edges, dtype=DT)
    if name == "distance":            # prefer near peers
        return -(d / RADIUS)
    if name == "invdist":             # sharply prefer near peers (inverse distance)
        return RADIUS / d.clamp_min(1.0)
    if name == "degree":              # prefer high-in-degree peers
        indeg = torch.bincount(g.dst_index, minlength=sc.N).to(DT)
        return (indeg[g.dst_index] / indeg.clamp_min(1).max())
    if name == "random":
        return torch.randn(g.num_edges, generator=gen, dtype=DT)
    raise ValueError(name)


def baseline_family_points(sc: Scenario, *, power_grid=7, n_grid=7, random_reps=4, seed=0) -> list:
    """All (F, D, E) operating points of the non-learned family for a scenario (a STRONG,
    dense baseline: 4 query heuristics + several random restarts x a 7x7 constant-(P,n) grid)."""
    g = sc.graph
    gen = torch.Generator().manual_seed(seed + sc.seed)
    powers = torch.linspace(CFG.p_min_dbm, CFG.p_max_dbm, power_grid, dtype=DT)
    blocks = torch.linspace(CFG.n_min, CFG.n_max, n_grid, dtype=DT)
    policies = [("uniform", None), ("distance", None), ("invdist", None), ("degree", None)]
    policies += [("random", r) for r in range(random_reps)]
    pts = []
    with torch.no_grad():
        for pname, _ in policies:
            s = _query_logits(pname, sc, gen)
            for P in powers:
                for n in blocks:
                    Pv = torch.full((sc.N,), float(P), dtype=DT)
                    nv = torch.full((sc.N,), float(n), dtype=DT)
                    o = evaluate_controls(g, s, Pv, nv, CFG)
                    pts.append((float(o["F"]), float(o["D"]), float(o["E"]), pname))
    return pts


def model_sweep_points(model, sc: Scenario, lambdas=MODEL_SWEEP) -> list:
    pts = []
    with torch.no_grad():
        for lam in lambdas:
            o = model_operating_point(model, sc.graph, sc.nf, sc.ef, torch.tensor(lam, dtype=DT), CFG)
            pts.append((float(o["F"]), float(o["D"]), float(o["E"])))
    return pts


# --------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------
def _normaliser(all_fde):
    # robust: lo = ideal (min), hi = 85th percentile so a few wasteful extremes (e.g. a grid
    # point that burns huge energy) do not inflate the range and game the comparison.
    A = np.asarray(all_fde, float)
    lo = A.min(0)
    hi = np.percentile(A, 85, axis=0)
    rng = (hi - lo)
    rng[rng < 1e-12] = 1.0
    return lo, rng


def _scalarise(fde, lam, lo, rng):
    z = (np.asarray(fde, float) - lo) / rng           # normalised objectives in [0,1], 0=best
    t = np.asarray(lam, float) * z
    return float(t.max() + RHO * t.sum())             # augmented weighted Chebyshev, lower=better


def _hypervolume_mc(points_fde, lo, rng, ref=1.0, samples=4000, seed=0):
    """MC dominated hypervolume of a normalised point set vs reference (ref,ref,ref)."""
    if not points_fde:
        return 0.0
    Z = (np.asarray(points_fde, float) - lo) / rng
    Z = Z[(Z <= ref).all(1)]
    if len(Z) == 0:
        return 0.0
    rng_s = np.random.RandomState(seed)
    U = rng_s.rand(samples, 3) * ref
    dominated = (Z[None, :, :] <= U[:, None, :]).all(2).any(1)
    return float(dominated.mean() * ref ** 3)


def _dominates(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return bool((a <= b).all() and (a < b).any())


def set_coverage(A_pts, B_pts) -> float:
    """C(A, B): fraction of B weakly dominated by some point of A (minimisation).  Normalisation-
    FREE Pareto-dominance metric: C(A,B) >> C(B,A) means A's front is genuinely better."""
    if not B_pts:
        return 0.0
    A = [p[:3] for p in A_pts]
    cov = sum(any(_dominates(a, b[:3]) or list(a) == list(b[:3]) for a in A) for b in B_pts)
    return cov / len(B_pts)


def per_scenario_scores(model, blind_model, untrained_model, sc: Scenario, *, family_seed=0) -> dict:
    """Return {method: {"scalar": mean over LAMBDAS, "hv": MC hypervolume}} for one scenario."""
    fam = baseline_family_points(sc, seed=family_seed)
    fam_fde = [p[:3] for p in fam]
    model_pts = model_sweep_points(model, sc)            # ~136 dense-lambda points (budget-matched)
    blind_pts = model_sweep_points(blind_model, sc)
    unt_pts = model_sweep_points(untrained_model, sc)
    model_canon = model_sweep_points(model, sc, lambdas=LAMBDAS)  # 10 canonical, for the diagnostic
    # MODEL-INDEPENDENT normalisation box: defined by the BASELINE family only, so the HV /
    # scalar comparison cannot be inflated by the model contributing most of the pool (the
    # adversarial-review fix).  HV/scalar then measure "in the region the baselines span, whose
    # front dominates more".  Robust (85th pct excludes wasteful family extremes).
    lo, rng = _normaliser(fam_fde)

    def front_scalar(points):
        """Fair FRONT-vs-FRONT achievement: best (min) scalarisation the method's OWN point
        set offers for each query preference, averaged (lower = better).  Budget-matched so a
        learned dense front competes fairly with the hand-tuned grid."""
        return float(np.mean([min(_scalarise(p, lam, lo, rng) for p in points) for lam in LAMBDAS]))

    # secondary diagnostic: does the model's lambda-CONDITIONED point land near the front-min?
    cond = float(np.mean([_scalarise(p, lam, lo, rng) for lam, p in zip(LAMBDAS, model_canon)]))

    fronts = {"best-fixed": fam_fde, "lambda-blind": blind_pts, "untrained": unt_pts}
    for pname in ("uniform", "distance", "invdist", "degree"):
        fronts[f"fixed-{pname}"] = [p[:3] for p in fam if p[3] == pname]

    out = {"model": {"scalar": front_scalar(model_pts), "hv": _hypervolume_mc(model_pts, lo, rng),
                     "conditioned_scalar": cond}}
    for name, fr in fronts.items():
        out[name] = {
            "scalar": front_scalar(fr),
            "hv": _hypervolume_mc(fr, lo, rng),
            "cov_model_over": set_coverage(model_pts, fr),   # C(model, baseline): model dominates baseline
            "cov_over_model": set_coverage(fr, model_pts),   # C(baseline, model): baseline dominates model
        }
    return out


def paired_significance(model_scores, base_scores, *, higher_better=False):
    """Paired comparison across scenarios.  ``higher_better=False`` (scalar regret): model wins
    where model < base.  ``higher_better=True`` (hypervolume): model wins where model > base.
    Returns Wilcoxon p (one-sided, model-better), median improvement, win rate, bootstrap CI."""
    from scipy import stats
    m = np.asarray(model_scores, float)
    b = np.asarray(base_scores, float)
    diff = (m - b) if higher_better else (b - m)   # positive => model better
    win_rate = float((diff > 0).mean())
    median_impr = float(np.median(diff))
    # rel improvement is meaningless when the baseline value is ~0 (e.g. coverage); report None
    rel_impr = (None if float(np.abs(b).max()) < 1e-6
                else float(np.median(diff / np.abs(b).clip(1e-9))))
    try:                                            # one-sided "model better" directly (robust)
        p_one = float(stats.wilcoxon(m, b, alternative=("greater" if higher_better else "less")).pvalue)
    except ValueError:                              # all-zero differences -> not significant
        p_one = 1.0
    # bootstrap 95% CI of the median improvement
    rs = np.random.RandomState(0)
    boot = [np.median(diff[rs.randint(0, len(diff), len(diff))]) for _ in range(2000)]
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {"wilcoxon_p_one_sided": float(p_one), "win_rate": win_rate,
            "median_improvement": median_impr, "median_rel_improvement": rel_impr, "ci95": ci}


def run_comparison(*, train_seeds, test_seeds, n_choices=(8, 10, 12), steps=900, seed=0) -> dict:
    train_sc = make_scenarios(train_seeds, n_choices)
    test_sc = make_scenarios(test_seeds, n_choices)
    model = train_model(train_sc, steps=steps, blind=False, seed=seed)
    blind = train_model(train_sc, steps=steps, blind=True, seed=seed)
    torch.manual_seed(seed + 777)
    untrained = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=32, layers=2).double()

    rows = [per_scenario_scores(model, blind, untrained, sc) for sc in test_sc]
    methods = list(rows[0].keys())
    baselines = [m for m in methods if m != "model"]
    scalar = {m: [r[m]["scalar"] for r in rows] for m in methods}
    hv = {m: [r[m]["hv"] for r in rows] for m in methods}
    cov_mo = {m: [r[m]["cov_model_over"] for r in rows] for m in baselines}   # model dominates baseline
    cov_om = {m: [r[m]["cov_over_model"] for r in rows] for m in baselines}   # baseline dominates model
    sig = {m: paired_significance(scalar["model"], scalar[m]) for m in baselines}
    sig_hv = {m: paired_significance(hv["model"], hv[m], higher_better=True) for m in baselines}
    sig_cov = {m: paired_significance(cov_mo[m], cov_om[m], higher_better=True) for m in baselines}
    return {"methods": methods, "scalar": scalar, "hv": hv, "cov_model_over": cov_mo,
            "cov_over_model": cov_om, "significance": sig, "significance_hv": sig_hv,
            "significance_cov": sig_cov, "n_test": len(test_sc), "test_seeds": list(test_seeds)}


def summarise(res) -> None:
    print(f"held-out scenarios: {res['n_test']}  (test seeds {res['test_seeds'][0]}..{res['test_seeds'][-1]})")
    print(f"{'method':14s} | {'C(mdl>b)':>8s} {'C(b>mdl)':>8s} {'cov.p':>7s} | "
          f"{'HV':>7s} {'hv.win':>6s} {'hv.p':>7s} | {'cheby':>7s} {'ch.win':>6s}")
    for m in res["methods"]:
        if m == "model":
            print(f"{m:14s} | {'--':>8s} {'--':>8s} {'--':>7s} | {np.mean(res['hv'][m]):7.3f}")
            continue
        cmo, com = np.mean(res["cov_model_over"][m]), np.mean(res["cov_over_model"][m])
        hp, sp = res["significance_hv"][m], res["significance"][m]
        cp = res["significance_cov"][m]
        print(f"{m:14s} | {cmo:8.3f} {com:8.3f} {cp['wilcoxon_p_one_sided']:7.4f} | "
              f"{np.mean(res['hv'][m]):7.3f} {hp['win_rate']*100:5.0f}% {hp['wilcoxon_p_one_sided']:7.4f} | "
              f"{np.mean(res['scalar'][m]):7.3f} {sp['win_rate']*100:5.0f}%")


if __name__ == "__main__":
    res = run_comparison(train_seeds=range(100, 116), test_seeds=range(200, 216), steps=900)
    summarise(res)
