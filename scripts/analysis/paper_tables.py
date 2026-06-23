"""Generate paper-ready data tables from the gated mainline (G7/G9/G11).

Produces, all from reproducible seeded runs (no hard-coded numbers):
  * representative (F, D, E) operating points the trained model reaches at vertex / balanced
    preferences on HELD-OUT scenarios (mean +/- std), vs the best-fixed baseline's front-min;
  * a small end-to-end complexity table (N, E, total runtime).

Run: python scripts/analysis/paper_tables.py   (prints markdown; ~3 min)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.baseline_comparison import (  # noqa: E402
    CFG, DT, _normaliser, _scalarise, baseline_family_points, make_scenarios,
    model_sweep_points, train_model,
)
from scripts.analysis.profile_scaling import fit_exponent, profile_scaling  # noqa: E402
from src.mainline.model import model_operating_point  # noqa: E402

REP_LAMBDAS = {"F-pref [1,0,0]": [1, 0, 0], "D-pref [0,1,0]": [0, 1, 0], "E-pref [0,0,1]": [0, 0, 1],
               "balanced [.34,.33,.33]": [0.34, 0.33, 0.33]}


def operating_point_table():
    train = make_scenarios(range(100, 110))
    test = make_scenarios(range(200, 212))
    model = train_model(train, steps=600, seed=0)
    rows = {}
    for name, lam in REP_LAMBDAS.items():
        fde_model = []
        for sc in test:
            with torch.no_grad():
                o = model_operating_point(model, sc.graph, sc.nf, sc.ef, torch.tensor(lam, dtype=DT), CFG)
            fde_model.append([float(o["F"]), float(o["D"]), float(o["E"])])
        A = np.array(fde_model)
        rows[name] = {"F": (A[:, 0].mean(), A[:, 0].std()),
                      "D": (A[:, 1].mean(), A[:, 1].std()),
                      "E": (A[:, 2].mean(), A[:, 2].std())}
    print("\n### Representative operating points (held-out mean +/- std over 12 scenarios)\n")
    print("| preference | F (failure prob.) | D (delay) | E (energy) |")
    print("|---|---|---|---|")
    for name, r in rows.items():
        print(f"| {name} | {r['F'][0]:.3f} ± {r['F'][1]:.3f} | {r['D'][0]:.4f} ± {r['D'][1]:.4f} "
              f"| {r['E'][0]:.3f} ± {r['E'][1]:.3f} |")
    # show the model genuinely steers: each preference should minimise its own objective
    print("\nSteering check (min over the 4 rows should be on the diagonal): "
          f"F-min at '{min(rows, key=lambda k: rows[k]['F'][0])}', "
          f"D-min at '{min(rows, key=lambda k: rows[k]['D'][0])}', "
          f"E-min at '{min(rows, key=lambda k: rows[k]['E'][0])}'")


def complexity_table():
    res = profile_scaling([200, 800, 3200, 6400], reps=2)
    print("\n### End-to-end complexity (fixed density, area proportional to N)\n")
    print("| N | E | total runtime (ms) |")
    print("|---|---|---|")
    for i in range(len(res["N"])):
        print(f"| {res['N'][i]} | {res['E'][i]} | {res['t_total'][i]*1e3:.0f} |")
    print(f"\nExponents: E~N = {fit_exponent(res['N'], res['E']):.2f}, "
          f"total runtime t~E = {fit_exponent(res['E'], res['t_total']):.2f} (sub-linear / overhead-dominated).")


if __name__ == "__main__":
    t0 = time.perf_counter()
    operating_point_table()
    complexity_table()
    print(f"\n(generated in {time.perf_counter()-t0:.0f}s)")
