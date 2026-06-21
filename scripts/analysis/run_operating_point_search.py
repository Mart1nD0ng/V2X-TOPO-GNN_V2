"""Track A — Non-trivial operating-point search (design contract:
docs/COUPLING_AND_OPERATING_POINT_DESIGN.md).

Goal G2: find an operating point (density x initial-confidence x interference-coupling
x active-degree) where the learned topology *decides success* — i.e. it is hard but
learnable AND beats every heuristic by a wide margin — instead of the trivial
sweet-spot crack (F~1e-12, any topology wins) or the failed basin (F~1, unlearnable).

The search scores each cell by gap_G = best_heuristic_F - learned_F (the quantity that
measures how much topology matters), trains/evaluates on the SAME constructor +
evaluator + coupled loss used in production (held-out layout seed to avoid one-graph
overfit), runs a coarse single-seed grid, shortlists the top cells by gap subject to a
learnability band, then confirms the shortlist across multiple seeds. Decision gates
A-G1/A-G2/A-G3 and the honest off-ramps are defined in the contract and emitted in the
JSON verdict.

Outputs result/<run-name>/: operating_point_search.json, figures/gap_heatmap.png,
RESULT.md. On A-pass the recommended cell is reported for freezing into
configs/operating_point_v1.yaml (the Track-B substrate).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from scripts.analysis.generalization_common import (  # noqa: E402
    best_heuristic_F,
    build_topology_layer,
    env_from_snapshot,
    evaluate,
    metrics,
    train_model,
)
from src.training.training_smoke import (  # noqa: E402
    TRAINING_PROFILES,
    _normalized_config,
    load_training_smoke_config,
)
from src.v2x_env.profiles import density_matched_vehicle_config  # noqa: E402
from src.v2x_env.vehicle_snapshot import generate_vehicle_snapshot  # noqa: E402

_FAIL_BAND = (0.1, 0.6)  # per-node F band that flags a non-trivial (intermediate) regime
_EVAL_SEED_OFFSET = 1000  # eval on a held-out layout (train_seed + offset)


# --------------------------------------------------------------------------- #
# Pure decision helpers (unit-tested in tests/analysis/test_operating_point_search.py)
# --------------------------------------------------------------------------- #
def cell_gap(learned_f: float, heuristic_f: float) -> float:
    """gap_G = best_heuristic_F - learned_F. Higher => topology matters more."""
    return float(heuristic_f) - float(learned_f)


def cell_ratio(learned_f: float, heuristic_f: float) -> float:
    """Heuristic/learned F ratio (>=1 means learned is better). Floor learned_F to avoid /0."""
    return float(heuristic_f) / max(float(learned_f), 1e-12)


def in_band(learned_f: float, low: float, high: float) -> bool:
    """A-i learnable & non-trivial band: not failed basin (F~1), not trivial (F~1e-12)."""
    return float(low) <= float(learned_f) <= float(high)


def topology_decides(
    gap: float, ratio: float, learned_f: float, *, gap_min: float, ratio_min: float, f_floor: float
) -> bool:
    """A-G2: large absolute gap AND >=ratio_min edge AND not trivially supercritical."""
    return float(gap) >= float(gap_min) and float(ratio) >= float(ratio_min) and float(learned_f) >= float(f_floor)


def select_shortlist(cells: list[dict], k: int, *, band_low: float, band_high: float) -> list[dict]:
    """Cells inside the learnability band, sorted by gap_G descending, top-k."""
    band = [
        c
        for c in cells
        if isinstance(c.get("learned_F"), (int, float)) and in_band(c["learned_F"], band_low, band_high)
    ]
    return sorted(band, key=lambda c: c["gap_G"], reverse=True)[: max(int(k), 0)]


def make_verdict(a_g1: bool, recommended: dict | None) -> str:
    """Three-branch verdict per the contract decision tree."""
    if recommended is not None:
        return (
            "ADOPT (A-G1 & A-G2 & A-G3 pass): operating point "
            f"density={recommended['density']:.0f} conf={recommended['confidence']} "
            f"coupling={recommended['coupling_db']:.0f}dB degree={recommended['degree']} "
            f"(gap_mean={recommended['gap_mean']:.4f}, ratio_mean={recommended['ratio_mean']:.2f}). "
            "Freeze into configs/operating_point_v1.yaml and proceed to Track B."
        )
    if a_g1:
        return (
            "REFRAME (A-G1 ok, A-G2/A-G3 fail): task is learnable but the topology edge is "
            "modest or unstable -> claim 'reliability-optimal sparse topology at scale'; DROP "
            "'topology decisively matters'. Consider a richer channel/quorum substrate before "
            "claiming more. Do NOT start Track B."
        )
    return (
        "NO LEARNABLE BAND (A-G1 fail): every cell is failed-basin or trivial -> the closed-form "
        "consensus is too bistable to host a non-trivial point. Escalate to a SUBSTRATE change "
        "(softer quorum / graded reliability). Do NOT start Track B."
    )


def _cell_key(cell: dict) -> tuple:
    """Identity of a (coarse or confirm) cell for resume de-duplication. Includes the
    training-step budget so re-runs at a different budget are distinct experiments
    (not served stale from the cache)."""
    return (
        round(float(cell["density"]), 6),
        str(cell["confidence"]),
        round(float(cell["coupling_db"]), 6),
        int(cell["degree"]),
        int(cell["node_count"]),
        int(cell["steps"]),
        int(cell["train_seed"]),
    )


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a partial final line from an interrupted run
    return rows


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True) + "\n")
        fh.flush()


# --------------------------------------------------------------------------- #
# Experiment primitives
# --------------------------------------------------------------------------- #
def _cell_cfg(base_config: dict, *, node_count: int, confidence: str, degree: int, coupling: float) -> dict:
    """Normalized cfg with this cell's operating-point overrides (does not mutate base)."""
    if confidence not in TRAINING_PROFILES:
        raise ValueError(f"confidence profile must be one of {sorted(TRAINING_PROFILES)}, got {confidence!r}")
    config = dict(base_config)
    config["vehicle_count"] = int(node_count)
    config["training_profile"] = str(confidence)
    config["max_out_degree"] = int(degree)
    physical = dict(config.get("physical", {}))
    physical["interference_density_coupling_db"] = float(coupling)
    config["physical"] = physical
    return _normalized_config(config)


def _make_env(cfg: dict, node_count: int, density: float, seed: int, coupling: float):
    snap = generate_vehicle_snapshot(density_matched_vehicle_config(node_count, density, seed=seed))
    return env_from_snapshot(snap, cfg, label=(density, seed), interference_coupling_db=coupling)


def _full_heuristic_F(env, layer, cfg) -> tuple[float, str]:
    """Strongest of 5 heuristics on the SAME constructor+evaluator: nearest-k,
    best-channel-k, best-success-k, best-sinr-k, random-k. Returns (min F, name)."""
    f = env["features"]
    dist = f["distance_m"].to(dtype=torch.float64)
    ef = f["edge_features"].to(dtype=torch.float64)
    scores: dict[str, torch.Tensor] = {"nearest": -dist}
    if ef.shape[1] > 2:
        scores["channel"] = ef[:, 2]
    if ef.shape[1] > 3:
        scores["success"] = ef[:, 3]
    if ef.shape[1] > 4:
        scores["sinr"] = ef[:, 4]
    torch.manual_seed(0)
    scores["random"] = torch.stack([torch.randn(dist.shape[0], dtype=torch.float64) for _ in range(3)]).mean(0)
    best_f, best_name = None, ""
    for name, score in scores.items():
        with torch.no_grad():
            fv = float(evaluate(score.reshape(-1), env, layer, cfg)["F_avalanche_node_mean"].mean())
        if best_f is None or fv < best_f:
            best_f, best_name = fv, name
    return float(best_f), best_name


def _eval_full(model, env, layer, cfg) -> dict:
    with torch.no_grad():
        ev = evaluate(model, env, layer, cfg)
    m = metrics(ev)
    pernode = ev["F_avalanche_node_mean"].reshape(-1)
    frac = float(((pernode >= _FAIL_BAND[0]) & (pernode <= _FAIL_BAND[1])).to(torch.float64).mean())
    return {"learned_F": m["F"], "learned_C": m["C"], "D": m["D"], "E": m["E"], "fail_frac_band": frac}


def run_cell(
    base_config: dict,
    *,
    node_count: int,
    density: float,
    confidence: str,
    degree: int,
    coupling: float,
    steps: int,
    train_seed: int,
    gate: dict,
) -> dict:
    """Train at the operating point, evaluate on a held-out layout, score gap/ratio/gates."""
    cell = {
        "density": float(density),
        "confidence": str(confidence),
        "coupling_db": float(coupling),
        "degree": int(degree),
        "node_count": int(node_count),
        "steps": int(steps),
        "train_seed": int(train_seed),
        "eval_seed": int(train_seed) + _EVAL_SEED_OFFSET,
    }
    try:
        cfg = _cell_cfg(base_config, node_count=node_count, confidence=confidence, degree=degree, coupling=coupling)
        layer = build_topology_layer(cfg)
        train_env = _make_env(cfg, node_count, density, train_seed, coupling)
        eval_env = _make_env(cfg, node_count, density, train_seed + _EVAL_SEED_OFFSET, coupling)
        model = train_model(cfg, [train_env], layer, steps, model_seed=train_seed)
        full = _eval_full(model, eval_env, layer, cfg)
        heur_full, heur_name = _full_heuristic_F(eval_env, layer, cfg)
        heur_pair = float(best_heuristic_F(eval_env, layer, cfg))
        lf = full["learned_F"]
        cell.update(full)
        cell.update(
            {
                "best_heuristic_F": heur_full,
                "best_heuristic_name": heur_name,
                "best_heuristic_F_pair": heur_pair,
                "gap_G": cell_gap(lf, heur_full),
                "gap_G_pair": cell_gap(lf, heur_pair),
                "ratio": cell_ratio(lf, heur_full),
                "ratio_pair": cell_ratio(lf, heur_pair),
            }
        )
        cell["in_band"] = in_band(lf, gate["band_low"], gate["band_high"])
        cell["topology_decides"] = topology_decides(
            cell["gap_G"], cell["ratio"], lf,
            gap_min=gate["gap_min"], ratio_min=gate["ratio_min"], f_floor=gate["f_floor"],
        )
    except Exception as exc:  # keep an unattended grid robust; record and continue
        cell["error"] = f"{type(exc).__name__}: {exc}"
        cell["learned_F"] = None
        cell["in_band"] = False
        cell["topology_decides"] = False
    return cell


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description="Track A non-trivial operating-point search")
    p.add_argument("--config", default="configs/production_training_v1.yaml")
    p.add_argument("--node-count", type=int, default=1500, help="coarse-grid node count")
    p.add_argument("--confirm-node-count", type=int, default=2000)
    p.add_argument("--densities", default="100,150,200,250,300,350,400")
    p.add_argument("--confidences", default="toy,hard_low_confidence")
    p.add_argument("--couplings", default="0,10,20")
    p.add_argument("--degrees", default="3,4")
    p.add_argument("--coarse-steps", type=int, default=80)
    p.add_argument("--confirm-steps", type=int, default=140)
    p.add_argument("--coarse-seed", type=int, default=7)
    p.add_argument("--confirm-seeds", default="7,42,123")
    p.add_argument("--shortlist", type=int, default=3)
    p.add_argument("--band-low", type=float, default=1e-3)
    p.add_argument("--band-high", type=float, default=1e-1)
    p.add_argument("--gap-min", type=float, default=0.02)
    p.add_argument("--ratio-min", type=float, default=2.0)
    p.add_argument("--f-floor", type=float, default=1e-4)
    p.add_argument("--reliability-target", type=float, default=0.01)
    p.add_argument("--limit-cells", type=int, default=0, help="0=all; >0 truncates the coarse grid (smoke)")
    p.add_argument("--skip-confirm", action="store_true", help="coarse pass only (smoke)")
    p.add_argument("--run-name", default="operating_point_v1")
    args = p.parse_args()

    base_config = dict(load_training_smoke_config(str(ROOT / args.config)))
    densities = [float(x) for x in str(args.densities).split(",") if x.strip()]
    confidences = [x.strip() for x in str(args.confidences).split(",") if x.strip()]
    couplings = [float(x) for x in str(args.couplings).split(",") if x.strip()]
    degrees = [int(x) for x in str(args.degrees).split(",") if x.strip()]
    confirm_seeds = [int(x) for x in str(args.confirm_seeds).split(",") if x.strip()]
    gate = {
        "band_low": args.band_low, "band_high": args.band_high,
        "gap_min": args.gap_min, "ratio_min": args.ratio_min, "f_floor": args.f_floor,
    }

    grid = [(d, conf, k, g) for d in densities for conf in confidences for k in couplings for g in degrees]
    if args.limit_cells > 0:
        grid = grid[: args.limit_cells]

    out_dir = ROOT / "result" / args.run_name
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    coarse_path = out_dir / "_coarse_cells.jsonl"
    confirm_path = out_dir / "_confirm_cells.jsonl"
    done_coarse = {_cell_key(c): c for c in _load_jsonl(coarse_path)}
    done_confirm = {_cell_key(c): c for c in _load_jsonl(confirm_path)}
    print(
        f"Coarse grid: {len(grid)} cells (N={args.node_count}, {args.coarse_steps} steps, seed {args.coarse_seed}); "
        f"{len(done_coarse)} cached; gates band=[{args.band_low},{args.band_high}] "
        f"gap>={args.gap_min} ratio>={args.ratio_min}",
        flush=True,
    )

    coarse_cells: list[dict] = []
    for i, (d, conf, k, g) in enumerate(grid):
        key = (round(float(d), 6), conf, round(float(k), 6), int(g), int(args.node_count),
               int(args.coarse_steps), int(args.coarse_seed))
        cell = done_coarse.get(key)
        cached = cell is not None
        if not cached:
            cell = run_cell(
                base_config, node_count=args.node_count, density=d, confidence=conf, degree=g,
                coupling=k, steps=args.coarse_steps, train_seed=args.coarse_seed, gate=gate,
            )
            _append_jsonl(coarse_path, cell)
            gc.collect()
        coarse_cells.append(cell)
        tag = "cached" if cached else "run"
        if cell.get("error"):
            print(f"[{i+1}/{len(grid)}] ({tag}) d={d:.0f} conf={conf} k={k:.0f}dB deg={g}: ERROR {cell['error']}",
                  flush=True)
        else:
            print(
                f"[{i+1}/{len(grid)}] ({tag}) d={d:.0f} conf={conf} k={k:.0f}dB deg={g}: "
                f"F={cell['learned_F']:.4f} heur={cell['best_heuristic_F']:.4f}({cell['best_heuristic_name']}) "
                f"gap={cell['gap_G']:+.4f} ratio={cell['ratio']:.2f} band={cell['in_band']} "
                f"decides={cell['topology_decides']}",
                flush=True,
            )

    shortlist = select_shortlist(coarse_cells, args.shortlist, band_low=args.band_low, band_high=args.band_high)
    print(f"\nShortlist ({len(shortlist)} cells by gap_G within band):", flush=True)
    for c in shortlist:
        print(f"  d={c['density']:.0f} conf={c['confidence']} k={c['coupling_db']:.0f}dB deg={c['degree']} "
              f"gap={c['gap_G']:+.4f}", flush=True)

    confirmed: list[dict] = []
    if not args.skip_confirm:
        for sc in shortlist:
            per_seed = []
            for s in confirm_seeds:
                key = (round(float(sc["density"]), 6), sc["confidence"], round(float(sc["coupling_db"]), 6),
                       int(sc["degree"]), int(args.confirm_node_count), int(args.confirm_steps), int(s))
                cell = done_confirm.get(key)
                if cell is None:
                    cell = run_cell(
                        base_config, node_count=args.confirm_node_count, density=sc["density"],
                        confidence=sc["confidence"], degree=sc["degree"], coupling=sc["coupling_db"],
                        steps=args.confirm_steps, train_seed=s, gate=gate,
                    )
                    _append_jsonl(confirm_path, cell)
                    gc.collect()
                per_seed.append(cell)
                tag = cell.get("error") or (f"F={cell['learned_F']:.4f} gap={cell['gap_G']:+.4f} "
                                            f"ratio={cell['ratio']:.2f} decides={cell['topology_decides']}")
                print(f"  confirm d={sc['density']:.0f} conf={sc['confidence']} k={sc['coupling_db']:.0f}dB "
                      f"deg={sc['degree']} seed={s}: {tag}", flush=True)
            ok = [c for c in per_seed if "error" not in c]
            gaps = [c["gap_G"] for c in ok]
            ratios = [c["ratio"] for c in ok]
            lfs = [c["learned_F"] for c in ok]
            all_pass = bool(ok) and all(c["topology_decides"] and c["in_band"] for c in per_seed)
            confirmed.append({
                "density": sc["density"], "confidence": sc["confidence"],
                "coupling_db": sc["coupling_db"], "degree": sc["degree"],
                "confirm_node_count": args.confirm_node_count, "confirm_steps": args.confirm_steps,
                "seeds": confirm_seeds, "per_seed": per_seed,
                "gap_mean": statistics.fmean(gaps) if gaps else None,
                "gap_std": statistics.pstdev(gaps) if len(gaps) > 1 else 0.0,
                "ratio_mean": statistics.fmean(ratios) if ratios else None,
                "learned_F_mean": statistics.fmean(lfs) if lfs else None,
                "A_G2_A_G3_pass": all_pass,
            })

    passing = [c for c in confirmed if c["A_G2_A_G3_pass"]]
    recommended = max(passing, key=lambda c: c["gap_mean"]) if passing else None
    a_g1 = any(
        c.get("in_band") and isinstance(c.get("best_heuristic_F"), (int, float))
        and (c["best_heuristic_F"] - c["learned_F"]) > 1e-2
        for c in coarse_cells
    )
    verdict = make_verdict(a_g1, recommended)

    summary = {
        "config": args.config,
        "gates": {**gate, "shortlist": args.shortlist, "reliability_target": args.reliability_target},
        "coarse": {"node_count": args.node_count, "steps": args.coarse_steps, "seed": args.coarse_seed,
                   "densities": densities, "confidences": confidences, "couplings": couplings, "degrees": degrees},
        "coarse_cells": coarse_cells,
        "shortlist": [{k: sc[k] for k in ("density", "confidence", "coupling_db", "degree", "gap_G", "learned_F")}
                      for sc in shortlist],
        "confirmed": confirmed,
        "A_G1_band_exists": bool(a_g1),
        "recommended_operating_point": recommended,
        "verdict": verdict,
    }

    (out_dir / "operating_point_search.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    _render(coarse_cells, densities, confidences, couplings, degrees, recommended,
            out_dir / "figures" / "gap_heatmap.png")
    _write_result_md(summary, out_dir / "RESULT.md")

    print(f"\nA-G1 (band exists): {a_g1}")
    print(f"VERDICT: {verdict}")
    print(f"Wrote {out_dir}")


def _render(coarse_cells, densities, confidences, couplings, degrees, recommended, out_path: Path) -> None:
    lut = {(c["confidence"], c["degree"], c["density"], c["coupling_db"]): c.get("gap_G") for c in coarse_cells}
    nrows, ncols = max(len(confidences), 1), max(len(degrees), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    for ri, conf in enumerate(confidences):
        for ci, g in enumerate(degrees):
            ax = axes[ri][ci]
            grid = np.full((len(couplings), len(densities)), np.nan)
            for yi, k in enumerate(couplings):
                for xi, d in enumerate(densities):
                    v = lut.get((conf, g, d, k))
                    if isinstance(v, (int, float)):
                        grid[yi, xi] = v
            im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(densities)))
            ax.set_xticklabels([f"{int(d)}" for d in densities])
            ax.set_yticks(range(len(couplings)))
            ax.set_yticklabels([f"{int(k)}" for k in couplings])
            ax.set_title(f"conf={conf}, degree={g}", fontsize=10)
            ax.set_xlabel("density (veh/km^2)")
            ax.set_ylabel("coupling (dB)")
            for yi in range(len(couplings)):
                for xi in range(len(densities)):
                    if not np.isnan(grid[yi, xi]):
                        ax.text(xi, yi, f"{grid[yi, xi]:.3f}", ha="center", va="center",
                                color="white", fontsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if recommended and recommended["confidence"] == conf and recommended["degree"] == g:
                if recommended["density"] in densities and recommended["coupling_db"] in couplings:
                    xi = densities.index(recommended["density"])
                    yi = couplings.index(recommended["coupling_db"])
                    ax.add_patch(plt.Rectangle((xi - 0.5, yi - 0.5), 1, 1, fill=False, edgecolor="red", lw=2.5))
    fig.suptitle("Operating-point search: gap_G = best_heuristic_F - learned_F  "
                 "(higher = topology decides; red box = recommended)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _write_result_md(summary: dict, out_path: Path) -> None:
    cells = [c for c in summary["coarse_cells"] if isinstance(c.get("gap_G"), (int, float))]
    top = sorted(cells, key=lambda c: c["gap_G"], reverse=True)[:10]
    lines = [
        "# Track A — Non-Trivial Operating-Point Search: Result",
        "",
        "Design contract: `docs/COUPLING_AND_OPERATING_POINT_DESIGN.md`. Harness: "
        "`scripts/analysis/run_operating_point_search.py` (`make operating-point-search`).",
        "",
        f"## Verdict: {summary['verdict']}",
        "",
        "Gates: learnability band F in "
        f"[{summary['gates']['band_low']}, {summary['gates']['band_high']}] (A-i); "
        f"topology-decides gap_G >= {summary['gates']['gap_min']} AND ratio >= "
        f"{summary['gates']['ratio_min']} AND F >= {summary['gates']['f_floor']} (A-G2); "
        "multi-seed stable (A-G3). gap_G uses the strongest of 5 heuristics "
        "(nearest/channel/success/sinr/random).",
        "",
        f"- A-G1 (a learnable band with heuristics missing exists): **{summary['A_G1_band_exists']}**",
        "",
        "## Top coarse cells by gap_G",
        "",
        "| density | conf | coupling dB | degree | learned_F | best_heur_F (name) | gap_G | ratio | in band | decides |",
        "|---:|---|---:|---:|---:|---|---:|---:|:--:|:--:|",
    ]
    for c in top:
        lines.append(
            f"| {c['density']:.0f} | {c['confidence']} | {c['coupling_db']:.0f} | {c['degree']} | "
            f"{c['learned_F']:.4f} | {c['best_heuristic_F']:.4f} ({c['best_heuristic_name']}) | "
            f"{c['gap_G']:+.4f} | {c['ratio']:.2f} | {c['in_band']} | {c['topology_decides']} |"
        )
    rec = summary.get("recommended_operating_point")
    lines += ["", "## Recommended operating point (multi-seed confirmed)", ""]
    if rec:
        lines += [
            f"- density **{rec['density']:.0f}** veh/km^2, confidence **{rec['confidence']}**, "
            f"coupling **{rec['coupling_db']:.0f} dB**, active degree **{rec['degree']}**",
            f"- gap_G mean **{rec['gap_mean']:.4f}** (std {rec['gap_std']:.4f}), "
            f"ratio mean **{rec['ratio_mean']:.2f}**, learned_F mean **{rec['learned_F_mean']:.4f}** "
            f"over seeds {rec['seeds']}",
            "",
            "Freeze this into `configs/operating_point_v1.yaml` (vehicle density, "
            "`training_profile`, `physical.interference_density_coupling_db`, `max_out_degree`) "
            "and use it as the Track-B substrate.",
        ]
    else:
        lines += ["- none (no shortlisted cell passed A-G2 & A-G3 across all seeds)."]
    lines += ["", "Artifacts: `operating_point_search.json`, `figures/gap_heatmap.png`.", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
