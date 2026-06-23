"""Seed-variance decomposition (P0-2): separate optimization/init variance from scene variance.

The S2 ablation reports +/- std over a single --seed that drives BOTH the scene layout and the model
init, so its error bars confound "is this scene hard" with "did this init converge". This harness
runs the FULL product arm x scene_seed x init_seed (reusing run_s2_temporal_ablation.py so the numbers
match result/s2_ablation*) and decomposes the held-out F variance:

  sigma_init[arm]  = mean over scenes of std(F over the init axis at fixed scene)
  sigma_scene[arm] = mean over inits  of std(F over the scene axis at fixed init)
  deltaF(a,b)      = mean over (scene,init) of (F[a] - F[b])   (paired cross-arm contrast)

This is the quantitative answer to "is the model unstable, or is the scene varying" (audit S3.2). The
protocol rule it enforces: if a reported arm ordering's |deltaF| < 2*sigma_init/sqrt(n) it is NOT
resolvable -> the report must say "indistinguishable", not give a ranking.

Usage:
    python scripts/analysis/run_seed_variance_decomposition.py --scene-seeds 7,42,123 --init-seeds 7,42,123 \
        --arms static,no_memory,full --node-count 400 --coupling-db 20 --epochs 12
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "scripts" / "analysis" / "run_s2_temporal_ablation.py"


def _run_one(arm: str, scene_seed: int, init_seed: int, extra: list[str]) -> dict | None:
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    cmd = [sys.executable, str(HARNESS), "--arm", arm, "--seed", str(scene_seed),
           "--scene-seed", str(scene_seed), "--init-seed", str(init_seed), *extra]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT arm={arm} scene={scene_seed} init={init_seed}", flush=True)
        return None
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON ")), None)
    if line is None:
        print(f"  FAIL arm={arm} scene={scene_seed} init={init_seed}: {proc.stderr.strip()[-200:]}", flush=True)
        return None
    rec = json.loads(line.split("RESULT_JSON ", 1)[1])
    rec["scene_seed"], rec["init_seed"] = int(scene_seed), int(init_seed)
    print(f"  done arm={arm:9s} scene={scene_seed} init={init_seed}  F={rec['F']:.4f}", flush=True)
    return rec


def _decompose(records: list[dict], arm: str, scene_seeds: list[int], init_seeds: list[int]) -> dict:
    grid = {(r["scene_seed"], r["init_seed"]): r["F"] for r in records if r["arm"] == arm}
    F = np.full((len(scene_seeds), len(init_seeds)), np.nan)
    for i, sc in enumerate(scene_seeds):
        for j, ic in enumerate(init_seeds):
            if (sc, ic) in grid:
                F[i, j] = grid[(sc, ic)]
    # sigma_init: std across the init axis (axis=1) at fixed scene, averaged over scenes.
    sigma_init = float(np.nanmean(np.nanstd(F, axis=1, ddof=1))) if F.shape[1] > 1 else 0.0
    # sigma_scene: std across the scene axis (axis=0) at fixed init, averaged over inits.
    sigma_scene = float(np.nanmean(np.nanstd(F, axis=0, ddof=1))) if F.shape[0] > 1 else 0.0
    finite = F[np.isfinite(F)]
    return {
        "F_mean": float(np.nanmean(F)) if finite.size else float("nan"),
        "F_min": float(np.nanmin(F)) if finite.size else float("nan"),
        "F_max": float(np.nanmax(F)) if finite.size else float("nan"),
        "sigma_init": sigma_init,
        "sigma_scene": sigma_scene,
        "init_range": float(np.nanmax(F) - np.nanmin(F)) if finite.size else float("nan"),
        "n": int(finite.size),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Seed-variance decomposition (sigma_init vs sigma_scene)")
    p.add_argument("--arms", default="static,no_memory,full")
    p.add_argument("--scene-seeds", default="7,42,123")
    p.add_argument("--init-seeds", default="7,42,123")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--run-name", default="seed_variance_decomposition_v1")
    p.add_argument("--node-count", type=int, default=400)
    p.add_argument("--num-frames", type=int, default=12)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--coupling-db", type=float, default=20.0)
    # P2-1: re-test on the REALISTIC substrate — turn on the hidden AR(1) shadow state so the temporal
    # question is live (vs the fully-observable default where memory has no theoretical edge).
    p.add_argument("--shadow-std-db", type=float, default=0.0)
    p.add_argument("--shadow-decorr-s", type=float, default=8.0)
    # Phase 1.3 fine-stage cell selection + churn substrate (forwarded to the S2 harness).
    p.add_argument("--churn-rate", type=float, default=0.0)
    p.add_argument("--density", type=float, default=None)
    p.add_argument("--training-profile", default=None)
    p.add_argument("--heuristic-F", type=float, default=None,
                   help="optional best_success_k held-out F for the sensitivity context line")
    args = p.parse_args()
    arms = [a for a in args.arms.split(",") if a.strip()]
    scene_seeds = [int(s) for s in args.scene_seeds.split(",") if s.strip()]
    init_seeds = [int(s) for s in args.init_seeds.split(",") if s.strip()]
    extra = ["--node-count", str(args.node_count), "--num-frames", str(args.num_frames),
             "--epochs", str(args.epochs), "--coupling-db", str(args.coupling_db),
             "--shadow-std-db", str(args.shadow_std_db), "--shadow-decorr-s", str(args.shadow_decorr_s),
             "--churn-rate", str(args.churn_rate)]
    if args.density is not None:
        extra += ["--density", str(args.density)]
    if args.training_profile is not None:
        extra += ["--training-profile", str(args.training_profile)]
    jobs = list(itertools.product(arms, scene_seeds, init_seeds))
    print(f"Seed-variance decomposition: {len(jobs)} runs "
          f"({len(arms)} arms x {len(scene_seeds)} scenes x {len(init_seeds)} inits), "
          f"N={args.node_count}, {args.coupling_db}dB, {args.epochs}ep", flush=True)

    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = [ex.submit(_run_one, a, sc, ic, extra) for (a, sc, ic) in jobs]
        for f in futs:
            r = f.result()
            if r is not None:
                records.append(r)

    decomposition = {arm: _decompose(records, arm, scene_seeds, init_seeds) for arm in arms}
    summary = {
        "arms": arms, "scene_seeds": scene_seeds, "init_seeds": init_seeds,
        "node_count": args.node_count, "num_frames": args.num_frames, "epochs": args.epochs,
        "coupling_db": args.coupling_db, "evaluator_currency": "quenched (train Q=11 / eval Q=21)",
        "shadow_std_db": args.shadow_std_db, "shadow_decorr_s": args.shadow_decorr_s,
        "heuristic_best_success_k_F": args.heuristic_F,
        "decomposition": decomposition, "raw": records,
    }
    out_dir = ROOT / "result" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== Seed-variance decomposition (quenched currency; lower F = better) ===")
    print(f"{'arm':10s} {'F_mean':>8s} {'sigma_init':>11s} {'sigma_scene':>12s} {'init_range':>11s} {'n':>4s}")
    for arm in arms:
        d = decomposition[arm]
        print(f"{arm:10s} {d['F_mean']:>8.4f} {d['sigma_init']:>11.4f} {d['sigma_scene']:>12.4f} "
              f"{d['init_range']:>11.4f} {d['n']:>4d}")
    # protocol verdict: PAIRED test over the shared (scene, init) grid — the same scene, init and
    # shadow realisation back both arms of every pair, so the paired design is the correct (and
    # most powerful) test. (Audit defect fix: the previous 2*sigma_init/sqrt(n) heuristic produced
    # a false positive when sigma_init was tiny — e.g. static-vs-full at 10 dB, paired p=0.29.)
    from scipy import stats as _stats  # local import keeps module import light

    pair_keys = sorted(set(itertools.product(scene_seeds, init_seeds)))
    grid = {(r["arm"], r["scene_seed"], r["init_seed"]): r["F"] for r in records}
    print("\n--- resolvability (paired t-test over the shared scene x init grid; claim an ordering only if p < 0.05) ---")
    paired_out = {}
    for a, b in itertools.combinations(arms, 2):
        diffs = np.array([
            grid[(a, s, i)] - grid[(b, s, i)]
            for (s, i) in pair_keys
            if (a, s, i) in grid and (b, s, i) in grid
        ])
        if diffs.size < 2:
            print(f"  {a:>10s} vs {b:<10s}  insufficient pairs ({diffs.size})")
            continue
        t_stat, p_t = _stats.ttest_1samp(diffs, 0.0)
        try:
            _, p_w = _stats.wilcoxon(diffs)
        except ValueError:
            p_w = float("nan")
        verdict = "RESOLVABLE" if p_t < 0.05 else "INDISTINGUISHABLE"
        paired_out[f"{a}_vs_{b}"] = {
            "mean_dF": float(diffs.mean()), "sd_dF": float(diffs.std(ddof=1)),
            "t": float(t_stat), "p_paired_t": float(p_t), "p_wilcoxon": float(p_w),
            "n_pairs": int(diffs.size), "verdict": verdict,
        }
        print(f"  {a:>10s} vs {b:<10s}  mean dF={diffs.mean():+.4f}  paired-t p={p_t:.4f}  "
              f"wilcoxon p={p_w:.3f}  n={diffs.size}  -> {verdict}")
    summary["paired_tests"] = paired_out
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    if args.heuristic_F is not None:
        print(f"\n  (context) best_success_k held-out F = {args.heuristic_F:.4f}; "
              f"arms at ~this level have no training headroom to resolve architecture)")
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
