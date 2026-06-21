"""Unified paper-figure generation (IEEE two-column style) from existing result/*.json.

Renders the replot/upgrade figures of docs/PAPER_FIGURE_PLAN.md into result/paper_figures/
(both .png at 300 dpi and vector .pdf). Every figure reads data already on disk — no training here.
New-experiment figures (F4.1, F4.5, F4.2, F4.3) live in their own runnable scripts.

Usage:  python -B scripts/analysis/make_paper_figures.py [--only F1.10,F2.2] [--out result/paper_figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# --------------------------------------------------------------------------- IEEE style
# Wong colour-blind-safe palette.
C_GNN = "#0072B2"      # blue  – learned / primary
C_HEUR = "#D55E00"     # vermillion – heuristic / baseline
C_GOOD = "#009E73"     # green – good / emission-on / optimized
C_BAD = "#7f0000"      # dark red – degenerate / collapse
C_ACC = "#CC79A7"      # purple – accent
C_ORANGE = "#E69F00"   # orange – third series
C_GREY = "#666666"

COL_W = 3.5    # IEEE single-column width (in)
DBL_W = 7.16   # IEEE double-column width (in)


def set_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "axes.grid": True,
        "axes.axisbelow": True,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _load(rel: str):
    return json.loads((RESULT / rel).read_text(encoding="utf-8"))


def save(fig, out_dir: Path, fid: str, title: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{fid}_{title}.{ext}")
    plt.close(fig)
    print(f"  wrote {fid}_{title}.png/.pdf", flush=True)


# =========================================================================== §0
def f0_2_floor(out: Path) -> None:
    """Protocol-floor reference: floor vs degree budget {4,8} x protocol variant."""
    ft = _load("protocol_floor_table/floor_table.json")
    rows = {r["degree"]: r["floors"] for r in ft["rows"]}
    variants = list(next(iter(rows.values())).keys())
    short = [v.split(" (")[0] for v in variants]
    degs = sorted(rows.keys())
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    x = np.arange(len(variants)); w = 0.38
    for i, d in enumerate(degs):
        vals = [rows[d][v] for v in variants]
        ax.bar(x + (i - 0.5) * w, vals, w, label=f"degree budget {d}",
               color=(C_GNN if d == degs[0] else C_GOOD), alpha=0.85)
    ax.axhline(0.01, ls="--", lw=0.8, color=C_BAD)
    ax.text(len(variants) - 1.0, 0.012, "F = 0.01 target", color=C_BAD, fontsize=6, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels(short, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("perfect-link failure floor")
    ax.set_title("Protocol floor (feasibility reference)")
    ax.legend(loc="upper left")
    save(fig, out, "F0.2", "protocol_floor")


# =========================================================================== §1.1
def f1_3_trajectory(out: Path) -> None:
    """How the operating point moves during training: F-D path, coloured by step."""
    tr = _load("production_report_operating_point/report.json")["trajectory"]
    F = np.array([t["F"] for t in tr]); D = np.array([t["D"] for t in tr]); step = np.array([t["step"] for t in tr])
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ax.plot(F, D, "-", color=C_GREY, lw=0.7, alpha=0.7, zorder=1)
    sc = ax.scatter(F, D, c=step, cmap="viridis", s=14, zorder=2, edgecolors="none")
    ax.scatter([F[0]], [D[0]], marker="o", s=55, facecolors="none", edgecolors=C_BAD, lw=1.3, zorder=3)
    ax.scatter([F[-1]], [D[-1]], marker="*", s=110, color=C_GOOD, zorder=3)
    ax.annotate("init", (F[0], D[0]), textcoords="offset points", xytext=(6, 2), fontsize=6, color=C_BAD)
    ax.annotate("converged", (F[-1], D[-1]), textcoords="offset points", xytext=(6, -2), fontsize=6, color=C_GOOD)
    # mark the early reorganisation (first 5 steps)
    ax.annotate("first ~5 steps\n(violent reorg.)", (F[3], D[3]), textcoords="offset points",
                xytext=(18, 18), fontsize=6, color=C_GREY,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_GREY))
    ax.set_yscale("log"); ax.set_xlabel("failure F"); ax.set_ylabel("delay D (eff-rounds, log)")
    ax.set_title("Operating-point trajectory during training")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03); cb.set_label("training step", fontsize=7)
    save(fig, out, "F1.3", "pareto_trajectory")


def f1_4_two_config(out: Path) -> None:
    """paper_env Pareto FLAT vs operating_point LIVE — justifies the two-config design."""
    pe = _load("production_report_paperenv/report.json")["pareto"]["rows"]
    op = _load("production_report_operating_point/report.json")["pareto"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6))
    for ax, rows, tag, live in ((axes[0], pe, "paper_environment_v1", False),
                                (axes[1], op, "operating_point_v1", True)):
        ws = [r["w_cost"] for r in rows]
        F = [r["holdout_F"]["mean"] for r in rows]; D = [r["holdout_D"]["mean"] for r in rows]
        sc = ax.scatter(F, D, c=ws, cmap="viridis", s=34, zorder=3, edgecolors="k", linewidths=0.3)
        for r in rows:
            ax.annotate(f"w={r['w_cost']:g}", (r["holdout_F"]["mean"], r["holdout_D"]["mean"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=5.5)
        ax.set_xlabel("failure F"); ax.set_title(f"{tag}\n{'live front' if live else 'flat (no D/E levers)'}")
        if live:
            ax.set_yscale("log")
        ax.set_ylabel("delay D" + (" (log)" if live else ""))
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03).set_label("cost weight w", fontsize=6)
    fig.suptitle("Two-config Pareto contrast", y=1.02, fontsize=9, fontweight="bold")
    save(fig, out, "F1.4", "two_config_pareto")


def f1_5_w0_dominated(out: Path) -> None:
    """rel-only (w=0) is strictly dominated: cost ~36x AND worse F."""
    rows = _load("production_report_operating_point/report.json")["pareto"]["rows"]
    ws = [r["w_cost"] for r in rows]
    F = [r["holdout_F"]["mean"] for r in rows]; D = [r["holdout_D"]["mean"] for r in rows]
    E = [r["holdout_E"]["mean"] for r in rows]
    i0 = ws.index(0.0)
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.6))
    # left: F-D, w=0 marked dominated; w>0 cluster labelled once
    ax = axes[0]
    for x, y, w in zip(F, D, ws):
        ax.scatter(x, y, color=(C_BAD if w == 0 else C_GNN), s=(48 if w == 0 else 30), zorder=3)
    Fpos = [f for f, w in zip(F, ws) if w > 0]; Dpos = [d for d, w in zip(D, ws) if w > 0]
    ax.annotate("w > 0\n(optimized regime)", (np.mean(Fpos), max(Dpos)),
                textcoords="offset points", xytext=(34, 6), fontsize=6, color=C_GNN, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_GNN))
    ax.set_yscale("log"); ax.set_xlabel("failure F"); ax.set_ylabel("delay D (eff-rounds, log)")
    ax.annotate("w=0 rel-only\n(dominated: worse F\n& ~36x cost)", (F[i0], D[i0]),
                textcoords="offset points", xytext=(-12, -38), fontsize=6, color=C_BAD,
                ha="center", arrowprops=dict(arrowstyle="->", lw=0.6, color=C_BAD))
    ax.set_xlim(min(F) - 0.03, max(F) + 0.05)
    ax.set_title("w=0 is off the achievable front")
    # right: cost blow-up bars (log)
    ax = axes[1]
    xb = np.arange(len(ws)); wbar = 0.38
    ax.bar(xb - wbar / 2, D, wbar, color=C_GNN, alpha=0.85, label="delay D")
    ax.bar(xb + wbar / 2, np.array(E) * 1e3, wbar, color=C_GOOD, alpha=0.85, label="energy E (mJ)")
    ax.set_yscale("log"); ax.set_xticks(xb); ax.set_xticklabels([f"{w:g}" for w in ws])
    ax.set_xlabel("cost weight w"); ax.set_ylabel("D / E (log)")
    ax.set_title("Cost blows up ~36x at w=0"); ax.legend()
    save(fig, out, "F1.5", "w0_dominated")


# =========================================================================== §1.2
def f1_9_transfer_regret(out: Path) -> None:
    """N-randomized planner vs from-scratch N=10000 expert; regret +0.0062."""
    st = _load("scale_transfer/scale_transfer.json")
    per = st["per_seed"]
    Fe = [p["F_expert"] for p in per]; Fp = [p["F_planner"] for p in per]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    ax.bar(0, np.mean(Fe), 0.5, yerr=np.std(Fe), color=C_GOOD, alpha=0.85, capsize=4, label="from-scratch N=10000 expert")
    ax.bar(1, np.mean(Fp), 0.5, yerr=np.std(Fp), color=C_GNN, alpha=0.85, capsize=4, label="N-randomized planner")
    ax.scatter([0] * len(Fe), Fe, color="k", s=12, zorder=5)
    ax.scatter([1] * len(Fp), Fp, color="k", s=12, zorder=5)
    regret = np.mean(Fp) - np.mean(Fe)
    ax.annotate(f"transfer regret\n+{regret:.4f}", (0.5, max(np.mean(Fe), np.mean(Fp))),
                textcoords="offset points", xytext=(0, 14), ha="center", fontsize=6.5, color=C_HEUR)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["expert", "planner"])
    ax.set_ylabel("failure F  (eval Q=21)")
    ax.set_title("Scale transfer at N=10000"); ax.legend(loc="lower center", fontsize=6)
    save(fig, out, "F1.9", "transfer_regret")


# =========================================================================== §1.3
_PROFILE_ORDER = ["toy", "near_target_synthetic", "hard_low_confidence"]
_PROFILE_LABEL = {"toy": "toy (high conf.)", "near_target_synthetic": "near-target", "hard_low_confidence": "hard (low conf.)"}


def f1_10_advantage_heatmap(out: Path) -> None:
    """27-cell advantage map: density x coupling, faceted by profile, colour=gap, robust outlined."""
    cells = _load("advantage_map/advantage_map.json")["cells"]
    densities = sorted({c["density"] for c in cells})
    couplings = sorted({c["coupling_db"] for c in cells})
    profiles = [p for p in _PROFILE_ORDER if any(c["profile"] == p for c in cells)]
    vmax = max(abs(c.get("gap_mean", c["best_heuristic_mean"] - c["F_gnn_mean"])) for c in cells)
    fig, axes = plt.subplots(1, len(profiles), figsize=(DBL_W, 2.5), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    im = None
    for ax, pr in zip(axes, profiles):
        M = np.full((len(couplings), len(densities)), np.nan)
        for c in cells:
            if c["profile"] != pr:
                continue
            i = couplings.index(c["coupling_db"]); j = densities.index(c["density"])
            M[i, j] = c.get("gap_mean", c["best_heuristic_mean"] - c["F_gnn_mean"])
            if c.get("label_robust"):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="k", lw=1.6, zorder=4))
        im = ax.imshow(M, origin="lower", cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(len(couplings)):
            for j in range(len(densities)):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.02f}", ha="center", va="center", fontsize=5.5,
                            color="k")
        ax.set_xticks(range(len(densities))); ax.set_xticklabels([int(d) for d in densities])
        ax.set_yticks(range(len(couplings))); ax.set_yticklabels([int(k) for k in couplings])
        ax.set_xlabel("density (veh/km$^2$)"); ax.set_title(_PROFILE_LABEL.get(pr, pr), fontsize=7)
        ax.grid(False)
    axes[0].set_ylabel("interference coupling (dB)")
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label("GNN advantage gap (bestH − F$_{GNN}$)", fontsize=6.5)
    fig.suptitle("Where learning wins (black outline = seed-robust)", y=1.04, fontsize=9, fontweight="bold")
    save(fig, out, "F1.10", "advantage_heatmap")


def f1_12_mc_survival(out: Path) -> None:
    """Does the advantage gap survive Monte-Carlo ground truth? slope gap_q -> gap_mc."""
    cells = _load("advantage_montecarlo/advantage_mc.json")["cells"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    for c in cells:
        gq, gm = c["gap_quenched"], c["gap_mc"]
        lab = f"d{int(c['density'])}/{c['profile'][:4]}/c{int(c['coupling_db'])}"
        col = C_GOOD if gm > 0 else C_BAD
        ax.plot([0, 1], [gq, gm], "-o", color=col, lw=1.0, ms=4)
        ax.annotate(lab, (1, gm), textcoords="offset points", xytext=(5, 0), fontsize=5.5, va="center")
    ax.axhline(0, ls="--", lw=0.7, color=C_GREY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["quenched\nsurrogate", "Monte-Carlo\nground truth"])
    ax.set_ylabel("GNN advantage gap"); ax.set_xlim(-0.2, 1.6)
    ax.set_title("Advantage survives MC (all gaps > 0)")
    save(fig, out, "F1.12", "mc_survival")


# =========================================================================== §2
_EMIT = {"full": ("pure memory (−emission)", C_BAD), "no_graph": ("−emission, −graph", C_HEUR),
         "filter": ("+emission", C_GOOD), "no_graph_filter": ("+emission, −graph", C_GNN),
         "filter_nomem": ("+emission, −memory", C_ACC)}
_ARM_ORDER = ["full", "no_graph", "filter", "no_graph_filter", "filter_nomem"]


def f2_2_sigma_init(out: Path) -> None:
    """sigma_init per arm: ~30-39x drop iff emission on."""
    dec = _load("emission_2x2_isolation/summary.json")["decomposition"]
    arms = [a for a in _ARM_ORDER if a in dec]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    vals = [dec[a]["sigma_init"] for a in arms]
    cols = [_EMIT[a][1] for a in arms]
    ax.bar(range(len(arms)), vals, color=cols, alpha=0.88)
    ax.set_yscale("log")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([_EMIT[a][0] for a in arms], rotation=22, ha="right", fontsize=6)
    ax.set_ylabel("$\\sigma_{init}$ (init-seed std of F)")
    ratio = dec["full"]["sigma_init"] / dec["filter"]["sigma_init"]
    ax.annotate(f"{ratio:.0f}× drop\nwith emission", (0.5, dec["filter"]["sigma_init"]),
                textcoords="offset points", xytext=(40, 30), fontsize=6.5, color=C_GOOD,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_GOOD))
    ax.set_title("Emission collapses init-variance")
    save(fig, out, "F2.2", "sigma_init")


def f2_3_probe(out: Path) -> None:
    """Mechanism probe: gate-input scale + gate gradient, +emission vs -emission, vs epoch."""
    ep = _load("emission_probe_collapse_regime/emission_probe.json")
    # average trajectories per arm over init seeds
    def avg(arm, key):
        runs = [r for r in ep["runs"] if r["arm"] == arm]
        T = min(len(r["trajectory"]) for r in runs)
        return np.array([np.mean([r["trajectory"][t][key] for r in runs]) for t in range(T)])
    arms = ep["arms"]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.5))
    for arm in arms:
        lab, col = ("+emission", C_GOOD) if arm == "filter" else ("−emission (pure memory)", C_BAD)
        e = np.arange(len(avg(arm, "joined_std")))
        axes[0].plot(e, avg(arm, "joined_std"), "-o", color=col, ms=3, label=lab)
        axes[1].plot(e, avg(arm, "gate_grad_norm"), "-o", color=col, ms=3, label=lab)
    axes[0].set_ylabel("recurrent gate-input scale (std)"); axes[0].set_xlabel("epoch")
    axes[0].set_title("Input distribution"); axes[0].legend(fontsize=6)
    axes[1].set_yscale("log"); axes[1].set_ylabel("gate-weight grad norm (log)"); axes[1].set_xlabel("epoch")
    axes[1].set_title("Gradient flow"); axes[1].legend(fontsize=6)
    fig.suptitle("Why pure memory collapses (mechanism probe)", y=1.03, fontsize=9, fontweight="bold")
    save(fig, out, "F2.3", "probe_divergence")


def f2_4_outcome_range(out: Path) -> None:
    """F outcome range per arm: -emission spans up to ~0.96 (collapse), +emission tight near 0.55."""
    dec = _load("emission_2x2_isolation/summary.json")["decomposition"]
    arms = [a for a in _ARM_ORDER if a in dec]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    for i, a in enumerate(arms):
        d = dec[a]; col = _EMIT[a][1]
        ax.plot([i, i], [d["F_min"], d["F_max"]], "-", color=col, lw=2.2, alpha=0.55)
        ax.scatter([i], [d["F_mean"]], color=col, s=26, zorder=4)
    ax.axhline(0.95, ls=":", lw=0.7, color=C_GREY); ax.text(len(arms) - 1, 0.955, "collapsed init", fontsize=6, ha="right", color=C_GREY)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([_EMIT[a][0] for a in arms], rotation=22, ha="right", fontsize=6)
    ax.set_ylabel("failure F  (min–mean–max over inits)")
    ax.set_title("Collapse vs escape across inits")
    save(fig, out, "F2.4", "outcome_range")


def f2_5_replication(out: Path) -> None:
    """Replication: sigma_init reduction (full/filter) across cells — severe at sparse+coupled, absent dense."""
    cands = {
        "d100 hard c20": "fine_stage_d100_hard_c20/summary.json",
        "d100 toy c20": "fine_stage_d100_toy_c20/summary.json",
        "d200 hard c20": "fine_stage_d200_hard_c20/summary.json",
        "d100 hard c0": "emission_replication_d100_c0/summary.json",
    }
    labels, ratios = [], []
    for lab, rel in cands.items():
        p = RESULT / rel
        if not p.exists():
            continue
        dec = json.loads(p.read_text(encoding="utf-8")).get("decomposition", {})
        if "full" in dec and "filter" in dec and dec["filter"]["sigma_init"] > 0:
            labels.append(lab); ratios.append(dec["full"]["sigma_init"] / dec["filter"]["sigma_init"])
    if not labels:
        print("  F2.5 skipped (no fine_stage decomposition found)"); return
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    cols = [C_BAD if r > 3 else C_GREY for r in ratios]
    ax.bar(range(len(labels)), ratios, color=cols, alpha=0.85)
    ax.axhline(1.0, ls="--", lw=0.7, color=C_GREY)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=6)
    ax.set_ylabel("$\\sigma_{init}$ reduction (−emission / +emission)")
    ax.set_title("Stabilization is regime-specific")
    save(fig, out, "F2.5", "replication")


# =========================================================================== §3.1
_DEC_COL = {"USE_GNN": C_GOOD, "USE_GNN_MARGINAL": C_GNN, "HEURISTIC_OK": C_ORANGE, "GNN_DEFAULT": C_GREY}
_DEC_LAB = {"USE_GNN": "use GNN\n(robust)", "USE_GNN_MARGINAL": "use GNN\n(marginal)",
            "HEURISTIC_OK": "heuristic\nsuffices", "GNN_DEFAULT": "GNN\ndefault"}


def f3_1_gating_map(out: Path) -> None:
    """Gating decision surface: density x coupling -> policy."""
    tab = _load("gating_demo/gating_table.json")
    dens = tab["densities"]; coup = tab["couplings"]
    order = list(_DEC_COL.keys())
    M = np.full((len(coup), len(dens)), -1)
    for e in tab["entries"]:
        i = coup.index(e["coupling_db"]); j = dens.index(e["density"])
        M[i, j] = order.index(e["decision"]) if e["decision"] in order else -1
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([_DEC_COL[o] for o in order])
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.imshow(M, origin="lower", cmap=cmap, vmin=0, vmax=len(order) - 1, aspect="auto")
    for e in tab["entries"]:
        i = coup.index(e["coupling_db"]); j = dens.index(e["density"])
        ax.text(j, i, _DEC_LAB.get(e["decision"], e["decision"]), ha="center", va="center", fontsize=5.5, color="k")
    ax.set_xticks(range(len(dens))); ax.set_xticklabels([int(d) for d in dens])
    ax.set_yticks(range(len(coup))); ax.set_yticklabels([int(k) for k in coup])
    ax.set_xlabel("density (veh/km$^2$)"); ax.set_ylabel("interference coupling (dB)")
    ax.set_title("Deployment gating decision map"); ax.grid(False)
    save(fig, out, "F3.1", "gating_map")


def f3_2_routing(out: Path) -> None:
    """Forward-only routing: realized heuristic F vs predicted band, per frame."""
    frames = _load("gating_demo/gating_demo_log.json")["frames"]
    fig, ax = plt.subplots(figsize=(DBL_W, 2.4))
    xs = np.arange(len(frames))
    lo = [f["expected_heuristic_F"][0] for f in frames]; hi = [f["expected_heuristic_F"][1] for f in frames]
    real = [f["realized_heuristic_F"] for f in frames]
    ax.fill_between(xs, lo, hi, color=C_GNN, alpha=0.2, label="predicted band")
    inb = [f["heuristic_F_in_band"] for f in frames]
    ax.scatter(xs, real, c=[C_GOOD if b else C_BAD for b in inb], s=20, zorder=4, label="realized heuristic F")
    # scenario separators
    scen = [f"d{int(f['scenario']['density'])}/c{int(f['scenario']['coupling_db'])}" for f in frames]
    seen = None
    for i, s in enumerate(scen):
        if s != seen:
            ax.axvline(i - 0.5, ls=":", lw=0.5, color=C_GREY)
            ax.text(i, ax.get_ylim()[1], s, fontsize=5.5, color=C_GREY, va="top")
            seen = s
    n_in = sum(inb)
    ax.set_xlabel("frame"); ax.set_ylabel("heuristic F")
    ax.set_title(f"Forward-only routing validation ({n_in}/{len(frames)} in band)")
    ax.legend(loc="upper right", fontsize=6)
    save(fig, out, "F3.2", "routing_validation")


def f3_3_context(out: Path) -> None:
    """Context estimation accuracy: estimated vs true density per frame."""
    frames = _load("gating_demo/gating_demo_log.json")["frames"]
    true = np.array([f["scenario"]["density"] for f in frames])
    est = np.array([f["estimated_density"] for f in frames])
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    lim = [min(true.min(), est.min()) * 0.9, max(true.max(), est.max()) * 1.05]
    ax.plot(lim, lim, "--", color=C_GREY, lw=0.7)
    ax.scatter(true, est, color=C_GNN, s=20, alpha=0.8)
    ax.set_xlabel("true density (veh/km$^2$)"); ax.set_ylabel("estimated density")
    ax.set_title("Deploy-time context estimation")
    save(fig, out, "F3.3", "context_estimation")


# =========================================================================== §3.2
def f3_4_retention(out: Path) -> None:
    """Retention by density: governed vs naive vs LOCO ceiling."""
    comp = _load("mixture_governed_gradnorm/envelope_governed.json")["comparison_by_density"]
    dens = sorted(comp.keys())
    series = [("naive", "naive", C_HEUR), ("governed", "governed", C_GNN), ("loco_ceiling", "LOCO ceiling", C_GOOD)]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.arange(len(dens)); w = 0.26
    for k, (key, lab, col) in enumerate(series):
        vals = [comp[d].get(key) or np.nan for d in dens]
        ax.bar(x + (k - 1) * w, vals, w, label=lab, color=col, alpha=0.88)
    ax.axhline(1.0, ls="--", lw=0.7, color=C_GREY)
    ax.set_xticks(x); ax.set_xticklabels([d.replace("d", "density ") for d in dens])
    ax.set_ylabel("in-grid retention"); ax.set_ylim(0, 1.15)
    ax.set_title("Governed mixture matches the ceiling"); ax.legend(fontsize=6)
    save(fig, out, "F3.4", "retention")


def f3_5_diagnostic(out: Path) -> None:
    """Gradient-conflict diagnostic: magnitude imbalance (norms) + positive cosines."""
    diag = _load("mixture_governed_gradnorm/envelope_governed.json")["diagnostic"]
    norms = diag["group_grad_norms"]; cos = diag["pairwise_cosines"]
    densk = sorted(norms.keys(), key=lambda x: float(x))
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.4))
    ax = axes[0]
    ax.bar(range(len(densk)), [norms[k] for k in densk], color=C_GNN, alpha=0.85)
    ax.set_xticks(range(len(densk))); ax.set_xticklabels([f"d{int(float(k))}" for k in densk])
    ax.set_ylabel("per-density gradient norm"); ax.set_title("Magnitude imbalance → GradNorm")
    lo = min(norms.values()); hi = max(norms.values())
    ax.annotate(f"{hi/lo:.1f}× imbalance", (0.5, hi), textcoords="offset points", xytext=(0, -2), fontsize=6.5, ha="center")
    ax = axes[1]
    pairs = list(cos.keys()); vals = [cos[p] for p in pairs]
    ax.bar(range(len(pairs)), vals, color=[C_GOOD if v > 0 else C_BAD for v in vals], alpha=0.85)
    ax.axhline(0, lw=0.7, color="k")
    ax.set_xticks(range(len(pairs))); ax.set_xticklabels([p.replace("|", "·\n") for p in pairs], fontsize=6)
    ax.set_ylabel("pairwise gradient cosine"); ax.set_ylim(-1, 1)
    ax.set_title("All cosines > 0 → no PCGrad need")
    fig.suptitle("Measure the conflict, then choose the tool", y=1.03, fontsize=9, fontweight="bold")
    save(fig, out, "F3.5", "gradient_diagnostic")


def f3_6_offgrid(out: Path) -> None:
    """Off-grid interpolation: gap_model at interpolation cells."""
    og = _load("mixture_governed_gradnorm/envelope_governed.json")["off_grid"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    dens = sorted({c["density"] for c in og})
    for d in dens:
        sub = [c for c in og if c["density"] == d]
        ax.scatter([d] * len(sub), [c["gap_model"] for c in sub], s=22, alpha=0.8,
                   color=C_GNN, label=f"d{int(d)}" if d == dens[0] else None)
    ax.axhline(0, ls="--", lw=0.7, color=C_GREY)
    n_pos = sum(1 for c in og if c["gap_model"] > 0.005)
    ax.set_xlabel("off-grid density (veh/km$^2$)"); ax.set_ylabel("interpolated advantage gap")
    ax.set_title(f"Off-grid interpolation ({n_pos}/{len(og)} positive)")
    save(fig, out, "F3.6", "offgrid_interpolation")


# =========================================================================== §4
def f4_1_currency(out: Path) -> None:
    """Currency faithfulness: F under mean-field / quenched / MC per cell, optimism annotated."""
    data = _load("currency_faithfulness/currency_faithfulness.json")
    cells = data["cells"]
    fig, ax = plt.subplots(figsize=(DBL_W * 0.62, 2.6))
    x = np.arange(len(cells)); w = 0.26
    series = [("F_meanfield", "mean-field (Q=1)", C_BAD), ("F_quenched", "quenched (Q=21)", C_GNN),
              ("F_mc", "Monte-Carlo", C_GOOD)]
    for k, (key, lab, col) in enumerate(series):
        ax.bar(x + (k - 1) * w, [c[key] for c in cells], w, label=lab, color=col, alpha=0.88)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"d{int(c['density'])}/{c['profile'][:4]}\nc{int(c['coupling_db'])}" for c in cells], fontsize=6)
    ax.set_ylabel("failure F (log)"); ax.legend(fontsize=6, loc="upper left")
    for i, c in enumerate(cells):
        ax.annotate(f"{c['meanfield_optimism_x']:.0f}× opt.", (x[i] - w, c["F_meanfield"]),
                    textcoords="offset points", xytext=(0, -12), fontsize=5.5, ha="center", color=C_BAD)
    mfr = data.get("meanfield_optimism_range"); qfr = data.get("quenched_fidelity_range")
    sub = (f"mean-field {mfr[0]:.0f}–{mfr[1]:.0f}× optimistic; quenched {qfr[0]:.2f}–{qfr[1]:.2f}× of MC"
           if mfr and qfr else "")
    ax.set_title("Currency faithfulness\n" + sub, fontsize=7.5)
    save(fig, out, "F4.1", "currency_faithfulness")


def f4_5_w0_band(out: Path) -> None:
    """w=0 rel-only multi-seed band vs w>0 optimized: F band + D/E blow-up."""
    s = _load("w0_seed_band/w0_seed_band.json")
    arms = list(s["arms"].keys())
    rel = [a for a in arms if a.startswith("w0")][0]
    opt = [a for a in arms if a != rel][0]
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W * 0.72, 2.7))
    # left: F per-seed spread (the honest story: rel-only is unstable across inits)
    ax = axes[0]
    sig = {}; jit = np.linspace(-0.12, 0.12, 5)
    for i, (a, col) in enumerate([(rel, C_BAD), (opt, C_GNN)]):
        ps = [r["F"] for r in s["arms"][a]["per_seed"]]
        sig[a] = float(np.std(ps))
        ax.scatter(i + jit[:len(ps)], ps, color=col, s=32, zorder=5, alpha=0.9, edgecolors="k", linewidths=0.3)
        ax.scatter([i], [np.mean(ps)], marker="_", s=900, color=col, lw=1.8, zorder=4)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["w=0\nrel-only", f"w={s['arms'][opt]['w_cost']:g}\noptimized"], fontsize=6.5)
    ax.set_ylabel("held-out failure F  (per init seed)")
    ax.annotate(f"$\\sigma$={sig[rel]:.2f}\n(bimodal:\ncollapse vs escape)", (0, max(r['F'] for r in s['arms'][rel]['per_seed'])),
                textcoords="offset points", xytext=(14, -2), fontsize=6, color=C_BAD, va="top")
    ax.annotate(f"$\\sigma$={sig[opt]:.2f}\n(tight)", (1, np.mean([r['F'] for r in s['arms'][opt]['per_seed']])),
                textcoords="offset points", xytext=(12, 0), fontsize=6, color=C_GNN, va="center")
    ax.set_xlim(-0.4, 1.7); ax.set_title("Cost terms remove init-variance")
    # right: D/E blow-up (robust)
    ax = axes[1]
    db = s["D_blowup_x"]; eb = s["E_blowup_x"]
    ax.bar([0, 1], [db, eb], 0.5, color=[C_ACC, C_ORANGE], alpha=0.85)
    ax.axhline(1.0, ls="--", lw=0.7, color=C_GREY)
    for i, v in enumerate([db, eb]):
        ax.annotate(f"{v:.0f}×", (i, v), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["delay D", "energy E"]); ax.set_ylabel("rel-only / optimized (×)")
    ax.set_title("Cost blow-up (robust, 26×)")
    fig.suptitle("w=0 reliability-only across 5 init seeds: unstable; cost terms regularize", y=1.03, fontsize=8.5, fontweight="bold")
    save(fig, out, "F4.5", "w0_seed_band")


def _s2_arms() -> dict:
    base = RESULT / "s2_temporal_null"
    arms = {}
    if base.exists():
        for p in sorted(base.glob("*.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            arms[d.get("arm", p.stem)] = d
    return arms


def f4_2_temporal_null(out: Path) -> None:
    """Temporal mean-F null: static / no_memory / full / filter mean F equal (memory buys stability not accuracy)."""
    arms = _s2_arms()
    if not arms:
        print("  F4.2 skipped (no result/s2_temporal_null/*.json)"); return
    order = [a for a in ["static", "no_memory", "full", "filter"] if a in arms] or list(arms)
    # init-variance envelope from the §2 emission study (capacity-matched arms) for an honest reference band
    sig_ref = 0.0
    try:
        dec = _load("emission_2x2_isolation/summary.json")["decomposition"]
        sig_ref = float(np.mean([dec[a]["sigma_init"] for a in ("full", "no_graph") if a in dec]))
    except Exception:  # noqa: BLE001
        pass
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    x = np.arange(len(order)); w = 0.4
    base = [arms[a]["F"] for a in order]
    ax.bar(x - w / 2, base, w, color=C_GNN, alpha=0.88, label="mean F (single init)")
    ax.bar(x + w / 2, [arms[a].get("worstF", np.nan) for a in order], w, color=C_ACC, alpha=0.7, label="worst-frame F")
    if sig_ref:
        ax.errorbar(x - w / 2, base, yerr=sig_ref, fmt="none", ecolor="k", elinewidth=0.8, capsize=3, zorder=5)
        ax.text(0.02, 0.97, f"error bar = §2 $\\sigma_{{init}}$≈{sig_ref:.2f}\n(arms within this envelope)",
                transform=ax.transAxes, fontsize=5.5, va="top", color=C_GREY)
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=12, fontsize=6.5)
    ax.set_ylabel("failure F"); ax.legend(fontsize=6, loc="upper right")
    ax.set_title("Temporal arms within the init-variance envelope")
    save(fig, out, "F4.2", "temporal_null")


def f4_3_fair_heuristics(out: Path) -> None:
    """Fair stream heuristics: GNN F vs channel-rank & carried-reliability on identical held-out frames."""
    arms = _s2_arms()
    arm = arms.get("filter") or arms.get("full") or (next(iter(arms.values())) if arms else None)
    if not arm or "heur_channel_F" not in arm:
        print("  F4.3 skipped (no s2 arm with heur baselines)"); return
    names = ["GNN\n(learned)", "channel-rank\nheuristic", "carried-rel.\nheuristic"]
    vals = [arm["F"], arm["heur_channel_F"], arm["heur_carried_F"]]
    cols = [C_GNN, C_HEUR, C_ORANGE]
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.bar(range(3), vals, 0.6, color=cols, alpha=0.88)
    # honest: parity with channel-rank, large win over carried-reliability
    win_carried = (vals[2] - vals[0]) / vals[2] * 100 if vals[2] > 0 else 0
    ax.annotate(f"−{win_carried:.0f}%\nvs carried-rel.", (0, vals[0]), textcoords="offset points",
                xytext=(20, 14), ha="center", fontsize=6, color=C_GNN,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=C_GNN))
    ax.annotate("≈ parity", (0.5, max(vals[0], vals[1])), textcoords="offset points",
                xytext=(0, 6), ha="center", fontsize=6, color=C_GREY)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylabel("held-out failure F")
    ax.set_title(f"Fair stream heuristics — same info, arm '{arm.get('arm','?')}'\n(matches channel-rank, beats carried-reliability)", fontsize=7)
    save(fig, out, "F4.3", "fair_heuristics")


# =========================================================================== driver
FIGS = {
    "F0.2": f0_2_floor,
    "F1.3": f1_3_trajectory, "F1.4": f1_4_two_config, "F1.5": f1_5_w0_dominated,
    "F1.9": f1_9_transfer_regret, "F1.10": f1_10_advantage_heatmap, "F1.12": f1_12_mc_survival,
    "F2.2": f2_2_sigma_init, "F2.3": f2_3_probe, "F2.4": f2_4_outcome_range, "F2.5": f2_5_replication,
    "F3.1": f3_1_gating_map, "F3.2": f3_2_routing, "F3.3": f3_3_context,
    "F3.4": f3_4_retention, "F3.5": f3_5_diagnostic, "F3.6": f3_6_offgrid,
    "F4.1": f4_1_currency, "F4.2": f4_2_temporal_null, "F4.3": f4_3_fair_heuristics, "F4.5": f4_5_w0_band,
}


def main() -> None:
    p = argparse.ArgumentParser(description="Generate paper figures (IEEE style) from result/*.json")
    p.add_argument("--only", default="", help="comma list of figure ids (default: all)")
    p.add_argument("--out", default="result/paper_figures")
    args = p.parse_args()
    set_ieee_style()
    out = ROOT / args.out
    want = [s.strip() for s in args.only.split(",") if s.strip()] or list(FIGS)
    print(f"Rendering {len(want)} figures -> {out}", flush=True)
    ok, fail = 0, []
    for fid in want:
        fn = FIGS.get(fid)
        if fn is None:
            print(f"  ?? unknown figure {fid}"); continue
        try:
            fn(out); ok += 1
        except Exception as e:  # noqa: BLE001
            fail.append((fid, repr(e))); print(f"  !! {fid} failed: {e!r}", flush=True)
    print(f"\ndone: {ok} ok, {len(fail)} failed")
    for fid, e in fail:
        print(f"  FAILED {fid}: {e}")


if __name__ == "__main__":
    main()
