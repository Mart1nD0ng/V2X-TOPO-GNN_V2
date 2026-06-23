"""Rebuild paper result figures with a unified IEEE journal style.

The script uses the existing experiment JSON artifacts under ``result/`` and
exports upgraded PDF/PNG figures into ``result/paper_figures``. With
``--sync-paper`` it also copies the same files into ``paper/figures`` so the
manuscript consumes the upgraded assets without changing LaTeX paths.

It does not rerun training or change any reported experimental value.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "result" / "paper_figures"
PAPER_DIR = ROOT / "paper" / "figures"


COL = {
    "ink": "#1F2A35",
    "muted": "#667587",
    "grid": "#D8E0E8",
    "light": "#F5F7FA",
    "blue": "#174A7C",
    "blue2": "#3B77AF",
    "teal": "#2A9D8F",
    "green": "#3A9D62",
    "amber": "#D49B2A",
    "orange": "#D9792B",
    "red": "#B9413E",
    "purple": "#6E5AA7",
    "pink": "#C0648A",
}
METHOD_COLORS = {
    "GNN": COL["blue"],
    "planner": COL["blue"],
    "expert": COL["teal"],
    "heuristic": COL["amber"],
    "floor": "#7F8C8D",
    "w0": COL["red"],
    "w5": COL["blue"],
    "emission_on": COL["teal"],
    "emission_off": COL["red"],
    "governed": COL["blue"],
    "naive": COL["orange"],
    "loco": COL["teal"],
}


def configure_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 140,
        }
    )


def j(path: str) -> Any:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13,
        1.08,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=COL["ink"],
    )


def clean(ax: plt.Axes, grid: bool = True) -> None:
    ax.tick_params(length=2.2, width=0.6, color=COL["muted"], labelcolor=COL["ink"], pad=1.8)
    ax.spines["left"].set_color(COL["muted"])
    ax.spines["bottom"].set_color(COL["muted"])
    if grid:
        ax.grid(True, axis="y", color=COL["grid"], lw=0.45, alpha=0.8)
        ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str, sync_paper: bool, *, dpi: int = 600) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    if sync_paper:
        PAPER_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, PAPER_DIR / pdf.name)
        shutil.copy2(png, PAPER_DIR / png.name)


def add_note(ax: plt.Axes, text: str, xy=(0.02, 0.98), ha="left", va="top") -> None:
    ax.text(
        xy[0],
        xy[1],
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=5.8,
        color=COL["ink"],
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=COL["grid"], lw=0.6, alpha=0.95),
    )


def short_variant(name: str) -> str:
    if name.startswith("small_realistic"):
        return "small"
    if name.startswith("wider_query"):
        return "wide"
    if name.startswith("deeper_quorum"):
        return "deep"
    if name.startswith("wider+deeper"):
        return "wide+deep"
    return name.split()[0]


def short_profile(name: str) -> str:
    return {
        "hard_low_confidence": "hard",
        "near_target_synthetic": "near-target",
        "toy": "toy",
        "filter": "+emission",
        "full": "-emission",
        "no_graph": "-emission\n(no graph)",
        "no_graph_filter": "+emission\n(no graph)",
        "filter_nomem": "+emission\n(no mem)",
        "static": "static",
        "no_memory": "no memory",
    }.get(str(name), str(name).replace("_", "\n"))


def confidence_interval(mean: float, ci: float) -> tuple[float, float]:
    return mean - ci, mean + ci


def figure_protocol_floor(sync_paper: bool) -> None:
    data = j("result/protocol_floor_table/floor_table.json")
    variants = list(data["rows"][0]["floors"].keys())
    profiles = sorted({r["profile"] for r in data["rows"]})
    degrees = sorted({r["degree"] for r in data["rows"]})
    fig, axes = plt.subplots(1, len(degrees), figsize=(3.55, 2.25), sharey=True)
    cmap = LinearSegmentedColormap.from_list("floor", ["#EAF2F8", "#73A9D8", "#174A7C"])
    for ax, deg, lab in zip(axes, degrees, ["a", "b"]):
        matrix = np.full((len(profiles), len(variants)), np.nan)
        feasible = np.zeros_like(matrix, dtype=bool)
        for row in data["rows"]:
            if int(row["degree"]) != int(deg):
                continue
            i = profiles.index(row["profile"])
            for k, v in enumerate(variants):
                matrix[i, k] = row["floors"][v]
                feasible[i, k] = bool(row["feasible_targets"][v])
        im = ax.imshow(-np.log10(matrix), cmap=cmap, vmin=1.0, vmax=2.6, aspect="auto")
        for i in range(matrix.shape[0]):
            for k in range(matrix.shape[1]):
                ax.text(k, i, f"{matrix[i,k]:.3f}", ha="center", va="center", fontsize=4.9, color=COL["ink"])
                if feasible[i, k]:
                    ax.add_patch(Rectangle((k - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec=COL["green"], lw=1.1))
        ax.set_title(f"degree budget {deg}")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels([short_variant(v) for v in variants], rotation=30, ha="right")
        ax.set_yticks(range(len(profiles)))
        ax.set_yticklabels([short_profile(p) for p in profiles])
        ax.tick_params(length=0)
        panel(ax, lab)
        ax.set_frame_on(False)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(r"$-\log_{10}$ perfect-link $F$")
    fig.suptitle("Protocol floor is degree- and profile-dependent", y=1.02, fontsize=8, fontweight="bold")
    save(fig, "F0.2_protocol_floor", sync_paper)


def pareto_rows(which: str) -> list[dict[str, Any]]:
    return j(f"result/{which}/report.json")["pareto"]["rows"]


def figure_two_config(sync_paper: bool) -> None:
    paper = pareto_rows("production_report_paperenv")
    op = pareto_rows("production_report_operating_point")
    fig = plt.figure(figsize=(3.55, 2.35))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.55], wspace=0.34)
    for ax, rows, title, lab in [
        (fig.add_subplot(gs[0, 0]), paper, "paper-env\n(no D/E lever)", "a"),
        (fig.add_subplot(gs[0, 1]), op, "operating point\n(live D/E lever)", "b"),
    ]:
        w = np.array([r["w_cost"] for r in rows], float)
        F = np.array([r["holdout_F"]["mean"] for r in rows])
        D = np.array([r["holdout_D"]["mean"] for r in rows])
        ax.plot(F, D, "-", color=COL["muted"], lw=0.8, zorder=1)
        sc = ax.scatter(F, D, c=w, cmap="viridis", s=22 + 10 * (w == 5), ec="white", lw=0.4, zorder=3)
        idx = int(np.argmin(np.abs(w - 5)))
        ax.scatter(F[idx], D[idx], s=55, marker="*", color=COL["red"], ec="white", lw=0.5, zorder=4)
        ax.set_yscale("log")
        ax.set_xlabel("failure F")
        ax.set_ylabel("delay D")
        ax.set_title(title)
        clean(ax)
        panel(ax, lab)
    ax = fig.add_subplot(gs[0, 2])
    paper_drop = paper[0]["holdout_D"]["mean"] / max(paper[-1]["holdout_D"]["mean"], 1e-9)
    op_drop = op[0]["holdout_D"]["mean"] / max(op[2]["holdout_D"]["mean"], 1e-9)
    bars = ax.bar([0, 1], [paper_drop, op_drop], color=[COL["muted"], COL["blue"]], width=0.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["flat", "live"], rotation=35, ha="right")
    ax.set_ylabel(r"$D(w=0)/D(w=5)$")
    clean(ax)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.03, f"{b.get_height():.0f}x", ha="center", va="bottom", fontsize=5.5)
    panel(ax, "c")
    fig.colorbar(sc, ax=fig.axes[:2], fraction=0.035, pad=0.03, label="cost weight w")
    save(fig, "F1.4_two_config_pareto", sync_paper)


def figure_pareto_swarm(sync_paper: bool) -> None:
    rows = pareto_rows("production_report_operating_point")
    rows_pos = [r for r in rows if float(r["w_cost"]) > 0]
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.25), sharex=True)
    rng = np.random.default_rng(3)
    for ax, metric, ylabel, lab in zip(axes, ["D", "E"], ["delay D (log)", "energy E (log)"], ["a", "b"]):
        for r in rows_pos:
            w = float(r["w_cost"])
            Fvals = np.array([s["F"] for s in r["holdout_per_scene"]])
            Yvals = np.array([s[metric] for s in r["holdout_per_scene"]])
            xj = Fvals + rng.normal(0, 0.0012, size=len(Fvals))
            ax.scatter(xj, Yvals, s=9, alpha=0.55, color=COL["muted"], lw=0)
            mF = r["holdout_F"]["mean"]
            mY = r[f"holdout_{metric}"]["mean"]
            cih = r[f"holdout_{metric}"]["ci_halfwidth"]
            color = COL["red"] if abs(w - 5) < 1e-6 else COL["blue2"]
            ax.errorbar(mF, mY, yerr=cih, fmt="o", ms=4.0, color=color, ecolor=color, capsize=2, lw=0.8)
            ax.text(mF, mY * 1.12, f"w={w:g}", fontsize=5.3, ha="center", color=color)
        ax.axhline(rows[0][f"holdout_{metric}"]["mean"], ls="--", lw=0.7, color=COL["red"], alpha=0.6)
        ax.set_yscale("log")
        ax.set_xlabel("failure F")
        ax.set_ylabel(ylabel)
        clean(ax)
        panel(ax, lab)
    add_note(axes[0], "w=0 is off-scale:\nF=0.73, D=6287\n(single init)", xy=(0.04, 0.96))
    fig.suptitle("Coupled cost-reliability front with scene-level spread", y=1.02, fontsize=8, fontweight="bold")
    save(fig, "F5.8_pareto_swarm", sync_paper)


def figure_pareto_trajectory(sync_paper: bool) -> None:
    tr = j("result/production_report_operating_point/report.json")["trajectory"]
    step = np.array([t["step"] for t in tr])
    F = np.array([t["F"] for t in tr])
    D = np.array([t["D"] for t in tr])
    grad = np.array([t["grad_norm"] for t in tr])
    parts = {k: np.array([t[k] for t in tr]) for k in ["wL_R", "wL_D", "wL_E"]}
    fig = plt.figure(figsize=(3.55, 2.55))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    sc = ax.scatter(F, D, c=step, cmap="viridis", s=7, zorder=3)
    ax.plot(F, D, color=COL["muted"], lw=0.45, alpha=0.8)
    ax.scatter(F[0], D[0], s=40, color=COL["red"], marker="x", lw=1.1, label="start")
    ax.scatter(F[-1], D[-1], s=40, color=COL["green"], marker="*", label="final")
    ax.set_yscale("log")
    ax.set_xlabel("failure F")
    ax.set_ylabel("delay D")
    clean(ax)
    panel(ax, "a")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
    cbar.set_label("step")
    ax = fig.add_subplot(gs[0, 1])
    ax.stackplot(step, parts["wL_D"], parts["wL_R"], parts["wL_E"], colors=[COL["blue2"], COL["teal"], COL["amber"]], alpha=0.82, labels=["D loss", "R loss", "E loss"])
    ax2 = ax.twinx()
    ax2.plot(step, grad, color=COL["red"], lw=0.9)
    ax2.set_yscale("log")
    ax2.set_ylabel("grad norm", color=COL["red"])
    ax.set_xlabel("training step")
    ax.set_ylabel("weighted loss")
    ax.legend(loc="upper right", fontsize=5.4)
    clean(ax)
    ax2.tick_params(labelsize=5.8, colors=COL["red"], length=2)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(COL["red"])
    panel(ax, "b")
    save(fig, "F1.3_pareto_trajectory", sync_paper)


def figure_transfer_regret(sync_paper: bool) -> None:
    data = j("result/scale_transfer/scale_transfer.json")
    rows = data["per_seed"]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(3.35, 1.95))
    for i, r in enumerate(rows):
        yi = y[i]
        ax.plot([r["F_expert"], r["F_planner"]], [yi, yi], color=COL["grid"], lw=2.0, zorder=1)
        ax.scatter(r["F_expert"], yi, s=28, color=COL["teal"], ec="white", lw=0.5, zorder=3)
        ax.scatter(r["F_planner"], yi, s=28, color=COL["blue"], ec="white", lw=0.5, zorder=3)
        ax.text(r["F_planner"] + 0.00022, yi, f"+{r['regret']:.4f}", va="center", fontsize=5.5, color=COL["red"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"seed {r['scene_seed']}" for r in rows])
    ax.set_xlabel("failure F at N=10000")
    ax.set_title("Scale transfer: one planner vs N-specific expert")
    ax.scatter([], [], color=COL["teal"], label="N=10000 expert")
    ax.scatter([], [], color=COL["blue"], label="N-randomized planner")
    ax.legend(loc="lower right", fontsize=5.5)
    clean(ax, grid=False)
    add_note(ax, f"mean regret = +{data['transfer_regret_mean']:.4f}", xy=(0.03, 0.12), va="bottom")
    save(fig, "F1.9_transfer_regret", sync_paper)


def cells_advantage() -> list[dict[str, Any]]:
    return j("result/advantage_map/advantage_map.json")["cells"]


def figure_advantage_bubble(sync_paper: bool) -> None:
    cells = cells_advantage()
    mc = j("result/advantage_montecarlo/advantage_mc.json")["cells"]
    mc_keys = {(c["profile"], float(c["density"]), float(c["coupling_db"])) for c in mc}
    profiles = ["hard_low_confidence", "near_target_synthetic", "toy"]
    densities = [100.0, 200.0, 300.0]
    couplings = [0.0, 10.0, 20.0]
    fig, axes = plt.subplots(1, 3, figsize=(6.95, 2.35), sharey=True)
    for ax, prof, lab in zip(axes, profiles, ["a", "b", "c"]):
        sub = [c for c in cells if c["profile"] == prof]
        for c in sub:
            x = couplings.index(float(c["coupling_db"]))
            y = densities.index(float(c["density"]))
            gap = float(c["gap_mean"])
            robust = bool(c["label_robust"])
            color = COL["green"] if robust and gap > 0 else (COL["amber"] if gap > 0 else COL["muted"])
            size = 60 + 1450 * abs(gap)
            ax.scatter(x, y, s=size, color=color, alpha=0.92, ec=COL["ink"] if robust else "white", lw=1.0 if robust else 0.45)
            ax.text(x, y, f"{gap:+.2f}", ha="center", va="center", fontsize=5.2, color="white" if robust else COL["ink"])
            if (prof, float(c["density"]), float(c["coupling_db"])) in mc_keys:
                ax.scatter(x, y, s=size + 45, facecolors="none", edgecolors=COL["blue"], lw=1.5)
        ax.set_xticks(range(len(couplings)))
        ax.set_xticklabels([f"{int(c)}" for c in couplings])
        ax.set_yticks(range(len(densities)))
        ax.set_yticklabels([f"{int(d)}" for d in densities])
        ax.set_xlabel("interference coupling (dB)")
        ax.set_title(short_profile(prof))
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(-0.55, 2.55)
        ax.grid(True, color=COL["grid"], lw=0.5)
        ax.set_axisbelow(True)
        panel(ax, lab)
    axes[0].set_ylabel("density (veh/km$^2$)")
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["green"], markeredgecolor=COL["ink"], markersize=7, label="seed-robust GNN gap"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=COL["amber"], markeredgecolor="white", markersize=7, label="positive but fragile"),
        plt.Line2D([0], [0], marker="o", color=COL["blue"], markerfacecolor="none", markersize=7, label="MC-audited"),
    ]
    axes[2].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.05, -0.05), fontsize=5.5)
    fig.suptitle("Advantage region is sparse, structured, and MC-audited only at the ringed cells", y=1.04, fontsize=8, fontweight="bold")
    save(fig, "F5.7_advantage_bubble_map", sync_paper)
    save(fig, "F1.10_advantage_heatmap", sync_paper)


def figure_mc_survival(sync_paper: bool) -> None:
    cells = j("result/advantage_montecarlo/advantage_mc.json")["cells"]
    fig, ax = plt.subplots(figsize=(3.15, 1.95))
    y0 = np.array([c["gap_quenched"] for c in cells])
    y1 = np.array([c["gap_mc"] for c in cells])
    labels = [short_profile(c["profile"]) for c in cells]
    for i, (a, b) in enumerate(zip(y0, y1)):
        ax.plot([0, 1], [a, b], color=COL["teal"], lw=1.0)
        ax.scatter([0, 1], [a, b], s=24, color=COL["teal"], ec="white", lw=0.5, zorder=3)
        ax.text(1.03, b, labels[i], fontsize=5.5, va="center")
    ax.axhline(0, color=COL["muted"], ls="--", lw=0.7)
    ax.set_xlim(-0.1, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["quenched\nsurrogate", "Monte-Carlo\nhold-out"])
    ax.set_ylabel("GNN advantage gap")
    ax.set_title("MC check preserves the advantage sign")
    clean(ax)
    save(fig, "F1.12_mc_survival", sync_paper)


def emission_summary(path: str = "result/emission_2x2_isolation/summary.json") -> dict[str, Any]:
    return j(path)["decomposition"]


def figure_emission_radar(sync_paper: bool) -> None:
    dec = emission_summary()
    arms = ["full", "no_graph", "filter", "no_graph_filter"]
    metrics = ["stability", "accuracy", "worst-case", "best-case", "consistency"]
    raw = {}
    for arm in arms:
        d = dec[arm]
        raw[arm] = np.array([
            -d["sigma_init"],
            -d["F_mean"],
            -d["F_max"],
            -d["F_min"],
            -d["init_range"],
        ])
    mat = np.vstack([raw[a] for a in arms])
    norm = (mat - mat.min(axis=0)) / (np.ptp(mat, axis=0) + 1e-12)
    theta = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    theta = np.r_[theta, theta[0]]
    fig = plt.figure(figsize=(3.25, 3.0))
    ax = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(top=0.82, bottom=0.24)
    colors = [COL["red"], COL["red"], COL["teal"], COL["teal"]]
    lss = ["-", "--", "-", "--"]
    for i, arm in enumerate(arms):
        vals = np.r_[norm[i], norm[i, 0]]
        ax.plot(theta, vals, color=colors[i], lw=1.2, ls=lss[i], label=short_profile(arm))
        ax.fill(theta, vals, color=colors[i], alpha=0.10 if "filter" in arm else 0.05)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(metrics, fontsize=5.8)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    ax.grid(color=COL["grid"], lw=0.6)
    ax.spines["polar"].set_color(COL["muted"])
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=2,
        fontsize=5.4,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )
    ax.set_title("Emission turns collapse into a stability polygon", y=1.12, fontsize=7.5, fontweight="bold")
    save(fig, "F5.3_ablation_radar", sync_paper)


def figure_sigma_init(sync_paper: bool) -> None:
    dec = emission_summary()
    arms = ["full", "no_graph", "filter", "no_graph_filter", "filter_nomem"]
    vals = np.array([dec[a]["sigma_init"] for a in arms])
    fig, ax = plt.subplots(figsize=(3.25, 2.05))
    colors = [COL["red"], COL["red"], COL["teal"], COL["teal"], COL["purple"]]
    bars = ax.bar(range(len(arms)), vals, color=colors, edgecolor="white", lw=0.5)
    ax.set_yscale("log")
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([short_profile(a) for a in arms], rotation=28, ha="right")
    ax.set_ylabel(r"$\sigma_{\rm init}$ of final F")
    ax.set_title("Emission collapses initialization variance")
    clean(ax)
    ratio = vals[0] / vals[2]
    ax.annotate(f"{ratio:.0f}x drop\nwith emission", xy=(2, vals[2]), xytext=(2.65, vals[0] * 0.55),
                arrowprops=dict(arrowstyle="->", color=COL["green"], lw=0.8), fontsize=5.8, color=COL["green"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*1.14, f"{v:.3f}", ha="center", va="bottom", fontsize=4.9)
    save(fig, "F2.2_sigma_init", sync_paper)


def figure_probe_divergence(sync_paper: bool) -> None:
    data = j("result/emission_probe_paperenv/emission_probe.json")
    fig, axes = plt.subplots(1, 2, figsize=(3.55, 2.05), sharex=True)
    for ax, key, ylabel, lab in zip(axes, ["joined_std", "gate_grad_norm"], ["recurrent input std", "gate grad norm"], ["a", "b"]):
        for arm, color in [("full", COL["red"]), ("filter", COL["teal"])]:
            runs = [r for r in data["runs"] if r["arm"] == arm]
            arr = np.array([[e[key] for e in r["trajectory"]] for r in runs])
            x = np.arange(arr.shape[1])
            mu = arr.mean(axis=0)
            sd = arr.std(axis=0)
            ax.plot(x, mu, color=color, lw=1.1, label=short_profile(arm))
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.16, lw=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log" if key == "gate_grad_norm" else "linear")
        clean(ax)
        panel(ax, lab)
    axes[0].legend(loc="upper left", fontsize=5.5)
    fig.suptitle("Mechanism probe: emission bounds recurrent state and gradients", y=1.04, fontsize=8, fontweight="bold")
    save(fig, "F2.3_probe_divergence", sync_paper)


def figure_outcome_range(sync_paper: bool) -> None:
    dec = emission_summary()
    arms = ["full", "no_graph", "filter", "no_graph_filter", "filter_nomem"]
    fig, ax = plt.subplots(figsize=(3.3, 2.0))
    for i, arm in enumerate(arms):
        d = dec[arm]
        color = COL["teal"] if "filter" in arm else COL["red"]
        ax.plot([i, i], [d["F_min"], d["F_max"]], color=color, lw=1.4, alpha=0.75)
        ax.scatter(i, d["F_mean"], s=28, color=color, ec="white", lw=0.5, zorder=3)
    ax.axhspan(0.5, 1.0, color=COL["red"], alpha=0.05, zorder=0)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([short_profile(a) for a in arms], rotation=28, ha="right")
    ax.set_ylabel("failure F range over inits")
    ax.set_title("Collapse vs escape across initializations")
    clean(ax)
    save(fig, "F2.4_outcome_range", sync_paper)


def figure_replication(sync_paper: bool) -> None:
    hard = emission_summary("result/emission_2x2_isolation/summary.json")
    easy = emission_summary("result/emission_replication_d100_c0/summary.json")
    regimes = ["d100 c20", "d100 c0"]
    full = [hard["full"]["sigma_init"], easy["full"]["sigma_init"]]
    filt = [hard["filter"]["sigma_init"], easy["filter"]["sigma_init"]]
    fig, ax = plt.subplots(figsize=(3.25, 2.0))
    x = np.arange(2)
    ax.plot(x, full, "-o", color=COL["red"], label="-emission", lw=1.0)
    ax.plot(x, filt, "-o", color=COL["teal"], label="+emission", lw=1.0)
    for i in range(2):
        ax.plot([x[i], x[i]], [filt[i], full[i]], color=COL["grid"], lw=2.5, zorder=0)
        ax.text(x[i] + 0.04, math.sqrt(full[i] * filt[i]), f"{full[i]/filt[i]:.0f}x", va="center", fontsize=5.5, color=COL["green"])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes)
    ax.set_ylabel(r"$\sigma_{\rm init}$")
    ax.set_title("Stabilization is regime-specific")
    ax.legend(loc="upper right")
    clean(ax)
    save(fig, "F2.5_replication", sync_paper)


def figure_gating_alluvial(sync_paper: bool) -> None:
    table = j("result/gating_demo/gating_table.json")
    entries = table["entries"]
    densities = sorted(table["densities"])
    decisions = ["USE_GNN", "USE_GNN_MARGINAL", "GNN_DEFAULT", "HEURISTIC_OK", "HEURISTIC_SUFFICES"]
    dlabels = {
        "USE_GNN": "use GNN",
        "USE_GNN_MARGINAL": "GNN marginal",
        "GNN_DEFAULT": "GNN default",
        "HEURISTIC_OK": "heuristic OK",
        "HEURISTIC_SUFFICES": "heuristic OK",
    }
    fig, ax = plt.subplots(figsize=(3.55, 2.2))
    ax.set_axis_off()
    y_density = {d: 0.82 - i * 0.31 for i, d in enumerate(densities)}
    y_dec = {d: 0.86 - i * 0.22 for i, d in enumerate(decisions)}
    for d in densities:
        ax.add_patch(Rectangle((0.02, y_density[d] - 0.055), 0.16, 0.11, fc="#EEF4F8", ec=COL["muted"], lw=0.7))
        ax.text(0.10, y_density[d], f"{int(d)}\nveh/km$^2$", ha="center", va="center", fontsize=5.8)
    active_decisions = [d for d in decisions if any(e["decision"] == d for e in entries)]
    for dec in active_decisions:
        col = COL["green"] if "GNN" in dec and "HEURISTIC" not in dec else COL["amber"]
        ax.add_patch(Rectangle((0.78, y_dec[dec] - 0.045), 0.19, 0.09, fc=col, ec="white", lw=0.7, alpha=0.85))
        ax.text(0.875, y_dec[dec], dlabels[dec], ha="center", va="center", fontsize=5.7, color="white" if dec != "HEURISTIC_SUFFICES" else COL["ink"])
    counts: dict[tuple[float, str], int] = {}
    gaps: dict[tuple[float, str], float] = {}
    for e in entries:
        key = (float(e["density"]), e["decision"])
        counts[key] = counts.get(key, 0) + 1
        lo, hi = e["expected_gap"]
        gaps[key] = gaps.get(key, 0.0) + 0.5 * (lo + hi)
    for (d, dec), cnt in counts.items():
        y0, y1 = y_density[d], y_dec[dec]
        width = 0.012 + cnt * 0.010
        color = COL["green"] if "GNN" in dec and "HEURISTIC" not in dec else COL["amber"]
        verts = [(0.18, y0), (0.36, y0), (0.58, y1), (0.78, y1)]
        path = MplPath(verts, [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
        ax.add_patch(PathPatch(path, lw=width * 90, edgecolor=color, facecolor="none", alpha=0.38, capstyle="round"))
    ax.text(0.02, 0.97, "deploy-time context", fontsize=6.5, fontweight="bold", color=COL["ink"])
    ax.text(0.78, 0.97, "policy decision", fontsize=6.5, fontweight="bold", color=COL["ink"])
    ax.text(0.50, 0.08, "flow width = number of coupling cells; green = learned constructor deployed", ha="center", fontsize=5.5, color=COL["muted"])
    save(fig, "F5.10_gating_sankey", sync_paper)


def figure_retention_governance(sync_paper: bool) -> None:
    gov = j("result/mixture_governed_gradnorm/envelope_governed.json")
    comp = gov["comparison_by_density"]
    dens = sorted(comp.keys())
    fig = plt.figure(figsize=(3.55, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 0.82], wspace=0.36)
    ax = fig.add_subplot(gs[0, 0])
    labels = [d.replace("d", "") for d in dens]
    series = ["naive", "governed", "loco_ceiling"]
    colors = [COL["orange"], COL["blue"], COL["teal"]]
    x = np.arange(len(dens))
    w = 0.24
    for i, s in enumerate(series):
        vals = [comp[d][s] for d in dens]
        ax.bar(x + (i - 1) * w, vals, width=w, color=colors[i], label=s.replace("_", " "), ec="white", lw=0.5)
    ax.axhline(1.0, color=COL["muted"], ls="--", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("density")
    ax.set_ylabel("retention")
    ax.legend(loc="lower right", fontsize=5.3)
    clean(ax)
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    norms = gov["diagnostic"]["group_grad_norms"]
    coss = gov["diagnostic"]["pairwise_cosines"]
    coords = {"100.0": (0.2, 0.75), "200.0": (0.75, 0.72), "300.0": (0.48, 0.25)}
    for pair, cos in coss.items():
        a, b = pair.split("|")
        xy1, xy2 = coords[a], coords[b]
        ax.plot([xy1[0], xy2[0]], [xy1[1], xy2[1]], color=COL["grid"], lw=1 + 3 * max(cos, 0), zorder=1)
        ax.text((xy1[0]+xy2[0])/2, (xy1[1]+xy2[1])/2, f"{cos:.2f}", fontsize=5.2, color=COL["muted"])
    max_norm = max(norms.values())
    for k, xy in coords.items():
        r = 0.065 + 0.055 * norms[k] / max_norm
        ax.add_patch(Circle(xy, r, fc=COL["blue2"], ec="white", lw=0.7, alpha=0.88, zorder=3))
        ax.text(xy[0], xy[1], k.split(".")[0], ha="center", va="center", color="white", fontsize=5.8, fontweight="bold")
    ax.set_title("gradient geometry")
    ax.set_axis_off()
    panel(ax, "b")
    fig.suptitle("Governed mixture recovers the per-density ceiling", y=1.02, fontsize=8, fontweight="bold")
    save(fig, "F5.9_retention_cosine_web", sync_paper)
    save(fig, "F3.4_retention", sync_paper)
    save(fig, "F3.5_gradient_diagnostic", sync_paper)


def figure_offgrid(sync_paper: bool) -> None:
    off = j("result/mixture_governed_gradnorm/envelope_governed.json")["off_grid"]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    for p in sorted({r["profile"] for r in off}):
        rows = [r for r in off if r["profile"] == p]
        x = [r["density"] + (r["coupling_db"] - 10) * 0.6 for r in rows]
        y = [r["gap_model"] for r in rows]
        ax.scatter(x, y, s=24, label=short_profile(p), alpha=0.88)
    ax.axhline(0, color=COL["muted"], ls="--", lw=0.7)
    ax.set_xlabel("off-grid density (jittered by coupling)")
    ax.set_ylabel("interpolated advantage gap")
    ax.set_title("Off-grid interpolation remains positive where headroom exists")
    ax.legend(loc="upper right", fontsize=5.4)
    clean(ax)
    save(fig, "F3.6_offgrid_interpolation", sync_paper)


def figure_gating_map(sync_paper: bool) -> None:
    table = j("result/gating_demo/gating_table.json")
    dens = sorted(table["densities"])
    coups = sorted(table["couplings"])
    decisions = {e["decision"] for e in table["entries"]}
    code = {"USE_GNN": 3, "USE_GNN_MARGINAL": 2, "GNN_DEFAULT": 1, "HEURISTIC_OK": 0, "HEURISTIC_SUFFICES": 0}
    mat = np.zeros((len(coups), len(dens)))
    for e in table["entries"]:
        mat[coups.index(e["coupling_db"]), dens.index(e["density"])] = code[e["decision"]]
    cmap = LinearSegmentedColormap.from_list("gate", [COL["amber"], COL["muted"], COL["blue2"], COL["green"]])
    fig, ax = plt.subplots(figsize=(3.1, 2.0))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=3, aspect="auto")
    for i, c in enumerate(coups):
        for jx, d in enumerate(dens):
            e = next(e for e in table["entries"] if e["coupling_db"] == c and e["density"] == d)
            ax.text(jx, i, e["decision"].replace("_", "\n").replace("SUFFICES", "OK"), ha="center", va="center", fontsize=5.0, color="white" if mat[i, jx] > 1.5 else COL["ink"])
    ax.set_xticks(range(len(dens)))
    ax.set_xticklabels([int(d) for d in dens])
    ax.set_yticks(range(len(coups)))
    ax.set_yticklabels([int(c) for c in coups])
    ax.set_xlabel("density")
    ax.set_ylabel("coupling (dB)")
    ax.set_title("Deployment gate decision surface")
    ax.tick_params(length=0)
    save(fig, "F3.1_gating_map", sync_paper)


def figure_routing_validation(sync_paper: bool) -> None:
    frames = j("result/gating_demo/gating_demo_log.json")["frames"]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    x = np.arange(len(frames))
    realized = [f["realized_heuristic_F"] for f in frames]
    lo = [f["expected_heuristic_F"][0] for f in frames]
    hi = [f["expected_heuristic_F"][1] for f in frames]
    for i in range(1, len(frames)):
        if frames[i]["gate_cell"]["density"] != frames[i - 1]["gate_cell"]["density"]:
            ax.axvline(i - 0.5, color=COL["grid"], lw=0.8, zorder=0)
    ax.fill_between(x, lo, hi, color=COL["amber"], alpha=0.18, label="expected heuristic band")
    ax.plot(x, realized, "-o", ms=3, lw=0.9, color=COL["teal"], label="realized heuristic F")
    ax.set_xlabel("deployment frame")
    ax.set_ylabel("failure F")
    ax.set_title("Forward-only routing consistency")
    ax.legend(loc="upper right", fontsize=5.5)
    clean(ax)
    save(fig, "F3.2_routing_validation", sync_paper)


def figure_context_estimation(sync_paper: bool) -> None:
    frames = j("result/gating_demo/gating_demo_log.json")["frames"]
    true = [f["gate_cell"]["density"] for f in frames]
    est = [f["estimated_density"] for f in frames]
    fig, ax = plt.subplots(figsize=(2.55, 2.0))
    ax.scatter(true, est, s=18, color=COL["blue"], ec="white", lw=0.4)
    mn, mx = 80, 320
    ax.plot([mn, mx], [mn, mx], ls="--", lw=0.8, color=COL["muted"])
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)
    ax.set_xlabel("true gate density")
    ax.set_ylabel("estimated density")
    ax.set_title("Deploy-time context estimation")
    clean(ax)
    save(fig, "F3.3_context_estimation", sync_paper)


def figure_currency(sync_paper: bool) -> None:
    cells = j("result/currency_faithfulness/currency_faithfulness.json")["cells"]
    fig, ax = plt.subplots(figsize=(3.35, 2.05))
    xs = [0, 1, 2]
    labels = ["mean-field", "quenched", "Monte-Carlo"]
    colors = [COL["red"], COL["blue"], COL["teal"]]
    for c in cells:
        vals = [c["F_meanfield"], c["F_quenched"], c["F_mc"]]
        lab = f"{short_profile(c['profile'])} d{int(c['density'])} c{int(c['coupling_db'])}"
        ax.plot(xs, vals, "-o", lw=1.0, ms=3.2, label=lab)
        ax.text(2.05, vals[-1], f"{c['quenched_fidelity_x']:.1f}x", fontsize=5.2, va="center")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("failure F (log)")
    ax.set_title("Failure rises with evaluator fidelity")
    ax.legend(loc="upper left", fontsize=4.9)
    clean(ax)
    save(fig, "F5.4_currency_slopegraph", sync_paper)
    save(fig, "F4.1_currency_faithfulness", sync_paper)


def figure_w0_dumbbell(sync_paper: bool) -> None:
    data = j("result/w0_seed_band/w0_seed_band.json")
    a0 = data["arms"]["w0_rel_only"]["per_seed"]
    a5 = data["arms"]["w5_optimized"]["per_seed"]
    by5 = {r["seed"]: r for r in a5}
    seeds = [r["seed"] for r in a0]
    fig = plt.figure(figsize=(3.55, 2.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.85], wspace=0.33)
    ax = fig.add_subplot(gs[0, 0])
    for i, r in enumerate(a0):
        s = r["seed"]
        y = len(seeds) - 1 - i
        ax.plot([r["F"], by5[s]["F"]], [y, y], color=COL["grid"], lw=2.0, zorder=1)
        ax.scatter(r["F"], y, color=COL["red"], s=22, ec="white", lw=0.4, zorder=3)
        ax.scatter(by5[s]["F"], y, color=COL["blue"], s=22, ec="white", lw=0.4, zorder=3)
        if s == 42:
            ax.text(r["F"] - 0.02, y + 0.20, "escape", fontsize=5.0, color=COL["green"], ha="right")
    ax.axvspan(data["F_degradation_rel_vs_opt"]["opt_band"][0], data["F_degradation_rel_vs_opt"]["opt_band"][1], color=COL["blue"], alpha=0.10)
    ax.axvspan(data["F_degradation_rel_vs_opt"]["rel_band"][0], data["F_degradation_rel_vs_opt"]["rel_band"][1], color=COL["red"], alpha=0.08)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f"s{s}" for s in seeds[::-1]])
    ax.set_xlabel("failure F")
    ax.set_title("per-seed convergence")
    ax.scatter([], [], color=COL["red"], label="w=0 rel-only")
    ax.scatter([], [], color=COL["blue"], label="w=5 optimized")
    ax.legend(loc="lower left", fontsize=5.1)
    clean(ax, grid=False)
    panel(ax, "a")
    ax = fig.add_subplot(gs[0, 1])
    vals = [data["D_blowup_x"], data["E_blowup_x"]]
    bars = ax.bar([0, 1], vals, color=[COL["amber"], COL["orange"]], width=0.62, ec="white", lw=0.5)
    ax.axhline(1, color=COL["muted"], ls="--", lw=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["D", "E"])
    ax.set_ylabel("w=0 / w=5")
    ax.set_title("cost blow-up")
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1.0, f"{v:.1f}x", ha="center", fontsize=5.5)
    clean(ax)
    panel(ax, "b")
    save(fig, "F5.5_w0_dumbbell", sync_paper)
    save(fig, "F4.5_w0_seed_band", sync_paper)


def figure_temporal_null(sync_paper: bool) -> None:
    arms = [j(f"result/s2_temporal_null/{a}.json") for a in ["static", "no_memory", "full", "filter"]]
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    x = np.arange(len(arms))
    F = [a["F"] for a in arms]
    worst = [a["worstF"] for a in arms]
    colors = [COL["blue"], COL["muted"], COL["purple"], COL["teal"]]
    ax.bar(x, F, color=colors, width=0.62, ec="white", lw=0.5, label="mean F")
    ax.scatter(x, worst, color=COL["red"], marker="_", s=80, label="worst-frame F")
    ax.axhline(arms[0]["heur_channel_F"], color=COL["amber"], ls="--", lw=0.8, label="channel heuristic")
    ax.set_xticks(x)
    ax.set_xticklabels([short_profile(a["arm"]) for a in arms], rotation=20, ha="right")
    ax.set_ylabel("failure F")
    ax.set_title("Temporal arms stay inside the variance envelope")
    ax.legend(loc="upper left", fontsize=5.2)
    clean(ax)
    save(fig, "F4.2_temporal_null", sync_paper)


def figure_fair_heuristics(sync_paper: bool) -> None:
    s = j("result/s2_temporal_null/static.json")
    vals = [s["F"], s["heur_channel_F"], s["heur_carried_F"]]
    labels = ["GNN\nlearned", "channel-rank\nheuristic", "carried-rel.\nheuristic"]
    fig, ax = plt.subplots(figsize=(3.15, 2.0))
    bars = ax.bar(range(3), vals, color=[COL["blue"], COL["orange"], COL["amber"]], ec="white", lw=0.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_ylabel("held-out failure F")
    ax.set_title("Fair-stream heuristic comparison")
    clean(ax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}", ha="center", fontsize=5.5)
    ax.annotate("-62% vs carried", xy=(0, vals[0]), xytext=(0.25, vals[2]*0.7),
                arrowprops=dict(arrowstyle="->", lw=0.8, color=COL["green"]), fontsize=5.5, color=COL["green"])
    save(fig, "F4.3_fair_heuristics", sync_paper)


def figure_w0_single(sync_paper: bool) -> None:
    rows = pareto_rows("production_report_operating_point")
    w0 = rows[0]
    w5 = next(r for r in rows if float(r["w_cost"]) == 5.0)
    fig, axes = plt.subplots(1, 2, figsize=(3.35, 2.0))
    ax = axes[0]
    ax.scatter(w0["holdout_F"]["mean"], w0["holdout_D"]["mean"], color=COL["red"], s=34, label="w=0")
    ax.scatter(w5["holdout_F"]["mean"], w5["holdout_D"]["mean"], color=COL["blue"], s=34, label="w=5")
    ax.plot([w0["holdout_F"]["mean"], w5["holdout_F"]["mean"]], [w0["holdout_D"]["mean"], w5["holdout_D"]["mean"]], color=COL["grid"], lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("failure F")
    ax.set_ylabel("delay D")
    ax.set_title("single-init Pareto corner")
    ax.legend(loc="lower left", fontsize=5.3)
    clean(ax)
    panel(ax, "a")
    ax = axes[1]
    vals = [w0["holdout_D"]["mean"] / w5["holdout_D"]["mean"], w0["holdout_E"]["mean"] / w5["holdout_E"]["mean"]]
    bars = ax.bar([0, 1], vals, color=[COL["amber"], COL["orange"]], width=0.62)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["D", "E"])
    ax.set_ylabel("w=0 / w=5")
    ax.set_title("cost blow-up")
    clean(ax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v+1.0, f"{v:.0f}x", ha="center", fontsize=5.5)
    panel(ax, "b")
    save(fig, "F1.5_w0_dominated", sync_paper)


def figure_emission_phase(sync_paper: bool) -> None:
    data = j("result/emission_probe_paperenv/emission_probe.json")
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    for arm, color in [("full", COL["red"]), ("filter", COL["teal"])]:
        for r in [r for r in data["runs"] if r["arm"] == arm]:
            x = [e["joined_std"] for e in r["trajectory"]]
            y = [e["gate_grad_norm"] for e in r["trajectory"]]
            ax.plot(x, y, "-o", ms=2.0, lw=0.65, alpha=0.55, color=color)
        ax.plot([], [], color=color, label=short_profile(arm))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("recurrent state/input scale")
    ax.set_ylabel("gate gradient norm")
    ax.set_title("Emission prevents high-scale gradient collapse")
    ax.legend(loc="upper right", fontsize=5.3)
    clean(ax)
    save(fig, "F5.6_emission_phase_portrait", sync_paper)


def build(sync_paper: bool) -> None:
    configure_style()
    figure_protocol_floor(sync_paper)
    figure_two_config(sync_paper)
    figure_pareto_swarm(sync_paper)
    figure_pareto_trajectory(sync_paper)
    figure_transfer_regret(sync_paper)
    figure_w0_single(sync_paper)
    figure_advantage_bubble(sync_paper)
    figure_mc_survival(sync_paper)
    figure_emission_radar(sync_paper)
    figure_sigma_init(sync_paper)
    figure_probe_divergence(sync_paper)
    figure_outcome_range(sync_paper)
    figure_replication(sync_paper)
    figure_emission_phase(sync_paper)
    figure_gating_map(sync_paper)
    figure_routing_validation(sync_paper)
    figure_context_estimation(sync_paper)
    figure_gating_alluvial(sync_paper)
    figure_retention_governance(sync_paper)
    figure_offgrid(sync_paper)
    figure_currency(sync_paper)
    figure_w0_dumbbell(sync_paper)
    figure_temporal_null(sync_paper)
    figure_fair_heuristics(sync_paper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync-paper", action="store_true", help="copy regenerated assets into paper/figures")
    args = parser.parse_args()
    build(sync_paper=bool(args.sync_paper))


if __name__ == "__main__":
    main()
