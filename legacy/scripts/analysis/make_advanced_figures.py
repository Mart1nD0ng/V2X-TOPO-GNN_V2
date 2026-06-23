"""Advanced / high-information-density paper figures (IEEE style):

  F5.1  physical_topology_hero   — the deployed optimal topology over the urban scene: vehicles coloured
                                   by per-node reliability F, selected links coloured by link-success and
                                   weighted by query weight, faint candidate skeleton beneath, a zoom inset
                                   of the densest cluster, scale bar.
  F5.2  logical_topology_graph   — the same topology as an abstract directed graph (Kamada-Kawai layout),
                                   node size ~ out-degree, colour = F, edges curved/weighted, community
                                   colouring, with a degree-distribution + structure-stats inset.
  F5.3  ablation_radar           — the emission 2×2 ablation as a radar/spider over {stability, mean acc.,
                                   worst-case, best-case, consistency}; +emission arms dominate the area.

Physical/logical re-deploy the trained production model (forward only, no training). Radar is JSON-only.

Usage:
  python -B scripts/analysis/make_advanced_figures.py [--only F5.1,F5.3] [--out result/paper_figures]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import networkx as nx  # noqa: E402

from scripts.analysis.make_paper_figures import (  # noqa: E402
    C_GNN, C_GOOD, C_BAD, C_ACC, C_ORANGE, C_GREY, COL_W, DBL_W, save, set_ieee_style,
)
from scripts.analysis.run_production_report import deploy_scene, train_main  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer  # noqa: E402
from src.training.training_smoke import _loss_config, _make_model, load_training_smoke_config  # noqa: E402

_RDYLGN_R = "RdYlGn_r"


def _load_or_train(config: str, model_pt: Path, viz_seed: int, train_n: int, max_steps: int):
    """Load the saved production model into a fresh model; retrain briefly if the state dict mismatches."""
    base = dict(load_training_smoke_config(str(ROOT / config)))
    probe = _scene_env(base, 100, viz_seed)
    model = _make_model(probe["cfg"])
    if model_pt.exists():
        try:
            model.load_state_dict(torch.load(model_pt, map_location="cpu", weights_only=False))
            print(f"  loaded {model_pt.name}", flush=True)
            return base, model
        except Exception as e:  # noqa: BLE001
            print(f"  state-dict load failed ({e!r:.60}); retraining briefly", flush=True)
    env = _scene_env(base, train_n, viz_seed)
    layer, caps = _topology_layer(env["cfg"], env["candidate"].num_nodes)
    loss_cfg = dict(_loss_config(env["cfg"]))
    loss_cfg["reliability_failure_target"] = 0.02
    loss_cfg["reliability_tail_failure_target"] = 0.02
    model, _ = train_main(env["cfg"], env, layer, caps, loss_cfg, max_steps, viz_seed)
    return base, model


def _segments(px, py, src, dst):
    return [[(px[int(s)], py[int(s)]), (px[int(d)], py[int(d)])] for s, d in zip(src, dst)]


def _indeg(dep):
    ind = np.bincount(np.asarray(dep["sel_dst"], dtype=int), minlength=dep["num_nodes"]).astype(float)
    return ind


# =========================================================================== F5.1
def fig_physical_hero(dep, out: Path) -> None:
    px, py = dep["px"], dep["py"]
    ind = _indeg(dep)
    fig, ax = plt.subplots(figsize=(DBL_W * 0.66, DBL_W * 0.6))
    lo, hi = float(min(px.min(), py.min())), float(max(px.max(), py.max()))
    span = hi - lo
    for g in np.linspace(lo, hi, 9):                       # faint urban-grid backdrop
        ax.axvline(g, color="#eef2f4", lw=0.5, zorder=0)
        ax.axhline(g, color="#eef2f4", lw=0.5, zorder=0)
    ax.add_collection(LineCollection(_segments(px, py, dep["cand_src"], dep["cand_dst"]),
                                     colors="#cfd8dc", linewidths=0.25, alpha=0.22, zorder=1))
    w = dep["sel_w"]; wn = w / (w.max() + 1e-12)
    segs = _segments(px, py, dep["sel_src"], dep["sel_dst"])
    ax.add_collection(LineCollection(segs, colors="#37474f", linewidths=(0.3 + 2.3 * wn), alpha=0.7, zorder=2))
    # nodes: size = in-degree (consensus hubs), colour = F (absolute scale; floor-saturated)
    norm = Normalize(vmin=0.0, vmax=0.12)
    sc = ax.scatter(px, py, s=(16 + 13 * ind), c=dep["node_F"], cmap=_RDYLGN_R, norm=norm,
                    edgecolors="#263238", linewidths=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02); cb.set_label("per-node failure F", fontsize=6.5)
    # in-degree size legend
    for k, dv in [(int(ind.min()), "min"), (int(np.median(ind)), "med"), (int(ind.max()), "max")]:
        ax.scatter([], [], s=(16 + 13 * k), c="#90a4ae", edgecolors="#263238", linewidths=0.4,
                   label=f"in-deg {k}")
    ax.legend(loc="lower right", fontsize=5.5, title="node size", title_fontsize=5.5, labelspacing=0.9, borderpad=0.6)
    # zoom inset (densest 1/4 window), top-left to avoid the colorbar
    H, xe, ye = np.histogram2d(px, py, bins=4)
    bi, bj = np.unravel_index(int(H.argmax()), H.shape)
    x0, x1, y0, y1 = xe[bi], xe[bi + 1], ye[bj], ye[bj + 1]; pad = 0.20 * (x1 - x0)
    axins = ax.inset_axes([0.015, 0.63, 0.35, 0.35])
    axins.add_collection(LineCollection(_segments(px, py, dep["cand_src"], dep["cand_dst"]),
                                        colors="#cfd8dc", linewidths=0.3, alpha=0.3))
    axins.add_collection(LineCollection(segs, colors="#37474f", linewidths=(0.6 + 3.0 * wn), alpha=0.85))
    axins.scatter(px, py, s=(28 + 18 * ind), c=dep["node_F"], cmap=_RDYLGN_R, norm=norm, edgecolors="#263238", linewidths=0.4)
    axins.set_xlim(x0 - pad, x1 + pad); axins.set_ylim(y0 - pad, y1 + pad)
    axins.set_xticks([]); axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor(C_GNN); s.set_linewidth(1.0)
    ax.indicate_inset_zoom(axins, edgecolor=C_GNN, lw=0.8, alpha=0.8)
    sb = 100.0                                              # scale bar
    ax.plot([lo + 0.05 * span, lo + 0.05 * span + sb], [lo + 0.04 * span] * 2, "-", color="#263238", lw=2)
    ax.text(lo + 0.05 * span + sb / 2, lo + 0.055 * span, "100 m", ha="center", fontsize=6, color="#263238")
    m = dep["metrics"]
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Deployed optimal topology over the urban scene (N={dep['num_nodes']})\n"
                 f"{len(dep['sel_src'])} links / {len(dep['cand_src'])} candidates · F={m['F']:.3f} "
                 f"(node size ~ in-degree; F uniformly near the protocol floor)", fontsize=7)
    save(fig, out, "F5.1", "physical_topology_hero")


# =========================================================================== F5.2
def fig_logical_hero(dep, out: Path) -> None:
    n = dep["num_nodes"]
    G = nx.DiGraph(); G.add_nodes_from(range(n))
    for s, d, w in zip(dep["sel_src"], dep["sel_dst"], dep["sel_w"]):
        G.add_edge(int(s), int(d), weight=float(w))
    indeg = np.array([G.in_degree(i) for i in range(n)], dtype=float)   # in-degree VARIES (out-deg is the fixed budget)
    pos = nx.spring_layout(G, k=2.4 / np.sqrt(max(n, 1)), iterations=250, seed=7)
    # community colouring (on the undirected projection)
    try:
        comms = list(nx.algorithms.community.greedy_modularity_communities(G.to_undirected()))
        comm_of = {u: ci for ci, c in enumerate(comms) for u in c}
        n_comm = len(comms)
        palette = [C_GNN, C_GOOD, C_ORANGE, C_ACC, "#1f77b4", "#8c564b", "#17becf", "#bcbd22"]
        halo = [palette[comm_of.get(i, 0) % len(palette)] for i in range(n)]
    except Exception:  # noqa: BLE001
        halo = ["#263238"] * n; n_comm = 0
    fig, ax = plt.subplots(figsize=(DBL_W * 0.6, DBL_W * 0.56))
    norm = Normalize(vmin=0.0, vmax=0.12)
    ew = np.array([G[u][v]["weight"] for u, v in G.edges()]) if G.number_of_edges() else np.array([])
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#b0bec5", alpha=0.35,
                           width=(0.2 + 1.8 * ew / (ew.max() + 1e-12)) if ew.size else 1.0,
                           arrowsize=4, arrowstyle="-|>", connectionstyle="arc3,rad=0.07")
    px_ = np.array([pos[i][0] for i in range(n)]); py_ = np.array([pos[i][1] for i in range(n)])
    ax.scatter(px_, py_, s=(70 + 95 * indeg), c=halo, alpha=0.30, edgecolors="none", zorder=2)  # community halo
    nodes = ax.scatter(px_, py_, s=(26 + 52 * indeg), c=dep["node_F"], cmap=_RDYLGN_R,
                       vmin=norm.vmin, vmax=norm.vmax, edgecolors="#263238", linewidths=0.4, zorder=3)
    cb = fig.colorbar(nodes, ax=ax, fraction=0.045, pad=0.02); cb.set_label("per-node failure F", fontsize=6.5)
    recip = nx.reciprocity(G) or 0.0
    clust = nx.average_clustering(G.to_undirected())
    stats = (f"N={n}  E={G.number_of_edges()}\nmean in-deg={indeg.mean():.2f}  max={int(indeg.max())}\n"
             f"reciprocity={recip:.2f}\nclustering={clust:.2f}\ncommunities={n_comm}")
    axin = ax.inset_axes([0.0, 0.0, 0.27, 0.2])
    axin.hist(indeg, bins=range(0, int(indeg.max()) + 2), color=C_GNN, alpha=0.85, align="left")
    axin.set_title("in-degree dist.", fontsize=5.5); axin.tick_params(labelsize=5)
    ax.text(0.0, 1.0, stats, transform=ax.transAxes, fontsize=6, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GREY, lw=0.5, alpha=0.85))
    ax.set_title("Optimal topology as a logical graph\n(node size ~ in-degree = consensus hubs; halo = community; colour = F)", fontsize=7)
    ax.axis("off")
    save(fig, out, "F5.2", "logical_topology_graph")


# =========================================================================== F5.3
def fig_ablation_radar(out: Path) -> None:
    dec = json.loads((ROOT / "result/emission_2x2_isolation/summary.json").read_text(encoding="utf-8"))["decomposition"]
    # the clean 2x2: emission {off,on} x graph {on,off}. (filter_nomem recurrence-control omitted — tracks +emission.)
    arms = [a for a in ["full", "no_graph", "filter", "no_graph_filter"] if a in dec]
    label = {"full": "−emission, +graph", "no_graph": "−emission, −graph",
             "filter": "+emission, +graph", "no_graph_filter": "+emission, −graph"}
    col = {"full": C_BAD, "no_graph": C_ORANGE, "filter": C_GOOD, "no_graph_filter": C_GNN}
    style = {"full": "-", "no_graph": "--", "filter": "-", "no_graph_filter": "--"}
    # all axes "higher = better", per-axis min-max normalised across the 4 arms
    raw = {
        "stability\n(−σ_init)": np.array([-dec[a]["sigma_init"] for a in arms]),
        "accuracy\n(−mean F)": np.array([-dec[a]["F_mean"] for a in arms]),
        "worst-case\n(−max F)": np.array([-dec[a]["F_max"] for a in arms]),
        "best-case\n(−min F)": np.array([-dec[a]["F_min"] for a in arms]),
        "consistency\n(−init range)": np.array([-dec[a]["init_range"] for a in arms]),
    }
    axes_labels = list(raw.keys())
    M = np.zeros((len(arms), len(axes_labels)))
    for j, k in enumerate(axes_labels):
        v = raw[k]; lo, hi = v.min(), v.max(); M[:, j] = (v - lo) / (hi - lo + 1e-12)
    ang = np.linspace(0, 2 * np.pi, len(axes_labels), endpoint=False); ang = np.concatenate([ang, ang[:1]])
    fig, ax = plt.subplots(figsize=(COL_W * 1.55, COL_W * 1.5), subplot_kw=dict(polar=True))
    for i, a in enumerate(arms):
        vals = np.concatenate([M[i], M[i, :1]])
        emit = a.startswith("filter") or a in ("filter", "no_graph_filter")
        ax.plot(ang, vals, style[a], color=col[a], lw=1.8, label=label[a], zorder=3)
        if emit:
            ax.fill(ang, vals, color=col[a], alpha=0.10, zorder=1)
    ax.set_xticks(ang[:-1]); ax.set_xticklabels(axes_labels, fontsize=6.2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels([]); ax.set_ylim(0, 1.08)
    ax.set_rlabel_position(0)
    ax.set_title("Emission 2×2 ablation — +emission fills every axis,\n−emission collapses to the centre (outer = better)", fontsize=7.5, pad=16)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, fontsize=6, columnspacing=1.2)
    save(fig, out, "F5.3", "ablation_radar")


# =========================================================================== F5.4 (alt to F4.1)
def fig_currency_slopegraph(out: Path) -> None:
    """Currency faithfulness as a slopegraph: mean-field -> quenched -> MC per cell (log F)."""
    cells = json.loads((ROOT / "result/currency_faithfulness/currency_faithfulness.json").read_text(encoding="utf-8"))["cells"]
    cols = [C_GNN, C_ORANGE, C_ACC]
    fig, ax = plt.subplots(figsize=(COL_W * 1.25, 2.7))
    xs = [0, 1, 2]
    for i, c in enumerate(cells):
        ys = [c["F_meanfield"], c["F_quenched"], c["F_mc"]]
        ax.plot(xs, ys, "-o", color=cols[i % 3], lw=1.3, ms=5,
                label=f"d{int(c['density'])}/{c['profile'][:4]}/c{int(c['coupling_db'])}")
        ax.annotate(f"{c['meanfield_optimism_x']:.0f}×", (0, ys[0]), textcoords="offset points",
                    xytext=(-4, 0), ha="right", fontsize=6, color=cols[i % 3])
        ax.annotate(f"{c['quenched_fidelity_x']:.1f}×", (2, ys[2]), textcoords="offset points",
                    xytext=(5, 0), ha="left", fontsize=6, color=cols[i % 3])
    ax.set_yscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels(["mean-field\n(Q=1)", "quenched\n(Q=21)", "Monte-Carlo\n(truth)"], fontsize=7)
    ax.set_ylabel("failure F (log)"); ax.set_xlim(-0.45, 2.7)
    ax.set_title("Currency faithfulness: optimism climbs left→right\n(mean-field 16–99× under truth; quenched 1.3–4.4×)", fontsize=7.5)
    ax.legend(fontsize=5.8, loc="lower right")
    save(fig, out, "F5.4", "currency_slopegraph")


# =========================================================================== F5.5 (alt to F4.5)
def fig_w0_dumbbell(out: Path) -> None:
    """w0 vs w5 per-seed dumbbell — convergence of the optimized arm vs the scattered rel-only arm."""
    s = json.loads((ROOT / "result/w0_seed_band/w0_seed_band.json").read_text(encoding="utf-8"))
    arms = s["arms"]; rel = [a for a in arms if a.startswith("w0")][0]; opt = [a for a in arms if a != rel][0]
    relp = {r["seed"]: r["F"] for r in arms[rel]["per_seed"]}; optp = {r["seed"]: r["F"] for r in arms[opt]["per_seed"]}
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(DBL_W * 0.7, 2.7), gridspec_kw={"width_ratios": [3, 1]})
    for sd in relp:
        collapse = relp[sd] > 0.4
        c = C_BAD if collapse else C_GOOD
        ax.plot([0, 1], [relp[sd], optp[sd]], "-", color=c, lw=1.0, alpha=0.7, zorder=2)
        ax.scatter([0], [relp[sd]], color=c, s=34, zorder=3)
        ax.scatter([1], [optp[sd]], color=C_GNN, s=34, zorder=3)
    ax.annotate("seed 42\n(escape)", (0, relp[42]), textcoords="offset points", xytext=(8, 0), fontsize=6, color=C_GOOD, va="center")
    ax.scatter([], [], color=C_BAD, s=34, label="w=0 collapse (4/5)"); ax.scatter([], [], color=C_GOOD, s=34, label="w=0 escape (1/5)")
    ax.scatter([], [], color=C_GNN, s=34, label="w=5 optimized")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["w=0\nrel-only", "w=5\noptimized"]); ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel("held-out failure F"); ax.legend(fontsize=5.8, loc="center right")
    ax.set_title(f"Per-seed convergence (rel-only σ={np.std(list(relp.values())):.2f} → optimized σ={np.std(list(optp.values())):.2f})", fontsize=7)
    # D/E blow-up strip
    axd.bar([0, 1], [s["D_blowup_x"], s["E_blowup_x"]], 0.55, color=[C_ACC, C_ORANGE], alpha=0.85)
    for i, v in enumerate([s["D_blowup_x"], s["E_blowup_x"]]):
        axd.annotate(f"{v:.0f}×", (i, v), textcoords="offset points", xytext=(0, 2), ha="center", fontsize=7)
    axd.axhline(1, ls="--", lw=0.7, color=C_GREY); axd.set_xticks([0, 1]); axd.set_xticklabels(["D", "E"])
    axd.set_ylabel("rel-only / opt (×)"); axd.set_title("cost blow-up", fontsize=7)
    save(fig, out, "F5.5", "w0_dumbbell")


# =========================================================================== F5.6 (alt to F2.3)
def fig_emission_phase(out: Path) -> None:
    """Emission collapse as a PHASE PORTRAIT: gate-input scale (x) vs gate-grad norm (y, log), per epoch."""
    ep = json.loads((ROOT / "result/emission_probe_collapse_regime/emission_probe.json").read_text(encoding="utf-8"))

    def avg(arm, key):
        runs = [r for r in ep["runs"] if r["arm"] == arm]
        T = min(len(r["trajectory"]) for r in runs)
        return np.array([np.mean([r["trajectory"][t][key] for r in runs]) for t in range(T)])
    fig, ax = plt.subplots(figsize=(COL_W * 1.2, 2.7))
    for arm in ep["arms"]:
        lab, col = ("+emission (bounded)", C_GOOD) if arm == "filter" else ("−emission (death spiral)", C_BAD)
        x = avg(arm, "joined_std"); y = np.clip(avg(arm, "gate_grad_norm"), 1e-4, None)
        ax.plot(x, y, "-", color=col, lw=1.2, alpha=0.8)
        ax.scatter(x, y, c=np.arange(len(x)), cmap=("Greens" if arm == "filter" else "Reds"), s=18, zorder=4, edgecolors="none")
        # direction arrows
        for t in range(0, len(x) - 1, max(1, len(x) // 5)):
            ax.annotate("", xy=(x[t + 1], y[t + 1]), xytext=(x[t], y[t]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=0.9, alpha=0.8))
        ax.scatter([x[0]], [y[0]], marker="o", s=55, facecolors="none", edgecolors=col, lw=1.3, zorder=5)
        ax.annotate(lab, (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0), fontsize=6.5, color=col, va="center")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("recurrent gate-input scale  (joined_std, log)")
    ax.set_ylabel("gate-weight grad norm (log)")
    ax.set_title("Emission collapse is a gate-gradient death spiral\n(−emission: input ↑ while gradient → 0; ○ = init)", fontsize=7.5)
    save(fig, out, "F5.6", "emission_phase_portrait")


# =========================================================================== F5.7 (alt to F1.10 + F1.12)
def fig_advantage_bubble(out: Path) -> None:
    """Advantage region as a bubble/parity map: size ~ gap, colour = class, robust=solid, MC-audited=ring."""
    amap = json.loads((ROOT / "result/advantage_map/advantage_map.json").read_text(encoding="utf-8"))["cells"]
    try:
        mc = json.loads((ROOT / "result/advantage_montecarlo/advantage_mc.json").read_text(encoding="utf-8"))["cells"]
        mc_keys = {(int(c["density"]), c["profile"], int(c["coupling_db"])): c for c in mc}
    except Exception:  # noqa: BLE001
        mc_keys = {}
    densities = sorted({c["density"] for c in amap}); couplings = sorted({c["coupling_db"] for c in amap})
    from scripts.analysis.make_paper_figures import _PROFILE_ORDER, _PROFILE_LABEL
    profiles = [p for p in _PROFILE_ORDER if any(c["profile"] == p for c in amap)]
    cls_col = {"GNN_ADVANTAGE": C_GOOD, "FLOOR_LIMITED": C_GREY, "HEURISTIC_PARITY": C_ORANGE,
               "GNN_DEFICIT": C_BAD, "TRAIN_DIVERGED": "#000000"}
    fig, axes = plt.subplots(1, len(profiles), figsize=(DBL_W, 2.7), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    gaps = [abs(c.get("gap_mean", 0.0)) for c in amap]; gmax = max(gaps) or 1.0
    for ax, pr in zip(axes, profiles):
        for c in amap:
            if c["profile"] != pr:
                continue
            j = densities.index(c["density"]); i = couplings.index(c["coupling_db"])
            gap = c.get("gap_mean", 0.0); size = 60 + 1400 * abs(gap) / gmax
            col = cls_col.get(c["cell_class"], C_GREY)
            ec = "k" if c.get("label_robust") else "none"
            ax.scatter(j, i, s=size, c=col, edgecolors=ec, linewidths=1.4, alpha=0.85, zorder=3)
            key = (int(c["density"]), c["profile"], int(c["coupling_db"]))
            if key in mc_keys:                      # MC-audited cell → bold outer ring + gap_mc tick
                ax.scatter(j, i, s=size + 230, facecolors="none", edgecolors=C_GNN, linewidths=1.6, zorder=4)
                ax.annotate(f"MC+{mc_keys[key]['gap_mc']:.02f}", (j, i), textcoords="offset points",
                            xytext=(0, -14), ha="center", fontsize=5, color=C_GNN)
            if abs(gap) > 0.02:
                ax.annotate(f"{gap:+.02f}", (j, i), textcoords="offset points", xytext=(0, 0), ha="center", va="center", fontsize=5)
        ax.set_xticks(range(len(densities))); ax.set_xticklabels([int(d) for d in densities])
        ax.set_yticks(range(len(couplings))); ax.set_yticklabels([int(k) for k in couplings])
        ax.set_title(_PROFILE_LABEL.get(pr, pr), fontsize=7)
        ax.set_xlim(-0.6, len(densities) - 0.4); ax.set_ylim(-0.6, len(couplings) - 0.4); ax.grid(alpha=0.25)
    axes[0].set_ylabel("interference coupling (dB)")
    fig.supxlabel("density (veh/km$^2$)", fontsize=8, y=0.04)
    handles = [plt.Line2D([], [], marker="o", ls="", color=cls_col[k], label=k.replace("_", " ").title())
               for k in ("GNN_ADVANTAGE", "FLOOR_LIMITED", "HEURISTIC_PARITY") if any(c["cell_class"] == k for c in amap)]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="seed-robust (solid edge)"),
                plt.Line2D([], [], marker="o", ls="", mfc="none", mec=C_GNN, label="MC-audited (ring)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=5.6, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Advantage region — bubble ~ |gap|, colour = class; 3 cells MC-audited (rings)", y=1.02, fontsize=8.5, fontweight="bold")
    save(fig, out, "F5.7", "advantage_bubble_map")


# =========================================================================== F5.1 (topology panel: physical + logical across N)
def fig_topology_panel(model, base, scales, seed, eval_q, out: Path) -> None:
    """2-row panel: physical deployed topology (row 1) + matching logical graph (row 2) across node counts.
    Pure topology — uniform node size/colour, no communication encoding."""
    NODEC, EDGEC = "#1f4e79", "#5a6b7a"
    ncol = len(scales)
    fig, axes = plt.subplots(2, ncol, figsize=(2.55 * ncol, 5.4))
    if ncol == 1:
        axes = axes.reshape(2, 1)
    for j, N in enumerate(scales):
        dep = deploy_scene(model, base, N, seed, eval_q)
        px, py = dep["px"], dep["py"]
        # ---- row 0: physical scene (real x,y) ----
        axp = axes[0][j]
        axp.add_collection(LineCollection(_segments(px, py, dep["sel_src"], dep["sel_dst"]),
                                          colors=EDGEC, linewidths=0.45, alpha=0.7, zorder=1))
        axp.scatter(px, py, s=max(4, 9 - N // 80), c=NODEC, edgecolors="white", linewidths=0.2, zorder=2)
        axp.set_aspect("equal"); axp.set_xticks([]); axp.set_yticks([])
        for s in axp.spines.values():
            s.set_edgecolor("#cccccc")
        axp.set_title(f"N = {N}   ({len(dep['sel_src'])} links)", fontsize=8)
        # ---- row 1: logical graph (force layout) ----
        axl = axes[1][j]
        G = nx.DiGraph(); G.add_nodes_from(range(dep["num_nodes"]))
        for s, d in zip(dep["sel_src"], dep["sel_dst"]):
            G.add_edge(int(s), int(d))
        pos = nx.spring_layout(G, k=2.6 / np.sqrt(max(dep["num_nodes"], 1)), iterations=200, seed=7)
        xs = [pos[i][0] for i in range(dep["num_nodes"])]; ys = [pos[i][1] for i in range(dep["num_nodes"])]
        nx.draw_networkx_edges(G, pos, ax=axl, edge_color=EDGEC, width=0.25, alpha=0.30, arrows=False)
        axl.scatter(xs, ys, s=max(3, 8 - N // 70), c=NODEC, edgecolors="none", zorder=3)
        axl.axis("off"); axl.set_aspect("equal")
        recip = nx.reciprocity(G) or 0.0
        clust = nx.average_clustering(G.to_undirected())
        axl.set_title(f"reciprocity {recip:.2f} · clustering {clust:.2f}", fontsize=6.5)
    fig.suptitle("Deployed topology across node-count scenarios (label-free; one model, one .backward())",
                 y=0.995, fontsize=9.5, fontweight="bold")
    fig.text(0.006, 0.71, "physical scene", rotation=90, va="center", ha="left", fontsize=8.5, fontweight="bold")
    fig.text(0.006, 0.28, "logical topology", rotation=90, va="center", ha="left", fontsize=8.5, fontweight="bold")
    fig.subplots_adjust(left=0.045, right=0.99, top=0.92, bottom=0.03, hspace=0.20, wspace=0.10)
    save(fig, out, "F5.1", "topology_panel")


# =========================================================================== F5.8 (Pareto + per-scene swarm)
def fig_pareto_swarm(out: Path) -> None:
    """Coupled cost-reliability front with per-scene swarm (operating_point live front)."""
    from scripts.analysis.make_paper_figures import _load
    rows = _load("production_report_operating_point/report.json")["pareto"]["rows"]
    ws = [r["w_cost"] for r in rows]; cmap = plt.get_cmap("viridis")
    def cw(w): return cmap((w - min(ws)) / (max(ws) - min(ws) + 1e-9))
    pos_rows = [r for r in rows if r["w_cost"] > 0]                # the actual front lives at w>0
    w0 = next((r for r in rows if r["w_cost"] == 0), None)
    off = [(-2, -8), (8, 8), (8, -10), (-2, 10), (10, 0)]         # de-clutter label offsets
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.9))
    for ax, key, lab in ((axes[0], "D", "delay D (eff-rounds)"), (axes[1], "E", "energy E (J)")):
        allx, ally = [], []
        for i, r in enumerate(pos_rows):
            w = r["w_cost"]
            xs = [m["F"] for m in r["holdout_per_scene"]]; ys = [m[key] for m in r["holdout_per_scene"]]
            ax.scatter(xs, ys, s=10, color=cw(w), alpha=0.30, zorder=2, edgecolors="none")   # per-scene swarm
            allx += xs; ally += ys
            mx = r["holdout_F"]["mean"]; my = r["holdout_" + key]["mean"]
            ax.errorbar(mx, my, xerr=r["holdout_F"]["ci_halfwidth"], yerr=r["holdout_" + key]["ci_halfwidth"],
                        fmt="o", ms=6, color=cw(w), ecolor="gray", elinewidth=0.9, capsize=3, zorder=4)
            ax.annotate(f"w={w:g}", (mx, my), textcoords="offset points", xytext=off[i % len(off)], fontsize=6, color="#222")
        pts = sorted((r["holdout_F"]["mean"], r["holdout_" + key]["mean"]) for r in pos_rows)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "--", color="#888", lw=0.9, zorder=1)   # front
        # zoom to the w>0 cluster; w=0 is off-scale (dominated) — noted, not plotted
        xpad = 0.012; ypad = 0.12 * (max(ally) - min(ally))
        ax.set_xlim(min(allx) - xpad, max(allx) + 3.2 * xpad); ax.set_ylim(min(ally) - ypad, max(ally) + ypad)
        if w0:
            ax.annotate(f"w=0 rel-only off-scale\n(F={w0['holdout_F']['mean']:.2f}, {key}="
                        f"{w0['holdout_'+key]['mean']:.0f}; dominated — see F5.5)",
                        (0.97, 0.96), xycoords="axes fraction", ha="right", va="top", fontsize=5.6, color=C_BAD)
        ax.set_xlabel("failure F"); ax.set_ylabel(lab); ax.set_title(f"F vs {key}  (w>0 front)")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(min(w for w in ws if w > 0), max(ws)))
    fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02).set_label("cost weight w", fontsize=6.5)
    fig.suptitle("Coupled cost–reliability front with per-scene swarm (operating point, 6 held-out scenes)",
                 y=1.02, fontsize=8.5, fontweight="bold")
    save(fig, out, "F5.8", "pareto_swarm")


# =========================================================================== F5.9 (retention + cosine web)
def fig_retention_cosine(out: Path) -> None:
    """Governed-generalist recovery (dumbbell naive->governed vs LOCO ceiling) + the gradient cosine web."""
    g = json.loads((ROOT / "result/mixture_governed_gradnorm/envelope_governed.json").read_text(encoding="utf-8"))
    comp = g["comparison_by_density"]; norms = g["diagnostic"]["group_grad_norms"]; cos = g["diagnostic"]["pairwise_cosines"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DBL_W, 3.0))
    # --- left: recovery dumbbell ---
    dens = sorted(comp.keys()); y = np.arange(len(dens))[::-1]
    for yi, d in zip(y, dens):
        c = comp[d]
        axL.plot([c["naive"], c["governed"]], [yi, yi], "-", color=C_GREY, lw=1.0, zorder=1)
        axL.scatter([c["naive"]], [yi], color=C_ORANGE, s=44, zorder=3, label="naive" if yi == y[0] else None)
        axL.scatter([c["governed"]], [yi], color=C_GNN, s=44, zorder=3, label="governed" if yi == y[0] else None)
        axL.scatter([c["loco_ceiling"]], [yi], marker="|", s=260, color=C_GOOD, lw=1.8, zorder=4,
                    label="LOCO ceiling" if yi == y[0] else None)
        axL.annotate(f"{c['naive']:.2f}→{c['governed']:.2f}", (c["governed"], yi), textcoords="offset points",
                     xytext=(6, 6), fontsize=6, color=C_GNN)
    axL.axvline(1.0, ls="--", lw=0.7, color=C_GREY)
    axL.set_yticks(y); axL.set_yticklabels([d.replace("d", "density ") for d in dens])
    axL.set_xlabel("in-grid retention"); axL.set_xlim(0.2, 1.15)
    axL.set_title("Governance recovers the ceiling\n(parity-recovery, not superiority)", fontsize=7.5)
    axL.legend(fontsize=6, loc="lower right")
    # --- right: gradient cosine web ---
    pos = {"100.0": (0.18, 0.85), "200.0": (0.85, 0.85), "300.0": (0.5, 0.12)}
    nv = {k: float(v) for k, v in norms.items()}
    nmax = max(nv.values())
    for pair, cv in cos.items():
        a, b = pair.split("|")
        (x0, y0), (x1, y1) = pos[a], pos[b]
        lw = 0.6 + 5.0 * abs(cv)
        axR.plot([x0, x1], [y0, y1], "-", color=(C_GOOD if cv > 0 else C_BAD), lw=lw, alpha=0.55, zorder=1)
        axR.annotate(f"cos={cv:.2f}", ((x0 + x1) / 2, (y0 + y1) / 2), fontsize=6.2, ha="center",
                     color="#333", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    for k, (x, y0) in pos.items():
        axR.scatter([x], [y0], s=300 + 900 * nv[k] / nmax, color=C_GNN, alpha=0.85, edgecolors="#263238", zorder=3)
        axR.annotate(f"d{int(float(k))}\n|g|={nv[k]:.0f}", (x, y0), ha="center", va="center", fontsize=6, color="white", zorder=4)
    axR.set_xlim(-0.05, 1.05); axR.set_ylim(-0.05, 1.0); axR.axis("off")
    axR.set_title("Per-density gradient web\n(all cos>0 => no directional conflict; node ~ |grad|, 2.1× imbalance => GradNorm)", fontsize=7)
    fig.suptitle("Why a governed mixture works: recovery (left) explained by the gradient geometry (right)",
                 y=1.03, fontsize=8.5, fontweight="bold")
    save(fig, out, "F5.9", "retention_cosine_web")


# =========================================================================== F5.10 (gating alluvial / Sankey)
def _ribbon(ax, xl, xr, ylt, ylb, yrt, yrb, color, alpha=0.55):
    from matplotlib.path import Path as MPath
    from matplotlib.patches import PathPatch
    xm = (xl + xr) / 2
    verts = [(xl, ylt), (xm, ylt), (xm, yrt), (xr, yrt), (xr, yrb), (xm, yrb), (xm, ylb), (xl, ylb), (xl, ylt)]
    codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
             MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=2))


def fig_gating_sankey(out: Path) -> None:
    """3-stage alluvial: density -> coupling -> deployment decision (gating table)."""
    tab = json.loads((ROOT / "result/gating_demo/gating_table.json").read_text(encoding="utf-8"))
    ent = [(int(e["density"]), int(e["coupling_db"]), e["decision"]) for e in tab["entries"]]
    dec_col = {"USE_GNN": C_GOOD, "USE_GNN_MARGINAL": C_GNN, "GNN_DEFAULT": C_GREY, "HEURISTIC_OK": C_ORANGE}
    dec_lab = {"USE_GNN": "USE_GNN", "USE_GNN_MARGINAL": "GNN marginal", "GNN_DEFAULT": "GNN default", "HEURISTIC_OK": "heuristic OK"}
    cols = {"density": sorted({e[0] for e in ent}), "coupling": sorted({e[1] for e in ent}),
            "decision": [d for d in ["USE_GNN", "USE_GNN_MARGINAL", "GNN_DEFAULT", "HEURISTIC_OK"] if any(e[2] == d for e in ent)]}
    xpos = {"density": 0.0, "coupling": 1.0, "decision": 2.0}
    keyfn = {"density": lambda e: e[0], "coupling": lambda e: e[1], "decision": lambda e: e[2]}
    order = sorted(ent)                                              # global ribbon order
    GAP, NW = 0.5, 0.12
    # node y-extents per column (stack downward, centred)
    ext = {}
    for col, nodes in cols.items():
        counts = {nd: sum(1 for e in ent if keyfn[col](e) == nd) for nd in nodes}
        total = sum(counts.values()) + GAP * (len(nodes) - 1); y = total / 2
        for nd in nodes:
            ext[(col, nd)] = [y, y - counts[nd]]; y -= counts[nd] + GAP
    # per-ribbon slice in each node (consistent across both segments), stacked top-down in global order
    slot = {}; offset = {k: ext[k][0] for k in ext}
    for col in cols:
        for e in order:
            key = (col, keyfn[col](e)); top = offset[key]; offset[key] = top - 1.0
            slot[(col, order.index(e))] = (top, top - 1.0)
    fig, ax = plt.subplots(figsize=(DBL_W * 0.82, 3.2))
    # nodes
    from matplotlib.patches import Rectangle
    for (col, nd), (yt, yb) in ext.items():
        ax.add_patch(Rectangle((xpos[col] - NW / 2, yb), NW, yt - yb, facecolor="#455a64", edgecolor="none", zorder=3))
        lab = (f"{nd} veh/km²" if col == "density" else f"{nd} dB" if col == "coupling" else dec_lab.get(nd, nd))
        ha = "right" if col == "density" else ("left" if col == "decision" else "center")
        dx = -0.09 if col == "density" else (0.09 if col == "decision" else 0)
        ax.text(xpos[col] + dx, (yt + yb) / 2, lab, ha=ha, va="center", fontsize=6.5,
                color=(dec_col.get(nd, "#333") if col == "decision" else "#333"))
    # ribbons (density->coupling, coupling->decision)
    for i, e in enumerate(order):
        col_c = dec_col[e[2]]
        sd = slot[("density", i)]; sc = slot[("coupling", i)]; se = slot[("decision", i)]
        _ribbon(ax, xpos["density"] + NW / 2, xpos["coupling"] - NW / 2, sd[0], sd[1], sc[0], sc[1], col_c)
        _ribbon(ax, xpos["coupling"] + NW / 2, xpos["decision"] - NW / 2, sc[0], sc[1], se[0], se[1], col_c)
    ax.set_xlim(-0.55, 2.55)
    yall = [v for pair in ext.values() for v in pair]
    ax.set_ylim(min(yall) - 0.5, max(yall) + 0.5); ax.axis("off")
    for col, lab in [("density", "density"), ("coupling", "interference"), ("decision", "gate decision")]:
        ax.text(xpos[col], max(yall) + 0.7, lab, ha="center", fontsize=7.5, fontweight="bold")
    ax.set_title("Deployment gating flow: estimable context → policy (ribbon colour = decision)", fontsize=8, y=1.04)
    save(fig, out, "F5.10", "gating_sankey")


def main() -> None:
    p = argparse.ArgumentParser(description="Advanced paper figures (topology hero + radar + de-bar-chart alternatives)")
    p.add_argument("--config", default="configs/paper_environment_v1.yaml")
    p.add_argument("--model", default="result/production_report_paperenv/model.pt")
    p.add_argument("--viz-n", type=int, default=110)
    p.add_argument("--scales", default="50,100,200,400", help="node counts for the F5.1 topology panel")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--only", default="")
    p.add_argument("--out", default="result/paper_figures")
    args = p.parse_args()
    set_ieee_style()
    out = ROOT / args.out
    want = [s.strip() for s in args.only.split(",") if s.strip()] or \
        ["F5.1", "F5.3", "F5.4", "F5.5", "F5.6", "F5.7", "F5.8", "F5.9", "F5.10"]
    json_only = {"F5.3": fig_ablation_radar, "F5.4": fig_currency_slopegraph, "F5.5": fig_w0_dumbbell,
                 "F5.6": fig_emission_phase, "F5.7": fig_advantage_bubble, "F5.8": fig_pareto_swarm,
                 "F5.9": fig_retention_cosine, "F5.10": fig_gating_sankey}

    if any(f in want for f in ("F5.1", "F5.1b", "F5.2")):     # any model-deploy figure
        base, model = _load_or_train(args.config, ROOT / args.model, args.seed, 2000, args.max_steps)
        eval_q = int(base.get("eval_quenched_quadrature", base.get("quenched_quadrature", 21)))
        if "F5.1" in want:                                     # the topology PANEL (physical row + logical row)
            scales = [int(x) for x in args.scales.split(",") if x.strip()]
            print(f"  building topology panel across N={scales}...", flush=True)
            fig_topology_panel(model, base, scales, args.seed, eval_q, out)
        if "F5.1b" in want or "F5.2" in want:                  # legacy single hero/logical (opt-in)
            dep = deploy_scene(model, base, args.viz_n, args.seed, eval_q)
            if "F5.1b" in want:
                fig_physical_hero(dep, out)
            if "F5.2" in want:
                fig_logical_hero(dep, out)
    for fid, fn in json_only.items():
        if fid in want:
            try:
                fn(out)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {fid} failed: {e!r}", flush=True)
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
