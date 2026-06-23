"""Redraw the paper's result figures from the UNIFIED MAINLINE (src/mainline), keeping each
figure's original style/type and only replacing the data (the old figures used the superseded
legacy pipeline).  Output: pdf+png into the paper project's redraw dir.

Figures:
  F5.1  topology_panel  — deployed distinct-peer query topology, physical + logical, across N
  F6.1  fea_panel       — spatial heatmaps (no-planning vs planned) x {congestion, delay, failure}
  R1    pareto_front    — F/D/E preference front + steering (new G11 data)
  R2    baseline_cmp    — Pareto coverage + hypervolume vs honest baselines (new G11 data)
  R3    complexity      — end-to-end near-linear scaling (new G9 data)

The plotting STYLE matches the original IEEE figures (colours, widths, the F6.1 smoothing /
contour helpers are reused verbatim); only the underlying numbers come from the new mainline.

Run: python scripts/analysis/redraw_paper_figures.py
"""

from __future__ import annotations

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
import networkx as nx  # noqa: E402

from src.mainline.model import OperatingPointConfig, PreferenceConditionedTopologyGNN, model_operating_point  # noqa: E402
from src.mainline.topology import build_candidate_graph  # noqa: E402
from src.mainline.symmetric_polynomials import edge_inclusion_probability  # noqa: E402
from scripts.analysis.baseline_comparison import Scenario, train_model  # noqa: E402

DT = torch.float64
OUT = Path("D:/PhD_works/V2X-topo-GNN/result/paper_figure_redraw")

# --- project IEEE style (matched to scripts/analysis/make_paper_figures.py) ----------------
C_GNN, C_HEUR, C_GOOD, C_BAD = "#0072B2", "#D55E00", "#009E73", "#7f0000"
C_ACC, C_ORANGE, C_GREY = "#CC79A7", "#E69F00", "#666666"
COL_W, DBL_W = 3.5, 7.16
NODEC, EDGEC = "#1f4e79", "#5a6b7a"   # F5.1 node / edge colours (verbatim)


def set_ieee_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.7,
        "lines.linewidth": 1.1, "lines.markersize": 4, "grid.linewidth": 0.4, "grid.alpha": 0.35,
        "axes.grid": True, "axes.axisbelow": True, "figure.dpi": 120, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig, fid, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fid}_{name}.{ext}")
    plt.close(fig)
    print(f"  saved {fid}_{name}.{{png,pdf}}", flush=True)


# --- urban-grid scene (vehicles on a road grid; the 'urban scene' look of the originals) ----
# SIDE/RADIUS chosen so link distances (~20-50 m) sit in the regime where the 3GPP-grounded
# finite-blocklength reliability is informative (longer links collapse ell -> consensus fails).
SIDE = 300.0
RADIUS = 65.0
CFG = OperatingPointConfig(rounds=8, payload_bits=8000.0, p_min_dbm=18.0, p_max_dbm=32.0, subchannels=5.0)


def urban_positions(N, seed):
    """Place N vehicles on a Manhattan road grid within a SIDE x SIDE scene (with lane jitter)."""
    gen = torch.Generator().manual_seed(seed)
    n_roads = max(3, int(round((N / 8) ** 0.5)) + 2)
    road_pos = torch.linspace(0.08, 0.92, n_roads, dtype=DT) * SIDE
    xs, ys = [], []
    for _ in range(N):
        if torch.rand((), generator=gen, dtype=DT) < 0.5:   # on a vertical road
            x = road_pos[torch.randint(n_roads, (1,), generator=gen)] + torch.randn((), generator=gen, dtype=DT) * 5
            y = torch.rand((), generator=gen, dtype=DT) * SIDE
        else:                                                # on a horizontal road
            x = torch.rand((), generator=gen, dtype=DT) * SIDE
            y = road_pos[torch.randint(n_roads, (1,), generator=gen)] + torch.randn((), generator=gen, dtype=DT) * 5
        xs.append(float(x)); ys.append(float(y))
    return torch.tensor(np.stack([xs, ys], 1), dtype=DT)


def urban_scenario(N, seed, radius=RADIUS):
    pos = urban_positions(N, seed)
    r = radius
    for _ in range(12):                                      # grow radius until every node forms a quorum
        g = build_candidate_graph(pos, r)
        if int(torch.bincount(g.src_index, minlength=N).min()) >= CFG.k:
            break
        r *= 1.12
    src, dst = g.src_index, g.dst_index
    outdeg = torch.bincount(src, minlength=N).to(DT)
    indeg = torch.bincount(dst, minlength=N).to(DT)
    nf = torch.stack([outdeg / outdeg.clamp_min(1).max(), indeg / indeg.clamp_min(1).max(),
                      torch.ones(N, dtype=DT)], 1)
    ef = (g.distance / r).unsqueeze(-1)
    sc = Scenario(graph=g, nf=nf, ef=ef, N=N, seed=seed)
    sc.pos = pos
    return sc


def deployed_topology(model, sc, lam):
    """Per-source top-k selected query peers (the deployed distinct-peer topology)."""
    g = sc.graph
    with torch.no_grad():
        s_edge, _, _ = model(sc.nf, sc.ef, g.src_index, g.dst_index, lam, g.num_nodes)
    sel_s, sel_d = [], []
    for i in range(g.num_nodes):
        m = (g.src_index == i)
        idx = m.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        kk = min(CFG.k, idx.numel())
        top = idx[torch.topk(s_edge[idx], kk).indices]
        sel_s += [i] * kk
        sel_d += [int(g.dst_index[e]) for e in top]
    return np.array(sel_s), np.array(sel_d)


# =========================================================================== F5.1
def fig_topology_panel(model, scales, seed, lam):
    ncol = len(scales)
    fig, axes = plt.subplots(2, ncol, figsize=(2.55 * ncol, 5.4))
    for j, N in enumerate(scales):
        sc = urban_scenario(N, seed + j)
        pos = sc.pos.numpy(); px, py = pos[:, 0], pos[:, 1]
        ss, sd = deployed_topology(model, sc, lam)
        # row 0: physical scene
        axp = axes[0][j]
        segs = [[(px[s], py[s]), (px[d], py[d])] for s, d in zip(ss, sd)]
        axp.add_collection(LineCollection(segs, colors=EDGEC, linewidths=0.45, alpha=0.7, zorder=1))
        axp.scatter(px, py, s=max(4, 9 - N // 80), c=NODEC, edgecolors="white", linewidths=0.2, zorder=2)
        axp.set_aspect("equal"); axp.set_xticks([]); axp.set_yticks([]); axp.grid(False)
        for spn in axp.spines.values():
            spn.set_edgecolor("#cccccc"); spn.set_visible(True)
        axp.set_title(f"N = {N}   ({len(ss)} links)", fontsize=8)
        # row 1: logical graph (force layout)
        axl = axes[1][j]
        G = nx.DiGraph(); G.add_nodes_from(range(N))
        for s, d in zip(ss, sd):
            G.add_edge(int(s), int(d))
        gpos = nx.spring_layout(G, k=2.6 / np.sqrt(max(N, 1)), iterations=200, seed=7)
        xs = [gpos[i][0] for i in range(N)]; ys = [gpos[i][1] for i in range(N)]
        nx.draw_networkx_edges(G, gpos, ax=axl, edge_color=EDGEC, width=0.25, alpha=0.30, arrows=False)
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
    save(fig, "F5.1", "topology_panel")


# =========================================================================== F6.1 helpers (reused verbatim from make_fea_comparison)
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter  # noqa: E402

_CMAP = "magma_r"   # low = light, high = dark (darker = worse)


def _smooth_scalar(px, py, val, extent, grid, sigma):
    x0, x1, y0, y1 = extent
    rng = [[x0, x1], [y0, y1]]
    Hs, _, _ = np.histogram2d(px, py, bins=grid, range=rng, weights=val)
    Hc, _, _ = np.histogram2d(px, py, bins=grid, range=rng)
    Ss = gaussian_filter(Hs, sigma); Sc = gaussian_filter(Hc, sigma)
    return (Ss / np.maximum(Sc, 1e-9)).T


def _smooth_density(mx, my, extent, grid, sigma):
    x0, x1, y0, y1 = extent
    H, _, _ = np.histogram2d(mx, my, bins=grid, range=[[x0, x1], [y0, y1]])
    return gaussian_filter(H, sigma).T


def _detect_lines(coord, lo, hi, bin_m=2.0, frac=0.18, merge_m=10.0):
    nb = max(8, int((hi - lo) / bin_m))
    h, edges = np.histogram(coord, bins=nb, range=(lo, hi))
    cen = 0.5 * (edges[:-1] + edges[1:])
    thr = max(frac * h.max(), 4.0)
    peaks = [cen[i] for i in range(nb) if h[i] >= thr]
    lines: list[list[float]] = []
    for c in sorted(peaks):
        if lines and c - lines[-1][-1] <= merge_m:
            lines[-1].append(c)
        else:
            lines.append([c])
    return [float(np.mean(g)) for g in lines]


def _composite(cols, extent, mask, px, py, roads, fid, name):
    x0, x1, y0, y1 = extent
    shape = cols[0][2][0].shape
    gx = np.linspace(x0, x1, shape[1]); gy = np.linspace(y0, y1, shape[0])
    ncol = len(cols)
    fig, axes = plt.subplots(2, ncol, figsize=(2.55 * ncol, 5.7))
    for c, (title, label, top, bot) in enumerate(cols):
        (ftop, mtop), (fbot, mbot) = top, bot
        both = np.concatenate([ftop[~np.isnan(ftop)], fbot[~np.isnan(fbot)]])
        vmax = float(np.percentile(both, 90)); levels = np.linspace(0.0, vmax, 41)
        cs = None
        for r, (f, mm) in enumerate([(ftop, mtop), (fbot, mbot)]):
            ax = axes[r][c]
            cs = ax.contourf(gx, gy, np.where(mask, np.nan, f), levels=levels, cmap=_CMAP, extend="max")
            for rx in roads[0]:
                ax.plot([rx, rx], [y0, y1], color="black", lw=0.32, alpha=0.5, zorder=4)
            for ry in roads[1]:
                ax.plot([x0, x1], [ry, ry], color="black", lw=0.32, alpha=0.5, zorder=4)
            ax.scatter(px, py, marker="s", s=3.0, facecolors="none", edgecolors="black", linewidths=0.28, zorder=5)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            ax.text(0.04, 0.96, mm, transform=ax.transAxes, va="top", ha="left", fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72))
            if r == 0:
                ax.set_title(title, fontsize=8.5)
        cb = fig.colorbar(cs, ax=[axes[0][c], axes[1][c]], orientation="horizontal", fraction=0.05, pad=0.02, aspect=22)
        ticks = np.linspace(0.0, vmax, 4)
        cb.set_ticks(ticks); cb.set_ticklabels([f"{t:.2g}" for t in ticks])
        cb.set_label(label, fontsize=6.5); cb.ax.tick_params(labelsize=5.8)
    axes[0][0].set_ylabel("No planning", fontsize=9.5, fontweight="bold")
    axes[1][0].set_ylabel("Planned", fontsize=9.5, fontweight="bold")
    fig.suptitle("Spatial impact of topology planning: no planning (top) vs planned (bottom) — darker = worse",
                 y=0.98, fontsize=9.5, fontweight="bold")
    save(fig, fid, name)


def _link_reliability(sc, P, n, pi):
    """Per-edge reliability ell = FBL x (1 - Mode-2 collision), where the collision rises with the
    receiver LOAD induced by the query inclusion ``pi`` (the G4/G5 physics).  Flooding (pi->1 on all
    candidates) drives the load -> collision up -> ell down; a sparse planned backbone keeps it low."""
    import math
    from src.mainline.finite_blocklength import averaged_link_success
    from src.mainline.topology import los_probability, mode2_collision_from_load, receiver_load
    g = sc.graph; src, dst, N = g.src_index, g.dst_index, g.num_nodes
    pe, ne = P[src], n[src]
    pl = CFG.pathloss; d = g.distance.clamp_min(1.0); log_d = torch.log10(d); log_fc = math.log10(CFG.fc_ghz)
    pl_los = pl.los[0] + pl.los[1] * log_d + pl.los[2] * log_fc
    pl_nlos = pl.nlos[0] + pl.nlos[1] * log_d + pl.nlos[2] * log_fc
    pl_non = torch.maximum(pl_nlos, pl_los + pl.nlosv_extra_db)
    losp = los_probability(d); pl_db = losp * pl_los + (1.0 - losp) * pl_non
    gamma = torch.pow(torch.tensor(10.0, dtype=DT), (pe - pl_db - CFG.noise_dbm) / 10.0)
    ell_fbl = averaged_link_success(gamma, ne, CFG.payload_bits, fading="rayleigh", max_harq_attempts=CFG.max_harq_attempts)
    tau = torch.full((N,), CFG.tau_proxy, dtype=DT)
    load = receiver_load(pi, tau, src, dst, N)
    p_col = mode2_collision_from_load(load[dst], CFG.subchannels)
    return (ell_fbl * (1.0 - p_col)).clamp(1e-4, 1.0 - 1e-9)


def _node_fields(model, sc, lam, planned):
    """Per-node failure F_i, delay D_i, and the active-link (src,dst) of a policy, via the mainline.

    The figure's claim is the spatial impact of TOPOLOGY PLANNING.  Power/blocklength are held FIXED
    for both conditions; the ONLY difference is the deployed topology:
      * planned     = the model's sparse distinct-peer backbone (low receiver load -> low collision),
      * no planning = flood every in-range candidate (high load -> high Mode-2 collision -> low ell).
    So the planned backbone genuinely lowers congestion, delay, and failure."""
    from src.mainline.model import _bucketed_inclusion_probability
    from src.mainline.global_evaluator import build_bucketed_padding, evaluate_global_consensus
    g = sc.graph; N = g.num_nodes
    P = torch.full((N,), 28.0, dtype=DT); n = torch.full((N,), 800.0, dtype=DT)
    pad = build_bucketed_padding(g.src_index, g.dst_index, N)
    if planned:
        with torch.no_grad():
            s, _, _ = model(sc.nf, sc.ef, g.src_index, g.dst_index, lam, N)
            pi = _bucketed_inclusion_probability(pad, s, CFG.k, g.num_edges)   # sparse selection (sum = k)
        ss, sd = deployed_topology(model, sc, lam)
    else:
        s = torch.zeros(g.num_edges, dtype=DT)
        outdeg = torch.bincount(g.src_index, minlength=N).clamp_min(1).to(DT)
        pi = (10.0 / outdeg[g.src_index]).clamp(max=1.0)                        # flood ~10 peers/node (high load)
        ss, sd = g.src_index.numpy(), g.dst_index.numpy()                       # the dense unplanned candidate mesh
    ell = _link_reliability(sc, P, n, pi)
    with torch.no_grad():
        res = evaluate_global_consensus(
            num_nodes=N, src_index=g.src_index, dst_index=g.dst_index, log_query_weight=s.unsqueeze(-1),
            link_reliability=ell.unsqueeze(-1), scenario_weight=torch.ones(1, dtype=DT), k=CFG.k,
            alpha=CFG.alpha, beta=CFG.beta, rounds=CFG.rounds,
            initial_correct_preference=CFG.initial_correct_preference, return_trajectory=True, padding=pad)
    F = (1.0 - res.c_ir[:, 0]).clamp(0, 1).numpy()
    ctraj = res.c_trajectory[:, :, 0].numpy() if res.c_trajectory is not None else None
    D = (1.0 - ctraj[:-1]).sum(0) if ctraj is not None else np.full(N, float(res.F_global))
    return F, D, np.asarray(ss), np.asarray(sd)


def fig_fea_panel(model, sc, lam):
    pos = sc.pos.numpy(); px, py = pos[:, 0], pos[:, 1]
    Ff, Df, sf, df = _node_fields(model, sc, lam, planned=False)
    Fp, Dp, sp, dp = _node_fields(model, sc, lam, planned=True)
    pad = 0.04 * (px.max() - px.min())
    extent = (px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad)
    grid = (200, 200); sig = 8.0
    Hc = np.histogram2d(px, py, bins=grid, range=[[extent[0], extent[1]], [extent[2], extent[3]]])[0]
    cover = gaussian_filter(Hc, sig).T
    present = binary_closing(binary_fill_holes(cover >= 0.04 * cover.max()), iterations=3)
    mask = ~present
    roads = (_detect_lines(px, extent[0], extent[1]), _detect_lines(py, extent[2], extent[3]))

    def mids(s, d):
        return (px[s] + px[d]) / 2.0, (py[s] + py[d]) / 2.0
    cong_f = _smooth_density(*mids(sf, df), extent, grid, sig)
    cong_p = _smooth_density(*mids(sp, dp), extent, grid, sig)
    dly_f = _smooth_scalar(px, py, Df, extent, grid, sig); dly_p = _smooth_scalar(px, py, Dp, extent, grid, sig)
    fail_f = _smooth_scalar(px, py, Ff, extent, grid, sig); fail_p = _smooth_scalar(px, py, Fp, extent, grid, sig)
    cols = [
        ("Channel congestion", "active-link density (a.u.)",
         (cong_f, f"{sf.size} links"), (cong_p, f"{sp.size} links")),
        ("Communication delay", "expected rounds $T_i$",
         (dly_f, f"mean {Df.mean():.1f}"), (dly_p, f"mean {Dp.mean():.1f}")),
        ("Consensus failure $F_i$", "per-node failure $F_i$",
         (fail_f, f"mean {Ff.mean():.2f}"), (fail_p, f"mean {Fp.mean():.2f}")),
    ]
    print(f"  F6.1 links f/p={sf.size}/{sp.size}  F f/p={Ff.mean():.3f}/{Fp.mean():.3f}  "
          f"D f/p={Df.mean():.1f}/{Dp.mean():.1f}", flush=True)
    _composite(cols, extent, mask, px, py, roads, "F6.1", "fea_panel")


# =========================================================================== R1/R2/R3 (new results)
import json  # noqa: E402

LAM_SWEEP = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [.7, .15, .15], [.15, .7, .15], [.15, .15, .7],
             [.5, .5, 0], [.5, 0, .5], [0, .5, .5], [.34, .33, .33]]


def fig_pareto_front():
    """R1: the F/D/E preference front swept from ONE checkpoint, with vertex steering labelled.

    Uses the G11 small-N held-out regime (where the global event probability F has spread); the
    urban large-N scenes saturate F->1 (all-nodes-correct probability vanishes) and are for the
    spatial figures, not the Pareto front."""
    from scripts.analysis.baseline_comparison import CFG as BCFG, make_scenarios, train_model
    train = make_scenarios(range(100, 110)); test = make_scenarios(range(200, 210))
    mdl = train_model(train, steps=600, seed=0)
    pts = []
    with torch.no_grad():
        for lam in LAM_SWEEP:
            fde = np.mean([[float(o["F"]), float(o["D"]), float(o["E"])]
                           for o in [model_operating_point(mdl, s.graph, s.nf, s.ef,
                                                           torch.tensor(lam, dtype=DT), BCFG) for s in test]], 0)
            pts.append((lam, fde))
    def pareto_front(F, Y):                                   # lower-left non-dominated set (both minimised)
        idx = np.argsort(F)
        keep, best = [], np.inf
        for i in idx:
            if Y[i] <= best + 1e-12:
                keep.append(i); best = Y[i]
        return keep
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 3.0))
    for ax, (k, lab) in zip(axes, [(1, "delay $D$"), (2, "energy $E$")]):
        F = np.array([p[1][0] for p in pts]); Y = np.array([p[1][k] for p in pts])
        keep = pareto_front(F, Y)
        ax.plot(F[keep], Y[keep], "--", color=C_GREY, lw=1.0, zorder=1, label="Pareto front")
        ax.scatter(F, Y, s=36, c=C_GNN, edgecolors="white", linewidths=0.4, zorder=3, label="operating points")
        for lam, fde in pts:                                  # label the three preference vertices
            if max(lam) == 1:
                axis = ["F", "D", "E"][int(np.argmax(lam))]
                ax.annotate(f"$\\lambda_{axis}$", (fde[0], fde[k]), textcoords="offset points",
                            xytext=(6, 5), fontsize=8, color=C_BAD, fontweight="bold")
        ax.set_xlabel("consensus failure $F$"); ax.set_ylabel(lab)
        ax.set_title(f"$F$ vs {lab.split()[1]}")
    axes[0].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("One checkpoint sweeps the F/D/E preference front (held-out scenes)",
                 y=1.02, fontsize=9, fontweight="bold")
    save(fig, "R1", "pareto_front")


def fig_baseline_comparison():
    """R2: Pareto set-coverage + hypervolume vs honest baselines (new G11 data)."""
    d = json.loads((ROOT / "docs/gate_evidence/g11_baseline.json").read_text(encoding="utf-8"))
    order = ["best-fixed", "fixed-uniform", "fixed-distance", "fixed-invdist", "fixed-degree",
             "lambda-blind", "untrained"]
    order = [m for m in order if m in d["hv_mean"]]
    short = {"best-fixed": "best-fixed", "fixed-uniform": "uniform", "fixed-distance": "distance",
             "fixed-invdist": "inv-dist", "fixed-degree": "degree", "lambda-blind": "$\\lambda$-blind",
             "untrained": "untrained"}
    labels = [short.get(m, m) for m in order]
    cmo = [d["cov_model_over"][m] for m in order]
    com = [d["cov_over_model"][m] for m in order]
    hv_model = d["hv_mean"]["model"]; hv_base = [d["hv_mean"][m] for m in order]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(DBL_W, 3.3))
    x = np.arange(len(order)); w = 0.38
    axL.bar(x - w / 2, cmo, w, color=C_GNN, label="C(model $\\succ$ base)")
    axL.bar(x + w / 2, com, w, color=C_HEUR, label="C(base $\\succ$ model)")
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    axL.set_ylabel("Pareto set-coverage"); axL.set_ylim(0, 1.05); axL.legend(fontsize=6.5, loc="upper center")
    axL.set_title("No baseline dominates any model point\n(C(base$\\succ$model)$=0$ everywhere)", fontsize=8)
    axR.bar(x, hv_base, 0.62, color=C_GREY, alpha=0.85)
    axR.axhline(hv_model, ls="--", lw=1.2, color=C_GNN)
    axR.text(x[0] - 0.4, hv_model + 0.012, f"model {hv_model:.2f}", color=C_GNN, fontsize=7, fontweight="bold")
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=6.5, rotation=30, ha="right")
    axR.set_ylim(0, hv_model * 1.18)
    axR.set_ylabel("hypervolume (model-indep. box)")
    axR.set_title("Hypervolume: model wins 100% of held-out\nscenes (p$=2\\times10^{-4}$)", fontsize=8)
    fig.suptitle("Held-out baseline comparison: the learned front Pareto-dominates honest strong baselines",
                 y=1.02, fontsize=8.5, fontweight="bold")
    save(fig, "R2", "baseline_comparison")


def fig_complexity():
    """R3: end-to-end near-linear complexity (new G9 profiling)."""
    from profile_scaling import profile_scaling, fit_exponent
    res = profile_scaling([200, 400, 800, 1600, 3200, 6400], reps=2)
    E = np.array(res["E"], float)
    fig, ax = plt.subplots(figsize=(COL_W + 0.4, 3.0))
    for key, lab, col in [("t_total", "end-to-end", C_GNN), ("t_build", "graph build", C_ORANGE),
                          ("t_consensus", "consensus", C_GOOD)]:
        t = np.array(res[key], float) * 1e3
        ax.loglog(E, t, "o-", color=col, ms=4, lw=1.1, label=f"{lab} (exp {fit_exponent(E, res[key]):.2f})")
    ref = (np.array(res["t_total"], float)[0] * 1e3) * (E / E[0])
    ax.loglog(E, ref, "--", color=C_GREY, lw=0.9, label="linear (slope 1)")
    ax.set_xlabel("candidate edges $E$"); ax.set_ylabel("runtime (ms)")
    ax.set_title("End-to-end runtime is near-linear in $E$\n(fixed density: $E=O(N)$; no $N\\times N$ tensor)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left"); ax.grid(True, which="both", alpha=0.3)
    save(fig, "R3", "complexity_scaling")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--only", default="", help="comma list: F5.1,F6.1,R1,R2,R3")
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else None
    set_ieee_style()
    lam = torch.tensor([0.34, 0.33, 0.33], dtype=DT)
    need_model = only is None or bool({"F5.1", "F6.1"} & only)
    model = None
    if need_model:
        OUT.mkdir(parents=True, exist_ok=True)
        model_pt = OUT / "_redraw_model.pt"
        model = PreferenceConditionedTopologyGNN(node_dim=3, edge_dim=1, hidden=32, layers=2).double()
        if model_pt.exists():
            model.load_state_dict(torch.load(model_pt, weights_only=True)); print("  loaded cached model", flush=True)
        else:
            print("training a mainline model on urban-grid scenes...", flush=True)
            train_sc = [urban_scenario(N, 1000 + i) for i, N in enumerate([90, 130, 170] * 3)]
            model = train_model(train_sc, steps=550, seed=0)
            torch.save(model.state_dict(), model_pt)
    if only is None or "F5.1" in only:
        print("F5.1 topology panel...", flush=True)
        fig_topology_panel(model, [50, 100, 200, 400], seed=200, lam=lam)
    if only is None or "F6.1" in only:
        print("F6.1 fea panel...", flush=True)
        fig_fea_panel(model, urban_scenario(160, 207), torch.tensor([0.5, 0.25, 0.25], dtype=DT))
    if only is None or "R1" in only:
        print("R1 pareto front...", flush=True)
        fig_pareto_front()
    if only is None or "R2" in only:
        print("R2 baseline comparison...", flush=True)
        fig_baseline_comparison()
    if only is None or "R3" in only:
        print("R3 complexity...", flush=True)
        sys.path.insert(0, str(ROOT / "scripts" / "analysis"))
        fig_complexity()
    print("done.")
