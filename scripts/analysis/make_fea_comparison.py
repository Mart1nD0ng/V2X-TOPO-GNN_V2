"""FEA-style spatial comparison: planned (learned) topology vs. no planning (flood).

Builds two topologies on one V2X scene and evaluates both through the SAME load-aware analytic
evaluator, then renders three smooth, continuous "finite-element"-style contour fields, each
comparing the two conditions side by side (darker = worse):

  F6.1  channel congestion  — spatial density of active communication links (channel contention)
  F6.2  communication delay — per-node expected consensus rounds (structural delay on the op config)
  F6.3  consensus failure F — per-node reliability failure (wrong + undecided)

"No planning" = every vehicle floods all in-range candidate peers (degree up to the candidate cap);
"planned" = the learned GNN top-k backbone (degree 4). Fields are produced by binning the per-node /
per-link quantity onto a fine grid and Gaussian-smoothing (Nadaraya-Watson), then drawn with filled
contours so boundaries are continuous rather than blocky.

Usage:
  python -B scripts/analysis/make_fea_comparison.py --config configs/operating_point_v1.yaml \
      --n 320 --out result/paper_figures
"""
from __future__ import annotations

import argparse
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
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter  # noqa: E402

from scripts.analysis.make_advanced_figures import _load_or_train  # noqa: E402
from scripts.analysis.make_paper_figures import save, set_ieee_style, DBL_W  # noqa: E402
from scripts.analysis.run_pareto_frontier import _scene_env, _topology_layer  # noqa: E402
from scripts.analysis.validate_operating_point_montecarlo import _evaluate_details, _select_topology  # noqa: E402

_CMAP = "magma_r"   # low = light, high = dark  (darker = worse)


def _topos(model, env):
    """Planned (model, degree budget) and flood (uniform scores, all candidates) topologies."""
    cfg = env["cfg"]; n = env["candidate"].num_nodes
    layer_p, caps_p = _topology_layer(cfg, n)
    with torch.no_grad():
        topo_planned = _select_topology(model, env, layer_p, caps_p)
    cand_cap = int(cfg.get("candidate_graph", {}).get("max_candidates_per_node", 12) or 12)
    cfg_flood = dict(cfg); cfg_flood["max_out_degree"] = cand_cap
    layer_f, caps_f = _topology_layer(cfg_flood, n)
    n_edges = env["features"]["src_index"].numel()
    uniform = torch.zeros(n_edges, dtype=torch.float64)        # equal scores -> uniform row-softmax over all candidates
    with torch.no_grad():
        topo_flood = _select_topology(uniform, env, layer_f, caps_f)
    return topo_planned, topo_flood


def _node_fields(topo, env, eval_q):
    """Per-node failure F, delay D (expected rounds), and the active-link list (src,dst)."""
    ev = _evaluate_details(topo, env, quenched_quadrature=eval_q)
    av = ev["avalanche_details"]
    F = (1.0 - av["node_p_correct_decision"]).detach().to(torch.float64).numpy()
    D = av["node_expected_rounds"].detach().to(torch.float64).numpy()
    sup = av["query_support"]
    src = sup.src_index.detach().numpy(); dst = sup.dst_index.detach().numpy()
    return F, D, src, dst


def _smooth_scalar(px, py, val, extent, grid, sigma):
    """Nadaraya-Watson field of a per-node scalar via binned Gaussian smoothing (continuous everywhere)."""
    x0, x1, y0, y1 = extent
    rng = [[x0, x1], [y0, y1]]
    Hs, _, _ = np.histogram2d(px, py, bins=grid, range=rng, weights=val)
    Hc, _, _ = np.histogram2d(px, py, bins=grid, range=rng)
    Ss = gaussian_filter(Hs, sigma); Sc = gaussian_filter(Hc, sigma)
    return (Ss / np.maximum(Sc, 1e-9)).T                      # ratio everywhere; outer mask applied at plot time


def _detect_lines(coord, lo, hi, bin_m=2.0, frac=0.18, merge_m=10.0):
    """Recover road centre-lines as position spikes (vehicles sit on the roads)."""
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


def _smooth_density(mx, my, extent, grid, sigma):
    """Smooth spatial density of points (link midpoints) -> congestion field."""
    x0, x1, y0, y1 = extent
    H, _, _ = np.histogram2d(mx, my, bins=grid, range=[[x0, x1], [y0, y1]])
    return gaussian_filter(H, sigma).T


def _panelpair(fields, extent, title, sub, label, out, fid, name, mask=None, px=None, py=None, roads=None):
    """Two side-by-side filled-contour panels (no planning | planned), shared scale, darker=worse.
    Optionally overlays the road grid (thin black lines) and vehicles (black hollow squares)."""
    (fa, ta), (fb, tb) = fields
    both = np.concatenate([fa[~np.isnan(fa)], fb[~np.isnan(fb)]])
    vmax = float(np.percentile(both, 90)); vmin = 0.0          # compress to the bulk so the mean gap reads clearly
    levels = np.linspace(vmin, vmax, 41)
    x0, x1, y0, y1 = extent
    gx = np.linspace(x0, x1, fa.shape[1]); gy = np.linspace(y0, y1, fa.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 3.5), sharey=True)
    cs = None
    for ax, f, t in ((axes[0], fa, ta), (axes[1], fb, tb)):
        ff = np.where(mask, np.nan, f) if mask is not None else f
        cs = ax.contourf(gx, gy, ff, levels=levels, cmap=_CMAP, extend="max")
        if roads is not None:                                  # road grid: thin black lines
            for rx in roads[0]:
                ax.plot([rx, rx], [y0, y1], color="black", lw=0.4, alpha=0.5, zorder=4)
            for ry in roads[1]:
                ax.plot([x0, x1], [ry, ry], color="black", lw=0.4, alpha=0.5, zorder=4)
        if px is not None:                                     # vehicles: black hollow squares
            ax.scatter(px, py, marker="s", s=5, facecolors="none", edgecolors="black", linewidths=0.35, zorder=5)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_title(t, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(cs, ax=axes, fraction=0.025, pad=0.02); cb.set_label(label, fontsize=7)
    fig.suptitle(f"{title}\n{sub}", y=1.02, fontsize=9.5, fontweight="bold")
    save(fig, out, fid, name)


def _composite(cols, extent, mask, px, py, roads, out, fid, name):
    """2 x len(cols) panel: row 0 = no planning, row 1 = planned; one shared colour scale per metric column.
    Each cell overlays the road grid (thin black) and vehicles (black hollow squares); darker = worse."""
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
        for r, (f, m) in enumerate([(ftop, mtop), (fbot, mbot)]):
            ax = axes[r][c]
            cs = ax.contourf(gx, gy, np.where(mask, np.nan, f), levels=levels, cmap=_CMAP, extend="max")
            for rx in roads[0]:
                ax.plot([rx, rx], [y0, y1], color="black", lw=0.32, alpha=0.5, zorder=4)
            for ry in roads[1]:
                ax.plot([x0, x1], [ry, ry], color="black", lw=0.32, alpha=0.5, zorder=4)
            ax.scatter(px, py, marker="s", s=3.0, facecolors="none", edgecolors="black", linewidths=0.28, zorder=5)
            ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.04, 0.96, m, transform=ax.transAxes, va="top", ha="left", fontsize=6,
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
    save(fig, out, fid, name)


def main() -> None:
    p = argparse.ArgumentParser(description="FEA-style spatial comparison: planned vs no-planning topology")
    p.add_argument("--config", default="configs/operating_point_v1.yaml")
    p.add_argument("--model", default="result/production_report_paperenv/model.pt")
    p.add_argument("--n", type=int, default=320)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--grid", type=int, default=200)
    p.add_argument("--sigma", type=float, default=8.0)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--out", default="result/paper_figures")
    args = p.parse_args()
    set_ieee_style()
    out = ROOT / args.out

    base, model = _load_or_train(args.config, ROOT / args.model, args.seed, 2000, args.max_steps)
    eval_q = int(base.get("eval_quenched_quadrature", base.get("quenched_quadrature", 21)))
    env = _scene_env(base, args.n, args.seed)
    nf = env["features"]["node_features"].detach().cpu().numpy()
    px, py = nf[:, 0] * 600.0, nf[:, 1] * 600.0
    print(f"  scene N={args.n}; evaluating planned vs flood under {args.config}...", flush=True)

    topo_p, topo_f = _topos(model, env)
    Fp, Dp, sp, dp = _node_fields(topo_p, env, eval_q)
    Ff, Df, sf, df = _node_fields(topo_f, env, eval_q)
    print(f"  links: planned={sp.size}, flood={sf.size} | F planned={Fp.mean():.3f}/flood={Ff.mean():.3f} | "
          f"D planned={Dp.mean():.2f}/flood={Df.mean():.2f}", flush=True)

    pad = 0.04 * (px.max() - px.min())
    extent = (px.min() - pad, px.max() + pad, py.min() - pad, py.max() + pad)
    grid = (args.grid, args.grid); sig = args.sigma

    # coverage mask: keep the scene footprint as ONE continuous region (fill inter-road holes),
    # mask only the exterior so boundaries are continuous rather than Swiss-cheese.
    Hc = np.histogram2d(px, py, bins=grid, range=[[extent[0], extent[1]], [extent[2], extent[3]]])[0]
    cover = gaussian_filter(Hc, sig).T
    present = cover >= (0.04 * cover.max())
    present = binary_closing(binary_fill_holes(present), iterations=3)
    mask = ~present
    road_x = _detect_lines(px, extent[0], extent[1]); road_y = _detect_lines(py, extent[2], extent[3])
    roads = (road_x, road_y)
    print(f"  detected {len(road_x)} vertical + {len(road_y)} horizontal roads", flush=True)

    def mids(src, dst):
        return (px[src] + px[dst]) / 2.0, (py[src] + py[dst]) / 2.0
    cong_f = _smooth_density(*mids(sf, df), extent, grid, sig)        # channel congestion = active-link density
    cong_p = _smooth_density(*mids(sp, dp), extent, grid, sig)
    dly_f = _smooth_scalar(px, py, Df, extent, grid, sig)            # delay = expected consensus rounds
    dly_p = _smooth_scalar(px, py, Dp, extent, grid, sig)
    fail_f = _smooth_scalar(px, py, Ff, extent, grid, sig)           # per-node consensus failure F_i
    fail_p = _smooth_scalar(px, py, Fp, extent, grid, sig)

    cols = [
        ("Channel congestion", "active-link density (a.u.)",
         (cong_f, f"{sf.size} links"), (cong_p, f"{sp.size} links")),
        ("Communication delay", "expected rounds $T_i$",
         (dly_f, f"mean {Df.mean():.1f}"), (dly_p, f"mean {Dp.mean():.1f}")),
        ("Consensus failure $F_i$", "per-node failure $F_i$",
         (fail_f, f"mean {Ff.mean():.2f}"), (fail_p, f"mean {Fp.mean():.2f}")),
    ]
    _composite(cols, extent, mask, px, py, roads, out, "F6.1", "fea_panel")
    print(f"done -> {out}")


if __name__ == "__main__":
    main()
