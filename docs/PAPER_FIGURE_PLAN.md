# Paper Figure Plan — manifest mapped to the v2 experiments chapter

One row per figure. The current figure set (14 distinct types) is too thin and too generic to carry the
argument; this plan brings it to a publishable **~30 figures**, of which the large majority are *already
backed by data on disk* — they need plotting code, not new experiments.

**Status legend**
- ✅ **EXISTS** — usable as-is (path given).
- ♻️ **REPLOT** — data already saved in `result/...`; needs new/upgraded plotting code only (cheap).
- 🆕 **NEW-RUN** — requires an experiment we have not run.
- ✏️ **SCHEMATIC** — hand-drawn / TikZ; no data (method/concept diagram).

**Priority:** **P1** = must-have in the main paper · **P2** = supplementary / appendix.

**Inventory now:** 14 distinct figures exist (8 production ×2 envs collapse to 8 types + 6 envelope).
**Plan:** ✅10 reuse · ♻️~17 replot-from-disk · 🆕~4 new-run · ✏️~4 schematic.

---

## §0 Setup

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F0.1** | **Pipeline schematic** | snapshot → candidate graph → GNN edge scores → hard-top-k constructor (straight-through) → analytic differentiable Avalanche evaluator (C/D/E) → coupled loss → **one** `.backward()`, **no labels**. The whole method on one page. | block diagram | ✏️ SCHEMATIC | **P1** — the method figure, currently missing |
| **F0.2** | **Protocol-floor reference** | perfect-link failure floor vs degree budget {4,8} × ic profile; the ~5× lever 4→8. Establishes the feasibility line every F is read against. | grouped bars / line | ♻️ REPLOT (upgrade `floor_feasibility.png`) | `protocol_floor_table` · **P1** |
| **F0.3** | **Example urban scene + candidate graph** | one snapshot: vehicles at (x,y) + faint candidate links *before* selection — shows the substrate the constructor prunes. | scatter + LineCollection | ♻️ REPLOT | scene env / deploy data · P2 |
| **F0.4** | **Currency ladder** | mean-field vs quenched vs MC as a conceptual ladder (optimism increasing). Small inset; the quantified version is F4.1. | concept strip | ✏️ SCHEMATIC | P2 (or fold into F4.1) |

## §1 — End-to-end constructor (MAIN RESULT)

### 1.1 Production model + coupled C/D/E Pareto

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F1.1** | Training curves | loss components (L_R/L_D/L_E) + C/D/E/F + grad norm vs step; convergence + early reorganization. | 2×2 multi-axis | ✅ EXISTS `production_report_*/figures/training_curves.png` | **P1** |
| **F1.2** | Coupled Pareto front | F-D and F-E (+ 3D F-D-E), log-axis, ±95% CI; the coupled loss controls D/E not only F. | scatter + front | ✅ EXISTS `pareto.png`, `pareto_3d.png` | **P1** |
| **F1.3** | **Pareto-point training trajectory** | the operating point's F-D path from init → converged: violent first-5-steps reorganization (D 267→96, grad 7.6e4→1e3) then glide. Answers "how the operating point *moves*." | annotated path on F-D plane | ♻️ REPLOT | `production_report_operating_point/report.json` (trajectory) · **P1, new** |
| **F1.4** | **Two-config contrast** | paper_env Pareto **flat** (no D/E levers, w inert) vs operating_point **live** front, side by side — justifies the two-config design. | 2-panel Pareto | ♻️ REPLOT | both `report.json` · **P1, new** |
| **F1.5** | **w=0 degenerate corner** | rel-only is strictly dominated: D/E ~36× higher *and* F worse; annotate it off the achievable front (log axis). Upgrades the current `de_ci.png`. | bar + dominated marker | ♻️ REPLOT (upgrade `de_ci.png`) | `report.json` pareto rows · **P1** |
| **F1.6** | Deployed optimal topology | physical scene (vehicles colored by per-node F, links by query weight) **+** logical graph (node size ∝ out-degree). The result you can *look at*. | scatter / network pair | ✅ EXISTS `physical_topology.png`, `logical_topology.png` | **P1** |
| **F1.7** | Deployed distributions | per-node F histogram, out-degree, per-link success. | 1×3 hist | ✅ EXISTS `distributions.png` | P2 |

### 1.2 Variable-N transferability

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F1.8** | Scalability vs N | one model: F/C/D/E flat across N=100→10000 (F 0.064–0.066); size-invariant scorer. | line, log-x | ✅ EXISTS `scalability.png` | **P1** |
| **F1.9** | **Transfer regret** | N-randomized planner vs from-scratch N=10000 expert at eval Q=21: **+0.0062 ± 0.0002**, per-seed dots + CI. The headline transfer number, currently *un-visualized*. | paired bars + per-seed | ♻️ REPLOT | `scale_transfer.json` · **P1, new** |

### 1.3 Advantage scope (where learning wins)

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F1.10** | **Advantage-region heatmap** | the 27-cell grid: density × coupling, faceted by confidence profile, color = gap (bestH − F_gnn); robust cells (gap−2σ>0) outlined. THE advantage-region figure. | 3-facet heatmap | ♻️ REPLOT | `advantage_map.json` · **P1, new** |
| **F1.11** | Gap with seed bands | the ~9 robust GNN-advantage cells with ±σ_seed error bars (gap +0.08…+0.11, ~13–18× σ). | bar + error bars | ✅ EXISTS `envelope_gap_bands.png` | **P1** |
| **F1.12** | **MC survival** | per cell, gap_quenched vs gap_mc — the advantage survives Monte-Carlo ground truth (+0.10 → +0.07–0.09, 3/3). Credibility. | paired scatter / slope | ♻️ REPLOT | `advantage_mc.json` · **P1, new** |
| **F1.13** | Advantage mechanism | gap vs sparsity and vs interference coupling — advantage lives on the channel **cliff** + where edges have global coupling. | scatter + trend | ♻️ REPLOT | `advantage_map.json` · P2 |
| **F1.14** | Area-confound control | density-300 at N=600 vs N=1800 (area fixed): density, not map size, is the driver. | grouped bars | ♻️ REPLOT | `advantage_map_area_control.json` · P2 |

## §2 — Emission-grounded stabilization (STANDOUT)

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F2.1** | **Emission feedback-loop schematic** | carried P(correct) (detached) re-grounds the recurrent state each frame → bounds the gate input → gradient survives. The mechanism in one diagram. | loop diagram | ✏️ SCHEMATIC | **P1**, missing |
| **F2.2** | **σ_init by arm (2×2 + control)** | full / no_graph / filter / no_graph_filter / filter_nomem: σ_init drops ~30–39× **iff** emission on, independent of graph coupling (p=0.86) and recurrence. | grouped bars + paired-p | ♻️ REPLOT (verify/replace `c3_mechanism.png`) | `emission_2x2_isolation/summary.json` · **P1** |
| **F2.3** | Mechanism probe divergence | gate-input scale (joined_std 0.35→11.8) + gate-weight gradient → 0, full vs filter over epochs. The money plot. | dual-axis vs epoch | ✅ EXISTS `c3_probe_divergence.png` | **P1** |
| **F2.4** | **Collapse vs escape** | F-trajectory: emission escapes (F≈0.96→0.57) vs pure-memory stuck at the collapsed init (F≈0.96). The outcome. | F vs epoch, 2 lines | ♻️ REPLOT | `emission_probe.json` / `fine_stage_*` · **P1, new** |
| **F2.5** | Replication across cells | σ_init reduction severe at sparse+coupled (39×, 27×), absent at dense d200 — regime-specific, not universal. | bars by cell | ♻️ REPLOT | `emission_replication_d100_c0`, `fine_stage_*` · P2 |

## §3 — Operationalizing the envelope

### 3.1 Domain-aware gating (selection)

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F3.1** | **Gating decision map** | density × coupling → policy (USE_GNN / marginal / heuristic-OK), colored regions; the advantage map turned into a runtime decision surface. | 2-D decision heatmap | ♻️ REPLOT | `gating_table.json` · **P1, new** |
| **F3.2** | **Forward-only routing validation** | per frame: realized heuristic F vs the gate's predicted band — **15/15 in band**. Proves the estimated context indexes the right cell. | band + points vs frame | ♻️ REPLOT | `gating_demo_log.json` · **P1, new** |
| **F3.3** | Context estimation accuracy | estimated vs true density per frame (99.7 vs 100) + mean-SINR proxy. | scatter / line | ♻️ REPLOT | `gating_demo_log.json` · P2 |

### 3.2 Governed mixture generalist (unification)

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F3.4** | Retention by density | governed vs naive vs LOCO ceiling: naive d100=0.34 (negative transfer) → governed ≈1.00 / 0.91 (ceiling). | grouped bars | ✅ EXISTS `governed_retention.png`, `generalization.png` | **P1** |
| **F3.5** | **Gradient-conflict diagnostic** | per-density gradient norms (d100 smallest, 802 vs 1709 → *magnitude* imbalance) + pairwise cosines (all positive, d100·d200=0.822 → *no directional* conflict) → recommends GradNorm not PCGrad. The "measure-then-choose-tool" visual. | bars + cosine matrix | ♻️ REPLOT | `mixture_governed_*/envelope_governed.json` (diagnostic) · **P1, new** |
| **F3.6** | Off-grid interpolation | the 12/12 interpolation cells with positive gap — generalist interpolates between grid densities/couplings. | scatter | ♻️ REPLOT | `envelope_governed.json` (off_grid) · P2 |

### 3.3 Synthesis

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F3.7** | Selection vs unification | trade-off table/radar: deploy-time cost, interpretability, off-grid coverage, single-artifact. | table / radar | ✏️ SCHEMATIC | P2 |

## §4 — Ablations, baselines, robustness

| ID | Figure | Visual argument | Type | Status | Source / priority |
|----|--------|-----------------|------|--------|-------------------|
| **F4.1** | **Currency faithfulness** | same topology under mean-field vs quenched vs MC: mean-field 22–40× optimistic, quenched within ~1.16× of MC. Licenses every absolute F. | grouped bars / ratio | ♻️ REPLOT (partial) → may need a small 🆕 run for the mean-field arm | `advantage_mc.json` + `validate_*_montecarlo` · **P1** |
| **F4.2** | Temporal mean-F null | static / no_memory / full / filter — equal mean F (paired n.s.); memory buys *stability* not accuracy. Pre-empts over-claim. | bars + paired-p | 🆕 NEW-RUN (s2 ablation) | P2 |
| **F4.3** | Fair stream heuristics | GNN vs channel-rank & carried-reliability on identical held-out frames/shadow (~19% rel.) — advantage isn't an info artifact. | bars | 🆕 NEW-RUN | P2 |
| **F4.4** | HARQ latency budget | D in ms vs w_D (33.5 → 11.2 ms under w_D×10), shaded 10–100 ms V2X budget. | line + budget band | ♻️ REPLOT | `production_report_operating_point/report.json` · P2 |
| **F4.5** | w=0 multi-seed band | 3–5 inits at w=0: converts "rel-only also degrades F" from single-init to banded headline (cost-blowup already robust). | bar + seed band | 🆕 NEW-RUN | P2 (the one open item) |

---

## Build order (recommended)

1. **Zero-cost wins (♻️ replot from disk, P1) — do first:** F1.3, F1.4, F1.5, F1.9, F1.10, F1.12, F2.2,
   F2.4, F3.1, F3.2, F3.5. These eleven are the figures that actually close the "too few" gap and they
   need *only plotting code* (all data on disk).
2. **Reuse as-is (✅):** F1.1, F1.2, F1.6, F1.7, F1.8, F1.11, F2.3, F3.4, F0.2 — drop into the draft now.
3. **Schematics (✏️):** F0.1 (pipeline) and F2.1 (emission loop) are P1 and worth hand-drawing early —
   they frame §0 and §2.
4. **New runs (🆕), lowest priority:** F4.2, F4.3, F4.5 (and the mean-field arm of F4.1). Only F4.5 is
   tied to an open scientific item; the rest are defensive supplementary.

## Per-section figure budget (main paper)

| § | P1 figures | of which new work |
|---|-----------|-------------------|
| §0 | F0.1, F0.2 | F0.1 schematic |
| §1 | F1.1–1.3, F1.5, F1.6, F1.8–1.12 | F1.3,1.4,1.9,1.10,1.12 (all replot) |
| §2 | F2.1–2.4 | F2.1 schematic, F2.2/2.4 replot |
| §3 | F3.1, F3.2, F3.4, F3.5 | F3.1,3.2,3.5 replot |
| §4 | F4.1 | partial replot + small run |

Net: **~22 P1 figures**, of which **~11 are replot-from-disk**, **2 schematics**, **~9 already exist**.
