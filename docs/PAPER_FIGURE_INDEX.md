# Paper Figure Index — generated files, captions, provenance

Generated set lives in `result/paper_figures/` (each figure as both `.png` @300dpi and vector `.pdf`).
Style: IEEE two-column, Wong colour-blind-safe palette, serif type — see
`scripts/analysis/make_paper_figures.py` (data figures) and `make_paper_schematics.py` (block diagrams).

Regenerate everything:
```
python -B scripts/analysis/make_paper_figures.py      # data figures (from result/*.json)
python -B scripts/analysis/make_paper_schematics.py   # F0.1, F2.1, F3.7
```

| file | § | one-line caption | data source |
|------|---|------------------|-------------|
| `F0.1_pipeline_schematic` | §0 | End-to-end differentiable constructor: snapshot→candidate→GNN→top-$k$→Avalanche evaluator→coupled loss→one `.backward()`, no labels. | schematic |
| `F0.2_protocol_floor` | §0 | Perfect-link failure floor vs degree budget {4,8} × protocol variant — the feasibility reference. | `protocol_floor_table` |
| `F1.3_pareto_trajectory` | §1.1 | The operating point's F–D path during training: violent first-~5-step reorganization then glide to convergence. | `production_report_operating_point` |
| `F1.4_two_config_pareto` | §1.1 | paper_env Pareto **flat** (no D/E levers) vs operating_point **live** front — justifies the two-config design. | both `production_report_*` |
| `F1.5_w0_dominated` | §1.1 | w=0 rel-only is strictly dominated (worse F **and** ~36× cost); cost blow-up bars. | `production_report_operating_point` |
| `F1.9_transfer_regret` | §1.2 | N-randomized planner vs from-scratch N=10000 expert: transfer regret **+0.0062**. | `scale_transfer` |
| `F1.10_advantage_heatmap` | §1.3 | 27-cell advantage map (density×coupling, faceted by profile); black outline = seed-robust GNN advantage. | `advantage_map` |
| `F1.12_mc_survival` | §1.3 | Per-cell advantage gap quenched→Monte-Carlo: survives ground truth (all gaps > 0). | `advantage_montecarlo` |
| `F2.1_emission_loop_schematic` | §2 | Emission feedback loop: carried $\hat P$(correct) re-grounds the recurrent state each frame. | schematic |
| `F2.2_sigma_init` | §2 | $\sigma_{init}$ per arm: ~39× drop iff emission on (independent of graph/recurrence). | `emission_2x2_isolation` |
| `F2.3_probe_divergence` | §2 | Mechanism probe: gate-input scale blows up & gate gradient vanishes without emission. | `emission_probe_collapse_regime` |
| `F2.4_outcome_range` | §2 | F outcome range over inits: −emission spans up to collapse (~0.96), +emission tight (~0.55). | `emission_2x2_isolation` |
| `F2.5_replication` | §2 | $\sigma_{init}$ reduction across cells — severe at sparse+coupled, absent dense (regime-specific). | `fine_stage_*`, `emission_replication_*` |
| `F3.1_gating_map` | §3.1 | Deployment gating decision surface: density×coupling → GNN/marginal/heuristic policy. | `gating_demo` |
| `F3.2_routing_validation` | §3.1 | Forward-only routing: realized heuristic F vs predicted band, 15/15 in band. | `gating_demo` |
| `F3.3_context_estimation` | §3.1 | Deploy-time context estimation: estimated vs true density per frame. | `gating_demo` |
| `F3.4_retention` | §3.2 | In-grid retention by density: governed vs naive vs LOCO ceiling. | `mixture_governed_gradnorm` |
| `F3.5_gradient_diagnostic` | §3.2 | Gradient-conflict diagnostic: magnitude imbalance + all-positive cosines → GradNorm not PCGrad. | `mixture_governed_gradnorm` |
| `F3.6_offgrid_interpolation` | §3.2 | Off-grid interpolation: positive advantage gap at interpolation cells. | `mixture_governed_gradnorm` |
| `F3.7_selection_vs_unification` | §3.3 | The two routes to serve the envelope: runtime gating (selection) vs governed generalist (unification). | schematic |
| `F4.1_currency_faithfulness` | §4 | Same topology under mean-field/quenched/MC: mean-field **16–99×** optimistic (blind at easy cells); quenched **1.3–4.4×** of MC (best 1.31× at the operating point). | `currency_faithfulness` |
| `F4.2_temporal_null` | §4 | Temporal arms (static/no_memory/full/filter) all within the §2 $\sigma_{init}$≈0.20 envelope — no mean-F win outside init noise (single-init; paired test pending). | `s2_temporal_null` |
| `F4.3_fair_heuristics` | §4 | GNN at **parity with channel-rank**, **−62% vs carried-reliability** on identical frames — not an unfair-info artifact (clean win is §1.3, not here). | `s2_temporal_null` |
| `F4.5_w0_seed_band` | §4 | w=0 rel-only across 5 inits: **bimodal/unstable** ($\sigma$=0.21) vs tight optimized ($\sigma$=0.02); cost blow-up **26× robust**. Cost terms regularize init-variance. | `w0_seed_band` |

## Advanced / high-information-density figures (de-bar-chart upgrades) — `make_advanced_figures.py`

Built to diversify chart types and add the missing topology heroes. The F5.4–F5.7 set are **richer
alternatives** to existing bar charts (kept non-destructively so you can compare and pick one per slot).

| file | replaces/adds | one-line caption | data source |
|------|---------------|------------------|-------------|
| `F5.1_topology_panel` | NEW (city scene + logical) | **2-row panel across N∈{50,100,200,400}:** row 1 = physical deployed topology (urban grid densifying), row 2 = matching logical graph (force layout) with reciprocity/clustering. **Pure topology — uniform node size/colour, no communication encoding.** | re-deploy `production_report_paperenv/model.pt` at 4 scales |
| `F5.3_ablation_radar` | NEW (radar) | Emission 2×2 ablation radar over {stability, accuracy, worst-case, best-case, consistency} — +emission fills every axis, −emission collapses to centre. | `emission_2x2_isolation` |
| `F5.4_currency_slopegraph` | alt to **F4.1** | Slopegraph mean-field→quenched→MC (log F): optimism climbs left→right; pairs the 16–99× / 1.3–4.4× story. | `currency_faithfulness` |
| `F5.5_w0_dumbbell` | alt to **F4.5** | Per-seed dumbbell: 4 collapsed + 1 escaped w=0 seeds all converge into the tight w=5 band; 26× cost strip. | `w0_seed_band` |
| `F5.6_emission_phase_portrait` | alt to **F2.3** | Phase portrait gate-input scale vs gate-grad (log). *Note: for this probe the dual-line F2.3 separates the arms more cleanly — keep F2.3 as the primary mechanism figure.* | `emission_probe_collapse_regime` |
| `F5.7_advantage_bubble_map` | alt to **F1.10 + F1.12** | Bubble/parity map: size ∝ |gap|, colour = cell class, solid edge = seed-robust, ring = MC-audited (folds in F1.12). | `advantage_map` + `advantage_montecarlo` |
| `F5.8_pareto_swarm` | NEW (2nd tier) | Coupled cost–reliability front (w>0) with per-scene swarm + CI; w=5 best, w=0 off-scale. | `production_report_operating_point` |
| `F5.9_retention_cosine_web` | NEW (2nd tier; folds F3.4+F3.5) | Governance recovers the ceiling (dumbbell naive→governed) **explained by** the gradient cosine web (all cos>0, 2.1× norm imbalance → GradNorm). | `mixture_governed_gradnorm` |
| `F5.10_gating_sankey` | NEW (2nd tier) | 3-stage alluvial density→interference→gate decision (ribbon colour = policy). | `gating_demo` |

**Bar-chart reduction plan (from the design workflow):** CONVERT F4.1→F5.4, F4.5→F5.5, F1.10→F5.7 (absorb
F1.12); KEEP F2.2 (it's the quantitative companion to the F5.3 radar), F2.4/F2.5/F4.2/F4.3 (controls/nulls).
Net: visible bar figures drop ~6→~3. Second-tier (F5.8–F5.10) now built: Pareto-swarm, retention+cosine-web,
gating alluvial.

## Spatial FEA-style comparison (planned vs no-planning) — `make_fea_comparison.py`

Smooth, continuous filled-contour fields over the V2X scene, comparing the **learned degree-4 backbone**
against **no planning** (every vehicle floods all in-range candidates). Both evaluated through the same
load-aware analytic evaluator (operating-point physics); fields are Nadaraya-Watson Gaussian-smoothed for
continuity. **Darker = worse.**

| file | one-line message | per-node quantity |
|------|------------------|-------------------|
| `F6.1_fea_congestion` | Channel congestion (active-link density): no-planning floods 2× the links → 2× contention. | link-midpoint density |
| `F6.2_fea_delay` | Communication delay: no-planning 14.6 vs planned 10.6 expected consensus rounds. | `node_expected_rounds` |
| `F6.3_fea_reliability` | Consensus failure $F$: no-planning mean 0.48 vs planned 0.33. | per-node $F$ = wrong+undecided |

Third metric chosen = **reliability failure $F$** (closes the causal chain congestion→delay→failure);
swap to energy via the script if an efficiency axis is preferred.

**Status:** 36 figures generated (24 base + 9 advanced + 3 FEA; F5.1 panel replaces the old F5.1 hero + F5.2).
§4 reflects honest results: F4.2/F4.3 surfaced
two honest negatives (temporal single-init inconclusive; GNN matches but doesn't beat channel-rank in
streaming) — documented rather than hidden. Only **F4.4** (HARQ ms-latency budget) remains deferred —
it needs the eff-rounds→ms factor wired in; the D/E story is already covered by F1.5/F4.5.
