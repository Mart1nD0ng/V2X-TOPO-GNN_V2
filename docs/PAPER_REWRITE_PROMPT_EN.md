## 0. Role and overall goal

You are a senior communications/ML paper author. Task: rewrite this V2X topology-optimisation
consensus paper's mathematical mainline, experiments, and conclusions — from the **superseded old
derivation** (mean-field node-mean `F`, iid beta-tail quorum, logistic-BLER, fixed degree) to the
**unified new mainline** (shared finite-mixture global `F` + generating-function quorum DP +
finite-blocklength `ℓ` + preference-conditioned GNN + global-risk emission) — and rewrite the
experiments/conclusions from the **verified experimental data**. Requirements: rigorous math, every
number drawn from the reproducible results provided below, **honest about limitations**, no
over-claiming. The prose is yours.

### Hard rules (must not be violated)
1. **Use only the numbers given in this document**; every table/inline value must match the values
   below (or be pulled from `docs/gate_evidence/*.json`). No fabrication.
2. **Use the equations in §2 as given** (you may change notation/typesetting, not the semantics).
3. **Delete the old conclusions listed in §3** and recompute/rewrite from the new data.
4. **State the limitations in §6 honestly** (this is a hard honesty requirement, not optional).
5. The old derivation may appear only as an **explicitly-labelled historical/contrast appendix**,
   never as the headline theorem.

---

## 1. One-sentence headline (paraphrase; do NOT copy verbatim)

> A single preference-conditioned topology GNN, built on a unified, self-consistent, end-to-end
> differentiable network-consensus mathematical mainline, covers the Pareto front of F (global
> consensus-failure probability) / D (completion delay) / E (network energy) with one checkpoint;
> on held-out scenarios its front **significantly outperforms** the strongest non-learned
> exhaustive baseline on two principled multi-objective metrics — normalisation-free Pareto
> dominance and hypervolume — with near-linear complexity in the network size.

---

## 2. The mathematical mainline: equations to put in the paper (replacing the old derivation)

**(M1) Weighted distinct-peer query policy (replaces any hard top-k / fixed-degree selection).**
The GNN emits logits `s_{ij}`, unnormalised weights `a_{ij}=exp(s_{ij})`. Each round draws exactly
`k` distinct peers; the subset distribution is the elementary symmetric polynomial form
`P(S_i=S)=(∏_{j∈S} a_{ij})/e_k(a_i)`, `|S|=k`; the per-edge inclusion probability `π_{ij}` satisfies
`Σ_j π_{ij}=k`. No degree cap, no top-k.

**(M2) Exact heterogeneous quorum DP (replaces betainc / iid beta-tail).**
Three-way (correct / wrong / no-response) generating function
`Ψ_i(z,x,y)=∏_j [1 + z·a_{ij}(p^0_{ij} + p^+_{ij} x + p^-_{ij} y)]`,
`P_i(m,n)=[z^k x^m y^n]Ψ_i / e_k(a_i)`, yielding decision probabilities `h^+_i, h^-_i, h^0_i`
(strong majority `2α>k`). Complexity `O(E k^3)`, differentiable. **No iid-with-replacement
beta-tail approximation is used.**

**(M3) Shared finite-mixture global event probability `F` (replaces `F=\overline{F_i}` node mean;
the H1 core).** A network-shared discrete latent `Z∈{1..Q}` (weights `ω_r`) couples the nodes;
given `Z=r` the terminal states are a conditional product.
`S_C=Σ_r ω_r ∏_{i∈H} c_{ir}`,
`F_global = 1 − S_C = −expm1( logsumexp_r( log ω_r + Σ_{i∈H} log c_{ir} ) )`, loss `= −log S_C`.
`c_{ir}` comes from the §6 graph-coupled Snowball finite-horizon recurrence (neighbour marginal
preferences enter each conditional probability). **State clearly that `F_global` is a strict global
event probability `P(∃ i: Y_i≠C)`, not a node average.**

**(M4) Rigorous finite-blocklength link reliability `ℓ` (replaces logistic-BLER sigmoid).**
PPV normal approximation `ε_FBL` with explicit channel dispersion
`V(γ)=(1−(1+γ)^{-2})(log_2 e)^2`; blocklength in real complex channel uses; request/response,
Mode-2 collision, half-duplex, finite HARQ modelled separately; the headline uses only the
3GPP TR 37.885-grounded path, no idealised channel.

**(M5) Independent D / E.** `D` = global order statistic of all-node correct completion (from the
`S(t)` trajectory); `E` = total joules across the network / rounds / attempts; power head
`P_i=P_min+(P_max−P_min)σ(r_i)`, blocklength head `n_i=n_min+(n_max−n_min)σ(b_i)`.

**(M6) Preference conditioning and scalarisation.** Preference `λ=(λ_F,λ_D,λ_E)∈Δ²` is a GNN input
(FiLM modulation); training uses augmented Chebyshev
`max_m λ_m (z_m−z_m^*)/s_m + ρ Σ_m λ_m (z_m−z_m^*)/s_m`. One checkpoint sweeps `λ` to trace the
whole front.

**(M7) Global-risk emission (replaces per-node confidence emission).**
`r_{ir}(t)=−log c_{ir}(t)` (the node's contribution to the global risk `−log S_r=Σ_i r_{ir}`); the
next-frame feature is `e_i^t=clip( sg[ Σ_r ρ_r r_{ir} ] / r_max , 0,1 )` (sg = stop-gradient).
**Do NOT claim "a bounded scalar automatically constrains all hidden state" — that claim is
falsified by the mechanism ablation** (see §5 Table F).

Full pipeline (suggested as one method figure):
`(a_θ,P_θ,n_θ) → π → Λ → γ → ℓ → (h^+,h^-,h^0) → c_{ir}(t) → (F_global, D, E)`.

---

## 3. Old conclusions to DELETE / recompute (spec §12)

Delete and replace with new data: the degree-4 protocol floor; the old `F=0.0635`-style node-mean
results; the mean-field / quenched / MC "currency" narrative; the hard top-k / spanning-tree
constructor; the old Pareto table; the per-node-confidence emission; the fixed-degree baseline
ranking. **Rewrite every sentence/table/figure that references any of these.**

---

## 4. Experiment design (write into the Experiments section; reorganise freely, facts as given)

- **Numerical-correctness validation** (supports method credibility): tabulate each new-mainline
  component's error against an independent brute-force / closed-form / scipy reference as an
  "exactness" table (Table A), arguing the mainline is an **exact implementation**, not a heuristic.
- **Multi-objective baseline comparison** (the headline experiment): a single preference-conditioned
  checkpoint is trained on TRAINING scenarios and evaluated on HELD-OUT scenarios. All methods
  (the learned model + every baseline) are scored through the **same physics pipeline**, differing
  ONLY in how the controls `(s,P,n)` are produced → fair.
  - Baselines (honest strong baselines; the fixed-degree ranking is FORBIDDEN): `best-fixed` = the
    per-scenario / per-preference oracle front over a 392-point exhaustive constant-policy family
    (4 query heuristics uniform/distance/inverse-distance/degree + random restarts × a 7×7 constant
    (P,n) grid) — the **strongest** non-learned baseline; `λ-blind` = the same architecture trained
    without preference conditioning (isolates the value of conditioning); `untrained` = random
    initialisation (discriminative control).
  - Metrics: (i) **Pareto set-coverage** (normalisation-free, primary): `C(A,B)` = fraction of B
    dominated by A; (ii) **hypervolume** (model-independent normalisation box);
    (iii) **Chebyshev front-scalar** (secondary — the model LOSES this, must be reported honestly).
  - Significance: paired Wilcoxon (one-sided) across held-out scenarios.
- **Ablations**: preference conditioning (vs λ-blind), training (vs untrained), and the emission's
  bounded-scalar mechanism ablation (Table F).
- **Complexity profiling**: fixed spatial density (area ∝ N ⇒ E=O(N)); sweep N, measure end-to-end
  runtime and peak tensor size; give the near-linear fit and figure.
- **Reproducibility**: fixed seeds; one-button `python scripts/gates/run_all_gates.py`; per-experiment
  `baseline_comparison.py` / `profile_scaling.py` / `paper_tables.py`.

---

## 5. Experimental data (**use as given**; you may re-typeset / select subsets / convert to LaTeX,
but do NOT change the values)

### Table A — Numerical correctness / exactness (new mainline vs independent reference; abs/rel error)
| Component | Check | Error |
|---|---|---|
| Global F (G1) | shared-mixture closed form vs brute-force joint | 0.00e+00 |
| Global F (G1) | log-domain vs direct | 3.3e-19 |
| Global F (G1) | failure-decomposition identity | 0.00e+00 |
| Global F (G1) | analytic S_C vs Monte-Carlo | 7.2e-6 |
| k-subset (G2) | normalisation \|Σp−1\| | 4.4e-16 |
| k-subset (G2) | inclusion prob. vs brute force | 6.7e-16 |
| k-subset (G2) | \|Σπ−k\| identity | 2.2e-15 |
| quorum DP (G3) | distribution vs brute-force enumeration | 2.9e-16 |
| FBL (G5) | V(γ) vs closed form | 4.4e-16 |
| FBL (G5) | ε_FBL vs scipy | 1.4e-17 |
| delay D (G6) | D vs CDF reference | 8.9e-16 |
| energy E (G6) | E vs brute force | 0.00e+00 |
| emission (G8) | −log S_r = Σ_i r_ir identity | 0.00e+00 |
> Purpose: argue the mainline is a **machine-precision exact** implementation. You may keep only a
> few representative rows + a "all components match independent references to machine precision"
> summary sentence.

### Table B — Multi-objective baseline comparison (headline; 12 held-out scenarios, paired Wilcoxon)
| Method | hypervolume↑ | C(model→·)↑ | C(·→model)↓ | Chebyshev↓ | HV win-rate | p (one-sided) |
|---|---|---|---|---|---|---|
| **Ours (model)** | **0.606** | — | — | 0.118 | — | — |
| best-fixed (exhaustive oracle, strongest baseline) | 0.509 | 0.392 | **0.000** | 0.061 | 100% | 2e-4 |
| fixed-uniform | 0.450 | 0.287 | 0.000 | 0.067 | 100% | 2e-4 |
| fixed-distance | 0.465 | 0.267 | 0.000 | 0.066 | 100% | 2e-4 |
| fixed-invdist | 0.438 | 0.480 | 0.000 | 0.097 | 100% | 2e-4 |
| fixed-degree | 0.450 | 0.287 | 0.000 | 0.067 | 100% | 2e-4 |
| λ-blind (ablation) | 0.357 | 0.000 | 0.000 | 0.230 | 100% | 2e-4 |
| untrained (control) | 0.267 | 0.990 | 0.000 | 0.280 | 100% | 2e-4 |
> Key narrative points (organise freely): (i) **C(·→model)=0.000 for ALL baselines** — no baseline
> point Pareto-dominates any model point; (ii) the model's C(model→best-fixed)=0.392; (iii) HV under
> a model-independent box is 0.606 vs 0.509 = **1.19×**, on 100% of held-out scenarios, p=2e-4;
> (iv) HV ordering model > best-fixed > λ-blind > untrained (clean gradient; ablations work).

### Table C — Ablation: value of preference conditioning and training
| Comparison | HV win-rate | p | Conclusion |
|---|---|---|---|
| model vs λ-blind | 100% | 2e-4 | preference conditioning adds real value |
| model vs untrained | 100% | 2e-4 | training adds real value; untrained dominated at C=0.99 (gate is discriminative) |

### Table D — Representative operating points (held-out 12-scenario mean ± std; shows genuine steering)
| Preference λ | F (failure prob.) | D (delay) | E (energy) |
|---|---|---|---|
| F-pref [1,0,0] | **0.355** ± 0.042 | 0.0517 ± 0.0011 | 0.629 ± 0.100 |
| D-pref [0,1,0] | 0.711 ± 0.071 | **0.0415** ± 0.0007 | 0.677 ± 0.094 |
| E-pref [0,0,1] | 0.513 ± 0.053 | 0.0529 ± 0.0010 | **0.168** ± 0.025 |
| balanced [.34,.33,.33] | 0.519 ± 0.053 | 0.0463 ± 0.0008 | 0.387 ± 0.057 |
> Steering is clean: each preference minimises its own objective on the diagonal (F-pref → lowest F,
> D-pref → lowest D, E-pref → lowest E). Use this for a 3D / projected Pareto scatter + preference arrows.

### Table E — End-to-end complexity (fixed density, area ∝ N)
| N | E | end-to-end runtime (ms) |
|---|---|---|
| 200 | 7,702 | 298 |
| 800 | 35,926 | 774 |
| 3,200 | 150,358 | 1,432 |
| 6,400 | 307,016 | 1,954 |
> Fit: E~N exponent 1.06; end-to-end t~E exponent 0.51 (overhead-dominated sub-linear, i.e. ≤ linear).
> The largest materialised tensor is the `O(E·k^3)` quorum cube, whose scaling exponent vs E is
> **1.000** (no N×N). Figure: `docs/gate_evidence/g9_scaling.png`.

### Table F — Emission bounded-scalar mechanism ablation (same bounded emission, ‖H‖ growth over 30 frames)
| recurrence cell | ‖H‖ growth |
|---|---|
| GRU (gated) | 0.81 |
| contractive (ρ=0.5) | 0.02 |
| expansive (ρ=1.3, non-contractive) | **2017** |
> Conclusion (state honestly): a bounded scalar input does **not** automatically constrain the hidden
> state — the expansive recurrence's ‖H‖ diverges under the same bounded emission; hidden-state
> boundedness is a property of the recurrence's contraction/gating (BIBO), not of the input. The
> paper must **not** claim "a bounded scalar auto-constrains all hidden state"; state the verified
> narrow properties instead (e∈[0,1] by construction, stop-gradient, emission exactly aligned with
> the global `−log S_C`).

### G7 steering (supporting sentence for method/ablation)
One checkpoint, sweeping λ: 10/10 mutually non-dominated, 3/3 argmin-at-vertex steering; the λ-blind
ablation collapses the steering.

---


## 7. Writing freedom (claudeprism decides)

- Section order, sub-headings, transitions, emphasis, the narrative arc of the abstract/intro, the
  angle of the related-work contrast.
- Which tables to merge/trim (e.g. keep only a few representative rows of Table A); figure forms
  (Pareto scatter, complexity curve, method pipeline diagram, ablation bars).
- The granularity of theorem/proposition statements and the notation system (keep semantics consistent).
- Organise the "key narrative points" above into a fluent argument; do not just list them.
- Submit as tracked edits to `paper/main.tex`; if the old derivation is retained, label it explicitly
  as historical/contrast.

**Bottom line**: the math per §2, the deletions per §3, the numbers per §5, and the limitations per
§6 are fixed; everything else is yours.
