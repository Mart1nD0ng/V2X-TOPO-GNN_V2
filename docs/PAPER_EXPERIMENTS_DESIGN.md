# Paper Experiments Chapter — design v2 (purpose · setup · process · expected · analysis)

Maps the project's evidence base to a publishable experiments chapter. This is **v2**, restructured
around four result sections (was eight). Two targets:
**Main paper** (IEEE TMC/TWC/JSAC): §0–§4. **Workshop/LoG** (the most novel ML finding): §2 + a
condensed §0/§1.3.

**Structure (v2).**
- **§0 Setup** — substrate; absorbs the currency declaration + protocol floor + baselines (these used to
  be a standalone "surrogate honesty" section; they are now reference machinery, not a headline).
- **§1 End-to-end constructor (MAIN RESULT)** — production model + coupled C/D/E Pareto, fused with
  variable-N transferability and the advantage-region scope. Theme: *one differentiable constructor that
  is controllable, transferable, and has a characterized advantage region.*
- **§2 Emission-grounded stabilization (STANDOUT RESULT)** — the most novel ML finding.
- **§3 Operationalizing the envelope** — two routes to serve the whole operating envelope with one
  system: runtime **gating** (selection) vs. a **governed generalist** (unification).
- **§4 Ablations, baselines, robustness** — defends the design, bounds the claims, validates the
  currency against Monte-Carlo ground truth.

> **Currency discipline (used everywhere).** Every F is reported in a stated evaluator currency
> (mean-field Q=1 / quenched train-Q=11,eval-Q=21 / Monte-Carlo). Relative orderings are
> currency-invariant; absolute F is not. This rule is declared in §0 and assumed thereafter.

> **Ordering note.** v2 puts the deployment system (§3) before ablations (§4) — positive results first,
> defensive section last. If the gating+generalist content is preferred *after* ablations, §3 and §4 are
> independent and swap freely.

Each section gives **目的 · 设置 · 过程 · 预期结果 · 分析**, plus the backing artifact (`result/...`,
figure). All artifact paths use the post-cleanup semantic names.

---

## §0 — Experimental setup (no claim; the substrate)

**目的.** Pin the environment, the three evaluator currencies, the feasibility floor, the baselines, and
the metrics — so every later number is interpretable and reproducible, and so the advantage/scope claims
in §1 have a reference line.

**设置.**
- **Environment:** `configs/paper_environment_v1.yaml` — TR 37.885 urban channel + geometric
  (axis-visibility) LOS + Xavier init. The streaming experiments (§2) add the frozen spec (AR(1) hidden
  shadow 4 dB / 8 s, churn 2 births/frame absorb_inject, intersection turning, dt 2 s). A second config,
  `configs/operating_point_v1.yaml`, additionally turns ON structural delay + retransmission energy; it
  is the only place the C/D/E Pareto has live levers (§1.1).
- **Pipeline (the method under test):** vehicle snapshot → candidate graph → GNN edge scores →
  hard-top-k constructor (straight-through backward) → analytic differentiable Avalanche evaluator
  (C/D/E) → coupled loss → one `.backward()`. **No labels.**
- **Evaluator currencies:** mean-field (Q=1), quenched (train Q=11 / eval Q=21), Monte-Carlo. Faithfulness
  is *quantified* in §4 (mean-field is severely optimistic; quenched tracks MC at the operating point).
- **Protocol floor (feasibility reference):** the perfect-link closed-form failure of a topology is
  bounded below by a floor governed by (ic profile × protocol × degree budget) — **not** by training.
  Degree budget 4 floors ≈ 0.064; budget 8 ≈ 0.011 (the degree budget is itself a ~5× reliability lever).
  Any "F ≤ x" claim is read against this floor; the advantage map (§1.3) uses it to label *floor-limited*
  cells. **Artifact:** `result/protocol_floor_table`.
- **Baselines:** protocol floor (perfect-link closed form), extended top-k heuristics
  (channel/success/sinr/nearest/random), fair stream heuristics (channel-rank, carried-reliability),
  per-cell screening experts, and (for transfer) a from-scratch scale-specific expert.
- **Metrics:** F (failure), C (correct), D (delay: eff-rounds + ms), E (retransmission energy),
  headroom (= bestH − floor), gap (= bestH − F_gnn), σ_init, transfer regret, retention. Stats: paired
  t + Wilcoxon over shared (scene, init) grids; 95% CIs; seed bands.

**分析.** Descriptive. The single load-bearing idea is the **currency-declaration rule** plus the
**floor as a feasibility reference** — together they make "where learning helps" (§1.3) a measurable,
not rhetorical, question.

---

## §1 — End-to-end differentiable topology constructor (MAIN RESULT, Contributions C4 + C2)

> One section, three fused claims: the constructor **(1.1) trains to a controllable C/D/E operating
> point**, **(1.2) deploys from one checkpoint across two orders of magnitude in N**, and **(1.3) has a
> measurable, characterized advantage region** over strong heuristics. The through-line is
> **transferability + advantage scope**.

### 1.1 Production model & coupled C/D/E Pareto

**目的.** Prove the end-to-end constructor trains to the target operating point and that D and E are
genuinely *optimized by the coupled objective*, not just F.

**设置.** Production training N=2000, full per-step logging (loss components L_R/L_D/L_E, C/D/E/F, grad
norm); cost-weight Pareto sweep (w ≡ w_D = w_E ∈ {0, 0.5, 1, 2, 5, 10}), one model per weight; 6
held-out scenes for CIs; a small physical scene (N=80) for the deployed-topology figures. Run on **both**
configs: `paper_environment_v1` (headline physics, reliability-floor + scale story) and
`operating_point_v1` (the only config with live D/E levers → a non-degenerate Pareto).

**过程.** (1) Train, logging every step. (2) Sweep w, train one model each, evaluate held-out → F-D and
F-E fronts (+ 3D F-D-E) with 95% CI. (3) Render the deployed optimal topology in the physical scene and
as a logical graph; per-node reliability, out-degree, link-success distributions.

**预期结果.**
- **paper_environment_v1:** converges to F = 0.0635, C = 0.9365, D = 7.64, E = 2.10e-2; F sits *on* the
  degree-4 protocol floor (≈0.064) → failure is protocol-limited, not topology-limited; deployed topology
  uses the full degree budget, link-success ≈ 1.0. Pareto is **flat** here (no D/E levers) — w does not
  move the point — which is *why* the operating-point config exists.
- **operating_point_v1:** a live front. Raising w marches the converged point from a degenerate corner
  into a good basin; **w = 5 is the sweet spot** (F = 0.427, D = 175, E = 0.48). At the operating point
  **D ∝ E** (both ride n_tx = 1/link_success), so the "front" is a *curve* in C/D/E space — shown in all
  three projections.
- **Training dynamics:** an early violent reorganization (grad norm 7.6e4 → ~1e3 in the first ~5 steps,
  D collapsing 267 → 96) then a slow glide to the converged operating point — i.e. the run sweeps from a
  high-cost interior point down-left toward the frontier.

**分析 (incl. the honest w = 0 corner — now multi-seed).** The w = 0 arm trains on reliability ONLY
(delay/energy unweighted). On `operating_point_v1` it is **degenerate and strictly dominated** at the
single-init operating point (D = 6287, E = 17.3, ≈36× the cost of w > 0; F = 0.730). A **5-init seed band**
(`result/w0_seed_band`, F4.5) sharpens this into the honest, stronger claim:
- the **cost blow-up is robust** — D and E are **26× larger** at the multi-seed mean (rel-only mean D/E vs
  the w = 5 optimized arm);
- the **reliability degradation is NOT a robust separation** — the rel-only F band overlaps the optimized
  band, because rel-only is **bimodal / init-unstable** (σ_init = 0.21: 4/5 seeds collapse to F ≈ 0.65–0.74,
  one escapes to F = 0.18), whereas the cost-weighted arm is tight (σ_init = 0.02).
So the correct headline is **"the coupled cost terms are a stabilizing regularizer that removes the
catastrophic init-variance of reliability-only training"** — not the weaker "rel-only degrades F." This
parallels the §2 emission story (cost terms collapse σ_init the way emission does). **Reporting rule:**
w = 0 is the *reliability-only ablation corner* (evidence that coupling is necessary), never an achievable
operating point; report the robust 26× cost blow-up and the σ_init collapse, not a mean-F separation.
**Artifacts:** `result/production_report_paperenv`, `result/production_report_operating_point`,
`result/w0_seed_band` (F4.5).

### 1.2 Variable-N transferability

**目的.** Prove ONE checkpoint deploys across two orders of magnitude in N, and quantify the honest cost
vs. a scale-specific expert (no overclaim).

**设置.** N-randomized training (log-spaced ladder 100–10000); evaluation across the ladder; a
from-scratch N=10000 expert as the anchor; eval Q = 21.

**过程.** Evaluate the single checkpoint at each N (F/C/D/E); train the from-scratch N=10000 expert;
compare both at N=10000 on shared scenes → **transfer regret**.

**预期结果.** F is flat across scale (production single-model panel: F = 0.064–0.066 across N = 100 →
10000; the dedicated N-randomized planner: F ≈ 0.0776 → 0.0700). Transfer regret **+0.0062 ± 0.0002** —
"transfers without divergence but underperforms a scale-specific expert by a small, measured margin."

**分析.** The message-passing scorer is size-invariant by construction; the small measured regret is the
precise, honest cost of N-generalization — neither "generalizes perfectly" nor a bare "transfers."
**Artifacts:** `result/planner_paperenv`, `result/scale_anchor_n10000`, `result/scale_transfer`, the
scalability panel of 1.1.

### 1.3 Advantage scope — *where* learning is valuable

**目的.** Prove the learned topology beats heuristics in a specific, characterizable region, is
gracefully at-parity (floor-limited) elsewhere, with honest seed error bars and Monte-Carlo ground truth.
This *scopes* the main claim instead of overselling it.

**设置.** 27-cell grid (density {100,200,300} × profile {toy, near, hard} × coupling {0,10,20} dB), 5
seeds, N = 600; floor/headroom/gap per cell; off-grid interpolation cells; the 3 robust d100 cells
re-checked under MC (1000 trials). An area-confound control re-runs density-300 at N = 1800 (area fixed).

**过程.** (1) Per cell, train a short GNN + score the heuristic set through the same constructor +
evaluator; classify by headroom and gap. (2) 5-seed re-run → gap mean ± σ, 2σ robustness gate. (3) Area
control. (4) MC band on the robust cells.

**预期结果.** ~9 **robust** GNN-advantage cells, all sparse (density 100), gap +0.08…+0.11 (≈13–18×
σ_seed); the density-200 boundary fragile/parity; density-300 robustly floor-limited; gap grows with
sparsity and interference coupling; **density (not map area)** is the driver; and the advantage
**survives MC** (gap_q +0.10 → gap_mc +0.07–0.09 on 3/3 cells).

**分析.** Physical mechanism: the GNN advantage lives where candidate links sit on the channel-reliability
**cliff** (sparse) and where edge choice has **global interference-coupling** consequences; it vanishes
when density saturates the graph (links all on the plateau → the problem reduces to protocol sampling).
The honest negative — 9 robust cells, not 18 — is a feature: it *scopes* the claim and directly feeds the
deployment gate (§3.1). **Artifacts:** `result/advantage_map`, `result/advantage_montecarlo`,
`result/advantage_map_ic_extremes`, `result/advantage_map_area_control`, figure
`result/envelope_figures/envelope_gap_bands.png`.

---

## §2 — Emission-grounded recurrence stabilization (STANDOUT RESULT, Contribution C3)

**目的.** Prove a one-feature carried-reliability **emission** eliminates the catastrophic pure-memory
collapse of a graph-coupled recurrent scorer, isolate *which* factor is responsible, and explain *how*.
This is the chapter's most novel ML finding and the workshop main result.

**设置.** Frozen streaming spec; the legacy `production_training_v1` physics (the regime where pure
memory collapses; paper-env physics does not — a stated boundary). Arms: full / no_graph / filter /
no_graph_filter / filter_nomem (the emission × graph-coupling 2×2 + a recurrence control); 3 scene × 3
init seeds; σ_init decomposition; a training-time mechanism probe; replication across 4 cells.

**过程.** (1) σ_init (init variance) per arm. (2) 2×2 isolation: emission on/off × graph on/off, plus
filter_nomem (emission, no recurrence). (3) Mechanism probe: instrument the graph-GRU gate inputs and
gate-weight gradients per epoch, full vs. filter, on the collapse init. (4) Replicate on additional cells
(vary profile, density, coupling).

**预期结果.** σ_init drops **~30–39×** iff emission is on, **independent of graph coupling** (filter ≈
no_graph_filter, paired p = 0.86) **and of recurrence** (filter_nomem stable). Probe: without emission the
recurrent gate-input scale blows up (joined_std 0.35 → 11.8) and the gate gradient vanishes to 0 (stuck
in the collapsed init, F ≈ 0.96); with emission the input stays bounded (~2) so the gradient survives and
the model escapes (F ≈ 0.57). Replication: severe at sparse+coupled (39×, 27×), absent at dense d200
(no collapse) — regime-specific. Temporal mean-F is null (memory does not improve accuracy — only
stability).

**分析.** Mechanism: the emission re-grounds the recurrent state in a calibrated [0,1] physical
observation each frame, **bounding the recurrent input distribution → keeping gradients flowing →
escaping collapse**. The contribution generalizes to any "recurrent model + differentiable physics
simulator" loop and is honestly scoped to the high-difficulty operating points where collapse occurs.
**Artifacts:** `result/emission_2x2_isolation`, `result/emission_replication_d100_c0`,
`result/emission_probe_collapse_regime`, `result/fine_stage_*`, figures `c3_mechanism.png`,
`c3_probe_divergence.png`.

---

## §3 — Operationalizing the envelope: gating vs. a governed generalist (Contributions C4 + C5)

> **Section thesis.** §1.3 *characterized* the operating envelope (where learning wins / is at parity).
> A deployed system must serve the *whole* envelope, not one cell. There are two routes, and we
> demonstrate both: **selection** — keep experts/heuristics and *route* among them at runtime from
> estimable context (3.1); and **unification** — train *one* generalist that works everywhere, using
> gradient governance to remove negative transfer (3.2). 3.3 contrasts them. The advantage map (§1.3) is
> the shared substrate both routes consume.

### 3.1 Domain-aware deployment gating (the *selection* route)

**目的.** Prove the seed-banded advantage map *is* a deployment gating table: from estimable per-frame
context the system selects the GNN-vs-heuristic policy and flags floor-limited regimes — with no
training at deploy time.

**设置.** Gating table derived from the seed-banded map; forward-only streaming demo over several
density × coupling scenarios; the confidence profile (ic/iw) flagged **configured-not-measured**
(unobservable at deploy).

**过程.** Per frame: estimate density (N / bbox-area from beacon positions) and an interference proxy
(mean edge SINR), look up the gate, route the policy, and validate by checking the realized heuristic F
lands in the gate's predicted band.

**预期结果.** Density recovered accurately (e.g. est 99.7 vs true 100); routing correct (d100 →
USE_GNN, d200 → marginal, d300 → heuristic suffices); **15/15 frames in the predicted band**.

**分析.** The gate is honestly 2-D (density × coupling, both estimable); the third axis (confidence
profile) is not deployment-observable, so it is a configured assumption with a sim-to-real caveat. The
gate turns §1.3's *characterization* into a runtime *decision*. **Artifact:** `result/gating_demo`.

### 3.2 Governed mixture generalist (the *unification* route)

**目的.** Prove ONE gradient-governed mixture policy spans the envelope — matching per-cell experts
in-grid, generalizing to held-out density, interpolating off-grid — and *explain then repair* the
negative transfer a naive mixture exhibits (a diagnostic-then-targeted-fix methodology, not just "a
mixture works").

**设置.** 27-cell mixture training; stratified leave-one-density-out (LOCO); off-grid interpolation
cells; the per-density gradient-conflict diagnostic; PCGrad vs. GradNorm governance (the per-env
evaluator-coupling harness fix is parity-tested).

**过程.** (1) Naive round-robin mixture → retention by density + LOCO + off-grid. (2) Gradient diagnostic:
per-density gradient norm + pairwise cosine at the shared init (a GO/NO-GO test). (3) Governed training
(PCGrad and GradNorm), re-measure retention.

**预期结果.** Naive mixture under-fits (d100 retention 0.34) — negative transfer. **Diagnostic: positive
cosines (d100·d200 = 0.822 → no directional conflict) but magnitude imbalance (d100 the smallest gradient
norm, 802 vs. d200 1709)** → the diagnostic *recommends GradNorm, not PCGrad*. Governed retention rises
to **≈1.00 (d100) / 0.91 (d200)**, matching the LOCO ceiling; LOCO held-out 0.85–0.99; off-grid 12/12
positive.

**分析.** The naive "negative transfer" is a round-robin **ordering artifact** — even no-op PCGrad
(= balanced multi-density batching) recovers the ceiling; GradNorm adds a magnitude-rebalance edge
*consistent with the diagnostic's own recommendation*. The contribution is the **measure-the-conflict →
choose-the-matching-tool** loop. **Artifacts:** `result/mixture_generalization`,
`result/mixture_governed_gradnorm`, `result/mixture_governed_pcgrad`, figures `generalization.png`,
`governed_retention.png`.

### 3.3 Selection vs. unification (synthesis)

**目的/分析.** Contrast the two routes as a deployment-design choice, honestly:
- **Gating (selection):** zero deploy-time training, interpretable (you can read off *why* a policy was
  chosen), but needs the expert/heuristic set on hand and only as good as the estimable context (2-D).
- **Governed generalist (unification):** one model, no runtime routing, interpolates off-grid; costs an
  offline governance step and hides the per-cell rationale.
A one-paragraph recommendation: gating where the context is reliably estimable and interpretability
matters; the governed generalist where off-grid coverage and a single artifact matter. This subsection is
the system-level payoff of having *characterized* the envelope in §1.3.

---

## §4 — Ablations, baselines, robustness (support)

**目的.** Defend the design choices, bound the claims, and validate the training currency against ground
truth.

**设置/过程/预期.**
- **Currency faithfulness (moved here from the old "surrogate honesty" section; measured, F4.1):** the same
  deployed topology under mean-field vs. quenched vs. MC (3 cells, 2000 MC trials). Mean-field is
  **16–99× optimistic** (and at easy cells underflows to ~0, i.e. *qualitatively blind*); **quenched tracks
  MC at 1.3–4.4×**, best (1.31×) at the operating-point-like cell, looser at hard low-confidence cells. →
  conclusions are read in the quenched/MC currency; mean-field is a training-only convenience.
  **Artifacts:** `result/currency_faithfulness`, `validate_*_montecarlo`, `result/advantage_montecarlo`.
- **w = 0 multi-seed band (done, F4.5):** 5 init seeds at w = 0 vs. w = 5 on `operating_point_v1`. Result:
  cost blow-up **26× robust**; reliability **not** robustly separated — rel-only is bimodal/init-unstable
  (σ_init = 0.21) while the cost-weighted arm is tight (σ_init = 0.02). Headline = "cost terms remove
  init-variance" (see §1.1). **Artifact:** `result/w0_seed_band`.
- **Temporal mean-F null (F4.2, single-init snapshot):** static / no_memory / full / filter mean F (0.235,
  0.342, 0.325, 0.265) all fall **within the §2 σ_init ≈ 0.20 envelope** — i.e. no arm shows a mean-F win
  outside init noise, consistent with "memory ⇒ stability (§2), not accuracy." Honest caveat: a *paired*
  multi-seed test (not just single inits) is needed to assert a strict null. **Artifact:**
  `result/s2_temporal_null`.
- **Fair stream heuristics (F4.3, honest result):** on identical held-out frames/shadow with matched
  information, the deployed GNN (filter arm, F = 0.265) is at **parity with the strong channel-rank
  heuristic (0.233)** and **−62% vs the carried-reliability heuristic (0.688)**. So the advantage is not an
  unfair-information artifact — but the *clean* win over the best heuristic lives in §1.3's advantage map
  (sparse cells, +0.08…0.11), **not** in this streaming comparison. Do not claim a blanket streaming win.
  **Artifact:** `result/s2_temporal_null` (same runs).
- **HARQ retry cap & ms latency:** structural/retransmission n_tx caps; D 33.5 ms → 11.2 ms under w_D×10
  (inside the 10–100 ms V2X budget); E saturates near the protocol minimum (honest PARTIAL).
- **Reproducibility:** byte-identical-when-off opt-in discipline, parity tests (incl. the per-env
  evaluator-coupling fix used in §3.2), seed-variance decomposition, currency scanner.

**分析.** These bound what is and is not claimed and make the headline results (§1–§3) auditable. The
currency-faithfulness check is what licenses every absolute F elsewhere; the temporal-null and
fair-heuristic checks pre-empt the two most obvious reviewer objections.

---

## Related-work positioning (why "domain adaptation" is NOT used)

A dedicated Related-Work/Discussion subsection: classic UDA presumes labelled source + unsupervised
target; here the analytic differentiable evaluator supplies the objective in **every** domain, so the UDA
premise does not arise. The correct framings are **operating-envelope characterization** (§1.3),
**domain-aware gating** (§3.1), **domain generalization via mixture + governance** (§3.2), and **scale
generalization** (§1.2). Adversarial DA (DANN/MMD/CORAL) is discussed only to explain why it is
inapplicable (and was measured null as FiLM conditioning). This turns a potential weakness into a
methodological point.

---

## Section → contribution → artifact map (v2)

| § | proves | contribution | key artifact |
|---|---|---|---|
| §0 | setup, currency rule, floor reference, baselines | (substrate) | protocol_floor_table |
| §1.1 | production model + coupled C/D/E Pareto | C4 | production_report_{paperenv, operating_point} |
| §1.2 | variable-N transfer + regret | C4 | scale_transfer, planner_paperenv, scale_anchor_n10000 |
| §1.3 | where learning wins (seed bands + MC) | C2 | advantage_map, advantage_montecarlo |
| §2 | emission stabilization + mechanism | C3 | emission_2x2_isolation, emission_probe_collapse_regime |
| §3.1 | domain-aware gating (selection) | C4 | gating_demo |
| §3.2 | governed mixture generalist (unification) | C5 | mixture_governed_{gradnorm, pcgrad}, mixture_generalization |
| §4 | currency faithfulness + ablations/robustness | support | validate_*_montecarlo, s2/fine/de_ablation/seed_variance |

**Main-paper order:** §0 → §1 (1.1 → 1.2 → 1.3) → §2 → §3 (3.1 → 3.2 → 3.3) → §4.
**Workshop order:** condensed §0 / §1.3 → §2 (with §3.2 as the optimization-side companion).
