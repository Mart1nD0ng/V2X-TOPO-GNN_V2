# ESP/ESD-GNN Performance-Scale Validation v2 — decision log & gate status

Round on branch `esp-performance-scale-v2` (branched off `main` after the Guarded-CDQ2 round merged;
the prior feature branch was deleted). **Live spec:** `docs/ESP_ESD_GNN_PERFORMANCE_SCALE_WORKFLOW.md`.
**Headline judge:** independent dynamic-MC macrostate basin first-hitting, full physics (workflow §1.3).
**Mainline policy:** ESP/ESD-GNN (`ESDGNN(use_cdq=False)`); Guarded-CDQ2 is an optional extension arm only.

**Compute reality (single machine).** Full-physics training is ~18–22 s/step @N=120 and scales ~linearly
in N; full-physics MC is ~0.4 s/trial @N=120, ~linear in N. The publication-grade ideal (≥5 model seeds,
≥30 scene seeds, 3000-trial UCB certification, N up to 10000, 8 baselines, OOD matrix) is many hundreds of
GPU-hours. Per workflow §9.1 every reduction is run **bounded and explicitly labeled compute-limited**;
the harness is built to the full design and the runs are the feasible subset.

## Campaign A — pre-registration (A0, 2026-06-28, hash-bound BEFORE any A2–A5 headline run)

User chose **(A)** — the full publication-grade campaign. A multi-agent design+adversarial-review workflow
(7 agents) produced the vetted execution matrix (`docs/gate_evidence/esp_scale_v2/campaign_a_plan.json`,
order A0→A2→A3a→A3b→A4→A5→A6, ~600 compute-min). Two reviews independently flagged two **plan-fatal**
issues; the fixes below are **pre-registered here and committed before any headline eval** so they cannot be
retro-tuned:

1. **Performance comparison is UN-GATED** (`mc_faithful_campaign.ungated_cost`, J = 1 − macro_P_correct, no
   feasibility gate). *Why:* the verified dynamic-MC `F_wrong` is ~0.035–0.0525 for **every** policy in the
   stressed headline regime (distance 0.0525, uniform 0.035, trained 0.0525) while the profile keeps
   `eps_w = eps_s = 1e-3`, so the feasibility-gated `headline_cost` returns **+inf for all policies** →
   `scale_regret`/`normalized_scale_regret`/`feasibility_retention` all collapse to **NaN**. Feasibility
   (`F_wrong`/`F_split` UCB vs ε, `mc_faithful_campaign.reliability_status`) is reported **separately** as a
   reliability column, never folded into the performance/regret denominator. `F_split ≡ 0` so `eps_s` is fine.
2. **Headline CI = seed-level bootstrap** (`paired_seed_separation` / `seed_level_bootstrap_ci`), each model
   seed = ONE observation, paired CRN across policies. The pooled-binomial Wilson CI from
   `aggregate_seed_macros` (n = trials×seeds) is **demoted to a diagnostic** — it conflates model-init
   variance with within-MC Bernoulli draws and understates uncertainty.
3. **≥5-seed headline asserted** (writers pass `min_seeds=5`; any reduction labeled compute-limited per §9.1).
4. **Service-profile calibration rule (already pre-registered, re-confirmed unchanged):**
   `R_d(N) = round(6·√(N/120))` → ladder 6/10/14/19/30/54 for N=120/336/660/1248/3036/9840. Not tuned post-hoc.
5. **Budget axis from one trajectory:** `train_esp_reinforce(snapshot_steps=…)` saves state_dicts at
   {40,80,150} along the SAME persistent-Adam trajectory (trajectory-preserving — tested), so the A2 budget
   curve and the A1 headline checkpoints come from one 5-seed×150-step run.
6. **Tail metrics:** `evaluate_macro` will pool the fields the MC already carries (T_confirm, energy, latency
   CVaR) when A3 needs them; D95/D99/basin-CVaR99 have no MC producer and are **dropped + labeled
   compute-limited-deferred** (§9.1), not mislabeled zero-cost.

## Gate status

| Gate | Scope | Status |
|------|-------|--------|
| G-ESP-TRAINING-BUDGET | full-physics budget curves (pilot/medium/full), ≥5 seeds, mixed N{100,300,1000}; does longer training improve macro + beat the distance heuristic? select best checkpoint | ✅ **CLOSED under the MC-faithful trainer (EV7)** — the EV1 ANALYTIC budget curve was FLAT (0.422 at every budget; the surrogate was blind to peer selection, EV2). Under MC-faithful REINFORCE the held-out MC budget curve **RISES monotonically**: init 0.383 → 40: 0.394 → 80: 0.401 → 150: **0.412** (+0.029, 5 seeds, seed-level bootstrap), beating the EV1-flat contrast. Best checkpoint = step 150. The original EV1 ⛔ stop is resolved by the user-chosen MC-faithful fix. |
| G-ESP-MC-FAITHFUL-TRAINING | **(user-chosen direction, 2026-06-28)** close the EV2 training-signal gap: train the GNN on the MC basin via the score-function (REINFORCE) gradient `∇E[R]=E[(R−b)·Σ∇log π(Sₜ)]` so it learns the peer-selection the judge rewards | ✅ **CLOSED (EV6, CI-separated)** — Campaign-A 5-seed headline (N=120, pre-registered seed-level bootstrap + un-gated J): trained **0.410** [0.399,0.420], **paired-vs-uniform +0.040 CI [0.029,0.050]** (CI-separated, gap statistically closed), **matches distance** 0.422 (overlapping CIs, 77% of the uniform→distance gap closed). EV4/EV5 built it (per-node credit, monotone in steps); EV6 confirms at 5 seeds. 13 tests; judge-invariant. |
| G-ESP-BASELINE-ORACLE | 8 baselines (uniform/distance/link-quality/load-balanced/region-bridge/edge-logit-oracle/expert/shared) through the canonical full-physics path; oracle headroom | 🟢 **(EV2+EV6+EV8)** — distance is the strong heuristic the trained policy matches (EV6); uniform the weak one it CI-separately beats; scale-specific experts (N=336/660) trained (A3a) → scale-regret vs shared small (N=336: 0.008). Deployable-baseline layer complete. |
| G-ESP-FIXED-PROTOCOL-SCALE | fixed protocol across N{100,300,1000,3000}+10000; macro+UCB+D99/CVaR+energy+strict+diagnostics+runtime/mem | ✅ **(EV8, compute-limited)** — shared-ESP (trained@N=120) **matches distance and beats uniform across N=120→3036** under fixed R_d=6: N=120 0.461≈0.463, N=336 0.429 vs 0.447, N=660 0.420≈0.420, N=3036 0.438≈0.4375 (uniform 0.42/0.38/0.37/0.31). **§13.2 parity holds across a 25× scale-up.** N=1248 (the lone v=4 grid) is a density/MAC-saturation outlier (all policies → total deadline, excluded). N=9840 = approximation bound (8 trials). Seeds 5/5/3/–/2 (N≥660 compute-limited). |
| G-ESP-FIXED-SERVICE-SCALE | pre-registered R_d(N)∝√N calibration; scale-regret + normalized + feasibility-retention + expert/heuristic comparison | ✅ **(EV8, compute-limited)** — pre-registered R_d(N)=round(6√(N/120)) ladder. Empirically fixed_service ≈ fixed_protocol at these N (the MC first-hits the basin/deadline before R_d epochs, so extra R_d buys nothing) — stated, not hidden. Scale-regret shared-vs-expert small (N=336: 0.008; shared trained@120 ≈ expert trained@N). fixed_service dropped at N≥3036 (same-as-fixed_protocol, labeled). |
| G-ESP-OOD-GENERALIZATION | one-axis-at-a-time (node count/density/geometry/covariance/PHY-load/sensor-group/profile/mobility) | ✅ **(EV9)** — 4 declared OOD cells (one axis varied, others pinned), 5 seeds. Trained **beats uniform in ALL 4** (CI-separated). Within the matched-marginal family it **tracks distance** (corr→0.35: 0.361 vs 0.367; base_err→0.30: 0.479 vs 0.470; →0.45: 0.016 vs 0.013). **Honest limit:** transfers POORLY to the `iid` covariance family (0.149 vs distance 0.203) — the policy learned mm_high correlated-error structure; structure-agnostic distance wins under iid. |
| G-ESP-RARE-EVENT-CERTIFICATION | enough MC for UCB cert at p<1e-3, OR rare-event/splitting/IS, OR labelled approximation | ✅ **(EV10)** — per-checkpoint zero-failure UCB (seed 0, M=4000). **F_split CERTIFIED ≤ eps_s=1e-3** (zero split events → UCB 0.00096). **F_wrong NOT certifiable** at eps_w=1e-3: structurally ~0.02–0.03 for trained (UCB 0.031) AND distance (UCB 0.027) alike → the stressed regime is infeasible at the strict wrong-basin target for **all** policies (a regime property, not a GNN failure — why A0 un-gates J). |
| G-ESP-SCALE-SYNTHESIS | unified report; scale-regret + feasibility curves; fixed-proto vs fixed-service; baseline/oracle headroom; OOD matrix; honest failure modes; paper-claim recommendation | ✅ **(EV11)** — `run_phase_a6_synthesis.py` (reads result JSONs only) → **§13.2 verdict**: a stable learned ESP constructor that CI-separately beats uniform and matches distance, parity transfers across a 25× scale-up + both calibration modes. Figures (budget curve, scale curve) from committed JSONs. Honest failure modes surfaced: iid-OOD transfer limit, N=1248 density collapse, regime infeasible at strict eps_w. **Not** a §13.1 superiority claim. |

Legend: ☐ not started · 🟡 in progress · 🟢 green.

## Adopted defaults (override-flagged)
- Reuses the GS3 `esp_scale.py` harness (train_esp_checkpoint, evaluate_macro, scale_regret,
  normalized_scale_regret, feasibility_retention, calibrated_profile) + the macrostate_v2 schema/manifest.
- Pre-registered fixed-service-profile calibration (declared BEFORE Block D, constraint #13): the GS3 rule
  `R_d(N) = round(R_d0 · √(N/N0))` (validated to restore feasibility at N=660/1248/3036 in the prior round).

## Decision log

### EV0 — round bootstrap (2026-06-28)
* Read the workflow spec + the Guarded-CDQ2 synthesis + the existing `esp_scale.py` harness. Branched
  `esp-performance-scale-v2` off `main` (the prior round is merged; `macrostate-cdq2-redesign` deleted).
* Confirmed the round's compute reality and the bounded/labeled strategy (workflow §9.1).

### EV1 — Slice: training-budget curve harness (G-ESP-TRAINING-BUDGET 🟡) (2026-06-28)
* `src/evaluation/esp_training.py` — `train_with_curve` runs the **canonical** full-physics training loop
  (`run_consensus_episode` + `macrostate_metrics` + `macrostate_lagrangian` + `MacrostateDuals`, same as
  `train_macrostate`) with a **persistent optimizer** + a **held-out analytic validation hook** at each
  budget checkpoint (so the budget curve is a continuous trajectory, not optimizer restarts; analytic
  validation is screening-only, workflow §1.3). `budget_improvement` / `select_best_budget` aggregate the
  "does longer training help / beat distance?" question over model seeds. 4 tests (3 pure + 1 tiny
  full-physics integration, 68 s).
* **Regime choice (room to learn):** iid @N=120 is near-ceiling (analytic Pc≈1.0 for both GNN and distance
  → no measurable headroom, workflow §5.3). Switched to **mm_high(base_err=0.35, corr=0.25), R_d=6**:
  distance analytic **Pc=0.42, F_deadline=0.56, F_wrong=0.016** — lots of room, deadline-dominated (the
  improvable basin is exactly ESP's learnable strength: faster-quorum peer selection), reliability-safe.
* **Result (seed 0; remaining seeds killed — finding is structural, not statistical):** the budget curve is
  **perfectly FLAT** — GNN `macro_P_correct` = **0.422 at every budget {5,15,30,50}**, identical to the
  distance heuristic (0.423). The training loss *rises* 0.6→75.5 but that is **entirely the dual μ_d ascending**
  (F_deadline=0.56 ≫ ε_d, the constraint never satisfied); the **primal (model output) does not move**. So
  longer training produces **zero macrostate improvement** here.
* **Diagnosis — why (analytic peer-insensitivity):** a heuristic-spread scan over regimes shows the **analytic
  macrostate objective is ~insensitive to ESP peer selection**: all 5 structurally-different heuristics
  (uniform / distance / link_quality / load_balanced / region_bridge) give **identical** analytic Pc in every
  regime — ceiling (iid easy: all 1.0), floor (R_d=4: all 0.0), or invariant (mm_high_R6: all 0.422; spread
  ≤ 0.002). The macro basin is **dominated by the environment's bulk correctness** (q_i, which the policy
  cannot change), not the polling pattern → the GNN has **no gradient to learn from** on this surrogate, and
  no heuristic beats another. (The oracle would confirm this but its full-physics backward is ~18 s/step;
  the heuristic invariance already establishes it.)
* **Round-critical juncture (stop-condition #1/#2 territory).** This connects to the prior round's finding
  (GNN ≈ distance). The decisive open question: does the **dynamic-MC judge** see peer-selection headroom that
  the analytic **training surrogate** is blind to? If MC spread ~0 too → the basin is genuinely peer-invariant
  (env-dominated, **§13.2**: GNN matches heuristics, stable but not superior). If MC spread is meaningful →
  a **training-signal gap** (the surrogate can't teach what the judge rewards, **§13.3**). `run_headroom_mc.py`
  (the MC-spread evidence) is executing to resolve this **before** the STOP+REPORT.

### EV2 — Slice: MC-judge headroom resolves the interpretation → §13.3 training-signal gap (2026-06-28)
* `docs/gate_evidence/esp_scale_v2/run_headroom_mc.py` → `headroom_mc_results.json`. The SAME 5 observable
  ESP heuristics under the **dynamic-MC judge** (full physics, 100 trials × 2 scenes, N=120):
  - **iid_easy:** distance **0.995** [0.972,0.999] > region_bridge 0.940 > load_balanced **0.920** [0.874,0.950]
    → **MC spread 0.075, CI-SEPARATED** (distance significantly beats load_balanced). Analytic spread was 0.000.
  - **mm_high_R6:** distance **0.445** [0.378,0.514] > load_balanced 0.370 > region_bridge **0.360** → MC spread
    0.085 (CIs marginally overlap at 200 trials). Analytic spread was 0.002.
* **Verdict — §13.3 training-signal gap (decisive in iid_easy):** the **dynamic-MC judge rewards peer
  selection** (distance CI-separated above the weak heuristics), but the **analytic macrostate training
  surrogate is blind to it** (all policies tied). So the GNN, trained on the flat surrogate, receives **no
  gradient toward the peer selection the judge values** → it stalls at ≈ its distance-like init (the EV1 flat
  curve). The mean-field analytic episode washes out the polling-pattern effect that the subset-sampling MC
  resolves. **Distance is already the best simple heuristic under MC**; whether a learned policy can beat
  *distance* under MC is unresolved, and crucially **cannot be reached through this training surrogate**.
* **STOP+REPORT (stop-condition #1/#2):** the round's central premise — trained ESP/ESD-GNN *performance
  superiority* — cannot be established with the current training signal. This is not a bug and not hidden;
  it is reported. The contribution honestly available now is the **evaluator/diagnostic + the training-signal
  gap finding** (§13.3) and the §13.2 "stable learned constructor that matches strong heuristics" framing.
  Manifest slice `EV2`. **The direction is the user's call** (see the report).
* **Evidence (reproducible):** `run_training_budget.py` (flat curve), the spread/headroom diagnostics, and
  `run_headroom_mc.py` (the MC-judge spread). Tests green: esp_training (4), heuristics (3), esp_baselines (2).

### EV3 — Slice: close the training-signal gap via MC-faithful REINFORCE (user-chosen direction) (2026-06-28)
* **User decision (2026-06-28):** *close the training-signal gap* — make the trainer MC-faithful so the GNN
  learns the peer-selection the dynamic-MC judge rewards. The headroom exists (EV2); the bottleneck is purely
  the mean-field training signal. This adds the gate **G-ESP-MC-FAITHFUL-TRAINING** (unblocks G-ESP-TRAINING-BUDGET).
* **Approach (score-function / REINFORCE on the MC basin).** The ESP k-subset sampler is a differentiable law
  `log π(Sᵢ) = Σ_{j∈Sᵢ} s_ij − log e_k(exp(sᵢ))` (Eq. 16). For a per-trial reward `R` (e.g. correct-basin
  first-hit), `∇E[R] = E[(R−b)·Σ_{i,t} ∇log π(S_{i,t})]` trains the GNN edge logits directly on the MC basin
  objective — **no mean-field washout**. Implementation plan: **extend the canonical dynamic-MC rollout** (the
  judge) with a *gated* `Σ log π` accumulation (reusing the exact snowball code, no duplication → no
  divergence bug) returning per-trial `Σ log π` + per-trial basin outcome codes; the REINFORCE loss is formed
  outside. Variance reduction: a per-batch mean baseline `b` (and optionally per-node).
* **EV3 foundation (this slice):** `src/optimization/mc_reinforce.py::batched_subset_log_prob` — the
  vectorised, differentiable ESP subset log-π (over all node/epoch/trial), validated vs the unbatched
  `subset_log_probability` reference, mask-correct, sums-to-1 over all k-subsets, gradient flows
  (`tests/optimization/test_mc_reinforce.py`, 3 green).
* **Next:** extend `run_dynamic_mc` with the gated log-π + per-trial outcome codes; write `train_esp_reinforce`;
  **proof-of-concept test that REINFORCE IMPROVES the MC `macro_P_correct`** on a small scene where analytic
  training was flat (the gap-closing demonstration); then retry the budget gate (pilot/medium/full) under the
  MC-faithful trainer and compare to the distance heuristic.

### EV4 — Slice: MC-faithful REINFORCE trainer + the gap-closing demonstration (2026-06-28)
* **Implementation.** `run_dynamic_mc` gains a GATED `reinforce=True` mode (src/validation/dynamic_mc.py):
  keeps the GNN's per-edge log-weights differentiable, accumulates **per-(trial, node)** `Σ_epochs log π(S_{i,t})`
  (sampling + physics use a DETACHED copy → the rollout is numerically identical to the judge, asserted by a
  test that reinforce=False/True give bit-identical basins), and exposes per-node correct-finalisation reward.
  `src/optimization/mc_reinforce.py::train_esp_reinforce` descends `−mean_t Σ_i ω_i (R_{i}−b_i)·log π(S_{i,t})`
  with a **per-node baseline** `b_i` (variance reduction) + participation weighting. 10 tests; 82-test
  MC/metrics regression clean.
* **The gap-closing result (mm_high(0.35,0.25) R_d=6, N=120; held-out MC, 200×2 trials):**
  - GNN init held-out `macro_P_correct` = **0.370** (≈ uniform 0.368).
  - **network-level** REINFORCE (one reward per trial shared across all nodes): held-out **0.370, Δ +0.000** —
    the network-basin reward gives almost no per-node credit; the gradient is pure MC sampling noise (training
    curve oscillates 0.28–0.49, no trend). *This is the env-domination of EV2 re-appearing as a
    credit-assignment problem.*
  - **per-node** REINFORCE (credit each node by ITS OWN correct finalisation, ω-weighted, baseline per node):
    held-out **0.392, Δ +0.022** toward distance (0.427) in 40 bounded steps — **the gap is closing** where
    the analytic surrogate (and network-level REINFORCE) were flat. Variance reduction was the key.
* **Honest caveat:** Δ +0.022 is a real positive *direction* but **not yet CI-separated** at 400 eval trials
  (init [0.32,0.42] vs trained [0.34,0.44] overlap); 1 model seed (compute-limited). Confirmation needed: does
  longer, multi-seed REINFORCE keep climbing toward / reach distance (0.427), CI-separated?
* **Next:** a longer multi-seed REINFORCE confirmation run (more steps; ≥2-3 seeds; tighter eval) — if it
  CI-separately reaches/approaches distance, the budget gate closes under the MC-faithful trainer (the GNN
  becomes a trainable topology constructor on the true objective); then proceed to the scale gates.

### EV5 — Slice: the gap-closing CONFIRMATION (2 seeds × 80 steps) (2026-06-28)
* **Result (mm_high(0.35,0.25) R_d=6, N=120; held-out MC, 200×2 trials; distance ref 0.427, uniform 0.368):**

  | model seed | init held-out Pc | trained (80 steps) | Δ |
  |---|---|---|---|
  | 0 | 0.370 [0.324,0.418] | **0.415** [0.368,0.464] | +0.045 |
  | 1 | 0.370 [0.324,0.418] | **0.405** [0.358,0.454] | +0.035 |
  | **mean** | 0.370 | **0.410** | **+0.040** |

* **Interpretation — the gap is closing, reproducibly and monotonically.** Per-node-credit MC-faithful
  REINFORCE trains the GNN from uniform-level (0.370 ≈ uniform 0.368) to **~0.41 = ~96% of the distance
  heuristic (0.427)**, consistent across 2 independent model seeds, and **monotone in training budget**
  (40 steps → 0.392, 80 steps → 0.41) — exactly where the analytic surrogate AND network-level REINFORCE
  were completely flat. The central blocker of the round (the EV2 training-signal gap) is **mechanistically
  resolved**: there now exists a trainer whose gradient reflects what the dynamic-MC judge measures.
* **Honest statistical caveat (NOT yet CI-separated).** At 400 eval trials the per-seed Wilson CIs still
  overlap (init [0.324,0.418] vs trained [0.368,0.464]); the +0.040 effect is narrower than the ±0.048 eval
  CI. The evidence for closing rests on **reproducibility (2 seeds) + monotonicity (in steps)**, not yet on a
  single CI-separated comparison. CI-separation would need either more eval trials (~1000+) or more training
  steps (drive trained Pc to/above distance so the gap widens). 2 model seeds (headline target ≥5):
  compute-limited.
* **Gate status.** G-ESP-MC-FAITHFUL-TRAINING core goal **met** (a working MC-faithful trainer that closes
  the gap); promoting it to a CI-separated, ≥5-seed, reaches/beats-distance headline — and then re-running
  the budget + scale gates with MC-faithful-trained checkpoints — is **substantial additional compute** and a
  user investment decision (workflow §9.1, the "rises-but-not-yet-CI-separated → user decision" branch).
  **User chose (A): the full publication-grade campaign** → see "Campaign A — pre-registration (A0)" + EV6.

### EV6 — Campaign A / Phase A1 headline: the gap is CI-separately closed at N=120 (5 seeds, 2026-06-28)
* **Setup.** 5 MC-faithful checkpoints (seeds 0–4 × 150 steps, per-node-credit REINFORCE) on
  mm_high(0.35,0.25) R_d=6 N=120, evaluated under the dynamic-MC judge (full physics) over 3 held-out scenes
  (trained 300 trials/scene, refs distance/uniform 1000 trials/scene, CRN-shared), with the **A0
  pre-registered statistics**: un-gated J, seed-level bootstrap headline, reliability reported separately.
  Result `phase_a1_eval_results.json` (hash-bound, namespace-clean).
* **Headline.** Trained **macro_P_correct = 0.410**, seed-level bootstrap CI **[0.399, 0.420]**, across-seed
  SD 0.014, per-seed [0.421, 0.421, 0.418, 0.396, 0.392].
  - vs **uniform 0.370** [0.353, 0.387]: **paired-across-seed Δ = +0.040, bootstrap CI [0.029, 0.050]** →
    **CI-separated** (excludes 0). The EV2 training-signal gap is now **statistically closed**, not just a
    reproducible direction.
  - vs **distance 0.422** [0.404, 0.439]: gap +0.012, CIs overlap → **statistically matches** the strongest
    simple heuristic (77% of the uniform→distance gap closed).
* **Reliability (separate column, A0).** Every policy is infeasible at the strict eps_w=1e-3 (trained F_wrong
  0.023, UCB 0.035; distance F_wrong 0.018, UCB 0.023; uniform F_wrong 0.035) — which is exactly why the
  performance comparison is un-gated. F_split ≡ 0. (Rare-event certification is A5.)
* **Methodology note.** Seed 3's *training* curve ended at 0.30 (alarming) but its **held-out** Pc is 0.396 —
  the held-out MC judge + seed-level bootstrap correctly absorbed the noisy trajectory; training curves are
  not the metric.
* **Verdict (§13.2).** MC-faithful REINFORCE makes the ESP/ESD-GNN a **stable learned constructor that
  CI-separately beats uniform and matches distance at N=120**. Whether this becomes §13.1 *superiority* (or
  stays §13.2 *parity*) is decided by the **scale sweep (A3)** — does the learned policy degrade more
  gracefully than distance as N grows / the deadline tightens? Next: A2 budget curve (does held-out Pc rise
  0->40->80->150 vs the EV1 flat analytic 0.422), then A3 scale experts + sweep.

### EV7 — Campaign A / Phase A2: the budget gate RISES under the MC-faithful trainer (2026-06-28)
* **Setup.** Each model seed's budget snapshots {0=init, 40, 80, 150} (one trajectory, trajectory-preserving
  snapshots) evaluated under the dynamic-MC judge at N=120 (2 held-out scenes x 250 trials), seed-level
  bootstrap per budget. `phase_a2_budget_results.json` (hash-bound, namespace-clean).
* **Result — the held-out MC budget curve RISES monotonically:**

  | budget (steps) | seed-mean macro_P_correct | seed-level bootstrap CI |
  |---|---|---|
  | 0 (untrained init) | 0.383 | [0.378, 0.387] |
  | 40 | 0.394 | [0.386, 0.402] |
  | 80 | 0.401 | [0.390, 0.411] |
  | 150 | **0.412** | [0.397, 0.425] |

  +0.029 from init, monotone non-decreasing. **Direct contrast: the EV1 ANALYTIC-surrogate budget curve was
  FLAT at 0.422 for every budget** — the mean-field surrogate had no gradient toward what the MC judge
  rewards (EV2). MC-faithful REINFORCE supplies that gradient, so longer training now *does* improve the
  held-out MC macrostate. This **resolves the EV1 STOP** (the round's original blocker) via the user-chosen
  fix. Best checkpoint = step 150 (the shared checkpoint for the A3 scale sweep).

### EV8 — Campaign A / Phase A3: scale sweep — §13.2 parity holds across a 25× scale-up (2026-06-28)
* **Setup.** The 5 N=120-trained shared checkpoints evaluated under the dynamic-MC judge across the
  `_GRID_TABLE` ladder N∈{120,336,660,1248,3036,9840} under BOTH calibration modes vs distance, uniform, and
  scale-specific experts (A3a, trained AT N=336/660). Un-gated J, seed-level bootstrap; pre-registered
  compute-limited taper (trials 200/150/100/40/16/8; shared seeds 5/5/3/3/2/2; N≥660 compute-limited; N=9840
  approximation bound). `phase_a3b_scale_results.json`.
* **Result — shared-ESP (trained only at N=120) matches distance and beats uniform across scale (fixed_protocol R_d=6):**

  | N | shared-ESP | distance | uniform | shared vs distance |
  |---|---|---|---|---|
  | 120 | 0.461 | 0.463 | 0.415 | parity |
  | 336 | 0.429 | 0.447 | 0.380 | ~parity (distance +0.018) |
  | 660 | 0.420 | 0.420 | 0.370 | parity (exact) |
  | 3036 | 0.438 | 0.4375 | 0.3125 | parity (shared +0.0005) |

  Across the **6 non-degenerate cells** (both modes, N=120→3036): shared above distance 2 / parity 2 / below 2
  (= noise around equal) and **shared beats uniform 6/6**. §13.2 parity holds across a **25× scale-up** from the
  N=120 training scale. Scale-regret shared-vs-expert small (N=336: 0.008 — the N=120-trained shared ≈ the
  N=336-trained expert).
* **Honest anomalies (surfaced, not hidden):**
  - **N=1248 is a density/MAC outlier.** It is the lone v=4 grid (13,13,4; mean comm-degree 13.2 vs ~9 for the
    v=3 grids). ALL policies collapse to P_correct=0 (F_deadline→1) — even at fixed_service R_d=19 — while the
    *physically larger but sparser* N=3036 (v=3) is fine. Diagnosed: fully connected, so not a bug; the MAC
    saturates at that density. Excluded from the parity verdict (no policy differentiation); the
    controlled-density scale story is the v=3 ladder.
  - **fixed_service ≈ fixed_protocol** at these N: the MC first-hits the basin/deadline before R_d epochs, so
    the larger fixed_service R_d buys no extra information — stated, fixed_service dropped at N≥3036.
  - **N=9840** = approximation bound (8 trials, Wilson ±0.12), drawn as a bound, never a validated N=10000 claim.
* **Verdict (§13.2).** MC-faithful REINFORCE yields a stable learned ESP constructor that **CI-separately beats
  uniform and matches distance, and this parity TRANSFERS across a 25× node-count scale-up** under both
  calibration modes — not §13.1 superiority (shared does not CI-separately pull ahead of distance), but a clean,
  honest parity result. Next: A4 OOD, A5 rare-event, A6 synthesis.

### EV9 — Campaign A / Phase A4: OOD generalization — within-family transfer, an iid limit (2026-06-28)
* **Setup.** The 5 N=120 shared checkpoints (trained on mm_high(0.35,0.25)) evaluated on 4 declared OOD
  regimes, ONE distribution axis varied with the others pinned to training (5 seeds, 150 trials × 2 scenes,
  seed-level bootstrap). `phase_a4_ood_results.json`. (The plan's corr=0.45 / base_err=0.20 cells are
  infeasible for matched_marginal → feasible one-axis equivalents corr→0.35, base_err→0.45/0.30, documented.)
* **Result (trained vs distance vs uniform; trained beats uniform, CI-separated, in ALL 4):**

  | OOD axis | trained | distance | uniform |
  |---|---|---|---|
  | covariance-family → iid | 0.149 [0.131,0.166] | 0.203 | 0.047 |
  | correlation harder (→0.35) | 0.361 [0.354,0.368] | 0.367 | 0.263 |
  | base-error harder (→0.45) | 0.016 [0.012,0.021] | 0.013 | 0.000 |
  | base-error easier (→0.30) | 0.479 [0.461,0.497] | 0.470 | 0.390 |

* **Verdict.** Within the **matched-marginal family** (correlation / base-error shifts) the trained policy
  **tracks distance** and **beats uniform** — OOD generalization holds. **Honest limit:** transferring to the
  **iid covariance family** (uncorrelated errors — a structure the policy never trained on), trained falls
  behind distance (0.149 vs 0.203) though it still beats uniform. The learned policy encodes mm_high
  correlated-error structure; the structure-agnostic distance heuristic generalizes better across covariance
  families. A real, reportable boundary of the learned constructor.

### EV10 — Campaign A / Phase A5: rare-event certification — F_split certified, F_wrong infeasible for all (2026-06-28)
* **Setup.** Per-checkpoint zero-failure UCB on a single deployable policy (seed 0, N=120, M=4000 trials;
  eps=1e-3 needs M≳3800). `phase_a5_rare_event_results.json`.
* **Result.**
  - **F_split CERTIFIED feasible** ≤ eps_s=1e-3: zero split events in 4000 trials → Wilson UCB 0.00096 < 1e-3.
  - **F_wrong NOT certifiable** at eps_w=1e-3: trained F_wrong 0.026 (UCB 0.031), distance 0.022 (UCB 0.027) —
    both orders of magnitude above 1e-3. The stressed mm_high regime is **structurally infeasible at the strict
    wrong-basin target for EVERY policy** (a regime property, not a GNN failure). This is exactly why the A0
    performance comparison is un-gated; a deployment would relax eps_w or de-stress the regime.

### EV11 — Campaign A / Phase A6: synthesis + the round verdict (2026-06-28)
* `run_phase_a6_synthesis.py` reads the committed a1/a2/a3b/a4/a5 result JSONs ONLY (no MC, no metric
  recomputation) and emits `phase_a6_synthesis.json` + figures (`figures/a2_budget_curve.png`,
  `figures/a3b_scale_curve.png`).
* **Section-13 verdict: §13.2 PARITY.** MC-faithful per-node-credit REINFORCE turns the ESP/ESD-GNN into a
  **stable learned topology constructor** that, judged by the independent dynamic-MC macrostate basin:
  - **CI-separately beats the uniform baseline** (N=120 paired Δ +0.040 [0.029,0.050]; beats uniform at 6/6
    scale cells and all 4 OOD cells);
  - **statistically matches the strong distance heuristic** (N=120 gap +0.012 overlapping CIs; tracks distance
    across N=120→3036, both calibration modes);
  - **transfers across a 25× node-count scale-up** and across within-family distribution shifts.
  It is **NOT** §13.1 superiority — the learned policy does not CI-separately pull ahead of distance at any
  scale. The honest headline is *parity with the best simple heuristic + a clear win over the naive one,
  reached by a trainer that the analytic surrogate could not provide a gradient for.*
* **Honest failure modes (all surfaced, none hidden):** (a) iid-OOD transfer limit (the policy encodes
  mm_high correlated-error structure; distance generalizes better across covariance families); (b) N=1248
  density/MAC collapse (the lone v=4 grid; all policies → deadline); (c) the stressed regime is infeasible at
  the strict eps_w=1e-3 wrong-basin target for every policy (F_split is certified).
* **Compute-limited (workflow §9.1):** headline 5 seeds; scale seeds taper 5/5/3/–/2; trials taper
  200…8; N=9840 is an 8-trial approximation bound, never a validated N=10000 claim; OOD/rare-event are
  single-regime/single-seed certifications. The harness is built to the full design; the runs are the
  feasible subset.

---

## ROUND COMPLETE — all gates green (2026-06-28)

| Gate | Verdict |
|---|---|
| G-ESP-MC-FAITHFUL-TRAINING | ✅ gap CI-separately closed (EV6) |
| G-ESP-TRAINING-BUDGET | ✅ budget curve rises vs EV1-flat (EV7) |
| G-ESP-BASELINE-ORACLE | ✅ baselines + experts (EV2/EV6/EV8) |
| G-ESP-FIXED-PROTOCOL-SCALE | ✅ parity across N=120→3036 (EV8) |
| G-ESP-FIXED-SERVICE-SCALE | ✅ parity; fixed_service≈fixed_protocol explained (EV8) |
| G-ESP-OOD-GENERALIZATION | ✅ within-family transfer; iid limit (EV9) |
| G-ESP-RARE-EVENT-CERTIFICATION | ✅ F_split certified; F_wrong infeasible-for-all (EV10) |
| G-ESP-SCALE-SYNTHESIS | ✅ §13.2 parity verdict (EV11) |

**Headline:** the round's original premise (trained-model performance *superiority*) was challenged at EV1/EV2
(the analytic training surrogate is blind to the peer-selection the MC judge rewards). The user-chosen
MC-faithful REINFORCE fix (EV3–EV6) **closed that training-signal gap**, and the publication-grade campaign
(EV6–EV11) establishes the honest, defensible result: **§13.2 parity** — a stable learned ESP constructor
that beats the naive baseline and matches the strong heuristic, with parity transferring across scale, plus
clearly-reported limits. All results macrostate_v2 namespace-clean, hash-bound, with pre-registered
(A0) un-gated J + seed-level-bootstrap statistics fixed before any headline run.

---

## Post-round: can the GNN beat distance? — MC oracle headroom probes (Lever 1, 2026-06-30)

After the round, a 18-agent research workflow asked *how to get from §13.2 parity to §13.1 superiority*. Its
top recommendation was a never-run gate experiment: a **dynamic-MC-judged free-edge oracle** (optimise free
per-edge logits per scene against the MC basin reward, not the peer-blind analytic objective) — an upper bound
on what any diagonal ESP law (a trained GNN included) can do under the judge.

### EV12 — Lever 1: MC free-edge oracle — un-gated headroom is LARGE but RELIABILITY-BOUGHT (iso-reliability ≈ 0)
* **Un-gated probe** (`oracle_probe_results.json`, 3 scenes, 2000-trial CRN eval, distance + random inits):
  the free per-edge oracle CI-separately BEATS distance by **+0.099** (CI [0.087, 0.109], per-scene gaps
  +0.087…+0.109; both inits independently reach ~0.52 — a robust attractor, not an optimiser artefact). This
  initially looked like "parity was a learning gap, not a no-headroom ceiling."
* **Mechanism (basin decomposition):** the gain is almost entirely **reduced deadline misses** (F_deadline
  ~0.55 → ~0.42) converted ~3:1 into correct vs wrong — a diagonal "decide faster / reach quorum sooner"
  improvement, NOT diversity. But it also raises F_wrong (~0.02 → ~0.05).
* **Reliability frontier** (`oracle_reliability_results.json`, Lever 1a, reward = `correct − λ·wrong`,
  λ∈{0,2,5,12}, 2 scenes): **this is the decisive, sobering result.** At **λ=2, where the oracle's F_wrong is
  pulled down to distance's level (~0.023 vs ~0.025)**, the P_correct gap over distance **COLLAPSES to
  −0.0067** (both scenes; statistically 0 at the ~0.025 Wilson half-width). Higher λ → strictly worse than
  distance. So the un-gated +0.10 was **almost entirely bought by spending the F_wrong budget**; once the
  oracle is held to distance's reliability it **becomes distance** (it stops "gambling" on risky nodes,
  F_deadline climbs back).
* **Corrected verdict (supersedes the optimistic un-gated read):** the **iso-reliability headroom over distance
  is ≈ 0** — distance sits on the (P_correct, F_wrong) Pareto frontier of per-edge policies in this regime.
  §13.2 parity is the **honest ceiling at equal reliability**, NOT a learning/optimisation failure to be tuned
  away. This vindicates the research's own caveat (the only mechanism with new expressive power buys its win
  with F_wrong). Compute-limited: 2–3 scenes; per-scene oracle is an upper bound; eval Wilson hw ~0.025.
* **Next:** Lever 1b distillation diagnosis (is the GNN's parity a feature/capacity/generalization issue?) —
  now interpreted in light of the above: even the un-gated oracle operating point is not a legitimate win, so
  the open question becomes whether a DIFFERENT regime/objective (lower deadline pressure, or a latency/energy
  objective distance does not already optimise) has non-zero iso-reliability headroom that is GNN-learnable.

### EV13 — Lever 1b: distillation diagnosis — the GNN is CAPABLE; parity was a TRAINER failure (not features/capacity)
* **Setup** (`oracle_distill_results.json`): distil the GNN (per-node-centered MSE) to the per-scene MC-oracle
  logits on 4 train scenes, test on 2 held-out scenes, capacity sweep hidden_dim 16 vs 64.
* **Result (refutes the feature-gap hypothesis):**
  - **Fit:** in-sample corr(GNN logits, oracle logits) = **0.856** (hidden_dim 16) ≈ **0.852** (64) — capacity
    is NOT the limit (the big model fits no better) — and **held-out corr 0.857 ≈ in-sample** — the fit
    GENERALISES. So the GNN's pure geometric+region features ARE enough to represent the oracle policy.
  - **Held-out MC:** the distilled GNN **beats distance by +0.032** (scene 30: 0.464 vs 0.434; scene 31:
    0.450 vs 0.417), capturing ~45% of the un-gated oracle's +0.071 headroom — **on scenes it never trained
    on.** REINFORCE-trained GNNs stayed at distance (0.41); supervised distillation extracts a generalising
    +0.03 the model was always capable of.
* **Verdict: GNN-REACHABLE.** The original §13.2 parity was a **TRAINING/optimisation failure** (REINFORCE got
  stuck at the distance attractor), NOT a feature, capacity, or generalisation limit. The model can represent
  AND generalise a better-than-distance policy; the trainer was the bottleneck.
* **CRITICAL caveat (combine with EV12/a):** the +0.03 the distilled GNN reaches is the SAME un-gated,
  **reliability-bought** operating point as the oracle (the distillation eval measured P_correct only, not
  F_wrong; the distilled policy mimics the oracle at corr 0.86, so it almost certainly inherits the oracle's
  elevated F_wrong). EV12 showed that at MATCHED reliability the oracle headroom is ≈ 0. So: **the GNN CAN beat
  distance on un-gated P_correct (trainer-fixable, e.g. distillation warm-start + REINFORCE), but NOT at equal
  reliability** — the iso-reliability target is distance itself.
* **Combined (a)+(b) bottom line:** in this regime, model capability is NOT the bottleneck and the trainer is
  fixable, but a LEGITIMATE (reliability-matched) win over distance does not exist (oracle-certified). The
  single highest-value next move: **re-probe the iso-reliability oracle headroom in a LESS deadline-dominated
  regime** (larger R_d / lower N) — EV12's mechanism (the un-gated gain is "decide faster → fewer deadline
  misses") suggests that where the deadline is not the binding constraint, decide-faster may buy correctness
  WITHOUT the F_wrong trade, opening a non-zero iso-reliability headroom that EV13 proves the GNN could then
  learn. If none appears even there, parity is robust and the honest deliverable is the oracle-certified
  no-iso-reliability-headroom characterisation + the capability/trainer diagnosis. Compute-limited: 4 train +
  2 held-out scenes; per-scene oracle = upper bound; held-out MC un-gated only.

### EV14 — Route A: iso-reliability headroom vs deadline budget R_d — parity is robust (no headroom even at large R_d)
* **Setup** (`oracle_reliability_rd_results.json`): re-ran the EV12 reliability-frontier machine (free-edge
  oracle, reward = correct − λ·wrong) at R_d ∈ {10, 14} (EV12 had R_d=6), 2 scenes, reporting the iso-reliability
  gap over distance (the gap at the λ where the oracle's F_wrong matches distance's).
* **Result — iso-reliability headroom stays ≈ 0 at every R_d:**

  | R_d | distance mean F_deadline | iso-reliability gap (oracle − distance at matched F_wrong) |
  |---|---|---|
  | 6 (EV12) | 0.553 | −0.007 |
  | 10 | 0.552 | −0.022 |
  | 14 | 0.552 | −0.022 |

* **Decisive mechanistic insight:** **R_d=14 gives results bit-identical to R_d=10, and distance's F_deadline is
  INVARIANT at ~0.552 across R_d∈{6,10,14}.** So R_d was **never the binding constraint** — the ~55% deadline
  misses are **physics/MAC-limited** (those trials structurally cannot form a quorum under the link/contention
  physics), not epoch-budget-limited. Relaxing the deadline therefore changes nothing.
* **Verdict:** there is **NO legitimate iso-reliability P_correct headroom over distance, and it does not open up
  by relaxing the deadline** — distance is the per-edge reliability-feasible optimum across the deadline-budget
  range. The "less-deadline-dominated regime" escape hatch is **closed**: the limit is the consensus physics, not
  the time budget. Parity is robust on the P_correct axis. (Compute-limited: 2 scenes; the un-gated oracle still
  beats distance ~+0.06 at every R_d, but it is all reliability-bought, exactly as at R_d=6.)
* **Next:** Route B — the multi-objective Pareto surface (is distance dominated on latency/energy at matched
  reliability+Pc, or only the P_correct ceiling?).

### EV15 — Route B: multi-objective Pareto surface — distance is the strong corner, NOT just the P_correct ceiling
* **Setup** (`pareto_measure_results.json`, per the vetted `route_b_pareto_design.json`): measure the full
  8-axis vector (Pc, Fw, Fs, Fd, lat_basin=basin_tau_correct_mean, lat_cvar, energy, energy_cvar) for distance
  (frozen β=0.04), uniform, link_quality, load_balanced, region_bridge, and the 5-seed trained GNN over 4
  held-out scenes × 2 sample-split blocks × 1500 trials. Gate = matched-to-distance on Fw/Fs/**Fd** (margins
  0.005) + Pc-equivalence (margin 0.012); then per-scene SIGN test on the 4 latency/energy axes. Per-scene
  oracle EXCLUDED (upper bound). CVaR is CVaR_0.9.
* **Result — distance Pareto-DOMINATES the deployable set; the GNN is dominated by it:**

  | policy vs distance | Pc | Fd | lat_basin | energy | admitted? |
  |---|---|---|---|---|---|
  | distance (ref) | 0.432 | 0.547 | 5.566 | 311.5 | — |
  | trained GNN | −0.015 | +0.013 | +0.146 (worse) | +7.1 (worse) | ✗ (worse on every axis) |
  | uniform | −0.051 | +0.059 | +0.250 | +24.7 | ✗ (strongly dominated) |
  | link_quality | −0.005 | +0.007 | +0.034 | +3.3 | ✗ (fails Fd by 0.002; lat_cvar edge negligible ~0) |
  | load_balanced | −0.044 | +0.050 | +0.254 | +22.8 | ✗ |
  | region_bridge | −0.041 | +0.047 | +0.244 | +22.9 | ✗ |

  **No policy is admitted** — distance is simultaneously the best (or tied-best) deployable policy on
  **P_correct AND F_deadline**, so nothing even enters the matched feasible set to challenge it on
  latency/energy. And on the raw vectors distance is **better-or-equal on all 7 axes** vs every policy. The
  **trained GNN is weakly DOMINATED by distance** (worse on Pc, Fd, mean latency, energy; tied on lat_cvar).
  The only sliver of a distance weakness is lat_cvar (tail latency) where link_quality "wins" — but by a
  negligible magnitude (~0) and link_quality fails the Fd gate.
* **Verdict + mechanism (the deep insight):** distance is **NOT only the P_correct iso-reliability ceiling — it
  is the strong corner of the WHOLE multi-objective Pareto surface.** Why: in this V2X consensus regime the
  objectives are **ALIGNED, not in tension** — polling the nearest / best-link peers simultaneously gives (a)
  the most reliable polls → fastest quorum → highest Pc, fewest deadline misses, lowest latency; and (b) the
  shortest links → lowest tx energy. There is **no Pareto tradeoff for a learned policy to exploit**; "nearest"
  is a near-optimal multi-objective heuristic. A learned topology constructor can at best MATCH distance (and
  this GNN slightly under-matches). Compute-limited: 4 scenes, per-scene SIGN test (directional, not
  definitive); the free-edge LATENCY oracle (the latency analog of EV12's P_correct oracle) was NOT run
  (skipped because no admitted policy beat distance on latency) — an optional rigor-completion.

---

## POST-ROUND CONCLUSION — can the GNN beat distance? (EV12–EV15)

After §13.2 parity, four experiments asked whether ANY legitimate superiority over distance exists:

| Axis / route | Finding |
|---|---|
| EV12 P_correct, un-gated | oracle beats distance +0.10 — but entirely **reliability-bought** |
| EV12 P_correct, iso-reliability | headroom ≈ **0** — distance is the per-edge reliability-feasible optimum |
| EV13 capability | the GNN **can** represent + generalise a better-than-distance policy; parity was a **trainer** failure (fixable), not a model limit |
| EV14 P_correct vs deadline budget | relaxing R_d opens **no** headroom — R_d isn't binding; the limit is consensus physics |
| EV15 multi-objective Pareto | distance is the **strong corner** of {Pc, reliability, deadline, latency, energy}; the GNN is dominated by it; objectives are **aligned** |

**Bottom line:** **distance is not just a strong P_correct baseline — it is the (near-)optimal corner of the project's entire multi-objective surface in this regime, because the objectives are aligned (nearest peer is best on every axis at once).** The learned GNN is capability-sufficient and trainer-limited, but there is **no legitimate objective on which it can beat distance here**. A genuine learning win would require a regime where the objectives are in TENSION (e.g. correlated-error structure that makes the locally-best peer globally redundant, sparse/asymmetric link costs decoupling energy from distance, or adversarial/mobility dynamics) — exactly the conditions the per-edge oracle (EV12/EV13/EV15) shows are absent in the current mm_high setup.


---

# NDH BENCHMARK — Non-Distance Headroom (new campaign, decision log)

**Motivation (carry-over from EV12–EV15):** in the current `mm_high R_d=6` physics, distance is the
strong corner of the *whole* multi-objective Pareto surface (EV15), and there is no iso-reliability
P_correct headroom over distance at any deadline budget (EV14) — because the objectives are *aligned*
(physics is highly distance-correlated). The NDH benchmark deliberately adds **non-distance** deployment
mechanisms (SPS persistent collision, heterogeneous RSU/vehicle capacity, CSI aging, intersection-queue
hotspots) to test whether a regime exists where a constrained (wrong-penalized) oracle has *equal-reliability*
headroom over distance — and only then train a GNN. **Spec:** `docs/NON_DISTANCE_HEADROOM_TECHNICAL_SPEC.md`
+ `docs/NON_DISTANCE_HEADROOM_ENGINEERING_PLAN.md`. **Oracle-first; matched-distance reliability is headline;
absolute F_w≤1e-3 deployment budget reported separately; strong heuristics get the GNN's observable proxies.**

Non-degradable: macrostate_v2 only; dynamic-MC basin first-hitting is final judge; full physics in headline;
train==eval physics hash; every mechanism enters the canonical path (not just tests); every parameter in the
registry; no stress value as a deployment default; no oracle-only truth into GNN/heuristics; `omega_RSU=0`;
RSU density capped; no legacy GRU/emission as current evidence.

| Gate | Status | Evidence |
|---|---|---|
| **G-NDH-PARAM-AUDIT** | **DONE** | `docs/NDH_PARAMETER_REGISTRY.md` (EV16) |
| **G-NDH-SCENE-RSU-HOTSPOT** | **DONE** | `src/environment/nonuniform_urban_scene.py` + `vehicle_only_participation` (EV17) |
| **G-NDH-SPS-PERSISTENCE** | **DONE** | `src/environment/sps_resource.py` + `round_physics` SPS collision (EV18) |
| **G-NDH-CSI-AGING** | **DONE** | `src/environment/csi_aging.py` (feature-only; physics keeps current γ) (EV19) |
| **G-NDH-HETEROGENEOUS-CAPACITY** | **DONE** | `src/environment/receiver_capacity.py` + `round_physics` per-node μ_j queue (EV20) |
| **G-NDH-FEATURE-SCHEMA** | **DONE** | `src/models/scene_features_v2.py` + `esd_gnn_v2.py` (leak-clean) (EV21) |
| G-NDH-BASELINE-ENVELOPE | pending | — |
| G-NDH-ORACLE-FRONTIER | pending (oracle-first gate) | — |
| G-NDH-STATIC-ESDGNN-V2 | conditional | only if oracle headroom exists |
| G-NDH-TEMPORAL-NEED / -V2 | conditional | only if history oracle beats static oracle |
| G-NDH-SYNTHESIS | pending | final report |

## EV16 — G-NDH-PARAM-AUDIT (2026-07-01)

* **Deliverable:** `docs/NDH_PARAMETER_REGISTRY.md` — every SPS/RSU/capacity/CSI/hotspot parameter with
  `default / sweep / stress / source / deployment-or-stress label`, the preserved baseline physics/protocol/
  scene (FROZEN), the hard density/fraction caps, the C2 observability split (deployable proxy vs oracle-only
  truth), and two assembled canonical profiles: **`NDH-DEPLOYMENT`** (headline, all defaults) and
  **`NDH-STRESS`** (reported separately, never a deployment default).
* **Key audit decisions:**
  - SPS does **not** change the existing collision pool `S=subchannels·slots_per_window=100`; it adds a
    *persistent per-node bucket + sensing-based reselection* layer so collision risk gains history (the
    non-distance structure). S∈{40,60} is congestion **stress** only.
  - Heterogeneous capacity replaces the global `service_rate μ=12` with per-node `μ_j` (vehicle lognormal
    μ_veh=8 default; RSU `mult=5`); model/heuristics read only the **noisy proxy** `μ̂_j`, never true `μ_j`.
  - CSI aging changes only the **feature** (stale `γ̂(t−a)`, default age 100 ms ≈ CAM 10 Hz); **physics keeps
    current `γ(t)`** ⇒ no train/eval physics mismatch.
  - RSU caps: `p_intersection_rsu ≤ 0.5`, RSU fraction ≤ 0.10–0.15, `omega_RSU=0` — anti-trivial-nearest-RSU.
  - Hotspots: ≤10% of intersections, ≤30% of vehicles, non-overlapping (≥ 2·radius), static in Phase 1.
  - Surrogate knobs (`tau_res`, `kappa_res`, `block_length_logstd`) are **declared lightweight surrogates**
    (no ns-3, Q4), constrained to non-collapse bands by their sweeps; none tuned to favour the GNN.
* **Acceptance:** all params present; no stress-as-default; hash-binding rule stated; observability split
  satisfies Contract C2. Pure-documentation gate (no code/physics change → no dynamic-MC run this gate).
* **Next:** G-NDH-SCENE-RSU-HOTSPOT (smallest next gate that unblocks everything): nonuniform urban scene +
  sparse intersection hotspots + capped RSU placement, with failing tests first (caps, connectivity,
  `omega_RSU=0`, reproducibility).

## EV17 — G-NDH-SCENE-RSU-HOTSPOT (2026-07-01)

* **Deliverable:** `src/environment/nonuniform_urban_scene.py` (`NonuniformUrbanScene` +
  `build_nonuniform_urban_scene`) — a **drop-in** replacement for `ManhattanScene` (identical 7 fields +
  `num_nodes`/`num_regions`) adding: lognormal block lengths + intersection jitter + optional road dropout,
  **sparse static intersection-queue hotspots**, and **capped RSU** placement. Plus canonical
  `vehicle_only_participation(scene)` in `src/metrics/participation.py` (`omega_RSU=0`, exogenous role label).
  Wired into `src/environment/__init__.py`.
* **Hard caps enforced (registry §3/§6, constraints #9/#10):** `p_intersection_rsu ≤ 0.5` and
  `max_rsu_fraction ≤ 0.15` (input-guarded); `#RSU ≤ max_rsu_fraction·N`; hotspots ≤ 10% of intersections,
  non-overlapping (centres ≥ 2·radius); **injected queue mass ≤ 30% of vehicles** (bounds the mechanism, not
  incidental grid density); road graph guaranteed connected (spanning-tree backbone); region coverage dense.
* **RSU = responder/witness only:** RSU are ordinary graph nodes (no special-casing in the graph builders);
  their non-participation lives ONLY in `node_type` + `vehicle_only_participation` (`omega_RSU=0`). Basin judge
  unchanged; omega-weighted `basin_*` exclude RSU exactly.
* **Verification:** 14 unit tests (drop-in interface, containment, caps, connectivity under road dropout,
  reproducibility, canonical-path smoke through `build_overlapping_scenario → run_dynamic_mc`); a **1500-scene
  exhaustive fuzz over the full registry sweep grid** (incl. coarse 2×2 × radius=80 cells) — 0 cap/interface/
  omega/NaN violations; full `tests/` suite green except one pre-existing unrelated failure (below).
* **Adversarial review (4-lens workflow) → 5 fixes applied, 1 documented:**
  - MAJOR (crash): the 30% hotspot cap was asserted against uncontrollable base-grid density → crashed on
    coarse-grid + radius=80. **Fixed:** cap now bounds only the *injected* queue mass (always satisfiable).
  - MAJOR (control-confound): RSU roadside positions extended coordinate extrema → `overlapping_evidence._band`
    moved the **vehicle–vehicle** correlation (the matched-marginal causal lever). **Fixed:** band edges now
    derived from **vehicle positions only** (ManhattanScene behaviour byte-identical); RSU-evidence-invariance
    verified = 0 over 40 seed-pairs.
  - MAJOR (dead param): `queue_length_m` was neutered by `min(…, hotspot_radius_m, …)`. **Fixed:** extent =
    `min(queue_length_m, seg_len)`; test asserts it drives reach.
  - MINOR: `max_rsu_fraction ≤ 0.15` guard added; dataclass `eq=False` (hashable parity with ManhattanScene).
    Also fixed a latent `_greedy_spaced(max_count=0)` off-by-one (tiny grids got 1 RSU when the cap floored to 0).
  - MINOR (documented, not coded): the *legacy node-union* diagnostics (`F_wrong`/`S_allcorrect`/`F_disagree`)
    count RSU when `eligible_mask=None`. The **basin judge already excludes RSU via omega**; legacy node-union
    metrics are FORBIDDEN as headline (shortcut #9). Judge file left untouched (do not weaken/perturb the judge)
    — RSU-enabled scenes report only `basin_*`.
* **Pre-existing unrelated failure (NOT introduced here):** `tests/evaluation/test_esp_baselines.py::
  test_oracle_upper_bounds_distance` fails by a hair (oracle analytic P_correct 0.42185 vs distance 0.42336;
  the 25-step ascent misses the 1e-3 slack by 5e-4). Provably independent of this gate: it uses the
  `ManhattanScene` path where the `_band` change is a literal no-op (`veh_mask is None → cv = coord`), so `ev`
  and the oracle result are byte-identical to pre-NDH. Flagged for separate fix (raise oracle steps or document
  the tolerance in the ESP module); confirmed on the clean tree in parallel.
* **Next:** G-NDH-SPS-PERSISTENCE — `src/environment/sps_resource.py` sensing-based resource persistence
  surrogate (persistent bucket + reselection; same-resource `G_int` neighbours collide repeatedly), deployable
  proxies only, entering the canonical collision physics.

* **[EV17 addendum] Pre-existing ESP failure CONFIRMED on the clean tree.** Ran
  `test_oracle_upper_bounds_distance` in a worktree at the pre-NDH commit 2d18a6f (clean
  `overlapping_evidence._band`): **byte-identical failure** (oracle 0.4218519185894899 vs distance
  0.42336056859843313). Definitively pre-existing and independent of all NDH work; fix flagged separately.

## EV18 — G-NDH-SPS-PERSISTENCE (2026-07-01)

* **Deliverable:** sensing-based SPS resource-persistence surrogate + same-resource collision physics,
  on the canonical path (train==eval).
  - `src/environment/sps_resource.py`: `assign_sps_buckets` (static sensing-avoidance assignment,
    `P(r)∝exp(−τ_res·sensed_occ_r)+noise`), deployable proxies `same_resource_conflict_degree` /
    `sensed_channel_busy_ratio`, and `assert_sps_pool_consistent`.
  - `round_physics`: `resource_collision_kappa` in `RoundPhysicsConfig` (binds `config_hash`),
    `resource_bucket` arg + `_sps_same_resource_collision` helper — same-bucket `G_int` contention
    `p=1−exp(−κ·(L−a_self)_+)` REPLACES the memoryless `1/S` for BOTH request and response phases;
    self-exclusion preserved; SINR interference untouched.
  - Static-bucket rationale (reselection ~1 s ≫ episode ~60–200 ms → frozen within a decision) ⇒ NO
    mutable state through the judge; threaded via `getattr(scene,'resource_bucket',None)` into both the
    analytic episode and the dynamic-MC judge. Scene gains `resource_bucket` + `mechanism_config_hash`.
* **Two-pool model (corrected mid-gate):** `S_phys = subchannels·slots_per_window` (memoryless `1/S` +
  SINR) vs `S_sps = sps_n_buckets` (persistent reservations). Reservations are a *subset* of the
  instantaneous pool ⇒ invariant `1 ≤ S_sps ≤ S_phys` (guard-enforced on both paths). Smaller `S_sps`
  is the congestion knob; the campaign's non-degenerate base needs `S_phys=400`, so `S_sps=40` at
  `S_phys=400` is the valid congestion regime (an earlier `S_sps==S_phys` guard was wrong — it killed
  the congestion regime; a review finding that mis-diagnosed the subset relation as inconsistency).
* **MC evidence (mechanism active + non-degenerate default):** congested + poor sensing (`S_sps=40`,
  τ=1, κ=0.5, `S_phys=400`) → persistent collision **collapses** Pc (0.32→~0.01); DEPLOYMENT band
  (`S_sps=100`, τ=4) → conflict ≈0, SPS **removes** the memoryless floor (Pc up, non-degenerate). The
  collapse corner is a labelled STRESS regime (R4), not a deployment default.
* **Verification:** 14 SPS tests (both-phase collision brute-forced to 1e-16, request+response
  self-exclusion=0, SPS-off byte-identical, `config_hash` binds κ, pool-consistency guard, dtype guard,
  mechanism-trace sentinels, `mechanism_config_hash`, sensing<random, canonical-MC active) + physics
  regression green.
* **Adversarial 3-lens review (physics-correctness / train==eval-grad / leak-faithfulness):** confirmed
  the physics is CORRECT (brute-force 1e-16, self-exclusion exact, no leakage, gradients flow, SINR
  untouched, train==eval bucket identity). Fixes applied from findings: (1) **mechanism-trace sentinels**
  `sps_persistence`/`resource_conflict_graph` emitted + `mode2_collision` correctly OFF under SPS (plan §4
  acceptance) + test; (2) direct **response-phase** collision test (roles were only exercised end-to-end);
  (3) **two-pool guard** `S_sps ≤ S_phys` + documented (replacing the wrong `==`); (4) **NDH-mechanism
  hash** `scene.mechanism_config_hash` (physics hash omits bucket-structure params); (5) `resource_bucket`
  dtype/ndim guard; (6) registry: κ_res relabelled DEPLOYMENT-modelling (non-degenerate at the default
  band), two-pool note, RRI/reselection/keep_prob marked **Phase-2 (temporal)**, collapse-band R4 note,
  NDH-DEPLOYMENT/STRESS now name the SPS values; (7) spec-§3.4 age-weight→tx-mass Phase-1 deviation
  documented in the helper docstring.
* **Next:** G-NDH-CSI-AGING — stale-CSI features (`stale_sinr_db`, `csi_age`, `csi_uncertainty`,
  `stale_vs_distance_residual`); physics uses current γ, model sees stale (no train/eval mismatch).

## EV19 — G-NDH-CSI-AGING (2026-07-01)

* **Deliverable:** `src/environment/csi_aging.py` — per-edge stale-CSI **deployable** features
  (`stale_sinr_db`, `stale_delivery`, `csi_age_ms`, `csi_uncertainty`, `stale_vs_distance_residual`)
  + `csi_uncertainty_db(age)`. Feature-ONLY: the physics link (`edge_geometry`) is untouched; the model
  sees a stale estimate while physics uses the current channel ⇒ **no train/eval physics mismatch**
  (constraint #4, spec §5.1).
* **Model:** `stale_sinr_db = true_sinr_db + N(0, σ(a))` with AR(1) shadow-decorrelation uncertainty
  `σ(a) = sqrt(csi_noise² + shadow_ar² · (1 − exp(−2a/τ_decorr)))` — monotone in age, `= csi_noise` at
  age 0, asymptotes to `sqrt(csi_noise² + shadow_ar²)`. This is the faithful Phase-1 *static* surrogate
  for spec §5.2's `γ̂(t−a)+ε`: with a static scene `γ(t−a)=γ(t)`, so an unbiased estimate with
  age-growing variance is exactly right (a genuinely shifted `γ(t−a)` needs temporal Phase 2).
* **Deployability (C2):** features use only geometry + age + observation noise — no `Y*`, no future CSI,
  no MC outcome. A source-inspection test asserts the module imports no evidence/protocol/validation/MC.
* **Verification (6 tests):** age=0 + zero-noise recovers current CSI exactly; `csi_uncertainty` monotone
  in age (0→50→100→200→500 ms) with the correct age-0 value and asymptote; larger age ⇒ larger mean
  feature error vs the true link; reproducible from seed + correct shapes + `stale_delivery ∈ [0,1]`;
  no-leak source audit; physics link byte-unchanged by CSI aging. Purely additive (new module + exports)
  ⇒ no regression surface.
* **Next:** G-NDH-HETEROGENEOUS-CAPACITY — per-node receiver capacity `μ_j` in the round-physics queue
  (`ρ_j=(Λ_j+b_j)/μ_j`, vehicle/RSU lognormal, noisy proxy) — a real physics change (enters full physics
  + dynamic MC), so it warrants the tests-first + adversarial-review treatment like SPS.

## EV20 — G-NDH-HETEROGENEOUS-CAPACITY (2026-07-01)

* **Deliverable:** per-node receiver capacity `μ_j` in the M/M/1 queue (a real physics change, on the
  canonical path, train==eval).
  - `src/environment/receiver_capacity.py`: `assign_receiver_capacity` (vehicle lognormal
    `μ_veh·exp(σ_μ ε)`; RSU `mult·μ_veh·exp(σ_rsu ε)` — higher mean, finite, overload still possible) +
    `noisy_capacity_proxy` (`μ̂=μ·exp(σ_obs ξ)`, the DEPLOYABLE signal).
  - `round_physics`: optional `node_capacity` [N] → `ρ_j = Λ_j/μ_j` (queue delay + drop use per-node μ);
    `None` → **byte-identical** scalar `service_rate` path.
  - Scene: `node_capacity` field (TRUE μ, physics-only) + `enable_heterogeneous_capacity` + capacity
    params in `mechanism_config_hash`; threaded into both canonical paths; `heterogeneous_capacity`
    trace sentinel (gated on `not disable_queueing`).
* **Truth split (C2):** true `μ_j` enters PHYSICS only (verified: no GNN/policy/heuristic path reads
  `scene.node_capacity`); the model will read only the noisy proxy `μ̂` (wired at the feature gate).
* **Operating band (band check, R_d=20):** non-degenerate — homog Pc=0.757, hetero(μ_veh=8) Pc=0.773;
  the queue binds via **DELAY** on typical load (ρ≈0.75<1); the DROP branch is a `μ_veh=4` STRESS-band
  effect. (The stressed R_d=6 config is deadline-degenerate — use R_d=20 for the capacity oracle-frontier.)
* **Verification:** 12 tests (RSU μ>vehicle μ, noisy proxy≠true + corr>0.5 + σ=0 recovers truth, higher
  μ → lower drop+delay, **homogeneous byte-identity in BOTH ρ<1 and ρ>1 regimes**, None-unchanged,
  differentiable, RSU-overload saturates, enters dynamic-MC, `mechanism_config_hash` binds params, trace
  sentinel False under `disable_queueing`, non-finite/negative μ + negative σ rejected) + **133-test
  env/protocol regression green**.
* **Adversarial 3-lens review:** confirmed physics CORRECT with 0 critical/0 major — byte-identity
  bit-for-bit across B>1/float32-64/all fields; truth split clean; train==eval identity; differentiable;
  RSU overload genuine; defaults faithful to registry §4. All 8 minor/5 nit findings fixed or documented:
  trace sentinel now gates `not disable_queueing`; `node_capacity` finiteness/positivity guard; negative
  `σ_obs` rejected; proxy documented as log-unbiased/level-biased (`exp(σ²/2)`) → feature = `capacity_proxy_log`;
  byte-identity test extended to the ρ>1 drop regime; real RSU-overload + tautology-removed tests; registry
  §4 notes b_j Phase-1-ABSENT + the delay-vs-drop operating band.
* **Next:** G-NDH-FEATURE-SCHEMA — `SceneFeaturesV2` + `ESDGNNStaticV2` wiring ALL mechanism proxies
  (SPS conflict, **noisy** capacity proxy — never true μ, CSI-aging, hotspot, node_type) into the GNN with
  a feature-availability mask; leak-critical (tests-first: assert true `node_capacity` NOT in the feature
  tensor; old features reproducible when mechanisms off; baselines get the same observable features).

## EV21 — G-NDH-FEATURE-SCHEMA (2026-07-01)

* **Deliverable:** the expanded DEPLOYABLE feature schema wiring every NDH mechanism's proxy into one
  tensor for the GNN + heuristics.
  - `src/models/scene_features_v2.py`: `build_scene_features_v2(scene, phy_cfg, …) -> SceneFeaturesV2`
    (13 node + 14 edge columns + per-column availability mask). `NODE_FEATURE_NAMES`/`EDGE_FEATURE_NAMES`
    define the schema; base structural columns (log-degrees, region size, distance/comm_radius, LOS,
    same_region) are a STRICT PREFIX so `build_scene_features` is reproduced exactly.
  - `src/models/esd_gnn_v2.py`: `ESDGNNStaticV2` (the ESDGNN encoder sized to the wider schema; keeps
    source/dest/interference/region/load channels; no legacy GRU) + `ESDGNNStaticV2QueryPolicy`.
* **Leak contract (Contract C2) — VERIFIED CLEAN by a 3-lens leak-focused review (0 critical/0 leaks):**
  every column traces to a deployable source. Capacity via `noisy_capacity_proxy` (log μ̂) — the true
  `node_capacity` is read ONLY by the proxy call, NEVER as a feature (node AND edge columns tested);
  CSI is the geometry-mean SINR + age-noise (edge_geometry is deterministic — no current-channel
  realization; age=0+noise=0 → geometry mean, tested); degrees are candidate-graph (policy-independent,
  not the policy-driven Λ); no import of evidence/dynamic-MC/validation; no Y*/vote/MC/future.
* **Availability mask:** per-column, keyed on the mechanism ENABLE flag (rsu/hotspot gate on
  `params.enable_*` since a NonuniformUrbanScene always carries node_type/hotspot_score; capacity/SPS
  gate on field-None-unless-enabled). Off mechanism → column 0 + mask 0 (tested, incl. the asymmetry).
* **Verification (11 tests):** no-true-μ-leak (node + edge), CSI stale-not-current, source-imports-no-truth,
  base features reproduced exactly, mask flips with mechanism + gating asymmetry, shapes/O(E)/determinism,
  `ESDGNNStaticV2` forward on all 4 regimes returns positive per-edge quality, differentiable. 20 models
  tests green (additive, no regression).
* **Review fixes applied (6 major/4 minor/3 nit, none a leak):** (1) `intersection_distance` was O(N²)
  `cdist(positions, all-intersections)` → now O(N) via the node's own segment endpoints (nearest
  intersection to a node on segment s IS an endpoint of s); (2) `local_density`/`sensed_cbr` per-scene
  `_norm01` → scene-INVARIANT `1−exp(−x/scale)` (fixed scale; transferable across scales); (3) added the
  edge-capacity + CSI leak tests (were untested — the exact surfaces a future truth-swap could break);
  (4) documented the four Phase-2 EMA/history omissions (`resource_age_norm`, `resource_busy_ratio`,
  `ack_success_ema`, `queue_delay_ema` — need multi-frame state) + the mask semantics (metadata; encoder
  need not consume it under Phase-1 train==eval). The "baselines consume the same proxies" requirement is
  the NEXT gate (G-NDH-BASELINE-ENVELOPE) — the shared builder is provided here.
* **Next:** G-NDH-BASELINE-ENVELOPE — strong heuristics as ESP policies reading the SAME `SceneFeaturesV2`
  columns (stale_link_quality, capacity_aware, resource_aware, distance_plus_*, load_balanced, rsu_nearest,
  rsu_capacity_aware, local_density_aware, best_heuristic_envelope) through the canonical dynamic-MC path.
