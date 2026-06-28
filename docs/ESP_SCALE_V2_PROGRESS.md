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

## Gate status

| Gate | Scope | Status |
|------|-------|--------|
| G-ESP-TRAINING-BUDGET | full-physics budget curves (pilot/medium/full), ≥5 seeds, mixed N{100,300,1000}; does longer training improve macro + beat the distance heuristic? select best checkpoint | ⛔ **STOP+REPORT** (EV1+EV2): flat training curve + a confirmed **training-signal gap** — the MC judge rewards peer selection (spread 0.075–0.085, CI-separated) but the analytic *training* surrogate is blind to it (spread ≤0.002). The GNN can't be trained to beat heuristics on this surrogate. Round premise (model superiority) challenged — user decision needed. |
| G-ESP-BASELINE-ORACLE | 8 baselines (uniform/distance/link-quality/load-balanced/region-bridge/edge-logit-oracle/expert/shared) through the canonical full-physics path; oracle headroom | ☐ |
| G-ESP-FIXED-PROTOCOL-SCALE | fixed protocol across N{100,300,1000,3000}+10000; macro+UCB+D99/CVaR+energy+strict+diagnostics+runtime/mem | ☐ |
| G-ESP-FIXED-SERVICE-SCALE | pre-registered R_d(N)∝√N calibration; scale-regret + normalized + feasibility-retention + expert/heuristic comparison | ☐ |
| G-ESP-OOD-GENERALIZATION | one-axis-at-a-time (node count/density/geometry/covariance/PHY-load/sensor-group/profile/mobility) | ☐ |
| G-ESP-RARE-EVENT-CERTIFICATION | enough MC for UCB cert at p<1e-3, OR rare-event/splitting/IS, OR labelled approximation | ☐ |
| G-ESP-SCALE-SYNTHESIS | unified report; scale-regret + feasibility curves; fixed-proto vs fixed-service; baseline/oracle headroom; OOD matrix; honest failure modes; paper-claim recommendation | ☐ |

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
