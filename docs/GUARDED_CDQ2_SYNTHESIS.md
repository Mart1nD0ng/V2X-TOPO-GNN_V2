# Guarded-CDQ2 round — final synthesis (G-FINAL-SYNTHESIS)

**Branch:** `macrostate-cdq2-redesign`. **Live specs:** `GUARDED_CDQ2_TECHNICAL_SPEC.md`,
`GUARDED_CDQ2_ENGINEERING_PLAN.md`, `MECHANISM_IDENTIFIABILITY_CONTRACT.md`. **Evidence manifest:**
`docs/gate_evidence/guarded_cdq2/manifest.json` (slices GS1–GS6). **Headline judge throughout:** the
independent **dynamic-MC macrostate basin first-hitting**, full physics (constraint #10) — never a
node-union/product surrogate (constraint #2). This document is the round's capstone; all numbers are
read from the committed result JSONs (constraint #13 — no metric is recomputed here).

---

## 1. The claim this round actually supports

> **The contribution is the characterization and *control* of the validity–liveness trade-off induced
> by diversity-aware consensus sampling — not "CDQ2 improves reliability."**

Concretely, the round establishes a **policy family with a single safety dial**:

1. **CDQ2 is the general query family**; **ESP is its reliability-first specialization at `η = 0`**
   (constraint #3 — ESP is the default).
2. **`η > 0` is a liveness/deadline extension**, not a reliability mechanism. Its benefit is **small and
   conditional**, and in the wrong regime it *raises* the wrong/split basins.
3. **Guarded-CDQ2** turns the dial automatically: it enables `η > 0` **only** with reliability slack AND
   deadline pressure, and falls back to ESP otherwise — so diversity is captured **where safe** and
   never violates the hard wrong/split budget (constraint #4/#5).

The honest through-line of the whole round: **the liveness benefit and the reliability/energy cost of
diversity come from the same mechanism** (CDQ2 polls physically more *distant* peers — measured mean
selected distance 49.2 m → 50.4 m as `η` 0 → 8, GS4). Distant peers give broader reach (faster quorum
under deadline slack) but worse links (more deadline misses when too tight) and more TX energy. So the
benefit is real but narrow, and the right deployment is a *guarded* one.

---

## 2. When to use ESP / CDQ2 / Guarded-CDQ2 (the decision)

| Situation | Use | Why (evidence) |
|---|---|---|
| **Default / unknown / strict reliability budget** (`ε_w` small) | **ESP** | reliability-first; `η=0`. In every regime where reliability has no slack, the guard itself selects ESP (GS5/GS6). |
| **Safety-critical / fail-safe-available** (huge wrong/split cost) | **ESP** | hazard selection → ESP at strict ε (GS6); CDQ2 ineligible via the feasibility gate. |
| **High evidence covariance + tight-but-feasible deadline + reliability slack** | **Guarded-CDQ2** | the guard enables `η` → deadline→correct gain while staying feasible (GS4 R_d=14 mm_high: `macro_F_deadline` 0.100→0.065; GS5 enable regime). |
| **Deadline-critical (high `c_d`) with reliability slack** | **Guarded-CDQ2** | hazard selection → Guarded-CDQ2, B=+1.749 (GS6). |
| **Balanced** | **Guarded-CDQ2** | hazard selection → Guarded-CDQ2, B=+0.165 (GS6). The adaptive choice: gains where safe, ESP elsewhere. |
| **Energy-constrained** | **ESP** | CDQ2 diversity costs ~1.2 % more TX energy (distant peers; 846→856 @N=336, GS6) → energy-weighted B<0 → ESP. |
| **Fixed-η CDQ2 as a standalone deployable policy** | **avoid** | it can violate the wrong/split UCB in high-covariance scenes (GS5 ε=0.05 enable regime: fixed-η infeasible while the guard stays feasible; GS5 safety-critical: fixed-η raises `F_wrong` 0.153→0.157). Deploy Guarded-CDQ2 instead. |

**Rule of thumb:** *Deploy ESP by default; deploy Guarded-CDQ2 when the application is deadline-leaning
and carries reliability slack; never deploy fixed-η CDQ2 unprotected.*

---

## 3. Evidence summary (GS1–GS6)

| Gate | Result | Honest scope |
|---|---|---|
| **GS1 G-METRIC-NAMESPACE** | canonical `macrostate_v2` schema + namespaces; ambiguous bare names banned from headline JSON; legacy gated behind `allow_legacy`; figure-guard. Adversarial audit found **2 critical holes** (top-level legacy smuggling; NaN sum-to-1 bypass) — fixed. | 27 tests; S15 migrated + archived. |
| **GS2 G-RESULT-MANIFEST** | every headline result carries physics/profile/evidence/scene/protocol/policy/checkpoint hashes + provenance + tracked seeds; train/eval mismatch fails fast (unless registered OOD); blocks single-seed headlines. | 17 tests; reuses `experiment_spec`. |
| **GS3 G-ESP-PERFORMANCE-SCALE** | trained 5-seed ESP/ESD-GNN **transfers** across scale (`macro_P_correct` ≈ 0.95 at N=120/336/660, reliability-safe); scale-regret −0.007 vs expert; **fixed-protocol deadline degradation recovered** by the fixed-service-profile `R_d∝√N` rule (Pc→1.0 at N=660/1248/3036). | real MC to N=3036; N=9840 documented approx; lightly-trained GNN ≈ distance heuristic (reported). |
| **GS4 G-ETA-RISK-LIVENESS** | the trade-off is **governed by the deadline regime**: η moves mass **deadline→correct** in the feasible-deadline window (R_d=14 mm_high: Fd 0.100→0.065@η=8) but **deadline-up** when too tight; iid flat. Mechanism **measured** (distant-peer selection). | conditional benefit; F_wrong flat ⇒ a *deadline* gain, not reliability. |
| **GS5 G-GUARDED-CDQ2** | hard+soft guard; **enables η where safe** (deadline +0.020 @ε=0.10, stays feasible) and **disables→ESP where risky** (fixed-η infeasible / raises F_wrong); **never violates the budget**. | small gain, narrow regime; **primary value = safety**. |
| **GS6 G-HAZARD-PROFILES** | **all 5 profiles select rationally**: safety/fail-safe→ESP, balanced/deadline-critical→Guarded-CDQ2, energy→ESP, safety-critical scene→all-ESP (feasibility gate). | deadline gain small; routing driven by the gate + costs + real energy cost. |

---

## 4. The honest negative / boundary results (reported, not hidden — constraint #12)

- **CDQ2 does not reduce the wrong/split basins.** Across the round, diversity's effect on `F_wrong` is
  flat-to-slightly-up. A deadline benefit is **never** counted as a reliability improvement
  (forbidden-shortcut #1).
- **The liveness benefit is small and conditional** — it requires covariance *and* a feasible-but-stressed
  deadline *and* reliability slack. Outside that window it vanishes or reverses (too-tight deadline → η
  *raises* `F_deadline`; GS4).
- **Diversity costs energy** (~1.2 %, distant peers) — a real deployment cost, not free liveness.
- **The lightly-trained ESP/ESD-GNN does not beat the distance heuristic** in the iid regime (GS3); the
  scale gate validates *transfer + the calibration rule*, not GNN superiority.
- **Under the strict default `ε_w=10⁻³`, no policy is feasible in the correlated-error regimes**, so the
  guard correctly defaults to ESP everywhere; the enable behaviour is demonstrated by sweeping ε as a
  per-application **service target** (never lowered to force a pass; forbidden-shortcut #13).

These boundaries are *why* the contribution is framed as trade-off **control** (the guard) rather than a
reliability win.

---

## 5. Acceptance (plan §8) + reproducibility

1. **ESP/ESD-GNN performance-scale validation complete** — GS3 (real macrostate-basin MC, N up to 3036 +
   documented approx; scale-regret + feasibility-retention; fixed-protocol vs fixed-service-profile). ✓
2. **CDQ2 role explicitly liveness/deadline scoped** — GS4 + GS5 (deadline-regime-governed; F_wrong flat). ✓
3. **Guarded-CDQ2 respects reliability constraints** — GS5 (feasible wherever ESP is; never violates the
   wrong/split UCB) + GS6 (feasibility gate). ✓
4. **No ambiguous metric names remain** — GS1 (every headline result is `macrostate_v2`; the ban-list +
   figure-guard are enforced in code, verified across all GS*-results JSONs: `forbidden=[] legacy=[]`). ✓
5. **All results reproducible through manifest hashes** — GS2 (every result record is hash-bound;
   `manifest.json` GS1–GS6 records modules/tests/evidence per slice). ✓

**Reproducible runs** (all under `docs/gate_evidence/guarded_cdq2/`, all `PYTHONPATH=. python …`):
`migrate_s15_to_v2.py`, `run_esp_performance_scale.py` (+ `run_esp_scale_largeN.py`),
`run_eta_risk_liveness_curve.py` (+ `run_eta_deadline_sensitivity.py`), `run_guard_calibration.py`,
`run_guarded_cdq2.py`, `run_hazard_profiles.py` → the corresponding `*_results.json`. Figures are produced
by `make_figures.py`, which **reads those JSONs only** (constraint #13).

**Code modules:** `src/metrics/{namespaces,schema,manifest}.py`; `src/evaluation/{esp_scale,eta_curve,
hazard_utility}.py`; `src/policies/guarded_cdq2.py`; `src/config/hazard_profile.py`. **Tests:**
`tests/metrics/{test_namespace_schema,test_result_manifest}.py`, `tests/evaluation/{test_esp_scale,
test_eta_curve,test_hazard_profiles}.py`, `tests/policies/test_guarded_cdq2.py`.

---

## 6. Paper role (what changes, what does not)

- **ESP/ESD-GNN remains the mainline reliability-first policy** and the headline.
- **CDQ2 enters as the unified query family** (`η=0 ≡ ESP`) and a **characterized, controlled** liveness
  extension — the paper's novelty is the **validity–liveness trade-off + its guard**, with the basin
  payoff honestly scoped.
- **Guarded-CDQ2 is the deployable artifact**: a one-dial, reliability-gated policy that captures the
  liveness gain where safe and is provably never worse than ESP on the hard constraints.
- The legacy node-union/global-product reliability headline is **not** revived (constraint #2); the
  macrostate basin first-hitting is the only headline metric.
