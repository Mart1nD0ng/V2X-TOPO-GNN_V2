# Engineering Plan: Guarded-CDQ2 Experiments, ESP Performance-Scale Validation, and Metric Cleanup

## 0. Execution Summary

This plan schedules three experimental tasks and one code-quality task:

1. \(\eta\)-risk-liveness curve.
2. Guarded-CDQ2.
3. Hazard-weighted service profiles.
4. ESP/ESD-GNN performance-scale validation.
5. Metric naming cleanup and code entropy reduction.

The correct implementation order is not the same as the conceptual order. The first engineering task must be metric cleanup, because the new experiments must not serialize ambiguous legacy metrics.

Recommended order:

\[
\boxed{
\text{Metric cleanup}
\rightarrow
\text{ESP performance-scale baseline}
\rightarrow
\eta\text{-curve}
\rightarrow
\text{Guarded-CDQ2}
\rightarrow
\text{Hazard profiles}
\rightarrow
\text{final synthesis}
}
\]

---

# 1. Phase 0: Freeze and Tag Current State

## Tasks

1. Tag the current branch state:
   - `macrostate-cdq2-v2-before-guarded`.
2. Archive current CDQ2 factorial results.
3. Mark current CDQ2 conclusion:
   - mechanism-identifiable;
   - deadline/liveness benefit;
   - no wrong-risk reduction.
4. Do not overwrite S15 / Phase-10 evidence.

## Exit

A reproducible baseline exists before new experiments begin.

---

# 2. Phase 1: Metric Naming Cleanup

## Gate: G-METRIC-NAMESPACE

### Tasks

1. Create canonical metric schema:
   - `src/metrics/schema.py`
   - `src/metrics/namespaces.py`
2. Migrate result serialization to namespace groups:
   - `macro`
   - `strict_audit`
   - `diagnostic`
   - `sampling`
   - `cdq`
   - `runtime`
3. Rename ambiguous fields:
   - `F_wrong` → `macro_F_wrong` or `surrogate_nodeunion_F_wrong`;
   - `S_allcorrect` → `surrogate_product_S_allcorrect` or `macro_P_correct`;
   - `F_disagree` → `macro_F_split` or `strict_any_disagreement`.
4. Enforce `metric_namespace_version="macrostate_v2"`.
5. Add compatibility shim only in legacy readers.

### Tests

- No headline JSON contains ambiguous keys.
- Legacy node-union/product fields cannot be emitted without `legacy=True`.
- Figure scripts fail if given legacy metrics.
- Macrostate metrics sum:
  \[
  P_C+F_w+F_s+F_d=1.
  \]

### Exit

All new experiments emit only namespace-clean metrics.

---

# 3. Phase 2: Result Manifest and Hash Enforcement

## Gate: G-RESULT-MANIFEST

### Tasks

1. Every experiment output must include:
   - git commit or manifest id;
   - service profile hash;
   - physics hash;
   - evidence hash;
   - scene distribution hash;
   - policy hash;
   - query family;
   - model checkpoint hash.
2. Add fail-fast check:
   - train/eval physics mismatch fails unless declared OOD.
3. Add result validation:
   - same metric namespace;
   - no missing macro outcomes;
   - no untracked model seed.

### Exit

All subsequent experiments are reproducible and hash-bound.

---

# 4. Phase 3: ESP/ESD-GNN Performance-Scale Validation

## Gate: G-ESP-PERFORMANCE-SCALE

This phase comes before new CDQ conclusions. ESP is the reliability-first default and must be validated as the mainline.

### 4.1 Policies

Evaluate:

1. ESP/ESD-GNN shared checkpoint.
2. ESP/ESD-GNN scale-specific expert.
3. Uniform ESP.
4. Distance/link-quality heuristic.
5. Direct topology oracle if feasible.
6. Guarded-CDQ2 placeholder disabled at this phase.

### 4.2 Training scales

Train shared models on:

\[
N\in\{100,300,1000\}.
\]

Train scale-specific experts on:

\[
N\in\{100,300,1000,3000\}.
\]

Optional stretch:

\[
N=10000
\]

expert only if compute permits.

### 4.3 Evaluation scales

\[
N\in\{100,300,1000,3000,10000\}.
\]

### 4.4 Two protocols

Run both:

1. Fixed-protocol scaling.
2. Fixed-service-profile scaling with protocol parameters calibrated by a pre-registered rule.

### 4.5 Metrics

- `macro_P_correct`
- `macro_F_wrong`
- `macro_F_split`
- `macro_F_deadline`
- `macro_D95`
- `macro_D99`
- `macro_CVaR99`
- `energy`
- `scale_regret`
- `normalized_scale_regret`
- `feasibility_retention`
- runtime and memory

### 4.6 Statistics

Minimum:

- 5 model seeds;
- 30 scene seeds;
- sufficient dynamic MC trials;
- nested bootstrap;
- UCB reporting for rare failure.

### 4.7 Acceptance

The gate is green only if:

1. performance is evaluated, not only runtime;
2. at least \(N=100,300,1000,3000\) have dynamic-MC performance;
3. \(N=10000\) has either dynamic MC or documented rare-event/statistical approximation;
4. scale regret is bounded and reported;
5. failure under fixed-protocol scaling is not hidden.

---

# 5. Phase 4: \(\eta\)-Risk-Liveness Curve

## Gate: G-ETA-RISK-LIVENESS

### 5.1 Script

Create:

```text
scripts/experiments/run_eta_risk_liveness_curve.py
```

Input:

- trained ESP/ESD-GNN checkpoint or fixed quality;
- diversity embeddings;
- service profile;
- environment family;
- physics mode;
- eta grid.

Output:

```text
docs/gate_evidence/guarded_cdq2/eta_curve_results.json
```

### 5.2 Eta grid

\[
\eta\in\{0,0.25,0.5,1,2,4,8,16\}.
\]

### 5.3 Environments

- iid;
- matched-marginal low covariance;
- matched-marginal high covariance;
- overlapping common-cause;
- balanced split-risk.

Each with:

- fixed-link ablation;
- full physics headline.

### 5.4 Metrics

- macro outcomes;
- D95/D99/CVaR;
- energy;
- selected correlation;
- progress/drift;
- minority-wrong exposure;
- receiver load.

### 5.5 Acceptance

The gate is green if it produces statistically stable curves and explicitly identifies whether CDQ2 shifts probability mass:

- from deadline to correct;
- from deadline to wrong;
- from split to correct;
- or no effect.

The gate is not judged by whether CDQ2 always improves reliability.

---

# 6. Phase 5: Guarded-CDQ2

## Gate: G-GUARDED-CDQ2

### 6.1 Implement policies

Add:

```text
src/policies/guarded_cdq2.py
```

with two modes:

1. Hard guard.
2. Soft differentiable guard.

### 6.2 Guard inputs

- `macro_F_wrong_UCB`
- `macro_F_split_UCB`
- `macro_F_deadline`
- risk slack
- deadline pressure
- optional load pressure

### 6.3 Policy

Hard guard:

\[
\eta=0
\quad
\text{if}
\quad
F_w^{UCB}>\epsilon_w-\delta_w
\quad\text{or}\quad
F_s^{UCB}>\epsilon_s-\delta_s.
\]

Soft guard:

\[
\eta = \eta_{\mathrm{raw}}
\sigma((m_w-\delta_w)/T_w)
\sigma((m_s-\delta_s)/T_s)
\sigma((p_d-\delta_d)/T_d).
\]

### 6.4 Evaluation arms

- ESP;
- fixed-CDQ2;
- learned-CDQ2;
- hard Guarded-CDQ2;
- soft Guarded-CDQ2;
- oracle-guarded upper bound.

### 6.5 Acceptance

Guarded-CDQ2 must:

1. satisfy wrong/split UCB constraints;
2. improve deadline or tail latency in high-covariance deadline-stressed scenes;
3. automatically fall back to ESP in safety-critical scenes;
4. expose guard activation statistics.

---

# 7. Phase 6: Hazard-Weighted Service Profiles

## Gate: G-HAZARD-PROFILES

### 7.1 Implement

Create:

```text
src/config/hazard_profile.py
src/evaluation/hazard_utility.py
scripts/experiments/run_hazard_profiles.py
```

### 7.2 Profiles

At least:

1. Safety-first.
2. Balanced.
3. Deadline-critical.
4. Fail-safe available.
5. Energy-constrained.

### 7.3 Utility

\[
B =
c_d(F_d^{ESP}-F_d^{policy})
+
c_T(D_q^{ESP}-D_q^{policy})
+
c_E(E^{ESP}-E^{policy})
-
c_w[\Delta F_w]_+
-
c_s[\Delta F_s]_+.
\]

A policy can be selected only if:

\[
F_w^{UCB}\le\epsilon_w,
\qquad
F_s^{UCB}\le\epsilon_s.
\]

### 7.4 Required output

For each profile:

- selected policy frequency;
- feasibility rate;
- hazard utility;
- macro outcomes;
- deadline and energy;
- explanation of selection.

### 7.5 Acceptance

The gate is green if policy selection changes rationally:

- safety-first → ESP / eta near 0;
- deadline-critical → CDQ2 or Guarded-CDQ2 when feasible;
- balanced → Guarded-CDQ2;
- fail-safe available → conservative.

---

# 8. Phase 7: Joint Synthesis

## Gate: G-FINAL-SYNTHESIS

### Tasks

1. Produce a unified report:
   - ESP performance-scale validation;
   - eta curve;
   - guarded policy;
   - hazard profiles.
2. Decide paper role:
   - ESP mainline;
   - CDQ2 as unified family and liveness extension;
   - Guarded-CDQ2 as deployable policy.
3. Generate figures:
   - eta risk-liveness curves;
   - guarded-CDQ2 frontier;
   - hazard profile policy map;
   - ESP scale-regret curve.

### Acceptance

The project is ready for paper rewriting only if:

1. ESP/ESD-GNN performance-scale validation is complete.
2. CDQ2 role is explicitly liveness/deadline scoped.
3. Guarded-CDQ2 respects reliability constraints.
4. No ambiguous metric names remain.
5. All results are reproducible through manifest hashes.

---

# 9. Required New Gates

```text
G-METRIC-NAMESPACE
G-RESULT-MANIFEST
G-ESP-PERFORMANCE-SCALE
G-ETA-RISK-LIVENESS
G-GUARDED-CDQ2
G-HAZARD-PROFILES
G-FINAL-SYNTHESIS
```

---

# 10. Loop Workflow

For each gate:

1. Read this plan and the technical document.
2. Choose one smallest gate slice.
3. Write API/mathematical contract.
4. Write failing tests.
5. Implement minimal canonical-path functionality.
6. Run unit, integration, dynamic-MC, and result-schema tests.
7. Run Mechanism Identifiability Contract audit where applicable.
8. Write evidence JSON and decision log.
9. Commit / manifest.
10. Continue.

---

# 11. Forbidden Shortcuts

1. Treat computational scaling as performance scaling.
2. Report CDQ2 as reliability-improving when it only reduces deadline.
3. Enable CDQ2 when wrong/split constraints are violated.
4. Use ambiguous metric names.
5. Use legacy product/node-union metrics in headline.
6. Train on ideal links and evaluate on full physics.
7. Use the same RNG seed and call it common random numbers without shared latent replay.
8. Use single model seed for headline.
9. Use small complete graphs as performance-scale evidence.
10. Let figure scripts recompute metrics with separate code.
11. Bury negative \(F_{\mathrm{wrong}}\) results.
12. Skip UCB reporting for rare failures.
13. Lower reliability thresholds to make Guarded-CDQ2 pass.

---

# 12. Stop Conditions

Stop and report if:

1. ESP performance does not scale beyond small \(N\).
2. Guarded-CDQ2 cannot satisfy wrong/split constraints in any service profile.
3. Hazard utilities depend on arbitrary cost ratios with no stable policy selection.
4. Metric namespace cleanup requires breaking too many existing APIs.
5. Dynamic MC contradicts analytic trend systematically.
6. Rare-event MC becomes computationally infeasible without importance sampling.
