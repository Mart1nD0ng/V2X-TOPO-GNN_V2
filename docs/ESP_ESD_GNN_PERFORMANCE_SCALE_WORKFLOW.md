# ESP/ESD-GNN Publication-Grade Performance-Scale Validation Workflow

## 0. Purpose

This workflow designs a publication-grade validation round for the **ESP/ESD-GNN reliability-first mainline**.

The previous Guarded-CDQ2 round established:

- CDQ2 is the general query family.
- ESP is the \(\eta=0\) reliability-first specialization.
- \(\eta>0\) CDQ2 is a liveness/deadline extension, not a reliability mechanism.
- Guarded-CDQ2 is optional and may be evaluated as an extension.
- The next unresolved core question is whether **trained ESP/ESD-GNN** has true performance-scale generalization, not merely runtime scalability.

This round must answer:

\[
\boxed{
\text{Does a trained ESP/ESD-GNN checkpoint preserve macrostate-basin performance across }N=100\ldots10000?
}
\]

and:

\[
\boxed{
\text{Does it outperform strong, capability-matched topology baselines under full physics and dynamic-MC judgment?}
}
\]

This workflow explicitly separates:

- computational scalability;
- performance scalability;
- OOD generalization;
- service-profile calibration;
- model superiority.

A run may pass computation scaling but fail performance scaling. That must be reported honestly.

---

## 1. Non-Negotiable Definitions

### 1.1 Headline metrics

Only `macrostate_v2` macrostate basin outcomes are headline:

```text
macro_P_correct
macro_F_wrong
macro_F_split
macro_F_deadline
macro_T_confirm
macro_D95
macro_D99
macro_CVaR99
macro_energy
```

### 1.2 Hard constraints

Reliability constraints:

\[
\macro_F_{\mathrm{wrong}}^{\mathrm{UCB}}\le\epsilon_w,
\]

\[
\macro_F_{\mathrm{split}}^{\mathrm{UCB}}\le\epsilon_s.
\]

Deadline may be a hard service constraint or an optimization target depending on the profile:

\[
\macro_F_{\mathrm{deadline}}^{\mathrm{UCB}}\le\epsilon_d.
\]

Within the feasible set, compare:

\[
\operatorname{CVaR}_{q}(T_{\mathrm{confirm}}),\quad
D_{99},\quad
E.
\]

### 1.3 Judge

The headline judge is:

\[
\boxed{
\text{independent dynamic-MC macrostate basin first-hitting under full physics}.
}
\]

Analytic surrogates may train and screen models, but they do not certify final results.

### 1.4 Query family

Mainline policy:

```text
ESP/ESD-GNN
```

CDQ2 and Guarded-CDQ2 are optional extension arms, not the mainline reliability claim.

---

## 2. Current State That Motivates This Round

The Guarded-CDQ2 round already completed metric namespace cleanup and result-hash enforcement. The prior scale evidence showed real macrostate outcomes for trained ESP/ESD-GNN across moderate scales, but the evidence is not yet publication-grade at the full target range.

Known limitations to fix:

1. Existing scale validation is partly computational and partly performance-level.
2. \(N=9840\) currently relies on documented approximation rather than full dynamic-MC certification.
3. The lightly trained ESP/ESD-GNN did not clearly beat the distance heuristic in the iid regime.
4. Fixed-protocol large-\(N\) performance can degrade through deadline miss and must be separated from fixed-service-profile scaling.
5. Training budgets were intentionally reduced in prior gates; they are not sufficient for final performance claims.

---

## 3. Experiment Overview

The validation round contains five experiment blocks.

```text
A. Training-budget calibration
B. Baseline and oracle calibration
C. Fixed-protocol scale validation
D. Fixed-service-profile scale validation
E. OOD generalization matrix
```

The final output is a synthesis report deciding:

- whether ESP/ESD-GNN shows true scale performance;
- where it beats heuristics;
- where it only matches heuristics;
- where protocol/service-profile calibration, not model generalization, explains success;
- whether Guarded-CDQ2 should remain an extension arm.

---

# 4. Block A — Training-Budget Calibration

## 4.1 Goal

Ensure the ESP/ESD-GNN checkpoint used in scale validation is not a lightly trained placeholder.

The minimum question is:

\[
\boxed{
\text{Does longer full-physics training materially improve macrostate performance?}
}
\]

## 4.2 Training arms

Train ESP/ESD-GNN under full physics with:

```text
steps ∈ {pilot, medium, full}
model_seeds ≥ 5
scene_seeds ≥ 30
```

Recommended concrete levels:

```text
pilot  : prior gate-level budget
medium : 5× pilot
full   : until validation macro objective plateaus or early-stop patience expires
```

The code may choose exact step counts based on runtime, but must log them.

## 4.3 Training scales

Use mixed-scale training:

\[
N_{\mathrm{train}}\in\{100,300,1000\}.
\]

Optional:

\[
N_{\mathrm{train}}\in\{100,300,1000,3000\}
\]

if compute allows.

## 4.4 Validation during training

Track:

```text
macro_P_correct
macro_F_wrong_UCB
macro_F_split_UCB
macro_F_deadline
macro_D99
macro_CVaR99
macro_energy
effective_sampling_progress
effective_sampling_drift
receiver_load
```

## 4.5 Acceptance

A trained checkpoint is publication-eligible only if:

1. medium/full training improves or confirms pilot performance;
2. training curves plateau or early-stop criteria are documented;
3. no model seed silently diverges;
4. training uses the same full-physics path as evaluation;
5. checkpoint manifest binds physics/profile/evidence/scene/query/model hashes.

If longer training does not improve over pilot, report that as a negative result and proceed with the best validated checkpoint.

---

# 5. Block B — Baseline and Oracle Calibration

## 5.1 Goal

A scale claim is meaningful only against strong baselines.

## 5.2 Required baselines

Evaluate at least:

1. `uniform_esp`
2. `distance_heuristic`
3. `link_quality_heuristic`
4. `load_balanced_heuristic`
5. `region_bridge_heuristic`
6. `direct_edge_logit_optimizer`
7. `scale_specific_esp_esd_gnn_expert`
8. `shared_esp_esd_gnn_checkpoint`

Optional extension:

9. `guarded_cdq2`

## 5.3 Direct topology oracle

The direct optimizer is not a deployable model. It estimates problem headroom:

\[
\text{Headroom}(N)=J_{\mathrm{heuristic}}(N)-J_{\mathrm{oracle}}(N).
\]

If oracle does not beat heuristics, then the scene/profile has little topology-learning room; do not claim GNN failure.

## 5.4 Capability matching

All policies must use:

- same candidate graph;
- same full physics;
- same service profile;
- same macrostate judge;
- same participation weights;
- same dynamic-MC seeds where possible.

No policy may access ground truth, peer votes, or simulator-only hidden state.

## 5.5 Acceptance

Baseline calibration is complete when:

1. each baseline has macrostate outcomes and UCBs;
2. direct optimizer headroom is measured;
3. scale-specific expert is available at each main scale;
4. heuristic strengths and weaknesses are documented;
5. shared checkpoint is compared to both heuristics and experts.

---

# 6. Block C — Fixed-Protocol Scale Validation

## 6.1 Goal

Evaluate what happens when the same protocol is used at all scales.

This answers:

\[
\boxed{
\text{Does the trained topology policy itself degrade as }N\text{ grows under a fixed protocol?}
}
\]

## 6.2 Node scales

Required:

\[
N\in\{100,300,1000,3000\}.
\]

Stretch:

\[
N=10000.
\]

If \(N=10000\) full dynamic-MC is infeasible, report a documented rare-event or statistical approximation, not a green full-performance claim.

## 6.3 Metrics

For each \(N\):

```text
macro_P_correct
macro_F_wrong_UCB
macro_F_split_UCB
macro_F_deadline
macro_D95
macro_D99
macro_CVaR99
macro_energy
strict_audit
effective_sampling_diagnostics
runtime
memory
```

## 6.4 Required plots

1. \(N\) vs macro outcomes.
2. \(N\) vs deadline miss.
3. \(N\) vs D99/CVaR.
4. \(N\) vs scale regret.
5. \(N\) vs feasibility rate.
6. Runtime/memory vs \(N\).

## 6.5 Acceptance

Fixed-protocol validation is successful if:

1. degradation, if present, is exposed rather than hidden;
2. ESP/ESD-GNN remains stable and does not collapse;
3. any loss of feasibility is attributed to protocol/service mismatch, not mislabeled as model failure;
4. shared checkpoint is compared against scale-specific experts and heuristics.

---

# 7. Block D — Fixed-Service-Profile Scale Validation

## 7.1 Goal

Evaluate deployable scale behavior when protocol parameters are calibrated by a pre-registered rule.

This answers:

\[
\boxed{
\text{Can the system maintain the same service target as }N\text{ changes?}
}
\]

## 7.2 Pre-registered calibration rule

Define a single rule before experiments, for example:

\[
R_d(N)=
\left\lceil
R_{d,0}\sqrt{\frac{N}{N_0}}
\right\rceil,
\]

or a more protocol-grounded function of density, candidate degree, and service target.

Do not tune \(R_d\) per result after seeing outcomes.

## 7.3 Node scales

Required:

\[
N\in\{100,300,1000,3000\}.
\]

Stretch:

\[
N=10000.
\]

## 7.4 Evaluation

For each \(N\), compare:

- shared checkpoint;
- scale-specific expert;
- heuristics;
- direct optimizer if feasible.

## 7.5 Scale regret

Define:

\[
J(\pi)
=
\operatorname{CVaR}_{q}(T_{\mathrm{confirm}})
+\lambda_E E
\]

among feasible policies only.

Scale regret:

\[
\operatorname{Regret}_{\mathrm{scale}}(N)
=
J_{\mathrm{shared}}(N)-J_{\mathrm{expert}}(N).
\]

Normalized scale regret:

\[
\operatorname{NRegret}(N)
=
\frac{
J_{\mathrm{shared}}(N)-J_{\mathrm{expert}}(N)
}{
J_{\mathrm{heuristic}}(N)-J_{\mathrm{expert}}(N)+\varepsilon
}.
\]

Feasibility retention:

\[
R_{\mathrm{feas}}(N)
=
\frac{
\Pr(\text{shared feasible at }N)
}{
\Pr(\text{expert feasible at }N)+\varepsilon
}.
\]

## 7.6 Acceptance

Fixed-service validation is successful if:

1. shared checkpoint remains feasible across required scales;
2. scale regret is bounded and reported;
3. expert comparison exists;
4. protocol calibration rule is pre-registered and not result-tuned;
5. \(N=10000\) is clearly labeled as full-MC, rare-event, or approximation.

---

# 8. Block E — OOD Generalization Matrix

## 8.1 Goal

Determine what the trained ESP/ESD-GNN generalizes over.

## 8.2 One-axis-at-a-time tests

Vary one axis at a time:

1. node count;
2. density;
3. road geometry;
4. evidence covariance;
5. PHY load/interference;
6. sensor group structure;
7. service profile;
8. mobility/handoff.

## 8.3 Metrics

For each OOD axis:

```text
feasibility_rate
scale_or_domain_regret
macro outcome deltas
D99/CVaR deltas
energy deltas
effective sampling deltas
```

## 8.4 Acceptance

Generalization is credible only if:

1. in-distribution performance is established first;
2. each OOD axis is isolated;
3. failure modes are reported;
4. no OOD claim relies only on runtime.

---

# 9. Statistical Protocol

## 9.1 Minimum seeds

Publication-grade minimum:

```text
model_seeds >= 5
scene_seeds >= 30
```

For expensive large-N tiers, any reduction must be explicitly labeled as a compute-limited approximation.

## 9.2 Dynamic-MC trials

For failure probability \(p\), a rough zero-failure 95% upper bound is:

\[
p_{\mathrm{upper}}\approx \frac{3}{M}.
\]

To certify:

\[
p<10^{-3},
\]

with zero failures requires:

\[
M\gtrsim3000.
\]

If \(M\) is too expensive, use:

- rare-event importance sampling;
- splitting/subset simulation;
- or report the large-N result as approximate, not certified.

## 9.3 Confidence intervals

Use:

- Wilson intervals for binomial outcomes;
- nested bootstrap for model-seed and scene-seed variance;
- paired comparisons with common random variables where possible;
- explicit UCBs for hard constraints.

## 9.4 Common randomness

Reusing the same RNG seed is not enough if policies consume different numbers of random variates. Use shared latent replay where feasible:

```text
scene latent variables
evidence variables
resource variables
fading variables
query random keys
```

---

# 10. Result Schema

Every result record must use `macrostate_v2` and include:

```json
{
  "metric_namespace_version": "macrostate_v2",
  "experiment_family": "esp_performance_scale_v2",
  "policy": "...",
  "query_family": "ESP",
  "scale_protocol": "fixed_protocol|fixed_service_profile",
  "node_count": 1000,
  "training_node_counts": [100, 300, 1000],
  "model_seed": 0,
  "scene_seed": 0,
  "dynamic_mc_trials": 3000,
  "macro": {},
  "strict_audit": {},
  "diagnostic": {},
  "sampling": {},
  "runtime": {},
  "manifest": {}
}
```

No bare legacy keys are allowed.

---

# 11. New Gates

```text
G-ESP-TRAINING-BUDGET
G-ESP-BASELINE-ORACLE
G-ESP-FIXED-PROTOCOL-SCALE
G-ESP-FIXED-SERVICE-SCALE
G-ESP-OOD-GENERALIZATION
G-ESP-RARE-EVENT-CERTIFICATION
G-ESP-SCALE-SYNTHESIS
```

---

# 12. Acceptance for the Full Round

The ESP/ESD-GNN performance-scale validation round is complete only if:

1. trained checkpoints are stronger than pilot placeholders or the failure to improve is documented;
2. full-physics train/eval compatibility is enforced;
3. performance-scale results use macrostate dynamic-MC, not only runtime;
4. required scales \(N=100,300,1000,3000\) have real performance results;
5. \(N=10000\) is clearly labeled as full-MC, rare-event, or approximation;
6. shared checkpoint is compared against scale-specific experts and strong heuristics;
7. fixed-protocol and fixed-service-profile results are both reported;
8. OOD generalization is tested one axis at a time;
9. all hard reliability constraints are evaluated with UCBs;
10. all results are hash-bound and metric-clean.

---

# 13. Interpretation Rules

## 13.1 If ESP/ESD-GNN beats heuristics and has low regret

Then claim:

> ESP/ESD-GNN provides real performance-scale topology learning under full physics and macrostate dynamic-MC evaluation.

## 13.2 If ESP/ESD-GNN matches heuristics but transfers well

Then claim:

> ESP/ESD-GNN is a stable learned topology constructor with scale transfer, but not a performance-superior model in this regime.

## 13.3 If ESP/ESD-GNN loses to heuristics

Then claim:

> The current model architecture is not yet competitive; the contribution should shift toward evaluator/workflow/mechanism characterization unless architecture is improved.

## 13.4 If fixed protocol fails but fixed service profile succeeds

Then claim:

> Large-scale performance requires service-profile protocol calibration; the GNN generalizes conditional on a feasible protocol.

## 13.5 If both fixed protocol and fixed service profile fail

Then stop model-claim escalation and revisit protocol, service profile, or event participation measure.

---

# 14. Forbidden Shortcuts

1. Do not use computational scaling as performance scaling.
2. Do not use old product/node-union metrics.
3. Do not use ambiguous metric names.
4. Do not train on ideal links and evaluate on full physics.
5. Do not omit strong heuristics.
6. Do not omit scale-specific experts.
7. Do not use only one model seed for headline.
8. Do not call \(N=10000\) validated if it is approximation-only.
9. Do not hide fixed-protocol deadline degradation.
10. Do not change service-profile calibration after seeing results.
11. Do not report feasibility without UCB.
12. Do not use small complete graphs as scale evidence.
13. Do not allow figure scripts to recompute metrics.
14. Do not let Guarded-CDQ2 replace ESP in this mainline round.

---

# 15. Stop Conditions

Stop and report if:

1. trained ESP/ESD-GNN cannot beat or match strong heuristics at small/mid scale;
2. scale-specific experts are not better than heuristics, implying little topology-learning headroom;
3. dynamic-MC certification is computationally infeasible without rare-event methods;
4. service-profile calibration rule cannot restore feasibility;
5. OOD axes produce systematic failures that invalidate the generalization claim;
6. metric namespace or manifest checks fail;
7. full physics train/eval compatibility fails.
