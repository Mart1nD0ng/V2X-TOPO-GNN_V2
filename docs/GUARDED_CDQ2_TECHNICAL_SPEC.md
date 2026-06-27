# CDQ2 as a Guarded Liveness Extension and ESP Performance-Scale Validation

## 0. Purpose

This document defines the next technical round for the `macrostate-cdq2-redesign` branch.

The previous Phase-10 result established a nuanced outcome:

- CDQ2 is mechanism-identifiable: it reacts to matched-marginal covariance structure.
- CDQ2 improves correct-basin probability in high-covariance settings mainly by reducing deadline miss.
- CDQ2 does **not** reduce wrong-basin risk in the tested majority-correct regime; it can slightly raise \(F_{\mathrm{wrong}}\).
- ESP remains the reliability-first default.
- CDQ2 should be treated as a unified query family whose \(\eta=0\) specialization is ESP, while \(\eta>0\) is a liveness/deadline-enhancing mode that must be guarded by reliability constraints.

This round therefore has three technical objectives:

1. Characterize the **\(\eta\)-risk-liveness curve**.
2. Implement and evaluate **Guarded-CDQ2**.
3. Evaluate **hazard-weighted service profiles**.

In parallel, this round must strengthen:

4. **ESP/ESD-GNN performance-scale validation** using trained checkpoints at multiple node scales.
5. **metric naming cleanup and code entropy reduction** to prevent legacy product/node-union metrics from re-entering the headline.

---

## 1. CDQ2 Query Family and ESP Specialization

CDQ2 defines a fixed-size \(k\)-DPP query distribution. For each source node \(i\), let candidate quality weights be

\[
a_{ij}=\exp(s_{ij})>0,
\qquad
D_i=\operatorname{diag}(a_{ij}),
\]

and let unit-normalized diversity embeddings be

\[
\bar z_{ij}=\frac{z_{ij}}{\|z_{ij}\|+\varepsilon}.
\]

Let \(Z_i\) stack the \(\bar z_{ij}\) rows. CDQ2 uses

\[
\boxed{
L_i
=
D_i^{1/2}
\left(I+\eta_i Z_iZ_i^\top\right)
D_i^{1/2},
\qquad
\eta_i\ge0.
}
\]

The fixed-size subset law is

\[
\boxed{
P_{\mathrm{CDQ2},i}(S)
=
\frac{\det\left((L_i)_S\right)}
{e_k(\lambda(L_i))},
\qquad |S|=k.
}
\]

When

\[
\eta_i=0,
\]

we have

\[
L_i=D_i,
\]

therefore

\[
\det((L_i)_S)=\prod_{j\in S}a_{ij},
\]

and

\[
\boxed{
P_{\mathrm{CDQ2},i}(S;\eta_i=0)
=
P_{\mathrm{ESP},i}(S).
}
\]

Thus ESP is exactly the reliability-first specialization of the CDQ2 family.

The correct interpretation is:

\[
\boxed{
\text{CDQ2 is the general query family; ESP is the safety/default mode.}
}
\]

A particular \(\eta>0\) CDQ2 operating point may trade validity for liveness, so it must not be deployed unless the reliability guard permits it.

---

## 2. Macrostate Outcomes and Constraints

The project no longer uses node-union or global-product failure as the headline.

For each trial, define the participation-weighted macrostate masses at polling epoch \(r\):

\[
C_r=\sum_i \omega_i \mathbf 1\{i\text{ correctly finalized}\},
\]

\[
W_r=\sum_i \omega_i \mathbf 1\{i\text{ wrongly finalized}\},
\]

\[
U_r=1-C_r-W_r.
\]

Basins are:

\[
\mathcal B_C=\{C_r\ge\rho_f\},
\]

\[
\mathcal B_W=\{W_r\ge\rho_f\},
\]

\[
\mathcal B_S=\{C_r\ge\rho_s,\ W_r\ge\rho_s\}.
\]

First-hitting outcomes are:

\[
P_C,
\quad
F_{\mathrm{wrong}},
\quad
F_{\mathrm{split}},
\quad
F_{\mathrm{deadline}}.
\]

The hard reliability constraints are:

\[
F_{\mathrm{wrong}}\le\epsilon_w,
\]

\[
F_{\mathrm{split}}\le\epsilon_s,
\]

\[
F_{\mathrm{deadline}}\le\epsilon_d.
\]

For liveness-oriented comparisons, \(F_{\mathrm{deadline}}\) may be part of the objective, but \(F_{\mathrm{wrong}}\) and \(F_{\mathrm{split}}\) must remain hard constraints.

---

## 3. Experiment A: \(\eta\)-Risk-Liveness Curve

### 3.1 Objective

Quantify how the CDQ2 diversity strength \(\eta\) moves the system along a validity-liveness trade-off curve.

The goal is not to show that CDQ2 universally beats ESP. The goal is to characterize:

\[
\eta\uparrow
\quad\Rightarrow\quad
\text{diversity}\uparrow,\ 
F_{\mathrm{deadline}}\downarrow,\ 
F_{\mathrm{wrong}}\text{ possibly }\uparrow.
\]

### 3.2 Arms

Use the same trained quality scores \(s_{ij}\) and same diversity embeddings \(Z_i\), but vary \(\eta\).

Recommended sweep:

\[
\eta\in
\{0,\ 0.25,\ 0.5,\ 1,\ 2,\ 4,\ 8,\ 16\}.
\]

Where:

- \(\eta=0\): ESP.
- \(\eta>0\): CDQ2 diversity mode.

Also include learned-\(\eta\) if already trained.

### 3.3 Environments

Run in at least four environment families:

1. iid evidence, no covariance structure.
2. region-block covariance.
3. overlapping common-cause covariance.
4. balanced split-risk environment.

For each environment, include both:

- fixed-link / idealized physical ablation;
- full physics.

All headline conclusions must be based on full physics.

### 3.4 Metrics

For each \(\eta\):

\[
P_C,
\quad
F_{\mathrm{wrong}},
\quad
F_{\mathrm{split}},
\quad
F_{\mathrm{deadline}},
\quad
D_{50},D_{95},D_{99},
\quad
\operatorname{CVaR}_{0.99}(T_{\mathrm{confirm}}),
\quad
E.
\]

Effective-sampling diagnostics:

\[
g_i=h_i^++h_i^-,
\]

\[
\Delta_i=h_i^+-h_i^-,
\]

\[
k_{\mathrm{eff}},
\quad
\text{selected pairwise evidence correlation},
\quad
\text{minority-wrong exposure},
\quad
\text{receiver load},
\quad
\text{cross-region mass}.
\]

### 3.5 Expected outcomes

A valid result may be:

- \(F_{\mathrm{deadline}}\) decreases with \(\eta\);
- \(F_{\mathrm{wrong}}\) increases mildly with \(\eta\);
- \(P_C\) increases only when the deadline reduction dominates;
- iid/no-covariance settings remain neutral.

This is not a failure; it is the expected validity-liveness trade-off.

### 3.6 Required plots

1. \(\eta\) vs \(F_{\mathrm{wrong}}\), \(F_{\mathrm{split}}\), \(F_{\mathrm{deadline}}\).
2. \(\eta\) vs \(P_C\).
3. \(\eta\) vs \(D_{95}\), \(D_{99}\), CVaR.
4. \(\eta\) vs selected evidence correlation.
5. \(\eta\) vs minority-wrong exposure.
6. \(\eta\) vs receiver load.
7. Trade-off front: \((F_{\mathrm{wrong}},F_{\mathrm{deadline}})\).

### 3.7 Acceptance

This experiment is complete if it identifies the shape of the trade-off with confidence intervals. It does not need to show CDQ2 improves every metric.

---

## 4. Experiment B: Guarded-CDQ2

### 4.1 Objective

Implement CDQ2 as a deployable policy that activates diversity only when reliability slack is sufficient.

The default reliability-first mode is ESP:

\[
\eta=0.
\]

CDQ2 mode:

\[
\eta>0
\]

is allowed only under a reliability guard.

### 4.2 Reliability slack

Let dynamic-MC or calibrated analytic UCB estimates be:

\[
\widehat F_w^{\mathrm{UCB}},
\qquad
\widehat F_s^{\mathrm{UCB}}.
\]

Define slack:

\[
m_w=\epsilon_w-\widehat F_w^{\mathrm{UCB}},
\]

\[
m_s=\epsilon_s-\widehat F_s^{\mathrm{UCB}}.
\]

### 4.3 Hard guard

The simplest deployable rule is:

\[
\boxed{
\eta_i^{\mathrm{guarded}}
=
\begin{cases}
0, & m_w<\delta_w \text{ or } m_s<\delta_s,\\
\eta_i^{\mathrm{raw}}, & \text{otherwise.}
\end{cases}
}
\]

where \(\delta_w,\delta_s>0\) are reliability margins.

### 4.4 Differentiable soft guard

For training, use:

\[
G_i
=
\sigma\left(\frac{m_w-\delta_w}{T_w}\right)
\sigma\left(\frac{m_s-\delta_s}{T_s}\right)
\sigma\left(\frac{p_d-\delta_d}{T_d}\right),
\]

where \(p_d\) is deadline pressure, e.g.

\[
p_d
=
\widehat F_{\mathrm{deadline}}
\quad\text{or}\quad
D_{99}/T_d.
\]

Then

\[
\boxed{
\eta_i^{\mathrm{guarded}}
=
G_i\eta_i^{\mathrm{raw}}.
}
\]

At deployment, use the hard guard unless differentiable control is required online.

### 4.5 Arms

Compare:

1. ESP.
2. Fixed-\(\eta\) CDQ2.
3. Learned-\(\eta\) CDQ2.
4. Guarded-CDQ2 hard guard.
5. Guarded-CDQ2 soft guard.
6. Oracle guarded CDQ2 using MC-estimated slack, for upper-bound analysis only.

### 4.6 Metrics

Primary:

\[
F_{\mathrm{wrong}}^{\mathrm{UCB}},
\quad
F_{\mathrm{split}}^{\mathrm{UCB}},
\quad
F_{\mathrm{deadline}},
\quad
P_C,
\quad
D_{99},
\quad
E.
\]

Secondary:

- percentage of nodes/episodes where guard disables CDQ2;
- \(\eta\) distribution;
- risk slack distribution;
- deadline pressure distribution;
- false activation and false suppression rates.

### 4.7 Acceptance

Guarded-CDQ2 is successful if it satisfies:

\[
F_{\mathrm{wrong}}^{\mathrm{UCB}}
\le
\epsilon_w,
\]

\[
F_{\mathrm{split}}^{\mathrm{UCB}}
\le
\epsilon_s,
\]

and improves deadline/liveness relative to ESP in high-covariance deadline-stressed settings:

\[
F_{\mathrm{deadline}}^{\mathrm{guarded}}
<
F_{\mathrm{deadline}}^{\mathrm{ESP}}
\]

or

\[
D_{99}^{\mathrm{guarded}}
<
D_{99}^{\mathrm{ESP}}.
\]

Guarded-CDQ2 must not be counted as successful if it buys deadline improvement by violating wrong/split constraints.

---

## 5. Experiment C: Hazard-Weighted Service Profiles

### 5.1 Objective

Determine when CDQ2 should be used based on application hazard costs, rather than treating every application as safety-first.

### 5.2 Hazard-weighted utility

Define the CDQ2 net benefit relative to ESP:

\[
B_{\mathrm{CDQ}}
=
c_d(F_d^{\mathrm{ESP}}-F_d^{\mathrm{CDQ2}})
+
c_T(D_{q}^{\mathrm{ESP}}-D_{q}^{\mathrm{CDQ2}})
+
c_E(E^{\mathrm{ESP}}-E^{\mathrm{CDQ2}})
-
c_w[F_w^{\mathrm{CDQ2}}-F_w^{\mathrm{ESP}}]_+
-
c_s[F_s^{\mathrm{CDQ2}}-F_s^{\mathrm{ESP}}]_+.
\]

CDQ2 is selected only if:

\[
B_{\mathrm{CDQ}}>0
\]

and

\[
F_w^{\mathrm{CDQ2,UCB}}\le\epsilon_w,
\]

\[
F_s^{\mathrm{CDQ2,UCB}}\le\epsilon_s.
\]

### 5.3 Suggested profiles

| Profile | \(c_w\) | \(c_s\) | \(c_d\) | Expected policy |
|---|---:|---:|---:|---|
| Safety-first | very high | very high | medium | ESP |
| Balanced | high | high | high | Guarded-CDQ2 |
| Deadline-critical | medium/high | high | very high | CDQ2 or Guarded-CDQ2 |
| Fail-safe available | very high | very high | low | ESP |
| Energy-constrained | high | high | medium | Guarded CDQ2 only if energy improves |

### 5.4 Required results

For each profile, report:

- selected policy frequency;
- feasibility rate;
- mean and UCB of \(F_{\mathrm{wrong}}\);
- mean and UCB of \(F_{\mathrm{split}}\);
- deadline miss;
- \(D_{99}\);
- energy;
- hazard-weighted objective.

### 5.5 Acceptance

A profile experiment is successful if it shows policy selection changes rationally with hazard weights:

- safety-first selects ESP or \(\eta\approx0\);
- deadline-critical selects CDQ2 or Guarded-CDQ2 when constraints allow;
- balanced selects Guarded-CDQ2 in high-covariance deadline-stressed scenes;
- fail-safe-available remains conservative.

---

# 6. ESP/ESD-GNN Performance-Scale Validation

## 6.1 Motivation

The project already has strong evidence for computational scaling. This is not enough.

Performance-scale validation must answer:

> Does a trained ESP/ESD-GNN checkpoint preserve macrostate-basin performance across node counts and scene distributions?

## 6.2 Policies

Evaluate:

1. ESP/ESD-GNN trained checkpoint.
2. Scale-specific ESP/ESD-GNN expert.
3. Uniform ESP.
4. Distance / link-quality heuristic.
5. Direct topology oracle if feasible.
6. Guarded-CDQ2 as extension, not headline.

## 6.3 Training scales

Train shared checkpoints on:

\[
N_{\mathrm{train}}\in\{100,300,1000\}.
\]

Optionally include a mixed-scale checkpoint.

Train scale-specific experts on:

\[
N\in\{100,300,1000,3000\}.
\]

## 6.4 Test scales

Evaluate on:

\[
N_{\mathrm{test}}\in\{100,300,1000,3000,10000\}.
\]

Use both:

### Fixed protocol

Same \(k,\alpha,\beta,R_d\) across all \(N\).

Purpose: expose degradation from scale.

### Fixed service profile

Protocol parameters are calibrated by a pre-registered rule to maintain the same service target.

Purpose: evaluate deployable scale behavior.

## 6.5 Metrics

Primary:

\[
P_C,
\quad
F_{\mathrm{wrong}},
\quad
F_{\mathrm{split}},
\quad
F_{\mathrm{deadline}},
\quad
D_{95},D_{99},
\quad
E.
\]

Scale regret:

\[
\operatorname{Regret}_{\mathrm{scale}}(N)
=
J_{\mathrm{shared}}(N)
-
J_{\mathrm{expert}}(N),
\]

where \(J\) is evaluated only among policies satisfying reliability constraints.

Normalized regret:

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
\Pr(\text{shared checkpoint feasible at }N)
}{
\Pr(\text{scale-specific expert feasible at }N)+\varepsilon
}.
\]

## 6.6 OOD axes

Test one OOD axis at a time:

- node count;
- density;
- road geometry;
- evidence covariance;
- PHY load;
- sensor-group structure;
- protocol profile;
- mobility and handoff.

## 6.7 Statistical design

Minimum:

- 5 model training seeds;
- 30 scene seeds;
- dynamic MC trials sufficient to estimate target failure probability;
- nested bootstrap separating model variance and scene variance;
- common random variables where possible, not merely reused RNG seed.

For rare failure probabilities, use upper confidence bounds:

\[
p_{\mathrm{upper}}
\approx
\frac{3}{M}
\quad
\text{when zero failures are observed.}
\]

If target \(p\le10^{-3}\), use at least 3000 zero-failure trials or rare-event estimation.

## 6.8 Acceptance

Performance-scale validation is successful if:

1. ESP/ESD-GNN remains feasible across target scales under fixed service profile.
2. Scale regret remains bounded and reported.
3. Failure modes under fixed protocol are explained, not hidden.
4. Computation remains near-linear.
5. Results are dynamic-MC judged, not analytic-only.

---

# 7. Metric Naming Cleanup and Code Entropy Reduction

## 7.1 Problem

The codebase currently risks mixing:

- macrostate basin outcomes;
- strict node-pair safety audits;
- old node-union/product surrogates;
- diagnostic node means;
- per-node spatial fields.

This creates a high risk that future paper text or plots accidentally revive legacy metrics.

## 7.2 Required metric namespaces

Use explicit namespaces:

### Headline macrostate metrics

```text
macro_P_correct
macro_F_wrong
macro_F_split
macro_F_deadline
macro_T_confirm
macro_D95
macro_D99
macro_CVaR99
```

### Strict audit metrics

```text
strict_any_disagreement
strict_any_wrong
strict_any_unfinalized
```

### Legacy / surrogate metrics

```text
surrogate_product_S_allcorrect
surrogate_nodeunion_F_wrong
surrogate_nodeunion_F_disagree
```

### Diagnostic node metrics

```text
diagnostic_node_failure_mean
diagnostic_node_wrong_mean
diagnostic_node_deadline_mean
diagnostic_spatial_F_i
```

### Effective-sampling metrics

```text
sampling_progress_g
sampling_drift_delta
sampling_keff
sampling_selected_corr
sampling_minority_exposure
sampling_receiver_load
```

### CDQ control metrics

```text
cdq_eta
cdq_guard_active
cdq_risk_slack_wrong
cdq_risk_slack_split
cdq_deadline_pressure
```

## 7.3 Forbidden names

Ban ambiguous names in headline code and result JSON:

```text
F
F_wrong
F_disagree
S_allcorrect
failure
reliability
D
delay
```

They may appear only in local variables inside small functions, never in serialized results or paper tables.

## 7.4 Result schema

Every result JSON must include:

```json
{
  "metric_namespace_version": "macrostate_v2",
  "policy": "...",
  "query_family": "ESP|CDQ2|Guarded-CDQ2",
  "physics_hash": "...",
  "service_profile_hash": "...",
  "evidence_hash": "...",
  "scene_distribution_hash": "...",
  "macro": {},
  "strict_audit": {},
  "diagnostic": {},
  "sampling": {},
  "cdq": {},
  "runtime": {}
}
```

## 7.5 Code entropy reduction

Refactor old overlapping utilities into a small number of canonical modules:

```text
metrics/macro.py
metrics/strict_audit.py
metrics/diagnostic_node.py
metrics/effective_sampling.py
metrics/schema.py

experiments/eta_curve.py
experiments/guarded_cdq2.py
experiments/hazard_profiles.py
experiments/esp_scale_validation.py
```

Deprecated names should raise warnings first, then errors.

---

# 8. Current Research Claim After This Round

The intended final claim after these experiments should be:

1. CDQ2 is the general query family.
2. ESP is the reliability-first specialization \(\eta=0\).
3. Positive \(\eta\) improves liveness/deadline behavior in correlated-evidence regimes.
4. Positive \(\eta\) can raise wrong risk in majority-correct settings.
5. Guarded-CDQ2 enables liveness gains only when wrong/split constraints have slack.
6. ESP/ESD-GNN remains the default reliability policy.
7. The project’s main contribution is not “CDQ always improves reliability,” but the characterization and control of the validity-liveness trade-off induced by diversity-aware sampling.
