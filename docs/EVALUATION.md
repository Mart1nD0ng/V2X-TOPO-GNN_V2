# Evaluation

Scalability claims require seeded, reproducible evaluation.

## Required Scales

- 100 nodes.
- 500 nodes.
- 2000 nodes.
- 5000 nodes.
- 10000 nodes.

## Required Baselines

- KNN sweep.
- MST+augmentation.
- Greedy SINR.
- Degree-capped spanner.
- NodeGNN-only once implemented.
- Full hierarchical model once implemented.

## Required Metrics

- M1 environment calibration: candidate edge count, candidate average degree, candidate giant component ratio, cell occupancy, candidate pair checks per node, cap hit ratio, build wall time, baseline mean out degree, baseline undirected average degree, baseline giant component ratio, link success quantiles, and `query_success_proxy`.
- M2.2 topology-aware evaluator bridge: node-wise `p_correct_query`, explicit `p_wrong_query`, `p_link_response`, `p_link_no_response`, `p_neutral_query`, `p_no_support_query`, edge-slot effective query degree, unique-peer effective query degree, duplicate edge count, isolated node count, support-size counts relative to `k`, node mean/min/10th-percentile correct decision probability, node mean wrong decision probability, undecided mean, and expected rounds mean. `p_no_response` is a legacy alias for `p_no_support_query`.
- M2.3 graph-coupled mean-field evaluator: node-wise correct, wrong, undecided, and expected-round outputs; graph-level mean/min/10th-percentile correct decision diagnostics; optional histories for preference propagation, quorum probabilities, and state mass error.
- M2.4 small-N exact joint reference validation: max absolute error between M2.3 and the exact joint recurrence for tiny graphs, exact-reference scope limits, deterministic agreement cases, and documented mean-field gap cases.
- M3 V2X consensus bridge: sparse V2X topology inputs, differentiable channel-proxy `link_success`, graph-coupled Avalanche C/D metrics, consensus energy proxy E, node-wise C/D/E outputs, graph quantiles, bridge/link degradation diagnostics, and saturation diagnostics before training.
- M3.1 V2X bridge calibration: M1-to-M3 sparse input adapter, failure-domain metrics `F = P_wrong + P_undecided`, reliability nines, reliability regime classification, and gradient-band diagnostics.
- M3.2 tail-aware bridge calibration: mean, tail, and global reliability weakening diagnostics; smooth failure-tail metrics; and toy, small-realistic, and realistic Avalanche sweep profiles.
- M2/M2.2/M2.3 closed-form evaluator phases: C_avalanche.
- M2/M2.2/M2.3 closed-form evaluator phases: D_avalanche.
- M3 evaluator bridge: E_avalanche as a metric only.
- Later coupled optimization phases: C/D/E may be consumed by a separate objective.
- Edge count.
- Average degree.
- Memory.
- Inference latency.

## Reproducibility

Every result must include seed list, config hash, git commit, mean, and standard deviation.

## Harness Status

`make scale-smoke` currently checks sparse-complexity guardrails. `make env-feasibility-check` runs non-mutating M1 smoke calibration before any GNN training is considered valid. `make env-feasibility-report` refreshes canonical M1 reports.

M2.2 topology bridge checks reject self-loops and duplicate directed peer edges by default, so topology query graphs are simple directed peer-query graphs unless a diagnostic path explicitly enables otherwise. If multi-edges are enabled, edge-slot diagnostics and unique-peer diagnostics must be interpreted separately. K-related feasibility warnings for hard distinct-peer querying should use unique-peer counts and effective unique-peer degree. The evaluator remains a static one-hop mean-field bridge before graph-coupled recurrence is added.

M2.3 adds deterministic graph-coupled mean-field recurrence over node marginal Snowball states. It is intended to reveal topology propagation effects such as weak cuts and bridge bottlenecks. It is not an exact full joint Avalanche safety proof because the full joint state process is exponential in node count.

M2.4 keeps the exact joint process out of production code. The reference is limited to `num_nodes <= 5`, `beta <= 3`, and `rounds <= 6`, and exists only in tests and explicit analysis scripts. `scripts/analysis/compare_mean_field_exact_snowball.py` writes its comparison report under `.agent/tmp/` when run directly.

M3 connects M1 channel proxies to M2.3 through `src/evaluation/v2x_consensus_bridge.py`. It accepts only sparse directed edge lists and never constructs dense node-by-node tensors. The bridge computes per-edge path loss, received power, SINR, packet success probability, and then passes explicit `link_success` plus explicit initial wrong-preference probabilities into the graph-coupled evaluator. Link success, SINR, and channel terms are internal diagnostics, not public objectives.

The M3 energy term is a deterministic proxy:

```text
per_edge_query_energy_ij = packet_duration_s *
    (tx_power_watt + rx_power_watt + processing_power_watt)
E_round_i = k * sum_j q_ij per_edge_query_energy_ij
E_consensus_i = node_expected_rounds_i * E_round_i
```

This is not a full NR sidelink power model. Reports must preserve `node_consensus_energy_j`, `E_consensus_node_mean`, `E_consensus_node_p90`, and `E_consensus_total` separately from C/D. Saturation diagnostics such as `saturated_correct_count`, expected-round ranges, and `flat_region_warning` must be checked before using these metrics in training.

M3.1 adds reliability-domain calibration. High `C_avalanche` is desirable. It should not be treated as bad merely because it is close to one. The useful reliability training domain is failure probability:

```text
F_i = P_wrong_i + P_undecided_i
```

The evaluator reports `F_avalanche_node_mean`, `F_avalanche_node_p90`, `F_avalanche_node_max`, and reliability nines `-log10(F + eps)`. Regimes are classified from mean failure and tail failure:

- `below_target`: failure is well above the configured target band.
- `near_target_boundary`: failure is near the configured target.
- `above_target_high_reliability`: failure is well below target, so reliability is already satisfied.

M3.2 makes reliability weakening tail-aware. High reliability is desirable, but future reliability pressure should weaken only when both mean and tail failure are already above target:

```text
mean_reliability_loss_should_weaken = mean_regime == above_target_high_reliability
tail_reliability_loss_should_weaken = tail_regime == above_target_high_reliability
global_reliability_loss_should_weaken =
    mean_reliability_loss_should_weaken and tail_reliability_loss_should_weaken
```

`reliability_loss_should_weaken` is kept as a backward-compatible alias for the global diagnostic flag. `optimize_delay_energy_next` is true only when the global flag is true. These are diagnostics for M4, not a training objective.

The evaluator also reports a smooth tail metric:

```text
F_soft_tail = tau * log(mean(exp(F_i / tau)))
reliability_nines_soft_tail = -log10(F_soft_tail + eps)
```

M4 should use failure-domain barrier terms over both mean and tail failure. A single bad-tail node should keep the global reliability weakening flag false even if mean failure is already low.

M4 implements the reliability barrier over evaluator outputs only:

```text
L_R = softplus((log(F + eps) - log(F_target)) / tau_R)
```

where `F = P_wrong + P_undecided`. The coupled objective uses mean and tail failure barriers, plus delay and energy softplus barriers over `D_avalanche` and `E_avalanche`. It does not optimize link success, SINR, BLER, HARQ, coverage, path loss, or average link reliability as direct loss terms. Once both mean and tail failure are below target, reliability pressure can weaken and delay/energy should dominate.

M4.1 hardens the tail gate. The default loss tail mode is `max`; the `softmax_tail` option is a conservative smooth maximum,

```text
F_soft_tail = tau * logsumexp(F_i / tau)
```

not the M3 diagnostic smooth mean. It intentionally does not subtract `log(num_nodes)`, so a single bad local failure probability keeps the tail barrier active. A normalized `softmean_tail` can be reported for diagnostics, but it is not the default reliability gate.

The loss layer validates failure-domain probabilities before computing barriers. Invalid evaluator outputs, such as negative failure probability, probabilities greater than one, or `P_wrong + P_undecided > 1`, raise `ValueError` except for tiny numerical roundoff. PCGrad and GradNorm remain standalone utilities until a later training integration phase.

M4.2 adds evaluator-loss integration validation. The tests run M3 sparse V2X evaluator outputs through the M4 coupled loss, check finite failure/delay/energy barriers, verify gradients through sparse topology weights, edge distances, and initial preferences, and smoke-test PCGrad/GradNorm on the actual loss components. It remains a test harness before M5 topology construction; no training loop or optimizer step is introduced.

M5 adds the first executable topology constructor. `TopologyConstructionLayer` consumes sparse candidate edge scores and emits a sparse deployable query distribution for M3. Constructor reports must include active edge count, degree cap hits, row-weight sums, selected/unselected candidate counts, top-k boundary margins, near-miss candidates, gradient coverage fraction, and zero-gradient-risk rows. The 10k constructor smoke uses O(NK) sparse candidates and asserts `active_edge_count <= N * max_out_degree` in top-k mode. M5.2 also evaluates all-candidate warm-start mode, where gradient coverage should be one over feasible sparse candidates.

M6 adds the first sparse model-side scorer. `HierarchicalGNNScorer` maps sparse V2X candidate graph features to candidate `edge_score` values using node/edge MLP encoders, sparse message passing, optional region pooling, and an edge score head. The model-check suite verifies a 10k sparse forward pass and a smaller backward pass. Integration tests prove GNN parameters receive finite nonzero gradients through:

```text
edge_score -> TopologyConstructionLayer -> M3 evaluator -> M4 coupled loss
```

Both `support_mode="all"` warm-start coverage and `support_mode="topk"` deployable support selection are checked. M6 does not add training loops, optimizers, checkpointing, datasets, or experiment runners.

M6.1 adds scorer hardening and plan-alignment checks. The scorer defaults to xavier initialization for normal use, supports kaiming as an alternate initializer, and keeps deterministic initialization only for tests/debugging. Region scale runs should pass explicit `num_regions` instead of relying on inferred max-region id. Model diagnostics now include node/edge embedding norm summaries, edge-score quantiles, source-row score entropy, incoming message counts, zero-incoming nodes, zero-outgoing candidate nodes, and candidate-graph isolated nodes. Environment-derived integration tests run M1 snapshot/candidate outputs through:

```text
HierarchicalGNNScorer -> TopologyConstructionLayer -> M3 evaluator -> M4 loss
```

and verify finite nonzero gradients to scorer parameters.

M6.2 adds structural head outputs and region-aware score bias:

- `node_budget_logits` / `node_budget_expected`
- `node_role_logits`
- `sector_preference_logits`
- sparse per-candidate `region_bridge_logits`
- `edge_score_base`, `structural_bias`, and final `edge_score`

The budget head is soft/diagnostic and is not converted to hard `per_node_budget` inside normal integration tests. Sector, role, and sparse region-bridge heads may bias candidate scores, but they are still scorer components, not final constrained optimizers. M6.2 does not add a training loop, does not add structural losses, and does not expose link/SINR/BLER/HARQ/coverage variables as objectives. The 10k model smoke now covers structured forward outputs with explicit `num_regions` and `edge_sector_id`.

M6.3 adds structural-head diagnostics and a deterministic budget-to-constructor bridge. `compute_structural_diagnostics(...)` reports budget expectation quantiles and entropy, role/sector entropy and argmax counts, sparse bridge-logit summaries, cross-region versus same-region bridge summaries, and structural-bias component magnitudes. `expected_budget_to_cap(...)` maps soft `node_budget_expected` values to integer constructor caps with explicit min/max clipping. The adapter is detached by default because integer caps are nondifferentiable; `detach=False` is intentionally unsupported.

Environment-derived budget integration now validates:

```text
HierarchicalGNNScorer
  -> node_budget_expected
  -> detached integer per_node_budget
  -> TopologyConstructionLayer
  -> M3 evaluator
  -> M4 loss
```

Gradients through the edge-score and structural-bias path remain finite, while the integer budget adapter does not claim gradients for the budget head. Structural ablation smoke tests compare no-bias, sector-only, role-only, bridge-only, and all-bias paths to ensure each enabled structural component can change sparse score distributions without introducing direct link/SINR/BLER/HARQ/coverage losses.

M6.4 adds structural objective and training-readiness diagnostics. The main C/D/E coupled loss is still the only evaluator objective. Structural auxiliary utilities are optional diagnostics for future curricula:

- budget entropy,
- budget target matching,
- sector entropy,
- role balance,
- bridge-logit regularization.

Tests prove that the main C/D/E path gives gradients to the edge scorer, sector head, role head, and bridge head when structural bias is enabled. Tests also prove that the budget head does not receive main-loss gradients through detached integer caps. Budget-head gradients require auxiliary objectives or a future differentiable surrogate. Cap diagnostics must include cap min/mean/max, histogram, saturation counts, and cap-change count when previous caps are available.

`configs/training_smoke.yaml` is a planning artifact for M7, not an executable trainer. It keeps warm-start `support_mode="all"` and capped `support_mode="topk"` phases explicit, lists C/D/E targets, and records optional structural auxiliary weights. M7 runs must not mix different graph rules between train, validation, and deployment within one experiment.

M7 makes that planning artifact executable as a tiny deterministic smoke only. `run_tiny_training_smoke(...)` reports per-step C/D/E loss terms, failure-domain reliability metrics, gradient norms, active edge counts, selected fractions, gradient coverage, cap diagnostics when caps are used, and separate edge-path, structural-head, and budget-head gradient maxima.

The smoke must be interpreted as a contract check, not model training. The all-candidate path is the phase-0 warm-start proof for high gradient coverage. The top-k path is the deployable support-selection proof and can have lower gradient coverage. Budget-head gradients through integer caps remain absent by design; if `budget_strategy="auxiliary"` is enabled, budget-head gradients come from the explicit structural auxiliary objective only.

M7.1 extends the report with trend and reproducibility diagnostics: loss delta, relative loss delta, nonincreasing-step count, best step, parameter-change L2 norm, per-step gradient norms, maximum gradients, finite/nonfinite metric counts, and C/F/D/E deltas. These trend fields are diagnostics, not pass/fail proof of convergence.

The default `toy` Avalanche profile is retained for speed in agent checks. The explicit `small_realistic` smoke uses `k=5`, `alpha=3`, `beta=5`, and `rounds=20`; it should pass before claiming the smoke is stable beyond toy consensus parameters. Optional gradient clipping reports pre- and post-clip norms without adding a scheduler or production optimizer framework.

The all-to-top-k curriculum smoke carries model parameters forward from phase 0 to phase 1. It must report active-edge-count changes, gradient-coverage changes, finite gradients in both phases, and parameter movement in both phases.

M7.2 requires curriculum reports to disclose phase-specific strategy choices. Phase 0 is the all-candidate warm-start path and defaults to fixed budget strategy with the budget head disabled. Phase 1 is the top-k path and may enable `phase_1_budget_strategy="auxiliary"` with `phase_1_train_budget_head=true`. If enabled, budget-head gradients are attributed only to the auxiliary objective; detached hard caps are still not treated as differentiable. Reports must include phase budget strategies, train-budget-head flags, final auxiliary losses, final budget-gradient maxima, and whether phase 1 auxiliary budget training was active.

M8 measures scale/runtime/memory viability for the current sparse pipeline; it does not measure model quality and does not replace real training or ablation studies. M8.1/M8.2 add tiered runtime checks:

- `--tier micro`: N=40, top-k support, toy Avalanche. This is a tiny dependency/profiling smoke for very slow machines.
- `--tier fast`: N=100, all/top-k support, toy Avalanche. This is CI/agent-check safe.
- `--tier medium`: N=100/500, all/top-k support, toy plus small-realistic Avalanche. This is explicit analysis.
- `--tier large`: N=100/500/2000/5000/10000, top-k toy by default. This is manual/explicit only.

`--smoke` is a backward-compatible alias for the fast tier. Reports write `.agent/tmp/scale_training_smoke.json` and `.agent/tmp/scale_training_smoke.md` by default. `--node-counts` can override a tier's node list for an explicit subset such as `100,500,2000,5000`, and `--topk-backend` can pin the top-k backend for explicit top-k runs. `--visualization-dir` writes deterministic SVG charts, a CSV table, a Markdown summary, and a visualization index for the completed sparse scale cases.

Each M8 case reports node count, candidate and active edge counts, mean out-degree, Avalanche profile, case status/error fields, optional configured timeout, case wall time, environment generation time, model initialization time, constructor setup time, scorer-forward time, topology-constructor time, evaluator time, loss time, backward time, postprocess time, tensor-memory proxy, gradient coverage, parameter and structural gradient maxima, C/D/E metrics, failure-domain reliability, and cap histograms when caps are used.

The tensor-memory proxy is deterministic tensor accounting over feature tensors, topology tensors, evaluator outputs, model parameters, and model gradients. It is not process RSS and not GPU peak memory. Timing values are environment-dependent and should be compared on the same machine; `total_step_time_s` means forward plus backward only, while `case_wall_time_s` includes environment and setup overhead.

M8.2 adds bottleneck summaries: dominant stage, bottleneck time, bottleneck fraction of forward/backward step time, and edge/node throughput proxies. Medium and large reports also include aggregate maxima, failed-case counts, status counts, and all-versus-top-k active-edge and gradient-coverage summaries.

`scripts/analysis/check_runtime_environment.py` is an explicit dependency sanity diagnostic. It reports the Python executable/version, NumPy/SciPy/PyYAML/Torch import status and versions, and CUDA availability. CUDA is not required.

All-mode is expected to improve gradient coverage because every feasible sparse candidate remains in the row-softmax support. Top-k mode is expected to reduce active edge count and memory at the cost of excluding candidates from the current gradient path. The small-realistic Avalanche profile is more expensive than toy because the graph-coupled recurrence uses larger beta and more rounds; realistic profiles should be explicit analysis runs, not normal agent checks.

`scale-training-fast-check` is included in normal `agent-check`. `scale-training-medium-check` and `scale-training-large-check` are explicit opt-in targets so contract checks remain practical.

M9 adds small-scale training stability and ablation reporting. `scripts/analysis/run_training_stability_smoke.py --smoke` runs the fast one-seed `baseline_all` case and writes `.agent/tmp/training_stability_smoke.json` plus `.agent/tmp/training_stability_smoke.md`. It reports initial/final/min total loss, C/D/E/F initial and final values, L_R/L_D/L_E initial and final values, reliability gate state, gradient coverage, active edge count, structural and budget gradient maxima, cap histogram, runtime, best step, nonincreasing-loss fraction, and metric direction deltas.

`--analysis` runs the baseline, top-k, structural-bias-off, budget-auxiliary-off, and short small-realistic cases across analysis seeds. The ablation comparisons report deltas against `baseline_all`. These runs remain deterministic harness checks; they do not add datasets, checkpointing, large-scale experiment management, or model selection.

M9.1 makes the stability harness tiered:

- `--tier smoke`: one seed, `baseline_all`, short step count. This is the agent-check tier.
- `--tier short`: one seed over baseline, top-k, structural-bias-off, and budget-auxiliary-off cases.
- `--tier analysis`: multiple seeds over the full ablation set.

`--smoke` and `--analysis` remain backward-compatible aliases. Reports include per-case status/error fields and aggregate counts by status. The recorded timeout field is informational and not hard enforcement. If a short or analysis case fails, the report keeps the error and continues remaining cases while setting `contract_ok=false`.

Ablation notes are intentionally conservative. Removing the budget auxiliary objective can lower total loss by removing a penalty; compare C/F/D/E, active edges, and gradient coverage separately. Structural-bias-off differences over 2-5 steps are smoke diagnostics, not final ablation conclusions.

M9.2 adds metric-focused ablation reporting. Each comparison includes total-loss delta, primary C/D/E loss delta excluding structural auxiliary terms, L_R/L_D/L_E deltas, structural auxiliary delta, C/F/D/E metric deltas, active-edge-count delta, gradient-coverage delta, and reliability-regime changes. Interpretation categories prevent common misreads:

- `auxiliary_removed_only` means total loss improved mainly because an auxiliary objective was removed.
- `support_change_confounded` means all-candidate and top-k support changed simultaneously, so topology support and metric effects are coupled.
- `structural_effect_small` means the short-run structural-bias toggle barely moved the reported metrics.
- `unstable_or_failed` means at least one compared case did not complete cleanly.

Seed-variance summaries report per-case final-loss, primary-loss, F, D, E, and parameter-change mean/std across seeds. Single-seed tiers report zero std and state that seed variance is not estimated.

M9.3 adds readiness diagnostics for deciding whether M10 can start. The readiness gate reports `ready_for_m10_minimal_experiment_harness`, `readiness_status`, reasons, blocker counts, ambiguous ablation count, support-confounded comparison count, and auxiliary-only improvement count. The gate blocks on failed cases, nonfinite losses, nonfinite gradients, zero-gradient cases, or exploding-loss diagnostics. It does not require loss decrease.

Per-case trend categories are deterministic diagnostics only: `decreasing`, `flat`, `increasing`, `unstable`, or `not_enough_steps`. They are based on short-run relative movement and are not statistical significance claims. `--summary-only --json-in <report>` can summarize an existing JSON report without rerunning cases.

M10 adds a minimal experiment harness over M9 reports:

```bash
python -B scripts/analysis/run_minimal_experiment.py --tier micro
python -B scripts/analysis/run_minimal_experiment.py --tier smoke
python -B scripts/analysis/run_minimal_experiment.py --tier short
```

The report includes per-case summaries, ablation comparisons against `baseline_all` within each seed, seed-variance summaries, readiness status, and conservative interpretation notes. Default outputs are `.agent/tmp/minimal_experiment.json` and `.agent/tmp/minimal_experiment.md`; canonical reports are not updated by default. M10 does not make final model-quality claims.

M10.1 moves the tier definitions into `configs/minimal_experiment.yaml` while keeping code defaults as a fallback. The report records the planned cases, planned seeds, max steps, and agent-check safety for each tier. It also renames stage semantics: `minimal_experiment_contract_ok` and `ready_for_next_stage` describe readiness for the next controlled ablation stage, while `ready_for_m10_minimal_experiment_harness` remains only as a deprecated M9 compatibility alias. A `decision_summary` states the recommended action and cautions without claiming production training readiness.

M11 adds `configs/controlled_ablation.yaml` and a controlled ablation runner. It compares baseline all-candidate support against top-k support, full structural-bias removal, sector/role/bridge bias removal, and budget-auxiliary removal. Each comparison is within the same seed and reports total-loss delta, primary-loss delta, F/D/E deltas, active-edge delta, gradient-coverage delta, and a conservative interpretation category. Failed cases are recorded and remaining cases continue; any failed, nonfinite, or zero-gradient case blocks the controlled ablation contract.

Only `controlled-ablation-micro-check` is intended for normal checks. Smoke and short controlled ablation tiers are explicit diagnostics because they run more cases.

M11.1 adds paired effect-size and consistency summaries. Reports include C/F/D/E, primary-loss, active-edge, gradient-coverage, and parameter-change deltas versus same-seed `baseline_all`. Multi-seed tiers aggregate mean/std/min/max deltas and sign-consistency fractions. Primary-benefit claims are disallowed when the comparison is support-confounded, auxiliary-only, failed, inconsistent across seeds, or single-seed-only. Top-k comparisons remain support-confounded unless active support and gradient coverage are controlled separately. Budget auxiliary removal must be interpreted through primary C/F/D/E metrics rather than total loss alone.

M12 adds selected ablation findings reports. `scripts/analysis/report_controlled_ablation_findings.py --json-in .agent/tmp/controlled_ablation.json` summarizes an existing controlled ablation JSON without running new cases by default. It can optionally run a selected `micro`, `smoke`, or `short` controlled-ablation tier when explicitly requested with `--run-tier`.

The findings report is a decision layer, not a trainer. It includes an executive summary, readiness status, case table, interpretation table, paired effect-size summary, confounded comparisons, auxiliary-only improvements, meaningful effects, inconclusive cases, recommended next actions, and a "what not to claim" section. Recommendation categories are conservative: `candidate_for_deeper_study`, `needs_multi_seed_validation`, `needs_longer_run`, `needs_controlled_support_match`, `ignore_for_now`, and `blocked_due_to_failure`.

M12 only recommends selected follow-up when primary-benefit claims are allowed by the controlled report. Support-confounded top-k cases are routed to controlled support-matching follow-up. Auxiliary-only total-loss drops are not treated as topology-quality improvement. Single-seed effects require multi-seed validation before being promoted.

M13 adds selected ablation evidence runs. `scripts/analysis/run_selected_ablation_evidence.py --preset smoke` runs the small baseline/structural pair and feeds the controlled report back through the M12 findings layer. `--preset short` runs a wider selected set across two seeds and remains explicit only.

M13 still does not create final model evidence. Its purpose is to collect stronger small-scale evidence for structural ablation cases before deciding whether deeper stability runs are worthwhile. Support-confounded top-k comparisons and auxiliary-only total-loss drops are retained in the report but are not promoted as structural follow-up cases. If no structural case is recommended, the report states `recommendation_status = no_structural_claim_yet` and recommends increasing steps or refining structural heads rather than forcing a claim.

M13.1 makes `configs/controlled_ablation.yaml:selected_evidence` the source of truth for selected evidence presets. Built-in presets are fallback only. Selected reports now include `no_recommendation_reasons`, `evidence_strength`, and `selected_evidence_readiness`. No recommendation is valid and should not be treated as failure when the evidence is negligible, single-seed only, support-confounded, or auxiliary-only. The short selected tier can optionally compare against a smoke report with `--compare-smoke-json`, but it remains explicit and is not part of the normal agent-check path.

M14 adds structural-head signal auditing. The audit measures base score magnitude versus structural bias magnitude, source-row top-1/top-k ranking changes, active top-k overlap with structural weights disabled, and one-pass gradient allocation across the base encoder, edge encoder, message layers, score head, sector head, role head, bridge head, and budget head. It then runs a tiny structural ablation micro-audit and classifies the result as too weak, gradient-starved, not metric-relevant under smoke, auxiliary/support-confounded, or observed.

The M14 classification is a diagnostic for follow-up design, not evidence of model quality. A `structural_signal_too_weak` result should trigger structural-weight refinement before longer runs; `structural_gradient_starved` should trigger head/path revision; `structural_signal_not_metric_relevant_under_smoke` should trigger longer or more targeted runs; and confounded results should defer structural claims.

M14.1 adds direction-aware audit fields. Reports include all-structural minus structural-zero deltas for primary loss, total loss, auxiliary loss, C, F, D, and E, plus `structural_effect_direction` and `structural_effect_magnitude`. A visible structural signal can be harmful if primary C/D/E loss increases and failure, delay, or energy worsen. Top-k support changes are reported as confounders because score bias can change both edge weights and active membership. Gradient allocation is split into main C/D/E loss, auxiliary structural loss, and combined loss; budget auxiliary gradients must not be interpreted as main reliability/delay/energy gradients.

M14.2 calibrates structural bias weights before longer evidence runs. The structural weight sweep compares each sector/role/bridge weight setting against the all-zero structural baseline under both all-candidate and top-k support. Each row reports primary-loss and C/F/D/E deltas, row-ranking changes, active support changes, structural-to-base score ratio, direction classification, magnitude, and safe-policy flags. A top-k setting is considered safe only when primary loss, failure probability, C, and top-k membership do not materially worsen. If no nonzero top-k setting is safe, top-k follow-up runs should disable structural score bias by default instead of treating visible structural heads as useful.

M15 uses the calibrated M14.2 policy in targeted selected runs. The report compares `all_low_structural` against `all_zero_structural`, keeps `topk_zero_structural` as the default deployable top-k policy, includes `topk_high_structural_unsafe_reference` only as an unsafe reference, and evaluates a calibrated all-to-top-k curriculum. Reports must include primary-loss and C/F/D/E deltas, active-edge and gradient-coverage deltas, structural policy fields, interpretation categories, claim gates, and recommendation status. Passing M15 means the calibrated policy is numerically healthy enough for selected follow-up; it is still not final model evidence.

M15.1 strengthens the calibrated follow-up report for the explicit short tier. The short tier aggregates each comparison across seeds and reports seed counts, successful/failed seed counts, mean/std/min/max deltas for primary loss, total loss, F, C, D, and E, sign-consistency fractions, claim-gate pass fractions, and recommendation consistency. The calibrated policy is considered ready for selected training smoke only when low all-mode structural weights and the calibrated curriculum are non-worse for every successful seed and the high-weight top-k reference remains stably unsafe. If any seed fails, readiness is at most conditional. If high-weight top-k is not stably unsafe, the policy must be re-audited. Top-k structural bias remains disabled by default.

M15.2 separates calibrated follow-up artifacts by tier. Smoke runs write
`.agent/tmp/calibrated_structural_followup_smoke.json` and `.md`; short runs
write `.agent/tmp/calibrated_structural_followup_short.json` and `.md`. Normal
agent checks run only the smoke tier and must not overwrite short-tier evidence.
Smoke reports can validate the code path but cannot authorize M16 selected
training smoke. M16 is allowed only from short-or-stronger evidence with at
least two successful seeds, zero failed seeds, stable all-low non-worse gates,
stable calibrated-curriculum non-worse gates, and a stably unsafe high-weight
top-k reference.

M16 adds `scripts/analysis/run_selected_training_smoke.py`, a selected
two-phase smoke harness for the calibrated policy. It refuses to run from a
missing or smoke-tier readiness artifact unless `--force` is supplied and
recorded. Phase 0 uses all-candidate support with low structural weights
`0.01/0.01/0.01`; phase 1 carries the same model parameters into top-k support
with structural score-bias weights `0.0/0.0/0.0`. The report records C/F/D/E,
loss, gradient, parameter-change, and readiness fields. Passing M16 validates
numerical trainability of the calibrated policy only; it is not model-quality or
production-training evidence.

M17 adds selected-training smoke interpretation. The report-only script reads
`.agent/tmp/selected_training_smoke.json` and writes
`.agent/tmp/selected_training_smoke_interpretation.json` plus `.md`. It reports
phase loss deltas, total selected-policy loss delta, phase-switch loss delta,
gradient coverage drop, top-k policy flags, readiness tier, finite loss/gradient
flags, parameter movement, and C/F/D/E deltas when the source report contains
initial and final metric trends. If those trends are absent, M17 reports the
`missing_metric_trends` caution rather than promoting the run.

M17 can classify a report as `blocked`, `requires_policy_revision`,
`numerically_healthy_only`, or `promising_but_insufficient`. It may authorize
only a selected short training smoke stage, and only under conservative
contract and metric-trend conditions. It does not claim model quality or replace
multi-seed selected training evidence.

M16.1 adds those source metric trends to the selected training smoke report.
Each phase now records initial, final, and delta values for mean C, mean failure
probability F, mean delay D, and mean consensus energy E. When available, the
same snapshot also records failure-tail, D-p90, and E-p90 trends. These fields
let M17 distinguish numerical health from primary-metric movement. Improvement
is necessary before promotion, but it is still not a model-quality claim without
selected short and multi-seed evidence.

M16.2 adds multi-seed selected smoke aggregation. The runner reuses M16 for each
configured seed, reports per-seed phase C/F/D/E/loss deltas, and aggregates
mean/std/min/max plus sign-consistency fractions. A run is ready for M18 only
when all seeds are numerically healthy, top-k structural bias is disabled, the
short-tier readiness artifact authorizes the run, at least three seeds succeed,
and C or F improves consistently in both phases without D/E worsening. Mixed but
healthy effects recommend repeating M16 with more steps rather than starting
selected short training.

M18 runs selected short training under the same calibrated policy. The default
readiness source is `.agent/tmp/selected_training_multiseed.json`; smoke-scale
or unready sources do not authorize the run unless explicitly forced. Phase 0
uses all-mode low structural weights `0.01/0.01/0.01`, and phase 1 uses top-k
support with structural score bias disabled. Reports aggregate per-seed C/F/D/E
and loss deltas, gradient coverage, finite-loss and finite-gradient fractions,
parameter movement, sign consistency, and whether the effects exceed the M16
tiny-effect threshold. This is still a controlled short harness and not a
model-quality claim.

M19 interprets the selected short report without rerunning training. It blocks
on failed seeds, nonfinite loss or gradients, contract failures, missing M18
readiness, enabled top-k structural bias, or high-weight top-k use. It requires
policy revision when C/F/D/E worsen consistently. Only when C and F move in the
favorable direction across both phases, D/E remain non-worse, and all safety
constraints hold can M19 set
`ready_for_M20_controlled_selected_short_ablation=true`. This is a readiness
signal for controlled ablation only, not a model-quality claim.

M20 adds controlled selected-short ablation around the calibrated policy.
`scripts/analysis/run_controlled_selected_short.py` requires the M19
interpretation artifact by default and compares the calibrated policy against a
zero-structural baseline plus a high-weight top-k unsafe reference. The report
uses paired deltas against `zero_structural_policy` for primary loss, C, F, D,
E, active support, and gradient coverage. Claims are blocked when changes are
tiny, support-confounded, total-loss-only, unsafe-reference, or when top-k
structural bias is enabled. M20 can only recommend deeper selected study; it
does not claim final model quality or production training readiness.

M20.1 adds `scripts/analysis/run_support_matched_audit.py` to isolate top-k
support membership from structural score effects. The audit records selected
sparse candidate-index overlap against `zero_structural_topk` and
`high_weight_topk`, classifies high-weight top-k as unsafe by policy, and uses
an analysis-only `support_matched_high_weight` proxy when exact support freezing
is not available. A support-effect observation can recommend a fixed-support
follow-up, but it must not promote high-weight top-k structural bias.

M20.2 adds `scripts/analysis/run_fixed_support_audit.py` for that fixed-support
follow-up. The audit freezes selected sparse candidate indices from a source
top-k policy and recomputes row-normalized weights on the same support under
zero, low, or high structural score policies. It reports support-source policy,
evaluation score policy, overlap against zero/high support, C/F/D/E and primary
loss deltas, and whether support-set or score-redistribution effects are
observed. All cases are analysis proxies with `claim_allowed=false`; they do
not authorize production training or high-weight top-k promotion.

`scripts/analysis/sweep_v2x_consensus_bridge.py` supports deterministic smoke profiles:

- `toy`: `k=1`, `alpha=1`, `beta=1`, `rounds=1`.
- `small-realistic`: `k=5`, `alpha=3`, `beta=5`, `rounds=20`.
- `realistic`: `k=20`, `alpha=14`, `beta=20`, `rounds=80`.

The realistic profile is intended for explicit analysis runs rather than normal agent checks. It remains deterministic and sparse, but has higher expected runtime because beta and rounds are larger.

## M21 Structural Redesign Review

M21 is a report-only decision layer:

```bash
python -B scripts/analysis/report_structural_redesign_review.py
```

It reads existing `.agent/tmp` audit artifacts and writes
`.agent/tmp/structural_redesign_review.json` plus `.md`. It does not rerun
training or produce new model-quality evidence.

The report summarizes each source artifact, records its limitations, and
produces per-head verdicts for sector, role, bridge, and budget heads. The
evaluation interpretation is conservative:

- high-weight top-k structural bias remains unsafe unless a future explicit
  audit changes the policy;
- no nonzero top-k structural weight is promoted by current reports;
- all-mode low structural weights are diagnostic-only when merely non-worse;
- budget heads are auxiliary-only until a differentiable budget surrogate is
  designed;
- fixed-support and support-matched audits are analysis proxies, not training
  benefits.

`ready_for_M22=true` means the evidence is sufficient to proceed to baseline
consolidation or structural redesign planning. It does not mean production
training, large-scale validation, or final topology-quality claims are ready.

## M22 Baseline Consolidation

M22 turns the M21 recommendation into an explicit baseline policy artifact:

```bash
python -B scripts/analysis/run_baseline_consolidation.py
```

The script reads `.agent/tmp/structural_redesign_review.json` by default and
writes `.agent/tmp/baseline_consolidation.json` plus `.md`. It is report-only
and does not run training or rerun evaluator cases.

The baseline report must disclose:

- the readiness artifact and whether it authorizes M22;
- the default all-mode pure edge-score warm-start policy;
- the default top-k pure edge-score deployment candidate;
- the structural-head status as diagnostic/redesign-only;
- the budget-head status as fixed, diagnostic, auxiliary, or future-surrogate
  only;
- allowed and disallowed claims.

The default deployment policy is `edge_score_topk_baseline`, where all
structural score-bias weights are zero. The default warm-start policy is
`edge_score_all_baseline`, also with structural weights zero. Low all-mode
structural weights remain a diagnostic reference only, and high-weight top-k
structural bias remains unsafe and unpromoted.

## M23 Pure Edge Baseline Stability

M23 runs the consolidated pure edge-score policies through a small deterministic
stability harness:

```bash
python -B scripts/analysis/run_pure_edge_baseline_stability.py
```

The harness reads `.agent/tmp/baseline_consolidation.json` by default and writes
`.agent/tmp/pure_edge_baseline_stability.json` plus `.md`. It evaluates the
all-mode warm-start baseline, the top-k deployment candidate, and an all-to-topk
curriculum that carries model parameters between phases. All structural
score-bias weights are zero in these pure baseline cases.

The report includes C/F/D/E deltas, loss deltas, finite gradient/loss flags,
parameter-change flags, gradient coverage, active edge count, and explicit
contract checks that structural bias does not affect `edge_score`. All-mode and
top-k losses must be interpreted with support and active-edge-count context; a
top-k loss value alone is not a model-quality conclusion.

## M24 Pure Edge Baseline Interpretation

M24 summarizes an existing M23 report without running new training:

```bash
python -B scripts/analysis/report_pure_edge_baseline_stability.py \
  --json-in .agent/tmp/pure_edge_baseline_stability.json
```

It writes `.agent/tmp/pure_edge_baseline_interpretation.json` plus `.md` by
default. The report separates healthy warm-start behavior, healthy deployment
candidate behavior, and all-to-topk curriculum transition health.

The all-to-topk transition is always marked support-change-confounded because
the constructor support rule changes from all sparse candidates to capped top-k.
Even large C/F/D/loss movement in that transition is not evidence that the model
learned a better topology. M24 may authorize only
`M25_pure_edge_selected_short_run`, and only when all pure baseline contract
checks pass.

## M25 Pure Edge Selected Short

M25 runs the authorized pure edge-score selected short harness:

```bash
python -B scripts/analysis/run_pure_edge_selected_short.py \
  --seeds 7,13,23
```

The default input is `.agent/tmp/pure_edge_baseline_interpretation.json`, which
must report `ready_for_M25_pure_edge_selected_short=true` unless the run is
explicitly forced. Default outputs are `.agent/tmp/pure_edge_selected_short.json`
and `.agent/tmp/pure_edge_selected_short.md`.

The M25 matrix contains all-mode, top-k, and all-to-topk pure edge-score cases.
All sector, role, and bridge structural score-bias weights are zero. Reports
aggregate loss and C/F/D/E deltas, finite loss and gradient fractions,
parameter-change fractions, gradient coverage, and active edge counts across
seeds.

The interpretation remains conservative: top-k and all-mode losses are not
direct quality rankings because support differs, and all-to-topk improvement is
support-rule-confounded. M25 establishes a clean reference baseline for later
interpretation; it does not validate production training or structural-head
benefits.

## M26 Pure Edge Selected Short Interpretation

M26 summarizes an existing M25 report without running new training:

```bash
python -B scripts/analysis/report_pure_edge_selected_short.py \
  --json-in .agent/tmp/pure_edge_selected_short.json
```

It writes `.agent/tmp/pure_edge_selected_short_interpretation.json` plus `.md`
by default. The report separates healthy all-mode warm-start behavior, healthy
top-k deployment-candidate behavior, and all-to-topk transition health.

The all-to-topk transition is always treated conservatively when active support
or gradient coverage changes between phases. Large C/F/D/loss movement in that
transition is not evidence that the model learned a better topology. M26 may
authorize only `M27_pure_edge_followup_training_or_ablation`, and only when all
pure-edge selected-short contract checks pass.

## M27 Pure Edge Follow-Up

M27 runs a stronger pure edge-score follow-up after M26 authorization:

```bash
python -B scripts/analysis/run_pure_edge_followup.py \
  --seeds 7,13,23
```

The default input is
`.agent/tmp/pure_edge_selected_short_interpretation.json`, which must report
`ready_for_M27_pure_edge_followup=true` unless the run is explicitly forced.
Default outputs are `.agent/tmp/pure_edge_followup.json` and
`.agent/tmp/pure_edge_followup.md`.

The M27 matrix contains all-mode, top-k, and all-to-topk pure edge-score
follow-up cases. All sector, role, and bridge structural score-bias weights are
zero. Reports aggregate C/F/D/E and loss deltas, finite loss and gradient
fractions, parameter-change fractions, gradient coverage, active edge counts,
and separate `health_claim_allowed` from `quality_claim_allowed`.

Quality claims remain disallowed. Top-k and all-mode losses are support-rule
dependent, and the all-to-topk transition is a deployment-transition health
check rather than evidence of model-quality improvement.

## M28 Pure Edge Reference Package

M28 consolidates existing pure edge-score artifacts into one reference package:

```bash
python -B scripts/analysis/build_pure_edge_reference_package.py
```

The package reads the default M22-M27 `.agent/tmp` artifacts and the manifest
`configs/pure_edge_baseline.yaml`. It writes
`.agent/tmp/pure_edge_reference_package.json` and
`.agent/tmp/pure_edge_reference_package.md`.

Each source artifact is reported with availability, status, key readiness
field, summary, and limitations. Missing artifacts produce
`incomplete_reference_package`; evidence is not fabricated. A complete package
requires the baseline consolidation contract, pure baseline stability contract,
M24 stable baseline interpretation, M26 stable selected-short interpretation,
and M27 stable follow-up status.

The package records allowed health claims and disallowed claims separately.
`ready_for_M29=true` is only a gate for the next controlled pure-edge baseline
extension, not a model-quality, production-training, or large-scale deployment
claim.

## M29 Pure Edge Baseline Extension

M29 runs the controlled pure edge-score extension:

```bash
python -B scripts/analysis/run_pure_edge_baseline_extension.py \
  --seeds 7,13,23 --max-steps-override 5
```

The default readiness source is `.agent/tmp/pure_edge_reference_package.json`.
It must report `ready_for_M29=true` unless the run is explicitly forced. The
policy matrix is all-mode, top-k, and all-to-topk pure edge-score cases with
all structural score-bias weights set to zero. Reports include C/F/D/E and loss
aggregates, finite loss/gradient fractions, gradient coverage, active edge
counts, node count, max steps, and Avalanche profile.

M29 can confirm only numerical health under a modest extension. It must not be
used to claim top-k superiority, optimal topology learning, all-to-topk quality
improvement, structural-head benefit, production training readiness, or
large-scale deployment validity.

## M30 Pure Edge Baseline Extension Interpretation

M30 summarizes an existing M29 report without running new training:

```bash
python -B scripts/analysis/report_pure_edge_baseline_extension.py \
  --json-in .agent/tmp/pure_edge_baseline_extension.json
```

It writes `.agent/tmp/pure_edge_baseline_extension_interpretation.json` and
`.md`. The report separates health claims from quality claims for
`all_mode_extension`, `topk_extension`, and `all_to_topk_extension`.

The all-mode extension can support only a warm-start numerical-health claim,
the top-k extension can support only a deployment-candidate numerical-health
claim, and all-to-topk movement remains support-change-confounded when active
edge count or gradient coverage changes. `ready_for_M31_pure_edge_benchmark_package`
is a packaging gate for the pure edge-score baseline benchmark, not evidence of
optimal topology learning, top-k superiority, structural-head benefit,
production readiness, or large-scale validity.

## M31 Pure Edge Benchmark Package

M31 packages the stable pure edge-score baseline as a benchmark reference:

```bash
python -B scripts/analysis/build_pure_edge_benchmark_package.py
```

The default inputs are `.agent/tmp/pure_edge_reference_package.json`,
`.agent/tmp/pure_edge_baseline_extension.json`, and
`.agent/tmp/pure_edge_baseline_extension_interpretation.json`. The manifest is
`configs/pure_edge_benchmark.yaml`.

The package records artifact status, benchmark policy, evidence summaries,
allowed health claims, disallowed claims, and a future comparison contract.
Future models must report C/F/D/E, active edge counts, gradient coverage,
support mode, structural-bias policy, direct physics loss absence, and
support-change confounders. Comparisons are blocked by missing baseline
artifacts, unlabeled structural bias, direct link/SINR/BLER/HARQ/coverage
losses, train/eval/deploy graph-rule mismatch, or dense NxN graph construction.

`ready_for_M32_formal_training_config=true` means the benchmark reference is
complete enough to freeze a formal training-v0 config. It is not a production
training, model-quality, top-k superiority, structural-head benefit, or
large-scale deployment claim.

## M32 Formal Training v0 Config Freeze

M32 validates the frozen `configs/training_v0.yaml` contract:

```bash
python -B scripts/analysis/validate_training_v0_config.py
```

The validator reads `.agent/tmp/pure_edge_benchmark_package.json` by default
and requires `ready_for_M32_formal_training_config=true` unless explicitly
forced. It confirms the all-mode warm-start phase, top-k candidate phase,
zero structural weights, disabled top-k structural bias, non-trainable
structural heads, diagnostic/disabled budget head, required C/F/D/E reporting,
and conservative claim boundaries.

M32 is not a training run. `ready_for_M33_formal_training_v0_smoke=true` means
the configuration is internally consistent with the benchmark package and can
be used for the next smoke-only formal harness. Direct link/SINR/BLER/HARQ/
coverage losses, dense NxN graph construction, sampling, PBFT, production
training, checkpointing, dataset loading, scheduling, and model selection
remain outside the contract.

## M33 Formal Training v0 Smoke

M33 runs the frozen v0 config through a controlled two-phase smoke:

```bash
python -B scripts/analysis/run_formal_training_v0_smoke.py \
  --max-steps-override 2
```

The default readiness source is
`.agent/tmp/training_v0_config_validation.json`; it must authorize M33 unless
`--force` is used and recorded. The report includes per-seed phase 0/phase 1
loss, C/F/D/E initial/final/delta values, L_R/L_D/L_E finals, active edge
counts, gradient coverage, finite-loss and finite-gradient fractions, parameter
movement, runtime, and Python/Torch/NumPy/SciPy/CUDA environment fields.

`formal_training_v0_smoke_status=smoke_passed` is a numerical contract signal
only. It can recommend `M34_formal_training_v0_baseline_run`, but it does not
claim model quality, top-k superiority, structural-head benefit, production
training readiness, or large-scale validity.

## M34 Formal Training v0 Baseline Run

M34 executes the M33-authorized frozen v0 policy with the configured phase
steps, or a small explicit override for checks:

```bash
python -B scripts/analysis/run_formal_training_v0_baseline.py \
  --max-steps-override 3
```

The default readiness source is `.agent/tmp/formal_training_v0_smoke.json`.
The output records per-step scalar loss, L_R/L_D/L_E, C/F/D/E, active edge
count, gradient coverage, and parameter-gradient norm trends for each seed and
phase. `formal_training_v0_baseline_status=baseline_run_completed` means the
longer controlled baseline run stayed numerically healthy under the frozen
pure edge-score policy. It is still not a production-training, model-quality,
top-k-superiority, structural-head-benefit, or large-scale validity claim.

## M35 Formal Training v0 Baseline Interpretation

M35 summarizes an existing M34 report without rerunning training:

```bash
python -B scripts/analysis/report_formal_training_v0_baseline.py \
  --json-in .agent/tmp/formal_training_v0_baseline.json
```

The report compares used M34 steps to the configured 50+50 phase counts in
`configs/training_v0.yaml`. A controlled 3+3 or other short override is listed
as `controlled_step_override_used`, but it does not block M36 when finite
losses, finite gradients, parameter movement, disabled structural bias, and
non-worse C/F/D/E gates all pass. `ready_for_M36_full_formal_training_v0_baseline`
authorizes only the full baseline run; it does not claim the full run has
already been validated or prove model quality.

## M36 Full Formal Training v0 Baseline Run

M36 runs the configured formal v0 baseline after M35 authorization:

```bash
python -B scripts/analysis/run_formal_training_v0_full.py
```

Fast checks may use:

```bash
python -B scripts/analysis/run_formal_training_v0_full.py \
  --max-steps-override 3
```

The default readiness source is
`.agent/tmp/formal_training_v0_baseline_interpretation.json`. Full mode uses
the configured 50+50 phase counts from `configs/training_v0.yaml`; override
mode is labeled `baseline_run_completed_with_override` and cannot be treated
as a completed full run. The report records per-seed per-step scalar trends,
phase switch deltas, finite-loss and finite-gradient fractions, parameter
movement, runtime, and Python/Torch/NumPy/SciPy/CUDA environment details.

`formal_training_v0_full_status=full_baseline_run_completed` is a numerical
contract result only. It can recommend M37 interpretation, but it does not
prove model quality, top-k superiority, production training readiness, or
large-scale validity.

## M37 Formal Training v0 Full Interpretation

M37 summarizes an existing M36 report without rerunning training:

```bash
python -B scripts/analysis/report_formal_training_v0_full.py \
  --json-in .agent/tmp/formal_training_v0_full.json
```

The report records the M36 source status, whether a step override was used,
the 50/50 step-count check, seed success counts, finite-loss and
finite-gradient fractions, runtime environment details, and per-phase loss and
C/F/D/E trend consistency. Phase 0 and phase 1 are interpreted separately.
Phase 1 improvement is a top-k phase training trend, not a direct quality
comparison with all-mode. Phase-transition effects remain support-rule
confounded.

`ready_for_M38_small_realistic_formal_baseline=true` means the toy-profile
formal v0 baseline is numerically healthy enough for the small-realistic
follow-up. It does not prove small-realistic behavior, production readiness,
top-k superiority, or model quality.

## M38 Small-Realistic Formal Training v0 Baseline Run

M38 runs the M37-authorized formal v0 baseline under the `small_realistic`
Avalanche profile:

```bash
python -B scripts/analysis/run_formal_training_v0_small_realistic.py
```

Fast checks may use:

```bash
python -B scripts/analysis/run_formal_training_v0_small_realistic.py \
  --max-steps-override 3
```

The report writes `.agent/tmp/formal_training_v0_small_realistic.json` and
`.md`. It records the readiness artifact, seed success counts, 50/50 or
override step counts, C/F/D/E and loss deltas, phase-switch deltas, finite loss
and gradient fractions, parameter movement, active edge and gradient coverage
trends, runtime, and Python/Torch/NumPy/SciPy/CUDA environment details.

`formal_training_v0_small_realistic_status=small_realistic_baseline_completed`
is a numerical contract result only. If a step override is used, the report is
not a full configured small-realistic run. M38 can recommend only M39
interpretation; it does not prove model quality, top-k superiority, production
readiness, or large-scale validity.

## M39 Small-Realistic Formal Training v0 Interpretation

M39 summarizes an existing M38 report without rerunning training:

```bash
python -B scripts/analysis/report_formal_training_v0_small_realistic.py \
  --json-in .agent/tmp/formal_training_v0_small_realistic.json
```

The report records the M38 source status, profile, step override state, 50/50
step-count check, seed success counts, finite-loss and finite-gradient
fractions, runtime environment details, per-phase C/F/D/E consistency, and
large phase-0 improvement diagnostics. Phase 0 and phase 1 are interpreted
separately. Phase 1 improvement is a top-k phase training trend, not a direct
quality comparison with all-mode. Phase-transition effects remain
support-rule-confounded.

`ready_for_M40_formal_training_v0_baseline_package=true` means toy and
small-realistic formal v0 baseline evidence can be consolidated into a package.
It does not prove model quality, top-k superiority, realistic deployment
behavior, production readiness, or large-scale validity.

## M40 Formal Training v0 Baseline Package

M40 packages the toy and small-realistic formal v0 baseline evidence:

```bash
python -B scripts/analysis/build_formal_training_v0_baseline_package.py
```

The package requires the pure edge benchmark package, M36 full toy run, M37 toy
interpretation, M38 small-realistic run, M39 small-realistic interpretation,
and the frozen configs. Missing evidence yields
`incomplete_formal_v0_baseline_package`; blocked or contract-inconsistent
evidence yields `blocked`.

The package includes a cross-profile comparison table for phase 0 and phase 1.
Small-realistic phase 0 can be described as stronger trainability evidence, but
the report must not claim realistic deployment behavior or model-quality proof.
Phase 1 is interpreted as a top-k phase training trend, not as top-k
superiority over all-mode.

Future comparisons against the formal v0 baseline must report C/F/D/E,
L_R/L_D/L_E, active edge count, gradient coverage, support mode,
structural-bias policy, Avalanche profile, node count, seed count,
direct-physics-loss absence, graph-rule consistency, and support-change
confounders. Direct link/SINR/BLER/HARQ/coverage losses, dense NxN graph
construction, train/eval/deploy graph-rule mismatch, unlabeled structural bias,
or unlabelled nonzero top-k structural bias block comparison.

## M41 Controlled v0 Baseline Extension

M41 runs the controlled formal v0 extension:

```bash
python -B scripts/analysis/run_controlled_v0_baseline_extension.py \
  --node-count 500 --profiles toy,small_realistic --seeds 7,13,23
```

Fast checks may use `--max-steps-override` and a smaller node count through the
dedicated `controlled-v0-baseline-extension-check` target.

The report writes `.agent/tmp/controlled_v0_baseline_extension.json` and
`.md`. It records per-profile and per-policy C/F/D/E and loss aggregates,
L_R/L_D/L_E final means when available, active edge counts, gradient coverage,
runtime, structural-bias guards, direct-physics-loss absence, dense-NxN absence,
and train/eval/deploy graph-rule consistency.

`controlled_extension_completed` allows only M42 interpretation. It does not
prove model quality, top-k superiority, production training readiness,
realistic deployment behavior, or deployment readiness.

## M42 Controlled v0 Baseline Extension Interpretation

M42 interprets the M41 extension:

```bash
python -B scripts/analysis/report_controlled_v0_baseline_extension.py \
  --json-in .agent/tmp/controlled_v0_baseline_extension.json
```

The report writes
`.agent/tmp/controlled_v0_baseline_extension_interpretation.json` and `.md`.
It summarizes the source contract, N=500 profiles, seed/case success counts,
structural-bias guards, direct-physics-loss absence, dense-NxN absence, graph
rule consistency, per-case C/F/D/E and loss deltas, gradient coverage, active
edge count, and support-change confounders.

`controlled_extension_confirmed` means the M41 source is healthy enough for the
M43 v0 baseline package v1.1 gate. It is still not a quality ranking, not a
top-k superiority claim, not 10k-scale validation, and not deployment or
production-training validation.

## M43 Formal Training v0 Baseline Package v1.1

M43 builds the consolidated v1.1 package:

```bash
python -B scripts/analysis/build_formal_training_v0_baseline_package_v1_1.py
```

The package summarizes M40 N=100 toy and small-realistic evidence plus the
M41/M42 N=500 controlled extension across all, top-k, and all-to-topk policies.
Its coverage matrix is a claim ledger, not a model ranking table: health claims
may be allowed, quality claims remain blocked, and all-to-topk rows remain
support-change-confounded.

`complete_v1_1_baseline_package` with
`ready_for_M44_scale_or_profile_extension=true` authorizes only M44
scale/profile extension planning. It does not validate 10k behavior,
production training, deployment readiness, or top-k superiority.

## M44 Scale/Profile Extension Plan

M44 plans the next diagnostic without running training:

```bash
python -B scripts/analysis/report_scale_profile_extension_plan.py \
  --json-in .agent/tmp/formal_training_v0_baseline_package_v1_1.json
```

The plan writes `.agent/tmp/scale_profile_extension_plan.json` and `.md`. It
requires the M43 package to be complete and ready for M44, then ranks
`small_realistic_topk_flatness_diagnostic`, `runtime_memory_scaling_check`,
`node_count_2000_toy_topk_smoke`, and
`node_count_500_small_realistic_more_steps`.

The default next stage is
`M45_small_realistic_topk_flatness_diagnostic`, because the current N=500
small-realistic top-k result is healthy but near-flat. M44 keeps 2k toy top-k
scale deferred until that ambiguity is explained, and it preserves the same
claim boundaries: no 10k validation, no top-k superiority, no structural-head
reintroduction, no production training, and no deployment validation.

## M45 Small-Realistic Top-k Flatness Diagnostic

M45 executes the selected diagnostic:

```bash
python -B scripts/analysis/run_topk_flatness_diagnostic.py \
  --max-steps-override 5
```

The report writes `.agent/tmp/topk_flatness_diagnostic.json` and `.md`. It
requires the M44 plan to recommend
`M45_small_realistic_topk_flatness_diagnostic` unless forced, then checks the
N=500 `small_realistic` top-k pure edge-score case across the configured seeds.

The evaluation output separates reliability saturation, loss-component
movement, support rigidity, gradient signal, and metric flatness. A flatness
classification is a diagnostic, not a quality claim. Larger-node tests,
longer-step probes, support-rigidity fixes, or loss-sensitivity diagnostics
remain follow-up stages and must preserve the same claim boundaries.

## M46 Runtime/Memory Scaling Diagnostic

M46 runs the next engineering diagnostic:

```bash
python -B scripts/analysis/run_runtime_memory_scaling.py
```

The default smoke writes `.agent/tmp/runtime_memory_scaling.json` and `.md`
when not run with `--no-write`. It requires the M45 flatness diagnostic to be
`reliability_saturated` or `metric_flat_but_healthy`, unless forced.

Each case reports wall time, environment/model/scorer/constructor/evaluator/
loss/backward timing, deterministic tensor-memory proxies, candidate and active
edge counts, gradient coverage, selected fraction, and sparse-to-dense edge
ratio against a dense `N*N` proxy. The proxy is a comparison scalar only, not a
dense graph allocation. The report can identify a runtime bottleneck or clear
the fast smoke path for a controlled N=2000 toy top-k follow-up, but it does not
make model-quality or deployment claims.

## M47 Node Count 2000 Toy Top-k Smoke

M47 runs the controlled follow-up scale point:

```bash
python -B scripts/analysis/run_node_count_2000_topk_smoke.py
```

The report writes `.agent/tmp/node_count_2000_topk_smoke.json` and `.md`. It
requires `.agent/tmp/runtime_memory_scaling.json` to report
`scaling_status=scaling_smoke_passed` unless forced, then checks N=2000
toy/top-k under the pure edge-score v0 policy.

The report includes active and candidate edge counts, sparse-to-dense ratio,
gradient coverage, finite-loss and finite-gradient fractions, parameter-change
fraction, C/F/D/E deltas when available, per-stage runtime, bottleneck stage,
throughput proxies, tensor-memory proxies, runtime environment, and claim
boundaries. Passing M47 can authorize only M48 interpretation or a controlled
next scale smoke; it is not model-quality, production-training, deployment, or
10k validation evidence.

## M48 Runtime/Memory Scaling Interpretation

M48 consolidates the M46 and M47 scaling artifacts without running new cases:

```bash
python -B scripts/analysis/report_runtime_memory_scaling.py
```

The report writes `.agent/tmp/runtime_memory_scaling_interpretation.json` and
`.md`. It checks source contracts, sparse-to-dense ratios, finite
loss/gradient evidence, structural-bias-disabled policy, top-k structural-bias
disablement, and dense-path absence before interpreting the N=100, N=500, and
N=2000 toy/top-k scaling trend.

The output includes a scaling table, active-edge, memory, and step-time growth
ratios, sparse-ratio trend, bottleneck-stage transition, and a conservative
M49 recommendation. Constructor/evaluator bottlenecks trigger diagnostic
recommendations before larger scale claims. M48 remains an engineering-health
interpretation, not a production-runtime, deployment, memory-peak, top-k
superiority, or 10k validation claim.

## M49 Constructor Bottleneck Diagnostic

M49 runs the constructor-only profiling diagnostic:

```bash
python -B scripts/analysis/profile_topology_constructor.py \
  --node-counts 500,2000
```

The report writes `.agent/tmp/constructor_bottleneck_diagnostic.json` and
`.md`. It profiles deterministic synthetic sparse inputs with candidate degree
8 and top-k cap 4 by default, records input size, active and feasible edges,
gradient coverage, cap-hit diagnostics, top-k boundary margins, zero-gradient
risk counts, and a timing decomposition across constructor stages.

M49 can recommend optimizing segmented top-k, diagnostics, validation, or row
loops before continuing scale. `--include-all-mode` adds all-mode comparison
timings, but that comparison remains an engineering diagnostic and does not
make top-k superiority, production-runtime, deployment, or 10k validation
claims.

## M50 Constructor Optimization Design

M50 summarizes the M49 constructor bottleneck evidence without running new
training or changing topology construction:

```bash
python -B scripts/analysis/report_constructor_optimization_design.py \
  --json-in .agent/tmp/constructor_bottleneck_diagnostic.json
```

The output records the M49 N=500 and N=2000 constructor timing summary, current
implementation risk summary, optimization options, and a correctness contract
for any future fast path. If top-k selection consumes more than half of the
N=2000 constructor time and M49 recommends `optimize_segmented_topk`, M50 can
authorize only `M50_1_segmented_topk_fast_path`. It does not implement the fast
path and does not claim 5k/10k behavior, production runtime, deployment
readiness, or top-k superiority.

## M50.1 Segmented Top-k Fast Path

M50.1 adds an opt-in constructor backend optimization and profiles it against
legacy behavior:

```bash
python -B scripts/analysis/profile_topology_constructor.py \
  --node-counts 500,2000 --topk-backend both
```

The profile reports `legacy_time_s`, `segmented_fast_time_s`, `speedup_ratio`,
selected-candidate equivalence, topology-weight equivalence, diagnostics
equivalence, and `ready_for_M51_5k_with_fast_path`. Readiness requires
equivalent sparse outputs, finite loss/gradient integration through the M3/M4
path, no dense graph allocation, and a fast path that is not meaningfully
slower than legacy. This is still an engineering constructor check, not a model
quality, production runtime, deployment, or 10k validation claim.

## M51 Node Count 5000 Toy Top-k Smoke

M51 runs the next controlled scale point with the segmented fast constructor
backend:

```bash
python -B scripts/analysis/run_node_count_5000_topk_smoke.py
```

The default input is `.agent/tmp/constructor_bottleneck_diagnostic.json`, which
must report `ready_for_M51_5k_with_fast_path=true` unless `--force` is used.
The default case is N=5000, toy Avalanche profile, top-k support, and
`topk_backend=segmented_fast`.

The report writes `.agent/tmp/node_count_5000_topk_smoke.json` and `.md`. It
records active and candidate edge counts, mean out-degree, sparse-to-dense
ratio against a scalar `N*N` proxy, finite-loss and finite-gradient fractions,
parameter movement, C/F/D/E deltas when available, per-stage runtime,
bottleneck stage, deterministic tensor-memory proxies, and runtime environment
fields.

`node_5000_topk_smoke_passed` is a controlled numerical and scaling-health
claim only. It does not validate N=10000 behavior, production runtime,
deployment readiness, realistic-profile behavior, or top-k superiority.

## M52 Runtime/Memory Scaling v1.1 Interpretation

M52 consolidates the observed sparse scaling path after the segmented_fast
constructor backend:

```bash
python -B scripts/analysis/report_runtime_memory_scaling_v1_1.py
```

The report reads M46, M48, M49, M50, and M51 artifacts and writes
`.agent/tmp/runtime_memory_scaling_v1_1_interpretation.json` plus `.md`. Its
scaling table covers the available toy/top-k rows for N=100, N=500, N=2000,
and N=5000, including backend context, active and candidate edges,
sparse-to-dense ratio, step time, constructor time, evaluator time, backward
time, memory proxy, bottleneck stage, and gradient coverage.

M52 can allow only controlled next-scale diagnostics. If the evaluator is a
moderate bottleneck, the recommended next step can be a N=10000 toy/top-k
forward smoke. If the evaluator bottleneck is severe, superlinear, or has a
high 10k projection, the report recommends an evaluator diagnostic first.
No M52 outcome validates N=10000 behavior, production runtime, deployment
readiness, RSS/GPU peak memory, or top-k superiority.

## M53 Node Count 10000 Toy Top-k Forward Smoke

M53 runs the controlled N=10000 toy/top-k forward-only smoke with the
segmented fast constructor backend:

```bash
python -B scripts/analysis/run_node_count_10000_topk_forward_smoke.py
```

The default input is
`.agent/tmp/runtime_memory_scaling_v1_1_interpretation.json`, which must
recommend `M53_node_count_10000_toy_topk_forward_smoke` and set
`ready_for_next_scale_step=true` unless `--force` is used. The script writes
`.agent/tmp/node_count_10000_topk_forward_smoke.json` and `.md`.

The report contains no backward or optimizer fields. It records sparse
candidate and active edge counts, mean out-degree, a scalar dense `N*N`
comparison proxy, forward runtime by scorer/constructor/evaluator/loss stage,
evaluator bottleneck fraction, deterministic tensor-memory proxy, finite
C/F/D/E metrics, and L_R/L_D/L_E if available.

`node_10000_topk_forward_passed` is a forward-only numerical and scaling-health
claim. `evaluator_bottleneck_confirmed` routes to an evaluator diagnostic, and
`memory_bottleneck_detected` routes to runtime/memory interpretation. No M53
outcome validates 10k training, production runtime, deployment readiness,
RSS/GPU peak memory, realistic-profile behavior, or top-k superiority.

## M54 Evaluator Bottleneck Diagnostic

M54 profiles evaluator runtime and memory before any 10k backward attempt:

```bash
python -B scripts/analysis/profile_v2x_evaluator.py --node-counts 5000,10000
```

The default readiness input is
`.agent/tmp/node_count_10000_topk_forward_smoke.json`, which must report
`evaluator_bottleneck_confirmed` or `node_10000_topk_forward_passed` unless
`--force` is used. The output is
`.agent/tmp/evaluator_bottleneck_diagnostic.json` plus `.md`.

The report decomposes evaluator time into channel/link proxy, topology
query-support, graph-coupled Avalanche recurrence, energy metric, metric
extraction, diagnostic postprocess, tensor conversion, and residual time. It
also records recurrence timing per round, query-support substage timing,
deterministic evaluator tensor-memory proxies, reliability gap against the
configured target, and N=5000 versus N=10000 scaling ratios.

`ready_for_10k_backward=true` is allowed only when evaluator cost is moderate,
outputs are finite, memory proxy is controlled, no dense graph path appears,
and no component dominates. A dominant graph-coupled or query-support component
routes to an optimization design first. M54 remains a diagnostic; it does not
validate 10k training, production runtime, deployment behavior, or
RSS/GPU-peak memory.

## M55 Query Support Optimization Design

M55 turns the M54 query-support bottleneck into a report-only optimization
design:

```bash
python -B scripts/analysis/report_query_support_optimization_design.py \
  --json-in .agent/tmp/evaluator_bottleneck_diagnostic.json
```

The report writes `.agent/tmp/query_support_optimization_design.json` and
`.md`. It requires M54 to identify `topology_query_support_bottleneck` and to
recommend `M55_query_support_optimization_design` unless forced.

M55 records the source N=5000/N=10000 evaluator, query-support, and
graph-coupled Avalanche times; evaluator and active-edge growth; reliability
state; and `ready_for_10k_backward`. It also documents the current sparse
query-support semantics:

```text
p_correct_query_i = sum_j q_ij link_success_ij u_j
p_wrong_query_i   = sum_j q_ij link_success_ij v_j
p_link_response_i = sum_j q_ij link_success_ij
p_no_support_i    = 1 - p_correct_query_i - p_wrong_query_i
```

The design can recommend a future fused sparse query-support path, diagnostics
fast path, CSR row-pointer path, row-normalization cache, or static edge/link
term cache. It does not implement those changes, run backward, or validate
10k training. Any future fast path must preserve explicit wrong-support
probabilities, neutral/no-support semantics, isolated-row behavior, gradients,
diagnostics when enabled, and no dense NxN or stochastic sampling paths.

## M55.1 Query Support Fast Path

M55.1 implements the query-support design as an explicit evaluator backend:
`query_support_backend="legacy"` preserves the previous path, while
`query_support_backend="fused_fast"` keeps the same sparse probability
semantics with cheaper segmented diagnostics. The evaluator bridge passes the
backend through `AvalancheBridgeConfig`; there is no separate train,
validation, or deployment evaluator rule.

M55.1 also adds `diagnostics_mode="full" | "lite" | "off"`. Full mode keeps
the current diagnostics. Lite and off reduce query-support diagnostic cost but
must not alter `p_correct_query`, `p_wrong_query`, `p_link_response`,
`p_neutral_query`, `p_no_support_query`, or the response aliases. Lite/off
outputs are clearly labeled and should not be treated as final diagnostic
evidence unless full diagnostics are rerun or the reduced mode is explicitly
accepted.

`scripts/analysis/profile_v2x_evaluator.py` can now run
`--query-support-backend both` to compare legacy/full and fused_fast/full on
the same sparse graph, then time fused_fast under full, lite, and off
diagnostics. This profile reports equivalence checks, speedup, and
`ready_for_M56_10k_backward_with_fast_query_support`. Passing M55.1 is an
implementation-health result for the evaluator backend only; it does not make
model-quality, 10k training, production-runtime, or deployment claims.

## M56 10k One-Step Backward Smoke

`scripts/analysis/run_node_count_10000_topk_backward_smoke.py` runs a single
controlled N=10000 toy/top-k backward smoke using
`topk_backend=segmented_fast`, `query_support_backend=fused_fast`, and
`diagnostics_mode=lite` by default. It requires M55.1 fast-query-support
readiness unless `--force` is passed.

The report writes `.agent/tmp/node_count_10000_topk_backward_smoke.json` and
`.md`. It records candidate and active sparse edge counts, loss and C/F/D/E
metrics, finite loss and finite gradient checks, parameter movement, forward
and backward timing, query-support timing, graph-coupled Avalanche timing, and
deterministic tensor-memory proxy.

M56 is backward feasibility evidence only. It does not run 10k training,
validate production runtime, validate deployment, or make any model-quality
claim. Lite diagnostics are explicitly labeled and do not replace full
diagnostic evidence for final claims.

## M57 10k Backward Interpretation

`scripts/analysis/report_node_count_10000_topk_backward.py` reads
`.agent/tmp/node_count_10000_topk_backward_smoke.json` and writes
`.agent/tmp/node_count_10000_topk_backward_interpretation.json` plus `.md`.
It does not rerun the smoke. It classifies the source as
`backward_feasibility_confirmed`, `backward_feasible_but_tiny_update`,
`numerically_healthy_only`, or `blocked`.

The interpretation records the source backends, sparse edge ratio, finite
loss/gradient checks, parameter-update category, reliability gap, runtime
bottleneck, and tensor-memory proxy. A tiny update recommends
`M58_gradient_step_size_diagnostic` by default, while diagnostics-lite evidence
adds an explicit caution.

## M58 Gradient Step-Size Diagnostic

`scripts/analysis/run_gradient_step_size_diagnostic.py` runs controlled
one-step probes over configured learning rates, defaulting to
`1e-4,1e-3,1e-2`. It requires the M57 interpretation to recommend
`M58_gradient_step_size_diagnostic` unless `--force` is passed.

For each learning rate, the diagnostic initializes the same model seed, runs
one forward pass, one backward pass, and one optimizer step, then records
finite-loss/gradient status, total/max/mean gradient norms, selected edge-score
and topology-weight gradient norms when available, parameter-update max/mean/L2,
update-to-parameter norm ratio, L_R/L_D/L_E contributions, C/F/D/E metrics, and
runtime. The aggregate classifies gradient scale and update scale before
choosing between a few-step 10k smoke, a broader learning-rate sweep, a
loss-scale diagnostic, or a reliability-gradient diagnostic.

M58 remains a diagnostic-only harness. It does not validate 10k training,
production runtime, deployment readiness, or model-quality improvement.

## M59 Gradient Source Diagnostic

M59 diagnoses whether M58's tiny updates come from the loss gradient, topology
weight gradient, scorer parameter gradient, optimizer scale, or loss scale:

```bash
python -B scripts/analysis/run_gradient_source_diagnostic.py \
  --loss-scales 1,10,100,1000 --learning-rates 1e-2,1e-1,1.0
```

The report writes `.agent/tmp/gradient_source_diagnostic.json` and `.md`. It
records the baseline total loss, L_R/L_D/L_E, C/F/D/E, reliability gap,
separate gradient norms for total loss and each loss component, gradient
localization through topology weights, edge scores, scorer parameters, and
score-head parameters, plus one-step loss-scale and learning-rate probes.

M59 is allowed to report that a calibrated one-step loss scale or learning rate
produces finite, meaningful updates. It is not allowed to claim 10k training
readiness, model quality, production runtime, deployment validity, or
diagnostics-lite equivalence to full evidence.

## M60 Loss-Scale Calibrated Backward Smoke

M60 runs the M59-selected calibrated one-step backward smoke:

```bash
python -B scripts/analysis/run_loss_scale_calibrated_backward_smoke.py
```

The report writes `.agent/tmp/loss_scale_calibrated_backward_smoke.json` and
`.md`. It requires the M59 source to report
`gradient_source_status=loss_scale_too_small` unless forced, then uses the
recommended scalar loss scale, expected `1000.0`, with `learning_rate=1e-2`.

The raw C/D/E coupled loss remains the reported objective. The scaled loss is a
single-step backward calibration scalar and is labeled separately. Passing M60
can authorize only a claim-limited M61 few-step smoke; it does not validate
10k training, production runtime, deployment readiness, model quality, or
diagnostics-lite as final evidence.

## M61 10k Few-Step Training Smoke

M61 runs the calibrated few-step smoke:

```bash
python -B scripts/analysis/run_tenk_few_step_training_smoke.py --max-steps 3
```

The report writes `.agent/tmp/tenk_few_step_training_smoke.json` and `.md`.
It requires M60 to set `ready_for_M61_10k_few_step_training_smoke=true` unless
forced. Defaults remain `node_count=10000`, `profile=toy`,
`support_mode=topk`, `topk_backend=segmented_fast`,
`query_support_backend=fused_fast`, `diagnostics_mode=lite`,
`loss_scale=1000.0`, and `learning_rate=0.01`.

For every step, M61 reports raw and scaled loss, L_R/L_D/L_E, C/F/D/E, finite
loss/gradient checks, gradient norms, per-step parameter movement, cumulative
parameter movement, active-edge count, gradient coverage, runtime, and tensor
memory proxy. The status can pass only as a short numerical smoke. It does not
validate 10k training, production runtime, deployment readiness, model quality,
top-k superiority, or diagnostics-lite as final evidence.

## M62 Metric Flatness Diagnostic

M62 runs:

```bash
python -B scripts/analysis/run_metric_flatness_diagnostic.py --max-steps 3
```

The report writes `.agent/tmp/metric_flatness_diagnostic.json` and `.md`. It
requires M61 to report
`tenk_few_step_training_status=numerically_healthy_but_no_metric_movement`
unless forced.

M62 evaluates the same toy/top-k sparse path and reports parameter, edge-score,
support, topology-weight, query-support, and C/F/D/E movement. It also runs
deterministic edge-score perturbation and edge-score-only probes to determine
whether the evaluator can move when scores are changed directly. These probes
are diagnostics only and do not validate 10k training, model quality,
production runtime, or diagnostics-lite as final evidence.

## M63 Top-k Support Sensitivity Probe

M63 runs:

```bash
python -B scripts/analysis/run_topk_support_sensitivity_probe.py
```

The report writes `.agent/tmp/topk_support_sensitivity_probe.json` and `.md`.
It requires M62 to classify the flatness as `topk_boundary_too_large` or
`support_rigid` unless forced.

The output records baseline top-k boundary margins, near-miss candidates,
M62's observed edge-score delta per step, deterministic perturbation sweeps,
score-scale probes, and analysis-only row-softmax temperature probes. It can
classify hard top-k as too rigid, trainable at the current scale, sensitive to
score scaling, or sensitive through selected-edge weight concentration.

All M63 probes are sensitivity diagnostics. They do not validate 10k training,
do not make top-k superiority or model-quality claims, and do not convert
temperature or score scaling into a deployable constructor rule.

## M64 Score-Scale Calibration

M64 runs:

```bash
python -B scripts/analysis/run_score_scale_calibration.py --score-scales 1,3,10,30,100
```

The report writes `.agent/tmp/score_scale_calibration.json` and `.md`. It
requires M63 to report `support_sensitivity_status=score_scale_sensitive`
unless forced.

For each score scale, M64 reports forward-only selected-support overlap,
row-top-k overlap, topology-weight entropy and deltas, query-support deltas,
C/F/D/E deltas, and one-step calibrated update health. The decision gate picks
the smallest safe scale that moves selected weights, query support, or metrics
without major support jumps or unstable updates. It remains a controlled
diagnostic and does not validate 10k training or production deployment.

## M65 Score-Scaled Few-Step Smoke

M65 runs:

```bash
python -B scripts/analysis/run_score_scaled_tenk_few_step_smoke.py --max-steps 3
```

The report writes `.agent/tmp/score_scaled_tenk_few_step_smoke.json` and
`.md`. It requires M64 to set
`ready_for_M65_score_scaled_10k_few_step_smoke=true` unless forced.

For each step, M65 reports raw/scaled loss, L_R/L_D/L_E, C/F/D/E, finite
loss/gradient checks, parameter update magnitude, edge-score deltas,
topology-weight deltas, query-support deltas, support overlap, runtime, and
tensor memory proxy. The report compares against `.agent/tmp/tenk_few_step_training_smoke.json`
when available. It remains a few-step diagnostic and cannot validate 10k
training, production deployment, model quality, or top-k superiority.

### M66 Score-Scaled Interpretation

M66 reports over the M65 score-scaled few-step JSON plus M61, M62, M63, and
M64 context artifacts:

```bash
python -B scripts/analysis/report_score_scaled_tenk_few_step.py --json-in .agent/tmp/score_scaled_tenk_few_step_smoke.json
```

The report writes `.agent/tmp/score_scaled_tenk_few_step_interpretation.json`
and `.md`. It is report-only and decides the next diagnostic strategy. It does
not run training, tune LR/loss scale, modify constructor semantics, or promote
analysis-only temperature probes to deployable behavior.

Current expected evidence can support `score_scale=3` being numerically safe
but insufficient for observable C/F/D/E movement. If higher M64 score scales
remain safe candidates or M63 temperature probes moved metrics, M66 recommends
a diagnostic score-scale/temperature grid before longer 10k training.

### M67 Score-Temperature Grid Probe

M67 runs:

```bash
python -B scripts/analysis/run_score_temperature_grid_probe.py --probe-steps 1
```

The report writes `.agent/tmp/score_temperature_grid_probe.json` and `.md`.
It requires M66 to report
`score_scaled_tenk_interpretation_status=score_scale_3_safe_but_insufficient`
unless forced.

For each score-scale and row-softmax-temperature pair, the report records
forward C/F/D/E sensitivity, topology-weight deltas, query-support deltas,
finite loss/gradient checks, update health, and a post-update reforward.
Temperature values other than `1.0` are labeled analysis-only and cannot be
promoted as deployable evidence. A deployable candidate requires temperature
`1.0`, stable support, finite gradients, and post-update metric movement.

### M68 Row-Softmax Temperature Design

M68 runs:

```bash
python -B scripts/analysis/report_row_softmax_temperature_design.py --json-in .agent/tmp/score_temperature_grid_probe.json
```

The report writes `.agent/tmp/row_softmax_temperature_design.json` and `.md`.
It requires M67 to report
`score_temperature_grid_status=only_analysis_candidates_found` and
`recommended_next_stage=M68_row_softmax_temperature_design` unless forced.

M68 is report-only. It summarizes the M67 diagnostic-only temperature evidence,
defines the candidate `row_softmax_temperature: float = 1.0` constructor
contract, records risks, and decides whether a prototype should be built. It
does not change evaluator metrics, does not rerun 10k training, and keeps
`ready_for_M69_temperature_aware_training_smoke=false` until implementation
and equivalence tests exist.

### M68.1 Row-Softmax Temperature Constructor Prototype

M68.1 adds `row_softmax_temperature` as a formal sparse constructor parameter.
The evaluator still consumes the same sparse `(src_index, dst_index,
topology_weight)` interface and the same Avalanche/Snowball closed-form path.
Temperature changes selected-edge weights before evaluation; it does not add
new evaluator objectives, direct link/SINR/BLER/HARQ/coverage losses, sampling,
or dense NxN graph tensors.

Temperature-aware evaluation evidence must report the constructor temperature.
`row_softmax_temperature=1.0` is legacy behavior. Non-1.0 values are only valid
as formal constructor settings when the same rule is used for train,
validation, and deployment.

### M69 Formal Constructor Temperature Grid

M69 runs:

```bash
python -B scripts/analysis/run_formal_temperature_grid_probe.py --probe-steps 1
```

It writes `.agent/tmp/formal_temperature_grid_probe.json` and `.md`. The
default readiness source is
`.agent/tmp/row_softmax_temperature_constructor_prototype.json`, which must
report `ready_for_M69_temperature_aware_10k_smoke=true` unless forced.

Unlike the earlier analysis-only grid, M69 applies
`row_softmax_temperature` through `TopologyConstructionLayer` for every
temperature row. Each row is labeled `analysis_only_temperature=false` and
`deployable_constructor_rule=true`; this means the grid can recommend a
temperature-aware follow-up smoke, not production training. The report compares
each temperature against the same score scale at temperature `1.0`, records
post-update reforward deltas, and blocks on nonfinite outputs, wrong backends,
structural bias, dense NxN paths, or direct physics losses.

### M70 Temperature-Aware 10k Few-Step Smoke

M70 runs:

```bash
python -B scripts/analysis/run_temperature_aware_tenk_few_step_smoke.py --max-steps 3
```

It writes `.agent/tmp/temperature_aware_tenk_few_step_smoke.json` and `.md`.
The default readiness source is `.agent/tmp/formal_temperature_grid_probe.json`,
which must report `ready_for_M70_temperature_aware_10k_few_step_smoke=true`
unless forced.

The smoke uses the M69 deployable candidate by default:
`score_scale=30.0`, `row_softmax_temperature=0.5`, segmented-fast top-k, fused
query support, and diagnostics-lite. The temperature is applied by the
constructor, not by analysis-only postprocessing. Each step records raw loss,
C/F/D/E, gradients, support overlap, topology-weight movement, query-support
movement, and a post-update reforward. The report compares movement with M61
unscaled and M65 score_scale=3 artifacts when they are present.

### M71 Temperature-Aware Interpretation

M71 runs:

```bash
python -B scripts/analysis/report_temperature_aware_tenk_few_step.py --json-in .agent/tmp/temperature_aware_tenk_few_step_smoke.json
```

It writes `.agent/tmp/temperature_aware_tenk_few_step_interpretation.json` and
`.md`. The report is artifact-only and does not rerun the smoke. It classifies
the M70 result as `temperature_aware_signal_confirmed`,
`numerically_healthy_but_tiny_signal`, `support_jump_risk`, `blocked`, or
`inconclusive`.

Readiness for M72 requires finite losses and gradients, formal deployable
constructor temperature, no analysis-only temperature, stable or minor support
movement, stable updates, improvement over the M65 score_scale=3 baseline, and
larger topology/query movement. Diagnostics-lite and below-target reliability
remain explicit cautions.

### M72 Temperature-Aware Longer 10k Smoke

M72 runs:

```bash
python -B scripts/analysis/run_temperature_aware_longer_tenk_smoke.py --max-steps 10
```

It writes `.agent/tmp/temperature_aware_longer_tenk_smoke.json` and `.md`. The
default readiness source is
`.agent/tmp/temperature_aware_tenk_few_step_interpretation.json`, which must
set `ready_for_M72_temperature_aware_longer_smoke=true` unless forced.

The smoke keeps the M70 formal constructor configuration:
`score_scale=30.0`, `row_softmax_temperature=0.5`, segmented-fast top-k, fused
query support, diagnostics-lite, toy profile, and structural bias disabled.
Each step records raw and scaled loss, L_R/L_D/L_E, C/F/D/E, reliability gap,
gradient norms, parameter movement, support overlap, topology/query movement,
runtime, and a post-update reforward. The report compares the longer movement
against M70, M65, and M61 artifacts when available.

Passing M72 can only authorize an M73 interpretation report. It does not
validate production training, deployment readiness, realistic-profile behavior,
or the finality of score_scale/temperature choices.

### M73 Temperature-Aware Longer Interpretation

M73 runs:

```bash
python -B scripts/analysis/report_temperature_aware_longer_tenk.py --json-in .agent/tmp/temperature_aware_longer_tenk_smoke.json
```

It writes `.agent/tmp/temperature_aware_longer_tenk_interpretation.json` and
`.md`. The report is artifact-only and does not rerun the smoke. It classifies
the M72 result as `temperature_signal_accumulates`,
`temperature_signal_accumulates_but_tiny`,
`numerically_healthy_but_insufficient`, `support_jump_risk`, `blocked`, or
`inconclusive`.

The report adds per-step movement estimates and a cautious linear reliability
gap projection when the gap is decreasing. A positive M73 result can authorize
only another controlled diagnostic, such as a 30-step temperature-aware smoke
or temperature/score-scale refinement. It cannot validate 10k training,
reliability target satisfaction, production readiness, or realistic-profile
behavior.

### M74 Temperature-Aware 30-Step 10k Smoke

M74 runs:

```bash
python -B scripts/analysis/run_temperature_aware_30step_tenk_smoke.py --max-steps 30
```

It writes `.agent/tmp/temperature_aware_30step_tenk_smoke.json` and `.md`. The
default readiness source is
`.agent/tmp/temperature_aware_longer_tenk_interpretation.json`, which must set
`ready_for_M74_temperature_aware_extended_smoke=true` unless forced.

The smoke keeps the formal constructor configuration from M72:
`score_scale=30.0`, `row_softmax_temperature=0.5`, segmented-fast top-k, fused
query support, diagnostics-lite, toy profile, and structural bias disabled.
Each step records raw and scaled loss, L_R/L_D/L_E, C/F/D/E, reliability gap,
gradient norms, parameter movement, support overlap, topology/query movement,
runtime, and a post-update reforward.

M74 adds trend and projection fields: C/F/reliability-gap deltas per step,
linear slopes, projected steps to close the reliability gap, projection status,
and movement ratios versus the M72 10-step artifact. Passing M74 can only
authorize M75 interpretation. It does not validate production training,
deployment readiness, realistic-profile behavior, or final score/temperature
settings.

### M75 Temperature-Aware 30-Step Interpretation

M75 runs:

```bash
python -B scripts/analysis/report_temperature_aware_30step_tenk.py --json-in .agent/tmp/temperature_aware_30step_tenk_smoke.json
```

It writes `.agent/tmp/temperature_aware_30step_interpretation.json` and `.md`.
The report reads the M74 30-step artifact and optional M72/M73/M70/M65
context artifacts. It classifies the result as an accumulating-but-slow signal,
a strong enough signal for an explicitly diagnostic extended smoke,
insufficient, support-jump risk, or blocked.

The M75 gate is conservative: if movement accumulates while projected
reliability closure remains impractically slow, the recommended next strategy
is all-mode warm-start or support smoothing rather than more blind hard-top-k
steps. It makes no model-quality, deployment, production-training, or
realistic-profile claim.

### M76 All-Mode Warm-Start 10k Probe

M76 runs:

```bash
python -B scripts/analysis/run_all_mode_warm_start_10k_probe.py --max-steps 30
```

It writes `.agent/tmp/all_mode_warm_start_10k_probe.json` and `.md`. The
default readiness source is
`.agent/tmp/temperature_aware_30step_interpretation.json`, which must
recommend `M76_all_mode_warm_start_10k_probe` unless forced.

The probe uses `support_mode=all`, fused query support, diagnostics-lite,
toy profile, structural bias disabled, `score_scale=1.0`, and
`row_softmax_temperature=1.0` by default. It records stepwise raw/scaled loss,
L_R/L_D/L_E, C/F/D/E, reliability gap, gradient coverage, topology/query
movement, active edge count, runtime, memory proxy, and a post-update
reforward. It compares all-mode movement, projected closure, gradient
coverage, active edges, runtime, and memory against the hard-top-k
temperature-aware M74/M75 path when those artifacts exist.

M76 is a diagnostic warm-start probe only. It cannot validate 10k training,
production readiness, deployment behavior, or realistic-profile model quality.

### M77 All-Mode Warm-Start Interpretation

M77 runs:

```bash
python -B scripts/analysis/report_all_mode_warm_start_10k.py --json-in .agent/tmp/all_mode_warm_start_10k_probe.json
```

It writes `.agent/tmp/all_mode_warm_start_10k_interpretation.json` and `.md`.
The report reads the M76 all-mode artifact and optional M74/M75/M72 context
artifacts. It classifies default all-mode as sufficient, insufficient,
impractically slow, blocked, or inconclusive, and compares movement against
the hard-top-k temperature-aware path.

The expected current interpretation is that default all-mode at
`score_scale=1.0` and `row_softmax_temperature=1.0` is numerically healthy but
weaker than M74 and impractically slow. That can authorize an
`M78_all_mode_temperature_score_grid_probe`, not production training or a
general rejection of calibrated all-mode warm-start.

### M78 Convergence Root-Cause Audit

M78 runs:

```bash
python -B scripts/analysis/run_convergence_root_cause_audit.py
```

It writes `.agent/tmp/convergence_root_cause_audit.json` and `.md`. The default
readiness source is
`.agent/tmp/temperature_aware_30step_interpretation.json`, which must recommend
`M76_all_mode_warm_start_10k_probe` or
`M78_all_mode_temperature_score_grid_probe` unless forced.

The audit reports evaluator sensitivity to deterministic query-support
perturbations, fixed-support selected-weight capacity, edge-score-only
optimizable movement, deterministic candidate-oracle capacity, loss-gradient
decomposition, and GNN score dynamic range. It remains diagnostic-only and
does not add direct physics objectives, stochastic sampling, binomial
enumeration, dense `N x N` graph operations, or long 10k training.

### M79 Scorer Parameterization Redesign

M79 runs:

```bash
python -B scripts/analysis/report_scorer_parameterization_redesign.py --json-in .agent/tmp/convergence_root_cause_audit.json
```

It writes `.agent/tmp/scorer_parameterization_redesign.json` and `.md`. The
default readiness source must have
`convergence_root_cause_status=scorer_parameterization_bottleneck` unless
forced.

The report is design-only. It records the current scorer bottleneck, missing
dynamic-range and layer-wise diagnostics, current scorer architecture summary,
safe redesign options, and correctness requirements. It can recommend
`M79_1_scorer_dynamic_range_diagnostic` or a later score-gain prototype, but it
does not run training, change `HierarchicalGNNScorer.forward`, modify topology
construction, add direct physics losses, or validate 10k model quality.

### M79.1 Scorer Dynamic Range Diagnostic

M79.1 runs:

```bash
python -B scripts/analysis/run_scorer_dynamic_range_diagnostic.py
```

It writes `.agent/tmp/scorer_dynamic_range_diagnostic.json` and `.md`. The
default readiness source is `.agent/tmp/scorer_parameterization_redesign.json`,
which must recommend `M79_1_scorer_dynamic_range_diagnostic` unless forced.

The report records edge-score distributions, selected/unselected top-k score
statistics, top-k boundary margins, one-step edge-score movement, layer-wise
parameter/gradient/update norms, score-head sensitivity, feature and embedding
scale statistics, and diagnostic score-scale what-if rows. It is a diagnostic
gate before any scorer redesign; it does not implement `score_output_gain`,
change GNN forward semantics, modify constructor rules, add loss terms, or
validate production 10k training.

### M80 Score Output Gain Probe

M80 runs:

```bash
python -B scripts/analysis/run_score_output_gain_probe.py --score-output-gains 1,10,30,100 --probe-steps 1
```

It writes `.agent/tmp/score_output_gain_probe.json` and `.md`. The default
readiness source is `.agent/tmp/scorer_dynamic_range_diagnostic.json`, which
must set `ready_for_M80_score_output_gain_prototype=true` unless forced.

The report records per-gain edge-score range, selected-score range, top-k
boundary margin, one-step edge-score movement, support stability,
topology-weight movement, query-support movement, C/F/D/E deltas, finite
loss/gradient status, update health, safe/unsafe gain candidates, and the
recommended next diagnostic stage. The probe uses the explicit model
`score_output_gain`; it does not use a hidden constructor score scale, add
loss terms, or validate production training.

### M81 Score-Gain Few-Step Smoke

M81 runs:

```bash
python -B scripts/analysis/run_score_gain_tenk_few_step_smoke.py --max-steps 3 --compare-gains 10,30
```

It writes `.agent/tmp/score_gain_tenk_few_step_smoke.json` and `.md`. The
default readiness source is `.agent/tmp/score_output_gain_probe.json`, which
must set `ready_for_M81_score_gain_tenk_smoke=true` unless forced.

The report records one few-step case per explicit scorer gain, with per-step
post-update reforward metrics, edge-score-to-top-k-margin ratios,
topology/query movement, support stability, update health, and comparisons
against M70/M74/M65/M61 artifacts when available. It is still a diagnostic
smoke and does not validate 10k training.

### M82 Score-Gain Few-Step Interpretation

M82 runs:

```bash
python -B scripts/analysis/report_score_gain_tenk_few_step.py --json-in .agent/tmp/score_gain_tenk_few_step_smoke.json
```

It writes `.agent/tmp/score_gain_tenk_few_step_interpretation.json` and `.md`.
The report is interpretation-only. It reads M81 and optional M80/M79.1/M70/M74
context artifacts, classifies gain 10 and gain 30 movement, compares gain
movement with M70 and M74 temperature-aware baselines, and recommends the next
diagnostic stage. It does not run additional 10k training, change the scorer,
change topology construction, or promote score gain as a production-ready
solution.

### M83 Combined Score-Gain And Temperature Probe

M83 runs:

```bash
python -B scripts/analysis/run_combined_gain_temperature_probe.py --probe-steps 3
```

It writes `.agent/tmp/combined_gain_temperature_probe.json` and `.md`. The
default readiness source is
`.agent/tmp/score_gain_tenk_few_step_interpretation.json`, whose
`recommended_next_stage` must be `M83_combined_gain_temperature_probe` unless
forced.

The report records one grid cell per `score_output_gain`,
`row_softmax_temperature`, and explicit `score_scale`. It compares each cell
against M81 gain-only, M74, M70, and M65 artifacts when available, then reports
support stability, update health, projected reliability closure, and a
conservative next-stage recommendation. M83 remains a diagnostic probe and does
not validate 10k training.

M83 writes deterministic visualization artifacts when output writing is enabled.
The main plots are fixed-node step trends, so a reviewer can inspect how loss,
C/F/D/E, reliability gap, gradients, edge-score movement, topology weights,
query support, and support stability evolve within the same node scale over the
diagnostic steps. Cross-configuration charts over gain/temperature/score-scale
cells remain available as secondary comparisons. These figures are report views
only and do not alter model, evaluator, loss, or constructor behavior.

## M84 Edge-Score Teacher Diagnostics

M84 evaluates deterministic teacher edge-score directions as reportable
diagnostics. The evaluator is still the same closed-form V2X consensus bridge,
and teacher modes are judged only by C/F/D/E deltas, topology/query movement,
support stability, and scorer-teacher alignment. The probe records whether
teacher directions move metrics, but this is not model-quality validation and
does not authorize a default teacher loss.

### M85 Edge-Score Teacher Interpretation

M85 runs:

```bash
python -B scripts/analysis/report_edge_score_teacher_probe.py --json-in .agent/tmp/edge_score_teacher_probe.json
```

It writes `.agent/tmp/edge_score_teacher_interpretation.json` and `.md`. The
report is artifact-only and does not rerun teacher probes. It classifies useful
gradient-teacher movement separately from unstable or wrong-direction oracle
movement, records static score/rank alignment separately from update-direction
alignment, and compares the result against M78 scorer-bottleneck and M83
combined-gain/temperature evidence.

A positive M85 result can authorize only an M86 teacher-guided scorer design or
probe. It cannot validate 10k training, promote candidate-oracle ranking, add a
default teacher objective, or make production/deployment claims.

### M86 Teacher-Guided Scorer Design

M86 runs:

```bash
python -B scripts/analysis/report_teacher_guided_scorer_design.py --json-in .agent/tmp/edge_score_teacher_interpretation.json
```

It writes `.agent/tmp/teacher_guided_scorer_design.json` and `.md`. The report
is design-only and requires M85 to classify the source as
`teacher_direction_found_but_update_path_bottleneck` unless forced.

The report records source teacher evidence, unsafe teacher exclusions,
candidate teacher-guided mechanisms, and a governance contract for future
prototypes. The expected current recommendation is
`edge_score_delta_distillation` with stop-gradient teacher targets and required
teacher-off, teacher-on, shuffled-teacher, and zero-teacher ablations. M86 does
not add a teacher objective to default training and keeps
`ready_for_default_training_change=false`.

### M87 Edge-Score Delta Distillation

M87 runs:

```bash
python -B scripts/analysis/run_edge_score_delta_distillation.py --steps 3 --teacher-weights 0.01,0.1,1.0
```

It writes `.agent/tmp/edge_score_delta_distillation.json` and `.md`. The
default readiness source is `.agent/tmp/teacher_guided_scorer_design.json`,
which must set `ready_for_M87_teacher_guided_prototype=true` while keeping
`ready_for_default_training_change=false` unless forced.

The diagnostic computes the unchanged raw coupled C/D/E loss, derives a
stop-gradient edge-score target from the local negative edge-score gradient,
and adds the teacher loss only inside this explicit M87 prototype. It reports
teacher-off, teacher-on, shuffled-teacher, and zero-teacher ablations, one
teacher-on run per configured teacher weight, support stability, topology/query
movement, C/F/D/E movement, finite loss/gradient checks, and whether teacher-on
beats the controls.

M87 cannot validate 10k training, promote a teacher objective into default
training, authorize `candidate_oracle_rank`, or make production/deployment
claims.

### M88 Edge-Score Delta Distillation Interpretation

M88 runs:

```bash
python -B scripts/analysis/report_edge_score_delta_distillation.py --json-in .agent/tmp/edge_score_delta_distillation.json
```

It writes `.agent/tmp/edge_score_delta_distillation_interpretation.json` and
`.md`. The report is artifact-only and does not rerun M87. It reads the
teacher-off, teacher-on, shuffled-teacher, and zero-teacher ablations, reports
per-run C/F/D/E deltas and teacher-loss movement, and records whether teacher
loss was numerically dominated by the scaled primary C/D/E objective.

M88 can recommend only another diagnostic teacher strategy. It cannot add a
default teacher objective, validate teacher-guided 10k training, promote
candidate-oracle ranking, or make production/deployment claims.

### M89 Teacher Loss Scale Probe

M89 runs:

```bash
python -B scripts/analysis/run_teacher_loss_scale_probe.py --steps 3 --teacher-weights 1,1e3,1e6,1e9,1e12
```

It writes `.agent/tmp/teacher_loss_scale_probe.json` and `.md`. The default
readiness source is
`.agent/tmp/edge_score_delta_distillation_interpretation.json`, which must
classify M87 as `teacher_loss_scale_too_small` and recommend
`M89_teacher_loss_scale_probe` unless forced.

For every teacher weight, M89 runs teacher-off, teacher-on,
shuffled-teacher, and zero-teacher controls under the same sparse top-k,
formal row-softmax-temperature, fused-query-support, C/D/E-loss path. The
report records unweighted teacher loss, scaled teacher loss, scaled primary
loss, diagnostic total loss, teacher/control C/F/D/E deltas, support
stability, and whether teacher-on beats all controls.

M89 remains an opt-in diagnostic. It cannot validate teacher-guided training,
promote a teacher objective into the default path, use candidate-oracle
ranking, or make production/deployment claims.

### M90 Teacher Loss Scale Interpretation

M90 runs:

```bash
python -B scripts/analysis/report_teacher_loss_scale_probe.py --json-in .agent/tmp/teacher_loss_scale_probe.json
```

It writes `.agent/tmp/teacher_loss_scale_interpretation.json` and `.md`. The
report is artifact-only and does not rerun M89. It classifies every teacher
weight as too small, distinguishable but not useful, useful and safe, unstable,
or invalid, then decides whether MSE-style edge-score delta distillation has a
safe useful scale.

The expected current result is that low weights are numerically dominated,
`1e6` is distinguishable but still below the useful threshold, and high weights
are support-jumpy. That authorizes only a next diagnostic such as directional
teacher alignment, not default teacher-guided training.

### M91 Directional Teacher Alignment Probe

M91 runs:

```bash
python -B scripts/analysis/run_directional_teacher_alignment_probe.py --steps 3 --teacher-weights 0.01,0.1,1.0,10.0
```

It writes `.agent/tmp/directional_teacher_alignment_probe.json` and `.md`.
The default readiness source is
`.agent/tmp/teacher_loss_scale_interpretation.json`, which must classify M90
as `mse_distillation_no_safe_useful_scale` and recommend
`M91_directional_teacher_alignment_probe` unless forced.

For every alignment mode, M91 runs teacher-off, aligned-teacher-on for each
configured teacher weight, shuffled-teacher, and zero-teacher controls under
the same sparse top-k, formal row-softmax-temperature, fused-query-support,
C/D/E-loss path. The report records alignment loss, alignment cosine, metric
deltas, topology/query movement, support stability, update health, and whether
an aligned-teacher-on weight beats all controls.

M91 remains an opt-in diagnostic. It cannot validate teacher-guided training,
promote a teacher objective into the default path, use candidate-oracle
ranking, use direct physics losses, sample teacher edges, or make
production/deployment claims.

### M92 Directional Teacher Alignment Interpretation

M92 runs:

```bash
python -B scripts/analysis/report_directional_teacher_alignment.py --json-in .agent/tmp/directional_teacher_alignment_probe.json
```

It writes `.agent/tmp/directional_teacher_alignment_interpretation.json` and
`.md`. The report is artifact-only and does not rerun M91.

M92 classifies the M91 result as control-confounded, not validated, validated
for another opt-in diagnostic, or blocked. It explicitly separates diagnostic
evidence from smoke, training, model-quality, and production-readiness
evidence. A recommendation for score-head-only teacher update or Jacobian
alignment remains diagnostic-only and cannot change the default training path.

### M93 Score-Head-Only Teacher Update Probe

M93 runs:

```bash
python -B scripts/analysis/run_score_head_only_teacher_update_probe.py --steps 3 --teacher-weights 0.01,0.1,1.0
```

It writes `.agent/tmp/score_head_only_teacher_update_probe.json` and `.md`.
The default readiness source is
`.agent/tmp/directional_teacher_alignment_interpretation.json`, which must
recommend `M93_score_head_only_teacher_update_probe` unless forced.

M93 requires both update scopes, `score_head_only` and `full_model`, and each
scope must run teacher-off, aligned-teacher-on, shuffled-teacher, and
zero-teacher controls. The report records C/F/D/E deltas, alignment movement,
support stability, score-head versus non-score-head parameter movement, and
whether the score-head-only teacher effect beats controls.

Passing M93 can only authorize an M94 interpretation report. Failing or
control-confounded M93 evidence routes to support-smoothing design rather than
default teacher training.

### M94 Support Smoothing Design

M94 runs:

```bash
python -B scripts/analysis/report_support_smoothing_design.py --json-in .agent/tmp/score_head_only_teacher_update_probe.json
```

It writes `.agent/tmp/support_smoothing_design.json` and `.md`. The report is
design-only and does not modify constructor code.

The design specifies explicit future constructor parameters such as
`support_smoothing_mode`, `support_smoothing_extra_per_row`,
`support_smoothing_temperature`, and `support_smoothing_stage`. Legacy defaults
must preserve current top-k/all behavior exactly. A positive M94 result can
authorize only `M95_support_smoothing_constructor_prototype`.

### M95 Support Smoothing Constructor Prototype

M95 runs:

```bash
python -B scripts/analysis/report_support_smoothing_constructor.py --readiness-artifact .agent/tmp/support_smoothing_design.json
```

It writes `.agent/tmp/support_smoothing_constructor_prototype.json` and `.md`.
The report proves only prototype constructor properties: legacy-equivalent
defaults, hard-top-k stage equivalence, deterministic sparse halo expansion,
row normalization, finite backward integration, train/eval/deploy forward-rule
consistency, and absence of dense N-by-N or sampling patterns.

Passing M95 can authorize only `M96_support_smoothing_10k_smoke`. M95 does
not provide training evidence, model-quality evidence, production-readiness
evidence, or permission to change the default training path.

### M96 Support Smoothing 10k Smoke

M96 runs:

```bash
python -B scripts/analysis/run_support_smoothing_10k_smoke.py --readiness-artifact .agent/tmp/support_smoothing_constructor_prototype.json
```

It writes `.agent/tmp/support_smoothing_10k_smoke.json` and `.md`. The smoke
uses explicit opt-in support-smoothing parameters and records C/F/D/E,
reliability gap, edge-score movement, topology-weight movement, query-support
movement, support overlap, gradients, updates, runtime, and memory proxies.

M96 compares against available M74 hard-top-k temperature, M76 all-mode, M81
score-gain, and M91 teacher-probe artifacts. If support smoothing improves
movement/projection without a support jump, it can authorize a controlled
strategy interpretation package. If movement is support-jumpy or not better,
it routes to `M97_loss_evaluator_sensitivity_review`.

### M97 Loss Evaluator Sensitivity Review

M97 runs:

```bash
python -B scripts/analysis/report_loss_evaluator_sensitivity_review.py --json-in .agent/tmp/support_smoothing_10k_smoke.json
```

It writes `.agent/tmp/loss_evaluator_sensitivity_review.json` and `.md`. The
review is report-only. It re-audits C/F sensitivity to query support, L_R
curve shape, L_R/L_D/L_E component movement, candidate graph capacity, and
evaluator threshold behavior without changing the objective or constructor.

If M97 finds that all available routes are flat, unstable, or support-jumpy,
it must stop at a human-review gate rather than authorizing stronger training
or objective changes.

The post-M97 gate package runs:

```bash
python -B scripts/analysis/report_human_review_gate_package.py --json-in .agent/tmp/loss_evaluator_sensitivity_review.json
```

It writes `.agent/tmp/human_review_gate_package.json` and `.md`. The package is
report-only and records that generic harness artifacts written later do not
supersede the active M91+ gate. It does not authorize a stronger run, objective
change, evaluator-threshold change, candidate-capacity change, default-path
change, or production-readiness claim.

### M98 Support-Smoothing Stabilization Design

M98 runs:

```bash
python -B scripts/analysis/report_support_smoothing_stabilization_design.py
```

It writes `.agent/tmp/human_review_decision_support_smoothing.json`,
`.agent/tmp/human_review_decision_support_smoothing.md`,
`.agent/tmp/support_smoothing_stabilization_design.json`, and
`.agent/tmp/support_smoothing_stabilization_design.md`. The report records that
support-smoothing stabilization research is authorized, but production
training, stronger runs, default-path changes, objective/evaluator changes,
dense support paths, stochastic support, direct physics losses, structural
bias in the default path, and hidden train-only rules remain unauthorized.

M98 is ready for M99 only when the support jump and not-full-10k limitations
remain explicit unresolved blockers, the conservative sweep grid is present,
and density-scaling requirements are documented. It does not authorize a full
10k smoke; M99 must first pass the support-stability gate.

### M99 Conservative Support-Smoothing Sweep

M99 runs:

```bash
python -B scripts/analysis/run_conservative_support_smoothing_sweep.py
```

It writes `.agent/tmp/conservative_support_smoothing_sweep.json` and `.md`.
The sweep is diagnostic only. It evaluates the M98 recommended grid over
existing explicit knobs and selects a candidate only when the case beats M74 or
M76 movement, reduces projected closure versus M75 when available, has finite
losses and gradients, keeps update health stable, preserves sparse O(Nk)
construction, and keeps active-edge change below the blocker threshold.

If M99 finds a stable candidate with an overridden node count, the only allowed
next step is a full node_count=10000 diagnostic smoke. If it finds no stable
candidate, the branch returns to human review rather than moving to objective,
evaluator, default-path, or production-training changes.

### M100 Full 10k Support Smoothing Smoke

M100 runs:

```bash
python -B scripts/analysis/run_full_10k_support_smoothing_smoke.py
```

It writes `.agent/tmp/full_10k_support_smoothing_smoke.json` and `.md`. The
smoke reuses the selected M99 parameters at `node_count=10000` with
`node_count_overridden=false` and reports support stability, C/F/D/E trends,
reliability-gap projection, runtime/memory proxies, sparse/dense ratio, density
stats, diagnostics-mode limitation, and comparisons with M74, M76, and M96.

Passing M100 can authorize only `M101_support_smoothing_full_10k_interpretation`.
Failing M100 is a human-review stop condition; it does not authorize objective,
evaluator, default-path, stronger-run, or production-training changes.

### M101 Support Smoothing Scale-Transfer Audit

M101 runs:

```bash
python -B scripts/analysis/run_support_smoothing_scale_transfer_audit.py
```

It writes `.agent/tmp/support_smoothing_scale_transfer_audit.json` and `.md`.
The audit requires the M100 artifact to show
`full_10k_support_smoothing_smoke_status=full_10k_movement_not_retained` unless
`--force` is used for diagnostic repair. It reruns the M99 selected candidate
over `node_counts=[100,500,2000,5000,10000]` and reports per-node C/F/D/E
movement, reliability-gap projection, edge-score/topology/query movement,
gradient and update norms, support overlap, active-edge change, sparse density,
runtime, and memory proxies.

M101 may identify movement attenuation, update-signal collapse, support
overdamping, or density-scaling ambiguity. It can recommend only a report/design
M102 follow-up and cannot validate support smoothing, authorize stronger
training, change defaults, change the objective/evaluator, or claim production
readiness.

### M102 Scale-Aware Update Normalization Design

M102 runs:

```bash
python -B scripts/analysis/report_scale_aware_update_normalization_design.py
```

It writes `.agent/tmp/scale_aware_update_normalization_design.json` and `.md`.
The report consumes the M101 scale-transfer audit, summarizes the 500-to-10000
node movement/update collapse, and specifies explicit opt-in normalization
formula candidates and a conservative M103 probe grid.

M102 is design evidence only. It cannot validate support smoothing, authorize
production training or stronger training, change defaults, change the
objective/evaluator, or weaken sparse O(Nk), no-sampling, no-direct-physics,
and train/validation/deployment consistency constraints.

### M103 Scale-Aware Update Normalization Probe

M103 runs:

```bash
python -B scripts/analysis/run_scale_aware_update_normalization_probe.py
```

It writes `.agent/tmp/scale_aware_update_normalization_probe.json` and `.md`.
The probe requires the M102 design artifact unless `--force` is used for
diagnostic repair. It compares no-normalization controls against bounded
node-count and active-edge-count normalization candidates at 500, 2000, and
10000 nodes.

M103 can report whether bounded normalization restores 10k diagnostic movement
relative to the no-normalization control while preserving finite gradients,
stable updates, sparse density, and support stability. It cannot validate
support smoothing, authorize production training, change defaults, change the
objective/evaluator, or claim model quality.

### R1 Scale-Law Root-Cause Audit

R1 runs:

```bash
python -B scripts/analysis/report_scale_law_root_cause_audit.py
```

It writes `.agent/tmp/scale_law_root_cause_audit.json` and
`reports/training/scale_law_root_cause_audit_README.md`. The audit reads the
M99-M103 artifacts and reports cross-scale C/F/D/E movement, reliability-gap
movement, edge-score/topology/query-support deltas, gradient and update norms,
support stability, degree stability, loss reductions, query-support
aggregation, smoothing mass, and density limitations.

R1 is report-only. It cannot authorize training, new losses, objective or
evaluator changes, default-path changes, or stronger sweeps.

### R2 Scale-Law Remediation Design

R2 runs:

```bash
python -B scripts/analysis/report_scale_law_remediation_design.py
```

It writes `.agent/tmp/scale_law_remediation_design.json`,
`.agent/tmp/scale_law_remediation_design.md`, and
`reports/training/scale_law_remediation_design_README.md`. The report consumes
R1 and compares remediation design options for loss/update scaling,
query-support sensitivity, smoothing mass/entropy, scorer update rescale, and
environment density contracts.

R2 can recommend only one human-reviewed R3 design or audit branch. It cannot
recommend implementation prototypes, production training, default-path changes,
objective/evaluator changes, direct physics losses, dense N-by-N graph paths,
or support-smoothing behavior changes.

### R3 Scale-Invariant Loss/Update Design

R3 runs:

```bash
python -B scripts/analysis/report_scale_invariant_loss_update_design.py
```

It writes `.agent/tmp/scale_invariant_loss_update_design.json`,
`.agent/tmp/scale_invariant_loss_update_design.md`, and
`reports/training/scale_invariant_loss_update_design_README.md`. The report
consumes R2, R1, and optional M103/M101 context, then designs scale-invariant
loss/update candidates while keeping raw C/D/E evaluator metrics and
`raw_total_loss` unchanged.

R3 can recommend only an R4 design review. It cannot authorize an
implementation prototype, production training, default-path changes,
objective/evaluator behavior changes, optimizer changes, constructor changes,
support-smoothing changes, stronger training, dense N-by-N graph paths, or
direct physics losses.

### R4 Scale-Invariant Design Review

R4 runs:

```bash
python -B scripts/analysis/report_scale_invariant_design_review.py
```

It writes `.agent/tmp/scale_invariant_design_review.json`,
`.agent/tmp/scale_invariant_design_review.md`, and
`reports/training/scale_invariant_design_review_README.md`. The report reviews
the R3 mixed design, ranks candidate branches, and records whether missing
extra-edge mass, row entropy, topology-to-query sensitivity, active-row query
sensitivity, active-failure counts, and spatial-density fields must be
collected before a prototype.

R4 can recommend only one bounded R5 diagnostic, prototype, or audit branch.
For the accepted R3 evidence, the expected branch is a topology smoothing mass
and query-sensitivity diagnostic. R4 cannot authorize a full mixed prototype,
production training, default-path changes, objective/evaluator behavior
changes, constructor changes, optimizer changes, scheduler changes, stronger
training, or direct physics losses.

### R5 Topology Smoothing Mass and Query-Sensitivity Diagnostic

R5 runs:

```bash
python -B scripts/analysis/run_topology_smoothing_mass_query_diagnostic.py
```

It writes `.agent/tmp/topology_smoothing_mass_query_diagnostic.json`,
`.agent/tmp/topology_smoothing_mass_query_diagnostic.md`, and
`reports/training/topology_smoothing_mass_query_diagnostic_README.md`.
The default diagnostic executes node counts 500, 2000, and 10000 for two
diagnostic steps using the accepted M99/M100/R4 support-smoothing candidate.
Harness targets may use a smaller forced node-count override for speed, but
the script default remains the full cross-scale diagnostic.

The output records extra-edge and base-top-k mass, row entropy and effective
degree, topology-weight deltas, query-support deltas, active-row sensitivity,
active-failure-node counts, cross-scale ratios, readiness flags, allowed
claims, and disallowed claims. R5 may classify smoothing-mass, entropy,
active-row, or query-sensitivity bottlenecks, but it cannot authorize
production training, default-path changes, objective/evaluator changes,
constructor changes, support-smoothing behavior changes, stronger training, or
model-quality claims.

### R6 Query-Support Sensitivity Counterfactual Probe

R6 runs:

```bash
python -B scripts/analysis/run_query_support_sensitivity_counterfactual_probe.py
```

It writes `.agent/tmp/query_support_sensitivity_counterfactual_probe.json`,
`.agent/tmp/query_support_sensitivity_counterfactual_probe.md`, and
`reports/training/query_support_sensitivity_counterfactual_probe_README.md`.
The default diagnostic executes node counts 500, 2000, and 10000 with
support-preserving topology-weight perturbation scales `1e-4`, `1e-3`, and
`1e-2`.

The perturbation modes are `uniform_extra_edge_mass_shift`,
`active_row_focused_mass_shift`, `entropy_reduction_shift`,
`entropy_increase_shift`, and `gradient_direction_topology_perturbation`.
They are deterministic post-construction perturbations over existing sparse
support. R6 reports row-normalization preservation, nonnegative weights,
query-support deltas, C/F/D/E deltas, active-row query deltas, best sensitivity
mode/scale, and whether the primary bottleneck is evaluator insensitivity or
model/update failure to produce effective topology perturbations.

R6 cannot authorize production training, default-path changes,
objective/evaluator behavior changes, constructor changes, support-smoothing
behavior changes, stronger training, prototype losses, or model-quality
claims.

### R7 Realistic Density Perturbation Capacity Diagnostic

R7 runs:

```bash
python -B scripts/analysis/run_realistic_density_perturbation_capacity_diagnostic.py
```

It writes `.agent/tmp/realistic_density_perturbation_capacity_diagnostic.json`,
`.agent/tmp/realistic_density_perturbation_capacity_diagnostic.md`, and
`reports/training/realistic_density_perturbation_capacity_diagnostic_README.md`.
The default diagnostic tests node counts 500, 2000, and 10000 for `toy` and
`small_realistic` profiles using the R6 gradient-direction topology
perturbation over existing sparse support.

R7 reports candidate/active degree stability, available spatial density
fields, profile density status, query-support and C/F/D/E perturbation
capacity, cross-profile toy-to-proxy ratios, and whether the toy conclusion
transfers beyond toy. `small_realistic` is treated as a proxy unless a real
production-like profile with an explicit scalable density contract is present.
R7 cannot authorize production training, a default-path change, evaluator or
constructor behavior changes, support-smoothing changes, or a training
prototype.

### R8 Production Density Profile Probe

R8 runs:

```bash
python -B scripts/analysis/run_production_density_profile_probe.py
```

It writes `.agent/tmp/production_density_profile_probe.json`,
`.agent/tmp/production_density_profile_probe.md`, and
`reports/training/production_density_profile_probe_README.md`. The default
probe checks node counts 100, 500, 2000, 5000, and 10000 for `toy`,
`small_realistic`, and `production_like_density_v0`.

`production_like_density_v0` is a diagnostic profile generator, not a dataset
loader. It scales spatial area with node count, uses an explicit candidate
radius/cap, targets stable candidate and active degree, reports link-success
quantiles, and requires reliability to remain unsolved at large N. R8 can mark
the profile as a production-density contract candidate only after this sanity
probe passes; production training, default-path changes, evaluator changes,
constructor changes, and model-quality claims remain disallowed.

## R9 Production Profile Perturbation Capacity Evaluation

R9 runs:

```bash
python -B scripts/analysis/run_production_profile_perturbation_capacity_probe.py
```

It writes `.agent/tmp/production_profile_perturbation_capacity_probe.json`,
`.agent/tmp/production_profile_perturbation_capacity_probe.md`, and
`reports/training/production_profile_perturbation_capacity_probe_README.md`.
The default probe uses `production_like_density_v0` across node counts 100,
500, 2000, 5000, and 10000 with deterministic support-preserving
gradient-direction topology perturbations.

R9 reports query-support movement, C/F/D/E deltas, row-normalization and
nonnegative-weight preservation, density fields, and the best sensitivity
scale. It does not validate training, model quality, default-path changes, or
final production readiness.

## R10 Production Profile Training Signal Evaluation

R10 runs:

```bash
python -B scripts/analysis/run_production_profile_training_signal_diagnostic.py
```

It writes `.agent/tmp/production_profile_training_signal_diagnostic.json`,
`.agent/tmp/production_profile_training_signal_diagnostic.md`, and
`reports/training/production_profile_training_signal_diagnostic_README.md`.
The default probe covers `production_like_density_v0` at node counts 100, 500,
2000, 5000, and 10000.

R10 reports raw loss, C/F/D/E metrics, loss-to-edge-score and
loss-to-topology-weight gradient norms, projected parameter update magnitudes,
and detached edge-score gradient counterfactual movement. It does not run an
optimizer step or validate production training.

## R11 Loss-to-Score Gradient Alignment Evaluation

R11 runs:

```bash
python -B scripts/analysis/run_loss_to_score_gradient_alignment_diagnostic.py
```

It writes `.agent/tmp/loss_to_score_gradient_alignment_diagnostic.json`,
`.agent/tmp/loss_to_score_gradient_alignment_diagnostic.md`, and
`reports/training/loss_to_score_gradient_alignment_diagnostic_README.md`.
The default probe covers `production_like_density_v0` at node counts 100, 500,
2000, 5000, and 10000.

R11 reports component gradients for `L_R`, `L_D`, `L_E`, and `total_loss`,
component-gradient cosine similarities, detached edge-score component
counterfactuals, topology-weight total-gradient counterfactuals, and an
alignment classification. It is a diagnostic only and does not validate
production training.

## R12 Constructor Jacobian Alignment Evaluation

R12 runs:

```bash
python -B scripts/analysis/run_constructor_jacobian_alignment_diagnostic.py
```

It writes `.agent/tmp/constructor_jacobian_alignment_diagnostic.json`,
`.agent/tmp/constructor_jacobian_alignment_diagnostic.md`, and
`reports/training/constructor_jacobian_alignment_diagnostic_README.md`. The
default probe covers `production_like_density_v0` at node counts 100, 500,
2000, 5000, and 10000 with 10000 as the focus count.

R12 reports fixed-support row-softmax alignment, full constructor re-forward
alignment, base-vs-extra smoothing mass transfer, row-level alignment, and
support-expression diagnostics. It is a diagnostic only and does not validate
production training.
