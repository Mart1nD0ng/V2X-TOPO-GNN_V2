# Architecture

The project builds topology plans in this order:

1. Calibrate a feasible 5G NR-V2X urban-grid environment.
2. Generate sparse O(Nk) candidate edges with physical and topological diversity.
3. Score candidate topology structure with hierarchical GNN components.
4. Construct one shared sparse topology from candidate edge scores under degree, sector, and cross-cell constraints.
5. Evaluate Avalanche/Snowball reliability, delay, and energy with deterministic closed-form math.
6. Use the M3 evaluator bridge to connect sparse V2X topology, channel proxy link success, graph-coupled Avalanche C/D, and consensus energy proxy E.
7. Optimize only C_avalanche, D_avalanche, and E_avalanche through a later coupled objective.
8. Validate scale claims through seeded baselines and sparse memory checks.

This harness intentionally stops before production model implementation.

## Invariants

- Environment feasibility comes before model training.
- Forward topology rules are identical in train, validation, and deployment.
- Consensus is Avalanche/Snowball, not phase-count consensus.
- Hidden physics variables stay out of the public loss contract.
- M3 is an evaluator bridge only: it reports differentiable C/D/E metrics and diagnostics, but it does not implement topology training or an optimization objective.
- M5 adds `TopologyConstructionLayer` v1. It is a deterministic sparse constructor, not a full hierarchical GNN, and maps candidate edge scores to row-normalized sparse query weights that can feed the M3 evaluator and M4 loss.
- The M5 constructor uses the same forward path in every module state; there is no separate validation-only threshold or repair path.
- M5's default is Mode B, a deployable sparse query distribution. Deterministic top-k support selection is optional and applied identically in every phase; hard-forward/ST construction remains a future Mode A alternative.
- M5.2 adds a GNN-readiness path: `support_mode="all"` keeps all sparse candidates for broader early score gradients, while `support_mode="topk"` enforces a fixed deployable cap. The constructor no longer performs per-edge CPU list grouping; top-k keeps only a unique-source-row loop.
- M6 adds `HierarchicalGNNScorer` as an encoder/scorer skeleton only. It maps sparse node and candidate-edge features to differentiable `edge_score` values for M5, using PyTorch MLP encoders, sparse `index_add` message passing, optional lightweight region pooling/broadcast, and an edge score head. It does not add training loops, optimizers, datasets, or checkpoints.
- M6.1 hardens the scorer for the next phase. Normal construction defaults to `init_mode="xavier"`; `init_mode="kaiming"` is also supported, and `init_mode="deterministic"` is reserved for tests/debug reproducibility. Region count can be passed explicitly through `num_regions` to avoid max-id inference in scale runs.
- M6.2 adds differentiable structural heads for node budget logits, role logits, sector preferences, and sparse candidate-edge region bridge logits. These heads can bias sparse edge scores, but they are not losses and do not yet replace the fixed constructor budgets.
- M6.3 adds structural diagnostics and a deterministic budget-to-constructor bridge. `expected_budget_to_cap(...)` maps soft budget expectations to integer caps for experiments, detaching by default because integer caps are nondifferentiable. This bridge must be used with the same rule in train, validation, and deployment for any experiment.
- M6.4 adds structural auxiliary objective utilities and training-readiness diagnostics. These are optional curriculum tools for structural heads; they are not the M4 C/D/E coupled objective and they do not make hard integer caps differentiable.
- M7 adds a tiny deterministic training smoke. It proves optimizer steps can pass through `HierarchicalGNNScorer -> TopologyConstructionLayer -> M3 evaluator -> M4 loss` and optional structural auxiliaries, but it is not a training system, dataset loader, checkpointing workflow, or experiment runner.
- M7.1 hardens that smoke with named Avalanche profiles, trend diagnostics, optional gradient clipping diagnostics, same-seed reproducibility checks, and an all-candidate to top-k curriculum smoke. It still is not a full trainer.
- M7.2 hardens curriculum semantics. Phase-specific fields override top-level smoke defaults during curriculum runs, and reports disclose each phase's budget strategy, budget-head training flag, auxiliary loss, and budget gradient status.
- M8 adds a scale/runtime/memory evaluation harness. M8.1 splits it into fast, medium, and large tiers so normal checks stay deterministic and portable. It measures computational viability and numerical stability, not model quality.
- M9 adds small deterministic stability and ablation smoke runs across seeds and simple configuration changes. It remains a harness over the existing M7 path, not a production training system or model-selection workflow.
- M9.1 splits stability reporting into smoke, short, and analysis tiers. Normal checks run only the smoke tier; short and analysis are explicit because they run more ablation cases.
- M9.3 adds a readiness gate for M10. The gate blocks on numerical failures, nonfinite values, zero gradients, or exploding-loss diagnostics; loss decrease is not a hard requirement.
- M10 adds a minimal experiment harness. It orchestrates existing M7/M9 smoke primitives over tiny seed/case/profile matrices and writes reports, but it is not a production training system or model-selection layer.

## M6 Sparse Model Skeleton

`src/models/hierarchical_gnn.py` is the first executable model-side component. Its input contract is sparse:

- `num_nodes`
- `src_index`
- `dst_index`
- `node_features`
- `edge_features`
- optional `region_id`
- optional `region_features`
- optional `num_regions`
- optional `edge_sector_id`
- optional `edge_is_cross_region`

The scorer never constructs dense node-by-node tensors. Candidate features are encoded, messages are aggregated with `index_add`, optional region context is pooled and broadcast by region id, and the edge score head emits one scalar per candidate edge.

M6.2 is still not a full RegionGNN training system. It now emits structural head outputs, but it does not run epochs, choose deployment budgets, perform repair, or optimize a constrained topology by itself.

Current structural outputs:

- `node_budget_logits`: soft per-node budget-bin scores.
- `node_budget_expected`: differentiable expected budget-bin value for diagnostics/curricula.
- `node_role_logits`: ordinary/relay/anchor-style role logits.
- `sector_preference_logits`: per-node sector preference logits.
- `region_bridge_logits`: sparse per-candidate-edge region-pair bridge logits.

The budget head is soft and diagnostic in M6.2. Hard mapping from predicted budgets into deployment caps is deferred to M6.3/M7. Region bridge, sector, and role heads are optional score-bias mechanisms in M6.2, not final constrained optimizers.

M6.3 provides the first deterministic bridge from soft budget heads to constructor caps:

```text
node_budget_expected -> expected_budget_to_cap(...) -> per_node_budget
```

The adapter returns integer caps and detaches by default. It does not create a gradient path through hard caps. Budget heads can be trained later only through differentiable surrogates, auxiliary/curriculum designs, or other explicit objectives that preserve the same forward rule.

Scorer diagnostics include score quantiles, score entropy by source, node and edge embedding norm summaries, incoming message counts, zero-incoming node count, zero-outgoing candidate node count, candidate-graph isolation, structural bias component summaries, structural head entropies, argmax counts, and bridge-logit summaries. These are debugging and gradient-health diagnostics; they are not loss terms.

M6.4 defines which structural heads can learn from which paths:

- Edge path, sector head, role head, and sparse bridge head can receive gradients from the main C/D/E loss when structural score bias is enabled.
- Budget head does not receive gradients through `expected_budget_to_cap(...)` because integer caps are detached.
- Budget head can receive gradients only through explicit auxiliary objectives or future differentiable budget surrogates.

Auxiliary structural objectives live in `src/models/structural_objectives.py` and are diagnostics/curriculum tools only. They are not public C/D/E objectives.

`configs/training_smoke.yaml` is a draft for later M7 smoke work. It separates all-candidate warm-start from capped top-k deployment-style experiments and requires cap histograms, gradient coverage, and structural gradient-path reporting.

## M7 Tiny Training Smoke

`src/training/training_smoke.py` contains a smoke-only optimizer harness. It generates one deterministic V2X snapshot, builds a sparse candidate graph, scores candidate edges, constructs a sparse topology, evaluates C/D/E, computes the M4 coupled loss, optionally adds structural auxiliary objectives, and takes a small fixed number of optimizer steps.

This is intentionally not a full trainer. It has no dataset abstraction, epochs, checkpointing, distributed execution, scheduler, or experiment manager.

The default phase uses `support_mode="all"` to maximize score-gradient coverage. A separate `support_mode="topk"` smoke validates the deployment-like capped path, where candidates excluded by top-k receive no score gradient in that forward pass. The budget head still does not learn through detached integer caps; budget-head gradients require the explicit auxiliary budget objective or a future differentiable surrogate.

M7.1 adds named smoke profiles:

- `toy`: `k=1`, `alpha=1`, `beta=2`, `rounds=3`; this is the fast default for normal checks.
- `small_realistic`: `k=5`, `alpha=3`, `beta=5`, `rounds=20`; this is an explicit smoke target before making stronger training-stability claims.

The curriculum smoke carries model parameters from a short `support_mode="all"` phase into a short `support_mode="topk"` phase. It reports the support-switch change in active edge count and gradient coverage. Loss decrease is diagnostic only, not a success criterion; the required contract is finite metrics, finite gradients, parameter movement, and no change to the shared train/validation/deployment graph rule.

M7.2 makes phase-specific curriculum semantics explicit. During curriculum only, `phase_0_budget_strategy`, `phase_0_train_budget_head`, `phase_1_budget_strategy`, and `phase_1_train_budget_head` override top-level smoke defaults. The default phase 1 config may activate auxiliary budget-head objectives, but hard top-k caps remain detached and nondifferentiable. Any phase 1 budget-head gradient must come from the auxiliary objective and must be reported separately from C/D/E.

The intended M6 proof path is:

```text
GNN parameters
  -> edge_score
  -> TopologyConstructionLayer
  -> V2X consensus evaluator
  -> coupled C/D/E loss
  -> backward gradients
```

This is still not a training system. M6 validates differentiability and sparse interfaces before later hierarchical GNN design and training phases.

## M8 Scale Evaluation Harness

`src/training/scale_smoke.py` and `scripts/analysis/run_scale_training_smoke.py` quantify the current sparse pipeline:

```text
V2X candidate graph
  -> HierarchicalGNNScorer
  -> TopologyConstructionLayer
  -> graph-coupled Avalanche evaluator
  -> coupled C/D/E loss
  -> backward diagnostics
```

M8 does not run optimizer steps, save checkpoints, select models, or introduce datasets. It measures runtime, tensor-memory proxy, active edge count, gradient coverage, C/D/E metrics, failure-domain reliability, and gradient maxima for sparse scale cases.

M8.1/M8.2 define four tiers:

- `micro`: N=40, top-k support, toy Avalanche. This is for very slow machines and dependency/profiling smoke.
- `fast`: N=100, `support_mode in {all, topk}`, toy Avalanche only. This is the only scale tier included in `agent-check`.
- `medium`: N=100/500, all/top-k, toy plus small-realistic Avalanche. This is an explicit analysis target.
- `large`: N=100/500/2000/5000/10000, top-k toy by default. This is manual/explicit only.

The all-candidate mode is expected to maximize gradient coverage at the cost of more active edges. The top-k mode is expected to reduce active edges and memory while lowering gradient coverage. Timing fields include case wall time, environment generation, model initialization, constructor setup, scorer forward, topology construction, evaluator, loss, backward, postprocess, and forward+backward step time. M8.2 reports the dominant bottleneck stage and edge/node throughput proxies. Memory is deterministic tensor accounting, not process RSS or GPU peak memory. `runtime-env-check` reports Python, NumPy, SciPy, PyYAML, Torch, and CUDA availability as an explicit diagnostic target.

The scale entrypoint can also write deterministic SVG/CSV visualization artifacts with `--visualization-dir`. These artifacts plot loss, runtime, tensor-memory proxy, active edges, gradient coverage, and C/F/D/E metrics over node count. They are generated from the completed sparse report rows and do not alter model, evaluator, loss, or topology-constructor behavior.

## M9 Stability And Ablation Harness

`src/training/stability_smoke.py` wraps the existing tiny smoke path. It runs deterministic small cases and summarizes whether losses and gradients stay finite, whether parameters move, and whether C/D/E/F metrics change in interpretable directions.

M9 cases are:

- `baseline_all`: N=100, all-candidate support, toy Avalanche, 20 steps by default.
- `baseline_topk`: N=100, top-k support, toy Avalanche.
- `structural_bias_off`: baseline all-candidate support with structural score bias disabled.
- `budget_aux_off`: baseline all-candidate support without the auxiliary budget objective.
- `small_realistic_short`: N=100, all-candidate support, small-realistic Avalanche, short step count.

The fast target runs one seed and `baseline_all`. Explicit analysis runs use seeds `[7, 13, 23]` and compare baseline against structural-bias-off, budget-auxiliary-off, and top-k cases. Loss decrease is diagnostic only; finite losses, finite gradients, and parameter movement are the contract.

M9.1 tiers are:

- `smoke`: one seed, `baseline_all`, short step count. This is the agent-check tier.
- `short`: one seed over baseline, top-k, structural-bias-off, and budget-auxiliary-off cases. This is explicit but still lightweight.
- `analysis`: multiple seeds over the full ablation set, including short small-realistic consensus. This is explicit only.

Each case reports `case_status`, `case_error`, `case_timeout_s`, and runtime. The timeout field is recorded for future orchestration but is not a hard timeout mechanism. Short and analysis tiers continue after a failed case and mark the aggregate contract as failed. Tiny-step ablation deltas are conservative diagnostics, not model-quality conclusions.

M9.2 hardens interpretation. Stability reports now separate `total_loss` from `L_primary`, where `L_primary` is the weighted C/D/E objective before structural auxiliary terms. Ablation summaries classify differences as metric improvement, auxiliary removal, support-mode confounding, small structural effect, unstable/failed, or inconclusive. Seed variance is reported per case; single-seed tiers explicitly mark variance as not estimated.

M9.3 turns the report into an engineering readiness gate for the M10 minimal experiment harness. `blocked` means a case failed, produced nonfinite loss or gradients, had zero gradients, or triggered exploding-loss diagnostics. `conditionally_ready` means the smoke contract passed but broader short/analysis evidence is confounded, auxiliary-only, tiny-effect, or not yet run. `ready` is reserved for required cases passing without numerical blockers or ambiguous ablation categories. This is not a scientific performance claim.

## M10 Minimal Experiment Harness

`src/training/minimal_experiment.py` and `scripts/analysis/run_minimal_experiment.py` orchestrate the existing stability harness over small tiers:

- `micro`: seed `[7]`, `baseline_all`, 3 steps, toy Avalanche.
- `smoke`: seed `[7]`, baseline/top-k/structural-bias-off/budget-auxiliary-off, toy Avalanche.
- `short`: seeds `[7, 13]`, the smoke cases plus the short small-realistic case through the existing stability tier.

The output is written to `.agent/tmp/minimal_experiment.json` and `.agent/tmp/minimal_experiment.md` by default. Summary-only mode reads an existing JSON report and does not rerun cases. M10 preserves M9 readiness and interpretation categories; it does not add a new optimizer path or broad experiment-management system.

M10.1 makes the tier plan explicit in `configs/minimal_experiment.yaml`. The report now includes a tier matrix, `minimal_experiment_contract_ok`, `ready_for_next_stage`, `next_recommended_stage`, `next_stage_reasons`, and a compact `decision_summary`. The old M9 `ready_for_m10_minimal_experiment_harness` field remains as a compatibility alias only. Passing M10 means the project can proceed to a controlled ablation runner such as M11; it is not production training readiness.

## M11 Controlled Ablation Harness

`src/training/controlled_ablation.py` and `scripts/analysis/run_controlled_ablation.py` add a manifest-driven ablation runner over the same tiny smoke primitive. The manifest is `configs/controlled_ablation.yaml`.

M11 tiers are:

- `micro`: seed `[7]`, baseline all-candidate support and structural-bias-off, toy Avalanche. This is the only agent-check safe tier.
- `smoke`: seed `[7]`, baseline, top-k, full structural-bias-off, sector-bias-off, role-bias-off, bridge-bias-off, and budget-auxiliary-off.
- `short`: seeds `[7, 13]`, the smoke cases plus a short small-realistic case.

The runner modifies smoke configuration only; it does not introduce a new training path. It records per-case C/F/D/E, primary loss, auxiliary loss, active support, gradient coverage, per-head gradients, status/error fields, and comparisons against `baseline_all` within each seed. Interpretation categories are conservative: support-change-confounded, auxiliary-removed-only, structural-effect-small, structural-effect-observed, inconclusive, or failed-case. A failed case is recorded and does not abort the remaining cases, but it blocks the controlled ablation contract.

## M12 Ablation Findings Reports

`src/training/ablation_report.py` and `scripts/analysis/report_controlled_ablation_findings.py` turn controlled ablation JSON outputs into selected follow-up recommendations. The default path reads `.agent/tmp/controlled_ablation.json` and writes `.agent/tmp/controlled_ablation_findings.json` plus `.agent/tmp/controlled_ablation_findings.md`; it does not run new cases unless `--run-tier` is passed.

M12 is a reporting and decision layer only. It does not add production training, datasets, checkpoints, schedulers, model selection, large experiment management, or new model code. Its job is to make the M11 interpretation categories actionable while preserving conservative claims.

Recommendation categories are:

- `candidate_for_deeper_study`
- `needs_multi_seed_validation`
- `needs_longer_run`
- `needs_controlled_support_match`
- `ignore_for_now`
- `blocked_due_to_failure`

Top-k comparisons remain support-confounded unless support size and gradient coverage are controlled. Budget auxiliary removal is reported as auxiliary-only when total loss changes without primary C/F/D/E improvement. Single-seed findings are routed to multi-seed validation instead of being treated as stable effects.

## M13 Selected Ablation Evidence

`scripts/analysis/run_selected_ablation_evidence.py` is a small evidence-collection wrapper over M11 controlled ablations and M12 findings. It does not duplicate training logic. It runs a selected preset, embeds the source controlled ablation report, embeds the M12 findings report, and writes `.agent/tmp/selected_ablation_evidence.json` plus `.agent/tmp/selected_ablation_evidence.md`.

M13 presets:

- `smoke`: seed `[7]`, baseline all-candidate support and structural-bias-off, toy Avalanche.
- `short`: seeds `[7, 13]`, baseline plus structural/sector/role/bridge/budget/top-k ablations, toy Avalanche.

The selected report filters M12 follow-up candidates so support-confounded top-k cases and auxiliary-only cases are reported but not promoted as structural-head follow-up evidence. If all structural effects remain negligible, single-seed-only, or confounded, the report sets `recommendation_status` to `no_structural_claim_yet` and recommends increasing steps or refining structural heads.

This is still not a production trainer, data pipeline, checkpointing workflow, scheduler, broad experiment suite, or model-selection layer.

M13.1 makes the `selected_evidence` section of `configs/controlled_ablation.yaml` the source of truth for the smoke and short selected evidence presets. Built-in defaults exist only as fallback when that section is absent. Selected reports expose `no_recommendation_reasons`, `evidence_strength`, `selected_evidence_readiness`, and optional short-vs-smoke comparison fields. `selected-ablation-smoke-check` is small enough for normal harness checks; `selected-ablation-short-check` is explicit because it runs more seeds and cases.

## M14 Structural Signal Audit

`src/models/structural_signal_audit.py` and `scripts/analysis/audit_structural_heads.py` measure whether sector, role, bridge, and budget structural heads are visible before scaling training. The audit is diagnostic only. It does not add a production trainer, dataset loader, checkpointing path, scheduler, model-selection system, or broad experiment suite.

M14 reports score contribution ratios, row-level ranking influence, top-k support overlap, gradient allocation across base encoders and structural heads, and a micro-ablation over structural-zero, sector-only, role-only, bridge-only, and all-structural score bias variants. Classifications are conservative: structural signal too weak, gradient starved, not metric-relevant under the tiny smoke graph, auxiliary/support confounded, or observed. Observed means visible, not beneficial.

M14.1 adds direction-aware interpretation. The audit now reports all-structural versus structural-zero deltas for primary C/D/E loss, total loss, auxiliary loss, C, F, D, and E. It classifies structural effect direction as beneficial, harmful, neutral, mixed, or unknown and reports top-k support changes as confounders. It also splits gradient allocation into main C/D/E loss, auxiliary structural loss, and combined loss so budget auxiliary gradients cannot be mistaken for main objective signal. Per-head status fields surface sector, role, bridge, and budget heads as active, harmful, neutral, gradient-starved, auxiliary-only, or missing. Recommended actions now include calibrating or disabling harmful structural weights, splitting auxiliary/main gradients, revising weak role features, or running selected follow-up only after primary metrics improve.

M14.2 adds a structural weight calibration sweep before longer selected runs. `scripts/analysis/sweep_structural_weights.py` evaluates small deterministic combinations of sector, role, and bridge score-bias weights under both `support_mode="all"` and `support_mode="topk"`. It reports primary-loss, C/F/D/E, ranking, support-change, and structural-to-base-ratio deltas against structural-zero. The conservative default policy is: do not assume structural heads are useful, disable structural score bias in top-k unless a nonzero setting leaves primary loss, failure probability, C, and selected support materially unchanged, and treat budget heads as auxiliary/diagnostic until a differentiable budget surrogate exists.

M15 adds targeted selected runs for that calibrated policy. `scripts/analysis/run_calibrated_structural_followup.py` tests all-zero all-mode, low-weight all-mode, zero-weight top-k, an unsafe high-weight top-k reference, and an all-to-top-k calibrated curriculum. The recommended policy is still conservative: all-mode may test `0.01/0.01/0.01` structural weights only if C/F/D/E are non-worse across seeds, while top-k remains zero-structural by default. The high-weight top-k case is an unsafe reference and is never promoted.

## M16 Selected Training Smoke

`scripts/analysis/run_selected_training_smoke.py` is a selected policy smoke,
not a production trainer. It requires short-tier calibrated structural
readiness from `.agent/tmp/calibrated_structural_followup_short.json` by
default. Smoke-tier calibrated artifacts are rejected because they validate only
the code path, not multi-seed readiness.

The selected policy has two phases over the existing tiny training primitive:

- phase 0: `support_mode="all"` with structural weights
  `sector=0.01`, `role=0.01`, `bridge=0.01`;
- phase 1: `support_mode="topk"` with all structural score-bias weights set to
  zero, carrying the phase 0 model parameters forward.

The M16 report records finite-loss and finite-gradient flags, parameter
movement, gradient coverage, C/F/D/E metrics, the readiness source artifact, and
whether top-k structural bias was disabled. It does not add datasets,
checkpoints, schedulers, production experiment management, or model selection.

## M17 Selected Training Interpretation

`scripts/analysis/report_selected_training_smoke.py` is a report-only decision
gate over `.agent/tmp/selected_training_smoke.json`. It computes phase loss
deltas, total selected-policy loss movement, phase-switch loss movement,
gradient-coverage drop, available C/F/D/E trend deltas, and the M16 contract
flags.

M17 blocks when the M16 report is not based on short-tier readiness, selected
training was not allowed, losses or gradients were nonfinite, parameters did not
move, the selected-training contract failed, top-k structural bias was not
disabled, or the high-weight top-k reference was used. It may recommend only
M18 selected short training smoke. It is not a production training or
model-quality gate.

M16.1 extends the selected training smoke report with per-phase initial, final,
and delta snapshots for C, F, D, and E, plus tail metrics when available. M17
uses those trend fields to remove the previous `missing_metric_trends` caution
and make a conservative readiness decision from primary metric movement.

M16.2 repeats the same selected training smoke over deterministic seeds such as
`[7, 13, 23]`. It aggregates phase C/F/D/E/loss deltas, sign consistency,
finite-loss and finite-gradient fractions, parameter movement, and top-k policy
flags. It can authorize only M18 selected short training when every seed is
healthy, the readiness artifact is short-tier, and C or F improves consistently
across both phases. It is still smoke-scale evidence, not selected short
training or model-quality proof.

## M18 Selected Short Training

`scripts/analysis/run_selected_short_training.py` runs a slightly longer
selected short harness using the calibrated M16 policy. The default readiness
source is `.agent/tmp/selected_training_multiseed.json`, and it must report
`ready_for_M18_selected_short_training=true` unless the run is explicitly
forced and marked as such.

M18 keeps the same two phase policy:

- phase 0: all-candidate support with structural weights
  `sector=0.01`, `role=0.01`, `bridge=0.01`;
- phase 1: top-k support with structural weights all zero, carrying phase 0
  model parameters forward.

Reports aggregate C/F/D/E and loss deltas across the selected seeds, finite
loss and gradient fractions, parameter movement, gradient coverage, and whether
effects exceed the previous M16 tiny-effect threshold. M18 is still a harness;
it does not add datasets, checkpointing, schedulers, production experiment
management, or model selection.

## M19 Selected Short Interpretation

`scripts/analysis/report_selected_short_training.py` is a report-only gate over
`.agent/tmp/selected_short_training.json`. It classifies M18 as blocked,
policy-revision-worthy, mixed/inconclusive, promising but short-horizon, or a
confirmed readiness signal for controlled selected-short ablation.

M19 may recommend only `M20_controlled_selected_short_ablation`. It does not
claim model quality, structural-head superiority, production-training readiness,
or large-scale validation. The allowed claim is limited to numerical health and
directionally consistent C/F/D trends across the selected short seeds while
top-k structural bias remains disabled.

## M20 Controlled Selected-Short Ablation

`scripts/analysis/run_controlled_selected_short.py` compares the calibrated
selected policy against conservative controls inside the same selected-short
harness. The core matrix is `calibrated_policy`, `zero_structural_policy`, and
`high_weight_topk_unsafe_reference`. The high-weight top-k case is diagnostic
only and is never promoted.

M20 is still not a trainer. It does not add datasets, checkpoints, schedulers,
large-scale experiment management, model selection, PBFT, sampling, dense
all-pairs graph construction, or direct link/SINR/BLER/HARQ/coverage losses.
The only possible promotion is a recommendation for deeper selected study when
primary C/F/D/E changes clear the conservative claim gates.

## M20.1 Support-Matched Audit

`scripts/analysis/run_support_matched_audit.py` is an audit layer over the
selected short top-k cases. It compares zero-structural top-k, calibrated
low-weight top-k, high-weight top-k, an analysis-only support-matched proxy,
and all-mode low-weight support. The support-matched proxy compares selected
sparse candidate-index overlap and evaluator metric deltas; it does not add a
fixed-support trainer or claim fixed-support training benefit.

M20.1 keeps high-weight top-k structural bias disabled by policy. If favorable
high-weight behavior is support-driven, the next step is a controlled
fixed-support design study, not promotion of the high-weight policy. If
calibrated low top-k remains negligible, top-k structural bias remains disabled.

## M20.2 Fixed-Support Top-k Audit

`scripts/analysis/run_fixed_support_audit.py` performs that design study as an
analysis proxy. It captures sparse `selected_candidate_index` support from a
source top-k policy, then recomputes row-normalized weights on exactly those
candidate edges under zero, low, or high structural score weights.

This is not a constructor mode and not a deployment rule. It separates
support-set selection effects from score redistribution effects within the same
fixed support. Every fixed-support case is marked `analysis_proxy=true`,
`claim_allowed=false`, and blocked from production claims. High-weight top-k
structural bias remains unsafe and disabled by policy.

## M21 Structural Redesign Review

`scripts/analysis/report_structural_redesign_review.py` is a report-only review
over M14-M20 structural-head artifacts. It does not run training, create
datasets, save checkpoints, change the GNN scorer, or change the topology
constructor.

M21 summarizes the structural signal audit, structural-weight sweep,
support-matched audit, fixed-support audit, calibrated structural follow-up,
controlled selected-short ablation, selected short training, and controlled
ablation reports when present. It classifies sector, role, bridge, and budget
heads separately for signal visibility, main-loss gradient presence, auxiliary
gradient presence, top-k safety, all-mode safety, confirmed benefit, observed
harm, support confounding, and recommended status.

The conservative default from the current evidence is:

- top-k structural score bias remains disabled by default;
- all-mode low structural weights are zero-or-low diagnostic settings, not a
  model-quality claim;
- budget heads are auxiliary-only until a differentiable budget surrogate is
  introduced;
- sector and bridge heads require redesign before promotion;
- role remains diagnostic or redesign-bound unless future main-loss evidence
  supports it.

M21 may mark the project ready for M22 baseline consolidation or structural
redesign planning. That readiness is an engineering planning gate, not
production training readiness or a structural-head benefit claim.

## M22 Baseline Consolidation

`scripts/analysis/run_baseline_consolidation.py` freezes the pure edge-score
baseline policy recommended by M21. It is report-only: it reads
`.agent/tmp/structural_redesign_review.json`, requires `ready_for_M22=true` and
`M22_recommended_scope="baseline_consolidation"` unless forced, and writes
`.agent/tmp/baseline_consolidation.json` plus `.md`.

The consolidated defaults are:

- `edge_score_all_baseline` for warm-start/training smoke: all sparse
  candidates are kept and structural score-bias weights are zero.
- `edge_score_topk_baseline` for deployable capped support: top-k structural
  score bias is disabled and structural weights are zero.

Structural heads may still be emitted for diagnostics, but they are not fed
into the default top-k score path. Sector and bridge remain redesign-required,
role remains diagnostic-only, and the budget head remains auxiliary or
future-surrogate-only. M22 does not add training logic, datasets, checkpoints,
schedulers, constructor behavior, or model-quality claims.

## M23 Pure Edge Baseline Stability

`scripts/analysis/run_pure_edge_baseline_stability.py` runs a small stability
harness over the M22 pure edge-score policies. It requires
`.agent/tmp/baseline_consolidation.json` with
`baseline_consolidation_contract_ok=true` unless explicitly forced.

The harness checks `edge_score_all_baseline`, `edge_score_topk_baseline`, and an
`all -> topk` curriculum that carries model parameters from all-mode phase 0 to
top-k phase 1. Sector, role, and bridge score-bias weights are exactly zero in
every pure baseline case. Structural heads may still be produced as diagnostics,
but they are not allowed to alter `edge_score`.

M23 reports finite loss/gradient health, parameter changes, C/F/D/E trends,
gradient coverage, active edge counts, and whether structural bias could affect
scores. It does not create a trainer, dataset loader, checkpoint, scheduler, or
structural-head benefit claim.

## M24 Pure Edge Baseline Interpretation

`scripts/analysis/report_pure_edge_baseline_stability.py` is a report-only
interpretation layer over `.agent/tmp/pure_edge_baseline_stability.json`. It
does not rerun training or change model, loss, or constructor behavior.

M24 classifies the all-mode baseline as a warm-start/training path, the top-k
baseline as the deployment candidate, and the all-to-topk curriculum as a
support-change-confounded transition. A large curriculum improvement is treated
as deployment-transition health only, because support mode and active-edge count
change at the phase boundary.

The report can set `ready_for_M25_pure_edge_selected_short=true` only when the
M23 contract is healthy, all structural score-bias weights are zero, top-k
structural bias is disabled, losses and gradients are finite, parameters move,
and no structural bias affects `edge_score`.

## M25 Pure Edge Selected Short

`scripts/analysis/run_pure_edge_selected_short.py` runs the authorized
selected-short pure edge-score policies. It requires the M24 interpretation
artifact by default and writes `.agent/tmp/pure_edge_selected_short.json` plus
`.md`.

The case matrix is pure edge-score only: all-mode zero structural weights,
top-k zero structural weights, and an all-to-topk transition that carries model
parameters from the all-mode phase into the top-k phase. Structural heads may be
computed as diagnostics, but sector, role, and bridge score-bias weights are
zero in every selected-short policy and must not affect `edge_score`.

M25 reports C/F/D/E and loss deltas, finite loss and gradient fractions,
parameter movement, gradient coverage, and active edge counts. Top-k and
all-mode results remain support-rule dependent, and the all-to-topk transition
is still support-confounded. M25 is not a production trainer and does not make
structural-head or model-quality claims.

## M26 Pure Edge Selected Short Interpretation

`scripts/analysis/report_pure_edge_selected_short.py` is a report-only
interpretation layer over `.agent/tmp/pure_edge_selected_short.json`. It does
not rerun training or change model, loss, or constructor behavior.

M26 classifies the all-mode selected-short case as a warm-start path, the top-k
case as a deployment candidate, and the all-to-topk case as a
support-change-confounded transition. Even when the transition has large C/F/D
movement, the report treats it as numerical transition health because active
support and gradient coverage change between phases.

The report can set `ready_for_M27_pure_edge_followup=true` only when the M25
contract is healthy, structural score-bias weights are zero, top-k structural
bias is disabled, losses and gradients are finite, parameters move, and no
structural bias affects `edge_score`.

## M27 Pure Edge Follow-Up

`scripts/analysis/run_pure_edge_followup.py` runs the M26-authorized pure
edge-score follow-up harness. It reads
`.agent/tmp/pure_edge_selected_short_interpretation.json` by default and
requires `ready_for_M27_pure_edge_followup=true` unless explicitly forced.

The case matrix remains pure edge-score only: all-mode warm-start, top-k
deployment candidate, and an all-to-topk deployment transition that carries
model parameters into the top-k phase. Sector, role, and bridge score-bias
weights are zero in every case, so structural heads may exist only as
diagnostics and must not affect `edge_score`.

M27 separates health claims from quality claims. A healthy all-mode, top-k, or
all-to-topk case may support numerical-path health, but `quality_claim_allowed`
remains false. The all-to-topk transition is still support-rule-confounded when
active support or gradient coverage changes, even if C/F/D/loss movement is
large.

## M28 Pure Edge Reference Package

`scripts/analysis/build_pure_edge_reference_package.py` is a report-only
consolidation layer over the M22-M27 pure edge-score artifacts. It reads
baseline consolidation, baseline stability, baseline interpretation,
selected-short, selected-short interpretation, and follow-up JSON files, then
writes `.agent/tmp/pure_edge_reference_package.json` plus `.md`.

The package policy is defined in `configs/pure_edge_baseline.yaml`. The
training/warm-start policy is `edge_score_all_baseline` with all structural
score-bias weights zero. The deployment candidate is `edge_score_topk_baseline`
with top-k structural bias disabled. The all-to-topk transition is kept as a
support-change-confounded health check.

M28 can set `ready_for_M29=true` only when every required artifact exists, the
M22-M27 contracts are healthy, structural weights remain zero, top-k structural
bias is disabled, and no source is blocked. It does not run training, redesign
structural heads, or make model-quality claims.

## M29 Pure Edge Baseline Extension

`scripts/analysis/run_pure_edge_baseline_extension.py` runs the M28-authorized
pure edge-score baseline extension. It reads
`.agent/tmp/pure_edge_reference_package.json` by default and requires
`ready_for_M29=true` unless explicitly forced.

The extension keeps the same frozen policy: all-mode warm-start, top-k
deployment candidate, and an all-to-topk transition with sector, role, and
bridge score-bias weights all zero. The node count, seed list, step count, and
Avalanche profile can be configured, but the harness remains small-scale and
diagnostic. Health claims remain separate from quality claims; the all-to-topk
transition is still support-change-confounded.

## M30 Pure Edge Baseline Extension Interpretation

`scripts/analysis/report_pure_edge_baseline_extension.py` is a report-only
interpretation layer over `.agent/tmp/pure_edge_baseline_extension.json`. It
does not rerun training or change model, loss, or constructor behavior.

The report classifies the M29 all-mode extension as a warm-start health path,
the top-k extension as a deployment-candidate health path, and the all-to-topk
extension as a support-change-confounded transition when active support or
gradient coverage changes. `ready_for_M31_pure_edge_benchmark_package=true`
only means the pure edge-score baseline can be packaged as a controlled
benchmark reference; it is not a model-quality, structural-head, production
training, or large-scale deployment claim.

## M31 Pure Edge Benchmark Package

`scripts/analysis/build_pure_edge_benchmark_package.py` packages the M28
reference package, M29 extension, and M30 extension interpretation into the
canonical pure edge-score benchmark reference. It reads
`configs/pure_edge_benchmark.yaml` and writes
`.agent/tmp/pure_edge_benchmark_package.json` plus `.md`.

M31 is report-only. It records the baseline policy, artifact availability,
claim ledger, and future comparison contract. Future models must report the
same C/F/D/E metrics, active edge counts, gradient coverage, support mode,
structural-bias policy, direct-physics-loss absence, and support-change
confounders. `ready_for_M32_formal_training_config=true` is only a
formal-configuration gate; it does not validate production training,
large-scale deployment, top-k superiority, or structural-head benefit.

## M32 Formal Training v0 Config Freeze

`configs/training_v0.yaml` freezes the formal v0 training configuration that
follows the M31 pure edge-score benchmark package. The validator
`scripts/analysis/validate_training_v0_config.py` is report-only and writes
`.agent/tmp/training_v0_config_validation.json` plus `.md`.

M32 fixes phase 0 as an all-mode warm-start and phase 1 as a top-k candidate
with parameters carried from phase 0. Structural score bias is disabled,
sector/role/bridge weights are zero, structural heads are not trainable, and
the budget head is diagnostic or disabled. `ready_for_M33_formal_training_v0_smoke`
only authorizes a formal smoke harness; it does not introduce production
training, checkpoints, dataset loaders, schedulers, model selection, or any
model-quality claim.

## M33 Formal Training v0 Smoke

`scripts/analysis/run_formal_training_v0_smoke.py` executes the frozen M32
configuration in a small two-phase smoke and writes
`.agent/tmp/formal_training_v0_smoke.json` plus `.md`. It requires
`.agent/tmp/training_v0_config_validation.json` with
`ready_for_M33_formal_training_v0_smoke=true` unless explicitly forced.

The harness reuses the existing tiny optimizer primitive. Phase 0 uses
`support_mode="all"`, phase 1 uses `support_mode="topk"`, and model parameters
are carried from phase 0 to phase 1. Structural score bias remains disabled,
sector/role/bridge weights stay zero, and top-k structural bias is disabled.
M33 reports C/F/D/E trends, loss components, active edges, gradient coverage,
runtime, and runtime environment details. Passing M33 authorizes only M34
baseline smoke work; it is not production training or model-quality evidence.

## M34 Formal Training v0 Baseline Run

`scripts/analysis/run_formal_training_v0_baseline.py` runs the same frozen
formal v0 policy authorized by M33, using configured phase step counts unless
a controlled `--max-steps-override` is supplied. The default readiness source
is `.agent/tmp/formal_training_v0_smoke.json`, which must report
`formal_training_v0_smoke_status=smoke_passed` unless explicitly forced.

M34 keeps the pure edge-score architecture unchanged: phase 0 is all-mode,
phase 1 is top-k with parameters carried from phase 0, structural score bias is
disabled, sector/role/bridge weights are zero, and top-k structural bias is
disabled. The report adds per-step scalar trends for loss, L_R/L_D/L_E,
C/F/D/E, active edge count, gradient coverage, and parameter-gradient norm. It
does not add production training, checkpointing, dataset loading, schedulers,
model selection, structural-head redesign, or model-quality claims.

## M35 Formal Training v0 Baseline Interpretation

`scripts/analysis/report_formal_training_v0_baseline.py` is a report-only
interpretation layer over `.agent/tmp/formal_training_v0_baseline.json`. It
compares the M34 step counts with the frozen `configs/training_v0.yaml`
configuration, classifies phase 0 and phase 1 C/F/D/E health, and writes
`.agent/tmp/formal_training_v0_baseline_interpretation.json` plus `.md`.

M35 can authorize only `M36_full_formal_training_v0_baseline_run`. A short
controlled step override is recorded as a caution rather than a blocker when
all numerical-health gates pass. M35 still makes no production-training,
checkpointing, scheduler, dataset, model-selection, top-k-superiority,
structural-head-benefit, or model-quality claim.

## M36 Full Formal Training v0 Baseline Run

`scripts/analysis/run_formal_training_v0_full.py` executes the M35-authorized
formal v0 baseline using the configured `configs/training_v0.yaml` phase step
counts by default: phase 0 all-mode warm-start and phase 1 top-k candidate,
with parameters carried from phase 0 to phase 1. The output path is
`.agent/tmp/formal_training_v0_full.json` plus `.md`.

M36 keeps the pure edge-score baseline frozen. Structural score bias is
disabled, sector/role/bridge weights are zero, top-k structural bias is
disabled, and direct physics losses remain forbidden. The harness records
compact per-step scalar curves for loss, L_R/L_D/L_E, C/F/D/E, active edge
count, gradient coverage, and parameter-gradient norm. A
`--max-steps-override` run is explicitly marked as an override and does not
claim the configured 50+50 run was completed. M36 is still not production
training, checkpointing, scheduling, dataset loading, model selection, or a
model-quality claim.

## M37 Formal Training v0 Full Interpretation

`scripts/analysis/report_formal_training_v0_full.py` is a report-only layer
over `.agent/tmp/formal_training_v0_full.json`. It does not rerun training or
change model, loss, or constructor behavior.

M37 interprets phase 0 and phase 1 separately because phase 0 uses all sparse
candidate support and phase 1 uses top-k support. It checks the full 50+50
step counts, finite losses and gradients, parameter movement, seed consistency,
disabled structural score bias, disabled top-k structural bias, and the graph
rule contract before allowing the small-realistic follow-up. The phase switch
remains support-rule-confounded and is not a top-k superiority claim.

## M38 Small-Realistic Formal Training v0 Baseline Run

`scripts/analysis/run_formal_training_v0_small_realistic.py` executes the same
frozen formal v0 baseline under the `small_realistic` Avalanche profile. The
default readiness source is `.agent/tmp/formal_training_v0_full_interpretation.json`,
which must report `ready_for_M38_small_realistic_formal_baseline=true` unless
the run is explicitly forced.

M38 keeps the architecture unchanged: phase 0 is all-mode warm-start, phase 1
is top-k with parameters carried from phase 0, structural score bias is
disabled, sector/role/bridge weights are zero, and top-k structural bias is
disabled. A `--max-steps-override` run is recorded as an override and cannot
claim the configured 50+50 small-realistic run. M38 is still not production
training, checkpointing, scheduling, dataset loading, model selection, or a
model-quality claim.

## M39 Small-Realistic Formal Training v0 Interpretation

`scripts/analysis/report_formal_training_v0_small_realistic.py` is a
report-only layer over `.agent/tmp/formal_training_v0_small_realistic.json`.
It does not rerun training or change the model, loss, or constructor.

M39 checks that the M38 source used the `small_realistic` profile without a
step override, completed the configured 50+50 phases, kept structural score
bias disabled, kept top-k structural bias disabled, and maintained finite
losses, finite gradients, parameter movement, and non-worse C/F/D/E trends
across seeds. Large phase 0 movement is interpreted as trainability evidence,
not final model quality. `ready_for_M40_formal_training_v0_baseline_package`
is only a packaging gate for baseline evidence consolidation.

## M40 Formal Training v0 Baseline Package

`scripts/analysis/build_formal_training_v0_baseline_package.py` consolidates
the M36/M37 toy-profile full baseline and the M38/M39 small-realistic full
baseline into one formal v0 baseline evidence package. It is report-only and
does not rerun training, load datasets, create checkpoints, add schedulers,
select models, redesign structural heads, or change topology construction.

The package records the frozen pure edge-score policy: phase 0 uses all-mode
support, phase 1 uses top-k support with parameters carried from phase 0, all
sector/role/bridge score-bias weights remain zero, and top-k structural bias is
disabled. It also records the conservative claim ledger and future comparison
requirements for C/F/D/E trends, L_R/L_D/L_E trends, active edge counts,
gradient coverage, support mode, structural-bias policy, profile, seed count,
direct-physics-loss absence, graph-rule consistency, and support-change
confounders.

`ready_for_M41_controlled_v0_baseline_extension=true` is a controlled-extension
gate only. It does not prove model quality, top-k superiority, production
training readiness, realistic deployment behavior, or structural-head benefit.

## M41 Controlled v0 Baseline Extension

`scripts/analysis/run_controlled_v0_baseline_extension.py` runs the M40-authorized
controlled extension over larger node counts, starting with `node_count=500`.
The case matrix is still pure edge-score only: all-mode, top-k, and
all-to-topk with structural score bias disabled and sector/role/bridge weights
fixed at zero.

M41 reports C/F/D/E and loss deltas, L_R/L_D/L_E finals when available, active
edge counts, gradient coverage, runtime, and support-change confounders for
toy and small-realistic profiles. It is not production training, checkpointing,
dataset loading, scheduling, model selection, structural-head redesign, or
large-scale deployment validation.

## M42 Controlled v0 Baseline Extension Interpretation

`scripts/analysis/report_controlled_v0_baseline_extension.py` is a report-only
layer over `.agent/tmp/controlled_v0_baseline_extension.json`. It interprets
the N=500 toy and small-realistic controlled extension without rerunning
training or changing the pure edge-score constructor.

M42 keeps the case semantics conservative: all-mode can be a larger-node
warm-start health path, top-k can be a larger-node deployment-candidate health
path, and all-to-topk remains support-change-confounded whenever support
changes. The report can authorize only `M43_v0_baseline_package_v1_1`; it does
not claim top-k superiority, 10k behavior, realistic deployment validity,
production readiness, or structural-head quality improvement.

## M43 Formal Training v0 Baseline Package v1.1

`scripts/analysis/build_formal_training_v0_baseline_package_v1_1.py` is a
report-only package builder over the M40 package and the M41/M42 controlled
extension artifacts. It records artifact availability, package identity,
evidence summaries, a N=100/N=500 coverage matrix, a conservative claim
ledger, and readiness for `M44_scale_or_profile_extension_planning`.

The package keeps the architecture boundary unchanged: pure edge-score
topology construction, structural bias disabled, structural weights fixed at
zero, top-k structural bias disabled, no dense NxN path, and no direct
link/SINR/BLER/HARQ/coverage loss. A complete v1.1 package is a planning gate
for additional scale/profile diagnostics, not a deployment or production
training validation.

## M44 Scale/Profile Extension Plan

`scripts/analysis/report_scale_profile_extension_plan.py` is a report-only
planner over `.agent/tmp/formal_training_v0_baseline_package_v1_1.json`. It
requires a complete M43 v1.1 package with `ready_for_M44_scale_or_profile_extension=true`,
disabled structural bias, disabled top-k structural bias, and blocked quality
claims before selecting the next diagnostic.

M44 ranks the v1.1 options and defaults to
`small_realistic_topk_flatness_diagnostic` because the N=500 small-realistic
top-k case is numerically healthy but near-flat. Runtime/memory scaling is
kept as the next engineering diagnostic, and larger toy top-k scale is deferred
until the profile-specific flatness ambiguity is explained. The plan does not
run training or authorize structural-head reintroduction, dense NxN paths,
direct physics losses, 10k validation, or deployment readiness claims.

## M45 Small-Realistic Top-k Flatness Diagnostic

`scripts/analysis/run_topk_flatness_diagnostic.py` runs the M44-selected
diagnostic for the N=500 `small_realistic` top-k pure edge-score case. It
reuses the existing formal v0 smoke phase primitive and reports reliability
saturation, loss-component movement, support rigidity, gradient signal, metric
flatness, and optional more-steps probe context.

M45 keeps the architecture boundary unchanged: no production trainer, dataset
loader, checkpointing, scheduler, model selection, structural-head redesign,
PBFT objective, sampling path, dense NxN graph construction, or direct
link/SINR/BLER/HARQ/coverage loss is added. Its output can diagnose why top-k
metrics are near-flat, but it cannot prove model quality, top-k superiority,
10k behavior, production readiness, or deployment validity.

## M46 Runtime/Memory Scaling Diagnostic

`scripts/analysis/run_runtime_memory_scaling.py` runs a controlled sparse
runtime and tensor-memory diagnostic for the pure edge-score v0 baseline. The
default smoke stays fast with N=100 and N=500 toy top-k cases; larger counts
such as N=2000, N=5000, and N=10000 require `--large` or explicit
`--node-counts`.

M46 records per-stage timing, deterministic tensor-memory proxies, active and
candidate edge ratios, gradient coverage, and a dense `N*N` proxy count for
comparison only. It does not allocate dense graph tensors and does not make
model-quality, deployment, production-training, or top-k superiority claims.
Structural bias remains disabled, structural weights remain zero, and top-k
structural bias remains disabled.

## M47 Node Count 2000 Toy Top-k Smoke

`scripts/analysis/run_node_count_2000_topk_smoke.py` runs the next controlled
pure edge-score scale point authorized by M46: N=2000, toy profile, top-k
support. It keeps structural score bias disabled, sector/role/bridge weights
fixed at zero, and top-k structural bias disabled.

M47 combines the per-stage sparse runtime/memory breakdown with a tiny
optimizer smoke for finite-loss, finite-gradient, parameter-movement, and
C/F/D/E delta context. It is still a controlled engineering-health diagnostic,
not production training, not deployment validation, not a top-k superiority
claim, and not evidence that N=2000 validates 10k behavior.

## M48 Runtime/Memory Scaling Interpretation

`scripts/analysis/report_runtime_memory_scaling.py` is a report-only
interpretation over the M46 runtime/memory scaling artifact and the M47 N=2000
toy top-k smoke artifact. It normalizes N=100, N=500, and N=2000 toy/top-k
rows into one sparse scaling table, computes edge, memory, and step-time
growth ratios, and records the largest-node bottleneck stage.

M48 does not run training or change topology construction. It can report that
the observed sparse path is healthy, healthy with a constructor/evaluator
bottleneck, requires a bottleneck diagnostic, or is blocked by source-contract,
structural-bias, dense-path, or nonfinite-loss/gradient evidence. A constructor
bottleneck at N=2000 recommends `M49_constructor_bottleneck_diagnostic` before
larger smoke runs; no interpretation authorizes 10k, production, deployment, or
structural-head validation claims.

## M49 Constructor Bottleneck Diagnostic

`scripts/analysis/profile_topology_constructor.py` profiles the existing
`TopologyConstructionLayer` internals over deterministic synthetic sparse
candidate graphs. It requires the M48 interpretation to recommend
`M49_constructor_bottleneck_diagnostic` unless forced, and writes
`.agent/tmp/constructor_bottleneck_diagnostic.json` plus `.md`.

M49 keeps the architecture unchanged. It times validation, source grouping,
top-k support selection, row softmax, diagnostics, and output packaging without
replacing `TopologyConstructionLayer.forward()`. It can recommend segmented
top-k optimization, diagnostics trimming, validation optimization, row-loop
vectorization, or continuing to a controlled 5k smoke. The diagnostic remains
engineering-only: no production trainer, dataset loader, new GNN architecture,
structural-head reintroduction, dense graph path, direct physics objective, or
deployment/10k validation claim is added.

## M50 Constructor Optimization Design

`scripts/analysis/report_constructor_optimization_design.py` is a report-only
design layer over `.agent/tmp/constructor_bottleneck_diagnostic.json`. It reads
the M49 timing decomposition, summarizes current `TopologyConstructionLayer`
top-k risks, lists safe optimization options, and writes
`.agent/tmp/constructor_optimization_design.json` plus `.md`.

M50 does not change the constructor implementation. It can recommend a future
segmented top-k fast path only when the source diagnosis shows top-k selection
dominates the N=2000 constructor time and M49 recommends
`optimize_segmented_topk`. The correctness contract for that future work must
preserve the same train/validation/deployment forward rule, deterministic
sparse top-k membership, duplicate/self-loop rejection, row normalization,
`selected_candidate_index` mapping, diagnostics, selected-support gradients,
and no-gradient behavior for excluded top-k candidates. It must not introduce
stochastic sampling, dense NxN graph tensors, structural-head reintroduction,
or direct hidden-physics objectives.

## M50.1 Segmented Top-k Fast Path

`TopologyConstructionLayer` now accepts `topk_backend="legacy"` or
`topk_backend="segmented_fast"`. The default remains `legacy`; the fast backend
is an implementation optimization for the existing top-k support rule, not a
new topology rule. `support_mode="all"` is unaffected by the backend choice.

The segmented fast path still consumes only sparse candidate edges. It keeps
the same validation, duplicate/self-loop rejection, per-node cap, selected
candidate mapping, row-softmax normalization, diagnostics, and
train/validation/deployment semantics. Backend comparison profiling with
`--topk-backend both` records legacy-vs-fast selected-index, topology-weight,
diagnostics, and gradient-integration checks before allowing the next
controlled 5k smoke.

## M51 Node Count 5000 Toy Top-k Smoke

`scripts/analysis/run_node_count_5000_topk_smoke.py` runs the controlled M51
scale point authorized by M50.1. The case remains the pure edge-score v0
baseline: `node_count=5000`, `profile=toy`, `support_mode=topk`, and
`topk_backend=segmented_fast`.

M51 keeps structural score bias disabled, keeps sector/role/bridge weights at
zero, keeps top-k structural bias disabled, and preserves the same sparse
train/validation/deployment constructor rule. The report records finite
loss/gradient evidence, parameter movement, C/F/D/E deltas when available,
active and candidate edge counts, sparse-to-dense proxy ratio, per-stage
runtime, bottleneck stage, throughput proxies, deterministic tensor-memory
proxy, and runtime environment fields.

Passing M51 is an engineering-health result for the observed N=5000 toy top-k
case. It does not validate production training, deployment readiness, top-k
superiority, realistic-profile behavior, or N=10000 behavior.

## M52 Runtime/Memory Scaling v1.1 Interpretation

`scripts/analysis/report_runtime_memory_scaling_v1_1.py` is a report-only
layer over the M46/M48 scaling artifacts, M49/M50 constructor artifacts, and
the M51 N=5000 segmented_fast smoke. It writes
`.agent/tmp/runtime_memory_scaling_v1_1_interpretation.json` plus `.md`.

M52 tracks the observed N=100/500/2000/5000 sparse scaling path, including
active and candidate edge growth, sparse-to-dense proxy ratios, tensor-memory
proxy growth, step-time growth, constructor-time growth, evaluator-time
growth, and bottleneck-stage transitions. It can report that the constructor
bottleneck was reduced after segmented_fast and that the evaluator is now the
largest observed bottleneck.

The architecture remains unchanged. M52 does not authorize production
training, deployment readiness, top-k superiority, realistic-profile behavior,
or N=10000 validation claims. Its only next-stage decision is whether the next
controlled diagnostic should be a 10k toy/top-k smoke or an evaluator
bottleneck diagnostic.

## M53 Node Count 10000 Toy Top-k Forward Smoke

`scripts/analysis/run_node_count_10000_topk_forward_smoke.py` runs the next
controlled scale point authorized by M52. The case is forward-only:
`node_count=10000`, `profile=toy`, `support_mode=topk`, and
`topk_backend=segmented_fast`.

M53 keeps the pure edge-score policy, structural score bias disabled, all
sector/role/bridge structural weights at zero, and top-k structural bias
disabled. It runs scorer, sparse topology construction, analytic evaluator,
and coupled loss once, but it does not run backward, take optimizer steps, or
make training claims.

The report records candidate and active edge counts, sparse-to-dense scalar
proxy ratio, finite forward C/F/D/E and loss components, forward-stage timing,
evaluator cost, bottleneck stage, deterministic tensor-memory proxy, and
runtime environment fields. Passing M53 is a forward-only engineering-health
result for the observed toy/top-k scale point; it does not validate production
training, deployment readiness, realistic-profile behavior, or backward
behavior at N=10000.

## M54 Evaluator Bottleneck Diagnostic

`scripts/analysis/profile_v2x_evaluator.py` profiles the evaluator path used
by the pure edge-score toy/top-k scale smokes. It reads the M53 forward smoke
as readiness and runs forward-only evaluator profiling for N=5000 and N=10000
by default.

M54 changes no model, constructor, or loss semantics. It reconstructs the
existing evaluator stages with timing hooks: channel/link proxy, topology
query-support normalization and sparse accumulation, graph-coupled Avalanche
recurrence, energy metric computation, metric extraction, and diagnostic
postprocess. It also reports deterministic evaluator tensor-memory proxies and
reliability state relative to the configured failure target.

The diagnostic can recommend graph-coupled evaluator optimization, query
support optimization, or a later 10k one-step backward smoke. It does not
itself authorize production training, deployment readiness, 10k training, or
realistic-profile claims.

## M55 Query Support Optimization Design

`scripts/analysis/report_query_support_optimization_design.py` is a
report-only design layer over `.agent/tmp/evaluator_bottleneck_diagnostic.json`.
It summarizes the M54 topology query-support bottleneck, records the current
sparse probability contract, lists safe optimization options, and writes
`.agent/tmp/query_support_optimization_design.json` plus `.md`.

M55 does not change query-support math, graph-coupled Avalanche recurrence,
topology construction, loss semantics, or training behavior. It can recommend a
future fused sparse query-support fast path, diagnostics fast path, CSR-style
row-pointer design, or no optimization when evidence is insufficient. Any
future implementation must preserve explicit `p_correct_query` and
`p_wrong_query` semantics, neutral/no-support probabilities, isolated-row
behavior, differentiability through sparse weights and preferences, diagnostics
when enabled, and the no-sampling/no-dense-NxN constraints.

## M55.1 Query Support Fast Path

M55.1 adds an implementation backend inside the evaluator query-support layer.
The architecture still uses the same sparse constructor output and the same
graph-coupled Avalanche evaluator. `query_support_backend="fused_fast"` is a
drop-in sparse backend for `query_support_backend="legacy"`; it is not a new
model, topology rule, loss term, or training path.

`diagnostics_mode` separates probability computation from diagnostic workload.
Full mode is the evidence-preserving mode. Lite/off modes are for controlled
runtime profiling and must be labeled when used. No M55.1 result validates
production training, deployment readiness, top-k quality, or 10k backward by
itself.

## M56 10k Backward Smoke

M56 adds a one-step feasibility smoke for the N=10000 toy/top-k pure edge-score
path. It uses `topk_backend="segmented_fast"`,
`query_support_backend="fused_fast"`, and `diagnostics_mode="lite"` by default.
The smoke performs one forward pass, one backward pass, and one optimizer
update only to check finite loss, finite gradients, parameter movement, sparse
edge counts, runtime decomposition, and tensor-memory proxy.

This is not a production trainer, model-quality result, or deployment result.
It preserves the same train/validation/deployment topology rule, keeps
structural bias disabled, and continues to forbid dense N-by-N graph paths and
direct hidden-physics loss objectives.

## M57 10k Backward Interpretation

M57 adds a report-only interpretation layer over the M56 backward smoke. It
classifies backward feasibility, parameter-update magnitude, reliability target
status, runtime bottleneck, and tensor-memory proxy without running additional
training or changing the model/evaluator/constructor path.

The report treats a tiny but nonzero update as computational feasibility with a
step-size caution, not as training readiness. Diagnostics-lite evidence is
explicitly labeled and remains separate from full diagnostic evidence.

## M58 Gradient Step-Size Diagnostic

M58 runs controlled one-step learning-rate probes over the same N=10000
toy/top-k pure edge-score path. It reuses a deterministic sparse environment,
reinitializes the model with the same seed for each learning rate, and records
gradient norms, loss-component contributions, update magnitudes, runtime, and
reliability state.

M58 is not multi-step training. It is allowed to recommend a conservative
few-step smoke only when update scale is healthy or learning-rate-sensitive
without large updates. It keeps structural bias disabled, uses
`topk_backend="segmented_fast"` and `query_support_backend="fused_fast"`, and
continues to forbid dense N-by-N graph paths and direct hidden-physics losses.

## M59 Gradient Source Diagnostic

M59 decomposes the M58 tiny-update result without changing the model, loss, or
constructor path. `scripts/analysis/run_gradient_source_diagnostic.py` reads
`.agent/tmp/gradient_step_size_diagnostic.json`, requires
`update_scale_status=updates_tiny_at_all_lrs` unless forced, and then runs
separate one-step probes for total loss, L_R, L_D, L_E, configured loss
scales, and wider learning rates.

The diagnostic keeps `node_count=10000`, `profile=toy`,
`support_mode=topk`, `topk_backend=segmented_fast`,
`query_support_backend=fused_fast`, and `diagnostics_mode=lite` by default.
It can recommend a calibrated loss-scale or optimizer-scale follow-up only
when a one-step probe produces finite gradients and a meaningful safe update.
It does not add multi-step training, production training infrastructure,
structural-head redesign, dense graph construction, direct hidden-physics
losses, or model-quality claims.

## M60 Loss-Scale Calibrated Backward Smoke

`scripts/analysis/run_loss_scale_calibrated_backward_smoke.py` validates the
M59 loss-scale recommendation with one controlled backward/update step. The
case remains N=10000, toy profile, top-k support, `topk_backend=segmented_fast`,
`query_support_backend=fused_fast`, and `diagnostics_mode=lite` by default.

M60 computes the raw coupled C/D/E loss normally, then uses
`scaled_loss = loss_scale * raw_total_loss` only for the single backward pass.
It records raw and scaled loss values separately, finite-gradient evidence,
parameter-update magnitude, runtime, and deterministic tensor-memory proxy. It
does not add a new objective, scheduler, optimizer policy, production trainer,
dense graph path, or direct hidden-physics loss.

## M61 10k Few-Step Training Smoke

M61 extends M60 only to a very short calibrated update sequence. The default
case remains N=10000, toy profile, top-k support,
`topk_backend=segmented_fast`, `query_support_backend=fused_fast`, and
`diagnostics_mode=lite`, with structural score bias disabled.

Each step computes the unchanged raw coupled C/D/E loss, uses
`scaled_total_loss = loss_scale * raw_total_loss` for backward, and records
raw loss, C/F/D/E, gradient norms, per-step and cumulative parameter movement,
runtime, memory proxy, active-edge count, and gradient coverage. M61 writes no
checkpoint, uses no scheduler, performs no model selection, and remains a
smoke diagnostic rather than 10k training evidence.

## M62 Metric Flatness Diagnostic

M62 adds a diagnostic layer for the M61 outcome where parameters move but
C/F/D/E remain effectively unchanged. It traces the signal chain from
parameter updates to edge-score deltas, top-k support overlap, row-softmax
topology weights, query-support probabilities, and final evaluator metrics.

The diagnostic also runs deterministic edge-score perturbation and
edge-score-only optimization probes. These probes are used only to localize
flatness; they do not change the model architecture, introduce a production
trainer, add a scheduler, reintroduce structural heads, or validate longer 10k
training. M62 explicitly avoids treating a larger learning rate or loss scale
as sufficient without signal-chain evidence.

## M63 Top-k Support Sensitivity Probe

M63 is a diagnostic-only layer over the M62 top-k boundary finding. It measures
baseline top-k margins, deterministic edge-score perturbation thresholds,
score-scale sensitivity, and analysis-only row-softmax temperature sensitivity
for the N=10000 toy/top-k pure edge-score path.

The probe preserves the existing sparse constructor and evaluator path:
`topk_backend=segmented_fast`, `query_support_backend=fused_fast`,
`diagnostics_mode=lite`, structural bias disabled, and top-k structural bias
disabled. Score scaling, perturbations, and temperature sweeps are sensitivity
probes only. They are not production training, not a new deployable constructor
rule, not a scheduler, and not model-quality evidence.

## M64 Score-Scale Calibration Diagnostic

M64 calibrates a scalar multiplier for the existing pure edge-score top-k path
after M63 identified score-scale sensitivity under rigid support. It runs a
forward-only score-scale sweep and isolated one-step calibrated update probes
with the same sparse constructor and evaluator backends.

The score scale is applied as `scaled_edge_score = score_scale * edge_score`
before `TopologyConstructionLayer`. The constructor rule, support mode,
selected-candidate mapping, structural-bias policy, and train/validation/
deployment graph rule remain unchanged. A passing M64 report can authorize only
an M65 score-scaled few-step smoke, not production training or a model-quality
claim.

## M65 Score-Scaled Few-Step Smoke

M65 runs the authorized short follow-up using the M64 `recommended_score_scale`
by applying `scaled_edge_score = score_scale * edge_score` before the unchanged
hard top-k constructor. The default path remains N=10000 toy/top-k with
`topk_backend=segmented_fast`, `query_support_backend=fused_fast`,
`diagnostics_mode=lite`, structural bias disabled, `loss_scale=1000.0`, and
`learning_rate=0.01`.

The report records raw loss and C/F/D/E trends, edge-score movement, selected
support overlap, topology-weight deltas, query-support deltas, runtime, and
memory proxies. It also compares against the unscaled M61 artifact when
available. M65 is still a diagnostic smoke: it does not change the loss
semantics, add a scheduler, run long 10k training, or make model-quality or
deployment claims.

## M66 Score-Scaled Interpretation

M66 is a report-only layer over the M65 score-scaled few-step artifact and the
M61/M62/M63/M64 context artifacts. It compares score_scale=3 movement against
the unscaled M61 smoke, checks whether M64 higher safe scale candidates remain
available, and records whether M63 analysis-only row-softmax temperature probes
moved C/F/D/E under fixed support.

M66 does not run training, tune learning rate or loss scale, add a new
constructor rule, or promote analysis-only temperature probes to deployment
semantics. If score_scale=3 remains flat while M64 has higher safe candidates,
the conservative next stage is a score-scale/temperature grid diagnostic before
any longer 10k run.

## M67 Score-Temperature Grid Probe

M67 runs a diagnostic-only grid over score scale and row-softmax temperature
for the existing N=10000 toy/top-k path. Score scale remains a scalar
multiplier on `edge_score` before the unchanged sparse top-k constructor.
Temperature values other than `1.0` are analysis/control probes only and are
not deployable constructor semantics.

Each grid cell records forward sensitivity against the scale-1, temperature-1
baseline, then performs a calibrated one-step update and a post-update
reforward. This distinguishes pre-update control sensitivity from actual
optimizer-induced topology/query/metric movement without adding long training,
new losses, schedulers, or model-selection behavior.

## M68 Row-Softmax Temperature Design

M68 is a report-only topology design layer over the M67 grid evidence. It
evaluates whether the analysis-only row-softmax temperature control should
become a formal `TopologyConstructionLayer` parameter named
`row_softmax_temperature`.

The proposed parameter applies only to row-softmax over the selected sparse
support. It must not change top-k membership directly, candidate graph
construction, sparse selected-candidate mapping, or structural-bias policy.
If implemented later, it must use identical train/validation/deployment
semantics, with `row_softmax_temperature=1.0` preserving the current behavior
exactly. M68 does not authorize temperature-aware training; it can only
authorize an M68.1 prototype and equivalence-test milestone.

## M68.1 Row-Softmax Temperature Prototype

M68.1 promotes `row_softmax_temperature` into an explicit
`TopologyConstructionLayer` parameter. The default value `1.0` preserves the
legacy constructor rule. Non-1.0 values change only the row-softmax
distribution over already selected sparse edges; they do not change candidate
graph construction, deterministic top-k membership, structural-bias policy, or
the shared train/validation/deployment graph rule.

This is still architecture scaffolding and diagnostic enablement, not 10k
training or model-quality evidence. Future artifacts using a non-1.0
temperature must report the value and use the same constructor rule across
train, validation, and deployment.

## M69 Formal Constructor Temperature Grid

M69 reruns the score-scale and row-softmax-temperature grid using the formal
`TopologyConstructionLayer(row_softmax_temperature=...)` parameter. This is a
diagnostic rerun of the M67 grid after M68.1, not a new trainer.

Each grid cell applies `score_scale * edge_score` before the sparse
segmented-fast top-k constructor, then applies the configured
`row_softmax_temperature` through the constructor itself. The report labels all
temperature rows as `analysis_only_temperature=false` and
`deployable_constructor_rule=true`, while still requiring a follow-up smoke
before any temperature-aware training claim. Deltas are compared against the
same score scale at `row_softmax_temperature=1.0`, preserving sparse
candidate-edge input, selected-candidate mapping, row normalization, zero
structural-bias weights, and the shared train/validation/deployment graph rule.

M69 does not add production training, checkpointing, schedulers, model
selection, dense N-by-N graph construction, stochastic sampling, or direct
link/SINR/BLER/HARQ/coverage loss terms.

## M70 Temperature-Aware 10k Few-Step Smoke

M70 runs the follow-up few-step smoke selected by the M69 formal constructor
grid. It uses `score_scale=30.0` and
`TopologyConstructionLayer(row_softmax_temperature=0.5)` with
`topk_backend="segmented_fast"`, `query_support_backend="fused_fast"`, and
`diagnostics_mode="lite"`.

The temperature is now a formal constructor rule, not analysis-only
postprocessing. Each step applies `score_scale * edge_score` before the sparse
top-k constructor, computes the existing coupled C/D/E loss, applies the
calibrated backward loss scale, performs one optimizer step, and then reruns a
post-update forward pass to measure actual topology/query/C/F/D/E movement.

M70 remains a controlled diagnostic smoke. It does not add checkpointing,
schedulers, model selection, dense N-by-N construction, stochastic topology
sampling, direct link/SINR/BLER/HARQ/coverage losses, or long 10k training.

## M71 Temperature-Aware 10k Interpretation

M71 is a report-only layer over the M70 temperature-aware few-step artifact.
It checks whether the formal constructor temperature signal improved relative
to the M61 unscaled and M65 score_scale=3 baselines while keeping support
stable or only slightly changed.

The report can authorize only a longer controlled temperature-aware smoke. It
does not run additional training, change the constructor, tune the temperature
or score scale, add schedules or checkpoints, or convert toy-profile evidence
into model-quality or deployment claims.

## M72 Temperature-Aware Longer 10k Smoke

M72 runs that authorized longer diagnostic smoke with the same formal
constructor rule selected by M69 and checked by M70/M71:
`score_scale=30.0`, `row_softmax_temperature=0.5`,
`topk_backend="segmented_fast"`, `query_support_backend="fused_fast"`, and
`diagnostics_mode="lite"` by default.

The run remains a controlled smoke, not production training. Each step computes
the unchanged raw coupled C/D/E loss, uses the calibrated scalar only for
backward, performs one optimizer step, and then reruns a post-update forward
pass to measure actual edge-score, topology-weight, query-support, and C/F/D/E
movement. It writes no checkpoints, uses no scheduler, performs no model
selection, keeps structural bias disabled, and preserves sparse O(Nk)
constructor/evaluator paths.

## M73 Temperature-Aware Longer Interpretation

M73 is report-only over the M72 longer-smoke artifact. It checks whether the
formal temperature-aware signal accumulates relative to the M70, M65, and M61
context artifacts, whether support remained stable or only slightly changed,
and whether the reliability-gap movement is large enough to justify another
controlled diagnostic.

The report does not run training, change `TopologyConstructionLayer`, tune
score scale or temperature, add schedulers/checkpoints, or promote toy-profile
evidence into a model-quality claim. A positive M73 result can authorize only a
claim-limited M74 diagnostic such as a 30-step smoke, temperature/score-scale
refinement, all-mode warm-start probe, or support-smoothing design.

## M74 Temperature-Aware 30-Step 10k Smoke

M74 runs the M73-authorized 30-step diagnostic using the same formal
temperature-aware constructor rule as M72: `score_scale=30.0`,
`row_softmax_temperature=0.5`, segmented-fast top-k, fused query support, and
diagnostics-lite by default. It is a controlled smoke, not production training.

Each step computes the unchanged raw coupled C/D/E loss, uses the calibrated
scalar only for backward, performs one optimizer step, and then reruns a
post-update forward pass. The report adds trend slopes, per-step C/F and
reliability-gap movement, a cautious reliability-gap projection, and movement
ratios versus M72. M74 writes no checkpoints, uses no scheduler, performs no
model selection, keeps structural bias disabled, and preserves sparse O(Nk)
constructor/evaluator paths.

## M75 Temperature-Aware 30-Step Interpretation

M75 is report-only over the M74 artifact. It reads the 30-step trend,
projection, support-stability, and M72/M70/M65 comparison fields, then selects
the next diagnostic strategy. It does not run additional training, tune
learning rate, tune loss scale, change `TopologyConstructionLayer`, or promote
toy-profile evidence into a model-quality claim.

If M74 movement accumulates but the projected reliability-gap closure remains
impractically slow, M75 routes away from blind hard-top-k extension toward
all-mode warm-start or support-smoothing diagnostics. A recommendation for any
M76 stage remains a controlled diagnostic decision, not production training
readiness.

## M76 All-Mode Warm-Start Probe

M76 runs a controlled all-mode warm-start diagnostic after M75 determines that
continuing the hard-top-k temperature-aware path is impractically slow. The
probe uses `support_mode="all"` so every feasible sparse candidate edge remains
active under the same topology constructor/evaluator/loss stack.

`topk_backend` is recorded as not applicable for the report, although the
shared constructor helper still receives an internal backend value for API
compatibility. The probe keeps structural bias disabled, uses fused query
support with diagnostics-lite by default, applies the same raw coupled C/D/E
loss, and uses the calibrated loss scale only as a backward scalar. Each step
performs a post-update reforward so all-mode edge-score, topology-weight,
query-support, and C/F/D/E movement can be compared against the M74/M75
hard-top-k temperature-aware artifacts.

M76 writes no checkpoints, uses no scheduler, performs no model selection, and
does not promote all-mode as a production or deployment rule. Passing M76 can
only authorize an M77 interpretation or curriculum-design report.

## M77 All-Mode Warm-Start Interpretation

M77 is report-only over the M76 all-mode warm-start artifact. It compares the
default `support_mode="all"`, `score_scale=1.0`,
`row_softmax_temperature=1.0` result against the M74/M75 hard-top-k
temperature-aware path, including C/F/D/E movement, reliability-gap projection,
gradient coverage, runtime, and active sparse edge count.

The report does not run additional 10k steps, tune score scale or temperature,
change `TopologyConstructionLayer`, alter the evaluator, or make all-mode a
deployment rule. If default all-mode is finite but weaker than M74 and
impractically slow, M77 routes to an all-mode score/temperature grid or support
smoothing diagnostic rather than blind continuation.

## M78 Convergence Root-Cause Audit

M78 is a bottom-to-top diagnostic audit over the accumulated 10k artifacts. It
compares hard-top-k unscaled, hard-top-k temperature-aware, and default all-mode
evidence, then records deterministic query-support, fixed-support,
edge-score-only, candidate-oracle, loss-gradient, and GNN score-range probes.

The audit is report/probe infrastructure, not a training loop. It does not
change model architecture, topology-constructor semantics, evaluator math, or
loss definitions. Its purpose is to decide whether the next architectural work
should focus on scorer parameterization, support smoothing, candidate graph
capacity, evaluator sensitivity, loss-gradient governance, or an all-mode to
top-k curriculum design.

## M79 Scorer Parameterization Redesign

M79 is report/design only. It reads the M78 root-cause audit plus optional
M59/M62/M75 context and summarizes why scorer parameterization is the next
architecture target when evaluator sensitivity, fixed-support capacity,
candidate-graph capacity, and reliability gradients exist but scorer parameter
updates produce too little edge-score movement.

The report inspects the current `HierarchicalGNNScorer` architecture at a
design level: node encoder, edge encoder, sparse message passing, optional
region pooling, score head, structural heads, initialization, ReLU activations,
and the pure edge-score output path. It explicitly records that the current
scorer has no internal score gain or row normalization and that later
`score_scale` diagnostics were external to the model.

M79 does not change the default scorer forward behavior. It documents redesign
options such as explicit `score_output_gain`, score-head initialization scale,
score row normalization, edge-score teacher probes, teacher distillation,
residual edge-feature score shortcuts, and score-head optimizer-group scaling.
Because current artifacts still lack edge-score distribution, selected-score
distribution, score-head parameter norms, layer-wise update norms, explicit
edge-score sensitivity, and feature normalization statistics, the conservative
next stage is `M79_1_scorer_dynamic_range_diagnostic` rather than an immediate
production-path scorer implementation.

## M79.1 Scorer Dynamic Range Diagnostic

M79.1 fills those missing scorer diagnostics without changing the model
architecture. It runs a single calibrated toy/top-k diagnostic pass, records
all-candidate, selected, and unselected edge-score distributions, measures
top-k boundary margins, records edge-score movement after one optimizer step,
and reports layer-wise parameter, gradient, and update norms.

The diagnostic also captures node/edge/message/final-score-input scale
statistics and score-head sensitivity estimates. Its score-scale sweep is a
what-if analysis only; it is not an internal scorer gain, a new constructor
rule, or a production training behavior. M79.1 can authorize only a later
score-output-gain prototype, score-head LR probe, initialization probe,
feature-normalization probe, or edge-score-teacher diagnostic.

## M80 Explicit Score Output Gain Prototype

M80 adds `score_output_gain` as an explicit fixed scalar in
`HierarchicalGNNScorer`. The default `score_output_gain=1.0` preserves the
legacy edge-score output exactly. Non-default gains multiply the final
`edge_score` tensor only; structural heads and structural-bias diagnostics are
not changed, and structural bias remains disabled in the M80 baseline.

The gain is a scorer parameterization setting, not a hidden training trick or a
constructor rule. It must be configured and reported consistently for
train/validation/deployment. M80's probe checks gains such as 10, 30, and 100
through the unchanged sparse top-k constructor, formal
`row_softmax_temperature`, fused query support, and coupled C/D/E loss. It can
only authorize a diagnostic M81 few-step smoke; it does not validate model
quality or production training.

## M81 Score-Gain Few-Step Smoke

M81 runs the explicit scorer `score_output_gain` setting through a short
temperature-aware 10k toy/top-k smoke. Each gain is an independent
`HierarchicalGNNScorer(score_output_gain=...)` initialization; the external
constructor score scale remains `1.0`, and the topology rule remains the formal
`TopologyConstructionLayer(row_softmax_temperature=0.5)` path.

The smoke reports per-step edge-score movement, top-k margin ratios,
topology-weight movement, query-support movement, C/F/D/E deltas, support
overlap, and post-update reforward metrics. It is a diagnostic bridge from the
M80 one-step gain probe to M82 interpretation, not a production-training or
model-quality milestone.

## M82 Score-Gain Interpretation

M82 is report-only. It reads the M81 score-gain few-step artifact and optional
M80, M79.1, M70, and M74 context artifacts. It decides whether explicit
`score_output_gain` alone is sufficient, whether gain 30 is promising but still
below temperature-aware baselines, or whether a combined scorer-gain and
constructor-temperature diagnostic is the next step.

M82 does not instantiate `HierarchicalGNNScorer`, call the constructor, run
training, change score-output-gain behavior, alter row-softmax temperature, or
add loss terms. Its architectural role is to keep scorer-gain evidence separate
from future combined gain/temperature probes and to preserve the current
train/validation/deployment topology-constructor contract.

## M83 Combined Gain/Temperature Probe

M83 runs a controlled diagnostic grid that combines the scorer's explicit
`score_output_gain` with the formal constructor `row_softmax_temperature`.
The external `score_scale` remains a separate reported control and defaults to
`1.0`. The probe uses the same sparse top-k constructor, fused query support,
and coupled C/D/E loss as M81/M74.

M83 is not production training or model-quality validation. It exists to decide
whether combining scorer output gain with formal constructor temperature
improves C/F/D/E movement without support instability before any longer or
broader strategy is considered.

M83 also introduces a reusable training-result visualization writer for
diagnostic reports with per-step rows. The primary visualization view is now a
fixed-node curve over training step for raw/scaled loss, C/F/D/E, reliability
gap, gradient norm, score/topology/query movement, and support stability. Cross
node-count or cross-candidate comparison charts can still be written as
secondary context, but they must not replace the fixed-node trend view when
stepwise training data are available.

## M84 Edge-Score Teacher Probe

M84 adds a diagnostic-only edge-score teacher probe after the combined
score-output-gain and row-softmax-temperature path remains metric-flat. The
probe lives outside the default training path and freezes the GNN whenever it
optimizes teacher logits or candidate edge scores directly. It preserves the
existing sparse candidate graph, hard top-k constructor, formal
`row_softmax_temperature` rule, and closed-form evaluator. Teacher modes are
used only to test capacity and scorer alignment; they are not architecture
changes and do not introduce production training behavior.

## M85 Edge-Score Teacher Interpretation

M85 is report-only over the M84 teacher artifact and scorer-bottleneck context.
It classifies each teacher mode as useful, flat, unstable, wrong-direction, or
invalid, then separates static scorer rank alignment from update-direction
alignment. This distinction matters when the current GNN scores rank similarly
to a teacher direction but ordinary scorer updates are too weak to realize that
direction.

The report can recommend a teacher-guided scorer update design for M86, but it
does not add a teacher objective, change default `HierarchicalGNNScorer`
behavior, alter topology construction, or validate 10k training.

## M86 Teacher-Guided Scorer Update Design

M86 is report/design only over the M85 interpretation. It evaluates how a
future prototype could use the useful `edge_score_gradient_direction` teacher
without changing the default training path. The expected current
recommendation is a stop-gradient `edge_score_delta_distillation` prototype,
with score-head-only teacher updates and alignment ablations as secondary
controls.

M86 explicitly excludes `candidate_oracle_rank` because M85 found it
wrong-direction and support-jumpy. It also excludes direct physics teachers,
stochastic sampled teachers, dense N-by-N teacher paths, and hidden train-only
teachers. A positive M86 report can authorize only an M87 prototype or
ablation harness; it does not add a teacher objective, redesign structural
heads, validate 10k training, or change train/validation/deployment topology
constructor semantics.

## M87 Edge-Score Delta Distillation Prototype

M87 implements that authorized prototype as an explicit opt-in diagnostic. It
constructs a detached local teacher target from the
`edge_score_gradient_direction` teacher, adds the reported teacher loss only
inside the M87 diagnostic run, and compares teacher-on against teacher-off,
shuffled-teacher, and zero-teacher ablations.

The default training path remains teacher-free. M87 does not promote
`candidate_oracle_rank`, add direct physics supervision, sample teacher edges,
allocate dense N-by-N teacher tensors, checkpoint models, add schedulers, or
validate 10k training. Any positive M87 result can authorize only an M88
interpretation or a tighter teacher-weight diagnostic before broader training
claims are considered.

## M88 Edge-Score Delta Distillation Interpretation

M88 is report-only over the M87 opt-in distillation artifact. It does not run
training, add a teacher objective to default training, change scorer behavior,
or change topology-constructor semantics.

The report compares teacher-on with teacher-off, shuffled-teacher, and
zero-teacher controls, then separates three cases: the current distillation has
no effect, the teacher term is numerically too small relative to the scaled
primary C/D/E objective, or the objective form remains too weak even at a
nontrivial reported scale. Any M88 recommendation is a next diagnostic strategy
such as teacher-loss scale probing, directional alignment probing,
score-head-only teacher update probing, or keeping teacher guidance diagnostic
only.

## M89 Teacher Loss Scale Probe

M89 is the M88-authorized opt-in scale sweep for the M87
`edge_score_delta_distillation` prototype. It repeats the same governed
teacher construction, but tests larger explicit teacher weights while running
teacher-off, teacher-on, shuffled-teacher, and zero-teacher controls for each
scale.

The default training path remains teacher-free. M89 does not change
`HierarchicalGNNScorer` defaults, alter `TopologyConstructionLayer`, promote
`candidate_oracle_rank`, add direct physics supervision, use stochastic
teachers, allocate dense N-by-N tensors, checkpoint models, add schedulers, or
validate 10k training. Its output can only authorize an M90 interpretation or a
more specific teacher-guidance diagnostic.

## M90 Teacher Loss Scale Interpretation

M90 is report-only over the M89 teacher-loss scale artifact. It reads the
per-scale teacher-on/off/shuffled/zero controls and classifies whether
MSE-style edge-score delta distillation has a safe useful scale.

The report does not run training, change `HierarchicalGNNScorer`, change
`TopologyConstructionLayer`, add a teacher objective to the default path,
promote candidate-oracle ranking, use direct physics losses, use stochastic
teachers, or allocate dense N-by-N teacher tensors. If low teacher weights are
too small, the mid-scale signal is distinguishable but below the useful
threshold, and high weights are support-jumpy, the next architecture step is a
different opt-in teacher diagnostic such as directional teacher alignment.

## M91 Directional Teacher Alignment Probe

M91 is the M90-authorized opt-in diagnostic that replaces tiny-delta MSE
regression with scale-normalized alignment against the detached
`edge_score_gradient_direction`. It tests `cosine_delta_alignment`,
`sign_alignment`, and `normalized_mse_direction` under the same sparse
top-k/temperature-aware constructor path and compares aligned-teacher-on
against teacher-off, shuffled-teacher, and zero-teacher controls.

M91 keeps the default training path unchanged and does not modify
`HierarchicalGNNScorer` defaults, `TopologyConstructionLayer` semantics,
candidate graph construction, structural-bias policy, or deployment rules.
The teacher direction is diagnostic-only, stop-gradient, explicitly reported,
and forbidden from using candidate-oracle ranking, direct physics supervision,
stochastic sampling, or dense N-by-N tensors. Its output can only authorize an
M92 interpretation or another explicitly opt-in teacher diagnostic.

## M92 Directional Teacher Alignment Interpretation

M92 is report-only over the M91 directional alignment artifact. It interprets
alignment movement, C/F movement, control separation, and support stability
without running additional optimizer steps or changing the model.

If alignment improves but aligned-teacher-on does not beat teacher-off,
shuffled-teacher, and zero-teacher controls, M92 treats the signal as
control-confounded rather than validated. A positive M92 recommendation can
authorize only another explicit diagnostic, such as a score-head-only teacher
update probe. It does not add a teacher objective to the default path, change
constructor semantics, validate 10k training, or make production-readiness
claims.

## M93 Score-Head-Only Teacher Update Probe

M93 is the M92-authorized opt-in diagnostic that isolates the teacher-guided
update to `HierarchicalGNNScorer.edge_score_head`. It compares
`update_scope="score_head_only"` with `update_scope="full_model"` while keeping
teacher-off, aligned-teacher-on, shuffled-teacher, and zero-teacher controls.

The score-head-only scope optimizes only edge-score-head parameters and reports
whether all non-score-head parameters stayed unchanged. The full-model scope is
comparison evidence only. M93 does not change scorer defaults, constructor
semantics, structural-bias policy, candidate graph construction, checkpointing,
schedulers, or the default teacher-free training path.

## M94 Support Smoothing Design

M94 is report/design only after M93 fails to validate a score-head-only teacher
effect or finds that the teacher path is unstable. It specifies a future sparse
deterministic support-smoothing constructor prototype without implementing it.

The design requires legacy defaults to preserve current behavior exactly. Any
future smoothing must be explicit, sparse, deterministic, reported in artifacts,
free of stochastic sampling and dense N-by-N allocation, and used with the same
train/validation/deployment forward rule within each curriculum stage. M94 can
authorize only an M95 constructor prototype, not training or production claims.

## M95 Support Smoothing Constructor Prototype

M95 adds explicit support-smoothing constructor parameters with inert defaults
and no default-training change. The prototype mode
`deterministic_topk_halo` is active only for the explicit
`expanded_sparse_support` stage with a positive per-row halo count.

The constructor still consumes only sparse candidate edges. In expanded mode it
selects the existing base top-k support and then adds bounded next-ranked sparse
halo candidates per active source row. The same configured forward rule is used
in train, validation, and deployment contexts; there is no `self.training`
branch, stochastic support sampling, candidate-oracle ranking, dense N-by-N
support expansion, direct physics loss, or structural-head default change.

M95 can authorize only a bounded M96 10k diagnostic smoke. It is not training
evidence, model-quality evidence, or production readiness.

## M96 Support Smoothing 10k Diagnostic

M96 exercises the M95 prototype as an explicit diagnostic configuration, not
as a default architecture change. It combines configured `score_output_gain`,
formal `row_softmax_temperature`, and the opt-in expanded sparse support stage
in one bounded smoke while preserving the shared train/validation/deployment
constructor rule.

If M96 movement comes from a support jump, the architecture path is not
validated and the next safe step is a loss/evaluator sensitivity review rather
than a stronger training run. A positive M96 would still be diagnostic smoke
evidence only and would require a controlled strategy interpretation package
before any longer or multi-seed run.

## M97 Human-Review Gate

M97 is report-only architecture governance. It does not introduce a new model,
constructor rule, loss, trainer, or default configuration. If teacher,
hard-top-k, all-mode, score-gain, and support-smoothing diagnostics fail to
produce stable movement, M97 records the blocker and pauses before any change
that would affect the mathematical objective, evaluator thresholds, or
candidate graph contract.

The post-M97 human-review gate package is also report-only. It records the
active M91+ stop condition as `.agent/tmp/human_review_gate_package.json` and
does not treat unrelated harness artifacts written later as authorization to
bypass M97. It cannot authorize a stronger run, change defaults, or claim
production readiness.

## M98 Support-Smoothing Stabilization Design

M98 records an explicit human decision to continue support-smoothing
stabilization research while keeping production training, stronger runs,
default-path changes, objective/evaluator changes, dense support paths,
stochastic support, structural bias in the default path, and hidden train-only
rules unauthorized. It writes
`.agent/tmp/human_review_decision_support_smoothing.json` and
`.agent/tmp/support_smoothing_stabilization_design.json`.

The M98 design keeps the architecture report-only. Its conservative next step
is a support-smoothing parameter sweep that tries to retain the M96 C/F
movement while reducing `active_edge_change_total_fraction` below explicit
support-jump thresholds. M98 also adds a density-scaling requirement: future
production training must report stable candidate and active degree across 100,
500, 2000, 5000, and 10000 nodes, and spatial density must be reported or
audited before any production-training claim.

## M99 Conservative Support-Smoothing Sweep

M99 executes the M98 conservative sweep as an opt-in diagnostic branch. It
reuses the M96 support-smoothing smoke runner for each explicit grid cell and
selects a candidate only if C/F and reliability-gap movement improve against
M74 or M76 while active-edge change remains below the support-jump blocker.

M99 is not an architecture default. It does not add a trainer, scheduler,
checkpoint path, new objective, evaluator change, structural bias, dense
support path, candidate-oracle ranking, or hidden train-only rule. If the
selected candidate is produced with an overridden node count, it can only
authorize a future full-10k diagnostic smoke; it cannot claim full-10k evidence
or production readiness.

## M100 Full-10k Support-Smoothing Smoke

M100 reruns the M99-selected explicit support-smoothing candidate at
`node_count=10000` with no node-count override. It remains a diagnostic smoke:
no production trainer, checkpointing, scheduler, default-path change,
objective/evaluator change, structural bias, or hidden train-only rule is
introduced.

M100 can only authorize an interpretation report if the full-10k run preserves
support stability and retains material C/F and reliability-gap movement. If
movement disappears or support becomes unstable at 10k, the architecture route
returns to human review rather than changing the constructor, evaluator, or
training objective.

## M101 Support-Smoothing Scale-Transfer Audit

M101 records the human decision to investigate the M100 scale-transfer failure
with a diagnostic-only branch. It reruns the exact M99-selected support
smoothing parameters across node counts 100, 500, 2000, 5000, and 10000 and
reports C/F/reliability movement, gradient/update norms, support stability,
sparse density, runtime, and tensor-memory proxies at each scale.

M101 does not add production training, checkpointing, schedulers, dataset
loading, objective/evaluator changes, structural bias, dense N-by-N paths,
stochastic sampling, direct physics losses, or default-path changes. Its only
allowed output is a scale-transfer diagnosis and a report/design-stage M102
recommendation such as scale-aware update normalization, support-smoothing
parameter rescale design, density-scaling environment audit, or strategy
rejection.

## M102 Scale-Aware Update Normalization Design

M102 is report/design only. It interprets the M101 finding that the M99
support-smoothing candidate retained small movement at 500 nodes but collapsed
at 10000 nodes, then defines explicit opt-in update-normalization candidates
for a future diagnostic probe.

M102 does not implement normalization, production training, checkpointing,
schedulers, model selection, objective/evaluator changes, structural bias, dense
N-by-N paths, stochastic sampling, direct physics losses, or default-path
changes. Its only allowed next step is a bounded M103 diagnostic probe with
reported multipliers, no-normalization controls, sparse O(Nk) checks, and the
same train/validation/deployment topology rule.

## M103 Scale-Aware Update Normalization Probe

M103 implements the bounded diagnostic probe designed by M102. It reruns the
M99 support-smoothing candidate across 500, 2000, and 10000 nodes with a
no-normalization control plus node-count and active-edge-count power
normalization candidates.

The architecture contract remains unchanged: normalization is an explicit
opt-in multiplier on the effective update/loss-scale path, not a default model
or constructor behavior. M103 does not add production training, checkpointing,
schedulers, model selection, objective/evaluator changes, structural bias,
dense N-by-N paths, stochastic sampling, direct physics losses, or hidden
train-only topology rules.

## R1/R2 Scale-Law Review Boundary

R1 is a report-only root-cause audit over the M99-M103 artifacts. It identifies
a mixed scale-law bottleneck: loss/update attenuation, query-support
sensitivity collapse, overdamped smoothing mass, scorer update shrinkage, and a
missing spatial-density contract. R1 does not change architecture behavior.

R2 is a remediation design report only. It compares scale-invariant
loss/update design, query-support sensitivity design, topology smoothing mass
design, scorer/update rescale design, and environment-density contract audit
options. The recommended next branch is an R3 design/audit decision, not a
production trainer or prototype implementation. R2 does not add dataset
loaders, checkpointing, schedulers, model selection, objective/evaluator
changes, structural-head redesign, dense N-by-N paths, direct physics losses,
support-smoothing behavior changes, or default training-path changes.

## R3 Scale-Invariant Loss/Update Design

R3 is report/design only. It accepts the R2 human decision to focus on
scale-invariant loss/update design and carries query-support sensitivity as a
required subdesign. R3 defines candidate designs for reference-scale normalized
sum, active-failure-node normalization, gradient-norm target normalization,
query-support coupled normalization, and topology smoothing mass-aware
normalization.

R3 does not implement any architecture, trainer, optimizer, objective,
evaluator, constructor, support-smoothing, dataset, scheduler, checkpoint, or
default-path change. The only allowed output is a human-reviewed R4 design
review recommendation with raw-vs-effective reporting requirements.

## R4 Scale-Invariant Design Review

R4 is report-only. It reviews the R3 mixed design and ranks the candidate
branches without implementing a prototype or approving a full mixed
implementation. The review selects a bounded R5 diagnostic branch when
extra-edge mass, row entropy, and query-support sensitivity diagnostics are
missing.

R4 keeps production training, default-path changes, objective/evaluator
changes, constructor changes, optimizer changes, scheduler changes, and
support-smoothing behavior changes false. The density contract remains
mandatory before production-readiness claims.

## R5 Topology Smoothing Mass and Query-Sensitivity Diagnostic

R5 is the first remediation-stage runnable diagnostic after R4. It executes the
accepted support-smoothing candidate across node counts and records new sparse
measurements for extra-edge mass, selected-edge mass, row entropy,
topology-weight-to-query-support sensitivity, active-row sensitivity, and
active-failure-node counts.

R5 does not implement a prototype loss, scale-invariant objective, trainer,
optimizer, scheduler, checkpoint, evaluator change, constructor behavior
change, support-smoothing behavior change, default-path change, dense N-by-N
path, direct physics loss, or production-training path. The diagnostic labels
base versus smoothing-extra edges only inside the analysis path so architecture
semantics remain unchanged.

## R6 Query-Support Sensitivity Counterfactual Probe

R6 is a runnable counterfactual diagnostic over the R5 mixed bottleneck. It
builds the same sparse support-smoothing topology, then applies deterministic
post-construction topology-weight perturbations over the existing sparse
support to test whether query support and C/F metrics can move when topology
weights are changed directly.

R6 does not train the model, update parameters, change evaluator behavior,
change objective/loss behavior, modify `TopologyConstructionLayer`, alter
support-smoothing semantics, add a default training path, allocate dense N-by-N
graphs, or add direct physics losses. The perturbations preserve support
indices and row normalization for valid cases and exist only inside the
diagnostic artifact.

## R7 Realistic Density Perturbation Capacity Diagnostic

R7 is a runnable diagnostic that moves the remediation evidence route away
from toy-only claims. It inspects the available profile/density contract for
toy and `small_realistic` proxy profiles, records graph-degree and spatial
density fields where available, and reruns the R6 gradient-direction
support-preserving topology perturbation across node counts 500, 2000, and
10000 by default.

R7 does not train the model, update parameters, change the evaluator,
objective/loss, constructor, support-smoothing semantics, default path, or use
dense N-by-N graph operations. The diagnostic may conclude that topology
perturbation capacity transfers to a realistic proxy, but production-density
claims require an explicit production density profile contract before any
production-training decision.

## R8 Production Density Profile Probe

R8 is a runnable density-contract sanity probe. It defines
`production_like_density_v0` as a diagnostic-only profile generator that scales
spatial area with node count, keeps node density approximately constant, uses
an explicit sparse candidate radius/cap, reports zero RSUs as a v0
simplification, and verifies hard top-k degree stability from 100 to 10000
nodes.

R8 also reruns toy and `small_realistic` profile checks to document why they
are not production-density evidence. It evaluates the existing Avalanche
metrics on a hard top-k baseline only; it does not train parameters, add a
dataset loader, alter evaluator/loss/constructor behavior, change support
smoothing, or change the default training path.

## R9 Production Profile Perturbation Capacity Probe

R9 is a runnable diagnostic that uses the R8 `production_like_density_v0`
candidate as the primary profile. It runs deterministic, support-preserving
topology-weight perturbations over the existing sparse support to test whether
query support and C/F metrics can move under the production-density candidate.

R9 does not train or update model parameters, add a dataset loader, alter the
evaluator, objective/loss, constructor, support-smoothing semantics, optimizer,
scheduler, or default path. It reports whether production-profile perturbation
capacity exists and routes any follow-up to another runnable diagnostic or
human review.

## R10 Production Profile Training Signal Diagnostic

R10 is a runnable diagnostic that measures the loss-to-topology, loss-to-edge
score, and loss-to-scorer-parameter gradient signal under
`production_like_density_v0`. It also runs detached edge-score gradient
counterfactuals to check whether the raw gradient direction can move query
support and C/F metrics.

R10 performs no optimizer step and does not mutate model parameters. It does
not change training, evaluator, loss, constructor, support smoothing, or
default behavior.

## R11 Loss-to-Score Gradient Alignment Diagnostic

R11 is a runnable diagnostic that reuses the R10 production-like profile path
and decomposes the existing C/D/E loss gradient by component. It records
edge-score and topology-weight gradient norms, cosine conflicts, component
edge-score counterfactuals, and a topology-weight total-gradient comparison to
localize whether the failure is in the loss components or the constructor
Jacobian from topology weights back to edge scores.

R11 does not train, update parameters, alter the evaluator, alter the
constructor, alter support smoothing, or change the default training path.

## R12 Constructor Jacobian Alignment Diagnostic

R12 is a runnable diagnostic that follows the R11 constructor-Jacobian finding.
It freezes the baseline support to isolate row-softmax score-to-weight mapping,
then compares that fixed-support result with a full constructor re-forward
through the existing top-k/support-smoothing path. It also records base top-k
versus support-smoothing extra-edge mass transfer, row-level alignment, and
support expressivity diagnostics.

R12 does not train, update parameters, alter the evaluator or objective, alter
constructor behavior, alter support smoothing, or change the default training
path. The fixed-support row-softmax recomputation is diagnostic-only and is not
a deployable constructor path.
