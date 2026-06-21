# Topology Constructor

The model constructs topology; it does not classify each candidate edge independently.

## Required Rule

Training, validation, and deployment use the same `forward()` path. The constructor must not train on one graph rule and validate or deploy on another.

## Implemented Default: M5 Mode B

M5 currently implements Mode B: a deployable sparse query distribution. It consumes sparse candidate edges, applies optional deterministic support selection, and emits nonnegative row-normalized query weights for the selected sparse edges.

The M5 forward rule is:

1. Keep only sparse feasible candidate edges.
2. Use `support_mode="topk"` to apply a deterministic score top-k cap per source node, or `support_mode="all"` to keep every feasible sparse candidate.
3. Apply row softmax over the selected candidates.
4. Return sparse `src_index`, `dst_index`, and `topology_weight`.

The same rule is used in train, validation, and deployment. The score top-k cap is not an eval-only threshold, and it is never applied only outside training.

top-k membership is nondifferentiable at score-order changes. Within the selected support, row-softmax query weights are differentiable with respect to `edge_score`. Candidates excluded by the cap are outside the gradient path for that forward pass, so M5 reports gradient-risk diagnostics.

Experiments that need wider score-gradient coverage may set `support_mode="all"`, `max_out_degree=None`, or a high cap, but that choice must be fixed for all train/validation/deployment phases of that experiment.

## M5.2 Training Readiness

M5.2 reduces constructor CPU synchronization. Row softmax is vectorized over sparse selected edges using tensor indexing and `index_add`. The all-candidate path avoids per-row Python loops. The top-k path still uses a small deterministic loop over unique source rows because PyTorch does not provide a simple segmented top-k primitive in the current project dependency set. It no longer converts every feasible edge source to a CPU list; remaining synchronizations are per active source row for the top-k budget and slice boundaries.

Recommended staged use:

1. Warm-start score learning with `support_mode="all"` or a high/no cap so all sparse candidate scores receive row-softmax gradients.
2. Move to a fixed `support_mode="topk"` cap if deployable query budgets require it.
3. Keep the chosen support mode and cap identical for train, validation, and deployment within each experiment.

## M6 Scorer Integration

M6 adds `HierarchicalGNNScorer`, which produces the `edge_score` tensor consumed by `TopologyConstructionLayer`. The scorer is not a topology constructor and does not apply thresholding, sampling, repair, or train/eval-specific graph rules. It consumes only sparse candidate graph tensors and returns one differentiable score per candidate edge.

M6.1 keeps the hierarchy deliberately small. The current scorer has node and edge feature encoders, sparse message passing, optional lightweight region context, and an edge score head. Topology construction still uses the fixed `max_out_degree`, `support_mode`, and optional feasibility mask supplied to `TopologyConstructionLayer`.

M6.2 moves from pure edge scoring toward a structured topology planner skeleton. The scorer can now emit:

- `node_budget_logits` and `node_budget_expected`,
- `node_role_logits`,
- `sector_preference_logits`,
- sparse per-candidate `region_bridge_logits`.

These outputs are differentiable model outputs, not loss terms. The budget head remains soft/diagnostic for now; hard deployment budget mapping is future M6.3/M7 work. Region bridge, sector preference, and role heads can optionally add a sparse structural bias to `edge_score`, but they do not yet enforce final bridge budgets, sector budgets, role constraints, or repair rules.

Initialization modes:

- `init_mode="xavier"`: default training-ready initialization.
- `init_mode="kaiming"`: alternate training-ready initialization.
- `init_mode="deterministic"`: test/debug mode for exact reproducibility only.

When `region_id` is provided, scale runs should also pass explicit `num_regions`. If `num_regions` is absent, the scorer falls back to inferring the count from `region_id` for convenience, which requires reading the maximum region id.

The recommended early integration path is:

1. Use `support_mode="all"` to maximize score-gradient coverage while validating scorer features.
2. Use `support_mode="topk"` with a fixed cap to validate the deployable sparse query path.
3. Keep the chosen constructor rule fixed across train, validation, and deployment for a given experiment.

M6 tests prove gradients can flow from the coupled C/D/E loss back into scorer parameters in both all-candidate and top-k constructor modes. They do not train the scorer over epochs.

Future M6.3/M7 work must define how structural heads become hard deployment constraints before replacing fixed constructor budgets:

- mapping soft node budgets to integer caps,
- enforcing cross-region bridge budgets,
- enforcing sector diversity constraints,
- converting role/anchor scores into deployable obligations,
- adding any deterministic repair inside the shared forward path.

No soft/hard train/eval mismatch is introduced by M6.2. Structural score bias, when enabled, is applied by the same scorer forward path for every caller state.

## M6.3 Budget Adapter and Structural Diagnostics

M6.3 adds two safe bridges toward structured topology planning:

- `compute_structural_diagnostics(...)` summarizes budget, role, sector, bridge, and structural-bias outputs.
- `expected_budget_to_cap(...)` maps `node_budget_expected` to integer `per_node_budget` caps for constructor experiments.

`expected_budget_to_cap(...)` is deterministic and supports `round`, `floor`, and `ceil` modes with explicit min/max clipping. It detaches by default because integer caps are nondifferentiable. Passing `detach=False` raises `NotImplementedError` instead of pretending that hard caps carry gradients.

When this adapter is used in an experiment, the same adapter mode, min/max caps, and constructor support mode must be used in train, validation, and deployment. It must not become a validation-only repair or threshold path.

M6.3 still keeps the constructor modes explicit:

1. `support_mode="all"` remains the warm-start path for maximum score-gradient coverage.
2. `support_mode="topk"` plus optional `per_node_budget` is the deployable capped path.

The budget head is still soft/diagnostic unless an experiment explicitly feeds detached caps into the constructor. Sector, role, and bridge heads currently bias sparse candidate scores; they are not hard constraints. Full constrained topology planning with learned budget enforcement, sector diversity, bridge quotas, and repair remains future M7/M8 work.

## M6.4 Structural Objective Contract

M6.4 makes the training-readiness contract explicit:

- The main C/D/E loss can train the base edge path and the sector, role, and bridge heads only when those heads affect `edge_score` through structural bias.
- The main C/D/E loss does not train the budget head through `expected_budget_to_cap(...)`; integer caps are detached and nondifferentiable.
- Budget-head learning requires a future differentiable surrogate, curriculum design, or explicit auxiliary objective.

Auxiliary utilities in `src/models/structural_objectives.py` are structural diagnostics, not topology losses:

- `budget_entropy_loss(...)`
- `budget_target_loss(...)`
- `sector_entropy_loss(...)`
- `role_balance_loss(...)`
- `bridge_logit_regularizer(...)`

They must not be confused with the M4 coupled C/D/E objective. They are optional tools for future curriculum experiments and should be reported separately.

`configs/training_smoke.yaml` is a draft, not a trainer. It recommends starting M7 with `support_mode="all"` for broad score-gradient coverage, then moving to a fixed `support_mode="topk"` cap strategy. Any experiment must keep the selected support mode, cap adapter mode, min/max caps, and constructor rule identical across train, validation, and deployment.

M7 training smoke must report:

- cap histogram and saturation counts,
- gradient coverage fraction,
- whether the budget head was trained by auxiliary/surrogate paths,
- sector, role, and bridge gradient presence,
- constructor mode and cap adapter settings.

M9 stability smoke reuses the same constructor forward path for all ablations. `baseline_all` keeps every feasible sparse candidate, while `baseline_topk` uses the same deterministic top-k cap in every caller state. These ablations compare gradient coverage and C/D/E/F movement; they do not introduce a separate validation or deployment topology rule.

M9.2 marks `baseline_all` versus `baseline_topk` as support-change-confounded by default. The capped case changes active support and gradient coverage along with C/D/E/F metrics, so short-run total-loss deltas are not enough to claim a better topology.

## M14 Structural Signal Audit

M14 audits whether structural score bias is large enough to matter before longer training runs. It compares `edge_score_base` with the decomposed `sector_bias`, `role_bias`, and `bridge_bias`, then measures whether row ordering and top-k support change when structural weights are enabled.

The audit preserves the same constructor rule. It never introduces a separate training/evaluation topology path, hard repair path, or dense candidate matrix. If top-k support overlap is near one while structural-to-base score ratio is tiny, the correct conclusion is to refine structural weights or head scaling before making structural claims.

M14.1 makes the audit direction-aware. `structural_signal_observed` means the
heads are visible in scores, rankings, or gradients; it does not mean they
improve topology quality. The report compares all-structural and structural-zero
variants under both `support_mode="all"` and `support_mode="topk"`. The all-mode
path changes sparse row-softmax weights without dropping candidates, while the
top-k path can change selected membership and is therefore support-confounded.

Main C/D/E gradients are reported separately from auxiliary structural
gradients. Budget-head gradients from auxiliary objectives must not be treated as
evidence that detached hard integer caps train the budget head through the main
constructor path.

M14.2 adds a disable-policy audit for structural score-bias weights. The sweep
evaluates sector, role, and bridge weights in both all-candidate and top-k
constructor modes using the same forward rule. All-mode may tolerate small
nonzero structural weights when primary C/F/D/E deltas are negligible. Top-k is
stricter because structural bias can change selected support; if no nonzero
top-k setting avoids primary-loss, failure, C, and support degradation, the safe
policy is to disable structural score bias in top-k runs and redesign or
recalibrate the harmful heads before M15/M7-style follow-up runs.

The budget head remains auxiliary and diagnostic in this policy. It should not
be treated as useful for hard caps until a differentiable budget surrogate or a
separate validated curriculum objective is introduced.

M15 applies the calibrated policy in selected follow-up runs. All-candidate
mode tests low structural weights `sector=0.01`, `role=0.01`, and
`bridge=0.01` against an all-zero baseline. Top-k mode uses zero structural
weights as the deployable default. The old `0.1/0.1/0.1` top-k structural
setting may appear only as an unsafe reference and must not be promoted. The
calibrated curriculum uses low structural weights in the all-candidate phase
and zero structural weights in the top-k phase, using the same constructor
forward rule in each phase.

M15.1 keeps the constructor policy unchanged and only hardens the evidence
summary. The short tier is explicit and aggregates the calibrated comparisons
across seeds. Low all-mode structural weights are still candidate-only unless
every successful short-tier seed passes the primary C/F/D/E non-worse gate.
Top-k structural bias remains disabled by default. If the high-weight top-k
reference ever stops being stably unsafe, the correct action is a renewed
structural-weight audit, not promotion of nonzero top-k structural weights.

M15.2 separates evidence artifacts by tier. Smoke-tier reports are written to
`calibrated_structural_followup_smoke.*` and are used only for path validation.
Short-tier reports are written to `calibrated_structural_followup_short.*` and
are the minimum evidence source allowed to authorize M16 selected training
smoke. Running the smoke target in agent-check must not overwrite the short
artifact. Even with short evidence, top-k structural bias remains disabled until
a future calibration finds a safe nonzero top-k policy.

M16 consumes that calibrated policy in a two-phase selected smoke. The
constructor first runs all-candidate support with low structural weights to keep
gradient coverage broad, then runs top-k support with structural score-bias
weights set to zero. The same constructor forward rule is used in both phases;
the support mode is an explicit experiment configuration, not a train/eval
branch. Passing this smoke validates that the calibrated topology policy can be
optimized numerically, not that structural heads improve model quality.

M17 is the report-only gate over that selected smoke. It verifies that top-k
structural bias stayed disabled, high-weight top-k structural bias was not used,
and the expected all-to-top-k gradient coverage drop was reported. If selected
short training is recommended, it is still a follow-up smoke over the same
constructor rule, not evidence that top-k structural bias should be enabled.

M16.1 adds C/F/D/E trend snapshots to that same report. These trends help
separate constructor-policy numerical health from actual primary metric
movement; they do not change the all-mode or top-k constructor rules.

M16.2 repeats the same all-mode to top-k selected policy across deterministic
seeds. The constructor rule is unchanged: all-mode keeps full sparse support in
phase 0, while top-k uses zero structural score bias in phase 1. Multi-seed
readiness depends on consistent C/F metric movement and numerical health, not on
enabling nonzero top-k structural bias.

M18 keeps that constructor policy for a longer selected short harness. It is
authorized by `.agent/tmp/selected_training_multiseed.json` rather than by a
single smoke report. The top-k phase still uses structural weights of zero, and
high-weight top-k structural bias remains unused. M18 may show stronger or still
tiny metric trends, but it must not be interpreted as permission to enable
nonzero top-k structural bias without a new calibration audit.

M19 interprets M18 without changing the constructor. It confirms only whether
the selected short report is ready for controlled selected-short ablation. If
top-k structural bias is enabled, or if high-weight top-k structural bias is
used, M19 blocks or requires policy revision. A positive M19 result preserves
the current deployable policy: all-mode low structural weights for warm-start
and top-k zero structural weights for capped support.

M20 controlled selected-short ablation keeps that same constructor policy. The
candidate case uses all-mode low structural weights in phase 0 and top-k zero
structural weights in phase 1; the baseline uses zero structural weights in
both phases. A high-weight top-k case is allowed only as an unsafe reference.
Claim gates explicitly block support-confounded effects and any candidate that
enables nonzero top-k structural score bias.

M20.1 audits the support confounder directly. It compares selected sparse
candidate-index overlap for zero, low, and high structural weights in top-k.
The `support_matched_high_weight` case is an analysis proxy over the observed
high-weight active support, not a new constructor mode and not a training rule.
Top-k structural bias remains disabled unless a future fixed-support or
support-matched study validates a safe nonzero policy.

M20.2 adds the fixed-support version of that audit. The analysis captures
`selected_candidate_index` from a source top-k policy and recomputes sparse
row-softmax weights only on those fixed candidate edges under alternate score
weights. This does not change `TopologyConstructionLayer.forward()` and must not
be used as a train/validation/deployment constructor rule. Its purpose is to
separate support-selection effects from score-redistribution effects before any
future support-proposal redesign.

M21 reviews the accumulated structural-head evidence without adding a
constructor mode. The current policy keeps top-k structural score bias disabled
by default, treats all-mode low structural weights as diagnostic only, and
routes budget heads to auxiliary or future differentiable-surrogate work. Any
future structural redesign must still preserve one shared sparse constructor
rule for train, validation, and deployment within an experiment.

M22 freezes the pure edge-score constructor policy before structural redesign.
The default training/warm-start baseline is `support_mode="all"` with
sector/role/bridge structural weights set to zero. The default deployment
candidate is `support_mode="topk"` with the same zero structural weights, so
top-k support is chosen only from the base edge score. All-mode low structural
weights may remain a diagnostic reference, but they are not a benefit claim and
do not authorize nonzero top-k structural bias.

Budget heads remain fixed, auxiliary, or diagnostic under M22. Detached integer
caps are still nondifferentiable; a future budget surrogate must be designed
explicitly before budget predictions can be treated as trainable constructor
constraints.

M23 tests the frozen constructor policies in a pure edge-score stability
harness. The all-mode baseline keeps every sparse candidate edge and uses zero
sector/role/bridge structural weights. The top-k baseline uses fixed top-k caps
and the same zero structural weights, so support is selected only by the base
edge score. The all-to-topk curriculum carries model parameters from all-mode to
top-k, but it does not mix constructor rules inside a phase and does not
introduce nonzero top-k structural bias.

M24 interprets those constructor results without adding a new constructor mode.
The all-mode result is a warm-start health signal, the top-k result is a
deployment-candidate health signal, and the all-to-topk result is explicitly
support-rule-confounded because active support changes between phases. M24 can
authorize a pure-edge selected short run only when all structural score-bias
weights remain zero and top-k structural bias stays disabled.

M25 runs that pure-edge selected short follow-up without changing the
constructor. The all-mode case keeps all sparse candidates with structural
weights zero. The top-k case uses capped support selected from the base edge
score with structural weights zero. The all-to-topk case carries model
parameters from the all-mode phase into the top-k phase, but the constructor
rule stays fixed within each phase. Structural heads may be present for
diagnostics, but they are not allowed to perturb top-k membership or row-softmax
weights in any pure-edge selected-short case.

M26 interprets those selected-short constructor results without adding a new
constructor mode. The all-mode result is warm-start health, the top-k result is
deployment-candidate health, and the all-to-topk result is support-confounded
when active edges or gradient coverage change between phases. M26 can authorize
only a pure-edge follow-up run or ablation, and only if structural weights
remain zero and top-k structural bias stays disabled.

M27 runs that authorized pure-edge follow-up without changing constructor
behavior. The all-mode case still keeps all sparse candidates, the top-k case
still selects capped support from the base edge score, and the all-to-topk case
still carries parameters across a support-rule switch. Sector, role, and bridge
score-bias weights remain zero throughout. M27 may allow numerical health
claims for these constructor paths, but quality claims remain blocked because
top-k and all-mode support differ and the transition is support-confounded.

M28 packages the current pure edge-score constructor policy without adding a new
constructor mode. The manifest records all-mode warm-start, top-k deployment,
and all-to-topk transition policies with structural score bias disabled. A
complete reference package can authorize only the next controlled pure-edge
extension; it does not imply top-k is better than all-mode, does not promote
structural heads, and does not alter `TopologyConstructionLayer.forward()`.

M29 runs that controlled pure-edge extension without changing constructor
behavior. All-mode keeps all sparse candidates with structural weights zero,
top-k selects capped support from the base edge score with structural weights
zero, and the all-to-topk transition carries parameters across the explicit
support-rule switch. The transition remains support-change-confounded whenever
active support or gradient coverage changes, even if C/F/D/loss movement looks
favorable.

M30 interprets the M29 extension without changing constructor behavior. It
keeps top-k structural bias disabled, requires all sector/role/bridge
score-bias weights to remain zero, and labels all-to-topk movement as a
support-change-confounded transition whenever the support rule changes active
edge count or gradient coverage. The report can authorize only pure edge-score
benchmark packaging; it does not introduce a fixed-support training rule,
structural-head promotion, or direct quality comparison between all-mode and
top-k support.

M31 packages this constructor policy as the benchmark reference without
changing `TopologyConstructionLayer.forward()`. The benchmark config keeps
all-mode warm-start, top-k deployment candidate, and all-to-topk transition
policies explicit with structural score bias disabled. Future comparisons are
blocked if they enable structural bias without an experimental label, mix
train/eval/deploy graph rules, introduce dense NxN graph construction, or omit
support-change confounder reporting.

M32 freezes the formal training-v0 constructor contract in
`configs/training_v0.yaml`. Phase 0 uses `support_mode=all`; phase 1 uses
`support_mode=topk`, carries parameters from phase 0, and requires top-k
structural bias to remain disabled. The same graph rule must be used across
train/eval/deploy for each phase. M32 does not add a fixed-support deployment
mode, dense NxN path, structural-head redesign, or direct physics-loss route.

M33 executes that frozen constructor contract in smoke form. The all-mode phase
keeps every sparse candidate edge for warm-start gradient coverage, then the
top-k phase applies the capped support rule with structural score-bias weights
still zero. The phase switch is explicit and recorded; within each phase the
same constructor forward rule is used. M33 does not add a new constructor mode,
support-freezing deployment path, dense NxN path, or nonzero top-k structural
bias.

M34 repeats the same constructor contract for the formal v0 baseline run with
configured or explicitly overridden step counts. It records active edge count
and gradient coverage per step, but it does not add a new support mode,
fixed-support deployment rule, dense NxN path, random sampling path, or
nonzero structural score bias. Phase 0 remains all-mode and phase 1 remains
top-k with structural bias disabled.

M35 interprets the M34 artifact without changing constructor behavior. It
checks that phase 0 remained all-mode, phase 1 remained top-k, structural
score-bias weights stayed zero, top-k structural bias stayed disabled, and the
same graph-rule contract remained intact before allowing the full M36 baseline
run. It does not add a new constructor mode or make a direct all-mode versus
top-k quality comparison.

M36 runs that full baseline under the same constructor contract. Phase 0 uses
all sparse candidate support for warm-start gradient coverage; phase 1 switches
to deterministic top-k support, carries parameters from phase 0, and keeps all
structural score-bias weights at zero. The report records active edge count and
gradient coverage per step, but M36 adds no fixed-support deployment rule,
dense NxN construction path, random sampling path, or nonzero top-k structural
bias.

M37 interprets the full-run constructor evidence without changing
`TopologyConstructionLayer.forward()`. It verifies that phase 0 stayed all-mode,
phase 1 stayed top-k, structural score-bias weights stayed zero, top-k
structural bias stayed disabled, and the same graph-rule contract remained
intact. It does not compare all-mode and top-k as direct quality rankings; the
phase transition remains support-rule-confounded.

M38 runs the same constructor contract under the `small_realistic` Avalanche
profile. Phase 0 still keeps all sparse candidate support for warm-start
gradient coverage, and phase 1 still uses deterministic top-k support with
structural score-bias weights fixed at zero. The stronger consensus profile
does not introduce a new constructor mode, fixed-support deployment path, dense
NxN path, random sampling path, or nonzero top-k structural bias. A step
override is a fast contract check only and cannot be reported as the full
configured small-realistic run.

M39 interprets the M38 constructor evidence without changing
`TopologyConstructionLayer.forward()`. It verifies that the small-realistic
source stayed all-mode in phase 0, top-k in phase 1, kept structural
score-bias weights at zero, kept top-k structural bias disabled, and completed
the configured step counts. Phase 1 remains a top-k phase training trend, not
proof that top-k is superior to all-mode, and the phase transition remains
support-rule-confounded.

M40 packages the toy and small-realistic formal v0 constructor evidence without
adding a new constructor mode. The formal baseline package keeps phase 0 as
all-mode support, phase 1 as deterministic top-k support, and structural
score-bias weights fixed at zero. Future comparisons must report support mode,
active edge counts, gradient coverage, graph-rule consistency, and
support-change confounders. Comparisons are blocked by dense NxN construction,
train/eval/deploy graph-rule mismatch, unlabeled structural bias, or top-k
structural bias enabled without an explicit experimental label.

M41 extends the same constructor contract to controlled larger-node baseline
checks. The all-mode case keeps all sparse candidates, the top-k case uses the
same deterministic capped support rule, and the all-to-topk case carries
parameters across an explicitly support-confounded phase switch. Structural
score-bias weights remain zero in every case, top-k structural bias remains
disabled, and the extension does not add a fixed-support deployment rule,
repair path, dense NxN construction path, or random topology sampling.

M42 interprets those larger-node constructor results without changing
`TopologyConstructionLayer.forward()`. All-mode remains a warm-start health
path, top-k remains a deployment-candidate health path, and all-to-topk remains
support-change-confounded when active support or gradient coverage changes.
Readiness for the next package stage requires structural weights to stay zero,
top-k structural bias to stay disabled, dense NxN paths to remain absent, and
the train/eval/deploy graph rule match to stay intact.

M43 packages the M40 N=100 and M41/M42 N=500 evidence without changing the
constructor. The v1.1 coverage matrix includes all-mode and top-k rows for
toy and small-realistic profiles, plus N=500 all-to-topk transition rows. It
preserves the same constructor claim boundary: top-k can be described only as
a numerically healthy deployment-candidate path, not as superior to all-mode or
validated for 10k deployment behavior.

M44 plans the next constructor diagnostic without changing
`TopologyConstructionLayer.forward()`. It prioritizes the small-realistic
top-k flatness diagnostic before larger-node toy scale because the current
top-k ambiguity could be caused by saturation, support rigidity, or limited
gradient coverage. Runtime/memory scaling and N=2000 toy top-k remain future
controlled diagnostics, not new support rules or deployment-validation claims.

M45 executes the selected top-k flatness diagnostic without changing
`TopologyConstructionLayer.forward()`. It measures selected candidate overlap,
active-edge change, row/top-k overlap when available, gradient coverage, and
zero-gradient risk context from the existing sparse top-k support path. The
diagnostic can identify support rigidity as a possible flatness driver, but it
does not introduce a fixed-support deployment rule, repair path, sampling path,
dense NxN construction, nonzero structural bias, or a top-k superiority claim.

M46 measures runtime and tensor-memory scaling without changing
`TopologyConstructionLayer.forward()`. It reports candidate and active edge
counts, mean out-degree, selected fraction, active/candidate edge-to-node
ratios, and sparse-to-dense edge ratio against a scalar dense `N*N` proxy. The
proxy is never materialized as a tensor. Top-k cases keep structural bias
disabled and structural weights at zero, so the diagnostic checks sparse
constructor cost rather than adding a new support rule.

M47 runs the same top-k constructor rule at N=2000 for the toy profile. The
constructor still receives sparse candidate edges and selects capped support
from the base edge score with structural score-bias weights fixed at zero. The
report checks active-edge scaling, sparse-to-dense ratio, gradient coverage,
and bottleneck timing without adding a fixed-support deployment mode, repair
path, dense NxN construction path, or nonzero top-k structural bias.

M48 is a report-only interpretation of the M46 and M47 constructor evidence.
It does not call the constructor or add a new support rule; it normalizes the
observed N=100, N=500, and N=2000 toy/top-k rows into a scaling table and
tracks candidate edges, active edges, mean out-degree, sparse-to-dense ratio,
gradient coverage, tensor-memory proxy, and bottleneck stage. A constructor
bottleneck at the largest observed node count recommends a focused M49
constructor diagnostic before a larger-node smoke, while preserving the same
sparse hard-forward rule and zero structural-bias weights.

M49 profiles the current constructor internals without changing
`TopologyConstructionLayer.forward()`. The diagnostic mirrors the existing
sparse validation, source grouping, deterministic top-k row loop, row softmax,
diagnostics, and output packaging stages over synthetic O(Nk) inputs. It
reports whether top-k selection, diagnostics, validation, row softmax, or the
remaining Python row loop dominates the constructor time. If a component
dominates at N=2000 or timing grows superlinearly from N=500 to N=2000, M49
recommends constructor optimization before a larger-node smoke; otherwise it
allows a controlled 5k toy top-k follow-up. It does not add a fixed-support
deployment rule, repair path, dense NxN construction, nonzero top-k structural
bias, or a new constructor mode.

M50 turns the M49 bottleneck diagnosis into a report-only optimization design.
It does not alter `TopologyConstructionLayer.forward()`. The recommended
future path is an optimized segmented top-k implementation only when M49 shows
top-k support selection dominates the N=2000 constructor timing. The design
also records alternatives: keeping the current implementation, CSR-style row
pointer precomputation, a fixed-degree candidate layout fast path for regular
K-per-source graphs, and all-mode warm-start. Any future implementation must
preserve the same hard-forward rule, sparse input/output contract,
deterministic top-k membership, row-softmax normalization, selected-candidate
mapping, diagnostics, and no-gradient behavior for excluded top-k candidates.
It must not allocate dense NxN graph tensors or add stochastic sampling.

M50.1 implements that backend as an opt-in constructor optimization:
`topk_backend="legacy"` preserves the prior implementation and remains the
default, while `topk_backend="segmented_fast"` uses a segmented sparse
row-layout to select capped top-k support without changing the topology rule.
The backend is ignored by `support_mode="all"` after validation because all
feasible sparse candidates remain selected in that mode.

The segmented fast path preserves the same sparse input contract,
train/validation/deployment forward rule, deterministic tie behavior, row
normalization, selected-candidate mapping, diagnostics, duplicate/self-loop
rejection, and excluded-candidate gradient behavior. Legacy-vs-fast equivalence
tests cover uniform and variable candidate degree, isolated rows, caps 1/2/4,
integer index dtypes, equal score ties, all-mode behavior, and an
evaluator/loss backward path.

M51 uses the same constructor rule at N=5000 with `support_mode="topk"` and
`topk_backend="segmented_fast"`. This is a scale smoke for the optimized
backend, not a new topology mode. The constructor still receives sparse
candidate edges, selects capped support from the pure edge score with
structural weights fixed at zero, row-normalizes selected edges, and reports
the same diagnostics. The dense `N*N` value in the report is only a scalar
comparison proxy and is not materialized as a graph tensor.

M52 is report-only and changes no constructor semantics. It consolidates the
N=100/500/2000/5000 toy/top-k scaling rows after segmented_fast, checks that
the constructor bottleneck was reduced relative to the legacy profile, and
records whether the evaluator is now the largest bottleneck. The report keeps
the same sparse candidate-edge contract, same train/validation/deployment
forward rule, structural weights at zero, and top-k structural bias disabled.
It does not claim that N=5000 validates N=10000 or that the evaluator will
scale without a further controlled diagnostic.

M53 applies the same segmented_fast constructor rule in a forward-only
N=10000 toy/top-k smoke. The constructor still consumes sparse candidate
edges, selects capped support from pure edge scores, row-normalizes selected
edges, preserves selected-candidate mapping, and emits the same diagnostics.
The run intentionally performs no backward pass and no optimizer step; the
reported gradient coverage is a support-coverage proxy, not a backward-gradient
claim. The dense `N*N` value remains a scalar comparison proxy only.

M54 profiles the evaluator that consumes the constructor output; it does not
change `TopologyConstructionLayer.forward()`. The profiler keeps the same
segmented_fast sparse top-k support, selected-candidate mapping, topology
weights, distance/LOS attribute lookup, and train/validation/deployment graph
rule. The N=5000 and N=10000 dense `N*N` values in the report remain scalar
comparison proxies and are not allocated as graph tensors.

M55 is a report-only design step for the evaluator's topology query-support
bridge. It does not change `TopologyConstructionLayer.forward()` or the
constructor output contract. Any future query-support optimization must still
consume the constructor's sparse `src_index`, `dst_index`, and
`topology_weight`, preserve selected-candidate attribute mapping upstream,
avoid dense NxN graph tensors, and keep the same evaluator path for
train/validation/deployment unless a future experiment explicitly documents a
single shared alternate path.

M55.1 implements that query-support optimization as an evaluator backend, not
as a constructor change. `TopologyConstructionLayer.forward()` still emits the
same sparse selected support and row-normalized topology weights. The evaluator
may consume those tensors through `query_support_backend="legacy"` or
`query_support_backend="fused_fast"`, but both backends must preserve the same
probability semantics and gradient behavior. `diagnostics_mode` controls only
query-support diagnostic cost and labeling; it does not alter constructor
support, selected-candidate mapping, or the train/validation/deployment graph
rule.

## Future Alternative: Mode A

Future hard-forward/ST construction may use the same hard-forward rule:

```python
A_hard = exact_degree_capped_topk(scores, budgets, constraints)
A_soft = differentiable_sparse_topk_surrogate(scores, budgets, tau)
A = A_hard + (A_soft - A_soft.detach())
```

In Mode A, forward metrics would consume `A_hard`; the soft term would exist only for straight-through gradients. Mode A is not the current M5 default; it remains a future Mode A option.

## M5 TopologyConstructionLayer v1

M5 introduces `src/topology/construction.py` as the first executable shared constructor. It consumes only sparse candidate edges:

- `num_nodes`
- `src_index`
- `dst_index`
- differentiable `edge_score`
- optional feasibility mask
- optional per-node budget or `max_out_degree`

The M5 default is a deterministic sparse soft query distribution. For each source node, the layer keeps feasible sparse candidate edges, optionally applies the same deterministic score top-k cap in every module state, and computes row-normalized nonnegative query weights with a row softmax over the remaining scores. There is no separate validation/deployment threshold path and no Bernoulli edge sampling.

The output is a sparse directed topology object with:

- `src_index`
- `dst_index`
- `topology_weight`
- `selected_candidate_index`
- `selected_edge_mask`
- diagnostics for edge count, active edge count, degree caps, row weight sums, isolated rows, cap hits, effective query degree, selected/unselected candidates, top-k boundary margins, near misses, gradient coverage, unique source counts, per-active-source candidate counts, and zero-gradient-risk rows.

`selected_candidate_index` maps output edges back to candidate-edge attributes such as distance and LOS flags before passing the topology into the M3 evaluator bridge.

Gradient-risk diagnostics are pragmatic warnings:

- `single_selected_row_count`: rows where softmax has one edge and selected score gradients can be zero for row-weight-only objectives.
- `unselected_candidate_count`: feasible candidates excluded by budget or top-k support selection.
- `topk_boundary_margin`: selected boundary score minus the best unselected score for capped rows.
- `near_miss_candidate_count`: unselected candidates just below the top-k boundary.
- `zero_gradient_risk_count`: rows with exactly one selected edge or active top-k exclusion.
- `gradient_coverage_fraction`: selected feasible candidates divided by all feasible candidates.
- `constructor_mode_code`: `1` for top-k support and `0` for all-candidate support.

## Constructor Constraints

- Per-node minimum and maximum degree.
- Directional diversity by sector.
- Cross-cell bridge budget.
- Deterministic repair inside the shared forward path, if repair is required.
- Sparse candidate graph input.
- Duplicate directed candidate edges are rejected by default.
- Self-loops are rejected by default and must be explicitly enabled for diagnostic-only cases.

## Banned Implementation Patterns

- Final topology from independent sigmoid edge probabilities.
- Train on soft graph, validate on thresholded graph.
- Validation-only topology repair.
- Per-edge Bernoulli topology sampling.
- Dense all-pairs candidate path for 10k nodes.

## M56 Constructor Use

M56 uses the existing `TopologyConstructionLayer` with
`support_mode="topk"` and `topk_backend="segmented_fast"` for the N=10000
toy/top-k one-step backward smoke. This is a backend reuse, not a topology rule
change. The constructor still consumes sparse candidate edges, applies the same
hard-forward top-k rule for the smoke path, preserves selected-candidate
mapping, keeps structural score bias disabled, and reports sparse diagnostics
without allocating dense N-by-N graph tensors.

## M57 Constructor Boundary

M57 does not invoke or modify `TopologyConstructionLayer`. It reads the M56
smoke artifact and verifies that the recorded path used
`topk_backend="segmented_fast"`, sparse top-k support, disabled structural bias,
and no dense N-by-N graph construction before interpreting the backward smoke.

## M58 Constructor Boundary

M58 reuses the same sparse top-k constructor policy as M56 for each one-step
learning-rate probe. The constructor receives sparse candidate edges, applies
`topk_backend="segmented_fast"` with fixed zero structural score bias, and
reports active-edge and gradient-coverage diagnostics. The diagnostic does not
add a new support rule, fixed-support training path, repair path, dense N-by-N
construction, stochastic sampling, or nonzero top-k structural bias.

## M59 Constructor Boundary

M59 reuses the same sparse top-k constructor output while diagnosing gradient
sources and loss scale. It does not change `TopologyConstructionLayer.forward()`,
does not freeze support as a new training rule, and does not add repair,
sampling, dense N-by-N construction, or nonzero top-k structural bias. Loss
component and loss-scale probes consume the same selected sparse support and
the same train/validation/deployment graph rule.

## M60 Constructor Boundary

M60 reuses the same N=10000 toy/top-k sparse constructor policy with
`topk_backend="segmented_fast"` and structural score-bias weights fixed at
zero. The loss-scale calibration happens after the evaluator/loss scalar is
computed; it does not alter selected support, row-softmax topology weights,
selected-candidate mapping, duplicate/self-loop handling, or the shared
train/validation/deployment graph rule.

## M61 Constructor Boundary

M61 runs a few calibrated update steps with the same sparse top-k constructor
policy used by M56 through M60. Each step calls `TopologyConstructionLayer`
with `support_mode="topk"` and `topk_backend="segmented_fast"`, consumes sparse
candidate edges, keeps structural score-bias weights at zero, and reports
active-edge count plus gradient coverage. The few-step loop does not add a new
support rule, support freezing path, repair path, dense N-by-N construction,
sampling, or nonzero top-k structural bias.

## M62 Constructor Boundary

M62 reuses the same constructor rule and records how much selected support and
row-softmax topology weights change after each calibrated step. It also applies
deterministic edge-score perturbations before the constructor and optionally
optimizes edge scores directly as a diagnostic variable. These probes preserve
the sparse candidate graph, selected-candidate mapping, row-softmax
normalization, duplicate/self-loop checks, and `segmented_fast` top-k backend.
They do not create a new train/eval/deploy topology rule, dense N-by-N
allocation, stochastic sampling path, or structural-bias reintroduction.

## M63 Constructor Boundary

M63 keeps `TopologyConstructionLayer.forward()` unchanged. It calls the same
top-k constructor with `topk_backend="segmented_fast"` to measure baseline
boundary margins, deterministic edge-score perturbation thresholds, and
score-scale sensitivity. The row-softmax temperature sweep is labeled
analysis-only; it measures selected-weight sensitivity and is not a deployable
constructor rule.

The diagnostic preserves sparse candidate inputs, selected-candidate mapping,
duplicate/self-loop rejection, zero structural-bias weights, top-k structural
bias disablement, and the shared train/validation/deployment topology rule. It
does not add support freezing, repair, dense N-by-N construction, stochastic
sampling, or nonzero top-k structural bias.

## M64 Constructor Boundary

M64 keeps `TopologyConstructionLayer.forward()` unchanged and calibrates only a
scalar multiplier on `edge_score` before the existing top-k call. The forward
sensitivity sweep and one-step update probes use `topk_backend="segmented_fast"`
with sparse candidates, zero structural-bias weights, and top-k structural bias
disabled.

A recommended score scale is a candidate for a follow-up smoke, not a new
deployable topology rule. M64 does not introduce support freezing, row-softmax
temperature as a constructor option, repair logic, dense N-by-N construction,
sampling, or nonzero top-k structural bias.

## M65 Constructor Boundary

M65 applies the M64 score scale before the existing constructor call:
`scaled_edge_score = score_scale * edge_score`. `TopologyConstructionLayer`
still receives sparse candidate edges, uses `support_mode="topk"` and
`topk_backend="segmented_fast"`, preserves selected-candidate mapping and
row-softmax normalization, and keeps structural-bias weights at zero.

The score scale is a diagnostic calibration of constructor input sensitivity.
It is not a new train/eval/deploy topology rule, not support smoothing, not an
analysis-only temperature rule, not repair logic, not stochastic sampling, and
not dense N-by-N construction.

## M66 Interpretation Boundary

M66 does not call `TopologyConstructionLayer` or change its forward rule. It
reads the M65 score-scaled artifact and the M63/M64 sensitivity artifacts to
decide the next diagnostic strategy. Higher score scales remain candidates for
additional controlled probes only; M66 does not make score_scale=10/30/100
training-safe and does not promote row-softmax temperature analysis into a
deployable constructor option.

## M67 Constructor Boundary

M67 calls the existing sparse `TopologyConstructionLayer` with
`support_mode="topk"` and `topk_backend="segmented_fast"`. Score scale is
applied only as `scaled_edge_score = score_scale * edge_score` before the
constructor. Temperature values other than `1.0` are recorded as
analysis-only row-softmax controls and must not be treated as deployable
constructor rules.

The grid preserves selected-candidate mapping, duplicate/self-loop rejection,
per-node degree caps, row normalization, zero structural-bias weights, and the
shared train/validation/deployment hard-forward rule. It allocates no dense
N-by-N graph and performs no stochastic topology sampling.

## M68 Temperature Design Boundary

M68 does not modify `TopologyConstructionLayer.forward()`. It proposes a
future explicit constructor parameter:

```python
row_softmax_temperature: float = 1.0
```

The design applies temperature only to the row-softmax over the selected sparse
support. It must not change candidate graph construction, selected top-k
membership directly, `selected_candidate_index` mapping, max-out-degree caps,
duplicate/self-loop rejection, stochastic sampling policy, or dense graph
allocation behavior. `row_softmax_temperature=1.0` must exactly preserve the
current constructor output.

If a prototype is implemented later, the same parameter value must be used for
training, validation, and deployment. `temperature <= 0` must be rejected, and
`support_mode=all` behavior must be documented and tested before any
temperature-aware smoke can be considered.

## M68.1 Row-Softmax Temperature Prototype

`TopologyConstructionLayer` now exposes `row_softmax_temperature: float = 1.0`
as the formal constructor parameter for selected-edge row-softmax temperature.
The older `temperature` keyword remains only as a compatibility alias. Future
diagnostics and smokes should report `row_softmax_temperature` explicitly.

For `support_mode="topk"`, top-k support membership is selected from the same
edge-score rule as before; `row_softmax_temperature` is applied only after
selection to the selected sparse row-softmax weights. For `support_mode="all"`,
the same temperature is applied to the row-softmax over all feasible sparse
candidate edges. `row_softmax_temperature=1.0` preserves legacy behavior.
Values below 1 sharpen selected-edge weights, values above 1 smooth them, and
values `<= 0` raise `ValueError`.

The parameter is a topology-constructor rule, not a hidden training trick. Any
non-1.0 value must be used identically in training, validation, and deployment.
It does not add stochastic sampling, direct dense NxN graph construction, new
candidate edges, direct top-k membership changes, or direct physics losses.
Constructor diagnostics include the temperature value, whether it affects
weights, whether it affects support, row entropy, and topology-weight min/max
fields.

## M69 Formal Constructor Temperature Grid

M69 uses the formal constructor temperature in the score-scale/temperature
grid. For every grid cell, `scaled_edge_score = score_scale * edge_score` is
passed to `TopologyConstructionLayer(row_softmax_temperature=temperature,
topk_backend="segmented_fast")`; no analysis-only temperature postprocessing is
used.

The M69 report compares each temperature against the same score scale at
`row_softmax_temperature=1.0`. Temperature can change selected sparse
row-softmax weights, query support, and C/F/D/E metrics, but it must not
directly change candidate graph construction, top-k membership selection,
selected-candidate mapping, duplicate/self-loop checks, stochastic sampling
policy, or dense graph allocation behavior. Any recommended candidate only
authorizes a controlled follow-up smoke that uses the same constructor rule in
train, validation, and deployment.

## M70 Constructor Boundary

M70 uses the M69 deployable candidate in a short N=10000 toy/top-k smoke:
`score_scale=30.0` and `row_softmax_temperature=0.5`. The constructor call is
formal:

```python
TopologyConstructionLayer(
    support_mode="topk",
    topk_backend="segmented_fast",
    row_softmax_temperature=0.5,
)
```

The score scale is applied to `edge_score` before construction, and the
temperature is applied only to the selected sparse row-softmax weights. The
smoke must report the temperature, confirm `analysis_only_temperature=false`
and `deployable_constructor_rule=true`, preserve selected-candidate mapping,
avoid dense N-by-N construction, and keep structural-bias weights disabled.

## M71 Constructor Boundary

M71 does not call or modify `TopologyConstructionLayer`. It reads the M70
artifact and verifies that the recorded path used the formal deployable
constructor temperature, not analysis-only temperature. It also checks that
support stayed stable or changed only slightly before recommending any longer
temperature-aware smoke.

The report cannot promote a new constructor rule, change top-k membership
semantics, add support smoothing, enable structural bias, allocate dense N-by-N
graphs, or treat score/temperature candidates as final deployment settings.

## M72 Constructor Boundary

M72 calls the same formal constructor rule as M70 for a longer controlled
diagnostic sequence:

```python
TopologyConstructionLayer(
    support_mode="topk",
    topk_backend="segmented_fast",
    row_softmax_temperature=0.5,
)
```

The score scale is applied before construction as
`scaled_edge_score = 30.0 * edge_score`. Temperature changes only selected
sparse row-softmax weights; it does not directly change top-k membership,
candidate graph construction, selected-candidate mapping, duplicate/self-loop
checks, stochastic sampling policy, or dense graph allocation behavior. M72
reports support overlap and active-edge change at every step and keeps
structural-bias weights at zero.

## M73 Constructor Boundary

M73 does not call or modify `TopologyConstructionLayer`. It reads the M72
artifact and checks that the longer smoke used the formal deployable
constructor temperature, not analysis-only temperature, while preserving
segmented-fast sparse top-k support and disabled structural bias.

The report may recommend another controlled diagnostic, but it cannot promote a
new constructor rule, change top-k membership semantics, add support smoothing,
enable structural bias, allocate dense N-by-N graph tensors, or treat
score_scale=30 / row_softmax_temperature=0.5 as final deployment settings.

## M74 Constructor Boundary

M74 calls the same formal constructor rule as M72 for a 30-step controlled
diagnostic:

```python
TopologyConstructionLayer(
    support_mode="topk",
    topk_backend="segmented_fast",
    row_softmax_temperature=0.5,
)
```

The score scale is applied before construction as
`scaled_edge_score = 30.0 * edge_score`. Temperature changes only selected
sparse row-softmax weights; it does not directly change top-k membership,
candidate graph construction, selected-candidate mapping, duplicate/self-loop
checks, stochastic sampling policy, or dense graph allocation behavior. M74
reports support overlap and active-edge change at every step and keeps
structural-bias weights at zero.

## M75 Constructor Boundary

M75 does not call or modify `TopologyConstructionLayer`. It reads the M74
artifact and checks that the 30-step smoke used the formal deployable
constructor temperature, not analysis-only temperature, while preserving
segmented-fast sparse top-k support and disabled structural bias.

The report may recommend a next diagnostic strategy, but it cannot promote a
new constructor rule, change top-k membership semantics, add support smoothing,
enable structural bias, allocate dense N-by-N graph tensors, or treat
score_scale=30 / row_softmax_temperature=0.5 as final deployment settings.

## M76 Constructor Boundary

M76 deliberately switches the diagnostic constructor policy to:

```python
TopologyConstructionLayer(
    support_mode="all",
    row_softmax_temperature=1.0,
)
```

All feasible sparse candidate edges are active, so hard top-k membership and
`topk_backend` are not part of the reported support rule. The shared runner may
pass an internal backend value only for constructor API compatibility, and the
report records this as `topk_backend="not_applicable"`.

The all-mode warm-start probe preserves sparse candidate construction,
row-normalized nonnegative topology weights, duplicate/self-loop rejection,
and the same train/validation/deployment forward rule for this diagnostic
configuration. It does not allocate dense N-by-N graph tensors, does not sample
edges, does not enable structural bias, and does not turn all-mode into a
deployment claim. Optional non-default `score_scale` or
`row_softmax_temperature` values must be reported explicitly.

## M77 Constructor Boundary

M77 does not call or modify `TopologyConstructionLayer`. It reads the M76
all-mode artifact and checks whether default all-mode
(`score_scale=1.0`, `row_softmax_temperature=1.0`) is stronger or weaker than
the M74/M75 hard-top-k temperature-aware path.

The report may recommend an all-mode score/temperature grid or support
smoothing design, but it cannot promote all-mode as final deployment behavior,
change support membership semantics, add stochastic sampling, allocate dense
N-by-N graph tensors, or treat the default all-mode result as evidence that
calibrated all-mode is generally invalid.

## M78 Constructor Boundary

M78 does not add a constructor mode. It audits existing hard-top-k and all-mode
artifacts and records deterministic capacity probes. Fixed-support probes hold
selected candidate indices fixed and only test selected-weight/logit capacity.
Candidate-oracle probes use deterministic rankings, never random sampling.

Any M78 recommendation is a next diagnostic choice, not a constructor contract
change. The audit must continue to report `dense_nxn_path_absent=true`, avoid
stochastic support exploration, and preserve the rule that train, validation,
and deployment use the same explicit topology construction semantics.

## M79 Constructor Boundary

M79 does not call or modify `TopologyConstructionLayer`. It is a report-only
scorer parameterization design layer over M78 evidence. A future scorer
`score_output_gain`, score normalization, or teacher-distillation path must be
explicitly reported and must preserve the same train/validation/deployment
constructor semantics.

M79 keeps structural bias disabled in the recommended mainline, does not add a
new support mode, does not change top-k membership semantics, does not sample
edges, and does not allocate dense N-by-N graph tensors. Compatibility with
`segmented_fast` top-k and formal `row_softmax_temperature` remains a required
contract for any later scorer prototype.

## M79.1 Constructor Boundary

M79.1 calls the existing sparse constructor only to measure scorer dynamic
range and top-k margin sensitivity. The default diagnostic path is
`support_mode="topk"`, `topk_backend="segmented_fast"`, fused query support,
and structural bias disabled. `row_softmax_temperature` is reported explicitly
when used, and the default raw scorer diagnostic uses `score_scale=1.0`.

The score-scale sweep in M79.1 multiplies `edge_score` before the existing
constructor as a diagnostic what-if probe. It does not implement internal
`score_output_gain`, does not change candidate construction, does not change
top-k membership semantics, does not introduce sampling, and does not allocate
dense N-by-N graph tensors. Any later scorer prototype must keep the same
explicit train/validation/deployment constructor rule.

## M80 Constructor Boundary

M80 introduces `score_output_gain` inside the scorer, not inside
`TopologyConstructionLayer`. The constructor continues to receive a single
sparse `edge_score` vector and applies the same explicit support rule:
`support_mode="topk"`, `topk_backend="segmented_fast"`, deterministic selected
candidate indices, and formal `row_softmax_temperature` over selected edges.

`score_output_gain=1.0` preserves the legacy constructor input exactly.
Non-default gains can change the numeric edge scores presented to the
constructor, but they do not change candidate graph construction, add sampling,
alter top-k semantics, or allocate dense N-by-N tensors. Future experiments
must report both `score_output_gain` and `row_softmax_temperature` so the same
train/validation/deployment topology rule is auditable.

## M81 Constructor Boundary

M81 uses the scorer's explicit `score_output_gain` in short diagnostic
few-step cases, but the constructor boundary remains unchanged. The constructor
still receives a sparse candidate-edge score vector, uses
`support_mode="topk"`, `topk_backend="segmented_fast"`, and applies the formal
`row_softmax_temperature=0.5` row-softmax over selected edges.

M81 must report `score_output_gain`, `row_softmax_temperature`, support overlap,
active-edge change fraction, and sparse-to-dense edge ratio. A support jump is
classified as unstable diagnostic behavior; it is not treated as validated
training progress.

## M82 Constructor Boundary

M82 does not call or modify `TopologyConstructionLayer`. It reads the M81
score-gain artifact and checks that the source smoke used the expected sparse
top-k constructor contract: `topk_backend="segmented_fast"`, structural bias
disabled, `row_softmax_temperature=0.5`, no dense N-by-N path, and no direct
physics loss.

The report may recommend a combined scorer-gain and constructor-temperature
probe, but it cannot silently change support membership semantics, promote
temperature or gain as final deployment behavior, introduce stochastic sampling,
or allocate dense N-by-N graph tensors.

## M83 Constructor Boundary

M83 calls the formal constructor in each combined grid cell. The scorer emits
`edge_score` with explicit `score_output_gain`; optional external `score_scale`
is applied before construction and reported separately; the constructor then
uses `support_mode="topk"`, `topk_backend="segmented_fast"`, and formal
`row_softmax_temperature` over selected sparse edges.

The probe must not change candidate graph construction, top-k membership rules,
sampling policy, train/validation/deployment semantics, or dense allocation
behavior. Any metric movement is interpreted together with support overlap and
active-edge-change diagnostics.

M83 visualization artifacts are generated after the probe from sparse report
rows. Fixed-node trend plots may show topology-weight deltas, query-support
deltas, selected support overlap, and active-edge-change fractions over
diagnostic steps, but they do not call a different constructor rule or replay
the graph with different train/eval/deploy semantics.

## M84 Teacher Probe Constructor Boundary

M84 teacher modes reuse the same sparse constructor contract. Gradient
direction, candidate-oracle, and edge-score-only teacher scores are passed
through `TopologyConstructionLayer` with `support_mode="topk"` and
`topk_backend="segmented_fast"`. The fixed-support teacher explicitly freezes
`selected_candidate_index` and only changes selected-edge row-softmax logits.
No teacher mode changes candidate graph construction, introduces sampling, or
allocates dense N-by-N tensors.

## M85 Teacher Interpretation Constructor Boundary

M85 does not call or modify `TopologyConstructionLayer`. It reads the M84
artifact and verifies that useful teacher evidence did not require structural
bias, stochastic sampling, direct physics losses, or dense N-by-N graph paths.
Wrong-direction or support-jumpy oracle evidence is kept out of the deployable
constructor path.

The report may recommend teacher-guided scorer update design, but it cannot
change top-k membership semantics, add support smoothing, promote candidate
oracle ranking, or alter the shared train/validation/deployment constructor
rule.

## M86 Teacher-Guided Scorer Design Constructor Boundary

M86 also does not call or modify `TopologyConstructionLayer`. It is a design
layer for future scorer-update prototypes only. Any recommended teacher-guided
prototype must preserve the same explicit sparse constructor semantics in
train, validation, and deployment, including segmented-fast top-k support,
formal `row_softmax_temperature`, selected-candidate mapping, duplicate and
self-loop rejection, row normalization, and disabled structural bias in the
recommended mainline.

Candidate-oracle ranking is excluded from the future constructor path because
M85 found it wrong-direction and support-jumpy. Dense N-by-N teacher paths,
stochastic teacher sampling, and hidden train-only teacher rules remain
forbidden.

## M87 Edge-Score Delta Distillation Constructor Boundary

M87 reuses the same sparse constructor path as the current teacher diagnostics:
`support_mode="topk"`, `topk_backend="segmented_fast"`, and formal
`row_softmax_temperature=0.5` by default. The scorer emits sparse candidate
edge scores, the M87 opt-in teacher term may nudge the scorer update, and the
constructor still applies the same deterministic top-k selection plus sparse
row-softmax rule.

The teacher target is an edge-score-space diagnostic target, not a constructor
rule. It does not change candidate graph construction, top-k membership
semantics, selected-candidate mapping, row normalization, duplicate/self-loop
behavior, train/validation/deployment constructor semantics, or sparse O(Nk)
constraints. `candidate_oracle_rank`, stochastic teacher sampling, and dense
N-by-N teacher paths remain blocked.

## M88 Edge-Score Delta Distillation Interpretation Constructor Boundary

M88 does not call or modify `TopologyConstructionLayer`. It reads the M87
artifact and verifies that teacher-guided diagnostics remained outside the
constructor rule: the default sparse top-k path, selected-candidate mapping,
formal `row_softmax_temperature`, row normalization, and train/validation/
deployment semantics are interpreted as source evidence only.

Any M88 recommendation is limited to another opt-in diagnostic teacher strategy
or keeping teacher guidance diagnostic-only. It cannot change support
membership semantics, promote candidate-oracle ranking, add stochastic teacher
sampling, or allocate dense N-by-N graph tensors.

## M89 Teacher Loss Scale Probe Constructor Boundary

M89 reuses the M87 sparse constructor path for every teacher-weight scale:
`support_mode="topk"`, `topk_backend="segmented_fast"`, and formal
`row_softmax_temperature=0.5` by default. The teacher target remains an
edge-score-space diagnostic target; larger teacher weights do not become
constructor rules.

For each teacher weight, M89 runs teacher-off, teacher-on, shuffled-teacher,
and zero-teacher controls through the same constructor rule. It reports support
overlap, active-edge-change fraction, and support-stability status so any
teacher-scale movement can be separated from support jumps. M89 does not
change candidate graph construction, top-k semantics, selected-candidate
mapping, row normalization, duplicate/self-loop behavior, train/validation/
deployment semantics, stochastic sampling policy, or sparse O(Nk) constraints.

## M90 Teacher Loss Scale Interpretation Constructor Boundary

M90 does not call or modify `TopologyConstructionLayer`. It reads the M89
scale artifact and interprets support overlap, active-edge-change fraction, and
support-stability status for each teacher weight.

If high teacher weights move metrics through support jump, M90 treats that as
unstable teacher-scale evidence, not a constructor improvement. The report
cannot change top-k membership semantics, add support smoothing, promote
candidate-oracle ranking, introduce stochastic teacher support, or allocate
dense N-by-N graph tensors.

## M91 Directional Teacher Alignment Constructor Boundary

M91 reuses the same sparse constructor path as M89:
`support_mode="topk"`, `topk_backend="segmented_fast"`, and formal
`row_softmax_temperature=0.5` by default. Directional teacher alignment
operates only in edge-score space before the normal constructor call; it does
not add a teacher-specific support rule.

For each alignment mode, M91 runs teacher-off, aligned-teacher-on for each
configured teacher weight, shuffled-teacher, and zero-teacher controls through
the same constructor rule. It reports minimum support overlap,
active-edge-change fraction, row-top-k overlap, and support-stability status so
any metric movement can be separated from support jumps.

M91 does not change candidate graph construction, top-k membership semantics,
selected-candidate mapping, row normalization, duplicate/self-loop behavior,
train/validation/deployment semantics, stochastic sampling policy, or sparse
O(Nk) constraints.

## M92 Directional Teacher Alignment Interpretation Constructor Boundary

M92 does not call or modify `TopologyConstructionLayer`. It reads M91 support
overlap, active-edge-change, row-top-k overlap, and support-stability fields to
separate teacher/control metric movement from support jumps.

Any M92 recommendation is a next diagnostic choice only. It cannot change
top-k membership semantics, add support smoothing, promote candidate-oracle
ranking, introduce stochastic teacher support, allocate dense N-by-N tensors,
or alter the shared train/validation/deployment constructor rule.

## M93 Score-Head-Only Teacher Update Constructor Boundary

M93 reuses the same sparse constructor path as M91:
`support_mode="topk"`, `topk_backend="segmented_fast"`, and formal
`row_softmax_temperature=0.5` by default. The only difference between
diagnostic cases is optimizer update scope: score-head-only versus full-model.

The teacher signal remains edge-score-space guidance before the normal
constructor call. M93 does not change candidate graph construction, top-k
membership semantics, selected-candidate mapping, row normalization,
duplicate/self-loop behavior, train/validation/deployment semantics,
stochastic sampling policy, or sparse O(Nk) constraints.

## M94 Support Smoothing Design Constructor Boundary

M94 is design-only and does not call or modify `TopologyConstructionLayer`. It
defines the contract for a future sparse deterministic support-smoothing
prototype with legacy defaults:

```text
support_smoothing_mode = "none"
support_smoothing_extra_per_row = 0
support_smoothing_temperature = 1.0
support_smoothing_stage = "hard_topk"
```

`support_smoothing_mode="none"` and `support_smoothing_extra_per_row=0` must
preserve current constructor behavior exactly. Non-default smoothing, if
implemented later, must select only additional sparse candidate edges
deterministically and must use the same train/validation/deployment forward
rule for the configured stage. M94 forbids dense N-by-N expansion, stochastic
support sampling, train-only smoothing, and candidate-oracle ranking.

## M95 Support Smoothing Constructor Prototype Boundary

M95 implements the M94 prototype as an opt-in extension of
`TopologyConstructionLayer`. Legacy defaults remain inert:

```text
support_smoothing_mode = "none"
support_smoothing_extra_per_row = 0
support_smoothing_temperature = 1.0
support_smoothing_stage = "hard_topk"
```

The only active prototype stage is
`support_smoothing_mode="deterministic_topk_halo"` with
`support_smoothing_stage="expanded_sparse_support"` and
`support_smoothing_extra_per_row > 0`. It first applies the existing
deterministic top-k rule to the sparse candidate graph, then adds at most the
configured number of next-ranked sparse candidates per active source row. It
does not allocate dense N-by-N tensors, sample support, use candidate-oracle
ranking, or add a train-only rule.

`support_smoothing_stage="hard_topk"` remains equivalent to the legacy hard
top-k path even if non-default smoothing parameters are supplied. Any expanded
support evidence is diagnostic/curriculum evidence only; deployment-quality
claims still require a final hard-top-k stage with matching validation and
deployment semantics.

## M96 Support Smoothing Smoke Constructor Boundary

M96 uses the M95 expanded sparse support stage only as an explicit diagnostic
configuration. The constructor must still consume sparse candidate edges, add
only bounded per-row halo candidates, and report all smoothing knobs in the
artifact. `support_smoothing_stage`, `support_smoothing_mode`,
`support_smoothing_extra_per_row`, and `support_smoothing_temperature` are
not hidden defaults.

M96 reports support overlap and active-edge-change fraction. If movement is
caused by support jump, the constructor path is not validated and no stronger
training run is authorized.

## M98 Support-Smoothing Stabilization Constructor Boundary

M98 is report/design only and does not modify `TopologyConstructionLayer`.
The human decision authorizes only a conservative support-smoothing
stabilization research branch. Default constructor behavior remains legacy:
`support_smoothing_mode="none"`,
`support_smoothing_extra_per_row=0`,
`support_smoothing_temperature=1.0`, and
`support_smoothing_stage="hard_topk"`.

The next proposed M99 sweep must keep smoothing explicit and opt-in, preserve
the same train/validation/deployment constructor rule for every configured
stage, and keep the candidate graph sparse O(Nk). It must report
`support_smoothing_extra_per_row`, `support_smoothing_temperature`, proposed
support-smoothing weight/blend if available, `score_output_gain`,
`row_softmax_temperature`, `loss_scale`, learning rate, active-edge-change
fraction, support overlap, candidate degree, active degree, runtime, and memory.

M98 does not authorize dense N-by-N expansion, stochastic support sampling,
candidate-oracle ranking, train-only smoothing, structural bias in the default
path, default-path changes, or production-readiness claims. Support-jump and
not-full-10k evidence remain unresolved blockers.

## M99 Conservative Support-Smoothing Constructor Boundary

M99 exercises only the already implemented opt-in
`deterministic_topk_halo`/`expanded_sparse_support` constructor path. It does
not add a support-smoothing weight/blend implementation; that proposed knob is
reported as unavailable and omitted from the executed sweep.

Every M99 case must keep smoothing explicit, sparse, deterministic, and
bounded by sparse candidate edges. The selected candidate must keep
`active_edge_change_total_fraction` below the support-jump blocker threshold,
report candidate and active degree, and preserve the same configured
train/validation/deployment constructor rule. M99 cannot change legacy
defaults or claim full-10k evidence when node count is overridden.

## M100 Full-10k Support-Smoothing Constructor Boundary

M100 runs the M99-selected explicit constructor parameters at
`node_count=10000` with no node-count override. It must report
candidate-edge count, active-edge count, mean candidate degree, mean active
degree, sparse/dense ratio, support overlap, and active-edge-change fraction.

M100 does not add new constructor knobs or alter legacy defaults. If the
full-10k smoke does not retain movement while preserving support stability, the
support-smoothing route is not validated for follow-up training and must stop
for review.

## M101 Support-Smoothing Scale-Transfer Constructor Boundary

M101 reruns the M99-selected explicit support-smoothing constructor parameters
across node counts 100, 500, 2000, 5000, and 10000. It reports candidate-edge
count, active-edge count, mean candidate degree, mean active degree,
sparse/dense ratio, support-smoothing sparse ratio, support overlap, and
active-edge-change fraction for every scale.

M101 does not add constructor parameters, change legacy defaults, alter
train/validation/deployment graph semantics, introduce stochastic support,
allocate dense N-by-N graph tensors, or promote support smoothing as a validated
deployment topology. A scale-transfer diagnosis can recommend only a future
report/design stage.

## M102 Scale-Aware Update Normalization Constructor Boundary

M102 is report/design only and does not modify `TopologyConstructionLayer` or
constructor defaults. It treats the M101 scale-transfer failure as an
update-signal problem to be probed later, not as permission to change hard-top-k
membership, smoothing semantics, sparse candidate construction, or
train/validation/deployment topology rules.

Any future M103 probe must keep normalization explicit and opt-in, report the
normalization basis, exponent, multiplier cap, active-edge change, support
overlap, sparse/dense ratio, and support-smoothing sparse ratio, and include a
no-normalization control. Dense N-by-N support expansion, stochastic support,
candidate-oracle ranking, structural bias defaults, and hidden train-only rules
remain forbidden.

## M103 Scale-Aware Update Normalization Constructor Boundary

M103 executes the M99 support-smoothing constructor candidate unchanged while
varying only the explicit update-normalization multiplier. It reports
candidate-edge count, active-edge count, mean candidate degree, mean active
degree, sparse/dense ratio, support overlap, and active-edge-change fraction
for every node-count and normalization case.

The probe does not modify `TopologyConstructionLayer`, support-smoothing
semantics, hard-top-k membership rules, deployment constructor semantics, or
legacy defaults. A successful M103 candidate can authorize only a follow-up
diagnostic smoke or interpretation report; it cannot promote dense support,
stochastic support, candidate-oracle ranking, structural bias defaults, or
production topology deployment.

## R1/R2 Scale-Law Constructor Boundary

R1 audits constructor-related scale evidence without changing
`TopologyConstructionLayer`. It records stable candidate degree and active
degree, but also reports overdamped smoothing mass at 10k and missing spatial
density evidence. Stable sparse graph degree is therefore not enough for a
production density claim.

R2 is design-only. It may propose future topology smoothing mass, entropy,
row-concentration, or support-jump guard designs, but it does not implement
them. Constructor defaults, hard-forward topology semantics, support-smoothing
behavior, sparse O(Nk) requirements, and train/validation/deployment topology
consistency remain unchanged until a separate human-reviewed R3 branch is
approved.

## R3 Scale-Invariant Loss/Update Constructor Boundary

R3 does not modify `TopologyConstructionLayer`, support-smoothing semantics,
hard-forward topology construction, or constructor defaults. Its candidate
designs may require future constructor diagnostics such as extra-edge mass,
row entropy, support overlap, and active-edge-change fraction, but those are
reporting requirements for a later reviewed branch, not R3 behavior changes.

Any future scale-invariant branch must preserve sparse O(Nk) construction and
the same train/validation/deployment topology rule. Dense N-by-N support,
stochastic support, candidate-oracle ranking, hidden train-only topology rules,
and default-path changes remain forbidden.

## R4 Scale-Invariant Design Review Constructor Boundary

R4 does not modify `TopologyConstructionLayer`, row normalization,
support-smoothing behavior, hard-forward construction, candidate generation, or
constructor defaults. It reviews which constructor-side diagnostics are missing
before a scale-invariant prototype can be selected.

The expected R5 diagnostic must record sparse topology smoothing mass and
query-sensitivity evidence, including `extra_edge_weight_mass_per_row`,
`row_entropy_mean`, `row_entropy_p90`,
`topology_weight_delta_to_query_support_delta_ratio`, and
`active_row_query_sensitivity`. Spatial density fields remain mandatory before
production-density claims, but R4 does not change constructor semantics.

## R5 Topology Smoothing Mass and Query-Sensitivity Constructor Boundary

R5 executes the existing sparse top-k support-smoothing candidate and compares
its selected support with a diagnostic-only no-smoothing top-k pass over the
same candidate graph and scores. That comparison labels base top-k edges versus
support-smoothing extra edges for measurement only; it does not change
`TopologyConstructionLayer`, row normalization, support-smoothing semantics,
hard-forward topology construction, sparse candidate generation, or constructor
defaults.

The diagnostic records extra-edge mass, base top-k mass, row entropy,
effective degree, topology-weight deltas, query-support deltas, active-row
sensitivity, active-failure-node counts, and sparse edge counts while
preserving O(Nk) construction. Dense N-by-N support, stochastic support,
candidate-oracle ranking, hidden train-only topology rules, direct physics
losses, and default-path changes remain forbidden.

## R6 Query-Support Counterfactual Constructor Boundary

R6 constructs the same sparse top-k support-smoothing topology as R5, then
applies deterministic post-construction perturbations to the already selected
`topology_weight` vector. The perturbations preserve `selected_candidate_index`,
`src_index`, `dst_index`, row normalization, and nonnegative weights for valid
cases.

R6 does not modify `TopologyConstructionLayer`, support-smoothing selection,
row-softmax construction, hard-forward topology rules, candidate generation, or
constructor defaults. The counterfactual modes are diagnostic-only and cannot
be treated as a train/eval/deploy topology rule. Dense N-by-N support,
stochastic support, candidate-oracle ranking, hidden train-only topology rules,
direct physics losses, and default-path changes remain forbidden.

## R7 Realistic Density Perturbation Constructor Boundary

R7 uses the existing sparse top-k support-smoothing construction path for toy
and realistic/proxy profiles, then applies the R6 gradient-direction
counterfactual over the already selected sparse support. Candidate support,
support-smoothing selection, row normalization, and constructor defaults remain
unchanged.

The diagnostic records candidate and active degree, available spatial density
fields, profile density status, and cross-profile perturbation capacity. A
`small_realistic` proxy can support a capacity-transfer diagnostic, but it is
not a production density contract by itself. Production-density claims require
an explicit scalable profile contract before any constructor or training-path
change is considered.

## R8 Production Density Profile Constructor Boundary

R8 defines `production_like_density_v0` as a diagnostic profile generator and
feeds its sparse candidate graph into the existing hard top-k constructor. The
profile scales spatial area with node count, uses an explicit candidate radius
and candidate cap, and checks candidate/active degree stability from 100 to
10000 nodes.

R8 does not modify `TopologyConstructionLayer`, row-softmax construction,
support-smoothing behavior, hard-forward topology rules, candidate graph
semantics, or constructor defaults. The profile can become a production-density
contract candidate only after the sanity probe passes and later human review
accepts it.

## R9 Production Profile Perturbation Constructor Boundary

R9 consumes the `production_like_density_v0` sparse candidate graph and uses
the existing top-k support-smoothing constructor. It then perturbs only the
already constructed sparse topology weights for diagnostic counterfactuals.
Support indices, row normalization, and nonnegative weights must remain
preserved for valid cases.

R9 does not modify `TopologyConstructionLayer`, support-smoothing selection,
candidate graph semantics, hard-forward topology rules, or constructor
defaults.

## R10 Production Profile Training Signal Constructor Boundary

R10 uses the existing sparse candidate graph and top-k support-smoothing
constructor for `production_like_density_v0`. Edge-score gradient
counterfactuals are reforwarded through the same constructor path to measure
whether the loss gradient direction can move topology/query metrics.

R10 does not modify `TopologyConstructionLayer`, support-smoothing behavior,
candidate graph semantics, hard-forward topology rules, or constructor
defaults.

## R11 Loss-to-Score Gradient Alignment Constructor Boundary

R11 uses the same sparse candidate graph and top-k support-smoothing
constructor as R10. Component edge-score counterfactuals reforward through the
existing constructor, while topology-weight counterfactuals perturb only the
already constructed sparse support for diagnostic comparison.

R11 does not modify `TopologyConstructionLayer`, support-smoothing selection,
row-softmax construction, hard-forward topology rules, candidate generation, or
constructor defaults. Any detected constructor-Jacobian bottleneck is a
diagnostic classification, not a behavior change.

## R12 Constructor Jacobian Alignment Boundary

R12 decomposes the constructor mapping without changing it. The diagnostic
first freezes the baseline support and recomputes row-softmax weights over that
support to separate score-to-weight mapping from support re-selection. It then
applies the same detached edge-score perturbation through the existing full
top-k/support-smoothing constructor and compares the resulting support and
weight deltas with direct topology-weight perturbations.

R12 also records base top-k versus support-smoothing extra-edge mass transfer,
row-level alignment fractions, near-boundary margins, smoothing-extra ranks,
and whether the useful direction is expressible in the current support. These
measurements are diagnostic-only; R12 does not modify `TopologyConstructionLayer`,
support smoothing, hard-forward topology rules, sparse candidate generation, or
constructor defaults.
