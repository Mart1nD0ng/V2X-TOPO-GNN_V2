# Loss Design

The loss contract consumes only Avalanche reliability, delay, and energy.

## Public Contract

```python
metrics = {
    "C_avalanche": C,
    "D_avalanche": D,
    "E_avalanche": E,
}
```

No other physics or link variable is a direct objective.

## M4 Failure-Domain Coupled Objective

```text
F_i = P_wrong_i + P_undecided_i
F_mean = mean(F_i)
F_tail = p90(F_i), conservative softmax_tail(F_i), or max(F_i)

L_R_mean = softplus((log(F_mean + eps) - log(F_target)) / tau_R)
L_R_tail = softplus((log(F_tail + eps) - log(F_tail_target)) / tau_R)
L_R = L_R_mean + lambda_tail * L_R_tail

L_D_mean = softplus((D_mean - D_target) / tau_D)
L_D_tail = softplus((D_p90 - D_p90_target) / tau_D)
L_D = L_D_mean + lambda_delay_tail * L_D_tail

L_E_mean = softplus((E_mean - E_target) / tau_E)
L_E_tail = softplus((E_p90 - E_p90_target) / tau_E)
L_E = L_E_mean + lambda_energy_tail * L_E_tail

L_total = w_R * L_R + w_D * L_D + w_E * L_E
```

High `C_avalanche` is desirable. Reliability is treated as a constraint in the failure domain, not as `-C`. If both mean and tail failure are below target, `w_R` may weaken to a configured floor so delay and energy can dominate. If either mean or tail failure is above target, the reliability barrier remains active.

M4.1 makes the reliability tail gate conservative. The default tail mode is `max`; `softmax_tail` is a smooth maximum,

```text
F_soft_tail = tau * logsumexp(F_i / tau)
```

with no `log(num_nodes)` normalization. The older normalized smooth average behavior is only available as `softmean_tail` and must not be used as the default reliability gate. A single unreliable node must keep reliability pressure active even when `F_mean` is below target.

Failure-domain inputs are validated as probabilities. `node_failure_probability`, `node_p_wrong_decision`, `node_p_undecided`, and `node_p_correct_decision` must be finite and in `[0, 1]`; `P_wrong + P_undecided` must be at most one. Only tiny numerical roundoff within tolerance is clamped. Large invalid evaluator outputs raise `ValueError` instead of being silently corrected.

M4.2 validates the end-to-end evaluator-to-loss path before topology construction. `evaluate_v2x_graph_consensus(...)` outputs must feed `compute_coupled_loss(...)` directly, produce finite C/D/E barrier components, and carry gradients through topology-relevant evaluator inputs such as sparse query weights, distances, and initial node preferences. This is a validation harness only; it does not add an optimizer loop or topology constructor.

## M7 Smoke Loss Use

M7 uses the M4 coupled loss without changing the public objective contract. The smoke harness may add structural auxiliary objectives from `src/models/structural_objectives.py`, but those are reported as `L_aux_structural` and remain separate from `L_R`, `L_D`, and `L_E`.

The first smoke phase uses `support_mode="all"` so sparse candidate scores receive broad row-softmax gradients. The `support_mode="topk"` smoke checks the deployment-like cap path and must report lower gradient coverage when candidates are excluded. The budget head is not trained by hard integer caps; a budget-head gradient is valid only when an explicit auxiliary budget objective is enabled.

Reliability remains failure-domain and tail-aware. C/D/E metrics are not replaced by link, SINR, BLER, HARQ, coverage, path-loss, or channel-success objectives.

M7.1 adds diagnostics around this loss use without changing the objective. Reports include loss trends, C/F/D/E deltas, finite metric counts, pre/post gradient-clipping norms when clipping is configured, and per-head gradient maxima for edge path, sector, role, bridge, and budget heads. Loss decrease is useful evidence but not guaranteed in a tiny smoke.

The `small_realistic` Avalanche smoke profile is an explicit check over less toy-like consensus parameters. It is separate from the fast default profile so normal contract checks remain quick. Budget-head gradients remain valid only through auxiliary objectives or future differentiable surrogates; detached hard caps still do not transmit budget gradients.

M7.2 keeps this loss boundary during curriculum runs. Phase-specific curriculum fields can enable a budget auxiliary objective in the top-k phase, but the M4 C/D/E terms remain unchanged. Reports must identify whether phase 1 budget-head gradients are active and must keep those gradients attributed to `L_aux_structural`, not to detached integer caps or hidden physics diagnostics.

M8 scale smoke consumes the same M4 coupled loss only as an evaluation signal for runtime, memory proxy, and gradient-health measurements. It adds no new objectives and performs no optimizer step. M8.1/M8.2 keep the fast tier in normal checks and leave micro/medium/large tiers explicit. Scale reports compare all-candidate versus top-k support by active edge count and gradient coverage while keeping reliability in the failure domain.

M9 stability smoke continues to use the same M4 coupled loss and optional structural auxiliaries from M7. Ablation reports may compare structural score bias and budget auxiliary settings, but they do not add new loss terms. Loss decrease is reported as a diagnostic, while finite C/D/E/F metrics, finite gradients, and parameter movement are the required stability checks.

M9.1 tiers keep that interpretation explicit. The smoke tier is a contract check. The short and analysis tiers add ablations and status reporting, but they are still harnesses. A lower total loss in `budget_aux_off` can simply mean the auxiliary budget penalty was removed; it is not evidence of a better C/D/E topology unless the failure, delay, energy, active-edge, and gradient-coverage metrics support that interpretation.

M9.2 reports `L_primary = weighted_L_R + weighted_L_D + weighted_L_E` separately from structural auxiliary terms. This does not change optimization; it only makes reports distinguish public C/D/E objective movement from auxiliary curriculum effects. Ablation tables must not interpret auxiliary removal, top-k support changes, or tiny structural-bias deltas as model-quality conclusions without matching C/F/D/E evidence.

M9.3 readiness uses numerical health, not monotonic loss decrease, as the blocker. Increasing or flat loss in a tiny deterministic smoke can still be conditionally ready if losses and gradients are finite, parameters move, and no exploding/zero-gradient diagnostics fire.

M10 reuses the same M4 loss and M9 interpretation layer. It records `total_loss`, `L_primary`, L_R/L_D/L_E trends, and structural auxiliary trends, but it does not introduce new objectives or direct hidden-physics losses.

M10.1 adds an experiment manifest and report decision summary only. The readiness fields now distinguish the minimal experiment contract from readiness for the next controlled ablation stage. They do not change the coupled loss and they do not imply production training readiness.

M11 controlled ablations still use the same M4 coupled loss and optional structural auxiliary terms. The runner toggles support mode and structural score-bias weights through configuration only. It reports `L_primary` separately from auxiliary terms so `budget_aux_off` cannot be misread as primary C/D/E improvement when total loss drops because an auxiliary penalty was removed.

M11.1 strengthens that reporting by adding paired deltas against same-seed `baseline_all` and multi-seed sign consistency. A lower `total_loss` is not enough for a primary benefit claim when auxiliary terms differ. Claims must be supported by primary C/F/D/E movement, effect magnitude, and consistency, and must not be blocked by support-change or auxiliary-removal confounders.

M16 uses the same M4 C/D/E loss inside a two-phase selected smoke. The selected
policy disables structural auxiliary weights and budget-head training in the
smoke path so the reported movement is dominated by the primary coupled
objective. Phase 0 uses low structural score bias in all-candidate support;
phase 1 disables structural score bias in top-k support. This does not add any
direct link, SINR, BLER, HARQ, coverage, path-loss, or channel-success loss.

M17 interprets the M16 report without changing the loss. It checks whether
losses and gradients were finite, parameters moved, phase loss increases stayed
within tolerance, and available C/F/D/E trends did not worsen. If C/F/D/E trend
deltas are missing, the report remains a numerical-health diagnostic and is not
promoted as primary-metric evidence. The next allowed stage is selected short
training smoke only; M17 does not add objectives or direct hidden-physics
losses.

M16.1 records the C/F/D/E trend inputs that M17 needs. The selected training
smoke now reports per-phase initial, final, and delta values for C, F, D, and E,
plus failure-tail, D-p90, and E-p90 when the evaluator/loss path exposes them.
These are diagnostics over the existing C/D/E failure-domain objective, not new
loss terms.

M16.2 aggregates those same diagnostic deltas across deterministic seeds. It
does not add objectives or change the M4 loss. Consistent tiny C/F improvement
across seeds can authorize only M18 selected short training smoke; direct
link/SINR/BLER/HARQ/coverage losses remain forbidden.

M18 selected short training still uses the same M4 C/D/E coupled objective and
the same calibrated two-phase policy. It only increases the selected step count
and aggregates per-seed C/F/D/E/loss trends. The report distinguishes
consistent improvement from consistent tiny improvement by comparing mean
effect magnitudes against the M16 tiny threshold. It does not add any new
objective and still forbids direct link, SINR, BLER, HARQ, coverage, path-loss,
or channel-success losses.

M19 is report-only. It interprets the M18 C/F/D/E trend aggregates and does not
change optimization. A readiness signal for M20 requires favorable C and F
movement, non-worse D/E, finite losses and gradients, and the same calibrated
top-k zero-structural policy. It does not add auxiliary objectives or hidden
physics losses.

M20 controlled selected-short ablation continues to use the same M4 C/D/E
coupled objective. It compares the calibrated selected policy with a
zero-structural baseline and a high-weight top-k unsafe reference. The report
separates primary-loss and C/F/D/E deltas from support and gradient-coverage
changes. A lower total loss is not enough for a claim, and the unsafe reference
cannot be promoted even if a short-run loss appears favorable.

M22 baseline consolidation does not add objectives and does not run optimizer
steps. It freezes pure edge-score all-mode and top-k policies after the M21
structural review. Structural heads are diagnostics or redesign targets, and
budget heads remain auxiliary or future-surrogate-only. The public objective
contract remains C/D/E; no structural-head benefit or budget-cap gradient claim
is introduced by the consolidation report.

M23 runs optimizer smoke steps for the pure edge-score baseline policies, but it
does not add new loss terms. The optimized objective remains the existing
coupled C/D/E loss from `compute_coupled_loss`, with structural auxiliary
weights disabled and budget strategy fixed. M23 compares loss and C/F/D/E trends
only as stability diagnostics; it does not treat top-k lower loss as automatic
model-quality improvement because support and active-edge counts differ.

M24 is report-only and introduces no objective. It interprets M23 loss and
C/F/D/E movement under conservative claim gates. The all-to-topk curriculum can
show a large loss decrease, but that decrease is support-rule-confounded and is
not a primary objective improvement claim. M24 readiness for M25 depends on
finite C/D/E loss behavior and pure edge-score policy health, not on adding any
direct link/SINR/BLER/HARQ/coverage loss.

M25 runs the pure edge-score selected short harness under the same M4 coupled
C/D/E objective. Structural auxiliary weights are disabled and sector/role/bridge
score-bias weights are zero, so structural heads cannot affect `edge_score`.
The report aggregates C/F/D/E and loss trends across seeds as baseline health
diagnostics. It does not add a new objective and does not turn all-to-topk
support-rule changes into model-quality claims.

M26 is report-only and introduces no objective. It interprets M25 C/F/D/E and
loss movement under conservative claim gates. The all-to-topk selected-short
case may show large loss and metric movement, but that movement remains
support-rule-confounded when active support or gradient coverage changes between
phases. M26 readiness for M27 depends on pure edge-score numerical health, not
on adding direct link/SINR/BLER/HARQ/coverage loss.

M27 runs additional pure edge-score follow-up optimizer smoke under the same M4
coupled C/D/E objective. It does not add new loss terms, structural-head
objectives, schedulers, or training infrastructure. Sector, role, and bridge
score-bias weights remain zero, so structural diagnostics cannot affect
`edge_score`. M27 reports health claims separately from quality claims:
finite gradients, parameter movement, and non-worse C/F/D trends can support
follow-up-path health, but top-k versus all-mode or all-to-topk loss movement
remains support-rule-confounded and cannot be treated as model-quality proof.

M28 is report-only and introduces no objective. It packages M22-M27 pure
edge-score evidence, records the C/D/E health claim ledger, and freezes the
manifested baseline policy for the next controlled stage. It does not rerun
optimization, add auxiliary losses, add hidden physics objectives, or promote
structural-head benefits. `ready_for_M29` depends on existing contract health
and structural score-bias being disabled, not on any new loss design.

M29 runs that controlled pure edge-score baseline extension with the same M4
C/D/E coupled objective. It does not add new loss terms, auxiliary structural
objectives, structural-head redesign, or hidden physics objectives. Sector,
role, and bridge score-bias weights remain zero in all-mode, top-k, and
all-to-topk extension policies. C/F/D/E and loss deltas are reported as
numerical-health diagnostics; top-k versus all-mode and all-to-topk movements
remain support-rule dependent and cannot be treated as model-quality proof.

M30 is report-only over the M29 extension artifact. It does not add optimizer
steps, loss terms, auxiliary structural objectives, or hidden physics
objectives. The interpretation gate can mark M29 as ready for pure edge-score
benchmark packaging only when C/F/D/E health remains finite and the pure
edge-score policy is intact; all-to-topk deltas remain support-rule-confounded
and quality claims remain blocked.

M31 is also report-only. It packages the pure edge-score benchmark policy and
future comparison contract without adding objectives, optimizer steps,
auxiliary structural losses, hidden physics losses, or structural-head
redesign. Future models must preserve the public C/F/D/E reporting contract and
must explicitly report absence of direct link/SINR/BLER/HARQ/coverage loss
terms before comparison against the benchmark.

M32 freezes `configs/training_v0.yaml` without adding a new objective. The
formal v0 config keeps the M4 coupled C/D/E objective targets explicit, records
`tail_mode=max`, requires L_R/L_D/L_E reporting, and forbids direct link/SINR/
BLER/HARQ/coverage losses. Structural score-bias weights are zero and
structural heads are not trainable in the baseline, so no auxiliary structural
loss is promoted into the main objective.

M33 executes that config with the existing M4 coupled C/D/E loss only. The
smoke records L_R/L_D/L_E and C/F/D/E trends for both all-mode warm-start and
top-k candidate phases, but it does not add an objective, auxiliary structural
loss, direct physics loss, scheduler, checkpoint, or model-selection rule.
Support-rule changes between phases remain interpretation context rather than
model-quality proof.

M34 runs the same frozen formal v0 objective for the baseline run and records
per-step scalar trends for total loss, L_R/L_D/L_E, and C/F/D/E. It still adds
no direct link/SINR/BLER/HARQ/coverage loss, no auxiliary structural objective,
no scheduler, no checkpointing path, and no model-selection rule. Longer
baseline-run deltas remain numerical-health evidence only; all-mode versus
top-k differences remain support-rule dependent.

M35 is report-only and adds no objective. It interprets M34 loss and C/F/D/E
trends against conservative gates before authorizing the full baseline run. A
short step override can be a readiness caution, but it does not create a new
loss term or allow claims that the configured full run, model quality, or
top-k superiority has already been validated.

M36 executes the configured formal v0 baseline with the same M4 coupled C/D/E
objective and records per-step scalar loss, L_R/L_D/L_E, and C/F/D/E trends.
It does not add auxiliary structural objectives, direct link/SINR/BLER/HARQ/
coverage losses, schedulers, checkpointing, or model-selection logic.
Structural score-bias weights remain zero, so full-run deltas remain
numerical-health evidence for M37 interpretation rather than model-quality or
top-k-superiority proof.

M37 is report-only and adds no objective. It interprets the M36 full-run loss
and C/F/D/E trends with conservative seed-consistency gates. A confirmed toy
baseline can authorize only a small-realistic formal baseline follow-up; it
does not validate a production objective, top-k superiority, or direct hidden
physics losses.

M38 executes that follow-up under the `small_realistic` Avalanche profile with
the same M4 coupled C/D/E objective. It does not add auxiliary structural
objectives, direct link/SINR/BLER/HARQ/coverage losses, schedulers,
checkpointing, or model-selection logic. The report records C/F/D/E and
L_R/L_D/L_E trends so M39 can interpret numerical health under stronger
consensus settings; it still cannot claim model quality or top-k superiority.

M39 is report-only and adds no objective. It interprets M38 loss and C/F/D/E
trends with conservative full-run and seed-consistency gates. Large phase 0
loss and reliability movement can indicate warm-start trainability or
reliability-barrier release, but it does not create a new loss term and does
not prove final model quality, realistic deployment behavior, or top-k
superiority.

M40 is also report-only and adds no objective. It packages the toy and
small-realistic formal v0 baseline evidence while preserving the C/D/E loss
contract and the L_R/L_D/L_E reporting requirements. The package explicitly
forbids interpreting the packaged evidence as a direct link/SINR/BLER/HARQ/
coverage objective, production training validation, top-k superiority, or
model-quality proof. Future comparisons must disclose direct physics loss
absence before they can be compared to the formal v0 baseline package.

M41 runs controlled larger-node formal v0 baseline cases with the same M4
C/D/E coupled objective only. It does not add direct link/SINR/BLER/HARQ/
coverage losses, auxiliary structural objectives, or objective weighting
changes. Its C/F/D/E and L_R/L_D/L_E aggregates are numerical-health
diagnostics; quality claims and top-k superiority claims remain disallowed.

M42 is report-only and adds no objective. It interprets the M41 C/F/D/E,
loss, L_R/L_D/L_E, gradient coverage, and active-edge aggregates as numerical
health evidence for N=500 toy and small-realistic controlled cases. It keeps
quality claims blocked, keeps all-to-topk support changes confounded, and
continues to require direct link/SINR/BLER/HARQ/coverage losses to remain
absent.

M43 is also report-only and adds no objective. The v1.1 package consolidates
M40 and M42 evidence into a coverage matrix and claim ledger while preserving
the M4 C/D/E loss boundary. It can report numerical-health claims for the pure
edge-score baseline, but it cannot convert those diagnostics into quality,
optimality, deployment, production-training, or direct-physics-loss claims.

M44 is report-only and adds no objective. It ranks the M43-recommended
scale/profile diagnostics and selects the small-realistic top-k flatness
diagnostic as the conservative next step when near-flat C/F deltas are present.
This planning decision does not change the C/D/E loss, does not add hidden
physics objectives, and does not turn a future longer or larger diagnostic into
a quality or deployment claim.

M45 runs that diagnostic with the same M4 C/D/E coupled objective and pure
edge-score top-k support rule. It reports L_R/L_D/L_E movement, reliability
gate context, gradient signal, support overlap, and C/F/D/E flatness to
separate saturation, support rigidity, gradient starvation, and loss gating.
It adds no loss term, no direct link/SINR/BLER/HARQ/coverage objective, no
auxiliary structural objective, no scheduler, no checkpoint, and no
model-selection rule. Any M45 health finding remains diagnostic-only and does
not prove model quality or top-k superiority.

M46 adds no objective and performs no production training. It measures runtime,
tensor-memory proxies, sparse edge ratios, and gradient finiteness for the pure
edge-score v0 baseline using one controlled sparse forward/backward diagnostic
step per case. The report continues to prohibit direct link/SINR/BLER/HARQ/
coverage losses, auxiliary structural objectives, schedulers, checkpoints, and
model-selection rules. A scaling smoke pass is an engineering-health signal
only; it does not prove model quality, top-k superiority, production readiness,
or 10k deployment behavior.

M47 keeps the same M4 C/D/E objective and pure edge-score top-k support at
N=2000. It adds no new loss term and no hidden-physics objective. The tiny
optimizer smoke is used only to report finite losses, finite gradients,
parameter movement, and C/F/D/E diagnostic deltas; these remain
numerical-health evidence and cannot be promoted to model-quality,
top-k-superiority, production-training, deployment, or 10k validation claims.

M48 is report-only and adds no objective. It reads the M46 and M47 scaling
artifacts, interprets sparse edge growth, tensor-memory proxy growth, and
runtime bottleneck movement, then recommends the next controlled diagnostic.
Its conclusion cannot alter the M4 C/D/E loss, cannot add hidden physics
objectives, and cannot turn runtime or sparse-memory evidence into model
quality, top-k superiority, production-readiness, deployment, or 10k validation
claims.

M49, M50, M50.1, M51, and M52 keep the same objective boundary. Constructor
profiling, constructor optimization design, segmented_fast top-k backend
equivalence, N=5000 smoke execution, and v1.1 scaling interpretation add no
loss term and no hidden-physics objective. M52 interprets runtime, memory
proxy, sparse edge growth, and bottleneck movement only; it cannot change the
M4 C/D/E loss and cannot promote scaling evidence into model-quality,
production-training, deployment, direct-physics-loss, or 10k validation claims.

M53 through M56 keep that boundary. The 10k forward smoke, evaluator/query
support diagnostics, fused query-support fast path, and M56 one-step backward
smoke do not add objectives and do not expose direct hidden-physics terms as
losses. M56 uses the existing coupled C/D/E objective once for a feasibility
check and reports finite loss, finite gradients, and parameter movement only.
It cannot be used as evidence for model quality, production training,
deployment readiness, top-k superiority, or realistic-profile behavior.

M57 is also report-only and does not alter the objective. It interprets the
M56 coupled-loss result by separating finite backward feasibility from update
magnitude and reliability-target status. A tiny parameter update routes to a
gradient/step-size diagnostic rather than a model-quality or training claim.

M58 still uses the same M4 C/D/E objective. It repeats one-step backward probes
at several learning rates and reports L_R/L_D/L_E values, relative loss
component contributions, gradient norms, and parameter-update scale. It does
not introduce a scheduler, optimizer policy, loss rescaling, hidden physics
objective, or multi-step training. Any recommendation for M59 is based on
diagnostic gradient/update health only.

M59 continues to use the same C/D/E objective and adds no new loss term. It
separately backpropagates total loss, L_R, L_D, and L_E for gradient-source
diagnostics, then multiplies the existing total loss by configured scalar
factors for one-step loss-scale probes. These scalar probes are diagnostic
calibration evidence only; they are not a scheduler, adaptive weighting system,
direct hidden-physics objective, or multi-step training policy.

M60 applies the selected scalar calibration once. It computes
`raw_total_loss` from the unchanged coupled C/D/E objective and uses
`scaled_total_loss = loss_scale * raw_total_loss` for one backward pass only.
The report must keep both values visible. This remains a loss-scale diagnostic,
not a new objective, not GradNorm or PCGrad integration, not a scheduler, and
not permission to optimize link/SINR/BLER/HARQ/coverage terms directly.

M61 repeats that same scalar calibration for a few steps only. The raw C/D/E
loss remains the objective reported for trends, while `scaled_total_loss` is
used only as the backward scalar. M61 is not a new loss function, not adaptive
loss weighting, not a scheduler, not production training, and not permission to
add direct link/SINR/BLER/HARQ/coverage objectives. Any C/F/D/E movement is a
claim-limited smoke trend, not model-quality evidence.

M62 does not introduce a new loss. It reuses the M61 calibrated scalar only to
trace where the existing C/D/E loss signal becomes flat. Direct edge-score
perturbation and edge-score-only optimization probes are sensitivity
diagnostics; they are not additional training objectives, direct physics
objectives, schedulers, or proof that larger learning rates or loss scales solve
the flatness issue.

M63 also adds no objective. It perturbs and rescales edge scores before the
unchanged sparse constructor, and it runs an analysis-only row-softmax
temperature sweep to measure selected-weight sensitivity. These probes only
measure whether the existing C/D/E evaluator metrics can react to score/support
movement. They are not loss terms, not adaptive weighting, not a scheduler, and
not permission to optimize link/SINR/BLER/HARQ/coverage directly.

M64 also adds no objective. It chooses a candidate scalar `score_scale` by
measuring how the unchanged coupled C/D/E loss and evaluator metrics respond
when `edge_score` is multiplied before the existing top-k constructor. The
one-step probe still computes the raw coupled loss and uses the configured
loss scale only as the backward scalar. Score scaling is not a direct physics
loss, not a scheduler, and not evidence that larger loss or learning-rate
values solve metric flatness by themselves.

M65 still adds no objective. It uses the M64 score scale only on the
constructor input, computes the same raw coupled C/D/E loss, and uses
`scaled_total_loss = loss_scale * raw_total_loss` for backward exactly as in
the calibrated smokes. The report keeps raw and scaled loss visible and
compares metric movement to M61. Any improvement is a claim-limited diagnostic
trend, not model-quality evidence or permission to add direct
link/SINR/BLER/HARQ/coverage objectives.

M66 is report-only and adds no objective. It interprets whether the M65
score_scale=3 smoke moved the existing C/D/E loss metrics enough to justify the
next diagnostic. It may recommend a score-scale/temperature grid probe, but it
does not treat larger score scale, learning rate, or loss scale as a solution
by itself and does not authorize direct hidden-physics objectives.

M67 also adds no objective. It evaluates a diagnostic grid over score scale and
row-softmax temperature using the same raw coupled C/D/E loss. The configured
loss scale remains only a backward scalar for the short update probe, and the
post-update reforward reports raw C/F/D/E movement. Temperature probes are
analysis-only controls and do not authorize new loss terms, direct
link/SINR/BLER/HARQ/coverage objectives, or blind longer training.

M68 is report-only and adds no objective. It defines the design contract for a
possible future row-softmax temperature constructor parameter, but it does not
change the coupled C/D/E loss, add adaptive weighting, tune learning rate or
loss scale, or authorize direct link/SINR/BLER/HARQ/coverage objectives.
Temperature remains a constructor sensitivity design candidate until a
prototype preserves the existing loss/evaluator semantics.

M68.1 implements the constructor prototype and still adds no objective.
`row_softmax_temperature` changes only the sparse row-softmax weights emitted
by the constructor. The raw coupled C/D/E loss, loss-scale handling, evaluator
metrics, and forbidden direct physics-objective policy remain unchanged.
Temperature-aware follow-up smokes must report the temperature value and must
not treat it as a hidden loss or training trick.

M69 still adds no objective. It reruns the score-scale/temperature grid with
formal constructor temperature and the same raw coupled C/D/E loss. The
configured loss scale remains only the backward scalar for the short update
probe; post-update reforward reports raw topology/query/C/F/D/E movement.
M69 must not promote learning-rate or loss-scale tuning as a solution by
itself, and it continues to forbid direct link/SINR/BLER/HARQ/coverage loss
terms.

M70 also adds no objective. It runs a few-step smoke with the M69 formal
constructor candidate (`score_scale=30.0`, `row_softmax_temperature=0.5`) and
the same raw coupled C/D/E loss. The configured loss scale remains only a
backward scalar for calibration; post-update reforward reports raw metric
movement. M70 must not be interpreted as long 10k training, production
readiness, or permission to add direct link/SINR/BLER/HARQ/coverage losses.

M71 is report-only and also adds no objective. It interprets M70 by comparing
raw C/F/D/E, topology-weight, and query-support movement against M61 and M65
artifacts. A positive interpretation can authorize only a longer controlled
temperature-aware smoke; it does not change the C/D/E loss, make
score_scale=30 or temperature=0.5 final, or allow direct hidden-physics
objectives.

M72 still adds no objective. It repeats the M70 formal temperature-aware
configuration for a longer controlled diagnostic sequence and reports whether
raw C/F/D/E and reliability-gap movement accumulates. The raw coupled C/D/E
loss remains the reported objective, while the configured loss scale remains
only a backward scalar. M72 does not add adaptive weighting, LR scheduling,
checkpointing, direct link/SINR/BLER/HARQ/coverage losses, or production 10k
training.

M73 is report-only and also adds no objective. It interprets M72 raw C/F/D/E
movement, support stability, and reliability-gap trend against M70/M65/M61
context artifacts. Its reliability-gap projection is a diagnostic extrapolation
only; it does not change the C/D/E objective, add adaptive weighting, authorize
direct hidden-physics objectives, or validate longer 10k training.

M74 still adds no objective. It runs a 30-step controlled diagnostic with the
same formal temperature-aware constructor configuration and the same raw
coupled C/D/E loss. The configured loss scale remains only the backward scalar,
and post-update reforward reports raw metric movement. M74's trend slopes and
reliability-gap projection are diagnostics only; they do not add adaptive
weighting, LR scheduling, checkpointing, direct link/SINR/BLER/HARQ/coverage
losses, or production 10k training.

M75 is report-only and also adds no objective. It interprets M74's raw
C/F/D/E movement, support stability, and reliability-gap projection before
choosing the next diagnostic strategy. A recommendation to pivot toward
all-mode warm-start, support smoothing, temperature-scale refinement, or an
extended smoke does not change the C/D/E objective and does not authorize
direct hidden-physics objectives or production training.

M76 still adds no objective. It changes the diagnostic support policy to
`support_mode=all` for an all-candidate warm-start probe, but keeps the same
raw coupled C/D/E loss and uses the calibrated scalar only for backward. The
probe reports whether all-mode gradients and metric movement are stronger than
the hard-top-k temperature-aware path; it does not add direct
link/SINR/BLER/HARQ/coverage losses, adaptive weighting, checkpointing,
schedulers, model selection, or production 10k training.

M77 is report-only and also adds no objective. It interprets the M76 all-mode
artifact against M74/M75 hard-top-k temperature-aware evidence. A recommendation
for an all-mode score/temperature grid, hard-top-k refinement, support
smoothing, or evaluator sensitivity review does not change the raw C/D/E loss,
does not add adaptive weighting, and does not authorize direct
link/SINR/BLER/HARQ/coverage objectives.

M78 is a convergence root-cause audit and also adds no objective. It decomposes
whether weak C/F/D/E movement is due to evaluator sensitivity, fixed-support
capacity, edge-score training path, candidate graph capacity, loss-gradient
conflict, or GNN score dynamic range. The loss-gradient section records
component norms and optional cosines for `L_R`, `L_D`, and `L_E`; it does not
modify weights, add GradNorm/PCGrad integration, introduce physics losses, or
authorize production training.

M79 is a scorer-parameterization redesign report and also adds no objective.
It may recommend score-output gain, score-head scale diagnostics, edge-score
teacher probes, or score-head optimizer-group probes, but those are future
design choices and not changes to the current coupled C/D/E loss. Any future
teacher signal must be derived from topology-objective evidence and must not
become a direct link/SINR/BLER/HARQ/coverage loss.

M79.1 also adds no objective. It computes the unchanged raw coupled C/D/E loss
and uses the configured `loss_scale` only as a backward scalar for one
diagnostic update. The layer-wise gradient/update table and score-head
sensitivity values are measurements of the current scorer path, not GradNorm,
PCGrad, a score-output gain, or a new auxiliary objective. Score-scale rows in
M79.1 are forward what-if diagnostics and must not be interpreted as direct
loss scaling or a deployable hidden training trick.

M80 also adds no objective. `score_output_gain` multiplies the scorer's final
edge-score output before the unchanged constructor/loss path, while the coupled
C/D/E loss remains the only training objective in the diagnostic probe. The
configured `loss_scale` is still only a backward scalar for controlled
diagnostics. M80 must not be interpreted as direct link/SINR/BLER/HARQ/coverage
loss, as GradNorm/PCGrad integration, or as production training validation.

M81 also adds no objective. It reuses the same raw coupled C/D/E loss and the
same calibrated `loss_scale` backward scalar while testing explicit scorer
`score_output_gain` values over a few diagnostic steps. The gain affects only
the model's emitted edge scores; it is not a direct reliability, link,
SINR, BLER, HARQ, coverage, GradNorm, PCGrad, or auxiliary teacher loss.

M82 also adds no objective. It is a report-only interpretation of the M81
artifact and optional context artifacts. It can recommend a combined
gain/temperature probe, a gain-30 longer smoke, a score-head LR probe, or an
edge-score-teacher probe, but those recommendations do not alter the current
coupled C/D/E loss or authorize direct physics losses, adaptive weighting, or
production training.

M83 also adds no objective. It combines explicit scorer `score_output_gain`
with formal constructor `row_softmax_temperature` while preserving the same raw
coupled C/D/E loss and the same calibrated backward `loss_scale`. The external
`score_scale` remains a reported diagnostic control, not a hidden loss term.
M83 must not be interpreted as a direct link/SINR/BLER/HARQ/coverage objective,
teacher objective, GradNorm/PCGrad integration, or production training
validation.

M83 visualizations are post-hoc report views. Fixed-node step curves expose the
same raw/scaled loss, L_R/L_D/L_E, C/F/D/E, reliability-gap, gradient, and
topology/query movement fields that the JSON already reports. They do not add a
loss component, tune objective weights, or change the backward scalar.

## Gradient Governance

- GradNorm adapts objective weights by matching training rates.
- PCGrad projects conflicting per-objective gradients.
- Logs must include per-objective gradient norms and cosine similarities.

`src/losses/gradnorm.py` and `src/losses/pcgrad.py` provide deterministic utilities only. They do not implement a training loop, optimizer integration, topology constructor, or GNN.

## Banned Direct Objectives

- Link reliability.
- SINR.
- BLER.
- HARQ.
- Coverage.
- Average reliability.

## M84 Teacher Probe Loss Boundary

M84 teacher modes use the existing coupled C/D/E objective for all optimized
diagnostic tensors. Fixed-support logits and edge-score-only tensors are
temporary probe variables; they are not new production parameters and they do
not add a supervised teacher loss to the default training path. Candidate
oracle ranking is deterministic and gradient-derived, not direct
SINR/BLER/HARQ/coverage supervision.

## M85 Teacher Interpretation Loss Boundary

M85 is report-only and adds no objective. It reads M84 teacher deltas and
alignment diagnostics to decide whether a valid teacher direction exists and
whether the bottleneck is in the scorer update path. It must not add an
auxiliary teacher loss, direct link/SINR/BLER/HARQ/coverage loss, adaptive
objective weighting, GradNorm/PCGrad integration, or any default training-path
change.

Any future M86 teacher-guided design must still keep the public C/D/E objective
boundary explicit and must distinguish diagnostic teacher supervision from a
validated training objective.

## M86 Teacher-Guided Scorer Design Loss Boundary

M86 is report-only and adds no objective. It designs possible future
teacher-guided scorer update mechanisms while keeping the coupled C/D/E loss
as the primary objective. The preferred prototype path,
`edge_score_delta_distillation`, is only a proposed stop-gradient auxiliary for
a later M87 prototype; it is not added to the default training path here.

M86 explicitly rejects candidate-oracle promotion, direct SINR/BLER/HARQ/
coverage or link-reliability teacher losses, stochastic sampled teachers, dense
N-by-N teacher tensors, and hidden train-only teachers. Any later prototype
must report teacher weight, teacher mode, teacher scope, and ablations with the
teacher off, on, shuffled, and zeroed before it can be interpreted.

## M87 Edge-Score Delta Distillation Loss Boundary

M87 adds an auxiliary teacher term only inside an explicit diagnostic
prototype. The primary objective remains the unchanged coupled C/D/E loss:

```text
scaled_primary_loss = loss_scale * raw_total_loss
teacher_loss = teacher_weight * mse(edge_score, teacher_edge_score.detach())
diagnostic_total_loss = scaled_primary_loss + teacher_loss
```

The teacher target is derived from the local edge-score gradient direction and
is detached before the diagnostic loss is formed. M87 does not add a default
training objective, does not use second-order gradients by default, does not
optimize link reliability, SINR, BLER, HARQ, coverage, or path loss, and does
not promote candidate-oracle ranking. Teacher-off, teacher-on,
shuffled-teacher, and zero-teacher controls must be reported before the result
can be interpreted.

## M88 Edge-Score Delta Distillation Interpretation Loss Boundary

M88 adds no objective and runs no optimizer steps. It reads the M87 loss terms
and compares teacher-loss magnitude against both the raw coupled C/D/E loss and
the scaled primary loss used for backward in the diagnostic prototype.

If the teacher term is orders of magnitude smaller than the scaled primary
loss, M88 may recommend a teacher-loss scale probe. If the teacher term is
finite and nontrivial but still does not alter edge-score, topology, query, or
C/F/D/E movement, M88 may recommend a directional alignment or score-head-only
teacher update diagnostic. It must never recommend adding teacher loss to the
default training path from M87 evidence alone.

## M89 Teacher Loss Scale Probe Loss Boundary

M89 keeps the same primary objective as M87:

```text
scaled_primary_loss = loss_scale * raw_total_loss
scaled_teacher_loss = teacher_weight * teacher_loss
diagnostic_total_loss = scaled_primary_loss + scaled_teacher_loss
```

The teacher term is present only inside the explicit M89 diagnostic probe.
M89 sweeps larger `teacher_weight` values to test whether the M87
stop-gradient edge-score target can become numerically visible while C/D/E
remains the primary objective. It reports both unweighted teacher loss and
scaled teacher loss, plus ratios against raw and scaled primary loss.

M89 does not add teacher loss to default training, does not add direct
link/SINR/BLER/HARQ/coverage objectives, does not introduce GradNorm/PCGrad
integration, and does not authorize a production teacher objective.

## M90 Teacher Loss Scale Interpretation Loss Boundary

M90 is report-only and adds no objective. It reads M89's unweighted teacher
loss, scaled teacher loss, scaled primary C/D/E loss, diagnostic total loss,
and teacher/control metric deltas. It decides whether increasing the MSE-style
edge-score delta distillation weight found a safe useful scale.

If higher teacher weights only become visible by causing support instability,
M90 must not recommend increasing teacher weight further or adding the teacher
term to default training. A recommendation for directional alignment or
score-head-only teacher diagnostics still preserves C/D/E as the primary
objective and forbids direct link/SINR/BLER/HARQ/coverage losses.

## M91 Directional Teacher Alignment Loss Boundary

M91 keeps C/D/E coupled loss as the primary objective and adds only an
explicit opt-in diagnostic alignment term:

```text
scaled_primary_loss = loss_scale * raw_total_loss
diagnostic_total_loss = scaled_primary_loss + teacher_weight * alignment_loss
```

The teacher direction is `normalize(-grad_edge_score).detach()` from the
current C/D/E loss. M91 tests scale-normalized `cosine_delta_alignment`,
`sign_alignment`, and `normalized_mse_direction` losses rather than the
unnormalized tiny-delta MSE objective used by M87/M89.

The alignment term is never added to default training. M91 does not introduce
direct link/SINR/BLER/HARQ/coverage objectives, does not use
candidate-oracle ranking, does not use stochastic teachers, and does not
authorize a production teacher objective.

## M92 Directional Teacher Alignment Interpretation Loss Boundary

M92 adds no objective and runs no optimizer steps. It reads M91's reported
`scaled_primary_loss`, alignment-loss movement, C/F/D/E deltas, and control
outcomes to decide whether another diagnostic teacher path is justified.

If M92 recommends score-head-only teacher updates or Jacobian alignment, that
recommendation is not a loss change. The coupled C/D/E objective remains the
primary objective, teacher guidance stays opt-in and ablated, and direct
link/SINR/BLER/HARQ/coverage losses remain forbidden.

## M93 Score-Head-Only Teacher Update Loss Boundary

M93 uses the same diagnostic alignment term as M91, but restricts one update
scope to the scorer's `edge_score_head` parameters:

```text
scaled_primary_loss = loss_scale * raw_total_loss
diagnostic_total_loss = scaled_primary_loss + teacher_weight * alignment_loss
```

The score-head-only restriction is an optimizer-parameter scope, not a new
objective. Full-model updates remain comparison controls. The alignment term is
not added to default training, C/D/E remains the primary objective, and direct
link/SINR/BLER/HARQ/coverage losses remain forbidden.

## M94 Support Smoothing Design Loss Boundary

M94 adds no objective and runs no optimizer steps. It only designs a future
constructor-side support-smoothing prototype. The C/D/E coupled loss remains
unchanged, and no teacher, direct physics, GradNorm, PCGrad, link/SINR/BLER/
HARQ/coverage, or candidate-oracle objective is introduced.

Any later prototype must prove that smoothing changes only sparse constructor
support semantics through explicit parameters, not the public C/D/E loss
contract.

## M95 Support Smoothing Constructor Prototype Loss Boundary

M95 changes only opt-in constructor support selection. It does not add or alter
any loss term. The coupled C/D/E objective remains the public training loss
contract, and M95 introduces no teacher objective, direct link/SINR/BLER/HARQ/
coverage objective, GradNorm/PCGrad integration, candidate-oracle ranking, or
physics-supervised shortcut.

The M95 backward-integration check exists only to prove that gradients remain
finite through scorer-score tensors, sparse constructor weights, the analytic
Avalanche evaluator, and the existing C/D/E loss. It is not training or model
quality evidence.

## M96 Support Smoothing Smoke Loss Boundary

M96 keeps the coupled C/D/E objective unchanged. It combines existing explicit
configuration knobs (`score_output_gain`, `row_softmax_temperature`,
`loss_scale`, and `learning_rate`) with opt-in support smoothing, but does not
add teacher loss, direct physics loss, candidate-oracle objectives, GradNorm,
PCGrad, or a new mathematical objective.

If M96 movement is support-jumpy, the result is treated as constructor/support
instability rather than a validated loss improvement. The next step must be a
sensitivity review, not a stronger training claim.

## M97 Loss Evaluator Sensitivity Review Boundary

M97 adds no loss term and changes no mathematical objective. It reviews
existing artifacts for C/F sensitivity to query support, L_R shape,
L_R/L_D/L_E conflicts, candidate graph capacity, and evaluator threshold
behavior.

If the review concludes that stable movement has not been validated, the
contract-compliant outcome is a human-review gate before any objective,
evaluator-threshold, or candidate-capacity change. Direct physics losses remain
forbidden.

The post-M97 human-review gate package adds no loss term and authorizes no
stronger training. It records the M97 stop condition and keeps any objective or
evaluator-threshold change behind explicit human review.

## M98 Support-Smoothing Stabilization Boundary

M98 adds no loss term and changes no mathematical objective. The human decision
authorizes only support-smoothing stabilization research, beginning with a
conservative parameter sweep. The design keeps `loss_scale`, learning rate,
score gain, row-softmax temperature, support-smoothing temperature, and a
proposed support-smoothing weight as explicit sweep knobs rather than hidden
training behavior.

M98 does not authorize direct physics losses, objective/evaluator changes,
production training, stronger training claims, default-path changes, structural
bias in the default path, dense support paths, stochastic support, or train-only
rules. Support-jump and not-full-10k evidence remain unresolved blockers.

## M99 Conservative Support-Smoothing Sweep Loss Boundary

M99 runs optimizer smoke steps only through the existing C/D/E objective and
existing explicit scale/learning-rate knobs. It does not introduce a new loss
term, teacher objective, direct physics loss, evaluator change, candidate
oracle, GradNorm/PCGrad integration, or default-path behavior change.

The M99 selection gate treats support-jump reduction as a diagnostic stability
condition, not as a new objective. A selected candidate may authorize a future
full-10k diagnostic smoke, but it does not authorize production training,
stronger training claims, or any objective/evaluator modification.

## M100 Full-10k Support-Smoothing Loss Boundary

M100 uses the unchanged C/D/E coupled objective with the explicit M99-selected
`loss_scale` and learning rate. It adds no teacher loss, direct physics loss,
candidate-oracle objective, evaluator change, or adaptive balancing method.

If the full-10k smoke loses C/F or reliability-gap movement, that is interpreted
as failed diagnostic transfer from the 500-node sweep, not as permission to
change the mathematical objective or evaluator thresholds.

## M101 Support-Smoothing Scale-Transfer Loss Boundary

M101 keeps the same coupled C/D/E objective and the same explicit M99-selected
`loss_scale` and learning rate while rerunning the candidate across node counts.
It adds no loss term, teacher objective, direct physics loss, candidate-oracle
objective, evaluator change, adaptive balancing method, scheduler, or
checkpointing.

Any diagnosed scale-transfer failure can authorize only a report/design M102
follow-up. It cannot justify changing the mathematical objective, adding direct
link/SINR/BLER/HARQ/coverage losses, or claiming production training readiness.

## M102 Scale-Aware Update Normalization Loss Boundary

M102 adds no loss term and runs no optimizer steps. It designs future
scale-aware update-normalization probes that may multiply an explicit update or
learning-rate-like factor in a bounded diagnostic, but the raw coupled C/D/E
objective, evaluator thresholds, and reported loss components remain unchanged.

The design requires future probes to report raw loss separately from any scaled
update behavior, include a no-normalization control, and block on non-finite
losses or gradients. It does not authorize direct physics losses,
candidate-oracle objectives, GradNorm/PCGrad integration, default-path changes,
or production-training claims.

## M103 Scale-Aware Update Normalization Loss Boundary

M103 applies normalization only as an explicit diagnostic multiplier on the
effective loss scale:

```text
effective_loss_scale = base_loss_scale * computed_normalization_multiplier
```

The raw coupled C/D/E loss is unchanged and reported separately from scaled
loss. The probe adds no new objective term, direct physics loss, teacher loss,
candidate-oracle objective, adaptive balancing method, scheduler, or default
training behavior. Any restored movement is diagnostic evidence only and does
not validate production training.

## R1/R2 Scale-Law Loss Boundary

R1 adds no objective and runs no optimizer steps. It audits existing evidence
and source reductions, finding that the current raw loss/update path is not
scale-invariant with respect to node count: C/D/E metrics and raw loss terms
are reduced through node means or tails, while the explicit `loss_scale` path
does not compensate for 10k-to-500 update attenuation.

R2 also adds no objective and runs no optimizer steps. It designs possible
future remediation branches for scale-invariant update scaling and
query-support sensitivity while preserving raw C/D/E semantics. Any future
branch must report raw L_R/L_D/L_E separately from effective backward scaling,
must keep direct link/SINR/BLER/HARQ/coverage variables out of the loss, and
must require explicit human approval before any implementation prototype.

## R3 Scale-Invariant Loss/Update Design Boundary

R3 adds no loss term and runs no optimizer steps. It designs candidate
scale-invariant backward-signal strategies while preserving the public C/D/E
loss contract. Raw evaluator metrics, raw L_R/L_D/L_E, and `raw_total_loss`
remain the comparable scientific outputs.

Any future approved branch may only affect an explicitly named
`effective_backward_loss` or equivalent reported backward scalar. R3 rejects a
multiplier-only follow-up because M103 bounded effective-loss-scale
normalization stayed flat at 10k. R3 also requires query-support sensitivity to
be carried as a subdesign and keeps direct link/SINR/BLER/HARQ/coverage losses
forbidden.

## R4 Scale-Invariant Design Review Boundary

R4 adds no loss term and runs no optimizer steps. It reviews the R3 mixed
design, ranks reference-scale, active-failure, gradient-target,
query-support-coupled, and smoothing-mass-aware candidates, then selects one
bounded next branch.

When query-support attenuation is far stronger than topology-weight
attenuation and smoothing mass/entropy diagnostics are missing, R4 must choose
diagnostics before any loss/update prototype. Raw C/D/E metrics, raw
L_R/L_D/L_E, `raw_total_loss`, objective semantics, evaluator semantics, and
the default training path remain unchanged.

## R5 Topology Smoothing Mass and Query-Sensitivity Loss Boundary

R5 runs a diagnostic with optimizer steps only to produce new measurements of
the existing support-smoothing candidate. It does not add a loss term, direct
physics objective, query-support auxiliary objective, scale-invariant
`effective_backward_loss`, scheduler, checkpoint, default-path change, or
production-training path.

The diagnostic keeps raw C/D/E metrics and the existing raw coupled loss as the
reported objective values. Any scaled backward scalar remains the pre-existing
configured `loss_scale` from the diagnostic candidate, not a new objective
design. R5's classification can route a future R6 decision toward smoothing
mass, query sensitivity, active-failure normalization, or human review, but it
cannot itself authorize a prototype loss or behavior change.

## R6 Query-Support Counterfactual Loss Boundary

R6 does not add or modify any loss. It may compute the existing coupled C/D/E
loss once to obtain a detached topology-weight gradient for the
`gradient_direction_topology_perturbation` diagnostic, but it performs no
optimizer step and does not update model parameters.

All R6 perturbations are post-construction counterfactual evaluator passes.
Raw C/D/E and F metrics remain the reported outputs. R6 cannot introduce an
`effective_backward_loss`, query-support auxiliary objective, direct
link/SINR/BLER/HARQ/coverage loss, default training-path change, scheduler,
checkpoint, or production-training path.

## R7 Realistic Density Perturbation Loss Boundary

R7 adds no loss term and runs no optimizer step. It reuses the R6
support-preserving topology-weight perturbation as a post-construction
diagnostic under toy and realistic/proxy profiles, then reports whether
query-support and C/F movement capacity transfers beyond toy evidence.

Raw C/D/E, F, and density/profile measurements remain evaluator outputs only.
R7 cannot introduce an `effective_backward_loss`, query-support auxiliary
objective, direct link/SINR/BLER/HARQ/coverage loss, training prototype,
default-path change, or production-training claim. If density/profile evidence
is missing, R7 must route to profile contract work before production readiness
can be claimed.

## R8 Production Density Profile Loss Boundary

R8 adds no loss term and runs no optimizer step. It evaluates existing raw
C/D/E and F metrics on toy, `small_realistic`, and
`production_like_density_v0` density cases to check whether the proposed
production-like profile is non-saturated and scale-stable.

The R8 profile generator is diagnostic-only. It cannot introduce an
`effective_backward_loss`, query-support auxiliary objective, direct
link/SINR/BLER/HARQ/coverage loss, training prototype, default-path change, or
production-training claim.

## R9 Production Profile Perturbation Loss Boundary

R9 computes a backward gradient only to form a detached diagnostic
gradient-direction topology perturbation. It runs no optimizer step and makes
no parameter update. Raw C/D/E and F metrics remain evaluator outputs.

R9 cannot introduce an `effective_backward_loss`, query-support auxiliary
objective, direct link/SINR/BLER/HARQ/coverage loss, training prototype,
default-path change, or production-training claim.

## R10 Production Profile Training Signal Loss Boundary

R10 computes the existing raw loss and its gradients to diagnose signal
availability under `production_like_density_v0`. It does not add or replace any
loss term and does not introduce `effective_backward_loss`.

Projected parameter updates and edge-score gradient counterfactuals are
reported as diagnostics only. R10 does not step an optimizer, update
parameters, add a query-support auxiliary loss, add direct physics losses, or
change default training behavior.

## R11 Loss-to-Score Gradient Alignment Boundary

R11 decomposes the unchanged coupled C/D/E loss into `L_R`, `L_D`, `L_E`, and
`total_loss` gradients with respect to `edge_score` and topology weights under
`production_like_density_v0`. It runs detached component counterfactuals to
diagnose whether the current loss descent direction improves or worsens C/F.

R11 adds no objective and runs no optimizer step. Component-gradient and
topology-weight perturbations are diagnostic only; they do not authorize
query-support auxiliary losses, direct link/SINR/BLER/HARQ/coverage losses,
GradNorm/PCGrad integration, default-path changes, or production training.

## R12 Constructor Jacobian Alignment Loss Boundary

R12 consumes the unchanged R11 loss gradients and uses them only as detached
diagnostic perturbation directions. It does not add a loss, redesign the
existing C/D/E objective, introduce `effective_backward_loss`, apply GradNorm or
PCGrad, or run an optimizer step.

The fixed-support row-softmax and full-constructor comparisons diagnose how the
existing constructor maps loss-derived edge-score directions into topology
weights. They do not authorize a loss redesign, query-support auxiliary loss,
direct physics loss, default-path change, or production training.
