# Gradient Governance — Rescuing Delay (D) Stability & Controllability — Design Contract & Technical Package

> **PHASE 0 EXECUTED (2026-06-04) — the leverage prediction was REFUTED; see
> `result/gradient_conflict_v1/RESULT.md`.** Measured at `operating_point_v1` (N=2000,
> 200 steps, seeds 7/42/123), gate never engaged:
> - **P0-G2 ⇒ D-diag-A on 2/3 seeds (NOT the predicted D-diag-B).** `∇L_D` is genuine
>   (seed 7 norm 2.1, ~12% of the dominant, **aligned with `∇L_R` +0.95**) but **swamped
>   by and anti-aligned with `∇L_E`** (norm 18.9, cos −0.66). D didn't move in
>   `de_ablation_v2` because it was **out-voted by energy**, not gradient-less. ⇒ **Track G2
>   is viable to TEST.** Caveat: `de_ablation_v2`'s `delay_heavy` arm (E off, w_D×10) still
>   left D flat, so the gradient may not translate to hard-forward D (straight-through /
>   floor) — G2 is "worth a measured attempt", not "likely win".
> - **P0-G1 ⇒ divergence did NOT reproduce (numerically fragile; seed 123 spiked to
>   max_F=0.45 and recovered) but conflict is pervasive** (negative cosines 34–95% of
>   steps, E systematically opposing R and D). PCGrad still well-motivated for the tail
>   risk + everywhere-conflicted updates. ⇒ **Track G1 still worth doing.**
>
> Net change vs the pre-Phase-0 plan: the pessimistic "D is zero-leverage → off-ramp"
> default is **lifted**; the recommended path is now **Track G1 (stability) + Track G2
> (governed D-control, genuinely worth testing now)**. Phase 0 is implemented:
> `src/losses/gradnorm.py::compute_task_gradient_vectors`,
> `scripts/analysis/run_gradient_conflict_diagnostic.py`.

> **TRACKS G1/G2 EXECUTED (2026-06-04) — see `result/de_ablation_v3_governed/RESULT.md`.**
> Integration layer `src/training/gradient_governance.py::coupled_backward` built (opt-in,
> default-off byte-identical; `run_de_ablation.py --governance`); `pcgrad` run × seeds
> 7/42/123.
> - **G1 (stability) EARNED.** Seed 123's `full` arm went from **F=0.506 (diverged in v2)
>   to F=0.0199 (in band)** under PCGrad; seeds 7/42 unchanged; C/E preserved (E reduction
>   +38/+23/+57%, independent 3/3). **Adopt PCGrad for the combined objective.** Caveat: the
>   divergence is numerically fragile (Phase 0 recovered ungoverned), so this is strong-but-
>   not-clean-causal; PCGrad attacks the measured root cause (pervasive E↔{R,D} conflict).
> - **G2 (D controllability) FAILS decisively → structural-delay OFF-RAMP confirmed.**
>   `delay_heavy` (w_D=10, **w_E=0 so no E-conflict**, R↔D aligned) **+ PCGrad** still moved
>   D by +0.05/+0.00/−0.20% (3/3). D given every advantage and immovable ⇒ structural
>   (straight-through top-k / β-floor), **not** conflict/magnitude. No gradient method can
>   fix it; `both`/GradNorm cannot beat 10×-weight-no-conflict. **Headline stays coupled
>   C/E, now stable under PCGrad; D needs a structural delay model — now specified in
>   `docs/STRUCTURAL_DELAY_MODEL_DESIGN.md` (D-fix-A: link-reliability/Q-function ARQ delay,
>   keeps PCGrad; honest caveat that D∝E unless the optional hop/path D-fix-B is added).**

Status: **design contract + technical package** (pre-implementation, house style of
`docs/COUPLING_AND_OPERATING_POINT_DESIGN.md`). Authorizes, the sanctioned way
(opt-in, default-off, unit-tested, training-validated), the **first wiring of PCGrad /
GradNorm into the training path** — a step that `docs/LOSS_DESIGN.md` has deferred ~30×
("`gradnorm.py`/`pcgrad.py` provide deterministic utilities only … no optimizer
integration"). It addresses the two delay-related defects measured in
`result/de_ablation_v2/`:

- **Goal D-S (stability):** the combined objective must not diverge. On the hardest
  layout (seed 123) the `full` (w_D=1, w_E=1) arm **blew up — F=0.506, C=0.494, D=14.8,
  E=12.4** — while every *isolated* arm trained fine (F≈0.021–0.026).
- **Goal D-C (controllability):** cranking w_D by 10× must move D. It **does not** —
  delay_relative_reduction = +0.003% / +0.13% / +0.83% across seeds 7/42/123, with D
  pinned at **1.32× its protocol floor**.

> **HONEST FRAMING — these are two DIFFERENT problems and gradient governance addresses
> them ASYMMETRICALLY. Read before implementing.**
> - **D-S is a gradient-*conflict* problem ⇒ PCGrad/GradNorm is the right tool.** The
>   tell: the *large* single-objective weights are stable (delay_heavy w_D=10 ✅,
>   energy_heavy w_E=10 ✅) but the *modest* combination (full 1,1) diverges ❌. A pair
>   that is fine alone and unstable together is the textbook signature of conflicting
>   per-task gradients — exactly what PCGrad projects out and GradNorm rebalances.
> - **D-C is (most likely) a *zero-leverage* problem ⇒ gradient governance probably
>   CANNOT fix it.** If `∂D/∂topology ≈ 0` (D = avalanche expected rounds, dynamics-bound
>   near the β floor, barely a function of edge choice), then no gradient surgery creates
>   leverage: PCGrad only removes *conflicting components* and GradNorm only rescales
>   *existing magnitude* — neither manufactures a gradient where the Jacobian is ~0.
>   Scaling 0 by any weight is still 0.
>
> Therefore this contract is **diagnostic-first**: Phase 0 *measures* `‖∇_θ L_D‖` and the
> inter-task cosines before either track commits. The stability win (D-S) is high-value
> and high-probability; the controllability win (D-C) is gated and, on current evidence,
> likely to honest-off-ramp to a **structural delay model** (not a gradient trick).

---

## 0. Why the defects occur (mechanism, from the committed evidence)

From `result/de_ablation_v2/` (N=2000, 200 steps, `configs/operating_point_v1.yaml`,
seeds 7/42/123) and the loss path in `src/losses/coupled_objective.py`:

1. **The training step is a single static-weight backward.** `compute_coupled_loss`
   forms `total = w_R·L_R + w_D·L_D + w_E·L_E` and callers `.backward()` one scalar
   (`effective_backward_loss`). Per-task gradients are **summed blindly**; if `∇L_D` and
   `∇L_E` point in opposing directions, the sum is a poor descent direction for *both* —
   and at a hard operating point that mis-step compounds into divergence (seed 123 full).

2. **The reliability gate can starve the picture further.** When F is satisfied the gate
   multiplies `w_R` by `min_reliability_weight_when_satisfied=0.05`
   (`coupled_objective.py:230`), abruptly changing the gradient mixture mid-training — a
   second source of conflict the static sum cannot manage.

3. **D has little-to-no topology lever at this operating point.** D ends at 1.32× its β
   floor and moves ≤0.83% under w_D×10. This is consistent with `∂D/∂topology≈0`, but it
   has **never been measured directly** — only inferred from D's immobility. Phase 0
   measures it.

**The two goals are therefore causally separate**, unlike Track A→B in the coupling
contract:

```
D-S (stability)  ── caused by per-task gradient CONFLICT ──► fixable by PCGrad/GradNorm
D-C (control)    ── caused by (likely) ZERO topology leverage on D ──► NOT fixable by
                    gradient governance; needs a structural delay model if pursued at all
Phase 0 diagnostic decides D-C's fate and confirms D-S's mechanism. Run it first.
```

---

# PHASE 0 — Gradient-Conflict & Leverage Diagnostic (run first; gates both tracks)

## P0.1 What it measures

At `configs/operating_point_v1.yaml`, N=2000, on the **`full` arm** (w_R=w_D=w_E=1, the
one that diverges), for seeds 7/42/123, log **every step** (cheap, reuses existing code):

| quantity | how | reuse |
|---|---|---|
| `‖∇_θ L_R‖, ‖∇_θ L_D‖, ‖∇_θ L_E‖` | per-task grad norms w.r.t. constructor params | `src/losses/gradnorm.py::compute_task_gradient_norms` (exists) |
| `cos(∇L_D,∇L_E)`, `cos(∇L_D,∇L_R)`, `cos(∇L_R,∇L_E)` | pairwise cosine of the per-task grad vectors | **new** `compute_task_gradient_vectors` (sibling of the norm fn) |
| gate state, weights, F/C/D/E per step | already in `compute_coupled_loss` output | — |

Feed the **weighted, scale-multiplied** per-task losses (`weighted_L_R/D/E ×
scale_backward_multiplier`, all already returned by `compute_coupled_loss`) so the
diagnostic reflects the *actual* training gradients, gate included.

## P0.2 Decision gates

- **P0-G1 — conflict is the divergence mechanism (for D-S):** on seed 123's `full` arm,
  there is a step window where `cos(∇L_D,∇L_E) < 0` (or `cos(∇L_*,∇L_R) < 0`) **and**
  divergence onset (F rising) coincides with it. ⇒ PCGrad/GradNorm is the right fix; go
  Track G1. *If F diverges with NO negative cosine*, the cause is LR/scale, not conflict
  ⇒ G1 off-ramp (LR warmup / `effective_backward` scale schedule), and PCGrad is not
  expected to help — recorded honestly.
- **P0-G2 — does D have any topology leverage (for D-C)?**
  - **D-diag-A (suppressed leverage):** `‖∇_θ L_D‖ ≥ 0.10 · max(‖∇L_R‖,‖∇L_E‖)` for ≥30%
    of steps **and** `∇L_D` is frequently conflicted/out-magnitude'd. ⇒ governance *can*
    plausibly free D; Track G2 is viable.
  - **D-diag-B (no leverage):** `‖∇_θ L_D‖ < 0.10 · max(‖∇L_R‖,‖∇L_E‖)` for ≥70% of steps
    (near-zero delay gradient). ⇒ **gradient governance cannot make D controllable**;
    Track G2 takes the structural-delay off-ramp. *(This is the predicted branch given
    D moves ≤0.83% under w_D×10.)*

## P0.3 Technical package (Phase 0)

- **New:** `src/losses/gradnorm.py::compute_task_gradient_vectors(losses, parameters)` —
  returns `{name: flat_vector}` (the same `autograd.grad(retain_graph=True)` pattern as
  `compute_task_gradient_norms`, without the norm reduction). Cosines computed in the
  script from these vectors. *(Add to `tests/loss/test_pcgrad_gradnorm.py`.)*
- **New:** `scripts/analysis/run_gradient_conflict_diagnostic.py`
  (`make gradient-conflict-diagnostic`). Reuses `training_smoke` helpers + the de_ablation
  forward path; trains the `full` arm logging the P0.1 table; emits
  `result/gradient_conflict_v1/{conflict.json, figures/cos_and_norms.png, RESULT.md}`.
- **Test:** `tests/analysis/test_gradient_conflict_diagnostic.py` (cosine of two known
  vectors; D-diag-A/B classification on a synthetic norm trace).

**Cost:** ~one de_ablation arm × 3 seeds at N=2000 (~3–4 min) + 2 small files. Pure
diagnostic — no default-path change. **This is the single highest-leverage next action:
it confirms the D-S fix target and decides whether D-C is even reachable by this method.**

---

# TRACK G1 — Stability of the Combined Objective (PCGrad / GradNorm)

## G1.1 Objective

The combined (w_D, w_E both on) objective trains to the operating-point band on **every**
seed — no divergence — without regressing the reliability/energy results already earned
in `result/de_ablation_v2/`.

## G1.2 The fix — wire gradient governance into the backward step (new integration layer)

**New:** `src/training/gradient_governance.py`

```python
@dataclass(frozen=True)
class GradientGovernanceConfig:
    pcgrad: bool = False          # opt-in; project conflicting per-task gradients
    gradnorm: bool = False        # opt-in; adaptive per-task reweighting (NOT for fixed-weight ablation arms)
    gradnorm_alpha: float = 0.5
    gradnorm_lr: float = 0.025
    log_conflict: bool = False    # emit per-task norms + pairwise cosines each step
    # default all-off  ⇒  byte-identical to today's effective_backward_loss.backward()

def coupled_backward(loss_out, parameters, governance, balancer=None) -> dict:
    """Populate parameter .grad from compute_coupled_loss output `loss_out`.
    - all-off  -> loss_out['effective_backward_loss'].backward()           (unchanged path)
    - pcgrad   -> per-task g_t = autograd.grad(weighted_L_t * scale_mult, params, retain_graph=True),
                  concat per task -> pcgrad_project([g_R,g_D,g_E]) -> sum -> scatter into .grad
    - gradnorm -> compute_task_gradient_norms -> balancer.update(losses, norms) -> new weights
                  (used to recombine this/next step); composes with pcgrad if both on
    Returns {task_norms, pairwise_cosines, weights} when log_conflict."""
```

Reuses verbatim: `pcgrad_project` / `merge_pcgrad` (`src/losses/pcgrad.py`),
`compute_task_gradient_norms` / `GradNormBalancer(tasks=("R","D","E"))`
(`src/losses/gradnorm.py`), and `compute_coupled_loss`'s already-exposed
`weighted_L_R/L_D/L_E` + `scale_backward_multiplier`. **No new math in the loss.**

**Invariant — this is gradient *plumbing*, not a new objective.** It changes *how* the
existing C/D/E gradients are combined, never *what* is optimized. No link/SINR/BLER/HARQ/
coverage term is introduced. `verify_no_link_reliability_loss.py` stays green (the helper
lives in `src/training/`, consumes only `L_R/L_D/L_E`).

## G1.3 Experiment spec

- **Harness:** `run_de_ablation.py` gains `--governance {none,pcgrad,gradnorm,both}`
  (default `none` ⇒ unchanged). The training loop branches to `coupled_backward`.
  *PCGrad keeps the fixed arm weights* (it only de-conflicts) so the w_D×10 / w_E×10 probe
  semantics survive; *GradNorm changes weights*, so the gradnorm/both modes are reported
  as a **governed-production** configuration, not as fixed-weight ablation arms.
- **Runs (N=2000, 200 steps, rel-target 0.02, seeds 7/42/123):**
  1. `--governance pcgrad --run-name de_ablation_v3_pcgrad` (the D-S test: does seed 123
     `full` stop diverging?).
  2. `--governance both  --run-name de_ablation_v3_both` (PCGrad + GradNorm production
     candidate).
- **Output:** `result/de_ablation_v3_governed/` (per-mode JSON + `RESULT.md`), with the
  `full`-arm F/C/D/E per seed vs the v2 baseline.

## G1.4 Decision gates (Track G1)

- **G1-G1 (stability restored):** seed 123 `full` arm is finite and **in band (F ≤ 0.10,
  target ≤ 0.02)** with `--governance pcgrad`; no seed's `full` arm diverges.
- **G1-G2 (no regression, off-path preserved):** with all governance flags off the run is
  byte-identical to v2 (assert in test); with pcgrad on, seeds 7/42 keep
  `energy_relative_reduction ≥ 0.02` and `F` no worse than the v2 `full` arm (± tol).
- **G1-G3 (multi-seed):** G1-G1 ∧ G1-G2 hold across seeds 7/42/123.

**Pass ⇒** adopt `gradient_governance.pcgrad: true` for the combined objective in
`configs/operating_point_v1.yaml` (opt-in, recommended-on whenever w_D and w_E are both
nonzero); re-stamp `result/de_ablation_v2` as superseded by the governed v3.

## G1.5 Decision tree (Track G1)

```
Run pcgrad (and both) on the full arm, seeds 7/42/123.
├─ G1-G1∧G1-G2∧G1-G3 pass → stability EARNED; PCGrad adopted for the combined objective.
├─ pcgrad partial (123 finite but out-of-band) → add GradNorm (both) and/or LR warmup;
│      if `both` passes, adopt both; else escalate to LR/scale schedule.
└─ pcgrad no effect AND P0-G1 found no negative cosine → divergence is LR/scale, not
       conflict → fix via warmup / effective-backward scale schedule (separate, small).
```

---

# TRACK G2 — Delay Controllability (gated on Phase 0; honest off-ramp built in)

## G2.1 Branch on the Phase-0 leverage verdict

- **If D-diag-B (no leverage — the predicted case): DO NOT attempt D-C via gradient
  governance.** State plainly in the RESULT: *PCGrad/GradNorm cannot create delay
  controllability because the topology has ~no gradient lever on D (avalanche-rounds delay
  is protocol-bound).* The only path to a controllable D is a **structural delay model** —
  make D depend on graph structure (hop-count / path-length / multi-hop relay delay) so
  edge selection has leverage. That is a **separate evaluator-model design** (cf. the v1
  D/E ablation recommendation and `docs/EVALUATOR_MODEL_AUDIT.md`); scoped out here.
  Track B's verdict stands: the earned claim is **coupled C/E**, D a monitored diagnostic.
- **If D-diag-A (suppressed leverage): pursue D-C via governance.** GradNorm up-weights the
  slow task (D) by training-rate matching; PCGrad stops R/E from cancelling D's gradient.

## G2.2 Experiment spec (only if D-diag-A)

- `run_de_ablation.py --governance both --run-name de_ablation_v3_dctrl` at
  `operating_point_v1`, seeds 7/42/123; **additionally** report
  `delay_relative_reduction` under governance vs the v2 static `delay_heavy` arm.
- Optionally add a `delay_heavy` governed arm (w_D=10 + PCGrad) to isolate D's freed
  response.

## G2.3 Decision gates (Track G2)

- **G2-G1 (D moves):** governed `delay_relative_reduction ≥ 0.02` (D drops ≥2% vs
  rel_only) — the same bar the v2 B-G1 failed.
- **G2-G2 (no reliability/energy regression):** `rel_only_meets_target` stays true; the
  earned `energy_relative_reduction ≥ 0.02` survives.
- **G2-G3 (multi-seed):** holds on seeds 7/42/123.

## G2.4 Decision tree (Track G2)

```
P0-G2 verdict?
├─ D-diag-B (≈zero leverage) → OFF-RAMP: structural delay model required; gradient
│      governance cannot help D-C. Claim stays "coupled C/E"; open a separate
│      EVALUATOR delay-model design. (Predicted outcome.)
└─ D-diag-A (suppressed leverage) → run governed D-control experiment.
        ├─ G2-G1∧G2-G2∧G2-G3 → "coupled C/D/E" finally EARNED at a non-trivial point;
        │      promote governed config; strongest possible result. Update the coupling
        │      contract banner from "C/E" to "C/D/E (under gradient governance)".
        └─ fail → D-control not achievable by governance either → same OFF-RAMP as above.
```

---

# Unified sequencing & top-level decision tree

```
PHASE 0  gradient-conflict diagnostic (full arm, seeds 7/42/123) ─────────────────────┐
  ├─ P0-G1: conflict drives the seed-123 divergence?  (expected: yes)                  │
  └─ P0-G2: does D have topology leverage?            (expected: no → D-diag-B)        │
        │                                                                              │
        ├──► TRACK G1 (stability): integrate PCGrad (opt-in) → kill the full-arm       │
        │     divergence. HIGH value, HIGH probability. Independent of P0-G2.          │
        │                                                                              │
        └──► TRACK G2 (D control): gated on P0-G2.                                     │
              ├─ D-diag-A → governed D-control experiment → maybe "C/D/E earned".      │
              └─ D-diag-B → OFF-RAMP to a structural delay model; "coupled C/E" stands.│
```

**Net:** Track G1 is worth doing regardless (it makes the combined objective trainable on
hard seeds and removes the only stability caveat in `result/de_ablation_v2/RESULT.md`).
Track G2 is honest about the likely ceiling: gradient governance is **necessary but
probably insufficient** for delay controllability, and the contract says so up front
rather than overselling a gradient trick.

---

## Guardrails (preserved invariants)

- **No new objective / no banned losses.** Gradient governance recombines existing
  `L_R/L_D/L_E` gradients only. No link/SINR/BLER/HARQ/coverage loss is added; the helper
  lives in `src/training/`, and `verify_no_link_reliability_loss.py` (scans `losses/`)
  stays green.
- **Opt-in, default-off, behaviour-preserving.** `GradientGovernanceConfig` defaults
  all-off ⇒ byte-identical to `effective_backward_loss.backward()`. A unit test asserts
  bit-equality of the off path; every existing test stays green.
- **train == val == deploy.** Governance touches only the *backward* gradient combination;
  the forward hard top-k row-softmax topology is unchanged everywhere. (Same spirit as the
  straight-through backward-only surrogate.)
- **Scale-invariant backward preserved.** Per-task gradients are taken on
  `weighted_L_t × scale_backward_multiplier`, so P1 scale-invariance is retained.
- **Determinism & sparsity.** `pcgrad_project` / `GradNormBalancer` are deterministic
  (already tested); per-task `autograd.grad` adds `retain_graph` passes but no Monte Carlo,
  no O(N²).
- **Multi-seed before promotion.** No governed config is promoted on a single seed.
- **Lifts the "utilities only" deferral the sanctioned way.** This is the first authorized
  PCGrad/GradNorm *integration*; per AGENTS.md it is allowed because it is opt-in,
  unit-tested, and training-validated. Update `docs/LOSS_DESIGN.md` §Gradient Governance to
  point here once G1 lands.

## Risks & mitigations

| risk | mitigation |
|---|---|
| `compute_task_gradient_norms`/vectors triple the backward cost (3 autograd.grad passes) | only when governance is on; develop at N≤2000; norms reuse one graph via `retain_graph`. |
| PCGrad de-conflicts but reliability suffers (projecting away useful R signal) | G1-G2 gate: pcgrad-on must not worsen seed-7/42 F vs v2; if it does, restrict PCGrad to {D,E} and leave R un-projected. |
| GradNorm reweighting breaks the fixed-weight ablation probe | keep GradNorm out of the ablation *arms*; use it only in governed-production runs; ablation uses PCGrad (weight-preserving). |
| Phase 0 says D-diag-A but G2 still fails | honest off-ramp is already in the tree; no goalposts move (the 2% bar is the same as v2 B-G1). |
| Over-claiming D-control from a gradient trick | the contract pre-commits to D-diag-B → structural-delay off-ramp; "C/E" stays unless G2-G1∧G2-G2∧G2-G3 pass on multi-seed. |
| seed-123 divergence is actually LR/scale, not conflict | P0-G1 distinguishes them; G1 tree routes to an LR/scale fix if no negative cosine is found. |

## Implementation checklist

**Phase 0 (do first)**
1. `src/losses/gradnorm.py::compute_task_gradient_vectors` (+ test in
   `tests/loss/test_pcgrad_gradnorm.py`).
2. `scripts/analysis/run_gradient_conflict_diagnostic.py` + `make gradient-conflict-diagnostic`.
3. `result/gradient_conflict_v1/` (JSON + cos/norm figure + `RESULT.md` with the
   P0-G1 / P0-G2 verdicts).
4. `tests/analysis/test_gradient_conflict_diagnostic.py`.

**Track G1 (stability)**
5. `src/training/gradient_governance.py`: `GradientGovernanceConfig` + `coupled_backward`.
6. `run_de_ablation.py`: `--governance` flag → `coupled_backward` in the loop (default
   `none` ⇒ unchanged).
7. `tests/training/test_gradient_governance.py`: off-path bit-identical; pcgrad changes
   `.grad`; projection removes negative cosine; gradnorm reweights; grads finite & flow to
   constructor params.
8. Runs → `result/de_ablation_v3_governed/RESULT.md`; apply G1 gates/tree.

**Track G2 (controllability — only if P0-G2 = D-diag-A)**
9. Governed D-control runs → `result/de_ablation_v3_dctrl/`; apply G2 gates/tree.
10. Docs: if C/D/E earned, update `docs/COUPLING_AND_OPERATING_POINT_DESIGN.md` banner
    ("C/E" → "C/D/E under governance"); else open a separate structural-delay evaluator
    design and record the off-ramp.

## Cost & recommendation

- **Phase 0:** ~3–4 min compute + 2 small files. **Do it first** — it is the gate for
  everything below and reuses code that already exists.
- **Track G1:** 1 integration module + 1 harness flag + 1 test + 2 short runs. Small, and
  it removes the only stability caveat in the Track-B result — **high value, high
  probability, recommended.**
- **Track G2:** gated; on current evidence (D ≤0.83% under w_D×10) expect the
  structural-delay off-ramp, i.e. gradient governance will **not** earn D-control. Pursue
  the governed experiment only if Phase 0 returns D-diag-A; otherwise log the off-ramp and
  leave the headline at **coupled C/E**.
- **Bottom line:** the defensible, evidence-backed plan is **Phase 0 → Track G1 (land the
  stability fix) → let Phase 0 decide G2.** Do not promise delay controllability from a
  gradient method; promise stability (which this delivers) and measure leverage before
  claiming more.
```
