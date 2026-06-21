# Coupling Revival & Non-Trivial Operating Point — Design Contract & Decision Package

> **OUTCOME (2026-06-04 — both tracks executed; the "pre-implementation" status below is historical).**
> - **Track A → ADOPTED.** A non-trivial, load-coupled operating point exists:
>   **200 veh/km² × hard_low_confidence × 20 dB coupling × degree 4**, frozen as
>   `configs/operating_point_v1.yaml`. 4/5 seeds beat the strongest of 5 heuristics by
>   2.3–3.3× (mean 2.59×); seed 42 is a 1.87× outlier reported honestly. See
>   `result/operating_point_v1/RESULT.md`.
> - **Track B → COUPLED C/E EARNED, D INERT.** On that point with retransmission-aware
>   energy ON, **E is topology-controllable (−27…65% under w_E×10) and independent of D
>   (+21…52% margin), 3/3 seeds**, while **D stays protocol-floor-bound (≤0.83%, 3/3)** —
>   the *mirror* of the partial branch this contract anticipated. Honest claim: a
>   **coupled C/E** objective (reliability ↔ energy), D as a monitored diagnostic.
>   Caveat: the `full` (1,1) arm diverges on the hardest seed (123) — a multi-objective
>   instability (PCGrad/GradNorm territory). See `result/de_ablation_v2/RESULT.md`. The
>   follow-up design contract for that instability (and an honest, diagnostic-first take on
>   whether gradient governance can rescue D *controllability*) is
>   `docs/GRADIENT_GOVERNANCE_DESIGN.md`.
> - **LATER EXTENDED → COUPLED C/D/E.** Gradient governance fixed the seed-123 instability
>   (PCGrad adopted) but could NOT make D controllable (D was structurally inert). The
>   **structural delay model** (`docs/STRUCTURAL_DELAY_MODEL_DESIGN.md`, D-fix-A: per-hop ARQ
>   delay `n_tx=1/link_success`) then made **D topology-controllable (+33% × 3 seeds)** —
>   `result/de_ablation_v4_structural/RESULT.md`. Final headline: **coupled C/D/E**,
>   link-quality-controllable, with **D ∝ E** (coupled levers, Option A), stable under PCGrad.

Status: **design contract + decision package** (pre-implementation, house style of
`docs/TEMPORAL_MODEL_DESIGN.md` / `docs/CURRICULUM_TRAINING_DESIGN.md`). Addresses the
two existential gaps from the project health check:

- **Goal G1 — revive the coupled C/D/E novelty:** make the topology *controllably*
  trade reliability for delay/energy, so "coupled C/D/E" is earned, not nominal.
- **Goal G2 — escape the sweet-spot crack:** find/construct a *hard, realistic*
  operating point where topology genuinely decides success, so the surviving
  positive result (GNN ≫ heuristic) is not confined to a trivial regime.

> **EVIDENCE CORRECTION (this reframes everything below).** The naive form of G1 —
> "re-run the D/E ablation with the load-aware interference enabler ON" — **has
> already been run, and already failed.** `result/de_ablation_v1/de_ablation.json`
> was produced with the default config `configs/production_training_v1.yaml`, which
> sets `physical.interference_density_coupling_db: 10.0` **and**
> `finite_blocklength_reliability: true`. The bridge gate
> (`v2x_consensus_bridge.py:544`: `coupling>0 AND link_success is None AND
> edge_count`) was satisfied, so the load-aware path was **active**. Result anyway:
> `delay_relative_reduction = 5.48e-5`, `energy_relative_reduction = 4.48e-5` (both ≪
> the `meaningful_threshold = 0.02`); verdict *"D/E LOWER-BOUND SATURATED."* **So
> turning the enabler on is necessary but not sufficient.** The contracts below
> target the two *structural* reasons it failed.

---

## 0. Why it failed, precisely (the two structural locks)

From the evaluator audit (`docs/EVALUATOR_MODEL_AUDIT.md` §lines 80–89) and the
committed ablation JSON:

1. **D is floor-saturated.** `D = 5.78` rounds vs the protocol floor `D_min = beta =
   5.0` (only +16 %). At the reliable operating point (`F = 9.99e-4`) virtually every
   node finalises near the `beta` floor, so **D has almost no cross-node spread for
   topology to optimise.** This is the *same* triviality that confines the headline
   result — a too-easy operating point. ⇒ **G1 depends on G2.**

2. **E ≈ D × const.** `_energy_proxy` computes
   `node_energy = node_expected_rounds × (k × Σ normalized_query_weight ×
   packet_duration_s × (P_tx+P_rx+P_proc))`. With fixed active degree `k=4` and
   query mass normalised per node, the bracket is a **constant**, so energy is just
   delay rescaled. **E has no lever independent of D**, no matter the operating
   point. ⇒ **G1 also needs one structural evaluator fix (give E its own lever).**

**Consequence — the two goals are causally ordered, not parallel:**

```
G2 (a hard, load-coupled, critical operating point)         ← prerequisite substrate
   └─ unlocks D's dynamic range (congestion → variable rounds → D spread)
G1 (D/E topology-controllable on that operating point)
   └─ requires, additionally, B-fix-2: break E ≈ D × const (retransmission energy)
```

So the **executable order is Track A (operating point) → Track B (D/E revival on
it)**, even though the user listed G1 first. Each track has independent decision
gates and an honest off-ramp.

---

# TRACK A — Non-Trivial Operating Point (Goal G2)

## A.1 Definition of the target (what "good" means, quantitatively)

An operating point is a 4-tuple **(density, initial-confidence, avalanche-profile,
interference-coupling)**. We seek one that *simultaneously* satisfies:

| criterion | meaning | metric |
|---|---|---|
| **A-i Learnable & non-trivial** | not the failed basin (F≈1), not the trivial-supercritical basin (F≈1e-12) | `learned_F ∈ [1e-3, 1e-1]` |
| **A-ii Topology decides** | heuristics fail where the learned model succeeds | `gap_G = best_heuristic_F − learned_F` large; `learned_F ≤ 0.5 × best_heuristic_F` (≥2× edge) |
| **A-iii Load-coupled** | congestion is real (precondition for D/E leverage in Track B) | `interference_density_coupling_db > 0` active |
| **A-iv Realistic** | density/spacing/speed map to a citable V2X scenario | pinned to a named scenario in the RESULT |

**Why these and not "F ≤ 0.01".** The existing sweeps show the *gap grows as the task
hardens*, while F crosses the old 0.01 target only in a thin band:

| density (veh/km²) | learned_F | best-heur_F | gap_G | regime |
|---:|---:|---:|---:|---|
| 100 | 0.087–0.099 | 0.11–0.124 | ~0.025 | subcritical edge |
| 150 | **0.046** | **0.100** | **0.054** | **critical, large gap, misses 0.01** |
| 200 | ~0.053 (nearest-k) | — | — | critical |
| 300 (production) | 0.0033 | 0.0153 | 0.012 | upper-critical, target met, modest gap |
| 600 | 7.8e-12 | 9.0e-6 | ~0 | trivial |
| 800 | 1e-17…1e-26 | ~1e-27 | ~0 | trivial |

(Sources: `result/density_generalization_v1`, `result/standard_generalization_v1`,
`result/density_contract_v1`.) The **150 veh/km²** point already shows the most
interesting story — a **2.2× edge (0.046 vs 0.100), gap 0.054** — even though it
misses the (arbitrary) 0.01 target. So the contract optimises **gap subject to
learnability**, not raw F. The recommended operating point = `argmax gap_G` over the
sweep subject to A-i.

## A.2 Experiment spec — the critical-band finder (new, complete component)

**New script:** `scripts/analysis/run_operating_point_search.py`
**Makefile target:** `operating-point-search`
Reuses, do not reinvent: `scripts/analysis/generalization_common.py`
(`train_model`, `learned_F`, `best_heuristic_F`, `env_from_snapshot`) and
`src/v2x_env/profiles.py::density_matched_vehicle_config`.

**Sweep grid (develop at N=2000; promote at N=10000):**

| axis | values | knob |
|---|---|---|
| density (veh/km²) | {100, 150, 200, 250, 300, 350, 400} | `density_matched_vehicle_config(N, density, seed)` |
| initial confidence | `toy {0.50,0.25}`, `hard_low_confidence {0.40,0.25}` | `TRAINING_PROFILES` |
| interference coupling (dB) | {0, 10, 20} | `physical.interference_density_coupling_db` |
| active degree k | {3, 4} | `max_out_degree` |
| avalanche | `small_realistic` (k=5,α=3,β=5,rounds=20) | fixed |
| seeds | {7, 42, 123} (for the shortlist) | — |

Coarse pass: 1 seed over the full grid; **shortlist** the top-3 cells by `gap_G`
subject to A-i; **confirm** the shortlist on 3 seeds.

**Per-cell record (JSON):**
```json
{"density": 150, "confidence_profile": "hard_low_confidence", "coupling_db": 10,
 "degree": 3, "learned_F": 0.046, "best_heuristic_F_pair": 0.100,
 "best_heuristic_F_full5": 0.092, "gap_G": 0.054, "ratio": 2.17,
 "D": 5.9, "E": 1.6e-2, "node_fail_fraction_in_band_0p1_0p6": 0.41,
 "meets_band_A_i": true, "topology_decides_A_ii": true}
```
**Summary block:** `{"F001_crossing_density": ~270, "gap_maximizing_cell": {...},
"recommended_operating_point": {...}, "multiseed_gap_mean_std": [...],
"verdict": "..."}`.

**Output dir:** `result/operating_point_v1/` (JSON + a density×coupling heatmap of
`gap_G` + `RESULT.md`).

> Relation to existing probes: `src/training/intermediate_reliability_band_probe.py`
> searches for a band by `active_failure_node_fraction ∈ [0.1,0.6]` but **does not**
> locate the F-crossing vs density or the gap-maximising cell. This new script is the
> missing piece: it scores by **gap_G** (the "topology decides" quantity), the thing
> G2 actually needs. Record `node_fail_fraction_in_band` too, for cross-validation
> against the existing probe.

## A.3 Decision gates (Track A)

- **A-G1 (a band exists):** ∃ cell with `learned_F ∈ [1e-3, 1e-1]` **and**
  `best_heuristic_F > learned_F + 1e-2`.
- **A-G2 (topology decides, non-trivial):** at the recommended cell,
  `gap_G ≥ 0.02` **and** `ratio = best_heuristic_F / learned_F ≥ 2.0` **and**
  `learned_F ≥ 1e-4` (rules out the trivial-supercritical basin).
- **A-G3 (multi-seed stable):** the recommended cell keeps `gap_G ≥ 0.02` and
  `ratio ≥ 2.0` across all 3 seeds (sign + band, not exact value).

**Pass = A-G1 ∧ A-G2 ∧ A-G3.** On pass, freeze the cell into
`configs/operating_point_v1.yaml`, re-run the headline multiseed result there
(`result/operating_point_v1/`), and proceed to Track B on this substrate.

## A.4 Decision tree (Track A)

```
Run operating-point-search.
├─ A passes → adopt P* = recommended cell; promote headline result to P*; go to Track B.
├─ A-G1 ok but A-G2 fails everywhere (gap ≤ 0.02 wherever learnable)
│      → the task is intrinsically easy wherever it is solvable; the GNN's edge is
│        genuinely modest. HONESTLY reframe the contribution to "reliability-optimal
│        sparse topology at scale, generalises with ~0 gap" and DROP the
│        "topology decisively matters" claim. Consider a richer channel/quorum
│        substrate (separate design) before claiming more.
└─ A-G1 fails (everything is failed-basin or trivial; no learnable band)
       → the closed-form consensus is too bistable to host a non-trivial point.
         Escalate to a SUBSTRATE change (softer quorum / graded reliability target)
         — out of scope here; record and stop.
```

---

# TRACK B — Coupled C/D/E Revival (Goal G1), on the Track-A operating point

## B.1 Objective

On `P*` (the Track-A operating point), make D and E **topology-controllable and
independently so**:
- `delay_relative_reduction ≥ 0.02` (w_D ×10 shaves D ≥2% vs `rel_only`),
- `energy_relative_reduction ≥ 0.02`,
- **E responds independently of D** (the `energy_heavy` arm reduces E by more than the
  `delay_heavy` arm does — proving E is not merely D rescaled).

## B.2 The structural fix (necessary beyond the operating point)

**B-fix-1 — open D's dynamic range: delivered by Track A, no new code.** At a
critical + load-coupled `P*`, congested receivers get worse SINR → more expected
rounds → cross-node D spread. The wiring already exists: `topology_weight` →
`in_load` (`index_add`) → `edge_interference` → SINR → finite-blocklength
`link_success` → `expected_rounds` → D. Track A simply moves the system to where
that chain has slope.

**B-fix-2 — give E its own lever: break `E ≈ D × const` (new, complete component).**
Add a **retransmission-aware energy** path so per-edge energy depends on
link quality (which topology controls), not only on round count.

- **Model (Option E1, recommended — differentiable ARQ proxy):**
  expected transmissions per edge `n_tx = 1 / clamp(link_success, min=ε_tx)`;
  `edge_energy = n_tx × per_edge_query_energy`. Because `link_success` depends on
  in-load (enabler) **and** on which receiver the topology selects (distance/SINR),
  a topology choosing high-SINR / low-load receivers now spends **less energy at the
  same D** — an independent lever. (E2 — per-node degree/power control — is more
  invasive, changes the constructor's degree budget; **defer**.)
- **Implementation:** `src/evaluation/v2x_consensus_bridge.py`, in `_energy_proxy`,
  gated by a new `EnergyProxyConfig.retransmission_aware: bool = False`
  (**opt-in, default off ⇒ behaviour-preserving**; existing tests stay green). When
  true, multiply `per_edge_query_energy` by `n_tx` before the `index_add`.
- **Invariant check (critical):** this is an **evaluator/energy metric**, NOT a loss
  term. It must live in `src/evaluation/`, never in `src/losses/`. It enters the loss
  only through the *existing* E penalty in `coupled_objective.py`. Confirm it does not
  trip `verify_no_link_reliability_loss.py` (that verifier scans `src/losses/`; a
  retransmission term in the evaluator is legitimate and preserves the "no direct
  link-reliability *loss*" invariant). If the name-based regex false-positives, scope
  the verifier to `losses/` and document it.

## B.3 Experiment spec

- **Config:** `configs/operating_point_v1.yaml` (Track-A output), with
  `energy.retransmission_aware: true`.
- **Harness:** existing `scripts/analysis/run_de_ablation.py`
  (`--config configs/operating_point_v1.yaml --run-name de_ablation_v2`). Same 4
  arms: `full (1,1)`, `rel_only (0,0)`, `delay_heavy (10,0)`, `energy_heavy (0,10)`.
- **Emit** the existing `de_ablation.json` schema **plus** one field:
  `energy_independent_of_delay = (energy_relative_reduction ≥ 0.02) AND
  ((E[delay_heavy] − E[energy_heavy]) / E[rel_only] ≥ 0.01)`.
- **Output:** `result/de_ablation_v2/` (JSON + `RESULT.md`).

## B.4 Decision gates (Track B)

- **B-G1 (D controllable):** `delay_relative_reduction ≥ 0.02`.
- **B-G2 (E controllable):** `energy_relative_reduction ≥ 0.02`.
- **B-G3 (E independent of D):** `energy_independent_of_delay == true`.
- **B-G4 (no reliability regression):** `rel_only_meets_target` stays true and the
  `full` arm `F` stays within the Track-A band.

## B.5 Decision tree (Track B)

```
Re-run D/E ablation on P* with retransmission-aware energy.
├─ B-G1 ∧ B-G2 ∧ B-G3 ∧ B-G4 → "coupled C/D/E" EARNED at a non-trivial operating
│        point. Promote (retransmission-aware energy + P*) to production; write the
│        paper-grade RESULT: "topology controllably trades reliability for delay AND
│        energy." STRONGEST available outcome.
├─ B-G1 ∧ B-G4 only (D moves, E does not even with retransmission energy)
│        → claim "coupled C/D" honestly; present E as a monitored diagnostic; drop E
│          from the headline. (Energy may be genuinely protocol-bound here.)
└─ none → D/E remain inert even at the hard operating point with an energy lever
          → HONEST OFF-RAMP: reliability-only constructor; D/E reported as
            diagnostics; the "coupled C/D/E" claim is retired in the writeup.
            (Consistent with O1 / temporal-G2 discipline.)
```

---

# Unified sequencing & top-level decision tree

```
TRACK A  operating-point-search ──────────────────────────────────────────────┐
  ├─ fail (no large-gap learnable band)                                        │
  │     → contribution = "reliability-optimal sparse topology at scale";       │
  │       STOP D/E work; (optional) substrate redesign. END.                   │
  └─ pass → adopt P*, re-home headline result at P*                            │
        TRACK B  D/E revival on P* (+ retransmission-aware energy)             │
          ├─ pass (D & E & independent) → "coupled C/D/E" earned at a          │
          │     non-trivial point → strongest result; promote; paper section.  │
          ├─ partial (C/D only) → "coupled C/D" + E diagnostic.                │
          └─ fail → reliability-only + D/E diagnostics (honest off-ramp).      │
```

**Either way the project comes out ahead of the health-check status quo:** Track A
alone resolves the "is the win trivial?" question (G2); Track B resolves the "is the
coupling real?" question (G1). Every branch ends in a defensible, evidence-backed
claim — including the negative branches.

---

## Guardrails (preserved invariants)

- **train == val == deploy:** topology is hard top-k row-softmax in the forward path
  everywhere; straight-through is backward-only. Unchanged.
- **Analytic avalanche, sparse O(Nk):** unchanged; sweeps develop at N≤2000.
- **No direct link/SINR/BLER/HARQ/coverage *loss*:** the retransmission term is an
  **evaluator energy metric**, not a loss; it reaches the loss only via the existing
  E penalty. Keep it in `src/evaluation/`. Verify against
  `verify_no_link_reliability_loss.py`.
- **Opt-in, behaviour-preserving:** `EnergyProxyConfig.retransmission_aware` defaults
  off; `interference_density_coupling_db` defaults 0. All existing tests stay green.
- **Multi-seed before promotion:** no operating point or energy model is promoted on a
  single seed.

## Risks & mitigations

| risk | mitigation |
|---|---|
| Chosen operating point is not realistic | Pin density/speed/spacing to a named, citable V2X scenario in the RESULT; report the scenario, not just the number. |
| Larger gap requires sparser degree (k=3) which destabilises training | degree is a Track-A sweep axis; pick the gap-maximising cell that is also multi-seed stable (A-G3). |
| `n_tx = 1/link_success` blows up as `link_success→0` | clamp `min=ε_tx`; unit-test gradient finiteness at low success. |
| Retransmission term trips the no-link-reliability-loss verifier (name match) | it lives in `evaluation/`, not `losses/`; scope the verifier to `losses/` and document if needed. |
| Moving the headline to P* invalidates prior 300-density numbers | keep the 300-density result as a secondary "easy-regime" datapoint; re-run multiseed at P*. |
| D moves but E still bound (E2 needed) | accept the B-G1-only branch honestly (claim C/D); E2 (degree/power control) is a scoped follow-up, not this contract. |

## Implementation checklist

**Track A**
1. `scripts/analysis/run_operating_point_search.py` (+ `make operating-point-search`),
   reusing `generalization_common.py` + `density_matched_vehicle_config`.
2. `configs/operating_point_v1.yaml` (frozen from the adopted cell).
3. `result/operating_point_v1/` — JSON, `gap_G` heatmap, `RESULT.md`.
4. Test: `tests/analysis/test_operating_point_search.py` (gap computation, band
   membership, summary selection on a tiny synthetic grid).

**Track B**
5. `src/evaluation/v2x_consensus_bridge.py`: `EnergyProxyConfig.retransmission_aware`
   + retransmission energy in `_energy_proxy` (opt-in).
6. `tests/evaluation/test_retransmission_energy.py` (default-off behaviour-preserving;
   gradient to `topology_weight`; E varies at fixed round-count; finite at low
   success).
7. Re-run `run_de_ablation.py --config configs/operating_point_v1.yaml --run-name
   de_ablation_v2`; write `result/de_ablation_v2/RESULT.md`.
8. Docs: stamp the verdict banner on this file; update `docs/PROJECT_ANALYSIS.md` /
   `docs/PROJECT_STATUS_AND_NEXT_STEPS.md` / `docs/EVALUATOR_MODEL_AUDIT.md`.

## Cost & recommendation

- **Track A:** 1 script + 1 config + 1 result dir + 1 test. Compute = density(7) ×
  confidence(2) × coupling(3) × degree(2) grid, 1 seed coarse + 3 seeds on a 3-cell
  shortlist, at N≤2000 — comparable to the existing generalisation sweeps (~half a day
  wall-clock on CPU).
- **Track B:** 1 evaluator flag + 1 test + 1 ablation re-run. Small.
- **Recommendation:** **do Track A first** — it is the prerequisite substrate for
  Track B *and* it independently answers G2 (the headline-triviality question). Only
  on A-pass start Track B. This pairing is the single highest-leverage next step from
  the health check: it directly attacks the two existential gaps (trivial operating
  point; nominal coupling) and every branch — pass or fail — yields a publishable,
  honest claim. Until then, the memoryless reliability-only constructor at 300 veh/km²
  remains the evidence-backed default.
