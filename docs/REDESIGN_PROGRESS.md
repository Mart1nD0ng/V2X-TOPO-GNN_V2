# ESD Redesign — Decision Log & Gate Status (G0–G12)

Tracks the effective-sampling-dynamics reconstruction (`EFFECTIVE_SAMPLING_DYNAMICS_TECHNICAL_SPEC.md`
+ `ESD_GNN_ENGINEERING_PLAN.md`). The legacy F/D/E mainline (old G1–G11, tag
`legacy-global-fde-v1`) is historical material only; its `docs/FINAL_REPORT_CN.md`
and `docs/REFACTOR_PROGRESS.md` do not map to these gates.

Branch: `effective-sampling-redesign`.

## Gate status

| Gate | Scope | Status |
|------|-------|--------|
| G0  | canonical execution closure (single `run_consensus_episode`) | ☐ not started |
| G1  | protocol semantics — true binary Snowball + single-node exact ref | 🟡 core done (per-node chain); joint exact ref pending (Phase 2) |
| G2  | correlated-evidence environment | 🟡 evidence model + correlation theory done; urban geometry/graphs pending |
| G3  | round-coupled full physics | ☐ |
| G4  | CDQ k-DPP subset exactness | ☐ |
| G5  | determinantal quorum exactness | ☐ |
| G6  | independent dynamic MC | ☐ |
| G7  | effective-sampling diagnostics | ☐ |
| G8  | protocol feasibility (perfect-link floors) | ☐ |
| G9  | model mechanism & ablations | ☐ |
| G10 | large-N complexity/performance | ☐ |
| G11 | reliability-constrained superiority | ☐ |
| G12 | temporal robustness | ☐ |

## Decision log

### D0 — Phase 0 contracts (2026-06-24)
* **Protocol:** adopt **true binary Snowball** (spec §3.1 option 1), not the legacy
  Snowflake streak automaton. Rationale + pseudocode + state def + safety/validity/
  deadline + threat model in `docs/PROTOCOL_SEMANTICS.md`.
* **Objective:** reliability-constrained CVaR tail-latency / energy, topology-only
  headline; primal-dual duals on the three reliability constraints. Thresholds adopted
  as **recommended defaults, flagged for user override** (`docs/OPTIMIZATION_CONTRACT.md`):
  `q=0.95`, `ε_s=1e-4`, `ε_v=1e-3`, `ε_d=1e-2`; `T_d`/`R_max` calibrated in Phase 5.
* **Did not stop on spec stop-condition #6** (protocol/threshold choice): the design docs
  already give a clear recommendation, so proceeded on documented defaults per the loop's
  "don't wait for non-essential confirmation". User may override the contract constants
  without touching any math/code.

### D1 — G1 true-Snowball per-node chain (2026-06-24)
* `src/protocol/binary_snowball.py`: finite-horizon reachable-state BFS, sparse
  (no `[B,S,S]`) differentiable `apply_round`, dense `transition_matrix` for tests,
  initial/readout/terminal helpers, deterministic reference `simulate_trajectory`.
* **Exactness boundary:** the per-node chain is the exact marginal evolution *given*
  `(h⁺,h⁻,h⁰)`. Joint/global exactness is only under the shared-latent model; the
  independent dynamic MC (G6) is the final judge (`PROTOCOL_SEMANTICS.md` §5).
* **Complexity:** `S = O(R_max·β)` (measured 62…2782 for β,R_max up to 20,40); per round
  `O(B·S)` via sparse scatter — clears spec stop-condition #5.
* **Evidence** (`tests/protocol/test_binary_snowball.py`, 8 passing):
  hand trace; row-stochastic + mass conservation; sparse==dense operator; exact chain ==
  brute-force `3^R` enumeration (atol 1e-12); **Snowball≠Snowflake** confidence-persistence
  sentinel (legacy flips, true Snowball sticks); differentiability in `h`.
* **Pending for G1 green:** small-`N` *joint* exact reference (`exact_small_n.py`,
  Phase 2) cross-checked against the dynamic MC under G6.

### D2 — G2 correlated-evidence model (2026-06-24)
* `src/environment/evidence_model.py`: `O_i = Y* ⊕ B_{g(i)} ⊕ E_i` (spec §6.2).
  - `sample()` per-instance generator for the dynamic MC (truth fields kept separate
    from observable region structure);
  - `analytic_scenarios()` exact shared-latent decomposition `(omega_r, init_cp[i,r])`
    over `2^G` region-bit configs — this is the analytic evaluator's shared latent `Z`;
  - `correct_observation_prob()` marginal `q_i`; `pairwise_correlation_theory()` exact.
* **Exactness boundary:** scenario enumeration is `2^G`; raises above `max_scenarios`
  (no silent truncation). Large-`N` many-region case uses dynamic MC / reduced scenarios.
* **Evidence** (`tests/environment/test_evidence_model.py`, 7 passing): empirical
  pairwise correlation matches theory (<0.01); zero-correlation control recovers
  independence; scenario decomposition reproduces the exact pairwise correlation (1e-12);
  weighted scenario marginal == `q_i` (1e-12); refuses too-many-regions; truth/observable
  separation.

## Next slice
Phase 3/4 geometry: `urban_scene.py` (Manhattan grid → positions + region assignment)
and `candidate_graph.py` / `interference_graph.py` (two physical graphs G_comm, G_int,
spec §7.1). Then the round-coupled canonical episode (G0/G3) ties protocol + evidence +
physics together. Geometry should reuse `src/mainline/topology.py::build_candidate_graph`
(no degree cap) as a starting point.
