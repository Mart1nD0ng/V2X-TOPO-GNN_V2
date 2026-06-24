# ESD Redesign — Decision Log & Gate Status (G0–G12)

Tracks the effective-sampling-dynamics reconstruction (`EFFECTIVE_SAMPLING_DYNAMICS_TECHNICAL_SPEC.md`
+ `ESD_GNN_ENGINEERING_PLAN.md`). The legacy F/D/E mainline (old G1–G11, tag
`legacy-global-fde-v1`) is historical material only; its `docs/FINAL_REPORT_CN.md`
and `docs/REFACTOR_PROGRESS.md` do not map to these gates.

Branch: `effective-sampling-redesign`.

## Gate status

| Gate | Scope | Status |
|------|-------|--------|
| G0  | canonical execution closure (single `run_consensus_episode`) | 🟡 analytic episode + mechanism trace + activation sentinels done; dynamic_mc + unused-config enforcement pending |
| G1  | protocol semantics — true binary Snowball + single-node exact ref | 🟡 core done (per-node chain); joint exact ref pending (Phase 2) |
| G2  | correlated-evidence environment | 🟢 evidence model + correlation theory + geometry/graphs + evidence scenarios done (geometric weak-cut/hub deferred to G7) |
| G3  | round-coupled full physics | 🟢 two graphs + round_physics + closed loop in canonical episode; all mechanisms causal-tested + activation-sentinelled |
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

### D3 — G2/G3 geometry + two physical graphs (2026-06-24)
* `src/environment/urban_scene.py`: Manhattan grid → vehicle positions clustered on road
  segments; region `g(i)` = segment id (spatial clustering ⇒ correlated evidence). Scene
  carries `comm_radius`/`int_radius`. Deterministic given a generator.
* `src/environment/candidate_graph.py`: `RadiusGraph` + cell-list `build_radius_graph`
  (`O(N+E)`, no `N×N`, **no degree cap**); `build_candidate_graph` = `G_comm`;
  generic `aggregate_over_graph` scatter.
* `src/environment/interference_graph.py`: `build_interference_graph` = `G_int` (larger
  radius); `non_intended_interferers` = `G_int \ G_comm`; `received_interference_mw`
  aggregates over **all** transmitters near a receiver (spec §7.1 cross-destination fix —
  the legacy destination-keyed aggregation drops these).
* **Complexity:** degree bounded & constant across scale (comm ~48, int ~200 for
  N=600→6300), so `E=O(N)` at fixed density (constraint #11). Builds 4000+ nodes fast.
* **Evidence** (`tests/environment/test_physical_graphs.py`, 7 passing): radius graph ==
  brute force, no self-loops, symmetric; no degree cap (dense cluster deg 29/29);
  `G_comm ⊆ G_int`; non-intended interferer sentinel `(t,j)∈G_int\G_comm`; interference
  over `G_int` strictly exceeds comm-only aggregation (external transmitter raises floor);
  Manhattan region containment; near-linear `E/N`; multi-thousand-node build.

### D4 — G2 evidence scenarios (2026-06-24)
* `src/environment/scenarios.py`: `build_scenario(name, scene)` →
  `EvidenceModel` for `all_correct` (perfect-evidence control), `iid` (independent),
  `one_biased_region` (shared region error), `two_opposing_regions` (median-x split into
  opposite opinion clusters). Geometric scenarios (`weak_cut`, `hub_congestion`) listed in
  `GEOMETRIC_SCENARIOS` and deferred to G7 (built by geometry edits) — catalogue explicit,
  not silently missing. G2 now 🟢.
* **Evidence** (`tests/environment/test_scenarios.py`, 5 passing): all-correct `q_i=1`;
  iid zero correlation, `q_i=1-p_node`; one-biased region mostly-wrong + positively
  correlated within region, zero cross-region; two-opposing has both opinion clusters;
  unknown/geometric name raises.

### D5 — G0/G3 round physics + canonical episode (2026-06-24)
* `src/environment/round_physics.py`: spec §7.2-§7.4 single-round chain — request/response
  distinct phases, cross-destination interference over `G_int`, Mode-2 collision,
  half-duplex (duty cycle over the `W`-slot window), M/M/1 queueing (drop+delay), FBL/HARQ;
  load/quality-coupled `τ` (no `tau_proxy`); contention over `S_eff = subchannels·W`.
  Response activity gated by full request-leg delivery (real hub-overload feedback).
* `src/sampling/esp_query.py` (exact per-edge `π_ij`), `baseline_policies.py`
  (uniform/distance, observable-only).
* `src/environment/canonical_episode.py::run_consensus_episode`: THE single entry. Closed
  loop `X_t→τ_t→Π→Λ_t→γ_t,ℓ_t→(h⁺,h⁻,h⁰)→X_{t+1}` over `[N,Q,S]`; shared-latent reliability
  `F_disagree`/`F_wrong`/`S_allcorrect` (spec §4.1-§4.2); runtime mechanism trace; ablation
  flags. `dynamic_mc` mode raises (G6, must not reuse analytic marginals).
* `evidence_model.analytic_scenarios` now enumerates only **non-degenerate** regions
  (`0<p_g<1`) ⇒ iid→1 scenario, one-biased→2 (tractable analytic `Q`).
* **Validation result**: perfect/iid evidence → `S_allcorrect≈1, F_wrong≈0`; one-biased
  region → `F_wrong≈0.40` (correlated wrong decisions) — the environment produces exactly
  the spec's safety/validity failures. Op-point `k=3,α=2,β=3,r_max≥16` finalizes cleanly.
* **Evidence** (`tests/environment/test_round_physics.py` 10 + `test_canonical_episode.py`
  9, all passing): mechanism causal directions; cross-destination interference; reliability
  ordering across scenarios; accurate runtime trace; each mechanism on the canonical path
  (activation sentinel); policy never sees truth/votes; differentiable in query weights.
* **Pending:** dynamic_mc (G6), unused-config enforcement (G0 full), CVaR/deadline metric
  (objectives slice), CDQ generalization of the quorum (G4/G5).

## Next slice
G6 **independent dynamic MC** (`src/validation/dynamic_mc.py`): per-trial sampling of query
subsets, fading, request/response and Snowball counters — must NOT reuse analytic terminal
marginals (spec §8). Three-way agreement (small-`N` exact chain / analytic / dynamic MC) and
common random numbers. This is the independent judge that validates the analytic episode.
