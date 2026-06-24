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
| G1  | protocol semantics — true binary Snowball + single-node exact ref | 🟢 per-node chain + small-N exact JOINT chain; MC matches exact within CI |
| G2  | correlated-evidence environment | 🟢 evidence model + correlation theory + geometry/graphs + evidence scenarios done (geometric weak-cut/hub deferred to G7) |
| G3  | round-coupled full physics | 🟢 two graphs + round_physics + closed loop in canonical episode; all mechanisms causal-tested + activation-sentinelled |
| G4  | CDQ k-DPP subset exactness | 🟢 normalizer/subset/inclusion/sampler exact vs brute force; diagonal recovers ESP; differentiable (math layer; wired into canonical path at G9) |
| G5  | determinantal quorum exactness | 🟢 P_i(m,n) via det(I+zLD_g) grid-interpolation; exact vs brute force + recovers quorum_dp (diagonal), grad rel-err<1e-4; (wired at G9) |
| G6  | independent dynamic MC | 🟢 forward MC + ranking agreement + exact-joint-chain agreement (MC unbiased, within CI); CRN/rare-event optional refinements |
| G7  | effective-sampling diagnostics | 🟢 response-conditioned π̃, progress/drift, ESS, region mixing+spectral gap, load; all hand-scenario directions pass |
| G8  | protocol feasibility (perfect-link floors) | 🟢 well-mixed perfect-link floor (MC-validated); FEASIBLE at N≤10000 for correct-majority≥0.6; 50/50 correctly infeasible → greenlights G9 |
| G9  | model mechanism & ablations | 🟡 G9a architecture + G9b primal-dual training done (trains toward oracle, generalizes); CDQ-MC confirmation + ablations + multi-seed headline pending |
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

### D5b — adversarial verification of G0/G3 (2026-06-24)
Ran a 6-lens multi-agent adversarial review (`verify-canonical-episode` workflow, 16 agents)
with per-finding independent verification. 5 confirmed-real findings (3 were the same bug):
* **[FIXED, high]** M/M/1 queueing was keyed off the `G_int` co-channel **contender count**
  (`load_req_node`, unweighted, larger radius) instead of the **addressed** receiver load
  `Λ_j = Σ_{i→j∈G_comm} active_i π_ij` (Eq. 33). Interference/collision correctly use `G_int`,
  but a service queue only enqueues requests addressed to `j`. Re-keyed `rho` to `recv_load`;
  kept `G_int` mass for collision/interference. Added a discriminating test
  (`test_queueing_keyed_on_addressed_load_not_interferer_count`) that fails under the old
  keying. My original test missed it (used `active=1` so both quantities were large).
* **[FIXED, med]** `τ` averaged request/response slots (`(Sr+Sp)/2·(ar+ap)`) instead of the
  per-leg `Sr·ar + Sp·ap`; masked at default equal slots, wrong for asymmetric profiles.
  Now per-leg (consistent with the energy term).
* **[deferred, med]** legacy `src/mainline/model.py::evaluate_controls` still uses fixed
  `tau_proxy`/`Q=1`/scalar initial pref. It is **off the new canonical path** (nothing in
  `src/environment` or `src/sampling` imports it) and disclaimed as historical, but spec §2.2
  names it for deletion. Tracked as a legacy-cleanup slice (delete + migrate the figure/
  baseline scripts) once the new path replaces all consumers.

### D6 — G6 independent dynamic MC (2026-06-24)
* `src/validation/dynamic_mc.py::run_dynamic_mc`: genuine round-by-round FORWARD simulation
  — samples evidence per trial → initial colours; each round samples `k`-subsets (exact ESP
  ancestral sampler), samples poll success ~ `Bern(ell_poll)` (fading-marginalised), reads
  peers' **actual** current colours, advances the **true** binary-Snowball counters
  (vectorised `_step` over `[T,N]`). Reports `F_disagree/F_wrong/S_allcorrect` + Wilson CIs,
  per-node decided freqs, finalisation latency. `physics_per_trial` knob (mean-active ℓ vs
  per-trial ℓ).
* **Independence (constraint #8):** never reads analytic terminal marginals `c_ir/w_ir` nor
  calls `run_consensus_episode` (sentinel-tested); shares only the system definition (policy
  + physics model). `Bern(ell)` is the exact fading-marginalised poll outcome.
* **Key findings (honest):**
  - all-correct evidence → analytic and MC agree exactly (`S_all≈1`).
  - **ranking preserved** (spec §8.3, NOT stop-condition #3): both rank `uniform` < `distance`
    for `F_wrong` (concentrating polls in-region raises correlated wrong evidence — the §9.1
    local-quality-vs-diversity tension).
  - the analytic mean-field is **optimistic under correlation** (one-biased: analytic
    `F_wrong≈0.08-0.40` vs MC `0.28-0.46`; iid: analytic `~0` vs MC `~0.7%`, the latter
    matching an independent union-bound estimate). This **confirms the MC's calibrating-judge
    role** — the analytic is the differentiable ranking surrogate; the MC sets absolute
    safety/deadline (spec §8.2/8.3). Documented as a known surrogate property, not faked away.
* **Evidence** (`tests/validation/test_dynamic_mc.py`, 7 passing): internal validity +
  reproducibility; all-correct exact agreement; failure-tail capture; ranking agreement;
  direction vs bias; independence sentinel; per-trial-physics path.
* **Pending:** small-`N` exact joint chain (finishes G1+G6 three-way), common-random-numbers
  paired comparison, rare-event estimators (spec §8.1 lvl 4).

### D6b — adversarial verification of G6 (2026-06-24)
`verify-dynamic-mc` workflow (4 lenses + per-finding verification). The highest-risk surface
— the vectorised true-Snowball `_step` equivalence — and the independence lens both passed
**clean** (no findings). Two confirmed bugs, **both in latency accounting only** (the hard
`F_disagree/F_wrong/S_allcorrect` stats were verified correct):
* **[FIXED, med]** off-by-one: the decisive round's duration was not accrued (one-round
  finishes reported `T_all=0`; desynced from the analytic clock). Now `running` is snapshotted
  at the START of the round (before the decision commit). Regression test
  `test_latency_charges_the_decisive_round` added.
* **[FIXED, low]** `physics_per_trial=True` collapsed round duration to a global scalar
  (`tau.max()` over N and T); now `tau.max(dim=0).values` → per-trial `[T]` (broadcasts in the
  mean-field mode).

### D7 — G1+G6 small-N exact joint chain + three-way agreement (2026-06-24)
* `src/protocol/exact_small_n.py::exact_joint_terminal`: exact enumeration of the true joint
  binary-Snowball Markov chain for tiny `N` under a FIXED link `ell` (time-homogeneous;
  per-joint-state quorum via the §5 DP with peer colours as indicators; joint transition =
  `p[x]`-weighted Kronecker product of per-node transition rows). Exact terminal
  `F_disagree/F_wrong/S_allcorrect`. Cost `O(R_max·S^{2N})`, capped (`N` tiny).
* Added an **ideal/fixed-link override** (`link_override`) to `round_physics` and threaded it
  through `run_consensus_episode` + `run_dynamic_mc` (trace records it; `full_physics=False`).
  Isolates the protocol layer for the three-way check AND provides the spec §3.3 perfect-link
  feasibility mode for G8. NOT the headline path (headline asserts `link_override is None`).
* **Decisive validation**: on a tiny coupled config (N=3, k=2, ℓ=0.9) the dynamic MC's S_all
  (0.5746, CI [0.5697,0.5794]) and F_wrong **bracket the exact** (0.5779 / 0.1022) — the MC
  is an **unbiased estimator of the true process** (validates the judge, finishes G1+G6). The
  analytic mean-field shows a large gap here (S_all 0.393) — expected at tiny strong coupling
  (mean-field is exact only in the weak-coupling/large-`N` limit); direction agrees.
* **Evidence** (`tests/protocol/test_exact_small_n.py`, 5 passing, fast): perfect+all-correct
  = 1 exactly; probabilities valid + monotone in link; **MC brackets exact** (S_all & F_wrong);
  analytic mean-field gap documented + direction agrees; link_override trace marks non-headline.

### D8 — G4 CDQ low-rank k-DPP query layer (2026-06-24)
* `src/sampling/dpp_query.py`: kernel `L=BBᵀ`, `B[j]=√(qⱼ)bⱼ` (spec §9.4).
  - `kdpp_normalizer` = `e_k(λ(L))` via the exact identity `Σ_{|T|=k} det(M_T)` (principal
    minors of `M=BᵀB`, r×r) — float64-exact, differentiable, `O(C(r,k)·k³)`.
  - `kdpp_inclusion` = `π_j = q_j·∂log e_k/∂q_j` (autograd; each principal minor is degree-1
    homogeneous in `qⱼ` for `j∈S`), `Σπ=k`. Single backward → all marginals.
  - `kdpp_subset_log_prob`, `enumerate_kdpp_distribution` (brute), `kdpp_sample`
    (Kulesza–Taskar exact sampler via the r×r dual eigendecomposition).
* **Diagonal kernel exactly recovers the §4 ESP product policy** (normalizer = elementary
  symmetric, inclusion = ESP inclusion, subset = ∏q/e_k) — CDQ is a strict generalization.
* **Evidence** (`tests/sampling/test_dpp_query.py`, 9 passing): normalizer/subset/inclusion
  vs brute force (<1e-10, measured ~1e-16); `Σπ=k`; diagonal→ESP (1e-10); diversity suppresses
  similar-peer co-selection (the mechanism ESP can't express); exact sampler matches
  distribution (40k samples, <0.02); differentiable in quality+diversity; `k≤r` enforced.
* Found a float32 test bug (`torch.tensor(pyfloat)` defaults float32) — fixed; impl is exact.
* **Note (constraint #13):** CDQ is the math layer; it is wired into the canonical query path
  when the ESD-GNN emits `(q, b)` at G9. Until then the canonical query remains the ESP policy.

### D8b — adversarial verification of G4 (2026-06-24)
`verify-cdq-kdpp` workflow (11 agents). Core math confirmed exact; 7 real **robustness** edge
-case defects found + all fixed (none affected the validated exact values, but all matter once
CDQ trains at G9 with padding/large logits/inference): (1) **[high]** `sqrt(0)` infinite
gradient in `low_rank_kernel` → now `clamp_min(eps)` (saturates → zero grad for masked/padded
candidates); (2) `kdpp_inclusion` crashed under `torch.no_grad()` → wrapped in
`enable_grad()`; (3) `k≤min(d,r)` now enforced (was only `k≤r`; `d<k` returned float noise);
(4) `kdpp_log_normalizer` now scale-stable (factors mean eigenvalue; no overflow at large
quality, still recovers ESP); (5) `k=0` inclusion short-circuits to zeros. 5 regression tests
added (no_grad, k>min(d,r), k=0, large-quality stability+ESP recovery, zero-quality finite
grad); 14 G4 tests passing.

### D9 — G5 CDQ determinantal heterogeneous quorum law (2026-06-24)
* `src/sampling/determinantal_quorum.py`: exact `P_i(m,n) = [zᵏxᵐyⁿ] det(I+z L D_g)/e_k(λ(L))`
  (spec §9.5). Low-rank: `[z^k]det(I_r+zM(x,y)) = e_k(λ(M(x,y)))`, `M=BᵀD_g B = C0+C+x+C-y`
  (r×r linear in x,y). Computed EXACTLY (no sampling) by evaluating that bivariate degree-≤k
  polynomial on a (k+1)² grid — each point is the G4 principal-minor normalizer of a numeric
  r×r matrix — and recovering coefficients via the inverse Vandermonde (2-D poly interpolation).
  `determinantal_quorum_decision` → `(h⁺,h⁻,h⁰)`; `bruteforce_determinantal_quorum` reference.
* **Diagonal kernel recovers `quorum_dp` exactly** (the §5 elementary-symmetric quorum) — CDQ
  generalizes both the ESP query (G4) and its quorum (G5).
* **Evidence** (`tests/sampling/test_determinantal_quorum.py`, 7 passing): P(m,n) & (h⁺,h⁻,h⁰)
  vs brute-force subset×ternary enumeration (<1e-10, dual-referenced); diagonal→quorum_dp
  (1e-10); unreachable `m+n>k` support ~0; gradient rel-err <1e-4 vs central differences;
  batched+differentiable; strict majority enforced. Vandermonde conditioning verified to k=6
  (maxerr 1.1e-12). Exactness boundary: exact for degree-≤k (only float64 Vandermonde
  conditioning, <1e-12 measured); not a sampled/straight-through surrogate.
* CDQ math layer complete (G4+G5); wired into the canonical query/quorum path at G9 when the
  ESD-GNN emits `(q, b)`. 21 sampling tests passing.
* **Verification:** G5 rests on TWO independent exact references (brute-force enumeration +
  the validated `quorum_dp` on the diagonal) agreeing at <1e-10, gradient FD <1e-4, and 5
  machine-precision edge-case probes (k=1, p0=0, degenerate kernel, quality~1e6, no-response)
  — no separate adversarial workflow needed (already triangulated).

### D10 — G8 perfect-link feasibility (2026-06-24) — VIABILITY GATE PASSED
* `src/validation/feasibility.py`: well-mixed perfect-link Snowball floor — self-consistent
  binomial-quorum recursion reusing the `binary_snowball` components (`wellmixed_terminal`),
  network floors over N exchangeable nodes via inclusion-exclusion (`network_floors`, log
  domain), `is_feasible`, `scan_feasibility`.
* **Result (greenlights G9, no stop-condition #1):** FEASIBLE at N∈{100,1000,10000} for any
  correct-majority init ≥ 0.6 with modest params (β=2–4, k=3–7, R_max=8–16); floors met with
  margin (F_wrong ~1e-6–1e-8 vs ε_v/10=1e-4). The 50/50 tie is correctly INFEASIBLE (gate has
  teeth). Exponential β-safety confirmed (per-node wrong w: 1.8e-7 @β=3 → 1.5e-27 @β=5).
* **Verification (the rigor):** the well-mixed recursion is validated against the INDEPENDENT
  dynamic MC in a complete-graph perfect-link setting at a measurable rate (recursion c/w match
  MC within finite-N bias). Exactness boundary: well-mixed mean-field floor (idealised best
  case, NOT the local-topology headline evaluator); the deep ~1e-7 tail rests on the
  MC-validated recursion + classic exp(-β) Snowball safety — direct MC of 1e-7 needs rare-event
  methods (spec §8.1 lvl 4).
* **Evidence** (`tests/validation/test_feasibility.py`, 7 passing): monotone in β/init; floor
  formulas; scan feasible at target N; 50/50 infeasible; recursion matches dynamic MC; +2
  regression tests from the `verify-feasibility` workflow.
* **D10b adversarial verification** (`verify-feasibility`, 8 agents): floor model sound; 2 LOW
  defects fixed — (1) `wellmixed_terminal` now guards the strict-majority precondition
  (`2α>k`; non-strict α silently inflated mass via the `h_zero` clamp); (2) `F_disagree` now
  uses the log-domain `1-(1-w)^N` form (was naive subtraction with ~1e-7 cancellation, below
  the gate but a docstring/code mismatch). No verdict impact; viability conclusion unchanged.

### D11 — CDQ wired into the canonical path (G9 infra) (2026-06-24)
* `src/sampling/cdq_query.py`: `cdq_edge_inclusion` (per-source bucketed `kdpp_inclusion` → π,
  Σπ=k) + `cdq_bucketed_quorum` (per-source bucketed `determinantal_quorum_decision` → h⁺,h⁻,h⁰
  with p⁺=ℓu, p⁻=ℓv, batched over Q). Masked/padded slots excluded EXACTLY by zeroing kernel
  rows (zero B row ⇒ no contribution, inclusion 0). `r≥k` (need not exceed degree — low-rank
  dual handles w>r). `DiagonalCDQPolicy` reproduces an ESP policy via diagonal kernel.
* `run_consensus_episode` now branches on `query_law` ("esp" default | "cdq"): CDQ policies emit
  `(quality[E], diversity[E,r])` via `.kernel(graph)`; same physics+protocol; trace records
  `query_law`. **CDQ (G4/G5) is now on the canonical headline path** (resolves constraint #13).
* **Evidence** (`tests/sampling/test_cdq_wiring.py`, 4 passing): bucketed inclusion/quorum match
  the single-source CDQ math (1e-10); **diagonal-CDQ episode == ESP episode bit-for-bit**
  (S_all/F_wrong/F_disagree/c_ir, 1e-9) — the wiring anchor; real non-diagonal CDQ episode runs
  and is differentiable in `(quality, diversity)`. 25 sampling tests passing.

### D12 — Phase 7 topology-oracle ceiling (stop-condition #2) — PASSED
* `src/optimization/topology_oracle.py`: direct per-scene edge-logit optimizer
  (`optimize_logweight_topology`, Adam on the analytic `F_disagree+F_wrong`) + `oracle_vs_
  heuristics` comparison with dynamic-MC confirmation. `LearnedLogWeightPolicy` (ESP, MC-
  supported). The oracle is a per-scene UPPER BOUND (uses the scene's evidence via the
  objective; NOT deployable — the GNN must approach it from observables, constraint #10).
* **Result (clears stop-condition #2; justifies G9):** on one-biased-region (perfect link,
  isolating the peer-selection lever) the oracle drives `F_wrong` 0.041→~0 analytically; the
  **distance heuristic is WORSE than uniform** (0.150 vs 0.041 — concentrating on nearby
  same-region peers = the §9.1 redundancy tension). Independent dynamic MC (5000 trials, 3
  scene seeds): oracle `F_wrong≈0.003` vs uniform `≈0.117`, **MC gain 0.110/0.118/0.123**,
  CIs cleanly separated → significant, MC-confirmed topology lever.
* **Evidence** (`tests/optimization/test_topology_oracle.py`, 3 passing): oracle ≫ heuristics
  analytically + non-overlapping MC CIs; distance worse than uniform under correlation; loss
  decreases. Manifest: `result/topology_oracle_ceiling/ceiling.json` (multi-seed, verdict PASS).
  Verified by the independent MC (multi-seed) — no separate workflow needed.

### D13 — G7 effective-sampling diagnostics (2026-06-24)
* `src/sampling/effective_dynamics.py` (spec §5, the namesake layer): `response_conditioned_
  marginal` (π̃=πℓ/Σπℓ, §5.2), `progress_drift` (g=h⁺+h⁻, Δ=h⁺−h⁻, ν_prog=g/τ, ν_drift=Δ/τ,
  §5.4-5.5), `effective_sample_size` (k_eff=1/(wᵀR_i w) with R from `pairwise_correlation_
  theory`, §5.6), `region_response_kernel`+`cross_region_response_mass`+`region_spectral_gap`
  (additive reversibilization, §5.7), `receiver_load` re-export (Λ). All differentiable, O(E)
  (ESS O(Σdeg²)), no N×N. Diagnostics/aux signals only — do not replace safety/deadline.
* **Evidence** (`tests/sampling/test_effective_dynamics.py`, 6 passing): all hand-scenario
  directions hold — symmetric loss→progress↓ (drift~0); opinion split→drift moves ±;
  redundant (correlated) peers→ESS↓; weak cut→cross-region mass↓ + spectral gap↓; hub→load↑.
  Reuses validated components (quorum_dp, pairwise-correlation, receiver_load) — direction
  -tested, no separate workflow. 31 sampling tests passing.
* These feed the G9 primal-dual auxiliary losses (spec §5.8).

### D14 — G9a ESD-GNN architecture (2026-06-24)
* `src/models/esd_gnn.py`: scene/scale-agnostic multi-graph encoder (spec §9.3) — per layer:
  source-side candidate competition + dest-side incoming load (G_comm), interference
  aggregation (G_int), vehicle↔region mean-pool/broadcast (region supergraph / correlation
  channel), residual+LayerNorm; `n_enc=3` layers + `n_refine=2` dynamics-in-the-loop
  refinements that feed the current kernel's inclusion-derived receiver load `Λ` back (spec
  §9.6) → CDQ heads (`quality=softplus+floor>0`, `diversity∈Rʳ`). `ESDGNNQueryPolicy`
  (`query_law="cdq"`). **Features are STRUCTURAL/observable only** (log-degrees, region size,
  distance, LOS, same-region — no `Y*`/vote/scene-ids), so one model transfers across N and
  scales; O(E) scatter, no N×N.
* **Evidence** (`tests/models/test_esd_gnn.py`, 6 passing): valid differentiable kernel;
  **no-truth-leak sentinel** (kernel identical across evidence biases — constraint #10); runs
  in the canonical CDQ episode with end-to-end gradient to model params; transfers across
  scales; kernel depends on observable structure; refinement load-feedback non-trivial.
  Architecture only (untrained); training quality is G9b. 107 tests passing.

### D15 — G9b primal-dual reliability-constrained training (2026-06-24)
* `src/optimization/primal_dual.py`: augmented-Lagrangian `L=ET+λE+Σμ_r(F_r−ε_r)` with dual
  ascent `μ_r←[μ_r+η(F_r−ε_r)]_+` (spec §4.5). `episode_metrics` (differentiable ET=Σ_t
  round_duration·P(not-all-correct), F_deadline=1−P(all-correct by deadline round) from the
  episode trajectory), `lagrangian`, `train_esd_gnn`. ET is the trainable analytic surrogate
  for `CVaR_q(T_all)`; the headline CVaR is the MC's job (G11).
* **Demonstration:** training the G9a ESD-GNN reduces `F_wrong` 0.015→0.0012 toward the D12
  oracle (0.0), `μ_v` actively enforcing the constraint — from OBSERVABLE features only.
* **Evidence** (`tests/optimization/test_primal_dual.py`, 5 passing): metrics+lagrangian
  differentiable; dual ascent direction correct (μ↑ when F>ε, ≥0 floor); training reduces
  failure <0.5× and ≤ oracle+0.02; generalises to a held-out scene; reproducible. 112 tests.

### D16 — CDQ dynamic MC (2026-06-24)
* `src/validation/dynamic_mc.py`: `run_dynamic_mc` now branches on `query_law` — for `"cdq"` it
  computes inclusion via `cdq_edge_inclusion` and samples subsets with `_CDQSubsetSampler`
  (per node: enumerate the exact k-DPP subset distribution once, draw fresh subsets each round
  by multinomial — exact, fast for dev-scale; raises above `max_subsets`). The trained CDQ
  ESD-GNN can now be externally MC-confirmed (constraint #14, spec §8.3). ESP path unchanged.
* **Evidence** (`tests/validation/test_cdq_mc.py`, 4 passing): CDQ sampler draws exactly-k
  distinct peers; **diagonal-CDQ MC reproduces the validated ESP MC** (CIs overlap, S_all
  within 0.03) — the consistency anchor against the adversarially-verified ESP sampler; CDQ MC
  agrees with the analytic CDQ episode on all-correct; runs for a real ESD-GNN + reproducible.
  116 tests passing.

### D17 — G9c ablation flags + the CDQ-vs-ESP finding (2026-06-24)
* `src/models/esd_gnn.py`: ablation switches `use_cdq` / `use_region` / `use_interference`
  (+ `n_refine`); `use_cdq=False` makes the policy ESP (`log_weights=log(quality)`, diagonal
  kernel, diversity unused — clean). `ESDGNNQueryPolicy.query_law` follows `use_cdq`.
* **Evidence** (`tests/models/test_esd_gnn_ablations.py`, 3 passing): each flag is RUNTIME
  -ACTIVE (changes the model output); no-CDQ switches query law to ESP; all variants valid+diff.
* **Ablation result (analytic, dev scale, one-biased-region):** `no_region` is **26× worse**
  (CR=88: 0.091 vs full 0.0035) — the region/correlation channel is the dominant mechanism.
  `no_refine` modestly worse. **BUT `no_cdq` (ESP) ≈/slightly-better than full CDQ** (0.0009 vs
  0.0035) — the central CDQ diversity shows NO analytic advantage.
* **Why (key methodological insight, recorded honestly per spec §12 / no-overclaim):** the
  **mean-field analytic episode is correlation-blind** — the quorum uses each selected peer's
  marginal `u_j` and treats selections as independent, so diverse vs redundant peers with the
  same `u_j` give identical analytic `h`. CDQ's diversity benefit (independent evidence ⇒ lower
  joint-error correlation) only manifests in the JOINT/MC dynamics. G9b trained on the analytic
  objective ⇒ CDQ had no gradient signal for diversity. The fix is the spec's own §5.8 ESS
  (`k_eff`) auxiliary loss (built in G7, encodes the correlation `R_i`) as the diversity
  training signal, with the MC as evaluator (§8.3). NOT a stop — the path is in-spec.

### D18 — CDQ fair-test investigation ⇒ STOP+report (central-claim direction decision)
Investigated whether CDQ's determinantal diversity can beat region-aware ESP. **Three
independent obstacles found, all evidence-backed:**
1. **Analytic blindness** (D17): the mean-field episode's quorum uses each peer's marginal
   `u_j` and treats selections as independent ⇒ diverse vs redundant peers give identical
   analytic `h`. No analytic gradient for diversity.
2. **The k_eff signal is non-differentiable.** `effective_sample_size` (the §5.6/§5.8 ESS,
   the *only* correlation-aware quantity) detaches the gradient (`float(w@R@w)`, line 95) —
   a diagnostic, not a usable training loss. The §5.8 aux loss needs a differentiable rewrite.
3. **The environment correlation is region-block / near-exchangeable.**
   `pairwise_correlation_theory`: cross-region corr = exactly 0; same-region corr depends only
   on the peers' marginal error rates. So the only correlation structure is region-level — which
   region-aware ESP already exploits (D17: no_region 26× worse). CDQ's extra lever (within-group
   heterogeneous-correlation diversity) is near-absent. Plus ESP with free weights can select any
   single best subset deterministically; CDQ's edge is only multi-round diverse *coverage*.
**Conclusion:** the CDQ-superiority claim is **unestablished and structurally obstructed in the
current §6.2 region-block design.** Not a defect to silently fix — it reshapes the headline, so
per spec §12 (no overclaiming) + stop-condition #6 it is a user direction decision. The exact CDQ
math (G4/G5) + region-aware ESD-GNN (G9a/b) remain valid and verified regardless.

### D19 — user chose **C (defer CDQ to scale)** + G10 large-scale complexity ✅ (2026-06-24)
User direction on D18: **C — proceed to G10/G11 first, keep both CDQ and ESP variants alive,
decide the CDQ-vs-ESP framing after seeing whether multi-round diverse-coverage effects emerge
at scale.** So the CDQ contribution question is OPEN (not reframed); G10/G11 run on both variants.
* **G10 large-scale complexity** (`scripts/analysis/scaling_benchmark.py` →
  `result/scaling/scaling.json`; `tests/validation/test_scaling.py`, 3 passing): the FULL
  canonical pipeline (radius graphs + ESD-GNN forward + dynamic MC) scales **near-linearly** N=96
  →9520: avg degree stays **bounded** (11.0→13.7 as N grows ~100×) ⇒ E=O(N) from the local radius
  (no degree cap needed, constraint #4); build/GNN/MC all ~linear; **per-trial-per-edge MC cost is
  ~constant (9.7→13.0 µs)**; N=9520 (E=130k) runs in seconds — **no N×N tensor** (#11). The
  analytic's 2^G scenario enumeration is intractable at scale (G~thousands) ⇒ confirms the **MC is
  the only large-N headline evaluator** (§8.3). **Stop-condition #4 (near-linear cost) does NOT
  trigger.** 122 tests passing.

### D20 — G11 headline verdict: ESD-GNN ≫ heuristics; **CDQ does NOT beat ESP at scale** (2026-06-24)
Ran the paired-CRN headline (`scripts/analysis/headline_comparison.py` → `result/headline/
headline.json`): 3 model seeds × {CDQ, ESP}, trained on N=48 scenes, evaluated on **16 disjoint
held-out N=336 scenes**, 350 MC trials, paired CRN, ideal link (topology isolation). Mean F_wrong:
uniform 0.106, distance 0.232, **esd_gnn_cdq 0.054, esd_gnn_esp 0.017**.
* **ESD-GNN ≫ baselines (PASS):** paired vs uniform — esd_gnn_esp **−0.088 [−0.102,−0.074]**,
  esd_gnn_cdq −0.052 [−0.078,−0.020], both significant & better; distance +0.127 (significantly
  *worse* — §9.1 tension). Trained N=48 → deployed N=336 ⇒ **scale transfer works** (#3/#5).
* **CDQ vs ESP (the deferred D18 question, decided):** paired diff CDQ−ESP **+0.036
  [+0.021,+0.056], significant ⇒ ESP is significantly BETTER than CDQ.** The user deferred to
  scale specifically to test whether multi-round diverse-coverage would help CDQ; **it does not.**
  Confirms the D18 structural analysis empirically and rigorously.
* **Verdict:** the deployable headline policy is the **region-aware ESD-GNN in ESP mode**; the
  determinantal-diversity CDQ provides no reliability advantage in the region-block V2X regime.
  The exact CDQ math (G4/G5) stands as a valid generalization (ESP = its diagonal special case).
  → Framing decision now due (the one the user deferred): **B reframe** (honest: exact CDQ math +
  region-aware ESD-GNN + the negative diversity finding) vs **A pursue** (finer-than-region
  correlation §6.3 to give CDQ a lever). Asking the user.

### D21 — framing decision: **B (reframe honestly + finish)** (2026-06-24)
User chose **B**. The project's contribution is reframed (honestly, on the verified results):
1. **Exact determinantal heterogeneous quorum / CDQ math (G4/G5)** — a correlation-aware
   generalization of independent ESP sampling (ESP = the diagonal special case), exact to machine
   precision, near-linear cost. A theoretical contribution.
2. **Region-aware multi-graph ESD-GNN (G9)** — the deployable headline policy, **in ESP mode**:
   significantly beats heuristics (−0.088 F_wrong vs uniform) and transfers train-small/deploy
   -large. The empirical systems contribution.
3. **Rigorous negative result** — determinantal diversity gives NO reliability advantage over
   region-aware ESP in the region-block V2X regime, with the structural explanation (D18: analytic
   correlation-blindness; near-exchangeable region-block correlation; ESP can select any best
   subset). Honest, explanatory, publishable.
**Canonical deployable policy = ESD-GNN with `use_cdq=False`.** CDQ stays as the exact-math layer,
NOT claimed as empirically superior. Finer-correlation pursuit (option A) parked as future work.

### D22 — G11 full-physics confirmation + energy/latency metrics ✅ (2026-06-24)
* **MC extended** (committed 10b5535): per-trial energy + CVaR_0.9 tail latency/energy
  (`_cvar_upper`); `tests/validation/test_mc_energy_latency.py` (3 passing). Headline harness
  records energy + latency_cvar; `evaluate_policies_paired(verbose=)` logs per-scene progress.
* **Full-physics headline** (`link_override=None`, real chain, N=96, 2 model seeds, 6 held-out
  scenes, 100 trials, paired CRN — a CONFIRMATION; the publication run is the same script with
  N_MODEL_SEEDS≥5/N_HELDOUT≥30, a multi-hour offline job beyond the 10-min dev cap). Mean F_wrong:
  uniform 0.112, distance 0.242, esd_gnn_cdq **0.110**, esd_gnn_esp **0.021**.
  - **esd_gnn_esp ≫ baselines:** −0.091 [−0.115,−0.067] vs uniform (sig, better). The robust
    deployable headline policy under the full physical chain.
  - **CDQ collapses under real physics:** esd_gnn_cdq −0.002 [−0.063,+0.088] vs uniform — **NOT
    significant** (≈ uniform) and **3× slower latency** (0.093 vs ESP 0.030). CDQ vs ESP: +0.089
    [+0.024,+0.172] sig ⇒ ESP decisively better. New honest insight: diversity picks worse
    -connected peers, hurting reliability AND latency under realistic links.
  - distance significantly worse than uniform (§9.1 tension — robust across link modes).
* **G11 verdict (robust across ideal-link D20 + full-physics D22):** the region-aware ESD-GNN in
  **ESP mode** is the headline policy — significantly beats heuristics, scale-transferable
  (N=48→336), robust to link noise. Determinantal CDQ provides no reliability benefit and *hurts*
  under full physics. Confirms the D18/D21 framing **B**. **G11 ✅ at dev rigor**; publication-grade
  seed/scene counts parameterized.

## Next slices (planned order)  — direction **C** + framing **B**; G10 ✅; **G11 ✅** (dev rigor).
Viability gates passed (#1 G8, #2 oracle); G7 ✅; G9a/b ✅; CDQ-MC ✅; G9c ✅; G10 ✅; G11 ✅.
1. **G12 temporal robustness** — evaluate the trained ESD-GNN(ESP) under temporal scene drift
   (vehicle mobility between rounds / across episodes); confirm reliability holds.
2. **Finalize**: honest report reflecting framing B; **G0 closure** (unused-config enforcement);
   legacy cleanup (delete `src/mainline/model.py::evaluate_controls`, migrate any figure scripts).
3. **(offline)** publication-grade G11: ≥5 seeds, ≥30 scenes, full physics, high trials.
1. **Framing decision (A/B)** then **G11 full rigor** (≥5 seeds, ≥30 scenes, full physics, CVaR
   latency + energy). 2. **G12** temporal robustness. Legacy cleanup: delete
   `src/mainline/model.py::evaluate_controls`.
* **(A) Pursue CDQ fairly** — differentiable k_eff rewrite (§5.8 aux loss) + enrich the env with
  finer-than-region correlation (§6.3 common-cause sensor-source groups making same-region peers
  non-exchangeable) + retrain + MC-eval. Multi-iteration, uncertain payoff, changes the headline env.
* **(B) Reframe honestly** — contribution = exact determinantal CDQ math (generalizes ESP) +
  region-aware ESD-GNN (beats heuristics, approaches oracle) + the honest finding that
  determinantal diversity ≈ region-aware ESP in region-block V2X. Then finish G10/G11 (scale +
  reliability headline) on this honest framing.
* **(C) Defer** the CDQ question to G10/G11 large-scale first (multi-round coverage may differ),
  decide after seeing scale results.
* Independent of A/B/C: **G10/G11** large-scale (N=100–10000) complexity + reliability-constrained
  superiority (dynamic-MC headline; ≥5 model seeds, ≥30 scene seeds, paired CRN, multiple-
  comparison); **G12** temporal; legacy cleanup (`src/mainline/model.py::evaluate_controls`).
