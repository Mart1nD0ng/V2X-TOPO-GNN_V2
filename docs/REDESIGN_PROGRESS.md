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

## Next slice
**G8 perfect-link feasibility scan** (`src/validation/feasibility.py` + a scan script, spec
§3.3, plan Phase 5) — the VIABILITY gate (loop stop-condition #1). Using `link_override=1`
(ideal link) calibrate `(k,α,β,R_max)` so the perfect-link floors meet `F_·^floor ≤ ε_·/10`
at the target `N` under the correlated-evidence scenarios; freeze `T_d`. If infeasible →
STOP + report (cannot let the GNN rescue an infeasible protocol). Then G7 effective-sampling
diagnostics (spec §5: response-conditioned `π̃`, progress/drift, ESS, mixing, load with
hand-scenario direction tests), then G9 ESD-GNN (wires CDQ into the canonical path).
