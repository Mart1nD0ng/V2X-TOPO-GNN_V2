# Non-Distance Headroom (NDH) Benchmark — Final Synthesis (G-NDH-SYNTHESIS)

**Campaign:** build a more deployment-realistic V2X consensus benchmark with **non-distance** mechanisms,
then use an **oracle-first** protocol to decide whether a learned topology (ESD-GNN) can legitimately beat
the distance heuristic under **matched reliability** — before training any GNN.

**Verdict:** across four realistic non-distance mechanisms, **distance remains the reliability-feasible
per-edge optimum**. No physics-changing NDH regime opens matched-reliability headroom for a diagonal ESP
policy, so **no GNN was trained** (the oracle-first gate correctly gated it). This reproduces the old-physics
EV12–15 conclusion in a deliberately non-distance-correlated environment.

Spec: `docs/NON_DISTANCE_HEADROOM_TECHNICAL_SPEC.md` + `..._ENGINEERING_PLAN.md`. Parameters:
`docs/NDH_PARAMETER_REGISTRY.md`. Decision log: `docs/ESP_SCALE_V2_PROGRESS.md` (EV16–24).

---

## 1. Parameter registry

All SPS / RSU / capacity / CSI / hotspot parameters are catalogued in
[`docs/NDH_PARAMETER_REGISTRY.md`](../../NDH_PARAMETER_REGISTRY.md) with `default / sweep / stress /
source / deployment-or-stress label`, the FROZEN preserved baseline physics, hard density/fraction caps,
the C2 observability split (deployable proxy vs oracle-only truth), and two assembled profiles
(`NDH-DEPLOYMENT` headline / `NDH-STRESS` reported separately). No stress value is a deployment default.

## 2. Mechanism evidence (each ACTIVE + non-degenerate, on the canonical full-physics dynamic-MC path)

| Mechanism | Gate | Enters physics? | Active + non-degenerate evidence |
|---|---|---|---|
| **Intersection hotspots + capped RSU + nonuniform road** | EV17 | scene geometry | 14 tests + 1500-scene fuzz; hard caps enforced (RSU ≤ 0.5 int-prob & ≤ 0.10–0.15 fraction; hotspots ≤ 10% intersections & ≤ 30% vehicles); `omega_RSU=0`; RSU-evidence-invariance = 0 |
| **SPS persistent collision** | EV18 | `round_physics` collision (same-bucket `G_int`) | MC: congested band (S_sps=40, τ=1) collapses Pc→~0; deployment band (S_sps=100, τ=4) non-degenerate; SPS-off byte-identical; 3-lens review confirmed physics correct |
| **CSI aging** | EV19 | FEATURE-only (physics keeps current γ) | uncertainty monotone in age; age=0 recovers current CSI; no train/eval physics change |
| **Heterogeneous receiver capacity** | EV20 | `round_physics` M/M/1 queue (per-node μ_j) | RSU μ > vehicle μ; homogeneous byte-identical; band @R_d=20 non-degenerate (binds via delay); 3-lens review: 0 critical/0 major |

Feature schema (EV21, `SceneFeaturesV2` + `ESDGNNStaticV2`): 13 node + 14 edge DEPLOYABLE columns +
availability mask; **leak-verified CLEAN** by a 3-lens review (capacity = noisy `log μ̂` never true μ; CSI
stale not current; degrees candidate-graph not policy-driven; no Y*/vote/MC/future).

## 3. Baseline envelope map (EV22)

11 strong heuristics as diagonal ESP policies reading the SAME deployable proxies as the GNN
(`src/evaluation/ndh_baselines.py`). Diagnostic (non-collapse band, directional): the reliability-bought
pattern already appears — heuristics beating distance on Pc (`distance_plus_resource` +0.040) raise
F_wrong; at matched reliability the best heuristic beats distance by only ~+0.003 (≈ parity).

## 4. Oracle headroom map (EV23) — the decisive measurement

Free-edge oracle (UPPER BOUND on any diagonal ESP policy, incl. a GNN) vs distance + best-heuristic
envelope, per physics-changing regime (CSI excluded — feature-only). 2 regimes × 2 scenes × λ∈{0,5}:

| regime | un-gated oracle Pc gain vs distance | reliability-bought? | **iso-reliability gap** |
|---|---|---|---|
| capacity (RSU+hotspots+μ_j, R_d=20) | +0.007 / −0.002 (≈ 0) | dFw +0.006 (never matches distance) | **none feasible → no headroom** |
| SPS (S_sps=40 @ S_phys=400, R_d=10) | **+0.024 / +0.038** | **dFw +0.013 / +0.016 (fully reliability-bought)** | **−0.056 vs distance, −0.0485 vs envelope** |

**`train_gnn = False`.** Capacity gives ~0 un-gated gain. SPS gives a real un-gated gain (+0.024–0.038)
but it is entirely reliability-bought; the two-point frontier {0,5} brackets the matched-reliability gap
(λ=0 above-but-higher-Fw, λ=5 below-with-lower-Fw), so the matched gap is ≈ parity-to-negative — never the
>+0.02 required. The per-scene oracle is an upper bound, excluded from deployable claims.

## 5. Decisions

* **Static ESD-GNN v2 — SKIPPED** (plan §16.3 stop condition #3: constrained oracle finds no matched-
  reliability headroom). Training a GNN would at best reproduce parity; the free-edge oracle already proves
  no diagonal law can beat distance at matched reliability, so there is nothing for the GNN to learn that a
  strong heuristic + distance don't already achieve. Not manufacturing a win.
* **Temporal-ESDGNN v2 — N/A** (not needed). A STATIC free-edge per-edge oracle already upper-bounds ANY
  diagonal (per-edge) query law, including a history-aware one: temporal memory can only change WHICH per-edge
  weights are chosen, and the free-edge oracle already searched all per-edge weightings. History cannot open
  per-edge headroom the free per-edge optimum lacks (plan §11 trigger not met).

## 6. Paper-claim recommendation

**Distance is the reliability-feasible per-edge optimum EVEN under added non-distance physics.** The deep
reason: this is a **wrong-basin-risk-dominated basin-consensus** task — the hard constraint is
`macro_F_wrong`, and ANY topology that raises `P_correct` (by polling more/faster/farther peers) also raises
wrong-basin risk. So at MATCHED reliability the per-edge optimum is ~distance-like **regardless of which
non-distance cost (collision persistence, capacity heterogeneity, contention) is added** — the added cost
shifts the absolute operating point but not the *reliability-feasible* per-edge frontier, on which nearest/
best-link polling is (near-)optimal.

**Where a legitimate learned-topology win could still come from (NOT per-edge topology):**
1. A **non-diagonal query law** — CDQ2 k-DPP **diversity** (polling globally-diverse, low-redundancy peers
   rather than locally-best ones). The free-edge oracle here is diagonal; it cannot represent diversity, so
   it does NOT upper-bound a diversity law. This is the single most promising unexplored direction.
2. A **task objective in genuine tension** — e.g. correlated-redundant evidence where the locally-best peer
   is globally redundant, or a multi-objective point where latency/energy trade against reliability
   (EV15 showed the current objectives are aligned, so this needs a new regime).

**Bottom line for the thesis:** the ESD-GNN's value proposition is NOT per-edge topology superiority over
distance in reliability-first consensus — the oracle caps that at parity across both the original physics
(EV12–15) and the new non-distance physics (EV23). Its value, if any, lies in the **diversity (CDQ2)** law
or a task with genuine multi-objective tension — which should be the next campaign.

---

*Compute-limited scope (stated honestly): 2 physics regimes × 2 scene seeds × λ∈{0,5} × 1000 eval trials;
per-scene directional (not CI-definitive). The direction is unambiguous and consistent across regimes,
scenes, and with the independent EV12–15 old-physics result; a wider λ grid / more seeds would tighten the
matched-reliability gap around parity but cannot produce the >+0.02 headroom absent here. All results
hash-bound (`ndh_oracle_frontier_results.json` manifest). The dynamic-MC macrostate basin judge was never
weakened; full physics throughout; no truth/future leak into any deployable policy.*
