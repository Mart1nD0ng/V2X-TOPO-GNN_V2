# Structural Delay Model — Making D Topology-Controllable via Link Reliability (Q-function) — Design Contract & Technical Package

> **OUTCOME (2026-06-04) — D-fix-A IMPLEMENTED & RE-TEST PASSED: "coupled C/D/E" EARNED.**
> See `result/de_ablation_v4_structural/RESULT.md`. With `delay.structural_delay: true` +
> `--governance pcgrad`, N=2000/200, seeds 7/42/123: **D is now topology-controllable —
> `delay_relative_reduction` = +32.7 / +34.1 / +33.8%** (was ≤0.83% with round-count D, and
> gradient governance could not move it in v3). SD-G1 ✅3/3, SD-G2 (C/E preserved) ✅3/3,
> SD-G3 (PCGrad stability incl. seed 123 full F=0.0255) ✅3/3. SD-G4 (independence) is noise
> around 0 by design — **D ∝ E confirmed** (per-arm D/E ratio constant ≈ 363–364), so D and E
> are coupled levers (Option A), not independent. Headline upgraded **C/E → C/D/E
> (link-quality-controllable, D∝E)**. Status line below is historical.

Status: **design contract + technical package** (pre-implementation, house style of
`docs/COUPLING_AND_OPERATING_POINT_DESIGN.md` / `docs/GRADIENT_GOVERNANCE_DESIGN.md`).
This is the **off-ramp confirmed by the gradient-governance work**: Track G2
(`result/de_ablation_v3_governed/RESULT.md`) proved D is **not** conflict- or
magnitude-bound (PCGrad gave `delay_heavy` w_D=10 + no E-conflict + de-confliction and D
still moved ≤0.2%), so no gradient method can rescue D. D is **structurally** immovable
because the metric the loss optimizes is a pure consensus-round count.

- **Goal:** make delay **D topology-controllable** by redefining it as a *structural*
  quantity built on the **existing link-reliability + finite-blocklength Q-function**, so
  edge selection has real leverage over D — mirroring the **proven** retransmission-aware
  energy fix (B-fix-2) that made E controllable (`result/de_ablation_v2/`: E −27…65%).
- **Keep PCGrad:** the re-test runs `--governance pcgrad` (the earned G1 stability win);
  the structural delay is an *evaluator metric* change, orthogonal to and composable with
  the gradient-plumbing PCGrad layer.

> **HONEST HEADLINE FINDING (decide the scope before implementing).** The cheapest
> link-reliability delay — per-hop ARQ retransmission delay `n_tx = 1/link_success` — is
> **algebraically proportional to the retransmission energy already in the evaluator**
> (both are `expected_rounds × weighted_mean(n_tx) × const`). So **D-fix-A makes D
> controllable but D ∝ E (perfectly correlated): D becomes a lever, but NOT one independent
> of E.** That still earns "**coupled C/D/E**, all link-quality-controllable" (a real
> upgrade from "C/E, D inert"), but the D⊥E independence gate (B-G3) fails *by
> construction*. To make D an **independent** lever you need a delay term E does not share —
> a **hop-count / path-length** structural delay (D-fix-B), which is more invasive. This
> contract specifies **D-fix-A as the primary, recommended step** (it is exactly "based on
> link reliability + Q-function", cheap, differentiable, proven mechanism) and **D-fix-B as
> a scoped optional extension** for independence.

> **DECISION (2026-06-04) — OPTION A CHOSEN.** Accept the D∝E coupling: claim **"coupled
> C/D/E, physically coupled through retransmissions"**. D-fix-B (independent hop/path delay)
> is **NOT pursued**. Consequences locked in below: (1) **C/D/E must be EARNED, not
> relabeled** — the existing "C/E" used the *round-count* D (inert, uncorrelated); you must
> implement D-fix-A *and* re-test that the structural D actually moves (SD-G1) before
> claiming C/D/E. (2) **SD-G4 (D⊥E independence) is informational only** — its expected
> FAIL is the *accepted design*, not a defect; D and E are reported as a physically-coupled
> reliability↔latency↔energy trade, both controlled by link quality (different units/targets,
> reported separately, as is standard). (3) Frame honestly in the writeup that
> `delay_relative_reduction ≈ energy_relative_reduction` because both ride the shared
> retransmission lever `n_tx = 1/link_success`.

---

## 0. Why D is inert today (mechanism, from the code)

In `src/evaluation/v2x_consensus_bridge.py` the delay the loss optimizes is a **pure round
count** with no structural dependence (line ~644):

```python
expected_rounds = avalanche_result["node_expected_rounds"]          # consensus rounds (quorum dynamics)
node_delay_seconds = expected_rounds * physical.single_hop_delay_s  # rounds x a CONSTANT per-hop time
...
"D_avalanche_rounds_mean": expected_rounds.mean()                   # <-- compute_coupled_loss reads THIS
```

`compute_coupled_loss` reads `D_avalanche_rounds_mean` (= `expected_rounds.mean()`). The
per-hop time is a **constant**, so D depends only on the avalanche quorum dynamics
(k/α/β/initial confidence), which sit at the protocol floor (1.32× β) and are nearly
edge-independent. **That is the entire reason w_D×10 moves D by ≤0.2%** — the loss is
optimizing a number the topology barely changes. Contrast E (`_energy_proxy`, line ~408),
which B-fix-2 made structural via `n_tx = 1/link_success` and which is now controllable.

**The fix is symmetric:** give the per-hop *time* the same `n_tx` factor (ARQ: a hop is
retransmitted until it succeeds), so D inherits a topology lever from `link_success`.

---

# THE STRUCTURAL DELAY MODEL

## D.1 D-fix-A — retransmission-aware structural delay (PRIMARY; built on link reliability + Q-function)

`link_success` is the finite-blocklength reliability `1 − Q((SINR_margin)/…)` already
computed by `_q_function` (line 163, `0.5·erfc(x/√2)`) from per-edge SINR — which depends
on the selected receiver's distance/load (topology). The model:

```
n_tx[e]              = 1 / clamp(link_success[e], min=floor)          # expected ARQ tx per hop  (>= 1)
per_edge_hop_delay[e]= single_hop_delay_s * n_tx[e]                   # seconds per hop on edge e
node_per_round_delay[v] = Σ_{e: src=v} w[e] * per_edge_hop_delay[e]   # w = normalized_query_weight (Σ=1 per node)
                       = single_hop_delay_s * weighted_mean_n_tx[v]   # seconds/round  (PARALLEL queries: NO ×k)
node_effective_rounds[v]   = expected_rounds[v] * weighted_mean_n_tx[v]            # rounds-like (>= expected_rounds)
node_structural_delay_s[v] = expected_rounds[v] * node_per_round_delay[v]          # seconds
D_avalanche_rounds_mean    = mean(node_effective_rounds)             # what the loss reads (rounds units), STRUCTURAL
```

- **Topology lever:** picking high-`link_success` receivers drives `n_tx → 1`, so
  `D_effective → expected_rounds` (minimised). A node stuck with low-SINR / high-load edges
  pays `n_tx > 1`. Edge selection now controls D — exactly the lever energy_heavy used.
- **Units preserved:** D stays in *rounds* (effective rounds), so `delay_target_rounds`
  still applies (likely recalibrate: D_eff ≈ rounds×n_tx ≈ 1.3–2.6× the current value; set a
  reachable target as `operating_point_v1` did for reliability, so the gate engages).
- **The k difference vs energy (physically important):** energy **sums** over the k queries
  (`_energy_proxy` multiplies by `avalanche.k`); delay is the **parallel** round latency, so
  it is the query-mass-weighted **mean** per-hop delay — **no ×k**. This is the one place
  the delay proxy must NOT copy the energy proxy verbatim.

### D.1.1 The honesty consequence — D ∝ E (correlated, not independent)

`node_energy = expected_rounds × per_attempt_energy × k × weighted_mean_n_tx` and
`node_structural_delay_s = expected_rounds × single_hop_delay_s × weighted_mean_n_tx`, so

```
node_structural_delay_s = node_energy × ( single_hop_delay_s / (per_attempt_energy × k) )   # a CONSTANT ratio
```

⇒ **D-fix-A makes D and E the same lever (perfectly correlated).** Re-test expectation:
**G2-G1 (D controllable) PASSES, but B-G3 (D⊥E independence) FAILS.** Honest claim becomes
*"coupled C/D/E, all controllable through link quality; D and E are correlated levers."*
Strictly better than today ("C/E, D inert"), but not three *independent* levers.

## D.2 D-fix-B — hop/path-length delay (OPTIONAL; restores D⊥E independence)

To make D move *independently* of E, add a delay term E does not share: a **differentiable
hop/path-length** component reflecting multi-hop propagation/queuing to reach quorum, which
depends on graph **structure** (path length / effective diameter) rather than per-link
energy. Sketch (deferred, scoped follow-up):

```
node_path_delay[v] = single_hop_delay_s * soft_expected_hops_to_quorum[v]
D_total[v] = node_structural_delay_s[v] (ARQ) + λ_path * node_path_delay[v]
```

`soft_expected_hops_to_quorum` is a differentiable estimate (e.g. a few iterations of a
soft message-passing reachability over the selected topology, O(Nk) and analytic — NO Monte
Carlo). This is more invasive (new structural estimator, new λ_path) and is **NOT** in the
primary scope; specify it only if the re-test shows D⊥E independence is required for the
contribution. **Recommend: ship D-fix-A first, measure, and only build D-fix-B if the
correlation is judged a real limitation.**

---

## Technical package (D-fix-A)

1. **New `DelayProxyConfig`** (`src/evaluation/v2x_consensus_bridge.py`, mirroring
   `EnergyProxyConfig`):
   ```python
   @dataclass(frozen=True)
   class DelayProxyConfig:
       single_hop_delay_s: float | None = None       # default -> physical.single_hop_delay_s
       structural_delay: bool = False                 # OPT-IN; default off = legacy expected_rounds (byte-identical)
       structural_success_floor: float = 1e-3
       @classmethod
       def from_mapping(cls, data): ...
   ```
2. **New `_delay_proxy(...)`** mirroring `_energy_proxy` (reuse `src_index`,
   `normalized_query_weight`, `node_expected_rounds`, `link_success`); returns
   `node_effective_rounds`, `node_structural_delay_seconds`, and the metric keys. **No ×k.**
3. **Wire it in** at line ~611 (next to the `_energy_proxy` call, same inputs available:
   `support.src_index`, `support.normalized_query_weight`, `channel["link_success"]`). When
   `structural_delay` is on, set `D_avalanche_rounds_mean` / `D_avalanche_rounds_p90` /
   `D_avalanche` from `node_effective_rounds`; **always** also expose
   `D_protocol_rounds_mean` (the legacy pure count) and `D_structural_rounds_mean` as
   diagnostics, so the RESULT can show both. Default off ⇒ legacy values unchanged.
4. **Plumb the config:** `evaluate_v2x_graph_consensus(..., delay_config=...)`;
   `src/training/training_smoke.py::_evaluator_delay_config` reads a YAML `delay:` block
   (mirroring `_evaluator_energy_config` / the `energy:` block).
5. **Config** `configs/operating_point_v1.yaml`: add
   ```yaml
   delay:
     structural_delay: true
     structural_success_floor: 0.001
   delay_target_rounds: <recalibrated, reachable>   # so the gate engages on D_effective
   ```
6. **Tests** `tests/evaluation/test_structural_delay.py`: default-off byte-identical (D ==
   expected_rounds.mean()); structural D ≥ legacy D (n_tx ≥ 1); worse link (lower SINR) →
   larger D; D differentiable to `topology_weight` (via link_success → SINR → in-load);
   finite at low success (floor clamp); **no ×k** (a 1-query vs k-query node has the same
   per-round delay scaling). Plus assert `node_structural_delay_s / node_energy` is constant
   (documents the D∝E correlation explicitly).

## Invariants preserved

- **No direct link-reliability loss.** `link_success` (the Q-function output) is used only
  as an **internal variable** to compute the evaluator's D metric; D reaches the loss solely
  via the existing `L_D` penalty in `coupled_objective.py`. The term lives in
  `src/evaluation/`, never `src/losses/` — identical to the B-fix-2 energy argument. Must
  stay green against `scripts/harness/verify_no_link_reliability_loss.py` (scans `losses/`).
- **Opt-in, default-off, behaviour-preserving.** `structural_delay=False` ⇒
  `D_avalanche_rounds_mean == expected_rounds.mean()` byte-identical; all existing tests/runs
  unaffected (assert in test).
- **Analytic, differentiable, sparse O(Nk).** `n_tx = 1/clamp(link_success)` + `index_add`;
  no Monte Carlo, no O(N²). Same complexity class as `_energy_proxy`.
- **train == val == deploy / PCGrad retained.** Forward topology unchanged; PCGrad
  (gradient plumbing) composes unchanged with the new evaluator metric.

---

## The re-test (the user's plan, keeping PCGrad)

Substrate `configs/operating_point_v1.yaml` + `delay.structural_delay: true`, **keep
`--governance pcgrad`**, N=2000, 200 steps, seeds 7/42/123:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'; $env:OMP_NUM_THREADS='1'   # OpenMP guard
python -B scripts/analysis/run_de_ablation.py --config configs/operating_point_v1.yaml `
  --node-count 2000 --max-steps 200 --seed 7 --reliability-target 0.02 `
  --governance pcgrad --run-name de_ablation_v4_structural_seed7
# repeat seeds 42, 123 as SEPARATE invocations (never wrap in `| Tee-Object` in a bg loop — it deadlocks)
```

### Decision gates

- **SD-G1 (D now controllable):** `delay_relative_reduction ≥ 0.02` (w_D×10 cuts D_effective
  ≥2% vs rel_only) — the bar G2 failed with the round-count D. *Expected PASS* (same lever as
  energy's +27…65%).
- **SD-G2 (reliability/energy not regressed):** `rel_only` still meets the (recalibrated)
  targets; the earned `energy_relative_reduction ≥ 0.02` survives.
- **SD-G3 (stability holds under PCGrad):** the `full` arm stays in band on all seeds incl.
  123 (re-confirm G1 — the loss landscape changed).
- **SD-G4 (independence, informational):** `energy_independent_of_delay`. *Expected FAIL for
  D-fix-A* (D ∝ E). A FAIL here is the trigger to consider D-fix-B, not a defect of D-fix-A.

### Decision tree

```
Re-test with structural_delay + pcgrad.
├─ SD-G1 ∧ SD-G2 ∧ SD-G3 pass → D is TOPOLOGY-CONTROLLABLE. Claim "coupled C/D/E,
│     link-quality-controllable" (D,E correlated). Promote structural_delay into the
│     operating point. Biggest result the project can reach without a new substrate.
│       ├─ SD-G4 also pass (unlikely for D-fix-A) → D⊥E independent: full "coupled C/D/E"
│       │     with three independent levers. Done.
│       └─ SD-G4 fail (expected) → D,E correlated. If independence is required, build
│             D-fix-B (hop/path delay); else report the correlated-levers claim honestly.
├─ SD-G1 fails (even structural D won't move) → the topology cannot pick better-SINR
│     receivers without breaking reliability at this operating point (link quality is
│     reliability-saturated). Then D's lever is genuinely absent here → revisit the
│     operating point (sparser degree / different density) or accept "coupled C/E" as final.
└─ SD-G3 fails (structural D destabilises training) → tighten PCGrad / recalibrate
      delay_target / add GradNorm; do not promote until stable.
```

---

## Risks & mitigations

| risk | mitigation |
|---|---|
| **D ∝ E correlation** (D-fix-A not independent) | Stated up front; SD-G4 measures it; D-fix-B (hop/path) is the scoped independence fix. The correlated-levers result is still a real upgrade. |
| `n_tx = 1/link_success` blows up as success→0 | `clamp(min=structural_success_floor)`; unit-test gradient finiteness at low success (same as energy). |
| Recalibrating `delay_target_rounds` looks like goalpost-moving | Set the reachable target *before* the re-test and record it (as `operating_point_v1` did for `reliability_target=0.02`); report D_protocol vs D_structural both. |
| Changing D's meaning breaks downstream readers of `D_avalanche_rounds_mean` | Keep the key name + rounds units; expose `D_protocol_rounds_mean` separately; default-off preserves legacy exactly. |
| Trips `verify_no_link_reliability_loss.py` (name match on link_success) | The term is in `evaluation/`, not `losses/`; same precedent as B-fix-2 energy, which is green. |
| Structural D changes the loss landscape → PCGrad stability no longer holds | SD-G3 re-confirms the `full` arm (esp. seed 123) under PCGrad before promotion. |

## Implementation checklist

1. `DelayProxyConfig` + `_delay_proxy` (no ×k) + wire-in at line ~611;
   `D_avalanche_rounds_mean` switched to `node_effective_rounds` when on; expose
   `D_protocol_rounds_mean` / `D_structural_rounds_mean` diagnostics.
2. `evaluate_v2x_graph_consensus(delay_config=...)` +
   `training_smoke._evaluator_delay_config` + `delay:` YAML block.
3. `tests/evaluation/test_structural_delay.py` (default-off byte-identical; n_tx≥1; worse
   link→larger D; differentiable; finite at floor; no ×k; D∝E constant-ratio assertion).
4. `configs/operating_point_v1.yaml`: `delay.structural_delay: true` + recalibrated
   `delay_target_rounds`.
5. Re-test: `run_de_ablation.py --governance pcgrad ... --run-name de_ablation_v4_structural_*`
   on seeds 7/42/123; `result/de_ablation_v4_structural/RESULT.md`; apply SD-G1..SD-G4.
6. Docs: stamp the verdict here; update `docs/GRADIENT_GOVERNANCE_DESIGN.md` (the off-ramp
   is now taken) and `docs/EVALUATOR_MODEL_AUDIT.md` (D is no longer floor-bound when
   structural).

## Cost & recommendation

- **D-fix-A:** 1 evaluator config + 1 proxy fn + wire-in + 1 test + 1 config edit + 1
   re-test (3 seeds, ~same cost as `de_ablation_v3`). Small; reuses the proven energy
   machinery and the existing Q-function link reliability verbatim.
- **D-fix-B:** deferred; only if SD-G4 independence is required.
- **Recommendation:** **implement D-fix-A and re-test with PCGrad.** It directly attacks the
   one open defect (D inert), uses exactly the link-reliability + Q-function basis requested,
   and on the energy precedent should make D controllable. Go in eyes-open that D will be
   *correlated* with E (not independent) — which is an honest, publishable "coupled C/D/E
   (link-quality-controllable)" and the natural stopping point unless independence is
   explicitly required (then D-fix-B).
```

---

## P1-2 follow-up: per-round MAX (slowest-query) latency aggregation

D-fix-A scaled per-hop time by `n_tx` and reduced it over a node's parallel query set with a
**query-mass-weighted MEAN** (`structural_delay_reduce="mean"`, the default, byte-identical). But the
round latency of *parallel* queries is set by the **slowest** query, not the average — the mean
under-states latency and weakens D's topology lever.

`structural_delay_reduce="max"` (opt-in) replaces the mean with a **differentiable soft-max**
(query-mass-weighted `logsumexp` over the per-edge `n_tx` hop delays; the mean is its `T→∞` limit, so
`D_max ≥ D_mean` always). The slowest query then controls the round latency, so the planner can lower D
by improving its *worst* link, not just the average. Gradients still flow through SINR → in-load →
`topology_weight`; a detached global shift keeps `exp` stable. Diagnostic: `D_structural_reduce_mode`.

**Validation** (`run_de_ablation.py --config operating_point_v1 --delay-reduce {mean,max}`, N=250,
40 steps, quenched eval Q=21): the `w_D×10` `delay_heavy` arm cuts D by **+81.8%** vs `rel_only` under
`max` (3262→592 rounds) — well past the ≥5% acceptance bar — and `max` magnifies the absolute latency
(rel_only D 3262 vs the mean's 1219), reflecting the worst-case parallel-query semantics. So D under
`max` is both a faithful latency model and strongly topology-controllable. Default stays `mean`
(byte-identical); contract tests in `tests/evaluation/test_structural_delay.py`.
