# CDQ 2.0 — Results & Contribution (macrostate / effective-sampling-redesign round)

**Framing (Plan A, user-decided 2026-06-27).** The contribution of this round is a **diversity-aware
consensus-query family (CDQ 2.0) that provably contains ESP, is mechanism-identifiable, exactly and
differentiably computable, and scale-generalising — together with a differentiable
pairwise-correlation training objective.** This is a *mechanism / methodology* contribution. Its
macrostate-basin **reliability** payoff in the binary-snowball consensus is **honestly scoped**: a
small, covariance-scoped, deadline-channel `P_correct` gain (not a wrong-/split-basin reduction). All
claims below are backed by the evidence manifest `docs/gate_evidence/macrostate/manifest.json` and
the reproducible scripts/JSON in that directory; all eleven gates of
`docs/MACROSTATE_CDQ2_ENGINEERING_PLAN.md` are green.

---

## 1. The headline contribution

CDQ 2.0 is the query kernel

```
L = D^{1/2} (I + η Z Z^T) D^{1/2},   D = diag(a_j = exp s_j) > 0,   Z = unit-normalised rows,   η ≥ 0,
```

a **full-rank diagonal quality `D` plus a rank-`r` diversity correction**. Selecting a distinct-peer
`k`-subset from the `k`-DPP `P(S) ∝ det(L_S)` gives a query law that:

1. **Exactly contains ESP.** `η = 0 ⇒ L = D ⇒ P_CDQ ≡ P_ESP`, for **any** embedding `Z` and **any**
   rank `r`. (The prior low-rank `L = BB^T` CDQ only recovered ESP at full rank `r = d`.) The model
   initialises at `η ≈ 0` and *learns* diversity only if the environment rewards it.
2. **Separates quality from diversity:** `det(L_S) = (∏_{j∈S} a_j)·det(I + η Z_S Z_S^T)`; for `k = 2`,
   `det = (1+η)² − η²(z_j·z_l)²` — more-similar peers get a smaller joint-selection weight, the lever
   ESP's product law lacks.

This is the first member of the consensus-query literature (in this codebase's lineage) that is *both*
diversity-aware *and* a strict, exact generalisation of the quality-only product policy.

---

## 2. Exactness (the math is not an approximation)

Validated against brute-force subset/assignment enumeration at `< 1e-10` (S12 / S12c / S12d):

| object | result |
|---|---|
| normaliser `e_k(λ(L))` | exact `< 1e-10` (incl. wide quality dynamic range, near-collinear rows, large η) |
| subset probability / inclusion `π_j` (`Σπ_j = k`) | exact `< 1e-10` |
| determinantal quorum `P(m,n)` (correct/wrong/no-response) | exact `< 1e-10` (two independent references) |
| exact `k`-DPP sampler (for the MC judge) | MC → exact law, `< 0.0015` at N=200k draws |
| η = 0 degeneracy of all of the above to ESP | machine-precision |

A central technical lesson, surfaced by adversarial audit and recorded: the spec's
matrix-determinant-lemma `z`-series normaliser is exact in real arithmetic but **catastrophically
cancellation-prone in float64** at realistic GNN logit ranges (the alternating `(I+zD_c)^{-1}` series).
We replaced it with an **eigenvalue route** — `e_k` of `eigvalsh(D_c^{1/2}(I+ηZZ^T)D_c^{1/2})` via a
cancellation-free positive recursion — which is *unconditionally stable* and *smoothly differentiable
everywhere* (incl. η = 0 and repeated eigenvalues, since `e_k` is a symmetric function), with no
eigen-gap denominators.

---

## 3. Differentiability + the correlation-objective methodology (G-CDQ2-GRADIENT)

The diversity head is trainable end-to-end. Autograd gradients of `log e_k`, the inclusion, and the
quorum match central finite differences at `< 1e-4` w.r.t. `a`, `Z`, and `η` (including at exactly
`η = 0`, the identifiability anchor). The training objective is the differentiable expected
within-poll correlation

```
L_corr = Σ_i Σ_{j<l} π^(2)_{i,jl} R_{jl}        (spec §11)
```

with `R` the deployment-observable evidence-error correlation and `π^(2)_{jl} = P(j,l ∈ S)` the
**pairwise inclusion**, computed *first-order-differentiably* via the **Schur-complement quotient
identity** `Σ_{S⊇T} det(L_S) = det(L_T)·e_{k−|T|}(λ(L/L_T))` — deliberately *not* a second-order
autograd of `e_k` (which the training back-prop would turn into a fragile third-order op). No
`float()/.item()/detached` score on the training path. The objective genuinely **discriminates
covariance under matched marginals** (identical `q_i`, different `R` ⇒ different gradient to the
diversity params) — the C3 identifiability evidence.

---

## 4. Mechanism identifiability (all five contract items)

The CDQ 2.0 components pass the full Mechanism Identifiability Contract:

- **C1 structure existence / C2 observability** (overlapping common-cause environment, S11): a
  crosscutting sensor/map/road common-cause model where same-road peers are *non-exchangeable*, with
  a **matched-marginal control** (same `q_i`, different covariance) and *deployment-observable* group
  labels (no truth leakage).
- **C3 gradient-reachable** (S13): the correlation gradient reaches and discriminates the diversity
  params (above).
- **C4 causal benefit** (S15): the matched-marginal MC factorial — ESP cannot distinguish the
  identical-marginal high/low-covariance arms; CDQ 2.0 can.
- **C5 mainline consistency** (S1/S8): deterministic config hashes + train==eval enforcement; `η = 0`
  reproduces the ESP canonical episode bit-for-bit.

---

## 5. Scale generalisation (G-SCALE-GENERALIZATION)

The canonical episode + dynamic MC + CDQ 2.0 are **near-linear in N, E with no N×N**, validated at
`N = 120 … 9840` (S16): log-log runtime slope **0.88 (ESP) / 0.81 (CDQ 2.0)** (≈ linear at the large
end), **max out-degree constant** (bounded by the comm radius), total padded cells `≤ 2E`. CDQ 2.0 is
a ~2× *constant* factor with the *same* asymptotic slope. **No N×N is structural, not merely empirical**
— a grep over `src/` finds no `(N,N)` / `num_nodes×num_nodes` / `eye(N)` allocation; the path is sparse
radius graphs + the bucketed `[m,w]` layout + per-source `[d,d]` kernels. Scaling N is the registered
`node_count` OOD axis, with a verified allow/block enforcement matrix (registered → allow; unregistered
/ protocol-mismatch → block; ideal/full-link mismatch → *always* block).

---

## 6. Temporal extension (G-TEMPORAL)

A **causal, differentiable, observable-driven** memory `m_t = (1−ρ) m_{t-1} + ρ x_t` over a
persistence-controlled, matched-marginal-in-time correlated-evidence sequence (S17). It passes the
temporal Mechanism Contract C1–C5 — causality (zero gradient w.r.t. the future, no leak),
differentiability, matched-marginal-in-time, `ρ = 1` / `η = 0` ⇒ static mainline — and **tracks the
persistent correlated structure** (estimate-quality high under persistence, ≥ 0.2 lower under
iid-in-time; scoped, matched-marginal-controlled).

---

## 7. The honest boundary: macrostate-basin reliability payoff

Judged by the **independent dynamic-MC macrostate basin first-hitting** (the headline judge; *not* the
legacy node-union `1−∏c_i`), the *reliability* payoff of CDQ 2.0's diversity is **scoped and marginal**:

- **Matched-marginal factorial (S15).** A genuine, **CI-separated, covariance-scoped `P_correct`
  benefit**: `matched_marginal_high` CDQ 2.0 `P_correct ∈ [0.634, 0.648]` vs ESP `[0.618, 0.632]`
  (CIs disjoint; holds under full physics), **neutral in iid** (no covariance ⇒ no lever). *But* the
  gain flows through **faster quorum (`F_deadline` ↓ ~0.025)**, **not** correlation-avoidance:
  **`F_wrong` is not reduced** (point estimate up ~0.009–0.014).
- **Temporal (S17).** At the basin headline the memory-driven diversity is a **null** vs ESP (memory ≈
  no-memory ≈ oracle ≈ ESP, all marginally below); the memory's value is at the *estimate / structure-
  tracking* level, not basin reliability.

**Why (fundamental, not a tuning artefact).** In a **majority-correct** regime, polling *diversely*
raises a node's exposure to the *minority* correlated-wrong clusters, so it does not lower the
wrong-consensus basin; it only rescues the few nodes already inside wrong clusters. Diversity buys
*speed* (reaching quorum), not *wrong-avoidance*. The sensor-only oracle embedding — the upper bound on
any embedding trained on `L_corr` — does not lower `F_wrong` either, so the correlation surrogate is a
faithful proxy for `P_correct` but **not** for the `F_wrong` reliability basin.

**Stance.** This boundary is reported, never hidden; no threshold was lowered and no test changed to
force a pass. The contribution stands on §§1–6 (exact, identifiable, differentiable, scalable
diversity query family + the correlation-objective methodology); the basin-reliability effect is
characterised honestly as a small, covariance-scoped, deadline-channel `P_correct` gain.

---

## 8. Reproducibility

- Evidence manifest: `docs/gate_evidence/macrostate/manifest.json` (per-slice S1–S17 + gate verdicts).
- Decision log: `docs/MACROSTATE_PROGRESS.md` (M0–M19, checkpoints #1–#12).
- Reproducible runs: `run_cdq2_factorial.py` / `run_scale_generalization.py` / `run_temporal_factorial.py`
  → `*_results.json` (all under `docs/gate_evidence/macrostate/`).
- Core modules: `src/sampling/{cdq2_kernel,cdq2_quorum,cdq2_correlation,cdq2_wiring}.py`,
  `src/environment/{overlapping_evidence,temporal_sequence}.py`,
  `src/metrics/{macrostate,basins,first_hitting,basin_surrogate,participation,temporal_memory}.py`,
  `src/config/{service_profile,experiment_spec}.py`, `src/evaluation/cdq2_factorial.py`.
- The legacy node-union `F` / global-risk paper headline is **not** modified by this round (it remains
  legacy); this document is the round's contribution statement.
