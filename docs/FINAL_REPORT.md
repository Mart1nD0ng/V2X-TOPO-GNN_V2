# Final Report — V2X-TOPO-GNN Mathematical Refactor (all 11 gates green)

> Generated 2026-06-22 at the completion of the gate-driven refactor (spec §8).
> Single source of truth: [`GLOBAL_CONSENSUS_MATH_REDESIGN.md`](GLOBAL_CONSENSUS_MATH_REDESIGN.md).
> Progress / decision log: [`REFACTOR_PROGRESS.md`](REFACTOR_PROGRESS.md).
> **This is an engineering report. It does NOT modify the paper.** Recommended paper changes are a
> separate proposal: [`PAPER_UPDATE_PROPOSAL.md`](PAPER_UPDATE_PROPOSAL.md).

## 1. Status: 11 / 11 acceptance gates green

| Gate | Constraint / module | Certifies |
|------|---------------------|-----------|
| G1 | H1 / §3.3 | `F_global` is a strict global event probability (shared finite-mixture log-domain product-mixture), not a per-node marginal |
| G2 | §3.1 | exact weighted distinct-peer k-subset query policy via elementary symmetric polynomials |
| G3 | §3.2 | exact heterogeneous quorum DP (three-way generating function), **no iid beta-tail** |
| G4 | H2 / §3.4 | physics-constrained adaptive topology — sparsity from cost, **no degree cap / top-k** |
| G5 | H3 / §3.5 | rigorous finite-blocklength `ℓ(γ,n,B)` with explicit dispersion `V(γ)`, 3GPP-grounded |
| G6 | §3.6 | independent global delay `D` and network energy `E` + power / blocklength heads |
| G7 | §3.7 | preference-conditioned GNN — one checkpoint sweeps the F/D/E Pareto front |
| G8 | §3.8 | global-risk emission `r_ir=-log c_ir` + stop-gradient; bounded-scalar claim **falsified** by ablation |
| G9 | H4 | end-to-end near-linear complexity (deterministic no-`N×N`-tensor guard) |
| G10 | H5 | single live mathematical mainline; legacy quarantined to `legacy/` |
| G11 | ultimate | baseline-comparison win on held-out scenarios with significance |

Each gate was hardened against an independent multi-lens **adversarial verification** (Workflow).
Across the rounds the verifications almost always found *gate-discriminativeness / reporting-honesty*
gaps rather than final implementation-correctness defects; every confirmed finding was fixed.

## 2. The unified mathematical mainline (`src/mainline/`)

```
(a_θ, P_θ, n_θ)  ->  π (G2, elementary symmetric polynomials)
                 ->  Λ receiver load (G4)  ->  γ (geometry + interference)  ->  ℓ (G5, FBL V(γ))
                 ->  (h+,h-,h0) three-way quorum DP (G3)
                 ->  graph-coupled Snowball recurrence  ->  c_ir(t)
                 ->  shared finite-mixture  F_global = 1 - Σ_r ω_r ∏_i c_ir   (G1, H1)
                 ->  D global order statistic, E network energy (G6)
                 ->  augmented-Chebyshev preference scalarisation (G7)
                 ->  global-risk emission e_i = clip(sg[Σ_r ρ_r r_ir]/r_max,0,1)  (G8)
```

Modules: `symmetric_polynomials.py` (G2), `quorum_dp.py` (G3), `snowball.py` + `global_evaluator.py`
(G1), `topology.py` (G4), `finite_blocklength.py` (G5), `objectives.py` (G6), `model.py` (G7),
`emission.py` (G8).  The whole live path imports **only** `src.mainline`; the superseded
mean-field / beta-tail / logistic-BLER / degree-cap derivations are frozen under `legacy/`
([`legacy/ARCHIVED.md`](../legacy/ARCHIVED.md)).

## 3. One-button reproduction

```bash
python scripts/gates/run_all_gates.py          # all 11 gates -> docs/gate_evidence/latest.json
python scripts/gates/run_all_gates.py G3 G9    # a subset
python -m pytest tests/ -q                     # the full unit-test suite
```

Per-area drivers (reproducible, seeded): `scripts/analysis/profile_scaling.py` (G9 figure),
`scripts/analysis/baseline_comparison.py` (G11 study).  All numbers in the gates come from these
reproducible scripts — none are hard-coded (anti-cheat rule §7).

## 4. H4 complexity evidence (G9)

End-to-end forward over `N = 200..6400` at fixed spatial density (area ∝ N ⇒ `E = O(N)`):
`E~N` exponent 1.06; per-stage `t~E` exponents build 1.05 / GNN 0.83 / consensus 0.55 / total 0.56
(no stage super-linear); largest materialised tensor is the `O(E·k³)` quorum cube — its scaling
exponent vs E is **1.000**, and a deterministic `TorchDispatch` guard fails any genuine `N×N`
(verified: injected N×N → exponent 1.92, cubic → 1.50, both FAIL the gate).  Figure:
[`gate_evidence/g9_scaling.png`](gate_evidence/g9_scaling.png).

## 5. G11 baseline comparison (the headline result)

A single preference-conditioned checkpoint, trained on TRAINING scenarios and evaluated on
**held-out** scenarios, against honest strong baselines (a 392-point exhaustive constant-policy
grid over 4 query heuristics + random restarts; a preference-blind ablation; an untrained
control).  All methods are scored through the **same** physics pipeline (`evaluate_controls`) —
they differ only in how the controls `(s, P, n)` are produced.

Primary metrics (12 held-out scenarios, paired Wilcoxon):

| metric | model vs strongest baseline (`best-fixed`) |
|--------|--------------------------------------------|
| **Pareto set-coverage** (normalisation-free) | C(model→grid) = **0.392** vs C(grid→model) = **0.000**, p = 0.0002 |
| **Hypervolume** (model-independent box) | 0.606 vs 0.509 = **1.19×**, win-rate **100%**, p = 0.0002 |
| vs preference-blind ablation | model wins HV 100%, p = 0.0002 (preference conditioning adds value) |
| vs untrained control | dominated, C(model→untrained) = 0.99 (gate is discriminative) |

**No baseline strictly Pareto-dominates any model point** (C(baseline→model) < 0.02; 0.000 at the
certified 600-step budget).  The win is seed-robust (verified across disjoint seed ranges) and
survives an aggressively strengthened 968-point baseline.

Reproduce: `python scripts/analysis/baseline_comparison.py` (full 16/16 × 900-step study).
Evidence: [`gate_evidence/g11_baseline.json`](gate_evidence/g11_baseline.json).

## 6. Honest limitations (reported, not hidden)

1. **Chebyshev extremes.** The exhaustive grid achieves better *per-preference single-point*
   (Chebyshev) values — including some balanced preferences — by brute-forcing power/blocklength
   to the limits.  Those points do **not** Pareto-dominate the model (they are worse on other
   objectives).  The model's win is on overall front quality + generalisation + efficiency (one
   checkpoint, no per-scenario search), not extreme corners.
2. **Dense-deployment regime.** G11 uses complete candidate graphs (all vehicles in radio range);
   the optimisation lever is the *polling* topology (which k peers to query, via the G2 k-subset
   distribution) + per-node power/blocklength.  At `SCALE ≫ RADIUS` the candidate graph is sparse
   and some nodes fall below quorum size `k` (needing the §7.2 candidate-shortage protocol); a
   600-step model does not beat exhaustive search in that harder regime — out of scope here.
3. **`F` is U-shaped in link reliability** (over-reliable links propagate wrong votes); recorded
   as a spec-§9.4 discrepancy for human review (does not affect any gate).
4. **`paper/main.tex` still narrates the superseded derivation** (betainc beta-tail quorum +
   node-mean `F`); reconciling it is the post-gate proposal [`PAPER_UPDATE_PROPOSAL.md`](PAPER_UPDATE_PROPOSAL.md).
5. **`src/mainline/` is not yet committed** (working tree); recommend committing so the gates run
   against an immutable tree.

## 7. Deliverables checklist (spec §8)

- [x] Unified mathematical-mainline code — `src/mainline/`
- [x] Full unit-test suite + one-button gate runner — `tests/`, `scripts/gates/run_all_gates.py`
- [x] Profiling report (H4 near-linear evidence) — G9 + `gate_evidence/g9_scaling.png`
- [x] Baseline comparison with reproduction command + significance — G11 + `baseline_comparison.py`
- [x] `REFACTOR_PROGRESS.md` final state (D1–D12, gate table all 🟢)
- [x] Paper-update **proposal** (not a direct edit) — `PAPER_UPDATE_PROPOSAL.md`
