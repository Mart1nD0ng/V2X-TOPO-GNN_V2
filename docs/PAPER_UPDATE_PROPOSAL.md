# Paper Update Proposal (PROPOSAL ONLY — does not modify `paper/main.tex`)

> Drafted 2026-06-22, after all 11 acceptance gates went green (spec §8 permits a paper-update
> **proposal** only once the gates pass; the spec forbids editing paper headline/claim text
> before that, §0/§5).  **Nothing in `paper/` has been changed.**  This document recommends the
> changes a human author should make to reconcile the manuscript with the unified mainline
> (`src/mainline/`).  Single source of truth: [`GLOBAL_CONSENSUS_MATH_REDESIGN.md`](GLOBAL_CONSENSUS_MATH_REDESIGN.md).

## A. Conflicting derivations in `paper/main.tex` to REPLACE (H5)

The G10 adversarial audit found `paper/main.tex` still narrates the **superseded** derivation,
which conflicts with the live mainline:

1. **Quorum (≈ lines 605–626, refs at 238/518/557–558/682/693).** The paper uses
   `H(x;k,α)=I_x(α,k−α+1)` "evaluated by `betainc`", self-described as an *iid-with-replacement
   beta-tail / mean-field surrogate* for the Poisson-binomial tail.
   → **Replace** with the exact heterogeneous **distinct-peer generating-function quorum DP**
   (`src/mainline/quorum_dp.py`, spec Eqs. 25–30):
   `Ψ_i(z,x,y)=∏_j[1+z a_ij(p0 + p+ x + p− y)]`, `P_i(m,n)=[z^k x^m y^n]Ψ_i / e_k(a_i)`,
   `h+/h−/h0` from Eqs. 27–29.  Remove all `betainc` / `I_x(α,k−α+1)` / "iid with replacement" /
   "mean-field surrogate" language, and the dormant commented `scipy.special.betainc` citation
   (≈ lines 1205–1206).

2. **Consensus `F` (≈ lines 574–595, 644–656, 668).** The paper uses a row-normalised support
   bridge `q_ij = w_ij/Σ w_ij` feeding `F = \overline{F_i}` (per-node **mean**), explicitly a
   "mean-field closure … treats neighbour states as independent".
   → **Replace** with the **shared finite-mixture global event probability**
   (`src/mainline/global_evaluator.py`, spec Eqs. 5–13):
   `F_global = 1 − Σ_r ω_r ∏_{i∈H} c_ir = −expm1(logsumexp_r(log ω_r + Σ_i log c_ir))`,
   loss `= −log S_C`.  Delete the `F = \overline{F_i}` node-mean definition; state clearly that
   `F_global` is a strict global event probability `P(∃ i: Y_i ≠ C)`, not a node average.

## B. Forbidden old results to REMOVE / RECOMPUTE (spec §12)

Do **not** carry over: the degree-4 protocol floor; the old `F = 0.0635` node-mean result; the
mean-field / quenched / MC "currency" narrative; the hard top-k / spanning-tree constructor; the
old Pareto table; the per-node-confidence emission; the fixed-degree baseline ranking.  All
quantitative claims must be recomputed from the mainline.

## C. Recommended new headline / results (from the gated mainline)

- **Method.** Preference-conditioned FiLM topology GNN producing a weighted distinct-peer query
  distribution (exact ESP, no top-k) + per-node power/blocklength heads; exact heterogeneous
  quorum DP; shared finite-mixture global `F`; rigorous finite-blocklength `ℓ(γ,n,B)` with
  explicit dispersion `V(γ)`; independent global delay `D` and network energy `E`; global-risk
  emission `r_ir = −log c_ir`.  One checkpoint sweeps the F/D/E Pareto front.
- **Complexity (H4).** End-to-end near-linear in `N, E` (G9); report the `g9_scaling.png` figure
  and the deterministic no-`N×N`-tensor guard.
- **Headline result (G11).** On held-out dense-deployment scenarios, a single checkpoint's
  Pareto front **dominates** the strongest non-learned baseline (an exhaustive 392-point
  constant-policy grid): normalisation-free Pareto set-coverage C(model→grid)=0.39 vs
  C(grid→model)=0.00 (p=2×10⁻⁴); hypervolume 1.19× under a model-independent box, 100% of
  held-out scenarios; and it beats a preference-blind ablation (preference conditioning adds
  value) and an untrained control.  Report exact reproduction command + seeds + Wilcoxon p.

## D. Honest limitations the paper MUST state (anti-degradation §7)

- The exhaustive grid wins the *per-preference single-point Chebyshev* metric (better
  single-objective extremes, incl. some balanced preferences); those points do **not**
  Pareto-dominate the model.  Frame the contribution as front quality (coverage + hypervolume)
  + generalisation + efficiency (one checkpoint, no per-scenario search), **not** corner optimality.
- G11 is on **dense-deployment** complete candidate graphs; the sparse-graph regime
  (`SCALE ≫ RADIUS`, sub-quorum nodes needing the §7.2 protocol) is out of scope and should be
  named as future work.
- `F` is **U-shaped** in link reliability (over-reliable links propagate wrong votes) — this is a
  genuine emergent property, not a bug; reconcile the §9.4 "↑P ⇒ ↓F" claim accordingly.

## E. Process note

Apply these as tracked manuscript edits with the recomputed numbers pulled from
`docs/gate_evidence/*.json` and the reproduction scripts; do not transcribe any number by hand.
The legacy derivation may be retained only as an explicitly-labelled historical/contrast appendix
(as `legacy/ARCHIVED.md` does for the code), never as the headline theorem.
