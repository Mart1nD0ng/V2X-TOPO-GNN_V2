# Optimization Contract — Reliability-Constrained Tail-Latency / Energy

**Status:** Phase 0 decision record (engineering plan §3).
**Design basis:** `EFFECTIVE_SAMPLING_DYNAMICS_TECHNICAL_SPEC.md` §4; `ESD_GNN_ENGINEERING_PLAN.md` §3.
**Supersedes:** the legacy `λ_F, λ_D, λ_E` equal-weight preference-conditioned Pareto,
which permitted trading error risk for speed and is forbidden going forward (spec §4.4).

---

## 1. Problem (spec §4.4)

Optimize **only the query topology** `π_θ` (which `k` peers each node polls). Transmit
power, MCS/blocklength, HARQ profile, and sub-channel pool are **fixed** in the headline
(spec §7.5) so that any performance change is attributable to topology, not resources.
Per-node power/blocklength heads are an *extension*, evaluated separately (spec §2.2).

```
min_θ  ( CVaR_q(T_all),  E_network )
s.t.   F_disagree(θ) ≤ ε_s
       F_wrong(θ)    ≤ ε_v
       F_deadline(T_d; θ) ≤ ε_d
```

Reliability is a **hard constraint**, not a Pareto axis. Only points inside the feasible
region are compared on `(D, E)` (spec §4.4; plan §17 prohibition #9).

---

## 2. Default thresholds (recommended — overridable)

Adopted from spec §4.4 recommended values. **Flagged for user override**; changing them
edits only this table and the `service_profile` config, never the math/code contracts.

| symbol | meaning | default |
|--------|---------|---------|
| `q`    | tail quantile for CVaR of `T_all` | `0.95` (also report `0.99`) |
| `ε_s`  | agreement-safety budget `F_disagree` | `1e-4` (`ε_s ≪ ε_v`) |
| `ε_v`  | validity budget `F_wrong`            | `1e-3` |
| `ε_d`  | deadline-miss budget `F_deadline`    | `1e-2` |
| `T_d`  | deadline (wall-clock)                | set per service profile (§4) |
| `R_max`| protocol horizon (rounds)            | scanned in Phase 5 feasibility |

Perfect-link feasibility floors (spec §3.3) must satisfy
`F_·^floor ≤ ε_·/10` before any model training (plan Phase 5 / G8 stop condition).

---

## 3. Primal–dual training objective (spec §4.5)

```
L_θ = CVaR_q(T_all) + λ_E E
      + μ_s (F_disagree − ε_s) + μ_v (F_wrong − ε_v) + μ_d (F_deadline − ε_d)
μ_r ← [ μ_r + η_μ (F_r − ε_r) ]_+
```

Fixed hand-tuned weights may **not** replace the dual ascent on `μ`. The auxiliary
effective-sampling losses (spec §5.8: progress / drift / ESS / mixing / load) are
*mechanism aids only*; the headline is judged on real safety, deadline, latency, energy.

---

## 4. Service profiles

A *service profile* fixes `(T_d, fixed PHY resources, R_max search range, ε_·)`. The
headline uses one default urban-V2X profile; the contract supports several so that
feasibility (Phase 5) can report which `N` / profile combinations are admissible.

| profile | `T_d` | PHY (fixed) | notes |
|---------|-------|-------------|-------|
| `urban_default` | TBD in Phase 5 from perfect-link floors | tx 23 dBm, PSSCH 600 cu / PSCCH 60 cu, 2-attempt HARQ, 5 sub-channels | headline |

`T_d` is calibrated in Phase 5 (protocol feasibility) as the smallest deadline meeting
`F_deadline^floor ≤ ε_d/10` under perfect links at the target `N`; it is then frozen for
all baselines/ablations/figures (single canonical evaluator, spec §2.3 / G0).

---

## 5. Comparison rules (spec §11, G11)

* Compare **only feasible** policies (all three constraints met) — dominated/ infeasible
  points are filtered before any latency/energy claim.
* Report `D_50, D_95, D_99, CVaR_0.99(T_all)`, `E_network`, feasibility rate.
* Significance: paired common-random-number dynamic-MC, ≥5 model seeds, ≥30 scene seeds,
  multiple-comparison correction (plan §13).
* The dynamic MC (spec §8) — not the analytic surrogate — produces the published numbers.
