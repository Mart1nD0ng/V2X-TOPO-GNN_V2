# ARCHIVED — historical reproduction material only (NOT the mainline)

> Quarantined on 2026-06-22 for **G10 (H5: single mathematical mainline)**, decision **D11**.
> Single source of truth for the live derivation: [`docs/GLOBAL_CONSENSUS_MATH_REDESIGN.md`](../docs/GLOBAL_CONSENSUS_MATH_REDESIGN.md).
> Progress / decision log: [`docs/REFACTOR_PROGRESS.md`](../docs/REFACTOR_PROGRESS.md).

Everything under `legacy/` is **frozen historical material**.  It encodes the *superseded*
derivations the refactor replaced and **must not** be used as a source of training results or
paper numbers (spec §8, §12).  The live mathematical mainline is **`src/mainline/`** only.

## Why these were quarantined (the forbidden / conflicting closures, H1–H5)

| Archived package | Forbidden closure it encodes | Replaced by (live mainline) |
|------------------|------------------------------|-----------------------------|
| `legacy/src/consensus/` | mean-field per-node-marginal `F` (`topology_query_support`), iid **beta-tail** quorum (`avalanche_closed_form`: regularized incomplete beta / beta-binomial) | `src/mainline/global_evaluator.py` (shared finite-mixture global `F`, H1) + `src/mainline/quorum_dp.py` (exact three-way generating-function DP, no beta-tail) |
| `legacy/src/evaluation/` | logistic **BLER-vs-SINR sigmoid** link reliability `ℓ` (`v2x_consensus_bridge`) | `src/mainline/finite_blocklength.py` (rigorous finite-blocklength `ℓ(γ,n,B)` with explicit dispersion `V(γ)`, H3) |
| `legacy/src/v2x_env/` | logistic `link_success`, **fixed degree caps / top-k** candidate truncation | `src/mainline/topology.py` (physics-constrained radius graph, sparsity from cost, no cap, H2) |
| `legacy/src/topology/` | fixed-degree / top-k spanning-tree constructor | `src/mainline/topology.py` (`build_candidate_graph`, no cap) |
| `legacy/src/{training,losses,models}/` | training/loss/model code built on the above closures (mean-field, hard-degree) | `src/mainline/model.py` + `objectives.py` + `emission.py` |
| `legacy/scripts/analysis/` (37 scripts), `legacy/scripts/harness/`, `legacy/scripts/calibrate_environment.py` | old experiment / verification tooling that imported the legacy closures and produced the discarded results (degree-4 floor, `F=0.0635` node-mean, old Pareto, etc., §12) | `scripts/gates/` (acceptance gates G1–G11) + `scripts/analysis/profile_scaling.py` |

## Invariants the G10 gate enforces (so this stays archival, not a parallel mainline)

- The live tree (`src/mainline/`, `scripts/gates/`, `scripts/analysis/profile_scaling.py`,
  `tests/`, `conftest.py`) imports **nothing** from `legacy/` (verified: import closure has
  zero `legacy.*` / `src.{consensus,evaluation,…}` references).
- `src/mainline/` contains **no** forbidden closure (no beta-tail `betainc`/`betabinomial`,
  no mean-field node-marginal `F`, no logistic-BLER `sigmoid((sinr-…))`, no degree cap / `topk`).
- `legacy/` is **not an importable package** on the live path (it carries no role in the
  mainline; these files are kept only for historical reproduction).

To reproduce a legacy result you must opt in explicitly (add `legacy/` to `sys.path`); it is
deliberately not wired into the live mainline.
