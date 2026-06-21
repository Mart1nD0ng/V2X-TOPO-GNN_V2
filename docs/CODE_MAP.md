# Code Map — Active Path vs Diagnostic Archive

The repo carries ~90k lines across ~140 `src` modules and ~120 analysis scripts.
The vast majority are one-off diagnostic milestones (M34–M103, R1–R12) that are
now **superseded by production training**. This map separates the small **active
path** that actually runs production training from the **diagnostic archive** so
future work is not lost in the accumulated entropy.

## Active path (the ~14 files that matter)

| concern | file |
|---|---|
| GNN edge scorer (P2: learnable gain + standardization) | `src/models/hierarchical_gnn.py` |
| Topology constructor (P0: straight-through gradient) | `src/topology/construction.py`, `src/topology/budget_adapter.py` |
| Coupled C/D/E loss (P1: scale-invariant backward) | `src/losses/coupled_objective.py` |
| Avalanche reliability (analytic, differentiable) | `src/consensus/graph_coupled_avalanche.py`, `src/consensus/avalanche_closed_form.py`, `src/consensus/topology_query_support.py` |
| Evaluator bridge | `src/evaluation/v2x_consensus_bridge.py`, `src/evaluation/v2x_bridge_inputs.py` |
| Environment / candidate graph / production density profiles | `src/v2x_env/` |
| Training loop (optimizer, ST/scale-invariant/scorer wiring, density profile) | `src/training/training_smoke.py` |
| Production config | `configs/production_training_v1.yaml` |
| Production runner + verdict | `scripts/analysis/run_production_training.py` |
| Standard density/time generalization | `scripts/analysis/run_standard_generalization.py` |
| Intermediate F/C operating-point scan | `scripts/analysis/run_intermediate_reliability_band_probe.py` |
| Result visualization | `scripts/analysis/visualize_production_training.py` |

Active tests: `tests/topology/test_straight_through_gradient.py`,
`tests/loss/test_scale_invariant_backward.py`,
`tests/models/test_scorer_dynamic_range_fix.py`,
`tests/training/test_intermediate_reliability_band_probe.py`,
`tests/training/test_de_ablation_bounds.py`,
`tests/training/test_production_training_verdict.py`, plus the core
`tests/{models,topology,loss,training,consensus,evaluation,integration}` suites.

Default gate: `make agent-check` → `harness-check` (invariants + core + remediation
+ production smoke). The remediation rationale is in `docs/REMEDIATION.md`.

## Diagnostic archive (do not extend; reference only)

Everything else under `src/training/*.py` (≈80 modules: `*_diagnostic.py`,
`*_probe.py`, `*_smoke.py`, `*_audit.py`, `*_design*.py`, `*_report.py`, the
`pure_edge_*`, `temperature_*`, `score_*`, `teacher_*`, `support_smoothing_*`
families) and the matching `scripts/analysis/*.py` are the historical milestone
chain. They are preserved and runnable via `make diagnostics-archive-check`, but:

- They are **report-only** — none changes the default training path.
- They localized the convergence root cause (constructor Jacobian, scale law,
  scorer dynamic range) but never shipped a fix; that is what P0/P1/P2 did.
- New work should go through the active path above, not by adding another probe.

## Entropy hygiene

- `make agent-check` no longer chains the ~120 milestone scripts.
- `python -m pyflakes src/` is kept near-clean (~90 unused imports removed). The
  few remaining `imported but unused` notes are **intentional re-exports**
  (marked `import X as X  # noqa: F401`, e.g. `load_default_config` /
  `load_training_v0_config` consumed by `scripts/analysis/run_*.py` CLIs).
- CAUTION: do NOT run `autoflake --remove-all-unused-imports` blindly — pyflakes
  3.x does not recognise the `import X as X` re-export convention, so autoflake
  will strip those re-exports and break the `run_*.py` CLIs and their tests.
  Verify with the import-check and the full `tests/training` suite after any
  automated import pruning.
- `.agent/tmp/` is git-ignored scratch; treat its contents as disposable.
- Historical milestone prompt files were moved from top-level `doc/` into
  `archive/legacy_milestone_prompts/`.
- Evaluator modeling changes should be recorded in
  `docs/EVALUATOR_MODEL_AUDIT.md`, not in another one-off milestone report.
