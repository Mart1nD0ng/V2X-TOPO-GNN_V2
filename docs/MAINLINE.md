# MAINLINE — the canonical V2X-topo-GNN pipeline (authoritative)

This is the single source of truth for **what is production / mainline** vs **what is the
opt-in historical milestone archive**. Everything not listed under "Mainline" is a
diagnostic/milestone artifact, kept for the research record but **not part of the
production path** (the AGENTS.md "~120 diagnostic milestone sweep" — `make
diagnostics-archive-check`, opt-in, does not gate work).

## Evaluator currency (P0-1 — authoritative)

Every reliability number (F) MUST be reported in one of three **currencies**, and the currency MUST be
stated next to the number (enforced by `scripts/harness/verify_no_mean_field_reliability_claim.py`):

| currency | what | use |
|---|---|---|
| **mean-field** (`quenched_quadrature=1`) | the legacy closed form | **22–40× optimistic, gradient-blind** to the query-spread lever — a SURROGATE only; **never** a reliability conclusion |
| **quenched** (`quenched_quadrature≥21`, train may use 11) | SSMC quenched-disorder closure | the **baseline reliability currency**; production/operating-point configs now train at Q=11 and report the headline at eval Q=21 |
| **Monte-Carlo** | direct Avalanche simulation | ground-truth validation band (`validate_*_montecarlo.py`); quenched is within ~1.16× of MC at the operating point |

Each `result/<run>/` should record `evaluator_currency` (the generators emit it). Pre-P0-1 result dirs
and any table citing an unlabelled F are **mean-field surrogates** pending re-measurement in the quenched
currency — see `docs/PROJECT_STATUS_AND_NEXT_STEPS.md` (now annotated). The **relative** GNN-vs-heuristic
and D/E-ablation orderings are currency-invariant (same evaluator on both arms); the **absolute** F is not.

## Standard paper environment (authoritative substrate for paper-facing numbers)

(Deployment phases and calibration order: `docs/DEPLOYMENT_ROADMAP.md`.)

`configs/paper_environment_v1.yaml` is THE environment paper-facing results must be generated on:
**TR 37.885 channel** (`channel.pathloss_model: tr37885`, one switch flips candidate graph AND
evaluator, parity-tested) + **geometric LOS** (`candidate_graph.los_model: axis_visibility`, fixing
the ~28% collinear-NLOS artifact of the legacy same-road-segment rule) + **quenched currency**
(train Q=11 / headline-eval Q=21). `configs/production_training_v1.yaml` keeps the LEGACY physics
(now in the quenched currency) for continuity with the historical headline; any number produced on
it must say so.

**Deployment environment spec v1 — FROZEN streaming values (Roadmap Phase 0.1).** Stream
experiments (temporal/filtering ablations, fine-stage calibration) use, unless a doc states an
explicit deviation: **AR(1) hidden shadow std 4 dB / decorrelation 8 s** (`--shadow-std-db 4
--shadow-decorr-s 8`), **churn 2 births/frame, absorb_inject boundary** (`--churn-rate 2`),
**intersection turning (0.8, 0.1, 0.1)**, dt 2 s. The all-mechanisms-on composition is contract-
tested (`tests/integration/test_deployment_spec_all_on.py`). Per-node carried/recurrent state MUST
be re-keyed by `node_id` across frames under churn (`remap_carried_by_node_id`).

**Reliability targets are FLOOR-ANCHORED (Roadmap Phase 0.3).** No F target may be stated without
the perfect-link protocol floor for the (ic profile × protocol × degree budget) cell —
`scripts/analysis/run_protocol_floor_table.py`, `result/protocol_floor_table`. At the deployed
degree budget 4: toy/hard floors ≈ 0.064 (F ≤ 0.01 is infeasible BY PROTOCOL there; feasible only
at ic = 0.90). Raising the budget to 8 lowers floors ~5× (toy 0.0118, hard 0.0109; with k=7
widening 0.0080/0.0070 ≤ 0.01) — i.e. the degree budget is itself a protocol-level reliability
lever, and target feasibility is a (profile, protocol, budget) decision, not a training goal.

## The canonical pipeline (end-to-end differentiable topology constructor)

```
vehicle snapshot (positions, speed, heading)            src/v2x_env/vehicle_snapshot.py, urban_grid.py, mobility.py, profiles.py
  → candidate graph (sparse O(Nk) edges + distance/LoS) src/v2x_env/candidate_graph.py, channel_model.py
  → node/edge features                                  src/training/training_smoke.py::_build_feature_tensors
  → HierarchicalGNNScorer  → edge_score   [LEARNED]     src/models/hierarchical_gnn.py
  → TopologyConstructionLayer (hard top-k row-softmax    src/topology/construction.py
        forward; straight-through full-candidate backward)   gradient_mode="straight_through_full_candidate"
  → evaluate_v2x_graph_consensus (analytic Avalanche;   src/evaluation/v2x_consensus_bridge.py
        C/D/E, link_success via Q-function, structural    src/consensus/avalanche_closed_form.py, graph_coupled_avalanche.py
        delay + retransmission energy)  [FIXED PHYSICS]
  → compute_coupled_loss (C/D/E task objective)         src/losses/coupled_objective.py
  → coupled_backward (opt-in PCGrad / GradNorm)         src/training/gradient_governance.py
  → optimizer.step()  → updates the GNN scorer
```

A single `.backward()` on the coupled C/D/E loss updates the GNN scorer end-to-end; the
evaluator is an analytic *differentiable* model (fixed physics, no learned params), and the
constructor is a parameter-free deterministic layer with a straight-through backward. No
intermediate supervised labels / teacher targets in the default path.

## Mainline source modules (the KEEP-set)

| package | mainline files | role |
|---|---|---|
| `src/v2x_env/` | **all** (vehicle_snapshot, urban_grid, mobility, profiles, candidate_graph, channel_model, feasibility, baselines) | sample/data generation |
| `src/models/` | hierarchical_gnn, temporal_scorer, structural_objectives, structural_diagnostics, `__init__` | the GNN scorer (+ aux structural loss used by training_smoke) |
| `src/topology/` | construction, budget_adapter, constructor_profile, temporal_metrics, `__init__` | the hard-top-k constructor |
| `src/consensus/` | avalanche_closed_form, graph_coupled_avalanche, topology_query_support, `__init__` | analytic consensus math |
| `src/evaluation/` | v2x_consensus_bridge, evaluator_profile, v2x_bridge_inputs, `__init__` | C/D/E evaluator (link reliability, structural delay, retransmission energy) |
| `src/losses/` | coupled_objective, gradnorm, pcgrad, horizon_objective, `__init__` | coupled objective + gradient-governance primitives |
| `src/training/` | **only** training_smoke, gradient_governance, `__init__` | shared training helpers + PCGrad/GradNorm integration |

**Verified mainline closure (2026-06-04):** the canonical scripts import only
`src.{evaluation,losses,topology}` + `src.training.{training_smoke,gradient_governance}`,
and `training_smoke` imports only `src.{evaluation,losses,models,topology}`. It imports
**none** of the ~100 other `src/training/*` modules — those are a self-contained diagnostic
cluster disconnected from the production path.

## Canonical scripts, configs, results

- **Scripts** (`scripts/analysis/`): `run_operating_point_search.py`, `run_de_ablation.py`,
  `run_gradient_conflict_diagnostic.py`, `run_production_training.py`,
  `run_production_multiseed.py`, `generalization_common.py`, `run_{standard,density,mobility}_generalization.py`.
  Harness verifiers: `scripts/harness/verify_*.py`.
- **Configs** (`configs/`): **`operating_point_v1.yaml`** (the adopted C/D/E operating point:
  200 veh/km² × hard_low_confidence × 20 dB × degree 4, retransmission energy + structural
  delay ON), `production_training_v1.yaml` (scale/convergence baseline).
- **Result lineage** (`result/`): `operating_point_v1` → `de_ablation_v2` (coupled C/E) →
  `de_ablation_v3_governed` (PCGrad stability) → **`de_ablation_v4_structural` (coupled
  C/D/E, the headline)**; plus `gradient_conflict_v1`, `production_multiseed`,
  `generalization_v1`, `density_generalization_v1`, `mobility_generalization_v1`.

## Authoritative design docs (the decision record)

`PROJECT_STATUS_AND_NEXT_STEPS` (note: predates the C/D/E arc),
`COUPLING_AND_OPERATING_POINT_DESIGN` (Track A/B), `GRADIENT_GOVERNANCE_DESIGN` (PCGrad
earned, D-control off-ramp), **`STRUCTURAL_DELAY_MODEL_DESIGN`** (D-fix-A → coupled C/D/E),
`ARCHITECTURE`, `CODE_MAP`, `AVALANCHE_CLOSED_FORM`, `EVALUATOR_MODEL_AUDIT`, `LOSS_DESIGN`.

## Mainline verification gate

```bash
make agent-check        # invariants + core (model/topology/loss/training-smoke/env/evaluation/integration) + source + skill
make production-check    # 2000 + 10000 node convergence
```

## The archive (NOT mainline — opt-in historical record)

- **≈100 diagnostic modules** in `src/training/*` (everything except training_smoke,
  gradient_governance, `__init__`) — e.g. `*_probe`, `*_diagnostic`, `*_report`, `*_design`,
  `formal_training_v0_*`, `pure_edge_*`, `score_*`, `temperature_aware_*`, `teacher_*`,
  `support_smoothing_*`, `scale_*`, `node_count_*`, `training_v0_config`, etc., plus the
  diagnostic `*_design/_report/_diagnostic` modules in `src/models|topology|consensus`.
- Their ~101 tests and the `diagnostics-archive-check` + per-milestone Makefile targets.
- `archive/legacy_milestone_prompts/` (≈60 milestone prompt records), `.agent/tmp/` (temp probe outputs).

**DONE (2026-06-04):** the cluster was physically relocated to `archive/diagnostics/`
(336 files: `src/`, `tests/`, diagnostic `scripts/`, and `Makefile.full`) after a `git init`
baseline. The 4 package `__init__.py` were slimmed to mainline-only exports and the Makefile
cut to **56 mainline targets**. Final shape: **36 mainline `src/` modules, 78 mainline tests,
39 mainline scripts**. Full mainline gate re-verified GREEN (7 invariant verifiers + all core
test dirs + remediation tests + production-train-smoke → READY_FOR_PRODUCTION_TRAINING).
Everything is recoverable via git (`git reset --hard 5ea1cb5` for the pre-cleanup baseline).
