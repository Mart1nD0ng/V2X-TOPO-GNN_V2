# Environment

The environment layer must establish that the simulation space contains feasible but non-trivial topology solutions before model training.

## Required Calibration

- Pin exact 3GPP TR 37.885 and TR 38.901 versions.
- Generate urban-grid road geometry, vehicles, RSUs, LOS/NLOS states, and physical-link assumptions.
- Build sparse O(Nk) candidate edges with explicit per-node caps.
- Search physical configs over Tx power, bandwidth, HARQ, MCS, candidate radius, RSU spacing, road stride, vehicle density, and SPS collision scale.
- Evaluate KNN sweep, MST+augmentation, greedy SINR, and degree-capped spanner baselines.

## M1 Output Paths

- `src/v2x_env/urban_grid.py`
- `src/v2x_env/vehicle_snapshot.py`
- `src/v2x_env/channel_model.py`
- `src/v2x_env/candidate_graph.py`
- `src/v2x_env/baselines.py`
- `src/v2x_env/feasibility.py`
- `scripts/calibrate_environment.py`
- `configs/physical_search.yaml`
- `reports/environment_feasibility.md`

Avoid adding legacy `core/map_env.py` paths unless a later architecture decision explicitly requires them.

## Acceptance Conditions

- Candidate graph connected or near-connected in at least 95 percent of seeds.
- At least one non-neural baseline reaches a usable `query_success_proxy` in enough seeds to show feasibility.
- The easiest low-degree baseline is not saturated.
- Average selected degree stays in the operational range.
- Gradient diagnostics are nonzero and bounded in the calibrated p-range.

## Harness Status

`make env-feasibility-check` runs deterministic environment tests, sparse-complexity guardrails, and the smoke calibration CLI with outputs under `.agent/tmp/`.

`make env-feasibility-report` is the only target that refreshes canonical M1 artifacts under `reports/` and `configs/environment_medium.yaml`.

The M1 feasibility proxy reports `query_success_proxy` and `consensus_readiness_proxy` style diagnostics only. Final Avalanche reliability belongs to the later closed-form evaluator phase.

M1.2 smoke calibration evaluates both 100 and 500 vehicles over seeds 7 and 13 with balanced coverage across the search dimensions. When a usable configuration is found, `make env-feasibility-report` writes `configs/environment_medium.yaml` with the selected full config, hash, mode, and seed list. If no usable config is found, the previous medium config is preserved unless `--allow-empty-medium` is explicitly passed.

Candidate graph reports include cell occupancy, pair checks before radius filtering, cap hit ratio, and build wall time. Baseline topology reports include directed edge count, mean out degree, undirected pair count, undirected average degree, reciprocity ratio, and undirected giant component ratio.

## Dependencies

The M1 environment harness uses only CPU Python packages listed in `requirements.txt`: NumPy and PyYAML.
