# NDH Parameter Registry (Gate G-NDH-PARAM-AUDIT)

Single source of truth for every parameter introduced by the **Non-Distance Headroom (NDH)
Benchmark** plus the **preserved baseline** physics/protocol it builds on. Closes gate
`G-NDH-PARAM-AUDIT` (engineering plan §2). Every later gate (`SCENE-RSU-HOTSPOT`,
`SPS-PERSISTENCE`, `CSI-AGING`, `HETEROGENEOUS-CAPACITY`, …) must take its parameter values
from this file and must add an entry here before adding a knob in code.

## 0. How to read this file

Every parameter row has the five mandated fields (engineering plan §2.2):

| field | meaning |
|---|---|
| **default** | the value used in the *canonical deployment-realistic* NDH config. Must be a value defensible as a real V2X deployment setting — **never a stress value** (forbidden shortcut #2). |
| **sweep** | the ablation grid used to map sensitivity / mechanism strength (C1/C4 of the Mechanism Identifiability Contract). |
| **stress** | the value(s) used **only** to manufacture a harder, congestion / aging / contention regime. Labelled stress; reported as stress; **never silently promoted to a headline deployment default.** |
| **source / rationale** | literature, 3GPP range, current-repo physics, or an explicitly-declared surrogate modelling choice. |
| **label** | `DEPLOYMENT` (default is a plausible real setting) or `STRESS-ONLY` (the knob exists only to build a hard regime; its "deployment" value is the benign end of the sweep). |

### Governance rules (non-negotiable, plan §2.2 + §15)

1. **No stress value may be a deployment default.** The canonical NDH-default profile (§7) is
   assembled entirely from the `default` column.
2. **Hash binding.** Every NDH experiment JSON records the physics `config_hash`
   (`RoundPhysicsConfig.config_hash`), the service-profile hash, and an NDH-mechanism config
   hash, so train==eval and figure-vs-result cannot silently diverge (Contract C5).
3. **Two reported operating points.** Headline = `default` (deployment) profile. Stress runs are
   reported *separately and labelled* (spec §17 / plan §9.2): matched-distance reliability is the
   headline judge; absolute `F_w ≤ 1e-3` deployment budget is reported alongside, not required.
4. **Observability split (Contract C2).** Each mechanism declares which of its quantities is a
   *deployable proxy* (legal model/heuristic input) vs *oracle-only truth* (diagnosis only). See
   §8. Strong heuristics get the **same** proxies the GNN gets (spec §11, constraint #19/#11).
5. **RSU is responder/witness only** for the headline macrostate: `omega_RSU = 0` (constraint #9).

---

## 1. Preserved baseline — FROZEN (must not change to manufacture a GNN win)

These are the current mainline physics/protocol/scene/service values
(`src/environment/round_physics.py`, `src/config/service_profile.py`,
`src/environment/urban_scene.py`). They are **not** an NDH contribution and **may not** be
re-tuned to create a non-distance advantage (forbidden shortcut #1, spec §2.2). Listed here so
the audit is complete and any drift is detectable via `config_hash`.

### 1.1 Fixed PHY resources (`RoundPhysicsConfig`)

| parameter | value | source / rationale |
|---|---|---|
| `fc_ghz` | 5.9 | 5.9 GHz ITS band (C-V2X / NR-V2X PC5) — spec §2.2 |
| `tx_power_dbm` | 23.0 | typical V2X sidelink Tx power — spec §2.2 |
| `noise_dbm` | −95.0 | thermal-noise floor surrogate — repo default |
| `subchannels` | 5 | Mode-2 sub-channels per slot — repo default |
| `slots_per_window` | 20 | resource-selection window slots — repo default |
| `resource_pool S = subchannels·slots_per_window` | 100 | collision pool (`~1/S` reuse) — matches spec §3.3 SPS default S=100 |
| `request_blocklength` / `response_blocklength` | 60 / 600 | FBL channel uses — repo default |
| `request_bits` / `response_bits` | 48 / 300 | request=BSM-sized payload, response larger — repo default |
| `max_harq_attempts` | 2 | HARQ in congested V2X — spec §2.2 (refs 4,5) |
| `harq_combining` | chase | repo default |
| `fading` / `use_shadow_fading` | rayleigh / True | repo default |
| `slot_time_s` | 1e-3 | one sidelink slot — repo default |
| `request_slots` / `response_slots` | 1 / 1 | base phase durations — repo default |
| `service_rate` (μ, GLOBAL) | 12.0 | **superseded by per-node μ_j in G-NDH-HETEROGENEOUS-CAPACITY** (§4) |
| `poll_window_s` (Δ_poll) | 0.01 | 10 ms polling epoch — spec §2.2 |
| `los_d0_m` | 50.0 | LOS proxy crossover distance — repo default |

> **Note (resource pool vs SPS).** The current physics already has a memoryless collision pool
> `S = subchannels·slots_per_window` (reuse prob `~1/S`). NDH-SPS (§2) does **not** change `S`; it
> adds a *persistent per-node resource bucket + sensing-based reselection* layer on top, so
> collision risk acquires history and is no longer `~1/S` i.i.d. (the non-distance structure).
> The ESP-scale campaign used `subchannels=10, slots_per_window=40` (S=400); the **NDH default is
> the repo S=100** (spec §3.3), with S∈{40,60} as congestion stress.

### 1.2 Protocol / service profile (`ConsensusServiceProfile.urban_default`)

| parameter | value | source / rationale |
|---|---|---|
| `k, alpha, beta` (quorum) | 3, 2, 3 | mainline ESP protocol — repo default |
| `max_poll_epochs` (R_d) | 20 default / **6 in mm_high stress** | deadline budget; R_d=6 is the stressed regime (EV14) — **R_d=6 is STRESS** |
| `correct_basin_mass` (ρ_f) | 0.60 | decisive majority (>½) — repo default |
| `max_wrong_basin_probability` (ε_w) | 1e-3 | reliability budget — repo default |
| `max_split_basin_probability` (ε_s) | 1e-3 | reliability budget — repo default |
| `max_deadline_miss_probability` (ε_d) | 1e-2 | deadline budget — repo default |
| `latency_quantile` | 0.95 | CVaR/quantile level for latency — repo default |
| `energy_budget` | inf (report-only) | soft budget — repo default |

> The macrostate_v2 dynamic-MC basin first-hitting judge, full physics, and these reliability
> constraints are **fixed**. NDH never weakens the judge (constraint #2).

### 1.3 Baseline scene geometry (`build_manhattan_scene`)

| parameter | value | source / rationale |
|---|---|---|
| `block_m` | 100.0 | Manhattan block length — repo default; **NDH makes it lognormal (§6)** |
| `lane_jitter_m` | 3.0 | lateral lane offset — repo default |
| `comm_radius` | 80.0 | communication candidate radius — repo default |
| `int_radius` | 160.0 | interference radius (≥ comm) — repo default |

---

## 2. Mechanism 1 — SPS persistent collision (`G-NDH-SPS-PERSISTENCE`, spec §3)

**Claimed structure (C1):** a node holds a *persistent* resource bucket; same-bucket neighbours on
`G_int` collide repeatedly, so collision risk is **non-local and history-correlated** —
`d_ij↓ ⇏ conflict↓`. Control = bucket re-randomised every round (no persistence) ⇒ recovers
`~1/S` i.i.d. collisions.

| parameter | default | sweep | stress | source / rationale | label |
|---|---|---|---|---|---|
| `resource_pool_size` S | 100 | {40, 60, 100} | 40, 60 (congestion) | matches repo S=100 (§1.1); spec §3.3 | STRESS-ONLY (low S) |
| `RRI_ms` (resource-reservation interval) | 100 | {50, 100, 200} | 50 (dense traffic) | BSM/CAM 10 Hz ⇒ 100 ms typical; spec §2.3 refs 3,4 | DEPLOYMENT |
| `reselection_interval_ms` | 1000 | {500, 1000, 2000} | 500 (more churn) | SPS reselection-counter range; spec §3.3 | DEPLOYMENT |
| `keep_probability` (prob_resource_keep) | 0.8 | {0.5, 0.8} | 0.5 | 3GPP SPS `probResourceKeep` typical 0.8; spec §3.3 | DEPLOYMENT |
| `sensing_noise_std` σ_sense | 0.1 | {0.0, 0.1, 0.2} | 0.2 | imperfect sensing surrogate; spec §3.3 | DEPLOYMENT |
| `tau_res` (selection temperature) | 4.0 | {2, 4, 8} | — | softmax sharpness of sensing-based selection `P(r)∝exp(−τ·occ_r)`; surrogate (spec §3.3 eq.) | DEPLOYMENT (modelling) |
| `kappa_res` (collision rate κ_res) | 0.5 | {0.25, 0.5, 1.0} | 1.0 | maps same-bucket contention load → collision prob `1−exp(−κ·(L−a)_+)`; surrogate (spec §3.4) | STRESS-ONLY (high κ) |

**Deployable proxies (model + heuristics, spec §3.5):** `resource_bucket_embedding`,
`resource_age_norm`, `sensed_CBR`, `same_resource_conflict_degree`, `recent_collision_ema`,
`recent_ack_success_ema`; edge: `src/dst_resource_age`, `same_resource_bucket`,
`source_resource_conflict_at_dst`, `responder_resource_conflict_at_src`,
`stale_resource_occupancy`. **Forbidden (truth, oracle-only):** `future_resource_id`,
`future_collision_outcome`, `future_delivery_success`, MC outcome.

---

## 3. Mechanism 2 — RSU placement (`G-NDH-SCENE-RSU-HOTSPOT`, spec §4.3)

**Claimed structure (C1):** witnesses with higher capacity sit at intersections/hotspots; "nearest
peer" ≠ "best responder". Control = no RSU (vehicles only). **Hard density caps prevent a trivial
nearest-RSU oracle** (risk R3).

| parameter | default | sweep | stress | source / rationale | label |
|---|---|---|---|---|---|
| `p_intersection_rsu` | 0.25 | {0.1, 0.25, 0.5} | — | intersection RSU coverage; **HARD CAP ≤ 0.5** (constraint #10) | DEPLOYMENT |
| `p_hotspot_rsu_boost` | 0.25 | {0.25, 0.5} | 0.5 | extra RSU near queue hotspots; spec §4.3 | DEPLOYMENT |
| `max_rsu_fraction_of_nodes` | 0.10 | {0.05, 0.10, 0.15} | — | **HARD CAP**: total RSU ≤ 15% nodes; spec §4.3 (anti-trivial) | DEPLOYMENT |
| `min_rsu_spacing_m` | 300 | {200, 300, 500} | 200 | RSU deployment spacing; spec §4.3 | DEPLOYMENT |
| `rsu_roadside_offset_m` | 5 | {5, 10} | — | roadside placement offset; spec §4.3 | DEPLOYMENT |
| `omega_RSU` (macrostate weight) | **0.0 (FIXED)** | — | — | RSU is responder/witness only (constraint #9) | DEPLOYMENT (fixed) |

**Hard constraints (enforced in scene builder + tested):** `p_intersection_rsu ≤ 0.5`;
`#RSU ≤ max_rsu_fraction·N`; RSU never enters macrostate C/W/U weights (`omega_RSU=0`).

---

## 4. Mechanism 3 — Heterogeneous receiver capacity (`G-NDH-HETEROGENEOUS-CAPACITY`, spec §4.4)

**Claimed structure (C1):** per-node service rate `μ_j` replaces the global constant, so queue delay
`ρ_j=(Λ_j+b_j)/μ_j` differs by responder; a slightly-farther high-μ peer can beat a near low-μ
peer. Control = all `μ_j=μ` (recovers current global queue). **Must enter full physics + dynamic
MC, not just tests** (plan §6 acceptance).

| parameter | default | sweep | stress | source / rationale | label |
|---|---|---|---|---|---|
| `mu_vehicle_base` μ_veh | 8 | {4, 6, 8, 12} | 4 (queue stress) | polls/window; sits below the frozen global μ=12 to leave RSU headroom; spec §4.4 | STRESS-ONLY (μ=4) |
| `vehicle_capacity_logstd` σ_μ | 0.5 | {0.25, 0.5, 0.75} | 0.75 | lognormal capacity spread; spec §4.4 | DEPLOYMENT |
| `rsu_capacity_multiplier` | 5 | {3, 5, 10} | — | RSU μ = mult·μ_veh (RSU stronger); spec §4.4 | DEPLOYMENT |
| `rsu_capacity_logstd` σ_rsu | 0.1 | {0.1, 0.25} | — | RSU capacity spread; spec §4.4 | DEPLOYMENT |
| `capacity_proxy_noise` σ_obs | 0.2 | {0.0, 0.2, 0.5} | 0.5 | noisy proxy `μ̂_j=μ_j·exp(σ_obs·ξ)`; spec §4.4/§4.5 | DEPLOYMENT |
| `background_load` b_j | 0.0 | {0.0, …} | — | extra processing load; 0 in Phase 1, reserved (spec §4.4) | DEPLOYMENT |

**Deployable proxy:** `capacity_proxy_log = log μ̂_j`, `capacity_proxy_uncertainty`,
`response_latency_ema`, `ack_success_ema`, `queue_delay_ema`, `local_rsu_density`; edge:
`receiver_capacity_proxy`, `predicted_receiver_queue_ratio`, `receiver_ack_history`,
`distance_to_nearest_rsu`. **Forbidden:** true `μ_j` (model reads only the noisy proxy; same for
heuristics — spec §4.5).

---

## 5. Mechanism 4 — CSI aging (`G-NDH-CSI-AGING`, spec §5)

**Claimed structure (C1):** the model sees **stale** CSI `γ̂_ij(t−a)`; physics uses **current**
`γ_ij(t)`. Distance/stale-link-quality stops being optimal where the channel has decorrelated;
age + uncertainty become informative. Control = `csi_age=0` (recovers current CSI feature). Phase 1
static (no GRU).

| parameter | default | sweep | stress | source / rationale | label |
|---|---|---|---|---|---|
| `csi_age_ms` | 100 | {0, 50, 100, 200, 500} | 200, 500 (deep staleness) | CAM 10 Hz ⇒ ≥100 ms inter-update; spec §5.2 | DEPLOYMENT |
| `csi_noise_std_db` | 1 | {0, 1, 2, 4} | 4 | CSI estimation/quantisation noise; spec §5.2 | DEPLOYMENT |
| `shadow_ar_std_db` | 4 | {2, 4, 6} | 6 | log-normal shadow std (consistent w/ PHY shadow); spec §5.2 | DEPLOYMENT |
| `shadow_decorrelation_s` | 3 | {1, 3, 8} | 1 (fast decorrelation) | shadow AR(1) decorrelation time; spec §5.2 | DEPLOYMENT |

**Deployable proxy (edge):** `stale_sinr_db`, `stale_link_delivery`, `csi_age_norm`,
`csi_uncertainty = σ_γ(a)`, `stale_vs_distance_residual`. **Forbidden:** current `γ_ij(t)`, any
future CSI. Physics path is unchanged (current γ); only the *feature* is stale (no train/eval
physics mismatch — constraint #4).

---

## 6. Mechanism 5 — Nonuniform road + intersection-queue hotspots (`G-NDH-SCENE-RSU-HOTSPOT`, spec §6)

**Claimed structure (C1):** perturbed urban grid + a *few* intersection-queue hotspots create local
density/contention non-uniformity that distance alone cannot read. Control = uniform Manhattan grid,
no hotspots. Phase 1 hotspots static.

| parameter | default | sweep | stress | source / rationale | label |
|---|---|---|---|---|---|
| `block_length_logstd` σ_block | 0.2 | {0.0, 0.2, 0.35} | 0.35 | `block ~ LogNormal(log 100, σ)`; surrogate urban irregularity (spec §6.2) | DEPLOYMENT |
| `intersection_jitter_m` | 20 | {0, 20} (U(−j,+j)) | — | intersection offset Uniform(−20,20); spec §6.2 | DEPLOYMENT |
| `road_presence_probability` | 1.0 | {0.8, 0.9, 1.0} | 0.8 (missing roads) | partial road grid; spec §6.2 | STRESS-ONLY (<1.0) |
| `num_hotspots` | 2 | {1, 2, 3} | 3 | sparse queue hotspots; **≤10% of intersections** (spec §6.3) | DEPLOYMENT |
| `hotspot_radius_m` | 50 | {30, 50, 80} | 80 | hotspot extent; spec §6.3 | DEPLOYMENT |
| `hotspot_vehicle_fraction` | 0.2 | {0.1, 0.2, 0.3} | 0.3 | **≤30% of vehicles** in hotspots (HARD CAP, spec §6.3) | DEPLOYMENT |
| `queue_length_m` | 100 | {50, 100, 150} | 150 | along-road queue extent; spec §6.3 | DEPLOYMENT |

**Hard constraints (enforced + tested):** hotspots ≤ 10% of intersections; hotspot vehicles ≤ 30%
of total; hotspot centres pairwise distance ≥ 2·`hotspot_radius_m` (non-overlapping); road graph
stays connected.

**Deployable proxy:** node `local_density`, `intersection_distance`, `hotspot_score`,
`queue_rank_along_road`, `local_CBR`, `local_candidate_degree`, `local_interference_degree`; edge
`edge_crosses_hotspot`, `src/dst_hotspot_score`, `edge_local_density_pair`. Strong heuristics read
the **same** local-density/hotspot proxies (spec §6.4).

---

## 7. Canonical NDH operating profiles (assembled from the columns above)

Two named, hash-bound profiles. **Headline runs use `NDH-DEPLOYMENT`.** Stress runs use
`NDH-STRESS` and are reported separately and labelled (rule §0.3).

### `NDH-DEPLOYMENT` (deployment-realistic; headline)

All parameters at their **`default`** column value. R_d = 20 (non-stressed deadline). S = 100.
μ_veh = 8, σ_μ = 0.5, rsu_mult = 5. csi_age = 100 ms. 2 hotspots, ≤30% hotspot vehicles. RSU
`p_int = 0.25`, fraction-cap 0.10, `omega_RSU = 0`.

### `NDH-STRESS` (hard regime; reported separately, never a deployment default)

S = 40 (congestion), κ_res = 1.0, keep_prob = 0.5; μ_veh = 4 (queue stress); csi_age = 500 ms,
shadow_decorr = 1 s; road_presence = 0.8; 3 hotspots, 30% hotspot vehicles, radius 80; R_d = 6
(stressed deadline). Used to *find* whether constrained-oracle equal-reliability headroom appears,
not to claim deployment behaviour.

> Single-mechanism regimes (`NDH-SPS`, `NDH-RSU-CAPACITY`, `NDH-CSI-AGING`, `NDH-HOTSPOT`) turn on
> exactly one mechanism over the `NDH-DEPLOYMENT` base (others at their structure-absent control),
> for the Mechanism Identifiability Contract C4 factorial.

---

## 8. Observability split (Contract C2) — which quantity is a legal model input

| quantity | role | legal model/heuristic input? |
|---|---|---|
| resource bucket id, age, sensed CBR, collision/ACK EMA, conflict degree | SPS proxy | **YES** (deployable) |
| `μ̂_j` noisy capacity, queue-delay EMA, response-latency/ACK EMA, RSU density | capacity proxy | **YES** (deployable) |
| stale γ̂ SINR, stale delivery, CSI age, CSI uncertainty, stale-vs-distance residual | CSI proxy | **YES** (deployable) |
| local density, intersection distance, hotspot score, queue rank | scene proxy | **YES** (deployable) |
| node type (vehicle/RSU), edge-to-RSU, distance-to-nearest-RSU | structural | **YES** (deployable) |
| true `μ_j`, current `γ_ij(t)`, future resource id, future collision/delivery, MC basin outcome, peer current vote | truth / future | **NO** — oracle diagnosis only (constraint #8, C2) |

The **free-edge oracle** and **wrong-penalized oracle** (`G-NDH-ORACLE-FRONTIER`) may optimise
against MC truth (they are upper-bound probes, excluded from deployable claims). The GNN and every
heuristic baseline see **only** the YES rows.

---

## 9. References (spec §13)

1. Harounabadi et al., "V2X in 3GPP Standardization: NR Sidelink in Rel-16 and Beyond," 2021.
2. Castañeda Garcia et al., "A Tutorial on 5G NR V2X Communications," 2021.
3. Dayal et al., "Adaptive Semi-Persistent Scheduling for Enhanced On-road Safety in Decentralized V2X Networks," 2021.
4. Jeon et al., "Reducing Message Collisions in Sensing-based Semi-Persistent Scheduling," 2018.
5. Fouda et al., "HARQ Retransmissions in C-V2X: A BSM Latency Analysis," 2023.
6. McCarthy et al., "OpenCV2X: Modelling of the V2X Cellular Sidelink and Performance Evaluation for Aperiodic Traffic," 2021.

---

## 10. Acceptance checklist (G-NDH-PARAM-AUDIT)

- [x] Every SPS / RSU / capacity / CSI / hotspot parameter has a registry entry with
      default / sweep / stress / source / label (§2–§6).
- [x] No stress value appears as a deployment default; the `NDH-DEPLOYMENT` profile (§7) is built
      only from the `default` column.
- [x] Preserved baseline physics/protocol/scene enumerated and marked FROZEN (§1).
- [x] Hard density / fraction caps recorded (RSU ≤ 0.5 intersection prob & ≤ 0.10–0.15 fraction;
      hotspots ≤ 10% intersections & ≤ 30% vehicles; `omega_RSU=0`) (§3, §6).
- [x] Observability split recorded: deployable proxy vs oracle-only truth (§8), satisfying
      Contract C2; heuristics get the same proxies as the GNN.
- [x] Config-hash binding rule stated (§0.2): all NDH result JSONs carry physics/profile/mechanism
      hashes (enforced per-experiment in later gates).

> **Source/rationale honesty.** Rows tagged "surrogate / modelling" (`tau_res`, `kappa_res`,
> `block_length_logstd`, sensing/collision maps) are **declared lightweight surrogates**, not
> measured 3GPP values — consistent with the no-ns-3 decision (Q4). They are constrained to
> non-collapse operating bands (risk R4) via their sweeps; none is tuned to favour the GNN.
