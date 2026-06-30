# Non-Distance Headroom Benchmark 工程工作流

## 0. 执行原则

本轮执行用户确认的两阶段路线：

\[
\boxed{
\text{环境机制实现}
\rightarrow
\text{oracle-first headroom 验证}
\rightarrow
\text{静态 ESD-GNN v2 训练}
\rightarrow
\text{必要时 Temporal-ESDGNN v2}
}
\]

禁止直接通过调环境制造 GNN 胜利。每个 regime 必须先证明：

\[
\text{wrong-penalized oracle has matched-reliability headroom over distance}.
\]

若 oracle 无 headroom，则该 regime 不进入 GNN headline。

---

# 1. Gate 总览

```text
G-NDH-PARAM-AUDIT
G-NDH-SCENE-RSU-HOTSPOT
G-NDH-SPS-PERSISTENCE
G-NDH-CSI-AGING
G-NDH-HETEROGENEOUS-CAPACITY
G-NDH-FEATURE-SCHEMA
G-NDH-BASELINE-ENVELOPE
G-NDH-ORACLE-FRONTIER
G-NDH-STATIC-ESDGNN-V2
G-NDH-TEMPORAL-NEED
G-NDH-TEMPORAL-ESDGNN-V2
G-NDH-SYNTHESIS
```

---

# 2. G-NDH-PARAM-AUDIT

## 目标

确保新增参数符合真实部署范围或有文献依据。

## 工作

1. 建立 `docs/NDH_PARAMETER_REGISTRY.md`。
2. 每个参数记录：
   - name；
   - default；
   - sweep；
   - stress；
   - source / rationale；
   - deployment vs stress 标签。
3. 保持以下 baseline 不随意改变：
   - carrier 5.9GHz；
   - Ptx 23 dBm；
   - poll window 10ms；
   - full physics；
   - dynamic MC judge；
   - macrostate_v2 metrics。

## 验收

- 所有新增参数有 registry entry；
- stress 参数不会作为 deployment default；
- 所有 experiment result 包含 config hash。

---

# 3. G-NDH-SCENE-RSU-HOTSPOT

## 目标

实现非均匀路网、少量 intersection queue hotspot、RSU 生成。

## 工作

### 3.1 非均匀路网

新增：

```text
src/environment/nonuniform_urban_scene.py
```

支持：

```text
block_length_lognormal
intersection_jitter
road_presence_probability
```

### 3.2 Intersection queue hotspot

支持：

```text
num_hotspots ∈ {1,2,3}
hotspot_radius_m
hotspot_vehicle_fraction
queue_length_m
```

约束：

```text
hotspots <= 10% intersections
hotspot vehicle fraction <= 30%
hotspots non-overlapping
```

### 3.3 RSU 生成

支持：

```text
intersection_rsu_probability <= 0.5
hotspot_rsu_boost
max_rsu_fraction
min_rsu_spacing_m
rsu_roadside_offset_m
```

默认：

```text
RSU responder-only
omega_RSU = 0
```

## 测试

- RSU 数量不超过上限；
- RSU 不参与 macrostate weights；
- hotspot 数量受限；
- road graph 连通；
- 非均匀路网不产生孤立候选图异常；
- scene seed reproducible。

## 验收

可生成：

```text
control_uniform_grid
ndh_hotspot_static
ndh_hotspot_rsu_static
```

---

# 4. G-NDH-SPS-PERSISTENCE

## 目标

实现 sensing-based SPS resource persistence surrogate。

## 工作

新增：

```text
src/environment/sps_resource.py
```

实现：

```text
resource_bucket
resource_age
time_to_reselection
sensed_occupancy
CBR
recent_collision_ema
```

resource selection：

```text
P(resource=r) ∝ exp(-tau_res * sensed_occupancy_r)
```

collision：

```text
same resource + interference graph neighbor => persistent collision risk
```

## 参数

```text
resource_pool_size ∈ {40,60,100}
RRI_ms ∈ {50,100,200}
reselection_interval_ms ∈ {500,1000,2000}
keep_probability ∈ {0.5,0.8}
sensing_noise_std ∈ {0.0,0.1,0.2}
```

## 测试

- same-resource conflict increases collision;
- different-resource conflict does not;
- reselection changes resource distribution;
- resource age increments and resets;
- future resource state not exposed to model;
- desired-edge self-exclusion remains valid.

## 验收

SPS trace appears in mandatory mechanism trace:

```text
sps_persistence=true
resource_conflict_graph=true
```

---

# 5. G-NDH-CSI-AGING

## 目标

让模型只能看到 stale CSI，而 physics 使用当前 CSI。

## 工作

新增：

```text
src/environment/csi_aging.py
```

实现 edge-level:

```text
stale_sinr_db
stale_delivery
csi_age_ms
csi_uncertainty
stale_vs_distance_residual
```

真实 physics 仍使用 current SINR。

## 参数

```text
csi_age_ms ∈ {0,50,100,200,500}
csi_noise_std_db ∈ {0,1,2,4}
shadow_ar_std_db ∈ {2,4,6}
shadow_decorrelation_s ∈ {1,3,8}
```

## 测试

- csi_age=0 recovers current CSI feature;
- larger age increases feature error;
- model feature never includes current future CSI;
- static GNN can read age and uncertainty;
- result JSON records CSI feature mode.

## 验收

CSI aging features appear in `SceneFeaturesV2`.

---

# 6. G-NDH-HETEROGENEOUS-CAPACITY

## 目标

实现车辆和 RSU receiver capacity 差异。

## 工作

修改 round physics queue model：

\[
\rho_j = \frac{\Lambda_j + b_j}{\mu_j}.
\]

新增 node capacity model：

```text
vehicle_capacity_lognormal
rsu_capacity_lognormal
noisy_capacity_proxy
response_latency_ema
ack_success_ema
```

## 参数

```text
mu_vehicle_base ∈ {4,6,8,12}
vehicle_capacity_logstd ∈ {0.25,0.5,0.75}
rsu_capacity_multiplier ∈ {3,5,10}
rsu_capacity_logstd ∈ {0.1,0.25}
capacity_proxy_noise ∈ {0.0,0.2,0.5}
```

## 测试

- higher receiver capacity reduces queue delay/drop;
- identical capacity recovers old behavior;
- noisy proxy differs from true capacity;
- RSU capacity higher on average than vehicle;
- RSU overload still possible.

## 验收

capacity mechanism enters full physics and dynamic MC, not just tests.

---

# 7. G-NDH-FEATURE-SCHEMA

## 目标

升级 ESD-GNN 输入特征和图结构。

## 工作

新增：

```text
src/models/scene_features_v2.py
src/models/esd_gnn_v2.py
```

node features：

```text
node_type_vehicle
node_type_rsu
capacity_proxy_log
capacity_uncertainty
resource_age_norm
resource_busy_ratio
same_resource_conflict_degree
ack_success_ema
queue_delay_ema
local_density
hotspot_score
intersection_distance
```

edge features：

```text
stale_sinr_db
stale_link_delivery
csi_age_norm
csi_uncertainty
same_resource_bucket
resource_conflict_count
edge_to_rsu
receiver_capacity_proxy
predicted_receiver_queue_ratio
stale_vs_distance_residual
edge_crosses_hotspot
```

graph channels：

```text
G_comm
G_int
G_region
G_resource_conflict
G_rsu_coverage
```

Phase 1 may encode `G_resource_conflict` and `G_rsu_coverage` as features; true graph aggregation can be added if oracle headroom is large.

## 测试

- no truth leakage;
- feature missing masks work;
- old scene features reproducible when mechanisms off;
- scaling remains O(E);
- all baselines get equal observable features.

## 验收

`ESDGNNStaticV2` can run in all four NDH regimes.

---

# 8. G-NDH-BASELINE-ENVELOPE

## 目标

建立强启发式集合，防止 GNN 只赢弱 baseline。

## 必须实现

```text
distance
stale_link_quality
capacity_aware
resource_aware
distance_plus_capacity
distance_plus_resource
distance_plus_csi_age
load_balanced
rsu_nearest
rsu_capacity_aware
local_density_aware
best_heuristic_envelope
```

## 测试

- 每个 baseline 只使用 deployable proxy；
- baseline 通过同一 canonical full-physics dynamic-MC path；
- best_heuristic_envelope 记录每个 scene 的 winning heuristic；
- no baseline uses oracle-only truth。

## 验收

GNN headline 必须相对 `best_heuristic_envelope` 报告。

---

# 9. G-NDH-ORACLE-FRONTIER

## 目标

每个 regime 先用 constrained oracle 判断是否存在等可靠性 headroom。

## 实验矩阵

Regimes:

```text
NDH-SPS
NDH-RSU-CAPACITY
NDH-CSI-AGING
NDH-HOTSPOT
NDH-SPS+RSU
NDH-SPS+CSI
NDH-RSU+HOTSPOT
NDH-FULL-STATIC
```

Scales:

```text
N ∈ {120,300,1000}
```

Oracle arms:

```text
free_edge_oracle
wrong_penalized_oracle
distance
best_heuristic_envelope
```

Penalty sweep:

```text
lambda_wrong ∈ {0,1,2,5,10,20,50}
```

## 判据

Primary:

\[
F_w^{oracle,UCB}
\le
F_w^{distance,UCB}+\delta_w
\]

and

\[
J^{oracle}
<
J^{distance}-\delta_J.
\]

Also report:

\[
F_w^{oracle,UCB}\le10^{-3}.
\]

## 验收

Only regimes with positive constrained oracle headroom proceed to GNN training.

---

# 10. G-NDH-STATIC-ESDGNN-V2

## 目标

在有 oracle headroom 的 regime 上训练静态 ESD-GNN v2。

## 训练

Allowed:

```text
MC-faithful per-node credit
oracle distillation warm start
self-critical distance baseline
reliability penalty
```

Not allowed:

```text
future CSI
future resource state
MC outcome as feature
truth labels in deployed model
```

## 比较对象

```text
distance
best_heuristic_envelope
oracle
wrong_penalized_oracle
ESDGNNStaticV2
```

## 验收

满足 C 或 D：

### C: dominance

```text
F_wrong not worse than distance
P_correct not worse than distance
D99 / energy / deadline strictly better
```

### D: oracle gap closure

```text
OGC >= 0.3 initial
OGC >= 0.5 strong
```

---

# 11. G-NDH-TEMPORAL-NEED

## 目标

决定是否进入 temporal branch。

## 实验

比较：

```text
static-observation oracle
history-aware oracle
static ESDGNN v2
```

History includes:

```text
ACK history
CBR history
resource conflict history
CSI time series
queue/load history
```

## 触发条件

进入 temporal only if:

```text
history-aware oracle CI-separately beats static-observation oracle
or static GNN fails to close headroom that history oracle exposes
```

否则 Phase 2 不做。

---

# 12. G-NDH-TEMPORAL-ESDGNN-V2

## 目标

重建 temporal model，不直接复用 legacy 结果。

## 要求

- macrostate_v2 metrics;
- dynamic-MC final judge;
- full physics;
- node/RSU track id carry;
- churn handling;
- feature masks;
- emission v2 stop-gradient;
- no old node-union metrics.

## 测试

- hidden state carry correct under churn;
- RSU hidden state persistent;
- new vehicle hidden state neutral;
- emission bounded;
- no truth leakage;
- temporal improves only if history oracle had headroom.

---

# 13. G-NDH-SYNTHESIS

## 输出

1. Parameter registry。
2. Mechanism evidence。
3. Oracle headroom map。
4. Baseline envelope map。
5. Static GNN v2 results。
6. Temporal need decision。
7. Recommendation for paper direction。

## 最终解释规则

- If no constrained oracle headroom: environment mechanism does not create legal superiority.
- If oracle headroom exists but GNN fails: model/training gap.
- If GNN beats best heuristic under matched reliability: valid superiority claim.
- If GNN only beats weak baselines: not a headline.
- If temporal oracle beats static oracle: Phase 2 justified.
- If temporal GNN wins without temporal oracle headroom: suspect leakage or evaluation bug.

---

# 14. 每轮 LOOP

1. Read technical spec and current progress.
2. Pick one smallest unfinished gate.
3. Write math/API contract.
4. Write failing tests first.
5. Implement only through canonical path.
6. Run unit, integration, dynamic-MC, metric-schema, hash checks.
7. Run Mechanism Identifiability Contract audit.
8. Run oracle/baseline sanity if mechanism affects physics.
9. Write evidence JSON and decision log.
10. Commit or manifest-bind.
11. Continue.

---

# 15. 禁止捷径

1. 不得为了让 GNN 赢而设置不现实参数。
2. 不得把 stress 参数写成 deployment default。
3. 不得跳过 oracle-first。
4. 不得只和 old distance/uniform 比。
5. 不得不给强启发式同等可观测特征。
6. 不得让 RSU 过密导致 nearest-RSU trivial。
7. 不得把 oracle-only truth 输入 GNN。
8. 不得复活 legacy GRU 结果作为当前证据。
9. 不得用 old node-union/global-product metrics。
10. 不得把 un-gated Pc gain 写成 reliability gain。
11. 不得训练/评估 physics mismatch。
12. 不得用 figure script 重算指标。
13. 不得用计算扩展替代性能扩展。

---

# 16. 停止条件

Stop and report if:

1. 参数缺乏真实依据；
2. SPS / RSU / CSI / hotspot 任一机制无法通过 no-leakage audit；
3. constrained oracle 无 headroom；
4. best heuristic envelope 已经吃掉全部 headroom；
5. static GNN 无法拟合 oracle despite features；
6. temporal history oracle 不优于 static oracle；
7. rare-event certification infeasible at required budget；
8. dynamic-MC 与 claimed mechanism 矛盾。
