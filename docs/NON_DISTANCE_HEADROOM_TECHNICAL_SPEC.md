# Non-Distance Headroom Benchmark 技术规范

## 0. 本轮目标

本轮不是“为了让 GNN 赢而随意加复杂环境”，而是建立一个更接近真实部署、并且能产生 **distance 无法解释的可靠性保持 headroom** 的 V2X 共识仿真基准。

当前项目已经证明：

- 在 `mm_high(0.35,0.25), R_d=6` 高压工况下，distance 是 safety-first Pareto frontier point。
- un-gated free-edge MC oracle 可以超过 distance，但主要通过降低 deadline miss，并伴随更高 `macro_F_wrong`。
- wrong-penalized oracle 回到 distance 水平，说明当前工况中等可靠性 headroom 近似为 0。
- 当前 ESD-GNN 结构有 source-side / destination-side / interference / region pooling / load refinement 通道，但输入特征仍主要是 degree、region size、distance、LOS、same-region，因此很容易学成 distance-like 策略。
- legacy 中存在 Graph-GRU / emission 机制，但它不属于当前 `macrostate_v2 + dynamic-MC + full-physics` 主线，不能直接当作新主线证据。

本轮目标是：

\[
\boxed{
\text{构造 Non-Distance Headroom Benchmark}
\rightarrow
\text{先用 constrained oracle 证明等可靠性 headroom}
\rightarrow
\text{再训练静态 ESD-GNN}
\rightarrow
\text{必要时进入 Temporal-ESDGNN v2}
}
\]

---

## 1. 用户决策固化

本轮采用以下决策。

| 编号 | 决策 |
|---|---|
| Q1 | 两阶段：先做 headroom benchmark，再训练 GNN |
| Q2 | 环境 + 模型共同贡献 |
| Q3 | 关键参数文献/标准可追溯 |
| Q4 | 继续轻量 surrogate，不接入 ns-3 |
| Q5 | SPS 采用 sensing-based resource selection surrogate |
| Q6 | 模型可观察统计 proxy + resource bucket / age |
| Q7 | RSU 只作为 responder / witness，不计入 macrostate participation |
| Q8 | Intersection RSU + hotspot RSU，RSU 概率最多 0.5，热点附近可更高但总量不能过多 |
| Q9 | node type + noisy capacity proxy + response/ACK history |
| Q10 | CSI aging 两阶段：先静态 proxy，必要时 temporal |
| Q11 | Temporal-ESDGNN v2 放在 Phase 2 |
| Q12 | hotspot 来源为 intersection queue；路网十字路口不要过于均匀；只设置少量热点 |
| Q13 | hotspot 先静态，之后再考虑动态建模 |
| Q14 | 扩展特征 + 增加 resource-conflict graph / RSU-coverage graph |
| Q15 | oracle 可用 truth 诊断；GNN 训练/部署只能用 deployable proxy |
| Q16 | 坚持 oracle-first |
| Q17 | headline 用 matched-distance reliability，同时报告绝对部署预算 |
| Q18 | 主胜利标准允许 C 或 D：支配 distance 或 oracle-gap closure |
| Q19 | 强启发式必须使用同等可观测信息 |
| Q20 | 首轮用小/中规模机制验证 |
| Q21 | 本轮是机制和 headroom 发现，不直接要求最终论文主结果 |

---

# 2. 真实部署参数原则

## 2.1 不引入 ns-3，但不能随意造参数

本轮继续使用轻量、近线性、可解释 surrogate。所有新增机制必须满足：

1. 参数落在真实 V2X 部署或文献常见范围内；
2. 关键参数在文档中记录来源；
3. 每个参数都有 default / stress / ablation 三档；
4. 任何 stress 参数都不能伪装成 deployment default；
5. 训练、评估、图表必须记录完整 config hash。

## 2.2 推荐保留的基础物理量

保持当前项目已有基本设定，除非后续参数审计发现不合理：

```text
carrier_frequency = 5.9 GHz
vehicle_tx_power = 23 dBm
poll_window = 10 ms
HARQ_attempts = 2
request_response_polling = enabled
finite_blocklength = enabled
full_physics = enabled
```

这些是当前仓库主线已经使用或接近使用的参数范围。它们在本轮不是创新点，不能通过改变它们制造虚假的 GNN 优势。

## 2.3 文献依据

本轮新增机制受到以下事实约束：

- C-V2X / NR-V2X sidelink 的 direct PC5 通信可覆盖 V2V / V2I / V2P，通常工作在 5.9 GHz ITS 频段。
- LTE-V2X Mode 4 与 NR-V2X Mode 2 都有分布式 / sensing-based resource selection 机制，semi-persistent scheduling 会引入资源记忆和持续碰撞问题。
- BSM/CAM 等周期性安全消息常用 SPS/RRI 思路，HARQ 在拥塞场景中会消耗额外资源并影响尾时延。
- Aperiodic traffic 会显著影响 sidelink scheduling performance，因此 background bursts 是合理的后续扩展。
- 本轮不复现完整 3GPP PHY/MAC，而是构造可解释 surrogate。

参考文献见文末。

---

# 3. 新环境机制一：SPS-Persistent-Collision Regime

## 3.1 科学目的

制造 distance 不能解释的非局部、历史相关资源冲突。

当前 distance 强，是因为距离几乎同时解释链路成功率、时延、能耗。SPS 持久碰撞会打破这个关系：

\[
d_{ij}\downarrow
\not\Rightarrow
\text{resource conflict risk}\downarrow.
\]

近邻可能长期占用冲突资源；稍远但资源干净的 peer 可能更优。

## 3.2 状态变量

每个节点 \(u\) 持有：

\[
r_u(t)\in\{1,\ldots,S\}
\]

表示当前 resource bucket。

\[
a_u^{res}(t)
\]

表示 resource age 或 time-to-reselection。

\[
\mathrm{CBR}_u(t)
\]

表示 sensed channel busy ratio。

\[
\mathrm{ACKEMA}_u(t)
\]

表示近期 ACK 成功率或 collision history。

## 3.3 资源选择 surrogate

每个节点每隔若干 RRI 或 reselection counter 到期时重新选择资源：

\[
P(r_u(t+1)=r \mid \mathrm{sensed\ occupancy})
\propto
\exp(-\tau_{res}\,\widehat{\mathrm{occ}}_r).
\]

其中 \(\widehat{\mathrm{occ}}_r\) 是 sensed occupancy，不是真实 future collision。

默认使用 sensing-based surrogate：

```text
resource_pool_size S ∈ {40, 60, 100}
RRI_ms ∈ {50, 100, 200}
reselection_interval_ms ∈ {500, 1000, 2000}
keep_probability ∈ {0.5, 0.8}
sensing_noise_std ∈ {0.0, 0.1, 0.2}
```

`S=100` 与当前 `subchannels × slots_per_window` 默认规模一致，可作为 default；`S=40/60` 用于 congestion stress。

## 3.4 Collision 机制

对 request edge \(i\to j\)，如果存在干扰 transmitter \(u\) 满足：

\[
r_u(t)=r_i(t)
\]

且 \(u\in\mathcal N_{int}(j)\)，则形成 same-resource interference。

定义：

\[
L_{j,r}^{res}(t)=
\sum_{u\in\mathcal N_{int}(j)}
\mathbf 1\{r_u(t)=r\}\,a_u(t).
\]

request collision risk：

\[
p_{ij}^{col,req}(t)
=
1-
\exp\left[-\kappa_{res}
\left(L_{j,r_i}^{res}(t)-a_i(t)\right)_+
\right].
\]

response phase 类似，用 responder \(j\) 的 resource bucket。

## 3.5 模型可观测特征

GNN 允许看到：

node features：

```text
resource_bucket_embedding
resource_age_norm
sensed_CBR
same_resource_conflict_degree
recent_collision_ema
recent_ack_success_ema
```

edge features：

```text
src_resource_age
dst_resource_age
same_resource_bucket
source_resource_conflict_at_dst
responder_resource_conflict_at_src
stale_resource_occupancy
```

禁止输入：

```text
future_resource_id
future_collision_outcome
future_delivery_success
MC outcome
```

## 3.6 预期 headroom

若该机制有效，应出现：

\[
P_C^{oracle} > P_C^{distance}+\delta
\]

同时：

\[
F_w^{oracle}\le F_w^{distance}+\epsilon.
\]

如果 constrained oracle 无法在该 regime 下超过 distance，则 SPS 机制不能作为 GNN superiority 主线。

---

# 4. 新环境机制二：RSU 与异质 Receiver Capacity

## 4.1 科学目的

让“最近 peer”不再等价于“最能快速响应的 peer”。

当前 receiver service rate 是全局常数，因此 distance 不需要区分 responder capacity。本轮引入：

\[
\mu_j
\]

作为节点异质响应能力。

## 4.2 节点类型

\[
type_j\in\{\mathrm{vehicle},\mathrm{RSU}\}.
\]

RSU headline 设置：

\[
\omega_{\mathrm{RSU}}=0.
\]

即 RSU 作为 responder/witness，不计入 macrostate participation。车辆仍是 consensus decision participants。

RSU 可返回证据 / preference，但不作为宏观 \(C,W,U\) 统计对象。

## 4.3 RSU 生成方式

采用 intersection + hotspot 组合：

1. 基础 intersection RSU：每个交叉口以概率 \(p_{int}\) 放置 RSU；
2. hotspot-near RSU：少量热点附近以更高概率放置 RSU；
3. 总 RSU 密度受上限约束，避免 RSU 过多导致 trivial solution。

参数：

```text
p_intersection_rsu ∈ {0.1, 0.25, 0.5}
p_hotspot_rsu_boost ∈ {0.25, 0.5}
max_rsu_fraction_of_nodes ∈ {0.05, 0.10, 0.15}
min_rsu_spacing_m ∈ {200, 300, 500}
rsu_roadside_offset_m ∈ {5, 10}
```

硬约束：

```text
p_intersection_rsu <= 0.5
RSU 不能密到让 nearest-RSU 成为 trivial oracle
```

## 4.4 Receiver capacity 分布

车辆 capacity：

\[
\mu_j^{veh}
=
\mu_{veh}\exp(\sigma_\mu \epsilon_j),
\quad
\epsilon_j\sim\mathcal N(0,1).
\]

RSU capacity：

\[
\mu_j^{rsu}
=
\mu_{rsu}\exp(\sigma_{rsu}\epsilon_j),
\quad
\mu_{rsu}>\mu_{veh}.
\]

推荐 sweep：

```text
mu_vehicle_base ∈ {4, 6, 8, 12} polls/window
vehicle_capacity_logstd ∈ {0.25, 0.5, 0.75}
rsu_capacity_multiplier ∈ {3, 5, 10}
rsu_capacity_logstd ∈ {0.1, 0.25}
```

Queue ratio：

\[
\rho_j(t)=\frac{\Lambda_j(t)+b_j(t)}{\mu_j}.
\]

其中 \(b_j(t)\) 是 background traffic / processing load，可先设为 0，后续扩展。

## 4.5 模型可观测特征

node features：

```text
node_type_vehicle
node_type_rsu
capacity_proxy_log
capacity_proxy_uncertainty
response_latency_ema
ack_success_ema
queue_delay_ema
local_rsu_density
```

edge features：

```text
edge_to_rsu
predicted_receiver_queue_ratio
receiver_capacity_proxy
receiver_ack_history
distance_to_nearest_rsu
```

GNN 不得直接读取真实 \(\mu_j\)，但可读取 noisy proxy：

\[
\widehat \mu_j=\mu_j\exp(\sigma_{obs}\xi_j).
\]

强启发式也必须可使用同样 proxy。

---

# 5. 新环境机制三：CSI Aging

## 5.1 科学目的

让部署时可见的 link-quality 与真实物理链路分离：

\[
\widehat{\gamma}_{ij}(t-\Delta)
\neq
\gamma_{ij}(t).
\]

这样 distance / stale link-quality 不再总是最优，GNN 需要根据 age、uncertainty 和历史统计做 robust topology。

## 5.2 Phase 1：静态 stale features

每条边有：

\[
a_{ij}^{CSI}\ge0
\]

表示 CSI age。

模型可见：

\[
\widehat{\gamma}_{ij}^{stale}
=
\gamma_{ij}(t-a_{ij}^{CSI})+\epsilon_{ij},
\]

\[
\widehat{\ell}_{ij}^{stale},
\quad
a_{ij}^{CSI},
\quad
\sigma_{\gamma}(a_{ij}^{CSI}).
\]

真实 physics 仍使用当前：

\[
\gamma_{ij}(t).
\]

默认参数：

```text
csi_age_ms ∈ {0, 50, 100, 200, 500}
csi_noise_std_db ∈ {0, 1, 2, 4}
shadow_ar_std_db ∈ {2, 4, 6}
shadow_decorrelation_s ∈ {1, 3, 8}
```

Phase 1 不启用 GRU。

## 5.3 Phase 2：Temporal-ESDGNN v2 条件

只有当：

\[
J_{\mathrm{history\ oracle}}
<
J_{\mathrm{static\ oracle}}-\delta
\]

或：

\[
P_C^{history\ oracle}
>
P_C^{static\ oracle}+\delta
\]

且差异 CI-separated 时，才进入 temporal branch。

Temporal-ESDGNN v2 不直接复用 legacy GRU，而是重新接入：

```text
macrostate_v2 metrics
dynamic-MC judge
full physics
ExperimentSpec hash
Mechanism Identifiability Contract
```

---

# 6. 新环境机制四：局部 Hotspot Density 与非均匀路网

## 6.1 科学目的

让局部密度、receiver contention、干扰和 resource conflict 出现局部非均匀结构，而不是均匀 Manhattan grid 中距离主导一切。

用户约束：

- hotspot 来源先用 intersection queue；
- 十字路口不要过于均匀；
- hotspot 数量少，不要整张图全是热点；
- Phase 1 hotspot 静态，之后再考虑动态。

## 6.2 路网生成

从规则 Manhattan grid 改成带扰动的 urban grid：

```text
block_length_m ~ LogNormal(log(100), σ_block)
intersection_offset_m ~ Uniform(-20, 20)
road_presence_probability ∈ {0.8, 0.9, 1.0}
```

保持道路拓扑仍可控，不引入复杂地图文件。

## 6.3 Intersection queue hotspot

选择少量交叉口作为 queue hotspot：

```text
num_hotspots ∈ {1, 2, 3}
hotspot_radius_m ∈ {30, 50, 80}
hotspot_vehicle_fraction ∈ {0.1, 0.2, 0.3}
queue_length_m ∈ {50, 100, 150}
```

车辆在 hotspot 附近沿道路排队，而非二维高斯随机散布。

生成约束：

```text
热点不超过全图交叉口的 10%
热点车辆不超过总车辆的 30%
热点之间距离 >= 2 * hotspot_radius
```

## 6.4 模型可观测特征

node features：

```text
local_density
intersection_distance
hotspot_score
queue_rank_along_road
local_CBR
local_candidate_degree
local_interference_degree
```

edge features：

```text
edge_crosses_hotspot
src_hotspot_score
dst_hotspot_score
edge_local_density_pair
```

强启发式必须可读取同等 local_density / hotspot proxy。

---

# 7. ESD-GNN 特征和图结构升级

## 7.1 当前问题

当前 ESD-GNN 结构上有多图聚合能力，但输入特征太窄。当前 node features 主要是：

```text
log(out_degree)
log(in_degree)
log(interference_degree)
log(region_size)
```

edge features 主要是：

```text
distance / comm_radius
LOS probability
same_region
```

这不足以学习 SPS、RSU capacity、CSI aging、hotspot 结构。

## 7.2 新增 SceneFeatures v2

新增 node features：

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

新增 edge features：

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

新增 graph：

```text
G_comm
G_int
G_region
G_resource_conflict
G_rsu_coverage
```

Phase 1 可以先把 `G_resource_conflict` / `G_rsu_coverage` 的统计量作为 node/edge features；如果 oracle headroom 明显，再增加真正的 graph aggregation channel。

## 7.3 模型结构最小改动

Phase 1 只做：

```text
ESDGNNStaticV2 = current ESDGNN + expanded feature encoder
```

保留：

- source-side aggregation；
- destination-side aggregation；
- interference aggregation；
- region pooling；
- load refinement。

新增：

- feature normalization；
- feature availability mask；
- optional resource conflict aggregation；
- optional RSU coverage pooling。

不要在 Phase 1 激活 legacy GRU。

---

# 8. Temporal-ESDGNN v2 规划

## 8.1 触发条件

只有当以下任一成立，才进入 Phase 2：

1. static GNN 无法吃到 constrained oracle headroom；
2. history-aware oracle 显著强于 static-observation oracle；
3. CSI aging 的主要信息来自多帧 ACK / CBR / resource persistence；
4. static proxy 无法稳定泛化。

## 8.2 Temporal state

每个 node / RSU 有 track id。

hidden state：

\[
h_i(t)=\mathrm{GRUCell}(x_i(t), h_i(t-1), h_{\mathcal N(i)}(t-1)).
\]

carry 规则：

- 车辆离开：drop hidden；
- 新车辆进入：initialize neutral；
- RSU：persistent hidden；
- occluded / stale vehicle：age hidden and decay。

## 8.3 Emission

旧 legacy 结果显示 emission 可以稳定 recurrent path，但新版本必须重做：

```text
emission_v2 = macrostate-compatible observable confidence / ACK / queue proxy
```

不要直接使用旧 global-risk emission 的旧指标。

候选 emission：

```text
ack_success_ema
queue_delay_ema
resource_conflict_ema
stale_csi_uncertainty
macrostate diagnostic confidence if available from previous episode
```

如果使用 consensus outcome feedback，必须 stop-gradient，并严格声明它是上一轮可观测系统输出，不是当前 truth leak。

---

# 9. Oracle-First 实验原则

每个机制都必须先通过 constrained oracle。

## 9.1 Oracle arms

对每个 regime，先评估：

```text
distance
link_quality_stale
capacity_aware
resource_aware
distance_plus_capacity
distance_plus_resource
distance_plus_csi_age
load_balanced
rsu_nearest
rsu_capacity_aware
best_heuristic_envelope
free_edge_oracle
wrong_penalized_oracle
```

所有启发式只能用 deployable proxy。

## 9.2 Headline 判据

主判据使用 matched-distance reliability：

\[
F_w^{policy,UCB}
\le
F_w^{distance,UCB}+\delta_w.
\]

同时报告绝对部署预算：

\[
F_w^{policy,UCB}
\le10^{-3}.
\]

本轮不要求严格 \(10^{-3}\) 必须通过，但必须报告。

## 9.3 胜利标准

采用 C 或 D：

### C：Pareto dominance

\[
F_w^{GNN}\le F_w^{distance}+\delta_w,
\]

\[
P_C^{GNN}\ge P_C^{distance}-\delta_C,
\]

并且至少一项：

\[
D_{99},\ E,\ F_{deadline}
\]

显著更好。

### D：Oracle-gap closure

\[
\mathrm{OGC}
=
\frac{
J_{\mathrm{distance}}-J_{\mathrm{GNN}}
}{
J_{\mathrm{distance}}-J_{\mathrm{oracle}}
+\varepsilon
}.
\]

要求：

\[
\mathrm{OGC}\ge 0.3
\]

作为初步成功；

\[
\mathrm{OGC}\ge 0.5
\]

作为强结果。

这里 \(J\) 必须是 reliability-compatible utility，而不是单纯 un-gated \(1-P_C\)。

---

# 10. 分阶段实验矩阵

## Phase 1：小/中规模机制验证

规模：

\[
N\in\{120,300,1000\}.
\]

每个机制先单独开启：

```text
NDH-SPS
NDH-RSU-CAPACITY
NDH-CSI-AGING
NDH-HOTSPOT
```

再组合：

```text
NDH-SPS+RSU
NDH-SPS+CSI
NDH-RSU+HOTSPOT
NDH-FULL-STATIC
```

每个 cell：

- 5 scene seeds；
- 3 model seeds for quick pass；
- dynamic MC trials as budget allows；
- full physics；
- macrostate_v2 only。

## Phase 2：Temporal 判定

仅当 static observations 不足时：

```text
Temporal-ESDGNN v2
history-aware oracle
static-observation oracle
static ESDGNN v2
```

比较是否历史信息带来可靠性保持 headroom。

---

# 11. 强启发式公平性

必须加入与 GNN 同信息的强启发式，否则 GNN 胜利不可信。

推荐 baselines：

```text
distance
stale_link_quality
capacity_aware
resource_aware
resource_and_capacity_hybrid
stale_csi_robust_distance
local_density_aware
rsu_nearest
rsu_capacity_aware
load_balanced
best_heuristic_envelope
free_edge_oracle
wrong_penalized_oracle
```

任何 headline claim 必须相对 `best_heuristic_envelope` 报告，而不能只比 old distance / uniform。

---

# 12. 主要风险

## R1：环境复杂但 GNN 仍学不到

应对：先 oracle-first，若 oracle 无 headroom，不训练 GNN。

## R2：GNN 赢弱 baseline，但输给新增强启发式

应对：headline 必须相对 best heuristic envelope。

## R3：RSU 过多导致 trivial nearest-RSU

应对：RSU 数量上限、RSU responder-only、RSU load/capacity有限、加 RSU overloading。

## R4：SPS 参数过强导致所有策略 collapse

应对：resource_pool / density / reselection sweep，保留 non-collapse operating band。

## R5：CSI aging 变成纯噪声

应对：给 age / uncertainty / ACK EMA，先比较 static oracle 与 history oracle。

## R6：Temporal branch 污染当前主线

应对：Phase 2 才做 Temporal-ESDGNN v2；legacy GRU 只作参考，不直接复用结果。

---

# 13. 参考资料

1. 3GPP / NR-V2X sidelink overview and Rel-16 context: Harounabadi et al., “V2X in 3GPP Standardization: NR Sidelink in Rel-16 and Beyond,” 2021.
2. 5G NR V2X tutorial: Castañeda Garcia et al., “A Tutorial on 5G NR V2X Communications,” 2021.
3. SPS and RRI in decentralized V2X: Dayal et al., “Adaptive Semi-Persistent Scheduling for Enhanced On-road Safety in Decentralized V2X Networks,” 2021.
4. Persistent collision in SPS: Jeon et al., “Reducing Message Collisions in Sensing-based Semi-Persistent Scheduling,” 2018.
5. HARQ latency / SPS interaction: Fouda et al., “HARQ Retransmissions in C-V2X: A BSM Latency Analysis,” 2023.
6. Aperiodic traffic in C-V2X sidelink: McCarthy et al., “OpenCV2X: Modelling of the V2X Cellular Sidelink and Performance Evaluation for Aperiodic Traffic,” 2021.
