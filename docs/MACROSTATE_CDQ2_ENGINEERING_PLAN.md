# V2X-TOPO-GNN：宏观首达目标与 CDQ 2.0 工程实施计划

## 0. 目标

本计划指导 `effective-sampling-redesign` 的下一阶段重构。执行顺序固定为：

\[
\boxed{
\text{冻结现状}
\rightarrow
\text{P0 物理与时延修正}
\rightarrow
\text{宏观首达目标}
\rightarrow
\text{独立 MC}
\rightarrow
\text{场景与相关性}
\rightarrow
\text{ESP 基准}
\rightarrow
\text{CDQ 2.0}
\rightarrow
\text{规模与泛化}
}
\]

论文继续冻结。当前论文仍基于旧的节点并集 global failure 和平权 \(F/D/E\) 叙事，不能用于描述新系统。

---

# 1. Phase 0：冻结、分支和结果隔离

## 工作

1. 给当前状态打 tag：
   - `pre-macrostate-cdq2`
2. 保存：
   - 当前 checkpoint；
   - 当前 ESP/CDQ 结果；
   - physics config；
   - gate evidence；
   - commit SHA。
3. 新建工作分支：
   - `macrostate-cdq2-redesign`
4. 当前结果只作为 legacy comparison，不得进入新 headline。

## Exit

- 历史结果可一键复现；
- 新分支不覆盖旧证据。

---

# 2. Phase 1：统一协议与 service profile

## 工作

更新：

- `docs/PROTOCOL_SEMANTICS.md`
- `docs/OPTIMIZATION_CONTRACT.md`

实现统一：

```text
ConsensusServiceProfile
  participation_weight_rule

  poll_window_ms
  k
  alpha
  beta
  max_poll_epochs

  correct_basin_mass
  split_basin_mass

  max_wrong_basin_probability
  max_split_basin_probability
  max_deadline_miss_probability

  latency_quantile
  energy_budget
```

明确：

- \(k\) 个 parallel unicast request-response polls；
- finalized nodes 不再发起 polls，但继续响应；
- epoch-end update；
- deadline 和 epoch 数的转换；
- uniform 与 application-weighted participation measure。

## 测试

- 配置阈值满足 basin disjointness；
- \(\sum_i\omega_i=1\)；
- policy 无法修改 \(\omega\)；
- profile hash 进入 checkpoint manifest。

## Exit

协议、范围、阈值和 deadline 只有一套定义。

---

# 3. Phase 2：修复 P0 物理与时延问题组

## P0-A：解析时延 off-by-one

替换：

```python
not_done = 1 - S_all_t[1:]
```

为基于 epoch 开始状态的 survival sum：

\[
E[T]
=
\sum_{r=0}^{R_d-1}
\Delta_{\mathrm{poll}}^{(r+1)}
P(T>r).
\]

### 必测

- \(\beta=1\)；
- 第一轮必定完成；
- 返回一轮 polling window，而不是 0。

## P0-B：并行 unicast 语义

实现：

\[
a_{ij}^{\mathrm{req}}=u_i\pi_{ij}.
\]

source activity：

\[
\sum_j a_{ij}^{\mathrm{req}}=ku_i.
\]

response 只在 request 成功后发生。

## P0-C：source/destination ownership

建立显式函数：

```python
scatter_source(...)
scatter_destination(...)
```

禁止复用无语义 scatter helper。

新增手工 star/asymmetric graph reference。

## P0-D：collision self-exclusion

实现：

\[
L_{j,-ij}=L_j-a_{ij}.
\]

单边、单请求情况下 collision 必须为 0。

## P0-E：poll-window success

将：

- request FBL；
- response FBL；
- HARQ；
- collision；
- half-duplex；
- queue；
- timeout；

统一成：

\[
\ell_{ij}(\Delta_{\mathrm{poll}}).
\]

不再把平均 edge service time 与 consensus epochs 任意相加。

## P0-F：dynamic MC 对齐

analytic 与 MC 使用相同：

- unicast semantics；
- energy ownership；
- poll window；
- finalized responder rule；
- collision rule。

## Exit

新增 `G-P0-PHYSICS` gate 全绿。

---

# 4. Phase 3：Canonical path 和配置闭环

## 工作

建立唯一入口：

```python
run_consensus_episode(
    scene,
    service_profile,
    query_policy,
    protocol,
    physics,
    evidence,
    mode="analytic" | "dynamic_mc",
    return_trace=True,
)
```

## 强制字段

```json
{
  "parallel_unicast": true,
  "poll_window_ms": 10,
  "source_destination_accounting": true,
  "collision_self_exclusion": true,
  "request_response": true,
  "dynamic_transient_load": true,
  "interference_graph": true,
  "half_duplex": true,
  "queueing": true,
  "finite_harq": true,
  "fbl_dispersion": true
}
```

## Train/eval 一致性

新增：

```text
ExperimentSpec
  protocol_hash
  service_profile_hash
  physics_hash
  evidence_hash
  scene_distribution_hash
  query_law
  allowed_ood_axes
```

规则：

- headline 训练和评价 physics hash 必须相同；
- ideal-link 只能进入显式 `ideal_link_ablation`；
- checkpoint 加载时强制 compatibility check；
- 禁止裸 `link_override` 进入 headline；
- unused config field 报错。

## Exit

`G-CANONICAL-CLOSURE` 全绿。

---

# 5. Phase 4：宏观 basin evaluator

## 工作

实现：

- participation measure；
- \(C_r,W_r,U_r\)；
- correct/wrong/split basin；
- first-hitting outcome；
- pairwise disagreement intensity；
- region disagreement。

建议模块：

```text
src/metrics/
  participation.py
  macrostate.py
  basins.py
  first_hitting.py
```

## Dynamic MC

每个 trial 记录：

```text
outcome = correct | wrong | split | deadline
tau
C_path
W_path
U_path
```

## Small-N exact

对 \(N\le8\)：

- 枚举联合 Snowball state；
- 计算 basin first-hitting probabilities；
- 与 dynamic MC 比较。

## 必测

- 四类 outcome 概率和为 1；
- basin 互斥；
- uniform weights 下复制相同 population 不产生 \(1-(1-p)^N\) 式机械变化；
- strict disagreement 仍可单独报告。

## Exit

`G-MACROSTATE` 全绿。

---

# 6. Phase 5：优化目标重构

## 工作

删除 headline 的平权 preference-conditioned \(F/D/E\) 目标。

实现：

\[
\min
\operatorname{CVaR}_{q}(T_{\mathrm{confirm}}\mid O=C)
+\lambda_EE
\]

subject to：

\[
F_{\mathrm{wrong}}\le\epsilon_w,
\quad
F_{\mathrm{split}}\le\epsilon_s,
\quad
F_{\mathrm{deadline}}\le\epsilon_d.
\]

使用 primal-dual。

## 防止 selection bias

同时报告：

- conditional correct latency；
- deadline-capped unconditional latency；
- deadline miss；
- energy per attempted instance；
- energy per successful instance。

## Exit

- dual variables 能自动响应 constraint violation；
- 不可行 policy 不进入 D/E Pareto；
- 不再通过降低可靠性换速度。

---

# 7. Phase 6：场景与相关结构重构

## 工作

修复：

- 固定 biased region 0；
- region-preserving pseudo-churn；
- 过于规则的 block correlation。

新增重叠共因模型：

\[
O_i
=
Y^\star
\oplus
B_{g(i)}^{\mathrm{road}}
\oplus
B_{s(i)}^{\mathrm{sensor}}
\oplus
B_{m(i)}^{\mathrm{map}}
\oplus
B_{q(i,t)}^{\mathrm{temporal}}
\oplus
E_i.
\]

## 场景矩阵

1. iid evidence；
2. single region bias；
3. randomized biased region；
4. multiple biased regions；
5. overlapping sensor-source correlation；
6. same marginal / different covariance；
7. weak cut；
8. true region handoff；
9. persistent common cause；
10. sudden common-cause change。

## Mechanism Contract

每个环境机制和模型组件必须填写：

- structure existence；
- deployment observability；
- gradient reachability；
- causal controls；
- mainline consistency。

## Exit

`G-CORRELATED-ENV` 与 contract audit 全绿。

---

# 8. Phase 7：建立新的 ESP full-physics 基准

在修改 CDQ 前先固定可信 ESP baseline。

## 训练

- full physics；
- canonical path；
- 宏观 basin objective；
- 至少 5 model seeds；
- 规模混合训练；
- no power/blocklength heads。

## 评价

- dynamic MC；
- 30+ scenes；
- adequate trials 或 rare-event MC；
- wrong/split/deadline；
- tail latency；
- energy；
- effective sampling diagnostics。

## 规模

- train：\(N=100,300,1000\)；
- test：\(N=100,300,1000,3000\)；
- scale：\(N=5000,10000\)。

## Exit

得到 P0 修复后的可信 ESP baseline。

---

# 9. Phase 8：CDQ 2.0 数学实现

## 核

\[
L
=
D^{1/2}(I+\eta ZZ^\top)D^{1/2}.
\]

## 工作

1. unit-normalized diversity embeddings；
2. adaptive \(\eta\)；
3. ESP exact fallback at \(\eta=0\)；
4. exact \(k\)-DPP normalizer；
5. exact sampler；
6. pairwise inclusion；
7. determinant quorum generating law；
8. autograd；
9. sparse degree buckets；
10. numerical stabilization。

## 必测

- \(\eta=0\) 与 ESP probability/gradient 一致 `<1e-10`；
- subset probability vs brute force `<1e-10`；
- quorum law vs brute force `<1e-10`；
- gradient relative error `<1e-4`；
- rank/degree stress；
- runtime 对 \(E\) 近线性。

## Exit

`G-CDQ2-MATH` 全绿。

---

# 10. Phase 9：相关性梯度

## 工作

实现 tensorized：

\[
\mathcal L_{\mathrm{corr}}
=
\sum_i\sum_{j<l}
\pi_{i,jl}^{(2)}R_{jl}.
\]

禁止 `.item()`、`float()`、detach。

## 测试

- nonzero gradient；
- gradient direction；
- zero-correlation 时 loss 为 0；
- correlation 增大时选择概率下降；
- shuffle controls。

## Exit

`G-CDQ2-GRADIENT` 全绿。

---

# 11. Phase 10：CDQ 公平验证

## 因子矩阵

| 轴 | 水平 |
|---|---|
| Environment | iid / region block / overlapping correlation |
| Physics | ideal ablation / full physics |
| Policy | ESP / old low-rank CDQ / CDQ 2.0 |
| Correlation objective | off / on |
| \(\eta\) | 0 / learned / fixed sweep |

所有 headline 必须 full-physics train + full-physics eval。

## 报告

- wrong basin；
- split basin；
- deadline miss；
- tail latency；
- energy；
- selected-link success；
- selected distance；
- pairwise evidence correlation；
- \(k_{\mathrm{eff}}\)；
- progress/drift；
- \(\eta\) distribution。

## 解释规则

- 若 \(\eta\to0\)，说明环境无需 diversity；
- 若 CDQ 只在 overlapping correlation 胜出，说明机制有效且 scoped；
- 若修正后仍不胜 ESP，接受负面结果；
- 不得用 ideal-trained checkpoint 评价 full physics。

---

# 12. Phase 11：规模与泛化

## 规模泛化

同时做：

### 固定协议

观察宏观 wrong/split/deadline 随 \(N\) 的变化。

### 固定 service profile

按预注册规则调整协议参数，维持相同服务目标。

## OOD 轴

- node count；
- density；
- road geometry；
- interference；
- evidence covariance；
- sensor-source composition；
- protocol；
- mobility/handoff。

每次只改变一个主轴。

## 统计

- 5+ model seeds；
- 30+ scenes；
- nested bootstrap；
- rare-event MC；
- explicit shared randomness，而不是仅重置 seed。

## Exit

`G-SCALE-GENERALIZATION` 全绿。

---

# 13. Phase 12：Temporal extension

只有静态主线完成后执行。

重新设计 temporal memory，使其面向：

- effective sampling drift；
- region correlation；
- load persistence；
- weak-cut persistence。

旧 global-risk emission 不自动进入新主线。

必须遵守 Mechanism Identifiability Contract。

---

# 14. Gate 清单

```text
G-P0-PHYSICS
G-CANONICAL-CLOSURE
G-MACROSTATE
G-CONSTRAINED-OBJECTIVE
G-CORRELATED-ENV
G-ESP-BASELINE
G-CDQ2-MATH
G-CDQ2-GRADIENT
G-CDQ2-EVALUATION
G-SCALE-GENERALIZATION
G-TEMPORAL
```

---

# 15. 每轮 LOOP

1. 阅读技术规范、工程计划、Mechanism Contract；
2. 选择一个最小未通过 gate slice；
3. 写数学/API contract 与 exactness boundary；
4. 先写失败测试；
5. 实现并接入 canonical path；
6. 运行 unit、integration、dynamic MC、scaling；
7. 执行 Mechanism Contract 审计；
8. 更新 evidence、manifest、decision log；
9. 原子 commit；
10. 继续下一 slice。

---

# 16. 严禁捷径

1. 继续使用 \(1-\prod_i c_i\) 作为跨规模主失败；
2. 使用 hard eligible set 掩盖范围问题；
3. 继续平权优化 \(F/D/E\)；
4. 用 ideal links 训练 full-physics headline；
5. 只修测试不修 canonical path；
6. 混淆 source/destination；
7. collision 不排除 desired transmission；
8. figure script 重写 evaluator；
9. 相关结构只存在于 simulator truth、模型不可观察；
10. correlation loss 被 detach；
11. 用相关与 marginal quality 同时变化的场景证明 CDQ；
12. 用低秩限制不同的模型做“不公平 CDQ vs ESP”；
13. 单训练 seed；
14. 小 complete graph headline；
15. 降低门槛让 gate 变绿；
16. 在新 gate 完成前修改论文 headline。

---

# 17. 停止与上报条件

仅在以下情况停止：

1. basin 定义或参与测度需要用户决策；
2. P0 物理语义存在无法兼容的协议解释；
3. exact CDQ 2.0 quorum 无法满足复杂度；
4. analytic surrogate 与 dynamic MC 排序系统性冲突；
5. full-physics ESP 没有 topology lever；
6. 可靠性约束在 perfect physics 下不可行；
7. 全部 gates 通过。

上报必须包含：

```text
STATUS:
CURRENT GATE:
BLOCKER OR COMPLETE:

EVIDENCE:
- tests
- equations/counterexample
- dynamic-MC results
- runtime/memory
- changed files/commit

WHY IT CANNOT BE SAFELY BYPASSED:

MINIMUM USER DECISION:

RECOMMENDED NEXT ACTION:
```
