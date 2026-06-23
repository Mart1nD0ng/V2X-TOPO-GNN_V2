# V2X-TOPO-GNN：有效抽样动力学重构工程计划书

**依赖技术规范**：`EFFECTIVE_SAMPLING_DYNAMICS_TECHNICAL_SPEC.md`  
**目标**：将当前项目从 preference-conditioned \(F/D/E\) proof-of-concept 重构为可靠性约束下的 Avalanche/Snowball 有效抽样拓扑控制系统。  
**原则**：先闭合协议和环境，再实现新模型；先证明问题存在优化空间，再追求模型性能；论文最后修改。

---

## 1. 分支和冻结策略

1. 给当前 `main` 打不可变 tag，例如：
   - `legacy-global-fde-v1`
2. 新建分支：
   - `effective-sampling-redesign`
3. 当前结果、checkpoint、figure 和 gate evidence 全部归档，不得作为新主线 headline。
4. `src/mainline` 暂不删除；以迁移方式逐模块替换。
5. 所有新结果必须绑定：
   - commit；
   - config hash；
   - checkpoint hash；
   - data/scene manifest；
   - runtime mechanism trace。

---

## 2. 建议的新目录

```text
src/
  protocol/
    binary_snowball.py
    protocol_state.py
    exact_small_n.py

  environment/
    urban_scene.py
    evidence_model.py
    candidate_graph.py
    interference_graph.py
    round_physics.py
    finite_blocklength.py
    canonical_episode.py

  sampling/
    esp_query.py
    dpp_query.py
    determinantal_quorum.py
    effective_dynamics.py

  models/
    baseline_gnn.py
    esd_gnn.py
    temporal_esd_gnn.py

  optimization/
    constrained_objective.py
    primal_dual.py

  validation/
    dynamic_mc.py
    rare_event.py
    mechanism_trace.py
    result_manifest.py
```

旧代码迁入：

```text
legacy/global_fde_v1/
```

---

## 3. Phase 0：问题规范和协议选择

### 任务

1. 写 `docs/PROTOCOL_SEMANTICS.md`：
   - 选定 true binary Snowball 或 Snowflake-like；
   - 给出逐步伪代码；
   - 定义 safety、validity、deadline finality；
   - 定义 threat model；
   - 定义 eligible set。
2. 写 `docs/OPTIMIZATION_CONTRACT.md`：
   - 可靠性阈值；
   - tail latency；
   - energy；
   - service profiles。
3. 删除新主线中的平权 \(F/D/E\) preference 语义。

### Exit

- 协议伪代码经 reference trace 验证；
- 所有指标无语义重叠；
- 目标阈值明确；
- perfect-link feasibility gate 可实现。

### Stop

若团队不决定 true Snowball 或 Snowflake-like，停止后续开发并上报。

---

## 4. Phase 1：Canonical execution path

### 任务

实现唯一入口：

```python
run_consensus_episode(
    scene,
    query_policy,
    protocol_cfg,
    phy_cfg,
    evidence_cfg,
    mode="analytic" | "dynamic_mc",
    return_trace=True,
)
```

所有 model、baseline、ablation、figure、gate 必须调用它。

### Mechanism trace

返回：

```json
{
  "protocol": "binary_snowball",
  "dynamic_transient_load": true,
  "interference_graph": true,
  "mode2_collision": true,
  "half_duplex": true,
  "request_response": true,
  "queueing": true,
  "finite_harq": true,
  "fbl_dispersion": true,
  "correlated_evidence": true,
  "query_policy": "esp|dpp",
  "scenario_count": 16
}
```

### 测试

- sentinel activation；
- 未消费配置字段报错；
- model/baseline/figure 使用同一入口；
- 禁止第二 evaluator。

### Exit

G0 全绿。

---

## 5. Phase 2：真实协议与独立 dynamic MC

### 任务

1. 实现 true binary Snowball finite-horizon state；
2. 保留当前 streak automaton 作为 Snowflake baseline；
3. 实现逐 trial dynamic MC：
   - 真实 query subset；
   - 真实 peer current state；
   - 真实 request/response；
   - 真实 resource/fading；
   - 真实 confidence counters。
4. 实现 \(N\le8\) exact joint reference。

### 测试

- 手工 trace；
- exact chain vs dynamic MC；
- MC 不能 import analytic terminal marginals；
- common random numbers；
- confidence interval coverage。

### Exit

G1、G6 全绿。

---

## 6. Phase 3：相关证据环境

### 任务

1. Manhattan road generator；
2. road segment / intersection / visibility region；
3. global truth \(Y^\star\)；
4. region-level shared error；
5. node-level independent error；
6. temporal region transitions；
7. observable correlation features。

### 场景集

- iid evidence；
- one biased region；
- two opposing regions；
- weak cut；
- hub congestion；
- sudden blockage；
- mobility/churn。

### 测试

- empirical pairwise correlation 与理论匹配；
- zero-correlation control；
- query policy 不读取 truth/vote；
- region split 可复现。

### Exit

G2 全绿。

---

## 7. Phase 4：完整逐轮物理链

### 任务

1. 构建 \(G_{\mathrm{comm}}\) 与 \(G_{\mathrm{int}}\)；
2. 动态 active/transient load；
3. all-transmitter interference；
4. Mode-2 resource overlap；
5. request/response 分离；
6. half-duplex；
7. queueing；
8. finite HARQ；
9. FBL with explicit dispersion；
10. wall-clock round duration 和 energy。

### 代码迁移

- 将 `load_coupled_link_reliability` 的有效逻辑迁入 `round_physics.py`；
- 删除 `tau_proxy`；
- 删除未使用的 queueing helper 或正式接入；
- 禁止 headline 直接调用旧 `evaluate_controls`。

### 测试

- 每个机制的单调性；
- cross-destination interference；
- request/response asymmetry；
- active mass 随 finalization 下降；
- mechanism sentinel；
- gradient activation。

### Exit

G3 全绿。

---

## 8. Phase 5：协议可行性扫描

### 任务

对：

\[
N=100,300,1000,3000,10000
\]

扫描：

\[
(k,\alpha,\beta,R_{\max}).
\]

分三层：

1. perfect link + global representative sampling；
2. full PHY + uniform query；
3. full PHY + direct per-scene topology optimizer。

### 输出

- safety floor；
- validity floor；
- deadline floor；
- minimum feasible parameter set；
- energy/latency cost。

### Exit

至少存在一个目标规模和 service profile 满足可靠性约束。

### Stop

若 perfect-link floor 不可行，禁止训练 GNN；上报需要修改协议或采用 committee/hierarchy。

---

## 9. Phase 6：有效抽样动力学层

### 任务

实现：

- response-conditioned distribution；
- progress \(g_i\)；
- drift \(\Delta_i\)；
- progress/drift per second；
- ESS；
- region mixing；
- query overlap；
- receiver load。

### 测试场景

- 对称独立丢包：仅 progress 下降；
- opinion-correlated link loss：drift 改变；
- weak cut：mixing 下降；
- hub：load 上升；
- redundant peers：ESS 下降。

### Exit

G7 全绿。

---

## 10. Phase 7：建立 topology optimization ceiling

在开发 ESD-GNN 前，先证明问题具有 topology lever。

### Baselines

1. uniform；
2. distance/SINR；
3. load-balanced；
4. bridge/conductance-aware；
5. current ESP；
6. direct per-scene edge-logit optimizer；
7. evolutionary optimizer。

### 判据

- direct optimizer 必须显著优于 heuristics；
- 优势必须在 dynamic MC 下保留；
- 满足 reliability constraints；
- 若 oracle 无优势，停止模型开发并重新审查场景/协议。

---

## 11. Phase 8：CDQ 数学层

### 任务

1. 低秩 k-DPP normalization；
2. inclusion marginals；
3. exact sampler；
4. determinant generating polynomial；
5. heterogeneous quorum coefficients；
6. autograd；
7. degree-bucketed/chunked implementation。

### 验收

- subset distribution vs brute force `<1e-10`；
- quorum distribution vs brute force `<1e-10`；
- gradient relative error `<1e-4`；
- diagonal kernel 恢复当前 ESP；
- runtime 对 \(E\) 近线性。

### Stop

若 exact determinant coefficient 的成本无法达到目标，停止并上报。不得静默改成 sampled training 后仍称 exact。

---

## 12. Phase 9：ESD-GNN

### 最小模型

- communication encoder；
- incoming-load encoder；
- evidence-correlation encoder；
- region pooling；
- quality head；
- diversity embedding head；
- CDQ layer；
- 2 次 dynamics refinement。

### Headline 输出

仅 query policy。power/blocklength 固定。

### 训练

- primal-dual constrained objective；
- curriculum：iid evidence → region correlation → weak cut → dynamic PHY；
- \(N=100,300,1000\) 混合训练；
- multiple model seeds；
- fixed validation protocol。

### Exit

G9 全绿。

---

## 13. Phase 10：大规模性能

### Headline 节点规模

\[
100,300,1000,3000
\]

Scale：

\[
5000,10000.
\]

### 评价

- feasibility rate；
- \(F_{\mathrm{disagree}}\)；
- \(F_{\mathrm{wrong}}\)；
- deadline miss；
- \(D_{50},D_{95},D_{99}\)；
- CVaR；
- energy；
- progress/drift/ESS/mixing diagnostics；
- runtime/memory。

### 统计

- 至少 5 model-training seeds；
- 至少 30 independent scene/physics seeds；
- paired common-random-number MC；
- confidence intervals；
- multiple-comparison correction。

### Exit

G10、G11 全绿。

---

## 14. Phase 11：Temporal extension

只有静态主线通过后执行。

### 比较

- no memory；
- scalar emission；
- GRU；
- contractive SSM；
- temporal ESD-GNN；
- shuffled memory；
- hidden normalization。

### 场景

- mobility；
- churn；
- persistent shadow；
- sudden weak cut；
- region-bias change。

### 指标

- recovery time；
- deadline miss；
- safety/validity；
- energy；
- variance across initialization；
- hidden-state norm。

### Exit

G12 全绿。若 scalar emission 无收益，从贡献中删除。

---

## 15. 新 gate catalogue

```text
G0  canonical execution closure
G1  protocol semantics
G2  correlated evidence environment
G3  round-coupled full physics
G4  CDQ subset exactness
G5  determinantal quorum exactness
G6  independent dynamic MC
G7  effective sampling diagnostics
G8  protocol feasibility
G9  model mechanism and ablations
G10 large-N complexity/performance
G11 reliability-constrained superiority
G12 temporal robustness
```

旧 G1–G11 结果作为 legacy evidence，不得直接映射为新 gate 通过。

---

## 16. 开发 LOOP

每次迭代只完成一个最小 gate slice。

1. **Read**
   - 技术规范；
   - 本计划；
   - 当前 gate 状态；
   - 相关代码。
2. **Specify**
   - 数学 contract；
   - exactness boundary；
   - complexity；
   - acceptance test。
3. **Test first**
   - reference；
   - invariant；
   - sentinel；
   - adversarial case。
4. **Implement**
   - 最小完整实现；
   - 只走 canonical path；
   - 无 silent fallback。
5. **Verify**
   - unit；
   - integration；
   - dynamic MC；
   - full gate；
   - runtime/memory。
6. **Adversarial audit**
   - 机制是否真的被调用；
   - 是否偷换协议；
   - 是否使用 truth；
   - 是否又出现小 \(N\)；
   - 是否 train/deploy mismatch。
7. **Document**
   - 更新 gate evidence；
   - 更新 decision log；
   - 原子 commit。
8. **Continue**
   - 进入下一个 slice，不等待非必要确认。

---

## 17. 严禁捷径

1. 用 \(N=8,10,12\) 作为性能 headline；
2. 用 complete graph 代替目标 V2X 稀疏图；
3. 继续用 `tau_proxy`；
4. 完整物理机制仅存在于 test；
5. 用解析 marginals 再采样冒充独立 MC；
6. 把 Snowflake 称为 Snowball；
7. 让 policy 读取 truth 或 peer vote；
8. 用 power/blocklength heads 掩盖 topology weakness；
9. 用 reliability weighted sum 允许不安全点；
10. 用 sampled query training 替代 CDQ exactness后仍宣称 exact；
11. 创建 \(N\times N\) dense tensor；
12. 使用未过滤 dominated points 的 coverage；
13. 单个训练 seed 做显著性；
14. 在 gate 未通过前更新论文 headline；
15. 配置字段未被消费却不报错；
16. figure script 自己重写 evaluator；
17. 通过降低阈值或缩小场景让 gate 变绿。

---

## 18. 完成定义

项目重构完成要求：

- 协议语义一致；
- canonical full physical path 唯一；
- dynamic MC 独立；
- perfect-link feasibility 通过；
- topology oracle 有显著空间；
- CDQ 数学与梯度通过；
- ESD-GNN 机制消融成立；
- \(N=100\sim10000\) 结果完整；
- 可靠性约束满足；
- tail latency/energy 优于能力匹配基线；
- runtime/memory 近线性；
- 所有结果可一键复现；
- 之后才重写论文。
