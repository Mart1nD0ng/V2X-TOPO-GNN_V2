# ESD 重构最终报告 — V2X-TOPO-GNN → Effective-Sampling-Dynamics GNN

> 生成于 2026-06-24，ESD 重构完成时（全部 gate G0–G12 绿）。
> 唯一设计依据：[`EFFECTIVE_SAMPLING_DYNAMICS_TECHNICAL_SPEC.md`](EFFECTIVE_SAMPLING_DYNAMICS_TECHNICAL_SPEC.md)
> + [`ESD_GNN_ENGINEERING_PLAN.md`](ESD_GNN_ENGINEERING_PLAN.md)。
> 进度 / 决策日志：[`REDESIGN_PROGRESS.md`](REDESIGN_PROGRESS.md)（D0–D25）。
> 分支 `effective-sampling-redesign`；legacy 冻结于 tag `legacy-global-fde-v1`。
> 旧的 [`FINAL_REPORT_CN.md`](FINAL_REPORT_CN.md) 是**被取代的历史报告**（旧 11-gate global-FDE 主线）。

本报告**诚实地**呈现已验证的结果（决策 D21 = framing B）。它不夸大；其核心既包含一个清晰的正面
系统性结果，也包含一个严格的、可解释的负面结果。

---

## 1. 三个贡献（全部经验证）

### 1.1 精确行列式异质 quorum / CDQ 数学（理论贡献，G4/G5）
低秩 k-DPP 关联感知查询核 `L=BBᵀ`（`B[j]=√(qⱼ)·bⱼ`）+ 精确行列式异质 quorum 律
`P_i(m,n)=[zᵏxᵐyⁿ]det(I+zLD_g)/e_k`。两者都是**独立 ESP 抽样的关联感知推广**：当核为对角时，
CDQ 逐位（bit-for-bit）退化为 ESP，quorum 退化为 `quorum_dp`。精确到机器精度（principal-minor
归一化 ~1e-16），复杂度对 `N,E` 近线性。对抗验证修复了 7 个鲁棒性缺陷。

### 1.2 区域感知多图 ESD-GNN（系统贡献，可部署 headline = ESP 模式）
多图编码器（G_comm 源/目的 + G_int 干扰 + 区域池化）+ dynamics-in-the-loop 精化 → 仅用**可观测
结构特征**（度/区域/几何，无 truth/vote，约束 #10）。primal-dual 可靠性约束训练
（`L=ET+λE+Σμ_r(F_r−ε_r)`）使其逼近 topology-oracle 上界。**显著优于启发式、跨规模迁移、对链路噪声
与时序漂移鲁棒**（见 §3）。

### 1.3 严格负面结果：行列式多样性在区域块关联 V2X 中无可靠性收益（D17/D18/D20/D22）
在已实现的 §6.2 区域块关联证据模型中，CDQ 的行列式**多样性**相对区域感知 ESP **没有可靠性优势**，
且在**完整物理链下反而有害**（变慢、选到链路更差的对等点）。三个独立、相互印证的结构原因：
1. 均场解析 episode 对关联**不敏感**——quorum 只用每个对等点的边缘 `u_j` 且把选择视为独立，故多样
   与冗余选择给出相同的解析 `h`（无多样性梯度）；
2. 唯一的关联感知量 `k_eff`（`effective_sample_size`）**不可微**（`float(w@R@w)`），§5.8 aux loss 需重写
   才能训练多样性；
3. 环境关联是区域块/近可交换的（跨区相关 0，同区相关只依赖边缘错误率）——唯一结构是区域级，区域感知
   ESP 已充分利用；且 ESP 用自由权重可确定性选出任意最优子集。
**这是该领域一个诚实、可解释、可发表的负面结论**，而非空结果。完整的关联感知 CDQ 数学仍是有效推广
（ESP 是其对角特例）；是否在更细关联（§6.3 共因传感器误差）下有收益，是被记录的 future work（选项 A，
用户选择 framing B 时暂缓）。

---

## 2. 验证层级（诚实角色，spec §8）

| 层 | 角色 | 关系 |
|----|------|------|
| 精确小-N 联合马氏链（`exact_small_n`） | ground truth（固定链路下的精确联合终态） | dynamic MC 在其 CI 内 |
| **独立 dynamic MC**（`run_dynamic_mc`） | **无偏裁判 / headline 评估器** | 不从解析终态边缘重采样（约束 #8）；CDQ 与 ESP 两种查询律都支持 |
| 解析 episode（`run_consensus_episode`） | 可微 ranking surrogate（强耦合下乐观、但保序） | 训练用；大 N 下 `2^G` 不可解 → MC 是唯一裁判 |

关键诚实结论：解析是**可微的排序代理**，MC 是**校准裁判**（spec §8.2/8.3）。

---

## 3. 关键量化结果

* **可行性（G8，viability）**：well-mixed 完美链路递推（MC 验证）→ 在 N≤10000、correct-majority≥0.6
  下 FEASIBLE；50/50 正确地 infeasible。无 stop-condition #1。
* **拓扑 oracle 杠杆（D12，viability #2）**：直接 per-scene 边权优化 oracle 显著优于启发式
  （MC F_wrong 0.003 vs uniform 0.117，3 seeds CI 分离）；distance 启发式比 uniform 更差（§9.1 张力）。
* **训练逼近 oracle（G9b）**：ESD-GNN F_wrong 0.015→0.0012，仅用可观测特征。
* **Headline（G11，paired CRN + Bonferroni）**：
  - ideal-link（D20，3 seeds/16 scenes/N=336）：`esd_gnn_esp` 比 uniform 低 **−0.088 [−0.102,−0.074]**（显著）；
    **CDQ vs ESP +0.036 [+0.021,+0.056]**（显著 → ESP 更好）。训练 N=48 → 部署 N=336（迁移成立）。
  - full-physics（D22，link_override=None，N=96）：`esd_gnn_esp` F_wrong **0.021**，比 uniform 低
    −0.091 [−0.115,−0.067]（显著）；**CDQ 崩塌到 ~uniform**（0.110，不显著）且**延迟 3×**；CDQ vs ESP
    +0.089（显著）。
* **大规模复杂度（G10）**：完整链路（图+GNN+MC）N=96→9520 **近线性**，平均度有界（11→14，E=O(N)，无
  degree cap），无 N×N；解析 `2^G` 在大 N 不可解 → 确认 MC 是唯一大-N 裁判。
* **时序鲁棒（G12）**：无记忆 ESD-GNN(ESP) 在漂移 {0→0.5+20%churn} 下 F_wrong **恒定 ~0.004**，而 uniform
  升到 0.16（重churn 下差距扩大到 38×）——每拓扑重算可观测特征实现时序泛化，无需隐状态记忆模型（§9.7 暂缓）。

---

## 4. 硬约束合规

| # | 约束 | 如何满足 |
|---|------|---------|
| 1 | 可靠性是硬约束 | primal-dual 把 F_disagree/F_wrong/F_deadline 作为约束，不与时延/能耗交换 |
| 2 | headline 只优化 query topology | power/blocklength 固定；headline 比较查询策略；资源控制未进入 headline |
| 3 | 训练=部署同一 policy | 同一 ESD-GNN（可观测特征）；迁移验证 N=48→336、跨漂移 |
| 4 | 无 degree cap/截断 | G0 gate（token 扫描 + 稠密图行为检查 out-deg==N−1）；几何决定稀疏度 |
| 5 | N=100~10000 | G10 近线性到 9520；小 N 仅用于 exactness |
| 6 | 单一 canonical evaluator | G0 gate：headline/oracle/train/scaling 全部经 run_consensus_episode/run_dynamic_mc |
| 7 | 完整物理链、无 tau_proxy | round_physics 逐轮耦合负载/SINR/FBL/HARQ/排队；G0 gate 扫描无数值 tau_proxy |
| 8 | 独立 dynamic MC | 前向仿真，不从解析边缘抽样（sentinel 测试） |
| 9 | 真实 binary Snowball | `binary_snowball` 置信累积有限视界链 |
| 10 | policy 不读 truth/vote | G0 gate 扫描策略模块无 truth/vote 引用 |
| 11 | 无 N×N、近线性 | 全程稀疏边表；G10 实测 N=9520 无 OOM |
| 12 | exact claim 注明随机模型边界 | exact_small_n / CDQ / quorum 均注明边界 |
| 13 | 未进入 canonical path 的机制不入结果 | G0 gate；legacy `model.py`（tau_proxy）被任何 redesign 模块导入 = 否 |
| 14 | 多 seed、能力匹配 baseline、dynamic-MC | headline 多 model-seed + uniform/distance baseline + MC 裁判 |

---

## 5. 诚实的局限与 future work

* **G11 严格度**：dev-loop 在 10 分钟工具上限内运行（ideal-link 3 seeds/16 scenes；full-physics 2 seeds/
  6 scenes）。**publication-grade 运行**（≥5 model seeds、≥30 scene seeds、full physics、高 trials）是同一脚本
  `scripts/analysis/headline_comparison.py` 仅调大常量的**离线多小时作业**——结论方向已在两种链路模式下显著且一致。
* **CDQ future work（选项 A）**：在更细于区域的关联（§6.3 共因传感器组，使同区对等点不可交换）下，配合
  可微 `k_eff`（§5.8 aux loss）重测 CDQ；当前证据下 framing B 暂缓。
* **时序记忆模型（§9.7）**：contractive GRU/SSM 记忆是被记录的扩展；静态策略已时序鲁棒，故按 plan §15
  仅在能证明收益时才加入。
* **legacy 图脚本**：携带独立 evaluator 的旧脚本（`baseline_comparison.py` 等）应隔离，避免与 redesign
  canonical path 混淆（非 gating，已记录）。

---

## 6. 可复现性

* 门槛：ESD G0–G12 全绿。测试：`tests/{environment,models,optimization,protocol,sampling,validation}`
  + `tests/test_g0_canonical_hygiene.py`（136 redesign+hygiene 测试通过）。
* 结果清单（`result/` 为可复现产物，由脚本重生成）：`scaling/scaling.json`（G10）、
  `headline/headline.json`（G11）、`topology_oracle_ceiling/ceiling.json`（D12）。
* 决策日志 D0–D25 在 `REDESIGN_PROGRESS.md`；每个 slice 一个原子 commit，分支 `effective-sampling-redesign`。
