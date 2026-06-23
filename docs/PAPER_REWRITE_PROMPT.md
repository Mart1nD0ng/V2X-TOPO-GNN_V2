# Prompt: 指导 claudeprism 重写论文（数学 / 结论 / 实验 / 数据表）

> 用途：把这份文档**整体作为 prompt** 交给 claudeprism，让它据此重写 `paper/main.tex` 的数学公式、
> 旧结论、实验设计 / 结果 / 分析与数据表格。
> 性质：**指导性 prompt** —— 给定关键事实、公式与数据（必须照用、不得编造），但**叙述结构、措辞、
> 强调点、图表标题、章节顺序由 claudeprism 自由决定**。
> 论文语言：`paper/main.tex` 是英文 LaTeX，输出英文；本 prompt 用中文给指导，公式 / 数据 / 术语保持其规范形式。
> 数据来源（均可复现、固定种子，**禁止手抄/篡改**）：`docs/gate_evidence/latest.json`、
> `docs/gate_evidence/g11_baseline.json`、`scripts/analysis/baseline_comparison.py`、`profile_scaling.py`、
> `paper_tables.py`。背景见 `docs/FINAL_REPORT_CN.md`。

---

## 0. 角色与总目标

你是资深通信/ML 论文作者。任务：把这篇 V2X 拓扑优化共识论文的数学主线、实验与结论，从**被取代的旧推导**
（mean-field 节点均值 `F`、iid beta-tail quorum、logistic-BLER、固定 degree）整体重写为**统一的新主线**
（共享有限混合全局 `F` + 生成函数 quorum DP + finite-blocklength `ℓ` + 偏好条件化 GNN + global-risk emission），
并据**已验证的实验数据**重写实验与结论。要求：数学严谨、数字全部来自下方提供的可复现结果、**如实陈述边界**、
不夸大。叙述自由发挥。

### 铁律（不可违反）
1. **只用本文档给出的数字**；任何表格/正文数值必须与下方一致（或注明从 `docs/gate_evidence/*.json` 拉取）。不得编造。
2. **数学公式照用 §2 给定形式**（可改记号/排版，不得改语义）。
3. **删除 §3 列出的旧结论**，全部按新数据重算重写。
4. **必须如实写出 §6 的局限**（这是诚实性要求，不是可选项）。
5. 旧推导只能作为**显式标注的历史/对照附录**出现，绝不作为 headline 定理。

---

## 1. 一句话 headline（供你改写，不要逐字照搬）

> 单个偏好条件化拓扑 GNN，在统一、自洽、端到端可微的网络共识数学主线上，用一个 checkpoint 覆盖
> F（全局共识失败概率）/ D（完成时延）/ E（全网能耗）的 Pareto 前沿；在留出场景上，其前沿在
> 归一化无关的 Pareto 支配与 hypervolume 两个原则性多目标指标上**显著胜过**最强的非学习穷举基线，
> 复杂度对网络规模近线性。

---

## 2. 数学主线：需写入论文的公式（替换旧推导）

**(M1) 加权 distinct-peer 查询策略（替换任何 hard top-k / 固定 degree 选择）。**
GNN 输出 logit `s_{ij}`，未归一化权重 `a_{ij}=exp(s_{ij})`。每轮精确抽取 `k` 个不同邻居，子集分布由
elementary symmetric polynomial 给出：
`P(S_i=S)=(∏_{j∈S} a_{ij})/e_k(a_i)`，`|S|=k`；边的包含概率 `π_{ij}` 满足 `Σ_j π_{ij}=k`。无 degree cap、无 top-k。

**(M2) 精确异质 quorum DP（替换 betainc / iid beta-tail）。**
三元（correct/wrong/no-response）生成函数
`Ψ_i(z,x,y)=∏_j [1 + z·a_{ij}(p^0_{ij} + p^+_{ij} x + p^-_{ij} y)]`，
`P_i(m,n)=[z^k x^m y^n]Ψ_i / e_k(a_i)`，由此得决策概率 `h^+_i, h^-_i, h^0_i`（强多数 `2α>k`）。
复杂度 `O(E k^3)`，可微。**不使用任何 iid-with-replacement beta-tail 近似。**

**(M3) 共享有限混合全局事件概率 `F`（替换 `F=\overline{F_i}` 节点均值，H1 核心）。**
网络共享离散潜变量 `Z∈{1..Q}`（权重 `ω_r`）耦合各节点；给定 `Z=r`，终态为条件乘积。
`S_C=Σ_r ω_r ∏_{i∈H} c_{ir}`，
`F_global = 1 − S_C = −expm1( logsumexp_r( log ω_r + Σ_{i∈H} log c_{ir} ) )`，loss `= −log S_C`。
`c_{ir}` 来自 §6 图耦合 Snowball 有限时域递推（用邻居 marginal preference，图耦合进入每个条件概率）。
**明确：`F_global` 是严格全局事件概率 `P(∃ i: Y_i≠C)`，不是节点平均。**

**(M4) 严格 finite-blocklength 链路 `ℓ`（替换 logistic-BLER sigmoid）。**
PPV 正态近似 `ε_FBL`，显式 channel dispersion `V(γ)=(1−(1+γ)^{-2})(log_2 e)^2`；blocklength 用真实
complex channel uses；request/response、Mode-2 collision、half-duplex、finite HARQ 分开建模；
headline 仅用 3GPP TR 37.885 标定路径，无理想化信道。

**(M5) 独立 D / E。** `D` = 全节点正确完成的全局 order statistic（由 `S(t)` 轨迹）；
`E` = 全网/全轮/全尝试总焦耳；功率头 `P_i=P_min+(P_max−P_min)σ(r_i)`、块长头 `n_i=n_min+(n_max−n_min)σ(b_i)`。

**(M6) 偏好条件化与标量化。** 偏好 `λ=(λ_F,λ_D,λ_E)∈Δ²` 作为 GNN 输入（FiLM 调制）；训练用 augmented
Chebyshev `max_m λ_m (z_m−z_m^*)/s_m + ρ Σ_m λ_m (z_m−z_m^*)/s_m`。一个 checkpoint 扫 `λ` 得整条前沿。

**(M7) Global-risk emission（替换 per-node confidence emission）。**
`r_{ir}(t)=−log c_{ir}(t)`（节点对全局风险 `−log S_r=Σ_i r_{ir}` 的贡献），下帧特征
`e_i^t=clip( sg[ Σ_r ρ_r r_{ir} ] / r_max , 0,1 )`（sg=stop-gradient）。
**注意：论文不得声称"有界标量自动约束全部 hidden state"——该主张已被机制 ablation 证伪**（见 §5 表 F）。

完整管线（建议作为一张 method 总图）：
`(a_θ,P_θ,n_θ) → π → Λ → γ → ℓ → (h^+,h^-,h^0) → c_{ir}(t) → (F_global, D, E)`。

---

## 3. 必须删除 / 重算的旧结论（规范 §12）

逐项删掉并按新数据替换：degree-4 protocol floor；旧 `F=0.0635` 等节点均值结果；mean-field / quenched /
MC "currency" 叙事；hard top-k / 生成树 constructor；旧 Pareto 表；per-node-confidence emission；
基于固定 degree 的 baseline 排名。**凡引用上述任一的句子/表/图都要重写。**

---

## 4. 实验设计（写入 Experiments 章；可重组，事实照用）

- **数值正确性验证**（method 的可信度支撑）：把新主线每个组件对照独立暴力枚举/闭式/scipy 的误差作为一张
  "exactness" 表（表 A），论证主线是**精确实现**而非启发式近似。
- **多目标基线对比**（headline 实验）：单个偏好条件化 checkpoint 在**训练**场景上训练、在**留出**场景上评估。
  所有方法（学习模型 + 全部基线）经**同一物理管线**打分，**仅**控制变量 `(s,P,n)` 的产生方式不同 → 公平。
  - 基线（诚实强基线，**禁用**固定 degree 排名）：`best-fixed` = per-scenario/per-preference 对一个
    392 点穷举常数策略族（4 种查询启发式 uniform/distance/inverse-distance/degree + 随机重启 × 7×7 常数 (P,n) 网格）
    取 oracle 前沿（**最强**非学习基线）；`λ-blind` = 同架构去偏好条件化（隔离偏好条件化价值）；
    `untrained` = 随机初始化（判别控制）。
  - 指标：① **Pareto set-coverage**（归一化无关，主指标）：`C(A,B)`=B 中被 A 支配的比例；
    ② **hypervolume**（模型无关归一化盒）；③ **Chebyshev front-scalar**（次要，模型在此**败**，须如实报告）。
  - 显著性：跨留出场景 paired Wilcoxon（单边）。
- **消融**：偏好条件化（vs λ-blind）、训练（vs untrained）、emission 的 bounded-scalar 机制 ablation（表 F）。
- **复杂度 profiling**：固定空间密度（面积 ∝ N ⇒ E=O(N)），扫 N 测端到端 runtime 与最大张量规模，给近线性拟合与图。
- **可复现**：固定种子；一键脚本 `python scripts/gates/run_all_gates.py`；分实验
  `baseline_comparison.py` / `profile_scaling.py` / `paper_tables.py`。

---

## 5. 实验数据（**照用**；可重排版/选取子集/转 LaTeX 表，不得改数值）

### 表 A — 数值正确性 / 精确性（新主线 vs 独立参照，绝对或相对误差）
| 组件 | 校验 | 误差 |
|---|---|---|
| 全局 F (G1) | 共享混合闭式 vs 暴力枚举 joint | 0.00e+00 |
| 全局 F (G1) | log 域 vs 直接 | 3.3e-19 |
| 全局 F (G1) | 失败分解恒等式 | 0.00e+00 |
| 全局 F (G1) | 解析 S_C vs Monte-Carlo | 7.2e-6 |
| k-subset (G2) | 归一化 \|Σp−1\| | 4.4e-16 |
| k-subset (G2) | 包含概率 vs 暴力 | 6.7e-16 |
| k-subset (G2) | \|Σπ−k\| 恒等式 | 2.2e-15 |
| quorum DP (G3) | 分布 vs 暴力枚举 | 2.9e-16 |
| FBL (G5) | V(γ) vs 闭式 | 4.4e-16 |
| FBL (G5) | ε_FBL vs scipy | 1.4e-17 |
| 时延 D (G6) | D vs CDF 参照 | 8.9e-16 |
| 能耗 E (G6) | E vs 暴力 | 0.00e+00 |
| emission (G8) | −log S_r = Σ_i r_ir 恒等式 | 0.00e+00 |
> 用途：论证主线是**机器精度精确**实现。可只保留代表性几行 + "all components match independent
> references to machine precision" 的总结句。

### 表 B — 多目标基线对比（headline，12 留出场景，paired Wilcoxon）
| 方法 | hypervolume↑ | C(model→·)↑ | C(·→model)↓ | Chebyshev↓ | HV 胜率 | p(单边) |
|---|---|---|---|---|---|---|
| **本文模型** | **0.606** | — | — | 0.118 | — | — |
| best-fixed（穷举 oracle，最强基线） | 0.509 | 0.392 | **0.000** | 0.061 | 100% | 2e-4 |
| fixed-uniform | 0.450 | 0.287 | 0.000 | 0.067 | 100% | 2e-4 |
| fixed-distance | 0.465 | 0.267 | 0.000 | 0.066 | 100% | 2e-4 |
| fixed-invdist | 0.438 | 0.480 | 0.000 | 0.097 | 100% | 2e-4 |
| fixed-degree | 0.450 | 0.287 | 0.000 | 0.067 | 100% | 2e-4 |
| λ-blind（消融） | 0.357 | 0.000 | 0.000 | 0.230 | 100% | 2e-4 |
| untrained（控制） | 0.267 | 0.990 | 0.000 | 0.280 | 100% | 2e-4 |
> 关键叙述点（自由组织）：① **C(·→model)=0.000 对所有基线** —— 无任一基线点 Pareto 支配任一模型点；
> ② 模型 C(model→best-fixed)=0.392；③ HV 模型无关盒下 0.606 vs 0.509 = **1.19×**，100% 留出场景，p=2e-4；
> ④ HV：模型 > best-fixed > λ-blind > untrained（梯度清晰，消融有效）。

### 表 C — 消融：偏好条件化与训练的价值
| 对比 | HV 胜率 | p | 结论 |
|---|---|---|---|
| 模型 vs λ-blind | 100% | 2e-4 | 偏好条件化带来真实价值 |
| 模型 vs untrained | 100% | 2e-4 | 训练带来真实价值；untrained 被 C=0.99 支配（gate 判别性） |

### 表 D — 代表性操作点（留出 12 场景 均值 ± 标准差；展示真实 steering）
| 偏好 λ | F（失败概率） | D（时延） | E（能耗） |
|---|---|---|---|
| F 优先 [1,0,0] | **0.355** ± 0.042 | 0.0517 ± 0.0011 | 0.629 ± 0.100 |
| D 优先 [0,1,0] | 0.711 ± 0.071 | **0.0415** ± 0.0007 | 0.677 ± 0.094 |
| E 优先 [0,0,1] | 0.513 ± 0.053 | 0.0529 ± 0.0010 | **0.168** ± 0.025 |
| 均衡 [.34,.33,.33] | 0.519 ± 0.053 | 0.0463 ± 0.0008 | 0.387 ± 0.057 |
> Steering 干净：每个偏好的目标在对角线取最小（F 优先→最低 F、D 优先→最低 D、E 优先→最低 E）。
> 可据此画一张 3D 或投影 Pareto 散点图 + 偏好箭头。

### 表 E — 端到端复杂度（固定密度，面积 ∝ N）
| N | E | 端到端 runtime (ms) |
|---|---|---|
| 200 | 7,702 | 298 |
| 800 | 35,926 | 774 |
| 3,200 | 150,358 | 1,432 |
| 6,400 | 307,016 | 1,954 |
> 拟合：E~N 指数 1.06；端到端 t~E 指数 0.51（被固定开销主导的次线性，即 ≤ 线性）。最大物化张量是
> `O(E·k^3)` 的 quorum cube，其对 E 的标度指数恰为 **1.000**（无 N×N）。图：`docs/gate_evidence/g9_scaling.png`。

### 表 F — Emission 的 bounded-scalar 机制 ablation（同一有界 emission，30 帧 ‖H‖ 增长）
| recurrence cell | ‖H‖ 增长 |
|---|---|
| GRU（门控） | 0.81 |
| contractive（ρ=0.5） | 0.02 |
| expansive（ρ=1.3，非收缩） | **2017** |
> 结论（如实写）：有界标量输入**不**自动约束 hidden state —— expansive recurrence 在同一有界 emission 下
> ‖H‖ 发散；hidden-state 有界性是 recurrence 的收缩/门控性质（BIBO），非输入性质。论文**不得**宣称
> "bounded scalar 自动约束全部 hidden state"，应陈述被验证的窄性质（e∈[0,1] 构造性、stop-gradient、
> emission 与全局 `−log S_C` 精确对齐）。

### G7 steering（可作为 method/ablation 支撑句）
单 checkpoint 扫 λ：10/10 互不支配，3/3 argmin-at-vertex steering；λ-blind 消融 steering 塌缩。

---

## 6. 必须如实写出的局限（诚实性，**不可省略**）

1. **Chebyshev 极值劣势。** 穷举网格在*每偏好单点*（Chebyshev）指标上更优（model 0.118 vs best-fixed 0.061，
   胜率 0%），靠暴力把功率/块长推到极限；但那些点**不 Pareto 支配**模型（在其它目标上更差，见 C(·→model)=0）。
   把贡献定位为**整体前沿质量（coverage + hypervolume）+ 泛化 + 效率**（单 checkpoint、无 per-scenario 搜索），
   **不是**角点最优。
2. **Dense-deployment 作用域。** 实验在完整候选图（所有车在射程内）上；优化杠杆是**轮询拓扑**（查询哪 k 个 peer）
   + per-node 功率/块长，非物理链路存在性；完整图上 `degree` 查询退化为 `uniform`（已在基线集声明）。
   `SCALE≫RADIUS` 的稀疏图使部分节点低于 quorum size k（需 §7.2 候选不足协议），列为 future work。
3. **`F` 对链路可靠性呈 U 形**（过可靠链路传播错误票）——真实涌现性质，需据此调和"↑P ⇒ ↓F"的旧说法。
4. emission 的 bounded-scalar claim 被**证伪**（表 F），论文据此修正措辞。

---

## 7. 写作自由度（claudeprism 自行决定）

- 章节顺序、小标题、过渡、强调点、摘要/引言的叙事弧、相关工作的对照角度。
- 哪些表整合/精简（如表 A 只留代表性几行）、图的形式（Pareto 散点、复杂度曲线、method pipeline 图、消融条形）。
- 定理/命题的陈述粒度、记号体系（保持语义一致即可）。
- 把上面的"关键叙述点"组织成流畅论证，不必逐条罗列。
- 提交为 `paper/main.tex` 的带追踪修改；旧推导若保留，须显式标注为历史/对照。

**底线**：数学按 §2、删除按 §3、数字按 §5、局限按 §6 —— 这四者不可动；其余皆可自由发挥。
