# Claude Code Loop Prompt：Global-F/D/E Project Redesign

你正在重构仓库：

`Mart1nD0ng/V2X-TOPO-GNN_V2`

先阅读并遵守：

1. `docs/GLOBAL_CONSENSUS_MATH_REDESIGN.md`
2. `docs/MAINLINE.md`
3. `docs/AUDIT_REPORT.md`
4. 当前主线代码：
   - `src/consensus/graph_coupled_avalanche.py`
   - `src/evaluation/v2x_consensus_bridge.py`
   - `src/topology/construction.py`
   - `src/v2x_env/channel_model.py`
   - `src/v2x_env/nr_v2x_sidelink.py`
   - `src/models/hierarchical_gnn.py`

本任务是项目级数学与实现重构，不是继续润色论文。除非所有验收门槛通过，不得更新论文 headline、摘要或实验结论。

---

## 1. 不可降级的硬约束

### H1：\(F\) 必须是严格全局事件概率

主指标必须是

\[
F_{\mathrm{global}}
=
P(\exists i\in\mathcal H:Y_i\neq C)
=
1-P(\forall i\in\mathcal H:Y_i=C).
\]

节点失败均值、正态尾、独立节点近似不得再命名为 consensus failure。所有 exactness claim 必须写明适用随机模型和结构假设。

### H2：不得存在人为固定节点度上限

mainline 中删除 fixed degree cap、degree-4、hard top-\(k\) support 和 candidate-count truncation。候选边只能由物理可达性产生。每轮 polling 数 \(k_{\mathrm{poll}}\) 是协议参数，不是图度上限。

### H3：端到端属性不得降级

必须保留完整梯度链：

\[
\theta
\to
\text{query/resource policy}
\to
\Lambda
\to
\gamma
\to
\ell
\to
\text{global consensus law}
\to
(F,D,E)
\to
\mathcal L.
\]

禁止 topology labels、learned reward surrogate、MC rollout training、部署时另行 hardening 造成 train/deploy mismatch。

### H4：复杂度必须对 \(N,E\) 近似线性

目标复杂度：

\[
O\!\left(
Q R_{\max}
(Ek_{\mathrm{poll}}^3+N\beta)
\right).
\]

禁止构造 \(N\times N\) dense tensor。固定密度和通信半径下应保持 \(E=O(N)\) 的期望扩展性。

### H5：项目只能有一套数学主线

新 mainline 只能存在一套：

- global \(F\) 定义；
- distinct-peer quorum；
- adaptive query policy；
- finite-blocklength model；
- \(D/E\) 定义；
- training/deployment policy。

mean-field、quenched currency、legacy node-mean \(F\)、hard top-\(k\)+ST 必须移至 archive，且任何 mainline 脚本不得 import。

---

## 2. 模型结构升级建议

以 `docs/GLOBAL_CONSENSUS_MATH_REDESIGN.md` 为规范，实现：

1. **Weighted distinct-peer query policy**
   - GNN 输出 \(a_{ij}=\exp(s_{ij})\)；
   - 使用 elementary symmetric polynomial 定义 exact weighted \(k\)-subset distribution；
   - 训练和部署使用同一分布与同一 sampler。

2. **Exact heterogeneous quorum DP**
   - correct / wrong / no-response 三元生成函数；
   - 不使用 iid beta tail；
   - 对 \(k_{\mathrm{poll}}\) 小常数实现 \(O(Ek^3)\) 可微 DP。

3. **Shared finite-mixture global evaluator**
   - 计算 conditional node-state recurrence；
   - 使用 log-domain product-mixture 计算 \(S_C\) 与 \(F_{\mathrm{global}}\)；
   - loss 使用 \(-\log S_C\)。

4. **Physics-constrained adaptive topology**
   - 不裁剪每节点候选数；
   - 由 inclusion probability 计算 receiver load；
   - 通过 interference、Mode-2 collision、half-duplex、queueing、maintenance cost 抑制 hub overload 和过密支持。

5. **Rigorous finite-blocklength path**
   - 显式实现 channel dispersion \(V(\gamma)\)；
   - blocklength 使用真实 complex channel uses；
   - request / response、collision、half-duplex、finite HARQ 分开建模；
   - headline 环境只使用 3GPP-grounded path。

6. **Independent \(D/E\)**
   - \(D\)：all-node correct completion 的全局 order statistic；
   - \(E\)：全网络、全轮次、全尝试的总 joule；
   - 新增 power head 与 blocklength/resource head，形成真实 F/D/E conflict。

7. **Preference-conditioned model**
   - 将 \(\lambda_F,\lambda_D,\lambda_E\) 输入 GNN；
   - 使用 FiLM 或 hypernetwork conditioning；
   - 一个 checkpoint 输出多个 Pareto operating points。

8. **Global-risk emission**
   - 将原 per-node marginal confidence 改为节点对 \(-\log S_C\) 的风险贡献；
   - 保持 stop-gradient；
   - 不得宣称 bounded scalar 自动约束全部 hidden state，必须用机制实验验证。

---

## 3. 最终交付目标

在独立重构分支完成以下交付：

### 数学与文档

- `docs/GLOBAL_CONSENSUS_MATH_REDESIGN.md`
- `docs/GLOBAL_FDE_IMPLEMENTATION_SPEC.md`
- `docs/GLOBAL_FDE_VALIDATION_REPORT.md`
- 更新后的 `docs/MAINLINE.md`
- legacy migration / deprecation 说明

### 新主线代码

建议模块：

```text
src/consensus/
  weighted_subset_quorum.py
  global_mixture_avalanche.py
  global_event_metrics.py

src/topology/
  elementary_symmetric.py
  adaptive_query_policy.py

src/v2x_env/
  finite_blocklength.py
  resource_coupled_sidelink.py
  channel_scenarios.py

src/evaluation/
  global_fde_bridge.py

src/losses/
  global_fde_objective.py
  pareto_conditioned_objective.py

src/models/
  global_risk_adaptive_gnn.py
  global_risk_emission.py
```

### 迁移

- 将 mean-field、legacy quenched closure、hard top-\(k\)、fixed budget 代码迁入 `archive/legacy_v1/`；
- mainline tests 和 imports 不得依赖 archive；
- 保留可复现历史 tag，但不得复用旧 headline 数字。

### 验证与实验

- brute-force subset exactness；
- small-\(N\) exact global chain；
- direct MC fidelity；
- finite-difference gradient；
- FBL monotonicity；
- no-hard-degree tests；
- runtime / memory scaling；
- matched-budget baselines；
- genuine multi-seed Pareto；
- cross-\(N\)、density、resource pool、protocol transfer；
- temporal global-risk emission ablation。

---

## 4. 完成判据

只有以下条件全部满足，任务才算完成：

1. quorum DP 与 brute-force 误差 `< 1e-10`；
2. global mixture formula 与显式枚举误差 `< 1e-12`；
3. autograd / finite-difference 相对误差 `< 1e-4`；
4. \(N\le8\) exact-joint validation 有完整报告；
5. MC topology ranking Spearman \(\rho\ge0.95\)；
6. \(Q\) 加倍后 `abs(delta log S_C) < 1e-3`；
7. 所有 headline metric 使用 `F_global`；
8. mainline 中不存在 fixed degree cap、active candidate truncation、hard top-k support；
9. 训练和部署使用同一 weighted-subset policy；
10. 主线中不出现 `mean-field currency` 或 legacy node-mean \(F\)；
11. FBL 显式包含 \(V(\gamma)\)，单位检查通过；
12. D/E gradient median cosine `< 0.95`；
13. 至少三个跨 seeds 稳定的非支配 Pareto 点；
14. 不创建 \(N\times N\) tensor，runtime 对 \(E\) 近似线性；
15. 所有图表和结果可由 commit、config、checkpoint、manifest 一键复现；
16. 完整 mainline test suite 与 production-scale smoke 全绿。

---

## 5. 每轮迭代工作流（LOOP）

持续循环，直到全部完成判据通过：

### L1. Inspect

- 读取数学规范、当前代码、测试和上轮状态；
- 选择一个最小且可验证的未完成 gate；
- 明确本轮不会破坏的现有接口。

### L2. Specify

在实现前写清：

- 数学定义；
- 输入输出 contract；
- exactness assumptions；
- 时间/空间复杂度；
- 数值稳定方案；
- 本轮 acceptance test。

必要时先更新 `GLOBAL_FDE_IMPLEMENTATION_SPEC.md`。

### L3. Test First

先新增失败测试：

- brute-force reference；
- invariants；
- gradient check；
- boundary cases；
- complexity guard。

没有可失败测试，不得直接改主线。

### L4. Implement

- 做满足测试的最小完整实现；
- 保持 sparse / chunked；
- 使用 log-domain、`logsumexp`、`expm1` 等稳定运算；
- 不留下 mainline TODO、stub 或 silent fallback。

### L5. Verify

依次运行：

1. targeted unit tests；
2. math exactness tests；
3. gradient tests；
4. integration tests；
5. full mainline gate；
6. runtime / memory smoke。

记录命令、commit、配置和结果。

### L6. Adversarial Audit

主动检查：

- 是否又把 marginal 指标命名为 global；
- 是否出现隐藏 fixed cap；
- 是否训练/部署分布不同；
- 是否引入 dense \(N^2\)；
- 是否把 approximation 写成 exact；
- 是否通过改名掩盖 legacy 数学；
- 是否新 \(D/E\) 实际仍成比例。

### L7. Document and Commit

更新：

- implementation spec；
- validation report；
- migration status；
- remaining gates；
- known limitations。

每轮形成一个原子 commit，message 写明通过的 gate。

### L8. Continue

选择下一个最小未完成 gate，继续循环。不要因任务较大而停止，也不要在每轮等待人工确认。

---

## 6. 严禁的捷径（FORBIDDEN）

1. 用节点均值、union bound、Normal tail 或独立乘积代替严格 global \(F\)；
2. 把条件独立结构隐去后声称 unrestricted exact Avalanche；
3. 保留 degree-4 / degree-8 作为主线限制；
4. 在物理半径后截取固定数量候选；
5. hard top-\(k\)+straight-through 继续作为 mainline；
6. 训练 soft policy、部署另一种 hard policy；
7. 用 Monte-Carlo reward 训练；
8. 学习一个 evaluator surrogate 替代解析物理与共识模型；
9. 用旧 mean-field / quenched 结果填新表；
10. 把 `D ∝ E` 的结果称为三目标 Pareto；
11. 删除 channel dispersion 后仍称 strict finite blocklength；
12. 用 logistic BLER 作为 headline FBL，却不做清楚标注；
13. 创建 \(N\times N\) adjacency、pair tensor 或全局 dense attention；
14. 只改文档不改实现，或只改实现不更新数学规范；
15. 为通过测试降低阈值、跳过测试、固定随机种子掩盖不稳定；
16. 编造实验数字、吞掉 NaN、silent clamp 无审计；
17. 在全部 gate 通过前修改论文 headline claim。

---

## 7. 停止与上报条件

仅在以下情况停止自动 LOOP，并输出结构化上报：

1. 数学规范内部存在无法同时满足的矛盾；
2. exactness、复杂度或端到端约束被证明不可同时满足；
3. 需要改变协议语义，例如 \(|\mathcal N_i|<k_{\mathrm{poll}}\) 的处理尚无授权选择；
4. 新全局 perfect-link floor 在目标规模上接近 1，导致现有 protocol 参数不可行；
5. small-\(N\) exact / MC 结果系统性否定当前 shared-mixture model，需要升级 joint structure；
6. 资源限制使必需验证无法完成，并已提供最小复现实验与资源估计；
7. 发现会破坏历史结果但无法安全迁移的 repository-level blocker；
8. 全部完成判据已通过。

上报格式必须包含：

```text
STATUS:
BLOCKER OR COMPLETE:

EVIDENCE:
- failing/passing tests
- equations or counterexample
- runtime/memory measurements
- relevant files and commits

WHY THIS CANNOT BE SAFELY BYPASSED:

MINIMUM DECISION NEEDED FROM USER:

RECOMMENDED NEXT ACTION:
```

除上述条件外，继续 LOOP，不要提前停止。
