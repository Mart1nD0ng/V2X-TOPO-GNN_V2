# 论文更新建议（仅建议 — 不修改 `paper/main.tex`）

> 起草于 2026-06-22，全部 11 个验收 gate 转绿之后（规范 §8 规定：gate 全绿后方可起草论文更新**建议**；
> 规范 §0/§5 禁止在此之前改动论文 headline/claim 文字）。**`paper/` 下未做任何改动。**
> 本文给出由人工作者执行、用以把手稿与统一主线（`src/mainline/`）调和一致的修改建议。
> 唯一规范来源：[`GLOBAL_CONSENSUS_MATH_REDESIGN.md`](GLOBAL_CONSENSUS_MATH_REDESIGN.md)。

## A. `paper/main.tex` 中需**替换**的冲突推导（H5）

G10 对抗审计发现 `paper/main.tex` 仍叙述**被取代**的旧推导，与 live 主线冲突：

1. **Quorum（约第 605–626 行，引用见 238/518/557–558/682/693）。** 论文用
   `H(x;k,α)=I_x(α,k−α+1)`“由 `betainc` 前向计算”，自述为 Poisson-binomial 尾的
   *iid-with-replacement beta-tail / mean-field 代理*。
   → **替换为**精确异质 **distinct-peer 生成函数 quorum DP**（`src/mainline/quorum_dp.py`，规范 Eq.25–30）：
   `Ψ_i(z,x,y)=∏_j[1+z a_ij(p0 + p+ x + p− y)]`，`P_i(m,n)=[z^k x^m y^n]Ψ_i / e_k(a_i)`，
   `h+/h−/h0` 由 Eq.27–29。删除全部 `betainc` / `I_x(α,k−α+1)` / “iid with replacement” /
   “mean-field surrogate” 措辞，以及约 1205–1206 行被注释掉的 `scipy.special.betainc` 引用。

2. **Consensus `F`（约第 574–595、644–656、668 行）。** 论文用行归一化 support bridge
   `q_ij = w_ij/Σ w_ij` 喂入 `F = \overline{F_i}`（per-node **均值**），明确是
   “mean-field closure … treats neighbour states as independent”。
   → **替换为共享有限混合的全局事件概率**（`src/mainline/global_evaluator.py`，规范 Eq.5–13）：
   `F_global = 1 − Σ_r ω_r ∏_{i∈H} c_ir = −expm1(logsumexp_r(log ω_r + Σ_i log c_ir))`，
   loss `= −log S_C`。删除 `F = \overline{F_i}` 节点均值定义；明确 `F_global` 是严格全局事件概率
   `P(∃ i: Y_i ≠ C)`，**非**节点平均。

## B. 需**删除 / 重算**的旧结论（规范 §12）

不得沿用：degree-4 protocol floor；旧 `F = 0.0635` 节点均值结果；mean-field / quenched / MC
“currency” 叙事；hard top-k / 生成树 constructor；旧 Pareto 表；per-node-confidence emission；
基于固定 degree 的 baseline 排名。所有定量主张必须由主线重算。

## C. 建议的新 headline / 结果（来自已 gate 化的主线）

- **方法。** 偏好条件化 FiLM 拓扑 GNN，输出加权 distinct-peer 查询分布（精确 ESP、无 top-k）+
  per-node 功率/块长头；精确异质 quorum DP；共享有限混合全局 `F`；严格 finite-blocklength
  `ℓ(γ,n,B)`（显式 dispersion `V(γ)`）；独立全局时延 `D` 与全网能耗 `E`；global-risk emission
  `r_ir = −log c_ir`。单个 checkpoint 扫出 F/D/E Pareto 前沿。
- **复杂度（H4）。** 端到端对 `N, E` 近线性（G9）；附 `g9_scaling.png` 图与确定性 no-`N×N` 守卫。
- **Headline 结果（G11）。** 在留出的 dense-deployment 场景上，单 checkpoint 的 Pareto 前沿
  **支配**最强非学习基线（穷举 392 点常数策略网格）：归一化无关的 Pareto set-coverage
  C(模型→网格)=0.39 vs C(网格→模型)=0.00（p=2×10⁻⁴）；模型无关盒下 hypervolume 1.19×、留出场景 100%
  胜出；并胜过去偏好条件化 ablation（偏好条件化有价值）与未训练控制。报告精确复现命令 + 种子 + Wilcoxon p。

## D. 论文**必须**如实陈述的边界（反降级 §7）

- 穷举网格在*每偏好单点 Chebyshev* 指标上更优（更好的单目标极值，含部分均衡偏好）；但那些点**不**
  Pareto 支配模型。把贡献定位为前沿质量（coverage + hypervolume）+ 泛化 + 效率（单 checkpoint、
  无 per-scenario 搜索），**不是**角点最优性。
- G11 在 **dense-deployment** 完整候选图上；稀疏图 regime（`SCALE ≫ RADIUS`、低于 quorum 的节点需
  §7.2 协议）作用域外，应列为 future work。
- `F` 对链路可靠性呈 **U 形**（过可靠的链路会传播错误票）——这是真实涌现性质，非 bug；据此调和
  §9.4 “↑P ⇒ ↓F” 的说法。

## E. 流程说明

将以上作为带追踪的手稿修改执行，重算的数字从 `docs/gate_evidence/*.json` 与复现脚本拉取，**不要手抄任何数字**。
旧推导只能作为显式标注的历史/对照附录保留（如同 `legacy/ARCHIVED.md` 对代码所做），绝不作为 headline 定理。
