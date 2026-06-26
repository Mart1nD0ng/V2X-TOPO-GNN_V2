> ⚠️ **已被取代（历史文档）。** 本报告记录旧的 11-gate global-FDE 主线（依据已废弃的
> `GLOBAL_CONSENSUS_MATH_REDESIGN.md`），冻结于 tag `legacy-global-fde-v1`。当前主线是
> Effective-Sampling-Dynamics 重构，见 [`ESD_FINAL_REPORT_CN.md`](ESD_FINAL_REPORT_CN.md)。

# 最终报告 — V2X-TOPO-GNN 数学重构（全部 11 个 gate 已绿）

> 生成于 2026-06-22，gate 驱动重构完成时（规范 §8）。
> 唯一规范来源：[`GLOBAL_CONSENSUS_MATH_REDESIGN.md`](GLOBAL_CONSENSUS_MATH_REDESIGN.md)。
> 进度 / 决策日志：[`REFACTOR_PROGRESS.md`](REFACTOR_PROGRESS.md)。
> **本文是工程报告，不修改论文。** 对论文的建议另见：[`PAPER_UPDATE_PROPOSAL_CN.md`](PAPER_UPDATE_PROPOSAL_CN.md)。

## 1. 状态：11 / 11 验收门槛全绿

| Gate | 约束 / 模块 | 证明了什么 |
|------|------------|-----------|
| G1 | H1 / §3.3 | `F_global` 是严格的全局事件概率（共享有限混合 log 域 product-mixture），**非** per-node 边缘均值 |
| G2 | §3.1 | 精确加权 distinct-peer k-subset 查询策略（elementary symmetric polynomial） |
| G3 | §3.2 | 精确异质 quorum DP（三元生成函数），**无 iid beta-tail** |
| G4 | H2 / §3.4 | 物理约束自适应拓扑——稀疏性由代价涌现，**无 degree cap / top-k** |
| G5 | H3 / §3.5 | 严格 finite-blocklength `ℓ(γ,n,B)`，显式 dispersion `V(γ)`，3GPP 标定 |
| G6 | §3.6 | 独立的全局时延 `D` 与全网能耗 `E` + 功率 / 块长头 |
| G7 | §3.7 | 偏好条件化 GNN——单个 checkpoint 扫出 F/D/E Pareto 前沿 |
| G8 | §3.8 | global-risk emission `r_ir=-log c_ir` + stop-gradient；bounded-scalar 主张被 ablation **证伪** |
| G9 | H4 | 端到端近线性复杂度（确定性 no-`N×N` 张量守卫） |
| G10 | H5 | 唯一 live 数学主线；legacy 归档至 `legacy/` |
| G11 | 终极 | 留出场景基线对比胜出，且有显著性 |

每个 gate 都经过独立的多 lens **对抗验证**（Workflow）硬化。历次验证发现的几乎都是
**gate 判别性 / 报告诚实性** gap，而非最终实现正确性缺陷；所有确认的发现均已修复。

## 2. 统一数学主线（`src/mainline/`）

```
(a_θ, P_θ, n_θ)  ->  π (G2，elementary symmetric polynomial)
                 ->  Λ 接收负载 (G4)  ->  γ (几何 + 干扰)  ->  ℓ (G5，FBL V(γ))
                 ->  (h+,h-,h0) 三元生成函数 quorum DP (G3)
                 ->  图耦合 Snowball 递推  ->  c_ir(t)
                 ->  共享有限混合  F_global = 1 - Σ_r ω_r ∏_i c_ir   (G1, H1)
                 ->  D 全局 order statistic、E 全网能耗 (G6)
                 ->  augmented-Chebyshev 偏好标量化 (G7)
                 ->  global-risk emission e_i = clip(sg[Σ_r ρ_r r_ir]/r_max,0,1)  (G8)
```

模块：`symmetric_polynomials.py` (G2)、`quorum_dp.py` (G3)、`snowball.py` + `global_evaluator.py`
(G1)、`topology.py` (G4)、`finite_blocklength.py` (G5)、`objectives.py` (G6)、`model.py` (G7)、
`emission.py` (G8)。整条 live 路径**只** import `src.mainline`；被取代的
mean-field / beta-tail / logistic-BLER / degree-cap 旧推导冻结在 `legacy/`
（[`legacy/ARCHIVED.md`](../legacy/ARCHIVED.md)）。

## 3. 一键复现

```bash
python scripts/gates/run_all_gates.py          # 全部 11 gate -> docs/gate_evidence/latest.json
python scripts/gates/run_all_gates.py G3 G9    # 跑子集
python -m pytest tests/ -q                     # 全套单元测试
```

分领域驱动脚本（可复现、固定种子）：`scripts/analysis/profile_scaling.py`（G9 图）、
`scripts/analysis/baseline_comparison.py`（G11 研究）。gate 里的所有数字都来自这些可复现脚本——
**无任何硬编码**（反作弊铁律 §7）。

## 4. H4 复杂度证据（G9）

端到端前向，`N = 200..6400`，固定空间密度（面积 ∝ N ⇒ `E = O(N)`）：
`E~N` 指数 1.06；分阶段 `t~E` 指数 build 1.05 / GNN 0.83 / consensus 0.55 / total 0.56
（无任一阶段超线性）；最大物化张量是 `O(E·k³)` 的 quorum cube——其对 E 的标度指数**恰为 1.000**，
且一个确定性 `TorchDispatch` 守卫会让任何真正的 `N×N` 失败（实测：注入 N×N → 指数 1.92、
cubic → 1.50，均 FAIL gate）。图：[`gate_evidence/g9_scaling.png`](gate_evidence/g9_scaling.png)。

## 5. G11 基线对比（headline 结果）

单个偏好条件化 checkpoint，在**训练**场景上训练、在**留出**场景上评估，对比诚实强基线
（4 种查询启发式 + 随机重启 × 7×7 常数 (P,n) 网格 = 392 点的穷举常数策略族；一个去偏好条件化的
ablation；一个未训练控制）。所有方法都经**同一物理管线**（`evaluate_controls`）打分——
**仅**控制变量 `(s, P, n)` 的产生方式不同。

主指标（12 个留出场景，paired Wilcoxon）：

| 指标 | 模型 vs 最强基线（`best-fixed`） |
|------|--------------------------------|
| **Pareto set-coverage**（归一化无关） | C(模型→网格) = **0.392** vs C(网格→模型) = **0.000**，p = 0.0002 |
| **Hypervolume**（模型无关归一化盒） | 0.606 vs 0.509 = **1.19×**，胜率 **100%**，p = 0.0002 |
| vs 去偏好条件化 ablation | 模型 HV 100% 胜，p = 0.0002（偏好条件化有真实价值） |
| vs 未训练控制 | 被支配，C(模型→未训练) = 0.99（gate 具判别性） |

**无任何基线点严格 Pareto 支配任一模型点**（C(基线→模型) < 0.02；认证的 600 步预算下实测 0.000）。
该胜出**跨种子稳健**（多个 disjoint 种子区间验证），并能抵住一个被激进强化到 968 点的基线。

复现：`python scripts/analysis/baseline_comparison.py`（完整 16/16 × 900 步研究）。
证据：[`gate_evidence/g11_baseline.json`](gate_evidence/g11_baseline.json)。

## 6. 诚实边界（如实记录，未隐藏）

1. **Chebyshev 极值。** 穷举网格在**每个偏好的单点（Chebyshev）**指标上更优——包括部分均衡偏好——
   靠把功率/块长暴力推到极限。但那些点**不 Pareto 支配**模型（它们在其它目标上更差）。模型的胜在
   整体前沿质量 + 泛化 + 效率（单 checkpoint、无 per-scenario 搜索），**非**极值角点。
2. **Dense-deployment regime。** G11 用完整候选图（所有车都在射程内）；优化杠杆是**轮询拓扑**
   （查询哪 k 个 peer，经 G2 k-subset 分布）+ per-node 功率/块长。当 `SCALE ≫ RADIUS` 时候选图稀疏、
   部分节点低于 quorum size `k`（需 §7.2 候选不足协议）；600 步模型在该更难 regime 下不胜——作用域外。
3. **`F` 对链路可靠性呈 U 形**（过可靠的链路会把错误票也传播）；作为规范 §9.4 偏差记入人工复核
   （不影响任何 gate）。
4. **`paper/main.tex` 仍叙述被取代的旧推导**（betainc beta-tail quorum + node-mean `F`）；调和它是
   gate 全绿后的建议 [`PAPER_UPDATE_PROPOSAL_CN.md`](PAPER_UPDATE_PROPOSAL_CN.md)。
5. **`src/mainline/` 尚未 commit**（工作树）；建议 commit 以便 gate 跑在不可变树上。

## 7. 交付清单（规范 §8）

- [x] 统一数学主线代码 — `src/mainline/`
- [x] 全套单测 + 一键 gate 复跑脚本 — `tests/`、`scripts/gates/run_all_gates.py`
- [x] Profiling 报告（H4 近线性证据）— G9 + `gate_evidence/g9_scaling.png`
- [x] 基线对比 + 复现命令 + 显著性 — G11 + `baseline_comparison.py`
- [x] `REFACTOR_PROGRESS.md` 最终态（D1–D12、gate 表全 🟢）
- [x] 论文更新**建议**（非直接改）— `PAPER_UPDATE_PROPOSAL_CN.md`
