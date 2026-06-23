# V2X-TOPO-GNN：面向 Avalanche 有效抽样动力学的技术重构规范

**文档用途**：项目数学、仿真环境、端到端模型和验证方法的统一技术依据。  
**审查基线**：GitHub `main`，审查时解析到提交 `7b193452a933dd706fecd7bc9291dc9308dac64a`。  
**论文状态**：暂停继续改写。项目完成本规范的关键验收后再重写论文。  
**核心决策**：项目由“平权优化 \(F/D/E\)”转为“可靠性约束下优化 Avalanche 有效抽样动力学、尾确认时延和能耗”。

---

## 1. 执行摘要

Avalanche/Snowball 与 PBFT 对不可靠链路的敏感点不同。PBFT 依赖确定性 quorum，链路丢失直接决定一个阶段能否推进；Snow 系列通过反复随机抽样积累置信度，偶发、对称、与意见无关的丢包通常首先降低每轮获得有效证据的概率，从而增加确认轮数、尾时延和能耗。

然而，在 V2X 中，链路失效并非独立、对称且与意见无关。局部高可靠链路会使实际响应样本偏向同一路段、同一遮挡区域、同一传感器误差来源；弱割会形成局部回音室；多个查询者共享接收节点会产生拥塞和相关丢包。因此，组网真正控制的是：

\[
\boxed{
\text{有效抽样动力学}
=
\text{响应成功率}
+
\text{抽样代表性}
+
\text{证据独立性}
+
\text{跨区域混合}
}
\]

它进一步决定：

\[
\text{有效 quorum 速率}
\rightarrow
\text{确认速度},
\]

\[
\text{正确方向漂移}
\rightarrow
\text{有效性与错误决定风险},
\]

\[
\text{弱割、分区和相关采样}
\rightarrow
\text{agreement 与 liveness 风险}.
\]

新的主问题应定义为：

> 在异质链路、资源竞争、相关感知和动态道路拓扑下，构造概率 peer-query policy，使 Snowball 在满足安全性、有效性和 deadline 可靠性约束的前提下，最小化全局确认尾时延与网络能耗。

---

## 2. 当前仓库的全局审查结论

### 2.1 可以保留的资产

当前项目已完成若干质量较高的局部组件：

1. 无固定 degree cap 的物理候选图构造；
2. 基于 elementary symmetric polynomial 的加权 distinct-peer \(k\)-subset policy；
3. 对该 product-weighted subset law 的异质 correct/wrong/no-response quorum DP；
4. 显式 channel dispersion 的 finite-blocklength normal approximation；
5. 对数域全局事件公式及数值稳定实现；
6. 稀疏、degree-bucketed 的近线性张量布局；
7. legacy 隔离与 gate 驱动的测试框架。

这些组件应作为重构起点，而不是推倒重来。

### 2.2 当前主线必须废弃或降级的设计

| 当前设计 | 根本问题 | 新定位 |
|---|---|---|
| \(\lambda_F,\lambda_D,\lambda_E\) 平权 preference-conditioned Pareto | 允许用错误风险换速度；不符合安全 V2X | 可靠性硬约束，优化尾时延/能耗 |
| 每节点 power、blocklength heads 参与 headline | 不能证明优势来自 topology；改变了论文问题 | 主线固定资源配置；作为后续扩展 |
| `evaluate_controls` | 未调用完整物理链；固定 `tau_proxy`；\(Q=1\) | 删除并由 canonical round-coupled evaluator 替代 |
| `load_coupled_link_reliability` | 仅 gate/test 调用，没有进入 headline | 接入唯一生产路径 |
| `queueing_utilisation` | 实现但未被消费 | 接入路径或删除 |
| \(N\in\{8,10,12\}\) 完全图 headline | 不是目标 V2X 规模，也没有弱割、稀疏性和区域结构 | headline 改为 \(N=100\sim10000\) |
| `snowball.py` 的 \(2\beta+2\) streak automaton | 实质是 Snowflake-style consecutive quorum，不是完整 Snowball confidence accumulation | 实现真实 binary Snowball 或明确改名 |
| \(Q=1\) product global \(F\) | 忽略节点间由图、共享物理和意见传播导致的相关性 | 仅作解析训练近似；dynamic MC 最终裁判 |
| 当前 Monte-Carlo | 从解析终态 marginals 再采样，属于自洽性测试 | 新增独立逐轮 dynamic MC |
| global-risk emission | 与旧 global product risk 对齐，但没有证明真实时序收益 | 暂停作为核心贡献 |
| 两层 source-side mean GNN | 看不见 receiver contention、证据相关性、弱割和长程混合 | 换成多图、多尺度、set-aware 模型 |

### 2.3 代码—机制闭环缺失

当前项目最大的工程风险不是“没有实现机制”，而是“机制实现、测试通过，但 headline 路径没有调用”。例如完整 interference/collision/half-duplex/request-response 物理链存在于 `src/mainline/topology.py`，但 `src/mainline/model.py::evaluate_controls` 仅计算几何 SINR、FBL、固定 `tau_proxy` 下的 receiver load 和 Mode-2 collision。

后续开发必须建立 **Mechanism-to-Mainline Contract**：

- 所有 headline、baseline、ablation、figure generation 只能调用一个 canonical evaluator；
- 每个被声称启用的机制必须有 runtime trace；
- 每个机制必须有 sentinel activation test；
- 每个机制必须有因果 ablation；
- 未被 canonical path 消费的配置字段必须报错；
- “文件存在”或“单元测试通过”不等于“机制进入系统”。

---

## 3. 协议语义必须先固定

### 3.1 当前实现不是完整 Snowball

当前状态只记录：

- 当前偏好侧；
- 连续相同 quorum 的 streak；
- correct/wrong absorbing state。

这对应 Snowflake-style consecutive quorum。完整 binary Snowball 还需要累计各颜色的 confidence，并根据累计 confidence 更新 preference。

必须二选一：

1. **推荐路线**：实现真实 binary Snowball；
2. 保留当前状态机，但项目、代码和论文统一称为 Snowflake-like probabilistic consensus。

不得继续把 Snowflake 状态机称为完整 Snowball。

### 3.2 推荐的 finite-horizon binary Snowball 状态

对二值 \(+\) / \(-\)，可用置信差而不是保存两个完整计数：

\[
d_i(t)=d_i^+(t)-d_i^-(t).
\]

单节点状态可写为：

\[
X_i(t)=
\left(
d_i(t),
\operatorname{pref}_i(t),
\operatorname{last}_i(t),
c_i(t),
\operatorname{decision}_i(t)
\right),
\]

其中：

- \(d_i(t)\in[-t,t]\)；
- \(c_i(t)\in\{0,\ldots,\beta-1\}\)；
- `decision` 为 undecided/correct/wrong。

有限 \(R_{\max}\) 下状态有限，可以构造稀疏 transition DP。复杂度会高于当前 \(2\beta+2\) 模型，但仍可在小 \(R_{\max}\) 下对每个节点求解。若生产训练需要近似，必须以真实 dynamic MC 和 small-\(N\) exact chain 校准。

### 3.3 协议可行性先于模型训练

对每个目标规模 \(N\)，先在 perfect-link、理想代表性抽样下标定：

\[
(k,\alpha,\beta,R_{\max}).
\]

必须满足：

\[
F_{\mathrm{safety}}^{\mathrm{floor}}\le\epsilon_s/10,
\qquad
F_{\mathrm{validity}}^{\mathrm{floor}}\le\epsilon_v/10,
\qquad
P(T_{\mathrm{all}}>T_d)^{\mathrm{floor}}\le\epsilon_d/10.
\]

若 perfect-link floor 不达标，停止模型训练并调整协议；不得让 GNN 补救不可行协议。

---

## 4. 新的可靠性与时延语义

设真实事实为 \(Y^\star\)，eligible honest nodes 为 \(\mathcal H\)，节点终态为 \(Y_i\in\{+,-,U\}\)。

### 4.1 Agreement safety

\[
\boxed{
F_{\mathrm{disagree}}
=
P\left(
\exists i,j\in\mathcal H:
Y_i\neq Y_j,\;
Y_i,Y_j\neq U
\right)
}
\]

### 4.2 Validity

\[
\boxed{
F_{\mathrm{wrong}}
=
P\left(
\exists i\in\mathcal H:
Y_i\neq Y^\star,\;
Y_i\neq U
\right)
}
\]

也可同时报告 all-wrong probability。

### 4.3 Deadline finality

\[
\boxed{
F_{\mathrm{deadline}}(T_d)
=
P(T_{\mathrm{all}}>T_d)
}
\]

其中

\[
T_{\mathrm{all}}
=
\inf\left\{
t:
\forall i\in\mathcal H,\;
Y_i(t)=Y^\star
\right\}.
\]

formal liveness 表示最终取得进展，不等价于工程 deadline。项目主指标应是 deadline finality 和尾确认时延。

### 4.4 优化目标

可靠性不再是可牺牲的 Pareto 轴。主问题为：

\[
\boxed{
\min_\theta
\left(
\operatorname{CVaR}_{q}(T_{\mathrm{all}}),
E_{\mathrm{network}}
\right)
}
\]

subject to

\[
F_{\mathrm{disagree}}(\theta)\le\epsilon_s,
\]

\[
F_{\mathrm{wrong}}(\theta)\le\epsilon_v,
\]

\[
F_{\mathrm{deadline}}(T_d;\theta)\le\epsilon_d.
\]

推荐：

\[
q=0.95\text{ 或 }0.99,
\quad
\epsilon_s\ll\epsilon_v\le10^{-3},
\quad
\epsilon_d\le10^{-2},
\]

具体阈值由应用场景确定。

### 4.5 可微 primal-dual 训练

\[
\mathcal L_\theta
=
\operatorname{CVaR}_{q}(T_{\mathrm{all}})
+
\lambda_E E
+
\mu_s(F_{\mathrm{disagree}}-\epsilon_s)
+
\mu_v(F_{\mathrm{wrong}}-\epsilon_v)
+
\mu_d(F_{\mathrm{deadline}}-\epsilon_d),
\]

\[
\mu_r
\leftarrow
\left[
\mu_r+\eta_\mu(F_r-\epsilon_r)
\right]_+.
\]

固定手工权重不能替代 dual update。只有满足约束的点才进入 D/E Pareto 比较。

---

## 5. 有效抽样动力学

### 5.1 Nominal query policy

节点 \(i\) 的 query-set policy 为

\[
\Pi_{\theta,i}(S\mid G_t),
\qquad
|S|=k.
\]

边 inclusion probability 为

\[
\pi_{ij}(t)=P_\theta(j\in S_i(t)).
\]

### 5.2 物理 thinning 与实际响应分布

完整 request-response 成功率为

\[
\ell_{ij}(t)
=
P(\text{valid response }j\to i).
\]

计划抽样与实际收到的证据不同。边的成功响应质量为

\[
r_{ij}(t)=\pi_{ij}(t)\ell_{ij}(t).
\]

归一化后的 response-conditioned marginal 为

\[
\boxed{
\widetilde\pi_{ij}(t)
=
\frac{
\pi_{ij}(t)\ell_{ij}(t)
}{
\sum_m\pi_{im}(t)\ell_{im}(t)+\varepsilon
}.
}
\]

Avalanche 实际看到的是 \(\widetilde\pi\)，不是 nominal \(\pi\)。

### 5.3 三元 query outcome

令 \(u_j(t)\)、\(v_j(t)\) 分别为节点 \(j\) 当前支持 correct/wrong 的条件概率，则

\[
p^+_{ij}(t)=\ell_{ij}(t)u_j(t),
\]

\[
p^-_{ij}(t)=\ell_{ij}(t)v_j(t),
\]

\[
p^0_{ij}(t)=1-p^+_{ij}(t)-p^-_{ij}(t).
\]

对 query-set distribution 求和后得到：

\[
h_i^+(t)=P(\text{correct quorum}),
\]

\[
h_i^-(t)=P(\text{wrong quorum}),
\]

\[
h_i^0(t)=1-h_i^+(t)-h_i^-(t).
\]

### 5.4 有效进展率

\[
\boxed{
g_i(t)=h_i^+(t)+h_i^-(t)
}
\]

表示一轮是否获得足以更新协议状态的 quorum。

若实际 round duration 为 \(\tau_i(t)\)，定义物理时间上的进展速率：

\[
\boxed{
\nu_i^{\mathrm{prog}}(t)
=
\frac{g_i(t)}{\tau_i(t)}.
}
\]

### 5.5 正确方向漂移

\[
\boxed{
\Delta_i(t)
=
h_i^+(t)-h_i^-(t)
}
\]

以及单位物理时间漂移：

\[
\boxed{
\nu_i^{\mathrm{drift}}(t)
=
\frac{\Delta_i(t)}{\tau_i(t)}.
}
\]

\(g_i\) 高但 \(\Delta_i\approx0\) 表示节点快速获得相互冲突的证据；仅优化 response rate 不足。

### 5.6 相关证据的有效样本量

设成功响应的归一化权重为 \(w_i\)，候选 peer 的证据相关矩阵为 \(R_i\)，其中 \(R_{jj}=1\)。定义：

\[
\boxed{
k_{\mathrm{eff},i}
=
\frac{1}{w_i^\top R_i w_i}.
}
\]

当 \(k\) 个响应等权且平均相关性为 \(\bar\rho\) 时：

\[
k_{\mathrm{eff}}
\approx
\frac{k}{1+(k-1)\bar\rho}.
\]

项目必须显式建模或估计 \(R_i\)，否则无法研究“抽样是否独立”。

### 5.7 Mixing 与 weak cut

构造 effective response kernel

\[
P^{\mathrm{resp}}_{ij}(t)=\widetilde\pi_{ij}(t).
\]

对区域 supergraph 计算：

- conductance；
- additive reversiblization spectral gap；
- cross-region response mass；
- query-set overlap。

这些是解释性指标和辅助训练信号，不替代最终 safety/deadline 指标。

### 5.8 推荐的辅助损失

\[
\mathcal L_{\mathrm{aux}}
=
\lambda_{\mathrm{prog}}
\operatorname{CVaR}_{0.05}[-\nu^{\mathrm{prog}}]
+
\lambda_{\mathrm{drift}}
\operatorname{CVaR}_{0.05}[-\nu^{\mathrm{drift}}]
+
\lambda_{\mathrm{ess}}
\operatorname{CVaR}_{0.05}[-k_{\mathrm{eff}}]
+
\lambda_{\mathrm{mix}}
[\gamma_{\mathrm{target}}-\gamma_{\mathrm{mix}}]_+
+
\lambda_{\mathrm{load}}
\operatorname{CVaR}_{0.95}(\Lambda).
\]

这些项只能作为机制辅助；论文 headline 仍以真实 safety、deadline、latency、energy 为准。

---

## 6. 仿真环境必须新增的证据模型

当前项目仅用统一 `initial_correct_preference`，没有真实事实、局部观测和相关错误来源。这样无法研究抽样代表性与证据多样性。

### 6.1 Ground truth

每个事件有真实二值状态：

\[
Y^\star\in\{+1,-1\}.
\]

### 6.2 区域共享错误

将道路划分为 road segment / intersection / visibility region，令节点 \(i\) 所属区域为 \(g(i)\)。

一种可解释模型为：

\[
O_i
=
Y^\star
\oplus
B_{g(i)}
\oplus
E_i,
\]

其中

\[
B_g\sim\operatorname{Bernoulli}(p_g)
\]

是区域级共享误差，

\[
E_i\sim\operatorname{Bernoulli}(p_i)
\]

是节点独立误差。

由此自然产生同一区域节点的 opinion correlation。

### 6.3 需要覆盖的场景

- 局部遮挡导致一片车辆共同误判；
- 同一传感器或地图源导致共因错误；
- 两个道路区域初始意见相反；
- weak cut 连接两个意见簇；
- hub receiver 的拥塞与共享丢包；
- 车辆移动导致相关区域变化；
- 无相关误差的理想控制组。

query policy 不得读取 \(Y^\star\) 或 peer 当前 vote；只能使用部署时可观测的几何、历史响应、可信度来源、道路区域和物理特征。

---

## 7. 仿真物理链的重新设计

### 7.1 两张物理图

必须区分：

1. **通信候选图** \(G_{\mathrm{comm}}\)：允许查询的 intended links；
2. **干扰传播图** \(G_{\mathrm{int}}\)：任一 active transmitter 对任一 susceptible receiver 的干扰路径。

当前按 intended destination 聚合 interference 会漏掉发向其他接收者、但仍干扰 \(j\) 的发射。

### 7.2 逐轮闭环

canonical analytic episode 必须执行：

\[
X_t
\rightarrow
\tau_t
\rightarrow
\Pi_\theta
\rightarrow
\Lambda_t
\rightarrow
\gamma_t,\ell_t
\rightarrow
(h_t^+,h_t^-,h_t^0)
\rightarrow
X_{t+1}.
\]

不得再使用固定 `tau_proxy`。

### 7.3 每轮步骤

1. 从协议状态计算 active/transient probability；
2. 根据 query policy 计算 intended request activity；
3. 计算 source transmission activity；
4. 在 \(G_{\mathrm{int}}\) 上计算 co-channel interference；
5. 计算 Mode-2 collision；
6. 计算 half-duplex blocking；
7. 分别计算 request 和 response 的 SINR/FBL/HARQ；
8. 计算 queueing delay 和 drop；
9. 得到完整 poll success \(\ell_{ij}(t)\)；
10. 计算 quorum distribution；
11. 更新真实 Snowball state；
12. 累积 wall-clock delay 和 total energy。

### 7.4 Request/response 必须分开

\[
\ell^{\mathrm{poll}}_{ij}(t)
=
(1-p^{\mathrm{col,req}}_{ij})
(1-p^{\mathrm{HD,req}}_{ij})
(1-\epsilon^{\mathrm{req}}_{ij})
(1-p^{\mathrm{col,resp}}_{ji})
(1-p^{\mathrm{HD,resp}}_{ji})
(1-\epsilon^{\mathrm{resp}}_{ji}).
\]

request 和 response 的发送节点、接收节点、干扰与队列不同，不能复用同一个 \(\gamma\)。

### 7.5 Headline 固定资源

主研究只优化 query topology。headline 中：

- transmit power 固定；
- MCS/blocklength 固定；
- HARQ profile 固定；
- subchannel pool 固定或作为场景参数。

per-node power/blocklength heads 移至 extension。这样性能变化才能归因于 topology。

---

## 8. Monte-Carlo 的正确角色

Monte-Carlo 不进入反向训练，但必须是独立裁判。

### 8.1 四级验证

1. **局部数学 reference**
   - subset/quorum brute force；
   - FBL scipy reference。
2. **small-\(N\) exact global chain**
   - \(N\le 8\)；
   - 枚举联合协议状态。
3. **independent dynamic MC**
   - 逐轮抽 query subset；
   - 逐轮抽资源、fading、request/response；
   - 使用真实 sampled peer state；
   - 推进真实 Snowball counters。
4. **rare-event estimation**
   - importance sampling；
   - splitting/subset simulation；
   - common random numbers。

### 8.2 Analytic evaluator 的角色

- 训练和梯度搜索；
- 快速筛选 topology；
- 输出有效抽样动力学 diagnostics。

### 8.3 Dynamic MC 的角色

- 校准绝对 safety/deadline；
- 验证 topology ranking；
- 检查 analytic gradient direction；
- 发布最终 headline 结果。

禁止“解析模型产生 marginals，再从这些 marginals 抽样”作为外部验证。

---

## 9. 新模型创新：ESD-GNN + CDQ

建议废弃 `Global-Risk-Budgeted Hierarchical Query GNN` 作为核心。其“global-risk budget”建立在旧 global product failure 上，不再符合新问题。

保留的只有“多尺度结构表示”思想，用于识别 weak cuts 与相关区域，而不是分配 global-risk budget。

### 9.1 核心场景挑战

最困难且最有研究价值的矛盾是：

\[
\boxed{
\text{局部高交付质量}
\quad\text{vs}\quad
\text{全局证据多样性与混合}
}
\]

最近、SINR 最高的 peers 往往：

- 属于同一道路区域；
- 共享相同遮挡和传感器错误；
- 被大量节点共同选择；
- 形成 receiver congestion；
- 对全局意见不具代表性。

当前 product-weighted \(k\)-subset law

\[
P(S)\propto\prod_{j\in S}a_j
\]

只有独立质量分数，没有 pairwise diversity，无法区分“5 个高质量但完全重复的证据”和“5 个略弱但互补的证据”。

### 9.2 模型名称

> **ESD-GNN：Effective-Sampling Dynamics Graph Neural Network**

核心数学层：

> **CDQ：Correlation-Aware Determinantal Query layer**

这是 SOTA-oriented 设计候选；是否达到 SOTA 必须由能力匹配基线和大规模实验决定，不能预先声称。

### 9.3 多图、多尺度编码器

模型同时编码：

1. \(G_{\mathrm{comm}}\)：query 可达性；
2. \(G_{\mathrm{int}}\)：资源/干扰关系；
3. \(G_{\mathrm{corr}}\)：证据与 shadow 相关关系；
4. \(G_{\mathrm{region}}\)：road segment / intersection supergraph。

每层包含：

- source-side candidate competition aggregation；
- destination-side incoming-load aggregation；
- evidence-correlation message passing；
- vehicle↔region cross-attention；
- residual + normalization。

模型必须能在有限层数内看见 weak cut 和 region-level evidence imbalance；普通两层 candidate-graph mean aggregation不够。

### 9.4 CDQ 的 k-DPP query law

对节点 \(i\) 的每个候选 peer \(j\)，模型输出：

- quality scalar \(q_{ij}>0\)；
- diversity embedding \(b_{ij}\in\mathbb R^r\)。

构造 PSD kernel：

\[
L_i=B_iB_i^\top,
\qquad
B_i[j,:]=\sqrt{q_{ij}}\,b_{ij}.
\]

要求 \(r\ge k\)。

query subset 的概率为：

\[
\boxed{
P_i(S)
=
\frac{
\det(L_{i,S})
}{
e_k(\lambda(L_i))
},
\qquad
|S|=k.
}
\]

解释：

- row norm 表示 peer quality；
- row angle 表示 peer similarity；
- 相似 peers 使 determinant 变小；
- 互补 peers 张成更大 volume，概率更高。

当前 ESP product law 是 diagonal-kernel k-DPP 的特殊情况：

\[
L_i=\operatorname{diag}(a_i)
\Rightarrow
\det(L_{i,S})=\prod_{j\in S}a_{ij}.
\]

因此 CDQ 是当前数学的严格推广，而不是不相关的新模块。

### 9.5 新的精确 quorum 生成式

定义：

\[
g_{ij}(x,y)
=
p^0_{ij}
+
p^+_{ij}x
+
p^-_{ij}y.
\]

令 \(D_g=\operatorname{diag}(g_{ij})\)。由 principal-minor expansion：

\[
\det(I+zL_iD_g)
=
\sum_{S}
z^{|S|}
\det(L_{i,S})
\prod_{j\in S}g_{ij}(x,y).
\]

因此：

\[
\boxed{
P_i(m,n)
=
\frac{
[z^k x^m y^n]
\det(I+zL_iD_g)
}{
e_k(\lambda(L_i))
}.
}
\]

这是 k-DPP query 下异质 correct/wrong/no-response quorum 的精确概率。

若 \(L_i=B_iB_i^\top\)，由 matrix determinant lemma：

\[
\det(I_d+zB_iB_i^\top D_g)
=
\det(I_r+zB_i^\top D_gB_i).
\]

因此 cubic 代价落在小 rank \(r\)，而不是候选度 \(d_i\)。

目标复杂度：

\[
O\left(
Er^2
+
N\operatorname{poly}(r,k)
\right)
\]

每轮；固定小 \(r,k\) 时对 \(N,E\) 近线性。

实现时需使用截断多项式矩阵和 Berkowitz/Bareiss 类无除法 determinant algorithm，并以 brute-force subset enumeration 验证。

### 9.6 Dynamics-in-the-loop refinement

ESD-GNN 可进行 \(L=2\sim3\) 次共享参数 refinement：

1. 生成初始 CDQ kernel；
2. 计算 \(\nu^{\mathrm{prog}},\nu^{\mathrm{drift}},k_{\mathrm{eff}},\Lambda,\gamma_{\mathrm{mix}}\)；
3. 将这些量作为 analytic feedback；
4. 更新 quality/diversity embeddings。

\[
h^{(\ell+1)}
=
h^{(\ell)}
+
\phi_\theta
\left(
h^{(\ell)},
\nu^{\mathrm{prog}},
\nu^{\mathrm{drift}},
k_{\mathrm{eff}},
\Lambda,
\gamma_{\mathrm{mix}}
\right).
\]

训练与部署执行相同 refinement 次数，保持 end-to-end 和无 train/deploy mismatch。

### 9.7 Temporal 模型

暂不以 emission 为核心。静态主线通过后，再加入 contractive temporal memory：

- region evidence-correlation state；
- receiver congestion state；
- observed response rate；
- weak-cut persistence。

允许使用 GRU/SSM，但必须：

- 明确 spectral/contractive control；
- 与 no-memory、scalar emission、shuffled memory 比较；
- 在完整 dynamic environment 上证明收益。

---

## 10. 强制基线与消融

### 10.1 Query-policy 基线

- uniform distinct-peer；
- distance/SINR/link-success；
- load-balanced heuristic；
- conductance/bridge-aware heuristic；
- current ESP product policy；
- DPP with hand-crafted correlation；
- direct per-scene logit optimizer；
- edge MLP；
- current two-layer GNN；
- ESD-GNN without CDQ；
- full ESD-GNN + CDQ。

### 10.2 机制消融

- 去掉 evidence-correlation graph；
- 去掉 interference graph；
- 去掉 region pooling；
- DPP→diagonal DPP（恢复 ESP）；
- 去掉 refinement；
- 去掉 drift auxiliary；
- 去掉 mixing auxiliary；
- fixed vs learned diversity embedding；
- topology-only vs optional resource-control extension。

### 10.3 评价规模

- exactness：\(N=4\sim8\)；
- development：\(N=32,64\)；
- training：\(N=100,300,1000\)；
- headline：\(N=100,300,1000,3000\)；
- scale：\(N=5000,10000\)。

不得以 \(N=8,10,12\) 作为性能 headline。

---

## 11. 新测试与开发 gate

### G0 Canonical execution closure

- 所有实验只调用 `run_consensus_episode`；
- runtime trace 显示全部机制；
- sentinel test 证明机制被调用；
- 未使用配置字段报错。

### G1 Protocol semantics

- 当前状态机与选定协议伪代码逐步一致；
- true Snowball confidence update 有 reference；
- small trace exact test。

### G2 Correlated evidence environment

- 区域相关矩阵与理论值一致；
- zero-correlation control 恢复独立情况；
- policy 不读取 ground truth/vote label。

### G3 Round-coupled physics

- transient mass 改变 load；
- interference graph包含非 intended transmitter；
- request/response、collision、half-duplex、queue、HARQ均激活；
- 每个机制有因果方向测试。

### G4 CDQ exactness

- k-DPP normalization；
- inclusion marginals；
- exact sampler；
- 与 brute force 误差 \(<10^{-10}\)。

### G5 Determinantal quorum exactness

- determinant coefficient law；
- 与子集×三元 outcome 枚举误差 \(<10^{-10}\)；
- gradient relative error \(<10^{-4}\)。

### G6 Dynamic MC independence

- MC 不调用 analytic terminal marginals；
- small-\(N\) exact chain、analytic、dynamic MC 三方比较；
- common random numbers 生效。

### G7 Effective sampling diagnostics

- response-conditioned distribution；
- progress/drift；
- ESS；
- mixing；
- load；
- 手工场景方向全部正确。

### G8 Protocol feasibility

- perfect-link floor 在目标 \(N\) 和安全阈值下通过；
- 不通过则停止训练。

### G9 Model mechanism

- full model 的增益能被 CDQ/region/load ablation 分解；
- topology-only 优势明确；
- 多训练 seeds。

### G10 Large-scale performance

- \(N=100\sim10000\)；
- 不出现 \(N^2\)；
- runtime/memory 对 \(E\) 近线性。

### G11 Reliability-constrained superiority

只比较满足 safety/validity/deadline constraints 的政策。在可行域内比较：

- \(D_{95},D_{99},\operatorname{CVaR}_{0.99}\)；
- energy；
- feasibility rate；
- paired dynamic-MC significance。

### G12 Temporal robustness

完整时序环境下评估：

- topology churn；
- shadow correlation；
- sudden weak-cut；
- recovery time；
- hidden-state stability；
- temporal generalization。

---

## 12. 允许和禁止的 claim

### 允许

- exact k-DPP subset probability；
- exact heterogeneous quorum probability under the defined CDQ law；
- differentiable analytic effective-sampling model；
- calibrated analytic surrogate；
- dynamic-MC-validated ranking；
- reliability-constrained latency/energy improvement。

### 禁止

- unrestricted exact Avalanche global reliability；
- 把 Snowflake streak automaton称为 Snowball；
- 用 \(Q=1\) product 作为真实 global failure；
- 用单元测试替代 dynamic protocol validation；
- 用小 complete graph 证明 V2X scalability；
- 将 power/blocklength 优势归因于 topology；
- 将实现但未进入 canonical path 的机制写入贡献；
- 在没有多训练 seed、能力匹配 baseline 时声称 SOTA。

---

## 13. 建议的新项目定位

推荐工作标题：

> **Reliability-Constrained Effective-Sampling Topology Control for Snowball Consensus over NR-V2X**

核心贡献假设：

1. 将无线拓扑作用表述为有效抽样动力学；
2. 提出 correlation-aware k-DPP query policy；
3. 推导 k-DPP 下异质三元 quorum 的 determinant generating law；
4. 设计多图、多尺度 ESD-GNN；
5. 在 safety/validity constraints 下优化尾确认时延和能耗；
6. 用独立 dynamic MC 校准和报告最终结果。

---

## 14. 参考方向

- Avalanche/Snowball 原始协议；
- *An Analysis of Avalanche Consensus*（2024）；
- *Quantifying Liveness and Safety of Avalanche's Snowball*（2024）；
- *Frosty: Bringing Strong Liveness Guarantees to the Snow Family*（2024）；
- *Determinantal Learning for Subset Selection in Wireless Networks*（2025）；
- 近期 V2X GNN/DRL resource-allocation 工作；
- DPP 与低秩 k-DPP 计算；
- graph conductance、spectral mixing 与 oversquashing/rewiring 工作。

这些工作为问题背景和基础工具提供参照；本项目的潜在新意应落在“相关证据、无线 thinning、Snowball quorum 与 topology GNN 的统一”上。
