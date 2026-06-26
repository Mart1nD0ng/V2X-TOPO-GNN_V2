# V2X-TOPO-GNN：宏观共识首达目标与 CDQ 2.0 技术规范

## 0. 文档目的

本文档统一下一轮项目重构所需的数学定义。它替换以下旧主线：

- 以 \(P(\exists i:Y_i\neq C)\) 或 \(1-\prod_i c_i\) 表示跨规模共识失败；
- 以 \(F/D/E\) 三目标平权 Pareto 作为安全系统的主优化问题；
- 用固定 eligible set 解决共识作用范围；
- 将低秩 \(BB^\top\) CDQ 与一般 ESP 视为严格嵌套；
- 将 polling round 的物理通信语义、deadline 与失败阈值彼此分开定义。

新的主线是：

\[
\boxed{
\text{外生参与测度}
\rightarrow
\text{宏观共识 basin 首达}
\rightarrow
\text{可靠性约束}
\rightarrow
\text{有效抽样动力学}
\rightarrow
\text{尾确认时延与能耗优化}
}
\]

项目继续保留端到端属性：

\[
\theta
\rightarrow
\Pi_\theta(S_i)
\rightarrow
\text{round-coupled PHY}
\rightarrow
\text{Snowball dynamics}
\rightarrow
\text{basin outcomes}
\rightarrow
\mathcal L.
\]

对任意组件的 exactness claim 必须注明其随机模型边界；独立 dynamic Monte-Carlo 是最终裁判。

---

# 1. 为什么废弃“任一节点失败”的主指标

在假设节点独立且边缘失败概率相同为 \(p\) 时，

\[
1-(1-p)^N
\]

确实等于至少一个节点失败的概率。问题不是代数，而是：

1. 该事件不是 Avalanche/Snowball 最自然的宏观共识事件；
2. 节点终态并不独立；
3. “存在任一失败节点”的事件必然随 \(N\) 单调扩大；
4. 它会把跨规模比较变成节点数量效应，而不是共识动力学比较。

因此主指标不能继续定义为

\[
P(\exists i:Y_i\neq C).
\]

严格的任意节点 disagreement 可继续作为固定规模安全审计指标，但不能作为跨规模主优化量。

---

# 2. 用外生参与测度替代 hard eligible set

完全取消作用范围是不可能的：任何共识实例都必须说明哪些节点的决定对该事件有意义。为避免人工 hard set，定义归一化参与测度

\[
\omega_i\ge0,
\qquad
\sum_{i=1}^{N}\omega_i=1.
\]

允许的设计包括：

### 全网均匀实例

\[
\omega_i=\frac1N.
\]

### 事件相关软范围

\[
\omega_i
=
\frac{
K_{\mathrm{app}}(d_i,\mathrm{TTC}_i,\mathrm{role}_i)
}{
\sum_j K_{\mathrm{app}}(d_j,\mathrm{TTC}_j,\mathrm{role}_j)
}.
\]

约束：

1. \(\omega\) 由应用或场景外生给定；
2. query policy 不得修改 \(\omega\)；
3. 训练、验证和部署使用同一作用范围规则；
4. \(\omega\) 不得使用 simulator-only truth；
5. 必须同时报告均匀测度和应用测度的敏感性。

---

# 3. 宏观共识状态

在 polling epoch \(r\) 结束后，定义：

\[
C_r
=
\sum_i\omega_i
\mathbf 1\{i\text{ 已正确最终化}\},
\]

\[
W_r
=
\sum_i\omega_i
\mathbf 1\{i\text{ 已错误最终化}\},
\]

\[
U_r=1-C_r-W_r.
\]

分别表示正确最终化质量、错误最终化质量和尚未最终化质量。

这些量属于 \([0,1]\)，不会通过节点事件并集与 \(N\) 机械耦合。节点数量仍会影响有限规模波动和动力学，但不会直接进入一个 \(1-(1-p)^N\) 乘积。

---

# 4. 正确、错误、分裂和 deadline basin

选定：

\[
\rho_f>\frac12,
\qquad
1-\rho_f<\rho_s<\frac12.
\]

定义：

\[
\mathcal B_C=\{C_r\ge\rho_f\},
\]

\[
\mathcal B_W=\{W_r\ge\rho_f\},
\]

\[
\mathcal B_S=
\{C_r\ge\rho_s,\;W_r\ge\rho_s\}.
\]

条件

\[
\rho_s>1-\rho_f
\]

确保正确/错误 basin 与 split basin 不会重叠。

首达时间为：

\[
\tau_C=\inf\{r:(C_r,W_r,U_r)\in\mathcal B_C\},
\]

\[
\tau_W=\inf\{r:(C_r,W_r,U_r)\in\mathcal B_W\},
\]

\[
\tau_S=\inf\{r:(C_r,W_r,U_r)\in\mathcal B_S\}.
\]

令最大 polling epochs 为 \(R_d\)。定义互斥运行结果：

\[
P_C
=
P\!\left(
\tau_C<
\min(\tau_W,\tau_S,R_d+1)
\right),
\]

\[
\boxed{
F_{\mathrm{wrong}}
=
P\!\left(
\tau_W<
\min(\tau_C,\tau_S,R_d+1)
\right),
}
\]

\[
\boxed{
F_{\mathrm{split}}
=
P\!\left(
\tau_S<
\min(\tau_C,\tau_W,R_d+1)
\right),
}
\]

\[
\boxed{
F_{\mathrm{deadline}}
=
P\!\left(
\min(\tau_C,\tau_W,\tau_S)>R_d
\right).
}
\]

在确定 tie-breaking 后：

\[
P_C+F_{\mathrm{wrong}}+F_{\mathrm{split}}+F_{\mathrm{deadline}}=1.
\]

## 连续 disagreement 诊断量

\[
D_{\mathrm{pair}}(r)
=
\frac{
2C_rW_r
}{
(C_r+W_r)^2+\varepsilon
}.
\]

它表示从已最终化质量中独立抽取两个参与者时决定不同的概率。

区域分歧可写为：

\[
D_{\mathrm{region}}(r)
=
\sum_g
\Omega_g
\left[
(C_{g,r}-C_r)^2+(W_{g,r}-W_r)^2
\right].
\]

## 固定规模严格安全审计

保留：

\[
F_{\mathrm{strict}}
=
P(\exists i,j:
Y_i\neq Y_j,\;
Y_i,Y_j\neq U).
\]

该指标用于固定 \(N\) 的协议安全审计，不作为跨规模主优化指标。

---

# 5. Polling epoch 与物理通信语义

## 5.1 统一的一轮定义

对每个未最终化节点 \(i\)，epoch \(r\)：

1. 从 query policy 中抽取恰好 \(k\) 个不同 peers；
2. 对这 \(k\) 个 peers 发起并行 unicast request-response polls；
3. 每个 peer 返回其当前 preference；
4. 只统计在 polling window \(\Delta_{\mathrm{poll}}\) 内完成且正确解码的响应；
5. 超时、collision、half-duplex、HARQ 未完成均记为 no-response；
6. window 结束时计算 quorum；
7. 更新 Snowball confidence、preference、streak；
8. 满足最终化条件则在 epoch 末吸收。

最终化节点：

- 不再主动发起新 polls；
- 继续响应其他节点；
- 返回其最终 preference。

解析模型采用同步 epoch abstraction；独立 dynamic MC 应验证更细粒度异步实现与该 abstraction 的差异。

## 5.2 轮内有效 poll 概率

\[
\boxed{
\ell_{ij}^{(r)}(\Delta_{\mathrm{poll}})
=
P\left(
T_{ij}^{\mathrm{req}}
+
T_{ji}^{\mathrm{resp}}
\le
\Delta_{\mathrm{poll}},
\text{ request/response 均正确解码}
\right).
}
\]

它必须包含：

- request FBL；
- response FBL；
- finite HARQ；
- resource collision；
- half-duplex；
- queueing；
- interference；
- poll-window timeout。

## 5.3 Deadline 与 epoch 数

若业务截止时间为 \(T_d\)：

\[
R_d
=
\left\lfloor
\frac{T_d}{\Delta_{\mathrm{poll}}}
\right\rfloor.
\]

若 window 可变：

\[
T_{\mathrm{confirm}}
=
\sum_{r=1}^{\tau_C}
\Delta_{\mathrm{poll}}^{(r)}.
\]

## 5.4 Source/destination 责任

对 edge \(i\to j\)：

| 量 | 所属端 |
|---|---|
| request TX energy | source \(i\) |
| request RX/queue | destination \(j\) |
| response TX energy | destination \(j\) |
| response RX energy | source \(i\) |
| poller epoch completion | source \(i\) |
| receiver congestion | destination \(j\) |

request activity：

\[
a_{ij}^{\mathrm{req}}(r)
=
u_i(r)\pi_{ij}(r).
\]

source activity：

\[
A_i^{\mathrm{req}}
=
\sum_j a_{ij}^{\mathrm{req}}
=
k\,u_i.
\]

receiver addressed load：

\[
\Lambda_j^{\mathrm{req}}
=
\sum_i a_{ij}^{\mathrm{req}}.
\]

response activity：

\[
a_{ji}^{\mathrm{resp}}
=
a_{ij}^{\mathrm{req}}
\ell_{ij}^{\mathrm{req}}.
\]

## 5.5 Collision 排除 desired transmission

\[
L_{j,-ij}^{\mathrm{req}}
=
L_j^{\mathrm{req}}
-
a_{ij}^{\mathrm{req}},
\]

\[
p_{ij}^{\mathrm{col,req}}
=
1-
\left(
1-\frac1{S_{\mathrm{eff}}}
\right)^{
L_{j,-ij}^{\mathrm{req}}
}.
\]

单一活动传输的 collision probability 必须严格为 0。response 同理。

---

# 6. 优化目标

可靠性是硬约束，不再作为与时延、能耗平权交换的 Pareto 轴。

\[
\boxed{
\min_\theta
\left[
\operatorname{CVaR}_{q}
\left(
T_{\mathrm{confirm}}\mid O=C
\right)
+
\lambda_E E_{\mathrm{network}}
\right]
}
\]

subject to：

\[
F_{\mathrm{wrong}}\le\epsilon_w,
\]

\[
F_{\mathrm{split}}\le\epsilon_s,
\]

\[
F_{\mathrm{deadline}}\le\epsilon_d.
\]

推荐通过 primal-dual 更新：

\[
\mathcal L_\theta
=
\operatorname{CVaR}_{q}(T_{\mathrm{confirm}}\mid O=C)
+
\lambda_EE
+
\mu_w(F_{\mathrm{wrong}}-\epsilon_w)
+
\mu_s(F_{\mathrm{split}}-\epsilon_s)
+
\mu_d(F_{\mathrm{deadline}}-\epsilon_d),
\]

\[
\mu_x
\leftarrow
\left[
\mu_x+\eta_\mu(F_x-\epsilon_x)
\right]_+.
\]

阈值不能由节点数量机械推导，应由 service hazard analysis 决定。

推荐统一配置：

```text
ConsensusServiceProfile
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

---

# 7. 有效抽样动力学

query-set policy：

\[
\Pi_{\theta,i}(S\mid G_r),
\qquad |S|=k.
\]

边 inclusion probability：

\[
\pi_{ij}(r)=P(j\in S_i(r)).
\]

response-conditioned marginal：

\[
\widetilde\pi_{ij}(r)
=
\frac{
\pi_{ij}(r)\ell_{ij}(r)
}{
\sum_m\pi_{im}(r)\ell_{im}(r)+\varepsilon
}.
\]

令邻居正确、错误 preference probabilities 为 \(u_j(r),v_j(r)\)：

\[
p_{ij}^{+}(r)=\ell_{ij}(r)u_j(r),
\]

\[
p_{ij}^{-}(r)=\ell_{ij}(r)v_j(r),
\]

\[
p_{ij}^{0}(r)=1-p_{ij}^{+}(r)-p_{ij}^{-}(r).
\]

经过 query law 的异质 quorum 计算：

\[
h_i^+(r)=P(\text{correct quorum}),
\]

\[
h_i^-(r)=P(\text{wrong quorum}),
\]

\[
h_i^0(r)=1-h_i^+(r)-h_i^-(r).
\]

## 有效进展率

\[
g_i(r)=h_i^+(r)+h_i^-(r),
\]

\[
\nu_i^{\mathrm{prog}}(r)
=
\frac{g_i(r)}{\Delta_{\mathrm{poll}}}.
\]

## 正确方向漂移

\[
\Delta_i(r)=h_i^+(r)-h_i^-(r),
\]

\[
\nu_i^{\mathrm{drift}}(r)
=
\frac{\Delta_i(r)}{\Delta_{\mathrm{poll}}}.
\]

## 相关证据有效样本量

设成功响应权重为 \(w_i\)，证据相关矩阵为 \(R_i\)：

\[
k_{\mathrm{eff},i}
=
\frac1{w_i^\top R_i w_i}.
\]

这些量作为辅助训练信号和解释性指标，不替代 basin outcome。

---

# 8. ESP 与 CDQ 的数学关系

## 8.1 ESP

对 candidate quality weights \(a_j>0\)，固定选择 \(k\) 个 peers：

\[
\boxed{
P_{\mathrm{ESP}}(S)
=
\frac{
\prod_{j\in S}a_j
}{
e_k(a)
},
\qquad |S|=k.
}
\]

ESP 只表达独立 item quality，不表达 pairwise similarity。

## 8.2 一般 CDQ / k-DPP

\[
\boxed{
P_{\mathrm{CDQ}}(S)
=
\frac{
\det(L_S)
}{
e_k(\lambda(L))
},
\qquad
L\succeq0,\ |S|=k.
}
\]

一般 full-rank k-DPP 包含 ESP：令

\[
L=\operatorname{diag}(a)
\]

即可恢复 ESP。

## 8.3 为什么旧 low-rank CDQ 不包含一般 ESP

若旧实现约束：

\[
L=BB^\top,
\qquad
B\in\mathbb R^{d\times r},
\qquad
r<d,
\]

则：

\[
\operatorname{rank}(L)\le r.
\]

而一般 ESP 的

\[
\operatorname{diag}(a_1,\ldots,a_d)
\]

通常 rank 为 \(d\)。因此旧 low-rank CDQ 的函数族并不包含一般 ESP。

---

# 9. CDQ 2.0：严格包含 ESP 的查询族

CDQ 本身采用嵌套设计，但这只是 CDQ 的具体设计选择，不是所有组件的通用开发合同。

令：

\[
D=\operatorname{diag}(a_1,\ldots,a_d),
\qquad
a_j=\exp(s_j)>0.
\]

模型输出 unit-normalized diversity embeddings：

\[
\bar z_j
=
\frac{z_j}{\|z_j\|+\varepsilon}.
\]

组成：

\[
Z=
[\bar z_1^\top;\ldots;\bar z_d^\top].
\]

定义：

\[
\boxed{
L
=
D^{1/2}
\left(
I+\eta ZZ^\top
\right)
D^{1/2},
\qquad
\eta\ge0.
}
\]

性质：

### ESP 精确退化

\[
\eta=0
\Rightarrow
L=D
\Rightarrow
P_{\mathrm{CDQ}}=P_{\mathrm{ESP}}.
\]

### Quality 与 diversity 分离

对 subset \(S\)：

\[
\boxed{
\det(L_S)
=
\left(
\prod_{j\in S}a_j
\right)
\det\left(
I+\eta Z_SZ_S^\top
\right).
}
\]

第一项是 ESP quality，第二项是 diversity correction。

对 \(k=2\)、单位向量 \(z_j,z_l\)：

\[
\det(I+\eta Z_SZ_S^\top)
=
(1+\eta)^2
-
\eta^2(z_j^\top z_l)^2.
\]

候选越相似，joint selection weight 越低。

### 自适应回退

\[
\eta_i
=
\eta_{\max}\sigma(\xi_i)
\]

或：

\[
\eta_i=\operatorname{softplus}(\xi_i).
\]

初始化为 \(\eta\approx0\)，使模型从 ESP 开始学习；环境不需要 diversity 时可以保持接近 ESP。

---

# 10. CDQ 下的异质 quorum 生成式

令：

\[
g_j(x,y)
=
p_j^0+p_j^+x+p_j^-y,
\]

\[
G(x,y)=\operatorname{diag}(g_j(x,y)).
\]

principal-minor identity 给出：

\[
\det(I+zLG)
=
\sum_S
z^{|S|}
\det(L_S)
\prod_{j\in S}g_j(x,y).
\]

因此：

\[
\boxed{
P(m,n)
=
\frac{
[z^kx^my^n]\det(I+zLG)
}{
e_k(\lambda(L))
}.
}
\]

它是 CDQ query law 下 heterogeneous correct/wrong/no-response quorum 的精确概率。

对 CDQ 2.0，可使用 determinant lemma：

\[
\det(I+zLG)
=
\det(I+zDG)
\det
\left[
I_r+
\eta z
Z^\top
D^{1/2}G
(I+zDG)^{-1}
D^{1/2}Z
\right].
\]

固定小 \(r,k\) 时，目标复杂度应对 candidate edges 近线性。

---

# 11. 相关性训练目标

环境必须提供：

- 真实可控的相关结构；
- 部署可观测的 correlation proxies；
- 可达 CDQ 参数的相关性梯度。

令：

\[
R_{jl}
=
\operatorname{Corr}(\text{peer }j,\text{peer }l\text{ 的证据错误}).
\]

对 fixed-size query set \(S_i\)，定义 pairwise inclusion：

\[
\pi^{(2)}_{i,jl}
=
P(j,l\in S_i).
\]

精确期望相关性成本：

\[
\boxed{
\mathcal L_{\mathrm{corr}}
=
\sum_i
\sum_{j<l}
\pi^{(2)}_{i,jl}R_{jl}.
}
\]

禁止：

- `float(tensor)`；
- `.item()`；
- detached correlation score；

出现在训练路径中。

环境必须在保持 marginal correctness、geometry 和 link-quality 分布匹配的条件下，仅改变 correlation，才能识别 CDQ 机制。

---

# 12. 训练用解析模型与最终验证

宏观 first-hitting outcome 是联合路径事件。任意图上的精确联合计算在大规模下不可行。

因此采用三级体系：

## Level 1：small-\(N\) exact reference

对 \(N\le8\) 构造联合协议状态链，验证 basin outcomes。

## Level 2：可微 analytic training surrogate

- true Snowball node-state recurrence；
- correlated latent factors；
- differentiable macrostate occupancy/hazard approximation；
- 明确标注 approximation；
- 输出有效抽样动力学 diagnostics。

不得再声称 unrestricted exact global Avalanche probability。

## Level 3：independent dynamic MC

逐轮抽取：

- evidence；
- query subset；
- request/response；
- PHY failures；
- peer actual state；
- Snowball update；
- basin first-hitting outcome。

最终 headline 的 wrong/split/deadline probabilities 来自 dynamic MC 或 rare-event MC。

---

# 13. Mechanism Identifiability Contract

所有组件必须遵守独立文档：

`MECHANISM_IDENTIFIABILITY_CONTRACT.md`

合同只包含：

1. 环境结构存在性；
2. 部署可观测性；
3. 梯度可达性；
4. 因果对照；
5. 主链一致性。

不要求每个复杂组件都包含简单组件，也不要求在实现前通过 oracle 获益测试。CDQ 严格包含 ESP 是 CDQ 的具体数学选择，而不是通用合同条款。
