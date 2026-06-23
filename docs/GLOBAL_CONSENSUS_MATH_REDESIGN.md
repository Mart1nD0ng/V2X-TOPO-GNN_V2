# V2X-TOPO-GNN：全局共识数学重构规范

## 0. 目的与适用范围

本文档给出下一代主线所需的数学定义，重点替换以下旧设计：

- 以节点边缘失败率均值代替全局共识失败概率；
- 固定 degree-\(K\) / hard top-\(k\) 拓扑；
- iid-with-replacement 的 beta-tail quorum 与 mean-field 节点边缘闭包；
- 缺失 channel dispersion 的 short-blocklength 代理；
- 高度同构的 delay / energy 代理。

新主线必须满足：

\[
\theta
\longrightarrow
\text{query/resource policy}
\longrightarrow
\Lambda
\longrightarrow
\gamma
\longrightarrow
\ell
\longrightarrow
\text{global consensus law}
\longrightarrow
(F,D,E)
\longrightarrow
\mathcal L .
\]

全文所称“精确”必须明确限定到所定义的随机模型。对不受限制的真实 Avalanche 联合过程，状态空间一般随 \(N\) 指数增长；任何多项式复杂度结果都必须显式声明其结构假设。

---

# 1. 严格定义全局共识失败概率

设 eligible honest node set 为 \(\mathcal H\)。静态场景通常取 \(\mathcal H=V\)；时序场景必须预先定义加入时间、离开时间与 grace period，不能在评价后动态改变集合。

节点 \(i\) 在截止轮次 \(R_{\max}\) 的终态为

\[
Y_i\in\{C,W,U\},
\]

其中：

- \(C\)：截止时间前正确完成；
- \(W\)：截止时间前错误完成；
- \(U\)：截止时间仍未完成。

定义全局成功事件

\[
\mathcal S_C=\{\forall i\in\mathcal H:Y_i=C\},
\qquad
S_C=P(\mathcal S_C).
\]

论文和项目中的主失败指标固定为

\[
\boxed{
F_{\mathrm{global}}
=
1-S_C
=
P\!\left(\exists i\in\mathcal H:Y_i\neq C\right).
}
\tag{1}
\]

该定义同时计入错误决定和未完成。

为区分 agreement 与 validity，建议同时报告

\[
S_W=P(\forall i\in\mathcal H:Y_i=W),
\]

\[
\boxed{
F_{\mathrm{agreement}}
=
1-S_C-S_W,
}
\tag{2}
\]

\[
\boxed{
F_{\mathrm{validity}}
=
S_W.
}
\tag{3}
\]

节点边缘量

\[
F_{\mathrm{node\ mean}}
=
\frac1{|\mathcal H|}
\sum_{i\in\mathcal H}
P(Y_i\neq C)
\]

只能作为诊断量，严禁再称为 consensus failure。

---

# 2. 为什么不能直接相乘节点边缘概率

一般情况下，

\[
P(\forall i:Y_i=C)
\neq
\prod_iP(Y_i=C),
\]

因为节点终态通过以下机制相关：

- 多个查询者共享同一响应节点；
- 同一节点在不同请求中返回一致状态；
- 共享 shadow、资源冲突、干扰和拥塞；
- 邻居状态沿图传播；
- weak cut、hub overload 或区域阻塞产生 cluster co-failure。

不受限制的精确联合过程可写为

\[
S_C
=
\boldsymbol\pi_0^\top
\left[
\prod_{t=0}^{R_{\max}-1}
\mathbf P_t(A)
\right]
\mathbf 1_{\mathcal C},
\tag{4}
\]

但若单节点状态数为 \(M=2\beta+2\)，全局状态空间为 \(M^N\)。

因此，“严格全局事件概率 + 可控复杂度”要求项目显式定义一个可计算的联合分布族，而不能继续先计算节点 marginals，再用均值、正态尾或独立乘积冒充全局概率。

---

# 3. 可控复杂度的全局联合模型

## 3.1 共享有限混合联合分布

引入整个网络共享的离散 latent scenario

\[
Z\in\{1,\ldots,Q\},
\qquad
P(Z=r)=\omega_r,
\qquad
\omega_r\ge0,\quad \sum_r\omega_r=1.
\]

\(Z\) 可表示：

- 全局或区域 shadow / blockage scenario；
- 资源拥塞状态；
- persistent interference condition；
- initial-condition regime。

在给定 \(Z=r\) 后，项目定义终态联合律为条件乘积分布

\[
P(Y_1,\ldots,Y_N\mid Z=r)
=
\prod_{i=1}^{N}P(Y_i\mid Z=r).
\tag{5}
\]

令

\[
c_{ir}=P(Y_i=C\mid Z=r),
\quad
w_{ir}=P(Y_i=W\mid Z=r),
\quad
u_{ir}=P(Y_i=U\mid Z=r).
\]

这些概率由 graph-coupled finite-horizon recurrence 计算；图耦合进入每个条件概率，但终态联合结构由式 (5) 明确定义。

## 3.2 全局失败闭式解

由全概率公式，

\[
\boxed{
S_C
=
\sum_{r=1}^{Q}
\omega_r
\prod_{i\in\mathcal H}c_{ir},
}
\tag{6}
\]

从而

\[
\boxed{
F_{\mathrm{global}}
=
1-
\sum_{r=1}^{Q}
\omega_r
\prod_{i\in\mathcal H}c_{ir}.
}
\tag{7}
\]

式 (7) 是所定义共享有限混合联合模型下的严格全局事件概率，不是节点均值。

还可严格分解为

\[
F_{\mathrm{any\ wrong}}
=
1-
\sum_r\omega_r
\prod_i(1-w_{ir}),
\tag{8}
\]

\[
F_{\mathrm{timeout\ without\ wrong}}
=
\sum_r\omega_r
\left[
\prod_i(1-w_{ir})
-
\prod_ic_{ir}
\right],
\tag{9}
\]

且

\[
F_{\mathrm{global}}
=
F_{\mathrm{any\ wrong}}
+
F_{\mathrm{timeout\ without\ wrong}}.
\tag{10}
\]

## 3.3 数值稳定计算

大规模 \(N\) 下禁止直接计算乘积。定义

\[
\log S_r
=
\sum_{i\in\mathcal H}\log(c_{ir}+\varepsilon),
\]

\[
\log S_C
=
\operatorname{logsumexp}_r
\left(
\log\omega_r+\log S_r
\right).
\tag{11}
\]

再计算

\[
\boxed{
F_{\mathrm{global}}
=
-\operatorname{expm1}(\log S_C).
}
\tag{12}
\]

训练时使用与 \(F_{\mathrm{global}}\) 严格单调等价的风险

\[
\boxed{
\mathcal L_F=-\log S_C.
}
\tag{13}
\]

其梯度为

\[
\nabla_\theta\mathcal L_F
=
-
\sum_r\rho_r
\sum_i\nabla_\theta\log c_{ir},
\tag{14}
\]

其中

\[
\rho_r
=
\frac{
\omega_r\prod_i c_{ir}
}{
\sum_s\omega_s\prod_i c_{is}
}.
\]

## 3.4 精确性边界

允许声称：

> Equation (7) is exact under the proposed shared finite-mixture conditional-product joint model.

禁止声称：

> Equation (7) is the exact global reliability of the unrestricted Avalanche process on an arbitrary graph.

必须使用：

- small-\(N\) exact global Markov chain；
- moderate-\(N\) direct Monte-Carlo；
- 必要时 rare-event importance sampling；

验证式 (7) 对真实采样过程的保真度。

若单一全局 latent scenario 无法捕获区域 co-failure，可升级为 bounded-treewidth regional latent factor graph，并通过 sum-product / junction tree 精确计算；复杂度必须对固定 treewidth 仍近似线性。

---

# 4. 无硬度上限的加权 distinct-peer polling

## 4.1 查询策略

节点 \(i\) 的物理可达候选集合为 \(\mathcal N_i\)。GNN 对每条候选边输出

\[
s_{ij}\in\mathbb R,
\qquad
a_{ij}=\exp(s_{ij})>0.
\tag{15}
\]

每轮从 \(\mathcal N_i\) 中选择恰好 \(k_{\mathrm{poll}}\) 个不同邻居。对子集 \(S\subseteq\mathcal N_i\)、\(|S|=k\)，定义

\[
\boxed{
P(S_i=S)
=
\frac{
\prod_{j\in S}a_{ij}
}{
e_k(a_i)
},
}
\tag{16}
\]

其中

\[
e_k(a_i)
=
\sum_{\substack{S\subseteq\mathcal N_i\\|S|=k}}
\prod_{j\in S}a_{ij}
\tag{17}
\]

是第 \(k\) 个 elementary symmetric polynomial。

该策略：

- 不使用 hard top-\(k\) support；
- 不设置固定节点度 cap；
- 所有物理候选边均处于可微路径；
- 训练与部署使用同一随机查询分布；
- 所有 \(a_{ij}\) 相等时退化为 uniform distinct-peer sampling。

部署采样必须使用与式 (16) 一致的 DP ancestral sampler；禁止用与式 (16) 不同的 Gumbel-top-\(k\) 或 Plackett–Luce 实现替代。

## 4.2 边包含概率与自适应有效度

边 \((i,j)\) 被选入一轮查询的概率为

\[
\boxed{
\pi_{ij}
=
\frac{
a_{ij}e_{k-1}(a_{i,-j})
}{
e_k(a_i)
}.
}
\tag{18}
\]

并且

\[
\sum_{j\in\mathcal N_i}\pi_{ij}=k_{\mathrm{poll}}.
\tag{19}
\]

模型的长期支持范围由 \(\pi_{ij}\) 自适应决定，而不是由 degree-\(K\) 人工裁剪。

在 \(H\) 轮内至少使用过邻居 \(j\) 的概率为

\[
m_{ij}^{(H)}
=
1-(1-\pi_{ij})^H,
\tag{20}
\]

期望 unique-neighbour count 为

\[
d_i^{(H)}
=
\sum_jm_{ij}^{(H)}.
\tag{21}
\]

该量可进入邻居维护、CSI、SCI、sensing 与 processing 开销，但不得变成固定上限约束。

---

# 5. 精确异质 quorum 概率

在 shared scenario \(r\)、round \(t\) 下，邻居 \(j\) 返回 correct、wrong、no-response 的概率分别为

\[
p^+_{ijr}(t)
=
\ell_{ijr}(t)u_{jr}(t),
\tag{22}
\]

\[
p^-_{ijr}(t)
=
\ell_{ijr}(t)v_{jr}(t),
\tag{23}
\]

\[
p^0_{ijr}(t)
=
1-p^+_{ijr}(t)-p^-_{ijr}(t).
\tag{24}
\]

定义生成函数

\[
\Psi_{ir}^{t}(z,x,y)
=
\prod_{j\in\mathcal N_i}
\left[
1+
za_{ij}
\left(
p^0_{ijr}
+p^+_{ijr}x
+p^-_{ijr}y
\right)
\right].
\tag{25}
\]

则在恰好选择 \(k\) 个 distinct peers 后，获得 \(m\) 个 correct、\(n\) 个 wrong response 的概率为

\[
\boxed{
P_{ir}^{t}(m,n)
=
\frac{
[z^kx^my^n]\Psi_{ir}^{t}(z,x,y)
}{
e_k(a_i)
}.
}
\tag{26}
\]

若 \(2\alpha>k\)，correct quorum 与 wrong quorum 不会同时发生：

\[
h^+_{ir}(t)
=
\sum_{m=\alpha}^{k}
\sum_{n=0}^{k-m}
P_{ir}^{t}(m,n),
\tag{27}
\]

\[
h^-_{ir}(t)
=
\sum_{n=\alpha}^{k}
\sum_{m=0}^{k-n}
P_{ir}^{t}(m,n),
\tag{28}
\]

\[
h^0_{ir}(t)
=
1-h^+_{ir}(t)-h^-_{ir}(t).
\tag{29}
\]

这替换当前 iid-with-replacement beta tail，并严格支持：

- heterogeneous links；
- distinct peers；
- different neighbour states；
- no-response；
- adaptive query weights。

## 5.1 可微动态规划

令 \(C_{j,q,m,n}\) 表示处理前 \(j\) 个候选、选中 \(q\) 个、其中 correct 为 \(m\)、wrong 为 \(n\) 的未归一化系数。递推为

\[
\begin{aligned}
C_{j,q,m,n}
=&\ C_{j-1,q,m,n}\\
&+a_{ij}p^0_{ij}C_{j-1,q-1,m,n}\\
&+a_{ij}p^+_{ij}C_{j-1,q-1,m-1,n}\\
&+a_{ij}p^-_{ij}C_{j-1,q-1,m,n-1}.
\end{aligned}
\tag{30}
\]

只保留

\[
q\le k,\qquad m+n\le q.
\]

单节点复杂度为

\[
O(|\mathcal N_i|k^3),
\]

全图、\(Q\) 个 shared scenarios、\(R_{\max}\) 轮的复杂度目标为

\[
\boxed{
O\!\left(
Q R_{\max}
\left(
E k_{\mathrm{poll}}^3+N\beta
\right)
\right).
}
\tag{31}
\]

在 \(Q,k_{\mathrm{poll}},\beta,R_{\max}\) 为小常数时，对 \(N,E\) 近似线性。

---

# 6. Graph-coupled finite-horizon recurrence

对每个节点 \(i\)、scenario \(r\)，维护 Snowball 状态向量

\[
\mathbf p_{ir}(t)\in\Delta^{2\beta+2}.
\]

从 \(\mathbf p_{ir}(t)\) 读出 conditional correct/wrong preference

\[
u_{ir}(t),\qquad v_{ir}(t).
\]

由式 (22)–(29) 计算

\[
h^+_{ir}(t),\quad h^-_{ir}(t),\quad h^0_{ir}(t),
\]

再通过单节点 Snowball transition matrix 更新

\[
\mathbf p_{ir}(t+1)
=
\mathbf p_{ir}(t)
\mathbf T_i
\left(
h^+_{ir}(t),h^-_{ir}(t),h^0_{ir}(t)
\right).
\tag{32}
\]

经过 \(R_{\max}\) 轮后得到

\[
c_{ir},\quad w_{ir},\quad u_{ir},
\]

并代入式 (6)–(12) 计算严格全局事件概率。

该 recurrence 对所定义 conditional-product model 是确定性、可微且无需 MC sampling；MC 只用于外部验证，不进入训练。

---

# 7. 无固定节点度上限的物理约束

候选图只能由物理可达性产生，例如：

- 通信半径；
- 接收灵敏度；
- 几何 LOS/NLOS/NLOSv；
- 频谱与设备能力。

禁止在半径过滤之后再截取每节点前 8、12 或其他固定数量候选。

固定车辆密度与固定通信半径下，应通过 spatial hashing / cell lists 构造候选边，使

\[
E=O(N)
\]

在期望意义上成立；实现中禁止构造 \(N\times N\) dense tensor。

## 7.1 接收端负载

由边包含概率计算接收端期望负载：

\[
\Lambda_{jr}(t)
=
\sum_i
\tau_{ir}(t)\pi_{ij},
\tag{33}
\]

其中 \(\tau_{ir}(t)\) 是节点 \(i\) 在 round \(t\) 仍处于 transient state 的概率。

物理链条必须为

\[
\Lambda
\longrightarrow
\text{resource overlap / collision / queueing / interference}
\longrightarrow
\gamma
\longrightarrow
\ell
\longrightarrow
(F,D,E).
\tag{34}
\]

高 hub load 和过密支持必须通过实际物理代价被抑制，而不是通过人工 degree cap。

## 7.2 候选数不足

若 \(|\mathcal N_i|<k_{\mathrm{poll}}\)，必须显式采用一种协议规则：

1. 该节点不能形成完整 quorum；
2. 预先定义 \(k_i=\min(k,|\mathcal N_i|)\) 与匹配的 \(\alpha_i\)；
3. 使用 RSU fallback。

严禁复制同一邻居来凑满 \(k\)，除非协议明确允许且相关性被正确建模。

---

# 8. 严谨 finite-blocklength link-delivery

## 8.1 最低可接受公式

对 complex AWGN channel，

\[
C(\gamma)=\log_2(1+\gamma),
\tag{35}
\]

\[
V(\gamma)
=
\left(
1-\frac1{(1+\gamma)^2}
\right)
(\log_2e)^2.
\tag{36}
\]

长度为 \(n\) complex channel uses、承载 \(B\) bits 时，normal approximation 为

\[
\boxed{
\epsilon_{\mathrm{FBL}}(\gamma,n,B)
\approx
Q\left(
\frac{
nC(\gamma)-B+\frac12\log_2n
}{
\sqrt{nV(\gamma)}
}
\right).
}
\tag{37}
\]

链路成功率为

\[
\ell_{\mathrm{FBL}}
=
1-\epsilon_{\mathrm{FBL}}.
\tag{38}
\]

严禁再把 \(V(\gamma)\) 吸收到常数 \(\sqrt n\) 中。

## 8.2 Blocklength 的物理单位

必须定义

\[
n
=
N_{\mathrm{RB}}
N_{\mathrm{SC/RB}}
N_{\mathrm{sym,data}},
\tag{39}
\]

并显式扣除：

- DMRS；
- SCI/control；
- guard；
- reserved RE；
- pilot/reference signals。

\(B\) 必须包括实际 payload、CRC 与需要纳入的协议头。

## 8.3 衰落、碰撞与 query 双向链路

若存在 quasi-static fading \(H\)，

\[
\ell_{ij}
=
\mathbb E_H
\left[
1-\epsilon_{\mathrm{FBL}}
\left(
\gamma_{ij}|H|^2,n,B
\right)
\right],
\tag{40}
\]

使用共享 scenario 或确定性 quadrature 计算。

一次有效 poll 至少包含 request 和 response：

\[
\boxed{
\ell_{ij}^{\mathrm{poll}}
=
(1-p_{\mathrm{col},ij})
(1-p_{\mathrm{HD},ij})
(1-\epsilon_{ij}^{\mathrm{req}})
(1-\epsilon_{ji}^{\mathrm{resp}}).
}
\tag{41}
\]

其中 collision、half-duplex 与 PHY decoding error 必须分开建模。

## 8.4 允许的论文表述

允许：

> finite-blocklength normal-approximation link-delivery model with explicit channel dispersion.

禁止：

> exact NR finite-blocklength BLER.

若需要严格信息论区间，应实现 RCU/DT achievability bound 与 meta-converse bound，并报告可靠性区间；否则必须把式 (37)称为 normal approximation。

---

# 9. 独立的全局 \(D\) 与网络 \(E\)

## 9.1 全局完成时延

定义

\[
S_r(t)
=
P(\forall i:Y_i(t)=C\mid Z=r)
=
\prod_i c_{ir}(t),
\tag{42}
\]

\[
S(t)
=
\sum_r\omega_rS_r(t).
\tag{43}
\]

令 \(T_{\mathrm{all}}\) 为所有 eligible nodes 正确完成的轮次。成功条件下的全局完成轮数为

\[
\boxed{
D_{\mathrm{round}}
=
\mathbb E[
T_{\mathrm{all}}
\mid T_{\mathrm{all}}\le R_{\max}
]
=
\frac{
\displaystyle
\sum_{t=0}^{R_{\max}-1}
\left[
S(R_{\max})-S(t)
\right]
}{
S(R_{\max})
}.
}
\tag{44}
\]

若每轮持续时间固定为 \(\tau_{\mathrm{round}}\)，

\[
D=\tau_{\mathrm{round}}D_{\mathrm{round}}.
\tag{45}
\]

该指标是最后一个节点完成的全局 order statistic，不是节点 expected rounds 的均值。

同时可报告 deadline-penalized 诊断量

\[
D_{\mathrm{cap}}
=
\tau_{\mathrm{round}}
\sum_{t=0}^{R_{\max}-1}
[1-S(t)].
\tag{46}
\]

## 9.2 并行 query 的 wall-clock 扩展

设每条 selected link 最多尝试 \(M\) 次，单次成功率为 \(\ell_{ij}\)。边在 \(m\) 次内完成的概率为

\[
f_{ij}(m)
=
1-(1-\ell_{ij})^m.
\tag{47}
\]

对式 (16) 的 weighted \(k\)-subset policy，一轮中全部 selected links 在 \(m\) 次内完成的概率为

\[
\boxed{
P(L_i\le m)
=
\frac{
e_k(a_i\odot f_i(m))
}{
e_k(a_i)
}.
}
\tag{48}
\]

因此慢est query 的截断期望尝试次数为

\[
\boxed{
\mathbb E[L_i]
=
\sum_{m=0}^{M-1}
\left[
1-
\frac{
e_k(a_i\odot f_i(m))
}{
e_k(a_i)
}
\right].
}
\tag{49}
\]

该公式可用于替换 weighted mean / ad hoc soft-max delay。

## 9.3 网络总能量

边 \(ij\) 的截断几何尝试次数期望为

\[
\bar n_{ij}
=
\sum_{m=0}^{M-1}(1-\ell_{ij})^m
=
\frac{
1-(1-\ell_{ij})^M
}{
\ell_{ij}
}.
\tag{50}
\]

令一次 request-response attempt 的真实能量为

\[
e_{ij}^{\mathrm{attempt}}
=
E_{ij}^{\mathrm{tx}}
+
E_{ij}^{\mathrm{rx}}
+
E_{ij}^{\mathrm{proc}}.
\tag{51}
\]

节点每轮期望能量为

\[
e_i^{\mathrm{round}}(t)
=
\sum_j
\pi_{ij}
\bar n_{ij}(t)
e_{ij}^{\mathrm{attempt}}(t)
+
E_i^{\mathrm{maint}}(t).
\tag{52}
\]

网络总期望能量为

\[
\boxed{
E
=
\sum_r\omega_r
\sum_{t=0}^{R_{\max}-1}
\sum_i
\tau_{ir}(t)
e_{ir}^{\mathrm{round}}(t).
}
\tag{53}
\]

式 (53) 利用期望的线性性，不需要把节点能量错误地当成全局最大值。

## 9.4 为什么 \(D\) 与 \(E\) 独立

- \(D\) 使用所有节点完成时间的全局 order statistic；
- \(E\) 使用所有节点、所有边、所有尝试的总和；
- \(D\) 对最慢节点和 weak cut 敏感；
- \(E\) 对支持范围、发射功率、重传和维护开销敏感。

二者不应再满足固定比例 \(D\propto E\)。

为产生真实 Pareto conflict，模型应增加独立控制变量：

### 发射功率头

\[
P_i
=
P_{\min}
+
(P_{\max}-P_{\min})\sigma(r_i).
\tag{54}
\]

### Blocklength / resource 头

\[
n_i
=
n_{\min}
+
(n_{\max}-n_{\min})\sigma(b_i).
\tag{55}
\]

通常：

- 增大 \(P_i\)：降低 \(F,D\)，提高 \(E\)；
- 增大 \(n_i\)：降低 \(F\)，提高单次 transmission time 与能耗；
- 改变 query distribution：影响 weak cuts、拥塞、\(D\) 与维护能耗。

---

# 10. 新主线的统一数学接口

最终 evaluator 应只保留一条主线：

\[
\boxed{
(a_\theta,P_\theta,n_\theta)
\rightarrow
\pi
\rightarrow
\Lambda
\rightarrow
\gamma
\rightarrow
\ell
\rightarrow
(h^+,h^-,h^0)
\rightarrow
\mathbf p_{ir}(t)
\rightarrow
(F_{\mathrm{global}},D,E).
}
\tag{56}
\]

训练目标可采用 preference-conditioned augmented Chebyshev：

\[
\mathcal L_\lambda
=
\max_{m\in\{F,D,E\}}
\lambda_m
\frac{z_m-z_m^\star}{s_m}
+
\rho
\sum_m
\lambda_m
\frac{z_m-z_m^\star}{s_m}.
\tag{57}
\]

其中 \(\lambda\in\Delta^2\) 作为模型输入，使一个模型覆盖多种 Pareto operating points。

原 emission 应升级为全局风险贡献。定义

\[
r_{ir}(t)
=
-\log(c_{ir}(t)+\varepsilon),
\tag{58}
\]

则

\[
-\log S_r(t)=\sum_i r_{ir}(t).
\]

可构造

\[
e_i^t
=
\operatorname{clip}
\left(
\operatorname{sg}
\left[
\sum_r\rho_r r_{ir}(t)
\right]/r_{\max},
0,1
\right),
\tag{59}
\]

作为下一帧输入，使 temporal emission 与全局 \(F\) 直接对齐。

---

# 11. 必须通过的数学验证

1. **Weighted-subset DP**
   - 与 brute-force 子集枚举误差 \(<10^{-10}\)。
2. **Global mixture formula**
   - 与显式 mixture enumeration 误差 \(<10^{-12}\)。
3. **Failure decomposition**
   - 式 (10) 数值恒等误差 \(<10^{-12}\)。
4. **Gradient**
   - autograd 与 central finite difference 相对误差 \(<10^{-4}\)。
5. **Small-\(N\) exact-joint**
   - 构造完整 global Markov chain，对 \(N\le8\) 比较。
6. **MC fidelity**
   - 报告 \(\log S_C\) MAE、Spearman \(\rho\)、topology top-1 accuracy 与置信区间。
7. **Scenario convergence**
   - \(Q\) 加倍后 \(|\Delta\log S_C|<10^{-3}\)。
8. **复杂度**
   - 不出现 \(N\times N\) tensor；runtime 对 \(E\) 近似线性。
9. **FBL**
   - 显式包含 \(V(\gamma)\)，所有单位可追踪。
10. **D/E independence**
    - \(\nabla D\) 与 \(\nabla E\) 不恒共线；
    - held-out scenes 的 median gradient cosine \(<0.95\)；
    - 至少三个跨 seeds 稳定的非支配 Pareto 点。
11. **主线唯一性**
    - mainline 不得再 import mean-field、hard degree cap 或 legacy node-mean \(F\)。

---

# 12. 禁止保留在新论文主线中的旧结论

重构后必须重新计算，不能沿用：

- degree-4 protocol floor；
- 旧 \(F=0.0635\) 等节点均值结果；
- mean-field / quenched / MC “currency”叙事；
- hard top-\(k\) / ST constructor；
- 旧 Pareto 表；
- 以 per-node confidence 为全局可靠性的 emission；
- 基于固定 degree 的 baseline 排名。

旧实现只能归档为历史复现材料，不得继续作为训练或论文结果来源。
