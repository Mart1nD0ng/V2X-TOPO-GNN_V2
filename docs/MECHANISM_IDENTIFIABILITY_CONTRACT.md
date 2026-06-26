# Mechanism Identifiability Contract

## 0. 目的

本合同适用于项目中所有声称具有特定作用机理的组件，包括：

- CDQ；
- evidence-correlation encoder；
- region channel；
- interference/load module；
- queueing；
- temporal memory；
- emission；
- weak-cut refinement；
- auxiliary loss；
- protocol adaptation。

组件只有同时满足以下五项，才能被称为“机制已验证”。

本合同**不要求**：

- 新组件必须在开发前通过 oracle 获益测试；
- 复杂组件必须严格退化成简单组件。

某个具体组件可以自主采用嵌套设计，例如 CDQ 2.0 严格包含 ESP，但这不是所有组件的通用要求。

---

# C1. 环境结构存在性

环境必须真实包含组件声称要利用或修复的结构。

## 必须说明

- 目标结构是什么；
- 结构由哪些随机变量或物理机制产生；
- 如何改变其强度；
- 如何构造结构缺失的控制组；
- 结构变化时哪些非目标变量保持不变。

## 例：CDQ

必须存在：

- marginal correctness 相同；
- link-quality 分布匹配；
- pairwise/shared evidence correlation 不同；

的场景。

仅改变某个 region 的 marginal error，不能充分识别 pairwise diversity 机制。

## 验收

- empirical structure statistic 与设计值一致；
- zero-structure control；
- strength sweep；
- matched-marginal causal comparison。

---

# C2. 部署可观测性

模型必须在部署时观察到目标结构的合法 proxy。

## 禁止

- ground truth；
- simulator-only latent variable；
- peer 当前 vote；
- 评价后才能得到的标签；
- 未来信息。

## 必须记录

- 原始可观测量；
- proxy 的计算方式；
- proxy 噪声；
- proxy 与真实结构的校准；
- OOD 下 proxy 的退化。

## 例：相关证据

可用：

- road region；
- sensor source；
- map source；
- historical agreement；
- shared shadow；
- response history。

不能直接输入“真实 correlation matrix”，除非部署系统确实可测得。

## 验收

- truth-leakage test；
- proxy shuffle；
- noisy-proxy robustness；
- missing-proxy fallback。

---

# C3. 梯度可达性

训练目标必须向组件参数提供非零、方向正确、数值稳定的梯度。

## 必须检查

\[
\left\|
\frac{\partial\mathcal L}
{\partial\theta_M}
\right\|>0
\]

在组件应发挥作用的 stress scene 中成立。

## 禁止

- `.item()`；
- `float(tensor)`；
- 无说明的 `detach()`；
- silent clamp 截断全部梯度；
- sampled metric 被错误当作可微目标。

## 验收

1. finite-difference gradient；
2. gradient norm trace；
3. expected-direction test；
4. zero-structure control 中梯度应消失或显著减弱；
5. stress-strength sweep 中梯度响应单调或符合理论。

---

# C4. 因果对照

必须通过只改变目标机制相关变量的对照，证明收益来自所声称结构。

## 必须包含

- mechanism on/off；
- target structure on/off；
- target structure strength sweep；
- structure shuffle；
- proxy shuffle；
- matched marginal/geometry/physics control。

## 结果矩阵

| Environment structure | Mechanism off | Mechanism on |
|---|---:|---:|
| absent | control | 不应产生虚假收益 |
| present | baseline | 应出现机制收益 |

## 解释规则

- 结构不存在时仍显著获益：机制解释可能错误；
- 结构存在但无收益：组件、可观测性或梯度可能失败；
- shuffle 后收益仍在：收益不是来自声称结构；
- 只有 marginal 改变时获益：不能归因于 correlation/diversity。

---

# C5. 主链一致性

机制必须在训练、验证、headline、ablation 和 figure generation 中走同一 canonical path。

## 必须一致

- protocol semantics；
- physics；
- environment structure；
- query law；
- feature availability；
- mechanism switch；
- normalization；
- service profile。

## 强制措施

1. checkpoint 保存完整 config hash；
2. train/eval compatibility check；
3. runtime mechanism trace；
4. sentinel activation test；
5. unused config field 报错；
6. figure script 只读取结果，不重写 evaluator；
7. OOD 实验必须显式登记允许改变的轴。

## 禁止

- ideal-link 训练、full-physics headline；
- 机制仅存在于 gate/test；
- 训练启用相关结构，评价关闭；
- ablation 使用另一套 evaluator；
- 相同名称对应不同数学定义。

---

# Contract Evidence Template

每个组件必须提交：

```text
MECHANISM:
CLAIMED EFFECT:

C1 ENVIRONMENT STRUCTURE:
- definition:
- generator:
- absent control:
- strength sweep:
- matched variables:
- tests:

C2 DEPLOYMENT OBSERVABILITY:
- observable proxies:
- forbidden information audit:
- proxy calibration:
- OOD robustness:
- tests:

C3 GRADIENT REACHABILITY:
- objective path:
- finite-difference result:
- gradient norm:
- direction test:
- zero-structure result:
- tests:

C4 CAUSAL CONTROLS:
- factorial design:
- matched controls:
- shuffle controls:
- results:
- interpretation:

C5 MAINLINE CONSISTENCY:
- canonical entry point:
- train hash:
- eval hash:
- runtime trace:
- sentinel:
- unused-config audit:

STATUS:
PASS | FAIL | INCOMPLETE

ALLOWED CLAIM:
```

只有五项全部 PASS，组件才能进入论文核心贡献。
