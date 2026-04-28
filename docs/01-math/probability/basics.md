# 概率基础

!!! info "参考资料"
    **主要资料**
    
    - [Deep Learning Book: Chapter 3 Probability and Information Theory](https://www.deeplearningbook.org/contents/prob.html) — Ian Goodfellow et al.
    - [Introduction to Probability](https://www.probabilitycourse.com/) — Hossein Pishro-Nik，免费在线教材

## 直觉 (Intuition)

概率是在量化"不确定性"。输入是一个随机事件，输出是一个 $[0,1]$ 之间的数，表示这件事发生的可能性。深度学习模型的输出几乎总是概率：分类器输出类别的概率，生成模型输出下一个 token 的概率，强化学习智能体输出动作的概率。理解概率论，就是在理解神经网络在表达什么。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\Omega$ | 样本空间，所有可能结果的集合 |
| $A, B$ | 事件（$\Omega$ 的子集） |
| $P(A)$ | 事件 $A$ 的概率 |
| $P(A \mid B)$ | 在 $B$ 发生的条件下，$A$ 的条件概率 |
| $P(A, B)$ | $A$ 和 $B$ 同时发生的联合概率 |

## 概率的三条公理

概率不是随便指定的数，需要满足三条公理（Kolmogorov 公理）。

第一，概率非负：

$$
P(A) \ge 0
$$

第二，整个样本空间的概率为 1：

$$
P(\Omega) = 1
$$

第三，互斥事件的概率可以相加。若 $A \cap B = \emptyset$，则：

$$
P(A \cup B) = P(A) + P(B)
$$

由这三条可以推导出：补事件 $P(\bar{A}) = 1 - P(A)$，以及一般加法公式 $P(A \cup B) = P(A) + P(B) - P(A \cap B)$。

## 条件概率

条件概率 $P(A \mid B)$ 回答的是：已知 $B$ 发生了，$A$ 发生的概率是多少。

$$
P(A \mid B)
=
\frac{P(A, B)}{P(B)}, \quad P(B) > 0
$$

把它理解为"在 $B$ 划定的范围内，$A$ 占多大比例"。从这个式子出发，可以得到乘法规则：

$$
P(A, B) = P(A \mid B)\,P(B) = P(B \mid A)\,P(A)
$$

!!! warning "常见误区"
    $P(A \mid B)$ 和 $P(B \mid A)$ 不是同一件事。"患者测出阳性的概率"和"阳性结果来自患者的概率"含义完全不同，混淆这两者在医学诊断中造成过真实错误（检察官谬误）。

## 全概率公式

如果事件集合 $\{B_1, B_2, \ldots, B_n\}$ 构成样本空间的一个完备划分（互斥且 $\bigcup_i B_i = \Omega$），则：

$$
P(A)
=
\sum_{i=1}^{n} P(A \mid B_i)\, P(B_i)
$$

直觉：把 $A$ 发生的所有路径加起来，每条路径按"走这条路"的概率加权。

## 贝叶斯公式

贝叶斯公式把条件的方向反过来：

$$
P(A \mid B)
=
\frac{P(B \mid A)\, P(A)}{P(B)}
$$

结合全概率公式，分母可以完全展开，不再依赖 $P(B)$ 的直接计算。

!!! note "贝叶斯公式的四要素"
    - **先验 (Prior)**：$P(A)$，还没看数据时的初始信念
    - **似然 (Likelihood)**：$P(B \mid A)$，假设 $A$ 成立时观测到 $B$ 的概率
    - **证据 (Evidence)**：$P(B)$，观测到数据的总概率（归一化常数）
    - **后验 (Posterior)**：$P(A \mid B)$，看到数据后的更新信念

## 独立性

若 $A$ 和 $B$ 相互独立，则：

$$
P(A, B) = P(A)\, P(B)
$$

等价地，$P(A \mid B) = P(A)$——知道 $B$ 发生了对 $A$ 的概率没有影响。独立性假设可以大幅简化模型设计，朴素贝叶斯分类器和自回归语言模型都依赖它。

## 代码验证

```python
import numpy as np

np.random.seed(0)
n = 1_000_000
rolls = np.random.randint(1, 7, size=n)

A = rolls % 2 == 0      # 偶数：{2, 4, 6}
B = rolls > 3           # 大于 3：{4, 5, 6}
AB = A & B              # 同时满足：{4, 6}

P_A = A.mean()          # ≈ 0.5
P_B = B.mean()          # ≈ 0.5
P_AB = AB.mean()        # ≈ 1/3

# 条件概率 P(A|B) = P(A,B)/P(B)
P_A_given_B = P_AB / P_B
print(P_A_given_B)      # ≈ 0.667，即 P({4,6}|{4,5,6}) = 2/3

# 贝叶斯公式验证：P(B|A) 两种方式相等
P_B_given_A_direct = P_AB / P_A
P_B_given_A_bayes  = P_A_given_B * P_B / P_A
print(np.isclose(P_B_given_A_direct, P_B_given_A_bayes))  # True
```

## 在深度学习中的应用

分类器的输出是条件概率 $P(\text{类别} \mid \text{输入})$，交叉熵损失就是在最大化它的对数。贝叶斯公式是贝叶斯神经网络、变分推断和概率生成模型的数学核心。独立性假设简化了自回归生成（每个 token 只依赖前面的 token）。

下一节讲随机变量。它把"离散事件"推广到"取值"，让我们在连续空间里谈概率。
