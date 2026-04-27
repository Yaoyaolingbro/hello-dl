# 随机过程与马尔科夫链

!!! info "参考资料"
    **主要资料**
    - [Deep Learning Book: Chapter 17](https://www.deeplearningbook.org/contents/graphical_models.html) — Ian Goodfellow et al.
    - Norris, *Markov Chains* — 经典教材，前两章覆盖本节内容
    - [Intro to Stochastic Processes](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/) — MIT 6.262

## 直觉 (Intuition)

随机过程把随机变量从"一个时刻"推广到"随时间演化的序列"。马尔科夫链是最简单的随机过程：下一个状态只依赖当前状态，不依赖更早的历史。强化学习里的 MDP、扩散模型的加噪/去噪过程、以及语言模型的自回归生成，都建立在这个"只看当前"的假设上。

## 主要符号

| 符号 | 含义 |
|------|------|
| $X_t$ | 时刻 $t$ 的随机变量 |
| $\mathcal{S}$ | 状态空间 |
| $P(X_{t+1} = j \mid X_t = i)$ | 状态转移概率 |
| $\mathbf{T}$ | 状态转移矩阵（$T_{ij} = P(j \mid i)$） |
| $\boldsymbol{\pi}$ | 平稳分布 |

## 随机过程

**随机过程** $\{X_t\}_{t \ge 0}$ 是一族随机变量，用时间索引。在每个时刻 $t$，$X_t$ 是一个随机变量，其分布可能与时间有关。

- **离散时间**：$t \in \{0, 1, 2, \ldots\}$，如语言模型的 token 序列
- **连续时间**：$t \in [0, \infty)$，如股票价格、扩散模型中的连续加噪过程

联合分布 $p(X_0, X_1, \ldots, X_T)$ 包含了过程的完整信息，但直接处理高维。马尔科夫假设是简化它的关键工具。

## 马尔科夫性质

**马尔科夫性质**（无记忆性）：给定当前状态，未来独立于过去：

$$
P(X_{t+1} \mid X_t, X_{t-1}, \ldots, X_0)
=
P(X_{t+1} \mid X_t)
$$

满足这个性质的随机过程叫**马尔科夫链**。

马尔科夫性质为什么有用？它把一个复杂的联合分布分解成简单的条件乘积：

$$
p(X_0, X_1, \ldots, X_T)
=
p(X_0) \prod_{t=0}^{T-1} P(X_{t+1} \mid X_t)
$$

每一步只需要一个条件分布 $P(X_{t+1} \mid X_t)$，而不是完整历史。

## 转移矩阵

对离散状态空间 $\mathcal{S} = \{1, 2, \ldots, n\}$，用**转移矩阵** $\mathbf{T} \in \mathbb{R}^{n \times n}$ 编码一步转移概率：

$$
T_{ij} = P(X_{t+1} = j \mid X_t = i)
$$

$\mathbf{T}$ 的每一行是一个概率分布（行和为 1），称为**随机矩阵**。

$k$ 步转移概率可以用矩阵幂计算：$P(X_{t+k} = j \mid X_0 = i) = (\mathbf{T}^k)_{ij}$。这与特征值分解直接关联——长期行为由 $\mathbf{T}$ 的最大特征值及其对应特征向量决定。

## 平稳分布

如果分布 $\boldsymbol{\pi}$ 满足：

$$
\boldsymbol{\pi}^\top \mathbf{T} = \boldsymbol{\pi}^\top
$$

则称 $\boldsymbol{\pi}$ 为平稳分布（稳态分布）。它是转移矩阵的左特征向量，对应特征值 1。

对不可约、非周期的马尔科夫链（遍历链），无论初始分布如何，$k$ 步后都会收敛到同一个平稳分布。MCMC 采样算法（如 Metropolis-Hastings）正是利用这个性质——设计转移核使其平稳分布等于目标分布。

## 随机微分方程（SDE）直觉

扩散模型（DDPM）的连续版本可以用随机微分方程（SDE）描述：

$$
dX_t = f(X_t, t)\, dt + g(t)\, dW_t
$$

其中 $f$ 是漂移项（确定性趋势），$g$ 是扩散系数，$W_t$ 是**布朗运动（维纳过程）**——连续时间的"随机游走"，每个增量 $W_{t+dt} - W_t \sim \mathcal{N}(0, dt)$。

扩散模型的前向过程（加噪）是一个已知的 SDE，逆向过程（去噪）也是 SDE，参数化形式由 Anderson 1982 年的反向 SDE 定理给出。学会用 SDE 视角看扩散模型，理解 score matching 会变得自然很多。

## 代码验证

```python
import numpy as np

# 构造一个简单马尔科夫链：3 个状态（晴天/多云/雨天）
T = np.array([
    [0.7, 0.2, 0.1],  # 晴天转移概率
    [0.3, 0.4, 0.3],  # 多云转移概率
    [0.2, 0.3, 0.5],  # 雨天转移概率
])

# 验证每行和为 1（随机矩阵）
print(T.sum(axis=1))  # [1. 1. 1.]

# 模拟马尔科夫链：从晴天出发，走 1000 步
state = 0  # 晴天
counts = np.zeros(3)
np.random.seed(42)
for _ in range(10000):
    state = np.random.choice(3, p=T[state])
    counts[state] += 1
empirical_pi = counts / counts.sum()
print("经验平稳分布:", empirical_pi.round(4))

# 解析平稳分布：左特征向量
eigenvalues, eigenvectors = np.linalg.eig(T.T)
# 找特征值为 1 的特征向量
idx = np.argmax(np.real(eigenvalues))
pi = np.real(eigenvectors[:, idx])
pi = pi / pi.sum()  # 归一化
print("解析平稳分布:", pi.round(4))
# 两种方法结果接近

# 验证平稳性：pi^T T = pi^T
print(np.allclose(pi @ T, pi, atol=1e-6))  # True

# k 步转移：矩阵幂
T_10 = np.linalg.matrix_power(T, 10)
print(T_10.round(4))
# 所有行接近相同（收敛到平稳分布）
```

## 在深度学习中的应用

强化学习的 MDP（马尔科夫决策过程）在马尔科夫链上加入动作和奖励，是 RL 的核心框架（Part 3 RL 章节详解）。扩散模型（DDPM）把生成过程建模为可逆马尔科夫链：前向链逐步加噪，逆向链逐步去噪。MCMC 采样（变分推断和 LLM 采样的基础）利用平稳分布性质从复杂分布中采样。

下一节讲状态估计与卡尔曼滤波，把贝叶斯推断和马尔科夫假设结合起来，在动态系统中实时更新状态信念。
