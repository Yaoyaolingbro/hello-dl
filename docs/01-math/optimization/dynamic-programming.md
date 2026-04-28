# 动态规划

!!! info "参考资料"
    **主要资料**
    
    - Bellman, *Dynamic Programming*, 1957 — 原始教材
    - Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. — Chapter 3–4（DP 与 Bellman 方程）
    - [David Silver RL Lectures](https://www.davidsilver.uk/teaching/) — Lecture 2–3，清晰的 DP 推导

## 直觉 (Intuition)

动态规划（Dynamic Programming）解决的问题是：一个多步决策序列，如何找到全局最优方案。核心洞察是**最优子结构**——最优策略的任意子序列本身也是最优的。这个性质让原本指数级的枚举问题变成多项式级的递推。Bellman 方程把"未来总收益"写成"当前收益 + 折扣后的未来收益"，是所有强化学习算法（Q-Learning、PPO、SAC）的数学基础。

## 主要符号

| 符号 | 含义 |
|------|------|
| $s \in \mathcal{S}$ | 状态 |
| $a \in \mathcal{A}$ | 动作 |
| $r(s, a)$ | 即时奖励 |
| $\gamma \in [0, 1)$ | 折扣因子 |
| $V^\pi(s)$ | 策略 $\pi$ 下状态 $s$ 的值函数 |
| $Q^\pi(s, a)$ | 策略 $\pi$ 下状态-动作对的 Q 函数 |

## 最优子结构

**最优子结构（optimal substructure）**：最优解的子问题的解，也是子问题的最优解。

经典例子：从 A 到 C 的最短路径，如果经过 B，则 A→B 和 B→C 分别也是各自的最短路径。这个性质允许我们把大问题分解：

$$
\text{opt}_{A \to C} = \min_B \left[\text{opt}_{A \to B} + \text{opt}_{B \to C}\right]
$$

不满足最优子结构的问题（如"最长非重复路径"）不能用 DP 求解。

## Bellman 期望方程

马尔科夫决策过程（MDP）中，策略 $\pi$ 下的**值函数**定义为从状态 $s$ 出发，按策略 $\pi$ 行动所获得的折扣累计奖励期望：

$$
V^\pi(s)
=
\mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k r(s_{t+k}, a_{t+k}) \;\Big|\; s_t = s\right]
$$

**Bellman 期望方程**把值函数写成递推形式：

$$
V^\pi(s)
=
\sum_{a} \pi(a \mid s)
\left[r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^\pi(s')\right]
$$

直觉：当前状态的价值 = 即时奖励 + 折扣后的后继状态价值的期望。这是一个关于 $V^\pi$ 的线性方程组（有限状态情况下），可以解析求解，也可以迭代求解。

## Bellman 最优方程

最优值函数 $V^*(s) = \max_\pi V^\pi(s)$ 满足**Bellman 最优方程**：

$$
V^*(s)
=
\max_{a}
\left[r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V^*(s')\right]
$$

最优 Q 函数 $Q^*(s, a)$ 满足：

$$
Q^*(s, a)
=
r(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, \max_{a'} Q^*(s', a')
$$

这是非线性方程（含 $\max$），一般需要迭代求解。**值迭代（Value Iteration）**从任意初始 $V_0$ 出发，反复应用 Bellman 算子直到收敛，理论上能收敛到 $V^*$（折扣 MDP 中 Bellman 算子是压缩映射）。

!!! note "为什么要折扣因子"
    $\gamma < 1$ 保证了无限时域的累计奖励是有限数；也可以理解为"近期奖励比远期更确定"的偏好。$\gamma \to 1$ 时 DP 趋向平均奖励问题；$\gamma = 0$ 时退化为贪心策略（只看当前奖励）。

## 从表格 DP 到函数近似

表格 DP 在有限状态空间 $|\mathcal{S}|$ 不大时可行：维护一张 $V(s)$ 的表格，迭代更新。状态空间大（如图像输入）时，表格无法存储，需要用**函数近似**：

$$
V^\pi(s) \approx V_\theta(s), \quad Q^\pi(s,a) \approx Q_\theta(s, a)
$$

用神经网络参数化值函数，然后通过 Bellman 残差优化参数：

$$
\mathcal{L}(\theta) = \mathbb{E}\!\left[\left(r + \gamma \max_{a'} Q_{\theta^-}(s', a') - Q_\theta(s, a)\right)^2\right]
$$

这就是 **DQN**（Deep Q-Network）的核心思想。

## 代码验证

```python
import numpy as np

# 简单网格世界：4 个状态，2 个动作（左/右），目标是到达状态 3
# 状态: 0, 1, 2, 3（3 是终止态，奖励 1；其他状态奖励 0）
gamma = 0.9
n_states = 4

# 转移矩阵：P[s, a, s'] = P(s'|s,a)
# 动作 0=左，动作 1=右；到边界则原地不动
P = np.zeros((n_states, 2, n_states))
for s in range(n_states):
    P[s, 0, max(0, s-1)]     = 1.0  # 左
    P[s, 1, min(n_states-1, s+1)] = 1.0  # 右

# 奖励：到达状态 3 得 1
R = np.zeros((n_states, 2))
for s in range(n_states):
    R[s, 1] = 1.0 if min(n_states-1, s+1) == 3 else 0.0

# 值迭代（Value Iteration）
V = np.zeros(n_states)
for _ in range(100):
    V_new = np.zeros(n_states)
    for s in range(n_states - 1):   # 状态 3 是终止态，值为 0
        V_new[s] = max(
            R[s, 0] + gamma * P[s, 0] @ V,  # 动作：左
            R[s, 1] + gamma * P[s, 1] @ V   # 动作：右
        )
    V = V_new

print("最优值函数 V*:", V.round(3))  # 离目标越近，值越大
# 最优策略：每个状态选值更大的动作（应该全选"右"）
policy = []
for s in range(n_states - 1):
    q_left  = R[s, 0] + gamma * P[s, 0] @ V
    q_right = R[s, 1] + gamma * P[s, 1] @ V
    policy.append('右' if q_right >= q_left else '左')
print("最优策略:", policy)  # ['右', '右', '右']
```

## 在深度学习中的应用

强化学习的全部理论基础是 Bellman 方程和 DP 原理（Part 3 RL 章节详细展开）。DQN 用神经网络近似 Q 函数。A3C/PPO 用值网络估计优势函数（Advantage），减小策略梯度的方差。AlphaGo/AlphaZero 把蒙特卡洛树搜索（MCTS）与神经网络值函数结合，在游戏 DP 问题上达到超人水平。

下一节讲拉格朗日乘子与 KKT 条件，解决"有约束时如何求最优"的问题，是 SVM、PPO 约束优化和 ADMM 的基础。
