# 强化学习基础

!!! info "参考资料"
    **主要教材**

    - [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) — Sutton & Barto, 2018（第二版，标准教材）
    - [CS 285: Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) — Sergey Levine, UC Berkeley

    **优质讲解**

    - [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/)
    - [David Silver RL 课程](https://www.davidsilver.uk/teaching/)

!!! note "前置依赖"
    本节里的 Bellman 方程和值迭代在 [动态规划](../../01-math/optimization/dynamic-programming.md) 里已从最优子结构角度推导过，建议先看那一节。这里从 RL 的视角重新出发，补充策略（Policy）和探索的概念。

## 直觉 (Intuition)

强化学习解决的问题是：一个 Agent 在环境里反复试错，学会做出最优决策。输入是当前的环境状态 $s$，输出是要执行的动作 $a$，目标是最大化长期累积奖励。和监督学习不同，RL 没有标准答案——只有执行动作后环境给的奖励信号，而且奖励可能稀疏、延迟。就像教小孩走路，不是告诉他每一步怎么迈，而是摔跤了就扣分，走成功了就加分。

## 马尔可夫决策过程

RL 的数学框架是马尔可夫决策过程（Markov Decision Process，MDP），由五元组 $(\mathcal{S}, \mathcal{A}, P, r, \gamma)$ 定义：

- $\mathcal{S}$：状态空间（state space）
- $\mathcal{A}$：动作空间（action space）
- $P(s'|s, a)$：状态转移概率，执行动作 $a$ 后从状态 $s$ 转移到 $s'$ 的概率
- $r(s, a)$：即时奖励函数
- $\gamma \in [0, 1)$：折扣因子（discount factor），控制对未来奖励的重视程度

"马尔可夫"的意思是：当前状态 $s_t$ 包含了所有用于决策的历史信息，$s_{t+1}$ 只依赖 $s_t$ 和 $a_t$，不依赖更早的状态。

## 策略、回报与价值函数

**策略（Policy）** $\pi(a|s)$ 是给定状态 $s$ 下选择动作 $a$ 的概率分布。目标是找到最优策略 $\pi^*$。

**折扣回报（Return）** 是从时刻 $t$ 出发的累积奖励：

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

折扣因子 $\gamma < 1$ 保证这个无穷和是有限的，同时也表达了"近期奖励比远期奖励更重要"的偏好。

**状态价值函数（State Value Function）** $V^\pi(s)$ 是从状态 $s$ 出发，按策略 $\pi$ 执行所能获得的期望回报：

$$V^\pi(s) = \mathbb{E}_\pi[G_t | s_t = s]$$

**动作价值函数（Action Value Function / Q-function）** $Q^\pi(s, a)$ 是在状态 $s$ 执行动作 $a$，之后按 $\pi$ 执行所能获得的期望回报：

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | s_t = s, a_t = a]$$

两者的关系：$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s, a)$。

## Bellman 方程

Bellman 方程是 DP（动态规划）在 RL 中的核心工具，把"长期价值"分解为"当前奖励 + 下一步价值"：

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) \left[ r(s,a) + \gamma V^\pi(s') \right]$$

对应的最优价值函数满足 Bellman 最优方程：

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) \left[ r(s,a) + \gamma V^*(s') \right]$$

最优策略直接从 $V^*$ 提取：$\pi^*(s) = \arg\max_a \sum_{s'} P(s'|s,a)[r(s,a) + \gamma V^*(s')]$。

这和 [动态规划章节](../../01-math/optimization/dynamic-programming.md) 里的 Bellman 最优方程一字不差——RL 的数学基础就是 DP，区别在于 RL 面对的是未知的 $P$ 和 $r$，必须通过与环境交互来估计。

!!! note "直觉小结"
    Bellman 方程告诉我们：当前状态的价值 = 当前奖励 + 下一状态的折扣价值。这是递归定义，可以用迭代方法（值迭代、策略迭代）求解。

## 探索与利用

RL 有一个监督学习里没有的难题：**探索-利用权衡（Exploration-Exploitation Tradeoff）**。

Agent 必须在两件事之间平衡：

- **利用（Exploitation）**：选当前估计价值最高的动作（贪心策略）
- **探索（Exploration）**：尝试价值估计不确定的动作，可能发现更好的策略

常用策略：$\varepsilon$-greedy，以概率 $\varepsilon$ 随机探索，以 $1-\varepsilon$ 利用当前最优。$\varepsilon$ 一般从大到小衰减。

!!! warning "常见误区"
    很多人以为"只要奖励够大，Agent 会自然去探索"。实际上在稀疏奖励环境（比如大多数时候奖励为 0 的棋盘游戏），纯贪心策略可能永远碰不到正奖励，永远不知道正确的方向在哪里。探索机制的设计是 RL 工程里最棘手的部分之一。

## 代码

用值迭代求解简单 GridWorld 的最优策略（复用 [动态规划章节](../../01-math/optimization/dynamic-programming.md) 的思路，加入折扣因子）：

```python
import numpy as np

# 4x4 格点世界，右下角(15)是终点，其余格子奖励-1
n_states, n_actions = 16, 4
gamma = 0.9

def get_next_state(s, a):
    """0=上,1=下,2=左,3=右"""
    row, col = s // 4, s % 4
    if a == 0 and row > 0: row -= 1
    elif a == 1 and row < 3: row += 1
    elif a == 2 and col > 0: col -= 1
    elif a == 3 and col < 3: col += 1
    return row * 4 + col

V = np.zeros(n_states)
for _ in range(500):
    V_new = np.copy(V)
    for s in range(n_states - 1):  # 终止状态 V=0
        rewards = []
        for a in range(n_actions):
            s_next = get_next_state(s, a)
            rewards.append(-1 + gamma * V[s_next])
        V_new[s] = max(rewards)
    V = V_new

# 提取最优策略
policy = np.zeros(n_states, dtype=int)
for s in range(n_states - 1):
    q_vals = [-1 + gamma * V[get_next_state(s, a)] for a in range(n_actions)]
    policy[s] = np.argmax(q_vals)
print(V.reshape(4, 4).round(2))
# 右下角价值最高，越靠近终点价值越大
```

## 在深度学习中的应用

RL 的框架现在已经渗透到语言模型训练里：RLHF（Reinforcement Learning from Human Feedback）把"人类打分"当作奖励信号，用 PPO 微调语言模型的策略，使其更符合人类偏好。DPO 进一步简化了这个流程（后面 DPO 章节详述）。

下一节讲策略梯度，它让策略 $\pi_\theta$ 用神经网络参数化，直接对期望回报做梯度上升——不需要显式计算 $V^*$，适合连续动作空间和部分可观测环境。
