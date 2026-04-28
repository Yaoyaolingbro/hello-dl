# 价值函数方法

!!! info "参考资料"
    **主要论文**

    - [Playing Atari with Deep RL (DQN)](https://arxiv.org/abs/1312.5602) — Mnih et al., Nature 2015
    - [Deep RL with Double Q-learning](https://arxiv.org/abs/1509.06461) — van Hasselt et al., AAAI 2016
    - [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) — Schaul et al., ICLR 2016

    **优质讲解**

    - [Lilian Weng: A (Long) Peek into RL](https://lilianweng.github.io/posts/2018-02-19-rl-overview/)
    - [Spinning Up: DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html)

## 直觉 (Intuition)

策略梯度直接学策略 $\pi_\theta(a|s)$，需要大量轨迹才能估计梯度方向。价值函数方法换了一个角度：学 Q 函数 $Q(s, a)$——每个（状态, 动作）对的"长期价值"——然后贪心地选价值最大的动作。输入是状态（和动作），输出是标量 Q 值。核心算法是 Q-learning，用 Bellman 最优方程做 TD 更新，不需要采集完整轨迹，每步都能更新。

## Q-learning

Q-learning 是一个 off-policy 的 TD 算法：不管当前用什么策略采样，更新的目标都是最优策略的 Q 值。

**更新规则**：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)\right]$$

其中 $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ 是 TD 目标（target），$[\ \cdot - Q(s_t, a_t)]$ 是 TD 误差。

**为什么用 $\max_{a'}$？** 因为我们要估计的是最优 Q 值——在 $s_{t+1}$ 之后执行最优策略能得到的价值。这和用来采样数据的策略（比如 $\varepsilon$-greedy）无关，所以叫 off-policy。

Q-learning 的收敛性在表格状态空间（Q 表）里有理论保证，但当状态空间大到无法枚举时，需要用神经网络来近似 Q 函数。

## DQN：深度 Q 网络

Mnih et al. (2015) 首次用深度神经网络代替 Q 表，成功在 49 个 Atari 游戏上达到或超过人类水平。

直接用神经网络做 Q 函数近似会有两个致命问题：

**问题一：数据相关性**。连续采集的 $(s_t, a_t, r_t, s_{t+1})$ 在时间上高度相关，神经网络训练假设数据独立同分布，用相关数据更新会导致不稳定甚至发散。

**解决：经验回放（Experience Replay）**。把采集到的转移 $(s, a, r, s')$ 存入一个回放池（replay buffer），每次更新时从中随机采样一个 mini-batch。这打破了时间相关性，让训练更稳定，同时也允许重复利用历史数据。

**问题二：目标不稳定**。TD 目标 $r + \gamma \max_{a'} Q_\theta(s', a')$ 里用的和被更新的是同一个网络 $Q_\theta$，每次更新 $\theta$ 目标也跟着变，形成一个"追着移动目标跑"的困境。

**解决：目标网络（Target Network）**。维护一个额外的目标网络 $Q_{\theta^-}$，参数每隔若干步从主网络复制（hard update）或缓慢跟踪（soft update：$\theta^- \leftarrow \tau \theta + (1-\tau)\theta^-$，$\tau \approx 0.005$）。TD 目标改用 $\theta^-$ 计算，使目标在一段时间内保持稳定。

## Double DQN

DQN 的 Q 值常常过估计。原因：TD 目标里对 $s'$ 的最优动作和 Q 值评估用同一个网络，系统性地高估了最大值（噪声里取 max 总会偏高）。

Double DQN 把"选动作"和"评估价值"分开：

$$y = r + \gamma Q_{\theta^-}\!\left(s',\; \underbrace{\arg\max_{a'} Q_\theta(s', a')}_{\text{主网络选动作}}\right)$$

主网络选择下一步的动作，目标网络评估这个动作的价值。原论文（Table 1）显示，Double DQN 在多数 Atari 游戏上比 DQN 有显著提升，过估计问题明显缓解。

!!! note "直觉小结"
    经验回放 + 目标网络是让 Q-learning 和神经网络配合工作的两个关键 trick。没有前者，数据相关会让训练不稳定；没有后者，目标漂移会导致训练发散。

## 代码

DQN 训练核心：经验回放采样 + TD 更新 + soft 目标网络更新。

```python
import torch
import torch.nn.functional as F
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buf.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s_next, done = zip(*batch)
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(a, dtype=torch.long),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(s_next, dtype=torch.float32),
                torch.tensor(done, dtype=torch.float32))

def dqn_update(q_net, q_target, optimizer, replay_buf, batch_size=64, gamma=0.99):
    if len(replay_buf.buf) < batch_size:
        return
    s, a, r, s_next, done = replay_buf.sample(batch_size)

    # 当前 Q 值
    q_vals = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

    # Double DQN 目标：主网络选动作，目标网络评估
    with torch.no_grad():
        a_best = q_net(s_next).argmax(1)            # 主网络选最优动作
        q_next = q_target(s_next).gather(1, a_best.unsqueeze(1)).squeeze(1)
        td_target = r + gamma * (1 - done) * q_next

    loss = F.mse_loss(q_vals, td_target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()

    # Soft 更新目标网络（tau=0.005）
    tau = 0.005
    for p, p_t in zip(q_net.parameters(), q_target.parameters()):
        p_t.data.copy_(tau * p.data + (1 - tau) * p_t.data)
```

## 局限与后续工作

DQN 系列只能处理**离散动作空间**：$\arg\max_a Q(s, a)$ 在动作数量很多（或连续）时不可行。对机器人控制等连续动作任务，要用 Actor-Critic 方法（SAC、TD3）。

DQN 的另一个限制是 on-screen state：直接从像素学，所有信息必须从像素中提取。现实机器人任务的状态空间更复杂，状态表示（state representation learning）是一个重要的研究方向。

下一节讲 PPO，它把策略梯度和价值函数方法的优点结合起来：用 Actor-Critic 架构，但在更新策略时加了显式的约束，防止一次更新步子太大破坏已有策略。
