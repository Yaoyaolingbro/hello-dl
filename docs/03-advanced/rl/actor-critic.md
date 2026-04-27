# Actor-Critic

!!! info "参考资料"
    **主要论文**

    - [Asynchronous Methods for Deep RL (A3C)](https://arxiv.org/abs/1602.01783) — Mnih et al., ICML 2016
    - [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1801.01290) — Haarnoja et al., ICML 2018
    - [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — Schulman et al., ICLR 2016（GAE）

    **优质讲解**

    - [Spinning Up: Actor-Critic Methods](https://spinningup.openai.com/en/latest/algorithms/sac.html)

## 直觉 (Intuition)

REINFORCE 用完整轨迹的回报估计梯度，方差大。Actor-Critic 用两个网络配合解决这个问题：Actor（策略网络）决定做什么，Critic（价值网络）评估这个决策有多好，然后用 Critic 的评估来指导 Actor 更新。输入是当前状态，Actor 输出动作，Critic 输出当前状态的价值。核心思路是用 TD（时序差分）估计替代 Monte Carlo 估计：不用等完整轨迹结束，每一步都能更新，方差更小，但会引入偏差。

## 时序差分与优势函数

**TD 目标（TD Target）**：不用完整轨迹，只走一步就做一个 bootstrap 估计：

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

这叫 TD 误差（TD Error）。$\delta_t > 0$ 表示这步比预期好，$\delta_t < 0$ 表示比预期差。

**优势函数（Advantage Function）**：

$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$

衡量"在状态 $s_t$ 执行动作 $a_t$，比从 $s_t$ 出发的平均期望好多少"。用 TD 误差近似优势函数：$A(s_t, a_t) \approx \delta_t$（单步 TD）。

**广义优势估计（Generalized Advantage Estimation，GAE）**：Schulman et al. 提出的折中方案，用 $\lambda$ 在单步 TD（低方差、高偏差）和 Monte Carlo（无偏差、高方差）之间平滑插值：

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

$\lambda = 0$ 退化为单步 TD，$\lambda = 1$ 退化为 Monte Carlo。实践中 $\lambda = 0.95$ 是常用值。

## A2C / A3C

A2C（Advantage Actor-Critic）：同步多环境，每个环境并行收集轨迹片段，用 GAE 估计优势，同时更新 Actor 和 Critic：

- Actor 损失：$\mathcal{L}_\text{actor} = -\mathbb{E}_t[\hat{A}_t \cdot \log \pi_\theta(a_t|s_t)]$
- Critic 损失：$\mathcal{L}_\text{critic} = \mathbb{E}_t[(r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t))^2]$
- 熵正则：$+\alpha \mathcal{H}[\pi_\theta(\cdot|s_t)]$，鼓励探索（防止策略过早收敛到确定性）

A3C（Asynchronous A2C）把同步改成异步：多个 worker 并行与环境交互，异步更新全局参数。速度快，但异步梯度可能过时。实践中 A2C 的同步版本已经足够快，A3C 现在很少用。

## SAC：最大熵强化学习

SAC（Soft Actor-Critic）是目前连续动作空间任务的默认选择，在 A2C 上做了两个关键改进：

**改进一：最大熵目标（Maximum Entropy RL）**。目标从最大化回报改为最大化回报 + 策略熵之和：

$$J(\pi) = \mathbb{E}_\tau\left[\sum_t \left(r_t + \alpha \mathcal{H}[\pi(\cdot|s_t)]\right)\right]$$

其中 $\alpha$ 是温度参数（可自动调节）。这个目标让策略保持一定随机性，自然地平衡探索与利用，避免过早陷入局部最优。

**改进二：Off-Policy 训练**。SAC 用经验回放池（replay buffer）存储历史数据，可以重复利用旧样本，样本效率远高于 on-policy 方法。Critic 用两个 Q 网络（取最小值）来减小 Q 值过估计问题（double Q-trick）。

!!! note "直觉小结"
    最大熵目标的直觉：在能获得高回报的前提下，策略应该尽量"随机"（保留更多可能性）。这让 SAC 在稀疏奖励和多峰奖励任务上表现比确定性策略强得多。

!!! tip "工程重点"
    SAC 的温度参数 $\alpha$ 不需要手动调——可以自动学习：设定目标熵 $\mathcal{H}_\text{target} = -|\mathcal{A}|$（连续动作时用 $-\dim(\mathcal{A})$），每步用梯度下降调节 $\alpha$ 使策略熵接近目标值。

## 代码

SAC 的 Critic 更新（双 Q 网络 + 目标网络）：

```python
import torch
import torch.nn.functional as F

def sac_critic_update(q1, q2, q1_target, q2_target, policy,
                      s, a, r, s_next, done, alpha, gamma=0.99):
    """SAC Critic 更新：双 Q + 软目标网络"""
    with torch.no_grad():
        # 从新策略采样下一动作
        a_next, log_prob_next = policy.sample(s_next)
        # 取两个目标网络 Q 的最小值，减少 Q 过估计
        q_next = torch.min(
            q1_target(s_next, a_next),
            q2_target(s_next, a_next)
        ) - alpha * log_prob_next                      # 最大熵加熵奖励
        td_target = r + gamma * (1 - done) * q_next

    loss_q1 = F.mse_loss(q1(s, a), td_target)
    loss_q2 = F.mse_loss(q2(s, a), td_target)
    return loss_q1 + loss_q2
```

## 局限与后续工作

Actor-Critic 的主要难点在于 Critic 的准确性：如果价值估计偏差大，Actor 的更新方向就会偏错。在奖励稀疏的任务（比如机械臂操作）里，Critic 长时间得不到有效信号，训练会停滞。

PPO 在 Actor-Critic 的基础上加了约束：限制每次策略更新的幅度，防止更新过大破坏已经学到的行为。这是目前最广泛使用的 RL 算法之一，下下节详细展开。

下一节先讲价值函数方法（DQN）——它不维护显式的策略，直接从 Q 值里提取最优动作，适合离散动作空间。
