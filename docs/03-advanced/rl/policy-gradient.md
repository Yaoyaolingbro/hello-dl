# 策略梯度

!!! info "参考资料"
    **主要论文**

    - [Policy Gradient Methods for RL with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — Sutton et al., NeurIPS 1999（策略梯度定理）
    - [Simple Statistical Gradient-Following Algorithms for Connectionist RL](https://link.springer.com/article/10.1007/BF00992696) — Williams 1992（REINFORCE 原始论文）

    **优质讲解**

    - [Lilian Weng: Policy Gradient Algorithms](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
    - [Spinning Up: Intro to RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

!!! note "前置依赖"
    策略梯度的方差分析用到 [蒙特卡洛采样与重要性采样](../../01-math/probability/sampling.md) 里的方差估计和基线减小方差的技巧。

## 直觉 (Intuition)

值迭代需要知道或估计转移概率 $P(s'|s,a)$，在高维连续状态空间里很难做。策略梯度方法完全绕开这个问题：直接把策略 $\pi_\theta(a|s)$ 参数化为神经网络，然后对期望回报做梯度上升。输入是当前状态，输出是动作的概率分布。关键推导是"策略梯度定理"：期望回报对参数 $\theta$ 的梯度可以写成一个可以用采样估计的期望式子，不需要对环境求导。

## 策略梯度定理

目标是最大化期望回报：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$$

其中 $\tau = (s_0, a_0, r_1, s_1, a_1, \ldots)$ 是一条轨迹（trajectory），$G(\tau) = \sum_t \gamma^t r_{t+1}$。

**策略梯度定理**给出了梯度的形式。推导用到一个 log-derivative trick：对任意分布 $p_\theta(\tau)$，$\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)$。

轨迹的对数概率：

$$\log p_\theta(\tau) = \log p(s_0) + \sum_t \left[\log \pi_\theta(a_t|s_t) + \log P(s_{t+1}|s_t, a_t)\right]$$

对 $\theta$ 求梯度时，$\log p(s_0)$ 和 $\log P(s_{t+1}|s_t, a_t)$（环境动态）不含 $\theta$，消去，得到：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G(\tau) \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

这就是策略梯度定理：**梯度方向 = 回报 × 动作对数概率的梯度之和**。直觉上就是：回报高的轨迹，增大其出现概率；回报低的轨迹，减小其出现概率。

## REINFORCE 算法

REINFORCE 是策略梯度的最基础实现：采集 $N$ 条完整轨迹，用 Monte Carlo 估计梯度：

$$\hat{\nabla}_\theta J(\theta) = \frac{1}{N} \sum_{n=1}^N G(\tau^{(n)}) \sum_t \nabla_\theta \log \pi_\theta(a_t^{(n)} | s_t^{(n)})$$

收集到轨迹后做一次梯度上升更新 $\theta$，然后用新策略重新采样，如此循环。

REINFORCE 的问题是方差极大——不同轨迹的回报差异悬殊，梯度估计不稳定，需要非常小的学习率和大量样本。

## 基线减小方差

用一个与动作无关的**基线（baseline）** $b(s_t)$ 减小方差，不改变梯度期望值：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \left(G_t - b(s_t)\right) \nabla_\theta \log \pi_\theta(a_t | s_t) \right]$$

其中 $G_t = \sum_{k \geq t} \gamma^{k-t} r_{k+1}$ 是从时刻 $t$ 起的折扣回报（因果性：$t$ 之前的动作不影响 $t$ 之后的奖励，所以去掉了）。

**为什么基线不改变梯度期望？** 因为：

$$\mathbb{E}_{a \sim \pi_\theta}\left[b(s) \nabla_\theta \log \pi_\theta(a|s)\right] = b(s) \nabla_\theta \underbrace{\sum_a \pi_\theta(a|s)}_{=1} = 0$$

最常用的基线是状态价值函数的估计 $\hat{V}(s_t)$（用一个单独的价值网络拟合）。此时 $(G_t - \hat{V}(s_t))$ 叫做**优势函数（Advantage Function）**的 Monte Carlo 估计，衡量"当前动作比平均水平好多少"。这是 Actor-Critic 方法的出发点。

!!! note "直觉小结"
    基线减小方差的原理和 [蒙特卡洛章节](../../01-math/probability/sampling.md) 里的控制变量（control variate）方法完全一致：减去一个不改变期望但和估计量相关的量，降低采样噪声。

## 代码

REINFORCE with baseline 的核心训练逻辑：

```python
import torch
import torch.nn as nn

def reinforce_update(policy_net, value_net, trajectories, optimizer_p, optimizer_v, gamma=0.99):
    """
    trajectories: list of (states, actions, rewards) per episode
    """
    policy_loss_total, value_loss_total = 0.0, 0.0
    for states, actions, rewards in trajectories:
        states  = torch.tensor(states,  dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        # 计算折扣回报 G_t（从后往前累积）
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)
        # 标准化回报，减小方差
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        values    = value_net(states).squeeze()
        advantage = returns - values.detach()   # detach：基线不参与策略梯度

        log_probs = policy_net(states).log_softmax(-1)
        selected  = log_probs[range(len(actions)), actions]
        policy_loss = -(selected * advantage).mean()

        value_loss = nn.functional.mse_loss(values, returns)
        policy_loss_total += policy_loss
        value_loss_total  += value_loss

    optimizer_p.zero_grad(); policy_loss_total.backward(retain_graph=True); optimizer_p.step()
    optimizer_v.zero_grad(); value_loss_total.backward(); optimizer_v.step()
```

## 局限与后续工作

REINFORCE 的两个核心问题：

第一，样本效率低。每次更新后必须丢掉旧轨迹（on-policy），采样成本高。PPO 通过重要性采样允许复用少量旧数据。

第二，回报估计方差大。用完整轨迹的 Monte Carlo 估计偏差小但方差大。Actor-Critic 用 bootstrap（TD 估计）替代 Monte Carlo 估计，大幅降低方差，代价是引入偏差。

下一节讲 Actor-Critic，它在策略梯度的基础上引入时序差分（Temporal Difference，TD）来估计优势函数，是目前大多数深度 RL 方法的基础。
