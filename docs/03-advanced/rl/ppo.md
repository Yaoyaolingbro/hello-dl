# PPO：近端策略优化

!!! info "参考资料"
    **主要论文**

    - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al., 2017（PPO 原始论文）
    - [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) — Schulman et al., ICML 2015（PPO 的前身）

    **优质讲解**

    - [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — ICLR 2022 Blog
    - [Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

!!! note "前置依赖"
    PPO 的重要性采样比 $r_t(\theta)$ 来自 [重要性采样](../../01-math/probability/sampling.md) 的基本思路——用旧策略采集的数据估计新策略下的期望。

## 直觉 (Intuition)

Actor-Critic 训练时，如果策略更新步子太大，可能一次就破坏了原来还不错的策略，之后很难恢复。TRPO 提出用 KL 散度约束更新幅度，但实现复杂（需要共轭梯度）。PPO 的贡献是用一个简单的 clip 操作达到类似效果：限制新旧策略的概率比在 $[1-\varepsilon, 1+\varepsilon]$ 之间，步子太大就截断，不再继续更新。输入是当前轨迹，输出是更新后的策略参数。PPO 是目前 RLHF（LLM 对齐）最主流的算法之一。

## 从 TRPO 到 PPO

**TRPO 的出发点**：Actor-Critic 每次更新后，旧数据就失效了（on-policy），但如果能用旧策略 $\pi_{\theta_\text{old}}$ 采集的数据来估计新策略 $\pi_\theta$ 下的期望回报，就可以多次复用数据。重要性采样给出了这个估计：

$$J(\theta) = \mathbb{E}_{a \sim \pi_{\theta_\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \hat{A}(s, a) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ 是重要性比（importance ratio）。

问题在于：当新旧策略差异大时，重要性采样估计的方差会爆炸（[采样章节](../../01-math/probability/sampling.md)里分析过有效样本量 ESS 会急剧下降）。TRPO 通过显式约束 $\text{KL}(\pi_{\theta_\text{old}} \| \pi_\theta) \leq \delta$ 来限制步长，但实现需要二阶优化，代码复杂。

## PPO-Clip

PPO 用 clip 操作直接限制 $r_t(\theta)$ 的范围，避免更新过大：

$$\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t,\; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]$$

其中 $\varepsilon$ 通常取 $0.1$ 或 $0.2$。

**分析两种情况**：

- 当 $\hat{A}_t > 0$（当前动作好于平均水平）：我们想增大 $\pi_\theta(a_t|s_t)$，但 clip 限制 $r_t(\theta) \leq 1 + \varepsilon$，增幅不超过 $\varepsilon$
- 当 $\hat{A}_t < 0$（当前动作差于平均水平）：我们想减小 $\pi_\theta(a_t|s_t)$，clip 限制 $r_t(\theta) \geq 1 - \varepsilon$，减幅也有上限

取 min 操作保证了这是一个悲观下界（pessimistic lower bound）：只在更新方向"安全"时才更新，超出范围就停止。

!!! note "直觉小结"
    PPO clip 的本质是：如果新策略和旧策略差异不大（$r_t \approx 1$），正常更新；如果差异超出阈值 $\varepsilon$，梯度截断，不再进一步偏离旧策略。这是一个保守更新的思路，宁可小步多走，不要一步走太远。

## PPO 完整目标

实践中 PPO 的完整损失包括三项：

$$\mathcal{L}(\theta) = \mathcal{L}^{\text{CLIP}}(\theta) - c_1 \mathcal{L}^{\text{VF}}(\theta) + c_2 \mathcal{H}[\pi_\theta]$$

- $\mathcal{L}^{\text{VF}}$：Critic 的价值函数损失（MSE）
- $\mathcal{H}[\pi_\theta]$：策略熵，鼓励探索
- $c_1, c_2$ 是超参数（典型值 $c_1 = 0.5, c_2 = 0.01$）

## 代码

PPO-Clip 的核心损失计算：

```python
import torch

def ppo_clip_loss(old_log_probs, new_log_probs, advantages, eps=0.2):
    """
    old_log_probs: 旧策略下的动作对数概率（采集数据时记录）
    new_log_probs: 当前策略下的动作对数概率（每次更新重新计算）
    advantages:   GAE 优势估计
    """
    # 重要性比（log 域相减再 exp 更数值稳定）
    ratio = torch.exp(new_log_probs - old_log_probs)

    # PPO-Clip 目标：取正常更新和截断更新的最小值
    clipped_ratio = ratio.clamp(1 - eps, 1 + eps)
    policy_loss = -torch.min(ratio * advantages,
                             clipped_ratio * advantages).mean()
    return policy_loss

def ppo_train_epoch(policy, value_net, rollout_data, optimizer,
                    eps=0.2, n_epochs=10):
    """对同一批数据重复更新 n_epochs 次（PPO 可多次复用旧数据）"""
    for _ in range(n_epochs):
        s, a, old_logp, adv, returns = rollout_data
        new_logp = policy.log_prob(s, a)

        policy_loss = ppo_clip_loss(old_logp, new_logp, adv, eps)
        value_loss  = 0.5 * ((value_net(s).squeeze() - returns) ** 2).mean()
        entropy     = policy.entropy(s).mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        optimizer.zero_grad(); loss.backward(); optimizer.step()
```

!!! tip "工程重点"
    PPO 有很多细节影响性能：GAE 的 $\lambda$ 值、advantage 标准化（减均值除方差）、梯度裁剪（`clip_grad_norm_`）、value loss 的 clip（参考 [37 Implementation Details](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)）。原论文报告的超参数不一定是最优的，很多工程细节是后来社区发现的。

## 局限与后续工作

PPO 仍然是 on-policy 方法（虽然允许少量复用），样本效率相比 SAC 等 off-policy 方法低。在需要海量数据的大规模问题（Minecraft 开放世界、RLHF 的大语言模型）上，PPO 的计算成本很高。

DPO（下一节）是一个绕开 PPO 的方法：在 LLM 对齐场景里，直接用偏好数据优化策略，不需要显式的奖励模型和 RL 训练循环，工程上简单很多。
