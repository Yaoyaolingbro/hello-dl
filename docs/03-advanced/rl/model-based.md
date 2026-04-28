# 基于模型的强化学习

!!! info "参考资料"
    **主要论文**

    - [World Models](https://arxiv.org/abs/1803.10122) — Ha & Schmidhuber, NeurIPS 2018
    - [Dream to Control (Dreamer)](https://arxiv.org/abs/1912.01603) — Hafner et al., ICLR 2020
    - [DreamerV3](https://arxiv.org/abs/2301.04104) — Hafner et al., 2023
    - [MBPO](https://arxiv.org/abs/1906.08253) — Janner et al., NeurIPS 2019

    **优质讲解**

    - [Lilian Weng: Model-Based RL](https://lilianweng.github.io/posts/2019-06-23-meta-rl/)

## 直觉 (Intuition)

Model-free RL（PPO、SAC）需要大量真实环境交互才能学习——每次走一步，得到一个经验。基于模型的 RL（Model-Based RL，MBRL）的思路是：先学一个环境的"世界模型"（world model），然后在世界模型里做想象（imagination），用虚拟经验来训练策略，减少真实交互次数。输入是少量真实轨迹，输出是一个可以"在脑子里模拟未来"的世界模型加策略。核心想法是以模型学习换样本效率。

## 问题背景

Model-free RL 的样本效率是工程上的主要瓶颈。在机器人控制里，一次真实机械臂实验可能需要几分钟，而学会一个任务需要几十万步——这是根本不现实的时间成本。

人类学习的效率高得多，部分原因是我们有关于物理世界的心理模型：不需要实际跌倒几万次才学会走路，可以在脑子里"模拟"会发生什么，做出预测性决策。MBRL 试图给 Agent 类似的能力。

## 世界模型的组成

一个标准的世界模型包含三个组件：

**表示模型（Representation Model）** 把高维观测 $\mathbf{o}_t$（像素图像）压缩为低维隐状态 $\mathbf{h}_t$：

$$\mathbf{h}_t = \text{Encoder}(\mathbf{o}_t)$$

**动态模型（Dynamics Model）** 预测执行动作 $\mathbf{a}_t$ 后的下一隐状态：

$$\hat{\mathbf{h}}_{t+1} = f(\mathbf{h}_t, \mathbf{a}_t)$$

**奖励模型（Reward Model）** 从隐状态预测即时奖励：

$$\hat{r}_t = g(\mathbf{h}_t)$$

有了这三个组件，就可以完全在隐空间里展开轨迹，不需要真实环境。

## Dreamer：在梦中训练

Dreamer 是目前 MBRL 在视觉任务上最成功的方法之一。它用一个 RSSM（Recurrent State Space Model）作为世界模型，把表示模型和动态模型统一为一个循环网络：

$$\mathbf{h}_t = f(\mathbf{h}_{t-1}, \mathbf{a}_{t-1}), \quad \mathbf{z}_t \sim q(\mathbf{z}_t | \mathbf{h}_t, \mathbf{o}_t)$$

其中 $\mathbf{z}_t$ 是随机性的"当前帧信息"（类似 VAE 的隐变量），$\mathbf{h}_t$ 是确定性的循环状态（携带历史信息）。两者合在一起构成完整的模型状态。

**Dreamer 的训练分两阶段**：

第一阶段：用真实环境数据训练世界模型（最大化 ELBO，同时预测奖励和终止状态）。

第二阶段：固定世界模型，在"想象轨迹"里训练策略和价值网络（Actor-Critic，完全不用真实环境）：

$$\text{imagine: } \hat{\mathbf{h}}_1, \hat{\mathbf{h}}_2, \ldots, \hat{\mathbf{h}}_H \text{ using dynamics model, compute } \hat{r}_1, \ldots, \hat{r}_H$$

原论文（Dreamer Table 1）显示，在 DMControl 任务上，Dreamer 用约 200k 真实步就能达到 PPO 需要 10M 步才能达到的性能——样本效率提升约 50 倍。

!!! note "直觉小结"
    世界模型相当于给 Agent 一个"模拟器"。真实数据用来更新模拟器，策略在模拟器里大量练习。真实交互很贵，模拟器里的推演很便宜——这是 MBRL 换取样本效率的根本逻辑。

## MBPO：模型数据增强

MBPO（Deisenroth et al. 的框架，Janner et al. 实现）是一个更简单的 MBRL 变体：不在隐空间想象，而是用世界模型生成短轨迹片段，和真实数据混在一起训练 SAC。

关键结论：世界模型只需要预测很短的展开（$k$ 步，通常 $k \leq 10$），预测误差不会累积到不可控的程度，又能提供足够多的虚拟数据。MBPO 在 MuJoCo 任务上样本效率比 SAC 高约 10-20 倍。

!!! warning "常见误区"
    "世界模型越精确越好"是直觉上合理但实践上不完全成立的想法。Dreamer 的世界模型其实相当粗糙（解码出来的图像模糊），但策略仍然学得很好。原因是策略只需要世界模型在"与奖励相关的关键信息"上准确，对无关细节的误差并不敏感。

## 代码

MBPO 风格的数据增强：用学好的动态模型从真实状态出发展开短轨迹。

```python
import torch

@torch.no_grad()
def model_rollout(dynamics_model, reward_model, real_states,
                  policy, rollout_length=5):
    """
    从真实状态出发，用世界模型展开 rollout_length 步
    生成虚拟轨迹数据用于策略训练
    """
    virtual_data = []
    s = real_states
    for _ in range(rollout_length):
        a = policy.sample(s)                         # 当前策略选动作
        s_next = dynamics_model(s, a)                # 预测下一状态
        r = reward_model(s, a)                       # 预测奖励

        virtual_data.append((s, a, r, s_next))
        s = s_next                                   # 继续展开

        # 短轨迹防止误差累积：超过 rollout_length 停止
    return virtual_data
```

## 局限与后续工作

MBRL 的核心难点是**模型误差累积（compounding error）**：动态模型每步有一点误差，展开很长的轨迹后误差指数级放大，策略在"梦境"里学到的行为到真实环境里可能完全失效。这就是为什么 MBPO 只做短展开，Dreamer 也只用 15 步想象轨迹。

另一个方向是直接用视频生成模型（如 Sora 的前身）作为世界模型，在视频帧级别做想象，这目前是 embodied AI 的重要研究方向。

下一节讲 DPO，它把强化学习"藏"进了语言模型的损失函数里，绕开了奖励建模和策略梯度这两个 RLHF 最麻烦的部分。
