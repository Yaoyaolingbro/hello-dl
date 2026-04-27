# DPO：直接偏好优化

!!! info "参考资料"
    **主要论文**

    - [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — Rafailov et al., NeurIPS 2023
    - [RLHF: Learning to Summarize with Human Feedback](https://arxiv.org/abs/2009.01325) — Stiennon et al., NeurIPS 2020（DPO 要取代的方法）

    **优质讲解**

    - [DPO 解析：从 RLHF 到直接优化](https://huggingface.co/blog/dpo-trl)

## 直觉 (Intuition)

RLHF 流程繁琐：先训练一个奖励模型，再用 PPO 做策略优化，两个训练过程独立，超参数多，调起来麻烦。DPO 的发现是：在 RLHF 的优化问题里，最优策略有解析形式，可以把奖励模型"吸收"进去，直接用偏好数据训练语言模型。输入是（prompt, 好回答, 坏回答）三元组，输出是更符合人类偏好的语言模型。DPO 不需要显式的奖励模型，也不需要 RL 训练，是目前开源 LLM 对齐微调最常用的方法。

## 问题背景

RLHF 训练语言模型对齐的标准流程：

1. 收集偏好数据：给定 prompt，让人类标注者从两个回答 $y_w$（preferred）和 $y_l$（less preferred）中选择更好的
2. 训练奖励模型 $r_\phi(x, y)$，拟合人类偏好（Bradley-Terry 模型）
3. 用 PPO 优化语言模型策略，目标是最大化奖励同时不偏离参考模型太远：

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi}[r_\phi(x, y)] - \beta \text{KL}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

DPO 证明这个优化问题可以跳过步骤 2 和 3，直接用偏好数据端到端训练。

## 方法推导

**第一步**：写出 RLHF 目标的最优解。上述 KL 约束的 RL 问题有解析最优策略：

$$\pi^*(y|x) = \frac{\pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)}{Z(x)}$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ 是归一化常数。

**第二步**：反解奖励。从上式可以把奖励 $r(x, y)$ 用策略表示：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

关键观察：$Z(x)$ 只依赖 $x$，在比较同一 prompt 下两个回答 $y_w$ 和 $y_l$ 时会被消去。

**第三步**：代入 Bradley-Terry 偏好模型。人类选择 $y_w$ 优于 $y_l$ 的概率为：

$$p^*(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

把第二步的 $r$ 代入，$Z(x)$ 消去，得到：

$$p^*(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**第四步**：写出 DPO 训练目标。最大化偏好数据的对数似然，用 $\pi_\theta$ 替代 $\pi^*$：

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

!!! note "直觉小结"
    DPO 损失的含义：对偏好回答 $y_w$ 相对参考模型的对数概率比，要大于非偏好回答 $y_l$ 的对数概率比。等价于：让模型在"好回答"上相对参考模型涨得比"坏回答"多。

## 梯度分析

DPO 损失的梯度告诉我们更新方向：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} \propto -\hat{\sigma} \left[ \beta \nabla_\theta \log \pi_\theta(y_w|x) - \beta \nabla_\theta \log \pi_\theta(y_l|x) \right]$$

其中 $\hat{\sigma} = \sigma(\beta \log(\pi_\theta(y_l|x)/\pi_{\text{ref}}(y_l|x)) - \beta \log(\pi_\theta(y_w|x)/\pi_{\text{ref}}(y_w|x)))$，可以理解为"当前对偏好方向判断错误的程度"——判断越错，梯度权重越大，更新力度越大。

## 代码

DPO 损失的实现：两次前向（win 和 lose），和参考模型对比计算对数概率比。

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_model, ref_model, prompt_ids, win_ids, lose_ids, beta=0.1):
    """
    policy_model: 要优化的模型
    ref_model:    冻结的参考模型（SFT 模型）
    beta:         KL 约束强度，控制偏离参考模型的幅度
    """
    def log_prob(model, input_ids, label_ids):
        """计算 label_ids 在 model 下的对数概率（序列级别）"""
        with torch.no_grad() if model is ref_model else torch.enable_grad():
            logits = model(input_ids).logits[:, :-1, :]
            log_p = F.log_softmax(logits, dim=-1)
            # 取每个位置上真实 token 的对数概率，求和得序列对数概率
            token_log_p = log_p.gather(2, label_ids[:, 1:].unsqueeze(2)).squeeze(2)
            return token_log_p.sum(1)

    # 计算策略模型和参考模型的对数概率
    pi_win  = log_prob(policy_model, prompt_ids, win_ids)
    pi_lose = log_prob(policy_model, prompt_ids, lose_ids)
    ref_win  = log_prob(ref_model, prompt_ids, win_ids)
    ref_lose = log_prob(ref_model, prompt_ids, lose_ids)

    # 对数概率比（implicit reward）
    log_ratio_win  = pi_win  - ref_win
    log_ratio_lose = pi_lose - ref_lose

    # DPO 损失：win 的隐式奖励 > lose 的隐式奖励
    loss = -F.logsigmoid(beta * (log_ratio_win - log_ratio_lose)).mean()
    return loss
```

!!! tip "工程重点"
    DPO 的 $\beta$ 值影响很大：太小会让模型过度偏离参考模型（模型"遗忘"）；太大则对偏好信号不敏感。典型值 $\beta \in [0.05, 0.5]$，需要根据数据质量调节。参考模型的选择也关键——通常用经过 SFT（Supervised Fine-Tuning）的模型，而不是预训练模型。

## 局限与后续工作

DPO 的数学推导依赖 Bradley-Terry 偏好模型的假设，以及偏好标注的一致性。现实中人类标注者不一致，偏好是部分可传递的，DPO 的理论保证在这些情况下会弱化。

RLHF + PPO 虽然流程更复杂，但可以在线采样（online RL）——让模型生成新回答，实时收集奖励信号，适应性更强。DPO 是离线的（offline RL），依赖预先收集的偏好数据集，对数据分布外的情况泛化较差。

SimPO、ORPO 等后续工作进一步简化了 DPO，去掉了对参考模型的依赖。这是 LLM 对齐领域当前最活跃的研究方向之一。

这是强化学习章节的最后一节，也是 Part 3 深入深度学习的核心内容。从策略梯度、Actor-Critic，到 PPO 和 DPO，这条线索在 Part 4 的语言模型对齐（RLHF）和生成模型奖励优化章节里还会再次出现。
