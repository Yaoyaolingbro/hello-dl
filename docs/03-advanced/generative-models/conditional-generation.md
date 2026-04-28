# 条件生成与 Classifier-Free Guidance

!!! info "参考资料"
    **主要论文**

    - [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — Ho & Salimans, NeurIPS 2021 Workshop
    - [Classifier Guidance](https://arxiv.org/abs/2105.05233) — Dhariwal & Nichol, NeurIPS 2021
    - [DALL·E 2](https://arxiv.org/abs/2204.06125) — Ramesh et al., 2022
    - [Photorealistic Text-to-Image with CLIP](https://arxiv.org/abs/2204.13807) — Nichol et al., 2022

    **优质讲解**

    - [Classifier-Free Diffusion Guidance 解析](https://sander.ai/2022/05/26/guidance.html) — Sander Dieleman

## 直觉 (Intuition)

无条件生成能产出多样的样本，但我们通常想要更精确的控制：给定文本"一只橘猫坐在月亮上"，生成符合描述的图像。条件生成的任务就是在采样时引入额外的条件信号 $\mathbf{c}$，让生成结果向 $\mathbf{c}$ 对应的方向偏移。输入是噪声加条件（文本、类别、图像等），输出是满足条件的高质量样本。Classifier-Free Guidance（CFG）是目前最主流的方法：同时训练有条件和无条件两个生成器，采样时在两者之间做加权外推，放大条件影响。

## 问题背景

生成模型的无条件版本学习 $p(\mathbf{x})$，条件版本学习 $p(\mathbf{x}|\mathbf{c})$。朴素的条件生成是把 $\mathbf{c}$ 直接拼接到输入——这能工作，但条件的影响往往太弱，生成的图像只是"隐约"符合条件。

更本质的问题是：我们希望采样时沿着"让 $p(\mathbf{x}|\mathbf{c})$ 更大"的方向走，也就是说，需要 $\nabla_\mathbf{x} \log p(\mathbf{x}|\mathbf{c})$ 的信息。

## Classifier Guidance

Dhariwal & Nichol (2021) 提出的方法：用 Bayes 定理把条件概率分解：

$$\nabla_\mathbf{x} \log p(\mathbf{x}|\mathbf{c}) = \nabla_\mathbf{x} \log p(\mathbf{x}) + \nabla_\mathbf{x} \log p(\mathbf{c}|\mathbf{x})$$

第一项是无条件扩散模型的 score，第二项是一个额外分类器在带噪图像 $\mathbf{x}_t$ 上的梯度。采样时在原来的去噪方向上叠加一个分类器梯度：

$$\tilde{\boldsymbol{\varepsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1-\bar{\alpha}_t} \cdot s \cdot \nabla_{\mathbf{x}_t} \log p_\phi(\mathbf{c}|\mathbf{x}_t)$$

其中 $s > 1$ 是 guidance 强度。

问题：需要训练一个对带噪图像有效的分类器，工程上很麻烦。

## Classifier-Free Guidance

Ho & Salimans (2021) 的 CFG 方法：不训练额外分类器，而是同时训练有条件（$\mathbf{c}$ 传入）和无条件（$\mathbf{c}$ 用空字符串 $\varnothing$ 替代）两个模式，共用同一个网络，训练时随机以概率 $p_{\text{uncond}}$（一般 10%~20%）丢弃条件输入。

采样时做加权外推（linear extrapolation）：

$$\tilde{\boldsymbol{\varepsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = (1 + w) \cdot \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - w \cdot \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t, \varnothing)$$

其中 $w \geq 0$ 是 guidance scale（也写作 $s - 1$，$s$ 是 classifier guidance 里的 guidance strength，两种写法等价）。

!!! note "直觉小结"
    CFG 做的事是：沿着"有条件方向"多走一点，同时沿着"无条件方向"往回退一步。$w = 0$ 是纯条件生成，$w$ 越大，生成结果越贴近条件 $\mathbf{c}$，但多样性越低、样本质量（FID）先降后升——guidance scale 需要根据任务调节。

## Cross-Attention：条件如何注入网络

主流的条件注入方式是 Cross-Attention，在 UNet（或 DiT）的每个模块里插入 cross-attention 层：

- Query（$\mathbf{Q}$）来自图像的中间特征
- Key（$\mathbf{K}$）和 Value（$\mathbf{V}$）来自文本（或其他条件）的编码器输出

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\right)\mathbf{V}$$

通过 cross-attention，图像的每个空间位置都能"查询"文本特征，建立局部对应关系（比如"月亮"这个词对应图像右上角的区域）。

## 代码

CFG 采样：同一网络分别做有条件和无条件前向，外推合并。

```python
@torch.no_grad()
def cfg_sample(model, x_T, text_emb, null_emb, guidance_scale=7.5,
               noise_schedule=None, steps=50):
    """
    CFG 采样：两次前向，外推到更强的条件方向
    guidance_scale: 典型值 5-15；越大越"符合文本"，但多样性下降
    """
    x = x_T
    # 把有条件和无条件 batch 合并，减少两次前向的开销
    for t_idx in reversed(range(steps)):
        t = torch.full((x.shape[0],), t_idx)

        # 有条件预测
        eps_cond = model(x, t, text_emb)
        # 无条件预测（用 null token 替代文本）
        eps_uncond = model(x, t, null_emb)

        # CFG 外推：沿"有条件-无条件"方向放大
        eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        # 用合并后的 eps 做 DDIM 去噪一步
        x = ddim_step(x, eps, t_idx, noise_schedule)
    return x
```

## 局限与后续工作

CFG 最明显的代价是推理速度：每步需要两次前向传播（有条件和无条件），吞吐量减半。Guidance Distillation 方法（比如 Consistency Distillation、DPO for diffusion）试图把 CFG 蒸馏进单次前向的模型里，减少这个开销。

另一个方向是更精细的条件控制：ControlNet 在 UNet 里增加可训练的旁路，让用户用边缘图、深度图、人体姿态等控制生成，不需要重训整个模型。

这是生成模型章节的最后一节。下一章讲强化学习，从 MDP 基础出发，推导策略梯度和 Actor-Critic，这些方法在生成模型的 RLHF 微调（对齐）和扩散模型的奖励优化里会再次用到。
