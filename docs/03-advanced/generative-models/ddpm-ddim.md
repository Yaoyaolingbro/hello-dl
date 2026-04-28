# DDPM 与 DDIM

!!! info "参考资料"
    **主要论文**

    - [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al., NeurIPS 2020
    - [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al., ICLR 2021
    - [Improved DDPM](https://arxiv.org/abs/2102.09672) — Nichol & Dhariwal, ICML 2021

    **优质讲解**

    - [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — Lilian Weng
    - [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — HuggingFace

!!! note "前置依赖"
    本节用到 [随机过程（马尔可夫链）](../../01-math/probability/stochastic-processes.md) 的基本概念，以及高斯分布的性质。

## 直觉 (Intuition)

扩散模型（Diffusion Model）的出发点有点反常识：它的训练目标不是"生成图像"，而是"去除噪声"。前向过程把一张真实图像逐步加噪直到变成纯高斯噪声；反向过程训练一个网络，让它学会从带噪声的图像中"猜测并去除"那些噪声。输入是噪声图，输出是干净图像。核心洞察是：去噪比生成更容易监督，而多步去噪的组合最终等价于从分布中采样。

## 问题背景

GAN 生成质量高，但训练不稳定，模式崩塌是常见问题。VAE 训练稳定，但生成的图像模糊。扩散模型的目标是：既稳定又高质量。

关键思路来自非平衡热力学（non-equilibrium thermodynamics）：把数据分布 $p_{\text{data}}$ 到高斯分布 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 的过程看成一个马尔可夫链，然后学习这个链的逆过程。

## 前向过程

前向过程（forward process）是一个固定的（不需要学习的）马尔可夫链，逐步给数据加高斯噪声：

$$q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t}\, \mathbf{x}_{t-1},\; \beta_t \mathbf{I})$$

其中 $\beta_t \in (0, 1)$ 是噪声调度（noise schedule），控制每步加噪的幅度，一般选一个从小到大的序列（比如 $\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$）。

**关键推导**：定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$，可以得到一个重要的封闭形式——从 $\mathbf{x}_0$ 直接跳到任意时刻 $\mathbf{x}_t$：

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t;\; \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0,\; (1-\bar{\alpha}_t) \mathbf{I})$$

等价地，

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\, \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

这个式子非常重要：它说明训练时不需要一步步模拟前向过程，可以直接从 $\mathbf{x}_0$ 一步采样到任意噪声级别 $t$ 的 $\mathbf{x}_t$。

## 反向过程与训练目标

反向过程（reverse process）是我们要学习的部分：从纯噪声 $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 出发，逐步去噪还原 $\mathbf{x}_0$。

理论上，反向过程的每一步也是高斯的（当 $\beta_t$ 足够小时），用参数化网络 $\boldsymbol{\mu}_\theta, \boldsymbol{\Sigma}_\theta$ 来逼近：

$$p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1};\; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t),\; \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))$$

对应的变分下界（ELBO）经过推导（原论文 Section 3.3），最终简化成一个**去噪目标**：

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\varepsilon}} \left[ \| \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) \|^2 \right]$$

其中 $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\, \mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\, \boldsymbol{\varepsilon}$，$\boldsymbol{\varepsilon}_\theta$ 是网络预测的噪声。

!!! note "直觉小结"
    训练目标退化成了：给定带噪图像和时间步 $t$，让网络预测被加进去的噪声 $\boldsymbol{\varepsilon}$。这比直接预测干净图像 $\mathbf{x}_0$ 更稳定，因为噪声的量级不随 $t$ 变化。

## DDPM 采样

训练好 $\boldsymbol{\varepsilon}_\theta$ 后，采样按以下步骤逐步去噪（$T$ 步，一般 $T = 1000$）：

$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z}$$

其中 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$（$t > 1$ 时），$\sigma_t^2 = \beta_t$（或 $\sigma_t^2 = \tilde{\beta}_t$，原论文有两种选择，差别不大）。

DDPM 的问题是需要 1000 步才能生成一张图，每步都要过一次神经网络——生成速度太慢。

## DDIM：确定性快速采样

DDIM（Denoising Diffusion Implicit Models）的贡献是：在**不重新训练模型**的前提下，把采样步数从 1000 步压缩到 50 步甚至 10 步。

DDIM 的关键发现是：DDPM 的训练目标对应的不只有一个采样过程。可以构造一系列非马尔可夫的前向过程，它们共享同样的边缘分布 $q(\mathbf{x}_t|\mathbf{x}_0)$，但允许更快的反向采样。

DDIM 的采样更新公式：

$$\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{\bar{\alpha}_t}}}_{\text{预测的 } \hat{\mathbf{x}}_0} + \sqrt{1-\bar{\alpha}_{t-1} - \sigma_t^2}\, \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t, t) + \sigma_t \boldsymbol{\varepsilon}$$

当 $\sigma_t = 0$ 时，采样过程完全确定性：同样的初始噪声总是生成同样的图像，这使得图像编辑（在隐空间插值）成为可能。

!!! tip "工程重点"
    实际部署中常用 DDIM 采样器配合 50～100 步，效果接近 DDPM 1000 步但快 10-20 倍。调节 $\sigma_t$ 可以在"确定性（$\sigma_t=0$）"和"随机性（$\sigma_t=\sqrt{\beta_t}$，退化成 DDPM）"之间连续插值。

## 代码

DDPM 训练和 DDIM 采样的核心逻辑：

```python
import torch

def ddpm_train_step(model, x0, noise_schedule):
    """训练：给定干净图像，随机选时间步加噪，预测噪声"""
    T = len(noise_schedule['alphas_bar'])
    t = torch.randint(0, T, (x0.shape[0],))

    alpha_bar_t = noise_schedule['alphas_bar'][t].view(-1, 1, 1, 1)
    eps = torch.randn_like(x0)
    # 前向过程一步到位：x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * eps

    eps_pred = model(x_t, t)
    return ((eps - eps_pred) ** 2).mean()  # 去噪 MSE

@torch.no_grad()
def ddim_sample(model, noise_schedule, x_T, steps=50):
    """DDIM 确定性采样（sigma=0）"""
    alphas_bar = noise_schedule['alphas_bar']
    timesteps = torch.linspace(len(alphas_bar)-1, 0, steps+1).long()
    x = x_T
    for i in range(steps):
        t_cur  = timesteps[i].item()
        t_prev = timesteps[i+1].item()
        ab_cur  = alphas_bar[t_cur]
        ab_prev = alphas_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        eps = model(x, torch.tensor([t_cur] * x.shape[0]))
        # 先从 x_t 和 eps 估计 x0
        x0_hat = (x - (1 - ab_cur).sqrt() * eps) / ab_cur.sqrt()
        x0_hat = x0_hat.clamp(-1, 1)
        # 再用 x0_hat 重建 x_{t-1}（sigma=0，确定性）
        x = ab_prev.sqrt() * x0_hat + (1 - ab_prev).sqrt() * eps
    return x
```

## 局限与后续工作

DDPM 的采样速度即使经过 DDIM 加速，在实时应用（游戏、视频流）里仍然太慢。Consistency Model 尝试用单步或少步推理替代多步去噪；Flow Matching（下一节）用更直的 ODE 轨迹替代扩散的曲折路径，采样效率更高。

DDPM 的另一个问题是图像级操作计算量大。Latent Diffusion Model（LDM/Stable Diffusion）把扩散过程移到 VAE 的隐空间，把图像维度从 $512^2 \times 3$ 压缩到 $64^2 \times 4$，节省了约 48 倍计算量。

下一节讲 Flow Matching，它把扩散过程从随机微分方程（SDE）的视角切换到确定性 ODE，轨迹更直，训练更稳定。
