# 生成对抗网络 (GAN)

!!! info "参考资料"
    **主要论文**

    - [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Goodfellow et al., NeurIPS 2014
    - [Conditional GAN](https://arxiv.org/abs/1411.1784) — Mirza & Osindero, 2014
    - [Wasserstein GAN](https://arxiv.org/abs/1701.07875) — Arjovsky et al., ICML 2017

    **优质讲解**

    - [李宏毅 GAN 讲座 2021](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)
    - [GAN Zoo](https://github.com/hindupuravinash/the-gan-zoo)

## 直觉 (Intuition)

在 GAN 出现之前，生成模型（比如 VAE 的前身）往往只能产出模糊的图像，根本原因是它们用像素级 MSE 来衡量"像不像"，而这个标准对高频细节不敏感。GAN 的思路完全不同：不定义一个固定的损失函数，而是训练一个判别器来当"鉴赏家"，生成器只需要骗过这个鉴赏家就够了。输入是随机噪声向量 $\mathbf{z}$，输出是和真实数据难以区分的样本。核心机制是两个网络的对抗博弈，判别器越强，生成器就被逼着生成越真实的样本。

## 问题背景

生成模型的目标是学会数据分布 $p_{\text{data}}(\mathbf{x})$，然后从中采样。直接建模高维分布（比如 $256 \times 256$ 图像）是不可能的——参数量太大，归一化常数也无法计算。

GAN 绕开了这个问题：不去显式建模 $p_{\text{data}}(\mathbf{x})$，而是学一个变换 $G: \mathbf{z} \mapsto \mathbf{x}$，让 $G$ 的输出分布尽量接近真实分布。如何衡量"接近"？用一个判别器 $D: \mathbf{x} \mapsto [0,1]$ 来估计。

## 方法推导

**第一步**：定义判别器目标。$D(\mathbf{x})$ 输出"这个样本是真实数据"的概率，我们希望它对真实样本输出 1，对生成样本输出 0：

$$\max_D \; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z} [\log(1 - D(G(\mathbf{z})))]$$

**第二步**：定义生成器目标。$G$ 的任务是让 $D$ 把生成样本判断成真实的——即最小化上面判别器的目标：

$$\min_G \; \mathbb{E}_{\mathbf{z} \sim p_z} [\log(1 - D(G(\mathbf{z})))]$$

合在一起，GAN 的训练目标是一个极小极大问题：

$$\min_G \max_D \; V(G, D) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} [\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z} [\log(1 - D(G(\mathbf{z})))]$$

**第三步**：分析最优判别器。对任意固定的 $G$，最优的 $D^*(\mathbf{x})$ 是：

$$D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_G(\mathbf{x})}$$

其中 $p_G(\mathbf{x})$ 是生成器诱导的分布。

**第四步**：代入最优判别器，整理生成器的目标。把 $D^*$ 代回 $V(G, D^*)$，经过整理可以得到：

$$V(G, D^*) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_G)$$

其中 $\text{JSD}$ 是 Jensen-Shannon 散度（取值 $[0, \log 2]$）。所以最优的 $G$ 使得 $p_G = p_{\text{data}}$，此时 JSD 为 0。

!!! note "直觉小结"
    GAN 本质上是在最小化生成分布和真实分布的 JS 散度，但通过对抗博弈来估计这个散度，而不是直接计算。

## 关键设计决策

**为什么不直接用 $\min_G -\mathbb{E}[\log D(G(\mathbf{z}))]$ 而不是 $\min_G \mathbb{E}[\log(1-D(G(\mathbf{z})))]$？**

训练初期 $D$ 很容易区分真假（$D(G(\mathbf{z})) \approx 0$），此时 $\log(1-D(G(\mathbf{z}))) \approx 0$，梯度几乎为零，生成器学不动。改成 $-\log D(G(\mathbf{z}))$ 后，初期梯度更大，训练更稳定。这是原论文里的一个实用 trick，消融上体现为训练早期收敛速度差异显著。

**为什么 WGAN 要替换 JS 散度？**

当 $p_{\text{data}}$ 和 $p_G$ 支撑集不重叠时（高维数据里很常见），JS 散度是常数 $\log 2$，梯度消失。Wasserstein 距离（Earth Mover's Distance）在支撑集不重叠时仍然有意义，训练更稳定。WGAN 的判别器（改名叫 critic）不再输出概率，去掉 sigmoid，并用权重裁剪或梯度惩罚保证 1-Lipschitz 约束。

## 代码

GAN 训练的核心循环：判别器和生成器交替更新。

```python
import torch
import torch.nn as nn

# 每一步训练判别器（多步）再训练生成器（一步）
def train_step(G, D, real_x, z, opt_G, opt_D):
    # 训练判别器：最大化 log D(x_real) + log(1 - D(G(z)))
    fake_x = G(z).detach()  # detach 防止梯度流入 G
    loss_D = -torch.mean(torch.log(D(real_x) + 1e-8) +
                         torch.log(1 - D(fake_x) + 1e-8))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()

    # 训练生成器：最小化 -log D(G(z))，即非饱和版本
    fake_x = G(z)
    loss_G = -torch.mean(torch.log(D(fake_x) + 1e-8))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()

    return loss_D.item(), loss_G.item()
```

## 局限与后续工作

GAN 的训练不稳定是公认的难题：模式崩塌（mode collapse）让生成器只学会生成少数几种样本；判别器过强会导致梯度消失，过弱则生成器无法被有效约束。超参数调节像魔法——改一个学习率，整个训练行为就变了。

WGAN 和 WGAN-GP 缓解了训练稳定性问题，StyleGAN 系列在图像质量上做到了极致，但根本上 GAN 仍然很难扩展到多样性要求高的任务。扩散模型的出现很大程度上是因为它规避了对抗训练的不稳定性，用确定性的去噪目标替代了博弈。

下一节讲自编码器（Autoencoder），它是 VAE 和扩散模型中隐空间压缩的基础组件。
