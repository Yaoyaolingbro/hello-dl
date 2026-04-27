# 变分自编码器 (VAE)

!!! info "参考资料"
    **主要论文**

    - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, ICLR 2014
    - [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) — Kingma & Welling, 2019（综述版）

    **优质讲解**

    - [Lilian Weng: From Autoencoder to Beta-VAE](https://lilianweng.github.io/posts/2018-08-12-vae/)
    - [李宏毅 VAE 讲座](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)

!!! note "前置依赖"
    本节用到 [变分推断与 ELBO](../../01-math/probability/variational-inference.md) 里的 ELBO 推导和重参数化技巧。如果还没读过，建议先看那一节。

## 直觉 (Intuition)

自编码器可以压缩数据，但没法生成新样本——隐空间里有大量空洞，随机采一个点解码出来是噪声。VAE 的贡献是把隐空间变成一个有结构的概率分布：编码器不再输出一个确定的 $\mathbf{z}$，而是输出一个高斯分布的参数 $(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$，再从这个分布里采样。输入是数据 $\mathbf{x}$，输出是重建的 $\hat{\mathbf{x}}$ 和隐分布参数。整个框架是"用变分推断来近似贝叶斯推断"，但实现起来和自编码器一样用反向传播训练。

## 问题背景

从概率的角度看，生成模型的任务是学习 $p(\mathbf{x})$。一个自然的做法是引入隐变量 $\mathbf{z}$，把 $p(\mathbf{x})$ 写成边缘化形式：

$$p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) \, d\mathbf{z}$$

其中 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ 是先验，$p_\theta(\mathbf{x}|\mathbf{z})$ 是解码器。

问题在于这个积分对高维 $\mathbf{z}$ 不可处理（intractable）。贝叶斯推断需要计算后验 $p(\mathbf{z}|\mathbf{x})$，这同样不可处理。VAE 用变分推断，引入近似后验 $q_\phi(\mathbf{z}|\mathbf{x})$（即编码器），来绕开这个问题。

## 方法推导

!!! info "符号约定"
    沿用原论文 [Kingma & Welling, 2014]：
    $\phi$ 是 encoder 参数，$\theta$ 是 decoder 参数，$\mathbf{z}$ 是隐变量。

**第一步**：从最大化 $\log p_\theta(\mathbf{x})$ 出发，引入近似后验 $q_\phi(\mathbf{z}|\mathbf{x})$，通过 Jensen 不等式得到变分下界 ELBO（Evidence Lower BOund）：

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

这个不等式的推导已在 [变分推断章节](../../01-math/probability/variational-inference.md) 中详细展开。

**第二步**：解读两项含义。

第一项 $\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z})]$ 是**重建项**：从编码后的分布里采样 $\mathbf{z}$，再解码回 $\mathbf{x}$，衡量重建质量。

第二项 $\text{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$ 是**正则项**：迫使编码分布接近标准高斯，防止 $\mathbf{z}$ 随意分散在隐空间的任意角落。

**第三步**：计算 KL 散度的解析形式。当 $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$，$p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ 时，KL 有闭合解：

$$\text{KL}(q \| p) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

这使得正则项不需要采样，可以直接计算。

**第四步**：重参数化技巧（Reparameterization Trick）。重建项里的采样 $\mathbf{z} \sim q_\phi(\mathbf{z}|\mathbf{x})$ 不可微，梯度没法通过采样操作流回 $\phi$。解决方法是把随机性移出计算图：

$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

现在 $\boldsymbol{\varepsilon}$ 是外部噪声，不参与反向传播，梯度可以顺利流过 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$。

!!! note "直觉小结"
    重参数化把"采样不可微"转化为"给确定计算加一个外部噪声"，这是深度学习里处理随机性的通用技巧，在 Diffusion Model、Flow Matching 里都会再次出现。

## 关键设计决策

**为什么先验选 $\mathcal{N}(\mathbf{0}, \mathbf{I})$？**

标准高斯先验有两个好处：KL 有解析解（不需要采样估计）；它是最"无信息"的先验，不引入不必要的偏置。实践中也有用 VQ-VAE 的离散先验或更复杂的流模型先验，但代价是 KL 计算复杂。

**$\beta$-VAE 为什么加一个系数 $\beta > 1$？**

原始 VAE 的 KL 项权重设为 1，有时编码器学到的 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 变化很大，隐空间不够"结构化"。$\beta$-VAE 把 KL 项乘以 $\beta > 1$，强迫隐空间更接近标准高斯，每个维度倾向于编码独立的因子。代价是重建质量下降。

## 代码

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.fc_mu    = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.encoder_base = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, input_dim), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_base(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)          # 随机性在 eps，不在 mu/std
        return mu + std * eps                # 梯度流过 mu 和 std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def elbo_loss(x, x_recon, mu, logvar):
    # 重建项：像素级 BCE，summed 而非 mean（和原论文一致）
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL 项：解析式，不需要采样
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

vae = VAE()
x = torch.rand(32, 784)
x_recon, mu, logvar = vae(x)
loss = elbo_loss(x, x_recon, mu, logvar)
print(f"ELBO loss: {loss.item():.1f}")  # 数量级约 ~20000（未训练）
```

## 局限与后续工作

VAE 生成的图像通常比较模糊。根本原因是重建项用了像素级损失，而人眼的感知是非线性的——一张图整体偏亮 1 像素，和一张图丢失了边缘轮廓，在 MSE 意义上可能相同，但后者明显更差。

VQ-VAE 用离散隐空间解决了隐空间"塌缩"问题（后验 $q$ 直接等于先验，导致 $\mathbf{z}$ 不携带信息）。Latent Diffusion Model（LDM，也就是 Stable Diffusion 的基础）用一个 VAE 把图像压到低维隐空间，再在隐空间里跑扩散模型，兼顾了 VAE 的压缩效率和扩散模型的生成质量。

下一节讲 DDPM，它不需要对抗训练，也不需要变分推断，用一个简单的去噪目标就能训练出高质量的生成模型。
