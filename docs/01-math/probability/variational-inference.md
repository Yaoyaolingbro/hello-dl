# 变分推断与 ELBO

!!! info "参考资料"
    **主要资料**
    - [Deep Learning Book: Chapter 19](https://www.deeplearningbook.org/contents/inference.html) — Ian Goodfellow et al.
    - Blei et al., "Variational Inference: A Review for Statisticians", *JASA* 2017 — 变分推断综述
    - Kingma & Welling, "Auto-Encoding Variational Bayes", ICLR 2014 — VAE 原始论文

    **工具文档**
    - [PyTorch: `torch.distributions`](https://pytorch.org/docs/stable/distributions.html)
    - [Pyro](https://pyro.ai/) — 概率编程库，变分推断的工程实现

## 直觉 (Intuition)

贝叶斯推断的核心困难是计算后验 $p(\mathbf{z} \mid \mathbf{x})$——分母 $p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$ 是一个高维积分，通常无法解析计算。变分推断的解法是：用一个参数化的简单分布 $q_\phi(\mathbf{z})$ 近似后验，把积分问题转化为优化问题——找最好的 $\phi$ 使 $q_\phi$ 尽量接近 $p(\mathbf{z} \mid \mathbf{x})$。VAE 就是这套思想的深度学习实现。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbf{x}$ | 观测变量（数据） |
| $\mathbf{z}$ | 隐变量（潜变量） |
| $p_\theta(\mathbf{x}, \mathbf{z})$ | 联合分布（生成模型） |
| $p(\mathbf{z} \mid \mathbf{x})$ | 真实后验（难以计算） |
| $q_\phi(\mathbf{z} \mid \mathbf{x})$ | 变分后验（近似分布，可微） |
| $\mathcal{L}(\theta, \phi; \mathbf{x})$ | ELBO（证据下界） |

## ELBO 的推导

目标是最大化对数似然 $\log p_\theta(\mathbf{x})$，对其做恒等变换：

$$
\log p_\theta(\mathbf{x})
=
\log \int p_\theta(\mathbf{x}, \mathbf{z})\, d\mathbf{z}
=
\log \int q_\phi(\mathbf{z} \mid \mathbf{x})\, \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\, d\mathbf{z}
$$

利用 Jensen 不等式（$\log$ 是凹函数，$\log \mathbb{E}[Y] \ge \mathbb{E}[\log Y]$）：

$$
\log p_\theta(\mathbf{x})
\ge
\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}\!\left[\log \frac{p_\theta(\mathbf{x}, \mathbf{z})}{q_\phi(\mathbf{z} \mid \mathbf{x})}\right]
=:
\mathcal{L}(\theta, \phi; \mathbf{x})
$$

右侧就是 **ELBO（Evidence Lower BOund，证据下界）**。等号成立当且仅当 $q_\phi(\mathbf{z} \mid \mathbf{x}) = p_\theta(\mathbf{z} \mid \mathbf{x})$。

展开 ELBO，得到两项解释：

$$
\mathcal{L}(\theta, \phi; \mathbf{x})
=
\underbrace{\mathbb{E}_{q_\phi}\!\left[\log p_\theta(\mathbf{x} \mid \mathbf{z})\right]}_{\text{重建项（Reconstruction）}}
-
\underbrace{D_\text{KL}\!\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p(\mathbf{z})\right)}_{\text{正则项（KL Divergence）}}
$$

重建项要求隐变量能还原观测，KL 项要求近似后验接近先验。

!!! note "ELBO 与对数似然的关系"
    $$
    \log p_\theta(\mathbf{x})
    =
    \mathcal{L}(\theta, \phi; \mathbf{x})
    +
    D_\text{KL}\!\left(q_\phi(\mathbf{z} \mid \mathbf{x}) \,\|\, p_\theta(\mathbf{z} \mid \mathbf{x})\right)
    $$
    因为 KL 散度 $\ge 0$，ELBO 始终是对数似然的下界。最大化 ELBO 同时实现两件事：提升 $\log p_\theta(\mathbf{x})$，并缩小 $q_\phi$ 与真实后验的 KL 差距。

## 重参数化技巧

ELBO 的梯度包含对 $q_\phi$ 期望的导数 $\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]$，不能直接通过采样估计（采样操作不可微）。**重参数化（Reparameterization Trick）** 将随机性从参数中分离：

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

梯度现在可以流过 $\boldsymbol{\mu}_\phi$ 和 $\boldsymbol{\sigma}_\phi$（确定性函数），随机性全部在 $\boldsymbol{\epsilon}$ 上（不含参数）：

$$
\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z})}[f(\mathbf{z})]
=
\mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I})}\!\left[\nabla_\phi f\!\left(\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}\right)\right]
$$

右侧可以用 MC 估计（通常只需 1 个样本），且完全可微。这是 VAE 能用反向传播训练的关键。

## 均场近似

当隐变量维度高时，常假设 $q_\phi$ 可以因子分解（**均场假设**）：

$$
q_\phi(\mathbf{z}) = \prod_{j=1}^d q_j(z_j)
$$

各维度独立，大大降低了参数量和优化难度，代价是忽略了后验中的相关性。对每个因子 $q_j$ 的最优解有解析形式（坐标上升变分推断，CAVI），不需要梯度方法。

## 代码验证

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# 最小 VAE：演示 ELBO 计算和重参数化
class VAE(nn.Module):
    def __init__(self, input_dim=2, latent_dim=1):
        super().__init__()
        # 编码器：输出均值和对数方差
        self.enc_mu    = nn.Linear(input_dim, latent_dim)
        self.enc_logvar = nn.Linear(input_dim, latent_dim)
        # 解码器：重建输入
        self.dec = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        return self.enc_mu(x), self.enc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        return mu + std * eps           # z = μ + σ * ε

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def elbo_loss(x, x_recon, mu, logvar):
    # 重建项：MSE（对应高斯解码器）
    recon = nn.functional.mse_loss(x_recon, x, reduction='sum')
    # KL 项：KL(N(μ,σ²) || N(0,1)) 的解析形式
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + kl

# 演示前向传播
vae = VAE()
x = torch.randn(16, 2)          # 16 个样本，2 维输入
x_recon, mu, logvar = vae(x)
loss = elbo_loss(x, x_recon, mu, logvar)
print(f"ELBO loss: {loss.item():.2f}")

# KL 散度解析公式验证
# KL(N(mu, sigma^2) || N(0,1)) = 0.5*(mu^2 + sigma^2 - 1 - log sigma^2)
mu_val = torch.tensor([1.0])
logvar_val = torch.tensor([0.0])   # sigma=1
kl_analytic = -0.5 * (1 + logvar_val - mu_val**2 - logvar_val.exp())
print(f"KL(N(1,1)||N(0,1)) = {kl_analytic.item():.4f}")  # 应为 0.5

# 验证：KL(N(0,1)||N(0,1)) = 0
mu_zero = torch.tensor([0.0])
logvar_zero = torch.tensor([0.0])
kl_zero = -0.5 * (1 + logvar_zero - mu_zero**2 - logvar_zero.exp())
print(f"KL(N(0,1)||N(0,1)) = {kl_zero.item():.4f}")  # 应为 0.0
```

## 在深度学习中的应用

VAE（Variational Autoencoder）是变分推断的直接实现：编码器参数化 $q_\phi(\mathbf{z} \mid \mathbf{x})$，解码器参数化 $p_\theta(\mathbf{x} \mid \mathbf{z})$，整体优化 ELBO。扩散模型可以被解释为一类分层 VAE，每个去噪步骤对应一个近似后验。大语言模型的 RLHF 训练中，KL 约束项（防止模型偏离 SFT 策略太远）在数学上等价于对 ELBO 的 KL 正则项。这是概率深度学习中最核心的优化框架。
