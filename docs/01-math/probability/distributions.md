# 常用分布

!!! info "参考资料"
    **主要资料**

    - [Deep Learning Book: Chapter 3.9](https://www.deeplearningbook.org/contents/prob.html) — Ian Goodfellow et al.
    - [Probability Distributions](https://en.wikipedia.org/wiki/List_of_probability_distributions) — Wikipedia 有各分布的完整公式表

    **工具文档**
    
    - [SciPy: `scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html)
    - [PyTorch: `torch.distributions`](https://pytorch.org/docs/stable/distributions.html)

## 直觉 (Intuition)

同一个"不确定性"可以有很多种形态：抛硬币是二值的，骰子是六值的，图像像素是连续的，点云坐标是有界的。不同形态对应不同的分布族。选对分布，模型才能合理描述数据；选错分布，再好的网络结构也会产生奇怪的输出（比如用高斯分布建模概率，会输出负数）。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\text{Bernoulli}(p)$ | 伯努利分布，参数为成功概率 $p$ |
| $\mathcal{N}(\mu, \sigma^2)$ | 高斯（正态）分布 |
| $\text{Cat}(\mathbf{p})$ | 类别分布，$K$ 个类别的概率向量 $\mathbf{p}$ |
| $\text{Beta}(\alpha, \beta)$ | Beta 分布 |
| $\text{Dir}(\boldsymbol{\alpha})$ | Dirichlet 分布 |

## 伯努利分布

伯努利分布 (Bernoulli Distribution) 描述"二选一"的随机变量，$X \in \{0, 1\}$：

$$
P(X = 1) = p, \quad P(X = 0) = 1 - p
$$

PMF 可以统一写成：

$$
p(x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

深度学习里，二分类的最后一层用 Sigmoid 激活，输出就是 $P(Y=1 \mid \mathbf{x})$，对应伯努利分布的参数 $p$。二元交叉熵损失就是伯努利分布的负对数似然。

## 类别分布

类别分布 (Categorical Distribution) 是伯努利分布的推广，$X \in \{1, 2, \ldots, K\}$：

$$
P(X = k) = p_k, \quad \sum_{k=1}^K p_k = 1
$$

多分类的最后一层用 Softmax 激活，输出就是 $K$ 个类别的概率向量，即类别分布的参数 $\mathbf{p} = (p_1, \ldots, p_K)$。语言模型每步输出的 token 概率向量（词表大小可达 32K~128K）本质上是一个类别分布。

## 高斯分布

高斯分布（正态分布，Normal Distribution）是连续变量里最重要的分布：

$$
p(x)
=
\frac{1}{\sqrt{2\pi}\sigma}
\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

其中 $\mu$ 是均值（分布的中心），$\sigma^2$ 是方差（分布的宽度）。记作 $X \sim \mathcal{N}(\mu, \sigma^2)$。

高斯分布无处不在的原因是**中心极限定理**：大量独立随机变量之和近似服从高斯分布，无论每个变量是什么分布。这使得高斯分布成为建模"许多小随机因素叠加"的默认选择。

多维高斯分布（Multivariate Gaussian）扩展到向量 $\mathbf{x} \in \mathbb{R}^d$：

$$
p(\mathbf{x})
=
\frac{1}{(2\pi)^{d/2}|\mathbf{\Sigma}|^{1/2}}
\exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)
$$

其中 $\boldsymbol{\mu}$ 是均值向量，$\mathbf{\Sigma}$ 是协方差矩阵（正定）。VAE 的先验就是标准多维高斯 $\mathcal{N}(\mathbf{0}, \mathbf{I})$。

## Beta 分布

Beta 分布是定义在 $[0,1]$ 上的连续分布，常用于建模概率本身：

$$
p(x)
=
\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0,1]
$$

其中 $B(\alpha, \beta)$ 是归一化常数（Beta 函数），$\alpha > 0$ 和 $\beta > 0$ 是形状参数。

- $\alpha = \beta = 1$：退化为均匀分布
- $\alpha, \beta > 1$：分布集中在 $(0,1)$ 中间某处
- $\alpha, \beta < 1$：分布集中在两端（0 或 1 附近）

Beta 分布是伯努利/二项分布的**共轭先验**——如果先验是 Beta，后验也是 Beta，参数可以解析更新，不需要积分。

## Dirichlet 分布

Dirichlet 分布是 Beta 分布在多维的推广，定义在概率单纯形 $\{\mathbf{p}: p_k \ge 0, \sum_k p_k = 1\}$ 上：

$$
p(\mathbf{p})
=
\frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K p_k^{\alpha_k - 1}
$$

参数 $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$ 控制分布的集中程度：$\alpha_k$ 越大，对应类别的概率越被强调；所有 $\alpha_k \to \infty$ 时退化为集中在均匀点上的分布。

Dirichlet 分布是类别分布的共轭先验，在主题模型（LDA）和 few-shot 分类中有应用。

!!! note "共轭先验的意义"
    共轭先验让后验计算变成更新参数，而不是做积分。这是传统贝叶斯统计的优雅之处，也是 VAE 选用高斯先验的原因之一——高斯先验在高斯似然下有解析后验。

## 代码验证

```python
import numpy as np
from scipy import stats
import torch
from torch.distributions import Normal, Categorical, Beta, Dirichlet

# 高斯分布：PDF 验证
mu, sigma = 0.0, 1.0
x = np.linspace(-4, 4, 1000)
pdf_scipy = stats.norm.pdf(x, mu, sigma)
print(np.trapz(pdf_scipy, x))  # ≈ 1.0，验证归一化

# 多维高斯采样（PyTorch）
mu_vec = torch.zeros(3)
cov = torch.eye(3)
dist = torch.distributions.MultivariateNormal(mu_vec, cov)
samples = dist.sample((1000,))
print(samples.mean(dim=0))   # ≈ [0, 0, 0]
print(samples.var(dim=0))    # ≈ [1, 1, 1]

# 类别分布：采样 token
probs = torch.tensor([0.1, 0.3, 0.5, 0.1])  # 4 个 token 的概率
cat = Categorical(probs=probs)
tokens = cat.sample((10,))
print(tokens)  # 大多数是 2（概率 0.5 的那个 token）

# Beta 分布：建模伯努利先验
alpha, beta_param = 2.0, 5.0
beta_dist = Beta(alpha, beta_param)
samples_beta = beta_dist.sample((10000,))
print(f"Beta(2,5) 均值: {samples_beta.mean():.3f}")  # ≈ 0.286 = 2/(2+5)
```

## 在深度学习中的应用

高斯分布是扩散模型（DDPM）加噪过程的核心，也是 VAE 隐变量的先验和近似后验。类别分布是所有分类任务和语言模型 next-token 预测的输出形式。Dirichlet 分布在 LDA 主题模型和 few-shot 元学习中用作先验。Beta 分布则在对比学习的 Mixup 数据增强中控制混合比例。

下一节讲期望与方差。它们是分布的"统计摘要"，让我们不用看整个分布就能理解它的主要特征。
