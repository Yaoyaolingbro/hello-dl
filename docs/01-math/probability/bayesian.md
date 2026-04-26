# 贝叶斯推断

!!! info "参考资料"
    **主要资料**
    - [Deep Learning Book: Chapter 5.6](https://www.deeplearningbook.org/contents/ml.html) — Ian Goodfellow et al.
    - [Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf) — Bishop, Chapter 1–2（经典教材）
    - [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers) — 代码驱动的入门

## 直觉 (Intuition)

贝叶斯推断是一套"用数据更新信念"的框架。你先有一个对参数的初始猜测（先验），然后看到数据，通过贝叶斯公式计算更新后的信念（后验）。和频率派的"参数是固定值，用数据估计它"不同，贝叶斯派把参数本身看成随机变量，维护一个完整的分布，而不只是一个点估计。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\theta$ | 模型参数（被视为随机变量） |
| $\mathcal{D}$ | 观测数据集 |
| $p(\theta)$ | 先验分布 |
| $p(\mathcal{D} \mid \theta)$ | 似然函数 |
| $p(\theta \mid \mathcal{D})$ | 后验分布 |

## 贝叶斯定理用于推断

把贝叶斯公式用在参数 $\theta$ 和数据 $\mathcal{D}$ 上：

$$
p(\theta \mid \mathcal{D})
=
\frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}
$$

因为 $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta)\, d\theta$ 是常数（不依赖 $\theta$），所以在优化时常省略，写成正比关系：

$$
\underbrace{p(\theta \mid \mathcal{D})}_{\text{后验}}
\propto
\underbrace{p(\mathcal{D} \mid \theta)}_{\text{似然}}
\cdot
\underbrace{p(\theta)}_{\text{先验}}
$$

## 最大似然估计（MLE）

**最大似然估计 (MLE)** 忽略先验，只最大化数据在参数 $\theta$ 下的似然：

$$
\hat{\theta}_\text{MLE}
=
\arg\max_\theta\, p(\mathcal{D} \mid \theta)
$$

假设数据独立同分布（i.i.d.），取对数让连乘变成求和：

$$
\hat{\theta}_\text{MLE}
=
\arg\max_\theta \sum_{i=1}^N \log p(x_i \mid \theta)
$$

神经网络训练中，最小化交叉熵损失等价于对类别分布做 MLE，最小化 MSE 损失等价于对高斯分布做 MLE。

## 最大后验估计（MAP）

**最大后验估计 (MAP)** 加入先验，最大化后验：

$$
\hat{\theta}_\text{MAP}
=
\arg\max_\theta\, p(\theta \mid \mathcal{D})
=
\arg\max_\theta \left[\log p(\mathcal{D} \mid \theta) + \log p(\theta)\right]
$$

!!! note "MLE vs MAP vs 完整贝叶斯"
    - **MLE**：只用数据，没有正则化，容易过拟合
    - **MAP**：加入先验，等价于正则化——高斯先验 $p(\theta) \propto e^{-\lambda\|\theta\|^2}$ 对应 L2 正则化，拉普拉斯先验对应 L1 正则化
    - **完整贝叶斯**：维护后验分布，不只是一个点，能定量表示不确定性，但计算代价高

## 共轭先验

当先验和似然属于特定的分布族时，后验和先验有相同的函数形式，只是参数不同。这种组合叫**共轭先验**，后验可以解析计算，不需要积分或采样。

常见共轭对：

| 似然 | 共轭先验 | 后验 |
|------|---------|------|
| 伯努利 $\text{Bernoulli}(p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha+\text{成功次数}, \beta+\text{失败次数})$ |
| 高斯（均值未知，方差已知） | 高斯 | 高斯 |
| 多项式（类别分布） | Dirichlet | Dirichlet |

## 贝叶斯更新是增量的

贝叶斯推断的一个重要特性是：先用 $n$ 个样本更新得到的后验，可以作为下一批数据的先验，继续更新。数据越多，后验越集中，不确定性越小，最终与 MLE 结果收敛。

## 代码验证

```python
import numpy as np
from scipy.stats import beta

# 抛硬币：估计正面概率 p，使用 Beta 共轭先验
# 先验：Beta(2, 2)，偏向 p=0.5
alpha_prior, beta_prior = 2.0, 2.0

# 观测数据
n_heads, n_tails = 7, 3  # 10 次中 7 次正面

# 后验更新：Beta(alpha + heads, beta + tails)
alpha_post = alpha_prior + n_heads   # 9.0
beta_post  = beta_prior  + n_tails   # 5.0

# 后验均值
posterior_mean = alpha_post / (alpha_post + beta_post)
mle_estimate   = n_heads / (n_heads + n_tails)
print(f"后验均值: {posterior_mean:.3f}")  # 0.643（先验拉向 0.5）
print(f"MLE 估计: {mle_estimate:.3f}")   # 0.700（纯数据估计）

# MAP 估计 = (alpha - 1) / (alpha + beta - 2)
map_estimate = (alpha_post - 1) / (alpha_post + beta_post - 2)
print(f"MAP 估计: {map_estimate:.3f}")   # 0.667

# 更多数据后，后验向 MLE 靠近
for n_total in [10, 100, 1000]:
    h = int(n_total * 0.7)  # 固定 70% 正面
    t = n_total - h
    a_post = alpha_prior + h
    b_post = beta_prior  + t
    post_mean = a_post / (a_post + b_post)
    print(f"n={n_total:4d}: 后验均值={post_mean:.4f} (MLE=0.700)")
# n=  10: 后验均值=0.6429 <- 先验影响大
# n= 100: 后验均值=0.6981 <- 先验影响小了
# n=1000: 后验均值=0.6999 <- 几乎等于 MLE
```

## 在深度学习中的应用

L2 正则化（Weight Decay）的贝叶斯解释是：给权重加高斯先验，做 MAP 估计。Dropout 可以解释为近似贝叶斯推断（MC Dropout：推理时开 Dropout，多次前向传播的方差估计不确定性）。VAE 的训练目标 ELBO 是贝叶斯推断的变分近似——这是后面"变分推断"节的主题。

下一节讲信息论。熵、KL 散度、互信息——这些是衡量分布之间"差距"的工具，也是生成模型损失函数的语言。
