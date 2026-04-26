# 蒙特卡洛与重要性采样

!!! info "参考资料"
    **主要资料**
    - [Deep Learning Book: Chapter 17](https://www.deeplearningbook.org/contents/monte_carlo.html) — Ian Goodfellow et al.
    - Robert & Casella, *Monte Carlo Statistical Methods* — 经典教材，前三章覆盖本节内容
    - [Arjovsky et al., "Wasserstein GAN"](https://arxiv.org/abs/1701.07875) — 重要性采样的实际应用

    **工具文档**
    - [NumPy: `numpy.random`](https://numpy.org/doc/stable/reference/random/index.html)
    - [PyTorch: `torch.distributions`](https://pytorch.org/docs/stable/distributions.html)

## 直觉 (Intuition)

当贝叶斯后验或期望积分无法解析计算时，蒙特卡洛方法用随机采样来近似：从分布中抽取样本，用样本均值代替积分。重要性采样解决的是"目标分布很难直接采样"的问题——换一个容易采样的提议分布，再用权重修正偏差。扩散模型的逆向采样、语言模型的 MCMC 解码、VAE 的 ELBO 梯度估计，都依赖这套思想。

## 主要符号

| 符号 | 含义 |
|------|------|
| $p(x)$ | 目标分布（难以采样） |
| $q(x)$ | 提议分布（容易采样） |
| $w(x)$ | 重要性权重 $p(x)/q(x)$ |
| $\hat{\mu}_N$ | 蒙特卡洛估计量 |
| $\text{ESS}$ | 有效样本数 |

## 蒙特卡洛估计

蒙特卡洛估计的核心思想是用样本均值近似期望：

$$
\mathbb{E}_p[f(X)]
=
\int f(x)\, p(x)\, dx
\approx
\hat{\mu}_N
=
\frac{1}{N} \sum_{i=1}^N f(x^{(i)}), \quad x^{(i)} \sim p
$$

大数定律保证 $\hat{\mu}_N \xrightarrow{p} \mathbb{E}_p[f(X)]$，中心极限定理给出误差量级：

$$
\hat{\mu}_N - \mathbb{E}_p[f] \sim \mathcal{N}\!\left(0,\, \frac{\text{Var}_p[f(X)]}{N}\right)
$$

误差以 $O(1/\sqrt{N})$ 收敛，与维度无关——这是蒙特卡洛在高维积分上的核心优势（数值积分的收敛速度依赖维度，而 MC 不）。

## 重要性采样

当目标分布 $p(x)$ 难以直接采样时，引入一个容易采样的**提议分布** $q(x)$，通过权重修正：

$$
\mathbb{E}_p[f(X)]
=
\int f(x)\, p(x)\, dx
=
\int f(x)\, \frac{p(x)}{q(x)}\, q(x)\, dx
=
\mathbb{E}_q\!\left[f(X)\, \frac{p(X)}{q(X)}\right]
$$

从 $q$ 采样，赋予每个样本**重要性权重** $w(x) = p(x)/q(x)$，估计量变为：

$$
\hat{\mu}_N^{\text{IS}}
=
\frac{1}{N} \sum_{i=1}^N f(x^{(i)})\, w(x^{(i)}), \quad x^{(i)} \sim q
$$

要求 $q(x) > 0$ 凡 $p(x) > 0$ 的地方（支撑覆盖）。

!!! note "提议分布的选择"
    理想的提议分布 $q(x) \propto |f(x)|\, p(x)$，使权重方差最小。实践中的原则：
    - $q$ 的尾部要比 $p$ 更重（否则权重方差无穷大）
    - $q$ 要容易采样，且能计算密度值
    - 高维时用神经网络拟合 $q$（归一化流、变分推断）

## 自归一化重要性采样

实践中 $p(x)$ 常只知道到常数倍（如未归一化的后验），用自归一化版本：

$$
\hat{\mu}_N^{\text{SIS}}
=
\frac{\sum_{i=1}^N f(x^{(i)})\, \tilde{w}(x^{(i)})}{\sum_{i=1}^N \tilde{w}(x^{(i)})},
\quad
\tilde{w}(x) = \frac{\tilde{p}(x)}{q(x)}
$$

其中 $\tilde{p}$ 是未归一化的目标。这是序列蒙特卡洛（SMC）和粒子滤波的基础。

## 有效样本数（ESS）

权重方差越大，等效的独立样本数越少，用**有效样本数**衡量：

$$
\text{ESS}
=
\frac{\left(\sum_{i=1}^N w_i\right)^2}{\sum_{i=1}^N w_i^2}
\in [1,\, N]
$$

$\text{ESS} \approx N$ 表示 $q \approx p$，采样效率高；$\text{ESS} \ll N$ 说明提议分布太差，需要换 $q$。

## 方差缩减技术

标准 MC 的方差 $\text{Var}(\hat{\mu}_N) = \text{Var}_p[f(X)]/N$，可以从分子 $\text{Var}_p[f(X)]$ 入手缩减：

**控制变量（Control Variates）**：找到一个期望已知的 $g(X)$（$\mathbb{E}_p[g] = \mu_g$），用：

$$
\hat{\mu}_N^{\text{CV}} = \frac{1}{N}\sum_{i=1}^N \left[f(x^{(i)}) - c\left(g(x^{(i)}) - \mu_g\right)\right]
$$

最优系数 $c^* = \text{Cov}(f, g)/\text{Var}(g)$，方差缩减量为 $\rho^2(f,g)\text{Var}(f)$，$\rho$ 越接近 1 效果越好。

**对偶变量（Antithetic Variables）**：对每个 $u \sim \text{Uniform}(0,1)$，同时用 $u$ 和 $1-u$ 生成两个负相关样本，均值不变但方差减小。

**拟蒙特卡洛（QMC）**：用低偏差序列（Sobol、Halton）替代随机序列，收敛速度提升到 $O(1/N)$（代替 $O(1/\sqrt{N})$），适用于光滑被积函数。

## 代码验证

```python
import numpy as np

np.random.seed(42)
N = 10000

# 目标：估计 E[X^2] under p = N(0,1)，真值 = 1.0
p_samples = np.random.randn(N)
mc_estimate = (p_samples ** 2).mean()
print(f"标准 MC 估计: {mc_estimate:.4f}  (真值: 1.0)")

# 重要性采样：用 q = N(0, 2^2) 估计同一期望
# w(x) = p(x)/q(x) = N(x;0,1) / N(x;0,4)
q_samples = np.random.normal(0, 2, N)  # 从 q 采样

def log_normal(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma)

log_w = log_normal(q_samples, 0, 1) - log_normal(q_samples, 0, 2)
w = np.exp(log_w - log_w.max())  # 数值稳定

f_vals = q_samples ** 2
is_estimate = (f_vals * w).sum() / w.sum()
print(f"重要性采样估计: {is_estimate:.4f}  (真值: 1.0)")

# 有效样本数
ess = w.sum() ** 2 / (w ** 2).sum()
print(f"ESS: {ess:.0f} / {N}  ({100*ess/N:.1f}%)")

# 控制变量：用 g(x)=x 作控制变量（E_p[x]=0 已知）
# f(x)=x^2, g(x)=x, Cov(f,g)=E[x^3]=0 -> 对 N(0,1) 效果为零
# 换成估计 E[exp(x)]，真值 = e^{0.5} ≈ 1.6487
# 用 g(x)=x 作控制变量
f2 = np.exp(p_samples)
g = p_samples  # E[g] = 0
c_star = np.cov(f2, g)[0, 1] / np.var(g)
cv_estimate = (f2 - c_star * g).mean()
print(f"\n控制变量估计 E[exp(X)]: {cv_estimate:.4f}  (真值: {np.exp(0.5):.4f})")
print(f"标准 MC 估计 E[exp(X)]: {f2.mean():.4f}")
print(f"标准差比 (CV/MC): {(f2 - c_star * g).std() / f2.std():.4f}")
```

## 在深度学习中的应用

扩散模型（DDPM/DDIM）的采样过程是从高斯先验出发，迭代运行逆向马尔科夫链，本质是用神经网络参数化的 MC 采样。语言模型推理阶段的 beam search、top-k/top-p 采样，都是从自回归分布中做 MC 采样的不同策略。VAE 的 ELBO 梯度中含有对后验 $q_\phi(z|x)$ 的期望，重参数化技巧使其可以用 MC 估计并反向传播——这是下一节变分推断的核心。

下一节讲变分推断与 ELBO，把积分问题转化为优化问题，是 VAE、扩散模型和贝叶斯深度学习的数学基础。
