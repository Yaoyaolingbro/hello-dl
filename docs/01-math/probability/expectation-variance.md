# 期望与方差

!!! info "参考资料"
    **主要资料**
    
    - [Deep Learning Book: Chapter 3.4–3.5](https://www.deeplearningbook.org/contents/prob.html) — Ian Goodfellow et al.
    - [Mathematics for Machine Learning](https://mml-book.github.io/) — Deisenroth et al., Chapter 6

## 直觉 (Intuition)

期望是分布的"重心"，方差是分布的"宽度"。有了这两个数，就可以在不看完整分布的情况下，大致了解一个随机变量的行为。损失函数的期望是训练的优化目标，梯度的方差决定了优化能否收敛，批归一化通过控制方差让网络训练更稳定——这些都依赖期望和方差的计算。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbb{E}[X]$ | 随机变量 $X$ 的期望 |
| $\text{Var}(X)$ | $X$ 的方差 |
| $\text{Cov}(X, Y)$ | $X$ 和 $Y$ 的协方差 |
| $\boldsymbol{\Sigma}$ | 协方差矩阵 |
| $\mathbb{E}_\theta[\cdot]$ | 在参数为 $\theta$ 的分布下计算期望 |

## 期望

随机变量 $X$ 的**期望（均值）**定义为所有取值按概率加权的平均：

离散情况：

$$
\mathbb{E}[X]
=
\sum_{x} x\, p(x)
$$

连续情况：

$$
\mathbb{E}[X]
=
\int x\, p(x)\, dx
$$

期望的关键性质是**线性**：

$$
\mathbb{E}[\alpha X + \beta Y]
=
\alpha\, \mathbb{E}[X] + \beta\, \mathbb{E}[Y]
$$

这条性质不要求 $X$ 和 $Y$ 独立，随时可用。

## 方差

**方差**度量随机变量围绕均值的分散程度：

$$
\text{Var}(X)
=
\mathbb{E}\!\left[(X - \mathbb{E}[X])^2\right]
$$

展开后得到常用计算公式：

$$
\text{Var}(X)
=
\mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

标准差 $\text{std}(X) = \sqrt{\text{Var}(X)}$ 和 $X$ 有相同的量纲，更直观。

!!! note "方差不是线性的"
    $\text{Var}(\alpha X) = \alpha^2 \text{Var}(X)$。两个独立随机变量之和的方差等于方差之和：$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$（仅当 $X, Y$ 独立时）。

## 协方差与协方差矩阵

**协方差**度量两个随机变量如何"一起变化"：

$$
\text{Cov}(X, Y)
=
\mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
=
\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

- 正协方差：$X$ 大时 $Y$ 也倾向于大
- 负协方差：$X$ 大时 $Y$ 倾向于小
- 协方差为 0：两个变量线性无关（但不一定独立）

对随机向量 $\mathbf{X} = (X_1, \ldots, X_d)$，**协方差矩阵**定义为：

$$
\boldsymbol{\Sigma}
=
\mathbb{E}\!\left[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top\right]
\in \mathbb{R}^{d \times d}
$$

其中第 $(i,j)$ 个元素是 $\text{Cov}(X_i, X_j)$。协方差矩阵一定是**对称正半定**的，前一章特殊矩阵节已证明。

## 相关系数

相关系数是归一化版本的协方差，去掉了量纲影响：

$$
\rho(X, Y)
=
\frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\, \text{Var}(Y)}}
\in [-1, 1]
$$

$|\rho| = 1$ 表示完全线性相关，$\rho = 0$ 表示线性无关。

## 蒙特卡洛估计

实际中经常无法直接计算解析期望，用**蒙特卡洛估计**（MC 估计）代替：从分布 $p$ 中采样 $N$ 个样本 $\{x^{(1)}, \ldots, x^{(N)}\}$，然后用样本均值近似期望：

$$
\mathbb{E}_{p}[f(X)]
\approx
\frac{1}{N} \sum_{i=1}^N f(x^{(i)})
$$

大数定律保证：当 $N \to \infty$ 时，样本均值依概率收敛到真实期望。深度学习的随机梯度下降（SGD）就是对梯度期望的 MC 估计——每次只用一个 mini-batch，而不是全部数据。

## 代码验证

```python
import numpy as np

np.random.seed(42)
n = 100_000

# 正态分布 N(2, 9)
mu_true, var_true = 2.0, 9.0
samples = np.random.normal(loc=mu_true, scale=np.sqrt(var_true), size=n)

print(f"样本均值: {samples.mean():.4f}  (真值: {mu_true})")  # ≈ 2.0
print(f"样本方差: {samples.var():.4f}  (真值: {var_true})")  # ≈ 9.0

# Var(X) = E[X^2] - (E[X])^2 验证
e_x2 = (samples ** 2).mean()
var_formula = e_x2 - samples.mean() ** 2
print(f"公式计算方差: {var_formula:.4f}")  # ≈ 9.0

# 协方差矩阵
X = np.random.multivariate_normal(
    mean=[0, 0],
    cov=[[1, 0.8], [0.8, 1]],
    size=10000
)
cov_matrix = np.cov(X.T)
print(cov_matrix.round(2))
# [[1.   0.8]
#  [0.8  1. ]]  <- 接近真实协方差矩阵

# SGD 是梯度期望的 MC 估计
full_grad = np.random.randn(1000).mean()     # 完整梯度（用全部 1000 个样本）
mini_batch_grad = np.random.randn(32).mean() # mini-batch 估计（32 个样本）
print(f"全批梯度: {full_grad:.4f}, mini-batch: {mini_batch_grad:.4f}")
```

## 在深度学习中的应用

批归一化（BatchNorm）在前向传播时计算批次的均值和方差，用它们对激活值归一化，让每层输入分布更稳定。梯度下降优化的是期望损失 $\mathbb{E}_{(\mathbf{x},y)\sim p_\text{data}}[\mathcal{L}]$，mini-batch 是对这个期望的 MC 近似。Adam 优化器维护梯度的一阶矩（期望）和二阶矩（非中心方差），用来自适应调整学习率。

下一节讲贝叶斯推断。它把先验信念和观测数据通过贝叶斯公式结合，是理解生成模型和不确定性量化的核心框架。
