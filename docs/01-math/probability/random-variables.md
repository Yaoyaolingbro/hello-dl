# 随机变量

!!! info "参考资料"
    **主要资料**
    - [Deep Learning Book: Chapter 3](https://www.deeplearningbook.org/contents/prob.html) — Ian Goodfellow et al.
    - [Introduction to Probability](https://www.probabilitycourse.com/) — Hossein Pishro-Nik

    **工具文档**
    - [SciPy: Statistical functions](https://docs.scipy.org/doc/scipy/reference/stats.html)

## 直觉 (Intuition)

随机变量把随机事件的结果映射成数字，让我们能用分析工具处理不确定性。输入是样本空间里的一个结果，输出是一个数（或向量）。连续变量描述像像素值、温度这样的连续量，离散变量描述类别标签、token 索引这样的计数量。神经网络的每个输出都是随机变量：同一张图输进去，在随机初始化的模型下，输出不同。

## 主要符号

| 符号 | 含义 |
|------|------|
| $X, Y$ | 随机变量（大写字母） |
| $x$ | 随机变量的取值（小写字母） |
| $p(x)$ | 概率质量函数（离散）或概率密度函数（连续） |
| $F(x)$ | 累积分布函数 |
| $\text{supp}(X)$ | 随机变量的支撑集（取值范围） |

## 离散随机变量

离散随机变量 $X$ 的取值是有限个或可数个。描述它用**概率质量函数 (PMF)**：

$$
p(x) = P(X = x)
$$

PMF 满足两个条件：所有取值的概率非负，且加起来等于 1：

$$
p(x) \ge 0, \quad \sum_{x} p(x) = 1
$$

典型例子：骰子的 PMF 是 $p(1)=p(2)=\cdots=p(6)=1/6$。神经网络输出类别概率时，实际上是在输出一个 PMF。

## 连续随机变量

连续随机变量 $X$ 可以取实数轴上的任意值。此时单个点的概率为零，用**概率密度函数 (PDF)** 描述：

$$
P(a \le X \le b)
=
\int_a^b p(x)\, dx
$$

PDF 满足：$p(x) \ge 0$，且 $\int_{-\infty}^{\infty} p(x)\, dx = 1$。

注意 PDF 的值本身不是概率，它可以大于 1——只有积分才是概率。

## 累积分布函数（CDF）

无论离散还是连续，都可以定义**累积分布函数 (CDF)**：

$$
F(x) = P(X \le x)
$$

CDF 是单调不减函数，从 0 增长到 1。

- 对连续变量：$p(x) = F'(x)$（PDF 是 CDF 的导数）
- 对离散变量：$F(x) = \sum_{x' \le x} p(x')$（求和）

CDF 在分位数计算、生成对抗网络的分布比较、以及逆变换采样中都会用到。

!!! note "逆变换采样"
    如果 $U \sim \text{Uniform}(0,1)$（均匀分布），则 $X = F^{-1}(U)$ 服从分布 $F$。这是从任意分布采样的基础方法，也是 Normalizing Flows 的逆变换操作所依赖的原理。

## 多维随机变量

深度学习里大多数情况是多维的。$d$ 维随机向量 $\mathbf{X} = (X_1, \ldots, X_d)$ 有联合分布 $p(\mathbf{x}) = p(x_1, \ldots, x_d)$。

**边际分布**：通过对其他变量积分（求和），得到单个变量的分布：

$$
p(x_1)
=
\int p(x_1, x_2)\, dx_2
$$

**条件分布**：固定一个变量后，另一个变量的分布：

$$
p(x_1 \mid x_2)
=
\frac{p(x_1, x_2)}{p(x_2)}
$$

自回归语言模型的核心就是建模条件分布 $p(\text{token}_t \mid \text{token}_{<t})$，每次预测下一个 token 时都在计算这个条件分布。

## 代码验证

```python
import numpy as np
from scipy import stats

# 离散随机变量：均匀骰子
pmf = np.array([1/6] * 6)
values = np.arange(1, 7)

print(pmf.sum())  # 1.0，验证 PMF 归一化

# 手动 CDF
cdf = np.cumsum(pmf)
print(cdf)  # [0.167 0.333 0.5 0.667 0.833 1.0]

# 连续随机变量：标准正态分布
x = np.linspace(-4, 4, 1000)
pdf = stats.norm.pdf(x, loc=0, scale=1)

# 验证 PDF 积分 = 1（数值积分）
dx = x[1] - x[0]
print(np.sum(pdf * dx))  # ≈ 1.0

# 逆变换采样：用均匀分布生成正态分布样本
u = np.random.uniform(0, 1, size=10000)
x_samples = stats.norm.ppf(u)  # 正态分布的逆 CDF
print(f"均值: {x_samples.mean():.3f}, 标准差: {x_samples.std():.3f}")
# 均值: ≈ 0.0, 标准差: ≈ 1.0
```

## 在深度学习中的应用

神经网络的随机性（Dropout、采样层、扩散模型的噪声添加）都可以用随机变量描述。VAE 的 encoder 输出的是分布参数（均值和方差），而不是确定值——这使得隐空间可以被采样和插值。语言模型生成时，每步都在从条件分布 $p(\text{token}_t \mid \text{context})$ 中采样。

下一节讲常用分布。它们是随机变量的具体形态，深度学习里的高斯分布、伯努利分布、类别分布都会在这里登场。
