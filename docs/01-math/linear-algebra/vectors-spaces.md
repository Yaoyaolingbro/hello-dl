# 向量与范数

!!! info "参考资料"
    **教材**

    - Gilbert Strang, *Introduction to Linear Algebra*, 5th ed. — Chapter 1–2
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 2.4–2.7，[免费 PDF](https://mml-book.github.io/)

    **延伸阅读**

    - 3Blue1Brown, *Essence of Linear Algebra*（YouTube 可视化系列，强烈推荐配合本节）

---

## 直觉 (Intuition)

向量可以理解为"有方向的箭头"，也可以理解为"一串数字"——两种视角各有用处，在深度学习里几乎都会用到。范数是给这根箭头量长度的方式，不同的量法对应不同的几何形状和优化行为。两个向量之间的内积则回答了一个问题：它们有多"同向"？

这三件事——向量加法与缩放、长度度量、方向度量——构成了后面几乎所有数学的基础语言。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $\mathbf{x}, \mathbf{y}$ | 列向量，$\in \mathbb{R}^n$ |
| $x_i$ | 向量 $\mathbf{x}$ 的第 $i$ 个分量 |
| $\mathbf{x}^\top$ | 行向量（$\mathbf{x}$ 的转置） |
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | 内积，等价于 $\mathbf{x}^\top \mathbf{y}$（实数域） |
| $\|\mathbf{x}\|_p$ | $\mathbf{x}$ 的 $L^p$ 范数 |
| $\|\mathbf{A}\|_F$ | 矩阵 $\mathbf{A}$ 的 Frobenius 范数 |

---

## 向量空间

向量空间是一个集合，里面的元素可以相加、可以被实数缩放，且这两种操作满足一些"合理的"性质（交换律、结合律、存在零元素等）。

最常见的向量空间是 $\mathbb{R}^n$：$n$ 个实数组成的有序列表。深度学习里每个样本（图片、文本 token、点云点）都可以表示为 $\mathbb{R}^n$ 中的一个向量。

**线性相关与线性无关：** 如果一组向量 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 中没有一个能被其他向量的线性组合表示，就称它们线性无关。向量空间的**维数**等于它的最大线性无关组的大小。

---

## 内积与正交

实向量空间上的标准内积定义为：

$$\langle \mathbf{x}, \mathbf{y} \rangle = \mathbf{x}^\top \mathbf{y} = \sum_{i=1}^n x_i y_i$$

内积有两个几何含义：

1. 测量方向相似度：$\langle \mathbf{x}, \mathbf{y} \rangle = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$，其中 $\theta$ 是两向量夹角
2. 测量投影长度：$\mathbf{y}$ 在 $\mathbf{x}$ 方向上的投影长度为 $\langle \mathbf{x}/\|\mathbf{x}\|, \mathbf{y} \rangle$

当 $\langle \mathbf{x}, \mathbf{y} \rangle = 0$ 时，称 $\mathbf{x}$ 与 $\mathbf{y}$ **正交**。正交向量在几何上垂直，在信息上互不重叠——这是 Transformer 注意力机制里 Query 和 Key 点积的直觉基础。

---

## 范数

范数是向量长度的推广。直觉上，范数 $\|\mathbf{x}\|$ 越大，向量"幅度"越大。

### $L^p$ 范数

$L^p$ 范数定义为：

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}, \quad p \geq 1$$

三个最常用的特例：

**$L^1$ 范数（曼哈顿距离）：**

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$$

$L^1$ 球的形状是一个菱形（2D）。由于它在原点处不可微，用它做正则化（LASSO）时会产生稀疏解——很多分量恰好为零。

**$L^2$ 范数（欧氏距离）：**

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top \mathbf{x}}$$

$L^2$ 球是圆（2D）。它处处可微，是梯度下降的自然配套，也是权重衰减（Weight Decay）的数学形式。

**$L^\infty$ 范数（最大范数）：**

$$\|\mathbf{x}\|_\infty = \max_i |x_i|$$

控制向量中绝对值最大的分量，在量化和数值稳定性分析中偶尔出现。

### Frobenius 范数

矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 的 Frobenius 范数是把矩阵"展平"后取 $L^2$ 范数：

$$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^\top \mathbf{A})}$$

深度学习里参数量级分析、LoRA 低秩误差分析都会用到它。

!!! note "Cauchy-Schwarz 不等式"
    对任意 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$：

    $$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$$

    等号成立当且仅当 $\mathbf{x}$ 与 $\mathbf{y}$ 平行（一个是另一个的标量倍）。

    这个不等式保证了余弦相似度 $\cos\theta = \langle \mathbf{x}, \mathbf{y} \rangle / (\|\mathbf{x}\|_2 \|\mathbf{y}\|_2)$ 的值域是 $[-1, 1]$，使其成为合法的相似度度量。

**证明思路：** 考虑二次函数 $f(t) = \|\mathbf{x} + t\mathbf{y}\|_2^2 = \|\mathbf{x}\|^2 + 2t\langle\mathbf{x},\mathbf{y}\rangle + t^2\|\mathbf{y}\|^2 \geq 0$。这个二次式对所有 $t$ 非负，说明判别式 $\leq 0$，即 $4\langle\mathbf{x},\mathbf{y}\rangle^2 - 4\|\mathbf{x}\|^2\|\mathbf{y}\|^2 \leq 0$。

---

## 向量的标准化与单位球

将向量除以自身的 $L^2$ 范数，得到单位向量：$\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|_2$。

标准化是深度学习里的高频操作：

- 余弦相似度 = 标准化后的内积
- Layer Normalization 把每层激活值的范数控制在可控范围内
- 对比学习（SimCLR、CLIP）的 loss 在特征标准化后才有意义

---

## 代码验证

```python
import numpy as np

x = np.array([3.0, 4.0])
y = np.array([1.0, 0.0])

# 验证三种范数的计算
l1 = np.linalg.norm(x, ord=1)   # 3 + 4 = 7.0
l2 = np.linalg.norm(x, ord=2)   # sqrt(9+16) = 5.0
linf = np.linalg.norm(x, ord=np.inf)  # max(3,4) = 4.0
print(l1, l2, linf)  # 7.0  5.0  4.0

# 验证 Cauchy-Schwarz：|<x,y>| <= ||x|| * ||y||
inner = np.dot(x, y)             # 3.0
bound = l2 * np.linalg.norm(y)  # 5.0 * 1.0 = 5.0
print(abs(inner) <= bound)  # True

# 余弦相似度：<x,y> / (||x|| ||y||)
cos_sim = inner / (l2 * np.linalg.norm(y))
print(cos_sim)  # 0.6  （夹角约 53.1°）

# Frobenius 范数
A = np.array([[1, 2], [3, 4]])
frob = np.linalg.norm(A, 'fro')  # sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
print(frob)  # 5.477...
```

```python
# L1 vs L2 正则化的稀疏性差异
# L1 惩罚产生稀疏解，L2 惩罚产生均匀收缩
from sklearn.linear_model import Lasso, Ridge
import numpy as np

np.random.seed(0)
X = np.random.randn(50, 10)
y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(50) * 0.1  # 只有两个特征有效

lasso = Lasso(alpha=0.1).fit(X, y)
ridge = Ridge(alpha=0.1).fit(X, y)
print("Lasso 系数（稀疏）：", lasso.coef_.round(2))
# [ 0.86  0.42  0.    0.    0.    0.    0.    0.    0.    0.  ]  <- 大多数为0
print("Ridge 系数（均匀）：", ridge.coef_.round(2))
# [ 0.82  0.39  0.01 -0.01  0.02 ...]  <- 所有分量都非零但小
```

!!! tip "在深度学习中的应用"

    - **损失函数**：MSE 用 $L^2$ 范数，MAE 用 $L^1$ 范数。$L^1$ 对离群点更鲁棒，因为误差不被平方放大。
    - **正则化**：L2 正则化（Weight Decay）= 约束参数向量的 $L^2$ 范数，防止模型过拟合。
    - **注意力机制**：$\text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$ 里的分母 $\sqrt{d_k}$ 就是为了控制内积的幅度——随着维度 $d_k$ 增大，点积绝对值也会增大，除以 $\sqrt{d_k}$ 让其保持在合理范围。
    - **对比学习**：CLIP、SimCLR 在计算相似度前都先做 $L^2$ 标准化，让内积直接等于余弦相似度。

!!! note "本节结论在后面的用处"
    $L^2$ 范数和内积是**矩阵求导**（第五节）的基础：梯度 $\nabla_\mathbf{x} \|\mathbf{x}\|^2 = 2\mathbf{x}$ 的推导直接用到这里的定义。Cauchy-Schwarz 不等式在分析**优化收敛速度**时也会反复出现。
