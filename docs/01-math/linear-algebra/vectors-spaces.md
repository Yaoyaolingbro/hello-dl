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

### 数值例子（2D）

取 $\mathbf{x} = \begin{pmatrix}3\\4\end{pmatrix}$，$\mathbf{y} = \begin{pmatrix}1\\2\end{pmatrix}$。

**内积：** $\langle \mathbf{x}, \mathbf{y} \rangle = 3 \cdot 1 + 4 \cdot 2 = 11$

**夹角：** $\|\mathbf{x}\|_2 = 5$，$\|\mathbf{y}\|_2 = \sqrt{5}$，所以 $\cos\theta = \dfrac{11}{5\sqrt{5}} \approx 0.984$，$\theta \approx 10.3°$——两向量几乎同向。

**$\mathbf{y}$ 在 $\mathbf{x}$ 方向上的投影：**

$$\text{proj}_\mathbf{x}\mathbf{y} = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\|^2}\mathbf{x} = \frac{11}{25}\begin{pmatrix}3\\4\end{pmatrix} = \begin{pmatrix}1.32\\1.76\end{pmatrix}$$

**正交分量（残差）：** $\mathbf{y} - \text{proj}_\mathbf{x}\mathbf{y} = \begin{pmatrix}-0.32\\0.24\end{pmatrix}$，验证：$\langle \mathbf{x},\, \mathbf{y} - \text{proj}_\mathbf{x}\mathbf{y} \rangle = 3(-0.32) + 4(0.24) = 0$ ✓

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

![不同 Lp 范数的单位球形状](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Vector_norms.svg/640px-Vector_norms.svg.png)

*$p=1$（菱形）、$p=2$（圆）、$p=\infty$（正方形）的单位"球"形状对比。正则化时，约束区域的角点会造成稀疏解（$L^1$），而圆形约束（$L^2$）则均匀收缩所有分量。来源：[Wikipedia](https://en.wikipedia.org/wiki/Norm_(mathematics))*

**$L^2$ 范数（欧氏距离）：**

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top \mathbf{x}}$$

$L^2$ 球是圆（2D）。它处处可微，是梯度下降的自然配套，也是权重衰减（Weight Decay）的数学形式。

**$L^\infty$ 范数（最大范数）：**

$$\|\mathbf{x}\|_\infty = \max_i |x_i|$$

控制向量中绝对值最大的分量，在量化和数值稳定性分析中偶尔出现。

### 数值例子：同一向量的三种范数

取 $\mathbf{x} = \begin{pmatrix}3\\-4\\0\end{pmatrix}$：

$$\|\mathbf{x}\|_1 = |3| + |-4| + |0| = 7, \quad \|\mathbf{x}\|_2 = \sqrt{9+16} = 5, \quad \|\mathbf{x}\|_\infty = \max(3,4,0) = 4$$

三者满足 $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1$（$4 \leq 5 \leq 7$）。单位向量 $\hat{\mathbf{x}} = \mathbf{x}/5 = (0.6,\,-0.8,\,0)^\top$。

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


!!! tip "在深度学习中的应用"

    - **损失函数**：MSE 用 $L^2$ 范数，MAE 用 $L^1$ 范数。$L^1$ 对离群点更鲁棒，因为误差不被平方放大。
    - **正则化**：L2 正则化（Weight Decay）= 约束参数向量的 $L^2$ 范数，防止模型过拟合。
    - **注意力机制**：$\text{softmax}(\mathbf{Q}\mathbf{K}^\top / \sqrt{d_k})$ 里的分母 $\sqrt{d_k}$ 就是为了控制内积的幅度——随着维度 $d_k$ 增大，点积绝对值也会增大，除以 $\sqrt{d_k}$ 让其保持在合理范围。
    - **对比学习**：CLIP、SimCLR 在计算相似度前都先做 $L^2$ 标准化，让内积直接等于余弦相似度。

!!! note "本节结论在后面的用处"
    $L^2$ 范数和内积是**矩阵求导**（第五节）的基础：梯度 $\nabla_\mathbf{x} \|\mathbf{x}\|^2 = 2\mathbf{x}$ 的推导直接用到这里的定义。Cauchy-Schwarz 不等式在分析**优化收敛速度**时也会反复出现。
