# 矩阵运算与性质

!!! info "参考资料"
    **教材**

    - Gilbert Strang, *Introduction to Linear Algebra*, 5th ed. — Chapter 2–3
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 2.2

---

## 直觉 (Intuition)

矩阵是线性变换的表示：把一个向量变成另一个向量。两个矩阵相乘，就是两个线性变换的复合。理解矩阵运算的关键是始终记住：每次矩阵乘法都在问"这个向量经过变换后去哪了"。

迹、行列式、秩这三个数各自捕捉了矩阵的一个全局性质：迹是特征值之和，行列式是特征值之积，秩是变换后空间的维数。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $\mathbf{A} \in \mathbb{R}^{m \times n}$ | $m$ 行 $n$ 列实矩阵 |
| $A_{ij}$ | 第 $i$ 行第 $j$ 列的元素 |
| $\mathbf{a}_j$ | 矩阵 $\mathbf{A}$ 的第 $j$ 列向量 |
| $\mathbf{A}^\top$ | $\mathbf{A}$ 的转置（行列互换） |
| $\mathbf{A}^{-1}$ | $\mathbf{A}$ 的逆矩阵（$\mathbf{A}$ 需方阵且可逆） |
| $\text{tr}(\mathbf{A})$ | $\mathbf{A}$ 的迹（主对角线元素之和） |
| $\det(\mathbf{A})$ | $\mathbf{A}$ 的行列式 |
| $\text{rank}(\mathbf{A})$ | $\mathbf{A}$ 的秩 |

---

## 矩阵乘法

$\mathbf{A} \in \mathbb{R}^{m \times k}$ 与 $\mathbf{B} \in \mathbb{R}^{k \times n}$ 的乘积 $\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}$，定义为：

$$C_{ij} = \sum_{l=1}^k A_{il} B_{lj} = \mathbf{a}_i^\top \mathbf{b}_j$$

其中 $\mathbf{a}_i^\top$ 是 $\mathbf{A}$ 的第 $i$ 行，$\mathbf{b}_j$ 是 $\mathbf{B}$ 的第 $j$ 列。

矩阵乘法有三种等价的看法，各有用处：

**逐元素视角：** $C_{ij}$ = $\mathbf{A}$ 第 $i$ 行与 $\mathbf{B}$ 第 $j$ 列的内积。

**列视角：** $\mathbf{C}$ 的第 $j$ 列 = $\mathbf{A}$ 的列的线性组合，系数由 $\mathbf{B}$ 的第 $j$ 列给出：

$$\mathbf{c}_j = \mathbf{A} \mathbf{b}_j = \sum_{l=1}^k B_{lj} \mathbf{a}_l$$

**外积和（最适合并行）：** $\mathbf{C} = \sum_{l=1}^k \mathbf{a}_l \mathbf{b}_l^\top$，每项是一个秩 1 矩阵。这个视角是现代矩阵乘法加速的基础。

### 数值例子：矩阵乘法的三种视角

取 $\mathbf{A} = \begin{pmatrix}1&2\\3&4\end{pmatrix}$，$\mathbf{B} = \begin{pmatrix}5&6\\7&8\end{pmatrix}$，计算 $\mathbf{C} = \mathbf{A}\mathbf{B}$。

**逐元素视角：** $C_{11} = [1,2]\cdot[5,7]^\top = 5+14 = 19$，$C_{12} = [1,2]\cdot[6,8]^\top = 22$，以此类推。

**外积和视角（最直观）：**

$$\mathbf{C} = \underbrace{\begin{pmatrix}1\\3\end{pmatrix}\begin{pmatrix}5&6\end{pmatrix}}_{\text{第1列} \times \text{第1行}} + \underbrace{\begin{pmatrix}2\\4\end{pmatrix}\begin{pmatrix}7&8\end{pmatrix}}_{\text{第2列} \times \text{第2行}} = \begin{pmatrix}5&6\\15&18\end{pmatrix} + \begin{pmatrix}14&16\\28&32\end{pmatrix} = \begin{pmatrix}19&22\\43&50\end{pmatrix}$$

每一项 $\mathbf{a}_l \mathbf{b}_l^\top$ 是一个**秩 1 矩阵**——整个矩阵乘法是若干秩 1 矩阵的叠加。

### 外积（Outer Product）

向量 $\mathbf{a} \in \mathbb{R}^m$，$\mathbf{b} \in \mathbb{R}^n$，它们的**外积**（outer product）是一个 $m \times n$ 矩阵：

$$\mathbf{a} \otimes \mathbf{b} = \mathbf{a}\mathbf{b}^\top, \quad (\mathbf{a}\mathbf{b}^\top)_{ij} = a_i b_j$$

外积是秩 1 矩阵，所有行都是 $\mathbf{b}^\top$ 的标量倍，所有列都是 $\mathbf{a}$ 的标量倍。

**2D 例子：** $\mathbf{a} = \begin{pmatrix}1\\2\end{pmatrix}$，$\mathbf{b} = \begin{pmatrix}3\\4\end{pmatrix}$：

$$\mathbf{a}\mathbf{b}^\top = \begin{pmatrix}1\\2\end{pmatrix}\begin{pmatrix}3&4\end{pmatrix} = \begin{pmatrix}1\cdot3 & 1\cdot4\\ 2\cdot3 & 2\cdot4\end{pmatrix} = \begin{pmatrix}3&4\\6&8\end{pmatrix}$$

注意行之比 $[3,4] : [6,8] = 1:2$ 恰好是 $a_1:a_2$，秩为 1。

与内积（$\mathbf{a}^\top\mathbf{b} = 11$，标量）对比：外积"保留了全部方向信息"，内积只保留了"相似度"。

!!! warning "矩阵乘法不可交换"
    $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$（一般情况下）。这是矩阵运算最容易出错的地方。反向传播公式里的转置来自于此：梯度的形状必须与原矩阵一致，推导时要时刻注意维度。

    但结合律成立：$(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$。

---

## 迹（Trace）

方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 的迹是主对角线元素之和：

$$\text{tr}(\mathbf{A}) = \sum_{i=1}^n A_{ii}$$

迹有两个重要性质：

**迹的循环置换性：** $\text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A})$

这个性质在推导矩阵求导时极其有用——可以通过循环置换把矩阵排列成方便求导的顺序。

**迹 = 特征值之和：** $\text{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i$（见特征值分解节）。

!!! note "Frobenius 范数与迹的关系"
    $\|\mathbf{A}\|_F^2 = \text{tr}(\mathbf{A}^\top \mathbf{A}) = \sum_{i,j} A_{ij}^2$

    这个等式在分析 LoRA 的近似误差、推导权重正则化梯度时会直接用到。

---

## 行列式（Determinant）

行列式 $\det(\mathbf{A})$ 是方阵的一个标量，几何含义是：变换后单位体积被缩放的倍数。

- $\det(\mathbf{A}) > 0$：保向变换（不翻转空间）
- $\det(\mathbf{A}) < 0$：翻转变换（镜像）
- $\det(\mathbf{A}) = 0$：矩阵将空间压缩到低维，信息丢失，**矩阵不可逆**

重要性质：

$$\det(\mathbf{A}\mathbf{B}) = \det(\mathbf{A}) \cdot \det(\mathbf{B})$$

$$\det(\mathbf{A}^{-1}) = 1 / \det(\mathbf{A})$$

$$\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i \quad \text{（特征值之积）}$$

在深度学习里，行列式最常出现在归一化流（Normalizing Flows）的 log-likelihood 计算中，需要计算 Jacobian 矩阵的行列式。

### 数值例子：行列式的几何含义

取 $\mathbf{A} = \begin{pmatrix}2&0\\0&3\end{pmatrix}$（纯缩放）：$\det(\mathbf{A}) = 6$，单位正方形被拉伸为 $2\times3$ 的矩形，面积正好是 6 倍。

取 $\mathbf{B} = \begin{pmatrix}1&2\\1&2\end{pmatrix}$（两列相同方向）：$\det(\mathbf{B}) = 2-2 = 0$，空间被压缩到一维直线，面积为 0，矩阵不可逆。

---

## 秩（Rank）

矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 的秩是其列向量空间（等价地，行向量空间）的维数：

$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$

当 $\text{rank}(\mathbf{A}) = \min(m, n)$ 时，称 $\mathbf{A}$ 为**满秩矩阵**。

秩的直觉：变换后空间的维数。一个秩为 2 的 $100 \times 100$ 矩阵，会把 $\mathbb{R}^{100}$ 里的所有向量都压缩到一个二维子空间里。

!!! tip "在深度学习中的应用"
    **LoRA（Low-Rank Adaptation）** 的核心假设是：大型预训练模型的权重更新矩阵 $\Delta\mathbf{W}$ 是低秩的，即 $\text{rank}(\Delta\mathbf{W}) \ll \min(m, n)$。所以可以用两个小矩阵 $\mathbf{A} \in \mathbb{R}^{d \times r}$ 和 $\mathbf{B} \in \mathbb{R}^{r \times d}$ 来代替 $\Delta\mathbf{W}$，其中 $r \ll d$。这把微调的可训练参数量从 $d^2$ 降到 $2dr$。

---

## 矩阵的逆

方阵 $\mathbf{A}$ 可逆当且仅当 $\det(\mathbf{A}) \neq 0$（等价于满秩）。逆矩阵满足：

$$\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$$

乘积的逆：$(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$（顺序反转）

转置的逆：$(\mathbf{A}^\top)^{-1} = (\mathbf{A}^{-1})^\top$

在深度学习里，很少直接计算逆矩阵（计算量 $O(n^3)$，且数值不稳定）。遇到需要解线性系统 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 的情形，通常用 LU 分解或共轭梯度法代替。

---

## 代码验证

```python
import numpy as np

A = np.array([[1.0, 2.0], [3.0, 4.0]])
B = np.array([[5.0, 6.0], [7.0, 8.0]])

# 矩阵乘法不可交换
print(A @ B)
# [[19. 22.]
#  [43. 50.]]
print(B @ A)
# [[23. 34.]
#  [31. 46.]]
# <- 结果不同

# 迹
print(np.trace(A))  # 1 + 4 = 5.0

# 迹的循环置换：tr(AB) = tr(BA)
print(np.trace(A @ B))  # 67.0
print(np.trace(B @ A))  # 67.0  <- 相等

# 行列式
print(np.linalg.det(A))  # 1*4 - 2*3 = -2.0

# 行列式的乘法性质：det(AB) = det(A) * det(B)
print(np.linalg.det(A @ B))              # ≈ -4.0
print(np.linalg.det(A) * np.linalg.det(B))  # -2 * 2 = -4.0  <- 相等

# 秩
C = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
print(np.linalg.matrix_rank(C))  # 2（第三行 = 第一行 + 2*(第二行 - 第一行)）

# 逆矩阵
A_inv = np.linalg.inv(A)
print(A_inv @ A)
# [[1. 0.]
#  [0. 1.]]  <- 确实是单位矩阵（浮点误差级别）
```

```python
# LoRA 低秩分解示意
import numpy as np

d, r = 512, 8   # d: 原矩阵维度，r: 低秩近似秩

# 假设权重更新矩阵是低秩的
A = np.random.randn(d, r) * 0.01   # d×r 的小矩阵
B = np.random.randn(r, d) * 0.01   # r×d 的小矩阵
delta_W = A @ B                     # d×d，但秩 <= r=8

print(f"原始参数量: {d*d:,}")        # 262,144
print(f"LoRA 参数量: {d*r + r*d:,}") # 8,192  <- 节省 32 倍
print(f"delta_W 的秩: {np.linalg.matrix_rank(delta_W)}")  # 8
```

!!! note "本节结论在后面的用处"
    **迹的循环置换性**在矩阵求导的推导（下一节）中会反复用到，是化简矩阵偏导数的主要工具。**秩**的概念是理解 SVD 低秩近似（后续章节）的直接前置知识。
