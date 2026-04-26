# 特征值分解

!!! info "参考资料"
    **教材**

    - Gilbert Strang, *Introduction to Linear Algebra*, 5th ed. — Chapter 6
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 4.2–4.4

    **可视化**

    - 3Blue1Brown, "Eigenvectors and eigenvalues"（YouTube）

---

## 直觉 (Intuition)

大多数向量经过矩阵变换后，方向和长度都会改变。但有一些"特殊"的向量，经过变换后方向不变，只是被拉伸或压缩了——这些向量就是特征向量，拉伸的倍数就是特征值。

特征值分解把矩阵"对角化"，把一个复杂的线性变换分解成：先旋转到自然坐标系，做纯缩放，再旋转回去。理解这个分解，是理解 PCA、谱聚类、GNN 谱卷积的共同基础。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $\lambda_i$ | 第 $i$ 个特征值（实数，对对称矩阵） |
| $\mathbf{v}_i$ | 对应 $\lambda_i$ 的特征向量（列向量） |
| $\mathbf{Q}$ | 特征向量矩阵（列为特征向量） |
| $\mathbf{\Lambda}$ | 特征值对角矩阵，$\Lambda_{ii} = \lambda_i$ |

---

## 定义

若向量 $\mathbf{v} \neq \mathbf{0}$ 满足：

$$\mathbf{A} \mathbf{v} = \lambda \mathbf{v}$$

则称 $\lambda$ 是 $\mathbf{A}$ 的**特征值**，$\mathbf{v}$ 是对应的**特征向量**。

几何含义：$\mathbf{v}$ 被矩阵 $\mathbf{A}$ 变换后只改变长度（缩放 $\lambda$ 倍），方向不变（$\lambda < 0$ 时方向反转）。

求特征值：$(A - \lambda \mathbf{I})\mathbf{v} = \mathbf{0}$ 有非零解的条件是 $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$（特征多项式）。

---

## 特征值分解（EVD）

若方阵 $\mathbf{A}$ 有 $n$ 个线性无关的特征向量 $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$，则可以分解为：

$$\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^{-1}$$

其中 $\mathbf{Q} = [\mathbf{v}_1, \ldots, \mathbf{v}_n]$（列为特征向量），$\mathbf{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_n)$。

矩阵乘以向量 $\mathbf{A}\mathbf{x}$ 的计算过程可以理解为：
1. $\mathbf{Q}^{-1}\mathbf{x}$：将 $\mathbf{x}$ 变换到特征基下
2. $\mathbf{\Lambda}(\cdot)$：在特征基下做纯缩放（每个分量乘以对应特征值）
3. $\mathbf{Q}(\cdot)$：变换回原坐标系

---

## 谱定理（最重要的结论）

!!! note "谱定理（Spectral Theorem）"
    实对称矩阵 $\mathbf{A} = \mathbf{A}^\top$ 一定可以**正交对角化**：

    $$\mathbf{A} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^\top$$

    其中 $\mathbf{Q}$ 是正交矩阵（$\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$），$\mathbf{\Lambda}$ 是实数对角矩阵。

    直觉：对称矩阵在它"自己的坐标系"里就是纯粹的缩放操作。没有旋转，没有剪切，只有沿各坐标轴的缩放。

**证明思路（归纳）：** 对 $n=1$ 显然成立。对一般 $n$，可以证明实对称矩阵至少有一个实特征值 $\lambda_1$，对应特征向量 $\mathbf{v}_1$。在 $\mathbf{v}_1$ 的正交补空间里，$\mathbf{A}$ 的限制仍然是对称矩阵，递归应用即可。

**推论：**

- 对称矩阵的所有特征值都是实数
- 不同特征值对应的特征向量相互正交
- 正定矩阵 $\Leftrightarrow$ 所有特征值 $> 0$（正定性等价条件）

---

## 矩阵的幂次与函数

特征值分解使得矩阵幂次的计算变得简单：

$$\mathbf{A}^k = \mathbf{Q} \mathbf{\Lambda}^k \mathbf{Q}^\top$$

其中 $\mathbf{\Lambda}^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$——只需对角线上每个数求幂。

更一般地，对任意函数 $f$：

$$f(\mathbf{A}) = \mathbf{Q} f(\mathbf{\Lambda}) \mathbf{Q}^\top = \mathbf{Q} \, \text{diag}(f(\lambda_1), \ldots, f(\lambda_n)) \, \mathbf{Q}^\top$$

这是矩阵指数、矩阵对数等操作的定义方式，在某些深度学习模型（如 SPD 网络）中会用到。

---

## PCA 与特征值分解

主成分分析（PCA）的核心就是对协方差矩阵做特征值分解。给定数据矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$（已中心化），协方差矩阵 $\mathbf{S} = \mathbf{X}^\top \mathbf{X} / n$ 是对称正半定的。

对 $\mathbf{S}$ 做特征值分解 $\mathbf{S} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$：

- 特征向量（$\mathbf{Q}$ 的列）是主成分方向
- 特征值是各方向上的方差
- 取前 $k$ 个最大特征值对应的特征向量，即可将数据从 $d$ 维降到 $k$ 维

---

## 代码验证

```python
import numpy as np

# 对称矩阵的特征值分解
A = np.array([[4.0, 2.0], [2.0, 3.0]])  # 对称正定矩阵

# eigvalsh 专门用于对称矩阵，返回实数特征值，比 eig 更稳定
eigenvalues, eigenvectors = np.linalg.eigh(A)
print("特征值:", eigenvalues)   # [1.697..., 5.302...]，全正（正定）

# 验证 A = Q Λ Q^T
Lambda = np.diag(eigenvalues)
Q = eigenvectors
reconstructed = Q @ Lambda @ Q.T
print(np.allclose(reconstructed, A))  # True

# 验证特征向量正交：Q^T Q = I
print(np.allclose(Q.T @ Q, np.eye(2)))  # True

# 验证特征方程：A v = λ v
for i in range(2):
    lhs = A @ Q[:, i]
    rhs = eigenvalues[i] * Q[:, i]
    print(np.allclose(lhs, rhs))  # True, True

# 用特征值分解快速计算矩阵幂次
A_power_5_direct = np.linalg.matrix_power(A, 5)
A_power_5_evd = Q @ np.diag(eigenvalues ** 5) @ Q.T
print(np.allclose(A_power_5_direct, A_power_5_evd))  # True
```

```python
# PCA 的完整实现（基于特征值分解）
import numpy as np

np.random.seed(42)
# 生成相关数据：主方向是 [1, 2] / sqrt(5)
n, d = 100, 2
X = np.random.randn(n, d) @ np.array([[2, 0], [1, 1]])  # 数据有相关性

# 中心化
X_centered = X - X.mean(axis=0)

# 协方差矩阵
S = X_centered.T @ X_centered / n

# 特征值分解（eigh 保证结果按升序排列）
eigenvalues, eigenvectors = np.linalg.eigh(S)

# PCA 降维：取最大特征值对应的方向（降到 1 维）
principal_component = eigenvectors[:, -1]  # 最大特征值对应的列
X_projected = X_centered @ principal_component
print(f"原始数据方差: {X_centered.var(axis=0)}")
print(f"主成分解释方差: {eigenvalues[-1]:.3f} / {eigenvalues.sum():.3f} = {eigenvalues[-1]/eigenvalues.sum():.1%}")
```

!!! tip "在深度学习中的应用"

    - **PCA 预处理**：高维特征降维，去除冗余信息，在 NLP 早期和图像处理中常用。
    - **谱图卷积**：GNN 的谱方法（ChebNet、GCN）依赖图拉普拉斯矩阵的特征分解，下一节详解。
    - **注意力矩阵分析**：分析 Transformer 的注意力头时，用特征值分析其"信息混合"方式。
    - **Hessian 分析**：优化理论里用 Hessian 的特征值分析 loss landscape 的曲率（尖锐最小值 vs 平坦最小值）。

!!! note "本节结论在后面的用处"
    谱定理是 **SVD**（下一节）的理论基础。SVD 把特征值分解推广到**非方阵**，是更一般的工具。特征值分解也是图拉普拉斯矩阵分析（本章最后一节）的核心。
