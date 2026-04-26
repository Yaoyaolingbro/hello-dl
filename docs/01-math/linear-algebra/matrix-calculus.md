# 矩阵求导

!!! info "参考资料"
    **教程**
    - [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) — 矩阵求导公式手册，遇到具体公式直接查
    - [Matrix Calculus](https://explained.ai/matrix-calculus/) (Parr & Howard) — 讲解最清晰的在线教程，强烈推荐

    **教材**
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 5.3–5.5

---

## 直觉 (Intuition)

标量对标量求导是大学微积分。矩阵求导是同一件事，只是输入和输出变成了向量或矩阵。梯度是"函数在哪个方向上增长最快"，Jacobian 是"一个向量函数的每个输出分量对每个输入分量的敏感度"，Hessian 是"函数的局部曲率"。

搞清楚布局约定（分子布局还是分母布局）是避免反向传播写错的关键，因为 PyTorch 和数学教材用的不一样。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $f: \mathbb{R}^n \to \mathbb{R}$ | 标量值函数 |
| $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ | 向量值函数 |
| $\nabla_\mathbf{x} f$ | $f$ 关于向量 $\mathbf{x}$ 的梯度，形状与 $\mathbf{x}$ 相同（列向量） |
| $\mathbf{J}$ | Jacobian 矩阵 |
| $\mathbf{H}$ | Hessian 矩阵 |

---

## 四种情形

根据输入和输出的形状，求导有四种情形：

| 输入 / 输出 | 标量 $f \in \mathbb{R}$ | 向量 $\mathbf{f} \in \mathbb{R}^m$ |
|------------|------------------------|-----------------------------------|
| 标量 $x \in \mathbb{R}$ | $\partial f / \partial x \in \mathbb{R}$ | $\partial \mathbf{f} / \partial x \in \mathbb{R}^m$ |
| 向量 $\mathbf{x} \in \mathbb{R}^n$ | 梯度 $\nabla_\mathbf{x} f \in \mathbb{R}^n$ | Jacobian $\mathbf{J} \in \mathbb{R}^{m \times n}$ |

矩阵对矩阵求导也有定义，但在深度学习里通常被分解为 Jacobian 的计算，不单独处理。

---

## 梯度

标量函数 $f(\mathbf{x})$ 关于列向量 $\mathbf{x} \in \mathbb{R}^n$ 的**梯度**定义为：

$$\nabla_\mathbf{x} f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^\top \in \mathbb{R}^n$$

梯度是列向量，形状与 $\mathbf{x}$ 完全相同。这个约定在推导反向传播时非常重要。

### 常用梯度公式

**线性函数：** $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$

$$\nabla_\mathbf{x} (\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$$

**二次型：** $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x}$，其中 $\mathbf{A}$ 是方阵

$$\nabla_\mathbf{x} (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top) \mathbf{x}$$

当 $\mathbf{A}$ 对称时，$\mathbf{A} + \mathbf{A}^\top = 2\mathbf{A}$，所以 $\nabla_\mathbf{x} (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = 2\mathbf{A}\mathbf{x}$。类比标量的 $d(ax^2)/dx = 2ax$。

**$L^2$ 范数平方：** $f(\mathbf{x}) = \|\mathbf{x}\|^2 = \mathbf{x}^\top \mathbf{x}$

$$\nabla_\mathbf{x} \|\mathbf{x}\|^2 = 2\mathbf{x}$$

这是 $\mathbf{A} = \mathbf{I}$ 的特例。

**线性变换：** $f(\mathbf{x}) = \|\mathbf{W}\mathbf{x} - \mathbf{b}\|^2$（最小二乘目标）

$$\nabla_\mathbf{x} f = 2\mathbf{W}^\top (\mathbf{W}\mathbf{x} - \mathbf{b})$$

推导：令 $\mathbf{r} = \mathbf{W}\mathbf{x} - \mathbf{b}$，则 $f = \mathbf{r}^\top \mathbf{r}$，$\nabla_\mathbf{r} f = 2\mathbf{r}$，再对 $\mathbf{x}$ 用链式法则（下一节）。

---

## Jacobian 矩阵

向量值函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 的 **Jacobian** 是一个 $m \times n$ 矩阵，第 $(i,j)$ 个元素是 $\partial f_i / \partial x_j$：

$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}$$

Jacobian 的几何含义：它是函数 $\mathbf{f}$ 在当前点附近的最佳线性近似。$\mathbf{f}(\mathbf{x} + \delta) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J} \delta$。

**线性变换的 Jacobian：** 若 $\mathbf{f}(\mathbf{x}) = \mathbf{W}\mathbf{x}$，则 $\mathbf{J} = \mathbf{W}$。

**逐元素激活函数的 Jacobian：** 若 $\mathbf{f}(\mathbf{x}) = \sigma(\mathbf{x})$（逐元素），则 $\mathbf{J}$ 是对角矩阵，对角线上是 $\sigma'(x_i)$。

---

## Hessian 矩阵

标量函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的 **Hessian** 是梯度的 Jacobian，即所有二阶偏导数构成的 $n \times n$ 矩阵：

$$\mathbf{H}_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

当 $f$ 足够光滑时，Hessian 是对称矩阵（混合偏导数顺序无关）。

Hessian 描述函数的**局部曲率**：正定 Hessian 对应局部极小值，不定 Hessian 对应鞍点。这在分析优化算法（SGD vs 牛顿法）时很重要，但 Hessian 的计算量是 $O(n^2)$，神经网络里通常无法直接使用。

---

## 布局约定（容易踩坑）

矩阵求导有两种布局约定，两者的结果互为转置：

- **分子布局（Numerator Layout）**：输出形状在前，结果的行数等于输出维度。PyTorch 的 `torch.autograd.functional.jacobian` 用的是这个。
- **分母布局（Denominator Layout）**：输入形状在前，梯度与输入形状相同。本节和大多数深度学习教材用的是这个。

!!! warning "布局混淆导致反向传播出错"
    看到 $\partial \mathbf{y} / \partial \mathbf{x}$ 时，要先确认作者用的是哪种布局。形状不对是反向传播代码里最常见的 bug 之一。本节始终用**分母布局**：梯度 $\nabla_\mathbf{x} f$ 的形状与 $\mathbf{x}$ 相同。

---

## 代码验证

```python
import numpy as np

# 验证线性函数的梯度：d(a^T x)/dx = a
a = np.array([1.0, 2.0, 3.0])
x = np.array([0.5, 1.0, -0.5])

f = a @ x  # 标量值
# 解析梯度
grad_analytical = a.copy()

# 数值梯度（有限差分验证）
eps = 1e-5
grad_numerical = np.zeros_like(x)
for i in range(len(x)):
    x_plus = x.copy(); x_plus[i] += eps
    x_minus = x.copy(); x_minus[i] -= eps
    grad_numerical[i] = (a @ x_plus - a @ x_minus) / (2 * eps)

print(np.allclose(grad_analytical, grad_numerical))  # True

# 验证二次型的梯度：d(x^T A x)/dx = (A + A^T) x
A = np.array([[2.0, 1.0], [3.0, 4.0]])
x2 = np.array([1.0, 2.0])

grad_analytical_2 = (A + A.T) @ x2
# 数值梯度
grad_numerical_2 = np.zeros_like(x2)
for i in range(len(x2)):
    x_p = x2.copy(); x_p[i] += eps
    x_m = x2.copy(); x_m[i] -= eps
    grad_numerical_2[i] = (x_p @ A @ x_p - x_m @ A @ x_m) / (2 * eps)

print(np.allclose(grad_analytical_2, grad_numerical_2))  # True
```

```python
import torch

# PyTorch autograd 验证 Jacobian
x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)  # 1×3
W = torch.tensor([[1.0, 0.0, -1.0], [2.0, 1.0, 0.0]])   # 2×3
y = W @ x.T   # 2×1

# Jacobian: dy/dx 应该等于 W（2×3 矩阵）
J = torch.autograd.functional.jacobian(lambda x: (W @ x.T).squeeze(), x)
print(J.shape)   # torch.Size([2, 3])
print(torch.allclose(J, W))  # True
```

!!! tip "在深度学习中的应用"
    - **线性层梯度**：$\mathbf{y} = \mathbf{W}\mathbf{x}$ 时，$\partial L / \partial \mathbf{x} = \mathbf{W}^\top (\partial L / \partial \mathbf{y})$，$\partial L / \partial \mathbf{W} = (\partial L / \partial \mathbf{y}) \mathbf{x}^\top$。这就是全连接层反向传播的完整公式。
    - **Weight Decay**：对损失加 $\frac{\lambda}{2}\|\mathbf{W}\|_F^2$ 后，梯度额外多了 $\lambda \mathbf{W}$，即参数向零衰减。
    - **二阶优化**：L-BFGS、K-FAC 等方法近似使用 Hessian 信息，在小批量或凸问题上收敛更快。

!!! note "本节结论在后面的用处"
    梯度公式 $\nabla_\mathbf{x} (\mathbf{W}\mathbf{x}) = \mathbf{W}^\top$ 是**链式法则（矩阵形式）**的基础构件。理解 Jacobian 的含义是读懂反向传播（Part 2）和 Normalizing Flows（Part 3 生成模型）的前提。
