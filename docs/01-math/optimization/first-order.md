# 一阶最优性

!!! info "参考资料"
    **主要资料**
    
    - [Deep Learning Book: Chapter 4](https://www.deeplearningbook.org/contents/numerical.html) — Ian Goodfellow et al.
    - [Mathematics for Machine Learning](https://mml-book.github.io/) — Deisenroth et al., Chapter 5–7
    - Nocedal & Wright, *Numerical Optimization*, 2nd ed. — 第 2 章，优化理论的标准教材

## 直觉 (Intuition)

梯度告诉你函数在当前点上升最快的方向；驻点是梯度为零的点——函数在此处"平坦"，可能是极小值、极大值或鞍点。一阶最优性条件是所有梯度类优化算法的基础：梯度下降、Adam、自然梯度，本质上都在"让梯度趋向零"。方向导数把梯度推广到任意方向，是理解 Taylor 展开、收敛分析和神经网络梯度流的关键工具。

## 主要符号

| 符号 | 含义 |
|------|------|
| $f: \mathbb{R}^n \to \mathbb{R}$ | 标量目标函数 |
| $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ | 梯度（偏导数构成的列向量） |
| $D_{\mathbf{v}} f(\mathbf{x})$ | 沿单位向量 $\mathbf{v}$ 的方向导数 |
| $\mathbf{x}^*$ | 驻点或最优点 |
| $L$ | 梯度的 Lipschitz 常数（光滑性参数） |

## 梯度

对多变量函数 $f(\mathbf{x}) = f(x_1, \ldots, x_n)$，**梯度（gradient）**是各偏导数构成的向量：

$$
\nabla f(\mathbf{x})
=
\left(\frac{\partial f}{\partial x_1},\, \frac{\partial f}{\partial x_2},\, \ldots,\, \frac{\partial f}{\partial x_n}\right)^\top
\in \mathbb{R}^n
$$

梯度的几何含义：$\nabla f(\mathbf{x})$ 指向 $f$ 在 $\mathbf{x}$ 处**上升最快的方向**，其模长是该方向的上升速率。

## 方向导数

函数沿单位向量 $\mathbf{v}$ 方向的变化率（**方向导数**）：

$$
D_{\mathbf{v}} f(\mathbf{x})
=
\lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{v}) - f(\mathbf{x})}{h}
=
\nabla f(\mathbf{x})^\top \mathbf{v}
$$

方向导数是梯度与方向向量的内积。由 Cauchy-Schwarz 不等式：

$$
D_{\mathbf{v}} f(\mathbf{x}) = \nabla f(\mathbf{x})^\top \mathbf{v} \le \|\nabla f(\mathbf{x})\| \cdot \|\mathbf{v}\|
$$

等号在 $\mathbf{v} = \nabla f(\mathbf{x}) / \|\nabla f(\mathbf{x})\|$ 时取到——梯度方向恰好是上升最快的方向，负梯度方向是下降最快的方向。梯度下降的合理性直接来自这里。

## 一阶 Taylor 展开

在 $\mathbf{x}_0$ 处的一阶 Taylor 展开：

$$
f(\mathbf{x}_0 + \boldsymbol{\delta})
\approx
f(\mathbf{x}_0) + \nabla f(\mathbf{x}_0)^\top \boldsymbol{\delta}
$$

这是优化步长分析的基础：给定步长 $\alpha$，沿梯度方向走一步 $\boldsymbol{\delta} = -\alpha \nabla f(\mathbf{x}_0)$，函数值的近似下降量为 $\alpha \|\nabla f(\mathbf{x}_0)\|^2$。

## 驻点与一阶最优性条件

**驻点（stationary point / critical point）**满足：

$$
\nabla f(\mathbf{x}^*) = \mathbf{0}
$$

!!! note "一阶必要条件"
    若 $\mathbf{x}^*$ 是无约束问题 $\min_\mathbf{x} f(\mathbf{x})$ 的**局部最小值**，则 $\nabla f(\mathbf{x}^*) = \mathbf{0}$。

    这是必要条件，不是充分条件。梯度为零的点可能是极小值、极大值或**鞍点（saddle point）**——需要二阶条件才能区分。

## Lipschitz 梯度（光滑性）

$f$ 的梯度是 $L$-Lipschitz 的，若存在常数 $L > 0$ 使得：

$$
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \le L\, \|\mathbf{x} - \mathbf{y}\|, \quad \forall\, \mathbf{x}, \mathbf{y}
$$

等价地，$\nabla^2 f(\mathbf{x}) \preceq L\mathbf{I}$（Hessian 的最大特征值 $\le L$）。这个常数 $L$ 决定了梯度下降的最大安全步长：步长 $\alpha \le 1/L$ 才能保证下降。学习率上界来自这里。

## 代码验证

```python
import numpy as np

# 数值梯度验证：用有限差分近似偏导数
def f(x):
    return x[0]**2 + 2*x[1]**2 + x[0]*x[1]  # 二次函数

def grad_f(x):
    # 解析梯度
    return np.array([2*x[0] + x[1], 4*x[1] + x[0]])

def numerical_grad(f, x, eps=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        grad[i] = (f(x + e) - f(x - e)) / (2 * eps)
    return grad

x0 = np.array([1.0, 2.0])
print("解析梯度:", grad_f(x0))         # [4. 9.]
print("数值梯度:", numerical_grad(f, x0))  # 应接近 [4. 9.]

# 验证方向导数 = 梯度 · 方向向量
v = np.array([1.0, 0.0])   # x 方向
dir_deriv_analytic = grad_f(x0) @ v
eps = 1e-5
dir_deriv_numerical = (f(x0 + eps*v) - f(x0 - eps*v)) / (2*eps)
print(f"\n方向导数 (解析): {dir_deriv_analytic:.4f}")
print(f"方向导数 (数值): {dir_deriv_numerical:.4f}")

# 梯度下降：找驻点
x = np.array([3.0, 3.0])
alpha = 0.1
for i in range(50):
    x = x - alpha * grad_f(x)
print(f"\n梯度下降后 x ≈ {x}")          # 应接近 [0, 0]
print(f"梯度范数: {np.linalg.norm(grad_f(x)):.2e}")  # 接近 0
```

## 在深度学习中的应用

反向传播（backprop）计算的就是损失函数对所有参数的梯度，本质是多变量链式法则（见线性代数章节的矩阵求导节）的高效实现。Adam、RMSProp 等自适应方法以梯度为输入，动态调整每个参数的学习率。RLHF 的策略梯度算法利用期望奖励对策略参数的梯度更新策略网络。

下一节讲二阶条件与曲率，Hessian 矩阵如何区分极小值、极大值和鞍点，以及为什么深度学习里的鞍点和平坦区域比局部最小值更难处理。
