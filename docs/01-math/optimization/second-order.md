# 二阶条件与曲率

!!! info "参考资料"
    **主要资料**
    - Nocedal & Wright, *Numerical Optimization*, 2nd ed. — 第 2–4 章（驻点分类与牛顿法）
    - Dauphin et al., "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization", NeurIPS 2014 — 深度学习鞍点问题的实证分析

## 直觉 (Intuition)

梯度为零只说明函数"平坦"，不说明是凹还是凸——Hessian 矩阵的特征值才能区分。正定 Hessian 意味着四面都在上坡，是极小值；不定 Hessian（有正有负特征值）是鞍点。深度学习的损失面基本上不存在局部最小值，大量的"停滞点"是鞍点，SGD 的随机噪声反而帮助逃出鞍点——这是 2014 年以来的共识，理解鞍点就是理解为什么深度学习能训起来。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbf{H} = \nabla^2 f(\mathbf{x})$ | Hessian 矩阵（$n \times n$ 实对称矩阵） |
| $\lambda_{\min}, \lambda_{\max}$ | Hessian 的最小/最大特征值 |
| $\kappa = \lambda_{\max}/\lambda_{\min}$ | 条件数（刻画优化难度） |

## Hessian 矩阵

对二阶可微函数 $f: \mathbb{R}^n \to \mathbb{R}$，**Hessian 矩阵**是二阶偏导数构成的矩阵：

$$
H_{ij}
=
\frac{\partial^2 f}{\partial x_i \partial x_j}
$$

在连续可微条件下，混合偏导数对称（Schwarz 定理），所以 $\mathbf{H}$ 是实对称矩阵，特征值全为实数。

二阶 Taylor 展开给出 Hessian 的几何含义：

$$
f(\mathbf{x} + \boldsymbol{\delta})
\approx
f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \boldsymbol{\delta}
+
\frac{1}{2}\boldsymbol{\delta}^\top \mathbf{H}(\mathbf{x})\, \boldsymbol{\delta}
$$

二次项 $\frac{1}{2}\boldsymbol{\delta}^\top \mathbf{H}\boldsymbol{\delta}$ 描述了函数的**曲率**——沿 Hessian 特征向量方向，曲率大小由对应特征值决定。

## 驻点的分类

在驻点 $\mathbf{x}^*$（即 $\nabla f(\mathbf{x}^*) = \mathbf{0}$），二次项主导局部行为：

| Hessian 特征值 | 驻点类型 |
|---------------|---------|
| 全为正（正定） | 严格局部最小值 |
| 全为负（负定） | 严格局部最大值 |
| 有正有负（不定） | **鞍点** |
| 含零特征值（半正/半负定） | 需要更高阶分析 |

!!! note "二阶充分条件"
    若 $\nabla f(\mathbf{x}^*) = \mathbf{0}$ 且 $\nabla^2 f(\mathbf{x}^*) \succ 0$，则 $\mathbf{x}^*$ 是严格局部最小值。若 $\nabla^2 f(\mathbf{x}^*)$ 不定，则 $\mathbf{x}^*$ 是鞍点——存在下降方向（负特征值对应的特征向量方向）。

## 条件数与优化难度

Hessian 的**条件数（condition number）**定义为：

$$
\kappa(\mathbf{H}) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

条件数衡量不同方向的曲率差异。$\kappa = 1$ 意味着所有方向曲率相同（球形等高线），梯度下降一步到位；$\kappa \gg 1$ 意味着不同方向曲率悬殊（细长椭圆等高线），梯度下降会在大曲率方向振荡而在小曲率方向前进缓慢，需要极小的步长。

深度学习中，批归一化（BatchNorm）通过控制激活值的尺度隐式改善了损失面的条件数，这也是 BatchNorm 加速训练的一个重要原因（除了均值/方差归一化之外）。

## 深度学习中的损失面

高维非凸函数的驻点类型分布与低维截然不同。Dauphin et al. (2014) 通过分析随机矩阵理论指出：在高维情况下，随机函数的驻点**几乎都是鞍点**，几乎不存在"差"的局部最小值（即显著高于全局最小值的局部最小值）。

这意味着：
- 梯度下降面临的主要挑战是**逃离鞍点**，而不是跳出局部极小值
- SGD 的梯度噪声有助于扰动参数逃离鞍点的平坦区域
- 真正难以逃离的是损失面上的**平坦区（plateaus）**，即梯度接近零但非驻点的区域

## 代码验证

```python
import numpy as np

# 构造鞍点函数 f(x,y) = x^2 - y^2，在原点处是鞍点
def f(xy):
    x, y = xy
    return x**2 - y**2

def hessian_saddle(xy):
    # H = [[2, 0], [0, -2]]
    return np.array([[2.0, 0.0], [0.0, -2.0]])

H = hessian_saddle([0, 0])
eigenvalues = np.linalg.eigvalsh(H)
print(f"鞍点的 Hessian 特征值: {eigenvalues}")  # [−2, 2]：有正有负 → 鞍点

# 构造极小值函数 f(x,y) = x^2 + 2y^2，在原点处是严格极小值
H_min = np.array([[2.0, 0.0], [0.0, 4.0]])
eigenvalues_min = np.linalg.eigvalsh(H_min)
print(f"极小值的 Hessian 特征值: {eigenvalues_min}")  # [2, 4]：全正 → 极小值

kappa = eigenvalues_min.max() / eigenvalues_min.min()
print(f"条件数 κ = {kappa:.1f}")  # 2.0（适中，梯度下降不会太慢）

# 对比：病态问题（高条件数）
H_ill = np.array([[0.01, 0.0], [0.0, 100.0]])
eigs_ill = np.linalg.eigvalsh(H_ill)
print(f"病态 Hessian 特征值: {eigs_ill}")  # [0.01, 100.]
print(f"病态条件数: {eigs_ill.max()/eigs_ill.min():.0f}")  # 10000
```

## 在深度学习中的应用

二阶方法（牛顿法、L-BFGS）用 Hessian 或其近似来自适应地缩放梯度，能处理病态条件数，但计算代价高（$n^2$ 存储 Hessian）。拟牛顿法（L-BFGS）用梯度历史近似 Hessian 逆，是大规模优化的实用工具。Fisher 信息矩阵是对数似然的期望 Hessian，是自然梯度法和 K-FAC 的基础（Part 3 大模型训练节会详细讨论）。

下一节讲梯度下降与收敛性，把一阶和二阶条件转化为实际的更新规则和收敛速度分析。
