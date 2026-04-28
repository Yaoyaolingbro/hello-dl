# 拉格朗日乘子与 KKT

!!! info "参考资料"
    **主要资料**
    
    - Boyd & Vandenberghe, *Convex Optimization* — 第 4–5 章（约束优化、KKT 条件的标准参考）
    - Nocedal & Wright, *Numerical Optimization*, 2nd ed. — 第 12 章

## 直觉 (Intuition)

无约束优化的最优条件是梯度为零；有约束时，最优点的梯度未必为零，但目标函数的梯度必须与约束"抗衡"——拉格朗日乘子正是度量这种抗衡力度的系数。KKT 条件是约束优化的统一最优性条件，SVM 的支持向量、PPO 的策略约束、控制系统的操作空间优化，背后都是这套框架。

## 主要符号

| 符号 | 含义 |
|------|------|
| $f(\mathbf{x})$ | 目标函数 |
| $h_i(\mathbf{x}) = 0$ | 等式约束 |
| $g_j(\mathbf{x}) \le 0$ | 不等式约束 |
| $\lambda_i, \mu_j \ge 0$ | 拉格朗日乘子 |
| $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})$ | 拉格朗日函数 |

## 等式约束：拉格朗日乘子法

问题：$\min_\mathbf{x} f(\mathbf{x})$，s.t. $h(\mathbf{x}) = 0$。

几何直觉：最优点处，目标函数的等高线与约束曲线相切，即两者的梯度平行：$\nabla f(\mathbf{x}^*) = \lambda \nabla h(\mathbf{x}^*)$。

**拉格朗日函数（Lagrangian）**把约束融入目标：

$$
\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda\, h(\mathbf{x})
$$

最优点满足：$\nabla_\mathbf{x} \mathcal{L} = \mathbf{0}$ 且 $\nabla_\lambda \mathcal{L} = h(\mathbf{x}) = 0$。引入乘子把一个约束问题转化为无约束问题的驻点问题，是核心思路。

## KKT 条件（含不等式约束）

一般问题：

$$
\min_\mathbf{x}\; f(\mathbf{x}), \quad
\text{s.t.}\; h_i(\mathbf{x}) = 0,\; g_j(\mathbf{x}) \le 0
$$

拉格朗日函数：

$$
\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})
=
f(\mathbf{x})
+ \sum_i \lambda_i h_i(\mathbf{x})
+ \sum_j \mu_j g_j(\mathbf{x})
$$

**KKT 条件**（局部最优的必要条件）：

$$
\nabla_\mathbf{x} \mathcal{L} = \mathbf{0} \qquad \text{（驻点条件）}
$$

$$
h_i(\mathbf{x}^*) = 0 \qquad \text{（等式可行性）}
$$

$$
g_j(\mathbf{x}^*) \le 0 \qquad \text{（不等式可行性）}
$$

$$
\mu_j \ge 0 \qquad \text{（对偶可行性）}
$$

$$
\mu_j\, g_j(\mathbf{x}^*) = 0 \qquad \text{（互补松弛条件）}
$$

!!! note "互补松弛的直觉"
    $\mu_j g_j(\mathbf{x}^*) = 0$ 意味着：要么约束 $g_j$ 不起作用（$g_j < 0$，则 $\mu_j = 0$），要么约束起作用（$g_j = 0$，则 $\mu_j \ge 0$）。只有"卡在约束边界"的约束（**活跃约束**）才有非零乘子——这正是 SVM 里只有支持向量贡献到决策边界的原因。

对**凸问题**（$f$ 和 $g_j$ 均凸，$h_i$ 线性），KKT 条件是**充要**最优条件。

## 对偶问题

**对偶函数（dual function）**：

$$
d(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \min_\mathbf{x}\, \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})
$$

**对偶问题**：$\max_{\boldsymbol{\lambda}, \boldsymbol{\mu} \ge 0}\, d(\boldsymbol{\lambda}, \boldsymbol{\mu})$

**弱对偶性**（总成立）：$d^* \le f^*$（对偶最优值不超过原始最优值）。

**强对偶性**（Slater 条件满足时成立）：$d^* = f^*$，原始问题等价于对偶问题。

对偶转化的意义：有时对偶问题更容易求解（如 SVM 的核化）；对偶变量 $\mu_j$ 给出约束"松动一点"时目标值的变化率（**影子价格**）。

## 代码验证

```python
import numpy as np
from scipy.optimize import minimize

# 约束优化：最小化 f(x,y) = (x-1)^2 + (y-2)^2
# 约束：x + y = 3（等式约束）
f = lambda x: (x[0]-1)**2 + (x[1]-2)**2
grad_f = lambda x: np.array([2*(x[0]-1), 2*(x[1]-2)])

# 解析解：拉格朗日条件
# ∇f = λ ∇h → [2(x-1), 2(y-2)] = λ [1, 1]
# → x-1 = y-2 → x = y-1；加上 x+y=3 → y=2, x=1.5
print("解析解: x=1.5, y=2.0")

# SciPy 数值验证
constraint = {'type': 'eq', 'fun': lambda x: x[0] + x[1] - 3}
result = minimize(f, [0, 0], constraints=constraint, jac=grad_f)
print(f"数值解: x={result.x[0]:.4f}, y={result.x[1]:.4f}")
print(f"拉格朗日乘子 λ ≈ {result.fun:.4f}")  # lambda 可从 x-1 = λ 推出

# 验证 KKT 驻点条件
x_opt = result.x
grad_at_opt = grad_f(x_opt)
grad_constraint = np.array([1.0, 1.0])  # ∇h
lambda_kkt = grad_at_opt[0] / grad_constraint[0]
print(f"\nKKT 乘子 λ = {lambda_kkt:.4f}")  # 应为 1.0
print(f"∇f = λ∇h 验证: {np.allclose(grad_at_opt, lambda_kkt * grad_constraint)}")  # True
```

## 在深度学习中的应用

SVM 的训练是 KKT 的直接应用：支持向量（卡在边界的样本）对应非零 $\mu_j$，KKT 互补松弛条件给出了核 SVM 的对偶形式（只涉及内积，可以核化）。PPO（Proximal Policy Optimization）的信任域约束 $D_\text{KL}(\pi_\text{old} \| \pi_\text{new}) \le \delta$ 通过拉格朗日松弛近似实现。操作空间控制中的零空间投影也本质上是约束优化的 KKT 条件。

下一节讲 ADMM，一种把约束优化分解为多个更小子问题交替求解的框架，特别适合大规模分布式优化和带稀疏约束的问题。
