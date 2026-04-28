# ADMM 及相关算法

!!! info "参考资料"
    **主要资料**
    
    - Boyd et al., "Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers", *Foundations and Trends in ML* 2011 — ADMM 标准综述，免费 PDF
    - Parikh & Boyd, "Proximal Algorithms", *Foundations and Trends in Optimization* 2014 — 近端算法综述

## 直觉 (Intuition)

ADMM（Alternating Direction Method of Multipliers，交替方向乘子法）把一个大优化问题分解成若干更小的子问题，交替求解。核心思想是：先把问题拆成两部分，各自独立优化，然后用乘子更新把它们"拉"向一致。这个框架特别适合分布式优化（每台机器只需解一个子问题）和含结构化稀疏约束的问题（如 LASSO），是联邦学习和分布式深度学习的基础方法之一。

## 主要符号

| 符号 | 含义 |
|------|------|
| $f, g$ | 待优化的两个函数（允许非光滑） |
| $\rho > 0$ | 增广拉格朗日罚参数 |
| $\mathbf{y}$ | 对偶变量（乘子） |
| $\mathbf{prox}_f(\mathbf{v})$ | $f$ 的近端算子 |

## 问题设定

ADMM 求解以下**分裂（separable）**结构的问题：

$$
\min_{\mathbf{x}, \mathbf{z}}\; f(\mathbf{x}) + g(\mathbf{z}), \quad \text{s.t.}\; A\mathbf{x} + B\mathbf{z} = \mathbf{c}
$$

$f$ 和 $g$ 各自可以非光滑（如 L1 范数），但各自"好处理"（有闭式近端算子）。

## 增广拉格朗日

对约束 $A\mathbf{x} + B\mathbf{z} = \mathbf{c}$ 引入乘子 $\mathbf{y}$，并加入二次罚项：

$$
\mathcal{L}_\rho(\mathbf{x}, \mathbf{z}, \mathbf{y})
=
f(\mathbf{x}) + g(\mathbf{z})
+ \mathbf{y}^\top(A\mathbf{x} + B\mathbf{z} - \mathbf{c})
+ \frac{\rho}{2}\|A\mathbf{x} + B\mathbf{z} - \mathbf{c}\|^2
$$

二次罚项使问题更容易数值求解（更好的条件数），但如果直接对 $(\mathbf{x}, \mathbf{z})$ 联合优化则失去了分裂结构。ADMM 的关键是**交替**最小化：

$$
\mathbf{x}^{k+1} = \arg\min_\mathbf{x}\, \mathcal{L}_\rho(\mathbf{x}, \mathbf{z}^k, \mathbf{y}^k)
$$

$$
\mathbf{z}^{k+1} = \arg\min_\mathbf{z}\, \mathcal{L}_\rho(\mathbf{x}^{k+1}, \mathbf{z}, \mathbf{y}^k)
$$

$$
\mathbf{y}^{k+1} = \mathbf{y}^k + \rho(A\mathbf{x}^{k+1} + B\mathbf{z}^{k+1} - \mathbf{c})
$$

x-步和 z-步各自可以利用 $f$、$g$ 的结构高效求解，乘子更新是梯度上升步。

## 近端算子

**近端算子（proximal operator）**是 ADMM 各子步的核心工具：

$$
\mathbf{prox}_{\lambda f}(\mathbf{v}) = \arg\min_\mathbf{x}\left[f(\mathbf{x}) + \frac{1}{2\lambda}\|\mathbf{x} - \mathbf{v}\|^2\right]
$$

直觉：在 $f$ 的惩罚和与 $\mathbf{v}$ 保持近距离之间权衡。许多常见函数有闭式近端算子：

| 函数 $f$ | $\mathbf{prox}_{\lambda f}(\mathbf{v})$ | 名称 |
|---------|----------------------------------------|------|
| $\lambda\|\mathbf{x}\|_1$ | $\text{sign}(\mathbf{v}) \cdot \max(|\mathbf{v}| - \lambda, 0)$ | 软阈值（Soft Thresholding） |
| $\frac{1}{2}\|\mathbf{x}\|_2^2$ | $\mathbf{v}/(1+\lambda)$ | L2 收缩 |
| $\delta_{\mathcal{C}}(\mathbf{x})$（集合指示函数） | $\Pi_\mathcal{C}(\mathbf{v})$（投影） | 投影算子 |

LASSO（$\min \frac{1}{2}\|A\mathbf{x}-\mathbf{b}\|^2 + \lambda\|\mathbf{x}\|_1$）的 ADMM 子步正是软阈值操作。

## LASSO 示例

LASSO 回归用 ADMM 求解的完整形式：令 $f(\mathbf{x}) = \frac{1}{2}\|A\mathbf{x}-\mathbf{b}\|^2$，$g(\mathbf{z}) = \lambda\|\mathbf{z}\|_1$，约束 $\mathbf{x} = \mathbf{z}$：

$$
\mathbf{x}^{k+1} = (A^\top A + \rho I)^{-1}(A^\top \mathbf{b} + \rho(\mathbf{z}^k - \mathbf{y}^k))
$$

$$
\mathbf{z}^{k+1} = \text{prox}_{\lambda/\rho \cdot \|\cdot\|_1}(\mathbf{x}^{k+1} + \mathbf{y}^k) = \text{SoftThresh}_{\lambda/\rho}(\mathbf{x}^{k+1} + \mathbf{y}^k)
$$

$$
\mathbf{y}^{k+1} = \mathbf{y}^k + \mathbf{x}^{k+1} - \mathbf{z}^{k+1}
$$

x-步是线性方程组（可以预分解 $A^\top A + \rho I$），z-步是软阈值，乘子更新是简单加法。

## 代码验证

```python
import numpy as np

def soft_threshold(v, threshold):
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)

def admm_lasso(A, b, lam=1.0, rho=1.0, n_iter=100):
    m, n = A.shape
    x = np.zeros(n)
    z = np.zeros(n)
    y = np.zeros(n)   # 对偶变量（scaled form）

    # 预计算：只需分解一次
    ATA = A.T @ A
    Atb = A.T @ b
    L = np.linalg.cholesky(ATA + rho * np.eye(n))

    for _ in range(n_iter):
        # x 步：解线性方程
        rhs = Atb + rho * (z - y)
        x = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        # z 步：软阈值（LASSO 近端算子）
        z = soft_threshold(x + y, lam / rho)
        # 乘子更新
        y = y + x - z

    return x

np.random.seed(42)
n, m = 50, 20
A = np.random.randn(m, n)
x_true = np.zeros(n)
x_true[:5] = [3, -2, 1.5, -1, 0.5]   # 稀疏真值
b = A @ x_true + 0.1 * np.random.randn(m)

x_admm = admm_lasso(A, b, lam=0.1, rho=1.0)
print(f"恢复误差: {np.linalg.norm(x_admm - x_true):.4f}")
print(f"非零元素数: {(np.abs(x_admm) > 1e-4).sum()}")  # 应接近 5
```

## 在深度学习中的应用

联邦学习（Federated Learning）用 ADMM 分布式优化：每台设备解局部子问题，服务器聚合乘子更新，避免直接传输原始数据。模型压缩中的结构化剪枝可以用 ADMM 把权重矩阵约束为低秩或稀疏形式。**近端梯度法**（如 ISTA）是 ADMM 的简化版本，在 L1 稀疏正则化的神经网络训练中有应用。

下一节讲插值与样条基础，从多项式插值到 B 样条，为机器人轨迹规划和图形学曲线表示打基础。
