# PDE 基础

!!! info "参考资料"
    **主要资料**
    
    - Strauss, *Partial Differential Equations: An Introduction* — 第 1–2 章，直觉友好的入门
    - Raissi et al., "Physics-Informed Neural Networks", *J. Computational Physics* 2019 — PINN 原始论文
    - Evans, *Partial Differential Equations* — 研究生水平参考，严格证明

## 直觉 (Intuition)

PDE 和 ODE 的区别在于变量个数：ODE 只有时间，PDE 同时依赖时间和空间。热方程描述热量如何从高温区扩散到低温区；波动方程描述振动如何传播；泊松方程描述静电场分布。这些方程是物理世界的语言，也是理解扩散模型（热方程类比）、Score Matching（泊松方程）和神经网络作为 PDE 求解器（PINN）的基础。

## 主要符号

| 符号 | 含义 |
|------|------|
| $u(\mathbf{x}, t)$ | 未知函数（场量，如温度、压强） |
| $\nabla u$ | 梯度（空间偏导数） |
| $\Delta u = \nabla^2 u$ | 拉普拉斯算子（$\sum_i \partial^2 u / \partial x_i^2$） |
| $\partial_t u$ | 时间偏导数 |
| $\Omega, \partial\Omega$ | 求解域和边界 |

## PDE 与 ODE 的区别

ODE：$\frac{du}{dt} = f(u, t)$，$u$ 只依赖时间 $t$

PDE：$\frac{\partial u}{\partial t} = f\!\left(u, \nabla u, \nabla^2 u, \mathbf{x}, t\right)$，$u$ 同时依赖空间 $\mathbf{x}$ 和时间 $t$

PDE 的解是一个**场**（函数空间中的元素），而不是一条轨迹。求解需要同时给定：
- **初始条件**：$u(\mathbf{x}, 0) = u_0(\mathbf{x})$（时间初值）
- **边界条件**：在 $\partial\Omega$ 上 $u$ 的约束（Dirichlet：固定值；Neumann：固定梯度）

## 热方程（扩散方程）

$$
\frac{\partial u}{\partial t} = \kappa\, \nabla^2 u
$$

物理含义：温度的变化率正比于局部温度的"弯曲程度"（拉普拉斯算子是离散拉普拉斯算子的连续版）。热量从高温区流向低温区，长时间后温度趋向均匀（平稳态）。

在无界空间中，热方程的解是初始条件与高斯核的卷积：

$$
u(\mathbf{x}, t) = \frac{1}{(4\pi\kappa t)^{d/2}} \int u_0(\mathbf{y})\, e^{-\|\mathbf{x}-\mathbf{y}\|^2 / (4\kappa t)}\, d\mathbf{y}
$$

**扩散模型的类比**：DDPM 的前向加噪过程 $q(\mathbf{x}_t \mid \mathbf{x}_0)$ 正是热方程的解——初始数据 $\mathbf{x}_0$ 经过 $t$ 步高斯扩散变成近似标准高斯噪声，扩散系数对应热扩散率 $\kappa$。

## 波动方程

$$
\frac{\partial^2 u}{\partial t^2} = c^2\, \nabla^2 u
$$

$c$ 是波速。与热方程的区别在于时间是二阶导数：系统有"惯性"，波动不会耗散，而是持续传播。声波、电磁波、水波都满足这个方程（近似）。

## 泊松方程

$$
\nabla^2 u = f(\mathbf{x})
$$

稳态问题（与时间无关），$f$ 是已知的"源项"。静电场中 $u$ 是电势，$f$ 是电荷分布。拉普拉斯方程（$f=0$）的解是**调和函数**，满足均值性质：每点的值等于周围邻域的均值。

Score function $\nabla_\mathbf{x} \log p(\mathbf{x})$（扩散模型去噪的关键量）与泊松方程有深刻联系——Score Matching 的理论推导依赖这里的分析。

## 数值方法概览

| 方法 | 思路 | 适用场景 |
|------|------|---------|
| 有限差分（FDM） | 用差商近似偏导数，在网格上求解 | 规则几何，简单边界 |
| 有限元（FEM） | 把函数空间分解为有限维子空间，变分原理 | 复杂几何，结构力学 |
| 谱方法 | 用 Fourier/Chebyshev 基展开 | 周期或光滑问题 |
| 物理信息神经网络（PINN） | 用神经网络近似解，把 PDE 残差放入损失函数 | 高维、逆问题、稀疏数据 |

## PINN 直觉

**Physics-Informed Neural Networks（PINN）**（Raissi et al., 2019）用神经网络 $u_\theta(\mathbf{x}, t)$ 近似 PDE 的解，损失函数由三部分组成：

$$
\mathcal{L} = \underbrace{\mathcal{L}_\text{PDE}}_{\text{PDE 残差}} + \underbrace{\mathcal{L}_\text{IC}}_{\text{初始条件}} + \underbrace{\mathcal{L}_\text{BC}}_{\text{边界条件}}
$$

PDE 残差通过自动微分（autograd）计算：$\partial_t u_\theta$ 和 $\nabla^2 u_\theta$ 直接对网络输出求导。优势是不需要网格，能处理高维问题；劣势是收敛慢、对病态问题不稳定。

## 代码验证

```python
import numpy as np
from scipy.ndimage import laplace

# 用有限差分模拟 1D 热方程
# ∂u/∂t = κ ∂²u/∂x²，周期边界

def heat_step(u, kappa, dt, dx):
    # 二阶中心差分：∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / dx^2
    d2u = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
    return u + kappa * dt * d2u

# 初始条件：尖峰（delta 函数近似）
N = 100
dx = 1.0 / N
x = np.linspace(0, 1, N)
u = np.zeros(N)
u[N//2] = 1.0 / dx   # 近似 delta

kappa, dt = 0.01, 0.0001
# CFL 条件：dt <= dx^2 / (2*kappa) 保证稳定
cfl = kappa * dt / dx**2
print(f"CFL 数: {cfl:.4f}  (需 < 0.5 保证稳定)")  # 应 < 0.5

# 积分 100 步
for _ in range(100):
    u = heat_step(u, kappa, dt, dx)

# 验证：热方程解是高斯函数
t_elapsed = 100 * dt
sigma = np.sqrt(2 * kappa * t_elapsed)
gaussian = np.exp(-(x - 0.5)**2 / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi)) * dx
print(f"\n理论峰值位置: x=0.5, 数值峰值位置: x={x[np.argmax(u)]:.2f}")
print(f"理论 σ={sigma:.4f}, 数值 σ≈{np.sqrt(np.sum((x-0.5)**2 * u * dx)):.4f}")
```

## 在深度学习中的应用

DDPM 的前向加噪过程与热方程（扩散方程）直接对应，理解热方程使得 DDPM 的设计动机变得透明。PINN 让神经网络成为 PDE 的代理求解器，是 AI4Science 的核心工具（气候模拟、分子动力学、流体力学）。图神经网络中消息传递可以理解为图上的离散扩散过程，连接了 GNN 和热方程。Score Matching 中 score function 的估计与泊松方程的格林函数理论有直接联系。

这是优化理论章节的最后一节，也是数学基础（Part 1）的最后一章。线性代数、几何、概率论、优化理论四章共同构成了后续内容的基石——Part 2 的神经网络训练直接用到梯度下降和反向传播，Part 3 的生成模型依赖变分推断和 SDE，Part 3 的强化学习依赖贝叶斯滤波和动态规划。
