# ODE 与向量场基础

!!! info "参考资料"
    **主要资料**
    
    - Strogatz, *Nonlinear Dynamics and Chaos* — 第 1–2 章，直觉驱动的 ODE 入门
    - Chen et al., "Neural Ordinary Differential Equations", NeurIPS 2018 — Neural ODE 原始论文
    - [SciPy: `solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)

## 直觉 (Intuition)

常微分方程（ODE）描述的是"状态如何随时间演化"——系统在每个时刻的变化率由当前状态决定。梯度下降可以看作一个离散 ODE，ResNet 的残差连接可以看作 ODE 的 Euler 数值积分，Neural ODE 把这个类比推到极限：用神经网络参数化 ODE 右端函数，连续时间的动力学成为可微分的计算图。Flow Matching 的向量场学习也完全建立在 ODE 语言上。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbf{z}(t) \in \mathbb{R}^n$ | 状态变量（时间 $t$ 的函数） |
| $\mathbf{f}(\mathbf{z}, t)$ | 右端函数（向量场） |
| $t_0, t_1$ | 初始/终止时间 |
| $\mathbf{z}_0 = \mathbf{z}(t_0)$ | 初始条件 |
| $h$ | 数值积分步长 |

## 一阶 ODE

**一阶常微分方程（ODE）**的一般形式：

$$
\frac{d\mathbf{z}}{dt} = \mathbf{f}(\mathbf{z}(t), t), \quad \mathbf{z}(t_0) = \mathbf{z}_0
$$

$\mathbf{f}$ 是**向量场（vector field）**：在状态空间的每个点 $(\mathbf{z}, t)$ 赋予一个速度向量，指示系统的运动方向。给定初始条件 $\mathbf{z}_0$，沿向量场积分得到**轨迹（trajectory）**。

**线性 ODE** $\dot{\mathbf{z}} = A\mathbf{z}$ 的解析解是 $\mathbf{z}(t) = e^{At}\mathbf{z}_0$，其中矩阵指数 $e^{At}$ 的稳定性由 $A$ 的特征值实部决定（实部全负则稳定）。

## 数值积分

大多数 ODE 没有解析解，用数值方法近似积分。

**Euler 法**（最简单的一阶方法）：

$$
\mathbf{z}_{k+1} = \mathbf{z}_k + h\, \mathbf{f}(\mathbf{z}_k, t_k)
$$

局部截断误差 $O(h^2)$，全局误差 $O(h)$。步长 $h$ 越小精度越高，但计算代价越大。

**Runge-Kutta 4 阶（RK4）**：

$$
\mathbf{z}_{k+1} = \mathbf{z}_k + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)
$$

其中 $k_1 = \mathbf{f}(\mathbf{z}_k, t_k)$，$k_2 = \mathbf{f}(\mathbf{z}_k + \frac{h}{2}k_1, t_k + \frac{h}{2})$，依此类推。全局误差 $O(h^4)$，是实践中最常用的定步长方法。

现代科学计算用自适应步长求解器（如 Dormand-Prince RK45、Adams-Bashforth）自动控制误差。

## ResNet 与 ODE 的对应

ResNet 的残差块：

$$
\mathbf{z}_{k+1} = \mathbf{z}_k + \mathbf{f}(\mathbf{z}_k, \theta_k)
$$

这正是 Euler 法，步长 $h=1$，向量场 $\mathbf{f}$ 由第 $k$ 层参数化。层数趋向无穷时，ResNet "极限"就是 Neural ODE——用一个连续的神经网络参数化向量场。

!!! note "Neural ODE"
    Chen et al. (2018) 提出用 ODE 求解器的**伴随方法（adjoint method）**反向传播梯度，使得整个连续动力系统端到端可训练，内存复杂度与层数无关。这开启了连续深度网络和 Flow Matching 的研究方向。

## 向量场的几何直觉

向量场在状态空间中定义了一族流线（integral curves）。几个关键概念：

- **不动点（fixed point）**：$\mathbf{f}(\mathbf{z}^*, t) = \mathbf{0}$，系统停在这里不动——对应优化的驻点
- **吸引子（attractor）**：附近轨迹都趋向的稳定不动点或极限环
- **相图（phase portrait）**：低维系统中向量场的可视化，直观展示系统的全局行为

扩散模型的前向过程（加噪）和逆向过程（去噪）都可以写成 ODE/SDE，向量场分别是已知的（高斯噪声漂移）和神经网络学习的（score function）。

## 代码验证

```python
import numpy as np
from scipy.integrate import solve_ivp

# 简谐振子：d^2x/dt^2 = -x
# 写成一阶系统：dz/dt = [z[1], -z[0]]（z=[x, v]）
def harmonic_oscillator(t, z):
    x, v = z
    return [v, -x]

# 初始条件：x(0)=1, v(0)=0 → 解析解 x(t)=cos(t)
sol = solve_ivp(harmonic_oscillator, t_span=(0, 2*np.pi),
                y0=[1.0, 0.0], method='RK45', dense_output=True)

# 验证：在 t=π 时 x 应为 -1
t_check = np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
x_numerical = sol.sol(t_check)[0]
x_analytic  = np.cos(t_check)
print("数值解:", x_numerical.round(4))
print("解析解:", x_analytic.round(4))
print(f"最大误差: {np.max(np.abs(x_numerical - x_analytic)):.2e}")  # 应 < 1e-5

# Euler 法（手动）vs. RK4 精度比较
def euler(f, z0, t0, t1, h):
    z, t = np.array(z0), t0
    while t < t1:
        z = z + h * np.array(f(t, z))
        t += h
    return z

z_euler_coarse = euler(harmonic_oscillator, [1.0, 0.0], 0, np.pi, h=0.5)
z_euler_fine   = euler(harmonic_oscillator, [1.0, 0.0], 0, np.pi, h=0.01)
print(f"\nEuler(h=0.5) x(π)={z_euler_coarse[0]:.4f}")  # 误差较大
print(f"Euler(h=0.01) x(π)={z_euler_fine[0]:.4f}")    # 接近 -1
print(f"真值 x(π)={np.cos(np.pi):.4f}")               # -1.0
```

## 在深度学习中的应用

Neural ODE（Chen et al., 2018）用 ODE 替代 ResNet 的离散层，提供了连续深度、参数量恒定的模型。Continuous Normalizing Flows（CNF）用 ODE 构建可逆流，避免了离散流模型的 Jacobian 行列式计算。Flow Matching（Lipman et al., 2022）直接回归从噪声到数据的连续向量场，比 DDPM 的采样效率高 100 倍以上，是最新一代图像/视频生成模型的基础。

下一节讲 PDE 基础，把 ODE 从"依赖时间"推广到"依赖时间和空间"，是物理仿真、科学机器学习（AI4Science）和热扩散方程与扩散模型类比的出发点。
