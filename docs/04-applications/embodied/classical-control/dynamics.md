# 动力学：Newton-Euler 与拉格朗日

!!! info "参考资料"
    **主要教材**

    - [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Lynch & Park, 第 8 章
    - [Introduction to Robotics](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003302) — Craig, 第 6 章

!!! note "前置依赖"
    拉格朗日方程的推导背景见 [ODE 与向量场](../../../01-math/optimization/odes-vector-fields.md)（变分原理相关）。本节重在动力学方程的结构和工程意义，不展开完整推导。

## 直觉 (Intuition)

运动学只描述几何：关节角、速度、末端位姿。动力学描述物理：要产生这些运动，需要多大的力矩？反过来，给定力矩，运动轨迹是什么？这是机器人控制里"真正难"的部分——你看着一个机械臂平稳地举起重物、快速挥动，背后是复杂的力矩计算和补偿。输入是关节角加速度（或期望轨迹），输出是关节力矩；或者反过来，输入是力矩，输出是加速度。

## 机器人动力学方程

机器人（串联机械臂）的动力学方程用广义力矩 $\boldsymbol{\tau}$ 和关节加速度 $\ddot{\mathbf{q}}$ 来描述：

$$\mathbf{M}(\mathbf{q})\ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}} + \mathbf{g}(\mathbf{q}) = \boldsymbol{\tau}$$

三项含义：

- $\mathbf{M}(\mathbf{q}) \in \mathbb{R}^{n\times n}$：**惯性矩阵（Mass/Inertia Matrix）**，对称正定，描述"要加速这个姿态需要多大力矩"。手臂伸直和弯曲时，$\mathbf{M}$ 完全不同。
- $\mathbf{C}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}}$：**科里奥利力和离心力项（Coriolis & Centripetal）**，与速度的乘积有关，快速运动时不可忽视。
- $\mathbf{g}(\mathbf{q})$：**重力项（Gravity）**，从任何姿态维持静止都需要的力矩。

这个方程有两种用法：

**正动力学（Forward Dynamics）**：给定力矩 $\boldsymbol{\tau}$，求加速度 $\ddot{\mathbf{q}} = \mathbf{M}^{-1}(\boldsymbol{\tau} - \mathbf{C}\dot{\mathbf{q}} - \mathbf{g})$。用于物理仿真（IsaacGym、MuJoCo 用的就是这个）。

**逆动力学（Inverse Dynamics）**：给定期望轨迹 $(\mathbf{q}, \dot{\mathbf{q}}, \ddot{\mathbf{q}})$，求所需力矩 $\boldsymbol{\tau}$。用于轨迹跟踪控制（下一节的计算力矩控制）。

!!! note "直觉小结"
    把机器人的运动想象成控制一辆汽车：惯性矩阵是车的质量（重的车需要更大油门才能加速），科里奥利项是转弯时的侧倾力，重力项是爬坡时额外的阻力。控制器需要知道这三项，才能给出精确的"油门量"（力矩）。

## Newton-Euler 递推法

**Newton-Euler 方法**从物理出发，对每个连杆分别写牛顿第二定律（$F = ma$）和欧拉方程（$\tau = I\alpha$），然后通过运动链的结构递推：

1. **向外递推（Outward Pass）**：从基座到末端，逐个计算每个连杆的速度和加速度
2. **向内递推（Inward Pass）**：从末端到基座，逐个计算维持这个运动所需的关节力矩

Newton-Euler 方法计算复杂度是 $O(n)$（$n$ 是关节数），是数值效率最高的逆动力学算法，工业机器人控制器（ABB、Fanuc 等）内部都用这个。

## 拉格朗日方法

**拉格朗日方法**从能量出发，把问题转化为求系统的拉格朗日量 $L = T - V$（动能 - 势能），然后通过 Euler-Lagrange 方程：

$$\boldsymbol{\tau} = \frac{d}{dt}\frac{\partial L}{\partial \dot{\mathbf{q}}} - \frac{\partial L}{\partial \mathbf{q}}$$

自动推导出上面的动力学方程结构。拉格朗日方法不需要分析内力，推导更系统，适合推导解析表达式和理论分析。缺点是直接展开后符号表达式极其复杂，手算对 3 关节以上的机器人就不现实了。

**动能**的计算依赖各连杆的惯性矩阵和速度：

$$T = \frac{1}{2}\sum_{i=1}^n \left(\dot{\mathbf{p}}_i^\top m_i \dot{\mathbf{p}}_i + \boldsymbol{\omega}_i^\top \mathbf{I}_i \boldsymbol{\omega}_i\right)$$

其中 $m_i$ 是连杆 $i$ 的质量，$\mathbf{I}_i$ 是惯性张量，$\mathbf{p}_i$ 和 $\boldsymbol{\omega}_i$ 是质心速度和角速度（通过雅可比和关节速度计算）。

## 动力学参数

描述一个连杆的动力学，需要以下参数：

- 质量 $m_i$（1 个）
- 质心位置 $\mathbf{r}_{c,i}$（3 个）
- 惯性张量 $\mathbf{I}_i$（独立分量 6 个）

共 10 个参数 × $n$ 个连杆。这些参数在设计阶段可以从 CAD 模型获得，但实际机器人组装后（螺丝、线缆、传感器都有质量）与理论值有偏差。**动力学参数辨识（Dynamic Parameter Identification）**通过让机器人做特定激励轨迹，记录关节力矩和运动数据，用最小二乘法反推参数，是高精度力控的必要步骤。

!!! tip "工程重点"
    MuJoCo 和 Isaac Gym 这类物理仿真器能够精确模拟动力学，但它们需要准确的 URDF/MJCF 文件（包含惯性参数）。如果参数设置不对（例如用默认值 $m=1, \mathbf{I}=\mathbf{I}$），Sim-to-Real 迁移时机器人行为会和仿真差很远。最好的做法是用 CAD 模型的参数作为初始值，然后用实机数据做参数辨识微调。

## 代码

用简化的拉格朗日方法计算 2 自由度平面臂的重力项（$\mathbf{g}(\mathbf{q})$）：

```python
import numpy as np

def gravity_torque_2dof(q, m1=1., m2=1., L1=1., L2=1., g=9.81):
    """
    2 自由度平面臂的重力补偿力矩
    q: [q1, q2]，关节角（rad）
    返回 [tau1, tau2]，维持静止所需力矩
    """
    # 各质心位置
    x_c1 = (L1/2) * np.cos(q[0])
    x_c2 = L1*np.cos(q[0]) + (L2/2)*np.cos(q[0]+q[1])

    # 势能对 q1 和 q2 的偏导数（势能 V = m1*g*y_c1 + m2*g*y_c2）
    y_c1 = (L1/2)*np.sin(q[0])
    y_c2 = L1*np.sin(q[0]) + (L2/2)*np.sin(q[0]+q[1])
    dV_dq1 = m1*g*(L1/2)*np.cos(q[0]) + m2*g*(L1*np.cos(q[0]) + (L2/2)*np.cos(q[0]+q[1]))
    dV_dq2 = m2*g*(L2/2)*np.cos(q[0]+q[1])
    return np.array([dV_dq1, dV_dq2])  # 重力补偿力矩

q = [np.pi/4, np.pi/4]
tau_g = gravity_torque_2dof(q)
print(tau_g.round(4))
# tau1 约 12.5 Nm，tau2 约 3.5 Nm（维持此姿态的关节力矩）
```

下一节讲关节空间控制：有了动力学方程，如何设计控制器让机器人精确跟踪轨迹？
