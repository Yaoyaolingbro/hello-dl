# 操作空间控制

!!! info "参考资料"
    **主要论文**

    - [A Unified Approach for Motion and Force Control of Robot Manipulators](https://ieeexplore.ieee.org/document/1087068) — Khatib, IEEE RA 1987（操作空间控制奠基论文）
    - [Whole-Body Dynamic Behavior and Control of Human-like Robots](https://journals.sagepub.com/doi/10.1177/0278364904049393) — Khatib et al., IJRR 2004

    **优质讲解**

    - [Stanford CS225A: Robot Control](https://cs.stanford.edu/groups/manips/teaching/cs225a/) — Khatib 教授（操作空间控制的创始人课程）

!!! note "前置依赖"
    本节整合了[速度雅可比](jacobian.md)的力学对偶关系和[动力学](dynamics.md)的方程结构，建议先读完那两节。

## 直觉 (Intuition)

关节空间控制让"关节角追目标"，但实际任务通常是末端层面的：让手爪以 5N 的力按压工件、让末端沿着工件表面滑动而不是扎进去、让机械臂拥抱一个人时感觉柔顺而不是硬邦邦。这些都是**操作空间（Operational Space / Task Space）**层面的目标，直接在关节空间描述和控制很不直观。操作空间控制把控制目标从关节角转移到末端的笛卡尔位置、速度和力，通过雅可比把末端的控制指令实时转换为关节力矩。

## 操作空间动力学

定义末端的广义坐标 $\mathbf{x} \in \mathbb{R}^m$（位置 + 朝向，$m \leq 6$），和对应的速度 $\dot{\mathbf{x}} = \mathbf{J}\dot{\mathbf{q}}$。

通过动力学变换，末端的操作空间动力学方程为：

$$\boldsymbol{\Lambda}(\mathbf{q})\ddot{\mathbf{x}} + \boldsymbol{\mu}(\mathbf{q},\dot{\mathbf{q}}) + \mathbf{p}(\mathbf{q}) = \mathbf{F}$$

其中：

- $\boldsymbol{\Lambda}(\mathbf{q}) = (\mathbf{J}\mathbf{M}^{-1}\mathbf{J}^\top)^{-1}$：**操作空间惯性矩阵**，描述末端的"有效质量"
- $\boldsymbol{\mu}$：操作空间的科里奥利项
- $\mathbf{p}$：操作空间的重力项
- $\mathbf{F}$：末端的广义力（通过 $\boldsymbol{\tau} = \mathbf{J}^\top \mathbf{F}$ 和关节力矩等价）

和关节空间动力学方程结构完全一样，只是把 $\mathbf{q}$ 替换成 $\mathbf{x}$，$\boldsymbol{\tau}$ 替换成 $\mathbf{F}$。

**核心操作**：把末端的控制力 $\mathbf{F}$ 转换成关节力矩：

$$\boldsymbol{\tau} = \mathbf{J}^\top \mathbf{F} + \mathbf{N}^\top \boldsymbol{\tau}_0$$

其中 $\mathbf{N}^\top = \mathbf{I} - \mathbf{J}^\top(\mathbf{J}^\top)^{\dagger}$ 是零空间投影，$\boldsymbol{\tau}_0$ 是次要任务的力矩（关节限位、冗余优化等）。

!!! note "直觉小结"
    操作空间控制把末端想象成一个"虚拟质量块"——给它施加一个虚拟力 $\mathbf{F}$（通过 CTC 一样的方法设计），再通过 $\mathbf{J}^\top$ 把这个力翻译成关节力矩。末端的控制设计和关节数无关，对冗余机械臂天然友好。

## 阻抗控制与导纳控制

当机械臂需要和环境（物体、人）接触时，纯位置控制（末端精确到达目标位置）是危险的——一旦发生碰撞，控制器还在努力"把末端推到目标位置"，会产生很大的接触力。

**阻抗控制（Impedance Control）**：让末端表现出像弹簧-阻尼系统一样的机械阻抗，建立末端位移和接触力之间的关系：

$$\mathbf{F}_\text{cmd} = \boldsymbol{\Lambda}_d(\ddot{\mathbf{x}}_d - \ddot{\mathbf{x}}) + \mathbf{D}_d(\dot{\mathbf{x}}_d - \dot{\mathbf{x}}) + \mathbf{K}_d(\mathbf{x}_d - \mathbf{x})$$

参数 $\boldsymbol{\Lambda}_d$（虚拟惯量）、$\mathbf{D}_d$（虚拟阻尼）、$\mathbf{K}_d$（虚拟刚度）可以独立调节。让刚度很小，末端就变得柔顺（被推动时会让步）；让阻尼很大，接触振动被快速衰减。

**导纳控制（Admittance Control）**是阻抗控制的对偶版本，从力传感器读取接触力，把力换算成位置修正量，常用于有力传感器的机器人（如 KUKA LBR iiwa、Franka Panda）。

!!! tip "工程重点"
    Franka Panda 机械臂内置了基于阻抗控制的柔顺模式（"Cartesian Impedance Control"），可以用 ROS 或 libfranka 直接设置刚度和阻尼参数。初学者常犯的错误是把刚度设得太高——末端一旦接触就产生很大力，可能损坏物体或机械臂本身。实际操作时建议从低刚度（$K_d \approx 100$ N/m）开始，逐渐增大。

## 力控制与混合力/位置控制

纯位置控制和纯力控制各有适用场景，很多实际任务需要同时控制末端的位置和力：

- 打磨任务：沿工件表面法线方向控制力（保持恒定压力），沿工件表面切线方向控制位置（走直线）
- 插入任务：沿插入方向控制力（避免卡住时过大力），垂直方向控制对准精度

**混合力/位置控制（Hybrid Force/Position Control，Raibert & Craig 1981）**：在任务空间里，不同方向上分别应用力控制或位置控制，用"选择矩阵"$\mathbf{S}$ 区分哪些自由度受力控，哪些受位置控。

$$\boldsymbol{\tau} = \mathbf{J}^\top(\mathbf{S}\mathbf{F}_\text{force} + (\mathbf{I}-\mathbf{S})\mathbf{F}_\text{position})$$

## 代码

操作空间控制器（CTC 版本）在仿真中的单步执行：

```python
import numpy as np

def operational_space_ctc(q, dq, x_d, dx_d, ddx_d, F_contact,
                          fk_func, jac_func, M_func, g_func,
                          Kp_x, Kd_x):
    """
    操作空间计算力矩控制
    x_d, dx_d, ddx_d: 末端目标位置/速度/加速度
    F_contact: 接触力（由力传感器读取），用于力控场景
    返回关节力矩
    """
    x  = fk_func(q)[:3, 3]    # 末端当前位置（简化用位置，忽略朝向）
    J  = jac_func(q)[3:, :]   # 线速度部分雅可比 (3×n)
    M  = M_func(q)
    g  = g_func(q)

    # 操作空间惯性矩阵
    M_inv = np.linalg.inv(M)
    Lambda = np.linalg.inv(J @ M_inv @ J.T)   # (3×3)

    # 位置误差
    e  = x_d - x
    de = dx_d - J @ dq

    # 操作空间 PD + 前馈
    F_cmd = Lambda @ (ddx_d + Kd_x @ de + Kp_x @ e)

    # 把末端力转换为关节力矩
    tau_task = J.T @ F_cmd
    tau_g    = g                             # 重力补偿
    tau      = tau_task + tau_g

    return tau
```

## 在现代具身智能中的应用

操作空间控制直接支撑了现代机器人操作研究的工程基础设施：

Franka Panda 的 Cartesian Impedance Controller、UR 系列的 Force Control 模式、Boston Dynamics Spot 的末端力控，都是操作空间控制的工程实现。

更重要的是，操作空间框架给学习策略（强化学习、模仿学习）提供了一个更好的动作空间：相比直接预测关节角度，预测末端的速度（$\Delta \mathbf{x}$）或力（$\mathbf{F}$）作为动作更接近任务层面，学习效率和泛化性都更好。OpenAI 的 Dactyl、DeepMind 的 MuJoCo 机械臂实验都用末端速度作为动作空间。

## 开放问题

传统操作空间控制依赖精确的动力学模型（$\mathbf{M}, \mathbf{C}, \mathbf{g}$）和实时力矩控制（通常需要 1kHz 以上的控制频率）。这两点对低成本机器人（力矩传感器贵、电机控制频率低）都是挑战。

低成本机器人（如 SO-ARM、LeRobot 的 Koch 系列）通常只有位置控制接口，没有力矩控制，操作空间控制难以直接应用。这是学术研究和实际部署之间目前最大的 Gap 之一，也是推动端到端学习策略（直接预测关节角）发展的重要工程原因。

这是传统运动控制章节的最后一节。从空间描述、正运动学、雅可比、逆运动学、轨迹规划，到动力学和控制器设计，这套工具链是机器人操作（Manipulation）、双足行走、无人机控制等所有传统机器人控制的数学基础。Part 4 后续章节里的 VLA、扩散策略等现代方法，都是在这套框架的基础上加入了感知和学习能力。
