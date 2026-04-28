# 正运动学：DH 参数法与指数积

!!! info "参考资料"
    **主要教材**

    - [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Lynch & Park, 第 4 章（PoE 方法）
    - [Introduction to Robotics](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003302) — Craig, 2005（DH 参数法经典来源）

    **优质讲解**

    - [Stanford CS223A: Forward Kinematics](https://cs.stanford.edu/groups/manips/teaching/cs223a/)

!!! note "前置依赖"
    本节依赖 [空间描述](spatial-descriptions.md) 里的齐次变换矩阵，以及 [Lie 群与 Lie 代数](../../../../01-math/geometry/lie-groups.md) 里的 $SE(3)$ 和 Exp 映射（指数积公式用到）。

## 直觉 (Intuition)

正运动学（Forward Kinematics，FK）问题是：给定机械臂每个关节的角度，求末端执行器（End-Effector）在空间中的位姿。想象一条手臂：肩膀转多少度、肘关节弯多少度、手腕转多少度——三者合起来，手在哪？输入是关节角度向量 $\mathbf{q} \in \mathbb{R}^n$，输出是末端位姿 $\mathbf{T} \in SE(3)$。正运动学总有唯一解，这是它和"逆运动学"的根本区别。

## 运动链的结构

机器人手臂是一条**运动链（Kinematic Chain）**：基座（Base）→ 连杆 1 → 关节 1 → 连杆 2 → 关节 2 → … → 末端。末端的位姿等于沿链的所有变换矩阵的乘积：

$$\mathbf{T}_{0n}(\mathbf{q}) = \mathbf{T}_{01}(q_1) \cdot \mathbf{T}_{12}(q_2) \cdots \mathbf{T}_{(n-1)n}(q_n)$$

问题是：如何系统地为每个关节定义 $\mathbf{T}_{i,i+1}(q_i)$？

## DH 参数法

**Denavit-Hartenberg（DH）参数**是 1955 年提出的经典方法，把每两相邻关节之间的变换用 4 个参数描述：

| 参数 | 含义 |
|------|------|
| $a_i$ | 沿 $x_i$ 轴，两关节轴的公垂线长度（连杆长度）|
| $\alpha_i$ | 绕 $x_i$ 轴，关节轴 $i$ 和 $i+1$ 之间的夹角（连杆扭角）|
| $d_i$ | 沿 $z_i$ 轴的偏置距离（关节偏置）|
| $\theta_i$ | 绕 $z_i$ 轴的关节角（**这是转动关节的变量**）|

每个关节的变换矩阵（标准 DH）：

$$\mathbf{T}_{i-1,i}(\theta_i) = \mathbf{R}_z(\theta_i) \cdot \mathbf{T}_z(d_i) \cdot \mathbf{T}_x(a_i) \cdot \mathbf{R}_x(\alpha_i)$$

优点是参数最少（每个关节 4 个），缺点是坐标系的选法有约束，对平行或相交关节轴不直观，也有改进版本（Modified DH，Khalil-Kleinfinger 约定）。

!!! warning "常见误区"
    DH 参数有"标准版"和"改进版"两种，两者的参数含义和矩阵形式都不同。ROS/MoveIt! 的 URDF 格式不用 DH 参数，而是直接用每个连杆的位姿偏移，切换时要注意对应关系。很多机器人 datasheet 给的 DH 表是改进 DH，直接套标准 DH 公式会出错。

## 指数积（Product of Exponentials）公式

现代机器人学更倾向于使用**指数积（PoE）公式**，因为它的几何意义更清晰，也更适合与 Lie 群框架配合使用。

每个旋转关节对应一个**关节螺旋轴（Screw Axis）** $\mathcal{S}_i \in \mathbb{R}^6$，描述该关节的旋转轴方向和位置。关节角变化 $\theta_i$ 对应 $SE(3)$ 上的指数映射：

$$e^{[\mathcal{S}_i] \theta_i} \in SE(3)$$

其中 $[\mathcal{S}_i]$ 是螺旋轴的 $4 \times 4$ 矩阵表示（在 [Lie 群章节](../../../../01-math/geometry/lie-groups.md) 里定义的 $\mathfrak{se}(3)$ 元素）。

末端位姿的空间型 PoE 公式：

$$\mathbf{T}_{0n}(\mathbf{q}) = e^{[\mathcal{S}_1]\theta_1} e^{[\mathcal{S}_2]\theta_2} \cdots e^{[\mathcal{S}_n]\theta_n} \cdot \mathbf{M}$$

其中 $\mathbf{M}$ 是所有关节角为零时末端的位姿（Home Configuration）。

PoE 的优点是，螺旋轴 $\mathcal{S}_i$ 在全局坐标系里定义，不需要像 DH 法那样为每个连杆手动选坐标系。对于数值计算和优化（如 IK），PoE 更方便。

!!! note "直觉小结"
    DH 法像"把手臂分成小块，逐段描述每块的变换"；PoE 法像"把每个关节的运动描述为空间里的一个螺旋，末端位姿是所有螺旋效果的叠加"。两种方法数学等价，只是建模角度不同。

## 代码

用 PoE 公式计算 2 自由度平面机械臂的末端位姿：

```python
import numpy as np

def exp_se3(S, theta):
    """SE(3) 的指数映射：S=[omega, v] 是螺旋轴，theta 是关节角"""
    omega = S[:3]
    v     = S[3:]
    # Rodrigues 公式计算旋转矩阵
    omega_hat = np.array([[0, -omega[2], omega[1]],
                          [omega[2], 0, -omega[0]],
                          [-omega[1], omega[0], 0]])
    R = (np.eye(3) + np.sin(theta) * omega_hat +
         (1 - np.cos(theta)) * omega_hat @ omega_hat)
    p = ((np.eye(3) - R) @ omega_hat @ v + np.outer(omega, omega) @ v * theta)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = p
    return T

# 2 自由度平面臂：两个关节都绕 z 轴，连杆长 L1=L2=1
L1, L2 = 1.0, 1.0
# 螺旋轴（空间型）：[omega; v]，omega=[0,0,1]，v = omega × r（r 是轴上一点）
S1 = np.array([0, 0, 1,  0,  0, 0])          # 关节 1 在原点
S2 = np.array([0, 0, 1, -L1,  0, 0])          # 关节 2 在 (L1,0,0)

# Home configuration：关节角=0 时末端在 (L1+L2, 0, 0)
M = np.eye(4); M[0, 3] = L1 + L2

def fk(q1, q2):
    T1 = exp_se3(S1, q1)
    T2 = exp_se3(S2, q2)
    return T1 @ T2 @ M

# 测试：两关节都转 pi/4
T = fk(np.pi/4, np.pi/4)
print(T[:3, 3].round(4))  # 末端位置
# 预期约 [0.7071, 1.7071, 0.]
```

## 开放问题

正运动学本身已经很成熟，但在实际机器人系统里的主要挑战在于**参数标定（Calibration）**：机器人出厂后，连杆长度、关节零点等参数和理论值会有偏差（加工公差、磨损），导致用标称参数计算的末端位姿与实际有几毫米到几厘米的误差。高精度装配（如电子元件焊接）对 FK 精度要求在 0.1mm 级别，标定是绕不开的工程问题。

下一节讲速度雅可比——从位置变到速度，讨论末端的速度如何由各关节速度合成。
