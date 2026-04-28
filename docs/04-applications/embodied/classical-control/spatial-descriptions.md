# 空间描述：旋转表示与齐次变换

!!! info "参考资料"
    **主要教材**

    - [Modern Robotics: Mechanics, Planning, and Control](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Lynch & Park, Cambridge 2017（第 3 章）
    - [A Mathematical Introduction to Robotic Manipulation](http://www.cds.caltech.edu/~murray/mlswiki/) — Murray, Li & Sastry, 1994（免费电子版）

    **优质讲解**

    - [Stanford CS223A Lecture Notes](https://cs.stanford.edu/groups/manips/teaching/cs223a/)

!!! note "前置依赖"
    本节内容和 [几何变换与旋转表示](../../../../01-math/geometry/geometry-transforms.md) 高度重叠，数学推导部分见那里。本节从工程和应用视角出发，聚焦于：在机器人程序里该选哪种表示，以及各种表示的坑在哪里。

## 直觉 (Intuition)

机器人控制的第一个问题是：如何描述一个物体在空间中的位姿（Pose）？位姿 = 位置（Position，3 个数）+ 朝向（Orientation，至少 3 个数）。位置很简单，难点在朝向——旋转有无数种表示方式，每种都有适合和不适合的场景。输入是坐标系或物体的旋转状态，输出是一种便于计算的数学表示。就像描述一个人站在哪、面朝哪个方向，只不过在三维空间里，"面朝哪"就变得微妙了。

## 为什么旋转很麻烦

旋转是三维的，但 3 个角度（比如欧拉角）直接参数化会遇到**万向节死锁（Gimbal Lock）**：某些姿态下，其中一个自由度会"消失"，导致在那个位置附近无法平滑旋转。1967 年阿波罗 11 号的制导系统差点因为这个问题失去对飞船方向的控制。

根本原因是：三维旋转群 SO(3) 是一个 3 维流形，但不能用 3 个全局参数平滑覆盖（类似于地球表面不能用两个坐标无奇点地覆盖）。于是我们有了多种表示，各有取舍：

| 表示方法 | 参数数量 | 优点 | 缺点 |
|----------|---------|------|------|
| 旋转矩阵 $\mathbf{R} \in SO(3)$ | 9 个（但 3 个自由度） | 无奇点、直接作用于向量 | 冗余、需要维持正交约束 |
| 欧拉角 | 3 个 | 直观易理解 | 万向节死锁、定义约定多 |
| 轴角（Axis-Angle）| 4 个（归一化后 3 个）| 直觉清晰 | 不能直接插值 |
| 四元数 | 4 个（单位四元数） | 无奇点、插值 SLERP 稳定 | 双覆盖（$q$ 和 $-q$ 代表同一旋转） |
| 李代数 $\mathfrak{so}(3)$ | 3 个向量 | 适合优化、局部线性 | 全局操作需要 Exp/Log |

## 旋转矩阵

旋转矩阵 $\mathbf{R} \in \mathbb{R}^{3\times 3}$ 满足 $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$，$\det(\mathbf{R}) = 1$，即属于特殊正交群 $SO(3)$。

它的作用是对向量做旋转：$\mathbf{v}' = \mathbf{R} \mathbf{v}$。绕 $z$ 轴旋转 $\theta$ 角的矩阵是：

$$\mathbf{R}_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

两次旋转的组合是矩阵乘法：先旋转 $\mathbf{R}_1$ 再旋转 $\mathbf{R}_2$，结果是 $\mathbf{R}_2 \mathbf{R}_1$（注意顺序）。

!!! warning "常见误区"
    旋转矩阵乘法是**不可交换**的：$\mathbf{R}_1 \mathbf{R}_2 \neq \mathbf{R}_2 \mathbf{R}_1$。先绕 $x$ 转再绕 $z$ 转，和先绕 $z$ 转再绕 $x$ 转结果完全不同。拿一本书动手试一试，能建立直觉。

## 欧拉角与万向节死锁

欧拉角用三次绕坐标轴旋转来分解任意旋转。常见约定有 ZYX（RPY，Roll-Pitch-Yaw）和 ZYZ 等，**不同软件用不同约定，混用是大坑**。

ZYX 约定：先绕 $z$ 转 $\psi$（Yaw），再绕 $y'$ 转 $\theta$（Pitch），再绕 $x''$ 转 $\phi$（Roll）。

万向节死锁发生在 Pitch = ±90° 时：绕 $y'$ 转了 90° 后，$z$ 和 $x$ 轴重合，失去一个自由度。这不是编程 bug，是欧拉角参数化本身的数学奇点。

!!! tip "工程重点"
    ROS（Robot Operating System）默认用 ZYX（RPY）欧拉角，但 URDF 文件里的 `rpy` 属性是 XYZ 固定轴旋转——两个不同的约定，读参数时必须确认。万向节死锁在实际机器人运动控制里大多可以通过限制关节范围避开，但在姿态估计和传感器融合（比如无人机飞越极点）里会真实发生。

## 四元数

单位四元数 $\mathbf{q} = w + x\mathbf{i} + y\mathbf{j} + z\mathbf{k}$，$w^2+x^2+y^2+z^2=1$，表示绕单位轴 $\hat{\mathbf{n}}$ 旋转 $\theta$ 角：$\mathbf{q} = (\cos(\theta/2),\; \hat{\mathbf{n}} \sin(\theta/2))$。

四元数最重要的工程价值是插值：球面线性插值（SLERP）在两个旋转之间做最短路径插值，且速度匀速：

$$\text{SLERP}(\mathbf{q}_1, \mathbf{q}_2, t) = \mathbf{q}_1 (\mathbf{q}_1^{-1} \mathbf{q}_2)^t$$

旋转矩阵插值没有标准的"最短路径"方法，直接线性插值会导致矩阵不再是旋转矩阵。所以轨迹规划里朝向的插值几乎总是用四元数。

!!! warning "常见误区"
    $\mathbf{q}$ 和 $-\mathbf{q}$ 代表同一个旋转（双覆盖问题）。在实际系统里，如果连续追踪旋转，相邻两帧四元数可能符号相反（"翻转"），直接做差会得到错误的角速度。解决方法是在每步检查 $\mathbf{q}_t \cdot \mathbf{q}_{t-1} < 0$ 时对 $\mathbf{q}_t$ 取反。

## 齐次变换矩阵

把旋转和平移打包成一个 $4 \times 4$ 矩阵：

$$\mathbf{T} = \begin{pmatrix} \mathbf{R} & \mathbf{p} \\ \mathbf{0}^\top & 1 \end{pmatrix} \in SE(3)$$

其中 $\mathbf{p} \in \mathbb{R}^3$ 是平移向量。对点 $\mathbf{p}_\text{old}$ 做变换：把它写成齐次坐标 $[\mathbf{p}_\text{old}^\top, 1]^\top$，然后乘以 $\mathbf{T}$。

多个变换的组合是矩阵乘法，逆变换：

$$\mathbf{T}^{-1} = \begin{pmatrix} \mathbf{R}^\top & -\mathbf{R}^\top\mathbf{p} \\ \mathbf{0}^\top & 1 \end{pmatrix}$$

在机器人里，$\mathbf{T}_{AB}$ 表示"把 B 坐标系中的点变换到 A 坐标系"，多个关节的变换链就是矩阵连乘：

$$\mathbf{T}_{0E} = \mathbf{T}_{01} \cdot \mathbf{T}_{12} \cdots \mathbf{T}_{(n-1)E}$$

数学推导已在 [几何变换章节](../../../../01-math/geometry/geometry-transforms.md) 展开，以及 Lie 群视角见 [Lie 群与 Lie 代数](../../../../01-math/geometry/lie-groups.md)。

## 坐标系约定

机器人控制里常见的两种坐标系约定：

**世界坐标系（World Frame）**：固定在地面，所有物体的绝对位姿都在这里描述。

**本体坐标系（Body Frame）**：固定在机器人身上随机器人运动，描述传感器数据（相机、IMU）常用。

两者之间的变换用 $\mathbf{T}_{WB}$（从 Body 到 World）表示。弄混坐标系是机器人代码里最常见的 bug 来源之一——ROS 的 `tf` 系统就是专门用来管理坐标系变换树的。

!!! tip "工程重点"
    命名 $\mathbf{T}_{AB}$ 时，下标的方向含义在不同教材里有时相反（有的是"B 相对于 A"，有的是"从 A 到 B 的变换"）。强烈建议在代码注释里明确写出每个变量的含义，比如 `T_world_camera`（把相机坐标系里的点变换到世界坐标系）。

下一节讲正运动学：知道了怎么表示每个关节的变换，如何把它们串起来算出机械臂末端的位姿？
