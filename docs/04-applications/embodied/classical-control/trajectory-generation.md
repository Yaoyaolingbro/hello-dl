# 轨迹规划

!!! info "参考资料"
    **主要教材**

    - [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Lynch & Park, 第 9 章
    - [Introduction to Robotics](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003302) — Craig, 第 7 章

!!! note "前置依赖"
    本节用到 [插值与样条基础](../../../01-math/optimization/interpolation.md)（三次样条、B 样条）以及 [贝塞尔曲线与 B 样条](../../../01-math/geometry/bezier-curves.md)（曲线参数化），建议先阅读。

## 直觉 (Intuition)

逆运动学给出了"目标位姿 → 关节角"的求解方法，但机器人运动不是瞬间跳到目标，而是要沿着一条平滑的路径连续运动。轨迹规划（Trajectory Generation）就是在起点和终点（或一系列路径点）之间，生成满足速度、加速度约束的平滑运动序列。输入是路径点集合和时间约束，输出是关节角（或末端位置）随时间的函数 $\mathbf{q}(t)$。轨迹规划直接决定运动的流畅程度和机械磨损，是工业机器人和服务机器人里最基础的工程模块之一。

## 关节空间 vs 笛卡尔空间

轨迹规划可以在两个空间里做，各有优缺点：

**关节空间（Joint Space）规划**：对每个关节角分别插值，生成 $q_1(t), q_2(t), \ldots, q_n(t)$。

- 优点：速度、加速度限制直接加在关节上，控制器执行简单；不存在奇异点问题
- 缺点：末端的实际运动路径是曲线，不直观；末端走的路径不是直线，穿过障碍物的风险更高

**笛卡尔空间（Cartesian Space）规划**：直接规划末端的位置和姿态轨迹 $\mathbf{p}(t), \mathbf{R}(t)$，再逐点求 IK 转换为关节角。

- 优点：末端走直线或指定曲线，路径可预见；方便约束末端姿态（如握着杯子不能倾斜）
- 缺点：在奇异点附近关节速度可能突变；每个时刻都需要 IK，计算量更大

工业焊接（必须沿焊缝走直线）用笛卡尔规划；Pick-and-Place（点到点，路径不重要）用关节空间规划。

## 时间参数化

路径（Path）是几何形状（一条曲线），轨迹（Trajectory）是路径加上时间参数——什么时刻在哪里。轨迹规划的核心是**时间参数化（Time Scaling）**：给定一条路径 $\mathbf{q}(s)$，$s \in [0,1]$，选择时间函数 $s(t)$ 使得运动满足速度和加速度限制。

**梯形速度曲线（Trapezoidal Velocity Profile）**：最常用的简单方案，三段：

- 匀加速阶段：速度从 0 线性增大
- 匀速阶段：以最大速度 $v_\max$ 匀速运动
- 匀减速阶段：速度线性减小至 0

对应的位移曲线是：加速段抛物线 + 匀速段直线 + 减速段抛物线（也叫 LSPB：Linear Segments with Parabolic Blends）。

**三次多项式（Cubic Polynomial）**：给定起止位置和速度，4 个约束确定三次多项式 $q(t) = a_3 t^3 + a_2 t^2 + a_1 t + a_0$。位置和速度连续，加速度在端点可能不为零（会有加速度突变，机械冲击大）。

**五次多项式（Quintic Polynomial）**：6 个约束（起止位置、速度、加速度），保证加速度连续，减少振动。

**三次样条（Cubic Spline）**：多段三次多项式拼接，满足 $C^0, C^1, C^2$ 连续（位置、速度、加速度全连续）。适合多路径点插值，是 CNC 数控机床和机械臂的标准轨迹格式之一。样条的数学背景已在 [插值章节](../../../01-math/optimization/interpolation.md) 详细推导。

!!! note "直觉小结"
    梯形曲线简单但有加速度突变（对电机有冲击）；三次多项式加速度连续但可能超出限制；五次多项式更平滑。实际工程里五次多项式和三次样条是最常见的选择。

## 姿态的插值

位置插值直接用向量插值，姿态插值要更小心。旋转矩阵直接线性插值不能保持正交性，正确的做法：

- **SLERP（Spherical Linear Interpolation）**：对四元数做球面插值，保证最短弧长路径，速度匀速：

$$\mathbf{q}(t) = \text{SLERP}(\mathbf{q}_0, \mathbf{q}_1, s(t))$$

- **Lie 群插值（Geodesic）**：在 $SO(3)$ 上用测地线插值，等价于用 Log 算出旋转向量，线性插值后再 Exp 回去。这是与 PoE/IK 框架最一致的方案，在连续轨迹优化里更常用。

!!! tip "工程重点"
    SLERP 假设角速度在插值段内匀速，适合点到点运动。连接多段旋转时，相邻段之间的角加速度会突变。更平滑的方案是 Spherical Cubic Spline（在 $SO(3)$ 上的 Catmull-Rom 样条），但实现复杂，大多数工业软件不支持。实际上很多机械臂控制器允许用户只指定关键帧的位置和朝向，姿态插值由控制器内部处理（往往就是 SLERP）。

## 代码

三次样条轨迹规划（多路径点关节角插值）：

```python
import numpy as np
from scipy.interpolate import CubicSpline

def joint_trajectory(waypoints, times):
    """
    waypoints: (M, n) 数组，M 个路径点，每点 n 个关节角
    times:     长度 M 的时间数组，单位秒
    返回轨迹函数，输入时间 t 返回关节角和速度
    """
    # scipy CubicSpline 自动处理 C2 连续性
    cs = CubicSpline(times, waypoints, bc_type='clamped')  # 边界速度=0
    def traj(t):
        q    = cs(t)          # 关节角
        dq   = cs(t, 1)       # 关节角速度（1 阶导数）
        return q, dq
    return traj

# 示例：3 关节臂，3 个路径点
waypoints = np.array([[0., 0., 0.],
                      [1., 0.5, -0.3],
                      [0., 1., 0.]])
times     = np.array([0., 1., 2.])
traj      = joint_trajectory(waypoints, times)

t_test    = 0.5
q, dq     = traj(t_test)
print(f"t={t_test:.1f}s: q={q.round(4)}, dq={dq.round(4)}")
# 在起点和第一个路径点之间的中间位置
```

## 开放问题

传统轨迹规划假设路径已知（通过离线运动规划预先算好），适合结构化环境（工厂流水线）。在动态环境（人和机器人共处、抓取移动物体）里，需要**实时重规划（Online Replanning）**：环境变化时能在几毫秒内生成新轨迹。

现代方法是把轨迹规划和运动规划结合：Motion Planning 部分（避障路径搜索，如 RRT*、CHOMP）负责全局路径，Trajectory Generation 部分（本节内容）负责把路径变成平滑可执行的关节命令。深度学习方法（如 Diffusion Policy）最近开始绕开这个分离的两阶段框架，直接用模仿学习端到端生成轨迹，但在精度和安全性约束上还没有传统方法成熟。

下一节讲动力学——不只是"运动到哪"，而是"需要多大的力/力矩才能实现这个运动"。
