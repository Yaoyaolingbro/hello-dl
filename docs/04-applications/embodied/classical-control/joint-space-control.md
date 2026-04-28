# 关节空间控制

!!! info "参考资料"
    **主要教材**

    - [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Lynch & Park, 第 11 章
    - [Robot Modeling and Control](https://www.wiley.com/en-us/Robot+Modeling+and+Control-p-9780471649908) — Spong, Hutchinson & Vidyasagar, 第 8 章

!!! note "前置依赖"
    本节的 PD 控制器稳定性分析用到 [凸函数与梯度下降](../../../../01-math/optimization/convex-basics.md) 里的 Lyapunov 稳定性直觉。计算力矩控制（CTC）依赖上一节的动力学方程。

## 直觉 (Intuition)

轨迹规划给出了期望的关节角序列 $\mathbf{q}_d(t)$，关节空间控制的任务是：设计力矩控制器，让实际关节角 $\mathbf{q}(t)$ 精确跟踪 $\mathbf{q}_d(t)$，即使存在扰动和模型误差。最朴素的思路是：偏差大就加大力矩；偏差在增大就更紧张。这就是 PD 控制的核心思路，也是工业机器人 90% 场景下的实际解决方案。

## PD 控制

定义跟踪误差 $\mathbf{e}(t) = \mathbf{q}_d(t) - \mathbf{q}(t)$，PD（比例-微分）控制器：

$$\boldsymbol{\tau} = \mathbf{K}_P \mathbf{e} + \mathbf{K}_D \dot{\mathbf{e}} + \mathbf{g}(\mathbf{q})$$

三项含义：

- $\mathbf{K}_P \mathbf{e}$：**比例项（P）**，偏差越大，力矩越大，往目标"拉"
- $\mathbf{K}_D \dot{\mathbf{e}}$：**微分项（D）**，误差变化越快，阻尼越大，防止超调振荡
- $\mathbf{g}(\mathbf{q})$：**重力补偿（Gravity Compensation）**，抵消重力，使控制器只需处理误差

其中 $\mathbf{K}_P, \mathbf{K}_D$ 是对角正定矩阵（每个关节独立增益）。

**稳定性**：可以用 Lyapunov 函数 $V = \frac{1}{2}\dot{\mathbf{q}}^\top \mathbf{M}(\mathbf{q})\dot{\mathbf{q}} + \frac{1}{2}\mathbf{e}^\top \mathbf{K}_P \mathbf{e}$（动能 + 弹性势能）证明，加重力补偿的 PD 控制在静止目标情况下是全局渐近稳定的。

**PD 控制的局限**：由于机器人动力学的非线性（$\mathbf{M}(\mathbf{q})$ 随姿态变化，科里奥利力随速度变化），固定增益的 PD 控制在快速运动时跟踪精度会下降——高速时惯性项和科里奥利力不能被忽略，而 PD 控制没有补偿它们。

!!! tip "工程重点"
    调节 PD 增益的经验规则：先调 $K_P$，让系统响应足够快但不振荡；再调 $K_D$，用阻尼消除振荡。实际关节控制器（如 Dynamixel 舵机、工业伺服驱动器）通常开放 P、D、I 三个增益，以及前馈速度和加速度增益。纯 PD 加重力补偿在低速精密操作（装配、精细操作）中完全够用，高速运动时才需要 CTC。

## 计算力矩控制（CTC）

**计算力矩控制（Computed Torque Control，CTC）**，也叫逆动力学控制，思路是：如果已知精确的动力学模型，可以用它"消掉"非线性，把机器人变成线性系统，然后设计简单的线性控制器。

设计步骤：

**第一步**：引入辅助输入 $\mathbf{a}$（叫做"外环控制输入"），然后用逆动力学计算所需力矩：

$$\boldsymbol{\tau} = \hat{\mathbf{M}}(\mathbf{q})\mathbf{a} + \hat{\mathbf{C}}(\mathbf{q},\dot{\mathbf{q}})\dot{\mathbf{q}} + \hat{\mathbf{g}}(\mathbf{q})$$

其中 $\hat{\mathbf{M}}, \hat{\mathbf{C}}, \hat{\mathbf{g}}$ 是动力学模型的估计值。

**第二步**：代入真实动力学方程（$\mathbf{M}\ddot{\mathbf{q}} + \mathbf{C}\dot{\mathbf{q}} + \mathbf{g} = \boldsymbol{\tau}$），若模型精确（$\hat{\mathbf{M}} = \mathbf{M}$），可以得到线性化后的双积分系统：

$$\ddot{\mathbf{q}} = \mathbf{a}$$

**第三步**：在线性系统上设计简单的 PD 外环控制器：

$$\mathbf{a} = \ddot{\mathbf{q}}_d + \mathbf{K}_D(\dot{\mathbf{q}}_d - \dot{\mathbf{q}}) + \mathbf{K}_P(\mathbf{q}_d - \mathbf{q})$$

可以证明，此时跟踪误差满足线性 ODE $\ddot{\mathbf{e}} + \mathbf{K}_D \dot{\mathbf{e}} + \mathbf{K}_P \mathbf{e} = 0$，只要 $\mathbf{K}_P, \mathbf{K}_D$ 正定，误差指数收敛。

!!! note "直觉小结"
    CTC 的核心思路：先用模型"预测"需要多少力矩来产生期望加速度（前馈，feedforward），再用 PD 处理预测误差（反馈，feedback）。前馈补偿了非线性，反馈处理残差——两者分工明确。

## 两种控制器的对比

| | PD + 重力补偿 | 计算力矩控制 (CTC) |
|--|--|--|
| 计算量 | 低（只需要 $\mathbf{g}(\mathbf{q})$） | 高（需要完整逆动力学）|
| 模型依赖 | 只需要重力模型 | 需要精确的完整动力学模型 |
| 跟踪精度 | 低速高精，高速下降 | 全速度范围高精度 |
| 鲁棒性 | 对模型误差鲁棒 | 模型误差直接影响精度 |
| 典型应用 | 工业装配、低速操作 | 高速精密轨迹跟踪 |

## 代码

CTC 控制器实现（仿真中的一步更新）：

```python
import numpy as np

def ctc_control(q, dq, q_d, dq_d, ddq_d, M_func, C_func, g_func,
                Kp, Kd):
    """
    CTC（计算力矩控制）的单步力矩计算
    M_func, C_func, g_func: 动力学模型函数
    返回关节力矩 tau
    """
    e   = q_d - q        # 位置误差
    de  = dq_d - dq      # 速度误差

    # 外环线性 PD：在线性化系统上设计
    a = ddq_d + Kd @ de + Kp @ e

    # 内环逆动力学：用模型把 a 转换成力矩
    M = M_func(q)                    # 惯性矩阵
    C = C_func(q, dq)                # 科里奥利矩阵
    g = g_func(q)                    # 重力项
    tau = M @ a + C @ dq + g        # 前馈 + 内环线性化

    return tau
```

## 开放问题

CTC 的核心假设是动力学模型精确，但实际机器人总有模型误差：连接电缆的质量、关节摩擦力（库仑摩擦、粘性摩擦）、连杆弹性等都不在模型里。

**自适应控制（Adaptive Control）**和**鲁棒控制（Robust Control）**是两条解决路线：前者在线估计动力学参数，后者设计对有界误差鲁棒的控制律（如滑模控制）。两种方法在现代机器人系统里都有应用，但相对传统 PD + 重力补偿的额外工程复杂度是否值得，需要根据任务精度要求判断。

下一节讲操作空间控制：控制目标从"关节角跟踪"变成"末端力/位置控制"，是接触操作和人机协作的基础。
