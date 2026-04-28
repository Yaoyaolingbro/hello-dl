# 速度雅可比

!!! info "参考资料"
    **主要教材**

    - [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Lynch & Park, 第 5 章
    - [Introduction to Robotics](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003302) — Craig, 第 5 章

!!! note "前置依赖"
    雅可比矩阵是多变量函数的导数矩阵，见 [矩阵微积分](../../../../01-math/linear-algebra/matrix-calculus.md)。奇异值分析见 [SVD](../../../../01-math/linear-algebra/svd.md)。

## 直觉 (Intuition)

正运动学给出"关节角度 → 末端位姿"的映射。速度雅可比回答速度版本的问题：如果各关节以速度 $\dot{\mathbf{q}}$ 运动，末端的线速度和角速度是多少？想象用手臂写字：每一笔的速度方向，是肩膀、肘、腕各关节速度的合成，雅可比就是这个合成关系的精确数学描述。输入是关节速度 $\dot{\mathbf{q}} \in \mathbb{R}^n$，输出是末端的六维速度（角速度 + 线速度）。它是逆运动学、轨迹控制和力控制的共同基础。

## 雅可比矩阵

末端速度用 6 维向量 $\mathcal{V} = [\boldsymbol{\omega}^\top, \mathbf{v}^\top]^\top$ 表示（角速度在前，线速度在后），雅可比矩阵 $\mathbf{J}(\mathbf{q}) \in \mathbb{R}^{6 \times n}$ 建立关节速度和末端速度之间的线性映射：

$$\mathcal{V} = \mathbf{J}(\mathbf{q}) \dot{\mathbf{q}}$$

雅可比依赖当前关节角 $\mathbf{q}$：同样的关节速度，在不同姿态下产生的末端速度不同。

用 PoE 公式，空间型雅可比的第 $i$ 列（第 $i$ 个关节对末端速度的贡献）为：

$$\mathbf{J}_i(\mathbf{q}) = \text{Ad}_{e^{[\mathcal{S}_1]\theta_1}\cdots e^{[\mathcal{S}_{i-1}]\theta_{i-1}}}(\mathcal{S}_i)$$

其中 $\text{Ad}$ 是伴随表示，把螺旋轴从局部坐标变换到全局坐标。直觉上：第 $i$ 个关节的贡献，是它的螺旋轴经过前 $i-1$ 个关节变换后的等效轴。

## 奇异性

当 $\mathbf{J}(\mathbf{q})$ 不满秩时，某些末端运动方向用关节速度无法实现，这叫**运动学奇异点（Kinematic Singularity）**。

典型奇异姿态：机械臂完全伸直（所有关节共线）或两个旋转轴对齐。此时维持末端速度需要无穷大的关节速度——实际电机做不到，控制器会产生异常大的力矩指令。

奇异性分析方法：对 $\mathbf{J}$ 做 SVD（见 [SVD 章节](../../../../01-math/linear-algebra/svd.md)），最小奇异值 $\sigma_\min \to 0$ 意味着接近奇异点。

**可操作性（Manipulability）**指标（Yoshikawa 1985）：

$$w = \sqrt{\det(\mathbf{J}\mathbf{J}^\top)}$$

$w = 0$ 表示奇异，$w$ 越大末端在各方向运动越"自由"。轨迹规划时可以把最大化 $w$ 作为约束，主动绕开奇异区域。

!!! tip "工程重点"
    工业机器人末端走直线时，在接近奇异点处会剧烈抖动——因为需要极大关节速度来维持末端速度。解决方案是使用**阻尼最小二乘（Damped Least Squares，DLS）伪逆**：

    $$\mathbf{J}^{\dagger}_\lambda = \mathbf{J}^\top(\mathbf{J}\mathbf{J}^\top + \lambda^2 \mathbf{I})^{-1}$$

    $\lambda > 0$ 给奇异值加了下限，用少量末端精度损失换取控制稳定性。$\lambda$ 一般在接近奇异时动态增大（可操作性自适应 DLS）。

## 静力学对偶

通过虚功原理，末端六维力/力矩向量 $\mathcal{F}$ 和关节力矩 $\boldsymbol{\tau}$ 的关系是雅可比转置：

$$\boldsymbol{\tau} = \mathbf{J}^\top(\mathbf{q}) \mathcal{F}$$

**静力学对偶（Static Duality）**：速度雅可比把关节速度映射到末端速度，力雅可比（$\mathbf{J}^\top$）把末端力映射到关节力矩。力控制依赖这个关系——让末端施加 5N 的力，只需要通过 $\mathbf{J}^\top$ 换算成关节力矩指令。

## 代码

数值雅可比（有限差分，不依赖解析 PoE，实际工程常用）：

```python
import numpy as np

def numerical_jacobian(fk_func, q, eps=1e-5):
    """
    fk_func: 输入关节角数组，返回 4×4 SE(3) 变换矩阵
    返回 6×n 雅可比（前 3 行角速度，后 3 行线速度）
    """
    n  = len(q)
    T0 = fk_func(q)
    J  = np.zeros((6, n))
    for i in range(n):
        dq    = np.zeros(n)
        dq[i] = eps
        T1    = fk_func(q + dq)
        dp = (T1[:3, 3] - T0[:3, 3]) / eps           # 线速度分量
        dR = (T1[:3, :3] - T0[:3, :3]) / eps
        omega_hat = dR @ T0[:3, :3].T                 # 近似 [omega]×
        omega = np.array([omega_hat[2,1], omega_hat[0,2], omega_hat[1,0]])
        J[:3, i] = omega
        J[3:, i] = dp
    return J
```

下一节讲逆运动学：已知末端目标位姿，如何求各关节角？
