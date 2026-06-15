# 逆运动学：解析解与数值法

!!! info "参考资料"
    **主要教材**

    - [Modern Robotics](http://hades.mech.northwestern.edu/index.php/Modern_Robotics) — Lynch & Park, 第 6 章
    - [Introduction to Robotics](https://www.pearson.com/en-us/subject-catalog/p/introduction-to-robotics-mechanics-and-control/P200000003302) — Craig, 第 4 章

    **工具**

    - [IKFast (OpenRAVE)](http://openrave.org/docs/latest_stable/openravepy/ikfast/) — 解析 IK 自动生成器

!!! note "前置依赖"
    本节的数值法用到 [一阶最优化](../../../01-math/optimization/first-order.md) 里的 Newton-Raphson 迭代（即梯度法在方程求解上的应用），以及上一节的雅可比矩阵。

## 直觉 (Intuition)

逆运动学（Inverse Kinematics，IK）是正运动学的反问题：给定末端目标位姿 $\mathbf{T}_\text{target} \in SE(3)$，求满足条件的关节角 $\mathbf{q}$。看上去只是方程求反，实际上难得多——解可能不存在（目标超出工作空间），可能有无穷多个解（冗余机械臂），也可能有多个离散解（非线性方程组的多个根）。机械臂抓取物体、机器人行走落脚，所有需要"末端到达指定位置"的任务都要求解 IK。

## 解析解

对于特殊结构的机械臂（满足 Pieper 条件：三个相邻关节轴相交或平行），存在解析封闭解。

**Pieper 条件**（1968）：若机械臂的后三个关节轴交于一点（或平行），可以把 IK 解耦为：先用前三个关节确定手腕位置，再用后三个关节确定手腕姿态。大多数 6 自由度工业机器人（如 KUKA、UR、Fanuc）都满足这个条件，所以有解析解。

解析解的优点是快（微秒级），解的集合有限可枚举（通常 ≤ 8 组），实时控制可以直接用。缺点是只对特定机器人结构有效，结构稍微改变就要重新推导。

## 数值解：Newton-Raphson 迭代

对任意结构的机械臂，用迭代数值法求解。把 IK 写成方程求根问题：找 $\mathbf{q}^*$ 使得 $f(\mathbf{q}^*) = \mathbf{0}$，其中：

$$f(\mathbf{q}) = \text{Log}\!\left(\mathbf{T}_\text{target}^{-1} \cdot \mathbf{T}_{0n}(\mathbf{q})\right) \in \mathbb{R}^6$$

$\text{Log}$ 是 $SE(3)$ 的对数映射（见 [Lie 群章节](../../../01-math/geometry/lie-groups.md)），把位姿误差转化为 6 维向量（位置误差 + 角度误差）。

Newton-Raphson 迭代：

$$\mathbf{q}_{k+1} = \mathbf{q}_k - \mathbf{J}^{\dagger}(\mathbf{q}_k) f(\mathbf{q}_k)$$

其中 $\mathbf{J}^{\dagger}$ 是雅可比的伪逆（满足 $n < 6$ 的欠驱动用 $(\mathbf{J}^\top \mathbf{J})^{-1}\mathbf{J}^\top$，冗余机械臂 $n > 6$ 用 $\mathbf{J}^\top(\mathbf{J}\mathbf{J}^\top)^{-1}$）。

每次迭代一步，误差一般以二次速率收敛。通常 10-50 步就能收敛到 $\|f\| < 10^{-6}$ 的精度。

!!! note "直觉小结"
    把末端当前位姿和目标位姿的差（6 维误差向量），通过雅可比的伪逆"反算"成关节角修正量，反复迭代。每次迭代都把末端往目标推近一点，直到误差足够小。

## 冗余机械臂与零空间

当自由度 $n > 6$ 时（如 7 自由度的 Franka Panda、人形机器人手臂），IK 有无穷多组解——这叫**运动学冗余（Kinematic Redundancy）**。

冗余的额外自由度可以用来：

- 避障（让手臂绕开障碍物）
- 避关节限位（让关节角远离硬止点）
- 优化可操作性（让姿态尽量远离奇异点）
- 自运动（Null-Space Motion）：在末端位置不变的情况下，手臂整体"扭动"

数学上，冗余的解可以写成特解加零空间分量：

$$\dot{\mathbf{q}} = \mathbf{J}^{\dagger} \mathcal{V} + (\mathbf{I} - \mathbf{J}^{\dagger}\mathbf{J}) \mathbf{z}$$

其中 $(\mathbf{I} - \mathbf{J}^{\dagger}\mathbf{J})$ 是投影到雅可比零空间的投影矩阵，$\mathbf{z}$ 是任意向量，用来指定次要目标（如关节角远离限位）。

!!! tip "工程重点"
    数值 IK 的初始值选择很重要：从不好的初始值出发可能收敛到错误的解或者发散。实际应用中常用两个策略：（1）把上一帧的关节角作为初始值（连续轨迹跟踪时效果好）；（2）离线预计算一个初始值库（配置空间 atlas），在线查最近邻初始化。

## 代码

Newton-Raphson 数值 IK 的核心迭代：

```python
import numpy as np

def ik_newton(fk_func, jac_func, T_target, q_init,
              max_iter=100, tol=1e-6):
    """
    fk_func:  输入关节角 q，返回 4×4 SE(3) 矩阵
    jac_func: 输入关节角 q，返回 6×n 雅可比矩阵
    T_target: 目标末端位姿，4×4 矩阵
    q_init:   初始关节角（好的初始值决定能否收敛到正确解）
    """
    q = q_init.copy()
    for i in range(max_iter):
        T_cur = fk_func(q)
        # 位姿误差：用对数映射转为 6 维向量
        T_err = np.linalg.inv(T_cur) @ T_target
        # 从 SE(3) 矩阵提取 6D 误差向量（简化版 Log）
        pos_err = T_err[:3, 3]
        R_err   = T_err[:3, :3]
        angle   = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        if abs(angle) > 1e-10:
            omega = angle/(2*np.sin(angle)) * np.array(
                [R_err[2,1]-R_err[1,2], R_err[0,2]-R_err[2,0], R_err[1,0]-R_err[0,1]])
        else:
            omega = np.zeros(3)
        err = np.concatenate([omega, pos_err])

        if np.linalg.norm(err) < tol:
            print(f"IK 收敛，迭代 {i} 步")
            return q, True

        J      = jac_func(q)
        # 阻尼最小二乘伪逆，防止奇异
        lam    = 1e-4
        J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(J.shape[0]))
        q += J_pinv @ err              # Newton 步骤

    return q, False   # 未收敛
```

## 开放问题

传统数值 IK 的局限在速度：上面的迭代算法每次求解需要毫秒到数十毫秒，在实时控制（1kHz 以上）或运动规划（需要大量 IK 求解）场景下可能太慢。

近年来，**学习型 IK** 成为研究热点：用神经网络直接学习 $\mathbf{T}_\text{target} \to \mathbf{q}$ 的映射（如 IKFlow、NDIKNet），推理时只需要单次前向传播（亚毫秒级），代价是对工作空间边界和奇异点的处理不够稳健。两种方法的结合（学习提供热启动初始值，传统迭代精化）是目前的主流方向。

下一节讲轨迹规划：知道了每个时刻需要到达的位姿，如何生成平滑的运动轨迹？
