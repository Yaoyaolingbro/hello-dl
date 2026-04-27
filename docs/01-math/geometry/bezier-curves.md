# 贝塞尔曲线与 B 样条

!!! info "参考资料"
    **主要资料**
    - Farin, *Curves and Surfaces for CAGD*, 5th ed. — 第 4–6 章（贝塞尔与 B 样条的标准参考）
    - de Boor, *A Practical Guide to Splines* — B 样条的权威教材
    - Prautzsch, Boehm & Paluszny, *Bézier and B-Spline Techniques* — 免费 PDF

    **工具文档**
    - [SciPy: `BSpline`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html)

## 直觉 (Intuition)

贝塞尔曲线不穿过控制点（除了端点），而是被控制点"吸引"——拖动控制点就像用橡皮筋拉曲线，感觉直觉。B 样条把贝塞尔曲线拼接起来：每段局部受少数控制点影响，移动一个控制点只改变附近的曲线，而不影响整体。这个**局部控制**性质使 B 样条成为字体、CAD 设计、游戏动画轨迹的基础。机器人轨迹规划用样条保证关节运动的速度和加速度连续；多项式插值和三次样条的代数理论在后面优化理论章节的插值节会系统展开，两节可对照阅读。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbf{P}_i$ | 第 $i$ 个控制点 |
| $n$ | 贝塞尔曲线次数（控制点数 = $n+1$） |
| $B_{i,n}(t)$ | Bernstein 基多项式 |
| $t \in [0, 1]$ | 曲线参数 |
| $k$ | B 样条的次数（阶数 = $k+1$） |
| $\mathbf{u}$ | 节点向量（knot vector） |

## Bernstein 多项式

**Bernstein 基多项式（Bernstein basis polynomials）**：

$$
B_{i,n}(t) = \binom{n}{i} t^i (1-t)^{n-i}, \quad i = 0, 1, \ldots, n,\; t \in [0,1]
$$

关键性质：
- **非负性**：$B_{i,n}(t) \ge 0$（$t \in [0,1]$）
- **单位分解**：$\sum_{i=0}^n B_{i,n}(t) = 1$（权重求和为 1）
- **端点条件**：$B_{0,n}(0) = B_{n,n}(1) = 1$，其他为 0

## 贝塞尔曲线

**$n$ 次贝塞尔曲线（Bézier curve）**由 $n+1$ 个控制点 $\mathbf{P}_0, \ldots, \mathbf{P}_n$ 定义：

$$
\mathbf{C}(t) = \sum_{i=0}^n \mathbf{P}_i\, B_{i,n}(t), \quad t \in [0, 1]
$$

性质：
- 曲线从 $\mathbf{P}_0$（$t=0$）出发，到 $\mathbf{P}_n$（$t=1$）结束
- 曲线在控制多边形（convex hull）内——**凸包性质**
- 端点切线方向：$\mathbf{C}'(0) = n(\mathbf{P}_1 - \mathbf{P}_0)$，$\mathbf{C}'(1) = n(\mathbf{P}_n - \mathbf{P}_{n-1})$

## de Casteljau 算法

de Casteljau 算法用递推构造贝塞尔曲线，数值稳定且有几何直觉：

$$
\mathbf{P}_i^{(0)} = \mathbf{P}_i, \quad
\mathbf{P}_i^{(r)}(t) = (1-t)\mathbf{P}_i^{(r-1)}(t) + t\mathbf{P}_{i+1}^{(r-1)}(t)
$$

$\mathbf{C}(t) = \mathbf{P}_0^{(n)}(t)$。直觉：在每段控制线上取 $t$ 比例的点，对这些点再连线取 $t$ 比例，重复 $n$ 次。每次迭代降低一次多边形次数，最终收敛到曲线上一点。

de Casteljau 还提供了一个免费的副产品：把贝塞尔曲线在 $t_0$ 处**分裂**成两段贝塞尔曲线，只需读取递推的中间结果。

## B 样条

**B 样条（B-Spline）**是贝塞尔曲线的推广，允许任意多段控制点，每段只受 $k+1$ 个控制点影响（局部支撑）：

$$
\mathbf{C}(t) = \sum_{i=0}^n \mathbf{P}_i\, N_{i,k}(t)
$$

$N_{i,k}(t)$ 是 **B 样条基函数（de Boor 递推）**，由**节点向量** $\mathbf{u} = (u_0, u_1, \ldots, u_{n+k+1})$ 决定：

$$
N_{i,0}(t) = \begin{cases} 1 & u_i \le t < u_{i+1} \\ 0 & \text{otherwise} \end{cases}
$$

$$
N_{i,k}(t) = \frac{t - u_i}{u_{i+k} - u_i} N_{i,k-1}(t) + \frac{u_{i+k+1} - t}{u_{i+k+1} - u_{i+1}} N_{i+1,k-1}(t)
$$

均匀节点向量给出均匀 B 样条；非均匀节点向量（NURBS 的基础）可以精确表示圆弧和二次曲线。

!!! note "B 样条 vs. 贝塞尔的关键区别"
    - 贝塞尔：修改任一控制点影响整条曲线；B 样条：只影响 $k+1$ 段范围
    - 贝塞尔：$n+1$ 个控制点只能有一段曲线；B 样条：任意多控制点，保持低次数（通常 3 次）
    - NURBS（非均匀有理 B 样条）= 加权 B 样条，能精确表示圆弧，是工业 CAD（CATIA、AutoCAD）的核心

## 代码验证

```python
import numpy as np
from scipy.interpolate import BSpline

def de_casteljau(P, t):
    """用 de Casteljau 算法求贝塞尔曲线上 t 处的点"""
    P = np.array(P, dtype=float)
    n = len(P) - 1
    for r in range(1, n + 1):
        P[:n-r+1] = (1 - t) * P[:n-r+1] + t * P[1:n-r+2]
    return P[0]

# 三次贝塞尔曲线（4 个控制点）
control_pts = [(0,0), (1,2), (2,2), (3,0)]

# 采样曲线
t_vals = np.linspace(0, 1, 50)
curve_pts = np.array([de_casteljau(control_pts, t) for t in t_vals])

# 验证端点
pt_start = de_casteljau(control_pts, 0)
pt_end   = de_casteljau(control_pts, 1)
print(f"起点: {pt_start} (应为 {control_pts[0]})")   # (0,0)
print(f"终点: {pt_end}  (应为 {control_pts[-1]})")  # (3,0)

# 凸包性质：所有曲线点在控制点凸包内
P = np.array(control_pts)
x_min, x_max = P[:,0].min(), P[:,0].max()
y_min, y_max = P[:,1].min(), P[:,1].max()
in_hull = ((curve_pts[:,0] >= x_min - 1e-10) &
           (curve_pts[:,0] <= x_max + 1e-10) &
           (curve_pts[:,1] >= y_min - 1e-10) &
           (curve_pts[:,1] <= y_max + 1e-10))
print(f"所有点在凸包内: {in_hull.all()}")  # True（近似凸包）

# B 样条（scipy）
k = 3   # 三次
t_knot = np.array([0,0,0,0, 0.5, 1,1,1,1], dtype=float)  # 夹紧节点向量
c_x = np.array([0, 0.5, 1, 1.5, 2])   # x 坐标控制点
c_y = np.array([0, 1,   1, 0.5, 0])   # y 坐标控制点

bsp_x = BSpline(t_knot, c_x, k)
bsp_y = BSpline(t_knot, c_y, k)
t_eval = np.linspace(0, 1, 20)
print(f"\nB 样条端点: ({bsp_x(0):.2f},{bsp_y(0):.2f}) → ({bsp_x(1):.2f},{bsp_y(1):.2f})")
```

## 在深度学习中的应用

机器人轨迹规划用 B 样条表示关节角度随时间的轨迹，节点向量控制曲线的时间分配，优化控制点即可优化整条轨迹。TrueType/OpenType 字体用三次贝塞尔曲线描述字符轮廓，神经网络辅助字体生成（如 DeepSVG）直接学习控制点序列。运动扩散模型（Motion Diffusion Model）用 B 样条参数化人体关节轨迹，用更少参数表达更长序列的运动。

这是几何基础章节的最后一节。下一章讲概率论，从 Kolmogorov 公理出发，经过贝叶斯推断、信息论、随机过程，到变分推断与 ELBO——为生成模型和强化学习打好数学基础。
