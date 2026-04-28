# 插值与样条基础

!!! info "参考资料"
    **主要资料**

    - de Boor, *A Practical Guide to Splines* — 样条的经典教材
    - [Numerical Recipes](http://numerical.recipes/) — 第 3 章，插值的实用实现
    - Farin, *Curves and Surfaces for CAGD* — 贝塞尔与 B 样条的图形学视角

    **工具文档**
    
    - [SciPy: `scipy.interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html)

## 直觉 (Intuition)

插值是"用已知点猜未知点"：给定一组数据点，找一条光滑曲线穿过它们，并能在中间取值。多项式插值最简单，但高次多项式在边界附近会剧烈震荡（Runge 现象）。分段多项式（样条）把曲线拆成若干段，每段低次多项式，接头处保证光滑连续——这样既灵活又稳定。机器人轨迹规划需要插值关节角度，生成时间上连续可微的运动曲线；神经网络的位置编码也涉及插值思想。

## 主要符号

| 符号 | 含义 |
|------|------|
| $(x_i, y_i)$，$i = 0, \ldots, n$ | 插值节点（数据点） |
| $p(x)$ | 插值多项式 |
| $L_i(x)$ | 第 $i$ 个 Lagrange 基函数 |
| $h_i = x_{i+1} - x_i$ | 相邻节点的间距 |

## 多项式插值

给定 $n+1$ 个数据点 $(x_0, y_0), \ldots, (x_n, y_n)$（$x_i$ 两两不同），**插值问题**是找一个次数 $\le n$ 的多项式 $p(x)$ 使得 $p(x_i) = y_i$。

**Lagrange 插值**给出显式公式：

$$
p(x) = \sum_{i=0}^n y_i\, L_i(x), \quad L_i(x) = \prod_{j \ne i} \frac{x - x_j}{x_i - x_j}
$$

$L_i(x_j) = \delta_{ij}$，每个基函数在对应节点处为 1，其余节点处为 0。

**Runge 现象**：在等距节点上用高次多项式插值，边界处误差会指数增大。解决方案是改用 Chebyshev 节点（集中在端点附近）或改用分段多项式（样条）。

## 三次样条

**三次样条（cubic spline）**是实践中最常用的插值方法：在每个子区间 $[x_i, x_{i+1}]$ 上用一段三次多项式，接头处满足：

- $C^0$：位置连续（值相等）
- $C^1$：一阶导数连续（速度连续）
- $C^2$：二阶导数连续（加速度连续）

$n+1$ 个节点，$n$ 段，每段 4 个系数共 $4n$ 个未知量，$C^0/C^1/C^2$ 条件给出 $4n-2$ 个方程，加上两端的**边界条件**（通常：端点二阶导为零，称"自然样条"；或指定端点一阶导）补全方程组。结果是一个稀疏三对角线性方程组，$O(n)$ 时间求解。

!!! note "三次样条 vs. 贝塞尔曲线"
    三次样条过所有数据点（插值），适合从离散数据恢复连续曲线。贝塞尔曲线由控制点"吸引"，通常不过控制点（逼近），适合设计场景（用户拖动控制点调整形状）。

## B 样条基础

**B 样条（B-Spline）**是样条的更一般形式，用一组局部支撑的基函数 $B_{i,k}(t)$ 表示曲线：

$$
\mathbf{C}(t) = \sum_{i=0}^n \mathbf{P}_i\, B_{i,k}(t)
$$

$k$ 是次数，$\mathbf{P}_i$ 是控制点，$B_{i,k}$ 由 de Boor 递推公式定义（每个基函数只在 $k+1$ 个节点区间上非零）。B 样条的**局部支撑性**使得移动一个控制点只影响局部曲线形状，而不影响整体，这是设计工具（CAD、字体）的核心需求。几何章节的贝塞尔曲线节已详细展开 B 样条的几何构造（de Boor 递推与控制点直觉），本节侧重插值视角的代数基础。

## 代码验证

```python
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')

# 插值带噪声的正弦数据
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 8)    # 8 个插值节点
y = np.sin(x) + 0.05 * np.random.randn(8)  # 带噪声的观测

# 三次样条插值
cs = CubicSpline(x, y, bc_type='natural')   # 自然样条

# 在密集网格上评估
x_fine = np.linspace(0, 2 * np.pi, 200)
y_fine = cs(x_fine)

# 验证：在节点处精确插值
print("节点处插值误差:", np.max(np.abs(cs(x) - y)))  # 应为 0（机器精度）

# 验证 C^2 连续性：检查二阶导数在内部节点的左右极限
y2_left  = cs(x[1:-1], 2)   # 从左侧计算二阶导数
y2_right = cs(x[1:-1], 2)   # 相同（连续）
print("C^2 连续性 (内部节点二阶导跳跃):", np.max(np.abs(y2_left - y2_right)))  # 接近 0

# 一阶导数（速度）和二阶导数（加速度）
v_at_nodes = cs(x, 1)    # 一阶导
a_at_nodes = cs(x, 2)    # 二阶导
print(f"端点速度: [{v_at_nodes[0]:.3f}, {v_at_nodes[-1]:.3f}]")  # 自然样条不约束端点速度
print(f"端点加速度: [{a_at_nodes[0]:.4f}, {a_at_nodes[-1]:.4f}]")  # 自然样条端点加速度 = 0
```

## 在深度学习中的应用

机器人轨迹规划用三次样条或五次多项式（满足位置/速度/加速度初末边界条件）生成关节空间轨迹，保证运动连续可微。神经辐射场（NeRF）和 3D Gaussian Splatting 里的位置编码用 Fourier 基函数插值空间频率。Flow Matching 模型的插值路径设计（从噪声到数据的流）本质上是概率意义上的轨迹插值问题，直接引用了样条插值思想。

下一节讲 ODE 与向量场，把从离散时刻的更新推广到连续时间动力系统，是 Neural ODE 和扩散模型的数学基础。
