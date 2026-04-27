# 叉积与刚体运动基础

!!! info "参考资料"
    **主要资料**
    - Murray, Li & Sastry, *A Mathematical Introduction to Robotic Manipulation* — 免费 PDF，第 2 章
    - [Deep Learning Book: Appendix](https://www.deeplearningbook.org/contents/linear_algebra.html) — 向量运算部分
    - Lynch & Park, *Modern Robotics* — 第 3 章，刚体运动的现代处理

## 直觉 (Intuition)

向量的叉积给出了"两个向量张成的平面的法向量"，模长是平行四边形面积。在物理上，叉积无处不在：力矩 = 力臂叉积力，角速度叉积位置向量 = 线速度，磁场中的洛伦兹力。刚体运动由平移和旋转组成，旋转不能简单加减（两个旋转之和未必还是旋转），这正是 Lie 群框架的出发点。理解叉积和角速度，是后续 Lie 群、位姿估计、机器人运动学的基础。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbf{a} \times \mathbf{b}$ | 向量叉积 |
| $[\mathbf{a}]_\times \in \mathbb{R}^{3\times 3}$ | $\mathbf{a}$ 的反对称矩阵（叉积矩阵） |
| $\boldsymbol{\omega}$ | 角速度向量（单位：rad/s） |
| $\mathbf{R} \in SO(3)$ | 旋转矩阵 |
| $\mathbf{p}$ | 位置向量 |

## 叉积

三维向量 $\mathbf{a} = (a_1, a_2, a_3)^\top$ 和 $\mathbf{b} = (b_1, b_2, b_3)^\top$ 的**叉积（cross product）**：

$$
\mathbf{a} \times \mathbf{b}
=
\begin{vmatrix}
\mathbf{e}_1 & \mathbf{e}_2 & \mathbf{e}_3 \\
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3
\end{vmatrix}
=
\begin{pmatrix}
a_2 b_3 - a_3 b_2 \\
a_3 b_1 - a_1 b_3 \\
a_1 b_2 - a_2 b_1
\end{pmatrix}
$$

关键性质：
- $\|\mathbf{a} \times \mathbf{b}\| = \|\mathbf{a}\|\|\mathbf{b}\|\sin\theta$（$\theta$ 是两向量夹角，模长 = 平行四边形面积）
- 方向由右手定则决定，垂直于 $\mathbf{a}$ 和 $\mathbf{b}$ 所张平面
- **反对称**：$\mathbf{a} \times \mathbf{b} = -\mathbf{b} \times \mathbf{a}$
- $\mathbf{a} \times \mathbf{a} = \mathbf{0}$（平行向量叉积为零）

## 叉积矩阵（反对称矩阵表示）

叉积 $\mathbf{a} \times \mathbf{b}$ 可以写成矩阵乘法：$\mathbf{a} \times \mathbf{b} = [\mathbf{a}]_\times \mathbf{b}$，其中：

$$
[\mathbf{a}]_\times
=
\begin{pmatrix}
0 & -a_3 & a_2 \\
a_3 & 0 & -a_1 \\
-a_2 & a_1 & 0
\end{pmatrix}
$$

$[\mathbf{a}]_\times$ 是**反对称矩阵**（$[\mathbf{a}]_\times^\top = -[\mathbf{a}]_\times$）。这个表示在 Lie 代数里至关重要——$\mathfrak{so}(3)$（$SO(3)$ 的 Lie 代数）恰好是全体 $3\times 3$ 反对称矩阵。

## 刚体运动：平移 + 旋转

**刚体（rigid body）**的运动保持任意两点的距离不变，可以完全由一个**旋转矩阵** $\mathbf{R} \in SO(3)$ 和一个**平移向量** $\mathbf{p} \in \mathbb{R}^3$ 描述：

$$
SO(3) = \{\mathbf{R} \in \mathbb{R}^{3\times 3} \mid \mathbf{R}^\top \mathbf{R} = \mathbf{I},\, \det(\mathbf{R}) = 1\}
$$

旋转矩阵保距离（等距变换），且保手性（行列式 +1）。$\mathbf{R}^\top = \mathbf{R}^{-1}$——旋转的逆等于转置。

## 角速度与速度

刚体旋转时，物体上一点 $\mathbf{r}$（相对旋转中心）的速度：

$$
\mathbf{v} = \boldsymbol{\omega} \times \mathbf{r} = [\boldsymbol{\omega}]_\times \mathbf{r}
$$

其中 $\boldsymbol{\omega}$ 是**角速度向量**（方向是旋转轴，模长是角速度大小）。旋转矩阵的时间导数：

$$
\dot{\mathbf{R}} = [\boldsymbol{\omega}]_\times \mathbf{R}
$$

这个方程把"旋转矩阵如何随时间变化"与叉积矩阵联系起来，是推导指数映射 $\mathbf{R}(t) = e^{[\boldsymbol{\omega}]_\times t}$ 的出发点。

!!! note "力矩与叉积"
    力矩 $\boldsymbol{\tau} = \mathbf{r} \times \mathbf{F}$：力 $\mathbf{F}$ 对距转轴 $\mathbf{r}$ 处的力矩，大小等于力臂长乘力的垂直分量。机器人操作臂的雅可比矩阵把关节速度映射到末端速度，其旋转部分正是 $\boldsymbol{\omega} = \mathbf{J}_\omega \dot{\mathbf{q}}$，利用了叉积的线性性。

## 代码验证

```python
import numpy as np

# 叉积验证
a = np.array([1.0, 0.0, 0.0])  # x 轴单位向量
b = np.array([0.0, 1.0, 0.0])  # y 轴单位向量
cross = np.cross(a, b)
print(f"x × y = {cross}")  # [0, 0, 1]（z 轴方向，右手定则）

# 叉积矩阵
def skew(v):
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ]
    ])

a_cross = skew(a)
print(f"\n[a]× @ b = {a_cross @ b}")  # 应等于 a × b = [0, 0, 1]
print(f"反对称性验证: {np.allclose(a_cross + a_cross.T, np.zeros((3,3)))}")  # True

# 旋转矩阵验证：绕 z 轴转 90 度
theta = np.pi / 2
Rz = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta),  np.cos(theta), 0],
    [0,             0,             1]
])
print(f"\nR @ R.T ≈ I: {np.allclose(Rz @ Rz.T, np.eye(3))}")  # True
print(f"det(R) = {np.linalg.det(Rz):.4f}")  # 1.0

# 旋转 x 轴单位向量应得到 y 轴方向
rotated = Rz @ a
print(f"R_z(90°) · [1,0,0] = {rotated.round(4)}")  # [0, 1, 0]

# 角速度到速度：v = ω × r
omega = np.array([0.0, 0.0, 1.0])  # 绕 z 轴旋转，角速度 = 1 rad/s
r = np.array([1.0, 0.0, 0.0])      # 点在 x 轴上距原点 1
v = np.cross(omega, r)
print(f"\n角速度 ω={omega}, 位置 r={r}, 速度 v={v}")  # v=[0, 1, 0]（切向速度）
```

## 在深度学习中的应用

叉积和反对称矩阵是 Lie 代数 $\mathfrak{so}(3)$ 的基础——下一节 Lie 群节会直接用到这里的 $[\cdot]_\times$ 表示。机器人雅可比矩阵（关节速度→末端速度）的推导需要叉积。等变神经网络（E(3)-Equivariant GNN，如 SEGNN、DiffSBDD）在分子和晶体表示中利用 SO(3) 的等变性，需要理解旋转群的代数结构。

下一节讲几何变换与相机模型，引入齐次坐标统一表示平移和旋转，推导 SO(3)/SE(3) 的参数化方式，以及计算机视觉里的投影变换。
