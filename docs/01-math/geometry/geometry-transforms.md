# 几何变换：齐次坐标、旋转矩阵与相机模型

!!! info "参考资料"
    **主要资料**
    
    - Hartley & Zisserman, *Multiple View Geometry in Computer Vision*, 2nd ed. — 第 1–2 章（齐次坐标与投影变换的标准参考）
    - Lynch & Park, *Modern Robotics* — 第 3 章
    - Szeliski, *Computer Vision: Algorithms and Applications* — 第 2 章，相机模型

## 直觉 (Intuition)

平移不能用普通矩阵乘法表示（矩阵乘法固定原点），旋转可以。**齐次坐标**给每个点加一个"1"变成 $n+1$ 维向量，把平移和旋转统一成一个矩阵乘法，这是计算机图形学和机器人学里处理几何变换的标准工具。相机把三维世界投影到二维图像，这个过程也是矩阵乘法——理解投影矩阵是 NeRF、3DGS、立体视觉等所有 3D 视觉任务的先决条件。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\tilde{\mathbf{p}} = (\mathbf{p}^\top, 1)^\top$ | 点 $\mathbf{p}$ 的齐次坐标表示 |
| $\mathbf{T} \in SE(3)$ | 刚体变换矩阵（$4\times 4$） |
| $\mathbf{K}$ | 相机内参矩阵（$3\times 3$） |
| $[\mathbf{R} \mid \mathbf{t}]$ | 相机外参矩阵（$3\times 4$） |
| $\mathbf{q} \in \mathbb{H}$ | 单位四元数（表示旋转） |

## 齐次坐标

$\mathbb{R}^3$ 中的点 $\mathbf{p} = (x, y, z)^\top$ 的**齐次坐标**为 $\tilde{\mathbf{p}} = (x, y, z, 1)^\top \in \mathbb{R}^4$。

刚体变换（旋转 + 平移）统一为 $4\times 4$ 矩阵乘法：

$$
\begin{pmatrix} \mathbf{R} & \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}
\begin{pmatrix} \mathbf{p} \\ 1 \end{pmatrix}
=
\begin{pmatrix} \mathbf{R}\mathbf{p} + \mathbf{t} \\ 1 \end{pmatrix}
$$

这个 $4\times 4$ 矩阵就是 $SE(3)$（特殊欧氏群）的元素，$\mathbf{T} \in SE(3)$。多个变换的复合就是矩阵乘法，求逆是：

$$
\mathbf{T}^{-1} = \begin{pmatrix} \mathbf{R}^\top & -\mathbf{R}^\top \mathbf{t} \\ \mathbf{0}^\top & 1 \end{pmatrix}
$$

齐次坐标也统一了无穷远点的表示：$(x, y, z, 0)^\top$ 表示无穷远处方向为 $(x,y,z)$ 的"理想点"，在投影几何里很有用。

## 旋转的参数化

旋转矩阵 $\mathbf{R} \in SO(3)$ 有 9 个元素，但只有 3 个自由度（旋转轴 2 个，旋转角 1 个）。几种参数化方式：

**轴角（Axis-Angle）**：单位旋转轴 $\hat{\mathbf{n}}$ 和旋转角 $\theta$，记 $\boldsymbol{\phi} = \theta \hat{\mathbf{n}}$。

通过 **Rodrigues 公式**恢复旋转矩阵：

$$
\mathbf{R} = \mathbf{I} + \sin\theta\, [\hat{\mathbf{n}}]_\times + (1-\cos\theta)\, [\hat{\mathbf{n}}]_\times^2
$$

这正是指数映射 $\mathbf{R} = e^{[\boldsymbol{\phi}]_\times}$ 的展开形式（下一节 Lie 群节会详细推导）。

**四元数（Quaternion）**：$\mathbf{q} = (q_w, q_x, q_y, q_z)$ 满足 $\|\mathbf{q}\| = 1$。旋转矩阵：

$$
\mathbf{R} = \begin{pmatrix}
1-2(q_y^2+q_z^2) & 2(q_xq_y - q_wq_z) & 2(q_xq_z + q_wq_y) \\
2(q_xq_y + q_wq_z) & 1-2(q_x^2+q_z^2) & 2(q_yq_z - q_wq_x) \\
2(q_xq_z - q_wq_y) & 2(q_yq_z + q_wq_x) & 1-2(q_x^2+q_y^2)
\end{pmatrix}
$$

四元数旋转插值（SLERP）平滑，适合动画；但有**双重覆盖**问题：$\mathbf{q}$ 和 $-\mathbf{q}$ 代表同一个旋转。

| 参数化 | 自由度 | 优点 | 缺点 |
|--------|--------|------|------|
| 旋转矩阵 $\in \mathbb{R}^{3\times 3}$ | 3（9 参数 +6 约束） | 直接变换点 | 冗余，优化需投影 |
| 轴角 $\boldsymbol{\phi} \in \mathbb{R}^3$ | 3 | 最小参数，$\boldsymbol{\phi}=0$ 是单位元 | $\|\boldsymbol{\phi}\| = \pi$ 处奇异 |
| 四元数 $\in \mathbb{R}^4$ | 3（+1 单位约束） | 旋转插值好，数值稳定 | 双重覆盖，需归一化 |
| 欧拉角 | 3 | 直觉最强 | 万向节死锁，顺序依赖 |

## 相机模型

**针孔相机（Pinhole Camera）**把三维点 $\mathbf{P}_w = (X, Y, Z)^\top$（世界坐标）投影到图像像素 $\mathbf{u} = (u, v)^\top$，分三步：

**第一步**（世界 → 相机坐标）：用外参矩阵 $[\mathbf{R} \mid \mathbf{t}]$：

$$
\mathbf{P}_c = \mathbf{R}\mathbf{P}_w + \mathbf{t}
$$

**第二步**（透视投影到归一化平面）：

$$
\mathbf{p} = \left(\frac{X_c}{Z_c},\, \frac{Y_c}{Z_c}\right)
$$

**第三步**（归一化平面 → 像素坐标）：用**内参矩阵** $\mathbf{K}$：

$$
\begin{pmatrix} u \\ v \\ 1 \end{pmatrix}
=
\underbrace{\begin{pmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{pmatrix}}_{\mathbf{K}}
\begin{pmatrix} X_c/Z_c \\ Y_c/Z_c \\ 1 \end{pmatrix}
$$

$f_x, f_y$ 是焦距（像素单位），$(c_x, c_y)$ 是主点（通常在图像中心）。完整的投影矩阵（$3\times 4$）：

$$
\lambda \begin{pmatrix} u \\ v \\ 1 \end{pmatrix}
=
\mathbf{K} [\mathbf{R} \mid \mathbf{t}]\, \tilde{\mathbf{P}}_w
$$

## 代码验证

```python
import numpy as np

# 齐次坐标：平移 + 旋转的统一矩阵乘法
def rotation_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

R = rotation_z(np.pi/4)         # 绕 z 轴 45°
t = np.array([1.0, 0.0, 0.5])   # 平移

# 4×4 变换矩阵
T = np.eye(4)
T[:3, :3] = R
T[:3, 3]  = t

p = np.array([1.0, 0.0, 0.0])   # 原始点
p_hom = np.append(p, 1)         # 齐次坐标
p_transformed = T @ p_hom
print(f"变换后的点: {p_transformed[:3].round(4)}")  # R·p + t

# 逆变换
T_inv = np.eye(4)
T_inv[:3, :3] = R.T
T_inv[:3, 3]  = -R.T @ t
p_recovered = T_inv @ p_transformed
print(f"逆变换恢复: {p_recovered[:3].round(4)}")  # 应还原为 [1,0,0]

# 相机投影
K = np.array([[500, 0, 320],
              [0, 500, 240],
              [0,   0,   1]], dtype=float)

# 世界坐标系中的点（相机正前方 2m，偏右 0.5m）
Pw = np.array([0.5, 0.0, 2.0])
Rt = np.hstack([np.eye(3), np.zeros((3,1))])  # 相机与世界对齐（外参为单位阵）
Pw_hom = np.append(Pw, 1)

# 投影
lam_uv1 = K @ Rt @ Pw_hom
uv = lam_uv1[:2] / lam_uv1[2]
print(f"\n投影像素坐标: u={uv[0]:.1f}, v={uv[1]:.1f}")
# u = 500*0.5/2 + 320 = 445, v = 240
```

## 在深度学习中的应用

NeRF、3D Gaussian Splatting 都以相机内外参为输入，把 3D 场景渲染到 2D 图像，整个可微渲染管线依赖这里的投影矩阵。6D 位姿估计（BundleSDF、FoundPose）输出相机的 $[\mathbf{R} \mid \mathbf{t}]$，训练损失需要在 SE(3) 上正确度量距离。等变网络（E(n)-GNN、SE(3)-Transformer）把分子/点云的旋转等变性编码进架构，内部运算正是 SO(3) 作用。

下一节讲 Lie 群与 SE(3)，解释为什么旋转矩阵不能直接做加减法，以及如何通过指数映射/对数映射在旋转群上做优化。
