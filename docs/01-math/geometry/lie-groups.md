# Lie 群与 SE(3)

!!! info "参考资料"
    **主要资料**
    - Murray, Li & Sastry, *A Mathematical Introduction to Robotic Manipulation* — 免费 PDF，第 2–3 章（SE(3) 的经典处理）
    - Sola et al., "A micro Lie theory for state estimation in robotics", 2018 — 实用且清晰，免费 PDF
    - Chirikjian, *Stochastic Models, Information Theory, and Lie Groups* — 深度参考

    **工具文档**
    - [Sophus](https://github.com/strasdat/Sophus) — C++ Lie 群库
    - [liegroups (Python)](https://github.com/utiasSTARS/liegroups) — Python 实现

## 直觉 (Intuition)

两个旋转矩阵之和未必还是旋转矩阵（行列式可能不为 1），所以旋转不能像普通向量那样直接加减。Lie 群提供了一套"在弯曲空间上做微积分"的框架：把局部的线性近似（Lie 代数，切空间）通过指数映射"包回"到非线性的群上。这样就能在旋转/位姿上做梯度下降、误差传播和插值——SLAM、位姿估计、等变网络的优化都需要这个工具。

## 主要符号

| 符号 | 含义 |
|------|------|
| $G$ | Lie 群（如 $SO(3)$，$SE(3)$） |
| $\mathfrak{g}$ | 对应的 Lie 代数（切空间） |
| $\text{Exp}: \mathfrak{g} \to G$ | 指数映射 |
| $\text{Log}: G \to \mathfrak{g}$ | 对数映射 |
| $\text{Ad}_X$ | 伴随映射（坐标系变换） |

## Lie 群的定义

**Lie 群（Lie group）**同时是一个光滑流形和一个群：
- 群结构：有乘法（复合变换）、单位元、逆元
- 流形结构：局部看起来像 $\mathbb{R}^n$，可以在上面做微积分

常见 Lie 群：

| 群 | 含义 | 维数 |
|---|------|------|
| $SO(2)$ | 二维旋转 | 1 |
| $SO(3)$ | 三维旋转（$3\times 3$ 正交矩阵，行列式 1） | 3 |
| $SE(3)$ | 三维刚体运动（旋转 + 平移） | 6 |
| $SL(n)$ | 行列式为 1 的 $n\times n$ 矩阵 | $n^2-1$ |

## Lie 代数

**Lie 代数（Lie algebra）** $\mathfrak{g}$ 是 Lie 群在单位元处的**切空间**。直觉：群描述有限变换，Lie 代数描述无穷小（微分）变换，是群的"线性化"。

$\mathfrak{so}(3)$：$SO(3)$ 的 Lie 代数，是全体 $3\times 3$ 反对称矩阵：

$$
\mathfrak{so}(3) = \{A \in \mathbb{R}^{3\times 3} \mid A^\top = -A\}
$$

维数为 3（与轴角参数化 $\boldsymbol{\phi} \in \mathbb{R}^3$ 同构），每个反对称矩阵 $[\boldsymbol{\phi}]_\times$ 对应轴角向量 $\boldsymbol{\phi} = \theta\hat{\mathbf{n}}$（见叉积节）。

$\mathfrak{se}(3)$：$SE(3)$ 的 Lie 代数，维数为 6，用扭量（twist）表示：

$$
\hat{\boldsymbol{\xi}} = \begin{pmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^\top & 0 \end{pmatrix} \in \mathbb{R}^{4\times 4}
$$

其中 $\boldsymbol{\omega} \in \mathbb{R}^3$ 是角速度，$\mathbf{v} \in \mathbb{R}^3$ 是线速度。

## 指数映射与对数映射

**指数映射（Exp）** 把 Lie 代数元素映射回群元素：

$$
\mathbf{R} = \text{Exp}(\boldsymbol{\phi}) = e^{[\boldsymbol{\phi}]_\times} = \mathbf{I} + \frac{\sin\theta}{\theta}[\boldsymbol{\phi}]_\times + \frac{1-\cos\theta}{\theta^2}[\boldsymbol{\phi}]_\times^2
$$

（这正是 Rodrigues 公式，$\theta = \|\boldsymbol{\phi}\|$）

**对数映射（Log）** 是 Exp 的逆：从旋转矩阵提取轴角向量：

$$
\boldsymbol{\phi} = \text{Log}(\mathbf{R}), \quad \theta = \arccos\!\left(\frac{\text{tr}(\mathbf{R}) - 1}{2}\right), \quad \hat{\mathbf{n}} = \frac{1}{2\sin\theta}\begin{pmatrix}R_{32}-R_{23}\\R_{13}-R_{31}\\R_{21}-R_{12}\end{pmatrix}
$$

!!! note "为什么需要 Exp/Log"
    旋转矩阵空间不是向量空间——你不能直接做 $\mathbf{R}_1 + \mathbf{R}_2$。但可以在 Lie 代数（线性空间）里做加减，再通过 Exp 映射回群：
    $$
    \mathbf{R}_1 \oplus \boldsymbol{\delta} = \text{Exp}(\text{Log}(\mathbf{R}_1) + \boldsymbol{\delta})
    $$
    SLAM 的优化、姿态平均、位姿插值都依赖这个"扰动 + 映射回群"的流程。

## 在优化中的应用

对位姿参数 $\mathbf{T} \in SE(3)$ 求最优，用**扰动模型**：

$$
\mathbf{T}^* = \arg\min_{\boldsymbol{\xi}} f\!\left(\text{Exp}(\boldsymbol{\xi}) \cdot \mathbf{T}_0\right)
$$

对 $\boldsymbol{\xi}$ 求导（在 $\boldsymbol{\xi}=0$ 处线性化），得到在 Lie 代数坐标系里的雅可比 $\mathbf{J}$，然后用高斯-牛顿或 LM 算法更新，再通过 Exp 映射回群。这是 g2o、Ceres、ORB-SLAM 等 SLAM 系统的标准优化流程。

## 代码验证

```python
import numpy as np

def exp_so3(phi):
    """轴角向量 phi → 旋转矩阵（Rodrigues 公式）"""
    theta = np.linalg.norm(phi)
    if theta < 1e-8:
        return np.eye(3)
    n = phi / theta
    K = np.array([[0,-n[2],n[1]],[n[2],0,-n[0]],[-n[1],n[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)

def log_so3(R):
    """旋转矩阵 → 轴角向量"""
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-8:
        return np.zeros(3)
    vee = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])
    return theta / (2 * np.sin(theta)) * vee

# 验证 Exp(Log(R)) = R
phi_true = np.array([0.1, 0.2, 0.3])  # 轴角向量
R = exp_so3(phi_true)
phi_recovered = log_so3(R)
print(f"原始轴角: {phi_true}")
print(f"Log(Exp(φ)) = {phi_recovered.round(6)}")  # 应还原
print(f"Exp(Log(R)) = R: {np.allclose(exp_so3(phi_recovered), R)}")  # True

# 旋转复合：R3 = R2 @ R1
phi1 = np.array([0.5, 0.0, 0.0])   # 绕 x 轴 0.5 rad
phi2 = np.array([0.0, 0.5, 0.0])   # 绕 y 轴 0.5 rad
R1, R2 = exp_so3(phi1), exp_so3(phi2)
R3 = R2 @ R1   # 先做 R1，再做 R2

# BCH 近似：log(R2 @ R1) ≈ phi1 + phi2（小角度时）
phi3 = log_so3(R3)
print(f"\nlog(R2 @ R1) = {phi3.round(4)}")
print(f"phi1 + phi2  = {(phi1+phi2).round(4)}")  # BCH 近似（小角度近似，不完全相等）
```

## 在深度学习中的应用

ORB-SLAM、VINS-Mono 等 SLAM 系统在 SE(3) 上做非线性最小二乘，位姿图优化（g2o/iSAM2）用 Lie 群参数化。SE(3)-Transformer、等变点云网络（E3NN）把 SO(3)/SE(3) 等变性编码进神经网络架构，使输出相对旋转/平移保持正确变换关系。6D 位姿估计的损失函数需要 SE(3) 上的距离度量（而非欧氏距离），Log 映射给出了合适的测地距离。

下一节讲贝塞尔曲线与 B 样条，从控制点出发用 de Casteljau 递推构造光滑曲线，为机器人轨迹规划和字体/图形设计提供数学工具。
