# 特殊矩阵

!!! info "参考资料"
    **教材**

    - Gilbert Strang, *Introduction to Linear Algebra*, 5th ed. — Chapter 6（对称矩阵与正定）
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 3.4–3.5

---

## 直觉 (Intuition)

大多数矩阵是没有特殊结构的，但深度学习里出现的矩阵很多都有额外性质：协方差矩阵是对称正半定的，旋转矩阵是正交的，注意力分数矩阵的求导会遇到对称矩阵的特殊性质。这些结构不只是数学上的美感，而是直接决定了哪些运算可以化简、哪些分解可以使用。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $\mathbf{A} \in \mathbb{R}^{n \times n}$ | 方阵 |
| $\mathbf{A} \succ 0$ | $\mathbf{A}$ 是正定矩阵 |
| $\mathbf{A} \succeq 0$ | $\mathbf{A}$ 是正半定矩阵 |
| $\lambda_i(\mathbf{A})$ | $\mathbf{A}$ 的第 $i$ 个特征值 |

---

## 对称矩阵

若 $\mathbf{A}^\top = \mathbf{A}$，则称 $\mathbf{A}$ 为**对称矩阵**。

对称矩阵的所有特征值都是**实数**，且不同特征值对应的特征向量**正交**。这个性质（谱定理）允许对其做正交对角化，是后面特征值分解章节的核心。

深度学习里的对称矩阵：

- 协方差矩阵 $\Sigma = \mathbb{E}[(\mathbf{x} - \mu)(\mathbf{x} - \mu)^\top]$
- 注意力分数矩阵（若 Q = K，则 $\mathbf{Q}\mathbf{K}^\top$ 对称）
- Hessian 矩阵（二阶偏导数矩阵，当函数足够光滑时）

---

## 正定与正半定矩阵

### 定义

对称矩阵 $\mathbf{A}$ 是**正定的**（Positive Definite，PD），如果对所有非零向量 $\mathbf{x}$：

$$\mathbf{x}^\top \mathbf{A} \mathbf{x} > 0$$

对称矩阵 $\mathbf{A}$ 是**正半定的**（Positive Semi-Definite，PSD），如果对所有向量 $\mathbf{x}$：

$$\mathbf{x}^\top \mathbf{A} \mathbf{x} \geq 0$$

直觉：$\mathbf{x}^\top \mathbf{A} \mathbf{x}$ 是一个二次型（quadratic form）。正定矩阵对应的二次型是一个"碗形"——在任何方向上都是凸的，碗底在原点。正半定矩阵允许碗底是一个子空间（某些方向上"平"）。

!!! note "正定性的等价条件"
    以下条件等价（对实对称矩阵 $\mathbf{A}$）：

    1. $\mathbf{A} \succ 0$（正定）
    2. 所有特征值 $\lambda_i > 0$
    3. 所有主子式（leading principal minors）为正（Sylvester 判据）
    4. 存在 Cholesky 分解 $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$，其中 $\mathbf{L}$ 是下三角矩阵

    Cholesky 分解是正定矩阵的"平方根"——比直接求逆快两倍，数值更稳定。

### 协方差矩阵一定是正半定的

对任意数据矩阵 $\mathbf{X} \in \mathbb{R}^{n \times d}$（$n$ 个样本，$d$ 维特征），样本协方差矩阵 $\mathbf{S} = \mathbf{X}^\top \mathbf{X} / (n-1)$ 是正半定的：

$$\mathbf{v}^\top \mathbf{S} \mathbf{v} = \frac{1}{n-1} \mathbf{v}^\top \mathbf{X}^\top \mathbf{X} \mathbf{v} = \frac{1}{n-1} \|\mathbf{X}\mathbf{v}\|^2 \geq 0$$

### 数值例子：2×2 正定矩阵验证

取 $\mathbf{A} = \begin{pmatrix}2&1\\1&3\end{pmatrix}$。

**Sylvester 判据：** $\Delta_1 = 2 > 0$，$\Delta_2 = \det(\mathbf{A}) = 6-1 = 5 > 0$，所以 $\mathbf{A}$ 是正定的。

**特征值：** $\det(\mathbf{A}-\lambda\mathbf{I}) = (2-\lambda)(3-\lambda) - 1 = \lambda^2 - 5\lambda + 5 = 0$，得 $\lambda = \frac{5\pm\sqrt{5}}{2}$，即 $\lambda_1 \approx 1.38 > 0$，$\lambda_2 \approx 3.62 > 0$ ✓

**二次型验证：** 取 $\mathbf{x} = \begin{pmatrix}1\\-1\end{pmatrix}$：$\mathbf{x}^\top\mathbf{A}\mathbf{x} = [1,-1]\begin{pmatrix}1\\-2\end{pmatrix} = 1\cdot1 + (-1)\cdot(-2) = 3 > 0$ ✓

---

## 正交矩阵

若方阵 $\mathbf{Q}$ 满足 $\mathbf{Q}^\top \mathbf{Q} = \mathbf{Q}\mathbf{Q}^\top = \mathbf{I}$，则称 $\mathbf{Q}$ 为**正交矩阵**。

等价地，$\mathbf{Q}^{-1} = \mathbf{Q}^\top$——求逆只需转置，计算上极其高效。

几何含义：正交矩阵只做**旋转**（$\det(\mathbf{Q}) = 1$）或**旋转加反射**（$\det(\mathbf{Q}) = -1$），不缩放长度。所以正交变换保长度：$\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$。

正交矩阵在深度学习里的用途：

- SVD 的左右奇异向量矩阵 $\mathbf{U}$、$\mathbf{V}$ 是正交的
- 旋转矩阵（SE(3) 群）是 $3 \times 3$ 正交矩阵（$\det = 1$）
- 正交初始化（Orthogonal Initialization）：让初始权重矩阵是正交的，防止梯度爆炸/消失

### 数值例子：2D 旋转矩阵

旋转 $\theta = 30°$ 的正交矩阵：

$$\mathbf{Q} = \begin{pmatrix}\cos 30° & -\sin 30° \\ \sin 30° & \cos 30°\end{pmatrix} = \begin{pmatrix}\frac{\sqrt{3}}{2} & -\frac{1}{2} \\ \frac{1}{2} & \frac{\sqrt{3}}{2}\end{pmatrix}$$

验证 $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$：$\frac{3}{4}+\frac{1}{4} = 1$，对角线为 1；非对角线 $\frac{\sqrt{3}}{4}-\frac{\sqrt{3}}{4} = 0$ ✓

取 $\mathbf{x} = \begin{pmatrix}1\\0\end{pmatrix}$（水平单位向量），旋转后：$\mathbf{Q}\mathbf{x} = \begin{pmatrix}\sqrt{3}/2 \\ 1/2\end{pmatrix}$，长度 $= \sqrt{3/4+1/4} = 1$ ✓（正交变换**保长度**）

---

## 投影矩阵

矩阵 $\mathbf{P}$ 是**投影矩阵**，如果 $\mathbf{P}^2 = \mathbf{P}$（幂等性）。

直觉：投影一次和投影两次的效果相同，因为已经投影到子空间里了，再投影不会有新变化。

**正交投影**（Orthogonal Projection）还满足 $\mathbf{P} = \mathbf{P}^\top$。向量 $\mathbf{x}$ 在列空间 $\text{span}(\mathbf{A})$ 上的正交投影为：

$$\mathbf{P} = \mathbf{A}(\mathbf{A}^\top \mathbf{A})^{-1}\mathbf{A}^\top$$

$$\hat{\mathbf{x}} = \mathbf{P}\mathbf{x}$$

这是最小二乘法的几何理解：$\hat{\mathbf{x}}$ 是 $\mathbf{x}$ 在 $\mathbf{A}$ 的列空间上的正交投影，残差 $\mathbf{x} - \hat{\mathbf{x}}$ 与列空间垂直。

### 数值例子：2D 正交投影

将向量 $\mathbf{v} = \begin{pmatrix}3\\4\end{pmatrix}$ 投影到 $x$ 轴方向 $\mathbf{a} = \begin{pmatrix}1\\0\end{pmatrix}$ 上：

$$\mathbf{P} = \mathbf{a}\mathbf{a}^\top = \begin{pmatrix}1\\0\end{pmatrix}\begin{pmatrix}1&0\end{pmatrix} = \begin{pmatrix}1&0\\0&0\end{pmatrix}$$

$$\hat{\mathbf{v}} = \mathbf{P}\mathbf{v} = \begin{pmatrix}3\\0\end{pmatrix}, \quad \text{残差：}\mathbf{v} - \hat{\mathbf{v}} = \begin{pmatrix}0\\4\end{pmatrix}$$

验证正交性：$\begin{pmatrix}1&0\end{pmatrix}\begin{pmatrix}0\\4\end{pmatrix} = 0$ ✓，验证幂等性：$\mathbf{P}^2 = \mathbf{P}$ ✓


!!! tip "在深度学习中的应用"

    - **BatchNorm / LayerNorm 的稳定性**：归一化操作本质上是把激活值映射到单位超球面，涉及正交变换。
    - **Attention 的数值稳定性**：Softmax 前除以 $\sqrt{d_k}$ 是为了让注意力分数矩阵的特征值不会过大，防止梯度饱和。
    - **权重初始化**：Orthogonal Initialization（PyTorch `nn.init.orthogonal_`）使初始权重保长度，防止深层网络的梯度爆炸/消失。
    - **Cholesky 分解**：高斯过程（Gaussian Process）和部分变分推断方法里，用 Cholesky 分解来高效计算正定协方差矩阵的逆和行列式。

!!! note "本节结论在后面的用处"
    正定性是理解**特征值分解**和**优化算法**（凸性、二阶方法）的基础。正交矩阵是 **SVD** 分解的构成部分。
