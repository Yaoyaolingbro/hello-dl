# 链式法则（矩阵形式）

!!! info "参考资料"
    **教程**

    - [Backpropagation, Intuitions](https://cs231n.github.io/optimization-2/) — Stanford CS231n，用计算图讲链式法则，非常直观
    - [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — Andrej Karpathy，说明为什么不能只靠自动微分

    **教材**

    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 5.6

---

## 直觉 (Intuition)

链式法则是反向传播的唯一数学基础。它回答了这个问题：如果 $L$ 通过 $\mathbf{y}$ 依赖于 $\mathbf{x}$，而 $\mathbf{y}$ 又依赖于 $\mathbf{x}$，那么 $L$ 对 $\mathbf{x}$ 的梯度怎么算？答案是把中间变量的 Jacobian 矩阵乘起来。

每一层神经网络的参数梯度，都是从 loss 开始、通过层层 Jacobian 矩阵相乘传递回来的。这就是为什么矩阵维度必须对齐——维度对不上就是 bug。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $\mathbf{x} \to \mathbf{y} \to L$ | 计算图（L 是标量 loss） |
| $\frac{\partial L}{\partial \mathbf{x}}$ | L 对向量 $\mathbf{x}$ 的梯度，形状与 $\mathbf{x}$ 相同 |
| $\mathbf{J}_{\mathbf{y} \to L}$ | L 对 $\mathbf{y}$ 的梯度（行向量形式），$= (\nabla_\mathbf{y} L)^\top$ |

---

## 标量链式法则（回顾）

对标量 $x \to y \to L$：

$$\frac{dL}{dx} = \frac{dL}{dy} \cdot \frac{dy}{dx}$$

矩阵形式是把这里的"乘"换成"矩阵乘法"。

---

## 向量链式法则

设 $\mathbf{x} \in \mathbb{R}^n$，$\mathbf{y} = \mathbf{f}(\mathbf{x}) \in \mathbb{R}^m$，$L = g(\mathbf{y}) \in \mathbb{R}$。

$\mathbf{f}$ 的 Jacobian 是 $\mathbf{J} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \in \mathbb{R}^{m \times n}$，$\nabla_\mathbf{y} L \in \mathbb{R}^m$ 是 $L$ 对 $\mathbf{y}$ 的梯度。

链式法则给出：

$$\nabla_\mathbf{x} L = \mathbf{J}^\top \nabla_\mathbf{y} L$$

注意 $\mathbf{J}^\top \in \mathbb{R}^{n \times m}$，乘以 $\nabla_\mathbf{y} L \in \mathbb{R}^m$，结果是 $\mathbb{R}^n$——正好与 $\mathbf{x}$ 的形状一致。

!!! note "转置的来源"
    转置是由布局约定决定的。我们约定梯度与变量形状相同（分母布局），所以从"输出方向"传回来的梯度需要乘 Jacobian 的转置才能让形状对齐。这个转置在反向传播代码里处处可见，不是偶然。

---

## 全连接层的反向传播推导

以全连接层 $\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ 为例，完整推导其反向传播公式。

已知：$\nabla_\mathbf{y} L$（从上层传来的梯度，形状 $m$）。

**第一步：求 $\nabla_\mathbf{x} L$。**

$\mathbf{y} = \mathbf{W}\mathbf{x} + \mathbf{b}$ 关于 $\mathbf{x}$ 的 Jacobian 是 $\mathbf{J}_\mathbf{x} = \mathbf{W} \in \mathbb{R}^{m \times n}$。

由链式法则：

$$\nabla_\mathbf{x} L = \mathbf{W}^\top \nabla_\mathbf{y} L$$

**第二步：求 $\nabla_\mathbf{W} L$。**

这里输出 $L$ 对矩阵 $\mathbf{W}$ 求导。用微分的方式推导：

$dL = \langle \nabla_\mathbf{y} L, d\mathbf{y} \rangle = \langle \nabla_\mathbf{y} L, d\mathbf{W} \cdot \mathbf{x} \rangle = \text{tr}(\nabla_\mathbf{y} L^\top \cdot d\mathbf{W} \cdot \mathbf{x})$

利用迹的循环置换性：$= \text{tr}(\mathbf{x} \cdot \nabla_\mathbf{y} L^\top \cdot d\mathbf{W})^\top = \text{tr}((\nabla_\mathbf{y} L \cdot \mathbf{x}^\top)^\top \cdot d\mathbf{W})$

所以：

$$\nabla_\mathbf{W} L = \nabla_\mathbf{y} L \cdot \mathbf{x}^\top$$

形状检查：$\nabla_\mathbf{y} L \in \mathbb{R}^m$，$\mathbf{x}^\top \in \mathbb{R}^{1 \times n}$，结果 $\in \mathbb{R}^{m \times n}$——与 $\mathbf{W}$ 形状一致。

**第三步：求 $\nabla_\mathbf{b} L$。**

$d\mathbf{y} = d\mathbf{b}$，所以 $\nabla_\mathbf{b} L = \nabla_\mathbf{y} L$（形状相同，直接传递）。

!!! note "三个梯度公式小结"

    $$\nabla_\mathbf{x} L = \mathbf{W}^\top \nabla_\mathbf{y} L$$
    $$\nabla_\mathbf{W} L = \nabla_\mathbf{y} L \cdot \mathbf{x}^\top$$
    $$\nabla_\mathbf{b} L = \nabla_\mathbf{y} L$$

    这三个公式值得记住。全连接层的反向传播就是这三行。

### 数值例子：手算一次完整的前向+反向

取一个最小的 2→2→1 网络，全部用具体数字。

**参数：** $\mathbf{W} = \begin{pmatrix}1&2\\0&1\end{pmatrix}$，$\mathbf{b} = \mathbf{0}$，输出权重 $\mathbf{w}_2 = \begin{pmatrix}1\\1\end{pmatrix}$

**输入和标签：** $\mathbf{x} = \begin{pmatrix}1\\1\end{pmatrix}$，真值 $y^* = 5$，损失 $L = (\hat{y} - y^*)^2$

**前向传播：**

$$\mathbf{h} = \mathbf{W}\mathbf{x} = \begin{pmatrix}3\\1\end{pmatrix}, \quad \hat{y} = \mathbf{w}_2^\top\mathbf{h} = 4, \quad L = (4-5)^2 = 1$$

**反向传播（从 $L$ 往回推）：**

$$\frac{\partial L}{\partial \hat{y}} = 2(4-5) = -2 \quad \Rightarrow \quad \nabla_\mathbf{h} L = \mathbf{w}_2 \cdot (-2) = \begin{pmatrix}-2\\-2\end{pmatrix}$$

$$\nabla_\mathbf{W} L = \nabla_\mathbf{h} L \cdot \mathbf{x}^\top = \begin{pmatrix}-2\\-2\end{pmatrix}\begin{pmatrix}1&1\end{pmatrix} = \begin{pmatrix}-2&-2\\-2&-2\end{pmatrix}$$

$$\nabla_\mathbf{x} L = \mathbf{W}^\top \nabla_\mathbf{h} L = \begin{pmatrix}1&0\\2&1\end{pmatrix}\begin{pmatrix}-2\\-2\end{pmatrix} = \begin{pmatrix}-2\\-6\end{pmatrix}$$

**验证（数值差分）：** 把 $W_{11}$ 从 $1$ 改到 $1.001$，重算 $\hat{y} = 1.001+2 = 3.001 + 1 = 4.001$，$L = (4.001-5)^2 = 0.998001$，$\Delta L/\Delta W_{11} \approx -2.0 \approx (\nabla_\mathbf{W}L)_{11}$ ✓

---

## 多层复合：梯度消失的数学根源

设三层网络 $\mathbf{x} \xrightarrow{\mathbf{W}_1} \mathbf{h}_1 \xrightarrow{\mathbf{W}_2} \mathbf{h}_2 \xrightarrow{\mathbf{W}_3} L$，则：

$$
\nabla_\mathbf{x} L = \mathbf{J}_1^\top \mathbf{J}_2^\top \mathbf{J}_3^\top \nabla_\mathbf{h}_3 L
$$

这是三个 Jacobian 矩阵的乘积。如果每个 $\|\mathbf{J}_i\| < 1$，乘积会指数级收缩→梯度消失；如果 $\|\mathbf{J}_i\| > 1$，乘积会指数级增长→梯度爆炸。

残差连接（ResNet）通过让 $\mathbf{J}_i = \mathbf{I} + \text{小量}$ 来保证梯度在传播时不消失，这是其成功的数学原因。


!!! tip "在深度学习中的应用"

    - **所有反向传播**：本节公式就是 `Linear` 层 `.backward()` 的底层实现。理解它才能调 batch size 时判断梯度累积对不对。
    - **梯度裁剪**：监控 Jacobian 乘积的范数，超过阈值就缩放梯度（`torch.nn.utils.clip_grad_norm_`）。
    - **残差连接**：让 Jacobian 包含单位矩阵分量，梯度传播路径上始终有"恒等捷径"。

!!! note "本节结论在后面的用处"
    链式法则是 Part 2 整个**反向传播**章节的数学核心。理解 Jacobian 矩阵的乘积关系，也有助于读懂 Part 3 中 Normalizing Flows 的对数似然公式（需要计算 Jacobian 行列式）。
