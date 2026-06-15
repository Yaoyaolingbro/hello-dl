# SVD 与低秩近似

!!! info "参考资料"
    **教材**

    - Gilbert Strang, *Introduction to Linear Algebra*, 5th ed. — Chapter 7
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 4.5

    **论文**

    - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

---

## 直觉 (Intuition)

特征值分解只对方阵有效，而深度学习里大多数权重矩阵都是长方形的。SVD 把特征值分解推广到任意矩阵，并给出了一个直觉上优雅的几何解读：任何线性变换都可以分解为"旋转 → 缩放 → 旋转"三步。

低秩近似是 SVD 最重要的应用之一：用前 $k$ 个奇异值对应的分量来近似一个矩阵，丢掉"次要信息"保留"主要结构"。LoRA 就建立在这个思想上。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $\mathbf{A} \in \mathbb{R}^{m \times n}$ | 待分解的矩阵（$m \geq n$） |
| $\sigma_i$ | 第 $i$ 个奇异值（非负实数，$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$） |
| $\mathbf{u}_i \in \mathbb{R}^m$ | 左奇异向量（$\mathbf{U}$ 的列） |
| $\mathbf{v}_i \in \mathbb{R}^n$ | 右奇异向量（$\mathbf{V}$ 的列） |
| $r = \text{rank}(\mathbf{A})$ | 矩阵的秩 |

---

## SVD 分解

!!! note "奇异值分解定理"
    任意矩阵 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 都可以分解为：

    $$\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$$

    其中：

    - $\mathbf{U} \in \mathbb{R}^{m \times m}$：正交矩阵，列为**左奇异向量**
    - $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$：仅主对角线非零的矩阵，对角元素为奇异值 $\sigma_1 \geq \cdots \geq \sigma_r \geq 0$
    - $\mathbf{V} \in \mathbb{R}^{n \times n}$：正交矩阵，列为**右奇异向量**

几何解读：$\mathbf{V}^\top$（旋转输入空间）→ $\mathbf{\Sigma}$（沿坐标轴缩放）→ $\mathbf{U}$（旋转输出空间）。任何线性变换都是这三步的复合。

![SVD 几何解释](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Singular-Value-Decomposition.svg/800px-Singular-Value-Decomposition.svg.png)

*单位圆（左）经 $\mathbf{V}^\top$ 旋转后，再被 $\mathbf{\Sigma}$ 沿两个坐标轴分别拉伸成椭圆，最后 $\mathbf{U}$ 再旋转一次得到最终形状（右）。奇异值就是椭圆的半轴长度。来源：[Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)*

### SVD 与特征值分解的关系

$\mathbf{A}^\top \mathbf{A} = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top)^\top (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top) = \mathbf{V} \mathbf{\Sigma}^\top \mathbf{U}^\top \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top = \mathbf{V}(\mathbf{\Sigma}^\top \mathbf{\Sigma})\mathbf{V}^\top$

这正是 $\mathbf{A}^\top \mathbf{A}$ 的特征值分解！右奇异向量 $\mathbf{V}$ 是 $\mathbf{A}^\top \mathbf{A}$ 的特征向量，奇异值 $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top \mathbf{A})}$。

---

## 外积展开形式

SVD 有一种等价的外积展开形式，对理解低秩近似更直观：

$$\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

每项 $\sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ 是一个**秩 1 矩阵**，代表矩阵 $\mathbf{A}$ 的一个"成分"，重要程度由 $\sigma_i$ 决定。

### 数值例子（3×2）

取 $\mathbf{A} = \begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix}$（3 行 2 列，秩为 2）。

**求奇异值：** $\mathbf{A}^\top\mathbf{A} = \begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix}^\top\begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix} = \begin{pmatrix}2&1\\1&2\end{pmatrix}$

$\mathbf{A}^\top\mathbf{A}$ 的特征值：$(2-\lambda)^2 - 1 = 0 \Rightarrow \lambda_1 = 3,\, \lambda_2 = 1$，奇异值 $\sigma_1 = \sqrt{3} \approx 1.732$，$\sigma_2 = 1$。

**右奇异向量**（$\mathbf{A}^\top\mathbf{A}$ 的特征向量）：$\mathbf{v}_1 = \dfrac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix}$，$\mathbf{v}_2 = \dfrac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix}$

**左奇异向量：** $\mathbf{u}_i = \mathbf{A}\mathbf{v}_i / \sigma_i$

$$\mathbf{u}_1 = \frac{1}{\sqrt{3}}\begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\end{pmatrix} = \frac{1}{\sqrt{6}}\begin{pmatrix}2\\1\\1\end{pmatrix}, \quad \mathbf{u}_2 = \frac{1}{1}\begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix}\frac{1}{\sqrt{2}}\begin{pmatrix}1\\-1\end{pmatrix} = \frac{1}{\sqrt{2}}\begin{pmatrix}0\\1\\-1\end{pmatrix}$$

**外积展开验证：**

$$\sigma_1\mathbf{u}_1\mathbf{v}_1^\top = \sqrt{3}\cdot\frac{1}{\sqrt{6}}\begin{pmatrix}2\\1\\1\end{pmatrix}\cdot\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\end{pmatrix} = \frac{\sqrt{3}}{\sqrt{12}}\begin{pmatrix}2&2\\1&1\\1&1\end{pmatrix} = \frac{1}{2}\begin{pmatrix}2&2\\1&1\\1&1\end{pmatrix} = \begin{pmatrix}1&1\\0.5&0.5\\0.5&0.5\end{pmatrix}$$

$$\sigma_2\mathbf{u}_2\mathbf{v}_2^\top = 1\cdot\frac{1}{\sqrt{2}}\begin{pmatrix}0\\1\\-1\end{pmatrix}\cdot\frac{1}{\sqrt{2}}\begin{pmatrix}1&-1\end{pmatrix} = \begin{pmatrix}0&0\\0.5&-0.5\\-0.5&0.5\end{pmatrix}$$

$$\mathbf{A}_1 + \mathbf{A}_2 = \begin{pmatrix}1&1\\1&0\\0&1\end{pmatrix} = \mathbf{A} \checkmark$$

**低秩近似的含义：** 如果只保留 $\mathbf{A}_1$（秩 1 近似），误差 $= \sigma_2 = 1$，而 $\mathbf{A}_1$ 捕获了 $\mathbf{A}$ 的主要方向——两列大体相似（正相关），这正是 $\sigma_1 > \sigma_2$ 告诉我们的信息。

---

## 低秩近似与 Eckart-Young 定理

!!! note "Eckart-Young 定理"
    在所有秩不超过 $k$ 的矩阵里，截断 SVD $\mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ 是 $\mathbf{A}$ 的**最优低秩近似**：

    $$\mathbf{A}_k = \arg\min_{\text{rank}(\mathbf{B}) \leq k} \|\mathbf{A} - \mathbf{B}\|_F$$

    近似误差为 $\|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$。

这意味着，如果一个矩阵的奇异值在前 $k$ 项后迅速衰减，就可以用 $k$ 个秩 1 矩阵的和来高精度近似它，同时把存储量从 $mn$ 降到 $k(m+n)$。

---

## LoRA 的数学基础

LoRA（Low-Rank Adaptation）的核心假设是：微调大模型时，权重的**变化量** $\Delta\mathbf{W}$ 是低秩的。

直觉来自 Eckart-Young 定理的逆命题：如果任务适应可以用少量"方向"描述，那么 $\Delta\mathbf{W}$ 的奇异值会快速衰减，可以用秩 $r \ll \min(m, n)$ 的矩阵来近似。

LoRA 的参数化：不直接训练 $\Delta\mathbf{W}$，而是训练两个小矩阵 $\mathbf{A} \in \mathbb{R}^{m \times r}$ 和 $\mathbf{B} \in \mathbb{R}^{r \times n}$，让 $\Delta\mathbf{W} = \mathbf{A}\mathbf{B}$。

| 方法 | 参数量 | $r=8, d=4096$ 时 |
|------|--------|-----------------|
| 全量微调 $\Delta\mathbf{W}$ | $mn$ | 16.7M |
| LoRA | $r(m+n)$ | 65.5K（**降低 255 倍**） |


!!! tip "在深度学习中的应用"

    - **LoRA 微调**：在 ChatGPT 之后几乎成为 LLM 微调的标准方法。秩 $r=8$ 在大多数任务上就能取得接近全量微调的效果。
    - **权重压缩**：对预训练权重做截断 SVD，可以减少推理时的存储和计算量。
    - **注意力矩阵的低秩近似**：Linformer 用低秩近似 attention 矩阵，把 Transformer 的复杂度从 $O(n^2)$ 降到 $O(n)$。
    - **PCA via SVD**：对数据矩阵 $\mathbf{X}$ 做 SVD，等价于对协方差矩阵做特征值分解（更数值稳定）。

!!! note "本节结论在后面的用处"
    低秩分解的思想在 Part 3 的**生成模型**（LoRA 作为条件控制）和 Part 4 的**LLM 微调**章节会直接用到。SVD 也是理解**度量学习**和**知识蒸馏**里矩阵近似误差的工具。
