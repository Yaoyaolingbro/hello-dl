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

### SVD 与特征值分解的关系

$\mathbf{A}^\top \mathbf{A} = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top)^\top (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top) = \mathbf{V} \mathbf{\Sigma}^\top \mathbf{U}^\top \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top = \mathbf{V}(\mathbf{\Sigma}^\top \mathbf{\Sigma})\mathbf{V}^\top$

这正是 $\mathbf{A}^\top \mathbf{A}$ 的特征值分解！右奇异向量 $\mathbf{V}$ 是 $\mathbf{A}^\top \mathbf{A}$ 的特征向量，奇异值 $\sigma_i = \sqrt{\lambda_i(\mathbf{A}^\top \mathbf{A})}$。

---

## 外积展开形式

SVD 有一种等价的外积展开形式，对理解低秩近似更直观：

$$\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

每项 $\sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ 是一个**秩 1 矩阵**，代表矩阵 $\mathbf{A}$ 的一个"成分"，重要程度由 $\sigma_i$ 决定。

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

---

## 代码验证

```python
import numpy as np

# 对随机矩阵做 SVD
A = np.random.randn(5, 4)
U, S, Vt = np.linalg.svd(A, full_matrices=False)  # 紧凑 SVD

print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")  # (5,4), (4,), (4,4)

# 验证重建
A_reconstructed = U @ np.diag(S) @ Vt
print(np.allclose(A, A_reconstructed))  # True

# 验证 U, V 是正交的
print(np.allclose(U.T @ U, np.eye(4)))   # True
print(np.allclose(Vt @ Vt.T, np.eye(4))) # True

# 低秩近似：只保留前 k 个奇异值
k = 2
A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
error = np.linalg.norm(A - A_k, 'fro')
theoretical_error = np.sqrt((S[k:] ** 2).sum())
print(np.allclose(error, theoretical_error))  # True，验证 Eckart-Young
print(f"近似误差: {error:.4f}，保留奇异值之和: {S[:k].sum():.4f}/{S.sum():.4f}")
```

```python
# LoRA 的参数量对比
def lora_params(d, r):
    return r * d + r * d  # A: d×r, B: r×d

d = 4096  # GPT-3 级别的隐藏维度
for r in [4, 8, 16, 32]:
    full = d * d
    lora = lora_params(d, r)
    print(f"r={r:2d}: LoRA {lora/1e3:.1f}K vs Full {full/1e6:.1f}M (节省 {full//lora}×)")
# r= 4: LoRA 32.8K vs Full 16.8M (节省 512×)
# r= 8: LoRA 65.5K vs Full 16.8M (节省 256×)
# r=16: LoRA 131.1K vs Full 16.8M (节省 128×)
# r=32: LoRA 262.1K vs Full 16.8M (节省 64×)
```

!!! tip "在深度学习中的应用"

    - **LoRA 微调**：在 ChatGPT 之后几乎成为 LLM 微调的标准方法。秩 $r=8$ 在大多数任务上就能取得接近全量微调的效果。
    - **权重压缩**：对预训练权重做截断 SVD，可以减少推理时的存储和计算量。
    - **注意力矩阵的低秩近似**：Linformer 用低秩近似 attention 矩阵，把 Transformer 的复杂度从 $O(n^2)$ 降到 $O(n)$。
    - **PCA via SVD**：对数据矩阵 $\mathbf{X}$ 做 SVD，等价于对协方差矩阵做特征值分解（更数值稳定）。

!!! note "本节结论在后面的用处"
    低秩分解的思想在 Part 3 的**生成模型**（LoRA 作为条件控制）和 Part 4 的**LLM 微调**章节会直接用到。SVD 也是理解**度量学习**和**知识蒸馏**里矩阵近似误差的工具。
