# 图的矩阵表示与拉普拉斯矩阵

!!! info "参考资料"
    **教材**

    - Spielman, *Spectral Graph Theory* — [课程讲义免费](http://www.cs.yale.edu/homes/spielman/sgt/)（进阶）
    - *Graph Neural Networks: Foundations, Frontiers, and Applications* — Chapter 3

    **论文**

    - Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
    - Defferrard et al., "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering" (ChebNet), NeurIPS 2016

---

## 直觉 (Intuition)

图是描述"关系"的数据结构：社交网络、分子结构、道路网络、知识图谱，都可以用节点和边来表示。把图变成矩阵，就可以用线性代数工具来分析它。

拉普拉斯矩阵是图上最核心的矩阵，它的特征值描述了图的"频率成分"：小特征值对应平滑的全局结构，大特征值对应局部的高频变化。GNN 的谱方法本质上就是在图的频域里做滤波。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $G = (V, E)$ | 图，$V$ 是节点集，$E$ 是边集，$|V| = n$ |
| $\mathbf{A} \in \mathbb{R}^{n \times n}$ | 邻接矩阵 |
| $d_i$ | 节点 $i$ 的度（与它相连的边数） |
| $\mathbf{D}$ | 度矩阵，对角矩阵 $D_{ii} = d_i$ |
| $\mathbf{L}$ | 图拉普拉斯矩阵 |
| $\hat{\mathbf{L}}$ | 归一化拉普拉斯矩阵 |

---

## 邻接矩阵

无权图 $G$ 的**邻接矩阵** $\mathbf{A} \in \{0, 1\}^{n \times n}$ 定义为：

$$A_{ij} = \begin{cases} 1 & \text{若节点 } i \text{ 和 } j \text{ 之间有边} \\ 0 & \text{否则} \end{cases}$$

对无向图，$\mathbf{A}$ 是对称矩阵（$A_{ij} = A_{ji}$）。有权图中，$A_{ij}$ 可以是边的权重。

邻接矩阵的一个直接应用：$(\mathbf{A}^k)_{ij}$ 给出从节点 $i$ 到节点 $j$ 经过恰好 $k$ 条边的路径数量。

---

## 度矩阵

节点 $i$ 的**度**是它相连的边数：$d_i = \sum_j A_{ij}$（即 $\mathbf{A}$ 第 $i$ 行的行和）。

**度矩阵** $\mathbf{D}$ 是对角矩阵，$D_{ii} = d_i$，非对角元素为零。

---

## 图拉普拉斯矩阵

图的**（组合）拉普拉斯矩阵**定义为：

$$\mathbf{L} = \mathbf{D} - \mathbf{A}$$

对角线元素 $L_{ii} = d_i$，非对角线 $L_{ij} = -A_{ij}$（若有边则为 $-1$，否则为 $0$）。

**为什么叫"拉普拉斯"？** 连续空间的拉普拉斯算子 $\Delta f = \nabla^2 f$ 描述函数在某点与周围的差异。图拉普拉斯是其离散版本：

$$(\mathbf{L}\mathbf{f})_i = d_i f_i - \sum_{j \in \mathcal{N}(i)} f_j = \sum_{j \in \mathcal{N}(i)} (f_i - f_j)$$

即节点 $i$ 的信号值与其邻居平均值之差，乘以度数。它衡量节点信号的"局部不光滑程度"。

!!! note "拉普拉斯矩阵是正半定的"
    对任意 $\mathbf{f} \in \mathbb{R}^n$：

    $$\mathbf{f}^\top \mathbf{L} \mathbf{f} = \sum_{(i,j) \in E} (f_i - f_j)^2 \geq 0$$

    **证明：**
    $\mathbf{f}^\top \mathbf{L} \mathbf{f} = \mathbf{f}^\top \mathbf{D} \mathbf{f} - \mathbf{f}^\top \mathbf{A} \mathbf{f} = \sum_i d_i f_i^2 - \sum_{i,j} A_{ij} f_i f_j = \sum_{(i,j) \in E} (f_i^2 + f_j^2 - 2f_i f_j) = \sum_{(i,j) \in E} (f_i - f_j)^2$

    这个二次型的值等于图上所有边的端点信号差的平方和。它为零当且仅当所有连通分量内 $\mathbf{f}$ 是常数。

    **推论：** $\mathbf{L}$ 的最小特征值是 $0$，对应特征向量是全 1 向量 $\mathbf{1}$。若图有 $k$ 个连通分量，则有 $k$ 个零特征值。

---

## 归一化拉普拉斯矩阵

组合拉普拉斯矩阵的问题是：度数大的节点对应的特征值也大，影响谱分析。**归一化拉普拉斯**消除了这种度偏差：

$$\hat{\mathbf{L}} = \mathbf{D}^{-1/2} \mathbf{L} \mathbf{D}^{-1/2} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}$$

归一化拉普拉斯的特征值在 $[0, 2]$ 之间，与图的规模无关。GCN 用的正是这个矩阵。

---

## 图信号与谱滤波

类比信号处理：如果把节点信号 $\mathbf{f}$ 看作时域信号，拉普拉斯矩阵的特征向量 $\{\mathbf{u}_i\}$ 就是图的"傅里叶基"，特征值 $\lambda_i$ 是对应的"频率"。

**图傅里叶变换：** $\hat{\mathbf{f}} = \mathbf{U}^\top \mathbf{f}$（将信号投影到特征向量基上）

**图逆傅里叶变换：** $\mathbf{f} = \mathbf{U}\hat{\mathbf{f}}$

**谱域卷积：** 在图上定义卷积 $\mathbf{f} * \mathbf{g} = \mathbf{U}(\hat{\mathbf{f}} \odot \hat{\mathbf{g}})$

谱图卷积网络（GCN）用一个参数化的频域滤波器 $h(\mathbf{\Lambda})$，将节点特征从邻居聚合：

$$\mathbf{H}' = h(\hat{\mathbf{L}}) \mathbf{H} = \mathbf{U} \, \text{diag}(h(\lambda_1), \ldots, h(\lambda_n)) \, \mathbf{U}^\top \mathbf{H}$$

Kipf & Welling 的 GCN 用一阶近似 $h(\lambda) \approx 1 - \lambda$，简化后得到：

$$\mathbf{H}' = \tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2} \mathbf{H} \mathbf{W}$$

其中 $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$（加自环），$\mathbf{W}$ 是可学习的权重矩阵。

---

## 代码验证

```python
import numpy as np

# 构建一个简单图：4 个节点，边 0-1, 1-2, 2-3, 3-0, 0-2
n = 4
A = np.zeros((n, n))
edges = [(0,1), (1,2), (2,3), (3,0), (0,2)]
for i, j in edges:
    A[i, j] = A[j, i] = 1.0

# 度矩阵
degrees = A.sum(axis=1)
D = np.diag(degrees)
print("度:", degrees)  # [3. 2. 3. 2.]

# 拉普拉斯矩阵
L = D - A
print(L)
# [[ 3. -1. -1. -1.]
#  [-1.  2. -1.  0.]
#  [-1. -1.  3. -1.]
#  [-1.  0. -1.  2.]]

# 验证正半定
eigenvalues = np.linalg.eigvalsh(L)
print("特征值:", eigenvalues.round(3))
# [0. 1. 3. 4.]  <- 最小特征值为 0（图连通，只有一个连通分量）
print(np.all(eigenvalues >= -1e-10))  # True

# 验证最小特征值对应全 1 向量
f_ones = np.ones(n)
print(np.allclose(L @ f_ones, np.zeros(n)))  # True  (L * 1 = 0)

# 验证二次型 = 边上差值平方和
f = np.array([1.0, 3.0, 2.0, 4.0])
quadratic = f @ L @ f
manual = sum((f[i] - f[j])**2 for i, j in edges)
print(np.isclose(quadratic, manual))  # True

# 归一化拉普拉斯
D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
L_norm = D_inv_sqrt @ L @ D_inv_sqrt
eig_norm = np.linalg.eigvalsh(L_norm)
print("归一化特征值范围:", eig_norm.min().round(3), "~", eig_norm.max().round(3))
# 0.0 ~ 2.0  <- 归一化后特征值在 [0, 2] 内
```

```python
# GCN 一层聚合的矩阵实现
import numpy as np

# 上面的图，加自环：A_tilde = A + I
A_tilde = A + np.eye(n)
d_tilde = A_tilde.sum(axis=1)
D_tilde_inv_sqrt = np.diag(1.0 / np.sqrt(d_tilde))

# GCN 传播矩阵：D^{-1/2} (A+I) D^{-1/2}
prop = D_tilde_inv_sqrt @ A_tilde @ D_tilde_inv_sqrt
print(prop.round(3))
# GCN 的一次消息传递：H_new = prop @ H @ W
# 其中 H 是节点特征矩阵，W 是可学习权重
```

!!! tip "在深度学习中的应用"

    - **图神经网络（GNN）**：GCN、GAT、GraphSAGE 都依赖邻接矩阵的各种变体来做消息传递。理解拉普拉斯矩阵是读懂 GCN 论文的关键。
    - **分子性质预测**：原子是节点，化学键是边，GNN 直接在分子图上学习分子性质（如药物活性）。
    - **知识图谱嵌入**：TransE、RotatE 等方法把实体和关系嵌入到向量空间，依赖图的邻接结构做归纳。
    - **谱聚类**：用拉普拉斯矩阵的特征向量对节点做聚类，是谱图理论最经典的应用。

!!! note "本节结论在后面的用处"
    拉普拉斯矩阵的正半定性和谱分解将在 Part 3 **GNN 章节**中直接使用。二次型 $\mathbf{f}^\top \mathbf{L} \mathbf{f}$ 是图上信号平滑度的度量，这个概念在图上扩散模型和图上注意力机制的分析中也会出现。
