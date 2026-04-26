# 度量空间

!!! info "参考资料"
    **教材**
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 2.3
    - Rudin, *Principles of Mathematical Analysis*, Chapter 2（完整数学处理，可选）

    **工程参考**
    - [FAISS 文档](https://faiss.ai/) — 工业界最常用的近似最近邻库，内置 L2/内积/余弦距离
    - [Annoy](https://github.com/spotify/annoy) — Spotify 开源，树结构 ANN

---

## 直觉 (Intuition)

上一节的 $L^2$ 范数给了我们欧氏距离，但距离的本质是什么？度量空间把"距离"这个概念抽象出来，只要满足三条公理的函数都可以叫做距离。这样做的好处是：只要你定义出一种合法的距离，就可以在这个空间上做最近邻搜索、聚类等所有依赖距离的操作。

在深度学习里，选择哪种距离并非无关紧要。图像检索用余弦相似度，目标检测用 IoU，语音识别用编辑距离——每种选择背后都有对该任务"什么叫相似"的理解。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $d(\mathbf{x}, \mathbf{y})$ | $\mathbf{x}$ 和 $\mathbf{y}$ 之间的距离（度量） |
| $B(\mathbf{x}, r)$ | 以 $\mathbf{x}$ 为中心、半径 $r$ 的开球 |
| $\text{sim}(\mathbf{x}, \mathbf{y})$ | 相似度（越大越相似，与距离方向相反） |

---

## 度量的三条公理

函数 $d: X \times X \to \mathbb{R}$ 是集合 $X$ 上的一个**度量**，当且仅当对所有 $\mathbf{x}, \mathbf{y}, \mathbf{z} \in X$：

1. **非负性**：$d(\mathbf{x}, \mathbf{y}) \geq 0$，且 $d(\mathbf{x}, \mathbf{y}) = 0 \Leftrightarrow \mathbf{x} = \mathbf{y}$
2. **对称性**：$d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$
3. **三角不等式**：$d(\mathbf{x}, \mathbf{z}) \leq d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$

每条公理都有直觉含义：距离不能是负数；从 A 到 B 和从 B 到 A 一样远；绕路不会更短。

任何满足这三条的函数都可以用来构建度量空间，进而使用依赖"距离"概念的算法。

---

## 三种主要距离及其几何含义

### $L^2$ 距离（欧氏距离）

$$d_2(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_i (x_i - y_i)^2}$$

$L^2$ 距离的等距面（所有与 $\mathbf{x}$ 等距的点）是一个**圆球**，对所有方向一视同仁。它适合特征各维度尺度相近的情况，也是最符合几何直觉的距离。

深度学习里的典型用途：回归任务的 MSE loss、k-NN 分类、基于 patch 的图像特征匹配。

### $L^1$ 距离（曼哈顿距离）

$$d_1(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1 = \sum_i |x_i - y_i|$$

名字来自曼哈顿街道：只能沿轴方向走，不能斜穿。$L^1$ 的等距面是一个**菱形**。

它对离群维度的敏感性低于 $L^2$——一个维度差 100 和 100 个维度各差 1，在 $L^1$ 下效果一样；而在 $L^2$ 下前者贡献是后者的 $\sqrt{100}$ 倍。这让 $L^1$ 在稀疏高维特征上有时表现更好。

### 余弦距离

余弦相似度和余弦距离的定义：

$$\text{sim}_{\cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$$

$$d_{\cos}(\mathbf{x}, \mathbf{y}) = 1 - \text{sim}_{\cos}(\mathbf{x}, \mathbf{y})$$

余弦相似度**只看方向，不看幅度**。两个特征向量如果只是被缩放了倍数，$L^2$ 距离会感知到这个差异，余弦距离不会。

!!! warning "余弦距离严格来说不是度量"
    $d_{\cos}$ 不满足三角不等式，严格意义上不是度量，而是伪度量（pseudometric）。在需要三角不等式的算法（如 HNSW）里，一般用 $\arccos(\text{sim}_{\cos})$（角度距离）来代替，后者是合法的度量。

### 三种距离的形状对比

在 2D 里，以原点为中心、"距离 = 1"的所有点构成的形状：

| 度量 | 单位球形状 |
|------|-----------|
| $L^2$ 欧氏 | 圆 |
| $L^1$ 曼哈顿 | 菱形（正方形斜放 45°） |
| $L^\infty$ | 正方形（轴对齐） |

---

## 为什么嵌入空间用余弦距离

神经网络学到的特征向量（embedding）生活在高维空间里。检索系统通常用余弦距离而非 $L^2$ 距离，原因在于训练方式：对比学习（如 CLIP）在 loss 计算前先对特征做 $L^2$ 归一化，归一化后内积直接等于余弦相似度。

归一化消除了幅度信息——两个语义相同但激活幅度不同的特征向量，在 $L^2$ 距离下可能很远，但余弦距离会判定它们相似。当幅度本身不含语义信息时，余弦距离是更合理的选择。

---

## 代码验证

```python
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([4.0, 0.0, 1.0])

# L2 欧氏距离
d_l2 = np.linalg.norm(x - y)
print(d_l2)   # sqrt(9+4+4) = sqrt(17) ≈ 4.123

# L1 曼哈顿距离
d_l1 = np.linalg.norm(x - y, ord=1)
print(d_l1)   # 3 + 2 + 2 = 7.0

# 余弦相似度
cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print(cos_sim)  # (4+0+3)/(sqrt(14)*sqrt(17)) ≈ 0.454

# 关键等式验证：归一化后的 L2 距离平方 = 2 - 2 * cos_sim
x_n = x / np.linalg.norm(x)
y_n = y / np.linalg.norm(y)
print(np.linalg.norm(x_n - y_n) ** 2)  # ≈ 1.091
print(2 - 2 * cos_sim)                  # ≈ 1.091  <- 两者完全相等
```

```python
# 工程实践：用 FAISS 做余弦最近邻检索
# pip install faiss-cpu
import numpy as np, faiss

d, n = 128, 10000
db = np.random.randn(n, d).astype(np.float32)
faiss.normalize_L2(db)          # L2 归一化，使内积 = 余弦相似度

index = faiss.IndexFlatIP(d)    # 内积索引
index.add(db)

query = np.random.randn(1, d).astype(np.float32)
faiss.normalize_L2(query)
scores, indices = index.search(query, 5)
print(scores)   # 最近邻的余弦相似度，范围 [-1, 1]
```

!!! tip "在深度学习中的应用"
    - **向量检索（ANN）**：FAISS 支持 $L^2$ 和内积两种索引，归一化后内积 = 余弦检索。工业界图文检索一般用余弦距离。
    - **HNSW 算法**：目前最主流的近似最近邻算法，依赖三角不等式来剪枝搜索路径，因此要用满足三角不等式的距离。
    - **RAG 文档检索**：文档 chunk 和 query 的相似度几乎全部用余弦距离，因为文本长度（幅度）不应影响语义相似度。

!!! note "本节结论在后面的用处"
    理解余弦距离的几何含义是读懂**对比学习**（Part 3 现代网络结构）的前提。度量的三角不等式在分析 SVD 近似误差时也会用到。
