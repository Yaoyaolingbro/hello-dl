---
level: basic
roles:
  - core
prerequisites:
  - 学习问题的基本形式
estimated_time: 30min
status: complete
---

# 无监督学习

!!! info "参考资料"
    - [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) — 第 13、14 章
    - [Clustering](https://scikit-learn.org/stable/modules/clustering.html) 与 [Decomposing signals](https://scikit-learn.org/stable/modules/decomposition.html) — scikit-learn User Guide

## 直觉 (Intuition)

无监督学习没有现成标签，但并非没有目标。我们会自己规定什么叫“结构”：相近的点属于同一簇，少数方向解释大部分变化，或者低概率样本算异常。结果是否有意义，取决于这个目标是否符合任务。K-means 和 PCA 是两个最小模型，分别代表聚类与降维。

## K-means：围绕中心分组

给定样本 $\{\mathbf x_i\}_{i=1}^{n}$ 和簇数 $K$，K-means 为每个样本分配簇编号 $c_i$，并为每个簇学习中心 $\boldsymbol\mu_k$。它最小化簇内平方距离：

$$
\min_{\{c_i\},\{\boldsymbol\mu_k\}}
\sum_{i=1}^{n}
\left\|\mathbf x_i-\boldsymbol\mu_{c_i}\right\|_2^2.
$$

直接同时优化离散分配和连续中心很难，算法便交替做两步：

1. 固定中心，把每个样本分给最近的中心；
2. 固定分配，把中心更新为簇内样本均值。

每一步都不会让目标变差，算法最终停在一个局部最优解。不同初始化可能得到不同结果，因此实践中会运行多次并选择目标值更低的一次。

K-means 隐含了很强的几何偏好：欧氏距离要有意义，簇大致紧凑，规模和密度不要相差太夸张。两个月牙形簇在人眼中很清楚，K-means 仍可能从中间切开。

!!! warning "常见误区"
    簇编号没有语义。一次运行的“簇 0”和另一次运行的“簇 0”不必对应；真正需要比较的是分组结构或下游效果。

## PCA：保留变化最大的方向

主成分分析 (Principal Component Analysis, PCA) 寻找低维线性子空间，使投影后的数据保留尽可能多的方差。数据中心化后，第一主方向满足

$$
\mathbf u_1
=\arg\max_{\|\mathbf u\|_2=1}
\frac1n\sum_{i=1}^{n}
(\mathbf u^\top\mathbf x_i)^2.
$$

解是协方差矩阵最大特征值对应的特征向量，也可以通过 SVD 得到。这里不重复矩阵分解，细节见 Part 1 的 [SVD 与低秩近似](../../01-math/linear-algebra/svd.md)。

保留前 $r$ 个主方向后，样本被压缩为 $r$ 维坐标。PCA 适合去冗余、可视化和构造线性基线，但“方差大”不等于“对任务有用”。一个变化很小的特征也可能恰好决定类别。

## 没有标签时怎样评价

聚类可以看簇内紧凑度、簇间分离度或稳定性，但这些内部指标仍然绑定某种几何假设。降维可以看重构误差，却不能证明低维表示保留了语义。

更可靠的做法通常是回到用途：聚类能否帮助检索或人工分析，表示能否提升下游分类，异常检测能否在有限复核预算下找到真正风险。没有标签会让评价更难，而不是让评价消失。

## 到表示学习的桥

传统方法直接在人工特征上找结构。深度表示学习则先用神经网络学习映射 $\mathbf z=f_\theta(\mathbf x)$，再让 $\mathbf z$ 适合重构、对比、聚类或下游任务。PCA 可以看成线性表示学习，K-means 可以看成最简单的离散表示。

这条线会在 Part 3 的表征学习中继续。现在先完成机器学习基础里最容易被低估的一件事：[泛化与数据](./generalization-and-data.md)。
