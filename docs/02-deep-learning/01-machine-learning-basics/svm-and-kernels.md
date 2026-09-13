---
level: basic
roles:
  - core
  - interview
prerequisites:
  - 线性模型
  - 优化基础
estimated_time: 35min
status: complete
---

# SVM 与核方法

!!! info "参考资料"
    - [Support-vector networks](https://doi.org/10.1007/BF00994018) — Cortes、Vapnik, *Machine Learning* 20, 1995
    - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html) — scikit-learn User Guide
    - [CS229 SVM notes](https://cs229.stanford.edu/notes_archive/cs229-notes3.pdf) — Stanford CS229

## 直觉 (Intuition)

一条直线也许能把训练样本分开，但可行的直线通常不止一条。支持向量机 (Support Vector Machine, SVM) 会选择离两类最近样本都更远的边界，给新样本留出余量。真实数据不能完美分开时，它允许一部分样本进入间隔甚至分错。核方法再把线性边界搬到一个更合适的特征空间里。

## 从分类边界到几何间隔

二分类标签记为 $y_i\in\{-1,+1\}$。线性打分函数为

$$
f(\mathbf x)=\mathbf w^\top\mathbf x+b.
$$

超平面 $f(\mathbf x)=0$ 是分类边界，$\mathbf w$ 是它的法向量。点 $\mathbf x$ 到边界的距离为

$$
\frac{|\mathbf w^\top\mathbf x+b|}{\|\mathbf w\|_2}.
$$

把打分和标签相乘，$y_if(\mathbf x_i)>0$ 表示分类正确。因为同时缩放 $\mathbf w,b$ 不会改变边界，我们可以固定最近样本的函数间隔为 1：

$$
y_i(\mathbf w^\top\mathbf x_i+b)\ge 1.
$$

此时两侧支持超平面之间的宽度是 $2/\|\mathbf w\|_2$。最大化间隔等价于最小化权重范数：

$$
\min_{\mathbf w,b}\ \frac12\|\mathbf w\|_2^2
\quad
\text{s.t.}\quad
y_i(\mathbf w^\top\mathbf x_i+b)\ge1.
$$

只有贴着间隔边界的样本会决定最终解，它们就是支持向量。离边界很远的样本继续向外移动，通常不会改变分类面。

## 软间隔与 hinge loss

噪声和重叠类别会让硬间隔约束无解。软间隔允许样本违反约束，并用 hinge loss 计价：

$$
\ell_i=\max\left(0,1-y_i f(\mathbf x_i)\right).
$$

一种常见的无约束写法是

$$
\min_{\mathbf w,b}
\frac{\lambda}{2}\|\mathbf w\|_2^2
+\frac1n\sum_{i=1}^{n}
\max(0,1-y_i f(\mathbf x_i)).
$$

$\lambda$ 控制边界平滑和训练违例之间的取舍。有些库改用参数 $C$ 乘在损失项前；在这种约定里，$C$ 越大越在意训练违例，正则化相对越弱。讨论超参数时先确认目标函数的写法，否则“大”和“小”很容易说反。

!!! tip "面试重点"
    分对不代表 hinge loss 为零。样本即使落在正确一侧，只要进入宽度为 1 的间隔，仍然会产生损失。

## 核技巧为什么可行

有些数据在原空间不能线性分开。设非线性映射 $\phi(\mathbf x)$ 把输入送到新的特征空间，分类器可以写成

$$
f(\mathbf x)=\mathbf w^\top\phi(\mathbf x)+b.
$$

SVM 的对偶问题只通过内积 $\phi(\mathbf x_i)^\top\phi(\mathbf x_j)$ 使用映射后的特征。如果能直接计算

$$
K(\mathbf x_i,\mathbf x_j)
=\phi(\mathbf x_i)^\top\phi(\mathbf x_j),
$$

就不必显式构造高维向量，这就是核技巧 (kernel trick)。RBF 核

$$
K(\mathbf x,\mathbf z)
=\exp(-\gamma\|\mathbf x-\mathbf z\|_2^2)
$$

让距离近的样本拥有较高相似度。$\gamma$ 太大时，每个样本只影响很小的邻域，边界容易弯得过头；太小时，模型接近过于平滑的边界。

## 使用时要注意什么

SVM 对特征尺度敏感。若一个特征范围是 $0$ 到 $10^6$，另一个只有 $0$ 到 $1$，距离和内积会被前者支配，所以通常先标准化。

核 SVM 的训练和预测成本会随样本数与支持向量数增长，大数据上未必合适。样本不多、维度较高、边界清楚时，它仍是很有竞争力的基线。若需要直接解释概率，还要注意 SVM 原始输出是间隔分数，概率往往来自额外校准。

下一章转向没有人工标签的情形：[无监督学习](./unsupervised-learning.md)。
