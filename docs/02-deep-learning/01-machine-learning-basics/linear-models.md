---
level: basic
roles:
  - core
  - interview
prerequisites:
  - 学习问题的基本形式
estimated_time: 35min
status: complete
---

# 线性模型

!!! info "参考资料"
    - [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) — 第 2、3、4 章
    - [Linear Regression](https://developers.google.com/machine-learning/crash-course/linear-regression) 与 [Logistic Regression](https://developers.google.com/machine-learning/crash-course/logistic-regression) — Google ML Crash Course

## 直觉 (Intuition)

线性模型给每个特征分配一个权重，再把结果相加。它表达能力有限，却把预测函数、损失、正则化和决策边界这些概念都摆在了明面上。神经网络中的线性层仍然做同一件事，只是前后多了非线性变换。先把线性模型吃透，后面的符号会轻松很多。

## 从一个加权和开始

输入 $\mathbf{x}\in\mathbb R^d$ 有 $d$ 个特征。线性模型的打分是

$$
z=\mathbf w^\top\mathbf x+b,
$$

其中 $\mathbf w\in\mathbb R^d$ 是权重，$b\in\mathbb R$ 是偏置。每个 $w_j$ 表示在其他特征不变时，第 $j$ 个特征增加一个单位会让打分改变多少。

“线性”描述的是参数和特征的组合方式，不代表模型只能画穿过原点的直线。偏置 $b$ 允许边界平移；加入 $x_1x_2$、$x_1^2$ 等人工特征后，模型对新特征仍然是线性的，但对原始输入已经能表示曲线。

## 线性回归

回归希望输出连续数值。最常见的做法是直接令预测值 $\hat y=z$，再用均方误差：

$$
\mathcal L(\mathbf w,b)
=\frac{1}{n}\sum_{i=1}^{n}
\left(\mathbf w^\top\mathbf x_i+b-y_i\right)^2.
$$

平方把正负误差都变成惩罚，也会放大大误差。若把偏置并入参数、将样本排成设计矩阵 $\mathbf X$，目标可以写成 $\|\mathbf X\mathbf w-\mathbf y\|_2^2$。对参数求导并令梯度为零：

$$
\mathbf X^\top(\mathbf X\mathbf w-\mathbf y)=0.
$$

当 $\mathbf X^\top\mathbf X$ 可逆时，得到闭式解：

$$
\mathbf w^*=(\mathbf X^\top\mathbf X)^{-1}\mathbf X^\top\mathbf y.
$$

实际计算通常不会显式求逆，而会用更稳定的线性方程求解或迭代优化。这里推导的意义是看清楚：最优参数取决于特征之间的相关性和特征与目标的相关性。

## 逻辑回归

二分类需要输出 $0$ 到 $1$ 之间的概率。逻辑回归先计算线性打分，再用 sigmoid 压到概率区间：

$$
p(y=1\mid\mathbf x)=\sigma(z)
=\frac{1}{1+e^{-z}}.
$$

它的名字里有“回归”，用途却是分类。因为模型拟合的是对数几率 (log-odds)：

$$
\log\frac{p(y=1\mid\mathbf x)}{1-p(y=1\mid\mathbf x)}
=\mathbf w^\top\mathbf x+b.
$$

给定二分类标签 $y\in\{0,1\}$，伯努利分布的负对数似然正好得到二元交叉熵：

$$
\ell
=-y\log p-(1-y)\log(1-p).
$$

所以逻辑回归不是先拍脑袋选 sigmoid，再随便配一个损失。概率模型和极大似然把两者连在了一起。交叉熵的统一写法和数值稳定实现留到[损失函数](../02-neural-network-foundations/loss-functions.md)。

## 正则化限制模型自由度

当特征多、样本少或特征高度相关时，模型可能用很大的权重贴合训练噪声。L2 正则化在经验风险后加一个平方范数：

$$
\mathcal J(\mathbf w,b)
=\mathcal L(\mathbf w,b)+\lambda\|\mathbf w\|_2^2.
$$

$\lambda>0$ 控制惩罚强度。它越大，权重通常越小，模型越平滑；太大则会欠拟合。L1 正则化使用 $\|\mathbf w\|_1$，常把一部分权重压到零，可用于稀疏特征选择。

!!! tip "面试重点"
    逻辑回归的决策边界仍是 $\mathbf w^\top\mathbf x+b=0$，所以它是线性分类器。sigmoid 改变的是分数到概率的映射，没有把线性边界变成曲线。

## 什么时候先试线性模型

线性模型训练快、容易解释，也是检查数据管道的好基线。如果复杂模型只比它好一点，先检查特征、标签和划分，而不是急着增加层数。它的短板也清楚：没有人工特征或非线性映射时，无法拟合异或这类非线性关系。

下一章从标签语义出发，区分[分类与回归](./classification-regression.md)。
