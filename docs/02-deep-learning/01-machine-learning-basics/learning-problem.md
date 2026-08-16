---
level: basic
roles:
  - core
  - interview
prerequisites: []
estimated_time: 25min
status: complete
---

# 学习问题的基本形式

!!! info "参考资料"
    - [The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/) — Hastie、Tibshirani、Friedman
    - [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course) — Google for Developers

## 直觉 (Intuition)

机器学习不是“把数据喂给模型”这么简单。我们先要说清楚输入是什么、希望模型输出什么，以及犯不同错误要付出多大代价。训练只负责在已有样本上调整参数；真正的目标，是让模型碰到新样本时仍然可靠。后面出现的神经网络、树模型和 SVM，都能放进这套框架。

## 五个对象

设一条样本写成 $(\mathbf{x},y)$。其中 $\mathbf{x}\in\mathbb{R}^d$ 是 $d$ 维输入，$y$ 是我们关心的目标。一个监督学习问题至少包含下面这些对象：

| 对象 | 它回答的问题 |
|---|---|
| 数据分布 $p(\mathbf{x},y)$ | 现实中的样本怎样产生？ |
| 数据集 $\mathcal D$ | 我们实际看到了哪些样本？ |
| 模型 $f_\theta$ | 允许用什么函数做预测？ |
| 损失 $\ell$ | 怎样衡量一次预测有多糟？ |
| 评价规则 | 模型是否值得部署？ |

参数 $\theta$ 可以是两个线性系数，也可以是数十亿个神经网络权重。模型变复杂以后，这五个对象没有变。

### 训练集只是分布的一次抽样

理想情况下，我们关心模型在真实分布上的平均损失：

$$
R(\theta)
=
\mathbb E_{(\mathbf{x},y)\sim p}
\left[\ell(f_\theta(\mathbf{x}),y)\right].
$$

$R(\theta)$ 叫真实风险 (true risk)。问题是 $p(\mathbf{x},y)$ 通常未知，这个期望没法直接计算。我们手里只有 $n$ 个样本组成的训练集：

$$
\mathcal D_{\mathrm{train}}
=\{(\mathbf{x}_i,y_i)\}_{i=1}^{n}.
$$

于是训练时改为最小化样本平均损失，也就是经验风险 (empirical risk)：

$$
\hat R(\theta)
=
\frac{1}{n}\sum_{i=1}^{n}
\ell(f_\theta(\mathbf{x}_i),y_i).
$$

训练算法求的是

$$
\theta^*=\arg\min_\theta \hat R(\theta),
$$

但项目成败取决于 $R(\theta^*)$。训练误差很低，只能说明模型解释了训练集，不能证明它学到了可迁移的规律。

!!! tip "面试重点"
    经验风险最小化 (ERM) 优化的是训练集上的平均损失。泛化讨论的是经验风险和真实风险之间的差距。把两者说成同一个量，是常见的概念错误。

## 学习算法做了什么

把模型训练拆开看，会更容易定位问题：

1. 先规定模型族，例如所有线性函数或某种神经网络；
2. 用损失把预测质量变成一个标量；
3. 用优化算法在模型族里寻找参数；
4. 用没有参与参数更新的数据检查结果。

如果模型族根本表达不了目标关系，换优化器也救不了；如果损失和业务代价不一致，训练得越好可能偏得越远。机器学习工程中，很多问题发生在第一步和第二步，而不是优化代码里。

## 监督、无监督与自监督

监督学习直接给出目标 $y$，例如类别或房价。无监督学习只有输入 $\mathbf{x}$，目标来自数据结构本身，比如把相似样本聚到一起。自监督学习也不依赖人工标签，但会从数据中构造监督信号，例如遮住一句话中的词再让模型预测。

三者的差别主要在训练信号从哪里来。到了优化阶段，仍然会出现模型、目标函数和参数更新。

## 为什么要分训练、验证和测试

训练集用于更新参数，验证集用于选择模型、超参数和决策阈值，测试集只用于最后一次独立评估。只要根据测试结果改过方案，测试集就已经参与了选择，结果会变得乐观。

数据怎样划分、泄漏怎样发生，会在[泛化与数据](./generalization-and-data.md)中展开。下一章先用最简单的函数族看看这套框架怎样落地：[线性模型](./linear-models.md)。
