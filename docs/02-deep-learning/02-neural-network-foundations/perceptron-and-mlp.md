---
level: basic
roles:
  - core
  - interview
prerequisites:
  - 线性模型
  - 分类与回归
estimated_time: 35min
status: complete
---

# 感知机与多层感知机

!!! info "参考资料"
    - [Understanding Deep Learning](https://udlbook.github.io/udlbook/) — Simon J. D. Prince，第 3、4 章
    - [Deep Learning](https://www.deeplearningbook.org/) — Goodfellow、Bengio、Courville，第 6 章

## 直觉 (Intuition)

线性模型只能用一个超平面切分空间。多层感知机 (Multilayer Perceptron, MLP) 把多个线性变换和非线性函数交替叠起来，让前一层先改写表示，后一层再做判断。隐藏单元不是预先命名的特征，而是训练时学出来的中间坐标。层数增加的是函数组合能力，不是简单地多做几次线性回归。

## 从感知机开始

一个二分类感知机先计算线性打分

$$
z=\mathbf w^\top\mathbf x+b,
$$

再用阶跃函数决定类别。它能解决线性可分问题，却无法表示异或 (XOR)：四个点中，对角点同类，没有一条直线能把它们分开。

MLP 用隐藏层先构造新特征。单隐藏层网络可以写成

$$
\mathbf h=\phi(\mathbf W_1\mathbf x+\mathbf b_1),
\qquad
\mathbf z=\mathbf W_2\mathbf h+\mathbf b_2.
$$

$\mathbf W_1\in\mathbb R^{d_h\times d_{in}}$ 把输入映射到 $d_h$ 维隐藏表示，$\phi$ 逐元素施加非线性，$\mathbf W_2\in\mathbb R^{d_{out}\times d_h}$ 产生输出分数。

## 没有激活函数会怎样

若去掉 $\phi$，两层网络变成

$$
\mathbf z
=\mathbf W_2(\mathbf W_1\mathbf x+\mathbf b_1)+\mathbf b_2
=(\mathbf W_2\mathbf W_1)\mathbf x
+(\mathbf W_2\mathbf b_1+\mathbf b_2).
$$

令 $\mathbf W'=\mathbf W_2\mathbf W_1$、$\mathbf b'=\mathbf W_2\mathbf b_1+\mathbf b_2$，它仍是一个线性层。无论堆多少层，都会折叠成一次仿射变换。非线性激活是深度真正产生表达能力的条件。

!!! tip "面试重点"
    “MLP 为什么需要激活函数”可以直接用上面的合并推导回答。不要只说“为了增加非线性”，要说明没有它时多层矩阵乘法仍等价于单层。

## 形状比公式更先出错

一批 $B$ 个样本组成 $\mathbf X\in\mathbb R^{B\times d_{in}}$。采用样本在行的约定时：

$$
\mathbf H=\phi(\mathbf X\mathbf W_1^\top+\mathbf b_1),
$$

输出形状是 $B\times d_h$，偏置沿 batch 维广播。写实现前先在纸上标形状，能提前发现大多数转置错误。

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)

x = torch.randn(16, 4)
logits = model(x)
print(logits.shape)  # torch.Size([16, 3])
```

输出叫 logits，因为它还没有变成概率。二分类、多分类和回归怎样连接损失，见[损失函数](./loss-functions.md)。参数怎样得到梯度，先从[计算图与自动微分](./computational-graphs.md)开始。
