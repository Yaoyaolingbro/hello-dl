---
level: basic
roles:
  - core
prerequisites:
  - 感知机与多层感知机
estimated_time: 30min
status: complete
---

# 计算图与自动微分

!!! info "参考资料"
    - [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — PyTorch Documentation
    - [Deep Learning](https://www.deeplearningbook.org/) — 第 6.5 节

## 直觉 (Intuition)

一个损失函数看起来很长，实际由加法、乘法、指数等小操作拼成。计算图记录这些中间值和依赖关系。前向计算从输入得到损失，反向计算沿相反方向传递“损失对当前值有多敏感”。自动微分只是替我们执行链式法则，不是在做符号推导或有限差分。

## 把表达式拆成节点

看一个标量例子：

$$
a=wx,qquad z=a+b,qquad L=z^2.
$$

若 $x=2,w=3,b=1$，前向得到 $a=6,z=7,L=49$。反向从 $\partial L/\partial L=1$ 开始。每个节点只需要自己的局部导数：

$$
\frac{\partial L}{\partial z}=2z=14,
$$

$$
\frac{\partial L}{\partial a}
=\frac{\partial L}{\partial z}\frac{\partial z}{\partial a}
=14,
$$

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial a}\frac{\partial a}{\partial w}
=14x=28.
$$

反向过程不需要把整个复合函数重新展开。节点收到上游梯度，乘上局部导数，再把结果传给父节点。

## 分支处为什么要相加

若一个变量沿两条路径影响损失，例如 $L=u^2+3u$，总导数是两条路径贡献之和：

$$
\frac{\partial L}{\partial u}=2u+3.
$$

这也是 PyTorch 默认累积梯度的数学原因。一个参数在网络中被多次使用，各条路径产生的梯度必须相加。

## 自动微分和数值微分不同

有限差分用

$$
\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}
$$

近似导数，会有截断误差和浮点误差。自动微分对基本操作使用精确导数，再通过链式法则组合；误差只来自普通浮点计算。有限差分仍适合做 gradient check，但不适合日常训练。

```python
import torch

w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(2.0)
b = torch.tensor(1.0)
loss = (w * x + b) ** 2
loss.backward()

print(w.grad)  # tensor(28.)
```

`requires_grad=True` 表示需要记录与 $w$ 有关的操作；`backward()` 从标量损失反向计算梯度。中间图何时保存、梯度为何清零，会在[反向传播](../03-training/backpropagation.md)和[训练循环](../03-training/minibatch-and-training-loop.md)中说明。

!!! warning "常见误区"
    自动微分算的是当前执行路径的导数。包含条件分支时，没走到的分支不在这次计算图里；这不是漏算，而是动态图对应的函数就是本次实际执行的程序。
