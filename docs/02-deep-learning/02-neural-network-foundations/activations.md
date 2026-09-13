---
level: basic
roles:
  - core
  - interview
prerequisites:
  - 感知机与多层感知机
estimated_time: 30min
status: complete
---

# 激活函数

!!! info "参考资料"
    - [Understanding Deep Learning](https://udlbook.github.io/udlbook/) — 第 3、4 章
    - [Non-linear activations](https://docs.pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) — PyTorch Documentation

## 直觉 (Intuition)

激活函数让多层线性变换无法再合并成一层。它也决定梯度怎样通过隐藏单元：某些区域几乎不传梯度，某些区域保持稳定，还有些函数用平滑换取更多计算。选择激活不是挑一条“形状好看”的曲线，而是在表达能力、梯度和计算成本之间取舍。

## Sigmoid 与 Tanh

Sigmoid 把实数压到 $(0,1)$：

$$
\sigma(x)=\frac{1}{1+e^{-x}},
\qquad
\sigma'(x)=\sigma(x)(1-\sigma(x)).
$$

当 $|x|$ 很大时，导数接近零。深层网络反复乘这些小导数，容易出现梯度消失。它适合二分类输出或门控，不再是隐藏层默认选择。

Tanh 输出 $(-1,1)$，且以零为中心：

$$
\tanh'(x)=1-\tanh^2(x).
$$

它仍会在两端饱和，但在循环网络的状态更新中很常见。

## ReLU 为什么常用

ReLU 定义为

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

正半轴导数为 1，不会像 sigmoid 那样随 $x$ 增大而趋近零；计算也很便宜。负半轴输出和梯度都是零，因此一个单元若长期落在负区间，可能再也激活不了，这叫 dying ReLU。

Leaky ReLU 给负半轴保留一个小斜率。GELU、SiLU 等平滑函数不会在零点突然折断，在 Transformer 等现代架构中很常见，但“更平滑”不保证在所有任务上更好。

| 激活 | 输出范围 | 梯度特点 | 常见位置 |
|---|---|---|---|
| Sigmoid | $(0,1)$ | 两端饱和 | 二分类输出、门控 |
| Tanh | $(-1,1)$ | 两端饱和、零中心 | 循环状态 |
| ReLU | $[0,\infty)$ | 正区稳定、负区为零 | CNN、MLP 隐藏层 |
| GELU/SiLU | 无界或近似无界 | 平滑 | Transformer、现代 MLP |

!!! tip "面试重点"
    ReLU 不能从数学上“解决”所有梯度消失。它只在正区间导数为 1；初始化、网络深度、归一化和残差路径仍会影响梯度。

输出层激活由任务决定：回归可能不加约束，二分类训练通常直接输出 logit，多分类交叉熵直接接收 logits。不要因为想看到概率，就把 softmax 写进模型后又交给包含 softmax 的损失。

激活是模型的一种结构假设。更广的结构假设会在[归纳偏置](./inductive-bias.md)中统一起来。
