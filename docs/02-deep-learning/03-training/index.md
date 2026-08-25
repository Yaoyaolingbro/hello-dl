---
level: intermediate
roles:
  - core
prerequisites: []
estimated_time: 10min
status: complete
---

# 03 · 模型训练

模型给出预测，损失衡量误差；训练要做的，是把这一个标量沿计算图送回每个参数，再用一批批数据反复更新参数。本目录把这条链路接完整：反向传播负责算梯度，训练循环管理状态，初始化、优化器、正则化与归一化共同决定更新是否稳定、是否能泛化。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|---|---|---|
| [反向传播](./backpropagation.md) | 从计算图推导反向模式自动微分和向量—雅可比积。 | 计算图与自动微分、损失函数 |
| [Mini-batch 与训练循环](./minibatch-and-training-loop.md) | 建立 epoch、batch、step、梯度累积和训练/评估模式的完整闭环。 | 反向传播 |
| [参数初始化](./initialization.md) | 用方差传播理解 Xavier、He 初始化及初始化失败的症状。 | 反向传播、激活函数 |
| [优化器与学习率](./optimization.md) | 比较 SGD、Momentum、Adam、AdamW 与常见学习率策略。 | 反向传播、Mini-batch 与训练循环 |
| [正则化](./regularization.md) | 区分权重衰减、Dropout、数据增强、早停和标签平滑的作用。 | 泛化与数据、优化器与学习率 |
| [归一化](./normalization.md) | 按归一化轴和训练行为比较 BatchNorm、LayerNorm 与 RMSNorm。 | Mini-batch 与训练循环 |

## 与前后章节的关系

前一目录已经讲过[计算图](../02-neural-network-foundations/computational-graphs.md)、[损失](../02-neural-network-foundations/loss-functions.md)和[激活函数](../02-neural-network-foundations/activations.md)；这里从它们给出的局部导数出发，不再重复定义。学习率与随机梯度的数学背景可回看 Part 1 的[梯度下降](../../01-math/optimization/gradient-descent.md)和[自适应方法](../../01-math/optimization/adaptive-methods.md)。

读完后可以进入[可复用组件](../04-components/index.md)。若训练已经失败，应按症状到后面的[训练故障诊断](../06-evaluation-debugging/diagnosing-training.md)排查；这一目录只解释机制，不重复故障清单。
