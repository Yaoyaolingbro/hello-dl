---
level: intermediate
roles:
  - core
prerequisites: []
estimated_time: 10min
status: complete
---

# 03 · 模型训练

这一部分把损失变成可执行的参数更新，并解释训练循环中最容易出问题的环节。

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

这一目录建立在前一阶段之上。概念已经在前文完整解释时，本目录只做必要提醒并链接原章。

学完后，继续回到 [Part 2 导读](../index.md) 查看下一阶段。
