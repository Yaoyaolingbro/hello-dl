---
level: intermediate
roles:
  - core
prerequisites: []
estimated_time: 10min
status: complete
---

# 05 · 模型架构

把前面的组件组合成完整模型，关注结构、数据流、训练特点和适用边界，而不是重复组件公式。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|---|---|---|
| [CNN 与 ResNet](./cnn-and-resnet.md) | 从卷积层、下采样和残差块组成视觉特征层级。 | 卷积、残差连接、归一化 |
| [RNN、LSTM 与 GRU](./rnn-lstm-gru.md) | 比较三种循环架构的状态路径、门控机制和序列限制。 | 循环状态、反向传播 |
| [Transformer](./transformer.md) | 从嵌入、注意力、前馈层、残差和归一化搭建完整 Transformer。 | 注意力、位置编码、残差连接、归一化 |
| [图神经网络](./graph-neural-networks.md) | 从消息传递层、读出和任务头组织 GNN，并解释过平滑。 | 图消息传递、归一化 |
| [如何选择架构](./architecture-selection.md) | 根据任务结构、数据规模、并行性、显存和延迟比较模型家族。 | CNN 与 ResNet、RNN、LSTM 与 GRU、Transformer、图神经网络 |

## 与前后章节的关系

这一目录建立在前一阶段之上。概念已经在前文完整解释时，本目录只做必要提醒并链接原章。

学完后，继续回到 [Part 2 导读](../index.md) 查看下一阶段。
