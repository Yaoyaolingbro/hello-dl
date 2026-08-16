---
level: intermediate
roles:
  - core
prerequisites: []
estimated_time: 10min
status: complete
---

# 06 · 评估与排错

训练出一个模型不等于解决了问题。这里讨论指标选择、训练故障、数据漂移、效率和实验可信度。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|---|---|---|
| [指标与决策阈值](./metrics-and-thresholds.md) | 从混淆矩阵出发选择 Precision、Recall、F1、AUC 和阈值。 | 分类与回归、泛化与数据 |
| [训练故障诊断](./diagnosing-training.md) | 按损失不降、验证变差、梯度异常和吞吐不足组织排查路径。 | 模型训练 |
| [数据与分布漂移](./data-and-distribution-shift.md) | 区分数据质量问题、协变量漂移、标签漂移与概念漂移。 | 泛化与数据、指标与决策阈值 |
| [效率基础](./efficiency-basics.md) | 估算参数量、计算量和显存，并区分训练与推理的性能瓶颈。 | 模型架构 |
| [可复现实验](./reproducible-experiments.md) | 说明随机性、版本、配置、基线、消融和方差报告如何影响结论。 | Mini-batch 与训练循环、指标与决策阈值 |

## 与前后章节的关系

这一目录建立在前一阶段之上。概念已经在前文完整解释时，本目录只做必要提醒并链接原章。

学完后，继续回到 [Part 2 导读](../index.md) 查看下一阶段。
