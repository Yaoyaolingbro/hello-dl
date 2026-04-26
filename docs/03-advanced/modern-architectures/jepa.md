# JEPA：联合嵌入预测架构

> 本节待编写。参考论文：LeCun 2022 白皮书 + I-JEPA (Assran et al., CVPR 2023) + V-JEPA (Bardes et al., 2024)。

!!! info "参考资料"
    **核心论文**
    - [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf) — LeCun, 2022（理论框架）
    - [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) — Assran et al., CVPR 2023（I-JEPA）
    - [V-JEPA: Latent Video Prediction for Visual Representation Learning](https://arxiv.org/abs/2404.08471) — Bardes et al., 2024（V-JEPA）

## 直觉 (Intuition)

待补充。核心问题：MAE 重建像素，但像素里大量信息是冗余的纹理——能否让模型在**表征空间**里做预测，只学语义？

JEPA 的回答：在潜在空间中预测被掩码区域的表征（而非像素），用 target encoder（EMA 更新）提供预测目标，避免模式坍塌。

## 问题背景

待补充。对比三条自监督路线：
1. 生成式（MAE）：重建像素，计算代价高，目标含冗余
2. 对比式（SimCLR / CLIP）：需要负样本或大 batch
3. JEPA：在表征空间预测，无需负样本，目标语义化

## 方法推导

待补充。

### 联合嵌入架构 (Joint Embedding Architecture)

待补充。Context encoder + Target encoder（EMA）+ Predictor 三部分的数据流。

### I-JEPA：图像版本

待补充。块状掩码策略（与 MAE 随机掩码的区别），为什么块状掩码更难预测、迫使模型学全局语义。

### V-JEPA：视频版本

待补充。时序预测的扩展，如何在时空 patch 上应用 JEPA 框架。

## 关键设计决策

待补充。
- EMA target encoder 为何优于梯度更新（防止坍塌）
- Predictor 的容量选择：太强会 shortcut，太弱学不好
- 块状掩码 vs. 随机掩码的消融

## 局限与后续工作

待补充。JEPA 目前在纯视觉表征上表现出色，但与语言对齐的结合（如 CLIP 路线）还处于早期；世界模型方向（LeCun 愿景的下一步）仍是开放问题。
