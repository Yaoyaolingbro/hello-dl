# MAE：掩码自编码器

> 本节待编写。参考论文：He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.

!!! info "参考资料"
    **主要论文**
    - [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) — He et al., CVPR 2022

    **相关工作**
    - BERT（NLP 掩码预训练的先驱）
    - BEiT（将 BERT 思路迁移到图像）

## 直觉 (Intuition)

待补充。核心问题：如何设计视觉自监督预训练，使其兼具高效性和高质量表征？

MAE 的回答：随机遮掩图像 75% 的 patch，只让 encoder 看可见部分，再用轻量 decoder 重建像素。极高的掩码率迫使模型学习语义而非纹理。

## 问题背景

待补充。介绍 BEiT / SimMIM 等前置工作的局限，MAE 出现时解决了什么问题。

## 方法推导

待补充。

### 非对称 Encoder-Decoder 结构

- Encoder：标准 ViT，**只处理可见 patch**（约 25%）
- Decoder：轻量 Transformer，输入可见 patch 表征 + mask token，重建全部 patch 的像素值

### 掩码策略

待补充。随机均匀采样 vs 结构化掩码的对比。

### 重建目标

待补充。像素级 MSE vs. 归一化像素值 vs. dVAE token（BEiT 路线）。

## 关键设计决策

待补充。原论文 Table 1–4 的消融：掩码率、decoder 深度、重建目标对下游任务的影响。

## 代码（可选）

```python
# 核心：随机掩码采样（保留 visible_ratio 比例的 patch）
# 待补充
```

## 局限与后续工作

待补充。像素重建 vs. 特征重建的讨论；引出 JEPA 路线的动机。
