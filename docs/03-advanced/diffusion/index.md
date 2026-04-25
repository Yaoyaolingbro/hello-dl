# 扩散模型

从加噪到去噪，扩散模型是当前图像/视频生成的主流方法。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [DDPM](ddpm.md) | 前向加噪过程、反向去噪、训练目标 | VAE、高斯分布 |
| [Score Matching](score-matching.md) | 分数函数视角、朗之万采样 | DDPM |
| [DDIM](ddim.md) | 确定性采样、加速推理 | DDPM |
| [Classifier-Free Guidance](cfg.md) | 条件生成、CFG 权重 | DDPM |
