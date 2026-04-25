# 生成模型

生成模型的目标是学会数据的分布，进而采样出新样本。从 GAN 的对抗博弈，到扩散模型的逐步去噪，再到 Flow Matching 的直线轨迹——思路各异，但终点相同。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [GAN](gan.md) | 生成器/判别器对抗训练 | 神经网络基础 |
| [自编码器](ae.md) | 隐空间压缩与重建 | 神经网络基础 |
| [VAE](vae.md) | ELBO、重参数化技巧 | AE、信息论 |
| [DDPM & DDIM](ddpm-ddim.md) | 扩散过程、去噪目标、加速采样 | VAE、高斯分布 |
| [Flow Matching](flow-matching.md) | 向量场、ODE 轨迹、CFM | DDPM |
| [条件生成与 Cross-Attention](conditional-generation.md) | 文本/标签条件、CFG | DDPM |
