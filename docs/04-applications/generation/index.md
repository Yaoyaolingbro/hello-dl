# 图像与视频生成

从扩散模型到视频生成，从条件控制到强化学习驱动的生成，这一章梳理当前生成式 AI 的核心技术路线。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [SD 与 FLUX](sd-flux.md) | Latent Diffusion、SDXL、FLUX 架构 | DDPM、VAE |
| [SVD 视频生成](svd.md) | Stable Video Diffusion、时序建模 | SD、Transformer |
| [DiT 架构](dit.md) | Diffusion Transformer、Scalable Diffusion | ViT、DDPM |
| [编辑与定制化](editing-customization.md) | ControlNet、DreamBooth、LoRA | SD |
| [条件生成](conditional-generation.md) | Classifier-Free Guidance、文本/图像条件 | DDPM |
| [流式生成](streaming-generation.md) | 实时生成、渐进式解码 | SD |
| [生成模型做感知任务](generation-for-perception.md) | 生成模型作为先验的分割/深度/法线估计 | SD、感知 |
| [扩散模型上的强化学习](diffusion-rl.md) | DiffusionDPO、DDPO、奖励引导生成 | PPO、扩散模型 |
