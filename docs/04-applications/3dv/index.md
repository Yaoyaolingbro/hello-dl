# 4.2 3D 视觉与空间智能

从多张 2D 图片重建 3D 世界。这一章按技术演进排列：经典几何 → 神经渲染 → 前向模型 → 3D 生成。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [相机模型与 Plücker Embedding](camera-embedding.md) | 内外参、投影变换、Plücker 坐标 | 几何变换 |
| [MVS 与 SfM](mvs.md) | 多视图重建、点云、经典几何管线 | 相机模型 |
| [SLAM](slam.md) | 视觉 SLAM、ORB-SLAM、神经 SLAM | 相机模型 |
| [Gaussian Splatting](gaussian-splatting.md) | 3D/4D GS、稀疏化、可微光栅化 | NeRF |
| [Feed-Forward 重建](feed-forward.md) | Dust3r、Cut3r、VGGT 端到端重建 | NeRF、Transformer |
| [后向优化方法](optical-methods.md) | RAFT 及迭代优化框架 | CNN |
| [三维模型生成](3d-generation/index.md) | TRELLIS、文本/图像驱动的 3D 生成 | 扩散模型、3DGS |
| [多模态大模型在 3D 中的应用](multimodal-3d.md) | 3D-LLM、SpatialVLM | VLM |
