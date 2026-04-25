# 三维视觉

从多张 2D 图片重建 3D 世界。这一章按技术演进顺序排列：先讲经典的几何方法，再到神经渲染，最后到最新的端到端前向模型。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [相机模型与 Plücker Embedding](camera-embedding.md) | 内外参、投影变换、Plücker 坐标作为神经网络输入 | 几何变换 |
| [多视图重建](mvs.md) | SfM、MVS、点云、经典几何管线 | 相机模型 |
| [NeRF](nerf.md) | 神经辐射场、体渲染方程 | MLP、概率论 |
| [3D Gaussian Splatting](3dgs.md) | 高斯点云表示、可微光栅化 | NeRF |
| [Feed-Forward 3D 重建](feed-forward.md) | 以 VGGT 为代表的端到端前向模型 | NeRF、Transformer |
