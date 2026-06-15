# 光流估计

!!! info "参考资料"
    **必读论文**

    - [Determining Optical Flow](https://doi.org/10.1016/0004-3702%2881%2990024-2) — Horn and Schunck, Artificial Intelligence 1981
    - [FlowNet: Learning Optical Flow with Convolutional Networks](https://openaccess.thecvf.com/content_iccv_2015/html/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.html) — Dosovitskiy et al., ICCV 2015
    - [PWC-Net: CNNs for Optical Flow Using Pyramid, Warping, and Cost Volume](https://openaccess.thecvf.com/content_cvpr_2018/html/Sun_PWC-Net_CNNs_for_CVPR_2018_paper.html) — Sun et al., CVPR 2018
    - [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3526_ECCV_2020_paper.php) — Teed and Deng, ECCV 2020
    - [SEA-RAFT: Simple, Efficient, Accurate RAFT for Optical Flow](https://arxiv.org/abs/2405.14793) — Wang et al., ECCV 2024

## 直觉 (Intuition)

光流要估计同一个视觉点从第一帧移动到第二帧的二维位移。输入是相邻图像 $\mathbf{I}_1$ 和 $\mathbf{I}_2$，输出是每个像素的运动向量。难点是纹理重复、遮挡、光照变化和大位移会让“同一个点”难以匹配。现代方法通常先比较两帧特征，再反复修正整张位移场。光流描述的是图像上的表观运动，不一定等于物体在三维世界中的真实运动。

## 任务定义

对第一帧中的像素位置 $\mathbf{x}=(x,y)$，光流场

$$
\mathbf{f}(\mathbf{x})=(u(\mathbf{x}),v(\mathbf{x}))
$$

表示它在第二帧中移动到 $\mathbf{x}+\mathbf{f}(\mathbf{x})$。$u$ 和 $v$ 分别是水平与垂直位移。

常用端点误差 (Endpoint Error, EPE) 是预测向量与真实向量的欧氏距离：

$$
\operatorname{EPE}(\mathbf{x})=
\left\|\hat{\mathbf{f}}(\mathbf{x})-\mathbf{f}^{*}(\mathbf{x})\right\|_2.
$$

其中 $\hat{\mathbf{f}}$ 是预测光流，$\mathbf{f}^{*}$ 是真值。KITTI 还常报告异常像素比例，因为自动驾驶更关心少量巨大错误。

## 发展脉络

### Horn-Schunck：把匹配写成全局优化

如果一个像素在短时间内亮度不变，就有亮度恒常假设：

$$
I_xu+I_yv+I_t=0.
$$

其中 $I_x$、$I_y$ 和 $I_t$ 是图像在空间与时间上的导数。一个方程无法唯一解出 $u$、$v$ 两个未知量，这就是 aperture problem：只看局部边缘，无法判断沿边缘方向的运动。

Horn-Schunck（[Paper](https://doi.org/10.1016/0004-3702%2881%2990024-2)）加入全局平滑约束，让邻近像素倾向于具有相似运动。它奠定了经典光流的能量优化范式：数据项负责匹配，正则项负责补足歧义。

平滑假设在物体边界会失效，亮度恒常也会被反光、阴影和曝光变化破坏。后来的经典方法不断设计更鲁棒的数据项、边缘保持正则和粗到细优化。

### FlowNet：第一次把光流作为监督学习问题

经典算法依赖手工设计的匹配代价和优化过程。FlowNet（[Paper](https://openaccess.thecvf.com/content_iccv_2015/html/Dosovitskiy_FlowNet_Learning_Optical_ICCV_2015_paper.html) | [Project](https://lmb.informatik.uni-freiburg.de/people/dosovits/code.html)）用 CNN 直接从两帧图像预测光流，并比较了直接堆叠输入与显式相关操作两种架构。

真实光流真值难以采集，论文使用合成数据训练。这是一个重要转折：数据生成方式开始与网络架构同样重要。FlowNet 证明端到端学习可行，但跨到真实场景时仍有明显域差异。

### PWC-Net：把经典结构写进网络

直接在原分辨率搜索大位移会产生很大的匹配空间。PWC-Net（[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sun_PWC-Net_CNNs_for_CVPR_2018_paper.html) | [Project](https://github.com/NVlabs/PWC-Net)）重新引入三类经典思想：

- Pyramid：在低分辨率先估计大运动
- Warping：用当前光流对齐第二帧特征
- Cost Volume：显式保存局部候选匹配的相似度

这些模块都是可学习管线的一部分。PWC-Net 比大型堆叠网络紧凑，也说明深度学习并不要求丢掉几何和优化先验。

粗到细方法的弱点是，小物体和细结构可能在金字塔底部消失。第一次粗估错误后，后续 warping 也可能把特征拉到错误位置。

### RAFT：在固定高分辨率上反复查匹配

RAFT（[Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/3526_ECCV_2020_paper.php) | [Project](https://github.com/princeton-vl/RAFT)）先计算第一帧与第二帧所有特征位置之间的相关性，形成 all-pairs correlation volume。网络从零流场开始，用循环更新模块多次查询相关体并修正预测。

它不再依赖传统的逐层粗到细解码器。高分辨率流场在迭代中持续存在，因此边界和小结构更容易保留。它的代价是相关体占用显存，推理时间也随更新次数增加。

### SEA-RAFT：改进初值、损失与预训练

SEA-RAFT（[Paper](https://arxiv.org/abs/2405.14793) | [Project](https://github.com/princeton-vl/SEA-RAFT)）保留 RAFT 的迭代框架，但直接回归更好的初始光流，使用混合 Laplace 分布描述误差，并加入刚体运动预训练。它说明 RAFT 之后的提升不一定来自更复杂的 Transformer，训练目标、初值和数据先验仍能显著改变效率与泛化。

## 核心方法

### 相关体

两帧特征 $\mathbf{g}_1(\mathbf{x})$ 与 $\mathbf{g}_2(\mathbf{y})$ 的相似度可以写成点积：

$$
C(\mathbf{x},\mathbf{y})=
\mathbf{g}_1(\mathbf{x})^\top\mathbf{g}_2(\mathbf{y}).
$$

其中 $C$ 是匹配代价。局部 cost volume 只比较邻域，显存低但依赖当前估计；all-pairs correlation 比较所有位置，搜索范围大但更耗内存。

### Warping 与迭代修正

给定当前光流，可以把第二帧特征采样回第一帧坐标系。对齐后仍不一致的位置提示网络继续修正。RAFT 类方法把这个过程写成学习到的优化器：每一步读取匹配证据和上下文，再输出增量 $\Delta\mathbf{f}$。

### 遮挡

第一帧中的点可能在第二帧被挡住，此时不存在可见对应点。训练时若把遮挡区域强行套进亮度恒常损失，模型会得到错误监督。前后向一致性、显式遮挡头和鲁棒损失都用于缓解这个问题。

## 工程实践

### 帧间隔定义了任务难度

相机帧率降低或丢帧后，同一物体的位移会变大。只在短间隔数据上训练的模型，不能默认处理任意间隔。部署时要固定采样策略，并在运动速度分布上评测。

### 输入尺寸通常要满足网络步长

RAFT 等网络会多次下采样。输入宽高不满足步长倍数时，需要 padding，输出后再正确裁剪。直接 resize 会改变位移尺度，恢复原尺寸时必须同步缩放 $u$、$v$。

!!! tip "工程重点"
    光流可视化颜色图只适合快速检查方向和大小。下游任务需要保存浮点向量，不能把彩色 PNG 再解码成光流。

### 合成数据与真实数据要分开看

合成数据能提供稠密真值，但材质、运动和模糊分布与真实相机不同。报告结果时要区分 synthetic pretraining、真实数据微调和 zero-shot 测试。

## 开放问题

以下判断基于截至 2026 年 6 月公开的论文与项目资料。

- **遮挡和出视野区域仍缺少直接证据。** 模型只能依赖上下文推断，错误却常被普通 EPE 平均掉。
- **跨域泛化依赖训练数据。** 动画、驾驶、人体、显微和水下场景的运动与成像差异很大，单一模型很难同时稳定。
- **高分辨率与长位移仍消耗大量内存。** 相关体和多次更新在 4K 视频或端侧设备上成本明显。
- **二维光流不保证三维一致。** 相机运动、物体运动和深度共同产生图像位移。机器人与自动驾驶通常还需要场景流、深度或相机位姿。
- **评测与下游价值不完全一致。** 更低的平均 EPE 不一定带来更好的视频分割、插帧或跟踪，需要面向下游任务设计评测。
