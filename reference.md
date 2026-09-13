# 参考资料

本文件记录写作过程中实际使用的资料。技术结论优先依据教材、论文和官方资料；知乎、博客、课程与视频归入“解释参考”，用于寻找讲解角度和常见误区。

## 全书通用

### 教材

- Ian Goodfellow, Yoshua Bengio, Aaron Courville, *Deep Learning*.

### 解释参考

- 3Blue1Brown：线性代数的本质、神经网络入门。
- 李宏毅：《机器学习》《生成模型》课程。
- Maximilian Du：课程笔记，<https://maximiliandu.com/course_notes.html>。
- LabML：论文实现讲解，<https://github.com/labmlai/annotated_deep_learning_paper_implementations>。
- 《Hello 算法》：<https://www.hello-algo.com/>；[开源仓库](https://github.com/krahets/hello-algo)。参考其“具体问题—分步图解—机制说明”的讲解节奏，以及用连续画面和播放控件呈现算法状态变化的做法。原项目采用 CC BY-NC-SA 4.0；本书只借鉴思路，视觉素材和播放器自行实现。

## Part 1 · 数学基础

### 解释参考

- Quant Wiki，<https://quant-wiki.com/>。

## Part 2 · 深度学习基础

### 教材

- Simon J. D. Prince, *Understanding Deep Learning*.
- Aston Zhang et al., *Dive into Deep Learning*.

### 论文

- Corinna Cortes, Vladimir Vapnik. “Support-vector networks.” *Machine Learning* 20, 273–297 (1995). DOI: [10.1007/BF00994018](https://doi.org/10.1007/BF00994018)。
- Leo Breiman. “Random Forests.” *Machine Learning* 45, 5–32 (2001). DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)。
- Tianqi Chen, Carlos Guestrin. “XGBoost: A Scalable Tree Boosting System.” KDD 2016. arXiv: [1603.02754](https://arxiv.org/abs/1603.02754)。
- Tom M. Mitchell. “The Need for Biases in Learning Generalizations.” 1980；后收入 *Readings in Machine Learning*。
- Peter W. Battaglia et al. “Relational inductive biases, deep learning, and graph networks.” arXiv: [1806.01261](https://arxiv.org/abs/1806.01261), 2018。
- Xavier Glorot, Yoshua Bengio. “[Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html).” AISTATS 2010：Xavier/Glorot 初始化与深层信号传播。
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. “[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852).” ICCV 2015：整流激活与 He 初始化。
- Diederik P. Kingma, Jimmy Ba. “[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).” ICLR 2015：Adam、矩估计与偏差修正。
- Ilya Loshchilov, Frank Hutter. “[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).” ICLR 2019：AdamW 及 L2 与权重衰减的等价边界。
- Nitish Srivastava et al. “[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html).” *JMLR* 15(56), 2014。
- Sergey Ioffe, Christian Szegedy. “[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167).” ICML 2015。
- Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton. “[Layer Normalization](https://arxiv.org/abs/1607.06450).” 2016。
- Biao Zhang, Rico Sennrich. “[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467).” NeurIPS 2019。
- Shibani Santurkar et al. “[How Does Batch Normalization Help Optimization?](https://papers.neurips.cc/paper_files/paper/2018/hash/905056c1ac1dad141560467e0a99e1cf-Abstract.html).” NeurIPS 2018：检验 internal covariate shift 解释并研究优化平滑性。
- Vincent Dumoulin, Francesco Visin. “[A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285).” 2018：stride、padding、dilation 与卷积输出尺寸关系。
- Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. “[Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html).” CVPR 2016：残差映射与恒等 shortcut。
- Razvan Pascanu, Tomas Mikolov, Yoshua Bengio. “[On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063).” ICML 2013：循环网络中的梯度消失、爆炸与梯度裁剪。
- Ashish Vaswani et al. “[Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need).” NeurIPS 2017：缩放点积注意力、mask、正弦位置编码与复杂度。
- Jianlin Su et al. “[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864).” 2021：旋转位置编码及其相对位置点积形式。
- Justin Gilmer et al. “[Neural Message Passing for Quantum Chemistry](https://proceedings.mlr.press/v70/gilmer17a.html).” ICML 2017：消息生成、邻域聚合与状态更新的一般框架。
- Manzil Zaheer et al. “[Deep Sets](https://papers.nips.cc/paper/6931-deep-sets).” NeurIPS 2017：集合函数的置换不变性。

### 官方资料

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)：线性模型、分类、数据与泛化。
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)：传统模型、模型选择与评价。
- [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：动态图和自动微分行为。
- [PyTorch Module notes](https://docs.pytorch.org/docs/stable/notes/modules.html)：模块训练/评估模式与基础训练循环。
- [PyTorch Optimizers](https://docs.pytorch.org/docs/stable/optim.html)：优化器和学习率调度器的官方行为。
- [PyTorch `torch.nn.init`](https://docs.pytorch.org/docs/stable/nn.init)：ReLU/LeakyReLU gain 与 Kaiming 初始化的 `a`、`fan_in`/`fan_out` 参数约定。
- [PyTorch `CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)：交叉熵与将 one-hot 目标混合均匀分布的 `label_smoothing` 约定。
- [PyTorch `Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)：二维互相关、权重形状、`groups` 与输出尺寸约定。
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：缩放点积注意力和布尔/加性 mask 的接口约定。
- 框架行为和 API 以 PyTorch 官方文档为准。

### 解释参考

知乎、博客、课程和视频只有在实际用于某章时才加入，并标明对应主题。

## Part 3 · 深入深度学习

## Part 4 · 现代 AI 应用
