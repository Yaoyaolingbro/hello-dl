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

### 官方资料

- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)：线性模型、分类、数据与泛化。
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)：传统模型、模型选择与评价。
- [PyTorch Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html)：动态图和自动微分行为。
- [PyTorch Module notes](https://docs.pytorch.org/docs/stable/notes/modules.html)：模块训练/评估模式与基础训练循环。
- [PyTorch Optimizers](https://docs.pytorch.org/docs/stable/optim.html)：优化器和学习率调度器的官方行为。
- 框架行为和 API 以 PyTorch 官方文档为准。

### 解释参考

知乎、博客、课程和视频只有在实际用于某章时才加入，并标明对应主题。

## Part 3 · 深入深度学习

## Part 4 · 现代 AI 应用
