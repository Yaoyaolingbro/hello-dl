---
title: AI 入门教程 · 设计文档
date: 2026-04-25
status: approved
---

# AI 入门教程设计文档

## 项目目标

面向大学数学入门水平（有基础线代和微积分）的 AI 新手，编写一份系统性的入门教程。
兼顾科研入门和工程面试两类读者。通过 MkDocs Material 管理，托管于 GitHub Pages。

## 目标读者

- 大学数学入门水平：学过线性代数和微积分，但没有在科研中深度使用过
- 希望看懂深度学习论文的在校学生
- 准备 AI 相关岗位面试的求职者

## 写作风格

见 `writing-style.md`。核心原则：直觉先行、穿插叙事、朴素直白、符号对齐论文。

---

## 项目结构

```
deep-learning/
├── mkdocs.yml
├── requirements.txt
├── reference.md                         # 全局参考资料汇总
├── writing-style.md                     # 写作规范（每章编写前必读）
├── docs/
│   ├── index.md                         # 首页 + 学习路线图
│   │
│   ├── 01-math/                         # Part 1: 基础数学
│   │   ├── index.md
│   │   ├── linear-algebra/
│   │   │   ├── index.md
│   │   │   ├── vectors-spaces.md        # 向量、内积、范数
│   │   │   ├── matrix-ops.md            # 矩阵运算、迹、行列式、秩
│   │   │   ├── special-matrices.md      # 对称/正定/正交/投影矩阵
│   │   │   ├── matrix-calculus.md       # 矩阵求导、Jacobian、Hessian
│   │   │   ├── chain-rule.md            # 链式法则矩阵形式（反向传播铺垫）
│   │   │   ├── eigenvalue.md            # 特征值分解、谱定理
│   │   │   └── svd.md                   # SVD、低秩近似、PCA 联系
│   │   ├── probability/
│   │   │   ├── index.md
│   │   │   ├── basics.md                # 概率公理、条件概率、贝叶斯公式
│   │   │   ├── random-variables.md      # 离散/连续随机变量、PDF/PMF/CDF
│   │   │   ├── distributions.md         # 高斯、伯努利、范畴、Beta、Dirichlet
│   │   │   ├── expectation-variance.md  # 期望、方差、协方差矩阵
│   │   │   ├── bayesian.md              # 贝叶斯推断、MLE vs MAP
│   │   │   └── information-theory.md    # 熵、交叉熵、KL 散度、互信息
│   │   └── optimization/
│   │       ├── index.md
│   │       ├── convex-basics.md         # 凸集、凸函数、Jensen 不等式
│   │       ├── first-order.md           # 梯度、方向导数、一阶最优性
│   │       ├── second-order.md          # Hessian、曲率、二阶条件
│   │       ├── gradient-descent.md      # GD/SGD、收敛性分析
│   │       ├── adaptive-methods.md      # Momentum、AdaGrad、Adam
│   │       └── lagrangian.md            # 拉格朗日乘子、KKT 条件
│   │
│   ├── 02-deep-learning/                # Part 2: 基础深度学习
│   │   ├── index.md
│   │   ├── gradient-descent.md          # 神经网络中的梯度下降
│   │   ├── backprop.md                  # 反向传播
│   │   ├── cnn.md                       # 卷积神经网络
│   │   ├── rnn.md                       # 循环神经网络、LSTM、GRU
│   │   └── transformer.md               # Transformer 完整讲解
│   │
│   ├── 03-advanced/                     # Part 3: 深入深度学习
│   │   ├── index.md
│   │   ├── ae-vae/
│   │   │   ├── index.md
│   │   │   ├── autoencoder.md
│   │   │   └── vae.md
│   │   ├── diffusion/
│   │   │   ├── index.md
│   │   │   ├── ddpm.md                  # Denoising Diffusion Probabilistic Models
│   │   │   ├── score-matching.md        # Score-based 视角
│   │   │   ├── ddim.md                  # 加速采样
│   │   │   └── cfg.md                   # Classifier-Free Guidance
│   │   └── rl/
│   │       ├── index.md
│   │       ├── rl-basics.md
│   │       ├── ppo.md
│   │       ├── dpo.md
│   │       └── grpo.md
│   │
│   └── 04-applications/                 # Part 4: 现代应用
│       ├── index.md
│       ├── 3dv/
│       │   ├── index.md
│       │   ├── nerf.md
│       │   ├── 3dgs.md
│       │   └── mvs.md
│       ├── computer-vision/
│       │   ├── index.md
│       │   ├── object-detection.md
│       │   ├── segmentation.md
│       │   └── vit.md
│       ├── multimodal/
│       │   ├── index.md
│       │   ├── clip.md
│       │   ├── vlm.md
│       │   └── llm-basics.md
│       ├── agent/
│       │   ├── index.md
│       │   ├── rag.md
│       │   ├── tool-use.md
│       │   └── multi-agent.md
│       └── embodied/
│           ├── index.md
│           └── manipulation.md
│
└── docs/assets/
    ├── css/custom.css
    └── js/katex.js
```

---

## 技术栈

| 工具 | 版本 | 用途 |
|---|---|---|
| mkdocs-material | 9.4.12 | 主题框架 |
| pymdownx.arithmatex + KaTeX | latest | 数学公式 |
| admonition + pymdownx.details | - | 折叠提示块 |
| pymdownx.superfences + mermaid2 | - | 流程图 |
| pymdownx.tabbed | - | 并排对比 |
| mkdocs-glightbox | latest | 图片放大灯箱 |
| git-revision-date-localized | - | 最后更新时间 |
| footnotes | - | 论文引用脚注 |

---

## 写作工作流

1. 开始每章前：读 `writing-style.md` → 搜优质 blog → 确认论文和符号
2. 章节结构：直觉段 → 穿插叙事（文字 + 公式 + 代码）
3. 完成后：在 `reference.md` 补充本章资源
