---
title: AI 入门教程 · 设计文档
date: 2026-04-25
updated: 2026-04-26
status: approved
---

# AI 入门教程设计文档

## 项目目标

面向大学数学入门水平（有基础线代和微积分）的 AI 新手，编写一份系统性的入门教程。
兼顾科研入门和工程面试两类读者。通过 MkDocs Material 管理，托管于 GitHub Pages。

远程仓库：`git@github.com:Yaoyaolingbro/deep-learing.git`（注意：仓库名拼写为 deep-learing）

## 目标读者

- 大学数学入门水平：学过线性代数和微积分，但没有在科研中深度使用过
- 希望看懂深度学习论文的在校学生
- 准备 AI 相关岗位面试的求职者

## 写作风格

见 `writing-style.md`。核心原则：直觉先行、穿插叙事、朴素直白、符号对齐论文、去除 AI 痕迹。

---

## 项目结构（当前实际状态，2026-04-26）

```
deep-learning/
├── mkdocs.yml
├── requirements.txt
├── reference.md
├── writing-style.md                     # 写作规范（每章编写前必读）
├── docs/
│   ├── index.md
│   │
│   ├── 01-math/                         # Part 1: 基础数学
│   │   ├── index.md
│   │   ├── linear-algebra/              # 9 节（纯代数）
│   │   │   ├── index.md
│   │   │   ├── vectors-spaces.md
│   │   │   ├── matrix-ops.md
│   │   │   ├── special-matrices.md
│   │   │   ├── matrix-calculus.md
│   │   │   ├── chain-rule.md
│   │   │   ├── eigenvalue.md
│   │   │   ├── svd.md
│   │   │   ├── complex-numbers.md
│   │   │   └── graph-laplacian.md
│   │   ├── geometry/                    # 3 节（空间几何，从 linear-algebra 拆出）
│   │   │   ├── index.md
│   │   │   ├── rigid-body-basics.md     # 叉积、SO(3)、刚体运动
│   │   │   ├── geometry-transforms.md   # 几何变换、相机模型
│   │   │   └── bezier-curves.md        # 贝塞尔曲线、B 样条
│   │   ├── probability/                 # 9 节
│   │   │   ├── index.md
│   │   │   ├── basics.md
│   │   │   ├── random-variables.md
│   │   │   ├── distributions.md
│   │   │   ├── expectation-variance.md
│   │   │   ├── bayesian.md
│   │   │   ├── information-theory.md
│   │   │   ├── stochastic-processes.md  # 随机过程、马尔科夫链
│   │   │   ├── sampling.md              # 蒙特卡洛、重要性采样
│   │   │   └── variational-inference.md # 变分推断、ELBO
│   │   └── optimization/               # 9 节
│   │       ├── index.md
│   │       ├── convex-basics.md
│   │       ├── first-order.md
│   │       ├── second-order.md
│   │       ├── gradient-descent.md      # GD / Mini-batch SGD / SGD 区别
│   │       ├── adaptive-methods.md      # Momentum、AdaGrad、Adam
│   │       ├── lagrangian.md            # 拉格朗日乘子、KKT
│   │       ├── admm.md                  # ADMM、近端梯度、LASSO
│   │       ├── interpolation.md         # 多项式插值、三次样条
│   │       └── odes-vector-fields.md    # ODE、向量场、数值求解器
│   │
│   ├── 02-deep-learning/               # Part 2: 基础深度学习（6 节）
│   │   ├── index.md
│   │   ├── neural-networks.md
│   │   ├── gradient-backprop.md
│   │   ├── cnn.md
│   │   ├── rnn.md
│   │   ├── transformer.md
│   │   └── gnn.md
│   │
│   ├── 03-advanced/                    # Part 3: 深入深度学习
│   │   ├── index.md
│   │   ├── modern-architectures/       # QFormer、UNet、ViT、DPT、等变网络
│   │   │   ├── index.md
│   │   │   ├── qformer.md
│   │   │   ├── unet.md
│   │   │   ├── vit.md
│   │   │   ├── dpt.md
│   │   │   └── equivariant.md
│   │   ├── generative-models/          # GAN、AE、VAE、DDPM/DDIM、Flow Matching、条件生成
│   │   │   ├── index.md
│   │   │   ├── gan.md
│   │   │   ├── ae.md
│   │   │   ├── vae.md
│   │   │   ├── ddpm-ddim.md
│   │   │   ├── flow-matching.md
│   │   │   └── conditional-generation.md
│   │   ├── imitation-learning/         # 行为克隆、DAgger、Diffusion Policy
│   │   │   ├── index.md
│   │   │   ├── bc.md
│   │   │   ├── dagger.md
│   │   │   └── diffusion-policy.md
│   │   └── rl/                         # RL 基础、PPO、DPO、GRPO
│   │       ├── index.md
│   │       ├── rl-basics.md
│   │       ├── ppo.md
│   │       ├── dpo.md
│   │       └── grpo.md
│   │
│   └── 04-applications/               # Part 4: 现代应用
│       ├── index.md
│       ├── 3dv/                        # 三维视觉
│       │   ├── index.md
│       │   ├── camera-embedding.md     # 相机模型与 Plücker Embedding
│       │   ├── mvs.md
│       │   ├── nerf.md
│       │   ├── 3dgs.md
│       │   ├── feed-forward.md         # VGGT 等前向重建
│       │   └── 3d-generation/          # 三维模型生成（新增）
│       │       ├── index.md
│       │       ├── trellis.md          # TRELLIS（结构化隐空间生成）
│       │       └── sam3d.md            # SAM3D（SAM 辅助 3D 理解）
│       ├── perception/                 # 传统感知
│       │   ├── index.md
│       │   ├── object-detection.md
│       │   ├── segmentation.md
│       │   ├── depth-estimation.md
│       │   └── pose-estimation.md
│       ├── multimodal/                 # 多模态大模型
│       │   ├── index.md
│       │   ├── decoding.md
│       │   ├── encoders/
│       │   │   ├── index.md
│       │   │   ├── clip.md
│       │   │   ├── blip.md
│       │   │   └── siglip.md
│       │   ├── vlm.md
│       │   ├── llm-basics.md
│       │   ├── cot.md                  # CoT 及其变种
│       │   └── lora.md
│       ├── agent/
│       │   ├── index.md
│       │   ├── agent-design.md         # ReAct、Reflexion 等设计范式
│       │   ├── rag.md
│       │   ├── tool-use.md
│       │   └── multi-agent.md
│       └── embodied/                   # 具身智能
│           ├── index.md
│           ├── classical-control/      # 传统运动控制（CS223A + Northwestern）
│           │   ├── index.md
│           │   ├── spatial-descriptions.md
│           │   ├── forward-kinematics.md
│           │   ├── jacobian.md
│           │   ├── inverse-kinematics.md
│           │   ├── trajectory-generation.md
│           │   ├── dynamics.md
│           │   ├── joint-space-control.md
│           │   └── operational-space-control.md
│           └── manipulation.md
│
└── docs/assets/
    ├── css/custom.css
    └── js/katex.js
```

---

## 技术栈

| 工具 | 用途 |
|---|---|
| mkdocs-material 9.4.12 | 主题框架，navigation.tabs 分 Part |
| pymdownx.arithmatex + KaTeX (unpkg CDN) | 数学公式渲染 |
| mermaid2 + pymdownx.superfences | 流程图（注意 YAML tag 为 `!!python/name:mermaid2.fence_mermaid`，IDE 误报可忽略） |
| mkdocs-glightbox | 图片放大灯箱 |
| git-revision-date-localized | 最后更新时间（需要完整 git 历史，fetch-depth: 0） |
| pymdownx.tabbed | 仅用于真正需要并排对比的场合 |
| GitHub Actions → gh-pages 分支 | 自动部署 |

---

## 写作工作流

见 `chapter-writing-workflow.md`（每次开始编写前 @ 该文件）。
