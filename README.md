# 深度学习入门指南

这是一个面向 AI 初学者的中文教程项目，用 MkDocs Material 构建并部署到 GitHub Pages。

这份 README 的主要作用是帮助下一次接手项目时快速恢复上下文：项目在写什么、目录怎么组织、写章节前要读哪些文件、常用命令是什么。

---

## 项目定位

教程目标是系统讲清楚从数学基础到现代 AI 应用的关键知识链路。

目标读者主要是两类人：

- 有大学数学基础，但还没有真正把数学用到 AI 里的学生
- 想补科研直觉和工程表达能力的 AI 初学者 / 求职者

项目追求不是堆知识点，而是把依赖关系讲清楚。读者应该能知道一个概念从哪里来、后面会在哪里用到。

---

## 仓库信息

- 本地路径：`/Users/yaoyaoling/Desktop/博士生资料/deep learning/`
- 远端仓库：`git@github.com:Yaoyaolingbro/deep-learing.git`
- 注意：远端仓库名是 `deep-learing`，不是 `deep-learning`
- 站点框架：MkDocs Material
- 部署方式：GitHub Actions 构建到 `gh-pages`

当前常用分支：

- `main`：主工作分支，后续通用写作优先从这里开始
- `claude-code`：辅助参考分支，保留过一版 `docs/01-math/linear-algebra/` 扩写稿

如果要重新让 Codex 从 `main` 独立写作，建议从 `main` 新建干净分支，不要直接复制 `claude-code` 的正文。

---

## 快速恢复顺序

如果下次继续写某一章，建议按这个顺序读文件：

1. `README.md`
2. `docs/superpowers/specs/2026-04-26-project-resume.md`
3. `chapter-writing-workflow.md`
4. `writing-style.md`
5. 对应 Part 的风格文件：
   - Part 1：`docs/superpowers/style-part1-math.md`
   - Part 2：`docs/superpowers/style-part2-dl-basics.md`
   - Part 3：`docs/superpowers/style-part3-advanced.md`
   - Part 4：`docs/superpowers/style-part4-applications.md`
6. `docs/` 下的 Markdown 目录树，排除 `docs/superpowers/` 和 `docs/assets/`
7. 目标章节所在目录的 `index.md`
8. 目标章节正文，以及前后相邻章节

推荐目录树命令：

```bash
find docs -path "docs/superpowers" -prune -o -path "docs/assets" -prune -o -name "*.md" -print | sort
```

---

## 常用命令

安装依赖：

```bash
pip install -r requirements.txt
```

严格构建站点：

```bash
mkdocs build --strict
```

本地预览当前分支：

```bash
mkdocs serve
```

同时比较两个分支的站点：

```bash
scripts/compare-branches.sh
```

默认比较：

- `codex`：http://127.0.0.1:8010
- `claude-code`：http://127.0.0.1:8011

只编译不启动浏览器：

```bash
scripts/compare-branches.sh --build-only --no-open
```

---

## 顶层目录

```text
.
├── .github/                         # GitHub Actions 配置
├── docs/                            # 站点正文与内部写作资料
├── scripts/                         # 本地辅助脚本
├── tests/                           # 脚本和辅助工具的轻量测试
├── mkdocs.yml                       # MkDocs 配置与站点导航
├── requirements.txt                 # Python 依赖
├── chapter-writing-workflow.md      # 章节写作工作流
├── writing-style.md                 # 通用写作规范
├── reference.md                     # 补充资源
└── README.md                        # 项目恢复入口
```

生成目录不会提交：

- `site/`
- `.branch-previews/`
- `.worktrees/`

---

## docs 结构

```text
docs/
├── index.md
├── 01-math/
│   ├── index.md
│   ├── linear-algebra/
│   ├── geometry/
│   ├── probability/
│   └── optimization/
├── 02-deep-learning/
│   ├── index.md
│   ├── neural-networks.md
│   ├── gradient-backprop.md
│   ├── cnn.md
│   ├── resnet.md
│   ├── rnn.md
│   ├── transformer.md
│   └── gnn.md
├── 03-advanced/
│   ├── index.md
│   ├── modern-architectures/
│   ├── generative-models/
│   ├── imitation-learning/
│   └── rl/
├── 04-applications/
│   ├── index.md
│   ├── perception/
│   ├── 3dv/
│   ├── audio/
│   ├── retrieval/
│   ├── generation/
│   ├── multimodal/
│   ├── agent/
│   ├── embodied/
│   └── mlsys/
├── assets/
│   ├── css/custom.css
│   └── js/katex.js
└── superpowers/
    ├── specs/
    ├── plan/
    ├── style-part1-math.md
    ├── style-part2-dl-basics.md
    ├── style-part3-advanced.md
    └── style-part4-applications.md
```

`docs/superpowers/` 是内部资料目录。它可以参与本地构建，但不应该加入 `mkdocs.yml` 的 `nav`。

---

## 四个 Part 的定位

### Part 1：数学基础

路径：`docs/01-math/`

目标是让读者能看懂深度学习论文里的公式，而不是通过数学考试。

主要模块：

- `linear-algebra/`：向量、矩阵、矩阵求导、谱分解、SVD、傅里叶、图拉普拉斯
- `geometry/`：刚体运动、几何变换、Lie 群、曲线
- `probability/`：随机变量、分布、贝叶斯、信息论、采样、变分推断
- `optimization/`：凸优化、一阶/二阶条件、梯度下降、Adam、KKT、ADMM、ODE/PDE

### Part 2：基础深度学习

路径：`docs/02-deep-learning/`

目标是把神经网络训练的基本机制讲清楚。

覆盖神经网络、反向传播、CNN、ResNet、RNN、Transformer、GNN。

### Part 3：深入深度学习

路径：`docs/03-advanced/`

目标是连接现代论文中的模型结构和训练方法。

主要模块：

- `modern-architectures/`
- `generative-models/`
- `imitation-learning/`
- `rl/`

### Part 4：现代 AI 应用

路径：`docs/04-applications/`

目标是按任务和应用场景组织现代 AI 方法。

主要模块：

- 感知：`perception/`
- 3D 视觉与空间智能：`3dv/`
- 语音与音频：`audio/`
- 搜索、排序与信息检索：`retrieval/`
- 图像与视频生成：`generation/`
- 多模态大模型：`multimodal/`
- Agent：`agent/`
- 具身智能：`embodied/`
- 机器学习系统：`mlsys/`

---

## 写作规范

通用规范见 `writing-style.md`。最重要的规则：

- 直觉先于定义
- 每个新符号第一次出现时必须解释
- 每个公式前先用一句话说明它在讲什么
- 中文行文，英文术语第一次出现时括号标注原文
- 正文不要写得像 AI 生成文案
- 面向初学者，但不能牺牲数学与工程上的严谨性

数学章节的额外规范见 `docs/superpowers/style-part1-math.md`：

- 优先用 NumPy 验证数学性质
- 重要结论用 `!!! note` 写成定理框
- 不写“显然”“不难证明”“容易验证”
- 每节末尾说明这个结论后面会在哪里用到

---

## 章节结构约定

普通章节建议使用这个结构：

```markdown
!!! info "参考资料"
    **主要论文**

    - ...

    **优质讲解**

    - ...

## 直觉 (Intuition)

...

## 正文内容

...

## 代码验证 / 工程实现
（数学章节里非必要，不出现）
...

!!! note "本节结论在后面的用处"
    ...
```

目录下的 `index.md` 只做导读，不写正文内容。

---

## 构建与渲染注意事项

- 行内公式用 `$...$`
- 块级公式用 `$$...$$`
- 无序列表前后要留空行
- 需要明确分段时，中间留一个空行
- Mermaid 代码块使用 ` ```mermaid `
- `git-revision-date-localized` 对新文件可能有 git log 警告，提交后通常会消失

---

## 重要内部资料

- `docs/superpowers/specs/2026-04-26-project-resume.md`：最完整的项目恢复说明
- `docs/superpowers/specs/2026-04-25-ai-tutorial-design.md`：早期设计文档
- `chapter-writing-workflow.md`：每次写章节前的操作流程
- `writing-style.md`：通用写作规范
- `docs/superpowers/style-part*.md`：各 Part 的写作补充规范

---

## 当前注意事项

- 不要默认“文件存在 = 内容已经完成”。很多章节只是骨架或待深入写作。
- `main` 是后续继续工作的基准分支。
- `claude-code` 可以作为参考，但如果目标是独立重写，应先从 `main` 新建分支。
- 修改站点结构时，实际目录、`mkdocs.yml` 的 `nav`、对应 `index.md` 需要一起检查。
