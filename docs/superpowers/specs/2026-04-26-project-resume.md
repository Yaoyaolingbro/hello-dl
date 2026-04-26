---
title: 项目恢复说明
date: 2026-04-26
updated: 2026-04-26
status: active
purpose: resume-context
---

# 项目恢复说明

> 这份文件的作用不是设计新功能，而是给下一次对话的模型快速恢复上下文。
> 如果要继续写教程、改结构或修文档，先读这份文件，再读后面列出的入口文件。

---

## 项目是什么

这是一个面向 AI 初学者的中文教程项目，目标是系统讲清楚：

- Part 1：数学基础
- Part 2：基础深度学习
- Part 3：深入深度学习
- Part 4：现代 AI 应用

站点用 MkDocs Material 构建，部署到 GitHub Pages。

目标读者主要有两类：

- 有大学数学基础、但还没真正把数学用到 AI 里的学生
- 想补科研直觉和工程表达能力的 AI 初学者 / 求职者

项目核心追求不是“堆知识点”，而是把知识点之间的依赖关系讲清楚，让读者能从数学一路走到现代模型与应用。

---

## 仓库信息

- 本地路径：`/Users/yaoyaoling/Desktop/博士生资料/deep learning/`
- 远端仓库：`git@github.com:Yaoyaolingbro/deep-learing.git`
- 注意：仓库名拼写是 `deep-learing`，不是 `deep-learning`
- 当前主工作分支：`main`
- 辅助分支：`claude-code`

截至 `2026-04-26`：

- `main` 是后续继续用 Codex 工作的主分支
- `claude-code` 额外包含一版 `docs/01-math/linear-algebra/` 的扩写与 Markdown 渲染修正

如果下次要继续写通用内容，优先从 `main` 开始。
如果要参考那版线性代数扩写，可额外查看 `claude-code`。

---

## 技术栈

- `mkdocs-material`：站点主题与导航
- `pymdownx.arithmatex` + KaTeX CDN：数学公式渲染
- `mermaid2` + `pymdownx.superfences`：流程图 / 结构图
- `glightbox`：图片放大
- `git-revision-date-localized`：页面更新时间
- GitHub Actions：构建并发布到 `gh-pages`

常用验证命令：

```bash
cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"
mkdocs build --strict
```

---

## 项目结构

当前项目已经形成比较完整的目录骨架，但不同章节的内容成熟度不一致。
不要默认“文件存在 = 内容已经写完”。

### 顶层关键文件

- `mkdocs.yml`：站点配置与 `nav`
- `writing-style.md`：通用写作规范
- `chapter-writing-workflow.md`：每次开写前给模型的工作流 prompt
- `reference.md`：补充资源
- `docs/superpowers/`：规范、spec、写作辅助文件；这些文件不进入站点导航

### `docs/` 下的主结构

- `docs/01-math/`
  - `linear-algebra/`
  - `geometry/`
  - `probability/`
  - `optimization/`
- `docs/02-deep-learning/`
  - 神经网络、反向传播、CNN、RNN、Transformer、GNN
- `docs/03-advanced/`
  - `modern-architectures/`
  - `generative-models/`
  - `imitation-learning/`
  - `rl/`
- `docs/04-applications/`
  - `perception/`
  - `3dv/`
  - `audio/`
  - `retrieval/`
  - `generation/`
  - `multimodal/`
  - `agent/`
  - `embodied/`
  - `mlsys/`

### 当前值得特别注意的部分

- Part 1 的目录已经比较完整，其中 `linear-algebra/` 在 `claude-code` 分支上有一版较大扩写
- Part 2 有完整骨架，但不同章节完成度差异明显
- Part 3 与 Part 4 的目录范围已经铺开，很多章节是待深入写作的主题卡位
- `docs/superpowers/specs/` 下现有文件更多是内部说明，不是面向站点读者的正文

---

## 当前工作流

每次开始编写章节前，优先读取：

1. `docs/superpowers/specs/2026-04-26-project-resume.md`
2. `chapter-writing-workflow.md`
3. `writing-style.md`
4. 对应 Part 的风格文件

Part 风格文件：

- Part 1：`docs/superpowers/style-part1-math.md`
- Part 2：`docs/superpowers/style-part2-dl-basics.md`
- Part 3：`docs/superpowers/style-part3-advanced.md`
- Part 4：`docs/superpowers/style-part4-applications.md`

### 工作流里新增的关键要求

在真正写某一章之前，模型必须先看一遍 `docs/` 下的 Markdown 目录树，但排除 `docs/superpowers/`。
目的不是数文件，而是先建立整本书的上下文，明确：

- 这章在全书里的位置
- 它依赖哪些前置章节
- 后续哪些章节会用到它

推荐命令已经写进 `chapter-writing-workflow.md`：

```bash
cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"
tree docs -P "*.md" -I "superpowers|assets|.obsidian"
```

如果没有 `tree`，则用：

```bash
cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"
find docs -path "docs/superpowers" -prune -o -path "docs/assets" -prune -o -name "*.md" -print | sort
```

---

## 写作原则

通用原则以 `writing-style.md` 为准，这里只提恢复上下文时最重要的几点：

- 直觉先于定义
- 每个新符号第一次出现时必须解释
- 每个公式前先说一句它在讲什么
- 正文不要写得像 AI 生成文案
- 面向初学者，但不能牺牲数学与工程上的严谨性

### MkDocs / Markdown 相关注意事项

- 无序列表前后要留空行
- 需要明确分段时，要留空行
- `superpowers` 目录下的文档可以参与构建，但不应该加入站点 `nav`

---

## 当前已发生的重要改动

最近一轮工作里，发生过这些关键变化：

- `main` 上补强了 `chapter-writing-workflow.md`
  - 写章节前先查看 `docs/` 目录树
  - 先总结章节在整本书里的位置与依赖关系
- `main` 上更新了 `writing-style.md`
  - 补充了 Markdown 渲染相关注意事项
- `main` 上删除了两个未继续使用的 stub：
  - `docs/03-advanced/rl/grpo.md`
  - `docs/04-applications/perception/pose-estimation.md`
- `main` 上移除了 Part 4 中暂不保留的两个章节域：
  - `docs/04-applications/ai4science/`
  - `docs/04-applications/quant/`
  - 同时清理了 `mkdocs.yml` 与 `docs/04-applications/index.md` 中对应入口
- `claude-code` 上保留了一版 `linear-algebra` 扩写稿及其 Markdown 修正

如果下次恢复对话时看到 `main` 和 `claude-code` 内容不完全一致，这通常是正常现象，不是工作区冲突。

---

## 下次恢复对话时建议怎么做

如果下次的任务是“继续写某一章”，推荐顺序：

1. 读这份 `project-resume`
2. 读 `chapter-writing-workflow.md`
3. 读 `writing-style.md`
4. 读对应 Part 的风格文件
5. 看 `docs/` 的 Markdown 目录树，排除 `superpowers`
6. 读目标章节所在目录的 `index.md`
7. 再读目标章节本身，以及前后相邻章节

如果下次的任务是“改站点结构 / 配置”，推荐顺序：

1. 读这份 `project-resume`
2. 读 `mkdocs.yml`
3. 看 `docs/` 实际目录结构
4. 再决定是否调整 `nav`

如果下次的任务是“恢复分支背景”，推荐顺序：

1. 看 `git branch -vv`
2. 看最近几次 `git log --oneline --decorate --all`
3. 确认当前是否在 `main`

---

## 恢复上下文时最值得先读的文件

这些文件是下次最可能需要重新加载的最小集合：

- `docs/superpowers/specs/2026-04-26-project-resume.md`
- `chapter-writing-workflow.md`
- `writing-style.md`
- `mkdocs.yml`
- `docs/01-math/index.md`
- `docs/02-deep-learning/index.md`
- `docs/03-advanced/index.md`
- `docs/04-applications/index.md`

如果要继续某个 Part，再补读对应的 Part 风格文件和该目录下的 `index.md`。

---

## 一句话总结

这是一个已经有完整章节骨架、正在逐步把内容写深写实的 AI 教程项目；下次恢复对话时，先建立全书级上下文，再开始写具体章节，不要直接埋头改某一个文件。
