# 章节编写工作流

> 每次开始编写新章节前，把这个文件 @ 给模型。

---

## 你是谁，你在做什么

你在帮我维护一份面向 AI 入门者的中文教程，托管于 MkDocs Material + GitHub Pages。
项目路径：`/Users/yaoyaoling/Desktop/博士生资料/deep learning/`
远程仓库：`git@github.com:Yaoyaolingbro/hello-dl.git`

写作风格规范在 `writing-style.md`，当前重写结构在
`docs/superpowers/specs/2026-08-16-mainline-rewrite-design.md`。

---

## 选择写作风格文件（必做第一步）

根据你要编写的章节所在的 Part，**额外 @ 对应的风格文件**。这些文件不重复通用规则，只补充该 Part 特有的要求。

| Part | 额外 @ 文件 | 核心侧重 |
|------|------------|---------|
| Part 1 数学基础（`01-math/`） | `docs/superpowers/style-part1-math.md` | 直觉先于定义、**无代码改用数值例子**、示意图必须插入、定理框、禁用"显然" |
| Part 2 基础深度学习（`02-deep-learning/`） | `docs/superpowers/style-part2-dl-basics.md` | 先展示可见变化、代码只在有助理解时使用、消融式解释 |
| Part 3 深入深度学习（`03-advanced/`） | `docs/superpowers/style-part3-advanced.md` | 以论文为主线、推导不跳步、严格对齐原文符号、引用消融实验 |
| Part 4 现代 AI 应用（`04-applications/`） | `docs/superpowers/style-part4-applications.md` | 时间脉络叙事、价值优先于推导、真实工程 tip、必写开放问题 |

**示例：** 编写 `04-applications/3dv/nerf.md` 时，同时 @ `writing-style.md` 和 `docs/superpowers/style-part4-applications.md`。

---

## 开始编写前（必做）

1. **先看 `docs/` 下的 Markdown 目录树（排除 `docs/superpowers/`）** — 先建立全书级上下文，只看 `.md` 文件即可。优先运行：

   ```bash
   cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"
   tree docs -P "*.md" -I "superpowers|assets|.obsidian"
   ```

   如果本机没有 `tree`，则改用：

   ```bash
   cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"
   find docs -path "docs/superpowers" -prune -o -path "docs/assets" -prune -o -name "*.md" -print | sort
   ```

   看完后，先写三行工作笔记：当前章节解决什么问题、依赖哪些章节、后续哪些章节会使用它。
   再搜索相邻章节是否已经解释过同一概念。已有完整解释时，只做简短提醒并链接原文。
2. **读 `writing-style.md`** — 重点看"直觉段模板"、"数学的写法四条规则"、"去除 AI 痕迹规则"
3. **读上表对应的 Part 风格文件** — 了解本 Part 的特殊要求和章节结构
4. **确认资料层级** — 教材、论文和官方资料用于核对事实；知乎、博客和视频用于寻找解释角度
5. **确认符号约定** — 所有公式符号与主要论文保持一致；有冲突时在 `!!! note` 里声明
6. **确认视觉任务** — 先写一句“这幅图让读者看见什么”。状态变化用 HTML/SVG 分步图，稳定关系用静态图；没有明确任务就不加图。
7. 把新资料记录到 `reference.md`。二手讲解中的公式、结论和版本信息必须回到权威来源核对。外部图片同时记录作者、来源和许可证。

---

## 章节标准结构

```markdown
!!! info "参考资料"
    **主要论文**

    - [论文标题](url) — 作者, 会议/期刊 年份

    **优质讲解**
    
    - 资源1
    - 资源2

## 直觉 (Intuition)

[问题是什么？]
[输入/输出各是什么？]
[核心思路一句话]
[可选：与已知概念类比]

## 正文内容

（文字 → 公式 → 代码，穿插叙事，不要分离）

!!! note "直觉小结"
    一句话总结这段推导的直觉含义

!!! tip "面试 / 工程重点"
    （点缀使用，不要每段都有）

!!! warning "常见误区"
    （只写真实存在的错误）
```

---

## 完成后的检查清单

### 内容质量
- [ ] 每个新符号第一次出现时有解释
- [ ] 每个公式前有一句话说明它在讲什么
- [ ] 直觉段不超过 5 句话
- [ ] 代码只写核心逻辑，注释说 WHY 不说 WHAT
- [ ] 术语第一次出现括号标注英文原文
- [ ] 每幅图都有学习目标，图注说明“看什么”
- [ ] 交互图有 `data-lesson-steps` 文字替代，并尊重“减少动态效果”
- [ ] 外部图片已记录来源与许可证，没有直接复制来路不明的素材

### 去除 AI 痕迹（对照 `writing-style.md` 中的检查项）
- [ ] 没有"此外"、"值得注意的是"、"不仅……而且……"
- [ ] 没有连续三个相同长度的句子
- [ ] 没有为了显得全面而强凑三项；事实本来有三项时不必改
- [ ] 没有宣传性语言（"充满活力的"、"令人叹为观止的"）
- [ ] 没有模糊归因（"专家认为"、"研究表明"）
- [ ] 段落结尾方式多样，不全是短金句

---

## 构建验证

每次完成一个文件后，运行内容检查；完成一个目录后，再做严格构建：

```bash
cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"
.venv/bin/python scripts/check_part2_content.py --root docs/02-deep-learning
.venv/bin/mkdocs build --strict
```

Part 2 全部完成后还要加 `--require-complete`，确保没有计划页或草稿页混入发布版本。

---

## Git 提交规范

```bash
cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"

# 查看改动
git diff --stat

# 提交（每个目录形成一个可独立检查的提交）
git add docs/02-deep-learning/目录 reference.md
git commit -m "docs(part2): 完成 XXX 目录"

# 推送触发 GitHub Actions 自动部署
git push origin main
```

提交信息前缀约定：
- `docs(section)`: 新增或重写章节内容
- `fix(section)`: 修正错误
- `refactor(section)`: 重构结构（不改内容）
- `chore`: mkdocs.yml、workflow 等配置改动

---

## 常见问题

**数学不渲染？**
检查 `docs/assets/js/katex.js` 是否存在，行内公式用 `$...$`，块级用 `$$...$$`。

**HTML/SVG 动画没有按钮？**
检查外层是否有 `data-lesson-visual`，舞台是否有 `data-lesson-stage`，步骤是否标记为 `data-step`。播放器脚本位于 `docs/assets/js/lesson-visuals.js`。

**git-revision-date 报 WARNING？**
新文件在 commit 之前会报"no git logs"，提交后自动消失，不影响构建。

**部署后页面没更新？**
等 GitHub Actions 跑完（约 2 分钟），或检查仓库 Actions tab 有无报错。
