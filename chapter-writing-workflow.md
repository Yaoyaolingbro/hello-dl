# 章节编写工作流

> 每次开始编写新章节前，把这个文件 @ 给模型。

---

## 你是谁，你在做什么

你在帮我维护一份面向 AI 入门者的中文教程，托管于 MkDocs Material + GitHub Pages。
项目路径：`/Users/yaoyaoling/Desktop/博士生资料/deep learning/`
远程仓库：`git@github.com:Yaoyaolingbro/deep-learing.git`（注意仓库名拼写）

写作风格规范在 `writing-style.md`，项目整体结构在 `docs/superpowers/specs/2026-04-25-ai-tutorial-design.md`。

---

## 选择写作风格文件（必做第一步）

根据你要编写的章节所在的 Part，**额外 @ 对应的风格文件**。这些文件不重复通用规则，只补充该 Part 特有的要求。

| Part | 额外 @ 文件 | 核心侧重 |
|------|------------|---------|
| Part 1 数学基础（`01-math/`） | `docs/superpowers/style-part1-math.md` | 直觉先于定义、NumPy 验证、定理框、禁用"显然" |
| Part 2 基础深度学习（`02-deep-learning/`） | `docs/superpowers/style-part2-dl-basics.md` | 每概念必须有 PyTorch 代码、注释说 WHY、消融式解释 |
| Part 3 深入深度学习（`03-advanced/`） | `docs/superpowers/style-part3-advanced.md` | 以论文为主线、推导不跳步、严格对齐原文符号、引用消融实验 |
| Part 4 现代 AI 应用（`04-applications/`） | `docs/superpowers/style-part4-applications.md` | 时间脉络叙事、价值优先于推导、真实工程 tip、必写开放问题 |

**示例：** 编写 `04-applications/3dv/nerf.md` 时，同时 @ `writing-style.md` 和 `docs/superpowers/style-part4-applications.md`。

---

## 开始编写前（必做）

1. **读 `writing-style.md`** — 重点看"直觉段模板"、"数学的写法四条规则"、"去除 AI 痕迹规则"
2. **读上表对应的 Part 风格文件** — 了解本 Part 的特殊要求和章节结构
3. **确认主要参考论文** — 在章节开头 `!!! info "参考资料"` 块里列出
4. **确认符号约定** — 所有公式符号与主要论文保持一致；有冲突时在 `!!! note` 里声明
5. 可以编写前联网搜索相关资料查找，但不要过于依赖。参考重要精华内容，讲出核心直觉和观点。

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

### 去除 AI 痕迹（对照 `writing-style.md` 中的检查项）
- [ ] 没有"此外"、"值得注意的是"、"不仅……而且……"
- [ ] 没有连续三个相同长度的句子
- [ ] 没有三段式列举（改成两项或四项）
- [ ] 没有宣传性语言（"充满活力的"、"令人叹为观止的"）
- [ ] 没有模糊归因（"专家认为"、"研究表明"）
- [ ] 段落结尾方式多样，不全是短金句

---

## 构建验证

每次完成一个文件后，运行：

```bash
cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"
mkdocs build 2>&1 | grep -E "(ERROR|WARNING(?!.*git-revision))"
```

只要没有 ERROR，git-revision-date 的 WARNING（新文件无 git log）可以忽略。

---

## Git 提交规范

```bash
cd "/Users/yaoyaoling/Desktop/博士生资料/deep learning"

# 查看改动
git diff --stat

# 提交（按章节提交，不要攒太多再提交）
git add docs/路径/文件.md
git commit -m "feat(章节标识): 添加 XXX 章节内容"

# 推送触发 GitHub Actions 自动部署
git push origin main
```

提交信息前缀约定：
- `feat(section)`: 新增章节内容
- `fix(section)`: 修正错误
- `refactor(section)`: 重构结构（不改内容）
- `chore`: mkdocs.yml、workflow 等配置改动

---

## 常见问题

**数学不渲染？**
检查 `docs/assets/js/katex.js` 是否存在，行内公式用 `$...$`，块级用 `$$...$$`。

**Mermaid 图不显示？**
代码块标记必须是 ` ```mermaid `，mkdocs.yml 里 superfences 的 tag 用 `!!python/name:mermaid2.fence_mermaid`（IDE 报 warning 是误报，运行时正常）。

**git-revision-date 报 WARNING？**
新文件在 commit 之前会报"no git logs"，提交后自动消失，不影响构建。

**部署后页面没更新？**
等 GitHub Actions 跑完（约 2 分钟），或检查仓库 Actions tab 有无报错。
