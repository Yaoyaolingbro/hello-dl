# Part 2 主线教材重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Part 2 重写为一条从传统机器学习桥梁、神经网络基础到训练、架构、评估和面试复习的完整学习主线，并发布到现有 MkDocs 站点。

**Architecture:** Part 2 使用七个编号目录表达固定阅读顺序，每个目录的 `index.md` 建立局部知识地图，子章节只承担一个主题。YAML 元数据声明难度、内容角色、前置知识、阅读时间和完成状态；内容检查脚本负责约束元数据、内部链接和占位文本，MkDocs 严格构建负责验证导航与渲染。

**Tech Stack:** MkDocs 1.6、Material for MkDocs 9.6、Python 3、PyYAML、Python `unittest`、GitHub Actions、Markdown + KaTeX

## Global Constraints

- 保留 MkDocs Material 框架、四个 Part 和 Part 4 现有科研方向。
- Part 2 采用“主线教材 + 标签化面试/科研内容”。
- 行文使用自然、直接的中文，避免固定套话、机械总结和大段重复。
- 关键公式给出必要推导；不堆砌与主线无关的公式。
- 每章编写前必须检查全书目录、当前目录 `index.md`、前置章节和相邻章节。
- 知乎和博客只用于寻找解释角度；公式、技术事实和前沿方法由教材、论文或官方资料交叉验证。
- 每个目录完成后独立运行内容检查、严格构建并提交 Git。
- Part 2 最终不得保留 `status: planned`、占位正文或失效内部链接。

---

## Target File Map

```text
docs/02-deep-learning/
├── index.md
├── 01-machine-learning-basics/
│   ├── index.md
│   ├── learning-problem.md
│   ├── linear-models.md
│   ├── classification-regression.md
│   ├── trees-and-ensembles.md
│   ├── svm-and-kernels.md
│   ├── unsupervised-learning.md
│   └── generalization-and-data.md
├── 02-neural-network-foundations/
│   ├── index.md
│   ├── perceptron-and-mlp.md
│   ├── computational-graphs.md
│   ├── loss-functions.md
│   ├── activations.md
│   └── inductive-bias.md
├── 03-training/
│   ├── index.md
│   ├── backpropagation.md
│   ├── minibatch-and-training-loop.md
│   ├── initialization.md
│   ├── optimization.md
│   ├── regularization.md
│   └── normalization.md
├── 04-components/
│   ├── index.md
│   ├── convolution.md
│   ├── residual-connections.md
│   ├── recurrence.md
│   ├── attention.md
│   ├── positional-encoding.md
│   └── graph-message-passing.md
├── 05-architectures/
│   ├── index.md
│   ├── cnn-and-resnet.md
│   ├── rnn-lstm-gru.md
│   ├── transformer.md
│   ├── graph-neural-networks.md
│   └── architecture-selection.md
├── 06-evaluation-debugging/
│   ├── index.md
│   ├── metrics-and-thresholds.md
│   ├── diagnosing-training.md
│   ├── data-and-distribution-shift.md
│   ├── efficiency-basics.md
│   └── reproducible-experiments.md
└── 07-interview-review/
    ├── index.md
    ├── core-concepts.md
    ├── essential-derivations.md
    ├── troubleshooting.md
    └── design-tradeoffs.md
```

Supporting files:

```text
chapter-writing-workflow.md
writing-style.md
reference.md
mkdocs.yml
requirements.txt
.github/workflows/deploy.yml
scripts/check_part2_content.py
tests/test_check_part2_content.py
```

---

### Task 1: Content Contract and Validation

**Files:**
- Create: `scripts/check_part2_content.py`
- Create: `tests/test_check_part2_content.py`
- Modify: `chapter-writing-workflow.md`
- Modify: `writing-style.md`
- Modify: `reference.md`
- Modify: `requirements.txt`
- Modify: `.github/workflows/deploy.yml`

**Interfaces:**
- Consumes: Markdown files under `docs/02-deep-learning/`.
- Produces: `validate_part2(root: Path, require_complete: bool = False) -> list[str]` and CLI flags `--root PATH`, `--require-complete`.
- Metadata contract: `level ∈ {basic, intermediate, advanced}`, `roles ⊆ {core, interview, research}`, `prerequisites` is a YAML list, `estimated_time` matches `^[1-9][0-9]*min$`, `status ∈ {planned, draft, complete}`.

- [ ] **Step 1: Write failing validator tests**

Create fixture files in temporary directories and assert exact failures:

```python
def test_rejects_invalid_role(self):
    errors = validate_part2(self.root)
    self.assertIn("chapter.md: unsupported role 'exam'", errors)

def test_require_complete_rejects_planned_page(self):
    errors = validate_part2(self.root, require_complete=True)
    self.assertIn("chapter.md: status must be complete", errors)

def test_rejects_missing_relative_link_target(self):
    errors = validate_part2(self.root)
    self.assertIn("chapter.md: missing link target missing.md", errors)
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest tests.test_check_part2_content -v`

Expected: FAIL because `scripts.check_part2_content` does not exist.

- [ ] **Step 3: Implement the validator**

Use `yaml.safe_load` for front matter, `Path.resolve()` for Markdown link targets, and skip external URLs, anchors and image links. Return sorted, human-readable errors; exit `1` when the CLI finds errors.

The CLI must support:

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
python scripts/check_part2_content.py --root docs/02-deep-learning --require-complete
```

- [ ] **Step 4: Update editorial rules and dependencies**

Add the exact metadata contract and pre-writing context checklist to `writing-style.md` and `chapter-writing-workflow.md`. Restructure `reference.md` into Part/topic sections with source type labels: `教材`, `论文`, `官方资料`, `解释参考`.

Reduce `requirements.txt` to the direct runtime dependencies used by `mkdocs.yml` and the validator:

```text
mkdocs==1.6.1
mkdocs-material==9.6.0
mkdocs-glightbox==0.4.0
mkdocs-git-revision-date-localized-plugin==1.2.1
PyYAML==6.0.1
```

Change GitHub Actions installation to `pip install -r requirements.txt`, then run the validator before deployment.

- [ ] **Step 5: Verify foundation**

Run:

```bash
python -m unittest discover -s tests -v
mkdocs build --strict
```

Expected: all validator fixture tests pass and MkDocs exits `0`. Do not run the validator against the legacy Part 2 tree yet: those pages intentionally lack the new metadata and are replaced in Task 2.

- [ ] **Step 6: Commit**

```bash
git add scripts/check_part2_content.py tests/test_check_part2_content.py chapter-writing-workflow.md writing-style.md reference.md requirements.txt .github/workflows/deploy.yml
git commit -m "build: add Part 2 content contract"
```

---

### Task 2: Part 2 Skeleton and Navigation

**Files:**
- Delete: `docs/02-deep-learning/cnn.md`
- Delete: `docs/02-deep-learning/gnn.md`
- Delete: `docs/02-deep-learning/gradient-backprop.md`
- Delete: `docs/02-deep-learning/neural-networks.md`
- Delete: `docs/02-deep-learning/resnet.md`
- Delete: `docs/02-deep-learning/rnn.md`
- Delete: `docs/02-deep-learning/transformer.md`
- Replace: `docs/02-deep-learning/index.md`
- Create: all directories and Markdown files in the Target File Map.
- Modify: `mkdocs.yml`

**Interfaces:**
- Consumes: metadata contract from Task 1.
- Produces: stable Part 2 URLs and the ordered dependency graph used by all later tasks.

- [ ] **Step 1: Add a failing navigation test**

Extend `tests/test_check_part2_content.py` to load `mkdocs.yml`, collect every Part 2 Markdown target, and assert it equals the set of Markdown files under `docs/02-deep-learning/`.

Expected failure: legacy flat pages do not match the target map.

- [ ] **Step 2: Replace the legacy skeleton**

Create every target page with valid metadata, `status: planned`, one H1, a concise responsibility statement and links to its immediate prerequisites. Do not copy legacy stub prose.

Use this exact page shape during the transition:

```markdown
---
level: basic
roles:
  - core
prerequisites: []
estimated_time: 20min
status: planned
---

# 学习问题的基本形式

本章负责说明输入、目标、模型、损失和评估之间的关系。具体内容将在本目录编写阶段完成。
```

- [ ] **Step 3: Write the Part 2 overview**

`docs/02-deep-learning/index.md` must define the reader profile, seven-stage route, three roles, prerequisite map and two suggested paths:

```text
初学主线：01 → 02 → 03 → 04 → 05 → 06
面试复习：02 → 03 → 05 → 06 → 07
```

- [ ] **Step 4: Replace the Part 2 nav block**

List every directory `index.md` and child page explicitly in dependency order. Titles use concise Chinese names; URLs retain the English filenames from the Target File Map.

- [ ] **Step 5: Verify skeleton**

Run:

```bash
python -m unittest discover -s tests -v
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
```

Expected: PASS without `--require-complete`; strict build exits `0` and reports no orphan Part 2 pages.

- [ ] **Step 6: Commit**

```bash
git add docs/02-deep-learning mkdocs.yml tests/test_check_part2_content.py
git commit -m "refactor(part2): establish mainline chapter structure"
```

---

### Task 3: 01 Machine Learning Basics

**Files:**
- Replace: `docs/02-deep-learning/01-machine-learning-basics/index.md`
- Replace: `docs/02-deep-learning/01-machine-learning-basics/learning-problem.md`
- Replace: `docs/02-deep-learning/01-machine-learning-basics/linear-models.md`
- Replace: `docs/02-deep-learning/01-machine-learning-basics/classification-regression.md`
- Replace: `docs/02-deep-learning/01-machine-learning-basics/trees-and-ensembles.md`
- Replace: `docs/02-deep-learning/01-machine-learning-basics/svm-and-kernels.md`
- Replace: `docs/02-deep-learning/01-machine-learning-basics/unsupervised-learning.md`
- Replace: `docs/02-deep-learning/01-machine-learning-basics/generalization-and-data.md`
- Modify: `reference.md`

**Interfaces:**
- Consumes: Part 1 probability, linear algebra and optimization concepts by link, without re-deriving them.
- Produces: empirical risk, train/validation/test split, generalization, bias/variance, linear prediction and classical baseline vocabulary used by all later directories.

- [ ] **Step 1: Inspect context and collect sources**

Read the full Part 2 tree, Part 1 index pages and every page in this directory. Record authoritative sources for statistical learning, linear/logistic regression, trees, SVM and clustering in `reference.md`.

- [ ] **Step 2: Write the directory overview and learning framework**

Explain the common interface `dataset → hypothesis → objective → optimization → evaluation`. Derive mean squared error as an empirical objective and explain why training loss is not the final goal.

- [ ] **Step 3: Write model-family chapters**

Cover these exact boundaries:

- `linear-models.md`: linear regression, logistic regression and regularization;
- `classification-regression.md`: task distinction, probability output, decision threshold and calibration intuition;
- `trees-and-ensembles.md`: split criteria, bagging, random forest and gradient boosting trade-offs;
- `svm-and-kernels.md`: margin, hinge loss, soft margin and kernel intuition;
- `unsupervised-learning.md`: clustering, dimensionality reduction and representation learning bridge.

- [ ] **Step 4: Write generalization and data chapter**

Explain data splits, leakage, distribution mismatch, capacity, underfitting/overfitting and the bias–variance trade-off. Link forward to metrics and distribution shift rather than duplicating them.

- [ ] **Step 5: Set all directory pages to `status: complete` and verify**

Run:

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
```

Expected: PASS; no planned page remains inside `01-machine-learning-basics/`.

- [ ] **Step 6: Commit**

```bash
git add docs/02-deep-learning/01-machine-learning-basics reference.md
git commit -m "docs(part2): write machine learning foundations"
```

---

### Task 4: 02 Neural Network Foundations

**Files:**
- Replace: all Markdown files under `docs/02-deep-learning/02-neural-network-foundations/`
- Modify: `reference.md`

**Interfaces:**
- Consumes: linear prediction, empirical risk and generalization from Task 3.
- Produces: tensor shape notation, computation graph, loss, activation and inductive-bias concepts used by training and architecture chapters.

- [ ] **Step 1: Inspect neighboring directories and collect sources**

Review Tasks 3, 5 and 6 chapter responsibilities. Add textbook, original-paper or official sources for backpropagation context, activation functions and inductive bias.

- [ ] **Step 2: Write `perceptron-and-mlp.md` and `computational-graphs.md`**

Derive the affine layer shape and show how stacked affine maps collapse without nonlinear activation. Use one scalar computation graph to introduce local derivatives; leave full reverse-mode derivation to `03-training/backpropagation.md`.

- [ ] **Step 3: Write `loss-functions.md`, `activations.md` and `inductive-bias.md`**

Cover MSE, binary/multiclass cross-entropy, numerical stability, ReLU-family and smooth activations. Explain how architecture and data assumptions create inductive bias; do not pre-explain CNN, RNN or GNN internals.

- [ ] **Step 4: Write the directory index and cross-links**

The knowledge map must connect classical linear models to MLPs, then connect computation graphs and losses to the training directory.

- [ ] **Step 5: Verify and commit**

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
git add docs/02-deep-learning/02-neural-network-foundations reference.md
git commit -m "docs(part2): write neural network foundations"
```

---

### Task 5: 03 Training

**Files:**
- Replace: all Markdown files under `docs/02-deep-learning/03-training/`
- Modify: `reference.md`

**Interfaces:**
- Consumes: computation graphs, losses and activations from Task 4; optimization basics from Part 1 by link.
- Produces: gradients, initialization, mini-batch dynamics, optimizer, regularization and normalization concepts used by every architecture and debugging chapter.

- [ ] **Step 1: Inspect the full dependency path and collect sources**

Read the Part 1 optimization index, Task 4 pages and Task 8 diagnostic responsibilities. Add primary sources for Xavier/He initialization, Adam/AdamW, dropout, batch normalization and layer normalization.

- [ ] **Step 2: Write backpropagation and the training loop**

Derive reverse-mode differentiation on a two-layer scalar example, then generalize to vector-Jacobian products. Explain epoch, batch, step, gradient accumulation, training/evaluation mode and a minimal training loop without tying the book to one framework.

- [ ] **Step 3: Write initialization and optimization**

Use variance propagation to motivate Xavier and He initialization. Compare SGD, Momentum, Adam and AdamW, including the difference between L2 regularization and decoupled weight decay. Explain learning-rate schedules and warmup only to Part 2 depth.

- [ ] **Step 4: Write regularization and normalization**

Separate the purposes of weight decay, dropout, augmentation, early stopping and label smoothing. Compare BatchNorm, LayerNorm and RMSNorm by normalized axes, train/inference behavior and suitable architectures.

- [ ] **Step 5: Verify and commit**

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
git add docs/02-deep-learning/03-training reference.md
git commit -m "docs(part2): explain model training end to end"
```

---

### Task 6: 04 Reusable Components

**Files:**
- Replace: all Markdown files under `docs/02-deep-learning/04-components/`
- Modify: `reference.md`

**Interfaces:**
- Consumes: tensor shapes, gradients, initialization and normalization from Tasks 4–5.
- Produces: convolution, residual, recurrence, attention, positional information and message-passing mechanisms combined by Task 7 architectures.

- [ ] **Step 1: Establish the component/architecture boundary**

For each component, describe the operation, shape flow, inductive bias, computational cost and failure mode. Do not narrate complete model families; those belong to Task 7.

- [ ] **Step 2: Write spatial and sequential components**

`convolution.md` derives output size and parameter sharing. `residual-connections.md` explains identity paths and gradient flow. `recurrence.md` derives the recurrent state update and explains long dependency difficulty.

- [ ] **Step 3: Write attention and positional encoding**

Derive scaled dot-product attention from query–key matching, explain the `1/√d_k` scale, tensor shapes, masks and quadratic sequence cost. Keep sinusoidal, learned and rotary position methods at mechanism-comparison depth.

- [ ] **Step 4: Write graph message passing**

Define neighborhood aggregation, permutation invariance and one generic message-passing equation. Defer named GNN architectures and oversmoothing analysis to Task 7.

- [ ] **Step 5: Verify and commit**

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
git add docs/02-deep-learning/04-components reference.md
git commit -m "docs(part2): explain reusable network components"
```

---

### Task 7: 05 Model Architectures

**Files:**
- Replace: all Markdown files under `docs/02-deep-learning/05-architectures/`
- Modify: `reference.md`

**Interfaces:**
- Consumes: reusable components from Task 6 and training behavior from Task 5.
- Produces: coherent CNN/ResNet, RNN/LSTM/GRU, Transformer and GNN model-family understanding used by evaluation and interview synthesis.

- [ ] **Step 1: Inspect component pages and collect original sources**

Use the original or authoritative papers for ResNet, LSTM, GRU, Transformer and representative message-passing GNNs. Link component definitions rather than rewriting them.

- [ ] **Step 2: Write CNN/ResNet and recurrent architecture chapters**

Explain how layers compose into feature hierarchies, receptive fields and downsampling stages. For recurrent models, compare vanilla RNN, LSTM and GRU state paths and practical sequence limitations.

- [ ] **Step 3: Write Transformer and GNN architecture chapters**

Build the Transformer from embeddings, attention, feed-forward blocks, residual paths and normalization; distinguish encoder, decoder and encoder–decoder layouts. Build GNNs from message-passing layers, readout and task heads; explain oversmoothing and depth limits.

- [ ] **Step 4: Write architecture selection**

Compare inductive bias, data scale, parallelism, memory, latency and task structure. Do not rank architectures universally; give a decision process and concrete counterexamples.

- [ ] **Step 5: Verify and commit**

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
git add docs/02-deep-learning/05-architectures reference.md
git commit -m "docs(part2): connect components into model architectures"
```

---

### Task 8: 06 Evaluation and Debugging

**Files:**
- Replace: all Markdown files under `docs/02-deep-learning/06-evaluation-debugging/`
- Modify: `reference.md`

**Interfaces:**
- Consumes: generalization from Task 3, training from Task 5 and architecture costs from Task 7.
- Produces: evaluation, failure diagnosis, data-shift, efficiency and reproducibility methods used by final interview review.

- [ ] **Step 1: Collect metric, reliability and systems sources**

Use authoritative definitions for precision/recall/F1, ROC-AUC, PR-AUC, calibration and distribution shift. Use framework or hardware documentation for memory and mixed-precision claims.

- [ ] **Step 2: Write metrics and training diagnosis**

Explain confusion-matrix metrics, thresholds, class imbalance and metric selection. Organize diagnosis by observed symptom: loss does not fall, validation degrades, gradients explode/vanish, training is unstable, throughput is low.

- [ ] **Step 3: Write data shift and efficiency basics**

Separate covariate, label and concept shift at introductory depth. Derive parameter memory and approximate dense-layer/attention compute; explain training versus inference memory, mixed precision, batching and latency/throughput trade-offs.

- [ ] **Step 4: Write reproducible experiments**

Cover seeds, deterministic limitations, data/version tracking, configuration capture, baselines, ablations and reporting variance. Keep it practical and avoid pretending exact reproducibility is always possible.

- [ ] **Step 5: Verify and commit**

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
git add docs/02-deep-learning/06-evaluation-debugging reference.md
git commit -m "docs(part2): add evaluation and training diagnostics"
```

---

### Task 9: 07 Interview Review

**Files:**
- Replace: all Markdown files under `docs/02-deep-learning/07-interview-review/`
- Modify: `reference.md`

**Interfaces:**
- Consumes: all Part 2 chapters.
- Produces: compact retrieval paths for concepts, derivations, troubleshooting and design trade-offs without duplicating full explanations.

- [ ] **Step 1: Build a coverage matrix**

Map every interview prompt to one canonical Part 2 chapter. The review page gives a short answer structure and a link; it does not repeat the full chapter.

- [ ] **Step 2: Write core concepts and essential derivations**

Cover definitions and comparisons that require precise language. Derivations include softmax cross-entropy gradient, backpropagation through an affine layer, convolution output size, attention scaling and normalization-axis comparisons.

- [ ] **Step 3: Write troubleshooting and design trade-offs**

Use symptom-driven scenarios and require the reader to state evidence, hypotheses, checks and fixes. Compare model families and training choices under data, compute, latency and interpretability constraints.

- [ ] **Step 4: Write the directory index as a review schedule**

Provide 1-day, 3-day and 7-day routes using links to canonical chapters. Keep each route realistic and identify prerequisites.

- [ ] **Step 5: Verify and commit**

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning
mkdocs build --strict
git add docs/02-deep-learning/07-interview-review reference.md
git commit -m "docs(part2): add interview synthesis and review paths"
```

---

### Task 10: Final Content and Site Audit

**Files:**
- Modify as required by evidence: `docs/02-deep-learning/**/*.md`, `mkdocs.yml`, `reference.md`
- Create: `docs/superpowers/reviews/2026-08-16-part2-audit.md`

**Interfaces:**
- Consumes: completed Tasks 1–9.
- Produces: auditable proof that Part 2 is complete and safe to merge.

- [ ] **Step 1: Require complete content**

Run:

```bash
python scripts/check_part2_content.py --root docs/02-deep-learning --require-complete
```

Expected: exit `0`; no planned/draft pages, invalid metadata, broken relative links or placeholder markers.

- [ ] **Step 2: Run the full local verification suite**

```bash
python -m unittest discover -s tests -v
mkdocs build --strict
git diff --check main...HEAD
```

Expected: all tests pass, MkDocs exits `0`, and Git reports no whitespace errors.

- [ ] **Step 3: Audit scope and repetition**

Record each target file, its status, canonical responsibility and incoming/outgoing links in the audit document. Search repeated long paragraphs and verify duplicated explanations are replaced by concise reminders and links.

- [ ] **Step 4: Inspect the rendered site**

Serve or open the generated site and inspect the Part 2 landing page, one formula-heavy chapter, each directory index, mobile navigation and previous/next links.

- [ ] **Step 5: Commit audit fixes**

```bash
git add docs/02-deep-learning docs/superpowers/reviews/2026-08-16-part2-audit.md mkdocs.yml reference.md
git commit -m "docs(part2): complete content and site audit"
```

---

### Task 11: Merge and Publish

**Files:**
- No planned content edits; only evidence-backed release fixes are allowed.

**Interfaces:**
- Consumes: verified `rewrite/part2-mainline` branch.
- Produces: merged local `main`, pushed `main`, and a successful GitHub Pages deployment.

- [ ] **Step 1: Confirm clean state and branch ancestry**

```bash
git status --short
git log --oneline --decorate --graph main..rewrite/part2-mainline
```

Expected: clean worktree and only intended rewrite commits.

- [ ] **Step 2: Merge into main**

```bash
git switch main
git merge --no-ff rewrite/part2-mainline -m "merge: publish Part 2 mainline rewrite"
```

- [ ] **Step 3: Re-run verification on main**

```bash
python -m unittest discover -s tests -v
python scripts/check_part2_content.py --root docs/02-deep-learning --require-complete
mkdocs build --strict
```

Expected: all commands exit `0` on the exact commit to be published.

- [ ] **Step 4: Push and verify deployment**

```bash
git push origin archive/pre-rewrite-20260816
git push origin main
```

Confirm the GitHub Actions deployment succeeds and the published site shows the new Part 2 navigation and pages.
