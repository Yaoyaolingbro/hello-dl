# OCR 与文档智能

!!! info "参考资料"
    **必读论文**

    - [An End-to-End Trainable Neural Network for Image-based Sequence Recognition](https://arxiv.org/abs/1507.05717) — Shi et al., TPAMI 2017
    - [Character Region Awareness for Text Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.html) — Baek et al., CVPR 2019
    - [LayoutLM: Pre-training of Text and Layout for Document Image Understanding](https://arxiv.org/abs/1912.13318) — Xu et al., KDD 2020
    - [TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models](https://arxiv.org/abs/2109.10282) — Li et al., AAAI 2023
    - [OCR-free Document Understanding Transformer](https://arxiv.org/abs/2111.15664) — Kim et al., ECCV 2022
    - [LayoutLMv3](https://arxiv.org/abs/2204.08387) — Huang et al., ACM MM 2022

## 直觉 (Intuition)

OCR 最基本的任务是把图像中的文字变成字符序列。真实文档还要求模型知道文字在哪里、阅读顺序是什么，以及“金额”“日期”“签名”分别对应哪一块。输入可以是街景、扫描件或多页 PDF，输出可能是纯文本、带坐标的文字行，也可能是结构化 JSON。困难来自字体、语言、透视、低清晰度和复杂版面。文档智能不是把 OCR 结果交给大模型就结束，错误会沿检测、识别、排序和抽取逐级传播。

## 任务定义

OCR 与文档智能通常包含几层任务：

| 层级 | 输入与输出 | 常用指标 |
|------|------------|----------|
| 文字检测 | 图像到文字框或多边形 | Precision、Recall、H-mean |
| 文字识别 | 裁剪文字图到字符序列 | Character Error Rate、Word Accuracy |
| 版面分析 | 页面到标题、段落、表格等区域 | mAP、mIoU |
| 信息抽取 | 页面到字段和值 | Entity F1、字段准确率 |
| 文档问答 | 页面和问题到答案 | ANLS、Exact Match |

字符错误率 (Character Error Rate, CER) 基于编辑距离：

$$
\operatorname{CER}=\frac{S+D+I}{N}.
$$

其中 $S$、$D$、$I$ 分别是替换、删除和插入字符数，$N$ 是真实字符总数。CER 能反映识别错误，但不能说明版面顺序和字段语义是否正确。

## 发展脉络

### CRNN：不再逐字符切割

传统 OCR 常先切出单个字符，再逐个分类。连写、字符宽度变化和切割错误会让后续识别失败。

CRNN（[Paper](https://arxiv.org/abs/1507.05717) | [Project](https://github.com/bgshih/crnn)）用 CNN 提取图像特征，把横向特征视为序列，再用循环网络和 CTC 损失输出可变长度文本。CTC 允许训练时只有整行转写，不要求每个字符的精确位置。

这条路线把检测后的文字行直接变成序列，成为很长时间内的 OCR 基线。它仍依赖规则裁剪，对弯曲文字、多方向文本和复杂语言建模有限。

### CRAFT：先找到字符区域，再把它们连成文字

场景文字可能弯曲、倾斜或字符间距变化。用水平矩形框表示整行文本会包含大量背景，也难以覆盖任意形状。

CRAFT（[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Baek_Character_Region_Awareness_for_Text_Detection_CVPR_2019_paper.html) | [Project](https://github.com/clovaai/CRAFT-pytorch)）预测字符区域热图和相邻字符的 affinity，再把字符连接成词或文本行。它用弱监督生成字符级标签，避免完全依赖昂贵的字符框标注。

CRAFT 改善了任意形状文字检测，但连接阈值和后处理会影响结果。密集排版、艺术字体和极小文字仍可能被错误合并或拆分。

### LayoutLM：文字内容和二维位置要一起建模

纯文本模型看不到“字段名在左、字段值在右”，也无法利用表格线、字体和页面区域。LayoutLM（[Paper](https://arxiv.org/abs/1912.13318) | [Project](https://github.com/microsoft/unilm/tree/master/layoutlm)）把 OCR token、二维框坐标和视觉特征共同输入预训练模型。

它把文档理解从字符串处理变成多模态表示学习。代价是上游 OCR 的漏字、错字和错误坐标会直接传给模型，长页面的 token 数量也容易超过上下文限制。

### TrOCR：视觉编码器直接生成文字

CRNN 使用 CNN、RNN 和 CTC，各模块带有明确的序列假设。TrOCR（[Paper](https://arxiv.org/abs/2109.10282) | [Project](https://github.com/microsoft/unilm/tree/master/trocr)）用预训练图像 Transformer 编码文字图，再用预训练文本 Transformer 自回归生成 wordpiece。

预训练让模型能利用视觉和语言两侧的知识，尤其适合印刷体、手写体和场景文字识别。自回归解码也可能“读得太像语言”：低清晰度输入下，模型会生成语言上合理、图像中不存在的字符。

### Donut：直接从页面图像生成结构化结果

OCR-based 文档理解需要检测、识别、阅读顺序和抽取多个模块。每一层都可能传播错误，调用外部 OCR 也增加延迟。

Donut（[Paper](https://arxiv.org/abs/2111.15664) | [Project](https://github.com/clovaai/donut)）用视觉编码器读取整页图像，文本解码器直接生成任务相关的结构化序列。它开辟了 OCR-free 文档理解路线。

“OCR-free”不等于模型不识字，而是没有显式的 OCR 中间接口。端到端模型减少了流水线错误，却更难定位错误来源，也不天然提供每个字段的文字坐标和置信度。

### LayoutLMv3：统一掩码文本与图像

早期文档预训练对文字和图像使用不同目标，跨模态对齐较复杂。LayoutLMv3（[Paper](https://arxiv.org/abs/2204.08387) | [Project](https://aka.ms/layoutlmv3)）同时对文字 token 和图像 patch 做掩码建模，并加入 word-patch alignment，让模型学习某个词与对应视觉区域的关系。

它说明 OCR-based 路线仍有价值：可搜索文本、精确坐标和视觉版面可以形成可解释的中间表示。文档智能因此不是 OCR-based 与 OCR-free 的简单替代关系，而是可控性、速度和端到端能力之间的选择。

## 核心方法

### CTC 与自回归解码

CTC 假设输出顺序沿输入序列单调前进，通过 blank token 合并重复路径。它并行、高效，适合清晰文字行。自回归解码逐 token 生成，能利用更强语言上下文，但速度较慢，也更容易根据语言先验补出图中没有的内容。

### 阅读顺序

OCR 框的坐标不能自动确定阅读顺序。双栏论文、票据、脚注和表格需要版面模型或规则排序。排序错误时，每个单词都可能识别正确，最终文本仍无法使用。

### OCR-based 与 OCR-free

OCR-based 系统保留文字、坐标和置信度，适合审计、检索和人工复核。OCR-free 系统能直接优化结构化任务，减少中间模块，但输出证据较弱。企业文档常使用混合方案：OCR 提供可追溯证据，视觉语言模型负责语义归纳和异常处理。

## 工程实践

### 先定义输出协议

“识别一张发票”可能表示纯文本、键值对、表格或符合 schema 的 JSON。输出协议不同，数据标注、模型和评测都会变化。字段缺失时也要区分空值、未检测和不适用。

### 多语言不是扩大字符表这么简单

中文、阿拉伯文、竖排文字和混合语言具有不同书写方向与分词规则。字符归一化、全角半角、繁简体和 Unicode 组合字符都要在训练与评测中统一。

### PDF 要保留原生信息

数字 PDF 可能已经包含文字层和坐标。直接把每页渲染成图片再 OCR，会丢失无损文本并增加错误。工程管线应先判断页面是否有可靠文字层，只对扫描区域或异常页面运行视觉 OCR。

!!! tip "工程重点"
    文档抽取系统要保存证据链：字段值、来源页码、文字框、OCR 置信度和原始裁剪。只有一个最终 JSON 时，线上错误很难审计。

### 长文档需要分层处理

把几十页高分辨率文档一次送入模型通常不可行。应先做页面分类、版面切分和候选召回，再对相关区域精读。切块时要保留页码、标题层级和跨页表格关系。

## 开放问题

以下判断基于截至 2026 年 6 月公开的论文与项目资料。

- **低资源语言和历史文献仍缺少数据。** 字体、拼写和纸张退化会同时影响识别与语言模型。
- **端到端模型存在幻觉风险。** 输出可能语义合理却不忠于像素，需要区域证据、字符级置信度或可验证解码。
- **复杂表格和阅读顺序仍不稳定。** 合并单元格、跨页表格、公式和图文混排难以用单一序列表示。
- **评测与业务目标脱节。** 字符级指标无法代表金额、日期和关键字段是否可用，应按字段代价设计评测。
- **隐私与数据治理限制训练。** 企业文档包含身份、财务和医疗信息，数据脱敏、授权和本地部署会直接决定可用方案。
