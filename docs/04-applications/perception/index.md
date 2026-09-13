# 4.1 感知

感知模型把像素变成机器可以继续使用的语义与几何信息。本章从整图标签出发，逐步增加输出的空间精度：边界框回答物体在哪里，掩码回答每个像素属于什么，光流和深度则描述运动与三维结构。OCR 最后展示如何把检测、识别、版面和语义组合成完整系统。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [图像分类](image-classification.md) | 从 AlexNet、ResNet 到 ViT，理解视觉表征怎样形成 | [CNN](../../02-deep-learning/05-architectures/cnn-and-resnet.md)、[ViT](../../03-advanced/modern-architectures/vit.md) |
| [目标检测](object-detection.md) | 两阶段、单阶段、anchor-free 与 DETR 集合预测 | 图像分类、[Transformer](../../02-deep-learning/05-architectures/transformer.md) |
| [图像分割](segmentation.md) | 语义、实例、全景与提示分割，FCN 到 SAM 3 | 目标检测、[U-Net](../../03-advanced/modern-architectures/unet.md) |
| [光流估计](optical-flow.md) | 从经典能量优化到 RAFT 的相关体和迭代更新 | [CNN](../../02-deep-learning/05-architectures/cnn-and-resnet.md)、优化基础 |
| [深度估计](depth-estimation.md) | 双目几何、单目相对/米制深度与空间基础模型 | [相机模型](../3dv/camera-embedding.md)、[DPT](../../03-advanced/modern-architectures/dpt.md) |
| [OCR 与文档智能](ocr.md) | 文字检测识别、版面预训练与 OCR-free 文档理解 | 目标检测、[Transformer](../../02-deep-learning/05-architectures/transformer.md) |

## 建议阅读顺序

第一次系统学习时，先读图像分类，再读目标检测与图像分割。这三节建立“整图、物体、像素”三个输出层级。

光流和深度更依赖几何，可以独立阅读。OCR 与文档智能会同时用到检测、序列建模和多模态表示，放在最后更容易看清完整工程管线。

## 本章知识地图

<figure class="lesson-visual" data-lesson-visual>
  <div data-lesson-stage role="img" aria-label="感知任务从整图语义分出物体定位、像素分割、二维运动、三维距离和文字识别六条路径">
    <svg class="lesson-visual__canvas--wide" viewBox="0 0 900 240" aria-hidden="true">
      <defs>
        <marker id="perception-map-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M 0 0 L 8 4 L 0 8 z" fill="var(--md-default-fg-color--lighter)"/>
        </marker>
      </defs>
    <g data-step data-step-label="分类">
      <rect x="26" y="92" width="128" height="56" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-primary-fg-color)" stroke-width="2"/>
      <text x="90" y="115" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">分类</text>
      <text x="90" y="136" text-anchor="middle" fill="var(--md-default-fg-color--light)" font-size="12">整图语义</text>
    </g>
    <g data-step data-step-label="检测">
      <path d="M 154 120 L 236 70" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="3" marker-end="url(#perception-map-arrow)"/>
      <rect x="236" y="42" width="128" height="56" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-primary-fg-color)" stroke-width="2"/>
      <text x="300" y="65" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">检测</text>
      <text x="300" y="86" text-anchor="middle" fill="var(--md-default-fg-color--light)" font-size="12">类别与框</text>
    </g>
    <g data-step data-step-label="分割">
      <path d="M 364 70 L 446 45" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="3" marker-end="url(#perception-map-arrow)"/>
      <rect x="446" y="17" width="128" height="56" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-primary-fg-color)" stroke-width="2"/>
      <text x="510" y="40" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">分割</text>
      <text x="510" y="61" text-anchor="middle" fill="var(--md-default-fg-color--light)" font-size="12">像素与实例</text>
    </g>
    <g data-step data-step-label="光流">
      <path d="M 154 120 L 236 175" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="3" marker-end="url(#perception-map-arrow)"/>
      <rect x="236" y="147" width="128" height="56" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-primary-fg-color)" stroke-width="2"/>
      <text x="300" y="170" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">光流</text>
      <text x="300" y="191" text-anchor="middle" fill="var(--md-default-fg-color--light)" font-size="12">二维运动</text>
    </g>
    <g data-step data-step-label="深度">
      <path d="M 154 120 L 446 175" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="3" marker-end="url(#perception-map-arrow)"/>
      <rect x="446" y="147" width="128" height="56" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-primary-fg-color)" stroke-width="2"/>
      <text x="510" y="170" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">深度</text>
      <text x="510" y="191" text-anchor="middle" fill="var(--md-default-fg-color--light)" font-size="12">三维距离</text>
    </g>
    <g data-step data-step-label="OCR">
      <path d="M 364 70 L 656 110" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="3" marker-end="url(#perception-map-arrow)"/>
      <path d="M 154 120 L 656 110" fill="none" stroke="var(--md-default-fg-color--lighter)" stroke-width="3" marker-end="url(#perception-map-arrow)"/>
      <rect x="656" y="82" width="128" height="56" rx="12" fill="var(--md-code-bg-color)" stroke="var(--md-primary-fg-color)" stroke-width="2"/>
      <text x="720" y="105" text-anchor="middle" fill="currentColor" font-size="15" font-weight="700">OCR</text>
      <text x="720" y="126" text-anchor="middle" fill="var(--md-default-fg-color--light)" font-size="12">文字定位与识别</text>
    </g>
    </svg>
  </div>
  <ol data-lesson-steps>
    <li>图像分类先把整张图压成语义表示。</li>
    <li>目标检测在分类基础上增加物体位置。</li>
    <li>图像分割把空间精度继续推进到像素和实例。</li>
    <li>光流从图像表征中估计二维运动。</li>
    <li>深度估计恢复三维距离线索。</li>
    <li>OCR 结合图像语义与目标定位，输出文字及其位置。</li>
  </ol>
  <figcaption>观察输出粒度怎样从整图标签一路细化到框、像素、运动、深度和文字。</figcaption>
</figure>
