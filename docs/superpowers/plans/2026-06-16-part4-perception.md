# Part 4 Perception Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the six existing Part 4 perception tutorials as research-oriented popular science articles with a curated paper lineage and official Paper/Project links.

**Architecture:** Keep the existing MkDocs navigation and page boundaries. Establish shared sourcing and writing rules in the Part 4 style guide, then research and write each task page independently before a cross-page consistency pass.

**Tech Stack:** Markdown, MkDocs Material, KaTeX, Mermaid, GitHub Actions.

---

### Task 1: Synchronize the Working Branch

**Files:**
- Merge: `main` into `codex`

- [ ] Fetch `origin` and inspect divergence.
- [ ] Merge the latest local `main` into `codex` without rewriting history.
- [ ] Resolve conflicts by preserving current `main` content and the committed perception design.
- [ ] Run `git status --short --branch` and confirm a clean merge result.

### Task 2: Strengthen the Part 4 Writing Rules

**Files:**
- Modify: `docs/superpowers/style-part4-applications.md`

- [ ] Add the curated-mainline selection criteria from the approved design.
- [ ] Add the required first-mention format: `Method（[Paper](...) | [Project](...)）`.
- [ ] Define source priority and the fallback from official project page to official repository.
- [ ] Add a pre-writing research workflow covering paper sections, benchmark verification, and current-status checks.
- [ ] Add research-popularization guidance that asks what bottleneck, change, evidence, and limitation each paper contributes.
- [ ] Add freshness rules for trends and open problems.
- [ ] Run `git diff --check`.
- [ ] Commit with `docs(part4): strengthen research writing guidelines`.

### Task 3: Write Image Classification

**Files:**
- Modify: `docs/04-applications/perception/image-classification.md`

- [ ] Research the primary papers and official projects for AlexNet, ResNet, EfficientNet, ViT, ConvNeXt, and a modern pretraining route where needed.
- [ ] Write task definition and metrics, distinguishing closed-set classification from representation learning.
- [ ] Explain the progression from convolutional inductive bias to scaling and Transformer-based vision.
- [ ] Add engineering guidance on data quality, transfer learning, class imbalance, calibration, and distribution shift.
- [ ] Add open problems with a June 2026 verification date.
- [ ] Check every selected paper's first mention for official links.

### Task 4: Write Object Detection

**Files:**
- Modify: `docs/04-applications/perception/object-detection.md`

- [ ] Research the primary papers and official projects for R-CNN/Faster R-CNN, YOLO, RetinaNet, FCOS, DETR, and a representative modern DETR improvement.
- [ ] Define boxes, confidence, IoU, NMS, AP, and the distinction between one-stage, two-stage, anchor-free, and set prediction.
- [ ] Build the paper lineage around latency, dense-label imbalance, hand-designed anchors, and matching-based prediction.
- [ ] Add engineering guidance on annotation policy, small objects, resolution, thresholds, and deployment latency.
- [ ] Add open problems with a June 2026 verification date.
- [ ] Check every selected paper's first mention for official links.

### Task 5: Write Image Segmentation

**Files:**
- Modify: `docs/04-applications/perception/segmentation.md`

- [ ] Research the primary papers and official projects for FCN, U-Net, DeepLab, Mask R-CNN, Mask2Former, and SAM.
- [ ] Distinguish semantic, instance, panoptic, and promptable segmentation.
- [ ] Explain the progression from dense prediction to instance masks, mask classification, and foundation segmentation.
- [ ] Add engineering guidance on label boundaries, crop strategy, class imbalance, mask post-processing, and prompt quality.
- [ ] Add open problems with a June 2026 verification date.
- [ ] Check every selected paper's first mention for official links.

### Task 6: Write Optical Flow

**Files:**
- Modify: `docs/04-applications/perception/optical-flow.md`

- [ ] Research the primary papers and official projects for Horn-Schunck, FlowNet, PWC-Net, RAFT, and FlowFormer or a better-supported representative successor.
- [ ] Define the flow field, brightness constancy, endpoint error, occlusion, and domain gap.
- [ ] Explain the transition from variational optimization to learned matching and recurrent refinement.
- [ ] Add engineering guidance on frame timing, augmentation, occlusion masks, resolution, and downstream use.
- [ ] Add open problems with a June 2026 verification date.
- [ ] Check every selected paper's first mention for official links.

### Task 7: Write Depth Estimation

**Files:**
- Modify: `docs/04-applications/perception/depth-estimation.md`

- [ ] Research the primary papers and official projects for stereo matching, Monodepth, DPT/MiDaS, ZoeDepth, Depth Anything, and a current metric-depth route where needed.
- [ ] Separate stereo depth, monocular relative depth, metric depth, and zero-shot generalization.
- [ ] Explain scale ambiguity, geometric cues, data mixing, and foundation-model supervision.
- [ ] Add engineering guidance on camera calibration, depth representation, invalid pixels, scale alignment, and edge quality.
- [ ] Add open problems with a June 2026 verification date.
- [ ] Check every selected paper's first mention for official links.

### Task 8: Write OCR and Document Intelligence

**Files:**
- Modify: `docs/04-applications/perception/ocr.md`

- [ ] Research the primary papers and official projects for CRAFT or DBNet, CRNN, TrOCR, LayoutLM, Donut, and a current document-understanding route where needed.
- [ ] Separate text detection, recognition, layout analysis, key-information extraction, and end-to-end document understanding.
- [ ] Explain the transition from modular OCR pipelines to pretrained multimodal document models.
- [ ] Add engineering guidance on multilingual text, reading order, image quality, long documents, structured outputs, and evaluation.
- [ ] Add open problems with a June 2026 verification date.
- [ ] Check every selected paper's first mention for official links.

### Task 9: Refresh the Perception Guide Page

**Files:**
- Modify: `docs/04-applications/perception/index.md`

- [ ] Update the introduction to explain the relationship among the six tasks.
- [ ] Correct each page description to match the completed content.
- [ ] Add or refine the reading order and prerequisite links without changing the six-page structure.
- [ ] Keep the page as navigation rather than duplicating tutorial content.

### Task 10: Cross-Page Review and Verification

**Files:**
- Review: `docs/04-applications/perception/*.md`
- Review: `docs/superpowers/style-part4-applications.md`

- [ ] Search for and remove all placeholder text.
- [ ] Verify paper titles, years, venues, benchmark claims, and official links against primary sources.
- [ ] Check that each page uses the agreed structure and explains all first-use terminology.
- [ ] Check that all current-trend and open-problem sections state the June 2026 verification date.
- [ ] Run `git diff --check`.
- [ ] Run `python -m unittest discover -s tests -v`.
- [ ] Run `mkdocs build --strict`.
- [ ] Review the complete diff for accidental edits.
- [ ] Commit the completed perception chapter.

### Task 11: Merge and Publish

**Files:**
- Merge: `codex` into `main`

- [ ] Fetch `origin` and verify `origin/main` has not advanced unexpectedly.
- [ ] Push `codex` so the completed branch is backed up remotely.
- [ ] Switch to `main`, fast-forward from `origin/main`, and merge `codex`.
- [ ] Run `python -m unittest discover -s tests -v` on the merged result.
- [ ] Run `mkdocs build --strict` on the merged result.
- [ ] Push `main` to `origin/main`.
- [ ] Report the pushed commit and verification results.
