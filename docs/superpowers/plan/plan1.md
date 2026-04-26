现在我需要对整个项目的结构进行更完善的修改
对于part二的部分，卷击神经网络之后，我希望添加resnet。循环神经网络里面需要扩充LRU、LSTM。transformer章节要扩写（1. Seq2Seq 到 Transformer 的历史动机
2. token、embedding、position encoding
3. Q/K/V attention 的矩阵解释
4. multi-head attention
5. layer norm、residual、FFN
6. encoder-only：BERT
7. decoder-only：GPT
8. encoder-decoder：T5
9. 长上下文问题：复杂度、RoPE

对于part三，现代网络结构里面，我希望添加clip代表的contrastive learning；强化学习这个章节，你可以参考互联网上关于CS 285的章的设计，不要太过复杂，最好面向与初学者可以层层递进的学明白，去除grpo。

Part 4的结构要大篇幅修改，并且在part4 的indexmd要有简单的章节为单位的mermaid知识导图：
Part 4 · 现代 AI 应用

4.1 感知
  - 图像分类
  - 目标检测
  - 语义 / 实例 / 全景分割、
  - 光流
  - 深度估计
  - OCR / 文档智能

4.2 3D 视觉与空间智能
  - 相机模型和Plunker Embedding
  - MVS / SfM
  - SLAM
  - Gaussian Splatting （3D、4D 、稀疏化）
  - Feed Forward Model（Dust3r、Cut3r、VGGT）
  - 后向优化的方法（RAFT为代表）
  - 3D 生成（Trellis）
  - 多模态大模型应用

4.3 语音与音频智能
  - ASR 语音识别
  - TTS 语音合成
  - 语音增强
  - 音视频多模态理解

4.4 搜索、排序与信息检索（我还希望有HNSW和ANN等算法，可以重构一下下面的内容）
  - Learning to Rank
  - dense retrieval
  - neural ranking
  - RAG
  - 多模态 RAG
  - agentic retrieval

4.5 图像和视频生成
  - SD和FLUX
  - SVD
  - DIT
  - 编辑和定制化
  - 条件生成
  - 流式生成
  - 生成模型做感知任务
  - 生成模型上的强化学习（diffusionNFT）

4.6 多模态大模型
  - 语言编码器
  - 视觉编码器:
      - CLIP:
      - BLIP & BLIP-2
      - SigLIP
   - 视觉语言模型（LLava、Qwen）
   - 大语言模型基础
   - 思维链（CoT，ToT）
   - LoRA 与参数高效微调
   - DPO、GRPO、DAPO
   - 多图理解

4.7 MLsys（这里我觉得你也可以自己设计更合理的章节结构，想给大家讲一下重要的mlsys相关的知识，包括训练和推理）
- 基本训练知识（不同的并行策略fsdp等）
- Quantization（INT8 / FP8 / GPTQ / AWQ）
- KV cache 优化
- flash-attn
- Sparse / MoE
- Speculative decoding
- 编译优化（TensorRT / vLLM）

4.8 Agent 与工具使用（包括但不限于下面内容，你可以自行在langchain这样的文档里学习了解，设计必要的内容即可）
  - Tool use
  - ReAct
  - planning
  - memory
  - multi-agent
  - workflow agent

4.9 具身智能与机器人
  - imitation learning（偏应用层面）
  - navigation
  - manipulation
  - locomotion
  - VLA model
  - Real2sim （数据映射）
  - Sim2real （RL token）
  - world model

4.10 AI for Science （这个我自己也不是很了解，你可以帮我设计结构）
  - 研究结构、序列和功能之间的联系
  - 蛋白质
  - 分子与材料
  - 气象预测
  - PDE / 科学计算代理模型（PINN等内容也考虑）

4.11 量化和金融（直接链接到quant-wiki）