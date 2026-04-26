# MLSys：训练与推理系统

模型写得再好，跑不起来也没用。这一章讲工程师真正会用到的系统知识：怎么让训练更快、让推理更省、让模型装进更小的设备。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [训练并行策略](training-parallelism.md) | DP / TP / PP / FSDP、ZeRO | Transformer、分布式基础 |
| [量化](quantization.md) | INT8 / FP8、GPTQ、AWQ、校准 | LLM 基础 |
| [KV Cache 优化](kv-cache.md) | MHA vs MQA vs GQA、PagedAttention | Transformer |
| [Flash Attention](flash-attention.md) | IO 感知注意力、Tiling、反向传播 | Attention 机制 |
| [Sparse 与 MoE](sparse-moe.md) | 稀疏激活、Expert Routing、负载均衡 | Transformer |
| [Speculative Decoding](speculative-decoding.md) | Draft-verify、投机采样加速 | 自回归解码 |
| [编译优化](compilation.md) | TensorRT、vLLM、torch.compile、XLA | 推理流水线 |
