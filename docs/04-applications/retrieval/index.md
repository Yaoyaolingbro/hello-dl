# 搜索、排序与信息检索

从 BM25 到神经检索，从向量索引到 RAG，这一章讲清楚"怎么找到相关信息"这个核心问题。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [ANN 与 HNSW](ann-hnsw.md) | 近似最近邻、HNSW 图索引、FAISS | 向量、内积 |
| [Learning to Rank](learning-to-rank.md) | Pointwise / Pairwise / Listwise | MLP |
| [稠密检索](dense-retrieval.md) | DPR、bi-encoder、向量召回 | BERT、对比学习 |
| [神经排序](neural-ranking.md) | Cross-encoder、ColBERT | Transformer |
| [RAG](rag.md) | 检索增强生成、chunking、reranking | LLM 基础 |
| [多模态 RAG](multimodal-rag.md) | 图文混合检索、多模态 embedding | 视觉编码器 |
| [Agentic Retrieval](agentic-retrieval.md) | 主动检索、迭代查询、工具调用 | Agent |
