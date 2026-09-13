---
level: intermediate
roles:
  - core
  - interview
prerequisites:
  - 感知机与多层感知机
  - 归一化
estimated_time: 45min
status: complete
---

# 注意力

!!! info "参考资料"
    - Ashish Vaswani et al., [Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need), NeurIPS 2017
    - [Scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — PyTorch Documentation；掩码与张量接口约定

## 直觉 (Intuition)

句子里出现“它”时，需要的信息可能在前一个词，也可能在二十个词之前。固定窗口按距离取信息；注意力则让当前位置提出一个 query，用它与候选位置的 key 匹配，再按匹配权重汇总对应的 value。

这三个名字分别回答：

- query：我现在要找什么？
- key：这个位置可以按什么特征被找到？
- value：一旦选中，这个位置实际提供什么内容？

一个候选位置的 key 与 value 来自同一位置，但含义不同；不能把“相似度用什么算”和“最终加权什么”混成一个张量。

## 从一次匹配推到矩阵公式

对一个 query $\mathbf q_i\in\mathbb R^{d_k}$ 和第 $j$ 个 key，先算点积分数

$$
s_{ij}=\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}}.
$$

一行分数经 softmax 变成非负、和为 1 的权重，再对 value 加权：

$$
\alpha_{ij}=\frac{\exp(s_{ij})}{\sum_{r=1}^{n_k}\exp(s_{ir})},
\qquad
\mathbf o_i=\sum_{j=1}^{n_k}\alpha_{ij}\mathbf v_j.
$$

把 $n_q$ 个 query、$n_k$ 个 key/value 堆成矩阵，得到缩放点积注意力：

$$
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V.
$$

$M$ 是可选掩码，下一节说明。先逐轴核对单个 head 的形状：

| 张量 | 形状 | 含义 |
|---|---|---|
| $Q$ | $[n_q,d_k]$ | $n_q$ 个查询 |
| $K$ | $[n_k,d_k]$ | $n_k$ 个可匹配键 |
| $V$ | $[n_k,d_v]$ | 与每个 key 对齐的内容 |
| $QK^\top$ | $[n_q,n_k]$ | 每个 query 对每个 key 的分数 |
| softmax 权重 | $[n_q,n_k]$ | 每一行对 key 轴归一化 |
| 输出 | $[n_q,d_v]$ | 每个 query 得到一个加权 value |

多头批量实现常用 $Q,K,V$ 形状 $[B,h,n_q,d_k]$、$[B,h,n_k,d_k]$、$[B,h,n_k,d_v]$。广播 batch 和 head 后，分数为 $[B,h,n_q,n_k]$，输出为 $[B,h,n_q,d_v]$。各头拼接后是 $[B,n_q,h d_v]$，通常再投影回模型维度；如何与前馈层和残差组合留给 [Transformer](../05-architectures/transformer.md)。

点“下一步”沿矩阵管线走一遍。形状写在节点里，便于核对每次变换有没有改错轴。

<figure class="lesson-visual" data-lesson-visual data-interval="1900">
  <div data-lesson-stage role="img" aria-label="多头注意力从 Q 和 K 计算分数，经缩放掩码与 softmax 后加权 V">
    <svg class="lesson-visual__canvas--wide" viewBox="0 0 1080 520" role="img" aria-hidden="true" style="color: var(--md-default-fg-color);">
      <defs>
        <marker id="attention-pipeline-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
        </marker>
      </defs>

      <g data-step data-step-label="Q 与 K 生成成对分数">
        <text x="35" y="35" font-size="18" font-weight="700" fill="currentColor">1. 匹配：每个 query 与每个 key 做点积</text>
        <g fill="none" stroke="currentColor" stroke-width="2">
          <rect x="45" y="75" width="220" height="70" rx="11" />
          <rect x="45" y="175" width="220" height="70" rx="11" />
          <rect x="385" y="115" width="260" height="90" rx="12" />
        </g>
        <g text-anchor="middle" fill="currentColor">
          <text x="155" y="104" font-size="20">Q</text><text x="155" y="128" font-size="16">B × h × n_q × d_k</text>
          <text x="155" y="204" font-size="20">K</text><text x="155" y="228" font-size="16">B × h × n_k × d_k</text>
          <text x="515" y="150" font-size="20">QKᵀ</text><text x="515" y="178" font-size="17">B × h × n_q × n_k</text>
        </g>
        <path d="M265 110 C315 110 325 145 385 150 M265 210 C315 210 325 175 385 170" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#attention-pipeline-arrow)" />
      </g>

      <g data-step data-step-label="缩放并在 logits 上加掩码">
        <text x="690" y="35" font-size="18" font-weight="700" fill="currentColor">2. 缩放与掩码</text>
        <rect x="720" y="90" width="260" height="76" rx="11" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="2" />
        <text x="850" y="120" text-anchor="middle" font-size="18" fill="currentColor">QKᵀ / √d_k + M</text>
        <text x="850" y="145" text-anchor="middle" font-size="16" fill="currentColor">形状仍为 B × h × n_q × n_k</text>
        <path d="M645 160 L720 135" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#attention-pipeline-arrow)" />
        <rect x="720" y="205" width="260" height="62" rx="11" fill="none" stroke="currentColor" stroke-width="2" />
        <text x="850" y="232" text-anchor="middle" font-size="17" fill="currentColor">M：允许位置 0</text>
        <text x="850" y="254" text-anchor="middle" font-size="17" fill="currentColor">禁止位置 −∞</text>
        <path d="M850 205 L850 166" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#attention-pipeline-arrow)" />
      </g>

      <g data-step data-step-label="沿 key 轴做 softmax">
        <text x="35" y="325" font-size="18" font-weight="700" fill="currentColor">3. 归一化：沿 n_k 轴做 softmax</text>
        <rect x="310" y="292" width="300" height="78" rx="12" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="2" />
        <text x="460" y="322" text-anchor="middle" font-size="20" fill="currentColor">注意力权重 α</text>
        <text x="460" y="349" text-anchor="middle" font-size="17" fill="currentColor">B × h × n_q × n_k</text>
        <path d="M850 267 C850 325 700 330 610 330" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#attention-pipeline-arrow)" />
        <text x="690" y="310" font-size="16" fill="currentColor">每行合法 key 权重和为 1</text>
      </g>

      <g data-step data-step-label="权重加权 V 得到输出">
        <text x="35" y="430" font-size="18" font-weight="700" fill="currentColor">4. 读取内容：αV</text>
        <rect x="240" y="405" width="230" height="72" rx="11" fill="none" stroke="currentColor" stroke-width="2" />
        <text x="355" y="434" text-anchor="middle" font-size="20" fill="currentColor">V</text>
        <text x="355" y="459" text-anchor="middle" font-size="16" fill="currentColor">B × h × n_k × d_v</text>
        <rect x="695" y="405" width="300" height="72" rx="11" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="2" />
        <text x="845" y="434" text-anchor="middle" font-size="20" fill="currentColor">输出 αV</text>
        <text x="845" y="459" text-anchor="middle" font-size="16" fill="currentColor">B × h × n_q × d_v</text>
        <path d="M470 441 L695 441 M520 370 C555 405 610 420 695 430" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#attention-pipeline-arrow)" />
      </g>
    </svg>
  </div>
  <ol data-lesson-steps>
    <li>Q 的 n_q 个 query 与 K 的 n_k 个 key 做点积，得到形状 B × h × n_q × n_k 的 QKᵀ。</li>
    <li>分数除以 √d_k，再加掩码 M；合法位置加 0，禁止位置加 −∞，形状不变。</li>
    <li>沿 n_k 轴做 softmax，得到同形状权重，每个 query 的合法 key 权重和为 1。</li>
    <li>权重乘 V，消去 n_k 轴，输出形状变为 B × h × n_q × d_v。</li>
  </ol>
  <figcaption>图 1：按 Q/K → score → mask/softmax → V/output 的顺序核对；n_k 在最后一次加权和中被消去。</figcaption>
</figure>

## 为什么除以 $\sqrt{d_k}$

假设 $q_r,k_r$ 各维相互独立、均值为 0、方差为 1。点积是

$$
\mathbf q^\top\mathbf k=\sum_{r=1}^{d_k}q_rk_r.
$$

每一项均值为 0、方差为 1，于是点积方差为 $d_k$，标准差随 $\sqrt{d_k}$ 增长。$d_k$ 大时，未缩放 logits 更容易出现很大的绝对值，把 softmax 推进接近 one-hot 的饱和区，较小权重对应的梯度会很弱。

除以 $\sqrt{d_k}$ 后，在上述近似下分数方差回到 1 的量级。这个推导解释了尺度，不声称真实训练中的 $Q,K$ 严格独立或单位方差；[归一化](../03-training/normalization.md)和初始化仍会影响实际数值。

## 掩码必须在 softmax 前生效

掩码定义“哪些 query–key 配对合法”。加性写法通常令合法位置 $M_{ij}=0$，非法位置 $M_{ij}=-\infty$，这样 softmax 后非法权重恰为 0。

常见的两类语义是：

- padding mask：屏蔽为了凑齐 batch 而添加的 key 位置；
- causal mask：第 $i$ 个位置不能读取未来的 $j>i$，防止训练时泄漏答案。

先 softmax 再把非法权重乘 0 会破坏“剩余权重和为 1”，除非重新归一化；工程上直接在 logits 上遮蔽更清楚。加性掩码可统一理解为“允许为 0、屏蔽为 $-\infty$”，但布尔掩码必须看 API：PyTorch `F.scaled_dot_product_attention` 的 `attn_mask=True` 表示该位置允许参与注意力；`nn.MultiheadAttention` 的布尔 `attn_mask` 或 `key_padding_mask=True` 则表示该位置被屏蔽。不要把一个函数的布尔张量原样传给另一个函数。若某个 query 的整行 key 都被遮蔽，softmax 没有合法分布可返回，常会产生 `NaN` 或未定义结果；数据管线应保证每行至少一个有效位置，或显式处理空行。

下面的手写计算只展示 `masked_fill` 的本地选择器语义；`blocked_selector=True` 表示把该 logit 填为 $-\infty$，它不是任何 PyTorch attention API 的 `attn_mask`：

```python
import math
import torch

B, heads, n, d = 2, 4, 6, 8
q = torch.randn(B, heads, n, d)
k = torch.randn(B, heads, n, d)
v = torch.randn(B, heads, n, d)

scores = q @ k.transpose(-2, -1) / math.sqrt(d)
blocked_selector = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
scores = scores.masked_fill(blocked_selector, float("-inf"))  # 本地选择器：True 屏蔽未来
weights = torch.softmax(scores, dim=-1)
out = weights @ v

print(weights.shape)  # torch.Size([2, 4, 6, 6])
print(out.shape)      # torch.Size([2, 4, 6, 8])
```

## 归纳偏置与二次成本

注意力偏爱按内容相似度路由信息，而不是预先规定固定邻域。在全局 self-attention 中，任意两个位置一层内就能建立直接路径；代价是每对位置都要产生分数。它本身若没有[位置编码](./positional-encoding.md)或位置相关 mask，并不知道 token 的先后次序。

单个批量多头操作的两次核心乘法成本约为

$$
O\!\left(Bh\,n_qn_k(d_k+d_v)\right).
$$

self-attention 取 $n_q=n_k=n$ 且 $hd_k,hd_v$ 与模型维度 $D$ 同量级时，常简写成 $O(Bn^2D)$。更关键的显存项是分数或权重矩阵

$$
O(Bh\,n_qn_k),
$$

全局 self-attention 因而有 $O(Bhn^2)$ 的 score memory。Q/K/V 和输出投影通常还有 $O(BnD^2)$ 级计算；只报 $n^2$ 而完全忽略特征投影，也可能误判短序列、宽模型的瓶颈。高效内核可以避免把全部中间矩阵长期写回显存，但不会改变每个 query 与所有 key 做全局精确匹配的基本成对工作量。

## 常见失败方式

- 忘记除以 $\sqrt{d_k}$，导致大维度下 logits 过尖、softmax 梯度弱。
- causal mask 方向写反，训练损失异常好，部署自回归时却崩溃；用一个 $4\times4$ 小矩阵打印验证最可靠。
- padding 只遮 query、不遮 key，使有效位置继续读取补齐内容。
- 一整行全被遮蔽，产生非有限权重。
- 长序列直接建立全局分数矩阵导致显存溢出；先按真实 $B,h,n$ 估算，而不是只看参数量。
- 把注意力权重直接当作可靠因果解释；它首先是计算中的路由系数，解释结论需要额外验证。

!!! tip "面试 / 工程重点"
    说明注意力复杂度时要分计算和内存：核心 self-attention 计算约 $O(Bn^2D)$，显式 score/weight 内存约 $O(Bhn^2)$。头数通常被包含在 $D=h d$ 的计算简写里，却仍直接出现在分数矩阵内存中。

下一页只比较顺序信息怎样进入这一操作；完整的多层 Transformer 结构不在本页展开。
