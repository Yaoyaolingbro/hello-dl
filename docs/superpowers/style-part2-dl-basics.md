---
title: Part 2 基础深度学习 · 写作规范
applies_to: 02-deep-learning/**
---

# Part 2 基础深度学习写作规范

> 本文件与 `writing-style.md` 配合使用，**不重复通用规则，只写 Part 2 特有的要求**。

---

## 核心定位

读完每一节，读者应该能**自己写出对应的 PyTorch 代码**，而不只是"听懂了"。

理论是为了让代码写对，代码是为了让理论落地。两者交替出现，不要先写完所有理论再写代码。

---

## 章节结构

```
## 直觉 (Intuition)

## 核心原理                  ← 数学推导，但以帮助理解代码为目的

## PyTorch 实现              ← 每个核心概念都有对应代码

## 训练实验 / 可视化         ← 展示关键超参数的影响（可选）

## 面试 / 工程要点           ← 高频考点和工程陷阱
```

---

## 代码规范（最重要的部分）

**每个核心概念必须有 PyTorch 代码。** 不是"可以有"，是"必须有"。

### 代码粒度

按从细到粗三个层次写：

1. **手动实现**：用 NumPy 或纯 PyTorch 手写关键算法（如手动反向传播、手写 Attention），帮助读者真正理解机制
2. **简洁实现**：用 PyTorch 标准 API 复现，展示生产级写法
3. **训练循环**：完整的 mini-batch 训练循环，标注每行在干什么

```python
# 手动实现 softmax（理解数值稳定性）
def softmax_manual(x):
    x = x - x.max(dim=-1, keepdim=True).values  # 减最大值防溢出
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=-1, keepdim=True)

# 对比 PyTorch 内置（生产代码用这个）
probs = F.softmax(logits, dim=-1)
```

### 代码注释原则

注释说**为什么**，不说是什么：

```python
# ✗ 差：# 计算 softmax
# ✓ 好：# 减最大值是为了防止 exp 溢出，不影响结果因为分子分母同除

hidden = self.norm(x + self.attn(x))   # 先残差再归一化（Pre-Norm）
# ✓ 也好：# Post-Norm 在原始 Transformer 里，Pre-Norm 训练更稳定
```

### 输出展示

关键代码的输出结果必须写在注释里，读者不跑代码也能知道结论：

```python
print(model.parameters().__next__().shape)  # torch.Size([512, 256])
print(f"参数量：{sum(p.numel() for p in model.parameters()):,}")  # 参数量：1,048,576
```

---

## 数学与代码的配合

数学公式和代码要紧挨着，不要把公式堆在前面、代码堆在后面：

```
公式（Attention 计算）
  ↓
代码（实现这个公式）
  ↓
文字（解释为什么这样实现，以及有什么需要注意的）
  ↓
公式（Multi-head Attention）
  ↓
代码（Multi-head 实现）
```

---

## 消融式解释

遇到超参数或设计选择时，**说清楚改掉会怎样**：

> 学习率设为 0.001 是经验值。太大（比如 0.1）会跳过最小值来回震荡；太小（比如 1e-6）收敛慢到不现实。实际调参时通常先跑 warmup，再衰减。

不要只说"我们设 lr=0.001"，要说"为什么是这个值，偏了会发生什么"。

---

## 面试 / 工程重点的取舍

Part 2 的面试题密度最高，但要**有节制**：

- 每节最多 2-3 个 `!!! tip "面试 / 工程重点"` 块
- 只写真正高频的问题，不要把所有边角细节都塞进去
- 工程陷阱（如 BatchNorm 在 eval 模式忘记切换）> 纯理论问题

```markdown
!!! tip "面试高频"
    反向传播里，为什么要先 `optimizer.zero_grad()`？
    
    PyTorch 默认**累积**梯度，不清零就会把多次 forward 的梯度加在一起。
    在 RNN 里有时候故意不清零来实现 truncated BPTT，但大多数情况这是 bug。
```

---

## 交稿前额外检查

- [ ] 每个核心概念有 PyTorch 代码吗？
- [ ] 代码注释说的是 WHY 而不是 WHAT 吗？
- [ ] 关键代码的输出写在注释里了吗？
- [ ] 有没有至少一处"改掉会怎样"的消融式解释？
- [ ] `!!! tip` 不超过 3 个吗？每个都是真正高频的考点吗？
