# 信息论

!!! info "参考资料"
    **主要资料**

    - [Deep Learning Book: Chapter 3.13](https://www.deeplearningbook.org/contents/prob.html) — Ian Goodfellow et al.
    - Shannon, "A Mathematical Theory of Communication", 1948 — 信息论的奠基论文

    **工具文档**
    
    - [SciPy: `scipy.stats.entropy`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)

## 直觉 (Intuition)

信息论回答的是：一条信息有多少"信息量"，两个分布有多大的差距。一件非常罕见的事发生了，它传递的信息量很大；一件必然发生的事发生了，它传递的信息量为零。熵、交叉熵、KL 散度是这套思想的三个核心量，它们直接出现在分类损失、生成模型目标和对比学习的 loss 里。

## 主要符号

| 符号 | 含义 |
|------|------|
| $H(P)$ | 分布 $P$ 的（香农）熵 |
| $H(P, Q)$ | 分布 $P$ 和 $Q$ 之间的交叉熵 |
| $D_\text{KL}(P \| Q)$ | $P$ 相对于 $Q$ 的 KL 散度 |
| $I(X; Y)$ | 随机变量 $X$ 和 $Y$ 的互信息 |

## 自信息

一个事件发生的**自信息（信息量）**定义为：

$$
I(x)
=
-\log p(x)
$$

概率越低的事件，信息量越大。当底数为 2 时单位是比特（bit），以 $e$ 为底时单位是奈特（nat），深度学习里通常用 $\ln$（自然对数）。

直觉：你对一件"几乎不可能发生"的事感到惊讶，说明它传递了大量信息。一件必然发生的事（$p=1$）信息量为 0。

## 熵

**熵（Shannon Entropy）**是自信息的期望，度量整个分布的不确定性：

$$
H(P)
=
-\sum_{x} p(x) \log p(x)
=
\mathbb{E}_P[-\log p(x)]
$$

- 均匀分布的熵最大：所有事件等可能，不确定性最高
- 确定性分布（一个事件概率为 1）熵为零：没有不确定性

对连续分布，熵的对应概念是**微分熵**：$h(P) = -\int p(x)\log p(x)\, dx$。

## 交叉熵

**交叉熵**度量"用分布 $Q$ 编码来自分布 $P$ 的数据"的平均编码长度：

$$
H(P, Q)
=
-\sum_{x} p(x) \log q(x)
=
\mathbb{E}_P[-\log q(x)]
$$

当 $Q = P$ 时，交叉熵等于熵 $H(P)$，这是最优编码。当 $Q \ne P$ 时，需要更多比特来编码。

深度学习里的**分类交叉熵损失**就是这个。真实标签 $P$ 是 one-hot 分布，模型输出 $Q$ 是 softmax 概率：

$$
\mathcal{L}_\text{CE}
=
-\sum_{k} y_k \log \hat{p}_k
=
-\log \hat{p}_{y}
$$

其中 $y$ 是正确类别，$\hat{p}_y$ 是模型给正确类别的概率。最小化交叉熵等价于最大化似然。

!!! note "交叉熵 = 熵 + KL 散度"
    $$H(P, Q) = H(P) + D_\text{KL}(P \| Q)$$

    因为真实标签的熵 $H(P)$ 是常数（不依赖模型参数），最小化交叉熵等价于最小化 KL 散度。

## KL 散度

**KL 散度（相对熵）**直接度量两个分布的差距：

$$
D_\text{KL}(P \| Q)
=
\sum_{x} p(x) \log \frac{p(x)}{q(x)}
=
\mathbb{E}_P\!\left[\log \frac{p(x)}{q(x)}\right]
$$

性质：

1. $D_\text{KL}(P \| Q) \ge 0$，等号成立当且仅当 $P = Q$（Gibbs 不等式）
2. **不对称**：$D_\text{KL}(P \| Q) \ne D_\text{KL}(Q \| P)$

不对称性很重要。$D_\text{KL}(P \| Q)$ 叫"前向 KL"，当 $q(x) = 0$ 而 $p(x) > 0$ 时值为无穷，迫使 $Q$ 覆盖 $P$ 的全部支撑（均值求解行为）。$D_\text{KL}(Q \| P)$ 叫"反向 KL"，迫使 $Q$ 集中在 $P$ 的某个模（众数求解行为）。VAE 的 ELBO 包含 $D_\text{KL}(q_\phi \| p)$（反向 KL）。

## 互信息

**互信息**度量两个随机变量之间共享的信息量：

$$
I(X; Y)
=
D_\text{KL}\!\left(p(X,Y) \| p(X)p(Y)\right)
=
\sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

$I(X; Y) = 0$ 当且仅当 $X$ 和 $Y$ 独立。互信息是相关系数的非线性推广，能捕捉任何形式的依赖关系，而不只是线性依赖。

对比学习（如 CPC、SimCLR 的理论分析）常用互信息作为表征学习的目标：让同一样本不同视角的 embedding 之间的互信息最大化。

## 代码验证

```python
import numpy as np
from scipy.stats import entropy

# 熵：均匀分布熵最大
p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
p_skewed  = np.array([0.97, 0.01, 0.01, 0.01])

print(f"均匀分布熵: {entropy(p_uniform):.4f}")  # 1.3863 nat（= log 4）
print(f"偏斜分布熵: {entropy(p_skewed):.4f}")   # 0.1416 nat

# 交叉熵损失
y_true = np.array([0, 1, 0, 0])    # one-hot，类别 1
y_pred = np.array([0.1, 0.7, 0.1, 0.1])  # 模型输出

ce = -np.sum(y_true * np.log(y_pred + 1e-9))
print(f"交叉熵: {ce:.4f}")  # -log(0.7) ≈ 0.3567

# KL 散度：scipy 的 entropy(p, q) = D_KL(P||Q)
p = np.array([0.4, 0.3, 0.2, 0.1])
q = np.array([0.25, 0.25, 0.25, 0.25])

kl_pq = entropy(p, q)  # D_KL(P||Q)
kl_qp = entropy(q, p)  # D_KL(Q||P)
print(f"KL(P||Q): {kl_pq:.4f}")  # 0.0853
print(f"KL(Q||P): {kl_qp:.4f}")  # 0.0870  <- 不对称

# 验证：H(P,Q) = H(P) + KL(P||Q)
hp  = entropy(p)
hpq = -np.sum(p * np.log(q))
print(np.isclose(hpq, hp + kl_pq))  # True
```

## 在深度学习中的应用

分类交叉熵是最常用的分类损失，本质是 KL 散度（固定数据分布时）。VAE 的训练目标 ELBO 包含 KL 散度项，约束隐变量分布接近高斯先验。知识蒸馏（Knowledge Distillation）用教师模型的软标签（soft label）替代 one-hot，最小化学生和教师输出分布的 KL 散度。互信息最大化是 SimCLR、CLIP 等对比学习方法的理论基础。

下一节讲随机过程与马尔科夫链，把概率从"一个时刻"推广到"随时间演化的序列"，是 RL 和扩散模型的基础语言。
