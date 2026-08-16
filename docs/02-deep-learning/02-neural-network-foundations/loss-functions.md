---
level: basic
roles:
  - core
  - interview
prerequisites:
  - 分类与回归
estimated_time: 35min
status: complete
---

# 损失函数

!!! info "参考资料"
    - [Understanding Deep Learning](https://udlbook.github.io/udlbook/) — 第 5 章
    - [BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) 与 [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — PyTorch Documentation

## 直觉 (Intuition)

损失函数把“预测得有多差”变成训练可以优化的数字。它不是评价指标的别名：训练需要可微、信号稳定，业务评价可能关心阈值后的成本。选择损失等于规定模型应该把能力花在哪里。先确定标签的统计含义，再选公式。

## 回归损失

均方误差 (MSE) 对单个样本写成

$$
\ell_{\mathrm{MSE}}=(\hat y-y)^2.
$$

对预测求导得到 $2(\hat y-y)$，误差越大，梯度越大。它会强烈追赶离群点；若噪声近似高斯，最小化 MSE 也对应最大化似然。

平均绝对误差 (MAE) 为

$$
\ell_{\mathrm{MAE}}=|\hat y-y|.
$$

它对大误差线性惩罚，更抗异常值，但在零点不可导，梯度大小也不随误差距离变化。Smooth L1/Huber loss 在小误差区使用平方项、大误差区使用线性项，是常见折中。

## 二分类交叉熵

模型给正类概率 $p$，标签 $y\in\{0,1\}$。伯努利负对数似然为

$$
\ell=-y\log p-(1-y)\log(1-p).
$$

当 $y=1$ 时，只剩 $-\log p$；模型越不相信正确类别，损失越大。若网络输出 logit $z$，概率是 $p=\sigma(z)$。

实现时不要先手算 sigmoid 再取 log。很大的正负 logit 会让概率舍入到 0 或 1，随后出现 $\log 0$。`BCEWithLogitsLoss` 把两步合并，用 log-sum-exp 形式保持数值稳定。

## 多分类交叉熵

模型输出 $K$ 个 logits $\mathbf z$。softmax 把它们变成概率：

$$
p_k=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}.
$$

真实类别为 $y$ 时，单样本损失是

$$
\ell=-\log p_y
=-z_y+\log\sum_{j=1}^{K}e^{z_j}.
$$

第二种写法解释了为什么框架直接接收 logits。PyTorch 的 `CrossEntropyLoss` 已经包含 `log_softmax`，输入前再做 softmax 会改变数值并让训练信号变差。

!!! tip "面试重点"
    softmax 交叉熵对 logit $z_k$ 的梯度是 $p_k-\mathbb I[k=y]$。预测概率与 one-hot 标签的差，直接成为输出层误差信号。

## 类别不平衡不是只换一个指标

少数类代价更高时，可以给正类或各类别加权，也可以调整采样。权重改变训练目标，阈值改变部署决策，两者不要混为一谈。指标和阈值的完整讨论见[指标与决策阈值](../06-evaluation-debugging/metrics-and-thresholds.md)。

损失决定输出端的信号；信号能否穿过多层网络，还取决于[激活函数](./activations.md)的梯度行为。
