---
level: intermediate
roles:
  - core
  - interview
prerequisites:
  - 泛化与数据
  - 优化器与学习率
estimated_time: 35min
status: complete
---

# 正则化

!!! info "参考资料"
    - Nitish Srivastava et al., [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html), JMLR 2014
    - Ilya Loshchilov, Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019

## 直觉 (Intuition)

训练集损失很低，只说明模型记住或解释了训练样本。正则化通过限制参数、扰动训练输入或改变停止时点，让模型不那么容易依赖训练集里的偶然细节。不同方法动的是不同环节：weight decay 改参数更新，Dropout 改网络路径，数据增强改输入分布，早停改训练时长，标签平滑改监督目标。它们不能只按“都防过拟合”混成一类旋钮。

数据划分、泄漏、欠拟合与过拟合的判断先见[泛化与数据](../01-machine-learning-basics/generalization-and-data.md)。若验证集已经参与了训练样本选择或增强泄漏，增加正则强度并不能修复评价协议。

## 五种方法分别约束什么

| 方法 | 直接作用对象 | 训练时发生什么 | 推理时是否额外操作 | 更强时的典型代价 |
|---|---|---|---|---|
| Weight decay | 参数大小 | 每步收缩指定参数 | 否 | 过强会欠拟合 |
| Dropout | 隐藏单元/连接 | 随机屏蔽并缩放保留值 | 关闭随机屏蔽 | 噪声过大，收敛变慢 |
| 数据增强 | 输入样本 | 采样保持标签语义的变换 | 通常否 | 错误变换会改掉标签 |
| 早停 | 训练时长 | 用验证表现选择停止点 | 使用保存的最佳检查点 | 偶然波动导致过早停止 |
| 标签平滑 | 监督目标 | one-hot 标签变得不那么极端 | 否 | 过强会压低可分性与置信度 |

## Weight decay：限制参数尺度

解耦权重衰减每一步直接做近似收缩：

$$
\theta\leftarrow(1-\eta\lambda)\theta+\text{优化器给出的更新}.
$$

$\lambda$ 太小几乎不起作用，太大则不断把有用权重拉回零。它与损失里 L2 惩罚的等价边界，已在[优化器与学习率](./optimization.md)中推导。实践中常只衰减矩阵型权重，不衰减偏置和归一化的 $\gamma,\beta$；应显式检查参数分组，不能凭优化器名字猜。

Weight decay 会影响参数范数，但“范数小”并不自动等于“泛化好”。在有归一化或尺度不变性的网络里，参数尺度与函数尺度的关系更复杂，因此仍要由独立验证集选择强度。

## Dropout：训练时随机删掉路径

设保留概率为 $q=1-p$，训练时为每个激活采样 $m_i\sim\operatorname{Bernoulli}(q)$，采用 inverted dropout：

$$
h_i'=\frac{m_i}{q}h_i.
$$

于是 $\mathbb E[h_i']=h_i$。推理时不再随机置零，也不必额外乘 $q$。例如 $p=0.2$ 时保留概率为 0.8，保留下来的激活乘 $1/0.8=1.25$；若误把 $p$ 当保留率，正则强度会完全相反。

随机屏蔽迫使表示不要只依赖某一条共适应路径，但也给梯度增加噪声。卷积网络配合 BatchNorm 时未必需要较大 Dropout；Transformer 则常在注意力权重、残差分支或 MLP 输出处使用。位置和概率都要由架构与数据规模决定。

```python
import torch
from torch import nn

drop = nn.Dropout(p=0.5)
x = torch.ones(8)

drop.train()
print(drop(x))  # 元素为 0 或 2；具体掩码每次不同

drop.eval()
print(drop(x))  # tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```

忘记切换评估模式会让同一输入得到随机预测。训练/评估状态的完整顺序见[Mini-batch 与训练循环](./minibatch-and-training-loop.md)。

## 数据增强：把不变性写进样本

图像左右翻转若不改变类别，就能把这种不变性通过训练样本告诉模型；语音加背景噪声、文本做受约束的扰动也属同一思路。它既是正则化，也是[归纳偏置](../02-neural-network-foundations/inductive-bias.md)：你在声明哪些变化不应改变输出。

增强必须与任务语义一致。数字 6 旋转后可能变成 9，医学影像的左右方向可能有临床含义，时间序列也不能随意打乱。验证和测试通常不采样随机训练增强，只做确定性的预处理；若使用 test-time augmentation，则要把它作为单独的推理方案报告。

## 早停：选择训练轨迹上的一个点

每隔固定 step 或 epoch 计算验证指标，保存到目前为止最好的检查点；若连续若干次没有超过最优值，就停止。这个等待次数常叫 patience。

早停需要三个细节：

- 监控真正关心的验证指标，并明确越大越好还是越小越好；
- 给随机波动留出 `min_delta` 与 patience，不要一次变差就停；
- 结束后恢复最佳检查点，而不是直接使用最后一次参数。

验证集参与了停止时点选择，因此不能再把它当最终无偏测试集。反复尝试许多配置也会逐渐对验证集过拟合。

## 标签平滑：让目标不过度极端

$K$ 类分类中，一种常见定义是

$$
\tilde{\mathbf y}=(1-\epsilon)\mathbf y_{\text{one-hot}}+\frac{\epsilon}{K}\mathbf 1.
$$

真实类目标从 1 变为 $1-\epsilon+\epsilon/K$，其余类为 $\epsilon/K$。也有实现把 $\epsilon$ 只分给错误类，阅读论文或 API 时要核对约定。

标签平滑能抑制过度尖锐的训练目标，并可能改善泛化；但它不是概率校准的保证，太强还会削弱模型区分类别的能力。类别本身存在不确定性时，软标签应尽量来自合理的标注分布，而不是把所有错误类一律当作同样相似。

!!! tip "面试 / 工程重点"
    正则化与归一化不是同义词。正则化主要约束可学习解或训练过程以改善泛化；归一化重标定中间激活，主要改变数值尺度与优化行为。BatchNorm 可能带来正则化副作用，但这不改变两类机制的区别。

## 怎么组合而不失去判断力

先用无正则或弱正则基线确认模型能拟合一个小数据子集，再逐项加入。训练损失升高、验证表现改善，往往说明约束起效；训练与验证都变差，可能已经过强。最终结论要靠受控消融，而不是同时把 weight decay、Dropout、增强与标签平滑全部加满。

这些方法都不负责重新定义激活的统计轴。下一页的[归一化](./normalization.md)专门解释 BatchNorm、LayerNorm 与 RMSNorm 在哪里计算统计量，以及为什么训练/推理行为不同。
