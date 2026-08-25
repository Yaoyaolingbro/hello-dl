---
level: intermediate
roles:
  - core
  - interview
prerequisites:
  - 反向传播
  - Mini-batch 与训练循环
estimated_time: 45min
status: complete
---

# 优化器与学习率

!!! info "参考资料"
    - Diederik P. Kingma, Jimmy Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), ICLR 2015
    - Ilya Loshchilov, Frank Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), ICLR 2019
    - [torch.optim](https://docs.pytorch.org/docs/stable/optim.html) — PyTorch Documentation

## 直觉 (Intuition)

反向传播给出当前 mini-batch 的梯度，优化器决定怎样把这份局部信息变成参数位移。SGD 直接走；Momentum 记住近期方向；Adam 还按每个参数的历史梯度尺度调整步幅；AdamW 在 Adam 之外独立收缩权重。它们没有脱离学习率：再聪明的方向估计，乘上不合适的步长仍会发散或停滞。

这一页只比较深度学习训练中的使用边界。梯度下降、随机梯度方差与收敛条件的完整背景见 Part 1 的[梯度下降](../../01-math/optimization/gradient-descent.md)和[自适应优化方法](../../01-math/optimization/adaptive-methods.md)。

## 从 SGD 到 Momentum

令第 $t$ 个 optimizer step 的 mini-batch 梯度为 $\mathbf g_t$。普通 SGD 更新为

$$
\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta_t\mathbf g_t.
$$

它几乎不保存额外状态，更新含义也最直接。损失面狭长时，梯度可能在陡峭方向左右摆动、在平缓方向前进很慢。Momentum 用速度变量积累一致方向：

$$
\mathbf v_t=\mu\mathbf v_{t-1}+\mathbf g_t,
\qquad
\boldsymbol\theta_{t+1}=\boldsymbol\theta_t-\eta_t\mathbf v_t.
$$

$\mu$ 常取接近 1 的值，如 0.9。它不是把学习率问题消掉：$\eta$ 太大时，惯性还可能让越界后的震荡持续更久。不同框架对动量是否带 $(1-\mu)$ 缩放有约定差异，比较公式或迁移超参数时要先核对实现。

## Adam 在每个坐标上调步幅

Adam 同时维护梯度的一阶矩与平方梯度的二阶原始矩：

$$
\mathbf m_t=\beta_1\mathbf m_{t-1}+(1-\beta_1)\mathbf g_t,
$$

$$
\mathbf v_t=\beta_2\mathbf v_{t-1}+(1-\beta_2)\mathbf g_t^2.
$$

初值为零会让训练早期的移动平均偏小，所以要做偏差修正：

$$
\hat{\mathbf m}_t=\frac{\mathbf m_t}{1-\beta_1^t},
\qquad
\hat{\mathbf v}_t=\frac{\mathbf v_t}{1-\beta_2^t}.
$$

逐元素更新为

$$
\boldsymbol\theta_{t+1}
=\boldsymbol\theta_t
-\eta_t\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\epsilon}.
$$

分母让历史平方梯度大的坐标步子变小，适合梯度尺度差异大或稀疏的参数。代价是每个参数通常要保存两份状态，且“有效步幅”受历史统计影响，不再能只看全局学习率。Adam 常是方便的起点，但不保证在每个任务上比调好的 SGD 泛化更好。

| 方法 | 保存的状态 | 每坐标自适应 | 常见取舍 |
|---|---:|---:|---|
| SGD | 无 | 否 | 省内存、易解释；对学习率和曲率较敏感 |
| Momentum | 1 份速度 | 否 | 抑制来回震荡；多一组状态和动量超参 |
| Adam | 一阶矩 + 二阶矩 | 是 | 早期进展常较快；状态内存大，需理解偏差修正 |
| AdamW | Adam 状态 | 是 | 另加解耦衰减；现代 Transformer 常用 |

## L2 正则与权重衰减何时等价

这两个词经常被 API 混用，但算法含义不同。给目标加 L2 惩罚

$$
L_{\text{total}}(\theta)=L(\theta)+\frac{\lambda}{2}\|\theta\|_2^2
$$

会让梯度变成 $\mathbf g_t+\lambda\boldsymbol\theta_t$。代入普通 SGD：

$$
\boldsymbol\theta_{t+1}
=(1-\eta_t\lambda)\boldsymbol\theta_t-\eta_t\mathbf g_t.
$$

这与每步先按 $1-\eta_t\lambda$ 收缩参数的 weight decay 相同；换句话说，对**普通 SGD**，两者在把衰减系数按学习率对应起来后等价。Momentum 的状态定义和框架实现会影响参数对应，不能脱离具体公式笼统声称完全相同。

在 Adam 中，把 $\lambda\theta$ 加进损失梯度后，它会进入一阶、二阶矩并被逐坐标预条件；不同坐标不再受到同样的直接收缩。AdamW 把衰减从梯度估计中拿出来：

$$
\boldsymbol\theta_{t+1}
=(1-\eta_t\lambda)\boldsymbol\theta_t
-\eta_t\frac{\hat{\mathbf m}_t}{\sqrt{\hat{\mathbf v}_t}+\epsilon}.
$$

所以 L2 惩罚与 decoupled weight decay 在 Adam 这类自适应优化器下通常不等价。AdamW 的 `weight_decay` 表示后者；具体框架是否对所有参数组衰减，仍要检查配置。偏置与归一化层的缩放/平移参数常被放入不衰减参数组，这是一种实践选择，不是 AdamW 公式自动做的事。

!!! tip "面试 / 工程重点"
    回答“L2 是否等于 weight decay”时要说出边界：对普通 SGD，适当对应学习率与系数后等价；对带逐坐标预条件的 Adam，一般不等价，AdamW 才把衰减从梯度更新中解耦。

## 学习率、warmup 与衰减

学习率决定每步能把当前局部近似相信到什么程度。过大时，训练损失会震荡、突然变成非有限值；过小时，曲线可能稳定却几乎不动。优化器选择不能替代学习率搜索。

常见调度只需抓住时间尺度：

| 策略 | 训练中的行为 | 容易踩的坑 |
|---|---|---|
| 固定学习率 | 全程不变，适合短基线 | 后期可能围绕较优区域震荡 |
| Step decay | 到里程碑后乘常数 | epoch 与 step 配错会提前衰减 |
| Cosine decay | 从峰值平滑降到较小值 | `T_max` 必须对应实际调用次数 |
| Warmup + decay | 前若干 step 从小值升到峰值，再衰减 | warmup 过长会浪费大量训练预算 |

Warmup 常用于大 batch、深层残差网络或自适应状态尚未稳定的训练开头。它缓和早期的大更新，不是修复错误梯度的工具。线性 warmup 可写成

$$
\eta_t=\eta_{\max}\frac{t}{T_w},\qquad 1\le t\le T_w.
$$

之后可接余弦衰减：

$$
\eta_t=\eta_{\min}
+\frac12(\eta_{\max}-\eta_{\min})
\left[1+\cos\left(\pi\frac{t-T_w}{T-T_w}\right)\right].
$$

调度器应按其定义在[训练循环](./minibatch-and-training-loop.md)里推进。用了梯度累积后，warmup 若按 optimizer step 计数，就只在真正 `optimizer.step()` 时前进，不能按每个 micro-batch 多走几次。

## 实际选择

- 先建立可比较的基线：固定模型、数据顺序、训练 step 与评价协议，再换优化器。
- CNN 等监督视觉任务常值得比较 SGD + Momentum 与 AdamW；Transformer 通常从 AdamW 起步，但仍需调学习率、warmup 和衰减。
- 同时改变优化器、峰值学习率、batch size 与训练轮数，最后无法知道收益来自哪里。
- 优化器负责寻找参数，weight decay 等手段怎样影响泛化，下一页[正则化](./regularization.md)单独讨论。
