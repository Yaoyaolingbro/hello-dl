# 自适应优化方法

!!! info "参考资料"
    **主要论文**

    - Kingma & Ba, "Adam: A Method for Stochastic Optimization", ICLR 2015 — Adam 原始论文
    - Reddi et al., "On the Convergence of Adam and Beyond", ICLR 2018 — Adam 的收敛性分析与 AMSGrad
    - Loshchilov & Hutter, "Decoupled Weight Decay Regularization", ICLR 2019 — AdamW

    **工具文档**
    
    - [PyTorch Optimizers](https://pytorch.org/docs/stable/optim.html)

## 直觉 (Intuition)

SGD 对所有参数用同一个学习率，但参数的重要性和更新频率差异很大——文本模型里罕见 token 的 embedding 需要大步更新，而频繁出现的 token 应该小步微调。自适应方法为每个参数维护自己的有效学习率，基于历史梯度信息动态调整。Adam 把动量（一阶矩）和自适应学习率（二阶矩）结合，成为深度学习的默认优化器；AdamW 修复了 Adam 的权重衰减实现错误，是现代大模型训练的标配。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbf{m}_t$ | 梯度的一阶矩估计（动量） |
| $\mathbf{v}_t$ | 梯度平方的二阶矩估计 |
| $\beta_1, \beta_2$ | 一阶/二阶矩的指数衰减率 |
| $\epsilon$ | 数值稳定项（防止除以零） |
| $\lambda$ | 权重衰减系数 |

## AdaGrad

**AdaGrad**（Duchi et al., 2011）累积历史梯度平方，为每个参数单独缩放学习率：

$$
\mathbf{G}_t = \mathbf{G}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t
$$

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{\mathbf{G}_t} + \epsilon} \odot \mathbf{g}_t
$$

历史梯度大的参数（频繁更新的维度）学习率自动缩小，历史梯度小的参数（罕见更新的维度）学习率保持大。问题是 $\mathbf{G}_t$ 单调递增，训练后期所有参数的学习率都趋向零，导致过早停止更新。

## RMSProp

**RMSProp**（Hinton, 2012）用指数移动平均替代累积，解决 AdaGrad 学习率归零的问题：

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1-\beta_2)\, \mathbf{g}_t \odot \mathbf{g}_t
$$

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{\mathbf{v}_t} + \epsilon} \odot \mathbf{g}_t
$$

$\beta_2$ 控制"遗忘速度"（通常取 0.99），使有效学习率随时间平稳而不衰减到零。

## Adam

**Adam**（Adaptive Moment Estimation）同时维护一阶矩（动量）和二阶矩（自适应缩放）：

$$
\mathbf{m}_t = \beta_1\, \mathbf{m}_{t-1} + (1-\beta_1)\, \mathbf{g}_t
$$

$$
\mathbf{v}_t = \beta_2\, \mathbf{v}_{t-1} + (1-\beta_2)\, \mathbf{g}_t \odot \mathbf{g}_t
$$

初始阶段 $\mathbf{m}_t, \mathbf{v}_t$ 偏向零，用**偏差修正（bias correction）**补偿：

$$
\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}
$$

更新规则：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\, \hat{\mathbf{m}}_t
$$

默认超参数：$\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$，$\alpha = 3 \times 10^{-4}$。

!!! note "Adam 的有效学习率"
    每个参数的有效学习率约为 $\alpha \cdot |\hat{m}_i| / \sqrt{\hat{v}_i}$。当梯度方向一致（$\hat{m}_i$ 大，$\hat{v}_i$ 小），步子大；当梯度方向来回震荡（$\hat{m}_i$ 小，$\hat{v}_i$ 大），步子小。Adam 自动识别"稳定方向"和"震荡方向"。

## AdamW：修复权重衰减

标准 Adam 里，L2 正则化（在损失上加 $\lambda\|\mathbf{x}\|^2$）会让权重衰减的梯度被自适应缩放所稀释，实际效果远弱于 SGD + L2。**AdamW** 把权重衰减从梯度中解耦出来，直接作用在参数上：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \frac{\alpha}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon}\, \hat{\mathbf{m}}_t - \alpha\lambda\, \mathbf{x}_t
$$

解耦后的权重衰减与梯度的大小无关，各参数的正则化强度均匀。BERT、GPT、LLaMA 等大模型全部使用 AdamW。

## Lion 与 Sophia

近年提出了一些更高效的变体：

- **Lion**（Chen et al., 2023）：只用梯度和动量的符号更新，内存占用比 Adam 少 1/3，在大模型上速度更快
- **Sophia**（Liu et al., 2023）：用 Hessian 对角线近似作为缩放因子，比 Adam 收敛更快但计算更贵

## 代码验证

```python
import numpy as np

# 手动实现 Adam，验证偏差修正的必要性
np.random.seed(42)

def adam(grad_fn, x0, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, n_steps=200):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    for t in range(1, n_steps + 1):
        g = grad_fn(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2
        m_hat = m / (1 - beta1 ** t)   # 偏差修正
        v_hat = v / (1 - beta2 ** t)   # 偏差修正
        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
    return x

# 最小化 f(x) = (x - 2)^2，最优解 x* = 2
grad_fn = lambda x: 2 * (x - 2)
x_opt = adam(grad_fn, x0=[0.0], lr=0.1)
print(f"Adam 最优解: x = {x_opt[0]:.4f}  (真值: 2.0)")

# 偏差修正的效果：没有修正时初始更新很小
m_no_correct = 0.0
v_no_correct = 0.0
beta1, beta2 = 0.9, 0.999
g0 = grad_fn(np.array([0.0]))[0]
m_no_correct = beta1 * 0 + (1 - beta1) * g0
v_no_correct = beta2 * 0 + (1 - beta2) * g0 ** 2
step_no_correct = m_no_correct / (np.sqrt(v_no_correct) + 1e-8)

m_hat = m_no_correct / (1 - beta1 ** 1)
v_hat = v_no_correct / (1 - beta2 ** 1)
step_corrected = m_hat / (np.sqrt(v_hat) + 1e-8)

print(f"\n无偏差修正第一步: {step_no_correct:.4f}")   # 很小
print(f"有偏差修正第一步: {step_corrected:.4f}")     # 正常大小
```

## 在深度学习中的应用

AdamW 是 GPT、BERT、LLaMA、Stable Diffusion 等所有主流大模型的默认优化器。Transformer 训练中的 Warmup + Cosine Annealing 学习率调度与 AdamW 结合使用。在强化学习里（如 PPO、SAC），Adam 是策略网络和价值网络的标准优化器。图像分类（ResNet、ViT）最终仍常用 SGD+Momentum 微调以获得更好的泛化性。

下一节讲动态规划，把优化从连续参数空间推广到序列决策问题，是强化学习 Bellman 方程的数学基础。
