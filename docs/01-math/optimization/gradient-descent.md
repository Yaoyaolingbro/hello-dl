# 梯度下降与收敛性

!!! info "参考资料"
    **主要资料**
    - Bottou et al., "Optimization Methods for Large-Scale Machine Learning", *SIAM Review* 2018 — 深度学习优化的权威综述
    - [Deep Learning Book: Chapter 8](https://www.deeplearningbook.org/contents/optimization.html) — Ian Goodfellow et al.
    - Bubeck, *Convex Optimization: Algorithms and Complexity*, 2015 — 收敛性分析的参考

## 直觉 (Intuition)

梯度下降是最简单的优化算法：沿梯度反方向走一小步，让函数值下降。步长（学习率）太大会震荡发散，太小收敛太慢——这个权衡是训练神经网络的核心工程问题。Mini-batch SGD 用一小批样本的梯度近似真实梯度，引入噪声；这个噪声在凸问题里有害，在非凸的深度学习里却有益：它帮助跳出鞍点、探索损失面。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\alpha$ | 学习率（步长） |
| $\mathbf{g}_t = \nabla f(\mathbf{x}_t)$ | 第 $t$ 步的梯度 |
| $B$ | Mini-batch 大小 |
| $L$ | 梯度的 Lipschitz 常数 |
| $m$ | 强凸参数（若存在） |

## 梯度下降（GD）

标准梯度下降（Gradient Descent）每步更新：

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha\, \nabla f(\mathbf{x}_t)
$$

**收敛性（凸函数，$L$-Lipschitz 梯度）**：步长 $\alpha \le 1/L$ 时，

$$
f(\mathbf{x}_T) - f(\mathbf{x}^*) \le \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2\alpha T}
$$

误差以 $O(1/T)$ 衰减（次线性收敛）。

**强凸情况**：若 $f$ 还是 $m$-强凸，误差以指数速度衰减：

$$
\|\mathbf{x}_T - \mathbf{x}^*\|^2 \le \left(1 - \frac{m}{L}\right)^T \|\mathbf{x}_0 - \mathbf{x}^*\|^2
$$

收敛速度由条件数 $\kappa = L/m$ 决定：条件数越大（损失面越"细长"），收敛越慢。

## 随机梯度下降（SGD）

每步只用一个样本（或 Mini-batch）计算梯度：

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{x}_t)
$$

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha\, \mathbf{g}_t
$$

Mini-batch 梯度是真实梯度的无偏估计（$\mathbb{E}[\mathbf{g}_t] = \nabla f(\mathbf{x}_t)$），但有方差。SGD 的收敛分析需要额外假设梯度方差有界：$\mathbb{E}[\|\mathbf{g}_t - \nabla f\|^2] \le \sigma^2$。

!!! note "学习率 vs. Batch Size 的权衡"
    大 batch 降低梯度方差（$\sigma^2/B$），但泛化性能可能下降（"sharp minima" 问题）。
    Linear Scaling Rule：batch size 增大 $k$ 倍，学习率也乘以 $k$，能维持相近的训练动态（Goyal et al., 2017），但有上限。

## 动量（Momentum）

普通 SGD 每步相互独立，容易在峡谷型损失面里震荡。**动量（Momentum）**积累历史梯度方向：

$$
\mathbf{m}_{t+1} = \beta\, \mathbf{m}_t + \nabla f(\mathbf{x}_t)
$$

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \alpha\, \mathbf{m}_{t+1}
$$

$\beta \in [0, 1)$ 是动量系数（通常取 0.9），$\mathbf{m}_t$ 是梯度的指数移动平均。直觉是：在一致方向上积累速度，在震荡方向上相互抵消。

**Nesterov 加速梯度（NAG）**在预测点计算梯度，理论上有更优的收敛界（$O(1/T^2)$ vs. $O(1/T)$，凸函数情况下），实践中常优于标准动量。

## 学习率调度

固定学习率在训练初期太小（收敛慢）或太大（不稳定）都不理想。常见调度策略：

| 策略 | 公式 / 行为 | 适用场景 |
|------|------------|---------|
| Step Decay | 每 $k$ 步乘以 $\gamma < 1$ | 图像分类（ResNet 标准训练） |
| Cosine Annealing | $\alpha_t = \alpha_\min + \frac{1}{2}(\alpha_\max - \alpha_\min)(1 + \cos(\pi t/T))$ | Transformer、ViT 预训练 |
| Warmup | 前 $w$ 步从零线性增加到 $\alpha_\max$ | 大模型训练（避免早期不稳定） |
| Cyclical LR | 周期性在 $[\alpha_\min, \alpha_\max]$ 来回 | 有助于逃离鞍点 |

## 代码验证

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 最小化 f(x) = x^2（全局最优 x*=0）
f  = lambda x: x ** 2
df = lambda x: 2 * x

# GD、SGD（加噪声）、Momentum 比较
def run_optimizer(optimizer, n_steps=100):
    x = 5.0
    history = [x]
    state = {}
    for _ in range(n_steps):
        x, state = optimizer(x, state)
        history.append(x)
    return history

def gd(x, state, lr=0.1):
    return x - lr * df(x), state

def sgd_noisy(x, state, lr=0.1, sigma=0.5):
    grad = df(x) + np.random.randn() * sigma
    return x - lr * grad, state

def momentum_sgd(x, state, lr=0.1, beta=0.9, sigma=0.3):
    m = state.get('m', 0.0)
    grad = df(x) + np.random.randn() * sigma
    m = beta * m + grad
    return x - lr * m, {'m': m}

gd_hist = run_optimizer(gd)
sgd_hist = run_optimizer(sgd_noisy)
mom_hist = run_optimizer(momentum_sgd)

print(f"GD 最终值:       x={gd_hist[-1]:.4f}")  # 接近 0
print(f"SGD 最终值:      x={sgd_hist[-1]:.4f}")  # 带噪声，不完全收敛
print(f"Momentum 最终值: x={mom_hist[-1]:.4f}")  # 通常更快收敛

# Cosine annealing 学习率
T = 100
alpha_max, alpha_min = 0.1, 1e-4
t_vals = np.arange(T)
cosine_lr = alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * t_vals / T))
print(f"\nCosine LR 初始: {cosine_lr[0]:.4f}, 末尾: {cosine_lr[-1]:.6f}")
```

## 在深度学习中的应用

SGD + Momentum 是图像分类训练的标准配置（ResNet、ViT 的最终 checkpoint 通常用 SGD 微调）。Cosine Annealing + Warmup 是 Transformer 和大语言模型的标准学习率调度。梯度裁剪（Gradient Clipping）把梯度范数上限设为 $c$，防止梯度爆炸，是 RNN 和 Transformer 训练的必备组件。

下一节讲自适应优化方法（Adam 家族），它们通过维护梯度的二阶矩来自动调整每个参数的学习率，基本取代了手工调 SGD 学习率的需求。
