# Flow Matching

!!! info "参考资料"
    **主要论文**

    - [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al., ICLR 2023
    - [Improving and Generalizing Flow-Matching](https://arxiv.org/abs/2302.00482) — Albergo & Vanden-Eijnden, 2023（Stochastic Interpolants）
    - [Stable Diffusion 3 / FLUX](https://arxiv.org/abs/2403.03206) — Esser et al., 2024（Flow Matching 的大规模应用）

    **优质讲解**

    - [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) — Lipman et al., 2024（官方教程）

!!! note "前置依赖"
    本节用到 [ODE 与向量场](../../01-math/optimization/odes-vector-fields.md) 里的 ODE 基本概念，建议先了解"向量场定义流"的直觉。

## 直觉 (Intuition)

扩散模型从噪声 $\mathbf{x}_T$ 走到数据 $\mathbf{x}_0$ 的路径是曲折的——因为每步只能走一小步，1000 步才能到达。Flow Matching 的想法是：能不能走直线？输入是噪声 $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$，输出是数据样本 $\mathbf{x}_1 \sim p_{\text{data}}$。核心思路是学一个速度场（velocity field）$v_\theta(\mathbf{x}, t)$，让沿着 ODE 从 $t=0$ 走到 $t=1$ 的轨迹恰好把噪声分布变换成数据分布。轨迹越直，需要的推理步数越少。

## 问题背景

DDPM 的采样过程本质上是在解一个随机微分方程（SDE）：

$$d\mathbf{x} = f(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{W}$$

Song et al. (2021) 证明，对任意 SDE 存在一个等价的 ODE（概率流 ODE），两者共享同样的边缘分布 $p_t(\mathbf{x})$：

$$\frac{d\mathbf{x}}{dt} = f(\mathbf{x}, t) - \frac{1}{2} g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})$$

但这个 ODE 的向量场很复杂，轨迹弯曲。Flow Matching 直接设计一个简单的向量场，目标就是把 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 变换到 $p_{\text{data}}$，而且轨迹尽量直。

## 方法推导

**第一步**：定义概率路径（Probability Path）。我们希望构造一族边缘分布 $p_t(\mathbf{x})$，满足 $p_0 = \mathcal{N}(\mathbf{0}, \mathbf{I})$，$p_1 = p_{\text{data}}$，中间连续过渡。对于每个数据点 $\mathbf{x}_1 \sim p_{\text{data}}$，定义一条从 $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 到 $\mathbf{x}_1$ 的路径：

$$\mathbf{x}_t = (1-t)\, \mathbf{x}_0 + t\, \mathbf{x}_1, \quad t \in [0, 1]$$

这就是最简单的线性插值（linear interpolation），轨迹是直线。

**第二步**：定义条件向量场。沿着这条直线，速度（时间导数）是常数：

$$u_t(\mathbf{x} | \mathbf{x}_1) = \frac{d\mathbf{x}_t}{dt} = \mathbf{x}_1 - \mathbf{x}_0$$

这是给定 $\mathbf{x}_1$（目标数据点）的条件向量场（conditional vector field）。

**第三步**：边缘向量场（Marginal Vector Field）。我们想要的是无条件向量场 $u_t(\mathbf{x})$——给定当前位置 $\mathbf{x}$，往哪个方向走。对所有可能的 $\mathbf{x}_1$ 取平均：

$$u_t(\mathbf{x}) = \mathbb{E}_{p(\mathbf{x}_1 | \mathbf{x}_t)}[\mathbf{x}_1 - \mathbf{x}_0]$$

直接学这个边缘向量场很难，但 Flow Matching 的关键发现是：**条件向量场的损失和边缘向量场的损失具有相同的梯度**。所以可以用条件损失来训练：

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_1} \left[ \| v_\theta(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \|^2 \right]$$

其中 $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$。

!!! note "直觉小结"
    训练目标简洁到令人惊讶：给定当前带噪版本 $\mathbf{x}_t$ 和时间步 $t$，让网络预测"从噪声到数据的方向"$\mathbf{x}_1 - \mathbf{x}_0$。线性路径让这个方向是常数，网络的学习任务比 DDPM 的去噪更直接。

## 关键设计决策

**线性路径 vs 余弦路径**

DDPM 的噪声调度（cosine schedule）对应弯曲的轨迹。Flow Matching 的线性插值使得轨迹是直线，采样时少量步骤（20-50 步）就能得到高质量图像，而 DDPM 需要 1000 步。

Stable Diffusion 3 和 FLUX 选用了 Flow Matching，原因之一正是直线轨迹大幅缩短了推理时间。

**为什么不直接用 $v_\theta$ 预测 $\mathbf{x}_0$？**

预测 $\mathbf{x}_0$（clean sample prediction）在 $t$ 接近 0 时方差很大，网络难以拟合。预测速度向量 $\mathbf{x}_1 - \mathbf{x}_0$ 的量级在整个训练过程中比较稳定。实践中预测速度的效果略好于预测 $\mathbf{x}_0$，但差别不大。

## 代码

Flow Matching 训练和推理（ODE 求解用欧拉法）：

```python
import torch

def flow_matching_loss(model, x1):
    """训练：随机插值，预测速度向量"""
    t = torch.rand(x1.shape[0], 1, 1, 1)        # 均匀采样时间步
    x0 = torch.randn_like(x1)                    # 从标准高斯采样噪声
    xt = (1 - t) * x0 + t * x1                   # 线性插值
    target = x1 - x0                             # 目标速度（常数）

    v_pred = model(xt, t.squeeze())
    return ((v_pred - target) ** 2).mean()

@torch.no_grad()
def flow_matching_sample(model, shape, steps=50):
    """推理：欧拉法积分 ODE"""
    x = torch.randn(shape)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i / steps)
        v = model(x, t)
        x = x + v * dt                           # 欧拉一步，沿速度场前进
    return x
```

## 局限与后续工作

Flow Matching 假设噪声和数据之间的最优配对是独立随机的（$\mathbf{x}_0$ 和 $\mathbf{x}_1$ 独立采样）。Conditional Flow Matching 的改进版本——Optimal Transport CFM（OT-CFM）——用最优传输（Optimal Transport）来找最优配对，使得轨迹更直，方差更小，需要更少步数。

Flow Matching 目前主要用于连续数据（图像、视频、音频、蛋白质结构）。离散数据（文本）上的对应方法是 Discrete Flow Matching，是当前的研究热点之一。

下一节讲条件生成：如何让生成模型受文本、类别标签等条件控制，以及 Classifier-Free Guidance（CFG）为什么有效。
