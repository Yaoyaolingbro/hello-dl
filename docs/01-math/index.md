# Part 1 · 数学基础

深度学习的本质是在高维空间里做优化。读懂论文里的公式，需要四块数学：线性代数（代数运算与分解）、几何基础（空间变换与曲线）、概率论（不确定性建模）、优化理论（如何训练）。

本章共 **36 个小节**，顺序经过仔细设计——每节只依赖前面已学的内容。

## 四块数学的定位

```mermaid
graph LR
    LA[线性代数\n11节] --> GEO[几何基础\n4节]
    LA --> OPT[优化理论\n11节]
    LA --> P[概率论\n10节]
    GEO --> OPT
    P --> OPT
```

## 章节速览

| 章节 | 小节数 | 核心内容 | 主要引用章节 |
|------|--------|----------|-------------|
| [线性代数](linear-algebra/index.md) | 11 | 向量 → 度量 → 矩阵 → 分解 → 复数 → 傅里叶 | 全部 |
| [几何基础](geometry/index.md) | 4 | 刚体运动 → 空间变换 → Lie 群 → 曲线设计 | 3DV、机器人、等变网络 |
| [概率论](probability/index.md) | 10 | 分布 → 推断 → 信息论 → 随机过程 → 状态估计 → ELBO | 生成模型、RL、SLAM |
| [优化理论](optimization/index.md) | 11 | 梯度 → 收敛 → 动态规划 → 约束 → ODE → PDE | 训练、RL、机器人、AI4Science |

## 全局符号约定

| 符号 | 含义 |
|------|------|
| $\mathbf{x}, \mathbf{y}, \mathbf{z}$ | 向量（粗体小写） |
| $\mathbf{W}, \mathbf{A}, \mathbf{H}$ | 矩阵（粗体大写） |
| $\mathcal{L}$ | 损失函数 |
| $\theta, \phi$ | 模型参数 |
| $p(\cdot), q(\cdot)$ | 概率分布 |
| $\mathbb{E}_{p}[\cdot]$ | 在分布 $p$ 下的期望 |
| $\nabla_\theta \mathcal{L}$ | 损失对参数的梯度 |
| $\|\mathbf{x}\|_2$ | L2 范数 |
| $\|\mathbf{A}\|_F$ | Frobenius 范数 |
| $\mathbb{R}^{m \times n}$ | $m \times n$ 实数矩阵空间 |
| $\mathbf{I}_n$ | $n \times n$ 单位矩阵 |
| $\mathbf{A}^\top$ | 矩阵转置 |
| $\mathbf{A}^{-1}$ | 矩阵逆 |
| $\det(\mathbf{A})$ | 行列式 |
| $\text{tr}(\mathbf{A})$ | 矩阵迹 |
| $\text{KL}(p \| q)$ | KL 散度 |
| $\mathcal{N}(\mu, \sigma^2)$ | 高斯分布 |
