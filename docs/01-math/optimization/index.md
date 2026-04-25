# 优化理论

训练神经网络本质上是求解一个高维优化问题。这一章讲清楚梯度下降为什么能工作，以及 Adam 为什么比 SGD 好。

## 你将学到

| 小节 | 核心内容 | 前置依赖 |
|------|----------|----------|
| [凸集与凸函数](convex-basics.md) | 凸性定义、Jensen 不等式 | 线性代数 |
| [一阶最优性](first-order.md) | 梯度、方向导数、驻点条件 | 矩阵求导 |
| [二阶条件与曲率](second-order.md) | Hessian、正定性、鞍点 | 一阶最优性 |
| [梯度下降与收敛性](gradient-descent.md) | GD/SGD、学习率、收敛分析 | 一阶最优性 |
| [自适应优化方法](adaptive-methods.md) | Momentum、AdaGrad、RMSProp、Adam | 梯度下降 |
| [拉格朗日乘子与 KKT](lagrangian.md) | 约束优化、KKT 条件 | 凸函数 |
