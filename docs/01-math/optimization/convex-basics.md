# 凸集与凸函数

!!! info "参考资料"
    **主要资料**
    - Boyd & Vandenberghe, *Convex Optimization* — 免费 PDF，第 2–3 章是标准参考
    - [Deep Learning Book: Chapter 4](https://www.deeplearningbook.org/contents/numerical.html) — Ian Goodfellow et al.

    **工具文档**
    - [CVXPY](https://www.cvxpy.org/) — Python 凸优化建模库

## 直觉 (Intuition)

凸性是优化问题的"好性质"：在凸函数上，局部最小值就是全局最小值，不存在"卡在山谷里出不去"的问题。深度学习的损失函数几乎从不是凸的，但理解凸优化能告诉我们什么时候优化有保证，以及神经网络训练为什么在实践中还能运作。Jensen 不等式是凸性最有用的一条推论，它直接出现在 ELBO 推导、信息不等式和 KL 散度下界的论证里。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathcal{C}$ | 凸集 |
| $f: \mathbb{R}^n \to \mathbb{R}$ | 目标函数 |
| $\nabla f(\mathbf{x})$ | 梯度（列向量） |
| $\nabla^2 f(\mathbf{x})$ | Hessian 矩阵 |
| $\text{epi}(f)$ | 函数 $f$ 的上图（epigraph） |

## 凸集

集合 $\mathcal{C} \subseteq \mathbb{R}^n$ 是**凸集（convex set）**，若对任意 $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ 和 $\theta \in [0,1]$，连接它们的线段都在集合内：

$$
\theta\, \mathbf{x} + (1-\theta)\, \mathbf{y} \in \mathcal{C}
$$

常见凸集：超平面 $\{\mathbf{x} \mid \mathbf{a}^\top \mathbf{x} = b\}$、半空间 $\{\mathbf{x} \mid \mathbf{a}^\top \mathbf{x} \le b\}$、球、多面体。凸集的交集仍是凸集，这是约束优化里构造可行域的基础。

## 凸函数

函数 $f$ 是**凸函数（convex function）**，若其定义域是凸集，且对任意 $\mathbf{x}, \mathbf{y}$ 和 $\theta \in [0,1]$：

$$
f(\theta\, \mathbf{x} + (1-\theta)\, \mathbf{y})
\le
\theta\, f(\mathbf{x}) + (1-\theta)\, f(\mathbf{y})
$$

等价地：函数曲线上任意两点之间的弦，都在曲线的上方或与曲线重合。

等价的判断准则：

**一阶条件**（$f$ 可微时）：

$$
f(\mathbf{y}) \ge f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x})
$$

即切平面始终在曲线下方——对所有点都低估函数值。

**二阶条件**（$f$ 二阶可微时）：$\nabla^2 f(\mathbf{x}) \succeq 0$（Hessian 半正定）。Hessian 正定（$\nabla^2 f \succ 0$）则 $f$ 是**严格凸函数**。

!!! note "局部最小 = 全局最小"
    凸函数的驻点（$\nabla f(\mathbf{x}^*) = \mathbf{0}$）一定是全局最小值点。这是凸优化"可靠"的根本原因：找到一个驻点就找到了解，不存在多个不同的局部最小值。

## Jensen 不等式

若 $f$ 是凸函数，则对任意随机变量 $X$：

$$
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]
$$

"先平均再计算"不超过"先计算再平均"。对严格凸函数，等号成立当且仅当 $X$ 几乎处处为常数。

对数函数 $-\log$ 是凸函数，Jensen 不等式给出 $\log(\mathbb{E}[X]) \ge \mathbb{E}[\log X]$。变分推断的 ELBO 推导正是对这一不等式取等号条件的分析，VAE 的证据下界直接来自这里（见概率论章节的变分推断节）。

## 强凸性

$f$ 是 $m$-**强凸（strongly convex）**函数，若存在 $m > 0$ 使得 $\nabla^2 f(\mathbf{x}) \succeq m\mathbf{I}$。强凸函数有**唯一最小值**，梯度下降在强凸函数上有线性收敛保证：误差以 $(1 - m/L)^k$ 指数衰减（其中 $L$ 是梯度的 Lipschitz 常数）。

## 常见凸函数

| 函数 | 凸性 | 原因 |
|------|------|------|
| $x^2,\; e^x,\; -\log x\;(x>0)$ | 凸 | 二阶导数 $\ge 0$ |
| $\|\mathbf{x}\|_p$（$p \ge 1$） | 凸 | 三角不等式 |
| 逐点上确界 $\max_i f_i(\mathbf{x})$（各 $f_i$ 凸） | 凸 | 上图交集仍是凸集 |
| 线性函数 $\mathbf{a}^\top \mathbf{x}$ | 既凸又凹 | Hessian $= 0$ |

## 代码验证

```python
import numpy as np

# 验证 Jensen 不等式：log(E[X]) >= E[log X]
np.random.seed(42)
X = np.random.exponential(scale=2.0, size=100000)

lhs = np.log(X.mean())        # log(E[X])
rhs = np.log(X).mean()        # E[log X]
print(f"log(E[X]) = {lhs:.4f}")   # ≈ 0.693
print(f"E[log X]  = {rhs:.4f}")   # ≈ 0.423（更小）
print(f"Jensen 成立: {lhs >= rhs}")  # True

# 验证凸性定义：f(θx + (1-θ)y) <= θf(x) + (1-θ)f(y)
f = lambda t: t ** 2   # 严格凸
x, y, theta = 1.0, 3.0, 0.4
lhs_c = f(theta * x + (1 - theta) * y)
rhs_c = theta * f(x) + (1 - theta) * f(y)
print(f"\nf(θx+(1-θ)y) = {lhs_c:.2f} <= θf(x)+(1-θ)f(y) = {rhs_c:.2f}: {lhs_c <= rhs_c}")

# 验证二阶条件：exp(x) 的二阶导数处处 > 0
x_vals = np.linspace(-3, 3, 100)
hessian_exp = np.exp(x_vals)   # d²/dx² exp(x) = exp(x) > 0
print(f"\nexp(x) Hessian > 0: {(hessian_exp > 0).all()}")  # True
```

## 在深度学习中的应用

虽然神经网络损失函数非凸，凸分析仍然关键：交叉熵损失对 logits 是凸的（这保证了 logistic regression 的全局可解性），L1/L2 正则化项是凸的，SVM 的核心是凸二次规划。对比学习损失中的 InfoNCE 可以通过凸对偶视角分析。Jensen 不等式直接出现在变分推断和扩散模型目标函数的推导里。

下一节讲一阶最优性条件，梯度和驻点的形式化定义是所有优化算法的出发点。
