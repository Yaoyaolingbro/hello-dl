# 状态估计与卡尔曼滤波

!!! info "参考资料"
    **主要资料**
    - Thrun et al., *Probabilistic Robotics* — Chapter 2–3（贝叶斯滤波和卡尔曼滤波的经典讲解）
    - Kalman, "A New Approach to Linear Filtering and Prediction Problems", 1960 — 原始论文

    **工程资料**
    - [FilterPy](https://github.com/rlabbe/filterpy) — Python 卡尔曼滤波库
    - [Roger Labbe, Kalman and Bayesian Filters in Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) — 免费 Jupyter 书

## 直觉 (Intuition)

状态估计解决的问题是：在传感器有噪声、运动有误差的情况下，如何实时推断系统的真实状态。贝叶斯滤波是通用框架，卡尔曼滤波是它在线性高斯假设下的解析解。输入是带噪声的观测序列，输出是对真实状态的概率估计（均值和不确定性）。SLAM、无人驾驶的传感器融合、机器人本体感知，都在实时运行这套算法。

## 主要符号

| 符号 | 含义 |
|------|------|
| $\mathbf{x}_t$ | 时刻 $t$ 的真实状态（隐变量） |
| $\mathbf{z}_t$ | 时刻 $t$ 的观测（带噪声的测量） |
| $p(\mathbf{x}_t \mid \mathbf{z}_{1:t})$ | 时刻 $t$ 的后验置信（belief） |
| $\mathbf{F}$ | 状态转移矩阵 |
| $\mathbf{H}$ | 观测矩阵 |
| $\mathbf{Q}, \mathbf{R}$ | 过程噪声和观测噪声的协方差矩阵 |
| $\mathbf{K}$ | 卡尔曼增益 |

## 状态空间模型

状态估计的标准数学框架是**隐马尔科夫模型（HMM）**或**状态空间模型**：

**状态转移方程**（系统动力学，有噪声）：

$$
\mathbf{x}_t = \mathbf{F}\mathbf{x}_{t-1} + \mathbf{w}_t, \quad \mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q})
$$

**观测方程**（传感器，有噪声）：

$$
\mathbf{z}_t = \mathbf{H}\mathbf{x}_t + \mathbf{v}_t, \quad \mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R})
$$

$\mathbf{w}_t$ 是过程噪声（运动本身的误差），$\mathbf{v}_t$ 是观测噪声（传感器误差）。

## 贝叶斯滤波

贝叶斯滤波是状态估计的通用框架，交替执行两步：

**预测步（Predict）**：利用运动模型，把 $t-1$ 的后验传播到 $t$：

$$
p(\mathbf{x}_t \mid \mathbf{z}_{1:t-1})
=
\int p(\mathbf{x}_t \mid \mathbf{x}_{t-1})\, p(\mathbf{x}_{t-1} \mid \mathbf{z}_{1:t-1})\, d\mathbf{x}_{t-1}
$$

直觉：根据"系统会怎么运动"，预测下一时刻状态可能在哪。预测后不确定性增大（因为运动有噪声）。

**更新步（Update）**：利用新观测 $\mathbf{z}_t$，用贝叶斯公式修正预测：

$$
p(\mathbf{x}_t \mid \mathbf{z}_{1:t})
\propto
p(\mathbf{z}_t \mid \mathbf{x}_t)\, p(\mathbf{x}_t \mid \mathbf{z}_{1:t-1})
$$

直觉：观测告诉我们系统实际在哪，和预测不符的地方就需要修正。更新后不确定性减小（观测提供了信息）。

## 卡尔曼滤波

在**线性高斯**假设下（状态转移和观测都是线性的，噪声都是高斯的），贝叶斯滤波有解析解，就是**卡尔曼滤波**。

后验始终是高斯分布：$p(\mathbf{x}_t \mid \mathbf{z}_{1:t}) = \mathcal{N}(\hat{\mathbf{x}}_t, \mathbf{P}_t)$，只需维护均值 $\hat{\mathbf{x}}_t$ 和协方差 $\mathbf{P}_t$。

**预测步**：

$$
\hat{\mathbf{x}}_{t|t-1} = \mathbf{F}\hat{\mathbf{x}}_{t-1}
$$

$$
\mathbf{P}_{t|t-1} = \mathbf{F}\mathbf{P}_{t-1}\mathbf{F}^\top + \mathbf{Q}
$$

**卡尔曼增益**（决定观测和预测各占多少权重）：

$$
\mathbf{K}_t
=
\mathbf{P}_{t|t-1}\mathbf{H}^\top
\left(\mathbf{H}\mathbf{P}_{t|t-1}\mathbf{H}^\top + \mathbf{R}\right)^{-1}
$$

**更新步**：

$$
\hat{\mathbf{x}}_t = \hat{\mathbf{x}}_{t|t-1} + \mathbf{K}_t(\mathbf{z}_t - \mathbf{H}\hat{\mathbf{x}}_{t|t-1})
$$

$$
\mathbf{P}_t = (\mathbf{I} - \mathbf{K}_t\mathbf{H})\mathbf{P}_{t|t-1}
$$

!!! note "卡尔曼增益的直觉"
    $\mathbf{K}_t$ 在 0 到 1 之间权衡预测和观测：
    - 若观测噪声 $\mathbf{R}$ 很大（传感器不可靠），$\mathbf{K}_t \to 0$，更信任预测
    - 若预测不确定性 $\mathbf{P}_{t|t-1}$ 很大（运动模型不准），$\mathbf{K}_t \to 1$，更信任观测

## 非线性扩展

实际系统大多是非线性的（如相机投影、旋转运动）。

- **扩展卡尔曼滤波（EKF）**：在当前估计点对非线性函数做一阶泰勒展开（Jacobian 线性化），然后用标准卡尔曼滤波框架。
- **无迹卡尔曼滤波（UKF）**：用 sigma 点集近似分布，处理更强的非线性，精度高于 EKF。
- **粒子滤波**：用一组带权重的粒子（样本）表示后验，适用于高度非线性或非高斯情况，计算代价较高。

SLAM（同步定位与建图）通常使用 EKF-SLAM 或基于图优化的方法（如 g2o/iSAM），在线估计机器人位姿和地图。

## 代码验证

```python
import numpy as np

# 一维匀速运动追踪
# 状态 x = [位置, 速度]，每 dt=0.1s 更新一次
dt = 0.1
F = np.array([[1, dt], [0, 1]])    # 状态转移
H = np.array([[1, 0]])             # 只观测位置
Q = np.eye(2) * 0.01               # 过程噪声（小）
R = np.array([[1.0]])              # 观测噪声（较大）

# 初始化
x_est = np.array([0.0, 1.0])      # 初始估计：位置 0，速度 1m/s
P = np.eye(2) * 1.0               # 初始不确定性

# 模拟真实轨迹和带噪声的观测
np.random.seed(42)
n_steps = 20
x_true = np.array([0.0, 1.0])
observations, estimates = [], []

for _ in range(n_steps):
    # 真实状态更新（有过程噪声）
    x_true = F @ x_true + np.random.multivariate_normal([0,0], Q)

    # 带噪声的观测
    z = H @ x_true + np.random.normal(0, np.sqrt(R[0,0]))
    observations.append(float(z))

    # 预测步
    x_pred = F @ x_est
    P_pred = F @ P @ F.T + Q

    # 更新步（卡尔曼增益）
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ (z - H @ x_pred)
    P = (np.eye(2) - K @ H) @ P_pred
    estimates.append(float(x_est[0]))

# 卡尔曼滤波估计比原始观测更平滑
print("观测:", [f"{o:.2f}" for o in observations[:5]])
print("估计:", [f"{e:.2f}" for e in estimates[:5]])
```

## 在深度学习中的应用

ViT + 卡尔曼滤波的混合架构在目标跟踪（如 ByteTrack）中广泛使用：深度特征提取 + 卡尔曼运动预测。SLAM 系统（ORB-SLAM、VINS-Mono）用 EKF 或因子图实时维护机器人的位姿后验。传感器融合（激光雷达 + 相机 + IMU）在自动驾驶中是核心模块，本质上是高维卡尔曼滤波。

下一节讲蒙特卡洛与重要性采样。这是当贝叶斯后验无法解析计算时的主要近似工具，也是扩散模型采样过程的数学基础。
