# 复数与 Euler 公式

!!! info "参考资料"
    **教材**
    - *Mathematics for Machine Learning* (Deisenroth et al.) — Chapter 3.9（简要）
    - Needham, *Visual Complex Analysis* — 最直观的复分析入门（可选延伸）

    **论文（应用）**
    - Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021

---

## 直觉 (Intuition)

实数只能在数轴上移动，但有些变换（特别是旋转）用实数描述很别扭。复数把平面变成了"带代数运算的二维空间"，旋转就变成了复数乘法。Euler 公式 $e^{i\theta} = \cos\theta + i\sin\theta$ 是这一思想的精华：乘以 $e^{i\theta}$ 就是旋转 $\theta$ 角。

这不只是数学技巧。Transformer 里的 RoPE 位置编码就是用复数旋转来编码位置信息；傅里叶变换的频域分析也建立在这里（下一节）。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $i$ | 虚数单位，$i^2 = -1$ |
| $z = a + bi$ | 复数（$a$ 实部，$b$ 虚部） |
| $\bar{z} = a - bi$ | $z$ 的共轭复数 |
| $|z| = \sqrt{a^2 + b^2}$ | $z$ 的模（绝对值） |
| $\arg(z)$ | $z$ 的辐角（与实轴夹角） |

---

## 复平面

复数 $z = a + bi$ 可以画在二维平面上：实部 $a$ 在横轴，虚部 $b$ 在纵轴。这个平面叫**复平面**（Argand 图）。

极坐标形式：$z = r e^{i\theta} = r(\cos\theta + i\sin\theta)$，其中 $r = |z|$ 是模，$\theta = \arg(z)$ 是辐角。

复数乘法的几何含义：

$$z_1 \cdot z_2 = r_1 r_2 \, e^{i(\theta_1 + \theta_2)}$$

**模相乘，辐角相加。** 这意味着乘以 $e^{i\theta}$（模为 1，辐角为 $\theta$）就是**旋转 $\theta$ 角，不改变长度**。

---

## Euler 公式

!!! note "Euler 公式"
    $$e^{i\theta} = \cos\theta + i\sin\theta$$

    **证明思路：** 对 $e^{i\theta}$、$\cos\theta$、$\sin\theta$ 分别展开 Taylor 级数，然后验证等式两端对应项相等：

    $$e^{i\theta} = \sum_{n=0}^\infty \frac{(i\theta)^n}{n!} = \underbrace{\sum_{k=0}^\infty \frac{(-1)^k \theta^{2k}}{(2k)!}}_{\cos\theta} + i \underbrace{\sum_{k=0}^\infty \frac{(-1)^k \theta^{2k+1}}{(2k+1)!}}_{\sin\theta}$$

**Euler 恒等式：** 取 $\theta = \pi$，得 $e^{i\pi} + 1 = 0$，联系了数学里五个最基本的常数。

---

## 复数的矩阵表示

复数乘法可以用 $2 \times 2$ 实矩阵来表示：

$$z = a + bi \quad \longleftrightarrow \quad \begin{bmatrix} a & -b \\ b & a \end{bmatrix}$$

特别地，模为 1 的复数 $e^{i\theta}$ 对应二维旋转矩阵：

$$e^{i\theta} \longleftrightarrow \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

这个对应关系说明：**复数乘法就是二维旋转**。

---

## 四元数（简介）

四元数（Quaternion）是复数在三维的推广，形如 $q = w + xi + yj + zk$，其中 $i^2 = j^2 = k^2 = ijk = -1$。

模为 1 的单位四元数可以表示三维旋转，且没有万向节死锁（Gimbal Lock）的问题。这让它在机器人学和 3D 生成领域广泛使用。

| 工具 | 表示维度 | 旋转维度 | 优势 |
|------|---------|---------|------|
| 欧拉角 | 3 | 3D | 直觉，但有 Gimbal Lock |
| 旋转矩阵 | 9 | 3D | 无奇点，但参数多、有正交约束 |
| 四元数 | 4 | 3D | 紧凑、无奇点、插值平滑 |
| 复数 | 2 | 2D | 最简洁的 2D 旋转表示 |

---

## RoPE：旋转位置编码

RoPE（Rotary Position Embedding）是目前 LLM 里最主流的位置编码方式（LLaMA、Mistral、Qwen 全部使用）。

核心思想：用复数旋转给每个位置的 Query 和 Key 向量编码位置信息，使得位置 $m$ 和 $n$ 的 Query-Key 内积只依赖于相对位置 $m-n$。

对 $d$ 维向量，将其两两配对分成 $d/2$ 对，第 $k$ 对用旋转角 $\theta_k = 10000^{-2k/d}$ 进行旋转：

$$\begin{bmatrix} q_{2k-1}' \\ q_{2k}' \end{bmatrix} = \begin{bmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{bmatrix} \begin{bmatrix} q_{2k-1} \\ q_{2k} \end{bmatrix}$$

用复数写更简洁：$q' = q \cdot e^{im\theta_k}$（将向量的每对分量看作一个复数，乘以旋转因子）。

相对位置的感知来自于：$\langle q_m', k_n' \rangle = \text{Re}(q_m \bar{k}_n \cdot e^{i(m-n)\theta_k})$，内积只与差值 $m-n$ 有关。

---

## 代码验证

```python
import numpy as np

# 验证 Euler 公式
theta = np.pi / 3   # 60°
z_euler = np.exp(1j * theta)
z_trig  = np.cos(theta) + 1j * np.sin(theta)
print(np.isclose(z_euler, z_trig))  # True

# 复数乘法 = 旋转
z = 1 + 1j          # 模 sqrt(2)，辐角 45°
rot = np.exp(1j * np.pi / 2)  # 旋转 90°
z_rotated = z * rot
print(z_rotated)     # (-1+1j)  <- 确实旋转了 90°（从 45° 到 135°）

# 复数的矩阵表示
def to_rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

theta = np.pi / 4
v = np.array([1.0, 0.0])
v_rotated_matrix = to_rotation_matrix(theta) @ v
v_rotated_complex = (v[0] + 1j * v[1]) * np.exp(1j * theta)
print(v_rotated_matrix)  # [0.707, 0.707]
print([v_rotated_complex.real, v_rotated_complex.imag])  # [0.707, 0.707]  <- 相同

# RoPE 的核心操作：对 Query 向量做旋转编码
def apply_rope(q, pos, theta=10000.0):
    """将 2D 向量 q 在位置 pos 处做 RoPE 旋转"""
    d = len(q) // 2
    q_complex = q[:d] + 1j * q[d:]  # 将向量对分成复数
    freqs = theta ** (-np.arange(d) / d)
    rotations = np.exp(1j * pos * freqs)
    q_rotated = q_complex * rotations
    return np.concatenate([q_rotated.real, q_rotated.imag])

q1 = np.random.randn(8)
q2 = q1.copy()
q1_encoded = apply_rope(q1, pos=3)
q2_encoded = apply_rope(q2, pos=5)
# 两个位置的 Query 向量，内积只依赖相对位置差 2
print(q1_encoded @ q2_encoded)
```

!!! tip "在深度学习中的应用"
    - **RoPE 位置编码**：LLaMA/Qwen/Mistral 的标准选择，支持外推到训练长度以外（相比绝对位置编码更灵活）。
    - **频域分析**：复数是傅里叶变换的数学语言，下一节会详细展开。
    - **3D 旋转**：四元数在机器人学、3D 生成（如 AnyDoor、ZeroScope）中用于表示和插值旋转。
    - **复数权重网络**：部分信号处理领域的神经网络直接在复数域上运算（Complex-Valued Neural Networks）。

!!! note "本节结论在后面的用处"
    复数旋转是**傅里叶变换**（下一节）的基础运算：DFT 本质上是用不同频率的复数旋转来分解信号。RoPE 的数学推导在 Part 3 的 Transformer 章节会展开。
