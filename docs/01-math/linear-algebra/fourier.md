# 傅里叶变换

!!! info "参考资料"
    **教程**
    - [An Interactive Introduction to Fourier Transforms](https://www.jezzamon.com/fourier/) — 可交互可视化，强烈推荐先看这个
    - 3Blue1Brown, "But what is the Fourier Transform?" （YouTube）

    **论文（应用）**
    - Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021
    - Su et al., "RoFormer", 2021（RoPE 与旋转频率）

---

## 直觉 (Intuition)

任何周期信号都可以分解为一系列正弦波的叠加，每个正弦波对应一个频率。这就是傅里叶变换在干的事：从时间域（信号随时间的变化）切换到频率域（信号中各频率成分的强度）。

在深度学习里，频域视角有三种用途：音频处理用 STFT 提取频率特征，Transformer 的位置编码（RoPE）用不同频率的旋转来区分位置，神经算子（FNO）直接在频域里学习 PDE 的解。

---

## 符号约定

| 符号 | 含义 |
|------|------|
| $x[n]$ | 离散信号，$n = 0, 1, \ldots, N-1$ |
| $X[k]$ | $x[n]$ 的 DFT，第 $k$ 个频率分量 |
| $\omega_N = e^{-2\pi i / N}$ | $N$ 次单位根 |
| $f_s$ | 采样率（每秒采样次数，单位 Hz） |

---

## 连续傅里叶变换

连续信号 $f(t)$ 的傅里叶变换将其分解为不同频率 $\xi$ 的复指数 $e^{2\pi i \xi t}$ 的叠加：

$$\hat{f}(\xi) = \int_{-\infty}^{\infty} f(t) \, e^{-2\pi i \xi t} \, dt$$

$\hat{f}(\xi)$ 是一个复数，其模 $|\hat{f}(\xi)|$ 表示频率 $\xi$ 的"强度"，辐角表示该频率的相位。

逆变换：$f(t) = \int_{-\infty}^{\infty} \hat{f}(\xi) \, e^{2\pi i \xi t} \, d\xi$（用各频率分量重建原信号）

---

## 离散傅里叶变换（DFT）

计算机处理的是离散信号。$N$ 点离散傅里叶变换定义为：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-2\pi i k n / N} = \sum_{n=0}^{N-1} x[n] \cdot \omega_N^{kn}$$

$X[k]$ 是信号中频率为 $k/N$ 个周期/采样点的正弦波分量的复数振幅。

DFT 可以写成矩阵乘法：$\mathbf{X} = \mathbf{F} \mathbf{x}$，其中 $F_{kn} = \omega_N^{kn}$，$\mathbf{F}$ 是 $N \times N$ 的 DFT 矩阵。

直接计算需要 $O(N^2)$ 次运算，而 FFT（快速傅里叶变换）通过分治把复杂度降到 $O(N \log N)$。

---

## 短时傅里叶变换（STFT）

纯傅里叶变换告诉我们信号"整体"含有哪些频率，但不知道每个频率在**哪个时刻**出现。语音信号里，不同时刻的音素不同，我们需要时频联合分析。

STFT 把信号分成若干短时窗（如 25ms），对每个窗做 DFT：

$$\text{STFT}\{x\}(m, k) = \sum_{n=0}^{L-1} x[n + mH] \cdot w[n] \cdot e^{-2\pi i k n / L}$$

其中 $w[n]$ 是窗函数（Hann/Hamming），$H$ 是帧移步长，$L$ 是窗长。

结果是一个二维时频表示（**谱图/Spectrogram**），横轴是时间帧，纵轴是频率，值是复数振幅（常用其模的平方，即**功率谱**）。

!!! tip "在深度学习中的应用"
    谱图是音频模型的标准输入：Whisper（OpenAI）、VALL-E（微软）、AudioLM（Google）都先把音频转为 mel 谱图，再喂给 Transformer。Mel 谱图用的是对数频率轴（mel 尺度），更接近人耳的感知方式。

---

## 卷积定理

!!! note "卷积定理"
    时域上的卷积等于频域上的逐元素乘法：

    $$\mathcal{F}\{f * g\} = \hat{f} \cdot \hat{g}$$

    其中 $(f * g)(t) = \int f(\tau) g(t - \tau) d\tau$ 是卷积操作。

这意味着：与其在时域做 $O(N^2)$ 的卷积，不如先做 FFT（$O(N \log N)$）、逐元素相乘（$O(N)$）、再做逆 FFT（$O(N \log N)$），总复杂度 $O(N \log N)$。

卷积定理是 CNN 的理论基础：CNN 的卷积核在频域看是一个**频率过滤器**，低通滤波器保留低频（平滑结构），高通滤波器保留高频（边缘细节）。

---

## 神经算子（FNO）中的傅里叶层

FNO（Fourier Neural Operator）用傅里叶变换来学习函数到函数的映射（如 PDE 的求解算子）：

1. 对输入做 FFT，得到频域表示
2. 在频域截取低频部分（丢弃高频噪声），做线性变换学习频域参数
3. 做逆 FFT，加上残差

这让 FNO 在不同分辨率之间具有**零样本泛化能力**——因为在频域学到的滤波器与网格分辨率无关。

---

## 代码验证

```python
import numpy as np
import matplotlib

# 合成信号：3Hz + 7Hz 正弦波叠加
fs = 100         # 采样率 100Hz
t = np.arange(0, 1, 1/fs)   # 1 秒，100 个采样点
signal = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t)

# DFT
X = np.fft.rfft(signal)         # rfft 利用实信号的对称性，只返回一半频率
freqs = np.fft.rfftfreq(len(t), 1/fs)  # 对应的频率轴

# 找峰值频率
magnitudes = np.abs(X)
peak_indices = np.argsort(magnitudes)[-2:]
print("峰值频率:", freqs[peak_indices])  # [3. 7.]  <- 正确找到了两个频率

# 验证卷积定理：时域卷积 == 频域乘法
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([0.5, 1.0, 0.5, 0.0])

conv_time = np.convolve(a, b, mode='full')
conv_freq = np.fft.irfft(np.fft.rfft(a, n=8) * np.fft.rfft(b, n=8))
print(np.allclose(conv_time, conv_freq[:len(conv_time)]))  # True
```

```python
# 音频的 mel 谱图（工程常用）
# pip install librosa
import librosa
import numpy as np

# 生成一段合成音频（实际使用时换成真实音频路径）
sr = 22050  # 采样率
duration = 1.0
t = np.linspace(0, duration, int(sr * duration))
audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz 正弦波（音符 A）

# 提取 mel 谱图
mel_spec = librosa.feature.melspectrogram(
    y=audio, sr=sr,
    n_fft=1024,     # FFT 窗长
    hop_length=256,  # 帧移步长（每步移动的采样点数）
    n_mels=80       # mel 频率通道数（Whisper 用 80 通道）
)
log_mel = librosa.power_to_db(mel_spec)  # 取对数，更接近人耳感知
print(log_mel.shape)  # (80, 87)：80 个 mel 通道，87 个时间帧
```

!!! tip "在深度学习中的应用"
    - **语音识别**：Whisper 用 80 通道 mel 谱图作为输入，窗长 25ms，帧移 10ms。
    - **音乐生成**：MusicLM、AudioLDM 在 mel 谱图空间里做扩散，再用 vocoder 转回波形。
    - **FNO**：直接在频域里学习 PDE 的解，在 Navier-Stokes 方程求解上比传统数值方法快 1000 倍。

!!! note "本节结论在后面的用处"
    傅里叶变换是 Part 4 **音频理解**章节（Whisper、VALL-E、AudioLDM）的数学基础，也是 Part 4 **AI4Science**章节（FNO、物理信息神经网络）的重要工具。RoPE 的频率衰减设计 $\theta_k = 10000^{-2k/d}$ 与傅里叶基函数的频率设计同源。
