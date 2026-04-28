# 自编码器 (Autoencoder)

!!! info "参考资料"
    **经典论文**

    - [Reducing the Dimensionality of Data with Neural Networks](https://www.science.org/doi/10.1126/science.1127647) — Hinton & Salakhutdinov, Science 2006
    - [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, ICLR 2014（VAE 的铺垫）

    **优质讲解**

    - [CS231n: Autoencoders](https://cs231n.github.io/unsupervised/)

## 直觉 (Intuition)

自编码器解决的问题是：如何用一个低维向量来"压缩"高维数据，同时又能从这个向量还原出原始数据？输入是高维数据 $\mathbf{x}$（比如图像），输出是重建的 $\hat{\mathbf{x}}$，中间产物是低维隐向量 $\mathbf{z}$。核心思路是用编码器把 $\mathbf{x}$ 压成 $\mathbf{z}$，再用解码器把 $\mathbf{z}$ 还原成 $\hat{\mathbf{x}}$，用重建误差训练整个网络。自编码器本身不是生成模型，但它学到的隐空间是 VAE 和 LDM（Latent Diffusion Model）的基础。

## 问题背景

监督学习需要大量标注数据，但标注成本高。自编码器提供了一种无监督的表示学习方式：让网络自己找到数据中的压缩表示，不需要任何标签。

同时，自编码器也是降维的非线性版本。PCA 找到的是线性子空间，自编码器（用非线性激活函数）可以找到弯曲的流形结构，在图像、音频、分子结构等数据上效果好得多。

## 基本结构

自编码器由两部分组成：

编码器将输入映射到隐空间：

$$\mathbf{z} = f_\phi(\mathbf{x})$$

解码器将隐向量还原为输入：

$$\hat{\mathbf{x}} = g_\theta(\mathbf{z})$$

训练目标是最小化重建误差。对连续数据常用均方误差：

$$\mathcal{L} = \|\mathbf{x} - g_\theta(f_\phi(\mathbf{x}))\|^2$$

对图像有时也用二元交叉熵（把像素值归一化到 $[0,1]$，当作独立的 Bernoulli 变量）。

!!! note "直觉小结"
    自编码器的瓶颈层（bottleneck）迫使网络只保留最重要的信息，丢弃冗余。瓶颈维度越小，压缩越激进，重建质量越低——这是信息论里率失真权衡的直接体现。

## 几种常见变体

**稀疏自编码器（Sparse AE）**：在隐向量上加 L1 正则，迫使任意时刻只有少数神经元激活。这和神经科学里大脑的稀疏编码有类比关系，学到的特征往往更具可解释性。

**去噪自编码器（Denoising AE，DAE）**：输入是加了噪声的 $\tilde{\mathbf{x}}$，目标是还原干净的 $\mathbf{x}$。这个任务比直接重建更难，迫使网络学到更鲁棒的特征。DDPM 里的去噪网络从思路上和 DAE 一脉相承。

**变分自编码器（VAE）**：下一节详细讲。简单说，VAE 不让 $\mathbf{z}$ 是一个确定的点，而是一个分布，使得隐空间连续可插值，从而支持生成新样本。

## 代码

最简单的全连接自编码器，验证瓶颈维度对重建质量的影响：

```python
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()  # 像素值归一化到 [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

model = Autoencoder(latent_dim=32)
x = torch.randn(16, 784)
x_recon, z = model(x)
print(z.shape)      # torch.Size([16, 32])，压缩了 24 倍
print(x_recon.shape)  # torch.Size([16, 784])
```

## 局限与后续工作

自编码器的隐空间没有结构约束：两个相邻的 $\mathbf{z}$ 解码出来的结果可能差异巨大，也可能某些区域根本没有被训练覆盖到。直接从隐空间随机采一个点解码，往往得到的是噪声。

这个问题推动了 VAE 的出现——通过给隐空间施加先验分布约束（$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$），使得整个隐空间都是有意义的，可以随机采样生成新样本。

下一节讲 VAE，它在自编码器的结构上加一层概率解释，把编码过程从"压缩到一个点"变成"压缩到一个分布"。
