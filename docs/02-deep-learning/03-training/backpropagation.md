---
level: intermediate
roles:
  - core
  - interview
prerequisites:
  - 计算图与自动微分
  - 损失函数
estimated_time: 45min
status: complete
---

# 反向传播

!!! info "参考资料"
    - [Deep Learning](https://www.deeplearningbook.org/) — 第 6.5 节
    - [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — PyTorch Documentation

## 直觉 (Intuition)

前向传播回答“参数现在给出什么预测”，反向传播回答“每个参数动一点，损失会怎样变”。如果把复合函数直接展开再求导，相同的中间导数会被反复计算。反向模式自动微分从标量损失出发，把每个节点收到的梯度只汇总一次，再乘局部导数送往上游；神经网络的反向传播就是这套过程在层状计算图上的名字。

先约定记号：对任意中间量 $v$，写

$$
\bar v=\frac{\partial L}{\partial v}.
$$

$\bar v$ 常叫伴随量或上游梯度。它说的是损失对 $v$ 的敏感度，不是 $v$ 所在节点自己的局部导数。

## 一个两层标量网络

用一个能手算完的网络看清全过程：

$$
a=w_1x+b_1,\qquad h=\operatorname{ReLU}(a),
$$

$$
s=w_2h+b_2,\qquad L=\frac12(s-y)^2.
$$

取 $x=2,w_1=3,b_1=1,w_2=4,b_2=-1,y=30$。前向计算得到 $a=7,h=7,s=27,L=4.5$。因为 $a>0$，本次执行路径上 $\operatorname{ReLU}'(a)=1$。

下面的图与这次计算一一对应。点“下一步”，先看损失把梯度交给 $s$，再沿实际执行路径逐段返回；分叉处会同时产生多份贡献。

<figure class="lesson-visual" data-lesson-visual data-interval="1900">
  <div data-lesson-stage role="img" aria-label="两层标量网络的梯度从损失累计返回各参数">
    <svg class="lesson-visual__canvas--wide" viewBox="0 0 1080 520" role="img" aria-hidden="true" style="color: var(--md-default-fg-color);">
      <defs>
        <marker id="backprop-path-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
        </marker>
      </defs>

      <g data-step data-step-label="前向值与反向起点">
        <text x="40" y="34" font-size="18" font-weight="700" fill="currentColor">前向：x → a → h → s → L</text>
        <path d="M160 105 L250 105 M390 105 L465 105 M605 105 L700 105 M840 105 L925 105" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#backprop-path-arrow)" />
        <g fill="none" stroke="currentColor" stroke-width="2">
          <rect x="40" y="72" width="120" height="66" rx="10" />
          <rect x="250" y="72" width="140" height="66" rx="10" />
          <rect x="465" y="72" width="140" height="66" rx="10" />
          <rect x="700" y="72" width="140" height="66" rx="10" />
          <rect x="925" y="72" width="115" height="66" rx="10" />
        </g>
        <g text-anchor="middle" font-size="18" fill="currentColor">
          <text x="100" y="100">x = 2</text><text x="100" y="124">w₁ = 3, b₁ = 1</text>
          <text x="320" y="100">a = 7</text><text x="320" y="124">w₁x + b₁</text>
          <text x="535" y="100">h = 7</text><text x="535" y="124">ReLU(a)</text>
          <text x="770" y="100">s = 27</text><text x="770" y="124">w₂h + b₂</text>
          <text x="982" y="100">L = 4.5</text><text x="982" y="124">L̄ = 1</text>
        </g>
      </g>

      <g data-step data-step-label="L 返回到 s，并在 s 处分叉">
        <text x="40" y="206" font-size="18" font-weight="700" fill="currentColor">反向第 1 段</text>
        <path d="M925 190 L840 190" fill="none" stroke="currentColor" stroke-width="3" marker-end="url(#backprop-path-arrow)" />
        <text x="882" y="178" text-anchor="middle" font-size="17" fill="currentColor">× (s-y) = -3</text>
        <rect x="700" y="158" width="140" height="64" rx="10" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="2" />
        <text x="770" y="184" text-anchor="middle" font-size="18" fill="currentColor">s̄ = -3</text>
        <text x="770" y="207" text-anchor="middle" font-size="16" fill="currentColor">向 h、w₂、b₂ 分叉</text>
        <path d="M700 190 C630 190 635 265 565 265 M700 190 C650 190 650 330 565 330 M700 190 C650 190 650 395 565 395" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#backprop-path-arrow)" />
        <g font-size="17" fill="currentColor">
          <text x="385" y="270">h̄ = -3 × w₂ = -12</text>
          <text x="385" y="335">w̄₂ = -3 × h = -21</text>
          <text x="385" y="400">b̄₂ = -3 × 1 = -3</text>
        </g>
      </g>

      <g data-step data-step-label="梯度穿过 ReLU 返回到 a">
        <text x="40" y="452" font-size="18" font-weight="700" fill="currentColor">反向第 2 段</text>
        <path d="M380 440 L250 440" fill="none" stroke="currentColor" stroke-width="3" marker-end="url(#backprop-path-arrow)" />
        <text x="315" y="426" text-anchor="middle" font-size="17" fill="currentColor">× ReLU′(7) = 1</text>
        <rect x="90" y="408" width="160" height="64" rx="10" fill="currentColor" fill-opacity="0.06" stroke="currentColor" stroke-width="2" />
        <text x="170" y="436" text-anchor="middle" font-size="18" fill="currentColor">ā = -12</text>
        <text x="170" y="459" text-anchor="middle" font-size="16" fill="currentColor">继续向 x、w₁、b₁ 分叉</text>
      </g>

      <g data-step data-step-label="a 的三份参数与输入梯度">
        <path d="M250 472 C340 505 430 487 515 468 M250 472 C405 530 605 512 715 468 M250 472 C460 545 770 528 915 468" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#backprop-path-arrow)" />
        <g font-size="17" fill="currentColor">
          <text x="520" y="472">w̄₁ = -12 × x = -24</text>
          <text x="720" y="472">b̄₁ = -12 × 1 = -12</text>
          <text x="920" y="472">x̄ = -12 × w₁ = -36</text>
        </g>
      </g>
    </svg>
  </div>
  <ol data-lesson-steps>
    <li>前向得到 a = 7、h = 7、s = 27、L = 4.5；反向从 L̄ = 1 开始。</li>
    <li>L 对 s 的局部导数是 s-y = -3，因此 s̄ = -3；s 再向 h、w₂、b₂ 分出三份贡献。</li>
    <li>h̄ = -12，且 ReLU′(7) = 1，所以 ā = -12。</li>
    <li>a 向三个输入返回：w̄₁ = -24、b̄₁ = -12、x̄ = -36。</li>
  </ol>
  <figcaption>图 1：看梯度从 L 沿反向路径逐段累积；每条边乘局部导数，遇到分叉就把贡献分别交给各输入。</figcaption>
</figure>

按相同顺序写成链式法则就是：

$$
\bar s=s-y=-3,
$$

$$
\bar w_2=\bar s\,h=-21,\qquad
\bar b_2=\bar s=-3,\qquad
\bar h=\bar s\,w_2=-12,
$$

$$
\bar a=\bar h\,\operatorname{ReLU}'(a)=-12,
$$

$$
\bar w_1=\bar a\,x=-24,\qquad
\bar b_1=\bar a=-12.
$$

例如 $w_1$ 稍微增大，当前损失会下降，所以梯度为负。优化器沿负梯度更新时反而会增大 $w_1$；符号与直觉一致。

## 反向模式怎样复用计算

反向传播需要前向时的 $x,h,w_2$ 等值，因此框架通常保存反向公式需要的中间量。调用反向计算后，图和缓存往往可以释放。若一边保留所有计算图、一边不断追加新迭代，显存会随迭代增长；把用于日志的损失张量长期保存也可能意外保住整张图。

一个变量若流向多条分支，传回的贡献必须相加。设 $u$ 同时进入 $p=u^2$ 与 $q=3u$，则 $L=p+q$ 给出

$$
\bar u=\bar p\frac{\partial p}{\partial u}
+\bar q\frac{\partial q}{\partial u}=2u+3.
$$

“乘局部导数”处理一条边，“在汇合点求和”处理多条路径。这也解释了参数共享为何可训练：同一个权重在多个位置使用时，各位置的梯度贡献会累加到同一参数。

## 从标量到向量—雅可比积

神经网络节点通常是向量。令 $\mathbf y=f(\mathbf x)$，其中 $\mathbf x\in\mathbb R^n$、$\mathbf y\in\mathbb R^m$，局部雅可比矩阵为

$$
\mathbf J_f=\frac{\partial\mathbf y}{\partial\mathbf x}\in\mathbb R^{m\times n}.
$$

损失 $L$ 最终仍是标量。若梯度用列向量表示，反向一步为

$$
\nabla_{\mathbf x}L
=\mathbf J_f^\top\nabla_{\mathbf y}L.
$$

在本页的列梯度约定下，反向计算是“局部雅可比转置乘上游列梯度”，即 $\mathbf J_f^\top\nabla_{\mathbf y}L$。若把上游梯度写成行协向量，同一个运算就是通常所说的向量—雅可比积 (vector–Jacobian product, VJP)：

$$
(\nabla_{\mathbf y}L)^\top\mathbf J_f.
$$

两种写法互为转置，得到的分量相同。框架只求这个乘积，不必构造可能很大的完整雅可比。

以线性层 $\mathbf y=\mathbf W\mathbf x+\mathbf b$ 为例，收到 $\bar{\mathbf y}=\nabla_{\mathbf y}L$ 后：

$$
\nabla_{\mathbf x}L=\mathbf W^\top\bar{\mathbf y},\qquad
\nabla_{\mathbf W}L=\bar{\mathbf y}\mathbf x^\top,\qquad
\nabla_{\mathbf b}L=\bar{\mathbf y}.
$$

若 $\mathbf W$ 形状为 $d_{out}\times d_{in}$，三项形状依次是 $d_{in}$、$d_{out}\times d_{in}$、$d_{out}$。反向代码出错时，先核对这三个形状，比盯着展开后的下标更有效。

## 用自动微分核对手算

```python
import torch

x = torch.tensor(2.0)
w1 = torch.tensor(3.0, requires_grad=True)
b1 = torch.tensor(1.0, requires_grad=True)
w2 = torch.tensor(4.0, requires_grad=True)
b2 = torch.tensor(-1.0, requires_grad=True)
y = torch.tensor(30.0)

h = torch.relu(w1 * x + b1)
score = w2 * h + b2
loss = 0.5 * (score - y) ** 2
loss.backward()

print(loss.item())                 # 4.5
print(w1.grad, b1.grad)            # tensor(-24.) tensor(-12.)
print(w2.grad, b2.grad)            # tensor(-21.) tensor(-3.)
```

自动微分省掉的是机械计算，不是数学含义。`backward()` 仍按图执行上面的 VJP；叶子参数的 `.grad` 默认做加法累积，因此真正训练时需要决定何时清零。完整状态顺序见[Mini-batch 与训练循环](./minibatch-and-training-loop.md)。

!!! tip "面试 / 工程重点"
    反向传播为什么适合神经网络训练？因为目标通常是一个标量损失、参数却很多。一次反向遍历可得到所有参数的梯度，代价通常与一次前向计算同量级，而不是为每个参数单独做一次前向差分。

!!! warning "梯度检查的边界"
    中心有限差分适合在小模型上核对实现，但 ReLU 的零点不可导，浮点步长也不能太大或太小。检查时避开折点、使用双精度，并比较相对误差；不要把有限差分当作训练算法。
