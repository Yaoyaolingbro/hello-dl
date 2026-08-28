---
level: basic
roles:
  - core
prerequisites:
  - 感知机与多层感知机
estimated_time: 30min
status: complete
---

# 计算图与自动微分

![前向计算与反向传递的概念示意图](../../assets/computational-graph-forward-backward.png)

*图 1：这是前向计算与反向传递的概念性总览；准确的计算图见下文。*

!!! info "参考资料"
    - [Autograd mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html) — PyTorch Documentation
    - [Deep Learning](https://www.deeplearningbook.org/) — 第 6.5 节

## 直觉 (Intuition)

一个损失函数写在纸上可能很长，程序执行时却只是乘、加、平方等小操作的接力。计算图把这些操作和中间结果记下来。这样一来，前向时算出损失，反向时就能逐个节点回答：这里变一点，损失会变多少？

自动微分做的正是这件事。它不是把整个式子交给符号工具化简，也不是靠有限差分猜导数；它沿着实际执行过的操作，重复使用链式法则。

## 把表达式拆成节点

先看一个只有标量的例子：

$$
a=wx,\qquad z=a+b,\qquad L=z^2.
$$

令 $x=2$、$w=3$、$b=1$。先从左向右代入：$a=6$，再得到 $z=7$，最后 $L=49$。这就是前向计算；它不神秘，就是按依赖关系把中间值算出来。

下面的图只保留依赖和数值。点“下一步”，先看输入怎样算出损失，再看梯度沿原路返回。

<figure class="lesson-visual" data-lesson-visual data-interval="1800">
  <div data-lesson-stage role="img" aria-label="标量计算图的前向计算与反向梯度">
    <svg class="lesson-visual__canvas--wide" viewBox="0 0 980 420" role="img" aria-hidden="true" style="color: var(--md-default-fg-color);">
      <defs>
        <marker id="forward-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
        </marker>
        <marker id="reverse-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="#d97706" />
        </marker>
      </defs>

      <g data-step data-step-label="输入 x、w、b">
        <text x="40" y="30" font-size="18" font-weight="700" fill="currentColor">输入</text>
        <rect x="40" y="48" width="120" height="54" rx="10" fill="none" stroke="currentColor" stroke-width="2" />
        <text x="100" y="82" text-anchor="middle" font-size="20" fill="currentColor">x = 2</text>
        <rect x="40" y="126" width="120" height="54" rx="10" fill="none" stroke="currentColor" stroke-width="2" />
        <text x="100" y="160" text-anchor="middle" font-size="20" fill="currentColor">w = 3</text>
        <rect x="40" y="245" width="120" height="54" rx="10" fill="none" stroke="currentColor" stroke-width="2" />
        <text x="100" y="279" text-anchor="middle" font-size="20" fill="currentColor">b = 1</text>
      </g>

      <g data-step data-step-label="乘法得到 a">
        <path d="M160 75 C210 75, 205 128, 255 128" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#forward-arrow)" />
        <path d="M160 153 C210 153, 205 138, 255 138" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#forward-arrow)" />
        <rect x="265" y="100" width="150" height="70" rx="12" fill="#e0f2fe" stroke="#0284c7" stroke-width="2" />
        <text x="340" y="126" text-anchor="middle" font-size="16" fill="#0c4a6e">乘法 wx</text>
        <text x="340" y="153" text-anchor="middle" font-size="22" font-weight="700" fill="#0c4a6e">a = 6</text>
      </g>

      <g data-step data-step-label="加法得到 z">
        <path d="M415 135 C455 135, 455 212, 500 212" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#forward-arrow)" />
        <path d="M160 272 C340 272, 370 222, 500 222" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#forward-arrow)" />
        <rect x="510" y="184" width="150" height="70" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2" />
        <text x="585" y="210" text-anchor="middle" font-size="16" fill="#14532d">加法 a + b</text>
        <text x="585" y="237" text-anchor="middle" font-size="22" font-weight="700" fill="#14532d">z = 7</text>
      </g>

      <g data-step data-step-label="平方得到损失 L">
        <path d="M660 219 L745 219" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#forward-arrow)" />
        <rect x="755" y="184" width="150" height="70" rx="12" fill="#f3e8ff" stroke="#9333ea" stroke-width="2" />
        <text x="830" y="210" text-anchor="middle" font-size="16" fill="#581c87">平方 z²</text>
        <text x="830" y="237" text-anchor="middle" font-size="22" font-weight="700" fill="#581c87">L = 49</text>
      </g>

      <g data-step data-step-label="梯度反向返回">
        <text x="910" y="327" text-anchor="end" font-size="17" font-weight="700" fill="#b45309">反向</text>
        <path d="M830 254 C830 334, 690 334, 610 260" fill="none" stroke="#d97706" stroke-width="3" marker-end="url(#reverse-arrow)" />
        <text x="735" y="315" text-anchor="middle" font-size="17" fill="#b45309">∂L/∂z = 14</text>
        <path d="M585 254 C565 363, 390 363, 354 177" fill="none" stroke="#d97706" stroke-width="3" marker-end="url(#reverse-arrow)" />
        <text x="485" y="363" text-anchor="middle" font-size="17" fill="#b45309">∂L/∂a = 14</text>
        <path d="M330 170 C310 395, 150 390, 115 186" fill="none" stroke="#d97706" stroke-width="3" marker-end="url(#reverse-arrow)" />
        <text x="250" y="400" text-anchor="middle" font-size="18" font-weight="700" fill="#b45309">∂L/∂w = 28</text>
      </g>
    </svg>
  </div>
  <ol data-lesson-steps>
    <li>输入是 x = 2、w = 3、b = 1。</li>
    <li>乘法节点计算 a = wx，得到 a = 6。</li>
    <li>加法节点计算 z = a + b，得到 z = 7。</li>
    <li>平方节点计算 L = z²，得到 L = 49。</li>
    <li>反向时梯度依次返回，最后得到 ∂L/∂w = 28。</li>
  </ol>
  <figcaption>图 2：看前向箭头如何把数值送到 L，再看梯度箭头如何把梯度送回 w。</figcaption>
</figure>

现在从 $L$ 往回看。反向计算先放一个单位的上游梯度 $\partial L/\partial L=1$ 在终点。到每个节点时，只做两件小事：把上游梯度乘上该节点的局部导数，再把结果交给它的输入。对本例来说：

$$
\frac{\partial L}{\partial z}=2z=14,
$$

$$
\frac{\partial L}{\partial a}
=\frac{\partial L}{\partial z}\frac{\partial z}{\partial a}
=14,
$$

$$
\frac{\partial L}{\partial w}
=\frac{\partial L}{\partial a}\frac{\partial a}{\partial w}
=14x=28.
$$

所以 $w$ 增加一点点时，$L$ 大约以 $28$ 倍的速度增加。这里没有重新展开整个复合函数；每个节点只知道自己的局部规则，接住并传递信息即可。这一页到此只讨论标量的局部导数。向量–雅可比积和完整的 reverse-mode 推导留在[反向传播](../03-training/backpropagation.md)。

## 分支处为什么要相加

一个变量也可能被用两次。设：

$$
L=u^2+3u.
$$

从反向方向看，平方分支和线性分支各自传回一份贡献；两份贡献先累加，再交给 $u$。点“下一步”，观察两条线最后在哪里会合。

<figure class="lesson-visual" data-lesson-visual data-interval="1800">
  <div data-lesson-stage role="img" aria-label="两个分支的梯度在同一节点累加">
    <svg class="lesson-visual__canvas--wide" viewBox="0 0 900 340" role="img" aria-hidden="true" style="color: var(--md-default-fg-color);">
      <defs>
        <marker id="branch-arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L9,3 z" fill="currentColor" />
        </marker>
      </defs>

      <g data-step data-step-label="平方分支贡献 2u">
        <rect x="55" y="45" width="190" height="68" rx="12" fill="#e0f2fe" stroke="#0284c7" stroke-width="2" />
        <text x="150" y="73" text-anchor="middle" font-size="17" fill="#0c4a6e">平方分支</text>
        <text x="150" y="99" text-anchor="middle" font-size="22" font-weight="700" fill="#0c4a6e">贡献 2u</text>
        <path d="M245 79 C390 79, 430 132, 558 161" fill="none" stroke="#0284c7" stroke-width="3" marker-end="url(#branch-arrow)" />
      </g>

      <g data-step data-step-label="线性分支贡献 3">
        <rect x="55" y="225" width="190" height="68" rx="12" fill="#dcfce7" stroke="#16a34a" stroke-width="2" />
        <text x="150" y="253" text-anchor="middle" font-size="17" fill="#14532d">线性分支</text>
        <text x="150" y="279" text-anchor="middle" font-size="22" font-weight="700" fill="#14532d">贡献 3</text>
        <path d="M245 259 C390 259, 430 206, 558 177" fill="none" stroke="#16a34a" stroke-width="3" marker-end="url(#branch-arrow)" />
      </g>

      <g data-step data-step-label="在同一点累加">
        <circle cx="590" cy="169" r="36" fill="#fef3c7" stroke="#d97706" stroke-width="3" />
        <text x="590" y="179" text-anchor="middle" font-size="30" font-weight="700" fill="#92400e">Σ</text>
        <text x="590" y="225" text-anchor="middle" font-size="16" fill="#92400e">同一累加点</text>
        <path d="M626 169 L695 169" fill="none" stroke="currentColor" stroke-width="3" marker-end="url(#branch-arrow)" />
        <rect x="705" y="133" width="150" height="72" rx="12" fill="#f3e8ff" stroke="#9333ea" stroke-width="2" />
        <text x="780" y="160" text-anchor="middle" font-size="16" fill="#581c87">传给 u</text>
        <text x="780" y="187" text-anchor="middle" font-size="20" font-weight="700" fill="#581c87">2u + 3</text>
      </g>
    </svg>
  </div>
  <ol data-lesson-steps>
    <li>平方分支传回贡献 2u。</li>
    <li>线性分支传回贡献 3。</li>
    <li>2u 与 3 在同一个梯度累加点相加，得到 ∂L/∂u = 2u + 3。</li>
  </ol>
  <figcaption>图 3：看蓝线与绿线落到同一个 Σ 累加点；反向传播在这里求和，不会任选一条分支。</figcaption>
</figure>

两份贡献的具体数值仍由局部导数给出：

$$
\frac{\partial L}{\partial u}=2u+3.
$$

也就是说，平方分支贡献 $2u$，线性分支贡献 $3$。同一张计算图里，若多条路径回到同一个变量，反向传播会把各条路径的贡献相加。不要把它和另一种累积混在一起：PyTorch 会把多次 `backward()` 的结果继续加到 `.grad` 中，除非主动清零。清零时机见[反向传播](../03-training/backpropagation.md)和[训练循环](../03-training/minibatch-and-training-loop.md)。

## 自动微分和数值微分不同

有限差分用

$$
\frac{f(x+\varepsilon)-f(x-\varepsilon)}{2\varepsilon}
$$

近似导数，会有截断误差和浮点误差。自动微分对基本操作使用精确导数，再通过链式法则组合；误差只来自普通浮点计算。有限差分仍适合做 gradient check，但不适合日常训练。

```python
import torch

w = torch.tensor(3.0, requires_grad=True)
x = torch.tensor(2.0)
b = torch.tensor(1.0)
loss = (w * x + b) ** 2
loss.backward()

print(w.grad)  # tensor(28.)
```

`requires_grad=True` 表示需要记录与 $w$ 有关的操作；`backward()` 从标量损失反向计算梯度。中间图何时保存、梯度为何清零，会在[反向传播](../03-training/backpropagation.md)和[训练循环](../03-training/minibatch-and-training-loop.md)中说明。

!!! warning "常见误区"
    自动微分算的是当前执行路径的导数。包含条件分支时，没走到的分支不在这次计算图里；这不是漏算，而是这次程序实际定义的函数就没有经过那条分支。
