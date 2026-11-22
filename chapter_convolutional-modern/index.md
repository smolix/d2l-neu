# Modern Convolutional Neural Networks
:label:`chap_modern_cnn`

With the mechanics of convolutional layers and LeNet in hand (:numref:`chap_cnn`), we now trace the ten-year arc of modern convolutional architecture design—from AlexNet's 2012 ImageNet breakthrough to the early-2020s dialogue between convnets and Vision Transformers. Rather than treating each landmark model as an isolated museum piece, we examine the concrete bottleneck each architecture solved and carry a cumulative toolkit forward from section to section:

1. **AlexNet** (:numref:`sec_alexnet`) shows why end-to-end learned representations overtook hand-crafted vision pipelines once large labeled datasets (ImageNet) and GPUs converged, introducing ReLU activations, dropout, and cosine learning-rate warmup while exposing a glaring parameter imbalance: 95% of its arithmetic sits in five convolutions, whereas 92% of its 46.8M weights sit in its dense classifier head.
2. **VGG, NiN, and GoogLeNet** (:numref:`sec_blocks`) replace ad-hoc layer lists with repeatable *blocks*: VGG stacks narrow $3 \times 3$ convolutions inside stage templates; Network in Network (NiN) introduces $1 \times 1$ channel-mixing convolutions and replaces the 43M-parameter dense head with parameter-free global average pooling; and GoogLeNet combines multi-scale parallel branches with $1 \times 1$ bottlenecks inside the stem–body–head macro layout that every subsequent convnet inherits.
3. **Normalization layers** (:numref:`sec_batch_norm`) stabilize deep optimization by standardizing intermediate activations—across the minibatch in *batch normalization* or within each example in *layer*, *group*, and *instance normalization*—making networks insensitive to weight scale and widening the usable learning-rate band by orders of magnitude.
4. **ResNet, ResNeXt, and DenseNet** (:numref:`sec_resnet`) solve the *degradation problem* of deep plain networks via additive residual shortcuts $\mathbf{x} + g(\mathbf{x})$ that nest function classes and keep the identity map at $g = 0$, then generalize the block through grouped convolutions (ResNeXt) and feature concatenation (DenseNet).
5. **Efficient convnets** (:numref:`sec_efficient_cnns`) bring latency, memory, and energy to the fore via depthwise-separable factorization (MobileNet), inverted bottlenecks (MobileNetV2), compound scaling and squeeze-and-excitation gates (EfficientNet), and train-time multi-branch blocks that algebraically fuse into single $3 \times 3$ convolutions for deployment (RepVGG)—while showing why multiply-add counts diverge from GPU wall-clock time.
6. **Modern training recipes** (:numref:`sec_training_recipes`) decouple training procedure from architecture: upgrading a fixed 2015 ResNet with AdamW, cosine warmup, label smoothing, Mixup, stochastic depth, and exponential moving averages accounts for over four ImageNet top-1 points on its own.
7. **ConvNeXt** (:numref:`sec_convnext`) holds the modern recipe fixed and modernizes a ResNet one component at a time—adopting $7 \times 7$ depthwise convolutions, $4\times$ inverted bottlenecks, a single GELU and LayerNorm per block, and separate downsampling layers—matching Vision Transformers without a single attention layer.
8. **Convolutional network design spaces** (:numref:`sec_cnn-design`) shift the unit of analysis from individual networks to *populations* sampled from the AnyNet template, comparing empirical error CDFs to distil the simple structural rules of the RegNet family.

Throughout the chapter, we evaluate every architecture and recipe on Fashion-MNIST under controlled schedules while citing large-scale ImageNet ablations wherever small $28 \times 28$ grayscale garments hit their ~94% dataset ceiling.

```toc
:maxdepth: 2

alexnet
blocks
batch-norm
resnet
efficient-convnets
training-recipes
convnext
cnn-design
```

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §7]{.kicker}

Modern Convolutional Neural Networks<br>**AlexNet · blocks · normalization · residual shortcuts · efficient convnets · training recipes · ConvNeXt · RegNet**
:::
:::

::: {.slide title="Why Stacking More LeNet Layers Fails in Practice"}
[The Central Problem]{.kicker}

::: {.cols .vc}
::: {.col}
- **Signal & degradation drift:** Sigmoids shrink gradients by $\le 1/4^L$; even with ReLU + BN, a 56-layer plain network trains worse than a 20-layer one ($0.27$ vs. $0.20$ loss).
- **92% dense-head bloat & FLOPs vs. latency:** AlexNet/VGG put $>92\%$ of weights in the head; depthwise convs cut MACs $8\times$ yet hit GPU memory bandwidth limits.
- **Confounded recipes:** Modern training recipes alone add $+4.3\%$ to an unchanged ResNet-50.
:::

::: {.col .fig .big}
@fig:ch07-mcnn-four-scaling-barriers
:::
:::
:::

::: {.slide title="Architectural Toolkit Across the Chapter"}
[Chapter Roadmap]{.kicker}

::: {.cols .vc}
::: {.col}
- **§§7.1–7.2 (AlexNet, VGG, NiN, GoogLeNet):** ReLU, dropout, $3 \times 3$ blocks, $1 \times 1$ bottlenecks, and global average pooling inside a stem–body–head template.
- **§§7.3–7.4 (Norms & Residuals):** BatchNorm, LayerNorm, GroupNorm, additive shortcuts $\mathbf{x} + g(\mathbf{x})$, grouped ResNeXt blocks, and DenseNet concatenation.
- **§§7.5–7.8 (Efficiency, Recipes, ConvNeXt, RegNet):** Depthwise factorization, RepVGG fusion, AdamW/Mixup/EMA, $7 \times 7$ ConvNeXt blocks, and population error CDFs.
:::

::: {.col .fig .big}
@fig:ch07-mcnn-decade-evolution-arc
:::
:::
:::

::: {.slide title="Three Unifying Design Invariants"}
[Core Mental Models]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. RF per FLOP & GAP:** Stacked $3 \times 3$ convs span $5 \times 5$ ($18c^2$ vs. $25c^2$) or $7 \times 7$ ($27c^2$ vs. $49c^2$) with more nonlinearities; GAP removes 43M-param heads.
- **2. Scale invariance & nested classes:** $\textrm{BN}(a\mathbf{w}^\top\mathbf{h}) = \textrm{BN}(\mathbf{w}^\top\mathbf{h})$, and $\mathbf{x} + g(\mathbf{x})$ keeps the identity at $g=0$ ($\mathcal{F} \subseteq \mathcal{F}'$).
- **3. Factorization vs. intensity:** Depthwise-separable convs cut MACs to $1/c_\textrm{o} + 1/k^2 \approx 1/9$, but read memory at $9$ vs. $9c_\textrm{o}$ MACs/read.
:::

::: {.col .fig .big}
@fig:ch07-mcnn-three-unifying-invariants
:::
:::
:::

::: {.slide title="Controlled Fashion-MNIST Running Benchmark"}
[Empirical Scoreboard]{.kicker}

::: {.cols .vc}
::: {.col}
- **AlexNet $\to$ Quarter-VGG / ResNet-18:** Accuracy climbs from $91\%$ to $93\text{--}94\%$ while parameters shrink from $46.8\textrm{M}$ to $11.2\textrm{M}$ (or $0.6\textrm{M}$ with GAP).
- **BNLeNet & ResNet-56:** BatchNorm widens the usable LR band $100\times$ ($\eta_0 \in [0.03, 10]$); residual skips reverse 56-layer degradation.
- **ConvNeXt & RegNetX32:** Tie ResNet-18 ($\approx 94\%$) with $3.38\textrm{M}$ and $0.52\textrm{M}$ parameters ($21\times$ smaller).
:::

::: {.col .fig .big}
@fig:ch07-mcnn-pareto-scoreboard
:::
:::
:::

