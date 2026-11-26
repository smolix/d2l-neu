# Modern Convolutional Neural Networks
:label:`chap_modern_cnn`

With the mechanics of convolutional layers and LeNet in hand (:numref:`chap_cnn`), we now trace a decade of convolutional architecture design, from AlexNet's 2012 ImageNet breakthrough to the early-2020s dialogue between convolutional networks and Vision Transformers. Rather than treating each landmark model as an isolated artifact, we ask what bottleneck prompted each design step and carry a cumulative toolkit forward across the chapter.

The opening sections move from hand-wired layers to modular building blocks. AlexNet (:numref:`sec_alexnet`) demonstrates how end-to-end feature learning overtook hand-engineered vision pipelines once large labeled datasets and GPUs converged, introducing ReLU activations, dropout, data augmentation, and a warmup-and-cosine learning-rate schedule reused throughout the chapter, while exposing a stark imbalance between convolutional compute and dense-head parameter count. VGG, Network in Network (NiN), and GoogLeNet (:numref:`sec_blocks`) address that imbalance by organizing networks into repeatable $3 \times 3$ blocks, mixing channels per pixel with $1 \times 1$ convolutions, replacing massive dense heads with global average pooling, and combining parallel multi-scale branches inside a stem--body--head layout.

The middle sections scale depth and hardware efficiency. Normalization layers (:numref:`sec_batch_norm`) standardize intermediate activations across the minibatch or within individual examples, rendering hidden layers insensitive to weight scale and widening usable learning rates by orders of magnitude. Residual networks (:numref:`sec_resnet`) resolve the optimization degradation of deep plain stacks through additive shortcuts $\mathbf{x} + g(\mathbf{x})$ that keep the identity mapping at $g = 0$, before extending the block via grouped convolutions (ResNeXt) and feature concatenation (DenseNet). Efficient convnets (:numref:`sec_efficient_cnns`) elevate latency, memory, and energy alongside accuracy through depthwise-separable factorization, inverted bottlenecks, compound scaling, squeeze-and-excitation gates, and structural re-parameterization, while showing why operation counts often diverge from GPU wall-clock time.

The closing sections disentangle training methodology from architecture. Modern training recipes (:numref:`sec_training_recipes`) show that contemporary optimization and regularization—AdamW, cosine warmup, label smoothing, Mixup, CutMix, stochastic depth, and weight averaging—account for several percentage points of ImageNet accuracy on an unchanged ResNet. Holding such a recipe fixed, ConvNeXt (:numref:`sec_convnext`) modernizes a ResNet step by step to match hierarchical Vision Transformers without attention, and convolutional design spaces (:numref:`sec_cnn-design`) compare empirical error distributions across sampled populations to distill the simple structural rules of the RegNet family.

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
- **Signal & degradation drift:** Sigmoids shrink gradients by $\le 1/4^L$; even with ReLU + BN, a 56-layer plain network trains worse than a 20-layer one ($\approx 0.28$ vs. $0.20$ loss).
- **92% dense-head bloat & FLOPs vs. latency:** AlexNet/VGG put $>92\%$ of weights in the head; depthwise convs cut MACs $8\times$ yet hit GPU memory bandwidth limits.
- **Confounded recipes:** A modern recipe alone (LAMB, BCE loss, RandAugment/Mixup/CutMix, 600 epochs) adds $4.3$ points to an unchanged ResNet-50 ($76.1\% \to 80.4\%$).
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
- **§§{sec_alexnet}–§{num:sec_blocks} (AlexNet, VGG, NiN, GoogLeNet):** ReLU, dropout, $3 \times 3$ blocks, $1 \times 1$ bottlenecks, and global average pooling inside a stem–body–head template.
- **§§{sec_batch_norm}–§{num:sec_resnet} (Norms & Residuals):** BatchNorm, LayerNorm, GroupNorm, additive shortcuts $\mathbf{x} + g(\mathbf{x})$, grouped ResNeXt blocks, and DenseNet concatenation.
- **§§{sec_efficient_cnns}–§{num:sec_cnn-design} (Efficiency, Recipes, ConvNeXt, RegNet):** Depthwise factorization, RepVGG fusion, AdamW/Mixup/EMA, $7 \times 7$ ConvNeXt blocks, and population error CDFs.
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
[Empirical Comparison]{.kicker}

::: {.cols .vc}
::: {.col}
- **AlexNet $\to$ Quarter-VGG / ResNet-18:** Accuracy climbs from $91\%$ to $93\text{--}94\%$ while parameters shrink from $46.8\textrm{M}$ to $11.2\textrm{M}$ (or $0.6\textrm{M}$ with GAP).
- **BNLeNet & ResNet-56:** BatchNorm widens the usable LR band $300\times$ ($\eta_0 \in [0.03, 10]$); residual skips reverse 56-layer degradation.
- **ConvNeXt & RegNetX32:** Tie ResNet-18 ($\approx 94\%$) with $3.38\textrm{M}$ and $0.52\textrm{M}$ parameters ($21\times$ smaller).
:::

::: {.col .fig .big}
@fig:ch07-mcnn-pareto-scoreboard
:::
:::
:::

