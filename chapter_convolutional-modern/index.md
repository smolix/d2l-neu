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

From 2012 to 2022, vision backbones scaled from 8 to 150+ layers while shrinking parameter and latency budgets, exposing four fundamental barriers:

- **Signal & gradient drift:** Saturating sigmoids shrink gradients by $\le 1/4$ per layer; even with ReLUs and batch normalization, a 56-layer *plain* network converges to higher training loss than a 20-layer one ($0.27$ vs. $0.20$ on Fashion-MNIST).
- **Parameter vs. compute asymmetry:** AlexNet and VGG concentrate over 92% of their parameters inside flattened dense classifier heads that contribute under 5% of total multiply-adds.
- **Hardware vs. theoretical FLOPs:** Low arithmetic-intensity operations (depthwise and grouped convolutions) can run slower on GPUs despite $8\times$ fewer multiply-adds.
- **Confounded benchmarks:** Modern training recipes alone boost an unchanged 2015 ResNet-50 by $>4$ ImageNet top-1 points.
:::

::: {.slide title="Architectural Toolkit Across the Chapter"}
[Chapter Roadmap]{.kicker}

Each section isolates one structural bottleneck and introduces a reusable primitive carried forward through the rest of the book:

| Section | Core Bottleneck Solved | Key Primitives Introduced |
|:--|:--|:--|
| **§7.1 AlexNet** | Hand-crafted features plateau on large data | End-to-end learned filters, ReLU, dropout, cosine warmup |
| **§7.2 VGG, NiN, GoogLeNet** | Ad-hoc layer design & 100M+ dense heads | Stacked $3\times 3$ blocks, $1\times 1$ mixing/bottlenecks, global average pooling |
| **§7.3 Normalization** | Scale drift & fragile learning-rate bands | BatchNorm (minibatch + running stats), LayerNorm, GroupNorm |
| **§7.4 ResNet, ResNeXt, DenseNet** | Deep plain networks degrade in training loss | Additive shortcuts $\mathbf{x} + g(\mathbf{x})$, grouped convs, concatenation |
| **§7.5 Efficient Convnets** | Mobile/edge latency, memory & energy limits | Depthwise-separable convs, inverted bottlenecks, SE gates, RepVGG fusion |
| **§7.6 Training Recipes** | Architecture gains confounded with training | AdamW, label smoothing, Mixup, stochastic depth, weight EMA |
| **§7.7 ConvNeXt** | Vision Transformers outscoring 2015 ResNets | Step-by-step ResNet modernization ($7\times 7$ depthwise, single GELU/LN) |
| **§7.8 Design Spaces (RegNet)** | Single-model search yields no general rules | Population error CDFs over AnyNet, linear stage-width staircases |
:::

::: {.slide title="Three Unifying Design Invariants"}
[Core Mental Models]{.kicker}

Every modern convolutional backbone relies on three quantitative principles that govern accuracy, trainability, and speed:

- **Receptive field per FLOP and stem–body–head layout:** Two stacked $3\times 3$ convolutions span the same $5\times 5$ window as one $5\times 5$ layer using $18c^2$ vs. $25c^2$ weights and two nonlinearities, while parameter-free global average pooling eliminates dense heads.
- **Scale invariance and nested function classes:** Normalization makes layer outputs invariant to weight scale ($\textrm{BN}(a\mathbf{w}^\top\mathbf{h}) = \textrm{BN}(\mathbf{w}^\top\mathbf{h})$), and residual shortcuts $\mathbf{x} + g(\mathbf{x})$ place the identity map at $g = 0$ so deeper models strictly contain shallower ones ($\mathcal{F} \subseteq \mathcal{F}'$).
- **Factorization ratio vs. arithmetic intensity:** Depthwise-separable convolutions cut multiply-adds to $1/c_{\textrm{o}} + 1/k^2 \approx 1/9$ for $3\times 3$ kernels, but read memory with only $9$ multiply-adds per activation element versus $9c_{\textrm{o}}$ for dense layers.
:::

::: {.slide title="Controlled Fashion-MNIST Running Benchmark"}
[Empirical Scoreboard]{.kicker}

We evaluate every architecture under controlled cosine schedules on Fashion-MNIST while pairing small-scale runs with large-scale ImageNet ablations:

| Architecture / Experiment | Parameters | Multiply-Adds | Fashion-MNIST Result & Takeaway |
|:--|--:|--:|:--|
| **AlexNet** (§7.1, $224^2$) | 46.8M | 938M | $\approx 91\%$ in 10 epochs ($3\times$ fewer epochs than LeNet, $2{,}250\times$ the FLOPs) |
| **VGG / NiN / GoogLeNet** (§7.2) | 0.6M–43.1M | 276M–798M | Quarter-VGG hits $93\%$; global average pooling cuts 99% of weights for $-2$ pts |
| **BNLeNet** (§7.3, $28^2$) | 61.9K | 0.42M | Trains cleanly across a $300\times$ learning-rate span ($\eta_0 \in [0.03, 10]$) |
| **Plain-56 vs. ResNet-56** (§7.4) | 0.85M | 125M | Plain-56 degrades ($0.27 > 0.20$ loss); ResNet-56 reverses ordering ($0.16$) |
| **Mini-MobileNet vs. Dense** (§7.5) | 0.54M / 76K | 50M | $7.7\times$ fewer FLOPs gives $2\text{--}3.5\times$ GPU speedup; narrow dense wins at 50M |
| **ResNet-18: 2015 vs. Modern** (§7.6) | 11.2M | 319M | Modern recipe (AdamW + Mixup + smoothing) wins every paired 10k-image run |
| **ConvNeXt vs. ResNet-18** (§7.7) | 3.38M | 44M | Modernized blocks tie at $\approx 94\%$; coarse $4\times 4$ patchify stem loses $2.5$ pts |
| **RegNetX32 vs. ResNet-18** (§7.8) | 0.52M | 112M | Matches ResNet-18 ($\approx 94\%$) with $<5\%$ of the weights and $1/3$ of the FLOPs |
:::

