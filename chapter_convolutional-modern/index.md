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

What made deep convnets trainable<br>
**AlexNet · blocks · normalization · residual connections · efficient convnets · training recipes · ConvNeXt · design spaces**
:::
:::

::: {.slide title="Stacking More LeNet Layers Does Not Give a Better Network"}
[The problem]{.kicker}

- LeNet: five layers, about 62,000 parameters, sigmoid units.
- Sigmoid units shrink the gradient at every layer. Even with ReLUs and
  normalization, a 56-layer *plain* network ends with a higher training
  loss than a 20-layer one (0.25–0.3 against 0.2 on Fashion-MNIST).
- Every new layer needs its own width, kernel size, and initialization.
- Reported accuracies mix architecture with the training procedure.

::: {.d2l-note}
Between 2012 and 2016 ImageNet winners grew from 8 layers to more than 150.
Which ideas made that possible, and which gains came from training?
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- count the parameters and multiply-adds of a convnet, and say where they sit;
- build networks from repeated blocks, 1 × 1 convolutions, and global pooling;
- implement batch normalization and explain its training and prediction modes;
- explain the degradation problem and why a residual block fixes it;
- trade accuracy for arithmetic with depthwise-separable convolutions;
- separate the effect of a training recipe from that of an architecture.

**Running example:** every network trained on Fashion-MNIST.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question |
|---|---|
| AlexNet | what let a deep network beat engineered features in 2012? |
| VGG, NiN, GoogLeNet | how do we build depth from repeated blocks? |
| Normalization Layers | how does normalizing activations speed up training? |
| ResNet, ResNeXt, DenseNet | why do deeper plain networks train worse, and what fixes it? |
| Efficient Convnets | how do we cut a layer's arithmetic tenfold, and at what cost? |
| Training Recipes | how much accuracy comes from the training procedure? |
| ConvNeXt | how far does a ResNet get, modernized one change at a time? |
| Design Spaces (RegNet) | what rules appear in populations of networks? |
:::

::: {.slide title="Every Section Ends With a Measured Result"}
[Running examples]{.kicker}

| section | result on Fashion-MNIST |
|---|---|
| AlexNet | 46.8M parameters; about 91% in a third of LeNet's epochs |
| VGG, NiN, GoogLeNet | 89–93% with 3.4M to 43M parameters; a pooled head on VGG's body loses about 2 points |
| Normalization Layers | with batch norm, LeNet trains at every peak rate from 0.03 to 10 |
| ResNet, ResNeXt, DenseNet | 56 plain layers train worse than 20, 56 residual layers better; ResNet-18 about 93% |
| Efficient Convnets | mini-MobileNet 91–92%; a narrow dense network of equal arithmetic about 93% |
| Training Recipes | the modern recipe wins every paired run, by 0.5–1.5 points |
| ConvNeXt | ConvNeXt blocks tie with a matched ResNet-18 (≈ 94%); the patchify stem loses 2 points |
| Design Spaces (RegNet) | the width rule is worth under a point; RegNetX32 ≈ 94% with a third of ResNet-18's multiply-adds |
:::
