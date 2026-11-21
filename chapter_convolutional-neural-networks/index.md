# Convolutional Neural Networks
:label:`chap_cnn`

Every image classifier we have built so far has ignored the spatial geometry of
images, flattening a two-dimensional pixel grid into a one-dimensional vector.
The MLP in :numref:`chap_perceptrons` treats a $28 \times 28$ Fashion-MNIST
photo as an unordered list of 784 numbers: if we applied the same fixed random
permutation to the pixels of every image and retrained, the MLP would reach the
exact same accuracy—even though the scrambled images would look like pure static
to human eyes. Humans recognize a sneaker by its outline and textures, which
are patterns formed by *neighboring* pixels, and an outline carries the same
meaning wherever in the frame it appears.

Neuroscience offered an early clue. In classic experiments on cats and monkeys,
Hubel and Wiesel discovered that neurons in the early visual cortex respond to
oriented edges and bars within localized patches of the visual field—each
neuron's *receptive field*
:cite:`Hubel.Wiesel.1959,Hubel.Wiesel.1962,Hubel.Wiesel.1968`. Later,
:citet:`Field.1987` modeled these responses on natural images using a small bank
of local filters swept across every image location (:numref:`field_visual`).

![Six receptive-field filters and their responses to a natural image, adapted from :citet:`Field.1987`. The left panel shows the filters, the middle panel the input image, and the right panel the filtered responses sampled at intervals proportional to each filter's size. Different filters respond to different local edge patterns.](../img/field-visual.png)
:label:`field_visual`

Inspired by these findings, Fukushima's Neocognitron :cite:`Fukushima.1980`
organized local detectors into hierarchical layers repeated across space, and
LeCun and collaborators showed how to train such architectures end to end via
backpropagation to recognize handwritten ZIP codes
:cite:`LeCun.Boser.Denker.ea.1989,LeCun.Jackel.Bottou.ea.1995`. Deeper layers of
trained vision networks even exhibit representational parallels with higher
stages of the primate visual pathway :cite:`Kuzovkin.Vicente.Petton.ea.2018`—though
biology served as architectural inspiration rather than a blueprint for silicon
computation.

At the heart of these architectures lies the *convolutional layer*, which slides
a small shared kernel of weights across every neighborhood of an image. Each
output unit depends only on nearby pixels, and shifting the input simply shifts
the output—building locality and translation equivariance directly into the
network while slashing parameter counts by orders of magnitude. These
*convolutional neural networks* (CNNs) drove the watershed breakthrough on
ImageNet :cite:`Deng.Dong.Socher.ea.2009,Krizhevsky.Sutskever.Hinton.2012` and
remain a cornerstone of computer vision alongside vision transformers
(:numref:`chap_transformers`).

In this chapter, we derive and build CNNs from first principles:

* :numref:`sec_why-conv` proves that enforcing locality and translation
  equivariance on a fully connected layer leaves essentially no linear choice
  but convolution.
* :numref:`sec_conv_layer` implements two-dimensional cross-correlation,
  trains a single kernel to discover an edge detector from data, and introduces
  *im2col* matrix multiplication and receptive fields.
* :numref:`sec_padding` introduces padding, stride, and dilation to control
  output resolution and expand receptive fields exponentially with depth.
* :numref:`sec_channels` extends convolutions to multiple input and output
  channels, accounting for parameter, FLOP, and activation-memory costs and
  introducing $1 \times 1$ bottleneck convolutions.
* :numref:`sec_pooling` introduces max-pooling, average pooling, and global
  pooling to downsample feature maps and build tolerance to small spatial shifts.
* :numref:`sec_lenet` assembles a complete CNN—LeNet—pits it against our MLP on
  Fashion-MNIST, visualizes its learned filters, and runs the pixel-shuffling
  experiment to isolate the value of spatial inductive bias.

Modern deep convolutional architectures (AlexNet, VGG, NiN, GoogLeNet, BatchNorm,
ResNet, DenseNet, and ConvNeXt) follow in :numref:`chap_modern_cnn`.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```

## Resources and Further Reading {.unnumbered}

The resources below accompany the sections of this chapter; each names the
section it goes with. All are freely accessible online except where noted.

**Books**

- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 9 develops sparse interactions, parameter sharing, and equivariance in detail (:numref:`sec_why-conv`, :numref:`sec_pooling`).
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF; Chapter 10 covers convolutions, stride, dilation, and receptive fields with unusually good figures (:numref:`sec_conv_layer`, :numref:`sec_padding`).
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap6.html) — free online; Chapter 6 introduces convolutional layers, shared weights, and pooling from first principles (:numref:`sec_conv_layer`, :numref:`sec_pooling`).

**Courses and video lectures**

- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/convolutional-networks/) — free notes; the "Convolutional Networks" module (layer arithmetic, parameter counting, layer patterns) is the most widely used online companion to this chapter (:numref:`sec_padding`, :numref:`sec_channels`).
- [Michigan EECS 498-007: Deep Learning for Computer Vision — Justin Johnson](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) — free lecture videos; the convolution and pooling lectures work through kernel/stride/padding arithmetic on the board at exactly this chapter's pace (:numref:`sec_conv_layer`, :numref:`sec_pooling`).

**Tutorials, notes, and interactive**

- [A guide to convolution arithmetic for deep learning — Dumoulin & Visin (2016)](https://arxiv.org/abs/1603.07285) — free, with the famous [animations](https://github.com/vdumoulin/conv_arithmetic); the definitive visual reference for padding, stride, dilation, and transposed convolutions (:numref:`sec_padding`).
- [CNN Explainer — Wang et al. (Polo Club)](https://poloclub.github.io/cnn-explainer/) — free, zero-install; an interactive convnet running in the browser where you can inspect every activation, kernel, and receptive field (:numref:`sec_channels`, :numref:`sec_lenet`).
- [Computing Receptive Fields of Convolutional Neural Networks — Araujo, Norris & Sim (2019), *Distill*](https://distill.pub/2019/computing-receptive-fields/) — free; derives the closed-form receptive-field arithmetic that :numref:`sec_conv_layer` introduces, including strided and multi-path cases.
- [Conv Nets: A Modular Perspective — Chris Olah](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) — free; a short classic building the "convolution as structured weight sharing" intuition (:numref:`sec_why-conv`).
- [Image Kernels Explained Visually — Victor Powell](https://setosa.io/ev/image-kernels/) — free, interactive; slide hand-designed kernels over an image, a perfect warm-up for the edge-detection example in :numref:`sec_conv_layer`.
- [Feature Visualization — Olah, Mordvintsev & Schubert (2017), *Distill*](https://distill.pub/2017/feature-visualization/) — free; what convnet units actually respond to, a companion to the activation-visualization exercise in :numref:`sec_lenet`.

**Foundational papers**

- [Gradient-Based Learning Applied to Document Recognition — LeCun, Bottou, Bengio & Haffner (1998), *Proc. IEEE*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) — free PDF; the LeNet-5 paper behind :numref:`sec_lenet`, still worth reading for its systems-level completeness.
- [Backpropagation Applied to Handwritten Zip Code Recognition — LeCun et al. (1989), *Neural Computation*](https://doi.org/10.1162/neco.1989.1.4.541) — the first convnet trained end-to-end with backprop (paywalled, noted; widely reproduced online) (:numref:`sec_lenet`).
- [Neocognitron — Fukushima (1980), *Biological Cybernetics*](https://doi.org/10.1007/BF00344251) — the pre-backprop ancestor of alternating convolution/pooling stages (paywalled, noted) (:numref:`sec_pooling`).
- [Receptive fields, binocular interaction and functional architecture in the cat's visual cortex — Hubel & Wiesel (1962), *J. Physiology*](https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/) — free (PMC); the biological root of local receptive fields and hierarchical feature detection (:numref:`sec_why-conv`).

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §6]{.kicker}

Convolutional Neural Networks: Geometry-Aware Vision Layers<br>**locality · translation equivariance · cross-correlation · padding · channels · pooling · LeNet**
:::
:::

::: {.slide title="Flattening Images Destroys Spatial Geometry"}
[Motivation & Inductive Bias]{.kicker}

Standard multilayer perceptrons flatten every 2D image grid into an unordered 1D vector, discarding which pixels sit next to one another:

- **Permutation blindness:** Applying one fixed random permutation to all 784 pixels of Fashion-MNIST leaves an MLP's accuracy unchanged ($\approx 87\%$), even though the scrambled images look like pure static to human eyes.
- **Parameter explosion at scale:** Mapping a 1-megapixel ($1000 \times 1000$) RGB image to a full-resolution 3-channel hidden representation requires $(3 \times 10^6)^2 = 9 \times 10^{12}$ weights (**36 TB** in `float32`; 3.6 TB even for a 1000-pixel grayscale map).
- **Biological and geometric clue:** Visual features (edges, corners, textures) are formed by *neighboring* pixels and carry the same meaning regardless of where they appear in the frame.

::: {.d2l-note}
Building **locality** and **translation equivariance** directly into the linear layer collapses trillions of free weights down to a small shared kernel of a few hundred parameters.
:::
:::

::: {.slide title="Receptive Fields in Biological and Silicon Vision"}
[From Visual Cortex to ConvNets]{.kicker}

Hubel and Wiesel discovered that early visual cortex neurons respond to oriented edges within localized *receptive fields*, inspiring hierarchical filter banks swept across space:

![](../img/field-visual.png){.nostretch width=76% fig-align="center"}

- **Local filter bank (left $\to$ right):** Six oriented edge filters convolved with a natural image produce sparse activation maps that highlight local contours at different orientations and scales.
- **End-to-end learning:** Rather than hand-crafting filter banks, convolutional neural networks (CNNs) learn the kernel weights across all layers simultaneously via backpropagation.
:::

::: {.slide title="Mathematical and Architectural Roadmap of Chapter 6"}
[Chapter Architecture]{.kicker}

Each section introduces one structural mechanism that transforms a dense layer into a complete convolutional classifier:

| Section | Core Mechanism | Mathematical Form / Invariant |
|---|---|---|
| **§6.1 Why Convolutions** | Translation equivariance & locality | $f(T_v \mathbf{X}) = T_v f(\mathbf{X}) \implies [\mathsf{V}]_{i,j,a,b} = [\mathbf{K}]_{a,b}$ |
| **§6.2 Conv Layers** | 2D cross-correlation & *im2col* | $[\mathbf{Y}]_{i,j} = \sum_{a,b} [\mathbf{K}]_{a,b}[\mathbf{X}]_{i+a,j+b}$, RF $r = 1 + \sum (k_l - 1)$ |
| **§6.3 Padding & Stride** | Boundary control, downsampling, dilation | Output size $\lfloor (n + 2p - d(k-1) - 1)/s \rfloor + 1$ |
| **§6.4 Channels** | Multi-channel tensors & $1 \times 1$ bottlenecks | $c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$ params (independent of $H \times W$) |
| **§6.5 Pooling** | Local shift tolerance & global invariance | Max/mean reduction over $k \times k$ windows (0 params) |
| **§6.6 LeNet** | Full CNN encoder + dense head | $61{,}706$ params, $416{,}520$ MACs, pixel-shuffle test |
:::

::: {.slide title="Concrete Benchmarks Across Chapter 6"}
[Quantitative Anchors]{.kicker}

Every section grounds its math and systems trade-offs in exact parameter, FLOP, and shape calculations:

| Section | Concrete Benchmark & Systems Takeaway |
|---|---|
| **§6.1 Why Convolutions** | A $3 \times 3$ layer mapping RGB ($c_\textrm{i}=3$) to $c_\textrm{o}=64$ channels uses $1{,}792$ parameters at any resolution |
| **§6.2 Conv Layers** | Gradient descent recovers the vertical edge kernel $[1, -1]$ from input–output pairs in 9 steps |
| **§6.3 Padding & Stride** | Three $3 \times 3$ layers with strides $(2,2,1)$ or dilations $(1,2,4)$ both achieve a 15-pixel receptive field |
| **§6.4 Channels** | A $1 \times 1 \to 3 \times 3 \to 1 \times 1$ bottleneck ($256 \to 64 \to 256$) cuts weights $8.5\times$ ($589{,}824 \to 69{,}632$) |
| **§6.5 Pooling** | $2 \times 2$ stride-2 pooling absorbs 50% of 1-pixel shifts; max-pooling pads borders with $-\infty$ |
| **§6.6 LeNet** | Convolutions hold 4% of LeNet's weights but do 86% of its MACs; pixel shuffling erases its gain over MLP |
:::

::: {.slide title="Concept Check & Chapter Study Strategy"}
[Check & Synthesis]{.kicker}

Before diving into §6.1, test the core scaling intuition separating dense layers from convolutions:

1. **Resolution scaling:** If you double an input image's height and width from $224 \times 224$ to $448 \times 448$, how do the parameter count and the multiply-add (MAC) count of a $3 \times 3$ convolutional layer change?
2. **Inductive bias test:** Why does a CNN beat an MLP on natural images, yet lose that advantage when all images undergo the same random pixel permutation?

::: {.d2l-note .fragment .fade-up}
**Answers:** (1) The parameter count is **completely unchanged** ($c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$ depends only on kernel and channel dimensions), whereas the MAC count and activation memory **quadruple** ($4\times$ as many output spatial positions). (2) A CNN's shared local kernels assume nearby pixels form meaningful patterns; permuting pixels destroys spatial locality while leaving an MLP's dense weight matrix statistically unchanged.
:::
:::
