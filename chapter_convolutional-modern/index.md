# Modern Convnets
:label:`chap_modern_cnn`

In :numref:`sec_lenet`, LeNet demonstrated that a convolutional network can
learn useful filters directly from raw pixels and edge out an MLP on
Fashion-MNIST with fewer than a third as many parameters. The natural next
question is how to scale that idea up—more layers, wider feature maps, and
harder datasets. For years, simply stacking more layers failed: deeper stacks
of sigmoid units trained *worse* than shallow ones because each sigmoid
attenuates the gradient passing back through it, and hand-picking kernel
sizes, channel counts, and initializations layer by layer quickly became
untenable. What breakthroughs were needed to make deep convolutional networks
both trainable and dramatically more accurate?

In 2012, the network that won the ImageNet competition—a benchmark of over a
million natural photographs across a thousand object categories—was only 8
layers deep, yet training a CNN end to end on raw pixels decisively beat every
system built on hand-crafted visual features. Just three years later, the
winning architecture exceeded 150 layers. Bridging that gap required a
sequence of conceptual leaps that dismantled one barrier at a time: composing
networks from repeatable *blocks* rather than bespoke layers, *normalizing*
intermediate activations so signal scales stay well behaved, and introducing
*residual connections* so each block learns a refinement to its input rather
than replacing it outright. The latter two ideas proved so fundamental that
they now appear inside virtually every modern Transformer.

Equally important, much of the progress over the following decade came not
just from inventing new layers, but from refining *how* networks are trained.
That creates an awkward pitfall when reading the literature: comparing two
published accuracy numbers across papers can easily confuse a stronger
training recipe with a genuinely better architecture.

This chapter unfolds in two halves. The first four sections trace the rise of
deep convnets chronologically: the 2012 breakthrough of AlexNet
(:numref:`sec_alexnet`), modular block-based design in VGG, NiN, and GoogLeNet
(:numref:`sec_blocks`), activation normalization (:numref:`sec_batch_norm`),
and residual architectures—ResNet, ResNeXt, and DenseNet
(:numref:`sec_resnet`), where we train plain and residual networks side by
side to see firsthand why depth alone degrades optimization. Next,
:numref:`sec_efficient_cnns` turns in the opposite direction, shrinking compute
enough to run on mobile hardware and testing how efficient depthwise-separable
designs hold up against a standard convnet of matched arithmetic cost. The
final three sections emphasize empirical methodology as much as layer design:
disentangling training recipes from architectures
(:numref:`sec_training_recipes`), systematically modernizing a ResNet one
change at a time in ConvNeXt (:numref:`sec_convnext`), and sampling
populations of models to uncover principled design spaces in RegNet
(:numref:`sec_cnn-design`).

To make every architecture concrete and reproducible on modest hardware, we
train each network in this chapter on Fashion-MNIST under a shared
learning-rate schedule defined in :numref:`sec_alexnet`. While a dataset of
this size lets us verify trainability and inspect compute and parameter
tradeoffs directly, we quote published ImageNet results from the primary
papers to show how each idea behaves at full scale. Vision Transformers
:cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` enter the picture in
:numref:`chap_transformers`; when given matched pretraining compute, modern
convnets match their classification accuracy :cite:`smith2023convnets`, and
convnets remain a workhorse whenever training data are limited, low latency is
critical, or dense pixel-level predictions are required.

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

## Resources and Further Reading {.unnumbered}

The references below cover the architectures of this chapter and current
practice: the landmark networks from AlexNet to ConvNeXt, the training
recipes that confound comparisons between them, and efficient networks for
deployment. All are freely accessible online.

**Books**

- [Deep Learning for Coders with fastai and PyTorch](https://github.com/fastai/fastbook), by Howard and Gugger. Free notebooks that build ResNets from scratch and apply the augmentation, schedules, and mixed precision of :numref:`sec_training_recipes` in working code.
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/), by Simon J. D. Prince. Chapter 11 analyzes why residual connections ease optimization, with loss-surface and gradient-propagation arguments that complement :numref:`sec_resnet`.

**Courses and video lectures**

- [Michigan EECS 498-007, lecture on CNN architectures](https://web.eecs.umich.edu/~justincj/teaching/eecs498/), by Justin Johnson. Covers the networks of the first four sections, from AlexNet to ResNet, with the same accounting of parameters and arithmetic.
- [MIT 6.5940: TinyML and Efficient Deep Learning](https://efficientml.ai/), by Song Han. Depthwise separability, quantization, pruning, and deployment on small devices: the systems side of :numref:`sec_efficient_cnns`.
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course). Current practice with pretrained backbones, ConvNeXt among them, and transfer learning.

**Tutorials, notes, and surveys**

- [timm (pytorch-image-models)](https://huggingface.co/docs/timm), by Ross Wightman and contributors. Implementations of the architectures in this chapter with trained weights, and [results tables](https://github.com/huggingface/pytorch-image-models/tree/main/results) that compare them under one evaluation.
- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), by Andrej Karpathy. The discipline of debugging first, which :numref:`sec_training_recipes` assumes.
- [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/), by Olah et al. (2020), *Distill*. Opens trained vision models to inspect the features they learn; a natural next step after the first-layer kernels of :numref:`sec_alexnet`.

**Foundational papers**

A selection of the primary sources for this chapter, all free on arXiv or the
proceedings site and worth reading in the original. The papers on VGG, NiN,
GoogLeNet, and RegNet are cited in their sections.

- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), by Krizhevsky, Sutskever, and Hinton (2012), *NeurIPS*. AlexNet (:numref:`sec_alexnet`).
- [Batch Normalization](https://arxiv.org/abs/1502.03167), by Ioffe and Szegedy (2015). Read it together with the later critique of its explanation in :numref:`sec_batch_norm`.
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), by He, Zhang, Ren, and Sun (2015). ResNet (:numref:`sec_resnet`), the most cited paper in deep learning.
- [Bag of Tricks for Image Classification with CNNs](https://arxiv.org/abs/1812.01187), by He et al. (2019). The first systematic demonstration that details of the recipe rival changes of architecture (:numref:`sec_training_recipes`).
- [ResNet Strikes Back](https://arxiv.org/abs/2110.00476), by Wightman, Touvron, and Jégou (2021). The controlled study of recipes and architecture behind :numref:`sec_training_recipes`.
- [MobileOne: An Improved One Millisecond Mobile Backbone](https://arxiv.org/abs/2206.04040), by Vasu et al. (2023), *CVPR*. Re-parameterized networks designed against measured phone latency; continues :numref:`sec_efficient_cnns` toward deployment.
- [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545), by Liu et al. (2022). ConvNeXt (:numref:`sec_convnext`), a cumulative ablation worth studying for its method as much as for its result.
- [ConvNets Match Vision Transformers at Scale](https://arxiv.org/abs/2310.16764), by Smith et al. (2023). With matched pretraining compute, convnets and vision transformers scale alike; read it with :numref:`chap_transformers`.

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
