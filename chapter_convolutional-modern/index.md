# Modern Convnets
:label:`chap_modern_cnn`

The LeNet of :numref:`sec_lenet` learns its convolution kernels from
examples, and it classifies Fashion-MNIST slightly better than the MLP with
fewer than a third of its parameters. A natural next step is to make it
larger, with more layers and more channels. For a long time this did not
work. Deeper stacks of sigmoid layers trained worse than shallow ones, since
every sigmoid shrinks the gradient that passes back through it, and every
new layer brought its own kernel size, width, and initialization to choose
by hand. What does it take to make a convolutional network much
deeper, and better?

The network that won the ImageNet competition in 2012 had 8 layers, and the
winner three years later had more than 150. ImageNet is a benchmark of more
than a million photographs in a thousand categories, and the 2012 winner, a
convolutional network trained end to end on raw pixels, beat every system
built on hand-designed features. Between the two networks lies a series of
ideas that removed one obstacle at a time: repeated blocks, layers that
keep the scale of the signals under control, and connections that let each
block add a correction to its input instead of replacing it. The last two
reappear in every transformer.

Later gains came as much from how networks are trained as from new layers.
This is awkward for anyone who compares two published accuracies, because
the better number may belong to the better training recipe and not to the
better network.

The first four sections follow the growth in depth: the network of 2012
(:numref:`sec_alexnet`), networks built from repeated blocks instead of layer
by layer (:numref:`sec_blocks`), the normalization of activations
(:numref:`sec_batch_norm`), and the residual connection
(:numref:`sec_resnet`), where we train plain and residual networks side by
side to see that depth alone was not enough.
:numref:`sec_efficient_cnns` goes the other way and makes a network cheap
enough for a phone, then asks what the saving is worth against a plain
network of equal cost. From there on the chapter is about evidence as much
as about layers. We ask how much of a reported accuracy comes from the
training procedure (:numref:`sec_training_recipes`), modernize a ResNet a
few changes at a time to find out which of them matter
(:numref:`sec_convnext`), and sample two small populations of networks to
test a design rule (:numref:`sec_cnn-design`).

Every network in the chapter is trained on Fashion-MNIST, under one
learning-rate schedule that :numref:`sec_alexnet` defines. A benchmark this
small shows that a network trains and what it costs. The evidence that an
idea matters at scale comes from the ImageNet results, which we quote from
the papers. Vision transformers :cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021`
follow in :numref:`chap_transformers`. With equal compute, convnets match
them on image classification :cite:`smith2023convnets`, and they remain the
default where data are limited, latency matters, or every pixel needs a
prediction.

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
