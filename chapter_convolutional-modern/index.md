# Modern Convnets
:label:`chap_modern_cnn`

The LeNet of :numref:`sec_lenet` has five layers and about 62,000
parameters. Stacking more of its layers does not give a better network.
Sigmoid units shrink the gradient at every layer, a deeper plain network
ends up with a *higher* training error than a shallower one, and every new
layer needs its own width, kernel size, and initialization chosen by hand.
Between 2012 and 2016 a series of ideas removed these obstacles one at a
time, and the networks that won ImageNet grew from 8 layers to more than
150. Later progress came as much from how networks are trained, and from
studying whole families of networks, as from new layers. Telling these
sources apart matters whenever we compare two published results or design a
network for a new problem.

:numref:`sec_alexnet` asks what let a deep network trained on raw pixels
beat engineered features in 2012. :numref:`sec_blocks` asks how to build
deep networks from repeated blocks instead of layer by layer, and what
$1 \times 1$ convolutions, global pooling, and parallel branches add.
:numref:`sec_batch_norm` asks how normalizing activations makes training
faster and less sensitive to the learning rate. :numref:`sec_resnet` asks
why deeper plain networks train worse and how a residual connection fixes
it. :numref:`sec_efficient_cnns` asks how to cut the arithmetic of a
convolutional layer by an order of magnitude, and what that costs in
accuracy. :numref:`sec_training_recipes` asks how much of a reported
accuracy comes from the training procedure rather than the architecture.
:numref:`sec_convnext` asks how far a ResNet gets when it is modernized one
change at a time. :numref:`sec_cnn-design` asks what rules appear when we
study populations of networks instead of single designs.

Every network in the chapter is trained on Fashion-MNIST in all four
frameworks, with the same model and training procedure in every framework;
ImageNet results are quoted from the papers. Vision transformers
:cite:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` now lead many large-scale
benchmarks and follow in :numref:`chap_transformers`. Convnets remain the
default where data are limited, latency matters, or every pixel needs a
prediction, and with matched compute they match transformers on image
classification :cite:`smith2023convnets`.

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

The references below trace the architectural developments covered in this
chapter and current practice: landmark networks from AlexNet to
ConvNeXt, the training recipes that confound naive comparisons between
them, and efficient networks for deployment. All are freely accessible
online except where noted.

**Books**

- [Deep Learning for Coders with fastai and PyTorch — Howard & Gugger](https://github.com/fastai/fastbook) — free notebooks; builds ResNets from scratch and applies modern training techniques (augmentation, schedules, mixed precision) of :numref:`sec_training_recipes` in working code.
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF; Chapter 11 (Residual networks) analyzes *why* residual connections ease optimization — loss-surface and gradient-propagation arguments complementing :numref:`sec_resnet`.

**Courses and video lectures**

- [Michigan EECS 498-007, Lecture: CNN Architectures — Justin Johnson](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) — free videos; walks AlexNet → VGG → GoogLeNet → ResNet with parameter/FLOP accounting, the same tour as this chapter's first half.
- [MIT 6.5940: TinyML and Efficient Deep Learning — Song Han](https://efficientml.ai/) — free lectures; depthwise separability, quantization, pruning, and edge deployment — the systems side of :numref:`sec_efficient_cnns`.
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course) — free; modern practice with pretrained backbones (including ConvNeXt) and transfer learning, the "what you actually do in 2026" companion to this chapter.

**Tutorials, notes, and surveys**

- [timm (pytorch-image-models) — Ross Wightman et al.](https://huggingface.co/docs/timm) — free; an implementation collection containing the architectures in this chapter, with trained weights and [results tables](https://github.com/huggingface/pytorch-image-models/tree/main/results) comparing them under consistent evaluation.
- [A Recipe for Training Neural Networks — Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/) — free; the debugging-first training discipline that :numref:`sec_training_recipes` systematizes.
- [Zoom In: An Introduction to Circuits — Olah et al. (2020), *Distill*](https://distill.pub/2020/circuits/zoom-in/) — free; opens trained vision models to inspect the features and circuits they learn, useful perspective once you can train the architectures in this chapter.

**Foundational papers**

All free on arXiv or the proceedings site; these are the primary sources this
chapter retells, worth reading in the original:

- [ImageNet Classification with Deep Convolutional Neural Networks — Krizhevsky, Sutskever & Hinton (2012), *NeurIPS*](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — AlexNet (:numref:`sec_alexnet`).
- [Batch Normalization — Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167) — read together with its critique in :numref:`sec_batch_norm`.
- [Deep Residual Learning for Image Recognition — He, Zhang, Ren & Sun (2015)](https://arxiv.org/abs/1512.03385) — ResNet (:numref:`sec_resnet`), the most-cited paper in deep learning.
- [Bag of Tricks for Image Classification with CNNs — He et al. (2019)](https://arxiv.org/abs/1812.01187) — the first systematic demonstration that recipe details rival architecture changes (:numref:`sec_training_recipes`).
- [ResNet Strikes Back — Wightman, Touvron & Jégou (2021)](https://arxiv.org/abs/2110.00476) — a controlled analysis of training recipes and architecture behind :numref:`sec_training_recipes`.
- [MobileOne: An Improved One Millisecond Mobile Backbone — Vasu et al. (2023), *CVPR*](https://arxiv.org/abs/2206.04040) — re-parameterized networks designed against measured phone latency; continues :numref:`sec_efficient_cnns` toward deployment.
- [A ConvNet for the 2020s — Liu et al. (2022)](https://arxiv.org/abs/2201.03545) — ConvNeXt (:numref:`sec_convnext`), a controlled ablation worth studying as method, not just result.
- [ConvNets Match Vision Transformers at Scale — Smith et al. (2023)](https://arxiv.org/abs/2310.16764) — with matched pretraining compute, convnets and vision transformers scale alike; read with :numref:`chap_transformers`.

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
- Sigmoid units shrink the gradient at every layer; a deeper *plain*
  network ends with a higher training error than a shallower one.
- Every new layer needs its own width, kernel size, and initialization.
- Reported accuracies mix architecture with the training procedure.

::: {.d2l-note}
Between 2012 and 2016 ImageNet winners grew from 8 layers to more than
150. Which ideas made that possible, and which gains came from training?
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

**Running example:** every network trained on Fashion-MNIST, with the same
model and procedure in all four frameworks.
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
| AlexNet | 46.8M parameters; about 90% in a third of LeNet's epochs |
| VGG, NiN, GoogLeNet | 81–90% with parameter counts from 2M to 43M |
| Normalization Layers | LeNet + batch norm about 88% where plain LeNet stalls |
| ResNet, ResNeXt, DenseNet | ResNet-18, 11.2M parameters, 90–91% |
| Efficient Convnets | Mini-MobileNet: 7.7× fewer multiply-adds, about 2 points lower |
| Training Recipes | the modern recipe wins every paired run, by 0.5–1.5 points |
| ConvNeXt | 2 points below a matched ResNet-18 at this scale |
| Design Spaces (RegNet) | within a point of ResNet-18 with 1% of its parameters |
:::
