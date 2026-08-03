# Modern Convnets
:label:`chap_modern_cnn`

This chapter studies the architectures and training practices that established
convolutional networks in computer vision. Since roughly 2021, vision
transformers have led many large-scale image classification benchmarks,
beginning with :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` and the Swin
Transformer :cite:`liu2021swin`; we cover them in
:numref:`chap_transformers`. Convnets remain important where latency is
constrained, training data is limited, or the prediction is dense, as in
segmentation :cite:`Long.Shelhamer.Darrell.2015` and object detection
:cite:`Redmon.Farhadi.2018`. With modern training procedures and matched
compute, they can also match transformers on image classification
:cite:`smith2023convnets`.

The first group of sections covers the architectural developments of
2012--2015. AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012` established
deep convolutional networks on ImageNet (:numref:`sec_alexnet`). VGG, NiN,
and GoogLeNet then introduced repeated blocks, $1 \times 1$ channel mixing,
global pooling, and multi-branch design (:numref:`sec_blocks`). Batch
normalization :cite:`Ioffe.Szegedy.2015` improved optimization
(:numref:`sec_batch_norm`), while residual connections
:cite:`He.Zhang.Ren.ea.2016` made substantially deeper networks practical
(:numref:`sec_resnet`). Normalization and residual connections subsequently
became standard components outside computer vision, including in transformers.

The later sections separate architectural choices from training and deployment
constraints. Efficient networks use depthwise convolution and structural
re-parameterization (:numref:`sec_efficient_cnns`). Modern training procedures
raise the accuracy of an unchanged ResNet-50 by more than four percentage
points :cite:`wightman2021resnet` (:numref:`sec_training_recipes`). ConvNeXt
applies these practices in a controlled modernization of ResNet
:cite:`liu2022convnet` (:numref:`sec_convnext`). Finally, RegNet studies
distributions over network designs rather than selecting a single architecture
:cite:`Radosavovic.Kosaraju.Girshick.ea.2020`
(:numref:`sec_cnn-design`).

The sections are arranged roughly chronologically because each design responds
to a limitation of its predecessors. This also makes it possible to distinguish
improvements due to architecture from those due to optimization, data
augmentation, and computational budget.

```toc
:maxdepth: 2

alexnet
blocks
batch-norm
resnet
training-recipes
convnext
efficient-convnets
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
- [Bag of Tricks for Image Classification with CNNs — He et al. (2019)](https://arxiv.org/abs/1812.01187) — the first systematic demonstration that recipe details rival architecture changes.
- [ResNet Strikes Back — Wightman, Touvron & Jégou (2021)](https://arxiv.org/abs/2110.00476) — a controlled analysis of training recipes and architecture behind :numref:`sec_training_recipes`.
- [A ConvNet for the 2020s — Liu et al. (2022)](https://arxiv.org/abs/2201.03545) — ConvNeXt (:numref:`sec_convnext`), a controlled ablation worth studying as method, not just result.
- [ConvNets Match Vision Transformers at Scale — Smith et al. (2023)](https://arxiv.org/abs/2310.16764) — the scaling-law resolution of the convnet-vs-transformer debate closing this chapter.
