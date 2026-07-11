# Modern Convolutional Neural Networks
:label:`chap_modern_cnn`

Now that we understand how to wire up convolutional networks, let's tour
the architectures that made them the dominant tool of computer vision,
and the practices that keep them competitive today. First, the honest
picture. Since roughly 2021, vision transformers have led large-scale
image classification benchmarks, beginning with
:citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` and the Swin Transformer
:cite:`liu2021swin`; we cover them in
:numref:`chap_attention-and-transformers`. Convnets remain the workhorse
wherever latency budgets are tight, training data is scarce, or the
prediction is dense, as in segmentation
:cite:`Long.Shelhamer.Darrell.2015` and object detection
:cite:`Redmon.Farhadi.2018`. When trained with modern recipes at matched
compute, they still match transformers on image classification
:cite:`smith2023convnets`. This chapter brings you to that state of the
art as of 2026.

The chapter tells the story in two eras. The first is the architecture
race of 2012--2015, set off when AlexNet
:cite:`Krizhevsky.Sutskever.Hinton.2012` won the 2012
[ImageNet competition](https://www.image-net.org/challenges/LSVRC/) by a
wide margin (:numref:`sec_alexnet`). The years that followed were about
*organizing* convolutions: repeated blocks in VGG
:cite:`Simonyan.Zisserman.2014`, $1 \times 1$ channel mixing and global
pooling in NiN :cite:`Lin.Chen.Yan.2013`, and multi-branch design in
GoogLeNet :cite:`Szegedy.Liu.Jia.ea.2015`, all covered in
:numref:`sec_blocks`. Batch normalization :cite:`Ioffe.Szegedy.2015`
made deep networks train reliably (:numref:`sec_batch_norm`), and
residual connections :cite:`He.Zhang.Ren.ea.2016` removed the
optimization barrier to depth (:numref:`sec_resnet`), with ResNeXt
:cite:`Xie.Girshick.Dollar.ea.2017` and DenseNet
:cite:`Huang.Liu.Van-Der-Maaten.ea.2017` as the surviving variations.
The race produced more than vision models: normalization layers and
residual connections escaped computer vision entirely and now sit
inside nearly every deep network, transformers included.

The second era, from 2016 to today, is one of maturation. Network
topology stopped changing quickly and progress moved elsewhere.
Deployment under latency and memory constraints produced a line of designs
from depthwise-separable MobileNets :cite:`howard2017mobilenet` to structural
re-parameterization (:numref:`sec_efficient_cnns`). Training recipes improved
so much that an unmodified ResNet-50 gains
over four points of ImageNet accuracy from the recipe alone
:cite:`wightman2021resnet`; :numref:`sec_training_recipes` teaches the
modern recipe, without which comparisons between architectures mislead.
Building on that recipe, ConvNeXt :cite:`liu2022convnet` modernized a
ResNet step by step, using only ideas from this book, into a network
that matches a vision transformer of equal cost
(:numref:`sec_convnext`). Finally,
network design itself became an empirical science: rather than crafting
a single network, the RegNet methodology
:cite:`Radosavovic.Kosaraju.Girshick.ea.2020` explores whole *design
spaces*, and :numref:`sec_cnn-design` uses it to close the chapter with
the big picture of where convnets and transformers each stand.

We present the architectures in roughly chronological order, partly to
convey a sense of the history so that you can form your own intuitions
about where the field is heading, and partly because each design
responds to a concrete failure of its predecessors. The networks in
this chapter are the product of intuition, a few mathematical insights,
and a lot of trial and error; knowing *why* each idea was introduced is
the best guide to when to reach for it.

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

The references below trace the architecture race this chapter narrates
and the modern practice it lands on: landmark networks from AlexNet to
ConvNeXt, the training recipes that confound naive comparisons between
them, and efficient networks for deployment. All are freely accessible
online except where noted.

**Books**

- [Deep Learning for Coders with fastai and PyTorch — Howard & Gugger](https://github.com/fastai/fastbook) — free notebooks; builds ResNets from scratch and applies the modern training tricks (augmentation, schedules, mixed precision) of :numref:`sec_training_recipes` in working code.
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF; Chapter 11 (Residual networks) analyzes *why* residual connections ease optimization — loss-surface and gradient-propagation arguments complementing :numref:`sec_resnet`.

**Courses and video lectures**

- [Michigan EECS 498-007, Lecture: CNN Architectures — Justin Johnson](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) — free videos; walks AlexNet → VGG → GoogLeNet → ResNet with parameter/FLOP accounting, the same tour as this chapter's first half.
- [MIT 6.5940: TinyML and Efficient Deep Learning — Song Han](https://efficientml.ai/) — free lectures; depthwise separability, quantization, pruning, and edge deployment — the systems side of :numref:`sec_efficient_cnns`.
- [Hugging Face Computer Vision Course](https://huggingface.co/learn/computer-vision-course) — free; modern practice with pretrained backbones (including ConvNeXt) and transfer learning, the "what you actually do in 2026" companion to this chapter.

**Tutorials, notes, and surveys**

- [timm (pytorch-image-models) — Ross Wightman et al.](https://huggingface.co/docs/timm) — free; the reference implementation zoo where every architecture in this chapter lives, with trained weights and [results tables](https://github.com/huggingface/pytorch-image-models/tree/main/results) comparing them under consistent evaluation.
- [A Recipe for Training Neural Networks — Andrej Karpathy](https://karpathy.github.io/2019/04/25/recipe/) — free; the debugging-first training discipline that :numref:`sec_training_recipes` systematizes.
- [Zoom In: An Introduction to Circuits — Olah et al. (2020), *Distill*](https://distill.pub/2020/circuits/zoom-in/) — free; opens trained vision models to inspect the features and circuits they learn, useful perspective once you can train the architectures in this chapter.

**Foundational papers**

All free on arXiv or the proceedings site; these are the primary sources this
chapter retells, worth reading in the original:

- [ImageNet Classification with Deep Convolutional Neural Networks — Krizhevsky, Sutskever & Hinton (2012), *NeurIPS*](https://papers.nips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) — AlexNet (:numref:`sec_alexnet`).
- [Batch Normalization — Ioffe & Szegedy (2015)](https://arxiv.org/abs/1502.03167) — read together with its critique in :numref:`sec_batch_norm`.
- [Deep Residual Learning for Image Recognition — He, Zhang, Ren & Sun (2015)](https://arxiv.org/abs/1512.03385) — ResNet (:numref:`sec_resnet`), the most-cited paper in deep learning.
- [Bag of Tricks for Image Classification with CNNs — He et al. (2019)](https://arxiv.org/abs/1812.01187) — the first systematic demonstration that recipe details rival architecture changes.
- [ResNet Strikes Back — Wightman, Touvron & Jégou (2021)](https://arxiv.org/abs/2110.00476) — the definitive recipe-vs-architecture accounting behind :numref:`sec_training_recipes`.
- [A ConvNet for the 2020s — Liu et al. (2022)](https://arxiv.org/abs/2201.03545) — ConvNeXt (:numref:`sec_convnext`), a controlled ablation worth studying as method, not just result.
- [ConvNets Match Vision Transformers at Scale — Smith et al. (2023)](https://arxiv.org/abs/2310.16764) — the scaling-law resolution of the convnet-vs-transformer debate closing this chapter.
