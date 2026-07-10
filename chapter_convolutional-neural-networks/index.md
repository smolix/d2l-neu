# Convolutional Neural Networks
:label:`chap_cnn`

Image data is represented as a two-dimensional grid of pixels, be the image
monochromatic or in color. Accordingly each pixel corresponds to one
or multiple numerical values respectively. So far we have ignored this rich
structure and treated images as vectors of numbers by *flattening* them, irrespective of the spatial relation between pixels. This
deeply unsatisfying approach was necessary in order to feed the
resulting one-dimensional vectors through a fully connected MLP.

Because these networks are invariant to the order of the features, we
could get similar results regardless of whether we preserve an order
corresponding to the spatial structure of the pixels or if we permute
the columns of our design matrix before fitting the MLP's parameters.
Ideally, we would leverage our prior knowledge that nearby pixels
are typically related to each other, to build efficient models for
learning from image data.

This chapter introduces *convolutional neural networks* (CNNs)
:cite:`LeCun.Jackel.Bottou.ea.1995`, a powerful family of neural networks that
are designed for precisely this purpose.
On the ImageNet collection
:cite:`Deng.Dong.Socher.ea.2009` it was the use of convolutional neural
networks, in short CNNs, that provided significant performance
improvements :cite:`Krizhevsky.Sutskever.Hinton.2012`, and CNN-based
architectures dominated computer vision from roughly 2012 to 2021.
Today they share the field with vision transformers
(:numref:`chap_attention-and-transformers`) and remain the default
where latency, small datasets, or dense prediction dominate.

Modern CNNs, as they are called colloquially, owe their design to
inspirations from biology, group theory, and a healthy dose of
experimental tinkering.  In addition to their sample efficiency in
achieving accurate models, CNNs tend to be computationally efficient,
both because they require fewer parameters than fully connected
architectures and because convolutions are easy to parallelize across
GPU cores :cite:`Chetlur.Woolley.Vandermersch.ea.2014`.  Consequently, practitioners often
apply CNNs whenever possible, and increasingly they have emerged as
credible competitors even on tasks with a one-dimensional sequence
structure, such as audio :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, text
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014`, and time series analysis
:cite:`LeCun.Bengio.ea.1995`, where recurrent neural networks are
conventionally used.  Some clever adaptations of CNNs have also
brought them to bear on graph-structured data :cite:`Kipf.Welling.2016` and
in recommender systems.

First, we will dive more deeply into the motivation for convolutional
neural networks. This is followed by a walk through the basic operations
that comprise the backbone of all convolutional networks.
These include the convolutional layers themselves,
nitty-gritty details including padding, stride, and dilation,
the pooling layers used to aggregate information
across adjacent spatial regions,
the use of multiple channels at each layer,
including grouped and depthwise-separable convolutions,
and a careful discussion of the structure of modern architectures.
We will conclude the chapter with a full working example of LeNet,
the first convolutional network successfully deployed,
long before the rise of modern deep learning.
In the next chapter, we will dive into full implementations
of some popular and comparatively recent CNN architectures
whose designs represent most of the techniques
commonly used by modern practitioners.

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

The references below develop the fundamentals of this chapter: why
convolutions, the mechanics of the operation (padding, stride, dilation,
channels), receptive fields, pooling, and the first convolutional networks.
All are freely accessible online except where noted.

**Books**

- [Deep Learning — Goodfellow, Bengio & Courville](https://www.deeplearningbook.org/) — free HTML; Chapter 9 (Convolutional Networks) is the canonical treatment of sparse interactions, parameter sharing, and equivariance — the same three-step argument as :numref:`sec_why-conv`, developed at length.
- [Understanding Deep Learning — Simon J. D. Prince](https://udlbook.github.io/udlbook/) — free PDF; Chapter 10 covers convolutions, stride, dilation, and receptive fields with unusually good figures.
- [Neural Networks and Deep Learning — Michael Nielsen](http://neuralnetworksanddeeplearning.com/chap6.html) — free online; Chapter 6 introduces convolutional layers, shared weights, and pooling from first principles, a gentle second telling of this chapter.

**Courses and video lectures**

- [Stanford CS231n: Deep Learning for Computer Vision](https://cs231n.github.io/convolutional-networks/) — free notes; the "Convolutional Networks" module (layer arithmetic, parameter counting, layer patterns) is the most widely used online companion to this chapter.
- [Michigan EECS 498-007: Deep Learning for Computer Vision — Justin Johnson](https://web.eecs.umich.edu/~justincj/teaching/eecs498/) — free lecture videos; the convolution and pooling lectures work through kernel/stride/padding arithmetic on the board at exactly this chapter's pace.

**Tutorials, notes, and interactive**

- [A guide to convolution arithmetic for deep learning — Dumoulin & Visin (2016)](https://arxiv.org/abs/1603.07285) — free, with the famous [animations](https://github.com/vdumoulin/conv_arithmetic); the definitive visual reference for padding, stride, dilation, and transposed convolutions.
- [CNN Explainer — Wang et al. (Polo Club)](https://poloclub.github.io/cnn-explainer/) — free, zero-install; an interactive convnet running in the browser where you can inspect every activation, kernel, and receptive field.
- [Computing Receptive Fields of Convolutional Neural Networks — Araujo, Norris & Sim (2019), *Distill*](https://distill.pub/2019/computing-receptive-fields/) — free; derives the closed-form receptive-field arithmetic that :numref:`sec_conv_layer` introduces, including strided and multi-path cases.
- [Conv Nets: A Modular Perspective — Chris Olah](https://colah.github.io/posts/2014-07-Conv-Nets-Modular/) — free; a short classic building the "convolution as structured weight sharing" intuition.
- [Image Kernels Explained Visually — Victor Powell](https://setosa.io/ev/image-kernels/) — free, interactive; slide hand-designed kernels over an image, a perfect warm-up for the edge-detection example in :numref:`sec_conv_layer`.
- [Feature Visualization — Olah, Mordvintsev & Schubert (2017), *Distill*](https://distill.pub/2017/feature-visualization/) — free; what convnet units actually respond to, a companion to the activation-visualization exercise in :numref:`sec_lenet`.

**Foundational papers**

- [Gradient-Based Learning Applied to Document Recognition — LeCun, Bottou, Bengio & Haffner (1998), *Proc. IEEE*](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf) — free PDF; the LeNet-5 paper behind :numref:`sec_lenet`, still worth reading for its systems-level completeness.
- [Backpropagation Applied to Handwritten Zip Code Recognition — LeCun et al. (1989), *Neural Computation*](https://doi.org/10.1162/neco.1989.1.4.541) — the first convnet trained end-to-end with backprop (paywalled, noted; widely reproduced online).
- [Neocognitron — Fukushima (1980), *Biological Cybernetics*](https://doi.org/10.1007/BF00344251) — the pre-backprop ancestor of alternating convolution/pooling stages (paywalled, noted).
- [Receptive fields, binocular interaction and functional architecture in the cat's visual cortex — Hubel & Wiesel (1962), *J. Physiology*](https://pmc.ncbi.nlm.nih.gov/articles/PMC1359523/) — free (PMC); the biological root of local receptive fields and hierarchical feature detection.

