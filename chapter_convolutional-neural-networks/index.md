# Convolutional Neural Networks
:label:`chap_cnn`

A 28 × 28 grayscale image has 784 pixels, and the MLP of
:numref:`chap_perceptrons` flattens them into a vector before its first
layer. Flattening throws away which pixels are neighbors: if we shuffled the
pixels of every image in the same way and retrained, the MLP would do exactly
as well, since it has no notion of space. Yet an edge, a stroke, or a texture
is a pattern among nearby pixels, and it means the same thing wherever in the
image it appears. A model that knew this would need far fewer parameters and
far less data. A one-megapixel color image fed to a hidden layer of 1,000
units needs three billion weights; a layer that looks only at small
neighborhoods, with the same weights everywhere, needs a few thousand.

*Convolutional neural networks* (CNNs) :cite:`LeCun.Jackel.Bottou.ea.1995`
build these two facts, locality and translation equivariance, into their
layers. On ImageNet :cite:`Deng.Dong.Socher.ea.2009` they brought the jump in
accuracy :cite:`Krizhevsky.Sutskever.Hinton.2012` that started the current
era of deep learning, and they remain the default where data are scarce,
latency matters, or every pixel needs a prediction; vision transformers
(:numref:`chap_transformers`) now share the field.

:numref:`sec_why-conv` asks which layer follows from locality and
translation equivariance alone. :numref:`sec_conv_layer` asks what that
layer computes, and how it learns an edge detector from data.
:numref:`sec_padding` asks how padding, stride, and dilation control the
size of the output and how far each output sees. :numref:`sec_channels`
asks how a layer combines several input channels into several output
channels. :numref:`sec_pooling` asks how to summarize a neighborhood so that
small shifts do not matter. :numref:`sec_lenet` asks what a complete
convolutional classifier looks like, and how it compares with the MLP on
Fashion-MNIST.

The chapter treats images and the classic building blocks; the deeper
architectures and training methods that made CNNs dominant follow in
:numref:`chap_modern_cnn`.

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

Layers that know an image is a grid<br>
**locality · translation equivariance · convolution · padding · channels · pooling · LeNet**
:::
:::

::: {.slide title="Flattening an Image Throws Away Its Geometry"}
[The problem]{.kicker}

- The MLP flattens a 28 × 28 image to 784 numbers; shuffling the pixels
  the same way in every image would not change how well it learns.
- An edge or a texture is a pattern among *neighboring* pixels, and it
  means the same thing wherever it appears.
- A one-megapixel color image into 1,000 hidden units: three billion
  weights. Local windows with shared weights: a few thousand.

::: {.d2l-note}
Build locality and translation equivariance into the layer, and the
parameter count collapses.
:::
:::

::: {.slide title="By the End of This Chapter You Can"}
[Goals]{.kicker}

- derive convolution from locality and translation equivariance;
- compute a cross-correlation by hand, and learn an edge detector from data;
- predict output sizes and receptive fields under padding, stride, and dilation;
- count the parameters and multiply-adds of a layer with many channels;
- explain what pooling buys, and pad it correctly;
- build LeNet and compare it with the MLP on Fashion-MNIST.

**Running example:** a 28 × 28 Fashion-MNIST image, from one kernel to a
complete classifier.
:::

::: {.slide title="Map of the Chapter"}
[Chapter map]{.kicker}

| section | question | adds |
|---|---|---|
| From Fully Connected Layers to Convolutions | which layer follows from locality and equivariance? | locality and translation equivariance give convolution |
| Convolutions for Images | what does the layer compute, and can it learn a kernel? | cross-correlation, learning a kernel, receptive fields |
| Padding, Stride, and Dilation | how large is the output, and how far does a unit see? | output size and how far each unit sees |
| Multiple Input and Multiple Output Channels | how do channels combine, and at what cost? | channels, 1 × 1 convolutions, cost |
| Pooling | how do we summarize a neighborhood? | downsampling and tolerance to small shifts |
| Convolutional Neural Networks (LeNet) | what does a whole CNN look like? | a complete CNN on Fashion-MNIST |
:::

::: {.slide title="Every Section Works on a Running Example With Real Numbers"}
[Running examples]{.kicker}

| section | running example |
|---|---|
| From Fully Connected Layers to Convolutions | an RGB-to-64-channel 3 × 3 layer has 1,792 parameters |
| Convolutions for Images | a vertical-edge kernel (1, −1) is learned from data in nine steps |
| Padding, Stride, and Dilation | strides (2, 2, 1) and dilations (1, 2, 4) both see 15 pixels |
| Multiple Input and Multiple Output Channels | a 3 × 3 layer with 256 channels in and out: 590,080 parameters |
| Pooling | max pooling pads with −∞, so negative inputs pool correctly |
| Convolutional Neural Networks (LeNet) | 61,706 parameters, about 88–90% against the MLP's 87% |
:::
