```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Multiple Input and Multiple Output Channels
:label:`sec_channels`

So far our convolutions have mapped a single 2D grid to a single 2D grid. Real
images and hidden representations have multiple *channels* on both sides: an RGB
photograph is a $3 \times h \times w$ tensor (requiring cross-channel comparisons
to detect a red-on-green boundary of equal luminance), and a hidden layer
extracts dozens or hundreds of distinct feature maps in parallel.

As derived in :numref:`sec_why-conv` (:eqref:`eq_conv-layer-channels`), a layer
with $c_\textrm{i}$ input channels and $c_\textrm{o}$ output channels maintains
one $c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}$ kernel per output
channel, cross-correlating across all input channels and summing the results. In
this section, we implement multi-channel convolutions, account for their
parameter, FLOP, and activation-memory costs (governed by the product
$c_\textrm{i} c_\textrm{o}$), and introduce $1 \times 1$ bottleneck convolutions.
```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Multiple Input Channels

Let us begin with the input side. When the input contains $c_\textrm{i}$
channels, the kernel must supply one $k_\textrm{h}\times k_\textrm{w}$ filter
for each input channel, giving a third-order kernel tensor of shape
$c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$. We cross-correlate
every input channel $\mathbf{X}_c$ with its corresponding kernel slice
$\mathbf{K}_c$ using the two-dimensional cross-correlation :eqref:`eq_corr2d`
(computed by `corr2d`) and sum the $c_\textrm{i}$ intermediate grids:

$$\mathbf{Y} = \sum_{c=1}^{c_\textrm{i}} \mathbf{X}_c \star \mathbf{K}_c.$$

Each entry of the resulting two-dimensional output is therefore a weighted sum
over the same spatial window across all input channels, combining
$c_\textrm{i} k_\textrm{h} k_\textrm{w}$ scalar products.
:numref:`fig_conv_multi_in` illustrates this calculation for two input
channels, where the shaded upper-left output entry equals
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Cross-correlation with two input channels: one cross-correlation per channel, then a sum.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`

Summing across channels is precisely what allows a kernel to compare
information carried by different channels, such as color. Consider a
$1\times 1$ kernel with weight $+1$ on the red channel and $-1$ on the green
channel: at every pixel it computes red minus green, producing positive values
in red regions, negative values in green regions, and zero in neutral gray
regions regardless of brightness. Wherever its sign flips between adjacent
pixels, a red region meets a green one---even if both regions have identical
luminance, in which case a grayscale brightness filter would see no edge at
all.

Implementing multi-input cross-correlation translates the formula directly
into Python: we call `corr2d` once per channel and sum the results.

```{.python .input #channels-multiple-input-channels-1}
def corr2d_multi_in(X, K):
    """Cross-correlate each channel of X with its channel of K and add."""
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

Let us verify the implementation on the input and kernel tensors from
:numref:`fig_conv_multi_in`; the upper-left output entry should be 56.

```{.python .input #channels-multiple-input-channels-2}
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Multiple Output Channels
:label:`subsec_multi-output-channels`

No matter how many input channels it reads, a single multi-input kernel
produces just one two-dimensional *feature map*
(:numref:`subsec_why-conv-channels`). To extract $c_\textrm{o}$ distinct
feature maps simultaneously, a convolutional layer maintains $c_\textrm{o}$
such kernels, each of shape
$c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$. Stacking them along a
new axis yields a fourth-order kernel tensor of shape
$c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$,
where output channel $o$ is given by

$$\mathbf{Y}_o = \sum_{c=1}^{c_\textrm{i}} \mathbf{X}_c \star \mathbf{K}_{o,c}, \quad o = 1, \ldots, c_\textrm{o}.$$

Notice that every output channel connects to *every* input channel through its
own $k_\textrm{h}\times k_\textrm{w}$ slice $\mathbf{K}_{o,c}$; this dense
all-to-all channel wiring is the primary driver of the layer's computational
cost. Written channel by channel, this matches :eqref:`eq_conv-layer-channels`
from :numref:`sec_why-conv`. In typical CNN architectures, the number of
channels grows as spatial resolution shrinks via strides or pooling, trading
fine-grained spatial layout for richer, jointly learned feature descriptions
at each location.

:begin_tab:`mxnet, pytorch`
Gluon and PyTorch store kernels channels-first as $(c_\textrm{o}, c_\textrm{i}, k_\textrm{h}, k_\textrm{w})$.
:end_tab:

:begin_tab:`tensorflow, jax`
Keras and Flax store kernels channels-last as $(k_\textrm{h}, k_\textrm{w}, c_\textrm{i}, c_\textrm{o})$.
:end_tab:
To implement multi-channel output cross-correlation, we iterate over the
output kernels, apply `corr2d_multi_in` to each, and stack the resulting
feature maps along axis 0.

```{.python .input #channels-multiple-output-channels-1}
def corr2d_multi_in_out(X, K):
    """Apply corr2d_multi_in once per output channel of K and stack."""
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

Let us test this function by stacking `K`, `K + 1`, and `K + 2` into a kernel
tensor with three output channels.

```{.python .input #channels-multiple-output-channels-2}
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

```{.python .input #channels-multiple-output-channels-3}
corr2d_multi_in_out(X, K)
```

The first output channel matches our earlier result because its kernel is the
original `K`. We can verify the second channel by mental arithmetic: adding 1
to every kernel weight increases each output entry by the sum of all elements
in its receptive window across both input channels. For the upper-left window
that sum is 20, so the second channel begins at $56 + 20 = 76$.

### Parameters and Cost

Adding channels gives a network expressive power, but it also scales up memory
and computation rapidly. Including one scalar bias per output channel, a
convolutional layer contains

$$c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$$

learnable parameters. Near the input of a network, this number is modest: the
$3\times 3$ layer mapping 3 RGB channels to 64 channels in
:numref:`sec_why-conv` has only 1,792 parameters. Deeper inside the network,
however, channel counts grow large: a $3\times 3$ layer mapping 256 channels
to 256 channels requires 589,824 weights plus 256 biases.

Turning to computational cost, each output element requires one multiply-add
for every weight in its output channel---$c_\textrm{i} k_\textrm{h} k_\textrm{w}$
in all. Across an output feature map of spatial size
$h_\textrm{o}\times w_\textrm{o}$ and $c_\textrm{o}$ output channels, the
layer performs

$$h_\textrm{o} w_\textrm{o} \cdot c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w}$$

multiply-adds. Applied to a $56\times 56$ feature map, our 256-to-256 layer
executes $56^2\cdot 589{,}824 \approx 1.8\times 10^9$ multiply-adds for a
single image. Both parameter count and arithmetic cost scale with the product
$c_\textrm{i} c_\textrm{o}$, meaning that doubling the width of every layer
quadruples both. Because the same weights are reused at every spatial
location, arithmetic cost also scales directly with the output area
$h_\textrm{o} w_\textrm{o}$.

A third resource cost is just as important during training yet easily
overlooked: *activation memory*. To compute gradients in the backward pass,
the framework must retain each layer's output activations, storing
$h_\textrm{o} w_\textrm{o} c_\textrm{o}$ numbers per training example
(:numref:`subsec_memory_during_training`). Unlike parameter memory, activation
memory is largest in the *earliest* layers, where spatial resolution is
highest even though parameter counts are tiny. For example, a 3-to-64 layer
producing a $224\times 224$ output must cache
$224^2\cdot 64 \approx 3.2\times 10^6$ activations per example---nearly two
thousand times more numbers than its 1,792 parameters.

## The $1\times 1$ Convolution
:label:`subsec_1x1`

At first glance, a $1\times 1$ convolution ($k_\textrm{h} = k_\textrm{w} = 1$)
might look pointless: because its spatial window covers only a single pixel,
it cannot detect edges, corners, or any other pattern spanning neighboring
locations. When multiple channels are present, however, it performs a very
useful operation---just as our red-minus-green color filter did earlier. At
every spatial coordinate $(i, j)$, a $1\times 1$ convolution multiplies the
$c_\textrm{i}$-dimensional channel vector by a shared
$c_\textrm{o}\times c_\textrm{i}$ weight matrix to produce a
$c_\textrm{o}$-dimensional output vector. In other words, it acts as a fully
connected layer applied independently at every pixel with weights tied across
all spatial positions. :numref:`fig_conv_1x1` depicts a $1\times 1$
convolution mapping 3 input channels to 2 output channels while leaving height
and width intact.

![A $1\times 1$ convolution with three input channels and two output channels. Each output element is a linear combination of the input elements at the same position.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Because the exact same linear map is applied at every pixel, we can compute
the entire $1\times 1$ convolution with a single matrix multiplication: we
reshape the input into a $c_\textrm{i}\times hw$ matrix, left-multiply by the
$c_\textrm{o}\times c_\textrm{i}$ kernel matrix, and reshape the output back
to $(c_\textrm{o}, h, w)$.

```{.python .input #channels-1-times-1-convolutional-layer-1}
def corr2d_multi_in_out_1x1(X, K):
    """A 1x1 convolution as one matrix product."""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    Y = d2l.matmul(d2l.reshape(K, (c_o, c_i)), d2l.reshape(X, (c_i, h * w)))
    return d2l.reshape(Y, (c_o, h, w))
```

Let us verify on random inputs that this matrix-multiplication formulation
agrees with `corr2d_multi_in_out`.

```{.python .input #channels-1-times-1-convolutional-layer-2}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
float(d2l.reduce_sum(d2l.abs(Y1 - Y2)))
```

```{.python .input #channels-1-times-1-convolutional-layer-2}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
float(d2l.reduce_sum(d2l.abs(Y1 - Y2)))
```

```{.python .input #channels-1-times-1-convolutional-layer-2}
%%tab jax
X = jax.random.normal(d2l.get_key(), (3, 3, 3))
K = jax.random.normal(d2l.get_key(), (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
float(d2l.reduce_sum(d2l.abs(Y1 - Y2)))
```

As expected, the two results match up to floating-point rounding error.

Why are $1\times 1$ convolutions so ubiquitous in modern architectures? Their
main strength is adjusting channel dimensionality cheaply so that expensive
spatial convolutions can operate on narrower representations. Return to our
256-to-256 $3\times 3$ layer. Instead of applying $3\times 3$ kernels directly
to all 256 channels, we can use a $1\times 1$ convolution to project 256
channels down to 64, apply the $3\times 3$ convolution across those 64
channels, and use a second $1\times 1$ convolution to expand back to 256.
Together, the three layers use only
$16{,}384 + 36{,}864 + 16{,}384 = 69{,}632$ weights instead of $589{,}824$---a
reduction of roughly $8.5\times$ in both parameters and multiply-adds. Such a
*bottleneck* restricts intermediate spatial mixing to a 64-channel subspace, a
tradeoff that architectures such as ResNet (:numref:`sec_resnet`) happily make
in exchange for much greater depth. Placing an activation function after a
$1\times 1$ convolution turns it into a tiny nonlinear MLP evaluated at every
pixel :cite:`Lin.Chen.Yan.2013`; without a nonlinearity, consecutive linear
convolutions would simply collapse into a single linear map.

Two other architectural mechanisms reduce the $c_\textrm{i} c_\textrm{o}$
factor by restricting which input channels feed into each output channel:
*grouped convolutions*, popularized by ResNeXt (:numref:`subsec_resnext`), and
*depthwise-separable convolutions*, which underpin lightweight mobile models
(:numref:`sec_depthwise_separable`). We will introduce each in the next
chapter as we build the architectures that rely on them.

Notice that channels alone leave one basic property unchanged: convolutional
outputs still shift pixel-for-pixel whenever the input shifts. In
:numref:`sec_pooling`, we introduce pooling layers to summarize local
neighborhoods and build tolerance to small spatial shifts.

## Summary

To handle multi-channel inputs, a convolutional layer cross-correlates each
input channel with its own spatial kernel and sums the results, allowing a
single filter to integrate cues across channels such as color. To produce
$c_\textrm{o}$ output feature maps, the layer stacks $c_\textrm{o}$ such
multi-input kernels into a tensor of shape
$c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$,
incurring $c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w} + c_\textrm{o}$
parameters and
$h_\textrm{o} w_\textrm{o} c_\textrm{o} c_\textrm{i} k_\textrm{h} k_\textrm{w}$
multiply-adds---both proportional to $c_\textrm{i} c_\textrm{o}$---while
training must also store $h_\textrm{o} w_\textrm{o} c_\textrm{o}$ activations
per example. A $1\times 1$ convolution acts as a shared fully connected layer
at every pixel, letting architectures mix channels and construct parameter-
and compute-efficient bottlenecks.

## Exercises

1. **Composition of convolutions.** Two convolutional layers with stride
   1, no padding, and no nonlinearity in between have kernels of shape
   $c_\textrm{m} \times c_\textrm{i} \times k_1 \times k_1$ and
   $c_\textrm{o} \times c_\textrm{m} \times k_2 \times k_2$.
    1. Prove that their composition is a single convolution, and give the
       shape of its kernel.
    1. Compare the number of weights of the pair with that of the single
       convolution. For which intermediate width $c_\textrm{m}$ is the
       pair cheaper?
    1. As :numref:`sec_conv_layer` showed, three stacked single-channel
       $3 \times 3$ kernels have a $7 \times 7$ receptive field. Decide
       whether every single-channel $7 \times 7$ kernel is the composition
       of three $3 \times 3$ kernels, and justify the answer.

1. **Three costs, three layers.** A network for $224\times 224$ RGB
   images begins with four $3\times 3$ convolutions with padding 1: from 3
   to 64 channels at stride 1, from 64 to 128 at stride 2, from 128 to 128
   at stride 1, and from 128 to 256 at stride 2.
    1. For each layer give the output shape, the number of weights, the
       number of multiply-adds per image, and the number of output values
       that training must keep for the backward pass.
    1. Which layer has the most weights, which performs the most
       multiply-adds, and which keeps the most output values? Explain why
       these are three different layers.
    1. The images are replaced by $448\times 448$ ones. Which of the three
       costs change, and by what factor?

1. [code] **Multi-channel im2col.** Extend the patch matrix of
   :numref:`subsec_conv_matmul` to several input and output channels.
   Rebuild the two-channel `X` of :numref:`fig_conv_multi_in` and the
   three-output-channel `K` of :numref:`subsec_multi-output-channels`,
   since later cells of this section overwrite both. Build the patch
   matrix, whose rows are input patches of shape
   $c_\textrm{i} \times k_\textrm{h} \times k_\textrm{w}$ flattened in
   that order, and arrange `K` into a
   $c_\textrm{i} k_\textrm{h} k_\textrm{w} \times c_\textrm{o}$ kernel
   matrix whose rows follow the same order. Confirm that their product,
   rearranged into the shape of the output, reproduces
   `corr2d_multi_in_out(X, K)`.

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problems 10.6 and 10.7.*

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/69)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/70)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/273)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17998)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §6.4]{.kicker}

Multiple Input and Multiple Output Channels<br>**sums over channels · feature maps · parameter counts · 1×1 layers**
:::
:::

::: {.slide title="One 3×3 Layer from 256 to 256 Channels Has 590,000 Weights"}
[The problem]{.kicker}

- A color image is three grids: $3\times 224\times 224$ for red, green,
  and blue.
- Single-channel cross-correlation takes one grid and returns one: it can
  neither compare colors nor detect more than one pattern.
- A real layer has $c_i$ input and $c_o$ output channels, and
  $c_o c_i k_h k_w$ weights.
- $256\to 256$ channels, $3\times 3$ kernels: 589,824 weights and
  $1.8\times 10^9$ multiply-adds per image on a $56\times 56$ output.

::: {.d2l-note}
Channels give a layer many detectors; their cost grows with
$c_i\cdot c_o$.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute a multi-channel cross-correlation by hand;
- give the shape of a layer's kernel tensor, and count its parameters and
  multiply-adds;
- write a $1\times 1$ convolution as one matrix product;
- compute the savings of a $1\times 1$ bottleneck.

**Running example:** LeNet's second convolution, from 6 to 16 channels with
$5\times 5$ kernels and a $10\times 10$ output.
:::

::: {.slide title="Channels Turn One Detector per Layer into Many"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| From Fully Connected Layers to Convolutions | locality and translation equivariance give convolution |
| Convolutions for Images | cross-correlation, learning a kernel, receptive fields |
| Padding, Stride, and Dilation | output size and how far each unit sees |
| **Multiple Input and Multiple Output Channels** | **channels, 1 × 1 convolutions, cost** |
| Pooling | downsampling, tolerance to small shifts, global pooling |
| LeNet | a complete CNN on Fashion-MNIST, its learned features, the shuffle test |
:::

::: {.col .narrow}
- **Uses:** `corr2d`; output shapes from padding and stride.
- **Skips:** pooling (next deck); networks built from these layers (next
  chapter).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Input channels]{.dtitle}

[one kernel per input channel, then a sum]{.dsub}
:::
:::

::: {.slide title="A Multi-Input Kernel Adds One Cross-Correlation per Channel"}
[Input channels]{.kicker}

$\mathbf{Y} = \sum_{c=1}^{c_i} \mathbf{X}_c \star \mathbf{K}_c$, with
$\star$ the 2-D cross-correlation computed by `corr2d`, kernel shape
$c_i\times k_h\times k_w$:

![Two input channels: one cross-correlation per channel, then a sum.](../img/conv-multi-in.svg){width=72%}

First output:
$(1{\cdot}1+2{\cdot}2+4{\cdot}3+5{\cdot}4)+(0{\cdot}0+1{\cdot}1+3{\cdot}2+4{\cdot}3)=56$.
:::

::: {.slide title="Summing over Channels Lets a Kernel Compare Colors"}
[Input channels]{.kicker}

- Take a $1\times 1$ kernel with weight $+1$ on red and $-1$ on green.
- Its output is red minus green at every pixel: positive in red regions,
  negative in green ones, zero in gray ones, however bright.
- Where the sign flips, red meets green, even if both are equally bright.
- A filter on the brightness alone sees no edge there.

::: {.d2l-note}
Each output value combines the same window in every input channel:
$c_i k_h k_w$ products.
:::
:::

::: {.slide title="Input Channels in Code: One corr2d per Channel"}
[Input channels]{.kicker}

@channels-multiple-input-channels-1

The input and kernel of the figure:

@!channels-multiple-input-channels-2
:::

::: {.slide title="Check: The Second Output Element"}
[Check]{.kicker}

Input channels $\begin{smallmatrix}0&1&2\\3&4&5\\6&7&8\end{smallmatrix}$
and $\begin{smallmatrix}1&2&3\\4&5&6\\7&8&9\end{smallmatrix}$, kernel
channels $\begin{smallmatrix}0&1\\2&3\end{smallmatrix}$ and
$\begin{smallmatrix}1&2\\3&4\end{smallmatrix}$. Compute the output in the
first row, second column.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $(1{\cdot}0+2{\cdot}1+4{\cdot}2+5{\cdot}3) +
(2{\cdot}1+3{\cdot}2+5{\cdot}3+6{\cdot}4) = 25 + 47 = 72$.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Output channels]{.dtitle}

[one kernel per feature map]{.dsub}
:::
:::

::: {.slide title="One Kernel per Output Channel, One Feature Map Each"}
[Output channels]{.kicker}

$$\mathbf{Y}_o = \sum_{c=1}^{c_i} \mathbf{X}_c \star \mathbf{K}_{o,c},
\qquad o = 1, \ldots, c_o$$

- Kernel tensor of shape $c_o\times c_i\times k_h\times k_w$.
- Every output channel reads every input channel, each through a kernel of
  its own.
- Each output channel is a *feature map*: one detector's responses at every
  position.
- Channels usually grow as resolution shrinks: LeNet goes from 1 to 6 to 16
  channels.
:::

::: {.slide title="Output Channels in Code: Stack One Output per Kernel"}
[Output channels]{.kicker}

@channels-multiple-output-channels-1

- Kernels `K`, `K + 1`, `K + 2` give three channels whose first elements
  are 56, 76, and 96.
- Adding 1 to every weight adds the window's sum over both input channels,
  20 for the first window.
:::

::: {.slide title="Parameters Grow with the Product of Channel Counts"}
[Output channels]{.kicker}

parameters $= c_o c_i k_h k_w + c_o$, multiply-adds
$= h_o w_o\cdot c_o c_i k_h k_w$

| layer | parameters |
|---|---|
| $3\to 64$, $3\times 3$ | 1,792 |
| $256\to 256$, $3\times 3$ | 590,080 |

- Doubling all channel counts quadruples both.
- The $256\to 256$ layer on a $56\times 56$ output:
  $56^2\cdot 589{,}824\approx 1.8\times 10^9$ multiply-adds per image.
:::

::: {.slide title="Check: LeNet's Second Convolution"}
[Check]{.kicker}

From 6 to 16 channels, $5\times 5$ kernels, a $10\times 10$ output. How
many parameters? How many multiply-adds?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $16\cdot 6\cdot 25 + 16 = 2{,}416$ parameters;
$10^2\cdot 16\cdot 6\cdot 25 = 240{,}000$ multiply-adds.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Cheaper channel mixing]{.dtitle}

[1×1 layers and bottlenecks]{.dsub}
:::
:::

::: {.slide title="A 1×1 Convolution Is a Dense Layer Applied at Every Pixel"}
[Cheaper mixing]{.kicker}

![A $1\times 1$ convolution from 3 to 2 channels.](../img/conv-1x1.svg){width=55%}

The same $c_o\times c_i$ matrix maps the channel vector at every position:
$c_o c_i$ weights and $c_o$ biases, no spatial pattern.
:::

::: {.slide title="A 1×1 Convolution Is One Matrix Product"}
[Cheaper mixing]{.kicker}

@channels-1-times-1-convolutional-layer-1

Total absolute difference from `corr2d_multi_in_out` on random data:

@!channels-1-times-1-convolutional-layer-2
:::

::: {.slide title="A 1×1 Bottleneck Has 8.5× Fewer Weights Than a 256-Channel 3×3 Layer"}
[Cheaper mixing]{.kicker}

| layer | weights |
|---|---|
| $1\times 1$, $256\to 64$ | 16,384 |
| $3\times 3$, $64\to 64$ | 36,864 |
| $1\times 1$, $64\to 256$ | 16,384 |
| **total** | **69,632** |
| one $3\times 3$, $256\to 256$ | 589,824 |

The multiply-adds fall by the same factor. The restriction: everything
passes through 64 channels (ResNet accepts it).
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- Input channels: one kernel channel each, results added; kernels can
  compare colors (red minus green).
- Output channels: kernel $c_o\times c_i\times k_h\times k_w$,
  $c_o c_i k_h k_w + c_o$ parameters; LeNet's second layer has 2,416.
- Cost $h_o w_o c_o c_i k_h k_w$: $1.8\times 10^9$ multiply-adds for
  $256\to 256$, $3\times 3$ at $56\times 56$.
- $1\times 1$: a dense layer at every pixel; a bottleneck has $8.5\times$
  fewer weights.

Next: pooling, which summarizes neighborhoods.
:::
