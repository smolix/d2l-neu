```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Multiple Input and Multiple Output Channels
:label:`sec_channels`

While we described the multiple channels
that comprise each image (e.g., color images have the standard RGB channels
to indicate the amount of red, green and blue) and convolutional layers for multiple channels in :numref:`subsec_why-conv-channels`,
until now, we simplified all of our numerical examples
by working with just a single input and a single output channel.
This allowed us to think of our inputs, convolution kernels,
and outputs each as two-dimensional tensors.

When we add channels into the mix,
our inputs and hidden representations
both become three-dimensional tensors.
For example, each RGB input image has shape $3\times h\times w$.
We refer to this axis, with a size of 3, as the *channel* dimension. The notion of
channels is as old as CNNs themselves: for instance LeNet-5 :cite:`LeCun.Jackel.Bottou.ea.1995` uses them. 
In this section, we will take a deeper look
at convolution kernels with multiple input and multiple output channels.

```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab jax
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

```{.python .input #channels-multiple-input-and-multiple-output-channels}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## Multiple Input Channels

When the input data contains multiple channels,
we need to construct a convolution kernel
with the same number of input channels as the input data,
so that it can perform cross-correlation with the input data.
Assuming that the number of channels for the input data is $c_\textrm{i}$,
the number of input channels of the convolution kernel also needs to be $c_\textrm{i}$. If our convolution kernel's window shape is $k_\textrm{h}\times k_\textrm{w}$,
then, when $c_\textrm{i}=1$, we can think of our convolution kernel
as just a two-dimensional tensor of shape $k_\textrm{h}\times k_\textrm{w}$.

However, when $c_\textrm{i}>1$, we need a kernel
that contains a tensor of shape $k_\textrm{h}\times k_\textrm{w}$ for *every* input channel. Concatenating these $c_\textrm{i}$ tensors together
yields a convolution kernel of shape $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$.
Since the input and convolution kernel each have $c_\textrm{i}$ channels,
we can perform a cross-correlation operation
on the two-dimensional tensor of the input
and the two-dimensional tensor of the convolution kernel
for each channel, adding the $c_\textrm{i}$ results together
(summing over the channels)
to yield a two-dimensional tensor.
This is the result of a two-dimensional cross-correlation
between a multi-channel input and
a multi-input-channel convolution kernel.

:numref:`fig_conv_multi_in` provides an example 
of a two-dimensional cross-correlation with two input channels.
The shaded portions are the first output element
as well as the input and kernel tensor elements used for the output computation:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.

![Cross-correlation computation with two input channels.](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


To make sure we really understand what is going on here,
we can implement cross-correlation operations with multiple input channels ourselves.
Notice that all we are doing is performing a cross-correlation operation
per channel and then adding up the results.

```{.python .input #channels-multiple-input-channels-1}
%%tab mxnet, pytorch, jax
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

```{.python .input #channels-multiple-input-channels-1}
%%tab tensorflow
def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)
```

We can construct the input tensor `X` and the kernel tensor `K`
corresponding to the values in :numref:`fig_conv_multi_in`
to validate the output of the cross-correlation operation.

```{.python .input #channels-multiple-input-channels-2}
X = d2l.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
               [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = d2l.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

corr2d_multi_in(X, K)
```

## Multiple Output Channels
:label:`subsec_multi-output-channels`

Regardless of the number of input channels,
so far we always ended up with one output channel.
However, as we discussed in :numref:`subsec_why-conv-channels`,
it turns out to be essential to have multiple channels at each layer.
In the most popular neural network architectures,
we actually increase the channel dimension
as we go deeper in the neural network,
typically downsampling to trade off spatial resolution
for greater *channel depth*.
Intuitively, you could think of each channel
as responding to a different set of features.
The reality is a bit more complicated than this. A naive interpretation would suggest 
that representations are learned independently per pixel or per channel. 
Instead, channels are optimized to be jointly useful.
This means that rather than mapping a single channel to an edge detector, it may simply mean 
that some direction in channel space corresponds to detecting edges.

Denote by $c_\textrm{i}$ and $c_\textrm{o}$ the number
of input and output channels, respectively,
and by $k_\textrm{h}$ and $k_\textrm{w}$ the height and width of the kernel.
To get an output with multiple channels,
we can create a kernel tensor
of shape $c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$
for *every* output channel.
We concatenate them on the output channel dimension,
so that the shape of the convolution kernel
is $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$.
In cross-correlation operations,
the result on each output channel is calculated
from the convolution kernel corresponding to that output channel
and takes input from all channels in the input tensor.

We implement a cross-correlation function
to calculate the output of multiple channels as shown below.

```{.python .input #channels-multiple-output-channels-1}
def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return d2l.stack([corr2d_multi_in(X, k) for k in K], 0)
```

We construct a trivial convolution kernel with three output channels
by concatenating the kernel tensor for `K` with `K+1` and `K+2`.

```{.python .input #channels-multiple-output-channels-2}
K = d2l.stack((K, K + 1, K + 2), 0)
K.shape
```

Below, we perform cross-correlation operations
on the input tensor `X` with the kernel tensor `K`.
Now the output contains three channels.
The result of the first channel is consistent
with the result of the previous input tensor `X`
and the multi-input channel,
single-output channel kernel.

```{.python .input #channels-multiple-output-channels-3}
corr2d_multi_in_out(X, K)
```

## $1\times 1$ Convolutional Layer
:label:`subsec_1x1`

At first, a $1 \times 1$ convolution, i.e., $k_\textrm{h} = k_\textrm{w} = 1$,
does not seem to make much sense.
After all, a convolution correlates adjacent pixels.
A $1 \times 1$ convolution obviously does not.
Nonetheless, they are popular operations that are sometimes included
in the designs of complex deep networks :cite:`Lin.Chen.Yan.2013,Szegedy.Ioffe.Vanhoucke.ea.2017`.
Let's see in some detail what it actually does.

Because the minimum window is used,
the $1\times 1$ convolution loses the ability
of larger convolutional layers
to recognize patterns consisting of interactions
among adjacent elements in the height and width dimensions.
The only computation of the $1\times 1$ convolution occurs
on the channel dimension.

:numref:`fig_conv_1x1` shows the cross-correlation computation
using the $1\times 1$ convolution kernel
with 3 input channels and 2 output channels.
Note that the inputs and outputs have the same height and width.
Each element in the output is derived
from a linear combination of elements *at the same position*
in the input image.
You could think of the $1\times 1$ convolutional layer
as constituting a fully connected layer applied at every single pixel location
to transform the $c_\textrm{i}$ corresponding input values into $c_\textrm{o}$ output values.
Because this is still a convolutional layer,
the weights are tied across pixel location.
Thus the $1\times 1$ convolutional layer requires $c_\textrm{o}\times c_\textrm{i}$ weights
(plus the bias). Also note that convolutional layers are typically followed 
by nonlinearities. This ensures that $1 \times 1$ convolutions cannot simply be 
folded into other convolutions. 

![The cross-correlation computation uses the $1\times 1$ convolution kernel with three input channels and two output channels. The input and output have the same height and width.](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

Let's check whether this works in practice:
we implement a $1 \times 1$ convolution
using a fully connected layer.
The only thing is that we need to make some adjustments
to the data shape before and after the matrix multiplication.

```{.python .input #channels-1-times-1-convolutional-layer-1}
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = d2l.reshape(X, (c_i, h * w))
    K = d2l.reshape(K, (c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = d2l.matmul(K, X)
    return d2l.reshape(Y, (c_o, h, w))
```

When performing $1\times 1$ convolutions,
the above function is equivalent to the previously implemented cross-correlation function `corr2d_multi_in_out`.
Let's check this with some sample data.

```{.python .input #channels-1-times-1-convolutional-layer-2}
%%tab mxnet, pytorch
X = d2l.normal(0, 1, (3, 3, 3))
K = d2l.normal(0, 1, (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-5
```

```{.python .input #channels-1-times-1-convolutional-layer-2}
%%tab tensorflow
X = d2l.normal((3, 3, 3), 0, 1)
K = d2l.normal((2, 3, 1, 1), 0, 1)
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-5
```

```{.python .input #channels-1-times-1-convolutional-layer-2}
%%tab jax
X = jax.random.normal(d2l.get_key(), (3, 3, 3))
K = jax.random.normal(d2l.get_key(), (2, 3, 1, 1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(d2l.reduce_sum(d2l.abs(Y1 - Y2))) < 1e-5
```

## Discussion

Channels allow us to combine the best of both worlds: MLPs that allow for significant nonlinearities and convolutions that allow for *localized* analysis of features. In particular, channels allow the CNN to reason with multiple features, such as edge and shape detectors at the same time. They also offer a practical trade-off between the drastic parameter reduction arising from translation invariance and locality, and the need for expressive and diverse models in computer vision. 

Note, though, that this flexibility comes at a price. Given an image of size $(h \times w)$, the cost for computing a $k \times k$ convolution is $\mathcal{O}(h \cdot w \cdot k^2)$. For $c_\textrm{i}$ and $c_\textrm{o}$ input and output channels respectively this increases to $\mathcal{O}(h \cdot w \cdot k^2 \cdot c_\textrm{i} \cdot c_\textrm{o})$. For a $256 \times 256$ pixel image with a $5 \times 5$ kernel and $128$ input and output channels respectively this amounts to over 53 billion operations (we count multiplications and additions separately). Later on we will encounter effective strategies to cut down on the cost, e.g., by requiring the channel-wise operations to be block-diagonal, leading to architectures such as ResNeXt :cite:`Xie.Girshick.Dollar.ea.2017`. 

## Exercises

1. Assume that we have two convolution kernels of size $k_1$ and $k_2$, respectively 
   (with no nonlinearity in between).
    1. Prove that the result of the operation can be expressed by a single convolution.
    1. What is the dimensionality of the equivalent single convolution?
    1. Is the converse true, i.e., can you always decompose a convolution into two smaller ones?
1. Assume an input of shape $c_\textrm{i}\times h\times w$ and a convolution kernel of shape 
   $c_\textrm{o}\times c_\textrm{i}\times k_\textrm{h}\times k_\textrm{w}$, padding of $(p_\textrm{h}, p_\textrm{w})$, and stride of $(s_\textrm{h}, s_\textrm{w})$.
    1. What is the computational cost (multiplications and additions) for the forward propagation?
    1. What is the memory footprint?
    1. What is the memory footprint for the backward computation?
    1. What is the computational cost for the backpropagation?
1. By what factor does the number of calculations increase if we double both the number of input channels 
   $c_\textrm{i}$ and the number of output channels $c_\textrm{o}$? What happens if we double the padding?
1. Are the variables `Y1` and `Y2` in the final example of this section exactly the same? Why?
1. Express convolutions as a matrix multiplication, even when the convolution window is not $1 \times 1$. 
1. Your task is to implement fast convolutions with a $k \times k$ kernel. One of the algorithm candidates 
   is to scan horizontally across the source, reading a $k$-wide strip and computing the $1$-wide output strip 
   one value at a time. The alternative is to read a $k + \Delta$ wide strip and compute a $\Delta$-wide 
   output strip. Why is the latter preferable? Is there a limit to how large you should choose $\Delta$?
1. Assume that we have a $c \times c$ matrix. 
    1. How much faster is it to multiply with a block-diagonal matrix if the matrix is broken up into $b$ blocks?
    1. What is the downside of having $b$ blocks? How could you fix it, at least partly?

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

::: {.slide title="Channels turn filters into feature banks"}
Real images have **channels**: RGB has 3, a modern CNN's
deep feature map has hundreds (64 → 2048 is typical).

Going deeper, networks trade spatial resolution for
channel depth — same information capacity, but
representing *kinds* of features instead of *places*.

This deck:

- **Multiple input channels** — kernels grow a 3rd axis.
- **Multiple output channels** — stack many filters.
- **$1 \times 1$ convolutions** — pure channel mixing.
:::

::: {.slide title="Multi-input-channel convolution"}
With $c_i$ input channels, the kernel becomes
$c_i \times k_h \times k_w$ — a 2D filter *per input channel*.
The output is the sum of per-channel cross-correlations:

$$Y = \sum_{c=1}^{c_i} X_c * K_c.$$

![Two input channels: per-channel cross-correlation, then sum. $(1{\cdot}1 + 2{\cdot}2 + 4{\cdot}3 + 5{\cdot}4) + (0{\cdot}0 + 1{\cdot}1 + 3{\cdot}2 + 4{\cdot}3) = 56$.](../img/conv-multi-in.svg){width=82%}
:::

::: {.slide title="Input channels in code"}
@channels-multiple-input-and-multiple-output-channels

@channels-multiple-input-channels-1
:::

::: {.slide title="Verify multi-input convolution"}
Verify against the figure — same numbers:

@channels-multiple-input-channels-2
:::

::: {.slide title="Multi-output-channel convolution"}
Each output channel comes from its *own* set of input-channel
filters. Stack $c_o$ such filter sets to get a 4-D kernel
of shape $c_o \times c_i \times k_h \times k_w$:

$$\mathbf{Y}_j = \sum_{c=1}^{c_i} \mathbf{X}_c * \mathbf{K}_{j, c} \quad\text{for}\quad j = 1, \ldots, c_o.$$

Intuition: each of the $c_o$ output channels is a *different
combination* of inputs, learned to detect a different feature.
The network discovers an entire "feature dictionary" per
layer.
:::

::: {.slide title="Output channels in code"}
Apply the multi-input-channel function $c_o$ times and stack
the results along a new leading axis:

@channels-multiple-output-channels-1
:::

::: {.slide title="Verify multi-output convolution"}
Build a 3-output-channel kernel by stacking three offset
copies:

@channels-multiple-output-channels-2

@channels-multiple-output-channels-3
:::

::: {.slide title="Parameter count"}
A conv layer with $c_o$ outputs, $c_i$ inputs, and a
$k_h \times k_w$ kernel has

$$c_o \cdot c_i \cdot k_h \cdot k_w \;+\; c_o$$

learnable parameters. Standard sizes:

- 3 → 64 channels, 3×3 kernel: 1.7k weights.
- 256 → 256 channels, 3×3 kernel: 590k weights.
- 512 → 2048 channels, 3×3 kernel: 9.4M weights.

Channel count drives parameter count *quadratically*. That's
why deeper layers in CNNs widen, but not too much.
:::

::: {.slide title="The 1×1 convolution"}
A 1×1 kernel has *no* spatial structure — it doesn't look at
neighbors. So why use it?

Because it acts as a **per-pixel fully connected layer
across channels**. At every spatial position, it computes a
linear combination of the $c_i$ input channels into the
$c_o$ output channels:

![1×1 conv: 3 input channels × 2 output channels. Each output pixel = a 2×3 matrix-vector product on the input channel vector at that position.](../img/conv-1x1.svg){width=82%}
:::

::: {.slide title="1×1 conv as matmul"}
At each pixel, the 1×1 conv applies the same $c_o \times c_i$
matrix to the input channel vector. Reshape the spatial axes
out and it's a single matrix multiply:

@channels-1-times-1-convolutional-layer-1

. . .

@channels-1-times-1-convolutional-layer-2
:::

::: {.slide title="Why 1×1 convs everywhere"}
Modern architectures use them constantly:

- **Bottlenecks** (ResNet) — squeeze channels with 1×1,
  do expensive 3×3 in the smaller space, expand back with
  1×1. Cuts compute by $\sim 4×$ for the same expressive
  power.
- **Pointwise convs** (MobileNet) — depthwise 3×3 +
  pointwise 1×1 splits a regular conv into two cheaper
  pieces.
- **Squeeze-and-Excitation, attention heads** — wherever
  you need cheap channel mixing without changing
  resolution.
:::

::: {.slide title="Cost of channel depth"}
A $h \times w$ image with $k \times k$ kernel and
$c_i \to c_o$ channels takes

$$\mathcal{O}(h \cdot w \cdot k^2 \cdot c_i \cdot c_o)$$

operations. For a 256×256 image, 5×5 kernel, 128→128
channels: ~53 *billion* multiply-adds. That's per layer;
multiply by depth.

This is why
- Convs benefit massively from GPU/TPU acceleration.
- Channel reduction tricks (1×1 bottlenecks, depthwise
  separable, group conv, ResNeXt) get serious attention
  in efficiency-driven architectures.
:::

::: {.slide title="Recap"}
- Multi-input-channel: per-channel kernel, results summed
  across input channels.
- Multi-output-channel: stack independent filter banks; one
  output channel per filter set.
- Total weights: $c_o c_i k_h k_w + c_o$.
- $1\times 1$ convs = per-pixel fully connected layer
  across channels; the workhorse of modern CNN
  architecture (bottlenecks, pointwise convs).
- Compute scales linearly with $c_i \cdot c_o$ — the
  dominant cost in deep CNNs.
:::
