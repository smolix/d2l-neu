```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Pooling
:label:`sec_pooling`

Convolutional layers answer local questions—whether an edge sits at pixel
$(i, j)$—whereas a classifier must answer a global question about the whole
image. Bridging from local feature maps to a global prediction requires both
wide spatial reach (:numref:`subsec_stride_rf`) and robustness to tiny spatial
shifts so a one-pixel camera jitter does not alter the prediction.

A *pooling layer* serves both goals without adding learnable parameters: it
slides a window across each channel independently and summarizes every patch by
its *maximum* or *average*. Because the maximum over a small block stays unchanged
as long as the peak response remains inside that block, local pooling provides
*tolerance* to small shifts, while *global pooling* across the entire feature map
collapses spatial coordinates into a single channel-wise vector.
```{.python .input #pooling}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #pooling}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #pooling}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #pooling}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #pooling-1}
%%tab tensorflow, jax
from functools import partial
```

## Maximum Pooling and Average Pooling

Like a convolutional layer, a pooling operator slides a fixed-shape window---the
*pooling window*---across its input and computes a single scalar output at
each position. Unlike a convolution, however, a pooling layer has no
learnable kernel weights: it applies a fixed deterministic reduction to the
elements inside the window. *Maximum pooling* (*max-pooling* for short)
returns the largest entry in the window, whereas *average pooling* returns
their arithmetic mean.

![Max-pooling with a pooling window shape of $2\times 2$. The shaded portions are the first output element as well as the input tensor elements used for the output computation: $\max(0, 1, 3, 4)=4$.](../img/pooling.svg)
:label:`fig_pooling`

In :numref:`fig_pooling`, a $2\times2$ window slides across a $3\times3$ input
with stride 1, moving from left to right and top to bottom. Because the window
fits in two positions along each axis, the output has shape $2\times2$, with
entries given by the maximum of each window:

$$
\begin{aligned}
\max(0, 1, 3, 4) &= 4,\\
\max(1, 2, 4, 5) &= 5,\\
\max(3, 4, 6, 7) &= 7,\\
\max(4, 5, 7, 8) &= 8.
\end{aligned}
$$

Notice that five of the nine input entries---0, 1, 2, 3, and 6---are never the
maximum of any window. Perturbing them slightly leaves the output completely
unchanged, because max-pooling depends only on the winning element in each
window. The same sliding-window construction generalizes immediately to any
$k_\textrm{h} \times k_\textrm{w}$ pooling window.

Below we implement forward propagation for two-dimensional pooling in
`pool2d`. Its nested loop mirrors `corr2d` from :numref:`sec_conv_layer`,
replacing the weighted inner product with either `max` or `mean`. We also
include a `stride` argument so we can examine downsampling next.

```{.python .input #pooling-maximum-pooling-and-average-pooling-1}
%%tab mxnet, pytorch
def pool2d(X, pool_size, mode='max', stride=1):
    k_h, k_w = pool_size
    Y = d2l.zeros(((X.shape[0] - k_h) // stride + 1,
                   (X.shape[1] - k_w) // stride + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            window = X[i * stride: i * stride + k_h,
                       j * stride: j * stride + k_w]
            Y[i, j] = window.max() if mode == 'max' else window.mean()
    return Y
```

```{.python .input #pooling-maximum-pooling-and-average-pooling-1}
%%tab tensorflow
def pool2d(X, pool_size, mode='max', stride=1):
    k_h, k_w = pool_size
    reduce = tf.reduce_max if mode == 'max' else tf.reduce_mean
    # Build Y from its entries, so that gradients flow through it
    return tf.stack([
        tf.stack([reduce(X[i: i + k_h, j: j + k_w])
                  for j in range(0, X.shape[1] - k_w + 1, stride)])
        for i in range(0, X.shape[0] - k_h + 1, stride)])
```

```{.python .input #pooling-maximum-pooling-and-average-pooling-1}
%%tab jax
def pool2d(X, pool_size, mode='max', stride=1):
    k_h, k_w = pool_size
    Y = jnp.zeros(((X.shape[0] - k_h) // stride + 1,
                   (X.shape[1] - k_w) // stride + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            window = X[i * stride: i * stride + k_h,
                       j * stride: j * stride + k_w]
            Y = Y.at[i, j].set(window.max() if mode == 'max'
                               else window.mean())
    return Y
```

Testing `pool2d` on the $3\times3$ tensor from :numref:`fig_pooling` confirms
that max-pooling yields 4, 5, 7, and 8.

```{.python .input #pooling-maximum-pooling-and-average-pooling-2}
X = d2l.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))
```

Switching to average pooling on the same input returns the four window means:
2, 3, 5, and 6.

```{.python .input #pooling-maximum-pooling-and-average-pooling-3}
pool2d(X, (2, 2), 'avg')
```

## Tolerance to Small Shifts

Just how much spatial shift can pooling absorb? Let us test the simplest
possible setting. A $2\times2$ pooling layer with stride 2 partitions its
input into non-overlapping $2\times2$ blocks and summarizes each block by a
single value. Imagine a $4\times4$ feature map containing a single nonzero
response of 1---like an edge detector firing at an isolated edge---and slide
that response one pixel at a time across row 1. Below we compare the
max-pooled and average-pooled outputs across three consecutive column
positions.

```{.python .input #pooling-tolerance-to-small-shifts}
def single_response(i, j, n=4):
    """An n x n feature map that is 1 at (i, j) and 0 elsewhere."""
    return d2l.tensor([[float(r == i and c == j) for c in range(n)]
                       for r in range(n)])

for j in range(3):
    Z = single_response(1, j)
    print(f'response at (1, {j}): max',
          d2l.numpy(pool2d(Z, (2, 2), stride=2)).tolist(),
          'avg', d2l.numpy(pool2d(Z, (2, 2), 'avg', stride=2)).tolist())
```

Shifting the response from column 0 to column 1 leaves both pooled outputs
completely unchanged because both locations fall inside the same $2\times2$
block. Shifting it one pixel further to column 2 crosses a block boundary,
causing the pooled response to jump to the adjacent output cell. Thus, a
one-pixel horizontal shift is absorbed whenever the response remains within
its block---half of all starting columns---and shifts the output by one
position otherwise. Local pooling therefore makes a representation *tolerant*
to certain small shifts, not strictly invariant to them.
:numref:`fig_pooling_shift` illustrates the max-pooled outputs across all
four column positions.

![A single response moved through the four columns of a $4\times4$ feature map (top), its $2\times2$ max-pooled map with stride 2 (middle), and its global maximum (bottom). Heavy lines mark the pooling blocks. The pooled response stays put while the response stays inside its block and jumps when it crosses into the next one; the global maximum never moves.](../img/pooling-shift.svg)
:label:`fig_pooling_shift`

Notice also how max-pooling and average pooling summarize the block
differently. Max-pooling preserves the full strength of the activation (1)
anywhere inside the block, effectively asking whether the feature appears
somewhere in that neighborhood. Average pooling outputs 0.25 because the
feature occupies only one of the four cells, measuring the feature's spatial
coverage and diluting a strong isolated response with its zero neighbors.

Subsampling with stride 2 also introduces a subtle signal-processing hazard
whenever a feature crosses a sampling boundary. In the extreme case, taking
every second sample of a high-frequency alternating signal $1, 0, 1, 0, \ldots$
produces either all ones or all zeros depending on a single-pixel shift. This
distortion---occurring whenever a signal varies faster than the sampling rate
can resolve---is called *aliasing*, and it affects every stride-2 downsampler,
whether pooled or convolutional. Applying a small smoothing filter before
subsampling mitigates aliasing :cite:`zhang2019making`, a technique explored
in the exercises below.

Historically, shift tolerance was the primary motivation behind pooling.
Fukushima's Neocognitron :cite:`Fukushima.1980` paired each layer of
feature-detecting cells with pooling-like cells that fired whenever a feature
appeared anywhere in a local neighborhood. An early version of max-pooling
appeared in speech recognition :cite:`Yamaguchi.Sakamoto.Akabane.ea.1990`, and
max-pooling was subsequently proposed in computational neuroscience as a model
of hierarchical evidence aggregation in the visual cortex
:cite:`Riesenhuber.Poggio.1999`.

## Global Pooling

What about the full translation invariance we desired in :numref:`sec_why-conv`
when asking whether Waldo appears *anywhere* in an image? As we just saw, a
small fixed window cannot guarantee it. Expanding the pooling window to cover
the *entire* spatial extent of each feature map can: the maximum or mean over
all spatial positions yields the exact same scalar regardless of where the
response sits (bottom row of :numref:`fig_pooling_shift`). This operation is
called *global pooling*, and it collapses each channel to a single number.


Inside a full CNN, however, global pooling sits atop zero-padded convolutional
layers, which are strictly shift-equivariant only away from the image borders
(:numref:`sec_why-conv`). As we saw in :numref:`subsec_padding_values`, an
interior pixel participates in all nine windows of a $3 \times 3$ kernel whereas
a border pixel participates in only six (and a corner pixel in four), so while a
global maximum over an isolated response stays constant across columns, a global
average drops when a feature moves onto the padded border. Convolutional stacks
ending in global pooling are therefore strictly invariant to shifts that keep
salient content away from the boundaries, and approximately invariant otherwise.

In the modern architectures of :numref:`chap_modern_cnn`, the two roles of
pooling are frequently separated. Intermediate spatial downsampling is often
handled by stride-2 convolutions, which learn how to weight each window rather
than committing to a fixed maximum or average, while the network terminates in
*global average pooling* :cite:`Lin.Chen.Yan.2013` before the final linear
classifier. LeNet, the classic architecture we build in the next section,
predates both conventions.

## Padding and Stride

Because pooling slides a window over spatial grids in exactly the same way as
convolution, deep learning frameworks parameterize pooling layers with the
same window shape, padding, and stride arguments, and the output shape obeys
the identical formula from :numref:`sec_padding`: padding $p$ pixels on each
side of an axis of length $n$ with window size $k$ and stride $s$ yields
$\lfloor(n+2p-k)/s\rfloor+1$ outputs along that axis. The one subtle
difference is what *value* fills the padded border. To demonstrate these
behaviors with built-in pooling layers, we first construct a four-dimensional
tensor holding a single $4\times4$ feature map with entries 0 through 15.

```{.python .input #pooling-padding-and-stride-1}
%%tab mxnet, pytorch
# (example, channel, height, width)
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 1, 4, 4))
X
```

```{.python .input #pooling-padding-and-stride-1}
%%tab tensorflow, jax
# (example, height, width, channel), as in the previous sections
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (1, 4, 4, 1))
X
```

Because pooling is usually used to summarize non-overlapping regions, most
frameworks default to setting the stride equal to the pooling window size so
the windows tile the input without overlap. On our $4\times4$ input, a
$3\times3$ window with stride 3 fits only once, returning the maximum of the
upper-left $3\times3$ block (10). Since pooling has no learnable weights, the
layer requires no parameter initialization.

```{.python .input #pooling-padding-and-stride-2}
%%tab mxnet
pool = nn.MaxPool2D(3)  # The stride defaults to the window shape
pool(X)
```

```{.python .input #pooling-padding-and-stride-2}
%%tab pytorch
pool = nn.MaxPool2d(3)  # The stride defaults to the window shape
pool(X)
```

```{.python .input #pooling-padding-and-stride-2}
%%tab tensorflow
pool = tf.keras.layers.MaxPool2D(3)  # The stride defaults to the window
pool(X)
```

```{.python .input #pooling-padding-and-stride-2}
%%tab jax
# The stride of nnx.max_pool defaults to 1, so we set it to the window shape
nnx.max_pool(X, window_shape=(3, 3), strides=(3, 3))
```

We can override these defaults whenever we want overlapping windows or
explicit padding. Setting padding to 1 and stride to 2 produces an output of
size $\lfloor(4+2-3)/2\rfloor+1 = 2$ along both height and width.

```{.python .input #pooling-padding-and-stride-3}
%%tab mxnet
pool = nn.MaxPool2D(3, padding=1, strides=2)
pool(X)
```

```{.python .input #pooling-padding-and-stride-3}
%%tab pytorch
pool = nn.MaxPool2d(3, padding=1, stride=2)
pool(X)
```

```{.python .input #pooling-padding-and-stride-3}
%%tab tensorflow
# Keras pooling layers pad only 'valid' or 'same'; tf.nn.max_pool2d takes
# the padding of each axis explicitly
pool = partial(tf.nn.max_pool2d, ksize=3, strides=2,
               padding=[[0, 0], [1, 1], [1, 1], [0, 0]])
pool(X)
```

```{.python .input #pooling-padding-and-stride-3}
%%tab jax
# One (before, after) pair of padding for each spatial axis
pool = partial(nnx.max_pool, window_shape=(3, 3), strides=(2, 2),
               padding=((1, 1), (1, 1)))
pool(X)
```

What value should fill the padded border of a pooling layer? In a convolution,
zero is the natural neutral element because multiplying by zero contributes
nothing to a weighted sum. For max-pooling, however, the neutral element is
$-\infty$: padded cells can never beat a real activation, ensuring every
output equals the maximum over the genuine input entries inside its window. In
our example above, the windows cover real rows and columns $\{0, 1\}$ and
$\{1, 2, 3\}$, yielding 5, 7, 13, and 15. Padding with $-\infty$ also
preserves shift equivariance in *value*: adding a constant to every input
entry shifts every pooled output by the exact same constant. We can verify
this on `X - 16`, whose entries are all strictly negative.

```{.python .input #pooling-padding-and-stride-5}
pool(X - 16)
```

Every output decreases by exactly 16, as expected. Had the layer padded with
zeros instead, the three windows touching the border would have returned 0,
corrupting the result simply because 0 happens to be larger than all negative
inputs.

Average pooling has no analogous neutral constant. Including padded zeros in
the denominator biases border averages toward 0; the cleaner alternative is to
divide only by the count of real input entries falling inside each window.

:begin_tab:`mxnet, pytorch, jax`
Average pooling here counts the padded zeros by default
(`count_include_pad=True`); `count_include_pad=False` averages only the real
entries.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.layers.AvgPool2D` with `padding='same'` averages only the real
entries under each window.
:end_tab:

## Multiple Channels

Whereas a convolutional layer mixes information across all input channels by
summing over them (:numref:`sec_channels`), a pooling layer operates
independently on each channel. Consequently, the output always has the exact
same number of channels as the input. To verify this behavior, we concatenate
`X` and `X + 1` along the channel axis to create a two-channel tensor.

```{.python .input #pooling-multiple-channels-1}
%%tab mxnet, pytorch
X = d2l.concat((X, X + 1), 1)
X
```

```{.python .input #pooling-multiple-channels-1}
%%tab tensorflow, jax
X = d2l.concat([X, X + 1], 3)  # the channel axis is last
X
```

Applying our earlier pooling operation (padding 1, stride 2) yields two output
channels: the first reproduces 5, 7, 13, and 15, while the second returns the
same values incremented by 1.

```{.python .input #pooling-multiple-channels-2}
pool(X)
```

:begin_tab:`tensorflow, jax`
The result is in the channels-last layout, so the two channels appear side
by side in the last axis.
:end_tab:

Because pooling has no learnable weights, it discovers no patterns on its
own---it can only summarize the features detected by preceding convolutional
layers. With convolutions, padding, stride, dilation, multi-channel feature
maps, and pooling all in hand, we now have every building block needed to
assemble and train a complete convolutional neural network and compare it
directly against our earlier MLPs.

## Summary

Pooling layers summarize each spatial window of an input feature map by its
maximum or average. They introduce zero learnable parameters, process each
channel independently, and follow the exact same padding and stride output-shape
rules as convolutions; max-pooling pads borders with $-\infty$ so that padded
positions never affect the maximum. A $2\times2$ pooling layer with stride 2
halves spatial resolution along each axis and absorbs small shifts that stay
within a block, while still jumping when a response crosses a block boundary.
Max-pooling preserves the peak activation in a neighborhood, whereas average
pooling dilutes isolated peaks across the window. Finally, global pooling
collapses each entire feature map to a single scalar, yielding full
translation invariance for interior patterns (and approximate invariance near
padded borders).

## Exercises

1. **Pooling alternatives.**
    1. Do we need a separate minimum-pooling layer, or can it be expressed
       through an operation already in this section?
    1. *Softmax pooling* with temperature $\tau > 0$ replaces the window
       entries $x_1, \ldots, x_m$ by their softmax-weighted average
       $$\sum_{i=1}^m x_i \frac{\exp(x_i / \tau)}{\sum_{j=1}^m \exp(x_j / \tau)}.$$
       Find its limits as $\tau \to 0$ and as $\tau \to \infty$. Give one
       cost that softmax pooling adds relative to max-pooling.

2. **Aliasing and blur-pool.** Naive stride-2 downsampling keeps
   every second entry of its input. Measure frequency in cycles per input
   sample, so that the one-dimensional signals $(1, 0, 1, 0, 1, 0)$ and
   $(0, 1, 0, 1, 0, 1)$ both have frequency $\tfrac{1}{2}$.
    1. Apply naive stride-2 downsampling to both signals, which differ by
       a shift of one position, and compare the two outputs. Why is this
       behavior called *aliasing*?
    1. Which input frequencies can a stride-2 subsampler represent
       faithfully?
    1. Blur-pool :cite:`zhang2019making` counteracts aliasing by applying
       a small averaging filter before subsampling. Work out what
       blur-pool with a two-tap box filter
       $(\tfrac{1}{2}, \tfrac{1}{2})$ computes on the two signals above.
       Which pooling operation from this section does it coincide with?

3. [code] **Backward pass of max-pooling.** Derive the gradient of
   two-dimensional max-pooling with respect to its input, for a
   $k_\textrm{h} \times k_\textrm{w}$ window and stride equal to the
   window size, and implement it without automatic differentiation.
   Verify against the gradient that automatic differentiation computes
   through the built-in max-pooling operation of this section on a small
   random tensor, to a relative error below $10^{-4}$.

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    ConvolutionalNetworks, naive max-pooling backward pass.*

4. **Max-pooling via ReLU.** ● A convolution is linear, so it cannot
   compute a maximum by itself.
    1. Express $\max(a, b)$ using only affine operations and
       $\textrm{ReLU}(x) = \max(0, x)$.
    1. Use this to implement max-pooling as a stack of convolutions and
       ReLU layers.
    1. Count the channels per layer and the ReLU layers that your
       construction uses for a $2 \times 2$ window and for a
       $3 \times 3$ window. How does the number of ReLU layers grow with
       the number of entries in the window?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/71)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/72)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/274)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17999)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §6.5]{.kicker}

Pooling<br>**max and average · tolerance to small shifts · global pooling · padding with −∞**
:::
:::

::: {.slide title="Shift the Image by One Pixel and Every Response Moves by One Pixel"}
[The problem]{.kicker}

- Convolutions answer local questions: is there an edge *here*?
- "Shirt or sneaker?" is one question about the whole $28\times28$ image.
- Reach is settled: fourteen $3\times3$ layers at full resolution, six
  with two halvings (the receptive-field formula with strides).
- New: an edge detector responds at the pixel of the edge. A one-pixel
  shift of the image moves every response, yet the answer must not change.

::: {.d2l-note .warn}
Pooling halves the resolution and decides what to keep: the maximum or the
mean of each window.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute max- and average pooling of a small input by hand;
- predict which one-pixel shifts a $2\times2$, stride-2 pool absorbs;
- say what global pooling removes, and what the border leaves;
- predict output shapes from window, padding, and stride;
- say why max-pooling pads with $-\infty$;
- count the channels and parameters of a pooling layer.

**Running example:** the $3\times3$ input $0, \ldots, 8$, then the
$4\times4$ input $0, \ldots, 15$.

**Idea:** replace each small window by its maximum or mean.
:::

::: {.slide title="Shrinking the Feature Maps, Between Channels and LeNet"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| From Fully Connected Layers to Convolutions | locality and translation equivariance give convolution |
| Convolutions for Images | cross-correlation, learning a kernel, receptive fields |
| Padding, Stride, and Dilation | output size and how far each unit sees |
| Multiple Input and Multiple Output Channels | channels, 1 × 1 convolutions, cost |
| **Pooling** | **downsampling, tolerance to small shifts, global pooling** |
| LeNet | a complete CNN on Fashion-MNIST, its learned features, the shuffle test |
:::

::: {.col .narrow}
- **Uses:** the sliding window of `corr2d`; the output-shape formula
  with padding and stride.
- **Skips:** downsampling with learned weights (strided convolutions,
  next chapter).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Maximum and Average Pooling]{.dtitle}

[a window, a reduction, no parameters]{.dsub}
:::
:::

::: {.slide title="Max-Pooling Replaces Each Window by Its Largest Entry"}
[Maximum and average pooling]{.kicker}

![2×2 max-pooling with stride 1 on a 3×3 input: $\max(0, 1, 3, 4) = 4$.](../img/pooling.svg){width=70%}

- The other windows give $\max(1,2,4,5)=5$, $\max(3,4,6,7)=7$,
  $\max(4,5,7,8)=8$.
- Average pooling takes the mean instead: 2, 3, 5, 6.
- Entries that are the largest in no window (0, 1, 2, 3, 6) do not
  affect the output.
:::

::: {.slide title="Pooling Is the Loop of Cross-Correlation with Max or Mean"}
[Maximum and average pooling]{.kicker}

@pooling-maximum-pooling-and-average-pooling-1

::: {.d2l-note}
No kernel, so nothing to learn: a pooling layer has **zero parameters**.
:::
:::

::: {.slide title="The Implementation Reproduces the Figure"}
[Maximum and average pooling]{.kicker}

Max-pooling of the $3\times3$ input: 4, 5, 7, 8.

@!pooling-maximum-pooling-and-average-pooling-2

Average pooling returns the four window means, 2, 3, 5, 6:

@!pooling-maximum-pooling-and-average-pooling-3
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Tolerance to Small Shifts]{.dtitle}

[what a 2×2, stride-2 pool absorbs, and what it does not]{.dsub}
:::
:::

::: {.slide title="A Response That Stays in Its 2×2 Block Leaves the Pooled Output Unchanged"}
[Tolerance to small shifts]{.kicker}

A $4\times4$ map with a single 1, moved along row 1; $2\times2$ windows,
stride 2:

@!pooling-tolerance-to-small-shifts

- Column 0 → 1: same block, **same output**.
- Column 1 → 2: next block, the output **moves**.
:::

::: {.slide title="Half of All One-Pixel Shifts Are Absorbed, the Other Half Move the Output"}
[Tolerance to small shifts]{.kicker}

- Blocks are 2 pixels wide: a response at an even column stays in its
  block when it moves right; at an odd column it crosses.
- Pooling gives **tolerance**, not invariance.
- The jump at a block boundary is *aliasing*; it affects every stride-2
  downsampler, pooled or convolutional. Blurring before subsampling
  reduces it (Zhang, 2019).

::: {.d2l-note .rule}
Max reports the response at full strength (1); average dilutes it over
the window (0.25).
:::
:::

::: {.slide title="Check: How Often Does a 3×3, Stride-3 Pool Absorb a One-Pixel Shift?"}
[Check]{.kicker}

A single response sits at a random column of a feature map. It moves one
pixel to the right. With $3\times3$ windows and stride 3, how often is the
pooled output unchanged? With $2\times2$ and stride 2?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** It stays in its block unless it sits in the last column of
the block: 2 of 3 positions for width 3, 1 of 2 for width 2. Wider
windows absorb more shifts and keep less resolution.
:::
:::
:::

::: {.slide title="Only a Window over the Whole Map Removes the Position"}
[Global pooling]{.kicker}

![](../img/pooling-shift.svg){width=80%}

Top: one response in columns 0 to 3. Middle: its $2\times2$ max-pool
jumps at the block boundary. Bottom: the maximum over all positions never
moves.

::: {.d2l-note}
*Global pooling* reduces each channel to one number: equivariant maps in,
an invariant answer out.
:::
:::

::: {.slide title="After a Padded Convolution, the Global Mean Depends on the Column"}
[Global pooling]{.kicker}

The same response in columns 0 to 3; a $3\times3$ kernel of ones with zero
padding 1; then the maximum and the mean over all 16 outputs:

@!pooling-global-pooling-2

::: {.d2l-note .rule}
Nine windows contain an interior response (mean 9/16), six a border one
(6/16). Convolution plus global pooling is invariant only to shifts that
stay away from the border.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Padding, Stride, and Channels]{.dtitle}

[the shape rules of convolutions, with one new padding value]{.dsub}
:::
:::

::: {.slide title="With Stride Equal to the Window, Pooling Windows Tile the Input"}
[Padding and stride]{.kicker}

The usual pooling stride is the window shape: no window overlaps another.
On the $4\times4$ input $0, \ldots, 15$:

@pooling-padding-and-stride-2

A $3\times3$ window fits once: the output is $\max$ of the top-left
$3\times3$ block, 10.
:::

::: {.slide title="Padding and Stride Follow the Convolution Formula"}
[Padding and stride]{.kicker}

Output height $\lfloor (n + 2p - k)/s \rfloor + 1$, with $p$ rows of
padding on each side. Here $n=4$, $k=3$, $p=1$, $s=2$:
$\lfloor 3/2 \rfloor + 1 = 2$.

@pooling-padding-and-stride-3
:::

::: {.slide title="Max-Pooling Pads with −∞, So Padding Never Wins"}
[Padding and stride]{.kicker}

Adding a constant to every input must add it to every output. On `X - 16`
(all entries below 0):

@pooling-padding-and-stride-5

- Each output is $5, 7, 13, 15$ minus 16.
- Zero padding would give 0 in the three windows that touch the padding.
- Average pooling has no neutral value: counted padded zeros pull border
  means toward 0, unless the layer averages only the real entries.
:::

::: {.slide title="Each Channel Is Pooled Separately" except="tensorflow,jax"}
[Channels]{.kicker}

A convolution sums over its input channels; pooling does not. Stack `X`
and `X + 1` as two channels and pool:

@!pooling-multiple-channels-2

Two channels in, two channels out: 5, 7, 13, 15 and the same plus 1.
:::

::: {.slide title="Each Channel Is Pooled Separately" only="tensorflow,jax"}
[Channels]{.kicker}

A convolution sums over its input channels; pooling does not. Stack `X`
and `X + 1` as two channels (last axis) and pool:

@!pooling-multiple-channels-2

Channels last: each innermost pair is one position, channel 1 then
channel 2. Channel 1 holds 5, 7, 13, 15; channel 2 the same plus 1.
:::

::: {.slide title="Check: What Does a 2×2, Stride-2 Max-Pool Do to 6@28×28?"}
[Check]{.kicker}

Output shape, number of parameters, and number of comparisons?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $6@14\times14$, 1,176 outputs. **Zero** parameters. Each
output takes 3 comparisons among 4 entries: 3,528 comparisons, against
117,600 multiply-adds for the $5\times5$, 1→6 convolution that produced
the input.
:::
:::
:::

::: {.slide title="Later Networks Learn Their Downsampling and Pool Once at the End"}
[Beyond this deck]{.kicker}

Pooling does two jobs, and later networks give them to different layers.

- **Halving the resolution:** often a convolution with stride 2, which
  learns how to combine each window instead of fixing max or mean.
- **Removing the position:** global average pooling of the last feature
  map, one number per channel, followed by a linear classifier.
- LeNet, in the next deck, does neither: it pools locally and ends in
  dense layers.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Pooling** replaces each window by its max or mean; no parameters;
  one channel at a time.
- **Shapes** follow the convolution formula; stride = window tiles the
  input.
- **Padding** for max-pooling is $-\infty$: pooling `X - 16` returns the
  outputs minus 16.
- **Tolerance:** a $2\times2$, stride-2 pool absorbs a one-pixel shift
  that stays in its block, half of all positions.
- **Global pooling:** the maximum over the whole map is 1 in every column;
  after a padded convolution the mean is 9/16 inside and 6/16 at the
  border.

**Next:** LeNet puts convolutions and pooling together into a classifier.
:::
