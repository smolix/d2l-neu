```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Padding and Stride
:label:`sec_padding`

Recall the example of a convolution in :numref:`fig_correlation`. 
The input had both a height and width of 3
and the convolution kernel had both a height and width of 2,
yielding an output representation with dimension $2\times2$.
Assuming that the input shape is $n_\textrm{h}\times n_\textrm{w}$
and the convolution kernel shape is $k_\textrm{h}\times k_\textrm{w}$,
the output shape will be $(n_\textrm{h}-k_\textrm{h}+1) \times (n_\textrm{w}-k_\textrm{w}+1)$: 
we can only shift the convolution kernel so far until it runs out
of pixels to apply the convolution to. 

In the following we will explore a number of techniques,
including padding, strided convolutions, and dilation,
that offer more control over the size of the output
and over the area of the input that each output element sees. 
As motivation, note that since kernels generally
have width and height greater than $1$,
after applying many successive convolutions,
we tend to wind up with outputs that are
considerably smaller than our input.
If we start with a $240 \times 240$ pixel image,
ten unpadded layers of $5 \times 5$ convolutions
reduce the representation to $200 \times 200$ pixels. The surviving outputs
are centered away from the original boundary, and boundary pixels influence
far fewer activations than interior pixels.
*Padding* is the most popular tool for handling this issue.
In other cases, we may want to reduce the dimensionality drastically,
e.g., if we find the original input resolution to be unwieldy.
*Strided convolutions* are a popular technique that can help in these instances.

```{.python .input #padding-and-strides-padding-and-stride}
%%tab mxnet
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #padding-and-strides-padding-and-stride}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input #padding-and-strides-padding-and-stride}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input #padding-and-strides-padding-and-stride}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Padding

As described above, one tricky issue when applying convolutional layers
is that we tend to lose pixels on the perimeter of our image. Consider :numref:`img_conv_reuse` that depicts the pixel utilization as a function of the convolution kernel size and the position within the image. The pixels in the corners are hardly used at all. 

![Pixel utilization for convolutions of size $1 \times 1$, $2 \times 2$, and $3 \times 3$ respectively.](../img/conv-reuse.svg)
:label:`img_conv_reuse`

Since we typically use small kernels,
for any given convolution
we might only lose a few pixels
but this can add up as we apply
many successive convolutional layers.
One straightforward solution to this problem
is to add extra pixels of filler around the boundary of our input image,
thus increasing the effective size of the image.
Typically, we set the values of the extra pixels to zero.
In :numref:`img_conv_pad`, we pad a $3 \times 3$ input,
increasing its size to $5 \times 5$.
The corresponding output then increases to a $4 \times 4$ matrix.
The shaded portions are the first output element as well as the input and kernel tensor elements used for the output computation: $0\times0+0\times1+0\times2+0\times3=0$.

![Two-dimensional cross-correlation with padding.](../img/conv-pad.svg)
:label:`img_conv_pad`

In general, if we add a total of $p_\textrm{h}$ rows of padding
(roughly half on top and half on bottom)
and a total of $p_\textrm{w}$ columns of padding
(roughly half on the left and half on the right),
the output shape will be

$$(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+1)\times(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+1).$$

This means that the height and width of the output
will increase by $p_\textrm{h}$ and $p_\textrm{w}$, respectively.

In many cases, we will want to set $p_\textrm{h}=k_\textrm{h}-1$ and $p_\textrm{w}=k_\textrm{w}-1$
to give the input and output the same height and width.
This will make it easier to predict the output shape of each layer
when constructing the network.
Assuming that $k_\textrm{h}$ is odd here,
we will pad $p_\textrm{h}/2$ rows on both sides of the height.
If $k_\textrm{h}$ is even, one possibility is to
pad $\lceil p_\textrm{h}/2\rceil$ rows on the top of the input
and $\lfloor p_\textrm{h}/2\rfloor$ rows on the bottom.
We will pad both sides of the width in the same way.

CNNs commonly use convolution kernels
with odd height and width values, such as 1, 3, 5, or 7.
Choosing odd kernel sizes has the benefit
that we can preserve the dimensionality
while padding with the same number of rows on top and bottom,
and the same number of columns on left and right.

Moreover, this practice of using odd kernels
and padding to precisely preserve dimensionality
offers a clerical benefit.
For any two-dimensional tensor `X`,
when the kernel's size is odd
and the number of padding rows and columns
on all sides are the same,
thereby producing an output with the same height and width as the input,
we know that the output `Y[i, j]` is calculated
by cross-correlation of the input and convolution kernel
with the window centered on `X[i, j]`.

In the following example, we create a two-dimensional convolutional layer
with a height and width of 3
and apply 1 pixel of padding on all sides.
Given an input with a height and width of 8,
we find that the height and width of the output is also 8.

```{.python .input #padding-and-strides-padding-1}
%%tab mxnet
# We define a helper function to calculate convolutions. It initializes 
# the convolutional layer weights and performs corresponding dimensionality 
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)
X = np.random.uniform(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-padding-1}
%%tab pytorch
# We define a helper function to calculate convolutions. It initializes the
# convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return Y.reshape(Y.shape[2:])

# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-padding-1}
%%tab tensorflow
# We define a helper function to calculate convolutions. It initializes
# the convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicates that batch size and the number of channels are both 1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # Strip the first two dimensions: examples and channels
    return tf.reshape(Y, Y.shape[1:3])
# 1 row and column is padded on either side, so a total of 2 rows or columns
# are added
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-padding-1}
%%tab jax
# We define a helper function to calculate convolutions. It initializes
# the convolutional layer weights and performs corresponding dimensionality
# elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, X.shape, 1) indicates that batch size and the number of channels are both 1
    key = d2l.get_key()
    X = X.reshape((1,) + X.shape + (1,))
    Y, _ = conv2d.init_with_output(key, X)
    # Strip the dimensions: examples and channels
    return Y.reshape(Y.shape[1:3])
# 1 row and column is padded on either side, so a total of 2 rows or columns are added
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='SAME')
X = jax.random.uniform(d2l.get_key(), shape=(8, 8))
comp_conv2d(conv2d, X).shape
```

When the height and width of the convolution kernel are different,
we can make the output and input have the same height and width
by setting different padding numbers for height and width.

```{.python .input #padding-and-strides-padding-2}
%%tab mxnet
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-padding-2}
%%tab pytorch
# We use a convolution kernel with height 5 and width 3. The padding on either
# side of the height and width are 2 and 1, respectively
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-padding-2}
%%tab tensorflow
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-padding-2}
%%tab jax
# We use a convolution kernel with height 5 and width 3. The padding on
# either side of the height and width are 2 and 1, respectively
conv2d = nn.Conv(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape
```

## Stride

When computing the cross-correlation,
we start with the convolution window
at the upper-left corner of the input tensor,
and then slide it over all locations both down and to the right.
In the previous examples, we defaulted to sliding one element at a time.
However, sometimes, either for computational efficiency
or because we wish to downsample,
we move our window more than one element at a time,
skipping the intermediate locations. This is particularly useful if the convolution 
kernel is large since it captures a large area of the underlying image.

We refer to the number of rows and columns traversed per slide as *stride*.
So far, we have used strides of 1, both for height and width.
Sometimes, we may want to use a larger stride.
:numref:`img_conv_stride` shows a two-dimensional cross-correlation operation
with a stride of 3 vertically and 2 horizontally.
The shaded portions are the output elements as well as the input and kernel tensor elements used for the output computation: $0\times0+0\times1+1\times2+2\times3=8$, $0\times0+6\times1+0\times2+0\times3=6$.
We can see that when the second element of the first column is generated,
the convolution window slides down three rows.
The convolution window slides two columns to the right
when the second element of the first row is generated.
When the convolution window continues to slide two columns to the right on the input,
there is no output because the input element cannot fill the window
(unless we add another column of padding).

![Cross-correlation with strides of 3 and 2 for height and width, respectively.](../img/conv-stride.svg)
:label:`img_conv_stride`

In general, when the stride for the height is $s_\textrm{h}$
and the stride for the width is $s_\textrm{w}$, the output shape is

$$\lfloor(n_\textrm{h}-k_\textrm{h}+p_\textrm{h}+s_\textrm{h})/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-k_\textrm{w}+p_\textrm{w}+s_\textrm{w})/s_\textrm{w}\rfloor.$$

If we set $p_\textrm{h}=k_\textrm{h}-1$ and $p_\textrm{w}=k_\textrm{w}-1$,
then the output shape can be simplified to
$\lfloor(n_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor$.
Going a step further, if the input height and width
are divisible by the strides on the height and width,
then the output shape will be $(n_\textrm{h}/s_\textrm{h}) \times (n_\textrm{w}/s_\textrm{w})$.

Below, we set the strides on both the height and width to 2,
thus halving the input height and width.

```{.python .input #padding-and-strides-stride-1}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-stride-1}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-stride-1}
%%tab tensorflow
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-stride-1}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 3), padding=1, strides=2)
comp_conv2d(conv2d, X).shape
```

Let's look at a slightly more complicated example.

```{.python .input #padding-and-strides-stride-2}
%%tab mxnet
conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-stride-2}
%%tab pytorch
conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-stride-2}
%%tab tensorflow
# tf.keras.Conv2D accepts only 'same'/'valid' for `padding`; we use a
# ZeroPadding2D layer to apply padding=(0, 1) (matching the MX/PT/JAX
# tabs) before the convolution.
conv2d = tf.keras.Sequential([
    tf.keras.layers.ZeroPadding2D(padding=(0, 1)),
    tf.keras.layers.Conv2D(1, kernel_size=(3, 5), padding='valid',
                           strides=(3, 4))])
comp_conv2d(conv2d, X).shape
```

```{.python .input #padding-and-strides-stride-2}
%%tab jax
conv2d = nn.Conv(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape
```

## Dilation
:label:`sec_dilation`

Padding and stride change where the kernel is applied. A third modification
changes the kernel's footprint itself: a *dilated convolution*
:cite:`yu2016dilated` places the kernel taps $d$ pixels apart
instead of at adjacent positions,
where $d$ is called the *dilation* (or *dilation rate*).
A $k_\textrm{h} \times k_\textrm{w}$ kernel with dilation $d$
still has $k_\textrm{h} k_\textrm{w}$ weights,
but along each axis the taps now span a window of

$$k' = k + (k-1)(d-1) = d(k-1)+1$$

pixels, the *effective kernel size*.
Setting $d=1$ recovers the ordinary convolution.
:numref:`fig_conv_dilation` shows a $3 \times 3$ kernel at dilations 1 and 2:
at dilation 2 the nine taps spread over a $5 \times 5$ footprint,
skipping every other pixel,
while the cost of applying the kernel is unchanged.

![A 3×3 kernel at dilation 1 and dilation 2: the nine taps spread over a 5×5 footprint while the cost stays fixed.](../img/arch-conv-dilation.svg)
:label:`fig_conv_dilation`

For output shapes, a dilated kernel behaves exactly like a dense kernel
of its effective size. Substituting $k'$ into the strided formula above
gives the general form covering padding, stride, and dilation at once:

$$\lfloor(n_\textrm{h}-d_\textrm{h}(k_\textrm{h}-1)+p_\textrm{h}+s_\textrm{h}-1)/s_\textrm{h}\rfloor \times \lfloor(n_\textrm{w}-d_\textrm{w}(k_\textrm{w}-1)+p_\textrm{w}+s_\textrm{w}-1)/s_\textrm{w}\rfloor.$$

With $d_\textrm{h}=d_\textrm{w}=1$ this reduces to the strided formula,
and with unit stride and no padding to the $(n-k+1)$ rule we started from.

The point of dilation is receptive-field growth on a budget.
In the receptive-field formula :eqref:`eq_receptive_field`,
each layer contributes $k'_i - 1 = d_i(k_i - 1)$ at stride 1,
so doubling the dilation at every layer,
$d_i = 1, 2, 4, \ldots, 2^{L-1}$ for $3 \times 3$ kernels,
yields a receptive field of side $2^{L+1}-1$:
it grows exponentially with depth
while every layer costs the same nine multiplications per output.
The alternative route to a large receptive field, striding,
pays for it by shrinking the output.
This makes dilation the standard tool for *dense prediction* tasks
such as semantic segmentation,
where every pixel needs wide context
but the output must keep the input's resolution;
we will meet it again in the fully convolutional networks
of :numref:`sec_fcn`.

Let's verify the effective-size claim numerically:
on our $8 \times 8$ input,
a $3 \times 3$ kernel at dilation 2
produces the same $4 \times 4$ output
as a dense $5 \times 5$ kernel.

```{.python .input #padding-and-strides-dilation}
%%tab mxnet
# Dilation 2 gives the 3x3 kernel a 5x5 footprint, so both convolutions
# shrink the 8x8 input to 4x4
conv2d = nn.Conv2D(1, kernel_size=3, dilation=2)
conv5 = nn.Conv2D(1, kernel_size=5)
comp_conv2d(conv2d, X).shape, comp_conv2d(conv5, X).shape
```

```{.python .input #padding-and-strides-dilation}
%%tab pytorch
# Dilation 2 gives the 3x3 kernel a 5x5 footprint, so both convolutions
# shrink the 8x8 input to 4x4
conv2d = nn.LazyConv2d(1, kernel_size=3, dilation=2)
conv5 = nn.LazyConv2d(1, kernel_size=5)
comp_conv2d(conv2d, X).shape, comp_conv2d(conv5, X).shape
```

```{.python .input #padding-and-strides-dilation}
%%tab tensorflow
# Dilation 2 gives the 3x3 kernel a 5x5 footprint, so both convolutions
# shrink the 8x8 input to 4x4
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='valid',
                                dilation_rate=2)
conv5 = tf.keras.layers.Conv2D(1, kernel_size=5, padding='valid')
comp_conv2d(conv2d, X).shape, comp_conv2d(conv5, X).shape
```

```{.python .input #padding-and-strides-dilation}
%%tab jax
# Dilation 2 gives the 3x3 kernel a 5x5 footprint, so both convolutions
# shrink the 8x8 input to 4x4
conv2d = nn.Conv(1, kernel_size=(3, 3), padding='VALID', kernel_dilation=2)
conv5 = nn.Conv(1, kernel_size=(5, 5), padding='VALID')
comp_conv2d(conv2d, X).shape, comp_conv2d(conv5, X).shape
```

## Summary and Discussion

Padding can increase the height and width of the output and is often chosen so
that input and output have the same spatial shape. It lets boundary pixels
affect outputs centered near the boundary, but does not make their use identical
to that of interior pixels: a corner pixel still belongs to fewer windows
containing real image values. Typically we pick symmetric padding on both sides
of the input height and width. In this case we refer to
$(p_\textrm{h}, p_\textrm{w})$ padding. Most commonly we set
$p_\textrm{h} = p_\textrm{w}$, in which case we simply state that we choose
padding $p$.

A similar convention applies to strides. When the vertical stride $s_\textrm{h}$ and horizontal stride $s_\textrm{w}$ match, we simply talk about stride $s$. The stride can reduce the resolution of the output, for example reducing the height and width of the output to only $1/n$ of the height and width of the input for $n > 1$. The same shorthand covers dilation: when $d_\textrm{h} = d_\textrm{w}$ we speak of dilation $d$. Dilation widens the window each output sees without adding weights or reducing resolution, which is what dense prediction needs. By default, the padding is 0 and the stride and dilation are 1.

So far all padding that we discussed simply extended images with zeros. This is cheap, and operators can be engineered to handle it implicitly without allocating a larger tensor, but it is not a neutral choice: zero padding is a *boundary condition*, an assertion that the world outside the image is black. Filters whose window overlaps the border therefore see systematically different inputs from filters in the interior, and the resulting artifacts propagate deep into the feature maps :cite:`Alsallakh.Kokhlikyan.Miglani.ea.2020`. Padding also leaks absolute position: convolution itself cannot tell where in the image it is, but a unit near the border can infer its location from how much zero "whitespace" it sees, and trained CNNs do exploit this signal. Alternatives such as reflect padding (mirroring the border rows and columns) or replicate padding (repeating them) remove the artificial black frame; in practice they are reached for when border artifacts become visible, for example in image generation.


## Exercises

1. Given the final code example in this section with kernel size $(3, 5)$, padding $(0, 1)$, and stride $(3, 4)$, 
   calculate the output shape to check if it is consistent with the experimental result.
1. For audio signals, what does a stride of 2 correspond to?
1. Implement mirror padding, i.e., padding where the border values are simply mirrored to extend tensors. 
1. What are the computational benefits of a stride larger than 1?
1. What might be statistical benefits of a stride larger than 1?
1. How would you implement a stride of $\frac{1}{2}$? What does it correspond to? When would this be useful? Compare your answer with the transposed convolutions of :numref:`sec_transposed_conv`.
1. A network stacks four $3 \times 3$ convolutions with stride 1 and dilations $1, 2, 4, 8$. Use :eqref:`eq_receptive_field` with each kernel replaced by its effective size to compute the receptive field of one output element. Which pixels inside that field does the output actually depend on? When does this *gridding* effect become a problem, and how would you choose a dilation schedule that avoids it?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/67)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/68)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/272)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17997)
:end_tab:

<!-- slides -->

::: {.slide title="Padding and stride control resolution"}
A plain convolution always **shrinks** its input. With an
$n \times n$ image and a $k \times k$ kernel:

$$n \times n \;\longrightarrow\; (n - k + 1) \times (n - k + 1).$$

Stack ten 5×5 layers on a 240×240 image:

$$240 \to 236 \to 232 \to \ldots \to 200.$$

30% of the area gone — *all* of it from the boundary. Two
knobs to control this: **padding** (fight the shrink) and
**stride** (lean into it, on purpose).
:::

::: {.slide title="The boundary problem"}
Even before stacking, single convs use boundary pixels much
less than central ones. Each pixel only contributes when the
kernel window covers it — interior pixels appear in many
windows, corner pixels in just one:

![Pixel utilization: 1×1, 2×2, and 3×3 kernels. Larger kernels make the boundary problem worse.](../img/conv-reuse.svg){width=88%}

Information at the edges is systematically underweighted.
Padding fixes both the shrinking *and* the underweighting.
:::

::: {.slide title="Padding: add zeros around the input"}
Add a "frame" of zero-valued pixels around the input. The
kernel can now slide further, including positions where its
window hangs off the original image:

![3×3 input padded to 5×5; 2×2 kernel gives a 4×4 output. The shaded element is $0{\cdot}0 + 0{\cdot}1 + 0{\cdot}2 + 0{\cdot}3 = 0$.](../img/conv-pad.svg){width=78%}

. . .

With $p_h$ rows and $p_w$ columns of padding total:

$$(n_h - k_h + p_h + 1) \times (n_w - k_w + p_w + 1).$$

To **preserve shape**: pick $p_h = k_h - 1$, $p_w = k_w - 1$.
:::

::: {.slide title="Why kernels are usually odd"}
For odd $k$, $(k-1)/2$ is an integer — we can pad
**symmetrically**, the same on both sides, and the output
position $(i, j)$ corresponds to a window **centered** on
input $(i, j)$. Clean to reason about.

- **Standard sizes**: 1, 3, 5, 7.
- "**SAME** padding" = $p = (k-1)/2$ → output shape = input shape.
- Even $k$ forces a left/right asymmetry — pad floor on one
  side, ceil on the other.

That's why every modern CNN you see uses 3×3 kernels with
padding 1, or 5×5 with padding 2, or 7×7 with padding 3.
:::

::: {.slide title="Padding helper"}
Define a helper to wrap input/output reshaping, then ask the
basic question: 8×8 input, 3×3 kernel, padding=1 — what's
the output shape?

@padding-and-strides-padding-and-stride
:::

::: {.slide title="Shape-preserving padding"}
@padding-and-strides-padding-1

Same shape — the padded conv is "shape-preserving". With an
asymmetric kernel, mirror the asymmetry in the padding:

@padding-and-strides-padding-2
:::

::: {.slide title="Stride: skipping positions on purpose"}
The opposite problem: sometimes the input is huge and we
*want* to shrink fast. Move the kernel by $s > 1$ at each step
— skipping intermediate positions:

![Cross-correlation with vertical stride 3 and horizontal stride 2. The kernel jumps three rows down and two columns right.](../img/conv-stride.svg){width=78%}

Computational benefit: $s\times$ fewer positions to evaluate
in each direction, so $s_h s_w \times$ fewer operations.
Statistical benefit: aggressive downsampling forces the
network to summarize.
:::

::: {.slide title="Stride formula"}
With strides $s_h$ vertically and $s_w$ horizontally:

$$\Big\lfloor \frac{n_h - k_h + p_h + s_h}{s_h} \Big\rfloor \times
  \Big\lfloor \frac{n_w - k_w + p_w + s_w}{s_w} \Big\rfloor.$$

Two helpful special cases:

- **SAME-padded** ($p = k - 1$): output is
  $\lfloor (n + s - 1)/s \rfloor$.
- **SAME + $s$ divides $n$**: output is exactly $n / s$.

So the standard "halve the resolution" recipe — kernel 3,
padding 1, stride 2 — turns $n \times n$ into $\lceil n/2 \rceil \times \lceil n/2 \rceil$.
:::

::: {.slide title="Stride in code"}
Halving an 8×8 input:

@padding-and-strides-stride-1

. . .

A more aggressive (and asymmetric) version — kernel 3×5,
padding 0×1, stride 3×4:

@padding-and-strides-stride-2

The output formula above predicts the shape; the code just
confirms it.
:::

::: {.slide title="Dilation: nine weights, a wider view"}
A **dilated** convolution spaces the kernel taps $d$ pixels
apart. Effective kernel size $k + (k-1)(d-1)$: a 3×3 kernel at
dilation 2 samples a 5×5 window at unchanged cost.

![A 3×3 kernel at dilation 1 and dilation 2.](../img/arch-conv-dilation.svg){width=80%}

@padding-and-strides-dilation

Doubling $d$ at every layer grows the receptive field
**exponentially** with depth. Standard tool in dense prediction
(FCN, DeepLab), where the output must keep full resolution.
:::

::: {.slide title="Three patterns to remember"}
Most production CNNs are built from these three:

| | kernel | padding | stride | output |
|---|---|---|---|---|
| **Preserve** | 3 | 1 | 1 | $n \times n$ |
| **Halve** | 3 | 1 | 2 | $n/2 \times n/2$ |
| **Patchify** | $k$ | 0 | $k$ | $n/k \times n/k$ |

*Preserve*: ResNet feature mixing. *Halve*: standard
downsample layer. *Patchify*: ViT turns 224×224 into a
14×14 grid of 16×16 patches in one shot.
:::

::: {.slide title="Recap"}
- Vanilla conv shrinks: $n \to n - k + 1$. Cumulative
  shrinkage destroys boundary information.
- **Padding $p$** grows the input — pick $p = k - 1$ to
  preserve shape ("SAME"). Odd kernels make this symmetric.
- **Stride $s$** skips positions — $s = 2$ is the standard
  downsampler; large $s$ patchifies.
- **Dilation $d$** spreads the taps: effective kernel
  $k + (k-1)(d-1)$, exponential receptive-field growth at
  fixed cost.
- One formula covers all three:
  $$\text{out} = \Big\lfloor \frac{n - d(k-1) - 1 + p}{s} \Big\rfloor + 1.$$
- Pad and stride per-axis; height and width are independent.
:::
