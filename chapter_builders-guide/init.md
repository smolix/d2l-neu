```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Initialization
:label:`sec_init_param`

Every neural network we have trained so far started from random weights, usually
drawn by the framework's default initializer. Yet :numref:`sec_numerical_stability`
showed just how unforgiving that starting point can be: if each layer slightly
amplifies or attenuates activations, a deep stack compounds the factor
exponentially, stalling training before it even begins.

That section also derived the remedy for a single layer: draw weights with
standard deviation $g/\sqrt{n_\textrm{in}}$, where $n_\textrm{in}$ is the
layer's input width and the *gain* $g$ compensates for the activation function
that follows—$g = \sqrt{2}$ before a ReLU (He initialization,
:numref:`subsec_he_init`) and $g = 1$ for a linear output. Under that rule, each
layer preserves the mean square of its input.

You might expect that applying this rule to every layer settles initialization
once and for all. Surprisingly, it does not. In a residual network, activations
still explode with depth even when every individual layer obeys He
initialization! Each residual block adds the output of its body $f$
(:numref:`sec_model_construction`) to its input. If the body preserves the input
mean square and produces an output uncorrelated with the input, their sum has
*double* the mean square. Across 32 residual blocks, the activation standard
deviation balloons by $2^{16} \approx 65{,}000$ before a single gradient step is
taken.

Sound initialization therefore depends on the *architecture's wiring*, not just
individual layer types, and we need clean mechanisms for applying custom rules
across a module tree. The first half of this section covers the mechanics: applying initializers
across an entire module tree, bounding extreme draws with truncated normals,
writing custom initializers, and editing weights in place. Slow down for the second half, where
we analyze variance growth across residual blocks and fix it by scaling down or
zeroing each block's output projection.

```{.python .input #init-initialization}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #init-initialization}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #init-initialization}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #init-initialization}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```


## Applying Initializers

As we saw in :numref:`sec_numerical_stability`, no framework defaults to He
initialization (:numref:`subsec_he_init`) for linear layers: Gluon draws from
$U(-0.07, 0.07)$, PyTorch from $U(-1/\sqrt{n_\textrm{in}}, 1/\sqrt{n_\textrm{in}})$
(one-sixth of He's variance), Keras from Glorot uniform, and NNX from LeCun
normal (half of He's variance). To apply He initialization—or the
depth-dependent residual scaling we derive shortly—across an entire model, we
walk the module tree (:numref:`sec_model_construction`). Let's compare default
versus He initialization on a 10-layer ReLU stack of width 256 (without biases)
fed 1,024 standard-normal inputs.

:begin_tab:`mxnet`
`net.initialize(init.MSRAPrelu(factor_type='in', slope=0), force_reinit=True)` applies He initialization ($g = \sqrt{2}$) across the block tree, overwriting existing weights:
:end_tab:

:begin_tab:`pytorch`
`net.apply(init_he)` invokes `init_he` on every submodule, using the in-place `nn.init.kaiming_normal_` (trailing underscore) so existing `nn.Parameter` references remain intact:
:end_tab:

:begin_tab:`tensorflow`
To reinitialize an existing Keras model, we loop over `model.layers` and `assign` a fresh draw in place—instantiating a new `initializer_class()` per layer because an unseeded Keras initializer instance replays the same numbers if reused:
:end_tab:

:begin_tab:`jax`
We walk `nnx.iter_modules(model)`, split a fresh random key per `nnx.Linear` layer, and update `layer.kernel[...]` in place with `nnx.initializers.he_normal()`:
:end_tab:
```{.python .input #init-applying-initializers-1}
%%tab mxnet
net = nn.Sequential()
for _ in range(10):
    net.add(nn.Dense(256, activation='relu', use_bias=False, in_units=256))
net.initialize()
X = mnp.random.normal(size=(1024, 256))
print(f'default: mean square of the output {float((net(X) ** 2).mean()):.2g}')

net.initialize(init.MSRAPrelu(factor_type='in', slope=0), force_reinit=True)
print(f'He:      mean square of the output {float((net(X) ** 2).mean()):.2g}')
```

```{.python .input #init-applying-initializers-1}
%%tab pytorch
net = nn.Sequential(*[module for _ in range(10) for module in
                      (nn.Linear(256, 256, bias=False), nn.ReLU())])
X = torch.randn(1024, 256)
print(f'default: mean square of the output {net(X).pow(2).mean():.2g}')

def init_he(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

net.apply(init_he)
print(f'He:      mean square of the output {net(X).pow(2).mean():.2g}')
```

```{.python .input #init-applying-initializers-1}
%%tab tensorflow
net = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', use_bias=False)
    for _ in range(10)])
X = tf.random.normal((1024, 256))
print(f'default: mean square of the output '
      f'{float(tf.reduce_mean(net(X) ** 2)):.2g}')

def reinit(model, initializer_class):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            init = initializer_class()   # one object per layer
            layer.kernel.assign(init(layer.kernel.shape))

reinit(net, tf.keras.initializers.HeNormal)
print(f'He:      mean square of the output '
      f'{float(tf.reduce_mean(net(X) ** 2)):.2g}')
```

```{.python .input #init-applying-initializers-1}
%%tab jax
rngs = nnx.Rngs(d2l.get_key())
net = nnx.Sequential(*[module for _ in range(10) for module in
                       (nnx.Linear(256, 256, use_bias=False, rngs=rngs),
                        nnx.relu)])
X = jax.random.normal(d2l.get_key(), (1024, 256))
print(f'default: mean square of the output {(net(X) ** 2).mean():.2g}')

def reinit(model, key, init_fn):
    layers = [m for _, m in nnx.iter_modules(model)
              if isinstance(m, nnx.Linear)]
    for layer, subkey in zip(layers, jax.random.split(key, len(layers))):
        layer.kernel[...] = init_fn(
            subkey, layer.kernel.shape, layer.kernel.dtype)

reinit(net, d2l.get_key(), nnx.initializers.he_normal())
print(f'He:      mean square of the output {(net(X) ** 2).mean():.2g}')
```

Under the default initializer, the output mean square collapses by orders of
magnitude because every layer multiplies variance by a factor strictly less than
one. Switching to He initialization brings the output mean square right back to
within roughly a factor of two of 1 (the input mean square)—the residual spread
coming simply from random finite-width fluctuations compounding across ten
layers.

:begin_tab:`pytorch`
This pattern is the whole interface. Later chapters package it as helper
functions, such as `init_cnn` for convolutional networks, but each is a
function like `init_he` handed to `apply`.
:end_tab:

### Truncated Normals
:label:`subsec_truncated_normals`

A standard Gaussian gets the variance right, but its tails extend to infinity.
For a tiny layer that rarely matters; across the $10^8$ weights of a large
model, however, dozens of entries are virtually guaranteed to land beyond five
standard deviations. To avoid extreme initial outliers, modern language and
vision models such as BERT and the Vision Transformer draw from a *truncated
normal* that rejects and redraws samples outside a fixed multiple of $\sigma$
:cite:`Devlin.Chang.Lee.ea.2018,Dosovitskiy.Beyer.Kolesnikov.ea.2021`. Clipping
the tails also trims the variance slightly, which some framework implementations
compensate for by rescaling and others leave as is.

:begin_tab:`mxnet`
Because `mxnet.init` has no built-in truncated normal, we redraw any entries outside $\pm 2\sigma$ in a loop:
:end_tab:

:begin_tab:`pytorch`
`nn.init.trunc_normal_` takes *absolute* cutoffs `a` and `b` (defaulting to $\pm 2$), so truncating at $\pm 2\sigma$ for `std=0.02` requires passing `a=-0.04, b=0.04`:
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.initializers.TruncatedNormal` truncates at $\pm 2\sigma$ without rescaling (whereas `HeNormal` truncates and rescales to hit target variance):
:end_tab:

:begin_tab:`jax`
`nnx.initializers.truncated_normal` takes cutoffs in multiples of $\sigma$ (`lower=-2.0, upper=2.0` by default):
:end_tab:
```{.python .input #init-truncated-normals}
%%tab mxnet
def trunc_normal(sigma, shape):
    w = mnp.random.normal(0, sigma, shape)
    tails = mnp.abs(w) > 2 * sigma
    while bool(tails.any()):
        w = mnp.where(tails, mnp.random.normal(0, sigma, shape), w)
        tails = mnp.abs(w) > 2 * sigma
    return w

w = mnp.random.normal(0, 0.02, (1000, 1000))
print(f'normal:    std {float(w.std()):.4f}, '
      f'max weight {float(mnp.abs(w).max()):.4f}')
w = trunc_normal(0.02, (1000, 1000))
print(f'truncated: std {float(w.std()):.4f}, '
      f'max weight {float(mnp.abs(w).max()):.4f}')
```

```{.python .input #init-truncated-normals}
%%tab pytorch
w = torch.empty(1000, 1000)
nn.init.normal_(w, std=0.02)
print(f'normal:    std {w.std():.4f}, max weight {w.abs().max():.4f}')
nn.init.trunc_normal_(w, std=0.02, a=-0.04, b=0.04)
print(f'truncated: std {w.std():.4f}, max weight {w.abs().max():.4f}')
```

```{.python .input #init-truncated-normals}
%%tab tensorflow
w = tf.keras.initializers.RandomNormal(stddev=0.02)((1000, 1000)).numpy()
print(f'normal:    std {w.std():.4f}, max weight {abs(w).max():.4f}')
w = tf.keras.initializers.TruncatedNormal(stddev=0.02)((1000, 1000)).numpy()
print(f'truncated: std {w.std():.4f}, max weight {abs(w).max():.4f}')
```

```{.python .input #init-truncated-normals}
%%tab jax
w = nnx.initializers.normal(stddev=0.02)(d2l.get_key(), (1000, 1000))
print(f'normal:    std {w.std():.4f}, max weight {jnp.abs(w).max():.4f}')
w = nnx.initializers.truncated_normal(stddev=0.02)(d2l.get_key(), (1000, 1000))
print(f'truncated: std {w.std():.4f}, max weight {jnp.abs(w).max():.4f}')
```

Out of one million standard normal draws with $\sigma = 0.02$, extreme weights
approach $5\sigma = 0.1$, whereas the truncated normal guarantees
$|w| \leq 0.04$. Clipping the tails lowers the empirical standard deviation from
$0.02$ to roughly $0.0176$. Fixed-$\sigma$ recipes like BERT simply accept that
slight reduction, while variance-preserving schemes rescale after truncation to
hit the exact target variance.

## Custom Initializers and Direct Edits

Whenever built-in initializers do not match what a paper or architecture calls
for, we can write our own initializer—or edit parameter tensors directly. Both
build on each framework's initializer interface:

:begin_tab:`mxnet`
In Gluon, an initializer is a subclass of `init.Initializer` whose
`_init_weight` method fills the array it is handed, in place.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, an initializer is a function that overwrites a parameter in place.
:end_tab:

:begin_tab:`tensorflow`
In Keras, an initializer is an object that maps `(shape, dtype)` to a tensor:
subclass `tf.keras.initializers.Initializer` and implement `__call__`. Layers
then accept an instance through their initializer arguments.
:end_tab:

:begin_tab:`jax`
In NNX, an initializer is any function with the signature `(key, shape, dtype)
-> array`.
:end_tab:

### A Rule with a Gain of Our Choice

Let's implement the gain-parameterized normal rule from the introduction: draw
weights with standard deviation $g/\sqrt{n_\textrm{in}}$ for any chosen gain
$g$. He initialization corresponds to $g = \sqrt{2}$, and our residual-network
experiments below will also test $g = 1$, $g = 1/\sqrt{N}$ (for $N$ blocks), and
$g = 0$ on each block's output projection. Writing one reusable initializer
handles all four cases cleanly.

:begin_tab:`mxnet`
The subclass stores the gain. A Gluon weight has shape
$(n_\textrm{out}, n_\textrm{in})$:
:end_tab:

:begin_tab:`pytorch`
The function follows the convention of `torch.nn.init`: it overwrites the
tensor it is given, and its name ends in an underscore. A PyTorch weight has
shape $(n_\textrm{out}, n_\textrm{in})$:
:end_tab:

:begin_tab:`tensorflow`
The subclass stores the gain. A Keras kernel has shape
$(n_\textrm{in}, n_\textrm{out})$:
:end_tab:

:begin_tab:`jax`
A function of the gain returns the initializer. An NNX kernel has shape
$(n_\textrm{in}, n_\textrm{out})$:
:end_tab:

```{.python .input #init-custom-initializers-1}
%%tab mxnet
class ScaledNormal(init.Initializer):
    def __init__(self, gain):
        super().__init__()
        self.gain = gain

    def _init_weight(self, name, data):
        std = self.gain / math.sqrt(data.shape[1])
        data[:] = std * mnp.random.normal(size=data.shape)
```

```{.python .input #init-custom-initializers-1}
%%tab pytorch
def scaled_normal_(weight, gain):
    nn.init.normal_(weight, std=gain / math.sqrt(weight.shape[1]))
```

```{.python .input #init-custom-initializers-1}
%%tab tensorflow
class ScaledNormal(tf.keras.initializers.Initializer):
    def __init__(self, gain):
        self.gain = gain

    def __call__(self, shape, dtype=None):
        std = self.gain / math.sqrt(shape[0])
        return std * tf.random.normal(shape, dtype=dtype or tf.float32)
```

```{.python .input #init-custom-initializers-1}
%%tab jax
def scaled_normal(gain):
    def init(key, shape, dtype=jnp.float32):
        std = gain / math.sqrt(shape[0])
        return std * jax.random.normal(key, shape, dtype)
    return init
```

### Direct Edits

Occasionally we want to modify parameter values directly—either right after
initialization or after loading weights. Because ReLU is positively homogeneous,
halving the final layer's weights in our ten-layer stack halves every output
activation and should therefore reduce the output mean square by a factor of
four.

:begin_tab:`mxnet`
`weight.data()[:] *= 0.5` edits the underlying array in place outside `autograd.record()`.
:end_tab:

:begin_tab:`pytorch`
Wrap in-place edits on leaf `nn.Parameter` tensors in `with torch.no_grad():` so autograd does not reject them.
:end_tab:

:begin_tab:`tensorflow`
`w.assign(0.5 * w)` updates the `tf.Variable` in place outside any `GradientTape`.
:end_tab:

:begin_tab:`jax`
`kernel[...] *= 0.5` mutates the array inside the tracked `nnx.Param` variable.
:end_tab:
```{.python .input #init-custom-initializers-2}
%%tab mxnet
before = float((net(X) ** 2).mean())
net[-1].weight.data()[:] *= 0.5
print(f'{before:.2g} before, {float((net(X) ** 2).mean()):.2g} after')
```

```{.python .input #init-custom-initializers-2}
%%tab pytorch
before = net(X).pow(2).mean()
with torch.no_grad():
    net[-2].weight *= 0.5
print(f'{before:.2g} before, {net(X).pow(2).mean():.2g} after')
```

```{.python .input #init-custom-initializers-2}
%%tab tensorflow
before = float(tf.reduce_mean(net(X) ** 2))
w = net.layers[-1].kernel
w.assign(0.5 * w)
print(f'{before:.2g} before, {float(tf.reduce_mean(net(X) ** 2)):.2g} after')
```

```{.python .input #init-custom-initializers-2}
%%tab jax
before = (net(X) ** 2).mean()
net.layers[-2].kernel[...] *= 0.5
print(f'{before:.2g} before, {(net(X) ** 2).mean():.2g} after')
```

Sure enough, the output mean square drops to exactly one-quarter of its prior
value.

## Initialization of Residual Networks

### Variance Growth across Blocks

Earlier we claimed that a deep residual stack explodes at initialization even
when every individual layer follows He's rule. Let's prove it. A residual block
computes $\mathbf{x}_{k+1} = \mathbf{x}_k + f_k(\mathbf{x}_k)$. Take the
two-layer body from :numref:`sec_model_construction`, expanded fourfold in the
middle as in many modern residual blocks and transformers: a linear layer from
width $d$ to $4d$, a ReLU, and a linear *output projection* from $4d$ back to
$d$. All biases start at zero. The first layer uses He initialization
($g = \sqrt{2}$) because a ReLU follows it, while the output projection uses
standard deviation $g/\sqrt{4d}$ for a gain $g$ that we leave free ($g = 1$
being the standard linear-layer rule from :numref:`sec_numerical_stability`).

Let $m_k$ denote the mean square of the entries of $\mathbf{x}_k$. Passing
through the first linear layer doubles $m_k$, and the ReLU halves it back to
$m_k$ (:numref:`subsec_he_init`). The output projection then scales the mean
square by $4d \cdot g^2/(4d) = g^2$, so the body output $f_k(\mathbf{x}_k)$ has
mean square $g^2 m_k$. Now comes the crucial step: because the output
projection's weights are zero-mean and independent of $\mathbf{x}_k$, the body
output $f_k(\mathbf{x}_k)$ is uncorrelated with the skip signal $\mathbf{x}_k$.
Their mean squares therefore add:

$$m_{k+1} = m_k + g^2 m_k = (1 + g^2)\, m_k.$$

Every block inflates the activation mean square by $1 + g^2 > 1$ whenever
$g \neq 0$. At $g = 1$, each block doubles the variance, so across $N$ blocks
the activation standard deviation grows by $2^{N/2}$: $2\times$ after 2 blocks,
$16\times$ after 8 blocks, and $65{,}536\times$ after 32 blocks. Every single
layer respected the single-layer variance rule, yet the network as a whole
blew up because that local rule is blind to the skip connection adding the input
back at every block.

### Scaling Down the Output Projection

Across $N$ blocks, the per-block factor compounds to $(1 + g^2)^N$. Keeping
activations well scaled at arbitrary depth therefore requires shrinking $g$ as
$N$ grows. Choosing $g^2 = 1/N$—scaling every body's output projection by
$1/\sqrt{N}$—does the trick:

$$m_N = \left(1 + \frac{1}{N}\right)^N m_0 \leq e\, m_0,$$

because $(1 + 1/N)^N$ increases monotonically to $e$. The activation standard
deviation now grows by at most $\sqrt{e} \approx 1.65$ regardless of depth
($1.5$ after 2 blocks, $1.60$ after 8, and $1.64$ after 32). This exact rule is
widely used in practice: GPT-2 :cite:`Radford.Wu.Child.ea.2019`, for example,
scales the output projection of every residual *branch* by $1/\sqrt{N}$, where
$N$ is the total number of residual branches in the network.

### Zero Output Projections

What if we push this idea to its limit and set $g = 0$? At first glance, zero
initialization sounds dangerous: zeroing *every* weight traps all hidden units
in identical symmetry forever (:numref:`sec_numerical_stability`). Zeroing *only*
the output projection of each residual body is completely different. At step
zero, every body outputs zero, so every block acts as an exact identity map and
the entire network starts out cleanly behaved while depth engages gradually
during training. Symmetry never freezes learning because the body's first layer
retains its random weights: on the first backward pass, the output projection
receives a nonzero gradient (since its ReLU inputs are nonzero), moving away
from zero immediately, and from the second step onward gradients flow freely
into the first layer as well.

### Measuring Variance across Depth
:label:`subsec_variance_across_depth`

Let's test all three predictions empirically. `WideResidualBlock` implements our
$d \to 4d \to d$ residual block using our custom `ScaledNormal` initializer—gain
$\sqrt{2}$ on the expansion layer and a configurable `out_gain` on the output
projection:

```{.python .input #init-watching-the-variance-compound-1}
%%tab mxnet
class WideResidualBlock(nn.Block):
    def __init__(self, d, out_gain):
        super().__init__()
        self.body = nn.Sequential()
        self.body.add(
            nn.Dense(4 * d, activation='relu',
                     weight_initializer=ScaledNormal(math.sqrt(2))),
            nn.Dense(d, weight_initializer=ScaledNormal(out_gain)))

    def forward(self, X):
        return X + self.body(X)
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab pytorch
class WideResidualBlock(nn.Module):
    def __init__(self, d, out_gain):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(),
                                  nn.Linear(4 * d, d))
        for layer, gain in ((self.body[0], math.sqrt(2)),
                            (self.body[2], out_gain)):
            scaled_normal_(layer.weight, gain)
            nn.init.zeros_(layer.bias)

    def forward(self, X):
        return X + self.body(X)
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab tensorflow
class WideResidualBlock(tf.keras.layers.Layer):
    def __init__(self, d, out_gain, **kwargs):
        super().__init__(**kwargs)
        self.body = tf.keras.Sequential([
            tf.keras.layers.Dense(
                4 * d, activation='relu',
                kernel_initializer=ScaledNormal(math.sqrt(2))),
            tf.keras.layers.Dense(
                d, kernel_initializer=ScaledNormal(out_gain))])

    def call(self, X):
        return X + self.body(X)
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab jax
class WideResidualBlock(nnx.Module):
    def __init__(self, d, out_gain, *, rngs):
        self.body = nnx.Sequential(
            nnx.Linear(d, 4 * d, kernel_init=scaled_normal(math.sqrt(2)),
                       rngs=rngs),
            nnx.relu,
            nnx.Linear(4 * d, d, kernel_init=scaled_normal(out_gain),
                       rngs=rngs))

    def __call__(self, X):
        return X + self.body(X)
```

Next, `std_across_depth` builds a stack of `num_blocks` blocks of width 64,
feeds in 256 $\mathcal{N}(0, 1)$ inputs, and tracks the activation standard
deviation at the input and after every block:

```{.python .input #init-watching-the-variance-compound-2}
%%tab mxnet
def std_across_depth(num_blocks, out_gain):
    net = nn.Sequential()
    net.add(*[WideResidualBlock(64, out_gain) for _ in range(num_blocks)])
    net.initialize()  # deferred: the draws happen at the forward pass
    X = mnp.random.normal(0, 1, (256, 64))
    stds = [float(X.std())]
    for block in net:
        X = block(X)
        stds.append(float(X.std()))
    return stds
```

```{.python .input #init-watching-the-variance-compound-2}
%%tab pytorch
def std_across_depth(num_blocks, out_gain):
    net = nn.Sequential(*[WideResidualBlock(64, out_gain)
                          for _ in range(num_blocks)])
    X = torch.randn(256, 64)
    stds = [X.std().item()]
    with torch.no_grad():
        for block in net:
            X = block(X)
            stds.append(X.std().item())
    return stds
```

```{.python .input #init-watching-the-variance-compound-2}
%%tab tensorflow
def std_across_depth(num_blocks, out_gain):
    net = tf.keras.Sequential([WideResidualBlock(64, out_gain)
                               for _ in range(num_blocks)])
    X = tf.random.normal((256, 64))
    stds = [float(tf.math.reduce_std(X))]
    for block in net.layers:
        X = block(X)
        stds.append(float(tf.math.reduce_std(X)))
    return stds
```

```{.python .input #init-watching-the-variance-compound-2}
%%tab jax
def std_across_depth(num_blocks, out_gain):
    rngs = nnx.Rngs(d2l.get_key())
    net = nnx.Sequential(*[WideResidualBlock(64, out_gain, rngs=rngs)
                           for _ in range(num_blocks)])
    X = jax.random.normal(rngs(), (256, 64))
    stds = [float(X.std())]
    for block in net.layers:
        X = block(X)
        stds.append(float(X.std()))
    return stds
```

We compare all three gain schedules—unscaled ($g = 1$), scaled
($g = 1/\sqrt{N}$), and zero ($g = 0$)—printing final standard deviations for
$N = 2, 8, 32$ and plotting the full depth profile at $N = 32$:

```{.python .input #init-watching-the-variance-compound-3}
gains = {'unscaled': lambda n: 1.0, 'scaled': lambda n: n ** -0.5,
         'zero': lambda n: 0.0}
profiles = {n: [std_across_depth(n, gain(n)) for gain in gains.values()]
            for n in (2, 8, 32)}
print(f'{"N":>3}' + ''.join(f'{name:>10}' for name in gains))
for n, stds in profiles.items():
    print(f'{n:>3}' + ''.join(f'{s[-1]:>10.3g}' for s in stds))
d2l.plot(list(range(33)), profiles[32], 'blocks', 'std of activations',
         legend=list(gains), yscale='log', figsize=(4.5, 3))
```

The empirical curves confirm our analysis. The unscaled stack grows by roughly
$\sqrt{2}$ per block—a clean straight line on the log scale—reaching anywhere
from about 20,000 to 200,000 after 32 blocks (most individual runs sit slightly
below the theoretical $\sqrt{\mathbb{E}[m_{32}]} = 65{,}536$ because multiplying
random positive factors produces a right-skewed distribution whose median falls
below the square root of its mean). With initial activations already in the tens
of thousands, there is little point even attempting to train the unscaled stack.
By contrast, the $1/\sqrt{N}$-scaled stack stays comfortably between 1.4 and 1.9
at every depth—right around our 1.5–1.65 prediction—while zeroing the output
projections holds the standard deviation flat at 1 across all 32 blocks.

Running a quick forward-pass diagnostic like this catches broken initializations
immediately before wasting GPU time on training. Notice, though, that it cannot
tell you whether $1/\sqrt{N}$ scaling or zero initialization will optimize
faster once training begins: both preserve signal scale at step zero, differing
only in whether the residual branches start active or wake up over the first few
updates.

Finally, remember that initialization controls only step zero. As gradient steps
update the weights, nothing in initialization alone guarantees that activations
stay well scaled—which is where normalization layers (:numref:`sec_batch_norm`)
step in.

## Summary

Framework default initializers do not use He initialization, so deep networks
without normalization layers require explicit initialization—applied via
constructor arguments, tree-wide callbacks, or direct in-place edits—and
truncated normals prevent rare extreme outliers in large models. Crucially,
per-layer variance rules (:numref:`sec_numerical_stability`) are not enough for
residual architectures: adding each block's body back to its input multiplies
the activation mean square by $1 + g^2$, causing $g = 1$ to blow up by
$\sqrt{2}$ per block. Scaling each block's output projection by $1/\sqrt{N}$
caps activation growth at $\sqrt{e}$ for any depth $N$, while zeroing the output
projection starts every block as the identity without trapping the network in
symmetry.

## Exercises

1. **Truncated normal.** For a normal distribution truncated at
   $\pm 2\sigma$: what fraction of draws does the cut discard, and by how
   much does it shrink the standard deviation? Neither quantity has an
   elementary closed form; express both in terms of the standard normal
   density and distribution function and evaluate them using tabulated
   values. Verify the standard deviation against the printed output of the
   demonstration in :numref:`subsec_truncated_normals`, and the discarded
   fraction by counting the entries of its plain normal draw that lie
   beyond $\pm 2\sigma$.

2. [code] **Normalized inputs.** Many residual networks normalize the
   input of each body: the body reads $\mathbf{x}_k / r_k$, where $r_k$
   is the root mean square of the entries of $\mathbf{x}_k$ for each
   example, instead of $\mathbf{x}_k$.
    1. Show that the body output then has mean square $g^2$, whatever the
       scale of $\mathbf{x}_k$, and that with $m_0 = 1$ the mean square
       after $N$ blocks is $1 + N g^2$. How does the growth with depth
       compare with the unnormalized blocks of this section?
    1. Which gain keeps the mean square at the end of the stack equal to 2
       for every $N$? Compare it with the scaling of GPT-2.
    1. Add the normalization to `WideResidualBlock`, measure the standard
       deviation across depth for $g = 1$ and $g = 1/\sqrt{N}$ at $N = 32$,
       and compare with your predictions and with the
       measured plot in :numref:`subsec_variance_across_depth`.

3. [code] **Kaiming initialization.** Implement the Kaiming (He)
   initializers of :numref:`subsec_he_init`, uniform and normal, with an
   explicit `nonlinearity` argument that switches the gain between `'relu'`
   and `'linear'`.
    1. Check both on the weight of a layer with 512 inputs and 256
       outputs by mean, standard deviation, and range: the normal
       initializer against the scaled normal initializer of this
       section, and the uniform initializer against its
       predicted bound and standard deviation. Explain any difference in
       range.
    1. Apply your normal initializer to the stack of
       :numref:`subsec_variance_across_depth` with the `'relu'` gain on
       both layers of every block. Predict the factor by which each block
       multiplies the mean square, and check it against a measured curve
       added to the plot of :numref:`subsec_variance_across_depth`.

    *Adapted from CMU 10-714, [Homework 2](https://github.com/dlsyscourse/hw2)
    and [Homework 4](https://github.com/dlsyscourse/hw4).*

4. [code] **Gradients across depth.** ● The backward pass through a block
   multiplies the gradient by the transposed Jacobian of
   $\mathbf{x} \mapsto \mathbf{x} + f(\mathbf{x})$.
    1. Repeat the argument of this section for the backward pass. Show that
       the body multiplies the mean square of the gradient by $g^2$,
       although the backward pass sums over the outputs of each layer
       rather than its inputs, so that each block again multiplies it by
       $1 + g^2$.
    1. Verify the prediction for the three choices of gain at $N = 32$:
       compute the gradient of the sum of the outputs of the stack with
       respect to its input, and compare its mean square with that of the
       gradient at the output.

5. [code] **Data-driven calibration.** ● Initialize the stack of
   :numref:`subsec_variance_across_depth` with the unscaled choice of
   gain, then calibrate it block by block on one minibatch of
   unit-variance inputs.
   For each block in turn, run the minibatch forward to that block,
   measure the standard deviation of its body output
   $f_k(\mathbf{x}_k)$, and rescale the block's output projection so that
   this standard deviation becomes $1/\sqrt{N}$. This adapts the
   layer-sequential unit-variance initialization of
   :citet:`Mishkin.Matas.2016` to a residual stack.
    1. Explain why the blocks must be calibrated in order rather than all
       at once from a single forward pass.
    1. Explain why the calibration targets the body output rather than
       the block output $\mathbf{x}_k + f_k(\mathbf{x}_k)$.
    1. Predict the standard deviation at the end of the calibrated stack.
       Plot the calibrated depth-versus-standard-deviation curve next to
       that of the scaled choice of gain, and explain the difference.


[Discussions](https://d2l.discourse.group/t/36754)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5.3]{.kicker}

Initialization<br>**library defaults · initializers over the tree · custom initializers · residual scaling**
:::
:::

::: {.slide title="He in Every Layer, Yet 32 Residual Blocks Grow Activations 65,000-Fold"}
[The problem]{.kicker}

- One layer: weights with standard deviation $g/\sqrt{n_\textrm{in}}$ keep the
  mean square of the activations; gain $g = \sqrt{2}$ before a ReLU (He),
  $g = 1$ when no nonlinearity follows.
- A residual block adds the output of its body $f_k$ to its input:
  $\mathbf{x}_{k+1} = \mathbf{x}_k + f_k(\mathbf{x}_k)$.
- Body and input of equal mean square, uncorrelated: the sum has twice
  the mean square.

::: {.d2l-note .warn}
32 blocks: the standard deviation grows by $2^{16} \approx 65{,}000$ before
training starts.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- check what your framework draws when you choose nothing;
- apply an initializer to a whole model or to selected parameters;
- write an initializer of your own;
- derive the factor $1 + g^2$ by which a residual block multiplies the mean
  square;
- choose the gain of the body's output projection for depth $N$.

**Running example:** a stack of $N \le 32$ residual blocks of width 64,
body $64 \to 256 \to 64$, fed 256 examples from $\mathcal{N}(0, 1)$.
:::

::: {.slide title="Choosing Starting Values, Between the State Tree and Custom Layers"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Modules and Model Construction | a model is a tree of modules, built from a config |
| Parameters, State, and Memory | parameters, buffers, bytes per parameter |
| **Initialization** | **library defaults, custom initializers, residual depth** |
| Custom Layers and Functions | new layers, state, and custom gradients |
| Numerical Formats | dtypes, mixed precision, loss scaling |
| Saving, Loading, and Pretrained Weights | safe files, checkpoints, pretrained weights |
| GPUs, Devices, and Memory | placement, asynchrony, training memory |
| Reproducibility and Inspection | seeds, determinism, hooks |
:::

::: {.col .narrow}
- **Uses:** Xavier and He initialization; the variance of a sum of
  independent terms; the module tree.
- **Skips:** normalization layers (Modern Convnets); what happens during
  training.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Defaults and Initializers]{.dtitle}

[what you get, and how to change it]{.dsub}
:::
:::

::: {.slide title="He Initialization Restores the Scale of Ten ReLU Layers"}
[Applying initializers]{.kicker}

The same ten layers, fed inputs of mean square 1. Mean square of the
output under the default, then after He initialization of every layer:

@init-applying-initializers-1
:::

::: {.slide title="An Unseeded Keras Initializer Returns the Same Values on Every Call" only="tensorflow"}
[Applying initializers]{.kicker}

- `RandomNormal(stddev=0.01)` without a seed draws its numbers once and
  repeats them on every call.
- Two layers sharing one object draw the same random numbers: layers of
  equal shape start identical.
- Create one initializer object per layer: a function that returns a new
  one, or a class handed to a loop that instantiates it per layer.

::: {.d2l-note .warn}
Shared initializer objects silently correlate layers: no error, no warning.
:::
:::

::: {.slide title="Truncated Normals Bound the Largest Weight"}
[Truncated normals]{.kicker}

- Among $10^8$ normal weights, dozens lie beyond $5\sigma$.
- Truncating at $2\sigma$ caps $|w|$ and lowers the standard deviation by
  the factor 0.88.
- BERT and the vision transformer use truncated normals.

$\sigma = 0.02$, a million draws, plain and truncated at $\pm 0.04$:

@!init-truncated-normals
:::

::: {.slide title="Check: Truncate, Then Rescale: How Large Can a Weight Get?"}
[Check]{.kicker}

Draw from $\mathcal{N}(0, 0.02^2)$, cut at $\pm 2\sigma$, then rescale so
that the standard deviation is 0.02 again. What is the largest possible
weight?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The cut lowers the standard deviation by the factor 0.88, so
the rescaling multiplies by $1/0.88 \approx 1.14$: the bound
$0.04$ becomes about $0.0455$, or $2.27\sigma$. A truncated LeCun normal
initializer does this.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Custom Initializers and Direct Edits]{.dtitle}

[a gain of our choice · direct edits]{.dsub}
:::
:::

::: {.slide title="A Custom Initializer Is a Few Lines of Code"}
[A custom initializer]{.kicker}

Normal with standard deviation $g/\sqrt{n_\textrm{in}}$ for any gain $g$
($g = \sqrt{2}$ is He); the residual blocks below use it with $g = 1$,
$1/\sqrt{N}$, and 0:

@-init-custom-initializers-1
:::

::: {.slide title="Parameters Can Also Be Edited in Place"}
[Direct edits]{.kicker}

Halve the weights of the last layer of the ReLU stack. The outputs halve,
so their mean square should fall by a factor of four:

@init-custom-initializers-2
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Residual Networks]{.dtitle}

[the factor 1 + g² per block]{.dsub}
:::
:::

::: {.slide title="The Body Passes On g² Times the Mean Square"}
[Variance growth]{.kicker}

Body: $d \to 4d$ (He, gain $\sqrt 2$), ReLU, $4d \to d$ (the *output
projection*, gain $g$); biases zero. Let $m_k$ be the mean square of
$\mathbf{x}_k$.

- First layer and ReLU pass on $m_k$ (He).
- Output projection: $4d$ inputs times variance $g^2/(4d)$: factor $g^2$.
- Body output has mean square $g^2 m_k$.
:::

::: {.slide title="Each Block Multiplies the Mean Square by 1 + g²"}
[Variance growth]{.kicker}

The body output is uncorrelated with $\mathbf{x}_k$: the projection's
weights have mean zero and are independent of $\mathbf{x}_k$. Mean squares
add:

$$m_{k+1} = m_k + g^2 m_k = (1 + g^2)\, m_k.$$

| $g$ | per block | std after $N$ blocks |
|---|---|---|
| 1 | $\times 2$ | $2^{N/2}$: 2, 16, 65,536 for $N = 2, 8, 32$ |
| $1/\sqrt N$ | $\times (1 + 1/N)$ | at most $\sqrt e \approx 1.65$ |
| 0 | $\times 1$ | 1 |

::: {.d2l-note .warn}
Every layer follows the per-layer rule; the skip connection is what it
ignores.
:::
:::

::: {.slide title="Check: Which Gain Keeps 100 Blocks within a Factor of 3?"}
[Check]{.kicker}

Find the largest gain $g$ for which the mean square after $N = 100$ blocks
is at most 3 times that of the input. Compare with $1/\sqrt{N}$.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $(1 + g^2)^{100} \le 3$ needs $g^2 \le \log 3 / 100 \approx
0.011$, so $g \le 0.105$. The choice $g = 1/\sqrt{100} = 0.1$ gives
$1.01^{100} \approx 2.70$.
:::
:::
:::

::: {.slide title="Scaling the Output Projection by 1/√N Bounds the Growth by √e"}
[Scaling down the output projection]{.kicker}

With $g^2 = 1/N$:

$$m_N = \left(1 + \frac{1}{N}\right)^N m_0 \le e\, m_0,$$

so the standard deviation grows by at most $\sqrt e \approx 1.65$ at any
depth: 1.5 for $N = 2$, 1.60 for 8, 1.64 for 32.

::: {.d2l-note}
GPT-2 scales the output projection of every residual body by
$1/\sqrt{N}$, with $N$ the number of bodies.
:::
:::

::: {.slide title="A Zero Output Projection Starts Every Block as the Identity"}
[Zero output projections]{.kicker}

- $g = 0$: the body contributes nothing; the network starts shallow.
- Not the symmetry trap: the body's first layer stays random.
- First backward pass: the zero projection gets a nonzero gradient (its
  input is nonzero); the first layer gets none through it.
- Once the projection moves off zero, gradient reaches the whole body.

::: {.d2l-note .warn}
Zeroing *every* layer is fatal: all units stay identical forever.
:::
:::

::: {.slide title="Measured: Unscaled Grows √2 per Block; Scaled and Zero Stay Flat"}
[Measuring variance across depth]{.kicker}

Standard deviation after the last block, and after every block for $N = 32$:

@!init-watching-the-variance-compound-3
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Defaults** are fixed rules, not He; set the initialization
  explicitly.
- **Applying:** constructor arguments, a function over every submodule, or
  assignment to chosen parameters; He keeps ten ReLU layers at scale.
- **Custom:** truncated normals, and a scaled normal with any gain in a few
  lines.
- **Residual blocks** multiply the mean square by $1 + g^2$: with $g = 1$
  the standard deviation after 32 blocks is predicted 65,536, measured
  between 20,000 and 200,000.
- **Scaling** the output projection by $1/\sqrt N$ bounds the growth by
  $\sqrt e \approx 1.65$; **zero** makes every block the identity.

**Next:** layers the library does not provide.
:::
