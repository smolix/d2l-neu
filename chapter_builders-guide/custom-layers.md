```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Custom Layers and Functions
:label:`sec_custom_layer`

The library ships over a hundred layers, yet every one of them started life as
code somebody wrote because the layer they needed did not exist. Sooner or
later you will be in the same position: a new normalization, an unusual
residual block, an operation with no gradient. This section shows what to do.
A custom layer is a subclass of the module class from
:numref:`sec_model_construction` with a forward method, and if you
register its state properly you inherit everything a built-in layer gets:
parameter tracking, serialization, and device movement, with no extra code. We
build up from a stateless five-liner to RMSNorm, a normalization used by many
current language models, then to layers with precomputed non-trainable state,
and finally to the case where the forward computation alone is not enough
because the gradient itself must be redefined.

```{.python .input #custom-layers-custom-layers-and-functions}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input #custom-layers-custom-layers-and-functions}
%%tab jax
import jax
from jax import numpy as jnp
from flax import nnx
from flax import serialization
from d2l import jax as d2l
```

```{.python .input #custom-layers-custom-layers-and-functions}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input #custom-layers-custom-layers-and-functions}
%%tab mxnet
import os
import tempfile
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

## Layers without Parameters

:begin_tab:`pytorch`
The smallest custom layer has no state at all. `CenteredLayer` subtracts the
mean from its input. To build it, we inherit from the base module class and
implement `forward`; there is nothing to set up, so `__init__` only calls the
parent constructor.
:end_tab:

:begin_tab:`jax`
The smallest custom layer has no state at all. `CenteredLayer` subtracts the
mean from its input. To build it, we inherit from the base module class and
implement `__call__`. This layer owns no variables and needs no constructor.
:end_tab:

:begin_tab:`tensorflow`
The smallest custom layer has no state at all. `CenteredLayer` subtracts the
mean from its input. To build it, we inherit from the base layer class and
implement `call`, which Keras invokes through its own `__call__`; there is
nothing to set up, so `__init__` only calls the parent constructor.
:end_tab:

:begin_tab:`mxnet`
The smallest custom layer has no state at all. `CenteredLayer` subtracts the
mean from its input. To build it, we inherit from the base block class and
implement `forward`; there is nothing to set up, so `__init__` only calls the
parent constructor.
:end_tab:

```{.python .input #custom-layers-layers-without-parameters-1}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input #custom-layers-layers-without-parameters-1}
%%tab jax
class CenteredLayer(nnx.Module):
    def __call__(self, X):
        return X - X.mean()
```

```{.python .input #custom-layers-layers-without-parameters-1}
%%tab tensorflow
class CenteredLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input #custom-layers-layers-without-parameters-1}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

Feeding data through confirms that it does what it says.

:begin_tab:`jax`
NNX modules are ordinary Python objects: calling the object invokes
`__call__` directly. Since `CenteredLayer` owns no variables, construction
needs neither an input shape nor an RNG stream.
:end_tab:

```{.python .input #custom-layers-layers-without-parameters-2}
%%tab pytorch
layer = CenteredLayer()
layer(torch.tensor([1.0, 2, 3, 4, 5]))
```

```{.python .input #custom-layers-layers-without-parameters-2}
%%tab jax
layer = CenteredLayer()
layer(jnp.array([1.0, 2, 3, 4, 5]))
```

```{.python .input #custom-layers-layers-without-parameters-2}
%%tab tensorflow
layer = CenteredLayer()
layer(tf.constant([1.0, 2, 3, 4, 5]))
```

```{.python .input #custom-layers-layers-without-parameters-2}
%%tab mxnet
layer = CenteredLayer()
layer(np.array([1.0, 2, 3, 4, 5]))
```

Nothing distinguishes this class from a built-in layer. We can place it inside
a `Sequential`, and the container neither knows nor cares that one of its
children is user code. The output mean should be zero; because we are adding
up floating-point numbers, we may see a very small nonzero value instead,
which is roundoff, not a bug.

:begin_tab:`jax`
NNX creates the `Linear` parameters in its constructor, so we specify both
feature dimensions there. `CenteredLayer` contributes no parameters.
:end_tab:

:begin_tab:`mxnet`
One gluon-specific step: parameter arrays are allocated lazily, so a network
must run `initialize()` before its first call. Our layer contributes nothing
to initialize, but the `Dense` beside it does.
:end_tab:

```{.python .input #custom-layers-layers-without-parameters-3}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()
```

```{.python .input #custom-layers-layers-without-parameters-3}
%%tab jax
net = nnx.Sequential(nnx.Linear(8, 128, rngs=nnx.Rngs(0)), CenteredLayer())
Y = net(jax.random.uniform(d2l.get_key(), (4, 8)))
Y.mean()
```

```{.python .input #custom-layers-layers-without-parameters-3}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input #custom-layers-layers-without-parameters-3}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
Y = net(np.random.uniform(size=(4, 8)))
Y.mean()
```

## Layers with Parameters: RMSNorm

:begin_tab:`pytorch`
A layer with something to learn must create its own parameters, and wrapping a
tensor in `nn.Parameter` is what registers them (:numref:`sec_parameters`).
We could show the mechanics by re-implementing `nn.Linear`, but that teaches
nothing the built-in does not already do. Instead we implement *RMSNorm*
:cite:`Zhang.Sennrich.2019`, a normalization used by many current language
models, in the same five lines.
:end_tab:

:begin_tab:`jax`
A layer with something to learn must create its own parameters, and
`nnx.Param` is what registers them in the object graph
(:numref:`sec_parameters`). We could show the mechanics by re-implementing
`nnx.Linear`, but that teaches nothing the built-in does not already do. Instead
we implement *RMSNorm* :cite:`Zhang.Sennrich.2019`, a normalization used by
many current language models, in the same handful of lines.
:end_tab:

:begin_tab:`tensorflow`
A layer with something to learn must create its own parameters, and
`add_weight` is what registers them (:numref:`sec_parameters`). Keras
splits creation off into a `build` method that runs on the first call, once
the input shape is known, so the layer need not be told its width in advance.
We could show the mechanics by re-implementing `Dense`, but that teaches
nothing the built-in does not already do. Instead we implement *RMSNorm*
:cite:`Zhang.Sennrich.2019`, a normalization used by many current language
models.
:end_tab:

:begin_tab:`mxnet`
A layer with something to learn must create its own parameters, and assigning
a `gluon.Parameter` to an attribute is what registers it
(:numref:`sec_parameters`). The parameter is declared with a name, a
shape, and an initializer; its array is allocated only when `initialize()`
runs, and the forward pass fetches the copy on the input's device with
`.data(X.device)`. We could show the mechanics by re-implementing `nn.Dense`,
but that teaches nothing the built-in does not already do. Instead we
implement *RMSNorm* :cite:`Zhang.Sennrich.2019`, a normalization used by many
current language models.
:end_tab:

Layer normalization standardizes each input vector: subtract the mean, divide
by the standard deviation, then apply a learned scale and shift. Zhang and
Sennrich observed that the re-centering contributes little and dropped it,
along with the shift. What remains is: divide by the root mean square,
multiply by a learned gain $\mathbf{g}$:

$$
\textrm{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \mathbf{g},
$$

where $d$ is the width of $\mathbf{x}$ and $\epsilon$ guards against division
by zero. One parameter vector, one reduction, no shift. Why dropping the mean
costs so little in practice is a question for the normalization discussion in
later chapters; here it is our stateful worked example.

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-1}
%%tab pytorch
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, X):
        rms = X.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.gain * X * rms
```

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-1}
%%tab jax
class RMSNorm(nnx.Module):
    def __init__(self, d, eps=1e-6):
        self.gain = nnx.Param(jnp.ones(d))
        self.eps = eps

    def __call__(self, X):
        rms = jnp.sqrt((X ** 2).mean(-1, keepdims=True) + self.eps)
        return self.gain * X / rms
```

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-1}
%%tab tensorflow
class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def build(self, X_shape):
        self.gain = self.add_weight(name='gain', shape=[X_shape[-1]],
                                    initializer='ones')

    def call(self, X):
        rms = tf.math.rsqrt(
            tf.reduce_mean(X ** 2, axis=-1, keepdims=True) + self.eps)
        return self.gain * X * rms
```

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-1}
%%tab mxnet
class RMSNorm(nn.Block):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.gain = gluon.Parameter('gain', shape=(d,), init=init.One())
        self.eps = eps

    def forward(self, X):
        rms = np.sqrt((X ** 2).mean(-1, keepdims=True) + self.eps)
        return self.gain.data(X.device) * X / rms
```

Let $m_2$ denote an input row's mean square. With gain one, the output mean
square is $m_2/(m_2 + \epsilon)$. It is therefore close to one when
$m_2 \gg \epsilon$, as the badly scaled inputs below verify.

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-2}
%%tab pytorch
norm = RMSNorm(8)
X = 100 * torch.randn(4, 8)
norm(X).pow(2).mean(-1)
```

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-2}
%%tab jax
norm = RMSNorm(8)
X = 100 * jax.random.normal(d2l.get_key(), (4, 8))
(norm(X) ** 2).mean(-1)
```

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-2}
%%tab tensorflow
norm = RMSNorm()
X = 100 * tf.random.normal((4, 8))
tf.reduce_mean(norm(X) ** 2, axis=-1)
```

```{.python .input #custom-layers-layers-with-parameters-rmsnorm-2}
%%tab mxnet
norm = RMSNorm(8)
norm.initialize()
X = 100 * np.random.randn(4, 8)
(norm(X) ** 2).mean(-1)
```

### The Composability Guarantee

The reason to write RMSNorm as a module, rather than as a function with a
gain tensor floating around beside it, is what registration buys. A correctly
written custom layer is indistinguishable from a built-in one along four
axes: its parameters are tracked, it composes inside containers, its state
serializes, and it moves across devices. We check each once.

:begin_tab:`pytorch`
First, the gain registered itself the moment we assigned an `nn.Parameter` in
`__init__`. Any optimizer handed `norm.parameters()` will find and update it.
:end_tab:

:begin_tab:`jax`
First, assigning `nnx.Param` registered the gain in the NNX object graph.
`nnx.state(norm, nnx.Param)` selects it, and an `nnx.Optimizer` configured
with `wrt=nnx.Param` will update it.
:end_tab:

:begin_tab:`tensorflow`
First, the gain registered itself the moment `add_weight` ran inside `build`.
Any optimizer handed `norm.trainable_variables` will find and update it.
:end_tab:

:begin_tab:`mxnet`
First, the gain registered itself the moment we assigned it to an attribute:
the block files every `gluon.Parameter` it is handed. A `Trainer` given
`norm.collect_params()` will find and update it.
:end_tab:

```{.python .input #custom-layers-the-composability-guarantee-1}
%%tab pytorch
list(norm.named_parameters())
```

```{.python .input #custom-layers-the-composability-guarantee-1}
%%tab jax
nnx.state(norm, nnx.Param)
```

```{.python .input #custom-layers-the-composability-guarantee-1}
%%tab tensorflow
norm.trainable_variables
```

```{.python .input #custom-layers-the-composability-guarantee-1}
%%tab mxnet
norm.collect_params()
```

Second, the layer drops into a `Sequential` next to built-ins.

```{.python .input #custom-layers-the-composability-guarantee-2}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), RMSNorm(8), nn.LazyLinear(2))
net(torch.randn(4, 20)).shape
```

```{.python .input #custom-layers-the-composability-guarantee-2}
%%tab jax
net = nnx.Sequential(nnx.Linear(20, 8, rngs=nnx.Rngs(0)), RMSNorm(8),
                     nnx.Linear(8, 2, rngs=nnx.Rngs(1)))
Y = net(jax.random.normal(d2l.get_key(), (4, 20)))
Y.shape
```

```{.python .input #custom-layers-the-composability-guarantee-2}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(8), RMSNorm(),
                           tf.keras.layers.Dense(2)])
net(tf.random.normal((4, 20))).shape
```

```{.python .input #custom-layers-the-composability-guarantee-2}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8), RMSNorm(8), nn.Dense(2))
net.initialize()
net(np.random.randn(4, 20)).shape
```

:begin_tab:`pytorch`
Third, its state serializes with everything else. A state dict written by one
instance loads into a fresh one, after which the two agree exactly (saving to
disk works the same way; see :numref:`sec_read_write`).
:end_tab:

:begin_tab:`jax`
Third, its state serializes with everything else. `flax.serialization` turns
the pure dictionary view of NNX state into bytes and back; restoring those
values into the state of a fresh model makes the two models agree exactly
(saving to disk works the same way; see :numref:`sec_read_write`).
:end_tab:

:begin_tab:`tensorflow`
Third, its state serializes with everything else. `get_weights` returns the
model's weights, gain included, as a list of arrays, and `set_weights` loads
that list into a clone once a first call has built it; after the copy the two
models agree exactly (saving to disk works the same way; see
:numref:`sec_read_write`).
:end_tab:

:begin_tab:`mxnet`
Third, its state serializes with everything else. `save_parameters` writes
every parameter, the gain included, to a file keyed by position in the block
tree, and `load_parameters` reads that file into a fresh clone, filling in
the deferred shapes as it goes; after the load the two models agree exactly
(:numref:`sec_read_write` covers the file formats).
:end_tab:

```{.python .input #custom-layers-the-composability-guarantee-3}
%%tab pytorch
clone = nn.Sequential(nn.LazyLinear(8), RMSNorm(8), nn.LazyLinear(2))
clone.load_state_dict(net.state_dict())
X = torch.randn(4, 20)
torch.equal(net(X), clone(X))
```

```{.python .input #custom-layers-the-composability-guarantee-3}
%%tab jax
graph, state = nnx.split(net)
raw = serialization.to_bytes(nnx.to_pure_dict(state))
X = jax.random.normal(d2l.get_key(), (4, 20))
clone = nnx.Sequential(nnx.Linear(20, 8, rngs=nnx.Rngs(2)), RMSNorm(8),
                       nnx.Linear(8, 2, rngs=nnx.Rngs(3)))
clone_graph, clone_state = nnx.split(clone)
restored = serialization.from_bytes(nnx.to_pure_dict(clone_state), raw)
nnx.replace_by_pure_dict(clone_state, restored)
clone = nnx.merge(clone_graph, clone_state)
bool(jnp.array_equal(net(X), clone(X)))
```

```{.python .input #custom-layers-the-composability-guarantee-3}
%%tab tensorflow
clone = tf.keras.Sequential([tf.keras.layers.Dense(8), RMSNorm(),
                             tf.keras.layers.Dense(2)])
X = tf.random.normal((4, 20))
clone(X)  # A first call builds the clone so its weights exist
clone.set_weights(net.get_weights())
bool(tf.reduce_all(net(X) == clone(X)))
```

```{.python .input #custom-layers-the-composability-guarantee-3}
%%tab mxnet
with tempfile.NamedTemporaryFile(suffix='.params', delete=False) as f:
    path = f.name
net.save_parameters(path)
clone = nn.Sequential()
clone.add(nn.Dense(8), RMSNorm(8), nn.Dense(2))
clone.load_parameters(path)
os.remove(path)
X = np.random.randn(4, 20)
bool((net(X) == clone(X)).all())
```

:begin_tab:`pytorch`
Fourth, it moves. `.to(device)` walks the module tree and carries every
registered tensor along, the gain included. On a machine with a GPU the cell
below reports `cuda:0` twice; on a CPU-only machine it reports `cpu` and runs
just the same.
:end_tab:

:begin_tab:`jax`
Fourth, it moves. NNX state is a pytree of `Variable` objects whose array
values can be placed on a chosen device and written back with `nnx.update`.
On a machine with a GPU (JAX's default device there) the cell below reports a
GPU device twice; on a CPU-only machine it reports `cpu` and runs just the
same.
:end_tab:

:begin_tab:`tensorflow`
Fourth, it moves, though TensorFlow settles this axis at creation rather than
on demand. There is no move-the-model call: `add_weight` places the gain on
the best available device the moment it runs (the GPU if one is visible), and
operations execute where their inputs live; a `tf.device` scope at
construction time overrides the default (:numref:`sec_use_gpu`). So there
is nothing to run here. The guarantee is the same: because the gain went
through the proper channel, placement is handled for it, custom layer or
built-in alike.
:end_tab:

:begin_tab:`mxnet`
Fourth, it moves. Gluon settles device placement at initialization:
`initialize(device=...)` puts every registered parameter, the gain included,
on the device you name, and `reset_device` moves an initialized block later.
The forward pass then fetches the copy living where the input does, which is
what `self.gain.data(X.device)` was for. :numref:`sec_use_gpu` exercises
these moves on real hardware, so we leave this axis as prose here. The
guarantee is the same: because the gain went through the proper channel,
every placement and move includes it, custom layer or built-in alike.
:end_tab:

```{.python .input #custom-layers-the-composability-guarantee-4}
%%tab pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
net[1].gain.device, net(torch.randn(4, 20, device=device)).device
```

```{.python .input #custom-layers-the-composability-guarantee-4}
%%tab jax
device = jax.devices()[0]
nnx.update(net, jax.device_put(nnx.state(net), device))
net.layers[1].gain.device, net(X).device
```

None of this took any code beyond the class definition. The guarantee comes
from the base class: subclass it, register state through the proper channels,
and containers, optimizers, checkpoints, and devices all treat your layer as
native.

### Checking against the Built-in

:begin_tab:`pytorch, jax, tensorflow`
RMSNorm proved useful enough that the library now ships its own
implementation. That gives us a referee. We copy a nontrivial gain into both
implementations, so that agreement cannot be an accident of default values,
and compare outputs.
:end_tab:

:begin_tab:`mxnet`
RMSNorm proved useful enough that most libraries now ship their own
implementation. This Gluon version is the exception: it has `nn.LayerNorm`
but no RMSNorm, so there is no referee
to check our five-liner against. The general rule for custom layers, build
one to understand it, then use the native implementation in production,
therefore resolves differently on this tab: when the library ships nothing
to prefer, keep the tested custom implementation.
:end_tab:

:begin_tab:`jax`
Parameters are mutable NNX variables, so we assign the same `gain` array to
our layer and to the native layer's `scale` parameter.
:end_tab:

:begin_tab:`tensorflow`
The native layer is `RMSNormalization`; it names its parameter `scale` and
its divisor guard `epsilon`, with the same $10^{-6}$ default as ours. Copying
is a weights-list operation: once both layers are built, `get_weights` and
`set_weights` carry the gain across.
:end_tab:

```{.python .input #custom-layers-checking-against-the-built-in}
%%tab pytorch
ours, native = RMSNorm(8), nn.RMSNorm(8, eps=1e-6)
with torch.no_grad():
    ours.gain.copy_(torch.linspace(0.5, 1.5, 8))
    native.weight.copy_(ours.gain)
X = torch.randn(16, 8)
torch.allclose(ours(X), native(X))
```

```{.python .input #custom-layers-checking-against-the-built-in}
%%tab jax
ours = RMSNorm(8)
native = nnx.RMSNorm(8, epsilon=1e-6, rngs=nnx.Rngs(0))
gain = jnp.linspace(0.5, 1.5, 8)
X = jax.random.normal(d2l.get_key(), (16, 8))
ours.gain[...] = gain
native.scale[...] = gain
bool(jnp.allclose(ours(X), native(X)))
```

```{.python .input #custom-layers-checking-against-the-built-in}
%%tab tensorflow
ours, native = RMSNorm(), tf.keras.layers.RMSNormalization(epsilon=1e-6)
X = tf.random.normal((16, 8))
ours(X)  # Build both so the weights exist
native.build(X.shape)
ours.gain.assign(tf.linspace(0.5, 1.5, 8))
native.set_weights(ours.get_weights())
bool(tf.experimental.numpy.allclose(ours(X), native(X)))
```

:begin_tab:`pytorch, jax, tensorflow`
They match to floating-point precision. This is the general rule for custom
layers: build one to understand it, then use the native implementation in
production. The native version may fuse the reduction and the scale into a
single kernel, and it will be maintained as the library evolves.
:end_tab:

## Precomputed State: Buffers

The compact mask below accepts square self-attention score matrices. Cached
decoding, where query and key lengths differ, needs an offset mask rather than
this $T \times T$ slice.

:begin_tab:`pytorch`
:numref:`sec_parameters` introduced buffers as the third kind of module
state: tensors that persist and travel with the model but receive no
gradient. Custom layers are where you create them. A common case is a layer
built around a precomputed table, for instance the causal mask that keeps
attention scores from looking at future positions. The mask is fixed, so a
parameter is wrong (the optimizer would update it) and a plain attribute is
wrong too (`.to(device)` would skip it and the state dict would omit it).
`register_buffer` is the correct channel.
:end_tab:

:begin_tab:`jax`
:numref:`sec_parameters` introduced state that persists and travels with
the model but receives no gradient. NNX distinguishes kinds of state through
`Variable` subclasses. `nnx.Param` marks trainable state; a custom subclass
can mark a buffer that an optimizer must skip. A common example is the causal
mask that keeps attention scores from looking at future positions. The mask
is fixed, so a parameter is wrong. Storing it in `Buffer` keeps it out of the
gradient computation while it still serializes and moves with the model.
BatchNorm uses the same idea for its running statistics through
`nnx.BatchStat`.
:end_tab:

:begin_tab:`tensorflow`
:numref:`sec_parameters` introduced state that persists and travels with
the model but receives no gradient. In Keras the channel is the `trainable`
flag: `add_weight(trainable=False)` creates a variable no optimizer will
touch that still counts among the layer's weights, so it serializes through
`get_weights` and gets its device placement like any parameter. A common case
is a layer built around a precomputed table, for instance the causal mask
that keeps attention scores from looking at future positions. The mask is
fixed, so a trainable weight is wrong (the optimizer would update it) and a
plain tensor attribute is wrong too (the weights list would omit it).
BatchNorm's moving statistics are non-trainable weights by exactly this
mechanism.
:end_tab:

:begin_tab:`mxnet`
:numref:`sec_parameters` introduced state that persists and travels with
the model but receives no gradient. Gluon's channel is `gluon.Constant`, a
`Parameter` subclass whose `grad_req` is fixed to `'null'`: autograd ignores
it and no `Trainer` will update it, yet it initializes, saves, and moves with
the rest of the block's parameters. A common case is a layer built around a
precomputed table, for instance the causal mask that keeps attention scores
from looking at future positions. The mask is fixed, so an ordinary parameter
is wrong (the optimizer would update it) and a plain array attribute is wrong
too (`save_parameters` would omit it and `reset_device` would skip it).
:end_tab:

```{.python .input #custom-layers-precomputed-state-buffers-1}
%%tab pytorch
class CausalMask(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        # Precompute once for the longest sequence; slice per call
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), 1)
        self.register_buffer('mask', mask)

    def forward(self, scores):
        T = scores.shape[-1]
        return scores.masked_fill(self.mask[:T, :T], float('-inf'))
```

```{.python .input #custom-layers-precomputed-state-buffers-1}
%%tab jax
class Buffer(nnx.Variable):
    pass

class CausalMask(nnx.Module):
    def __init__(self, max_len):
        self.mask = Buffer(jnp.triu(
            jnp.ones((max_len, max_len), dtype=bool), 1))

    def __call__(self, scores):
        T = scores.shape[-1]
        return jnp.where(self.mask[:T, :T], -jnp.inf, scores)
```

```{.python .input #custom-layers-precomputed-state-buffers-1}
%%tab tensorflow
class CausalMask(tf.keras.layers.Layer):
    def __init__(self, max_len):
        super().__init__()
        # Precompute once for the longest sequence; slice per call
        self.mask = self.add_weight(name='mask', shape=(max_len, max_len),
                                    dtype=tf.bool, trainable=False,
                                    initializer='zeros')
        self.mask.assign(
            tf.range(max_len)[None, :] > tf.range(max_len)[:, None])

    def call(self, scores):
        T = scores.shape[-1]
        return tf.where(self.mask[:T, :T], float('-inf'), scores)
```

```{.python .input #custom-layers-precomputed-state-buffers-1}
%%tab mxnet
class CausalMask(nn.Block):
    def __init__(self, max_len):
        super().__init__()
        # Precompute once for the longest sequence; slice per call
        self.mask = gluon.Constant(np.triu(np.ones((max_len, max_len)), 1))

    def forward(self, scores):
        T = scores.shape[-1]
        return np.where(self.mask.data(scores.device)[:T, :T].astype(bool),
                        -np.inf, scores)
```

:begin_tab:`pytorch`
The layer masks the strict upper triangle, has no parameters for an optimizer
to touch, and still lists the mask in its state dict.
:end_tab:

:begin_tab:`jax`
The layer masks the strict upper triangle, has no `nnx.Param` variables for
the optimizer, and still carries the mask as `Buffer` state.
:end_tab:

:begin_tab:`tensorflow`
The layer masks the strict upper triangle, has an empty trainable-weights
list for an optimizer to consult, and still carries the mask among its
weights.
:end_tab:

:begin_tab:`mxnet`
The layer masks the strict upper triangle, and its one entry in
`collect_params` is a `Constant` with `grad_req` `'null'`, so a `Trainer`
skips it while serialization and device moves still carry it.
:end_tab:

```{.python .input #custom-layers-precomputed-state-buffers-2}
%%tab pytorch
mask = CausalMask(max_len=8)
print(mask(torch.zeros(3, 3)))
print(list(mask.parameters()), list(mask.state_dict()))
```

```{.python .input #custom-layers-precomputed-state-buffers-2}
%%tab jax
mask = CausalMask(max_len=8)
print(mask(jnp.zeros((3, 3))))
print(nnx.state(mask, nnx.Param), nnx.state(mask, Buffer))
```

```{.python .input #custom-layers-precomputed-state-buffers-2}
%%tab tensorflow
mask = CausalMask(max_len=8)
print(mask(tf.zeros((3, 3))))
print(mask.trainable_weights, [w.name for w in mask.weights])
```

```{.python .input #custom-layers-precomputed-state-buffers-2}
%%tab mxnet
mask = CausalMask(max_len=8)
mask.initialize()
print(mask(np.zeros((3, 3))))
print(mask.collect_params())
```

## Custom Gradients

So far we customized the forward computation and let automatic
differentiation derive the backward pass. That derivation is literal: it
differentiates exactly the operations you ran. Occasionally literalness is
the problem. Quantization rounds values to a grid, and rounding is flat
almost everywhere, so its true derivative is zero. Any parameter sitting
behind a rounding operation stops learning:

```{.python .input #custom-layers-custom-gradients-1}
%%tab pytorch
w = torch.tensor([0.9, 1.4, 2.6], requires_grad=True)
w.round().sum().backward()
w.grad
```

```{.python .input #custom-layers-custom-gradients-1}
%%tab jax
w = jnp.array([0.9, 1.4, 2.6])
jax.grad(lambda w: jnp.round(w).sum())(w)
```

```{.python .input #custom-layers-custom-gradients-1}
%%tab tensorflow
w = tf.Variable([0.9, 1.4, 2.6])
with tf.GradientTape() as tape:
    y = tf.reduce_sum(tf.round(w))
tape.gradient(y, w, unconnected_gradients=tf.UnconnectedGradients.ZERO)
```

```{.python .input #custom-layers-custom-gradients-1}
%%tab mxnet
w = np.array([0.9, 1.4, 2.6])
w.attach_grad()
with autograd.record():
    y = np.round(w).sum()
y.backward()
w.grad
```

:begin_tab:`tensorflow`
TensorFlow declares rounding non-differentiable outright, so without the
`unconnected_gradients` flag the tape reports the disconnection as `None`;
the flag asks for the zeros that calculus prescribes.
:end_tab:

Zero gradient, exactly as calculus demands, and useless for training. The
*straight-through estimator* :cite:`Bengio.Leonard.Courville.2013` resolves
this with a controlled lie: keep the rounding in the forward pass, but
pretend in the backward pass that it was the identity.

:begin_tab:`pytorch`
No automatic system can derive a lie for us, so we override the chain rule
with `torch.autograd.Function`, supplying both directions ourselves as static
methods, the split :numref:`fig_bg_ste` draws explicitly.
:end_tab:

:begin_tab:`jax`
No automatic system can derive a lie for us, so we override the chain rule
with `jax.custom_vjp`, attaching a hand-written backward rule (a
vector-Jacobian product) to an ordinary function, the split
:numref:`fig_bg_ste` draws explicitly.
:end_tab:

:begin_tab:`tensorflow`
No automatic system can derive a lie for us, so we override the chain rule
with `@tf.custom_gradient`, a decorator under which the function returns its
backward rule alongside its output, the split :numref:`fig_bg_ste` draws
explicitly.
:end_tab:

:begin_tab:`mxnet`
No automatic system can derive a lie for us, so we override the chain rule
with `autograd.Function`, supplying both directions ourselves as methods of a
class, the split :numref:`fig_bg_ste` draws explicitly.
:end_tab:

![The straight-through estimator, forward versus backward. Forward keeps the true staircase round(x), close to but not the same as the identity it approximates; backward substitutes a constant surrogate gradient of 1, the identity's own derivative, for the true gradient, which is zero almost everywhere and would stop all learning.](../img/bg-ste.svg)
:label:`fig_bg_ste`

```{.python .input #custom-layers-custom-gradients-2}
%%tab pytorch
class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X):
        return X.round()

    @staticmethod
    def backward(ctx, grad_output):
        # Pretend forward was the identity: pass the gradient straight through
        return grad_output
```

```{.python .input #custom-layers-custom-gradients-2}
%%tab jax
@jax.custom_vjp
def round_ste(X):
    return jnp.round(X)

def round_ste_fwd(X):
    return jnp.round(X), None  # (output, residuals); we need no residuals

def round_ste_bwd(residuals, grad_output):
    # Pretend forward was the identity: pass the gradient straight through
    return (grad_output,)

round_ste.defvjp(round_ste_fwd, round_ste_bwd)
```

```{.python .input #custom-layers-custom-gradients-2}
%%tab tensorflow
@tf.custom_gradient
def round_ste(X):
    def grad(upstream):
        # Pretend forward was the identity: pass the gradient straight through
        return upstream
    return tf.round(X), grad
```

```{.python .input #custom-layers-custom-gradients-2}
%%tab mxnet
class RoundSTE(autograd.Function):
    def forward(self, X):
        return np.round(X)

    def backward(self, grad_output):
        # Pretend forward was the identity: pass the gradient straight through
        return grad_output
```

:begin_tab:`pytorch`
Two rules govern the class. First, invoke it through `RoundSTE.apply(X)`,
never by calling `forward` directly: `apply` is what inserts the operation
into the autograd graph, while a direct call computes the same values with no
bookkeeping, and `backward` then never runs. This is the most common
first-time bug with custom functions, and it fails without an error message.
Second, `backward` receives the loss gradient with respect to the output and
must return the gradient with respect to each input. Ours ignores the input
values entirely; a backward that needs them must stash them in `forward` with
`ctx.save_for_backward` and retrieve them from `ctx`.
:end_tab:

:begin_tab:`jax`
Two rules govern the definition. First, the forward rule must return a
*pair*: the output plus whatever residuals the backward rule will need. Ours
needs nothing and returns `None`; returning the output alone is the most
common first-time bug, and JAX reports it as a puzzling structure mismatch
rather than pointing at the missing residuals. Second, the backward rule
receives those residuals and the loss gradient with respect to the output,
and must return a tuple holding one gradient per argument of the original
function, hence `(grad_output,)` rather than `grad_output`. An argument that
is not differentiable data, say a configuration flag, must be declared
through `nondiff_argnums` so it is routed past the gradient machinery.
:end_tab:

:begin_tab:`tensorflow`
Two rules govern the decorator. First, the decorated function returns a
*pair*: the output and the gradient function. Because the gradient function
is a closure defined inside the forward pass, anything the backward
computation needs, the input values say, is captured for free; there is no
separate residual mechanism, and the decorated function is invoked like any
other, the decorator handling the tape bookkeeping. Second, `grad` receives
the loss gradient with respect to the output and must return one gradient per
argument of the decorated function. The genuine trap is state: if the wrapped
computation reads a `tf.Variable`, the gradient function must instead accept
a `variables` keyword argument and return the variables' gradients as a
second result; TensorFlow raises a `TypeError` naming this requirement if it
is missing.
:end_tab:

:begin_tab:`mxnet`
Two rules govern the class. First, a `Function` instance is single-use: the
library asserts that each instance is called exactly once, so create a fresh
`RoundSTE()` at every application rather than reusing one, or the second call
fails with an assertion naming this rule. Second, `backward` receives the
loss gradient with respect to each output of `forward` and must return the
gradient with respect to each input. Ours ignores the input values entirely;
a backward that needs them must stash them in `forward` with
`self.save_for_backward` and read them back from `self.saved_tensors`.
:end_tab:

A small example verifies that the gradient now flows. We multiply the rounded
values by known weights, so we know what gradient to expect at `w`.

```{.python .input #custom-layers-custom-gradients-3}
%%tab pytorch
w.grad = None
y = (RoundSTE.apply(w) * torch.tensor([1.0, 2.0, 3.0])).sum()
y.backward()
w.grad
```

```{.python .input #custom-layers-custom-gradients-3}
%%tab jax
jax.grad(lambda w: (round_ste(w) * jnp.array([1.0, 2.0, 3.0])).sum())(w)
```

```{.python .input #custom-layers-custom-gradients-3}
%%tab tensorflow
with tf.GradientTape() as tape:
    y = tf.reduce_sum(round_ste(w) * tf.constant([1.0, 2.0, 3.0]))
tape.gradient(y, w)
```

```{.python .input #custom-layers-custom-gradients-3}
%%tab mxnet
with autograd.record():
    y = (RoundSTE()(w) * np.array([1.0, 2.0, 3.0])).sum()
y.backward()
w.grad
```

The downstream gradient arrives at `w` untouched, as if the rounding were the
identity, while the forward pass still computed with rounded values. Some
estimators instead zero the gradient where a saved input had saturated, while
ordinary gradient clipping bounds a large upstream gradient independently of
the quantizer. These are different approximations. Neither is a universal
default; the surrogate backward rule must match the training objective it is
meant to approximate.

:begin_tab:`jax`
JAX offers a shortcut for this particular lie. `jax.lax.stop_gradient` is an
identity whose gradient is zero, so `X + stop_gradient(round(X) - X)`
computes `round(X)` in the forward pass while the backward pass sees only the
leading `X`. Three lines, no `custom_vjp`; the general mechanism earns its
keep when the surrogate is not the identity, an input-dependent saturation
rule for instance. The two definitions agree in value and in gradient:
:end_tab:

```{.python .input #custom-layers-custom-gradients-4}
%%tab jax
def round_ste2(X):
    return X + jax.lax.stop_gradient(jnp.round(X) - X)

def loss(f, w):
    return (f(w) * jnp.array([1.0, 2.0, 3.0])).sum()

(bool(jnp.allclose(round_ste(w), round_ste2(w))),
 bool(jnp.allclose(jax.grad(loss, 1)(round_ste, w),
                   jax.grad(loss, 1)(round_ste2, w))))
```

A scope note to close. A custom gradient redefines the gradient of a
computation assembled from existing tensor operations; it does not create new
ones. Writing new *kernels*, code that runs on the accelerator itself, is a
separate craft that we do not cover in this book; :numref:`chap_performance`
discusses how to get performance out of the operations you already have.

## Summary

:begin_tab:`pytorch`
A custom layer is a module subclass: `forward` defines the computation,
`nn.Parameter` registers learnable state, and `register_buffer` registers
persistent state that no optimizer should touch. Registration is what buys
composability; a correctly written layer gets parameter tracking, container
compatibility, serialization, and device movement for free, as we verified on
RMSNorm axis by axis. When the chain rule itself must be overridden, as in
the straight-through estimator, `torch.autograd.Function` lets you supply
`forward` and `backward` as a pair, invoked through `apply`. Build custom
implementations to understand them; prefer the native ones in production.
:end_tab:

:begin_tab:`jax`
A custom layer is a module subclass: `__call__` defines the computation,
`nnx.Param` registers learnable state, and another `nnx.Variable` subclass
registers persistent state that the optimizer does not touch. Registration
is what buys composability; a correctly written layer
gets parameter tracking, container compatibility, serialization, and device
movement for free, as we verified on RMSNorm axis by axis. When the chain
rule itself must be overridden, as in the straight-through estimator,
`jax.custom_vjp` attaches a forward and a backward rule to a function, and
`jax.lax.stop_gradient` covers the identity-surrogate case in a single
expression. Build custom implementations to understand them; prefer the
native ones in production.
:end_tab:

:begin_tab:`tensorflow`
A custom layer is a layer subclass: `call` defines the computation,
`add_weight` registers learnable state, and `add_weight(trainable=False)`
registers persistent state that no optimizer should touch. Registration is
what buys composability; a correctly written layer gets parameter tracking,
container compatibility, and serialization for free, with device placement
settled at creation, as we verified on RMSNorm axis by axis. When the chain
rule itself must be overridden, as in the straight-through estimator,
`@tf.custom_gradient` lets the forward function return its own backward rule.
Build custom implementations to understand them; prefer the native ones in
production.
:end_tab:

:begin_tab:`mxnet`
A custom layer is a block subclass: `forward` defines the computation,
assigning a `gluon.Parameter` registers learnable state, and `gluon.Constant`
registers persistent state that autograd and the `Trainer` ignore.
Registration is what buys composability; a correctly written layer gets
parameter tracking, container compatibility, serialization, and device
movement for free, as we verified on RMSNorm axis by axis. When the chain
rule itself must be overridden, as in the straight-through estimator,
`autograd.Function` lets you supply `forward` and `backward` as a pair, one
fresh instance per call. Build custom implementations to understand them;
prefer the native ones in production, and where gluon ships none, as with
RMSNorm, keep your own.
:end_tab:

## Exercises

:begin_tab:`pytorch`
1. Add an optional learned bias to `RMSNorm` (a shift applied after the
   scale, restoring part of what LayerNorm had). Verify that the state dict
   of a model containing it grows by the expected entry, and that a state
   dict saved without the bias no longer loads with `strict=True`.
1. Implement `Dropout` from scratch as a custom layer that zeroes each entry
   with probability $p$ and rescales the survivors during training, but is
   the identity during evaluation. Where does the `training` flag your
   `forward` consults live, and how does calling `.eval()` on an enclosing
   `Sequential` reach your layer?
1. Implement a clamp with a custom gradient: an `autograd.Function` whose
   forward is `X.clamp(lo, hi)` and whose backward passes the gradient only
   where the input lay strictly inside the clamp range. Compare its gradients
   with those of the native `torch.clamp` for inputs inside, outside, and
   exactly on the boundary. Which convention does the native operation use at
   the boundary?
1. Write a layer that returns the leading half of the Fourier coefficients of
   its input. It has no parameters, so nothing registers. What do you still
   gain by wrapping the computation in a module instead of calling the
   transform inline wherever you need it?
:end_tab:

:begin_tab:`jax`
1. Add an optional learned bias to `RMSNorm` (a shift applied after the
   scale, restoring part of what LayerNorm had). Verify that the parameter
   state of a model containing it grows by the expected entry, and observe
   what `flax.serialization.from_bytes` does when it restores bytes saved
   without the bias into the new state structure.
1. Implement `Dropout` from scratch as a custom layer that zeroes each entry
   with probability $p$ and rescales the survivors during training, but is
   the identity during evaluation. Store `self.deterministic = False`, give
   the module an `nnx.Rngs` stream, and implement
   `set_view(self, *, deterministic)` to update the flag. Verify that
   `nnx.view(model, deterministic=True)` reaches a dropout layer nested in an
   `nnx.Sequential`.
1. Implement a clamp with a custom gradient: a `custom_vjp` function whose
   forward is `jnp.clip(X, lo, hi)` and whose backward passes the gradient
   only where the input lay strictly inside the clamp range; the forward rule
   must save `X` as a residual. Compare its gradients with those of the
   native `jnp.clip` for inputs inside, outside, and exactly on the boundary.
   Which convention does the native operation use at the boundary?
1. Write a layer that returns the leading half of the Fourier coefficients of
   its input. It has no parameters, so nothing registers. What do you still
   gain by wrapping the computation in a module instead of calling the
   transform inline wherever you need it?
:end_tab:

:begin_tab:`tensorflow`
1. Add an optional learned bias to `RMSNorm` (a shift applied after the
   scale, restoring part of what LayerNorm had). Verify that `get_weights` on
   a model containing it grows by the expected entry, and observe what
   `set_weights` does when handed a list saved without the bias.
1. Implement `Dropout` from scratch as a custom layer that zeroes each entry
   with probability $p$ and rescales the survivors during training, but is
   the identity during evaluation. Keras passes a `training` argument to
   `call`; how does calling an enclosing `Sequential` with `training=False`
   reach your layer?
1. Implement a clamp with a custom gradient: a `@tf.custom_gradient` function
   whose forward is `tf.clip_by_value(X, lo, hi)` and whose backward passes
   the gradient only where the input lay strictly inside the clamp range; the
   gradient closure must capture `X`. Compare its gradients with those of the
   native `tf.clip_by_value` for inputs inside, outside, and exactly on the
   boundary. Which convention does the native operation use at the boundary?
1. Write a layer that returns the leading half of the Fourier coefficients of
   its input. It has no parameters, so nothing registers. What do you still
   gain by wrapping the computation in a module instead of calling the
   transform inline wherever you need it?
:end_tab:

:begin_tab:`mxnet`
1. Add an optional learned bias to `RMSNorm` (a shift applied after the
   scale, restoring part of what LayerNorm had). Verify that the file written
   by `save_parameters` for a model containing it grows by the expected
   entry, and observe what `load_parameters` does when handed a file saved
   without the bias. Which of its `allow_missing` and `ignore_extra` flags
   changes the outcome?
1. Implement `Dropout` from scratch as a custom layer that zeroes each entry
   with probability $p$ and rescales the survivors during training, but is
   the identity during evaluation. Gluon has no `.train()`/`.eval()` switch;
   the flag your `forward` consults is `autograd.is_training()`. What sets
   that flag, and what does it imply for calling the layer outside
   `autograd.record()`?
1. Implement a clamp with a custom gradient: an `autograd.Function` whose
   forward is `np.clip(X, lo, hi)` and whose backward passes the gradient
   only where the input lay strictly inside the clamp range; `forward` must
   stash `X` with `save_for_backward`. Compare its gradients with those of
   the native `np.clip` for inputs inside, outside, and exactly on the
   boundary. Which convention does the native operation use at the boundary?
1. Write a layer that returns the leading half of the Fourier coefficients of
   its input. It has no parameters, so nothing registers. What do you still
   gain by wrapping the computation in a module instead of calling the
   transform inline wherever you need it?
:end_tab:
