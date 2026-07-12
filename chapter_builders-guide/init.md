```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Initialization
:label:`sec_init_param`

Every parameter in a network has a value before training starts: a fully
connected layer mapping 512 inputs to 256 outputs brings a $256 \times 512$
weight matrix filled with random numbers. :numref:`sec_numerical_stability` explains why the scale of those
numbers decides whether a deep network trains at all, and derives the Xavier
:cite:`Glorot.Bengio.2010` and He :cite:`He.Zhang.Ren.ea.2015` schemes that
keep signal variance stable across layers. This section is the practical
companion: what the library does when you say nothing, how to impose a scheme
of your own on a whole model or a single block, what transformer-era code adds
on top of Xavier and He, and how to write an initializer that no menu provides.

```{.python .input #init-initialization}
%%tab pytorch
import math
import torch
from torch import nn
```

```{.python .input #init-initialization}
%%tab jax
import math
import jax
from jax import numpy as jnp
from flax import nnx
```

```{.python .input #init-initialization}
%%tab tensorflow
import math
import tensorflow as tf
```

```{.python .input #init-initialization}
%%tab mxnet
import math
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

## Defaults and When to Override Them

You can usually ignore initialization because the default is sensible.

:begin_tab:`pytorch`
PyTorch initializes an `nn.Linear` the moment it is constructed: weight and
bias are both drawn from the uniform distribution $U(-b, b)$ with
$b = 1/\sqrt{n_\textrm{in}}$, where $n_\textrm{in}$ is the number of input
features. (The documentation describes the weight rule as a variant of He
initialization, `kaiming_uniform_` with `a=math.sqrt(5)`; the algebra collapses
to the bound above.) A uniform distribution on $[-b, b]$ has standard deviation
$b/\sqrt{3}$, so both claims are cheap to check against a fresh layer:
:end_tab:

:begin_tab:`jax`
NNX creates parameters when `nnx.Linear` is constructed, using the explicit
input width to determine fan-in. Its default is LeCun normal: the weight comes
from a normal distribution with standard deviation
$1/\sqrt{n_\textrm{in}}$, truncated so that no entry lands in the far tails
(the initializer rescales after clipping to keep the standard deviation on target, which
puts the hard bound near $2.3$ standard deviations), and the bias starts at
zero. Both claims are cheap to check against a fresh layer:
:end_tab:

:begin_tab:`tensorflow`
Keras attaches no numbers at construction either: `Dense(256)` fixes only the
output width, and the kernel comes into existence at build time, when the
layer first learns its input shape (every Keras layer is lazy in the sense of
:numref:`sec_lazy_init`). The build draws Glorot uniform, the Xavier scheme
of :numref:`subsec_xavier` in its uniform variant: the kernel comes from
$U(-b, b)$ with $b = \sqrt{6/(n_\textrm{in} + n_\textrm{out})}$, and the bias
starts at zero. A uniform distribution on $[-b, b]$ has standard deviation
$b/\sqrt{3}$, so all three claims are cheap to check against a fresh layer:
:end_tab:

:begin_tab:`mxnet`
Gluon separates construction from initialization: `nn.Dense(256)` carries no
numbers until you call `initialize()`, and when the input width is left
unspecified even that call only records your choice, with the draw deferred to
the first forward pass (:numref:`sec_lazy_init`). What it draws is the legacy
exception in the table below: every weight comes from $U(-0.07, 0.07)$
regardless of the layer's shape, a fixed constant rather than a fan-aware
bound, and the bias starts at zero. A uniform distribution on $[-b, b]$ has
standard deviation $b/\sqrt{3}$, so all three claims are cheap to check
against a fresh layer (passing `in_units` pins the input width, so nothing is
deferred here):
:end_tab:

```{.python .input #init-defaults-and-when-to-override-them}
%%tab pytorch
torch.manual_seed(0)
layer = nn.Linear(512, 256)
bound = 1 / math.sqrt(layer.in_features)
w = layer.weight
print(f'range [{w.min():.4f}, {w.max():.4f}], predicted bound {bound:.4f}')
print(f'std {w.std():.4f}, predicted {bound / math.sqrt(3):.4f}')
```

```{.python .input #init-defaults-and-when-to-override-them}
%%tab jax
layer = nnx.Linear(512, 256, rngs=nnx.Rngs(0))
w = layer.kernel
print(f'std {w.std():.4f}, predicted {1 / math.sqrt(512):.4f}')
print(f'max |w| {jnp.abs(w).max():.4f}, '
      f'bias all zero: {bool((layer.bias == 0).all())}')
```

```{.python .input #init-defaults-and-when-to-override-them}
%%tab tensorflow
tf.keras.utils.set_random_seed(0)
layer = tf.keras.layers.Dense(256)
layer.build((None, 512))
w = layer.kernel.numpy()
bound = math.sqrt(6 / (512 + 256))
print(f'range [{w.min():.4f}, {w.max():.4f}], predicted bound {bound:.4f}')
print(f'std {w.std():.4f}, predicted {bound / math.sqrt(3):.4f}')
print(f'bias all zero: {bool((layer.bias.numpy() == 0).all())}')
```

```{.python .input #init-defaults-and-when-to-override-them}
%%tab mxnet
np.random.seed(0)
layer = nn.Dense(256, in_units=512)
layer.initialize()
w = layer.weight.data()
print(f'range [{float(w.min()):.4f}, {float(w.max()):.4f}], '
      f'predicted bound 0.0700')
print(f'std {float(w.std()):.4f}, predicted {0.07 / math.sqrt(3):.4f}')
print(f'bias all zero: {bool((layer.bias.data() == 0).all())}')
```

Other libraries make different but equally fan-aware choices, with one legacy
exception:

| Library | Default for a dense layer's weight |
|:--|:--|
| PyTorch | $U(-1/\sqrt{n_\textrm{in}},\, 1/\sqrt{n_\textrm{in}})$, bias likewise |
| Flax (JAX) | LeCun normal: truncated normal, std $1/\sqrt{n_\textrm{in}}$; zero bias |
| Keras (TensorFlow) | Glorot uniform: $U(\pm\sqrt{6/(n_\textrm{in} + n_\textrm{out})})$; zero bias |
| MXNet Gluon | $U(-0.07, 0.07)$, not fan-aware, so override it; zero bias |

:begin_tab:`pytorch`
When should you override? Four situations recur: a deep network without
normalization layers, where variance compounds across depth (we demonstrate
this below); parameters you created by hand, since
`nn.Parameter(torch.empty(...))` contains whatever bytes the allocator
happened to return; reproducing a paper whose results depend on its
initialization recipe; and architecture-specific corrections such as the
residual scaling later in this section. Note that the defaults are computed
from fan-in at construction time; a lazy layer (:numref:`sec_lazy_init`) has
no fan-in until its first forward pass, so it runs the same rule at
materialization instead.
:end_tab:

:begin_tab:`jax`
When should you override? Four situations recur: a deep network without
normalization layers, where variance compounds across depth (we demonstrate
this below); reproducing a paper whose results depend on its initialization
recipe; architecture-specific corrections such as the residual scaling later
in this section; and parameters you create by hand, though here NNX forces
the choice on you: construct the value explicitly and wrap it in `nnx.Param`.
NNX layers receive their input and output widths in the constructor, and
their initializers run there as well; no later `init` pass infers fan-in from
the first input.
:end_tab:

:begin_tab:`tensorflow`
When should you override? Four situations recur: a deep network without
normalization layers, where variance compounds across depth (we demonstrate
this below); reproducing a paper whose results depend on its initialization
recipe; architecture-specific corrections such as the residual scaling later
in this section; and parameters you create by hand, where Keras quietly
substitutes a default of its own, since `add_weight` falls back to Glorot
uniform unless you pass an `initializer`, a sensible scale for a weight
matrix and the wrong one for almost anything else. There is no
construction-time fine print to remember: fan-in and fan-out are read off
the input shape when `build` runs.
:end_tab:

:begin_tab:`mxnet`
When should you override? Here the first candidate is the default itself:
$\pm 0.07$ was tuned for the hidden widths of an earlier era and ignores
fan-in, so a layer much wider than those starts with too much variance, a
much narrower one with too little, and either error compounds across depth by
the analysis of :numref:`sec_numerical_stability`. Beyond that, the usual
situations recur: a deep network without normalization layers, where variance
compounds even when each layer is right (we demonstrate this below);
reproducing a paper whose results depend on its initialization recipe; and
architecture-specific corrections such as the residual scaling later in this
section. The lazy fine print runs the other way from PyTorch's: with the
input width unspecified, `initialize()` merely records which initializer to
use, and the draw happens at the first forward pass, once the layer knows its
fan-in.
:end_tab:

## Applying Initializers

:begin_tab:`pytorch`
To impose a scheme we do not touch layers one at a time. `net.apply(fn)` walks
the module tree of :numref:`sec_model_construction` and calls `fn` on every
submodule, children first. The function inspects each module's type and
decides what, if anything, to do with it. The routines in `torch.nn.init`
follow PyTorch's convention that a trailing underscore means *in place*:
`nn.init.normal_(w)` overwrites `w` rather than returning a fresh tensor. In
place is what we want here, since the parameter must remain the same tensor
object the module registered; replacing it would leave `net.parameters()` and
any existing optimizer pointing at the old one.
:end_tab:

:begin_tab:`jax`
NNX affine layers such as `Linear` and `Conv` accept `kernel_init` and
`bias_init`, each a function
`(key, shape, dtype) -> array`, and `nnx.initializers` supplies the standard
menu. These functions run in the constructor. For a model that already
exists, we can walk its modules and assign new values to selected parameters.
For a parameter in a custom module, call an initializer directly and wrap its
result in `nnx.Param`. We show the first two routes here.
:end_tab:

:begin_tab:`tensorflow`
To impose a scheme we declare it where the layer is declared: every Keras
layer accepts `kernel_initializer` and `bias_initializer` arguments, each an
object mapping a shape and dtype to a tensor, and `tf.keras.initializers`
supplies the standard menu (the string `'zeros'` names the same object as
`tf.keras.initializers.Zeros()`). The arguments are stored at construction
and run at build time, so they wait for lazy shapes automatically. That
covers models you are about to build. The second route, for a model that
already exists, rests on the fact that a kernel is a mutable variable:
iterate over `model.layers`, decide by type what to do with each layer, and
overwrite the kernels you select in place with `assign`. We show both,
starting with constructor arguments.
:end_tab:

:begin_tab:`mxnet`
To impose a scheme we hand an `Initializer` object to the model:
`net.initialize(init=..., force_reinit=True)` walks every parameter of the
block tree of :numref:`sec_model_construction` and runs the initializer on
it (`force_reinit` lifts the guard against silently clobbering a model that
already has values). The object is a default, not a decree: a parameter
constructed with its own initializer keeps it, and every `Dense` bias is
constructed with `zeros`, which is why the normal draw below leaves the
biases at zero. Dispatching by layer type, the pattern PyTorch users write,
is possible (iterate over `net` and test `isinstance`), but Gluon's native
unit of selection is the subtree: every block, and every individual
parameter, has an `initialize` method of its own. We show both, starting with
the whole net.
:end_tab:

```{.python .input #init-applying-initializers-1}
%%tab pytorch
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

def init_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input #init-applying-initializers-1}
%%tab jax
init_normal = nnx.initializers.normal(0.01)
net = nnx.Sequential(
    nnx.Linear(4, 8, kernel_init=init_normal,
               bias_init=nnx.initializers.zeros, rngs=nnx.Rngs(0)),
    nnx.relu,
    nnx.Linear(8, 1, kernel_init=init_normal,
               bias_init=nnx.initializers.zeros, rngs=nnx.Rngs(1)))

X = jnp.ones((2, 4))
net.layers[0].kernel[:, 0], net.layers[0].bias[0]
```

```{.python .input #init-applying-initializers-1}
%%tab tensorflow
init_normal = tf.keras.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential([
    tf.keras.layers.Dense(8, kernel_initializer=init_normal,
                          bias_initializer='zeros'),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1, kernel_initializer=init_normal,
                          bias_initializer='zeros')])

X = tf.ones((2, 4))
net(X)
net.layers[0].kernel[:, 0], net.layers[0].bias[0]
```

```{.python .input #init-applying-initializers-1}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize(init=init.Normal(sigma=0.01))

X = np.ones((2, 4))
net(X)  # the input width was deferred, so the draw happens here
net[0].weight.data()[0], net[0].bias.data()[0]
```

:begin_tab:`pytorch`
The `isinstance` dispatch is what makes the pattern compose: one function can
treat `nn.Linear`, `nn.Conv2d`, and `nn.Embedding` differently, and it
silently skips containers and activations. Because `apply` runs on any
subtree, we can also mix schemes per block. Below we give the first layer
Xavier initialization (:numref:`subsec_xavier`) and set the last to a
constant, a poor idea for training by the symmetry argument of
:numref:`sec_numerical_stability`, but it makes the mechanics visible:
:end_tab:

:begin_tab:`jax`
Mixing schemes per block needs no dispatch at all: each layer names its own
initializer, so the choice sits exactly where the layer is declared. Below we
give the first layer Xavier initialization (:numref:`subsec_xavier`) and set
the last to a constant, a poor idea for training by the symmetry argument of
:numref:`sec_numerical_stability`, but it makes the mechanics visible:
:end_tab:

:begin_tab:`tensorflow`
Mixing schemes per block needs no dispatch while the model is under
construction: each layer names its own initializer, so the choice sits
exactly where the layer is declared. Below we give the first layer Xavier
initialization (:numref:`subsec_xavier`), which Keras spells `GlorotUniform`
after its author, and set the last to a constant, a poor idea for training by
the symmetry argument of :numref:`sec_numerical_stability`, but it makes the
mechanics visible:
:end_tab:

:begin_tab:`mxnet`
Mixing schemes per block is `initialize` on the subtree: `net[0]` is a block
and `net[0].weight` is a parameter, and both accept an initializer directly,
with `force_reinit=True` because we are overwriting live values. Below we
give the first layer Xavier initialization (:numref:`subsec_xavier`) and set
the second to a constant, a poor idea for training by the symmetry argument
of :numref:`sec_numerical_stability`, but it makes the mechanics visible:
:end_tab:

```{.python .input #init-applying-initializers-2}
%%tab pytorch
def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
net[0].weight.data[0], net[2].weight.data
```

```{.python .input #init-applying-initializers-2}
%%tab jax
net = nnx.Sequential(
    nnx.Linear(4, 8, kernel_init=nnx.initializers.xavier_uniform(),
               rngs=nnx.Rngs(0)), nnx.relu,
    nnx.Linear(8, 1, kernel_init=nnx.initializers.constant(42.0),
               rngs=nnx.Rngs(1)))

net.layers[0].kernel[:, 0], net.layers[2].kernel
```

```{.python .input #init-applying-initializers-2}
%%tab tensorflow
net = tf.keras.Sequential([
    tf.keras.layers.Dense(
        8, kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42.0))])

net(X)
net.layers[0].kernel[:, 0], net.layers[2].kernel[:, 0]
```

```{.python .input #init-applying-initializers-2}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
net[0].weight.data()[0], net[1].weight.data()
```

:begin_tab:`pytorch`
This ten-line pattern is the entire API. Later chapters wrap it into
one-liners, `init_cnn` for convolutional networks and `init_seq2seq` for
encoder-decoder models, but each is just a function like `init_normal` handed
to `apply`.
:end_tab:

:begin_tab:`jax`
The second route walks the existing object graph. `nnx.iter_modules` yields
each child with its path, and one fresh key per matching layer keeps the draws
independent. The function below re-draws every linear kernel of the
constant-42 network; biases pass through untouched:
:end_tab:

:begin_tab:`tensorflow`
The second route re-initializes a model that already exists. The loop below
is the whole mechanism: iterate `model.layers`, dispatch with `isinstance` so
that activations and anything else without a kernel are skipped, and `assign`
a fresh draw into each kernel. `assign` overwrites the variable's buffer in
place, which is what we want, since the parameter must remain the same
variable the model tracks and any existing optimizer updates. The function
repairs the constant-42 network from the previous cell; biases pass through
untouched:
:end_tab:

:begin_tab:`mxnet`
There is no second route to learn, because `force_reinit=True` already is the
route for a model with live values: the two cells above re-initialized the
same network three times. Whole net, block, or single parameter, the call is
the same, and nothing else is needed.
:end_tab:

```{.python .input #init-applying-initializers-3}
%%tab jax
def reinit(model, key, init_fn):
    layers = [m for _, m in nnx.iter_modules(model)
              if isinstance(m, nnx.Linear)]
    for layer, subkey in zip(layers, jax.random.split(key, len(layers))):
        layer.kernel[...] = init_fn(
            subkey, layer.kernel.shape, layer.kernel.dtype)

reinit(net, jax.random.key(1), nnx.initializers.xavier_uniform())
net.layers[2].kernel[:3]
```

```{.python .input #init-applying-initializers-3}
%%tab tensorflow
def reinit(model, init):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer.kernel.assign(init(layer.kernel.shape))

reinit(net, tf.keras.initializers.GlorotUniform(seed=1))
net.layers[2].kernel[:3]
```

## Modern Schemes: Truncation, Depth, and Zeros

Xavier and He set the variance of a single layer. The schemes below, standard
in transformer codebases, adjust what happens in the distribution's tails,
across depth, and at a block's start.

### Truncated Normals

A Gaussian gets the variance right, but its tails are unbounded. That is
harmless for one draw and a near-certainty at scale: among the $10^8$ weights
of a BERT-sized model, dozens land beyond five standard deviations. Large
draws also consume disproportionate dynamic range once a model is cast to low
precision (:numref:`sec_numerics`). BERT and implementations in the ViT
lineage use truncated-normal initialization
:cite:`Devlin.Chang.Lee.ea.2018,Dosovitskiy.Beyer.Kolesnikov.ea.2021`: the
tails are cut off at a fixed multiple of the nominal standard deviation. Raw
truncated-normal initializers do not necessarily restore the variance removed
with the tails; fan-aware variance-scaling initializers often do.

:begin_tab:`pytorch`
`nn.init.trunc_normal_` takes absolute cutoffs `a` and `b` (defaulting to
$\pm 2$, which means $\pm 2\sigma$ only when `std=1`), so a clip at two
standard deviations must state them explicitly:
:end_tab:

:begin_tab:`jax`
`nnx.initializers.truncated_normal` states its cutoffs in units of the
standard deviation (`lower=-2.0`, `upper=2.0` by default), so the clip at two
standard deviations is what you get with no extra arguments. Truncation is
the house preference throughout Flax: the `variance_scaling` factory behind
`lecun_normal`, `xavier_normal`, and `he_normal` draws its normal variants
truncated as well.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.initializers.TruncatedNormal` fixes its cutoff at two standard
deviations: draws landing outside the bound are discarded and redrawn, and no
rescaling follows. Truncation also backs the fan-aware factory behind
`HeNormal` and its relatives, which draws a truncated normal and then
rescales it so the standard deviation lands on target despite the missing
tails:
:end_tab:

:begin_tab:`mxnet`
`mxnet.init` ships no truncated normal; the menu ends just before the scheme
we want. The escape hatch is the one this section closes with: an initializer
is a subclass of `init.Initializer` that fills an array inside its
`_init_weight` method, so the missing entry costs a dozen lines. We draw
normal values and redraw every entry that lands outside two standard
deviations until none does, the same discard-and-redraw semantics as Keras,
and keep the drawing function separate so it can double as the demonstration:
:end_tab:

```{.python .input #init-truncated-normals}
%%tab pytorch
torch.manual_seed(0)
w = torch.empty(1000, 1000)
nn.init.normal_(w, std=0.02)
print(f'normal:    std {w.std():.4f}, max weight {w.abs().max():.4f}')
nn.init.trunc_normal_(w, std=0.02, a=-0.04, b=0.04)
print(f'truncated: std {w.std():.4f}, max weight {w.abs().max():.4f}')
```

```{.python .input #init-truncated-normals}
%%tab jax
key = jax.random.key(0)
w = nnx.initializers.normal(stddev=0.02)(key, (1000, 1000))
print(f'normal:    std {w.std():.4f}, max weight {jnp.abs(w).max():.4f}')
w = nnx.initializers.truncated_normal(stddev=0.02)(key, (1000, 1000))
print(f'truncated: std {w.std():.4f}, max weight {jnp.abs(w).max():.4f}')
```

```{.python .input #init-truncated-normals}
%%tab tensorflow
init = tf.keras.initializers.RandomNormal(stddev=0.02, seed=0)
w = init((1000, 1000)).numpy()
print(f'normal:    std {w.std():.4f}, max weight {abs(w).max():.4f}')
init = tf.keras.initializers.TruncatedNormal(stddev=0.02, seed=0)
w = init((1000, 1000)).numpy()
print(f'truncated: std {w.std():.4f}, max weight {abs(w).max():.4f}')
```

```{.python .input #init-truncated-normals}
%%tab mxnet
def trunc_normal(sigma, shape):
    w = np.random.normal(0, sigma, shape)
    tails = np.abs(w) > 2 * sigma
    while bool(tails.any()):
        w = np.where(tails, np.random.normal(0, sigma, shape), w)
        tails = np.abs(w) > 2 * sigma
    return w

class TruncatedNormal(init.Initializer):
    def __init__(self, sigma=0.02):
        super().__init__(sigma=sigma)
        self.sigma = sigma

    def _init_weight(self, name, arr):
        arr[:] = trunc_normal(self.sigma, arr.shape)

np.random.seed(0)
w = np.random.normal(0, 0.02, (1000, 1000))
print(f'normal:    std {float(w.std()):.4f}, '
      f'max weight {float(np.abs(w).max()):.4f}')
w = trunc_normal(0.02, (1000, 1000))
print(f'truncated: std {float(w.std()):.4f}, '
      f'max weight {float(np.abs(w).max()):.4f}')
```

A million plain-normal draws produce entries near $5\sigma = 0.1$; the
truncated version guarantees $|w| \leq 0.04$. Its printed standard deviation
dips slightly below the nominal 0.02 because truncation removes tail mass;
practice ignores the difference.

### Scaling Down Residual Branches

A residual network computes $\mathbf{x}_{k+1} = \mathbf{x}_k +
f_k(\mathbf{x}_k)$, so its output is the input plus the sum of $N$ block
contributions. If every block is initialized identically, each contribution
has variance proportional to that of the already-inflated stream it reads, and
the stream's variance compounds geometrically with depth. This is the
depth-wise cousin of the layer-wise explosion in
:numref:`sec_numerical_stability`. GPT-2's fix is to shrink only the *last*
linear layer of each residual branch, the output projection that writes into
the stream, by a factor of $1/\sqrt{N}$: with $N$ roughly independent
contributions each scaled down that way, the sum's variance stays $O(1)$
regardless of depth :cite:`Radford.Wu.Child.ea.2019`. The published recipe is
a normal with std $0.02/\sqrt{2N}$ on the residual projections (the factor is
$2N$ because each of the $N$ transformer layers writes to the stream twice,
once from attention and once from the MLP). Unlike Xavier or He, the base 0.02
is not derived from fan-in; it is an empirical constant that worked at GPT-2's
widths and stuck.

:numref:`fig_bg_residual-stream` draws the two regimes side by side.

![A residual stream with additive block contributions, unscaled (left) versus scaled by 1/sqrt(N) (right). Under the independent-contribution approximation, unscaled variance grows with N and scaling keeps the sum O(1). In an unnormalized stack, each branch reads an already inflated stream, so growth can be faster, as the experiment below shows.](../img/bg-residual-stream.svg)
:label:`fig_bg_residual-stream`

### Starting a Block at Zero

Zero-initializing *every* weight is fatal: all units in a layer compute the
same output, receive the same gradient, and stay identical forever
(:numref:`sec_numerical_stability`). Zero-initializing just the *last* layer
of a residual block is a different and useful move. The branch then
contributes exactly nothing, each block is the identity map, and the network
starts as a shallow function whose depth switches on gradually during
training. Symmetry is not a problem because the branch's earlier layers keep
their random weights. On the first backward pass, the zeroed projection
receives a nonzero gradient because its input is nonzero, while the earlier
layers receive zero gradient through that projection. After the projection
moves off zero, gradient reaches the whole branch. Keep this scheme distinct
from the previous one: GPT-2 makes every residual projection *small but
nonzero*, whereas zero-init makes one layer *exactly zero* so the block starts
as an exact identity.

### Watching the Variance Compound

Claims about variance at depth are cheap to test. We reuse a compact residual
block (it repeats the one from :numref:`sec_model_construction`) and stack
it $N$ deep:

```{.python .input #init-watching-the-variance-compound-1}
%%tab pytorch
class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(),
                                  nn.Linear(4 * d, d))

    def forward(self, X):
        return X + self.body(X)
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab jax
he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'truncated_normal')

class ResidualBlock(nnx.Module):
    def __init__(self, d, out_init=he, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.fc1 = nnx.Linear(d, 4 * d, kernel_init=he, rngs=rngs)
        self.fc2 = nnx.Linear(4 * d, d, kernel_init=out_init, rngs=rngs)

    def __call__(self, X):
        return X + self.fc2(nnx.relu(self.fc1(X)))
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab tensorflow
class ResidualBlock(tf.keras.Model):
    def __init__(self, d, out_init='he_normal'):
        super().__init__()
        self.body = tf.keras.Sequential([
            tf.keras.layers.Dense(4 * d, kernel_initializer='he_normal'),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(d, kernel_initializer=out_init)])

    def call(self, X):
        return X + self.body(X)
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab mxnet
he = init.MSRAPrelu(factor_type='in', slope=0.0)

class ResidualBlock(nn.Block):
    def __init__(self, d, out_init=he):
        super().__init__()
        self.body = nn.Sequential()
        self.body.add(nn.Dense(4 * d, activation='relu',
                               weight_initializer=he))
        self.body.add(nn.Dense(d, weight_initializer=out_init))

    def forward(self, X):
        return X + self.body(X)
```

:begin_tab:`pytorch`
Every linear layer gets He initialization, appropriate for the ReLU inside
the branch. Then a `named_parameters` loop, the same shape as nanoGPT's
initialization code, singles out each block's output projection and applies
one of three treatments: leave it alone, scale it by $1/\sqrt{N}$, or zero it.
:end_tab:

:begin_tab:`jax`
Every dense layer gets He initialization, appropriate for the ReLU inside
the branch; `variance_scaling(2.0, 'fan_in', 'truncated_normal')` is exactly
what `nnx.initializers.he_normal()` expands to. The block takes its output
projection's initializer as a constructor argument, so each treatment is a
different initializer.
Leaving it alone means He again, scaling by $1/\sqrt{N}$ is a closure over
the depth that wraps the same three-argument signature, and zeroing is
`nnx.initializers.zeros`.
:end_tab:

:begin_tab:`tensorflow`
Every dense layer gets He initialization, appropriate for the ReLU inside
the branch; the string `'he_normal'` names the same object as
`tf.keras.initializers.HeNormal()`. The treatment is a constructor argument
as well: the block takes its output projection's initializer as a parameter,
so each of the three treatments is just a different initializer. Leaving it
alone means He again and zeroing is `'zeros'`, but scaling by $1/\sqrt{N}$
has no menu entry, so we subclass `Initializer`: the instance stores the
depth, and its `__call__` shrinks a fresh He draw, behind the same
shape-and-dtype signature everything on the menu implements.
:end_tab:

:begin_tab:`mxnet`
Every dense layer gets He initialization through its `weight_initializer`
constructor argument, which sets the per-parameter initializer that
`net.initialize()` respects. The menu entry is `init.MSRAPrelu`, named after
the He et al. paper's lab: with `factor_type='in'` and `slope=0.0` (the PReLU
slope, zero for a plain ReLU) it draws a Gaussian with variance
$2/n_\textrm{in}$, exactly He normal. The treatment is a constructor argument
as well: the block takes its output projection's initializer, so each of the
three treatments is just a different initializer. Leaving it alone means He
again and zeroing is `init.Zero()`, but scaling by $1/\sqrt{N}$ has no menu
entry, so once more we subclass `Initializer`: the instance stores the depth,
and its `_init_weight` shrinks a fresh He draw in place.
:end_tab:

```{.python .input #init-watching-the-variance-compound-2}
%%tab pytorch
def init_he(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        nn.init.zeros_(module.bias)

def output_std(num_blocks, tweak):
    torch.manual_seed(0)
    net = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
    net.apply(init_he)
    for name, param in net.named_parameters():
        if name.endswith('body.2.weight'):  # each block's output projection
            tweak(param, num_blocks)
    torch.manual_seed(1)
    with torch.no_grad():
        return net(torch.randn(256, 64)).std().item()
```

```{.python .input #init-watching-the-variance-compound-2}
%%tab jax
def scaled_he(num_blocks):
    def init(key, shape, dtype=jnp.float32):
        return he(key, shape, dtype) * num_blocks ** -0.5
    return init

def output_std(num_blocks, out_init):
    rngs = nnx.Rngs(0)
    net = nnx.Sequential(*[
        ResidualBlock(64, out_init=out_init(num_blocks), rngs=rngs)
        for _ in range(num_blocks)])
    X = jax.random.normal(jax.random.key(1), (256, 64))
    return net(X).std().item()
```

```{.python .input #init-watching-the-variance-compound-2}
%%tab tensorflow
class ScaledHe(tf.keras.initializers.Initializer):
    def __init__(self, num_blocks):
        self.num_blocks = num_blocks

    def __call__(self, shape, dtype=None):
        w = tf.keras.initializers.HeNormal()(shape, dtype)
        return w * self.num_blocks ** -0.5

def output_std(num_blocks, out_init):
    tf.keras.utils.set_random_seed(0)
    net = tf.keras.Sequential(
        [ResidualBlock(64, out_init=out_init(num_blocks))
         for _ in range(num_blocks)])
    X = tf.random.normal((256, 64), seed=1)
    return float(tf.math.reduce_std(net(X)))
```

```{.python .input #init-watching-the-variance-compound-2}
%%tab mxnet
class ScaledHe(init.Initializer):
    def __init__(self, num_blocks):
        super().__init__(num_blocks=num_blocks)
        self.num_blocks = num_blocks

    def _init_weight(self, name, arr):
        he._init_weight(name, arr)
        arr *= self.num_blocks ** -0.5

def output_std(num_blocks, out_init):
    np.random.seed(0)
    net = nn.Sequential()
    net.add(*[ResidualBlock(64, out_init=out_init(num_blocks))
              for _ in range(num_blocks)])
    net.initialize()  # deferred: the draws happen at the forward pass
    X = np.random.normal(0, 1, (256, 64))
    return float(net(X).std())
```

One forward pass on unit-variance inputs per depth and treatment:

```{.python .input #init-watching-the-variance-compound-3}
%%tab pytorch
tweaks = {'default': lambda p, n: None,
          'scaled': lambda p, n: p.data.mul_(n ** -0.5),
          'zero': lambda p, n: nn.init.zeros_(p)}
print(f'{"N":>3}' + ''.join(f'{name:>10}' for name in tweaks))
for n in (2, 8, 32):
    stds = (output_std(n, tweak) for tweak in tweaks.values())
    print(f'{n:>3}' + ''.join(f'{s:>10.3g}' for s in stds))
```

```{.python .input #init-watching-the-variance-compound-3}
%%tab jax
tweaks = {'default': lambda n: he,
          'scaled': scaled_he,
          'zero': lambda n: nnx.initializers.zeros}
print(f'{"N":>3}' + ''.join(f'{name:>10}' for name in tweaks))
for n in (2, 8, 32):
    stds = (output_std(n, tweak) for tweak in tweaks.values())
    print(f'{n:>3}' + ''.join(f'{s:>10.3g}' for s in stds))
```

```{.python .input #init-watching-the-variance-compound-3}
%%tab tensorflow
tweaks = {'default': lambda n: 'he_normal',
          'scaled': ScaledHe,
          'zero': lambda n: 'zeros'}
print(f'{"N":>3}' + ''.join(f'{name:>10}' for name in tweaks))
for n in (2, 8, 32):
    stds = (output_std(n, tweak) for tweak in tweaks.values())
    print(f'{n:>3}' + ''.join(f'{s:>10.3g}' for s in stds))
```

```{.python .input #init-watching-the-variance-compound-3}
%%tab mxnet
tweaks = {'default': lambda n: he,
          'scaled': ScaledHe,
          'zero': lambda n: init.Zero()}
print(f'{"N":>3}' + ''.join(f'{name:>10}' for name in tweaks))
for n in (2, 8, 32):
    stds = (output_std(n, tweak) for tweak in tweaks.values())
    print(f'{n:>3}' + ''.join(f'{s:>10.3g}' for s in stds))
```

The default column multiplies by a roughly constant factor per block in this
unnormalized stack, reaching tens of millions by $N=32$. Such initial
activations make optimization impractical. The scaled column stays near a
small constant over the depths tested: scaling cuts each branch's variance by
a factor of $N$, following the independent-contribution approximation above.
The zero column reproduces the input's standard deviation, since every block
is the identity. This forward-pass test does not prove how an optimizer will
behave, but it catches an unusable initialization before training begins.

## Custom Initializers

Occasionally the menu has nothing you need.

:begin_tab:`pytorch`
An initializer is just a function that mutates a parameter, so writing one is
no harder than using one.
:end_tab:

:begin_tab:`jax`
An initializer is just a function `(key, shape, dtype) -> array`, the
signature everything in `nnx.initializers` shares, so writing one is no harder
than using one.
:end_tab:

:begin_tab:`tensorflow`
An initializer is just an object mapping `(shape, dtype)` to a tensor:
subclass `tf.keras.initializers.Initializer`, implement `__call__`, and every
layer will accept an instance, so writing one is no harder than using one.
:end_tab:

:begin_tab:`mxnet`
By now this is familiar: an initializer is a subclass of `init.Initializer`
whose `_init_weight` method fills the array it is handed, in place, and this
section has already written two (the truncated normal and the depth-scaled
He). Writing one is no harder than using one.
:end_tab:

Suppose, to make the point vividly, we want weights distributed as

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`pytorch`
Draw uniformly from $U(-10, 10)$, then zero every entry of magnitude below 5:
:end_tab:

:begin_tab:`jax`
Split the key, draw a magnitude from $U(5, 10)$, and draw a factor from
$\{-1, 0, 0, 1\}$, which zeroes the entry with probability one half and keeps
either sign with probability one quarter. Handing the function to
`kernel_init` makes it official; `init` gives every layer an independent key:
:end_tab:

:begin_tab:`tensorflow`
Draw uniformly from $U(-10, 10)$, then zero every entry of magnitude below 5.
Handing an instance to `kernel_initializer` makes it official:
:end_tab:

:begin_tab:`mxnet`
Draw uniformly from $U(-10, 10)$, then zero every entry of magnitude below 5.
Handing an instance to `net.initialize` makes it official:
:end_tab:

```{.python .input #init-custom-initializers-1}
%%tab pytorch
def my_init(module):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -10, 10)
        with torch.no_grad():
            module.weight *= module.weight.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input #init-custom-initializers-1}
%%tab jax
def my_init(key, shape, dtype=jnp.float32):
    mag_key, sign_key = jax.random.split(key)
    mag = jax.random.uniform(mag_key, shape, dtype, minval=5, maxval=10)
    sign = jax.random.choice(sign_key, jnp.array([-1., 0., 0., 1.], dtype),
                             shape)
    return mag * sign

net = nnx.Sequential(
    nnx.Linear(4, 8, kernel_init=my_init, rngs=nnx.Rngs(0)), nnx.relu,
    nnx.Linear(8, 1, rngs=nnx.Rngs(1)))
net.layers[0].kernel[:, :2]
```

```{.python .input #init-custom-initializers-1}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        w = tf.random.uniform(shape, -10, 10, dtype=dtype)
        return w * tf.cast(tf.abs(w) >= 5, w.dtype)

net = tf.keras.Sequential([
    tf.keras.layers.Dense(8, kernel_initializer=MyInit()),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1)])

net(X)
net.layers[0].kernel[:2]
```

```{.python .input #init-custom-initializers-1}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize(MyInit())
net(X)
net[0].weight.data()[:2]
```

:begin_tab:`pytorch`
The `torch.no_grad()` block is required: parameters are leaf tensors that
track gradients, and PyTorch rejects in-place arithmetic on them unless we
declare that the mutation is not part of any computation to differentiate.
(The `nn.init` routines run under their own `no_grad` internally, which is why
the first line needs none.) The same escape hatch handles one-off surgery,
such as offsetting a whole matrix or pinning a single entry:
:end_tab:

:begin_tab:`jax`
NNX parameters are mutable variables. Assignment updates the value owned by
the layer while preserving the `nnx.Param` object that optimizers and
checkpoints track. One-off surgery therefore looks like ordinary indexed
assignment:
:end_tab:

:begin_tab:`tensorflow`
No guard is needed on the way in, because `assign` sits outside automatic
differentiation: TensorFlow records gradients on a `GradientTape`, and an
assignment is a write to the variable's buffer, not an operation on the tape.
The same door handles one-off surgery, such as offsetting a whole matrix or
pinning a single entry, since slicing a variable composes with `assign`:
:end_tab:

:begin_tab:`mxnet`
No guard is needed on the way in, because gradients are only recorded inside
an `autograd.record()` block: outside one, indexed assignment is a plain
array write. `weight.data()` hands back the array the parameter holds on the
device, so the same door handles one-off surgery, such as offsetting a whole
matrix or pinning a single entry:
:end_tab:

```{.python .input #init-custom-initializers-2}
%%tab pytorch
with torch.no_grad():
    net[0].weight[:] += 1
    net[0].weight[0, 0] = 42
net[0].weight[0]
```

```{.python .input #init-custom-initializers-2}
%%tab jax
net.layers[0].kernel[...] += 1
net.layers[0].kernel[0, 0] = 42
net.layers[0].kernel[0]
```

```{.python .input #init-custom-initializers-2}
%%tab tensorflow
w = net.layers[0].kernel
w.assign(w + 1)
w[0, 0].assign(42)
w[0]
```

```{.python .input #init-custom-initializers-2}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

:begin_tab:`pytorch`
One caveat when building with lazy layers (:numref:`sec_lazy_init`): before
the first forward pass their parameters are placeholders with no shape, so
`apply`-based initializers and direct surgery alike must come *after* the dry
run that materializes them.
:end_tab:

:begin_tab:`jax`
There is no ordering caveat: NNX parameters are created in the constructor,
so both graph walks and direct assignment can run immediately.
:end_tab:

:begin_tab:`tensorflow`
The ordering caveat of the lazy world (:numref:`sec_lazy_init`) splits by
route here: a constructor initializer is stored and runs at build time, so it
can never fire too early, but `assign` surgery needs a kernel to exist and
must therefore follow the first call (or an explicit `build`). The cells
above respected that order by running `net(X)` before touching
`net.layers[0].kernel`.
:end_tab:

:begin_tab:`mxnet`
The ordering caveat of the lazy world (:numref:`sec_lazy_init`) splits by
route here too: `initialize` may run before shapes are known, since it only
records which initializer to use and the draw waits for the first forward
pass, but `weight.data()` raises a `DeferredInitializationError` until that
pass has happened. The cells above respected the order by running `net(X)`
before touching `net[0].weight.data()`.
:end_tab:

## Summary

:begin_tab:`pytorch`
PyTorch initializes parameters at construction with fan-aware defaults;
override them when depth, hand-made parameters, or a paper's recipe demands
it. The mechanism is one pattern: a function that dispatches on module type
and mutates parameters in place with the trailing-underscore `nn.init`
routines, walked over the model by `net.apply`.
:end_tab:

:begin_tab:`jax`
NNX initializes parameters in layer constructors with fan-aware defaults;
override them when depth or a paper's recipe demands it. An initializer is a function
`(key, shape, dtype) -> array`, handed to a layer as `kernel_init` at
construction or run over selected modules in an existing object graph.
:end_tab:

:begin_tab:`tensorflow`
Keras initializes parameters at build time, when a layer first learns its
input shape, with fan-aware defaults; override them when depth or a paper's
recipe demands it. The mechanism is one pattern: an initializer is an object
mapping `(shape, dtype)` to a tensor, handed to a layer as
`kernel_initializer` at construction or, for a model that already exists,
written into each selected kernel by a `model.layers` walk and `assign`.
:end_tab:

:begin_tab:`mxnet`
Gluon initializes parameters when you ask it to: `initialize()` records the
plan, the first forward pass supplies the shapes, and the default draw,
$U(-0.07, 0.07)$, is a legacy constant rather than a fan-aware rule, so
override it more readily than elsewhere. The mechanism is one pattern: an
`Initializer` object handed to `initialize` on the net, a block, or a single
parameter, with `force_reinit=True` to overwrite live values.
:end_tab:

On top of Xavier and He,
transformer-era code truncates normal tails to bound outliers, shrinks each
residual output projection by $1/\sqrt{N}$ so the stream's variance stays
$O(1)$ at any depth, and zero-initializes a block's last layer to start it as
the identity.

:begin_tab:`pytorch`
Anything the menu lacks is a few lines of tensor code under
`torch.no_grad()`.
:end_tab:

:begin_tab:`jax`
Anything the menu lacks is a few lines of `jax.random` code behind the same
three-argument signature.
:end_tab:

:begin_tab:`tensorflow`
Anything the menu lacks is an `Initializer` subclass with a few lines of
`tf.random` code in its `__call__`.
:end_tab:

:begin_tab:`mxnet`
Anything the menu lacks, truncation included, is an `init.Initializer`
subclass with a few lines of array code in its `_init_weight`.
:end_tab:

## Exercises

1. Instrument the residual stack: record the standard deviation of the
   activation after every block (run the stack one block at a time, or
   capture per-block activations with the tools of :numref:`sec_repro`)
   for the default and scaled treatments at $N=32$, and plot it against
   depth. Which curve matches the geometric-growth prediction?
1. Zero-initialize *all* layers of every block instead of just the output
   projection. The forward pass still returns the input, but what can the
   network learn? Work out which parameters receive a nonzero gradient, and
   relate the answer to the symmetry-breaking argument of
   :numref:`sec_numerical_stability`.
1. Write an initializer that fills each parameter from a dictionary keyed by
   parameter name (walk `net.named_parameters()` in PyTorch, `net.weights`
   in TensorFlow, `nnx.iter_modules(net)` in JAX, `net.collect_params()`
   in MXNet). You have re-invented part of checkpoint loading,
   which :numref:`sec_read_write` covers.
1. For a normal distribution truncated at $\pm 2\sigma$: what fraction of
   draws does the clip discard, and by how much does it shrink the standard
   deviation? Verify both numbers against the printed output of the truncation
   demo above.
