```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Initialization
:label:`sec_init_param`

:numref:`sec_numerical_stability` showed how to initialize one layer. Draw
each weight with standard deviation $g/\sqrt{n_\textrm{in}}$, where
$n_\textrm{in}$ is the number of inputs of the layer and the *gain* $g$
depends on what follows it: $g = \sqrt{2}$ before a ReLU, which is He
initialization, and $g = 1$ when no nonlinearity follows. Each layer then
passes
on the mean square of its input unchanged, so a stack of ten ReLU layers
keeps its activations roughly at the scale of its input. Xavier
initialization uses the average of the numbers of inputs and outputs
instead, so that the gradients keep their scale in the backward pass as
well.

In a residual network the activations grow even though every layer follows
this rule. Each residual block adds the output of a small network, its
*branch*, to its input. If the branch passes on the mean square of its
input, and its output is uncorrelated with that input, the sum has twice
the mean square of the input. After 32 blocks the standard deviation of the
activations has grown by a factor of $2^{16} \approx 65{,}000$, before
training has started.

The right initialization therefore depends on the architecture, not only on
each layer, and we need a way to apply a chosen rule to chosen parameters of
a model. This section first shows what the framework does when we choose
nothing, and then how to apply initializers to a whole model or to parts of
it. With these tools it fixes the residual network, by scaling down or
zeroing the last layer of each branch. It ends with initializers beyond the
defaults, truncated normals and arbitrary distributions, and with editing
parameters directly.

```{.python .input #init-initialization}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import init, np as mnp, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #init-initialization}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #init-initialization}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input #init-initialization}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import math
import jax
from jax import numpy as jnp
from flax import nnx
```

## Library Defaults

A framework initializes a layer when we do not, with a fixed default rule.
The default is not He initialization, so in general it does not keep the
scale of a deep stack of ReLU layers (:numref:`sec_numerical_stability`). We
check the default on a layer with 512 inputs and 256 outputs. A uniform
distribution on $[-b, b]$ has standard deviation $b/\sqrt{3}$.

:begin_tab:`mxnet`
Gluon separates construction from initialization: `nn.Dense(256)` carries no
numbers until we call `initialize()`, and when the input width is left
unspecified, even that call only records our choice and the draw waits for
the first forward pass (:numref:`sec_lazy_init`). Passing `in_units` fixes
the input width, so nothing is deferred here. The draw is $U(-0.07, 0.07)$
for every weight, a constant rather than a bound that depends on the size of
the layer, and the bias starts at zero:
:end_tab:

:begin_tab:`pytorch`
PyTorch initializes an `nn.Linear` when it is constructed: weight and bias
are both drawn from $U(-b, b)$ with $b = 1/\sqrt{n_\textrm{in}}$. (The
documentation describes the weight rule as `kaiming_uniform_` with
`a=math.sqrt(5)`, a variant of He initialization; the algebra reduces to the
bound above.)
:end_tab:

:begin_tab:`tensorflow`
Keras attaches no numbers at construction: `Dense(256)` fixes only the
output width, and the kernel is created at build time, when the layer first
learns its input shape (every Keras layer is lazy in the sense of
:numref:`sec_lazy_init`). The build draws Xavier uniform, which Keras calls
Glorot uniform after the first author of :citet:`Glorot.Bengio.2010`, and
the bias starts at zero:
:end_tab:

:begin_tab:`jax`
NNX creates parameters when `nnx.Linear` is constructed, using the input
width it is given. Its default is LeCun normal: a normal distribution with
standard deviation $1/\sqrt{n_\textrm{in}}$, truncated at two standard
deviations and then rescaled to restore the standard deviation, which puts
the largest possible entry near $2.3/\sqrt{n_\textrm{in}}$. The bias starts
at zero:
:end_tab:

```{.python .input #init-defaults-and-when-to-override-them}
%%tab mxnet
layer = nn.Dense(256, in_units=512)
layer.initialize()
w = layer.weight.data()
print(f'range [{float(w.min()):.4f}, {float(w.max()):.4f}], '
      f'predicted bound 0.0700')
print(f'std {float(w.std()):.4f}, predicted {0.07 / math.sqrt(3):.4f}')
print(f'bias all zero: {bool((layer.bias.data() == 0).all())}')
```

```{.python .input #init-defaults-and-when-to-override-them}
%%tab pytorch
layer = nn.Linear(512, 256)
bound = 1 / math.sqrt(layer.in_features)
w = layer.weight
print(f'range [{w.min():.4f}, {w.max():.4f}], predicted bound {bound:.4f}')
print(f'std {w.std():.4f}, predicted {bound / math.sqrt(3):.4f}')
```

```{.python .input #init-defaults-and-when-to-override-them}
%%tab tensorflow
layer = tf.keras.layers.Dense(256)
layer.build((None, 512))
w = layer.kernel.numpy()
bound = math.sqrt(6 / (512 + 256))
print(f'range [{w.min():.4f}, {w.max():.4f}], predicted bound {bound:.4f}')
print(f'std {w.std():.4f}, predicted {bound / math.sqrt(3):.4f}')
print(f'bias all zero: {bool((layer.bias.numpy() == 0).all())}')
```

```{.python .input #init-defaults-and-when-to-override-them}
%%tab jax
layer = nnx.Linear(512, 256, rngs=nnx.Rngs(d2l.get_key()))
w = layer.kernel
print(f'std {w.std():.4f}, predicted {1 / math.sqrt(512):.4f}')
print(f'max |w| {jnp.abs(w).max():.4f}, '
      f'bias all zero: {bool((layer.bias == 0).all())}')
```

When should we override the default? Four situations recur: a deep network
without normalization layers, where the factor of each layer compounds
across depth; architectures whose structure the per-layer rule ignores, such
as the residual networks of this section; reproducing a paper whose results
depend on its initialization; and parameters we create by hand.

:begin_tab:`mxnet`
For Gluon the first candidate is the default itself: $\pm 0.07$ ignores the
number of inputs, so it matches He initialization at one width only, and
layers much wider or much narrower start with too much or too little
variance.
:end_tab:

:begin_tab:`pytorch`
Parameters created by hand need particular care in PyTorch:
`nn.Parameter(torch.empty(...))` holds whatever bytes the allocator happened
to return.
:end_tab:

:begin_tab:`tensorflow`
Parameters created by hand get a default in Keras too: `add_weight` falls
back to Glorot uniform unless we pass an `initializer`, a sensible scale for
a weight matrix and an unsuitable one for a parameter with another role.
:end_tab:

:begin_tab:`jax`
Parameters created by hand get no default in NNX: we construct the value
ourselves and wrap it in `nnx.Param`.
:end_tab:

## Applying Initializers

:begin_tab:`mxnet`
To impose a scheme we hand an `Initializer` object to the model:
`net.initialize(init=..., force_reinit=True)` walks every parameter of the
block tree of :numref:`sec_model_construction` and runs the initializer on
it (`force_reinit` lifts the guard against overwriting a model that already
has values). The object is a default, not a decree: a parameter constructed
with its own initializer keeps it, and every `Dense` bias is constructed
with `zeros`, which is why the normal draw below leaves the biases at zero.
:end_tab:

:begin_tab:`pytorch`
To impose a scheme we do not touch layers one at a time. `net.apply(fn)`
walks the module tree of :numref:`sec_model_construction` and calls `fn` on
every submodule, children first. The function inspects each module's type
and decides what, if anything, to do with it. The routines in
`torch.nn.init` follow PyTorch's convention that a trailing underscore means
*in place*: `nn.init.normal_(w)` overwrites `w` rather than returning a new
tensor. The parameter must remain the tensor object that the module
registered; replacing it would leave `net.parameters()` and any existing
optimizer pointing at the old one.
:end_tab:

:begin_tab:`tensorflow`
To impose a scheme we declare it where the layer is declared: every Keras
layer accepts `kernel_initializer` and `bias_initializer` arguments, each an
object that maps a shape and a dtype to a tensor, and
`tf.keras.initializers` supplies the standard menu (the string `'zeros'`
names `tf.keras.initializers.Zeros()`, the default for biases). The
arguments are stored at construction and run at build time, so they wait
for lazy shapes. One rule
needs care: an initializer object created without a seed returns the same
values every time it is called. Two layers that share one such object draw
the same sequence of random numbers, so layers of equal shape start with
identical weights. Each layer therefore gets an object of its own:
:end_tab:

:begin_tab:`jax`
NNX layers such as `Linear` and `Conv` accept `kernel_init` and `bias_init`,
each a function `(key, shape, dtype) -> array`, and `nnx.initializers`
supplies the standard menu; biases start at zero by default. These
functions run in the constructor, and each layer draws with its own key
from `rngs`:
:end_tab:

```{.python .input #init-applying-initializers-1}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize(init=init.Normal(sigma=0.01))

X = mnp.ones((2, 4))
net(X)  # the input width was deferred, so the draw happens here
net[0].weight.data()[0], net[0].bias.data()[0]
```

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
%%tab tensorflow
def init_normal():   # a new initializer object for every layer
    return tf.keras.initializers.RandomNormal(stddev=0.01)

net = tf.keras.Sequential([
    tf.keras.layers.Dense(8, kernel_initializer=init_normal()),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(1, kernel_initializer=init_normal())])
X = tf.ones((2, 4))
net(X)
net.layers[0].kernel[:, 0].numpy(), net.layers[0].bias[0].numpy()
```

```{.python .input #init-applying-initializers-1}
%%tab jax
rngs = nnx.Rngs(d2l.get_key())
init_normal = nnx.initializers.normal(0.01)
net = nnx.Sequential(
    nnx.Linear(4, 8, kernel_init=init_normal, rngs=rngs), nnx.relu,
    nnx.Linear(8, 1, kernel_init=init_normal, rngs=rngs))

net.layers[0].kernel[:, 0], net.layers[0].bias[0]
```

:begin_tab:`mxnet`
Mixing schemes per block is `initialize` on the subtree: `net[0]` is a block
and `net[0].weight` is a parameter, and both accept an initializer directly,
with `force_reinit=True` because we overwrite live values. Below we give the
first layer Xavier initialization (:numref:`subsec_xavier`) and set the
second to a constant. By the symmetry argument of
:numref:`sec_numerical_stability`, a constant is a poor choice for training,
but it makes the mechanics visible:
:end_tab:

:begin_tab:`pytorch`
The `isinstance` dispatch is what makes the pattern compose: one function
can treat `nn.Linear`, `nn.Conv2d`, and `nn.Embedding` differently, and it
skips containers and activations. Because `apply` runs on any subtree, we
can also mix schemes per block. Below we give the first layer Xavier
initialization (:numref:`subsec_xavier`) and set the last to a constant. By
the symmetry argument of :numref:`sec_numerical_stability`, a constant is a
poor choice for training, but it makes the mechanics visible:
:end_tab:

:begin_tab:`tensorflow`
Mixing schemes per block needs no dispatch while the model is under
construction: each layer names its own initializer. Below we give the first
layer Xavier initialization (:numref:`subsec_xavier`) and set the last to a
constant. By the symmetry argument of :numref:`sec_numerical_stability`, a
constant is a poor choice for training, but it makes the mechanics visible:
:end_tab:

:begin_tab:`jax`
Mixing schemes per block needs no dispatch: each layer names its own
initializer where it is declared. Below we give the first layer Xavier
initialization (:numref:`subsec_xavier`) and set the last to a constant. By
the symmetry argument of :numref:`sec_numerical_stability`, a constant is a
poor choice for training, but it makes the mechanics visible:
:end_tab:

```{.python .input #init-applying-initializers-2}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
net[0].weight.data()[0], net[1].weight.data()
```

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
%%tab tensorflow
net = tf.keras.Sequential([
    tf.keras.layers.Dense(
        8, kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42.0))])

net(X)
net.layers[0].kernel[:, 0].numpy(), net.layers[2].kernel[:, 0].numpy()
```

```{.python .input #init-applying-initializers-2}
%%tab jax
net = nnx.Sequential(
    nnx.Linear(4, 8, kernel_init=nnx.initializers.xavier_uniform(),
               rngs=rngs), nnx.relu,
    nnx.Linear(8, 1, kernel_init=nnx.initializers.constant(42.0),
               rngs=rngs))

net.layers[0].kernel[:, 0], net.layers[2].kernel[:, 0]
```

:begin_tab:`mxnet`
There is no second route to learn, because `force_reinit=True` is already
the route for a model with live values: the two cells above initialized the
same network three times. Whole net, block, or single parameter, the call is
the same.
:end_tab:

:begin_tab:`pytorch`
This pattern is the whole interface. Later chapters package it as helper
functions, `init_cnn` for convolutional networks and `init_seq2seq` for
encoder-decoder models, but each is a function like `init_normal` handed to
`apply`.
:end_tab:

:begin_tab:`tensorflow`
The second route re-initializes a model that already exists. The loop below
is the whole mechanism: iterate over `model.layers`, dispatch with
`isinstance` so that activations and other layers without a kernel are
skipped, and `assign` a fresh draw to each kernel. `assign` overwrites the
variable in place, so it remains the variable that the model and any
optimizer track. Following the rule above, the function receives an
initializer *class* and creates one object per layer. It repairs the
constant-42 network of the previous cell; biases pass through untouched:
:end_tab:

:begin_tab:`jax`
The second route walks the existing object graph. `nnx.iter_modules` yields
each submodule with its path, and one fresh key per matching layer keeps the
draws independent. The function below draws every linear kernel of the
constant-42 network again; biases pass through untouched:
:end_tab:

```{.python .input #init-applying-initializers-3}
%%tab tensorflow
def reinit(model, initializer_class):
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            init = initializer_class()   # one object per layer
            layer.kernel.assign(init(layer.kernel.shape))

reinit(net, tf.keras.initializers.GlorotUniform)
net.layers[2].kernel[:3]
```

```{.python .input #init-applying-initializers-3}
%%tab jax
def reinit(model, key, init_fn):
    layers = [m for _, m in nnx.iter_modules(model)
              if isinstance(m, nnx.Linear)]
    for layer, subkey in zip(layers, jax.random.split(key, len(layers))):
        layer.kernel[...] = init_fn(
            subkey, layer.kernel.shape, layer.kernel.dtype)

reinit(net, d2l.get_key(), nnx.initializers.xavier_uniform())
net.layers[2].kernel[:3]
```

## Initialization of Residual Networks

### Variance Growth across Blocks

A residual block computes $\mathbf{x}_{k+1} = \mathbf{x}_k +
f_k(\mathbf{x}_k)$, where the branch $f_k$ is a small network. We use the
branch of the blocks of :numref:`sec_model_construction`, widened fourfold
as in many residual networks: a linear layer from $d$ to $4d$ features, a
ReLU, and a linear layer from $4d$ back to $d$, the *output projection* of
the branch.
The first layer gets He initialization, gain $\sqrt{2}$, because a ReLU
follows it. The output projection gets standard deviation
$g/\sqrt{4d}$ with a gain $g$ that we leave open for now; $g = 1$ is the
choice of :numref:`sec_numerical_stability` for a layer that feeds no
nonlinearity. All biases start at zero.

How does the mean square of the activations change across one block? Write
$m_k$ for the mean square of the entries of $\mathbf{x}_k$. The first layer
and the ReLU pass on $m_k$, by the argument for He initialization
(:numref:`subsec_he_init`), and the output projection multiplies it by
$4d \cdot g^2/(4d) = g^2$, so the branch output has mean square $g^2 m_k$.
The branch output is uncorrelated with $\mathbf{x}_k$, because the weights
of the output projection have mean zero and are independent of
$\mathbf{x}_k$. The mean square of the sum is therefore the sum of the mean
squares,

$$m_{k+1} = m_k + g^2 m_k = (1 + g^2)\, m_k.$$

Each block multiplies the mean square by $1 + g^2$, a factor above one for
every $g \neq 0$. With $g = 1$ it doubles, and after $N$ blocks the standard
deviation has grown by $2^{N/2}$: a factor of 2 after 2 blocks, 16 after 8,
and $65{,}536$ after 32. Every layer was initialized by the rule for one
layer, and the network as a whole still explodes, because the rule ignores
the skip connection, which adds the input back in every block.

### Scaling Down Residual Branches

The factor $1 + g^2$ per block compounds to $(1 + g^2)^N$ after $N$ blocks.
To keep the product bounded as the network gets deeper, the gain must
shrink with $N$. The choice $g^2 = 1/N$, that is, scaling the output
projection by $1/\sqrt{N}$, gives

$$m_N = \left(1 + \frac{1}{N}\right)^N m_0 \leq e\, m_0,$$

since $(1 + 1/N)^N$ increases toward $e$. The standard deviation then grows
by at most $\sqrt{e} \approx 1.65$, however deep the network: by 1.5 after
2 blocks, 1.60 after 8, and 1.64 after 32. The language model GPT-2
:cite:`Radford.Wu.Child.ea.2019` initializes its residual branches this way:
it scales the output projection of every branch by $1/\sqrt{N}$, where $N$
counts the branches.

### Starting a Block at Zero

The extreme choice is $g = 0$. Zero-initializing *every* weight is fatal:
all units in a layer compute the same output, receive the same gradient, and
stay identical forever (:numref:`sec_numerical_stability`). Zero-initializing
only the output projection is different. The branch then contributes
nothing, each block is the identity map, and the network starts as a
shallow function whose depth comes into play as training proceeds. Symmetry
is not a problem, because the branch's first layer keeps its random weights.
On the first backward pass, the output projection receives a nonzero
gradient, because its input is nonzero, while the first layer of the branch
receives zero gradient through the zero projection. Once the projection has
moved off zero, gradient reaches the whole branch.

### Measuring Variance across Depth
:label:`subsec_variance_across_depth`

We now check the three predictions by measurement. `WideResidualBlock` is
the residual block of :numref:`sec_model_construction` with its branch
widened fourfold and an explicit initialization: the gain of the output
projection is a constructor argument, and each layer gets its own normal
initializer with the standard deviation that the predictions assume:

```{.python .input #init-watching-the-variance-compound-1}
%%tab mxnet
class WideResidualBlock(nn.Block):
    def __init__(self, d, out_gain):
        super().__init__()
        self.body = nn.Sequential()
        self.body.add(
            nn.Dense(4 * d, activation='relu',
                     weight_initializer=init.Normal(math.sqrt(2 / d))),
            nn.Dense(d, weight_initializer=init.Normal(
                out_gain / math.sqrt(4 * d))))

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
            std = gain / math.sqrt(layer.in_features)
            nn.init.normal_(layer.weight, std=std)
            nn.init.zeros_(layer.bias)

    def forward(self, X):
        return X + self.body(X)
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab tensorflow
class WideResidualBlock(tf.keras.layers.Layer):
    def __init__(self, d, out_gain, **kwargs):
        super().__init__(**kwargs)
        normal = tf.keras.initializers.RandomNormal   # one object per layer
        self.body = tf.keras.Sequential([
            tf.keras.layers.Dense(
                4 * d, activation='relu',
                kernel_initializer=normal(stddev=math.sqrt(2 / d))),
            tf.keras.layers.Dense(
                d, kernel_initializer=normal(
                    stddev=out_gain / math.sqrt(4 * d)))])

    def call(self, X):
        return X + self.body(X)
```

```{.python .input #init-watching-the-variance-compound-1}
%%tab jax
class WideResidualBlock(nnx.Module):
    def __init__(self, d, out_gain, *, rngs):
        normal = nnx.initializers.normal
        self.body = nnx.Sequential(
            nnx.Linear(d, 4 * d, kernel_init=normal(math.sqrt(2 / d)),
                       rngs=rngs),
            nnx.relu,
            nnx.Linear(4 * d, d,
                       kernel_init=normal(out_gain / math.sqrt(4 * d)),
                       rngs=rngs))

    def __call__(self, X):
        return X + self.body(X)
```

The function below stacks `num_blocks` blocks of width 64, feeds them 256
examples with entries drawn from $\mathcal{N}(0, 1)$, and records the
standard deviation of the activations at the input and after every block:

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

The three treatments, unscaled, scaled, and zero, set the gain of the output projection to 1, to
$1/\sqrt{N}$, and to 0. We print the standard deviation after the last
block for $N = 2, 8, 32$ and plot it after every block for $N = 32$:

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

The measurements follow the predictions. The unscaled column grows by about
$\sqrt{2}$ per block, a straight line on the logarithmic axis. After 32
blocks it lands between about 20,000 and 200,000, in two runs out of three
below the predicted 65,536. The prediction is the square root of the
expected mean square; the factor of each block fluctuates with its random
weights, the fluctuations compound, and the typical run lies below the
square root of the average. Initial activations this large
make optimization impractical. The scaled column stays between about 1.4 and
1.9 at every depth, near the predicted 1.5 to 1.65, and the zero column keeps
the standard deviation of the input, 1, since every block is the identity.
A forward pass does not show how an optimizer will behave, but it catches an
unusable initialization before training begins.

## Custom Initializers and Direct Edits

Beyond the defaults, models use initializers of other shapes: truncated
normals and arbitrary distributions, which we write ourselves when the
library lacks them. Sometimes we also set values by hand. An initializer
we write ourselves has a fixed interface:

:begin_tab:`mxnet`
In Gluon, an initializer is a subclass of `init.Initializer` whose
`_init_weight` method fills the array it is handed, in place.
:end_tab:

:begin_tab:`pytorch`
In PyTorch, an initializer is a function that overwrites a parameter in
place.
:end_tab:

:begin_tab:`tensorflow`
In Keras, an initializer is an object that maps `(shape, dtype)` to a
tensor: subclass `tf.keras.initializers.Initializer` and implement
`__call__`. Layers then accept an instance through their initializer
arguments.
:end_tab:

:begin_tab:`jax`
In NNX, an initializer is any function with the signature
`(key, shape, dtype) -> array`.
:end_tab:

### Truncated Normals
:label:`subsec_truncated_normals`

A normal distribution gets the variance right, but its tails are unbounded.
That is harmless for one draw, while at scale a value far in the tail is a
near-certainty: among the $10^8$ weights of a large model, dozens land beyond
five standard deviations. Large weights also use up dynamic range once a
model is stored in a low-precision format (:numref:`sec_numerics`). Large
language and vision models such as BERT and the vision transformer therefore
use a *truncated normal*, which cuts the tails off at a fixed multiple of
the nominal standard deviation
:cite:`Devlin.Chang.Lee.ea.2018,Dosovitskiy.Beyer.Kolesnikov.ea.2021`. Cutting
the tails also removes variance, which some initializers restore by
rescaling and others do not.

:begin_tab:`mxnet`
`mxnet.init` provides no truncated normal, so we write the drawing function.
It draws normal values and redraws every entry that lands outside two
standard deviations until none does, a discard-and-redraw rule. Wrapped in
an `init.Initializer` subclass, as in the next subsection, it becomes an
initializer:
:end_tab:

:begin_tab:`pytorch`
`nn.init.trunc_normal_` takes absolute cutoffs `a` and `b`, which default
to $\pm 2$ and so mean $\pm 2$ standard deviations only when `std=1`. A cut
at two standard deviations must state them:
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.initializers.TruncatedNormal` cuts at two standard deviations:
draws outside the bound are discarded and redrawn, and no rescaling follows.
The factory behind `HeNormal` and its relatives draws a truncated normal too
and then rescales it so that the standard deviation lands on target:
:end_tab:

:begin_tab:`jax`
`nnx.initializers.truncated_normal` states its cutoffs in units of the
standard deviation (`lower=-2.0`, `upper=2.0` by default), so the cut at two
standard deviations needs no extra arguments. Flax truncates by default:
the `variance_scaling` factory behind `lecun_normal`, `xavier_normal`, and
`he_normal` draws its normal variants truncated and rescaled:
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

A million plain normal draws produce entries near $5\sigma = 0.1$; the
truncated version guarantees $|w| \leq 0.04$. Its standard deviation falls
from 0.02 to about 0.0176, because truncation removes the mass in the
tails. Recipes with a fixed standard deviation, such as BERT's, accept this
smaller variance; a recipe whose derivation needs a target variance should
rescale after truncating.

### A Custom Distribution

Suppose we want weights distributed as

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

that is, half of the weights are zero and the others have magnitudes between
5 and 10. We draw uniformly from $U(-10, 10)$ and then zero every entry of
magnitude below 5.

:begin_tab:`mxnet`
Handing an instance to `net.initialize` makes it the initializer of every
weight:
:end_tab:

:begin_tab:`pytorch`
The function overwrites each linear layer's weight:
:end_tab:

:begin_tab:`tensorflow`
Handing an instance to `kernel_initializer` makes it the initializer of
that layer:
:end_tab:

:begin_tab:`jax`
Handing the function to `kernel_init` makes it the initializer of that
layer:
:end_tab:

```{.python .input #init-custom-initializers-1}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        data[:] = mnp.random.uniform(-10, 10, data.shape)
        data *= mnp.abs(data) >= 5

net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'), nn.Dense(1))
net.initialize(MyInit())
net(X)
net[0].weight.data()[:2]
```

```{.python .input #init-custom-initializers-1}
%%tab pytorch
def my_init(module):
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -10, 10)
        with torch.no_grad():
            module.weight *= module.weight.abs() >= 5

net.apply(my_init)
net[0].weight.data[:2]
```

```{.python .input #init-custom-initializers-1}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        w = tf.random.uniform(shape, -10, 10, dtype=dtype)
        return w * tf.cast(tf.abs(w) >= 5, w.dtype)

net = tf.keras.Sequential([
    tf.keras.layers.Dense(8, 'relu', kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1)])
net(X)
net.layers[0].kernel[:, :2]
```

```{.python .input #init-custom-initializers-1}
%%tab jax
def my_init(key, shape, dtype=jnp.float32):
    w = jax.random.uniform(key, shape, dtype, minval=-10, maxval=10)
    return w * (jnp.abs(w) >= 5)

net = nnx.Sequential(
    nnx.Linear(4, 8, kernel_init=my_init, rngs=rngs), nnx.relu,
    nnx.Linear(8, 1, rngs=rngs))
net.layers[0].kernel[:, :2]
```

### Direct Edits

:begin_tab:`mxnet`
No guard is needed above, because gradients are recorded only inside an
`autograd.record()` block: outside one, indexed assignment is a plain array
write. `weight.data()` hands back the array that the parameter holds, so the
same interface supports direct edits of parameters, such as offsetting a
whole matrix or setting a single entry:
:end_tab:

:begin_tab:`pytorch`
The `torch.no_grad()` block above is required: parameters are leaf tensors
that track gradients, and PyTorch rejects in-place arithmetic on them unless
we declare that the change is not part of any computation to differentiate.
(The `nn.init` routines run under their own `no_grad` internally, which is
why the first line needs none.) The same context supports direct edits of
parameters, such as offsetting a whole matrix or setting a single entry:
:end_tab:

:begin_tab:`tensorflow`
No guard is needed above, because `assign` sits outside automatic
differentiation: TensorFlow records gradients on a `GradientTape`, and an
assignment writes to the variable without recording anything. The same
interface supports direct edits of parameters, such as offsetting a whole
matrix or setting a single entry, because slicing a variable composes with
`assign`:
:end_tab:

:begin_tab:`jax`
NNX parameters are mutable variables. Assignment updates the value owned by
the layer while keeping the `nnx.Param` object that optimizers and
checkpoints track. Direct edits of parameters, such as offsetting a whole
matrix or setting a single entry, therefore use ordinary indexed
assignment:
:end_tab:

```{.python .input #init-custom-initializers-2}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input #init-custom-initializers-2}
%%tab pytorch
with torch.no_grad():
    net[0].weight[:] += 1
    net[0].weight[0, 0] = 42
net[0].weight.data[0]
```

```{.python .input #init-custom-initializers-2}
%%tab tensorflow
w = net.layers[0].kernel
w.assign(w + 1)
w[0, 0].assign(42)
w[:, 0]
```

```{.python .input #init-custom-initializers-2}
%%tab jax
net.layers[0].kernel[...] += 1
net.layers[0].kernel[0, 0] = 42
net.layers[0].kernel[:, 0]
```

:begin_tab:`mxnet`
The order matters for lazy layers (:numref:`sec_lazy_init`): `initialize`
may run before the shapes are known, since it only records which initializer
to use, but `weight.data()` raises a `DeferredInitializationError` until the
first forward pass has happened. The cells above respected this order by
running `net(X)` before touching `net[0].weight.data()`.
:end_tab:

:begin_tab:`pytorch`
The order matters for lazy layers (:numref:`sec_lazy_init`): before the
first forward pass their parameters are placeholders with no shape, so
initialization with `apply` and direct assignment must follow the dry run
that creates them.
:end_tab:

:begin_tab:`tensorflow`
The order matters for lazy layers (:numref:`sec_lazy_init`): a constructor
initializer is stored and runs when the variables are created, but a direct
`assign` needs a kernel that exists and must follow the first call (or an
explicit `build`). The cells above respected this order by running `net(X)`
before touching `net.layers[0].kernel`.
:end_tab:

:begin_tab:`jax`
The order of operations never matters here: NNX creates parameters in the
constructor, so walks over the graph and direct assignment can run at once.
:end_tab:

We can now apply any initializer, from the library or of our own, to any
part of a model, and we have seen that the right choice depends on how the
layers are connected. Initialization only sets the starting point. Keeping
the activations at scale during training takes more, such as the
normalization layers of :numref:`sec_batch_norm`.

## Summary

A framework initializes its layers by default, with a fixed rule that is
not He initialization, so a model whose behavior matters sets its
initialization explicitly. Initializers are
applied over the tree of a model: as constructor arguments, as a function
applied to every submodule, or by assigning new values to selected
parameters. The per-layer rules of :numref:`sec_numerical_stability` are
not enough for a residual network: each block adds its branch to its input
and multiplies the mean square of the activations by $1 + g^2$, where $g$ is
the gain of the branch's output projection. With $g = 1$ the standard
deviation grows by $\sqrt{2}$ per block; scaling the output projection by
$1/\sqrt{N}$ bounds the growth by $\sqrt{e}$ at any depth $N$, and a zero
output projection makes every block start as the identity. Truncated normals
bound the largest weights, and custom distributions take a few lines of
code.

## Exercises

:begin_tab:`mxnet`
1. [code] **Default scale.** Gluon draws every `Dense` weight from
   $U(-0.07, 0.07)$, whatever the layer's shape, and starts the bias at
   zero.
    1. Compute, as a function of the width $n$, the factor by which an
       $n \times n$ layer followed by a ReLU multiplies the second moment
       of its input. At which width does it equal the factor of one that He
       initialization achieves (:numref:`subsec_he_init`)?
    1. Stack ten `nn.Dense(n, activation='relu')` layers, keep the default
       initialization, and push unit-variance inputs through, once for
       $n = 512$ and once for $n = 4096$. Compare the second moment of the
       output, the mean of its squared entries, with your predictions.
:end_tab:

:begin_tab:`pytorch`
1. [code] **Default scale.** PyTorch draws both the weight and the bias of
   an `nn.Linear` from $U(-1/\sqrt{n_\textrm{in}}, 1/\sqrt{n_\textrm{in}})$.
    1. With the biases set to zero, compute the factor by which one such
       layer followed by a ReLU multiplies the second moment of its input,
       and compare it with the factor of one that He initialization
       achieves (:numref:`subsec_he_init`).
    1. Stack ten `nn.Linear(512, 512)` layers, each followed by a ReLU,
       keep the default weights, zero the biases, and push unit-variance
       inputs through. Compare the second moment of the output, the mean
       of its squared entries, with your prediction.
    1. Restore the default biases and repeat the measurement. Explain why
       the second moment stops shrinking after a few layers, and predict
       the value at which it levels off.
:end_tab:

:begin_tab:`tensorflow`
1. [code] **Default scale.** Keras draws the kernel of a `Dense` layer
   from Glorot uniform, $U(\pm\sqrt{6/(n_\textrm{in} + n_\textrm{out})})$,
   and starts the bias at zero.
    1. For a layer with $n_\textrm{in} = n_\textrm{out}$, compute the
       factor by which the layer followed by a ReLU multiplies the second
       moment of its input, and compare it with the factor of one that He
       initialization achieves (:numref:`subsec_he_init`).
    1. Stack ten `Dense(512)` layers, each followed by a ReLU, with their
       default initialization, and push unit-variance inputs of width 512
       through. Compare the second moment of the output, the mean of its
       squared entries, with your prediction.
:end_tab:

:begin_tab:`jax`
1. [code] **Default scale.** NNX draws the kernel of an `nnx.Linear` from
   LeCun normal, a truncated normal rescaled to standard deviation
   $1/\sqrt{n_\textrm{in}}$, and starts the bias at zero.
    1. Compute the factor by which one such layer followed by a ReLU
       multiplies the second moment of its input, and compare it with the
       factor of one that He initialization
       achieves (:numref:`subsec_he_init`).
    1. Stack ten `nnx.Linear(512, 512)` layers, each followed by a ReLU,
       with their default initialization, and push unit-variance inputs
       through. Compare the second moment of the output, the mean of its
       squared entries, with your prediction.
:end_tab:

2. **Truncated normal.** For a normal distribution truncated at
   $\pm 2\sigma$: what fraction of draws does the cut discard, and by how
   much does it shrink the standard deviation? Neither quantity has an
   elementary closed form; express both in terms of the standard normal
   density and distribution function and evaluate them using tabulated
   values. Verify the standard deviation against the printed output of the
   demonstration in :numref:`subsec_truncated_normals`, and the discarded
   fraction by counting the entries of its plain normal draw that lie
   beyond $\pm 2\sigma$.

1. **Zero initialization.** Zero-initialize *all* layers of every block
   instead of just the output projection. The forward pass still returns the
   input, but what can the network learn? Work out which parameters receive
   a nonzero gradient, and relate the answer to the symmetry argument
   of :numref:`sec_numerical_stability`.

1. [code] **Normalized branches.** Many residual networks normalize the
   input of each branch: the branch reads $\mathbf{x}_k / r_k$, where $r_k$
   is the root mean square of the entries of $\mathbf{x}_k$ for each
   example, instead of $\mathbf{x}_k$.
    1. Show that the branch output then has mean square $g^2$, whatever the
       scale of $\mathbf{x}_k$, and that with $m_0 = 1$ the mean square
       after $N$ blocks is $1 + N g^2$. How does the growth with depth
       compare with the unnormalized blocks of this section?
    1. Which gain keeps the mean square at the end of the stack equal to 2
       for every $N$? Compare it with the scaling of GPT-2.
    1. Add the normalization to `WideResidualBlock`, measure the standard
       deviation across depth for $g = 1$ and $g = 1/\sqrt{N}$ at $N = 32$,
       and compare with your predictions and with the
       measured plot in :numref:`subsec_variance_across_depth`.

1. [code] **Kaiming initialization.** Implement the Kaiming (He)
   initializers of :numref:`subsec_he_init`, uniform and normal, with an
   explicit `nonlinearity` argument that switches the gain between `'relu'`
   and `'linear'`.
    1. Check both on the weight of a layer with 512 inputs and 256
       outputs by mean, standard deviation, and range: the normal
       initializer against the explicit normal initializers of
       `WideResidualBlock`, and the uniform initializer against its
       predicted bound and standard deviation. Explain any difference in
       range.
    1. Apply your normal initializer to the stack of
       :numref:`subsec_variance_across_depth` with the `'relu'` gain on
       both layers of every block. Predict the factor by which each block
       multiplies the mean square, and check it against a measured curve
       added to the plot of :numref:`subsec_variance_across_depth`.

    *Adapted from CMU 10-714, [Homework 2](https://github.com/dlsyscourse/hw2)
    and [Homework 4](https://github.com/dlsyscourse/hw4).*

1. [code] **Gradients across depth.** ● The backward pass through a block
   multiplies the gradient by the transposed Jacobian of
   $\mathbf{x} \mapsto \mathbf{x} + f(\mathbf{x})$.
    1. Repeat the argument of this section for the backward pass. Show that
       the branch multiplies the mean square of the gradient by $g^2$,
       although the backward pass sums over the outputs of each layer
       rather than its inputs, so that each block again multiplies it by
       $1 + g^2$.
    1. Verify the prediction for the three treatments at $N = 32$: compute
       the gradient of the sum of the outputs of the stack with respect to
       its input, and compare its mean square with that of the gradient at
       the output.

1. [code] **Data-driven calibration.** ● Initialize the stack of
   :numref:`subsec_variance_across_depth` with the unscaled treatment, then
   calibrate it block by block on one minibatch of unit-variance inputs.
   For each block in turn, run the minibatch forward to that block,
   measure the standard deviation of its branch output
   $f_k(\mathbf{x}_k)$, and rescale the block's output projection so that
   this standard deviation becomes $1/\sqrt{N}$. This adapts the
   layer-sequential unit-variance initialization of
   :citet:`Mishkin.Matas.2016` to a residual stack.
    1. Explain why the blocks must be calibrated in order rather than all
       at once from a single forward pass.
    1. Explain why the calibration targets the branch output rather than
       the block output $\mathbf{x}_k + f_k(\mathbf{x}_k)$.
    1. Predict the standard deviation at the end of the calibrated stack.
       Plot the calibrated depth-versus-standard-deviation curve next to
       that of the scaled treatment, and explain the difference.

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5.3]{.kicker}

Initialization<br>**library defaults · initializers over the tree · residual scaling · custom initializers**
:::
:::

::: {.slide title="He in Every Layer, Yet 32 Residual Blocks Grow Activations 65,000-Fold"}
[The problem]{.kicker}

- One layer: weights with standard deviation $g/\sqrt{n_\textrm{in}}$ keep the
  mean square of the activations; gain $g = \sqrt{2}$ before a ReLU (He),
  $g = 1$ when no nonlinearity follows.
- A residual block adds a branch to its input:
  $\mathbf{x}_{k+1} = \mathbf{x}_k + f_k(\mathbf{x}_k)$.
- Branch and input of equal mean square, uncorrelated: the sum has twice
  the mean square.

::: {.d2l-note .warn}
32 blocks: the standard deviation grows by $2^{16} \approx 65{,}000$ before
training starts.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- check what your framework draws when you choose nothing;
- apply an initializer to a whole model, one block, or one parameter;
- derive the factor $1 + g^2$ by which a residual block multiplies the mean
  square;
- choose the gain of the branch's output projection for depth $N$;
- write an initializer the library lacks.

**Running example:** a stack of $N \le 32$ residual blocks of width 64,
branch $64 \to 256 \to 64$, fed 256 examples from $\mathcal{N}(0, 1)$.
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
| Number Formats and Mixed Precision | dtypes, mixed precision, loss scaling |
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

::: {.slide title="The Default Initialization Is Not He" only="mxnet"}
[Library defaults]{.kicker}

Gluon `Dense`: weight $U(\pm 0.07)$, whatever the shape of the layer; zero
bias.

A $512 \to 256$ layer, checked against the prediction ($U(\pm b)$ has
standard deviation $b/\sqrt{3}$):

@!init-defaults-and-when-to-override-them
:::

::: {.slide title="The Default Initialization Is Not He" only="pytorch"}
[Library defaults]{.kicker}

PyTorch `nn.Linear`: weight and bias $U(\pm 1/\sqrt{n_\textrm{in}})$.

A $512 \to 256$ layer, checked against the prediction ($U(\pm b)$ has
standard deviation $b/\sqrt{3}$):

@!init-defaults-and-when-to-override-them
:::

::: {.slide title="The Default Initialization Is Not He" only="tensorflow"}
[Library defaults]{.kicker}

Keras `Dense`: Xavier (Glorot) uniform,
$U(\pm\sqrt{6/(n_\textrm{in} + n_\textrm{out})})$; zero bias.

A $512 \to 256$ layer, checked against the prediction ($U(\pm b)$ has
standard deviation $b/\sqrt{3}$):

@!init-defaults-and-when-to-override-them
:::

::: {.slide title="The Default Initialization Is Not He" only="jax"}
[Library defaults]{.kicker}

Flax `nnx.Linear`: LeCun normal, truncated at two standard deviations and
rescaled to standard deviation $1/\sqrt{n_\textrm{in}}$; zero bias.

A $512 \to 256$ layer, checked against the prediction ($U(\pm b)$ has
standard deviation $b/\sqrt{3}$):

@!init-defaults-and-when-to-override-them
:::

::: {.slide title="Override the Default in Four Situations"}
[Library defaults]{.kicker}

- deep networks without normalization layers: the factor of each layer
  compounds;
- architectures the per-layer rule ignores, such as residual networks;
- reproducing a paper whose results depend on its initialization;
- parameters created by hand.

::: {.d2l-note .rule}
A model whose behavior should not depend on the framework sets its
initialization explicitly.
:::
:::

::: {.slide title="Check: How Fast Does a Uniform Default Shrink Ten ReLU Layers?"}
[Check]{.kicker}

Ten layers $512 \to 512$, weights from $U(\pm 1/\sqrt{n_\textrm{in}})$, zero
biases, each followed by a ReLU. By which factor does one layer multiply the
mean square of the activations, and ten layers?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The weight variance is $b^2/3 = 1/(3 n_\textrm{in})$, so one
layer gives $n_\textrm{in} \cdot \frac{1}{3 n_\textrm{in}} \cdot \frac12 =
\frac16$, and ten layers $6^{-10} \approx 1.7 \times 10^{-8}$.
:::
:::
:::

::: {.slide title="An Initializer Is Applied over the Model's Tree"}
[Applying initializers]{.kicker}

Every weight from $\mathcal{N}(0, 0.01^2)$, every bias zero:

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

::: {.slide title="Each Block Can Get Its Own Scheme"}
[Applying initializers]{.kicker}

First layer Xavier, last layer the constant 42 (a poor choice for training,
by symmetry, but easy to see):

@init-applying-initializers-2
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Residual Networks]{.dtitle}

[the factor 1 + g² per block]{.dsub}
:::
:::

::: {.slide title="The Branch Passes On g² Times the Mean Square"}
[Variance growth]{.kicker}

Branch: $d \to 4d$ (He, gain $\sqrt 2$), ReLU, $4d \to d$ (the *output
projection*, gain $g$); biases zero. Let $m_k$ be the mean square of
$\mathbf{x}_k$.

- First layer and ReLU pass on $m_k$ (He).
- Output projection: $4d$ inputs times variance $g^2/(4d)$: factor $g^2$.
- Branch output has mean square $g^2 m_k$.
:::

::: {.slide title="Each Block Multiplies the Mean Square by 1 + g²"}
[Variance growth]{.kicker}

The branch output is uncorrelated with $\mathbf{x}_k$: the projection's
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
[Scaling down residual branches]{.kicker}

With $g^2 = 1/N$:

$$m_N = \left(1 + \frac{1}{N}\right)^N m_0 \le e\, m_0,$$

so the standard deviation grows by at most $\sqrt e \approx 1.65$ at any
depth: 1.5 for $N = 2$, 1.60 for 8, 1.64 for 32.

::: {.d2l-note}
GPT-2 scales the output projection of every residual branch by
$1/\sqrt{N}$, with $N$ the number of branches.
:::
:::

::: {.slide title="A Zero Output Projection Starts Every Block as the Identity"}
[Starting a block at zero]{.kicker}

- $g = 0$: the branch contributes nothing; the network starts shallow.
- Not the symmetry trap: the branch's first layer stays random.
- First backward pass: the zero projection gets a nonzero gradient (its
  input is nonzero); the first layer gets none through it.
- Once the projection moves off zero, gradient reaches the whole branch.

::: {.d2l-note .warn}
Zeroing *every* layer is fatal: all units stay identical forever.
:::
:::

::: {.slide title="Measured: Unscaled Grows √2 per Block; Scaled and Zero Stay Flat"}
[Measuring variance across depth]{.kicker}

Standard deviation after the last block, and after every block for $N = 32$:

@!init-watching-the-variance-compound-3
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Custom Initializers and Direct Edits]{.dtitle}

[truncated normals · any distribution · direct edits]{.dsub}
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

::: {.slide title="A Custom Initializer Is a Few Lines of Code"}
[A custom distribution]{.kicker}

Half of the weights zero, the others with magnitude in $[5, 10]$: draw from
$U(-10, 10)$, zero every entry below 5 in magnitude.

@init-custom-initializers-1
:::

::: {.slide title="Parameters Can Also Be Edited in Place"}
[Direct edits]{.kicker}

Offset the whole first layer by 1, then set one entry to 42:

@init-custom-initializers-2
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

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Defaults** are fixed rules, not He; set the initialization
  explicitly.
- **Applying:** constructor arguments, a function over every submodule, or
  assignment to chosen parameters.
- **Residual blocks** multiply the mean square by $1 + g^2$: with $g = 1$
  the standard deviation after 32 blocks is predicted 65,536, measured
  20,000 to 200,000.
- **Scaling** the output projection by $1/\sqrt N$ bounds the growth by
  $\sqrt e \approx 1.65$; **zero** makes every block the identity.
- **Custom:** truncated normals and any distribution in a few lines.

**Next:** layers the library does not provide.
:::
