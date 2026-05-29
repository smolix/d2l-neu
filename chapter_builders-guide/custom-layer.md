```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Custom Layers

One factor behind deep learning's success
is the availability of a wide range of layers
that can be composed in creative ways
to design architectures suitable
for a wide variety of tasks.
For instance, researchers have invented layers
specifically for handling images, text,
looping over sequential data,
and
performing dynamic programming.
Sooner or later, you will need
a layer that does not exist yet in the deep learning framework.
In these cases, you must build a custom layer.
In this section, we show you how.

```{.python .input #custom-layer-custom-layers}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #custom-layer-custom-layers}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #custom-layer-custom-layers}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #custom-layer-custom-layers}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Layers without Parameters

To start, we construct a custom layer
that does not have any parameters of its own.
This should look familiar if you recall our
introduction to modules in :numref:`sec_model_construction`.
The following `CenteredLayer` class simply
subtracts the mean from its input.
To build it, we simply need to inherit
from the base layer class and implement the forward propagation function.

```{.python .input #custom-layer-layers-without-parameters-1}
%%tab mxnet
class CenteredLayer(nn.Block):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input #custom-layer-layers-without-parameters-1}
%%tab pytorch
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
```

```{.python .input #custom-layer-layers-without-parameters-1}
%%tab tensorflow
class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, X):
        return X - tf.reduce_mean(X)
```

```{.python .input #custom-layer-layers-without-parameters-1}
%%tab jax
class CenteredLayer(nn.Module):
    def __call__(self, X):
        return X - X.mean()
```

Let's verify that our layer works as intended by feeding some data through it.

```{.python .input #custom-layer-layers-without-parameters-2}
layer = CenteredLayer()
layer(d2l.tensor([1.0, 2, 3, 4, 5]))
```

We can now incorporate our layer as a component
in constructing more complex models.

```{.python .input #custom-layer-layers-without-parameters-3}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

```{.python .input #custom-layer-layers-without-parameters-3}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(128), CenteredLayer())
```

```{.python .input #custom-layer-layers-without-parameters-3}
%%tab tensorflow
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])
```

```{.python .input #custom-layer-layers-without-parameters-3}
%%tab jax
net = nn.Sequential([nn.Dense(128), CenteredLayer()])
```

As an extra sanity check, we can send random data
through the network and check that the mean is in fact 0.
Because we are dealing with floating point numbers,
we may still see a very small nonzero number
due to floating-point rounding.

:begin_tab:`jax`
Here we utilize the `init_with_output` method which returns both the output of
the network as well as the parameters. In this case we only focus on the
output.
:end_tab:

```{.python .input #custom-layer-layers-without-parameters-4}
%%tab pytorch, mxnet
Y = net(d2l.rand(4, 8))
Y.mean()
```

```{.python .input #custom-layer-layers-without-parameters-4}
%%tab tensorflow
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)
```

```{.python .input #custom-layer-layers-without-parameters-4}
%%tab jax
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (4, 8)))
Y.mean()
```

## Layers with Parameters

Now that we know how to define simple layers,
let's move on to defining layers with parameters
that can be adjusted through training.
We can use built-in functions to create parameters, which
provide some basic housekeeping functionality.
In particular, they govern access, initialization,
sharing, saving, and loading model parameters.
This way, among other benefits, we will not need to write
custom serialization routines for every custom layer.

Now let's implement our own version of the  fully connected layer.
Recall that this layer requires two parameters,
one to represent the weight and the other for the bias.
In this implementation, we bake in the ReLU activation as a default.
This layer requires two input arguments: `in_units` and `units`, which
denote the number of inputs and outputs, respectively.

```{.python .input #custom-layer-layers-with-parameters-1}
%%tab mxnet
from mxnet import gluon

class MyDense(nn.Block):
    def __init__(self, units, in_units):
        super().__init__()
        self.weight = gluon.Parameter('weight', shape=(in_units, units))
        self.bias = gluon.Parameter('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data(ctx=x.ctx)) + self.bias.data(
            ctx=x.ctx)
        return npx.relu(linear)
```

```{.python .input #custom-layer-layers-with-parameters-1}
%%tab pytorch
class MyDense(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        # Scaled init (Xavier-ish) keeps activations bounded on size-64 inputs
        self.weight = nn.Parameter(torch.randn(in_units, units) / in_units**0.5)
        self.bias = nn.Parameter(torch.zeros(units,))
        
    def forward(self, X):
        linear = torch.matmul(X, self.weight) + self.bias
        return F.relu(linear)
```

```{.python .input #custom-layer-layers-with-parameters-1}
%%tab tensorflow
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)
```

```{.python .input #custom-layer-layers-with-parameters-1}
%%tab jax
class MyDense(nn.Module):
    in_units: int
    units: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.normal(stddev=1),
                                 (self.in_units, self.units))
        self.bias = self.param('bias', nn.initializers.zeros, self.units)

    def __call__(self, X):
        linear = jnp.matmul(X, self.weight) + self.bias
        return nn.relu(linear)
```

:begin_tab:`mxnet, tensorflow, jax`
Next, we instantiate the `MyDense` class
and access its model parameters.
:end_tab:

:begin_tab:`pytorch`
Next, we instantiate the `MyDense` class
and access its model parameters.
:end_tab:

```{.python .input #custom-layer-layers-with-parameters-2}
%%tab mxnet
dense = MyDense(units=3, in_units=5)
dense.params
```

```{.python .input #custom-layer-layers-with-parameters-2}
%%tab pytorch
dense = MyDense(5, 3)
dense.weight
```

```{.python .input #custom-layer-layers-with-parameters-2}
%%tab tensorflow
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()
```

```{.python .input #custom-layer-layers-with-parameters-2}
%%tab jax
dense = MyDense(5, 3)
params = dense.init(d2l.get_key(), jnp.zeros((3, 5)))
params
```

We can directly carry out forward propagation calculations using custom layers.

```{.python .input #custom-layer-layers-with-parameters-3}
%%tab mxnet
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

```{.python .input #custom-layer-layers-with-parameters-3}
%%tab pytorch
dense(torch.rand(2, 5))
```

```{.python .input #custom-layer-layers-with-parameters-3}
%%tab tensorflow
dense(tf.random.uniform((2, 5)))
```

```{.python .input #custom-layer-layers-with-parameters-3}
%%tab jax
dense.apply(params, jax.random.uniform(d2l.get_key(),
                                       (2, 5)))
```

We can also construct models using custom layers.
Once we have that we can use it just like the built-in fully connected layer.

```{.python .input #custom-layer-layers-with-parameters-4}
%%tab mxnet
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

```{.python .input #custom-layer-layers-with-parameters-4}
%%tab pytorch
net = nn.Sequential(MyDense(64, 8), MyDense(8, 1))
net(torch.rand(2, 64))
```

```{.python .input #custom-layer-layers-with-parameters-4}
%%tab tensorflow
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))
```

```{.python .input #custom-layer-layers-with-parameters-4}
%%tab jax
net = nn.Sequential([MyDense(64, 8), MyDense(8, 1)])
Y, _ = net.init_with_output(d2l.get_key(), jax.random.uniform(d2l.get_key(),
                                                              (2, 64)))
Y
```

## Summary

We can design custom layers via the basic layer class. This allows us to define flexible new layers that behave differently from any existing layers in the library.
Once defined, custom layers can be invoked in arbitrary contexts and architectures.
Layers can have local parameters, which can be created through built-in functions.


## Exercises

1. Design a layer that takes an input and computes a tensor reduction,
   i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
1. Design a layer that returns the leading half of the Fourier coefficients of the data.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/58)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/59)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/279)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17993)
:end_tab:

<!-- slides -->

::: {.slide title="Custom Layers"}
`torch.nn` ships 100+ layers, but occasionally — a new
architecture, an unusual normalization, a custom block —
you need one the framework doesn't have.

Writing one is trivial: subclass `nn.Module`, override
`forward`. Two flavors:

- **Stateless** — pure transforms. Just override `forward`.
- **Stateful** — your own `Linear`, low-rank weight, etc.
  Wrap learnable tensors in `nn.Parameter`.

The custom layer composes with built-ins automatically —
`Sequential`, `parameters()`, `to(device)`, checkpointing.
:::

::: {.slide title="Stateless layer: a centering operator"}
Subtract the row-wise mean from each input. Nothing to
learn — pure transform:

@custom-layer-custom-layers

. . .

@custom-layer-layers-without-parameters-1

. . .

Standalone use:

@custom-layer-layers-without-parameters-2

The output mean is (numerically) zero — by construction.
:::

::: {.slide title="Composes with built-ins"}
Drop the custom layer into a `Sequential` like any other:

@custom-layer-layers-without-parameters-3

. . .

@custom-layer-layers-without-parameters-4

The framework can't tell `CenteredLayer` apart from
`Linear` or `ReLU` — they're all just `nn.Module`s.
:::

::: {.slide title="Stateful layer: hand-rolled Linear"}
Implement a fully-connected layer from scratch. The
*one* important step: wrap learnable tensors in
`nn.Parameter` so they're auto-registered for training:

@custom-layer-layers-with-parameters-1

. . .

@custom-layer-layers-with-parameters-2
:::

::: {.slide title="What `nn.Parameter` buys you"}
After `linear = MyLinear(5, 3)`:

- `linear.weight` and `linear.bias` are tracked parameters.
- `linear.parameters()` yields both — feed to the optimizer.
- `state_dict()` saves them; `linear.to('cuda')` moves them.

All for free, just by declaring `nn.Parameter` in
`__init__`.
:::

::: {.slide title="Test drive"}
@custom-layer-layers-with-parameters-3

. . .

Stack two `MyLinear`s — same `Sequential` plumbing as
built-in layers:

@custom-layer-layers-with-parameters-4
:::

::: {.slide title="When to write a custom layer"}
Real-world cases that justify a custom layer:

- **Novel architectural blocks** — gated linear units,
  factorized weight matrices, low-rank parameterizations
  (LoRA).
- **Custom normalization** — group norm with non-standard
  groups, layer-norm variants.
- **Tied/shared weights with structure** — embedding +
  output projection sharing in language models.
- **Frozen "buffers"** — running statistics in BatchNorm,
  position-specific masks. Use `register_buffer` for
  non-trainable tensors that should still travel with
  the module (saved, moved to GPU, etc.).
:::

::: {.slide title="Recap"}
- Custom layer = `nn.Module` subclass with a `forward`.
- Stateless: just override `forward`. Stateful: wrap
  learnable tensors in `nn.Parameter`.
- Use `register_buffer` for non-trainable state that
  should still travel with the module.
- Composes with built-in layers exactly the same as a
  built-in. No special handling.
- The escape hatch when the standard layer zoo doesn't
  cover what you actually need.
:::
