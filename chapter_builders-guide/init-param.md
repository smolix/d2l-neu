```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Parameter Initialization

Now that we know how to access the parameters,
let's look at how to initialize them properly.
We discussed the need for proper initialization in :numref:`sec_numerical_stability`.
The deep learning framework provides default random initializations to its layers.
However, we often want to initialize our weights
according to various other protocols. The framework provides most commonly
used protocols, and also allows one to create a custom initializer.

```{.python .input #init-param-parameter-initialization-1}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #init-param-parameter-initialization-1}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input #init-param-parameter-initialization-1}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input #init-param-parameter-initialization-1}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

:begin_tab:`mxnet`
By default, MXNet initializes weight parameters by randomly drawing from a uniform distribution $U(-0.07, 0.07)$,
clearing bias parameters to zero.
MXNet's `init` module provides a variety
of preset initialization methods.
:end_tab:

:begin_tab:`pytorch`
By default, PyTorch initializes weight and bias matrices
uniformly by drawing from a range that is computed according to the input and output dimension.
PyTorch's `nn.init` module provides a variety
of preset initialization methods.
:end_tab:

:begin_tab:`tensorflow`
By default, Keras initializes weight matrices uniformly by drawing from a range that is computed according to the input and output dimension, and the bias parameters are all set to zero.
TensorFlow provides a variety of initialization methods both in the root module and the `keras.initializers` module.
:end_tab:

:begin_tab:`jax`
By default, Flax initializes weights using `jax.nn.initializers.lecun_normal`,
i.e., by drawing samples from a truncated normal distribution centered on 0 with
the standard deviation set as the squared root of $1 / \textrm{fan}_{\textrm{in}}$
where `fan_in` is the number of input units in the weight tensor. The bias
parameters are all set to zero.
Jax's `nn.initializers` module provides a variety
of preset initialization methods.
:end_tab:

```{.python .input #init-param-parameter-initialization-2}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input #init-param-parameter-initialization-2}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input #init-param-parameter-initialization-2}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input #init-param-parameter-initialization-2}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])
X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## Built-in Initialization

Let's begin by calling on built-in initializers.
The code below initializes all weight parameters
as Gaussian random variables
with standard deviation 0.01, while bias parameters are cleared to zero.

```{.python .input #init-param-built-in-initialization-1}
%%tab mxnet
# Here force_reinit ensures that parameters are freshly initialized even if
# they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input #init-param-built-in-initialization-1}
%%tab pytorch
def init_normal(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input #init-param-built-in-initialization-1}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input #init-param-built-in-initialization-1}
%%tab jax
weight_init = nn.initializers.normal(0.01)
bias_init = nn.initializers.zeros

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(d2l.get_key(), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

We can also initialize all the parameters
to a given constant value (say, 1).

```{.python .input #init-param-built-in-initialization-2}
%%tab mxnet
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

```{.python .input #init-param-built-in-initialization-2}
%%tab pytorch
def init_constant(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```

```{.python .input #init-param-built-in-initialization-2}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]
```

```{.python .input #init-param-built-in-initialization-2}
%%tab jax
weight_init = nn.initializers.constant(1)

net = nn.Sequential([nn.Dense(8, kernel_init=weight_init, bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=weight_init, bias_init=bias_init)])

params = net.init(d2l.get_key(), X)
layer_0 = params['params']['layers_0']
layer_0['kernel'][:, 0], layer_0['bias'][0]
```

We can also apply different initializers for certain blocks.
For example, below we initialize the first layer
with the Xavier initializer
and initialize the second layer
to a constant value of 42.

```{.python .input #init-param-built-in-initialization-3}
%%tab mxnet
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[0].weight.data()[0])
print(net[1].weight.data())
```

```{.python .input #init-param-built-in-initialization-3}
%%tab pytorch
def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 42)

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
```

```{.python .input #init-param-built-in-initialization-3}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.Constant(42)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])
```

```{.python .input #init-param-built-in-initialization-3}
%%tab jax
net = nn.Sequential([nn.Dense(8, kernel_init=nn.initializers.xavier_uniform(),
                              bias_init=bias_init),
                     nn.relu,
                     nn.Dense(1, kernel_init=nn.initializers.constant(42),
                              bias_init=bias_init)])

params = net.init(d2l.get_key(), X)
params['params']['layers_0']['kernel'][:, 0], params['params']['layers_2']['kernel']
```

### Custom Initialization

Sometimes, the initialization methods we need
are not provided by the deep learning framework.
In the example below, we define an initializer
for any weight parameter $w$ using the following strange distribution:

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \textrm{ with probability } \frac{1}{4} \\
            0    & \textrm{ with probability } \frac{1}{2} \\
        U(-10, -5) & \textrm{ with probability } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

:begin_tab:`mxnet`
Here we define a subclass of the `Initializer` class.
Usually, we only need to implement the `_init_weight` function
which takes a tensor argument (`data`)
and assigns to it the desired initialized values.
:end_tab:

:begin_tab:`pytorch`
Again, we implement a `my_init` function to apply to `net`.
:end_tab:

:begin_tab:`tensorflow`
Here we define a subclass of `Initializer` and implement the `__call__`
function that return a desired tensor given the shape and data type.
:end_tab:

:begin_tab:`jax`
Jax initialization functions take as arguments the `PRNGKey`, `shape` and
`dtype`. Here we implement the function `my_init` that returns a desired
tensor given the shape and data type.
:end_tab:

```{.python .input #init-param-custom-initialization-1}
%%tab mxnet
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[:2]
```

```{.python .input #init-param-custom-initialization-1}
%%tab pytorch
def my_init(module):
    if isinstance(module, nn.Linear):
        print("Init", *[(name, param.shape)
                        for name, param in module.named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        with torch.no_grad():
            module.weight *= module.weight.abs() >= 5

net.apply(my_init)
net[0].weight[:2]
```

```{.python .input #init-param-custom-initialization-1}
%%tab tensorflow
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])
```

```{.python .input #init-param-custom-initialization-1}
%%tab jax
def my_init(key, shape, dtype=jnp.float_):
    data = jax.random.uniform(key, shape, minval=-10, maxval=10)
    return data * (jnp.abs(data) >= 5)

net = nn.Sequential([nn.Dense(8, kernel_init=my_init), nn.relu, nn.Dense(1)])
params = net.init(d2l.get_key(), X)
print(params['params']['layers_0']['kernel'][:, :2])
```

:begin_tab:`mxnet, pytorch, tensorflow`
Note that we always have the option
of setting parameters directly.
:end_tab:

:begin_tab:`jax`
When initializing parameters in JAX and Flax, the dictionary of parameters
returned has a `flax.core.frozen_dict.FrozenDict` type. It is not advisable in
the Jax ecosystem to directly alter the values of an array, hence the datatypes
are generally immutable. One might use `params.unfreeze()` to make changes.
:end_tab:

```{.python .input #init-param-custom-initialization-2}
%%tab mxnet
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

```{.python .input #init-param-custom-initialization-2}
%%tab pytorch
with torch.no_grad():
    net[0].weight[:] += 1
    net[0].weight[0, 0] = 42
net[0].weight[0]
```

```{.python .input #init-param-custom-initialization-2}
%%tab tensorflow
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]
```

## Summary

We can initialize parameters using built-in and custom initializers.

## Exercises

Look up the online documentation for more built-in initializers.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/8089)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/8090)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/8091)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17991)
:end_tab:

<!-- slides -->

::: {.slide title="Initialization Matters"}
Initialization isn't cosmetic — it determines whether a deep
network trains *at all*.

- Zero weights → every neuron in a layer computes the same
  thing, gets the same gradient ("symmetry breaking" fails).
- Too large → activations blow up.
- Too small → activations and gradients vanish through depth.

The fix: choose the scale so signal variance stays roughly
constant from layer to layer.
:::

::: {.slide title="Why scale matters"}
Consider $y = Wx$ with i.i.d. zero-mean $x_i$, variance
$\sigma_x^2$, and weights with variance $\sigma_w^2$:

$$\text{Var}(y_i) = n_{\text{in}} \cdot \sigma_w^2 \cdot \sigma_x^2.$$

Stack $L$ layers and the signal variance scales by
$(n_{\text{in}} \sigma_w^2)^L$ — keep it stable by picking
$\sigma_w^2 \approx 1/n_{\text{in}}$.
:::

::: {.slide title="Xavier and Kaiming"}
- **Xavier (Glorot 2010)** —
  $\sigma_w^2 = \dfrac{2}{n_{\text{in}} + n_{\text{out}}}$.
  Balances forward variance with backward gradient variance.
  Designed for $\tanh$ / sigmoid.
- **Kaiming/He (2015)** —
  $\sigma_w^2 = \dfrac{2}{n_{\text{in}}}$.
  Compensates for ReLU killing half the signal. Default for
  modern CNNs / Transformers.

Bias usually starts at 0.
:::

::: {.slide title="The framework defaults"}
Each framework picks one of these by default:

| Framework | Default for `Linear`/`Dense` |
|---|---|
| PyTorch | Kaiming-uniform on weight; uniform $\pm 1/\sqrt{\text{fan-in}}$ on bias |
| Flax (JAX) | LeCun-normal (~Kaiming for $\tanh$) |
| Keras (TF) | Glorot-uniform |
| MXNet | Uniform $\pm 0.07$ (legacy; you should override) |

Bottom line: every modern framework picks something
fan-in/fan-out aware. You can usually leave it alone.
Override when you need a non-standard scheme.
:::

::: {.slide title="Setup"}
@init-param-parameter-initialization-1

. . .

@init-param-parameter-initialization-2
:::

::: {.slide title="The universal pattern: net.apply(fn)"}
Override the default by walking the module tree and applying
an initializer to each leaf module. PyTorch: `net.apply(fn)`
calls `fn(module)` recursively for every submodule:

@init-param-built-in-initialization-1

. . .

Constants are an anti-pattern (kills symmetry-breaking) but
illustrate the API:

@init-param-built-in-initialization-2
:::

::: {.slide title="Different scheme per layer"}
Dispatch on layer type or layer index — Xavier for the first
linear, constant 42 for the second:

@init-param-built-in-initialization-3

The pattern: take a `(name, module)` tuple, decide what to do.
Same machinery used for freezing layers (`requires_grad =
False`), discriminative learning rates, and BERT-style "warm
up the head, not the backbone".
:::

::: {.slide title="Custom initialization"}
For non-standard schemes, write the init function yourself.
Here a heavy-tailed sample with thresholding:

$$w \sim U(-10, 10),\quad w \leftarrow w \cdot \mathbb{1}_{|w| \ge 5}.$$

@init-param-custom-initialization-1

. . .

For one-off surgery — loading specific weights, replacing a
single layer's tensor — assign to `.data` directly:

@init-param-custom-initialization-2
:::

::: {.slide title="When to override defaults"}
Most of the time, don't. Cases where you should:

- **Loading pretrained weights** — `load_state_dict` is the
  ultimate "initialization" override.
- **Custom layers** — you wrote a new layer with a different
  variance budget, e.g. small-residual init that puts
  ResBlocks at near-identity.
- **Reproducibility / ablations** — comparing init schemes
  systematically.
- **Architecture-specific tricks** — e.g. zero-init the last
  BN $\gamma$ in each ResNet block (FixUp / Skip-init).
:::

::: {.slide title="Recap"}
- Init scale matters: set it so signal variance stays roughly
  constant across depth.
- **Xavier**: $\frac{2}{n_{in}+n_{out}}$ for $\tanh$/sigmoid.
- **Kaiming/He**: $\frac{2}{n_{in}}$ for ReLU.
- Framework defaults are sane; override via
  `net.apply(init_fn)` and write per-type rules in the
  function.
- Direct `with torch.no_grad(): layer.weight[...] = ...` for
  one-off tensor surgery.
:::
