```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Parameter Management

Once we have chosen an architecture
and set our hyperparameters,
we proceed to the training loop,
where our goal is to find parameter values
that minimize our loss function.
After training, we will need these parameters
in order to make future predictions.
Additionally, we will sometimes wish
to extract the parameters
perhaps to reuse them in some other context,
to save our model to disk so that
it may be executed in other software,
or for examination in the hope of
gaining scientific understanding.

Most of the time, we will be able
to ignore the nitty-gritty details
of how parameters are declared
and manipulated, relying on deep learning frameworks
to do the heavy lifting.
However, when we move away from
stacked architectures with standard layers,
we will sometimes need to get into the weeds
of declaring and manipulating parameters.
In this section, we cover the following:

* Accessing parameters for debugging, diagnostics, and visualizations.
* Sharing parameters across different model components.

```{.python .input #parameters-parameter-management-1}
%%tab mxnet
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #parameters-parameter-management-1}
%%tab pytorch
import torch
from torch import nn
```

```{.python .input #parameters-parameter-management-1}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input #parameters-parameter-management-1}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

We start by focusing on an MLP with one hidden layer.

```{.python .input #parameters-parameter-management-2}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Dense(8, activation='relu'))
net.add(nn.Dense(1))
net.initialize()  # Use the default initialization method

X = np.random.uniform(size=(2, 4))
net(X).shape
```

```{.python .input #parameters-parameter-management-2}
%%tab pytorch
net = nn.Sequential(nn.LazyLinear(8),
                    nn.ReLU(),
                    nn.LazyLinear(1))

X = torch.rand(size=(2, 4))
net(X).shape
```

```{.python .input #parameters-parameter-management-2}
%%tab tensorflow
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X).shape
```

```{.python .input #parameters-parameter-management-2}
%%tab jax
net = nn.Sequential([nn.Dense(8), nn.relu, nn.Dense(1)])

X = jax.random.uniform(d2l.get_key(), (2, 4))
params = net.init(d2l.get_key(), X)
net.apply(params, X).shape
```

## Parameter Access
:label:`subsec_param-access`

Let's start with how to access parameters
from the models that you already know.

:begin_tab:`mxnet, pytorch, tensorflow`
When a model is defined via the `Sequential` class,
we can first access any layer by indexing
into the model as though it were a list.
Each layer's parameters are conveniently
located in its attribute.
:end_tab:

:begin_tab:`jax`
Flax and JAX decouple the model and the parameters as you
might have observed in the models defined previously.
When a model is defined via the `Sequential` class,
we first need to initialize the network to generate
the parameters dictionary. We can access
any layer's parameters through the keys of this dictionary.
:end_tab:

We can inspect the parameters of the second fully connected layer as follows.

```{.python .input #parameters-parameter-access}
%%tab mxnet
net[1].params
```

```{.python .input #parameters-parameter-access}
%%tab pytorch
net[2].state_dict()
```

```{.python .input #parameters-parameter-access}
%%tab tensorflow
net.layers[2].weights
```

```{.python .input #parameters-parameter-access}
%%tab jax
params['params']['layers_2']
```

We can see that this fully connected layer
contains two parameters,
corresponding to that layer's
weights and biases, respectively.


### Targeted Parameters

Note that each parameter is represented
as an instance of the parameter class.
To do anything useful with the parameters,
we first need to access the underlying numerical values.
There are several ways to do this.
Some are simpler while others are more general.
The following code extracts the bias
from the second neural network layer, which returns a parameter class instance, and
further accesses that parameter's value.

```{.python .input #parameters-targeted-parameters-1}
%%tab mxnet
type(net[1].bias), net[1].bias.data()
```

```{.python .input #parameters-targeted-parameters-1}
%%tab pytorch
type(net[2].bias), net[2].bias.data
```

```{.python .input #parameters-targeted-parameters-1}
%%tab tensorflow
type(net.layers[2].weights[1]), tf.convert_to_tensor(net.layers[2].weights[1])
```

```{.python .input #parameters-targeted-parameters-1}
%%tab jax
bias = params['params']['layers_2']['bias']
type(bias), bias
```

:begin_tab:`mxnet,pytorch`
Parameters are complex objects,
containing values, gradients,
and additional information.
That is why we need to request the value explicitly.

In addition to the value, each parameter also allows us to access the gradient. Because we have not invoked backpropagation for this network yet, it is in its initial state.
:end_tab:

:begin_tab:`jax`
Unlike the other frameworks, JAX does not keep a track of the gradients over the
neural network parameters, instead the parameters and the network are decoupled.
It allows the user to express their computation as a
Python function, and use the `grad` transformation for the same purpose.
:end_tab:

```{.python .input #parameters-targeted-parameters-2}
%%tab mxnet
net[1].weight.grad()
```

```{.python .input #parameters-targeted-parameters-2}
%%tab pytorch
net[2].weight.grad == None
```

### All Parameters at Once

When we need to perform operations on all parameters,
accessing them one-by-one can grow tedious.
The situation can grow especially unwieldy
when we work with more complex modules (e.g., nested ones),
since we would need to recurse
through the entire tree to extract
each sub-module's parameters.
Below we demonstrate accessing the parameters of all layers.

```{.python .input #parameters-all-parameters-at-once}
%%tab mxnet
net.collect_params()
```

```{.python .input #parameters-all-parameters-at-once}
%%tab pytorch
[(name, param.shape) for name, param in net.named_parameters()]
```

```{.python .input #parameters-all-parameters-at-once}
%%tab tensorflow
net.get_weights()
```

```{.python .input #parameters-all-parameters-at-once}
%%tab jax
jax.tree_util.tree_map(lambda x: x.shape, params)
```

## Tied Parameters

Often, we want to share parameters across multiple layers.
Let's see how to do this elegantly.
In the following we allocate a fully connected layer
and then use its parameters specifically
to set those of another layer.
Here we need to run the forward propagation
`net(X)` before accessing the parameters.

```{.python .input #parameters-tied-parameters}
%%tab mxnet
net = nn.Sequential()
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = np.random.uniform(size=(2, 20))

net(X)
# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

```{.python .input #parameters-tied-parameters}
%%tab pytorch
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.LazyLinear(8)
net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.LazyLinear(1))

net(X)
# Check whether the parameters are the same object (tied, not just equal)
assert net[2].weight is net[4].weight
net[2].weight.data[0, 0] = 100
# Modifying one affects the other since they share the same tensor
assert net[2].weight.data[0, 0] == net[4].weight.data[0, 0]
```

```{.python .input #parameters-tied-parameters}
%%tab tensorflow
# Keras keeps both references to the shared layer in net.layers,
# but the shared layer's parameters are tied
shared = tf.keras.layers.Dense(8, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(8, activation=tf.nn.relu),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
# Check whether the parameters are the same object
print(net.layers[2].weights[0] is net.layers[3].weights[0])
```

```{.python .input #parameters-tied-parameters}
%%tab jax
# We need to give the shared layer a name so that we can refer to its
# parameters
shared = nn.Dense(8)
net = nn.Sequential([nn.Dense(8), nn.relu,
                     shared, nn.relu,
                     shared, nn.relu,
                     nn.Dense(1)])

params = net.init(d2l.get_key(), X)

# Check whether the parameters are different
print(len(params['params']) == 3)
```

This example shows that the parameters
of the second and third layer are tied.
They are not just equal, they are
represented by the same exact tensor.
Thus, if we change one of the parameters,
the other one changes, too.

:begin_tab:`mxnet, pytorch, tensorflow`
You might wonder,
when parameters are tied
what happens to the gradients?
Since the model parameters contain gradients,
the gradients of the second hidden layer
and the third hidden layer are added together
during backpropagation.
:end_tab:


## Summary

We have several ways of accessing and tying model parameters.


## Exercises

1. Use the `NestMLP` model defined in :numref:`sec_model_construction` and access the parameters of the various layers.
1. Construct an MLP containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
1. Why is sharing parameters a good idea?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/56)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/57)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/269)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17990)
:end_tab:

<!-- slides -->

::: {.slide}
A neural network is a tree of **parameters** — the weight
matrices and bias vectors that gradient descent updates.
Training is one thing you do with them; this deck covers
all the others.

- **Inspection** — debug a network, sanity-check
  initialization, look at learned features.
- **Iteration** — every optimizer needs to walk every
  parameter; weight decay needs them; checkpointing needs
  them.
- **Sharing** ("tying") — make two layers refer to the
  *same* tensor: encoder/decoder weights in autoencoders,
  input/output embeddings in language models.

The mental model behind the API: a module is a **tree**,
parameters live at the leaves, and the framework gives you
both leaf-by-name access and recursive traversal.
:::

::: {.slide title="The parameter tree"}
A nested module is just a tree. Each module is a node;
each parameter is a leaf:

```
net  (Sequential)
├─ 0: Linear      ├─ weight  (8, 4)
│                 └─ bias    (8,)
├─ 1: ReLU         (no params)
└─ 2: Linear      ├─ weight  (1, 8)
                  └─ bias    (1,)
```

Two access patterns:

- **By path**: `net[2].weight` — direct.
- **By traversal**: walk the tree, yield every leaf.

Frameworks give you both, plus serialization built on the
same traversal.
:::

::: {.slide title="A toy model"}
@parameters-parameter-management-1

. . .

@parameters-parameter-management-2
:::

::: {.slide title="Direct access"}
Index into a `Sequential` like a list; each layer exposes
its parameters as attributes:

@parameters-parameter-access

Two parameters per `Linear` layer — weight matrix and bias
vector. The output object is a `Parameter` (PyTorch) or
similar wrapper that carries the tensor + gradient + extra
metadata.
:::

::: {.slide title="Tensor inside the parameter"}
`.data` (PyTorch) unwraps the parameter to a plain tensor
for inspection:

@parameters-targeted-parameters-1

. . .

`.grad` is the gradient buffer — populated by `backward()`,
otherwise `None`. Useful for custom optimizers or
diagnosing dead neurons:

@parameters-targeted-parameters-2
:::

::: {.slide title="Recursive traversal"}
For everything-at-once, use `named_parameters()`. It walks
the whole tree and yields `(name, param)` pairs at the
leaves — names use dotted paths through the nesting:

@parameters-all-parameters-at-once

This is the iterator `optim.SGD(net.parameters(), …)`
consumes. It's also what gets pickled when you save a
checkpoint with `state_dict()`. Walk-tree-once, use
many ways.
:::

::: {.slide title="Parameter tying"}
Reuse the *same* module instance at multiple positions in
your architecture, and the framework treats them as one
parameter set — same memory, gradients accumulate across
uses.

Common cases:

- **Tied embeddings**: input embedding and output softmax
  projection in a language model share weights — saves
  $|V| \cdot d$ parameters.
- **Autoencoders**: decoder uses transposed encoder
  weights.
- **Recurrent layers**: same kernel applied at every time
  step (the original tying mechanism).

@parameters-tied-parameters

Modify `net[2].weight` and `net[4].weight` reflects the
same change — they *are* the same tensor, not just equal.
:::

::: {.slide title="Recap"}
- A module is a tree; parameters live at the leaves.
- Direct access: `net[i].weight`, `.bias`, `.grad`.
- Recursive traversal: `named_parameters()` /
  `state_dict()` walks the whole tree.
- Same iterator powers optimizers, weight decay,
  checkpointing.
- Tied parameters = reuse the same module instance —
  gradients accumulate; one buffer in memory.
:::
