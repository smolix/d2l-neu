```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Implementation of Multilayer Perceptrons
:label:`sec_mlp-implementation`

Multilayer perceptrons (MLPs) are not much more complex to implement than simple linear models. The key conceptual
difference is that we now concatenate multiple layers.

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab jax
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
```

## Implementation from Scratch

Let's begin again by implementing such a network from scratch.

### Initializing Model Parameters

Recall that Fashion-MNIST contains 10 classes,
and that each image consists of a $28 \times 28 = 784$
grid of grayscale pixel values.
As before we will disregard the spatial structure
among the pixels for now,
so we can think of this as a classification dataset
with 784 input features and 10 classes.
To begin, we will implement an MLP
with one hidden layer and 256 hidden units.
Both the number of layers and their width are adjustable
(they are considered hyperparameters).
Typically, we choose the layer widths to be divisible by larger powers of 2.
This is computationally efficient due to the way
memory is allocated and addressed in hardware.

Again, we will represent our parameters with several tensors.
Note that *for every layer*, we must keep track of
one weight matrix and one bias vector.
As always, we allocate memory
for the gradients of the loss with respect to these parameters.

:begin_tab:`mxnet`
In the code below, we first define and initialize the parameters
and then enable gradient tracking.
:end_tab:

:begin_tab:`pytorch`
In the code below we use `nn.Parameter`
to automatically register
a class attribute as a parameter to be tracked by `autograd` (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`tensorflow`
In the code below we use `tf.Variable`
to define the model parameter.
:end_tab:

:begin_tab:`jax`
In the code below we use `flax.linen.Module.param`
to define the model parameter.
:end_tab:

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = np.random.randn(num_inputs, num_hiddens) * sigma
        self.b1 = np.zeros(num_hiddens)
        self.W2 = np.random.randn(num_hiddens, num_outputs) * sigma
        self.b2 = np.zeros(num_outputs)
        for param in self.get_scratch_params():
            param.attach_grad()
```

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(torch.randn(num_inputs, num_hiddens) * sigma)
        self.b1 = nn.Parameter(torch.zeros(num_hiddens))
        self.W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs) * sigma)
        self.b2 = nn.Parameter(torch.zeros(num_outputs))
```

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = tf.Variable(
            tf.random.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = tf.Variable(tf.zeros(num_hiddens))
        self.W2 = tf.Variable(
            tf.random.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = tf.Variable(tf.zeros(num_outputs))
```

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab jax
class MLPScratch(d2l.Classifier):
    num_inputs: int
    num_outputs: int
    num_hiddens: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.W1 = self.param('W1', nn.initializers.normal(self.sigma),
                             (self.num_inputs, self.num_hiddens))
        self.b1 = self.param('b1', nn.initializers.zeros, self.num_hiddens)
        self.W2 = self.param('W2', nn.initializers.normal(self.sigma),
                             (self.num_hiddens, self.num_outputs))
        self.b2 = self.param('b2', nn.initializers.zeros, self.num_outputs)
```

### Model

To make sure we know how everything works,
we will implement the ReLU activation ourselves
rather than invoking the built-in `relu` function directly.

```{.python .input #mlp-implementation-model-1}
%%tab mxnet
def relu(X):
    return np.maximum(X, 0)
```

```{.python .input #mlp-implementation-model-1}
%%tab pytorch
def relu(X):
    return torch.clamp(X, min=0)
```

```{.python .input #mlp-implementation-model-1}
%%tab tensorflow
def relu(X):
    return tf.math.maximum(X, 0)
```

```{.python .input #mlp-implementation-model-1}
%%tab jax
def relu(X):
    return jnp.maximum(X, 0)
```

Since we are disregarding spatial structure,
we `reshape` each two-dimensional image into
a flat vector of length  `num_inputs`.
Finally, we implement our model
with just a few lines of code. Since we use the framework built-in autograd this is all that it takes.

```{.python .input #mlp-implementation-model-2}
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Training

Fortunately, the training loop for MLPs
is exactly the same as for softmax regression. We define the model, data, and trainer, then finally invoke the `fit` method on model and data.

```{.python .input #mlp-implementation-training}
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

## Concise Implementation

As you might expect, by relying on the high-level APIs, we can implement MLPs even more concisely.

### Model

Compared with our concise implementation
of softmax regression
(:numref:`sec_softmax_concise`),
the only difference is that we add
*two* fully connected layers where we previously added only *one*.
The first is the hidden layer,
the second is the output layer.

```{.python .input #mlp-implementation-model-2-2}
%%tab mxnet
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize()
```

```{.python .input #mlp-implementation-model-2-2}
%%tab pytorch
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))
```

```{.python .input #mlp-implementation-model-2-2}
%%tab tensorflow
class MLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu'),
            tf.keras.layers.Dense(num_outputs)])
```

```{.python .input #mlp-implementation-model-2-2}
%%tab jax
class MLP(d2l.Classifier):
    num_outputs: int
    num_hiddens: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_hiddens)(X)
        X = nn.relu(X)
        X = nn.Dense(self.num_outputs)(X)
        return X
```

Previously, we defined `forward` methods for models to transform input using the model parameters.
These operations are essentially a pipeline:
you take an input and
apply a transformation (e.g.,
matrix multiplication with weights followed by bias addition),
then repetitively use the output of the current transformation as
input to the next transformation.
However, you may have noticed that 
no `forward` method is defined here.
In fact, `MLP` inherits the `forward` method from the `Module` class (:numref:`subsec_oo-design-models`) to 
simply invoke `self.net(X)` (`X` is input),
which is now defined as a sequence of transformations
via the `Sequential` class.
The `Sequential` class abstracts the forward process
enabling us to focus on the transformations.
We will further discuss how the `Sequential` class works in :numref:`subsec_model-construction-sequential`.


### Training

The training loop is exactly the same
as when we implemented softmax regression.
This modularity enables us to separate
matters concerning the model architecture
from orthogonal considerations.

```{.python .input #mlp-implementation-training-2}
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Summary

Now that we have more practice in designing deep networks, the step from a single to multiple layers of deep networks does not pose such a significant challenge any longer. In particular, we can reuse the training algorithm and data loader. Note, though, that implementing MLPs from scratch is nonetheless messy: naming and keeping track of the model parameters makes it difficult to extend models. For instance, imagine wanting to insert another layer between layers 42 and 43. This might now be layer 42b, unless we are willing to perform sequential renaming. Moreover, if we implement the network from scratch, it is much more difficult for the framework to perform meaningful performance optimizations.

Nonetheless, you have now reached the state of the art of the late 1980s when fully connected deep networks were the method of choice for neural network modeling. Our next conceptual step will be to consider images. Before we do so, we need to review a number of statistical basics and details on how to compute models efficiently.


## Exercises

1. Change the number of hidden units `num_hiddens` and plot how its number affects the accuracy of the model. What is the best value of this hyperparameter?
1. Try adding a hidden layer to see how it affects the results.
1. Why is it a bad idea to insert a hidden layer with a single neuron? What could go wrong?
1. How does changing the learning rate alter your results? With all other parameters fixed, which learning rate gives you the best results? How does this relate to the number of epochs?
1. Let's optimize over all hyperparameters jointly, i.e., learning rate, number of epochs, number of hidden layers, and number of hidden units per layer.
    1. What is the best result you can get by optimizing over all of them?
    1. Why it is much more challenging to deal with multiple hyperparameters?
    1. Describe an efficient strategy for optimizing over multiple parameters jointly.
1. Compare the speed of the framework and the from-scratch implementation for a challenging problem. How does it change with the complexity of the network?
1. Measure the speed of tensor--matrix multiplications for well-aligned and misaligned matrices. For instance, test for matrices with dimension 1024, 1025, 1026, 1028, and 1032.
    1. How does this change between GPUs and CPUs?
    1. Determine the memory bus width of your CPU and GPU.
1. Try out different activation functions. Which one works best?
1. Is there a difference between weight initializations of the network? Does it matter?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/227)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17985)
:end_tab:

<!-- slides -->

::: {.slide title="Implementing an MLP two ways"}
The simplest **multilayer perceptron** — two affine layers
with a ReLU between them — trained end-to-end on
Fashion-MNIST (28×28 grayscale, 10 classes).

```
  X (batch, 784)
       │ Linear  784 → 256
       │ ReLU
       │ Linear  256 → 10
       ▼
  logits (batch, 10)
```

We'll build it twice — **from scratch** (manage the
weights by hand) and **concise** (`nn.Sequential`) — to
make concrete what the framework's abstraction buys you.
:::

::: {.slide title="Why one hidden layer of 256 is reasonable"}
For Fashion-MNIST (784 inputs → 10 outputs):

- **256 hidden units** = roughly 200k parameters. Big
  enough to memorize the training set in principle, small
  enough to actually train fast.
- **Powers of 2** for layer widths are a habit, not magic —
  matmul kernels are tuned for them; nothing breaks if you
  use 250 instead.
- **Single hidden layer** because Fashion-MNIST is easy. A
  proper deep net wouldn't help much without convolutions
  (next chapter).

These are *hyperparameters* — not learned. We set them by
hand, train, and see what works.
:::

::: {.slide title="Setup"}
@mlp-implementation-implementation-of-multilayer-perceptrons
:::

::: {.slide title="Parameters from scratch"}
Two weight matrices, two bias vectors. Init: small Gaussian
$\mathcal{N}(0, \sigma^2)$ for weights, zero for biases.

$$\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times 256},\quad
  \mathbf{b}^{(1)} \in \mathbb{R}^{256},$$
$$\mathbf{W}^{(2)} \in \mathbb{R}^{256 \times 10},\quad
  \mathbf{b}^{(2)} \in \mathbb{R}^{10}.$$

Total: $784 \cdot 256 + 256 + 256 \cdot 10 + 10 = 203\,530$
parameters.

@mlp-implementation-initializing-model-parameters
:::

::: {.slide title="ReLU and forward pass"}
First, our own ReLU — just `max(X, 0)` elementwise:

@mlp-implementation-model-1

. . .

Then the forward pass:

$$\mathbf{H} = \mathrm{ReLU}(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}),\quad
  \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.$$

Image pixels are flattened to a 784-vector first — we're
ignoring spatial structure. (CNNs in the next chapter
fix this.)

@mlp-implementation-model-2
:::

::: {.slide title="Training"}
Same `Trainer`, same Fashion-MNIST loaders, same
cross-entropy loss as softmax regression. *Only* the model
class changed:

@mlp-implementation-training

About 1–2 percentage points better than plain softmax
regression on the same data. A nonlinearity earns its keep.
:::

::: {.slide title="The concise version"}
Stack the same architecture using the framework's container.
Lazy linear layers infer input shapes; `ReLU` is built in:

@mlp-implementation-model-2-2

That's the whole architecture: 6 layers in a `Sequential`
(`Flatten` + 2 `Linear` + 1 `ReLU` + glue), zero hand-rolled
parameter management.

Both versions produce the *same* model. The framework just
removes the bookkeeping.
:::

::: {.slide title="Same training, same accuracy"}
@mlp-implementation-training-2

Identical convergence behavior. Built-in `Linear` and
`ReLU` give you exactly what the from-scratch version
computes — one of them is just easier to read and harder
to bug.
:::

::: {.slide title="What's left to learn"}
We have a working MLP — but the real questions are open:

- **Initialization** — pick $\sigma$ so activations don't
  explode or vanish through depth.
- **Generalization** — why does it do well on unseen data?
- **Regularization** — dropout, weight decay, etc.
- **Backprop** — how gradients flow through arbitrary stacks.

Each is the topic of one of the next decks.
:::

::: {.slide title="Recap"}
- An MLP is a softmax classifier with one or more
  **hidden layers + nonlinearity** between affine
  transforms.
- From scratch: 4 parameter tensors, hand-rolled ReLU,
  explicit matmuls. Useful to understand; tedious to ship.
- Concise: `Sequential(Flatten, Linear, ReLU, Linear)` —
  same model, less bookkeeping.
- Hyperparameters (depth, width, learning rate) live
  outside the model class; the same training loop works
  for any of them.
- Beats softmax regression on Fashion-MNIST by a small
  but real margin — first taste of "depth helps".
:::
