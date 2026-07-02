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
with one hidden layer and 256 hidden units
(:numref:`fig_mdl-mlp-arch`).
Both the number of layers and their width are adjustable
(they are considered hyperparameters).
Typically, we choose each layer width to be a multiple of a large power of 2.
This improves computational efficiency because the matrix-multiplication kernels
on modern hardware (CPUs and GPUs) are tuned for operand dimensions that align
to SIMD vector widths and tensor-core tile sizes.

![The two-layer MLP of this section: a batched input is flattened to 784 features, mapped by an affine layer to a 256-dimensional hidden representation, passed through a ReLU, then mapped by a second affine layer to 10 logits.](../img/mdl-mlp-arch.svg)
:label:`fig_mdl-mlp-arch`

Again, we will represent our parameters with several tensors.
Note that *for every layer*, we must keep track of
one weight matrix and one bias vector.
As always, we allocate memory
for the gradients of the loss with respect to these parameters.
We use small Gaussian noise ($\sigma = 0.01$) as a simple starting point;
principled strategies for choosing this scale are the subject of
:numref:`sec_numerical_stability`.

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
    return torch.maximum(X, torch.zeros_like(X))
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
trainer = d2l.Trainer(max_epochs=30)
trainer.fit(model, data)
```

You should see the validation accuracy settle around $0.87$ — a real, if
modest, improvement over the $\approx 0.83$ that softmax regression reached
on the same data, bought by the hidden layer and its ReLU.

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
Note that while the two versions compute the same *architecture*, they do not
start from the same *parameters*: the scratch model draws its weights from
$\mathcal{N}(0, 0.01^2)$, whereas the concise version uses the framework's
fan-in-scaled default initializer (:numref:`sec_numerical_stability`), so
their training trajectories, and final accuracies, can differ slightly.

```{.python .input #mlp-implementation-training-2}
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Summary

We have now built and trained a working multilayer perceptron, a network with a hidden layer and a nonlinearity, in both a from-scratch and a concise form. The from-scratch version makes the new ingredients concrete: two weight matrices, two bias vectors, a hand-rolled ReLU, and a two-step forward computation. The concise version shows that `nn.Sequential` absorbs all of that bookkeeping into a four-element stack. The training loop, the loss function, and the data loader are unchanged from softmax regression, a first sign that the modular design pays off.

The from-scratch version also exposes why we reach for the high-level API: naming and tracking parameters by hand quickly becomes awkward. Imagine inserting another layer between layers 42 and 43; we would be stuck renaming or improvising a "layer 42b". Hand-rolled forward passes are also harder for the framework to optimize. `nn.Sequential` removes both problems at once.

Three questions remain open, and each is the subject of one of the next sections:

* **How do gradients flow through this stack, and what can go wrong as it gets deeper?** (:numref:`sec_backprop`, :numref:`sec_numerical_stability`)
* **Why does such a flexible model generalize to unseen data at all?** (:numref:`sec_generalization_deep`)
* **How can we regularize it to generalize better?** (:numref:`sec_dropout`)

Answering them turns this small working model into a reliable building block.


## Exercises

1. Change the number of hidden units `num_hiddens` and plot how its number affects the accuracy of the model. What is the best value of this hyperparameter?
1. Try adding a hidden layer to see how it affects the results. In particular, add one to the *from-scratch* model while keeping its $\sigma = 0.01$ Gaussian initialization: you may find that the deeper network trains *worse*. Why that happens, and what to do about it, is the subject of :numref:`sec_numerical_stability`.
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
1. Try out different activation functions. Which one works best on Fashion-MNIST? Compare at least ReLU, tanh, sigmoid, and GELU (`torch.nn.functional.gelu` in PyTorch, `jax.nn.gelu` in JAX). (*Hint:* for sigmoid and tanh you may need to retune the learning rate.) GELU is the default in modern transformer architectures; can you see why from how it behaves on this task?
1. Compare the effect of three initialization scales on training: (a) small Gaussian noise with $\sigma = 0.001$; (b) the value used in this section, $\sigma = 0.01$; (c) large Gaussian noise with $\sigma = 0.1$. Plot the training and validation curves for each. Why does $\sigma$ matter? (*Hint:* consider what happens to the activations on the very first forward pass.) The principled answer is developed in :numref:`sec_numerical_stability`.

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

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §5.2]{.kicker}

Implementing a **multilayer perceptron**<br>One hidden layer, two ways to build it --- and a scoreboard: **$\approx 0.87$ where softmax regression managed $\approx 0.83$**.
:::
:::

::: {.slide title="The whole model on one slide"}
[What we are building]{.kicker}

::: {.cols .vc}
::: {.col}
A batched image is **flattened** to 784 features, mapped by
an **affine layer + ReLU** to a 256-dim hidden vector, then
by a **second affine layer** to 10 logits.

- One hidden layer, one nonlinearity.
- Same loss, loaders, and `Trainer` as softmax regression.

::: {.d2l-note .rule}
That ReLU between the two affine maps is the *entire* difference
from a linear classifier — and softmax regression scored
$\approx 0.83$ on this data. The hidden layer must beat that
number, or it is 200k wasted parameters. Keep score.
:::
:::

::: {.col .fig .big}
![](../img/mdl-mlp-arch.svg)
:::
:::
:::

::: {.slide title="Why these sizes?"}
[Design choices]{.kicker}

::: {.cols .vc}
::: {.col}
Fashion-MNIST: $784$ inputs, $10$ classes. We pick **256**
hidden units, giving $\approx 200\text{k}$ parameters.

- **Width 256:** big enough to fit the data, small enough
  to train in seconds.
- **A power of 2** because matmul kernels are tuned for
  SIMD/tensor-core tile sizes (nothing breaks at 250).
- **One hidden layer** suffices here; spatial structure
  waits for convolutions.

::: {.d2l-note .rule}
Depth, width, and learning rate are **hyperparameters**:
chosen by hand, not learned.
:::
:::

::: {.col .fig}
![](../img/mdl-mlp-arch.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[From Scratch]{.dtitle}

[parameters, ReLU, and forward by hand]{.dsub}
:::
:::

::: {.slide title="Parameters: two weights, two biases"}
[From Scratch]{.kicker}

::: {.cols .vc}
::: {.col}
@mlp-implementation-initializing-model-parameters
:::

::: {.col .narrow}
Weights start as small Gaussian noise ($\sigma=0.01$) to break symmetry, biases at zero:

$$\mathbf{W}^{(1)}\!\in\mathbb{R}^{784\times256},\;
  \mathbf{b}^{(1)}\!\in\mathbb{R}^{256}$$
$$\mathbf{W}^{(2)}\!\in\mathbb{R}^{256\times10},\;
  \mathbf{b}^{(2)}\!\in\mathbb{R}^{10}$$

::: {.d2l-note}
$784\cdot256 + 256 + 256\cdot10 + 10 = 203{,}530$ learnable numbers.
:::
:::
:::
:::

::: {.slide title="ReLU, by hand"}
[From Scratch]{.kicker}

To see there is no magic, we write the activation ourselves
rather than calling the built-in. It is just $\max(x, 0)$,
applied elementwise:

@mlp-implementation-model-1

::: {.d2l-note}
Zero out the negatives, pass the positives through. That one
kink is what lets a stack of affine maps bend.
:::
:::

::: {.slide title="The forward pass is two lines"}
[From Scratch]{.kicker}

Flatten, then an affine-ReLU, then a second affine, exactly
the data flow in the diagram:

$$\mathbf{H} = \mathrm{ReLU}(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}),
\qquad
  \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.$$

@mlp-implementation-model-2
:::

::: {.slide title="Promise kept: ≈0.87, four points over softmax"}
[From Scratch · payoff]{.kicker}

The loss, the loaders, and the `Trainer` are **unchanged**
from softmax regression. Only the model class is new:

@!mlp-implementation-training

::: {.d2l-note .rule}
Validation accuracy settles around $\approx 0.87$ over 30 epochs —
a real, if modest, gain over softmax regression's $\approx 0.83$ on
the same data, bought by one hidden layer and its ReLU.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Concise]{.dtitle}

[let the framework keep the books]{.dsub}
:::
:::

::: {.slide title="The same model, declared" except="jax"}
[Concise]{.kicker}

::: {.cols .vc}
::: {.col}
Stack the layers in the framework's container. Lazy linear
layers infer their input size; `ReLU` and `Flatten` come
built in:

@mlp-implementation-model-2-2
:::

::: {.col .narrow}
::: {.d2l-note .rule}
`Sequential` *is* the forward pass: apply each layer in
turn. No hand-written `forward`, no parameter bookkeeping.
:::

Same architecture as the diagram, four lines instead of two
classes.
:::
:::
:::

::: {.slide title="The same model, declared" only="jax"}
[Concise]{.kicker}

::: {.cols .vc}
::: {.col}
Flax builds the network inside one `@nn.compact` method:
declare each layer where it is applied and the parameters
are registered for you.

@mlp-implementation-model-2-2
:::

::: {.col .narrow}
::: {.d2l-note .rule}
No hand-written parameter dictionary: `nn.Dense` registers
its weights the first time it is called.
:::

Same architecture as the diagram, one compact method.
:::
:::
:::

::: {.slide title="Same loop, same result"}
[Concise]{.kicker}

Reusing the very same `trainer` and data, the concise model
converges just like the scratch one:

@mlp-implementation-training-2

::: {.d2l-note}
Same *architecture*, different init (framework default vs
$\mathcal{N}(0, 0.01^2)$), so trajectories differ slightly.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[What's next]{.dtitle}

[from a working MLP to a reliable one]{.dsub}
:::
:::

::: {.slide title="Four open questions"}
[Where this goes]{.kicker}

We have a working MLP. Making it *reliable* is the rest of
this chapter:

- **Backprop (§5.3):** how gradients flow through an arbitrary
  stack.
- **Initialization (§5.4):** choose $\sigma$ so signals neither
  vanish nor explode through depth.
- **Generalization (§5.5):** why a flexible model does well on
  unseen data at all.
- **Regularization (§5.6):** dropout, and friends.

::: {.d2l-note}
Each question gets its own section — and exercise 2 hands you the
cliffhanger: add a second hidden layer while keeping
$\sigma = 0.01$, and the deeper net trains *worse*. §5.4 explains.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- An **MLP** = a linear classifier plus a hidden layer and
  a **nonlinearity** between affine maps.
- **From scratch:** four parameter tensors, a hand-rolled
  ReLU, a two-line forward. Concrete, but tedious to ship.
- **Concise:** declare the layer stack; the framework owns
  the parameters and the forward pass.
:::

::: {.col}
- Both forms declare the **same architecture** (inits differ).
- The **training loop is unchanged** from softmax
  regression (modularity paying off).
- Hyperparameters (depth, width, lr) live **outside** the
  model; the same loop trains any of them.
- Scoreboard settled: **$\approx 0.87$ vs $\approx 0.83$** — the
  first measured payoff of depth.
:::
:::

::: {.d2l-note}
Next (§5.3): open the black box — what `backward()` actually
computes, by hand and then verified.
:::
:::
