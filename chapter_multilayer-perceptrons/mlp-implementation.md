```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Implementation of Multilayer Perceptrons
:label:`sec_mlp-implementation`

An MLP implementation extends a linear classifier by composing affine layers
with nonlinear activations. We first write this composition explicitly and
then use the corresponding framework layers.

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
from flax import nnx
from jax import numpy as jnp
```

## Implementation from Scratch

We begin with an implementation using tensors and automatic differentiation.

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
Powers-of-two widths often align well with accelerator kernels, although the
fastest dimensions depend on the hardware and numerical precision.

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
In the code below, `nnx.Param` marks each array as a trainable parameter.
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
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr,
                 sigma=0.01, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.W1 = nnx.Param(
            rngs.params.normal((num_inputs, num_hiddens)) * sigma)
        self.b1 = nnx.Param(jnp.zeros(num_hiddens))
        self.W2 = nnx.Param(
            rngs.params.normal((num_hiddens, num_outputs)) * sigma)
        self.b2 = nnx.Param(jnp.zeros(num_outputs))
```

### Model

To expose the complete forward computation, we implement ReLU directly rather
than invoking the built-in `relu` function.

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
The forward method then requires only the flattening and two affine
computations. Automatic differentiation supplies the backward pass.

```{.python .input #mlp-implementation-model-2}
@d2l.add_to_class(MLPScratch)
def forward(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))
    H = relu(d2l.matmul(X, self.W1) + self.b1)
    return d2l.matmul(H, self.W2) + self.b2
```

### Training

The training loop is the same as for softmax regression. We define the model,
data, and trainer, then invoke `fit` on the model and data.

```{.python .input #mlp-implementation-training}
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=30)
trainer.fit(model, data)
```

In the displayed run, validation accuracy settles near $0.87$, above the earlier
softmax-regression run. This comparison is illustrative: initialization,
shuffling, framework defaults, and optimization settings also affect the result.

## Concise Implementation

High-level framework layers provide a more concise implementation of the same
architecture.

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
    def __init__(self, num_outputs, num_hiddens, lr, num_inputs=784,
                 rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.hidden = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)
        self.output = nnx.Linear(num_hiddens, num_outputs, rngs=rngs)

    def forward(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        return self.output(nnx.relu(self.hidden(X)))
```

A `forward` method applies each model transformation to the output of the
preceding one. Here, `MLP` inherits the `Module` implementation
(:numref:`subsec_oo-design-models`), which invokes `self.net(X)`. The
`Sequential` container applies its registered transformations in order and
therefore defines the forward computation. :numref:`subsec_model-construction-sequential`
examines this container in detail.


### Training

The training loop is the same as for softmax regression.
This modularity enables us to separate
matters concerning the model architecture
from orthogonal considerations.
Note that while the two versions compute the same *architecture*, they do not
start from the same *parameters*: the scratch model draws its weights from
$\mathcal{N}(0, 0.01^2)$, whereas the concise version uses the library's
default initializer. Defaults differ across libraries and layer types, so the
two versions need not start at comparable scales. Their training trajectories
and final accuracies can therefore differ.

```{.python .input #mlp-implementation-training-2}
model = MLP(num_outputs=10, num_hiddens=256, lr=0.1)
trainer.fit(model, data)
```

## Summary

We have built and trained a multilayer perceptron in both from-scratch and concise
forms. The from-scratch version exposes two weight matrices, two bias vectors, a
ReLU, and the two affine computations. The concise version represents the same
architecture as a four-element `nn.Sequential` stack. The training loop, loss,
and data loader remain unchanged from softmax regression.

The from-scratch version also shows why high-level containers are useful:
manually naming and tracking parameters becomes cumbersome as layers are added
or reordered. `nn.Sequential` handles both registration and execution order.

Three questions remain open, and each is the subject of one of the next sections:

* **How do gradients flow through this stack, and what can go wrong as it gets deeper?** (:numref:`sec_backprop`, :numref:`sec_numerical_stability`)
* **Why does such a flexible model generalize to unseen data at all?** (:numref:`sec_generalization_deep`)
* **How can we regularize it to generalize better?** (:numref:`sec_dropout`)

The following sections address these requirements for reliable training.


## Exercises

1. [code] **Hidden-layer width.** Vary the number of hidden units `num_hiddens`
   and plot how the accuracy of the model depends on it. Which value of this
   hyperparameter gives the best result?
1. **Parameter count.** The one-hidden-layer model of this section maps 784
   inputs through $h$ hidden units to 10 outputs.
    1. Give a closed-form expression for the total number of parameters as a
       function of $h$.
    1. At which $h$ does the parameter count equal the number of training
       examples in Fashion-MNIST (60,000)? Compare this value with the 256
       units used in this section.
1. [code] **Adding a layer.** Add a second hidden layer to the *from-scratch*
   model while keeping its $\sigma = 0.01$ Gaussian initialization and observe
   how the results change. You may find that the deeper network trains
   *worse*. Why this happens, and what to do about it, is the subject of
   :numref:`sec_numerical_stability`.
1. **Single hidden unit.** Why is it a bad idea to insert a hidden layer with
   a single neuron? What could go wrong?
1. [code] **Learning rate.** How does changing the learning rate alter your
   results?
    1. With all other hyperparameters fixed, which learning rate gives the
       best result?
    1. How does the best learning rate relate to the number of training
       epochs?
1. [extended] **Joint hyperparameter search.** Hyperparameters interact: the
   best value of one depends on the values of the others.
    1. Train the model over a grid of learning rates $\{0.01, 0.03, 0.1,
       0.3\}$ and hidden widths $\{32, 64, 128, 256\}$, keeping the number of
       epochs fixed, and plot the final validation accuracy as a heatmap.
       Does the best learning rate shift systematically with width?
    1. Now optimize over all hyperparameters jointly: learning rate, number
       of epochs, number of hidden layers, and number of hidden units per
       layer. What is the best result you can achieve?
    1. Why is tuning several hyperparameters jointly much harder than tuning
       each one in isolation?
    1. Describe an efficient strategy for searching over multiple
       hyperparameters jointly.
1. [code] **Gradient check.** Approximate the derivative of the loss with
   respect to each entry $w$ of $\mathbf{W}^{(1)}$ by a central difference,

    $$\frac{\partial \ell}{\partial w} \approx \frac{\ell(w + \epsilon) - \ell(w - \epsilon)}{2\epsilon},$$

    evaluated on a single minibatch, and compare the result with the gradient
    computed by automatic differentiation. Report the maximum relative error.
    In double precision a correct implementation stays below $10^{-5}$.

    *Adapted from Stanford CS231n,
    [Assignment 1](https://cs231n.github.io/assignments2023/assignment1/),
    two-layer network.*
1. [code] **Benchmarking implementations.** Compare the speed of the
   framework and from-scratch implementations on a common benchmark: train
   both at hidden widths 256, 1024, and 4096 for three epochs each on
   Fashion-MNIST, recording the wall-clock time per epoch. Plot the ratio of
   from-scratch to framework time as a function of width. Does the ratio
   grow, shrink, or stay flat as the network gets larger? Suggest an
   explanation.
1. [code] **Memory alignment.** Measure the speed of tensor--matrix
   multiplications for well-aligned and misaligned matrices, for instance
   with dimensions 1024, 1025, 1026, 1028, and 1032.
    1. How do the results differ between GPUs and CPUs?
    1. Determine the memory bus width of your CPU and GPU.
1. [code] **Activation functions.** Try out different activation functions.
   Which one works best on Fashion-MNIST? Compare at least ReLU, tanh,
   sigmoid, and GELU; for sigmoid and tanh you may need to retune the
   learning rate. GELU is used in BERT and GPT-2-style Transformers, while
   many recent language models use gated SiLU (SwiGLU) blocks. Does this
   small image task provide enough evidence to choose among them for a
   Transformer?
1. [code] **Initialization scale.** The weights are initialized with Gaussian
   noise of standard deviation $\sigma$.
    1. Train the one-hidden-layer model with $\sigma = 0.001$, with
       $\sigma = 0.01$ (the value used in this section), and with
       $\sigma = 0.1$. Plot the training and validation curves for each. Why
       does $\sigma$ matter? Consider what happens to the activations on the
       first forward pass.
    1. Before running anything, predict which of a two-hidden-layer and a
       four-hidden-layer version of the from-scratch model (with the same
       total width) is more sensitive to $\sigma$, and why. Then verify:
       sweep $\sigma$ over $\{0.001, 0.003, 0.01, 0.03, 0.1\}$ for both
       depths and report the range of $\sigma$ for which each network reaches
       at least 70% training accuracy within five epochs. The principled
       answer is developed in :numref:`sec_numerical_stability`.

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2023/assignment2/),
    FullyConnectedNets Inline Question 1.*

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

**Implementing a Multilayer Perceptron**<br>From-scratch and framework-layer implementations; the displayed run reaches $\approx 0.87$ validation accuracy
:::
:::

::: {.slide title="The MLP architecture"}
[What we are building]{.kicker}

::: {.cols .vc}
::: {.col}
A batched image is **flattened** to 784 features, mapped by
an **affine layer + ReLU** to a 256-dim hidden vector, then
by a **second affine layer** to 10 logits.

- One hidden layer, one nonlinearity.
- Same loss, loaders, and `Trainer` as softmax regression.

::: {.d2l-note .rule}
That ReLU between the two affine maps, together with the hidden
layer, is the *entire* difference from a linear classifier like
softmax regression.
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

- **Width 256:** a representative capacity for this example.
- **A power of 2:** often favorable for accelerator kernels,
  although performance depends on hardware and precision.
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

We write the activation directly as $\max(x, 0)$,
applied elementwise:

@mlp-implementation-model-1

::: {.d2l-note}
Map negative inputs to zero and retain positive inputs. This
nonlinearity prevents the affine maps from collapsing into one.
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

::: {.slide title="Training the from-scratch model"}
[From Scratch]{.kicker}

The loss, the loaders, and the `Trainer` are **unchanged**
from softmax regression. Only the model class is new:

@!mlp-implementation-training

::: {.d2l-note .rule}
In this run, validation accuracy settles around $\approx 0.87$ over 30 epochs,
a modest gain over the softmax regression baseline on the same data,
for a model with one hidden layer and a ReLU.
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Concise]{.dtitle}

[framework layers register parameters and compose operations]{.dsub}
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
NNX stores each layer as an ordinary attribute and registers its
parameters automatically:

@mlp-implementation-model-2-2
:::

::: {.col .narrow}
::: {.d2l-note .rule}
No hand-written parameter dictionary: each `nnx.Linear` owns
its weights from construction onward.
:::

Same architecture as the diagram, one compact method.
:::
:::
:::

::: {.slide title="Reusing the training loop"}
[Concise]{.kicker}

The concise model reuses the same `trainer` and data. Its
trajectory can differ because its initializer is different:

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

- **Backprop (the forward/backward-propagation section):** how gradients
  flow through an arbitrary stack.
- **Initialization (the numerical-stability section):** choose $\sigma$ so
  signals neither vanish nor explode through depth.
- **Generalization (the generalization-in-deep-learning section):** why a
  flexible model does well on unseen data at all.
- **Regularization (the dropout section):** dropout, and friends.

::: {.d2l-note}
Each question gets its own section, and exercise 2 hands you the
next question: add a second hidden layer while keeping
$\sigma = 0.01$, and the deeper net trains *worse*. The numerical-stability
section explains.
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
- **Concise:** declare the layer stack; `Sequential` holds
  the parameters and defines the forward pass.
:::

::: {.col}
- Both forms declare the **same architecture** (inits differ).
- The **training loop is unchanged** from softmax
  regression (modularity paying off).
- Hyperparameters (depth, width, lr) live **outside** the
  model; the same loop trains any of them.
- The displayed run settles around **$\approx 0.87$**; a controlled comparison
  would also match seeds, initialization, and optimization settings.
:::
:::

::: {.d2l-note}
The next section derives the gradients computed by `backward()` and verifies
them numerically.
:::
:::
