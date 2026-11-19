```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Dropout
:label:`sec_dropout`

An MLP trained for long enough will memorize whatever labels it is given—wrong
ones included—and lose accuracy on unseen data as it does so
(:numref:`sec_generalization_deep`). The two regularizers we have encountered so
far combat this tendency in different ways: weight decay
(:numref:`sec_weight_decay`) penalizes large weights, while early stopping
(:numref:`subsec_early_stopping`) halts training before mislabeled examples are
memorized. *Dropout* takes a third, more drastic approach: it injects noise into
the network while it learns. At every training step, we literally *drop out*
some fraction of the hidden units, setting each unit to zero independently with
some probability and computing the output from the survivors alone. At test
time, we run the full network.

The technique grew out of a hypothesis about how neural networks overfit.
Neither weight decay nor early stopping directly governs how hidden units divide
the work among themselves. A unit can become useful only in tandem with specific
partner units—for instance, by correcting their errors on particular training
examples—and such brittle arrangements need not transfer to new data.
:citet:`Hinton.Srivastava.Krizhevsky.ea.2012` called this phenomenon
*co-adaptation*. If hidden units vanish at random on every step, no unit can
rely on any particular partner being present.

In this section, we define the dropout operation, implement it from scratch and
via high-level APIs in a two-hidden-layer MLP, and train that network on
Fashion-MNIST. To see dropout's regularization effect clearly, we then return to
the noisy-label setting of the previous section, where a small training set
contains many corrupted labels. Finally, we examine three standard explanations
for why dropout works, noting what each one captures and where each falls short.

```{.python .input #dropout}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #dropout}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #dropout}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #dropout}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Noise in Hidden Layers

### Simplicity and Robustness

So far we have measured a model's simplicity in two ways: by the number of its
features (:numref:`subsec_polynomial-curve-fitting`) and by the norm of its
weights, which weight decay keeps small. A third notion of simplicity is
*smoothness*—the requirement that a function should not be overly sensitive to
small perturbations in its inputs. For instance, when classifying images, adding
a bit of random noise to the pixels should be mostly harmless; we can encourage
this behavior directly by training on noisy inputs while still asking the model
to predict the clean labels.

For a linear model, this reduces to something familiar. Adding an input noise
vector $\boldsymbol{\epsilon}$ with independent, zero-mean entries of variance
$s^2$ increases the expected squared error by $s^2\|\mathbf{w}\|^2$—the variance
of $\mathbf{w}^\top\boldsymbol{\epsilon}$—so training on noisy inputs is
equivalent to weight decay (Tikhonov regularization). For nonlinear models with
small noise, the induced penalty approximately penalizes how rapidly the output
varies with the input :cite:`Bishop.1995`. Dropout carries this intuition from
the input layer into the hidden layers, while changing the form of the noise:
rather than adding a small perturbation to every unit, it zeroes out a random
subset of units completely. Because noise of that magnitude lies outside
Bishop's small-perturbation analysis, we treat his result as conceptual
motivation rather than an exact equivalence.

### Masking and Rescaling

Zeroing out hidden units reduces the magnitude of the signal reaching the next
layer. If half the units are dropped, a weighted sum over them shrinks by half
on average, so dropout rescales the surviving units to keep the expected value
unchanged. Consider a single hidden unit with activation $h_j$, the $j^\textrm{th}$
entry of an example's hidden representation $\mathbf{h}$. At each training step,
dropout draws a binary *mask* $\xi_j$ equal to $0$ with probability $p$ and $1$
with probability $1 - p$, independently for every unit and every example, and
replaces $h_j$ with

$$
h_j' = \frac{\xi_j h_j}{1 - p} =
\begin{cases}
0 & \textrm{with probability } p,\\
\dfrac{h_j}{1 - p} & \textrm{with probability } 1 - p.
\end{cases}
$$
:eqlabel:`eq_dropout`

Here $p$ is the *dropout probability*. Dividing the retained units by $1 - p$
*debiases* the layer: for a fixed activation $h_j$, the expectation over the
mask is

$$\mathbb{E}[h_j'] = p \cdot 0 + (1 - p)\frac{h_j}{1 - p} = h_j,$$

so the next layer receives the same activation on average as it would without
dropout, and $1/(1 - p)$ is the unique factor with this property. The
fluctuations around that mean are substantial, however. Since
$\mathbb{E}[h_j'^2] = (1 - p)\, h_j^2/(1 - p)^2 = h_j^2/(1 - p)$, the variance is

$$\textrm{Var}[h_j'] = \mathbb{E}[h_j'^2] - h_j^2 = \frac{p}{1 - p}\, h_j^2.$$

The noise grows with both the dropout probability $p$ and the magnitude of the
activation. At $p = 0.5$, each surviving unit is doubled while each dropped unit
vanishes, giving $h_j'$ a standard deviation equal to $|h_j|$ itself. Unlike the
small additive input noise discussed above, dropout perturbs a unit by as much
as its own value.

The original paper :cite:`Hinton.Srivastava.Krizhevsky.ea.2012` left kept
activations unscaled during training and instead multiplied each unit's outgoing
weights by $1 - p$ at test time. Modern deep learning frameworks rescale during
training as in :eqref:`eq_dropout`, a convention known as *inverted dropout*.
Its advantage is practical: at inference time, the network runs unmodified, with
no dropout scaling required.

### Training and Inference

At test time we typically want deterministic predictions, so we disable dropout:
the full network runs with no masks and no rescaling. Does this full network
compute the average output of the random thinned networks sampled during
training? For a single unit, yes—each activation equals the expected value of
its dropped counterpart. The same holds through the next linear transformation:
a pre-activation $z = \sum_j w_j h_j'$ satisfies
$\mathbb{E}[z] = \sum_j w_j h_j$, matching the full network. Once a nonlinear
activation function is applied, however, expectation and nonlinearity no longer
commute.

A two-unit example makes the discrepancy concrete. Suppose $h_1 = h_2 = 1$ feed
into $z = h_1' - h_2'$ with dropout probability $p = 0.5$. Each $h_j'$ takes the
values $0$ or $2$ with equal probability, so $z$ takes the values $-2$, $0$,
$0$, and $2$ with probability $1/4$ each, yielding $\mathbb{E}[z] = 0$—the same
value computed by the full network. After applying a ReLU, however, the average
over masks is $\mathbb{E}[\textrm{ReLU}(z)] = \frac{1}{4} \cdot 2 = 0.5$,
whereas the full network produces $\textrm{ReLU}(0) = 0$. Thus, in a nonlinear
network, running the full network at test time does not compute the exact
average prediction of the networks sampled during training; it is only an
approximation, whose quality we revisit at the end of this section. To estimate
that average directly—or to gauge predictive uncertainty—one can keep dropout
active at test time and average over many stochastic forward passes (exercise
:exref:`ex_dropout-monte-carlo-dropout`).

## Implementation

### The Dropout Layer

To implement dropout on a tensor `X` with probability `dropout`, we draw a
uniform random number from $[0, 1)$ for each element and retain entries whose
samples are at least `dropout`, which occurs with probability $1 -$ `dropout`.
Multiplying `X` by this binary mask and dividing by $1 -$ `dropout` applies
:eqref:`eq_dropout` elementwise. We handle `dropout` $= 1$ separately by
returning an all-zeros tensor to avoid division by zero.

:begin_tab:`jax`
JAX has no global random state, so `dropout_layer` takes a random key as an
argument, and each call needs a fresh one (:numref:`sec_ndarray`).
:end_tab:

```{.python .input #dropout-implementation-from-scratch-1}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return mnp.zeros_like(X)
    mask = mnp.random.uniform(0, 1, X.shape, device=X.device) >= dropout
    return mask.astype(X.dtype) * X / (1.0 - dropout)
```

```{.python .input #dropout-implementation-from-scratch-1}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = torch.rand_like(X) >= dropout
    return mask.to(X.dtype) * X / (1.0 - dropout)
```

```{.python .input #dropout-implementation-from-scratch-1}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(tf.shape(X)) >= dropout
    return tf.cast(mask, X.dtype) * X / (1.0 - dropout)
```

```{.python .input #dropout-implementation-from-scratch-1}
%%tab jax
def dropout_layer(X, dropout, key):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) >= dropout
    return mask.astype(X.dtype) * X / (1.0 - dropout)
```

Let's test `dropout_layer` on a $2 \times 8$ tensor with dropout probabilities
$0$, $0.5$, and $1$. We also apply $p = 0.2$ to a vector of one million ones and
verify that its empirical mean and variance match the theoretical values $1$ and
$p/(1 - p) = 0.25$.

```{.python .input #dropout-implementation-from-scratch-2}
%%tab pytorch, tensorflow, mxnet
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (2, 8))
for p in (0, 0.5, 1):
    print(f'p = {p}:', dropout_layer(X, p))
Y = dropout_layer(d2l.ones(10**6), 0.2)
print(f'mean {float(d2l.reduce_mean(Y)):.3f}, '
      f'variance {float(d2l.reduce_mean((Y - 1) ** 2)):.3f}')
```

```{.python .input #dropout-implementation-from-scratch-2}
%%tab jax
X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
for p in (0, 0.5, 1):
    print(f'p = {p}:', dropout_layer(X, p, d2l.get_key()))
Y = dropout_layer(jnp.ones(10**6), 0.2, d2l.get_key())
print(f'mean {float(Y.mean()):.3f}, '
      f'variance {float(((Y - 1) ** 2).mean()):.3f}')
```

At $p = 0.5$, each entry is either zeroed out or doubled, with the exact count
of dropped entries fluctuating from run to run. On the vector of one million
ones with $p = 0.2$, the sample mean and variance match $1$ and $0.25$ to within
about $0.001$.

### The Model

We apply dropout to the output of each hidden layer immediately after its
activation function. :numref:`fig_dropout2` illustrates the effect on the MLP
from :numref:`fig_mlp` (four inputs, five hidden units, three outputs) for a
single mask draw in which $h_2$ and $h_5$ are dropped. The outputs no longer
depend on $h_2$ or $h_5$, and because multiplying by zero blocks gradients
during backpropagation, the weights entering and leaving those two units receive
no update from this example. On the next step, a fresh mask is drawn and a
different subset of units is silenced.

![Dropping the hidden units $h_2$ and $h_5$ of an MLP removes them together with their connections.](../img/dropout2.svg)
:label:`fig_dropout2`

Our model uses two hidden layers of 256 ReLU units each. Dropout probabilities
can be set independently per layer; a common heuristic is to drop fewer units
closer to the input layer, so we choose $p = 0.2$ after the first hidden layer
and $p = 0.5$ after the second. Following :numref:`sec_softmax_scratch` and
:numref:`subsec_he_init`, we register the parameters with the framework and
initialize the weights using He initialization—normal draws with variance
$2/n_\textrm{in}$, where $n_\textrm{in}$ is the layer's fan-in—and zero biases.
Crucially, dropout is applied only during training.

:begin_tab:`mxnet`
`autograd.is_training()` is true inside the `autograd.record()` block in
which the `Trainer` computes the loss for an update, and false when it
evaluates the model.
:end_tab:

:begin_tab:`pytorch`
`self.training` is set by `train()` and `eval()`, which the `Trainer` calls
before updating and before evaluating the model.
:end_tab:

:begin_tab:`tensorflow`
`self.training` records the `training` argument with which the `Trainer`
calls the model: `True` for an update, `False` for an evaluation.
:end_tab:

:begin_tab:`jax`
The `Trainer` evaluates a view of the model, made by `nnx.view`, which calls
`set_view` to set `deterministic` to `True`. The keys for the masks come from
the stream `rngs`, which advances at every call, so every step draws new
masks.
:end_tab:

```{.python .input #dropout-defining-the-model}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784):
        super().__init__()
        self.save_hyperparameters()
        he = lambda name, n_in, n_out: gluon.Parameter(  # Variance 2 / n_in
            name, shape=(n_in, n_out), init=init.Normal((2 / n_in) ** 0.5))
        zero = lambda name, n: gluon.Parameter(name, shape=(n,),
                                               init=init.Zero())
        self.W1 = he('W1', num_inputs, num_hiddens_1)
        self.W2 = he('W2', num_hiddens_1, num_hiddens_2)
        self.W3 = he('W3', num_hiddens_2, num_outputs)
        self.b1 = zero('b1', num_hiddens_1)
        self.b2 = zero('b2', num_hiddens_2)
        self.b3 = zero('b3', num_outputs)
        self.initialize()

    def forward(self, X):
        X = X.reshape(-1, self.num_inputs)
        H1 = npx.relu(mnp.dot(X, self.W1.data()) + self.b1.data())
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = npx.relu(mnp.dot(H1, self.W2.data()) + self.b2.data())
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return mnp.dot(H2, self.W3.data()) + self.b3.data()
```

```{.python .input #dropout-defining-the-model}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784):
        super().__init__()
        self.save_hyperparameters()
        he = lambda n_in, n_out: nn.Parameter(  # Variance 2 / n_in
            d2l.normal(0, (2 / n_in) ** 0.5, (n_in, n_out)))
        self.W1 = he(num_inputs, num_hiddens_1)
        self.W2 = he(num_hiddens_1, num_hiddens_2)
        self.W3 = he(num_hiddens_2, num_outputs)
        self.b1 = nn.Parameter(d2l.zeros(num_hiddens_1))
        self.b2 = nn.Parameter(d2l.zeros(num_hiddens_2))
        self.b3 = nn.Parameter(d2l.zeros(num_outputs))

    def forward(self, X):
        X = X.reshape(-1, self.num_inputs)
        H1 = torch.relu(X @ self.W1 + self.b1)
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = torch.relu(H1 @ self.W2 + self.b2)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return H2 @ self.W3 + self.b3
```

```{.python .input #dropout-defining-the-model}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784):
        super().__init__()
        self.save_hyperparameters()
        def he(name, n_in, n_out):  # Normal weights with variance 2 / n_in
            normal = tf.random_normal_initializer(stddev=(2 / n_in) ** 0.5)
            return self.add_weight(name=name, shape=(n_in, n_out),
                                   initializer=normal)
        zero = lambda name, n: self.add_weight(name=name, shape=(n,),
                                               initializer='zeros')
        self.W1 = he('W1', num_inputs, num_hiddens_1)
        self.W2 = he('W2', num_hiddens_1, num_hiddens_2)
        self.W3 = he('W3', num_hiddens_2, num_outputs)
        self.b1 = zero('b1', num_hiddens_1)
        self.b2 = zero('b2', num_hiddens_2)
        self.b3 = zero('b3', num_outputs)

    def forward(self, X):
        X = tf.reshape(X, (-1, self.num_inputs))
        H1 = tf.nn.relu(X @ self.W1 + self.b1)
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = tf.nn.relu(H1 @ self.W2 + self.b2)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return H2 @ self.W3 + self.b3
```

```{.python .input #dropout-defining-the-model}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        self.rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = lambda n_in, n_out: nnx.Param(  # Variance 2 / n_in
            self.rngs.params.normal((n_in, n_out)) * (2 / n_in) ** 0.5)
        self.W1 = he(num_inputs, num_hiddens_1)
        self.W2 = he(num_hiddens_1, num_hiddens_2)
        self.W3 = he(num_hiddens_2, num_outputs)
        self.b1 = nnx.Param(jnp.zeros(num_hiddens_1))
        self.b2 = nnx.Param(jnp.zeros(num_hiddens_2))
        self.b3 = nnx.Param(jnp.zeros(num_outputs))
        self.deterministic = False

    def set_view(self, *, deterministic):
        self.deterministic = deterministic

    def forward(self, X):
        X = X.reshape(-1, self.num_inputs)
        H1 = nnx.relu(X @ self.W1 + self.b1)
        if not self.deterministic:
            H1 = dropout_layer(H1, self.dropout_1, self.rngs.dropout())
        H2 = nnx.relu(H1 @ self.W2 + self.b2)
        if not self.deterministic:
            H2 = dropout_layer(H2, self.dropout_2, self.rngs.dropout())
        return H2 @ self.W3 + self.b3
```

### Training

We train the model just as we trained softmax regression in
:numref:`sec_softmax_concise`, using minibatch SGD with learning rate $0.1$ and
minibatches of 256 images, now for 20 epochs.

```{.python .input #dropout-training}
hparams = {'num_outputs': 10, 'num_hiddens_1': 256, 'num_hiddens_2': 256,
           'dropout_1': 0.2, 'dropout_2': 0.5, 'lr': 0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=20, num_gpus=1)
trainer.fit(model, data)
```

After 20 epochs, validation accuracy reaches about 87%, fluctuating by a point
or more from epoch to epoch just as in :numref:`sec_mlp-implementation`. Notice
one apparent quirk in the curves: during the first few epochs, the training loss
sits *above* the validation loss. This is not a bug. The training loss is
measured with dropout masks active, evaluating the noisier thinned subnetworks,
whereas the validation loss is evaluated on the full deterministic network. By
the end of training, the training loss drops slightly below the validation loss.

### Concise Implementation

High-level deep learning APIs provide dropout as a standard layer parameterized
by the dropout probability. During training it applies :eqref:`eq_dropout`, and
during evaluation it acts as the identity, eliminating the need to check
training mode manually. We insert a dropout layer after each hidden activation
and configure He initialization (normal weights with variance $2/n_\textrm{in}$)
to match our scratch implementation.

:begin_tab:`mxnet`
The framework files this initialization under `init.Xavier`, with
`rnd_type='gaussian'`, `factor_type='in'`, and `magnitude=2`.
:end_tab:

:begin_tab:`pytorch`
The `Trainer` initializes the lazy layers through the model's `weight_init`
method, as in :numref:`sec_softmax_concise`.
:end_tab:

:begin_tab:`tensorflow`
Each layer gets an initializer of its own: an instance created without a
seed returns the same random values on every call, so layers that shared one
would start from related weights.
:end_tab:

:begin_tab:`jax`
NNX dropout layers draw their keys from `rngs`, and `nnx.view` switches them
off for evaluation, as for the scratch model; it reaches them inside
`nnx.Sequential` as well.
:end_tab:

```{.python .input #dropout-concise-implementation-1}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation='relu'),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation='relu'),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize(init.Xavier(rnd_type='gaussian', factor_type='in',
                                        magnitude=2))
```

```{.python .input #dropout-concise-implementation-1}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)
```

```{.python .input #dropout-concise-implementation-1}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        # One initializer per layer: an instance repeats its random values
        he = lambda: tf.keras.initializers.VarianceScaling(
            2.0, 'fan_in', 'untruncated_normal')
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation='relu',
                                  kernel_initializer=he()),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation='relu',
                                  kernel_initializer=he()),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs, kernel_initializer=he())])
```

```{.python .input #dropout-concise-implementation-1}
%%tab jax
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        self.net = nnx.Sequential(
            nnx.Linear(num_inputs, num_hiddens_1, kernel_init=he, rngs=rngs),
            nnx.relu, nnx.Dropout(dropout_1, rngs=rngs),
            nnx.Linear(num_hiddens_1, num_hiddens_2, kernel_init=he,
                       rngs=rngs),
            nnx.relu, nnx.Dropout(dropout_2, rngs=rngs),
            nnx.Linear(num_hiddens_2, num_outputs, kernel_init=he, rngs=rngs))

    def forward(self, X):
        return self.net(X.reshape((X.shape[0], -1)))  # Flatten each image
```

Rather than re-running the same Fashion-MNIST fit here, we put `DropoutMLP` to
work in the controlled experiment that follows.

## Effects of Dropout

### Too Little to Regularize

Does dropout actually improve accuracy for our MLP? On the full 60,000-image
Fashion-MNIST training set, there is very little overfitting to remove in the
first place: after 20 epochs, the training loss ends only slightly below the
validation loss. To see what dropout truly does, we need a setting where the
network strongly overfits. We already encountered one in
:numref:`subsec_early_stopping`: an MLP trained for 200 epochs on 1,000 images
with corrupted labels memorized the wrong labels and suffered a clear drop in
validation accuracy.

### Noisy Labels

Let's repeat that experiment with our two-hidden-layer MLP and push the noise
even harder: we redraw 40% of the training labels at random, leaving roughly 36%
of them incorrect—more than one in three training examples now teaches the
network the wrong class. We reuse the utilities from
:numref:`subsec_early_stopping`: `d2l.corrupt` randomizes the labels,
`d2l.ArrayData` serves NumPy arrays in minibatches, and `d2l.Tracker` logs
accuracy on named evaluation sets after every epoch. We evaluate on the first
2,000 validation images.

```{.python .input #dropout-noisy-labels-1}
images = d2l.FashionMNIST(batch_size=2000)
X, y = (d2l.numpy(a) for a in next(iter(images.train_dataloader())))
X_val, y_val = (d2l.numpy(a) for a in next(iter(images.val_dataloader())))
X, y, X_val = X[:1000].reshape(1000, -1), y[:1000], X_val.reshape(2000, -1)
labels = d2l.corrupt(y, 0.4, np.random.default_rng())
wrong = labels != y  # The training labels that are now wrong
```

We train `DropoutMLP` twice for 200 epochs with minibatches of 64 images—once
with dropout probability $0.5$ in both hidden layers and once with dropout
disabled ($p = 0$). After each epoch, we record both the fraction of wrong
training labels memorized and the accuracy on the clean validation images.

```{.python .input #dropout-noisy-labels-2}
acc = {}
for p in (0, 0.5):
    trainer = d2l.Tracker(max_epochs=200, datasets={
        'wrong labels': d2l.ArrayData(X[wrong], labels[wrong], 1000),
        'validation': d2l.ArrayData(X_val, y_val, 1000)})
    trainer.fit(DropoutMLP(**{**hparams, 'dropout_1': p, 'dropout_2': p}),
                d2l.ArrayData(X, labels))
    acc[p] = trainer.accuracy
    print(f'dropout {p}: wrong labels reproduced '
          f'{acc[p]["wrong labels"][-1]:.2f}, validation accuracy over the '
          f'last 10 epochs {np.mean(acc[p]["validation"][-10:]):.3f}')
d2l.plot(list(range(1, 201)), [acc[0]['wrong labels'], acc[0.5]['wrong labels'],
                               acc[0]['validation'], acc[0.5]['validation']],
         'epoch', 'accuracy', fmts=('C0--', 'C1--', 'C0-', 'C1-'),
         legend=['wrong labels, p = 0', 'wrong labels, p = 0.5',
                 'validation, p = 0', 'validation, p = 0.5'],
         legend_loc='outside', figsize=(4.5, 3))
```

Without dropout, the network memorizes every corrupted label by around epoch
100, and its validation accuracy drops by roughly ten percentage points from its
early peak. With dropout enabled, memorization proceeds much more slowly: even
after 200 epochs, the network fits only three-quarters to four-fifths of the
wrong labels, and its validation accuracy over the final ten epochs stays three
to five percentage points higher than without dropout. In a noisy, data-scarce
regime, dropout does exactly what a regularizer should: it biases optimization
away from fitting label noise.

Notice also what dropout does *not* do. It slows memorization rather than
preventing it altogether. The wrong-label curve is still climbing at epoch 200,
and the dropout-trained network still loses validation accuracy over time—just
later and less severely. Neither run finishes as accurate as it was during its
first fifty epochs, underscoring that dropout complements early stopping rather
than replacing it.

## Interpretations

Why does dropout help—both here and across the image, speech, and text
benchmarks where :citet:`Srivastava.Hinton.Krizhevsky.ea.2014` demonstrated
lower test error? Our experiment shows *what* dropout does, not *why*. Three
explanations are common in the literature; each illuminates part of the
mechanism, and none tells the whole story on its own.

### Noise Injection

The first view treats dropout as structured noise injection: forcing a network
to function despite noisy hidden activations pushes it toward smoother, more
resilient mappings. For a linear model we can work out the exact consequence.
Suppose we apply dropout to the inputs of $\mathbf{w}^\top\mathbf{x} + b$ under
squared error loss. The masked inputs $x_j'$ are independent with mean $x_j$ and
variance $\frac{p}{1 - p}x_j^2$, so the prediction has mean
$\mathbf{w}^\top\mathbf{x} + b$ and variance
$\frac{p}{1-p}\sum_j w_j^2 x_j^2$. By the bias–variance decomposition, the
expected squared loss equals the squared error of the mean plus this variance:

$$\mathbb{E}\left[\left(\mathbf{w}^\top\mathbf{x}' + b - y\right)^2\right] = \left(\mathbf{w}^\top\mathbf{x} + b - y\right)^2 + \frac{p}{1 - p}\sum_j x_j^2 w_j^2.$$

In expectation, input dropout in linear regression is therefore an adaptive form
of weight decay :cite:`Srivastava.Hinton.Krizhevsky.ea.2014`, distinct from the
uniform $\ell_2$ penalty induced by additive Gaussian noise. Summed over the
training set, the penalty on $w_j$ scales with the sum of squares of feature
$j$: weights tied to large-magnitude features are penalized more heavily, while
the weight of an identically zero feature incurs no penalty at all. In deep
networks, however, the hidden activations themselves evolve throughout training,
so no analogous closed-form penalty exists.

### Co-adaptation

The second explanation is the intuition that originally inspired dropout. When
units cannot rely on specific partners being active, each unit is pressured to
learn features that remain useful on their own, which should generalize better
than fragile, tightly coupled combinations. While this picture is appealing, our
experiment neither proves nor disproves it: any regularizer that slows the
memorization of noisy labels would produce similar learning curves. It is best
viewed as helpful intuition rather than an established mechanism.

### An Implicit Ensemble

Our network contains 512 hidden units, corresponding to $2^{512}$ possible binary
masks. Each mask defines a *thinned* subnetwork, all of them sharing a single
pool of weights, and each training step updates a different subnetwork for every
example in the minibatch. From this perspective, running the full network at
test time approximates averaging the predictions of an exponentially large
*ensemble* of subnetworks—capturing the variance-reduction benefits of
ensembling while training only one model. Again, the analogy is loose. Unlike a
true ensemble, the thinned subnetworks are not trained independently, and as our
two-unit ReLU counterexample demonstrated, the full network does not compute
their exact arithmetic average. When dropout is applied solely to the layer
immediately preceding the output, the full network computes a normalized
weighted geometric mean of the subnetworks' softmax predictions (exercise
:exref:`ex_dropout-geometric-mean`); once earlier hidden layers are also
dropped, even that identity breaks down.

### Dropout in Practice

When, then, is dropout most effective? Precisely in the regime highlighted by
our experiment: far more parameters than training examples, combined with many
passes over the training data. That was the exact setting of the massive fully
connected layers in early-2010s image classifiers such as AlexNet
(:numref:`sec_alexnet`), for which dropout was originally designed. Modern
architectures tend to use smaller dropout rates (often around $0.1$), and models
trained on vast corpora where each example is seen only once—such as large
language models—frequently omit dropout altogether. One practical caveat is
worth remembering: because dropout multiplies the mean square of activations by
$1/(1 - p)$ during training but leaves them unscaled at inference time, layers
that normalize activation statistics (:numref:`sec_batch_norm`) can suffer from
this train–test variance shift :cite:`Li.Chen.Hu.ea.2019`.

In the next section, we encounter the high-parameter, small-data regime outside
computer vision: an MLP with roughly nineteen parameters for every house whose
sale price it tries to predict.

## Summary

Dropout regularizes neural networks by zeroing out each hidden unit
independently with probability $p$ during training and rescaling the surviving
units by $1/(1 - p)$ so that every unit's expected value stays unchanged; at
test time, dropout is turned off and the full network runs. Because expectation
does not pass through nonlinear activations, the full network only approximates
the average prediction of the thinned training networks. On the full
Fashion-MNIST dataset, our MLP overfits too little for dropout to make much
difference, but on 1,000 heavily mislabeled images, dropout markedly slows the
memorization of corrupted labels and preserves higher validation accuracy. For
linear models, input dropout acts in expectation as a feature-scaled weight
decay; for deep networks, noise injection, reduced co-adaptation, and implicit
ensembling each offer useful—though incomplete—intuitions for its behavior.

## Exercises

1. [code] **Dropout probabilities.** The model of this section drops units
   with probability 0.2 after the first hidden layer and 0.5 after the
   second.
    1. Train it with the two probabilities swapped, with both set to 0.5,
       and with both set to 0.8. For each run, report the best validation
       accuracy. Which layer tolerates the higher probability?
    1. The `Trainer` computes the training loss that it plots with the masks
       switched on. After training, evaluate the loss on the training images
       with the masks switched off, and compare the gap between training and
       validation loss for the networks trained with and without dropout.
1. **Dropout and He initialization.** The He initialization of
   :numref:`subsec_he_init` chooses the variance of the weights so that the
   mean square of the pre-activations stays the same from layer to layer.
    1. During training, by what factor does dropout with probability $p$
       change the mean square of the pre-activations of the next layer?
       Give the factor for each of the two dropout layers of this section's
       model.
    1. In a network with $K - 1$ hidden layers, each followed by dropout
       with probability $p$, how does the factor grow with $K$? Should the
       initialization correct for it, given that the network runs without
       dropout at test time? Relate your answer to the mismatch between
       training and test that dropout causes before a batch normalization
       layer.
1. [code] **Less noise, less dropout.** The noisy-label experiment of this
   section redrew 40% of the labels and dropped units with probability 0.5
   in both hidden layers.
    1. Repeat it with 20% of the labels redrawn, as in
       :numref:`subsec_early_stopping`, once with probability 0.5 in both
       layers and once with the 0.2 and 0.5 of the full-data run. In each
       case, does dropout still raise the validation accuracy of the last
       epochs, or does it only delay the memorization?
    1. In every run, with and without dropout, find the best validation
       accuracy and the epoch at which it is reached. Which helps more
       against wrong labels, dropout or stopping early? Do the two combine?
1. **Ensemble size.** The two hidden layers of this section's model have
   512 units together, so they admit $2^{512}$ joint dropout masks, and the
   ensemble interpretation views each mask as a thinned network.
    1. Training runs for 20 epochs over the 60,000 Fashion-MNIST training
       images, and a fresh mask is drawn for every example. How many joint
       masks are sampled during training, and what fraction of all joint
       masks is that? How likely is it that any mask is drawn twice? The
       masks are not equally likely, since the first layer drops units with
       probability 0.2: bound the probability that two draws coincide by
       the probability of the most likely joint mask.
    1. Using these numbers, explain in what sense dropout does, and does
       not, train an ensemble of all these networks.
1. [code] **Monte Carlo dropout.** Instead of switching dropout off at test
   time, keep it on in the trained `DropoutMLPScratch` and run $T = 20$
   forward passes per validation image, then average the softmax outputs
   :cite:`Gal.Ghahramani.2016`. Compare the accuracy and the calibration
   (the mean of the largest softmax probability against the accuracy) with
   those of the single pass of the full network. How does the accuracy
   change as $T$ grows from 1 to 20?
   :exlabel:`ex_dropout-monte-carlo-dropout`
1. [code] **Dropout and weight decay.** Weight decay and dropout both
   regularize the model of this section, by different mechanisms. Measure
   each run by its best validation accuracy and by the gap between training
   and validation loss, with both losses evaluated with the masks switched
   off.
    1. With both dropout probabilities set to zero, add weight decay as in
       :numref:`sec_weight_decay` at strengths
       $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$. Compare these runs
       with the runs with and without dropout of this section.
    1. Keep the dropout probabilities at 0.2 and 0.5, and add weight decay
       at the same strengths. Are the effects of the two regularizers
       additive, do they show diminishing returns, or do they cancel each
       other out?
1. **Dropout at test time.** ● Set `dropout_1` to zero, so that only the
   second hidden layer is masked, and let $\boldsymbol{\xi}$ range over its
   masks. Let $\mathbf{o}(\boldsymbol{\xi})$ and $\hat{\mathbf{y}}(\boldsymbol{\xi})$
   be the logits and the softmax output under mask $\boldsymbol{\xi}$, which
   occurs with probability $P(\boldsymbol{\xi})$. Show that the logits of the
   full network equal $\sum_{\boldsymbol{\xi}} P(\boldsymbol{\xi})\, \mathbf{o}(\boldsymbol{\xi})$.
   Show further that its softmax output equals the normalized weighted
   geometric mean

    $$\tilde{y}_j = \frac{\prod_{\boldsymbol{\xi}} \hat{y}_j(\boldsymbol{\xi})^{P(\boldsymbol{\xi})}}{\sum_k \prod_{\boldsymbol{\xi}} \hat{y}_k(\boldsymbol{\xi})^{P(\boldsymbol{\xi})}}.$$

    Explain why neither equality holds in general once `dropout_1` is
    nonzero.
   :exlabel:`ex_dropout-geometric-mean`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/261)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17987)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.6]{.kicker}

Dropout<br>**random masks · inverted dropout · noisy labels**
:::
:::

::: {.slide title="An MLP Trained Long Enough Memorizes Its Wrong Labels"}
[The problem]{.kicker}

- The MLP of this chapter fits 1,000 Fashion-MNIST images even with
  **random** labels, and it loses accuracy on new images as it memorizes
  wrong ones.
- **Weight decay** limits the size of the weights; **early stopping**
  limits the length of training.
- **Dropout** disturbs the network while it learns: at every step, each
  hidden unit is set to zero with probability $p$.

::: {.d2l-note}
The motivation (Hinton et al., 2012): a unit may be useful only together
with particular partners. Such *co-adapted* combinations fit the training
set and need not carry over to new images. With random units missing at
every step, no unit can count on a partner.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- compute the mean and the variance of a dropped unit;
- explain why the network at test time is **not** the average of the
  thinned networks, with a two-unit counterexample;
- implement a dropout layer and an MLP that uses it;
- measure what dropout changes when there are wrong labels to memorize,
  and what it does not;
- show that, for a linear model, dropout is a form of weight decay.

**Running example:** Fashion-MNIST. Softmax regression: 7,850 parameters,
about 84% after 30 epochs. Here: an MLP with two hidden layers of 256 units,
269,322 parameters, trained for 20 epochs at learning rate 0.1.
:::

::: {.slide title="Dropout Is the Third Regularizer of the Book"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Multilayer Perceptrons | hidden layers, universal approximation, activation functions |
| Implementation | an MLP trained on Fashion-MNIST |
| Backpropagation | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | vanishing gradients, Xavier and He initialization |
| Generalization in Deep Learning | memorization, double descent, implicit bias, early stopping |
| **Dropout** | **random masks, slower memorization of noisy labels** |
| Predicting House Prices on Kaggle | the workflow on real data |
:::

::: {.col .narrow}
- **Uses:** weight decay, He initialization, the `Trainer`, Fashion-MNIST.
- **Skips:** variants such as DropConnect and stochastic depth, and the
  Bayesian reading of dropout beyond one exercise.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Noise in Hidden Layers]{.dtitle}

[masks, rescaling, and the network at test time]{.dsub}
:::
:::

::: {.slide title="For a Linear Model, Input Noise Is Weight Decay"}
[Noise · motivation]{.kicker}

Perturb the input, $\mathbf{x} + \boldsymbol{\epsilon}$, with independent
entries of mean 0 and variance $s^2$:

$$\mathbb{E}\left[\left(\mathbf{w}^\top(\mathbf{x} + \boldsymbol{\epsilon}) + b - y\right)^2\right] = \left(\mathbf{w}^\top\mathbf{x} + b - y\right)^2 + s^2\|\mathbf{w}\|^2.$$

- The cross term vanishes: $\mathbb{E}[\boldsymbol{\epsilon}] = \mathbf{0}$.
- Training on noisy inputs = squared loss + weight decay $s^2\|\mathbf{w}\|^2$.

::: {.d2l-note}
Dropout moves the noise to the hidden units, and makes it large: it removes
units instead of nudging them.
:::
:::

::: {.slide title="Dropout Zeros a Unit with Probability p and Rescales the Rest"}
[Noise · the operation]{.kicker}

With a mask $\xi_j$ that is 0 with probability $p$ and 1 otherwise:

$$h_j' = \frac{\xi_j h_j}{1 - p} = \begin{cases} 0 & \textrm{with probability } p,\\ h_j/(1 - p) & \textrm{with probability } 1 - p.\end{cases}$$

- A fresh mask for every unit, every example, every step.
- $\mathbb{E}[h_j'] = p \cdot 0 + (1 - p)\frac{h_j}{1-p} = h_j$: the expected value
  is kept, and $1/(1 - p)$ is the only factor that keeps it.

::: {.d2l-note}
**Inverted dropout** rescales during training, so the test network is the
plain network. The original version (Hinton et al., 2012) scaled the
weights by $1 - p$ at test time instead.
:::
:::

::: {.slide title="The Noise Is as Large as the Activation"}
[Noise · the variance]{.kicker}

$$\mathbb{E}[h_j'^2] = \frac{h_j^2}{1 - p}, \qquad \textrm{Var}[h_j'] = \frac{p}{1 - p}\, h_j^2.$$

| $p$ | kept unit scaled by | variance for $h_j = 1$ |
|---|---:|---:|
| 0.2 | 1.25 | 0.25 |
| 0.5 | 2 | 1 |

::: {.d2l-note}
At $p = 0.5$ the standard deviation of $h_j'$ equals $|h_j|$: this is not
a small perturbation.
:::
:::

::: {.slide title="The Test Network Is Not the Average of the Thinned Networks"}
[Noise · test time]{.kicker}

Two units, $h_1 = h_2 = 1$, $p = 0.5$, pre-activation $z = h_1' - h_2'$:

| $(h_1', h_2')$ | (0,0) | (2,0) | (0,2) | (2,2) |
|---|---:|---:|---:|---:|
| $z$ | 0 | 2 | −2 | 0 |
| $\textrm{ReLU}(z)$ | 0 | 2 | 0 | 0 |

- Linear layer: $\mathbb{E}[z] = 0$, the full network's value.
- After the ReLU: $\mathbb{E}[\textrm{ReLU}(z)] = 0.5$, but the full network gives
  $\textrm{ReLU}(0) = 0$.

::: {.d2l-note}
The expected value is kept unit by unit, not through nonlinear layers.
:::
:::

::: {.slide title="Check: How Large Is the Noise at p = 0.8?"}
[Check]{.kicker}

A unit has activation $h_j = 1$ and dropout probability $p = 0.8$. What is
a kept unit scaled to, and what is the variance of $h_j'$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** A kept unit becomes $1/(1 - 0.8) = 5$, kept one time in five.
The variance is $0.8/0.2 = 4$: a standard deviation twice the
activation.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Implementation]{.dtitle}

[a dropout layer and an MLP that uses it]{.dsub}
:::
:::

::: {.slide title="A Dropout Layer in Three Lines"}
[Implementation · the layer]{.kicker}

Draw a uniform number per entry, keep the entry if it is at least $p$,
scale the survivors by $1/(1 - p)$:

@dropout-implementation-from-scratch-1
:::

::: {.slide title="A Million Ones at p = 0.2: Mean 1, Variance 0.25"}
[Implementation · check]{.kicker}

@!dropout-implementation-from-scratch-2

- $p = 0.5$: each entry is 0 with probability $1/2$, otherwise doubled.
- The sample mean and variance match $1$ and $p/(1 - p) = 0.25$.
:::

::: {.slide title="Dropout Goes after Each Hidden Activation, Only in Training"}
[Implementation · the model]{.kicker}

::: {.cols .vc}
::: {.col}
`Linear → ReLU → dropout(0.2) → Linear → ReLU → dropout(0.5) → Linear`

- Dropped units get no gradient on that example.
- Fewer units dropped near the input, more deeper.
- He initialization (variance $2/n_\textrm{in}$), zero biases, in every
  framework.
:::

::: {.col .fig .big}
![Dropping $h_2$ and $h_5$ removes them and their connections for one step.](../img/dropout2.svg)
:::
:::
:::

::: {.slide title="Training: 20 Epochs on Fashion-MNIST"}
[Implementation · training]{.kicker}

Two hidden layers of 256 units, $p = 0.2$ and $0.5$, SGD at learning rate
0.1, minibatches of 256:

@!dropout-training

::: {.d2l-note}
The training loss is computed with masks, on thinned networks; the
validation loss on the full network.
:::
:::

::: {.slide title="Concise: One Dropout Layer after Each Activation"}
[Implementation · concise]{.kicker}

A framework `Dropout(p)` layer masks and rescales in training mode and passes
its input through at test time; the `Trainer` switches the mode.

@dropout-concise-implementation-1
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Effects of Dropout]{.dtitle}

[too little to regularize · noisy labels]{.dsub}
:::
:::

::: {.slide title="On All 60,000 Images There Is Too Little to Regularize"}
[Effects · the full data]{.kicker}

- A regularizer can only remove overfitting that is there.
- After 20 epochs on all of Fashion-MNIST, the training loss, even with the
  masks switched on, ends only a little below the validation loss.
- To see what dropout does, we need a network that overfits.

::: {.d2l-note}
The last deck supplied one: 1,000 images with redrawn labels, 200 epochs.
The MLP memorized the wrong labels and lost validation accuracy doing so.
:::
:::

::: {.slide title="With Noisy Labels, Dropout Slows Memorization"}
[Effects · noisy labels]{.kicker}

1,000 images, 40% of the labels redrawn (about 36% wrong), 200 epochs,
minibatches of 64; dropout 0.5 in both hidden layers against none:

@!dropout-noisy-labels-2
:::

::: {.slide title="Dropout Slows the Memorization and Does Not Stop It"}
[Effects · reading]{.kicker}

| | no dropout | dropout 0.5 |
|---|---|---|
| wrong labels reproduced | all, by about epoch 100 | three quarters to four fifths after 200 epochs, still rising |
| validation accuracy, last ten epochs | about ten points below its early best | three to five points higher |

::: {.d2l-note}
Dropout changes which fit training finds when there is noise to fit. Both
networks end below their early best: early stopping is still worth having.
:::
:::

::: {.slide title="Check: How Often Can a Unit Count on a Given Partner?"}
[Check]{.kicker}

The second hidden layer has 256 units and $p = 0.5$. How many of them does
an example keep on average, and in what fraction of the steps are two given
units of that layer both present?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $256 \times 0.5 = 128$ units. Two given units are both kept with
probability $0.5^2 = 0.25$: a unit can count on a partner in one step of
four. In the first layer ($p = 0.2$): about 205 units, a given pair in 64%
of the steps.
:::
:::
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Interpretations]{.dtitle}

[noise, co-adaptation, an implicit ensemble]{.dsub}
:::
:::

::: {.slide title="For Linear Regression, Dropout Is Weight Decay Scaled by x²"}
[Interpretations · noise]{.kicker}

Drop the inputs $x_j' = \xi_j x_j/(1 - p)$: mean $x_j$, variance
$\frac{p}{1-p}x_j^2$, independent.

$$\mathbb{E}\left[\left(\mathbf{w}^\top\mathbf{x}' + b - y\right)^2\right] = \left(\mathbf{w}^\top\mathbf{x} + b - y\right)^2 + \frac{p}{1 - p}\sum_j x_j^2 w_j^2.$$

- Expected loss = loss of the mean + variance of the prediction.
- Each weight is penalized by the square of its input; no penalty where
  $x_j = 0$.

::: {.d2l-note}
For deep networks the penalty depends on activations that change during
training: no closed form.
:::
:::

::: {.slide title="Three Explanations, None Complete"}
[Interpretations · overview]{.kicker}

| view | what it says | limit |
|---|---|---|
| noise injection | tolerate noise → smoother function | exact only for linear models |
| co-adaptation | units must work alone | plausible; not tested by our experiment |
| implicit ensemble | $2^{512}$ thinned networks, shared weights | test network ≠ their average |

::: {.d2l-note}
Only the expected value of each unit is exact. With 512 hidden units there
are $2^{512} \approx 10^{154}$ masks.
:::
:::

::: {.slide title="Dropout Today: Sparingly, at About 0.1, or Not at All"}
[Interpretations · practice]{.kicker}

- **Where it matters:** many more parameters than examples, and many passes
  over each example.
- **Early-2010s image classifiers** (AlexNet): dropout kept large fully
  connected layers from overfitting.
- **Later architectures:** small probabilities, around 0.1.
- **Models that see each example once** (large language models): often 0.
- **Caution:** dropout before a normalization layer changes the mean square
  by $1/(1-p)$ between training and test.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Operation:** zero with probability $p$, scale by $1/(1-p)$;
  $\mathbb{E}[h_j'] = h_j$, $\textrm{Var}[h_j'] = \frac{p}{1-p}h_j^2$.
- **Test time:** full network, no masks; not the average of the thinned
  networks once a nonlinearity intervenes.
:::

::: {.col}
- **Fashion-MNIST, 2 × 256, 20 epochs:** about 87%; too little overfitting
  for dropout to matter.
- **1,000 images, 40% of labels redrawn:** no dropout memorizes every wrong
  label; dropout 0.5 memorizes more slowly and ends three to five points
  higher.
- **Linear model:** dropout = weight decay scaled by $x_j^2$.
- **Why it works:** noise, co-adaptation, ensemble; each a partial account.
:::
:::

**Next:** house prices on Kaggle, a table of mixed numbers and categories,
where preprocessing and validation decide the result.
:::
