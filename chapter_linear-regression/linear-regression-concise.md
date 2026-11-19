```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation
:label:`sec_linear_concise`

In :numref:`sec_linear_scratch`, we built linear regression completely from
scratch—allocating parameters by hand, writing out the forward pass and squared
loss, and coding minibatch stochastic gradient descent ourselves—and in
:numref:`sec_oo-design`, we organized those pieces into reusable `Module`,
`DataModule`, and `Trainer` classes. Writing everything by hand is feasible when
a model has just three parameters, and doing so once is invaluable for
understanding how deep learning works under the hood. Yet as models grow, it
quickly becomes impractical—much like coding a blog engine from scratch is a
great learning exercise, while building every production website that way would
be a waste of time. A standard image classifier such as ResNet-50
(:numref:`sec_resnet`), for instance, contains $25.6$ million parameters spread
across $161$ tensors and relies on stateful optimizers that maintain running
averages for every single parameter. Worse still, subtle mistakes in hand-rolled
layers or optimizers rarely crash—the network simply trains more slowly or less
accurately without telling you why.

This is why modern deep learning frameworks—MXNet :cite:`Chen.Li.Li.ea.2015`,
JAX :cite:`Frostig.Johnson.Leary.2018` (with Flax and Optax), PyTorch
:cite:`Paszke.Gross.Massa.ea.2019`, and TensorFlow
:cite:`Abadi.Barham.Chen.ea.2016`—ship battle-tested, hardware-tuned
implementations of standard layers, loss functions, and optimizers behind clean
high-level APIs. Swapping one loss or optimizer for another becomes a one-line
edit. In this section, we replace our hand-written model parameters, loss, and
optimizer with each framework's built-in counterparts while leaving
`d2l.Trainer` and `SyntheticRegressionData` completely untouched. The only
detail requiring a sharp eye is the scaling factor of the built-in loss.

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## The Fully Connected Layer

Recall from :numref:`fig_single_neuron` that linear regression is simply a
single *fully connected* (or *dense*) neural network layer: every input feature
connects to the output via a matrix--vector product $\mathbf{X}\mathbf{w} + b$.
Every framework provides this layer out of the box, creating, storing, and
registering its own weight and bias tensors automatically. All we need to specify
is the output dimensionality (`1`) and the initialization scheme—matching our
from-scratch model with zero-mean normal weights of standard deviation `0.01`
and a bias initialized to `0`.

:begin_tab:`mxnet`
In Gluon, the fully connected layer is `nn.Dense(1)`. Conveniently, it infers
the input dimensionality automatically on the first forward pass, instantiating
and initializing its weights according to `init.Normal(sigma=0.01)` and its bias
to zero.
:end_tab:

:begin_tab:`pytorch`
PyTorch provides `nn.LazyLinear(1)`, which defers creating its weight matrix
until the first forward pass reveals the input dimension $d$—saving us from
tedious shape bookkeeping in deeper networks. We define a `weight_init` helper
that applies our $\mathcal{N}(0, 0.01^2)$ weight and zero-bias initialization to
any linear layer (starting in :numref:`sec_use_gpu`, our trainer invokes it
automatically after lazy initialization).
:end_tab:

:begin_tab:`tensorflow`
In Keras, the fully connected layer is `tf.keras.layers.Dense(1)`. Like Gluon,
it infers the input dimension from the first batch it sees, draws weights from
`kernel_initializer`, and initializes the bias to zero by default.
:end_tab:

:begin_tab:`jax`
In Flax NNX, the fully connected layer is `nnx.Linear(num_inputs, 1)`. Because
NNX initializes parameters eagerly rather than lazily, we pass `num_inputs`
explicitly alongside `kernel_init=nnx.initializers.normal(0.01)` and an `rngs`
key stream (defaulting to a fresh key from `d2l.get_key()`); the bias defaults
to zero.
:end_tab:

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab pytorch
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab mxnet
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Dense(1)
        self.net.initialize(init.Normal(sigma=0.01))
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab tensorflow
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        init = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.layers.Dense(1, kernel_initializer=init)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, num_inputs, lr, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(
            num_inputs, 1, kernel_init=nnx.initializers.normal(0.01),
            rngs=rngs)
```

Notice that `LinearRegression` does not even need to define `forward`: it
inherits `Module.forward` from :numref:`sec_oo-design`, which delegates directly
to `self.net(X)`.

## The Loss Function

In our from-scratch implementation, we averaged *half* the squared error across
each minibatch $\mathcal{B}$. Most deep learning libraries instead compute the
standard *mean squared error* (MSE) without the leading factor of $\frac{1}{2}$:

$$\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left(\hat{y}^{(i)} - y^{(i)}\right)^2,$$

whose square root (RMSE) directly expresses the typical prediction error in the
units of the target. Dropping the $\frac{1}{2}$ leaves the global minimizer
unchanged, but it doubles every gradient. Recall from
:numref:`subsec_linear-regression-minibatch-sgd` that a learning rate is
meaningful only relative to the scale of its loss: to take the exact same SGD
steps as our scratch model took at $\eta = 0.03$, we must halve the learning
rate to $\eta = 0.015$.

It is also worth checking how each library's loss handles the silent
broadcasting trap from :numref:`sec_linear_scratch`—passing 1D labels of shape
`(32,)` alongside column predictions of shape `(32, 1)`:

:begin_tab:`mxnet`
Gluon's `L2Loss` includes the $\frac{1}{2}$ factor per example, so we multiply
by `2` and take `.mean()` to match the standard MSE used across the other
frameworks. Internally, `L2Loss` reshapes the labels to match the predictions,
preventing the silent broadcasting bug automatically.
:end_tab:

:begin_tab:`pytorch`
PyTorch's `nn.MSELoss` computes the mean squared error over all elements. Beware:
if `y_hat` and `y` have mismatched shapes `(32, 1)` and `(32,)`, `nn.MSELoss`
emits a warning and *still* broadcasts them into a $32 \times 32$ error matrix,
silently training on the wrong objective! Always heed those warnings and keep
target shapes aligned.
:end_tab:

:begin_tab:`tensorflow`
Note the argument order of Keras's `tf.keras.losses.MeanSquaredError`: true
labels `y` come *first* and predictions `y_hat` come *second*. Keras aligns their
shapes before subtracting, preventing the broadcasting trap.
:end_tab:

:begin_tab:`jax`
In JAX, losses are plain array functions: `optax.squared_error(y_hat, y)`
computes elementwise squared errors, which we average with `.mean()`. Better
yet, `optax.squared_error` raises an error if shapes disagree, turning a silent
bug into an immediate catch.
:end_tab:

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = gluon.loss.L2Loss()
    return 2 * fn(y_hat, y).mean()  # L2Loss includes the 1/2
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = tf.keras.losses.MeanSquaredError()
    return fn(y, y_hat)
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    return optax.squared_error(y_hat, y).mean()
```

## The Optimizer

Next, we replace our hand-written `SGD` class from :numref:`sec_oo-design` with
the framework's native SGD optimizer. Every step applies the same rule—subtracting
$\eta$ times the minibatch gradient from each parameter—except now the optimizer
discovers every layer's parameters automatically. Even better, because every
framework exposes momentum, Adam, and other modern optimizers behind the exact
same interface, switching optimization algorithms later takes only a one-line
change inside `configure_optimizers`.

:begin_tab:`mxnet`
In Gluon, the optimizer object is named `gluon.Trainer` (not to be confused with
our `d2l.Trainer` training loop). We pass `self.collect_params()`, the algorithm
name `'sgd'`, and `{'learning_rate': self.lr}`. Because Gluon's `step(batch_size)`
divides gradients by `batch_size`, our loop passes `step(1)` since our loss is
already averaged.
:end_tab:

:begin_tab:`pytorch`
`torch.optim.SGD` accepts `self.parameters()`—automatically gathering the weight
and bias of `self.net` (and of any deeper layers in future models)—alongside the
learning rate `self.lr`.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.optimizers.SGD` takes `self.lr`; during each training step, our
trainer passes the model's `trainable_variables` paired with their gradients to
`apply_gradients`.
:end_tab:

:begin_tab:`jax`
`optax.sgd(self.lr)` returns a stateless *gradient transformation* (pure `init`
and `update` functions) that our `Trainer` wraps in `nnx.Optimizer` to update
`model`'s parameters (:numref:`sec_oo-design`).
:end_tab:

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return gluon.Trainer(self.collect_params(),
                         'sgd', {'learning_rate': self.lr})
```

## Training

Now let us train our concise model on the exact same synthetic problem
($\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, $\sigma = 0.01$, $1{,}000$ training
examples in minibatches of $32$, and $1{,}000$ validation examples), using
`lr=0.015` to compensate for the doubled MSE loss. Started from identical
initial weights and fed identical minibatches, this concise model would take the
exact same updates as our scratch model (as you will verify in the exercises);
here we draw fresh data, batches, and initial weights, so we expect the same
qualitative curve and accuracy rather than bit-identical digits.

:begin_tab:`pytorch`
Before starting `fit`, we pass a single example `data.X[:1]` through `model` to
materialize the parameters of `nn.LazyLinear`, and then apply `model.weight_init`
so the layer uses our $\mathcal{N}(0, 0.01^2)$ initialization instead of
PyTorch's default uniform initialization.
:end_tab:

As before, we set `trainer.board.yscale = 'log'` so the loss remains readable
across five orders of magnitude.

```{.python .input #linear-regression-concise-training-1}
%%tab pytorch
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(lr=0.015)
model(data.X[:1])  # Creates the parameters of the lazy layer
model.net.apply(model.weight_init)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab mxnet, tensorflow
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(lr=0.015)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab jax
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(2, lr=0.015)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

The training and validation curves fall in lockstep at the exact same rate we
predicted in :numref:`sec_linear_scratch`—roughly $7\times$ per epoch—before
flattening out near $10^{-4}$ around epoch 8. Notice why the floor sits at
$\sigma^2 = 10^{-4}$ instead of $\sigma^2/2 = 5 \times 10^{-5}$: the MSE loss
omits the factor of $\frac{1}{2}$, doubling every loss value. Let us inspect the
final validation loss stored in `trainer.history`:

```{.python .input #linear-regression-concise-training-4}
print(f"validation loss: {trainer.history['val_loss'][-1]:.2e}")
```

The final validation loss lands within roughly $15\%$ of the theoretical noise
floor $\sigma^2 = 10^{-4}$.

To verify parameter recovery directly, we add a small `get_w_b` helper that
extracts the learned weight matrix and bias vector from `self.net` (remembering
that PyTorch and Gluon store weights as $1 \times d$ row matrices whereas Keras
and Flax store them as $d \times 1$ `kernel` columns, so we reshape before
subtracting $\mathbf{w}^*$):

```{.python .input #linear-regression-concise-training-2}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.weight.detach(), self.net.bias.detach()
```

```{.python .input #linear-regression-concise-training-2}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.weight.data(), self.net.bias.data()
```

```{.python .input #linear-regression-concise-training-2}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.kernel, self.net.bias
```

```{.python .input #linear-regression-concise-training-2}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.kernel[...], self.net.bias[...]
```

```{.python .input #linear-regression-concise-training-3}
w, b = model.get_w_b()
w_error = d2l.numpy(d2l.reshape(w, data.w.shape) - data.w)
b_error = d2l.numpy(b - data.b)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

Once again, the errors in $\hat{\mathbf{w}}$ and $\hat{b}$ are only a few times
$10^{-4}$, matching our scratch run because the dominant error sources (label
noise across $1{,}000$ examples and the remaining distance after $10$ epochs) are
identical regardless of who wrote the layer.

Throughout the rest of this book, we will build models from high-level framework
components whenever they exist—writing custom code from scratch only to demystify
a new mechanism or when inventing something no library yet provides. Yet one
crutch we leaned on here must now be left behind: checking learned parameters
against known ground truth. On real-world datasets, nobody hands us
$\mathbf{w}^*$ and $b^*$; all we can observe is how well our model generalizes to
unseen data—the subject of :numref:`sec_generalization_basics`.

## Summary

* Modern deep learning frameworks (PyTorch, JAX/Flax/Optax, TensorFlow/Keras,
  and MXNet/Gluon) provide optimized, well-tested layers, losses, and optimizers
  that replace hand-written boilerplate while plugging seamlessly into our
  `Module`, `DataModule`, and `Trainer` abstractions.
* Built-in mean squared error losses omit the textbook factor of $\frac{1}{2}$,
  doubling both the loss values (moving the noise floor from $\sigma^2/2$ to
  $\sigma^2$) and the gradients—so we halve the learning rate ($\eta = 0.015$)
  to match our scratch optimization steps.
* Using framework building blocks saves effort and boosts speed, while knowing
  how to write components from scratch remains indispensable whenever you need
  to debug unexpected behavior or pioneer new architectures.

## Exercises

1. [code] **Loss scaling.** The framework loss in `LinearRegression.loss`
   is twice the loss of `LinearRegressionScratch`.
    1. For one minibatch $\mathcal{B}$ of `SyntheticRegressionData`,
       compute the gradient of `LinearRegression.loss` with respect to the
       weights by automatic differentiation, and compare it entry by entry
       with the formula
       $\frac{2}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \mathbf{x}^{(i)}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)$
       at the same parameters.
    1. Give `LinearRegression` and `LinearRegressionScratch` the same
       initial parameters and the same minibatches, for example by setting
       `batch_size` to `num_train` so that every step uses the whole
       training set. Train the first at `lr=0.015` and the second at
       `lr=0.03`, confirm that the trained parameters agree, and explain
       why the two training-loss curves still differ.

1. [code] **A layer without a bias.** Build `LinearRegression` with a
   layer that has no bias (every framework's dense layer has an argument
   that turns the bias off), and train it as in this section.
    1. Before training, predict the value at which the validation loss
       levels off, from the expected loss
       :eqref:`eq_linreg-scratch-expected-loss` doubled for the mean
       squared error. Then check your prediction.
    1. Predict what the weights converge to and how far they end from
       $\mathbf{w}^*$. Since the features have mean zero, the missing bias
       acts on the weights like label noise of standard deviation
       $b^* = 4.2$; it enters both the estimation error of
       :numref:`subsec_synthetic-regression-data-recovery` and the noise
       ball of minibatch SGD (:numref:`sec_linear_regression`). Compare
       your prediction with the errors of the trained weights.
   :exlabel:`ex_linear-regression-concise-no-bias`

1. [code] **Several outputs.** Change the output width of the layer in
   `LinearRegression` from 1 to 2. After constructing `data`, append to
   `data.y` a second label column
   $y_2 = \mathbf{x}^\top \mathbf{w}_2^* + b_2^* + \epsilon_2$, computed
   from `data.X` with generating parameters of your choice and noise of
   the same size.
    1. Show that the squared loss summed over the two outputs splits into
       two independent linear regressions, one per output. Conclude that
       the parameters learned for one output do not depend on the labels of
       the other.
    1. The framework losses average over every entry of the
       $|\mathcal{B}| \times 2$ matrix of errors of a minibatch
       $\mathcal{B}$, not only over its examples. State by which factor
       the gradient of each output's parameters shrinks relative to the
       single-output model, and which learning rate reproduces the
       single-output training run. Confirm this by training both models
       from the same initial parameters on the same minibatches.

1. [code] **Feature scale and step size.** Multiply the first column of
   `data.X` by 100 after constructing `data`, leaving `data.y` unchanged,
   so that the generating weight of that feature becomes $0.02$. Recall
   from exercise :exref:`ex_linear-regression-scratch-stability` that
   gradient descent diverges once $\eta$ exceeds $2/\mu_{\max}$, where
   $\mu_{\max}$ is the largest eigenvalue of the Hessian $\mathbf{H}$ of
   the loss.
    1. Compute $\mathbf{H}$ for `LinearRegression.loss` and estimate its
       eigenvalues before and after the rescaling.
    1. Predict the largest learning rate for which training does not
       diverge, before and after the rescaling, and check both predictions
       by training at learning rates around them.
    1. At a learning rate just below the new limit, predict how many
       updates the second weight needs to reduce its error by a factor of
       100, and compare the prediction with a measurement. Then standardize
       the rescaled feature as in :numref:`sec_pandas`, and show that
       `lr=0.015` works again.

1. [code] **Optimization versus estimation error.** ● Train
   `LinearRegression` with `lr=0.015` and `max_epochs=10` on
   `SyntheticRegressionData` with `num_train` set to each of
   $5, 50, 500, 5{,}000$ and `batch_size=min(32, num_train)`. For each run,
   compare the distance $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$ of the
   trained weights from the generating weights $\mathbf{w}^*$, stored in
   `data.w`, with the same distance for the least-squares solution on the
   same training data, computed as in the recovery subsection
   (:numref:`subsec_synthetic-regression-data-recovery`).
    1. Count the parameter updates in each run, and explain the ratio of
       the two errors at each `num_train`.
    1. Find the number of epochs after which the run with `num_train=5`
       comes within a factor of two of its least-squares error.


:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/204)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17977)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.5]{.kicker}

Linear regression from framework parts<br>**layer · loss · optimizer · the same trainer**
:::
:::

::: {.slide title="Hand-Written Parts Do Not Scale to 25 Million Parameters"}
[The problem]{.kicker}

- From scratch: we created, initialized, and updated **3 numbers**, `w`
  and `b`, with our own code, then packed that code into classes.
- ResNet-50, a standard image classifier: **25.6 million** parameters in
  **161 tensors**, trained with update rules that keep a running average
  of past gradients for every parameter.
- A slip in a hand-written part rarely raises an error. The model trains,
  only less well.

::: {.d2l-note}
Frameworks ship layers, losses, and optimizers: tested by many users, fast
on every device, and swapped by changing one line.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- name the framework part that replaces each hand-written part;
- build linear regression from a layer, a loss, and an optimizer;
- convert a learning rate between losses that differ by a constant factor;
- say what each framework loss does with labels of the wrong shape;
- predict where the loss levels off, and judge the trained parameters.

**Running example:** $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, noise
$\sigma = 0.01$; 1,000 training examples in minibatches of 32.
:::

::: {.slide title="The Concise Version Reuses Everything but the Model's Parts"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| Synthetic Data | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | SGD and the training loop by hand |
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| **Concise Implementation** | **framework layers and losses** |
| Generalization | training vs. test error |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** `Module`, `Trainer`, `SyntheticRegressionData`, the scratch
  model.
- **Skips:** how layers infer shapes and initialize (the Computation
  chapter); other update rules (the Optimization chapter).
:::
:::
:::

::: {.slide title="Each Hand-Written Part Has a Framework Counterpart"}
[Overview]{.kicker}

| from scratch | framework |
|---|---|
| `w`, `b`, and `X @ w + b` | one fully connected **layer** |
| $\tfrac{1}{2}$ squared error, averaged | built-in **mean squared error** |
| our `SGD` class | an **optimizer** object |
| `Module`, `DataModule`, `Trainer` | unchanged |

::: {.d2l-note}
The trainer calls only the methods that every `Module` has. It cannot tell
the two models apart.
:::
:::

::: {.slide title="LazyLinear(1) Is the Whole Model" only="pytorch"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-1}
self.net = nn.LazyLinear(1)  # 1 output
```

- `LazyLinear` takes the number of inputs from its first input.
- Its parameters do not exist before that first call, so we initialize
  them after it, from $\mathcal{N}(0, 0.01^2)$ and with bias 0.
:::

::: {.slide title="Dense(1) Is the Whole Model" only="mxnet"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-2}
self.net = nn.Dense(1)  # 1 output
self.net.initialize(init.Normal(sigma=0.01))
```

- `Dense` takes the number of inputs from its first input.
- `initialize` records the initializer; the parameters are created at the
  first forward pass, the bias as 0.
:::

::: {.slide title="Dense(1) Is the Whole Model" only="tensorflow"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-3}
init = tf.initializers.RandomNormal(stddev=0.01)
self.net = tf.keras.layers.Dense(1, kernel_initializer=init)
```

- `Dense` takes the number of inputs from its first input.
- `kernel_initializer` draws the weights; the bias starts at 0.
:::

::: {.slide title="nnx.Linear(2, 1) Is the Whole Model" only="jax"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-4}
self.net = nnx.Linear(
    num_inputs, 1, kernel_init=nnx.initializers.normal(0.01),
    rngs=rngs)
```

- NNX has no lazy layers: we pass the number of inputs.
- `rngs` supplies the random keys; the parameters exist as soon as the
  constructor returns. The bias starts at 0.
:::

::: {.slide title="The Framework Loss Is Twice Ours, So η Halves" except="mxnet"}
[Loss]{.kicker}

@linear-regression-concise-defining-the-loss-function

| | loss | gradient |
|---|---|---|
| ours | $\frac{1}{2\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $g$ |
| framework | $\frac{1}{\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $2g$ |

Same minimizer, but every gradient doubles. A step is $\eta$ times the
gradient, so $\eta = 0.03/2 = 0.015$ takes the scratch steps.
:::

::: {.slide title="The Framework Loss Is Twice Ours, So η Halves" only="mxnet"}
[Loss]{.kicker}

@linear-regression-concise-defining-the-loss-function

| | loss | gradient |
|---|---|---|
| ours | $\frac{1}{2\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $g$ |
| doubled `L2Loss` | $\frac{1}{\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $2g$ |

`L2Loss` includes the $\frac{1}{2}$; doubled, it is the mean squared
error. Every gradient doubles, so $\eta = 0.03/2 = 0.015$ takes the
scratch steps.
:::

::: {.slide title="Check: A Summed Loss"}
[Check]{.kicker}

A library's loss **sums** $(\hat{y} - y)^2$ over the minibatch of 32,
with no $\frac{1}{2}$. Which $\eta$ reproduces the scratch steps at
$\eta = 0.03$?

::: {.d2l-note .fragment .fade-up}
**Answer.** The sum is 32 times the mean, and the mean is twice our loss:
the gradient is $64g$. So $\eta = 0.03/64 \approx 4.7 \times 10^{-4}$.
:::
:::

::: {.slide title="MSELoss Warns About a Shape Mismatch, Then Trains on the Wrong Loss" only="pytorch"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ broadcast
to a $32 \times 32$ matrix of differences.

- `nn.MSELoss` prints a warning, then averages all 1,024 entries.
- That loss pulls every prediction toward the mean of the 32 labels.

::: {.d2l-note .warn}
Read the warnings. Reshape the labels to the predictions' shape.
:::
:::

::: {.slide title="L2Loss Reshapes the Labels, So the Shape Bug Cannot Occur" only="mxnet"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ would
broadcast to a $32 \times 32$ matrix of differences.

- `L2Loss` first reshapes the labels to the shape of the predictions.
- The difference has shape $(32, 1)$, as intended.
:::

::: {.slide title="Keras Aligns the Shapes, So the Shape Bug Cannot Occur" only="tensorflow"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ would
broadcast to a $32 \times 32$ matrix of differences.

- `MeanSquaredError` brings both to one shape before subtracting.
- Note the order: label first, prediction second.
:::

::: {.slide title="optax.squared_error Refuses Mismatched Shapes" only="jax"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ would
broadcast to a $32 \times 32$ matrix of differences.

- `optax.squared_error` raises an error instead.
- The bug is no longer silent: fix the shape and run again.
:::

::: {.slide title="An Optimizer Object Replaces Our SGD Class"}
[Optimizer]{.kicker}

@linear-regression-concise-defining-the-optimization-algorithm

- Each step: $\theta \leftarrow \theta - \eta\, \nabla_\theta L$ for every
  parameter $\theta$ of every layer.
- Another update rule from the same module: change this one line.
:::

::: {.slide title="The Same Trainer Fits the Framework Model"}
[Training]{.kicker}

@-linear-regression-concise-training-1

- Data drawn as before; a new model class at $\eta = 0.015$.
- `Trainer` and `fit` are unchanged: they call only the methods that
  every `Module` has.
:::

::: {.slide title="The Loss Falls About 7× per Epoch, Then Levels Off at σ² = 10⁻⁴"}
[Training]{.kicker}

::: {.cols .vc}
::: {.col}
Ten epochs at $\eta = 0.015$, log scale.

A straight line, the per-epoch factor of the scratch run, then the floor:
the noise variance $\sigma^2$.
:::

::: {.col .fig .big}
@!linear-regression-concise-training-1
:::
:::
:::

::: {.slide title="Ten Epochs Recover w* and b* to Within About 10⁻³"}
[Parameters]{.kicker}

The parameters live inside the layer; `get_w_b` reads them out of `net`:

@linear-regression-concise-training-2

@!linear-regression-concise-training-3

::: {.d2l-note}
Errors of a few times $10^{-4}$, the size the scratch run left, from the
same sources: the noise in the training labels and what ten epochs leave
of the initial error.
:::
:::

::: {.slide title="Check: Why Does This Loss Level Off Twice as High?"}
[Check]{.kicker}

The scratch run levels off near $5 \times 10^{-5}$, this run near
$10^{-4}$, on data from the same generator. Why the factor 2? And what
does the square root of this run's floor tell us about the predictions?

::: {.d2l-note .fragment .fade-up}
**Answer.** Even the true parameters leave the noise, of variance
$\sigma^2$. The mean squared error levels off at $\sigma^2$; our loss
carries the $\frac{1}{2}$ and levels off at $\sigma^2/2$. And
$\sqrt{10^{-4}} = 0.01 = \sigma$: a typical prediction misses its label by
the noise level, in the units of the label.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| | from scratch | framework |
|---|---|---|
| parameters, forward | `w`, `b`, `X @ w + b` | fully connected layer |
| loss | $\frac{1}{2}$ mean squared error | mean squared error |
| update | our `SGD`, $\eta = 0.03$ | optimizer, $\eta = 0.015$ |
| training loop | `Trainer` | the same `Trainer` |
| loss floor | $\sigma^2/2 = 5 \times 10^{-5}$ | $\sigma^2 = 10^{-4}$ |

**Next:** how well does a trained model predict examples it has not seen?
:::
