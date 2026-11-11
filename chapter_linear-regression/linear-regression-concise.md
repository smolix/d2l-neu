```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation
:label:`sec_linear_concise`

In :numref:`sec_linear_scratch` we wrote every part of linear regression
by hand, from the initialization of the parameters to the update of
minibatch stochastic gradient descent. In :numref:`sec_oo-design` we
arranged the same parts into classes and trained the model once more
through `d2l.Trainer`. Both steps were feasible because the model has three
parameters. The models of later chapters are far larger. The image
classifier ResNet-50 of :numref:`sec_resnet`, for instance, has 25.6
million parameters in 161 tensors, and it is trained with update rules that
keep a running average of past gradients for every parameter. Writing and
testing all of this by hand for every new model would take most of our
effort. Worse, a mistake in any part rarely raises an error. The model
still trains, only less well.

Deep learning frameworks such as MXNet :cite:`Chen.Li.Li.ea.2015`, JAX
:cite:`Frostig.Johnson.Leary.2018` with its libraries Flax and Optax,
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, and TensorFlow
:cite:`Abadi.Barham.Chen.ea.2016` therefore provide these parts ready-made.
Their layers, losses, and optimizers are tested by many users and run on
code tuned for each kind of hardware, and one loss or optimizer can be
swapped for another by changing a line. This section swaps our hand-written
parts for the framework's. A fully connected layer holds the weights and
the bias and computes the predictions, a built-in loss computes the squared
error, and an optimizer object applies the update. The `Trainer` runs
unchanged, because it calls only the methods that every `Module` has. Only
the scale of the loss needs some care.

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

Linear regression is a single fully connected layer
(:numref:`fig_single_neuron`), and every framework provides that layer.
For a minibatch with design matrix $\mathbf{X}$ it computes the predictions
$\mathbf{X}\mathbf{w} + b$ that the `forward` method of our scratch model
wrote out, and it creates and stores its own parameters. All that is left
for us to choose is the number of outputs, here 1, and how the parameters
are initialized. As in the scratch model, the weights are drawn from a
normal distribution with mean 0 and standard deviation 0.01, and the bias
starts at 0.

:begin_tab:`mxnet`
In Gluon the fully connected layer is `nn.Dense(1)`, which takes the number
of inputs from the first input it receives and creates its parameters then.
The weights are drawn as `initialize` prescribes, and the bias starts at
zero.
:end_tab:

:begin_tab:`pytorch`
PyTorch's `nn.LazyLinear(1)` takes the number $d$ of inputs from the first
input it receives and has no parameters until then, which spares us
counting the inputs of every layer in deeper networks. The method
`weight_init` states our initialization for any linear layer; from
:numref:`sec_use_gpu` on, the trainer applies it after the first forward
pass.
:end_tab:

:begin_tab:`tensorflow`
In Keras the fully connected layer is `tf.keras.layers.Dense(1)`, which
takes the number of inputs from the first input it receives and creates the
weights then, drawn from the `kernel_initializer`. The bias starts at zero
by default.
:end_tab:

:begin_tab:`jax`
In Flax NNX the fully connected layer is `nnx.Linear(d, 1)`, which needs
the number $d$ of inputs, since NNX has no lazy layers. It draws its
weights from `kernel_init` with random keys from `rngs`, by default from
`d2l.get_key()`, which starts from a new random seed on every run; the bias
starts at zero.
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

The model needs no `forward` method of its own. It inherits the `forward`
of `Module` (:numref:`sec_oo-design`), which passes its input to
`self.net`, here the layer.

## The Loss Function

Our scratch loss was half the squared error, averaged over the minibatch
$\mathcal{B}$. Most frameworks provide the *mean squared error* instead,

$$\frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \left(\hat{y}^{(i)} - y^{(i)}\right)^2,$$

which has no factor $\frac{1}{2}$. It is the usual measure of prediction
quality in regression: its square root is the typical size of an error, in
the units of the label. The two losses have the same minimizer, so what we
fit does not change. The steps do. A learning rate has a meaning only
together with the scale of its loss
(:numref:`subsec_linear-regression-minibatch-sgd`), and the framework loss
is twice ours. To repeat the steps that the scratch model took at
$\eta = 0.03$, this model must therefore be trained at $\eta = 0.015$.

The loss also decides what becomes of the silent error of
:numref:`sec_linear_scratch`, in which labels of shape $(32,)$ are broadcast
against predictions of shape $(32, 1)$ into a $32 \times 32$ matrix of
differences.

:begin_tab:`mxnet`
Gluon's `L2Loss` computes $\frac{1}{2}(\hat{y} - y)^2$ for each example,
so we double it and average over the minibatch to obtain the mean squared
error. It first reshapes the labels to the shape of the predictions, so the
shape error cannot occur.
:end_tab:

:begin_tab:`pytorch`
`nn.MSELoss` averages the squared errors over all entries of its inputs,
here the examples of the minibatch. If labels and predictions differ in
shape, it warns and then averages over the $32 \times 32$ matrix of
differences anyway, so training proceeds on the wrong loss.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.losses.MeanSquaredError` takes the label first and the prediction
second. It brings both to the same shape before subtracting, so the shape
error cannot occur.
:end_tab:

:begin_tab:`jax`
In JAX a loss is an ordinary function of arrays: `optax.squared_error`
computes the elementwise squared errors, which we average over the
minibatch. It refuses labels and predictions that differ in shape, so the
error is no longer silent.
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

The optimizer replaces our `SGD` class of :numref:`sec_oo-design`. It
receives the parameters and the learning rate $\eta$, and at each step it subtracts $\eta$ times the
gradient of the minibatch loss from each parameter, the update we wrote by
hand. The optimizer finds the parameters through the layers of the model,
so a model with more layers needs no change here. Each framework offers the
other common update rules behind the same interface, and replacing minibatch
SGD by one of them changes only this one line.

:begin_tab:`mxnet`
In Gluon the optimizer is called `Trainer`, although it only applies
updates; it takes the parameters, collected by `self.collect_params()`, the
name of the update rule, and a dictionary of its hyperparameters. Its
method `step(batch_size)` divides the gradient by `batch_size`, so our
trainer calls `step(1)`, since our loss is already a minibatch average.
:end_tab:

:begin_tab:`pytorch`
`torch.optim.SGD` takes the parameters as `self.parameters()`, which
collects the parameters of every layer of the model, here the weight and
the bias of `net`.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.optimizers.SGD` takes the learning rate. The parameters arrive at
each step together with their gradients, through `apply_gradients`, which
our trainer calls with the model's `trainable_variables`.
:end_tab:

:begin_tab:`jax`
`optax.sgd` returns a *gradient transformation*: a pair of pure functions,
one that creates the optimizer's state and one that turns gradients into
parameter updates. Our trainer wraps it in an `nnx.Optimizer`, which applies
the updates to the parameters of the model (:numref:`sec_oo-design`).
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

We train on data drawn as before, from the generator of
:numref:`sec_synthetic-regression-data` with $\mathbf{w}^* = (2, -3.4)$,
$b^* = 4.2$, and noise of standard deviation $\sigma = 0.01$: 1,000
training examples, served in minibatches of 32, and 1,000 validation
examples. From the same initial parameters and on the same minibatches,
the two models would compute the same updates (an exercise arranges this).
Here the examples, their order, and the initial weights are all drawn
afresh, so we should expect the behavior of the scratch run but not its
digits.

:begin_tab:`pytorch`
Before training we call the model once on one example to create the
parameters of the lazy layer, and `apply` then calls `weight_init` on every
layer of `net`. Without this line the layer would keep PyTorch's default
initialization, a uniform distribution whose width shrinks with the number
of inputs.
:end_tab:

We also put the trainer's plot, `trainer.board`, on a logarithmic scale, so
that the loss stays visible after it has fallen by several orders of
magnitude.

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

The plot shows the mean training loss of each epoch and the validation loss
after each epoch. Both curves fall along a straight line, by a factor of
about 7 per epoch, and level off near $10^{-4}$ after about eight epochs.
We have seen this plot before. The slope is the one that
:numref:`sec_linear_scratch` predicted and measured, as it must be if the
steps are the same. Only the height differs, since every loss is now
doubled. In particular the floor, the loss of the true parameters, is the
noise variance $\sigma^2 = 10^{-4}$ and no longer $\sigma^2/2$. To compare
the run with this number we read the final validation loss from
`trainer.history`.

```{.python .input #linear-regression-concise-training-4}
print(f"validation loss: {trainer.history['val_loss'][-1]:.2e}")
```

The final validation loss lies within about 15% of $\sigma^2 = 10^{-4}$.

What about the parameters themselves? The scratch model kept them as
`self.w` and `self.b`. Here they belong to the layer, and `get_w_b` reads
them out of `net`. A layer stores its weight as a matrix (a $1 \times d$ row
in Gluon and PyTorch, a $d \times 1$ column called the `kernel` in Keras and
Flax), so the comparison below first reshapes it to the shape of
$\mathbf{w}^*$.

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

The parameter errors $\hat{\mathbf{w}} - \mathbf{w}^*$ and $\hat{b} - b^*$
are typically a few times $10^{-4}$ and rarely exceed $10^{-3}$. The
scratch model left errors of the same size, and they have the same sources
(:numref:`sec_linear_scratch`), chiefly the noise in the 1,000 training
labels and what ten epochs leave of the initial error. Neither depends on
who wrote the layer.

The rest of the book builds its models this way, from the layers, losses,
and optimizers of the framework, and writes a part by hand only to show how
it works or when the framework does not provide it. One habit of this
chapter cannot be kept, however: judging a trained model by its distance
from parameters that we chose ourselves. With real data there are no true
parameters to compare with, and what matters is how well a model predicts
examples it has not seen, the question of
:numref:`sec_generalization_basics`.

## Summary

Deep learning frameworks provide tested and fast versions of the parts we
wrote by hand for linear regression. A fully connected layer holds the
weights and the bias and computes the predictions, a loss function computes
the mean squared error, and an optimizer applies the SGD update to every
parameter of the model. Replacing each hand-written part by its framework
counterpart leaves `Module`, `DataModule`, and `Trainer` unchanged. A
framework loss may differ from ours by a constant factor, and the learning
rate must then be rescaled to match. Writing a part by hand remains the way
to understand it, and the way to build one that the framework lacks.

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
