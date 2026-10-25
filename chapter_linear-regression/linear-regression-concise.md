```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Concise Implementation of Linear Regression
:label:`sec_linear_concise`

In :numref:`sec_linear_scratch` we wrote every part of linear regression
ourselves: the parameters and their initialization, the forward pass, the
squared loss, and the update of minibatch stochastic gradient descent.
Writing them out showed what each part computes, and we could check every
line against the formulas of :numref:`sec_linear_regression`. That model had
two weights and a bias. ResNet-50, a standard network for classifying
images, has 25.6 million parameters in 161 tensors, and it is trained with
update rules that keep a running average of past gradients for every
parameter. Writing and testing all of this by hand for every new model would
take most of our effort. Worse, a mistake in any part rarely raises an
error. The model still trains, only less well.

Deep learning frameworks such as MXNet :cite:`Chen.Li.Li.ea.2015`, JAX
:cite:`Frostig.Johnson.Leary.2018` with its libraries Flax and Optax,
PyTorch :cite:`Paszke.Gross.Massa.ea.2019`, and TensorFlow
:cite:`Abadi.Barham.Chen.ea.2016` therefore provide these parts ready-made.
Their layers, losses, and optimizers are tested by many users and run on
code tuned for each kind of hardware, and one loss or optimizer can be
swapped for another by changing a line. This section trains the same model
on the same synthetic data with framework parts only. A fully connected
layer holds the weights
and the bias and computes the predictions, a built-in loss computes the
squared error, and an optimizer object applies the update. The `Trainer`
of :numref:`sec_oo-design` runs unchanged, because it calls only the methods
that every `Module` has. One difference needs care: the framework loss is
twice ours, which doubles the step that a given learning rate takes.

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #linear-regression-concise-concise-implementation-of-linear-regression}
%%tab jax
from d2l import jax as d2l
from flax import nnx
import optax
```

## The Fully Connected Layer

Linear regression is a network with a single layer: every input feature is
connected to the one output, which computes $\mathbf{w}^\top \mathbf{x} + b$
(:numref:`fig_single_neuron`). A layer in which every input is connected to
every output is called fully connected, or dense. For a minibatch with
design matrix $\mathbf{X}$ it computes the predictions
$\mathbf{X}\mathbf{w} + b$ that the `forward` method of our scratch model
wrote out. The layer creates and stores its own parameters. We choose only
the number of outputs, here 1, and how the parameters are initialized. As in
the scratch model, the weights are drawn from a normal distribution with
mean 0 and standard deviation 0.01, and the bias starts at 0.

:begin_tab:`mxnet`
In Gluon the fully connected layer is `nn.Dense(1)`. Gluon takes the number
of inputs from the first input that the layer receives. `initialize`
records how the parameters will be initialized, here with
`init.Normal(sigma=0.01)` for the weights, and Gluon creates them with the
right shape at the first forward pass. The bias is initialized with zeros by
default.
:end_tab:

:begin_tab:`pytorch`
In PyTorch the fully connected layer is `nn.Linear(d, 1)`, which needs the
number $d$ of inputs, or `nn.LazyLinear(1)`, which takes $d$ from the first
input that the layer receives. We use the lazy version. In deeper networks
the number of inputs of a layer is the number of outputs of the layer before
it, and working it out by hand is tedious. A lazy layer has no parameters
until its first call, so we initialize them after that call, just before
training.
:end_tab:

:begin_tab:`tensorflow`
In Keras the fully connected layer is `tf.keras.layers.Dense(1)`. Keras
takes the number of inputs from the first input that the layer receives and
creates the weights then, drawing them from the `kernel_initializer`. The
bias starts at zero by default.
:end_tab:

:begin_tab:`jax`
In Flax NNX the fully connected layer is `nnx.Linear(d, 1)`. NNX has no lazy
layers, so we pass the number $d$ of inputs, and `rngs`, a source of random
keys from which the layer draws its initial weights. The parameters exist as
soon as the constructor returns. `kernel_init` sets the distribution of the
weights, and the bias starts at zero by default. If no `rngs` is given, the
model takes a key from `d2l.get_key()`, whose stream starts from a new random
seed on every run, as the random number generators of the other frameworks
do.
:end_tab:

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab pytorch
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)
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
the units of the label. Our loss and the framework loss have the
same minimizer, but the framework loss and its gradient are twice ours. This
factor of 2 returns when we choose the learning rate.

:begin_tab:`mxnet`
Gluon has no mean squared error loss. Its `L2Loss` computes
$\frac{1}{2}(\hat{y} - y)^2$ for each example, our scratch loss before the
average. To compute the same loss as the other frameworks, we double it and
average over the minibatch. `L2Loss` first reshapes the labels to the shape
of the predictions, so labels of shape $(32,)$ cannot be broadcast against
predictions of shape $(32, 1)$ into a $32 \times 32$ matrix of differences.
:end_tab:

:begin_tab:`pytorch`
`nn.MSELoss` computes the mean squared error, averaged over all entries of
its inputs, which here are the examples of the minibatch. If the labels and
the predictions differ in shape, for example $(32,)$ against $(32, 1)$, it
warns, and then computes the mean over the $32 \times 32$ matrix of
differences anyway; training proceeds on the wrong loss.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.losses.MeanSquaredError` computes the mean squared error over the
minibatch. Keras losses take the label first and the prediction second, and
they bring both to the same shape before subtracting, so labels of shape
$(32,)$ cannot be broadcast against predictions of shape $(32, 1)$ into a
$32 \times 32$ matrix of differences.
:end_tab:

:begin_tab:`jax`
In JAX a loss is an ordinary function of arrays. Optax's
`optax.squared_error` computes the elementwise squared errors, and we
average them over the minibatch. It raises an error if labels and
predictions differ in shape, so labels of shape $(32,)$ cannot be broadcast
against predictions of shape $(32, 1)$ into a $32 \times 32$ matrix of
differences.
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

The optimizer replaces our `SGD` class. It receives the parameters and the
learning rate $\eta$, and at each step it subtracts $\eta$ times the
gradient of the minibatch loss from each parameter, the update we wrote by
hand. The optimizer finds the parameters through the layers of the model,
so a model with more layers needs no change here, and replacing minibatch
SGD by another update rule changes only this one line.

:begin_tab:`mxnet`
In Gluon the optimizer is called `Trainer`. Despite its name it only applies
updates; our `Trainer` from :numref:`sec_oo-design` runs the training loop
that calls it. Gluon's `Trainer` takes the parameters, collected from every
layer by `self.collect_params()`, the name of the update rule, `'sgd'`, and
a dictionary of its hyperparameters. Its method `step(batch_size)` divides
the gradient by `batch_size` before the update. Our trainer calls `step(1)`,
since our loss is already averaged over the minibatch.
:end_tab:

:begin_tab:`pytorch`
`torch.optim.SGD` takes the parameters as `self.parameters()`, which
collects the parameters of every layer of the model, here the weight and
the bias of `net`. The module `torch.optim` provides the other common update
rules with the same interface.
:end_tab:

:begin_tab:`tensorflow`
`tf.keras.optimizers.SGD` takes the learning rate. The parameters arrive at
each step together with their gradients, through `apply_gradients`, which
our trainer calls with the model's `trainable_variables`. The module
`tf.keras.optimizers` provides the other common update rules with the same
interface.
:end_tab:

:begin_tab:`jax`
`optax.sgd` returns a *gradient transformation*: a pair of pure functions,
one that creates the optimizer's state and one that turns gradients into
parameter updates. Our trainer wraps it in an `nnx.Optimizer`, which applies
the updates to the parameters of the model (:numref:`sec_oo-design`). Optax
provides the other common update rules with the same interface.
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

We train on the synthetic data of
:numref:`sec_synthetic-regression-data`: 1,000 training and 1,000
validation examples whose two features are drawn from a standard normal
distribution and whose labels are
$y = \mathbf{x}^\top \mathbf{w}^* + b^* + \epsilon$, with
$\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, and noise $\epsilon$ of standard
deviation $\sigma = 0.01$. The data module serves the training examples in
minibatches of 32.

The learning rate needs one adjustment. Multiplying a loss by a constant $c$
multiplies its gradient by $c$, so SGD on the loss $cL$ with learning rate
$\eta$ takes the same steps as SGD on $L$ with learning rate $c\eta$
(:numref:`subsec_linear-regression-minibatch-sgd`). The framework loss is
twice our scratch loss, so the learning rate 0.015 reproduces the steps
that the scratch model took with 0.03. From the same initial parameters and
on the same minibatches, the two models compute the same updates. With 0.03
the concise model would take the steps of the scratch model at 0.06. On this
problem training would still converge, but it would no longer repeat the
computation of the scratch model.

:begin_tab:`pytorch`
Before training we call the model once on one example. This call creates
the parameters of the lazy layer, which we then initialize as in the
scratch model with functions from `nn.init`. Without these lines the layer
would keep PyTorch's default initialization, a uniform distribution whose
width shrinks with the number of inputs.
:end_tab:

We also put the trainer's plot, `trainer.board`, on a logarithmic scale, so
that the loss stays visible after it has fallen by several orders of
magnitude.

```{.python .input #linear-regression-concise-training-1}
%%tab pytorch
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(lr=0.015)
model(data.X[:1])  # Creates the parameters of the lazy layer
nn.init.normal_(model.net.weight, std=0.01)
nn.init.zeros_(model.net.bias)
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
This is the behavior of the scratch model with every loss doubled.

In :numref:`sec_linear_scratch` each step of the scratch model at
$\eta = 0.03$ multiplied the error of the parameters by $1 - \eta = 0.97$
on average, which compounds to a factor of about 7 per epoch in the loss.
The framework gradient is twice as large and the learning rate half as
large, so each step multiplies the error by the same factor,
$1 - 2\eta = 0.97$. The floor is
the loss of the true parameters, which leave the noise $\epsilon$
unexplained. For the mean squared error that loss is the noise variance
$\sigma^2 = 10^{-4}$, twice the floor $\sigma^2/2$ of the scratch loss.
The trainer keeps the numbers behind the plot in `trainer.history`, a
dictionary of per-epoch lists.

```{.python .input #linear-regression-concise-training-4}
print(f"validation loss: {trainer.history['val_loss'][-1]:.2e}")
```

The final validation loss lies within about 15% of $\sigma^2 = 10^{-4}$.

Since we generated the data, we can also compare the learned parameters with
$\mathbf{w}^*$ and $b^*$. The scratch model kept its parameters as `self.w`
and `self.b`. Here they belong to the layer, and `get_w_b` reads them out of
`net`.

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
are typically a few times $10^{-4}$ and rarely exceed $10^{-3}$, as for the
scratch model. Errors of this size are expected: even the exact
least-squares solution misses each parameter by about
$\sigma/\sqrt{n} = 0.01/\sqrt{1000} \approx 3 \times 10^{-4}$, the
standard error of an average of $n = 1{,}000$ noisy terms. And as in the
scratch model, the bias, which starts 4.2 below $b^*$, is still about
$4.2 \times 0.97^{320} \approx 2.5 \times 10^{-4}$ short of it after the
320 steps of ten epochs.

We can now assemble a model from framework parts and train it with the
loop of the previous sections. The only adjustment was the learning rate,
which we scaled with the loss. The rest of the book builds its models the
same way, from the layers, losses, and optimizers of the framework, and
writes a part by hand only to show how it works or when the framework does
not provide it. Here we judged the trained model by its distance from
parameters that we chose ourselves. With real data the true parameters are
unknown, and what matters is how well a model predicts examples it has not
seen. :numref:`sec_generalization_basics` takes up that question.

## Summary

Deep learning frameworks provide tested and fast versions of the parts we
wrote by hand for linear regression. A fully connected layer holds the
weights and the bias and computes the predictions, a loss function computes
the mean squared error, and an optimizer applies the SGD update to every
parameter of the model. Replacing each hand-written part by its framework
counterpart leaves `Module`, `DataModule`, and `Trainer` unchanged. The
framework loss omits the factor $\frac{1}{2}$ of our squared loss, so its
gradient is twice ours, and the learning rate 0.015 reproduces the steps of
the scratch model at 0.03. The trained model recovers $\mathbf{w}^*$ and
$b^*$ to within a few times $10^{-4}$, rarely more than $10^{-3}$, close to
the estimation error of least squares on 1,000 noisy examples, and its
validation loss
levels off at the noise variance $\sigma^2 = 10^{-4}$. Writing a part by
hand remains the way to understand it, and the way to build one that the
framework lacks.

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

1. [code] **Scratch versus concise.** Time `LinearRegressionScratch` and
   `LinearRegression` on the same `SyntheticRegressionData` for 10, 100,
   and 1,000 epochs, and report the time per epoch for each. Both run the
   same `fit_epoch`; identify which operations differ between the two runs
   and determine, by timing those operations in isolation, whether they
   account for the measured gap.

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
   so that the generating weight of that feature becomes $0.02$. The
   Hessian of `LinearRegression.loss` with respect to the weights is
   $\frac{2}{|\mathcal{B}|}\mathbf{X}^\top\mathbf{X}$ for a minibatch
   $\mathcal{B}$ with design matrix $\mathbf{X}$; the bias nearly
   decouples from the weights because the features have mean zero. On a
   quadratic loss with Hessian $\mathbf{H}$, a step of gradient descent
   multiplies the parameter error by $\mathbf{I} - \eta\mathbf{H}$,
   as exercise :exref:`ex_linear-regression-scratch-stability` shows, so training
   diverges once $\eta$ exceeds $2/\mu_{\max}$ for the largest
   eigenvalue $\mu_{\max}$ of $\mathbf{H}$.
    1. Estimate the eigenvalues of this Hessian before and after the
       rescaling.
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
  and `b`, with our own code.
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
- predict how fast the loss falls, and where it levels off;
- read the trained parameters out of a layer and judge their errors.

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
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| Synthetic Regression Data | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | SGD and the training loop by hand |
| **Concise Implementation** | **framework layers and losses** |
| Generalization | training vs. test error |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** `Module`, `Trainer`, the synthetic data, the scratch model.
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

::: {.slide title="The Framework Loss Is Twice Ours" except="mxnet"}
[Loss]{.kicker}

@linear-regression-concise-defining-the-loss-function

| | loss | gradient |
|---|---|---|
| ours | $\frac{1}{2\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $g$ |
| framework | $\frac{1}{\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $2g$ |

Same minimizer; every gradient doubles.
:::

::: {.slide title="The Framework Loss Is Twice Ours" only="mxnet"}
[Loss]{.kicker}

@linear-regression-concise-defining-the-loss-function

| | loss | gradient |
|---|---|---|
| ours | $\frac{1}{2\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $g$ |
| doubled `L2Loss` | $\frac{1}{\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $2g$ |

`L2Loss` includes the $\frac{1}{2}$; doubling it gives the mean squared
error of the other frameworks.
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

::: {.slide title="An Optimizer Object Replaces Our SGD Class"}
[Optimizer]{.kicker}

@linear-regression-concise-defining-the-optimization-algorithm

- Each step: $\theta \leftarrow \theta - \eta\, \nabla_\theta L$ for every
  parameter $\theta$ of every layer.
- Another update rule from the same module: change this one line.
:::

::: {.slide title="Halving η to 0.015 Reproduces the Scratch Steps"}
[Learning rate]{.kicker}

Multiplying the loss by $c$ multiplies the gradient by $c$:

$$\eta_\textrm{fw} \cdot 2g = \eta_\textrm{scratch} \cdot g \quad\Longleftrightarrow\quad \eta_\textrm{fw} = \frac{\eta_\textrm{scratch}}{2} = \frac{0.03}{2} = 0.015.$$

- Same initial parameters, same minibatches: the same updates.
- Features of variance 1: $\nabla L \approx 2(\theta - \hat{\theta})$,
  with $\hat{\theta}$ the minimizer, so each step multiplies the error
  $\theta - \hat{\theta}$ by $1 - 2\eta = 0.97$, the scratch model's
  $1 - 0.03$.
:::

::: {.slide title="Check: How Fast Should the Loss Fall?"}
[Check]{.kicker}

The loss exceeds its minimum by about the squared distance to the
minimizer. An epoch has 32 minibatches. By what factor does the
excess loss fall per epoch at $\eta = 0.015$? And at $\eta = 0.03$?

::: {.d2l-note .fragment .fade-up}
**Answer.** At $\eta = 0.015$: $(0.97^2)^{32} \approx 0.14$, about
$\div 7$ per epoch. At $\eta = 0.03$: $(0.94^2)^{32} \approx 0.02$, about
$\div 50$.
:::
:::

::: {.slide title="The Loss Falls About 7× per Epoch, Then Levels Off at σ² = 10⁻⁴"}
[Training]{.kicker}

::: {.cols .vc}
::: {.col}
Ten epochs at $\eta = 0.015$, log scale.

A straight line, a constant factor per epoch, then the floor: the noise
variance $\sigma^2$. The scratch curve with every loss doubled.
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
Errors of the size the scratch run left, a few times $10^{-4}$: at
$\eta = 0.015$ the concise model takes the scratch model's steps.
:::
:::

::: {.slide title="Check: Why Does This Loss Level Off Twice as High?"}
[Check]{.kicker}

The scratch run levels off near $5 \times 10^{-5}$, this run near
$10^{-4}$, with the same data and the same updates. Why the factor 2? And
what does the square root of this run's floor tell us about the
predictions?

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
| per step | distance $\times\, 0.97$ | distance $\times\, 0.97$ |
| loss floor | $\sigma^2/2 = 5 \times 10^{-5}$ | $\sigma^2 = 10^{-4}$ |

**Next:** how well does a trained model predict examples it has not seen?
:::
