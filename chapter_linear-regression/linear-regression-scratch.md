```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Linear Regression Implementation from Scratch
:label:`sec_linear_scratch`

The previous sections derived linear regression on paper, built classes for
models, data, and training, and generated data with a known answer. We now
turn the derivation into a program that trains the model. A training program
can be wrong without failing. Subtracting a label vector of shape $(32,)$
from predictions of shape $(32, 1)$, for example, raises no error, because
broadcasting forms the $32 \times 32$ matrix of all pairwise differences.
Training on the loss computed from it lowers that loss steadily while
learning the wrong weights. Such bugs show only when we compare the result
with what a correct program must produce.

We therefore write the four components of training ourselves, using nothing
but tensors and automatic differentiation: the linear model, the squared
loss, minibatch stochastic gradient descent, and the training loop that runs
them. Writing each component explicitly shows what it computes and what it
passes to the next one. We train on the synthetic data of
:numref:`sec_synthetic-regression-data`, whose labels were generated as
$\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ with
$\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, and Gaussian noise of standard
deviation $\sigma = 0.01$. Since we know how the data arose, we can predict
what a correct implementation must achieve and then check it. The loss must
fall to a floor set by the noise, and the learned parameters must come out
close to $\mathbf{w}^*$ and $b^*$. The training loop written here becomes the
loop of `d2l.Trainer` for the rest of the book; the next section replaces the
other components with their framework counterparts.

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np as mnp, npx
npx.set_np()
```

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=5}
%%tab jax
%matplotlib inline
import collections
from d2l import jax as d2l
from flax import nnx
import jax
from jax import numpy as jnp
import optax
```

## Model, Loss, and Gradient

### The Parameters

The model has two parameters, the weight vector $\mathbf{w} \in \mathbb{R}^2$
and the bias $b$. We initialize the weights with random numbers drawn from a
normal distribution with mean 0 and standard deviation 0.01, and the bias
with 0. For linear regression the starting point matters little: the loss is
a quadratic function of the parameters with a single minimum, which gradient
descent reaches from any start (exercise
:exref:`ex_linear-regression-scratch-initialization`). For deep networks the
scale of the initial weights matters a great deal, which is why the standard
deviation is an argument, `sigma`.

The parameters live in a subclass of `d2l.Module`, the base class of models
from :numref:`sec_oo-design`, so that the trainer can call the model's
methods. We register them as parameters of the module with the framework's
parameter type. Registration lets the framework's own tools find them: they
are listed, saved, and moved to a GPU together with the rest of the model.
It computes nothing for us; the loss, the gradient check, and the update
below are still written by hand.

:begin_tab:`pytorch`
`nn.Parameter` wraps a tensor and records the operations on it for
differentiation, as `requires_grad=True` did in :numref:`sec_autograd`.
Assigned as an attribute of a module, it is registered as a parameter of
that module, so `model.parameters()` and `model.state_dict()` contain `w`
and `b`.
:end_tab:

:begin_tab:`mxnet`
A Gluon `Parameter` assigned as an attribute of a block is registered with
it, so `model.collect_params()` lists `w` and `b`. Gluon allocates the memory
of a parameter, and of its gradient, when the parameter is initialized, here
from a normal distribution and with zeros. The array that holds the value of
a parameter is returned by `data()`, its gradient by `grad()`.
:end_tab:

:begin_tab:`tensorflow`
`add_weight` creates a variable, a tensor that can be changed in place, and
registers it with the Keras model, so `model.trainable_variables` and
`model.weights` list `w` and `b`. A `GradientTape` records the operations on
trainable variables, so that their gradients can be computed
(:numref:`sec_autograd`).
:end_tab:

:begin_tab:`jax`
`nnx.Param` marks an array as a trainable parameter of an NNX module, so
`nnx.state(model)` contains `w` and `b`, and `nnx.grad` differentiates with
respect to them. The argument `rngs` supplies the random key for the
initialization; without it, the model takes a key from the `d2l` library,
which draws a new random seed on every run, as the other frameworks do.
:end_tab:

```{.python .input #linear-regression-scratch-defining-the-model-1  n=6}
%%tab pytorch
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = nn.Parameter(d2l.normal(0, sigma, (num_inputs, 1)))
        self.b = nn.Parameter(d2l.zeros(1))
```

```{.python .input #linear-regression-scratch-defining-the-model-1  n=6}
%%tab mxnet
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.w = gluon.Parameter('w', shape=(num_inputs, 1),
                                 init=init.Normal(sigma))
        self.b = gluon.Parameter('b', shape=(1,), init=init.Zero())
        self.initialize()
```

```{.python .input #linear-regression-scratch-defining-the-model-1  n=6}
%%tab tensorflow
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        normal = tf.keras.initializers.RandomNormal(mean=0, stddev=sigma)
        self.w = self.add_weight(name='w', shape=(num_inputs, 1),
                                 initializer=normal)
        self.b = self.add_weight(name='b', shape=(1,), initializer='zeros')
```

```{.python .input #linear-regression-scratch-defining-the-model-1  n=7}
%%tab jax
class LinearRegressionScratch(d2l.Module):  #@save
    """The linear regression model implemented from scratch."""
    def __init__(self, num_inputs, lr, sigma=0.01, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.w = nnx.Param(
            rngs.params.normal((num_inputs, 1)) * sigma)
        self.b = nnx.Param(jnp.zeros(1))
```

### The Forward Pass

The forward pass computes the predictions for a minibatch,
$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$. For a minibatch of 32
examples, $\mathbf{X}$ is the $32 \times 2$ matrix whose rows are the
examples' features. Its product with the weights, stored as a $2 \times 1$
matrix, is a $32 \times 1$ column of weighted sums, and broadcasting adds the
scalar $b$ to each of its entries (:numref:`subsec_broadcasting`). The
decorator `add_to_class` of :numref:`sec_oo-design` adds this `forward`
method to the class defined above.

:begin_tab:`mxnet`
The MXNet version reads the values of the parameters with `data()`.
:end_tab:

```{.python .input #linear-regression-scratch-defining-the-model-2  n=8}
%%tab pytorch, tensorflow, jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w) + self.b
```

```{.python .input #linear-regression-scratch-defining-the-model-2}
%%tab mxnet
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    return d2l.matmul(X, self.w.data()) + self.b.data()
```

### The Loss

The loss of one example is half its squared error,
$\frac{1}{2}(\hat{y} - y)^2$, as in :numref:`sec_linear_regression`, and the
loss of a minibatch is the mean over its examples. Before subtracting, the
loss reshapes the labels into a column of shape $(|\mathcal{B}|, 1)$, the
shape of the predictions. Our data loader already provides that shape, and
the reshape rules out the bug of the opening; exercise
:exref:`ex_linear-regression-scratch-reshape-loss` works out what that bug
does to training.

```{.python .input #linear-regression-scratch-defining-the-loss-function  n=9}
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2
    return d2l.reduce_mean(l)
```

### The Gradient

Before we hand the loss to automatic differentiation, we write down the
gradient that it must return. :numref:`subsec_linear-regression-minibatch-sgd`
derived it with the chain rule: for one example with error
$r = \hat{y} - y$,

$$\frac{\partial l}{\partial \mathbf{w}} = (\hat{y} - y)\, \mathbf{x} \qquad \textrm{and} \qquad \frac{\partial l}{\partial b} = \hat{y} - y.$$
:eqlabel:`eq_linreg_grad_by_hand`

In words, *the gradient is the error-weighted input*: each weight receives
the error times its feature, and the bias the error itself. For a minibatch
$\mathcal{B}$ we collect the errors in the vector
$\mathbf{r} = \mathbf{X}\mathbf{w} + b - \mathbf{y}$. In this section $L$
denotes the loss of the minibatch, the average of the per-example losses
over $\mathcal{B}$ instead of over the whole training set. The averages of
the per-example gradients become matrix products,

$$\nabla_{\mathbf{w}} L = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top \mathbf{r} \qquad \textrm{and} \qquad \frac{\partial L}{\partial b} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} r^{(i)}.$$

One matrix--vector product computes the weight gradient of the whole
minibatch. It is the gradient $2\mathbf{X}^\top\mathbf{r}$ of the line
model in :numref:`sec_calculus`, divided by $2|\mathcal{B}|$, because our
loss is half the mean of the squared errors instead of their sum.

To check that automatic differentiation computes these gradients, we take
one minibatch of the synthetic data and a freshly initialized model.

```{.python .input #linear-regression-scratch-the-gradient-1}
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
X, y = next(iter(data.train_dataloader()))
model = LinearRegressionScratch(2, lr=0.03)
```

We compute the gradient with respect to $(\mathbf{w}, b)$ once by automatic
differentiation and once with the formulas above, and print the gradient
and the largest difference between the two.

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab pytorch
model.loss(model(X), y).backward()
with torch.no_grad():
    r = model(X) - y                         # One error per example
    by_hand = torch.cat([X.T @ r, r.sum(0, keepdim=True)]) / len(y)
    by_autograd = torch.cat([model.w.grad, model.b.grad[None]])
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(abs(by_hand - by_autograd).max()):.1g}')
```

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab mxnet
with autograd.record():
    loss = model.loss(model(X), y)
loss.backward()
r = model(X) - y                             # One error per example
by_hand = mnp.concatenate([X.T @ r, r.sum(0, keepdims=True)]) / len(y)
by_autograd = mnp.concatenate([model.w.data().grad,
                               model.b.data().grad[None]])
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(abs(by_hand - by_autograd).max()):.1g}')
```

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab tensorflow
with tf.GradientTape() as tape:
    loss = model.loss(model(X), y)
grad_w, grad_b = tape.gradient(loss, [model.w, model.b])
r = model(X) - y                             # One error per example
by_hand = tf.concat([tf.transpose(X) @ r,
                     tf.reduce_sum(r, 0, keepdims=True)], 0) / len(y)
by_autograd = tf.concat([grad_w, grad_b[None]], 0)
difference = tf.reduce_max(abs(by_hand - by_autograd))
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(difference):.1g}')
```

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab jax
grads = nnx.grad(lambda m: m.loss(m(X), y))(model)
r = model(X) - y                             # One error per example
by_hand = jnp.concatenate([X.T @ r, r.sum(0, keepdims=True)]) / len(y)
by_autograd = jnp.concatenate([grads.w[...], grads.b[...][None]])
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(abs(by_hand - by_autograd).max()):.1g}')
```

The largest difference is zero or at most about $10^{-6}$: the two
computations agree up to the rounding error of float32 arithmetic. This is
the gradient that the backward pass computes in every training step.

## The Optimizer and the Training Step

### The Update

Linear regression could be solved in closed form, with the normal equations
of :numref:`sec_linear_regression`. We train it with minibatch stochastic
gradient descent instead, because that is how every later model in the book
is trained, and here its result can be checked against the known parameters.
Each step draws a minibatch and moves the parameters against the gradient of
the minibatch loss $L$,

$$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla_{(\mathbf{w}, b)} L,$$

where the learning rate $\eta > 0$ sets the size of the step. Each
parameter moves against its own partial derivative, scaled by $\eta$. Since
the loss is a mean over the minibatch, $\eta$ needs no change when the
minibatch size changes; with a summed loss it would have to shrink in
proportion to $|\mathcal{B}|$
(:numref:`subsec_linear-regression-minibatch-sgd`).

:begin_tab:`pytorch`
Our `SGD` class has the interface of PyTorch's built-in optimizers. The
constructor receives the parameters and the learning rate, `step` updates the
parameters in place, and `zero_grad` resets their gradients to zero. The
class derives from `d2l.HyperParameters` (:numref:`sec_oo-design`) so that
`save_hyperparameters` stores `params` and `lr` as attributes.
:end_tab:

:begin_tab:`mxnet`
Our `SGD` class has the interface of Gluon's built-in `Trainer`. The
constructor receives the parameters and the learning rate, and `step` updates
the parameters in place. Like Gluon's `step`, it takes a batch size, which
Gluon uses to divide the gradient; our version ignores it, since our loss is
already a mean. The class derives from `d2l.HyperParameters`
(:numref:`sec_oo-design`) so that `save_hyperparameters` stores `params` and
`lr` as attributes.
:end_tab:

:begin_tab:`tensorflow`
Our `SGD` class has the interface of Keras optimizers: `apply_gradients`
receives pairs of a gradient and a variable, and updates each variable in
place with `assign_sub`. The class derives from `d2l.HyperParameters`
(:numref:`sec_oo-design`) so that `save_hyperparameters` stores `lr` as an
attribute.
:end_tab:

:begin_tab:`jax`
JAX arrays cannot be changed in place, so a JAX optimizer only computes the
updates, and a separate step applies them. Optax, JAX's optimization library,
describes an optimizer by two pure functions: `init` creates the optimizer
state, and `update` turns gradients into updates, which `nnx.Optimizer` then
adds to the model's parameters. Plain SGD keeps no state, and its update is
$-\eta$ times the gradient.
:end_tab:

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-1  n=11}
%%tab mxnet
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self, _):
        for param in self.params:
            param -= self.lr * param.grad
```

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-1  n=11}
%%tab pytorch
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()
```

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-1  n=12}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
```

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-1  n=13}
%%tab jax
class SGD(d2l.HyperParameters):  #@save
    """Minibatch stochastic gradient descent."""
    # An optax optimizer is a pair of pure functions: `init` creates the
    # optimizer state, and `update` turns gradients into parameter updates.
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        del params  # SGD keeps no state
        return optax.EmptyState()

    def update(self, updates, state, params=None):
        del params
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state
```

The trainer obtains the optimizer from the model's `configure_optimizers`
method (:numref:`sec_oo-design`), which returns an instance of this class.

:begin_tab:`mxnet`
The MXNet version hands the optimizer the arrays returned by `data()`. The
`grad` attribute of each of these arrays holds the parameter's gradient, so
`SGD` updates the parameters in place.
:end_tab:

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-2  n=14}
%%tab pytorch
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w, self.b], self.lr)
```

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-2  n=14}
%%tab tensorflow
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD(self.lr)
```

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-2  n=14}
%%tab jax
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD(self.lr)
```

```{.python .input #linear-regression-scratch-defining-the-optimization-algorithm-2  n=14}
%%tab mxnet
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    return SGD([self.w.data(), self.b.data()], self.lr)
```

### One Training Step

One step of minibatch SGD consists of four operations. It computes the
predictions and the loss of the minibatch while the framework records the
operations, clears any stored gradients, runs the backward pass to compute
the gradient of the loss, and updates the parameters. The order matters in two
places. The gradient used in the update must belong to the current minibatch
alone, so a gradient that the framework keeps from the previous step must be
cleared before the backward pass. And the update must not be recorded: it is
not part of the loss, and recording it would make the next backward pass
differentiate through every earlier update.

:begin_tab:`pytorch`
PyTorch adds the gradient of each backward pass to what `.grad` already holds
(:numref:`sec_autograd`), so the trainer calls `zero_grad` before `backward`.
PyTorch also refuses to change a parameter in place while it records, so the
update runs inside `torch.no_grad()`.
:end_tab:

:begin_tab:`mxnet`
MXNet records only inside `autograd.record()`, and `backward` overwrites the
stored gradient. Nothing has to be cleared, and the update, which runs
outside the recording block, is not recorded.
:end_tab:

:begin_tab:`tensorflow`
A `GradientTape` records only inside its `with` block, and it returns the
gradients as new tensors without storing them. Nothing has to be cleared, and
the update after the block is not recorded.
:end_tab:

:begin_tab:`jax`
`nnx.grad` returns the gradients as new arrays and stores nothing, so nothing
has to be cleared. The update is a separate function of the parameters and
the gradients, outside any differentiation.
:end_tab:

## The Training Loop

### Epochs and Averages

The training loop repeats the step, epoch after epoch. With 1,000 training
examples and minibatches of 32, an epoch, one pass through the training
data, takes 32 steps. After each epoch the trainer measures the loss on the
1,000 validation examples, on which the model never trains. The `Trainer` of
:numref:`sec_oo-design` leaves these two passes to the methods `fit_epoch`
and `validate`. Its `fit` method calls both once per epoch, keeps the epoch
means that they return in `trainer.history`, and plots them. We now write the
two methods, after `prepare_batch`, a hook that a later chapter overrides to
move each minibatch to a GPU; here it returns the minibatch unchanged.

```{.python .input #linear-regression-scratch-training-1  n=15}
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

`fit_epoch` and `validate` report the mean of each metric over all
examples. Reading a number from an accelerator makes Python wait until the
device has finished all the work queued for it (:numref:`sec_oo-design`), so
the steps return values that stay on the device. `Averager` adds up each
metric on the device and converts the sums into Python numbers once per
epoch, in `mean`. It weights each minibatch by its size, so the shorter last
minibatch counts correctly.

```{.python .input #linear-regression-scratch-training-5}
class Averager:  #@save
    """Example-weighted means of batch metrics, summed on the device."""
    def __init__(self):
        self.sums, self.n = {}, 0

    def add(self, metrics, n):
        for name, value in metrics.items():
            self.sums[name] = self.sums.get(name, 0) + value * n
        self.n += n

    def mean(self):
        """One host synchronization per metric."""
        return {name: float(s) / self.n for name, s in self.sums.items()}
```

### Training and Validation Passes

Both methods call the model's `training_step` or `validation_step`, which
`Module` provides (:numref:`sec_oo-design`) and which return the loss of the
minibatch. `d2l.as_metrics` turns a bare loss into the dictionary
`{'loss': loss}`, so that the trainer handles models that report one metric
and models that report several in the same way.

:begin_tab:`pytorch`
`fit_epoch` puts the model into training mode, runs the four operations of a
step for every minibatch, and adds the minibatch metrics to an `Averager`. Two
lines serve later chapters: gradient clipping, which limits the size of the
gradient, and the optional learning-rate schedule of :numref:`sec_oo-design`,
which changes $\eta$ after every step. The metrics are added after
`detach()`, since a running sum of losses that are still attached to the
computation graph would keep the graph of every minibatch in memory until the
end of the epoch. `validate` puts the model into evaluation mode and computes
the validation metrics without recording.
:end_tab:

:begin_tab:`mxnet`
`fit_epoch` records the forward pass inside `autograd.record()`, computes the
gradients, updates the parameters, and adds the minibatch metrics to an
`Averager`. We pass a batch size of 1 to `step`, because our loss is already
a mean. The line on gradient clipping, which limits the size of the
gradient, serves later chapters. `validate` computes the validation metrics;
outside `autograd.record()` nothing is recorded.
:end_tab:

:begin_tab:`tensorflow`
A function compiled into a graph with `tf.function` runs much faster than
the same operations issued one at a time from Python. `prepare_model`, which
replaces the one of :numref:`sec_oo-design`, therefore compiles the training
step (the model's `training_step`, the gradients, and the update) and the
validation step, once per call of `fit`. It first runs one forward pass,
because Keras creates a layer's variables only when the layer first sees
data. The compiled steps return their metrics as tensors, which `fit_epoch`
and `validate` average. Gradient clipping, which limits the size of the
gradient, serves later chapters, and the fallback to the variables that the
tape watched serves models whose variables Keras does not track.
:end_tab:

:begin_tab:`jax`
JAX differentiates and compiles pure functions, while an NNX module is an
object that owns its parameters. `nnx.grad` and `nnx.jit` connect the two:
they split the module into its fixed structure and its arrays, apply the JAX
transformation to a function of the arrays, and write the results back into
the module. We compile one training step (forward pass, loss, gradients, and
the update `optimizer.update(model, grads)`) into a single function,
`_trainer_train_step`, with a companion `_trainer_validation_step`. With
`has_aux=True`, `nnx.grad` differentiates the `'loss'` entry of the step's
metrics and also returns the whole dictionary. Both return the metrics as an
`OrderedDict`: `jit` hands back a plain dictionary with its keys sorted, which
would draw `acc` before `loss` and give the curves other colors than in the
other frameworks. Both functions are defined at
module level, and each is compiled once per minibatch shape in every call of
`fit`. They receive the two views of the model, for training and for
evaluation, that `prepare_model` creates in :numref:`sec_oo-design`.
:end_tab:

```{.python .input #linear-regression-scratch-training-2  n=16}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()
    metrics = d2l.Averager()
    for batch in map(self.prepare_batch, self.train_dataloader):
        out = d2l.as_metrics(self.model.training_step(batch))
        self.optim.zero_grad()
        out['loss'].backward()
        if self.gradient_clip_val > 0:  # To be discussed later
            self.clip_gradients(self.gradient_clip_val, self.model)
        with torch.no_grad():
            self.optim.step()
        if self.scheduler is not None:
            self.scheduler.step()
        metrics.add({k: v.detach() for k, v in out.items()}, len(batch[-1]))
    return metrics.mean()

@d2l.add_to_class(d2l.Trainer)  #@save
def validate(self):
    self.model.eval()
    metrics = d2l.Averager()
    with torch.no_grad():
        for batch in map(self.prepare_batch, self.val_dataloader):
            metrics.add(d2l.as_metrics(self.model.validation_step(batch)),
                        len(batch[-1]))
    return metrics.mean()
```

```{.python .input #linear-regression-scratch-training-2  n=17}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    metrics = d2l.Averager()
    for batch in map(self.prepare_batch, self.train_dataloader):
        with autograd.record():
            out = d2l.as_metrics(self.model.training_step(batch))
        out['loss'].backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        metrics.add(out, len(batch[-1]))
    return metrics.mean()

@d2l.add_to_class(d2l.Trainer)  #@save
def validate(self):
    metrics = d2l.Averager()
    for batch in map(self.prepare_batch, self.val_dataloader):
        metrics.add(d2l.as_metrics(self.model.validation_step(batch)),
                    len(batch[-1]))
    return metrics.mean()
```

```{.python .input #linear-regression-scratch-training-2  n=18}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    self.model = model
    batch = self.prepare_batch(next(iter(self.train_dataloader)))
    model(*batch[:-1], training=False)  # Keras creates the variables here
    self._train_step = tf.function(self.train_step, reduce_retracing=True)
    self._val_step = tf.function(self.val_step, reduce_retracing=True)

@d2l.add_to_class(d2l.Trainer)  #@save
def train_step(self, batch):
    with tf.GradientTape() as tape:
        out = d2l.as_metrics(self.model.training_step(batch))
    params = self.model.trainable_variables or tape.watched_variables()
    grads = tape.gradient(out['loss'], params)
    if self.gradient_clip_val > 0:
        grads = self.clip_gradients(self.gradient_clip_val, grads)
    self.optim.apply_gradients(zip(grads, params))
    return out

@d2l.add_to_class(d2l.Trainer)  #@save
def val_step(self, batch):
    return d2l.as_metrics(self.model.validation_step(batch))

@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    metrics = d2l.Averager()
    for batch in map(self.prepare_batch, self.train_dataloader):
        metrics.add(self._train_step(batch), len(batch[-1]))
    return metrics.mean()

@d2l.add_to_class(d2l.Trainer)  #@save
def validate(self):
    metrics = d2l.Averager()
    for batch in map(self.prepare_batch, self.val_dataloader):
        metrics.add(self._val_step(batch), len(batch[-1]))
    return metrics.mean()
```

```{.python .input #linear-regression-scratch-training-2  n=19}
%%tab jax
@nnx.jit  #@save
def _trainer_train_step(model, optimizer, batch):
    def loss_fn(model):
        out = collections.OrderedDict(
            d2l.as_metrics(model.training_step(batch)))
        return out['loss'], out
    grads, out = nnx.grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return out

@nnx.jit  #@save
def _trainer_validation_step(model, batch):
    return collections.OrderedDict(
        d2l.as_metrics(model.validation_step(batch)))

@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    metrics = d2l.Averager()
    for batch in map(self.prepare_batch, self.train_dataloader):
        metrics.add(_trainer_train_step(self.train_model, self.optim, batch),
                    len(batch[-1]))
    return metrics.mean()

@d2l.add_to_class(d2l.Trainer)  #@save
def validate(self):
    metrics = d2l.Averager()
    for batch in map(self.prepare_batch, self.val_dataloader):
        metrics.add(_trainer_validation_step(self.val_model, batch),
                    len(batch[-1]))
    return metrics.mean()
```

## Checking the Implementation

### The Expected Loss

Before training, we predict what a correct implementation must produce. The
label of a new example is $y = \mathbf{x}^\top \mathbf{w}^* + b^* + \epsilon$,
so the error of parameters $(\mathbf{w}, b)$ on it is
$r = \mathbf{x}^\top(\mathbf{w} - \mathbf{w}^*) + (b - b^*) - \epsilon$. The
two features of the synthetic data are independent standard normal
variables, and the noise $\epsilon$ is independent of them. All three have
mean zero, each feature has variance 1, and the product of any two of them
averages to zero. Squaring $r$ and taking the expectation therefore leaves
only the squared terms,

$$E\left[\tfrac{1}{2} r^2\right] = \tfrac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|^2 + \tfrac{1}{2}(b - b^*)^2 + \tfrac{1}{2}\sigma^2.$$
:eqlabel:`eq_linreg-scratch-expected-loss`

The expected loss is half the squared distance between the parameters and
the true ones, plus a floor of $\sigma^2/2 = 5 \times 10^{-5}$. No choice of
parameters gets below the floor, since even the true parameters leave the noise
$\epsilon$ unexplained. The validation loss, an average over 1,000 examples
that the model does not train on, estimates the expected loss.

The equation also predicts how fast training approaches the floor. We call
$(\mathbf{w} - \mathbf{w}^*, b - b^*)$ the *parameter error*; for the
least-squares solution it is the estimation error of
:numref:`subsec_synthetic-regression-data-recovery`. The gradient of the
expected loss with respect to $(\mathbf{w}, b)$ is the parameter error
itself, so a step along this gradient with learning rate $\eta$ multiplies
the parameter error by $1 - \eta$. The minibatch gradient equals this
gradient on average. At $\eta = 0.03$ the parameter error therefore shrinks
by a factor of about $0.97$ per step, and the loss above the floor, half its
square, by $0.97^2$. Over an epoch of 32 steps the excess loss shrinks by
$0.97^{64} \approx 0.14$, a factor of about 7. Training starts with weights
close to zero and a zero bias, about
$\frac{1}{2}(2^2 + 3.4^2 + 4.2^2) \approx 16.6$ above the floor, so after
seven epochs about $16.6 \times 0.14^7 \approx 2 \times 10^{-5}$
should remain above it.

### The Training Run

We train a fresh model for ten epochs with learning rate 0.03. The loss falls
by more than five orders of magnitude, so we plot it on a logarithmic axis,
on which a constant factor per epoch is a straight line.

```{.python .input #linear-regression-scratch-training-3  n=20}
model = LinearRegressionScratch(2, lr=0.03)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

The loss falls along a straight line, by a factor of about 7 per epoch, and
levels off near $5 \times 10^{-5}$ after seven or eight epochs, as predicted.
The board plots the mean training loss of each epoch in the middle of the
epoch, since it is averaged while the parameters change, and the validation
loss at the end of the epoch. The two curves lie on top of each other. With
three parameters and 1,000 examples, the model can fit almost none of the
noise in its training labels, so it does as well on new examples as on the
old ones. The numbers behind the curves are kept in `trainer.history`; for
example, `trainer.history['val_loss'][-1]` is the final validation loss.

### Recovered Parameters

Because we generated the data ourselves, we can also compare the learned
parameters with the true ones. We print the parameter errors
$\hat{\mathbf{w}} - \mathbf{w}^*$ and $\hat{b} - b^*$.

```{.python .input #linear-regression-scratch-training-4  n=21}
%%tab pytorch, tensorflow
w_error = d2l.numpy(d2l.reshape(model.w, data.w.shape) - data.w)
b_error = d2l.numpy(model.b - data.b)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

```{.python .input #linear-regression-scratch-training-4}
%%tab mxnet
w_error = d2l.numpy(d2l.reshape(model.w.data(), data.w.shape) - data.w)
b_error = d2l.numpy(model.b.data() - data.b)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

```{.python .input #linear-regression-scratch-training-4  n=23}
%%tab jax
w_error = d2l.numpy(d2l.reshape(model.w[...], data.w.shape) - data.w)
b_error = d2l.numpy(model.b[...] - data.b)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

The parameter errors are a few times $10^{-4}$ and rarely exceed $10^{-3}$.
They cannot be much smaller: the noise in the 1,000 training labels moves
even the exact least-squares fit about
$\sigma/\sqrt{n} = 0.01/\sqrt{1000} \approx 3 \times 10^{-4}$ away from
the true value in each coordinate, the standard error derived in
:numref:`subsec_synthetic-regression-data-recovery`. Minibatch SGD adds two
contributions of similar size. With a constant learning rate its parameters
never settle; they keep fluctuating in a small cloud around the minimizer,
the noise ball of :numref:`sec_linear_regression`. And ten epochs leave a
remainder of the initial parameter error: the 320 steps multiply it by
$0.97^{320} \approx 6 \times 10^{-5}$. The bias starts 4.2 below $b^*$, so
about $4.2 \times 0.97^{320} \approx 2.5 \times 10^{-4}$ of that distance
remains, which makes the error of the bias negative in about three runs out
of four.

The loss does not reveal errors of this size. By
:eqref:`eq_linreg-scratch-expected-loss`, a parameter error of
$5 \times 10^{-4}$ in each of the three coordinates raises the expected
loss by $\frac{1}{2} \cdot 3 \cdot (5 \times 10^{-4})^2 \approx 4 \times 10^{-7}$,
less than 1% of the floor and less than the fluctuation of a validation loss
over 1,000 examples. A comparison of parameters detects implementation errors
that a loss curve hides, which is why new code is best tested on synthetic
data whose parameters we know. Such a comparison is possible only when a
single set of parameters minimizes the loss, as here, where no feature is a
combination of the others. Deep networks have many parameter settings that
make the same predictions, so we judge them by their predictions on new
data.

### The Learning Rate

For steps along the gradient of the expected loss, the factor $1 - \eta$
also tells us what other learning rates do. For $0 < \eta < 1$ every step
shrinks the parameter error, the faster the larger $\eta$ is, and a step with
$\eta = 1$ lands on the minimum of the expected loss. For $1 < \eta < 2$ the
steps overshoot the minimum, but the parameter error still shrinks, since
$|1 - \eta| < 1$. For $\eta > 2$ every step multiplies the parameter error by
a number larger than 1 in magnitude, so it grows without bound. Minibatch
noise and the finite training set lower this limit a little (exercise
:exref:`ex_linear-regression-scratch-stability`). We train with four
learning rates, a factor of 10 apart, and print the final validation loss of
each run.

```{.python .input #linear-regression-scratch-the-learning-rate}
for lr in [0.003, 0.03, 0.3, 3]:
    trainer = d2l.Trainer(max_epochs=10)
    trainer.board.display = False
    trainer.fit(LinearRegressionScratch(2, lr=lr), data)
    val_loss = trainer.history['val_loss'][-1]
    print(f'lr = {lr:<6} validation loss: {val_loss:.1e}')
```

At $\eta = 0.003$ the parameter error shrinks by only $0.997$ per step, and
after ten epochs the loss is still about 2 to 3, as the prediction
$16.6 \times 0.997^{640} \approx 2.4$ says. At $\eta = 0.3$ the loss
reaches the floor within the first epoch. At $\eta = 3$ every step
multiplies the parameter error by about $-2$, and the loss overflows to
`inf` or `nan`.

We have written every component of training by hand and checked the result
against an answer known in advance. The loss reaches the noise floor at the
predicted rate, and the parameters agree with the true ones to the precision
that 1,000 noisy examples allow. Much of this code is generic, and the next
section replaces our parameters and forward pass, our loss, and our optimizer
with the framework's layers, losses, and optimizers. The check relied on
knowing the true parameters, which real data never provide. How to judge a
model by data it has not seen is the subject of
:numref:`sec_generalization_basics`.

## Summary

Training linear regression takes four components: parameters with a forward
pass, the squared loss, minibatch stochastic gradient descent, and a training
loop. The gradient of the squared loss is the error-weighted input, and
automatic differentiation reproduces it. A training step computes the loss,
clears old gradients, runs the backward pass, and updates the parameters
outside the recorded computation. The loop averages the minibatch metrics,
weighted by minibatch size, and reads them from the device once per epoch.
On synthetic data with standard normal features, the expected loss is half
the squared distance to the true parameters plus $\sigma^2/2$; it predicts
the outcome of training, and that learning rates above about 2 diverge. At
$\eta = 0.03$ the loss falls by a factor of about 7 per epoch to the floor
of $5 \times 10^{-5}$, and the learned parameters come within a few times
$10^{-4}$ of the true ones, rarely more than $10^{-3}$.

## Exercises

1. [code] **Initialization at the extremes.** `LinearRegressionScratch`
   draws its weights with standard deviation `sigma=0.01`. Predict whether
   training reaches the noise floor when the weights are initialized to
   exactly zero, and when `sigma=1000`. Use
   :eqref:`eq_linreg-scratch-expected-loss` to estimate how many more epochs
   the second case needs at learning rate 0.03, then check your estimate.
   :exlabel:`ex_linear-regression-scratch-initialization`

1. **Shapes in the loss.** Suppose the data loader yielded labels `y` of
   shape $(|\mathcal{B}|,)$ while the predictions `y_hat` have shape
   $(|\mathcal{B}|, 1)$, and `loss` computed `(y_hat - y) ** 2 / 2` without
   reshaping `y`.
    1. Determine whether an error is raised, and find the shape and the
       entries of `y_hat - y`.
    1. Show that this loss is smallest when the model predicts the mean of
       the minibatch labels for every example. Determine approximately the
       weights and the bias that training approaches on
       `SyntheticRegressionData`, and the value at which the loss levels
       off.
   :exlabel:`ex_linear-regression-scratch-reshape-loss`

1. [code] **Second derivatives.** Compute by hand the Hessian of the
   minibatch loss with respect to $(\mathbf{w}, b)$, and state which
   quantities it depends on. Then compute it with automatic
   differentiation, differentiating through the gradient as in
   :numref:`sec_autograd`, for one fixed minibatch of
   `SyntheticRegressionData`, once at initialization and once after
   training. Compare both results with your formula and with the curvature
   of :eqref:`eq_linreg-scratch-expected-loss`.

1. [code] **Stability and minibatch size.** ● The expected loss predicts
   that steps along its gradient diverge for $\eta > 2$. Let
   $\tilde{\mathbf{x}} = (\mathbf{x}, 1)$ be an input with a $1$ appended
   for the bias.
    1. For full-batch gradient descent, show that each step multiplies the
       difference between the parameters and the least-squares solution by
       the matrix $\mathbf{I} - \eta \mathbf{H}$, where $\mathbf{H}$ is the
       average of $\tilde{\mathbf{x}}\tilde{\mathbf{x}}^\top$ over the
       training set. Compute the eigenvalues of $\mathbf{H}$ and predict the
       largest stable learning rate.
    1. Train with minibatches of 1,000, 32, and 1 example, giving each run the
       same number of updates, say 320. For each minibatch size, find the
       largest learning rate in a grid of your choice at which the final
       training loss is below $10^{-3}$.
    1. Show that a step on a single example multiplies the component of the
       parameter error along $\tilde{\mathbf{x}}$ by
       $1 - \eta \|\tilde{\mathbf{x}}\|^2$, and use
       $E\|\tilde{\mathbf{x}}\|^2 = 3$ to explain the limit you found for
       minibatches of one example.
   :exlabel:`ex_linear-regression-scratch-stability`

1. **Why reshuffle.** The training loader draws a fresh random order at
   every epoch. Explain which property of the minibatch gradient this
   preserves. Then construct a fixed ordering of the training set, such as
   sorting by label, under which minibatch SGD without reshuffling behaves
   differently, and describe the trajectory of $(\mathbf{w}, b)$ within an
   epoch and across epochs.
   :exlabel:`ex_linear-regression-scratch-why-reshuffle`

1. [code] **Robust losses.** ● Replace `loss` in `LinearRegressionScratch`
   with the mean absolute error
   `d2l.reduce_mean(d2l.abs(y_hat - d2l.reshape(y, (-1, 1))))`.
    1. Train on `SyntheticRegressionData` and compare the recovered
       parameters and the loss curve with those of the squared loss.
    1. Set one training label to $y^{(5)} = 10{,}000$, the corruption used in
       :numref:`subsec_linear-regression-loss-function`, and train with
       both losses. Record $b$ after every minibatch update, plot it
       against the iteration for both runs, and explain the pattern of the
       squared-loss curve, including what determines its final value.
    1. On the corrupted data, replace `loss` with the minibatch average of
       the Huber loss

        $$l_\delta(y, y') = \begin{cases}\frac{1}{2}(y-y')^2 & \textrm{ if } |y-y'| \leq \delta \\ \delta \left(|y-y'| - \frac{\delta}{2}\right) & \textrm{ otherwise,}\end{cases}$$

        whose quadratic part matches the squared loss of this section.
        Train for $\delta \in \{1, 10, 10^2, 10^3, 10^4\}$, and plot the
        distance $|\hat{b} - b^*|$ of the learned bias from the generating
        bias $b^* = 4.2$ against $\delta$ on logarithmic axes. Explain the
        shape of the curve from the derivative of $l_\delta$ with respect
        to $y'$, and relate the result to the penalty curves
        of :numref:`fig_linreg-loss-menu`.
   :exlabel:`ex_linear-regression-scratch-robust-losses`

1. **Uncleared gradients.** ● Suppose the gradient buffers accumulate and
   are never cleared, so that the update at step $t$ subtracts
   $\eta(\mathbf{g}_1 + \cdots + \mathbf{g}_t)$, the sum of all minibatch
   gradients so far. For the one-dimensional loss $\frac{h}{2}(w - c)^2$
   with $h > 0$, whose minimum lies at $w = c$, and with exact gradients,
   derive a recurrence for the error $e_t = w_t - c$, and determine for
   which values of $\eta h$ the error converges to zero. State what this
   predicts for minibatch SGD on `LinearRegressionScratch` under the same
   accumulation.

1. [code] **Planck's law.** ● Use
   [Planck's law](https://en.wikipedia.org/wiki/Planck%27s_law) to
   determine the temperature of a black body from measurements of its
   spectral radiance at several wavelengths. The spectral radiance is

    $$B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1},$$

    where $\lambda$ is the wavelength, $T$ is the temperature, $c$ is the
    speed of light, $h$ is Planck's constant, and $k$ is the Boltzmann
    constant. Generate measurements at a known $T$, then write a
    `d2l.Module` subclass whose `forward` computes $B(\lambda, T)$ with $T$
    as its only parameter and fit it with `d2l.Trainer`. Choose units in
    which $B$ and $T$ are of order one before fitting.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/201)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17976)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.4]{.kicker}

Linear regression with tensors and autograd<br>
**parameters · the gradient by hand · minibatch SGD · the training loop · a known answer**
:::
:::

::: {.slide title="A Wrong Training Loop Can Still Lower the Loss"}
[The problem]{.kicker}

- Labels of shape $(32,)$ minus predictions of shape $(32, 1)$: no error.
  Broadcasting forms a $32 \times 32$ matrix of differences.
- Ten epochs on that loss lower it steadily, from about 17 to about 8,
  while $\mathbf{w}$ ends near $(0.06, -0.1)$ instead of $(2, -3.4)$.
- Nothing in the loss curve says that the program is wrong.

::: {.d2l-note}
Train on data with a **known answer**:
$\mathbf{y} = \mathbf{X}\mathbf{w}^* + b^* + \boldsymbol{\epsilon}$ with
$\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, noise $\sigma = 0.01$. A correct
program must reach the loss floor $\sigma^2/2 = 5 \times 10^{-5}$ and recover
$\mathbf{w}^*$ and $b^*$.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write parameters, forward pass, squared loss, and SGD with tensors and
  autograd;
- derive the gradient by hand and check it against autograd;
- put the four operations of a training step in the right order;
- predict the loss curve and the parameter error before training;
- say which learning rates converge and which diverge.

**Running example:** $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$,
$\sigma = 0.01$; 1,000 training examples in minibatches of 32.
:::

::: {.slide title="Every Part of Training, Written by Hand"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| Synthetic Regression Data | data with known $\mathbf{w}^*, b^*$ |
| **Implementation from Scratch** | **SGD and the training loop by hand** |
| Concise Implementation | framework layers and losses |
| Generalization | training vs. test error |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** the squared loss and the SGD update, autograd, the `Trainer`
  classes, the synthetic data.
- **Skips:** framework layers, losses, and optimizers (the next deck), and
  how to judge a model on unseen data.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The model and its gradient]{.dtitle}

[parameters · forward pass · loss · the gradient by hand]{.dsub}
:::
:::

::: {.slide title="Parameters: Small Random Weights and a Zero Bias"}
[The model]{.kicker}

Entries of $\mathbf{w}$ drawn from $\mathcal{N}(0, 0.01^2)$, $b = 0$, both
registered with the module:

| framework | parameter type | listed by |
|---|---|---|
| PyTorch | `nn.Parameter(tensor)` | `model.parameters()` |
| TensorFlow | `self.add_weight(...)` | `model.trainable_variables` |
| MXNet | `gluon.Parameter(...)` | `model.collect_params()` |
| JAX | `nnx.Param(array)` | `nnx.state(model)` |

::: {.d2l-note}
Registered parameters are recorded for differentiation, saved, and moved to
a GPU with the model. Any start works here: the loss is a quadratic with a
single minimum.
:::
:::

::: {.slide title="A Prediction Is One (32×2)(2×1) Product; the Loss a Mean"}
[The model]{.kicker}

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b, \qquad L = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \tfrac{1}{2}\bigl(\hat{y}^{(i)} - y^{(i)}\bigr)^2$$

@linear-regression-scratch-defining-the-model-2

@linear-regression-scratch-defining-the-loss-function

::: {.d2l-note}
Broadcasting adds $b$ to all 32 predictions. The loss first reshapes
$\mathbf{y}$ into a $(32, 1)$ column: a $(32,)$ vector would broadcast to
$32 \times 32$.
:::
:::

::: {.slide title="The Gradient Is the Error-Weighted Input"}
[The gradient]{.kicker}

One example, error $r = \hat{y} - y$:

$$\frac{\partial l}{\partial \mathbf{w}} = r\, \mathbf{x}, \qquad \frac{\partial l}{\partial b} = r$$

. . .

A minibatch, errors $\mathbf{r} = \mathbf{X}\mathbf{w} + b - \mathbf{y}$:

$$\nabla_{\mathbf{w}} L = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top \mathbf{r}, \qquad \frac{\partial L}{\partial b} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} r^{(i)}$$

::: {.d2l-note .rule}
One matrix–vector product per minibatch: the line model's
$2\mathbf{X}^\top\mathbf{r}$, divided by $2|\mathcal{B}|$.
:::
:::

::: {.slide title="Autograd Returns the Gradient Derived by Hand"}
[The gradient]{.kicker}

One minibatch of 32 examples, a fresh model:

@linear-regression-scratch-the-gradient-2

The largest difference is zero or at most about $10^{-6}$: float32 rounding.
:::

::: {.slide title="Check: The Gradient at w = 0, b = 0"}
[Check]{.kicker}

Start from $\mathbf{w} = \mathbf{0}$, $b = 0$. Averaged over many examples,
what is the gradient, and where does its negative point? The features are
independent standard normal.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The errors are $r = -y$, so the gradient is
$E[\mathbf{x}\, r] = -E[\mathbf{x}\mathbf{x}^\top]\mathbf{w}^* = -\mathbf{w}^* = (-2, 3.4)$
for $\mathbf{w}$ and $E[r] = -b^* = -4.2$ for $b$. The negative gradient
points straight at the true parameters.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Minibatch SGD]{.dtitle}

[the update · one step, in order]{.dsub}
:::
:::

::: {.slide title="SGD Subtracts η Times the Gradient, in Place" only="pytorch"}
[The optimizer]{.kicker}

$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla_{(\mathbf{w}, b)} L$,
with learning rate $\eta$. The two methods of our `SGD` class:

```{.python #linear-regression-scratch-exercises-6}
def step(self):                      # runs inside torch.no_grad()
    for param in self.params:
        param -= self.lr * param.grad

def zero_grad(self):
    for param in self.params:
        if param.grad is not None:
            param.grad.zero_()
```
:::

::: {.slide title="SGD as an Optax Transformation: Updates −η·g" only="jax"}
[The optimizer · JAX]{.kicker}

JAX arrays are immutable. An optax optimizer is two pure functions: `init`
makes the state, `update` maps gradients to updates, and `nnx.Optimizer`
adds them to the parameters. The two methods of our `SGD` class:

```{.python #linear-regression-scratch-exercises-5}
def init(self, params):                  # SGD keeps no state
    return optax.EmptyState()

def update(self, updates, state, params=None):
    updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
    return updates, state
```
:::

::: {.slide title="Four Operations per Step, in a Fixed Order" except="mxnet,tensorflow"}
[One step]{.kicker}

1. **Forward and loss**, recorded for differentiation.
2. **Clear** gradients kept from the previous step.
3. **Backward**: the gradient of this minibatch's loss.
4. **Update** the parameters, *not* recorded.

| framework | old gradients | unrecorded update |
|---|---|---|
| PyTorch | added to `.grad`: `zero_grad()` | `torch.no_grad()` |
| MXNet | overwritten by `backward` | outside `autograd.record()` |
| TensorFlow | returned fresh by the tape | outside the tape |
| JAX | returned fresh by `nnx.grad` | separate function |
:::

::: {.slide title="Four Operations per Step, in a Fixed Order" only="mxnet"}
[One step]{.kicker}

::: {.cols}
::: {.col}
1. **Forward and loss**, recorded for differentiation.
2. **Clear** gradients kept from the previous step.
3. **Backward**: the gradient of this minibatch's loss.
4. **Update** the parameters, *not* recorded.
:::

::: {.col}
Step 4, $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla L$,
in our `SGD` class:

```{.python #linear-regression-scratch-exercises-7}
def step(self, _):  # batch size ignored
    for param in self.params:  # data() arrays
        param -= self.lr * param.grad
```
:::
:::

| framework | old gradients | unrecorded update |
|---|---|---|
| PyTorch | added to `.grad`: `zero_grad()` | `torch.no_grad()` |
| MXNet | overwritten by `backward` | outside `autograd.record()` |
| TensorFlow | returned fresh by the tape | outside the tape |
| JAX | returned fresh by `nnx.grad` | separate function |
:::

::: {.slide title="Four Operations per Step, in a Fixed Order" only="tensorflow"}
[One step]{.kicker}

::: {.cols}
::: {.col}
1. **Forward and loss**, recorded for differentiation.
2. **Clear** gradients kept from the previous step.
3. **Backward**: the gradient of this minibatch's loss.
4. **Update** the parameters, *not* recorded.
:::

::: {.col}
Step 4, $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla L$,
in our `SGD` class:

```{.python #linear-regression-scratch-exercises-8}
def apply_gradients(self, grads_and_vars):
    for grad, param in grads_and_vars:
        param.assign_sub(self.lr * grad)
```
:::
:::

| framework | old gradients | unrecorded update |
|---|---|---|
| PyTorch | added to `.grad`: `zero_grad()` | `torch.no_grad()` |
| MXNet | overwritten by `backward` | outside `autograd.record()` |
| TensorFlow | returned fresh by the tape | outside the tape |
| JAX | returned fresh by `nnx.grad` | separate function |
:::

::: {.slide title="Check: Forgetting to Clear the Gradients"}
[Check]{.kicker}

Suppose `backward` added each new gradient to the stored one, as PyTorch's
does, and nothing ever cleared it. What does step $t$ subtract, and what
happens to the parameters?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta(\mathbf{g}_1 + \cdots + \mathbf{g}_t)$: every earlier
gradient is applied again at every step. The parameters run past the
minimum, and on a quadratic they oscillate around it without settling.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[The training loop]{.dtitle}

[epochs · weighted means · one sync per epoch]{.dsub}
:::
:::

::: {.slide title="An Epoch Mean Weighs Every Example Equally"}
[The loop]{.kicker}

The two methods of `Averager`:

```{.python #linear-regression-scratch-exercises-9}
def add(self, metrics, n):
    for name, value in metrics.items():
        self.sums[name] = self.sums.get(name, 0) + value * n
    self.n += n

def mean(self):                      # one host sync per metric
    return {k: float(s) / self.n for k, s in self.sums.items()}
```

::: {.d2l-note}
Each minibatch counts with its size, so the shorter last one counts
correctly. The sums stay on the device; `mean` reads them once per epoch.
:::
:::

::: {.slide title="fit_epoch Runs the Four Operations on Every Minibatch" only="pytorch"}
[The loop · simplified]{.kicker}

```{.python #linear-regression-scratch-exercises-1}
for batch in self.train_dataloader:
    loss = self.model.training_step(batch)   # 1. forward, loss
    self.optim.zero_grad()                   # 2. clear
    loss.backward()                          # 3. backward
    with torch.no_grad():
        self.optim.step()                    # 4. update, unrecorded
    metrics.add({'loss': loss.detach()}, len(batch[-1]))
return metrics.mean()                        # one sync per epoch
```

`validate` runs the same loop without steps 2–4, under `torch.no_grad()`.
:::

::: {.slide title="fit_epoch Runs the Four Operations on Every Minibatch" only="mxnet"}
[The loop · simplified]{.kicker}

```{.python #linear-regression-scratch-exercises-2}
for batch in self.train_dataloader:
    with autograd.record():
        loss = self.model.training_step(batch)  # 1. forward, loss
    loss.backward()                             # 2.+3. overwrite
    self.optim.step(1)                          # 4. update
    metrics.add({'loss': loss}, len(batch[-1]))
return metrics.mean()                           # one sync per epoch
```

`validate` runs the same loop without recording or updates.
:::

::: {.slide title="One Compiled Function Runs the Four Operations" only="tensorflow"}
[The loop · simplified]{.kicker}

```{.python #linear-regression-scratch-exercises-3}
@tf.function                                     # compiled once
def train_step(batch):
    with tf.GradientTape() as tape:
        loss = model.training_step(batch)        # 1. forward, loss
    grads = tape.gradient(loss, params)          # 2.+3. fresh grads
    optim.apply_gradients(zip(grads, params))    # 4. update
    return loss
```

`fit_epoch` calls it on every minibatch and averages the returned losses.
:::

::: {.slide title="One Compiled Function Runs the Four Operations" only="jax"}
[The loop · simplified]{.kicker}

```{.python #linear-regression-scratch-exercises-4}
@nnx.jit                                         # compiled once per shape
def train_step(model, optimizer, batch):
    loss, grads = nnx.value_and_grad(            # 1.-3. fresh grads
        lambda m: m.training_step(batch))(model)
    optimizer.update(model, grads)               # 4. update
    return loss
```

`fit_epoch` calls it on every minibatch and averages the returned losses.
:::

::: {.slide}
::: {.divider}
[04]{.dnum}

[Checking the result]{.dtitle}

[a prediction · the run · the parameters · the learning rate]{.dsub}
:::
:::

::: {.slide title="The Expected Loss Is Half the Squared Parameter Error Plus σ²/2"}
[A prediction]{.kicker}

Standard normal features, independent of the noise:

$$E\bigl[\tfrac{1}{2} r^2\bigr] = \tfrac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|^2 + \tfrac{1}{2}(b - b^*)^2 + \tfrac{1}{2}\sigma^2$$

- Floor: $\sigma^2/2 = 5 \times 10^{-5}$; nothing gets below it.
- Gradient $= (\mathbf{w} - \mathbf{w}^*, b - b^*)$: a step along it
  multiplies the parameter error by $1 - \eta = 0.97$.
- Excess loss per epoch of 32 steps: $0.97^{64} \approx 0.14$, a factor
  of 7. Start: $\tfrac{1}{2}(2^2 + 3.4^2 + 4.2^2) \approx 16.6$.
:::

::: {.slide title="The Loss Falls 7× per Epoch to the Floor of 5×10⁻⁵"}
[The run]{.kicker}

::: {.cols .vc}
::: {.col}
Ten epochs at $\eta = 0.03$, log scale.

A straight line, then the floor after seven or eight epochs, as predicted.
Training and validation coincide.
:::

::: {.col .fig .big}
@!linear-regression-scratch-training-3
:::
:::
:::

::: {.slide title="The Parameters Come Within About 10⁻³ of (2, −3.4) and 4.2"}
[The parameters]{.kicker}

Parameter errors $\hat{\mathbf{w}} - \mathbf{w}^*$ and $\hat{b} - b^*$:

@!linear-regression-scratch-training-4

- Even exact least squares misses by $\sigma/\sqrt{n} = 0.01/\sqrt{1000} \approx 3 \times 10^{-4}$.
- The loss cannot see this: $\tfrac{1}{2} \cdot 3 \cdot (5 \times 10^{-4})^2 \approx 4 \times 10^{-7}$,
  under 1% of the floor.

::: {.d2l-note .rule}
Comparing parameters catches bugs that a loss curve hides.
:::
:::

::: {.slide title="η = 0.003 Crawls, η = 3 Diverges"}
[The learning rate]{.kicker}

Along the averaged gradient, the parameter error is multiplied by
$1 - \eta$ per step: it shrinks for $0 < \eta < 2$ and grows for
$\eta > 2$. With minibatches the limit is somewhat lower.

@linear-regression-scratch-the-learning-rate

::: {.d2l-note}
At $\eta = 3$ the parameter error is multiplied by about $-2$ per step, and
the loss overflows.
:::
:::

::: {.slide title="Check: Ten Times More Noise"}
[Check]{.kicker}

Generate the data with $\sigma = 0.1$ instead of $0.01$ and train the same
way. Where does the loss level off, how large are the parameter errors, and
how fast does the loss fall?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The floor $\sigma^2/2$ rises 100-fold, to $5 \times 10^{-3}$.
The parameter errors grow 10-fold, to about
$\sigma/\sqrt{n} \approx 3 \times 10^{-3}$.
The rate, a factor of 7 per epoch, is unchanged: the noise does not change
the gradient of the expected loss.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Model:** $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$; loss: the mean of
  $\tfrac{1}{2}(\hat{y} - y)^2$.
- **Gradient:** the error-weighted input, $\frac{1}{|\mathcal{B}|}\mathbf{X}^\top\mathbf{r}$;
  autograd agrees up to float32 rounding.
- **Step:** forward, clear, backward, unrecorded update; epoch means
  weighted by minibatch size, read once per epoch.
- **Check:** loss $16.6 \to 5 \times 10^{-5}$ at 7× per epoch; parameter
  errors a few times $10^{-4}$, rarely more than $10^{-3}$.
- **Learning rate:** the averaged gradient converges for $0 < \eta < 2$;
  with minibatches the limit is somewhat lower; $\eta = 3$ diverges.

**Next:** the same model with the framework's layers, losses, and
optimizers.
:::
