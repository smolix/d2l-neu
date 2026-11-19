```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Object-Oriented Design for Implementation
:label:`sec_oo-design`

In :numref:`sec_linear_scratch`, we trained linear regression from scratch with
a dozen-line `train` function. That simplicity came at a cost: `train` was wired
directly to one model and one dataset, hard-coding `linreg`, `squared_loss`, `w`,
`b`, `X_train`, and `X_val` right inside its body. Any new architecture or
dataset would force us to copy the entire loop and tweak a handful of lines. Yet
strip away those specific names, and the underlying skeleton never changes:

```text
for epoch in range(num_epochs):
    for X, y in train_data:                     # data
        loss = loss_fn(net(X), y)               # model and loss
        update(params, gradient(loss, params))  # optimizer
    val_loss = mean_loss(net, val_data)         # model on held-out data
    record(epoch, val_loss)
```

Only the commented lines vary from one problem to the next; the scaffolding around
them is universal. Across this book, nearly thirty sections train models—most of
them across multiple frameworks—which would mean maintaining close to a hundred
nearly identical copies of this loop. Adding a single capability such as gradient
clipping or learning-rate scheduling would require editing every copy, and they
would inevitably drift out of sync.

The standard engineering remedy is object-oriented modularity, inspired by
open-source libraries such as [Lightning](https://lightning.ai/). We write the
training loop *once* and cleanly separate the moving parts into three core
abstractions (:numref:`fig_oo_design`):

* **`Module`** encapsulates a model: its learnable parameters, forward pass
  (`forward`), loss function, and optimizer (`configure_optimizers`).
* **`DataModule`** encapsulates a dataset and serves it as reusable training and
  validation minibatch loaders.
* **`Trainer`** runs the training and validation loops, orchestrates gradient
  updates, and logs epoch metrics into a history dictionary.

In terms of our pseudocode above, `DataModule` supplies `train_data` and
`val_data`, `Module` supplies `net`, `loss_fn`, and `update`, and `Trainer`
drives the nested loops and records the losses. From this point on, every model
in the book subclasses `Module`, every dataset subclasses `DataModule`, and we
only revisit `Trainer` when introducing new hardware or optimization machinery
such as GPUs and gradient clipping.

![The three base classes. The `Trainer` takes minibatches from a `DataModule` and passes each to a `Module`, whose `training_step` returns the loss; the trainer updates the parameters and keeps the epoch means of the losses in its `history`. New models and datasets are written as subclasses of `Module` and `DataModule`.](../img/mdl-linreg-oo-classes.svg)
:label:`fig_oo_design`

In this section, we refactor the from-scratch linear regression code of
:numref:`sec_linear_scratch` into these three classes and re-run training to
verify that we recover the exact same results. A few arguments and hooks exist
solely for upcoming chapters; we flag them whenever they appear.

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #oo-design-object-oriented-design-for-implementation-1}
import collections
```

## Utilities
:label:`oo-design-utilities`

Before defining our three base classes, we introduce two small Python utilities
designed specifically to keep notebook code readable. Neither is specific to
machine learning.

First, standard Python expects an entire class definition to live inside one
contiguous block of code, whereas explaining a complex class in a notebook works
far better when each method sits in its own short cell next to the prose
describing it. The decorator `add_to_class` dynamically attaches a function as a
method on an already-defined class—even *after* instances of that class have been
created! (Think of this strictly as pedagogical notebook scaffolding; in regular
Python source files, always define a class in one place.)

```{.python .input #oo-design-utilities-1}
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        return obj
    return wrapper
```

Let us try it out on a toy class `A` and an already-instantiated object `a`:

```{.python .input #oo-design-utilities-2}
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Now we attach a new method `do` in a separate cell. Because Python resolves
methods on the class at call time, our existing instance `a` immediately gains
access to `do`:

```{.python .input #oo-design-utilities-3}
@add_to_class(A)
def do(self):
    print('Instance attribute "b" is', self.b)

a.do()
```

Our second helper eliminates boilerplate in constructors, which otherwise fill up
with repetitive assignments like `self.lr = lr` and `self.batch_size = batch_size`.
Because most constructor arguments in deep learning are hyperparameters, we
inherit from `d2l.HyperParameters` and call `self.save_hyperparameters()`, which
automatically binds every constructor argument (except any listed in `ignore`) as
an attribute on `self`. Under the hood it inspects the calling stack frame using
Python's `inspect` module; its implementation lives in :numref:`sec_utils`.

```{.python .input #oo-design-utilities-5}
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('self.c exists:', hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

Every class we build in the remainder of this section inherits from
`HyperParameters`.

## Models
:label:`subsec_oo-design-models`

Every model in this book subclasses `Module`. A subclass initializes its
parameters in `__init__` and implements three core methods: `forward` (which maps
an input minibatch to predictions—by default delegating to `self.net` if present),
`loss` (which compares predictions against labels), and `configure_optimizers`
(which constructs and returns the optimizer). In return, `Module` supplies
`training_step` and `validation_step` for the trainer to call. Alongside
`HyperParameters`, `Module` inherits from each framework's native neural network
base class:

:begin_tab:`mxnet`
In MXNet, that base class is Gluon's `nn.Block`. Calling `model(X)` invokes
`forward`, and the block automatically tracks any parameters stored as attributes
(:numref:`sec_model_construction`).
:end_tab:

:begin_tab:`pytorch`
In PyTorch, that base class is `nn.Module`. Calling `model(X)` invokes `forward`,
and the module automatically tracks all registered parameters and submodules
stored as attributes (:numref:`sec_model_construction`).
:end_tab:

:begin_tab:`tensorflow`
In TensorFlow, that base class is `tf.keras.Model`, which invokes `call` when an
instance is called. Our `call` records the `training` flag (needed later for
layers like dropout and batch normalization) and forwards to `forward` so our
subclasses look uniform across frameworks. Notice also that `__init__` clears the
default `loss = None` attribute set by Keras 3 so it does not shadow our `loss`
method.
:end_tab:

:begin_tab:`jax`
In JAX, that base class is Flax's `nnx.Module`, a standard Python object whose
attributes hold its parameters directly where `nnx.jit` and `nnx.grad` can find
them. We wire `__call__` to `forward` for consistency with the other frameworks.
:end_tab:

```{.python .input #oo-design-models}
%%tab pytorch
class Module(nn.Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def training_step(self, batch):
        """The mean loss on one minibatch, or a dict of batch-mean metrics
        with key 'loss'."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def validation_step(self, batch):
        """The mean loss on one minibatch of validation data, or a dict of
        batch-mean metrics with key 'loss'."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input #oo-design-models}
%%tab mxnet
class Module(nn.Block, d2l.HyperParameters):  #@save
    """The base class of models."""
    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def training_step(self, batch):
        """The mean loss on one minibatch, or a dict of batch-mean metrics
        with key 'loss'."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def validation_step(self, batch):
        """The mean loss on one minibatch of validation data, or a dict of
        batch-mean metrics with key 'loss'."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input #oo-design-models}
%%tab tensorflow
class Module(tf.keras.Model, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self):
        super().__init__()
        self.training = None
        self.__dict__.pop('loss', None)

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def call(self, X, *args, training=None, **kwargs):
        if training is not None:
            self.training = training
        return self.forward(X, *args, **kwargs)

    def training_step(self, batch):
        """The mean loss on one minibatch, or a dict of batch-mean metrics
        with key 'loss'."""
        return self.loss(self(*batch[:-1], training=True), batch[-1])

    def validation_step(self, batch):
        """The mean loss on one minibatch of validation data, or a dict of
        batch-mean metrics with key 'loss'."""
        return self.loss(self(*batch[:-1], training=False), batch[-1])

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input #oo-design-models}
%%tab jax
class Module(nnx.Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X, *args, **kwargs):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X, *args, **kwargs)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)

    def training_step(self, batch):
        """The mean loss on one minibatch, or a dict of batch-mean metrics
        with key 'loss'."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def validation_step(self, batch):
        """The mean loss on one minibatch of validation data, or a dict of
        batch-mean metrics with key 'loss'."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        raise NotImplementedError
```

By convention, every minibatch is a tuple whose final element is the label tensor
and whose preceding elements are the model inputs. Both `training_step` and
`validation_step` unpack `*batch[:-1]` into `self(...)`, compute the loss against
`batch[-1]`, and simply *return* the result—performing no printing, plotting, or
logging of their own. Keeping models completely unaware of the trainer is a
deliberate design choice: it means any custom loop can train or evaluate a
`Module` without hidden side effects.

In this chapter, each step returns a single scalar—the mean minibatch loss.
Starting in :numref:`sec_softmax_scratch`, steps may instead return a dictionary
mapping metric names to minibatch means (always including `'loss'`, alongside
quantities like `'acc'`). Our tiny helper `as_metrics` normalizes a bare scalar
loss into `{'loss': out}` so the trainer handles both uniformly.

```{.python .input #oo-design-models-1}
def as_metrics(out):  #@save
    """A step returns a loss or a dict of metrics; return a dict."""
    return out if isinstance(out, dict) else {'loss': out}
```

### Linear Regression as a Module

Let us now rewrite the from-scratch model of :numref:`sec_linear_scratch` as a
subclass `LinearRegressionScratch(d2l.Module)`. Its constructor creates the same
initial weights and bias as `init_params`, wrapping them in the framework's
dedicated parameter type so that framework utilities can automatically discover,
inspect, save, and move them onto a GPU. Nothing about the underlying math
changes: we still compute the predictions, loss, and SGD update explicitly by
hand.

:begin_tab:`pytorch`
Wrapping tensors in `nn.Parameter` both enables gradient tracking (just like
`requires_grad=True`) and registers them on the `nn.Module` so
`model.parameters()` yields `w` and `b`.
:end_tab:

:begin_tab:`mxnet`
Assigning a Gluon `Parameter` as an attribute registers it with the `nn.Block`,
and `self.initialize()` allocates its value and gradient arrays (accessed via
`.data()`).
:end_tab:

:begin_tab:`tensorflow`
Calling `self.add_weight(...)` creates a mutable `tf.Variable` and registers it
with the Keras model so `model.trainable_variables` includes `w` and `b`.
:end_tab:

:begin_tab:`jax`
Wrapping arrays in `nnx.Param` marks them as trainable parameters for `nnx.grad`.
An optional `rngs` argument supplies the PRNG stream for initialization (falling
back to `d2l.get_key()` when omitted).
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

Next, we attach `forward` and `loss`—identical to `linreg` and `squared_loss`
from :numref:`sec_linear_scratch`, simply reading `self.w` and `self.b` off the
model instance.

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

```{.python .input #linear-regression-scratch-defining-the-loss-function  n=9}
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2
    return d2l.reduce_mean(l)
```

We package our one-line SGD parameter update into a small `SGD` optimizer class
matching each framework's native optimizer interface, so that in
:numref:`sec_linear_concise` we can swap in the framework's built-in optimizer
without touching a single line of `Trainer`.

:begin_tab:`pytorch`
Mirroring PyTorch's built-in optimizers, `step()` updates each parameter in place
and `zero_grad()` resets accumulated gradients to zero.
:end_tab:

:begin_tab:`mxnet`
Mirroring Gluon's `Trainer`, `step` updates parameters in place and accepts an
ignored batch-size argument (since our loss is already a minibatch mean).
:end_tab:

:begin_tab:`tensorflow`
Mirroring Keras optimizers, `apply_gradients` iterates over `(grad, param)` pairs
and updates each variable in place with `assign_sub`.
:end_tab:

:begin_tab:`jax`
Mirroring Optax gradient transformations, our `SGD` class defines pure `init` and
`update` methods that return parameter updates $-\eta \mathbf{g}$, which
`nnx.Optimizer` adds to the model's parameters.
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

Finally, `configure_optimizers` wires the model to our `SGD` optimizer:

:begin_tab:`mxnet`
We pass `SGD` the underlying arrays returned by `.data()`, whose `.grad`
attributes receive the gradients during backpropagation.
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

## Data
:label:`oo-design-data`

Our data base class `DataModule` is even simpler. Subclasses typically download
and preprocess raw datasets inside `__init__` (with `root` specifying the local
cache directory and `num_workers` setting the number of background data-loading
processes) and implement a single method, `get_dataloader(train)`, which returns a
reusable data loader yielding `(inputs..., labels)` minibatches. The convenience
methods `train_dataloader` and `val_dataloader` simply forward to it.

```{.python .input #oo-design-data}
class DataModule(d2l.HyperParameters):  #@save
    """The base class of data."""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)
```

Next, we turn our synthetic dataset from :numref:`sec_synthetic-regression-data`
into a `DataModule` subclass whose constructor calls `d2l.synthetic_data` to
generate `num_train + num_val` examples (by default $1{,}000$ for training and
$1{,}000$ for validation).

:begin_tab:`jax`
When `key` is omitted, the constructor draws a fresh one from `d2l.get_key()`, so
each instantiation yields a new dataset.
:end_tab:

```{.python .input #synthetic-regression-data-generating-the-dataset-1}
%%tab pytorch, tensorflow, mxnet
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.X, self.y = d2l.synthetic_data(w, b, noise, num_train + num_val)
```

```{.python .input #synthetic-regression-data-generating-the-dataset-1}
%%tab jax
class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32, key=None):
        super().__init__()
        self.save_hyperparameters()
        key = d2l.get_key() if key is None else key  # A new dataset per run
        self.X, self.y = d2l.synthetic_data(w, b, noise, num_train + num_val,
                                            key)
```

### A Loader for Every Epoch

What kind of object should `get_dataloader` return? Our simple `minibatches`
generator from :numref:`sec_linear_scratch` will not suffice here: `Trainer` asks
the `DataModule` for its loaders once up front and then iterates over them across
every epoch. A one-shot generator would run for the first epoch and silently yield
zero batches for every epoch thereafter!

Instead, we need a *reusable* data loader that can be iterated over epoch after
epoch, reshuffling the training set into a fresh random order on each pass while
reporting its batch count via `len`. (For massive datasets that cannot fit in RAM
or require on-the-fly image decoding, framework data loaders also stream files
from disk and prefetch batches using background worker threads while the GPU
trains on the current batch. Our synthetic tensors fit comfortably in memory, so
we keep things lightweight for now.)

Below, we add `get_tensorloader` to `DataModule` to serve in-memory tensors—using
the framework's native loader where it fits cleanly, and a tiny `ArrayLoader`
helper otherwise. Validation splits are always read in their stored order since
we only average metrics over them.

:begin_tab:`jax`
Because JAX focuses on array transformations rather than data pipelines, it does
not ship a built-in data loader. Our `ArrayLoader` keeps the arrays as NumPy
arrays on the host and splits a fresh PRNG key on every shuffled epoch so no
random order is repeated. Keeping the final partial minibatch triggers at most
one extra JIT compilation per training run (not per epoch).
:end_tab:

```{.python .input #synthetic-regression-data-reusable-data-loaders-1}
%%tab jax
class ArrayLoader:  #@save
    """Minibatches of in-memory arrays, reshuffled each pass if shuffle."""
    def __init__(self, arrays, batch_size, shuffle=False, drop_last=False,
                 key=None):
        self.arrays = tuple(np.asarray(a) for a in arrays)  # On the host
        self.batch_size, self.shuffle = batch_size, shuffle
        self.drop_last = drop_last
        if shuffle and key is None:  # A new order on each run
            key = d2l.get_key()
        self.key = key

    def __len__(self):
        n = len(self.arrays[0])
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        n = len(self.arrays[0])
        if self.shuffle:  # A new key, and hence a new order, on every pass
            self.key, key = jax.random.split(self.key)
            order = np.asarray(jax.random.permutation(key, n))
        else:
            order = np.arange(n)
        for i in range(len(self)):
            start = i * self.batch_size
            batch_indices = order[start : start + self.batch_size]
            yield tuple(a[batch_indices] for a in self.arrays)
```

```{.python .input #synthetic-regression-data-reusable-data-loaders-1}
%%tab mxnet
class ArrayLoader:  #@save
    """Minibatches of in-memory arrays, reshuffled each pass if shuffle."""
    def __init__(self, arrays, batch_size, shuffle=False, drop_last=False):
        self.arrays = tuple(arrays)
        self.batch_size, self.shuffle = batch_size, shuffle
        self.drop_last = drop_last

    def __len__(self):
        n = len(self.arrays[0])
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = mnp.arange(len(self.arrays[0]), dtype='int32')
        if self.shuffle:  # A new random order on every pass
            mnp.random.shuffle(order)
        for i in range(len(self)):
            start = i * self.batch_size
            batch_indices = order[start : start + self.batch_size]
            yield tuple(a[batch_indices] for a in self.arrays)
```

:begin_tab:`mxnet`
Gluon's `DataLoader` assembles minibatches one sample at a time; our `ArrayLoader`
fetches an entire batch with a single vectorized slice per tensor, which scales
much better with batch size.
:end_tab:

:begin_tab:`pytorch`
`TensorDataset` aligns tensors along their leading example axis, and `DataLoader`
slices them into minibatches—reshuffling automatically on every pass whenever
`shuffle=True`.
:end_tab:

:begin_tab:`tensorflow`
`from_tensor_slices` pairs examples across tensors, `shuffle` samples from a
buffer (using the full split size for uniform training shuffles and `1` to
preserve validation order), and `batch` groups them into minibatches.
:end_tab:

:begin_tab:`jax, mxnet`
`get_tensorloader` slices the requested split and wraps the tensors in an
`ArrayLoader`.
:end_tab:

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab pytorch
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    dataset = torch.utils.data.TensorDataset(*tensors)
    return torch.utils.data.DataLoader(dataset, self.batch_size,
                                       shuffle=train)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab tensorflow
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    shuffle_buffer = tensors[0].shape[0] if train else 1
    return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
        buffer_size=shuffle_buffer).batch(self.batch_size)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab jax
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    return ArrayLoader(tensors, self.batch_size, shuffle=train)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-1}
%%tab mxnet
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    return ArrayLoader(tensors, self.batch_size, shuffle=train)
```

Now `SyntheticRegressionData.get_dataloader` is just two lines—slicing either the
training split or the validation split and delegating to `get_tensorloader`:

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-2}
@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

Let us instantiate our synthetic dataset and repeat the two-pass test that
tripped up our plain generator earlier:

```{.python .input #oo-design-a-loader-for-every-epoch-1}
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-3  n=4}
loader = data.train_dataloader()
sizes = [len(y) for X, y in loader]
print(f'{len(sizes)} minibatches, the last of size {sizes[-1]}')
print(f'second pass: {len([y for X, y in loader])} minibatches')
for _ in range(2):  # Each new pass starts in a new random order
    X, y = next(iter(loader))
    print('a new pass starts with labels', d2l.numpy(y[:3, 0]).round(2))
```

This time, both passes yield all $32$ minibatches (ending with the $8$-example
remainder), and every pass begins in a fresh random order. Better yet, we can
query `len(loader)` directly without iterating through the dataset:

```{.python .input #synthetic-regression-data-concise-implementation-of-the-data-loader-4}
len(loader)
```

## Training
:label:`oo-design-training`

### The Training Loop and the History

The `Trainer` class brings `Module` and `DataModule` together to fit parameters
over `max_epochs` epochs. (Two additional constructor arguments wait for later
chapters: `num_gpus` in :numref:`sec_use_gpu` and `gradient_clip_val` in
:numref:`sec_rnn-scratch`.) `Trainer` also owns a `ProgressBoard` for live
training curves, defined shortly. Three small setup hooks—`prepare_data`,
`prepare_model`, and `prepare_optimizer`—are separated into their own methods so
subsequent chapters can override individual steps (for instance, moving models to
GPUs) without touching `fit`.

```{.python .input #oo-design-training}
class Trainer(d2l.HyperParameters):  #@save
    """Fit a Module to a DataModule, keeping a per-epoch history."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.board = ProgressBoard(xlabel='epoch', ylabel='loss')

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_model(self, model):
        self.model = model

    def prepare_optimizer(self, model):
        self.optim = model.configure_optimizers()
```

:begin_tab:`pytorch`
To support learning-rate schedules in :numref:`sec_scheduler`, `prepare_optimizer`
also unpacks an optional `(optimizer, scheduler)` tuple when returned by
`configure_optimizers`.
:end_tab:

:begin_tab:`jax`
`prepare_model` creates two parameter-sharing views of the model—one for training
and one for evaluation—so layers like dropout and batch normalization behave
appropriately in each mode. `prepare_optimizer` wraps the Optax gradient
transformation in `nnx.Optimizer`, caching the transformation structure so
repeated fits can reuse compiled JIT kernels.
:end_tab:

```{.python .input #oo-design-training-3}
%%tab pytorch
@d2l.add_to_class(Trainer)  #@save
def prepare_optimizer(self, model):
    """`configure_optimizers` may also return a learning-rate scheduler."""
    # PyTorch steps a schedule with a separate object (sec_scheduler)
    optim = model.configure_optimizers()
    self.optim, self.scheduler = (optim if isinstance(optim, tuple)
                                  else (optim, None))
```

```{.python .input #oo-design-training-3}
%%tab jax
@d2l.add_to_class(Trainer)  #@save
def prepare_model(self, model):
    # Two views that share the parameters: layers such as dropout behave
    # differently in training and in evaluation
    self.model = model
    self.train_model = nnx.view(model, deterministic=False,
                                use_running_average=False,
                                raise_if_not_found=False)
    self.val_model = nnx.view(model, deterministic=True,
                              use_running_average=True,
                              raise_if_not_found=False)

#@save
optimizer_cache = {}  # The most recent optax transformation, by its key

# Reusing the transformation lets a later fit of a model with the same
# structure reuse the compiled training step

@d2l.add_to_class(Trainer)  #@save
def prepare_optimizer(self, model):
    key = (jax.tree.structure(model), type(model).configure_optimizers)
    if key not in optimizer_cache:
        optimizer_cache.clear()  # One entry: it may refer to a model
        optimizer_cache[key] = model.configure_optimizers()
    self.optim = nnx.Optimizer(model, optimizer_cache[key], wrt=nnx.Param)
```

The `fit(model, data)` method implements the exact outer loop from our opening
pseudocode. After running the three setup hooks, it loops over `max_epochs`
epochs: in each epoch, `fit_epoch()` sweeps through the training loader and
returns mean training metrics, `validate()` evaluates the validation loader (when
present), and `record()` logs and plots both.

```{.python .input #oo-design-training-1}
@d2l.add_to_class(Trainer)  #@save
def fit(self, model, data):
    self.prepare_data(data)
    self.prepare_model(model)
    self.prepare_optimizer(model)
    self.history = collections.defaultdict(list)
    self.board.clear()
    self.board.xlim = [0, self.max_epochs]
    for self.epoch in range(self.max_epochs):
        # The training mean averages over the epoch, so it sits mid-epoch
        self.record('train', self.fit_epoch(), x=self.epoch + 0.5)
        if self.val_dataloader is not None:
            self.record('val', self.validate(), x=self.epoch + 1)
        self.board.show(final=False)
    self.board.show()

@d2l.add_to_class(Trainer)  #@save
def record(self, split, metrics, x):
    for name in sorted(metrics, key=lambda name: (name != 'loss', name)):
        self.history[f'{split}_{name}'].append(metrics[name])
        self.board.draw(x, metrics[name], f'{split}_{name}')
```

Notice that `fit` exposes the current epoch number on `self.epoch` and returns
nothing directly—every logged curve lives in `trainer.history`, a dictionary of
lists keyed by split and metric name (`'train_loss'`, `'val_loss'`, and later
`'val_acc'`). Thus `trainer.history['val_loss'][-1]` immediately retrieves the
final validation loss. Sorting `'loss'` first ensures consistent curve colors on
the progress board across models.

### Epoch Means

When `fit_epoch` and `validate` average metrics across an epoch, every *example*
should carry equal weight—not every *minibatch*. That distinction matters whenever
the batch size does not divide the dataset evenly: with $1{,}000$ examples and
`batch_size=32`, the final batch holds only $8$ examples, so unweighted minibatch
averaging would count each of those $8$ examples four times as heavily as the
other $992$! To avoid this bias, we weight each minibatch's reported mean by its
size `len(batch[-1])` (which is why every `training_step` and `validation_step`
must return a minibatch *mean*, never a sum).

Our small `Averager` class accumulates these example-weighted sums directly on the
compute device and converts them to Python floats only *once* per epoch inside
`mean()`. On a GPU, avoiding per-batch host synchronization prevents stalling the
asynchronous execution queue (:numref:`subsec_async_dispatch`); on a CPU with
three parameters it makes no speed difference, so you can read `Averager` simply
as a clean weighted-mean accumulator.

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
        """One host synchronization per metric; the loss comes first."""
        names = sorted(self.sums, key=lambda name: (name != 'loss', name))
        return {name: float(self.sums[name]) / self.n for name in names}
```

### Training and Validation Passes

`fit_epoch` and `validate` translate the inner loop of :numref:`sec_linear_scratch`
into our three-class interface: minibatches come from the data loader, pass
through `prepare_batch` (currently an identity hook that :numref:`sec_use_gpu`
will override to transfer batches onto GPUs), flow through `training_step` or
`validation_step`, and accumulate into `Averager`.

```{.python .input #linear-regression-scratch-training-1  n=15}
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

Because this exact trainer will run every model throughout the rest of the book,
it is slightly longer than bare linear regression requires—yet its algorithmic
core is still the exact four-step SGD cycle of :numref:`sec_linear_scratch`
(forward pass and loss, backward pass, unrecorded update, and gradient clearing),
with a few forward-looking hooks for later chapters that remain dormant here.

:begin_tab:`pytorch`
The two `if` blocks are hooks for gradient clipping and learning-rate schedulers.
Calling `.detach()` before passing metrics to `Averager` prevents the running
loss sum from retaining the computational graph of every minibatch in memory, and
`model.train()`/`model.eval()` toggle layers such as dropout and batch
normalization once we introduce them.
:end_tab:

:begin_tab:`mxnet`
`self.optim.step(1)` passes a batch size of `1` since our loss is already a
minibatch mean. Calling `wait_to_read()` on the previous batch's loss keeps
Python at most one minibatch ahead of MXNet's asynchronous backend, bounding
memory usage.
:end_tab:

:begin_tab:`tensorflow`
Because executing TensorFlow operations eagerly one by one from Python incurs
heavy overhead, this `prepare_model` override runs one dummy forward pass (so
Keras materializes all layer variables) and wraps `train_step` and `val_step`
with `tf.function` to compile them into fast fused graphs.
:end_tab:

:begin_tab:`jax`
Decorating `_trainer_train_step` and `_trainer_validation_step` with `@nnx.jit`
compiles the forward pass, loss, gradient, and optimizer update into a single
fused XLA computation. Passing `has_aux=True` lets `nnx.grad` differentiate
`out['loss']` while returning the full metrics dictionary alongside the
gradients.
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
    metrics, previous = d2l.Averager(), None
    for batch in map(self.prepare_batch, self.train_dataloader):
        with autograd.record():
            out = d2l.as_metrics(self.model.training_step(batch))
        out['loss'].backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        metrics.add(out, len(batch[-1]))
        if previous is not None:  # Python stays one step ahead at most
            previous.wait_to_read()
        previous = out['loss']
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
    loader = (self.train_dataloader if self.train_dataloader is not None
              else self.val_dataloader)  # `evaluate` has no training data
    batch = self.prepare_batch(next(iter(loader)))
    model(*batch[:-1], training=False)  # Keras creates the variables here
    self._train_step = tf.function(self.train_step, reduce_retracing=True)
    self._val_step = tf.function(self.val_step, reduce_retracing=True)

@d2l.add_to_class(d2l.Trainer)  #@save
def train_step(self, batch):
    with tf.GradientTape() as tape:
        out = d2l.as_metrics(self.model.training_step(batch))
    params = self.model.trainable_variables
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
        out = d2l.as_metrics(model.training_step(batch))
        return out['loss'], out
    grads, out = nnx.grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return out

@nnx.jit  #@save
def _trainer_validation_step(model, batch):
    return d2l.as_metrics(model.validation_step(batch))

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

### The Progress Board

To visualize training live inside a notebook (a lightweight nod to tools like
[TensorBoard](https://www.tensorflow.org/tensorboard)), `Trainer` logs every
epoch mean to `ProgressBoard`. Its interface is minimal: `draw(x, y, label)`
appends a point to curve `label`, `show()` renders the figure (implemented in
:numref:`sec_utils`), and `clear()` resets it. You can customize plotting before
calling `fit`—for example, setting `trainer.board.yscale = 'log'` for logarithmic
loss plots or `trainer.board.display = False` when running silent sweeps.

Notice one subtle plotting detail: the training loss mean averages over batches
processed *while* the parameters were improving throughout the epoch, whereas the
validation loss is evaluated using the updated parameters at the *end* of the
epoch. Plotting the training mean at the midpoint of each epoch (`0.5, 1.5, ...`)
and the validation mean at the end (`1, 2, ...`) aligns the two curves honestly on
the same timeline.

```{.python .input #oo-design-utilities-6}
class ProgressBoard(d2l.HyperParameters):  #@save
    """Curves that grow during training; `show` redraws them."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', figsize=(3.5, 2.5),
                 display=True):
        self.save_hyperparameters()
        self.clear()

    def clear(self):
        """Forget all curves; the next `show` starts a new figure."""
        self.data, self.handle = {}, None

    def draw(self, x, y, label):
        """Append the point (x, y) to the curve called `label`."""
        xs, ys = self.data.setdefault(label, ([], []))
        xs.append(x)
        ys.append(y)

    def show(self, final=True):
        raise NotImplementedError
```

## Training Linear Regression Again

Let us now retrain the exact same linear regression model from
:numref:`sec_linear_scratch` using our new modular classes. Notice how cleanly
every hyperparameter now lives with the component that owns it: the learning rate
`lr=0.03` on the model, the batch size `32` (default) on `data`, and `max_epochs=10`
on `d2l.Trainer` (which pairs our `Trainer` class with the full plotting routine
in :numref:`sec_utils`).

```{.python .input #oo-design-training-linear-regression-again-1}
model = LinearRegressionScratch(2, lr=0.03)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

The validation loss traces the exact same trajectory as in
:numref:`sec_linear_scratch`, plunging roughly $7\times$ per epoch until it hits
the noise floor near $5 \times 10^{-5}$, with the training curve lying right on
top of it (with only $3$ parameters and $1{,}000$ examples, the model barely
memorizes any label noise). Let us verify the final validation loss and parameter
recovery errors numerically:

```{.python .input #oo-design-training-linear-regression-again-2}
%%tab pytorch, tensorflow
w_error = d2l.numpy(d2l.reshape(model.w, data.w.shape) - data.w)
b_error = d2l.numpy(model.b - data.b)
print(f"validation loss: {trainer.history['val_loss'][-1]:.1e}")
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

```{.python .input #oo-design-training-linear-regression-again-2}
%%tab mxnet
w_error = d2l.numpy(d2l.reshape(model.w.data(), data.w.shape) - data.w)
b_error = d2l.numpy(model.b.data() - data.b)
print(f"validation loss: {trainer.history['val_loss'][-1]:.1e}")
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

```{.python .input #oo-design-training-linear-regression-again-2}
%%tab jax
w_error = d2l.numpy(d2l.reshape(model.w[...], data.w.shape) - data.w)
b_error = d2l.numpy(model.b[...] - data.b)
print(f"validation loss: {trainer.history['val_loss'][-1]:.1e}")
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

Everything matches: the validation loss sits right at the $5 \times 10^{-5}$
noise floor, and all three parameter errors lie around a few times $10^{-4}$
(comfortably inside our $10^{-3}$ threshold). For a single one-off model, all this
object-oriented scaffolding might seem like overkill—but it pays off immediately
in the very next section, where we swap in built-in framework layers, losses, and
optimizers while reusing `d2l.Trainer` and `SyntheticRegressionData` without
changing a single line.

## Summary

* Nearly every model in this book shares the same outer training loop, so we
  implement it once in `Trainer` and isolate problem-specific details into
  subclasses of `Module` (parameters, `forward`, `loss`, `configure_optimizers`)
  and `DataModule` (reusable training and validation data loaders).
* Keeping `training_step` and `validation_step` free of logging or plotting side
  effects decouples models from the trainer; `Averager` computes unbiased
  example-weighted epoch means (handling partial final minibatches cleanly), and
  `Trainer` records them in `trainer.history` and plots them on `ProgressBoard`.
* Refactored into these three classes (stored in the `d2l` library), our
  from-scratch linear regression reproduces the exact noise floor and parameter
  recovery of :numref:`sec_linear_scratch`—and lets future chapters swap out
  models, losses, optimizers, or datasets independently.

## Exercises

1. [code] **Methods added later.** Register a second method `do` on class
   `A` with a different body, predict which version `a.do()` runs, and
   check. Then attach a function `greet(self)` to the instance instead, with
   `setattr(a, 'greet', greet)`, call `a.greet()`, and explain the error.

1. **Optimizer and model.** The `Module` class keeps the optimizer in
   `configure_optimizers`, a *method of the model*, rather than passing it
   as an argument to `Trainer`. State the advantages of this design choice,
   and describe a training setup in which putting the optimizer on the
   model becomes awkward.

1. **Reading the training curves.** The trainer draws the mean training
   loss of each epoch at the middle of the epoch and the validation loss
   at its end.
    1. Early in training, the training curve often lies above the
       validation curve, although both sets of examples come from the same
       distribution. Explain why from the way the two means are computed.
    1. Suppose `fit` computed the training loss like the validation loss,
       with the parameters at the end of each epoch. What would this cost
       per epoch, and what would the gap between the two curves then
       measure?

1. [code] **Early stopping.** Training for a fixed number of epochs may
   stop too early or too late. Write a subclass of `d2l.Trainer` whose
   `fit` ends training once the validation loss has not decreased for
   `patience` consecutive epochs, deciding from `self.history` alone.
   Explain why no model or dataset class has to change. Test your trainer
   on the synthetic regression data with a large `max_epochs`.

1. [code] **Test split.** Write a subclass of `SyntheticRegressionData`
   whose constructor also draws a test set from the same distribution and
   whose method `test_dataloader` serves it. Train a model on the original
   data, and compute its mean loss on the test set once, after the last
   epoch, with the model's `validation_step`. Explain why the test set is
   evaluated once at the end rather than after every epoch like the
   validation data, and what would go wrong if the epoch with the best test
   loss were reported. :numref:`sec_generalization_basics` returns to this
   question.

1. [code] **Planck's law.** ● The same `Module` and `Trainer` fit any model
   whose loss can be differentiated, not only a linear one. Use
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
7. **A partial last minibatch.** With 1,000 examples per split and
   `batch_size=32`, each loader ends a pass with a partial minibatch of 8
   examples. `ArrayLoader` discards it when given `drop_last=True`. The
   training loader reshuffles every epoch, whereas the validation loader
   reads in a fixed order. For each loader, state which examples a
   discarded partial minibatch leaves out over many epochs, and decide for
   which loader discarding is acceptable.
:end_tab:

:begin_tab:`pytorch`
7. **A partial last minibatch.** With 1,000 examples per split and
   `batch_size=32`, each loader ends a pass with a partial minibatch of 8
   examples. `DataLoader` discards it when given `drop_last=True`. The
   training loader reshuffles every epoch, whereas the validation loader
   reads in a fixed order. For each loader, state which examples a discarded
   partial minibatch leaves out over many epochs, and decide for which
   loader discarding is acceptable.
:end_tab:

:begin_tab:`tensorflow`
7. **A partial last minibatch.** With 1,000 examples per split and
   `batch_size=32`, each loader ends a pass with a partial minibatch of 8
   examples. `Dataset.batch` discards it when given `drop_remainder=True`.
   The training loader reshuffles every epoch, whereas the validation loader
   reads in a fixed order. For each loader, state which examples a discarded
   partial minibatch leaves out over many epochs, and decide for which
   loader discarding is acceptable.
:end_tab:

:begin_tab:`jax`
7. **A partial last minibatch.** With 1,000 examples per split and
   `batch_size=32`, each loader ends a pass with a partial minibatch of 8
   examples. `ArrayLoader` discards it when given `drop_last=True`.
    1. The training loader reshuffles every epoch, whereas the validation
       loader reads in a fixed order. For each loader, state which examples
       a discarded partial minibatch leaves out over many epochs, and decide
       for which loader discarding is acceptable.
    1. A compiled step is compiled once for every new input shape. Count
       the compilations of the training and validation steps that keeping
       the partial minibatch causes over ten epochs, and compare their cost
       with that of discarding 8 examples in every epoch.
:end_tab:

8. [code] **On-the-fly generation.** Write a `DataModule` subclass that
   stores no $\mathbf{X}$ or $\mathbf{y}$. Its training loader instead draws
   every minibatch fresh from the data-generating process when the
   minibatch is requested.
    1. A stream of fresh minibatches has no natural end, yet the `Trainer`
       ends an epoch only when a pass over the training loader ends. Decide
       how many minibatches one epoch yields, make every pass stop there,
       and make `len` return that number.
    1. Decide whether the validation loader should also draw fresh examples
       on every pass, and implement your choice. Justify it by what the
       validation loss is used for.

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/6645)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/6646)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/6647)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17974)
:end_tab:


<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.4]{.kicker}

One training loop for every model<br>
**Module · DataModule · Trainer · epoch means · the same result again**
:::
:::

::: {.slide title="A Loop Written for One Model Would Need Close to 100 Copies"}
[The problem]{.kicker}

The dozen-line `train` of the previous deck names `linreg`,
`squared_loss`, `w`, `b`, and `X_train`. Stripped of the names:

```{.python #oo-design-exercises-3}
for epoch in range(num_epochs):
    for X, y in train_data:                     # data
        loss = loss_fn(net(X), y)               # model and loss
        update(params, gradient(loss, params))  # optimizer
    val_loss = mean_loss(net, val_data)         # model on held-out data
    record(epoch, val_loss)
```

- Only the commented lines change between models.
- Nearly 30 sections train a model, most in several frameworks.

::: {.d2l-note}
Clipping gradients would mean editing every copy. Write the loop **once**.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- say which part of training `Module`, `DataModule`, and `Trainer` own;
- write a model whose steps return the mean loss of a minibatch;
- compute an epoch mean that counts every example equally;
- read `trainer.history` after training;
- train the linear model of the previous deck through the classes and get
  the same result.

**Running example:** $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$,
$\sigma = 0.01$; 1,000 training and 1,000 validation examples in
minibatches of 32.
:::

::: {.slide title="Three Classes Turn the Loop Into Reusable Code"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| Synthetic Data | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | SGD and the training loop by hand |
| **Object-Oriented Design** | **`Module`, `DataModule`, `Trainer`** |
| Concise Implementation | framework layers and losses |
| Generalization | training vs. test error |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** the dozen-line training loop and the generator of the previous
  deck; Python classes.
- **Skips:** the drawing code of the progress board; asynchronous execution,
  covered in the chapter on computation.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Model and data]{.dtitle}

[Module · DataModule · a loader for every epoch]{.dsub}
:::
:::

::: {.slide title="Each Line of the Loop Has One Owner"}
[Three classes]{.kicker}

![](../img/mdl-linreg-oo-classes.svg){.nostretch width=90%}

| in the loop | owner | through |
|--------------------------|-----------|------------------------------|
| `train_data`, `val_data` | `DataModule` | `train_dataloader()`, `val_dataloader()` |
| `net`, `loss_fn`, the update rule | `Module` | `forward`, `loss`, `configure_optimizers()` |
| both `for` loops, gradients, `record` | `Trainer` | `fit`, `history` |
:::

::: {.slide title="A Model's Steps Return the Mean Loss and Do Nothing Else"}
[Module]{.kicker}

| method | returns | called |
|---|---|---|
| `forward(X)` | predictions | inside the steps |
| `loss(y_hat, y)` | the mean loss over the minibatch | inside the steps |
| `training_step(batch)` | the loss, or a dict with key `'loss'` | per training minibatch |
| `validation_step(batch)` | the loss, or a dict of metrics | per validation minibatch |
| `configure_optimizers()` | the optimizer | once, before training |

A minibatch is a tuple `(inputs..., labels)`.

::: {.d2l-note}
No printing, plotting, or recording: nothing in the model refers to the
trainer, so **any loop** can train or evaluate it.
:::
:::

::: {.slide title="add_to_class Adds a Method to a Class That Already Exists"}
[Notebook utilities]{.kicker}

@oo-design-utilities-2

@oo-design-utilities-3

- `a`, created before `do`, has it: Python looks a method up on the class
  when it is called.
- `save_hyperparameters()` stores every constructor argument:
  `Trainer(max_epochs=10)` has `self.max_epochs`.
:::

::: {.slide title="LinearRegressionScratch Registers w and b With the Module"}
[Module]{.kicker}

@linear-regression-scratch-defining-the-model-1

- `forward` and `loss` are `linreg` and `squared_loss` of the previous
  deck, now methods that read the parameters from the model.

::: {.d2l-note}
Registered parameters are listed, saved, and moved to a GPU with the model.
The loss and the update are still ours.
:::
:::

::: {.slide title="Keras Calls call, So Module Redirects It to forward" only="tensorflow"}
[Module · TensorFlow]{.kicker}

```{.python #oo-design-exercises-4}
def call(self, X, *args, training=None, **kwargs):
    if training is not None:
        self.training = training
    return self.forward(X, *args, **kwargs)

def training_step(self, batch):
    return self.loss(self(*batch[:-1], training=True), batch[-1])
```

- `validation_step` passes `training=False`; dropout, written later, reads
  `self.training`.
- Keras 3 sets an attribute `loss = None`; the constructor deletes it so
  that the method `loss` is found.
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

`configure_optimizers` returns `SGD([self.w, self.b], self.lr)`.
:::

::: {.slide title="SGD Subtracts η Times the Gradient, in Place" only="mxnet"}
[The optimizer]{.kicker}

$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla_{(\mathbf{w}, b)} L$,
with learning rate $\eta$. The one method of our `SGD` class:

```{.python #linear-regression-scratch-exercises-7}
def step(self, _):  # batch size ignored: the loss is a mean
    for param in self.params:  # data() arrays
        param -= self.lr * param.grad
```

`configure_optimizers` returns `SGD([self.w.data(), self.b.data()], self.lr)`.
:::

::: {.slide title="SGD Subtracts η Times the Gradient, in Place" only="tensorflow"}
[The optimizer]{.kicker}

$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla_{(\mathbf{w}, b)} L$,
with learning rate $\eta$. The one method of our `SGD` class:

```{.python #linear-regression-scratch-exercises-8}
def apply_gradients(self, grads_and_vars):
    for grad, param in grads_and_vars:
        param.assign_sub(self.lr * grad)
```

The same interface as a Keras optimizer; `configure_optimizers` returns
`SGD(self.lr)`.
:::

::: {.slide title="In JAX, SGD Returns the Update −η·g Instead of Applying It" only="jax"}
[The optimizer]{.kicker}

JAX arrays are immutable. An optax optimizer computes updates, and
`nnx.Optimizer` adds them to the parameters. The two methods of our `SGD`
class:

```{.python #linear-regression-scratch-exercises-5}
def init(self, params):                  # SGD keeps no state
    return optax.EmptyState()

def update(self, updates, state, params=None):
    updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
    return updates, state
```
:::

::: {.slide title="Check: Which Class Holds Each Setting?"}
[Check]{.kicker}

The running example trains for 10 epochs with learning rate
$\eta = 0.03$. Which class holds

1. the learning rate $\eta = 0.03$?
2. the minibatch size 32?
3. the 10 epochs?
4. the true $\mathbf{w}^*$ and the noise level $\sigma = 0.01$?
5. the squared loss?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) `Module`: `LinearRegressionScratch(2, lr=0.03)` hands it
to `SGD`. (2) `DataModule`: the default `batch_size=32`. (3) `Trainer`:
`max_epochs=10`. (4) `DataModule`: the synthetic data. (5) `Module`: its
`loss`.
:::
:::
:::

::: {.slide title="A Generator Runs Out After One Pass; a Loader Does Not"}
[DataModule]{.kicker}

A `DataModule` subclass writes one method, `get_dataloader(train)`, which
returns a **data loader**: an object that yields minibatches `(X, y)`.

| | our generator | framework loader |
|---|---|---|
| a second pass | yields nothing | a new random order |
| `len(loader)` | needs a full pass | 32, without a pass |
| data larger than memory | must fit in memory | read from disk |
| assembling minibatches | in the training process | parallel workers, ahead of time |

::: {.d2l-note}
The trainer iterates over the same loader in every epoch: a generator would
train for one epoch, then silently do nothing.
:::
:::

::: {.slide title="The New Loader Yields 32 Minibatches on Every Pass"}
[DataModule]{.kicker}

The test of the generator, then the first labels of two new passes:

@!synthetic-regression-data-concise-implementation-of-the-data-loader-3

The loader reports its length without a pass:

@synthetic-regression-data-concise-implementation-of-the-data-loader-4
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The trainer]{.dtitle}

[the loop · history · epoch means]{.dsub}
:::
:::

::: {.slide title="The Trainer Writes the Loop Once"}
[Trainer]{.kicker}

```{.python #oo-design-exercises-1}
def fit(self, model, data):
    ...  # prepare the data, the model, and the optimizer
    self.history = collections.defaultdict(list)
    for self.epoch in range(self.max_epochs):
        # The training mean averages over the epoch: draw it mid-epoch
        self.record('train', self.fit_epoch(), x=self.epoch + 0.5)
        self.record('val', self.validate(), x=self.epoch + 1)
        self.board.show(final=False)
```

- `fit_epoch` and `validate` return the means over one pass.
- The training mean uses parameters that change during the epoch: drawn
  at 0.5, 1.5, …; validation at 1, 2, ….
:::

::: {.slide title="In JAX the Trainer Holds Two Views and the Optimizer State" only="jax"}
[Trainer · JAX]{.kicker}

```{.python #oo-design-exercises-2}
tx = model.configure_optimizers()       # an optax transformation
self.optim = nnx.Optimizer(model, tx, wrt=nnx.Param)
self.train_model = nnx.view(model, deterministic=False, ...)
self.val_model = nnx.view(model, deterministic=True, ...)
```

- `prepare_model` creates two views that share the parameters; layers
  such as dropout read the mode from them.
- `nnx.Optimizer` holds the optimizer state next to the model it updates.
:::

::: {.slide title="history Keys Join the Split and the Metric Name"}
[Trainer]{.kicker}

A step returns a loss or a dict; `as_metrics` turns both into a dict:

@oo-design-models-1

```{.python #oo-design-exercises-6}
def record(self, split, metrics, x):
    for name in sorted(metrics, key=lambda name: (name != 'loss', name)):
        self.history[f'{split}_{name}'].append(metrics[name])
        self.board.draw(x, metrics[name], f'{split}_{name}')
```

- Keys: `'train_loss'`, `'val_loss'`; a classifier adds `'val_acc'`.
- `fit` returns nothing: read `trainer.history['val_loss'][-1]`.
:::

::: {.slide title="Weighting by Size Counts Every Example 1/1,000"}
[Epoch means]{.kicker}

1,000 examples in minibatches of 32: 31 full ones and one of 8.

| how the epoch mean is formed | one example of a full minibatch | one of the last 8 |
|---|---|---|
| minibatch means, equal weights | $1/32 \cdot 1/32 = 1/1024$ | $1/32 \cdot 1/8 = 1/256$ |
| weighted by size (the trainer) | $1/1000$ | $1/1000$ |

- Equal weights count each of the last 8 examples **4×** as much.
- The trainer adds $n_b \times$ mean per minibatch, divides by 1,000, and
  reads the sums once per epoch.

::: {.d2l-note .warn}
So a step returns a **mean**: a loss summed over 32 examples would enter the
epoch mean 32× too large.
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
previous = None
for batch in self.train_dataloader:
    with autograd.record():
        loss = self.model.training_step(batch)  # 1. forward, loss
    loss.backward()                             # 2.+3. overwrite
    self.optim.step(1)                          # 4. update
    metrics.add({'loss': loss}, len(batch[-1]))
    if previous is not None:                    # queue: one step long
        previous.wait_to_read()
    previous = loss
return metrics.mean()                           # one read per epoch
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

::: {.slide title="Check: What Does the Trainer Report?"}
[Check]{.kicker}

Validation: 1,000 examples, minibatches of 32. The 31 full minibatches have
mean loss 0.010 and the last one, of 8 examples, mean loss 0.050.

1. Which epoch mean does the trainer report?
2. Which would equal weights per minibatch report?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) $(992 \cdot 0.010 + 8 \cdot 0.050)/1000 \approx 0.0103$.
(2) $(31 \cdot 0.010 + 0.050)/32 \approx 0.0113$: 9% too high, because 8
examples count as much as 32.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Linear regression again]{.dtitle}

[the same model · the same data · through the classes]{.dsub}
:::
:::

::: {.slide title="Through the Classes, the Loss Again Falls 7× per Epoch to 5×10⁻⁵"}
[Training again]{.kicker}

@oo-design-training-linear-regression-again-1

- Training and validation curves lie on top of each other: three
  parameters cannot fit the noise of 1,000 labels.
:::

::: {.slide title="The Parameters Again Come Within About 10⁻³ of (2, −3.4) and 4.2"}
[Training again]{.kicker}

@!oo-design-training-linear-regression-again-2

- Validation loss at the floor $\sigma^2/2 = 5 \times 10^{-5}$.
- Parameter errors typically a few times $10^{-4}$, as with the
  dozen-line loop: only the organization of the code has changed.
:::

::: {.slide title="Check: Which Class Changes?"}
[Check]{.kicker}

Where does each change go?

1. Train a new network on a new dataset.
2. Report accuracy as well as the loss.
3. Use a different optimizer.
4. Stop when the validation loss stops improving.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** (1) New subclasses of `Module` and `DataModule`. (2) The
model's `validation_step` returns a dict with `'loss'` and `'acc'`. (3) The
model's `configure_optimizers`. (4) `Trainer.fit`, once, for every model.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| class | owns | contract |
|-----------|----------------------|--------------------------------|
| `Module` | parameters, `forward`, `loss`, optimizer | steps **return** the mean loss or a dict of means |
| `DataModule` | training and validation data | `get_dataloader(train)` yields minibatches on every pass |
| `Trainer` | the loop, `history`, `board` | means over examples: each weighs 1/1,000 |

- Re-run through the classes: loss $\approx 5 \times 10^{-5}$ after 10
  epochs at 7× per epoch; parameter errors within about $10^{-3}$.

**Next:** the concise implementation: a framework layer, loss, and
optimizer in the `Module`, trained by the same `Trainer`.
:::
