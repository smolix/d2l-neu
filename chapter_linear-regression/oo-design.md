```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Object-Oriented Design for Implementation
:label:`sec_oo-design`

In :numref:`sec_linear_scratch`, we saw both sides of implementing a machine learning model: first a standalone twelve-line `train` function wired directly to one model and one dataset, and then a concise four-line `trainer.fit(model, data)` call. As we move on to multilayer perceptrons, convolutional networks, recurrent networks, and Transformers, rewriting data loaders, epoch loops, metric averagers, and live plots in every section would bury the new ideas under repetitive plumbing.

To keep our focus on modeling while keeping every line of the training harness transparent and modifiable, we organize the `d2l` library around three clean object-oriented abstractions (:numref:`fig_oo_design`), inspired by frameworks like PyTorch Lightning :cite:`Falcon.ea.2019`:

1. **`Module` (models):** Holds trainable parameters, defines the `forward` pass, computes the `loss`, specifies `training_step` and `validation_step`, and returns its optimizer via `configure_optimizers`.
2. **`DataModule` (data):** Encapsulates dataset downloading, preprocessing, and splitting, exposing `train_dataloader()` and `val_dataloader()` iterables.
3. **`Trainer` (training loop):** Connects any `Module` to any `DataModule`, running the epoch-by-epoch minibatch training and validation passes, averaging metrics across batches, and plotting live learning curves on a `ProgressBoard`.

![The three core classes of the `d2l` training harness: `DataModule` supplies training and validation minibatches, `Module` defines the model, loss, and optimizer, and `Trainer` drives the training and validation loops.](../img/mdl-linreg-oo-classes.svg)
:label:`fig_oo_design`

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

Before building the three core classes, we introduce two small Python utilities that keep notebook definitions readable and self-contained:

1. **`@add_to_class(Class)`:** In a textbook or notebook, presenting a 60-line class in one giant code block makes it hard to explain each method as it appears. The decorator `@add_to_class(Class)` lets us define a class's skeleton first and attach new methods to it incrementally in later code cells.
2. **`HyperParameters` (`save_hyperparameters()`):** Writing `self.a = a; self.b = b; ...` at the top of every `__init__` method is pure boilerplate. Inheriting from `HyperParameters` and calling `self.save_hyperparameters()` inspects the caller's frame and automatically binds every `__init__` argument (except any listed in `ignore`) as an attribute on `self`.

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

The `Module` class is the base class of every model we implement. It inherits from each framework's native neural network container (`nn.Module` in PyTorch, `nn.Block` in MXNet, `tf.keras.Model` in TensorFlow, or `nnx.Module` in Flax/JAX) as well as `d2l.HyperParameters`. By default:

- Calling `model(X)` invokes `model.forward(X)`.
- `training_step(batch)` passes the features `batch[:-1]` through the model and returns a dictionary `{'loss': l}` comparing the predictions against the labels `batch[-1]`.
- `validation_step(batch)` computes the same validation loss dictionary (and in :numref:`sec_softmax_scratch`, `Classifier` will override `validation_step` to report both `'loss'` and `'acc'`).
- The helper `as_metrics` normalizes any single scalar or dictionary return value into a metric dictionary so `Trainer` can log arbitrary named quantities uniformly.

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

To see how `Module` works in practice before using built-in framework layers, let us wrap our from-scratch linear regression parameters $(\mathbf{w}, b)$, forward pass, squared loss, and from-scratch `SGD` updater into `LinearRegressionScratch(d2l.Module)`:

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

Next, we wrap our from-scratch minibatch SGD update rule into a reusable `SGD(d2l.HyperParameters)` class and return it from `LinearRegressionScratch.configure_optimizers`:

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

The `DataModule` base class is the counterpart to `Module` for datasets. Each concrete dataset subclass implements `train_dataloader()` and `val_dataloader()` (typically delegating both to `get_dataloader(train=True/False)`):

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

Let us package our synthetic regression generator from :numref:`sec_synthetic-regression-data` into `SyntheticRegressionData(d2l.DataModule)`. It generates `num_train + num_val` examples up front (1,000 training and 1,000 validation examples by default) so that `Trainer` can track both training loss and held-out validation loss after every epoch:

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


Inside `get_dataloader`, we want an object that can be iterated over once per epoch—reshuffling the training split at the start of every epoch while reading the validation split in deterministic order. We provide both a from-scratch `ArrayLoader` class (wrapping our slice-by-slice index permutation logic from :numref:`sec_linear_scratch`) and a concise `get_tensorloader` method that delegates to each framework's native `DataLoader` / `tf.data.Dataset`:

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

Now we build `Trainer`, the engine that wires a `Module` and a `DataModule` together. Calling `trainer.fit(model, data)` prepares the data loaders, attaches the model and its optimizer, and runs `max_epochs` passes. In every epoch, `fit_epoch` sweeps through the training minibatches (updating parameters and averaging training metrics) and then calls `validate` over the validation minibatches, recording the epoch averages in `self.history` and updating a live `ProgressBoard` plot:

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

Because the last minibatch of an epoch is often smaller than `batch_size` (for instance, $1{,}000 = 31 \times 32 + 8$), an unweighted average of per-batch mean losses would give the 8 examples in the final batch the exact same vote as the 32 examples in every earlier batch. Our `Averager` helper accumulates each batch's metric weighted by its batch size `n`, so `avg.means()` always returns the exact per-example average across the entire split:

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

Next, `validate` and `fit_epoch` execute the per-batch validation and training loops across our four frameworks (with optional gradient clipping `gradient_clip_val` ready for recurrent networks in :numref:`chap_rnn`):

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

Finally, `ProgressBoard` renders live epoch-by-epoch learning curves during notebook execution (and can be silenced via `trainer.board.display = False` whenever we run silent sweeps):

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

Let us test our newly assembled `Module`, `DataModule`, and `Trainer` classes end-to-end by training `LinearRegressionScratch` on `SyntheticRegressionData` for 10 epochs and checking the recovered parameters:

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

Both training and validation loss drop smoothly to the $0.00005$ noise floor, and the recovered parameters match $(\mathbf{w}^*, b^*)$ to within about a thousandth—confirming that our object-oriented harness reproduces our standalone scratch loop while freeing every future chapter from boilerplate.

## Summary

- **Three core abstractions:** The `d2l` library separates modeling (`Module`), data loading (`DataModule`), and optimization loops (`Trainer`), flanked by `@add_to_class`, `HyperParameters`, `Averager`, and `ProgressBoard`.
- **Exact epoch averages:** `Averager` weights each minibatch's metrics by its example count so partial final minibatches never skew epoch-level losses or accuracies.
- **Drop-in reuse:** Any model subclassing `Module` (whether built from raw tensors like `LinearRegressionScratch` or framework layers like `LinearRegression` in :numref:`sec_linear_concise`) trains and validates through the exact same `trainer.fit(model, data)` call.

## Exercises

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
1. **A partial last minibatch.** With 1,000 examples per split and
   `batch_size=32`, each loader ends a pass with a partial minibatch of 8
   examples. `ArrayLoader` discards it when given `drop_last=True`. The
   training loader reshuffles every epoch, whereas the validation loader
   reads in a fixed order. For each loader, state which examples a
   discarded partial minibatch leaves out over many epochs, and decide for
   which loader discarding is acceptable.
:end_tab:

:begin_tab:`pytorch`
1. **A partial last minibatch.** With 1,000 examples per split and
   `batch_size=32`, each loader ends a pass with a partial minibatch of 8
   examples. `DataLoader` discards it when given `drop_last=True`. The
   training loader reshuffles every epoch, whereas the validation loader
   reads in a fixed order. For each loader, state which examples a discarded
   partial minibatch leaves out over many epochs, and decide for which
   loader discarding is acceptable.
:end_tab:

:begin_tab:`tensorflow`
1. **A partial last minibatch.** With 1,000 examples per split and
   `batch_size=32`, each loader ends a pass with a partial minibatch of 8
   examples. `Dataset.batch` discards it when given `drop_remainder=True`.
   The training loader reshuffles every epoch, whereas the validation loader
   reads in a fixed order. For each loader, state which examples a discarded
   partial minibatch leaves out over many epochs, and decide for which
   loader discarding is acceptable.
:end_tab:

:begin_tab:`jax`
1. **A partial last minibatch.** With 1,000 examples per split and
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

1. [code] **On-the-fly generation.** Write a `DataModule` subclass that
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
