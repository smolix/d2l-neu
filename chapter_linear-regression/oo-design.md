```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Object-Oriented Design for Implementation
:label:`sec_oo-design`

Almost every model in this book follows the same loop: load data, run a forward
pass, compute the loss, update the parameters, and repeat. If we wrote that loop
from scratch for each new model, a small change to the training procedure (say,
adding gradient clipping or a learning-rate schedule) would force us to touch
every chapter. The fix, borrowed from libraries such as
[Lightning](https://lightning.ai/), is to write the loop once inside a reusable
`Trainer` class and let the model and the data vary as subclasses of `Module` and
`DataModule`:

* **`Module`** holds the model parameters, the `forward` pass, the loss, and the
  optimizer. Every model in the book is a subclass.
* **`DataModule`** holds a dataset and serves its training and validation data
  loaders. Every dataset is a subclass.
* **`Trainer`** owns the epoch loop: it feeds batches from the `DataModule` into
  the `Module` and applies the optimizer.

Most chapters subclass only `Module` and `DataModule`; we return to the `Trainer`
itself when we reach GPUs, parallel training, and optimization.
:numref:`fig_oo_design` shows how the three fit together.

![The three base classes and how they fit together. `Trainer.fit` drives a `Module` (which holds the model, loss, and optimizer) over data served by a `DataModule`; new models and datasets are written as subclasses.](../img/mdl-linreg-oo-classes.svg)
:label:`fig_oo_design`

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab mxnet
import time
import numpy as np
from d2l import mxnet as d2l
from mxnet.gluon import nn
```

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab pytorch
import time
import numpy as np
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab tensorflow
import time
import numpy as np
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #oo-design-object-oriented-design-for-implementation}
%%tab jax
from dataclasses import field
from d2l import jax as d2l
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp
import numpy as np
import jax
import time
from typing import Any
```

## Utilities
:label:`oo-design-utilities`

We need a few utilities to simplify object-oriented programming in Jupyter notebooks. One of the challenges is that class definitions tend to be fairly long blocks of code. Notebook readability demands short code fragments, interspersed with explanations, a requirement incompatible with the style of programming common for Python libraries. The first
utility function allows us to register functions as methods in a class *after* the class has been created. In fact, we can do so *even after* we have created instances of the class! It allows us to split the implementation of a class into multiple code blocks.

```{.python .input #oo-design-utilities-1}
def add_to_class(Class):  #@save
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        return obj
    return wrapper
```

Let's have a quick look at how to use it. We plan to implement a class `A` with a method `do`. Instead of having code for both `A` and `do` in the same code block, we can first declare the class `A` and create an instance `a`.

```{.python .input #oo-design-utilities-2}
class A:
    def __init__(self):
        self.b = 1

a = A()
```

Next we define the method `do` as we normally would, but not in class `A`'s scope. Instead, we decorate this method by `add_to_class` with class `A` as its argument. In doing so, the method is able to access the member variables of `A` just as we would expect had it been included as part of `A`'s definition. Let's see what happens when we invoke it for the instance `a`.

```{.python .input #oo-design-utilities-3}
@add_to_class(A)
def do(self):
    print('Class attribute "b" is', self.b)

a.do()
```

The second one is a utility class that saves all arguments in a class's `__init__` method as class attributes. This allows us to extend constructor call signatures implicitly without additional code.

```{.python .input #oo-design-utilities-4}
class HyperParameters:  #@save
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplementedError
```

The stub above fixes the *interface*; :numref:`sec_utils` fills in the implementation on this same class, which is why saving even a `NotImplementedError` body with `#@save` is worthwhile. To use it, we define our class that inherits from `HyperParameters` and calls `save_hyperparameters` in the `__init__` method.

```{.python .input #oo-design-utilities-5}
# Call the fully implemented HyperParameters class saved in d2l
class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])
        print('self.a =', self.a, 'self.b =', self.b)
        print('There is no self.c =', not hasattr(self, 'c'))

b = B(a=1, b=2, c=3)
```

The final utility allows us to plot experiment progress interactively while it is going on. In deference to the much more powerful (and complex) [TensorBoard](https://www.tensorflow.org/tensorboard) we name it `ProgressBoard`. The  implementation is deferred to :numref:`sec_utils`. For now, let's simply see it in action.

The `draw` method records a point `(x, y)` to be shown in the figure, with `label` specified in the legend. The optional `every_n` smooths the line: it shows one point per $n$ calls to `draw`, plotting the average of the last $n$ recorded values. As we explain just below, `draw` is *asynchronous*: it merely schedules the point and returns immediately, so that plotting never slows down training.

```{.python .input #oo-design-utilities-6}
class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplementedError
```

Why does `draw` go to the trouble of *scheduling* work instead of plotting
right away? The answer previews a theme that runs through the entire book.
Modern frameworks get their speed by *compiling* the training computation into a
graph that runs on the accelerator with little Python in the loop. Compilation
imposes two rules we must respect. First, a compiled step has to be *pure*: a
`print` or a plotting call inside it cannot be captured by the compiler, forcing a
fallback to slower eager execution. Second, the accelerator runs *asynchronously*,
ahead of Python; the instant we ask for a concrete number (to print or plot it),
Python must *block* until the device catches up, stalling the very pipeline we
worked to speed up.

Real-time monitoring therefore seems to be at odds with efficiency. `ProgressBoard`
resolves the tension by *decoupling* the two: `draw` hands the value to a queue and
returns at once, while a background thread performs the device-to-host transfer and
the (comparatively slow) rendering at its own pace, dropping points if it falls
behind, since a live loss curve needs no more than a few updates per second. The
training loop stays compiled and the device stays busy, yet we still watch the loss
go down as it happens. The pattern is general: keep the hot path pure and compiled,
and push logging, plotting, and checkpointing off to the side.

In the following example, we draw `sin` and `cos` with different smoothness. If you
run this code block interactively, you will see the lines grow in animation. Because
drawing is asynchronous, we finish with `board.flush()`, which waits for the queue
to drain and renders the final figure; `Trainer.fit` (developed below) does this for
you.

```{.python .input #oo-design-utilities-7}
board = d2l.ProgressBoard('x')
for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)
board.flush()  # wait for the queued points, then render the final figure
```

## Models
:label:`subsec_oo-design-models`

The `Module` class is the base class of all models we will implement. At the very least we need three methods. The first, `__init__`, stores the learnable parameters, the `training_step` method accepts a data batch to return the loss value, and finally, `configure_optimizers` returns the optimization method, or a list of them, that is used to update the learnable parameters. Optionally we can define `validation_step` to report the evaluation measures.
Sometimes we put the code for computing the output into a separate `forward` method to make it more reusable.
The `plot` method converts the trainer's batch
counter into a *fractional epoch* for the x-coordinate, so that the training
loss (recorded several times per epoch) and the validation loss (recorded once
per epoch) can share a single x-axis, with `every_n` chosen so each curve shows
a fixed number of points per epoch.

:begin_tab:`jax`
With the introduction of [dataclasses](https://docs.python.org/3/library/dataclasses.html)
in Python 3.7, classes decorated with `@dataclass` automatically add magic
methods such as `__init__` and `__repr__`. The member variables are defined
using type annotations. All Flax modules are Python 3.7 dataclasses.
:end_tab:

```{.python .input #oo-design-models}
%%tab pytorch
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        # Pass a thunk, not the value: the device-to-host transfer is deferred
        # to the board's drawing thread so this loop never blocks on it.
        self.board.draw(x, lambda v=value: d2l.numpy(d2l.to(v, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input #oo-design-models}
%%tab mxnet
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        # MXNet's engine is NOT safe for GPU->host transfers issued from a
        # foreign thread (under load it corrupts the CUDA context -> error 999).
        # So resolve the value on THIS (main) thread and enqueue the host
        # scalar; the board's drawing thread then does only matplotlib,
        # never an MXNet GPU op.
        self.board.draw(x, d2l.numpy(value), (
            'train_' if train else 'val_') + key, every_n=int(n))
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input #oo-design-models}
%%tab tensorflow
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
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
        return self.forward(X, *args)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        # Defer the device-to-host transfer to the board's drawing thread.
        self.board.draw(x, lambda v=value: d2l.numpy(v), (
            'train_' if train else 'val_') + key, every_n=int(n))
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=True)
        return l

    def _report_train(self, loss):
        self.plot('loss', loss, train=True)

    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('loss', l, train=False)

    def _report_val(self, y_hat, batch):
        self.plot('loss', self.loss(y_hat, batch[-1]), train=False)

    def configure_optimizers(self):
        raise NotImplementedError
```

```{.python .input #oo-design-models}
%%tab jax
class Module(d2l.nn_Module, d2l.HyperParameters):  #@save
    """The base class of models."""
    # No need for save_hyperparam when using Python dataclass
    plot_train_per_epoch: int = field(default=2, init=False)
    plot_valid_per_epoch: int = field(default=1, init=False)
    # Use default_factory to make sure new plots are generated on each run
    board: ProgressBoard = field(default_factory=lambda: ProgressBoard(),
                                 init=False)

    def loss(self, y_hat, y):
        raise NotImplementedError

    # JAX & Flax do not have a forward-method-like syntax. Flax uses setup
    # and built-in __call__ magic methods for forward pass. Adding here
    # for consistency
    def forward(self, X, *args, **kwargs):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X, *args, **kwargs)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        # Defer the device-to-host transfer to the board's drawing thread, so
        # this loop never blocks on it (the board itself filters via every_n).
        self.board.draw(x, lambda v=value: d2l.numpy(d2l.to(v, d2l.cpu())),
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, params, batch, state):
        l, grads = jax.value_and_grad(self.loss)(params, batch[:-1],
                                                 batch[-1], state)
        self.plot("loss", l, train=True)
        return l, grads

    def validation_step(self, params, batch, state):
        l = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        
    def apply_init(self, dummy_input, key):
        """To be defined later in :numref:`sec_lazy_init`"""
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
```

:begin_tab:`mxnet`
You may notice that `Module` is a subclass of `nn.Block`, the base class of neural networks in Gluon.
It provides convenient features for handling neural networks. For example, if we define a `forward` method, such as `forward(self, X)`, then for an instance `a` we can invoke this method by `a(X)`. This works since it calls the `forward` method in the built-in `__call__` method. You can find more details and examples about `nn.Block` in :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`pytorch`
You may notice that `Module` is a subclass of `nn.Module`, the base class of neural networks in PyTorch.
It provides convenient features for handling neural networks. For example, if we define a `forward` method, such as `forward(self, X)`, then for an instance `a` we can invoke this method by `a(X)`. This works since it calls the `forward` method in the built-in `__call__` method. You can find more details and examples about `nn.Module` in :numref:`sec_model_construction`.
:end_tab:

:begin_tab:`tensorflow`
You may notice that `Module` is a subclass of `tf.keras.Model`, the base class of neural networks in TensorFlow.
It provides convenient features for handling neural networks. For example, it invokes the `call` method in the built-in `__call__` method. Here we redirect `call` to the `forward` method, saving its arguments as a class attribute, consistent with the `forward` convention used elsewhere in the book.
Note that in `__init__` we remove the `loss` instance attribute
that Keras 3 sets to `None`,
since it would shadow the `loss` method
defined by our subclasses.
:end_tab:

:begin_tab:`jax`
You may notice that `Module` is a subclass of `linen.Module`, the base class of neural networks in Flax.
It provides convenient features for handling neural networks. For example, it handles the model parameters, provides the `nn.compact` decorator to simplify code, invokes the `__call__` method among other things.
Here we also redirect `__call__` to the `forward` method, consistent with the `forward` convention used elsewhere in the book.
:end_tab:

##  Data
:label:`oo-design-data`

The `DataModule` class is the base class for data. Quite frequently the `__init__` method is used to prepare the data. This includes downloading and preprocessing if needed. The `train_dataloader` returns the data loader for the training dataset. A data loader is a (Python) generator that yields a data batch each time it is used. This batch is then fed into the `training_step` method of `Module` to compute the loss. There is an optional `val_dataloader` to return the validation dataset loader. It behaves in the same manner, except that it yields data batches for the `validation_step` method in `Module`.

```{.python .input #oo-design-data}
%%tab pytorch
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

```{.python .input #oo-design-data}
%%tab tensorflow
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

```{.python .input #oo-design-data}
%%tab jax
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

```{.python .input #oo-design-data}
%%tab mxnet
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

## Training
:label:`oo-design-training`

:begin_tab:`pytorch, mxnet, tensorflow`
The `Trainer` class trains the learnable parameters in the `Module` class with data specified in `DataModule`. The key method is `fit`, which accepts two arguments: `model`, an instance of `Module`, and `data`, an instance of `DataModule`. It iterates over the entire dataset `max_epochs` times to train the model. Note that `fit_epoch` is left abstract here; it is implemented just two sections later, in :numref:`sec_linear_scratch`.
:end_tab:

:begin_tab:`tensorflow`
Before the first epoch, `fit` calls `_compile_steps`
to wrap the training and validation steps
with `tf.function`.
This compiles them into fused computational graphs
that TensorFlow can execute far more efficiently
than dispatching each operation one at a time
in Python.
We will discuss `_compile_steps` in detail
in :numref:`sec_linear_scratch`.
:end_tab:

:begin_tab:`jax`
The `Trainer` class trains the learnable parameters `params` with data specified in `DataModule`. The key method is `fit`, which accepts three arguments: `model`, an instance of `Module`, `data`, an instance of `DataModule`, and `key`, a JAX `PRNGKeyArray`. We make the `key` argument optional here to simplify the interface, but it is recommended to always pass and initialize the model parameters with a root key in JAX and Flax. It iterates over the entire dataset `max_epochs` times to train the model. Note that `fit_epoch` is left abstract here; it is implemented just two sections later, in :numref:`sec_linear_scratch`.
:end_tab:

```{.python .input #oo-design-training}
%%tab pytorch
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        self.model.board.flush()  # drain queued points; render the final figure

    def fit_epoch(self):
        raise NotImplementedError
```

```{.python .input #oo-design-training}
%%tab tensorflow
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        self._compile_steps()
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        self.model.board.flush()  # drain queued points; render the final figure

    def fit_epoch(self):
        raise NotImplementedError
```

```{.python .input #oo-design-training}
%%tab jax
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data, key=None):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()

        if key is None:
            root_key = d2l.get_key()
        else:
            root_key = key
        params_key, dropout_key = jax.random.split(root_key)
        key = {'params': params_key, 'dropout': dropout_key}

        dummy_input = next(iter(self.train_dataloader))[:-1]
        variables = model.apply_init(dummy_input, key=key)
        params = variables['params']

        if 'batch_stats' in variables.keys():
            # Here batch_stats will be used later (e.g., for batch norm)
            batch_stats = variables['batch_stats']
        else:
            batch_stats = {}

        # Flax uses optax under the hood for a single state obj TrainState.
        # More will be discussed later in the dropout and batch
        # normalization section
        class TrainState(train_state.TrainState):
            batch_stats: Any
            dropout_rng: jax.Array

        self.state = TrainState.create(apply_fn=model.apply,
                                       params=params,
                                       batch_stats=batch_stats,
                                       dropout_rng=dropout_key,
                                       tx=model.configure_optimizers())
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        self.model.board.flush()  # drain queued points; render the final figure

    def fit_epoch(self):
        raise NotImplementedError
```

```{.python .input #oo-design-training}
%%tab mxnet
class Trainer(d2l.HyperParameters):  #@save
    """The base class for training models with data."""
    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        assert num_gpus == 0, 'No GPU support yet'

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()
        self.num_train_batches = len(self.train_dataloader)
        self.num_val_batches = (len(self.val_dataloader)
                                if self.val_dataloader is not None else 0)

    def prepare_model(self, model):
        model.trainer = self
        model.board.xlim = [0, self.max_epochs]
        self.model = model

    def fit(self, model, data):
        self.prepare_data(data)
        self.prepare_model(model)
        self.optim = model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.max_epochs):
            self.fit_epoch()
        self.model.board.flush()  # drain queued points; render the final figure

    def fit_epoch(self):
        raise NotImplementedError
```

## Summary

The classes above sketch the object-oriented design
for our deep learning implementations: how their objects
store data and interact with each other.
We will keep enriching implementations of these classes,
such as via `@add_to_class`,
in the rest of the book.
Moreover,
these fully implemented classes
are saved in the [D2L library](https://github.com/d2l-ai/d2l-en/tree/master/d2l),
a *lightweight toolkit* that makes structured modeling for deep learning easy. 
In particular, it facilitates reusing many components between projects without changing much at all. This modularity keeps implementations concise: you can swap just the optimizer, the model, or the dataset, and it can do the same for your own projects. 


## Exercises

1. The `add_to_class` decorator works by calling `setattr(Class, obj.__name__, obj)`. (a) Add a method `greet(self)` to the existing class `A` *after* the instance `a` has been created, using `@add_to_class(A)`, and verify that `a.greet()` works. (b) What happens if you define `greet` *without* the decorator and then call `a.greet()`? Why?
1. The `Module` class keeps the optimizer in `configure_optimizers`, a *method of the model*, rather than passing it as an argument to `Trainer`. What are the advantages of this design choice? Can you think of a case where putting the optimizer on the model is awkward?
1. Extend `DataModule` with a `test_dataloader` method and extend `Trainer.fit` to run a final evaluation pass on the test set after training. What invariant must the test loader satisfy that the validation loader need not?
1. The `save_hyperparameters` implementation uses Python's `inspect` module to capture the caller's local variables. Can you implement a version that does *not* use `inspect`, for example by requiring the caller to pass the local namespace explicitly? What are the trade-offs?
1. (Advanced) The `ProgressBoard.draw` method is *asynchronous*: it hands values to a background thread rather than plotting immediately. Sketch a synchronous alternative. Under what conditions would the synchronous version be slower? When would the two perform identically?
1. Remove the `save_hyperparameters` statement in the `B` class. Can you still print `self.a` and `self.b`? Optional: if you have studied the full implementation of the `HyperParameters` class, can you explain why?

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
[Dive into Deep Learning · §3.2]{.kicker}

Write the training loop *once*<br>**let every new model and dataset be a subclass · Module · DataModule · Trainer**.
:::
:::

::: {.slide title="One loop, written once"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
Almost every model in this book runs the **same loop**: load a batch,
forward, compute loss, update, repeat.

Rewrite that loop per model and one tweak (gradient clipping, an LR
schedule) means touching *every* chapter. Instead, factor it into three
collaborating classes:

::: {.d2l-note}
**`Module`** is the model · **`DataModule`** is the data ·
**`Trainer`** owns the loop. New work = a new *subclass*.
:::
:::

::: {.col .fig .big}
![](../img/mdl-linreg-oo-classes.svg)
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Notebook-friendly utilities]{.dtitle}

[three helpers that make classes teachable]{.dsub}
:::
:::

::: {.slide title="Define a class, then grow it"}
[Utilities]{.kicker}

A notebook wants short cells, so declare the **shell** first and instantiate it...

@oo-design-utilities-2

. . .

...then attach a method later with `@add_to_class`, which `setattr`s it onto the class, so the bound method sees `self`:

@oo-design-utilities-3
:::

::: {.slide title="`add_to_class`, in three lines"}
[Utilities]{.kicker}

::: {.cols .vc}
::: {.col}
The whole trick: a decorator that writes the function onto a class
object. Python's class namespace is mutable, so this works even on a
class that already has instances.

@oo-design-utilities-1
:::

::: {.col .narrow}
::: {.d2l-note .rule}
We use it throughout the book to split one class across several cells,
each next to the prose that explains it.
:::
:::
:::
:::

::: {.slide title="Stop hand-copying constructor args"}
[Utilities]{.kicker}

Every `__init__` is full of `self.lr = lr; self.n = n; ...`. The
`HyperParameters` mixin captures the caller's arguments and saves them
as attributes automatically:

@oo-design-utilities-5

. . .

One `save_hyperparameters()` call and `self.a`, `self.b` exist; an
`ignore=` list opts arguments out. (Full implementation in the
Utilities appendix.)
:::

::: {.slide title="`ProgressBoard`: the loss curve, animated"}
[Utilities]{.kicker}

::: {.cols .vc}
::: {.col}
`draw(x, y, label)` records a point and the curve grows as training
runs; `every_n` thins a noisy series by plotting the average of the last
$n$ values:

@-oo-design-utilities-7
:::

::: {.col .fig}
@!oo-design-utilities-7
:::
:::

::: {.d2l-note}
Why `draw` merely *schedules* the point (and `flush()` waits for the
queue) is the point of the next slide.
:::
:::

::: {.slide title="Watching the loss live should be impossible"}
[Utilities · compilation & async]{.kicker}

Frameworks earn their speed by **compiling** the training step into a
graph and letting the device run **ahead** of Python. That imposes two
rules:

- A compiled step must be **pure**: a `print` or plot inside it cannot
  be captured by the compiler, forcing a fallback to slower eager execution.
- The instant Python asks for a concrete number, it must **block** until
  the device catches up, stalling the very pipeline we built.

. . .

::: {.d2l-note .warn}
So every naïve "plot the loss each batch" either breaks the compiled
graph or drains the device pipeline. Real-time monitoring and efficiency
seem to be at war.
:::
:::

::: {.slide title="Resolution: queue now, render elsewhere"}
[Utilities · compilation & async]{.kicker}

`ProgressBoard` decouples the two: `draw` hands the value to a **queue**
and returns at once; a background thread does the device-to-host copy
and the slow matplotlib rendering at its own pace, dropping points if
it falls behind, since a live curve needs only a few updates per second.

. . .

The training loop stays compiled, the device stays busy, and the loss
still falls before your eyes.

::: {.d2l-note .rule}
The pattern to remember, book-wide: **keep the hot path pure and
compiled; push logging, plotting, and checkpointing off to the side.**
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The three base classes]{.dtitle}

[Module · DataModule · Trainer]{.dsub}
:::
:::

::: {.slide title="`Module`: the model, its loss, its optimizer" except="jax"}
[Base classes]{.kicker}

::: {.cols .vc}
::: {.col}
Every model subclasses `Module` and supplies three things:

- **`forward`** / `loss`: the prediction and how wrong it is.
- **`training_step`**: loss on one batch (plots it for free).
- **`configure_optimizers`**: the optimizer to use.

::: {.d2l-note}
`Module` extends the framework's own neural-network base class, so an
instance is callable: `model(X)` runs `forward`.
:::
:::

::: {.col .fig}
![](../img/mdl-linreg-oo-classes.svg)
:::
:::
:::

::: {.slide title="`Module`: the same contract, written functionally" only="jax"}
[Base classes]{.kicker}

::: {.cols .vc}
::: {.col}
In Flax a module **is a dataclass**: no `__init__`, fields declared by
type annotation. The same contract, written functionally:

- `forward` / `__call__`: the prediction.
- **`training_step`** returns *(loss, grads)* via
  `jax.value_and_grad`, since parameters are passed in, not stored.
- `configure_optimizers`: the optax optimizer.

::: {.d2l-note .rule}
Parameters live *outside* the module; every step is a pure function of
them.
:::
:::

::: {.col .fig}
![](../img/mdl-linreg-oo-classes.svg)
:::
:::
:::

::: {.slide title="`DataModule`: where batches come from"}
[Base classes]{.kicker}

A `DataModule` serves a **train** and a **validation** loader, both
through one `get_dataloader(train)` hook that subclasses override. This
is the *entire* base class:

@oo-design-data

::: {.d2l-note}
A loader is a generator yielding one batch at a time, fed straight into
`Module.training_step`.
:::
:::

::: {.slide title="`Trainer`: it owns the loop" except="jax"}
[Base classes]{.kicker}

::: {.cols .vc}
::: {.col}
`fit(model, data)` wires the two together: prepare the loaders, hand the
optimizer over, then run `fit_epoch` for `max_epochs`. The body is short:

```{.python #oo-design-exercises-1}
def fit(self, model, data):
    self.prepare_data(data)
    self.prepare_model(model)
    self.optim = model.configure_optimizers()
    for self.epoch in range(self.max_epochs):
        self.fit_epoch()
```

::: {.d2l-note}
`fit_epoch` stays abstract here; we enrich `Trainer` for GPUs and
parallel training in later chapters.
:::
:::

::: {.col .fig}
![](../img/mdl-linreg-oo-classes.svg)
:::
:::
:::

::: {.slide title="`Trainer`: threading state, the JAX way" only="jax"}
[Base classes]{.kicker}

::: {.cols .vc}
::: {.col}
JAX has no mutable `self.params`, so `fit` takes an explicit PRNG `key`,
splits it for init and dropout, and bundles parameters, optimizer, and
RNG into a single immutable `TrainState`:

::: {.d2l-note .rule}
Same `fit(model, data)` contract; the state is passed *through* each
step rather than stored on the object.
:::
:::

::: {.col .narrow}
```{.python #oo-design-exercises-2}
root = key or d2l.get_key()
p_key, d_key = jax.random.split(root)
params = model.apply_init(
    dummy, key=...)['params']
self.state = TrainState.create(
    apply_fn=model.apply,
    params=params, tx=optim, ...)
```
:::
:::
:::

::: {.slide title="Recap" except="jax"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Three classes** scaffold every model: `Module` (the model),
  `DataModule` (the data), `Trainer` (the loop).
- New model or dataset = a **subclass**; the loop is written once.
- `add_to_class` splits a class across notebook cells;
  `HyperParameters` kills `__init__` boilerplate.
:::

::: {.col}
- `ProgressBoard` plots the loss live yet never blocks: **keep the hot
  path pure and compiled; push logging off to the side**, a theme that
  recurs all book.
:::
:::
:::

::: {.slide title="Recap" only="jax"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Three classes** scaffold every model: `Module` (the model),
  `DataModule` (the data), `Trainer` (the loop).
- New model or dataset = a **subclass**; the loop is written once.
- `add_to_class` splits a class across notebook cells;
  `HyperParameters` kills `__init__` boilerplate.
:::

::: {.col}
- `ProgressBoard` plots the loss live yet never blocks: **keep the hot
  path pure and compiled; push logging off to the side**, a theme that
  recurs all book.
- **Watch the framing:** JAX is functional (a dataclass `Module`,
  parameters and a `TrainState` threaded through `fit`).
:::
:::
:::
