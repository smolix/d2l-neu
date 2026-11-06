DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import jax
# Force JAX to initialize CUDA (and load its own cuBLAS/cuSOLVER) before any
# subsequent `import tensorflow` pulls in TF's bundled CUDA libraries. Without
# this, TF's older cuBLAS is loaded first and jax.xla_bridge falls back to CPU.
_jax_devices = jax.devices()
_jax_uses_gpu = any(device.platform == 'gpu' for device in _jax_devices)
import tensorflow as _tf
# TF is only used for data loading in JAX notebooks; hide GPUs from it so it
# doesn't fight JAX for the pre-allocated GPU memory (avoids transient
# "Dst tensor is not initialized" OOMs seen when multiple notebooks share a
# device).
if _jax_uses_gpu:
    try:
        _tf.config.set_visible_devices([], 'GPU')
    except RuntimeError:
        for _tf_gpu in _tf.config.list_physical_devices('GPU'):
            _tf.config.experimental.set_memory_growth(_tf_gpu, True)
del _jax_devices, _jax_uses_gpu
del _tf
from jax import numpy as jnp
from flax import nnx
import random as _random

# Seed the d2l PRNG once per interpreter, from the operating system's
# entropy, so that every run draws new random numbers, like the unseeded
# samplers of the other frameworks. Every d2l.get_key() call splits and
# advances _master_key, so the whole notebook draws from a single
# continuous stream of uncorrelated sub-keys. Do not build a fresh
# jax.random.PRNGKey(seed) at each call site — independently seeded keys
# can produce correlated samples in the JAX counter-based PRNG.
_master_key = jax.random.PRNGKey(_random.SystemRandom().getrandbits(32))
_seed_rng = _random.Random()


def get_key():
    """Return a fresh sub-key split from the module-level master key."""
    global _master_key
    _master_key, sub = jax.random.split(_master_key)
    return sub


def get_seed():
    """Return a fresh int seed from the d2l stream, for non-JAX APIs that
    want an int (numpy.random.seed, tf.random.set_seed, stdlib random)."""
    return _seed_rng.randint(0, 10**6)

nn_Module = nnx.Module


#################   WARNING   ################
# The below part is generated automatically through:
#    python tools/build_lib.py
# Don't edit it directly

import sys
d2l = sys.modules[__name__]

import inspect
import collections
from collections import defaultdict
from dataclasses import asdict
from IPython import display
import json
import math
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib_inline import backend_inline
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import gzip
import hashlib
d2l = sys.modules[__name__]

from functools import partial
from flax import nnx
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
from typing import Any, Callable

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_log_ticks(axis):
    """Label all major ticks of a log axis in one format.

    Defined in :numref:`sec_calculus`"""
    lo, hi = axis.get_view_interval()
    if 0 < lo and hi < 100 * lo:  # Under two decades: ticks at 1, 2, 5
        axis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ticks = axis.get_major_locator().tick_values(lo, hi)
    plain = all(1e-4 <= t < 1e5 for t in ticks if lo <= t <= hi)

    def label(v, _):
        if plain:  # 0.001, 0.01, ..., 10000
            return f'{v:g}'
        m, k = f'{v:.3e}'.split('e')  # 2.500e-05 -> '2.500', '-05'
        m, power = f'{float(m):g}', f'10^{{{int(k)}}}'  # '2.5', '10^{-5}'
        return f'${power}$' if m == '1' else f'${m}\\times{power}$'

    axis.set_major_formatter(ticker.FuncFormatter(label))
    axis.set_minor_formatter(ticker.NullFormatter())

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    for axis, scale in ((axes.xaxis, xscale), (axes.yaxis, yscale)):
        if scale == 'log':
            set_log_ticks(axis)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""
    legend = [] if legend is None else legend

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    # Extend the formats with new colors so that no curve is dropped
    fmts = list(fmts) + [f'C{i % 10}' + ('-', '--', '-.', ':')[i % 4]
                         for i in range(len(fmts), len(Y))]
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

def add_to_class(Class):
    """Register functions as methods in created class.

    Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        return obj
    return wrapper

class HyperParameters:
    """The base class of hyperparameters.

    Defined in :numref:`sec_oo-design`"""

    def save_hyperparameters(self, ignore=None):
        """Save function arguments into instance attributes.

        Defined in :numref:`sec_utils`"""
        ignore = [] if ignore is None else ignore
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class Module(d2l.nn_Module, d2l.HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X, *args, **kwargs):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X, *args, **kwargs)

    def __call__(self, X, *args, **kwargs):
        return self.forward(X, *args, **kwargs)

    def training_step(self, batch):
        """The loss on one batch, or a dict of metrics with key 'loss'."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def validation_step(self, batch):
        """The loss on one batch, or a dict of batch-mean metrics."""
        return self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        return optax.sgd(self.lr)

def as_metrics(out):
    """A step returns a loss or a dict of metrics; return a dict.

    Defined in :numref:`sec_oo-design`"""
    return out if isinstance(out, dict) else {'loss': out}

class DataModule(d2l.HyperParameters):
    """The base class of data.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, root='../data', num_workers=4):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        return ArrayLoader(tensors, self.batch_size, shuffle=train)

class Trainer(d2l.HyperParameters):
    """Fit a Module to a DataModule, keeping a per-epoch history.

    Defined in :numref:`sec_oo-design`"""

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

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

    def record(self, split, metrics, x):
        for name in sorted(metrics, key=lambda name: (name != 'loss', name)):
            self.history[f'{split}_{name}'].append(metrics[name])
            self.board.draw(x, metrics[name], f'{split}_{name}')

    def evaluate(self, model, data):
        """Validation metrics of `model` on `data`, averaged over examples.

        Defined in :numref:`sec_oo-design`"""
        self.train_dataloader = None
        self.val_dataloader = data.val_dataloader()
        self.prepare_model(model)
        return self.validate()

    def prepare_model(self, model):
        self.model = model
        self.train_model = nnx.view(model, deterministic=False,
                                    use_running_average=False,
                                    raise_if_not_found=False)
        self.val_model = nnx.view(model, deterministic=True,
                                  use_running_average=True,
                                  raise_if_not_found=False)

    def fit_epoch(self):
        metrics = d2l.Averager()
        for batch in map(self.prepare_batch, self.train_dataloader):
            metrics.add(_trainer_train_step(self.train_model, self.optim, batch),
                        len(batch[-1]))
        return metrics.mean()

    def validate(self):
        metrics = d2l.Averager()
        for batch in map(self.prepare_batch, self.val_dataloader):
            metrics.add(_trainer_validation_step(self.val_model, batch),
                        len(batch[-1]))
        return metrics.mean()

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
        self.board = d2l.ProgressBoard(xlabel='epoch', ylabel='loss')

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return batch

    def prepare_optimizer(self, model):
        val = self.gradient_clip_val
        key = (jax.tree.structure(model), type(model).configure_optimizers, val)
        if key not in d2l.optimizer_cache:
            tx = model.configure_optimizers()
            if val > 0:  # The closure holds only val, not the trainer
                clip = lambda grads, params: clip_gradients(val, grads)
                tx = optax.chain(optax.stateless(clip), tx)
            d2l.optimizer_cache.clear()
            d2l.optimizer_cache[key] = tx
        self.optim = nnx.Optimizer(model, d2l.optimizer_cache[key],
                                   wrt=nnx.Param)

class ProgressBoard(d2l.HyperParameters):
    """Curves that grow during training; `show` redraws them.

    Defined in :numref:`sec_oo-design`"""
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
        """Redraw all curves; the book build renders only the final call.

        Defined in :numref:`sec_utils`"""
        if not self.display or (
                not final and os.environ.get('D2L_NB_CAPTURE') == '1'):
            return
        d2l.use_svg_display()
        fig = Figure(figsize=self.figsize)
        self._plot(fig.add_subplot())
        if self.handle is None:
            self.handle = display.display(fig, display_id=True)
        else:
            self.handle.update(fig)

    def _plot(self, ax):
        """Losses on the left axis, accuracies on a right one.

        Defined in :numref:`sec_utils`"""
        acc_ax = None
        for i, (label, (xs, ys)) in enumerate(self.data.items()):
            target = ax
            if label.endswith('acc'):
                acc_ax = acc_ax or ax.twinx()
                target = acc_ax
            # Mark validation points, and any curve that has a single point
            marked = label.startswith('val') or len(xs) == 1
            target.plot(xs, ys, ('-', '--', '-.', ':')[i % 4], color=f'C{i % 10}',
                        marker='o' if marked else None, markersize=3,
                        clip_on=not marked, label=label)
        ax.set_xlabel(self.xlabel), ax.set_ylabel(self.ylabel)
        ax.set_xscale(self.xscale), ax.set_yscale(self.yscale)
        if self.xlim: ax.set_xlim(self.xlim)
        if self.ylim: ax.set_ylim(self.ylim)
        if self.yscale == 'log': d2l.set_log_ticks(ax.yaxis)
        if acc_ax is not None: acc_ax.set_ylabel('accuracy')
        lines = list(ax.get_lines()) + list(acc_ax.get_lines() if acc_ax else [])
        labels = [line.get_label() for line in lines]
        if acc_ax is not None:  # Above the axes: no curve of either axis is hidden
            ax.legend(lines, labels, loc='lower center', ncol=len(lines),
                      bbox_to_anchor=(0.5, 1), frameon=False)
        elif lines:
            ax.legend(lines, labels)
        ax.grid()

optimizer_cache = {}  # The most recent optax transformation, by its key

class SyntheticRegressionData(d2l.DataModule):
    """Synthetic data for linear regression.

    Defined in :numref:`sec_synthetic-regression-data`"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32, key=None):
        super().__init__()
        self.save_hyperparameters()
        if key is None:  # A new seed on each run
            key = jax.random.key(random.getrandbits(32))
        n = num_train + num_val
        key1, key2 = jax.random.split(key)
        self.X = jax.random.normal(key1, (n, w.shape[0]))
        eps = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + eps

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

class ArrayLoader:
    """Minibatches of in-memory arrays, reshuffled each pass if shuffle.

    Defined in :numref:`sec_synthetic-regression-data`"""
    def __init__(self, arrays, batch_size, shuffle=False, drop_last=False,
                 key=None):
        self.arrays = tuple(np.asarray(a) for a in arrays)  # On the host
        self.batch_size, self.shuffle = batch_size, shuffle
        self.drop_last = drop_last
        if shuffle and key is None:  # A new seed on each run
            key = jax.random.key(random.getrandbits(32))
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

class LinearRegressionScratch(d2l.Module):
    """The linear regression model implemented from scratch.

    Defined in :numref:`sec_linear_scratch`"""
    def __init__(self, num_inputs, lr, sigma=0.01, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.w = nnx.Param(
            rngs.params.normal((num_inputs, 1)) * sigma)
        self.b = nnx.Param(jnp.zeros(1))

    def forward(self, X):
        return d2l.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2
        return d2l.reduce_mean(l)

    def configure_optimizers(self):
        return SGD(self.lr)

class SGD(d2l.HyperParameters):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_linear_scratch`"""
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

class Averager:
    """Example-weighted means of batch metrics, summed on the device.

    Defined in :numref:`sec_linear_scratch`"""
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

@nnx.jit
def _trainer_train_step(model, optimizer, batch):
    def loss_fn(model):
        out = d2l.as_metrics(model.training_step(batch))
        return out['loss'], out
    grads, out = nnx.grad(loss_fn, has_aux=True)(model)
    optimizer.update(model, grads)
    return out

@nnx.jit
def _trainer_validation_step(model, batch):
    return d2l.as_metrics(model.validation_step(batch))

class LinearRegression(d2l.Module):
    """Linear regression built from framework components.

    Defined in :numref:`sec_linear_concise`"""
    def __init__(self, num_inputs, lr, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(
            num_inputs, 1, kernel_init=nnx.initializers.normal(0.01),
            rngs=rngs)

    def loss(self, y_hat, y):
        return optax.squared_error(y_hat, y).mean()

    def configure_optimizers(self):
        return optax.sgd(self.lr)

    def get_w_b(self):
        return self.net.kernel[...], self.net.bias[...]

class FashionMNIST(d2l.DataModule):
    """The Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()

    def text_labels(self, indices):
        """Return text labels.

        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        X, y = self.train if train else self.val
        X = tf.expand_dims(X, axis=3) / 255  # Channel axis last; values in [0, 1]
        resize = lambda X, y: (tf.image.resize(X, self.resize), y)
        return TensorFlowDataLoader(tf.data.Dataset.from_tensor_slices(
            (X, tf.cast(y, tf.int32))).shuffle(len(y) if train else 1).batch(
                self.batch_size).map(resize))

    def visualize(self, batch, nrows=1, ncols=8, labels=None):
        """Show the images of a batch in grayscale, titled by their labels.

        Defined in :numref:`sec_fashion_mnist`"""
        X, y = batch
        if labels is None:
            labels = self.text_labels(y)
        X = d2l.reshape(X, (-1, *self.resize))  # Drop the channel axis
        d2l.show_images(X, nrows, ncols, titles=labels, cmap='gray')

class TensorFlowDataLoader:
    """Expose a tf.data.Dataset as re-iterable NumPy batches.

    Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.as_numpy_iterator()

    def __len__(self):
        return len(self.dataset)

class Classifier(d2l.Module):
    """The base class of classification models.

    Defined in :numref:`sec_classification`"""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        return {'loss': self.loss(Y_hat, batch[-1]),
                'acc': self.accuracy(Y_hat, batch[-1])}

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the fraction of correct predictions.

        Defined in :numref:`sec_classification`"""
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
        compare = d2l.astype(preds == d2l.reshape(Y, (-1,)), d2l.float32)
        return d2l.reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = d2l.reshape(Y, (-1,))
        fn = optax.softmax_cross_entropy_with_integer_labels
        return fn(Y_hat, Y).mean() if averaged else fn(Y_hat, Y)

    def layer_summary(self, X_shape):
        X = jnp.zeros(X_shape)
        for layer in self.net.layers:
            X = layer(X)
            # Activations and pooling are plain functions: print their names
            name = getattr(layer, '__name__', type(layer).__name__)
            print(f'{name:<22} {tuple(X.shape)}')

def cross_entropy(y_hat, y):
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = jnp.clip(y_hat[jnp.arange(len(y_hat)), y], min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

class SoftmaxRegression(d2l.Classifier):
    """The softmax regression model.

    Defined in :numref:`sec_softmax_concise`"""
    def __init__(self, num_outputs, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(num_inputs, num_outputs,
                              kernel_init=nnx.initializers.normal(0.01),
                              rngs=rngs)

    def forward(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        return self.net(X)

def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return jax.devices('cpu')[0]

def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return jax.devices('gpu')[i]

def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    try:
        return jax.device_count('gpu')
    except:
        return 0  # No GPU backend found

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if num_gpus() >= i + 1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.

    Defined in :numref:`sec_use_gpu`"""
    devices = [gpu(i) for i in range(num_gpus())]
    return devices if devices else [cpu()]

class Benchmark:
    """Time a callable: warmup, then device-synchronized average seconds.

    Defined in :numref:`sec_use_gpu`"""
    def __init__(self, f, warmup=3, repeats=10, desc='time'):
        self.desc, out = desc, None
        for _ in range(warmup):
            out = f()
        jax.block_until_ready(out)
        t0 = time.perf_counter()
        for _ in range(repeats):
            out = f()
        jax.block_until_ready(out)
        self.time = (time.perf_counter() - t0) / repeats

    def __repr__(self):
        return f'{self.desc}: {1000 * self.time:.2f} ms/call'

def corr2d(X, K):
    """Compute 2D cross-correlation.

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    Y = jnp.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y = Y.at[i, j].set((X[i:i + h, j:j + w] * K).sum())
    return Y

class LeNet(d2l.Classifier):
    """The layout of LeNet-5, with sigmoid activations and average pooling.

    Defined in :numref:`sec_lenet`"""
    def __init__(self, lr=0.1, num_classes=10, kernel_init=None, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs', 'kernel_init'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        kernel_init = (nnx.initializers.xavier_uniform() if kernel_init is None
                       else kernel_init)

        def avg_pool(x):  # 2x2 windows, stride 2
            return nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2))

        def flatten(x):
            return x.reshape((x.shape[0], -1))

        self.net = nnx.Sequential(
            nnx.Conv(1, 6, kernel_size=(5, 5), padding=2,
                     kernel_init=kernel_init, rngs=rngs), nnx.sigmoid,
            avg_pool,
            nnx.Conv(6, 16, kernel_size=(5, 5), padding='VALID',
                     kernel_init=kernel_init, rngs=rngs), nnx.sigmoid,
            avg_pool,
            flatten,
            nnx.Linear(400, 120, kernel_init=kernel_init, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(120, 84, kernel_init=kernel_init, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(84, num_classes, kernel_init=kernel_init, rngs=rngs))

def cosine_warmup(t, num_steps, warmup_steps):
    """Learning-rate factor: linear warmup from 0, then a half-cosine to 0.

    Defined in :numref:`sec_alexnet`"""
    t = jnp.asarray(t, jnp.float32)  # Arguments may be traced arrays
    warmup = jnp.where(t < warmup_steps, t / warmup_steps, 1.0)
    progress = jnp.clip((t - warmup_steps) / (num_steps - warmup_steps), 0, 1)
    return warmup * 0.5 * (1 + jnp.cos(jnp.pi * progress))

class Residual(nnx.Module):
    """The Residual block of ResNet models.

    Defined in :numref:`sec_resnet`"""
    def __init__(self, num_channels, use_1x1conv=False, strides=(1, 1),
                 in_channels=None, rngs=None):
        in_channels = num_channels if in_channels is None else in_channels
        strides = (strides, strides) if isinstance(strides, int) else strides
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        # He initialization; no conv biases, since the batch normalization
        # after each conv removes them
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        # padding=1 on every side: 'SAME' pads only right and bottom when the
        # stride is 2 and the sizes are even
        self.conv1 = nnx.Conv(in_channels, num_channels, kernel_size=(3, 3),
                              padding=1, strides=strides, use_bias=False,
                              kernel_init=he, rngs=rngs)
        self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3),
                              padding='same', use_bias=False,
                              kernel_init=he, rngs=rngs)
        # Running averages decay 0.9 and a two-pass variance (the Flax
        # defaults are 0.99 and a one-pass variance)
        bn = lambda: nnx.BatchNorm(num_channels, momentum=0.9,
                                   use_fast_variance=False, rngs=rngs)
        # Auto-enable 1x1 conv when downsampling so the residual shape matches.
        if use_1x1conv or any(s != 1 for s in strides):
            self.conv3 = nnx.Conv(in_channels, num_channels,
                                  kernel_size=(1, 1), strides=strides,
                                  use_bias=False, kernel_init=he, rngs=rngs)
            self.bn3 = bn()
        else:
            self.conv3 = None
        self.bn1, self.bn2 = bn(), bn()

    def __call__(self, X):
        Y = nnx.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.bn3(self.conv3(X))
        Y += X
        return nnx.relu(Y)

class ResNeXtBlock(nnx.Module):
    """The ResNeXt block.

    Defined in :numref:`sec_resnet`"""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=(1, 1), in_channels=None, rngs=None):
        in_channels = num_channels if in_channels is None else in_channels
        strides = (strides, strides) if isinstance(strides, int) else strides
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        bot_channels = int(round(num_channels * bot_mul))
        # He initialization; no conv biases, since the batch normalization
        # after each conv removes them
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        self.conv1 = nnx.Conv(in_channels, bot_channels, kernel_size=(1, 1),
                              strides=(1, 1), use_bias=False,
                              kernel_init=he, rngs=rngs)
        self.conv2 = nnx.Conv(bot_channels, bot_channels,
                              kernel_size=(3, 3), strides=strides,
                              padding=1, feature_group_count=groups,
                              use_bias=False, kernel_init=he, rngs=rngs)
        self.conv3 = nnx.Conv(bot_channels, num_channels, kernel_size=(1, 1),
                              strides=(1, 1), use_bias=False,
                              kernel_init=he, rngs=rngs)
        # Running averages decay 0.9 and a two-pass variance (the Flax
        # defaults are 0.99 and a one-pass variance)
        bn = lambda n: nnx.BatchNorm(n, momentum=0.9, use_fast_variance=False,
                                     rngs=rngs)
        self.bn1, self.bn2 = bn(bot_channels), bn(bot_channels)
        self.bn3 = bn(num_channels)
        # Auto-enable 1x1 conv when downsampling so the residual shape matches.
        if use_1x1conv or any(s != 1 for s in strides):
            self.conv4 = nnx.Conv(in_channels, num_channels,
                                  kernel_size=(1, 1), strides=strides,
                                  use_bias=False, kernel_init=he, rngs=rngs)
            self.bn4 = bn(num_channels)
        else:
            self.conv4 = None

    def __call__(self, X):
        Y = nnx.relu(self.bn1(self.conv1(X)))
        Y = nnx.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nnx.relu(Y + X)

def num_params(model):
    """The number of trainable parameters.

    Defined in :numref:`sec_efficient_cnns`"""
    return sum(p.size for _, p in nnx.state(model, nnx.Param).flat_state())

def mixup(key, X, y, alpha):
    """Mix a minibatch with a shuffled copy; return both label sets.

    Defined in :numref:`sec_training_recipes`"""
    key_lam, key_perm = jax.random.split(key)
    lam = jax.random.beta(key_lam, alpha, alpha)
    perm = jax.random.permutation(key_perm, X.shape[0])
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam

def drop_path(key, Y, p, training):
    """Zero each example of Y with probability p, rescale the rest.

    Defined in :numref:`sec_training_recipes`"""
    if not training or p == 0:
        return Y
    shape = (Y.shape[0],) + (1,) * (Y.ndim - 1)  # One draw per example
    keep = jax.random.bernoulli(key, 1 - p, shape)
    return Y * keep / (1 - p)

class TimeMachine(d2l.DataModule):
    """The Time Machine dataset.

    Defined in :numref:`sec_text-sequence`"""
    def _download(self):
        fname = d2l.download(d2l.DATA_URL + 'timemachine.txt', self.root,
                             '090b5e7e70c295757f55df93cb0a180b9691891a')
        with open(fname) as f:
            return f.read()

    def _preprocess(self, text):
        return re.sub('[^A-Za-z]+', ' ', text).lower()

    def _tokenize(self, text):
        return list(text)

    def build(self, raw_text, vocab=None):
        tokens = self._tokenize(self._preprocess(raw_text))
        if vocab is None: vocab = Vocab(tokens)
        corpus = [vocab[token] for token in tokens]
        return corpus, vocab

    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000,
                 tokenization='bpe', vocab_size=1024):
        super(d2l.TimeMachine, self).__init__()
        self.save_hyperparameters()
        raw_text = self._download()
        if tokenization == 'bpe':
            self.tokenizer = d2l.BPETokenizer(
                vocab_size, pattern=d2l.BPETokenizer.GPT2_PATTERN)
            self.tokenizer.train(raw_text)
            corpus, self.vocab = self.tokenizer.encode(raw_text), self.tokenizer
        else:  # 'char': the character-level pipeline
            corpus, self.vocab = self.build(raw_text)
        windows = [corpus[i:i+num_steps+1] for i in range(len(corpus)-num_steps)]
        # Drop the windows that straddle the split, so that no validation token
        # is a training target
        windows = windows[:num_train] + windows[num_train+num_steps:]
        array = d2l.tensor(windows)
        self.X, self.Y = array[:,:-1], array[:,1:]

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

import regex

class BPETokenizer:
    """Byte-level BPE tokenizer trained by iterated most-frequent-pair
    merges.

    Token ids 0..255 are raw bytes; learned merges get ids
    256..vocab_size-1; special tokens sit above those and are never
    produced by BPE itself. `pattern` is an optional pre-tokenization
    regex: text is split into chunks and merges never cross chunk
    boundaries.

    Defined in :numref:`sec_text-sequence`"""

    GPT2_PATTERN = (r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+"
                    r"| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")

    def __init__(self, vocab_size=1024, pattern=None,
                 specials=('<pad>', '<bos>', '<eos>')):
        self.vocab_size, self.pattern = vocab_size, pattern
        self.merges = {}                                  # (id, id) -> new id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes
        self.byte_ids = list(range(256))                  # byte value -> id
        self.specials = {s: vocab_size + i for i, s in enumerate(specials)}

    def __len__(self):
        return self.vocab_size + len(self.specials)

    @property
    def pad(self):
        return self.specials['<pad>']

    @property
    def bos(self):
        return self.specials['<bos>']

    @property
    def eos(self):
        return self.specials['<eos>']

    def _chunks(self, text):
        return [text] if self.pattern is None else regex.findall(
            self.pattern, text)

    def _merge(self, ids, pair, new_id):
        """Replace every occurrence of pair in ids by new_id.

        Defined in :numref:`sec_text-sequence`"""
        out, i = [], 0
        while i < len(ids):
            try:  # list.index scans for the next candidate at C speed
                j = ids.index(pair[0], i)
            except ValueError:
                j = len(ids)
            out.extend(ids[i:j])
            if j < len(ids) - 1 and ids[j + 1] == pair[1]:
                out.append(new_id)
                i = j + 2
            elif j < len(ids):
                out.append(ids[j])
                i = j + 1
            else:
                i = j
        return out

    def train(self, text):
        """Learn vocab_size - 256 merges, most frequent pair first.

        Defined in :numref:`sec_text-sequence`"""
        chunk_freq = collections.Counter(self._chunks(text))
        seqs = [list(chunk.encode('utf-8')) for chunk in chunk_freq]
        for new_id in range(256, self.vocab_size):
            pairs = collections.Counter()
            for seq, w in zip(seqs, chunk_freq.values()):
                if w == 1:  # Counter.update counts at C speed
                    pairs.update(zip(seq, seq[1:]))
                else:  # weight pair counts by chunk frequency
                    for pair in zip(seq, seq[1:]):
                        pairs[pair] += w
            if not pairs:
                break  # nothing left to merge
            pair = max(pairs, key=pairs.get)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            seqs = [self._merge(seq, pair, new_id) for seq in seqs]

    def _encode_chunk(self, text_bytes):
        ids = [self.byte_ids[b] for b in text_bytes]
        while len(ids) > 1:
            # The lowest-rank merge applicable anywhere in this chunk
            pair = min(zip(ids, ids[1:]),
                       key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text, allow_special=False):
        if allow_special and self.specials:
            pat = '(' + '|'.join(regex.escape(s) for s in self.specials) + ')'
            ids = []
            for part in regex.split(pat, text):
                if part in self.specials:
                    ids.append(self.specials[part])
                elif part:
                    ids.extend(self.encode(part))
            return ids
        return [i for chunk in self._chunks(text)
                for i in self._encode_chunk(chunk.encode('utf-8'))]

    def decode(self, ids):
        specials = {i: s.encode('utf-8') for s, i in self.specials.items()}
        data = b''.join(self.vocab[i] if i in self.vocab else specials[i]
                        for i in ids)
        return data.decode('utf-8', errors='replace')

    @classmethod
    def from_tiktoken(cls, name):
        """Load a published bytes->rank table (e.g. 'gpt2') into our encoder.

        Defined in :numref:`sec_text-sequence`"""
        import tiktoken  # lazy: d2l itself does not require tiktoken
        enc = tiktoken.get_encoding(name)
        ranks = enc._mergeable_ranks
        tok = cls(vocab_size=len(ranks), pattern=enc._pat_str, specials=())
        tok.specials = dict(enc._special_tokens)
        tok.vocab = {rank: b for b, rank in ranks.items()}
        tok.byte_ids = [ranks[bytes([b])] for b in range(256)]
        for token, rank in ranks.items():
            if len(token) > 1:  # recover which pair merged into this token
                parts = [bytes([b]) for b in token]
                while len(parts) > 2:
                    a, b = min(zip(parts, parts[1:]),
                               key=lambda p: ranks.get(p[0] + p[1],
                                                       float('inf')))
                    i = list(zip(parts, parts[1:])).index((a, b))
                    parts[i:i + 2] = [a + b]
                tok.merges[ranks[parts[0]], ranks[parts[1]]] = rank
        return tok

    def __getitem__(self, token):
        return self.specials[token]

    def to_tokens(self, ids):
        return [self.decode([int(i)]) for i in ids]

import textwrap

def print_wrapped(*args, width=76):
    """Print like `print`, folding each line to the width of a page.

    Defined in :numref:`sec_text-sequence`"""
    for line in ' '.join(str(a) for a in args).split('\n'):
        print(textwrap.fill(line, width, subsequent_indent='    '))

class Vocab:
    """Vocabulary for text.

    Defined in :numref:`sec_text-sequence`"""
    def __init__(self, tokens=(), min_freq=0, reserved_tokens=()):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens, ordered by descending frequency.
        # Reserve <unk> at index 0 so vocab[0] is the unknown token.
        self.idx_to_token = ['<unk>'] + list(reserved_tokens) + [
            token for token, freq in self.token_freqs
            if freq >= min_freq and token not in reserved_tokens]
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if isinstance(indices, (list, tuple)) or getattr(indices, 'ndim', 0):
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[int(indices)]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

def kneser_ney(train, tokens, vocab_size, n=3):
    """Interpolated Kneser-Ney log-probabilities (nats) of each of tokens.

    Defined in :numref:`sec_language-model`"""
    counts = {n: collections.Counter(zip(*(train[i:] for i in range(n))))}
    for k in range(n - 1, 0, -1):  # Lower orders: continuation counts
        counts[k] = collections.Counter(g[1:] for g in counts[k + 1])
    stats = {}
    for k, c in counts.items():
        c1, c2 = (sum(x == r for x in c.values()) for r in (1, 2))
        N, S = collections.Counter(), collections.Counter()
        for g, x in c.items():
            N[g[:-1]] += x
            S[g[:-1]] += 1
        stats[k] = c, N, S, c1 / max(c1 + 2 * c2, 1)
    logp = []
    for i in range(len(tokens)):
        p = 1 / vocab_size  # Below the unigram: uniform
        for k in range(1, min(n, i + 1) + 1):
            c, N, S, D = stats[k]
            g = tuple(tokens[i - k + 1:i + 1])
            if N[g[:-1]]:  # An unseen context passes p down unchanged
                p = (max(c[g] - D, 0) + D * S[g[:-1]] * p) / N[g[:-1]]
        logp.append(math.log(p))
    return logp

class RNNScratch(nnx.Module):
    """The RNN model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, rngs=None):
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.sigma = sigma
        self.W_xh = nnx.Param(
            rngs.params.normal((num_inputs, num_hiddens)) * sigma)
        self.W_hh = nnx.Param(
            rngs.params.normal((num_hiddens, num_hiddens)) * sigma)
        self.b_h = nnx.Param(jnp.zeros(num_hiddens))

    def __call__(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = jnp.zeros((inputs.shape[1], self.num_hiddens))
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
            state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                             d2l.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

def check_len(a, n):
    """Check the length of a list.

    Defined in :numref:`sec_rnn-scratch`"""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'

def check_shape(a, shape):
    """Check the shape of a tensor.

    Defined in :numref:`sec_rnn-scratch`"""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

class RNNLMScratch(d2l.Classifier):
    """The RNN-based language model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0,
                 rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rnn', 'rngs'])
        self.rnn = rnn
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.E = nnx.Param(rngs.params.normal((vocab_size, rnn.num_inputs)))
        self.W_hq = nnx.Param(rngs.params.normal(
            (rnn.num_hiddens, vocab_size)) * sigma)
        self.b_q = nnx.Param(jnp.zeros(vocab_size))
        self.drop = nnx.Dropout(dropout, rngs=rngs)
        self.emb_drop = nnx.Dropout(emb_dropout, rngs=rngs)

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])

    def embedding(self, X):
        # Output shape: (num_steps, batch_size, num_inputs)
        return self.E[X.T]

    def output_layer(self, rnn_outputs):
        # A list of per-step states, or a tensor whose first axis is time
        H = d2l.stack(list(rnn_outputs), 1)  # (batch_size, num_steps, num_hiddens)
        return d2l.matmul(H, self.W_hq) + self.b_q

    def forward(self, X, state=None):
        embs = self.emb_drop(self.embedding(X))
        rnn_outputs, _ = self.rnn(embs, state)
        H = d2l.stack(list(rnn_outputs), 0)  # (num_steps, batch_size, num_hiddens)
        return self.output_layer(self.drop(H))

    def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0):
        model = nnx.view(self, deterministic=True, use_running_average=True,
                         raise_if_not_found=False)
        outputs, state = tok.encode(prefix), None
        for i in range(len(outputs) - 1):  # Warm up on the prefix
            X = d2l.tensor([[outputs[i]]])
            _, state = model.rnn(model.embedding(X), state)
        for _ in range(num_tokens):  # Generate num_tokens continuation tokens
            X = d2l.tensor([[outputs[-1]]])
            rnn_outputs, state = model.rnn(model.embedding(X), state)
            logits = model.output_layer(rnn_outputs)[0, 0]
            if temperature == 0:  # Greedy: the highest-scoring token
                token = logits.argmax()
            else:  # Sample from the softmax of logits / temperature
                token = jax.random.categorical(d2l.get_key(), logits / temperature)
            outputs.append(int(token))
        return tok.decode(outputs)

def clip_gradients(grad_clip_val, grads):
    """Rescale the gradients to global norm at most grad_clip_val.

    Defined in :numref:`sec_rnn-scratch`"""
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)

class RNN(nnx.Module):
    """The RNN model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01, rngs=None):
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.sigma = sigma
        normal = nnx.initializers.normal(sigma)
        cell = nnx.SimpleCell(num_inputs, num_hiddens, kernel_init=normal,
                              recurrent_kernel_init=normal, rngs=rngs)
        self.rnn = nnx.RNN(cell, time_major=True, return_carry=True,
                           rngs=rngs)

    def __call__(self, inputs, H=None):
        H, outputs = self.rnn(inputs, initial_carry=H)
        return outputs, H

class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0,
                 rngs=None):
        d2l.Classifier.__init__(self)
        self.save_hyperparameters(ignore=['rnn', 'rngs'])
        self.rnn = rnn
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.emb = nnx.Embed(vocab_size, rnn.num_inputs,
                             embedding_init=nnx.initializers.normal(1.0),
                             rngs=rngs)
        self.linear = nnx.Linear(
            rnn.num_hiddens, vocab_size,
            kernel_init=nnx.initializers.normal(sigma), rngs=rngs)
        self.drop = nnx.Dropout(dropout, rngs=rngs)
        self.emb_drop = nnx.Dropout(emb_dropout, rngs=rngs)

    def embedding(self, X):
        return self.emb(X.T)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

def beam_search(step_fn, prefix, num_tokens, beam_size=4,
                alpha=0.75, eos_id=None):
    """Length-normalized beam search (score = log P / len^alpha).
    Returns candidates sorted best-first; [0][1] is the argmax sequence.

    Defined in :numref:`sec_beam-search`"""
    def log_softmax(logits):
        logits = [float(l) for l in logits]
        m = max(logits)
        lse = m + math.log(sum(math.exp(l - m) for l in logits))
        return [l - lse for l in logits]
    score = lambda logp, ids: logp / max(1, len(ids) - len(prefix))**alpha
    beams = [(0.0, list(prefix), False)]  # (log-prob, sequence, finished)
    for _ in range(num_tokens):
        if all(done for _, _, done in beams):
            break
        candidates = [b for b in beams if b[2]]  # Finished pass through
        for logp, ids, _ in (b for b in beams if not b[2]):
            logprobs = log_softmax(step_fn(ids))
            best = sorted(range(len(logprobs)),
                          key=lambda i: -logprobs[i])[:beam_size]
            candidates += [(logp + logprobs[i], ids + [i], i == eos_id)
                           for i in best]
        beams = sorted(candidates, key=lambda b: score(b[0], b[1]),
                       reverse=True)[:beam_size]
    return sorted([(score(logp, ids), ids) for logp, ids, _ in beams],
                  reverse=True)

def train_2d(trainer, steps=20, f_grad=None, **hparams):
    """Run a 2D update rule from (-5, -2) and return the iterates.

    Defined in :numref:`sec_gd`"""
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1, s2: state of stateful optimizers
    results = [(x1, x2)]
    for _ in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad, **hparams)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, **hparams)
        results.append((x1, x2))
    print(f'step {steps}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):
    """Show the trace of 2D variables over the contours of f.

    Defined in :numref:`sec_gd`"""
    x1s, x2s = zip(*[(float(a), float(b)) for a, b in results])
    grids = []  # A window around start and minimum, widened to fit the trace
    for (lo, hi), xs in zip([(-5.5, 1.0), (-3.0, 3.0)], (x1s, x2s)):
        lo, hi = min(lo, min(xs) - 0.5), max(hi, max(xs) + 0.5)
        grids.append(d2l.arange(lo, hi, (hi - lo) / 100))
    x1, x2 = d2l.meshgrid(*grids, indexing='ij')
    width, height = (float(g[-1] - g[0]) for g in grids)
    equal = 0.25 < height / width < 4  # Equal scales unless the trace diverged
    d2l.set_figsize((3, 2.8) if equal else (3.5, 2.5))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='0.6', linewidths=0.8)
    d2l.plt.plot(x1s, x2s, '-o', color='C1', markersize=4)
    d2l.plt.xlabel('$x_1$')
    d2l.plt.ylabel('$x_2$')
    if equal:
        d2l.plt.gca().set_aspect('equal')

d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

def get_data_ch11(batch_size=10, n=1500):
    """Load the standardized airfoil self-noise data as minibatches.

    Defined in :numref:`sec_minibatch_sgd`"""
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (jnp.array(data[:n, :-1]), jnp.array(data[:n, -1])),
        batch_size, is_train=True)
    return data_iter, data.shape[1]-1

class Timer:
    """Record multiple running times.

    Defined in :numref:`sec_minibatch_sgd`"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

def record_run(step, full_loss, data_iter, num_epochs=2, plot=True):
    """Call step(X, y) on every minibatch; about every 200 examples, record
    full_loss() and the training time so far. Return the history.

    Defined in :numref:`sec_minibatch_sgd`"""
    if plot:
        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                xlim=[0, num_epochs], ylim=[0.22, 0.35])
    history = {'examples': [], 'loss': []}
    n, i, timer = 0, 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            out = step(X, y)
            n, i = n + len(X), i + 1
            if i % max(1, 200 // len(X)) == 0:  # About every 200 examples
                jax.block_until_ready(out)  # Wait for the queued steps
                timer.stop()
                l = float(full_loss())
                history['examples'].append(n)
                history['loss'].append(l)
                if plot:
                    animator.add(i / len(data_iter), (l,))
                timer.start()
    history['time'] = timer.cumsum()
    print(f'loss: {l:.3f}, {timer.sum() / num_epochs:.3f} sec/epoch')
    return history

def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2, plot=True):
    """Train linear regression with an update rule; return its history.

    Defined in :numref:`sec_minibatch_sgd`"""
    w = jnp.array(np.random.normal(scale=0.01, size=(feature_dim, 1)),
                  dtype=jnp.float32)
    b = jnp.zeros(1)
    loss = lambda w, b, X, y: d2l.squared_loss(d2l.linreg(X, w, b), y).mean()
    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))
    Xs, ys = (jnp.concatenate(t) for t in zip(*data_iter))  # The whole data
    full_loss = jax.jit(lambda w, b: loss(w, b, Xs, ys))

    def step(X, y):
        nonlocal w, b
        grads = grad_fn(w, b, X, y)
        w, b = trainer_fn([w, b], list(grads), states, hyperparams)
        return w, b

    X, y = next(iter(data_iter))  # One step on copies compiles everything
    copy = lambda tree: jax.tree.map(lambda a: a, tree)
    trainer_fn([w, b], list(grad_fn(w, b, X, y)), copy(states),
               copy(hyperparams))
    full_loss(w, b)
    return d2l.record_run(step, lambda: full_loss(w, b), data_iter,
                          num_epochs, plot)

def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2,
                       plot=True):
    """Train linear regression with an optax optimizer factory; return its
    history.

    Defined in :numref:`sec_minibatch_sgd`"""
    Xs, ys = (jnp.concatenate(t) for t in zip(*data_iter))  # The whole data
    net = nnx.Linear(Xs.shape[1], 1,
                     kernel_init=nnx.initializers.normal(0.01),
                     rngs=nnx.Rngs(random.getrandbits(32)))
    optimizer = nnx.Optimizer(net, trainer_fn(**hyperparams), wrt=nnx.Param)
    loss = lambda net, X, y: d2l.squared_loss(net(X), y).mean()

    @nnx.jit
    def step_fn(net, optimizer, X, y):
        l, grads = nnx.value_and_grad(loss)(net, X, y)
        optimizer.update(net, grads)
        return l

    full_loss = nnx.jit(lambda net: loss(net, Xs, ys))
    X, y = next(iter(data_iter))  # One step on copies compiles everything
    step_fn(nnx.clone(net), nnx.clone(optimizer), X, y)
    full_loss(net)
    return d2l.record_run(lambda X, y: step_fn(net, optimizer, X, y),
                          lambda: full_loss(net), data_iter, num_epochs, plot)

class WikiText103(d2l.DataModule):
    """WikiText-103 as UTF-8 bytes, cut into windows of num_steps bytes.

    Defined in :numref:`sec_adam`"""
    def __init__(self, batch_size, num_steps, num_train=None, num_val=None):
        super().__init__()
        self.save_hyperparameters()
        self.vocab = d2l.BPETokenizer(vocab_size=256, specials=())  # Bytes
        self.train = self._read('train-00000-of-00002',
                                '3cbc792dfc0eeedf8d7333acdc15a9f536032d9a')
        self.val = self._read('validation-00000-of-00001',
                              '49d15019b0476afc752bd5f010b5a2aab2de89ca')
        self.train, self.val = self.train[:num_train], self.val[:num_val]

    def _read(self, split, sha1):
        url = ('https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
               f'main/wikitext-103-raw-v1/{split}.parquet')
        fname = d2l.download(url, os.path.join(self.root, 'wikitext-103'),
                             sha1)
        lines = pd.read_parquet(fname)['text']  # Each line ends with '\n'
        return np.frombuffer(b''.join(s.encode() for s in lines), np.uint8)

    def windows(self, corpus, starts):
        """Inputs and next-byte targets of the windows at `starts`."""
        idx = np.asarray(starts)[:, None] + np.arange(self.num_steps + 1)
        array = corpus[idx].astype(np.int32)
        return array[:, :-1], array[:, 1:]

    def train_dataloader(self):  # A pass of minibatches at random offsets
        n = len(self.train) - self.num_steps
        rng = np.random.default_rng(random.getrandbits(32))
        for _ in range(n // (self.batch_size * self.num_steps)):
            yield self.windows(self.train,
                               rng.integers(n, size=self.batch_size))

    def val_dataloader(self):  # Consecutive, non-overlapping windows
        starts = np.arange(0, len(self.val) - self.num_steps, self.num_steps)
        for i in range(0, len(starts), self.batch_size):
            yield self.windows(self.val, starts[i:i + self.batch_size])

@jax.custom_vjp
def embedding_rows(table, X):
    """The rows X of table.

    Defined in :numref:`sec_adam`"""
    return table[X]

def _embedding_rows_fwd(table, X):
    return table[X], (table, X)

def _embedding_rows_bwd(residuals, grad):
    """Gradient of the table: the one-hot rows of X times the gradients.

    Defined in :numref:`sec_adam`"""
    table, X = residuals
    one_hot = jax.nn.one_hot(X, len(table), dtype=grad.dtype)
    return jnp.einsum('...v,...d->vd', one_hot, grad,
                      precision='highest').astype(table.dtype), None

embedding_rows.defvjp(_embedding_rows_fwd, _embedding_rows_bwd)

class Embed(nnx.Embed):  #@save
    """nnx.Embed whose gradient is a matrix product for small vocabularies.

    Defined in :numref:`sec_adam`"""
    def __call__(self, X):
        table = self.embedding[...]
        if len(table) > 1024:  # A one-hot row would cost too much
            return super().__call__(X)
        return embedding_rows(table, X).astype(self.dtype or table.dtype)

class TinyLM(nnx.Module):
    """A small decoder-only transformer language model.

    Defined in :numref:`sec_adam`"""
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_blks=2,
                 max_len=64, *, rngs):
        self.num_heads = num_heads
        def linear(d_in, d_out):
            a = 1 / math.sqrt(d_in)  # Weights and biases ~ U(-a, a)
            init = lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -a, a)
            return nnx.Linear(d_in, d_out, kernel_init=init, bias_init=init,
                              rngs=rngs)
        def embed(n):  # Embedding rows ~ N(0, I)
            return Embed(n, d_model, rngs=rngs,
                         embedding_init=nnx.initializers.normal(1.0))
        def norm():
            return nnx.LayerNorm(d_model, epsilon=1e-5, rngs=rngs)
        self.token_emb, self.pos_emb = embed(vocab_size), embed(max_len)
        self.blks = nnx.List([nnx.Dict(
            norm1=norm(), qkv=linear(d_model, 3 * d_model),
            proj=linear(d_model, d_model), norm2=norm(),
            mlp1=linear(d_model, 4 * d_model),
            mlp2=linear(4 * d_model, d_model))
            for _ in range(num_blks)])
        self.norm = norm()
        self.head = linear(d_model, vocab_size)

    def attention(self, blk, X):
        B, T, D = X.shape
        q, k, v = jnp.split(blk['qkv'](X), 3, axis=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1) for u in (q, k, v))
        Y = jax.nn.dot_product_attention(q, k, v, is_causal=True)
        return blk['proj'](Y.reshape(B, T, D))

    def __call__(self, X):
        H = self.token_emb(X) + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = H + self.attention(blk, blk['norm1'](H))
            H = H + blk['mlp2'](jax.nn.gelu(blk['mlp1'](blk['norm2'](H)),
                                            approximate=False))
        return self.head(self.norm(H))

@nnx.jit
def _lm_train_step(model, optimizer, X, Y):
    def loss_fn(model):
        logits = model(X)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits.reshape(-1, logits.shape[-1]), Y.reshape(-1)).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

def train_lm(model, data, optimizer, num_steps):
    """Train a model on next-token prediction for a fixed number of steps.

    Defined in :numref:`sec_adam`"""
    losses, pending, step = [], [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            pending.append(_lm_train_step(model, optimizer, jnp.asarray(X),
                                          jnp.asarray(Y)))
            step += 1
            if len(pending) == 50 or step >= num_steps:  # Sync now and then
                losses.extend(float(l) for l in jax.device_get(pending))
                pending = []
            if step >= num_steps:
                return losses

@jax.jit(static_argnums=0)
def lm_loss_sum(graphdef, state, X, Y):
    """Summed next-token cross-entropy of a batch, and the model's state.

    Defined in :numref:`sec_adam`"""
    model = nnx.merge(graphdef, state)
    loss = optax.softmax_cross_entropy_with_integer_labels(model(X), Y).sum()
    return loss, nnx.state(model)

def lm_perplexity(model, batches):
    """Perplexity of a language model on the minibatches (X, Y) of batches.

    Defined in :numref:`sec_adam`"""
    # A view with dropout off, sharing the weights; one walk through it
    graphdef, state = nnx.split(nnx.view(model, deterministic=True,
                                         raise_if_not_found=False))
    total, count = 0, 0
    for X, Y in batches:
        loss, state = lm_loss_sum(graphdef, state, X, Y)
        total, count = total + loss, count + Y.size
    nnx.update(model, state)  # Keep statistics the model records, if any
    return math.exp(float(total) / count)

def final_loss(losses, k=100):
    """Mean of the last k losses; infinity if the run diverged.

    Defined in :numref:`sec_adam`"""
    v = sum(losses[-k:]) / k
    return v if math.isfinite(v) else float('inf')

def smooth(losses, k=25):
    """Means over consecutive blocks of k losses, for plotting.

    Defined in :numref:`sec_adam`"""
    return [sum(losses[i:i + k]) / k
            for i in range(0, len(losses) - k + 1, k)]

class FashionCNN(nnx.Module):
    """A compact CNN for Fashion-MNIST: two conv-pool stages, two linear.

    Defined in :numref:`sec_adam`"""
    def __init__(self, *, rngs):
        def init(fan_in):  # Weights and biases ~ U(-a, a), a = 1/sqrt(fan_in)
            a = 1 / math.sqrt(fan_in)
            f = lambda key, shape, dtype: jax.random.uniform(
                key, shape, dtype, -a, a)
            return dict(kernel_init=f, bias_init=f, rngs=rngs)
        self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), **init(9))
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), **init(9 * 32))
        self.fc1 = nnx.Linear(64 * 7 * 7, 128, **init(64 * 7 * 7))
        self.fc2 = nnx.Linear(128, 10, **init(128))

    def __call__(self, X):
        X = nnx.max_pool(nnx.relu(self.conv1(X)), window_shape=(2, 2),
                         strides=(2, 2))
        X = nnx.max_pool(nnx.relu(self.conv2(X)), window_shape=(2, 2),
                         strides=(2, 2))
        return self.fc2(nnx.relu(self.fc1(X.reshape(X.shape[0], -1))))

@nnx.jit
def num_correct(model, X, y):
    """Number of examples in (X, y) that model classifies correctly.

    Defined in :numref:`sec_adam`"""
    return (model(X).argmax(axis=1) == y).sum()

def evaluate_accuracy(model, batches):
    """Fraction of the examples in batches that model classifies correctly.

    Defined in :numref:`sec_adam`"""
    # A view with dropout off and batch norm on running statistics
    model = nnx.view(model, deterministic=True, use_running_average=True,
                     raise_if_not_found=False)
    correct, total = 0, 0
    for X, y in batches:
        correct += num_correct(model, jnp.asarray(X), jnp.asarray(y))
        total += len(y)
    return float(correct) / total

def lr_sweep(make_model, make_tx, data, lrs, seed,
             num_steps=2000, *, evaluate=None, metric='score', every=None):
    """Train make_model(rngs=...) with the optax transformation make_tx(lr)
    for each lr, from one initialization; return the loss curves and the
    evaluations.

    Defined in :numref:`sec_adam`"""
    every = every or num_steps
    curves, scores = {}, {}
    for lr in lrs:  # The same initialization for every lr
        model = make_model(rngs=nnx.Rngs(seed))
        optimizer = nnx.Optimizer(model, make_tx(lr), wrt=nnx.Param)
        curves[lr], scores[lr] = [], []
        for _ in range(num_steps // every):
            curves[lr] += train_lm(model, data, optimizer, every)
            if evaluate is not None:
                scores[lr].append(evaluate(model))
        line = f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}'
        if scores[lr]:
            line += f', {metric} {scores[lr][-1]:.3f}'
        print(line)
    return curves, scores

def masked_softmax(X, mask=None):
    """Softmax over the last axis with zero weight where mask is False.

    Defined in :numref:`sec_attention-scoring-functions`"""
    if mask is None:
        return jax.nn.softmax(X, axis=-1)
    # Most negative finite value: zero weight at any precision, no NaN
    X = jnp.where(mask, X, jnp.finfo(X.dtype).min)
    weights = jax.nn.softmax(X, axis=-1)
    # A query without any valid key gets zero weights, not a uniform row
    return jnp.where(mask.any(-1, keepdims=True), weights, 0.0)

def padding_mask(valid_lens, num_keys):
    """Boolean mask that is True for the first valid_lens keys.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # valid_lens of shape (batch,) gives (batch, 1, keys); per-query
    # lengths of shape (batch, queries) give (batch, queries, keys)
    if valid_lens.ndim == 1:
        valid_lens = valid_lens[:, None]
    return jnp.arange(num_keys) < valid_lens[..., None]

def causal_mask(num_queries, num_keys=None):
    """Boolean (queries, keys) mask, True where a query may see a key.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # Queries are the last positions, as when new tokens attend to a
    # cache: query i sees key j if j <= i + num_keys - num_queries
    num_keys = num_queries if num_keys is None else num_keys
    i = jnp.arange(num_queries)[:, None]
    j = jnp.arange(num_keys)[None, :]
    return j <= i + (num_keys - num_queries)

class DotProductAttention(nnx.Module):
    """Scaled dot-product attention over the last two axes.

    Defined in :numref:`sec_attention-scoring-functions`"""
    def __init__(self, dropout=0.0, *, rngs=None):
        # rngs is needed only to draw dropout masks (dropout > 0)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        # A Variable, so that views of the module (nnx.view) share it
        self.last_weights = nnx.Intermediate(None)

    # Leading axes (batch, and heads if present) are carried along.
    # Shape of queries: (..., no. of queries, d)
    # Shape of keys: (..., no. of key-value pairs, d)
    # Shape of values: (..., no. of key-value pairs, value dimension)
    # mask: boolean, broadcastable to (..., no. of queries, no. of keys),
    # True where a query may attend to a key
    def __call__(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        scores = queries @ keys.swapaxes(-2, -1) / math.sqrt(d)
        weights = masked_softmax(scores, mask)
        if not isinstance(weights, jax.core.Tracer):
            # Record the weights of eager calls for inspection; inside
            # jit-compiled code skip it, so the module state keeps a
            # fixed structure and the compiled step is reused
            self.last_weights.set_value(weights)
        return self.dropout(weights) @ values

    @property
    def attention_weights(self):
        """Weights of the last eager call: (..., queries, keys)."""
        return self.last_weights.get_value()

class Linear(nnx.Linear):
    """nnx.Linear with weights and biases drawn from U(-a, a), where
    a = 1/sqrt(in_features).

    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self, in_features, out_features, *, use_bias=True, rngs,
                 **kwargs):
        a = in_features ** -0.5
        init = lambda key, shape, dtype=jnp.float32: jax.random.uniform(
            key, shape, dtype, -a, a)
        super().__init__(in_features, out_features, use_bias=use_bias,
                         kernel_init=init, bias_init=init, rngs=rngs,
                         **kwargs)

class MultiHeadAttention(nnx.Module):
    """Multi-head attention with heads as an explicit axis; num_kv_heads <
    num_heads shares each key-value head across a group of query heads.

    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self, num_hiddens, num_heads, dropout=0.0, bias=False,
                 num_kv_heads=None, *, rngs):
        assert num_hiddens % num_heads == 0, 'heads must divide num_hiddens'
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0
        kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
        self.attention = d2l.DotProductAttention(dropout, rngs=rngs)
        self.W_q = Linear(num_hiddens, num_hiddens, use_bias=bias, rngs=rngs)
        self.W_k = Linear(num_hiddens, kv_hiddens, use_bias=bias, rngs=rngs)
        self.W_v = Linear(num_hiddens, kv_hiddens, use_bias=bias, rngs=rngs)
        self.W_o = Linear(num_hiddens, num_hiddens, use_bias=bias, rngs=rngs)

    def __call__(self, queries, keys, values, mask=None):
        # queries: (batch, no. of queries, num_hiddens); keys and values:
        # (batch, no. of key-value pairs, num_hiddens). mask: boolean,
        # broadcastable to (batch, no. of queries, no. of keys), shared by
        # all heads, or to (batch, num_heads, no. of queries, no. of keys)
        q = self.split_heads(self.W_q(queries), self.num_heads)
        k = self.split_heads(self.W_k(keys), self.num_kv_heads)
        v = self.split_heads(self.W_v(values), self.num_kv_heads)
        group = self.num_heads // self.num_kv_heads
        k, v = jnp.repeat(k, group, axis=1), jnp.repeat(v, group, axis=1)
        if mask is not None and mask.ndim == 3:
            mask = mask[:, None]           # The same mask for every head
        # Shape of Y: (batch, num_heads, no. of queries, head width)
        Y = self.attention(q, k, v, mask)
        return self.W_o(self.merge_heads(Y))

    @property
    def attention_weights(self):
        """Weights of the last call: (batch, num_heads, queries, keys)."""
        return self.attention.attention_weights

    def split_heads(self, X, num_heads):
        """(batch, length, num_heads * width) -> (batch, num_heads, length,
        width).

        Defined in :numref:`sec_multihead-attention`"""
        return X.reshape(X.shape[0], X.shape[1], num_heads, -1).swapaxes(1, 2)

    def merge_heads(self, X):
        """Reverse split_heads: (batch, num_heads, length, width) -> (batch,
        length, num_heads * width).

        Defined in :numref:`sec_multihead-attention`"""
        return X.swapaxes(1, 2).reshape(X.shape[0], X.shape[2], -1)

def sinusoidal_encoding(max_len, num_hiddens):
    """The (max_len, num_hiddens) table of sinusoidal position codes.

    Defined in :numref:`sec_positional-information`"""
    theta = jnp.arange(max_len)[:, None] / 10000 ** (
        jnp.arange(0, num_hiddens, 2) / num_hiddens)
    return jnp.stack([jnp.sin(theta), jnp.cos(theta)],
                     -1).reshape(max_len, num_hiddens)

def rope(x, offset=0, base=10000.0, scale=None):
    """Rotate feature pairs of x (..., num_steps, d) by position angles.

    Defined in :numref:`sec_positional-information`"""
    d = x.shape[-1]
    pos = jnp.arange(x.shape[-2], dtype=jnp.float32) + offset
    omega = base ** (-jnp.arange(0, d, 2) / d)
    if scale is not None:  # One angle factor per pair, shape (d/2,)
        omega = omega * scale
    theta = pos[:, None] * omega[None, :]
    # Angles in fp32, tables in the dtype of x: bf16 queries stay bf16
    cos, sin = jnp.cos(theta).astype(x.dtype), jnp.sin(theta).astype(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], -1).reshape(x.shape)

class TinyCharLM(nnx.Module):
    """Attention-only language model.

    Defined in :numref:`sec_positional-information`"""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                 pos='rope', max_len=512, bias=False, *, rngs):
        assert pos in ('learned', 'sinusoidal', 'rope', 'alibi', 'none')
        self.num_heads, self.pos = num_heads, pos
        init = nnx.initializers.normal(0.02)
        self.token_emb = d2l.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        if pos == 'learned':
            self.pos_emb = nnx.Embed(max_len, num_hiddens,
                                     embedding_init=init, rngs=rngs)
        if pos == 'sinusoidal':
            self.P = nnx.Cache(sinusoidal_encoding(max_len, num_hiddens))
        if pos == 'rope':  # Base and per-pair angle factors of the rotation
            self.rope_base = 10000.0
            self.rope_scale = nnx.Cache(
                jnp.ones(num_hiddens // num_heads // 2))
        self.blks = nnx.List([nnx.Dict(
            qkv=d2l.Linear(num_hiddens, 3 * num_hiddens, use_bias=bias,
                           rngs=rngs),
            proj=d2l.Linear(num_hiddens, num_hiddens, use_bias=bias,
                            rngs=rngs))
            for _ in range(num_blks)])

    def _alibi(self, T):
        h = jnp.arange(1, self.num_heads + 1)
        slopes = 2.0 ** (-8.0 * h / self.num_heads)
        pos = jnp.arange(T, dtype=jnp.float32)
        return slopes[:, None, None] * (pos[None, :] - pos[:, None])

    def _attend(self, blk, H):
        B, T, D = H.shape
        q, k, v = jnp.split(blk['qkv'](H), 3, axis=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).swapaxes(1, 2)
                   for u in (q, k, v))
        if self.pos == 'rope':
            q = rope(q, base=self.rope_base, scale=self.rope_scale[...])
            k = rope(k, base=self.rope_base, scale=self.rope_scale[...])
        scores = q @ k.swapaxes(-2, -1) / math.sqrt(q.shape[-1])
        if self.pos == 'alibi':
            scores = scores + self._alibi(T)
        weights = d2l.masked_softmax(scores, d2l.causal_mask(T))
        out = (weights @ v).swapaxes(1, 2).reshape(B, T, D)
        return blk['proj'](out), weights

    def _embed(self, X):
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(jnp.arange(X.shape[1]))
        if self.pos == 'sinusoidal':  # Scale tokens up to the table
            H = H * math.sqrt(H.shape[-1]) + self.P[:X.shape[1]]
        return H

    def __call__(self, X):
        H = self._embed(X)
        for blk in self.blks:
            out, _ = self._attend(blk, H)
            H = H + out
        return self.token_emb.attend(H)  # Tied output head

    def attention_weights(self, X):
        """Per-block attention maps, each (batch, num_heads, T, T)."""
        H, maps = self._embed(X), []
        for blk in self.blks:
            out, weights = self._attend(blk, H)
            maps.append(weights)
            H = H + out
        return maps

class FeedForward(nnx.Module):
    """Position-wise FFN: GELU MLP or SwiGLU at matched parameter count.

    Defined in :numref:`sec_transformer-block`"""
    def __init__(self, num_hiddens, act='swiglu', bias=False, *, rngs):
        assert act in ('swiglu', 'gelu'), f'unknown act: {act!r}'
        self.act = act
        if act == 'gelu':
            width = 4 * num_hiddens
        else:  # 'swiglu': three matrices; width 8d/3 matches the MLP budget
            width = round(8 * num_hiddens / 3)
            self.W_g = d2l.Linear(num_hiddens, width, use_bias=bias,
                                  rngs=rngs)
        self.W_1 = d2l.Linear(num_hiddens, width, use_bias=bias, rngs=rngs)
        self.W_2 = d2l.Linear(width, num_hiddens, use_bias=bias, rngs=rngs)

    def __call__(self, X):
        if self.act == 'gelu':  # The tanh approximation, as in GPT-2
            return self.W_2(jax.nn.gelu(self.W_1(X), approximate=True))
        return self.W_2(jax.nn.silu(self.W_g(X)) * self.W_1(X))

class TransformerBlock(nnx.Module):
    """Configurable transformer block: attention + FFN on a residual
    stream.

    Defined in :numref:`sec_transformer-block`"""
    def __init__(self, num_hiddens, num_heads, *, dropout=0, norm='rms',
                 act='swiglu', pre_norm=True, bias=False, attn_factory=None,
                 ffn_factory=None, rngs):
        assert norm in ('rms', 'layer'), f'unknown norm: {norm!r}'
        assert num_hiddens % num_heads == 0
        self.pre_norm = pre_norm
        make_norm = nnx.RMSNorm if norm == 'rms' else nnx.LayerNorm
        self.norm1 = make_norm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.norm2 = make_norm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.attention = (d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, bias=bias,
                                                 rngs=rngs)
                          if attn_factory is None else attn_factory(rngs))
        self.ffn = (FeedForward(num_hiddens, act, bias=bias, rngs=rngs)
                    if ffn_factory is None else ffn_factory(rngs))
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, X, mask=None):
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.dropout(self.attention(Y, Y, Y, mask))
            return X + self.dropout(self.ffn(self.norm2(X)))
        X = self.norm1(X + self.dropout(self.attention(X, X, X, mask)))
        return self.norm2(X + self.dropout(self.ffn(X)))

class GPT(nnx.Module):
    """Decoder-only transformer language model built from configurable
    blocks.

    Defined in :numref:`sec_gpt`"""

    class CausalAttention(nnx.Module):
        """Causal self-attention, optionally rotary; num_kv_heads <
        num_heads shares each key-value head across a group of queries."""
        def __init__(self, num_hiddens, num_heads, *, bias=False, rope=False,
                     num_kv_heads=None, rngs):
            self.num_heads, self.rope = num_heads, rope
            self.num_kv_heads = num_kv_heads or num_heads
            self.kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
            self.W_qkv = d2l.Linear(num_hiddens,
                                    num_hiddens + 2 * self.kv_hiddens,
                                    use_bias=bias, rngs=rngs)
            self.W_o = d2l.Linear(num_hiddens, num_hiddens, use_bias=bias,
                                  rngs=rngs)

        def qkv(self, X, offset=0):
            """Queries, keys, and values of X, each (batch, heads, length,
            head width), at positions offset, offset + 1, ..."""
            B, T, D = X.shape
            q, k, v = jnp.split(self.W_qkv(X), [D, D + self.kv_hiddens],
                                axis=-1)
            q = q.reshape(B, T, self.num_heads, -1).swapaxes(1, 2)
            k, v = (u.reshape(B, T, self.num_kv_heads, -1).swapaxes(1, 2)
                    for u in (k, v))
            if self.rope:
                q, k = d2l.rope(q, offset), d2l.rope(k, offset)
            return q, k, v

        @staticmethod
        @jax.checkpoint  # The backward pass recomputes the T x T weights
        def attend(q, k, v):
            """Causal attention of (batch, length, heads, head width)
            queries, keys, and values."""
            return jax.nn.dot_product_attention(q, k, v, is_causal=True)

        def __call__(self, X, keys=None, values=None, mask=None):
            # Self-attention: X also serves as the keys and the values, and
            # the mask is causal, so a block's mask argument must stay None
            if mask is not None:
                raise ValueError('CausalAttention applies its own mask')
            q, k, v = self.qkv(X)
            Y = self.attend(*(u.swapaxes(1, 2) for u in (q, k, v)))
            return self.W_o(Y.reshape(Y.shape[0], Y.shape[1], -1))

    def __init__(self, vocab_size, *, num_hiddens=256, num_heads=8,
                 num_blks=6, max_len=1024, pos='rope', norm='rms',
                 act='swiglu', pre_norm=True, bias=False, dropout=0,
                 num_kv_heads=None, ffn_factory=None, rngs):
        assert pos in ('rope', 'learned'), f'unknown pos: {pos!r}'
        self.pos, self.max_len = pos, max_len
        init = nnx.initializers.normal(0.02)
        self.token_emb = d2l.Embed(vocab_size, num_hiddens,
                                   embedding_init=init, rngs=rngs)
        if pos == 'learned':
            self.pos_emb = nnx.Embed(max_len, num_hiddens,
                                     embedding_init=init, rngs=rngs)
        attn = lambda rngs: self.CausalAttention(
            num_hiddens, num_heads, bias=bias, rope=pos == 'rope',
            num_kv_heads=num_kv_heads, rngs=rngs)
        self.blks = nnx.List([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout=dropout,
                                 norm=norm, act=act, pre_norm=pre_norm,
                                 bias=bias, attn_factory=attn,
                                 ffn_factory=ffn_factory, rngs=rngs)
            for _ in range(num_blks)])
        self.norm = (nnx.RMSNorm if norm == 'rms'
                     else nnx.LayerNorm)(num_hiddens, epsilon=1e-5, rngs=rngs)
        for path, m in nnx.iter_graph(self):  # GPT-2's initialization
            if isinstance(m, nnx.Linear):  # W_o and W_2 write to the stream
                std = 0.02 / (2 * num_blks) ** 0.5 if path[-1] in (
                    'W_o', 'W_2') else 0.02
                m.kernel[...] = std * jax.random.normal(rngs.params(),
                                                        m.kernel.shape)
                if m.bias is not None:
                    m.bias[...] = jnp.zeros(m.bias.shape)

    def __call__(self, X):
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(jnp.arange(X.shape[1]))
        for blk in self.blks:
            H = blk(H)
        return self.token_emb.attend(self.norm(H))

    def generate(self, prefix, num_tokens, temperature=1.0, top_k=None):
        """Sample a continuation of the token-id list prefix.

        Defined in :numref:`sec_gpt`"""
        n, ctx = self.max_len, list(prefix)[-self.max_len:]
        # One buffer of fixed shape (1, max_len): the step compiles once
        buf, t = jnp.asarray([ctx + [0] * (n - len(ctx))]), len(ctx)
        tokens = []
        for key in jax.random.split(d2l.get_key(), num_tokens):
            buf, token = _gpt_sample_next(self, buf, t, key, temperature, top_k)
            tokens.append(token)
            t = min(t + 1, n)
        return list(prefix) + [int(i) for i in jax.device_get(tokens)]

def lm_decay_mask(params):
    """Decay the weight matrices; exempt embeddings, gains, and biases.

    Defined in :numref:`sec_gpt`"""
    return jax.tree_util.tree_map_with_path(
        lambda path, p: p.ndim == 2 and 'emb' not in str(path), params)

@optax.inject_hyperparams
def lm_adamw(lr=0.0, num_steps=1, warmup=0, weight_decay=0.0, max_norm=1.0):
    """The transformation of LMOptimizer, with its hyperparameters as state.

    Defined in :numref:`sec_gpt`"""
    schedule = lambda t: lr * d2l.cosine_warmup(t, num_steps, warmup)
    return optax.chain(
        optax.clip_by_global_norm(max_norm),
        optax.adamw(schedule, weight_decay=weight_decay, mask=lm_decay_mask))

class LMOptimizer(nnx.Optimizer):
    """AdamW with weight decay on the weight matrices only, a linear warmup
    and a cosine decay of the learning rate, and gradient-norm clipping.

    Defined in :numref:`sec_gpt`"""
    tx = lm_adamw()  # Shared by all instances: the step compiles once
    def __init__(self, model, lr, num_steps, warmup, weight_decay=0.1,
                 max_norm=1.0):
        super().__init__(model, LMOptimizer.tx, wrt=nnx.Param)
        hyperparams = dict(lr=lr, num_steps=num_steps, warmup=warmup,
                           weight_decay=weight_decay, max_norm=max_norm)
        for name, value in hyperparams.items():
            self.opt_state.hyperparams[name][...] = value

@nnx.jit(static_argnames='top_k')
def _gpt_sample_next(model, buf, t, key, temperature, top_k):
    """Sample the token after the first t tokens of buf and append it.

    A full buffer (t equal to its length) drops its oldest token first.

    Defined in :numref:`sec_gpt`"""
    logits = model(buf)[0, t - 1] / temperature
    if top_k is not None:
        cutoff = jax.lax.top_k(logits, top_k)[0][-1]
        logits = jnp.where(logits < cutoff, -jnp.inf, logits)
    token = jax.random.categorical(key, logits)
    n = buf.shape[1]
    buf = jnp.where(t == n, jnp.roll(buf, -1, axis=1), buf)
    return buf.at[0, jnp.minimum(t, n - 1)].set(token), token

GPT2_URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/'
d2l.DATA_HUB['gpt2-weights'] = (
    GPT2_URL + 'model.safetensors',
    '89a76996d7c6ee89b86618a265483aab73e61d50')
d2l.DATA_HUB['gpt2-merges'] = (
    GPT2_URL + 'merges.txt', '396d4d8ec90cb02f4d56e049e0e4add868bcd943')
d2l.DATA_HUB['gpt2-encoder'] = (
    GPT2_URL + 'vocab.json', 'f0223209235343bc067d7da838328bced8085ae1')

def gpt2_tokenizer():
    """GPT-2's byte-level BPE tokenizer, assembled by tiktoken.

    Defined in :numref:`sec_gpt`"""
    import tiktoken  # lazy: d2l itself does not require tiktoken
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks
    return tiktoken.Encoding(
        'gpt2', explicit_n_vocab=50257,
        pat_str=d2l.BPETokenizer.GPT2_PATTERN,
        mergeable_ranks=data_gym_to_mergeable_bpe_ranks(
            d2l.download('gpt2-merges', '../data/gpt2'),
            d2l.download('gpt2-encoder', '../data/gpt2')),
        special_tokens={'<|endoftext|>': 50256})

def load_gpt2():
    """GPT-2 small (124M) as a GPT with the released weights, in eval
    mode.

    Defined in :numref:`sec_gpt`"""
    from safetensors.numpy import load_file  # lazy, like tiktoken above
    gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
               max_len=1024, pos='learned', norm='layer', act='gelu',
               pre_norm=True, bias=True, rngs=nnx.Rngs(d2l.get_key()))
    weights = {k: jnp.asarray(v) for k, v in load_file(
        d2l.download('gpt2-weights', '../data/gpt2')).items()}
    gpt2.token_emb.embedding[...] = weights['wte.weight']
    gpt2.pos_emb.embedding[...] = weights['wpe.weight']
    gpt2.norm.scale[...] = weights['ln_f.weight']
    gpt2.norm.bias[...] = weights['ln_f.bias']
    for i, blk in enumerate(gpt2.blks):
        for norm, key in ((blk.norm1, f'h.{i}.ln_1'),
                          (blk.norm2, f'h.{i}.ln_2')):
            norm.scale[...] = weights[key + '.weight']
            norm.bias[...] = weights[key + '.bias']
        for linear, key in ((blk.attention.W_qkv, f'h.{i}.attn.c_attn'),
                            (blk.attention.W_o, f'h.{i}.attn.c_proj'),
                            (blk.ffn.W_1, f'h.{i}.mlp.c_fc'),
                            (blk.ffn.W_2, f'h.{i}.mlp.c_proj')):
            # Conv1D stores (in, out), exactly nnx.Linear's kernel layout
            linear.kernel[...] = weights[key + '.weight']
            linear.bias[...] = weights[key + '.bias']
    gpt2.eval()
    return gpt2

def kv_cache_bytes(model, num_tokens, batch_size=1, bytes_per_elem=4):
    """Bytes of the keys and values of num_tokens positions in every block
    of a GPT, for batch_size sequences.

    Defined in :numref:`sec_kv-cache`"""
    attn = model.blks[0].attention
    return (2 * len(model.blks) * attn.kv_hiddens * num_tokens * batch_size
            * bytes_per_elem)

class PatchEmbedding(nnx.Module):
    """Image-to-sequence stem of the vision transformer.

    Defined in :numref:`sec_vision-transformer`"""
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512,
                 num_channels=3, *, rngs):
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        # A partial trailing patch would silently shrink the token grid
        assert img_size[0] % patch_size[0] == 0 and \
            img_size[1] % patch_size[1] == 0, \
            'image size must be divisible by the patch size'
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        # PyTorch's default initialization: U(-a, a), a = 1/sqrt(fan_in)
        a = (num_channels * patch_size[0] * patch_size[1]) ** -0.5
        init = lambda key, shape, dtype=jnp.float32: jax.random.uniform(
            key, shape, dtype, -a, a)
        self.conv = nnx.Conv(num_channels, num_hiddens, kernel_size=patch_size,
                             strides=patch_size, padding='VALID',
                             kernel_init=init, bias_init=init, rngs=rngs)

    def __call__(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        X = self.conv(X)
        return X.reshape((X.shape[0], -1, X.shape[3]))

class ViTBlock(nnx.Module):
    """Pre-norm transformer block with a GELU MLP.

    Defined in :numref:`sec_vision-transformer`"""
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads, dropout,
                 use_bias=False, *, rngs):
        self.ln1 = nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs)
        self.mlp = nnx.Sequential(
            d2l.Linear(num_hiddens, mlp_num_hiddens, rngs=rngs),
            lambda X: jax.nn.gelu(X, approximate=False),  # Exact GELU
            nnx.Dropout(dropout, rngs=rngs),
            d2l.Linear(mlp_num_hiddens, num_hiddens, rngs=rngs),
            nnx.Dropout(dropout, rngs=rngs))

    def __call__(self, X, mask=None):
        X_norm = self.ln1(X)
        X = X + self.attention(X_norm, X_norm, X_norm, mask)
        return X + self.mlp(self.ln2(X))

class ViT(d2l.Classifier):
    """Vision transformer.

    Defined in :numref:`sec_vision-transformer`"""
    def __init__(self, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, emb_dropout, blk_dropout, num_steps,
                 lr=1e-3, use_bias=False, num_classes=10, num_channels=1, *,
                 rngs):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens, num_channels, rngs=rngs)
        self.cls_token = nnx.Param(jnp.zeros((1, 1, num_hiddens)))
        num_tokens = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable, initialized to small noise
        self.pos_embedding = nnx.Param(
            rngs.params.normal((1, num_tokens, num_hiddens)) * 0.02)
        self.embedding_dropout = nnx.Dropout(emb_dropout, rngs=rngs)
        self.blks = nnx.List([
            ViTBlock(num_hiddens, mlp_num_hiddens, num_heads, blk_dropout,
                     use_bias, rngs=rngs) for _ in range(num_blks)])
        self.head = nnx.Sequential(
            nnx.LayerNorm(num_hiddens, epsilon=1e-5, rngs=rngs),
            d2l.Linear(num_hiddens, num_classes, rngs=rngs))

    def forward(self, X):
        X = self.patch_embedding(X)
        X = d2l.concat((jnp.tile(self.cls_token, (X.shape[0], 1, 1)), X), 1)
        X = self.embedding_dropout(X + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

    def configure_optimizers(self):
        # AdamW decays weight matrices and kernels; it exempts biases, norm
        # parameters, the class token, and the position embeddings
        decay = lambda params: jax.tree_util.tree_map_with_path(
            lambda path, p: p.ndim >= 2 and path[0].key not in (
                'cls_token', 'pos_embedding'), params)
        # Linear warmup over the first tenth of the steps, then cosine
        return optax.adamw(lambda t: self.lr * d2l.cosine_warmup(
            t, self.num_steps, self.num_steps // 10), weight_decay=0.05,
            mask=decay)

class GRU(d2l.RNN):
    """The multilayer GRU model implemented with high-level APIs.

    Defined in :numref:`sec_gru`"""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0,
                 rngs=None):
        rngs = nnx.Rngs(random.getrandbits(32)) if rngs is None else rngs
        self.num_inputs, self.num_hiddens = num_inputs, num_hiddens
        self.num_layers = num_layers
        self.rnns = nnx.List([
            nnx.RNN(nnx.GRUCell(num_inputs if i == 0 else num_hiddens,
                                num_hiddens, rngs=rngs),
                    time_major=True, return_carry=True, rngs=rngs)
            for i in range(num_layers)])
        self.dropouts = nnx.List([
            nnx.Dropout(dropout, rngs=rngs) for _ in range(num_layers - 1)])

    def __call__(self, X, state=None):
        outputs = X
        state = [None] * self.num_layers if state is None else state
        new_state = []
        for i, rnn in enumerate(self.rnns):
            carry, outputs = rnn(outputs, initial_carry=state[i])
            new_state.append(carry)
            if i < self.num_layers - 1:
                outputs = self.dropouts[i](outputs)
        return outputs, new_state

def delayed_recall(rng, batch_size, num_steps, num_symbols=8,
                   blank_fillers=False):
    """A symbol, then num_steps - 1 fillers; the target is the symbol.

    Defined in :numref:`sec_gru`"""
    Y = rng.integers(0, num_symbols, batch_size)
    X = rng.integers(num_symbols, 2 * num_symbols, (batch_size, num_steps))
    if blank_fillers:  # The same filler token at every step
        X[:] = num_symbols
    X[:, 0] = Y
    return X, Y

class RecallModel(nnx.Module):
    """Embedding, a sequence layer, and a classifier of its last output.

    Defined in :numref:`sec_gru`"""
    def __init__(self, layer, num_hiddens, vocab_size=16, num_inputs=32,
                 num_outputs=8, *, rngs):
        self.embedding = d2l.Embed(vocab_size, num_inputs, rngs=rngs,
            embedding_init=nnx.initializers.normal(num_inputs ** -0.5))
        self.layer = layer
        self.head = nnx.Linear(num_hiddens, num_outputs, rngs=rngs)

    def __call__(self, X):
        return self.head(self.layer(self.embedding(X))[:, -1])

@nnx.jit
def recall_step(model, optimizer, X, Y):
    """One Adam step on the cross-entropy of the classifier's output.

    Defined in :numref:`sec_gru`"""
    def loss_fn(model):
        return optax.softmax_cross_entropy_with_integer_labels(
            model(X), Y).mean()
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss

@nnx.jit
def recall_predict(model, X):
    """The predicted class of every sequence.

    Defined in :numref:`sec_gru`"""
    return model(X).argmax(-1)

def train_delayed_recall(model, num_steps, blank_fillers=False,
                         num_updates=1500, lr=3e-3):
    """Train on fresh batches of delayed recall; return the test accuracy.

    Defined in :numref:`sec_gru`"""
    rng = np.random.default_rng(0)
    batch = lambda rng, n: delayed_recall(rng, n, num_steps,
                                          blank_fillers=blank_fillers)
    optimizer = nnx.Optimizer(model, optax.chain(
        optax.clip_by_global_norm(1.0), optax.adam(lr)), wrt=nnx.Param)
    step = nnx.cached_partial(recall_step, model, optimizer)  # Compile once
    for _ in range(num_updates):
        step(*batch(rng, 128))
    X, Y = batch(np.random.default_rng(1), 2048)
    return float((recall_predict(model, X) == Y).mean())

class SeqLM(nnx.Module):
    """A language model of residual blocks: a sequence mixer, then an MLP.

    Defined in :numref:`sec_gru`"""
    def __init__(self, vocab_size, num_hiddens, mixers, *, rngs):
        d = num_hiddens
        self.embedding = d2l.Embed(vocab_size, d, rngs=rngs,  # Unit length
            embedding_init=nnx.initializers.normal(d ** -0.5))
        self.mixers = nnx.List(mixers)
        norm = lambda: nnx.LayerNorm(d, epsilon=1e-5, rngs=rngs)
        self.norms = nnx.List([norm() for _ in mixers])
        self.mlps = nnx.List([nnx.Sequential(
            norm(), nnx.Linear(d, 4 * d, rngs=rngs),
            lambda x: nnx.gelu(x, approximate=False),  # Exact GELU
            nnx.Linear(4 * d, d, rngs=rngs)) for _ in mixers])
        self.norm = norm()
        self.head = nnx.Linear(d, vocab_size, rngs=rngs)

    def __call__(self, X):
        H = self.embedding(X)  # (batch_size, num_steps, num_hiddens)
        for norm, mixer, mlp in zip(self.norms, self.mixers, self.mlps):
            H = H + mixer(norm(H))
            H = H + mlp(H)
        return self.head(self.norm(H))  # (batch_size, num_steps, vocab_size)

class PennTreebank(d2l.DataModule):
    """Character-level Penn Treebank, cut into windows of num_steps.

    Defined in :numref:`sec_gru`"""
    def __init__(self, batch_size, num_steps):
        super().__init__()
        self.save_hyperparameters()
        data_dir = d2l.download_extract('ptb')
        def read(split):
            with open(os.path.join(data_dir, f'ptb.{split}.txt')) as f:
                return f.read()
        train, val = read('train'), read('valid')
        self.vocab = d2l.Vocab(list(train))
        self.train, self.val = self.windows(train), self.windows(val)

    def windows(self, text):
        corpus = d2l.tensor(self.vocab[list(text)])
        n = (len(corpus) - 1) // self.num_steps * self.num_steps
        return (corpus[:n].reshape(-1, self.num_steps),  # Inputs
                corpus[1:n + 1].reshape(-1, self.num_steps))  # Next tokens

    def get_dataloader(self, train):
        return self.get_tensorloader(self.train if train else self.val, train)

def fit_lm(model, data, num_steps=3000, lr=2e-3):
    """Train with AdamW for num_steps steps; return validation perplexity.

    Defined in :numref:`sec_gru`"""
    optimizer = nnx.Optimizer(model, optax.adamw(lr, weight_decay=0.01),
                              wrt=nnx.Param)
    d2l.train_lm(model, data, optimizer, num_steps)
    return d2l.lm_perplexity(model, data.val_dataloader())

def linear_scan(a, b, h0=None):
    """All states of h_t = a_t * h_{t-1} + b_t; time is the second axis.

    Defined in :numref:`sec_ssm`"""
    a = jnp.broadcast_to(a, b.shape)  # Decays may omit axes; broadcast them
    if h0 is not None:  # Fold the initial state into the first step
        b = b.at[:, 0].add(a[:, 0] * h0)
    def combine(earlier, later):
        (a1, b1), (a2, b2) = earlier, later
        return a1 * a2, a2 * b1 + b2
    return jax.lax.associative_scan(combine, (a, b), axis=1)[1]

class MinGRU(nnx.Module):
    """The minimal GRU: its gate and candidate read only the input.

    Defined in :numref:`sec_ssm`"""
    def __init__(self, num_inputs, num_hiddens, *, rngs):
        self.W_z = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)  # Decay
        self.W_h = nnx.Linear(num_inputs, num_hiddens, rngs=rngs)

    def coefficients(self, X):
        """Decays a_t and writes b_t of the linear recurrence."""
        Z = jax.nn.sigmoid(self.W_z(X))
        return Z, (1 - Z) * jnp.tanh(self.W_h(X))

    def __call__(self, X):  # X: (batch_size, num_steps, num_inputs)
        return linear_scan(*self.coefficients(X))

class S4D(nnx.Module):
    """A diagonal state space layer: N states per channel, fixed dynamics.

    Defined in :numref:`sec_ssm`"""
    def __init__(self, num_hiddens, num_states=16, dt_min=1e-3, dt_max=1e-1,
                 *, rngs):
        d, N = num_hiddens, num_states
        # Rates lambda_n = -(n + 1), stored as log(-lambda) to stay negative
        self.log_neg_rate = nnx.Param(
            jnp.tile(jnp.log(jnp.arange(1., N + 1)), (d, 1)))
        self.log_dt = nnx.Param(  # Log-uniform in [dt_min, dt_max]
            rngs.params.uniform((d, 1)) * math.log(dt_max / dt_min)
            + math.log(dt_min))
        self.C = nnx.Param(rngs.params.normal((d, N)) / math.sqrt(N))
        self.D = nnx.Param(jnp.ones(d))

    def discretize(self):
        """Zero-order hold with B = 1: decays a_bar and input weights b_bar."""
        rate = -jnp.exp(self.log_neg_rate[...])  # (d, N)
        dt_rate = jnp.exp(self.log_dt[...]) * rate
        return jnp.exp(dt_rate), jnp.expm1(dt_rate) / rate

    def __call__(self, X):  # X: (batch_size, num_steps, d)
        T = X.shape[1]
        a_bar, b_bar = self.discretize()
        powers = a_bar[..., None] ** jnp.arange(T)
        K = ((self.C[...] * b_bar)[..., None] * powers).sum(1)  # Kernel, (d, T)
        Y = jnp.fft.irfft(jnp.fft.rfft(X, n=2 * T, axis=1) *
                          jnp.fft.rfft(K.T, n=2 * T, axis=0), n=2 * T, axis=1)
        return Y[:, :T] + self.D[...] * X

    def step(self, x, h):
        """One step: x is (batch_size, d); the state h is (batch_size, d, N).

        Defined in :numref:`sec_ssm`"""
        a_bar, b_bar = self.discretize()
        h = a_bar * h + b_bar * x[..., None]
        return (h * self.C[...]).sum(-1) + self.D[...] * x, h

def init_timescales(layer, key, dt_min=1e-3, dt_max=1e-1):
    """Start a minGRU's gates at the decays exp(-dt) of S4D.

    Defined in :numref:`sec_ssm`"""
    dt = dt_min * (dt_max / dt_min) ** jax.random.uniform(
        key, layer.W_z.bias.shape)
    layer.W_z.bias[...] = -jnp.log(jnp.expm1(dt))  # sigmoid(b) = exp(-dt)
    return layer

@jax.custom_vjp
def _adjoint_scan(a, b):
    return d2l.linear_scan(a, b)

def adjoint_scan_forward(a, b):
    """Forward pass: the states, and what the backward pass needs.

    Defined in :numref:`sec_mamba`"""
    h = d2l.linear_scan(a, b)
    return h, (a, h)

def adjoint_scan_backward(saved, grad_h):
    """Backward pass: gradients of a and b by a scan in reverse time.

    Defined in :numref:`sec_mamba`"""
    a, h = saved
    # g_t = grad_h_t + a_{t+1} g_{t+1}: a linear recurrence from the end
    a_next = jnp.concatenate([a[:, 1:], jnp.ones_like(a[:, :1])], 1)
    g = d2l.linear_scan(a_next[:, ::-1], grad_h[:, ::-1])[:, ::-1]
    h_prev = jnp.concatenate([jnp.zeros_like(h[:, :1]), h[:, :-1]], 1)
    return g * h_prev, g

_adjoint_scan.defvjp(adjoint_scan_forward, adjoint_scan_backward)

def adjoint_scan(a, b):  #@save
    """All states of h_t = a_t * h_{t-1} + b_t, differentiated by a scan.

    Defined in :numref:`sec_mamba`"""
    return _adjoint_scan(jnp.broadcast_to(a, b.shape), b)

class SelectiveSSM(nnx.Module):
    """A diagonal SSM whose step sizes and write and read vectors depend on
    the input: N states for each of num_channels channels.

    Defined in :numref:`sec_mamba`"""
    def __init__(self, num_channels, num_states=16, dt_min=1e-3, dt_max=1e-1,
                 *, rngs):
        H, N, R = num_channels, num_states, math.ceil(num_channels / 16)
        self.log_neg_rate = nnx.Param(  # Rates -1, ..., -N in every channel
            jnp.tile(jnp.log(jnp.arange(1., N + 1)), (H, 1)))
        def dt_bias(key, shape, dtype=jnp.float32):  # softplus(bias) = dt
            dt = dt_min * (dt_max / dt_min) ** jax.random.uniform(key, shape)
            return (dt + jnp.log(-jnp.expm1(-dt))).astype(dtype)
        def small(key, shape, dtype=jnp.float32):  # U(-R^-1/2, R^-1/2)
            return jax.random.uniform(key, shape, dtype, -R ** -0.5, R ** -0.5)
        self.W_dt = nnx.Sequential(  # Step size per channel, through rank R
            nnx.Linear(H, R, use_bias=False, rngs=rngs),
            nnx.Linear(R, H, kernel_init=small, bias_init=dt_bias, rngs=rngs))
        self.W_B = nnx.Linear(H, N, use_bias=False, rngs=rngs)  # Write vector
        self.W_C = nnx.Linear(H, N, use_bias=False, rngs=rngs)  # Read vector
        self.D = nnx.Param(jnp.ones(H))

    def coefficients(self, U):
        """Log-decays and writes, each of shape (batch_size, T, H, N).

        Defined in :numref:`sec_mamba`"""
        dt = jax.nn.softplus(self.W_dt(U))  # Step sizes, (batch_size, T, H)
        log_a = dt[..., None] * -jnp.exp(self.log_neg_rate[...])
        return log_a, (dt * U)[..., None] * self.W_B(U)[..., None, :]

    def __call__(self, U):  # U: (batch_size, num_steps, num_channels)
        log_a, b = self.coefficients(U)
        H = adjoint_scan(jnp.exp(log_a), b)  # States, (batch_size, T, H, N)
        return (H * self.W_C(U)[..., None, :]).sum(-1) + self.D[...] * U

class Mamba(nnx.Module):
    """The Mamba layer: a gated selective SSM between two projections.

    Defined in :numref:`sec_mamba`"""
    def __init__(self, num_hiddens, num_states=16, expand=2, conv_width=4,
                 *, rngs, **ssm_args):
        H = expand * num_hiddens
        self.W_in = nnx.Linear(num_hiddens, 2 * H, use_bias=False, rngs=rngs)
        self.conv = nnx.Conv(H, H, (conv_width,), feature_group_count=H,
                             padding=((conv_width - 1, 0),), rngs=rngs)  # Causal
        self.ssm = SelectiveSSM(H, num_states, rngs=rngs, **ssm_args)
        self.W_out = nnx.Linear(H, num_hiddens, use_bias=False, rngs=rngs)

    def ssm_input(self, X):
        """The input of the SSM and the gate branch."""
        U, Z = jnp.split(self.W_in(X), 2, -1)
        return jax.nn.silu(self.conv(U)), Z

    def __call__(self, X):  # X: (batch_size, num_steps, num_hiddens)
        U, Z = self.ssm_input(X)
        return self.W_out(self.ssm(U) * jax.nn.silu(Z))

def decay_mask(log_a):
    """L[t, s] = a_{s+1} * ... * a_t for s <= t, and 0 for s > t.

    Defined in :numref:`sec_matrix-state`"""
    cum = log_a.cumsum(-1)
    log_L = cum[..., :, None] - cum[..., None, :]  # Sum of log a over (s, t]
    causal = jnp.tril(jnp.ones(log_L.shape[-2:], dtype=bool))
    return jnp.exp(jnp.where(causal, log_L, -jnp.inf))

def chunked_linear_attention(Q, K, V, log_a, chunk_size):
    """y_t = S_t q_t with S_t = S_{t-1} diag(a_t) + v_t k_t^T, in chunks.

    Q, K: (batch_size, T, d_k); V: (batch_size, T, d_v); log_a: (batch_size,
    T, d_k) for a decay per key coordinate, or (batch_size, T) for one per step

    Defined in :numref:`sec_matrix-state`"""
    B, T, m = Q.shape[0], Q.shape[1], chunk_size
    if log_a.ndim == 2:  # The same decay for every key coordinate
        log_a = log_a[..., None]
    Q, K, V, log_a = (X.reshape(B, T // m, m, -1) for X in (Q, K, V, log_a))
    # 1. Within chunks: the dual form, with one mask per key coordinate
    L = decay_mask(log_a.mT)  # (B, T / m, 1 or d_k, m, m), entries <= 1
    if L.shape[2] == 1:  # One mask for all coordinates: a masked Q K^T
        Y = (L[:, :, 0] * (Q @ K.mT)) @ V
    else:
        Y = (L * Q.mT[..., None] * K.mT[..., None, :]).sum(2) @ V
    # 2. Across chunks: the writes of each chunk, decayed to its end, and
    #    a scan over chunks for the state at every chunk's end
    cum = log_a.cumsum(2)  # Log decay from the chunk's start to each step
    writes = V.mT @ (K * jnp.exp(cum[:, :, -1:] - cum))
    S = d2l.linear_scan(jnp.exp(cum[:, :, -1:]), writes)
    S = jnp.concatenate([jnp.zeros_like(S[:, :1]), S[:, :-1]], 1)  # At start
    # 3. Every step reads the state at its chunk's start, decayed to it
    return (Y + (Q * jnp.exp(cum)) @ S.mT).reshape(B, T, -1)

class GatedLinearAttention(nnx.Module):
    """Multi-head linear attention with a gated decay per key coordinate.

    Defined in :numref:`sec_matrix-state`"""
    gate_scale = 16  # log a = log sigmoid(z) / 16 keeps decays near one

    def __init__(self, num_hiddens, num_heads, chunk_size=16, conv_width=4,
                 *, rngs):
        d = num_hiddens
        self.num_heads, self.chunk_size = num_heads, chunk_size
        self.W_qkv = nnx.Linear(d, 3 * d, use_bias=False, rngs=rngs)
        self.conv = nnx.Conv(3 * d, 3 * d, (conv_width,),
                             feature_group_count=3 * d,  # One filter each
                             padding=((conv_width - 1, 0),), rngs=rngs)  # Causal
        self.W_a = self.decay_gate(d, rngs)
        self.W_g = nnx.Linear(d, d, rngs=rngs)  # Output gate
        self.W_o = nnx.Linear(d, d, use_bias=False, rngs=rngs)
        self.init_decays(4, 256)

    def decay_gate(self, d, rngs, rank=16):
        """Low-rank projection to one gate per key coordinate."""
        return nnx.Sequential(nnx.Linear(d, rank, use_bias=False, rngs=rngs),
                              nnx.Linear(rank, d, rngs=rngs))

    def init_decays(self, tau_min, tau_max):
        """Start the gates at decays exp(-1/tau), tau log-uniform in range."""
        b = self.W_a.layers[-1].bias
        tau = jnp.logspace(math.log10(tau_min), math.log10(tau_max), b.shape[0])
        b[...] = -jnp.log(jnp.expm1(self.gate_scale / tau))

    def decays(self, X):
        """Log decays, (batch_size * num_heads, T, d_k)."""
        return self.split_heads(jax.nn.log_sigmoid(self.W_a(X))
                                / self.gate_scale)

    def split_heads(self, Z):
        """(batch_size, T, num_heads * e) -> (batch_size * num_heads, T, e)."""
        B, T, _ = Z.shape
        return Z.reshape(B, T, self.num_heads, -1).swapaxes(1, 2).reshape(
            B * self.num_heads, T, -1)

    def memory(self, X, Q, K, V, log_a):
        """Outputs of the decayed matrix memory of every head."""
        return chunked_linear_attention(Q, K, V, log_a, self.chunk_size)

    def __call__(self, X):  # X: (batch_size, num_steps, num_hiddens)
        B, T, d = X.shape
        QKV = jax.nn.silu(self.conv(self.W_qkv(X)))
        Q, K, V = (self.split_heads(Z) for Z in jnp.split(QKV, 3, -1))
        unit = lambda Z: Z / jnp.maximum(
            jnp.linalg.norm(Z, axis=-1, keepdims=True), 1e-12)
        Q, K = unit(Q), unit(K)  # Unit length
        Y = self.memory(X, Q, K, V, self.decays(X))  # (B * num_heads, T, d / h)
        Y = Y / jnp.sqrt((Y ** 2).mean(-1, keepdims=True) + 1e-6)
        Y = Y.reshape(B, self.num_heads, T, -1).swapaxes(1, 2).reshape(B, T, d)
        return self.W_o(Y * jax.nn.silu(self.W_g(X)))

def delta_rule_writes(K, V, beta, log_a):
    """W and Z with corrected values U = W - Z S_0^T, for every chunk.

    K: (..., m, d_k); V: (..., m, d_v); beta, log_a: (..., m)

    Defined in :numref:`sec_deltanet`"""
    L = d2l.decay_mask(log_a)  # Products of decays within the chunk
    gamma = jnp.exp(log_a.cumsum(-1))[..., None]  # Decay since start
    beta = beta[..., None]
    M = jnp.eye(K.shape[-2]) + (
        beta * jnp.tril(L, -1) * (K @ K.mT))  # Lower triangular
    W = jax.scipy.linalg.solve_triangular(M, beta * V, lower=True)
    Z = jax.scipy.linalg.solve_triangular(M, beta * gamma * K, lower=True)
    return W, Z

def chunked_delta_rule(Q, K, V, beta, log_a, chunk_size):
    """Outputs y_t = S_t q_t of the gated delta rule, computed in chunks.

    Q, K: (batch_size, T, d_k); V: (batch_size, T, d_v);
    beta, log_a: (batch_size, T)

    Defined in :numref:`sec_deltanet`"""
    B, T, m = Q.shape[0], Q.shape[1], chunk_size
    Q, K, V = (X.reshape(B, T // m, m, -1) for X in (Q, K, V))
    beta, log_a = beta.reshape(B, T // m, m), log_a.reshape(B, T // m, m)
    W, Z = delta_rule_writes(K, V, beta, log_a)
    L, cum = d2l.decay_mask(log_a), log_a.cumsum(-1)
    # Each chunk maps the memory at its start to S A + W_end at its end
    to_end = jnp.exp(cum[..., -1:] - cum)[..., None]
    I = jnp.eye(K.shape[-1])
    A = jnp.exp(cum[..., -1, None, None]) * I - (Z * to_end).mT @ K
    def combine(earlier, later):  # A scan over chunks, with matrix products
        (A1, W1), (A2, W2) = earlier, later
        return A1 @ A2, W1 @ A2 + W2
    W_end = jax.lax.associative_scan(combine, (A, (W * to_end).mT @ K),
                                     axis=1)[1]
    S = jnp.concatenate([jnp.zeros_like(W_end[:, :1]), W_end[:, :-1]], 1)
    U = W - Z @ S.mT  # Corrected values, with the memory at each start
    Y = jnp.exp(cum)[..., None] * (Q @ S.mT) + (L * (Q @ K.mT)) @ U
    return Y.reshape(B, T, -1)

class GatedDeltaNet(d2l.GatedLinearAttention):
    """Gated linear attention with one decay per head, whose memory is
    written by the delta rule.

    Defined in :numref:`sec_deltanet`"""
    gate_scale = 1

    def __init__(self, num_hiddens, num_heads, chunk_size=32, conv_width=4,
                 *, rngs):
        super().__init__(num_hiddens, num_heads, chunk_size, conv_width,
                         rngs=rngs)
        self.W_beta = nnx.Linear(num_hiddens, num_heads, rngs=rngs)

    def decay_gate(self, d, rngs):
        return nnx.Sequential(nnx.Linear(d, self.num_heads, rngs=rngs))

    def decays(self, X):  # (batch_size * num_heads, T)
        return self.split_heads(jax.nn.log_sigmoid(self.W_a(X)))[..., 0]

    def memory(self, X, Q, K, V, log_a):
        beta = self.split_heads(jax.nn.sigmoid(self.W_beta(X)))[..., 0]
        return chunked_delta_rule(Q, K, V, beta, log_a, self.chunk_size)

class ResNet18(nnx.Module):
    """A slightly modified ResNet-18 (small stem, no max-pool).

    Defined in :numref:`sec_multi_gpu_concise`"""
    def __init__(self, num_classes=10, in_channels=1, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        # Stem conv without bias (the batch normalization after it removes
        # it); running averages decay 0.9, two-pass variance
        self.net = nnx.Sequential(
            nnx.Conv(in_channels, 64, (3, 3), (1, 1), padding='same',
                     use_bias=False, kernel_init=he, rngs=rngs),
            nnx.BatchNorm(64, momentum=0.9, use_fast_variance=False,
                          rngs=rngs), nnx.relu,
            d2l.Residual(64, in_channels=64, rngs=rngs),
            d2l.Residual(64, in_channels=64, rngs=rngs),
            d2l.Residual(128, use_1x1conv=True, strides=(2, 2),
                         in_channels=64, rngs=rngs),
            d2l.Residual(128, in_channels=128, rngs=rngs),
            d2l.Residual(256, use_1x1conv=True, strides=(2, 2),
                         in_channels=128, rngs=rngs),
            d2l.Residual(256, in_channels=256, rngs=rngs),
            d2l.Residual(512, use_1x1conv=True, strides=(2, 2),
                         in_channels=256, rngs=rngs),
            d2l.Residual(512, in_channels=512, rngs=rngs),
            lambda x: x.mean(axis=(1, 2)),
            nnx.Linear(512, num_classes, kernel_init=he, rngs=rngs))

    def __call__(self, x):
        return self.net(x)

def windows(corpus, batch_size, num_steps=128, *, sharding=None):
    """Endless minibatches of windows at random offsets, each position with
    the next byte as its target; `sharding` spreads them over devices.

    Defined in :numref:`sec_fast_transformer`"""
    offsets = jnp.arange(num_steps + 1)
    while True:
        starts = jax.random.randint(d2l.get_key(), (batch_size, 1), 0,
                                    len(corpus) - num_steps)
        if sharding is not None:
            starts = jax.device_put(starts, sharding)
        W = corpus[starts + offsets].astype(jnp.int32)
        yield W[:, :-1], W[:, 1:]

class TabularMDP:
    """A finite MDP as dense arrays: P[s, a, s'] and r[s, a].

    Defined in :numref:`sec_mdp`"""
    def __init__(self, P, r, gamma):
        self.P, self.r, self.gamma = P, r, gamma
        self.num_states, self.num_actions = r.shape

    @classmethod
    def from_gym(cls, env, gamma):
        """Read the transition table Gymnasium exposes as env.unwrapped.P."""
        n_s, n_a = env.observation_space.n, env.action_space.n
        P, r = np.zeros((n_s, n_a, n_s)), np.zeros((n_s, n_a))
        for s, actions in env.unwrapped.P.items():
            for a, outcomes in actions.items():
                for p, s_next, reward, _ in outcomes:
                    P[s, a, s_next] += p      # several outcomes may share s'
                    r[s, a] += p * reward     # r(s,a) is the expected reward
        return cls(P, r, gamma)

    def backup(self, V):
        """Q(s, a) = r(s, a) + gamma * sum_{s'} P(s'|s, a) V(s') for every
        state-action pair: the bracket of the Bellman equations.

        Defined in :numref:`sec_valueiter`"""
        return self.r + self.gamma * self.P @ V

    def policy_value(self, pi):
        """V^pi = (I - gamma P_pi)^{-1} r_pi exactly, for pi[s, a] the
        probability of action a at state s (one-hot rows: deterministic).
        Needs gamma < 1 here: from_gym's terminal states loop on themselves.

        Defined in :numref:`sec_valueiter`"""
        P_pi = np.einsum('sa,sat->st', pi, self.P)
        r_pi = (pi * self.r).sum(axis=1)
        return np.linalg.solve(np.eye(self.num_states) - self.gamma * P_pi,
                               r_pi)

class VerifierTask:
    """Four prompts, nine candidate responses shared by all of them, an
    exact-match verifier, and a reference policy that rarely hedges.

    Defined in :numref:`sec_mdp`"""
    prompts = ['2+2', '3*4', '8*8', '9*9']
    answers = ['4', '12', '64', '81']
    resp = ['1', '3', '4', '7', '12', '19', '23', '35',
            'one of 4, 12, 64 or 81']          # the last one is the hedge

    def __init__(self):
        self.theta_ref = np.zeros((len(self.prompts), len(self.resp)))
        self.theta_ref[:, -1] = -4.0           # the reference rarely hedges
        self.pi_ref = self.policy(self.theta_ref)

    def verify(self, x, y):
        """The verifier: reward 1 iff response y is prompt x's answer."""
        return float(self.resp[y] == self.answers[x])

    @staticmethod
    def policy(theta):
        """pi(y | x): a softmax over each prompt's row of preferences."""
        z = np.exp(theta - theta.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def success(self, pi, reward=None):
        """E r(x, y), y ~ pi(. | x), averaged over the prompts; r = verify
        unless another reward function is given."""
        reward = self.verify if reward is None else reward
        R = np.array([[reward(x, y) for y in range(len(self.resp))]
                      for x in range(len(self.prompts))])
        return (pi * R).sum(1).mean()

def value_iteration(mdp, num_iters):
    """Sweep V <- max_a backup(V); return the whole history of iterates.

    Defined in :numref:`sec_valueiter`"""
    V, history = np.zeros(mdp.num_states), []
    for _ in range(num_iters):
        V = mdp.backup(V).max(axis=1)
        history.append(V)
    return np.array(history)

def policy_evaluation(mdp, pi, num_iters):
    """Same sweep with the max replaced by an average under pi(a|s).

    Defined in :numref:`sec_valueiter`"""
    V, history = np.zeros(mdp.num_states), []
    for _ in range(num_iters):
        V = (pi * mdp.backup(V)).sum(axis=1)
        history.append(V)
    return np.array(history)

def evaluate(env, policy, num_episodes, gamma=1.0, rng=None):
    """Mean discounted return of an acting policy over Monte Carlo episodes.

    A policy is any function `policy(obs, rng) -> action`; a deterministic
    policy ignores `rng`, a stochastic one draws from it.

    Defined in :numref:`sec_valueiter`"""
    total = 0.0
    for _ in range(num_episodes):
        obs, done, discount = env.reset()[0], False, 1.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs, rng))
            done = terminated or truncated
            total, discount = total + discount * reward, discount * gamma
    return total / num_episodes

def epsilon_greedy(q, epsilon, rng):
    """Explore with probability epsilon, else act greedily on the values q.

    Defined in :numref:`sec_qlearning`"""
    if rng.random() < epsilon:
        return int(rng.integers(len(q)))
    # Ties are coin flips: np.argmax would always pick action 0 on a zero
    # table, and an agent that only ever moves left never finds the goal.
    return int(rng.choice(np.flatnonzero(q == q.max())))

def linear_schedule(start, end, num_steps):
    """step -> value, interpolated from start to end, then held at end.

    Defined in :numref:`sec_qlearning`"""
    return lambda step: end + (start - end) * max(0.0, 1.0 - step / num_steps)

def q_learning(seed, Q, visits, env, num_episodes, gamma,
               alpha=lambda n: 1 / (1 + 0.1 * n), epsilon=None):
    """Tabular Q-learning; updates Q in place, yields each episode's return.
    `alpha(n)` is the step size at the n-th visit of a pair; `epsilon(k)` is
    the exploration rate in episode k (default: 1 to 0.05 over the first
    half of the episodes, then held).

    Defined in :numref:`sec_qlearning`"""
    rng = np.random.default_rng(seed)
    if epsilon is None:
        epsilon = linear_schedule(1.0, 0.05, num_episodes // 2)
    env.reset(seed=seed)
    for episode in range(num_episodes):
        s, done, ret = env.reset()[0], False, 0.0
        while not done:
            a = epsilon_greedy(Q[s], epsilon(episode), rng)
            s_next, r, terminated, truncated, _ = env.step(a)
            visits[s, a] += 1
            delta = r + gamma * (1 - terminated) * Q[s_next].max() - Q[s, a]
            Q[s, a] += alpha(visits[s, a]) * delta
            s, done, ret = s_next, terminated or truncated, ret + r
        yield ret

class Policy(nnx.Module):
    """A policy network and its optimizer; pi(.|s) is the softmax of the
    network's outputs at s.

    Defined in :numref:`sec_imitation`"""
    def __init__(self, policy, lr=1e-2):
        self.policy = policy
        self.opt_pi = nnx.Optimizer(policy, d2l.shared_adam(lr),
                                    wrt=nnx.Param)

    def log_prob(self, obs, act, policy=None):
        """log pi(a|s). Gradients flow w.r.t. the module you differentiate;
        the update functions pass that module back in as `policy`."""
        policy = self.policy if policy is None else policy
        logp = jax.nn.log_softmax(policy(obs), axis=-1)
        return jnp.take_along_axis(logp, act[:, None], axis=-1).squeeze(-1)

    @classmethod
    def tabular(cls, num_states, num_actions, lr=0.1, seed=None):
        """One preference theta_{s,a} per state-action pair: an embedding."""
        rngs = nnx.Rngs(d2l.get_key() if seed is None else seed)
        return cls(nnx.Embed(num_states, num_actions, rngs=rngs,
                             embedding_init=nnx.initializers.zeros_init()),
                   lr)

    def act(self, obs, rng):
        """Sample an action; numpy in, int out, the acting protocol of evaluate.

        Defined in :numref:`sec_imitation`"""
        logits = d2l.forward_np(self.policy, obs)
        probs = np.exp(logits - logits.max())
        return int(rng.choice(len(probs), p=probs / probs.sum()))

    def act_greedy(self, obs, rng=None):
        return int(d2l.forward_np(self.policy, obs).argmax())

    def log_prob_np(self, obs, act):
        """log pi(a|s) as numpy, on a batch padded to a bucket length, so that
        each operation compiles once per bucket rather than once per length.

        Defined in :numref:`sec_imitation`"""
        n = len(act)
        size = d2l._bucket(n)
        return np.asarray(self.log_prob(jnp.asarray(d2l._pad(obs, size)),
                                        jnp.asarray(d2l._pad(act, size))))[:n]

@nnx.jit
def _policy_pass(ac, obs, act, w, n):
    """-sum(w * log pi(act|obs)) / n and one optimizer step; padded rows
    carry w = 0 and add nothing.

    Defined in :numref:`sec_imitation`"""
    loss, grads = nnx.value_and_grad(
        lambda policy: -(w * ac.log_prob(obs, act, policy)).sum() / n)(
            ac.policy)
    ac.opt_pi.update(ac.policy, grads)
    return loss

def policy_step(ac, obs, act, weight):
    """One ascent step on (1/n) sum_t w_t log pi(a_t|s_t) over a batch of
    n states and actions; the weights w arrive as numpy, that is, as data.
    The batch is padded to a bucket length, so the compiled pass is reused.

    Defined in :numref:`sec_imitation`"""
    n = len(weight)
    size = d2l._bucket(n)
    return float(_policy_pass(ac, d2l._pad(obs, size), d2l._pad(act, size),
                              d2l._pad(np.float32(weight), size), n))

def verifier_step(task, theta, K, reward, rng, weighting, lr=2.0):
    """One REINFORCE step on a VerifierTask: K responses per prompt, each
    score e_y - pi(.|x) (eq_softmax_score) weighted by weighting(rewards of
    the prompt's K responses), averaged over the K, one ascent step.

    Defined in :numref:`sec_policygradient`"""
    pi, g = task.policy(theta), np.zeros_like(theta)
    eye = np.eye(len(task.resp))
    for x in range(len(task.prompts)):         # one group of K per prompt
        ys = rng.choice(len(task.resp), K, p=pi[x])
        w = weighting(np.array([reward(x, y) for y in ys]))
        g[x] = (eye[ys] - pi[x]).T @ w / K
    return theta + lr * g

class Batch:
    """A flat batch of transitions, plus the episode boundaries.

    Defined in :numref:`sec_policygradient`"""
    def __init__(self, obs, act, rew, next_obs, term, ep_ends):
        self.obs, self.act, self.rew = obs, act, rew
        self.next_obs, self.term = next_obs, term
        self.ep_ends = ep_ends    # one past the last step of each episode

    def __len__(self):
        return len(self.rew)

    def episodes(self):
        """Yield one slice per episode."""
        start = 0
        for end in self.ep_ends:
            yield slice(start, end)
            start = end

    def episode_returns(self, gamma=1.0):
        """R(tau), the discounted return, one number per episode."""
        return np.array([(gamma ** np.arange(ep.stop - ep.start)
                          * self.rew[ep]).sum() for ep in self.episodes()])

    def per_step(self, values):
        """Repeat one value per episode over that episode's steps."""
        return np.repeat(values, np.diff(self.ep_ends, prepend=0))

    def backward_scan(self, x, factor):
        """y_t = x_t + factor * y_{t+1}, restarted at every episode boundary.

        Defined in :numref:`sec_baselines`"""
        y = np.zeros_like(x)
        for ep in self.episodes():
            running = 0.0
            for t in reversed(range(ep.start, ep.stop)):
                running = x[t] + factor * running
                y[t] = running
        return y

    def reward_to_go(self, gamma):
        """G_t: the discounted return of the rest of its episode, by one scan.

        Defined in :numref:`sec_baselines`"""
        return self.backward_scan(self.rew, gamma)

    def truncation_bootstrap(self, gamma, V):
        """gamma^(T-t) (1 - term_T) V(s_T) for every step t of each episode:
        the discounted value of the future that a time-limit cut did not see.
        V maps a batch of numpy states to numpy values.

        Defined in :numref:`sec_deeprl`"""
        tail = np.zeros_like(self.rew)
        last = self.ep_ends - 1
        tail[last] = gamma * (1 - self.term[last]) * V(self.next_obs[last])
        return self.backward_scan(tail, gamma)

    def td_target(self, bootstrap, gamma):
        """r_t + gamma (1 - terminated) V(s'), by a numpy bootstrap.

        Defined in :numref:`sec_actorcritic`"""
        return self.rew + gamma * (1 - self.term) * bootstrap(self.next_obs)

    def gae(self, value_fn, gamma, lam):
        """GAE(gamma, lam): the reward-to-go scan, run on the TD errors.

        Defined in :numref:`sec_actorcritic`"""
        delta = self.td_target(value_fn, gamma) - value_fn(self.obs)
        return self.backward_scan(delta, gamma * lam)

def rollout(env, policy, num_episodes, rng):
    """Collect complete episodes from `policy(obs, rng) -> action` as a
    Batch. `term` records `terminated`, never `truncated`: a time-limit cut
    is not a terminal state. All sampling runs through the numpy `rng`.

    Defined in :numref:`sec_policygradient`"""
    cols, ep_ends = [[] for _ in range(5)], []
    for _ in range(num_episodes):
        obs, done = env.reset()[0], False
        while not done:
            act = policy(obs, rng)
            next_obs, reward, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            for col, val in zip(cols, (obs, act, reward, next_obs,
                                       float(terminated))):
                col.append(val)
            obs = next_obs
        ep_ends.append(len(cols[0]))
    obs, act, rew, next_obs, term = (np.asarray(c) for c in cols)
    return Batch(obs, act, rew.astype(np.float32), next_obs,
                 term.astype(np.float32), np.asarray(ep_ends))

def normalize(x, eps=1e-8):
    """Center a batch of weights and rescale them to unit spread.

    Defined in :numref:`sec_baselines`"""
    return (x - x.mean()) / (x.std() + eps)

def group_step(task, theta, K, reward, rng, lr=2.0):
    """verifier_step with the group-normalized weight (GRPO): the rewards
    of each prompt's K responses, standardized within the group.

    Defined in :numref:`sec_baselines`"""
    return d2l.verifier_step(task, theta, K, reward, rng, normalize, lr)

class ActorCritic(d2l.Policy):
    """A policy and a value function, each with its own optimizer.

    Defined in :numref:`sec_deeprl`"""
    def __init__(self, policy, value, lr=1e-2):
        super().__init__(policy, lr)
        self.value = value
        self.opt_v = nnx.Optimizer(value, d2l.shared_adam(lr), wrt=nnx.Param)

    def V(self, obs, value=None):
        value = self.value if value is None else value
        return value(obs).squeeze(-1)

    def value_np(self, obs):
        """V(s) for a batch of states, padded to a bucket length."""
        out = d2l.forward_np(self.value, d2l._pad(obs, d2l._bucket(len(obs))))
        return out[:len(obs), 0]

    @classmethod
    def tabular(cls, num_states, num_actions, lr=0.1, seed=None):
        """Preference and value tables: two zero-initialized embeddings."""
        rngs = nnx.Rngs(d2l.get_key() if seed is None else seed)
        zeros = nnx.initializers.zeros_init()
        return cls(nnx.Embed(num_states, num_actions,
                             embedding_init=zeros, rngs=rngs),
                   nnx.Embed(num_states, 1, embedding_init=zeros, rngs=rngs),
                   lr)

    @classmethod
    def mlp(cls, obs_dim, num_actions, hidden=64, lr=1e-2, seed=None):
        """The same container with the tables replaced by one-hidden-layer nets.

        Defined in :numref:`sec_deeprl`"""
        rngs = nnx.Rngs(d2l.get_key() if seed is None else seed)
        def net(out):
            return nnx.Sequential(d2l.Linear(obs_dim, hidden, rngs=rngs), jnp.tanh,
                                  d2l.Linear(hidden, out, rngs=rngs))
        return cls(net(num_actions), net(1), lr)

@nnx.jit
def _value_pass(value, opt, obs, target, mask):
    """Masked mean squared error over the real rows, one optimizer step.

    Defined in :numref:`sec_deeprl`"""
    loss, grads = nnx.value_and_grad(lambda v: (mask * (
        v(obs).squeeze(-1) - target) ** 2).sum() / mask.sum())(value)
    opt.update(value, grads)
    return loss

def fit_value(ac, obs, target, num_steps=1):
    """Regress the value head on a fixed target: eq_value_baseline for nets.
    The batch is padded to a bucket length, so the compiled pass is reused.

    Defined in :numref:`sec_deeprl`"""
    size = d2l._bucket(len(target))
    mask = (np.arange(size) < len(target)).astype(np.float32)
    obs, target = d2l._pad(obs, size), d2l._pad(np.float32(target), size)
    for _ in range(num_steps):
        loss = _value_pass(ac.value, ac.opt_v, obs, target, mask)
    return float(loss)

class GaussianHead(nnx.Module):
    """A Gaussian over actions: a trunk of hidden features, a linear mean,
    and a log standard deviation that is one learned vector or, with
    `state_std=True`, a second linear head clamped to [-5, 2]. The default
    trunk is one tanh layer; any module with `hidden` outputs will do.
    `zero_mean=True` sets the mean layer to zero, so that mu = 0 at first.

    Defined in :numref:`sec_deeprl`"""
    def __init__(self, obs_dim, act_dim, hidden, rngs, state_std=False,
                 trunk=None, zero_mean=False):
        self.trunk = trunk if trunk is not None else nnx.Sequential(
            d2l.Linear(obs_dim, hidden, rngs=rngs), jnp.tanh)
        self.mean = d2l.Linear(hidden, act_dim, rngs=rngs)
        if zero_mean:
            self.mean.kernel[...] = 0.0
            self.mean.bias[...] = 0.0
        self.state_std = state_std
        self.log_std = (d2l.Linear(hidden, act_dim, rngs=rngs) if state_std
                        else nnx.Param(jnp.zeros(act_dim)))

    def __call__(self, obs):
        h = self.trunk(obs)
        if self.state_std:
            return self.mean(h), jnp.exp(jnp.clip(self.log_std(h), -5, 2))
        return self.mean(h), jnp.exp(self.log_std[...])

class GaussianPolicy(ActorCritic):
    """The same interface over a Normal instead of a softmax; nothing that
    consumes the interface changes. `head_args` go to GaussianHead.

    Defined in :numref:`sec_deeprl`"""
    def __init__(self, obs_dim, act_dim, hidden=64, lr=1e-2, seed=None,
                 **head_args):
        rngs = nnx.Rngs(d2l.get_key() if seed is None else seed)
        super().__init__(GaussianHead(obs_dim, act_dim, hidden, rngs,
                                      **head_args),
                         nnx.Sequential(d2l.Linear(obs_dim, hidden, rngs=rngs),
                                        jnp.tanh,
                                        d2l.Linear(hidden, 1, rngs=rngs)), lr)

    def log_prob(self, obs, act, policy=None):
        mean, std = (self.policy if policy is None else policy)(obs)
        return jax.scipy.stats.norm.logpdf(act, mean, std).sum(-1)

    def act(self, obs, rng):
        mean, std = d2l.forward_np(self.policy, obs)   # compiled, as in tabular
        return mean + std * rng.standard_normal(mean.shape, dtype=np.float32)

    def act_greedy(self, obs, rng=None):
        return d2l.forward_np(self.policy, obs)[0]

@nnx.jit
def _ppo_step(policy, opt, obs, act, adv, logp_old, mask, epsilon,
              entropy_coef, use_clip):
    def loss_fn(policy):
        logp_all = jax.nn.log_softmax(policy(obs), axis=-1)
        logp = jnp.take_along_axis(logp_all, act[:, None], -1).squeeze(-1)
        rho = jnp.exp(logp - logp_old)
        surr = jnp.where(use_clip, jnp.minimum(
            rho * adv, jnp.clip(rho, 1 - epsilon, 1 + epsilon) * adv),
            rho * adv)
        entropy = -(jnp.exp(logp_all) * logp_all).sum(-1)
        loss = -(mask * (surr + entropy_coef * entropy)).sum() / mask.sum()
        return loss, (rho, logp, entropy)
    (_, (rho, logp, entropy)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True)(policy)
    opt.update(policy, grads)
    n = mask.sum()
    return ((mask * (jnp.abs(rho - 1) > epsilon)).sum() / n,
            (mask * (logp_old - logp)).sum() / n, (mask * entropy).sum() / n)

def ppo_epochs(ac, obs, act, adv, logp_old, epsilon, num_epochs,
               entropy_coef=0.01, use_clip=True):
    """num_epochs clipped-surrogate passes on one frozen batch of states
    and actions; returns [num_epochs, 3] numpy diagnostics, recorded before
    each pass: fraction of ratios outside the band, approximate KL, mean
    policy entropy.

    Defined in :numref:`sec_ppo`"""
    size = d2l._bucket(len(adv))
    mask = (np.arange(size) < len(adv)).astype(np.float32)
    obs, act, adv, logp_old = (d2l._pad(x, size) for x in
                               (obs, act, adv, logp_old))
    step = nnx.cached_partial(_ppo_step, ac.policy, ac.opt_pi)
    return np.array([step(obs, act, adv, logp_old, mask, epsilon,
                          entropy_coef, use_clip)
                     for _ in range(num_epochs)])

class ReplayBuffer:
    """A ring of transitions in preallocated numpy; sample() returns a Batch.

    Defined in :numref:`sec_dqn`"""
    def __init__(self, capacity, obs_dim):
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros(capacity, np.int64)
        self.rew = np.zeros(capacity, np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.term = np.zeros(capacity, np.float32)
        self.capacity, self.size, self.ptr = capacity, 0, 0

    def add(self, obs, act, rew, next_obs, term):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.next_obs[i], self.term[i] = next_obs, term
        self.ptr, self.size = (i + 1) % self.capacity, min(self.size + 1,
                                                           self.capacity)

    def __len__(self):
        return self.size

    def sample(self, batch_size, rng):
        """batch_size transitions drawn uniformly, each a one-step
        fragment: ep_ends marks every transition as its own episode."""
        i = rng.integers(self.size, size=batch_size)
        return d2l.Batch(self.obs[i], self.act[i], self.rew[i],
                         self.next_obs[i], self.term[i],
                         np.arange(1, batch_size + 1))

class SquashedGaussianPolicy(nnx.Module):
    """d2l.GaussianHead with SAC's network (two ReLU layers and a
    state-dependent standard deviation), squashed through
    a = scale * tanh(u).

    Defined in :numref:`sec_sac`"""
    def __init__(self, obs_dim, act_dim, hidden=64, scale=2.0, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        trunk = nnx.Sequential(
            d2l.Linear(obs_dim, hidden, rngs=rngs), jax.nn.relu,
            d2l.Linear(hidden, hidden, rngs=rngs), jax.nn.relu)
        self.policy = d2l.GaussianHead(obs_dim, act_dim, hidden, rngs,
                                       state_std=True, trunk=trunk)
        self.scale = scale

    def log_prob_u(self, u, mean, std):
        """log pi at a = scale tanh(u), from the pre-squash u the sampler
        keeps."""
        logdet = 2 * (jnp.log(2.0) - u - jax.nn.softplus(-2 * u))
        return (jax.scipy.stats.norm.logpdf(u, mean, std)
                - logdet - jnp.log(self.scale)).sum(-1)

    def sample(self, obs, key):
        """A reparameterized action and its log-probability, differentiable."""
        mean, std = self.policy(obs)
        u = mean + std * jax.random.normal(key, std.shape)
        return self.scale * jnp.tanh(u), self.log_prob_u(u, mean, std)

    def act(self, obs, rng):
        mean, std = d2l.forward_np(self.policy, obs)   # compiled forward
        u = mean + std * rng.standard_normal(mean.shape, dtype=np.float32)
        return self.scale * np.tanh(u)

    def act_logp(self, obs, rng):
        """A sampled action and its log-probability, numpy in and out."""
        mean, std = d2l.forward_np(self.policy, obs)
        u = mean + std * rng.standard_normal(mean.shape, dtype=np.float32)
        return (self.scale * np.tanh(u),
                np.asarray(self.log_prob_u(u, mean, std)))

    def act_greedy(self, obs, rng=None):
        return self.scale * np.tanh(d2l.forward_np(self.policy, obs)[0])

class ReplayBufferC(d2l.ReplayBuffer):
    """d2l.ReplayBuffer with the action column widened to float vectors.

    Defined in :numref:`sec_sac`"""
    def __init__(self, capacity, obs_dim, act_dim):
        super().__init__(capacity, obs_dim)
        self.act = np.zeros((capacity, act_dim), np.float32)

@nnx.jit
def update_D(X, Z, net_D, net_G, optimizer_D):
    """Update the critic; return the summed log loss, halved.

    Defined in :numref:`sec_basic_gan`"""
    batch_size = X.shape[0]
    fake_X = net_G(Z)  # computed outside the loss: no gradient to net_G
    def loss_D_fn(model_D):
        real_Y = model_D(X).squeeze()
        fake_Y = model_D(fake_X).squeeze()
        return (jnp.sum(optax.sigmoid_binary_cross_entropy(
                    real_Y, jnp.ones(batch_size))) +
                jnp.sum(optax.sigmoid_binary_cross_entropy(
                    fake_Y, jnp.zeros(batch_size)))) / 2
    loss_D, grads_D = nnx.value_and_grad(loss_D_fn)(net_D)
    optimizer_D.update(net_D, grads_D)
    return loss_D

@nnx.jit
def update_G(Z, net_D, net_G, optimizer_G):
    """Update the generator on the non-saturating loss; return its sum.

    Defined in :numref:`sec_basic_gan`"""
    # Both networks are arguments, so critics with mutable state (batch
    # norm) work too; argnums=0 differentiates the generator only
    def loss_G_fn(model_G, model_D):
        fake_Y = model_D(model_G(Z)).squeeze()
        return jnp.sum(optax.sigmoid_binary_cross_entropy(
            fake_Y, jnp.ones(Z.shape[0])))
    loss_G, grads_G = nnx.value_and_grad(loss_G_fn, argnums=0)(net_G, net_D)
    optimizer_G.update(net_G, grads_G)
    return loss_G

def train_gan(net_D, net_G, data_iter, num_epochs, latent_dim, lr_D, lr_G,
              step_G=update_G, seed=None):
    """Alternate update_D and step_G over the minibatches with Adam
    (beta_1 = 0.5); return the per-sample (loss_D, loss_G) of each epoch.

    Defined in :numref:`sec_basic_gan`"""
    key = d2l.get_key() if seed is None else jax.random.key(seed)
    optimizer_D = nnx.Optimizer(net_D, optax.adam(lr_D, b1=0.5),
                                wrt=nnx.Param)
    optimizer_G = nnx.Optimizer(net_G, optax.adam(lr_G, b1=0.5),
                                wrt=nnx.Param)
    history = []
    for epoch in range(num_epochs):
        loss_D, loss_G, n = 0.0, 0.0, 0  # summed on the device, read once
        for X, *_ in data_iter:  # labels, if any, are ignored
            key, subkey = jax.random.split(key)
            Z = jax.random.normal(subkey, (X.shape[0], latent_dim))
            loss_D += update_D(X, Z, net_D, net_G, optimizer_D)
            loss_G += step_G(Z, net_D, net_G, optimizer_G)
            n += X.shape[0]
        history.append((loss_D / n, loss_G / n))
    return np.array(history)

def mmd2_unbiased(K_pp, K_qq, K_pq):
    """Unbiased MMD^2 from the kernel matrices within p, within q, and
    across; the diagonals (a point with itself) are left out.

    Defined in :numref:`sec_gan_objectives`"""
    m, n = len(K_pp), len(K_qq)
    return ((K_pp.sum() - K_pp.diagonal().sum()) / (m * (m - 1))
            + (K_qq.sum() - K_qq.diagonal().sum()) / (n * (n - 1))
            - 2 * K_pq.mean())

def rpgan_loss_D(critic, real, fake):
    """Relativistic pairing critic loss -E[log sigma(D(x) - D(x'))].

    Defined in :numref:`sec_gan_convergence`"""
    return jax.nn.softplus(critic(fake) - critic(real)).mean()

def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator.

    Defined in :numref:`sec_gan_convergence`"""
    return jax.nn.softplus(critic(real) - critic(fake)).mean()

def gan_loss_D(critic, real, fake):
    """Logistic critic loss -E[log sigma(D(x))] - E[log sigma(-D(x'))]: the
    batch means, where update_D reports the halved sum.

    Defined in :numref:`sec_gan_convergence`"""
    return (jax.nn.softplus(-critic(real)).mean()
            + jax.nn.softplus(critic(fake)).mean())

def gan_loss_G(critic, real, fake):
    """Non-saturating generator loss -E[log sigma(D(x'))]; real is unused.

    Defined in :numref:`sec_gan_convergence`"""
    return jax.nn.softplus(-critic(fake)).mean()

def r1_r2_penalty(critic, real, fake):
    """Per-sample squared critic input gradients on real (R1) and fake (R2),
    before the gamma/2 scale.

    Defined in :numref:`sec_gan_convergence`"""
    def sq_grad_norm(x):
        # Rows are independent, so the gradient of the sum is per-sample
        grad = jax.grad(lambda x: critic(x).sum())(jax.lax.stop_gradient(x))
        return (grad.reshape(x.shape[0], -1) ** 2).sum(axis=1)
    return sq_grad_norm(real), sq_grad_norm(fake)

@nnx.jit(static_argnums=2)
def critic_step(net_D, opt_D, loss_D, real, fake, gamma, y=None):
    """One critic step on loss_D + gamma/2 (R1 + R2); return the loss and
    the batch means of R1 and R2.

    Defined in :numref:`sec_gan_convergence`"""
    def objective(net_D):
        critic = net_D if y is None else lambda x: net_D(x, y)
        r1, r2 = r1_r2_penalty(critic, real, fake)
        loss = loss_D(critic, real, fake)
        return (loss + gamma / 2 * (r1 + r2).mean(),
                (loss, r1.mean(), r2.mean()))
    (_, logs), grads = nnx.value_and_grad(objective, has_aux=True)(net_D)
    opt_D.update(net_D, grads)
    return logs

@nnx.jit(static_argnums=2)
def generator_step(net_G, opt_G, loss_G, net_D, real, z, y=None):
    """One generator step on loss_G against the critic; return the loss.

    Defined in :numref:`sec_gan_convergence`"""
    def objective(net_G):
        critic = net_D if y is None else lambda x: net_D(x, y)
        return loss_G(critic, real, net_G(z) if y is None else net_G(z, y))
    loss, grads = nnx.value_and_grad(objective)(net_G)
    opt_G.update(net_G, grads)
    return loss

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

# Fan-in initialization with the leaky-ReLU gain (normal, not truncated)
fan_in_init = nnx.initializers.variance_scaling(2 / (1 + 0.2 ** 2),
                                                'fan_in', 'normal')

class Generator(nnx.Module):
    """Learned 4x4 constant + projected latent (+ extra 4x4 input maps),
    bilinearly upsampled to resolution x resolution; no normalization.

    Defined in :numref:`sec_dcgan`"""
    def __init__(self, latent_dim, resolution, base_ch, *, extra_ch=0,
                 const_ch=128, rngs):
        self.latent_dim = latent_dim
        self.const = nnx.Param(
            jax.random.normal(rngs.params(), (1, 4, 4, const_ch)))
        self.z_proj = nnx.Linear(latent_dim, 16 * latent_dim,
                                 kernel_init=fan_in_init, rngs=rngs)
        self.mix = nnx.Conv(const_ch + latent_dim + extra_ch, base_ch,
                            (3, 3), padding='SAME', kernel_init=fan_in_init,
                            rngs=rngs)
        num_stages = int(np.log2(resolution / 4))       # 64: four doublings
        chans = [base_ch // 2 ** i for i in range(num_stages + 1)]
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=fan_in_init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nnx.Conv(chans[-1], 3, (3, 3), padding='SAME',
                               kernel_init=fan_in_init, rngs=rngs)

    def __call__(self, z, extra=None):
        zc = self.z_proj(z).reshape(-1, 4, 4, self.latent_dim)
        const = jnp.broadcast_to(self.const[...],
                                 (z.shape[0],) + self.const.shape[1:])
        maps = [const, zc] + ([] if extra is None else [extra])
        x = nnx.leaky_relu(self.mix(jnp.concatenate(maps, axis=-1)), 0.2)
        for conv in self.stages:                     # 4 -> 8 -> ... -> 64
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, 2 * h, 2 * w, c), method='bilinear')
            x = nnx.leaky_relu(conv(x), 0.2)
        return self.to_rgb(x)                        # no tanh: raw output

class Discriminator(nnx.Module):
    """Mirror of the generator: conv + leaky ReLU + bilinear downsampling to
    a 4x4 feature map phi(x), then a linear head on the flattened map.

    Defined in :numref:`sec_dcgan`"""
    def __init__(self, resolution, base_ch, *, rngs):
        num_stages = int(np.log2(resolution / 4))
        chans = [3] + [base_ch * 2 ** i for i in range(num_stages)]
        self.stages = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME',
                      kernel_init=fan_in_init, rngs=rngs)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nnx.Conv(chans[-1], 2 * chans[-1], (3, 3),
                            padding='SAME', kernel_init=fan_in_init,
                            rngs=rngs)
        self.head = nnx.Linear(2 * chans[-1] * 4 * 4, 1,
                               kernel_init=fan_in_init, rngs=rngs)

    def phi(self, x):
        for conv in self.stages:                     # 64 -> 32 -> ... -> 4
            x = nnx.leaky_relu(conv(x), 0.2)
            b, h, w, c = x.shape
            x = jax.image.resize(x, (b, h // 2, w // 2, c),
                                 method='bilinear')
        return nnx.leaky_relu(self.mix(x), 0.2)

    def __call__(self, x):
        return self.head(self.phi(x).reshape(x.shape[0], -1))

class EMA:
    """Exponential moving average of a model's parameters, given as a
    half-life in steps. Other state such as batch-norm statistics is copied;
    the average is kept in evaluation mode.

    Defined in :numref:`sec_dcgan`"""
    def __init__(self, model, half_life):
        self.model = nnx.clone(model)
        self.model.eval()
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        self._average(self.model, model, self.decay)

    @staticmethod
    @nnx.jit
    def _average(avg, model, decay):
        params = jax.tree.map(lambda a, p: decay * a + (1 - decay) * p,
                              nnx.state(avg, nnx.Param),
                              nnx.state(model, nnx.Param))
        nnx.update(avg, params, nnx.state(model, nnx.Not(nnx.Param)))

def sample_real(key, images, n, labels=None):
    """Draw a training batch with random horizontal flips (and its labels).

    Defined in :numref:`sec_dcgan`"""
    k1, k2 = jax.random.split(key)
    idx = jax.random.randint(k1, (n,), 0, len(images))
    batch = images[idx]
    flip = jax.random.bernoulli(k2, 0.5, (n,))
    batch = jnp.where(flip[:, None, None, None], batch[:, :, ::-1, :], batch)
    return batch if labels is None else (batch, labels[idx])

def ranking_accuracy(critic, real, fake):
    """Fraction of aligned (real, generated) pairs with D(real) > D(fake).

    Defined in :numref:`sec_dcgan`"""
    return float((critic(real) > critic(fake)).mean())

def fid_score(feat_p, feat_q):
    """Frechet distance between Gaussians fitted to two feature sets.

    Defined in :numref:`sec_dcgan`"""
    feat_p, feat_q = np.float64(feat_p), np.float64(feat_q)
    mu_p, mu_q = feat_p.mean(0), feat_q.mean(0)
    C_p = np.cov(feat_p, rowvar=False)
    C_q = np.cov(feat_q, rowvar=False)
    lam, U = np.linalg.eigh(C_p)
    sqrt_C_p = (U * np.sqrt(np.maximum(lam, 0))) @ U.T
    M = sqrt_C_p @ C_q @ sqrt_C_p
    cross = np.linalg.eigvalsh((M + M.T) / 2)   # symmetric up to rounding
    return (((mu_p - mu_q) ** 2).sum() + np.trace(C_p) + np.trace(C_q)
            - 2 * np.sqrt(np.maximum(cross, 0)).sum())

def kid_score(feat_p, feat_q):
    """Unbiased MMD^2 between two feature sets, cubic polynomial kernel.

    Defined in :numref:`sec_dcgan`"""
    d = feat_p.shape[1]
    kernel = lambda a, b: (a @ b.T / d + 1) ** 3
    return d2l.mmd2_unbiased(kernel(feat_p, feat_p), kernel(feat_q, feat_q),
                             kernel(feat_p, feat_q))

class GaussianMixture:
    """A mixture of isotropic Gaussians with closed-form density and score.

    Defined in :numref:`sec_diffusion-ebm-training`"""
    def __init__(self, means, weights, std):
        self.means = jnp.asarray(means, dtype=jnp.float32)
        self.weights = jnp.asarray(weights, dtype=jnp.float32)
        self.std = std

    def sample(self, key, n):
        k1, k2 = jax.random.split(key)
        k = jax.random.choice(k1, len(self.weights), (n,), p=self.weights)
        return self.means[k] + self.std * jax.random.normal(
            k2, (n, self.means.shape[1]))

    def log_prob(self, x, scale=1.0, sigma=0.0):
        """Log density of scale * x0 + sigma * eps, where x0 is a mixture draw."""
        var = (scale * self.std) ** 2 + sigma ** 2
        d2 = ((x[:, None, :] - scale * self.means[None]) ** 2).sum(-1)
        logs = (jnp.log(self.weights)[None] - d2 / (2 * var)
                - 0.5 * x.shape[-1] * math.log(2 * math.pi * var))
        return jax.nn.logsumexp(logs, 1)

    def score(self, x, scale=1.0, sigma=0.0):
        """The gradient of log_prob with respect to x, in closed form."""
        var = (scale * self.std) ** 2 + sigma ** 2
        d2 = ((x[:, None, :] - scale * self.means[None]) ** 2).sum(-1)
        r = jax.nn.softmax(jnp.log(self.weights)[None] - d2 / (2 * var), 1)
        return (r[:, :, None]
                * (scale * self.means[None] - x[:, None, :])).sum(1) / var

class ScoreNet(nnx.Module):
    """A two-dimensional score model: a multilayer perceptron with two outputs.

    Defined in :numref:`sec_diffusion-score-matching`"""
    def __init__(self, hidden=128, rngs=None):
        self.h1 = nnx.Linear(2, hidden, rngs=rngs)
        self.h2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.out = nnx.Linear(hidden, 2, rngs=rngs)

    def __call__(self, x):
        return self.out(nnx.silu(self.h2(nnx.silu(self.h1(x)))))

def sinusoidal_embedding(t, dim):
    """Map integer steps t (shape (n,)) to sinusoidal features (n, dim).

    Defined in :numref:`sec_diffusion-annealed`"""
    freqs = jnp.exp(-math.log(10000) * jnp.arange(dim // 2) / (dim // 2))
    args = t.astype(jnp.float32)[:, None] * freqs[None]
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], 1)

class ResBlock(nnx.Module):
    """Two 3x3 convolutions with group normalization and an added embedding.

    Defined in :numref:`sec_diffusion-annealed`"""
    def __init__(self, c_in, c_out, emb_dim, rngs):
        # epsilon 1e-5 and a two-pass variance in every group normalization
        self.norm1 = nnx.GroupNorm(c_in, num_groups=8, epsilon=1e-5,
                                   use_fast_variance=False, rngs=rngs)
        self.conv1 = nnx.Conv(c_in, c_out, (3, 3), padding='SAME', rngs=rngs)
        self.emb = nnx.Linear(emb_dim, c_out, rngs=rngs)
        self.norm2 = nnx.GroupNorm(c_out, num_groups=8, epsilon=1e-5,
                                   use_fast_variance=False, rngs=rngs)
        self.conv2 = nnx.Conv(c_out, c_out, (3, 3), padding='SAME', rngs=rngs)
        self.skip = (nnx.Conv(c_in, c_out, (1, 1), rngs=rngs) if c_in != c_out
                     else None)

    def __call__(self, x, emb):
        h = self.conv1(nnx.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, None, None, :]
        h = self.conv2(nnx.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)

class UNet(nnx.Module):
    """A small U-Net for 28x28 images (NHWC), conditioned on a step index t.

    Defined in :numref:`sec_diffusion-annealed`"""
    def __init__(self, ch=32, emb_dim=128, in_channels=1, out_channels=1,
                 rngs=None):
        self.emb_dim = emb_dim
        self.emb1 = nnx.Linear(emb_dim, emb_dim, rngs=rngs)
        self.emb2 = nnx.Linear(emb_dim, emb_dim, rngs=rngs)
        self.inc = nnx.Conv(in_channels, ch, (3, 3), padding='SAME', rngs=rngs)
        self.down1 = ResBlock(ch, ch, emb_dim, rngs)                   # 28 x 28
        # Strided convs pad 1 on every side: 'SAME' pads only right and
        # bottom when the stride is 2 and the sizes are even
        self.pool1 = nnx.Conv(ch, ch, (3, 3), strides=2, padding=1,
                              rngs=rngs)                                # -> 14 x 14
        self.down2 = ResBlock(ch, 2 * ch, emb_dim, rngs)
        self.pool2 = nnx.Conv(2 * ch, 2 * ch, (3, 3), strides=2,
                              padding=1, rngs=rngs)                     # -> 7 x 7
        self.mid = ResBlock(2 * ch, 2 * ch, emb_dim, rngs)
        self.up1 = ResBlock(4 * ch, ch, emb_dim, rngs)                 # 14 x 14
        self.up0 = ResBlock(2 * ch, ch, emb_dim, rngs)                 # 28 x 28
        self.norm = nnx.GroupNorm(ch, num_groups=8, epsilon=1e-5,
                                  use_fast_variance=False, rngs=rngs)
        self.outc = nnx.Conv(ch, out_channels, (3, 3), padding='SAME', rngs=rngs)

    def __call__(self, x, t, cond=None):
        emb = sinusoidal_embedding(t, self.emb_dim)
        if cond is not None:
            emb = emb + cond
        emb = self.emb2(nnx.silu(self.emb1(emb)))
        h1 = self.down1(self.inc(x), emb)
        h2 = self.down2(self.pool1(h1), emb)
        h = self.mid(self.pool2(h2), emb)
        h = jnp.repeat(jnp.repeat(h, 2, axis=1), 2, axis=2)
        h = self.up1(jnp.concatenate([h, h2], -1), emb)  # skip connection
        h = jnp.repeat(jnp.repeat(h, 2, axis=1), 2, axis=2)
        h = self.up0(jnp.concatenate([h, h1], -1), emb)  # skip connection
        return self.outc(nnx.silu(self.norm(h)))

class Encoder(nnx.Module):
    """The base encoder interface for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    # Later there can be additional arguments (e.g., length excluding padding)
    def __call__(self, X, *args):
        raise NotImplementedError

class Decoder(nnx.Module):
    """The base decoder interface for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError

class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_all_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]

    def predict_step(self, batch, num_steps, save_attention_weights=False):
        model = nnx.view(self, deterministic=True, use_running_average=True,
                         raise_if_not_found=False)
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = model.encoder(src, src_valid_len)
        enc_attention_weights = (getattr(model.encoder, 'attention_weights', [])
                                 if save_attention_weights else [])
        dec_state = model.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
        for _ in range(num_steps):
            Y, dec_state = model.decoder(outputs[-1], dec_state)
            outputs.append(d2l.argmax(Y, 2))
            if save_attention_weights:
                attention_weights.append(model.decoder.attention_weights)
        return d2l.concat(outputs[1:], 1), (attention_weights,
                                            enc_attention_weights)

class MTFraEng(d2l.DataModule):
    """The English-French dataset, tokenized with a shared byte-level BPE.

    Defined in :numref:`sec_machine_translation`"""
    def __init__(self, batch_size, num_steps=20, num_train=1024, num_val=128,
                 vocab_size=4000):
        super().__init__()
        self.save_hyperparameters()
        pairs = self._pairs(self._preprocess(self._download()))
        # Train ONE shared byte-level BPE over both languages.
        m = min(5000, len(pairs))
        self.tokenizer = d2l.BPETokenizer(
            vocab_size, pattern=d2l.BPETokenizer.GPT2_PATTERN)
        self.tokenizer.train('\n'.join([p[0] for p in pairs[:m]] +
                                       [p[1] for p in pairs[:m]]))
        # src_vocab / tgt_vocab both refer to the shared tokenizer.
        self.src_vocab = self.tgt_vocab = self.tokenizer
        pairs = pairs[:num_train + num_val]
        self.src_sents = [p[0] for p in pairs]
        self.tgt_sents = [p[1] for p in pairs]
        self.arrays = self._encode(pairs)

    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL + 'fra-eng.zip', self.root,
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text):
        # Normalize spaces, lowercase, and put a space before punctuation.
        text = text.replace(' ', ' ').replace('\xa0', ' ').lower()
        no_space = lambda c, prev: c in ',.!?' and prev != ' '
        return ''.join([' ' + c if i > 0 and no_space(c, text[i - 1]) else c
                        for i, c in enumerate(text)])

    def _pairs(self, text):
        return [ln.split('\t') for ln in text.split('\n') if ln.count('\t') == 1]

    def _encode(self, pairs):
        tok, t = self.tokenizer, self.num_steps
        def row(sent, is_tgt):
            ids = tok.encode(sent)
            ids = (ids[:t - 1] + [tok.eos] if len(ids) >= t else
                   ids + [tok.eos] + [tok.pad] * (t - 1 - len(ids)))
            return [tok.bos] + ids if is_tgt else ids
        src = d2l.tensor([row(s, False) for s, _ in pairs])
        tgt = d2l.tensor([row(t, True) for _, t in pairs])
        valid_len = d2l.reduce_sum(d2l.astype(src != tok.pad, d2l.int32), 1)
        return src, tgt[:, :-1], valid_len, tgt[:, 1:]

    def build(self, src_sentences, tgt_sentences):
        return self._encode([(self._preprocess(s), self._preprocess(t))
                             for s, t in zip(src_sentences, tgt_sentences)])

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)

def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs.

    Defined in :numref:`sec_machine_translation`"""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence-to-sequence learning.

    Defined in :numref:`sec_machine_translation`"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, rngs=None):
        rngs = nnx.Rngs(params=0, dropout=1, carry=2) if rngs is None else rngs
        self.embedding = nnx.Embed(vocab_size, embed_size, rngs=rngs)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout,
                           rngs=rngs)

    def __call__(self, X, *args):
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        outputs, state = self.rnn(embs)
        # outputs: (num_steps, batch_size, num_hiddens)
        # state: (num_layers, batch_size, num_hiddens)
        return outputs, state

class Seq2Seq(d2l.EncoderDecoder):
    """The RNN encoder-decoder for sequence-to-sequence learning.

    Defined in :numref:`sec_machine_translation`"""
    def __init__(self, encoder, decoder, tgt_pad, lr):
        super().__init__(encoder, decoder)
        self.tgt_pad, self.lr = tgt_pad, lr

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        return optax.adam(learning_rate=self.lr)

def chrf(pred, label, n=6, beta=2):
    """chrF (Popovic, 2015): character n-gram F-score.

    Defined in :numref:`sec_seq2seq_training`"""
    def ngrams(s, k):
        s = s.replace(' ', '')
        return collections.Counter(s[i:i + k] for i in range(len(s) - k + 1))
    prec = rec = 0.0
    for k in range(1, n + 1):
        p, r = ngrams(pred, k), ngrams(label, k)
        overlap = sum((p & r).values())
        if sum(p.values()) and sum(r.values()):
            prec += overlap / sum(p.values())
            rec += overlap / sum(r.values())
    prec, rec = prec / n, rec / n
    if prec + rec == 0:
        return 0.0
    return (1 + beta**2) * prec * rec / (beta**2 * prec + rec)

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.

    Defined in :numref:`sec_seq2seq_training`"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, min(k, len_pred) + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

def read_ptb():
    """Load the PTB dataset into a list of text lines.

    Defined in :numref:`sec_word2vec_data`"""
    data_dir = d2l.download_extract('ptb')
    # Read the training set
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

def subsample(sentences, vocab):
    """Subsample high-frequency words.

    Defined in :numref:`sec_word2vec_data`"""
    # Exclude unknown tokens ('<unk>')
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = collections.Counter([
        token for line in sentences for token in line])
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram.

    Defined in :numref:`sec_word2vec_data`"""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights.

    Defined in :numref:`sec_word2vec_data`"""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling.

    Defined in :numref:`sec_word2vec_data`"""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling.

    Defined in :numref:`sec_word2vec_data`"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))

def _pad_ptb(all_centers, all_contexts, all_negatives):
    """Pre-pad all skip-gram examples to the global max length.

    Returns four NumPy arrays: centers (N,), contexts_negatives (N, L),
    masks (N, L), labels (N, L), where L = max(len(c) + len(n)).

    Defined in :numref:`sec_word2vec_data`"""
    import numpy as _np
    n = len(all_centers)
    max_len = max(len(c) + len(neg)
                  for c, neg in zip(all_contexts, all_negatives))
    centers = _np.asarray(all_centers, dtype=_np.int64)
    contexts_negatives = _np.zeros((n, max_len), dtype=_np.int64)
    masks = _np.zeros((n, max_len), dtype=_np.float32)
    labels = _np.zeros((n, max_len), dtype=_np.float32)
    for i, (c, neg) in enumerate(zip(all_contexts, all_negatives)):
        cur_len = len(c) + len(neg)
        contexts_negatives[i, :cur_len] = c + neg
        masks[i, :cur_len] = 1.
        labels[i, :len(c)] = 1.
    return centers, contexts_negatives, masks, labels

def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory.

    Defined in :numref:`sec_word2vec_data`"""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    centers, cn, masks, labels = _pad_ptb(
        all_centers, all_contexts, all_negatives)
    centers = centers.reshape(-1, 1)
    n = len(centers)

    class PTBDataIter:
        def __len__(self):
            return math.ceil(n / batch_size)
        def __iter__(self):
            import numpy as _np
            idx = _np.random.permutation(n)
            for i in range(0, n, batch_size):
                b = idx[i:i + batch_size]
                yield (jnp.asarray(centers[b]), jnp.asarray(cn[b]),
                       jnp.asarray(masks[b]), jnp.asarray(labels[b]))

    return PTBDataIter(), vocab

d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')

class TokenEmbedding:
    """Token Embedding.

    Defined in :numref:`sec_synonyms`"""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)

def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs.

    Defined in :numref:`sec_bert`"""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments

class BERTEncoder(nnx.Module):
    """BERT encoder.

    Defined in :numref:`sec_bert`"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, rngs=None):
        rngs = nnx.Rngs(params=0, dropout=1) if rngs is None else rngs
        self.token_embedding = nnx.Embed(vocab_size, num_hiddens, rngs=rngs)
        self.segment_embedding = d2l.Embed(2, num_hiddens, rngs=rngs)
        self.blks = nnx.List([d2l.TransformerEncoderBlock(
            num_hiddens, ffn_num_hiddens, num_heads, dropout, True, rngs=rngs)
            for _ in range(num_blks)])
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (1, max_len, num_hiddens)) * 0.02)

    def __call__(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X, _ = blk(X, valid_lens)
        return X

class MaskLM(nnx.Module):
    """The masked language model task of BERT.

    Defined in :numref:`sec_bert`"""
    def __init__(self, vocab_size, num_hiddens, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.dense1 = nnx.Linear(num_hiddens, num_hiddens, rngs=rngs)
        self.layer_norm = nnx.LayerNorm(num_hiddens, rngs=rngs)
        self.dense2 = nnx.Linear(num_hiddens, vocab_size, rngs=rngs)

    def __call__(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = jnp.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `jnp.array([0, 0, 0, 1, 1, 1])`
        batch_idx = jnp.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.dense1(masked_X)
        mlm_Y_hat = nnx.relu(mlm_Y_hat)
        mlm_Y_hat = self.layer_norm(mlm_Y_hat)
        mlm_Y_hat = self.dense2(mlm_Y_hat)
        return mlm_Y_hat

class NextSentencePred(nnx.Module):
    """The next sentence prediction task of BERT.

    Defined in :numref:`sec_bert`"""
    def __init__(self, num_hiddens, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        self.output = nnx.Linear(num_hiddens, 2, rngs=rngs)

    def __call__(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)

class BERTModel(nnx.Module):
    """The BERT model.

    Defined in :numref:`sec_bert`"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, max_len=1000, rngs=None):
        rngs = nnx.Rngs(params=0, dropout=1) if rngs is None else rngs
        self.encoder = BERTEncoder(
            vocab_size, num_hiddens, ffn_num_hiddens, num_heads, num_blks,
            dropout, max_len=max_len, rngs=rngs)
        self.hidden = nnx.Linear(num_hiddens, num_hiddens, rngs=rngs)
        self.mlm = MaskLM(vocab_size, num_hiddens, rngs=rngs)
        self.nsp = NextSentencePred(num_hiddens, rngs=rngs)

    def __call__(self, tokens, segments, valid_lens=None, pred_positions=None,
                 training=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(
            jnp.tanh(self.hidden(encoded_X[:, 0, :])))
        return encoded_X, mlm_Y_hat, nsp_Y_hat

WIKITEXT_2_URL = ('https://huggingface.co/datasets/Salesforce/wikitext/'
                  'resolve/main/wikitext-2-v1/train-00000-of-00001.parquet')
WIKITEXT_2_SHA1 = '98ee727e59fcc34fddaadae93e15b1f8ed5561a4'

def _read_wiki(data_dir=None):
    import contextlib
    import io
    import pandas as pd
    with contextlib.redirect_stdout(io.StringIO()):
        fname = d2l.download(WIKITEXT_2_URL, folder='../data',
                              sha1_hash=WIKITEXT_2_SHA1)
    lines = pd.read_parquet(fname)['text'].tolist()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs

def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next

def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # For the input of a masked language model, make a new copy of tokens and
    # replace some of them by '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]

def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=np.int32))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype=np.int32))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype=np.float32))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=np.int32))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=np.float32))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=np.int32))
        nsp_labels.append(np.array(is_next, dtype=np.int32))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataset:
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)

def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset.

    Defined in :numref:`sec_bert-dataset`"""
    paragraphs = _read_wiki()
    train_set = _WikiTextDataset(paragraphs, max_len)
    # Create an index array and shuffle it
    indices = list(range(len(train_set)))
    random.shuffle(indices)

    # Return a callable so each call yields a fresh iterator (one-shot
    # generators can't be re-entered for multi-epoch training).
    def data_iter():
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            if len(batch_indices) < batch_size:
                continue
            batch = [train_set[idx] for idx in batch_indices]
            # Stack on the host, then copy each field to the device once
            yield tuple(jnp.asarray(np.stack([b[k] for b in batch]))
                        for k in range(7))
    return data_iter, train_set.vocab

def _get_batch_loss_bert(net, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = optax.softmax_cross_entropy_with_integer_labels(
        mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1))
    mlm_l = (mlm_l * mlm_weights_X.reshape(-1)).sum() / (
        mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = optax.softmax_cross_entropy_with_integer_labels(
        nsp_Y_hat, nsp_y).mean()
    l = mlm_l + nsp_l
    return l, (mlm_l, nsp_l)

d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels.

    Defined in :numref:`sec_sentiment`"""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset.

    Defined in :numref:`sec_sentiment`"""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.asarray([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens],
                                dtype=np.int32)
    test_features = np.asarray([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens],
                               dtype=np.int32)
    train_iter = d2l.load_array((train_features, np.asarray(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, np.asarray(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence.

    Defined in :numref:`sec_sentiment_rnn`"""
    sequence = jnp.array(vocab[sequence.split()])
    label = jnp.argmax(net(sequence.reshape(1, -1)), axis=1)
    return 'positive' if label == 1 else 'negative'

d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels

class SNLIDataset:
    """A customized dataset to load the SNLI dataset.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.asarray(dataset[2], dtype=np.int32)
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.asarray([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines], dtype=np.int32)

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)

def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = d2l.load_array(
        (train_set.premises, train_set.hypotheses, train_set.labels),
        batch_size, is_train=True)
    test_iter = d2l.load_array(
        (test_set.premises, test_set.hypotheses, test_set.labels),
        batch_size, is_train=False)
    return train_iter, test_iter, train_set.vocab

def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis.

    Defined in :numref:`sec_natural-language-inference-attention`"""
    premise = jnp.array(vocab[premise]).reshape((1, -1))
    hypothesis = jnp.array(vocab[hypothesis]).reshape((1, -1))
    label = jnp.argmax(nnx.view(net, deterministic=True)(premise, hypothesis),
                       axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'

@nnx.jit
def train_batch_ch13(net, optimizer, X, y):
    """Train for a minibatch with JAX (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    def compute_loss(model):
        logits = model(X)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, y).mean()
        return loss, logits
    (loss, logits), grads = nnx.value_and_grad(
        compute_loss, has_aux=True)(net)
    optimizer.update(net, grads)
    train_loss_sum = loss * X.shape[0]
    train_acc_sum = (logits.argmax(axis=-1) == y).sum()
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, optimizer, num_epochs):
    """Train a model with JAX (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    num_batches = int(train_iter.cardinality().numpy())
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()

    train_net = nnx.view(net, use_running_average=False,
                         raise_if_not_found=False)
    eval_net = nnx.view(net, use_running_average=True,
                        raise_if_not_found=False)

    @nnx.jit
    def eval_step(model, X):
        return model(X)

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        loss_sum = jnp.array(0.0)
        train_correct = jnp.array(0.0)
        num_examples = 0
        timer.start()
        for i, (features, labels) in enumerate(
                train_iter.as_numpy_iterator()):
            l, acc = train_batch_ch13(
                train_net, optimizer, jnp.array(features), jnp.array(labels))
            n = features.shape[0]
            loss_sum += l
            train_correct += acc
            num_examples += n
        # One transfer per epoch also waits for all dispatched training work,
        # keeping the throughput measurement meaningful without synchronizing
        # every minibatch.
        loss_sum, train_correct = jax.device_get((loss_sum, train_correct))
        timer.stop()
        # Evaluate on test set
        correct, total = jnp.array(0), 0
        for X, y in test_iter.as_numpy_iterator():
            logits = eval_step(eval_net, jnp.array(X))
            correct += (logits.argmax(axis=-1) == y).sum()
            total += y.shape[0]
        correct = int(jax.device_get(correct))
        train_loss = float(loss_sum) / num_examples
        train_acc = float(train_correct) / num_examples
        test_acc = correct / total
        animator.add(epoch + 1, (train_loss, train_acc, test_acc))
    print(f'loss {train_loss:.3f}, train acc '
          f'{train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{num_examples * num_epochs / timer.sum():.1f} examples/sec')
    return net

d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip', 
                         'fba480ffa8aa7e0febbb511d181409f899b9baa5')

def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height).

    Defined in :numref:`sec_bbox`"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right).

    Defined in :numref:`sec_bbox`"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format.

    Defined in :numref:`sec_bbox`"""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel.

    Defined in :numref:`sec_anchor`"""
    in_height, in_width = data.shape[-2:]
    num_sizes, num_ratios = len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = jnp.array(sizes)
    ratio_tensor = jnp.array(ratios)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (jnp.arange(in_height) + offset_h) * steps_h
    center_w = (jnp.arange(in_width) + offset_w) * steps_w
    shift_y, shift_x = jnp.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = jnp.concatenate((size_tensor * jnp.sqrt(ratio_tensor[0]),
                         sizes[0] * jnp.sqrt(ratio_tensor[1:])))\
                         * in_height / in_width  # Handle rectangular inputs
    h = jnp.concatenate((size_tensor / jnp.sqrt(ratio_tensor[0]),
                         sizes[0] / jnp.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = jnp.tile(jnp.stack((-w, -h, w, h)).T,
                                    (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = jnp.repeat(jnp.stack([shift_x, shift_y, shift_x, shift_y],
                          axis=1), boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return jnp.expand_dims(output, axis=0)

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes.

    Defined in :numref:`sec_anchor`"""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes.

    Defined in :numref:`sec_anchor`"""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = jnp.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = jnp.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = jnp.clip(inter_lowerrights - inter_upperlefts, 0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes.

    Defined in :numref:`sec_anchor`"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    jaccard = box_iou(anchors, ground_truth)
    anchors_bbox_map = jnp.full((num_anchors,), -1, dtype=jnp.int32)
    max_ious = jnp.max(jaccard, axis=1)
    indices = jnp.argmax(jaccard, axis=1)
    mask = max_ious >= iou_threshold
    anchors_bbox_map = jnp.where(mask, indices, anchors_bbox_map)
    col_discard = jnp.full((num_anchors,), -1.0)
    row_discard = jnp.full((num_gt_boxes,), -1.0)

    # Use lax.fori_loop so JIT does not unroll (and re-trace) per gt box
    def body(_, carry):
        jaccard, anchors_bbox_map = carry
        max_idx = jnp.argmax(jaccard)
        box_idx = max_idx % num_gt_boxes
        anc_idx = max_idx // num_gt_boxes
        anchors_bbox_map = anchors_bbox_map.at[anc_idx].set(box_idx)
        jaccard = jaccard.at[:, box_idx].set(col_discard)
        jaccard = jaccard.at[anc_idx, :].set(row_discard)
        return (jaccard, anchors_bbox_map)

    _, anchors_bbox_map = jax.lax.fori_loop(
        0, num_gt_boxes, body, (jaccard, anchors_bbox_map))
    return anchors_bbox_map

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets.

    Defined in :numref:`sec_anchor`"""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes.

    Defined in :numref:`sec_anchor`"""
    anchors = anchors.squeeze(axis=0)
    num_anchors = anchors.shape[0]

    def per_image(label):
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, None)
        bbox_mask = jnp.tile(
            jnp.expand_dims((anchors_bbox_map >= 0).astype(jnp.float32),
                            axis=-1), (1, 4))
        valid = anchors_bbox_map >= 0
        safe_idx = jnp.maximum(anchors_bbox_map, 0)
        class_labels = jnp.where(
            valid, label[safe_idx, 0].astype(jnp.int32) + 1,
            jnp.zeros(num_anchors, dtype=jnp.int32))
        assigned_bb = jnp.where(
            valid[:, None], label[safe_idx, 1:],
            jnp.zeros((num_anchors, 4), dtype=jnp.float32))
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        return offset.reshape(-1), bbox_mask.reshape(-1), class_labels

    # vmap over batch instead of Python for loop: one compiled kernel
    # is reused for every image instead of unrolling 32x
    bbox_offset, bbox_mask, class_labels = jax.vmap(per_image)(labels)
    return (bbox_offset, bbox_mask, class_labels)

def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets.

    Defined in :numref:`sec_anchor`"""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes.

    Defined in :numref:`sec_anchor`"""
    # Work in NumPy so the Python while-loop isn't forcing a host/device
    # sync every iteration.
    boxes_np = np.asarray(boxes)
    B = np.argsort(-np.asarray(scores))
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = int(B[0])
        keep.append(i)
        if B.size == 1: break
        rest = B[1:]
        # Pairwise IoU between box i and every remaining box, in NumPy
        box_i, rest_boxes = boxes_np[i], boxes_np[rest]
        lt = np.maximum(box_i[:2], rest_boxes[:, :2])
        rb = np.minimum(box_i[2:], rest_boxes[:, 2:])
        inter = np.clip(rb - lt, 0, None).prod(axis=1)
        area_i = (box_i[2:] - box_i[:2]).prod()
        area_rest = (rest_boxes[:, 2:] - rest_boxes[:, :2]).prod(axis=1)
        iou = inter / (area_i + area_rest - inter)
        B = rest[iou <= iou_threshold]
    return jnp.array(keep, dtype=jnp.int32)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression.

    Defined in :numref:`sec_anchor`"""
    batch_size = cls_probs.shape[0]
    anchors = anchors.squeeze(axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = jnp.max(cls_prob[1:], axis=0), jnp.argmax(
            cls_prob[1:], axis=0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = jnp.arange(num_anchors, dtype=jnp.int32)
        combined = jnp.concatenate((keep, all_idx))
        unique, counts = jnp.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = jnp.concatenate((keep, non_keep))
        class_id = class_id.at[non_keep].set(-1)
        class_id = class_id[all_id_sorted].astype(jnp.float32)
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id = jnp.where(below_min_idx, -1, class_id)
        conf = jnp.where(below_min_idx, 1 - conf, conf)
        pred_info = jnp.concatenate((jnp.expand_dims(class_id, axis=1),
                                     jnp.expand_dims(conf, axis=1),
                                     predicted_bb), axis=1)
        out.append(pred_info)
    return jnp.stack(out)

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels.

    Defined in :numref:`sec_object-detection-dataset`"""
    from PIL import Image
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        img = Image.open(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}'))
        images.append(jnp.array(img).transpose(2, 0, 1))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, jnp.expand_dims(jnp.array(targets), axis=1) / 256

class BananasDataset:
    """A customized dataset to load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].astype(jnp.float32), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """Load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    train_dataset = BananasDataset(is_train=True)
    val_dataset = BananasDataset(is_train=False)
    train_iter = d2l.ArrayDataLoader(
        jnp.stack(train_dataset.features), train_dataset.labels,
        batch_size=batch_size, shuffle=True)
    val_iter = d2l.ArrayDataLoader(
        jnp.stack(val_dataset.features), val_dataset.labels,
        batch_size=batch_size)
    return train_iter, val_iter

d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    from PIL import Image
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(np.array(Image.open(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg'))))
        labels.append(np.array(Image.open(os.path.join(
            voc_dir, 'SegmentationClass', f'{fname}.png')).convert('RGB')))
    return features, labels

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

def voc_colormap2label():
    """Build the mapping from RGB to class indices for VOC labels.

    Defined in :numref:`sec_semantic_segmentation`"""
    colormap2label = np.zeros(256 ** 3, dtype=np.int32)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices.

    Defined in :numref:`sec_semantic_segmentation`"""
    colormap = colormap.astype(np.int32)
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    # feature and label are HWC numpy arrays
    h, w = feature.shape[0], feature.shape[1]
    top = np.random.randint(0, h - height + 1)
    left = np.random.randint(0, w - width + 1)
    feature = feature[top:top+height, left:left+width, :]
    label = label[top:top+height, left:left+width, :]
    return feature, label

class VOCSegDataset:
    """A customized dataset to load the VOC dataset.

    Defined in :numref:`sec_semantic_segmentation`"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return (img.astype(np.float32) / 255 - self.rgb_mean) / self.rgb_std

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[0] >= self.crop_size[0] and
            img.shape[1] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (jnp.array(feature.transpose(2, 0, 1)),
                jnp.array(voc_label_indices(label, self.colormap2label)))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset.

    Defined in :numref:`sec_semantic_segmentation`"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    train_dataset = VOCSegDataset(True, crop_size, voc_dir)
    test_dataset = VOCSegDataset(False, crop_size, voc_dir)
    train_iter = d2l.ArrayDataLoader(train_dataset, batch_size=batch_size,
                                     shuffle=True, drop_last=True)
    test_iter = d2l.ArrayDataLoader(test_dataset, batch_size=batch_size,
                                    drop_last=True)
    return train_iter, test_iter

d2l.DATA_HUB['cifar10_tiny'] = (d2l.DATA_URL + 'kaggle_cifar10_tiny.zip',
                                '2068874e4b9a9f0fb07ebe0ad2b29754449ccacd')

def read_csv_labels(fname):
    """Read `fname` to return a filename to label dictionary.

    Defined in :numref:`sec_kaggle_cifar10`"""
    with open(fname, 'r') as f:
        # Skip the file header line (column name)
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))

def copyfile(filename, target_dir):
    """Copy a file into a target directory.

    Defined in :numref:`sec_kaggle_cifar10`"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def reorg_train_valid(data_dir, labels, valid_ratio):
    """Split the validation set out of the original training set.

    Defined in :numref:`sec_kaggle_cifar10`"""
    # The number of examples of the class that has the fewest examples in the
    # training dataset
    n = collections.Counter(labels.values()).most_common()[-1][1]
    # The number of examples per class for the validation set
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label

def reorg_test(data_dir):
    """Organize the testing set for data loading during prediction.

    Defined in :numref:`sec_kaggle_cifar10`"""
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))

d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d')

def rbfkernel(x1, x2, ls=4.):
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls**2 / 2) * (dist ** 2))

class HPOTrainer(d2l.Trainer):
    def validation_error(self):
        return 1 - self.validate()['acc']

class HPOSearcher(d2l.HyperParameters):
    def sample_configuration(self) -> dict:
        raise NotImplementedError

    def update(self, config: dict, error: float, additional_info=None):
        pass

class RandomSearcher(HPOSearcher):
    def __init__(self, config_space: dict, initial_config=None):
        self.save_hyperparameters()

    def sample_configuration(self) -> dict:
        if self.initial_config is not None:
            result = self.initial_config
            self.initial_config = None
        else:
            result = {
                name: domain.rvs()
                for name, domain in self.config_space.items()
            }
        return result

class HPOScheduler(d2l.HyperParameters):
    def suggest(self) -> dict:
        raise NotImplementedError
    
    def update(self, config: dict, error: float, info=None):
        raise NotImplementedError

class BasicScheduler(HPOScheduler):
    def __init__(self, searcher: HPOSearcher):
        self.save_hyperparameters()

    def suggest(self) -> dict:
        return self.searcher.sample_configuration()

    def update(self, config: dict, error: float, info=None):
        self.searcher.update(config, error, additional_info=info)

class HPOTuner(d2l.HyperParameters):
    def __init__(self, scheduler: HPOScheduler, objective: callable):
        self.save_hyperparameters()
        self.incumbent, self.incumbent_error = None, None  # Best so far
        self.incumbent_trajectory, self.records = [], []
        self.cumulative_runtime, self.current_runtime = [], 0

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i}: " + ", ".join(  # Floats to 4 significant digits
                f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}"
                for k, v in config.items()))
            error = float(self.objective(**config))
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error:.4f}, runtime = {runtime:.1f} s")

    def bookkeeping(self, config: dict, error: float, runtime: float):
        self.records.append({"config": config, "error": error, "runtime": runtime})
        # Check if the last hyperparameter configuration performs better 
        # than the incumbent
        if self.incumbent is None or self.incumbent_error > error:
            self.incumbent = config
            self.incumbent_error = error
        # Add current best observed performance to the optimization trajectory
        self.incumbent_trajectory.append(self.incumbent_error)
        # Update runtime
        self.current_runtime += runtime
        self.cumulative_runtime.append(self.current_runtime)

def hpo_objective_lenet(learning_rate, batch_size, max_epochs=10):
    model = d2l.LeNet(lr=learning_rate, num_classes=10)
    trainer = d2l.HPOTrainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return validation_error

class SuccessiveHalvingScheduler(d2l.HPOScheduler):
    def __init__(self, searcher, eta, r_min, r_max, prefact=1):
        self.save_hyperparameters()
        # Compute K, which is later used to determine the number of configurations
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        # Define the rungs
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        if r_max not in self.rung_levels:
            # The final rung should be r_max
            self.rung_levels.append(r_max)
            self.K += 1
        # Bookkeeping
        self.observed_error_at_rungs = defaultdict(list)
        self.all_observed_error_at_rungs = defaultdict(list)
        # Our processing queue
        self.queue = []

    def suggest(self):
        if len(self.queue) == 0:
            # Start a new round of successive halving
            # Number of configurations for the first rung:
            n0 = int(self.prefact * self.eta ** self.K)
            for _ in range(n0):
                config = self.searcher.sample_configuration()
                config["max_epochs"] = self.r_min  # Set r = r_min
                self.queue.append(config)
        # Return an element from the queue
        return self.queue.pop()

    def update(self, config: dict, error: float, info=None):
        ri = int(config["max_epochs"])  # Rung r_i
        # Update our searcher, e.g if we use Bayesian optimization later
        self.searcher.update(config, error, additional_info=info)
        self.all_observed_error_at_rungs[ri].append((config, error))
        if ri < self.r_max:
            # Bookkeeping
            self.observed_error_at_rungs[ri].append((config, error))
            # Determine how many configurations should be evaluated on this rung
            ki = self.K - self.rung_levels.index(ri)
            ni = int(self.prefact * self.eta ** ki)
            # If we observed all configuration on this rung r_i, we estimate the
            # top 1 / eta configuration, add them to queue and promote them for
            # the next rung r_{i+1}
            if len(self.observed_error_at_rungs[ri]) >= ni:
                kiplus1 = ki - 1
                niplus1 = int(self.prefact * self.eta ** kiplus1)
                best_performing_configurations = self.get_top_n_configurations(
                    rung_level=ri, n=niplus1
                )
                riplus1 = self.rung_levels[self.K - kiplus1]  # r_{i+1}
                # Queue may not be empty: insert new entries at the beginning
                self.queue = [
                    dict(config, max_epochs=riplus1)
                    for config in best_performing_configurations
                ] + self.queue
                self.observed_error_at_rungs[ri] = []  # Reset

    def get_top_n_configurations(self, rung_level, n):
        rung = self.observed_error_at_rungs[rung_level]
        if not rung:
            return []
        sorted_rung = sorted(rung, key=lambda x: x[1])
        return [x[0] for x in sorted_rung[:n]]

import os
from matplotlib.figure import Figure

def plot_curves(curves, xlabel, ylabel, smooth=1, reference=None,
                ylim=None, x=None, xscale='linear', yscale='linear',
                band='range'):
    """One panel per call. `curves` maps a label to an array of shape
    [seed, step] (or [step] for a single seed); plots the seed median and a
    shaded band over the seeds: min to max (`band='range'`) or the
    interquartile range (`band='iqr'`, the better summary from five seeds
    on). `smooth`, if greater than 1, replaces each point by the mean of the
    `smooth` steps that end there, before the seed statistics; the first
    `smooth - 1` steps lack a full window and are dropped. `x` gives the
    position of each step (default 0, 1, 2, ...). `reference`, if given,
    draws a dashed horizontal line; `ylim` fixes the y-axis range; `xscale`
    and `yscale` take matplotlib's scale names, e.g. 'log'.

    Defined in :numref:`sec_utils`"""
    set_figsize((6, 4))
    for name, values in curves.items():
        values = np.atleast_2d(values)
        xs = np.arange(values.shape[1]) if x is None else np.asarray(x)
        if smooth > 1:
            kernel = np.ones(smooth) / smooth
            values = np.stack([np.convolve(v, kernel, 'valid')
                               for v in values])
            xs = xs[smooth - 1:]      # each window sits at its last step
        line, = plt.plot(xs, np.median(values, axis=0), label=name)
        if values.shape[0] > 1:
            lo, hi = ((values.min(axis=0), values.max(axis=0))
                      if band == 'range' else
                      np.percentile(values, [25, 75], axis=0))
            plt.fill_between(xs, lo, hi, alpha=0.2, color=line.get_color())
    if reference is not None:
        plt.axhline(reference, linestyle='--', color='gray')
    if ylim is not None:
        plt.ylim(ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def plot_hist(samples, xlabel, ylabel, bins=60, vlines=()):
    """Overlaid histograms of `samples` (label -> values), dashed at `vlines`.

    Defined in :numref:`sec_utils`"""
    set_figsize((6, 4))
    for name, x in samples.items():
        plt.hist(x, bins=bins, alpha=0.5, label=name)
    for v in vlines:
        plt.axvline(v, linestyle='--', color='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

def show_grid(desc, values, policy=None, titles=None,
              diverging=False, fmt='{:.2f}'):
    """A gridworld in the style of the chapter's figures. Each cell is
    shaded by `values` (white to blue, or with `diverging=True` red to blue
    around zero, for differences) and shows its state index (top left), its
    value (bottom left) and an arrow for the action of `policy`, if given.
    Holes are grey and the goal green; S, H and G sit at the bottom right.
    The grid shape is read from `desc` (e.g. `env.unwrapped.desc`).
    `values` and `policy` may each be a single frame, shape (num_states,),
    or a sequence of frames, shape (num_frames, num_states); `titles`, if
    given, labels each frame.

    Defined in :numref:`sec_utils`"""
    h, w = desc.shape
    values = np.atleast_2d(values).reshape(-1, h * w)
    if policy is not None:
        policy = np.atleast_2d(policy).reshape(-1, h * w)
    if diverging:
        top = max(np.abs(values).max(), 1e-12)
        cmap, norm = plt.get_cmap('RdBu'), plt.Normalize(-top, top)
    else:
        cmap = plt.get_cmap('Blues')
        norm = plt.Normalize(min(0.0, values.min()),
                             max(values.max(), 1e-12))
    move = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}   # (dx, dy)
    fig, axes = plt.subplots(1, len(values), squeeze=False,
                             figsize=(3.6 * len(values), 3.6))
    for k, ax in enumerate(axes[0]):
        for s in range(h * w):
            y, x = divmod(s, w)
            cell, v = desc[y, x].decode(), values[k, s]
            fill = {'H': ('#565D66', 0.55), 'G': ('#2CA02C', 0.35)}.get(
                cell, (cmap(norm(v)), 1.0))
            ax.add_patch(plt.Rectangle((x, y), 1, 1, fc=fill[0],
                                       alpha=fill[1], ec='black', lw=1.0))
            shade = np.abs(norm(v) - 0.5) * 2 if diverging else norm(v)
            ink = 'white' if cell == 'H' or (cell != 'G' and shade > 0.6) \
                else 'black'
            ax.text(x + 0.07, y + 0.07, str(s), ha='left', va='top',
                    fontsize=9, color=ink)
            if cell not in 'HG' or np.abs(v) > 1e-9:   # terminal: 0
                ax.text(x + 0.07, y + 0.93, fmt.format(v), ha='left',
                        va='bottom', fontsize=10, color=ink)
            if cell in 'SHG':
                ax.text(x + 0.82, y + 0.80, cell, ha='center', va='center',
                        fontsize=13, fontweight='bold', color=ink)
            if policy is not None and cell not in 'HG':
                dx, dy = move[int(policy[k, s])]
                ax.annotate('', xy=(x + 0.5 + 0.22 * dx, y + 0.45 + 0.22 * dy),
                            xytext=(x + 0.5 - 0.22 * dx,
                                    y + 0.45 - 0.22 * dy),
                            arrowprops=dict(arrowstyle='-|>', color=ink,
                                            lw=1.6, mutation_scale=14))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect('equal')
        ax.axis('off')
        if titles is not None:
            ax.set_title(titles[k])
    fig.tight_layout()

def run_seeds(train, num_seeds, **kwargs):
    """Run train(seed, **kwargs), a generator of curve points, per seed.

    Defined in :numref:`sec_utils`"""
    return np.array([list(train(seed, **kwargs)) for seed in range(num_seeds)])

def on_grid(runs, grid, col, smooth=1):
    """Resample per-episode rows (env step, value, ...) onto an env-step
    grid. For each run, drop the rows whose column `col` is NaN, average
    the column over the last `smooth` episodes, and interpolate linearly at
    the grid points; returns [run, grid point], the input of plot_curves.

    Defined in :numref:`sec_utils`"""
    out = []
    for r in runs:
        r = np.asarray(r, dtype=float)
        r = r[~np.isnan(r[:, col])]
        v = np.convolve(r[:, col], np.ones(smooth) / smooth, 'valid')
        out.append(np.interp(grid, r[smooth - 1:, 0], v))
    return np.stack(out)

_forwards = {}                          # one compiled forward per net structure

import weakref
_splits = weakref.WeakKeyDictionary()  # per net: its forward and variables

def forward_np(net, x):
    """net(x) for numpy x, as numpy, through one compiled forward per net
    structure and input shape. The net is split once and its variables are
    kept, so each call reads their current values: optimizer steps show up
    without splitting again.

    Defined in :numref:`sec_utils`"""
    if net not in _splits:
        graph, state = nnx.split(net)
        tree = jax.tree.structure(state)
        if (graph, tree) not in _forwards:
            _forwards[graph, tree] = jax.jit(lambda leaves, x: nnx.merge(
                graph, jax.tree.unflatten(tree, leaves))(x))
        _splits[net] = _forwards[graph, tree], jax.tree.leaves(
            state, is_leaf=lambda v: isinstance(v, nnx.Variable))
    forward, variables = _splits[net]
    leaves = [v.get_raw_value() for v in variables]
    return jax.tree.map(np.asarray, forward(leaves, x))

def _pad(x, size):
    """Zero-pad axis 0 of an array to `size` rows.

    Defined in :numref:`sec_utils`"""
    x = np.asarray(x)
    return np.pad(x, ((0, size - len(x)),) + ((0, 0),) * (x.ndim - 1))

def _bucket(n):
    """Padded batch length: the next power of two, at least 64, so that a
    compiled pass is reused across batches of different lengths.

    Defined in :numref:`sec_utils`"""
    return 1 << max(6, (n - 1).bit_length())

_adams = {}

def shared_adam(lr):
    """One optax.adam per learning rate, shared by all agents: compiled
    updates are keyed on the optimizer, so a new agent reuses the compiled
    code instead of compiling it again.

    Defined in :numref:`sec_utils`"""
    return _adams.setdefault(lr, optax.adam(lr))

_sgds = {}

def plain_sgd(net, lr):
    """A plain SGD optimizer over `net`, one optax.sgd per learning rate.

    Defined in :numref:`sec_utils`"""
    return nnx.Optimizer(net, _sgds.setdefault(lr, optax.sgd(lr)),
                         wrt=nnx.Param)

def load_cifar10_arrays():
    """CIFAR-10 on the default device, pixel values scaled to [-1, 1].
    Returns train_X, train_y, test_X, test_y.

    Defined in :numref:`sec_utils`"""
    import pickle, warnings  # not in the d2l library preamble
    d2l.DATA_HUB['cifar10'] = (
        'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        '874905e36347c8536514d0a26261acf3bff89bc7')
    cifar_dir = d2l.download_extract('cifar10', 'cifar-10-batches-py')
    def load_batch(name):
        with open(f'{cifar_dir}/{name}', 'rb') as f, warnings.catch_warnings():
            warnings.simplefilter('ignore')   # legacy NumPy pickle format
            batch = pickle.load(f, encoding='bytes')
        X = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return (jnp.asarray(X, dtype=jnp.float32) / 127.5 - 1,
                jnp.asarray(batch[b'labels']))
    Xs, ys = zip(*[load_batch(f'data_batch_{i}') for i in range(1, 6)])
    test_X, test_y = load_batch('test_batch')
    return jnp.concatenate(Xs), jnp.concatenate(ys), test_X, test_y

class FeatureCNN(nnx.Module):
    """Three conv blocks, global average pooling, and a linear head. Inputs
    are first resized bilinearly to resolution x resolution.

    Defined in :numref:`sec_utils`"""
    def __init__(self, feature_dim=128, resolution=32, rngs=None):
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.resolution = resolution
        widths = [3, 32, 64, feature_dim]
        self.convs = nnx.List(
            [nnx.Conv(c_in, c_out, (3, 3), padding='SAME', rngs=rngs)
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        # Running averages decay 0.9; the variance is computed in two passes
        self.norms = nnx.List([nnx.BatchNorm(c, momentum=0.9,
                                             use_fast_variance=False,
                                             rngs=rngs)
                               for c in widths[1:]])
        self.head = nnx.Linear(feature_dim, 10, rngs=rngs)

    def features(self, x):
        if x.shape[1] != self.resolution:
            x = jax.image.resize(
                x, (x.shape[0], self.resolution, self.resolution, x.shape[3]),
                method='bilinear')
        for conv, norm in zip(self.convs, self.norms):
            x = nnx.max_pool(nnx.relu(norm(conv(x))),
                             window_shape=(2, 2), strides=(2, 2))
        return x.mean(axis=(1, 2))

    def __call__(self, x):
        return self.head(self.features(x))

def train_feature_cnn(cnn, X, y, num_epochs=15, max_lr=0.003,
                      batch_size=256, shift=0, seed=0):
    """Train on random horizontal flips, plus random shifts of up to `shift`
    pixels, with a one-cycle learning rate; end in evaluation mode.

    Defined in :numref:`sec_utils`"""
    schedule = optax.cosine_onecycle_schedule(
        num_epochs * (len(X) // batch_size), max_lr)
    opt = nnx.Optimizer(cnn, optax.adam(schedule), wrt=nnx.Param)
    h, w = X.shape[1:3]

    @nnx.jit
    def step(cnn, opt, Xb, yb, key):
        k1, k2 = jax.random.split(key)
        flip = jax.random.bernoulli(k1, 0.5, (len(Xb), 1, 1, 1))
        Xb = jnp.where(flip, Xb[:, :, ::-1], Xb)
        a, b = jax.random.randint(k2, (2,), 0, 2 * shift + 1)
        Xb = jnp.pad(Xb, ((0, 0), (shift, shift), (shift, shift), (0, 0)),
                     mode='reflect')
        Xb = jax.lax.dynamic_slice(Xb, (0, a, b, 0), (len(Xb), h, w, 3))
        def loss_fn(cnn):
            return optax.softmax_cross_entropy_with_integer_labels(
                cnn(Xb), yb).mean()
        opt.update(cnn, nnx.grad(loss_fn)(cnn))

    cnn.train()
    key = jax.random.key(seed)
    for epoch in range(num_epochs):
        key, kp = jax.random.split(key)
        order = jax.random.permutation(kp, len(X))
        for i in range(0, len(order) - batch_size + 1, batch_size):
            key, ka = jax.random.split(key)
            idx = order[i:i + batch_size]
            step(cnn, opt, X[idx], y[idx], ka)
    cnn.eval()

def linreg(X, w, b):
    """The linear regression model.

    Defined in :numref:`sec_utils`"""
    return d2l.matmul(X, w) + b

def squared_loss(y_hat, y):
    """Squared loss.

    Defined in :numref:`sec_utils`"""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2

class ArrayDataLoader(d2l.ArrayLoader):
    """`d2l.ArrayLoader` that yields JAX arrays and also accepts a dataset.

    Pass the arrays as positional arguments, or one dataset object with
    `__len__` and `__getitem__` whose items are tuples of fields. Like
    `d2l.ArrayLoader`, it keeps the last, partial minibatch unless
    `drop_last=True`.

    Defined in :numref:`sec_utils`"""
    def __init__(self, *args, batch_size=32, shuffle=False, drop_last=False):
        if len(args) == 1 and not hasattr(args[0], 'shape'):  # A dataset
            items = [args[0][i] for i in range(len(args[0]))]
            if isinstance(items[0], (tuple, list)):
                args = tuple(np.array([item[j] for item in items])
                             for j in range(len(items[0])))
            else:
                args = (np.array(items),)
        super().__init__(args, batch_size, shuffle=shuffle,
                         drop_last=drop_last)

    def __iter__(self):
        for batch in super().__iter__():
            yield tuple(jnp.asarray(a) for a in batch)

def load_array(data_arrays, batch_size, is_train=True):
    """Minibatches of JAX arrays, reshuffled each pass if is_train.

    Defined in :numref:`sec_utils`"""
    return d2l.ArrayDataLoader(*data_arrays, batch_size=batch_size,
                               shuffle=is_train)

class Animator:
    """For plotting data in animation.

    Defined in :numref:`sec_utils`"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

class Accumulator:
    """For accumulating sums over `n` variables.

    Defined in :numref:`sec_utils`"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.

    Defined in :numref:`sec_utils`"""
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_utils`"""
    if len(line) > num_steps:
        return line[:num_steps]
    return line + [padding_token] * (num_steps - len(line))

def download_extract(name, folder=None):
    """Download and extract a zip/tar file.

    Defined in :numref:`sec_utils`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    target = os.path.join(base_dir, folder) if folder else data_dir
    marker = fname + '.extracted'
    if os.path.exists(marker):
        return target
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted.'
    fp.extractall(base_dir)
    open(marker, 'w').close()
    return target

def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_utils`"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5,
                cmap=None):
    """Plot a list of images (cmap colors single-channel images).

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = d2l.numpy(img)
        except:
            pass
        ax.imshow(img, cmap=cmap)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def show_heatmaps(matrices, xlabel, ylabel, titles=None,
                  figsize=(2.5, 2.5), cmap='Reds', cbar=True,
                  xticklabels=None, yticklabels=None):
    """Show heatmaps of matrices; cbar is True (shared), 'each' or False.

    Defined in :numref:`sec_utils`"""
    d2l.use_svg_display()
    matrices = d2l.numpy(matrices)
    num_rows, num_cols, height, width = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    # All panels share one color scale (one norm) unless each gets its own
    lims = {} if cbar == 'each' else dict(norm=d2l.plt.Normalize(
        matrices.min(), matrices.max()))
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix, cmap=cmap, **lims)
            if cbar == 'each':
                fig.colorbar(pcm, ax=ax, shrink=0.6)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    # Shared axes: setting the ticks of one panel sets them for all
    for axis, n, labels in ((ax.xaxis, width, xticklabels),
                            (ax.yaxis, height, yticklabels)):
        if labels is None:
            axis.set_major_locator(d2l.plt.MaxNLocator(
                'auto', integer=True, steps=[1, 2, 2.5, 5, 10]))
        else:
            axis.set_ticks(range(n), labels)
    if cbar is True:
        fig.colorbar(pcm, ax=axes, shrink=0.6);

def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    if not url.startswith('http'):
        # For back compatibility
        url, sha1_hash = DATA_HUB[url]
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    import tempfile
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    fd, tmp = tempfile.mkstemp(dir=folder)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(r.content)
        os.replace(tmp, fname)
    except BaseException:
        os.unlink(tmp)
        raise
    return fname

def extract(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    import tempfile, shutil
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), 'Only support zip/tar files.'
    opener = zipfile.ZipFile if ext == '.zip' else tarfile.open
    if folder is None:
        folder = base_dir
    tmp = tempfile.mkdtemp(dir=folder)
    try:
        with opener(filename, 'r') as fp:
            fp.extractall(tmp)
        for name in os.listdir(tmp):
            src = os.path.join(tmp, name)
            dst = os.path.join(folder, name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            os.rename(src, dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

class PositionWiseFFN(nnx.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs,
                 ffn_num_inputs=None, rngs=None):
        rngs = nnx.Rngs(0) if rngs is None else rngs
        ffn_num_inputs = (ffn_num_hiddens if ffn_num_inputs is None
                          else ffn_num_inputs)
        self.dense1 = nnx.Linear(ffn_num_inputs, ffn_num_hiddens, rngs=rngs)
        self.dense2 = nnx.Linear(ffn_num_hiddens, ffn_num_outputs, rngs=rngs)

    def __call__(self, X):
        return self.dense2(nnx.relu(self.dense1(X)))

class AddNorm(nnx.Module):
    """The residual connection followed by layer normalization."""
    def __init__(self, num_hiddens, dropout, rngs=None):
        rngs = nnx.Rngs(params=0, dropout=1) if rngs is None else rngs
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.ln = nnx.LayerNorm(num_hiddens, rngs=rngs)

    def __call__(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nnx.Module):
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False, rngs=None):
        rngs = nnx.Rngs(params=0, dropout=1) if rngs is None else rngs
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias, rngs=rngs)
        self.addnorm1 = AddNorm(num_hiddens, dropout, rngs=rngs)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens,
                                   num_hiddens, rngs=rngs)
        self.addnorm2 = AddNorm(num_hiddens, dropout, rngs=rngs)

    def __call__(self, X, valid_lens):
        mask = (None if valid_lens is None
                else d2l.padding_mask(valid_lens, X.shape[1]))
        Y = self.addnorm1(X, self.attention(X, X, X, mask))
        # Attention weights are recorded on eager (un-jitted) calls only
        return self.addnorm2(Y, self.ffn(Y)), self.attention.attention_weights


ones_like = jnp.ones_like
ones = jnp.ones
zeros_like = jnp.zeros_like
zeros = jnp.zeros
arange = jnp.arange
meshgrid = jnp.meshgrid
sin = jnp.sin
sinh = jnp.sinh
cos = jnp.cos
cosh = jnp.cosh
tanh = jnp.tanh
linspace = jnp.linspace
exp = jnp.exp
log = jnp.log
tensor = jnp.array
expand_dims = jnp.expand_dims
matmul = jnp.matmul
int32 = jnp.int32
int64 = jnp.int64
float32 = jnp.float32
concat = jnp.concatenate
stack = jnp.stack
abs = jnp.abs
eye = jnp.eye
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.astype(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)
nn_Module = nnx.Module
to = jax.device_put
numpy = np.asarray
transpose = lambda a: a.T
sigmoid = jax.nn.sigmoid
size = lambda x, *args, **kwargs: x.size
