DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import jax
# Force JAX to initialize CUDA (and load its own cuBLAS/cuSOLVER) before any
# subsequent `import tensorflow` pulls in TF's bundled CUDA libraries. Without
# this, TF's older cuBLAS is loaded first and jax.xla_bridge falls back to CPU.
jax.devices()
import tensorflow as _tf
# TF is only used for data loading in JAX notebooks; hide GPUs from it so it
# doesn't fight JAX for the pre-allocated GPU memory (avoids transient
# "Dst tensor is not initialized" OOMs seen when multiple notebooks share a
# device).
try:
    _tf.config.set_visible_devices([], 'GPU')
except RuntimeError:
    for _tf_gpu in _tf.config.list_physical_devices('GPU'):
        _tf.config.experimental.set_memory_growth(_tf_gpu, True)
del _tf
import flax
from jax import numpy as jnp
from flax import linen as nn
import random as _random

# Seed the d2l PRNG once per interpreter. Every d2l.get_key() call splits
# and advances _master_key, so the whole notebook draws from a single
# continuous stream of uncorrelated sub-keys. Do not build a fresh
# jax.random.PRNGKey(seed) at each call site — independently seeded keys
# can produce correlated samples in the JAX counter-based PRNG.
_master_key = jax.random.PRNGKey(0)
_seed_rng = _random.Random(0)


def get_key():
    """Return a fresh sub-key split from the module-level master key."""
    global _master_key
    _master_key, sub = jax.random.split(_master_key)
    return sub


def get_seed():
    """Return a fresh deterministic int seed, for non-JAX APIs that want
    an int (numpy.random.seed, tf.random.set_seed, stdlib random)."""
    return _seed_rng.randint(0, 10**6)

nn_Module = nn.Module


#################   WARNING   ################
# The below part is generated automatically through:
#    python tools/build_lib.py
# Don't edit it directly

import sys
d2l = sys.modules[__name__]

import inspect
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
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
import hashlib
d2l = sys.modules[__name__]

from dataclasses import field
from functools import partial
import flax
from flax import linen as nn
from flax.training import train_state
import jax
from jax import numpy as jnp
from jax import grad, vmap
import numpy as np
import optax
import tensorflow as tf
import tensorflow_datasets as tfds
from types import FunctionType
from typing import Any

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points.

    Defined in :numref:`sec_calculus`"""

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
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
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

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.

        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class ProgressBoard(d2l.HyperParameters):
    """The board that plots data points in animation.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, xlabel=None, ylabel=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=['-', '--', '-.', ':'], colors=['C0', 'C1', 'C2', 'C3'],
                 fig=None, axes=None, figsize=(3.5, 2.5), display=True):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        Point = collections.namedtuple('Point', ['x', 'y'])
        if not hasattr(self, 'raw_points'):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []    
        points = self.raw_points[label]
        line = self.data[label]
        points.append(Point(x, y))
        if len(points) != every_n:
            return    
        mean = lambda x: sum(x) / len(x)
        line.append(Point(mean([p.x for p in points]), 
                          mean([p.y for p in points])))
        points.clear()
        if not self.display: 
            return
        d2l.use_svg_display()
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(self.data.items(), self.ls, self.colors):        
            plt_lines.append(d2l.plt.plot([p.x for p in v], [p.y for p in v], 
                                          linestyle=ls, color=color)[0])
            labels.append(k)        
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        if not self.xlabel: self.xlabel = self.x    
        axes.set_xlabel(self.xlabel)
        axes.set_ylabel(self.ylabel)
        axes.set_xscale(self.xscale)
        axes.set_yscale(self.yscale)
        axes.legend(plt_lines, labels)    
        display.display(self.fig)
        display.clear_output(wait=True)

class Module(d2l.nn_Module, d2l.HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
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
        # Skip device sync unless this point will be plotted; every_n
        # buckets ensure alignment with the board's own filter
        if train and int(n) > 1 and self.trainer.train_batch_idx % int(n):
            return
        self.board.draw(x, d2l.to(value, d2l.cpu()),
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
        

    def configure_optimizers(self):
        return optax.sgd(self.lr)

    def apply_init(self, dummy_input, key):
        params = self.init(key, *dummy_input)  # dummy_input tuple unpacked
        return params

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
        # Use Tensorflow Datasets & Dataloader. JAX or Flax do not provide
        # any dataloading functionality. `drop_remainder=train` keeps every
        # *training* minibatch the same shape, so a `@jax.jit`'d step
        # function compiles once per epoch instead of recompiling for the
        # smaller last batch (a common source of multi-minute slowdowns on
        # NLP datasets where the last batch is a different shape every time).
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(tensors).shuffle(
                buffer_size=shuffle_buffer
            ).batch(self.batch_size, drop_remainder=train))

class Trainer(d2l.HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`sec_oo-design`"""

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

    def fit_epoch(self):
        self.model.training = True
        # Some hand-rolled optimizers (e.g. LinearRegressionScratch.SGD) are not
        # JIT-traceable: detect that by probing opt_state and skip the JIT path.
        jit_ok = isinstance(self.state.opt_state, tuple)
        if self.state.batch_stats:
            # Mutable states will be used later (e.g., for batch norm)
            for batch in self.train_dataloader:
                (_, mutated_vars), grads = self.model.training_step(
                    self.state.params, self.prepare_batch(batch), self.state)
                if jit_ok:
                    self.state = _trainer_update_with_bn(
                        self.state, grads, mutated_vars['batch_stats'])
                else:
                    self.state = self.state.apply_gradients(grads=grads).replace(
                        dropout_rng=jax.random.split(self.state.dropout_rng)[0],
                        batch_stats=mutated_vars['batch_stats'])
                self.train_batch_idx += 1
        else:
            for batch in self.train_dataloader:
                _, grads = self.model.training_step(
                    self.state.params, self.prepare_batch(batch), self.state)
                if jit_ok:
                    self.state = _trainer_update(self.state, grads)
                else:
                    self.state = self.state.apply_gradients(grads=grads).replace(
                        dropout_rng=jax.random.split(self.state.dropout_rng)[0])
                self.train_batch_idx += 1

        if self.val_dataloader is None:
            return
        self.model.training = False
        for batch in self.val_dataloader:
            self.model.validation_step(self.state.params,
                                       self.prepare_batch(batch),
                                       self.state)
            self.val_batch_idx += 1

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

    def prepare_batch(self, batch):
        if self.gpus:
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return batch

    def clip_gradients(self, grad_clip_val, grads):
        grad_leaves, _ = jax.tree_util.tree_flatten(grads)
        norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
        clip = lambda grad: jnp.where(norm < grad_clip_val,
                                      grad, grad * (grad_clip_val / norm))
        return jax.tree_util.tree_map(clip, grads)

class SyntheticRegressionData(d2l.DataModule):
    """Synthetic data for linear regression.

    Defined in :numref:`sec_synthetic-regression-data`"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32, key=jax.random.PRNGKey(0)):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        key1, key2 = jax.random.split(key)
        self.X = jax.random.normal(key1, (n, w.shape[0]))
        noise = jax.random.normal(key2, (n, 1)) * noise
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

class LinearRegressionScratch(d2l.Module):
    """The linear regression model implemented from scratch.

    Defined in :numref:`sec_linear_scratch`"""
    num_inputs: int
    lr: float
    sigma: float = 0.01

    def setup(self):
        self.w = self.param('w', nn.initializers.normal(self.sigma),
                            (self.num_inputs, 1))
        self.b = self.param('b', nn.initializers.zeros, (1))

    def forward(self, X):
        return d2l.matmul(X, self.w) + self.b

    def loss(self, params, X, y, state):
        y_hat = state.apply_fn({'params': params}, *X)  # X unpacked from a tuple
        l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
        return d2l.reduce_mean(l)

    def configure_optimizers(self):
        return SGD(self.lr)

class SGD(d2l.HyperParameters):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_linear_scratch`"""
    # The key transformation of Optax is the GradientTransformation
    # defined by two methods, the init and the update.
    # The init initializes the state and the update transforms the gradients.
    # https://github.com/deepmind/optax/blob/master/optax/_src/transform.py
    def __init__(self, lr):
        self.save_hyperparameters()

    def init(self, params):
        # Delete unused params
        del params
        return optax.EmptyState

    def update(self, updates, state, params=None):
        del params
        # When state.apply_gradients method is called to update flax's
        # train_state object, it internally calls optax.apply_updates method
        # adding the params to the update equation defined below.
        updates = jax.tree_util.tree_map(lambda g: -self.lr * g, updates)
        return updates, state

    def __call__(self):
        return optax.GradientTransformation(self.init, self.update)

@jax.jit
def _trainer_update(state, grads):
    return state.apply_gradients(grads=grads).replace(
        dropout_rng=jax.random.split(state.dropout_rng)[0])

@jax.jit
def _trainer_update_with_bn(state, grads, batch_stats):
    return state.apply_gradients(grads=grads).replace(
        dropout_rng=jax.random.split(state.dropout_rng)[0],
        batch_stats=batch_stats)

class LinearRegression(d2l.Module):
    """The linear regression model implemented with high-level APIs.

    Defined in :numref:`sec_linear_concise`"""
    lr: float

    def setup(self):
        self.net = nn.Dense(1, kernel_init=nn.initializers.normal(0.01))

    def forward(self, X):
        return self.net(X)

    def loss(self, params, X, y, state):
        y_hat = state.apply_fn({'params': params}, *X)
        return d2l.reduce_mean(jnp.square(y_hat - y))

    def configure_optimizers(self):
        return optax.sgd(self.lr)

    def get_w_b(self, state):
        net = state.params['net']
        return net['kernel'], net['bias']

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
        data = self.train if train else self.val
        process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                                tf.cast(y, dtype='int32'))
        resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
        shuffle_buf = len(data[0]) if train else 1
        # `drop_remainder=train` for the same reason as the TF tab — JAX
        # also retraces a `@jax.jit`'d step function per unique input shape.
        return tfds.as_numpy(
            tf.data.Dataset.from_tensor_slices(process(*data)).shuffle(
                shuffle_buf).batch(self.batch_size,
                                   drop_remainder=train).map(resize_fn))

    def visualize(self, batch, nrows=1, ncols=8, labels=None):
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

class Classifier(d2l.Module):
    """The base class of classification models.

    Defined in :numref:`sec_classification`"""
    def training_step(self, params, batch, state):
        # Here value is a tuple since models with BatchNorm layers require
        # the loss to return auxiliary data
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot("loss", l, train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        # Discard the second returned value. It is used for training models
        # with BatchNorm layers since loss also returns auxiliary data
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('loss', l, train=False)
        self.plot('acc', self.accuracy(params, batch[:-1], batch[-1], state),
                  train=False)

    @partial(jax.jit, static_argnums=(0, 5))
    def accuracy(self, params, X, Y, state, averaged=True):
        """Compute the number of correct predictions.

        Defined in :numref:`sec_classification`"""
        Y_hat = state.apply_fn({'params': params,
                                'batch_stats': state.batch_stats},  # BatchNorm Only
                               *X)
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
        compare = d2l.astype(preds == d2l.reshape(Y, (-1,)), d2l.float32)
        return d2l.reduce_mean(compare) if averaged else compare

    def layer_summary(self, X_shape, key=d2l.get_key()):
        X = jnp.zeros(X_shape)
        params = self.init(key, X)
        bound_model = self.clone().bind(params, mutable=['batch_stats'])
        _ = bound_model(X)
        for layer in bound_model.net.layers:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

    @partial(jax.jit, static_argnums=(0, 5))
    def loss(self, params, X, Y, state, averaged=True):
        Y_hat, updates = state.apply_fn({'params': params,
                                         'batch_stats': state.batch_stats},
                                        *X, mutable=['batch_stats'],
                                        rngs={'dropout': state.dropout_rng})
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = d2l.reshape(Y, (-1,))
        fn = optax.softmax_cross_entropy_with_integer_labels
        return (fn(Y_hat, Y).mean(), updates) if averaged else (fn(Y_hat, Y), updates)

def cross_entropy(y_hat, y):
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = jnp.clip(y_hat[list(range(len(y_hat))), y], min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

class SoftmaxRegression(d2l.Classifier):
    num_outputs: int
    lr: float

    @nn.compact
    def __call__(self, X):
        X = X.reshape((X.shape[0], -1))  # Flatten
        X = nn.Dense(self.num_outputs)(X)
        return X

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
    return [gpu(i) for i in range(num_gpus())]

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
    """The LeNet-5 model.

    Defined in :numref:`sec_lenet`"""
    lr: float = 0.1
    num_classes: int = 10
    kernel_init: FunctionType = nn.initializers.xavier_uniform

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(features=6, kernel_size=(5, 5), padding='SAME',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nn.Conv(features=16, kernel_size=(5, 5), padding='VALID',
                    kernel_init=self.kernel_init()),
            nn.sigmoid,
            lambda x: nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nn.Dense(features=120, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=84, kernel_init=self.kernel_init()),
            nn.sigmoid,
            nn.Dense(features=self.num_classes, kernel_init=self.kernel_init())
        ])

class Residual(nn.Module):
    """The Residual block of ResNet models.

    Defined in :numref:`sec_resnet`"""
    num_channels: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        self.conv1 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same', strides=self.strides)
        self.conv2 = nn.Conv(self.num_channels, kernel_size=(3, 3),
                             padding='same')
        # Auto-enable 1x1 conv when downsampling so the residual shape matches.
        if self.use_1x1conv or any(s != 1 for s in self.strides):
            self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                 strides=self.strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return nn.relu(Y)

class ResNeXtBlock(nn.Module):
    """The ResNeXt block.

    Defined in :numref:`sec_resnet`"""
    num_channels: int
    groups: int
    bot_mul: int
    use_1x1conv: bool = False
    strides: tuple = (1, 1)
    training: bool = True

    def setup(self):
        bot_channels = int(round(self.num_channels * self.bot_mul))
        self.conv1 = nn.Conv(bot_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.conv2 = nn.Conv(bot_channels, kernel_size=(3, 3),
                               strides=self.strides, padding='same',
                               feature_group_count=self.groups)
        self.conv3 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                               strides=(1, 1))
        self.bn1 = nn.BatchNorm(not self.training)
        self.bn2 = nn.BatchNorm(not self.training)
        self.bn3 = nn.BatchNorm(not self.training)
        if self.use_1x1conv:
            self.conv4 = nn.Conv(self.num_channels, kernel_size=(1, 1),
                                       strides=self.strides)
            self.bn4 = nn.BatchNorm(not self.training)
        else:
            self.conv4 = None

    def __call__(self, X):
        Y = nn.relu(self.bn1(self.conv1(X)))
        Y = nn.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return nn.relu(Y + X)

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

    def __init__(self, batch_size, num_steps, num_train=10000, num_val=5000):
        super(d2l.TimeMachine, self).__init__()
        self.save_hyperparameters()
        corpus, self.vocab = self.build(self._download())
        array = d2l.tensor([corpus[i:i+num_steps+1] 
                            for i in range(len(corpus)-num_steps)])
        self.X, self.Y = array[:,:-1], array[:,1:]

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(
            self.num_train, self.num_train + self.num_val)
        return self.get_tensorloader([self.X, self.Y], train, idx)

class Vocab:
    """Vocabulary for text.

    Defined in :numref:`sec_text-sequence`"""
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        # Flatten a 2D list if needed
        if tokens and isinstance(tokens[0], list):
            tokens = [token for line in tokens for token in line]
        # Count token frequencies
        counter = collections.Counter(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                  reverse=True)
        # The list of unique tokens, ordered by descending frequency.
        # Reserve <unk> at index 0 so vocab[0] is the unknown token.
        self.idx_to_token = ['<unk>'] + reserved_tokens + [
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
        if hasattr(indices, '__len__') and len(indices) > 1:
            return [self.idx_to_token[int(index)] for index in indices]
        return self.idx_to_token[indices]

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx['<unk>']

class RNNScratch(nn.Module):
    """The RNN model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens,))

    def __call__(self, inputs, state=None):
        if state is not None:
            state, = state
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
            state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
                d2l.matmul(state, self.W_hh) if state is not None else 0)
                             + self.b_h)
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
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)

    def one_hot(self, X):    
        # Output shape: (num_steps, batch_size, vocab_size)    
        return jax.nn.one_hot(X.T, self.vocab_size)

    def output_layer(self, rnn_outputs):
        outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return d2l.stack(outputs, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab, params):
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = d2l.tensor([[outputs[-1]]])
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                                embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.apply({'params': params}, rnn_outputs,
                               method=self.output_layer)
                outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), ())))
        return ''.join([vocab.idx_to_token[i] for i in outputs])

class RNN(nn.Module):
    """The RNN model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    num_hiddens: int

    @nn.compact
    def __call__(self, inputs, H=None):
        if H is None:
            batch_size = inputs.shape[1]
            H = nn.SimpleCell(features=self.num_hiddens).initialize_carry(
                jax.random.PRNGKey(0), (batch_size, self.num_hiddens))

        SimpleRNN = nn.scan(nn.SimpleCell, variable_broadcast="params",
                            in_axes=0, out_axes=0,
                            split_rngs={"params": False})

        H, outputs = SimpleRNN(features=self.num_hiddens)(H, inputs)
        return outputs, H

class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    training: bool = True

    def setup(self):
        self.linear = nn.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        return d2l.swapaxes(self.linear(hiddens), 0, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state, self.training)
        return self.output_layer(rnn_outputs)

class GRU(d2l.RNN):
    """The multilayer GRU model.

    Defined in :numref:`sec_deep_rnn`"""
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    @nn.compact
    def __call__(self, X, state=None, training=False):
        outputs = X
        new_state = []
        if state is None:
            batch_size = X.shape[1]
            state = [nn.GRUCell(features=self.num_hiddens).initialize_carry(
                jax.random.PRNGKey(0),
                (batch_size, self.num_hiddens))] * self.num_layers

        GRU = nn.scan(nn.GRUCell, variable_broadcast="params",
                      in_axes=0, out_axes=0, split_rngs={"params": False})

        # Introduce a dropout layer after every GRU layer except last
        for i in range(self.num_layers - 1):
            layer_i_state, X = GRU(features=self.num_hiddens)(state[i], outputs)
            new_state.append(layer_i_state)
            X = nn.Dropout(self.dropout, deterministic=not training)(X)

        # Final GRU layer without dropout
        out_state, X = GRU(features=self.num_hiddens)(state[-1], X)
        new_state.append(out_state)
        return X, jnp.array(new_state)

class MTFraEng(d2l.DataModule):
    """The English-French dataset.

    Defined in :numref:`sec_machine_translation`"""
    def _download(self):
        d2l.extract(d2l.download(
            d2l.DATA_URL+'fra-eng.zip', self.root, 
            '94646ad1522d915e7b0f9296181140edcf86a4f5'))
        with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()

    def _preprocess(self, text):
        # Replace non-breaking space with space
        text = text.replace('\u202f', ' ').replace('\xa0', ' ')
        # Insert space between words and punctuation marks
        no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
        out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
               for i, char in enumerate(text.lower())]
        return ''.join(out)

    def _tokenize(self, text, max_examples=None):
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i >= max_examples: break
            parts = line.split('\t')
            if len(parts) == 2:
                # Skip empty tokens
                src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
        return src, tgt

    def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
        super(MTFraEng, self).__init__()
        self.save_hyperparameters()
        self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
            self._download())

    def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
        def _build_array(sentences, vocab, is_tgt=False):
            pad_or_trim = lambda seq, t: (
                seq[:t-1] + ['<eos>'] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
            sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
            if is_tgt:
                sentences = [['<bos>'] + s for s in sentences]
            if vocab is None:
                vocab = d2l.Vocab(sentences, min_freq=2)
            array = d2l.tensor([vocab[s] for s in sentences])
            valid_len = d2l.reduce_sum(
                d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
            return array, vocab, valid_len
        src, tgt = self._tokenize(self._preprocess(raw_text), 
                                  self.num_train + self.num_val)
        src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
        tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
        return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),
                src_vocab, tgt_vocab)

    def get_dataloader(self, train):
        idx = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader(self.arrays, train, idx)

    def build(self, src_sentences, tgt_sentences):
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
            src_sentences, tgt_sentences)])
        arrays, _, _ = self._build_arrays(
            raw_text, self.src_vocab, self.tgt_vocab)
        return arrays

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

class Encoder(nn.Module):
    """The base encoder interface for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def __call__(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def setup(self):
        raise NotImplementedError

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def __call__(self, X, state):
        raise NotImplementedError

class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder--decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    encoder: nn.Module
    decoder: nn.Module

    def __call__(self, enc_X, dec_X, *args, training=False):
        enc_all_outputs = self.encoder(enc_X, *args, training=training)
        dec_state = self.decoder.init_state(enc_all_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=training)[0]

    def predict_step(self, params, batch, num_steps,
                     save_attention_weights=False):
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs, inter_enc_vars = self.encoder.apply(
            {'params': params['encoder']}, src, src_valid_len, training=False,
            mutable='intermediates')
        # Save encoder attention weights if inter_enc_vars containing encoder
        # attention weights is not empty. (to be covered later)
        enc_attention_weights = []
        if bool(inter_enc_vars) and save_attention_weights:
            # Encoder Attention Weights saved in the intermediates collection
            enc_attention_weights = inter_enc_vars[
                'intermediates']['enc_attention_weights'][0]

        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [d2l.expand_dims(tgt[:,0], 1), ], []
        for _ in range(num_steps):
            (Y, dec_state), inter_dec_vars = self.decoder.apply(
                {'params': params['decoder']}, outputs[-1], dec_state,
                training=False, mutable='intermediates')
            outputs.append(d2l.argmax(Y, 2))
            # Save attention weights (to be covered later)
            if save_attention_weights:
                # Decoder Attention Weights saved in the intermediates collection
                dec_attention_weights = inter_dec_vars[
                    'intermediates']['dec_attention_weights'][0]
                attention_weights.append(dec_attention_weights)
        return d2l.concat(outputs[1:], 1), (attention_weights,
                                            enc_attention_weights)

class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence-to-sequence learning.

    Defined in :numref:`sec_seq2seq`"""
    vocab_size: int
    embed_size: int
    num_hiddens: int
    num_layers: int
    dropout: float = 0

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.embed_size)
        self.rnn = d2l.GRU(self.num_hiddens, self.num_layers, self.dropout)

    def __call__(self, X, *args, training=False):
        # X shape: (batch_size, num_steps)
        embs = self.embedding(d2l.astype(d2l.transpose(X), d2l.int64))
        # embs shape: (num_steps, batch_size, embed_size)
        outputs, state = self.rnn(embs, training=training)
        # outputs shape: (num_steps, batch_size, num_hiddens)
        # state shape: (num_layers, batch_size, num_hiddens)
        return outputs, state

class Seq2Seq(d2l.EncoderDecoder):
    """The RNN encoder--decoder for sequence to sequence learning.

    Defined in :numref:`sec_seq2seq_decoder`"""
    encoder: nn.Module
    decoder: nn.Module
    tgt_pad: int
    lr: float

    @partial(jax.jit, static_argnums=(0, 5))
    def loss(self, params, X, Y, state, averaged=False):
        Y_hat = state.apply_fn({'params': params}, *X, training=True,
                               rngs={'dropout': state.dropout_rng})
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = d2l.reshape(Y, (-1,))
        fn = optax.softmax_cross_entropy_with_integer_labels
        l = fn(Y_hat, Y)
        mask = d2l.astype(Y != self.tgt_pad, d2l.float32)
        return d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask), {}

    def validation_step(self, params, batch, state):
        # Evaluate with dropout disabled (training=False); training=True path
        # is used by self.loss during fit.
        Y_hat = state.apply_fn({'params': params}, *batch[:-1],
                               training=False)
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = d2l.reshape(batch[-1], (-1,))
        fn = optax.softmax_cross_entropy_with_integer_labels
        l = fn(Y_hat, Y)
        mask = d2l.astype(Y != self.tgt_pad, d2l.float32)
        l = d2l.reduce_sum(l * mask) / d2l.reduce_sum(mask)
        self.plot('loss', l, train=False)

    def configure_optimizers(self):
        # Adam optimizer is used here
        return optax.adam(learning_rate=self.lr)

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

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.

    Defined in :numref:`sec_queries-keys-values`"""
    d2l.use_svg_display()
    num_rows, num_cols, _, _ = matrices.shape
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix, cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = jnp.arange((maxlen),
                          dtype=jnp.float32)[None, :] < valid_len[:, None]
        return jnp.where(mask, X, value)

    if valid_lens is None:
        return nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if valid_lens.ndim == 1:
            valid_lens = jnp.repeat(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.softmax(X.reshape(shape), axis=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Defined in :numref:`sec_attention-scoring-functions`"""
    dropout: float

    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    @nn.compact
    def __call__(self, queries, keys, values, valid_lens=None,
                 training=False):
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.swapaxes(1, 2)
        scores = queries@(keys.swapaxes(1, 2)) / math.sqrt(d)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        return dropout_layer(attention_weights)@values, attention_weights

class AdditiveAttention(nn.Module):
    num_hiddens: int
    dropout: float

    def setup(self):
        self.W_k = nn.Dense(self.num_hiddens, use_bias=False)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=False)
        self.w_v = nn.Dense(1, use_bias=False)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = jnp.expand_dims(queries, axis=2) + jnp.expand_dims(keys, axis=1)
        features = nn.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        attention_weights = masked_softmax(scores, valid_lens)
        dropout_layer = nn.Dropout(self.dropout, deterministic=not training)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return dropout_layer(attention_weights)@values, attention_weights

class AttentionDecoder(d2l.Decoder):
    """The base attention-based decoder interface.

    Flax modules are dataclasses, so the base class deliberately omits
    `__init__`; subclasses declare their fields as class-level
    annotations and (optionally) a `setup()` method.

    Defined in :numref:`sec_seq2seq_attention`"""
    @property
    def attention_weights(self):
        raise NotImplementedError

class MultiHeadAttention(nn.Module):
    num_hiddens: int
    num_heads: int
    dropout: float
    bias: bool = False

    def setup(self):
        self.attention = d2l.DotProductAttention(self.dropout)
        self.W_q = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_k = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_v = nn.Dense(self.num_hiddens, use_bias=self.bias)
        self.W_o = nn.Dense(self.num_hiddens, use_bias=self.bias)

    @nn.compact
    def __call__(self, queries, keys, values, valid_lens, training=False):
        # Shape of queries, keys, or values:
        # (batch_size, no. of queries or key-value pairs, num_hiddens)
        # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        # After transposing, shape of output queries, keys, or values:
        # (batch_size * num_heads, no. of queries or key-value pairs,
        # num_hiddens / num_heads)
        queries = self.transpose_qkv(self.W_q(queries))
        keys = self.transpose_qkv(self.W_k(keys))
        values = self.transpose_qkv(self.W_v(values))

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            valid_lens = jnp.repeat(valid_lens, self.num_heads, axis=0)

        # Shape of output: (batch_size * num_heads, no. of queries,
        # num_hiddens / num_heads)
        output, attention_weights = self.attention(
            queries, keys, values, valid_lens, training=training)
        # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
        output_concat = self.transpose_output(output)
        return self.W_o(output_concat), attention_weights

    def transpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads.

        Defined in :numref:`sec_multihead-attention`"""
        # Shape of input X: (batch_size, no. of queries or key-value pairs,
        # num_hiddens). Shape of output X: (batch_size, no. of queries or
        # key-value pairs, num_heads, num_hiddens / num_heads)
        X = X.reshape((X.shape[0], X.shape[1], self.num_heads, -1))
        # Shape of output X: (batch_size, num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        X = jnp.transpose(X, (0, 2, 1, 3))
        # Shape of output: (batch_size * num_heads, no. of queries or key-value
        # pairs, num_hiddens / num_heads)
        return X.reshape((-1, X.shape[2], X.shape[3]))

    def transpose_output(self, X):
        """Reverse the operation of transpose_qkv.

        Defined in :numref:`sec_multihead-attention`"""
        X = X.reshape((-1, self.num_heads, X.shape[1], X.shape[2]))
        X = jnp.transpose(X, (0, 2, 1, 3))
        return X.reshape((X.shape[0], X.shape[1], -1))

class PositionalEncoding(nn.Module):
    """Positional encoding.

    Defined in :numref:`sec_self-attention-and-positional-encoding`"""
    num_hiddens: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        # Create a long enough P
        self.P = d2l.zeros((1, self.max_len, self.num_hiddens))
        X = d2l.arange(self.max_len, dtype=jnp.float32).reshape(
            -1, 1) / jnp.power(10000, jnp.arange(
            0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens)
        self.P = self.P.at[:, :, 0::2].set(jnp.sin(X))
        self.P = self.P.at[:, :, 1::2].set(jnp.cos(X[:, :self.num_hiddens // 2]))

    @nn.compact
    def __call__(self, X, training=False, offset=0):
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'P', self.P)
        # `offset` lets autoregressive decoders advance the encoding position
        # past tokens already emitted, instead of always slicing from 0.
        X = X + self.P[:, offset:offset + X.shape[1], :]
        return nn.Dropout(self.dropout)(X, deterministic=not training)

class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network.

    Defined in :numref:`sec_transformer`"""
    ffn_num_hiddens: int
    ffn_num_outputs: int

    def setup(self):
        self.dense1 = nn.Dense(self.ffn_num_hiddens)
        self.dense2 = nn.Dense(self.ffn_num_outputs)

    def __call__(self, X):
        return self.dense2(nn.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """The residual connection followed by layer normalization.

    Defined in :numref:`sec_transformer`"""
    dropout: float

    @nn.compact
    def __call__(self, X, Y, training=False):
        return nn.LayerNorm()(
            nn.Dropout(self.dropout)(Y, deterministic=not training) + X)

class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block.

    Defined in :numref:`sec_transformer`"""
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.attention = d2l.MultiHeadAttention(self.num_hiddens, self.num_heads,
                                                self.dropout, self.use_bias)
        self.addnorm1 = AddNorm(self.dropout)
        self.ffn = PositionWiseFFN(self.ffn_num_hiddens, self.num_hiddens)
        self.addnorm2 = AddNorm(self.dropout)

    def __call__(self, X, valid_lens, training=False):
        output, attention_weights = self.attention(X, X, X, valid_lens,
                                                   training=training)
        Y = self.addnorm1(X, output, training=training)
        return self.addnorm2(Y, self.ffn(Y), training=training), attention_weights

class TransformerEncoder(d2l.Encoder):
    """The Transformer encoder.

    Defined in :numref:`sec_transformer`"""
    vocab_size: int
    num_hiddens:int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float
    use_bias: bool = False

    def setup(self):
        self.embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(self.num_hiddens, self.dropout)
        self.blks = [TransformerEncoderBlock(self.num_hiddens,
                                             self.ffn_num_hiddens,
                                             self.num_heads,
                                             self.dropout, self.use_bias)
                     for _ in range(self.num_blks)]

    def __call__(self, X, valid_lens, training=False):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.embedding(X) * math.sqrt(self.num_hiddens)
        X = self.pos_encoding(X, training=training)
        attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X, attention_w = blk(X, valid_lens, training=training)
            attention_weights[i] = attention_w
        # Flax sow API is used to capture intermediate variables
        self.sow('intermediates', 'enc_attention_weights', attention_weights)
        return X

def annotate(text, xy, xytext):
    d2l.plt.gca().annotate(text, xy=xy, xytext=xytext,
                           arrowprops=dict(arrowstyle='->'))

def train_2d(trainer, steps=20, f_grad=None):
    """Optimize a 2D objective function with a customized trainer.

    Defined in :numref:`sec_gd`"""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):
    """Show the trace of 2D variables during optimization.

    Defined in :numref:`sec_gd`"""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')

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

d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (jnp.array(data[:n, :-1]), jnp.array(data[:n, -1])),
        batch_size, is_train=True)
    return data_iter, data.shape[1]-1

def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = jnp.array(np.random.normal(scale=0.01, size=(feature_dim, 1)),
                  dtype=jnp.float32)
    b = jnp.zeros(1)
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    # JIT only the grad computation; the optimizer update runs eagerly so
    # that stateful optimizers can mutate `states` without triggering JAX
    # tracer-leak errors from closure side-effects inside jit.
    @jax.jit
    def compute_grads(w, b, X, y):
        def loss_fn(w, b):
            return d2l.squared_loss(d2l.linreg(X, w, b), y).mean()
        return jax.grad(loss_fn, argnums=(0, 1))(w, b)
    # Pre-stack the full dataset on device so the periodic evaluate_loss
    # stays inside one compiled call instead of looping in Python.
    eval_batches = [(jnp.array(X), jnp.array(y)) for X, y in data_iter]
    Xs = jnp.concatenate([X for X, _ in eval_batches], axis=0)
    ys = jnp.concatenate([y for _, y in eval_batches], axis=0)
    @jax.jit
    def full_eval(w, b):
        out = d2l.linreg(Xs, w, b)
        y_r = ys.reshape(out.shape)
        return ((out - y_r) ** 2 / 2).mean()
    for _ in range(num_epochs):
        for X, y in data_iter:
            X, y = jnp.array(X), jnp.array(y)
            grads = compute_grads(w, b, X, y)
            w, b = trainer_fn([w, b], list(grads), states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (float(full_eval(w, b)),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]

def train_concise_ch11(trainer_fn, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Dense(1)
    key = jax.random.PRNGKey(0)
    X_dummy = jnp.ones((1, 5))
    params = net.init(key, X_dummy)

    optimizer = trainer_fn(**hyperparams)
    opt_state = optimizer.init(params)

    loss = lambda pred, y: jnp.mean((pred - y) ** 2) / 2
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    # JIT-fuse the per-batch optimizer update so per-step Python overhead
    # stays out of the inner loop.
    @jax.jit
    def step(params, opt_state, X, y):
        def loss_fn(params):
            out = net.apply(params, X)
            y_reshaped = y.reshape(out.shape)
            return jnp.mean((out - y_reshaped) ** 2) / 2
        l, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, l

    # Pre-stack the full dataset on device so the periodic full-loss
    # evaluation is a single compiled call.
    eval_batches = [(jnp.array(X), jnp.array(y)) for X, y in data_iter]
    Xs = jnp.concatenate([X for X, _ in eval_batches], axis=0)
    ys = jnp.concatenate([y for _, y in eval_batches], axis=0)
    @jax.jit
    def full_eval(params):
        out = net.apply(params, Xs)
        y_r = ys.reshape(out.shape)
        return jnp.mean((out - y_r) ** 2) / 2
    for _ in range(num_epochs):
        for X, y in data_iter:
            X, y = jnp.array(X), jnp.array(y)
            params, opt_state, _ = step(params, opt_state, X, y)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (float(full_eval(params)),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')

class Benchmark:
    """For measuring running time.

    Defined in :numref:`sec_hybridize`"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.timer = d2l.Timer()
        return self

    def __exit__(self, *args):
        print(f'{self.description}: {self.timer.stop():.4f} sec')

def split_batch(X, y, num_devices):
    """Split `X` and `y` across devices by reshaping.

    Defined in :numref:`sec_multi_gpu`"""
    assert X.shape[0] == y.shape[0]
    batch_size = X.shape[0]
    # Reshape (batch, ...) -> (num_devices, batch_per_device, ...)
    def _reshape(a):
        return a.reshape(num_devices, batch_size // num_devices, *a.shape[1:])
    return _reshape(X), _reshape(y)

class ResNet18(nn.Module):
    """A slightly modified ResNet-18 model.

    Defined in :numref:`sec_multi_gpu_concise`"""
    num_classes: int = 10
    training: bool = True

    def setup(self):
        self.net = nn.Sequential([
            nn.Conv(64, kernel_size=(3, 3), strides=(1, 1), padding='same'),
            nn.BatchNorm(not self.training),
            nn.relu,
            # ResNet blocks
            d2l.Residual(64, training=self.training),
            d2l.Residual(64, training=self.training),
            d2l.Residual(128, use_1x1conv=True, strides=(2, 2),
                         training=self.training),
            d2l.Residual(128, training=self.training),
            d2l.Residual(256, use_1x1conv=True, strides=(2, 2),
                         training=self.training),
            d2l.Residual(256, training=self.training),
            d2l.Residual(512, use_1x1conv=True, strides=(2, 2),
                         training=self.training),
            d2l.Residual(512, training=self.training),
            # Global average pooling and classifier
            lambda x: x.mean(axis=(1, 2)),
            nn.Dense(self.num_classes),
        ])

    def __call__(self, x):
        return self.net(x)

@partial(jax.jit, static_argnums=(3, 4))  # net, loss_fn are static
def train_batch_ch13(state, X, y, net, loss_fn):
    """Train for a minibatch with JAX (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    def compute_loss(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            X, mutable=['batch_stats'])
        loss = loss_fn(logits, y).mean()
        return loss, (logits, updates)
    (loss, (logits, updates)), grads = jax.value_and_grad(
        compute_loss, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
    train_loss_sum = loss * X.shape[0]
    train_acc_sum = (logits.argmax(axis=-1) == y).sum()
    return state, train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss_fn, state, num_epochs):
    """Train a model with JAX (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    num_batches = sum(1 for _ in tfds.as_numpy(train_iter))
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()

    # Use a separate eval module (training=False) so BatchNorm uses
    # running stats instead of batch stats at test time. Params are shared
    # with the training network; only the `training` flag differs.
    eval_net = net.clone(training=False)

    @jax.jit
    def eval_step(params, batch_stats, X):
        logits = eval_net.apply(
            {'params': params, 'batch_stats': batch_stats}, X)
        return logits

    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(tfds.as_numpy(train_iter)):
            timer.start()
            state, l, acc = train_batch_ch13(
                state, jnp.array(features), jnp.array(labels), net, loss_fn)
            n = features.shape[0]
            metric.add(float(l), float(acc), n, n)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        # Evaluate on test set
        correct, total = 0, 0
        for X, y in tfds.as_numpy(test_iter):
            logits = eval_step(state.params, state.batch_stats, jnp.array(X))
            correct += int((logits.argmax(axis=-1) == y).sum())
            total += y.shape[0]
        test_acc = correct / total
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')
    return state

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

class BERTEncoder(nn.Module):
    """BERT encoder.

    Defined in :numref:`sec_bert`"""
    vocab_size: int
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        self.token_embedding = nn.Embed(self.vocab_size, self.num_hiddens)
        self.segment_embedding = nn.Embed(2, self.num_hiddens)
        self.blks = [d2l.TransformerEncoderBlock(
            self.num_hiddens, self.ffn_num_hiddens, self.num_heads,
            self.dropout, True) for _ in range(self.num_blks)]
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.param('pos_embedding',
                                        nn.initializers.normal(0.02),
                                        (1, self.max_len, self.num_hiddens))

    def __call__(self, tokens, segments, valid_lens, training=False):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X, _ = blk(X, valid_lens, training=training)
        return X

class MaskLM(nn.Module):
    """The masked language model task of BERT.

    Defined in :numref:`sec_bert`"""
    vocab_size: int
    num_hiddens: int

    @nn.compact
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
        mlm_Y_hat = nn.Dense(self.num_hiddens)(masked_X)
        mlm_Y_hat = nn.relu(mlm_Y_hat)
        mlm_Y_hat = nn.LayerNorm()(mlm_Y_hat)
        mlm_Y_hat = nn.Dense(self.vocab_size)(mlm_Y_hat)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT.

    Defined in :numref:`sec_bert`"""
    @nn.compact
    def __call__(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return nn.Dense(2)(X)

class BERTModel(nn.Module):
    """The BERT model.

    Defined in :numref:`sec_bert`"""
    vocab_size: int
    num_hiddens: int
    ffn_num_hiddens: int
    num_heads: int
    num_blks: int
    dropout: float
    max_len: int = 1000

    def setup(self):
        self.encoder = BERTEncoder(
            self.vocab_size, self.num_hiddens, self.ffn_num_hiddens,
            self.num_heads, self.num_blks, self.dropout,
            max_len=self.max_len)
        self.hidden = nn.Dense(self.num_hiddens)
        self.mlm = MaskLM(self.vocab_size, self.num_hiddens)
        self.nsp = NextSentencePred()

    def __call__(self, tokens, segments, valid_lens=None, pred_positions=None,
                 training=False):
        encoded_X = self.encoder(tokens, segments, valid_lens,
                                 training=training)
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

def _read_wiki(data_dir=None):
    import contextlib
    import io
    import pandas as pd
    with contextlib.redirect_stdout(io.StringIO()):
        fname = d2l.download(WIKITEXT_2_URL, folder='../data')
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
        all_token_ids.append(jnp.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=jnp.int32))
        all_segments.append(jnp.array(segments + [0] * (
            max_len - len(segments)), dtype=jnp.int32))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(jnp.array(len(token_ids), dtype=jnp.float32))
        all_pred_positions.append(jnp.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=jnp.int32))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            jnp.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=jnp.float32))
        all_mlm_labels.append(jnp.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=jnp.int32))
        nsp_labels.append(jnp.array(is_next, dtype=jnp.int32))
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
            yield (jnp.stack([b[0] for b in batch]),
                   jnp.stack([b[1] for b in batch]),
                   jnp.stack([b[2] for b in batch]),
                   jnp.stack([b[3] for b in batch]),
                   jnp.stack([b[4] for b in batch]),
                   jnp.stack([b[5] for b in batch]),
                   jnp.stack([b[6] for b in batch]))
    return data_iter, train_set.vocab

def _get_batch_loss_bert(params, net, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y, rng):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net.apply(params, tokens_X, segments_X,
                                        valid_lens_x.reshape(-1),
                                        pred_positions_X, training=True,
                                        rngs={'dropout': rng})
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
    train_features = jnp.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = jnp.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, jnp.array(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, jnp.array(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

def predict_sentiment(net, params, vocab, sequence):
    """Predict the sentiment of a text sequence.

    Defined in :numref:`sec_sentiment_rnn`"""
    sequence = jnp.array(vocab[sequence.split()])
    label = jnp.argmax(net.apply(params, sequence.reshape(1, -1)), axis=1)
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
        self.labels = jnp.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return jnp.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

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

def predict_snli(net, params, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis.

    Defined in :numref:`sec_natural-language-inference-attention`"""
    premise = jnp.array(vocab[premise]).reshape((1, -1))
    hypothesis = jnp.array(vocab[hypothesis]).reshape((1, -1))
    label = jnp.argmax(net.apply(params, premise, hypothesis, training=False),
                       axis=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'

def rbfkernel(x1, x2, ls=4.):
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls**2 / 2) * (dist ** 2))

class HPOTrainer(d2l.Trainer):
    def validation_error(self):
        self.model.training = False
        accuracy = 0
        val_batch_idx = 0
        for batch in self.val_dataloader:
            batch = self.prepare_batch(batch)
            accuracy += self.model.accuracy(
                self.state.params, batch[:-1], batch[-1], self.state)
            val_batch_idx += 1
        return 1 - accuracy / val_batch_idx

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
        # Bookkeeping results for plotting
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        self.records = []

    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            print(f"Trial {i}: config = {config}")
            error = self.objective(**config)
            error = float(error)
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)
            print(f"    error = {error}, runtime = {runtime}")

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

from functools import partial

@partial(jax.jit, static_argnames=('net_D', 'net_G', 'optimizer_G'))
def update_G(Z, net_D, net_G, params_D, params_G, loss_fn, opt_state_G,
             optimizer_G):
    """Update generator.

    Defined in :numref:`sec_basic_gan`"""
    batch_size = Z.shape[0]
    ones = jnp.ones((batch_size,))
    def loss_G_fn(params_G):
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G.apply(params_G, Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D.apply(params_D, fake_X).squeeze()
        loss_G = jnp.sum(optax.sigmoid_binary_cross_entropy(fake_Y, ones))
        return loss_G
    loss_G, grads_G = jax.value_and_grad(loss_G_fn)(params_G)
    updates, opt_state_G = optimizer_G.update(grads_G, opt_state_G, params_G)
    params_G = optax.apply_updates(params_G, updates)
    return loss_G, params_G, opt_state_G

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

def linreg(X, w, b):
    """The linear regression model.

    Defined in :numref:`sec_utils`"""
    return d2l.matmul(X, w) + b

def squared_loss(y_hat, y):
    """Squared loss.

    Defined in :numref:`sec_utils`"""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a JAX-compatible data iterator.

    For training, the last partial batch is dropped so every minibatch
    has the same shape — this lets a `@jax.jit`'d step function compile
    once per epoch instead of recompiling for the smaller last batch.
    Validation iterators yield all batches (the last may be smaller),
    matching how PyTorch / TF / MXNet behave.

    Defined in :numref:`sec_utils`"""
    n = data_arrays[0].shape[0]
    indices = np.arange(n)
    last = n - (n % batch_size) if is_train else n
    def data_iter():
        if is_train:
            np.random.shuffle(indices)
        for i in range(0, last, batch_size):
            batch_indices = indices[i: min(i + batch_size, n)]
            yield tuple(jnp.array(a[batch_indices]) for a in data_arrays)
    class DataIter:
        def __iter__(self):
            return data_iter()
        def __len__(self):
            return last // batch_size if is_train else (n + batch_size - 1) // batch_size
    return DataIter()

class ArrayDataLoader:
    """A simple data loader for JAX that batches arrays or dataset objects.

    Defined in :numref:`sec_utils`"""
    def __init__(self, *args, batch_size=32, shuffle=False, drop_last=False,
                 **kwargs):
        if len(args) == 1 and hasattr(args[0], '__len__') and hasattr(args[0], '__getitem__'):
            dataset = args[0]
            items = [dataset[i] for i in range(len(dataset))]
            if isinstance(items[0], (tuple, list)):
                self.arrays = tuple(np.array([item[j] for item in items])
                                    for j in range(len(items[0])))
            else:
                self.arrays = (np.array(items),)
        else:
            self.arrays = tuple(np.asarray(a) for a in args)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        n = self.arrays[0].shape[0]
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, n, self.batch_size):
            end = i + self.batch_size
            if self.drop_last and end > n:
                break
            batch_idx = indices[i:min(end, n)]
            yield tuple(jnp.array(a[batch_idx]) for a in self.arrays)

    def __len__(self):
        n = self.arrays[0].shape[0]
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

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

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = d2l.numpy(img)
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

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
nn_Module = nn.Module
to = jax.device_put
numpy = np.asarray
transpose = lambda a: a.T
sigmoid = jax.nn.sigmoid
size = lambda x, *args, **kwargs: x.size
