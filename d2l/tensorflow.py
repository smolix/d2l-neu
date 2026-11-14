DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import os as _os
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import tensorflow as tf
import keras

# TensorFlow's Python logger prints performance hints as warnings, e.g.
# "5 out of the last 5 calls to <function> triggered tf.function retracing"
# for every eager grouped convolution. They are not errors, and in a notebook
# they land in the cell output; errors still surface as exceptions.
tf.get_logger().setLevel('ERROR')

# Enumerating physical GPUs calls cuInit even in TensorFlow's CPU runtime.
# An explicit CPU selection must therefore skip GPU discovery entirely.
if _os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)

nn_Module = tf.keras.Model

#################   WARNING   ################
# The below part is generated automatically through:
#    python tools/build_lib.py
# Don't edit it directly

import sys
d2l = sys.modules[__name__]

def split_batch(X, y, devices):
    """Split `X` and `y` into multiple devices.

    Defined in :numref:`sec_multi_gpu`"""
    assert X.shape[0] == y.shape[0]
    return (tf.split(X, len(devices)), tf.split(y, len(devices)))

def resnet18(num_classes, in_channels=1):
    """A slightly modified ResNet-18 model built with Keras.

    Defined in :numref:`sec_multi_gpu_concise`"""
    def resnet_block(num_channels, num_residuals, first_block=False):
        blk = tf.keras.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.add(d2l.Residual(num_channels, use_1x1conv=True,
                                     strides=2))
            else:
                blk.add(d2l.Residual(num_channels))
        return blk

    # He initialization, one initializer instance per layer
    he = lambda: tf.keras.initializers.VarianceScaling(
        2.0, 'fan_in', 'untruncated_normal')
    # Small stem conv without bias (the batch normalization after it removes
    # it) and no max-pooling. Running averages decay 0.9 and epsilon is 1e-5
    # (the Keras defaults are 0.99 and 1e-3)
    net = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same',
                               use_bias=False, kernel_initializer=he()),
        tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5),
        tf.keras.layers.Activation('relu'),
        resnet_block(64, 2, first_block=True),
        resnet_block(128, 2),
        resnet_block(256, 2),
        resnet_block(512, 2),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, kernel_initializer=he()),
    ])
    return net

class HyperParameters:
    """The base class of hyperparameters.

    Defined in :numref:`sec_utils`"""

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

import numpy as np
import tensorflow as tf

def synthetic_data(w, b, noise, n):
    """n examples with standard normal features and labels Xw + b + noise.

    Defined in :numref:`sec_synthetic-regression-data`"""
    X = tf.random.normal((n, w.shape[0]))
    eps = tf.random.normal((n, 1)) * noise
    return X, d2l.matmul(X, d2l.reshape(w, (-1, 1))) + b + eps

def add_to_class(Class):
    """Register functions as methods in created class.

    Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        return obj
    return wrapper

class Module(tf.keras.Model, d2l.HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
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
        return tf.keras.optimizers.SGD(float(self.lr))

def as_metrics(out):
    """A step returns a loss or a dict of metrics; return a dict.

    Defined in :numref:`sec_oo-design`"""
    return out if isinstance(out, dict) else {'loss': out}

class LinearRegressionScratch(d2l.Module):
    """The linear regression model implemented from scratch.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        normal = tf.keras.initializers.RandomNormal(mean=0, stddev=sigma)
        self.w = self.add_weight(name='w', shape=(num_inputs, 1),
                                 initializer=normal)
        self.b = self.add_weight(name='b', shape=(1,), initializer='zeros')

    def forward(self, X):
        return d2l.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2
        return d2l.reduce_mean(l)

    def configure_optimizers(self):
        return SGD(self.lr)

class SGD(d2l.HyperParameters):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)

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
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)

class SyntheticRegressionData(d2l.DataModule):
    """Synthetic data for linear regression.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000,
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        self.X, self.y = d2l.synthetic_data(w, b, noise, num_train + num_val)

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)

class Trainer(d2l.HyperParameters):
    """Fit a Module to a DataModule, keeping a per-epoch history.

    Defined in :numref:`sec_oo-design`"""

    def prepare_data(self, data):
        self.train_dataloader = data.train_dataloader()
        self.val_dataloader = data.val_dataloader()

    def prepare_optimizer(self, model):
        self.optim = model.configure_optimizers()

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

    def prepare_model(self, model):
        self.model = model
        loader = (self.train_dataloader if self.train_dataloader is not None
                  else self.val_dataloader)  # `evaluate` has no training data
        batch = self.prepare_batch(next(iter(loader)))
        model(*batch[:-1], training=False)  # Keras creates the variables here
        self._train_step = tf.function(self.train_step, reduce_retracing=True)
        self._val_step = tf.function(self.val_step, reduce_retracing=True)

    def train_step(self, batch):
        with tf.GradientTape() as tape:
            out = d2l.as_metrics(self.model.training_step(batch))
        params = self.model.trainable_variables
        grads = tape.gradient(out['loss'], params)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, params))
        return out

    def val_step(self, batch):
        return d2l.as_metrics(self.model.validation_step(batch))

    def fit_epoch(self):
        metrics = d2l.Averager()
        for batch in map(self.prepare_batch, self.train_dataloader):
            metrics.add(self._train_step(batch), len(batch[-1]))
        return metrics.mean()

    def validate(self):
        metrics = d2l.Averager()
        for batch in map(self.prepare_batch, self.val_dataloader):
            metrics.add(self._val_step(batch), len(batch[-1]))
        return metrics.mean()

    def evaluate(self, model, data, train=False):
        """Metrics of `model` on the validation data of `data` (on its training
        data if `train`), averaged over examples.

        Defined in :numref:`sec_softmax_scratch`"""
        self.train_dataloader = None
        self.val_dataloader = (data.train_dataloader() if train
                               else data.val_dataloader())
        self.prepare_model(model)
        return self.validate()

    def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
        self.save_hyperparameters()
        self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]
        self.board = d2l.ProgressBoard(xlabel='epoch', ylabel='loss')

    def prepare_batch(self, batch):
        if self.gpus:
            # tf.data.Dataset emits batches on CPU. Re-wrap them inside the
            # GPU device context so subsequent ops keep their inputs on-device
            # rather than incurring an implicit copy each step.
            with self.gpus[0]:
                batch = [tf.identity(a) for a in batch]
        return batch

    def clip_gradients(self, grad_clip_val, grads):
        grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
        new_grads = [tf.convert_to_tensor(grad) if isinstance(
            grad, tf.IndexedSlices) else grad for grad in grads]
        norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
        scale = tf.minimum(1.0, grad_clip_val / norm)
        return [grad * scale for grad in new_grads]

class Averager:
    """Example-weighted means of batch metrics, summed on the device.

    Defined in :numref:`sec_oo-design`"""
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

class LinearRegression(d2l.Module):
    """Linear regression built from framework components.

    Defined in :numref:`sec_linear_concise`"""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        init = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.layers.Dense(1, kernel_initializer=init)

    def loss(self, y_hat, y):
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)

    def configure_optimizers(self):
        return tf.keras.optimizers.SGD(self.lr)

    def get_w_b(self):
        return self.net.kernel, self.net.bias

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
        labels = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        X, y = self.train if train else self.val
        X = tf.expand_dims(X, axis=3) / 255  # Channel axis last; values in [0, 1]
        resize = lambda X, y: (tf.image.resize(X, self.resize), y)
        return tf.data.Dataset.from_tensor_slices(
            (X, tf.cast(y, tf.int32))).shuffle(len(y) if train else 1).batch(
                self.batch_size).map(resize)

    def visualize(self, batch, nrows=1, ncols=8, labels=None):
        """Show the images of a batch in grayscale, titled by their labels.

        Defined in :numref:`sec_fashion_mnist`"""
        X, y = batch
        if labels is None:
            labels = self.text_labels(y)
        X = d2l.reshape(X, (-1, *self.resize))  # Drop the channel axis
        d2l.show_images(X, nrows, ncols, titles=labels, cmap='gray')

class Classifier(d2l.Module):
    """The base class of classification models.

    Defined in :numref:`sec_softmax_scratch`"""
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1], training=False)
        return {'loss': self.loss(Y_hat, batch[-1]),
                'acc': self.accuracy(Y_hat, batch[-1])}

    def accuracy(self, Y_hat, Y, averaged=True):
        """Compute the fraction of correct predictions.

        Defined in :numref:`sec_softmax_scratch`"""
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
        compare = d2l.astype(preds == d2l.reshape(Y, (-1,)), d2l.float32)
        return d2l.reduce_mean(compare) if averaged else compare

    def loss(self, Y_hat, Y, averaged=True):
        Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
        Y = d2l.reshape(Y, (-1,))
        reduction = (tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
                     if averaged else tf.keras.losses.Reduction.NONE)
        fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=reduction)
        return fn(Y, Y_hat)

    def layer_summary(self, X_shape):
        X = d2l.normal(X_shape)
        for layer in self.net.layers:
            X = layer(X)
            name = getattr(layer, '__name__', type(layer).__name__)
            print(f'{name:<22} {tuple(X.shape)}')

def cross_entropy(y_hat, y):
    p = tf.gather(y_hat, y, batch_dims=1)
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    return -tf.reduce_mean(tf.math.log(tf.maximum(p, 1e-12)))

class SoftmaxRegression(d2l.Classifier):
    """The softmax regression model.

    Defined in :numref:`sec_softmax_concise`"""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential()
        self.net.add(tf.keras.layers.Flatten())
        self.net.add(tf.keras.layers.Dense(
            num_outputs,
            kernel_initializer=tf.initializers.RandomNormal(stddev=0.01)))

    def forward(self, X):
        return self.net(X)

class ArrayData(d2l.DataModule):
    """Minibatches of the arrays X and y; training batches are shuffled.

    Defined in :numref:`sec_generalization_deep`"""
    def __init__(self, X, y, batch_size=64):
        super().__init__()
        self.save_hyperparameters()

    def get_dataloader(self, train):
        return self.get_tensorloader(
            (d2l.tensor(self.X), d2l.tensor(self.y)), train)

class Tracker(d2l.Trainer):
    """A Trainer that records the accuracy on named datasets every epoch.

    Defined in :numref:`sec_generalization_deep`"""
    def __init__(self, max_epochs, datasets):
        super().__init__(max_epochs, num_gpus=1)
        self.datasets = datasets
        self.accuracy = {name: [] for name in datasets}
        self.board.display = False  # We plot the accuracies ourselves

    def validate(self):
        for name, dataset in self.datasets.items():
            self.val_dataloader = dataset.val_dataloader()
            self.accuracy[name].append(super().validate()['acc'])
        return {}

def corrupt(y, fraction, rng):
    """Replace each label, with probability fraction, by a random class.

    Defined in :numref:`sec_generalization_deep`"""
    redraw = rng.random(len(y)) < fraction
    y = y.copy()
    y[redraw] = rng.integers(0, 10, redraw.sum())
    return y

def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return tf.device('/CPU:0')

def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return tf.device(f'/GPU:{i}')

def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return len(tf.config.experimental.list_physical_devices('GPU'))

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
        self.desc = desc
        for _ in range(warmup):
            f()
        tf.test.experimental.sync_devices()
        t0 = time.perf_counter()
        for _ in range(repeats):
            f()
        tf.test.experimental.sync_devices()
        self.time = (time.perf_counter() - t0) / repeats

    def __repr__(self):
        return f'{self.desc}: {1000 * self.time:.2f} ms/call'

def corr2d(X, K):
    """Compute 2D cross-correlation.

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    # Build Y from its entries, so that gradients flow through it
    return tf.stack([
        tf.stack([tf.reduce_sum(X[i: i + h, j: j + w] * K)
                  for j in range(X.shape[1] - w + 1)])
        for i in range(X.shape[0] - h + 1)])

class LeNet(d2l.Classifier):
    """The layout of LeNet-5, with sigmoid activations and average pooling.

    Defined in :numref:`sec_lenet`"""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(6, kernel_size=5, padding='same'),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(16, kernel_size=5),
            tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120), tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(84), tf.keras.layers.Activation('sigmoid'),
            tf.keras.layers.Dense(num_classes)])

def cosine_warmup(t, num_steps, warmup_steps):
    """Learning-rate factor: linear warmup from 0, then a half-cosine to 0.

    Defined in :numref:`sec_alexnet`"""
    t = tf.cast(t, tf.float32)  # Also works on the step counter of a graph
    warmup = tf.where(t < warmup_steps, t / warmup_steps, 1.0)
    progress = tf.clip_by_value(
        (t - warmup_steps) / (num_steps - warmup_steps), 0.0, 1.0)
    return warmup * 0.5 * (1 + tf.cos(math.pi * progress))

class Residual(tf.keras.Model):
    """The Residual block of ResNet models.

    Defined in :numref:`sec_resnet`"""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # He initialization, one initializer instance per layer
        he = lambda: tf.keras.initializers.VarianceScaling(
            2.0, 'fan_in', 'untruncated_normal')
        # No conv biases: the batch normalization after each conv removes
        # them. conv1 pads 1 on every side: 'same' pads only right and bottom
        # when the stride is 2 and the sizes are even
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                   strides=strides, use_bias=False,
                                   kernel_initializer=he())])
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3,
            padding='same', use_bias=False, kernel_initializer=he())
        # Running averages decay 0.9 and epsilon 1e-5 (the Keras defaults are
        # 0.99 and 1e-3)
        bn = lambda: tf.keras.layers.BatchNormalization(momentum=0.9,
                                                        epsilon=1e-5)
        self.conv3 = None
        # Auto-enable 1x1 conv when downsampling so the residual shape matches.
        if use_1x1conv or strides != 1:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1,
                strides=strides, use_bias=False, kernel_initializer=he())
            self.bn3 = bn()
        self.bn1, self.bn2 = bn(), bn()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.bn3(self.conv3(X))
        Y += X
        return tf.keras.activations.relu(Y)

class ResNeXtBlock(tf.keras.Model):
    """The ResNeXt block.

    Defined in :numref:`sec_resnet`"""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        # He initialization, one initializer instance per layer; no conv
        # biases, since the batch normalization after each conv removes them
        he = lambda: tf.keras.initializers.VarianceScaling(
            2.0, 'fan_in', 'untruncated_normal')
        self.conv1 = tf.keras.layers.Conv2D(bot_channels, 1, strides=1,
            use_bias=False, kernel_initializer=he())
        self.conv2 = tf.keras.Sequential([  # Pad 1 on every side
            tf.keras.layers.ZeroPadding2D(1),
            tf.keras.layers.Conv2D(bot_channels, 3, strides=strides,
                                   groups=groups, use_bias=False,
                                   kernel_initializer=he())])
        self.conv3 = tf.keras.layers.Conv2D(num_channels, 1, strides=1,
            use_bias=False, kernel_initializer=he())
        # Running averages decay 0.9 and epsilon 1e-5 (the Keras defaults are
        # 0.99 and 1e-3)
        bn = lambda: tf.keras.layers.BatchNormalization(momentum=0.9,
                                                        epsilon=1e-5)
        self.bn1, self.bn2, self.bn3 = bn(), bn(), bn()
        # Auto-enable 1x1 conv when downsampling so the residual shape matches.
        if use_1x1conv or strides != 1:
            self.conv4 = tf.keras.layers.Conv2D(num_channels, 1,
                strides=strides, use_bias=False, kernel_initializer=he())
            self.bn4 = bn()
        else:
            self.conv4 = None

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = tf.keras.activations.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return tf.keras.activations.relu(Y + X)

def num_params(model):
    """The number of trainable parameters.

    Defined in :numref:`sec_depthwise_separable`"""
    return sum(int(tf.size(w)) for w in model.trainable_variables)

def mixup(X, y, alpha):
    """Mix a minibatch with a shuffled copy; return both label sets.

    Defined in :numref:`sec_training_recipes`"""
    g1, g2 = tf.random.gamma([], alpha), tf.random.gamma([], alpha)
    lam = g1 / (g1 + g2)  # Beta(alpha, alpha) as a ratio of Gammas
    perm = tf.random.shuffle(tf.range(tf.shape(X)[0]))
    return (lam * X + (1 - lam) * tf.gather(X, perm),
            y, tf.gather(y, perm), lam)

def drop_path(Y, p, training):
    """Zero each example of Y with probability p, rescale the rest.

    Defined in :numref:`sec_training_recipes`"""
    if not training or p == 0:
        return Y
    shape = [tf.shape(Y)[0]] + [1] * (len(Y.shape) - 1)
    keep = tf.random.uniform(shape) > p
    return Y * tf.cast(keep, Y.dtype) / (1 - p)

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

    def __init__(self, batch_size, num_steps, num_train=50000, num_val=5000,
                 tokenization='bpe', vocab_size=1024):
        super(d2l.TimeMachine, self).__init__()
        self.save_hyperparameters()
        raw_text = self._download()
        if tokenization == 'bpe':
            self.tokenizer = d2l.BPETokenizer(
                vocab_size, pattern=d2l.BPETokenizer.GPT2_PATTERN)
            self.tokenizer.train(raw_text)
            corpus, self.vocab = self.tokenizer.encode(raw_text), self.tokenizer
        else:  # tokenization='char': characters, via TimeMachine.build
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
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)  # Merge the pair and skip past it
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, text):
        """Learn vocab_size - 256 merges, most frequent pair first.

        Defined in :numref:`sec_text-sequence`"""
        chunk_freq = collections.Counter(self._chunks(text))
        seqs = [list(chunk.encode('utf-8')) for chunk in chunk_freq]
        for new_id in range(256, self.vocab_size):
            pairs = collections.Counter()
            for seq, w in zip(seqs, chunk_freq.values()):
                for pair in zip(seq, seq[1:]):
                    pairs[pair] += w  # A chunk that occurs w times counts w times
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

import textwrap

def print_wrapped(*args, width=76):
    """Print like `print`, folding each line to the width of a page.

    Defined in :numref:`sec_text-sequence`"""
    for line in ' '.join(str(a) for a in args).split('\n'):
        print(textwrap.fill(line, width, subsequent_indent='    '))

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

class RNNScratch(d2l.Module):
    """The RNN model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W_xh = self.add_weight(  # One initializer per weight
            name='W_xh', shape=(num_inputs, num_hiddens),
            initializer=tf.keras.initializers.RandomNormal(stddev=sigma))
        self.W_hh = self.add_weight(
            name='W_hh', shape=(num_hiddens, num_hiddens),
            initializer=tf.keras.initializers.RandomNormal(stddev=sigma))
        self.b_h = self.add_weight(name='b_h', shape=(num_hiddens,),
                                   initializer='zeros')

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = tf.zeros((tf.shape(inputs)[1], self.num_hiddens))
        outputs = []
        for X in tf.unstack(inputs):  # Shape: (num_steps, batch_size, num_inputs)
            state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                             d2l.matmul(state, self.W_hh) + self.b_h)
            outputs.append(state)
        return outputs, state

def check_shape(a, shape):
    """Check the shape of a tensor.

    Defined in :numref:`sec_rnn-scratch`"""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

class RNNLMScratch(d2l.Classifier):
    """The RNN-based language model implemented from scratch.

    Defined in :numref:`sec_rnn-scratch`"""
    def __init__(self, rnn, vocab_size, lr=0.01, dropout=0, emb_dropout=0):
        super().__init__()
        self.save_hyperparameters()
        self.sigma = getattr(rnn, 'sigma', 0.01)  # Scale of the output weights
        self.init_params()

    def init_params(self):
        self.E = self.add_weight(
            name='E', shape=(self.vocab_size, self.rnn.num_inputs),
            initializer=tf.keras.initializers.RandomNormal(stddev=1))
        self.W_hq = self.add_weight(
            name='W_hq', shape=(self.rnn.num_hiddens, self.vocab_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.sigma))
        self.b_q = self.add_weight(name='b_q', shape=(self.vocab_size,),
                                   initializer='zeros')

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1], training=False), batch[-1])

    def embedding(self, X):
        # Output shape: (num_steps, batch_size, num_inputs)
        return tf.gather(self.E, tf.transpose(X))

    def output_layer(self, rnn_outputs):
        # A list of per-step states, or a tensor whose first axis is time
        H = d2l.stack(list(rnn_outputs), 1)  # (batch_size, num_steps, num_hiddens)
        return d2l.matmul(H, self.W_hq) + self.b_q

    def forward(self, X, state=None):
        embs = self.embedding(X)
        if self.training:  # tf.nn.dropout is always active; apply it in training
            embs = tf.nn.dropout(embs, self.emb_dropout)
        rnn_outputs, _ = self.rnn(embs, state)
        if self.training:  # A list of per-step states (scratch) or one tensor
            rnn_outputs = tf.nest.map_structure(
                lambda H: tf.nn.dropout(H, self.dropout), rnn_outputs)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0):
        outputs, state = tok.encode(prefix), None
        for i in range(len(outputs) - 1):  # Warm up on the prefix
            X = d2l.tensor([[outputs[i]]])
            _, state = self.rnn(self.embedding(X), state)
        for _ in range(num_tokens):  # Generate num_tokens continuation tokens
            X = d2l.tensor([[outputs[-1]]])
            rnn_outputs, state = self.rnn(self.embedding(X), state)
            logits = self.output_layer(rnn_outputs)[0]  # Shape: (1, vocab_size)
            if temperature == 0:  # Greedy: the highest-scoring token
                token = tf.argmax(logits[0])
            else:  # Sample from the softmax of logits / temperature
                token = tf.random.categorical(logits / temperature, 1)[0, 0]
            outputs.append(int(token))
        return tok.decode(outputs)

class RNN(d2l.Module):
    """The RNN model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        normal = tf.keras.initializers.RandomNormal
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, kernel_initializer=normal(stddev=sigma),
            recurrent_initializer=normal(stddev=sigma),
            return_sequences=True, return_state=True)

    # Keras builds the layer by calling `call` with named arguments, which
    # `Module.call(X, *args)` cannot pass on, so we name the state here
    def call(self, inputs, H=None, training=None):
        return self.forward(inputs, H)

    def forward(self, inputs, H=None):
        # The keras layer is batch-major: transpose in and out
        outputs, H = self.rnn(tf.transpose(inputs, perm=[1, 0, 2]), H)
        return tf.transpose(outputs, perm=[1, 0, 2]), H

class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def init_params(self):
        normal = tf.keras.initializers.RandomNormal
        self.emb = tf.keras.layers.Embedding(
            self.vocab_size, self.rnn.num_inputs,
            embeddings_initializer=normal(stddev=1))
        self.linear = tf.keras.layers.Dense(
            self.vocab_size, kernel_initializer=normal(stddev=self.sigma))

    def embedding(self, X):
        return self.emb(tf.transpose(X))

    def output_layer(self, hiddens):
        return d2l.transpose(self.linear(hiddens), (1, 0, 2))

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

d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

def train_batch_ch13(net, X, y, loss, optimizer):
    """Train for a minibatch with Keras (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    with tf.GradientTape() as tape:
        pred = net(X, training=True)
        l = loss(y, pred)
    grads = tape.gradient(l, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    train_loss_sum = tf.reduce_sum(l)
    train_acc_sum = tf.reduce_sum(
        tf.cast(tf.argmax(pred, axis=1) == tf.cast(y, tf.int64), tf.float32))
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, optimizer, num_epochs):
    """Train a model with Keras (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    num_batches = sum(1 for _ in train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(net, features, labels, loss, optimizer)
            n = features.shape[0]
            metric.add(float(l), float(acc), n, n)
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        # Evaluate on test set
        correct, total = 0, 0
        for X, y in test_iter:
            logits = net(X, training=False)
            correct += int(tf.reduce_sum(tf.cast(
                tf.argmax(logits, axis=1) == tf.cast(y, tf.int64),
                tf.float32)))
            total += y.shape[0]
        test_acc = correct / total
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec')

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
    size_tensor = np.array(sizes, dtype=np.float32)
    ratio_tensor = np.array(ratios, dtype=np.float32)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (np.arange(in_height) + offset_h) * steps_h
    center_w = (np.arange(in_width) + offset_w) * steps_w
    shift_y, shift_x = np.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:])))\
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = np.repeat(np.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1), boxes_per_pixel, axis=0)
    output = tf.constant(out_grid + anchor_manipulations, dtype=tf.float32)
    return tf.expand_dims(output, axis=0)

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
    inter_upperlefts = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = tf.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = tf.clip_by_value(inter_lowerrights - inter_upperlefts, 0,
                              float('inf'))
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes.

    Defined in :numref:`sec_anchor`"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious = tf.reduce_max(jaccard, axis=1).numpy()
    indices = tf.argmax(jaccard, axis=1).numpy()
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # Use a plain Python loop — this runs once per sample, not in training
    col_discard = np.full((num_anchors,), -1.0)
    row_discard = np.full((num_gt_boxes,), -1.0)
    jaccard_np = jaccard.numpy()
    for _ in range(num_gt_boxes):
        max_idx = int(np.argmax(jaccard_np))
        box_idx = max_idx % num_gt_boxes
        anc_idx = max_idx // num_gt_boxes
        anchors_bbox_map[anc_idx] = box_idx
        jaccard_np[:, box_idx] = col_discard
        jaccard_np[anc_idx, :] = row_discard
    return tf.constant(anchors_bbox_map, dtype=tf.int32)

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
    batch_size, anchors = labels.shape[0], tf.squeeze(anchors, axis=0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    num_anchors = anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, None)
        bbox_mask = tf.tile(
            tf.expand_dims(
                tf.cast(anchors_bbox_map >= 0, dtype=tf.float32), axis=-1),
            [1, 4])
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = tf.zeros(num_anchors, dtype=tf.int32)
        assigned_bb = tf.zeros((num_anchors, 4), dtype=tf.float32)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = tf.cast(
            tf.squeeze(tf.where(anchors_bbox_map >= 0), axis=1), tf.int32)
        bb_idx = tf.gather(anchors_bbox_map, indices_true)
        class_labels = tf.tensor_scatter_nd_update(
            class_labels, tf.expand_dims(indices_true, 1),
            tf.cast(tf.gather(label[:, 0], bb_idx), tf.int32) + 1)
        assigned_bb = tf.tensor_scatter_nd_update(
            assigned_bb, tf.expand_dims(indices_true, 1),
            tf.gather(label[:, 1:], bb_idx))
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(tf.reshape(offset, [-1]))
        batch_mask.append(tf.reshape(bbox_mask, [-1]))
        batch_class_labels.append(class_labels)
    bbox_offset = tf.stack(batch_offset)
    bbox_mask = tf.stack(batch_mask)
    class_labels = tf.stack(batch_class_labels)
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
    return tf.constant(keep, dtype=tf.int32)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression.

    Defined in :numref:`sec_anchor`"""
    batch_size = cls_probs.shape[0]
    anchors = tf.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob = cls_probs[i]
        offset_pred = tf.reshape(offset_preds[i], (-1, 4))
        conf = tf.reduce_max(cls_prob[1:], axis=0)
        class_id = tf.cast(tf.argmax(cls_prob[1:], axis=0), tf.int32)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = tf.cast(tf.range(num_anchors), tf.int32)
        combined = tf.concat((keep, all_idx), axis=0)
        unique, _, counts = tf.unique_with_counts(combined)
        non_keep = tf.boolean_mask(unique, counts == 1)
        all_id_sorted = tf.concat((keep, non_keep), axis=0)
        # Set non-kept boxes' class to -1
        updates = tf.fill([tf.shape(non_keep)[0]], -1)
        class_id = tf.tensor_scatter_nd_update(
            class_id, tf.expand_dims(non_keep, 1), updates)
        class_id = tf.cast(tf.gather(class_id, all_id_sorted), tf.float32)
        conf = tf.gather(conf, all_id_sorted)
        predicted_bb = tf.gather(predicted_bb, all_id_sorted)
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = conf < pos_threshold
        class_id = tf.where(below_min_idx, tf.fill(tf.shape(class_id), -1.0),
                            class_id)
        conf = tf.where(below_min_idx, 1 - conf, conf)
        pred_info = tf.concat((tf.expand_dims(class_id, axis=1),
                               tf.expand_dims(conf, axis=1),
                               predicted_bb), axis=1)
        out.append(pred_info)
    return tf.stack(out)

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
        images.append(tf.constant(np.array(img), dtype=tf.float32))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, tf.expand_dims(tf.constant(targets, dtype=tf.float32),
                                  axis=1) / 256

class BananasDataset:
    """A customized dataset to load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """Load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    train_dataset = BananasDataset(is_train=True)
    val_dataset = BananasDataset(is_train=False)
    # Stack images: result shape is (N, H, W, C) — NHWC for TF
    train_images = tf.stack(train_dataset.features)
    val_images = tf.stack(val_dataset.features)
    train_iter = tf.data.Dataset.from_tensor_slices(
        (train_images, train_dataset.labels))
    # `drop_remainder=True` keeps every training minibatch the same
    # shape so the SSD `train_step` (`@tf.function`-wrapped at the call
    # site in :numref:`sec_ssd`) traces once per epoch shape instead of
    # retracing for the smaller last batch.
    train_iter = train_iter.shuffle(len(train_dataset.features)).batch(
        batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    val_iter = tf.data.Dataset.from_tensor_slices(
        (val_images, val_dataset.labels))
    val_iter = val_iter.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_iter, val_iter

class TinySSD(keras.Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.cls_loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)
        for i in range(5):
            # Equivalent to: self.blk_i, self.cls_i, self.bbox_i
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', cls_predictor(num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(num_anchors))

    def call(self, X, training=False):
        anchors_list = [None] * 5
        cls_preds_list = [None] * 5
        bbox_preds_list = [None] * 5
        # X is expected in NHWC layout (the Keras default)
        for i in range(5):
            blk = getattr(self, f'blk_{i}')
            cls_pred = getattr(self, f'cls_{i}')
            bbox_pred = getattr(self, f'bbox_{i}')
            X, anchors_list[i], cls_preds_list[i], bbox_preds_list[i] = \
                blk_forward(X, blk, sizes[i], ratios[i],
                            cls_pred, bbox_pred, training=training)
        anchors = tf.concat(anchors_list, axis=1)
        cls_preds = concat_preds(cls_preds_list)
        cls_preds = tf.reshape(cls_preds,
                               (tf.shape(cls_preds)[0], -1,
                                self.num_classes + 1))
        bbox_preds = concat_preds(bbox_preds_list)
        return anchors, cls_preds, bbox_preds

    def _compute_ssd_loss(self, cls_preds, cls_labels, bbox_preds,
                          bbox_labels, bbox_masks):
        num_classes = self.num_classes + 1
        cls = self.cls_loss(
            tf.reshape(cls_labels, [-1]),
            tf.reshape(cls_preds, [-1, num_classes]))
        bbox = tf.reduce_mean(
            tf.abs((bbox_preds * bbox_masks) -
                   (bbox_labels * bbox_masks)))
        return cls + bbox

    def train_step(self, data):
        features, target = data
        with tf.GradientTape() as tape:
            anchors, cls_preds, bbox_preds = self(features, training=True)
            bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(
                anchors, target)
            loss = self._compute_ssd_loss(cls_preds, cls_labels,
                                          bbox_preds, bbox_labels, bbox_masks)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        # Accumulate metrics without syncing large tensors to host
        cls_correct = tf.reduce_sum(
            tf.cast(tf.argmax(cls_preds, axis=-1) ==
                    tf.cast(cls_labels, tf.int64), tf.int64))
        cls_total = tf.cast(tf.size(cls_labels), tf.int64)
        bbox_abs = tf.reduce_sum(
            tf.abs((bbox_preds * bbox_masks) - (bbox_labels * bbox_masks)))
        bbox_total = tf.cast(tf.size(bbox_labels), tf.float32)
        return {'loss': loss,
                'cls_correct': cls_correct, 'cls_total': cls_total,
                'bbox_abs': bbox_abs, 'bbox_total': bbox_total}

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
    # Use NumPy for the random crop so this is safe to run from worker
    # threads (tf.image.random_crop holds graph-side state that breaks
    # under tf.data parallel py_function calls).
    H, W = feature.shape[0], feature.shape[1]
    top = int(np.random.randint(0, H - height + 1))
    left = int(np.random.randint(0, W - width + 1))
    feat = feature[top:top + height, left:left + width, :]
    lab = label[top:top + height, left:left + width, :]
    return feat, lab

class VOCSegDataset:
    """A customized dataset to load the VOC dataset.

    Defined in :numref:`sec_semantic_segmentation`"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.rgb_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.rgb_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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
        # feature: HWC float32; label: HWC uint8 RGB -> class indices HW int32
        label_idx = voc_label_indices(label, self.colormap2label)
        return (feature.astype(np.float32),
                label_idx.astype(np.int32))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset.

    Defined in :numref:`sec_semantic_segmentation`"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    train_dataset = VOCSegDataset(True, crop_size, voc_dir)
    test_dataset = VOCSegDataset(False, crop_size, voc_dir)
    n_train = len(train_dataset)
    n_test = len(test_dataset)
    # Drop last partial batch
    n_train = (n_train // batch_size) * batch_size
    n_test = (n_test // batch_size) * batch_size
    def make_tf_dataset(dataset, n, shuffle):
        # Cropping/labeling is plain NumPy; wrap in tf.py_function and run it
        # in parallel so the GPU isn't starved between batches. We use
        # from_tensor_slices(indices) + map(parallel) instead of a single
        # serial Python generator.
        feat0, label0 = dataset[0]
        feat_shape, label_shape = feat0.shape, label0.shape
        def load(i):
            feat, label = dataset[int(i)]
            return feat, label
        def tf_load(i):
            feat, label = tf.py_function(
                load, [i], (tf.float32, tf.int32))
            feat.set_shape(feat_shape)
            label.set_shape(label_shape)
            return feat, label
        ds = tf.data.Dataset.from_tensor_slices(
            np.arange(len(dataset), dtype=np.int64))
        if shuffle:
            ds = ds.shuffle(buffer_size=min(n, 1000),
                            reshuffle_each_iteration=True)
        ds = ds.take(n)
        ds = ds.map(tf_load, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    train_iter = make_tf_dataset(train_dataset, n_train, shuffle=True)
    test_iter = make_tf_dataset(test_dataset, n_test, shuffle=False)
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

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_utils`"""
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib.

    Defined in :numref:`sec_utils`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_log_ticks(axis):
    """Label all major ticks of a log axis in one format.

    Defined in :numref:`sec_utils`"""
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

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend,
             legend_loc=None):
    """Set the axes for matplotlib. legend_loc is a matplotlib legend
    location, or 'outside' for beside the axes, clear of the curves.

    Defined in :numref:`sec_utils`"""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    for axis, scale in ((axes.xaxis, xscale), (axes.yaxis, yscale)):
        if scale == 'log':
            set_log_ticks(axis)
    if legend and legend_loc == 'outside':
        axes.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
    elif legend:
        axes.legend(legend, loc=legend_loc)  # None: matplotlib's best spot
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None,
         legend_loc=None):
    """Plot data points.

    Defined in :numref:`sec_utils`"""
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
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend,
             legend_loc)

import os
from matplotlib.figure import Figure

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a TensorFlow data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

def sgd(params, grads, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_utils`"""
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_utils`"""
    mnist_train, mnist_test = tf.keras.datasets.fashion_mnist.load_data()
    # Divide all numbers by 255 so that all pixel values are between
    # 0 and 1, add a batch dimension at the last. And cast label to int32
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (
        tf.image.resize_with_pad(X, resize, resize) if resize else X, y)
    return (
        tf.data.Dataset.from_tensor_slices(process(*mnist_train)).batch(
            batch_size).shuffle(len(mnist_train[0])).map(resize_fn),
        tf.data.Dataset.from_tensor_slices(process(*mnist_test)).batch(
            batch_size).map(resize_fn))

class TrainCallback(tf.keras.callbacks.Callback):
    """A callback to visualize the training progress.

    Defined in :numref:`sec_utils`"""
    def __init__(self, net, train_iter, test_iter, num_epochs, device_name):
        self.timer = d2l.Timer()
        self.animator = d2l.Animator(
            xlabel='epoch', xlim=[1, num_epochs], legend=[
                'train loss', 'train acc', 'test acc'])
        self.net = net
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.num_epochs = num_epochs
        self.device_name = device_name
    def on_epoch_begin(self, epoch, logs=None):
        self.timer.start()
    def on_epoch_end(self, epoch, logs):
        self.timer.stop()
        test_acc = self.net.evaluate(
            self.test_iter, verbose=0, return_dict=True)['accuracy']
        metrics = (logs['loss'], logs['accuracy'], test_acc)
        self.animator.add(epoch + 1, metrics)
        if epoch == self.num_epochs - 1:
            batch_size = next(iter(self.train_iter))[0].shape[0]
            num_examples = batch_size * tf.data.experimental.cardinality(
                self.train_iter).numpy()
            print(f'loss {metrics[0]:.3f}, train acc {metrics[1]:.3f}, '
                  f'test acc {metrics[2]:.3f}')
            print(f'{num_examples / self.timer.avg():.1f} examples/sec on '
                  f'{str(self.device_name)}')

def train_ch6(net_fn, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""
    device_name = device._device_name
    strategy = tf.distribute.OneDeviceStrategy(device_name)
    with strategy.scope():
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        net = net_fn()
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    callback = TrainCallback(net, train_iter, test_iter, num_epochs,
                             device_name)
    net.fit(train_iter, epochs=num_epochs, verbose=0, callbacks=[callback])
    return net

def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset.

    Defined in :numref:`sec_utils`"""
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5,
                cmap=None, vmin=None, vmax=None):
    """Plot a list of images (cmap colors single-channel images, on the
    color scale from vmin to vmax if given).

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = d2l.numpy(img)
        except:
            pass
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
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
        else:  # Named ticks; names along the x-axis read vertically
            axis.set_ticks(range(n), labels,
                           rotation=90 if axis is ax.xaxis else 0)
    if cbar is True:
        fig.colorbar(pcm, ax=axes, shrink=0.6);

def linreg(X, w, b):
    """The linear regression model.

    Defined in :numref:`sec_utils`"""
    return d2l.matmul(X, w) + b

def squared_loss(y_hat, y):
    """Squared loss.

    Defined in :numref:`sec_utils`"""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2

def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset.

    Defined in :numref:`sec_utils`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

class Animator:
    """For plotting data in animation.

    Defined in :numref:`sec_utils`"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
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

def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_utils`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    elif (len(y_hat.shape) > 1 and y_hat.shape[-1] == 1) or y_hat.dtype != y.dtype:
        # Binary classification with float scores (logits or probabilities):
        # threshold at 0 (logits) to get class labels, then reshape to match y.
        y_hat = d2l.astype(y_hat > 0, y.dtype).reshape(y.shape)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

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

def download_extract(name, folder=None):
    """Download and extract a zip/tar file.

    Defined in :numref:`sec_utils`"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    target = os.path.join(base_dir, folder) if folder else data_dir
    # Skip re-extraction if a completion marker exists (extracting many small
    # files is slow and unnecessary when the archive is already unpacked).
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

def tokenize(lines, token='word'):
    """Split text lines into word or character tokens.

    Defined in :numref:`sec_utils`"""
    assert token in ('word', 'char'), 'Unknown token type: ' + token
    return [line.split() if token == 'word' else list(line) for line in lines]

def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset.

    Defined in :numref:`sec_utils`"""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]

def grad_clipping(grads, theta):
    """Clip the gradient.

    Defined in :numref:`sec_utils`"""
    theta = tf.constant(theta, dtype=tf.float32)
    new_grad = []
    for grad in grads:
        if isinstance(grad, tf.IndexedSlices):
            new_grad.append(tf.convert_to_tensor(grad))
        else:
            new_grad.append(grad)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, theta):
        for i, grad in enumerate(new_grad):
            new_grad[i] = grad * theta / norm
    else:
        new_grad = new_grad
    return new_grad

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

def read_data_nmt():
    """Load the English-French dataset.

    Defined in :numref:`sec_utils`"""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
        return f.read()

def preprocess_nmt(text):
    """Preprocess the English-French dataset.

    Defined in :numref:`sec_utils`"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset.

    Defined in :numref:`sec_utils`"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_utils`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches.

    Defined in :numref:`sec_utils`"""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset.

    Defined in :numref:`sec_utils`"""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_utils`"""
    maxlen = X.shape[1]
    mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
        None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)
    
    if len(X.shape) == 3:
        return tf.where(tf.expand_dims(mask, axis=-1), X, value)
    else:
        return tf.where(mask, X, value)

class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_utils`"""
    def __init__(self, valid_len):
        super().__init__(reduction='none')
        self.valid_len = valid_len
    
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def call(self, label, pred):
        weights = tf.ones_like(label, dtype=tf.float32)
        weights = sequence_mask(weights, self.valid_len)
        label_one_hot = tf.one_hot(label, depth=pred.shape[-1])
        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')(label_one_hot, pred)
        weighted_loss = tf.reduce_mean((unweighted_loss*weights), axis=1)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    animator = d2l.Animator(xlabel="epoch", ylabel="loss",
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x for x in batch]
            bos = tf.reshape(tf.constant([tgt_vocab['<bos>']] * Y.shape[0]),
                             shape=(-1, 1))
            dec_input = tf.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            with tf.GradientTape() as tape:
                Y_hat, _ = net(X, dec_input, X_valid_len, training=True)
                l = MaskedSoftmaxCELoss(Y_valid_len)(Y, Y_hat)
            gradients = tape.gradient(l, net.trainable_variables)
            gradients = d2l.grad_clipping(gradients, 1)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            num_tokens = tf.reduce_sum(Y_valid_len).numpy()
            metric.add(tf.reduce_sum(l), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device._device_name)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    save_attention_weights=False):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = tf.constant([len(src_tokens)])
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = tf.expand_dims(src_tokens, axis=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len, training=False)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = tf.expand_dims(tf.constant([tgt_vocab['<bos>']]), axis=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state, training=False)
        # We use the token with the highest prediction likelihood as input
        # of the decoder at the next time step
        dec_X = tf.argmax(Y, axis=2)
        pred = tf.squeeze(dec_X, axis=0)
        # Save attention weights
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred.numpy())
    return ' '.join(tgt_vocab.to_tokens(tf.reshape(output_seq, shape = -1).numpy().tolist())), attention_weight_seq

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
    trainer = d2l.HPOTrainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=batch_size)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validation_error()
    return float(validation_error)

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

d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

d2l.DATA_HUB['aclImdb'] = (d2l.DATA_URL + 'aclImdb_v1.tar.gz', 
                          '01ada507287d82875905620988597833ad4e0903')

d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')


reshape = tf.reshape
ones_like = tf.ones_like
ones = tf.ones
zeros_like = tf.zeros_like
zeros = tf.zeros
meshgrid = tf.meshgrid
sin = tf.sin
sinh = tf.sinh
cos = tf.cos
cosh = tf.cosh
tanh = tf.tanh
linspace = tf.linspace
exp = tf.exp
normal = tf.random.normal
rand = tf.random.uniform
matmul = tf.matmul
reduce_sum = tf.reduce_sum
reduce_mean = tf.reduce_mean
argmax = tf.argmax
tensor = tf.constant
arange = tf.range
astype = tf.cast
int32 = tf.int32
int64 = tf.int64
float32 = tf.float32
transpose = tf.transpose
concat = tf.concat
stack = tf.stack
abs = tf.abs
eye = tf.eye
log = tf.math.log
sigmoid = tf.sigmoid
expand_dims = tf.expand_dims
repeat = tf.repeat
batch_matmul = tf.matmul
numpy = lambda x, *args, **kwargs: x.numpy(*args, **kwargs)
size = lambda a: tf.size(a).numpy()
