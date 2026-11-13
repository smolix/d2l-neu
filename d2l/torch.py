DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

import numpy as np
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms

nn_Module = nn.Module

#################   WARNING   ################
# The below part is generated automatically through:
#    python tools/build_lib.py
# Don't edit it directly

import sys
d2l = sys.modules[__name__]

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
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from scipy.spatial import distance_matrix

def synthetic_data(w, b, noise, n):
    """n examples with standard normal features and labels Xw + b + noise.

    Defined in :numref:`sec_synthetic-regression-data`"""
    X = d2l.randn(n, len(w))
    eps = d2l.randn(n, 1) * noise
    return X, d2l.matmul(X, d2l.reshape(w, (-1, 1))) + b + eps

def add_to_class(Class):
    """Register functions as methods in created class.

    Defined in :numref:`sec_oo-design`"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        return obj
    return wrapper

class Module(nn.Module, d2l.HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
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
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        """Run a dry pass to fix lazy shapes, then apply `init`.

        Defined in :numref:`sec_lazy_init`"""
        with torch.no_grad():
            self(*inputs)
        if init is not None:
            self.apply(init)

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
        self.w = nn.Parameter(d2l.normal(0, sigma, (num_inputs, 1)))
        self.b = nn.Parameter(d2l.zeros(1))

    def forward(self, X):
        return d2l.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2
        return d2l.reduce_mean(l)

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr)

class SGD(d2l.HyperParameters):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_oo-design`"""
    def __init__(self, params, lr):
        self.save_hyperparameters()

    def step(self):
        for param in self.params:
            param -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

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
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)

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
        """`configure_optimizers` may also return a learning-rate scheduler.

        Defined in :numref:`sec_oo-design`"""
        # PyTorch steps a schedule with a separate object (sec_scheduler)
        optim = model.configure_optimizers()
        self.optim, self.scheduler = (optim if isinstance(optim, tuple)
                                      else (optim, None))

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

    def validate(self):
        self.model.eval()
        metrics = d2l.Averager()
        with torch.no_grad():
            for batch in map(self.prepare_batch, self.val_dataloader):
                metrics.add(d2l.as_metrics(self.model.validation_step(batch)),
                            len(batch[-1]))
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
            batch = [d2l.to(a, self.gpus[0]) for a in batch]
        return batch

    def prepare_model(self, model):
        self.model = model
        if self.gpus:
            model.to(self.gpus[0])
            # Pick convolution algorithms by timing, unless determinism is wanted
            torch.backends.cudnn.benchmark = not (
                torch.are_deterministic_algorithms_enabled()
                or torch.backends.cudnn.deterministic)
        # Lazy parameters get their shapes from a dry pass on the device, then
        # the model's weight_init, if it has one, before the optimizer exists
        if any(isinstance(p, nn.parameter.UninitializedParameter)
               for p in model.parameters()):
            loader = (self.train_dataloader if self.train_dataloader is not None
                      else self.val_dataloader)  # `evaluate` has no training data
            batch = self.prepare_batch(next(iter(loader)))
            model.apply_init(batch[:-1], getattr(model, 'weight_init', None))

    def clip_gradients(self, grad_clip_val, model):
        params = [p for p in model.parameters() if p.requires_grad]
        norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
        scale = torch.clamp(grad_clip_val / norm, max=1.0)  # no `if` on the host
        for param in params:
            param.grad.mul_(scale)

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
        self.net = nn.LazyLinear(1)

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

    def get_w_b(self):
        return self.net.weight.detach(), self.net.bias.detach()

class FashionMNIST(d2l.DataModule):
    """The Fashion-MNIST dataset.

    Defined in :numref:`sec_fashion_mnist`"""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)

    def text_labels(self, indices):
        """Return text labels.

        Defined in :numref:`sec_fashion_mnist`"""
        labels = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat',
                  'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
        return [labels[int(i)] for i in indices]

    def get_dataloader(self, train):
        data = self.train if train else self.val
        return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                           num_workers=self.num_workers)

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
        Y_hat = self(*batch[:-1])
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
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        X = d2l.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            name = getattr(layer, '__name__', type(layer).__name__)
            print(f'{name:<22} {tuple(X.shape)}')

def cross_entropy(y_hat, y):
    # Tiny clip to keep log finite when softmax outputs underflow to 0.
    p = y_hat[list(range(len(y_hat))), y].clamp(min=1e-12)
    return -d2l.reduce_mean(d2l.log(p))

class SoftmaxRegression(d2l.Classifier):
    """The softmax regression model.

    Defined in :numref:`sec_softmax_concise`"""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))

    def weight_init(self, module):
        # The Trainer calls this after the lazy layer has learned its shape
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

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

def save_checkpoint(path, model, optimizer, step, cfg=None, rng=None):
    """Atomically write a resumable training checkpoint.

    Defined in :numref:`sec_read_write`"""
    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step': step,
            'cpu_rng': torch.get_rng_state()}
    if cfg is not None:
        ckpt['cfg'] = asdict(cfg)
    if rng is not None:                        # a torch.Generator
        ckpt['rng'] = rng.get_state()
    if torch.cuda.is_available():
        ckpt['cuda_rng'] = torch.cuda.get_rng_state_all()
    tmp = path + '.tmp'
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

def load_checkpoint(path, model, optimizer=None, rng=None):
    """Restore the state written by save_checkpoint; return the raw dict.

    Defined in :numref:`sec_read_write`"""
    ckpt = torch.load(path, weights_only=True)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer'])
    if rng is not None:
        rng.set_state(ckpt['rng'])
    torch.set_rng_state(ckpt['cpu_rng'])
    if torch.cuda.is_available() and 'cuda_rng' in ckpt:
        torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    return ckpt

def cpu():
    """Get the CPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device('cpu')

def gpu(i=0):
    """Get a GPU device.

    Defined in :numref:`sec_use_gpu`"""
    return torch.device(f'cuda:{i}')

def num_gpus():
    """Get the number of available GPUs.

    Defined in :numref:`sec_use_gpu`"""
    return torch.cuda.device_count()

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
    `devices` lists the GPUs to wait for, by default the current one.

    Defined in :numref:`sec_use_gpu`"""
    def __init__(self, f, warmup=3, repeats=10, desc='time', devices=None):
        def wait():  # CUDA runs asynchronously; the CPU does not
            if torch.cuda.is_available():
                for device in devices or [None]:
                    torch.cuda.synchronize(device)
        self.desc = desc
        for _ in range(warmup):
            f()
        wait()
        t0 = time.perf_counter()
        for _ in range(repeats):
            f()
        wait()
        self.time = (time.perf_counter() - t0) / repeats

    def __repr__(self):
        return f'{self.desc}: {1000 * self.time:.2f} ms/call'

def peak_memory(f, *args):
    """Extra peak GPU memory allocated by f(*args), in bytes.

    Defined in :numref:`sec_use_gpu`"""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    f(*args)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - base

def corr2d(X, K):
    """Compute 2D cross-correlation.

    Defined in :numref:`sec_conv_layer`"""
    h, w = K.shape
    Y = d2l.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = d2l.reduce_sum((X[i: i + h, j: j + w] * K))
    return Y

def init_cnn(module):
    """Initialize weights for CNNs.

    Defined in :numref:`sec_lenet`"""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)

class LeNet(d2l.Classifier):
    """The layout of LeNet-5, with sigmoid activations and average pooling.

    Defined in :numref:`sec_lenet`"""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120), nn.Sigmoid(),
            nn.LazyLinear(84), nn.Sigmoid(),
            nn.LazyLinear(num_classes))

    def weight_init(self, module):
        # The Trainer calls this after the first minibatch fixed the shapes
        init_cnn(module)  # Xavier-uniform weights
        if type(module) in (nn.Linear, nn.Conv2d) and module.bias is not None:
            nn.init.zeros_(module.bias)

def cosine_warmup(t, num_steps, warmup_steps):
    """Learning-rate factor: linear warmup from 0, then a half-cosine to 0.

    Defined in :numref:`sec_alexnet`"""
    if t < warmup_steps:
        return t / warmup_steps
    progress = min((t - warmup_steps) / (num_steps - warmup_steps), 1)
    return 0.5 * (1 + math.cos(math.pi * progress))

class Residual(nn.Module):
    """The Residual block of ResNet models.

    Defined in :numref:`sec_resnet`"""
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # No conv biases: the batch normalization after each conv removes them
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   stride=strides, bias=False)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1,
                                   bias=False)
        # Auto-enable 1x1 conv when downsampling so the residual shape matches.
        if use_1x1conv or strides != 1:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides, bias=False)
            self.bn3 = nn.LazyBatchNorm2d()
        else:
            self.conv3 = None
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.bn3(self.conv3(X))
        Y += X
        return F.relu(Y)

class ResNeXtBlock(nn.Module):
    """The ResNeXt block.

    Defined in :numref:`sec_resnet`"""
    def __init__(self, num_channels, groups, bot_mul, use_1x1conv=False,
                 strides=1):
        super().__init__()
        bot_channels = int(round(num_channels * bot_mul))
        # No conv biases: the batch normalization after each conv removes them
        self.conv1 = nn.LazyConv2d(bot_channels, kernel_size=1, stride=1,
                                   bias=False)
        self.conv2 = nn.LazyConv2d(bot_channels, kernel_size=3,
                                   stride=strides, padding=1,
                                   groups=groups, bias=False)
        self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=1,
                                   bias=False)
        self.bn1 = nn.LazyBatchNorm2d()
        self.bn2 = nn.LazyBatchNorm2d()
        self.bn3 = nn.LazyBatchNorm2d()
        # Auto-enable 1x1 conv when downsampling so the residual shape matches.
        if use_1x1conv or strides != 1:
            self.conv4 = nn.LazyConv2d(num_channels, kernel_size=1,
                                       stride=strides, bias=False)
            self.bn4 = nn.LazyBatchNorm2d()
        else:
            self.conv4 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = F.relu(self.bn2(self.conv2(Y)))
        Y = self.bn3(self.conv3(Y))
        if self.conv4:
            X = self.bn4(self.conv4(X))
        return F.relu(Y + X)

def num_params(model):
    """The number of trainable parameters.

    Defined in :numref:`sec_depthwise_separable`"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mixup(X, y, alpha):
    """Mix a minibatch with a shuffled copy; return both label sets.

    Defined in :numref:`sec_training_recipes`"""
    lam = random.betavariate(alpha, alpha)
    perm = torch.randperm(X.shape[0], device=X.device)
    return lam * X + (1 - lam) * X[perm], y, y[perm], lam

def drop_path(Y, p, training):
    """Zero each example of Y with probability p, rescale the rest.

    Defined in :numref:`sec_training_recipes`"""
    if not training or p == 0:
        return Y
    shape = (Y.shape[0],) + (1,) * (Y.ndim - 1)  # One draw per example
    keep = torch.rand(shape, device=Y.device) > p
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
        self.W_xh = nn.Parameter(
            d2l.randn(num_inputs, num_hiddens) * sigma)
        self.W_hh = nn.Parameter(
            d2l.randn(num_hiddens, num_hiddens) * sigma)
        self.b_h = nn.Parameter(d2l.zeros(num_hiddens))

    def forward(self, inputs, state=None):
        if state is None:
            # Initial state with shape: (batch_size, num_hiddens)
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        outputs = []
        for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs)
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
        self.E = nn.Parameter(
            d2l.randn(self.vocab_size, self.rnn.num_inputs))
        self.W_hq = nn.Parameter(
            d2l.randn(self.rnn.num_hiddens, self.vocab_size) * self.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size))

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
        embs = F.dropout(self.embedding(X), self.emb_dropout, self.training)
        rnn_outputs, _ = self.rnn(embs, state)
        H = d2l.stack(list(rnn_outputs), 0)  # (num_steps, batch_size, num_hiddens)
        return self.output_layer(F.dropout(H, self.dropout, self.training))

    @torch.inference_mode()  # Generation needs no autograd graph
    def predict(self, prefix, num_tokens, tok, device=None, temperature=0.0):
        outputs, state = tok.encode(prefix), None
        for i in range(len(outputs) - 1):  # Warm up on the prefix
            X = d2l.tensor([[outputs[i]]], device=device)
            _, state = self.rnn(self.embedding(X), state)
        for _ in range(num_tokens):  # Generate num_tokens continuation tokens
            X = d2l.tensor([[outputs[-1]]], device=device)
            rnn_outputs, state = self.rnn(self.embedding(X), state)
            logits = self.output_layer(rnn_outputs)[0, 0]
            if temperature == 0:  # Greedy: the highest-scoring token
                token = logits.argmax()
            else:  # Sample from the softmax of logits / temperature
                token = torch.distributions.Categorical(
                    logits=logits / temperature).sample()
            outputs.append(int(token))
        return tok.decode(outputs)

class RNN(d2l.Module):
    """The RNN model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        for name, param in self.rnn.named_parameters():
            if name.startswith('weight'):
                nn.init.normal_(param, std=sigma)
            else:  # Both bias vectors start at zero
                nn.init.zeros_(param)

    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)

class RNNLM(d2l.RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""
    def init_params(self):
        self.emb = nn.Embedding(self.vocab_size, self.rnn.num_inputs)
        self.linear = nn.Linear(self.rnn.num_hiddens, self.vocab_size)
        nn.init.normal_(self.linear.weight, std=self.sigma)
        nn.init.zeros_(self.linear.bias)

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

VALLEY_CURVATURES = (0.2, 4.0)

def f_valley(x1, x2):
    """The ill-conditioned quadratic valley h1/2 x1^2 + h2/2 x2^2, with
    (h1, h2) = VALLEY_CURVATURES.

    Defined in :numref:`sec_gd`"""
    h1, h2 = VALLEY_CURVATURES
    return h1 / 2 * x1 ** 2 + h2 / 2 * x2 ** 2

def valley_grad(x1, x2):
    """Gradient of f_valley.

    Defined in :numref:`sec_gd`"""
    h1, h2 = VALLEY_CURVATURES
    return h1 * x1, h2 * x2

def train_2d(trainer, f_grad, steps=20, **hparams):
    """Run a 2D update rule from (-5, -2) and return the iterates.

    Defined in :numref:`sec_gd`"""
    x1, x2, s1, s2 = -5, -2, 0, 0  # s1, s2: state of stateful optimizers
    results = [(x1, x2)]
    for _ in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad, **hparams)
        results.append((x1, x2))
    print(f'step {steps}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def gd_2d(x1, x2, s1, s2, f_grad, eta):
    """One step of gradient descent in 2D; the state stays zero.

    Defined in :numref:`sec_gd`"""
    g1, g2 = f_grad(x1, x2)
    return x1 - eta * g1, x2 - eta * g2, 0, 0

d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

def get_airfoil(batch_size=10, n=1500):
    """Load the standardized airfoil self-noise data as minibatches.

    Defined in :numref:`sec_minibatch_sgd`"""
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    dataset = torch.utils.data.TensorDataset(
        *(t.to(d2l.try_gpu()) for t in (data[:n, :-1], data[:n, -1])))
    sampler = torch.utils.data.BatchSampler(  # Whole minibatches of indices
        torch.utils.data.RandomSampler(dataset), batch_size, drop_last=False)
    data_iter = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                            batch_size=None)
    return data_iter, data.shape[1]-1

class Timer:
    """Record multiple running times.

    Defined in :numref:`sec_minibatch_sgd`"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.perf_counter()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.perf_counter() - self.tik)
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

def record_run(step, full_loss, data_iter, num_epochs=2, plot=True, *,
               ylim=None):
    """Call step(X, y) on every minibatch; about every 200 examples, record
    full_loss() and the training time so far, and plot the loss in the range
    ylim. Return the history.

    Defined in :numref:`sec_minibatch_sgd`"""
    # Wait until the device has finished (CUDA runs asynchronously)
    wait = torch.cuda.synchronize if torch.cuda.is_available() else (
        lambda: None)
    if plot:
        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                xlim=[0, num_epochs], ylim=ylim)
    history = {'examples': [], 'loss': []}
    n, i, timer = 0, 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            step(X, y)
            n, i = n + len(X), i + 1
            if i % max(1, 200 // len(X)) == 0:  # About every 200 examples
                wait()
                timer.stop()
                with torch.no_grad():
                    l = float(full_loss())
                history['examples'].append(n)
                history['loss'].append(l)
                if plot:
                    animator.add(i / len(data_iter), (l,))
                timer.start()
    history['time'] = timer.cumsum()
    print(f'loss: {l:.3f}, {timer.sum() / num_epochs:.3f} sec/epoch')
    return history

def train_with_rule(trainer_fn, states, hyperparams, data_iter,
                    feature_dim, num_epochs=2, plot=True,
                    ylim=(0.22, 0.35)):
    """Train linear regression with an update rule; return its history. The
    default ylim is the range of the airfoil losses.

    Defined in :numref:`sec_minibatch_sgd`"""
    Xs, ys = (torch.cat(t) for t in zip(*data_iter))  # The whole data
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     device=Xs.device, requires_grad=True)
    b = torch.zeros(1, device=Xs.device, requires_grad=True)
    loss = lambda X, y: d2l.squared_loss(d2l.linreg(X, w, b), y).mean()

    def step(X, y):
        loss(X, y).backward()
        trainer_fn([w, b], states, hyperparams)

    loss(Xs, ys).backward()  # Warm-up, then clear the gradients
    w.grad, b.grad = None, None
    return d2l.record_run(step, lambda: loss(Xs, ys), data_iter, num_epochs,
                          plot, ylim=ylim)

def train_with_optimizer(trainer_fn, hyperparams, data_iter, num_epochs=2,
                         plot=True, ylim=(0.22, 0.35)):
    """Train linear regression with a torch.optim optimizer class; return
    its history. The default ylim is the range of the airfoil losses.

    Defined in :numref:`sec_minibatch_sgd`"""
    Xs, ys = (torch.cat(t) for t in zip(*data_iter))  # The whole data
    net = nn.Linear(Xs.shape[1], 1).to(Xs.device)
    nn.init.normal_(net.weight, std=0.01)
    nn.init.zeros_(net.bias)
    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = lambda X, y: d2l.squared_loss(net(X), y).mean()

    def step(X, y):
        optimizer.zero_grad()
        loss(X, y).backward()
        optimizer.step()

    loss(Xs, ys).backward()  # Warm-up; zero_grad clears it
    return d2l.record_run(step, lambda: loss(Xs, ys), data_iter, num_epochs,
                          plot, ylim=ylim)

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
        array = torch.from_numpy(corpus[idx].astype(np.int64))
        return array[:, :-1], array[:, 1:]

    def train_dataloader(self):  # A pass of minibatches at random offsets
        n = len(self.train) - self.num_steps
        for _ in range(n // (self.batch_size * self.num_steps)):
            yield self.windows(self.train,
                               torch.randint(n, (self.batch_size,)).numpy())

    def val_dataloader(self):  # Consecutive, non-overlapping windows
        starts = np.arange(0, len(self.val) - self.num_steps, self.num_steps)
        for i in range(0, len(starts), self.batch_size):
            yield self.windows(self.val, starts[i:i + self.batch_size])

class TinyLM(nn.Module):
    """A small decoder-only transformer language model.

    Defined in :numref:`sec_adam`"""
    def __init__(self, vocab_size, d_model=128, num_heads=2, num_blks=2,
                 max_len=64):
        super().__init__()
        self.num_heads = num_heads
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blks = nn.ModuleList([nn.ModuleDict(dict(
            norm1=nn.LayerNorm(d_model),
            qkv=nn.Linear(d_model, 3 * d_model),
            proj=nn.Linear(d_model, d_model),
            norm2=nn.LayerNorm(d_model),
            mlp=nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                              nn.Linear(4 * d_model, d_model))))
            for _ in range(num_blks)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def attention(self, blk, X):
        B, T, D = X.shape
        q, k, v = blk['qkv'](X).chunk(3, dim=-1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                   for u in (q, k, v))
        Y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return blk['proj'](Y.transpose(1, 2).reshape(B, T, D))

    def forward(self, X):
        H = self.token_emb(X) + self.pos_emb(torch.arange(X.shape[1],
                                                          device=X.device))
        for blk in self.blks:
            H = H + self.attention(blk, blk['norm1'](H))
            H = H + blk['mlp'](blk['norm2'](H))
        return self.head(self.norm(H))

def train_steps(model, data, optimizer, num_steps, *, stop=None):
    """Minimize the mean cross-entropy of the logits model(X) for the labels
    Y over the minibatches (X, Y) of data, leaving out the labels -100, for
    num_steps steps or until stop(step) is true; return every step's loss.

    Defined in :numref:`sec_adam`"""
    device = d2l.try_gpu()
    model.to(device)
    losses, step = [], 0
    while step < num_steps:
        for X, Y in data.train_dataloader():
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   Y.reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            step += 1
            if step >= num_steps or (stop is not None and stop(step)):
                return torch.stack(losses).tolist()

def lm_perplexity(model, batches):
    """Perplexity of a language model on the minibatches (X, Y) of batches.

    Defined in :numref:`sec_adam`"""
    device = next(model.parameters()).device
    modes = {m: m.training for m in model.modules()}  # Restored at the end
    model.eval()  # Dropout off while we measure
    total, count = 0, 0
    with torch.inference_mode():
        for X, Y in batches:
            logits = model(X.to(device))
            total += F.cross_entropy(logits.flatten(0, 1),
                                     Y.to(device).flatten(), reduction='sum')
            count += Y.numel()
    for m, training in modes.items():
        m.training = training
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

class FashionCNN(nn.Module):
    """A compact CNN for Fashion-MNIST: two conv-pool stages, two linear.

    Defined in :numref:`sec_adam`"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), 2)
        X = F.max_pool2d(F.relu(self.conv2(X)), 2)
        return self.fc2(F.relu(self.fc1(X.flatten(1))))

def evaluate_accuracy(model, batches):
    """Fraction of the examples in batches that model classifies correctly.

    Defined in :numref:`sec_adam`"""
    device = next(model.parameters()).device
    training, correct, total = model.training, 0, 0
    model.eval()  # Dropout off, batch norm on running statistics
    with torch.inference_mode():
        for X, y in batches:
            y_hat = model(X.to(device)).argmax(dim=1)
            correct += (y_hat == y.to(device)).sum()
            total += len(y)
    model.train(training)
    return float(correct) / total

def lr_sweep(make_model, make_optimizer, data, lrs, seed,
             num_steps=2000, *, evaluate=None, metric='score', every=None,
             verbose=True):
    """Train make_model() with make_optimizer(model, lr) for each lr, from
    one initialization; return the loss curves and the evaluations.

    Defined in :numref:`sec_adam`"""
    every = every or num_steps
    curves, scores = {}, {}
    for lr in lrs:
        torch.manual_seed(seed)  # The same initialization for every lr
        model = make_model()
        optimizer = make_optimizer(model, lr)
        curves[lr], scores[lr] = [], []
        for _ in range(num_steps // every):
            curves[lr] += train_steps(model, data, optimizer, every)
            if evaluate is not None:
                scores[lr].append(evaluate(model))
        line = f'lr {lr:g}: final loss {final_loss(curves[lr]):.3f}'
        if scores[lr]:
            line += f', {metric} {scores[lr][-1]:.3f}'
        if verbose:
            print(line)
    return curves, scores

def lenet_bn():
    """LeNet with ReLU, max-pooling, batch normalization, Xavier weights.

    Defined in :numref:`sec_scheduler`"""
    net = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.BatchNorm2d(6),
        nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2), nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.BatchNorm1d(120), nn.ReLU(),
        nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(),
        nn.Linear(84, 10))
    for m in net.modules():  # Xavier weights, zero biases
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net

def masked_softmax(X, mask=None):
    """Softmax over the last axis with zero weight where mask is False.

    Defined in :numref:`sec_attention-scoring-functions`"""
    if mask is None:
        return F.softmax(X, dim=-1)
    # Most negative finite value: zero weight at any precision, no NaN
    X = X.masked_fill(~mask, torch.finfo(X.dtype).min)
    weights = F.softmax(X, dim=-1)
    # A query without any valid key gets zero weights, not a uniform row
    return torch.where(mask.any(-1, keepdim=True), weights, 0.0)

def padding_mask(valid_lens, num_keys):
    """Boolean mask that is True for the first valid_lens keys.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # valid_lens of shape (batch,) gives (batch, 1, keys); per-query
    # lengths of shape (batch, queries) give (batch, queries, keys)
    if valid_lens.dim() == 1:
        valid_lens = valid_lens[:, None]
    keys = torch.arange(num_keys, device=valid_lens.device)
    return keys < valid_lens[..., None]

def causal_mask(num_queries, num_keys=None, device=None):
    """Boolean (queries, keys) mask, True where a query may see a key.

    Defined in :numref:`sec_attention-scoring-functions`"""
    # Queries are the last positions, as when new tokens attend to a
    # cache: query i sees key j if j <= i + num_keys - num_queries
    num_keys = num_queries if num_keys is None else num_keys
    i = torch.arange(num_queries, device=device)[:, None]
    j = torch.arange(num_keys, device=device)[None, :]
    return j <= i + (num_keys - num_queries)

class DotProductAttention(nn.Module):
    """Scaled dot-product attention over the last two axes.

    Defined in :numref:`sec_attention-scoring-functions`"""
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    # Leading axes (batch, and heads if present) are carried along.
    # Shape of queries: (..., no. of queries, d)
    # Shape of keys: (..., no. of key-value pairs, d)
    # Shape of values: (..., no. of key-value pairs, value dimension)
    # mask: boolean, broadcastable to (..., no. of queries, no. of keys),
    # True where a query may attend to a key
    def forward(self, queries, keys, values, mask=None):
        d = queries.shape[-1]
        scores = queries @ keys.transpose(-2, -1) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, mask)
        return self.dropout(self.attention_weights) @ values

class MultiHeadAttention(nn.Module):
    """Multi-head attention with heads as an explicit axis; num_kv_heads <
    num_heads shares each key-value head across a group of query heads.

    Defined in :numref:`sec_multihead-attention`"""
    def __init__(self, num_hiddens, num_heads, dropout=0.0, bias=False,
                 num_kv_heads=None):
        super().__init__()
        assert num_hiddens % num_heads == 0, 'heads must divide num_hiddens'
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        assert num_heads % self.num_kv_heads == 0
        kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
        self.attention = d2l.DotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, kv_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, kv_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, mask=None):
        # queries: (batch, no. of queries, num_hiddens); keys and values:
        # (batch, no. of key-value pairs, num_hiddens). mask: boolean,
        # broadcastable to (batch, no. of queries, no. of keys), shared by
        # all heads, or to (batch, num_heads, no. of queries, no. of keys)
        q = self.split_heads(self.W_q(queries), self.num_heads)
        k = self.split_heads(self.W_k(keys), self.num_kv_heads)
        v = self.split_heads(self.W_v(values), self.num_kv_heads)
        group = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        if mask is not None and mask.dim() == 3:
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
        return X.reshape(X.shape[0], X.shape[1], num_heads, -1).transpose(1, 2)

    def merge_heads(self, X):
        """Reverse split_heads: (batch, num_heads, length, width) -> (batch,
        length, num_heads * width).

        Defined in :numref:`sec_multihead-attention`"""
        return X.transpose(1, 2).reshape(X.shape[0], X.shape[2], -1)

def sinusoidal_encoding(max_len, num_hiddens):
    """The (max_len, num_hiddens) table of sinusoidal position codes.

    Defined in :numref:`sec_positional-information`"""
    theta = torch.arange(max_len)[:, None] / 10000 ** (
        torch.arange(0, num_hiddens, 2) / num_hiddens)
    return torch.stack([torch.sin(theta), torch.cos(theta)],
                       -1).reshape(max_len, num_hiddens)

def rope_tables(n, d, offset=0, base=10000.0, scale=None,
                device=None):
    """Cosines and sines, each (n, d/2), of the angles of the positions
    offset, ..., offset + n - 1.

    Defined in :numref:`sec_positional-information`"""
    pos = torch.arange(n, dtype=torch.float32, device=device) + offset
    omega = base ** (-torch.arange(0, d, 2, device=device) / d)
    if scale is not None:  # One angle factor per pair, shape (d/2,)
        omega = omega * scale
    theta = pos[:, None] * omega[None, :]
    return torch.cos(theta), torch.sin(theta)

def rope(x, offset=0, base=10000.0, scale=None, tables=None):
    """Rotate feature pairs of x (..., num_steps, d) by position angles;
    tables from rope_tables take the place of offset, base, and scale.

    Defined in :numref:`sec_positional-information`"""
    if tables is None:
        tables = rope_tables(*x.shape[-2:], offset, base, scale, x.device)
    # Angles in fp32, tables in the dtype of x: bf16 queries stay bf16
    cos, sin = (u.to(x.dtype) for u in tables)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], -1).flatten(-2)

class TinyCharLM(nn.Module):
    """Attention-only language model.

    Defined in :numref:`sec_positional-information`"""
    def __init__(self, vocab_size, num_hiddens=128, num_heads=4, num_blks=2,
                 pos='rope', max_len=512, bias=False):
        super().__init__()
        assert pos in ('learned', 'sinusoidal', 'rope', 'alibi', 'none')
        self.num_heads, self.pos = num_heads, pos
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pos == 'learned':
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        if pos == 'sinusoidal':
            self.register_buffer('P', sinusoidal_encoding(max_len,
                                                          num_hiddens))
        if pos == 'rope':  # Base and per-pair angle factors of the rotation
            self.rope_base = 10000.0
            self.register_buffer('rope_scale',
                                 torch.ones(num_hiddens // num_heads // 2))
        self.blks = nn.ModuleList([nn.ModuleDict(dict(
            qkv=nn.Linear(num_hiddens, 3 * num_hiddens, bias=bias),
            proj=nn.Linear(num_hiddens, num_hiddens, bias=bias)))
            for _ in range(num_blks)])

    def stream(self, X, num_blks=None):
        """Residual stream after the first num_blks blocks (default: all),
        and the attention weights of those blocks."""
        H, maps = self.embed(X), []
        for blk in self.blks[:num_blks]:
            out, weights = self.attend(blk, H)
            H, maps = H + out, maps + [weights]
        return H, maps

    def forward(self, X):
        return F.linear(self.stream(X)[0], self.token_emb.weight)  # Tied

    def attention_weights(self, X):
        """Per-block attention maps, each (batch, num_heads, T, T)."""
        return self.stream(X)[1]

    def embed(self, X):
        """Token embeddings plus the absolute position code, if any.

        Defined in :numref:`sec_positional-information`"""
        H = self.token_emb(X)
        if self.pos == 'learned':
            H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
        if self.pos == 'sinusoidal':  # Scale tokens up to the table
            H = H * math.sqrt(H.shape[-1]) + self.P[:X.shape[1]]
        return H

    def alibi(self, T, device):
        """ALiBi bias -m_h (i - j) of every head, (num_heads, T, T).

        Defined in :numref:`sec_positional-information`"""
        h = torch.arange(1, self.num_heads + 1, device=device)
        slopes = 2.0 ** (-8.0 * h / self.num_heads)
        pos = torch.arange(T, device=device, dtype=torch.float32)
        return slopes[:, None, None] * (pos[None, :] - pos[:, None])

    def attend(self, blk, H):
        """Output of attention block blk on the stream H, and its weights.

        Defined in :numref:`sec_positional-information`"""
        B, T, D = H.shape
        q, k, v = blk['qkv'](H).chunk(3, -1)
        q, k, v = (u.reshape(B, T, self.num_heads, -1).transpose(1, 2)
                   for u in (q, k, v))
        if self.pos == 'rope':
            q = rope(q, base=self.rope_base, scale=self.rope_scale)
            k = rope(k, base=self.rope_base, scale=self.rope_scale)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        if self.pos == 'alibi':
            scores = scores + self.alibi(T, H.device)
        weights = d2l.masked_softmax(scores, d2l.causal_mask(T, device=H.device))
        out = (weights @ v).transpose(1, 2).reshape(B, T, D)
        return blk['proj'](out), weights

class FeedForward(nn.Module):
    """Position-wise FFN: GELU MLP or SwiGLU at matched parameter count.

    Defined in :numref:`sec_transformer-block`"""
    def __init__(self, num_hiddens, act='swiglu', bias=False):
        super().__init__()
        assert act in ('swiglu', 'gelu'), f'unknown act: {act!r}'
        self.act = act
        if act == 'gelu':
            width = 4 * num_hiddens
        else:  # 'swiglu': three matrices; width 8d/3 matches the MLP budget
            width = round(8 * num_hiddens / 3)
            self.W_g = nn.Linear(num_hiddens, width, bias=bias)
        self.W_1 = nn.Linear(num_hiddens, width, bias=bias)
        self.W_2 = nn.Linear(width, num_hiddens, bias=bias)

    def forward(self, X):
        if self.act == 'gelu':  # The tanh approximation, as in GPT-2
            return self.W_2(F.gelu(self.W_1(X), approximate='tanh'))
        return self.W_2(F.silu(self.W_g(X)) * self.W_1(X))

class TransformerBlock(nn.Module):
    """Configurable transformer block: attention + FFN on a residual
    stream.

    Defined in :numref:`sec_transformer-block`"""
    def __init__(self, num_hiddens, num_heads, *, dropout=0, norm='rms',
                 act='swiglu', pre_norm=True, bias=False, attn_factory=None,
                 ffn_factory=None):
        super().__init__()
        assert norm in ('rms', 'layer'), f'unknown norm: {norm!r}'
        assert num_hiddens % num_heads == 0
        self.pre_norm = pre_norm
        make_norm = nn.RMSNorm if norm == 'rms' else nn.LayerNorm
        self.norm1 = make_norm(num_hiddens, eps=1e-5)
        self.norm2 = make_norm(num_hiddens, eps=1e-5)
        self.attention = (d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout, bias=bias)
                          if attn_factory is None else attn_factory())
        self.ffn = (FeedForward(num_hiddens, act, bias=bias)
                    if ffn_factory is None else ffn_factory())
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None, **attn_args):
        # attn_args go on to the attention, e.g. the RoPE tables of d2l.GPT
        if self.pre_norm:
            Y = self.norm1(X)
            X = X + self.dropout(self.attention(Y, Y, Y, mask, **attn_args))
            return X + self.dropout(self.ffn(self.norm2(X)))
        X = self.norm1(X + self.dropout(self.attention(X, X, X, mask,
                                                       **attn_args)))
        return self.norm2(X + self.dropout(self.ffn(X)))

class CausalAttention(nn.Module):
    """Fused causal self-attention, optionally rotary; num_kv_heads <
    num_heads shares each key-value head across a group of queries.

    Defined in :numref:`sec_gpt`"""
    def __init__(self, num_hiddens, num_heads, *, bias=False, rope=False,
                 num_kv_heads=None):
        super().__init__()
        self.num_heads, self.rope = num_heads, rope
        self.num_kv_heads = num_kv_heads or num_heads
        self.kv_hiddens = self.num_kv_heads * (num_hiddens // num_heads)
        self.W_qkv = nn.Linear(num_hiddens, num_hiddens + 2 * self.kv_hiddens,
                               bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def qkv(self, X, offset=0, tables=None):
        """Queries, keys, and values of X, each (batch, heads, length, head
        width), at positions offset, offset + 1, ...; tables, if given, are
        the d2l.rope_tables of these positions."""
        B, T, D = X.shape
        q, k, v = self.W_qkv(X).split([D, self.kv_hiddens, self.kv_hiddens],
                                      -1)
        q = q.reshape(B, T, self.num_heads, -1).transpose(1, 2)
        k, v = (u.reshape(B, T, self.num_kv_heads, -1).transpose(1, 2)
                for u in (k, v))
        if self.rope:
            q, k = (d2l.rope(u, offset, tables=tables) for u in (q, k))
        return q, k, v

    def forward(self, X, keys=None, values=None, mask=None, tables=None):
        # Self-attention: X also serves as the keys and the values, and the
        # mask is causal, so a block's mask argument must stay None
        if mask is not None:
            raise ValueError('CausalAttention applies its own mask')
        q, k, v = self.qkv(X, tables=tables)
        Y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            enable_gqa=self.num_kv_heads < self.num_heads)
        return self.W_o(Y.transpose(1, 2).flatten(2))  # (batch, length, d)

class GPT(nn.Module):
    """Decoder-only transformer language model built from configurable
    blocks.

    Defined in :numref:`sec_gpt`"""
    CausalAttention = CausalAttention  # A subclass may replace it

    def __init__(self, vocab_size, *, num_hiddens=256, num_heads=8,
                 num_blks=6, max_len=1024, pos='rope', norm='rms',
                 act='swiglu', pre_norm=True, bias=False, dropout=0,
                 num_kv_heads=None, ffn_factory=None):
        super().__init__()
        assert pos in ('rope', 'learned'), f'unknown pos: {pos!r}'
        self.pos, self.max_len, self.num_hiddens = pos, max_len, num_hiddens
        self.head_width = num_hiddens // num_heads
        self.token_emb = nn.Embedding(vocab_size, num_hiddens)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        if pos == 'learned':
            self.pos_emb = nn.Embedding(max_len, num_hiddens)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        attn = lambda: self.CausalAttention(
            num_hiddens, num_heads, bias=bias, rope=pos == 'rope',
            num_kv_heads=num_kv_heads)
        self.blks = nn.ModuleList([
            d2l.TransformerBlock(num_hiddens, num_heads, dropout=dropout,
                                 norm=norm, act=act, pre_norm=pre_norm,
                                 bias=bias, attn_factory=attn,
                                 ffn_factory=ffn_factory)
            for _ in range(num_blks)])
        self.norm = (nn.RMSNorm if norm == 'rms'
                     else nn.LayerNorm)(num_hiddens, eps=1e-5)
        self.init_weights()

    def forward(self, X):                     # X: (batch, T) token ids
        H, tables = self.token_emb(X), None   # (batch, T, d)
        if self.pos == 'learned':
            H = H + self.pos_emb(torch.arange(X.shape[1], device=X.device))
        else:  # RoPE angles of positions 0, ..., T - 1, for every block
            tables = d2l.rope_tables(X.shape[1], self.head_width,
                                     device=X.device)
        for blk in self.blks:                 # L blocks, causal attention
            H = blk(H, tables=tables)
        return F.linear(self.norm(H), self.token_emb.weight)  # (batch, T, V)

    def init_weights(self):
        """GPT-2's initialization of the linear layers: N(0, 0.02^2) weights and
        zero biases; W_o and W_2, which write to the stream, get std
        0.02 / sqrt(2L).

        Defined in :numref:`sec_gpt`"""
        L = len(self.blks)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                std = 0.02 / (2 * L) ** 0.5 if name.endswith(
                    ('W_o', 'W_2')) else 0.02
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.inference_mode()
    def generate(self, prefix, num_tokens, temperature=1.0, top_k=None):
        """Sample a continuation of the token-id list prefix.

        Defined in :numref:`sec_gpt`"""
        ids = torch.tensor(prefix, device=next(self.parameters()).device)
        for _ in range(num_tokens):  # The tokens stay on the device
            logits = self(ids[None, -self.max_len:])[0, -1] / temperature
            if top_k is not None:
                cutoff = torch.topk(logits, top_k).values[-1]
                logits[logits < cutoff] = -torch.inf
            ids = torch.cat((ids, torch.multinomial(F.softmax(logits, -1), 1)))
        return ids.tolist()

class LMOptimizer:
    """AdamW with weight decay on the weight matrices only, a linear warmup
    and a cosine decay of the learning rate, and gradient-norm clipping.

    Defined in :numref:`sec_gpt`"""
    def __init__(self, model, lr, num_steps, warmup, weight_decay=0.1,
                 max_norm=1.0):
        named = list(model.named_parameters())
        # Embedding tables are named *emb* (token_emb, pos_emb) by convention
        decay = [p for n, p in named if p.ndim == 2 and 'emb' not in n]
        rest = [p for n, p in named if p.ndim != 2 or 'emb' in n]
        self.optimizer = torch.optim.AdamW(
            [{'params': decay, 'weight_decay': weight_decay},
             {'params': rest, 'weight_decay': 0.0}], lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda t: d2l.cosine_warmup(t, num_steps, warmup))
        self.params, self.max_norm = decay + rest, max_norm

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        nn.utils.clip_grad_norm_(self.params, self.max_norm)
        self.optimizer.step()
        self.scheduler.step()

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
    from safetensors.torch import load_file  # lazy, like tiktoken above
    gpt2 = GPT(vocab_size=50257, num_hiddens=768, num_heads=12, num_blks=12,
               max_len=1024, pos='learned', norm='layer', act='gelu',
               pre_norm=True, bias=True)
    weights = load_file(d2l.download('gpt2-weights', '../data/gpt2'))
    with torch.no_grad():
        gpt2.token_emb.weight.copy_(weights['wte.weight'])
        gpt2.pos_emb.weight.copy_(weights['wpe.weight'])
        gpt2.norm.weight.copy_(weights['ln_f.weight'])
        gpt2.norm.bias.copy_(weights['ln_f.bias'])
        for i, blk in enumerate(gpt2.blks):
            modules = {f'h.{i}.ln_1': blk.norm1, f'h.{i}.ln_2': blk.norm2,
                       f'h.{i}.attn.c_attn': blk.attention.W_qkv,
                       f'h.{i}.attn.c_proj': blk.attention.W_o,
                       f'h.{i}.mlp.c_fc': blk.ffn.W_1,
                       f'h.{i}.mlp.c_proj': blk.ffn.W_2}
            for key, module in modules.items():
                W = weights[key + '.weight']  # Conv1D: (in, out)
                module.weight.copy_(W.T if W.ndim == 2 else W)
                module.bias.copy_(weights[key + '.bias'])
    return gpt2.eval()

def kv_cache_bytes(model, num_tokens, batch_size=1, bytes_per_elem=4):
    """Bytes of the keys and values of num_tokens positions in every block
    of a GPT, for batch_size sequences.

    Defined in :numref:`sec_kv-cache`"""
    attn = model.blks[0].attention
    return (2 * len(model.blks) * attn.kv_hiddens * num_tokens * batch_size
            * bytes_per_elem)

class PatchEmbedding(nn.Module):
    """Image-to-sequence stem of the vision transformer.

    Defined in :numref:`sec_vision-transformer`"""
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512,
                 num_channels=3):
        super().__init__()
        # A partial trailing patch would silently shrink the token grid
        assert img_size % patch_size == 0, 'patches must tile the image'
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.Conv2d(num_channels, num_hiddens,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, X):  # (batch, c, h, w) -> (batch, patches, d)
        return self.conv(X).flatten(2).transpose(1, 2)

class ViTBlock(d2l.TransformerBlock):
    """Pre-norm block with LayerNorm and an exact-GELU MLP.

    Defined in :numref:`sec_vision-transformer`"""
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__(
            num_hiddens, num_heads, dropout=dropout, norm='layer',
            bias=use_bias, ffn_factory=lambda: nn.Sequential(
                nn.Linear(num_hiddens, mlp_num_hiddens),
                nn.GELU(),  # Exact GELU
                nn.Dropout(dropout),
                nn.Linear(mlp_num_hiddens, num_hiddens)))

class ViT(d2l.Classifier):
    """Vision transformer.

    Defined in :numref:`sec_vision-transformer`"""
    def __init__(self, *, img_size, patch_size, num_hiddens, mlp_num_hiddens,
                 num_heads, num_blks, num_steps, emb_dropout=0.0,
                 blk_dropout=0.0, lr=1e-3, use_bias=False, num_classes=10,
                 num_channels=1):
        super().__init__()
        self.save_hyperparameters()
        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens, num_channels)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, num_hiddens))
        num_tokens = self.patch_embedding.num_patches + 1  # Add the cls token
        # Positional embeddings are learnable, initialized to small noise
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_tokens, num_hiddens) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.ModuleList([
            ViTBlock(num_hiddens, mlp_num_hiddens, num_heads, blk_dropout,
                     use_bias=use_bias) for _ in range(num_blks)])
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens, eps=1e-5),
                                  nn.Linear(num_hiddens, num_classes))

    def forward(self, X):
        X = self.patch_embedding(X)  # (batch, patches, d)
        cls = self.cls_token.expand(X.shape[0], -1, -1)
        X = self.dropout(torch.cat((cls, X), 1) + self.pos_embedding)
        for blk in self.blks:
            X = blk(X)
        return self.head(X[:, 0])

    def configure_optimizers(self):
        # AdamW decays weight matrices and kernels; it exempts biases, norm
        # parameters, the class token, and the position embeddings
        named = list(self.named_parameters())
        decay = lambda n, p: p.ndim >= 2 and n not in ('cls_token',
                                                        'pos_embedding')
        optim = torch.optim.AdamW(
            [{'params': [p for n, p in named if decay(n, p)]},
             {'params': [p for n, p in named if not decay(n, p)],
              'weight_decay': 0.0}], lr=self.lr, weight_decay=0.05)
        # Linear warmup over the first tenth of the steps, then cosine
        return optim, torch.optim.lr_scheduler.LambdaLR(optim, lambda t: (
            d2l.cosine_warmup(t, self.num_steps, self.num_steps // 10)))

def delayed_recall(rng, batch_size, num_steps, num_symbols=8,
                   blank_fillers=False):
    """A symbol, then num_steps - 1 fillers; the target is the symbol.

    Defined in :numref:`sec_gru`"""
    Y = rng.integers(0, num_symbols, (batch_size, 1))  # One query, at the end
    X = rng.integers(num_symbols, 2 * num_symbols, (batch_size, num_steps))
    if blank_fillers:  # The same filler token at every step
        X[:] = num_symbols
    X[:, :1] = Y
    return X, Y

def init_lecun(module):
    """LeCun normal weights and zero biases for all linear, convolutional,
    and recurrent layers.

    Defined in :numref:`sec_gru`"""
    for layer in module.modules():
        if isinstance(layer, (nn.Linear, nn.Conv1d)):
            fan_in = layer.weight[0].numel()  # Inputs of one output unit
            nn.init.normal_(layer.weight, std=fan_in ** -0.5)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.RNNBase):  # Input and recurrent matrices
            for name, param in layer.named_parameters():
                if name.startswith('weight'):
                    nn.init.normal_(param, std=param.shape[1] ** -0.5)
                else:
                    nn.init.zeros_(param)
    return module

class Recurrent(nn.Module):
    """A batch-first recurrent layer that returns only its outputs.

    Defined in :numref:`sec_gru`"""
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, X):
        return self.rnn(X)[0]

class RecallModel(nn.Module):
    """Embedding, a sequence layer, and a classifier at every position.

    Defined in :numref:`sec_gru`"""
    def __init__(self, layer, num_hiddens, vocab_size=16, num_inputs=32,
                 num_outputs=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, num_inputs)
        nn.init.normal_(self.embedding.weight, std=num_inputs ** -0.5)
        self.layer = layer
        self.head = init_lecun(nn.Linear(num_hiddens, num_outputs))

    def forward(self, X):  # Logits, (batch_size, num_steps, num_outputs)
        return self.head(self.layer(self.embedding(X)))

def train_queries(model, batch, num_steps, optimizer,
                  scheduler=None):
    """Train on fresh batches; the queries are the last Y.shape[1] steps.

    Defined in :numref:`sec_gru`"""
    device = next(model.parameters()).device
    for step in range(num_steps):
        X, Y = (torch.as_tensor(A, device=device) for A in batch(step))
        logits = model(X)[:, -Y.shape[1]:]  # Predictions at the queries
        loss = F.cross_entropy(logits.flatten(0, 1), Y.flatten())
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

def query_accuracy(model, X, Y):
    """Fraction of queries answered correctly.

    Defined in :numref:`sec_gru`"""
    device = next(model.parameters()).device
    X, Y = (torch.as_tensor(A, device=device) for A in (X, Y))
    with torch.inference_mode():
        return (model(X)[:, -Y.shape[1]:].argmax(-1) == Y).float().mean()

def train_delayed_recall(model, num_steps, blank_fillers=False,
                         num_updates=1500, lr=3e-3):
    """Train on fresh batches of delayed recall; return the test accuracy.

    Defined in :numref:`sec_gru`"""
    task = lambda rng, n: delayed_recall(rng, n, num_steps,
                                         blank_fillers=blank_fillers)
    rng = np.random.default_rng(0)
    model.to(d2l.try_gpu())
    train_queries(model, lambda step: task(rng, 128), num_updates,
                  torch.optim.Adam(model.parameters(), lr=lr))
    return float(query_accuracy(model, *task(np.random.default_rng(1), 2048)))

class SeqLM(nn.Module):
    """A language model of residual blocks: a sequence mixer, then an MLP.

    Defined in :numref:`sec_gru`"""
    def __init__(self, vocab_size, num_hiddens, mixers):
        super().__init__()
        d = num_hiddens
        self.embedding = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embedding.weight, std=d ** -0.5)  # Unit length
        self.mixers = nn.ModuleList(mixers)
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in mixers])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
            nn.Linear(4 * d, d)) for _ in mixers])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size)
        for module in (self.mlps, self.head):  # Mixers initialize themselves
            init_lecun(module)

    def forward(self, X):
        H = self.embedding(X)  # (batch_size, num_steps, num_hiddens)
        for norm, mixer, mlp in zip(self.norms, self.mixers, self.mlps):
            H = H + mixer(norm(H))
            H = H + mlp(H)
        return self.head(self.norm(H))  # (batch_size, num_steps, vocab_size)

    def first_mixer_input(self, X):
        """What the first mixer reads: the normalized token embeddings.

        Defined in :numref:`sec_ssm`"""
        return self.norms[0](self.embedding(X))

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=0.01)
    d2l.train_steps(model, data, optimizer, num_steps)
    return d2l.lm_perplexity(model, data.val_dataloader())

def associative_scan(combine, elems):
    """All prefixes of elems under an associative combine; time is axis 1.

    Defined in :numref:`sec_ssm`"""
    s = 1
    while s < elems[0].shape[1]:  # Combine every position with the one s back
        earlier = tuple(x[:, :-s] for x in elems)
        later = tuple(x[:, s:] for x in elems)
        elems = tuple(torch.cat([x[:, :s], y], 1)
                      for x, y in zip(elems, combine(earlier, later)))
        s *= 2
    return elems

def linear_scan(a, b, h0=None):
    """All states of h_t = a_t * h_{t-1} + b_t; time is the second axis.

    Defined in :numref:`sec_ssm`"""
    a = a.expand_as(b)  # Decays may omit axes on which they do not vary
    if h0 is not None:  # Fold the initial state into the first step
        b = torch.cat([a[:, :1] * h0.unsqueeze(1) + b[:, :1], b[:, 1:]], 1)
    def combine(earlier, later):
        (a1, b1), (a2, b2) = earlier, later
        return a1 * a2, a2 * b1 + b2
    return associative_scan(combine, (a, b))[1]

class MinGRU(nn.Module):
    """The minimal GRU: its gate and candidate read only the input.

    Defined in :numref:`sec_ssm`"""
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.W_z = nn.Linear(num_inputs, num_hiddens)  # Update gate (decay)
        self.W_h = nn.Linear(num_inputs, num_hiddens)  # Candidate state
        d2l.init_lecun(self)

    def coefficients(self, X):
        """Decays a_t and writes b_t of the linear recurrence."""
        Z = torch.sigmoid(self.W_z(X))
        return Z, (1 - Z) * torch.tanh(self.W_h(X))

    def forward(self, X):  # X: (batch_size, num_steps, num_inputs)
        return linear_scan(*self.coefficients(X))

class S4D(nn.Module):
    """A diagonal state space layer: N states per channel, fixed dynamics.

    Defined in :numref:`sec_ssm`"""
    def __init__(self, num_hiddens, num_states=16, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        d, N = num_hiddens, num_states
        # Rates lambda_n = -(n + 1), stored as log(-lambda) to stay negative
        self.log_neg_rate = nn.Parameter(
            torch.log(torch.arange(1., N + 1)).repeat(d, 1))
        self.log_dt = nn.Parameter(  # Log-uniform in [dt_min, dt_max]
            torch.rand(d, 1) * math.log(dt_max / dt_min) + math.log(dt_min))
        self.C = nn.Parameter(torch.randn(d, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.ones(d))

    def discretize(self):
        """Zero-order hold with B = 1: decays a_bar and input weights b_bar."""
        rate = -torch.exp(self.log_neg_rate)  # (d, N)
        dt_rate = torch.exp(self.log_dt) * rate
        return torch.exp(dt_rate), torch.expm1(dt_rate) / rate

    def forward(self, X):  # X: (batch_size, num_steps, d)
        T = X.shape[1]
        a_bar, b_bar = self.discretize()
        powers = a_bar.unsqueeze(-1) ** torch.arange(T, device=X.device)
        K = ((self.C * b_bar).unsqueeze(-1) * powers).sum(1)  # Kernel, (d, T)
        Y = torch.fft.irfft(torch.fft.rfft(X, n=2 * T, dim=1) *
                            torch.fft.rfft(K.T, n=2 * T, dim=0), n=2 * T, dim=1)
        return Y[:, :T] + self.D * X

    def step(self, x, h):
        """One step: x is (batch_size, d); the state h is (batch_size, d, N).

        Defined in :numref:`sec_ssm`"""
        a_bar, b_bar = self.discretize()
        h = a_bar * h + b_bar * x.unsqueeze(-1)
        return (h * self.C).sum(-1) + self.D * x, h

def init_timescales(layer, dt_min=1e-3, dt_max=1e-1):
    """Start a minGRU's gates at the decays exp(-dt) of S4D.

    Defined in :numref:`sec_ssm`"""
    dt = dt_min * (dt_max / dt_min) ** torch.rand(layer.W_z.bias.shape)
    with torch.no_grad():  # sigmoid(b) = exp(-dt)
        layer.W_z.bias.copy_(-torch.log(torch.expm1(dt)))
    return layer

class AdjointScan(torch.autograd.Function):
    """linear_scan whose backward pass is a second scan, run backward.

    Defined in :numref:`sec_mamba`"""
    @staticmethod
    def forward(ctx, a, b):
        h = d2l.linear_scan(a, b)
        ctx.save_for_backward(a, h)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        a, h = ctx.saved_tensors
        # g_t = grad_h_t + a_{t+1} g_{t+1}: a linear recurrence from the end
        a_next = torch.cat([a[:, 1:], torch.ones_like(a[:, :1])], 1)
        g = d2l.linear_scan(a_next.flip(1), grad_h.flip(1)).flip(1)
        h_prev = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)
        return g * h_prev, g

def adjoint_scan(a, b):
    """All states of h_t = a_t * h_{t-1} + b_t, differentiated by a scan.

    Defined in :numref:`sec_mamba`"""
    return AdjointScan.apply(a.expand_as(b), b)

class SelectiveSSM(nn.Module):
    """A diagonal SSM whose step sizes and write and read vectors depend on
    the input: N states for each of num_channels channels.

    Defined in :numref:`sec_mamba`"""
    def __init__(self, num_channels, num_states=16, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        H, N, R = num_channels, num_states, math.ceil(num_channels / 16)
        self.log_neg_rate = nn.Parameter(  # Rates -1, ..., -N in every channel
            torch.log(torch.arange(1., N + 1)).repeat(H, 1))
        self.W_dt = nn.Sequential(  # Step size per channel, through rank R
            nn.Linear(H, R, bias=False), nn.Linear(R, H))
        self.W_B = nn.Linear(H, N, bias=False)  # Write vector
        self.W_C = nn.Linear(H, N, bias=False)  # Read vector
        d2l.init_lecun(self)
        dt = dt_min * (dt_max / dt_min) ** torch.rand(H)
        with torch.no_grad():  # Small up-projection, softplus(bias) = dt
            nn.init.uniform_(self.W_dt[1].weight, -R ** -0.5, R ** -0.5)
            self.W_dt[1].bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(H))

    def log_coefficients(self, U):
        """Log-decays and writes, each of shape (batch_size, T, H, N).

        Defined in :numref:`sec_mamba`"""
        dt = F.softplus(self.W_dt(U))  # Step sizes, (batch_size, T, H)
        log_a = dt.unsqueeze(-1) * -torch.exp(self.log_neg_rate)
        return log_a, (dt * U).unsqueeze(-1) * self.W_B(U).unsqueeze(-2)

    def forward(self, U):  # U: (batch_size, num_steps, num_channels)
        log_a, b = self.log_coefficients(U)
        H = adjoint_scan(torch.exp(log_a), b)  # States, (batch_size, T, H, N)
        return (H * self.W_C(U).unsqueeze(-2)).sum(-1) + self.D * U

class Mamba(nn.Module):
    """The Mamba layer: a gated selective SSM between two projections.

    Defined in :numref:`sec_mamba`"""
    def __init__(self, num_hiddens, num_states=16, expand=2, conv_width=4,
                 **ssm_args):
        super().__init__()
        H = expand * num_hiddens
        self.W_in = nn.Linear(num_hiddens, 2 * H, bias=False)
        self.conv = nn.Conv1d(H, H, conv_width, groups=H,  # One per channel
                              padding=conv_width - 1)
        self.W_out = nn.Linear(H, num_hiddens, bias=False)
        d2l.init_lecun(self)
        self.ssm = SelectiveSSM(H, num_states, **ssm_args)  # Has its own init

    def ssm_input(self, X):
        """The input of the SSM and the gate branch."""
        U, Z = self.W_in(X).chunk(2, -1)
        U = self.conv(U.transpose(1, 2))[..., :X.shape[1]]  # Keep it causal
        return F.silu(U.transpose(1, 2)), Z

    def forward(self, X):  # X: (batch_size, num_steps, num_hiddens)
        U, Z = self.ssm_input(X)
        return self.W_out(self.ssm(U) * F.silu(Z))

def decay_mask(log_a):
    """L[t, s] = a_{s+1} * ... * a_t for s <= t, and 0 for s > t.

    Defined in :numref:`sec_matrix-state`"""
    cum = log_a.cumsum(-1)
    log_L = cum[..., :, None] - cum[..., None, :]  # Sum of log a over (s, t]
    causal = torch.ones(log_L.shape[-2:], dtype=torch.bool,
                        device=log_a.device).tril()
    return torch.exp(log_L.masked_fill(~causal, -torch.inf))

def chunked_linear_attention(Q, K, V, log_a, chunk_size):
    """y_t = S_t q_t with S_t = S_{t-1} diag(a_t) + v_t k_t^T, in chunks.

    Q, K: (batch_size, T, d_k); V: (batch_size, T, d_v); log_a: (batch_size,
    T, d_k) for a decay per key coordinate, or (batch_size, T) for one per step

    Defined in :numref:`sec_matrix-state`"""
    B, T, m = Q.shape[0], Q.shape[1], chunk_size
    if log_a.dim() == 2:  # The same decay for every key coordinate
        log_a = log_a.unsqueeze(-1)
    Q, K, V, log_a = (X.reshape(B, T // m, m, -1) for X in (Q, K, V, log_a))
    # 1. Within chunks: the dual form, with one mask per key coordinate
    L = decay_mask(log_a.mT)  # (B, T / m, 1 or d_k, m, m), entries <= 1
    if L.shape[2] == 1:  # One mask for all coordinates: a masked Q K^T
        Y = (L[:, :, 0] * (Q @ K.mT)) @ V
    else:
        Y = (L * Q.mT.unsqueeze(-1) * K.mT.unsqueeze(-2)).sum(2) @ V
    # 2. Across chunks: the writes of each chunk, decayed to its end, and
    #    a scan over chunks for the state at every chunk's end
    cum = log_a.cumsum(2)  # Log decay from the chunk's start to each step
    writes = V.mT @ (K * torch.exp(cum[:, :, -1:] - cum))
    S = d2l.linear_scan(torch.exp(cum[:, :, -1:]), writes)
    S = torch.cat([torch.zeros_like(S[:, :1]), S[:, :-1]], 1)  # At the start
    # 3. Every step reads the state at its chunk's start, decayed to it
    return (Y + (Q * torch.exp(cum)) @ S.mT).reshape(B, T, -1)

class GatedLinearAttention(nn.Module):
    """Multi-head linear attention with a gated decay per key coordinate.

    Defined in :numref:`sec_matrix-state`"""
    gate_scale = 16  # log a = log sigmoid(z) / 16 keeps decays near one

    def __init__(self, num_hiddens, num_heads, chunk_size=16, conv_width=4,
                 timescales=(4, 256)):
        super().__init__()
        d = num_hiddens
        self.num_heads, self.chunk_size = num_heads, chunk_size
        self.W_qkv = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(3 * d, 3 * d, conv_width, groups=3 * d,
                              padding=conv_width - 1)  # One filter per channel
        self.W_a = self.decay_gate(d)
        self.W_g = nn.Linear(d, d)  # Output gate
        self.W_o = nn.Linear(d, d, bias=False)
        d2l.init_lecun(self)
        self.init_decays(*timescales)  # Memories of 4 to 256 steps by default

    def decay_gate(self, d, rank=16):
        """Low-rank projection to one gate per key coordinate."""
        return nn.Sequential(nn.Linear(d, rank, bias=False), nn.Linear(rank, d))

    def init_decays(self, tau_min, tau_max):
        """Start the gates at decays exp(-1/tau), tau log-uniform in range."""
        b = self.W_a[-1].bias
        tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), len(b))
        with torch.no_grad():  # log sigmoid(b) / gate_scale = -1 / tau
            b.copy_(-torch.log(torch.expm1(self.gate_scale / tau)))

    def decays(self, X):
        """Log decays, (batch_size * num_heads, T, d_k).

        Defined in :numref:`sec_matrix-state`"""
        return heads_to_batch(F.logsigmoid(self.W_a(X)) / self.gate_scale,
                              self.num_heads)

    def memory(self, X, Q, K, V, log_a):
        """Outputs of the decayed matrix memory of every head. X is unused here;
        the delta rule of the next section reads its write strengths from it.

        Defined in :numref:`sec_matrix-state`"""
        return chunked_linear_attention(Q, K, V, log_a, self.chunk_size)

    def forward(self, X):  # X: (batch_size, num_steps, num_hiddens)
        B, T = X.shape[:2]
        QKV = self.conv(self.W_qkv(X).mT)[..., :T].mT  # Causal: drop the tail
        Q, K, V = (heads_to_batch(Z, self.num_heads)
                   for Z in F.silu(QKV).chunk(3, -1))
        Q, K = F.normalize(Q, dim=-1), F.normalize(K, dim=-1)  # Unit length
        Y = self.memory(X, Q, K, V, self.decays(X))  # (B * num_heads, T, d / h)
        Y = F.rms_norm(Y, Y.shape[-1:], eps=1e-6)  # Unit RMS per head
        return self.W_o(batch_to_heads(Y, B) * F.silu(self.W_g(X)))

def heads_to_batch(Z, num_heads):
    """(batch_size, T, num_heads * e) -> (batch_size * num_heads, T, e).

    Defined in :numref:`sec_matrix-state`"""
    B, T, _ = Z.shape
    return Z.reshape(B, T, num_heads, -1).transpose(1, 2).flatten(0, 1)

def batch_to_heads(Y, batch_size):
    """(batch_size * num_heads, T, e) -> (batch_size, T, num_heads * e).

    Defined in :numref:`sec_matrix-state`"""
    _, T, e = Y.shape
    return Y.reshape(batch_size, -1, T, e).transpose(1, 2).flatten(2)

def delta_rule_writes(K, V, beta, log_a):
    """W and Z with corrected values U = W - Z S_0^T, for every chunk.

    K: (..., m, d_k); V: (..., m, d_v); beta, log_a: (..., m)

    Defined in :numref:`sec_deltanet`"""
    L = d2l.decay_mask(log_a)  # Products of decays within the chunk
    gamma = torch.exp(log_a.cumsum(-1)).unsqueeze(-1)  # Decay since start
    beta = beta.unsqueeze(-1)
    M = torch.eye(K.shape[-2], device=K.device) + (
        beta * L.tril(-1) * (K @ K.mT))  # Lower triangular
    W = torch.linalg.solve_triangular(M, beta * V, upper=False)
    Z = torch.linalg.solve_triangular(M, beta * gamma * K, upper=False)
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
    to_end = torch.exp(cum[..., -1:] - cum).unsqueeze(-1)
    I = torch.eye(K.shape[-1], device=K.device)
    A = torch.exp(cum[..., -1, None, None]) * I - (Z * to_end).mT @ K
    def combine(earlier, later):  # A scan over chunks, with matrix products
        (A1, W1), (A2, W2) = earlier, later
        return A1 @ A2, W1 @ A2 + W2
    W_end = d2l.associative_scan(combine, (A, (W * to_end).mT @ K))[1]
    S = torch.cat([torch.zeros_like(W_end[:, :1]), W_end[:, :-1]], 1)
    U = W - Z @ S.mT  # Corrected values, with the memory at each start
    Y = torch.exp(cum).unsqueeze(-1) * (Q @ S.mT) + (L * (Q @ K.mT)) @ U
    return Y.reshape(B, T, -1)

class GatedDeltaNet(d2l.GatedLinearAttention):
    """Gated linear attention with one decay per head, whose memory is
    written by the delta rule.

    Defined in :numref:`sec_deltanet`"""
    gate_scale = 1  # For init_decays: log sigmoid(b) = -1 / tau

    def __init__(self, num_hiddens, num_heads, chunk_size=32, conv_width=4,
                 timescales=(4, 256)):
        super().__init__(num_hiddens, num_heads, chunk_size, conv_width,
                         timescales)
        self.W_beta = nn.Linear(num_hiddens, num_heads)  # Write strength
        d2l.init_lecun(self.W_beta)

    def decay_gate(self, d):  # One gate per head
        # Sequential, so that init_decays finds the bias at W_a[-1]
        return nn.Sequential(nn.Linear(d, self.num_heads))

    def decays(self, X):  # (batch_size * num_heads, T)
        return d2l.heads_to_batch(F.logsigmoid(self.W_a(X)),
                                  self.num_heads).squeeze(-1)

    def memory(self, X, Q, K, V, log_a):
        beta = d2l.heads_to_batch(torch.sigmoid(self.W_beta(X)),
                                  self.num_heads).squeeze(-1)
        return chunked_delta_rule(Q, K, V, beta, log_a, self.chunk_size)

def gelu_ish(x):
    """The tanh approximation of GELU with rounded constants: a chain of
    eight elementwise operations.

    Defined in :numref:`sec_perf_model`"""
    return 0.5 * x * (1 + torch.tanh(0.8 * (x + 0.04 * x**3)))

def kernel_times(f, steps=10):
    """Device milliseconds per call of f, for every kernel that f runs.

    Defined in :numref:`sec_perf_model`"""
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            acc_events=True) as prof:
        for _ in range(steps):
            f()
        torch.cuda.synchronize()
    return {e.key: e.device_time_total / 1e3 / steps
            for e in prof.key_averages()
            if e.device_type == torch.autograd.DeviceType.CUDA}

def show_profile(kernels, step_ms, rows=6):
    """Busy device time against the wall-clock time step_ms of a step, the
    share of the matrix products, and the kernels with the most time.

    Defined in :numref:`sec_perf_model`"""
    busy = sum(kernels.values())
    products = sum(ms for name, ms in kernels.items()
                   if re.search('gemm|xmma|cutlass|matmul|sdpa', name, re.I))
    print(f'wall clock {step_ms:.2f} ms per step, device busy {busy:.2f} ms '
          f'({busy / step_ms:.0%}), matrix products {products / busy:.0%}')
    for name, ms in sorted(kernels.items(), key=lambda kv: -kv[1])[:rows]:
        name = re.sub(r'^void |\(anonymous namespace\)::|\w+::|\(.*', '', name)
        print(f'{ms:6.2f} ms {ms / busy:4.0%}  {name[:52]}')

import glob
import subprocess
import tempfile

def torchrun(script, nproc, *args, env=None):
    """Run Python source in nproc processes; return their JSON lines.

    Defined in :numref:`sec_multi_gpu_concise`"""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'script.py')
        with open(path, 'w') as f:
            f.write(script)
        launcher = os.path.join(os.path.dirname(sys.executable), 'torchrun')
        # Each process prints to a file of its own (--redirects 1), so that
        # the lines of different processes cannot interleave
        run = subprocess.run(
            [launcher, '--standalone', f'--nproc-per-node={nproc}',
             '--log-dir', tmp, '--redirects', '1', path, *map(str, args)],
            capture_output=True, text=True, env={**os.environ, **(env or {})})
        if run.returncode != 0:
            raise RuntimeError(run.stderr[-3000:])
        logs = sorted(glob.glob(f'{tmp}/*/attempt_0/*/stdout.log'),
                      key=lambda log: int(log.split(os.sep)[-2]))  # By rank
        return [json.loads(line) for log in logs for line in open(log)
                if line.startswith('{')]

def init_distributed():
    """Join the process group of torchrun; return the rank, the number of
    processes, and the GPU of this process.

    Defined in :numref:`sec_multi_gpu_concise`"""
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group('nccl', device_id=device)
    return (torch.distributed.get_rank(), torch.distributed.get_world_size(),
            device)

def resnet18(num_classes, in_channels=1):
    """ResNet-18 for small images: a 3x3 stem without max-pooling, He
    initialization.

    Defined in :numref:`sec_multi_gpu_concise`"""
    def stage(num_channels, first=False):
        return [d2l.Residual(num_channels, use_1x1conv=not first,
                             strides=1 if first else 2),
                d2l.Residual(num_channels)]
    net = nn.Sequential(
        # No stem bias: the batch normalization after the conv removes it
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(),
        *stage(64, first=True), *stage(128), *stage(256), *stage(512),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
    with torch.no_grad():  # Create the lazy parameters of the blocks
        net.eval()(torch.zeros(1, in_channels, 32, 32))
    for m in net.modules():  # He initialization, zero biases
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return net.train()

def windows(corpus, batch_size, num_steps=128):
    """Endless minibatches of windows at random offsets, each position with
    the next byte as its target.

    Defined in :numref:`sec_fast_transformer`"""
    offsets = torch.arange(num_steps + 1, device=corpus.device)
    while True:
        starts = torch.randint(len(corpus) - num_steps, (batch_size, 1),
                               device=corpus.device)
        W = corpus[starts + offsets].long()
        yield W[:, :-1], W[:, 1:]

def make_step(model, forward=None, bf16=False):
    """An AdamW training step of `model` on a minibatch (X, Y); `forward`
    replaces the model's forward pass, `bf16` turns on autocast.

    Defined in :numref:`sec_fast_transformer`"""
    forward = model if forward is None else forward
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=0.01, fused=True)
    def step(X, Y):
        with torch.autocast('cuda', torch.bfloat16, enabled=bf16):
            loss = F.cross_entropy(forward(X).flatten(0, 1), Y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss
    return step

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
        self.R = self.reward_table(self.verify)   # R[x, y] = r(x, y)

    def verify(self, x, y):
        """The verifier: reward 1 iff response y is prompt x's answer."""
        return float(self.resp[y] == self.answers[x])

    def reward_table(self, reward):
        """reward(x, y) for every prompt x (rows) and response y (columns)."""
        return np.array([[reward(x, y) for y in range(len(self.resp))]
                         for x in range(len(self.prompts))])

    @staticmethod
    def policy(theta):
        """pi(y | x): a softmax over each prompt's row of preferences."""
        z = np.exp(theta - theta.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    def success(self, pi, reward=None):
        """E r(x, y), y ~ pi(. | x), averaged over the prompts; r = verify
        unless another reward function is given."""
        R = self.R if reward is None else self.reward_table(reward)
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

class seeded:
    """In a `with seeded(seed):` block, PyTorch's CPU generator starts from
    seed; on exit it resumes its previous state, so the seed affects only
    the draws inside the block. seed=None draws from the global stream.

    Defined in :numref:`sec_imitation`"""
    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        if self.seed is not None:
            self.state = torch.get_rng_state()
            torch.default_generator.manual_seed(self.seed)

    def __exit__(self, *exc):
        if self.seed is not None:
            torch.set_rng_state(self.state)

class Policy(nn.Module):
    """A policy network and its optimizer; pi(.|s) is the softmax of the
    network's outputs at s.

    Defined in :numref:`sec_imitation`"""
    def __init__(self, policy, lr=1e-2):
        super().__init__()
        self.policy = policy
        self.opt_pi = torch.optim.Adam(policy.parameters(), lr=lr)

    def log_prob(self, obs, act):
        """log pi(a|s) for a batch of states and the actions taken there."""
        return torch.log_softmax(self.policy(obs), dim=-1) \
                    .gather(-1, act[:, None]).squeeze(-1)

    @classmethod
    def tabular(cls, num_states, num_actions, lr=0.1, seed=None):
        """One preference theta_{s,a} per state-action pair: an embedding."""
        with seeded(seed):
            policy = nn.Embedding(num_states, num_actions)
        nn.init.zeros_(policy.weight)
        return cls(policy, lr)

    def act(self, obs, rng):
        """Sample an action; numpy in, int out, the acting protocol of evaluate.

        Defined in :numref:`sec_imitation`"""
        with torch.inference_mode():
            probs = torch.softmax(self.policy(torch.as_tensor(obs)), -1).numpy()
        return int(rng.choice(len(probs), p=probs))

    def act_greedy(self, obs, rng=None):
        """The most probable action; the same protocol as act.

        Defined in :numref:`sec_imitation`"""
        with torch.inference_mode():
            return int(self.policy(torch.as_tensor(obs)).argmax())

    def log_prob_np(self, obs, act):
        """log pi(a|s) for numpy batches of states and actions, as numpy.

        Defined in :numref:`sec_imitation`"""
        with torch.inference_mode():
            return self.log_prob(torch.as_tensor(obs),
                                 torch.as_tensor(act)).numpy()

    def probs(self, num_states, num_actions):
        """pi(a|s) for every pair, as a (num_states, num_actions) numpy array.

        Defined in :numref:`sec_imitation`"""
        s, a = np.divmod(np.arange(num_states * num_actions), num_actions)
        return np.exp(self.log_prob_np(s, a)).reshape(num_states, num_actions)

def policy_step(ac, obs, act, weight):
    """One ascent step on (1/n) sum_t w_t log pi(a_t|s_t) over a batch of
    n states and actions; the weights w arrive as numpy, that is, as data.

    Defined in :numref:`sec_imitation`"""
    w = torch.as_tensor(weight)
    loss = -(w * ac.log_prob(torch.as_tensor(obs),
                             torch.as_tensor(act))).mean()
    ac.opt_pi.zero_grad()
    loss.backward()
    ac.opt_pi.step()
    return loss.item()

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

    def lambda_return(self, value_fn, gamma, lam):
        """The critic's target for GAE: the advantage plus the value it is
        measured from.

        Defined in :numref:`sec_actorcritic`"""
        return self.gae(value_fn, gamma, lam) + value_fn(self.obs)

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

def train_reinforce_tabular(ac, env, gamma, seed, num_updates=256,
                            batch_episodes=16):
    """REINFORCE: a fresh batch from the current policy, every step of a
    trajectory weighted by that trajectory's return, one ascent step.
    Yields the batch's mean return and its number of steps.

    Defined in :numref:`sec_policygradient`"""
    rng = np.random.default_rng(seed)      # one stream for all sampling
    env.reset(seed=seed)
    for _ in range(num_updates):
        batch = rollout(env, ac.act, batch_episodes, rng)
        R = batch.episode_returns(gamma)
        d2l.policy_step(ac, batch.obs, batch.act, batch.per_step(R))
        yield float(R.mean()), len(batch)

def policy_objective(theta, P, r, gamma):
    """J(theta) = V(s0) of the softmax policy, a differentiable linear solve.

    Defined in :numref:`sec_policygradient`"""
    pi = torch.softmax(theta, -1)
    P_pi = torch.einsum('sa,sat->st', pi, P)
    return torch.linalg.solve(torch.eye(len(P)) - gamma * P_pi,
                              (pi * r).sum(-1))[0]

def tabular_score(obs, act, w, pi):
    """sum_t w_t (e_{a_t} - pi(.|s_t)), each term added into row s_t: the
    weighted softmax score (eq_softmax_score) of a batch, the shape of pi.

    Defined in :numref:`sec_policygradient`"""
    u = np.zeros(pi.shape)
    np.add.at(u, obs, w[:, None] * (np.eye(pi.shape[1])[act] - pi[obs]))
    return u

def standardize(x, eps=1e-8):
    """Center a NumPy batch of weights and divide by its standard deviation
    (NumPy's std, which divides by the batch size).

    Defined in :numref:`sec_baselines`"""
    return (x - x.mean()) / (x.std() + eps)

def group_step(task, theta, K, reward, rng, lr=2.0):
    """verifier_step with the group-normalized weight (GRPO): the rewards
    of each prompt's K responses, standardized within the group.

    Defined in :numref:`sec_baselines`"""
    return d2l.verifier_step(task, theta, K, reward, rng, standardize, lr)

class ActorCritic(d2l.Policy):
    """A policy and a value function, each with its own optimizer.

    Defined in :numref:`sec_deeprl`"""
    def __init__(self, policy, value, lr=1e-2):
        super().__init__(policy, lr)
        self.value = value
        self.opt_v = torch.optim.Adam(value.parameters(), lr=lr)

    def V(self, obs):
        """V(s) for a batch of states, as a tensor."""
        return self.value(obs).squeeze(-1)

    def value_np(self, obs):
        """V(s) for a numpy batch of states, as numpy."""
        with torch.inference_mode():
            return self.V(torch.as_tensor(obs)).numpy()

    @classmethod
    def tabular(cls, num_states, num_actions, lr=0.1, seed=None):
        """Preference and value tables: two zero-initialized embeddings."""
        with d2l.seeded(seed):
            policy, value = (nn.Embedding(num_states, num_actions),
                             nn.Embedding(num_states, 1))
        nn.init.zeros_(policy.weight), nn.init.zeros_(value.weight)
        return cls(policy, value, lr)

    @classmethod
    def mlp(cls, obs_dim, num_actions, hidden=64, lr=1e-2, seed=None):
        """The same container with the tables replaced by one-hidden-layer nets.

        Defined in :numref:`sec_deeprl`"""
        with d2l.seeded(seed):
            policy, value = (tanh_mlp(obs_dim, hidden, num_actions),
                             tanh_mlp(obs_dim, hidden, 1))
        return cls(policy, value, lr)

def tanh_mlp(in_dim, hidden, out_dim):
    """A network with one tanh hidden layer.

    Defined in :numref:`sec_deeprl`"""
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                         nn.Linear(hidden, out_dim))

def fit_value(ac, obs, target, num_steps=1):
    """Regress the value head on a fixed target: eq_value_baseline for nets.

    Defined in :numref:`sec_deeprl`"""
    obs, target = torch.as_tensor(obs), torch.as_tensor(target)
    for _ in range(num_steps):
        loss = ((ac.V(obs) - target) ** 2).mean()
        ac.opt_v.zero_grad()
        loss.backward()
        ac.opt_v.step()
    return loss.item()

class GaussianHead(nn.Module):
    """A Gaussian over actions: a trunk of hidden features, a linear mean,
    and a log standard deviation that is one learned vector or, with
    `state_std=True`, a second linear head clamped to [-5, 2]. The default
    trunk is one tanh layer; any module with `hidden` outputs will do.
    `zero_mean=True` sets the mean layer to zero, so that mu = 0 at first.

    Defined in :numref:`sec_deeprl`"""
    def __init__(self, obs_dim, act_dim, hidden, state_std=False, trunk=None,
                 zero_mean=False):
        super().__init__()
        self.trunk = trunk if trunk is not None else nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act_dim)
        if zero_mean:
            nn.init.zeros_(self.mean.weight), nn.init.zeros_(self.mean.bias)
        self.state_std = state_std
        self.log_std = (nn.Linear(hidden, act_dim) if state_std
                        else nn.Parameter(torch.zeros(act_dim)))

    def forward(self, obs):
        h = self.trunk(obs)
        if self.state_std:
            return self.mean(h), self.log_std(h).clamp(-5, 2).exp()
        return self.mean(h), self.log_std.exp()

class GaussianPolicy(ActorCritic):
    """The same interface over a Normal instead of a softmax; nothing that
    consumes the interface changes. `head_args` go to GaussianHead.

    Defined in :numref:`sec_deeprl`"""
    def __init__(self, obs_dim, act_dim, hidden=64, lr=1e-2, seed=None,
                 **head_args):
        with d2l.seeded(seed):
            policy = GaussianHead(obs_dim, act_dim, hidden, **head_args)
            value = tanh_mlp(obs_dim, hidden, 1)
        super().__init__(policy, value, lr)

    def log_prob(self, obs, act):
        """log pi(a|s): the Normal density, summed over action dimensions."""
        mean, std = self.policy(obs)
        return torch.distributions.Normal(mean, std).log_prob(act).sum(-1)

    def act(self, obs, rng):
        """Sample mu(s) + sigma(s) z, z ~ N(0, I) from the numpy rng."""
        with torch.inference_mode():
            mean, std = self.policy(torch.as_tensor(obs))
        return mean.numpy() + std.numpy() * rng.standard_normal(
            mean.shape, dtype=np.float32)

    def act_greedy(self, obs, rng=None):
        """The mean action mu(s)."""
        with torch.inference_mode():
            return self.policy(torch.as_tensor(obs))[0].numpy()

def steps_to(run, level, window):
    """Env steps at which the mean return of the last `window` rows of
    run = [(env step, return, ...), ...] first reaches `level`; inf if it
    never does.

    Defined in :numref:`sec_actorcritic`"""
    mean = np.convolve(run[:, 1], np.ones(window) / window, 'valid')
    hit = np.flatnonzero(mean >= level)
    return run[hit[0] + window - 1, 0] if len(hit) else np.inf

def clipped_surrogate(rho, adv, epsilon):
    """PPO's per-sample objective eq_ppo_clip: min(rho A, clip(rho) A).

    Defined in :numref:`sec_ppo`"""
    return torch.minimum(rho * adv,
                         rho.clamp(1 - epsilon, 1 + epsilon) * adv)

def plain_surrogate(rho, adv, epsilon):
    """The unclipped per-sample objective rho A.

    Defined in :numref:`sec_ppo`"""
    return rho * adv

def ppo_epochs(ac, obs, act, adv, logp_old, epsilon, num_epochs,
               entropy_coef=0.01, surrogate=clipped_surrogate):
    """num_epochs passes on the mean surrogate plus an entropy bonus, on one
    frozen batch of states and actions; returns [num_epochs, 3] numpy
    diagnostics, recorded before each pass: fraction of ratios outside the
    band, approximate KL, mean policy entropy.

    Defined in :numref:`sec_ppo`"""
    obs, act, adv, logp_old = (torch.as_tensor(x)
                               for x in (obs, act, adv, logp_old))
    diag = []
    for _ in range(num_epochs):
        logp_all = torch.log_softmax(ac.policy(obs), dim=-1)
        logp = logp_all.gather(-1, act[:, None]).squeeze(-1)
        rho = torch.exp(logp - logp_old)
        entropy = -(logp_all.exp() * logp_all).sum(-1).mean()
        loss = -surrogate(rho, adv, epsilon).mean() - entropy_coef * entropy
        ac.opt_pi.zero_grad()
        loss.backward()
        ac.opt_pi.step()
        diag.append(torch.stack([((rho - 1).abs() > epsilon).float().mean(),
                                 (logp_old - logp).mean(), entropy]).detach())
    return torch.stack(diag).numpy()

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

class SquashedGaussianPolicy(nn.Module):
    """d2l.GaussianHead with SAC's network (two ReLU layers and a
    state-dependent standard deviation), squashed through
    a = scale * tanh(u).

    Defined in :numref:`sec_sac`"""
    def __init__(self, obs_dim, act_dim, hidden=64, scale=2.0, seed=None):
        super().__init__()
        with d2l.seeded(seed):
            trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
            self.policy = d2l.GaussianHead(obs_dim, act_dim, hidden,
                                           state_std=True, trunk=trunk)
        self.scale = scale

    def log_prob_u(self, u, mean, std):
        """log pi at a = scale tanh(u), from the pre-squash u the sampler
        keeps."""
        logdet = 2 * (np.log(2) - u - nn.functional.softplus(-2 * u))
        return (torch.distributions.Normal(mean, std).log_prob(u)
                - logdet - np.log(self.scale)).sum(-1)

    def sample(self, obs):
        """A reparameterized action and its log-probability, differentiable."""
        mean, std = self.policy(obs)
        u = mean + std * torch.randn_like(std)
        return self.scale * torch.tanh(u), self.log_prob_u(u, mean, std)

    @torch.inference_mode()
    def _draw(self, obs, rng):
        """A pre-squash sample u with the head's mean and std, from numpy
        states and the noise of rng."""
        mean, std = self.policy(torch.as_tensor(obs))
        noise = rng.standard_normal(mean.shape, dtype=np.float32)
        return mean + std * torch.as_tensor(noise), mean, std

    def act(self, obs, rng):
        """A sampled action, numpy in and out."""
        return self.scale * np.tanh(self._draw(obs, rng)[0].numpy())

    @torch.inference_mode()
    def act_logp(self, obs, rng):
        """A sampled action and its log-probability, numpy in and out."""
        u, mean, std = self._draw(obs, rng)
        return (self.scale * np.tanh(u.numpy()),
                self.log_prob_u(u, mean, std).numpy())

    @torch.inference_mode()
    def act_greedy(self, obs, rng=None):
        """The squashed mean scale * tanh(mean), numpy in and out."""
        mean = self.policy(torch.as_tensor(obs))[0].numpy()
        return self.scale * np.tanh(mean)

class ReplayBufferC(d2l.ReplayBuffer):
    """d2l.ReplayBuffer with the action column widened to float vectors.

    Defined in :numref:`sec_sac`"""
    def __init__(self, capacity, obs_dim, act_dim):
        super().__init__(capacity, obs_dim)
        self.act = np.zeros((capacity, act_dim), np.float32)

def update_D(X, Z, net_D, net_G, trainer_D):
    """Update the critic; return the summed log loss, halved.

    Defined in :numref:`sec_basic_gan`"""
    trainer_D.zero_grad()
    real_Y = net_D(X)
    # The generator is the fixed player here: detach its output
    fake_Y = net_D(net_G(Z).detach())
    loss_D = (F.binary_cross_entropy_with_logits(
                  real_Y, torch.ones_like(real_Y), reduction='sum') +
              F.binary_cross_entropy_with_logits(
                  fake_Y, torch.zeros_like(fake_Y), reduction='sum')) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D

def update_G(Z, net_D, net_G, trainer_G):
    """Update the generator on the non-saturating loss; return its sum.

    Defined in :numref:`sec_basic_gan`"""
    trainer_G.zero_grad()
    # net_D changed in its half-step, so recompute the scores
    fake_Y = net_D(net_G(Z))
    loss_G = F.binary_cross_entropy_with_logits(
        fake_Y, torch.ones_like(fake_Y), reduction='sum')
    loss_G.backward()
    trainer_G.step()
    return loss_G

def train_gan(net_D, net_G, data_iter, num_epochs, latent_dim, lr_D, lr_G,
              step_G=update_G, seed=None, on_epoch=None):
    """Alternate update_D and step_G over the minibatches with Adam
    (beta_1 = 0.5), calling on_epoch(epoch) after each epoch if given;
    return the per-sample (loss_D, loss_G) of each epoch.

    Defined in :numref:`sec_basic_gan`"""
    if seed is not None:
        torch.manual_seed(seed)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D,
                                 betas=(0.5, 0.999))
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G,
                                 betas=(0.5, 0.999))
    history = []
    for epoch in range(1, num_epochs + 1):
        loss_D, loss_G, n = 0, 0, 0  # summed on the device, read once
        for X, *_ in data_iter:  # labels, if any, are ignored
            Z = torch.randn(X.shape[0], latent_dim, device=X.device)
            loss_D += update_D(X, Z, net_D, net_G, trainer_D).detach()
            loss_G += step_G(Z, net_D, net_G, trainer_G).detach()
            n += X.shape[0]
        history.append(torch.stack([loss_D, loss_G]) / n)
        if on_epoch is not None:
            on_epoch(epoch)
    return torch.stack(history).cpu().numpy()

def make_mlp(sizes, act):
    """Linear layers of the given widths with the activation module act
    between them; weights uniform in +-1/sqrt(fan_in), zero biases.

    Defined in :numref:`sec_basic_gan`"""
    layers = []
    for d_in, d_out in zip(sizes[:-1], sizes[1:]):
        linear = nn.Linear(d_in, d_out)  # weights uniform by default
        nn.init.zeros_(linear.bias)
        layers += [linear, act]
    return nn.Sequential(*layers[:-1])

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
    return F.softplus(critic(fake) - critic(real)).mean()

def rpgan_loss_G(critic, real, fake):
    """Non-saturating pairing loss for the generator.

    Defined in :numref:`sec_gan_convergence`"""
    return F.softplus(critic(real) - critic(fake)).mean()

def gan_loss_D(critic, real, fake):
    """Logistic critic loss -E[log sigma(D(x))] - E[log sigma(-D(x'))]: the
    batch means, where update_D reports the halved sum.

    Defined in :numref:`sec_gan_convergence`"""
    return F.softplus(-critic(real)).mean() + F.softplus(critic(fake)).mean()

def gan_loss_G(critic, real, fake):
    """Non-saturating generator loss -E[log sigma(D(x'))]; real is unused.

    Defined in :numref:`sec_gan_convergence`"""
    return F.softplus(-critic(fake)).mean()

def r1_r2_penalty(critic, real, fake):
    """Per-sample squared critic input gradients on real (R1) and fake (R2),
    before the gamma/2 scale.

    Defined in :numref:`sec_gan_convergence`"""
    def sq_grad_norm(x):
        x = x.detach().requires_grad_(True)
        grad, = torch.autograd.grad(critic(x).sum(), x, create_graph=True)
        return grad.reshape(x.shape[0], -1).pow(2).sum(dim=1)
    return sq_grad_norm(real), sq_grad_norm(fake)

def critic_step(net_D, opt_D, loss_D, real, fake, gamma, y=None):
    """One critic step on loss_D + gamma/2 (R1 + R2); return the loss and
    the batch means of R1 and R2.

    Defined in :numref:`sec_gan_convergence`"""
    critic = net_D if y is None else lambda x: net_D(x, y)
    fake = fake.detach()  # the generator is held fixed
    r1, r2 = r1_r2_penalty(critic, real, fake)
    loss = loss_D(critic, real, fake)
    opt_D.zero_grad()
    (loss + gamma / 2 * (r1 + r2).mean()).backward()
    opt_D.step()
    return loss.detach(), r1.mean().detach(), r2.mean().detach()

def generator_step(net_G, opt_G, loss_G, net_D, real, z, y=None):
    """One generator step on loss_G against the critic; return the loss.

    Defined in :numref:`sec_gan_convergence`"""
    critic = net_D if y is None else lambda x: net_D(x, y)
    loss = loss_G(critic, real, net_G(z) if y is None else net_G(z, y))
    opt_G.zero_grad()
    loss.backward()
    opt_G.step()
    return loss.detach()

d2l.DATA_HUB['pokemon'] = (d2l.DATA_URL + 'pokemon.zip',
                           'c065c0e2593b8b161a2d7873e42418bf6a21106c')

def load_pokemon(resolution=64):
    """All Pokemon sprites as one (n, 3, h, w) tensor in [-1, 1].

    Defined in :numref:`sec_dcgan`"""
    import warnings  # not in the d2l library preamble
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((resolution, resolution)),
        torchvision.transforms.PILToTensor()])
    dataset = torchvision.datasets.ImageFolder(
        d2l.download_extract('pokemon'), transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, num_workers=d2l.get_dataloader_workers())
    with warnings.catch_warnings():  # palette sprites with transparency
        warnings.filterwarnings('ignore', message='Palette images')
        return torch.cat([X for X, _ in loader]).float() / 127.5 - 1

def fan_in_init(module):
    """Fan-in normal weights with the leaky-ReLU gain, zero offsets.

    Defined in :numref:`sec_dcgan`"""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=0.2)
        nn.init.zeros_(module.bias)

class Generator(nn.Module):
    """Learned 4x4 constant + projected latent (+ extra 4x4 input maps),
    bilinearly upsampled to resolution x resolution; no normalization.

    Defined in :numref:`sec_dcgan`"""
    def __init__(self, latent_dim, resolution, base_ch, *, extra_ch=0,
                 const_ch=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.const = nn.Parameter(torch.randn(1, const_ch, 4, 4))
        self.z_proj = nn.Linear(latent_dim, 16 * latent_dim)
        self.mix = nn.Conv2d(const_ch + latent_dim + extra_ch, base_ch, 3,
                             padding=1)
        num_stages = int(np.log2(resolution / 4))       # 64: four doublings
        chans = [base_ch // 2 ** i for i in range(num_stages + 1)]
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.to_rgb = nn.Conv2d(chans[-1], 3, 3, padding=1)
        self.act = nn.LeakyReLU(0.2)
        self.apply(fan_in_init)

    def forward(self, z, extra=None):
        zc = self.z_proj(z).reshape(-1, self.latent_dim, 4, 4)
        const = self.const.expand(z.shape[0], -1, -1, -1)
        maps = [const, zc] + ([] if extra is None else [extra])
        x = self.act(self.mix(torch.cat(maps, dim=1)))
        for conv in self.stages:                     # 4 -> 8 -> ... -> 64
            x = F.interpolate(x, scale_factor=2, mode='bilinear')
            x = self.act(conv(x))
        return self.to_rgb(x)                        # no tanh: raw output

class Discriminator(nn.Module):
    """Mirror of the generator: conv + leaky ReLU + bilinear downsampling to
    a 4x4 feature map phi(x), then a linear head on the flattened map.

    Defined in :numref:`sec_dcgan`"""
    def __init__(self, resolution, base_ch):
        super().__init__()
        num_stages = int(np.log2(resolution / 4))
        chans = [3] + [base_ch * 2 ** i for i in range(num_stages)]
        self.stages = nn.ModuleList(
            [nn.Conv2d(c_in, c_out, 3, padding=1)
             for c_in, c_out in zip(chans[:-1], chans[1:])])
        self.mix = nn.Conv2d(chans[-1], 2 * chans[-1], 3, padding=1)
        self.head = nn.Linear(2 * chans[-1] * 4 * 4, 1)
        self.act = nn.LeakyReLU(0.2)
        self.apply(fan_in_init)

    def phi(self, x):
        for conv in self.stages:                     # 64 -> 32 -> ... -> 4
            x = F.interpolate(self.act(conv(x)), scale_factor=0.5,
                              mode='bilinear', antialias=True)
        return self.act(self.mix(x))

    def forward(self, x):
        return self.head(self.phi(x).flatten(1))

class EMA:
    """Exponential moving average of a model's parameters, given as a
    half-life in steps. Buffers such as batch-norm statistics are copied;
    the average is kept in evaluation mode.

    Defined in :numref:`sec_dcgan`"""
    def __init__(self, model, half_life):
        import copy  # the d2l library preamble does not import it
        self.model = copy.deepcopy(model).requires_grad_(False).eval()
        self.decay = 0.5 ** (1 / half_life)

    def update(self, model):
        with torch.no_grad():
            for avg, p in zip(self.model.parameters(), model.parameters()):
                avg.lerp_(p, 1 - self.decay)
            for avg, b in zip(self.model.buffers(), model.buffers()):
                avg.copy_(b)

def sample_real(images, n, labels=None):
    """Draw a training batch with random horizontal flips; return it with
    its labels (None without labels).

    Defined in :numref:`sec_dcgan`"""
    idx = torch.randint(0, len(images), (n,), device=images.device)
    batch = images[idx]
    flip = torch.rand(n, device=images.device) < 0.5
    batch = torch.where(flip.view(-1, 1, 1, 1), batch.flip(-1), batch)
    return batch, (None if labels is None else labels[idx])

def pair_stats(net_D, real, fake, y=None):
    """Ranking accuracy P(D(x) > D(x')) and mean rank weight
    E[sigma(D(x') - D(x))] over aligned (real, generated) pairs; given
    labels y, the critic is D(., y).

    Defined in :numref:`sec_dcgan`"""
    critic = net_D if y is None else lambda x: net_D(x, y)
    with torch.no_grad():
        diff = critic(fake) - critic(real)
    return float((diff < 0).float().mean()), float(torch.sigmoid(diff).mean())

def train_r3gan(net_G, net_D, loss_D, loss_G, gamma, num_steps, images,
                labels=None, *, eval_images=None, eval_labels=None,
                eval_size=256, batch_size=64, lr=2e-4, half_life=500,
                log_every=250, seed=None):
    """Train with Adam (beta_1 = 0), alternating critic_step and
    generator_step on flipped real batches; a generated batch takes the
    labels of its real batch, if any. Every log_every steps, log the losses,
    R1, R2 and pair_stats on fresh pairs from eval_images (default: images).
    Return the EMA of the generator and the history, a dict of arrays.

    Defined in :numref:`sec_dcgan`"""
    if seed is not None:
        torch.manual_seed(seed)
    if eval_images is None:
        eval_images, eval_labels = images, labels
    opt_G, opt_D = [torch.optim.Adam(net.parameters(), lr, betas=(0.0, 0.99))
                    for net in (net_G, net_D)]
    noise = lambda n: torch.randn(n, net_G.latent_dim, device=images.device)
    generate = lambda z, y: net_G(z) if y is None else net_G(z, y)
    ema = EMA(net_G, half_life)
    history = {k: [] for k in ('step', 'loss_D', 'loss_G', 'R1', 'R2',
                               'accuracy', 'rank_weight')}
    for step in range(1, num_steps + 1):
        X, y = sample_real(images, batch_size, labels)
        with torch.no_grad():
            fake = generate(noise(batch_size), y)
        loss_d, r1, r2 = d2l.critic_step(net_D, opt_D, loss_D, X, fake,
                                         gamma, y)
        X, y = sample_real(images, batch_size, labels)
        loss_g = d2l.generator_step(net_G, opt_G, loss_G, net_D, X,
                                    noise(batch_size), y)
        ema.update(net_G)
        if step % log_every == 0:  # fresh pairs of real and generated images
            X, y = sample_real(eval_images, eval_size, eval_labels)
            with torch.no_grad():
                fake = generate(noise(eval_size), y)
            acc, weight = pair_stats(net_D, X, fake, y)
            logs = (step, loss_d, loss_g, r1, r2, acc, weight)
            for k, v in zip(history, logs):
                history[k].append(float(v))
    return ema.model, {k: np.array(v) for k, v in history.items()}

def tile_images(imgs, num_rows):
    """Tile images (n, 3, h, w) in [-1, 1] into one (rows * h, cols * w, 3)
    array in [0, 1].

    Defined in :numref:`sec_dcgan`"""
    n, c, h, w = imgs.shape
    grid = (imgs.clamp(-1, 1) + 1) / 2
    grid = grid.reshape(num_rows, n // num_rows, c, h, w)
    grid = grid.permute(0, 3, 1, 4, 2)  # (rows, h, cols, w, c)
    return grid.reshape(num_rows * h, -1, c).cpu().numpy()

def fid_score(feat_p, feat_q):
    """Frechet distance between Gaussians fitted to two feature sets.

    Defined in :numref:`sec_dcgan`"""
    feat_p = np.asarray(feat_p, np.float64)
    feat_q = np.asarray(feat_q, np.float64)
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

def batched(f, X, batch_size=1000):
    """Apply f to X in batches, without gradients, and concatenate.

    Defined in :numref:`sec_dcgan`"""
    with torch.no_grad():
        return torch.cat([f(X_k) for X_k in X.split(batch_size)])

class GaussianMixture:
    """A mixture of isotropic Gaussians, exact at every noise level.

    Defined in :numref:`sec_diffusion-denoising`"""
    def __init__(self, means, weights, std, device=None):
        self.means = torch.tensor(means, dtype=torch.float32, device=device)
        self.weights = torch.tensor(weights, dtype=torch.float32,
                                    device=device)
        self.std = std

    @staticmethod
    def three_modes(device=None):
        """The chapter's running example, with mode weights 0.5/0.3/0.2."""
        return GaussianMixture([[-2.5, -1.5], [2.5, -1.5], [0.0, 2.5]],
                               [0.5, 0.3, 0.2], 0.5, device)

    def sample(self, n):
        """n draws from the mixture."""
        k = torch.multinomial(self.weights, n, replacement=True)
        return self.means[k] + self.std * torch.randn(
            n, self.means.shape[1], device=self.means.device)

    def _modes(self, x, alpha, sigma):
        """Per-mode log densities, offsets x - alpha mu_k, variances, alpha."""
        a, s = (torch.as_tensor(v, dtype=x.dtype, device=x.device)
                .reshape(-1, 1, 1) for v in (alpha, sigma))
        var = (a * self.std) ** 2 + s ** 2
        r = x[:, None] - a * self.means
        log_p = (torch.log(self.weights) - (r ** 2).sum(-1) / (2 * var[..., 0])
                 - x.shape[1] / 2 * torch.log(2 * torch.pi * var[..., 0]))
        return log_p, r, var, a

    def log_prob(self, x, alpha=1.0, sigma=0.0):
        """log p(x) for x = alpha x0 + sigma eps."""
        return torch.logsumexp(self._modes(x, alpha, sigma)[0], 1)

    def score(self, x, alpha=1.0, sigma=0.0):
        """The gradient of log p(x) for x = alpha x0 + sigma eps."""
        log_p, r, var, _ = self._modes(x, alpha, sigma)
        return -(torch.softmax(log_p, 1)[..., None] * r / var).sum(1)

    def posterior_mean(self, x, alpha, sigma):
        """E[x0 | x]; finite at alpha = 0 and at sigma = 0."""
        log_p, r, var, a = self._modes(x, alpha, sigma)
        return (torch.softmax(log_p, 1)[..., None]
                * (self.means + a * self.std ** 2 / var * r)).sum(1)

    def velocity(self, x, t):
        """E[eps - x0 | x_t] on the linear path x_t = (1 - t) x0 + t eps."""
        log_p, r, var, a = self._modes(x, 1 - t, t)
        eps = (1 - a) * r / var                        # E[eps | x_t, mode k]
        x0 = self.means + a * self.std ** 2 / var * r  # E[x0 | x_t, mode k]
        return (torch.softmax(log_p, 1)[..., None] * (eps - x0)).sum(1)

    def nll(self, x):
        """Mean negative log-likelihood of clean points, in nats.

        Defined in :numref:`sec_diffusion-denoising`"""
        return -self.log_prob(x).mean()

    def summary(self, x):
        """Mode weights by nearest mean, stray fraction, spread of the rest.

        Defined in :numref:`sec_diffusion-denoising`"""
        d2 = ((x[:, None] - self.means) ** 2).sum(-1)
        near, k = d2.min(1)
        inside = near < (3 * self.std) ** 2
        weights = torch.bincount(k, minlength=len(self.weights)) / len(x)
        return dict(weights=[round(w, 3) for w in weights.tolist()],
                    stray=round(1 - inside.float().mean().item(), 4),
                    spread=round(near[inside].mean().item(), 3))

def scatter_panels(samples, titles=None, data=None, lim=4.5,
                   size=3.0):
    """Scatter 2-D point sets side by side on equal, shared axes.

    Defined in :numref:`sec_diffusion-denoising`"""
    _, axes = d2l.plt.subplots(1, len(samples), squeeze=False,
                               figsize=(size * len(samples), size))
    for i, (ax, pts) in enumerate(zip(axes[0], samples)):
        if data is not None:
            ax.scatter(*d2l.numpy(data).T, s=2, c='lightgray')
        ax.scatter(*d2l.numpy(pts).T, s=2)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect='equal')
        if titles:
            ax.set_title(titles[i])
    return axes[0]

def time_embedding(t, dim):
    """Sines and cosines of t at dim / 2 geometric frequencies.

    Defined in :numref:`sec_diffusion-denoising`"""
    freqs = torch.exp(-math.log(10000) / (dim // 2)
                      * torch.arange(dim // 2, device=t.device))
    args = t.float()[:, None] * freqs
    return torch.cat([args.sin(), args.cos()], 1)

class TimeMLP(nn.Module):
    """An MLP on a point x and the time features of one or more times.

    Defined in :numref:`sec_diffusion-denoising`"""
    def __init__(self, hidden=256, depth=3, *, dim=2, num_times=1,
                 time_dim=64):
        super().__init__()
        self.time_dim = time_dim
        self.embed = d2l.make_mlp([time_dim * num_times, hidden, hidden],
                                  nn.SiLU())
        self.inp = nn.Linear(dim, hidden)
        self.body = nn.Sequential(  # depth layers, each after a SiLU
            nn.SiLU(), d2l.make_mlp([hidden] * depth + [dim], nn.SiLU()))

    def forward(self, x, *ts):
        feats = [time_embedding(t, self.time_dim) for t in ts]
        return self.body(self.inp(x) + self.embed(torch.cat(feats, 1)))

def train_regression(model, loss_fn, batch_fn, num_steps, *,
                     lr=1e-3, cosine=False, ema_half_life=1000,
                     compiled=True):
    """Adam on loss_fn(model, batch_fn()) -> (model, EMA model, losses).

    Defined in :numref:`sec_diffusion-denoising`"""
    params = list(model.parameters())  # fused Adam: one kernel per step on a GPU
    opt = torch.optim.Adam(params, lr=lr, fused=params[0].is_cuda)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_steps)
             if cosine else None)
    if compiled and params[0].is_cuda:  # loss and gradient as CUDA graphs
        loss_fn = torch.compile(loss_fn, mode='reduce-overhead',
                                dynamic=False)
    ema, losses = d2l.EMA(model, ema_half_life), []
    for _ in range(num_steps):
        loss = loss_fn(model, batch_fn())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched:
            sched.step()
        ema.update(model)
        losses.append(loss.detach().clone())  # the graph reuses its output
    return model, ema.model, torch.stack(losses)

@torch.no_grad()
def langevin(score_fn, x, step, num_steps):
    """Langevin steps x <- x + step / 2 * score + sqrt(step) * noise.

    Defined in :numref:`sec_diffusion-denoising`"""
    for _ in range(num_steps):
        x = x + step / 2 * score_fn(x) + step ** 0.5 * torch.randn_like(x)
    return x

def load_fashion_mnist_pm1(binary=False):
    """Fashion-MNIST (X, y, X_test, y_test), images in [-1, 1] or binary.

    Defined in :numref:`sec_diffusion-denoising`"""
    data, device = d2l.FashionMNIST(), d2l.try_gpu()  # on the GPU if any
    def prep(X):  # (n, 1, 28, 28) floats or (n, 28, 28) binary pixels
        return (X > 127).long() if binary else X[:, None].float() / 127.5 - 1
    return (prep(data.train.data).to(device), data.train.targets.to(device),
            prep(data.val.data).to(device), data.val.targets.to(device))

def show_gray(imgs, num_rows, num_cols, titles=None, scale=1.0):
    """Show images (n, 1, h, w) in [-1, 1], or binary (n, h, w), in gray.

    Defined in :numref:`sec_diffusion-denoising`"""
    imgs = imgs.detach().cpu()
    imgs = ((imgs[:, 0] + 1) / 2).clamp(0, 1) if imgs.is_floating_point() \
        else imgs.float()
    return d2l.show_images(imgs[..., None].expand(-1, -1, -1, 3), num_rows,
                           num_cols, titles, scale)

class GaussianPath:
    """x_t = alpha(t) x0 + sigma(t) eps, data at t = 0 and noise at t = 1.

    Defined in :numref:`sec_diffusion-gaussian-paths`"""
    def __init__(self, alpha, sigma):
        self.alpha, self.sigma = alpha, sigma

    @staticmethod
    def vp(beta_min=0.1, beta_max=20.0):
        """Variance preserving, Ho et al.'s linear beta in continuous time."""
        log_abar = lambda t: -(beta_min * t + (beta_max - beta_min) * t**2 / 2)
        return GaussianPath(lambda t: torch.exp(log_abar(t) / 2),
                            lambda t: torch.sqrt(-torch.expm1(log_abar(t))))

    @staticmethod
    def linear():
        """The straight line of flow matching and rectified flow."""
        return GaussianPath(lambda t: 1 - t, lambda t: t)

    @staticmethod
    def cosine(s=0.008):
        """Variance preserving, Nichol and Dhariwal's cosine schedule."""
        c = lambda t: torch.cos(math.pi / 2 * (t + s) / (1 + s))
        g = lambda t: c(t) / c(torch.zeros_like(t))   # exactly 1 at t = 0
        # and exactly 0 at t = 1 (cos(pi/2) is not), keeping the slope
        f = lambda t: torch.where(t < 1, g(t), g(t) - g(t).detach())
        return GaussianPath(f, lambda t: torch.sqrt(1 - f(t) ** 2))

    def log_snr(self, t):
        """lambda_t = log(alpha_t^2 / sigma_t^2), falling from +inf at 0."""
        return 2 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))

    def sample(self, x0, t, eps):
        """x_t for per-example times t of shape (n,)."""
        t = t.reshape(-1, *[1] * (x0.dim() - 1))
        return self.alpha(t) * x0 + self.sigma(t) * eps

    def grid(self, steps, t_start=1.0, t_end=0.0):
        """Lists of alpha and sigma at steps + 1 even times, t_start to t_end.

        Defined in :numref:`sec_diffusion-gaussian-paths`"""
        t = torch.linspace(t_start, t_end, steps + 1, dtype=torch.float64)
        return self.alpha(t).tolist(), self.sigma(t).tolist()

    def rates(self, t):
        """The time derivatives (alpha'(t), sigma'(t)), by forward-mode AD.

        Defined in :numref:`sec_diffusion-gaussian-paths`"""
        ones = torch.ones_like(t)
        return (torch.func.jvp(self.alpha, (t,), (ones,))[1],
                torch.func.jvp(self.sigma, (t,), (ones,))[1])

    def target(self, kind, x0, eps, t):
        """The regression target of a kind: eps, x0, v, score or velocity.

        Defined in :numref:`sec_diffusion-gaussian-paths`"""
        t = t.reshape(-1, *[1] * (x0.dim() - 1))
        a, s = self.alpha(t), self.sigma(t)
        if kind == 'eps': return eps
        if kind == 'x0': return x0
        if kind == 'v': return a * eps - s * x0
        if kind == 'score': return -eps / s
        if kind != 'velocity': raise ValueError(f'unknown kind {kind!r}')
        da, ds = self.rates(t)                      # velocity = d x_t / dt
        return da * x0 + ds * eps

    def x0_eps(self, kind, out, x, t):
        """The (x0, eps) that a prediction `out` of a kind implies at (x, t).

        Defined in :numref:`sec_diffusion-gaussian-paths`"""
        t = t.reshape(-1, *[1] * (x.dim() - 1))
        a, s = self.alpha(t), self.sigma(t)
        if kind == 'eps': return (x - s * out) / a, out
        if kind == 'x0': return out, (x - a * out) / s
        if kind == 'v': return ((a * x - s * out) / (a**2 + s**2),
                                (s * x + a * out) / (a**2 + s**2))
        if kind == 'score': return (x + s ** 2 * out) / a, -s * out
        if kind != 'velocity': raise ValueError(f'unknown kind {kind!r}')
        da, ds = self.rates(t)
        det = a * ds - s * da
        return (ds * x - s * out) / det, (a * out - da * x) / det

    def time_at(self, log_snr):
        """The time with a given log-SNR, by bisection (lambda_t falls in t).

        Defined in :numref:`sec_diffusion-gaussian-paths`"""
        lo, hi = torch.zeros_like(log_snr), torch.ones_like(log_snr)
        for _ in range(50):
            mid = (lo + hi) / 2
            later = self.log_snr(mid) > log_snr      # the answer lies after mid
            lo, hi = torch.where(later, mid, lo), torch.where(later, hi, mid)
        return (lo + hi) / 2

    def sigma_bars(self, steps):
        """sigma_bar = sigma / alpha on the grid of steps; inf where alpha = 0.

        Defined in :numref:`sec_diffusion-gaussian-paths`"""
        a, s = self.grid(steps)
        return [s_k / a_k if a_k > 0 else math.inf for a_k, s_k in zip(a, s)]

def regression_loss(path, kind):
    """Squared error of net(x_t, t) against the target of a kind, t ~ U(0, 1).

    Defined in :numref:`sec_diffusion-gaussian-paths`"""
    def loss_fn(net, x0):
        t = torch.rand(len(x0), device=x0.device)
        eps = torch.randn_like(x0)
        error = net(path.sample(x0, t, eps), t) - path.target(kind, x0, eps, t)
        return (error ** 2).flatten(1).sum(1).mean()  # ||error||^2 per example
    return loss_fn

def make_pred(net, path, kind):
    """pred(x, alpha, sigma) -> (x0, eps) from net(x, t) of a kind on path.

    Defined in :numref:`sec_diffusion-gaussian-paths`"""
    def pred(x, alpha, sigma):
        log_snr = 2 * math.log(alpha / sigma) if alpha > 0 else -math.inf
        t = path.time_at(torch.tensor(log_snr, dtype=torch.float64))
        x = x * (path.sigma(t).item() / sigma)  # the same point on our path
        t = torch.full((len(x),), t.item(), device=x.device)
        return path.x0_eps(kind, net(x, t), x, t)
    return pred

def pred_bar(pred, xb, r):
    """pred at x = alpha x_bar for sigma_bar = r; at r = inf, x_bar is noise.

    Defined in :numref:`sec_diffusion-gaussian-paths`"""
    if math.isinf(r):
        return pred(xb, 0.0, 1.0)
    a = 1 / math.sqrt(1 + r ** 2)
    return pred(xb * a, a, r * a)

@torch.no_grad()
def ode_sample(pred, x, sigma_bars):
    """Euler (= DDIM) on d x_bar / d sigma_bar = eps, from x to data; x is the
    noisy point at the first level, unit-variance scale (noise if inf).

    Defined in :numref:`sec_diffusion-gaussian-paths`"""
    r0 = sigma_bars[0]
    xb = x if math.isinf(r0) else x * math.sqrt(1 + r0 ** 2)  # x / alpha
    for r_t, r_s in zip(sigma_bars[:-1], sigma_bars[1:]):
        x0, eps = pred_bar(pred, xb, r_t)
        xb = x0 + r_s * eps                    # the Euler (= DDIM) step
    return xb

@torch.no_grad()
def ancestral_sample(pred, x, path, steps, *, t_start=1.0):
    """Ho et al.'s sampler from x at t_start (1 is noise) to data at t = 0.

    Defined in :numref:`sec_diffusion-gaussian-paths`"""
    a, s = path.grid(steps, t_start)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        ratio = s[i + 1] * a[i] / (a[i + 1] * s[i])
        x = a[i + 1] * x0 + s[i + 1] * ratio * eps      # the posterior mean
        if s[i + 1] > 0:                                # no noise at t = 0
            var = s[i] ** 2 - (a[i] / a[i + 1] * s[i + 1]) ** 2  # forward step
            x = x + math.sqrt(var) * torch.randn_like(x)
    return x

class UNetBlock(nn.Module):
    """Two 3x3 convolutions with group norm, a residual, and an embedding.

    Defined in :numref:`sec_diffusion-images`"""
    def __init__(self, c_in, c_out, emb_dim=None):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, c_out) if emb_dim else None
        self.norm2 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = (nn.Conv2d(c_in, c_out, 1) if c_in != c_out
                     else nn.Identity())

    def forward(self, x, emb=None):
        h = self.conv1(F.silu(self.norm1(x)))
        if self.emb is not None:
            h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    """A small U-Net on time t, a label y (num_classes = null), and cond.

    Defined in :numref:`sec_diffusion-images`"""
    def __init__(self, ch=64, num_classes=None, in_channels=1,
                 out_channels=1, emb_dim=128):
        super().__init__()
        self.emb_dim, self.num_classes = emb_dim, num_classes
        self.label_emb = (nn.Embedding(num_classes + 1, emb_dim)
                          if num_classes else None)
        self.emb_mlp = d2l.make_mlp([emb_dim] * 3, nn.SiLU())
        self.inc = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down1 = UNetBlock(ch, ch, emb_dim)                   # 28 x 28
        self.pool1 = nn.Conv2d(ch, ch, 3, stride=2, padding=1)    # 14 x 14
        self.down2 = UNetBlock(ch, 2 * ch, emb_dim)
        self.pool2 = nn.Conv2d(2 * ch, 2 * ch, 3, stride=2, padding=1)  # 7 x 7
        self.mid = UNetBlock(2 * ch, 2 * ch, emb_dim)
        self.up1 = UNetBlock(4 * ch, ch, emb_dim)                 # 14 x 14
        self.up0 = UNetBlock(2 * ch, ch, emb_dim)                 # 28 x 28
        self.outc = nn.Sequential(nn.GroupNorm(8, ch), nn.SiLU(),
                                  nn.Conv2d(ch, out_channels, 3, padding=1))

    def forward(self, x, t, y=None, cond=None):
        emb = d2l.time_embedding(t, self.emb_dim)
        if y is not None:
            emb = emb + self.label_emb(y)
        if cond is not None:
            emb = emb + cond
        emb = self.emb_mlp(emb)
        up = lambda h, skip: torch.cat([F.interpolate(
            h, size=skip.shape[-2:], mode='nearest-exact'), skip], 1)
        h1 = self.down1(self.inc(x), emb)
        h2 = self.down2(self.pool1(h1), emb)
        h = self.mid(self.pool2(h2), emb)
        h = self.up1(up(h, h2), emb)  # skip connection
        h = self.up0(up(h, h1), emb)  # skip connection
        return self.outc(h)

def cfg(net, c, gamma):
    """Classifier-free guidance: (x, t) -> u + gamma (u_c - u), u unlabeled.

    Defined in :numref:`sec_diffusion-images`"""
    null = torch.full_like(c, net.num_classes)
    def guided(x, t):
        out_c, out = net(torch.cat([x, x]), torch.cat([t, t]),
                         torch.cat([c, null])).chunk(2)
        return out + gamma * (out_c - out)
    return guided

def fashion_feature_cnn(X, y):
    """The feature network that judges Fashion-MNIST samples in this chapter.

    Defined in :numref:`sec_diffusion-images`"""
    cnn = d2l.FeatureCNN(resolution=28).to(X.device)
    d2l.train_feature_cnn(cnn, X.expand(-1, 3, -1, -1), y)
    return cnn

def fd_features(cnn, imgs):
    """Features of gray images (n, 1, 28, 28) in [-1, 1], on the CPU.

    Defined in :numref:`sec_diffusion-images`"""
    rgb = lambda x: x.clamp(-1, 1).expand(-1, 3, -1, -1)
    return d2l.batched(lambda x: cnn.features(rgb(x)), imgs).cpu()

@torch.no_grad()
def ddim_sample(pred, x, path, steps, eta=0.0, *, t_start=1.0, t_end=0.0):
    """DDIM from x at t_start (1 is noise) to t_end; eta = 0 is deterministic.
    eta = 1 draws the posterior variance: not the ancestral sampler's.

    Defined in :numref:`sec_diffusion-samplers`"""
    a, s = path.grid(steps, t_start, t_end)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        ratio = s[i + 1] * a[i] / (a[i + 1] * s[i])     # kappa of the text
        var = (eta * s[i + 1]) ** 2 * (1 - ratio ** 2)   # fresh noise
        x = a[i + 1] * x0 + math.sqrt(s[i + 1] ** 2 - var) * eps
        if var > 0:
            x = x + math.sqrt(var) * torch.randn_like(x)
    return x

def edm_grid(steps, sigma_max=80.0, sigma_min=0.002, rho=7):
    """Karras et al.'s levels sigma_bar: even in sigma_bar^(1/rho), then 0.

    Defined in :numref:`sec_diffusion-samplers`"""
    hi, lo = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return [(hi + i / (steps - 1) * (lo - hi)) ** rho
            for i in range(steps)] + [0.0]

@torch.no_grad()
def heun_sample(pred, x, sigma_bars):
    """Heun on d x_bar / d sigma_bar = eps: an Euler trial, then the average
    of the two slopes; from x at the first level to data, as ode_sample.

    Defined in :numref:`sec_diffusion-samplers`"""
    r0 = sigma_bars[0]
    xb = x if math.isinf(r0) else x * math.sqrt(1 + r0 ** 2)  # x / alpha
    for r_t, r_s in zip(sigma_bars[:-1], sigma_bars[1:]):
        x0, eps = d2l.pred_bar(pred, xb, r_t)
        x_next = x0 + r_s * eps                       # the Euler step
        if r_s > 0 and not math.isinf(r_t):           # no slope at 0 or inf
            x_next = xb + (r_s - r_t) * (
                eps + d2l.pred_bar(pred, x_next, r_s)[1]) / 2
        xb = x_next
    return xb

@torch.no_grad()
def ddim_invert(pred, x, path, steps, t_end=1.0):
    """Deterministic DDIM run backwards, from data x at t = 0 to t_end.

    Defined in :numref:`sec_diffusion-samplers`"""
    a, s = path.grid(steps, 0.0, t_end)
    for i in range(steps):
        eps = pred(x, a[i + 1], s[i + 1])[1]    # x read at the next level
        x = a[i + 1] * (x - s[i] * eps) / a[i] + s[i + 1] * eps
    return x

class Autoencoder(nn.Module):
    """Residual convolutional autoencoder from 28x28x1 images to a 7x7
    Gaussian latent and back; returns the reconstruction and the KL term.

    Defined in :numref:`sec_diffusion-latent`"""
    def __init__(self, latent_channels=4, ch=64, num_blocks=2):
        super().__init__()
        blocks = lambda a, b: [d2l.UNetBlock(a, b)] + [
            d2l.UNetBlock(b, b) for _ in range(num_blocks - 1)]
        down = lambda c: nn.Conv2d(c, c, 3, stride=2, padding=1)
        head = lambda a, b: [nn.GroupNorm(8, a), nn.SiLU(),
                             nn.Conv2d(a, b, 3, padding=1)]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, ch, 3, padding=1), *blocks(ch, ch),
            down(ch), *blocks(ch, 2 * ch),                       # 14 x 14
            down(2 * ch), *blocks(2 * ch, 2 * ch),               # 7 x 7
            *head(2 * ch, 2 * latent_channels))  # mean and log variance
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 2 * ch, 3, padding=1),
            *blocks(2 * ch, 2 * ch),
            nn.Upsample(scale_factor=2), *blocks(2 * ch, ch),    # 14 x 14
            nn.Upsample(scale_factor=2), *blocks(ch, ch),        # 28 x 28
            *head(ch, 1))

    def encode(self, x):
        mean, log_var = self.encoder(x).chunk(2, dim=1)
        return mean, log_var.clamp(-30, 20)

    def forward(self, x):
        mean, log_var = self.encode(x)
        z = mean + (0.5 * log_var).exp() * torch.randn_like(mean)
        kl = 0.5 * (mean ** 2 + log_var.exp() - 1 - log_var).mean()
        return self.decoder(z), kl

def train_step_gflops(net, loss_fn, shape, batch_size=128):
    """GFLOP of one training step, forward and backward, on a batch of zeros
    of the given shape.

    Defined in :numref:`sec_diffusion-latent`"""
    from torch.utils.flop_counter import FlopCounterMode  # not in the preamble
    x = torch.zeros(batch_size, *shape, device=next(net.parameters()).device)
    with FlopCounterMode(display=False) as counter:
        loss_fn(net, x).backward()
    return counter.get_total_flops() / 1e9

class DiTBlock(d2l.ViTBlock):
    """ViTBlock with adaLN-Zero: the condition c sets a shift, a scale and a
    gate for each sublayer; the gates start at zero.

    Defined in :numref:`sec_diffusion-latent`"""
    def __init__(self, num_hiddens, mlp_num_hiddens, num_heads):
        super().__init__(num_hiddens, mlp_num_hiddens, num_heads, dropout=0.0,
                         use_bias=True)
        ln = lambda: nn.LayerNorm(num_hiddens, elementwise_affine=False,
                                  eps=1e-6)
        self.norm1, self.norm2 = ln(), ln()
        self.ada = nn.Sequential(nn.SiLU(),
                                 nn.Linear(num_hiddens, 6 * num_hiddens))
        nn.init.zeros_(self.ada[1].weight), nn.init.zeros_(self.ada[1].bias)

    def forward(self, X, c):
        shift1, scale1, gate1, shift2, scale2, gate2 = (
            self.ada(c)[:, None].chunk(6, dim=-1))
        X_mod = self.norm1(X) * (1 + scale1) + shift1
        X = X + gate1 * self.attention(X_mod, X_mod, X_mod)
        return X + gate2 * self.ffn(self.norm2(X) * (1 + scale2) + shift2)

class DiT(nn.Module):
    """Diffusion transformer conditioned on a time t in [0, 1] and, with
    num_classes, on a label y; label num_classes means no label.

    Defined in :numref:`sec_diffusion-latent`"""
    def __init__(self, img_size=28, patch_size=4, num_channels=1, width=256,
                 depth=6, num_heads=4, num_classes=None):
        super().__init__()
        self.width, self.num_classes = width, num_classes
        self.p, self.g = patch_size, img_size // patch_size
        self.patch = d2l.PatchEmbedding(img_size, patch_size, width,
                                        num_channels)
        self.pos = nn.Parameter(0.02 * torch.randn(1, self.g ** 2, width))
        self.time_mlp = d2l.make_mlp([width] * 3, nn.SiLU())
        self.label_emb = (nn.Embedding(num_classes + 1, width)
                          if num_classes else None)
        self.blks = nn.ModuleList([DiTBlock(width, 4 * width, num_heads)
                                   for _ in range(depth)])
        self.ln = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.out = nn.Linear(width, patch_size ** 2 * num_channels)
        for m in (self.ada[1], self.out):
            nn.init.zeros_(m.weight), nn.init.zeros_(m.bias)

    def forward(self, x, t, y=None):
        c = self.time_mlp(d2l.time_embedding(t, self.width))
        if y is not None:
            c = c + self.label_emb(y)
        X = self.patch(x) + self.pos
        for blk in self.blks:
            X = blk(X, c)
        shift, scale = self.ada(c)[:, None].chunk(2, dim=-1)
        X = self.out(self.ln(X) * (1 + scale) + shift)  # (n, g*g, p*p*c)
        n, g, p = len(x), self.g, self.p  # tokens back to pixels
        X = X.reshape(n, g, g, p, p, -1).permute(0, 5, 1, 3, 2, 4)
        return X.reshape(n, -1, g * p, g * p)

def flow_map_error(u, teacher, x0, eps, s, t):
    """Per-example squared error of the average velocity u(x_t, s, t) against
    the stop-gradient target v - (t - s) du/dt; v from teacher, or eps - x0.

    Defined in :numref:`sec_diffusion-few-step`"""
    x_t = d2l.GaussianPath.linear().sample(x0, t, eps)
    if teacher is None:
        v = eps - x0
    else:
        with torch.no_grad():
            v = teacher(x_t, t)
    u_t, du_dt = torch.func.jvp(lambda x, r: u(x, s, r), (x_t, t),
                                (v, torch.ones_like(t)))
    gap = (t - s).reshape(-1, *[1] * (x0.dim() - 1))
    target = (v - gap * du_dt).detach()
    return ((u_t - target) ** 2).flatten(1).sum(1)

def flow_map_sample(u, x, steps, t_start=1.0):
    """Jump from x at t_start (1 is noise) to data at t = 0 in equal steps.

    Defined in :numref:`sec_diffusion-few-step`"""
    ts = torch.linspace(t_start, 0, steps + 1, device=x.device)
    for t, s in zip(ts[:-1], ts[1:]):
        x = x - (t - s) * u(x, s.expand(len(x)), t.expand(len(x)))
    return x

class UNetFlowMap(nn.Module):
    """u(x, s, t): a copy of a velocity U-Net, plus a zero-initialized gap.

    Defined in :numref:`sec_diffusion-few-step`"""
    def __init__(self, unet):
        super().__init__()
        import copy  # the d2l library preamble does not import it
        # the teacher is an EMA copy with gradients off; the student trains
        self.unet = copy.deepcopy(unet).requires_grad_(True)
        self.gap = nn.Linear(unet.emb_dim, unet.emb_dim)
        nn.init.zeros_(self.gap.weight), nn.init.zeros_(self.gap.bias)

    def forward(self, x, s, t):
        gap = self.gap(d2l.time_embedding(t - s, self.unet.emb_dim))
        return self.unet(x, t, cond=gap)

def safetensors_header(model_id, revision=None, timeout=30):
    """Shape and dtype of every tensor of a Hub checkpoint, read by HTTP
    range requests without fetching tensor data (in production:
    `huggingface_hub.get_safetensors_metadata`).

    Defined in :numref:`sec_lm_hf_models`"""
    import urllib.error
    import urllib.request
    import huggingface_hub
    revision = revision or d2l.HF_REVISIONS.get(model_id)
    if revision is None:
        revision = huggingface_hub.HfApi().model_info(model_id).sha
        print(f'unpinned: {model_id}@{revision}')
    token = huggingface_hub.get_token()  # for gated repositories

    class HubRedirect(urllib.request.HTTPRedirectHandler):
        """Keep the token on redirects within the Hub, not to the CDN."""
        def redirect_request(self, request, *args):
            new = super().redirect_request(request, *args)
            auth = request.unredirected_hdrs.get('Authorization')
            if new and auth and new.host == request.host:
                new.add_unredirected_header('Authorization', auth)
            return new

    opener = urllib.request.build_opener(HubRedirect)

    def fetch(filename, start=None, stop=None):  # bytes start, ..., stop - 1
        request = urllib.request.Request(huggingface_hub.hf_hub_url(
            model_id, filename, revision=revision))
        if start is not None:
            request.add_header('Range', f'bytes={start}-{stop - 1}')
        if token:  # unredirected: dropped when a redirect leaves the Hub
            request.add_unredirected_header('Authorization', f'Bearer {token}')
        with opener.open(request, timeout=timeout) as response:
            if start is None or response.status == 206:
                return response.read()
            return response.read(stop)[start:]  # the server ignored the range

    try:  # a sharded checkpoint lists its shards in an index
        index = json.loads(fetch('model.safetensors.index.json'))
        shards = sorted(set(index['weight_map'].values()))
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
        shards = ['model.safetensors']
    tensors = {}
    for shard in shards:
        n = int.from_bytes(fetch(shard, 0, 8), 'little')  # header length
        header = json.loads(fetch(shard, 8, 8 + n))
        header.pop('__metadata__', None)
        tensors.update({name: (tuple(t['shape']), t['dtype'])
                        for name, t in header.items()})
    return tensors

HF_REVISIONS = {
    'Qwen/Qwen3-0.6B': 'c1899de289a04d12100db370d81485cdf75e47ca',
    'Qwen/Qwen3-0.6B-Base': 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd',
    'Qwen/Qwen3-1.7B': '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
    'Qwen/Qwen3-1.7B-Base': 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1',
    'Qwen/Qwen3-4B': '1cfa9a7208912126459214e8b04321603b3df60c',
    'Qwen/Qwen3-8B': 'b968826d9c46dd6066d109eabc6255188de91218',
    'Qwen/Qwen3.5-0.8B': '2fc06364715b967f1860aea9cf38778875588b17',
    'Qwen/Qwen3.5-0.8B-Base': 'dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68',
    'Qwen/Qwen3.5-2B': '15852e8c16360a2fea060d615a32b45270f8a8fc',
    'Qwen/Qwen3.5-4B': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
    'google/gemma-4-E2B-it': '4adf15748de9e1a0097b6274cee9c15836c41c37'}

LM_DATA_URL = 'https://d2l-data.smola.org/lm/'

d2l.DATA_HUB['lm-hf-memory-speed-v1'] = (
    d2l.LM_DATA_URL + 'lm-hf-memory-speed-v1.json',
    'fce26dfa4ddb9d0463fcfef3ddcb3861ec7226d7')

_TEXT_KEY_MAPPING = {  # checkpoint key renames for a multimodal file's text
    'gemma4_text': {r'^model\.language_model\.': 'model.'}}

def load_hf(model_id, revision=None, dtype=torch.bfloat16, device_map='auto'):
    """A pinned causal language model from the Hugging Face Hub, text part
    only, in eval mode, and its tokenizer; prints a load report. The
    tokenizer's end token joins the stop tokens of `generate`.

    Defined in :numref:`sec_lm_hf_models`"""
    import collections
    import warnings
    import transformers
    if revision is None:
        revision = d2l.HF_REVISIONS.get(model_id)
    if revision is None:
        import huggingface_hub
        revision = huggingface_hub.HfApi().model_info(model_id).sha
        print(f'unpinned: {model_id}@{revision}')
    if device_map == 'auto':  # one GPU: everything on it; none: the CPU
        n_gpus = torch.cuda.device_count()
        device_map = ('cpu' if n_gpus == 0 else {'': 0} if n_gpus == 1
                      else 'auto')
    pin = dict(revision=revision, trust_remote_code=False)
    config = transformers.AutoConfig.from_pretrained(model_id, **pin)
    text = config.get_text_config()  # the language model's configuration
    renames = {}
    if text is not config and text.model_type in _TEXT_KEY_MAPPING:
        renames['key_mapping'] = _TEXT_KEY_MAPPING[text.model_type]
    model, info = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, config=text, dtype=dtype, device_map=device_map,
        use_safetensors=True, output_loading_info=True, **pin, **renames)
    if info['missing_keys']:  # these modules would stay randomly initialized
        del model  # a kept traceback would otherwise hold its memory
        raise RuntimeError(f'no weights for {sorted(info["missing_keys"])}')
    model.revision = revision
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **pin)
    # Without generation_config.json, `generate` stops only at the
    # configuration's end token, which need not be the one ending a chat turn
    stop = model.generation_config.eos_token_id
    stop = ([] if stop is None else [stop] if isinstance(stop, int)
            else list(stop))
    if tokenizer.eos_token_id not in [*stop, None]:
        stop.append(tokenizer.eos_token_id)
        model.generation_config.eos_token_id = stop
    unexpected = collections.Counter(  # keys in the file the model ignores
        '.'.join(key.split('.')[:2]) for key in info['unexpected_keys'])
    print(f'{model_id}@{revision[:12]}: {type(model).__name__}, 0 missing '
          f'keys, unexpected keys: {dict(unexpected) or "none"}, stop tokens '
          f'{stop}')
    devices = set(getattr(model, 'hf_device_map', {}).values())
    if len(devices) > 1:
        warnings.warn(f'{model_id} is spread over {sorted(map(str, devices))}')
    return model.eval(), tokenizer

def _hf_text_config(config):
    """A configuration and its text part, as plain dicts.

    Defined in :numref:`sec_lm_hf_models`"""
    c = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    return c, c.get('text_config') or c

def _hf_layers(t):
    """Kind, head dimension, KV heads and KV sharing of every layer.

    Defined in :numref:`sec_lm_hf_models`"""
    L, H = t['num_hidden_layers'], t['num_attention_heads']
    first_shared = L - (t.get('num_kv_shared_layers') or 0)
    overrides = {int(i): o
                 for i, o in (t.get('per_layer_config') or {}).items()}
    layers = []
    for i, kind in enumerate(t.get('layer_types') or ['full_attention'] * L):
        d_h = t.get('head_dim') or t['hidden_size'] // H
        H_kv = t.get('num_key_value_heads') or H
        if kind == 'full_attention':  # Gemma 4's global layers are wider
            d_h = t.get('global_head_dim') or d_h
            if t.get('attention_k_eq_v'):
                H_kv = t.get('num_global_key_value_heads') or H_kv
        o = overrides.get(i, {})
        layers.append((kind, o.get('head_dim', d_h),
                       o.get('num_key_value_heads', H_kv), i >= first_shared))
    return layers

def config_params(config, include_vision=False, include_mtp=False,
                  by_group=False):
    """Parameters of a dense or hybrid model (Qwen3, Qwen3.5, Gemma 4),
    counted from its configuration; other layouts raise.

    Defined in :numref:`sec_lm_hf_models`"""
    c, t = _hf_text_config(config)
    if (t.get('num_experts') or t.get('n_routed_experts')
            or t.get('enable_moe_block')):
        raise NotImplementedError('use d2l.config_stats (14.1)')
    known = ('qwen3', 'qwen3_5_text', 'gemma4_text')  # QK norms, no biases
    if (t['model_type'] not in known or t.get('attention_bias')
            or t.get('mlp_bias')):
        raise NotImplementedError(f'{t["model_type"]} layout, not one of '
                                  f'{known} without biases')
    d, f, H, V = (t['hidden_size'], t['intermediate_size'],
                  t['num_attention_heads'], t['vocab_size'])
    gemma = t['model_type'] == 'gemma4_text'
    gate = 2 if (t.get('attn_output_gate')  # gated attention output
                 or t['model_type'] == 'qwen3_5_text') else 1
    g = dict.fromkeys(['embed', 'per_layer_embed', 'attention',
                       'linear_attention', 'mlp', 'per_layer_proj', 'norm',
                       'lm_head', 'vision', 'mtp'], 0)
    g['embed'] = V * d
    g['lm_head'] = 0 if t.get('tie_word_embeddings', True) else V * d

    def attention(kind, d_h, H_kv, shared):  # Q (+ gate), K, V, O, QK norms
        n_kv = (0 if shared else
                1 if t.get('attention_k_eq_v') and kind == 'full_attention'
                else 2)
        norms = d_h * (1 if shared else 2)
        return gate * d * H * d_h + n_kv * d * H_kv * d_h + H * d_h * d + norms

    for kind, d_h, H_kv, shared in _hf_layers(t):
        if kind == 'linear_attention':  # Gated DeltaNet
            H_v, d_v = t['linear_num_value_heads'], t['linear_value_head_dim']
            K = t['linear_num_key_heads'] * t['linear_key_head_dim']
            U, C = H_v * d_v, 2 * K + H_v * d_v  # C: convolution channels
            g['linear_attention'] += (d * C + 2 * d * U + 2 * d * H_v + 2 * H_v
                                      + C * t['linear_conv_kernel_dim'] + d_v)
        else:
            g['attention'] += attention(kind, d_h, H_kv, shared)
        wide = 2 if shared and t.get('use_double_wide_mlp') else 1
        g['mlp'] += 3 * d * f * wide
        g['norm'] += (4 if gemma else 2) * d
    g['norm'] += d  # final norm
    p, L = t.get('hidden_size_per_layer_input') or 0, t['num_hidden_layers']
    if p:  # Gemma 4: per-layer embeddings, their projections and norms
        g['per_layer_embed'] = t['vocab_size_per_layer_input'] * L * p
        g['per_layer_proj'] = d * L * p + L * 2 * d * p
        g['norm'] += p + L * d
    if include_mtp and t.get('mtp_num_hidden_layers'):  # Qwen3.5
        block = attention('full_attention', t.get('head_dim') or d // H,
                          t['num_key_value_heads'], False) + 3 * d * f + 2 * d
        g['mtp'] = t['mtp_num_hidden_layers'] * block + 2 * d * d + 3 * d
    v = c.get('vision_config')
    if include_vision and v:
        if (v['model_type'] not in ('qwen3_5', 'qwen3_5_vision')
                or v.get('deepstack_visual_indexes')):
            raise NotImplementedError(f'vision tower of {v["model_type"]}')
        D, I, m = v['hidden_size'], v['intermediate_size'], (
            v['spatial_merge_size'] ** 2)
        patch = (v['in_channels'] * v['temporal_patch_size']
                 * v['patch_size'] ** 2)
        block = 4 * D * D + 2 * D * I + 9 * D + I  # with biases, LayerNorms
        g['vision'] = ((patch + 1 + v['num_position_embeddings']) * D
                       + v['depth'] * block + 2 * D
                       + (D * m + 1) * (D * m + v['out_hidden_size']))
    return g if by_group else sum(g.values())

def config_memory(config, seq_len, batch_size, dtype_bytes=2, *,
                  state_bytes=4, conv_bytes=None, conv_taps=None,
                  window_cached=None, include_vision=False, include_mtp=False,
                  logits_to_keep=1):
    """Bytes to run a model on `batch_size` sequences of `seq_len` tokens.

    The defaults store what Transformers 5.22.2 stores: a sliding layer keeps
    `sliding_window - 1` positions, a linear-attention layer all
    `linear_conv_kernel_dim` convolution inputs. `activations` estimates one
    layer's tensors with fused attention; `logits_to_keep` of None or 0 keeps
    the logits of every position.

    Defined in :numref:`sec_lm_hf_models`"""
    import warnings
    c, t = _hf_text_config(config)
    n, B, b = seq_len, batch_size, dtype_bytes
    if n > t.get('max_position_embeddings', n):
        warnings.warn(f'seq_len {n} exceeds max_position_embeddings '
                      f'{t["max_position_embeddings"]}')
    if window_cached is None and t.get('sliding_window'):
        window_cached = t['sliding_window'] - 1
    conv_bytes = b if conv_bytes is None else conv_bytes
    if conv_taps is None:
        conv_taps = t.get('linear_conv_kernel_dim')
    kv = state = width = 0  # per sequence; width: Q, K, V entries per token
    for kind, d_h, H_kv, shared in _hf_layers(t):
        if kind == 'linear_attention':  # fixed matrix state + conv inputs
            H_v, d_v = t['linear_num_value_heads'], t['linear_value_head_dim']
            C = 2 * t['linear_num_key_heads'] * t['linear_key_head_dim'] + (
                H_v * d_v)
            state += (H_v * t['linear_key_head_dim'] * d_v * state_bytes
                      + C * conv_taps * conv_bytes)
            continue
        width = max(width, (t['num_attention_heads'] + 2 * H_kv) * d_h)
        if not shared:  # a KV-sharing layer reads an earlier layer's cache
            kept = min(n, window_cached) if kind == 'sliding_attention' else n
            kv += 2 * H_kv * d_h * b * kept
    f = t['intermediate_size'] * (2 if t.get('use_double_wide_mlp')
                                  and t.get('num_kv_shared_layers') else 1)
    params = config_params(config, include_vision, include_mtp)
    n_out = min(logits_to_keep, n) if logits_to_keep else n
    memory = {'params': params, 'weights': params * b,
              'kv_cache': B * kv, 'recurrent_state': B * state,
              'activations': B * n * b * (2 * t['hidden_size'] + 3 * f
                                          + width),
              'logits': B * n_out * t['vocab_size'] * b}
    memory['total'] = sum(memory.values()) - params
    return memory

d2l.DATA_HUB['lm-qwen3-port-prompts-v1'] = (
    d2l.LM_DATA_URL + 'lm-qwen3-port-prompts-v1.json',
    'b5840fad4ad3d8d1897c12d522def18f6eb8fa06')
d2l.DATA_HUB['lm-qwen3-0p6b-logit-ref-v1'] = (
    d2l.LM_DATA_URL + 'lm-qwen3-0p6b-logit-ref-v1.pt',
    '2fa5449638a8a8924bc6b35384dc996c23513cb6')

def qwen3_rope_tables(n, head_dim, theta, offset=0, device=None):
    """Cosines and sines, each (n, head_dim / 2) in fp32, of the RoPE angles
    of positions offset, ..., offset + n - 1, with the frequencies computed
    as Transformers computes them.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    # 1 / theta^(2i/d) on the CPU, as Transformers builds its buffer; the
    # book's theta^(-2i/d) is equal in exact arithmetic but not in fp32
    inv = 1.0 / float(theta) ** (torch.arange(
        0, head_dim, 2, dtype=torch.float32, device='cpu') / head_dim)
    pos = torch.arange(n, dtype=torch.float32, device=device) + offset
    angles = pos[:, None] * inv.to(pos.device)[None, :]
    return torch.cos(angles), torch.sin(angles)

class Qwen3Attention(d2l.CausalAttention):
    """Causal attention with separate projections, an explicit head width,
    optional RMSNorm of queries and keys per head before RoPE (QK-norm),
    and a step that reads and writes a key-value cache.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    def __init__(self, num_hiddens, num_heads, num_kv_heads, *, head_dim=None,
                 qk_norm=False, rope_theta=10000.0, norm_eps=1e-6, bias=False,
                 rope=True, causal=True):
        nn.Module.__init__(self)  # CausalAttention's would allocate W_qkv
        if not causal:
            raise ValueError('Qwen3Attention is causal only')
        self.num_heads, self.num_kv_heads = num_heads, num_kv_heads
        self.head_dim = head_dim or num_hiddens // num_heads
        self.qk_norm, self.rope, self.rope_theta = qk_norm, rope, rope_theta
        self.kv_hiddens = num_kv_heads * self.head_dim
        q_hiddens = num_heads * self.head_dim  # may differ from num_hiddens
        self.W_q = nn.Linear(num_hiddens, q_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, self.kv_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, self.kv_hiddens, bias=bias)
        self.W_o = nn.Linear(q_hiddens, num_hiddens, bias=bias)
        if qk_norm:  # one gain per coordinate of a head, shared by the heads
            self.q_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=norm_eps)

    def qkv(self, X, offset=0, tables=None):
        B, T, _ = X.shape
        q = self.W_q(X).reshape(B, T, self.num_heads, self.head_dim)
        k = self.W_k(X).reshape(B, T, self.num_kv_heads, self.head_dim)
        v = self.W_v(X).reshape(B, T, self.num_kv_heads, self.head_dim)
        if self.qk_norm:  # normalize each head's query and key
            q, k = self.q_norm(q), self.k_norm(k)
        q, k, v = (u.transpose(1, 2) for u in (q, k, v))  # (B, heads, T, d_h)
        if self.rope:
            if tables is None:
                tables = qwen3_rope_tables(T, self.head_dim, self.rope_theta,
                                           offset, X.device)
            q, k = (d2l.rope(u, tables=tables) for u in (q, k))
        return q, k, v

    def forward_step(self, X, cache, tables=None):
        """Attention output for X at positions t, ..., t + T - 1, where cache
        = (ks, vs, t) holds this block's buffers, each (batch, kv heads,
        max_len, head_dim); writes the keys and values of X into them."""
        ks, vs, t = cache
        T = X.shape[1]
        q, k, v = self.qkv(X, t, tables)
        ks[:, :, t:t + T], vs[:, :, t:t + T] = k, v
        Y = F.scaled_dot_product_attention(
            q, ks[:, :, :t + T], vs[:, :, :t + T],
            attn_mask=d2l.causal_mask(T, t + T, X.device),
            enable_gqa=self.num_kv_heads < self.num_heads)
        return self.W_o(Y.transpose(1, 2).flatten(2))

class SwiGLU(d2l.FeedForward):
    """SwiGLU feed-forward network with an explicit hidden width.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    def __init__(self, num_hiddens, hidden, bias=False):
        nn.Module.__init__(self)  # FeedForward's would allocate width 8d/3
        self.act = 'swiglu'
        self.W_g = nn.Linear(num_hiddens, hidden, bias=bias)
        self.W_1 = nn.Linear(num_hiddens, hidden, bias=bias)
        self.W_2 = nn.Linear(hidden, num_hiddens, bias=bias)

class Qwen3GPT(d2l.GPT):
    """d2l.GPT with Qwen3's head width, QK-norm, RoPE base, feed-forward
    width and normalization epsilon, and an optional untied output head.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    def __init__(self, vocab_size, num_hiddens, num_heads, num_kv_heads,
                 num_blks, *, head_dim=None, qk_norm=True, rope_theta=1e6,
                 ffn_hidden=None, norm_eps=1e-6, tie_embeddings=True,
                 max_len=40960, bias=False):
        head_dim = head_dim or num_hiddens // num_heads
        ffn_hidden = ffn_hidden or round(8 * num_hiddens / 3)
        # read by CausalAttention below while GPT builds the blocks; a plain
        # attribute may be set before nn.Module.__init__ runs
        self.attn_options = dict(head_dim=head_dim, qk_norm=qk_norm,
                                 rope_theta=rope_theta, norm_eps=norm_eps)
        super().__init__(
            vocab_size, num_hiddens=num_hiddens, num_heads=num_heads,
            num_blks=num_blks, max_len=max_len, bias=bias,
            num_kv_heads=num_kv_heads,
            ffn_factory=lambda: SwiGLU(num_hiddens, ffn_hidden, bias))
        for blk in self.blks:
            blk.norm1.eps = blk.norm2.eps = norm_eps
        self.norm.eps = norm_eps
        self.head_width, self.rope_theta = head_dim, rope_theta
        if not tie_embeddings:
            self.lm_head = nn.Linear(num_hiddens, vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, std=0.02)

    def CausalAttention(self, d, H, *, bias, rope, num_kv_heads):
        """The attention GPT builds in each block (a method, not a lambda,
        so that the model can be pickled)."""
        return Qwen3Attention(d, H, num_kv_heads, bias=bias, rope=rope,
                              **self.attn_options)

    def forward(self, X, logits_to_keep=None, tables=None):
        if tables is None:  # RoPE angles of positions 0, ..., T - 1
            tables = qwen3_rope_tables(X.shape[1], self.head_width,
                                       self.rope_theta, device=X.device)
        H = self.token_emb(X)
        for blk in self.blks:
            H = blk(H, tables=tables)
        if logits_to_keep is not None:  # the head only where logits are read
            H = H[:, -logits_to_keep:]
        H = self.norm(H)
        if hasattr(self, 'lm_head'):
            return self.lm_head(H)
        return F.linear(H, self.token_emb.weight)

    def init_cache(self, batch_size, max_len):
        """Zeroed keys and values for every block, each (blocks, batch, kv
        heads, max_len, head_dim)."""
        attn = self.blks[0].attention
        shape = (len(self.blks), batch_size, attn.num_kv_heads, max_len,
                 attn.head_dim)
        p = self.token_emb.weight
        return (torch.zeros(shape, device=p.device, dtype=p.dtype),
                torch.zeros(shape, device=p.device, dtype=p.dtype))

    @torch.inference_mode()
    def forward_cached(self, X, ks, vs, t):
        """Logits after the last of the new tokens X, which sit at positions
        t, t + 1, ...; writes their keys and values into the cache."""
        tables = qwen3_rope_tables(X.shape[1], self.head_width,
                                   self.rope_theta, t, X.device)
        H = self.token_emb(X)
        for i, blk in enumerate(self.blks):  # the modules of blk.forward
            H = H + blk.attention.forward_step(blk.norm1(H), (ks[i], vs[i], t),
                                               tables)
            H = H + blk.ffn(blk.norm2(H))
        H = self.norm(H[:, -1])
        if hasattr(self, 'lm_head'):
            return self.lm_head(H), ks, vs
        return F.linear(H, self.token_emb.weight), ks, vs

    @torch.inference_mode()
    def generate_cached(self, prefix, num_tokens, temperature=1.0, top_k=None):
        """Sample as generate does, but never recompute the prefix."""
        assert len(prefix) + num_tokens <= self.max_len, 'cache overflow'
        X = torch.tensor(prefix, device=self.token_emb.weight.device)[None]
        ks, vs = self.init_cache(1, len(prefix) + num_tokens)
        logits, ks, vs = self.forward_cached(X, ks, vs, 0)
        tokens = []
        for t in range(len(prefix), len(prefix) + num_tokens):
            tokens.append(d2l.sample_logits(logits, 'sample', temperature,
                                            k=top_k))
            if t + 1 < len(prefix) + num_tokens:  # logits for the next draw
                logits, ks, vs = self.forward_cached(tokens[-1][:, None], ks,
                                                     vs, t)
        return list(prefix) + torch.cat(tokens).tolist()

def qwen3_gpt_kwargs(cfg):
    """Qwen3GPT's arguments from a Qwen3 configuration (a dict or a
    Transformers config); raises on settings the port does not implement.
    A dict without head_dim, tie_word_embeddings or a RoPE base gets
    Transformers' Qwen3Config defaults (128, untied, 10,000).

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    c = cfg if isinstance(cfg, dict) else cfg.to_dict()
    kind = c.get('model_type')
    if kind == 'qwen3_moe':
        raise NotImplementedError('qwen3_moe: expert layers are not ported')
    if kind and kind.startswith('qwen3_5'):
        raise NotImplementedError(
            f'{kind}: its Gated DeltaNet layers are ported in '
            'sec_lm_hybrids_at_scale')
    if kind != 'qwen3':
        raise ValueError(f'not a Qwen3 configuration: {kind!r}')
    rope = c.get('rope_parameters') or {}  # where Transformers 5 keeps theta
    for name, value in [('use_sliding_window', c.get('use_sliding_window')),
                        ('rope_scaling', c.get('rope_scaling')),
                        ('rope_type', rope.get('rope_type', 'default')
                         != 'default'),
                        ('attention_bias', c.get('attention_bias'))]:
        if value:
            raise ValueError(f'{name} is not implemented in Qwen3GPT')
    return dict(
        vocab_size=c['vocab_size'], num_hiddens=c['hidden_size'],
        num_heads=c['num_attention_heads'],
        num_kv_heads=c['num_key_value_heads'],
        num_blks=c['num_hidden_layers'], head_dim=c.get('head_dim', 128),
        qk_norm=True, rope_theta=float(c.get('rope_theta') or rope.get(
            'rope_theta', 1e4)),
        ffn_hidden=c['intermediate_size'], norm_eps=c['rms_norm_eps'],
        tie_embeddings=c.get('tie_word_embeddings', False),
        max_len=c['max_position_embeddings'], bias=False)

def qwen3_rope_permutation(head_dim):
    """Index idx with idx[2i] = i and idx[2i + 1] = i + head_dim / 2: taking
    row idx[j] of a head's rows as row j turns Transformers' split-half
    RoPE pairs into the book's interleaved pairs.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    half = torch.arange(head_dim // 2)
    return torch.stack([half, half + head_dim // 2], 1).flatten()

def map_qwen3_state_dict(hf_state, cfg, rope_layout='permute'):
    """A Qwen3GPT state dict from a Transformers Qwen3 state dict. Renames
    every tensor and, per head, permutes the rows of q_proj and k_proj and
    the gains of q_norm and k_norm into the book's RoPE pairing
    (rope_layout='none' skips this). A KeyError lists every tensor not
    consumed and every key not produced.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    assert rope_layout in ('permute', 'none'), rope_layout
    kw = qwen3_gpt_kwargs(cfg)
    d_h = kw['head_dim'] or kw['num_hiddens'] // kw['num_heads']
    idx = qwen3_rope_permutation(d_h)

    def permute(W):  # W: (heads * d_h, ...) rows, or one head's (d_h,) gains
        if rope_layout == 'none':
            return W
        return W.unflatten(0, (-1, d_h))[:, idx.to(W.device)].flatten(0, 1)

    keep = lambda W: W
    layer = {'input_layernorm': ('norm1', keep),
             'self_attn.q_proj': ('attention.W_q', permute),
             'self_attn.k_proj': ('attention.W_k', permute),
             'self_attn.v_proj': ('attention.W_v', keep),
             'self_attn.o_proj': ('attention.W_o', keep),
             'self_attn.q_norm': ('attention.q_norm', permute),
             'self_attn.k_norm': ('attention.k_norm', permute),
             'post_attention_layernorm': ('norm2', keep),
             'mlp.gate_proj': ('ffn.W_g', keep),
             'mlp.up_proj': ('ffn.W_1', keep),
             'mlp.down_proj': ('ffn.W_2', keep)}
    names = {'model.embed_tokens': ('token_emb', keep)}
    for i in range(kw['num_blks']):
        names.update({f'model.layers.{i}.{hf}': (f'blks.{i}.{ours}', f)
                      for hf, (ours, f) in layer.items()})
    names['model.norm'] = ('norm', keep)
    if not kw['tie_embeddings']:
        names['lm_head'] = ('lm_head', keep)
    left, state, missing = dict(hf_state), {}, []
    head = left.get('lm_head.weight')
    emb = left.get('model.embed_tokens.weight')
    if kw['tie_embeddings'] and head is not None:  # a stored copy of emb
        if not (head.is_meta or emb is None or torch.equal(head, emb)):
            raise ValueError('lm_head.weight differs from the tied embedding')
        del left['lm_head.weight']
    for hf, (ours, f) in names.items():
        if hf + '.weight' in left:
            state[ours + '.weight'] = f(left.pop(hf + '.weight'))
        else:
            missing.append(f'{ours}.weight (from {hf}.weight)')
    if left or missing:
        raise KeyError(f'not consumed: {sorted(left)}; '
                       f'not produced: {missing}')
    return state

def _qwen3_files(model_id, revision):
    """Local paths of config.json, generation_config.json (None if the
    repository has none) and the safetensors files of a Hub checkpoint at a
    revision, downloaded into the Hugging Face cache.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    import huggingface_hub
    # the Hub answered that the file is absent; LocalEntryNotFoundError
    # (offline, not cached) is a sibling subclass and must propagate
    from huggingface_hub.errors import RemoteEntryNotFoundError

    def get(filename, optional=False):
        try:
            return huggingface_hub.hf_hub_download(model_id, filename,
                                                   revision=revision)
        except RemoteEntryNotFoundError:
            if optional:
                return None
            raise

    index = get('model.safetensors.index.json', optional=True)
    if index is None:
        weights = [get('model.safetensors')]
    else:  # a sharded checkpoint lists its shards in the index
        with open(index) as fh:
            shards = sorted(set(json.load(fh)['weight_map'].values()))
        weights = [get(shard) for shard in shards]
    return (get('config.json'), get('generation_config.json', optional=True),
            weights)

def load_qwen3(model_id='Qwen/Qwen3-0.6B', revision=None, *,
               dtype=torch.bfloat16, device=None):
    """A pinned Qwen3 checkpoint from the Hugging Face Hub as a Qwen3GPT in
    eval mode: built on the meta device, then its mapped weights assigned.
    The tokenizer is loaded separately, at model.revision.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    from safetensors.torch import load_file  # lazy, as in load_gpt2
    revision = revision or d2l.HF_REVISIONS[model_id]
    config_file, generation_file, weight_files = _qwen3_files(model_id,
                                                              revision)
    with open(config_file) as fh:
        cfg = json.load(fh)
    with torch.device('meta'):  # shapes only: no memory is allocated
        model = Qwen3GPT(**qwen3_gpt_kwargs(cfg))
    hf_state = {}
    for path in weight_files:
        hf_state.update(load_file(path))
    device = device or d2l.try_gpu()
    state = {name: W.to(device, dtype)
             for name, W in map_qwen3_state_dict(hf_state, cfg).items()}
    # assign=True: meta tensors cannot be copied into, so take the tensors
    model.load_state_dict(state, strict=True, assign=True)
    model.hf_config, model.revision = cfg, revision
    model.generation_config = None
    if generation_file is not None:
        import transformers
        with open(generation_file) as fh:
            model.generation_config = transformers.GenerationConfig.from_dict(
                json.load(fh))
    return model.eval()

Generation = collections.namedtuple(  # finish_reason: 'eos', 'stop', 'length'
    'Generation', 'text token_ids logprobs finish_reason')

def truncated_probs(logits, temperature=1.0, k=None, p=None, min_p=None):
    """Rows of logits (batch, vocab) -> the temperature-scaled distribution
    truncated by top-k, top-p and min-p and renormalized; ties in rank go
    to the smaller token id. As in Transformers and vLLM (and unlike
    sec_decoding's sample_next), top-p measures mass within the top k.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    probs = F.softmax(logits.float() / temperature, -1)
    ranked, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    keep = torch.full_like(order[:, :1], probs.shape[-1])  # tokens kept
    if k is not None:  # top-k: a fixed count
        keep = keep.clamp(max=k)
    if p is not None:  # top-p: the smallest head of mass at least p of
        cum = ranked.double().cumsum(-1)  # the top-k mass (1 without k)
        head = (cum < p * cum.gather(-1, keep - 1)).sum(-1, keepdim=True) + 1
        keep = torch.minimum(keep, head)
    if min_p is not None:  # min-p: a fraction of the top probability
        keep = torch.minimum(keep, (ranked >= min_p * ranked[:, :1]).sum(
            -1, keepdim=True))
    ranked = ranked * (torch.arange(probs.shape[-1], device=probs.device)
                       < keep)
    probs = torch.zeros_like(probs).scatter(-1, order, ranked)
    return probs / probs.sum(-1, keepdim=True)

def sample_logits(logits, strategy='greedy', temperature=1.0, k=None,
                  p=None, min_p=None, generator=None):
    """One token id per row of logits (batch, vocab): the argmax (ties to
    the smaller id), or a draw from truncated_probs by inverting its
    distribution function at a uniform from generator (a torch.Generator,
    or one per row).

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    if strategy == 'greedy':
        return logits.argmax(-1)
    if strategy != 'sample':
        raise ValueError(f'unknown strategy: {strategy!r}')
    cdf = truncated_probs(logits, temperature, k, p, min_p).double().cumsum(-1)
    if isinstance(generator, (list, tuple)):  # one uniform per row
        u = torch.cat([torch.rand(1, generator=g, dtype=torch.float64)
                       for g in generator])
    else:
        u = torch.rand(len(cdf), generator=generator, dtype=torch.float64)
    u = u.to(cdf.device)[:, None] * cdf[:, -1:]  # in [0, total mass)
    return torch.searchsorted(cdf, u, right=True)[:, 0]

def _lm_cached_steps(model, ids, num_new):
    """Prefill equal-length prompts ids (lists of token ids) into the
    model's cache; returns the last logits and step(next_ids) -> logits.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    X = torch.tensor(ids, device=model.token_emb.weight.device)
    ks, vs = model.init_cache(len(ids), X.shape[1] + num_new)
    t = X.shape[1]
    logits = model.forward_cached(X, ks, vs, 0)[0]

    def step(next_ids):  # next_ids: (batch,), at position t
        nonlocal t
        logits = model.forward_cached(next_ids[:, None], ks, vs, t)[0]
        t += 1
        return logits
    return logits, step

def _lm_recompute_steps(model, ids):
    """For any model mapping token ids to logits: each step runs the whole
    sequence again.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    X = torch.tensor(ids, device=next(model.parameters()).device)

    def step(next_ids):
        nonlocal X
        X = torch.cat([X, next_ids[:, None]], 1)
        return model(X)[:, -1]
    return model(X)[:, -1], step

def _lm_hf_steps(model, ids, pad_id):
    """Prompts of any lengths through a Transformers model and its cache:
    padded on the left, with positions counted from each prompt's first
    token, and logits computed at the last position only.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    T = max(map(len, ids))
    X = torch.tensor([[pad_id] * (T - len(r)) + r for r in ids],
                     device=model.device)
    mask = torch.tensor([[0] * (T - len(r)) + [1] * len(r) for r in ids],
                        device=model.device)
    cache = None

    def run(X):
        nonlocal cache
        pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)[:, -X.shape[1]:]
        out = model(input_ids=X, attention_mask=mask, position_ids=pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = out.past_key_values
        return out.logits[:, -1]

    def step(next_ids):
        nonlocal mask
        mask = torch.cat([mask, mask.new_ones(len(ids), 1)], 1)
        return run(next_ids[:, None])
    return run(X), step

def _lm_finish(tokens, logprobs, tokenizer, eos_ids, stop):
    """One row's Generation: a final end token is dropped, and the text is
    cut before the first stop string.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    reason = 'length'
    if tokens and tokens[-1] in eos_ids:
        tokens, reason = tokens[:-1], 'eos'
    text = tokenizer.decode(tokens)
    cuts = [text.find(s) for s in stop or () if s in text]
    if cuts:
        text, reason = text[:min(cuts)], 'stop'
    if logprobs is not None:
        logprobs = logprobs[:len(tokens)]
    return Generation(text, tokens, logprobs, reason)

def _lm_unwrap(model):
    """The model inside a torch.compile or PEFT wrapper (which pass every
    keyword argument through to it), or model itself.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    while True:
        inner = getattr(model, '_orig_mod', None)
        if inner is None and hasattr(model, 'get_base_model'):
            inner = model.get_base_model()
        if inner is None or inner is model:
            return model
        model = inner

def _lm_bos_ids(tokenizer):
    """The special tokens the tokenizer puts before a text (a BOS, or
    none). Those it appends after a text, such as an EOS, are left out:
    a prompt continues where its text ends.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    with_sp, plain = (tokenizer(['a'], add_special_tokens=s)['input_ids'][0]
                      for s in (True, False))
    return next((with_sp[:k] for k in range(len(with_sp) - len(plain) + 1)
                 if with_sp[k:k + len(plain)] == plain), [])

@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens=None, *,
             strategy='greedy', temperature=1.0, k=None, p=None, min_p=None,
             seed=0, eos_token_ids=None, stop=None, chat=False,
             enable_thinking=None, return_logprobs=False, batch_size=8):
    """Continue each prompt (a string, or with chat=True also a list of
    messages) and return one Generation per prompt, in input order.
    strategy is 'greedy' or 'sample' (truncated_probs's temperature, k, p,
    min_p). stop and eos_token_ids take one item or a list. seed is an
    int or a torch.Generator, either one stream for the call, or a list
    of one int per prompt: an item's random draws then depend only on its
    seed; the
    sampled text is batch-invariant only up to bf16 kernel drift, because
    the logits depend on batch and padding composition
    (:numref:`subsec_lm_nondeterminism`).

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    config = getattr(model, 'generation_config', None)
    if max_new_tokens is None:
        max_new_tokens = getattr(config, 'max_new_tokens', None) or 256
    if eos_token_ids is None:  # the tokenizer's end token and the model's
        eos = getattr(config, 'eos_token_id', None)
        eos_token_ids = [tokenizer.eos_token_id,
                         *(eos if isinstance(eos, list) else [eos])]
    if isinstance(eos_token_ids, (int, np.integer)):
        eos_token_ids = [eos_token_ids]
    eos_ids = {int(i) for i in eos_token_ids if i is not None}
    if isinstance(stop, str):  # one string, not its characters
        stop = [stop]
    if chat:  # a string is one user message; the template adds no BOS
        opts = {} if enable_thinking is None else {
            'enable_thinking': enable_thinking}
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': x}] if isinstance(x, str) else x,
            tokenize=False, add_generation_prompt=True, **opts)
                   for x in prompts]
    head = [] if chat else _lm_bos_ids(tokenizer)  # a BOS, never an EOS
    ids = [head + tokenizer(x, add_special_tokens=False)['input_ids']
           for x in prompts]
    if not all(ids):  # Qwen3's tokenizer adds no BOS to predict from
        raise ValueError('an empty prompt has no position to predict from')
    hf = sys.modules.get('transformers')  # loaded if model is one of its
    if hf is not None and isinstance(_lm_unwrap(model), hf.PreTrainedModel):
        pad = getattr(tokenizer, 'pad_token_id', None) or 0
        start = lambda rows: _lm_hf_steps(model, rows, pad)
        groups = [list(range(len(ids)))]  # left padding: input order
    else:  # no attention mask: batch only prompts of equal length
        if hasattr(model, 'forward_cached'):
            start = lambda rows: _lm_cached_steps(model, rows, max_new_tokens)
        else:
            start = lambda rows: _lm_recompute_steps(model, rows)
        groups = [[i for i in range(len(ids)) if len(ids[i]) == n]
                  for n in sorted({len(r) for r in ids})]
    shared = isinstance(seed, (int, np.integer, torch.Generator))
    if isinstance(seed, torch.Generator):
        generator = seed
    elif shared:
        generator = torch.Generator().manual_seed(int(seed))
    else:
        assert len(seed) == len(ids), 'one seed per prompt'
        generators = [torch.Generator().manual_seed(int(s)) for s in seed]
    out = [None] * len(ids)
    for rows in (g[j:j + batch_size] for g in groups
                 for j in range(0, len(g), batch_size)):
        logits, step = start([ids[i] for i in rows])
        gen = generator if shared else [generators[i] for i in rows]
        tokens = [[] for _ in rows]
        logprobs = [[] for _ in rows] if return_logprobs else None
        active = set(range(len(rows)))
        for n in range(max_new_tokens):
            nxt = sample_logits(logits, strategy, temperature, k, p, min_p,
                                gen)
            if return_logprobs:  # under the untruncated distribution
                lp = F.log_softmax(logits.float(), -1).gather(
                    -1, nxt[:, None])[:, 0].tolist()
            for j, token in enumerate(nxt.tolist()):
                if j in active:
                    tokens[j].append(token)
                    if return_logprobs:
                        logprobs[j].append(lp[j])
                    text = tokenizer.decode(tokens[j]) if stop else ''
                    if token in eos_ids or any(s in text for s in stop or ()):
                        active.discard(j)
            if not active or n == max_new_tokens - 1:
                break
            logits = step(nxt)
        for j, i in enumerate(rows):
            out[i] = _lm_finish(tokens[j], logprobs and logprobs[j],
                                tokenizer, eos_ids, stop)
    return out

_HIDDEN_POINTS = ('attn_out', 'resid_mid', 'mlp_out', 'resid_post')

def collect_hiddens(model, input_ids):
    """Per layer, in execution order: the attention output, the residual
    stream after it, the FFN output, and the stream after the block; each
    (layers, batch, T, d) on the CPU, fp64 for an fp64 model, else fp32.
    model is a d2l.GPT or a Transformers Qwen3 model.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    if isinstance(model, d2l.GPT):
        layers = [(b.attention, b.norm2, b.ffn, b) for b in model.blks]
    else:  # Transformers: attention, second norm and MLP of each layer
        layers = [(b.self_attn, b.post_attention_layernorm, b.mlp, b)
                  for b in model.model.layers]
    dtype = (torch.float64 if next(model.parameters()).dtype == torch.float64
             else torch.float32)
    caps = {name: [] for name in _HIDDEN_POINTS}

    def save(name, x):  # Transformers modules may return tuples
        x = x[0] if isinstance(x, tuple) else x
        caps[name].append(x.detach().to('cpu', dtype))

    handles = []
    try:
        for attn, norm2, ffn, blk in layers:
            handles += [
                attn.register_forward_hook(
                    lambda m, a, y: save('attn_out', y)),
                norm2.register_forward_pre_hook(  # the stream entering norm2
                    lambda m, a: save('resid_mid', a[0])),
                ffn.register_forward_hook(lambda m, a, y: save('mlp_out', y)),
                blk.register_forward_hook(
                    lambda m, a, y: save('resid_post', y))]
        with torch.inference_mode():
            model(input_ids)
    finally:
        for handle in handles:
            handle.remove()
    return {name: torch.stack(caps[name]) for name in _HIDDEN_POINTS}

def layer_errors(ref, ours, skip_first=1):
    """Relative Frobenius error of each capture of collect_hiddens against a
    reference over positions skip_first, ..., T - 1, per layer, capture
    point and prompt: an array (layers, 4, batch).

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    errors = []
    for name in _HIDDEN_POINTS:
        r, o = (u[:, :, skip_first:].double().flatten(2)
                for u in (ref[name], ours[name]))
        errors.append(((o - r).norm(dim=-1) / r.norm(dim=-1)).numpy())
    return np.stack(errors, 1)

def first_divergence(errors, floor, factor=10.0):
    """The first (layer, capture point) in execution order whose median
    error over prompts exceeds factor times the median floor there (the
    clean port's error); None if there is none.

    Defined in :numref:`sec_lm_d2l_checkpoint`"""
    above = np.median(errors, -1) > factor * np.median(floor, -1)
    if not above.any():
        return None
    layer, point = np.argwhere(above)[0]  # row-major: layer by layer
    return int(layer), _HIDDEN_POINTS[point]

def _label_sep(prompt, word):
    """A space between a prompt and a label word, except after whitespace
    and next to a wide (Chinese, Japanese or Korean) character.

    Defined in :numref:`sec_lm_prompting`"""
    import unicodedata
    wide = any(unicodedata.east_asian_width(c) in ('W', 'F')
               for c in prompt[-1:] + word[:1])
    return '' if wide or prompt[-1:].isspace() else ' '

def _label_token_ids(tokenizer, prompts, label_words, sep=None):
    """The prompts' token ids, and per prompt the id of each label word's
    first token as the next token, the word following the prompt after
    sep (by default _label_sep's choice). Raises if (a) the prompt's end
    merges with a word, (b) two words share a first token, (c) a word is
    several tokens, checked in this order.

    Defined in :numref:`sec_lm_prompting`"""
    enc = lambda texts: tokenizer(texts, add_special_tokens=False)[
        'input_ids']
    ids, tails = enc(prompts), [[] for _ in prompts]
    seps = [[_label_sep(p, w) if sep is None else sep for w in label_words]
            for p in prompts]
    for j, w in enumerate(label_words):
        full = enc([p + s[j] + w for p, s in zip(prompts, seps)])
        for i, (p, h, f) in enumerate(zip(prompts, ids, full)):
            if f[:len(h)] != h:  # (a) the prompt ends inside a token
                raise ValueError(
                    f'prompt {i} ends inside a token ({p[-12:]!r} merges '
                    f'with {w!r}): strip trailing whitespace from the '
                    'prompt, the space belongs to the label')
            if len(f) == len(h):
                raise ValueError(f'label word {w!r} adds no token')
            tails[i].append(f[len(h):])
    for t in tails:  # (b) two words, one first token
        first = {}
        for w, s in zip(label_words, t):
            if s[0] in first:
                raise ValueError(
                    f'label words {first[s[0]]!r} and {w!r} share the '
                    f'first token {tokenizer.decode(s[:1])!r} ({s[0]})')
            first[s[0]] = w
    for t, seps_i in zip(tails, seps):  # (c) a word of several tokens
        for w, s, sp in zip(label_words, t, seps_i):
            if len(s) > 1:
                raise ValueError(
                    f'label word {w!r} after {sp!r} is {len(s)} tokens '
                    f'{[tokenizer.decode([u]) for u in s]}: use a word '
                    'that is one token there, or another sep')
    return ids, [[s[0] for s in t] for t in tails]

def label_logprobs(model, tokenizer, prompts, label_words, *, batch_size=8,
                   chat=False, enable_thinking=False, sep=None):
    """Log-probabilities (len(prompts), len(label_words)), float32 on the
    CPU, of each label word's first token as the next token, normalized
    over the label tokens only. A word follows its prompt after sep, by
    default a space, but none after whitespace or next to Chinese text.
    With chat=True a prompt is a message list (a string is one user
    message), rendered with the generation prefix. Transformers models
    get left-padded batches; a d2l GPT, which takes no attention mask,
    gets batches of prompts of equal length.

    Defined in :numref:`sec_lm_prompting`"""
    if chat:
        opts = {} if enable_thinking is None else {
            'enable_thinking': enable_thinking}
        prompts = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': m}] if isinstance(m, str) else m,
            tokenize=False, add_generation_prompt=True, **opts)
                   for m in prompts]
    ids, labels = _label_token_ids(tokenizer, prompts, label_words, sep)
    if not chat:  # as d2l.generate: a plain prompt gets the BOS, if any
        head = d2l._lm_bos_ids(tokenizer)  # and never an EOS
        ids = [head + r for r in ids]
    if not all(ids):
        raise ValueError('an empty prompt has no position to predict from')
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(d2l._lm_unwrap(model),
                                       hf.PreTrainedModel)
    if hf:  # one padded group, in input order
        groups = [list(range(len(ids)))]
        pad = getattr(tokenizer, 'pad_token_id', None) or 0
    else:
        groups = [[i for i in range(len(ids)) if len(ids[i]) == n]
                  for n in sorted({len(r) for r in ids})]
    device = next(model.parameters()).device
    out = torch.empty(len(ids), len(label_words))
    for rows in (g[j:j + batch_size] for g in groups
                 for j in range(0, len(g), batch_size)):
        with torch.inference_mode():
            if hf:  # logits_to_keep=1 and no cache: no full-vocabulary
                T = max(len(ids[i]) for i in rows)  # logits per position
                X = torch.tensor([[pad] * (T - len(ids[i])) + ids[i]
                                  for i in rows], device=device)
                mask = torch.tensor([[0] * (T - len(ids[i])) +
                                     [1] * len(ids[i]) for i in rows],
                                    device=device)
                pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
                z = model(input_ids=X, attention_mask=mask,
                          position_ids=pos, use_cache=False,
                          logits_to_keep=1).logits[:, -1]
            else:
                X = torch.tensor([ids[i] for i in rows], device=device)
                z = (model(X, logits_to_keep=1) if isinstance(
                    d2l._lm_unwrap(model), d2l.Qwen3GPT) else model(X))[:, -1]
            lab = torch.tensor([labels[i] for i in rows], device=z.device)
            lp = z.gather(-1, lab).float().log_softmax(-1).cpu()
        out[rows] = lp  # out is an ordinary tensor the caller may modify
    return out

def fewshot_prompt(demos, query, template='plain', *, fmt=None,
                   label_map=None, system=None):
    """Demonstrations demos, a list of (x, y), and a query x as a prompt.
    'plain' (fmt, default 'Input: {x}\\nOutput: {y}') and 'qa'
    ('Q: {x}\\nA: {y}') give a string whose blocks are separated by a blank
    line and which ends after the query's label field, without a space;
    'chat_single' gives the same text as one user message, 'chat_multi'
    one user and one assistant message per demonstration. label_map maps
    each y to its label word.

    Defined in :numref:`sec_lm_prompting`"""
    if template not in ('plain', 'qa', 'chat_single', 'chat_multi'):
        raise ValueError(f'unknown template: {template!r}')
    if fmt is not None:
        import string
        if template not in ('plain', 'chat_single'):
            raise ValueError(f'template {template!r} takes no fmt')
        parts = list(string.Formatter().parse(fmt))
        fields = [f for _, f, _, _ in parts if f is not None]
        tail = parts[-1][0] if parts and parts[-1][1] is None else ''
        if (set(fields) != {'x', 'y'} or fields.count('y') > 1 or
                fields[-1] != 'y' or tail.strip()):
            raise ValueError('fmt needs the fields {x} and {y}, with {y} '
                             f'once and last: {fmt!r}')

    def word(y):
        if label_map is not None:
            if y not in label_map:
                raise KeyError(f'label {y!r} is not in label_map')
            y = label_map[y]
        if not isinstance(y, str):
            raise TypeError(f'label {y!r} is not a string: pass label_map')
        return y
    demos = [(x, word(y)) for x, y in demos]
    head = [] if system is None else [{'role': 'system', 'content': system}]
    if template == 'chat_multi':
        return head + [{'role': r, 'content': c} for x, y in demos
                       for r, c in (('user', x), ('assistant', y))] + [
                           {'role': 'user', 'content': query}]
    fmt = 'Q: {x}\nA: {y}' if template == 'qa' else (
        fmt or 'Input: {x}\nOutput: {y}')
    text = '\n\n'.join([fmt.format(x=x, y=y) for x, y in demos] +
                       [fmt.format(x=query, y='').rstrip()])
    if template == 'chat_single':
        return head + [{'role': 'user', 'content': text}]
    return text if system is None else system + '\n\n' + text

d2l.DATA_HUB['ag-news-train'] = (
    'https://huggingface.co/datasets/sh0416/ag_news/resolve/'
    '70e3fa1915be9a8daebec5e840f20df9a8e18793/train.jsonl',
    '4b8aa80c8d0e1bea19ec0924b720317a6a61b040')
d2l.DATA_HUB['ag-news-test'] = (
    'https://huggingface.co/datasets/sh0416/ag_news/resolve/'
    '70e3fa1915be9a8daebec5e840f20df9a8e18793/test.jsonl',
    '6c874e265fe5ff35861f27eaf3e681dda147ba12')

def calibrate_labels(logprobs, content_free_logprobs):
    """Contextual calibration: subtract the mean log-probabilities (M, L)
    of content-free inputs from logprobs (N, L) and renormalize over the
    L labels; one content-free input may be given as an (L,) row. The
    mean is taken in log space; Zhao et al. (2021) average the
    probabilities instead.

    Defined in :numref:`sec_lm_prompting`"""
    logprobs = torch.as_tensor(logprobs)
    cf = torch.as_tensor(content_free_logprobs)
    if cf.ndim not in (1, 2) or cf.shape[-1] != logprobs.shape[-1]:
        raise ValueError(f'content-free log-probabilities {tuple(cf.shape)}'
                         f' are not (M, {logprobs.shape[-1]}) or '
                         f'({logprobs.shape[-1]},)')
    return torch.log_softmax(logprobs - cf.reshape(-1, cf.shape[-1]).mean(0),
                             dim=-1)

# Several of these files share a name: fetch them with d2l.download_keyed
d2l.DATA_HUB['arc-challenge-test'] = (
    'https://huggingface.co/datasets/allenai/ai2_arc/resolve/'
    '210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/'
    'test-00000-of-00001.parquet',
    '029435570dd5b680c95603f2f51a93218fa30cb2')
d2l.DATA_HUB['arc-challenge-train'] = (
    'https://huggingface.co/datasets/allenai/ai2_arc/resolve/'
    '210d026faf9955653af8916fad021475a3f00453/ARC-Challenge/'
    'train-00000-of-00001.parquet',
    '43a863338c7852a03fc5ab79f7c80b88e01a192d')
d2l.DATA_HUB['gsm8k-train'] = (
    'https://huggingface.co/datasets/openai/gsm8k/resolve/'
    '740312add88f781978c0658806c59bc2815b9866/main/'
    'train-00000-of-00001.parquet',
    '2328586d74d5426d030f29ac85a485f58514efbd')
d2l.DATA_HUB['gsm8k-test'] = (
    'https://huggingface.co/datasets/openai/gsm8k/resolve/'
    '740312add88f781978c0658806c59bc2815b9866/main/'
    'test-00000-of-00001.parquet',
    'b0c27c2f1cb845ba83b135c7e2575070b33ca967')
d2l.DATA_HUB['wikitext-2-raw-test'] = (
    'https://huggingface.co/datasets/Salesforce/wikitext/resolve/'
    'b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-2-raw-v1/'
    'test-00000-of-00001.parquet',
    'ab23bd60c20d2e169a8dc2d139ab6e7bf8b74e91')
# Fixed inputs: lm-evaluation-harness's scores on 200 ARC items
# (tools/oneshot/make_lm_eval_arc_fixture.py) and the held-out texts
# (tools/oneshot/make_lm_heldout_texts.py)
d2l.DATA_HUB['lm-eval-reference-arc-v1'] = (
    d2l.LM_DATA_URL + 'lm-eval-reference-arc-v1.json',
    'c0bdeb8593b0ba75406025d06e8cbea9016a2c71')
d2l.DATA_HUB['lm-heldout-texts-v1'] = (
    d2l.LM_DATA_URL + 'lm-heldout-texts-v1.json',
    '11b0ed4fead96910fba1078b2d9f5ee5b6a59b69')
# This section's measured results, written by a run with D2L_RERUN=1
d2l.DATA_HUB['lm-bpb-v1'] = (
    d2l.LM_DATA_URL + 'lm-bpb-v1.csv',
    '7c2ceeeda10b3f4ca6774821ed426c896442c3d3')
d2l.DATA_HUB['lm-gsm8k-samples-v1'] = (
    d2l.LM_DATA_URL + 'lm-gsm8k-samples-v1.jsonl',
    '11b561b6fb07d19016f5a00b4a5dcf35d3dbbb07')
d2l.DATA_HUB['lm-arc-generations-v1'] = (
    d2l.LM_DATA_URL + 'lm-arc-generations-v1.jsonl',
    '0b58a8a81e87794c5a564bd958142ede4288d195')
d2l.DATA_HUB['lm-arc-scores-v1'] = (
    d2l.LM_DATA_URL + 'lm-arc-scores-v1.parquet',
    'e4e1ea4b45a7a5288ae3b914459e60228b5e2931')

def download_keyed(key, root=os.path.join('..', 'data', 'lm')):
    """Download DATA_HUB[key] into root/key/ and return its local path.
    Files from different datasets often share a name (Hugging Face's
    'test-00000-of-00001.parquet'), and d2l.download keeps one file per
    name and folder, so each key gets a folder of its own.

    Defined in :numref:`sec_lm_scoring`"""
    return d2l.download(key, os.path.join(root, key))

def format_mc(question, options, fmt='cloze', *, shots=()):
    """A multiple-choice item as (prompt, choices). 'cloze': the prompt
    'Question: {question}\\nAnswer:' and each option after a space;
    'letter': 'Question: {question}', a line 'A. {option}' per option and
    'Answer:', with the letters ' A', ' B', ... as choices (OLMES's
    multiple-choice formulation); 'mmlu': the same without 'Question: ',
    as the MMLU paper and lm-evaluation-harness's mmlu tasks write it. Each
    shot, a tuple (question, options, answer_index), is rendered the same
    way with its answer and put before the item, separated by a blank line.
    Callers that want a header (such as MMLU's subject line) prepend it
    themselves.

    Defined in :numref:`sec_lm_scoring`"""
    if fmt not in ('cloze', 'letter', 'mmlu'):
        raise ValueError(f'unknown fmt: {fmt!r}')

    def render(q, opts):
        opts = [str(o) for o in opts]  # also numpy arrays, pandas Series
        if not opts:
            raise ValueError(f'no options for {q!r}')
        if fmt == 'cloze':
            return f'Question: {q}\nAnswer:', [' ' + o for o in opts]
        if len(opts) > 26:
            raise ValueError(f'{len(opts)} options, but 26 letters')
        letters = [chr(ord('A') + i) for i in range(len(opts))]
        lines = [q if fmt == 'mmlu' else f'Question: {q}'] + [
            f'{a}. {o}' for a, o in zip(letters, opts)]
        return '\n'.join(lines + ['Answer:']), [' ' + a for a in letters]
    blocks = []
    for q, opts, a in shots:
        prompt, choices = render(q, opts)
        if not 0 <= a < len(choices):
            raise ValueError(f'answer index {a} for {len(choices)} options')
        blocks.append(prompt + choices[a])
    prompt, choices = render(question, options)
    return '\n\n'.join(blocks + [prompt]), choices

ScoreComponents = collections.namedtuple(  # each (N, C_max), padded with
    'ScoreComponents', 'll n_tokens n_chars n_bytes')  # -inf (ll) and 0

def _lm_encode(tokenizer, texts):
    """Token ids of each text without special tokens: a Transformers
    tokenizer is called with add_special_tokens=False, any other one (such
    as tiktoken's) through its plain encode.

    Defined in :numref:`sec_lm_scoring`"""
    hf = sys.modules.get('transformers')
    texts = [str(t) for t in texts]  # also numpy arrays, pandas Series
    if not texts:
        return []
    if hf is not None and isinstance(tokenizer, hf.PreTrainedTokenizerBase):
        return tokenizer(texts, add_special_tokens=False)['input_ids']
    return [list(tokenizer.encode(t)) for t in texts]

def _lm_boundary_id(tokenizer):
    """The token that separates documents in pretraining: <|endoftext|> if
    the tokenizer has it, else its EOS, else None. Qwen3: 151643, Qwen3.5:
    248044, for base and post-trained checkpoints alike.

    Defined in :numref:`sec_lm_scoring`"""
    conv = getattr(tokenizer, 'convert_tokens_to_ids', None)
    if conv is not None:
        i = conv('<|endoftext|>')
        if i is not None and i != getattr(tokenizer, 'unk_token_id', None):
            return int(i)
    elif '<|endoftext|>' in getattr(tokenizer, 'special_tokens_set', ()):
        return tokenizer.encode_single_token('<|endoftext|>')  # tiktoken
    return getattr(tokenizer, 'eos_token_id', None)

def _lm_max_len(model):
    """The longest input the model takes, or None if it does not say.

    Defined in :numref:`sec_lm_scoring`"""
    model = d2l._lm_unwrap(model)
    cfg = getattr(model, 'config', None)
    if cfg is not None and hasattr(cfg, 'get_text_config'):
        cfg = cfg.get_text_config()
    return (getattr(cfg, 'max_position_embeddings', None) or
            getattr(model, 'max_len', None))

def score_choices(model, tokenizer, prompts, choices, norm='token', *,
                  batch_size=8, return_components=False):
    """Log-likelihood scores (N, C_max) of each choice after its prompt,
    -inf in padded columns; the prediction is argmax(-1). choices holds N
    lists (ragged) or one list shared by all prompts. norm divides the
    summed log-probability ll: 'sum' by 1, 'token' by the continuation's
    tokens, 'char' and 'byte' by the length and UTF-8 length of the choice
    without one leading whitespace character (lm-evaluation-harness's
    acc_norm and acc_bytes). Trailing whitespace of a prompt moves to the
    front of its choices; the continuation is the suffix of
    encode(prompt + choice) beyond encode(prompt) (a warning says when the
    prompt's last token merges into the choice's, which this rule scores as
    a different text), with no BOS (Qwen has none; for a tokenizer that
    needs one, start each prompt with tokenizer.bos_token); an empty
    prompt is preceded by the document-boundary token. With
    return_components=True, returns ScoreComponents, from which every norm
    follows. Log-softmax is taken in fp32 over the continuation positions
    only. In bf16 an item's score depends on which items share its batch,
    so a different call over the same items (another batch size or item
    set) agrees only up to that batch drift, not bit for bit.

    Defined in :numref:`sec_lm_scoring`"""
    import warnings
    if norm not in ('sum', 'token', 'char', 'byte'):
        raise ValueError(f'unknown norm: {norm!r}')
    prompts, choices = list(prompts), list(choices)  # also arrays, Series
    if len(choices) and isinstance(choices[0], str):  # one shared list
        choices = [choices] * len(prompts)
    choices = [list(cs) for cs in choices]
    if len(choices) != len(prompts):
        raise ValueError(f'{len(prompts)} prompts, {len(choices)} choices')
    heads = [p.rstrip() for p in prompts]
    rows = [(i, j) for i, cs in enumerate(choices) for j in range(len(cs))]
    conts = [prompts[i][len(heads[i]):] + choices[i][j] for i, j in rows]
    texts = [c[1:] if c[:1].isspace() else c for c in conts]
    ctx_ids = _lm_encode(tokenizer, heads)
    full = _lm_encode(tokenizer, [heads[i] + c
                                  for (i, _), c in zip(rows, conts)])
    max_len, cut_items, inputs, cont_ids = _lm_max_len(model), set(), [], []
    merged = []  # pairs whose prompt tokens are not a prefix of the whole
    for (i, j), f in zip(rows, full):
        ctx, cont = ctx_ids[i], f[len(ctx_ids[i]):]
        if not cont:
            raise ValueError(f'item {i}, choice {j}: empty continuation')
        if f[:len(ctx)] != ctx:
            merged.append((i, j))
        if not ctx:
            b = _lm_boundary_id(tokenizer)
            if b is None:
                raise ValueError(f'item {i}: empty prompt and no boundary '
                                 'token to put before it')
            ctx = [b]
        cut = len(ctx) + len(cont) - 1 - (max_len or float('inf'))
        if cut > 0:  # cut the prompt from the left
            if cut >= len(ctx):
                raise ValueError(f'item {i}, choice {j}: the continuation '
                                 f'does not fit the context of {max_len}')
            if i not in cut_items:
                warnings.warn(f'score_choices: prompt of item {i} cut by '
                              f'{cut} tokens to fit the context of {max_len}')
                cut_items.add(i)
            ctx = ctx[cut:]
        inputs.append((ctx + cont)[:-1])  # the last token predicts nothing
        cont_ids.append(cont)
    if merged:
        warnings.warn(
            f'score_choices: in {len(merged)} of {len(rows)} prompt-choice '
            f'pairs (first: item {merged[0][0]}, choice {merged[0][1]}) the '
            "prompt's last token merges with the choice's first, so the "
            'scored tokens do not spell prompt + choice; end the prompts '
            'differently (for example without a newline)')
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(d2l._lm_unwrap(model),
                                       hf.PreTrainedModel)
    qwen3 = isinstance(d2l._lm_unwrap(model), d2l.Qwen3GPT)
    pad = (getattr(tokenizer, 'pad_token_id', None) or 0) if hf else 0
    device = next(model.parameters()).device
    order = sorted(range(len(rows)), key=lambda r: (-len(inputs[r]), r))
    ll = torch.empty(len(rows), dtype=torch.float64)
    for s in range(0, len(order), batch_size):
        b = order[s:s + batch_size]
        T = max(len(inputs[r]) for r in b)
        m = [len(cont_ids[r]) for r in b]
        with torch.inference_mode():
            if hf:  # left padding: the continuations end every row
                X = torch.tensor([[pad] * (T - len(inputs[r])) + inputs[r]
                                  for r in b], device=device)
                mask = (torch.arange(T, device=device) >= torch.tensor(
                    [T - len(inputs[r]) for r in b], device=device)[:, None]
                        ).long()
                pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
                z = model(input_ids=X, attention_mask=mask, position_ids=pos,
                          use_cache=False, logits_to_keep=max(m)).logits
                Z = torch.cat([z[k, z.shape[1] - mk:]
                               for k, mk in enumerate(m)])
            else:  # right padding, exact under the causal mask
                X = torch.tensor([inputs[r] + [pad] * (T - len(inputs[r]))
                                  for r in b], device=device)
                starts = [len(inputs[r]) - mk for r, mk in zip(b, m)]
                off = min(starts) if qwen3 else 0
                out = model(X, logits_to_keep=T - off) if qwen3 else model(X)
                z = getattr(out, 'logits', out)
                Z = torch.cat([z[k, st - off:st - off + mk]
                               for k, (st, mk) in enumerate(zip(starts, m))])
            y = torch.tensor([t for r in b for t in cont_ids[r]],
                             device=Z.device)
            lp = torch.cat([  # fp32 log-softmax, at most 4,096 rows at once
                Z[i:i + 4096].float().log_softmax(-1).gather(
                    -1, y[i:i + 4096, None])[:, 0]
                for i in range(0, len(y), 4096)])
            sums = torch.stack([v.double().sum() for v in lp.split(m)])
        ll[b] = sums.cpu()
    C = max(map(len, choices), default=0)
    comps = [torch.zeros(len(prompts), C, dtype=dt)
             for dt in (torch.float32, torch.long, torch.long, torch.long)]
    comps[0].fill_(-math.inf)
    for r, (i, j) in enumerate(rows):
        for t, v in zip(comps, (ll[r].item(), len(cont_ids[r]),
                                len(texts[r]),
                                len(texts[r].encode('utf-8')))):
            t[i, j] = v
    comps = ScoreComponents(*comps)
    if return_components:
        return comps
    return comps.ll / {'sum': 1, 'token': comps.n_tokens,
                       'char': comps.n_chars, 'byte': comps.n_bytes}[norm]

def bits_per_byte(model, tokenizer, texts, ctx_len=2048, stride=None,
                  batch_size=4):
    """Bits per UTF-8 byte of texts: total negative log-likelihood in nats
    over ln 2 times the total bytes. Each text is put in Unicode NFC form
    (as Qwen's tokenizers do themselves, so the bytes counted are those
    scored), tokenized whole (no BOS) and preceded by one unscored
    document-boundary token (Qwen3: 151643, Qwen3.5: 248044; see
    _lm_boundary_id); without one, its first token, and any more up to the
    end of the character it starts, are unscored and their bytes leave the
    denominator. Windows of ctx_len input tokens (capped at the model's
    maximum input length, with a warning) advance by stride (default
    ctx_len // 2) and score every token once, the first window ctx_len
    tokens and each later one stride tokens, each with at least
    ctx_len - stride + 1 tokens of context; texts are scored independently.
    A Transformers model with a linear output layer runs its decoder on
    batch_size windows and projects 1,024 positions at a time; any other
    model runs one window at a time.

    Defined in :numref:`sec_lm_scoring`"""
    import unicodedata
    import warnings
    texts = [texts] if isinstance(texts, str) else list(texts)
    texts = [unicodedata.normalize('NFC', str(t)) for t in texts]
    max_len = _lm_max_len(model)
    if max_len and ctx_len > max_len:
        warnings.warn(f'bits_per_byte: ctx_len {ctx_len} capped at the '
                      f"model's maximum input length {max_len}")
        ctx_len = max_len
    stride = ctx_len // 2 if stride is None else stride
    if ctx_len < 1 or not 1 <= stride <= ctx_len:
        raise ValueError(f'need 1 <= stride <= ctx_len, got {stride}, '
                         f'{ctx_len}')
    b, windows, n_bytes = _lm_boundary_id(tokenizer), [], 0
    for text, ids in zip(texts, _lm_encode(tokenizer, texts)):
        if not ids:
            continue
        n_bytes += len(text.encode('utf-8'))
        first = 1  # unscored tokens: the boundary, or the text's first ones
        if b is None:  # the first token has no context; if it ends inside
            while first < len(ids):  # a character, so do the next ones
                head = tokenizer.decode(ids[:first])
                if text.startswith(head) or '\ufffd' not in head:
                    break
                first += 1
            n_bytes -= len(tokenizer.decode(ids[:first]).encode('utf-8'))
        seq = ids if b is None else [b] + ids
        done = first - 1  # targets seq[first:done + 1] are scored
        while done < len(seq) - 1:  # inputs seq[start:end] predict
            end = min(max(ctx_len, done + 1) if done == first - 1
                      else done + stride, len(seq) - 1)
            start = max(end - ctx_len, 0)  # seq[start + 1:end + 1]
            windows.append((seq[start:end], seq[done + 1:end + 1]))
            done = end
    if not n_bytes:
        raise ValueError('no text to score')
    inner = d2l._lm_unwrap(model)
    hf = sys.modules.get('transformers')
    hf = hf is not None and isinstance(inner, hf.PreTrainedModel)
    head = inner.get_output_embeddings() if hf else None
    cfg = inner.config.get_text_config() if hf else None
    split = (hf and isinstance(head, nn.Linear) and
             getattr(cfg, 'final_logit_softcapping', None) is None)
    device, nll = next(model.parameters()).device, 0.0

    def add_nll(H, y, project):  # H: (positions, d) or logits; y: targets
        total = 0.0
        for i in range(0, len(y), 1024):
            z = project(H[i:i + 1024]).float()
            total += F.cross_entropy(z, y[i:i + 1024], reduction='sum').item()
        return total
    with torch.inference_mode():
        for s in range(0, len(windows), batch_size if split else 1):
            w = windows[s:s + (batch_size if split else 1)]
            T = max(len(x) for x, _ in w)
            X = torch.tensor([x + [0] * (T - len(x)) for x, _ in w],
                             device=device)
            y = torch.tensor([t for _, ys in w for t in ys], device=device)
            if split:  # decoder states, then the head in chunks
                mask = (torch.arange(T, device=device) < torch.tensor(
                    [len(x) for x, _ in w], device=device)[:, None]).long()
                dec = (inner.get_decoder() if hasattr(inner, 'get_decoder')
                       else inner.model)
                H = dec(input_ids=X, attention_mask=mask,
                        use_cache=False).last_hidden_state
                H = torch.cat([H[k, len(x) - len(ys):len(x)]
                               for k, (x, ys) in enumerate(w)])
                nll += add_nll(H, y, head)
            else:  # one window: logits for its scored positions
                n = len(y)
                if hf:
                    z = model(input_ids=X, use_cache=False,
                              logits_to_keep=n).logits[0]
                elif isinstance(inner, d2l.Qwen3GPT):
                    z = model(X, logits_to_keep=n)[0]
                else:
                    out = model(X)
                    z = getattr(out, 'logits', out)[0, -n:]
                nll += add_nll(z, y, lambda v: v)
    return nll / (math.log(2) * n_bytes)

def extract_answer(text, kind='gsm8k', strict=False):
    """The answer a generated text commits to, as a string, or None.
    'gsm8k': the first number after the last '####', else after the last
    'the answer is' (any case), else, unless strict, the last number in the
    text; commas and '$' are removed and the number is normalized ('1,000'
    and '1000.00' give '1000', '-0.50' and '−0.50' give '-0.5'); a hyphen
    right after a letter, digit or point is not a sign ('10-15' gives
    '15'). 'boxed': the content
    of the last \\boxed{...} whose braces match, stripped.

    Defined in :numref:`sec_lm_scoring`"""
    if kind == 'boxed':
        found, i = None, text.find('\\boxed{')
        while i != -1:
            depth = 0
            for j in range(i + 6, len(text)):  # from the opening brace
                depth += {'{': 1, '}': -1}.get(text[j], 0)
                if depth == 0:
                    found = text[i + 7:j].strip()
                    break
            i = text.find('\\boxed{', i + 1)
        return found or None
    if kind != 'gsm8k':
        raise ValueError(f'unknown kind: {kind!r}')
    from decimal import Decimal
    number = re.compile(r'(?:(?<![\w.])[-\u2212])?\$?'  # sign, not hyphen
                        r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)')

    def normalize(s):
        d = Decimal(s.translate({ord(','): None, ord('$'): None,
                                 0x2212: '-'}))
        if d == d.to_integral_value():
            return str(int(d))
        return format(d.normalize(), 'f')
    for marker in ('####', 'the answer is'):
        ends = [k.end() for k in re.finditer(re.escape(marker), text, re.I)]
        m = number.search(text, ends[-1]) if ends else None
        if m:
            return normalize(m.group())
    found = None if strict else number.findall(text)
    return normalize(found[-1]) if found else None

def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples of which c are correct: the
    probability that k of them, drawn without replacement, include a
    correct one, 1 - C(n - c, k) / C(n, k), in the product form
    1 - prod_{i = n - c + 1}^{n} (1 - k / i). Integer arrays broadcast.

    Defined in :numref:`sec_lm_scoring`"""
    n, c, k = np.broadcast_arrays(*(np.asarray(v) for v in (n, c, k)))
    if not all(np.issubdtype(v.dtype, np.integer) for v in (n, c, k)):
        raise TypeError('n, c and k must be integers')
    for bad, msg in ((c < 0, 'c < 0'), (c > n, 'c > n'), (k < 1, 'k < 1'),
                     (k > n, 'k > n')):
        if np.any(bad):
            raise ValueError(f'pass_at_k needs 0 <= c <= n, 1 <= k <= n: '
                             f'{msg}')
    out = np.empty(n.shape)
    for idx in np.ndindex(n.shape):
        ni, ci, ki = int(n[idx]), int(c[idx]), int(k[idx])
        out[idx] = 1.0 if ni - ci < ki else 1.0 - np.prod(
            1.0 - ki / np.arange(ni - ci + 1, ni + 1))
    return float(out) if out.ndim == 0 else out

def pass_at_k_variance(n, p, k):
    """Variance of pass_at_k(n, c, k) when c ~ Binomial(n, p): the sum over
    c of Binomial(c; n, p) (U(n, c, k) - (1 - (1 - p)^k))^2.

    Defined in :numref:`sec_lm_scoring`"""
    if not 0 <= p <= 1 or not 1 <= k <= n:
        raise ValueError(f'need 0 <= p <= 1 and 1 <= k <= n: {n}, {p}, {k}')
    if p in (0, 1):
        return 0.0
    c = np.arange(n + 1)
    log_pmf = (math.lgamma(n + 1) - np.array([math.lgamma(i + 1) + math.lgamma(
        n - i + 1) for i in c]) + c * math.log(p) + (n - c) * math.log1p(-p))
    U = pass_at_k(np.full(n + 1, n), c, np.full(n + 1, k))
    return float(np.sum(np.exp(log_pmf) * (U - (1 - (1 - p) ** k)) ** 2))

def run_python_tests(program, test_code, timeout=5.0):
    """True iff program followed by test_code runs to its end and the
    process exits with code 0, in a fresh `python -I` process, in an empty
    temporary directory, with stdin closed, within timeout seconds of
    wall-clock time and, on POSIX, within limits on CPU time and address
    space (1 GiB). Exiting early (sys.exit, os._exit, a unittest.main() of
    the program's own) fails, as in human-eval; on POSIX, any process the
    code started is killed when the call returns. This is not a sandbox:
    the code runs as the caller, can read and write the caller's files and
    reach the network. Run code you did not write in a container.

    Defined in :numref:`sec_lm_scoring`"""
    import secrets
    import signal
    import subprocess
    import tempfile
    token = secrets.token_hex(16)  # written only once test_code has run
    limits = ''
    if os.name == 'posix':  # set in the child, before the program runs
        cpu = math.ceil(timeout) + 1
        limits = (
            'import resource\n'
            'for r, v in ((resource.RLIMIT_CPU, %d), '
            '(resource.RLIMIT_AS, 1 << 30)):\n'
            '    try:\n        resource.setrlimit(r, (v, v))\n'
            '    except (ValueError, OSError):\n        pass\n' % cpu)
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'main.py'), 'w') as f:
            f.write(program + '\n' + test_code + '\n')
        done = os.path.join(tmp, 'done')  # absolute: the code may chdir
        proc = subprocess.Popen(
            [sys.executable, '-I', '-c', limits + 'import runpy\n'
             "runpy.run_path('main.py', run_name='__main__')\n"
             f'open({done!r}, "w").write({token!r})'],
            cwd=tmp, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=os.name == 'posix')
        try:
            if proc.wait(timeout=timeout) != 0:
                return False
            with open(done) as f:
                return f.read() == token
        except (subprocess.TimeoutExpired, OSError):
            return False
        finally:
            if os.name == 'posix':  # the process and any it started
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                proc.kill()
            proc.wait()

# Run-time downloads only (licences): fetch them with d2l.download_keyed
d2l.DATA_HUB['gsm-symbolic-test'] = (
    'https://huggingface.co/datasets/apple/GSM-Symbolic/resolve/'
    '93b5b3758d9d9841ffe81d6cd2ae2b030685b078/main/test.jsonl',
    '6d7beda4cc95f9d04f219a99daaeaf445b53c9d1')
d2l.DATA_HUB['mt-bench-questions'] = (
    'https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/resolve/'
    'e3a795c5e9a82ee40611c416b8a7786c73198991/raw/question.jsonl',
    '1b5197cafba020598f6c60e9401a5098bcd81d27')
d2l.DATA_HUB['dolly-15k'] = (
    'https://huggingface.co/datasets/databricks/databricks-dolly-15k/'
    'resolve/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/'
    'databricks-dolly-15k.jsonl',
    '34356d82b686ae9fb3b230f428ba644f17e5e652')
d2l.DATA_HUB['mt-bench-human'] = (
    'https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/'
    'f7d2896d2cc5d80f8b55c2bbc722613555233c25/data/'
    'human-00000-of-00001-25f4910818759289.parquet',
    'e19c24360cb79f1919cfecd980a7a4951571fcdb')
d2l.DATA_HUB['mt-bench-gpt4-pair'] = (
    'https://huggingface.co/datasets/lmsys/mt_bench_human_judgments/resolve/'
    'f7d2896d2cc5d80f8b55c2bbc722613555233c25/data/'
    'gpt4_pair-00000-of-00001-c0b431264a82ddc0.parquet',
    '8928cda0399b6a0e40589cfbd3df4b7ccebdf242')

BootstrapResult = collections.namedtuple(  # ci = (lo, hi); p_two_sided is
    'BootstrapResult',                      # None for a single sample
    'mean ci se p_two_sided n n_clusters n_boot')

def _bootstrap_replicates(scores, n_boot, cluster, seed, chunk=2 * 10**7):
    """Scores y, n_boot bootstrap means and the number of clusters G. Each
    replicate draws G clusters with replacement: with C (n_boot, G) the
    counts, sums y_sum and sizes m of the clusters, the replicate means
    are the ratios (C @ y_sum) / (C @ m). C is built in chunks of at most
    chunk entries.

    Defined in :numref:`sec_lm_eval_suites`"""
    y = np.asarray(scores, dtype=float).ravel()
    if np.isnan(y).any():
        raise ValueError('scores contain NaN')
    if len(y) < 2:
        raise ValueError(f'{len(y)} items: the bootstrap needs at least 2')
    if cluster is None:  # every item its own cluster
        y_sum, m = y, np.ones_like(y)
    else:
        cluster = np.asarray(cluster).ravel()
        if len(cluster) != len(y):
            raise ValueError(f'{len(y)} scores, {len(cluster)} cluster ids')
        g = np.unique(cluster, return_inverse=True)[1]
        y_sum, m = np.bincount(g, weights=y), np.bincount(g).astype(float)
        if len(m) < 30:
            raise ValueError(f'{len(m)} clusters: the cluster bootstrap '
                             'needs at least 30')
    G = len(m)
    rng, rows, means = np.random.default_rng(seed), max(1, chunk // G), []
    for b in range(0, n_boot, rows):  # a loop over chunks, not replicates
        C = rng.multinomial(G, np.full(G, 1 / G), size=min(rows, n_boot - b))
        means.append(C @ y_sum / (C @ m))
    return y, np.concatenate(means), G

def bootstrap_ci(scores, n_boot=10000, cluster=None, *, seed=0, alpha=0.05):
    """Percentile bootstrap interval for the mean of scores. With cluster
    (one label per score), whole clusters are resampled and each
    replicate's mean is the ratio estimator, so clusters may differ in
    size; at least 30 clusters are required.

    Defined in :numref:`sec_lm_eval_suites`"""
    import warnings
    y, means, G = _bootstrap_replicates(scores, n_boot, cluster, seed)
    if len(y) < 100:
        warnings.warn(f'bootstrap_ci: {len(y)} items; below 100 the '
                      'percentile interval is unreliable')
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(y.mean()), (float(lo), float(hi)),
                           float(means.std(ddof=1)), None, len(y), G, n_boot)

def paired_bootstrap(scores_a, scores_b, n_boot=10000, cluster=None, *,
                     seed=0, alpha=0.05):
    """bootstrap_ci of the per-item differences scores_a - scores_b on the
    same items, plus a bootstrap p-value, 2 min(P*(mean <= 0),
    P*(mean >= 0)) over the replicates; the sign-flip test of the
    statistics appendix is its permutation counterpart.

    Defined in :numref:`sec_lm_eval_suites`"""
    import warnings
    a, b = (np.asarray(s, dtype=float).ravel() for s in (scores_a, scores_b))
    if a.shape != b.shape:
        raise ValueError(f'{len(a)} and {len(b)} scores: pair them by item')
    d, means, G = _bootstrap_replicates(a - b, n_boot, cluster, seed)
    if len(d) < 100:
        warnings.warn(f'paired_bootstrap: {len(d)} items; below 100 the '
                      'percentile interval is unreliable')
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    p = min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))
    return BootstrapResult(float(d.mean()), (float(lo), float(hi)),
                           float(means.std(ddof=1)), float(p), len(d), G,
                           n_boot)

def min_detectable_effect(sd=None, n=None, *, se=None, deff=1.0,
                          alpha=0.05, power=0.8):
    """Smallest true difference that a two-sided level-alpha test detects
    with the given power: (z_{1-alpha/2} + z_power) sd sqrt(deff / n), or
    from a standard error, (z_{1-alpha/2} + z_power) se sqrt(deff); arrays
    broadcast.

    Defined in :numref:`sec_lm_eval_suites`"""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    if se is not None:
        if sd is not None or n is not None:
            raise ValueError('pass sd and n, or se alone')
        mde = z * np.asarray(se, dtype=float) * np.sqrt(deff)
    elif sd is None or n is None:
        raise ValueError('pass sd and n, or se alone')
    else:
        mde = z * np.asarray(sd, dtype=float) * np.sqrt(
            np.asarray(deff, dtype=float) / np.asarray(n, dtype=float))
    return float(mde) if mde.ndim == 0 else mde

EvalTask = collections.namedtuple('EvalTask', [
    'id', 'dataset', 'revision', 'files', 'kind', 'adapter', 'split',
    'config', 'n', 'subset_seed', 'fmt', 'norm', 'template', 'chat',
    'enable_thinking', 'shots', 'shots_from', 'max_new_tokens', 'sampling',
    'stop', 'extract', 'cluster', 'since'], defaults=(
        'test', None, None, 0, 'cloze', 'token', None, False, None, 0, None,
        None, None, (), None, None, None))

def subset_ids(n_total, n, seed=0):
    """Row indices of a seeded subset of n of n_total items: the first n
    of one permutation, so a smaller subset is a prefix of a larger one;
    n=None keeps every row, in order.

    Defined in :numref:`sec_lm_eval_suites`"""
    if n is None:
        return np.arange(n_total)
    if not 0 <= n <= n_total:
        raise ValueError(f'a subset of {n} from {n_total} items')
    return np.random.default_rng(seed).permutation(n_total)[:n]

def item_seed(seed, task_id, item_id):
    """A 63-bit seed for one item's random draws, from sha256 of
    'seed/task_id/item_id' (Python's hash() is salted per process).

    Defined in :numref:`sec_lm_eval_suites`"""
    digest = hashlib.sha256(f'{seed}/{task_id}/{item_id}'.encode()).digest()
    return int.from_bytes(digest[:8], 'big') >> 1

class EvalSuite:
    """Evaluation tasks (EvalTask records, or ids of d2l.EVAL_TASKS) with
    their items; n and seed, unless None, replace every task's n and
    subset_seed. batch_size is a run setting, not part of the suite.

    Defined in :numref:`sec_lm_eval_suites`"""
    def __init__(self, tasks, name='suite', batch_size=8, *, n=None, seed=0):
        self.name, self.batch_size, self._items = name, batch_size, {}
        resolved = []
        for t in tasks:
            if isinstance(t, str):
                t = d2l.EVAL_TASKS[t]
            t = EvalTask(**(t if isinstance(t, dict) else t._asdict()))
            t = t._replace(**{k: v for k, v in (('n', n), ('subset_seed',
                                                           seed))
                              if v is not None})
            t = d2l._eval_canonical(t)  # JSON values, files and stop tuples
            if t.id in self._items:
                raise ValueError(f'task {t.id!r} appears twice')
            rows = d2l._eval_load_rows(t)
            ids = subset_ids(len(rows), t.n, t.subset_seed)
            self._items[t.id] = d2l._eval_items(t, rows, ids)
            resolved.append(t)
        self.tasks = tuple(resolved)

    @property
    def hash(self):
        """sha256 of the canonical JSON of every task's fields, followed by
        the sha256 of each task's items (ids, rendered prompts, choices and
        references) in id order: a new dataset revision, subset or prompt
        rendering gives a new hash."""
        h = hashlib.sha256(json.dumps([t._asdict() for t in self.tasks],
                                      sort_keys=True).encode())
        for t in self.tasks:
            items = sorted(self._items[t.id], key=lambda item: item['id'])
            h.update(hashlib.sha256(json.dumps(
                items, sort_keys=True, default=str).encode()).digest())
        return h.hexdigest()

    def items(self, task_id):
        """The items of one task in subset order: dicts with the item's
        'id' and the prompt fields of the task's kind."""
        return self._items[task_id]

    def _score(self, task, items, model, tokenizer, seed):
        """Result rows (task, item_id, cluster, score, pred, finish_reason)
        for some items of one task, by the task's kind.

        Defined in :numref:`sec_lm_eval_suites`"""
        preds, reasons = [None] * len(items), [None] * len(items)
        if task.kind == 'mc':  # the highest-scoring choice; ties to the first
            S = d2l.score_choices(model, tokenizer, [it['prompt'] for it in items],
                                  [it['choices'] for it in items], task.norm,
                                  batch_size=self.batch_size)
            preds = S.argmax(-1).tolist()
            scores = [float(p == it['gold']) for p, it in zip(preds, items)]
        elif task.kind == 'mc_partial':  # one continuation after each prompt
            S = d2l.score_choices(
                model, tokenizer, [p for it in items for p in it['prompts']],
                [[it['continuation']] for it in items for _ in it['prompts']],
                task.norm, batch_size=self.batch_size)[:, 0]
            preds = [int(s.argmax()) for s in S.split(
                [len(it['prompts']) for it in items])]
            scores = [float(p == it['gold']) for p, it in zip(preds, items)]
        elif task.kind in ('gen', 'code'):
            sampling = dict(task.sampling or {})  # None: greedy
            if sampling:
                sampling.setdefault('strategy', 'sample')
            gens = d2l.generate(
                model, tokenizer, [it['prompt'] for it in items],
                task.max_new_tokens, chat=task.chat,
                enable_thinking=task.enable_thinking, stop=list(task.stop),
                batch_size=self.batch_size, **sampling,
                seed=[item_seed(seed, task.id, it['id']) for it in items])
            reasons = [g.finish_reason for g in gens]
            if task.kind == 'gen':
                preds = [d2l.extract_answer(g.text, task.extract or 'gsm8k')
                         for g in gens]
                scores = [float(p is not None and p == it['reference'])
                          for p, it in zip(preds, items)]
            else:  # the program is the prompt and its completion
                scores = [float(d2l.run_python_tests(it['prompt'] + g.text,
                                                     it['test']))
                          for g, it in zip(gens, items)]
        elif task.kind == 'ppl':  # bits per byte per text (mean: not per byte)
            scores = [d2l.bits_per_byte(model, tokenizer, [it['text']])
                      for it in items]
        else:
            raise ValueError(f'unknown kind: {task.kind!r}')
        return [{'task': task.id, 'item_id': it['id'],
                 'cluster': str(it[task.cluster] if task.cluster else it['id']),
                 'score': s, 'pred': None if p is None else str(p),
                 'finish_reason': r}
                for it, s, p, r in zip(items, scores, preds, reasons)]

    def run(self, model, tokenizer, *, seed=0, cache_dir=None, rerun=False,
            sandbox=False):
        """Score every item and return an EvalResult. With cache_dir, results
        are written per chunk under the model, its revision, the suite hash
        and a run key (seed, dtype, batch size, library versions), and a run
        with the same key resumes (rerun=True starts over). Sampled tasks
        draw item i of task t with the generator seeded item_seed(seed, t, i).
        Code tasks execute model-written programs: sandbox=True says that the
        process runs in a sandbox.

        Defined in :numref:`sec_lm_eval_suites`"""
        if not sandbox and any(t.kind == 'code' for t in self.tasks):
            raise RuntimeError('code tasks run model-written programs; run them '
                               'in a sandbox and pass sandbox=True')
        model_id, revision = d2l._eval_model_id(model)
        hf = sys.modules.get('transformers')
        settings = {'seed': seed, 'batch_size': self.batch_size,
                    'dtype': str(next(model.parameters()).dtype),
                    'torch': torch.__version__,
                    'transformers': getattr(hf, '__version__', None)}
        run_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()
                                 ).hexdigest()[:16]
        suite_hash = self.hash  # covers the rendered prompts
        path = None if cache_dir is None else d2l._eval_cache_path(
            cache_dir, model_id, revision, suite_hash, run_key)
        rows = [] if path is None or rerun else d2l._eval_cache_read(path)
        done, start = {(r['task'], r['item_id']) for r in rows}, time.time()
        for task in self.tasks:
            todo = [it for it in self.items(task.id)
                    if (task.id, it['id']) not in done]
            step = 32 * self.batch_size  # items between two cache writes
            for i in range(0, len(todo), step):
                rows += self._score(task, todo[i:i + step], model, tokenizer, seed)
                if path is not None:
                    d2l._eval_cache_write(path, rows)
        meta = {'model': model_id, 'revision': revision, 'suite': self.name,
                'suite_hash': suite_hash, 'run_key': run_key, **settings,
                'tasks': [t._asdict() for t in self.tasks],
                'wall_clock': time.time() - start}
        return EvalResult(pd.DataFrame(rows), meta)

    @classmethod
    def from_manifest(cls, name_or_path, batch_size=8):
        """The suite of a frozen manifest: a name in d2l.EVAL_SUITES or a JSON
        file written by to_manifest. Its tasks' n and subset_seed are kept.

        Defined in :numref:`sec_lm_eval_suites`"""
        if name_or_path in d2l.EVAL_SUITES:
            manifest = d2l.EVAL_SUITES[name_or_path]
        else:
            with open(name_or_path, encoding='utf-8') as f:
                manifest = json.load(f)
        return cls(manifest['tasks'], manifest['name'], batch_size, n=None,
                   seed=None)

    def to_manifest(self, path=None):
        """The suite as a JSON-ready manifest {'name', 'tasks'}; with path,
        also written there as sorted-key JSON.

        Defined in :numref:`sec_lm_eval_suites`"""
        manifest = json.loads(json.dumps(
            {'name': self.name, 'tasks': [t._asdict() for t in self.tasks]}))
        if path is not None:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, sort_keys=True, indent=1)
                f.write('\n')
        return manifest

class EvalResult:
    """Per-item results of one model on one suite (per_item: task, item_id,
    cluster, score, pred, finish_reason, sorted by task and item id) and
    the run's settings (meta).

    Defined in :numref:`sec_lm_eval_suites`"""
    def __init__(self, per_item, meta):
        self.per_item = per_item.sort_values(['task', 'item_id'],
                                             ignore_index=True)
        self.meta = meta

    def scores(self, task=None):
        """One task's scores ordered by item id, or a dict over tasks. Ids
        are strings ('10' sorts before '2'): align other data by id."""
        if task is None:
            return {t: self.scores(t) for t in self.per_item.task.unique()}
        return self.per_item.score[self.per_item.task == task].to_numpy()

    @staticmethod
    def _clusters(task, cluster):
        """The cluster labels if the items share clusters, else None (one
        cluster per item); fewer than 30 shared clusters raise."""
        k = cluster.nunique()
        if k == len(cluster):
            return None
        if k < 30:
            raise ValueError(f'task {task}: {k} clusters; the cluster '
                             'bootstrap needs at least 30')
        return cluster.to_numpy()

    def summary(self, n_boot=2000):
        """Per task: n, mean and its 95% interval (clusters resampled)."""
        rows = []
        for task, df in self.per_item.groupby('task'):
            r = bootstrap_ci(df.score, n_boot,
                             self._clusters(task, df.cluster))
            rows.append({'task': task, 'n': r.n, 'mean': r.mean,
                         'lo': r.ci[0], 'hi': r.ci[1],
                         'clusters': r.n_clusters})
        return pd.DataFrame(rows)

    def macro(self, n_boot=10000, *, seed=0, alpha=0.05):
        """The mean of the task means, with a bootstrap that resamples
        items (or clusters) within each task."""
        reps = [_bootstrap_replicates(df.score, n_boot,
                                      self._clusters(t, df.cluster),
                                      [seed, k])
                for k, (t, df) in enumerate(self.per_item.groupby('task'))]
        means = np.mean([m for _, m, _ in reps], axis=0)
        lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
        return BootstrapResult(
            float(np.mean([y.mean() for y, _, _ in reps])),
            (float(lo), float(hi)), float(means.std(ddof=1)), None,
            sum(len(y) for y, _, _ in reps), len(reps), n_boot)

    def compare(self, other, *, n_boot=10000, seed=0):
        """Per task, paired_bootstrap of this model's scores minus other's,
        aligned on (task, item_id); the two runs must share suite and
        items."""
        if self.meta['suite_hash'] != other.meta['suite_hash']:
            raise ValueError(f"different suites: {self.meta['suite_hash']} "
                             f"and {other.meta['suite_hash']}")
        m = self.per_item.merge(other.per_item, on=['task', 'item_id'],
                                how='outer', suffixes=('', '_b'),
                                indicator=True)
        bad = m[m._merge != 'both']
        if len(bad):
            side = {'left_only': 'self', 'right_only': 'other'}
            first = [(t, i, side[w]) for t, i, w in zip(
                bad.task, bad.item_id, bad._merge.astype(str))][:5]
            raise ValueError(f'item sets differ in {len(bad)} items; first '
                             f'(task, item_id, only in): {first}')
        return {task: paired_bootstrap(df.score, df.score_b, n_boot,
                                       self._clusters(task, df.cluster),
                                       seed=seed)
                for task, df in m.groupby('task')}

    def save(self, path):
        """Write per_item to path (parquet, rows sorted) and meta, without the
        wall clock, to path with the suffix .json (sorted keys).

        Defined in :numref:`sec_lm_eval_suites`"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        _eval_sorted(self.per_item).to_parquet(path, index=False)
        meta = {k: v for k, v in self.meta.items() if k != 'wall_clock'}
        with open(os.path.splitext(path)[0] + '.json', 'w',
                  encoding='utf-8') as f:
            json.dump(meta, f, sort_keys=True, indent=1)
            f.write('\n')

    @classmethod
    def load(cls, path):
        """An EvalResult written by save.

        Defined in :numref:`sec_lm_eval_suites`"""
        with open(os.path.splitext(path)[0] + '.json', encoding='utf-8') as f:
            meta = json.load(f)
        return cls(pd.read_parquet(path), meta)

JUDGE_RUBRIC_V1 = (
    'Below are a question and two answers to it. Decide which answer is '
    'more helpful and more correct for the person who asked. Do not '
    'explain your decision.\n\n[Question]\n{question}\n\n[Answer A]\n'
    '{first}\n\n[Answer B]\n{second}\n\nWhich answer is better? Reply with '
    'a single letter: A or B.')

JUDGE_PAD_V1 = (
    'I hope this answer is useful. It responds to the question that was '
    'asked, and the points above are meant to cover what the question asks '
    'for. If any part of it is unclear, reading the answer above again '
    'should help, since it already contains everything this response has '
    'to say on the topic.')

JudgeResult = collections.namedtuple(  # verdict: 'a', 'b' or 'tie'
    'JudgeResult', 'pref_a delta position_bias consistent verdict logits')

def judge_pairwise(judge, tokenizer, question, answer_a, answer_b, *,
                   rubric=JUDGE_RUBRIC_V1, batch_size=8,
                   max_batch_tokens=None, return_mass=False):
    """Ask a chat model which answer is better, once with answer_a in the
    first slot (log-odds l1 of the token 'A' over 'B') and once with
    answer_b there (l2): delta = (l1 - l2) / 2, position_bias =
    (l1 + l2) / 2. Strings give one JudgeResult, lists one per item.
    Prompts are sorted by length and batched (at most max_batch_tokens
    tokens including padding, if given); answers are cut at 1,500 tokens.
    The logits of A and B are recomputed in fp32 from the input of the
    judge's output head, since bf16 logits are rounded. With
    return_mass=True, also returns P(A) + P(B) per item and order.

    Defined in :numref:`sec_lm_eval_suites`"""
    import warnings
    single = isinstance(question, str)
    if single:
        question, answer_a, answer_b = [question], [answer_a], [answer_b]
    if not len(question) == len(answer_a) == len(answer_b):
        raise ValueError('one question and two answers per item')
    if not question:
        return ([], np.zeros((0, 2))) if return_mass else []
    letters = [tokenizer.encode(c, add_special_tokens=False) for c in 'AB']
    if any(len(ids) != 1 for ids in letters) or letters[0] == letters[1]:
        raise ValueError(f'A and B are not two single tokens: {letters}')

    def cut(text, i, name):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= 1500:
            return text
        warnings.warn(f'judge_pairwise: item {i}: {name} cut from '
                      f'{len(ids)} to 1500 tokens')
        return tokenizer.decode(ids[:1500])

    def render(q, first, second):
        content = rubric.format(question=q, first=first, second=second)
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
    texts = []  # two per item: answer_a first, then answer_b first
    for i, (q, a, b) in enumerate(zip(question, answer_a, answer_b)):
        a, b = cut(a, i, 'answer_a'), cut(b, i, 'answer_b')
        texts += [render(q, a, b), render(q, b, a)]
    unique = list(dict.fromkeys(texts))  # identical prompts run once
    ids = [tokenizer(t, add_special_tokens=False)['input_ids']
           for t in unique]
    order = sorted(range(len(ids)), key=lambda r: (-len(ids[r]), r))
    batches = [[]]
    for r in order:  # longest first, so a batch's first row sets its length
        b = batches[-1]
        if b and (len(b) == batch_size or max_batch_tokens is not None and
                  len(ids[b[0]]) * (len(b) + 1) > max_batch_tokens):
            batches.append([])
        batches[-1].append(r)
    pad = tokenizer.pad_token_id
    pad = tokenizer.eos_token_id if pad is None else pad
    device = next(judge.parameters()).device
    AB = [letters[0][0], letters[1][0]]
    # bf16 logits are rounded (steps of 0.125 near 20): the A and B logits
    # are redone in fp32 from the output head's input, if it has one
    head = getattr(judge, 'get_output_embeddings', lambda: None)()
    seen, warned = [], False
    hook = (head.register_forward_hook(lambda m, x, y: seen.append(x[0]))
            if isinstance(head, nn.Linear) else None)
    # per prompt: log P(A), log P(B) and the log-odds of A over B
    lp = torch.empty(len(ids), 3, dtype=torch.float64)
    try:
        for b in batches:
            T = len(ids[b[0]])
            X = torch.tensor([[pad] * (T - len(ids[r])) + ids[r] for r in b],
                             device=device)  # padded on the left
            mask = torch.tensor([[0] * (T - len(ids[r])) + [1] * len(ids[r])
                                 for r in b], device=device)
            pos = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)
            seen.clear()
            with torch.inference_mode():
                z = judge(input_ids=X, attention_mask=mask, position_ids=pos,
                          use_cache=False, logits_to_keep=1).logits[:, -1]
                zab = z[:, AB].float()
                if len(seen) == 1:
                    h = seen[0][:, -1].float()
                    w = head.weight[AB].to(h.device, torch.float32)
                    exact = h @ w.T if head.bias is None else (
                        h @ w.T + head.bias[AB].to(h.device, torch.float32))
                    exact = exact.to(zab.device)
                    if torch.allclose(exact, zab, rtol=2**-7, atol=1e-3):
                        zab = exact
                    elif not warned:  # a head that is not the last step
                        warned = True
                        warnings.warn('judge_pairwise: the output head does '
                                      'not give the logits; the log-odds '
                                      'use the logits as returned')
            lp[b, :2] = z.float().log_softmax(-1)[:, AB].double().cpu()
            lp[b, 2] = (zab[:, 0] - zab[:, 1]).double().cpu()
    finally:
        if hook is not None:
            hook.remove()
    row = {t: k for k, t in enumerate(unique)}
    l1, l2 = (torch.stack([lp[row[t]] for t in texts[j::2]])
              for j in (0, 1))
    results = []
    for o1, o2 in zip(l1[:, 2].tolist(), l2[:, 2].tolist()):
        # o1, o2: log-odds that the first slot wins, in either order
        delta, bias = (o1 - o2) / 2, (o1 + o2) / 2
        verdict = 'a' if o1 > 0 > o2 else 'b' if o1 < 0 < o2 else 'tie'
        results.append(JudgeResult(0.5 * (1 + math.tanh(delta / 2)), delta,
                                   bias, verdict != 'tie', verdict, (o1, o2)))
    mass = torch.stack([l1[:, :2].exp().sum(-1), l2[:, :2].exp().sum(-1)],
                       1).numpy()
    out = results[0] if single else results
    return (out, mass[0] if single else mass) if return_mass else out

def bradley_terry(winners, losers, n_items, *, weights=None, l2=1e-3,
                  n_iter=50):
    """Bradley-Terry strengths s, P(i beats j) = sigmoid(s_i - s_j): the
    maximizer of the weighted log-likelihood minus l2 ||s||^2 by Newton's
    method, centred. weights (N,) give (n_items,); weights (B, N), such as
    bootstrap counts, give B fits at once, (B, n_items).

    Defined in :numref:`sec_lm_eval_suites`"""
    N = len(winners)
    X = np.zeros((N, n_items))  # +1 for the winner, -1 for the loser
    np.add.at(X, (np.arange(N), np.asarray(winners)), 1)
    np.add.at(X, (np.arange(N), np.asarray(losers)), -1)
    W = np.ones(N) if weights is None else np.asarray(weights, dtype=float)
    single, W = W.ndim == 1, np.atleast_2d(W)
    s = np.zeros((len(W), n_items))
    for _ in range(n_iter):
        p = 0.5 * (1 + np.tanh(s @ X.T / 2))  # (B, N): P(winner wins)
        grad = (W * (1 - p)) @ X - 2 * l2 * s
        H = np.einsum('bn,nk,nj->bkj', W * p * (1 - p), X, X, optimize=True
                      ) + 2 * l2 * np.eye(n_items)  # minus the Hessian
        step = np.linalg.solve(H, grad[..., None])[..., 0]
        s = s + step
        if np.abs(step).max() < 1e-10:
            break
    s = s - s.mean(-1, keepdims=True)
    return s[0] if single else s

d2l.HF_REVISIONS['Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50'] = (
    'ce1a79d9c5163932d65c417380e53230e1086370')

d2l.DATA_HUB['lm-ud-ewt-train'] = (
    'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/'
    '6b021622f3ab7902730f313fa5955a8b8cdd52b8/en_ewt-ud-train.conllu',
    '59e9ff791f828f7cd3dcd2e55dc4f30ea7bbd5f9')
d2l.DATA_HUB['lm-ud-ewt-dev'] = (
    'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/'
    '6b021622f3ab7902730f313fa5955a8b8cdd52b8/en_ewt-ud-dev.conllu',
    '1e3226c7b533354e4f79e7bae7a1aa56a7dd9c94')
d2l.DATA_HUB['lm-ud-ewt-test'] = (
    'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/'
    '6b021622f3ab7902730f313fa5955a8b8cdd52b8/en_ewt-ud-test.conllu',
    '41b279558051f482d78fff4ea7b3c8c98c91ecd4')
# The recall pairs (tools/oneshot/build_lm_recall_pairs.py)
d2l.DATA_HUB['lm-interp-recall-v1'] = (
    d2l.LM_DATA_URL + 'lm-interp-recall-v1.json',
    'e3025b0216df5384d8a14f1e8e10c4cab080ebbf')

class _InterveneStop(Exception):
    """Raised right after the stop_after site, to end a forward early.

    Defined in :numref:`sec_lm_interpretability`"""

class _Intervention:
    """The context manager that d2l.intervene returns.

    Defined in :numref:`sec_lm_interpretability`"""
    KINDS = ('resid_post', 'resid_mid', 'attn_out', 'mlp_out', 'head_z')

    def __init__(self, model, sites, record, record_grad, stop_after):
        assert isinstance(model, d2l.GPT), 'intervene instruments a d2l.GPT'
        for blk in model.blks:
            assert blk.pre_norm, 'intervene needs pre-norm blocks'
            assert not (blk.training and blk.dropout.p > 0), (
                'dropout must be inactive: call model.eval()')
        self.model, self.record_grad = model, record_grad
        self.sites = {self._check(s): fn for s, fn in (sites or {}).items()}
        for site, fn in self.sites.items():
            if fn is None:  # recording is record='s job
                raise ValueError(f'site {site} maps to None; use record=')
        self.stop_after = None if stop_after is None else self._check(
            stop_after)
        self.record = {self._check(s) for s in record}
        if self.stop_after is not None:  # a stopped forward is for reading
            self.record.add(self.stop_after)
        self.cache = {s: [] for s in self.record}
        self.wanted = set(self.sites) | self.record

    def _check(self, site):
        if not (isinstance(site, tuple) and len(site) == 2
                and site[0] in self.KINDS
                and isinstance(site[1], (int, np.integer))
                and 0 <= site[1] < len(self.model.blks)):
            raise ValueError(f'not a site of this model: {site!r}')
        return site[0], int(site[1])

    @property
    def grad_cache(self):
        """The gradient at each recorded tensor; read after a backward."""
        return {s: [x.grad for x in xs] for s, xs in self.cache.items()}

    def _site(self, site, x):
        """Apply the site's fn to x and record the result. Returns fn's
        value (None without a fn) and a term to add to what the module
        passes on (None without record_grad)."""
        self.fired.add(site)
        fn, y, zero = self.sites.get(site), None, None
        if fn is not None:
            y = fn(x)
            if not (torch.is_tensor(y) and y.shape == x.shape
                    and y.dtype == x.dtype):
                raise ValueError(f'{site}: fn must return a tensor of shape '
                                 f'{tuple(x.shape)} and dtype {x.dtype}')
        if site in self.record:
            v = (x if y is None else y).detach()
            if self.record_grad:
                if not torch.is_grad_enabled():
                    raise RuntimeError('record_grad needs gradients enabled')
                # leaf - leaf.detach() is exactly 0: adding it leaves the
                # forward unchanged and makes leaf.grad the gradient there
                v.requires_grad_()
                zero = v - v.detach()
            self.cache[site].append(v)
        if site == self.stop_after:
            raise _InterveneStop(self)
        return y, zero

    def _write_hook(self, inner, outer, base):
        """Forward hook on a module whose output a block adds to its stream
        (W_o, the FFN): site inner sees that output, site outer the stream
        after the add, r = base[0] + output."""
        def hook(module, args, out):
            new = out
            if inner in self.wanted:
                y, zero = self._site(inner, out)
                new = new if y is None else y
                new = new if zero is None else new + zero
            if outer in self.wanted:
                r = base[0] + new  # the stream, as the block's add makes it
                y, zero = self._site(outer, r)
                if y is not None:  # the block's own add then yields about y,
                    new = new + (y - r)  # exactly r wherever y equals r
                new = new if zero is None else new + zero
            return None if new is out else new
        return hook

    def _z_hook(self, site):
        """Forward pre-hook on W_o: site head_z sees its input."""
        def hook(module, args):
            y, zero = self._site(site, args[0])
            new = args[0] if y is None else y
            new = new if zero is None else new + zero
            return None if new is args[0] else (new,)
        return hook

    @staticmethod
    def _store_hook(box):
        """Forward pre-hook that keeps the module's input in box[0]."""
        def hook(module, args):
            box[0] = args[0]
        return hook

    def __enter__(self):
        self.fired, self.handles = set(), []
        for l, blk in enumerate(self.model.blks):
            kinds = {kind for kind, layer in self.wanted if layer == l}
            W_o, hooks = blk.attention.W_o, []
            if 'head_z' in kinds:
                hooks.append(W_o.register_forward_pre_hook(
                    self._z_hook(('head_z', l))))
            if kinds & {'attn_out', 'resid_mid'}:
                x_in = [None]  # the block's input, the stream before it
                hooks += [
                    blk.norm1.register_forward_pre_hook(
                        self._store_hook(x_in)),
                    W_o.register_forward_hook(self._write_hook(
                        ('attn_out', l), ('resid_mid', l), x_in))]
            if kinds & {'mlp_out', 'resid_post'}:
                mid = [None]  # the stream after attention, as norm2 sees it
                hooks += [
                    blk.norm2.register_forward_pre_hook(
                        self._store_hook(mid)),
                    blk.ffn.register_forward_hook(self._write_hook(
                        ('mlp_out', l), ('resid_post', l), mid))]
            self.handles += hooks
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        if isinstance(exc, _InterveneStop):
            return exc.args[0] is self  # another context's stop goes on
        if exc_type is None:
            unfired = sorted(self.wanted - self.fired)
            if unfired:
                raise RuntimeError(f'sites that never fired: {unfired}')
        return False

def intervene(model, sites=None, *, record=(), record_grad=False,
              stop_after=None):
    """Context manager that reads and overwrites activations of a d2l.GPT
    with pre-norm blocks, in the full forward and in forward_cached. A
    site is (kind, layer), kind one of 'resid_post', 'resid_mid',
    'attn_out', 'mlp_out', 'head_z' (the per-head input of W_o, (B, T,
    heads * head_dim)). sites maps a site to fn(x) -> x_new (same shape and
    dtype), applied in every call; at a resid site the block's own add
    then carries x_new (rounded), and exactly x wherever x_new equals x.
    Nested contexts apply the outer fn first, and each context reads a
    block at its own hooks: an outer context's resid_mid or resid_post is
    the stream before an inner context's edit of attn_out or mlp_out in
    that block, so record and edit in one context. ctx.cache[site]
    lists, for each recorded site, one tensor per call (after fn). With
    record_grad=True, ctx.grad_cache[site] lists their gradients after
    the caller's backward. stop_after=site ends the forward right after
    that site (recorded) without an error. On a normal exit, a site that
    never fired raises.

    Defined in :numref:`sec_lm_interpretability`"""
    return _Intervention(model, sites, record, record_grad, stop_after)

def logit_lens(model, hidden):
    """The model's final norm and output head applied in fp32 to hidden
    (..., d), a stream read at any block; returns logits (..., V).

    Defined in :numref:`sec_lm_interpretability`"""
    norm, h = model.norm, hidden.float()
    f32 = lambda p: None if p is None else p.float()  # no affine, no bias
    if isinstance(norm, nn.RMSNorm):
        h = F.rms_norm(h, norm.normalized_shape, f32(norm.weight), norm.eps)
    else:  # nn.LayerNorm
        h = F.layer_norm(h, norm.normalized_shape, f32(norm.weight),
                         f32(norm.bias), norm.eps)
    head = model.lm_head if hasattr(model, 'lm_head') else model.token_emb
    return F.linear(h, head.weight.float())

class TunedLens(nn.Module):
    """One affine translator h + A_l h + b_l per block but the last (the
    identity), applied before the logit lens; A_l and b_l start at zero,
    so a fresh lens is the logit lens.

    Defined in :numref:`sec_lm_interpretability`"""
    def __init__(self, d_model, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.translators = nn.ModuleList(
            nn.Linear(d_model, d_model) for _ in range(num_layers - 1))
        for t in self.translators:
            nn.init.zeros_(t.weight)
            nn.init.zeros_(t.bias)

    def forward(self, hidden, layer):
        if not 0 <= layer < self.num_layers:
            raise IndexError(f'layer {layer} of {self.num_layers}')
        if layer == self.num_layers - 1:
            return hidden
        t = self.translators[layer]
        h = hidden.to(t.weight.dtype)
        return h + t(h)

    def _check(self, model):
        if len(model.blks) != self.num_layers:
            raise ValueError(f'a lens of {self.num_layers} layers for a '
                             f'model of {len(model.blks)} blocks')

    def logits(self, model, hidden, layer):
        self._check(model)
        return d2l.logit_lens(model, self(hidden, layer))

    def fit(self, model, batches, *, steps, lr=1e-3, frac=0.25, seed=0,
            chunk=512):
        """Adam on KL(final || lens), summed over the layers, at a random
        fraction frac of the positions >= 1 of each sequence of each batch
        (equal-length (B, T) ids) of the frozen model. Each layer's loss is
        backpropagated alone, chunk positions at a time, so the memory
        holds one chunk of logits. Returns the loss of each step."""
        assert not any(p.requires_grad for p in model.parameters()), (
            'freeze the model first: model.requires_grad_(False)')
        self._check(model)
        L, dev = self.num_layers, self.translators[0].weight.device
        sites = [('resid_post', l) for l in range(L)]
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        gen, batches, losses = torch.Generator().manual_seed(seed), iter(
            batches), []
        for _ in range(steps):
            X = next(batches, None)
            if X is None:
                raise ValueError(f'batches ran out before {steps} steps')
            with torch.no_grad(), d2l.intervene(
                    model, record=sites, stop_after=sites[-1]) as ctx:
                model(X.to(model.token_emb.weight.device))
            B, T = X.shape
            n = max(1, round(frac * (T - 1)))  # positions per sequence
            cols = torch.stack([torch.randperm(T - 1, generator=gen)[:n] + 1
                                for _ in range(B)]).flatten()
            rows = torch.arange(B).repeat_interleave(n)
            H = [ctx.cache[s][0][rows, cols].to(dev, torch.float32)
                 for s in sites]  # each (B * n, d), detached
            del ctx
            optimizer.zero_grad()
            total = 0.0
            for i in range(0, len(rows), chunk):
                with torch.no_grad():  # the final distribution, the target
                    target = F.log_softmax(
                        d2l.logit_lens(model, H[-1][i:i + chunk]), -1)
                for l in range(L - 1):
                    lens = F.log_softmax(self.logits(model, H[l][i:i + chunk],
                                                     l), -1)
                    kl = F.kl_div(lens, target, reduction='sum',
                                  log_target=True) / len(rows)
                    kl.backward()
                    total += kl.item()
            optimizer.step()
            losses.append(total)
        return losses

class TopKSAE(nn.Module):
    """Top-k sparse autoencoder with Qwen-Scope's parameter names and
    shapes: W_enc (d_sae, d_model), b_enc, W_dec (d_model, d_sae), b_dec.
    With center=True the encoder subtracts b_dec from its input first.
    seed fixes the random initial dictionary (default: the global RNG).

    Defined in :numref:`sec_lm_interpretability`"""
    def __init__(self, d_model, d_sae, k, *, center=False, seed=None):
        super().__init__()
        self.d_model, self.d_sae, self.k = d_model, d_sae, k
        self.center = center  # a plain attribute: the encoder's convention
        gen = None if seed is None else torch.Generator().manual_seed(seed)
        W = torch.randn(d_model, d_sae, generator=gen)
        W = W / W.norm(dim=0, keepdim=True)  # unit decoder columns
        self.W_dec = nn.Parameter(W)
        self.W_enc = nn.Parameter(W.T.clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def pre_acts(self, x):
        x = x.to(self.W_enc.dtype)
        return F.linear(x - self.b_dec if self.center else x, self.W_enc,
                        self.b_enc)

    @staticmethod
    def topk_relu(pre, k):
        """ReLU of the k largest entries of each row, zero elsewhere; equal
        values at the cut go to the lower indices."""
        cut = pre.topk(k, dim=-1).values[..., -1:]
        keep = pre >= cut
        ties = keep.sum(-1) > k  # rows where the cut value repeats
        if ties.any():
            p, c = pre[ties], cut[ties]
            above, at = p > c, p == c
            keep[ties] = above | (at & (at.cumsum(-1)
                                        <= k - above.sum(-1, keepdim=True)))
        return torch.where(keep, F.relu(pre), 0.0)

    def encode(self, x):
        return self.topk_relu(self.pre_acts(x), self.k)

    def decode(self, z):
        return F.linear(z, self.W_dec, self.b_dec)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

    def loss(self, x, dead=None, k_aux=256):
        """Squared error normalized by the batch variance, and the auxiliary
        loss: the same for the residual x - x_hat reconstructed from the
        k_aux largest pre-activations of the dead features (a mask)."""
        x = x.to(self.W_enc.dtype).reshape(-1, self.d_model)
        pre = self.pre_acts(x)
        x_hat = self.decode(self.topk_relu(pre, self.k))
        nmse = lambda u, v: ((u - v) ** 2).sum(-1).mean() / (
            (v - v.mean(0)) ** 2).sum(-1).mean()
        aux = x_hat.new_zeros(())
        if dead is not None and dead.any():
            e = (x - x_hat).detach()
            z = self.topk_relu(pre.masked_fill(~dead, -torch.inf),
                               min(k_aux, int(dead.sum())))
            aux = nmse(F.linear(z, self.W_dec), e)
        return nmse(x_hat, x), aux

    @torch.no_grad()
    def normalize_decoder_(self):
        """Project each decoder column onto the unit sphere (columns already
        of norm 1 to within 1e-6 are left as they are)."""
        n = self.W_dec.norm(dim=0, keepdim=True)
        self.W_dec.div_(torch.where((n - 1).abs() > 1e-6, n.clamp_min(1e-12),
                                    1.0))

    @torch.no_grad()
    def remove_parallel_grad_(self):
        """Remove from W_dec.grad each column's component along the column,
        the part a unit-norm projection would undo."""
        W, G = self.W_dec, self.W_dec.grad
        if G is not None:
            G.sub_(W * (W * G).sum(0, keepdim=True) / (W * W).sum(
                0, keepdim=True))

def train_sae(sae, activations, *, steps, batch_size=4096, lr=2e-4,
              aux_coef=1/32, dead_after=200, seed=0):
    """Train a TopKSAE with Adam on random rows of activations, an (N, d)
    CPU tensor (fp16 allowed), with unit decoder columns and the auxiliary
    loss of the features that have not fired for dead_after steps. seed
    fixes the batches; the initial dictionary is the SAE's own (see
    TopKSAE's seed). Returns the loss of each step.

    Defined in :numref:`sec_lm_interpretability`"""
    dev, dtype = sae.W_dec.device, sae.W_dec.dtype
    gen, N = torch.Generator().manual_seed(seed), len(activations)
    with torch.no_grad():  # b_dec at the data mean, W_enc = W_dec^T
        sample = activations[torch.randperm(N, generator=gen)[:65536]]
        sae.b_dec.copy_(sample.to(dev, dtype).mean(0))
        sae.W_dec.div_(sae.W_dec.norm(dim=0, keepdim=True))
        sae.W_enc.copy_(sae.W_dec.T)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    idle = torch.zeros(sae.d_sae, dtype=torch.long, device=dev)
    losses = []
    for _ in range(steps):
        x = activations[torch.randint(N, (batch_size,), generator=gen)].to(
            dev, dtype)
        mse, aux = sae.loss(x, idle >= dead_after)
        loss = mse + aux_coef * aux
        with torch.no_grad():  # steps since each feature last fired
            fired = (sae.encode(x) > 0).any(0)
            idle = torch.where(fired, 0, idle + 1)
        optimizer.zero_grad()
        loss.backward()
        sae.remove_parallel_grad_()
        optimizer.step()
        sae.normalize_decoder_()
        losses.append(loss.item())
    return losses

def load_qwen_scope_sae(repo_id, layer, revision=None, *, device=None,
                        dtype=torch.float32, center=False):
    """The TopKSAE of one layer of a Qwen-Scope repository, downloaded
    at the pinned revision and loaded with weights_only=True.

    Defined in :numref:`sec_lm_interpretability`"""
    from huggingface_hub import hf_hub_download
    revision = revision or d2l.HF_REVISIONS[repo_id]
    get = lambda f: hf_hub_download(repo_id, f, revision=revision)
    with open(get('config.json')) as f:
        cfg = json.load(f)
    if cfg.get('model_type') != 'topk_sae':
        raise ValueError(f'not a top-k SAE: {cfg.get("model_type")!r}')
    if layer not in cfg['layers']:
        raise ValueError(f'{repo_id} has no layer {layer}: {cfg["layers"]}')
    state = torch.load(get(f'layer{layer}.sae.pt'), weights_only=True,
                       map_location='cpu')
    d, m = cfg['d_model'], cfg['d_sae']
    shapes = {'W_enc': (m, d), 'b_enc': (m,), 'W_dec': (d, m), 'b_dec': (d,)}
    if not isinstance(state, dict) or set(state) != set(shapes):
        found = sorted(state) if isinstance(state, dict) else type(state)
        raise KeyError(f'expected the tensors {sorted(shapes)}, found {found}')
    for name, shape in shapes.items():
        if tuple(state[name].shape) != shape:
            raise ValueError(f'{name}: shape {tuple(state[name].shape)}, '
                             f'expected {shape}')
    with torch.device('meta'):  # no random initialization to overwrite
        sae = TopKSAE(d, m, cfg['k'], center=center)
    sae.load_state_dict(state, assign=True)
    return sae.to(device=device, dtype=dtype)

@torch.no_grad()
def sae_metrics(sae, model, token_batches, layer):
    """Reconstruction statistics of sae on the stream after block layer,
    over token_batches (equal-length (B, T) ids), at the positions >= 1
    whose stream norm is at most 10 times the batch median: L0, pooled FVU,
    the next-token cross-entropy of the model, with x_hat spliced in, and
    with the mean activation, loss recovered, the fraction of features
    that never fire, each feature's firing rate, and per-row values:
    sse and sst over the n kept positions, mean ce_orig and ce_splice
    over the n_ce kept positions that have a next token (NaN if none).
    The cross-entropy goes one row at a time, so the fp32 buffers hold
    one row of logits.

    Defined in :numref:`sec_lm_interpretability`"""
    if sae.W_dec.shape[0] != model.num_hiddens:
        raise ValueError(f'SAE width {sae.W_dec.shape[0]} != model width '
                         f'{model.num_hiddens}')
    site, dev = ('resid_post', layer), model.token_emb.weight.device
    token_batches = list(token_batches)  # read twice

    def ce(logits, X, keep):  # per row: summed loss and count, kept t
        loss = torch.stack([F.cross_entropy(row[:-1].float(), x[1:],
                                            reduction='none')
                            for row, x in zip(logits, X)])
        m = keep[:, :-1]
        return (loss * m).sum(1), m.sum(1)

    keeps, ce_orig, total, n = [], [], 0.0, 0
    for X in token_batches:  # pass 1: the clean run and the kept positions
        X = X.to(dev)
        with d2l.intervene(model, record=[site]) as ctx:
            logits = model(X)
        x = ctx.cache[site][0].float()
        norms = x.norm(dim=-1)
        keep = norms <= 10 * norms[:, 1:].median()
        keep[:, 0] = False
        keeps.append(keep)
        ce_orig.append(ce(logits, X, keep))
        del logits  # not alive during the next forward
        total = total + x[keep].double().sum(0)
        n += int(keep.sum())
    mean = (total / n).float()
    rows = {'sse': [], 'sst': []}
    fired = torch.zeros(sae.d_sae, device=sae.W_dec.device)
    nnz, ce_splice, ce_mean = 0, [], []
    for X, keep in zip(token_batches, keeps):  # pass 2: the two splices
        X, k = X.to(dev), keep[..., None]

        def splice(x):
            nonlocal nnz, fired
            x_hat, z = sae(x)
            xf, x_hat = x.float(), x_hat.float()
            rows['sse'].append((((x_hat - xf) ** 2).sum(-1) * keep).sum(1))
            rows['sst'].append((((xf - mean) ** 2).sum(-1) * keep).sum(1))
            active = (z > 0) & k
            nnz += int(active.sum())
            fired = fired + active.flatten(0, 1).sum(0)
            return torch.where(k, x_hat.to(x.dtype), x)

        with d2l.intervene(model, {site: splice}):
            ce_splice.append(ce(model(X), X, keep))
        with d2l.intervene(model, {site: lambda x: torch.where(
                k, mean.to(x.dtype), x)}):
            ce_mean.append(ce(model(X), X, keep))
    cat = lambda xs: torch.cat(xs).double().cpu().numpy()
    per_row = lambda pairs: cat([s / c for s, c in pairs])  # 0/0: NaN
    pooled = lambda pairs: (sum(float(s.sum()) for s, _ in pairs)
                            / sum(int(c.sum()) for _, c in pairs))
    ce_o, ce_s, ce_m = pooled(ce_orig), pooled(ce_splice), pooled(ce_mean)
    sse, sst = cat(rows['sse']), cat(rows['sst'])
    density = (fired / n).cpu().numpy()
    return {'l0': nnz / n, 'fvu': float(sse.sum() / sst.sum()),
            'ce_orig': ce_o, 'ce_splice': ce_s, 'ce_mean_ablate': ce_m,
            'loss_recovered': (ce_m - ce_s) / (ce_m - ce_o),
            'unfired_frac': float((density == 0).mean()), 'density': density,
            'n_tokens': n,
            'per_doc': {'sse': sse, 'sst': sst,
                        'n': cat([keep.sum(1) for keep in keeps]),
                        'n_ce': cat([c for _, c in ce_orig]),
                        'ce_orig': per_row(ce_orig),
                        'ce_splice': per_row(ce_splice)}}

def request_hash(base_url, model, messages, params, replica=0):
    """sha256 of the canonical JSON (sorted keys, no whitespace, UTF-8) of
    what can change a response: the endpoint's hostname, the model, the
    messages, the merged parameters (None values dropped) without the
    transport fields stream, stream_options, timeout and extra_headers, and
    the client-side replica number.

    Defined in :numref:`sec_lm_api`"""
    from urllib.parse import urlsplit  # not in the d2l library preamble
    transport = ('stream', 'stream_options', 'timeout', 'extra_headers')
    params = {k: v for k, v in d2l._chat_merge(params, {}).items()
              if k not in transport}
    key = {'host': urlsplit(base_url).hostname, 'model': model,
           'messages': messages, 'params': params, 'replica': replica}
    blob = json.dumps(key, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()

PricePoint = collections.namedtuple(  # USD per `per` tokens
    'PricePoint', ['input', 'output', 'cached_input', 'per'],
    defaults=(None, 1_000_000))

def api_cost(usage, price):
    """Cost in USD of one response, ((p - c) input + c cached_input +
    o output) / per, with p prompt, c cached and o completion tokens from
    usage (reasoning tokens are inside o). cached_input None means no
    discount; missing cached tokens count as 0; missing prompt or
    completion tokens raise ValueError.

    Defined in :numref:`sec_lm_api`"""
    p, o = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if p is None or o is None:
        raise ValueError(f'usage lacks prompt or completion tokens: {usage}')
    c = usage.get('cached_tokens') or 0
    cached = price.input if price.cached_input is None else price.cached_input
    return ((p - c) * price.input + c * cached + o * price.output) / price.per

def backoff_delay(attempt, base=1.0, cap=60.0, rng=None, retry_after=None):
    """Full-jitter sleep after failed attempt `attempt` (0, 1, ...): uniform
    on [0, min(cap, base 2^attempt)], and at least the server's
    Retry-After.

    Defined in :numref:`sec_lm_api`"""
    rng = rng or random
    return max(retry_after or 0, rng.uniform(0, min(cap, base * 2**attempt)))

async def chat_batch(api, requests, *, concurrency=8):
    """Send requests (dicts with 'messages', optionally 'replica' and request
    parameters) through api with at most `concurrency` in flight; results
    come in input order, a failed request as its exception. A replay miss
    or a capped call is not a result: once every request has finished (the
    paid ones are cached), the first d2l.ReplayMiss or d2l.CostCapExceeded
    is raised, with the whole list as its `results` attribute.

    Defined in :numref:`sec_lm_api`"""
    import asyncio  # not in the d2l library preamble
    semaphore = asyncio.Semaphore(concurrency)
    async def one(request):
        async with semaphore:
            try:
                request = dict(request)
                return await api.achat(request.pop('messages'), **request)
            except Exception as e:
                return e
    results = await asyncio.gather(*(one(r) for r in requests))
    for r in results:
        if isinstance(r, (d2l.ReplayMiss, d2l.CostCapExceeded)):
            r.results = results
            raise r
    return results

def closed_loop_stats(results, c):
    """Throughput and latency of a closed loop that kept c requests in
    flight. Exceptions count as n_failed and are excluded. For the N
    remaining results let s_i = t_send, f_i = s_i + t_total and f_(1) <= ...
    <= f_(N) the sorted completion times; N <= 2c raises ValueError. The
    steady window runs from t_a = f_(c) to t_b = f_(N-c). X counts the
    completions with t_a < f_i <= t_b per second of the window and X_tokens
    their completion tokens; W is the mean of f_i - s_i over requests with
    s_i >= t_a and f_i <= t_b; Lbar = sum_i max(0, min(f_i, t_b) -
    max(s_i, t_a)) / (t_b - t_a) is the mean number in flight. little =
    X W / c checks the client's bookkeeping, sustained = Lbar / c that it
    kept c in flight. X_tokens is None when a counted completion has no
    usage. ttft_p50 and ttft_p90 (from t_first) are over all N. The results
    must come from one closed loop: a d2l.ReplayMiss or d2l.CostCapExceeded
    among them, a mix of cached and live results, more than c requests in
    flight at once, or send times whose wall and monotonic clocks disagree
    by more than 2 s (two recording runs) raise ValueError.

    Defined in :numref:`sec_lm_api`"""
    d2l._check_closed_loop(results, c)  # one loop of c, from one run
    ok = [r for r in results if not isinstance(r, BaseException)]
    N = len(ok)
    if N <= 2 * c:
        raise ValueError(f'{N} results at c = {c}; need more than 2c')
    s = np.array([r.t_send for r in ok])
    f = s + np.array([r.t_total for r in ok])
    t_a, t_b = np.sort(f)[[c - 1, N - c - 1]]
    window = t_b - t_a
    if window <= 0:
        raise ValueError('the steady window is empty')
    done = (f > t_a) & (f <= t_b)  # completions inside the window
    inside = (s >= t_a) & (f <= t_b)  # requests wholly inside it
    tokens = [r.usage.get('completion_tokens') for r, d in zip(ok, done) if d]
    X = done.sum() / window
    W = (f - s)[inside].mean() if inside.any() else float('nan')
    Lbar = np.clip(np.minimum(f, t_b) - np.maximum(s, t_a), 0, None).sum()
    ttft = [r.t_first - r.t_send for r in ok if r.t_first is not None]
    p50, p90 = (map(float, np.percentile(ttft, [50, 90])) if ttft
                else (None, None))
    X_tokens = None if None in tokens else float(sum(tokens) / window)
    return {'X': float(X), 'X_tokens': X_tokens,
            'W': float(W), 'Lbar': float(Lbar / window),
            'little': float(X * W / c), 'sustained': float(Lbar / window / c),
            'ttft_p50': p50, 'ttft_p90': p90,
            'n_429': sum(getattr(r, 'n_429', 0) or 0 for r in results),
            'n_retries': sum(getattr(r, 'n_retries', 0) or 0 for r in results),
            'n_failed': len(results) - N, 'window': float(window),
            'n_window': int(done.sum())}

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum() if l.numel() > 1 else l * y.numel()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
               devices=d2l.try_all_gpus()):
    """Train a model with multiple GPUs (defined in Chapter 13).

    Defined in :numref:`sec_image_augmentation`"""
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of examples
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {metric[0] / metric[2]:.3f}, train acc '
          f'{metric[1] / metric[3]:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on '
          f'{str(devices)}')

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
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)

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
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
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
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = torch.div(max_idx, num_gt_boxes, rounding_mode='floor')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
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
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
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
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression.

    Defined in :numref:`sec_anchor`"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)

d2l.DATA_HUB['banana-detection'] = (
    d2l.DATA_URL + 'banana-detection.zip',
    '5de26c8fce5ccdea9f91267273464dc968d20d72')

def read_data_bananas(is_train=True):
    """Read the banana detection dataset images and labels.

    Defined in :numref:`sec_object-detection-dataset`"""
    data_dir = d2l.download_extract('banana-detection')
    csv_fname = os.path.join(data_dir, 'bananas_train' if is_train
                             else 'bananas_val', 'label.csv')
    csv_data = pd.read_csv(csv_fname)
    csv_data = csv_data.set_index('img_name')
    images, targets = [], []
    for img_name, target in csv_data.iterrows():
        images.append(torchvision.io.read_image(
            os.path.join(data_dir, 'bananas_train' if is_train else
                         'bananas_val', 'images', f'{img_name}')))
        # Here `target` contains (class, upper-left x, upper-left y,
        # lower-right x, lower-right y), where all the images have the same
        # banana class (index 0)
        targets.append(list(target))
    return images, torch.tensor(targets).unsqueeze(1) / 256

class BananasDataset(torch.utils.data.Dataset):
    """A customized dataset to load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    def __init__(self, is_train):
        self.features, self.labels = read_data_bananas(is_train)
        print('read ' + str(len(self.features)) + (f' training examples' if
              is_train else f' validation examples'))

    def __getitem__(self, idx):
        return (self.features[idx].float(), self.labels[idx])

    def __len__(self):
        return len(self.features)

def load_data_bananas(batch_size):
    """Load the banana detection dataset.

    Defined in :numref:`sec_object-detection-dataset`"""
    train_iter = torch.utils.data.DataLoader(BananasDataset(is_train=True),
                                             batch_size, shuffle=True)
    val_iter = torch.utils.data.DataLoader(BananasDataset(is_train=False),
                                           batch_size)
    return train_iter, val_iter

d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

def read_voc_images(voc_dir, is_train=True):
    """Read all VOC feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    txt_fname = os.path.join(voc_dir, 'ImageSets', 'Segmentation',
                             'train.txt' if is_train else 'val.txt')
    mode = torchvision.io.image.ImageReadMode.RGB
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        features.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'JPEGImages', f'{fname}.jpg')))
        labels.append(torchvision.io.read_image(os.path.join(
            voc_dir, 'SegmentationClass' ,f'{fname}.png'), mode))
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
    colormap2label = torch.zeros(256 ** 3, dtype=torch.long)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[
            (colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label

def voc_label_indices(colormap, colormap2label):
    """Map any RGB values in VOC labels to their class indices.

    Defined in :numref:`sec_semantic_segmentation`"""
    colormap = colormap.permute(1, 2, 0).numpy().astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]

def voc_rand_crop(feature, label, height, width):
    """Randomly crop both feature and label images.

    Defined in :numref:`sec_semantic_segmentation`"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

class VOCSegDataset(torch.utils.data.Dataset):
    """A customized dataset to load the VOC dataset.

    Defined in :numref:`sec_semantic_segmentation`"""

    def __init__(self, is_train, crop_size, voc_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_voc_images(voc_dir, is_train=is_train)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        self.colormap2label = voc_colormap2label()
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return [img for img in imgs if (
            img.shape[1] >= self.crop_size[0] and
            img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = voc_rand_crop(self.features[idx], self.labels[idx],
                                       *self.crop_size)
        return (feature, voc_label_indices(label, self.colormap2label))

    def __len__(self):
        return len(self.features)

def load_data_voc(batch_size, crop_size):
    """Load the VOC semantic segmentation dataset.

    Defined in :numref:`sec_semantic_segmentation`"""
    voc_dir = d2l.download_extract('voc2012', os.path.join(
        'VOCdevkit', 'VOC2012'))
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        VOCSegDataset(True, crop_size, voc_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        VOCSegDataset(False, crop_size, voc_dir), batch_size,
        drop_last=True, num_workers=num_workers)
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

def show_trace_2d(f, results):
    """Show the trace of 2D variables over the contours of f.

    Defined in :numref:`sec_utils`"""
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

def plot_intervals(x, series, xlabel, ylabel, *, kind='errorbar',
                   reference=None, xscale='linear', axes=None):
    """Estimates with intervals: `series` maps a label to (mean, lo, hi).

    Defined in :numref:`sec_utils`"""
    if kind not in ('errorbar', 'band') or xscale not in ('linear', 'log',
                                                          'log2'):
        raise ValueError(f'unknown kind {kind!r} or xscale {xscale!r}')
    x = np.asarray(x)
    categorical = not np.issubdtype(x.dtype, np.number)
    pos = np.arange(len(x), dtype=float) if categorical else x.astype(float)
    logx = xscale != 'linear' and not categorical
    # Side-by-side offsets, in the units in which the x-axis is even
    to_axis, from_axis = (np.log, np.exp) if logx else (np.asarray,) * 2
    gaps = np.diff(np.unique(to_axis(pos)))
    width = 0.6 * (gaps.min() if len(gaps) else 1.0) / len(series)
    if axes is None:
        d2l.set_figsize()
        axes = d2l.plt.gca()
    for i, (label, (mean, lo, hi)) in enumerate(series.items()):
        mean, lo, hi = (np.asarray(v, dtype=float) for v in (mean, lo, hi))
        if not mean.shape == lo.shape == hi.shape == pos.shape:
            raise ValueError(f'{label}: mean, lo and hi need {len(pos)} '
                             'values each')
        err = np.stack([mean - lo, hi - mean])
        if np.any(err < -1e-9):  # nan (a missing point) passes
            raise ValueError(f'{label}: an interval misses its estimate')
        color = f'C{i % 10}'
        if kind == 'band':
            axes.plot(pos, mean, '-', color=color, label=label)
            axes.fill_between(pos, lo, hi, color=color, alpha=0.2,
                              linewidth=0)
        else:
            at = from_axis(to_axis(pos) + (i - (len(series) - 1) / 2) * width)
            axes.errorbar(at, mean, yerr=np.clip(err, 0, None), fmt='o',
                          color=color, markersize=4, capsize=3, label=label)
    if reference is not None:
        axes.axhline(reference, linestyle='--', color='black', linewidth=0.8)
    if logx:
        axes.set_xscale('log', base=2 if xscale == 'log2' else 10)
    if categorical or logx:  # One tick per x value
        axes.set_xticks(pos, [str(v) if categorical else f'{v:g}' for v in x])
        axes.xaxis.set_minor_locator(ticker.NullLocator())
    if categorical and len(x) > 3:  # Slanted names do not collide
        axes.set_xticklabels(axes.get_xticklabels(), rotation=30, ha='right')
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    if len(series) > 1:
        axes.legend()
    axes.grid(axis='y' if categorical else 'both')  # No line through a bar
    return axes

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

def plain_sgd(net, lr):
    """A plain SGD optimizer over the parameters of `net`.

    Defined in :numref:`sec_utils`"""
    return torch.optim.SGD(net.parameters(), lr=lr)

def load_cifar10_arrays():
    """CIFAR-10 on the GPU if there is one, pixel values scaled to [-1, 1].
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
        X = torch.tensor(batch[b'data'], dtype=torch.float32)
        return X.reshape(-1, 3, 32, 32) / 127.5 - 1, torch.tensor(
            batch[b'labels'])
    Xs, ys = zip(*[load_batch(f'data_batch_{i}') for i in range(1, 6)])
    test_X, test_y = load_batch('test_batch')
    device = d2l.try_gpu()
    return (torch.cat(Xs).to(device), torch.cat(ys).to(device),
            test_X.to(device), test_y.to(device))

class FeatureCNN(nn.Module):
    """Three conv blocks, global average pooling, and a linear head. Inputs
    are first resized bilinearly to resolution x resolution.

    Defined in :numref:`sec_utils`"""
    def __init__(self, feature_dim=128, resolution=32):
        super().__init__()
        self.resolution = resolution
        widths = [3, 32, 64, feature_dim]
        self.blocks = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1),
                           nn.BatchNorm2d(c_out), nn.ReLU(),
                           nn.MaxPool2d(2))
             for c_in, c_out in zip(widths[:-1], widths[1:])])
        self.head = nn.Linear(feature_dim, 10)

    def features(self, x):
        if x.shape[-1] != self.resolution:
            x = F.interpolate(x, size=(self.resolution, self.resolution),
                              mode='bilinear', antialias=True)
        for block in self.blocks:
            x = block(x)
        return x.mean(dim=(2, 3))

    def forward(self, x):
        return self.head(self.features(x))

def train_feature_cnn(cnn, X, y, num_epochs=15, max_lr=0.003,
                      batch_size=256, shift=0, seed=0):
    """Train on random horizontal flips, plus random shifts of up to `shift`
    pixels, with a one-cycle learning rate; end in evaluation mode.

    Defined in :numref:`sec_utils`"""
    gen = torch.Generator(X.device).manual_seed(seed)  # local random streams
    gen_shift = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(cnn.parameters())
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr, total_steps=num_epochs * (len(X) // batch_size),
        cycle_momentum=False)
    h, w = X.shape[2:]
    cnn.train()
    for epoch in range(num_epochs):
        order = torch.randperm(len(X), device=X.device, generator=gen)
        for i in range(0, len(order) - batch_size + 1, batch_size):
            Xb, yb = X[order[i:i + batch_size]], y[order[i:i + batch_size]]
            flip = torch.rand(len(Xb), 1, 1, 1, device=X.device,
                              generator=gen) < 0.5
            Xb = torch.where(flip, Xb.flip(-1), Xb)
            a, b = torch.randint(0, 2 * shift + 1, (2,),
                                 generator=gen_shift).tolist()
            Xb = F.pad(Xb, (shift,) * 4, mode='reflect')[..., a:a + h, b:b + w]
            loss = F.cross_entropy(cnn(Xb), yb)
            opt.zero_grad(), loss.backward(), opt.step(), sched.step()
    cnn.eval()

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent.

    Defined in :numref:`sec_utils`"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def get_dataloader_workers():
    """Use 4 processes to read the data.

    Defined in :numref:`sec_utils`"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load it into memory.

    Defined in :numref:`sec_utils`"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (torch.utils.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                        num_workers=get_dataloader_workers()),
            torch.utils.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                        num_workers=get_dataloader_workers()))

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), d2l.size(y))
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """Train a model with a GPU (defined in Chapter 6).

    Defined in :numref:`sec_utils`"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

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
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]

def grad_clipping(net, theta):
    """Clip the gradient.

    Defined in :numref:`sec_utils`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

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
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks.

    Defined in :numref:`sec_utils`"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def __init__(self):
        super().__init__(reduction='none')

    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = d2l.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # Make the loss scalar for `backward`
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_utils`"""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

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
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
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

d2l.DATA_HUB['ml-100k'] = (
    'https://files.grouplens.org/datasets/movielens/ml-100k.zip',
    'cd4dcac4241c8a4ad7badc7ca635da8a69dddb83')

def read_data_ml100k():
    data_dir = d2l.download_extract('ml-100k')
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), sep='\t',
                       names=names, engine='python')
    num_users = data.user_id.unique().shape[0]
    num_items = data.item_id.unique().shape[0]
    return data, num_users, num_items

def split_data_ml100k(data, num_users, num_items,
                      split_mode='random', test_ratio=0.1):
    """Split the dataset in random mode or seq-aware mode."""
    if split_mode == 'seq-aware':
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u, []).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        # O(N) set-membership filter instead of O(N^2) list-membership.
        test_set = set(test_data)
        train_data = [item for item in train_list if item not in test_set]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        # Seed for deterministic splits across frameworks; uses Python's
        # `random` for cross-framework portability (some frameworks' numpy
        # shim lacks `np.random.default_rng`).
        rng = random.Random(0)
        mask = [rng.random() < 1 - test_ratio for _ in range(len(data))]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data

def load_data_ml100k(data, num_users, num_items, feedback='explicit'):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == 'explicit' else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == 'explicit' else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == 'implicit':
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter

def split_and_load_ml100k(split_mode='seq-aware', feedback='explicit',
                          test_ratio=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_ratio)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback)
    train_set = torch.utils.data.TensorDataset(
        torch.tensor(train_u), torch.tensor(train_i),
        torch.tensor(train_r).float())
    test_set = torch.utils.data.TensorDataset(
        torch.tensor(test_u), torch.tensor(test_i),
        torch.tensor(test_r).float())
    train_iter = torch.utils.data.DataLoader(
        train_set, shuffle=True, drop_last=False,
        batch_size=batch_size)
    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size)
    return num_users, num_items, train_iter, test_iter

def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    net = net.to(devices[0])
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        net.train()
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            users, items, ratings = [v.to(devices[0]) for v in values]
            trainer.zero_grad()
            preds = net(users, items)
            ls = loss(preds, ratings)
            ls.backward()
            trainer.step()
            l += ls.item()
            metric.add(ls.item() * users.shape[0], users.shape[0],
                       users.shape[0])
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')

class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()

    def forward(self, positive, negative):
        distances = positive - negative
        loss = -torch.sum(torch.log(torch.sigmoid(distances)), dim=0,
                          keepdim=True)
        return loss

class HingeLossbRec(nn.Module):
    def __init__(self):
        super(HingeLossbRec, self).__init__()

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = torch.sum(torch.clamp(-distances + margin, min=0))
        return loss

def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc

def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_items)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, scores = [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x = [torch.tensor(user_ids)]
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([torch.tensor(item_ids)])
        test_data_iter = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*x), shuffle=False,
            batch_size=1024)
        for values in test_data_iter:
            values = [v.to(devices[0]) for v in values]
            # `net` returns a 1-D tensor of logits per batch; ravel
            # ensures we always extend with scalars regardless of whether
            # the model emits shape (B,) or (B, 1).
            scores.extend(net(*values).detach().cpu().numpy().ravel().tolist())
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return sum(hit_rate) / len(hit_rate), sum(auc) / len(auc)

def train_ranking(net, train_iter, test_iter, loss, optimizer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['test hit rate', 'test AUC'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        for i, values in enumerate(train_iter):
            input_data = [v.to(devices[0]) for v in values]
            p_pos = net(*input_data[:-1])
            p_neg = net(*input_data[:-2], input_data[-1])
            ls = loss(p_pos, p_neg)
            optimizer.zero_grad()
            ls.backward()
            optimizer.step()
            # Per-batch loss only; accumulating across batches inside `l`
            # turned the printed train-loss into a quadratic sum.
            metric.add(ls.item(), values[0].shape[0], values[0].numel())
            timer.stop()
        with torch.no_grad():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
                animator.add(epoch + 1, (hit_rate, auc))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')

d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

class CTRDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = torch.zeros(self.NUM_FEATS, dtype=torch.long)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                instance['y'] = [float(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(sorted(feat_values))}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = torch.tensor(
            (0, *torch.cumsum(self.field_dims, dim=0).numpy()[:-1]))

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        feat = torch.tensor([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                             for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, torch.tensor(self.data[idx]['y'])

class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError

class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture.

    Defined in :numref:`sec_encoder-decoder`"""
    def __init__(self):
        super().__init__()

    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
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

    def predict_step(self, batch, device, num_steps,
                     save_attention_weights=False):
        self.eval()
        batch = [d2l.to(a, device) for a in batch]
        src, tgt, src_valid_len, _ = batch
        enc_all_outputs = self.encoder(src, src_valid_len)
        dec_state = self.decoder.init_state(enc_all_outputs, src_valid_len)
        outputs, attention_weights = [d2l.expand_dims(tgt[:, 0], 1), ], []
        for _ in range(num_steps):
            Y, dec_state = self.decoder(outputs[-1], dec_state)
            outputs.append(d2l.argmax(Y, 2))
            if save_attention_weights:
                attention_weights.append(self.decoder.attention_weights)
        return d2l.concat(outputs[1:], 1), attention_weights

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

def init_seq2seq(module):
    """Initialize weights for sequence-to-sequence learning.

    Defined in :numref:`sec_machine_translation`"""
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
    if type(module) == nn.GRU:
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])

class Seq2SeqEncoder(d2l.Encoder):
    """The RNN encoder for sequence-to-sequence learning.

    Defined in :numref:`sec_machine_translation`"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = d2l.GRU(embed_size, num_hiddens, num_layers, dropout)
        self.apply(init_seq2seq)

    def forward(self, X, *args):
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
        self.save_hyperparameters()

    def validation_step(self, batch):
        return self.loss(self(*batch[:-1]), batch[-1])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

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
    num_workers = d2l.get_dataloader_workers()
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
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(centers).unsqueeze(1),
        torch.from_numpy(cn), torch.from_numpy(masks),
        torch.from_numpy(labels))
    data_iter = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, num_workers=num_workers)
    return data_iter, vocab

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
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module(f"{i}", d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    """The masked language model task of BERT.

    Defined in :numref:`sec_bert`"""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.LazyLinear(num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.LazyLinear(vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat

class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT.

    Defined in :numref:`sec_bert`"""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.LazyLinear(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)

class BERTModel(nn.Module):
    """The BERT model.

    Defined in :numref:`sec_bert`"""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, 
                 num_heads, num_blks, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_blks, dropout,
                                   max_len=max_len)
        self.hidden = nn.Sequential(nn.LazyLinear(num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
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
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)

class _WikiTextDataset(torch.utils.data.Dataset):
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
    num_workers = d2l.get_dataloader_workers()
    paragraphs = _read_wiki()
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab

def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss (mask per-token losses before summing)
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y).mean()
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l

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
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab

def predict_sentiment(net, vocab, sequence):
    """Predict the sentiment of a text sequence.

    Defined in :numref:`sec_sentiment_rnn`"""
    sequence = torch.tensor(vocab[sequence.split()], device=d2l.try_gpu())
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
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

class SNLIDataset(torch.utils.data.Dataset):
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
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)

def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary.

    Defined in :numref:`sec_natural-language-inference-and-dataset`"""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab

def predict_snli(net, vocab, premise, hypothesis):
    """Predict the logical relationship between the premise and hypothesis.

    Defined in :numref:`sec_natural-language-inference-attention`"""
    net.eval()
    premise = torch.tensor(vocab[premise], device=d2l.try_gpu())
    hypothesis = torch.tensor(vocab[hypothesis], device=d2l.try_gpu())
    label = torch.argmax(net([premise.reshape((1, -1)),
                           hypothesis.reshape((1, -1))]), dim=1)
    return 'entailment' if label == 0 else 'contradiction' if label == 1 \
            else 'neutral'

class PositionWiseFFN(nn.Module):
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module):
    """The residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        mask = (None if valid_lens is None
                else d2l.padding_mask(valid_lens, X.shape[1]))
        Y = self.addnorm1(X, self.attention(X, X, X, mask))
        return self.addnorm2(Y, self.ffn(Y))

class GRU(d2l.RNN):
    """The multilayer GRU model implemented with high-level APIs.

    Defined in :numref:`sec_gru`"""
    def __init__(self, num_inputs, num_hiddens, num_layers=1, dropout=0):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = nn.GRU(num_inputs, num_hiddens, num_layers,
                          dropout=dropout)

_EVAL_ROWS = {}  # rows of the files read from the Hub in this process

def _eval_load_rows(task, split=None):
    """The rows (dicts) of the file of task.files for split (default
    task.split): the one file whose path has the split name as a word.
    Read from the Hugging Face Hub at task.revision, or from the folder
    task.dataset if that is a local directory; parquet or JSON lines.

    Defined in :numref:`sec_lm_eval_suites`"""
    split = split or task.split
    word = re.compile(rf'(^|[/_.-]){re.escape(split)}([/_.-]|$)')
    files = [f for f in task.files if word.search(f)]
    if len(files) != 1:
        raise ValueError(f'task {task.id}: {len(files)} files of split '
                         f'{split!r} in {task.files}')
    local = os.path.isdir(task.dataset)
    key = (task.dataset, task.revision, files[0])
    if local or key not in _EVAL_ROWS:
        if local:
            path = os.path.join(task.dataset, files[0])
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(task.dataset, files[0], repo_type='dataset',
                                   revision=task.revision)
        if path.endswith('.parquet'):
            rows = pd.read_parquet(path).to_dict('records')
        else:
            with open(path, encoding='utf-8') as f:
                rows = [json.loads(line) for line in f if line.strip()]
        if local:
            return rows
        _EVAL_ROWS[key] = rows
    return _EVAL_ROWS[key]

def _eval_arc(row, i):
    """ARC: the question, its options and the index of the answer key.

    Defined in :numref:`sec_lm_eval_suites`"""
    labels = [str(a) for a in row['choices']['label']]
    return {'id': str(row['id']), 'question': row['question'],
            'options': [str(o) for o in row['choices']['text']],
            'gold': labels.index(str(row['answerKey']))}

def _eval_hellaswag(row, i):
    """HellaSwag, cleaned as lm-evaluation-harness does: 'label: context'
    as the question, the four endings as options. With fmt='continuation'
    the context is the prompt and each ending continues it, as in the
    harness.

    Defined in :numref:`sec_lm_eval_suites`"""
    def clean(text):
        text = text.strip().replace(' [title]', '. ')
        return re.sub(r'\[.*?\]', '', text).replace('  ', ' ')
    ctx = row['ctx_a'] + ' ' + row['ctx_b'].capitalize()
    return {'id': str(i), 'question': clean(row['activity_label'] + ': ' +
                                            ctx),
            'options': [clean(e) for e in row['endings']],
            'gold': int(row['label'])}

def _eval_mmlu(row, i):
    """MMLU: the question, four options, the answer index, the subject
    (for same-subject demonstrations and as the cluster) and the subject
    header that precedes the prompt. With fmt='mmlu' and five
    'dev:same_subject' shots the prompt is lm-evaluation-harness's.

    Defined in :numref:`sec_lm_eval_suites`"""
    subject = str(row['subject'])
    return {'id': str(i), 'question': row['question'].strip(),
            'options': [str(o) for o in row['choices']],
            'gold': int(row['answer']), 'subject': subject,
            'header': 'The following are multiple choice questions (with '
                      f"answers) about {subject.replace('_', ' ')}.\n\n"}

def _eval_mmlu_pro(row, i):
    """MMLU-Pro: up to ten options, the category as the subject.

    Defined in :numref:`sec_lm_eval_suites`"""
    subject = str(row['category'])
    return {'id': str(row['question_id']), 'question': row['question'].strip(),
            'options': [str(o) for o in row['options']],
            'gold': int(row['answer_index']), 'subject': subject,
            'header': 'The following are multiple choice questions (with '
                      f'answers) about {subject}.\n\n'}

def _eval_piqa(row, i):
    """PIQA: the goal and its two solutions.

    Defined in :numref:`sec_lm_eval_suites`"""
    return {'id': str(i), 'question': row['goal'],
            'options': [row['sol1'], row['sol2']], 'gold': int(row['label'])}

def _eval_sciq(row, i):
    """SciQ: the support paragraph as a header, three distractors and the
    correct answer last, as lm-evaluation-harness orders them.

    Defined in :numref:`sec_lm_eval_suites`"""
    support = (row['support'] or '').strip()
    return {'id': str(i), 'question': row['question'],
            'options': [row['distractor1'], row['distractor2'],
                        row['distractor3'], row['correct_answer']],
            'gold': 3, 'header': support + '\n' if support else ''}

def _eval_winogrande(row, i):
    """WinoGrande, partially scored: the sentence up to the blank with
    either option as two prompts, the rest of the sentence as their
    shared continuation.

    Defined in :numref:`sec_lm_eval_suites`"""
    s = row['sentence']
    k = s.index('_')
    return {'id': str(i), 'prompts': [s[:k] + row['option1'],
                                      s[:k] + row['option2']],
            'continuation': ' ' + s[k + 1:].strip(),
            'gold': int(row['answer']) - 1}

def _eval_gsm8k(row, i):
    """GSM8K: the question, the worked answer without its calculator
    annotations <<...>> (a demonstration's target) and the final number
    after '####' (the reference).

    Defined in :numref:`sec_lm_eval_suites`"""
    return {'id': str(i), 'question': row['question'],
            'target': re.sub(r'<<[^>]*>>', '', row['answer']),
            'reference': d2l.extract_answer(row['answer'], 'gsm8k',
                                            strict=True)}

def _eval_humaneval(row, i):
    """HumanEval: the function header and docstring as the prompt; the
    tests call check() on the entry point.

    Defined in :numref:`sec_lm_eval_suites`"""
    return {'id': str(row['task_id']), 'prompt': row['prompt'],
            'test': f"{row['test']}\n\ncheck({row['entry_point']})\n",
            'entry_point': row['entry_point']}

EVAL_ADAPTERS = {  # dataset row and its index -> an item
    'arc': _eval_arc, 'hellaswag': _eval_hellaswag, 'mmlu': _eval_mmlu,
    'mmlu_pro': _eval_mmlu_pro, 'piqa': _eval_piqa, 'sciq': _eval_sciq,
    'winogrande': _eval_winogrande, 'gsm8k': _eval_gsm8k,
    'humaneval': _eval_humaneval}

def _eval_canonical(task):
    """task with canonical JSON values (numpy numbers and arrays become
    Python ones; files and stop become tuples, a string being one entry).
    Refuses sampling keys that run() sets itself.

    Defined in :numref:`sec_lm_eval_suites`"""
    def plain(x):
        if hasattr(x, 'tolist'):
            return x.tolist()
        raise TypeError(f'task {task.id}: {x!r} is not JSON')
    d = json.loads(json.dumps(task._asdict(), default=plain))
    for key in ('files', 'stop'):
        d[key] = (d[key],) if isinstance(d[key], str) else tuple(d[key] or ())
    clash = sorted(set(d['sampling'] or ()) & {
        'seed', 'stop', 'chat', 'enable_thinking', 'batch_size',
        'max_new_tokens'})
    if clash:
        raise ValueError(f'task {task.id}: sampling sets {clash}, which '
                         'run() derives from the task and its seed')
    return task._replace(**d)

def _eval_items(task, rows, ids):
    """The items rows[ids] of task, adapted by EVAL_ADAPTERS[task.adapter]
    and rendered: 'mc' items get a prompt and choices from format_mc in
    task.fmt with task.shots demonstrations (after the item's header), or,
    with fmt='continuation', the question as the prompt and each option
    after a space as a choice (demonstrations: question and answer, blank
    lines between); 'gen' items get a prompt from fewshot_prompt in
    task.template. The
    demonstrations come from task.shots_from: {'split', 'ids'} (rows of
    that split) or '<split>:same_subject' (the first rows of that split
    with the item's subject).

    Defined in :numref:`sec_lm_eval_suites`"""
    if task.kind not in ('mc', 'mc_partial', 'gen', 'code', 'ppl'):
        raise ValueError(f'task {task.id}: unknown kind {task.kind!r}')
    if task.template in ('chat_single', 'chat_multi') and not task.chat:
        raise ValueError(f'task {task.id}: template {task.template!r} '
                         'needs chat=True')
    adapt = d2l.EVAL_ADAPTERS[task.adapter]
    items = [adapt(rows[int(i)], int(i)) for i in ids]
    src, k = task.shots_from, task.shots or 0
    if k and src is None:
        raise ValueError(f'task {task.id}: {k} shots but no shots_from')
    if isinstance(src, dict):
        if len(src['ids']) != k:
            raise ValueError(f"task {task.id}: shots={k} but "
                             f"{len(src['ids'])} demonstration ids")
        pool = d2l._eval_load_rows(task, src['split'])
        fixed = [adapt(pool[j], j) for j in src['ids']]
    elif isinstance(src, str):
        split, rule = src.split(':')
        if rule != 'same_subject':
            raise ValueError(f'task {task.id}: unknown rule {rule!r}')
        pool = d2l._eval_load_rows(task, split)
        by_subject = collections.defaultdict(list)
        for j, row in enumerate(pool):
            shot = adapt(row, j)
            if len(by_subject[shot['subject']]) < k:
                by_subject[shot['subject']].append(shot)
    for item in items:
        shots = [] if not k else fixed if isinstance(src, dict) else (
            by_subject[item['subject']])
        if k and len(shots) < k:
            raise ValueError(f"task {task.id}: {len(shots)} of {k} shots "
                             f"for item {item['id']}")
        if task.kind == 'mc' and task.fmt == 'continuation':
            demos = [s['question'] + ' ' + s['options'][s['gold']]
                     for s in shots]
            prompt = '\n\n'.join(demos + [item['question']])
            item['prompt'] = item.get('header', '') + prompt
            item['choices'] = [' ' + o for o in item['options']]
        elif task.kind == 'mc':
            prompt, item['choices'] = d2l.format_mc(
                item['question'], item['options'], task.fmt,
                shots=[(s['question'], s['options'], s['gold'])
                       for s in shots])
            item['prompt'] = item.get('header', '') + prompt
        elif task.kind == 'gen':
            demos = [(s['question'], s['target']) for s in shots]
            item['prompt'] = (item['question'] if task.template is None
                              else d2l.fewshot_prompt(demos, item['question'],
                                                      task.template))
    return items

EVAL_TASKS = {t.id: t for t in (  # pinned revisions and explicit files
    EvalTask('arc_challenge', 'allenai/ai2_arc',
             '210d026faf9955653af8916fad021475a3f00453',
             ('ARC-Challenge/test-00000-of-00001.parquet',), 'mc', 'arc',
             config='ARC-Challenge', norm='char'),
    EvalTask('arc_easy', 'allenai/ai2_arc',
             '210d026faf9955653af8916fad021475a3f00453',
             ('ARC-Easy/test-00000-of-00001.parquet',), 'mc', 'arc',
             config='ARC-Easy', norm='char'),
    EvalTask('hellaswag', 'Rowan/hellaswag',
             '218ec52e09a7e7462a5400043bb9a69a41d06b76',
             ('data/validation-00000-of-00001.parquet',), 'mc', 'hellaswag',
             split='validation', fmt='continuation', norm='char'),
    EvalTask('mmlu', 'cais/mmlu', 'c30699e8356da336a370243923dbaf21066bb9fe',
             ('all/test-00000-of-00001.parquet',
              'all/dev-00000-of-00001.parquet'), 'mc', 'mmlu', config='all',
             fmt='mmlu', norm='sum', shots=5,
             shots_from='dev:same_subject', cluster='subject'),
    EvalTask('mmlu_pro', 'TIGER-Lab/MMLU-Pro',
             'b189ec765aa7ed75c8acfea42df31fdae71f97be',
             ('data/test-00000-of-00001.parquet',
              'data/validation-00000-of-00001.parquet'), 'mc', 'mmlu_pro',
             fmt='letter', norm='sum', shots=5,
             shots_from='validation:same_subject'),
    EvalTask('gsm8k', 'openai/gsm8k',
             '740312add88f781978c0658806c59bc2815b9866',
             ('main/test-00000-of-00001.parquet',
              'main/train-00000-of-00001.parquet'), 'gen', 'gsm8k',
             config='main', template='qa', shots=4,
             shots_from={'split': 'train', 'ids': [0, 1, 2, 3]},
             max_new_tokens=384, stop=('\n\nQ:',), extract='gsm8k'),
    EvalTask('humaneval', 'openai/openai_humaneval',
             '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544',
             ('openai_humaneval/test-00000-of-00001.parquet',), 'code',
             'humaneval', max_new_tokens=512,
             stop=('\nclass', '\ndef', '\n#', '\nif', '\nprint')),
    EvalTask('piqa', 'baber/piqa', '142f6d7367fd9877f0fb3b5734ea6a545f54cdd1',
             ('piqa_validation.parquet',), 'mc', 'piqa', split='validation',
             norm='char'),
    EvalTask('sciq', 'allenai/sciq',
             '2c94ad3e1aafab77146f384e23536f97a4849815',
             ('data/test-00000-of-00001.parquet',), 'mc', 'sciq',
             norm='char'),
    EvalTask('winogrande', 'allenai/winogrande',
             '01e74176c63542e6b0bcb004dcdea22d94fb67b5',
             ('winogrande_xl/validation-00000-of-00001.parquet',),
             'mc_partial', 'winogrande', split='validation',
             config='winogrande_xl', norm='sum'))}

EVAL_SUITES = {  # frozen manifests: a change here is a new suite version
    'dev_v1': {'name': 'dev_v1', 'tasks': [
        EvalTask('arc_challenge', 'allenai/ai2_arc',
                 '210d026faf9955653af8916fad021475a3f00453',
                 ('ARC-Challenge/test-00000-of-00001.parquet',), 'mc', 'arc',
                 config='ARC-Challenge', n=500, norm='char'),
        EvalTask('gsm8k', 'openai/gsm8k',
                 '740312add88f781978c0658806c59bc2815b9866',
                 ('main/test-00000-of-00001.parquet',
                  'main/train-00000-of-00001.parquet'), 'gen', 'gsm8k',
                 config='main', n=100, template='chat_multi', chat=True,
                 enable_thinking=False, shots=4,
                 shots_from={'split': 'train', 'ids': [0, 1, 2, 3]},
                 max_new_tokens=384, extract='gsm8k')]},
    'report_v1': {'name': 'report_v1', 'tasks': [
        EvalTask('mmlu_pro', 'TIGER-Lab/MMLU-Pro',
                 'b189ec765aa7ed75c8acfea42df31fdae71f97be',
                 ('data/test-00000-of-00001.parquet',
                  'data/validation-00000-of-00001.parquet'), 'mc',
                 'mmlu_pro', n=1000, fmt='letter', norm='sum', shots=5,
                 shots_from='validation:same_subject'),
        EvalTask('mmlu', 'cais/mmlu',
                 'c30699e8356da336a370243923dbaf21066bb9fe',
                 ('all/test-00000-of-00001.parquet',
                  'all/dev-00000-of-00001.parquet'), 'mc', 'mmlu',
                 config='all', n=2000, fmt='mmlu', norm='sum', shots=5,
                 shots_from='dev:same_subject', cluster='subject'),
        EvalTask('gsm8k', 'openai/gsm8k',
                 '740312add88f781978c0658806c59bc2815b9866',
                 ('main/test-00000-of-00001.parquet',
                  'main/train-00000-of-00001.parquet'), 'gen', 'gsm8k',
                 config='main', template='qa', shots=4,
                 shots_from={'split': 'train', 'ids': [0, 1, 2, 3]},
                 max_new_tokens=384, stop=('\n\nQ:',), extract='gsm8k')]}}

def _eval_model_id(model):
    """(model id, revision) of a Transformers model, from its config;
    other models may set the attributes name_or_path and revision, else
    their class name and 'local' are used.

    Defined in :numref:`sec_lm_eval_suites`"""
    m = d2l._lm_unwrap(model)
    cfg = getattr(m, 'config', None)
    name = (getattr(m, 'name_or_path', None) or
            getattr(cfg, '_name_or_path', None) or type(m).__name__)
    revision = (getattr(m, 'revision', None) or
                getattr(cfg, '_commit_hash', None) or 'local')
    return str(name), str(revision)

def _eval_cache_path(cache_dir, model_id, revision, suite_hash, run_key):
    """<cache_dir>/<model id, '/' as '--'>@<revision>/<suite hash>/<run
    key>.parquet

    Defined in :numref:`sec_lm_eval_suites`"""
    return os.path.join(cache_dir, f"{model_id.replace('/', '--')}@"
                        f'{revision}', suite_hash, f'{run_key}.parquet')

def _eval_sorted(rows):
    """Result rows as a DataFrame sorted by task and item id.

    Defined in :numref:`sec_lm_eval_suites`"""
    return pd.DataFrame(rows).sort_values(['task', 'item_id'],
                                          ignore_index=True)

def _eval_cache_read(path):
    """The result rows cached at path, or [] if there are none.

    Defined in :numref:`sec_lm_eval_suites`"""
    if not os.path.exists(path):
        return []
    return [{k: (None if isinstance(v, float) and k != 'score' and
                 math.isnan(v) else v) for k, v in r.items()}
            for r in pd.read_parquet(path).to_dict('records')]

def _eval_cache_write(path, rows):
    """Write result rows to path atomically, sorted, without timestamps.

    Defined in :numref:`sec_lm_eval_suites`"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _eval_sorted(rows).to_parquet(path + '.tmp', index=False)
    os.replace(path + '.tmp', path)

ChatResult = collections.namedtuple('ChatResult', [  # one chat completion
    'text', 'finish_reason', 'usage', 'logprobs', 'model', 'provider',
    'system_fingerprint', 'reasoning', 'request_hash', 'from_cache',
    't_send', 't_first', 't_first_content', 't_total', 'itl', 'n_retries',
    'n_429', 'created'])

class ReplayMiss(KeyError):
    """A request that is not in the cache of a client in replay mode: a
    build that replays never reaches the network.

    Defined in :numref:`sec_lm_api`"""
    def __str__(self):
        return str(self.args[0]) if self.args else ''

class CostCapExceeded(RuntimeError):
    """A live call whose worst-case cost would cross the client's cost cap;
    raised before any HTTP request is made.

    Defined in :numref:`sec_lm_api`"""

def _chat_merge(base, update):
    """Merge request parameters: keys of update win, nested dicts merge, and
    keys whose value is None are dropped at every level (so None unsets a
    default).

    Defined in :numref:`sec_lm_api`"""
    out = {}
    for k in list(base) + [k for k in update if k not in base]:
        v = d2l._chat_plain(update[k] if k in update else base[k])
        if isinstance(v, dict):
            inner = base.get(k) if k in update else None
            v = d2l._chat_merge(inner if isinstance(inner, dict) else {}, v)
        if v is not None:
            out[k] = v
    return out

def _chat_plain(v):
    """v with NumPy (or torch) scalars, also inside lists and tuples, turned
    into Python numbers, so that they hash and serialize like them.

    Defined in :numref:`sec_lm_api`"""
    if isinstance(v, (list, tuple)):
        return [d2l._chat_plain(a) for a in v]
    if getattr(v, 'ndim', None) == 0 and hasattr(v, 'item'):
        return v.item()
    return v

def _chat_usage(u):
    """The usage block of a response as prompt, completion, cached and
    reasoning tokens and the provider's cost; missing values are None,
    never 0.

    Defined in :numref:`sec_lm_api`"""
    u = u or {}
    prompt = u.get('prompt_tokens_details') or {}
    completion = u.get('completion_tokens_details') or {}
    return {'prompt_tokens': u.get('prompt_tokens'),
            'completion_tokens': u.get('completion_tokens'),
            'cached_tokens': prompt.get('cached_tokens'),
            'reasoning_tokens': completion.get(
                'reasoning_tokens', u.get('reasoning_tokens')),
            'cost': u.get('cost')}

def _chat_logprobs(lp):
    """Generated-token log-probabilities as (token, logprob, top) triples,
    top a list of (token, logprob); None when the response has none.

    Defined in :numref:`sec_lm_api`"""
    if not lp or not lp.get('content'):
        return None
    return [(t['token'], t['logprob'],
             [(a['token'], a['logprob']) for a in t.get('top_logprobs') or []])
            for t in lp['content']]

def _chat_parse(r):
    """The fields of a (non-streamed) chat-completions response dict that
    d2l.ChatResult keeps; unknown top-level fields (such as prompt
    log-probabilities) and the texts of further choices go to 'extra'.

    Defined in :numref:`sec_lm_api`"""
    choices = r.get('choices') or [{}]
    choice, known = choices[0], {'id', 'object', 'created', 'model', 'usage',
                                 'choices', 'provider', 'system_fingerprint'}
    message = choice.get('message') or {}
    extra = {k: v for k, v in r.items() if k not in known and v is not None}
    if len(choices) > 1:
        extra['choices'] = [{'text': (c.get('message') or {}).get('content'),
                             'finish_reason': c.get('finish_reason')}
                            for c in choices]
    return {'text': message.get('content') or '',
            'reasoning': (message.get('reasoning')
                          or message.get('reasoning_content')),
            'finish_reason': choice.get('finish_reason'),
            'logprobs': d2l._chat_logprobs(choice.get('logprobs')),
            'usage': d2l._chat_usage(r.get('usage')), 'model': r.get('model'),
            'provider': r.get('provider'),
            'system_fingerprint': r.get('system_fingerprint'), 'extra': extra,
            't_first': None, 't_first_content': None, 'itl': None,
            'n_chunks': None}

def _retry_after(value):
    """Seconds in a Retry-After header (delta seconds or an HTTP date), or
    None when the header is absent or unreadable.

    Defined in :numref:`sec_lm_api`"""
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        import datetime  # not in the d2l library preamble
        import email.utils
        try:
            t = email.utils.parsedate_to_datetime(value)
            now = datetime.datetime.now(datetime.timezone.utc)
            return max(0.0, (t - now).total_seconds())
        except (TypeError, ValueError):
            return None

def _check_closed_loop(results, c):
    """Raise ValueError unless results can be one closed loop of c recorded
    in one run: no d2l.ReplayMiss or d2l.CostCapExceeded among them, not a
    mix of cached and live responses, never more than c in flight, and
    t_send minus the wall-clock `created` the same within 2 s (a monotonic
    clock of another boot or a suspended machine shifts it).

    Defined in :numref:`sec_lm_api`"""
    import calendar  # not in the d2l library preamble
    bad = [r for r in results
           if isinstance(r, (d2l.ReplayMiss, d2l.CostCapExceeded))]
    if bad:
        raise ValueError(f'{len(bad)} requests got no response: {bad[0]!r}')
    ok = [r for r in results if not isinstance(r, BaseException)]
    if len({r.from_cache for r in ok} - {None}) > 1:
        raise ValueError('the results mix cached and live responses; record '
                         'the level in one run, into a fresh cache')
    s = np.array([r.t_send for r in ok], dtype=float)
    f = s + np.array([r.t_total for r in ok], dtype=float)
    step = np.concatenate([np.ones(len(ok)), -np.ones(len(ok))])
    order = np.lexsort((step, np.concatenate([s, f])))  # ends before starts
    if len(ok) and np.cumsum(step[order]).max() > c:
        raise ValueError(f'more than c = {c} requests were in flight at once: '
                         f'not one closed loop')
    lag = [t - calendar.timegm(time.strptime(r.created, '%Y-%m-%dT%H:%M:%SZ'))
           for t, r in zip(s, ok) if r.created is not None]
    if lag and max(lag) - min(lag) > 2:
        raise ValueError('the send times come from more than one run '
                         '(monotonic and wall clocks disagree)')

class ChatAPI:
    """A measuring and replaying client for OpenAI-compatible chat
    completions. Every request is keyed by d2l.request_hash. Mode 'replay'
    answers from the JSONL cache and raises d2l.ReplayMiss on a miss,
    'record' answers from the cache and goes live on a miss, 'live' always
    goes live and appends; a replay returns the first recorded entry. Every
    live HTTP attempt, failed or not, is appended to <cache stem>.calls.jsonl.
    The SDK client (openai.AsyncOpenAI, max_retries=0) is created at the
    first live call, so constructing and replaying never imports openai or
    reads a key. api_key is a string, a function returning one (called at
    the first live call) or None (a placeholder for local servers); it never
    enters the hash, cache, calls log or manifest. With price set, a live
    call needs max_tokens (or max_completion_tokens); with cost_cap_usd it
    reserves its worst-case cost before sending and settles to the real
    cost after. http_client may also be an httpx transport (such as
    d2l.FakeChatServer().transport()). backoff_base, backoff_cap and rng
    (None: the random module) set the retry schedule.

    Defined in :numref:`sec_lm_api`"""
    RETRY_STATUS = (408, 429)  # and every 5xx

    def __init__(self, base_url, model, api_key=None, *, cache=None,
                 mode='replay', price=None, cost_cap_usd=None, timeout=120.0,
                 max_retries=6, default_params=None, http_client=None):
        import threading  # not in the d2l library preamble
        if mode not in ('replay', 'record', 'live'):
            raise ValueError(f"mode must be 'replay', 'record' or 'live', "
                             f'not {mode!r}')
        if cost_cap_usd is not None and price is None:
            raise ValueError('a cost cap needs a price')
        self.base_url, self.model, self.api_key = base_url, model, api_key
        self.mode, self.price, self.cost_cap_usd = mode, price, cost_cap_usd
        self.timeout, self.max_retries = timeout, max_retries
        self.default_params = d2l._chat_merge(default_params or {}, {})
        self.http_client, self.cache_path = http_client, cache
        self.calls_path = (None if cache is None else
                           os.path.splitext(cache)[0] + '.calls.jsonl')
        self.backoff_base, self.backoff_cap, self.rng = 1.0, 60.0, None
        self._lock, self._cache = threading.Lock(), {}
        self._sdk = self._loop = self._sdk_version = None
        self._sdks = {}  # event loop -> SDK client
        self._spent = self._reserved = 0.0
        self._stats, self._costs = collections.Counter(), collections.Counter()
        self._served, self._providers = (collections.Counter(),
                                         collections.Counter())
        self._trimmed = set()  # files whose torn last line was dealt with
        if cache is not None and os.path.exists(cache):
            import warnings  # not in the d2l library preamble
            with open(cache, 'rb') as f:
                lines = f.read().split(b'\n')
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line) if line.strip() else None
                except ValueError:
                    if i < len(lines) - 1:  # a whole line: a damaged cache
                        raise ValueError(f'{cache}: line {i + 1} is not JSON')
                    warnings.warn(f'{cache}: skipping its torn last line (a '
                                  f'run killed while writing it)')
                    continue
                if entry is not None:
                    self._cache.setdefault(entry['hash'], []).append(entry)

    @property
    def spent_usd(self):
        """USD settled by this client's live calls in this session."""
        return self._spent

    def entries(self, request_hash):
        """The cache entries (dicts) recorded for one request hash."""
        return self._cache.get(request_hash, [])

    def chat(self, messages, *, replica=0, stream=False, **params):
        """Synchronous d2l.ChatAPI.achat, run on the client's private event
        loop in a worker thread (so it also works inside Jupyter)."""
        return self._run(self.achat(messages, replica=replica, stream=stream,
                                    **params))

    def models(self):
        """The model ids the endpoint lists (GET /models; never replayed)."""
        if self.mode == 'replay':
            raise d2l.ReplayMiss(f'{self.model}: models() needs a live '
                                 f'endpoint, the client is in replay mode')
        async def ids():
            return [m.id async for m in self._client().models.list()]
        return self._run(ids())

    def manifest(self):
        """This session's counts (calls, cache hits, live calls, HTTP
        attempts, retries, 429s, failed attempts), the spent and reconciled
        cost (sum of api_cost against sum of usage.cost over the results
        returned), and the served model ids and providers of the results."""
        from urllib.parse import urlsplit  # not in the d2l library preamble
        keys = ('calls', 'cache_hits', 'live_calls', 'attempts', 'retries',
                'n_429', 'failed_attempts')
        return {'host': urlsplit(self.base_url).hostname,
                'model': self.model, 'mode': self.mode,
                'cache': (None if self.cache_path is None else
                          os.path.basename(self.cache_path)),
                'cache_entries': sum(map(len, self._cache.values())),
                **{k: self._stats[k] for k in keys},
                'spent_usd': self._spent, 'cost_cap_usd': self.cost_cap_usd,
                'price': None if self.price is None else self.price._asdict(),
                **{k: self._costs[k] for k in (
                    'n_api_cost', 'sum_api_cost', 'n_usage_cost',
                    'sum_usage_cost')},
                'served_models': dict(self._served),
                'providers': dict(self._providers), 'sdk': self._sdk_version}

    def _run(self, coro):
        import asyncio  # not in the d2l library preamble
        import threading
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                threading.Thread(target=self._loop.run_forever,
                                 daemon=True).start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def _client(self):
        """The SDK client of the running event loop, created on first use:
        one per open loop (the private loop of chat() and a notebook's), the
        clients of closed loops dropped."""
        import asyncio  # not in the d2l library preamble
        loop = asyncio.get_running_loop()
        with self._lock:
            for old in [l for l in self._sdks if l.is_closed()]:
                del self._sdks[old]
            if loop in self._sdks:
                return self._sdks[loop]
            import openai  # not in the d2l library preamble
            http = self.http_client
            if http is not None and hasattr(http, 'handle_async_request'):
                import httpx2  # the httpx flavour of openai >= 3
                http = httpx2.AsyncClient(transport=http)
            key = self.api_key() if callable(self.api_key) else self.api_key
            self._sdk = self._sdks[loop] = openai.AsyncOpenAI(
                base_url=self.base_url, api_key=key or 'none', max_retries=0,
                timeout=self.timeout, http_client=http)
            self._sdk_version = openai.__version__
            return self._sdk

    def _lookup(self, h, messages):
        """The first cached result of h unless the mode is 'live'; a miss
        raises d2l.ReplayMiss in replay mode and returns None otherwise."""
        if self.mode != 'live' and self._cache.get(h):
            self._stats['cache_hits'] += 1
            return self._result(self._cache[h][0], from_cache=True)
        if self.mode == 'replay':
            last = next((m.get('content') for m in reversed(messages)
                         if m.get('role') == 'user'), '')
            raise d2l.ReplayMiss(f'{self.model}: request {h[:12]} is not in '
                                 f'the cache (last user message '
                                 f'{str(last)[:80]!r})')
        return None

    def _reserve(self, messages, params):
        """Reserve the worst case: one prompt token per UTF-8 byte of the
        messages' JSON (a byte-level BPE token covers at least one byte, and
        the JSON's quoting outweighs a chat template's few special tokens per
        message) and max_tokens completion tokens for each of the n
        choices."""
        if self.price is None:
            self._stats['live_calls'] += 1
            return 0.0
        out = params.get('max_tokens', params.get('max_completion_tokens'))
        if out is None:
            raise ValueError('a priced live call needs max_tokens')
        n = len(json.dumps(messages, ensure_ascii=False).encode('utf-8'))
        worst = d2l.api_cost({'prompt_tokens': n, 'completion_tokens': out * (
            params.get('n') or 1)}, self.price)
        with self._lock:
            if (self.cost_cap_usd is not None and self._spent + self._reserved
                    + worst > self.cost_cap_usd):
                raise d2l.CostCapExceeded(
                    f'{self.model}: spent {self._spent:.6f} + reserved '
                    f'{self._reserved:.6f} + this call {worst:.6f} USD would '
                    f'cross the cap of {self.cost_cap_usd} USD')
            self._reserved += worst
        self._stats['live_calls'] += 1
        return worst

    def _settle(self, reserved, cost):
        with self._lock:
            self._reserved -= reserved
            self._spent += cost

    def _cost(self, usage, reserved=None):
        """Real cost of a response: the larger of usage.cost and api_cost
        from the price; the reservation when neither is known."""
        costs = [usage.get('cost')]
        if self.price is not None:
            try:
                costs.append(d2l.api_cost(usage, self.price))
            except ValueError:
                pass
        costs = [c for c in costs if c is not None]
        return max(costs) if costs else (reserved or 0.0)

    def _retry_info(self, e):
        """(retry?, HTTP status, Retry-After seconds) of a failed attempt:
        connection errors, timeouts, RETRY_STATUS and every 5xx are retried,
        and so is an error event inside a stream (the SDK raises a bare
        APIError; its numeric code, if any, is treated as the status)."""
        def retry(status):
            return status in self.RETRY_STATUS or 500 <= status < 600
        status = getattr(e, 'status_code', None)
        if status is not None:
            headers = getattr(getattr(e, 'response', None), 'headers', {})
            return (retry(status), status,
                    d2l._retry_after(headers.get('retry-after')))
        openai = sys.modules.get('openai')
        if openai is not None and isinstance(e, openai.APIConnectionError):
            return True, None, None
        if openai is not None and type(e) is openai.APIError:  # mid-stream
            code = str(getattr(e, 'code', None) or '')
            if code.isdigit():
                return retry(int(code)), int(code), None
            return True, None, None
        transport = [getattr(sys.modules.get(m), 'TransportError', None)
                     for m in ('httpx2', 'httpx')]
        return any(t is not None and isinstance(e, t)
                   for t in transport), None, None

    async def _send(self, messages, params, stream, h, replica, attempt):
        """One HTTP attempt, logged to the calls log whatever its outcome;
        returns the parsed response with its chunk times."""
        kwargs = {k: v for k, v in params.items()
                  if k not in ('stream', 'stream_options')}
        log = {'created': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'hash': h, 'model': self.model, 'replica': replica,
               'attempt': attempt, 'stream': stream}
        self._stats['attempts'] += 1
        self._stats['retries'] += attempt > 0
        t0 = time.monotonic()
        try:
            create = self._client().chat.completions.create
            if stream:
                options = d2l._chat_merge({'include_usage': True},
                                          params.get('stream_options') or {})
                out = await self._read_stream(await create(
                    model=self.model, messages=messages, stream=True,
                    stream_options=options, **kwargs))
            else:
                response = await create(model=self.model, messages=messages,
                                        **kwargs)
                out = d2l._chat_parse(response.model_dump())
                out['t_end'] = time.monotonic()
        except Exception as e:
            _, status, retry_after = self._retry_info(e)
            self._stats['failed_attempts'] += 1
            self._stats['n_429'] += status == 429
            self._append(self.calls_path, {
                **log, 'status': status, 'retry_after': retry_after,
                'error': f'{type(e).__name__}: {e}'[:200],
                'latency': time.monotonic() - t0})
            raise
        self._append(self.calls_path, {
            **log, 'status': 200, 'latency': out['t_end'] - t0,
            'finish_reason': out['finish_reason'], 'usage': out['usage'],
            'cost': self._cost(out['usage']), 'served_model': out['model'],
            'provider': out['provider']})
        return out

    async def _read_stream(self, chunks):
        """Concatenate a streamed response; t_first is the first chunk with
        reasoning or content, t_first_content the first with content, and
        itl the gaps between such chunks."""
        out = dict.fromkeys(('finish_reason', 'usage', 'model', 'provider',
                             'system_fingerprint', 't_first',
                             't_first_content'))
        text, reasoning, logprobs, times = [], [], [], []
        now = time.monotonic()
        async for chunk in chunks:
            c, now = chunk.model_dump(), time.monotonic()
            for k in ('model', 'provider', 'system_fingerprint', 'usage'):
                out[k] = c.get(k) or out[k]
            for choice in c.get('choices') or []:
                if choice.get('index', 0) != 0:
                    continue
                delta = choice.get('delta') or {}
                r = delta.get('reasoning') or delta.get('reasoning_content')
                if r or delta.get('content'):
                    times.append(now)
                    if out['t_first'] is None:
                        out['t_first'] = now
                if delta.get('content'):
                    text.append(delta['content'])
                    if out['t_first_content'] is None:
                        out['t_first_content'] = now
                if r:
                    reasoning.append(r)
                logprobs += d2l._chat_logprobs(choice.get('logprobs')) or []
                out['finish_reason'] = (choice.get('finish_reason')
                                        or out['finish_reason'])
        out.update(text=''.join(text), reasoning=''.join(reasoning) or None,
                   logprobs=logprobs or None, extra={}, t_end=now,
                   usage=d2l._chat_usage(out['usage']), n_chunks=len(times),
                   itl=[b - a for a, b in zip(times, times[1:])])
        return out

    def _record(self, h, messages, params, replica, out, t_send, created,
                n_retries, n_429, reserved):
        """Settle the cost of a live response, append it to the cache and
        return it as a d2l.ChatResult."""
        from urllib.parse import urlsplit  # not in the d2l library preamble
        self._settle(reserved, self._cost(out['usage'], reserved))
        fields = ('text', 'finish_reason', 'usage', 'logprobs', 'model',
                  'provider', 'system_fingerprint', 'reasoning', 't_first',
                  't_first_content', 'itl')
        result = {**{k: out[k] for k in fields}, 'request_hash': h,
                  't_send': t_send, 't_total': out['t_end'] - t_send,
                  'n_retries': n_retries, 'n_429': n_429, 'created': created}
        transport = ('stream', 'stream_options', 'timeout', 'extra_headers')
        request = {'host': urlsplit(self.base_url).hostname,
                   'model': self.model, 'messages': messages,
                   'replica': replica, 'params': {
                       k: v for k, v in params.items() if k not in transport}}
        entry = json.loads(json.dumps(  # what a reload of the file gives
            {'hash': h, 'request': request, 'result': result,
             'extra': out['extra'], 'n_chunks': out['n_chunks'],
             'sdk': self._sdk_version}, ensure_ascii=False))
        self._cache.setdefault(h, []).append(entry)
        self._append(self.cache_path, entry)
        return self._result(entry, from_cache=False)

    def _result(self, entry, from_cache):
        r = dict(entry['result'], from_cache=from_cache)
        if r['logprobs'] is not None:
            r['logprobs'] = [(t, lp, [tuple(a) for a in top])
                             for t, lp, top in r['logprobs']]
        result = d2l.ChatResult(**r)
        self._served[result.model] += 1
        self._providers[result.provider] += 1
        if result.usage.get('cost') is not None:
            self._costs['n_usage_cost'] += 1
            self._costs['sum_usage_cost'] += result.usage['cost']
        if self.price is not None:
            try:
                self._costs['sum_api_cost'] += d2l.api_cost(result.usage,
                                                            self.price)
                self._costs['n_api_cost'] += 1
            except ValueError:
                pass
        return result

    def _append(self, path, obj):
        """Append one JSON line (the key scrubbed, should it ever appear)."""
        if path is None:
            return
        line = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        key = getattr(self._sdk, 'api_key', None)
        if isinstance(key, str) and len(key) >= 8:
            line = line.replace(key, '[api key]')
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            if path not in self._trimmed:
                d2l._jsonl_trim_torn(path)
                self._trimmed.add(path)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

    async def achat(self, messages, *, replica=0, stream=False, **params):
        """One chat completion as a d2l.ChatResult. The cache answers first when
        the mode allows; a live call reserves its worst-case cost, retries
        connection errors, timeouts, 408, 429 and 5xx responses and error
        events inside a stream with full-jitter backoff (Retry-After as a
        floor), and raises at once on any other status.

        Defined in :numref:`sec_lm_api`"""
        import asyncio  # not in the d2l library preamble
        params = d2l._chat_merge(self.default_params, params)
        h = d2l.request_hash(self.base_url, self.model, messages, params, replica)
        self._stats['calls'] += 1
        hit = self._lookup(h, messages)  # raises ReplayMiss in replay mode
        if hit is not None:
            return hit
        reserved = self._reserve(messages, params)  # raises CostCapExceeded
        t_send, n_retries, n_429 = time.monotonic(), 0, 0
        created = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    out = await self._send(messages, params, stream, h, replica,
                                           attempt)
                    break
                except Exception as e:
                    retry, status, retry_after = self._retry_info(e)
                    n_429 += status == 429
                    if not retry or attempt == self.max_retries:
                        e.n_retries, e.n_429 = n_retries, n_429
                        raise
                    n_retries += 1
                    await asyncio.sleep(d2l.backoff_delay(
                        attempt, self.backoff_base, self.backoff_cap, self.rng,
                        retry_after))
        except BaseException:
            self._settle(reserved, 0.0)
            raise
        return self._record(h, messages, params, replica, out, t_send, created,
                            n_retries, n_429, reserved)

def _jsonl_trim_torn(path):
    """End a JSONL file with a newline before appending to it: a torn last
    line (no newline, as a run killed mid-write leaves it) is cut off unless
    it is complete JSON, which only gets its newline.

    Defined in :numref:`sec_lm_api`"""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, 'rb+') as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b'\n':
            return
        f.seek(0)
        data = f.read()
        cut = data.rfind(b'\n') + 1
        try:
            json.loads(data[cut:])
            f.write(b'\n')
        except ValueError:
            f.truncate(cut)

def _api_window(override, day, hhmm):
    """Whether a pricing.overrides row covers a UTC weekday and HHMM time;
    utc_end 0 means midnight, and a window may wrap past it.

    Defined in :numref:`sec_lm_api`"""
    if override.get('utc_days') and day not in override['utc_days']:
        return False
    start, end = override.get('utc_start'), override.get('utc_end')
    if start is None and end is None:
        return True
    start, end = start or 0, end or 2400
    return (start <= hhmm < end) if start < end else (
        hhmm >= start or hhmm < end)

def price_from_listing(listing, model_id, provider, *, when=None):
    """The d2l.PricePoint (USD per million tokens) of one provider's endpoint
    of model_id in an OpenRouter listing snapshot: listing['endpoints']
    [model_id] holds the endpoint rows of /api/v1/models/<id>/endpoints (or
    that whole response). provider matches a row's tag exactly, else its
    provider_name or the slug before the tag's '/'. Listed prices are USD
    per token with the row's `discount` already applied (Novita's 0.165 per
    million at discount 0.45 is 0.3 x 0.55), so it is not applied again.
    Rows with time-window pricing.overrides need when, an aware datetime
    whose UTC weekday and time select the window.

    Defined in :numref:`sec_lm_api`"""
    import datetime  # not in the d2l library preamble
    import decimal
    rows = listing['endpoints'][model_id]
    if isinstance(rows, dict):
        rows = rows['data']['endpoints']
    p = provider.lower()
    match = [r for r in rows if str(r.get('tag', '')).lower() == p] or [
        r for r in rows if p in (str(r.get('provider_name', '')).lower(),
                                 str(r.get('tag', '')).lower().split('/')[0])]
    if len(match) != 1:
        raise ValueError(f'{len(match)} endpoints of {model_id} match '
                         f'{provider!r}: {[r.get("tag") for r in match]}')
    pricing = match[0]['pricing']
    if pricing.get('overrides'):
        if when is None or when.tzinfo is None:
            raise ValueError(f'{provider} prices {model_id} by time window: '
                             f'pass when, an aware datetime')
        t = when.astimezone(datetime.timezone.utc)
        day = ('monday', 'tuesday', 'wednesday', 'thursday', 'friday',
               'saturday', 'sunday')[t.weekday()]
        window = next((o for o in pricing['overrides'] if d2l._api_window(
            o, day, 100 * t.hour + t.minute)), {})
        pricing = {**pricing, **window}
    def per_million(key):
        if pricing.get(key) is None:
            return None
        value = decimal.Decimal(str(pricing[key])) * 1_000_000
        if value < 0:
            raise ValueError(f'{provider}: no fixed {key} price ({value})')
        return float(value)
    return d2l.PricePoint(per_million('prompt'), per_million('completion'),
                          per_million('input_cache_read'))

def _openrouter_key():
    """The OpenRouter key, read from the environment only (never a file).

    Defined in :numref:`sec_lm_api`"""
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('a live OpenRouter call needs OPENROUTER_API_KEY '
                           'in the environment')
    return key

def openrouter_api(model, provider, *, listing, cache=None, mode='replay',
                   cost_cap_usd=None, reasoning=None, when=None, **kwargs):
    """A d2l.ChatAPI for one model on OpenRouter with one provider pinned
    (no fallbacks, require_parameters on), priced from the listing snapshot
    (None: unpriced). reasoning (e.g. {'enabled': False}) is sent only when
    given. The key is read from OPENROUTER_API_KEY at the first live call;
    further keyword arguments go to d2l.ChatAPI.

    Defined in :numref:`sec_lm_api`"""
    body = {'provider': {'only': [provider], 'allow_fallbacks': False,
                         'require_parameters': True}}
    if reasoning is not None:
        body['reasoning'] = reasoning
    defaults = d2l._chat_merge({'extra_body': body},
                               kwargs.pop('default_params', None) or {})
    kwargs.setdefault('api_key', d2l._openrouter_key)
    price = (None if listing is None else
             d2l.price_from_listing(listing, model, provider, when=when))
    return d2l.ChatAPI('https://openrouter.ai/api/v1', model, cache=cache,
                       mode=mode, price=price, cost_cap_usd=cost_cap_usd,
                       default_params=defaults, **kwargs)

class LocalChatServer:
    """An OpenAI-compatible server (such as `transformers serve`) run as a
    subprocess in its own process group. '{port}' in cmd becomes port (a
    free one when None); start() polls http://host:port/v1/models until it
    answers 200 and returns the base URL, or kills the server and raises
    with the last 20 log lines; stop() terminates the whole group, deletes
    the temporary log and returns the exit code. Also a context manager.

    Defined in :numref:`sec_lm_api`"""
    def __init__(self, cmd, *, port=None, host='127.0.0.1',
                 ready_timeout=900.0):
        self.cmd, self.port, self.host = list(cmd), port, host
        self.ready_timeout = ready_timeout
        self.proc = self.base_url = self.log_path = self.load_time = None

    def start(self):
        import atexit  # not in the d2l library preamble
        import socket
        import subprocess
        import tempfile
        import urllib.request
        if self.proc is not None:
            return self.base_url
        if self.port is None:
            with socket.socket() as s:
                s.bind((self.host, 0))
                self.port = s.getsockname()[1]
        cmd = [str(a).replace('{port}', str(self.port)) for a in self.cmd]
        fd, self.log_path = tempfile.mkstemp(prefix='d2l-server-',
                                             suffix='.log')
        t0 = time.monotonic()
        with os.fdopen(fd, 'wb') as log:
            self.proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
        atexit.register(self.stop)
        url = f'http://{self.host}:{self.port}/v1'
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            if self.proc.poll() is not None:  # its children may live on
                code, tail = self.proc.returncode, self._tail()
                self.stop()
                raise RuntimeError(f'server exited with code {code} before '
                                   f'it was ready:\n{tail}')
            left = self.ready_timeout - (time.monotonic() - t0)
            if left <= 0:
                tail = self._tail()
                self.stop()
                raise TimeoutError(f'server not ready after '
                                   f'{self.ready_timeout} s:\n{tail}')
            try:
                with opener.open(url + '/models',
                                 timeout=max(0.5, min(5.0, left))) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(min(0.5, max(left, 0.0)))
        self.load_time, self.base_url = time.monotonic() - t0, url
        return url

    def stop(self):
        import atexit  # not in the d2l library preamble
        import signal
        import subprocess
        if self.proc is None:
            return None
        proc, self.proc = self.proc, None
        atexit.unregister(self.stop)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            code = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
        try:  # children that outlived the leader
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if self.log_path is not None and os.path.exists(self.log_path):
            os.remove(self.log_path)
        return code

    def _tail(self, n=20):
        with open(self.log_path, encoding='utf-8', errors='replace') as f:
            return ''.join(f.readlines()[-n:])

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

class FakeChatServer:
    """An in-process chat-completions server for tests. Each request takes
    service_time seconds in one of capacity slots, so a closed loop has
    W_0 = service_time and X_max = capacity / service_time; a busy slot
    starts its next request when the previous one ended (not when the event
    loop gets round to it), so the capacity is exact. script lists
    statuses returned in order before normal behaviour (429s carry
    Retry-After: 0); fail_rate answers 503 at random; rate_limit answers 429
    beyond that many requests per second. A request with thinking switched
    on gets reasoning first ('reasoning', or 'reasoning_content' when it
    carries chat_template_kwargs, as a local server); max_tokens, stop, n,
    logprobs and stream_options.include_usage are honoured, and requests
    with a provider pin get a provider and usage.cost. transport() plugs it
    into the SDK's http_client; n_requests, statuses, bodies and
    max_in_flight record what it saw.

    Defined in :numref:`sec_lm_api`"""
    PRICE = (0.1, 0.4)  # USD per million prompt and completion tokens

    def __init__(self, service_time, capacity, *, fail_rate=0.0,
                 rate_limit=None, script=None, seed=0):
        self.service_time, self.capacity = service_time, capacity
        self.fail_rate, self.rate_limit = fail_rate, rate_limit
        self.script, self.rng = list(script or []), random.Random(seed)
        self.n_requests = self.in_flight = self.max_in_flight = 0
        self.statuses, self.bodies = [], []
        self._admitted = collections.deque()
        self._sem = self._sem_loop = self._free = None

    def transport(self):
        """An httpx.AsyncBaseTransport of the flavour the SDK uses."""
        try:
            import httpx2 as httpx  # openai >= 3; not in the d2l preamble
        except ImportError:
            import httpx
        server = self
        class Transport(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                return await server._handle(request, httpx)
        return Transport()

    async def _acquire(self, arrival):
        """Wait for a slot; the service starts at arrival or when the slot's
        previous request ended, whichever is later."""
        import asyncio  # not in the d2l library preamble
        import heapq
        loop = asyncio.get_running_loop()
        if self._sem_loop is not loop:
            self._sem, self._sem_loop = asyncio.Semaphore(self.capacity), loop
            self._free = [0.0] * self.capacity
        await self._sem.acquire()
        return max(arrival, heapq.heappop(self._free))

    def _release(self, end):
        import heapq  # not in the d2l library preamble
        heapq.heappush(self._free, end)
        self._sem.release()

    def _status(self):
        if self.script:
            return self.script.pop(0), True
        if self.rate_limit is not None:
            now = time.monotonic()
            while self._admitted and now - self._admitted[0] >= 1.0:
                self._admitted.popleft()
            if len(self._admitted) >= self.rate_limit:
                return 429, False
            self._admitted.append(now)
        if self.fail_rate and self.rng.random() < self.fail_rate:
            return 503, False
        return 200, False

    def _reply(self, body):
        """Tokens ((kind, text) pairs), finish reason and usage of a reply."""
        messages = body.get('messages') or [{}]
        last = str(messages[-1].get('content', ''))
        words = (f'Fake reply to {len(last)} characters: the answer is '
                 f'{len(last) % 10}.').split(' ')
        content, finish = [w if i == 0 else ' ' + w
                           for i, w in enumerate(words)], 'stop'
        stops = body.get('stop') or []
        for i in range(len(content)):
            text = ''.join(content[:i + 1])
            cut = min((text.find(s) for s in ([stops] if isinstance(
                stops, str) else stops) if s in text), default=None)
            if cut is not None:
                prefix = text[:cut]
                content = [prefix] if prefix else []
                break
        r = body.get('reasoning')
        think = (r.get('enabled', True) if isinstance(r, dict) else bool(
            (body.get('chat_template_kwargs') or {}).get('enable_thinking')))
        tokens = ([('r', w) for w in ('Let', ' me', ' think.')] if think
                  else []) + [('c', w) for w in content]
        limit = body.get('max_tokens', body.get('max_completion_tokens'))
        if limit is not None and len(tokens) > limit:
            tokens, finish = tokens[:limit], 'length'
        n_prompt = sum(len(str(m.get('content', '')).split())
                       for m in messages)
        usage = {'prompt_tokens': n_prompt, 'completion_tokens': len(tokens),
                 'total_tokens': n_prompt + len(tokens),
                 'prompt_tokens_details': {'cached_tokens': 0},
                 'completion_tokens_details': {'reasoning_tokens': sum(
                     k == 'r' for k, _ in tokens)}}
        if 'provider' in body:
            usage['cost'] = (n_prompt * self.PRICE[0]
                             + len(tokens) * self.PRICE[1]) / 1e6
        return tokens, finish, usage

    def _logprobs(self, body, words):
        if not body.get('logprobs'):
            return None
        k = body.get('top_logprobs') or 0
        return {'content': [
            {'token': w, 'logprob': -0.1 * (i + 1), 'top_logprobs': [
                {'token': w if j == 0 else f'{w}#{j}',
                 'logprob': -0.1 * (i + 1) - j} for j in range(k)]}
            for i, w in enumerate(words)]}

    async def _handle(self, request, httpx):
        import asyncio  # not in the d2l library preamble
        arrival = time.monotonic()
        self.n_requests += 1
        if request.method == 'GET' and request.url.path.endswith('/models'):
            self.statuses.append(200)
            return httpx.Response(200, json={'object': 'list', 'data': [
                {'id': 'fake-model', 'object': 'model', 'created': 0,
                 'owned_by': 'd2l'}]})
        body = json.loads((await request.aread()) or b'{}')
        self.bodies.append(body)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        status, scripted = self._status()
        self.statuses.append(status)
        if status != 200:
            self.in_flight -= 1
            scripted_429 = scripted and status == 429
            headers = {'retry-after': '0'} if scripted_429 else {}
            return httpx.Response(status, headers=headers, json={'error': {
                'message': f'fake status {status}', 'code': status}})
        tokens, finish, usage = self._reply(body)
        head = {'id': 'fake-1', 'created': int(time.time()),
                'model': body.get('model'), 'system_fingerprint': 'fp_fake'}
        if 'provider' in body:
            head['provider'] = (body['provider'].get('only') or ['Fake'])[0]
        field = ('reasoning_content' if 'chat_template_kwargs' in body
                 else 'reasoning')
        if body.get('stream'):
            return httpx.Response(
                200, headers={'content-type': 'text/event-stream'},
                content=self._stream(arrival, body, head, field, tokens,
                                     finish, usage))
        try:
            end = await self._acquire(arrival) + self.service_time
            try:
                await asyncio.sleep(max(0.0, end - time.monotonic()))
            finally:
                self._release(end)
        finally:
            self.in_flight -= 1
        reasoning = ''.join(w for k, w in tokens if k == 'r')
        words = [w for k, w in tokens if k == 'c']
        message = {'role': 'assistant', 'content': ''.join(words)}
        if reasoning:
            message[field] = reasoning
        choice = {'message': message, 'finish_reason': finish,
                  'logprobs': self._logprobs(body, words)}
        return httpx.Response(200, json={
            **head, 'object': 'chat.completion', 'usage': usage,
            'choices': [{'index': i, **choice}
                        for i in range(body.get('n') or 1)]})

    async def _stream(self, arrival, body, head, field, tokens, finish,
                      usage):
        """Server-sent events: a role chunk, the first token half the
        service time after the start, the rest spread over the other half."""
        import asyncio  # not in the d2l library preamble
        def event(choices, **more):
            return 'data: {}\n\n'.format(json.dumps({
                **head, 'object': 'chat.completion.chunk', 'choices': choices,
                **more})).encode()
        async def until(t):
            await asyncio.sleep(max(0.0, t - time.monotonic()))
        try:
            start = await self._acquire(arrival)
            gap = self.service_time / 2 / max(1, len(tokens))
            try:
                yield event([{'index': 0, 'finish_reason': None,
                              'delta': {'role': 'assistant', 'content': ''}}])
                await until(start + self.service_time / 2)
                for i, (kind, w) in enumerate(tokens):
                    lp = self._logprobs(body, [w]) if kind == 'c' else None
                    key = 'content' if kind == 'c' else field
                    yield event([{'index': 0, 'finish_reason': None,
                                  'logprobs': lp, 'delta': {key: w}}])
                    await until(start + self.service_time / 2 + (i + 1) * gap)
                yield event([{'index': 0, 'delta': {},
                              'finish_reason': finish}])
                if (body.get('stream_options') or {}).get('include_usage'):
                    yield event([], usage=usage)
                yield b'data: [DONE]\n\n'
            finally:
                self._release(start + self.service_time)
        finally:
            self.in_flight -= 1


ones_like = torch.ones_like
ones = torch.ones
zeros_like = torch.zeros_like
zeros = torch.zeros
tensor = torch.tensor
arange = torch.arange
sin = torch.sin
sinh = torch.sinh
cos = torch.cos
cosh = torch.cosh
tanh = torch.tanh
linspace = torch.linspace
exp = torch.exp
log = torch.log
normal = torch.normal
rand = torch.rand
randn = torch.randn
matmul = torch.matmul
int32 = torch.int32
int64 = torch.int64
float32 = torch.float32
concat = torch.cat
stack = torch.stack
abs = torch.abs
eye = torch.eye
sigmoid = torch.sigmoid
batch_matmul = torch.bmm
numpy = lambda x, *args, **kwargs: x.detach().cpu().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)
expand_dims = lambda x, *args, **kwargs: x.unsqueeze(*args, **kwargs)
swapaxes = lambda x, *args, **kwargs: x.swapaxes(*args, **kwargs)
repeat = lambda x, *args, **kwargs: x.repeat(*args, **kwargs)
nn_Module = nn.Module
meshgrid = lambda *args, **kwargs: torch.meshgrid(*args, **{'indexing': 'ij', **kwargs})
