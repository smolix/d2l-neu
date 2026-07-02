```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# The Image Classification Dataset
:label:`sec_fashion_mnist`



One widely used benchmark for image classification is [MNIST](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998`, a dataset of 70,000 handwritten-digit images ($28 \times 28$ pixels, 10 classes). MNIST shaped a generation of machine learning research, but today even simple models exceed 95% accuracy (and a linear classifier already tops 90%), so differences between strong and weak models are hard to see. To compare models meaningfully we need a dataset where a linear baseline is clearly outpaced by a richer one.

We therefore use **Fashion-MNIST** :cite:`Xiao.Rasul.Vollgraf.2017`, a drop-in replacement released in 2017. It has exactly the same structure (60,000 training and 10,000 test images of $28 \times 28$ grayscale pixels, in 10 classes) but the classes are clothing categories (t-shirt, trouser, pullover, and so on) that are harder to tell apart, which makes accuracy differences between models clearly visible. For large-scale experiments the standard benchmark is ImageNet :cite:`Deng.Dong.Socher.ea.2009` (1.2 million images, 1000 classes), but it is too large to keep our examples interactive; Fashion-MNIST teaches the same lessons at a fraction of the compute cost.

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab mxnet
%matplotlib inline
import time
from d2l import mxnet as d2l
from mxnet import gluon, npx
from mxnet.gluon.data.vision import transforms
npx.set_np()

d2l.use_svg_display()
```

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab pytorch
%matplotlib inline
import time
from d2l import torch as d2l
import torch
import torchvision
from torchvision import transforms

d2l.use_svg_display()
```

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab tensorflow
%matplotlib inline
import time
from d2l import tensorflow as d2l
import tensorflow as tf

d2l.use_svg_display()
```

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
import numpy as np
import time
import tensorflow as tf
import tensorflow_datasets as tfds

d2l.use_svg_display()
```

## Loading the Dataset

Since the Fashion-MNIST dataset is so useful, all major frameworks provide preprocessed versions of it. We can download and read it into memory using built-in framework utilities.

```{.python .input #image-classification-dataset-loading-the-dataset-1}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.vision.FashionMNIST(
            train=True).transform_first(trans)
        self.val = gluon.data.vision.FashionMNIST(
            train=False).transform_first(trans)
```

```{.python .input #image-classification-dataset-loading-the-dataset-1}
%%tab pytorch
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.FashionMNIST(
            root=self.root, train=False, transform=trans, download=True)
```

```{.python .input #image-classification-dataset-loading-the-dataset-1}
%%tab tensorflow, jax
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train, self.val = tf.keras.datasets.fashion_mnist.load_data()
```

Fashion-MNIST consists of images from 10 categories, each represented
by 6000 images in the training dataset and by 1000 in the test dataset.
A *test dataset* is used for evaluating model performance (it must not be used for training).
Consequently the training set and the test set
contain 60,000 and 10,000 images, respectively.

```{.python .input #image-classification-dataset-loading-the-dataset-2}
%%tab mxnet, pytorch
data = FashionMNIST(resize=(32, 32))
len(data.train), len(data.val)
```

```{.python .input #image-classification-dataset-loading-the-dataset-2}
%%tab tensorflow, jax
data = FashionMNIST(resize=(32, 32))
len(data.train[0]), len(data.val[0])
```

We instantiated the dataset with `resize=(32, 32)`, so each image is delivered as a single-channel tensor of spatial size $32 \times 32$. There is one subtlety worth pinning down now: where the channel axis lives. PyTorch and MXNet use the *channel-first* convention $c \times h \times w$ ($c$ color channels, then height and width); TensorFlow and JAX use *channel-last* $h \times w \times c$. The `get_dataloader` method below produces the right layout for each framework, so the rest of this chapter never has to think about it; we confirm the per-image shape once the loader is in place, below.

A single grayscale image, so $c = 1$. Most modern photographs have $c = 3$ channels (red, green, blue); hyperspectral sensors such as HyMap record over 100.



The categories of Fashion-MNIST have human-understandable names. 
The following convenience method converts between numeric labels and their names.

```{.python .input #image-classification-dataset-loading-the-dataset-4}
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Reading a Minibatch

To make our life easier when reading from the training and test sets,
we use the built-in data iterator rather than creating one from scratch.
Recall that at each iteration, a data iterator
reads a minibatch of data with size `batch_size`.
We also randomly shuffle the examples for the training data iterator.

```{.python .input #image-classification-dataset-reading-a-minibatch-1}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 num_workers=self.num_workers)
```

```{.python .input #image-classification-dataset-reading-a-minibatch-1}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train,
                                       num_workers=self.num_workers)
```

```{.python .input #image-classification-dataset-reading-a-minibatch-1}
%%tab tensorflow
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    process = lambda X, y: (tf.expand_dims(X, axis=3) / 255,
                            tf.cast(y, dtype='int32'))
    resize_fn = lambda X, y: (tf.image.resize_with_pad(X, *self.resize), y)
    shuffle_buf = len(data[0]) if train else 1
    # `drop_remainder=train` keeps every training minibatch the same
    # shape so Keras `model.fit` / a `@tf.function`'d train-step compile
    # once and stop retracing for the smaller last batch (a major
    # speedup for HPO loops where a fresh model is fit per trial).
    return tf.data.Dataset.from_tensor_slices(process(*data)).shuffle(
        shuffle_buf).batch(self.batch_size,
                           drop_remainder=train).map(resize_fn)
```

```{.python .input #image-classification-dataset-reading-a-minibatch-1}
%%tab jax
@d2l.add_to_class(FashionMNIST)  #@save
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
```

Now that the loader is defined, let us read one image and confirm where the channel axis lands.

```{.python .input #image-classification-dataset-loading-the-dataset-3}
%%tab mxnet, pytorch
X, y = next(iter(data.train_dataloader()))
X[0].shape  # channel-first: (channels, height, width)
```

```{.python .input #image-classification-dataset-loading-the-dataset-3}
%%tab tensorflow, jax
X, y = next(iter(data.train_dataloader()))
X[0].shape  # channel-last: (height, width, channels)
```

To see how this works, let's load a minibatch of images by invoking the `train_dataloader` method. It contains 64 images.

```{.python .input #image-classification-dataset-reading-a-minibatch-2}
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

Let us time one full pass through the training set. The exact number (a few seconds on a CPU-only machine for PyTorch, under a second once the TF/JAX pipeline is compiled) matters less than the comparison: a single forward and backward pass over a minibatch typically takes 10 to 100 times longer than the corresponding I/O, so the loader is not the bottleneck. If loading *were* slower than training, you would overlap I/O with compute via prefetching (`prefetch_factor` in PyTorch, `.prefetch()` in `tf.data`) or raise `num_workers`.

```{.python .input #image-classification-dataset-reading-a-minibatch-3}
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Visualization

We will often be using the Fashion-MNIST dataset. A convenience function `show_images` lays out a list of images in a grid with optional per-image titles. The `d2l` library provides it; here we show only its interface. The full implementation (matplotlib grid layout) lives in the library source, so the cell below is a stub: knowing how to call it is what matters for our purposes.

```{.python .input #image-classification-dataset-visualization-1}
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """Plot a list of images."""
    # Full implementation lives in the d2l library (d2l/torch.py et al.).
    # This stub declares the interface; the rendered notebook uses the library version.
    raise NotImplementedError
```

Let's put it to good use. In general, it is a good idea to visualize and inspect data that you are training on. 
Humans are very good at spotting oddities and because of that, visualization serves as an additional safeguard against mistakes and errors in the design of experiments. Here are the images and their corresponding labels (in text)
for the first few examples in the training dataset.

```{.python .input #image-classification-dataset-visualization-2}
%%tab pytorch
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=None):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

```{.python .input #image-classification-dataset-visualization-2}
%%tab tensorflow
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=None):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

```{.python .input #image-classification-dataset-visualization-2}
%%tab jax
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=None):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(jnp.squeeze(X), nrows, ncols, titles=labels)

batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

```{.python .input #image-classification-dataset-visualization-2}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=None):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)
batch = next(iter(data.val_dataloader()))
data.visualize(batch)
```

We are now ready to work with the Fashion-MNIST dataset in the sections that follow.

## Summary

We now have a slightly more realistic dataset to use for classification. Fashion-MNIST is an apparel classification dataset consisting of images representing 10 categories. We will use this dataset in subsequent sections and chapters to evaluate various network designs, from a simple linear model to advanced residual networks. As we commonly do with images, we read them as a tensor of shape (batch size, number of channels, height, width). For now, we only have one channel as the images are grayscale (the visualization above uses a false color palette for improved visibility). A well-implemented data iterator keeps this loading off the critical path, so that training speed is set by the model rather than by I/O.


## Exercises

1. Time one full training epoch at `batch_size` of 1, 16, 64, 256, and 1024. Plot throughput (images per second) against `batch_size`. Why does throughput rise with batch size up to a point and then plateau?
1. Set `num_workers=0` (single-threaded loading) and compare against the default multi-worker setting. Under what conditions does increasing `num_workers` stop helping?
1. PyTorch stores tensors in channel-first order $(c, h, w)$, while TensorFlow and JAX use channel-last $(h, w, c)$. Read the `get_dataloader` implementations for all four frameworks. Which step introduces the channel dimension, and where does the layout differ?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/48)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/49)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/224)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17980)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.2]{.kicker}

The Image Classification Dataset<br>**Fashion-MNIST**, the workhorse we will classify for the rest of this chapter.

@!image-classification-dataset-visualization-2
:::
:::

::: {.slide title="Why a new benchmark?"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **MNIST** (handwritten digits) is solved: a *linear* model already tops 95%, so models are hard to tell apart.
- We want data where a weak model is **clearly outpaced** by a richer one.
- **Fashion-MNIST**: a drop-in replacement with the same shape and API but harder clothing classes ($28\times28$ grayscale, 10 classes, 60 k / 10 k).
:::

::: {.col .fig .big}
@!image-classification-dataset-visualization-2
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Loading the Data]{.dtitle}

[a reusable DataModule per framework]{.dsub}
:::
:::

::: {.slide title="Wrap it once, reuse everywhere"}
[Loading]{.kicker}

A `DataModule` owns this framework's download, transform, and `train`/`val` splits, so every model we build later just asks for batches:

@image-classification-dataset-loading-the-dataset-1
:::

::: {.slide title="60 000 train, 10 000 test"}
[Loading]{.kicker}

Instantiate it, resizing to $32\times32$ to match the ConvNet inputs in later chapters:

@image-classification-dataset-loading-the-dataset-2

::: {.d2l-note}
Ten classes $\times$ 6 000 train images each $= 60\,000$; 1 000 each in test.
:::
:::

::: {.slide title="One image: channel-first" only="pytorch,mxnet"}
[Loading · layout]{.kicker}

PyTorch and MXNet store images **channel-first**, $c \times h \times w$, with the color axis before height and width:

@-image-classification-dataset-loading-the-dataset-3

::: {.d2l-note .rule}
Shape is `(1, 32, 32)`: one grayscale channel, then $32\times32$ pixels.
:::
:::

::: {.slide title="One image: channel-last" only="tensorflow,jax"}
[Loading · layout]{.kicker}

TensorFlow and JAX store images **channel-last**, $h \times w \times c$, with the color axis at the end:

@image-classification-dataset-loading-the-dataset-3

::: {.d2l-note .rule}
Same image, axes reordered to `(32, 32, 1)`. `get_dataloader` hands each framework its native layout, so later chapters never think about it.
:::
:::

::: {.slide title="Labels as words, not integers"}
[Loading]{.kicker}

The dataset stores labels as integers 0–9. A tiny helper maps them to names so our spot-checks are readable:

@image-classification-dataset-loading-the-dataset-4
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Reading Minibatches]{.dtitle}

[the iterator that feeds training]{.dsub}
:::
:::

::: {.slide title="The data iterator"}
[Minibatches]{.kicker}

`get_dataloader` shuffles the training split and serves a `batch_size`-sized minibatch each step:

@image-classification-dataset-reading-a-minibatch-1
:::

::: {.slide title="What one minibatch looks like" except="mxnet"}
[Minibatches]{.kicker}

Pull one batch and read its shapes off directly:

@image-classification-dataset-reading-a-minibatch-2

::: {.d2l-note}
64 images, one grayscale channel, $32\times32$ pixels, plus 64 integer labels. A full pass over the training set is I/O-cheap (a second or two), so loading is **not** the training bottleneck.
:::
:::

::: {.slide title="What one minibatch looks like" only="mxnet"}
[Minibatches]{.kicker}

Pull one batch and read its shapes off directly:

@-image-classification-dataset-reading-a-minibatch-2

::: {.d2l-note}
`(64, 1, 32, 32) float32` images and `(64,) int32` labels: 64 channel-first images plus their labels. A full pass over the training set is I/O-cheap, so loading is **not** the training bottleneck.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Looking at the Data]{.dtitle}

[always eyeball what you train on]{.dsub}
:::
:::

::: {.slide title="See the data before you model it"}
[Visualization]{.kicker}

A `visualize` method tiles one validation batch, each image captioned with its class name. Eyeballing data is a cheap, powerful sanity check:

@image-classification-dataset-visualization-2
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

::: {.cols}
::: {.col}
- **Fashion-MNIST**: 10 clothing classes, $28\times28$ grayscale, harder than MNIST but the same size and API.
- A `DataModule` owns each framework's download, transforms, and `train`/`val` loaders.
:::

::: {.col}
- **Channel axis** differs: PyTorch/MXNet $c\times h\times w$, TensorFlow/JAX $h\times w\times c$ (the loader hides it).
- Always **look at your data**; loading stays off the training critical path.
:::
:::
:::
