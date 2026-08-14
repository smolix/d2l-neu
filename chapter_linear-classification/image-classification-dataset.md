```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# The Image Classification Dataset
:label:`sec_fashion_mnist`



[MNIST](https://en.wikipedia.org/wiki/MNIST_database) :cite:`LeCun.Bottou.Bengio.ea.1998` is a widely used image-classification benchmark with 70,000 handwritten-digit images ($28 \times 28$ pixels, 10 classes). Many simple models now achieve high accuracy on it, which makes differences among model classes less visible.

We therefore use **Fashion-MNIST** :cite:`Xiao.Rasul.Vollgraf.2017`, a 2017 replacement with the same structure: 60,000 training and 10,000 test images, each containing $28 \times 28$ grayscale pixels from one of 10 clothing categories. These categories provide a more discriminating comparison of the models developed in this chapter. ImageNet :cite:`Deng.Dong.Socher.ea.2009` supports larger-scale experiments with 1.2 million images and 1000 classes, but Fashion-MNIST keeps the examples interactive.

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

d2l.use_svg_display()
```

## Loading the Dataset

The library provides a preprocessed version of Fashion-MNIST that we can download and read into memory.

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

Fashion-MNIST images are natively $28\times28$. We use
`resize=(32, 32)` to match the spatial dimensions used by later convolution
examples; interpolation changes the pixels and increases per-image storage and
compute, but not the labels. Each image is delivered as a single-channel tensor
of spatial size $32 \times 32$. One subtlety matters here: where the channel
axis lives. There are two conventions, *channel-first* $c \times h \times w$
and *channel-last* $h \times w \times c$. The `get_dataloader` method below
produces the appropriate layout, which we confirm after constructing the loader.
Here $c = 1$ because the images are grayscale; most photographs have $c = 3$.



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

We use the built-in data iterator to read the training and test sets.
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
    # `drop_remainder=train` keeps every training minibatch the same
    # shape, so JAX does not retrace the `@jax.jit`'d step function for
    # a smaller last batch.
    dataset = (tf.data.Dataset.from_tensor_slices(process(*data)).shuffle(
        shuffle_buf).batch(self.batch_size, drop_remainder=train).map(
            resize_fn))
    return d2l.TensorFlowDataLoader(dataset)
```

We read one image to confirm the location of the channel axis.

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

Here is the same batch again, now at batch granularity: the first axis is the batch dimension (64 images per step by default), followed by the per-image shape we just confirmed, and the labels arrive as a matching vector of 64 integers.

```{.python .input #image-classification-dataset-reading-a-minibatch-2}
X, y = next(iter(data.train_dataloader()))
print(X.shape, X.dtype, y.shape, y.dtype)
```

We time one full pass through the training set as a local loader check.
The result depends on storage, worker count, framework, and hardware; it tells
us whether loading is a bottleneck only for this run. If loading is slower than
training on a target system, prefetching or additional loader workers may help.

```{.python .input #image-classification-dataset-reading-a-minibatch-3}
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

## Visualization

We will often be using the Fashion-MNIST dataset. The `d2l` library provides a convenience function `show_images` that lays out a list of images in a grid with optional per-image titles.

We visualize a minibatch before training. This check can reveal mislabeled examples, unexpected transformations, or shape errors early. Here are the images and their corresponding labels (in text)
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

Fashion-MNIST contains grayscale apparel images from 10 categories. A batch has
a batch axis, two spatial axes, and one channel axis: PyTorch and MXNet use
$(n,c,h,w)$, while TensorFlow and JAX use $(n,h,w,c)$. The visualization uses
false color, but the underlying data have one grayscale channel. Loader
throughput must be measured together with model computation on the target
system; it is not guaranteed to be faster than training.


## Exercises

1. [code] **Throughput versus batch size.** Time one full training epoch at
   `batch_size` of 1, 16, 64, 256, and 1024. Plot throughput (images per
   second) against `batch_size`. Explain why throughput rises with batch
   size up to a point and then plateaus.
1. [code] **Worker ablation.** Set `num_workers=0` (single-threaded
   loading) and compare against the default multi-worker setting. State
   under what conditions increasing `num_workers` stops helping.
1. **Channel layout.** PyTorch stores tensors in channel-first order
   $(c, h, w)$, while TensorFlow and JAX use channel-last $(h, w, c)$. Read
   the `get_dataloader` implementations for all four frameworks. Identify
   the step that introduces the channel dimension and the exact place where
   the layouts diverge.
1. [code] **Cost of resizing.** Measure the wall-clock and per-image memory
   cost of the `resize=(32, 32)` step against using the native
   $28 \times 28$ images. Report whether a simple linear classifier's
   validation accuracy changes measurably between the two.
1. **Shuffling and validation.** `get_dataloader` sets `shuffle=train`, so
   only the training loader reshuffles. Explain what would go wrong with
   reported validation metrics and with run-to-run comparability if the
   validation loader were shuffled every epoch as well.
1. [extended] **Loader or compute.** Sweep a grid of `num_workers` and
   `batch_size` values, measure loader throughput at each point, and
   compare it against the forward-pass throughput of the linear classifier
   of :numref:`sec_softmax_scratch`. Produce a heatmap marking the region
   where loading rather than compute is the bottleneck, and state the
   crossover point on your hardware.

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

The Image Classification Dataset<br>**Fashion-MNIST**, the dataset we will classify for the rest of this chapter.

@!image-classification-dataset-visualization-2
:::
:::

::: {.slide title="Fashion-MNIST distinguishes model capacity"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- On **MNIST**, even simple models exceed 95% and a *linear* one tops 90%: models are hard to tell apart.
- **Fashion-MNIST**: a drop-in replacement, same shape and API, but harder clothing classes ($28\times28$ grayscale, 10 classes, 60 k / 10 k).

::: {.d2l-note}
Under the configuration in the softmax-from-scratch section, the linear model reaches about **82%** accuracy; later chapters compare more expressive models on the same data.
:::
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

::: {.slide title="A reusable Fashion-MNIST data module"}
[Loading]{.kicker}

A `DataModule` packages this framework's download, transforms, and `train`/`val` splits behind a common minibatch interface:

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
Same image, axes reordered to `(32, 32, 1)`: one grayscale channel at the end.
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
64 images, one grayscale channel, $32\times32$ pixels, plus 64 integer
labels arriving as a matching vector.
:::
:::

::: {.slide title="What one minibatch looks like" only="mxnet"}
[Minibatches]{.kicker}

Pull one batch and read its shapes off directly:

@-image-classification-dataset-reading-a-minibatch-2

::: {.d2l-note}
`(64, 1, 32, 32) float32` images and `(64,) int32` labels: 64 channel-first
images plus a matching vector of integer labels.
:::
:::

::: {.slide title="Measuring data-loading throughput" except="mxnet"}
[Minibatches · timing]{.kicker}

Time one full pass over all 60,000 training images:

@image-classification-dataset-reading-a-minibatch-3

::: {.d2l-note .rule}
This measurement is specific to the current storage, worker count, framework,
and hardware. If loading limits training throughput, prefetch batches or
increase `num_workers`.
:::
:::

::: {.slide title="Measuring data-loading throughput" only="mxnet"}
[Minibatches · timing]{.kicker}

Time one full pass over all 60,000 training images:

@-image-classification-dataset-reading-a-minibatch-3

::: {.d2l-note .rule}
This measurement is specific to the current storage, worker count, framework,
and hardware. If loading limits training throughput, prefetch batches or
increase `num_workers`.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Looking at the Data]{.dtitle}

[inspect examples before training]{.dsub}
:::
:::

::: {.slide title="Inspecting a minibatch"}
[Visualization]{.kicker}

A `visualize` method tiles one validation batch, each image captioned with its class name. Inspecting examples can reveal label, transformation, and layout errors:

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
- Inspect the data before training, and measure whether loading or model computation limits throughput on the target system.
- Next: a linear classifier on this data and the accuracy it attains under the
  configuration used in the softmax-from-scratch section.
:::
:::
:::
