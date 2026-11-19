```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Image Classification Data
:label:`sec_fashion_mnist`

Our softmax regression model from :numref:`sec_softmax` is ready to train—all
it needs is data. For image classification, choosing the right benchmark
shapes everything our experiments can teach us. Ideally, a teaching dataset
should be challenging enough that genuinely stronger architectures score
visibly higher, yet compact enough that every model trains in minutes
inside an interactive notebook.

The two most famous image benchmarks each satisfy only half of that
bargain. [MNIST](https://en.wikipedia.org/wiki/MNIST_database)
:cite:`LeCun.Bottou.Bengio.ea.1998`—70,000 grayscale $28 \times 28$ images
of handwritten digits across 10 classes—was a formidable challenge in the
1990s (when a top-of-the-line Bell Labs workstation boasted 64 MB of RAM
and 5 MFLOPs, and digit recognition powered automated USPS mail sorting).
Today, however, MNIST has become a sanity check rather than a benchmark:
convolutional networks clear 99% test accuracy and even simple classical
methods hover around 97%, leaving almost no daylight between weak and
strong architectures. At the opposite extreme, ImageNet
:cite:`Deng.Dong.Socher.ea.2009`—1.2 million color photographs across
1,000 categories—separates modern models cleanly, yet training a network on
it requires hours to days of GPU compute, far too slow for interactive
exploration.

To strike the sweet spot, we turn to Fashion-MNIST
:cite:`Xiao.Rasul.Vollgraf.2017`, designed specifically as a drop-in,
harder replacement for MNIST. It keeps the exact same footprint—70,000
grayscale $28 \times 28$ images in 10 classes—but replaces digits with
articles of clothing. Garments are substantially trickier to separate: at
low resolution, a shirt can easily resemble a T-shirt, a pullover, or a
coat. Linear classifiers reach roughly 83% accuracy while strong
convolutional networks surpass 90%, giving us plenty of room to watch
architectural improvements pay off. We will use Fashion-MNIST throughout
this chapter and in many chapters to come.

In this section, we build a reusable `FashionMNIST` data module across all
four frameworks, load it in minibatches, inspect its tensor shapes and I/O
throughput, and visually check a grid of images before letting any model
near them. Taking care with the data pipeline now pays off later, because
every image classifier in the book will read its inputs through the class
we construct here.

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #image-classification-dataset-image-classification-data-1}
%%tab mxnet
import gzip
import os
import time
```

```{.python .input #image-classification-dataset-image-classification-data-1}
%%tab pytorch
import time
import torchvision
from torchvision import transforms
```

```{.python .input #image-classification-dataset-image-classification-data-1}
%%tab tensorflow
import time
```

```{.python .input #image-classification-dataset-image-classification-data-1}
%%tab jax
import tensorflow as tf
import time
```

```{.python .input #image-classification-dataset-image-classification-data-2}
d2l.use_svg_display()
```

## Loading the Dataset

Fashion-MNIST contains 7,000 images for each of the ten clothing categories
introduced in :numref:`sec_softmax` (from T-shirt to ankle boot), split
into 60,000 training images and 10,000 test images—6,000 and 1,000 per
class, respectively. Each pixel stores an integer intensity between 0 and
255, with 0 representing the dark background.

Following our object-oriented design from :numref:`sec_oo-design`, we wrap
the dataset in a `DataModule` subclass that supplies minibatches to
`Trainer` via `train_dataloader` and `val_dataloader`. Our `FashionMNIST`
class downloads the files automatically on first use and keeps both splits
in memory. It accepts two hyperparameters: `batch_size` (examples per
minibatch) and `resize` (the target spatial height and width). By default
we preserve the native $28 \times 28$ resolution; deeper vision
architectures in later chapters will ask for upscaled images via `resize`.

:begin_tab:`pytorch`
The `torchvision` library handles downloading Fashion-MNIST and exposes it
as a standard dataset object. Passing a `transform` pipeline resizes each
image and converts its pixel intensities into floating-point tensors in
$[0, 1]$.
:end_tab:

:begin_tab:`mxnet`
Although Gluon ships its own Fashion-MNIST helper, its split does not match
the official release: most of its 10,000 test images actually come from the
published training set. To keep our numbers comparable with the other
frameworks and the literature, we read the official gzipped files directly.
Each file begins with a brief binary header (16 bytes for images, 8 bytes
for labels), followed by one byte per pixel or label. The `to_dataset`
method converts all images up front into 32-bit floats in $[0, 1]$, leaving
resizing for the minibatch loader so that upscaling to, say,
$224 \times 224$ does not inflate the 60,000 training images to 12 GB in
memory.
:end_tab:

:begin_tab:`tensorflow, jax`
Keras downloads Fashion-MNIST and returns each split directly as a pair of
NumPy arrays holding the raw images and integer labels; our data loader
handles scaling and resizing on the fly.
:end_tab:

```{.python .input #image-classification-dataset-loading-the-dataset-1}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        self.train = self.to_dataset(*self.read(train=True))
        self.val = self.to_dataset(*self.read(train=False))

    def read(self, train):
        """Images and labels of one split, read from the published files."""
        url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
        folder = os.path.join(self.root, 'FashionMNIST', 'raw')
        arrays = []
        for kind, header in (('images-idx3', 16), ('labels-idx1', 8)):
            fname = f"{'train' if train else 't10k'}-{kind}-ubyte.gz"
            if not os.path.exists(os.path.join(folder, fname)):
                d2l.download(url + fname, folder)
            with gzip.open(os.path.join(folder, fname)) as f:
                data = bytearray(f.read())[header:]  # One byte per value
            arrays.append(d2l.tensor(data, dtype='uint8'))
        images, labels = arrays
        return images.reshape(-1, 28, 28, 1), labels.astype('int32')

    def to_dataset(self, images, labels):
        """Convert all images at once: channel axis first, values in [0, 1]."""
        X = images.asnumpy().transpose(0, 3, 1, 2).astype('float32') / 255
        return gluon.data.ArrayDataset(X, labels.asnumpy())
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

Let us instantiate the dataset and verify the size of each split.

```{.python .input #image-classification-dataset-loading-the-dataset-2}
%%tab mxnet, pytorch
data = FashionMNIST()
len(data.train), len(data.val)
```

```{.python .input #image-classification-dataset-loading-the-dataset-2}
%%tab tensorflow, jax
data = FashionMNIST()
len(data.train[0]), len(data.val[0])
```

Notice that Fashion-MNIST ships with only two splits—no dedicated
validation set. Following standard practice in this book, we use the 10,000
test images as our validation split so that `Trainer` can report validation
loss after each epoch. That convenience comes with an important caveat: a
test set consulted repeatedly while comparing models ceases to be a truly
unbiased final benchmark. We will measure exactly how much optimism
repeated reuse introduces in :numref:`chap_classification_generalization`.

Because the stored labels are integer indices from 0 to 9, let us add a
small helper that translates numeric indices back into human-readable
garment names.

```{.python .input #image-classification-dataset-loading-the-dataset-4}
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Reading Minibatches

Inside a deep learning framework, an image is represented as a
three-dimensional tensor with $c$ *channels*, height $h$, and width $w$.
Grayscale images have $c = 1$, standard color photographs have $c = 3$
(red, green, and blue), and hyperspectral satellite sensors can have over a
hundred channels. Frameworks differ in where they place the channel axis:
*channel-first* libraries store an image with shape $c \times h \times w$,
whereas *channel-last* libraries use $h \times w \times c$.

Rather than building our own iterator from scratch, we rely on each
framework's built-in data loader. During training, the loader reshuffles
the examples at the start of every epoch so minibatches vary across epochs;
during validation, it reads examples deterministically in their stored
order. Both loaders convert pixel intensities from 0--255 integers to
floats in $[0, 1]$ and resize images when requested. Because 64 does not
divide 60,000 evenly, the final minibatch of each training epoch is smaller
(32 images here), and both loaders retain it so every image is seen once
per epoch.

:begin_tab:`pytorch`
PyTorch's `DataLoader` batches and shuffles examples automatically. Its
`num_workers` background processes (4 by default in `DataModule`) load and
transform upcoming minibatches in parallel while the model trains on the
current one.
:end_tab:

:begin_tab:`mxnet`
Gluon's `DataLoader` reads the dataset in minibatches. Because `to_dataset`
already converted every image to a float array in memory, assembling
batches directly in the main process is faster than sending them across
worker processes, so `FashionMNIST` ignores `DataModule.num_workers`. The
custom `batchify_fn` stacks each minibatch and resizes it in a single
vectorized call whenever `resize` differs from `(28, 28)`.
:end_tab:

:begin_tab:`tensorflow`
Our loader uses a `tf.data` pipeline whose shuffle buffer matches the full
training set size, guaranteeing a uniformly random permutation on every
epoch.
:end_tab:

:begin_tab:`jax`
Because JAX does not ship its own image loader, we reuse TensorFlow's
`tf.data` pipeline and wrap it in a lightweight `TensorFlowDataLoader`
helper that yields re-iterable NumPy minibatches directly consumable by
JAX.
:end_tab:

```{.python .input #image-classification-dataset-reading-minibatches-1}
%%tab jax
class TensorFlowDataLoader:  #@save
    """Expose a tf.data.Dataset as re-iterable NumPy batches."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return self.dataset.as_numpy_iterator()

    def __len__(self):
        return len(self.dataset)
```

```{.python .input #image-classification-dataset-reading-a-minibatch-1}
%%tab mxnet
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    data = self.train if train else self.val
    return gluon.data.DataLoader(data, self.batch_size, shuffle=train,
                                 batchify_fn=self.batchify)

@d2l.add_to_class(FashionMNIST)  #@save
def batchify(self, examples):
    """Stack examples into a minibatch and resize it if asked to."""
    X, y = gluon.data.batchify.Stack()(examples)
    if X.shape[2:] != tuple(self.resize):  # Resizing wants channels last
        X = npx.image.resize(X.transpose(0, 2, 3, 1), self.resize[::-1])
        X = X.transpose(0, 3, 1, 2)
    return X, y
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
    X, y = self.train if train else self.val
    X = tf.expand_dims(X, axis=3) / 255  # Channel axis last; values in [0, 1]
    resize = lambda X, y: (tf.image.resize(X, self.resize), y)
    return tf.data.Dataset.from_tensor_slices(
        (X, tf.cast(y, tf.int32))).shuffle(len(y) if train else 1).batch(
            self.batch_size).map(resize)
```

```{.python .input #image-classification-dataset-reading-a-minibatch-1}
%%tab jax
@d2l.add_to_class(FashionMNIST)  #@save
def get_dataloader(self, train):
    X, y = self.train if train else self.val
    X = tf.expand_dims(X, axis=3) / 255  # Channel axis last; values in [0, 1]
    resize = lambda X, y: (tf.image.resize(X, self.resize), y)
    return TensorFlowDataLoader(tf.data.Dataset.from_tensor_slices(
        (X, tf.cast(y, tf.int32))).shuffle(len(y) if train else 1).batch(
            self.batch_size).map(resize))
```

Let us pull a single minibatch and inspect the layout of its first image.

```{.python .input #image-classification-dataset-loading-the-dataset-3}
%%tab mxnet, pytorch
X, y = next(iter(data.train_dataloader()))
X[0].shape  # Channel-first: (channels, height, width)
```

```{.python .input #image-classification-dataset-loading-the-dataset-3}
%%tab tensorflow, jax
X, y = next(iter(data.train_dataloader()))
X[0].shape  # Channel-last: (height, width, channels)
```

Stacking examples into a minibatch prepends a leading batch axis to the
images, while the labels arrive as a one-dimensional vector with one entry
per image.

```{.python .input #image-classification-dataset-reading-a-minibatch-2}
print(X.shape, X.dtype, y.shape, y.dtype)
```

The images arrive as 32-bit floats and the labels as integer class indices.
A single minibatch of 64 images occupies $64 \times 784 \times 4$ bytes
(roughly 200 KB), and the entire 60,000-image training set in float32 takes
only about 190 MB. By modern standards that is tiny: the whole dataset sits
comfortably in RAM, so the loader spends its time converting, resizing, and
collating arrays rather than waiting on disk.

Still, is our data pipeline fast enough? Reading data should never be the
bottleneck that leaves a GPU or CPU idling between steps. Let us time one
complete pass over the training set with no model attached.

```{.python .input #image-classification-dataset-reading-a-minibatch-3}
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

The exact timing depends on your hardware, framework, and worker count, so
treat the number above as specific to the machine that executed this
notebook. Whatever it reads, it establishes a hard floor: no training epoch
can finish faster than reading its data. Whenever training feels sluggish,
benchmarking the loader alone is the first diagnostic to run—if a full
training epoch takes barely longer than a data-only pass, your model is
I/O-bound, and adding worker processes or prefetching will yield bigger
speedups than tweaking the network itself.

## Inspecting the Images

Always look at your data before training on it. Human eyes are remarkably
good at catching bugs that loss curves never flag: misaligned labels,
upside-down or transposed images, or a misplaced channel axis. A neural
network will happily train on corrupted inputs without throwing an
exception—it will just perform poorly. Let us add a `visualize` method to
`FashionMNIST` that displays a grid of grayscale images titled with their
text labels via `d2l.show_images`. Because Matplotlib otherwise renders
single-channel arrays using a false-color palette, we explicitly request
`cmap='gray'`, and because $c = 1$, reshaping drops the channel axis
cleanly in both channel-first and channel-last layouts.

```{.python .input #image-classification-dataset-visualization-2}
@d2l.add_to_class(FashionMNIST)  #@save
def visualize(self, batch, nrows=1, ncols=8, labels=None):
    """Show the images of a batch in grayscale, titled by their labels."""
    X, y = batch
    if labels is None:
        labels = self.text_labels(y)
    X = d2l.reshape(X, (-1, *self.resize))  # Drop the channel axis
    d2l.show_images(X, nrows, ncols, titles=labels, cmap='gray')
```

Let us display the first three validation images of every class—one column
per category—taking advantage of the fact that the validation loader always
returns images in deterministic order.

```{.python .input #image-classification-dataset-inspecting-the-images-1}
X, y = next(iter(data.val_dataloader()))
labels = d2l.numpy(y)
idx = [int(np.flatnonzero(labels == c)[r]) for r in range(3) for c in range(10)]
titles = data.text_labels(range(10)) + [''] * 20
data.visualize([d2l.stack([X[i] for i in idx]), None], nrows=3, ncols=10,
               labels=titles)
```

Everything checks out: the labels match the garments, every item is
upright, and the background is dark as expected. Even a quick glance,
however, reveals why Fashion-MNIST is far richer than handwritten digits.
T-shirts, pullovers, coats, and shirts share nearly identical silhouettes
and differ only in collars, buttons, or sleeve lengths—subtle cues that
$28 \times 28$ pixels barely resolve—while sandals, sneakers, and ankle
boots form a second look-alike cluster. Trousers and bags, by contrast,
stand out unmistakably. Already we can predict where a linear classifier
will struggle most; in :numref:`sec_softmax_scratch`, we will train one and
see whether our intuition bears out.

## Summary

Fashion-MNIST provides a compact yet genuinely challenging image benchmark:
60,000 training and 10,000 test grayscale images of $28 \times 28$ pixels
spanning 10 apparel categories—fast enough for interactive notebooks yet
hard enough to separate weak models from strong ones. Our `FashionMNIST`
data module loads the published dataset uniformly across all four
frameworks, yields minibatches as four-dimensional float tensors in
$[0, 1]$ alongside integer class indices, reshuffles training data each
epoch, and retains the final short minibatch so every image is visited.
Finally, benchmarking the data loader and inspecting labeled images by eye
are essential sanity checks before any model begins training.

## Exercises

1. [code] **Class means.** Compute the mean image of each class over the
   training set, and show the ten means as a grid of images.
    1. Compute the Euclidean distance between every pair of class means and
       present the distances as a $10 \times 10$ table. Which pairs are
       closest, and which class is farthest from all others?
    1. Predict from the table which classes a linear classifier will confuse
       most often and which it will almost never confuse. Keep your
       prediction; :numref:`sec_softmax_scratch` computes the answer.
    1. Name a pair of classes that a person tells apart easily although
       their means are close, and say which pixels carry the difference.

1. **Shuffling and validation.** `get_dataloader` reshuffles the training
   split every epoch and reads the validation split in the same order every
   epoch. Suppose the validation loader, with the default `batch_size=64`,
   were reshuffled every epoch as well.
    1. Explain whether the validation loss and accuracy, averaged over all
       10,000 validation examples, would change.
    1. The `Trainer` weights each validation batch by its number of
       examples. Suppose it took the unweighted mean of the per-batch
       values instead. Explain whether that mean would change.
    1. Explain whether the images that `visualize` shows for the first
       validation batch would change.

1. [code] **Reshaping versus transposing.** `visualize` removes the
   channel axis with a reshape, which is correct only because $c = 1$.
    1. A minibatch of color images has shape $(n, 3, h, w)$ in channel-first
       layout. Show that reshaping it to $(n, h, w, 3)$ does not move the
       channel axis to the end, by computing which stored value lands at
       position $(0, 0, 0, 1)$ of the result.
    1. Write the transposition that converts it correctly, apply both
       versions to a minibatch of 8 random arrays of shape $(3, 4, 5)$, and
       check which one agrees with indexing the original array.

:begin_tab:`pytorch`
4. [code] **Loader throughput.** The time of one pass over the training
   set depends on the minibatch size and on how many processes prepare the
   minibatches.
    1. Time one full pass of `train_dataloader` for `FashionMNIST`
       constructed with `batch_size` of 1, 16, 64, 256, and 1024, and plot
       the throughput in images per second against `batch_size`. Explain
       the shape of the curve in terms of per-batch overhead and per-image
       work.
    1. `get_dataloader` passes the `num_workers` attribute, which
       `DataModule` sets to 4 by default, to `torch.utils.data.DataLoader`.
       For `data = FashionMNIST()`, time one pass of
       `data.train_dataloader()` with `data.num_workers` set to 0, 1, 2, 4,
       and 8, where 0 loads in the main process. State under what
       conditions more workers stop helping.
:end_tab:

:begin_tab:`jax`
4. [code] **Loader throughput.** The time of one pass over the training
   set depends on the minibatch size and on how much of the pipeline runs
   in parallel.
    1. Time one full pass of `train_dataloader` for `FashionMNIST`
       constructed with `batch_size` of 1, 16, 64, 256, and 1024, and plot
       the throughput in images per second against `batch_size`. Explain
       the shape of the curve in terms of per-batch overhead and per-image
       work.
    1. The `tf.data` pipeline in `get_dataloader` does not use the
       `num_workers` attribute of `DataModule`; its parallelism is set per
       stage. Time one pass as written, then with
       `num_parallel_calls=tf.data.AUTOTUNE` in the `map` call and a
       trailing `.prefetch(tf.data.AUTOTUNE)`. State under what conditions
       more parallelism stops helping.
:end_tab:

:begin_tab:`tensorflow`
4. [code] **Loader throughput.** The time of one pass over the training
   set depends on the minibatch size and on how much of the pipeline runs
   in parallel.
    1. Time one full pass of `train_dataloader` for `FashionMNIST`
       constructed with `batch_size` of 1, 16, 64, 256, and 1024, and plot
       the throughput in images per second against `batch_size`. Explain
       the shape of the curve in terms of per-batch overhead and per-image
       work.
    1. The `tf.data` pipeline in `get_dataloader` does not use the
       `num_workers` attribute of `DataModule`; its parallelism is set per
       stage. Time one pass as written, then with
       `num_parallel_calls=tf.data.AUTOTUNE` in the `map` call and a
       trailing `.prefetch(tf.data.AUTOTUNE)`. State under what conditions
       more parallelism stops helping.
:end_tab:

:begin_tab:`mxnet`
4. [code] **Loader throughput.** The time of one pass over the training
   set depends on the minibatch size and on the size of the images that
   the loader delivers.
    1. Time one full pass of `train_dataloader` for `FashionMNIST`
       constructed with `batch_size` of 1, 16, 64, 256, and 1024, and plot
       the throughput in images per second against `batch_size`. Explain
       the shape of the curve in terms of per-batch overhead and per-image
       work.
    1. `batchify` resizes each minibatch as the loader reads it. Time one
       pass of `train_dataloader` for `FashionMNIST(resize=(s, s))` with
       $s$ = 28, 56, 112, and 224, and plot the time against the number of
       pixels per image. Which part of the time grows with the number of
       pixels, and which does not?
:end_tab:

5. [code] **Upsampling and linear models.** ● Treat the resizing from
   $28 \times 28$ to $32 \times 32$ as a linear map
   $\mathbf{x}' = \mathbf{A}\mathbf{x}$ from the 784 original pixels to the
   1024 resized ones, where the columns of
   $\mathbf{A} \in \mathbb{R}^{1024 \times 784}$ are linearly independent.
    1. Show that softmax regression on $\mathbf{x}'$ can represent exactly
       the classifiers that softmax regression on $\mathbf{x}$ can.
    1. After :numref:`sec_softmax_scratch`, train `SoftmaxRegressionScratch`
       on data from `FashionMNIST()` and from
       `FashionMNIST(resize=(32, 32))`, with `num_inputs` set to the number
       of pixels per image. Compare the validation accuracies and explain
       any difference in light of the previous part.

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
[Dive into Deep Learning · §3.2]{.kicker}

Image Classification Data<br>**Fashion-MNIST · minibatches · a look at the images**
:::
:::

::: {.slide title="MNIST Is Too Easy, ImageNet Too Big for a Notebook"}
[The problem]{.kicker}

| dataset | images | classes | what goes wrong |
|---|---|---|---|
| MNIST | 70,000 digits, $28\times28$, gray | 10 | CNNs above 99%, classical methods about 97%: models barely differ |
| ImageNet | 1.2 million photos, color | 1,000 | hours to days of GPU time per run |
| **Fashion-MNIST** | **70,000 garments, $28\times28$, gray** | **10** | **linear models about 83%, good CNNs above 90%** |

::: {.d2l-note}
Fashion-MNIST keeps MNIST's format, loads in seconds, and leaves room
between weak and strong models.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- load Fashion-MNIST in your framework, with the same images as in the
  other three;
- read the shape of a minibatch in channel-first or channel-last layout;
- time the loader, a lower bound on the time of an epoch;
- check images and labels by eye before training, and say which classes
  resemble each other.

**Running example:** 60,000 training and 10,000 test images; minibatches
of 64 images of $28 \times 28$ pixels, 32-bit floats.
:::

::: {.slide title="The Data Every Classifier in This Chapter Reads"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Softmax Regression | probabilities over classes, cross-entropy |
| **Image Classification Data** | **Fashion-MNIST in minibatches** |
| Softmax Regression from Scratch | softmax regression by hand, the `Classifier` class |
| Concise Implementation | framework layers, a stable loss |
| Measuring Classifiers | precision, recall, costs, calibration |
| Generalization in Classification | error bars, test-set reuse, VC dimension |
| Environment and Distribution Shift | correcting for shifted data |
:::

::: {.col .narrow}
- **Uses:** `DataModule` and `Trainer` from linear regression.
- **Skips:** data augmentation, color images, data larger than memory.
:::
:::
:::

::: {.slide title="Ten Classes, 6,000 Training and 1,000 Test Images Each"}
[The dataset]{.kicker}

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| T-shirt | trouser | pullover | dress | coat | sandal | shirt | sneaker | bag | ankle boot |

Labels are the class indices 0–9; a lookup gives the names:

@image-classification-dataset-loading-the-dataset-4
:::

::: {.slide title="Every Framework Loads the Same Published Files"}
[The dataset]{.kicker}

- **PyTorch:** torchvision; **TensorFlow, JAX:** Keras; **MXNet:** reads
  the published gzip files itself.
- `FashionMNIST(batch_size=64, resize=(28, 28))` keeps both splits in
  memory.

@!image-classification-dataset-loading-the-dataset-2

::: {.d2l-note .warn}
The 10,000 **test** images serve as validation data. A test set consulted
for every comparison stops being an unbiased test.
:::
:::

::: {.slide title="Gluon's Own Copy Has a Different Test Set" only="mxnet"}
[The dataset]{.kicker}

- Gluon ships a copy of Fashion-MNIST, split differently from the published
  dataset: about 80% of its 10,000 test images are **training** images of
  the published one.
- Results on it cannot be compared with published numbers.
- So `FashionMNIST.read` opens the published files: a 16-byte header, then
  $28 \times 28 = 784$ bytes per image (8 bytes and one byte per label for
  the labels).
:::

::: {.slide title="The Training Loader Reshuffles; Both Keep the Short Last Minibatch"}
[Reading minibatches]{.kicker}

@image-classification-dataset-reading-a-minibatch-1

- Training: a new random order every epoch, so the minibatches differ from
  epoch to epoch. Validation: the stored order every time.
- Pixel values become floats in $[0, 1]$; images are resized if asked.
- The last minibatch of an epoch is short (32 images at a batch size of
  64), and both loaders keep it, so every epoch covers every image.
:::

::: {.slide title="Check: How Many Minibatches per Epoch?"}
[Check]{.kicker}

With minibatches of **256**, how many minibatches does

1. one training epoch have, and how large is the last one?
2. one pass over the validation set have?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.**

1. $60{,}000 = 234 \times 256 + 96$: 235 minibatches, the last of 96.
2. $10{,}000 = 39 \times 256 + 16$: 40 minibatches, the last of 16.
:::
:::
:::

::: {.slide title="Channel-First or Channel-Last, Depending on the Framework"}
[Reading minibatches]{.kicker}

An image is $c \times h \times w$ or $h \times w \times c$, with
$c$ channels (1 for gray, 3 for color) and $h \times w$ pixels:

@image-classification-dataset-loading-the-dataset-3

::: {.d2l-note .rule}
Same image in every framework; only the position of the channel axis
differs.
:::
:::

::: {.slide title="A Minibatch Is a 4-D Tensor of Images and a Vector of Labels"}
[Reading minibatches]{.kicker}

@!image-classification-dataset-reading-a-minibatch-2

- PyTorch, MXNet: $(n, c, h, w)$; TensorFlow, JAX: $(n, h, w, c)$.
- Images: 32-bit floats in $[0, 1]$. Labels: integers 0–9.
- One minibatch of images: $64 \times 784 \times 4$ bytes $\approx$ 200 KB.
:::

::: {.slide title="Check: How Much Memory?"}
[Check]{.kicker}

1. A minibatch of shape $(256, 1, 28, 28)$ in 32-bit floats?
2. All 60,000 training images as 32-bit floats? As bytes, as stored?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.**

1. $256 \times 784 \times 4 = 802{,}816$ bytes, about 0.8 MB.
2. $60{,}000 \times 784 \times 4 \approx 188$ MB as floats; a quarter of
   that, 47 MB, as bytes. Both fit in memory easily.
:::
:::
:::

::: {.slide title="Reading One Epoch Takes Seconds"}
[Loader speed]{.kicker}

One pass over the 60,000 training images, with no model:

@image-classification-dataset-reading-a-minibatch-3

::: {.d2l-note .rule}
The time depends on storage, workers, framework, and hardware: it holds only
for the machine that ran it.
:::
:::

::: {.slide title="An Epoch Takes at Least as Long as Its Slower Stage"}
[Loader speed]{.kicker}

Loader workers prepare the next minibatches while the model trains on the
current one, so

$$\textrm{time per epoch} \;\ge\; \max(\textrm{loading time},\ \textrm{compute time}).$$

- Loading slower: the processor waits. Add workers, prefetch, keep the
  data in memory.
- Compute slower: the loader is fast enough; speed up the model instead.
:::

::: {.slide title="Look at the Images Before Training"}
[Inspecting the images]{.kicker}

@-image-classification-dataset-visualization-2

- Grayscale, as the model receives them, each titled with its label.
- `reshape` drops the channel axis in either layout, since $c = 1$.
:::

::: {.slide title="Shirts and Coats Look Alike at 28 × 28 Pixels"}
[Inspecting the images]{.kicker}

The first three validation images of each class, one class per column:

@!image-classification-dataset-inspecting-the-images-1

::: {.d2l-note}
Similar classes (shirt, coat, pullover, T-shirt) are where a classifier
will make most of its mistakes.
:::
:::

::: {.slide title="Check: Which Bug Would the Gallery Show?"}
[Check]{.kicker}

Which of these bugs does a gallery reveal before training, and which only a
poor loss after it?

1. labels shifted by one class;
2. images transposed;
3. pixels left in $[0, 255]$;
4. every image delivered twice per epoch.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answers.** 1 and 2 show at a glance: wrong titles, sideways garments.
3 shows in the printed values, not the picture, since the color map
rescales. 4 shows in neither: count the images per epoch.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| step | running example |
|---|---|
| dataset | 60,000 + 10,000 garments, 10 classes, $28 \times 28$ gray |
| split | the published split; test images as validation |
| minibatch | 64 images as a four-dimensional tensor, floats in $[0, 1]$ |
| epoch | 938 minibatches, reshuffled every time, the last one short |
| checks | loader time, a lower bound on the epoch; a gallery with labels: the four tops look alike |

**Next:** softmax regression from scratch, trained on these images.
:::
