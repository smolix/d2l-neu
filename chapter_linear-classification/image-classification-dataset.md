```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Image Classification Data
:label:`sec_fashion_mnist`

To train and test softmax regression, we need an image classification benchmark that is rich enough to separate weak models from strong ones yet compact enough to train in seconds inside an interactive notebook. Classic [MNIST](https://en.wikipedia.org/wiki/MNIST_database) handwritten digits :cite:`LeCun.Bottou.Bengio.ea.1998` ($70{,}000$ grayscale $28 \times 28$ images across 10 digit classes) was a landmark benchmark in the 1990s, but today even trivial models clear $97\%$ accuracy and convolutional networks exceed $99\%$, leaving almost no headroom to see architectural improvements work. At the opposite extreme, ImageNet :cite:`Deng.Dong.Socher.ea.2009` ($1.2$ million color photos across $1{,}000$ categories) requires hours of GPU compute per run.

*Fashion-MNIST* :cite:`Xiao.Rasul.Vollgraf.2017` hits the sweet spot: it keeps MNIST's exact footprint—$60{,}000$ training and $10{,}000$ test grayscale images of size $28 \times 28$ across $10$ balanced classes—but replaces digits with articles of clothing (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot). On Fashion-MNIST, linear softmax regression reaches roughly $83\%$ accuracy while convolutional networks climb past $90\%$. In this section, we build the `FashionMNIST(d2l.DataModule)` class that will feed every image classifier across the next several chapters.

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

Fashion-MNIST contains $6{,}000$ training images and $1{,}000$ test images for each of its $10$ garment categories ($60{,}000$ and $10{,}000$ in total), stored as 8-bit unsigned integers in $[0, 255]$ ($0$ = dark background). Following our object-oriented design (:numref:`sec_oo-design`), `FashionMNIST` subclasses `d2l.DataModule` and accepts `batch_size` and an optional spatial `resize` tuple (defaulting to $(28, 28)$):

:begin_tab:`pytorch`
`torchvision.datasets.FashionMNIST` downloads the dataset and applies a `transforms.Compose` pipeline that resizes each image and scales pixel intensities to float32 tensors in $[0, 1]$.
:end_tab:

:begin_tab:`mxnet`
Because Gluon's built-in Fashion-MNIST helper uses a nonstandard train/test split, `FashionMNIST.read` downloads and parses the official gzipped IDX byte files directly (skipping the 16-byte image header and 8-byte label header), and `to_dataset` scales the pixels to float32 in $[0, 1]$.
:end_tab:

:begin_tab:`tensorflow, jax`
`tf.keras.datasets.fashion_mnist.load_data()` returns the official training and test splits as `(images, labels)` NumPy arrays.
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


Because Fashion-MNIST ships with only two splits, we use the $10{,}000$-image test split as our validation set during training (we analyze the statistical effect of repeated test-set evaluation in :numref:`chap_classification_generalization`). Let us also attach `text_labels` to translate integer class indices $0, \ldots, 9$ into human-readable garment names:

```{.python .input #image-classification-dataset-loading-the-dataset-4}
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['T-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Reading Minibatches

In deep learning frameworks, each image in a minibatch of size $n$ with $c$ channels ($c = 1$ for grayscale, $c = 3$ for RGB color), height $h$, and width $w$ is stored either in *channel-first* layout $(n, c, h, w)$ or *channel-last* layout $(n, h, w, c)$, depending on the framework.

:begin_tab:`mxnet`
MXNet stores images channel-first.
:end_tab:

:begin_tab:`pytorch`
PyTorch stores images channel-first.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow stores images channel-last.
:end_tab:

:begin_tab:`jax`
JAX stores images channel-last. We wrap `tf.data.Dataset` in a lightweight `TensorFlowDataLoader` helper that exposes `.as_numpy_iterator()` and `__len__` so each epoch yields NumPy minibatches directly:
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

Now we attach `get_dataloader(self, train)` to `FashionMNIST`, wrapping the framework's native data loader so it reshuffles the training split on every epoch and retains the final partial minibatch ($60{,}000 = 937 \times 64 + 32$) so every image is visited once per epoch:

:begin_tab:`mxnet`
A `batchify` helper resizes each minibatch as the loader reads it.
:end_tab:

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


Let us fetch one minibatch `(X, y)` from `train_dataloader()` and inspect both the single-image layout and the full minibatch tensor shapes and dtypes:

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


At $64 \times 28 \times 28 \times 4 \approx 200\textrm{ KB}$ per minibatch (and $\sim 190\textrm{ MB}$ for all $60{,}000$ float32 training images), the entire dataset fits comfortably in RAM. Still, benchmarking the data loader by itself—timing one complete pass over `train_dataloader()` with no model attached—is a vital habit: no training epoch can ever run faster than its data pipeline!

```{.python .input #image-classification-dataset-reading-a-minibatch-3}
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```


## Inspecting the Images

Always inspect a sample of labeled images by eye before training any model: misaligned labels, inverted intensities, or a scrambled reshape will ruin training without raising a runtime error. We attach a `visualize` method to `FashionMNIST` that drops the singleton channel axis and plots a grid of grayscale images titled with their text labels, and then display the first three validation images of each of the ten classes:

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


Even a quick visual scan previews what our linear classifier is up against: trousers and bags have unmistakable silhouettes, sandals/sneakers/ankle boots form a distinct footwear cluster, and the four upper-body garments (T-shirt, pullover, coat, and shirt) share almost the same boxy torso outline, differing only in subtle collar, sleeve, and button details that $28 \times 28$ resolution barely resolves.

## Summary

- **Fashion-MNIST benchmark:** $60{,}000$ training and $10{,}000$ test grayscale $28 \times 28$ images across $10$ balanced apparel classes, challenging enough to separate linear models ($\sim 83\%$) from convolutional networks ($>90\%$) while running in seconds.
- **`FashionMNIST` data module:** Delivers float32 image minibatches in $[0, 1]$ of shape $(n, 1, h, w)$ (channel-first) or $(n, h, w, 1)$ (channel-last) alongside integer class labels, with built-in resizing and grayscale visualization.

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


:begin_tab:`pytorch`
2. [code] **Loader throughput.** The time of one pass over the training
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
2. [code] **Loader throughput.** The time of one pass over the training
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
2. [code] **Loader throughput.** The time of one pass over the training
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
2. [code] **Loader throughput.** The time of one pass over the training
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

3. [code] **Upsampling and linear models.** ● Treat the resizing from
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

The Image Classification Dataset: Fashion-MNIST & Data Pipelines<br>**Fashion-MNIST benchmark · `DataModule` API · tensor layouts · minibatch loaders · pipeline throughput · visual debugging**
:::
:::

::: {.slide title="Why Fashion-MNIST Is Our Running Vision Benchmark"}
[Benchmark Selection]{.kicker}

::: {.cols .vc}
::: {.col}
An effective development benchmark must separate weak models from strong architectures while fitting in memory and training in seconds:

- **MNIST (1998) is too easy:** Linear models reach $\approx 92\%$ and trivial nets exceed $99\%$, leaving almost no headroom to test new ideas.
- **ImageNet (2009) is too heavy:** $1.2\text{M}$ color photos across $1{,}000$ classes require hours of multi-GPU compute per experiment.
- **Fashion-MNIST (2017) hits the sweet spot:** Drop-in $70{,}000$-image ($28\times 28$) footprint where linear softmax reaches $\sim 83\%$ and CNNs exceed $90\%$.
:::

::: {.col .fig .big}
@fig:ch03-cls-benchmark-sweetspot
:::
:::
:::

::: {.slide title="Ten Balanced Apparel Categories"}
[Class Taxonomy]{.kicker}

::: {.cols .vc}
::: {.col}
Fashion-MNIST contains $6{,}000$ training and $1{,}000$ test images for each of $q = 10$ clothing categories ($0$-based indices $0, \ldots, 9$):

@image-classification-dataset-loading-the-dataset-4

- **Silhouette clusters:** $4$ boxy upper-body tops (`0, 2, 4, 6`), $3$ footwear items (`5, 7, 9`), and $3$ distinct shapes (`1` trouser, `3` dress, `8` bag).
:::

::: {.col .fig}
@fig:ch03-cls-garment-taxonomy
:::
:::
:::

::: {.slide title="Encapsulating Fashion-MNIST in a DataModule"}
[Object-Oriented Data Pipeline]{.kicker}

`FashionMNIST` subclasses `d2l.DataModule`; `resize=(h, w)` changes resolution without new training code:

@-image-classification-dataset-loading-the-dataset-1
:::

::: {.slide title="Rescaling Pixels at Transform Time"}
[Object-Oriented Data Pipeline]{.kicker}

::: {.cols .vc}
::: {.col}
- **Normalization at transform time:** Raw 8-bit pixels (`uint8` in $[0, 255]$, dark background $= 0$) rescale to `float32` in $[0, 1]$.
- **One place to change:** Because the transform lives in the `DataModule`, every model that loads Fashion-MNIST sees the same $[0, 1]$ inputs.
:::

::: {.col .fig}
@fig:ch03-cls-pixel-rescale
:::
:::
:::

::: {.slide title="Verifying the Official Train and Validation Splits"}
[Dataset Splits & Pitfalls]{.kicker}

::: {.cols .vc}
::: {.col}
Instantiating `FashionMNIST()` loads the official $60{,}000$-image training split and $10{,}000$-image evaluation split into memory:

@image-classification-dataset-loading-the-dataset-2

- **Compact memory footprint:** At $784$ pixels/image, $60{,}000$ training images take $\approx 47\text{ MB}$ in `uint8` or $\approx 188\text{ MB}$ in `float32`.
- **Test-set reuse caveat:** Because Fashion-MNIST ships with only two splits, `data.val` uses the $10\text{k}$ test split (§{chap_classification_generalization} quantifies reuse inflation).
:::

::: {.col .fig}
@fig:ch03-cls-dataset-splits
:::
:::
:::

::: {.slide title="Why MXNet Reads the Raw IDX Byte Files Directly" only="mxnet"}
[Reproducibility Across Frameworks]{.kicker}

::: {.cols .vc}
::: {.col}
Why does MXNet parse the official `*-ubyte.gz` files directly instead of calling `gluon.data.vision.FashionMNIST`?

- **Nonstandard split in Gluon's helper:** Gluon's built-in helper draws roughly $80\%$ of its $10{,}000$ "test" images from the published **training** split!
- **Cross-framework comparability:** Parsing the official IDX binary files directly (skipping the $16$-byte image header and $8$-byte label header) guarantees identical $60\text{k} + 10\text{k}$ splits across all frameworks.
:::

::: {.col .fig .big}
@fig:ch03-cls-idx-header
:::
:::
:::

::: {.slide title="Building the Minibatch Loader"}
[Shuffling & Batching]{.kicker}

The `get_dataloader(train)` method wraps the dataset in an iterable minibatch loader:

@image-classification-dataset-reading-a-minibatch-1
:::

::: {.slide title="Epoch Reshuffling and the Short Final Batch"}
[Shuffling & Batching]{.kicker}

::: {.cols .vc}
::: {.col}
- **Epoch reshuffling (`shuffle=train`):** Training reshuffles all $60{,}000$ indices every epoch; validation iterates in deterministic order.
- **Retaining the short final minibatch:** At `batch_size=64`, $60{,}000 = 937 \times 64 + 32$ yields $938$ batches per epoch with zero dropped examples.
:::

::: {.col .fig}
@fig:ch03-cls-epoch-batches
:::
:::
:::

::: {.slide title="Single-Image Layouts: Channel-First vs. Channel-Last" only="pytorch,mxnet"}
[Framework Tensor Conventions]{.kicker}

::: {.cols .vc}
::: {.col}
A single grayscale image ($c = 1$, $h = 28$, $w = 28$) is stored as a 3D tensor whose channel-axis position depends on the framework:

@image-classification-dataset-loading-the-dataset-3

- **Channel-first $(c, h, w) = (1, 28, 28)$:** This framework puts the channel axis first (`NCHW` for a minibatch).
- **The other convention:** Channel-last $(h, w, c)$ (`NHWC`) stores the channels of each pixel together; both flatten to the same $784$ features.
:::

::: {.col .fig}
@fig:ch03-cls-nchw-vs-nhwc
:::
:::
:::

::: {.slide title="Single-Image Layouts: Channel-First vs. Channel-Last" only="tensorflow,jax"}
[Framework Tensor Conventions]{.kicker}

::: {.cols .vc}
::: {.col}
A single grayscale image ($c = 1$, $h = 28$, $w = 28$) is stored as a 3D tensor whose channel-axis position depends on the framework:

@image-classification-dataset-loading-the-dataset-3

- **Channel-last $(h, w, c) = (28, 28, 1)$:** This framework puts the channel axis last (`NHWC` for a minibatch).
- **The other convention:** Channel-first $(c, h, w)$ (`NCHW`) stores each channel as one contiguous plane; both flatten to the same $784$ features.
:::

::: {.col .fig}
@fig:ch03-cls-nchw-vs-nhwc
:::
:::
:::

::: {.slide title="Inspecting a Full Minibatch of Images and Labels"}
[4D Image Tensors & 1D Label Vectors]{.kicker}

::: {.cols .vc}
::: {.col}
Stacking $n = 64$ images prepends a leading batch axis to `X` and pairs it with a 1D integer vector `y`:

@image-classification-dataset-reading-a-minibatch-2

- **Minibatch footprint:** $64 \times 1 \times 28 \times 28 \times 4\text{ B} \approx 200\text{ KB}$ ($\approx 803\text{ KB}$ at $n = 256$).
- **Flattening for linear models:** Reshaping `X` to `(-1, 784)` produces $\mathbf{X} \in \mathbb{R}^{n \times 784}$ in both `NCHW` and `NHWC`.
:::

::: {.col .fig}
@fig:ch03-cls-minibatch-flatten
:::
:::
:::

::: {.slide title="Benchmarking Loader Throughput in Isolation"}
[Systems Bottleneck Analysis]{.kicker}

Before attaching any neural network, always time one complete pass over `train_dataloader()` by itself:

@image-classification-dataset-reading-a-minibatch-3

- **Hard lower bound on epoch time:** Because data loading and model execution form a producer--consumer pipeline, no training epoch can ever run faster than the data loader alone.
:::

::: {.slide title="Diagnosing Data Pipeline Bottlenecks"}
[Producer–Consumer Overlap]{.kicker}

::: {.cols .vc}
::: {.col}
When background CPU workers prefetch batch $b + 1$ while the GPU computes on batch $b$, epoch wall-clock time obeys:

$$T_{\text{epoch}} \;\ge\; \max\!\big(T_{\text{load}},\; T_{\text{compute}}\big)$$

- **Small batch sizes ($n = 1$ or $16$):** Per-batch Python dispatch and IPC overhead dominate (`T_load >> T_compute`), starving the GPU.
- **Worker scaling:** `num_workers=0` blocks serially between steps; once $T_{\text{load}} < T_{\text{compute}}$, extra workers add zero speedup.
:::

::: {.col .fig}
@fig:ch03-cls-producer-consumer
:::
:::
:::

::: {.slide title="Visualizing Labeled Minibatches Before Training"}
[Defensive Data Inspection]{.kicker}

::: {.cols .vc}
::: {.col}
Silent data bugs—shifted label mappings, transposed spatial axes, or scrambled channels—never raise syntax errors. Always plot a labeled gallery first:

@-image-classification-dataset-visualization-2

- **Dropping the singleton channel:** Reshaping `X` to `(-1, *self.resize)` strips the length-$1$ channel axis from either `(n, 1, h, w)` or `(n, h, w, 1)`.
:::

::: {.col .fig}
@fig:ch03-cls-visual-debug
:::
:::
:::

::: {.slide title="What the Garment Silhouettes Predict About Model Errors"}
[Visual Error Preview]{.kicker}

Displaying the first three validation images of each category (one column per class) immediately reveals where a linear template classifier will struggle:

@!image-classification-dataset-inspecting-the-images-1

::: {.d2l-note}
**Silhouette clusters:** Trousers (`1`) and bags (`8`) have unmistakable outlines; sandals (`5`), sneakers (`7`), and ankle boots (`9`) form a footwear cluster; and the four upper-body tops (**T-shirt `0`, pullover `2`, coat `4`, shirt `6`**) share nearly identical boxy silhouettes at $28 \times 28$ resolution.
:::
:::

::: {.slide title="Concept Check: Shapes, Batches & Visual Debugging"}
[Concept Check]{.kicker}

Test your understanding of the `FashionMNIST` pipeline on three practical questions:

1. **Minibatch counts:** When we switch to `batch_size=256` in §{sec_softmax_scratch}, how many minibatches does one training epoch ($60{,}000$ images) produce, and how many images are in the final minibatch?
2. **Visual vs. numerical bugs:** Which of these four bugs will a `data.visualize` gallery expose immediately by eye, and which will it miss: *(a)* labels shifted by $+1$, *(b)* images transposed $(h \leftrightarrow w)$, *(c)* pixels left unnormalized in $[0, 255.0]$, *(d)* every image duplicated twice per epoch?
3. **Upsampling invariance:** If we set `resize=(32, 32)` ($1{,}024$ pixels via bilinear interpolation $\mathbf{x}' = \mathbf{A}\mathbf{x}$), does a linear softmax classifier gain any new expressive power?

::: {.d2l-note .fragment .fade-up}
**Answers:** **(1)** $60{,}000 = 234 \times 256 + 96$, so $235$ minibatches per epoch with $96$ images in the last batch (validation has $39 \times 256 + 16 \to 40$ batches). **(2)** *(a)* and *(b)* are obvious by eye (wrong titles, sideways clothes); *(c)* is hidden because `imshow` auto-rescales `[vmin, vmax]`; *(d)* requires checking `len(dataloader)`. **(3)** No: any linear score $\mathbf{w}'^\top(\mathbf{A}\mathbf{x}) + b = (\mathbf{A}^\top\mathbf{w}')^\top\mathbf{x} + b$ is already a linear score on the original $784$ pixels.
:::
:::

::: {.slide title="Summary: Fashion-MNIST Data Pipeline"}
[Section Summary]{.kicker}

| Pipeline Stage | Specification / Shape | Systems & Modelling Takeaway |
|:--|:--|:--|
| **Dataset splits** | $60{,}000$ train / $10{,}000$ val ($6\text{k} / 1\text{k}$ per class) | Perfectly balanced across $10$ apparel categories; $\approx 188\text{ MB}$ in `float32` |
| **Pixel encoding** | Raw `uint8` $\in [0, 255]$ $\to$ `float32` $\in [0, 1]$ | Dark background $= 0$; rescaling keeps initial linear logits $\mathbf{W}^\top\mathbf{x}$ near $0$ |
| **Minibatch tensor `X`** | `(n, 1, 28, 28)` (channel-first) or `(n, 28, 28, 1)` (channel-last), per the framework | $n = 64 \to 200\text{ KB}$; $n = 256 \to 803\text{ KB}$; flattens to `(n, 784)` for linear layers |
| **Label vector `y`** | 1D integer tensor `(n,)` with values in $\{0, \ldots, 9\}$ | Stores $0$-based class index directly; `text_labels` maps indices to garment names |
| **Epoch iteration** | $938$ batches at $n=64$ ($235$ at $n=256$) | Shuffles training split every epoch; retains partial final batch so no data are dropped |
| **Throughput bound** | $T_{\text{epoch}} \ge \max(T_{\text{load}}, T_{\text{compute}})$ | Always benchmark the standalone loader; tune `batch_size` and worker parallelism |
:::
