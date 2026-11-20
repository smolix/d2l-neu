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

In deep learning frameworks, each image in a minibatch of size $n$ with $c$ channels ($c = 1$ for grayscale, $c = 3$ for RGB color), height $h$, and width $w$ is stored either in *channel-first* layout $(n, c, h, w)$ (PyTorch and MXNet) or *channel-last* layout $(n, h, w, c)$ (TensorFlow and JAX). Our `get_dataloader(train)` method wraps each framework's native data loader, reshuffling the training split on every epoch and keeping the final partial minibatch ($60{,}000 = 937 \times 64 + 32$) so every image is visited once per epoch:

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
- **`FashionMNIST` data module:** Delivers float32 image minibatches in $[0, 1]$ of shape $(n, 1, h, w)$ (channel-first) or $(n, h, w, 1)$ (channel-last) alongside integer class labels, with built-in resizing, worker parallelism, and grayscale visualization.

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
