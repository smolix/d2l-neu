```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# The Image Classification Dataset
:label:`sec_fashion_mnist`

To develop and compare classifiers we need labeled images, and the choice of
dataset decides what our experiments can show. A good dataset for this book
is hard enough that a better model scores visibly higher, and small enough
that a model trains in minutes inside a notebook.
MNIST :cite:`LeCun.Bottou.Bengio.ea.1998`, 70,000 grayscale images of
handwritten digits with $28 \times 28$ pixels in 10 classes, is small but too
easy. Convolutional networks classify more than 99% of its test digits
correctly, and even classical methods reach about 97%, so models that differ
in kind differ in accuracy by a few points at most.
ImageNet :cite:`Deng.Dong.Socher.ea.2009`, with 1.2 million color images in
1,000 classes, separates models well, but training a network on it takes
hours to days of GPU time.

Fashion-MNIST :cite:`Xiao.Rasul.Vollgraf.2017` lies between the two. It keeps
the format of MNIST, 70,000 grayscale images of $28 \times 28$ pixels in 10
classes, but shows garments instead of digits, and its classes are harder to
tell apart: a shirt can look like a T-shirt, a pullover, or a coat. Linear
classifiers reach about 83% accuracy on it and good convolutional networks
more than 90%, which leaves room to tell models apart. We use it for the rest of
this chapter and for much of the book.

This section loads Fashion-MNIST in each framework, reads it in minibatches,
checks the shape of what the loader delivers, and looks at a few images
before any model touches them. Every later classifier in the book reads its
data through the class we write here.

```{.python .input #image-classification-dataset-the-image-classification-dataset}
%%tab mxnet
%matplotlib inline
import gzip
import os
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
import time
from d2l import jax as d2l
import tensorflow as tf

d2l.use_svg_display()
```

## Loading the Dataset

Fashion-MNIST has 7,000 images in each of its 10 classes: t-shirt, trouser,
pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boot. Its
authors split them into 60,000 training and 10,000 test images, 6,000 and
1,000 per class. Each pixel stores a brightness between 0 and 255, with the
background at 0.

As in :numref:`sec_oo-design`, a `DataModule` holds the data and hands
minibatches to the `Trainer` through its methods `train_dataloader` and
`val_dataloader`. The class `FashionMNIST` downloads the dataset on first use
and keeps both splits in memory. It has two hyperparameters: `batch_size`,
the number of examples per minibatch, and `resize`, the height and width of
the images that the loader delivers. The images keep their native $28 \times 28$ pixels by default;
networks in later chapters that are built for larger inputs ask for
enlarged images instead.

:begin_tab:`pytorch`
The torchvision library downloads Fashion-MNIST and serves it as a dataset
object. Its `transform` argument resizes each image and converts it into a
tensor with values in $[0, 1]$.
:end_tab:

:begin_tab:`mxnet`
Gluon downloads its own copy of Fashion-MNIST, but that copy is split
differently from the published dataset: most of its 10,000 test images are
training images of the published dataset. To train and test on the
published split, we read the published files ourselves.
Each file is compressed with gzip and starts with a short header, 16 bytes
for images and 8 bytes for labels, followed by one byte per pixel or label.
`transform_first` then resizes each image and converts it into an array
with values in $[0, 1]$.
:end_tab:

:begin_tab:`tensorflow, jax`
Keras downloads Fashion-MNIST and returns each split as a pair of NumPy
arrays, images and labels. The loader below converts them.
:end_tab:

```{.python .input #image-classification-dataset-loading-the-dataset-1}
%%tab mxnet
class FashionMNIST(d2l.DataModule):  #@save
    """The Fashion-MNIST dataset."""
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        self.train = gluon.data.ArrayDataset(
            *self.read(train=True)).transform_first(trans)
        self.val = gluon.data.ArrayDataset(
            *self.read(train=False)).transform_first(trans)

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

We load the dataset and count the images in each split.

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

Fashion-MNIST has no validation split of its own, so the 10,000 test images
serve as our validation data: the `Trainer` reports the loss on them after
every epoch. A test set that we consult again and again while we compare
models stops being an unbiased test of the model we finally choose;
:numref:`chap_classification_generalization` measures how much this costs.

The labels are class indices from 0 to 9. To read them, we map each index
to the name of its class.

```{.python .input #image-classification-dataset-loading-the-dataset-4}
@d2l.add_to_class(FashionMNIST)  #@save
def text_labels(self, indices):
    """Return text labels."""
    labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
              'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [labels[int(i)] for i in indices]
```

## Reading Minibatches

The loader for the training split draws the examples in a new random order
in every epoch, so that the minibatches differ from epoch to epoch; the
loader for the validation split reads them in their stored order. Both
convert the pixel values from integers between 0 and 255 into floating-point
numbers between 0 and 1, and resize the images if asked to.

With minibatches of 64, the 60,000 training images make 937 full
minibatches and a last, incomplete one of 32 images
(60,000 = 937 $\times$ 64 + 32). Both loaders keep that last minibatch, so
that every epoch trains on every training image and the validation loss
covers every test image (10,000 = 156 $\times$ 64 + 16). Because the order
changes, a different set of 32 images lands in the short minibatch in each
epoch. Dropping it instead, an option every framework's loader offers, would
give every step the same shape at the price of leaving 32 images out of each
epoch.

:begin_tab:`pytorch`
`DataLoader` reads the dataset in minibatches. Its `num_workers` processes,
4 by default in `DataModule`, load and transform the next minibatches in
parallel while the model works on the current one.
:end_tab:

:begin_tab:`mxnet`
Gluon's `DataLoader` reads the dataset in minibatches. Its `num_workers`
processes, 4 by default in `DataModule`, load and transform the next
minibatches in parallel while the model works on the current one.
:end_tab:

:begin_tab:`tensorflow`
The loader is a `tf.data` pipeline. It adds a channel axis to the images,
scales them to $[0, 1]$, pairs them with the labels, shuffles them with a
buffer that holds all training examples (a uniformly random order, drawn
anew on every pass), groups them into minibatches, and resizes each
minibatch. The `Trainer` traces its compiled training step once for the
full minibatches and once more for the shorter last one.
:end_tab:

:begin_tab:`jax`
JAX has no loader for image datasets, so we build a pipeline with
TensorFlow's `tf.data`: it adds a channel axis to the images, scales them
to $[0, 1]$, pairs them with the labels, shuffles them with a buffer that
holds all training examples (a uniformly random order, drawn anew on every
pass), groups them into minibatches, and resizes each minibatch. The class
`TensorFlowDataLoader` wraps the pipeline: iterating over it yields each
minibatch as NumPy arrays, which JAX functions accept, and `len` gives the
number of minibatches. The `Trainer` compiles its training step once for
the full minibatches and once more for the shorter last one, since JAX
compiles a function anew for each input shape.
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

An image is a three-dimensional tensor: a number of *channels* $c$, one for
each color, and the height $h$ and width $w$ of the image in pixels. Here
$c = 1$ because the images are grayscale; color photographs have $c = 3$.
Frameworks disagree on where the channel axis goes: *channel-first* layout
stores an image as $c \times h \times w$, *channel-last* layout as
$h \times w \times c$. We read one minibatch and look at the shape of its
first image.

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

A minibatch adds an axis for the examples in front, and its labels form a
vector with one entry per example.

```{.python .input #image-classification-dataset-reading-a-minibatch-2}
print(X.shape, X.dtype, y.shape, y.dtype)
```

The images arrive as 32-bit floats and the labels as integers. One minibatch
of images takes $64 \times 784 \times 4$ bytes, about 200 KB, and the whole
training set as floats about 190 MB.

A loader that is slower than the model leaves the processor waiting for data.
We time one pass over the training set without any model.

```{.python .input #image-classification-dataset-reading-a-minibatch-3}
tic = time.time()
for X, y in data.train_dataloader():
    continue
f'{time.time() - tic:.2f} sec'
```

The time depends on the storage, the number of loader workers, the
framework, and the hardware, so the number above holds only for the machine
that ran this notebook. It is the least time that an epoch of training can
take. When we train models in the following sections, compare it with the
time per epoch: if the two are close, loading limits training, and more
loader workers or prefetching of the next minibatches help.

## Inspecting the Images

We look at the data before we train on it. A look catches errors that no
loss curve reveals: labels attached to the wrong images, images flipped or
transposed, a channel axis in the wrong place. The method `visualize` shows
the images of a minibatch, each titled with its label. It calls the library
function `d2l.show_images`, which lays out a list of images in a grid with
optional titles. Matplotlib would draw single-channel images in false
color, so `visualize` sets the gray color map. Since every image has a
single channel, reshaping removes the channel axis in either layout.

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

Here are the first 16 images of the validation set, which the validation
loader always delivers in the same order.

```{.python .input #image-classification-dataset-inspecting-the-images-1}
batch = next(iter(data.val_dataloader()))
data.visualize(batch, nrows=2)
```

The labels match the garments, the garments are upright, and the
background is dark, as stored. Some garments of different classes look
alike: the shirts and the coats above differ in details that
$28 \times 28$ pixels barely resolve. Classes like these are where a
classifier will make most of its mistakes.

We can now load Fashion-MNIST in minibatches in each of the four
frameworks, with the same images and labels in each. To compare classifiers
on it we still need to say how to measure them, which is the subject of the
next section.

## Summary

Fashion-MNIST has 60,000 training and 10,000 test images of garments in 10
classes, grayscale with $28 \times 28$ pixels. It is small enough to train
on in minutes, and hard enough that better models score visibly higher,
unlike MNIST. The class `FashionMNIST` loads the published files, and its
loaders deliver images as floats in $[0, 1]$ and labels as class indices;
the training loader reshuffles every epoch, and both loaders keep the last,
incomplete minibatch. A minibatch of images is a four-dimensional tensor,
with one axis each for the examples, the channels, the height, and the
width. Before training, we look at the images
with their labels, and we check that loading an epoch takes less time than
training on it.

## Exercises

1. [code] **Throughput versus batch size.** Time one full pass of
   `train_dataloader` over the training set for `FashionMNIST` constructed
   with `batch_size` of 1, 16, 64, 256, and 1024, and count the images
   that each pass delivers. Plot throughput (images per second) against
   `batch_size` and explain the shape of the curve in terms of per-batch
   overhead and per-image work.
   :exlabel:`ex_image-classification-dataset-throughput-batch-size`

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

:begin_tab:`pytorch`
3. [code] **Worker ablation.** `get_dataloader` passes the `num_workers`
   attribute, which `DataModule` sets to 4 by default, to
   `torch.utils.data.DataLoader`. For `data = FashionMNIST()`, time one pass
   of `data.train_dataloader()` with `data.num_workers` set to 0, 1, 2, 4,
   and 8, where 0 loads in the main process. State under what conditions
   increasing `num_workers` stops helping.
   :exlabel:`ex_image-classification-dataset-loader-parallelism`
:end_tab:

:begin_tab:`jax`
3. [code] **Parallel loading.** The `tf.data` pipeline in `get_dataloader`
   does not use the `num_workers` attribute of `DataModule`; its
   parallelism is set per stage. Time one pass of `train_dataloader` as
   written, then with `num_parallel_calls=tf.data.AUTOTUNE` in the `map`
   call and a trailing `.prefetch(tf.data.AUTOTUNE)`. State under what
   conditions additional parallelism stops helping.
   :exlabel:`ex_image-classification-dataset-loader-parallelism`
:end_tab:

:begin_tab:`tensorflow`
3. [code] **Parallel loading.** The `tf.data` pipeline in `get_dataloader`
   does not use the `num_workers` attribute of `DataModule`; its
   parallelism is set per stage. Time one pass of `train_dataloader` as
   written, then with `num_parallel_calls=tf.data.AUTOTUNE` in the `map`
   call and a trailing `.prefetch(tf.data.AUTOTUNE)`. State under what
   conditions additional parallelism stops helping.
   :exlabel:`ex_image-classification-dataset-loader-parallelism`
:end_tab:

:begin_tab:`mxnet`
3. [code] **Worker ablation.** `get_dataloader` passes the `num_workers`
   attribute, which `DataModule` sets to 4 by default, to
   `gluon.data.DataLoader`. For `data = FashionMNIST()`, time one pass of
   `data.train_dataloader()` with `data.num_workers` set to 0, 1, 2, 4,
   and 8, where 0 loads in the main process. State under what conditions
   increasing `num_workers` stops helping.
   :exlabel:`ex_image-classification-dataset-loader-parallelism`
:end_tab:

4. [code] **Reshaping versus transposing.** `visualize` removes the
   channel axis with a reshape, which is correct only because $c = 1$.
    1. A minibatch of color images has shape $(n, 3, h, w)$ in channel-first
       layout. Show that reshaping it to $(n, h, w, 3)$ does not move the
       channel axis to the end, by computing which stored value lands at
       position $(0, 0, 0, 1)$ of the result.
    1. Write the transposition that converts it correctly, apply both
       versions to a minibatch of 8 random arrays of shape $(3, 4, 5)$, and
       check which one agrees with indexing the original array.

1. [code] **Upsampling and linear models.** ● Treat the resizing from
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

1. [extended] **Loader or compute.** ● Combine the two sweeps of problems
   :exref:`ex_image-classification-dataset-throughput-batch-size` and
   :exref:`ex_image-classification-dataset-loader-parallelism` into a grid
   of batch sizes and loader-parallelism settings, and measure loader
   throughput in images per second at each point. After
   :numref:`sec_softmax_scratch`, measure the throughput of a full training
   step of `SoftmaxRegressionScratch` (forward pass, loss, gradient, and
   parameter update) on a minibatch held in memory, at the same batch sizes.
   Draw a heatmap of the ratio of loader throughput to training throughput
   over the grid. Identify the region of the grid in which loading is the
   bottleneck, and state whether a crossover exists on your hardware.

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

The Image Classification Dataset<br>**Fashion-MNIST · minibatches · a look at the images**
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
- count the minibatches of an epoch, including the short last one;
- time the loader and compare it with training;
- check images and labels by eye before training.

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
| **The Image Classification Dataset** | **Fashion-MNIST in minibatches** |
| Measuring Classifiers | accuracy, costs, calibration, the `Classifier` class |
| Implementation from Scratch | softmax regression by hand |
| Concise Implementation | framework layers, a stable loss |
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
| t-shirt | trouser | pullover | dress | coat | sandal | shirt | sneaker | bag | ankle boot |

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

- $60{,}000 = 937 \times 64 + 32$: 938 minibatches, the last of 32 images,
  a different 32 in every epoch.
- Validation: $10{,}000 = 156 \times 64 + 16$, so 157 minibatches.
- Compiled steps (TensorFlow, JAX) trace once more for the short shape.
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

The first 16 validation images, in the order the loader always delivers
them:

@!image-classification-dataset-inspecting-the-images-1

::: {.d2l-note}
Similar classes (shirt, coat, pullover, t-shirt) are where a classifier
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
| epoch | 938 minibatches, the last of 32 images, different each time |
| checks | loader time against training time; a gallery with labels |

**Next:** accuracy, and the base class that reports it with the loss.
:::
