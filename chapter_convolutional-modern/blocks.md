```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Blocks, Bottlenecks, and Branches: VGG, NiN, GoogLeNet
:label:`sec_blocks`

AlexNet (:numref:`sec_alexnet`) specifies each of its eight layers by hand:
an $11 \times 11$ kernel here, 384 channels there, a pooling layer after the
first, second, and fifth convolutions. That works for eight layers, but a
designer who wants twenty needs a rule for choosing them. AlexNet also puts
92% of its 47 million parameters into its fully connected head, where they
do 5% of the arithmetic and have to be held back from memorizing the training
set by dropout. Three networks of 2013 and 2014 answered these two problems
with ideas that every later convolutional network uses.

VGG :cite:`Simonyan.Zisserman.2014` builds the network from one repeated
*block*, a few $3 \times 3$ convolutions followed by pooling, so that a whole
network is a short list of block sizes. Network in network (NiN)
:cite:`Lin.Chen.Yan.2013` removes the fully connected head: $1 \times 1$
convolutions mix the channels at every position, and an average over all
positions turns the last feature map into class scores. GoogLeNet
:cite:`Szegedy.Liu.Jia.ea.2015` runs convolutions of several sizes in
parallel inside one block and makes the expensive ones affordable by first
reducing the number of channels with $1 \times 1$ convolutions. We build each
network, count its parameters and multiply-adds as we did for AlexNet, and
train it on Fashion-MNIST under the same procedure, so that the four networks
can be compared on one table.

```{.python .input #blocks-imports}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np as mnp, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #blocks-imports}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input #blocks-imports}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #blocks-imports}
%%tab jax
from d2l import jax as d2l
from flax import nnx
from jax import numpy as jnp
import optax
```

## VGG: Blocks as the Unit of Design
:label:`sec_vgg`

VGG replaces individually chosen layers with one block, repeated at growing
channel counts. The block answers two questions at once: how to make a
network deeper without losing its resolution too soon, and how large its
kernels should be.

### Stacks of $3 \times 3$ Convolutions
:label:`subsec_vgg-blocks`

If every convolution were followed by pooling with stride 2, an input of
width $d$ would permit at most $\lfloor\log_2 d\rfloor$ convolutions before
its spatial extent collapses: for a 224-pixel image, 224, 112, 56, 28, 14, 7,
3, 1, and the network is over after seven halvings. To go deeper, a network
must place several convolutions at each resolution, with stride 1 and
padding that keeps the size.

VGG places small kernels there. Two stacked $3 \times 3$ convolutions see a
$5 \times 5$ window of their input, the same as one $5 \times 5$
convolution, but with $c$ input and $c$ output channels they have
$2 \cdot 9c^2 = 18c^2$ weights instead of $25c^2$. Three of them see
$7 \times 7$ with $27c^2$ weights instead of $49c^2$, and they apply three
nonlinearities instead of one. :citet:`Simonyan.Zisserman.2014` trained
networks of 11 to 19 layers built this way and found that the deeper ones
classified ImageNet more accurately. Stacks of $3 \times 3$ convolutions
remained the default for most of the following decade, and GPU libraries implement them with
specialized algorithms :cite:`lavin2016fast`.

A *VGG block* is therefore a sequence of `num_convs` $3 \times 3$
convolutions with padding 1, each followed by a ReLU, and a $2 \times 2$
max-pooling with stride 2 at the end. The convolutions keep the height and
the width; the pooling halves them.

```{.python .input #vgg-vgg-blocks  n=2}
%%tab mxnet
def vgg_block(num_convs, num_channels):
    blk = nn.HybridSequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1,
                          activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input #vgg-vgg-blocks  n=3}
%%tab pytorch
def vgg_block(num_convs, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.LazyConv2d(out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)
```

```{.python .input #vgg-vgg-blocks  n=4}
%%tab tensorflow
def he():  # A new He-normal initializer for each layer
    return dict(kernel_initializer=tf.keras.initializers.VarianceScaling(
        2.0, 'fan_in', 'untruncated_normal'))

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                       padding='same', activation='relu',
                                       **he()))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

```{.python .input #vgg-vgg-blocks}
%%tab jax
def vgg_block(num_convs, in_channels, out_channels, init):
    layers = []
    for _ in range(num_convs):
        layers.append(nnx.Conv(in_channels, out_channels, kernel_size=(3, 3),
                               padding=(1, 1), **init))
        layers.append(nnx.relu)
        in_channels = out_channels
    layers.append(max_pool)
    return nnx.Sequential(*layers)

def max_pool(x):  # 2x2 windows, stride 2
    return nnx.max_pool(x, window_shape=(2, 2), strides=(2, 2))
```

### The VGG Network
:label:`subsec_vgg-network`

A VGG network is a sequence of such blocks followed by the fully connected
head of AlexNet: two hidden layers of 4096 units with dropout, and an output
layer. :numref:`fig_vgg` places VGG-11 beside AlexNet. Every block halves the
resolution, so five blocks take a $224 \times 224$ input to $7 \times 7$,
and the number of channels doubles from block to block up to 512.

![AlexNet (left) and VGG-11 (right) for single-channel $224 \times 224$ images and ten classes; the input is at the bottom. AlexNet's layers are chosen one by one; VGG repeats one block, several $3 \times 3$ convolutions followed by max-pooling, at growing channel counts.](../img/arch-vgg.svg)
:label:`fig_vgg`

Since a block is fixed by two numbers, the number of convolutions and their
output channels, the convolutional part of the network is fixed by a list of
such pairs. The class `VGG` takes this list as its argument `arch` and builds
one block per entry. VGG is therefore a *family* of networks: VGG-11, VGG-16,
and VGG-19 differ only in `arch`, and :citet:`Simonyan.Zisserman.2014`
compared six members of the family. We initialize all networks of this
section as AlexNet in :numref:`sec_alexnet`: He-normal weights and zero
biases.

:begin_tab:`mxnet`
As for AlexNet, the networks are built from `HybridSequential` containers
and `HybridBlock`s and compiled into one graph by `hybridize()`. Gluon's
`Dense` layer flattens its input, so the head needs no flattening layer.
:end_tab:

:begin_tab:`pytorch`
As for AlexNet, the `Trainer` calls `weight_init` on every layer once the
first minibatch has fixed the shapes of the lazy layers.
:end_tab:

:begin_tab:`tensorflow`
`he()` gives every layer its own initializer, since an unseeded Keras
initializer returns the same values on every call.
:end_tab:

:begin_tab:`jax`
The dictionary `init` carries the weight initializer and the random number
streams to every layer.
:end_tab:

```{.python .input #vgg-vgg-network-1  n=5}
%%tab mxnet
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.HybridSequential()
        for (num_convs, num_channels) in arch:
            self.net.add(vgg_block(num_convs, num_channels))
        self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                     nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                     nn.Dense(num_classes))
        self.net.initialize(init.MSRAPrelu('in', 0))
        self.net.hybridize()
```

```{.python .input #vgg-vgg-network-1  n=5}
%%tab pytorch
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        conv_blks = [vgg_block(num_convs, out_channels)
                     for (num_convs, out_channels) in arch]
        self.net = nn.Sequential(
            *conv_blks, nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(0.5),
            nn.LazyLinear(num_classes))

    def weight_init(self, module):  # He-normal weights, zero biases
        if type(module) in (nn.Linear, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)
```

```{.python .input #vgg-vgg-network-1  n=5}
%%tab tensorflow
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        layers = tf.keras.layers
        self.net = tf.keras.models.Sequential(
            [vgg_block(num_convs, num_channels)
             for (num_convs, num_channels) in arch] +
            [layers.Flatten(),
             layers.Dense(4096, activation='relu', **he()),
             layers.Dropout(0.5),
             layers.Dense(4096, activation='relu', **he()),
             layers.Dropout(0.5),
             layers.Dense(num_classes, **he())])
```

```{.python .input #vgg-vgg-network-1  n=5}
%%tab jax
class VGG(d2l.Classifier):
    def __init__(self, arch, lr=0.1, num_steps=4690, num_classes=10,
                 input_shape=(224, 224, 1), rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = (nnx.Rngs(params=d2l.get_key(), dropout=d2l.get_key())
                if rngs is None else rngs)
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        init = dict(kernel_init=he, rngs=rngs)
        conv_blks, in_channels = [], input_shape[-1]
        for num_convs, out_channels in arch:
            conv_blks.append(vgg_block(num_convs, in_channels, out_channels,
                                       init))
            in_channels = out_channels
        # Each block halves the height and the width
        height, width = (n // 2 ** len(arch) for n in input_shape[:2])
        self.net = nnx.Sequential(
            *conv_blks,
            flatten,
            nnx.Linear(height * width * in_channels, 4096, **init), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, 4096, **init), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, num_classes, **init))

def flatten(x):  # One vector per example
    return x.reshape((x.shape[0], -1))
```

:begin_tab:`jax`
`VGG` takes the image size as `input_shape`, since NNX layers need their
input sizes when they are built: the first fully connected layer reads
$h w c$ numbers for a final feature map of $h \times w$ pixels and $c$
channels.
:end_tab:

VGG-11, the smallest network of the paper, has five blocks: one convolution
each in the first two and two in the other three, with 64, 128, 256, 512,
and 512 output channels. Its eight convolutional and three fully connected
layers give it its name. We check the shapes on one $224 \times 224$ image.

```{.python .input #vgg-vgg-network-2  n=6}
%%tab mxnet, pytorch
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 1, 224, 224))
```

```{.python .input #vgg-vgg-network-2  n=7}
%%tab tensorflow, jax
VGG(arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))).layer_summary(
    (1, 224, 224, 1))
```

Each block halves the height and the width, from 224 to 7, and the last
block hands $512 \times 7 \times 7 = 25{,}088$ numbers to the fully connected
head. Counted as for AlexNet, VGG-11 has 128.8 million parameters and
performs 7.5 billion multiply-adds per image. Its eight convolutional layers
hold 9.2 million of the parameters and perform 98% of the arithmetic, eight
times AlexNet's convolutional arithmetic. Its head holds 119.6 million
parameters, 93% of the total; the first fully connected layer alone,
$25{,}088 \times 4096$, holds 103 million, about 410 MB in 32-bit floating
point.

### Training

VGG-11 performs eight times the arithmetic of AlexNet, so we train a
narrower member of the family on Fashion-MNIST, with a quarter of the
channels in every block: 16, 32, 64, 128, and 128. It performs 512 million
multiply-adds per image, 470 million of them in the convolutions: about half
of AlexNet's 938 million. Everything else
follows :numref:`sec_alexnet`: images upsampled to $224 \times 224$,
minibatches of 128, and 10 epochs of SGD whose learning rate rises linearly
to a peak of 0.03 over the first epoch and then decays to zero along the
half-cosine :eqref:`eq_cosine_warmup`, stepped after every minibatch. VGG
gets the same `configure_optimizers` as AlexNet.

```{.python .input #blocks-training-1}
%%tab mxnet
@d2l.add_to_class(VGG)
def configure_optimizers(self):
    schedule = lambda t: self.lr * d2l.cosine_warmup(  # MXNet counts from 1
        t - 1, self.num_steps, self.num_steps // 10)
    return gluon.Trainer(self.collect_params(), 'sgd', {
        'learning_rate': None, 'lr_scheduler': schedule})
```

```{.python .input #blocks-training-1}
%%tab pytorch
@d2l.add_to_class(VGG)
def configure_optimizers(self):
    optim = torch.optim.SGD(self.parameters(), lr=self.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(optim, lambda t: (
        d2l.cosine_warmup(t, self.num_steps, self.num_steps // 10)))
    return optim, schedule  # The Trainer steps it after every minibatch
```

```{.python .input #blocks-training-1}
%%tab tensorflow
@d2l.add_to_class(VGG)
def configure_optimizers(self):
    schedule = lambda: self.lr * d2l.cosine_warmup(
        optim.iterations, self.num_steps, self.num_steps // 10)
    optim = tf.keras.optimizers.SGD(schedule)
    return optim
```

```{.python .input #blocks-training-1}
%%tab jax
@d2l.add_to_class(VGG)
def configure_optimizers(self):
    return optax.sgd(lambda t: self.lr * d2l.cosine_warmup(
        t, self.num_steps, self.num_steps // 10))
```

```{.python .input #vgg-training  n=8}
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
model = VGG(arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128)), lr=0.03,
            num_steps=10 * len(data.train_dataloader()))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
val_acc = {'VGG': trainer.history['val_acc']}  # for the comparison below
```

The narrow VGG reaches a validation accuracy of about 92%, half a point to a
point above AlexNet with half of its arithmetic, and the training and
validation losses stay close. Narrowing the blocks shrank the convolutions
to 0.6 million parameters, but the head still maps
$128 \times 7 \times 7 = 6272$ numbers to 4096 units and holds 42.5 million
of the network's 43.1 million parameters: 99%. Blocks organized the body of the network; the head is
untouched.

## NiN: $1 \times 1$ Convolutions and Global Average Pooling
:label:`sec_nin`

The fully connected head is expensive because it connects every position of
the last feature map to every hidden unit, and it needs those connections
only to turn a feature map into one score per class. Inserting fully
connected layers earlier, to let the network compute more per position,
would destroy the spatial structure and cost even more. NiN solves both
problems with layers that the reader has met in :numref:`subsec_1x1`.

### NiN Blocks

A $1 \times 1$ convolution is a fully connected layer applied independently
at every pixel: it maps the $c_\textrm{i}$ channels at a position to
$c_\textrm{o}$ channels with the same $c_\textrm{i} \times c_\textrm{o}$
weight matrix at every position, and leaves the spatial structure untouched.
A *NiN block* follows an ordinary convolution with two $1 \times 1$
convolutions, each with a ReLU. The two together form a small MLP with one
hidden layer that runs at every pixel, hence the name *network in network*.
:numref:`fig_nin` compares the head of VGG with the end of NiN.

![The head of VGG (left) and the end of NiN (right), both for ten classes; the input is at the bottom. VGG flattens its last feature map into fully connected layers. After dropout, NiN's last block, a $3 \times 3$ convolution and two $1 \times 1$ convolutions with no ReLU after the last one, emits one channel per class, and global average pooling turns each channel into one score.](../img/arch-nin.svg)
:label:`fig_nin`

```{.python .input #nin-nin-blocks}
%%tab mxnet
def nin_block(num_channels, kernel_size, strides, padding, final=False):
    blk = nn.HybridSequential()
    blk.add(nn.Conv2D(num_channels, kernel_size, strides, padding,
                      activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
            nn.Conv2D(num_channels, kernel_size=1,
                      activation=None if final else 'relu'))
    return blk
```

```{.python .input #nin-nin-blocks}
%%tab pytorch
def nin_block(out_channels, kernel_size, strides, padding, final=False):
    layers = [
        nn.LazyConv2d(out_channels, kernel_size, strides, padding), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1), nn.ReLU(),
        nn.LazyConv2d(out_channels, kernel_size=1)]
    return nn.Sequential(*layers, *([] if final else [nn.ReLU()]))
```

```{.python .input #nin-nin-blocks}
%%tab tensorflow
def nin_block(out_channels, kernel_size, strides, padding, final=False):
    layers = tf.keras.layers
    return tf.keras.models.Sequential([
        layers.Conv2D(out_channels, kernel_size, strides=strides,
                      padding=padding, activation='relu', **he()),
        layers.Conv2D(out_channels, 1, activation='relu', **he()),
        layers.Conv2D(out_channels, 1, activation=None if final else 'relu',
                      **he())])
```

```{.python .input #nin-nin-blocks}
%%tab jax
def nin_block(in_channels, out_channels, kernel_size, strides, padding,
              init, final=False):
    layers = [
        nnx.Conv(in_channels, out_channels, kernel_size, strides=strides,
                 padding=padding, **init), nnx.relu,
        nnx.Conv(out_channels, out_channels, kernel_size=(1, 1), **init),
        nnx.relu,
        nnx.Conv(out_channels, out_channels, kernel_size=(1, 1), **init)]
    return nnx.Sequential(*layers, *([] if final else [nnx.relu]))
```

### Global Average Pooling

NiN uses the kernel sizes and channel counts of AlexNet's first layers: blocks
with $11 \times 11$, $5 \times 5$, and $3 \times 3$ convolutions and 96, 256,
and 384 channels, each followed by $3 \times 3$ max-pooling with stride 2.
Then it differs. Its last block has as many output channels as there are
classes, ten for Fashion-MNIST, and *global average pooling* averages each of
these channels over all positions. The ten averages are the logits. The head
therefore has no parameters at all: the last block must learn to make the
channel of the correct class large on average over the image. Its last
$1 \times 1$ convolution outputs these class scores directly, without a ReLU,
and `nin_block` takes the flag `final` to drop it. A ReLU there would switch a
class off for good: once a class's pre-activations are negative at every
position of every image, its logit is 0, its gradient is zero, and training
can never bring it back. Our last block is a simplification of the original
network, whose last block stays wide, with 1,024 channels on ImageNet and 192
on CIFAR-10, until its final $1 \times 1$ convolution maps them to one channel
per class; ours narrows the whole block to ten channels.

Global average pooling has a second property. A convolution commutes with
translations, so shifting the input shifts each feature map, and an average
over all positions does not change when its entries are permuted. Up to the
borders of the image and the strides of the network, the scores therefore do
not depend on where in the image an object appears. A fully connected head,
by contrast, has a separate weight for every position.

```{.python .input #nin-nin-model-1}
%%tab mxnet
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.HybridSequential()
        self.net.add(
            nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2D(pool_size=3, strides=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1,
                      final=True),
            nn.GlobalAvgPool2D(),
            nn.Flatten())
        self.net.initialize(init.MSRAPrelu('in', 0))
        self.net.hybridize()
```

```{.python .input #nin-nin-model-1}
%%tab pytorch
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nin_block(96, kernel_size=11, strides=4, padding=0),
            nn.MaxPool2d(3, stride=2),
            nin_block(256, kernel_size=5, strides=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            nin_block(384, kernel_size=3, strides=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding=1,
                      final=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten())

    weight_init = VGG.weight_init  # He-normal weights, zero biases
```

```{.python .input #nin-nin-model-1}
%%tab tensorflow
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            nin_block(96, kernel_size=11, strides=4, padding='valid'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            nin_block(256, kernel_size=5, strides=1, padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            nin_block(384, kernel_size=3, strides=1, padding='same'),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
            tf.keras.layers.Dropout(0.5),
            nin_block(num_classes, kernel_size=3, strides=1, padding='same',
                      final=True),
            tf.keras.layers.GlobalAvgPool2D()])
```

```{.python .input #nin-nin-model-1}
%%tab jax
class NiN(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = (nnx.Rngs(params=d2l.get_key(), dropout=d2l.get_key())
                if rngs is None else rngs)
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        init = dict(kernel_init=he, rngs=rngs)

        def max_pool(x):  # 3x3 windows, stride 2
            return nnx.max_pool(x, (3, 3), strides=(2, 2))

        self.net = nnx.Sequential(
            nin_block(1, 96, (11, 11), (4, 4), (0, 0), init), max_pool,
            nin_block(96, 256, (5, 5), (1, 1), (2, 2), init), max_pool,
            nin_block(256, 384, (3, 3), (1, 1), (1, 1), init), max_pool,
            nnx.Dropout(0.5, rngs=rngs),
            nin_block(384, num_classes, (3, 3), (1, 1), (1, 1), init,
                      final=True),
            global_avg_pool)

def global_avg_pool(x):  # Average over height and width (NHWC)
    return x.mean(axis=(1, 2))
```

We check the shapes on one $224 \times 224$ image.

```{.python .input #nin-nin-model-2}
%%tab mxnet, pytorch
NiN().layer_summary((1, 1, 224, 224))
```

```{.python .input #nin-nin-model-2}
%%tab tensorflow, jax
NiN().layer_summary((1, 224, 224, 1))
```

The resolution falls from 224 to 54, 26, 12, and 5, as in AlexNet, and the
last block emits ten channels of $5 \times 5$, which the pooling reduces to
ten numbers. NiN has 2.0 million parameters, 4% of AlexNet's, and performs
762 million multiply-adds per image, a little less than AlexNet's 938
million: the $1 \times 1$ convolutions add arithmetic to the body while the
head disappears.

### Training

We train NiN with the procedure of AlexNet and VGG, so it takes VGG's
`configure_optimizers`: images upsampled to $224 \times 224$, minibatches of
128, and 10 epochs of SGD whose learning rate warms up to a peak of 0.03
over the first epoch and then decays to zero. Of the peaks we tried, 0.03 is
the largest at which NiN trains reliably: at 0.05 an occasional run stays
near chance for several epochs, and at 0.1 some runs diverge.

```{.python .input #nin-training}
NiN.configure_optimizers = VGG.configure_optimizers  # AlexNet's schedule
data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
model = NiN(lr=0.03, num_steps=10 * len(data.train_dataloader()))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
val_acc['NiN'] = trainer.history['val_acc']
```

NiN ends at a validation accuracy of 84–88% after the same 10 epochs, a few
points below AlexNet and VGG, with 4% of AlexNet's parameters. Its training
loss ends between 0.3 and 0.45, against about 0.22 for AlexNet, and its
validation loss stays close to it: in 10 epochs NiN fits the training data
less well, and it does not overfit.

NiN is also the network that depends most on the learning-rate schedule.
Trained for 10 epochs at a constant rate of 0.01, it is still improving at
the end, and where a run ends, anywhere between about 78% and 87%, depends
on how quickly it leaves its first plateau. The warmup allows a peak three
times higher, which lets a slow start catch up, and the decay removes the
drops of single late epochs.

The two ideas that NiN introduced outlived the network: $1 \times 1$
convolutions for mixing channels, and global average pooling as the head of
a convolutional network.

## GoogLeNet: Multi-Branch Blocks
:label:`sec_googlenet`

Which kernel size should a layer use? A $1 \times 1$ convolution sees one
position, a $5 \times 5$ convolution sees a neighborhood, and objects in
photographs come at every size. GoogLeNet, which won the ImageNet challenge
of 2014 with a top-5 error of 6.7%, does not choose: its *Inception block*
applies several kernel sizes in parallel to the same input and concatenates
their outputs. To keep the large kernels affordable it borrows NiN's
$1 \times 1$ convolution and uses it to reduce the number of channels before
each of them.

### The Inception Block

The Inception block of :numref:`fig_inception` has four parallel branches.
The first applies a $1 \times 1$ convolution. The second and third apply a
$1 \times 1$ convolution that reduces the number of channels, followed by a
$3 \times 3$ or a $5 \times 5$ convolution. The fourth applies $3 \times 3$
max-pooling with stride 1, followed by a $1 \times 1$ convolution. Every
branch pads its input so that its output keeps the height and the width, and
the block concatenates the four outputs along the channel dimension. The
numbers of output channels of the branches are the block's hyperparameters:
they decide how much of the block's capacity goes to each kernel size. The
name of the block comes from the film *Inception* and its line "we need to go
deeper".

![The Inception block; the input is at the bottom. Four parallel branches read the same input with different kernel sizes; $1 \times 1$ convolutions reduce the number of channels before the $3 \times 3$ and $5 \times 5$ convolutions; the branch outputs are concatenated along the channel dimension. The shapes are those of the first Inception block of GoogLeNet on a $224 \times 224$ input.](../img/arch-inception-block.svg)
:label:`fig_inception`

```{.python .input #googlenet-inception-blocks}
%%tab mxnet
class Inception(nn.HybridBlock):
    # c1--c4 are the numbers of output channels of the four branches
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        self.b2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.b2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1,
                              activation='relu')
        self.b3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.b3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2,
                              activation='relu')
        self.b4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.b4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return mnp.concatenate((b1, b2, b3, b4), axis=1)
```

```{.python .input #googlenet-inception-blocks}
%%tab pytorch
class Inception(nn.Module):
    # c1--c4 are the numbers of output channels of the four branches
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = nn.LazyConv2d(c1, kernel_size=1)
        self.b2_1 = nn.LazyConv2d(c2[0], kernel_size=1)
        self.b2_2 = nn.LazyConv2d(c2[1], kernel_size=3, padding=1)
        self.b3_1 = nn.LazyConv2d(c3[0], kernel_size=1)
        self.b3_2 = nn.LazyConv2d(c3[1], kernel_size=5, padding=2)
        self.b4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.b4_2 = nn.LazyConv2d(c4, kernel_size=1)

    def forward(self, x):
        b1 = F.relu(self.b1_1(x))
        b2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b3 = F.relu(self.b3_2(F.relu(self.b3_1(x))))
        b4 = F.relu(self.b4_2(self.b4_1(x)))
        return torch.cat((b1, b2, b3, b4), dim=1)
```

```{.python .input #googlenet-inception-blocks}
%%tab tensorflow
class Inception(tf.keras.Model):
    # c1--c4 are the numbers of output channels of the four branches
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        def Conv2D(*args, **kwargs):  # ReLU, He-normal initializer
            return tf.keras.layers.Conv2D(*args, activation='relu',
                                          **kwargs, **he())
        self.b1_1 = Conv2D(c1, 1)
        self.b2_1 = Conv2D(c2[0], 1)
        self.b2_2 = Conv2D(c2[1], 3, padding='same')
        self.b3_1 = Conv2D(c3[0], 1)
        self.b3_2 = Conv2D(c3[1], 5, padding='same')
        self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.b4_2 = Conv2D(c4, 1)

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.concat((b1, b2, b3, b4), axis=-1)
```

```{.python .input #googlenet-inception-blocks}
%%tab jax
class Inception(nnx.Module):
    # c1--c4 are the numbers of output channels of the four branches
    def __init__(self, in_channels, c1, c2, c3, c4, init):
        self.b1_1 = nnx.Conv(in_channels, c1, kernel_size=(1, 1), **init)
        self.b2_1 = nnx.Conv(in_channels, c2[0], kernel_size=(1, 1), **init)
        self.b2_2 = nnx.Conv(c2[0], c2[1], kernel_size=(3, 3),
                             padding='SAME', **init)
        self.b3_1 = nnx.Conv(in_channels, c3[0], kernel_size=(1, 1), **init)
        self.b3_2 = nnx.Conv(c3[0], c3[1], kernel_size=(5, 5),
                             padding='SAME', **init)
        self.b4_2 = nnx.Conv(in_channels, c4, kernel_size=(1, 1), **init)

    def __call__(self, x):
        b1 = nnx.relu(self.b1_1(x))
        b2 = nnx.relu(self.b2_2(nnx.relu(self.b2_1(x))))
        b3 = nnx.relu(self.b3_2(nnx.relu(self.b3_1(x))))
        pooled = nnx.max_pool(x, window_shape=(3, 3), strides=(1, 1),
                              padding='SAME')
        b4 = nnx.relu(self.b4_2(pooled))
        return jnp.concatenate((b1, b2, b3, b4), axis=-1)
```

The first Inception block of GoogLeNet shows the role of the $1 \times 1$
reductions. On a $224 \times 224$ input it receives 192 channels at
$28 \times 28$ resolution, the shapes annotated in :numref:`fig_inception`.
With branch outputs $c_1 = 64$, $c_2 = (96, 128)$, $c_3 = (16, 32)$, and
$c_4 = 32$, it emits $64 + 128 + 32 + 32 = 256$ channels at the same
$28 \times 28$ resolution. A $5 \times 5$ convolution straight from 192 to
32 channels would need $25 \cdot 192 \cdot 32 = 153{,}600$ weights. Reducing
to 16 channels first costs $192 \cdot 16 = 3{,}072$ weights for the
reduction and $25 \cdot 16 \cdot 32 = 12{,}800$ for the convolution, 15,872
in total: a tenth. The number of multiply-adds falls by the same factor,
since every weight is used at all $28 \times 28$ positions. Reducing the
channels before an expensive convolution recurs throughout this chapter.

### Stem, Body, and Head

GoogLeNet has three parts, an organization that most later vision networks
share. The *stem* processes the raw image with a few ordinary convolutions
and reduces its resolution: a $7 \times 7$ convolution with stride 2 and 64
channels, max-pooling, a $1 \times 1$ convolution, a $3 \times 3$
convolution with 192 channels, and max-pooling again, each pooling with a
$3 \times 3$ window and stride 2. The *body* consists of nine Inception
blocks in three groups of two, five, and two, with max-pooling between the
groups. The *head* is NiN's: global average pooling, followed by a single
fully connected layer that maps the channels to the classes.

The channel allocations of the nine blocks come from the paper, which chose
them by hand. As for VGG, we store them as data, one tuple per block and one
tuple of blocks per group:

```{.python .input #blocks-stem-body-and-head-1}
googlenet_arch = (
    ((64, (96, 128), (16, 32), 32), (128, (128, 192), (32, 96), 64)),
    ((192, (96, 208), (16, 48), 64), (160, (112, 224), (24, 64), 64),
     (128, (128, 256), (24, 64), 64), (112, (144, 288), (32, 64), 64),
     (256, (160, 320), (32, 128), 128)),
    ((256, (160, 320), (32, 128), 128), (384, (192, 384), (48, 128), 128)))
```

`GoogleNet` takes this tuple as its argument `arch`, as `VGG` does, and
composes the stem, the three groups with pooling in between, and the head.

```{.python .input #blocks-stem-body-and-head-2}
%%tab mxnet
class GoogleNet(d2l.Classifier):
    def __init__(self, arch=googlenet_arch, lr=0.1, num_steps=4690,
                 num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        pool = lambda: nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        stem = nn.HybridSequential()
        stem.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                           activation='relu'), pool(),
                 nn.Conv2D(64, kernel_size=1, activation='relu'),
                 nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
                 pool())
        body = []
        for group in arch:
            blk = nn.HybridSequential()
            blk.add(*[Inception(*c) for c in group])
            body.append(blk)
        self.net = nn.HybridSequential()
        self.net.add(stem, body[0], pool(), body[1], pool(), body[2],
                     nn.GlobalAvgPool2D(), nn.Dense(num_classes))
        self.net.initialize(init.MSRAPrelu('in', 0))
        self.net.hybridize()
```

```{.python .input #blocks-stem-body-and-head-2}
%%tab pytorch
class GoogleNet(d2l.Classifier):
    def __init__(self, arch=googlenet_arch, lr=0.1, num_steps=4690,
                 num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        pool = lambda: nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stem = nn.Sequential(
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3), nn.ReLU(),
            pool(),
            nn.LazyConv2d(64, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(192, kernel_size=3, padding=1), nn.ReLU(), pool())
        body = [nn.Sequential(*[Inception(*c) for c in group])
                for group in arch]
        head = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                             nn.LazyLinear(num_classes))
        self.net = nn.Sequential(stem, body[0], pool(), body[1], pool(),
                                 body[2], head)

    weight_init = VGG.weight_init  # He-normal weights, zero biases
```

```{.python .input #blocks-stem-body-and-head-2}
%%tab tensorflow
class GoogleNet(d2l.Classifier):
    def __init__(self, arch=googlenet_arch, lr=0.1, num_steps=4690,
                 num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        layers = tf.keras.layers
        # Pad 1 on every side: 'same' pads unevenly at stride 2
        pool = lambda: tf.keras.Sequential([layers.ZeroPadding2D(1),
                                            layers.MaxPool2D(3, strides=2)])
        stem = tf.keras.Sequential([
            layers.ZeroPadding2D(3),
            layers.Conv2D(64, 7, strides=2, activation='relu', **he()),
            pool(),
            layers.Conv2D(64, 1, activation='relu', **he()),
            layers.Conv2D(192, 3, padding='same', activation='relu', **he()),
            pool()])
        body = [tf.keras.Sequential([Inception(*c) for c in group])
                for group in arch]
        head = tf.keras.Sequential([layers.GlobalAvgPool2D(),
                                    layers.Dense(num_classes, **he())])
        self.net = tf.keras.Sequential([stem, body[0], pool(), body[1],
                                        pool(), body[2], head])
```

```{.python .input #blocks-stem-body-and-head-2}
%%tab jax
class GoogleNet(d2l.Classifier):
    def __init__(self, arch=googlenet_arch, lr=0.1, num_steps=4690,
                 num_classes=10, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        init = dict(kernel_init=he, rngs=rngs)

        def pool(x):  # 3x3 windows, stride 2; 'SAME' would pad unevenly
            return nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2),
                                padding=((1, 1), (1, 1)))

        stem = nnx.Sequential(
            nnx.Conv(1, 64, kernel_size=(7, 7), strides=(2, 2), padding=3,
                     **init), nnx.relu, pool,
            nnx.Conv(64, 64, kernel_size=(1, 1), **init), nnx.relu,
            nnx.Conv(64, 192, kernel_size=(3, 3), padding='SAME', **init),
            nnx.relu, pool)
        body, in_channels = [], 192
        for group in arch:
            blocks = []
            for c1, c2, c3, c4 in group:
                blocks.append(Inception(in_channels, c1, c2, c3, c4, init))
                in_channels = c1 + c2[1] + c3[1] + c4
            body.append(nnx.Sequential(*blocks))
        head = nnx.Sequential(global_avg_pool,
                              nnx.Linear(in_channels, num_classes, **init))
        self.net = nnx.Sequential(stem, body[0], pool, body[1], pool,
                                  body[2], head)
```

GoogLeNet has five stride-2 steps, three in the stem (the first
convolution and two poolings) and two between the groups, so it reduces the
resolution by a factor of 32. On a $96 \times 96$ input, a resolution at which Fashion-MNIST
training stays cheap, the groups operate at $12 \times 12$, $6 \times 6$, and
$3 \times 3$; the number of channels grows from 192 after the stem to 480,
832, and 1024 after the three groups.

```{.python .input #googlenet-googlenet-model-7}
%%tab mxnet, pytorch
GoogleNet().layer_summary((1, 1, 96, 96))
```

```{.python .input #googlenet-googlenet-model-7}
%%tab tensorflow, jax
GoogleNet().layer_summary((1, 96, 96, 1))
```

GoogLeNet is the deepest network of this section: 22 layers with weights lie
on its longest path, three in the stem, two in each of the nine Inception
blocks, and the output layer. We train it with the same procedure and the
same `configure_optimizers` as the other three networks, on
$96 \times 96$ images, at which its 6.0 million parameters perform 276
million multiply-adds per image.

```{.python .input #blocks-stem-body-and-head-3}
GoogleNet.configure_optimizers = VGG.configure_optimizers
data = d2l.FashionMNIST(batch_size=128, resize=(96, 96))
model = GoogleNet(lr=0.03, num_steps=10 * len(data.train_dataloader()))
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
val_acc['GoogLeNet'] = trainer.history['val_acc']
```

GoogLeNet ends at a validation accuracy of about 91% after 10 epochs, as
AlexNet does. Its training loss, below 0.2 after 10 epochs, is lower than
AlexNet's, and its validation loss, about 0.25, is the same.

## Comparing the Networks

Since the learning rate decays to zero, the validation accuracy of every
network changes little over the last epochs, and we compare the networks by
their accuracy after the last epoch:

```{.python .input #blocks-comparing-the-networks}
{name: round(acc[-1], 3) for name, acc in val_acc.items()}  # After 10 epochs
```

:numref:`tab_blocks_scoreboard` collects these accuracies for the networks
of this section and AlexNet, all trained on Fashion-MNIST for 10 epochs with
the same schedule, together with LeNet and the counts of parameters and
multiply-adds at the input size used for training.

:Parameters, multiply-adds, and Fashion-MNIST validation accuracy of the networks of :numref:`sec_lenet`, :numref:`sec_alexnet`, and this section. Multiply-adds are per image at the training input size. Accuracies are after 10 epochs, as ranges over repeated runs; a single value means that the runs agree to within half a point. LeNet's are after 30 epochs at a constant learning rate.
:label:`tab_blocks_scoreboard`

| network | input | parameters | multiply-adds | validation accuracy |
|:--|:--|--:|--:|:--|
| LeNet | $28 \times 28$ | 0.06M | 0.4M | 88–90% (30 epochs) |
| AlexNet | $224 \times 224$ | 46.8M | 938M | 91% |
| VGG, quarter width | $224 \times 224$ | 43.1M | 512M | 91.5–92% |
| NiN | $224 \times 224$ | 2.0M | 762M | 84–88% |
| GoogLeNet | $96 \times 96$ | 6.0M | 276M | 91% |

Three observations hold across repeated runs. First, the parameters of a
dense head add little accuracy here: NiN and GoogLeNet, which replace it by
global average pooling, have 23 and 8 times fewer parameters than AlexNet;
GoogLeNet matches its accuracy, and NiN ends a few points below it. Second,
arithmetic alone does not decide accuracy: the narrow VGG is the most
accurate network with about half of AlexNet's multiply-adds, while NiN
performs almost as many as AlexNet and fits the training data more slowly.
Third, GoogLeNet matches AlexNet with less than a third of its arithmetic,
although it sees $96 \times 96$ images instead of $224 \times 224$. Ten
epochs on upsampled $28 \times 28$ images are a small test; on ImageNet, in
the challenge of 2014, GoogLeNet won with about 7 million parameters per
network and a lower error than VGG, whose networks had up to 144 million.

The three networks contributed the parts from which convolutional networks
are still assembled. VGG made the repeated block the unit of design and a
network a list of block sizes. NiN introduced the $1 \times 1$ convolution
for mixing channels and global average pooling as the head. GoogLeNet ran
branches of different kernel sizes in parallel, reduced channels with
$1 \times 1$ convolutions before expensive operations, and divided a network
into stem, body, and head. Its hand-tuned branches are rare in current
networks; two of its ideas survive in a more regular form. Grouped
convolutions, as in ResNeXt (:numref:`subsec_resnext`), are multi-branch
blocks whose branches are identical, and RepVGG :cite:`ding2021repvgg` trains
with parallel branches and merges them into a single convolution for
inference (:numref:`sec_efficient_cnns`).

None of these networks normalizes its activations, and none is much deeper
than twenty layers. VGG's authors trained their deepest networks by first
training shallower ones and copying their weights, and GoogLeNet added
auxiliary classifiers in the middle of the network to keep its gradients
from vanishing; our implementations omit both. The next section introduces
normalization layers, which let much deeper networks train without such
devices.

## Summary

After AlexNet, networks were built from blocks rather than individual layers.
VGG repeats a block of $3 \times 3$ convolutions and pooling, so that a
network is a list of block sizes and depth comes from stacking small kernels.
NiN mixes channels with $1 \times 1$ convolutions and replaces the fully
connected head, which holds over 90% of the parameters of AlexNet and VGG,
by global average pooling with no parameters. GoogLeNet runs kernels of
several sizes in parallel, makes them affordable with $1 \times 1$
reductions, and organizes the network into stem, body, and head. Under the
same training procedure on Fashion-MNIST, the narrow VGG is the most
accurate of the four; GoogLeNet matches AlexNet and NiN ends a few points
below it, each with a small fraction of its parameters.

## Exercises

1. [code] **Global average pooling versus a dense head.** Replace the
   global average pooling of `NiN` by a fully connected layer that maps
   the flattened $10 \times 5 \times 5$ output of the last `nin_block` to
   the 10 logits.
    1. Compare the number of parameters, validation accuracy, and time per
       epoch with those of `NiN`.
    1. Shift every resized $224 \times 224$ validation image horizontally
       by 32 pixels, filling with zeros, and compare how much the
       validation accuracy of each network drops. Relate the result to
       the argument about translations in this section.

1. [code] **The $1 \times 1$ convolutions of NiN.** Remove both
   $1 \times 1$ convolutions from every `nin_block`, so that each block is
   a single convolution followed by a ReLU, except in the last block, whose
   convolution outputs the class scores. Train the network as `NiN`.
    1. How many parameters and multiply-adds does the network lose?
    1. Compare the validation accuracy with that of `NiN`. Which of the
       four blocks needs its $1 \times 1$ convolutions most? Test your
       answer by restoring them in one block at a time.

1. **VGG-16 from its configuration.** VGG-16 has 13 convolutional layers in
   five blocks with `arch=((2, 64), (2, 128), (3, 256), (3, 512), (3,
   512))`, and three fully connected layers of 4096, 4096, and 1000 units,
   on $224 \times 224$ color images.
    1. Count its parameters and multiply-adds by hand, separately for the
       convolutional and the fully connected layers, and check your total
       against the 138 million parameters reported by
       :citet:`Simonyan.Zisserman.2014`.
    1. Which layer has the most parameters? Which layers perform the most
       multiply-adds, and why do several of them tie?

1. **Depthwise Inception.** Suppose the four branches of the first body
   block of `GoogleNet`, which maps 192 input channels to
   $64 + 128 + 32 + 32 = 256$ output channels, are replaced by a single
   $7 \times 7$ depthwise convolution followed by a $1 \times 1$
   convolution (:numref:`sec_depthwise_separable`). Compare the parameter
   count and the number of multiplications of the two blocks at
   the same input and output sizes.

1. [code] **Native resolution.** Fashion-MNIST images are
   $28 \times 28$, and the training runs of this section upsample them
   eightfold to $224 \times 224$, which spends computation without adding
   information.
    1. Train `VGG` as in this section, with
       `arch=((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))` and
       `lr=0.03`, on images resized to $56 \times 56$ and to
       $84 \times 84$. The input size of the first fully connected layer changes with the
       resolution.
       Compare validation accuracy and time per epoch with the same model
       trained at $224 \times 224$, and give the spatial size of the
       feature map that enters the fully connected layers at each of the
       three resolutions.
    1. `GoogleNet` accepts a $28 \times 28$ input as written, because its
       convolutions and pooling layers pad so that every stride-2 step
       halves the resolution, rounding up. Compute the resolution at which
       each of the three groups of Inception blocks operates on a
       $28 \times 28$ input. What do the $3 \times 3$ and $5 \times 5$
       branches compute at the smallest of these resolutions?

1. **Where the savings come from.** ● Use the counts of this section and
   the table of AlexNet's layers, :numref:`tab_alexnet_cost`.
    1. For VGG-11, `NiN`, and `GoogleNet` on a single-channel
       $224 \times 224$ image with ten classes, split the parameters and
       the multiply-adds between the convolutional layers and the
       classification head.
    1. Relative to VGG-11, does `NiN` save more parameters in its body or
       in its head? And `GoogleNet`? Answer the same questions for the
       multiply-adds, and name the design choice responsible for each
       saving.

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 10.18.*

1. [code] **Single-branch ablation.** ● Build a variant of `GoogleNet` whose
   Inception blocks keep only the $3 \times 3$ branch with its
   $1 \times 1$ reduction. Let the $3 \times 3$ convolution emit as many
   channels as the original block, and choose the width of the
   $1 \times 1$ reduction so that each block's parameter count stays
   within 10% of the original. Train both networks as `GoogleNet` is
   trained in this section and compare validation accuracy. What does the
   result say about the value of mixing kernel sizes within one block?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/77)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/78)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/277)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18002)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §7.2]{.kicker}

Blocks, Bottlenecks, and Branches<br>**VGG blocks · 1×1 convolutions · global average pooling · Inception**
:::
:::

::: {.slide title="AlexNet Sets Each Layer by Hand and Keeps 92% of Its Weights in the Head"}
[The problem]{.kicker}

- Eight layers, each kernel size and width chosen individually: no rule
  for building a network of twenty layers.
- The fully connected head holds 43 M of AlexNet's 47 M parameters and
  does 5% of its arithmetic.
- The head needs dropout to keep from memorizing the training set.

::: {.d2l-note .warn}
How should a deeper network be designed, and does it need the head at all?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- build a network from one repeated block and a list of block sizes;
- compare stacked 3×3 kernels with one large kernel;
- replace the dense head by 1×1 convolutions and global average pooling;
- cost an Inception block with and without its 1×1 reductions;
- compare four networks trained the same way.

**Running example:** the Fashion-MNIST scoreboard. AlexNet: 46.8 M
parameters, 938 M multiply-adds, ≈ 91%.
:::

::: {.slide title="Three Networks of 2013–2014 Organized the Convnet"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| AlexNet | learned features, ReLU, dropout, GPUs |
| **VGG, NiN, GoogLeNet** | **blocks, 1 × 1 convolutions, global pooling** |
| Normalization Layers | batch, layer, group normalization |
| ResNet, ResNeXt, DenseNet | residual connections make depth trainable |
| Efficient Convnets | separable blocks, scaling, re-parameterization |
| Training Recipes | schedules, augmentation, weight averaging |
| ConvNeXt | a ResNet modernized one change at a time |
| Design Spaces (RegNet) | rules from populations of networks |
:::

::: {.col .narrow}
- **Uses:** AlexNet and its parameter count; 1×1 convolutions;
  `Classifier`, `Trainer`.
- **Skips:** ImageNet training; VGG's staged training and GoogLeNet's
  auxiliary classifiers.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[VGG]{.dtitle}

[one block, repeated]{.dsub}
:::
:::

::: {.slide title="Pooling After Every Convolution Would Stop a 224-Pixel Network at Seven Layers"}
[VGG]{.kicker}

- Stride-2 pooling halves the side: 224 → 112 → 56 → 28 → 14 → 7 → 3 → 1.
- At most $\lfloor \log_2 d \rfloor$ such stages for an input of width $d$.
- To go deeper, place **several stride-1 convolutions at each resolution**,
  padded to keep the size.
:::

::: {.slide title="Two 3×3 Convolutions See 5×5 With 18c² Weights Instead of 25c²"}
[Check]{.kicker}

$c$ input and $c$ output channels, stride 1. Two 3×3 layers see a 5×5
window with $2 \cdot 9c^2 = 18c^2$ weights; one 5×5 layer needs $25c^2$.

What window do **three** 3×3 layers see, and how many weights do they need
against one layer of that size?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** 7×7, with $27c^2$ weights against $49c^2$, and three
nonlinearities instead of one.
:::
:::
:::

::: {.slide title="A VGG Block Is a Few 3×3 Convolutions and One Pooling" except="tensorflow"}
[VGG]{.kicker}

@vgg-vgg-blocks

`num_convs` convolutions keep the size; the pooling halves it.
:::

::: {.slide title="A VGG Block Is a Few 3×3 Convolutions and One Pooling" only="tensorflow"}
[VGG]{.kicker}

```{.python #blocks-exercises}
def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels, kernel_size=3,
                                       padding='same', activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk
```

Excerpt without the `kernel_initializer` argument. `num_convs`
convolutions keep the size; the pooling halves it.
:::

::: {.slide title="A VGG Network Is a List of Block Sizes"}
[VGG]{.kicker}

`arch` lists (convolutions, channels) per block. VGG-11:
`((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))`

| block | convolutions | channels | output |
|---|--:|--:|---|
| 1 | 1 | 64 | 64@112×112 |
| 2 | 1 | 128 | 128@56×56 |
| 3 | 2 | 256 | 256@28×28 |
| 4 | 2 | 512 | 512@14×14 |
| 5 | 2 | 512 | 512@7×7 |

Then AlexNet's head: 4096, 4096, 10. VGG-16 only changes `arch`:
`((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))`.
:::

::: {.slide title="VGG-11 Does 8× AlexNet's Convolutional Arithmetic; Its Head Holds 93% of the Weights"}
[VGG]{.kicker}

| VGG-11, one 224×224 image | parameters | multiply-adds |
|---|--:|--:|
| 8 convolutions | 9.2 M | 7.43 G (98%) |
| head, 25,088 → 4096 → 4096 → 10 | **119.6 M** (93%) | 0.12 G |
| total | 128.8 M | 7.5 G |

The first dense layer alone: $25{,}088 \cdot 4096 = 103$ M weights.
:::

::: {.slide title="Check: Where Does a Quarter-Width VGG Keep Its Weights?"}
[Check]{.kicker}

Channels 16, 32, 64, 128, 128 instead of 64 … 512. The last block emits
128@7×7. How many weights in the first dense layer (to 4096 units)?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $128 \cdot 7 \cdot 7 \cdot 4096 \approx 25.7$ M. The whole head
holds 42.5 M of 43.1 M parameters: **99%**. The convolutions shrank to
0.6 M; the head did not.
:::
:::
:::

::: {.slide title="The Quarter-Width VGG Reaches About 92% at Half of AlexNet's Arithmetic"}
[VGG]{.kicker}

@!vgg-training

Same procedure as AlexNet: 224×224, 10 epochs of SGD, warmup over the
first epoch to a peak of 0.03, cosine decay to 0.
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[NiN]{.dtitle}

[1×1 convolutions and no dense head]{.dsub}
:::
:::

::: {.slide title="A 1×1 Convolution Is a Dense Layer Applied at Every Pixel"}
[NiN]{.kicker}

@nin-nin-blocks

Two 1×1 convolutions after each convolution: a one-hidden-layer MLP
over the channels at every position, hence *network in network*.
:::

::: {.slide title="Global Average Pooling Turns Ten Channels Into Ten Scores With No Parameters"}
[NiN]{.kicker}

![VGG's head (left) and NiN's end (right): the last block emits one channel per class, and each channel is averaged over all positions.](../img/arch-nin.svg){width=80%}

The average does not change when a shift permutes the positions.
:::

::: {.slide title="NiN Has 2.0 M Parameters, 4% of AlexNet's"}
[NiN]{.kicker}

- AlexNet's first layers as NiN blocks: 11×11, 5×5, 3×3 with 96, 256,
  384 channels, each followed by max-pooling.
- Shapes: 224 → 54 → 26 → 12 → 5, then a block with **10 channels**
  and global average pooling.
- The last 1×1 convolution has **no ReLU**: it outputs the class scores.
  A ReLU there can switch a class off for good (logit 0, zero gradient).
- 2.0 M parameters; 762 M multiply-adds, a little less than AlexNet's
  938 M.
:::

::: {.slide title="NiN Reaches 84–88% With 4% of AlexNet's Parameters"}
[NiN]{.kicker}

@!nin-training

Same procedure as AlexNet and VGG. Training loss 0.3–0.45 against
AlexNet's ≈ 0.22: NiN fits more slowly and does not overfit.
:::

::: {.slide title="NiN Needs the Warmup and the Decay Most"}
[NiN]{.kicker}

- Constant rate 0.01: still improving at epoch 10; a run ends anywhere
  between ≈ 78% and 87%, depending on how fast it leaves its first plateau.
- Warmup + cosine, peak 0.03: 84–88%; a slow start catches up, and the
  decay removes single-epoch drops at the end.
- Higher peaks fail: at 0.05 an occasional run stays near chance for
  several epochs; at 0.1 some runs diverge.
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[GoogLeNet]{.dtitle}

[parallel branches and 1×1 bottlenecks]{.dsub}
:::
:::

::: {.slide title="The Inception Block Runs Four Kernel Sizes in Parallel"}
[GoogLeNet]{.kicker}

![The Inception block; input at the bottom.](../img/arch-inception-block.svg){width=70%}

Each branch keeps the size; the outputs are concatenated along channels.
:::

::: {.slide title="Check: What Does the 1×1 Reduction Save?"}
[Check]{.kicker}

First Inception block: 192 channels in at 28×28. The 5×5 branch emits 32
channels. Weights with and without a 16-channel 1×1 reduction?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Direct: $25 \cdot 192 \cdot 32 = 153{,}600$. Reduced:
$192 \cdot 16 + 25 \cdot 16 \cdot 32 = 15{,}872$, **a tenth**, and the
multiply-adds fall by the same factor.
:::
:::
:::

::: {.slide title="Stem, Body, and Head: Nine Inception Blocks in Three Groups"}
[GoogLeNet]{.kicker}

- **Stem:** 7×7 convolution with stride 2, pooling, 1×1 and 3×3
  convolutions to 192 channels, pooling.
- **Body:** Inception blocks in groups of 2, 5, and 2, pooling between
  groups; channels 192 → 480 → 832 → 1024.
- **Head:** NiN's global average pooling, then one dense layer.
- Branch widths: a hand-chosen tuple, stored as data like VGG's `arch`.
:::

::: {.slide title="GoogLeNet Ends at About 91% With 6 M Parameters, at 96×96"}
[GoogLeNet]{.kicker}

@!blocks-stem-body-and-head-3

AlexNet's accuracy with an eighth of its parameters and less than a third
of its arithmetic.
:::

::: {.slide title="The Dense Head Adds Little Accuracy; VGG Is Most Accurate at Half of AlexNet's Arithmetic"}
[Scoreboard]{.kicker}

| network | input | parameters | multiply-adds | accuracy |
|---|---|--:|--:|---|
| LeNet (30 epochs, lr 2.0) | 28×28 | 0.06 M | 0.4 M | 88–90% |
| AlexNet | 224×224 | 46.8 M | 938 M | 91% |
| VGG, quarter width | 224×224 | 43.1 M | 512 M | 91.5–92% |
| NiN | 224×224 | 2.0 M | 762 M | 84–88% |
| GoogLeNet | 96×96 | 6.0 M | 276 M | 91% |

Validation accuracy after 10 epochs (LeNet: 30); SGD with warmup to 0.03
and cosine decay, He initialization.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **VGG:** a network is a list of blocks; stacked 3×3 kernels replace
  large ones.
- **NiN:** 1×1 convolutions mix channels; global average pooling
  replaces a head that held over 90% of the weights.
- **GoogLeNet:** parallel kernel sizes, 1×1 reductions (the 5×5 branch
  10× cheaper), stem–body–head.
- **Fashion-MNIST** (one schedule for all): narrow VGG 92%; GoogLeNet
  matches AlexNet's 91% with 8× fewer parameters, NiN ends a few points
  below with 23× fewer.

**Next:** normalization layers, which let much deeper networks train.
:::
