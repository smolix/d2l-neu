```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# AlexNet and Learned Image Representations
:label:`sec_alexnet`

In :numref:`sec_lenet`, LeNet learned every operation in its pipeline directly
from training examples—both its convolutional filters and its final
classifier. Although Yann LeCun and his colleagues had already trained such a
network end to end in 1989, for the next two decades mainstream computer
vision moved in a different direction. Classical vision pipelines split
representation from prediction: hand-engineered descriptors such as SIFT
:cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, HOG
:cite:`Dalal.Triggs.2005`, and bags of visual words
:cite:`Sivic.Zisserman.2003` converted raw pixels into fixed feature vectors,
which were then fed into a linear model or kernel classifier. Because the
training objective could adjust only the classifier and never the features,
progress depended largely on human ingenuity in designing better feature
extractors. By contrast, a representation is *learned* when the loss function
shapes the feature extractors jointly with the classifier. Why did learned
representations take over twenty years to overtake engineered ones, and what
finally tipped the balance?

The turning point arrived at the 2012 ImageNet challenge, which evaluated
models on sorting natural photographs into 1000 object categories using
*top-5 error*—the fraction of validation images whose true label does not
appear among the model's five highest-probability predictions. *AlexNet*
:cite:`Krizhevsky.Sutskever.Hinton.2012` replaced the traditional two-stage
pipeline with a single deep convolutional network trained end to end on raw
pixels, cutting top-5 error to 15.3% compared with 26.2% for the strongest
hand-engineered entry. Whereas the winning error had improved by roughly two
percentage points the year before, AlexNet slashed it by more than ten in a
single leap.

Architecturally, AlexNet is essentially LeNet scaled up: a stack of
convolutional stages that trade spatial resolution for channel depth, topped
by a block of fully connected layers. Because that blueprint was already more
than two decades old, its sudden triumph points straight to the real
catalysts. Three ingredients converged in 2012: a labeled dataset large enough
to constrain tens of millions of parameters, GPUs fast enough to train them in
days rather than months, and key modeling practices—ReLU activations, dropout,
and data augmentation—that made deep networks both trainable and resistant to
overfitting.

In this section, we examine why each of these three ingredients mattered,
implement AlexNet, account for where its parameters and arithmetic reside, and
train it on Fashion-MNIST.

```{.python .input #alexnet-deep-convolutional-neural-networks-alexnet  n=2}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #alexnet-deep-convolutional-neural-networks-alexnet  n=3}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #alexnet-deep-convolutional-neural-networks-alexnet  n=4}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #alexnet-deep-convolutional-neural-networks-alexnet}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Learned Features, Data, and Hardware

Learning visual features from data was not brand new in 2012: beyond LeNet,
unsupervised sparse coding had already demonstrated that edge-like image
filters can emerge without labels :cite:`olshausen1996emergence`. What changed
in 2012 was that hierarchically learned features decisively outperformed
hand-crafted descriptors on a large-scale natural-image benchmark. Inspecting
AlexNet's first convolutional layer reveals what backpropagation discovered on
its own. :numref:`fig_filters` displays its 96 learned $11 \times 11$ kernels:
oriented edge and frequency detectors alongside color-opponent blobs—strikingly
reminiscent of the oriented-gradient filters built into SIFT and HOG by hand.
No engineer programmed these filters; the network discovered them purely to
minimize classification loss, and then composed them layer by layer into
higher-level parts and object representations that no human could specify by
hand.

![The 96 kernels of the first convolutional layer of AlexNet, learned on ImageNet. Reproduction courtesy of :citet:`Krizhevsky.Sutskever.Hinton.2012`.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

### Data

Deep models with tens of millions of parameters quickly memorize small
benchmarks such as PASCAL VOC, which contained only on the order of ten
thousand images. The ImageNet dataset :cite:`Deng.Dong.Socher.ea.2009` broke
that bottleneck: its 2012 competition split :cite:`russakovsky2015imagenet`
provided roughly 1.2 million labeled training photographs spanning 1000
categories. Even so, 60 million parameters still outnumber 1.2 million images
by fifty to one, so AlexNet's authors expanded the effective dataset via data
augmentation—extracting random $224 \times 224$ crops from images resized to
$256 \times 256$ along with horizontal reflections. Because a crop can start
at any of roughly $32 \times 32$ offsets and be flipped or unflipped, each
photograph yields up to $32 \cdot 32 \cdot 2 = 2{,}048$ distinct training
views. Although these views overlap strongly, they significantly boost sample
diversity—yet even with augmentation, AlexNet still overfit severely without
dropout :cite:`Krizhevsky.Sutskever.Hinton.2012`.

### Hardware

Training a deep CNN means executing the same simple arithmetic operations in
massive parallel batches: applying a kernel across every spatial position and
output channel involves millions of independent multiply-adds. A *graphics
processing unit* (GPU) is tailor-made for this workload. Rather than dedicating
silicon to a handful of complex, high-clock cores optimized for branching
control flow, a GPU packs hundreds or thousands of streamlined arithmetic
units behind a wide, high-bandwidth memory bus. Each of the two NVIDIA GTX
580 GPUs used to train AlexNet delivered roughly 1.6 trillion single-precision
floating-point operations per second (TFLOPS) and 192 GB/s of memory
bandwidth, compared with about 0.2 TFLOPS and 20 GB/s for a 2011 quad-core
desktop CPU.

A quick back-of-the-envelope estimate shows how decisive that gap was. A
forward pass through AlexNet on one $224 \times 224$ RGB image requires about
a billion multiply-adds. A full training step costs roughly three times as
much: one pass forward, plus two comparable passes backward per layer (one to
compute weight gradients and another to propagate gradients to the layer
below). Counting two floating-point operations per multiply-add, training for
90 epochs across 1.2 million images demands approximately
$10^9 \cdot 3 \cdot 2 \cdot 1.2 \times 10^6 \cdot 90 \approx 6.5 \times 10^{17}$
operations. At theoretical peak, two GTX 580s could finish in two to three
days, and the actual run took five to six days—whereas a CPU running at peak
throughput would have required more than a month for every single experiment.
In 2012, Alex Krizhevsky had to write custom CUDA kernels by hand
([cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)) to harness
those GPUs; today, the exact same architecture takes a dozen lines of
framework code.

## The AlexNet Architecture

:numref:`fig_alexnet` compares LeNet (:numref:`sec_lenet`) side by side with
the AlexNet variant we train on Fashion-MNIST below. Both follow the same
high-level design—a stack of convolutional stages feeding into a dense
classification head—but AlexNet scales up every dimension and modernizes three
critical components: the activation function, pooling, and regularization.

![LeNet as built in :numref:`sec_lenet` (left) and AlexNet as built in this section (right), both for single-channel Fashion-MNIST images and ten classes; the input is at the bottom. AlexNet is deeper and wider, uses ReLU activations and max-pooling, and regularizes its dense layers with dropout.](../img/arch-alexnet.svg)
:label:`fig_alexnet`

### Depth and Width

AlexNet comprises eight parameterized layers—five convolutional layers and
three fully connected layers—compared with LeNet's two and three. Because
ImageNet photographs are eight times wider and taller than Fashion-MNIST
images, the first layer deploys a large $11 \times 11$ kernel with stride 4 to
capture broad spatial patterns while immediately downsampling the feature map.
The second layer narrows to $5 \times 5$ kernels, and the final three
convolutional layers use $3 \times 3$ kernels. Overlapping $3 \times 3$
max-pooling layers with stride 2 follow the first, second, and fifth
convolutions, replacing LeNet's average pooling so strong local activations
are preserved rather than diluted (:numref:`sec_pooling`). Layer widths expand
dramatically as well: convolutional channels climb from 96 to 384 (versus 6
and 16 in LeNet), and the hidden dense layers widen to 4096 units each (versus
120 and 84).

For clarity, our implementation omits two historical artifacts from the 2012
paper that are no longer needed. First, because a single GTX 580 had only
3 GB of memory, the original authors split each layer across two GPUs—a
workaround unnecessary on modern hardware. Second, the paper included local
response normalization across neighboring channels, a heuristic that later
architectures abandoned in favor of batch normalization.

### ReLU Activations

Whereas LeNet relied on saturating nonlinearities (sigmoid in our version,
scaled $\tanh$ in 1998), every sigmoid in a backward pass shrinks gradients by
at least a factor of $1/4$, forcing LeNet to use an unusually large learning
rate of 2.0 (:numref:`sec_lenet`). AlexNet replaces saturating units with the
rectified linear unit, $\textrm{ReLU}(x) = \max(x, 0)$
(:numref:`subsec_activation-functions`). Because its derivative is identically
1 for all positive inputs, ReLU passes gradients backward across active units
without attenuation, leaving only the weight-matrix scaling controlled by
proper initialization. ReLU is also faster to evaluate because it avoids
expensive exponentials. The empirical speedup was dramatic: on a small
CIFAR-10 convnet, :citet:`Krizhevsky.Sutskever.Hinton.2012` reached 25%
training error six times faster with ReLU than with $\tanh$.

### Dropout and Augmentation

Because AlexNet's two 4096-unit fully connected layers contain the vast
majority of its parameters, they are the primary site of potential
overfitting. AlexNet guards against memorization by applying dropout with
probability 0.5 after each hidden dense layer (:numref:`sec_dropout`), whereas
LeNet used no dropout at all. Alongside dropout, random cropping, horizontal
flipping, and color jittering artificially multiply the diversity of the
training images—an idea that has since evolved into an essential pillar of
modern vision training pipelines (:numref:`sec_training_recipes`).

### Implementation

Our implementation translates :numref:`fig_alexnet` directly from input to
output. Rather than relying on framework defaults, we initialize all weights
explicitly. While Xavier initialization preserves activation variance across
linear layers, a ReLU zeros out roughly half of its inputs and thus halves the
output second moment. *He initialization* (:numref:`subsec_he_init`) offsets
this factor of two by drawing weights for a layer with $n_\textrm{in}$ inputs
per output ($n_\textrm{in} = c_\textrm{i} k^2$ for a convolution) from a
zero-mean Gaussian with variance $2/n_\textrm{in}$. All biases are initialized
to zero (already the default in every framework below except PyTorch).

:begin_tab:`mxnet`
`init.MSRAPrelu('in', 0)` draws normal weights with variance
$2/n_\textrm{in}$, He initialization for a ReLU (a PReLU with slope 0).
The network is a `HybridSequential`, and `hybridize()` compiles it into a
single graph.
:end_tab:

:begin_tab:`pytorch`
The lazy layers have no weights until the first minibatch fixes their shapes.
As for LeNet, the `Trainer` then calls the model's `weight_init` on every
layer.
:end_tab:

:begin_tab:`tensorflow`
Every layer receives its own initializer, since an unseeded Keras initializer
returns the same values on every call.
:end_tab:

:begin_tab:`jax`
NNX layers draw their weights in the constructor, so every layer receives the
initializer.
:end_tab:

```{.python .input #alexnet-capacity-control-and-preprocessing-1  n=5}
%%tab mxnet
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.HybridSequential()
        self.net.add(
            nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(num_classes))
        self.net.initialize(init.MSRAPrelu('in', 0))
        self.net.hybridize()  # Compile the network into one graph
```

```{.python .input #alexnet-capacity-control-and-preprocessing-1  n=5}
%%tab pytorch
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.LazyConv2d(96, kernel_size=11, stride=4), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(384, kernel_size=3, padding=1), nn.ReLU(),
            nn.LazyConv2d(256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(4096), nn.ReLU(), nn.Dropout(p=0.5),
            nn.LazyLinear(num_classes))

    def weight_init(self, module):
        # The Trainer calls this after the first minibatch fixed the shapes
        if type(module) in (nn.Linear, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)
```

```{.python .input #alexnet-capacity-control-and-preprocessing-1  n=5}
%%tab tensorflow
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        layers = tf.keras.layers

        def he():  # A new He-normal initializer for each layer
            init = tf.keras.initializers.VarianceScaling(
                2.0, 'fan_in', 'untruncated_normal')
            return dict(kernel_initializer=init)

        self.net = tf.keras.models.Sequential([
            layers.Conv2D(96, kernel_size=11, strides=4, activation='relu',
                          **he()),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(256, kernel_size=5, padding='same',
                          activation='relu', **he()),
            layers.MaxPool2D(pool_size=3, strides=2),
            layers.Conv2D(384, kernel_size=3, padding='same',
                          activation='relu', **he()),
            layers.Conv2D(384, kernel_size=3, padding='same',
                          activation='relu', **he()),
            layers.Conv2D(256, kernel_size=3, padding='same',
                          activation='relu', **he()),
            layers.MaxPool2D(pool_size=3, strides=2), layers.Flatten(),
            layers.Dense(4096, activation='relu', **he()), layers.Dropout(0.5),
            layers.Dense(4096, activation='relu', **he()), layers.Dropout(0.5),
            layers.Dense(num_classes, **he())])
```

```{.python .input #alexnet-capacity-control-and-preprocessing-1}
%%tab jax
class AlexNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_steps=4690, num_classes=10,
                 rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = (nnx.Rngs(params=d2l.get_key(), dropout=d2l.get_key())
                if rngs is None else rngs)
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        init = dict(kernel_init=he, rngs=rngs)

        def max_pool(x):  # 3x3 windows, stride 2
            return nnx.max_pool(x, window_shape=(3, 3), strides=(2, 2))

        def flatten(x):  # One vector per example
            return x.reshape((x.shape[0], -1))

        self.net = nnx.Sequential(
            nnx.Conv(1, 96, kernel_size=(11, 11), strides=4, padding='VALID',
                     **init), nnx.relu, max_pool,
            nnx.Conv(96, 256, kernel_size=(5, 5), **init), nnx.relu,
            max_pool,
            nnx.Conv(256, 384, kernel_size=(3, 3), **init), nnx.relu,
            nnx.Conv(384, 384, kernel_size=(3, 3), **init), nnx.relu,
            nnx.Conv(384, 256, kernel_size=(3, 3), **init), nnx.relu,
            max_pool,
            flatten,  # 5 x 5 x 256 values per image
            nnx.Linear(5 * 5 * 256, 4096, **init), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, 4096, **init), nnx.relu,
            nnx.Dropout(0.5, rngs=rngs),
            nnx.Linear(4096, num_classes, **init))
```

To verify that our code matches :numref:`fig_alexnet`, we feed a single-channel
$224 \times 224$ dummy image through `layer_summary` and inspect each output
tensor shape.

```{.python .input #alexnet-capacity-control-and-preprocessing-2  n=6}
%%tab mxnet, pytorch
AlexNet().layer_summary((1, 1, 224, 224))
```

```{.python .input #alexnet-capacity-control-and-preprocessing-2  n=7}
%%tab tensorflow, jax
AlexNet().layer_summary((1, 224, 224, 1))
```

Every dimension matches the formulas of :numref:`sec_padding`. The initial
stride-4 convolution maps 224 to $\lfloor (224 - 11)/4 \rfloor + 1 = 54$, and
each overlapping $3 \times 3$ max-pooling layer with stride 2 maps $n$ to
$\lfloor (n - 3)/2 \rfloor + 1$ ($54 \to 26 \to 12 \to 5$). Because the
intermediate padded convolutions preserve spatial dimensions, the final
convolutional stage flattens into a $256 \times 5 \times 5 = 6400$-element
vector for the dense classifier head.

### Parameters and Multiply-Adds

With the layer shapes confirmed, we can tally parameters and arithmetic cost
just as we did for LeNet (:numref:`sec_lenet`): a convolutional layer reuses
each of its $c_\textrm{i} c_\textrm{o} k^2$ weights at every spatial output
position, whereas a fully connected layer multiplies each weight once per
example. :numref:`tab_alexnet_cost` summarizes the results for our
single-channel, ten-class AlexNet.

:Parameters and multiply-adds per example of the layers of AlexNet, for one single-channel $224 \times 224$ image and ten classes. Pooling, activation, and dropout layers have no parameters.
:label:`tab_alexnet_cost`

| layer | output | parameters | multiply-adds |
|:--|:--|--:|--:|
| $11\times11$ convolution, 1 to 96, stride 4 | 96@54×54 | $11{,}712$ | $33.9$M |
| $5\times5$ convolution, 96 to 256 | 256@26×26 | $614{,}656$ | $415.3$M |
| $3\times3$ convolution, 256 to 384 | 384@12×12 | $885{,}120$ | $127.4$M |
| $3\times3$ convolution, 384 to 384 | 384@12×12 | $1{,}327{,}488$ | $191.1$M |
| $3\times3$ convolution, 384 to 256 | 256@12×12 | $884{,}992$ | $127.4$M |
| fully connected, 6400 to 4096 | 4096 | $26{,}218{,}496$ | $26.2$M |
| fully connected, 4096 to 4096 | 4096 | $16{,}781{,}312$ | $16.8$M |
| fully connected, 4096 to 10 | 10 | $40{,}970$ | $0.04$M |
| total | | $46{,}764{,}746$ | $938.1$M |

The split between parameters and computation is even more lopsided than in
LeNet. The five convolutional layers account for only 8% of AlexNet's 47
million parameters yet execute 95% of its multiply-adds. Meanwhile, the three
fully connected layers consume 92% of the parameters—roughly 170 MB in 32-bit
floating point—while performing just 5% of the arithmetic. Compared with
LeNet's 61,706 parameters and 416,520 multiply-adds per image, AlexNet scales
parameters up by roughly $760\times$ and arithmetic by $2{,}250\times$. On
ImageNet (3 RGB channels and 1000 classes), the full model reaches roughly 60
million parameters, still heavily concentrated in the same dense layers.

## Training on Fashion-MNIST

Because training AlexNet on ImageNet takes days even on modern GPUs, we train
it on Fashion-MNIST to keep experiments interactive. Since Fashion-MNIST's
native $28 \times 28$ images are too small to survive an $11 \times 11$
stride-4 first layer and three pooling stages, we upsample them to
$224 \times 224$ via the `resize` argument of `d2l.FashionMNIST`. In real
applications, upsampling 64-fold adds zero visual information while
multiplying compute; here we accept that overhead solely so we can run the
unmodified AlexNet architecture.

We train with minibatch SGD for 10 epochs (a third of LeNet's 30) using a
batch size of 128. Thanks to ReLU activations, gradients flow without sigmoid
attenuation, so a much smaller learning rate than LeNet's 2.0 works cleanly.

### Learning-Rate Schedule

A fixed learning rate is rarely optimal at both ends of training. Early on,
when weights are random, an overly aggressive step can destabilize
optimization; near the end, a large step keeps the parameters bouncing around
a minimum rather than settling into it. AlexNet's original authors handled
decay manually, dividing the learning rate by 10 whenever validation error
plateaued. Instead, we adopt a smooth, hands-free schedule that has become a
modern standard—and we reuse this exact schedule throughout the rest of the
chapter so all models can be compared cleanly. Specifically, a linear *warmup*
ramps the learning rate from 0 to its peak $\eta_0$ over the first
$t_\textrm{w}$ steps, followed by a half-cosine decay back to 0 at the final
step $T$:

$$
\eta_t = \frac{\eta_0}{2} \left(1 + \cos \frac{\pi (t - t_{\textrm{w}})}{T - t_{\textrm{w}}}\right)
\quad \textrm{for } t \geq t_{\textrm{w}}.
$$
:eqlabel:`eq_cosine_warmup`

At $t = t_\textrm{w}$, the cosine term equals 1 and $\eta_t = \eta_0$; halfway
through the decay it drops to $\eta_0/2$; and at $t = T$ it reaches 0.
Warming up gently allows us to use a higher peak learning rate than would be
safe from step 1. Here we train AlexNet with peak rate $\eta_0 = 0.03$ (each
architecture in :numref:`sec_blocks` likewise uses the highest peak rate at
which it trains reliably). Below, `cosine_warmup` computes the multiplicative
schedule factor at step $t$; we save it in `d2l` for reuse across the chapter
and study learning-rate schedules systematically in :numref:`sec_scheduler`.

```{.python .input #alexnet-training-on-fashion-mnist-1}
%%tab mxnet, pytorch
def cosine_warmup(t, num_steps, warmup_steps):  #@save
    """Learning-rate factor: linear warmup from 0, then a half-cosine to 0."""
    if t < warmup_steps:
        return t / warmup_steps
    progress = min((t - warmup_steps) / (num_steps - warmup_steps), 1)
    return 0.5 * (1 + math.cos(math.pi * progress))
```

```{.python .input #alexnet-training-on-fashion-mnist-1}
%%tab tensorflow
def cosine_warmup(t, num_steps, warmup_steps):  #@save
    """Learning-rate factor: linear warmup from 0, then a half-cosine to 0."""
    t = tf.cast(t, tf.float32)  # Also works on the step counter of a graph
    warmup = tf.where(t < warmup_steps, t / warmup_steps, 1.0)
    progress = tf.clip_by_value(
        (t - warmup_steps) / (num_steps - warmup_steps), 0.0, 1.0)
    return warmup * 0.5 * (1 + tf.cos(math.pi * progress))
```

```{.python .input #alexnet-training-on-fashion-mnist-1}
%%tab jax
def cosine_warmup(t, num_steps, warmup_steps):  #@save
    """Learning-rate factor: linear warmup from 0, then a half-cosine to 0."""
    t = jnp.asarray(t, jnp.float32)  # Arguments may be traced arrays
    warmup = jnp.where(t < warmup_steps, t / warmup_steps, 1.0)
    progress = jnp.clip((t - warmup_steps) / (num_steps - warmup_steps), 0, 1)
    return warmup * 0.5 * (1 + jnp.cos(jnp.pi * progress))
```

`AlexNet` updates this schedule after every minibatch: `num_steps` specifies
the total step count $T$ across the run, and warmup spans the first tenth of
training (one epoch out of ten).

```{.python .input #alexnet-training-on-fashion-mnist-2}
%%tab mxnet
@d2l.add_to_class(AlexNet)
def configure_optimizers(self):
    schedule = lambda t: self.lr * cosine_warmup(  # MXNet counts from 1
        t - 1, self.num_steps, self.num_steps // 10)
    return gluon.Trainer(self.collect_params(), 'sgd', {
        'learning_rate': None, 'lr_scheduler': schedule})
```

```{.python .input #alexnet-training-on-fashion-mnist-2}
%%tab pytorch
@d2l.add_to_class(AlexNet)
def configure_optimizers(self):
    optim = torch.optim.SGD(self.parameters(), lr=self.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(optim, lambda t: (
        cosine_warmup(t, self.num_steps, self.num_steps // 10)))
    return optim, schedule  # The Trainer steps it after every minibatch
```

```{.python .input #alexnet-training-on-fashion-mnist-2}
%%tab tensorflow
@d2l.add_to_class(AlexNet)
def configure_optimizers(self):
    schedule = lambda: self.lr * cosine_warmup(
        optim.iterations, self.num_steps, self.num_steps // 10)
    optim = tf.keras.optimizers.SGD(schedule)
    return optim
```

```{.python .input #alexnet-training-on-fashion-mnist-2}
%%tab jax
@d2l.add_to_class(AlexNet)
def configure_optimizers(self):
    return optax.sgd(lambda t: self.lr * cosine_warmup(
        t, self.num_steps, self.num_steps // 10))
```

### Comparison with LeNet

Before running the training loop, it helps to calibrate expectations.
Fashion-MNIST has neither the dataset scale nor the visual complexity needed
to exercise AlexNet's 47 million parameters, so we should expect only a modest
accuracy gain over LeNet—the point here is to verify the implementation,
inspect its compute profile, and examine what the first layer learns. To make
that inspection possible, we snapshot the initial 96 first-layer kernels
before training begins so we can measure exactly how optimization modifies
them.

```{.python .input #alexnet-comparison-with-lenet-1}
%%tab mxnet
def first_kernels(model):
    """The 96 kernels of 11 x 11 weights of the first layer, as NumPy."""
    return d2l.numpy(model.net[0].weight.data())[:, 0]

data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
model = AlexNet(lr=0.03, num_steps=10 * len(data.train_dataloader()))
model(next(iter(data.train_dataloader()))[0][:1])  # Create the weights now
K0 = first_kernels(model)
```

```{.python .input #alexnet-comparison-with-lenet-1}
%%tab pytorch
def first_kernels(model):
    """The 96 kernels of 11 x 11 weights of the first layer, as NumPy."""
    return d2l.numpy(model.net[0].weight)[:, 0]

data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
model = AlexNet(lr=0.03, num_steps=10 * len(data.train_dataloader()))
model.apply_init([next(iter(data.train_dataloader()))[0][:1]],
                 model.weight_init)  # Create the weights now
K0 = first_kernels(model)
```

```{.python .input #alexnet-comparison-with-lenet-1}
%%tab tensorflow
def first_kernels(model):
    """The 96 kernels of 11 x 11 weights of the first layer, as NumPy."""
    kernel = d2l.numpy(model.net.layers[0].kernel)  # (11, 11, 1, 96)
    return kernel[:, :, 0].transpose(2, 0, 1)

data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
model = AlexNet(lr=0.03, num_steps=10 * len(data.train_dataloader()))
model(next(iter(data.train_dataloader()))[0][:1])  # Create the weights now
K0 = first_kernels(model)
```

```{.python .input #alexnet-comparison-with-lenet-1}
%%tab jax
def first_kernels(model):
    """The 96 kernels of 11 x 11 weights of the first layer, as NumPy."""
    kernel = d2l.numpy(model.net.layers[0].kernel[...])  # (11, 11, 1, 96)
    return kernel[:, :, 0].transpose(2, 0, 1)

data = d2l.FashionMNIST(batch_size=128, resize=(224, 224))
model = AlexNet(lr=0.03, num_steps=10 * len(data.train_dataloader()))
K0 = first_kernels(model)
```

```{.python .input #alexnet-training  n=8}
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
trainer.fit(model, data)
```

After 10 epochs, AlexNet reaches roughly 91% validation accuracy, with a
training loss near 0.22 and a validation loss near 0.25. Because the cosine
schedule anneals the learning rate toward zero over the final epochs, accuracy
fluctuates by less than a percentage point at the end, making the final epoch
a reliable summary of the run. Compared with LeNet's ~88--90% after 30 epochs,
AlexNet edges ahead by one or two percentage points in a third as many epochs,
at the price of $2{,}250\times$ more arithmetic per image—and even that modest
gain reflects its ReLU activations, He initialization, and learning-rate
schedule as much as its larger capacity.

### The Learned First Layer

Whereas LeNet's $5 \times 5$ kernels were too tiny to read visually
(:numref:`sec_lenet`), AlexNet's 96 kernels of size $11 \times 11$ are large
enough to interpret directly as small images, as we saw on ImageNet in
:numref:`fig_filters`. To see what our model learned on Fashion-MNIST, we
subtract the initial random weights `K0` from the trained first-layer kernels
and plot each weight difference on its own grayscale range.

```{.python .input #alexnet-what-the-first-layer-learned-1}
change = first_kernels(model) - K0
d2l.show_images(change, 6, 16, scale=0.6, cmap='gray')
print(f'mean size of the change: {np.abs(change).mean() / np.abs(K0).mean():.3f}'
      ' of the mean size of the initial weights')
```

Ten epochs shift the first-layer weights by only about 4% of their initial
magnitude, so plotting the raw trained kernels would still look mostly like
initial random noise. The *weight updates* themselves, however, are distinctly
structured: smooth brightness gradients and broad oriented bars—low-frequency
counterparts of the crisp ImageNet edge filters in :numref:`fig_filters`. Both
missing ingredients (fine high-frequency edges and color blobs) follow
directly from the input data: our images are grayscale, and eightfold
upsampling leaves no genuine detail finer than roughly eight pixels, which an
$11 \times 11$ kernel perceives as a gentle gradient. In short, the first
layer adapts faithfully to the spatial statistics of its dataset.

Finally, recall from :numref:`tab_alexnet_cost` that 92% of AlexNet's
parameters reside in its massive fully connected head, which contributes
little arithmetic yet requires heavy dropout to curb memorization. In the next
section, we ask how to build deep architectures systematically from reusable
blocks—and whether that giant dense head is needed at all.

## Summary

End-to-end learned visual representations surpassed hand-crafted feature
pipelines when three factors aligned in 2012: large-scale labeled datasets
(ImageNet), massively parallel GPUs capable of training tens of millions of
parameters in days, and modern training components (ReLU activations, dropout,
and data augmentation). Scaling up LeNet's basic template with these tools,
AlexNet slashed ImageNet top-5 error from 26.2% to 15.3%. Accounting for its
layers reveals a stark asymmetry: the convolutional trunk executes 95% of the
arithmetic while the dense head holds 92% of the parameters. On Fashion-MNIST,
which cannot utilize such vast capacity, AlexNet surpasses LeNet by one to two
percentage points in a third of the epochs, at roughly $2{,}250\times$ the
arithmetic per image.

## Exercises

1. **Memory and compute ledger.** Extend :numref:`tab_alexnet_cost` to the
   memory that `AlexNet` needs in 32-bit floating point.
    1. Compute the number of activations that the forward pass of one
       $224 \times 224$ image stores, layer by layer, counting the output
       of every convolutional, pooling, and fully connected layer. Where is
       the memory for activations concentrated, and where is the memory for
       parameters concentrated?
    1. During training, the parameters, their gradients, and the stored
       activations of a whole minibatch must fit in memory. At which batch
       size does the memory for activations exceed the memory for
       parameters and gradients?
    1. Repeat the parameter count of :numref:`tab_alexnet_cost` for the
       ImageNet version, with three input channels, 1000 classes, and
       $227 \times 227$ inputs, which make the last convolutional stage
       $6 \times 6$ instead of $5 \times 5$. Compare your total with the 60
       million parameters reported by
       :citet:`Krizhevsky.Sutskever.Hinton.2012`.

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 10.16.*
   :exlabel:`ex_alexnet-memory-compute-ledger`

1. **Receptive fields of the early layers.** A stack of layers with kernel
   sizes $k_i$ and strides $s_i$ gives its top unit a receptive field of
   side $r = 1 + \sum_{i} (k_i - 1) \prod_{j<i} s_j$
   (:numref:`sec_padding`). Apply it to `AlexNet`, counting each
   max-pooling layer as a layer with its own kernel size and stride, to
   compute the receptive field of one unit after each of the five
   convolutional layers. What fraction of the $224 \times 224$ input side
   does each receptive field span, and where does a unit first see the
   whole image?

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problem 10.17.*

1. [code] **Change per layer.** Training moved the first-layer kernels by
   about 4% of the size of their initial values.
    1. Keep a copy of every weight array before training and compute the
       same ratio for each of the eight layers with weights. Which layers
       moved most, and which least? Relate the order to the number of
       weights in each layer.
    1. The text attributes the smoothness of the first layer's change to
       the eightfold upsampling. Test this: train on images resized to
       $112 \times 112$, which are upsampled fourfold (the last
       convolutional stage is then $2 \times 2$, so the first fully
       connected layer has 1024 inputs), and draw the change of the first
       layer again. At which scale does it vary now?

1. [code] **Dropout.** The fully connected layers of `AlexNet` hold 92% of
   its parameters, and dropout is its only regularizer.
    1. Train `AlexNet` as in this section with both dropout layers removed,
       and compare the training loss, the validation loss, and the
       validation accuracy after 10 epochs with those of the original.
    1. Train both networks on a random subset of 6,000 training images for
       the same number of gradient steps. How do the differences change, and
       why?

1. **Arithmetic intensity.** ● The *arithmetic intensity* of a layer is
   the number of multiply-adds it performs per byte of parameters it reads
   from memory. Use :numref:`tab_alexnet_cost`.
    1. Compute the arithmetic intensity of every layer of `AlexNet`, once
       for a batch of 1 image and once for a batch of 128 images, assuming
       that each parameter is read once per batch.
    1. Consider a chip that performs 10 multiply-adds in the time it reads
       one byte from memory. Assume that each layer takes the longer of its
       arithmetic time and its parameter-reading time. A designer can spend
       extra chip area on doubling either the arithmetic throughput or the
       memory bandwidth. Which choice speeds up single-image inference more,
       and which speeds up training at batch size 128 more?

1. [code] **Batch size, throughput, and memory.** ● Train `AlexNet` as in
   this section while varying only the batch size.
    1. Using your answer to problem
       :exref:`ex_alexnet-memory-compute-ledger`, predict the largest
       training batch size that fits in the memory of your GPU.
    1. Starting from a batch size of 32 and doubling it, find the largest
       batch size that trains without running out of memory. Compare it
       with your prediction and name the main source of any discrepancy.
    1. Measure the throughput in images per second over a fixed number of
       training steps for at least four batch sizes up to that limit.
       Beyond which batch size does throughput stop growing, and why?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/276)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18001)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §7.1]{.kicker}

AlexNet<br>**learned features · data and GPUs · ReLU and dropout · where the parameters sit**
:::
:::

::: {.slide title="Hand-Designed Features Stalled at 26% Top-5 Error on ImageNet"}
[The problem]{.kicker}

- ImageNet challenge: 1000 classes, 1.2 M training photographs.
- Top-5 error: the correct label is not among the five best guesses.
- Winners: **28%** (2010), **26%** (2011).
- Their pipeline: SIFT, HOG, bags of words → linear or kernel classifier.
- The loss trains the classifier, **never the features**.

::: {.d2l-note .warn}
Can one network learn the features and the classifier together?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- explain why data, GPUs, and ReLU arrived together in 2012;
- estimate the compute needed to train AlexNet;
- build AlexNet and trace its shapes;
- count its parameters and multiply-adds layer by layer;
- read its Fashion-MNIST result against LeNet's.

**Running example:** a Fashion-MNIST scoreboard of parameters,
multiply-adds, and validation accuracy. LeNet: 62k, 0.42 M, ≈ 88–90%.
:::

::: {.slide title="The First Modern Convnet, Before Blocks and Normalization"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| **AlexNet** | **learned features, ReLU, dropout, GPUs** |
| VGG, NiN, GoogLeNet | blocks, 1 × 1 convolutions, global pooling |
| Normalization Layers | batch, layer, group normalization |
| ResNet, ResNeXt, DenseNet | residual connections make depth trainable |
| Efficient Convnets | separable blocks, scaling, re-parameterization |
| Training Recipes | schedules, augmentation, weight averaging |
| ConvNeXt | a ResNet modernized one change at a time |
| Design Spaces (RegNet) | rules from populations of networks |
:::

::: {.col .narrow}
- **Uses:** LeNet, `Classifier`, `Trainer`, dropout, He
  initialization.
- **Skips:** training on ImageNet; the paper's two-GPU split.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Learned Representations]{.dtitle}

[what made 2012 different: data and hardware]{.dsub}
:::
:::

::: {.slide title="AlexNet's First Layer Learned Edge and Color Detectors"}
[Representations]{.kicker}

![The 96 first-layer kernels of AlexNet, 11×11 pixels each, learned on ImageNet.](../img/filters.png){width=55%}

Oriented edges and color patches: close to what SIFT and HOG compute by
design, found by the loss alone.
:::

::: {.slide title="1.2 Million Images Made Tens of Millions of Parameters Learnable"}
[Data]{.kicker}

- PASCAL VOC, the benchmark of the time: about 10,000 images, 20 classes.
- ImageNet challenge: 1.2 M training images, 1000 classes.
- Random 224×224 crops of 256×256 images, and their mirror images:
  about $32 \cdot 32 \cdot 2 = 2{,}048$ inputs per image.
- Even so, the network overfit without dropout.
:::

::: {.slide title="Check: How Long Does Training AlexNet Take?"}
[Check]{.kicker}

- Forward pass: about $10^9$ multiply-adds per 224×224 color image.
- Training step: ×3 (forward and two backward products); ×2 operations
  per multiply-add; 90 epochs of 1.2 M images.

How many operations in all? How many days on two GTX 580s at 1.6 TFLOPS
each, and on a 0.2-TFLOPS CPU?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $10^9 \cdot 6 \cdot 1.2 \times 10^6 \cdot 90 \approx 6.5
\times 10^{17}$: **2–3 days** on the GPUs at peak (the run took 5–6),
**more than a month** on the CPU.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[The AlexNet Architecture]{.dtitle}

[LeNet's template, deeper and wider]{.dsub}
:::
:::

::: {.slide title="AlexNet Is LeNet's Template, Deeper and Wider"}
[Architecture]{.kicker}

| | LeNet | AlexNet |
|---|---|---|
| input | 28×28 | 224×224 |
| convolutions | 2: 5×5; 6, 16 channels | 5: 11×11 (stride 4), 5×5, three 3×3; 96–384 channels |
| pooling | 2×2 average | 3×3 max, stride 2 |
| activation | sigmoid | ReLU |
| dense head | 120, 84, 10 | 4096, 4096, 10, dropout 0.5 |
| parameters | 62 k | 47 M |
:::

::: {.slide title="Five Convolutions, Three Poolings, Then a 4096-Wide Head"}
[Architecture]{.kicker}

- Conv 11×11, 96 channels, stride 4 → max-pool 3×3, stride 2
- Conv 5×5, 256, padding 2 → max-pool
- Conv 3×3, 384 → 384 → 256, padding 1 → max-pool
- Flatten → dense 4096 → dense 4096 → dense 10, dropout 0.5 after each
  hidden dense layer
- ReLU after every hidden layer

Our version drops the paper's two-GPU split and its normalization across
channels.
:::

::: {.slide title="A Sigmoid Multiplies the Gradient by at Most 1/4; an Active ReLU by 1"}
[Architecture]{.kicker}

- Sigmoid: $\sigma'(x) = \sigma(x)(1 - \sigma(x)) \le \tfrac14$; the
  $K$ sigmoid factors on the backward path multiply to at most $4^{-K}$.
- ReLU: $\max(x, 0)$ has derivative 1 for every $x > 0$. The weights
  still scale the gradient; He initialization controls that.
- LeNet needed learning rate 2.0; AlexNet trains with a peak of 0.03.
- Small CIFAR-10 convnet: 25% training error **6× faster** with ReLU
  than with tanh.
:::

::: {.slide title="The Encoder Shrinks 224×224 to 5×5 While Channels Grow to 384"}
[Shapes]{.kicker}

| layer | output |
|---|---|
| input | 1@224×224 |
| conv 11×11, stride 4 → max-pool | 96@54×54 → 96@26×26 |
| conv 5×5, padding 2 → max-pool | 256@26×26 → 256@12×12 |
| three conv 3×3, padding 1 | 384, 384, 256 @12×12 |
| max-pool → flatten | 256@5×5 → 6400 |
| dense layers | 4096 → 4096 → 10 |
:::

::: {.slide title="Check: An 11×11 Kernel With Stride 4 on a 224×224 Image"}
[Check]{.kicker}

One input channel, 96 output channels, no padding. Output size?
Parameters? Multiply-adds?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\lfloor (224 - 11)/4 \rfloor + 1 = 54$: **96@54×54**.
Weights $96 \cdot 121 = 11{,}616$, plus 96 biases. Each weight is used at
$54^2 = 2{,}916$ positions: **33.9 M** multiply-adds.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Cost and Training]{.dtitle}

[where the parameters sit, and what Fashion-MNIST shows]{.dsub}
:::
:::

::: {.slide title="Convolutions Do 95% of the Arithmetic; the Dense Head Holds 92% of the Weights"}
[Parameters and multiply-adds]{.kicker}

| layers | parameters | multiply-adds |
|---|--:|--:|
| 5 convolutions | 3.7 M (8%) | 895 M (95%) |
| dense 6400 → 4096 | **26.2 M** | 26 M |
| dense 4096 → 4096 | 16.8 M | 17 M |
| dense 4096 → 10 | 0.04 M | 0.04 M |
| total | 46.8 M | 938 M |

The dense head is 170 MB in float32 and does 5% of the work.
:::

::: {.slide title="AlexNet Has 760× LeNet's Parameters and Does 2,250× Its Arithmetic"}
[Scoreboard]{.kicker}

| | LeNet | AlexNet |
|---|--:|--:|
| input | 28×28 | 224×224 (upsampled) |
| parameters | 61,706 | 46.8 M |
| multiply-adds per image | 0.42 M | 938 M |
| activation | sigmoid | ReLU |
| learning rate, epochs | 2.0, 30 | peak 0.03, 10 |
:::

::: {.slide title="Warm Up, Then Decay the Learning Rate to Zero"}
[Training]{.kicker}

- **Warmup:** the rate rises linearly from 0 to its peak $\eta_0$ over
  the first $t_\textrm{w}$ steps. Small first steps while the weights are
  random; the peak can then be higher.
- **Decay:** a half-cosine from $\eta_0$ to 0 at the last step $T$:
  $\eta_t = \tfrac{\eta_0}{2}\bigl(1 + \cos\tfrac{\pi(t - t_\textrm{w})}{T - t_\textrm{w}}\bigr)$.
  Small last steps let the weights settle.
- AlexNet's authors divided the rate by 10 by hand. The cosine schedule is
  a later default: nothing to tune but $\eta_0$ and $t_\textrm{w}$.
- Stepped after every minibatch; warmup over the first epoch of 10.
:::

::: {.slide title="Upsample 28×28 to 224×224 and Train for 10 Epochs"}
[Training]{.kicker}

@-alexnet-training

- Upsampling multiplies the pixels by 64 and adds no information; it
  keeps the architecture unchanged.
- Minibatch SGD, 128 images, peak learning rate 0.03; He-normal weights
  (variance $2/n_\textrm{in}$), zero biases.
:::

::: {.slide title="AlexNet Passes LeNet's Accuracy in a Third of the Epochs"}
[Training]{.kicker}

@!alexnet-training

- About 91% after 10 epochs; LeNet reaches 88–90% after 30.
- The runs differ in activation, initialization, and schedule as well as
  size. Fashion-MNIST cannot show what AlexNet's capacity is for.
:::

::: {.slide title="Training Changed the First Layer by About 4%, Smoothly"}
[Learned features]{.kicker}

@!alexnet-what-the-first-layer-learned-1

- The trained kernels still look like their random start; their change is
  smooth ramps and broad bars.
- No sharp edges, no color: the images were upsampled eightfold and have one
  channel.
:::

::: {.slide title="Check: How Large Is the 6400 × 4096 Matrix?"}
[Check]{.kicker}

How many megabytes in float32? What share of the parameters is it?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $6400 \cdot 4096 = 26.2$ M weights, $\times 4$ bytes $\approx$
**105 MB**: 56% of all 46.8 M parameters, in one layer that does 3% of
the arithmetic.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Learned features:** one network trained end to end cut ImageNet top-5
  error from 26% to 15%.
- **Enablers:** 1.2 M labeled images; GPUs (≈ $6.5 \times 10^{17}$
  operations in days); ReLU, dropout, augmentation.
- **Architecture:** LeNet's template with 5 convolutions and a 4096-wide
  dense head.
- **Cost:** convolutions do 95% of the 938 M multiply-adds; the head holds
  92% of the 46.8 M parameters.
- **Fashion-MNIST:** ≈ 91% in 10 epochs with warmup and cosine decay, a
  point or two above LeNet at 2,250× the arithmetic.

**Next:** blocks, and networks without the giant dense head.
:::
