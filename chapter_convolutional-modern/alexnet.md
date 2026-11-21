```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# AlexNet and Learned Image Representations
:label:`sec_alexnet`

In :numref:`sec_lenet`, LeNet trained every stage of its pipeline—both convolutional filters and final classifier—end to end from raw pixels. Yet for two decades after LeCun's 1989 results, mainstream computer vision split feature extraction from classification: hand-engineered descriptors such as SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, HOG :cite:`Dalal.Triggs.2005`, and bags of visual words :cite:`Sivic.Zisserman.2003` mapped pixels into fixed feature vectors that fed a linear or kernel classifier. Because the training loss updated only the classifier and never the features, progress relied on human feature engineering.

The watershed came at the 2012 ImageNet challenge (1,000 object categories evaluated by *top-5 error*, the fraction of validation images whose true label falls outside the model's top five predictions). *AlexNet* :cite:`Krizhevsky.Sutskever.Hinton.2012` trained a single eight-layer convolutional network end to end on raw pixels, slashing top-5 error from 26.2% (for the best hand-engineered pipeline) to 15.3% in a single leap. Architecturally, AlexNet is LeNet scaled up, so its sudden triumph points straight to three catalysts that converged in 2012: a million-image labeled dataset (ImageNet), GPUs fast enough to train tens of millions of weights in days rather than months, and key modeling practices—ReLU activations, dropout, and data augmentation—that made deep networks trainable and resistant to memorization.

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

What did backpropagation discover on ImageNet that hand-crafted descriptors missed? :numref:`fig_filters` shows the 96 learned $11 \times 11$ kernels of AlexNet's first convolutional layer: oriented edge and frequency detectors alongside color-opponent blobs, found purely by minimizing classification loss and composed by deeper layers into textures, parts, and object categories.

![The 96 kernels of the first convolutional layer of AlexNet, learned on ImageNet. Reproduction courtesy of :citet:`Krizhevsky.Sutskever.Hinton.2012`.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

### Data and Hardware

Models with 60 million parameters quickly memorize small benchmarks such as PASCAL VOC (~$10^4$ images). The 2012 ImageNet training split :cite:`Deng.Dong.Socher.ea.2009,russakovsky2015imagenet` supplied 1.2 million labeled photographs across 1,000 categories, and AlexNet multiplied effective sample diversity via data augmentation—extracting random $224 \times 224$ crops and horizontal reflections from $256 \times 226$ images (up to $32 \cdot 32 \cdot 2 = 2{,}048$ views per photograph).

Hardware supplied the second half of the breakthrough. Convolving a filter across every spatial location and channel executes millions of independent multiply-adds—the exact workload graphics processing units (GPUs) were built to run across hundreds of parallel cores connected to high-bandwidth memory. Each of the two NVIDIA GTX 580 GPUs used for AlexNet delivered ~1.6 TFLOPS and 192 GB/s of memory bandwidth, versus ~0.2 TFLOPS and 20 GB/s for a contemporary quad-core CPU. One forward pass of AlexNet on a $224 \times 224$ RGB image costs ~$10^9$ multiply-adds; counting a backward pass at twice the forward pass and 2 FLOPs per multiply-add, 90 epochs over 1.2 million images require

$$10^9 \cdot 3 \cdot 2 \cdot 1.2 \times 10^6 \cdot 90 \approx 6.5 \times 10^{17} \text{ FLOPs}$$

—five to six days on two GTX 580s using Krizhevsky's custom CUDA kernels ([cuda-convnet](https://code.google.com/archive/p/cuda-convnet/)), compared with well over a month on a CPU.

## The AlexNet Architecture

:numref:`fig_alexnet` compares LeNet (:numref:`sec_lenet`) side by side with
the AlexNet variant we train on Fashion-MNIST below. Both follow the same
high-level design—a stack of convolutional stages feeding into a dense
classification head—but AlexNet scales up every dimension and modernizes three
critical components: the activation function, pooling, and regularization.

![LeNet as built in :numref:`sec_lenet` (left) and AlexNet as built in this section (right), both for single-channel Fashion-MNIST images and ten classes; the input is at the bottom. AlexNet is deeper and wider, uses ReLU activations and max-pooling, and regularizes its dense layers with dropout.](../img/arch-alexnet.svg)
:label:`fig_alexnet`

### Depth, Width, ReLU, and Dropout

AlexNet expands LeNet from five parameterized layers to eight—five convolutions and three dense layers (:numref:`fig_alexnet`). Because $224 \times 224$ inputs are eight times wider than $28 \times 28$ thumbnails, layer 1 uses an $11 \times 11$ kernel with stride 4, layer 2 uses a $5 \times 5$ kernel, and layers 3--5 use $3 \times 3$ kernels, with overlapping $3 \times 3$ stride-2 max-pooling after the first, second, and fifth convolutions. Channel widths grow from 96 to 384 (vs. 6 and 16 in LeNet), and the two hidden dense layers widen to 4096 units each (vs. 120 and 84). Our implementation omits two historical 2012 artifacts: splitting layers across two 3 GB GPUs and local response normalization (superseded by batch normalization in :numref:`sec_batch_norm`).

Three modeling choices made this larger network effective:

1. **ReLU activations:** Replacing saturating sigmoids (whose derivative $\le 1/4$ attenuates deep gradients by $4^{-K}$) with $\textrm{ReLU}(x) = \max(x, 0)$ keeps derivatives at 1 on active units, reaching 25% CIFAR-10 training error six times faster than $\tanh$ :cite:`Krizhevsky.Sutskever.Hinton.2012`.
2. **Dropout and augmentation:** Applying dropout ($p = 0.5$, :numref:`sec_dropout`) after both 4096-unit dense layers prevents the classifier head from memorizing the training set.
3. **He initialization:** Because ReLU zeroes out roughly half its inputs, we draw weights from $\mathcal{N}(0, 2/n_{\textrm{in}})$ (:numref:`subsec_he_init`, with $n_{\textrm{in}} = c_{\textrm{i}} k^2$ for convolutions) and initialize biases to zero.

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

1. **Memory and compute ledger.** Extend :numref:`tab_alexnet_cost` to the memory that `AlexNet` needs in 32-bit floating point.
    1. Compute the number of activations that the forward pass of one $224 \times 224$ image stores, layer by layer, counting the output of every convolutional, pooling, and fully connected layer. Where is activation memory concentrated, and where is parameter memory concentrated?
    1. During training, parameters, their gradients, and stored activations for a whole minibatch must fit in memory. At what batch size does activation memory exceed the memory for parameters and gradients?
    1. Repeat the parameter count of :numref:`tab_alexnet_cost` for the ImageNet version (3 input channels, 1000 classes, $227 \times 227$ inputs so the final convolutional stage is $6 \times 6$ instead of $5 \times 5$), and compare your total against the 60 million parameters reported by :citet:`Krizhevsky.Sutskever.Hinton.2012`.
   :exlabel:`ex_alexnet-memory-compute-ledger`

1. **Receptive fields of the early layers.** A stack of layers with kernel sizes $k_i$ and strides $s_i$ gives its top unit a receptive field of side $r = 1 + \sum_{i} (k_i - 1) \prod_{j<i} s_j$ (:numref:`sec_padding`). Counting each max-pooling layer with its own kernel size and stride, compute the receptive field of one unit after each of the five convolutional layers of `AlexNet`. What fraction of the $224 \times 224$ input side does each receptive field span, and at which layer does a unit first see the entire image?

1. **Arithmetic intensity.** ● The *arithmetic intensity* of a layer is the number of multiply-adds it performs per byte of parameters read from memory. Using :numref:`tab_alexnet_cost`:
    1. Compute the arithmetic intensity of every layer of `AlexNet` for a batch of 1 image and for a batch of 128 images (assuming each parameter is read once per batch).
    1. Consider an accelerator that performs 10 multiply-adds in the time it reads one byte from memory, where each layer takes the maximum of its compute time and parameter-read time. Does doubling compute throughput or doubling memory bandwidth speed up single-image inference ($B = 1$) more? Which speeds up training at $B = 128$ more?

1. [code] **Batch size, throughput, and memory.** ● Train `AlexNet` as in this section while varying only the batch size.
    1. Using problem :exref:`ex_alexnet-memory-compute-ledger`, predict the largest training batch size that fits in your GPU's memory.
    1. Starting from batch size 32 and doubling, find the largest batch size that trains without running out of memory and explain the main source of any gap against your prediction.
    1. Measure training throughput (images/second) across at least four batch sizes up to that limit. Beyond which batch size does throughput plateau, and why?

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
