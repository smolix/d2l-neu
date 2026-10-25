```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Convolutional Neural Networks (LeNet)
:label:`sec_lenet`

We can now combine convolution, pooling, and fully connected layers into a
complete image classifier. In our earlier experiments with image data, we applied
a linear model with softmax regression (:numref:`sec_softmax_scratch`)
and an MLP (:numref:`sec_mlp-implementation`)
to pictures of clothing in the Fashion-MNIST dataset.
Both models first flattened each image from a $28\times28$ matrix into a
$784$-dimensional vector and processed it with fully connected layers. Such
layers treat the pixels as an unordered list: shuffling the pixels in the same
way in every image would not change what the MLP can learn. After 30 epochs,
the MLP reached about 87% validation accuracy.
Convolutional layers instead preserve the spatial organization and share
parameters across locations, so they need far fewer parameters. This section
asks whether a network built from them also classifies the images more
accurately.

We answer it with *LeNet*, one of the early CNNs to demonstrate strong
performance on a practical computer vision task.
The model was introduced by (and named for) Yann LeCun,
then a researcher at AT&T Bell Labs,
for the purpose of recognizing handwritten digits in images :cite:`LeCun.Bottou.Bengio.ea.1998`.
This work represented the culmination
of a decade of research developing the technology;
LeCun's team published the first study to successfully
train CNNs via backpropagation :cite:`LeCun.Boser.Denker.ea.1989`.

LeNet matched the performance of support vector machines, then a dominant
approach to handwritten-digit recognition, with an error rate below 1% per
digit. Variants were deployed to read handwritten amounts on bank checks and
deposit slips. The deployment demonstrated that a trained convolutional model
could replace a substantial hand-engineered recognition pipeline.

```{.python .input #lenet-convolutional-neural-networks-lenet}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input #lenet-convolutional-neural-networks-lenet}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input #lenet-convolutional-neural-networks-lenet}
%%tab tensorflow
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #lenet-convolutional-neural-networks-lenet}
%%tab jax
from d2l import jax as d2l
from flax import nnx
from jax import numpy as jnp
```

## LeNet

At a high level, LeNet (LeNet-5) consists of two parts:
(i) a convolutional encoder consisting of two convolutional layers; and
(ii) a dense block consisting of three fully connected layers.
The architecture is summarized in :numref:`img_lenet`.

![Data flow in LeNet. The input is a handwritten digit, the output is a probability over 10 possible outcomes.](../img/lenet.svg)
:label:`img_lenet`

The basic units in each convolutional block
are a convolutional layer, a sigmoid activation function,
and a subsequent average pooling operation.
ReLUs and max-pooling were not part of LeNet-5 and had not yet become the
standard choices for trained CNNs.
Each convolutional layer uses a $5\times 5$ kernel.
These layers map spatially arranged inputs
to a number of two-dimensional feature maps, typically
increasing the number of channels.
The first convolutional layer has 6 output channels,
while the second has 16.
Each $2\times2$ pooling operation (stride 2)
reduces dimensionality by a factor of $4$ via spatial downsampling.
The convolutional block emits an output with shape (batch size, number
of channels, height, width) in the PyTorch/MXNet convention; TensorFlow
and JAX use the channels-last layout (batch, height, width, channels).

To pass the convolutional block's output to the dense block, we flatten each
example. This transforms the four-dimensional tensor into a matrix whose first
axis indexes examples and whose second axis contains the flattened features.
LeNet's dense block has three fully connected layers,
with 120, 84, and 10 outputs, respectively.
Because we are still performing classification,
the 10-dimensional output layer corresponds
to the number of possible output classes.

Modern frameworks express LeNet as a `Sequential` block containing the layers
described above. We use Xavier initialization, introduced in
:numref:`subsec_xavier`, in all four frameworks.

:begin_tab:`pytorch`
Lazy layers have no weights until the first minibatch fixes their shapes, so
the model cannot initialize them in its constructor. Before training, we
therefore call `apply_init` (:numref:`sec_lazy_init`), which runs one
minibatch through the model and then applies `init_cnn` to every layer.
:end_tab:

:begin_tab:`tensorflow`
Xavier initialization, under the name `glorot_uniform`, is the Keras default
for convolutional and dense layers, so the model needs no initialization code.
:end_tab:

```{.python .input #lenet-1}
%%tab pytorch
def init_cnn(module):  #@save
    """Initialize weights for CNNs."""
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        nn.init.xavier_uniform_(module.weight)
```

```{.python .input #lenet-2}
%%tab pytorch
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
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
```

```{.python .input #lenet-2}
%%tab mxnet
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(
            nn.Conv2D(channels=6, kernel_size=5, padding=2,
                      activation='sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2),
            nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            nn.AvgPool2D(pool_size=2, strides=2),
            nn.Dense(120, activation='sigmoid'),
            nn.Dense(84, activation='sigmoid'),
            nn.Dense(num_classes))
        self.net.initialize(init.Xavier())
```

```{.python .input #lenet-2}
%%tab tensorflow
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                   activation='sigmoid', padding='same'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                                   activation='sigmoid'),
            tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(120, activation='sigmoid'),
            tf.keras.layers.Dense(84, activation='sigmoid'),
            tf.keras.layers.Dense(num_classes)])
```

```{.python .input #lenet-2}
%%tab jax
class LeNet(d2l.Classifier):  #@save
    """The LeNet-5 model."""
    def __init__(self, lr=0.1, num_classes=10, kernel_init=None, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs', 'kernel_init'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        kernel_init = (nnx.initializers.xavier_uniform() if kernel_init is None
                       else kernel_init)
        self.net = nnx.Sequential(
            nnx.Conv(1, 6, kernel_size=(5, 5), padding='SAME',
                     kernel_init=kernel_init, rngs=rngs),
            nnx.sigmoid,
            lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            nnx.Conv(6, 16, kernel_size=(5, 5), padding='VALID',
                     kernel_init=kernel_init, rngs=rngs),
            nnx.sigmoid,
            lambda x: nnx.avg_pool(x, window_shape=(2, 2), strides=(2, 2)),
            lambda x: x.reshape((x.shape[0], -1)),  # flatten
            nnx.Linear(400, 120, kernel_init=kernel_init, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(120, 84, kernel_init=kernel_init, rngs=rngs),
            nnx.sigmoid,
            nnx.Linear(84, num_classes, kernel_init=kernel_init, rngs=rngs))
```

This is the classic teaching version of LeNet-5, with sigmoid activations
and average pooling. It is not an exact reproduction: the 1998 network used a
scaled tanh, trainable subsampling, partial connections between the two
convolutional layers, and radial-basis output units. Its alternating
convolution--pooling structure and its 6, 16, 120, and 84 channels and hidden
units are those of LeNet-5.

To check that the operations line up with :numref:`img_lenet_vert`, we pass a
single-channel (black and white) $28 \times 28$ image through the network and
print the output shape after each layer.

:begin_tab:`jax`
Because an NNX model already owns its initialized layers, we can pass an array
through the callables stored in `Sequential` and print each intermediate shape
directly.
:end_tab:

![Compressed notation for LeNet. Beside each arrow is the shape of the tensor it carries, as channels@height×width.](../img/lenet-vert.svg)
:label:`img_lenet_vert`

```{.python .input #lenet-3}
%%tab mxnet, pytorch
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
        
model = LeNet()
model.layer_summary((1, 1, 28, 28))
```

```{.python .input #lenet-3}
%%tab tensorflow
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = d2l.normal(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

```{.python .input #lenet-3}
%%tab jax
@d2l.add_to_class(d2l.Classifier)  #@save
def layer_summary(self, X_shape):
    X = jnp.zeros(X_shape)
    for layer in self.net.layers:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

model = LeNet()
model.layer_summary((1, 28, 28, 1))
```

Every convolutional and pooling layer except the first convolutional layer
reduces the height and width of the representation.
The first convolutional layer uses two pixels of padding
to compensate for the reduction in height and width
that would otherwise result from using a $5 \times 5$ kernel.
As an aside, the image size of $28 \times 28$ pixels in the original
MNIST OCR dataset is a result of *trimming* two pixel rows (and columns) from the
original scans that measured $32 \times 32$ pixels. This was done primarily to
save space (a 30% reduction) at a time when megabytes mattered.

In contrast, the second convolutional layer forgoes padding,
and thus the height and width are both reduced by four pixels.
As we go up the stack of layers,
the number of channels increases layer-over-layer
from 1 in the input to 6 after the first convolutional layer
and 16 after the second convolutional layer.
However, each pooling layer halves the height and width.
Finally, each fully connected layer reduces dimensionality,
ultimately emitting an output whose dimension
matches the number of classes.


## Training

We now train LeNet on Fashion-MNIST with the loss and the optimizer of the
MLP, cross-entropy minimized by minibatch stochastic gradient descent, and for
the same 30 epochs. Does the convolutional network reach a higher validation
accuracy than the MLP?

LeNet has fewer than a third of the MLP's parameters, yet it performs about
twice as many multiply-adds per image, because each convolution kernel is
applied at every position of its input. Using a GPU can accelerate training.
The `d2l.Trainer` class handles device placement and the training loop
(:numref:`subsec_device_aware_trainer`).

The learning rate needs more care than for the MLP. The derivative of the
sigmoid, $\sigma'(x) = \sigma(x)(1 - \sigma(x))$, is at most $1/4$, and the
gradient that reaches the first convolutional layer has passed backward
through four sigmoids, each of which multiplies it by at most $1/4$. With the
MLP's learning rate of 0.1, LeNet therefore learns slowly: its validation
accuracy stays near chance for the first two or three epochs, and after 10
epochs it is still below that of the MLP. We use a learning rate of 2.0
instead. Even then, the accuracy can sit near chance for the first epochs
before it climbs.

```{.python .input #lenet-training}
%%tab pytorch
trainer = d2l.Trainer(max_epochs=30, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=2.0)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)
trainer.fit(model, data)
```

```{.python .input #lenet-training}
%%tab mxnet, jax
trainer = d2l.Trainer(max_epochs=30, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = LeNet(lr=2.0)
trainer.fit(model, data)
```

```{.python .input #lenet-training}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=30)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = LeNet(lr=2.0)
    trainer.fit(model, data)
```

After 30 epochs, LeNet reaches a validation accuracy of about 89--90%. The MLP
of :numref:`sec_mlp-implementation` reached about 87% after the same number of
epochs, with more than three times as many parameters. A convolution kernel
learns a local pattern once and detects it at every position, whereas the MLP
must learn the pattern separately for each position at which it occurs.

## Summary

LeNet combines a convolutional encoder, in which the resolution shrinks while
the number of channels grows, with a small dense classification head. Trained
with the loss, the optimizer, and the 30 epochs of the MLP of
:numref:`sec_mlp-implementation`, it reaches about 89--90% validation accuracy
on Fashion-MNIST, against about 87% for the MLP, with fewer than a third of
the MLP's parameters. Its sigmoid activations need a learning rate twenty
times that of the MLP, because each sigmoid shrinks the backward gradient by a
factor of at least four. Modern frameworks also make LeNet's implementation
substantially shorter than the specialized systems required for the original
experiments :cite:`Bottou.Le-Cun.1988`.

The overall encoder--classifier organization remains common, although modern
networks replace most of LeNet's individual components:

:Component substitutions from LeNet to modern CNNs.
:label:`tab_lenet_modern`

| LeNet (this section) | Modern (2020s) | Effect of the change |
|:--|:--|:--|
| sigmoid activation | ReLU or GELU | gradients that survive depth instead of shrinking at every layer |
| average pooling | max-pooling or strided convolution | selective or learned local downsampling |
| no normalization | batch or layer normalization | stable activation scales, so much deeper stacks train |
| dense head on flattened features | global average pooling plus one linear layer | removes most of the parameters (the $400 \times 120$ block here) |
| Xavier initialization | He initialization | variance matched to the ReLU |

The next chapter examines these changes through AlexNet
(:numref:`sec_alexnet`), NiN (:numref:`sec_nin`), batch normalization
(:numref:`sec_batch_norm`), and ResNet (:numref:`sec_resnet`). He
initialization, introduced in :numref:`sec_init_param`, accompanies the use of
ReLU activations.

## Exercises

1. [code] **Modern components.** Retrain `LeNet` with the changes below,
   using the training settings of this section, and compare the
   validation accuracy of each variant after 30 epochs with that of
   `LeNet`.
    1. Replace average pooling with max-pooling.
    1. Replace the sigmoid activations with ReLU. If the loss diverges or
       the accuracy stays at chance, lower the learning rate until training
       succeeds, and report the learning rate you used.
    1. Make both changes together, and call the result the modernized
       LeNet. Which of the two changes accounts for most of the difference
       in accuracy?
    1. ● At initialization, measure for `LeNet` and for the modernized
       LeNet the norm of the gradient of the loss with respect to the first
       convolutional kernel, and the standard deviation of the logits
       across a minibatch. The derivation of Xavier initialization in
       :numref:`subsec_xavier` assumes inputs with zero mean and no
       nonlinearity. Use the mean of the sigmoid's output and its slope
       $\sigma'(0) = 1/4$ to explain your measurements and the learning
       rate that `LeNet` needs.
   :exlabel:`ex_lenet-modern-components`

1. [code] **Original MNIST.** Write an `MNIST` data class by adapting
   `d2l.FashionMNIST`, and train both `LeNet` and the MLP of
   :numref:`sec_mlp-implementation` on it, with the settings that each
   uses for Fashion-MNIST. :citet:`LeCun.Bottou.Bengio.ea.1998` report a
   test error rate below 1% for LeNet-5 on MNIST.
    1. Compare the test error rate of `LeNet` with this figure.
    1. For each dataset, compute the difference and the ratio of the test
       error rates of the two networks. :numref:`sec_fashion_mnist` states
       that MNIST makes differences among model classes less visible. Does
       each of the two measures support this statement?

1. [code] **Activation maps.** Train the modernized LeNet of
   problem :exref:`ex_lenet-modern-components` and extract the feature maps of its first and second
   convolutional layers, each taken after its ReLU.
    1. Display them for test images from two different classes, for
       example pullovers and coats.
    1. Measure the maximum activation of each of the two layers on
       Fashion-MNIST test images, on out-of-distribution photos such as a
       cat or a car, and on uniform random noise. Convert each photo to a
       $28 \times 28$ grayscale image, and keep every input in $[0, 1]$,
       the pixel range of `d2l.FashionMNIST`. Do the maxima for photos and
       noise fall inside or clearly outside the range observed on
       Fashion-MNIST test images?

1. [code] **Overfitting sanity check.** Before trusting a full training
   run, check that a network can memorize a small dataset. Fix a subset
   of 50 Fashion-MNIST training images and train on it with all 50
   images in one minibatch, for at most 500 gradient steps.
    1. For both `LeNet` and the modernized LeNet of problem :exref:`ex_lenet-modern-components`,
       record the number of steps until training accuracy exceeds 99%. If
       a network does not get there, diagnose which architectural or
       optimization choice is responsible.
    1. Name one bug that this check would catch and one that it would
       miss, and explain why the check cannot detect the second.

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2024/assignment2/),
    Convolutional Networks notebook.*

1. **Equivariance versus augmentation.** A stride-1 convolution on an
   infinite grid is exactly translation equivariant.
    1. Go through the layers of `LeNet` in order. Name each component that
       breaks exact translation equivariance of the feature maps, and the
       components that prevent the final prediction from being
       translation invariant.
    1. Explain why training `LeNet` on translated copies of its training
       images can improve test accuracy even though its convolutional
       layers share weights across all locations.

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6.*

1. [code] **The dense head.** ● :numref:`tab_lenet_modern` states that
   replacing the dense head with global average pooling removes most of
   LeNet's parameters.
    1. Using the weight count of :numref:`sec_channels`, one bias per
       output channel or unit, and the layer shapes of
       :numref:`img_lenet_vert`, count the parameters of the two
       convolutional layers combined, of the $400 \times 120$ dense layer,
       and of all of `LeNet`. What fraction of the total do the
       convolutional layers account for, and what fraction the
       $400 \times 120$ layer?
    1. Replace the entire dense head, from the flatten through the output
       layer, by global average pooling followed by a single linear layer
       from 16 to 10 units. How many parameters does the network now
       have? Before training, predict whether the test accuracy changes
       by more than one percentage point relative to `LeNet`; then
       retrain and compare.
    1. Instead connect the flattened 400-dimensional convolutional output
       directly to a ten-way linear output layer. Retrain, and compare the
       accuracy of the two reduced heads with each other and with your
       prediction. What does the comparison say about where LeNet's
       capacity is used?

    *Adapted from Michael Nielsen,
    [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html),
    Chapter 6.*

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/73)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/74)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/275)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/18000)
:end_tab:

<!-- slides -->

::: {.slide title="The MLP ignores which pixels are neighbors"}
The MLP of the previous chapter flattens each 28×28 Fashion-MNIST image into
784 numbers. **Shuffling the pixels the same way in every image would not
change what it can learn.**

- 256 hidden units, about 204k parameters
- 30 epochs of SGD → **≈ 87%** validation accuracy

Can a network that keeps the 2-D layout do better with fewer parameters?
:::

::: {.slide title="LeNet set the CNN template"}
**LeNet-5** (Yann LeCun et al., 1989; deployed in the 1990s) recognized
handwritten digits on bank checks.

Its organization, a **convolutional encoder** in which spatial dimensions
shrink and channels grow followed by a **dense head**, influenced many later
CNN architectures.
:::

::: {.slide title="Spatial size shrinks while channels grow"}
![LeNet data flow on a 28×28 handwritten digit.](../img/lenet.svg){width=92%}
:::

::: {.slide title="Two conv blocks, three dense layers"}
- Conv1: 1→6 channels, 5×5 kernel, padding 2 (28→28), sigmoid
- AvgPool: 2×2, stride 2 → 14×14
- Conv2: 6→16 channels, 5×5, no padding → 10×10, sigmoid
- AvgPool: 2×2, stride 2 → 5×5
- Flatten → 16·5·5 = 400 → 120 → 84 → 10 logits

LeNet-5 itself used a scaled tanh; the sigmoid σ(x) = ½(1 + tanh(x/2)) is
the same S-curve, shifted to positive outputs and flatter at 0.
:::

::: {.slide title="Pooling halves the resolution; channels go 1 → 6 → 16"}
![Shapes as channels@height×width.](../img/lenet-vert.svg){width=23%}
:::

::: {.slide title="Pyramid shape, dense bottleneck"}
- **Pyramid shape:** spatial dimensions halve at each pooling layer while the
  number of channels increases. Many later CNNs retain this pattern.
- **The bottleneck is the flatten:** `400 × 120 = 48000`
  weights from conv block to first dense layer. Modern
  CNNs replace the dense stack with *global average
  pooling*, which is much cheaper.
:::

::: {.slide title="LeNet in code"}
@lenet-2
:::

::: {.slide title="Xavier init, applied after the dry run" only="pytorch"}
Lazy layers get their weights from the first minibatch, so `apply_init`
runs one minibatch through the model and then applies `init_cnn`:

@lenet-1
:::

::: {.slide title="Tracing shapes through the network"}
A dummy 28×28 single-channel input, shape after each layer
(28→28→14→10→5→flatten→120→84→10, as in the diagram):

@!lenet-3
:::

::: {.slide title="Sigmoid layers need a 20× larger learning rate"}
σ′(x) = σ(x)(1 − σ(x)) ≤ ¼, and the gradient reaching Conv1 passes
backward through **four** sigmoids:

- each multiplies it by at most ¼
- at the MLP's learning rate 0.1, LeNet sits near chance for 2–3 epochs
  and trails the MLP after 10

**Use learning rate 2.0.**
:::

::: {.slide title="Same loss, optimizer, and 30 epochs as the MLP"}
Cross-entropy and SGD; only the model and the learning rate change:

@lenet-training
:::

::: {.slide title="LeNet beats the MLP with a third of the parameters"}
| | MLP (256 hidden) | LeNet |
|---|---|---|
| parameters | ≈ 204k | ≈ 62k |
| multiply-adds per image | ≈ 0.20M | ≈ 0.42M |
| learning rate | 0.1 | 2.0 |
| validation accuracy, 30 epochs | ≈ 87% | ≈ 89–90% |

**A kernel learns a local pattern once and detects it at every position.**
:::

::: {.slide title="Check: fewer parameters, more arithmetic"}
LeNet has a third of the MLP's parameters but does twice the multiply-adds
per image. Why?

. . .

Each convolution weight is reused at every output position: conv2's
16·6·25 = 2,400 weights × 10·10 positions = 240,000 multiply-adds. A dense
weight is used once per image.
:::

::: {.slide title="Thirty years changed most components, not the template"}
| LeNet (this deck) | Modern (2020s) |
|---|---|
| sigmoid activation | ReLU / GELU |
| average pooling | max pool / strided conv |
| no normalization | BatchNorm / LayerNorm |
| dense head | global average pool + 1 linear |
| Xavier init | He init |
| 5 layers, ~60k params | 50+ layers, millions of params |

Each substitution is a section of the next chapter, Modern CNNs. The
*convolutional encoder + head* organization stays.
:::

::: {.slide title="Recap"}
- LeNet-5 demonstrated a CNN in a deployed recognition system.
- Template: conv encoder (spatial ↓, channels ↑) → flatten → dense head.
- Sigmoids shrink gradients by ≥ 4× per layer: LeNet needs learning rate
  2.0, twenty times the MLP's.
- Then, in the MLP's 30 epochs: ≈ 89–90% vs ≈ 87%, with a third of the
  parameters.
- The next chapter swaps every component for its modern
  equivalent and goes much deeper.
:::
