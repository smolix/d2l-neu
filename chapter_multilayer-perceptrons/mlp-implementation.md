```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Implementation of Multilayer Perceptrons
:label:`sec_mlp-implementation`

Softmax regression scores each Fashion-MNIST class with a weighted sum of
the 784 pixel values of an image. With 7,850 parameters it reaches about 83%
validation accuracy after ten epochs, and most of its errors confuse garments
of similar outline, such as shirts, pullovers, and coats
(:numref:`sec_softmax_scratch`). The multilayer perceptron (MLP) of
:numref:`sec_mlp` first computes features of the pixels in a hidden layer
and then classifies these features with a linear layer. With 256 hidden
units it has 203,530 parameters and performs 203,264 multiply-adds per
image, 26 times as many as softmax regression. Whether the hidden layer
makes the classifier more accurate on real images, and by how much, is a
question for experiment.

We implement the MLP twice and train it as we trained softmax regression.
The first implementation writes the four parameter arrays, the activation
function, and the forward pass with tensor operations, and it checks the
untrained model before any training time is spent. The second assembles the
same model from the framework's layers. The data, the cross-entropy loss,
minibatch stochastic gradient descent (SGD), and the `Trainer` stay as they
were, so any change in accuracy comes from the hidden layer. Finally we
compare the MLP with softmax regression trained under the same settings, in
accuracy and in training time. Automatic differentiation computes
all gradients in this section; the next section shows how.

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, np as mnp, npx
from mxnet.gluon import nn
import time
npx.set_np()
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab pytorch
from d2l import torch as d2l
import time
import torch
from torch import nn
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab tensorflow
from d2l import tensorflow as d2l
import time
import tensorflow as tf
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab jax
from d2l import jax as d2l
from flax import nnx
from jax import numpy as jnp
import time
```

## Implementation from Scratch

### The Parameters

Each $28 \times 28$ image becomes a vector of $d = 784$ pixel values, and
there are $q = 10$ classes. We give the hidden layer $h = 256$ units
(:numref:`fig_mdl-mlp-arch`). The width is a hyperparameter, like the
learning rate: training does not choose it, and Exercise
:exref:`ex_mlp-implementation-width` compares other widths. In the notation
of :numref:`sec_mlp`, each layer has a weight matrix with one row per input
and one column per output, and a bias vector with one entry per output:

$$\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times 256}, \quad \mathbf{b}^{(1)} \in \mathbb{R}^{256}, \quad \mathbf{W}^{(2)} \in \mathbb{R}^{256 \times 10}, \quad \mathbf{b}^{(2)} \in \mathbb{R}^{10}.$$

Together these hold $785 \cdot 256 + 257 \cdot 10 = 203{,}530$ numbers, and
all but 2,570 of them belong to the hidden layer.

![The MLP of this section. A minibatch of $n$ images is flattened to an $n \times 784$ matrix $\mathbf{X}$, an affine map followed by the ReLU computes the $n \times 256$ hidden representation $\mathbf{H}$, and a second affine map computes the $n \times 10$ logits $\mathbf{O}$.](../img/mdl-mlp-arch.svg)
:label:`fig_mdl-mlp-arch`

As for softmax regression, we draw the weights from a normal distribution
with mean 0 and standard deviation 0.01 and set the biases to 0. The
weights must be random: hidden units that start with equal weights receive
equal gradients and stay identical throughout training, as
:numref:`sec_numerical_stability` shows. Small weights keep the
logits close to 0 at first, so the untrained model gives every class a
probability close to 1/10.

We register the four arrays as parameters of the module with the
framework's parameter type, as in :numref:`sec_softmax_scratch`. The
default optimizer that `Classifier` inherits from `Module`, minibatch SGD at
the learning rate `lr`, finds registered parameters by itself, so the model
needs no `configure_optimizers` of its own. The argument `std` sets the
standard deviation of the initial weights.

:begin_tab:`pytorch`
`nn.Parameter` registers each tensor with the module, so
`model.parameters()` lists `W1`, `b1`, `W2`, and `b2`, and the default
optimizer, `torch.optim.SGD`, updates them.
:end_tab:

:begin_tab:`mxnet`
A Gluon `Parameter` assigned as an attribute of a block is registered with
it, so `model.collect_params()` lists the four arrays, and the default
optimizer, Gluon's `Trainer` with the `'sgd'` update, updates them.
`initialize` allocates their values, which `data()` returns.
:end_tab:

:begin_tab:`tensorflow`
`add_weight` creates each variable and registers it with the Keras model, so
`model.trainable_variables` lists the four arrays, and the default
optimizer, Keras's `SGD`, updates them. A Keras initializer object returns
the same values every time it is called. With one initializer shared by
both weight matrices, $\mathbf{W}^{(2)}$ would repeat the first entries of
$\mathbf{W}^{(1)}$, so the method `normal` creates a new initializer for
each of them.
:end_tab:

:begin_tab:`jax`
`nnx.Param` marks each array as a trainable parameter, so
`nnx.state(model)` contains the four arrays, and the default optimizer,
`optax.sgd`, updates them. As in :numref:`sec_softmax_scratch`, the argument
`rngs` supplies the random key for the initialization; without it, the
model draws a new one on every run.
:end_tab:

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab pytorch
class MLPScratch(d2l.Classifier):
    """An MLP with one hidden layer, implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, std=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = nn.Parameter(d2l.normal(0, std, (num_inputs, num_hiddens)))
        self.b1 = nn.Parameter(d2l.zeros(num_hiddens))
        self.W2 = nn.Parameter(d2l.normal(0, std, (num_hiddens, num_outputs)))
        self.b2 = nn.Parameter(d2l.zeros(num_outputs))
```

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab mxnet
class MLPScratch(d2l.Classifier):
    """An MLP with one hidden layer, implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, std=0.01):
        super().__init__()
        self.save_hyperparameters()
        normal, zero = init.Normal(std), init.Zero()
        self.W1 = gluon.Parameter('W1', shape=(num_inputs, num_hiddens),
                                  init=normal)
        self.b1 = gluon.Parameter('b1', shape=(num_hiddens,), init=zero)
        self.W2 = gluon.Parameter('W2', shape=(num_hiddens, num_outputs),
                                  init=normal)
        self.b2 = gluon.Parameter('b2', shape=(num_outputs,), init=zero)
        self.initialize()
```

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab tensorflow
class MLPScratch(d2l.Classifier):
    """An MLP with one hidden layer, implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, std=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.W1 = self.add_weight(name='W1', shape=(num_inputs, num_hiddens),
                                  initializer=self.normal())
        self.b1 = self.add_weight(name='b1', shape=(num_hiddens,),
                                  initializer='zeros')
        self.W2 = self.add_weight(name='W2', shape=(num_hiddens, num_outputs),
                                  initializer=self.normal())
        self.b2 = self.add_weight(name='b2', shape=(num_outputs,),
                                  initializer='zeros')

    def normal(self):
        # A new initializer for each weight matrix: one initializer object
        # returns the same values every time it is called
        return tf.keras.initializers.RandomNormal(stddev=self.std)
```

```{.python .input #mlp-implementation-initializing-model-parameters}
%%tab jax
class MLPScratch(d2l.Classifier):
    """An MLP with one hidden layer, implemented from scratch."""
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, std=0.01,
                 rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.W1 = nnx.Param(
            rngs.params.normal((num_inputs, num_hiddens)) * std)
        self.b1 = nnx.Param(jnp.zeros(num_hiddens))
        self.W2 = nnx.Param(
            rngs.params.normal((num_hiddens, num_outputs)) * std)
        self.b2 = nnx.Param(jnp.zeros(num_outputs))
```

### The Forward Pass

We write the activation function ourselves instead of calling the
framework's built-in function. The ReLU keeps the positive entries and
zeros the others, so it multiplies each entry $z$ by 1 if $z > 0$ and by 0
otherwise. In this form automatic differentiation gives the derivative 0 at
$z = 0$, as for the built-in ReLU of every framework
(:numref:`subsec_activation-functions`); for $\max(z, 0)$ it would give
$1/2$ or 1 there, depending on the framework.

```{.python .input #mlp-implementation-model-1}
%%tab mxnet
def relu(X):
    return X * (X > 0)
```

```{.python .input #mlp-implementation-model-1}
%%tab pytorch
def relu(X):
    return X * (X > 0)
```

```{.python .input #mlp-implementation-model-1}
%%tab tensorflow
def relu(X):
    return X * tf.cast(X > 0, X.dtype)
```

```{.python .input #mlp-implementation-model-1}
%%tab jax
def relu(X):
    return X * (X > 0)
```

For a minibatch of $n$ images, flattened into the rows of
$\mathbf{X} \in \mathbb{R}^{n \times 784}$, the model computes

$$\mathbf{H} = \mathrm{ReLU}\left(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}\right), \qquad \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.$$

Row $i$ of $\mathbf{H} \in \mathbb{R}^{n \times 256}$ is the hidden
representation of image $i$, and row $i$ of
$\mathbf{O} \in \mathbb{R}^{n \times 10}$ holds its logits; the biases are
added to every row by broadcasting. The method `hidden` computes
$\mathbf{H}$, and `forward` computes $\mathbf{O}$ from it. The model returns
logits, not probabilities. The loss that every `Classifier` inherits from
:numref:`sec_softmax_concise` takes the logits and computes the
cross-entropy from them without forming the probabilities, so that no
exponential overflows. Automatic differentiation provides the gradients of
all four parameter arrays, and the model needs no backward pass of its own.

:begin_tab:`mxnet`
The MXNet version reads the values of the parameters with `data()`.
:end_tab:

```{.python .input #mlp-implementation-model-2}
%%tab pytorch, tensorflow, jax
@d2l.add_to_class(MLPScratch)
def hidden(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))  # One flattened image per row
    return relu(d2l.matmul(X, self.W1) + self.b1)

@d2l.add_to_class(MLPScratch)
def forward(self, X):
    return d2l.matmul(self.hidden(X), self.W2) + self.b2  # The logits
```

```{.python .input #mlp-implementation-model-2}
%%tab mxnet
@d2l.add_to_class(MLPScratch)
def hidden(self, X):
    X = d2l.reshape(X, (-1, self.num_inputs))  # One flattened image per row
    return relu(d2l.matmul(X, self.W1.data()) + self.b1.data())

@d2l.add_to_class(MLPScratch)
def forward(self, X):
    return d2l.matmul(self.hidden(X), self.W2.data()) + self.b2.data()
```

### Checks before Training

Training takes minutes, and a mistake in the model may only show as poor
accuracy at the end. A forward pass through the untrained model on a few
images tests three predictions in a fraction of a second. First, each image
should have a hidden representation of 256 entries and one logit per class.
Second, since the logits start close to 0, the loss should be close to
$\log 10 \approx 2.30$, the cross-entropy of a prediction that gives every
class the probability 1/10. Third, about half of the hidden activations
should be positive. With zero biases, the pre-activation of a hidden unit is
a sum of pixel values times weights drawn symmetrically around 0, so it is
positive or negative with equal probability, and the ReLU sets the negative
ones to 0.

```{.python .input #mlp-implementation-checks-before-training}
data = d2l.FashionMNIST(batch_size=256)
model = MLPScratch(num_inputs=784, num_outputs=10, num_hiddens=256, lr=0.1)
X, y = next(iter(data.train_dataloader()))
X, y = X[:8], y[:8]  # Eight images suffice for the checks
H, O = model.hidden(X), model(X)
active = d2l.reduce_mean(d2l.astype(H > 0, d2l.float32))
print('H:', H.shape, 'O:', O.shape)
print(f'loss {d2l.numpy(model.loss(O, y)):.3f}, '
      f'fraction of active units {d2l.numpy(active):.2f}')
```

The shapes are $8 \times 256$ and $8 \times 10$, the loss is close to 2.30,
and about half of the hidden units are active. A weight matrix with its axes
swapped would stop the forward pass with an error, a wrong number of inputs
would show up in the shapes, and a loss far from 2.30 would point to weights
that are too large or to a mistake in the loss.

### Training

We train with the minibatch size and the learning rate of softmax regression
in :numref:`sec_softmax_scratch`, 256 images and 0.1, but for 30 epochs
instead of 10. The comparison at the end of this section shows why the MLP
needs the longer run. With `num_gpus=1` the `Trainer` trains on a GPU if the
machine has one and on the CPU otherwise (:numref:`sec_use_gpu`).

```{.python .input #mlp-implementation-training}
trainer = d2l.Trainer(max_epochs=30, num_gpus=1)
trainer.fit(model, data)
```

The training loss falls over all 30 epochs. After the first few epochs the
validation loss lies above it and falls more slowly, and the validation
accuracy climbs to about 87%. From one epoch to the next the accuracy moves
by about a point, and a single epoch can fall several points below its
neighbors, since minibatch SGD at a constant learning rate keeps moving the
parameters.

## Concise Implementation

A fully connected layer of the framework holds a weight matrix and a bias
vector and computes the affine map of one layer, as in the concise softmax
regression of :numref:`sec_softmax_concise`. The concise MLP puts four
operations into the framework's sequential container, which applies them in
order: flattening, a fully connected layer with 256 outputs, the ReLU, and a
fully connected layer with 10 outputs. Compared with `SoftmaxRegression`, it
adds the hidden layer and the ReLU. We initialize it like the scratch model,
with weights of standard deviation 0.01 and zero biases, in every framework.
Without this choice each framework would draw the weights from a default
distribution of its own. The four defaults differ, so the scratch and the
concise model would start from different scales.

:begin_tab:`mxnet`
The `Dense` layer flattens its input itself, and its argument `activation`
applies the ReLU to its output. `initialize` draws the weights of both
layers from `init.Normal`, whose argument `sigma` is the standard deviation.
The biases start at 0.
:end_tab:

:begin_tab:`pytorch`
`nn.LazyLinear` creates its weights at the first forward pass, once the
number of inputs is known. As for `SoftmaxRegression`, the `Trainer` then
calls the model's method `weight_init` on every submodule, and ours draws
the weights of both linear layers and sets their biases to 0.
:end_tab:

:begin_tab:`tensorflow`
The argument `activation` of the first `Dense` layer applies the ReLU to its
output. Each `Dense` layer gets an initializer of its own, for the reason
given for the scratch model. The biases start at 0.
:end_tab:

:begin_tab:`jax`
`nnx.Sequential` applies its layers in order, and `nnx.relu` enters the
sequence as a plain function. NNX layers create their parameters when they
are constructed, so each `nnx.Linear` needs its number of inputs up front.
`kernel_init` sets the distribution of the weights, and the biases start at
0.
:end_tab:

```{.python .input #mlp-implementation-model-2-2}
%%tab pytorch
class MLP(d2l.Classifier):
    """An MLP with one hidden layer, built from framework layers."""
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(), nn.LazyLinear(num_hiddens),
                                 nn.ReLU(), nn.LazyLinear(num_outputs))

    def weight_init(self, module):
        # The Trainer calls this once the lazy layers know their shapes
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)
```

```{.python .input #mlp-implementation-model-2-2}
%%tab mxnet
class MLP(d2l.Classifier):
    """An MLP with one hidden layer, built from framework layers."""
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens, activation='relu'),
                     nn.Dense(num_outputs))
        self.net.initialize(init.Normal(sigma=0.01))
```

```{.python .input #mlp-implementation-model-2-2}
%%tab tensorflow
class MLP(d2l.Classifier):
    """An MLP with one hidden layer, built from framework layers."""
    def __init__(self, num_outputs, num_hiddens, lr):
        super().__init__()
        self.save_hyperparameters()
        normal = tf.keras.initializers.RandomNormal
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens, activation='relu',
                                  kernel_initializer=normal(stddev=0.01)),
            tf.keras.layers.Dense(num_outputs,
                                  kernel_initializer=normal(stddev=0.01))])
```

```{.python .input #mlp-implementation-model-2-2}
%%tab jax
class MLP(d2l.Classifier):
    """An MLP with one hidden layer, built from framework layers."""
    def __init__(self, num_outputs, num_hiddens, lr, num_inputs=784,
                 rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        normal = nnx.initializers.normal(0.01)
        self.net = nnx.Sequential(
            nnx.Linear(num_inputs, num_hiddens, kernel_init=normal,
                       rngs=rngs),
            nnx.relu,
            nnx.Linear(num_hiddens, num_outputs, kernel_init=normal,
                       rngs=rngs))

    def forward(self, X):
        return self.net(X.reshape((X.shape[0], -1)))  # Flatten each image
```

:begin_tab:`mxnet, pytorch, tensorflow`
The model needs no `forward` method of its own. It inherits the `forward`
of `Module` (:numref:`subsec_oo-design-models`), which calls `self.net(X)`,
and the sequential container defines the computation.
:numref:`subsec_model-construction-sequential` looks inside such a
container.
:end_tab:

:begin_tab:`jax`
Its `forward` flattens each image and passes the result through the
sequence of the hidden layer, the ReLU, and the output layer.
:end_tab:

The scratch model spelled out every array and every operation; here the
layers hold the parameters and the container fixes the order of the
operations. With a deeper network, or one whose layers we want to reorder,
the concise form saves us from naming and registering each array by hand.

## Comparison with Softmax Regression

### Accuracy

We now train the concise MLP and, for comparison, the concise softmax
regression of :numref:`sec_softmax_concise`, `d2l.SoftmaxRegression`, with
the same trainer: 30 epochs, minibatches of 256, and learning rate 0.1. All
three models start from weights of standard deviation 0.01 and zero biases.
We time each run and keep its validation accuracy after every epoch from
`trainer.history`. The trainer's plot is switched off, and we draw the
validation curves of the three models, the scratch MLP included, in one plot
instead.

```{.python .input #mlp-implementation-accuracy-1}
curves, seconds = {'MLP from scratch': trainer.history['val_acc']}, {}
trainer.board.display = False
models = {'concise MLP': MLP(num_outputs=10, num_hiddens=256, lr=0.1),
          'softmax regression': d2l.SoftmaxRegression(num_outputs=10, lr=0.1)}
for name, model in models.items():
    start = time.perf_counter()
    trainer.fit(model, data)
    seconds[name] = (time.perf_counter() - start) / trainer.max_epochs
    curves[name] = trainer.history['val_acc']
```

```{.python .input #mlp-implementation-accuracy-2}
d2l.plot(list(range(1, trainer.max_epochs + 1)), list(curves.values()),
         'epoch', 'validation accuracy', legend=list(curves))
```

```{.python .input #mlp-implementation-accuracy-3}
print('validation accuracy averaged over the last 5 epochs')
for name, acc in curves.items():  # A single epoch can land a few points low
    print(f'  {name}: {sum(acc[-5:]) / 5:.3f}')
```

The scratch and the concise MLP follow the same curve up to the variation
from run to run: they are the same model, written in two ways. Both start
more slowly than softmax regression and usually trail it by a few points
after the first epoch. With weights of standard deviation 0.01, the hidden activations
are small, and every gradient that reaches $\mathbf{W}^{(1)}$ is multiplied
by entries of $\mathbf{W}^{(2)}$, whose standard deviation is 0.01 as well,
so at first the hidden layer learns slowly. After ten epochs, the training
length of :numref:`sec_softmax_scratch`, the MLP is ahead by about a point
on average, within its variation from epoch to epoch. Since any single
epoch can land a few points low, we compare the models by their validation
accuracy averaged over the last five epochs: softmax regression levels off at
about 84%, while the MLP climbs to about 87%. The hidden layer removes about
a fifth of the errors of softmax regression.

### Parameters, Computation, and Time

The following table collects what the hidden layer adds. The multiply-adds
are those of the forward pass for one image, $784 \cdot 10 = 7{,}840$ for
softmax regression and $784 \cdot 256 + 256 \cdot 10 = 203{,}264$ for the
MLP.

| model | parameters | multiply-adds per image | validation accuracy, epochs 26–30 |
|:--|--:|--:|--:|
| softmax regression | 7,850 | 7,840 | about 84% |
| MLP, 256 hidden units | 203,530 | 203,264 | about 87% |

The hidden layer multiplies the parameters and the arithmetic by 26. We
compare the measured times per epoch:

```{.python .input #mlp-implementation-parameters-computation-and-time}
for name, s in seconds.items():
    print(f'{name}: {s:.1f} s per epoch')
```

On a GPU the two times are about the same, although the MLP does 26 times
the arithmetic. For models this small, nearly all of an epoch goes to
reading the images and to the fixed cost of every step, which do not grow
with the width of the model; Exercise :exref:`ex_mlp-implementation-width`
measures how much.

We can now write an MLP with tensor operations or with framework layers,
check it before training, and train it with the loop of softmax regression.
On Fashion-MNIST, trained for 30 epochs, its hidden layer raises the
validation accuracy from about 84% to about 87%. The experiment
leaves three questions open. Automatic differentiation computed the
gradients of all 203,530 parameters without our looking inside; the next
section derives them. Weights of standard deviation 0.01 slowed the start of
training, and with more hidden layers they stop training altogether
(Exercise :exref:`ex_mlp-implementation-initialization-scale-depth`). And a
model with 3.4 parameters per training image could memorize its training
set, yet its validation accuracy keeps rising for all 30 epochs.

## Summary

A multilayer perceptron with one hidden layer adds a weight matrix, a bias
vector, and an activation function to softmax regression, and everything
else in the training code stays as it was. From scratch, it consists of four
registered parameter arrays, the ReLU written by hand, and a forward
pass of two lines that returns logits. A forward pass through the untrained
model checks the shapes, a loss near $\log 10 \approx 2.30$, and that about
half of the hidden units are active. The concise version stacks the same
operations in the framework's sequential container, with the same
initialization in every framework. On Fashion-MNIST, after 30 epochs, the
hidden layer of 256 units raises the validation accuracy from about 84% to
about 87%, with 26 times the parameters and multiply-adds of softmax
regression and, on a GPU, about the same training time per epoch.

## Exercises

1. [code] **A hidden layer without activation.** Replace `relu` by the
   identity in `MLPScratch` and train the resulting model as in this
   section.
    1. Compare its validation accuracy after 30 epochs with that of the MLP
       and of softmax regression.
    1. After training, compute $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$
       and $\mathbf{b} = \mathbf{W}^{(2)\top}\mathbf{b}^{(1)} + \mathbf{b}^{(2)}$,
       and verify on a minibatch that $\mathbf{X}\mathbf{W} + \mathbf{b}$
       reproduces the logits of the model.
    1. Exercise :exref:`ex_mlp-linear-networks` showed that such a network
       computes exactly the affine maps whose weight matrix has rank at most
       the hidden width. Explain why this restriction costs nothing here,
       and state below which hidden width it would reduce what the model
       can represent on Fashion-MNIST.
1. [code] **Which errors the hidden layer removes.** Softmax regression
   confuses mostly tops with tops and shoes with shoes
   (:numref:`sec_softmax_scratch`).
    1. Compute the column-normalized confusion matrix $\mathbf{C}$ of
       :numref:`sec_softmax_scratch` for the trained softmax regression and
       for the trained MLP of this section, and plot the recall of each
       class, the diagonal of $\mathbf{C}$, for both models.
    1. Which classes gain the most from the hidden layer, and which errors
       remain the most frequent? Relate your answer to what a hidden unit
       can compute and a weighted sum of pixels cannot.
    1. Repeat the comparison with both models trained for 10 epochs, as
       in :numref:`sec_softmax_scratch`. Which of the differences you found
       are larger than the change between two runs of the same model?
1. [code] **Single hidden unit.** Set `num_hiddens=1` in `MLPScratch`, so
   that its hidden layer has a single ReLU unit with weights
   $\mathbf{w}^{(1)} \in \mathbb{R}^{784 \times 1}$ and bias
   $b^{(1)} \in \mathbb{R}$. Each input $\mathbf{x} \in \mathbb{R}^{1 \times 784}$
   is then reduced to the scalar
   $t = \operatorname{ReLU}(\mathbf{x}\mathbf{w}^{(1)} + b^{(1)}) \geq 0$,
   and the logits are $\mathbf{o} = t\, \mathbf{w}^{(2)} + \mathbf{b}^{(2)}$
   with $\mathbf{w}^{(2)}, \mathbf{b}^{(2)} \in \mathbb{R}^{1 \times 10}$.
    1. Show that each class is predicted on at most one interval of $t$, and
       that the order of these intervals along the $t$ axis is fixed by the
       entries of $\mathbf{w}^{(2)}$. Describe the regions of input space
       that the model assigns to the classes, and state what they imply for
       separating the ten Fashion-MNIST classes.
    1. Suppose that after some update
       $\mathbf{x}\mathbf{w}^{(1)} + b^{(1)} < 0$ for every training
       example. Which parameters receive nonzero gradients from then on, and
       what does the network predict? Why does the same event do little
       harm when it hits one of the 256 hidden units of this section?
    1. Train the model five times, and report for each run the validation
       accuracy and the fraction of validation examples on which $t > 0$,
       which `model.hidden` gives you. Compare any run in which this
       fraction is zero with your answer to (b).
1. [code] **Width.** Train `MLP` as in this section, for 30 epochs at
   learning rate 0.1, with hidden widths
   $h \in \{16, 64, 256, 1024, 4096\}$, and record the final validation
   accuracy and the time per epoch.
    1. Plot both against $h$ on a logarithmic axis. Beyond which width does
       the accuracy stop improving by more than its variation from epoch to
       epoch?
    1. The multiply-adds grow in proportion to $h$. Time one pass over the
       training data that only reads the minibatches, without any model,
       and use it to explain the part of the time curve that does not grow
       with $h$. From which width on does the arithmetic dominate on your
       hardware?
    1. Repeat the training at widths 64 and 1024 with learning rates
       $\{0.03, 0.1, 0.3\}$. Does the best learning rate depend on the
       width?
   :exlabel:`ex_mlp-implementation-width`
1. [code] **Initialization scale and depth.** ● `MLPScratch` draws its
   weights with standard deviation 0.01, which its argument `std` sets.
    1. Train the one-hidden-layer model with standard deviations 0.001,
       0.01, and 0.1, and plot the training and validation curves for each.
       How much do the curves differ, and does the difference persist as
       training continues? Relate your observation to the scale of the
       hidden activations and the logits on the first forward pass.
    1. Extend `MLPScratch` to two and to four hidden layers, each of width
       256. Before running anything, predict which depth is more sensitive
       to the standard deviation, and why.
    1. Sweep the standard deviation over
       $\{0.001, 0.003, 0.01, 0.03, 0.1\}$ for both depths. For each depth,
       report the range for which the network reaches at least 70%
       validation accuracy within five epochs.
       :numref:`sec_numerical_stability` develops the principled answer.

    *Adapted from Stanford CS231n,
    [Assignment 2](https://cs231n.github.io/assignments2023/assignment2/),
    FullyConnectedNets Inline Question 1.*
   :exlabel:`ex_mlp-implementation-initialization-scale-depth`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/92)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/93)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/227)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17985)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.2]{.kicker}

Implementation of Multilayer Perceptrons<br>**four parameter arrays · checks before training · framework layers · a comparison with softmax regression**
:::
:::

::: {.slide title="A Hidden Layer of 256 Units Multiplies the Arithmetic by 26"}
[The problem]{.kicker}

Softmax regression on Fashion-MNIST: one weighted sum of the 784 pixels per
class.

- 7,850 parameters, 7,840 multiply-adds per image, about 83% validation
  accuracy after ten epochs.
- Most errors confuse tops of similar outline: shirts, pullovers, coats.

. . .

An MLP with one hidden layer of 256 ReLU units:

- 203,530 parameters, 203,264 multiply-adds per image: 26 times as many.

::: {.d2l-note}
Does the hidden layer make the classifier more accurate, and by how much?
Train both models under the same settings and measure.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the four parameter arrays of an MLP and register them with your
  framework;
- write the ReLU and a forward pass of two lines that returns logits;
- check an untrained model in a second: shapes, a loss of $\log 10$, half
  of the hidden units active;
- build the same model from framework layers with the same initialization;
- compare the MLP with softmax regression in accuracy, parameters,
  multiply-adds, and time per epoch.

**Running example:** Fashion-MNIST in minibatches of 256 at learning rate
0.1; softmax regression, 7,850 parameters and about 83% after ten epochs,
against the MLP.
:::

::: {.slide title="The First MLP Trained on Real Images"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Multilayer Perceptrons | hidden layers, activation functions, universal approximation |
| **Implementation of Multilayer Perceptrons** | **an MLP trained on Fashion-MNIST** |
| Backpropagation | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | vanishing gradients, Xavier and He initialization |
| Generalization in Deep Learning | memorization, double descent, implicit bias, early stopping |
| Dropout | random masks, robustness to missing units |
| Predicting House Prices on Kaggle | the workflow on real data |
:::

::: {.col .narrow}
- **Uses:** $\mathbf{H} = \phi(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$,
  $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$; the
  `Classifier` and its loss on logits; the `Trainer`; `SoftmaxRegression`.
- **Skips:** the backward pass (next deck); how to choose the initial
  scale (two decks on).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[From scratch]{.dtitle}

[parameters · forward pass · checks · training]{.dsub}
:::
:::

::: {.slide title="203,530 Parameters, All but 2,570 in the Hidden Layer"}
[From scratch]{.kicker}

::: {.cols .vc}
::: {.col}
$d = 784$ pixels, $h = 256$ hidden units, $q = 10$ classes:

- $\mathbf{W}^{(1)}$: $784 \times 256$; $\mathbf{b}^{(1)}$: 256 entries
- $\mathbf{W}^{(2)}$: $256 \times 10$; $\mathbf{b}^{(2)}$: 10 entries

$$785 \cdot 256 + 257 \cdot 10 = 203{,}530$$

::: {.d2l-note}
The width $h$ is a hyperparameter: training does not choose it.
:::
:::

::: {.col .fig .big}
![](../img/mdl-mlp-arch.svg)
:::
:::
:::

::: {.slide title="Random Weights Keep the Hidden Units Different"}
[From scratch]{.kicker}

Weights from $\mathcal{N}(0, 0.01^2)$, biases 0, as for softmax regression.

- **Why random:** if all weights of a layer were equal, every hidden unit
  would compute the same activation and receive the same gradient. The
  units would stay identical: one unit, 256 times.
- **Why small:** the logits start near 0, so every class gets probability
  about 1/10.

::: {.d2l-note .warn}
Keras: one initializer object returns the same values on every call. Shared
by $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$, it would make
$\mathbf{W}^{(2)}$ repeat the first entries of $\mathbf{W}^{(1)}$. Give each
weight its own.
:::
:::

::: {.slide title="Registered Parameters Need No Optimizer Code"}
[From scratch]{.kicker}

The default minibatch SGD of `Module` finds every registered array:

| framework | parameter type | found via |
|---|---|---|
| PyTorch | `nn.Parameter(tensor)` | `model.parameters()` |
| TensorFlow | `self.add_weight(...)` | `model.trainable_variables` |
| MXNet | `gluon.Parameter(...)`, read with `data()` | `model.collect_params()` |
| JAX | `nnx.Param(array)` | `nnx.state(model)` |

::: {.d2l-note}
Four arrays, `W1`, `b1`, `W2`, `b2`; the argument `std` sets the standard
deviation of the weights.
:::
:::

::: {.slide title="The ReLU Keeps the Positive Entries and Zeros the Rest"}
[From scratch]{.kicker}

Written by hand instead of calling the built-in function: each entry times
1 if it is positive, times 0 otherwise.

@mlp-implementation-model-1

::: {.d2l-note}
The derivative at $z = 0$ is then 0, as for every built-in ReLU. Written as
$\max(z, 0)$, autograd would give $1/2$ or 1 there, by framework.
:::
:::

::: {.slide title="The Forward Pass Is Two Lines and Returns Logits"}
[From scratch]{.kicker}

$$\mathbf{H} = \mathrm{ReLU}(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \qquad \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$$

@mlp-implementation-model-2

- Rows of $\mathbf{X}$: flattened images; rows of $\mathbf{O}$: logits.
- The inherited loss takes logits and never forms probabilities; autograd
  supplies the backward pass.
:::

::: {.slide title="Check: What Should the Untrained Model Report?"}
[Check]{.kicker}

Pass the first 8 images of a minibatch through the untrained model, with
weights from $\mathcal{N}(0, 0.01^2)$ and zero biases. Predict:

1. the shapes of $\mathbf{H}$ and $\mathbf{O}$;
2. the cross-entropy loss;
3. the fraction of positive hidden activations.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $8 \times 256$ and $8 \times 10$. The logits are near 0, so
every class gets about 1/10 and the loss is $\log 10 \approx 2.30$. Each
pre-activation is a sum of pixels times weights of random sign, positive or
negative with equal probability: about half.
:::
:::
:::

::: {.slide title="The Untrained Model Passes All Three Checks"}
[From scratch]{.kicker}

@!mlp-implementation-checks-before-training

- A weight matrix with swapped axes stops the forward pass with an error.
- A wrong number of inputs shows in the shapes.
- A loss far from 2.30 points to weights that are too large or to a mistake
  in the loss.

::: {.d2l-note}
A second of checking before minutes of training.
:::
:::

::: {.slide title="Thirty Epochs at Learning Rate 0.1 Reach About 87%"}
[From scratch]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Minibatches of 256, learning rate 0.1, as for softmax regression, but 30
epochs instead of 10.

- The training loss falls throughout.
- Validation accuracy: about 87% at the end, moving by about a point from
  epoch to epoch and sometimes dipping several points for one epoch.
:::

::: {.col .fig .big}
@!mlp-implementation-training
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Framework layers]{.dtitle}

[the same model, declared]{.dsub}
:::
:::

::: {.slide title="The Same Model Is Four Operations in a Sequential Container"}
[Framework layers]{.kicker}

Flatten, fully connected to 256, ReLU, fully connected to 10:

| framework | container and layers | initialization |
|---|---|---|
| PyTorch | `nn.Sequential(Flatten, LazyLinear, ReLU, LazyLinear)` | `weight_init` hook, after the first forward pass |
| TensorFlow | `Sequential([Flatten, Dense(relu), Dense])` | a `RandomNormal` per layer |
| MXNet | `nn.Sequential(Dense(relu), Dense)` | `initialize(init.Normal(0.01))` |
| JAX | `nnx.Sequential(Linear, relu, Linear)` | `kernel_init=normal(0.01)` |

::: {.d2l-note}
The layers hold the parameters, and the container fixes the order of the
operations: nothing to name or register by hand.
:::
:::

::: {.slide title="Framework Defaults Would Start the Four Tabs at Different Scales"}
[Framework layers]{.kicker}

Standard deviation of the default initial weights, for 784 and for 256
inputs:

| framework | default | layer 1 | layer 2 |
|---|---|---:|---:|
| PyTorch | uniform, bound $1/\sqrt{\text{inputs}}$ | 0.021 | 0.036 |
| TensorFlow | Glorot uniform | 0.044 | 0.087 |
| MXNet | uniform on $[-0.07, 0.07]$ | 0.040 | 0.040 |
| JAX | LeCun normal | 0.036 | 0.0625 |

::: {.d2l-note .rule}
Every tab sets $\mathcal{N}(0, 0.01^2)$ and zero biases, as the scratch
model does, so all implementations start from the same distribution.
:::
:::

::: {.slide title="Check: Which Model Learns Faster in the First Epoch?"}
[Check]{.kicker}

Softmax regression and the MLP both start from weights
$\mathcal{N}(0, 0.01^2)$ and zero biases and train at learning rate 0.1.
Which one has the lower average training loss in its first epoch, and why?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Softmax regression: about 0.79, against about 1.04 for the MLP.
The hidden activations start small, and every gradient that reaches
$\mathbf{W}^{(1)}$ is multiplied by entries of $\mathbf{W}^{(2)}$ of size
about 0.01, so the hidden layer learns slowly at first.
:::
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Comparison]{.dtitle}

[accuracy · parameters · multiply-adds · time]{.dsub}
:::
:::

::: {.slide title="The MLP Starts Behind and Ends Three Points Ahead"}
[Comparison]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Same data, loss, learning rate, and 30 epochs for all three models.

- Scratch and concise MLP: one curve, up to run-to-run noise.
- Usually behind after one epoch; about a point ahead after ten.
- Averaged over epochs 26–30: about 87% against about 84%.
:::

::: {.col .fig .big}
@!mlp-implementation-accuracy-2
:::
:::
:::

::: {.slide title="26 Times the Parameters and Arithmetic Remove a Fifth of the Errors"}
[Comparison]{.kicker}

| model | parameters | multiply-adds per image | accuracy, epochs 26–30 |
|:--|--:|--:|--:|
| softmax regression | 7,850 | 7,840 | about 84% |
| MLP, 256 hidden units | 203,530 | 203,264 | about 87% |

Measured time per epoch:

@!mlp-implementation-parameters-computation-and-time

::: {.d2l-note}
On a GPU the times are about the same: reading the images and the fixed
cost of each step take nearly all of an epoch.
:::
:::

::: {.slide title="Check: What Do 1,024 Hidden Units Cost?"}
[Check]{.kicker}

Widen the hidden layer from 256 to 1,024 units. How many parameters and
multiply-adds per image does the MLP have, and how do they compare with the
256-unit MLP and with softmax regression?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $785 \cdot 1024 + 1025 \cdot 10 = 814{,}090$ parameters and
$784 \cdot 1024 + 1024 \cdot 10 = 813{,}056$ multiply-adds: 4 times the
256-unit MLP and about 104 times softmax regression.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Model:** $\mathbf{W}^{(1)}$ ($784 \times 256$), $\mathbf{b}^{(1)}$,
  $\mathbf{W}^{(2)}$ ($256 \times 10$), $\mathbf{b}^{(2)}$: 203,530
  registered parameters, from $\mathcal{N}(0, 0.01^2)$ and zeros.
- **Forward:** $\mathbf{H} = \mathrm{ReLU}(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)})$,
  $\mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}$, logits into
  the stable cross-entropy.
- **Checks:** shapes $8 \times 256$ and $8 \times 10$, loss
  $\log 10 \approx 2.30$, half of the units active.
- **Concise:** flatten, linear, ReLU, linear in a sequential container, with
  the same initialization in every framework.
- **Result after 30 epochs:** about 87% against 84% for softmax regression;
  26 times the parameters and multiply-adds, about the same time per epoch
  on a GPU.

**Next:** how automatic differentiation computes the 203,530 gradients.
:::
