```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Implementation
:label:`sec_mlp-implementation`

In :numref:`sec_mlp`, we hand-crafted the weights of a two-unit hidden
layer to solve XOR—a trick that works for four corners of a square and
little else. Now it is time to let gradient descent learn the hidden
features from real images. On Fashion-MNIST, softmax regression reached
roughly 83% validation accuracy in ten epochs, with most of its errors
coming from look-alike garments such as shirts, pullovers, and coats
(:numref:`sec_softmax_scratch`). Separating those categories requires
combining pixels into higher-level patterns—exactly what a hidden layer is
built to do. How many mistakes does a single hidden layer eliminate, which
classes benefit, and what price do we pay? On paper the bill looks steep:
adding 256 hidden units multiplies both parameter count and arithmetic by
26 (:numref:`sec_mlp`). Only an experiment can tell whether the accuracy
jump earns its keep.

Implementing an MLP turns out to be scarcely harder than implementing
softmax regression—the main conceptual step is chaining layers together.
We build the network twice: first from scratch using raw parameter arrays,
a hand-written ReLU, and quick pre-training sanity checks, and then
concisely using each framework's built-in layers. Because the dataset,
cross-entropy loss, minibatch SGD optimizer, and `Trainer` carry over
unchanged, any performance shift isolates the effect of the hidden layer
itself. We then pit our MLP head to head against softmax regression on
accuracy, per-class recall, and wall-clock training time, and peer inside
the hidden weights to see what features SGD actually invented. Automatic
differentiation handles every gradient here; in the next section we open
the hood to see how.

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #mlp-implementation-implementation-of-multilayer-perceptrons}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #mlp-implementation-implementation-1}
import time
```

## Implementation from Scratch

### The Parameters

Each Fashion-MNIST image flattens into $d = 784$ grayscale pixel features
mapped to $q = 10$ classes. We insert a single hidden layer of $h = 256$
units (:numref:`fig_mdl-mlp-arch`)—choosing a power of two is a common
practical habit because it aligns cleanly with hardware memory layouts,
though width itself is an adjustable hyperparameter (explored in Exercise
:exref:`ex_mlp-implementation-width`). Following the layout of
:numref:`sec_mlp`, each layer pairs a weight matrix (inputs $\times$
outputs) with a bias vector:

$$\mathbf{W}^{(1)} \in \mathbb{R}^{784 \times 256}, \quad \mathbf{b}^{(1)} \in \mathbb{R}^{256}, \quad \mathbf{W}^{(2)} \in \mathbb{R}^{256 \times 10}, \quad \mathbf{b}^{(2)} \in \mathbb{R}^{10}.$$

Together these four arrays hold all 203,530 parameters we counted in
:numref:`sec_mlp`, with all but 2,570 belonging to the hidden layer.

![The MLP of this section. A minibatch of $n$ images is flattened to an $n \times 784$ matrix $\mathbf{X}$, an affine map followed by the ReLU computes the $n \times 256$ hidden representation $\mathbf{H}$, and a second affine map computes the $n \times 10$ logits $\mathbf{O}$.](../img/mdl-mlp-arch.svg)
:label:`fig_mdl-mlp-arch`

How should we initialize these four arrays? As in softmax regression, we
sample weights from $\mathcal{N}(0, 0.01^2)$ and set biases to zero.
Randomness is strictly mandatory here: if two hidden units start with
identical weights, they receive identical gradients and stay locked
together forever, a symmetry trap we examine in
:numref:`sec_numerical_stability`. Small weights keep the initial logits
near zero so the untrained network assigns roughly $1/10$ probability to
every class. Note, however, that `std=0.01` is simply borrowed from
softmax regression to keep our comparison clean—it is actually a poor scale
for a multi-layer network, and the sluggish opening epochs will make that
plain.

Just as in :numref:`sec_softmax_scratch`, registering all four arrays with
the framework lets `Classifier`'s inherited SGD optimizer track and update
them automatically, with no custom `configure_optimizers` needed.

:begin_tab:`mxnet`
Calling `initialize()` allocates storage for the four registered parameters,
whose tensors we read via `data()`.
:end_tab:

:begin_tab:`jax`
As in :numref:`sec_softmax_scratch`, `rngs` supplies the PRNG key used to
draw the initial weights.
:end_tab:

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

First, let us implement the ReLU activation by hand rather than calling a
built-in op. Multiplying `X` by the boolean mask `X > 0` keeps positive
entries intact and zeroes out the rest—and cleanly produces derivative 0 at
$z = 0$ under automatic differentiation, matching every framework's
built-in ReLU (whereas taking a maximum with 0 can yield 1 or $1/2$
depending on the library, though as :numref:`subsec_activation-functions`
noted, exact zeros rarely occur in practice).

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

Given a minibatch of $n$ images flattened into $\mathbf{X} \in \mathbb{R}^{n \times 784}$,
the network evaluates

$$\mathbf{H} = \mathrm{ReLU}\left(\mathbf{X}\mathbf{W}^{(1)} + \mathbf{b}^{(1)}\right), \qquad \mathbf{O} = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)},$$

with each bias broadcast across rows. We factor the hidden layer into a
`hidden` method (handy for inspecting activations) and let `forward` map
$\mathbf{H}$ to the raw logits $\mathbf{O}$—remembering that `Classifier`'s
inherited cross-entropy loss expects logits, not probabilities
(:numref:`sec_softmax_concise`).

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

### Checks before Training

Training a deep network can take minutes—or hours on larger datasets—and a
subtle wiring bug often reveals itself only as puzzlingly poor accuracy at
the very end. Running a single forward pass through the *untrained* model
on a handful of images lets us verify three theoretical predictions in a
fraction of a second:

1. **Shapes:** Every image should produce a 256-dimensional hidden vector
   and 10 class logits.
1. **Initial loss:** Near-zero logits yield near-uniform probabilities
   ($\approx 1/10$), so cross-entropy should start right around
   $\log 10 \approx 2.30$.
1. **Active ReLU fraction:** Because biases start at zero and weights are
   symmetric around zero, roughly half of all hidden pre-activations should
   be positive, leaving ~50% of ReLU units active.

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

All three checks pass: shapes are $8 \times 256$ and $8 \times 10$, the
initial loss sits right near $2.30$, and roughly half of the hidden units
fire. A transposed weight matrix would have crashed immediately, wrong
dimensions would jump out of the shapes, and an initial loss far from 2.30
would immediately betray mis-scaled weights or a broken loss function.

### Training

Our training loop is identical to softmax regression's
(:numref:`sec_softmax_scratch`): minibatches of 256 images and learning
rate 0.1, now run for 30 epochs instead of 10 because our small `std=0.01`
initialization makes the hidden layer warm up slowly (as we will see
momentarily). Passing `num_gpus=1` instructs `Trainer` to run on a GPU if
available and fall back to the CPU otherwise (:numref:`sec_use_gpu`).

```{.python .input #mlp-implementation-training}
trainer = d2l.Trainer(max_epochs=30, num_gpus=1)
trainer.fit(model, data)
```

Training loss drops steadily across all 30 epochs, while validation loss
sits modestly above it and validation accuracy climbs through most of the
run. Notice that the validation curve jitters noticeably from epoch to
epoch—sometimes dipping several points in a single epoch. Nothing is wrong:
minibatch SGD with a fixed learning rate keeps bouncing around the minimum,
and each epoch's validation point simply captures a snapshot of wherever
the final minibatch step left the weights.

## Concise Implementation

Registering weight matrices manually gets tedious fast—imagine maintaining
a 50-layer network by hand and wanting to insert a new layer between layers
42 and 43! Using each framework's high-level APIs, building an MLP is just
as concise as softmax regression (:numref:`sec_softmax_concise`): we simply
stack *two* fully connected layers instead of one inside a `Sequential`
container—flattening, a 256-unit hidden layer, ReLU, and a 10-unit output
layer. As before, we explicitly match our scratch initialization
($\mathcal{N}(0, 0.01^2)$ weights and zero biases) so every framework runs
the exact same experiment.

:begin_tab:`mxnet`
Gluon's `Dense` layer flattens its input automatically and applies ReLU
directly via its `activation='relu'` argument.
:end_tab:

:begin_tab:`pytorch`
As in `SoftmaxRegression`, PyTorch's `LazyLinear` layers defer parameter
allocation until the first forward pass, after which `Trainer` runs
`weight_init`.
:end_tab:

:begin_tab:`tensorflow`
Keras's first `Dense` layer fuses the ReLU activation directly via
`activation='relu'`.
:end_tab:

:begin_tab:`jax`
Flax NNX accepts `nnx.relu` directly as a callable stage inside
`nnx.Sequential`; each `nnx.Linear` declares its input width explicitly,
and `forward` flattens the input batch before calling `self.net`.
:end_tab:

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
Notice that `MLP` defines no `forward` method of its own: it inherits
`Module.forward` (:numref:`subsec_oo-design-models`), which simply pipes
`X` straight through `self.net(X)`. We unpack how `Sequential` works under
the hood in :numref:`subsec_model-construction-sequential`.
:end_tab:

Where the scratch version spelled out every matrix and operation by hand,
here the layers own their parameters and `Sequential` orchestrates the
pipeline—making it effortless to deepen, widen, or rearrange architectures
without renaming a single variable.

## Comparison with Softmax Regression

### Accuracy

Let us now run a controlled head-to-head comparison: train both the concise
`MLP` and `d2l.SoftmaxRegression` (:numref:`sec_softmax_concise`) under
identical settings—30 epochs, batch size 256, learning rate 0.1, and the
same `std=0.01` initialization—timing each run and plotting all three
validation accuracy curves (including our scratch MLP) on the same axes.

```{.python .input #mlp-implementation-accuracy-1}
curves, seconds = {'MLP from scratch': trainer.history['val_acc']}, {}
trainer.board.display = False
models = {'concise MLP': MLP(num_outputs=10, num_hiddens=256, lr=0.1),
          'softmax regression': d2l.SoftmaxRegression(num_outputs=10, lr=0.1)}
for name, net in models.items():
    start = time.perf_counter()
    trainer.fit(net, data)
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
train_acc = trainer.evaluate(model, data, train=True)['acc']
print(f'training accuracy of the MLP from scratch: {train_acc:.3f}')
```

As expected, the scratch and concise MLPs trace the exact same trajectory
up to run-to-run randomness. Interestingly, *both* MLPs lag behind softmax
regression after the first epoch! Our `std=0.01` initialization is the
culprit: tiny first-layer weights produce tiny hidden activations, and
every gradient flowing back into $\mathbf{W}^{(1)}$ gets multiplied by
equally tiny entries of $\mathbf{W}^{(2)}$, putting the hidden layer into
slow motion at the start. Even at epoch 10 (where we stopped in
:numref:`sec_softmax_scratch`), the MLP leads by barely a percentage
point—easy to mistake for noise. By epochs 26–30, however, the gap is
unmistakable: averaging the last five epochs to smooth out single-epoch
jitter, softmax regression plateaus around 84% whereas the MLP climbs to
about 87%. Three percentage points might sound small until you flip them
into *error rates*: dropping from 16% error to 13% error cuts away roughly
one-fifth of all mistakes! And on the training set the MLP scores around
89%—only a couple of points above validation, showing virtually no severe
overfitting despite its 203,530 parameters.

### Errors by Class

Where did that one-fifth reduction in errors come from? Naturally we might
hope it fixed the shirts confused with pullovers and coats, yet headline
accuracy cannot tell us. Let us compute the per-class *recall* (the
diagonal of the confusion matrix from :numref:`sec_softmax_scratch`) for
softmax regression alongside *both* identically configured MLPs—having two
independent MLP runs side by side immediately shows how much per-class
scores fluctuate from run to run.

```{.python .input #mlp-implementation-errors-by-class}
def recall(net):
    """Fraction of the validation images of each class that net gets right."""
    correct, total = np.zeros(10), np.zeros(10)
    for X, y in map(trainer.prepare_batch, data.val_dataloader()):
        y_hat = d2l.numpy(d2l.argmax(net(X), axis=1))
        y = d2l.numpy(y).astype(int)
        total += np.bincount(y, minlength=10)
        correct += np.bincount(y, weights=y_hat == y, minlength=10)
    return correct / total

recalls = {'softmax': recall(models['softmax regression']),
           'MLP, scratch': recall(model),
           'MLP, concise': recall(models['concise MLP'])}
names = data.text_labels(range(10))
print(f'{"class":11s}' + ''.join(f'{name:>14s}' for name in recalls))
for j, name in enumerate(names):
    print(f'{name:11s}' + ''.join(f'{r[j]:14.2f}' for r in recalls.values()))
```

The breakdown reveals a subtler picture than a simple "shirts got fixed."
Across the six classes softmax regression already handled well (trousers,
dresses, bags, and the three shoe categories), recall nudges up slightly
and combined errors drop. Across the four tops (T-shirts, pullovers, coats,
and shirts), however, the two identical MLPs can swing by ten percentage
points or more against *each other* on an individual class! What you are
seeing is a decision boundary still in motion: at a constant learning rate,
the final minibatch steps nudge the boundaries between look-alike tops back
and forth, so whatever recall one top gains, a neighboring top temporarily
loses. Only their combined total stabilizes: summed across all four tops,
the MLP generally cuts errors by a few hundred (providing the bulk of its
overall gain), though an unlucky final epoch can momentarily reverse even
that. No single top owns the victory—the four tops remain the hardest part
of Fashion-MNIST.

### The Learned Features

In :numref:`sec_softmax_scratch`, reshaping the ten columns of $\mathbf{W}$
revealed ten whole-garment templates. Now let us peek inside the MLP's
hidden layer to see what features gradient descent discovered. Each column
$j$ of $\mathbf{W}^{(1)}$ holds the 784 pixel weights feeding hidden unit
$j$: reshaped to $28 \times 28$, bright garment pixels landing on red
weights excite unit $j$ while those on blue weights inhibit it. Let us
visualize the ten hidden units with the largest weight norms (each on its
own symmetric color scale), labeled by whichever class logit they boost
most strongly in $\mathbf{W}^{(2)}$.

```{.python .input #mlp-implementation-what-the-hidden-units-compute-1}
%%tab pytorch, tensorflow
W1, W2 = d2l.numpy(model.W1), d2l.numpy(model.W2)
```

```{.python .input #mlp-implementation-what-the-hidden-units-compute-1}
%%tab mxnet
W1, W2 = d2l.numpy(model.W1.data()), d2l.numpy(model.W2.data())
```

```{.python .input #mlp-implementation-what-the-hidden-units-compute-1}
%%tab jax
W1, W2 = d2l.numpy(model.W1[...]), d2l.numpy(model.W2[...])
```

```{.python .input #mlp-implementation-what-the-hidden-units-compute-2}
units = np.argsort(np.linalg.norm(W1, axis=0))[::-1][:10]  # Largest weights
images = W1[:, units].T.reshape(10, 28, 28)
images /= np.abs(images).max(axis=(1, 2), keepdims=True)  # Own color scale
d2l.show_images(images, 2, 5, titles=[names[k] for k in W2[units].argmax(1)],
                cmap='RdBu_r', vmin=-1, vmax=1);
```

These look strikingly different from whole-garment templates. Rather than
matching an entire silhouette, most high-norm hidden units specialize in
localized *parts and contrasts*—a collar opening, the outline of a sleeve,
or the dark gap between trouser legs—with vivid red and blue regions
sitting side by side to detect local edges and contrasts rather than raw
brightness. While a linear template asks one global question ("does this
whole image look like a shirt?"), the hidden layer asks 256 focused
questions about distinct garment features, and the output layer weighs
those answers together. Notice the titles too: most of these heaviest-moving
units feed into the four tops (especially shirts), suggesting that gradient
descent invested its greatest representational effort right where softmax
regression struggled most.

### Parameters, Computation, and Time

Let us put the two models side by side. Forward-pass multiply-adds per
image jump from $784 \cdot 10 = 7{,}840$ for softmax regression to
$784 \cdot 256 + 256 \cdot 10 = 203{,}264$ for our MLP:

| model | parameters | multiply-adds per image | validation accuracy, epochs 26–30 |
|:--|--:|--:|--:|
| softmax regression | 7,850 | 7,840 | about 84% |
| MLP, 256 hidden units | 203,530 | 203,264 | about 87% |

With 26 times as many parameters and floating-point operations, does the
MLP take 26 times as long per epoch? Let us check our timings:

```{.python .input #mlp-implementation-parameters-computation-and-time}
for name, s in seconds.items():
    print(f'{name}: {s:.1f} s per epoch')
```

Surprisingly, on a GPU the two epoch times are nearly identical! For
networks this compact, data loading and per-step kernel launch overhead
completely dominate wall-clock time, so multiplying the matrix arithmetic
by 26 comes virtually for free (Exercise
:exref:`ex_mlp-implementation-width` probes where compute finally overtakes
overhead).

This single experiment puts the three deep-learning questions from our
chapter introduction squarely on the table. First, automatic
differentiation computed all 203,530 gradients effortlessly behind the
scenes—how does it do so in a single backward pass (:numref:`sec_backprop`)?
Second, our `std=0.01` initialization noticeably slowed the opening epochs,
and in deeper stacks it stalls learning completely
(:numref:`sec_numerical_stability`). Third, with 3.4 parameters per
training image, our MLP had ample capacity to memorize Fashion-MNIST—yet it
generalized within a couple of points of its training accuracy
(:numref:`sec_generalization_deep`).

## Summary

Stepping from softmax regression to a one-hidden-layer MLP requires adding
just one affine layer and a nonlinearity while reusing the exact same data
loader, loss, and `Trainer`—putting us roughly at the state of the art of
late-1980s neural network modeling. Before training, a quick forward pass
on a few images catches shape bugs, verifies the $\log 10 \approx 2.30$
initial loss, and confirms that roughly half of ReLU units activate. Using
each framework's `Sequential` container eliminates manual parameter
bookkeeping and makes deeper architectures trivial to assemble. On
Fashion-MNIST over 30 epochs, 256 ReLU hidden units trim roughly a fifth of
softmax regression's errors (~87% vs. ~84% validation accuracy) at
virtually identical GPU time per epoch, with the strongest hidden units
specializing in localized garment parts and pixel contrasts rather than
whole-image templates.

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
1. [code] **Recall across runs.** The recall table of this section holds
   one run of softmax regression and two runs of the MLP, too few to say
   which differences between the models are real.
    1. Train each model five times and report, for every class, the mean and
       the range of its recall. For which classes is the difference between
       the two models larger than the spread within either of them?
    1. Compute the column-normalized confusion matrix $\mathbf{C}$ of
       :numref:`sec_softmax_scratch` for one run of each model. Which
       confusions among the four tops does the hidden layer reduce, and
       which does it only move from one pair of classes to another?
    1. Keep a copy of the parameters of the MLP after each of its last five
       epochs, and classify each validation image by the average of the five
       predicted probabilities. Does the recall of the tops vary less
       between runs than that of a single epoch?
1. [code] **Single hidden unit.** Set `num_hiddens=1` in `MLPScratch`, so
   that its hidden layer has a single ReLU unit with weights
   $\mathbf{w}^{(1)} \in \mathbb{R}^{784 \times 1}$ and bias
   $b^{(1)} \in \mathbb{R}$. Each input $\mathbf{x} \in \mathbb{R}^{1 \times 784}$
   is then reduced to the scalar
   $t = \operatorname{ReLU}(\mathbf{x}\mathbf{w}^{(1)} + b^{(1)}) \geq 0$,
   and the logits are $\mathbf{o} = t\, \mathbf{w}^{(2)} + \mathbf{b}^{(2)}$
   with $\mathbf{w}^{(2)} \in \mathbb{R}^{1 \times 10}$ and
   $\mathbf{b}^{(2)} \in \mathbb{R}^{10}$.
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
    1. Extend `MLPScratch` to two and to four hidden layers, each of
       width 256. Before running anything, predict which depth is more
       sensitive to the standard deviation, and why.
    1. Sweep the standard deviation over
       $\{0.001, 0.003, 0.01, 0.03, 0.1\}$ for both depths. For each depth,
       report the range for which the network reaches at least 70%
       validation accuracy within five epochs.
       :numref:`sec_numerical_stability` explains the pattern and trains a
       network with ten hidden layers.

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

Implementation<br>**four parameter arrays · checks before training · framework layers · a comparison with softmax regression**
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
- compare the MLP with softmax regression in accuracy, class by class, and
  in parameters, multiply-adds, and time per epoch;
- read the weights of a hidden unit as an image.

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
| Multilayer Perceptrons | hidden layers, universal approximation, activation functions |
| **Implementation** | **an MLP trained on Fashion-MNIST** |
| Backpropagation | how the gradients are computed, and at what memory cost |
| Numerical Stability and Initialization | vanishing gradients, Xavier and He initialization |
| Generalization in Deep Learning | memorization, double descent, implicit bias, early stopping |
| Dropout | random masks, slower memorization of noisy labels |
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

::: {.slide title="Random Weights Keep the Hidden Units Different" except="tensorflow"}
[From scratch]{.kicker}

Weights from $\mathcal{N}(0, 0.01^2)$, biases 0, as for softmax regression.

- **Why random:** if all weights of a layer were equal, every hidden unit
  would compute the same activation and receive the same gradient. The
  units would stay identical: one unit, 256 times.
- **Why small:** the logits start near 0, so every class gets probability
  about 1/10.
:::

::: {.slide title="Random Weights Keep the Hidden Units Different" only="tensorflow"}
[From scratch]{.kicker}

Weights from $\mathcal{N}(0, 0.01^2)$, biases 0, as for softmax regression.

- **Why random:** if all weights of a layer were equal, every hidden unit
  would compute the same activation and receive the same gradient. The
  units would stay identical: one unit, 256 times.
- **Why small:** the logits start near 0, so every class gets probability
  about 1/10.

::: {.d2l-note .warn}
A Keras initializer object returns the same values on every call. Shared
by $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$, it would make
$\mathbf{W}^{(2)}$ repeat the first entries of $\mathbf{W}^{(1)}$. Give each
weight its own.
:::
:::

::: {.slide title="Registered Parameters Need No Optimizer Code" only="mxnet"}
[From scratch]{.kicker}

The default minibatch SGD of `Module` finds every registered array:

- A `gluon.Parameter(...)` assigned as an attribute is registered with the
  block; `initialize` allocates it, and `data()` reads its value.
- `model.collect_params()` lists the four arrays, and Gluon's `Trainer`
  with the `'sgd'` update updates them.

::: {.d2l-note}
Four arrays, `W1`, `b1`, `W2`, `b2`; the argument `std` sets the standard
deviation of the weights.
:::
:::

::: {.slide title="Registered Parameters Need No Optimizer Code" only="pytorch"}
[From scratch]{.kicker}

The default minibatch SGD of `Module` finds every registered array:

- `nn.Parameter(tensor)` registers a tensor with the module.
- `model.parameters()` lists the four arrays, and `torch.optim.SGD`
  updates them.

::: {.d2l-note}
Four arrays, `W1`, `b1`, `W2`, `b2`; the argument `std` sets the standard
deviation of the weights.
:::
:::

::: {.slide title="Registered Parameters Need No Optimizer Code" only="tensorflow"}
[From scratch]{.kicker}

The default minibatch SGD of `Module` finds every registered array:

- `self.add_weight(...)` creates a variable and registers it with the
  Keras model.
- `model.trainable_variables` lists the four arrays, and Keras's `SGD`
  updates them.

::: {.d2l-note}
Four arrays, `W1`, `b1`, `W2`, `b2`; the argument `std` sets the standard
deviation of the weights.
:::
:::

::: {.slide title="Registered Parameters Need No Optimizer Code" only="jax"}
[From scratch]{.kicker}

The default minibatch SGD of `Module` finds every registered array:

- `nnx.Param(array)` marks an array as a trainable parameter.
- `nnx.state(model)` contains the four arrays, and `optax.sgd` updates
  them.

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
The derivative at $z = 0$ is then 0, the value that the built-in ReLU uses.
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

::: {.slide title="The Same Model Is Four Operations in a Sequential Container" only="mxnet"}
[Framework layers]{.kicker}

Flatten, fully connected to 256, ReLU, fully connected to 10:

- `nn.Sequential()` with `Dense(256, activation='relu')` and `Dense(10)`;
  `Dense` flattens its input itself.
- Initialization: `initialize(init.Normal(sigma=0.01))`.

::: {.d2l-note}
The layers hold the parameters, and the container fixes the order of the
operations: nothing to name or register by hand.
:::
:::

::: {.slide title="The Same Model Is Four Operations in a Sequential Container" only="pytorch"}
[Framework layers]{.kicker}

Flatten, fully connected to 256, ReLU, fully connected to 10:

- `nn.Sequential(nn.Flatten(), nn.LazyLinear(256), nn.ReLU(),
  nn.LazyLinear(10))`.
- Initialization: the `Trainer` calls `weight_init` after the first
  forward pass, once the lazy layers know their shapes.

::: {.d2l-note}
The layers hold the parameters, and the container fixes the order of the
operations: nothing to name or register by hand.
:::
:::

::: {.slide title="The Same Model Is Four Operations in a Sequential Container" only="tensorflow"}
[Framework layers]{.kicker}

Flatten, fully connected to 256, ReLU, fully connected to 10:

- `Sequential([Flatten(), Dense(256, activation='relu'), Dense(10)])`.
- Initialization: a `RandomNormal(stddev=0.01)` per layer, since one
  initializer object repeats its values.

::: {.d2l-note}
The layers hold the parameters, and the container fixes the order of the
operations: nothing to name or register by hand.
:::
:::

::: {.slide title="The Same Model Is Four Operations in a Sequential Container" only="jax"}
[Framework layers]{.kicker}

Flatten, fully connected to 256, ReLU, fully connected to 10:

- `nnx.Sequential(nnx.Linear(784, 256), nnx.relu, nnx.Linear(256, 10))`;
  `forward` flattens each image first.
- Initialization: `kernel_init=nnx.initializers.normal(0.01)`.

::: {.d2l-note}
The layers hold the parameters, and the container fixes the order of the
operations: nothing to name or register by hand.
:::
:::

::: {.slide title="The Default Initialization Would Start at a Different Scale" only="mxnet"}
[Framework layers]{.kicker}

- `initialize()` by default: uniform on $[-0.07, 0.07]$ for every layer.
- Standard deviation 0.040 in both layers, against 0.01 in the scratch
  model.

::: {.d2l-note .rule}
`init.Normal(sigma=0.01)` and zero biases match the scratch model, so both
implementations start from the same distribution.
:::
:::

::: {.slide title="The Default Initialization Would Start at a Different Scale" only="pytorch"}
[Framework layers]{.kicker}

- `nn.LazyLinear` by default: uniform with bound $1/\sqrt{\text{inputs}}$.
- Standard deviation 0.021 for 784 inputs, 0.036 for 256 inputs, against
  0.01 in the scratch model.

::: {.d2l-note .rule}
`weight_init` sets $\mathcal{N}(0, 0.01^2)$ and zero biases, as the scratch
model does, so both implementations start from the same distribution.
:::
:::

::: {.slide title="The Default Initialization Would Start at a Different Scale" only="tensorflow"}
[Framework layers]{.kicker}

- `Dense` by default: Glorot uniform, variance $2/(\text{inputs} + \text{outputs})$.
- Standard deviation 0.044 for the first layer (784 to 256), 0.087 for the
  second (256 to 10), against 0.01 in the scratch model.

::: {.d2l-note .rule}
A `RandomNormal(stddev=0.01)` per layer and zero biases match the scratch
model, so both implementations start from the same distribution.
:::
:::

::: {.slide title="The Default Initialization Would Start at a Different Scale" only="jax"}
[Framework layers]{.kicker}

- `nnx.Linear` by default: LeCun normal, variance $1/\text{inputs}$.
- Standard deviation 0.036 for 784 inputs, 0.0625 for 256 inputs, against
  0.01 in the scratch model.

::: {.d2l-note .rule}
`kernel_init=normal(0.01)` and zero biases match the scratch model, so both
implementations start from the same distribution.
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

[accuracy · classes · hidden units · cost]{.dsub}
:::
:::

::: {.slide title="The MLP Starts Behind and Ends Three Points Ahead"}
[Comparison]{.kicker}

::: {.cols .vc}
::: {.col .narrow}
Same data, loss, learning rate, and 30 epochs for all three models.

- Scratch and concise MLP: one curve, up to run-to-run noise.
- Usually behind after one epoch; about a point ahead after ten.
- Averaged over epochs 26–30: about 87% against about 84%, a fifth of the
  errors removed.
- Training accuracy of the MLP: about 89%.
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

::: {.slide title="Recall by Class: No Single Class Gains Reliably"}
[Comparison · classes]{.kicker}

Which fifth of the errors? Recall: the fraction of each class's validation
images classified correctly. The two MLPs are the same model, trained twice.

@!mlp-implementation-errors-by-class

::: {.d2l-note}
The six classes outside the tops gain a few points. Among the four tops the
two MLPs differ by ten points or more: SGD moves the boundaries, and only
the sum over the tops is stable.
:::
:::

::: {.slide title="Hidden Units Respond to Parts of Garments"}
[Comparison · features]{.kicker}

The ten units with the largest incoming weights: column $j$ of
$\mathbf{W}^{(1)}$ as a $28 \times 28$ image (red raises the unit, blue
lowers it), titled by the class whose logit the unit raises most.

@!mlp-implementation-what-the-hidden-units-compute-2

::: {.d2l-note}
Softmax regression: ten templates, each a blurred garment. Here: necklines,
sleeve edges, the gap between trouser legs, red beside blue. Most of the
ten serve one of the four tops.
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
  the initialization of the scratch model.
- **Result after 30 epochs:** about 87% against 84% for softmax regression;
  26 times the parameters and multiply-adds, about the same time per epoch
  on a GPU.
- **Where:** no single class gains reliably, and the tops keep most of the
  errors; the hidden units with the largest weights respond to parts of
  garments.

**Next:** how automatic differentiation computes the 203,530 gradients.
:::
