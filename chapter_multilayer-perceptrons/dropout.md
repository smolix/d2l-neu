```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Dropout
:label:`sec_dropout`

When an overparameterized network trains for many epochs on scarce or noisy data, it eventually memorizes training quirks and corrupted labels (:numref:`sec_generalization_deep`). Where weight decay (:numref:`sec_weight_decay`) penalizes large weights and early stopping (:numref:`subsec_early_stopping`) halts training before late-stage memorization, *dropout* :cite:`Hinton.Srivastava.Krizhevsky.ea.2012,Srivastava.Hinton.Krizhevsky.ea.2014` regularizes a network by injecting multiplicative noise directly into its hidden layers during training: on every forward pass, each hidden unit is independently zeroed out with probability $p$, forcing every unit to learn features that work reliably without counting on specific partner units (*co-adaptation*). At test time, dropout is disabled and the full network runs deterministically.

In this section, we formulate inverted dropout masking and rescaling, implement it from scratch and concisely in a two-hidden-layer MLP on Fashion-MNIST, and demonstrate on a heavily corrupted label-noise experiment how dropout slows memorization and preserves validation accuracy.

```{.python .input #dropout}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #dropout}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #dropout}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #dropout}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```


## Noise in Hidden Layers

### Masking and Rescaling (Inverted Dropout)

Injecting random noise into inputs during training encourages smooth, perturbation-resilient functions :cite:`Bishop.1995`—in fact, for a linear model $\mathbf{w}^\top\mathbf{x} + b$ under squared error, adding zero-mean independent noise of variance $s^2$ to the inputs adds $s^2\|\mathbf{w}\|_2^2$ to the expected loss, which is literally $\ell_2$ weight decay! Dropout extends noise injection to hidden layers via random binary masking.

For a hidden activation $h_j$ and *dropout probability* $p \in [0, 1)$, on each training step we draw an independent Bernoulli mask $\xi_j \in \{0, 1\}$ with $P(\xi_j = 0) = p$ and $P(\xi_j = 1) = 1 - p$, and replace $h_j$ with

$$
h_j' = \frac{\xi_j h_j}{1 - p} =
\begin{cases}
0 & \textrm{with probability } p,\\
\dfrac{h_j}{1 - p} & \textrm{with probability } 1 - p.
\end{cases}
$$
:eqlabel:`eq_dropout`

Dividing the surviving activations by $1 - p$ during training (*inverted dropout*) keeps the expectation unbiased:

$$\mathbb{E}[h_j'] = p \cdot 0 + (1 - p)\,\frac{h_j}{1 - p} = h_j, \qquad \mathrm{Var}[h_j'] = \frac{p}{1 - p}\,h_j^2.$$

Because $\mathbb{E}[h_j'] = h_j$ is preserved during training, **at inference time we simply turn dropout off ($h_j' = h_j$) without any weight rescaling**. (Note that for a linear regression model with input dropout, the variance formula $\mathrm{Var}[x_j'] = \frac{p}{1-p}x_j^2$ adds an adaptive ridge penalty $\mathbb{E}[(\mathbf{w}^\top\mathbf{x}' + b - y)^2] = (\mathbf{w}^\top\mathbf{x} + b - y)^2 + \frac{p}{1-p}\sum_j x_j^2 w_j^2$ to the squared loss; in a deep nonlinear network, because $\mathbb{E}[\mathrm{ReLU}(z)] \neq \mathrm{ReLU}(\mathbb{E}[z])$, the deterministic test-time network is an effective approximation to the ensemble of $2^h$ thinned subnetworks—see Exercises :exref:`ex_dropout-monte-carlo-dropout` and :exref:`ex_dropout-geometric-mean`.)

## Implementation

### The Dropout Layer

Our from-scratch `dropout_layer(X, dropout)` draws a uniform tensor in $[0, 1)$, keeps entries $\geq$ `dropout`, and divides by `1.0 - dropout` (returning zeros when `dropout == 1`). Let us test it on a $2 \times 8$ array for $p \in \{0, 0.5, 1\}$ and verify on $10^6$ ones at $p = 0.2$ that the sample mean is $1.000$ and the sample variance is $p/(1 - p) = 0.250$:

```{.python .input #dropout-implementation-from-scratch-1}
%%tab mxnet
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return mnp.zeros_like(X)
    mask = mnp.random.uniform(0, 1, X.shape, device=X.device) >= dropout
    return mask.astype(X.dtype) * X / (1.0 - dropout)
```

```{.python .input #dropout-implementation-from-scratch-1}
%%tab pytorch
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return torch.zeros_like(X)
    mask = torch.rand_like(X) >= dropout
    return mask.to(X.dtype) * X / (1.0 - dropout)
```

```{.python .input #dropout-implementation-from-scratch-1}
%%tab tensorflow
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1: return tf.zeros_like(X)
    mask = tf.random.uniform(tf.shape(X)) >= dropout
    return tf.cast(mask, X.dtype) * X / (1.0 - dropout)
```

```{.python .input #dropout-implementation-from-scratch-1}
%%tab jax
def dropout_layer(X, dropout, key):
    assert 0 <= dropout <= 1
    if dropout == 1: return jnp.zeros_like(X)
    mask = jax.random.uniform(key, X.shape) >= dropout
    return mask.astype(X.dtype) * X / (1.0 - dropout)
```

Let's test `dropout_layer` on a $2 \times 8$ tensor with dropout probabilities
$0$, $0.5$, and $1$. We also apply $p = 0.2$ to a vector of one million ones and
verify that its empirical mean and variance match the theoretical values $1$ and
$p/(1 - p) = 0.25$.

```{.python .input #dropout-implementation-from-scratch-2}
%%tab pytorch, tensorflow, mxnet
X = d2l.reshape(d2l.arange(16, dtype=d2l.float32), (2, 8))
for p in (0, 0.5, 1):
    print(f'p = {p}:', dropout_layer(X, p))
Y = dropout_layer(d2l.ones(10**6), 0.2)
print(f'mean {float(d2l.reduce_mean(Y)):.3f}, '
      f'variance {float(d2l.reduce_mean((Y - 1) ** 2)):.3f}')
```

```{.python .input #dropout-implementation-from-scratch-2}
%%tab jax
X = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
for p in (0, 0.5, 1):
    print(f'p = {p}:', dropout_layer(X, p, d2l.get_key()))
Y = dropout_layer(jnp.ones(10**6), 0.2, d2l.get_key())
print(f'mean {float(Y.mean()):.3f}, '
      f'variance {float(((Y - 1) ** 2).mean()):.3f}')
```


### Defining and Training the Model from Scratch

We apply dropout to the outputs of each hidden layer right after its ReLU activation (:numref:`fig_dropout2`)—active *only* during training (`autograd.is_training()` in MXNet, `self.training` in PyTorch/TensorFlow, and `not self.deterministic` in Flax/JAX). Below we define our He-initialized two-hidden-layer classifier `DropoutMLPScratch` ($784 \to 256 \to 256 \to 10$) with separate dropout probabilities `dropout_1` and `dropout_2`:

![Dropout silences a random subset of hidden units ($h_2$ and $h_5$ shown) and their incident connections on each training step.](../img/dropout2.svg)
:label:`fig_dropout2`

```{.python .input #dropout-defining-the-model}
%%tab mxnet
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784):
        super().__init__()
        self.save_hyperparameters()
        he = lambda name, n_in, n_out: gluon.Parameter(  # Variance 2 / n_in
            name, shape=(n_in, n_out), init=init.Normal((2 / n_in) ** 0.5))
        zero = lambda name, n: gluon.Parameter(name, shape=(n,),
                                               init=init.Zero())
        self.W1 = he('W1', num_inputs, num_hiddens_1)
        self.W2 = he('W2', num_hiddens_1, num_hiddens_2)
        self.W3 = he('W3', num_hiddens_2, num_outputs)
        self.b1 = zero('b1', num_hiddens_1)
        self.b2 = zero('b2', num_hiddens_2)
        self.b3 = zero('b3', num_outputs)
        self.initialize()

    def forward(self, X):
        X = X.reshape(-1, self.num_inputs)
        H1 = npx.relu(mnp.dot(X, self.W1.data()) + self.b1.data())
        if autograd.is_training():
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = npx.relu(mnp.dot(H1, self.W2.data()) + self.b2.data())
        if autograd.is_training():
            H2 = dropout_layer(H2, self.dropout_2)
        return mnp.dot(H2, self.W3.data()) + self.b3.data()
```

```{.python .input #dropout-defining-the-model}
%%tab pytorch
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784):
        super().__init__()
        self.save_hyperparameters()
        he = lambda n_in, n_out: nn.Parameter(  # Variance 2 / n_in
            d2l.normal(0, (2 / n_in) ** 0.5, (n_in, n_out)))
        self.W1 = he(num_inputs, num_hiddens_1)
        self.W2 = he(num_hiddens_1, num_hiddens_2)
        self.W3 = he(num_hiddens_2, num_outputs)
        self.b1 = nn.Parameter(d2l.zeros(num_hiddens_1))
        self.b2 = nn.Parameter(d2l.zeros(num_hiddens_2))
        self.b3 = nn.Parameter(d2l.zeros(num_outputs))

    def forward(self, X):
        X = X.reshape(-1, self.num_inputs)
        H1 = torch.relu(X @ self.W1 + self.b1)
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = torch.relu(H1 @ self.W2 + self.b2)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return H2 @ self.W3 + self.b3
```

```{.python .input #dropout-defining-the-model}
%%tab tensorflow
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784):
        super().__init__()
        self.save_hyperparameters()
        def he(name, n_in, n_out):  # Normal weights with variance 2 / n_in
            normal = tf.random_normal_initializer(stddev=(2 / n_in) ** 0.5)
            return self.add_weight(name=name, shape=(n_in, n_out),
                                   initializer=normal)
        zero = lambda name, n: self.add_weight(name=name, shape=(n,),
                                               initializer='zeros')
        self.W1 = he('W1', num_inputs, num_hiddens_1)
        self.W2 = he('W2', num_hiddens_1, num_hiddens_2)
        self.W3 = he('W3', num_hiddens_2, num_outputs)
        self.b1 = zero('b1', num_hiddens_1)
        self.b2 = zero('b2', num_hiddens_2)
        self.b3 = zero('b3', num_outputs)

    def forward(self, X):
        X = tf.reshape(X, (-1, self.num_inputs))
        H1 = tf.nn.relu(X @ self.W1 + self.b1)
        if self.training:
            H1 = dropout_layer(H1, self.dropout_1)
        H2 = tf.nn.relu(H1 @ self.W2 + self.b2)
        if self.training:
            H2 = dropout_layer(H2, self.dropout_2)
        return H2 @ self.W3 + self.b3
```

```{.python .input #dropout-defining-the-model}
%%tab jax
class DropoutMLPScratch(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        self.rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = lambda n_in, n_out: nnx.Param(  # Variance 2 / n_in
            self.rngs.params.normal((n_in, n_out)) * (2 / n_in) ** 0.5)
        self.W1 = he(num_inputs, num_hiddens_1)
        self.W2 = he(num_hiddens_1, num_hiddens_2)
        self.W3 = he(num_hiddens_2, num_outputs)
        self.b1 = nnx.Param(jnp.zeros(num_hiddens_1))
        self.b2 = nnx.Param(jnp.zeros(num_hiddens_2))
        self.b3 = nnx.Param(jnp.zeros(num_outputs))
        self.deterministic = False

    def set_view(self, *, deterministic):
        self.deterministic = deterministic

    def forward(self, X):
        X = X.reshape(-1, self.num_inputs)
        H1 = nnx.relu(X @ self.W1 + self.b1)
        if not self.deterministic:
            H1 = dropout_layer(H1, self.dropout_1, self.rngs.dropout())
        H2 = nnx.relu(H1 @ self.W2 + self.b2)
        if not self.deterministic:
            H2 = dropout_layer(H2, self.dropout_2, self.rngs.dropout())
        return H2 @ self.W3 + self.b3
```

We now instantiate `DropoutMLPScratch` with `dropout_1 = 0.2` after the first hidden layer, `dropout_2 = 0.5` after the second, and learning rate `lr = 0.1` (packing the hyperparameters into `hparams` for reuse below), and train it for $20$ epochs on Fashion-MNIST:

```{.python .input #dropout-training}
hparams = {'num_outputs': 10, 'num_hiddens_1': 256, 'num_hiddens_2': 256,
           'dropout_1': 0.2, 'dropout_2': 0.5, 'lr': 0.1}
model = DropoutMLPScratch(**hparams)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=20, num_gpus=1)
trainer.fit(model, data)
```


Notice that during the opening epochs, training loss sits *above* validation loss because training loss is evaluated with noisy dropout masks active while validation loss runs the full deterministic network.

### Concise Implementation

Using framework `Dropout` layers inside `Sequential`, we can express the exact same two-hidden-layer architecture in a few lines:

```{.python .input #dropout-concise-implementation-1}
%%tab mxnet
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential()
        self.net.add(nn.Dense(num_hiddens_1, activation='relu'),
                     nn.Dropout(dropout_1),
                     nn.Dense(num_hiddens_2, activation='relu'),
                     nn.Dropout(dropout_2),
                     nn.Dense(num_outputs))
        self.net.initialize(init.Xavier(rnd_type='gaussian', factor_type='in',
                                        magnitude=2))
```

```{.python .input #dropout-concise-implementation-1}
%%tab pytorch
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(num_hiddens_1), nn.ReLU(),
            nn.Dropout(dropout_1), nn.LazyLinear(num_hiddens_2), nn.ReLU(),
            nn.Dropout(dropout_2), nn.LazyLinear(num_outputs))

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            nn.init.zeros_(module.bias)
```

```{.python .input #dropout-concise-implementation-1}
%%tab tensorflow
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr):
        super().__init__()
        self.save_hyperparameters()
        # One initializer per layer: an instance repeats its random values
        he = lambda: tf.keras.initializers.VarianceScaling(
            2.0, 'fan_in', 'untruncated_normal')
        self.net = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(num_hiddens_1, activation='relu',
                                  kernel_initializer=he()),
            tf.keras.layers.Dropout(dropout_1),
            tf.keras.layers.Dense(num_hiddens_2, activation='relu',
                                  kernel_initializer=he()),
            tf.keras.layers.Dropout(dropout_2),
            tf.keras.layers.Dense(num_outputs, kernel_initializer=he())])
```

```{.python .input #dropout-concise-implementation-1}
%%tab jax
class DropoutMLP(d2l.Classifier):
    def __init__(self, num_outputs, num_hiddens_1, num_hiddens_2,
                 dropout_1, dropout_2, lr, num_inputs=784, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        he = nnx.initializers.variance_scaling(2.0, 'fan_in', 'normal')
        self.net = nnx.Sequential(
            nnx.Linear(num_inputs, num_hiddens_1, kernel_init=he, rngs=rngs),
            nnx.relu, nnx.Dropout(dropout_1, rngs=rngs),
            nnx.Linear(num_hiddens_1, num_hiddens_2, kernel_init=he,
                       rngs=rngs),
            nnx.relu, nnx.Dropout(dropout_2, rngs=rngs),
            nnx.Linear(num_hiddens_2, num_outputs, kernel_init=he, rngs=rngs))

    def forward(self, X):
        return self.net(X.reshape((X.shape[0], -1)))  # Flatten each image
```


## Effects of Dropout under Label Noise

On the full $60{,}000$-image Fashion-MNIST dataset over 20 epochs, an unregularized MLP barely overfits, leaving little room for dropout to shine. To watch dropout combat severe memorization, we return to the $1{,}000$-image noisy-label benchmark from :numref:`subsec_early_stopping`, corrupting $40\%$ of the training labels at random via `d2l.corrupt` ($\sim 36\%$ genuinely wrong labels, indexed by the Boolean mask `wrong = labels != y`):

```{.python .input #dropout-noisy-labels-1}
images = d2l.FashionMNIST(batch_size=2000)
X, y = (d2l.numpy(a) for a in next(iter(images.train_dataloader())))
X_val, y_val = (d2l.numpy(a) for a in next(iter(images.val_dataloader())))
X, y, X_val = X[:1000].reshape(1000, -1), y[:1000], X_val.reshape(2000, -1)
labels = d2l.corrupt(y, 0.4, np.random.default_rng())
wrong = labels != y  # The training labels that are now wrong
```

We now train `DropoutMLP` for $200$ epochs on `(X, labels)` with dropout probability `p = 0` (no dropout) versus `p = 0.5` in both hidden layers, using `d2l.Tracker` to record accuracy on the corrupted training subset `X[wrong]` alongside held-out validation accuracy on `X_val`:

```{.python .input #dropout-noisy-labels-2}
acc = {}
for p in (0, 0.5):
    trainer = d2l.Tracker(max_epochs=200, datasets={
        'wrong labels': d2l.ArrayData(X[wrong], labels[wrong], 1000),
        'validation': d2l.ArrayData(X_val, y_val, 1000)})
    trainer.fit(DropoutMLP(**{**hparams, 'dropout_1': p, 'dropout_2': p}),
                d2l.ArrayData(X, labels))
    acc[p] = trainer.accuracy
    print(f'dropout {p}: wrong labels reproduced '
          f'{acc[p]["wrong labels"][-1]:.2f}, validation accuracy over the '
          f'last 10 epochs {np.mean(acc[p]["validation"][-10:]):.3f}')
d2l.plot(list(range(1, 201)), [acc[0]['wrong labels'], acc[0.5]['wrong labels'],
                               acc[0]['validation'], acc[0.5]['validation']],
         'epoch', 'accuracy', fmts=('C0--', 'C1--', 'C0-', 'C1-'),
         legend=['wrong labels, p = 0', 'wrong labels, p = 0.5',
                 'validation, p = 0', 'validation, p = 0.5'],
         legend_loc='outside', figsize=(4.5, 3))
```


Look at the two trajectories:

- **Without dropout ($p = 0$):** By epoch 100, the MLP memorizes $100\%$ of the corrupted training labels, and validation accuracy degrades by $\sim 10$ percentage points from its early peak.
- **With dropout ($p = 0.5$):** Memorization of wrong labels is strongly damped, leaving validation accuracy over the final 10 epochs **3 to 5 percentage points higher** than the unregularized network!

In practice, dropout is most valuable precisely when parameter count dwarfs dataset size and training runs for many epochs (as in early-2010s vision classifiers like AlexNet in :numref:`sec_alexnet`); in single-pass foundation-model pretraining or directly before batch normalization layers (:numref:`sec_batch_norm`, where dropout's $1/(1-p)$ training variance inflation creates a train--test variance shift :cite:`Li.Chen.Hu.ea.2019`), dropout is typically reduced or omitted.

## Summary

- **Inverted dropout:** Zeroing each hidden activation independently with probability $p$ and rescaling survivors by $1/(1 - p)$ during training preserves $\mathbb{E}[h_j'] = h_j$ while injecting variance $\frac{p}{1-p}h_j^2$, allowing the full unscaled network to run deterministically at test time.
- **Resistance to noise memorization:** On small datasets with corrupted labels, dropout substantially slows the memorization of wrong labels and preserves higher held-out accuracy, complementing early stopping and weight decay.

## Exercises

1. [code] **Dropout probabilities.** The model of this section drops units
   with probability 0.2 after the first hidden layer and 0.5 after the
   second.
    1. Train it with the two probabilities swapped, with both set to 0.5,
       and with both set to 0.8. For each run, report the best validation
       accuracy. Which layer tolerates the higher probability?
    1. The `Trainer` computes the training loss that it plots with the masks
       switched on. After training, evaluate the loss on the training images
       with the masks switched off, and compare the gap between training and
       validation loss for the networks trained with and without dropout.
1. **Dropout and He initialization.** The He initialization of
   :numref:`subsec_he_init` chooses the variance of the weights so that the
   mean square of the pre-activations stays the same from layer to layer.
    1. During training, by what factor does dropout with probability $p$
       change the mean square of the pre-activations of the next layer?
       Give the factor for each of the two dropout layers of this section's
       model.
    1. In a network with $K - 1$ hidden layers, each followed by dropout
       with probability $p$, how does the factor grow with $K$? Should the
       initialization correct for it, given that the network runs without
       dropout at test time? Relate your answer to the mismatch between
       training and test that dropout causes before a batch normalization
       layer.

1. [code] **Monte Carlo dropout.** Instead of switching dropout off at test
   time, keep it on in the trained `DropoutMLPScratch` and run $T = 20$
   forward passes per validation image, then average the softmax outputs
   :cite:`Gal.Ghahramani.2016`. Compare the accuracy and the calibration
   (the mean of the largest softmax probability against the accuracy) with
   those of the single pass of the full network. How does the accuracy
   change as $T$ grows from 1 to 20?
   :exlabel:`ex_dropout-monte-carlo-dropout`
1. [code] **Dropout and weight decay.** Weight decay and dropout both
   regularize the model of this section, by different mechanisms. Measure
   each run by its best validation accuracy and by the gap between training
   and validation loss, with both losses evaluated with the masks switched
   off.
    1. With both dropout probabilities set to zero, add weight decay as in
       :numref:`sec_weight_decay` at strengths
       $\lambda \in \{10^{-4}, 10^{-3}, 10^{-2}\}$. Compare these runs
       with the runs with and without dropout of this section.
    1. Keep the dropout probabilities at 0.2 and 0.5, and add weight decay
       at the same strengths. Are the effects of the two regularizers
       additive, do they show diminishing returns, or do they cancel each
       other out?
1. **Dropout at test time.** ● Set `dropout_1` to zero, so that only the
   second hidden layer is masked, and let $\boldsymbol{\xi}$ range over its
   masks. Let $\mathbf{o}(\boldsymbol{\xi})$ and $\hat{\mathbf{y}}(\boldsymbol{\xi})$
   be the logits and the softmax output under mask $\boldsymbol{\xi}$, which
   occurs with probability $P(\boldsymbol{\xi})$. Show that the logits of the
   full network equal $\sum_{\boldsymbol{\xi}} P(\boldsymbol{\xi})\, \mathbf{o}(\boldsymbol{\xi})$.
   Show further that its softmax output equals the normalized weighted
   geometric mean

    $$\tilde{y}_j = \frac{\prod_{\boldsymbol{\xi}} \hat{y}_j(\boldsymbol{\xi})^{P(\boldsymbol{\xi})}}{\sum_k \prod_{\boldsymbol{\xi}} \hat{y}_k(\boldsymbol{\xi})^{P(\boldsymbol{\xi})}}.$$

    Explain why neither equality holds in general once `dropout_1` is
    nonzero.
   :exlabel:`ex_dropout-geometric-mean`

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/100)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/101)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/261)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17987)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.6]{.kicker}

Dropout: Regularization via Multiplicative Hidden Noise<br>**co-adaptation · inverted dropout $1/(1-p)$ · train vs. test semantics · slowing noisy-label memorization**
:::
:::

::: {.slide title="Why Inject Random Noise Into Hidden Units?"}
[Motivation & Co-Adaptation]{.kicker}

In §4.5 we saw that overparameterized MLPs trained for many epochs on noisy data eventually memorize wrong labels and degrade on unseen images. How can we disrupt memorization *during* each forward pass?

- **Three complementary regularizers:** Weight decay (§2.7) penalizes large weight norms $\|\mathbf{W}\|_{\textrm{F}}^2$; early stopping (§4.5) halts training before late-stage noise memorization; **dropout** (Hinton et al., 2012; Srivastava et al., 2014) randomly zeroes out each hidden unit with probability $p$ on every training step.
- **Preventing hidden-unit co-adaptation:** When units are always present together, groups of neurons can conspire to fix isolated training errors (*co-adaptation*). Random masking forces every unit to extract features that remain useful alongside random subsets of partners.
:::

::: {.slide title="Warm-Up: Input Noise Equals L2 Weight Decay"}
[Noise as Regularization]{.kicker}

Why does injecting zero-mean noise during training act as a regularizer? Consider a linear predictor $\mathbf{w}^\top\mathbf{x} + b$ trained under squared error when inputs are perturbed by $\mathbf{x} + \boldsymbol{\epsilon}$ with i.i.d. noise of mean $0$ and variance $s^2$ (Bishop, 1995):

$$\mathbb{E}_{\boldsymbol{\epsilon}}\!\left[\bigl(\mathbf{w}^\top(\mathbf{x} + \boldsymbol{\epsilon}) + b - y\bigr)^2\right] = (\mathbf{w}^\top\mathbf{x} + b - y)^2 + 2(\mathbf{w}^\top\mathbf{x} + b - y)\,\mathbf{w}^\top\underbrace{\mathbb{E}[\boldsymbol{\epsilon}]}_{\mathbf{0}} + \mathbb{E}\!\left[(\mathbf{w}^\top\boldsymbol{\epsilon})^2\right]$$

$$\mathbb{E}_{\boldsymbol{\epsilon}}\!\left[\bigl(\mathbf{w}^\top(\mathbf{x} + \boldsymbol{\epsilon}) + b - y\bigr)^2\right] = (\mathbf{w}^\top\mathbf{x} + b - y)^2 + s^2\|\mathbf{w}\|_2^2$$

::: {.d2l-note .rule}
Expectation of noisy squared error equals clean squared error **plus an exact $\ell_2$ ridge penalty $s^2\|\mathbf{w}\|_2^2$**! Dropout extends noise injection from small additive input jitter to multiplicative binary masking across every hidden layer.
:::
:::

::: {.slide title="Inverted Dropout: Masking and 1/(1 − p) Rescaling"}
[Mathematical Formulation]{.kicker}

For each hidden activation $h_j$ and dropout probability $p \in [0, 1)$, on every training forward pass we draw an independent Bernoulli gate $\xi_j \sim \mathrm{Bernoulli}(1 - p)$ ($P(\xi_j = 0) = p$, $P(\xi_j = 1) = 1 - p$) and compute:

$$h_j' = \frac{\xi_j\, h_j}{1 - p} = \begin{cases} 0 & \textrm{with probability } p \\ \dfrac{h_j}{1 - p} & \textrm{with probability } 1 - p \end{cases}$$

- **Unbiased expectation ($\mathbb{E}[h_j'] = h_j$):** Dividing surviving units by $1 - p$ during training ensures $\mathbb{E}[h_j'] = p \cdot 0 + (1 - p)\frac{h_j}{1 - p} = h_j$.
- **Zero test-time overhead:** Because training activations already have the full expected scale $h_j$, **at inference time we simply disable dropout ($h_j' = h_j$)** with no weight rescaling needed.
:::

::: {.slide title="Variance Inflation Under Inverted Dropout"}
[Second-Moment Scaling]{.kicker}

While inverted dropout preserves the mean $\mathbb{E}[h_j'] = h_j$, it injects substantial multiplicative variance proportional to $h_j^2$:

$$\mathbb{E}\bigl[(h_j')^2\bigr] = (1 - p)\,\frac{h_j^2}{(1 - p)^2} = \frac{h_j^2}{1 - p}, \qquad \mathrm{Var}[h_j'] = \mathbb{E}\bigl[(h_j')^2\bigr] - h_j^2 = \frac{p}{1 - p}\,h_j^2$$

| Dropout Rate $p$ | Survival Rate $1 - p$ | Survivor Scale $\frac{1}{1-p}$ | Second-Moment Inflation $\frac{1}{1-p}$ | Variance $\mathrm{Var}[h_j']$ (for $h_j = 1$) |
|:--:|:--:|:--:|:--:|:--:|
| $p = 0.0$ | $100\%$ | $1.00\times$ | $1.00\times$ | $0.00$ |
| $p = 0.2$ | $80\%$ | $1.25\times$ | $1.25\times$ | $0.25$ ($\textrm{std} = 0.5$) |
| $p = 0.5$ | $50\%$ | $2.00\times$ | $2.00\times$ | $1.00$ ($\textrm{std} = 1.0$) |
| $p = 0.8$ | $20\%$ | $5.00\times$ | $5.00\times$ | $4.00$ ($\textrm{std} = 2.0$) |
:::

::: {.slide title="Adaptive Ridge Penalty vs. Nonlinear Test-Time Mismatch"}
[Linear Exactness vs. ReLU Approximation]{.kicker}

How does the deterministic test-time network ($h_j' = h_j$) relate to the $2^h$ random subnetworks sampled during training?

::: {.cols}
::: {.col}
**1. Exact adaptive ridge on a linear layer:**

$$\mathbb{E}_{\boldsymbol{\xi}}\!\left[(\mathbf{w}^\top\mathbf{x}' + b - y)^2\right] = (\mathbf{w}^\top\mathbf{x} + b - y)^2 + \frac{p}{1 - p}\sum_{j=1}^d x_j^2 w_j^2$$

- Penalizes each weight $w_j^2$ in proportion to its feature energy $x_j^2$ (frequent features are regularized more strongly).
:::

::: {.col}
**2. Nonlinear expectation mismatch ($\mathbb{E}[\mathrm{ReLU}(z)] \neq \mathrm{ReLU}(\mathbb{E}[z])$):**

| $(h_1', h_2')$ at $p = 0.5$ | $(0, 0)$ | $(2, 0)$ | $(0, 2)$ | $(2, 2)$ |
|:--|--:|--:|--:|--:|
| Pre-activation $z = h_1' - h_2'$ | $0$ | $+2$ | $-2$ | $0$ |
| $\mathrm{ReLU}(z) = \max(z, 0)$ | $0$ | $2$ | $0$ | $0$ |

- Average of thinned ReLUs is $\mathbb{E}[\mathrm{ReLU}(z)] = 0.5$, whereas deterministic test network gives $\mathrm{ReLU}(\mathbb{E}[z]) = \mathrm{ReLU}(0) = 0$.
:::
:::
:::

::: {.slide title="Implementing Inverted Dropout in Three Lines of Code"}
[From-Scratch Layer]{.kicker}

Our `dropout_layer(X, dropout)` draws uniform random numbers in $[0, 1)$, keeps entries $\ge p$, and rescales by $1 / (1 - p)$:

@dropout-implementation-from-scratch-1

@!dropout-implementation-from-scratch-2
:::

::: {.slide title="Where Dropout Sits in a Multi-Layer MLP"}
[Architecture & Mode Switching]{.kicker}

We insert dropout immediately after each hidden layer's ReLU activation—active **only during training** and bypassed during validation/inference:

::: {.cols .vc}
::: {.col}
- **Two-hidden-layer Fashion-MNIST MLP:** $784 \to 256 \to 256 \to 10$ ($269{,}322$ parameters, He-initialized).
- **Layerwise dropout schedule:** Lower dropout near the input (`dropout_1 = 0.2`) to preserve raw feature signal; stronger dropout deeper (`dropout_2 = 0.5`) where co-adaptation risk is highest.
- **Backward pass automatically masked:** Because $h_j' = \frac{\xi_j}{1 - p}h_j$, autograd multiplies gradients through dropped units by $0$ and surviving units by $\frac{1}{1 - p}$.
:::

::: {.col .fig .big}
![](../img/dropout2.svg){.nostretch width=100%}
:::
:::
:::

::: {.slide title="Defining and Training DropoutMLPScratch"}
[From-Scratch Model]{.kicker}

In `DropoutMLPScratch`, we guard each `dropout_layer` call with the framework's training-mode flag (`self.training`, `autograd.is_training()`, or `not self.deterministic`):

@dropout-defining-the-model
:::

::: {.slide title="20-Epoch Training Curve on Full Fashion-MNIST"}
[Full-Dataset Benchmark]{.kicker}

Training `DropoutMLPScratch` for $20$ epochs on all $60{,}000$ Fashion-MNIST images ($n = 256, \eta = 0.1$) reaches $\sim 87\%$ validation accuracy:

::: {.cols .vc}
::: {.col}
- **Why training loss sits *above* validation loss early on:** `train_loss` is evaluated on noisy, thinned networks ($p_1 = 0.2, p_2 = 0.5$), whereas `val_loss` uses the clean unmasked ensemble approximation.
- **Little overfitting on $60{,}000$ clean images:** Over $20$ epochs, even an unregularized MLP barely overfits the full dataset—to see dropout's true power, we test under label noise next.
:::

::: {.col .fig .big}
@!dropout-training
:::
:::
:::

::: {.slide title="Concise Implementation with Framework Dropout Layers"}
[High-Level API]{.kicker}

Framework `Sequential` containers interleave `Dropout(p)` modules after each `ReLU`, automatically toggling masks between `train` and `eval` modes inside `Trainer`:

@dropout-concise-implementation-1
:::

::: {.slide title="Combatting Memorization Under 40% Label Corruption"}
[Label-Noise Stress Test]{.kicker}

We return to the $1{,}000$-image noisy-label benchmark from §4.5, redrawing $40\%$ of the training labels at random ($\sim 36\%$ genuinely wrong labels) and training `DropoutMLP` for $200$ epochs with $p = 0$ vs. $p = 0.5$ in both hidden layers:

::: {.cols .vc}
::: {.col}
- **No dropout ($p = 0$, blue):** Memorizes $100\%$ of the wrong labels by epoch 100 (`C0--`), dragging validation accuracy (`C0-`) down by $\sim 10$ points from its early peak.
- **With dropout ($p = 0.5$, orange):** Strongly damps wrong-label memorization (`C1--`) and finishes **3 to 5 percentage points higher** in validation accuracy (`C1-`)!
:::

::: {.col .fig .big}
@!dropout-noisy-labels-2
:::
:::
:::

::: {.slide title="Three Complementary Interpretations of Dropout"}
[Theoretical Perspectives]{.kicker}

Why does random binary masking improve generalization across diverse tasks? Three perspectives each capture part of the mechanism:

| Theoretical View | Core Mechanism | Scope & Limitation |
|:--|:--|:--|
| **1. Adaptive Noise Regularization** | Injects multiplicative variance $\mathrm{Var}[h_j'] = \frac{p}{1-p}h_j^2$, penalizing high-energy weights $\propto \frac{p}{1-p}h_j^2 w_j^2$ | Exact closed-form ridge penalty for linear regression; state-dependent in deep nets |
| **2. Anti-Co-Adaptation (Hinton et al.)** | Any pair of units co-occurs with probability $(1 - p)^2$ ($25\%$ at $p = 0.5$), preventing fragile multi-unit conspiracies | Explains why dropout slows the memorization of isolated wrong labels |
| **3. Implicit Geometric Ensemble** | Shares weights across $2^h$ thinned subnetworks ($2^{512} \approx 10^{154}$ masks in our 2-layer MLP) | Exact geometric mean of softmax probabilities only for the final hidden layer; approximate across deep ReLUs |
:::

::: {.slide title="When to Use Dropout—and When to Avoid It"}
[Modern Engineering Guidelines]{.kicker}

Comparing our two experiments with modern deep-learning architectures clarifies where dropout belongs in your toolkit:

| Regime / Architecture | Typical Dropout Rate $p$ | Engineering Rationale |
|:--|:--:|:--|
| **Large dense layers on small/noisy data** (e.g., AlexNet classifier head, tabular MLPs) | $p \in [0.2, 0.5]$ | Parameters dwarf sample count over many epochs; strong co-adaptation risk |
| **Multi-epoch Transformers / Vision backbones** | $p \approx 0.1$ | Light regularization on residual/attention outputs across multi-epoch runs |
| **Single-epoch LLM pretraining** (trillions of tokens, $1$ pass per document) | $p = 0.0$ | Every gradient step sees fresh data; memorization cannot build across epochs |
| **Directly before Batch Normalization (§7.3)** | **Avoid ($p = 0.0$)** | Dropout inflates training second moment by $\frac{1}{1-p}$, causing a train/test variance shift in BatchNorm's running statistics (Li et al., 2019) |
:::

::: {.slide title="Concept Check: Inverted Dropout, Partners & Variance"}
[Check]{.kicker}

Test your command of dropout mathematics and systems behavior:

1. **Extreme dropout ($p = 0.8$):** If a hidden unit has activation $h_j = 1.0$ and we apply inverted dropout with $p = 0.8$, what value does a surviving unit take, and what are $\mathbb{E}[h_j']$ and $\mathrm{Var}[h_j']$?
2. **Pairwise co-adaptation probability:** In a 256-unit hidden layer with $p = 0.5$, how many units survive on average per example, and on what fraction of training steps are three specific partner units $(j_1, j_2, j_3)$ *all* active together?
3. **Forgetting to turn off dropout at test time:** What symptoms appear if you accidentally leave `self.training = True` during validation?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. **At $p = 0.8$:** Survivors are scaled to $1 / (1 - 0.8) = 5.0$ (occurring $20\%$ of the time). The mean is $\mathbb{E}[h_j'] = 0.2 \times 5.0 = 1.0$, and the variance is $\mathrm{Var}[h_j'] = \frac{0.8}{0.2}(1.0)^2 = 4.0$ (standard deviation $2.0 = 2|h_j|$).
2. **Partner survival:** On average $256 \times 0.5 = 128$ units survive, and any triple $(j_1, j_2, j_3)$ co-occurs on only $(1 - 0.5)^3 = 1/8 = 12.5\%$ of steps.
3. **Test-time bug:** Validation predictions become non-deterministic across calls and suffer lower single-pass accuracy (unless averaged across $T \ge 20$ Monte Carlo passes).
:::
:::

::: {.slide title="Summary: Dropout Regularization"}
[Key Takeaways]{.kicker}

| Aspect | Mathematical / Code Specification | Key Property | Deep Learning Role |
|:--|:--|:--|:--|
| **Inverted Dropout Mask** | $h_j' = \frac{\xi_j h_j}{1 - p}$ with $\xi_j \sim \mathrm{Bernoulli}(1 - p)$ | $\mathbb{E}[h_j'] = h_j$ during training | Leaves test-time network completely unscaled ($h_j' = h_j$) |
| **Injected Noise Variance** | $\mathrm{Var}[h_j'] = \frac{p}{1 - p}h_j^2$, $\mathbb{E}[(h_j')^2] = \frac{h_j^2}{1 - p}$ | $\textrm{std}(h_j') = |h_j|$ at $p = 0.5$ | Large multiplicative perturbation prevents co-adaptation |
| **Linear Regression Limit** | $\mathbb{E}[(\mathbf{w}^\top\mathbf{x}' + b - y)^2] = \textrm{MSE} + \frac{p}{1-p}\sum_j x_j^2 w_j^2$ | Feature-scaled $\ell_2$ ridge penalty | Explains why dropout regularizes high-energy features most |
| **Label-Noise Benchmark** | $1{,}000$ images, $40\%$ labels redrawn, $p = 0.5$ | Cuts wrong-label memorization; $+3$ to $+5$ pts val acc | Complements early stopping when training data is noisy/scarce |
| **Interaction with He Init / BN** | Multiplies pre-activation mean square by $\frac{1}{1 - p}$ in train | $1.25\times$ at $p = 0.2$; $2\times$ at $p = 0.5$ | Explains why dropout before BatchNorm causes train-test shift |
:::

