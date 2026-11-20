```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Numerical Stability and Initialization
:label:`sec_numerical_stability`

In :numref:`sec_mlp-implementation`, initializing our one-hidden-layer MLP with $\mathcal{N}(0, 0.01^2)$ weights caused a sluggish first epoch, yet the network recovered and reached $87\%$ accuracy. What happens if we stack *ten* hidden layers of width $256$ with that same $\mathcal{N}(0, 0.01^2)$ initialization? Ten epochs later, the loss has not budged from $\log 10 \approx 2.30$ and accuracy sits at $10\%$ random guessing! Conversely, if we initialize the weights with $\mathcal{N}(0, 1)$, activations and gradients overflow to `inf` and `NaN` on the very first step.

As we saw in :numref:`sec_backprop`, forward activations and backward gradients pass through a product of $K$ weight and slope matrices across a $K$-layer network. Unless each layer preserves the signal variance—keeping its multiplicative factor near $1$—signals either *vanish* exponentially to zero or *explode* exponentially to infinity. In this section, we quantify per-layer variance propagation, derive *Xavier* and *He* initialization, verify them on a 10-layer MLP on Fashion-MNIST, and prove why initial weights must be *random* to break permutation symmetry among hidden units.

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```


## Vanishing and Exploding Gradients

### A Product over Layers

In a $K$-layer MLP with hidden layers $\mathbf{Z}^{(k)} = \mathbf{H}^{(k-1)}\mathbf{W}^{(k)} + \mathbf{b}^{(k)}$ and $\mathbf{H}^{(k)} = \phi(\mathbf{Z}^{(k)})$ of width $h$, the backward recurrence from :numref:`sec_backprop` propagates activation gradients one layer down via

$$
\frac{\partial L}{\partial \mathbf{H}^{(k-1)}} = \Bigl(\frac{\partial L}{\partial \mathbf{H}^{(k)}} \odot \phi'\bigl(\mathbf{Z}^{(k)}\bigr)\Bigr)\, \mathbf{W}^{(k)\top}.
$$
:eqlabel:`eq_grad-step`

Unrolling :eqref:`eq_grad-step` from the top hidden layer $K - 1$ down to layer $k$ expresses the gradient at layer $k$ as a product of $K - 1 - k$ matrix--slope factors:

$$
\frac{\partial L}{\partial \mathbf{H}^{(k)}} = \frac{\partial L}{\partial \mathbf{H}^{(K-1)}}\;
\underbrace{\odot\, \phi'\bigl(\mathbf{Z}^{(K-1)}\bigr)\, \mathbf{W}^{(K-1)\top}}_{\textrm{layer } K-1}
\;\cdots\;
\underbrace{\odot\, \phi'\bigl(\mathbf{Z}^{(k+1)}\bigr)\, \mathbf{W}^{(k+1)\top}}_{\textrm{layer } k+1}.
$$
:eqlabel:`eq_grad-product`

For a single example with upstream gradient row $\mathbf{g}'$ and downstream gradient row $\mathbf{g}$ across a layer of width $h$ whose weights $w_{ij}$ are drawn i.i.d. with mean $0$ and variance $\tau^2$, each coordinate $g_i = \sum_{j=1}^h w_{ij}\,\phi'(z_j)\,g'_j$ has expected square (treating weights, slopes, and upstream gradients as approximately independent at initialization)

$$\mathbb{E}[g_i^2] = \sum_{j=1}^h \mathbb{E}[w_{ij}^2]\,\mathbb{E}[\phi'(z_j)^2]\,\mathbb{E}[(g'_j)^2] = \underbrace{h\, \tau^2\, \mathbb{E}\bigl[\phi'(z)^2\bigr]}_{\textrm{per-layer factor}}\, \mathbb{E}[(g'_j)^2].$$
:eqlabel:`eq_grad-factor`

Across $t$ layers, the gradient's mean square is multiplied by $\big(h\,\tau^2\,\mathbb{E}[\phi'(z)^2]\big)^t$! For symmetric pre-activations and ReLU, $\phi'(z) \in \{0, 1\}$ is $1$ half the time, so $\mathbb{E}[\phi'(z)^2] = 1/2$ and the per-layer multiplier is $\frac{1}{2}h\tau^2$. At width $h = 256$:
- With $\tau = 0.01$: the multiplier is $256 \times 10^{-4} / 2 = 0.0128$, which after $9$ hops shrinks the gradient mean square by $0.0128^9 \approx 10^{-17}$ (**vanishing gradients**).
- With $\tau = 1$: the multiplier is $256 \times 1 / 2 = 128$, which after $9$ hops inflates the gradient mean square by $128^9 \approx 10^{19}$ (**exploding gradients**).
- With sigmoid activations ($\sigma'(z) \leq 1/4$, so $\mathbb{E}[\sigma'(z)^2] \leq 1/16$): even if $h\tau^2 = 1$, every layer shrinks the gradient mean square by at least a factor of $16$ (:cite:`bengio1994learning`).

Let us verify these three predictions directly on a 10-layer, width-256 network at initialization using `moments`:

```{.python .input #numerical-stability-and-init-vanishing-gradients}
%%tab mxnet
def moments(phi, dphi, scale, depth=10, width=256, n=256):
    """Mean squares of the activations and of their gradients, per layer."""
    Ws = [scale * mnp.random.normal(size=(width, width))
          for _ in range(depth)]
    H, Zs, acts = mnp.random.normal(size=(n, width)), [], []
    for W in Ws:  # forward pass, keeping the pre-activations
        Zs.append(H @ W)
        H = phi(Zs[-1])
        acts.append(float((H ** 2).mean()))
    G, grads = mnp.random.normal(size=(n, width)), []  # dL/dH at the top
    for W, Z in zip(reversed(Ws), reversed(Zs)):  # backward pass
        grads.insert(0, float((G ** 2).mean()))
        G = (G * dphi(Z)) @ W.T  # dL/dH one layer down
    return acts, grads

relu = (npx.relu, lambda Z: (Z > 0).astype('float32'))
sigmoid = (npx.sigmoid, lambda Z: npx.sigmoid(Z) * (1 - npx.sigmoid(Z)))
```

```{.python .input #numerical-stability-and-init-vanishing-gradients}
%%tab pytorch
def moments(phi, dphi, scale, depth=10, width=256, n=256):
    """Mean squares of the activations and of their gradients, per layer."""
    Ws = [scale * torch.randn(width, width) for _ in range(depth)]
    H, Zs, acts = torch.randn(n, width), [], []
    for W in Ws:  # forward pass, keeping the pre-activations
        Zs.append(H @ W)
        H = phi(Zs[-1])
        acts.append(float((H ** 2).mean()))
    G, grads = torch.randn(n, width), []  # dL/dH at the top layer
    for W, Z in zip(reversed(Ws), reversed(Zs)):  # backward pass
        grads.insert(0, float((G ** 2).mean()))
        G = (G * dphi(Z)) @ W.T  # dL/dH one layer down
    return acts, grads

relu = (torch.relu, lambda Z: (Z > 0).float())
sigmoid = (torch.sigmoid, lambda Z: torch.sigmoid(Z) * (1 - torch.sigmoid(Z)))
```

```{.python .input #numerical-stability-and-init-vanishing-gradients}
%%tab tensorflow
def moments(phi, dphi, scale, depth=10, width=256, n=256):
    """Mean squares of the activations and of their gradients, per layer."""
    Ws = [scale * tf.random.normal((width, width)) for _ in range(depth)]
    H, Zs, acts = tf.random.normal((n, width)), [], []
    for W in Ws:  # forward pass, keeping the pre-activations
        Zs.append(H @ W)
        H = phi(Zs[-1])
        acts.append(float(tf.reduce_mean(H ** 2)))
    G, grads = tf.random.normal((n, width)), []  # dL/dH at the top layer
    for W, Z in zip(reversed(Ws), reversed(Zs)):  # backward pass
        grads.insert(0, float(tf.reduce_mean(G ** 2)))
        G = (G * dphi(Z)) @ tf.transpose(W)  # dL/dH one layer down
    return acts, grads

relu = (tf.nn.relu, lambda Z: tf.cast(Z > 0, tf.float32))
sigmoid = (tf.sigmoid, lambda Z: tf.sigmoid(Z) * (1 - tf.sigmoid(Z)))
```

```{.python .input #numerical-stability-and-init-vanishing-gradients}
%%tab jax
def moments(phi, dphi, scale, depth=10, width=256, n=256):
    """Mean squares of the activations and of their gradients, per layer."""
    Ws = [scale * jax.random.normal(d2l.get_key(), (width, width))
          for _ in range(depth)]
    H, Zs, acts = jax.random.normal(d2l.get_key(), (n, width)), [], []
    for W in Ws:  # forward pass, keeping the pre-activations
        Zs.append(H @ W)
        H = phi(Zs[-1])
        acts.append(float((H ** 2).mean()))
    G, grads = jax.random.normal(d2l.get_key(), (n, width)), []  # dL/dH
    for W, Z in zip(reversed(Ws), reversed(Zs)):  # backward pass
        grads.insert(0, float((G ** 2).mean()))
        G = (G * dphi(Z)) @ W.T  # dL/dH one layer down
    return acts, grads

relu = (jax.nn.relu, lambda Z: (Z > 0).astype(jnp.float32))
sigmoid = (jax.nn.sigmoid,
           lambda Z: jax.nn.sigmoid(Z) * (1 - jax.nn.sigmoid(Z)))
```

Let us compare three setups: ReLU with $\tau = 0.01$ (as in
:numref:`sec_mlp-implementation`), ReLU with $\tau = 1$, and sigmoid with
$\tau^2 = 1/256$ (so $h\tau^2 = 1$, isolating pure activation attenuation).

```{.python .input #numerical-stability-and-init-exploding-gradients}
runs = {'ReLU, N(0, 0.01²)': moments(*relu, 0.01),
        'ReLU, N(0, 1)': moments(*relu, 1.0),
        'sigmoid, N(0, 1/256)': moments(*sigmoid, 1 / 16)}
for name, (acts, grads) in runs.items():
    print(f'{name}: gradient mean square at layer 1: {grads[0]:.1e}')
d2l.plot(list(range(1, 11)), [grads for acts, grads in runs.values()],
         'layer', 'mean square of gradient', legend=list(runs),
         yscale='log', figsize=(4.5, 3))
```


On a logarithmic vertical axis, all three curves are straight lines whose slopes match our analytical multiplier :eqref:`eq_grad-factor` ($10^{-17}$, $10^{19}$, and $5 \times 10^{-12}$ at layer 1).

## Parameter Initialization

### Xavier Initialization
:label:`subsec_xavier`

How do we choose the initial weight variance $\tau^2$ so that every layer's multiplier equals $1$? Consider a linear layer $o_j = \sum_{i=1}^{n_{\textrm{in}}} x_i w_{ij}$ with $n_{\textrm{in}}$ inputs (the layer's *fan-in*) and $n_{\textrm{out}}$ outputs (its *fan-out*), where weights $w_{ij}$ are i.i.d. with mean $0$ and variance $\tau^2$ and inputs $x_i$ are uncorrelated with mean $0$ and variance $\gamma^2$. In the forward pass, $\mathbb{E}[o_j] = 0$ and

$$\mathrm{Var}[o_j] = \mathbb{E}[o_j^2] = \sum_{i=1}^{n_{\textrm{in}}} \mathbb{E}[x_i^2]\,\mathbb{E}[w_{ij}^2] = n_{\textrm{in}}\, \tau^2 \gamma^2.$$

Preserving forward variance across the layer requires $n_{\textrm{in}}\tau^2 = 1$. By :eqref:`eq_grad-factor` with unit slope, preserving backward gradient variance requires $n_{\textrm{out}}\tau^2 = 1$. When $n_{\textrm{in}} \neq n_{\textrm{out}}$, *Xavier initialization* (or *Glorot initialization*, :cite:`Glorot.Bengio.2010`) averages the two fan counts:

$$\frac{1}{2}(n_{\textrm{in}} + n_{\textrm{out}})\,\tau^2 = 1 \quad \Longleftrightarrow \quad \tau^2 = \frac{2}{n_{\textrm{in}} + n_{\textrm{out}}},$$

drawing weights either from $\mathcal{N}\!\left(0, \frac{2}{n_{\textrm{in}} + n_{\textrm{out}}}\right)$ or from the variance-matched uniform interval $U\!\left(-\sqrt{\frac{6}{n_{\textrm{in}} + n_{\textrm{out}}}},\, \sqrt{\frac{6}{n_{\textrm{in}} + n_{\textrm{out}}}}\right)$. Xavier initialization is the standard choice for linear and zero-centered unit-slope activations such as $\tanh$.

### He Initialization
:label:`subsec_he_init`

For ReLU layers, zeroing out the negative half of a symmetric zero-mean pre-activation $z$ cuts its mean square in half: $\mathbb{E}[\mathrm{ReLU}(z)^2] = \frac{1}{2}\mathbb{E}[z^2]$. Because $\mathbb{E}[o_j^2] = n_{\textrm{in}}\tau^2\,\mathbb{E}[x_i^2]$ holds for zero-mean independent weights even when the inputs $x_i = \mathrm{ReLU}(z_i)$ have positive mean, the activation mean square $v^{(k)} = \mathbb{E}[(h_i^{(k)})^2]$ evolves across a ReLU layer as

$$v^{(k)} = \frac{1}{2}\, n_{\textrm{in}}\, \tau^2\, v^{(k-1)}.$$

Keeping $v^{(k)} = v^{(k-1)}$ constant across depth therefore requires doubling the variance:

$$\tau^2 = \frac{2}{n_{\textrm{in}}}.$$

This is *He initialization* (or *Kaiming initialization*, :cite:`He.Zhang.Ren.ea.2015`), the default for ReLU networks (note that standard framework layers often default to $1/(3n_{\textrm{in}})$, $1/n_{\textrm{in}}$, or Xavier unless He is requested explicitly).

### Activations, Gradients, and Training across Depth
:label:`subsec_variance_propagation`

Let us compare $\mathcal{N}(0, 0.01^2)$, Xavier ($	au^2 = 1/256$), and He ($	au^2 = 2/256$) across a 10-layer ReLU network—first checking the forward activation and backward gradient mean squares at initialization, and then training a 10-hidden-layer `DeepMLP` on Fashion-MNIST for 10 epochs at learning rate $\eta = 0.05$:

```{.python .input #numerical-stability-and-init-depth-sweep}
runs = {'N(0, 0.01²)': moments(*relu, 0.01),
        'Xavier': moments(*relu, (1 / 256) ** 0.5),
        'He': moments(*relu, (2 / 256) ** 0.5)}
for name, (acts, grads) in runs.items():
    print(f'{name}: activations at layer 10: {acts[-1]:.1e}, '
          f'gradient at layer 1: {grads[0]:.1e}')
d2l.plot(list(range(1, 11)), [acts for acts, grads in runs.values()],
         'layer', 'mean square of activations', legend=list(runs),
         yscale='log', figsize=(4.5, 3))
```


```{.python .input #numerical-stability-and-init-training-a-deep-network-1}
%%tab mxnet
class DeepMLP(d2l.Classifier):
    """A deep MLP whose weights have the standard deviation std(n_in, n_out)."""
    def __init__(self, std, lr, depth=10, width=256):
        super().__init__()
        self.save_hyperparameters(ignore=['std'])
        sizes = [784] + [width] * depth + [10]
        self.net = nn.Sequential()
        for k, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            self.net.add(nn.Dense(
                n_out, activation='relu' if k < depth else None,
                weight_initializer=init.Normal(std(n_in, n_out))))
        self.net.initialize()
```

```{.python .input #numerical-stability-and-init-training-a-deep-network-1}
%%tab pytorch
class DeepMLP(d2l.Classifier):
    """A deep MLP whose weights have the standard deviation std(n_in, n_out)."""
    def __init__(self, std, lr, depth=10, width=256):
        super().__init__()
        self.save_hyperparameters(ignore=['std'])
        sizes = [784] + [width] * depth + [10]
        layers = [nn.Flatten()]
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layer = nn.Linear(n_in, n_out)
            nn.init.normal_(layer.weight, std=std(n_in, n_out))
            nn.init.zeros_(layer.bias)
            layers += [layer, nn.ReLU()]
        self.net = nn.Sequential(*layers[:-1])  # No ReLU after the logits
```

```{.python .input #numerical-stability-and-init-training-a-deep-network-1}
%%tab tensorflow
class DeepMLP(d2l.Classifier):
    """A deep MLP whose weights have the standard deviation std(n_in, n_out)."""
    def __init__(self, std, lr, depth=10, width=256):
        super().__init__()
        self.save_hyperparameters(ignore=['std'])
        sizes = [784] + [width] * depth + [10]
        self.net = tf.keras.models.Sequential([tf.keras.layers.Flatten()])
        for k, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            normal = tf.keras.initializers.RandomNormal(stddev=std(n_in, n_out))
            self.net.add(tf.keras.layers.Dense(
                n_out, activation='relu' if k < depth else None,
                kernel_initializer=normal))
```

```{.python .input #numerical-stability-and-init-training-a-deep-network-1}
%%tab jax
class DeepMLP(d2l.Classifier):
    """A deep MLP whose weights have the standard deviation std(n_in, n_out)."""
    def __init__(self, std, lr, depth=10, width=256, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['std', 'rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        sizes = [784] + [width] * depth + [10]
        layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            normal = nnx.initializers.normal(std(n_in, n_out))
            layers += [nnx.Linear(n_in, n_out, kernel_init=normal, rngs=rngs),
                       nnx.relu]
        self.net = nnx.Sequential(*layers[:-1])  # No ReLU after the logits

    def forward(self, X):
        return self.net(X.reshape((X.shape[0], -1)))  # Flatten each image
```

All three runs share the exact same architecture and hyperparameters,
varying only `std(n_in, n_out)`. Let us compare their training curves and
final validation accuracies.

```{.python .input #numerical-stability-and-init-training-a-deep-network-2}
inits = {'N(0, 0.01²)': lambda n_in, n_out: 0.01,
         'Xavier': lambda n_in, n_out: (2 / (n_in + n_out)) ** 0.5,
         'He': lambda n_in, n_out: (2 / n_in) ** 0.5}
data = d2l.FashionMNIST(batch_size=256)
losses = {}
for name, std in inits.items():
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
    trainer.board.display = False
    trainer.fit(DeepMLP(std, lr=0.05), data)
    losses[name] = trainer.history['train_loss']
    print(f'{name}: validation accuracy {trainer.history["val_acc"][-1]:.2f}')
d2l.plot(list(range(1, 11)), list(losses.values()), 'epoch', 'training loss',
         legend=list(losses), figsize=(4.5, 3))
```


Look at how closely the 10-epoch training curves follow our variance analysis:
- Under $\mathcal{N}(0, 0.01^2)$, layer-10 activations ($\sim 10^{-19}$) and layer-1 gradients ($\sim 10^{-17}$) vanish completely, leaving the training loss frozen at $\log 10 \approx 2.30$ and validation accuracy at $10\%$.
- Under Xavier, losing a factor of $2$ per ReLU layer shrinks signals by $2^{-10} \approx 10^{-3}$, causing a slower first epoch before training recovers to $\sim 85\%$ accuracy.
- Under He initialization, both activations and gradients stay near order $1$ across all 10 layers and training drops smoothly from the very first epoch.

## Symmetry among Hidden Units

Why must initial weights be *random* rather than setting every entry of $\mathbf{W}^{(1)}$ to the deterministic constant $c = \sqrt{2 / n_{\textrm{in}}}$? Because hidden units within a layer are permutation-symmetric: if all columns of $\mathbf{W}^{(1)}$ start identical (and all rows of $\mathbf{W}^{(2)}$ start identical), then every hidden unit computes the exact same forward activation, receives the exact same backward gradient, and takes the exact same SGD update step forever—collapsing a 256-unit layer into a single effective neuron! Let us verify on a 16-unit layer that constant He-scaled weights give identical gradient columns (`0.00` difference between units), whereas random He-scaled weights give every unit a distinct gradient direction:

```{.python .input #numerical-stability-and-init-symmetry-among-hidden-units}
%%tab mxnet
def hidden_grad(W1, W2, X, y):
    """Gradient of the squared loss with respect to the hidden weights."""
    W1 = W1.copy()
    W1.attach_grad()
    with autograd.record():
        L = ((npx.relu(X @ W1) @ W2 - y) ** 2).mean() / 2
    L.backward()
    return W1.grad

d, h = 20, 16
X, y = mnp.random.normal(size=(64, d)), mnp.random.normal(size=(64, 1))
c, c_out = (2 / d) ** 0.5, (1 / h) ** 0.5
inits = {'constant': (mnp.full((d, h), c), mnp.full((h, 1), c_out)),
         'random': (c * mnp.random.normal(size=(d, h)),
                    c_out * mnp.random.normal(size=(h, 1)))}
for name, (W1, W2) in inits.items():
    G = hidden_grad(W1, W2, X, y)
    print(f'{name}: largest difference between two hidden units '
          f'{float(mnp.abs(G - G[:, :1]).max()):.2f}')
```

```{.python .input #numerical-stability-and-init-symmetry-among-hidden-units}
%%tab pytorch
def hidden_grad(W1, W2, X, y):
    """Gradient of the squared loss with respect to the hidden weights."""
    W1 = W1.clone().requires_grad_(True)
    L = ((torch.relu(X @ W1) @ W2 - y) ** 2).mean() / 2
    L.backward()
    return W1.grad

d, h = 20, 16
X, y = torch.randn(64, d), torch.randn(64, 1)
c, c_out = (2 / d) ** 0.5, (1 / h) ** 0.5
inits = {'constant': (torch.full((d, h), c), torch.full((h, 1), c_out)),
         'random': (c * torch.randn(d, h), c_out * torch.randn(h, 1))}
for name, (W1, W2) in inits.items():
    G = hidden_grad(W1, W2, X, y)
    print(f'{name}: largest difference between two hidden units '
          f'{float((G - G[:, :1]).abs().max()):.2f}')
```

```{.python .input #numerical-stability-and-init-symmetry-among-hidden-units}
%%tab tensorflow
def hidden_grad(W1, W2, X, y):
    """Gradient of the squared loss with respect to the hidden weights."""
    W1 = tf.Variable(W1)
    with tf.GradientTape() as tape:
        L = tf.reduce_mean((tf.nn.relu(X @ W1) @ W2 - y) ** 2) / 2
    return tape.gradient(L, W1)

d, h = 20, 16
X, y = tf.random.normal((64, d)), tf.random.normal((64, 1))
c, c_out = (2 / d) ** 0.5, (1 / h) ** 0.5
inits = {'constant': (tf.fill((d, h), c), tf.fill((h, 1), c_out)),
         'random': (c * tf.random.normal((d, h)),
                    c_out * tf.random.normal((h, 1)))}
for name, (W1, W2) in inits.items():
    G = hidden_grad(W1, W2, X, y)
    print(f'{name}: largest difference between two hidden units '
          f'{float(tf.reduce_max(tf.abs(G - G[:, :1]))):.2f}')
```

```{.python .input #numerical-stability-and-init-symmetry-among-hidden-units}
%%tab jax
def hidden_grad(W1, W2, X, y):
    """Gradient of the squared loss with respect to the hidden weights."""
    L = lambda W1: ((jax.nn.relu(X @ W1) @ W2 - y) ** 2).mean() / 2
    return jax.grad(L)(W1)

d, h = 20, 16
X = jax.random.normal(d2l.get_key(), (64, d))
y = jax.random.normal(d2l.get_key(), (64, 1))
c, c_out = (2 / d) ** 0.5, (1 / h) ** 0.5
inits = {'constant': (jnp.full((d, h), c), jnp.full((h, 1), c_out)),
         'random': (c * jax.random.normal(d2l.get_key(), (d, h)),
                    c_out * jax.random.normal(d2l.get_key(), (h, 1)))}
for name, (W1, W2) in inits.items():
    G = hidden_grad(W1, W2, X, y)
    print(f'{name}: largest difference between two hidden units '
          f'{float(jnp.abs(G - G[:, :1]).max()):.2f}')
```


## Summary

- **Exponential variance scaling:** Across a $K$-layer MLP of width $h$ and weight variance $\tau^2$, each layer multiplies activation and gradient mean squares by roughly $h\tau^2\,\mathbb{E}[\phi'(z)^2]$, causing exponential vanishing or explosion unless this factor is close to $1$.
- **Xavier and He initialization:** Xavier initialization sets $\tau^2 = 2/(n_{\textrm{in}} + n_{\textrm{out}})$ for linear and $\tanh$ layers; He initialization sets $\tau^2 = 2/n_{\textrm{in}}$ to compensate for ReLU halving the mean square, enabling deep ReLU networks to train from step 1.
- **Symmetry breaking:** Initializing weights with identical constants locks all hidden units in a layer into identical gradients and updates forever; random initialization is essential to break permutation symmetry.

## Exercises

1. **Spectral norm bounds.** For a single example, written as a column
   vector, the step :eqref:`eq_grad-step` multiplies the gradient
   $\mathbf{g}'$ above layer $k$ by the matrix
   $\mathbf{M}^{(k)} = \mathbf{W}^{(k)} \operatorname{diag}\bigl(\phi'(\mathbf{z}^{(k)})\bigr)$,
   where $\operatorname{diag}(\cdot)$ places the entries of a vector on the
   diagonal of an otherwise zero matrix, so the gradient in
   :eqref:`eq_grad-product` is a product of $K - 1 - k$ such matrices times
   the gradient at the top. The spectral norm (the largest singular value)
   is submultiplicative:
   $\|\mathbf{A}\mathbf{B}\|_2 \le \|\mathbf{A}\|_2 \|\mathbf{B}\|_2$.
    1. Prove submultiplicativity from the definition
       $\|\mathbf{A}\|_2 = \max_{\mathbf{x} \neq 0} \|\mathbf{A}\mathbf{x}\| / \|\mathbf{x}\|$.
    1. Use it to bound the norm of the gradient in terms of the norms
       $\|\mathbf{M}^{(k)}\|_2$ of the factors. Which condition on these
       norms keeps the bound from growing geometrically with depth?
    1. This bound is only an upper bound. Explain why the condition of the
       previous part rules out exploding gradients but not vanishing ones,
       and give two $2 \times 2$ matrices that satisfy the condition with
       equality but whose product has spectral norm strictly smaller than
       one.
1. **Other symmetries.** Permuting the hidden units is not the only change
   of parameters that leaves the function of an MLP unchanged. Consider a
   one-hidden-layer MLP in which hidden unit $j$ has incoming weights
   $\mathbf{w}_j$, bias $b_j$, and outgoing weights $\mathbf{v}_j$.
    1. Prove that $\textrm{ReLU}(\alpha z) = \alpha\, \textrm{ReLU}(z)$ for
       $\alpha > 0$. Use this identity to show that replacing
       $(\mathbf{w}_j, b_j, \mathbf{v}_j)$ by
       $(\alpha \mathbf{w}_j, \alpha b_j, \mathbf{v}_j / \alpha)$ leaves the
       function of a ReLU network unchanged.
    1. Find a transformation of the parameters of a single hidden unit that
       leaves the function of a $\tanh$ network unchanged.
    1. Identical hidden units are a fixed point of the permutation, and
       gradient descent started at that fixed point stays there. Find the
       parameters of a unit that the transformation of the previous part
       leaves unchanged. Then show that if all weights and biases of a
       $\tanh$ network are initialized to zero, gradient descent changes
       only the output bias, at every step.

    *Adapted from Simon Prince,
    [Understanding Deep Learning](https://udlbook.github.io/udlbook/),
    Problems 3.5, 4.3, and 7.15.*
1. **Fan-in versus averaged fans.** Following
   :citet:`He.Zhang.Ren.ea.2015`, consider a linear network with $K$ layers
   in which layer $k$ maps $n_{k-1}$ inputs to $n_k$ outputs and draws its
   weights with variance $\tau_k^2$. By the computation of
   :numref:`subsec_xavier`, layer $k$ multiplies the variance of the forward
   signal by $n_{k-1}\tau_k^2$ and that of the backward signal by
   $n_k\tau_k^2$.
    1. Set $\tau_k^2 = 1/n_{k-1}$ in every layer. Show that the forward
       variance is preserved and that the backward variance changes by the
       total factor $n_K / n_0$ at any depth.
    1. Let the widths alternate as $100, 400, 100, 400, \ldots$, and use
       Xavier initialization in every layer. Compute the factor by which
       one $100 \to 400 \to 100$ pair of layers multiplies the forward
       variance, and the factor after 50 such pairs.
    1. Which of the two rules keeps both variances within a
       depth-independent factor of their initial values for every sequence
       of widths?

1. **Variance under ReLU.** ● The derivation in :numref:`subsec_he_init`
   tracks the mean square $\mathbb{E}[\textrm{ReLU}(z)^2]$ rather than the variance,
   because the next layer's pre-activation $o = \sum_i x_i w_i$, with inputs $x_i = \textrm{ReLU}(z_i)$ and
   independent zero-mean weights has
   $\textrm{Var}[o] = n_\textrm{in}\tau^2 \mathbb{E}[x_i^2]$ for any mean of
   $x_i$.
    1. For $z \sim \mathcal{N}(0, \nu^2)$, compute $\mathbb{E}[\textrm{ReLU}(z)]$ and
       $\textrm{Var}[\textrm{ReLU}(z)]$, and show that the variance equals
       $\left(\tfrac{1}{2} - \tfrac{1}{2\pi}\right)\nu^2$ rather than
       $\tfrac{1}{2}\nu^2$.
    1. A tempting shortcut substitutes $\textrm{Var}[x_i]$ for $\mathbb{E}[x_i^2]$ in
       the Xavier computation. Show that this shortcut prescribes
       $\tau^2 = 2 / \bigl(n_\textrm{in}(1 - 1/\pi)\bigr)$ instead of
       $2/n_\textrm{in}$. Compute the factor by which each ReLU layer
       initialized this way actually multiplies the mean square, and the
       total factor over fifty layers.
1. [code] **Tanh networks.** ● Repeat the measurement of
   :numref:`subsec_variance_propagation` with $\tanh$ in place of ReLU, with
   Xavier initialization and fifty layers. A layer then maps the mean square
   $m$ of its input to $m' = \mathbb{E}[\tanh(z)^2]$, where $z$ is approximately
   $\mathcal{N}(0, m)$.
    1. Use $\tanh(z)^2 \approx z^2 - \tfrac{2}{3}z^4$ for small $z$ to show
       that $m' \approx m - 2m^2$. Over many layers, does the mean square
       stay constant, decay geometrically, or decay more slowly? Show that
       the mean square at layer $k$ is about $1/(2k)$ for large $k$.
    1. Check the prediction by plotting the measured mean square against
       the layer on logarithmic axes. What happens to the gradient?

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/235)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17986)
:end_tab:

