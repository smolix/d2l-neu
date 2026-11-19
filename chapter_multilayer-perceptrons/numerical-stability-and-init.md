```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Numerical Stability and Initialization
:label:`sec_numerical_stability`

Until now, we took parameter initialization for granted—drawing weights
from $\mathcal{N}(0, 0.01^2)$ without asking whether that scale actually
matters. You might even have gotten the impression that initialization is a
minor detail. On the contrary, once we stack hidden layers to harness the
power of depth (:numref:`sec_mlp`), initialization and the choice of
activation function together decide whether a network learns rapidly or
refuses to move at all.

Try initializing a ten-hidden-layer MLP of width 256 with that same
$\mathcal{N}(0, 0.01^2)$ rule. Ten epochs later, its training loss will not
have budged from where it started. Worse, the network was doomed *before*
the very first gradient update. As we saw in :numref:`sec_backprop`,
backpropagation multiplies the error signal by one weight matrix per layer
on its journey back to layer 1. With weights of standard deviation 0.01,
each hop shrinks the mean square of the gradient roughly eightyfold: across
nine hops, the mean square at layer 1 plunges by $10^{17}$, shrinking
typical gradient entries by more than $10^8$. The forward pass suffers the
exact same collapse, arriving at the top hidden layer practically equal to
zero for every image. Crank the initial standard deviation up to 1, and
both signals explode at a comparable exponential rate instead.

Keeping activations and gradients alive across depth requires matching the
initial weight variance to both the layer width and the activation
function. In this section, we measure how signals shrink or blow up across
layers, derive the two classic variance-preserving schemes—*Xavier* and
*He* initialization—verify them numerically on random ten-layer stacks, and
then train a ten-layer MLP on Fashion-MNIST to watch our analysis predict
success and failure firsthand. Finally, we prove a second requirement we
asserted without proof in :numref:`sec_mlp-implementation`: initial weights
must be *random* to break permutation symmetry among hidden units.

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

Why is depth so sensitive to weight scale? The backward pass derived in
:numref:`sec_backprop` gives the answer immediately: as the error signal
travels downward, every layer multiplies it by a matrix of activation
slopes and a transposed weight matrix. Consider a $K$-layer MLP with
$K - 1$ hidden layers of width $h$, $\mathbf{H}^{(0)} = \mathbf{X}$,
pre-activations
$\mathbf{Z}^{(k)} = \mathbf{H}^{(k-1)}\mathbf{W}^{(k)} + \mathbf{b}^{(k)}$,
and activations $\mathbf{H}^{(k)} = \phi(\mathbf{Z}^{(k)})$ (setting
weight decay to zero so the objective is just $L$). Combining the
activation and matrix-product rules moves the activation gradient down one
layer:

$$
\frac{\partial L}{\partial \mathbf{H}^{(k-1)}} = \Bigl(\frac{\partial L}{\partial \mathbf{H}^{(k)}} \odot \phi'\bigl(\mathbf{Z}^{(k)}\bigr)\Bigr)\, \mathbf{W}^{(k)\top}.
$$
:eqlabel:`eq_grad-step`

Unrolling this recurrence from the top hidden layer $K - 1$ down to layer
$k$ exposes a chain of $K - 1 - k$ matrix--slope factors:

$$
\frac{\partial L}{\partial \mathbf{H}^{(k)}} = \frac{\partial L}{\partial \mathbf{H}^{(K-1)}}\;
\underbrace{\odot\, \phi'\bigl(\mathbf{Z}^{(K-1)}\bigr)\, \mathbf{W}^{(K-1)\top}}_{\textrm{layer } K-1}
\;\cdots\;
\underbrace{\odot\, \phi'\bigl(\mathbf{Z}^{(k+1)}\bigr)\, \mathbf{W}^{(k+1)\top}}_{\textrm{layer } k+1},
$$
:eqlabel:`eq_grad-product`

applied left to right—and every weight gradient at layer $k$,
$\mathbf{H}^{(k-1)\top}\bigl(\partial L / \partial \mathbf{H}^{(k)} \odot \phi'(\mathbf{Z}^{(k)})\bigr)$,
inherits this entire chain as a factor.

Whenever you multiply many factors together, the product stays well-behaved
only if each factor hovers right around 1. If every layer rescales the
signal by a constant $\rho$, then $t$ layers rescale it by $\rho^t$ (the
exact exponential behavior of matrix powers we saw in
:numref:`sec_linear-algebra`): even a mild $\rho = 0.9$ shrinks gradients
200-fold across fifty layers, while $\rho = 1.1$ inflates them 117-fold.

What rescaling factor does a single hidden layer apply? Because examples do
not interact across rows of :eqref:`eq_grad-step`, consider one example
with upstream gradient row $\mathbf{g}'$, downstream row $\mathbf{g}$, and
pre-activations $z_j$. Each entry $g_i = \sum_j w_{ij}\, \phi'(z_j)\, g'_j$
squares to a double sum over $(j, j')$ with terms
$w_{ij} w_{ij'}\, \phi'(z_j)\phi'(z_{j'})\, g'_j g'_{j'}$. Assuming the
weights are drawn independently with mean zero and variance $\tau^2$, and
treating them as approximately independent of the slopes and upstream
gradients, every off-diagonal term ($j \neq j'$) averages to zero while the
$h$ diagonal terms ($j = j'$) each contribute
$\tau^2\, \mathbb{E}[\phi'(z_j)^2\, (g'_j)^2]$. Treating slopes and
upstream gradients as independent as well yields
$\mathbb{E}[g_i^2] = h\, \tau^2\, \mathbb{E}[\phi'(z)^2]\, \mathbb{E}[(g'_j)^2]$.
Each layer therefore multiplies the mean square of the gradient by roughly

$$h\, \tau^2\, \mathbb{E}\bigl[\phi'(z)^2\bigr].$$
:eqlabel:`eq_grad-factor`

Notice the deliberate qualifier "roughly": assuming weights are independent
of the slopes $\phi'(z_j)$ is a heuristic leap, since those very weights
helped compute $z_j$ during the forward pass. In wide layers—where any
single weight $w_{ij}$ is only one of $h$ terms entering $z_j$—the
approximation is remarkably accurate, as we are about to test numerically.

Two knobs govern :eqref:`eq_grad-factor`: weight scale ($h\tau^2$) and
activation slope ($\mathbb{E}[\phi'(z)^2]$). When pre-activations are
symmetric around zero, ReLU has slope 1 half the time and 0 the other half,
so $\mathbb{E}[\phi'(z)^2] = 1/2$. For width $h = 256$, our previous
$\tau = 0.01$ produces $256 \times 10^{-4} / 2 = 0.0128$ per layer, while
$\tau = 1$ produces a massive $128$ per layer!

Let us put this formula to the test on a ten-hidden-layer, width-256
network (without biases) at initialization. Our helper `moments` runs a
forward pass on 256 standard normal inputs, seeds the top hidden layer with
a unit-variance random gradient so we isolate pure depth attenuation, steps
backward via :eqref:`eq_grad-step`, and records the mean squares of both
activations and gradients at every layer.

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

On a log scale, all three curves march along crisp straight lines—confirming
that each layer rescales the gradient by a constant multiplier. Across the
nine hops from layer 10 down to layer 1, :eqref:`eq_grad-factor` forecasts
$0.0128^9 \approx 10^{-17}$ for small ReLU weights and
$128^9 \approx 10^{19}$ for large ones, and our simulation hits both right
on the nose! Our independence heuristic passes with flying colors. Even
with $h\tau^2 = 1$, the sigmoid sheds more than an order of magnitude per
layer, slashing the layer-1 gradient mean square to roughly
$5 \times 10^{-12}$.

### Vanishing Gradients

The two plummeting lines visualize the *vanishing gradient* problem we
first glimpsed for sigmoids in :numref:`subsec_activation-functions`:
early layers receive gradients orders of magnitude tinier than top layers,
so any learning rate small enough to keep upper layers stable leaves the
bottom layers frozen in place. Both under-scaled weights and saturating
activations cause vanishing gradients. Because $\sigma'(z) \le 1/4$
everywhere, $\mathbb{E}[\sigma'(z)^2] \le 1/16$, meaning every sigmoid
layer cuts the gradient's mean square by at least a factor of 16 even when
$h\tau^2 = 1$. Trying to compensate with bigger weights backfires
immediately: larger pre-activations push units out of the narrow Goldilocks
zone near zero and into the flat saturating tails where $\sigma'(z) \approx 0$
:cite:`bengio1994learning`. ReLU, by contrast, preserves unit slope for
every positive pre-activation, leaving weight scale as our sole knob—a
major reason ReLU displaced sigmoid in hidden layers.

### Exploding Gradients

The climbing curve displays the mirror disaster: *exploding gradients*.
With a layer-1 mean square $10^{19}$ times larger than layer 10's, typical
gradient entries at the bottom are roughly $3 \times 10^9$ times larger
than at the top! No single learning rate can bridge that gulf: shrink it
for layer 1 and top layers freeze; tune it for top layers and layer 1
catapults to infinity. Worse, forward activations grow at the exact same
rate (:numref:`subsec_he_init`): once values exceed float32's ceiling of
roughly $3.4 \times 10^{38}$, they overflow to `inf` and poison the loss
with `NaN`. Whenever you see `NaN` loss on step 1 of training, oversized
initial weights are suspect number one.

## Parameter Initialization

### Xavier Initialization
:label:`subsec_xavier`

How should we pick $\tau^2$ to pin every layer's multiplier right at 1?
Begin with a linear layer (no activation) having $n_\textrm{in}$ inputs
$x_i$ and $n_\textrm{out}$ outputs,

$$o_j = \sum_{i=1}^{n_\textrm{in}} x_i w_{ij}.$$

Draw weights $w_{ij}$ i.i.d. from any zero-mean distribution with variance
$\tau^2$ (not necessarily Gaussian), and take zero-mean inputs $x_i$ of
variance $\gamma^2$, independent of the weights and of one another. Then
$\mathbb{E}[o_j] = \sum_i \mathbb{E}[x_i]\,\mathbb{E}[w_{ij}] = 0$, and

$$
\begin{aligned}
\textrm{Var}[o_j] = \mathbb{E}[o_j^2]
&= \sum_{i=1}^{n_\textrm{in}} \sum_{i'=1}^{n_\textrm{in}} \mathbb{E}[x_i x_{i'} w_{ij} w_{i'j}] \\
&= \sum_{i=1}^{n_\textrm{in}} \mathbb{E}[x_i^2]\, \mathbb{E}[w_{ij}^2]
= n_\textrm{in}\, \tau^2 \gamma^2,
\end{aligned}
$$

since all cross terms ($i \neq i'$) vanish by independence and zero mean.
Preserving forward activation variance therefore demands
$n_\textrm{in}\tau^2 = 1$.

Now check the backward pass: by :eqref:`eq_grad-factor` (with slope 1 and
summation over $n_\textrm{out}$ outputs), preserving gradient variance
demands $n_\textrm{out}\tau^2 = 1$. Whenever a layer changes width
($n_\textrm{in} \neq n_\textrm{out}$), we cannot satisfy both conditions
simultaneously! *Xavier initialization* (named after Xavier Glorot,
:cite:`Glorot.Bengio.2010`) strikes a clean compromise by averaging the two:

$$
\frac{1}{2}(n_\textrm{in} + n_\textrm{out})\, \tau^2 = 1,
\qquad \textrm{that is,} \qquad
\tau^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}.
$$

For our Fashion-MNIST MLP's first layer ($n_\textrm{in} = 784$,
$n_\textrm{out} = 256$), Xavier prescribes
$\tau = \sqrt{2/1040} \approx 0.044$—more than four times the 0.01 we used
earlier. If we prefer uniform weights over $[-a, a]$ (whose variance is
$a^2/3$), matching the same variance gives

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}},\ \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

Although derived for linear layers, Xavier works well for zero-centered
activations with unit slope at the origin—such as $\tanh$—as long as
pre-activations remain modest.

### He Initialization
:label:`subsec_he_init`

ReLU breaks linearity dramatically: it zeroes out roughly half of its
inputs! Fortunately, the variance calculation above actually requires only
zero-mean independent weights: $\mathbb{E}[o_j^2] = n_\textrm{in}\tau^2 \mathbb{E}[x_i^2]$
holds even when inputs $x_i$ have *nonzero* mean (as nonnegative ReLU
outputs do!). So we simply track the *mean square* across layers. Because
symmetric weights yield pre-activations $z$ symmetric about zero, ReLU
keeps the positive half and drops the negative half, cutting the mean
square exactly in half:
$\mathbb{E}[\textrm{ReLU}(z)^2] = \tfrac{1}{2}\mathbb{E}[z^2]$.

Denoting the mean square of layer $k$'s activations $\mathbf{H}^{(k)}$ by
$v^{(k)}$, each affine-plus-ReLU block updates it as

$$v^{(k)} = \frac{1}{2}\, n_\textrm{in}\, \tau^2\, v^{(k-1)}.$$

Keeping $v^{(k)}$ steady across depth therefore requires *doubling* the
linear-layer variance:

$$\tau^2 = \frac{2}{n_\textrm{in}}.$$

This is *He initialization* (or *Kaiming initialization*,
:cite:`He.Zhang.Ren.ea.2015`), the universal default for ReLU networks. In
the backward pass, :eqref:`eq_grad-factor` with $\mathbb{E}[\phi'(z)^2] = 1/2$
yields the twin condition $\tau^2 = 2/n_\textrm{out}$, matching exactly on
equal-width layers (the single final logit layer has no ReLU, introducing
at most one harmless factor of 2 that never compounds). For our
Fashion-MNIST first layer, He initialization sets
$\tau = \sqrt{2/784} \approx 0.05$. A simple rule of thumb serves nearly
every standard architecture: *Xavier for $\tanh$, He for ReLU*.

### Default Initialization

What happens if you let your framework pick initial weights on its own?
Comparing each library's default linear-layer initializer against ReLU's
$2/n_\textrm{in}$ reveals a surprise: none of the four defaults to He
initialization, which is why our models from the next section onward set He
initialization explicitly.

:begin_tab:`mxnet`
Gluon's default `initialize()` draws weights uniformly from $[-0.07, 0.07]$
regardless of layer width (zero biases). Its variance,
$0.07^2/3 \approx 0.0016$, matches $2/n_\textrm{in}$ only around
$n_\textrm{in} \approx 1{,}200$.
:end_tab:

:begin_tab:`pytorch`
PyTorch's `nn.Linear` draws weights and biases uniformly from
$[-1/\sqrt{n_\textrm{in}}, 1/\sqrt{n_\textrm{in}}]$, giving weight
variance $1/(3 n_\textrm{in})$—just one-sixth of He's $2/n_\textrm{in}$.
:end_tab:

:begin_tab:`tensorflow`
Keras's `Dense` layer defaults to uniform Xavier initialization with zero
biases—half the variance of He initialization on equal-width layers.
:end_tab:

:begin_tab:`jax`
Flax NNX's `nnx.Linear` defaults to *LeCun normal* (truncated normal of
variance $1/n_\textrm{in}$) with zero biases—again half the variance of He
initialization.
:end_tab:

### Activations and Gradients across Depth
:label:`subsec_variance_propagation`

Let us re-run our ten-layer ReLU diagnostic comparing $\mathcal{N}(0, 0.01^2)$,
Xavier ($\tau^2 = 1/256$), and He ($\tau^2 = 2/256$), now plotting forward
activation mean squares across depth and printing both the top-layer
activations and bottom-layer gradients.

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

He initialization holds both forward activations and backward gradients
within roughly a factor of 3 of unity all the way across ten layers (small
random layer-to-layer fluctuations keep the line from being ruler-flat).
Xavier loses ReLU's factor of $1/2$ at every layer, landing right near the
predicted $2^{-10} \approx 10^{-3}$ at layer 10 and
$2^{-9} \approx 2 \times 10^{-3}$ at layer 1. Meanwhile,
$\mathcal{N}(0, 0.01^2)$ collapses layer-10 activations to a mean square
around $10^{-19}$—passing microscopic values of roughly $3 \times 10^{-10}$
to the output layer for every single input.

### Training a Deep Network

Now for the real verdict: do these initialization diagnostics actually
predict training behavior? Let us train our ten-hidden-layer, width-256
ReLU MLP (`DeepMLP`) on Fashion-MNIST for ten epochs (batch size 256) under
all three initializations. We use a learning rate of 0.05—half that of
:numref:`sec_mlp-implementation`—because at 0.1 a ten-layer plain MLP
occasionally diverges mid-training, for a reason we explain right after the
run.

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

Our analysis nailed the outcome. Under $\mathcal{N}(0, 0.01^2)$, the
ten-layer network is dead on arrival: its loss stays pinned at
$\log 10 \approx 2.30$ across all ten epochs and its validation accuracy
never rises above 10% (pure guessing). Both Xavier and He train smoothly—with
He pulling cleanly ahead on epoch 1 (training loss ~0.85 versus ~1.5 for
Xavier, which pays the per-layer factor-of-2 ReLU tax) and both reaching
around 85% validation accuracy by epoch 10. We did not expect ten dense
layers to outperform a single hidden layer on Fashion-MNIST; the point is
that initialization alone makes the difference between a dead network and a
trainable one.

Remember one crucial caveat: Xavier and He govern only step zero, when
weights are independent of one another and of the inputs. As soon as SGD
begins updating parameters, correlations build up across layers and nothing
automatically locks the per-layer factor at 1—which is why a slightly too
large learning rate can still destabilize a properly initialized deep
network. Keeping signal scales healthy *throughout* training requires
architectural tools we meet in later chapters, notably normalization layers
(:numref:`sec_batch_norm`) and residual connections.

## Symmetry among Hidden Units

Variance preservation fixes the *magnitude* of initial weights, but why
must they be *random*? Wouldn't setting every weight to the same constant
$c$ yield the exact same mean square?

Consider a one-hidden-layer MLP with a single output where every weight in
$\mathbf{W}^{(1)}$ starts at $c$ and every weight in $\mathbf{W}^{(2)}$
starts at $c'$. Because hidden units are completely interchangeable
(permuting them along with their incoming and outgoing weights leaves the
network function identical), every hidden unit computes the exact same
activation in the forward pass, receives the exact same gradient in the
backward pass, and takes the exact same SGD update! Induction guarantees
that all $h$ hidden units stay clones of one another forever: 256 hidden
units collapse into a single neuron copied 256 times. Gradient descent
cannot *break the symmetry* on its own unless random initialization breaks
it first.

Let us verify this directly on a small network ($d = 20, h = 16$, 64 random
examples, squared loss), comparing the maximum difference across columns of
$\partial L / \partial \mathbf{W}^{(1)}$ for constant weights at He scale
$c = \sqrt{2/d}$ versus random Gaussian weights at the exact same scale.

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

Under constant weights, all 16 columns of the gradient agree to the very
last bit (`0.00` difference), locking all 16 units together; under random
weights, every unit receives a distinct gradient from step 1 onward (biases
can safely start at zero because the random weights already break symmetry).

Keeping gradients alive lets us *optimize* a deep network, yet optimization
is only half the battle. In the next section, we turn to the deepest puzzle
of modern deep learning: why do networks with far more parameters than
training examples generalize instead of simply memorizing their data?

## Summary

Because backpropagation multiplies gradients by one activation-slope matrix
and one transposed weight matrix per layer, signals in deep networks vanish
or explode exponentially unless each layer's variance multiplier stays near
1. For a ReLU layer of width $h$ and weight variance $\tau^2$, both forward
activations and backward gradients scale by roughly $h\tau^2/2$ per layer;
saturating activations like sigmoid shrink gradients rapidly even when
$h\tau^2 = 1$. *Xavier initialization* ($\tau^2 = 2/(n_\textrm{in} + n_\textrm{out})$)
preserves variance across linear and $\tanh$ layers, while *He
initialization* ($\tau^2 = 2/n_\textrm{in}$) accounts for ReLU halving the
mean square—enabling our ten-hidden-layer Fashion-MNIST MLP to train from
epoch 1 where $\mathcal{N}(0, 0.01^2)$ stalls completely at $\log 10$.
Finally, initial weights must be sampled randomly to break permutation
symmetry among hidden units so that they can learn distinct features.

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
1. [code] **Linear networks.** Rerun the measurement of
   :numref:`subsec_variance_propagation` without activation function, by
   passing the identity and its derivative to `moments`, with the weight
   scales of $\mathcal{N}(0, 0.01^2)$, Xavier, and He.
    1. Compute the factor $n_\textrm{in}\tau^2$ per layer for each of the
       three scales, and check the factors against the slopes of the
       measured curves.
    1. Compare the ranking of the three initializations with the ranking
       for the ReLU network and explain the difference.
1. [code] **Layerwise adaptive scaling.** ● Write
   $\mathbf{g}^{(k)} = \partial L / \partial \mathbf{W}^{(k)}$ for the
   gradient of the loss $L$ with respect to the weights of layer $k$.
   When the gradients of different layers differ by orders of magnitude, a
   natural remedy is to give each layer a step proportional to its own
   weights :cite:`You.Gitman.Ginsburg.2017`. It replaces the SGD update
   $\mathbf{W}^{(k)} \leftarrow \mathbf{W}^{(k)} - \eta\, \mathbf{g}^{(k)}$ by

    $$\mathbf{W}^{(k)} \leftarrow \mathbf{W}^{(k)} - \eta \, \frac{\|\mathbf{W}^{(k)}\|_\textrm{F}}{\|\mathbf{g}^{(k)}\|_\textrm{F}} \, \mathbf{g}^{(k)}.$$

    The scaled update has norm $\eta\, \|\mathbf{W}^{(k)}\|_\textrm{F}$:
    every layer moves by the same fraction $\eta$ of its weight norm,
    however large or small its gradient. An exploding gradient therefore
    changes only the direction of the update, not its size.
    1. Build a three-layer linear network of width 100 with
       $\mathcal{N}(0, 1)$ weights. Draw one fixed minibatch of 256 inputs
       and 256 targets in $\mathbb{R}^{100}$ from
       $\mathcal{N}(\mathbf{0}, \mathbf{I})$, and let $L$ be the mean
       squared error on this minibatch. Train with plain SGD at
       $\eta = 0.1$, record $L$ and $\|\mathbf{g}^{(1)}\|_\textrm{F}$ over
       the first ten steps, and show that both diverge.
    1. Replace the update by the scaled one, with the same $\eta$, and
       repeat the ten steps. Does the loss decrease? Relate the outcome to
       the scale of the network's outputs at this initialization.
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

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §4.4]{.kicker}

Numerical Stability and Initialization<br>**a product over layers · vanishing and exploding gradients · Xavier and He · symmetry**
:::
:::

::: {.slide title="Ten Layers with N(0, 0.01²) Weights Shrink the Gradient over 10⁸-Fold"}
[The problem]{.kicker}

The Fashion-MNIST MLP has one hidden layer of width 256 and weights from
$\mathcal{N}(0, 0.01^2)$. Give it ten hidden layers, initialized the same
way, and compare the gradient's mean square at the first and the last
hidden layer:

- weights $\mathcal{N}(0, 0.01^2)$: first/last $\approx 10^{-17}$, so the
  gradient itself shrinks more than $10^8$-fold;
- weights $\mathcal{N}(0, 1)$: first/last $\approx 10^{19}$, a gradient
  about $3 \times 10^9$ times larger.

::: {.d2l-note .warn}
The first layers either never move or overshoot. The initial scale decides
whether a deep network trains at all: with $\mathcal{N}(0, 0.01^2)$, ten
epochs leave the loss where it started.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the gradient at layer $k$ as a product of one factor per layer;
- compute the factor $h\tau^2 \mathbb{E}[\phi'(z)^2]$ by which one layer scales the
  gradient, and predict the gradient after ten layers;
- derive Xavier and He initialization from a variance count;
- check the predictions by measurement, and by training the deep network
  with each initialization;
- explain why constant weights fail at any scale.

**Running example:** the Fashion-MNIST MLP of width 256, deepened to ten
hidden layers; minibatches of 256.
:::

::: {.slide title="The Scale of the Initial Weights, Between Backpropagation and Generalization"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Multilayer Perceptrons | hidden layers, universal approximation, activation functions |
| Implementation | an MLP trained on Fashion-MNIST |
| Backpropagation | how the gradients are computed, and at what memory cost |
| **Numerical Stability and Initialization** | **vanishing gradients, Xavier and He initialization** |
| Generalization in Deep Learning | memorization, double descent, implicit bias, early stopping |
| Dropout | random masks, slower memorization of noisy labels |
| Predicting House Prices on Kaggle | the workflow on real data |
:::

::: {.col .narrow}
- **Uses:** the backward rules of the last deck; the variance of a sum of
  independent terms; the slopes of ReLU and sigmoid.
- **Skips:** normalization layers and residual connections (later
  chapters).
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[Vanishing and Exploding Gradients]{.dtitle}

[a product of one factor per layer]{.dsub}
:::
:::

::: {.slide title="The Gradient at Layer k Is a Product of One Factor per Layer Above"}
[A product over layers]{.kicker}

The $K$-layer step of the last deck, for the activations (no weight decay,
so $J = L$): multiply entry by entry by the slopes, then by the transposed
weights.

$$\frac{\partial L}{\partial \mathbf{H}^{(k-1)}} = \Bigl(\frac{\partial L}{\partial \mathbf{H}^{(k)}} \odot \phi'\bigl(\mathbf{Z}^{(k)}\bigr)\Bigr)\, \mathbf{W}^{(k)\top}$$

In code: `G = (G * dphi(Z)) @ W.T`. From the last hidden layer $K - 1$ down
to layer $k$, the step is applied $K - 1 - k$ times.

::: {.d2l-note .rule}
A product of many factors is huge or tiny unless they are near 1:
$0.9^{50} \approx 1/200$, $\;1.1^{50} \approx 117$.
:::
:::

::: {.slide title="Each Layer Multiplies the Gradient's Mean Square by hτ²𝔼[φ′(z)²]"}
[A product over layers]{.kicker}

Each entry one layer down sums $h$ terms (weight × slope × gradient). With
independent weights of mean 0 and variance $\tau^2$, the sum has $h$ times
the mean square of a term, and a term has $\tau^2 \mathbb{E}[\phi'(z)^2]$ times that
of the gradient above.

| network ($h = 256$) | factor per layer | after 9 layers |
|---|---|---|
| ReLU, $\tau = 0.01$ | $256 \cdot 10^{-4} \cdot \tfrac12 = 0.0128$ | $\approx 10^{-17}$ |
| ReLU, $\tau = 1$ | $256 \cdot \tfrac12 = 128$ | $\approx 10^{19}$ |
| sigmoid, $h\tau^2 = 1$ | $\mathbb{E}[\sigma'(z)^2] \le 1/16$ | $< 10^{-10}$ |

ReLU: slope 1 on half of the units, so $\mathbb{E}[\phi'^2] = 1/2$.
:::

::: {.slide title="Measured: Straight Lines with the Predicted Slopes"}
[A product over layers]{.kicker}

Ten layers of width 256, minibatch of 256; the backward pass starts with a
gradient of mean square 1 at layer 10.

@!numerical-stability-and-init-exploding-gradients
:::

::: {.slide title="Sigmoid Gradients Vanish Even with Correctly Scaled Weights"}
[Vanishing gradients]{.kicker}

- $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 1/4$, so $\mathbb{E}[\sigma'(z)^2] \le 1/16$.
- With $h\tau^2 = 1$, each sigmoid layer divides the mean square by at
  least 16; measured: about $5 \times 10^{-12}$ after nine layers.
- Larger weights push pre-activations into the tails, where
  $\sigma' \approx 0$.
- ReLU has slope 1 on every active unit: only the weights set its factor.

::: {.d2l-note}
Small weights and small slopes both make gradients vanish. Only the first
can be cured by rescaling the weights.
:::
:::

::: {.slide title="Exploding Gradients Leave No Learning Rate That Suits All Layers"}
[Exploding gradients]{.kicker}

- First-layer gradient about $3 \times 10^9$ times the last (mean square
  $10^{19}$): a step size that keeps the first layers stable freezes the
  last ones, and the reverse overshoots.
- The forward pass grows by the same factor per ReLU layer.
- Above $3.4 \times 10^{38}$, the largest float32 number, values become
  `inf` and the loss becomes `nan`.

::: {.d2l-note .warn}
A loss that is `nan` from the first step often means an initialization that
is too large.
:::
:::

::: {.slide title="Check: Which τ Keeps a ReLU Layer of Width 256 Flat?"}
[Check]{.kicker}

Choose the weight standard deviation $\tau$ for which a ReLU layer of width
256 leaves the gradient's mean square unchanged.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $h\tau^2 \cdot \tfrac12 = 1$, so $\tau^2 = 2/256$ and
$\tau \approx 0.088$, nearly nine times the 0.01 of the one-layer MLP.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Variance-Preserving Initialization]{.dtitle}

[Xavier · He · the measurement repeated]{.dsub}
:::
:::

::: {.slide title="A Sum of n_in Independent Terms Has n_in Times Their Variance"}
[Xavier initialization]{.kicker}

Output $o_j = \sum_{i=1}^{n_\textrm{in}} x_i w_{ij}$; weights independent,
mean 0, variance $\tau^2$; inputs mean 0, variance $\gamma^2$.

$$\mathbb{E}[o_j] = 0, \qquad
\textrm{Var}[o_j] = \sum_{i} \mathbb{E}[x_i^2]\,\mathbb{E}[w_{ij}^2] = n_\textrm{in}\,\tau^2\gamma^2.$$

Cross terms $\mathbb{E}[x_i x_{i'} w_{ij} w_{i'j}]$, $i \neq i'$, vanish: independent
weights with mean 0.

::: {.d2l-note .rule}
The layer scales the variance by $n_\textrm{in}\tau^2$; keep it at 1.
:::
:::

::: {.slide title="Forward Needs n_in τ² = 1, Backward Needs n_out τ² = 1: Xavier Averages"}
[Xavier initialization]{.kicker}

The backward pass multiplies by $\mathbf{W}^\top$ and sums over the
$n_\textrm{out}$ outputs: factor $n_\textrm{out}\tau^2$.

$$\tau^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}
\qquad\text{or}\qquad
U\Bigl(-\sqrt{\tfrac{6}{n_\textrm{in} + n_\textrm{out}}},\ \sqrt{\tfrac{6}{n_\textrm{in} + n_\textrm{out}}}\Bigr).$$

First layer of the MLP, $784 \to 256$: $\tau = \sqrt{2/1040} \approx 0.044$,
four times the 0.01 it used. Derived for tanh, slope 1 near zero.
:::

::: {.slide title="The ReLU Halves the Mean Square, So He Doubles the Variance"}
[He initialization]{.kicker}

Symmetric pre-activations: $\textrm{ReLU}(z)^2$ is $z^2$ for half of them,
0 for the rest, so $\mathbb{E}[\textrm{ReLU}(z)^2] = \tfrac12 \mathbb{E}[z^2]$.

$$v^{(k)} = \tfrac12\, n_\textrm{in}\, \tau^2\, v^{(k-1)}
\quad\Longrightarrow\quad
\tau^2 = \frac{2}{n_\textrm{in}},$$

with $v^{(k)}$ the mean square of layer $k$. The backward pass gives the
same condition with $n_\textrm{out}$. First layer of the MLP:
$\tau \approx 0.05$.

::: {.d2l-note}
Rule of thumb: Xavier for tanh, He for ReLU.
:::
:::

::: {.slide title="Measured: Xavier Halves per Layer, He Stays Near 1"}
[He initialization]{.kicker}

Ten ReLU layers of width 256; mean square of the activations per layer:

@!numerical-stability-and-init-depth-sweep
:::

::: {.slide title="Trained: N(0, 0.01²) Never Leaves log 10; He Trains at Once"}
[Initialization · training]{.kicker}

Ten hidden ReLU layers of width 256 on Fashion-MNIST, ten epochs, SGD at
learning rate 0.05, minibatches of 256; only the standard deviation of the
initial weights differs:

@!numerical-stability-and-init-training-a-deep-network-2

::: {.d2l-note}
The measurements predicted it. $\mathcal{N}(0, 0.01^2)$: loss 2.30 and 10%
accuracy throughout. Xavier starts more slowly than He; both reach about
85%, no better than one hidden layer.
:::
:::

::: {.slide title="The Default Initialization Is Not He" only="mxnet"}
[Default initialization]{.kicker}

- Gluon's `initialize` draws weights from $U(\pm 0.07)$, whatever the
  width, and sets the biases to 0: weight variance $0.07^2/3$.
- Per ReLU layer of width 256, the mean square of the activations is
  multiplied by $\tfrac{1}{2} \cdot 256 \cdot 0.07^2/3 \approx 0.21$.

::: {.d2l-note .rule}
A deep plain ReLU stack shrinks under this default. Set the initialization
explicitly; this book does.
:::
:::

::: {.slide title="The Default Initialization Is Not He" only="pytorch"}
[Default initialization]{.kicker}

- `nn.Linear` draws weights and biases from
  $U(\pm 1/\sqrt{n_\textrm{in}})$: weight variance $1/(3 n_\textrm{in})$.
- Per ReLU layer of width 256, the weights multiply the mean square of the
  activations by $\tfrac{1}{2} \cdot 256 \cdot \tfrac{1}{3 \cdot 256} = 1/6$
  (biases aside).

::: {.d2l-note .rule}
A deep plain ReLU stack shrinks under this default. Set the initialization
explicitly; this book does.
:::
:::

::: {.slide title="The Default Initialization Is Not He" only="tensorflow"}
[Default initialization]{.kicker}

- A Keras `Dense` layer draws its weights from the uniform Xavier
  distribution, variance $2/(n_\textrm{in} + n_\textrm{out})$, and sets
  its biases to 0.
- Per ReLU layer of width 256, the mean square of the activations is
  multiplied by $\tfrac{1}{2} \cdot 256 \cdot \tfrac{2}{512} = 1/2$.

::: {.d2l-note .rule}
A deep plain ReLU stack shrinks under this default. Set the initialization
explicitly; this book does.
:::
:::

::: {.slide title="The Default Initialization Is Not He" only="jax"}
[Default initialization]{.kicker}

- `nnx.Linear` draws its weights from LeCun normal, a truncated normal with
  variance $1/n_\textrm{in}$, and sets its biases to 0.
- Per ReLU layer of width 256, the mean square of the activations is
  multiplied by $\tfrac{1}{2} \cdot 256 \cdot \tfrac{1}{256} = 1/2$.

::: {.d2l-note .rule}
A deep plain ReLU stack shrinks under this default. Set the initialization
explicitly; this book does.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Symmetry among Hidden Units]{.dtitle}

[why the weights must be random]{.dsub}
:::
:::

::: {.slide title="Constant Weights Give Every Hidden Unit the Same Gradient"}
[Symmetry]{.kicker}

Every weight equals $c = \sqrt{2/d}$, which has He's mean square. Then every
hidden unit computes the same function, gets the same gradient, takes the
same step, and stays identical: $h$ units act as one.

$d = 20$, $h = 16$, squared loss; largest difference between the gradients
of two hidden units:

@!numerical-stability-and-init-symmetry-among-hidden-units
:::

::: {.slide title="Random Draws Break the Symmetry; the Variance Count Sets Their Scale"}
[Symmetry]{.kicker}

- Hidden units are interchangeable: permuting them with their weights
  leaves the function unchanged.
- Gradient descent preserves an exact tie; only the initialization can
  break it.
- Random weights at the Xavier or He scale do; biases may start at 0.
:::

::: {.slide title="Check: Do Equal Biases Break the Tie?"}
[Check]{.kicker}

All hidden weights equal $c$. Does setting all hidden biases to the same
value $b \ne 0$ break the symmetry? What about distinct random biases?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Equal biases do not: every unit still computes the same
function. Distinct biases do, since units then have different
pre-activations and receive different gradients. The standard choice is
random weights and zero biases.
:::
:::
:::

::: {.slide title="Initialization Sets the Scale Only at the Start"}
[Limits]{.kicker}

- Xavier and He assume independent weights and inputs: true at
  initialization.
- Training makes the weights depend on the data, and nothing then keeps
  each layer's factor near 1: at learning rate 0.1 the ten-layer network
  now and then diverges, so the run above uses 0.05.
- Normalization layers and residual connections, in later chapters, keep
  the scale under control during training as well.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Product:** the gradient at layer $k$ is the top gradient, multiplied
  by the slopes $\phi'(\mathbf{Z})$ and the transposed weights of every
  layer above it.
- **Factor:** $h\tau^2 \mathbb{E}[\phi'(z)^2]$ per layer on the mean square; ten
  ReLU layers of width 256 multiply it by $10^{-17}$ with $\tau = 0.01$ and
  by $10^{19}$ with $\tau = 1$.
- **Sigmoid:** $\mathbb{E}[\sigma'^2] \le 1/16$ shrinks the gradient even when
  $h\tau^2 = 1$.
- **Xavier:** $\tau^2 = 2/(n_\textrm{in} + n_\textrm{out})$; **He:**
  $\tau^2 = 2/n_\textrm{in}$, flat through ten ReLU layers.
- **Trained:** ten hidden layers learn nothing from
  $\mathcal{N}(0, 0.01^2)$ and train at once with He.
- **Symmetry:** constant weights keep all units identical at any scale;
  draw them at random.

**Next:** a network that trains is not yet one that generalizes.
:::
