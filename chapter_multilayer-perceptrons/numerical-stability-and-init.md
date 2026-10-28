```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Numerical Stability and Initialization
:label:`sec_numerical_stability`

The MLP of :numref:`sec_mlp-implementation` has a single hidden layer and
draws its initial weights from a normal distribution with standard deviation
0.01. A network with ten such hidden layers of width 256, initialized the
same way, does not learn. The backward pass of :numref:`sec_backprop`
multiplies the gradient by one transposed weight matrix per layer on its way
down to the first layer, and with weights this small each multiplication
shrinks the magnitude of the gradient about ninefold. By the time it reaches
the first layer, the gradient is smaller than at the top by a factor of more
than $10^8$, and the first layers barely change. With weights of standard
deviation 1, each multiplication enlarges the gradient instead, and the
updates overshoot.

Whether a deep network can be trained at all thus depends on the scale of
its initial weights, and the right scale depends on the width of each layer
and on the activation function. This section measures how the gradient
shrinks or grows from layer to layer, derives the scales that keep it
constant, known as Xavier and He initialization, and checks them with the
same measurement. It then explains a second requirement, stated without
proof in :numref:`sec_mlp-implementation`, that no choice of scale can meet:
the initial weights of different hidden units must differ, or the units
remain identical throughout training. Both remedies concern the
start of training; architectures that keep the scale under control during
training follow in later chapters.

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np as mnp, npx
npx.set_np()
```

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input #numerical-stability-and-init-numerical-stability-and-initialization}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import jax
from jax import numpy as jnp
```

## Vanishing and Exploding Gradients

### Products of Jacobians

Consider an MLP with $K$ layers, $K - 1$ of them hidden, of width $h$, in the notation of
:numref:`sec_mlp`. Starting from $\mathbf{H}^{(0)} = \mathbf{X}$, hidden
layer $k$ computes the pre-activation
$\mathbf{Z}^{(k)} = \mathbf{H}^{(k-1)}\mathbf{W}^{(k)} + \mathbf{b}^{(k)}$
and the activation $\mathbf{H}^{(k)} = \phi(\mathbf{Z}^{(k)})$. We follow the
gradient of the loss $L$ with respect to the activations of each layer. By
the rules of :numref:`sec_backprop`, one step of the backward pass
multiplies the gradient entry by entry by the slopes of the activation
function and then by the transposed weights.

For a single example, layer $k$ computes
$\mathbf{h}^{(k)} = \phi(\mathbf{z}^{(k)})$ with
$\mathbf{z}^{(k)} = \mathbf{W}^{(k)\top}\mathbf{h}^{(k-1)} + \mathbf{b}^{(k)}$,
and the step reads

$$
\nabla_{\mathbf{h}^{(k-1)}} L = \mathbf{W}^{(k)}\bigl(\phi'(\mathbf{z}^{(k)}) \odot \nabla_{\mathbf{h}^{(k)}} L\bigr) = \mathbf{M}^{(k)}\, \nabla_{\mathbf{h}^{(k)}} L,
\qquad
\mathbf{M}^{(k)} = \mathbf{W}^{(k)} \operatorname{diag}\bigl(\phi'(\mathbf{z}^{(k)})\bigr),
$$
:eqlabel:`eq_grad-step`

where $\operatorname{diag}(\cdot)$ places the entries of a vector on the
diagonal of an otherwise zero matrix. The transpose of $\mathbf{M}^{(k)}$
is the Jacobian of layer $k$, the matrix of derivatives of its outputs
$\mathbf{h}^{(k)}$ with respect to its inputs $\mathbf{h}^{(k-1)}$. The step
is thus the chain rule of :numref:`sec_calculus`: the gradient with respect
to the inputs of a function is its transposed Jacobian times the gradient
with respect to its outputs. Repeating :eqref:`eq_grad-step` from the last
hidden layer, $K - 1$, down to layer $k$ gives

$$
\nabla_{\mathbf{h}^{(k)}} L = \mathbf{M}^{(k+1)}\, \mathbf{M}^{(k+2)} \cdots \mathbf{M}^{(K-1)}\, \nabla_{\mathbf{h}^{(K-1)}} L.
$$
:eqlabel:`eq_grad-product`

The gradient that reaches layer $k$ is the gradient at the last hidden layer
multiplied by one transposed Jacobian for each of the $K - 1 - k$ layers
above it. The gradient with respect to the weights of layer $k$, the outer
product
$\mathbf{h}^{(k-1)} \bigl(\phi'(\mathbf{z}^{(k)}) \odot \nabla_{\mathbf{h}^{(k)}} L\bigr)^\top$,
contains this product as a factor.

A product of many factors is very large or very small unless the factors are
close to 1. If each factor scales the vectors it multiplies by about the
same amount $\rho$, a product of $t$ of them scales them by about $\rho^t$.
As for the repeated matrix products of :numref:`sec_linear-algebra`, with
$\rho = 0.9$ the gradient shrinks 200-fold over fifty layers, and with
$\rho = 1.1$ it grows 117-fold. The factor of one layer
follows from the scale of its weights. Each entry of the gradient one layer
down is a sum of $h$ terms, one per unit of the layer above, and each term is
the product of a weight, a slope $\phi'(z)$, and an entry of the gradient
above. If the weights are independent with mean zero and variance $\tau^2$,
the cross terms of the square of this sum vanish, so the sum has $h$ times
the mean square of one term, and one term has $\tau^2 E\bigl[\phi'(z)^2\bigr]$
times the mean square of the gradient above (:numref:`subsec_xavier`
derives this). Per layer, the mean square of the gradient is therefore
multiplied by about

$$h\, \tau^2\, E\bigl[\phi'(z)^2\bigr].$$
:eqlabel:`eq_grad-factor`

Two things set the factor: the scale of the weights, through $h\tau^2$, and
the slope of the activation function, through $E[\phi'(z)^2]$. When the
pre-activations are symmetric about zero, the ReLU has slope 1 for half of
them and 0 for the other half, so $E[\phi'(z)^2] = 1/2$. For a ReLU layer of
width $h = 256$ with the weights of :numref:`sec_mlp-implementation`,
$\tau = 0.01$, the factor is $256 \times 10^{-4} / 2 = 0.0128$; with
$\tau = 1$ it is $128$.

We measure these factors in a network of ten hidden layers of width 256,
without biases, at initialization. The function `moments` runs the forward
pass on a minibatch of 256 random inputs, keeping the pre-activations, and
then the backward pass. As the loss we take
$L = \sum_{i, j} v_{ij} [\mathbf{H}^{(K-1)}]_{ij}$, a weighted sum of the
activations of the last hidden layer with random weights $v_{ij}$ of mean
square 1. The backward pass then starts from
$\partial L / \partial \mathbf{H}^{(K-1)} = \mathbf{V}$, a gradient of mean
square 1 whatever the scale of $\mathbf{H}^{(K-1)}$, and each of its steps
applies :eqref:`eq_grad-step` to all examples at once. The function returns the mean squares of the activations and
of the gradients at every layer. Each activation function is passed together
with its derivative.

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

We compare three networks: ReLU with the weights of
:numref:`sec_mlp-implementation`, $\tau = 0.01$; ReLU with $\tau = 1$; and
the sigmoid with $\tau^2 = 1/256$, so that $h\tau^2 = 1$ and only the slope
of the activation function changes the scale. The plot shows the mean square
of the gradient at each layer; the top layer starts at 1.

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

On a logarithmic axis the curves are straight lines: every layer multiplies
the mean square by the same factor. Over the nine steps from the top layer
down to the first, the factors of :eqref:`eq_grad-factor` predict
$0.0128^9 \approx 10^{-17}$ for the small weights and
$128^9 \approx 10^{19}$ for the large ones, and the measurement agrees. The
sigmoid loses more than an order of magnitude per layer even with
$h\tau^2 = 1$, and the mean square of its gradient at the first layer is
about $5 \times 10^{-12}$ of that at the top.

### Vanishing Gradients

The two falling curves show vanishing gradients, which
:numref:`subsec_activation-functions` met for the sigmoid: the weights of the
first layers receive gradients many orders of magnitude smaller than those
of the last layers. Any learning rate small enough to keep the last layers stable
then leaves the first layers almost where they started. Small weights cause
this, and so do activation functions with small slopes. The derivative of
the sigmoid, $\sigma'(z) = \sigma(z)\bigl(1 - \sigma(z)\bigr)$, never
exceeds $1/4$, its value at $z = 0$,
so $E[\sigma'(z)^2] \le 1/16$ and each sigmoid layer with $h\tau^2 = 1$
divides the mean square of the gradient by at least 16. Larger weights do
not simply compensate, because they push more pre-activations into the tails
of the sigmoid, where its slope is close to zero. Vanishing gradients made
deep networks with sigmoid and tanh units hard to train
:cite:`bengio1994learning`. The ReLU has slope 1 for every active unit, so
only the scale of the weights sets its factor, which is one reason it
replaced the sigmoid in hidden layers.

### Exploding Gradients

The rising curve shows an *exploding gradient*. With a mean square $10^{19}$
times larger at the first layer than at the top, the gradient there is about
$3 \times 10^9$ times larger in magnitude, and no single learning rate
suits all layers: one small enough for the first layers leaves the last ones
frozen, and one suited to the last layers makes the updates of the first
ones overshoot so far that the loss grows instead of falling. The forward
pass grows along with the gradient, since a ReLU layer also multiplies the
mean square of the activations by $h\tau^2/2$, as :numref:`subsec_he_init`
shows. Once values exceed about $3.4 \times 10^{38}$, the largest number
that 32-bit floating point can represent, they become infinite, and the loss
becomes `nan`. A loss that is `nan` from the first step is often the sign of
an initialization that is too large.

## Parameter Initialization

### Default Initialization

A framework initializes the weights of a layer when we do not. Its default
is not designed for a deep stack of ReLU layers, so this book sets the
initialization of its models explicitly. The variance argument that follows
says which scale to choose. Below, $n_\textrm{in}$ is the number of inputs
of a layer.

:begin_tab:`mxnet`
Gluon's `initialize` draws the weights uniformly from $[-0.07, 0.07]$,
whatever the size of the layer, and sets the biases to 0.
:end_tab:

:begin_tab:`pytorch`
`nn.Linear` draws its weights and its biases uniformly from the interval
$[-1/\sqrt{n_\textrm{in}}, 1/\sqrt{n_\textrm{in}}]$, so the weights have
variance $1/(3 n_\textrm{in})$.
:end_tab:

:begin_tab:`tensorflow`
A Keras `Dense` layer draws its weights from the uniform version of the
Xavier initialization derived below and sets its biases to 0.
:end_tab:

:begin_tab:`jax`
`nnx.Linear` draws its weights from a truncated normal distribution with
variance $1/n_\textrm{in}$, called LeCun normal, and sets its biases to 0.
:end_tab:

### Xavier Initialization
:label:`subsec_xavier`

We start with a layer without activation function. It has $n_\textrm{in}$
inputs $x_i$ and $n_\textrm{out}$ outputs, and output $j$ is

$$o_j = \sum_{i=1}^{n_\textrm{in}} x_i w_{ij}.$$

The weights $w_{ij}$ are drawn independently from the same distribution,
with mean zero and variance $\tau^2$; the distribution need not be Gaussian.
The inputs $x_i$ have mean zero and variance $\gamma^2$, and they are
independent of the weights and of each other. Then $o_j$ has mean
$E[o_j] = \sum_i E[x_i]\,E[w_{ij}] = 0$ and variance

$$
\begin{aligned}
\textrm{Var}[o_j] = E[o_j^2]
&= \sum_{i=1}^{n_\textrm{in}} \sum_{i'=1}^{n_\textrm{in}} E[x_i x_{i'} w_{ij} w_{i'j}] \\
&= \sum_{i=1}^{n_\textrm{in}} E[x_i^2]\, E[w_{ij}^2]
= n_\textrm{in}\, \tau^2 \gamma^2.
\end{aligned}
$$

In the second line, the cross terms with $i \neq i'$ vanish because the
weights $w_{ij}$ and $w_{i'j}$ are independent with mean zero, and the
remaining terms use $E[w_{ij}^2] = \tau^2$ and $E[x_i^2] = \gamma^2$, which
hold because both have mean zero. A sum of $n_\textrm{in}$ independent
terms has $n_\textrm{in}$ times their variance, so the layer multiplies the
variance of the signal by $n_\textrm{in}\tau^2$. To keep it unchanged, we
need $n_\textrm{in}\tau^2 = 1$.

The backward pass multiplies the gradient by $\mathbf{W}^\top$, which sums
over the $n_\textrm{out}$ outputs of the layer instead. The same computation
shows that the variance of the gradient is multiplied by
$n_\textrm{out}\tau^2$, so keeping it unchanged needs
$n_\textrm{out}\tau^2 = 1$. Both conditions hold only if
$n_\textrm{in} = n_\textrm{out}$. *Xavier initialization*
:cite:`Glorot.Bengio.2010` satisfies their average,

$$
\frac{1}{2}(n_\textrm{in} + n_\textrm{out})\, \tau^2 = 1,
\qquad \textrm{that is,} \qquad
\tau^2 = \frac{2}{n_\textrm{in} + n_\textrm{out}}.
$$

For the first layer of the MLP of :numref:`sec_mlp-implementation`, with
$n_\textrm{in} = 784$ and $n_\textrm{out} = 256$, this gives
$\tau = \sqrt{2/1040} \approx 0.044$, four times the 0.01 used there. The
weights may also be drawn uniformly. The uniform distribution on $[-a, a]$
has variance $a^2/3$, so Xavier initialization draws them from

$$U\left(-\sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}},\ \sqrt{\frac{6}{n_\textrm{in} + n_\textrm{out}}}\right).$$

The derivation assumed a layer without activation function. It also holds
approximately for an activation function with slope 1 near zero, such as
tanh, as long as the pre-activations stay small, and Glorot and Bengio
derived it for tanh networks.

### He Initialization
:label:`subsec_he_init`

A ReLU is far from linear: it sets about half of its inputs to zero. To
extend the argument, we note what it used. With independent weights of mean
zero, the output $o_j$ has mean zero and $E[o_j^2] = n_\textrm{in}\tau^2
E[x_i^2]$ for *any* inputs, whatever their mean: the inputs enter only
through their mean square $E[x_i^2]$. So we track the mean square of the
activations through the ReLU. If the pre-activations $z$ are symmetric about
zero, as they are when the weights are drawn from a symmetric distribution
independently of the inputs, then $\textrm{ReLU}(z)^2$ equals $z^2$ for the
positive half of them and $0$ for the negative half, so the ReLU halves the
mean square: $E[\textrm{ReLU}(z)^2] = \tfrac{1}{2}E[z^2]$.

Let $m^{(k)}$ be the mean square of the entries of $\mathbf{H}^{(k)}$, the activations of
layer $k$. Its pre-activations have mean square
$n_\textrm{in}\tau^2 m^{(k-1)}$, and the ReLU halves it:

$$m^{(k)} = \frac{1}{2}\, n_\textrm{in}\, \tau^2\, m^{(k-1)}.$$

The mean square stays constant from layer to layer if we double the weight
variance of the linear case:

$$\tau^2 = \frac{2}{n_\textrm{in}}.$$

This is *He initialization* :cite:`He.Zhang.Ren.ea.2015`, also called
Kaiming initialization, the standard choice for networks with ReLU
activations. For the backward pass, :eqref:`eq_grad-factor` with
$E[\phi'(z)^2] = 1/2$ gives the same condition with $n_\textrm{out}$ in
place of $n_\textrm{in}$, and for layers of equal width the two coincide.
The factor 2 compensates for the ReLU between two layers. A network has one
weight layer more than it has ReLU layers, so one layer receives the factor
without a ReLU to undo it. In the count above it is the output layer, whose
logits have twice the mean square of the activations below them. In the
count of pre-activations of :citet:`He.Zhang.Ren.ea.2015` it is the first
layer. Either way the scale is off by a single factor of 2, which does not
compound with depth. For the first layer of the MLP of
:numref:`sec_mlp-implementation`, He initialization gives
$\tau = \sqrt{2/784} \approx 0.05$. Xavier and He differ by the factor 2
and by which fan size they use. The rule of thumb is Xavier for tanh and He for
ReLU, and most libraries provide both as named initializers.

### Activations and Gradients across Depth
:label:`subsec_variance_propagation`

We now repeat the measurement with ReLU and three initializations: the
$\mathcal{N}(0, 0.01^2)$ of :numref:`sec_mlp-implementation`, Xavier, and
He. With $n_\textrm{in} = n_\textrm{out} = 256$, Xavier gives
$\tau^2 = 1/256$ and He gives $\tau^2 = 2/256$. This time we plot the mean
square of the activations at each layer and print, for each initialization,
its value at the last layer together with the mean square of the gradient at
the first layer.

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

He initialization keeps the mean squares of the activations and of the
gradients within a factor of about 3 of 1 through all ten layers. The curve
is not exactly flat, because the factor of each layer fluctuates around 1
with the random weights and the fluctuations compound. Xavier initialization
loses the factor $1/2$ of the ReLU at every layer, in both directions: the
activations at the last layer are about $2^{-10} \approx 10^{-3}$, and the
gradient at the first layer about $2^{-9} \approx 2 \times 10^{-3}$.
With the weights of :numref:`sec_mlp-implementation`, the activations at the
last layer are about $10^{-19}$, and the network is useless before training
begins.

The derivations assumed that the weights are independent of each other and
of the inputs. This holds at initialization, but training changes the
weights and breaks the assumption, and nothing then keeps the factor of each
layer near 1. Normalization layers (:numref:`sec_batch_norm`) and residual
connections, introduced with deeper architectures later in the book, keep
the scale under control during training as well.

## Symmetry among Hidden Units

The variance argument fixes the scale of the initial weights. It does not
say why they should be random, and a constant would have the same mean
square. Consider an MLP with one hidden layer and a single output, and
suppose that every weight of the hidden layer equals the same constant $c$
and every weight of the output layer the same constant $c'$. Every hidden
unit then computes the same function of the input. In the backward pass,
every hidden unit receives the same gradient from the output, and the
weights into each unit receive the same gradient. After an update of
minibatch stochastic gradient descent, the units are still identical, and by
the same argument after every later update. The $h$ hidden units behave like
a single unit, however many of them the network has. The hidden units of an
MLP are interchangeable: permuting them, together with their incoming and
outgoing weights, does not change the function that the network computes.
Gradient descent cannot break this symmetry if the initialization does not.

We check this with $d = 20$ inputs, $h = 16$ hidden ReLU units, a minibatch
of 64 random examples with random targets, and the squared loss. The constant
$c = \sqrt{2/d}$ gives the pre-activations the same mean square as He
initialization. We compute the gradient with respect to the weights of the
hidden layer and print the largest difference between the gradients of two
hidden units, that is, between two columns of the gradient, for the constant
weights and for random weights at the same scale.

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

With constant weights, all 16 columns of the gradient are identical, so all
hidden units take the same step and stay identical. With random weights,
the units receive different gradients from the first step on. Random
initialization breaks the symmetry, and the scale of the random draws comes
from the variance argument; the biases can start at zero, since the weights
already differ between units. Dropout, introduced later in this chapter,
also breaks the symmetry, because it silences a different random subset of
units at every step.

We can now choose the initial weights of a deep MLP so that activations and
gradients keep their size through every layer, and we know why they must be
random. A network that can be trained is not yet a network that predicts
well on new data. The next section asks why deep networks, with many more
parameters than training examples, generalize at all.

## Summary

The backward pass multiplies the gradient by one Jacobian per layer, so the
gradient that reaches the first layers of a deep network is a long product.
Each ReLU layer of width $h$ with weights of variance $\tau^2$ multiplies its
mean square by about $h\tau^2/2$. With the $\mathcal{N}(0, 0.01^2)$ weights
of a one-hidden-layer MLP, ten layers of width 256 shrink the mean square
of the gradient by a factor of about $10^{17}$, and with
$\mathcal{N}(0, 1)$ weights they enlarge it by about $10^{19}$. Saturating
activation functions such as the sigmoid shrink it even when the weights
are scaled correctly. Xavier initialization,
$\tau^2 = 2/(n_\textrm{in} + n_\textrm{out})$, keeps the variance constant
through linear layers, and He initialization, $\tau^2 = 2/n_\textrm{in}$,
compensates for the ReLU, which halves the mean square. The initial weights
must also be random, because hidden units that start identical receive
identical gradients and stay identical.

## Exercises

1. **Spectral norm bounds.** The gradient in :eqref:`eq_grad-product` is a
   product of $K - 1 - k$ transposed Jacobians. The spectral norm (the largest singular
   value) is submultiplicative:
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
       total factor $n_K / n_0$, whatever the depth.
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
   Layerwise adaptive rate scaling :cite:`You.Gitman.Ginsburg.2017` replaces
   the SGD update
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
   tracks the mean square $E[\textrm{ReLU}(z)^2]$ rather than the variance,
   because the next layer's pre-activation $o = \sum_i x_i w_i$, with inputs $x_i = \textrm{ReLU}(z_i)$ and
   independent zero-mean weights has
   $\textrm{Var}[o] = n_\textrm{in}\tau^2 E[x_i^2]$ whatever the mean of
   $x_i$.
    1. For $z \sim \mathcal{N}(0, \nu^2)$, compute $E[\textrm{ReLU}(z)]$ and
       $\textrm{Var}[\textrm{ReLU}(z)]$, and show that the variance equals
       $\left(\tfrac{1}{2} - \tfrac{1}{2\pi}\right)\nu^2$ rather than
       $\tfrac{1}{2}\nu^2$.
    1. A tempting shortcut substitutes $\textrm{Var}[x_i]$ for $E[x_i^2]$ in
       the Xavier computation. Show that this shortcut prescribes
       $\tau^2 = 2 / \bigl(n_\textrm{in}(1 - 1/\pi)\bigr)$ instead of
       $2/n_\textrm{in}$. Compute the factor by which each ReLU layer
       initialized this way actually multiplies the mean square, and the
       total factor over fifty layers.
1. [code] **Tanh networks.** ● Repeat the measurement of
   :numref:`subsec_variance_propagation` with $\tanh$ in place of ReLU, with
   Xavier initialization and fifty layers. A layer then maps the mean square
   $m$ of its input to $m' = E[\tanh(z)^2]$, where $z$ is approximately
   $\mathcal{N}(0, m)$.
    1. Use $\tanh(z)^2 \approx z^2 - \tfrac{2}{3}z^4$ for small $z$ to show
       that $m' \approx m - 2m^2$. Over many layers, does the mean square
       stay constant, decay geometrically, or decay more slowly? Show that
       $m^{(k)} \approx 1/(2k)$ for large $k$.
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

Numerical Stability and Initialization<br>**products of Jacobians · vanishing and exploding gradients · Xavier and He · symmetry**
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
whether a deep network trains at all.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write the gradient at layer $k$ as a product of one factor per layer;
- compute the factor $h\tau^2 E[\phi'(z)^2]$ by which one layer scales the
  gradient, and predict the gradient after ten layers;
- derive Xavier and He initialization from a variance count;
- check the predictions by measurement;
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
| Multilayer Perceptrons | hidden layers, activation functions, universal approximation |
| Implementation | an MLP trained on Fashion-MNIST |
| Backpropagation | how the gradients are computed, and at what memory cost |
| **Numerical Stability and Initialization** | **vanishing gradients, Xavier and He initialization** |
| Generalization in Deep Learning | memorization, double descent, implicit bias, early stopping |
| Dropout | random masks, robustness to missing units |
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
[Products of Jacobians]{.kicker}

One backward step per layer: multiply by the slopes $\phi'(\mathbf{z})$,
then by the weights. For one example, from the last hidden layer $K - 1$
down:

$$\nabla_{\mathbf{h}^{(k)}} L = \mathbf{M}^{(k+1)} \cdots \mathbf{M}^{(K-1)}\, \nabla_{\mathbf{h}^{(K-1)}} L,
\qquad \mathbf{M}^{(k)} = \mathbf{W}^{(k)} \operatorname{diag}\bigl(\phi'(\mathbf{z}^{(k)})\bigr).$$

$\mathbf{M}^{(k)}$ is the transposed Jacobian of layer $k$.

::: {.d2l-note .rule}
A product of many factors is huge or tiny unless they are near 1:
$0.9^{50} \approx 1/200$, $\;1.1^{50} \approx 117$.
:::
:::

::: {.slide title="Each Layer Multiplies the Gradient's Mean Square by hτ²E[φ′(z)²]"}
[Products of Jacobians]{.kicker}

Each entry one layer down sums $h$ terms (weight × slope × gradient). With
independent weights of mean 0 and variance $\tau^2$, the sum has $h$ times
the mean square of a term, and a term has $\tau^2 E[\phi'(z)^2]$ times that
of the gradient above.

| network ($h = 256$) | factor per layer | after 9 layers |
|---|---|---|
| ReLU, $\tau = 0.01$ | $256 \cdot 10^{-4} \cdot \tfrac12 = 0.0128$ | $\approx 10^{-17}$ |
| ReLU, $\tau = 1$ | $256 \cdot \tfrac12 = 128$ | $\approx 10^{19}$ |
| sigmoid, $h\tau^2 = 1$ | $E[\sigma'(z)^2] \le 1/16$ | $< 10^{-10}$ |

ReLU: slope 1 on half of the units, so $E[\phi'^2] = 1/2$.
:::

::: {.slide title="Measured: Straight Lines with the Predicted Slopes"}
[Products of Jacobians]{.kicker}

Ten layers of width 256, minibatch of 256; the backward pass starts with a
gradient of mean square 1 at layer 10.

@!numerical-stability-and-init-exploding-gradients
:::

::: {.slide title="Sigmoid Gradients Vanish Even with Correctly Scaled Weights"}
[Vanishing gradients]{.kicker}

- $\sigma'(z) = \sigma(z)(1 - \sigma(z)) \le 1/4$, so $E[\sigma'(z)^2] \le 1/16$.
- With $h\tau^2 = 1$, each sigmoid layer divides the mean square by at
  least 16; measured: about $5 \times 10^{-12}$ after nine layers.
- Larger weights push pre-activations into the tails, where
  $\sigma' \approx 0$.
- ReLU has slope 1 on every active unit: only the weights set its factor.

::: {.d2l-note}
Vanishing gradients long made deep sigmoid and tanh networks hard to train.
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

$$E[o_j] = 0, \qquad
\textrm{Var}[o_j] = \sum_{i} E[x_i^2]\,E[w_{ij}^2] = n_\textrm{in}\,\tau^2\gamma^2.$$

Cross terms $E[x_i x_{i'} w_{ij} w_{i'j}]$, $i \neq i'$, vanish: independent
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
0 for the rest, so $E[\textrm{ReLU}(z)^2] = \tfrac12 E[z^2]$.

$$m^{(k)} = \tfrac12\, n_\textrm{in}\, \tau^2\, m^{(k-1)}
\quad\Longrightarrow\quad
\tau^2 = \frac{2}{n_\textrm{in}},$$

with $m^{(k)}$ the mean square of layer $k$. The backward pass gives the
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
- Dropout, later in this chapter, also breaks ties: it silences a different
  random subset of units at every step.
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
  each layer's factor near 1.
- Normalization layers and residual connections, in later chapters, keep
  the scale under control during training as well.
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Product:** the gradient at layer $k$ is the top gradient times one
  factor $\mathbf{W}\operatorname{diag}(\phi'(\mathbf{z}))$ per layer
  above it.
- **Factor:** $h\tau^2 E[\phi'(z)^2]$ per layer on the mean square; ten
  ReLU layers of width 256 multiply it by $10^{-17}$ with $\tau = 0.01$ and
  by $10^{19}$ with $\tau = 1$.
- **Sigmoid:** $E[\sigma'^2] \le 1/16$ shrinks the gradient even when
  $h\tau^2 = 1$.
- **Xavier:** $\tau^2 = 2/(n_\textrm{in} + n_\textrm{out})$; **He:**
  $\tau^2 = 2/n_\textrm{in}$, flat through ten ReLU layers.
- **Symmetry:** constant weights keep all units identical at any scale;
  draw them at random.

**Next:** a network that trains is not yet one that generalizes.
:::
