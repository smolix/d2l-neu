```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Linear Regression Implementation
:label:`sec_linear_scratch`

Now that we have the theory of linear regression (:numref:`sec_linear_regression`) and a synthetic dataset with known ground-truth parameters (:numref:`sec_synthetic-regression-data`), we are ready to put both into code. We will implement the entire training pipeline twice in this section:

1. **From scratch:** First, we build every component—parameter initialization, the linear forward pass, the squared loss, automatic differentiation, the minibatch iterator, and the SGD update loop—relying only on basic tensor arithmetic and `autograd`. Writing each step by hand strips away all magic so you can see exactly how gradients flow and how hyperparameters like the learning rate shape convergence.
2. **Concisely with framework APIs (:numref:`sec_linear_concise`):** Once every moving part is transparent, we re-express the exact same model in a handful of lines using high-level deep learning abstractions—fully connected layers, built-in loss functions, optimizers, and the reusable `Module`/`Trainer` harness whose internal design we unpack in :numref:`sec_oo-design`.

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=2}
%%tab mxnet
import math
import numpy as np
from mxnet import autograd, gluon, init, np as mnp, npx
from mxnet.gluon import nn
from d2l import mxnet as d2l
npx.set_np()
```

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=3}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=4}
%%tab tensorflow
import math
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l
```

```{.python .input #linear-regression-scratch-linear-regression-implementation-from-scratch  n=5}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Model, Loss, and Gradient

Let us start by generating our 1,000-example synthetic dataset with true parameters $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, and noise $\sigma = 0.01$:

```{.python .input #linear-regression-scratch-model-loss-and-gradient}
%%tab pytorch, tensorflow, mxnet
w_star, b_star, noise = d2l.tensor([2, -3.4]), 4.2, 0.01
X, y = d2l.synthetic_data(w_star, b_star, noise, 2000)
X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]
```

```{.python .input #linear-regression-scratch-model-loss-and-gradient}
%%tab jax
w_star, b_star, noise = d2l.tensor([2, -3.4]), 4.2, 0.01
X, y = d2l.synthetic_data(w_star, b_star, noise, 2000, d2l.get_key())
X_train, y_train, X_val, y_val = X[:1000], y[:1000], X[1000:], y[1000:]
```

### The Parameters

Our model has a weight vector $\mathbf{w} \in \mathbb{R}^2$ and a scalar bias $b \in \mathbb{R}$. We initialize the weights with small random draws from $\mathcal{N}(0, 0.01^2)$ and set the bias to $0$ (for linear regression, starting from all zeros works just as well—see Exercise :exref:`ex_linear-regression-scratch-initialization`—though breaking symmetries with random weights becomes essential once we stack hidden layers in :numref:`chap_perceptrons`).

:begin_tab:`pytorch`
We set `requires_grad=True` on both tensors so PyTorch's autograd engine records every operation involving them.
:end_tab:

:begin_tab:`mxnet`
We call `attach_grad()` on both parameters to allocate gradient buffers and instruct MXNet to track operations inside `autograd.record()`.
:end_tab:

:begin_tab:`tensorflow`
We wrap both tensors in `tf.Variable`, which TensorFlow's `GradientTape` watches automatically.
:end_tab:

:begin_tab:`jax`
JAX arrays are immutable values; `jax.grad` differentiates pure functions with respect to their arguments directly without mutating flags on the arrays.
:end_tab:

```{.python .input #linear-regression-scratch-the-parameters}
%%tab pytorch
def init_params(sigma=0.01):
    w = torch.normal(0, sigma, (2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b
```

```{.python .input #linear-regression-scratch-the-parameters}
%%tab mxnet
def init_params(sigma=0.01):
    w, b = mnp.random.normal(0, sigma, (2, 1)), mnp.zeros(1)
    w.attach_grad()
    b.attach_grad()
    return w, b
```

```{.python .input #linear-regression-scratch-the-parameters}
%%tab tensorflow
def init_params(sigma=0.01):
    w = tf.Variable(tf.random.normal((2, 1), stddev=sigma))
    b = tf.Variable(tf.zeros(1))
    return w, b
```

```{.python .input #linear-regression-scratch-the-parameters}
%%tab jax
def init_params(sigma=0.01):
    return jax.random.normal(d2l.get_key(), (2, 1)) * sigma, jnp.zeros(1)
```

### The Forward Pass

Given a minibatch design matrix $\mathbf{X} \in \mathbb{R}^{|\mathcal{B}| \times d}$, parameters $\mathbf{w} \in \mathbb{R}^{d \times 1}$, and scalar bias $b$, the linear prediction $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$ is a single matrix-vector product plus a broadcasted addition:

```{.python .input #linear-regression-scratch-the-forward-pass}
def linreg(X, w, b):
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

### The Loss

Next we implement the mean squared error $L(\mathbf{w}, b) = \frac{1}{2|\mathcal{B}|}\sum_{i \in \mathcal{B}} (\hat{y}^{(i)} - y^{(i)})^2$. We reshape `y` to `y_hat.shape` as a defensive guard: if one tensor had shape `(n, 1)` and the other `(n,)`, subtracting them would silently broadcast into an `(n, n)` matrix of all pairwise differences!

```{.python .input #linear-regression-scratch-the-loss}
def squared_loss(y_hat, y):
    """Half the squared error, averaged over the minibatch."""
    return d2l.reduce_mean((y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2)
```

### The Gradient

Before letting automatic differentiation take over, let us check on a tiny 4-example slice that `autograd` produces the exact same gradients as our hand-derived formulas from :eqref:`eq_linreg_batch_update`:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{|\mathcal{B}|} \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y}), \qquad \frac{\partial L}{\partial b} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (\hat{y}^{(i)} - y^{(i)}).$$
:eqlabel:`eq_linreg_grad_by_hand`

Let us slice out the first four examples `(X_b, y_b)` and compare the analytical gradients with `autograd`:

```{.python .input #linear-regression-scratch-the-gradient-1}
Xb, yb = X_train[:32], y_train[:32]
w, b = init_params()
```

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab pytorch
squared_loss(linreg(Xb, w, b), yb).backward()
with torch.no_grad():
    r = linreg(Xb, w, b) - yb                # One error per example
    by_hand = torch.cat([Xb.T @ r, r.sum(0, keepdim=True)]) / len(yb)
    by_autograd = torch.cat([w.grad, b.grad[None]])
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(abs(by_hand - by_autograd).max()):.1g}')
```

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab mxnet
with autograd.record():
    loss = squared_loss(linreg(Xb, w, b), yb)
loss.backward()
r = linreg(Xb, w, b) - yb                    # One error per example
by_hand = mnp.concatenate([Xb.T @ r, r.sum(0, keepdims=True)]) / len(yb)
by_autograd = mnp.concatenate([w.grad, b.grad[None]])
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(abs(by_hand - by_autograd).max()):.1g}')
```

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab tensorflow
with tf.GradientTape() as tape:
    loss = squared_loss(linreg(Xb, w, b), yb)
grad_w, grad_b = tape.gradient(loss, [w, b])
r = linreg(Xb, w, b) - yb                    # One error per example
by_hand = tf.concat([tf.transpose(Xb) @ r,
                     tf.reduce_sum(r, 0, keepdims=True)], 0) / len(yb)
by_autograd = tf.concat([grad_w, grad_b[None]], 0)
difference = tf.reduce_max(abs(by_hand - by_autograd))
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(difference):.1g}')
```

```{.python .input #linear-regression-scratch-the-gradient-2}
%%tab jax
loss_wb = lambda w, b: squared_loss(linreg(Xb, w, b), yb)
grad_w, grad_b = jax.grad(loss_wb, argnums=(0, 1))(w, b)
r = linreg(Xb, w, b) - yb                    # One error per example
by_hand = jnp.concatenate([Xb.T @ r, r.sum(0, keepdims=True)]) / len(yb)
by_autograd = jnp.concatenate([grad_w, grad_b[None]])
print('gradient:', d2l.numpy(by_autograd).reshape(-1).round(4))
print(f'largest difference: {float(abs(by_hand - by_autograd).max()):.1g}')
```

The largest entrywise discrepancy is zero (or at most $\sim 10^{-6}$, `float32`
rounding precision), confirming that autograd computes the exact minibatch
gradient we expect.

The analytical and automatic gradients match to machine precision. From here on, we let `autograd` differentiate our forward pass and loss automatically.

## Training from Scratch

### The Update

At each step, we update the parameters in the direction opposite their gradients, $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\,(\nabla_{\mathbf{w}} L, \nabla_b L)$, taking care not to record the parameter update itself inside the computational graph (and zeroing out any accumulated gradient buffers before the next step).

### Minibatches

To sweep through the dataset in random minibatches of size $|\mathcal{B}|$, our generator `minibatches` draws a fresh random permutation of the indices $0, \ldots, n - 1$ at the start of each epoch and yields successive slices of size `batch_size` (Exercise :exref:`ex_linear-regression-scratch-why-reshuffle`):

```{.python .input #linear-regression-scratch-minibatches-1}
%%tab pytorch, mxnet, jax
def minibatches(X, y, batch_size):
    """Minibatches of (X, y), in a new random order on every call."""
    indices = np.random.permutation(len(X))
    for i in range(0, len(indices), batch_size):
        batch = indices[i : i + batch_size]
        yield X[batch], y[batch]
```

```{.python .input #linear-regression-scratch-minibatches-1}
%%tab tensorflow
def minibatches(X, y, batch_size):
    """Minibatches of (X, y), in a new random order on every call."""
    indices = np.random.permutation(len(X))
    for i in range(0, len(indices), batch_size):
        batch = indices[i : i + batch_size]
        yield tf.gather(X, batch), tf.gather(y, batch)
```

With $1{,}000$ training examples and `batch_size=32`, one epoch yields
$\lceil 1000 / 32 \rceil = 32$ minibatches: $31$ full batches of $32$ examples
plus one remainder batch of $8$. Let us inspect one pass over the generator—and
see what happens if we accidentally iterate over the *same* generator a second
time.

```{.python .input #synthetic-regression-data-reading-the-data-in-minibatches-1}
loader = minibatches(X_train, y_train, 32)
sizes = [len(y) for X, y in loader]
print(f'{len(sizes)} minibatches, the last of size {sizes[-1]}')
print(f'second pass: {len([y for X, y in loader])} minibatches')
```

Each yielded pair `(X_b, y_b)` has shapes `(32, 2)` and `(32, 1)`. With $n = 1{,}000$ and `batch_size = 32`, each pass over the dataset (one *epoch*) produces $\lceil 1000 / 32 \rceil = 32$ minibatches (31 batches of 32 examples and a final batch of 8).

### The Training Loop

We now assemble the complete training loop in `train(lr, batch_size, num_epochs)`. In each epoch, we iterate over `minibatches(X, y, batch_size)`, compute the minibatch loss, differentiate it with respect to `w` and `b`, and apply the SGD step:

```{.python .input #linear-regression-scratch-the-training-loop}
%%tab pytorch
def train(lr, num_epochs=10, batch_size=32):
    w, b = init_params()
    val_losses = []
    for epoch in range(num_epochs):
        for Xb, yb in minibatches(X_train, y_train, batch_size):
            loss = squared_loss(linreg(Xb, w, b), yb)  # forward pass, loss
            loss.backward()                            # backward pass
            with torch.no_grad():                      # not recorded:
                for param in (w, b):
                    param -= lr * param.grad           # update
                    param.grad.zero_()                 # clear
        with torch.no_grad():
            val_losses.append(squared_loss(linreg(X_val, w, b), y_val).item())
    return w, b, val_losses
```

```{.python .input #linear-regression-scratch-the-training-loop}
%%tab mxnet
def train(lr, num_epochs=10, batch_size=32):
    w, b = init_params()
    val_losses = []
    for epoch in range(num_epochs):
        for Xb, yb in minibatches(X_train, y_train, batch_size):
            with autograd.record():                    # forward pass, loss
                loss = squared_loss(linreg(Xb, w, b), yb)
            loss.backward()                            # backward pass
            for param in (w, b):
                param -= lr * param.grad               # update
        val_losses.append(float(squared_loss(linreg(X_val, w, b), y_val)))
    return w, b, val_losses
```

```{.python .input #linear-regression-scratch-the-training-loop}
%%tab tensorflow
def train(lr, num_epochs=10, batch_size=32):
    w, b = init_params()
    val_losses = []
    for epoch in range(num_epochs):
        for Xb, yb in minibatches(X_train, y_train, batch_size):
            with tf.GradientTape() as tape:            # forward pass, loss
                loss = squared_loss(linreg(Xb, w, b), yb)
            grads = tape.gradient(loss, [w, b])        # backward pass
            for param, grad in zip((w, b), grads):
                param.assign_sub(lr * grad)            # update
        val_losses.append(float(squared_loss(linreg(X_val, w, b), y_val)))
    return w, b, val_losses
```

```{.python .input #linear-regression-scratch-the-training-loop}
%%tab jax
@jax.jit
def sgd_step(w, b, X, y, lr):
    loss_wb = lambda w, b: squared_loss(linreg(X, w, b), y)  # forward, loss
    grad_w, grad_b = jax.grad(loss_wb, argnums=(0, 1))(w, b)  # backward
    return w - lr * grad_w, b - lr * grad_b                   # update

def train(lr, num_epochs=10, batch_size=32):
    w, b = init_params()
    val_losses = []
    for epoch in range(num_epochs):
        for Xb, yb in minibatches(X_train, y_train, batch_size):
            w, b = sgd_step(w, b, Xb, yb, lr)
        val_losses.append(float(squared_loss(linreg(X_val, w, b), y_val)))
    return w, b, val_losses
```

Notice that the core training loop is only a dozen lines of code—and those exact same four steps (forward pass $\rightarrow$ loss $\rightarrow$ backward pass $\rightarrow$ parameter update) will train every deep network in the rest of this book.

## Checking the Implementation

### The Expected Loss

How small should the loss become when training succeeds? Even at the true parameters $(\mathbf{w}^*, b^*)$, the residuals equal the Gaussian noise $\hat{y}^{(i)} - y^{(i)} = -\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$, so the expected loss is the *irreducible noise floor*:

$$\mathbb{E}\left[L(\mathbf{w}^*, b^*)\right] = \frac{1}{2}\,\mathbb{E}\left[\epsilon^2\right] = \frac{1}{2}\sigma^2 = \frac{1}{2}(0.01)^2 = 0.00005.$$
:eqlabel:`eq_linreg-scratch-expected-loss`

At initialization ($\mathbf{w} \approx \mathbf{0}, b = 0$), by contrast, the residual is $y = \mathbf{x}^\top\mathbf{w}^* + b^* + \epsilon$, which has mean $b^* = 4.2$ and variance $\|\mathbf{w}^*\|_2^2 + \sigma^2 = 2^2 + (-3.4)^2 + 0.01^2 = 15.5601$. Thus the expected initial loss is

$$\mathbb{E}\left[L(\mathbf{0}, 0)\right] = \frac{1}{2}\left((b^*)^2 + \|\mathbf{w}^*\|_2^2 + \sigma^2\right) = \frac{1}{2}(17.64 + 15.5601) \approx 16.6.$$

Over the course of training, we therefore expect the loss to plunge by five orders of magnitude—from roughly $16.6$ down to the noise floor $0.00005$.

### The Training Run

Let us run `train` for 3 epochs with learning rate $\eta = 0.03$ and minibatch size $|\mathcal{B}| = 32$, printing and plotting the loss after each epoch:

```{.python .input #linear-regression-scratch-training-3  n=20}
w, b, val_losses = train(lr=0.03)
d2l.plot(list(range(1, 11)), val_losses, xlabel='epoch',
         ylabel='validation loss', yscale='log')
```

Sure enough, the validation loss plunges along a straight line—dropping roughly
$7\times$ per epoch—before flattening out right near the predicted floor of
$5 \times 10^{-5}$ around epoch 7 or 8. Let us compare the measured excess loss
against our closed-form prediction epoch by epoch over the first seven epochs.

```{.python .input #linear-regression-scratch-the-training-run-1}
floor = noise ** 2 / 2                              # sigma^2 / 2
excess = (float(d2l.reduce_sum(w_star ** 2)) + b_star ** 2) / 2  # 16.6
for epoch, loss in enumerate(val_losses[:7], start=1):
    excess *= (1 - 0.03) ** (2 * 32)                # 32 steps per epoch
    print(f'epoch {epoch}: measured {loss - floor:.1e}, '
          f'predicted {excess:.1e}')
```

Measured and predicted excess losses track each other tightly over the early
epochs and gradually diverge by a modest factor of 3 or 4 by epoch 7. Why the

The loss drops from roughly $16.6$ at initialization to under $0.05$ after a single epoch, and settles right at the $0.00005$ noise floor by epoch 3!

### Recovered Parameters

Because our dataset is synthetic, we can also compare the learned $(\mathbf{w}, b)$ directly against the true $\mathbf{w}^* = (2, -3.4)^\top$ and $b^* = 4.2$:

```{.python .input #linear-regression-scratch-training-4  n=21}
w_error = d2l.numpy(d2l.reshape(w, w_star.shape) - w_star)
b_error = d2l.numpy(b - b_star)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

All three parameter errors sit at a few times $10^{-4}$—comfortably inside the

Both the weights and the bias are recovered to within a thousandth—matching the statistical estimation floor $\sigma / \sqrt{n} \approx 3 \times 10^{-4}$ we found with the exact least-squares solver in :numref:`sec_synthetic-regression-data`.

### The Learning Rate

What happens if we choose the learning rate poorly? Let us run the exact same 3-epoch loop across four learning rates spanning two orders of magnitude, $\eta \in \{0.003, 0.03, 0.3, 1.5\}$, and plot the loss trajectory over all 96 minibatch steps on a logarithmic scale (Exercise :exref:`ex_linear-regression-scratch-stability`):

```{.python .input #linear-regression-scratch-the-learning-rate}
for lr in [0.003, 0.03, 0.3, 3]:
    val_loss = train(lr)[2][-1]
    print(f'lr = {lr:<6} validation loss: {val_loss:.1e}')
```

The results match our theory across the board. At $\eta = 0.003$, the error
contracts by only $0.997$ per step, leaving the loss around $2$ to $3$ after ten
epochs (right near $16.6 \times 0.997^{640} \approx 2.4$). At $\eta = 0.3$, the
model rockets down to the noise floor within the very first epoch. And at
$\eta = 3$, every update roughly doubles the error with alternating sign, quickly
overflowing to `nan`. Keep in mind that the stability ceiling of $2$ is a property
of our *data scaling*, not a universal constant of SGD: it arises because our
features have unit variance. Scaling a feature by $s > 1$ increases the loss
curvature along that weight by $s^2$ and shrinks the maximum stable learning rate
to roughly $2 / s^2$—a major practical reason why we standardize input features
before training (:numref:`sec_pandas`).

Everything we checked in this section leaned on knowing the ground-truth
parameters, a luxury we never have on real datasets; in
:numref:`sec_generalization_basics`, we tackle how to judge models strictly from
held-out data. First, however, :numref:`sec_oo-design` packages the training
machinery we just built into reusable object-oriented classes so that later
chapters never have to rewrite this boilerplate.


The four curves tell a classic optimization story:
- **Too small ($\eta = 0.003$):** Each step barely nudges the parameters; after 96 updates the loss is still above $1$, needing ten times as many epochs to reach the floor.
- **Well tuned ($\eta = 0.03$ and $\eta = 0.3$):** The loss plunges exponentially down to the $5 \times 10^{-5}$ noise floor, with the larger learning rate arriving inside the first 20 steps (and jittering in a slightly wider noise ball once there).
- **Too large ($\eta = 1.5$):** Each step overshoots the valley and lands higher up on the opposite wall, causing the loss to explode exponentially past $10^7$!

## Concise Implementation
:label:`sec_linear_concise`

Building every piece from raw tensors taught us how linear regression works under the hood, but rewriting parameter allocation, data iterators, and training loops from scratch for every new architecture would quickly become tedious and error-prone. Deep learning frameworks provide pre-built layers, numerically tested loss functions, and standard optimizers, and in :numref:`sec_oo-design` we package our model, data, and training loop into three reusable base classes (`d2l.Module`, `d2l.DataModule`, and `d2l.Trainer`). Let us see how concisely we can express linear regression using those high-level building blocks.

### The Fully Connected Layer

In our concise implementation, a subclass `LinearRegression(d2l.Module)` wraps a single built-in fully connected layer (`Dense` or `Linear` / `LazyLinear`) with one output unit, initializing its weights from $\mathcal{N}(0, 0.01^2)$ and its bias to zero.

:begin_tab:`pytorch`
PyTorch's `nn.LazyLinear(1)` requires only the output dimension at construction time and infers the number of input features $d$ automatically on the very first forward pass!
:end_tab:

:begin_tab:`mxnet`
Gluon's `nn.Dense(1)` likewise infers its input dimension lazily when the first minibatch passes through it.
:end_tab:

:begin_tab:`tensorflow`
Keras's `layers.Dense(1)` also infers its input width lazily on the first call.
:end_tab:

:begin_tab:`jax`
Flax NNX layers allocate parameters eagerly upon construction, so `nnx.Linear` takes both the input dimension (`num_inputs=2` by default) and the output dimension (`1`).
:end_tab:

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab pytorch
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.LazyLinear(1)

    def weight_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab mxnet
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Dense(1)
        self.net.initialize(init.Normal(sigma=0.01))
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab tensorflow
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        init = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.layers.Dense(1, kernel_initializer=init)
```

```{.python .input #linear-regression-concise-defining-the-model-1}
%%tab jax
class LinearRegression(d2l.Module):  #@save
    """Linear regression built from framework components."""
    def __init__(self, num_inputs, lr, rngs=None):
        super().__init__()
        self.save_hyperparameters(ignore=['rngs'])
        rngs = nnx.Rngs(d2l.get_key()) if rngs is None else rngs
        self.net = nnx.Linear(
            num_inputs, 1, kernel_init=nnx.initializers.normal(0.01),
            rngs=rngs)
```

Notice that `LinearRegression` does not even need to define `forward`: it
inherits `Module.forward` from :numref:`sec_oo-design`, which delegates directly
to `self.net(X)`.


### The Loss Function

Instead of writing $(\hat{y} - y)^2 / 2$ by hand, we delegate `loss` to each framework's built-in mean squared error primitive (`nn.MSELoss` in PyTorch, `gluon.loss.L2Loss` in MXNet, `MeanSquaredError` in Keras, or `optax.squared_error` in JAX). Note that PyTorch, TensorFlow, and Optax omit the cosmetic $\frac{1}{2}$ factor in their squared error definitions (dividing it out here keeps the numerical loss values identical to our scratch implementation):

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = gluon.loss.L2Loss()
    return 2 * fn(y_hat, y).mean()  # L2Loss includes the 1/2
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    fn = tf.keras.losses.MeanSquaredError()
    return fn(y, y_hat)
```

```{.python .input #linear-regression-concise-defining-the-loss-function}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def loss(self, y_hat, y):
    return optax.squared_error(y_hat, y).mean()
```


### The Optimizer

Similarly, `configure_optimizers` returns each framework's built-in minibatch SGD instance (`torch.optim.SGD`, `gluon.Trainer`, `tf.keras.optimizers.SGD`, or `optax.sgd`), automatically wired to the module's registered parameters and learning rate `self.lr`:

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return optax.sgd(self.lr)
```

```{.python .input #linear-regression-concise-defining-the-optimization-algorithm}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def configure_optimizers(self):
    return gluon.Trainer(self.collect_params(),
                         'sgd', {'learning_rate': self.lr})
```


### Training with Framework Components

With `LinearRegression` defined, training reduces to four lines: instantiate the model, instantiate `d2l.SyntheticRegressionData`, create a `d2l.Trainer(max_epochs=3)`, and call `trainer.fit(model, data)`:

```{.python .input #linear-regression-concise-training-4}
print(f"validation loss: {trainer.history['val_loss'][-1]:.2e}")
```

The final validation loss lands within roughly $15\%$ of the theoretical noise
floor $\sigma^2 = 10^{-4}$.

To verify parameter recovery directly, we add a small `get_w_b` helper that

The validation and training loss curves drop right down to the $0.00005$ noise floor, mirroring our from-scratch implementation. Let us add a small helper method `get_w_b` to extract the learned weight tensor and bias from the framework layer and compare them against $(\mathbf{w}^*, b^*)$:

```{.python .input #linear-regression-concise-training-2}
%%tab pytorch
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.weight.detach(), self.net.bias.detach()
```

```{.python .input #linear-regression-concise-training-2}
%%tab mxnet
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.weight.data(), self.net.bias.data()
```

```{.python .input #linear-regression-concise-training-2}
%%tab tensorflow
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.kernel, self.net.bias
```

```{.python .input #linear-regression-concise-training-2}
%%tab jax
@d2l.add_to_class(LinearRegression)  #@save
def get_w_b(self):
    return self.net.kernel[...], self.net.bias[...]
```

```{.python .input #linear-regression-concise-training-3}
w, b = model.get_w_b()
w_error = d2l.numpy(d2l.reshape(w, data.w.shape) - data.w)
b_error = d2l.numpy(b - data.b)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

Once again, the errors in $\hat{\mathbf{w}}$ and $\hat{b}$ are only a few times
$10^{-4}$, matching our scratch run because the dominant error sources (label

## Summary

- **From-scratch implementation:** A complete deep learning training loop requires only four primitives—a parameterized forward pass ($\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$), a scalar loss ($\frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$), reverse-mode automatic differentiation, and a minibatch SGD parameter update.
- **Verification via the noise floor:** On synthetic data with additive noise $\mathcal{N}(0, \sigma^2)$, the expected loss at the true parameters is $\frac{1}{2}\sigma^2$ ($0.00005$ for $\sigma = 0.01$), providing an exact quantitative test that optimization has converged.
- **Learning-rate sensitivity:** Too small a learning rate crawls slowly; a well-chosen range converges exponentially to the noise floor; too large a learning rate overshoots the quadratic bowl and diverges.
- **Concise framework implementation:** High-level framework layers (`LazyLinear` / `Dense` / `nnx.Linear`), built-in MSE losses, and `SGD` optimizers—organized inside `d2l.Module` and driven by `d2l.Trainer`—reproduce the exact same model and parameter recovery in a fraction of the code.

## Exercises

1. [code] **Initialization at the extremes.** `init_params`
   draws the weights with standard deviation `sigma=0.01`. Predict whether
   training reaches the noise floor when the weights are initialized to
   exactly zero, and when `sigma=1000`. Use
   :eqref:`eq_linreg-scratch-expected-loss` to estimate how many more epochs
   the second case needs at learning rate 0.03, then check your estimate.
   :exlabel:`ex_linear-regression-scratch-initialization`

1. [code] **Second derivatives.** Compute by hand the Hessian of the
   minibatch loss with respect to $(\mathbf{w}, b)$, and state which
   quantities it depends on. Then compute it with automatic
   differentiation, differentiating through the gradient as in
   :numref:`sec_autograd`, for one fixed minibatch of the synthetic data,
   once at initialization and once after
   training. Compare both results with your formula and with the curvature
   of :eqref:`eq_linreg-scratch-expected-loss`.

1. [code] **Stability and minibatch size.** ● The expected loss predicts
   that steps along its gradient diverge for $\eta > 2$. Let
   $\tilde{\mathbf{x}} = (\mathbf{x}, 1)$ be an input with a $1$ appended
   for the bias.
    1. For full-batch gradient descent, show that each step multiplies the
       difference between the parameters and the least-squares solution by
       the matrix $\mathbf{I} - \eta \mathbf{H}$, where $\mathbf{H}$ is the
       average of $\tilde{\mathbf{x}}\tilde{\mathbf{x}}^\top$ over the
       training set. Compute the eigenvalues of $\mathbf{H}$ and predict the
       largest stable learning rate.
    1. Train with minibatches of 1,000, 32, and 1 example, giving each run the
       same number of updates, say 320. For each minibatch size, find the
       largest learning rate in a grid of your choice at which the final
       training loss is below $10^{-3}$.
    1. Show that a step on a single example multiplies the component of the
       parameter error along $\tilde{\mathbf{x}}$ by
       $1 - \eta \|\tilde{\mathbf{x}}\|^2$, and use
       $\mathbb{E}\|\tilde{\mathbf{x}}\|^2 = 3$ to explain the limit you found for
       minibatches of one example.
   :exlabel:`ex_linear-regression-scratch-stability`

1. **Why reshuffle.** `minibatches` draws a fresh random order in every
   epoch. Explain which property of the minibatch gradient this
   preserves. Then construct a fixed ordering of the training set, such as
   sorting by label, under which minibatch SGD without reshuffling behaves
   differently, and describe the trajectory of $(\mathbf{w}, b)$ within an
   epoch and across epochs.
   :exlabel:`ex_linear-regression-scratch-why-reshuffle`

1. [code] **Robust losses.** ● Replace `squared_loss` with the mean
   absolute error
   `d2l.reduce_mean(d2l.abs(y_hat - d2l.reshape(y, (-1, 1))))`.
    1. Train on the synthetic data and compare the recovered
       parameters and the loss curve with those of the squared loss.
    1. Set one training label to $y^{(5)} = 10{,}000$, the corruption used in
       :numref:`subsec_linear-regression-loss-function`, and train with
       both losses. Record $b$ after every minibatch update, plot it
       against the iteration for both runs, and explain the pattern of the
       squared-loss curve, including what determines its final value.
    1. On the corrupted data, replace the loss with the minibatch average of
       the Huber loss

        $$l_\delta(y, y') = \begin{cases}\frac{1}{2}(y-y')^2 & \textrm{ if } |y-y'| \leq \delta \\ \delta \left(|y-y'| - \frac{\delta}{2}\right) & \textrm{ otherwise,}\end{cases}$$

        whose quadratic part matches the squared loss of this section.
        Train for $\delta \in \{1, 10, 10^2, 10^3, 10^4\}$, and plot the
        distance $|\hat{b} - b^*|$ of the learned bias from the generating
        bias $b^* = 4.2$ against $\delta$ on logarithmic axes. Explain the
        shape of the curve from the derivative of $l_\delta$ with respect
        to $y'$, and relate the result to the penalty curves
        of :numref:`fig_linreg-loss-menu`.
   :exlabel:`ex_linear-regression-scratch-robust-losses`

1. [code] **A layer without a bias.** Build `LinearRegression` with a
   layer that has no bias (every framework's dense layer has an argument
   that turns the bias off), and train it as in this section.
    1. Before training, predict the value at which the validation loss
       levels off, from the expected loss
       :eqref:`eq_linreg-scratch-expected-loss` doubled for the mean
       squared error. Then check your prediction.
    1. Predict what the weights converge to and how far they end from
       $\mathbf{w}^*$. Since the features have mean zero, the missing bias
       acts on the weights like label noise of standard deviation
       $b^* = 4.2$; it enters both the estimation error of
       :numref:`subsec_synthetic-regression-data-recovery` and the noise
       ball of minibatch SGD (:numref:`sec_linear_regression`). Compare
       your prediction with the errors of the trained weights.
   :exlabel:`ex_linear-regression-concise-no-bias`

1. [code] **Several outputs.** Change the output width of the layer in
   `LinearRegression` from 1 to 2. After constructing `data`, append to
   `data.y` a second label column
   $y_2 = \mathbf{x}^\top \mathbf{w}_2^* + b_2^* + \epsilon_2$, computed
   from `data.X` with generating parameters of your choice and noise of
   the same size.
    1. Show that the squared loss summed over the two outputs splits into
       two independent linear regressions, one per output. Conclude that
       the parameters learned for one output do not depend on the labels of
       the other.
    1. The framework losses average over every entry of the
       $|\mathcal{B}| \times 2$ matrix of errors of a minibatch
       $\mathcal{B}$, not only over its examples. State by which factor
       the gradient of each output's parameters shrinks relative to the
       single-output model, and which learning rate reproduces the
       single-output training run. Confirm this by training both models
       from the same initial parameters on the same minibatches.

1. [code] **Feature scale and step size.** Multiply the first column of
   `data.X` by 100 after constructing `data`, leaving `data.y` unchanged,
   so that the generating weight of that feature becomes $0.02$. Recall
   from exercise :exref:`ex_linear-regression-scratch-stability` that
   gradient descent diverges once $\eta$ exceeds $2/\mu_{\max}$, where
   $\mu_{\max}$ is the largest eigenvalue of the Hessian $\mathbf{H}$ of
   the loss.
    1. Compute $\mathbf{H}$ for `LinearRegression.loss` and estimate its
       eigenvalues before and after the rescaling.
    1. Predict the largest learning rate for which training does not
       diverge, before and after the rescaling, and check both predictions
       by training at learning rates around them.
    1. At a learning rate just below the new limit, predict how many
       updates the second weight needs to reduce its error by a factor of
       100, and compare the prediction with a measurement. Then standardize
       the rescaled feature as in :numref:`sec_pandas`, and show that
       `lr=0.015` works again.

1. [code] **Optimization versus estimation error.** ● Train
   `LinearRegression` with `lr=0.015` and `max_epochs=10` on
   `SyntheticRegressionData` with `num_train` set to each of
   $5, 50, 500, 5{,}000$ and `batch_size=min(32, num_train)`. For each run,
   compare the distance $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_2$ of the
   trained weights from the generating weights $\mathbf{w}^*$, stored in
   `data.w`, with the same distance for the least-squares solution on the
   same training data, computed as in the recovery subsection
   (:numref:`subsec_synthetic-regression-data-recovery`).
    1. Count the parameter updates in each run, and explain the ratio of
       the two errors at each `num_train`.
    1. Find the number of epochs after which the run with `num_train=5`
       comes within a factor of two of its least-squares error.


:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/44)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/45)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/204)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17977)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.3]{.kicker}

Linear regression with tensors and autograd<br>
**parameters · the gradient by hand · minibatch SGD · a dozen-line loop · a known answer**
:::
:::

::: {.slide title="A Wrong Training Loop Can Still Lower the Loss"}
[The problem]{.kicker}

- Labels of shape $(32,)$ minus predictions of shape $(32, 1)$: no error.
  Broadcasting forms a $32 \times 32$ matrix of differences.
- Ten epochs on that loss lower it steadily, while $\mathbf{w}$ ends far
  from $(2, -3.4)$.
- Nothing in the loss curve says that the program is wrong.

::: {.d2l-note}
Train on data with a **known answer**:
$\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, noise $\sigma = 0.01$. A correct
program must reach the loss floor $\sigma^2/2 = 5 \times 10^{-5}$ and
recover $\mathbf{w}^*$ and $b^*$ to within about $10^{-3}$.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- write parameters, forward pass, loss, minibatch SGD, and a dozen-line
  training loop with tensors and autograd;
- derive the gradient by hand and check it against autograd;
- put the four operations of a training step in the right order;
- predict the loss curve and the parameter error before training;
- say which learning rates converge and which diverge.

**Running example:** $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$,
$\sigma = 0.01$; 1,000 training examples in minibatches of 32.
:::

::: {.slide title="Every Part of Training, Written by Hand"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| Synthetic Data | data with known $\mathbf{w}^*, b^*$ |
| **Implementation from Scratch** | **SGD and the training loop by hand** |
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| Concise Implementation | framework layers and losses |
| Generalization | training vs. test error |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** the squared loss and the SGD update, autograd, the synthetic
  data and its threshold of $10^{-3}$.
- **Skips:** the classes that package this code (the next deck) and
  framework layers.
:::
:::
:::

::: {.slide}
::: {.divider}
[01]{.dnum}

[The model and its gradient]{.dtitle}

[parameters · forward pass · loss · the gradient by hand]{.dsub}
:::
:::

::: {.slide title="A Prediction Is One (32×2)(2×1) Product; the Loss a Mean"}
[The model]{.kicker}

Start from $\mathbf{w} \sim \mathcal{N}(0, 0.01^2)$, $b = 0$; a minibatch
$\mathcal{B}$ of 32 examples:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b, \qquad L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} \tfrac{1}{2}\bigl(\hat{y}^{(i)} - y^{(i)}\bigr)^2$$

@linear-regression-scratch-the-forward-pass

@linear-regression-scratch-the-loss

::: {.d2l-note}
The loss first reshapes $\mathbf{y}$ into a $(32, 1)$ column: a $(32,)$
vector would broadcast to $32 \times 32$.
:::
:::

::: {.slide title="The Gradient Is the Error-Weighted Input"}
[The gradient]{.kicker}

One example, error $r = \hat{y} - y$:

$$\frac{\partial l}{\partial \mathbf{w}} = r\, \mathbf{x}, \qquad \frac{\partial l}{\partial b} = r$$

. . .

A minibatch $\mathcal{B}$: features $\mathbf{X}$, labels $\mathbf{y}$,
errors $\mathbf{r} = \mathbf{X}\mathbf{w} + b - \mathbf{y}$:

$$\nabla_{\mathbf{w}} L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top \mathbf{r}, \qquad \frac{\partial L_{\mathcal{B}}}{\partial b} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} r^{(i)}$$

::: {.d2l-note .rule}
One matrix–vector product per minibatch.
:::
:::

::: {.slide title="Autograd Returns the Gradient Derived by Hand"}
[The gradient]{.kicker}

One minibatch of 32 examples, fresh parameters:

@linear-regression-scratch-the-gradient-2

The largest difference is zero or at most about $10^{-6}$: float32 rounding.
:::

::: {.slide title="Check: The Gradient at w = 0, b = 0"}
[Check]{.kicker}

Start from $\mathbf{w} = \mathbf{0}$, $b = 0$. Averaged over many examples,
what is the gradient, and where does its negative point? The features are
independent standard normal.

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The errors are $r = -y$, so the gradient is
$\mathbb{E}[\mathbf{x}\, r] = -\mathbb{E}[\mathbf{x}\mathbf{x}^\top]\mathbf{w}^* = -\mathbf{w}^* = (-2, 3.4)$
for $\mathbf{w}$ and $\mathbb{E}[r] = -b^* = -4.2$ for $b$. The negative gradient
points straight at the true parameters.
:::
:::
:::

::: {.slide}
::: {.divider}
[02]{.dnum}

[Minibatch SGD and the loop]{.dtitle}

[one step, in order · random minibatches · a dozen lines]{.dsub}
:::
:::

::: {.slide title="Four Operations per Step, in a Fixed Order"}
[One step]{.kicker}

1. **Forward pass and loss** $L_{\mathcal{B}}$, recorded for
   differentiation.
2. **Backward pass**: the gradient of this minibatch's loss.
3. **Update** $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla L_{\mathcal{B}}$,
   *not* recorded.
4. **Clear** any gradient the framework keeps.

::: {.d2l-note .rule}
The next backward pass must see only its own minibatch, so kept gradients
are cleared before it. The update is not part of the loss, so it is never
recorded.
:::
:::

::: {.slide title="Check: Forgetting to Clear the Gradients"}
[Check]{.kicker}

Suppose `backward` added each new gradient to the stored one, and nothing
ever cleared it. What does step $t$ subtract, and what happens to the
parameters?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta(\mathbf{g}_1 + \cdots + \mathbf{g}_t)$: every earlier
gradient is applied again at every step. The parameters run past the
minimum, and on a quadratic they oscillate around it without settling.
:::
:::
:::

::: {.slide title="A Random Minibatch Points the Right Way; a Sorted One Need Not"}
[Minibatches]{.kicker}

At $\mathbf{w} = \mathbf{0}$, $b = 0$ a minibatch's gradient for the bias is
$\partial L_{\mathcal{B}} / \partial b = -(\text{mean label})$. The labels
have mean 4.2 and standard deviation $\sqrt{2^2 + 3.4^2} \approx 3.9$.

| minibatch of 32 | mean label | $\partial L_{\mathcal{B}} / \partial b$ | step on $b$ |
|---|---|---|---|
| drawn at random | $4.2 \pm 0.7$ | $\approx -4.2$ | up, toward 4.2 |
| the 32 smallest labels | $\approx -4.6$ | $\approx +4.6$ | down, away from 4.2 |

::: {.d2l-note}
A random minibatch's gradient is an unbiased estimate of the full
gradient. Minibatches in a fixed order, sorted by label, bias every step.
:::

[The lowest 3.2% of a normal distribution average 2.2 standard deviations below its mean.]{.dgm-cap}
:::

::: {.slide title="A Minibatch Loader Shuffles, Then Cuts Into Pieces of 32"}
[Minibatches]{.kicker}

@linear-regression-scratch-minibatches-1

- A new random order on every call; each example is used once per epoch.
- `yield` makes the function a **generator**: it hands the loop one
  minibatch at a time.
:::

::: {.slide title="32 Minibatches per Epoch, and a Generator Runs Out After One"}
[Minibatches]{.kicker}

$\lceil 1000/32 \rceil = 32$ minibatches: $31 \times 32 + 8$.

@synthetic-regression-data-reading-the-data-in-minibatches-1

::: {.d2l-note .warn}
A generator is exhausted after one pass, without an error, so the training
loop calls `minibatches` anew in every epoch.
:::
:::

::: {.slide title="A Dozen Lines Train Linear Regression"}
[The loop]{.kicker}

@linear-regression-scratch-the-training-loop

::: {.d2l-note}
The comments mark the operations of a step; the other lines run the two
loops and record the validation loss after every epoch.
:::
:::

::: {.slide}
::: {.divider}
[03]{.dnum}

[Checking the result]{.dtitle}

[a prediction · the run · the parameters · the learning rate]{.dsub}
:::
:::

::: {.slide title="The Expected Loss Is Half the Squared Parameter Error Plus σ²/2"}
[A prediction]{.kicker}

Standard normal features, independent of the noise:

$$\mathbb{E}\bigl[\tfrac{1}{2} r^2\bigr] = \tfrac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|^2 + \tfrac{1}{2}(b - b^*)^2 + \tfrac{1}{2}\sigma^2$$

- Floor: $\sigma^2/2 = 5 \times 10^{-5}$; nothing gets below it.
- Gradient $= (\mathbf{w} - \mathbf{w}^*, b - b^*)$: a step multiplies the
  parameter error by $1 - \eta$, the excess loss by $(1 - \eta)^2$.
- Start: $\tfrac{1}{2}(2^2 + 3.4^2 + 4.2^2) \approx 16.6$ above the floor.

At $\eta = 0.03$, by how much should one epoch of 32 steps lower the excess?

::: {.fragment .fade-up}
::: {.d2l-note}
$0.97^{64} \approx 0.14$: a factor of about 7 per epoch.
:::
:::
:::

::: {.slide title="The Loss Falls 7× per Epoch to the Floor of 5×10⁻⁵"}
[The run]{.kicker}

::: {.cols .vc}
::: {.col}
Ten epochs at $\eta = 0.03$; the validation loss on a log scale.

A straight line, then the floor after seven or eight epochs, as predicted.
:::

::: {.col .fig .big}
@!linear-regression-scratch-training-3
:::
:::
:::

::: {.slide title="Measured Excess Tracks the Prediction for Seven Epochs"}
[The run]{.kicker}

Validation loss minus $\sigma^2/2$, against $16.6 \times 0.97^{64k}$ after
epoch $k$:

@!linear-regression-scratch-the-training-run-1

::: {.d2l-note}
The two drift apart by at most a factor of 3 or 4. The prediction takes
$\mathbf{H}$, the average of $(\mathbf{x}, 1)(\mathbf{x}, 1)^\top$, to be
$\mathbf{I}$; its eigenvalues lie within about 10% of 1, and the smallest
sets the late rate.
:::
:::

::: {.slide title="The Parameters Come Within About 10⁻³ of (2, −3.4) and 4.2"}
[The parameters]{.kicker}

Parameter errors $\hat{\mathbf{w}} - \mathbf{w}^*$ and $\hat{b} - b^*$:

@!linear-regression-scratch-training-4

- Even exact least squares misses by $\sigma/\sqrt{n} = 0.01/\sqrt{1000} \approx 3 \times 10^{-4}$.
- The loss cannot see this: $\tfrac{1}{2} \cdot 3 \cdot (5 \times 10^{-4})^2 \approx 4 \times 10^{-7}$,
  under 1% of the floor.

::: {.d2l-note .rule}
Comparing parameters catches bugs that a loss curve hides.
:::
:::

::: {.slide title="η = 0.003 Crawls, η = 3 Diverges"}
[The learning rate]{.kicker}

Along the averaged gradient, the parameter error is multiplied by
$1 - \eta$ per step: it shrinks for $0 < \eta < 2$ and grows for
$\eta > 2$. With minibatches the limit is somewhat lower.

@linear-regression-scratch-the-learning-rate

::: {.d2l-note}
At $\eta = 3$ the parameter error is multiplied by about $-2$ per step, and
the loss overflows.
:::
:::

::: {.slide title="Check: Ten Times More Noise"}
[Check]{.kicker}

Generate the data with $\sigma = 0.1$ instead of $0.01$ and train the same
way. Where does the loss level off, how large are the parameter errors, and
how fast does the loss fall?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** The floor $\sigma^2/2$ rises 100-fold, to $5 \times 10^{-3}$.
The parameter errors grow 10-fold, to about
$\sigma/\sqrt{n} \approx 3 \times 10^{-3}$.
The rate, a factor of 7 per epoch, is unchanged: the noise does not change
the gradient of the expected loss.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **Model:** $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$; loss
  $L_{\mathcal{B}}$, the mean of $\tfrac{1}{2}(\hat{y} - y)^2$.
- **Gradient:** the error-weighted input, $\frac{1}{|\mathcal{B}|}\mathbf{X}^\top\mathbf{r}$;
  autograd agrees up to float32 rounding.
- **Step:** forward and loss, backward, unrecorded update, clear kept
  gradients; a dozen lines train the model.
- **Check:** loss $16.6 \to 5 \times 10^{-5}$ at 7× per epoch; parameter
  errors a few times $10^{-4}$, within $10^{-3}$.
- **Learning rate:** the averaged gradient converges for $0 < \eta < 2$;
  with minibatches the limit is somewhat lower; $\eta = 3$ diverges.

**Next:** object-oriented design, the same code as `Module`,
`DataModule`, `Trainer`, trained once more.
:::

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.5]{.kicker}

Linear regression from framework parts<br>**layer · loss · optimizer · the same trainer**
:::
:::

::: {.slide title="Hand-Written Parts Do Not Scale to 25 Million Parameters"}
[The problem]{.kicker}

- From scratch: we created, initialized, and updated **3 numbers**, `w`
  and `b`, with our own code, then packed that code into classes.
- ResNet-50, a standard image classifier: **25.6 million** parameters in
  **161 tensors**, trained with update rules that keep a running average
  of past gradients for every parameter.
- A slip in a hand-written part rarely raises an error. The model trains,
  only less well.

::: {.d2l-note}
Frameworks ship layers, losses, and optimizers: tested by many users, fast
on every device, and swapped by changing one line.
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- name the framework part that replaces each hand-written part;
- build linear regression from a layer, a loss, and an optimizer;
- convert a learning rate between losses that differ by a constant factor;
- say what each framework loss does with labels of the wrong shape;
- predict where the loss levels off, and judge the trained parameters.

**Running example:** $\mathbf{w}^* = (2, -3.4)$, $b^* = 4.2$, noise
$\sigma = 0.01$; 1,000 training examples in minibatches of 32.
:::

::: {.slide title="The Concise Version Reuses Everything but the Model's Parts"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Linear Regression | model, loss, closed form, minibatch SGD |
| Synthetic Data | data with known $\mathbf{w}^*, b^*$ |
| Implementation from Scratch | SGD and the training loop by hand |
| Object-Oriented Design | `Module`, `DataModule`, `Trainer` |
| **Concise Implementation** | **framework layers and losses** |
| Generalization | training vs. test error |
| Weight Decay | a penalty on large weights |
:::

::: {.col .narrow}
- **Uses:** `Module`, `Trainer`, `SyntheticRegressionData`, the scratch
  model.
- **Skips:** how layers infer shapes and initialize (the Computation
  chapter); other update rules (the Optimization chapter).
:::
:::
:::

::: {.slide title="Each Hand-Written Part Has a Framework Counterpart"}
[Overview]{.kicker}

| from scratch | framework |
|---|---|
| `w`, `b`, and `X @ w + b` | one fully connected **layer** |
| $\tfrac{1}{2}$ squared error, averaged | built-in **mean squared error** |
| our `SGD` class | an **optimizer** object |
| `Module`, `DataModule`, `Trainer` | unchanged |

::: {.d2l-note}
The trainer calls only the methods that every `Module` has. It cannot tell
the two models apart.
:::
:::

::: {.slide title="LazyLinear(1) Is the Whole Model" only="pytorch"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-1}
self.net = nn.LazyLinear(1)  # 1 output
```

- `LazyLinear` takes the number of inputs from its first input.
- Its parameters do not exist before that first call, so we initialize
  them after it, from $\mathcal{N}(0, 0.01^2)$ and with bias 0.
:::

::: {.slide title="Dense(1) Is the Whole Model" only="mxnet"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-2}
self.net = nn.Dense(1)  # 1 output
self.net.initialize(init.Normal(sigma=0.01))
```

- `Dense` takes the number of inputs from its first input.
- `initialize` records the initializer; the parameters are created at the
  first forward pass, the bias as 0.
:::

::: {.slide title="Dense(1) Is the Whole Model" only="tensorflow"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-3}
init = tf.initializers.RandomNormal(stddev=0.01)
self.net = tf.keras.layers.Dense(1, kernel_initializer=init)
```

- `Dense` takes the number of inputs from its first input.
- `kernel_initializer` draws the weights; the bias starts at 0.
:::

::: {.slide title="nnx.Linear(2, 1) Is the Whole Model" only="jax"}
[Layer]{.kicker}

A *fully connected* layer computes $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
and holds $\mathbf{w}$ and $b$ itself. Our model's constructor
creates it:

```{.python #linear-regression-concise-exercises-4}
self.net = nnx.Linear(
    num_inputs, 1, kernel_init=nnx.initializers.normal(0.01),
    rngs=rngs)
```

- NNX has no lazy layers: we pass the number of inputs.
- `rngs` supplies the random keys; the parameters exist as soon as the
  constructor returns. The bias starts at 0.
:::

::: {.slide title="The Framework Loss Is Twice Ours, So η Halves" except="mxnet"}
[Loss]{.kicker}

@linear-regression-concise-defining-the-loss-function

| | loss | gradient |
|---|---|---|
| ours | $\frac{1}{2\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $g$ |
| framework | $\frac{1}{\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $2g$ |

Same minimizer, but every gradient doubles. A step is $\eta$ times the
gradient, so $\eta = 0.03/2 = 0.015$ takes the scratch steps.
:::

::: {.slide title="The Framework Loss Is Twice Ours, So η Halves" only="mxnet"}
[Loss]{.kicker}

@linear-regression-concise-defining-the-loss-function

| | loss | gradient |
|---|---|---|
| ours | $\frac{1}{2\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $g$ |
| doubled `L2Loss` | $\frac{1}{\vert\mathcal{B}\vert}\sum_i (\hat{y}^{(i)} - y^{(i)})^2$ | $2g$ |

`L2Loss` includes the $\frac{1}{2}$; doubled, it is the mean squared
error. Every gradient doubles, so $\eta = 0.03/2 = 0.015$ takes the
scratch steps.
:::

::: {.slide title="Check: A Summed Loss"}
[Check]{.kicker}

A library's loss **sums** $(\hat{y} - y)^2$ over the minibatch of 32,
with no $\frac{1}{2}$. Which $\eta$ reproduces the scratch steps at
$\eta = 0.03$?

::: {.d2l-note .fragment .fade-up}
**Answer.** The sum is 32 times the mean, and the mean is twice our loss:
the gradient is $64g$. So $\eta = 0.03/64 \approx 4.7 \times 10^{-4}$.
:::
:::

::: {.slide title="MSELoss Warns About a Shape Mismatch, Then Trains on the Wrong Loss" only="pytorch"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ broadcast
to a $32 \times 32$ matrix of differences.

- `nn.MSELoss` prints a warning, then averages all 1,024 entries.
- That loss pulls every prediction toward the mean of the 32 labels.

::: {.d2l-note .warn}
Read the warnings. Reshape the labels to the predictions' shape.
:::
:::

::: {.slide title="L2Loss Reshapes the Labels, So the Shape Bug Cannot Occur" only="mxnet"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ would
broadcast to a $32 \times 32$ matrix of differences.

- `L2Loss` first reshapes the labels to the shape of the predictions.
- The difference has shape $(32, 1)$, as intended.
:::

::: {.slide title="Keras Aligns the Shapes, So the Shape Bug Cannot Occur" only="tensorflow"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ would
broadcast to a $32 \times 32$ matrix of differences.

- `MeanSquaredError` brings both to one shape before subtracting.
- Note the order: label first, prediction second.
:::

::: {.slide title="optax.squared_error Refuses Mismatched Shapes" only="jax"}
[Loss · shapes]{.kicker}

Labels of shape $(32,)$ against predictions of shape $(32, 1)$ would
broadcast to a $32 \times 32$ matrix of differences.

- `optax.squared_error` raises an error instead.
- The bug is no longer silent: fix the shape and run again.
:::

::: {.slide title="An Optimizer Object Replaces Our SGD Class"}
[Optimizer]{.kicker}

@linear-regression-concise-defining-the-optimization-algorithm

- Each step: $\theta \leftarrow \theta - \eta\, \nabla_\theta L$ for every
  parameter $\theta$ of every layer.
- Another update rule from the same module: change this one line.
:::

::: {.slide title="The Same Trainer Fits the Framework Model"}
[Training]{.kicker}

@-linear-regression-concise-training-1

- Data drawn as before; a new model class at $\eta = 0.015$.
- `Trainer` and `fit` are unchanged: they call only the methods that
  every `Module` has.
:::

::: {.slide title="The Loss Falls About 7× per Epoch, Then Levels Off at σ² = 10⁻⁴"}
[Training]{.kicker}

::: {.cols .vc}
::: {.col}
Ten epochs at $\eta = 0.015$, log scale.

A straight line, the per-epoch factor of the scratch run, then the floor:
the noise variance $\sigma^2$.
:::

::: {.col .fig .big}
@!linear-regression-concise-training-1
:::
:::
:::

::: {.slide title="Ten Epochs Recover w* and b* to Within About 10⁻³"}
[Parameters]{.kicker}

The parameters live inside the layer; `get_w_b` reads them out of `net`:

@linear-regression-concise-training-2

@!linear-regression-concise-training-3

::: {.d2l-note}
Errors of a few times $10^{-4}$, the size the scratch run left, from the
same sources: the noise in the training labels and what ten epochs leave
of the initial error.
:::
:::

::: {.slide title="Check: Why Does This Loss Level Off Twice as High?"}
[Check]{.kicker}

The scratch run levels off near $5 \times 10^{-5}$, this run near
$10^{-4}$, on data from the same generator. Why the factor 2? And what
does the square root of this run's floor tell us about the predictions?

::: {.d2l-note .fragment .fade-up}
**Answer.** Even the true parameters leave the noise, of variance
$\sigma^2$. The mean squared error levels off at $\sigma^2$; our loss
carries the $\frac{1}{2}$ and levels off at $\sigma^2/2$. And
$\sqrt{10^{-4}} = 0.01 = \sigma$: a typical prediction misses its label by
the noise level, in the units of the label.
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

| | from scratch | framework |
|---|---|---|
| parameters, forward | `w`, `b`, `X @ w + b` | fully connected layer |
| loss | $\frac{1}{2}$ mean squared error | mean squared error |
| update | our `SGD`, $\eta = 0.03$ | optimizer, $\eta = 0.015$ |
| training loop | `Trainer` | the same `Trainer` |
| loss floor | $\sigma^2/2 = 5 \times 10^{-5}$ | $\sigma^2 = 10^{-4}$ |

**Next:** how well does a trained model predict examples it has not seen?
:::
