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

Let us start by generating $2{,}000$ synthetic examples, $1{,}000$ for training and $1{,}000$ for validation, with true parameters $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, and noise $\sigma = 0.01$:

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

Before letting automatic differentiation take over, let us check on a single minibatch of 32 examples that `autograd` produces the exact same gradients as our hand-derived formulas from :eqref:`eq_linreg_batch_update`:

$$\frac{\partial L}{\partial \mathbf{w}} = \frac{1}{|\mathcal{B}|} \mathbf{X}^\top (\hat{\mathbf{y}} - \mathbf{y}), \qquad \frac{\partial L}{\partial b} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} (\hat{y}^{(i)} - y^{(i)}).$$
:eqlabel:`eq_linreg_grad_by_hand`

Let us slice out the first 32 training examples `(Xb, yb)` and compare the analytical gradients with `autograd`:

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
gradient we expect. From here on, we let `autograd` differentiate our forward pass and loss automatically.

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

Each yielded pair `(Xb, yb)` has shapes `(32, 2)` and `(32, 1)`. The first pass (one *epoch*) yields all $32$ minibatches as expected, while the second pass
yields zero! A Python generator is single-use and exhausts silently once
consumed, which is why our training loop calls `minibatches(...)` fresh at the
start of every epoch.

### The Training Loop

We now assemble the complete training loop in `train(lr, num_epochs, batch_size)`. In each epoch, we iterate over `minibatches(X, y, batch_size)`, compute the minibatch loss, differentiate it with respect to `w` and `b`, and apply the SGD step:

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

The same computation also predicts how *fast* the loss should fall. At arbitrary
parameters, the residual is
$r = \mathbf{x}^\top(\mathbf{w} - \mathbf{w}^*) + (b - b^*) - \epsilon$. Because
the two features are independent standard normal variables and the noise is
independent of both, every cross term has expectation zero, and

$$\mathbb{E}\left[L(\mathbf{w}, b)\right] = \tfrac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|_2^2 + \tfrac{1}{2}(b - b^*)^2 + \tfrac{1}{2}\sigma^2.$$

The gradient of this expected loss with respect to $(\mathbf{w}, b)$ is simply
the *parameter error* $(\mathbf{w} - \mathbf{w}^*, b - b^*)$, so a gradient step
of size $\eta$ shrinks the parameter error by the factor $1 - \eta$ and the
*excess loss* above the floor by $(1 - \eta)^2$. Minibatch gradients match the
expected gradient only on average, so this gives an approximate trajectory rather
than an exact identity. At $\eta = 0.03$, one epoch of $32$ minibatch updates
should reduce the excess loss by a factor of roughly
$1 / (1 - 0.03)^{64} \approx 7$.

### The Training Run

Let us run `train` for 10 epochs with learning rate $\eta = 0.03$ and minibatch size $|\mathcal{B}| = 32$, plotting the validation loss after each epoch on a logarithmic scale:

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
epochs and gradually drift apart, by at most a modest factor of 3 or 4 by
epoch 7. Why the slight drift? Our idealized prediction assumed the sample
covariance matrix
$\mathbf{H} = \frac{1}{n}\sum_i \tilde{\mathbf{x}}^{(i)}(\tilde{\mathbf{x}}^{(i)})^\top$
(see exercise :exref:`ex_linear-regression-scratch-stability`) equals the exact
identity matrix. On a finite sample of $1{,}000$ examples, its eigenvalues sit
within roughly $10\%$ of $1$, and over several epochs these small deviations
compound, so the empirical run drifts slightly ahead of or behind the idealized
prediction. Both the decay rate and the noise floor match theory.

### Recovered Parameters

Because our dataset is synthetic, we can also compare the learned $(\mathbf{w}, b)$ directly against the true $\mathbf{w}^* = (2, -3.4)^\top$ and $b^* = 4.2$:

```{.python .input #linear-regression-scratch-training-4  n=21}
w_error = d2l.numpy(d2l.reshape(w, w_star.shape) - w_star)
b_error = d2l.numpy(b - b_star)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

Both the weights and the bias are recovered to within a thousandth—matching the statistical estimation floor $\sigma / \sqrt{n} \approx 3 \times 10^{-4}$ we found with the exact least-squares solver in :numref:`sec_synthetic-regression-data`.

### The Learning Rate

What happens if we choose the learning rate poorly? Let us run the exact same 10-epoch loop across four learning rates spanning three orders of magnitude, $\eta \in \{0.003, 0.03, 0.3, 3\}$, and print the final validation loss of each run (Exercise :exref:`ex_linear-regression-scratch-stability`):

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
Flax NNX layers allocate parameters eagerly upon construction, so `nnx.Linear` takes both the input dimension (`num_inputs`, which we set to `2`) and the output dimension (`1`).
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

Instead of writing $(\hat{y} - y)^2 / 2$ by hand, we delegate `loss` to each framework's built-in mean squared error primitive (`nn.MSELoss` in PyTorch, `gluon.loss.L2Loss` in MXNet, `MeanSquaredError` in Keras, or `optax.squared_error` in JAX). Note that PyTorch, TensorFlow, and Optax omit the cosmetic $\frac{1}{2}$ factor in their squared error definitions (and we multiply Gluon's `L2Loss`, which includes it, by $2$), so these losses are twice our scratch values and level off at $\sigma^2 = 10^{-4}$ rather than $\sigma^2 / 2$:

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

With `LinearRegression` defined, training reduces to four lines: instantiate the model, instantiate `d2l.SyntheticRegressionData`, create a `d2l.Trainer(max_epochs=10)`, and call `trainer.fit(model, data)`:

```{.python .input #linear-regression-concise-training-1}
%%tab pytorch
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(lr=0.015)
model(data.X[:1])  # Creates the parameters of the lazy layer
model.net.apply(model.weight_init)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab mxnet, tensorflow
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(lr=0.015)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-1}
%%tab jax
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
model = LinearRegression(2, lr=0.015)
trainer = d2l.Trainer(max_epochs=10)
trainer.board.yscale = 'log'
trainer.fit(model, data)
```

```{.python .input #linear-regression-concise-training-4}
print(f"validation loss: {trainer.history['val_loss'][-1]:.2e}")
```

The final validation loss lands within roughly $15\%$ of the theoretical noise
floor $\sigma^2 = 10^{-4}$.

The validation and training loss curves drop right down to the $10^{-4}$ noise floor, mirroring our from-scratch implementation. Let us add a small helper method `get_w_b` to extract the learned weight tensor and bias from the framework layer and compare them against $(\mathbf{w}^*, b^*)$:

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

Once again, the errors in $\hat{\mathbf{w}}$ and $\hat{b}$ stay within about a
thousandth, matching our scratch run because the dominant error sources (label
noise across $1{,}000$ examples and the remaining distance after $10$ epochs) are
identical regardless of who wrote the layer.

Everything we checked in this section leaned on knowing the ground-truth
parameters, a luxury we never have on real datasets; in
:numref:`sec_generalization_basics`, we tackle how to judge models strictly from
held-out data. First, however, :numref:`sec_oo-design` opens up the `Module`,
`DataModule`, and `Trainer` classes we just used, which package the training
machinery we built by hand so that later chapters never have to rewrite this
boilerplate.

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
