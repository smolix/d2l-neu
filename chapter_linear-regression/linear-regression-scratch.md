```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Linear Regression Implementation from Scratch
:label:`sec_linear_scratch`

Now that we have the theory of linear regression (:numref:`sec_linear_regression`) and a synthetic dataset with known ground-truth parameters (:numref:`sec_synthetic-regression-data`), we are ready to put both into code. In this section, we build every component of the training pipeline from scratch—parameter initialization, the linear forward pass, the squared loss, automatic differentiation, the minibatch iterator, and the SGD update loop—relying only on basic tensor arithmetic and `autograd`. Writing each step by hand strips away all magic so you can see exactly how gradients flow and how hyperparameters like the learning rate shape convergence. Once every moving part is transparent, :numref:`sec_oo-design` will organize these components into reusable object-oriented classes (`Module`, `DataModule`, and `Trainer`), and :numref:`sec_linear_concise` will re-express the exact same model in a handful of lines using high-level framework layers, built-in loss functions, and optimizers.

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

Let us start by generating $2{,}000$ synthetic examples—$1{,}000$ for training and $1{,}000$ for validation—with true parameters $\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, and noise $\sigma = 0.01$:

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

Everything we checked in this section leaned on knowing the ground-truth
parameters, a luxury we never have on real datasets; in
:numref:`sec_generalization_basics`, we tackle how to judge models strictly from
held-out data. First, however, :numref:`sec_oo-design` packages the training
machinery we just built into reusable object-oriented classes (`Module`,
`DataModule`, and `Trainer`), and :numref:`sec_linear_concise` shows how to swap
our hand-written tensors, loss, and updater for ready-made framework components.

## Summary

- **From-scratch implementation:** A complete deep learning training loop requires only four primitives—a parameterized forward pass ($\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$), a scalar loss ($\frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$), reverse-mode automatic differentiation, and a minibatch SGD parameter update.
- **Verification via the noise floor:** On synthetic data with additive noise $\mathcal{N}(0, \sigma^2)$, the expected loss at the true parameters is $\frac{1}{2}\sigma^2$ ($0.00005$ for $\sigma = 0.01$), providing an exact quantitative test that optimization has converged and recovered $(\mathbf{w}^*, b^*)$ within $O(\sigma / \sqrt{n})$.
- **Learning-rate sensitivity:** Too small a learning rate crawls slowly; a well-chosen range converges exponentially to the noise floor; too large a learning rate ($\eta > 2 / \mu_{\max}$) overshoots the quadratic bowl and diverges.

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

:begin_tab:`mxnet`
[Discussions](https://d2l.discourse.group/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://d2l.discourse.group/t/201)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/17976)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §2.3]{.kicker}

Linear Regression Implementation from Scratch<br>
**tensor parameters · broadcasting guards · autograd verification · minibatch SGD loop · convergence floor**
:::
:::

::: {.slide title="Why a Falling Loss Curve Never Proves Correctness"}
[The Systems Debugging Hazard]{.kicker}

In deep learning code, many severe bugs raise no runtime exceptions and still produce a steadily decreasing training loss:

- **Silent broadcasting bug:** Subtracting a 1D label tensor `y` of shape `(32,)` from a 2D prediction tensor `y_hat` of shape `(32, 1)` broadcasts into a `(32, 32)` matrix of all $1{,}024$ pairwise differences!
- **Deceptive loss curve:** Minimizing the mean of that $32 \times 32$ matrix pulls every prediction toward the batch average $\bar{y}$, causing the reported loss to drop smoothly while $\mathbf{w}$ converges toward $\mathbf{0}$ instead of $\mathbf{w}^* = (2, -3.4)^\top$.
- **Quantitative ground-truth test:** By training on our synthetic dataset ($\mathbf{w}^* = (2, -3.4)^\top$, $b^* = 4.2$, $\sigma = 0.01$), we demand both that the validation loss hit $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$ and that $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_\infty \lesssim 10^{-3}$.
:::

::: {.slide title="Initializing Parameters With Gradient Tracking"}
[From-Scratch Setup]{.kicker}

We allocate a weight column vector $\mathbf{w} \in \mathbb{R}^{2 \times 1}$ drawn from $\mathcal{N}(0, 0.01^2)$ and a scalar bias $b = 0 \in \mathbb{R}^1$, enabling automatic differentiation on both tensors:

@linear-regression-scratch-the-parameters

- **Framework gradient hooks:** PyTorch sets `requires_grad=True`, MXNet calls `attach_grad()`, TensorFlow wraps tensors in `tf.Variable`, and JAX differentiates pure functions of immutable arrays directly via `jax.grad`.
- **Why random initialization:** For convex linear regression, starting at $\mathbf{w} = \mathbf{0}$ converges identically, but small random weights will become essential in Chapter 4 to break symmetry across hidden units.
:::

::: {.slide title="Forward Pass and the Defensive Reshape in Squared Loss"}
[Model & Loss Primitives]{.kicker}

On a minibatch $\mathbf{X} \in \mathbb{R}^{32 \times 2}$, the forward pass is a single $(32 \times 2)(2 \times 1)$ matrix-vector product plus a broadcasted scalar bias:

@linear-regression-scratch-the-forward-pass

@linear-regression-scratch-the-loss

::: {.d2l-note .warn}
**Defensive shape alignment:** `squared_loss` explicitly calls `d2l.reshape(y, (-1, 1))` (or `y.reshape(y_hat.shape)`) before subtracting, guaranteeing an elementwise `(32, 1)` residual vector even if a caller passes a 1D label vector of shape `(32,)`.
:::
:::

::: {.slide title="Verifying Autograd Against the Analytical Gradient"}
[Gradient Check]{.kicker}

For residual vector $\mathbf{r} = \mathbf{X}\mathbf{w} + b - \mathbf{y} \in \mathbb{R}^{|\mathcal{B}| \times 1}$, differentiating $L_{\mathcal{B}} = \frac{1}{2|\mathcal{B}|}\|\mathbf{r}\|_2^2$ by hand yields $\nabla_{\mathbf{w}} L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top\mathbf{r}$ and $\partial_b L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{1}^\top\mathbf{r}$:

@linear-regression-scratch-the-gradient-2

::: {.d2l-note .rule}
Comparing reverse-mode `autograd` against the analytical formula on a 32-example minibatch confirms agreement to within `float32` machine epsilon ($\sim 10^{-7}$).
:::
:::

::: {.slide title="Why Reshuffling Every Epoch Is Essential for SGD"}
[Stochastic Sampling Invariant]{.kicker}

Why must our minibatch iterator draw a fresh random permutation of $\{0, \ldots, n-1\}$ at the start of every epoch rather than slicing examples in a fixed order?

| Minibatch Selection ($|\mathcal{B}| = 32$, at $\mathbf{w}=\mathbf{0}, b=0$) | Batch Mean Label $\bar{y}_{\mathcal{B}}$ | Bias Gradient $\partial_b L_{\mathcal{B}} = -\bar{y}_{\mathcal{B}}$ | Resulting Parameter Step on $b$ |
|---|---|---|---|
| **Uniform random minibatch** | $4.2 \pm 0.7$ | $\approx -4.2$ | Steps upward toward $b^* = 4.2$ |
| **Sorted order (32 smallest labels)** | $\approx -4.6$ | $\approx +4.6$ | Steps *downward*, away from $b^* = 4.2$! |

- **Unbiasedness requires random sampling:** A uniformly sampled minibatch satisfies $\mathbb{E}[\nabla L_{\mathcal{B}}] = \nabla L$. Iterating over examples sorted by label or time feeds systematically biased gradients for dozens of consecutive steps, causing the parameters to cycle wildly across each epoch.
:::

::: {.slide title="The Minibatch Generator and One-Pass Exhaustion"}
[Data Iteration Pitfall]{.kicker}

Our `minibatches` function permutes indices with `np.random.permutation(len(X))` and `yield`s slices of size $32$ ($\lceil 1000 / 32 \rceil = 31$ batches of $32$ plus $1$ remainder batch of $8$):

@linear-regression-scratch-minibatches-1

@synthetic-regression-data-reading-the-data-in-minibatches-1

::: {.d2l-note .warn}
**Python generator exhaustion:** A generator object (`loader = minibatches(...)`) runs out after a single pass and silently yields $0$ batches on a second `for` loop! The training loop must call `minibatches(X_train, y_train, batch_size)` fresh inside every epoch (or use a reusable `DataLoader`, §2.4).
:::
:::

::: {.slide title="The Four-Step Update and Dozen-Line Training Loop"}
[Complete From-Scratch Loop]{.kicker}

Every minibatch update executes four operations in strict sequence: (1) forward pass and loss, (2) backward pass, (3) unrecorded parameter step, and (4) clearing stored gradients:

@linear-regression-scratch-the-training-loop

::: {.d2l-note .rule}
**Two critical autograd guards:** In imperative frameworks (PyTorch), parameter updates must run inside `torch.no_grad()` so the update itself is not appended to the computation graph, and `param.grad.zero_()` must clear accumulated gradients before the next minibatch backward pass.
:::
:::

::: {.slide title="Predicting the Convergence Rate Before Running Code"}
[Quantitative Theory of the Loss Curve]{.kicker}

Because our features are independent standard normals ($\mathbb{E}[\mathbf{x}\mathbf{x}^\top] = \mathbf{I}_2$, $\mathbb{E}[\mathbf{x}] = \mathbf{0}$) independent of $\epsilon \sim \mathcal{N}(0, \sigma^2)$, the expected population loss has an exact analytical form:

$$\mathbb{E}\bigl[L(\mathbf{w}, b)\bigr] = \frac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|_2^2 + \frac{1}{2}(b - b^*)^2 + \frac{1}{2}\sigma^2$$

- **Initial excess loss:** Starting near $(\mathbf{w}, b) \approx (\mathbf{0}, 0)$, the initial loss is $\frac{1}{2}(2^2 + (-3.4)^2 + 4.2^2) + 5 \times 10^{-5} \approx 16.6$, more than five orders of magnitude above the noise floor $\frac{1}{2}\sigma^2 = 5 \times 10^{-5}$.
- **Per-step contraction factor:** The expected gradient is $(\mathbf{w} - \mathbf{w}^*, b - b^*)$, so each step of size $\eta = 0.03$ multiplies the parameter error by $(1 - \eta) = 0.97$ and the excess loss by $(1 - \eta)^2 = 0.97^2$.
- **Per-epoch contraction ($32$ steps):** One epoch of $32$ minibatch steps shrinks the excess loss by $0.97^{64} \approx 0.142$—a drop of roughly **$7\times$ per epoch**!
:::

::: {.slide title="Training Run: Exponential Decay to the Noise Floor"}
[Empirical Verification]{.kicker}

Running `train(lr=0.03)` for 10 epochs confirms both the $7\times$-per-epoch exponential slope and the plateau at the theoretical noise floor:

::: {.cols .vc}
::: {.col}
@!linear-regression-scratch-the-training-run-1
:::

::: {.col .fig}
@!linear-regression-scratch-training-3
:::
:::

::: {.d2l-note}
On a log-scale plot, exponential convergence $(1 - \eta)^{64k}$ appears as a straight downward line for the first 7 epochs until the excess loss drops below the irreducible noise floor $\sigma^2/2 = 5 \times 10^{-5}$.
:::
:::

::: {.slide title="Verifying Parameter Recovery Toward (2, −3.4) and 4.2"}
[Parameter Accuracy Check]{.kicker}

Inspecting $\hat{\mathbf{w}} - \mathbf{w}^*$ and $\hat{b} - b^*$ after 10 epochs confirms recovery within a few times $10^{-4}$ ($\approx \sigma/\sqrt{n}$):

@!linear-regression-scratch-training-4

- **Statistical vs. numerical error:** Even exact least squares (`lstsq`) deviates from $(\mathbf{w}^*, b^*)$ by $\sigma / \sqrt{n} = 0.01 / \sqrt{1000} \approx 3.2 \times 10^{-4}$.
- **Why loss alone is blind to $10^{-3}$ parameter drift:** A parameter error of $5 \times 10^{-4}$ raises the validation loss by only $\frac{1}{2} \cdot 3 \cdot (5 \times 10^{-4})^2 \approx 4 \times 10^{-7}$—less than $1\%$ of the $5 \times 10^{-5}$ noise floor!

::: {.d2l-note .rule}
Comparing recovered parameters against ground truth catches subtle gradient and broadcasting bugs that a loss plot hides.
:::
:::

::: {.slide title="Learning-Rate Stability: Crawling vs. Divergence"}
[Step-Size Limits]{.kicker}

Sweeping $\eta \in \{0.003, 0.03, 0.3, 3\}$ over two orders of magnitude maps out the stability boundary of minibatch SGD:

@linear-regression-scratch-the-learning-rate

- **Too small ($\eta = 0.003$):** Parameter error contracts by only $0.997$ per step, leaving the validation loss around $2.4$ after 10 epochs ($16.6 \times 0.997^{640} \approx 2.4$).
- **Well tuned ($\eta = 0.03$ and $\eta = 0.3$):** Loss plunges exponentially to the $5 \times 10^{-5}$ noise floor within $1$ to $7$ epochs.
- **Unstable ($\eta = 3 > 2 / \mu_{\max}$):** Because each step multiplies error along eigendirection $j$ of the Hessian $\mathbf{H} \approx \mathbf{I}$ by $(1 - \eta \mu_j)$, at $\eta = 3$ the error multiplies by $1 - 3 = -2$ per step, oscillating with doubling amplitude until overflow (`nan`).
:::

::: {.slide title="Concept Check: Gradients, Loss Scaling & Stability"}
[Check Your Understanding]{.kicker}

Test your understanding of training loop mechanics on these three scenarios:

1. **Forgotten `zero_grad()`:** In PyTorch, if you omit `param.grad.zero_()` inside the training loop, what vector does step $t$ subtract from `param`, and how do the iterates behave on a quadratic bowl?
2. **Summed loss vs. mean loss:** Suppose a loss function computes the *sum* $\sum_{i \in \mathcal{B}} (\hat{y}^{(i)} - y^{(i)})^2$ over a minibatch of $|\mathcal{B}| = 32$ without dividing by $2|\mathcal{B}|$. Which learning rate $\eta$ reproduces our scratch step at $\eta_{\textrm{scratch}} = 0.03$?
3. **Rescaled feature stability:** If you multiply feature $x_1$ by $10$ (so its variance becomes $100$ instead of $1$), what happens to the maximum stable learning rate $\eta_{\max}$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Without clearing, step $t$ subtracts $\eta \sum_{s=1}^t \mathbf{g}_s$ (the running sum of all past gradients!), acting like undiscounted momentum that overshoots the minimum and oscillates without converging.
2. The summed loss is $2|\mathcal{B}| = 64$ times larger than our scratch loss, so its gradient is $64\mathbf{g}$; we must divide the learning rate by $64$ to $\eta = 0.03 / 64 \approx 4.69 \times 10^{-4}$.
3. Scaling $x_1$ by $10$ increases the largest eigenvalue of the Hessian $\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$ by $10^2 = 100$, shrinking the stability ceiling $\eta_{\max} = 2 / \mu_{\max}$ by $100\times$ (from $\approx 2$ down to $\approx 0.02$).
:::
:::

::: {.slide title="Section 2.3 Summary & Key Takeaways"}
[Summary]{.kicker}

| Training Stage | Mathematical Operation | Systems / Implementation Guard | Quantitative Check ($\sigma = 0.01, n = 1000$) |
|---|---|---|---|
| **1. Forward & Loss** | $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$, $L_{\mathcal{B}} = \frac{1}{2|\mathcal{B}|}\|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$ | Reshape `y` to `(-1, 1)` to prevent `(B,) - (B, 1) -> (B, B)` broadcasting | Initial loss $\mathbb{E}[L(\mathbf{0}, 0)] \approx 16.6$ |
| **2. Backward Pass** | $\nabla_{\mathbf{w}} L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y})$ | Verify `autograd` against hand formula on 1 batch | Max discrepancy $\lesssim 10^{-7}$ (`float32` eps) |
| **3. SGD Update** | $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \nabla L_{\mathcal{B}}$ | Run update outside autograd tape (`no_grad`) and clear `.grad` | Stable for $0 < \eta < 2/\mu_{\max}$; $\eta=0.03$ drops loss $7\times$/epoch |
| **4. Epoch Reshuffle** | Uniform permutation of $\{0, \ldots, n-1\}$ | Re-create generator each epoch so it does not exhaust after epoch 1 | Loss plateaus at $\sigma^2/2 = 5 \times 10^{-5}$; $\|\hat{\mathbf{w}} - \mathbf{w}^*\|_\infty \lesssim 10^{-3}$ |

::: {.d2l-note}
**Up next (§2.4 Object-Oriented Design & §2.5 Concise Implementation):** We first package this bare loop into reusable `Module`, `DataModule`, and `Trainer` classes (`LinearRegressionScratch`, §2.4), and then swap in high-level framework layers, MSE losses, and optimizers (`LinearRegression`, §2.5).
:::
:::
