```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Implementation from Scratch
:label:`sec_linear_scratch`

We now have the mathematics of linear regression on paper, a synthetic dataset
whose true parameters we know (:numref:`sec_synthetic-regression-data`), and a
concrete verification threshold of roughly $10^{-3}$. It is time to turn those
equations into a working program—completely from scratch, relying on nothing but
tensors and automatic differentiation.

Even though modern deep learning frameworks can automate almost all of this
boilerplate (as we will see in :numref:`sec_linear_concise`), implementing the
entire pipeline by hand at least once is the surest way to know what is truly
happening under the hood. It also exposes a notorious pitfall of machine learning
code: *a training program can be completely wrong without ever raising an error*.
Consider a classic example: subtracting a label vector of shape `(32,)` from
predictions of shape `(32, 1)` succeeds silently because broadcasting creates a
$32 \times 32$ matrix of all pairwise differences! Training on the resulting loss
produces a smooth, steadily falling loss curve—while learning completely bogus
weights. Bugs like this reveal themselves only when we compare our program's
output against quantitative predictions of what a *correct* program must achieve.

In this section, we write all four pieces of training as plain Python functions:
(i) the parameterized linear model, (ii) the squared loss, (iii) minibatch
stochastic gradient descent, and (iv) a dozen-line training loop that stitches
them together. Before running our loop, we predict both the exact noise floor the
validation loss must hit and the rate at which it should drop, and then verify
that our learned parameters recover the ground truth within our $10^{-3}$
threshold.

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

We start by generating $2{,}000$ synthetic examples via `d2l.synthetic_data`
(:numref:`sec_synthetic-regression-data`), splitting them evenly into $1{,}000$
training examples and $1{,}000$ held-out validation examples.

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

Before we can optimize our model's parameters—the weight vector
$\mathbf{w} \in \mathbb{R}^2$ and scalar bias $b$—we need to initialize them. We
draw the initial weights from a zero-mean normal distribution with standard
deviation `sigma=0.01` and initialize the bias to zero. For linear regression,
the starting point barely matters because the squared loss is a convex quadratic
with a single global minimum reachable from anywhere (exercise
:exref:`ex_linear-regression-scratch-initialization`). For deep networks, however,
initialization scale is critical, so we expose `sigma` as an argument. (It is
purely a coincidence that both our initial weight scale and the dataset's label
noise $\sigma$ equal `0.01`; they are unrelated quantities.) Wrapping parameter
creation in `init_params` makes it easy to start fresh across multiple
experiments.

:begin_tab:`pytorch`
Passing `requires_grad=True` tells PyTorch to record operations involving `w` and
`b` on the computational graph (:numref:`sec_autograd`), so that `backward()` can
populate their `.grad` attributes.
:end_tab:

:begin_tab:`mxnet`
Calling `attach_grad()` allocates gradient buffers for `w` and `b`, enabling
automatic differentiation for operations executed inside an `autograd.record()`
scope (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`tensorflow`
Wrapping `w` and `b` in `tf.Variable` marks them as mutable trainable parameters
that `tf.GradientTape` watches automatically (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`jax`
Because JAX arrays are immutable and carry no hidden gradient state, `jax.grad`
differentiates pure functions directly with respect to their arguments. We
therefore store `w` and `b` as ordinary JAX arrays, passing them explicitly to
every function that uses them, and supply a fresh key from `d2l.get_key()` for
the random initialization.
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

Next, we implement the linear model $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$
from :eqref:`eq_linreg-y-vec`. Given a minibatch of 32 examples, `X` has shape
$(32, 2)$ and `w` has shape $(2, 1)$, so `d2l.matmul(X, w)` produces a $(32, 1)$
column vector of predictions; adding the scalar bias `b` broadcasts it across
every example (:numref:`subsec_broadcasting`).

```{.python .input #linear-regression-scratch-the-forward-pass}
def linreg(X, w, b):
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

### The Loss

We implement half the mean squared error, averaging $\frac{1}{2}(\hat{y} - y)^2$
across the minibatch (:numref:`sec_linear_regression`). Notice one defensive
detail: before subtracting, we explicitly reshape `y` to `(-1, 1)` to match the
column shape of `y_hat`. Although our synthetic labels already have shape
$(|\mathcal{B}|, 1)$, this zero-cost reshape guarantees that a 1D label vector
can never trigger the silent $(|\mathcal{B}|, |\mathcal{B}|)$ broadcasting bug
described at the start of this section (see exercise
:exref:`ex_linear-regression-scratch-reshape-loss` to see what that bug actually
learns).

```{.python .input #linear-regression-scratch-the-loss}
def squared_loss(y_hat, y):
    """Half the squared error, averaged over the minibatch."""
    return d2l.reduce_mean((y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2)
```

### The Gradient

Before trusting automatic differentiation blindly, let us derive the exact
gradient we expect so we can verify our loss and model against it. From
:numref:`subsec_linear-regression-minibatch-sgd`, the chain rule gives the
per-example gradients for an error $r = \hat{y} - y$:

$$\frac{\partial l}{\partial \mathbf{w}} = (\hat{y} - y)\, \mathbf{x} \qquad \textrm{and} \qquad \frac{\partial l}{\partial b} = \hat{y} - y.$$
:eqlabel:`eq_linreg_grad_by_hand`

Put simply, *the gradient is the error-weighted input*: each weight's gradient is
the prediction error multiplied by its corresponding feature, and the bias
gradient is the error itself. Averaging across a minibatch $\mathcal{B}$ with
feature matrix $\mathbf{X}$, label vector $\mathbf{y}$, and error vector
$\mathbf{r} = \mathbf{X}\mathbf{w} + b - \mathbf{y}$ turns the minibatch loss
$L_{\mathcal{B}}$ gradients into a single matrix--vector product and a sum:

$$\nabla_{\mathbf{w}} L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top \mathbf{r} \qquad \textrm{and} \qquad \frac{\partial L_{\mathcal{B}}}{\partial b} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} r^{(i)}.$$

Notice that this is just the $2\mathbf{X}^\top \mathbf{r}$ gradient from
:numref:`sec_calculus` divided by $2|\mathcal{B}|$, because here we take half the
*mean* of the squared errors rather than their sum.

Let us verify on a 32-example minibatch with freshly initialized parameters that
automatic differentiation matches these closed-form expressions.

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

## Training

### The Update

Although linear regression can be solved directly via the normal equations
(:numref:`sec_linear_regression`), we train it here with minibatch stochastic
gradient descent because SGD is the universal workhorse for every deep network in
this book—and linear regression on synthetic data gives us a rare opportunity to
verify our SGD loop against a known ground truth. At each step, we take a random
minibatch $\mathcal{B}$ and update the parameters opposite the minibatch gradient:

$$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla_{(\mathbf{w}, b)} L_{\mathcal{B}},$$

where $\eta > 0$ is the learning rate. Because $L_{\mathcal{B}}$ averages over the
minibatch, $\eta$ does not need to be rescaled whenever we adjust the batch size
(:numref:`subsec_linear-regression-minibatch-sgd`).

Every single training step consists of four operations executed in strict sequence:

1. **Forward pass:** compute predictions and the minibatch loss while the
   framework records the computational graph;
2. **Backward pass:** invoke automatic differentiation to compute gradients with
   respect to every parameter;
3. **Parameter update:** step each parameter against its gradient *outside* the
   recorded graph;
4. **Clear gradients:** zero out any persistent gradient buffers before the next
   iteration.

Notice why both rules of ordering matter: clearing stale gradients ensures each
update reflects only the current minibatch, and keeping the update out of the
computational graph prevents subsequent backward passes from trying to
differentiate through previous optimization steps.

### Minibatches

For the minibatch gradient to remain an unbiased estimator of the full gradient
(:numref:`subsec_linear-regression-minibatch-sgd`), examples must be visited in
random order; feeding minibatches in a fixed order—say, sorted by label—would
systematically bias every step (exercise
:exref:`ex_linear-regression-scratch-why-reshuffle`). The standard practice is to
reshuffle the entire training set at the start of each *epoch* (one complete pass
through the dataset) and slice the shuffled indices into consecutive chunks of
size `batch_size`, guaranteeing that every example is seen once per epoch in a
fresh grouping.

We implement this with a Python generator function, `minibatches`. Because its
body uses `yield`, calling `minibatches(...)` returns an iterator that lazily
hands back one `(X_batch, y_batch)` tuple on each iteration of a `for` loop.

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

The first pass yields all $32$ minibatches as expected, while the second pass
yields zero! A Python generator is single-use and exhausts silently once
consumed, which is why our training loop calls `minibatches(...)` fresh at the
start of every epoch.

### The Training Loop

We now have every ingredient needed to write our complete training loop. In each
epoch, we iterate over all minibatches produced by `minibatches(X_train, y_train, batch_size)`,
applying the four-step SGD update to every batch; at the end of each epoch, we
evaluate the validation loss on the $1{,}000$ held-out examples that the model
never trains on. The function `train` initializes fresh parameters, runs for
`num_epochs` epochs at learning rate `lr`, and returns `(w, b, val_losses)`.
Inline comments highlight the four steps of SGD; the rest is just bookkeeping.

:begin_tab:`pytorch`
Because PyTorch accumulates gradients in `.grad` across backward passes
(:numref:`sec_autograd`), we explicitly call `param.grad.zero_()` after every
update. We wrap the in-place parameter update inside `with torch.no_grad():`
because PyTorch forbids modifying tracked leaf tensors while gradient recording
is active.
:end_tab:

:begin_tab:`mxnet`
MXNet records operations only inside `with autograd.record():`, so the update
immediately outside that block runs unrecorded, and `backward()` overwrites the
gradient buffer automatically.
:end_tab:

:begin_tab:`tensorflow`
Because `tf.GradientTape` records operations only inside its `with` block, the
in-place `assign_sub` update outside the block is not tracked, and `tape.gradient`
returns freshly created gradient tensors that require no clearing.
:end_tab:

:begin_tab:`jax`
In JAX, `jax.grad` returns fresh gradient arrays without mutating state, so the
update simply computes and returns new `(w, b)` arrays. Wrapping the step in
`sgd_step` with `@jax.jit` compiles it once for each distinct minibatch shape.
:end_tab:

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

## Checking the Implementation

### The Expected Loss

Before hitting "run," let us predict quantitatively what a *correct* training run
must look like—otherwise we would have no way to distinguish a right loss curve
from a merely plausible-looking buggy one. For a fresh example with label
$y = \mathbf{x}^\top \mathbf{w}^* + b^* + \epsilon$, the prediction error of
parameters $(\mathbf{w}, b)$ is
$r = \mathbf{x}^\top(\mathbf{w} - \mathbf{w}^*) + (b - b^*) - \epsilon$.
Because the two features are independent standard normal variables and the noise
$\epsilon$ is independent of both, every cross term has expectation zero while
each squared feature has expectation $1$. Squaring $r$ and taking expectations
therefore yields the clean decomposition

$$\mathbb{E}\left[\tfrac{1}{2} r^2\right] = \tfrac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|^2 + \tfrac{1}{2}(b - b^*)^2 + \tfrac{1}{2}\sigma^2.$$
:eqlabel:`eq_linreg-scratch-expected-loss`

Look at what :eqref:`eq_linreg-scratch-expected-loss` tells us. The expected loss
equals half the squared Euclidean distance from our parameters to the ground
truth, plus an irreducible *noise floor* of $\sigma^2 / 2 = 5 \times 10^{-5}$. No
model can beat that floor on unseen data, because even the exact true parameters
cannot predict the independent noise $\epsilon$. Our validation loss—averaged
over $1{,}000$ unseen examples—directly estimates this expected loss.

Better still, :eqref:`eq_linreg-scratch-expected-loss` predicts the *rate* at
which training should approach that floor! Call
$(\mathbf{w} - \mathbf{w}^*, b - b^*)$ the *parameter error* (for the exact
least-squares fit, this is the estimation error analyzed in
:numref:`subsec_synthetic-regression-data-recovery`). Because the gradient of the
expected loss with respect to $(\mathbf{w}, b)$ is simply the parameter error
itself, taking a gradient step of size $\eta$ shrinks the parameter error by the
factor $1 - \eta$, and thus shrinks the *excess loss* above the floor by
$(1 - \eta)^2$. Minibatch gradients match the expected gradient only on average,
so this gives an approximate trajectory rather than an exact identity. At
$\eta = 0.03$, one epoch of $32$ minibatch updates should reduce the excess loss
by a factor of roughly $1 / (1 - 0.03)^{64} \approx 7$. Starting from weights near
zero and zero bias, the initial excess loss is
$\frac{1}{2}(\|\mathbf{w}^*\|^2 + (b^*)^2) \approx 16.6$, giving us a concrete
prediction for every epoch before we run a single line of training!

### The Training Run

Let us train for $10$ epochs at learning rate $\eta = 0.03$. Because the
validation loss drops across more than five orders of magnitude, we plot it on a
logarithmic $y$-axis, where exponential decay (a constant factor per epoch)
appears as a straight line.

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
slight drift? Our idealized prediction assumed the sample covariance matrix
$\mathbf{H} = \frac{1}{n}\sum_i \tilde{\mathbf{x}}^{(i)}(\tilde{\mathbf{x}}^{(i)})^\top$
(see exercise :exref:`ex_linear-regression-scratch-stability`) equals the exact
identity matrix. On a finite sample of $1{,}000$ examples, its eigenvalues sit
within roughly $10\%$ of $1$, and after several epochs the smallest eigenvalue
sets the asymptotic decay rate—causing the empirical run to lag slightly behind
the idealized prediction. Both the decay rate and the noise floor match theory.

### Recovered Parameters

Reaching the noise floor is encouraging, yet it is still indirect evidence.
Because we generated these data ourselves, we can inspect the learned parameters
directly and compare them with the ground truth $\mathbf{w}^* = (2, -3.4)$ and
$b^* = 4.2$:

```{.python .input #linear-regression-scratch-training-4  n=21}
w_error = d2l.numpy(d2l.reshape(w, w_star.shape) - w_star)
b_error = d2l.numpy(b - b_star)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

All three parameter errors sit at a few times $10^{-4}$—comfortably inside the
$10^{-3}$ verification threshold we derived in
:numref:`subsec_synthetic-regression-data-recovery`. Notice why they cannot get
much smaller than this:

1. Even the exact least-squares fit is shifted by $\sigma / \sqrt{n} \approx 3 \times 10^{-4}$
   due to random noise in the $1{,}000$ training labels;
2. Constant-step-size minibatch SGD fluctuates in a small noise ball around that
   least-squares fit (:numref:`sec_linear_regression`);
3. Ten epochs ($320$ steps) leave a small optimization remainder from the initial
   offset—for the bias, which starts $4.2$ away from $b^*$, roughly
   $4.2 \times 0.97^{320} \approx 2.5 \times 10^{-4}$ still remains.

Notice how sensitive parameter inspection is compared to looking at the loss
alone: by :eqref:`eq_linreg-scratch-expected-loss`, an error of $5 \times 10^{-4}$
in every coordinate increases the expected loss by only
$\frac{1}{2} \cdot 3 \cdot (5 \times 10^{-4})^2 \approx 4 \times 10^{-7}$—less
than $1\%$ of the noise floor, completely buried in the finite-sample fluctuation
of a validation loss. Checking recovered parameters on synthetic data catches
subtle implementation bugs that a loss curve easily hides. Of course, we should
not take exact parameter recovery for granted in general: even in linear
regression it requires linearly independent features, and deep neural networks
possess vast numbers of distinct parameter settings that produce equally accurate
predictions. In machine learning, we ultimately judge models by their predictive
accuracy on unseen data :cite:`Vapnik.1992`.

### The Learning Rate

Our expected-loss contraction factor $1 - \eta$ also predicts what happens as we
sweep the learning rate across orders of magnitude:

* For $0 < \eta < 1$, every step shrinks the parameter error monotonically (faster
  as $\eta$ increases, hitting the minimum of the expected loss in one step at
  $\eta = 1$).
* For $1 < \eta < 2$, steps overshoot and oscillate across the minimum, yet still
  converge because $|1 - \eta| < 1$.
* For $\eta > 2$, $|1 - \eta| > 1$, so every step *amplifies* the error until
  training explodes to infinity! (Finite-sample eigenvalues and minibatch noise
  lower this stability limit slightly; see exercise
  :exref:`ex_linear-regression-scratch-stability`.)

Let us test four learning rates spaced by factors of ten and inspect the final
validation loss of each run.

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

## Summary

* A complete linear regression training pipeline from scratch requires just four
  plain components: parameterized forward pass (`linreg`), squared loss
  (`squared_loss`), minibatch shuffling (`minibatches`), and a dozen-line SGD
  training loop (`train`).
* The gradient of the squared loss is the error-weighted input
  ($\frac{1}{|\mathcal{B}|}\mathbf{X}^\top \mathbf{r}$ for $\mathbf{w}$ and the
  mean error for $b$), which automatic differentiation matches up to `float32`
  rounding.
* Every minibatch SGD step runs four operations in order: recorded forward pass
  and loss, backward pass, unrecorded parameter update, and gradient clearing.
* On our synthetic Gaussian dataset, the expected loss decomposes into
  $\frac{1}{2}\|\mathbf{w}-\mathbf{w}^*\|^2 + \frac{1}{2}(b-b^*)^2 + \frac{1}{2}\sigma^2$,
  predicting the noise floor ($\sigma^2/2 = 5 \times 10^{-5}$), the decay rate per
  epoch, and the learning-rate stability limit. Comparing learned parameters
  directly against $(\mathbf{w}^*, b^*)$ verifies the code to a few times
  $10^{-4}$—catching subtle errors that a loss curve alone would hide.

## Exercises

1. [code] **Initialization at the extremes.** `init_params`
   draws the weights with standard deviation `sigma=0.01`. Predict whether
   training reaches the noise floor when the weights are initialized to
   exactly zero, and when `sigma=1000`. Use
   :eqref:`eq_linreg-scratch-expected-loss` to estimate how many more epochs
   the second case needs at learning rate 0.03, then check your estimate.
   :exlabel:`ex_linear-regression-scratch-initialization`

1. **Shapes in the loss.** Suppose the data loader yielded labels `y` of
   shape $(|\mathcal{B}|,)$ while the predictions `y_hat` have shape
   $(|\mathcal{B}|, 1)$, and `squared_loss` computed `(y_hat - y) ** 2 / 2` without
   reshaping `y`.
    1. Determine whether an error is raised, and find the shape and the
       entries of `y_hat - y`.
    1. Show that this loss is smallest when the model predicts the mean of
       the minibatch labels for every example. Determine approximately the
       weights and the bias that training approaches on the synthetic
       data, and the value at which the loss levels off.
   :exlabel:`ex_linear-regression-scratch-reshape-loss`

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

1. **Uncleared gradients.** ● Suppose the gradient buffers accumulate and
   are never cleared, so that the update at step $t$ subtracts
   $\eta(\mathbf{g}_1 + \cdots + \mathbf{g}_t)$, the sum of all minibatch
   gradients so far. For the one-dimensional loss $\frac{h}{2}(w - c)^2$
   with $h > 0$, whose minimum lies at $w = c$, and with exact gradients,
   derive a recurrence for the error $e_t = w_t - c$, and determine for
   which values of $\eta h$ the error converges to zero. State what this
   predicts for `train` under the same accumulation.

1. **Planted bug.** A faulty variant of `minibatches` shuffles the rows of
   `X` and the entries of `y` with two independent permutations. We collect
   the pairs that it yields in one epoch over the training data, whose
   parameters are $\mathbf{w}^* = (2, -3.4)$ and $b^* = 4.2$ with noise
   standard deviation $0.01$. We then fit $\hat{\mathbf{w}}$ and $\hat{b}$
   to these pairs by least squares.
    1. Predict the approximate values of $\hat{\mathbf{w}}$ and $\hat{b}$,
       and explain your prediction.
    1. Estimate the mean squared error of this fit on the collected pairs,
       and compare it with the smallest mean squared error that any
       predictor can attain on correctly paired data.

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
