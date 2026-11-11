```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

# Implementation from Scratch
:label:`sec_linear_scratch`

So far linear regression exists on paper, next to a dataset whose true
parameters we know (:numref:`sec_synthetic-regression-data`) and a threshold of about $10^{-3}$ that a correct fit
must meet. We now turn the derivation into a program. The trouble
with training programs is that they can be wrong without failing.
Subtracting a label vector of shape $(32,)$ from predictions of shape
$(32, 1)$, for example, raises no error, because broadcasting forms the
$32 \times 32$ matrix of all pairwise differences. Training on the loss
computed from it lowers that loss steadily while learning the wrong weights.
Nothing on the screen looks amiss. Such bugs show only when we compare the
result with what a correct program must produce.

We therefore write the four components of training ourselves, using nothing
but tensors and automatic differentiation: the linear model, the squared
loss, minibatch stochastic gradient descent, and the training loop that runs
them. Writing each component explicitly shows what it computes and what it
passes to the next one. Every part of the program is a plain function, and
the training loop is a dozen lines. Since we train on data whose origin we
know, we can predict what a correct implementation must achieve and then
check it: the loss must fall to a floor set by the noise, and the learned
parameters must come within the threshold of the true ones.

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

We draw the data as in :numref:`sec_synthetic-regression-data`, with
`d2l.synthetic_data`: 2,000 examples, of which the first 1,000 are for
training and the other 1,000 for validation.

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

The model has two parameters, the weight vector $\mathbf{w} \in \mathbb{R}^2$
and the bias $b$. We initialize the weights with random numbers drawn from a
normal distribution with mean 0 and standard deviation 0.01, and the bias
with 0. For linear regression the starting point matters little: the loss is
a quadratic function of the parameters with a single minimum, which gradient
descent reaches from any start (exercise
:exref:`ex_linear-regression-scratch-initialization`). For deep networks the
scale of the initial weights matters a great deal, which is why the standard
deviation is an argument, `sigma`. (It has nothing to do with the noise
level $\sigma$ of the data. That both are 0.01 is a coincidence.) We will
start from fresh parameters several times, so `init_params` creates them.

:begin_tab:`pytorch`
`requires_grad=True` asks PyTorch to record the operations on a tensor, as
in :numref:`sec_autograd`, so that `backward` can store the gradient in its
`grad` attribute.
:end_tab:

:begin_tab:`mxnet`
`attach_grad` allocates memory for the gradient of an array, so that the
operations recorded inside `autograd.record()` can be differentiated with
respect to it (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`tensorflow`
A `tf.Variable` is a tensor that can be changed in place, and a
`GradientTape` records the operations on it, so that their gradients can be
computed (:numref:`sec_autograd`).
:end_tab:

:begin_tab:`jax`
JAX arrays cannot be changed, and they record nothing. Instead, `jax.grad`
differentiates a function with respect to its arguments, so we keep `w` and
`b` as plain arrays and pass them to every function that uses them. The
random weights need a key, which `d2l.get_key()` supplies.
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

The forward pass computes the predictions for a minibatch,
$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b$. For a minibatch of 32
examples, $\mathbf{X}$ is the $32 \times 2$ matrix whose rows are the
examples' features. Its product with the weights, stored as a $2 \times 1$
matrix, is a $32 \times 1$ column of weighted sums, and broadcasting adds the
scalar $b$ to each of its entries (:numref:`subsec_broadcasting`).

```{.python .input #linear-regression-scratch-the-forward-pass}
def linreg(X, w, b):
    """The linear regression model."""
    return d2l.matmul(X, w) + b
```

### The Loss

The loss of one example is half its squared error,
$\frac{1}{2}(\hat{y} - y)^2$, as in :numref:`sec_linear_regression`, and the
loss of a minibatch is the mean over its examples. Before subtracting, the
loss reshapes the labels into a column of shape $(|\mathcal{B}|, 1)$, the
shape of the predictions. Our labels already have that shape, so the
reshape is a precaution. It costs nothing, and it rules out the bug of the
opening. What that bug does to training is worked out in exercise
:exref:`ex_linear-regression-scratch-reshape-loss`.

```{.python .input #linear-regression-scratch-the-loss}
def squared_loss(y_hat, y):
    """Half the squared error, averaged over the minibatch."""
    return d2l.reduce_mean((y_hat - d2l.reshape(y, (-1, 1))) ** 2 / 2)
```

### The Gradient

Before we hand the loss to automatic differentiation, we write down the
gradient that it must return. Otherwise we would have nothing to check the
result against. :numref:`subsec_linear-regression-minibatch-sgd` derived it
with the chain rule: for one example with error $r = \hat{y} - y$,

$$\frac{\partial l}{\partial \mathbf{w}} = (\hat{y} - y)\, \mathbf{x} \qquad \textrm{and} \qquad \frac{\partial l}{\partial b} = \hat{y} - y.$$
:eqlabel:`eq_linreg_grad_by_hand`

In words, *the gradient is the error-weighted input*: each weight receives
the error times its feature, and the bias the error itself. For a minibatch
$\mathcal{B}$, as in the forward pass, $\mathbf{X}$ and $\mathbf{y}$ hold
the features and labels of its examples, and we collect their errors in the
vector $\mathbf{r} = \mathbf{X}\mathbf{w} + b - \mathbf{y}$. The
*minibatch loss* $L_{\mathcal{B}}$ is the average of the per-example losses
over $\mathcal{B}$, so its gradient is the average of the per-example
gradients, and these averages become matrix products,

$$\nabla_{\mathbf{w}} L_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\mathbf{X}^\top \mathbf{r} \qquad \textrm{and} \qquad \frac{\partial L_{\mathcal{B}}}{\partial b} = \frac{1}{|\mathcal{B}|}\sum_{i \in \mathcal{B}} r^{(i)}.$$

One matrix--vector product computes the weight gradient of the whole
minibatch. It is the gradient $2\mathbf{X}^\top\mathbf{r}$ of the line
model in :numref:`sec_calculus`, divided by $2|\mathcal{B}|$, because our
loss is half the mean of the squared errors instead of their sum.

Does automatic differentiation return these gradients? To find out, we take
a minibatch of 32 training examples and fresh parameters.

```{.python .input #linear-regression-scratch-the-gradient-1}
Xb, yb = X_train[:32], y_train[:32]
w, b = init_params()
```

We compute the gradient with respect to $(\mathbf{w}, b)$ once by automatic
differentiation and once with the formulas above, and print the gradient
and the largest difference between the two.

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

The largest difference is zero or at most about $10^{-6}$: the two
computations agree up to the rounding error of float32 arithmetic. This is
the gradient that the backward pass computes in every training step.

## Training

### The Update

Strictly speaking, we need no optimizer at all: linear regression can be
solved in closed form, with the normal equations of
:numref:`sec_linear_regression`. We train it with minibatch stochastic
gradient descent nevertheless, because that is how every later model in the
book is trained, and because here, for once, its result can be checked
against known parameters. Each step draws a minibatch $\mathcal{B}$ and
moves the parameters against the gradient of its loss,

$$(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta\, \nabla_{(\mathbf{w}, b)} L_{\mathcal{B}},$$

where the learning rate $\eta > 0$ sets the size of the step. Since the loss
is a mean over the minibatch, $\eta$ needs no change when the minibatch size
changes (:numref:`subsec_linear-regression-minibatch-sgd`).

One step of minibatch SGD consists of four operations, which the training
loop below performs in this order:

1. the *forward pass* computes the predictions and the loss of the
   minibatch, while the framework records the operations;
1. the *backward pass* computes the gradient of the loss from that record;
1. the *update* moves the parameters against the gradient, without being
   recorded;
1. any gradient that the framework keeps is cleared.

These four operations are the whole algorithm. Two rules fix their order.
The gradient used in an update must belong to the current minibatch alone,
so a gradient that the framework keeps must be cleared before the next
backward pass. And the update must not be recorded: it is not part of the
loss, and recording it would make the next backward pass differentiate
through every earlier update.

### Minibatches

The minibatches must be drawn at random, or their average gradient is no
longer an unbiased estimate of the full gradient
(:numref:`subsec_linear-regression-minibatch-sgd`); minibatches taken in a
fixed order, say sorted by label, would bias every step. The usual way is to
shuffle the training examples at the start of every epoch, one pass through
the training data, and to cut the shuffled sequence into consecutive pieces
of `batch_size` examples. Every example is then used once per epoch, and
the minibatches differ from one epoch to the next.

The function `minibatches` does this. Because its body contains `yield`,
calling it returns a generator, an object that runs the body piece by piece
as a loop iterates over it: each `yield` hands the loop one minibatch, a
pair of features and labels.

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

With 1,000 training examples and minibatches of 32, an epoch consists of
$\lceil 1000/32 \rceil = 32$ minibatches: 31 full ones and a last one with
the remaining 8 examples. We count the minibatches of one pass over a
generator, and then try a second pass over the same generator.

```{.python .input #synthetic-regression-data-reading-the-data-in-minibatches-1}
loader = minibatches(X_train, y_train, 32)
sizes = [len(y) for X, y in loader]
print(f'{len(sizes)} minibatches, the last of size {sizes[-1]}')
print(f'second pass: {len([y for X, y in loader])} minibatches')
```

The count matches. The second pass, however, yields nothing: a generator
can be iterated only once, and after one pass it is exhausted, without an
error. The training loop below therefore calls `minibatches` anew in every
epoch.

### The Training Loop

The training loop repeats the step on every minibatch of an epoch, epoch
after epoch. After each epoch it measures the loss on the 1,000 validation
examples, on which the model never trains. `train` starts from fresh
parameters, trains with learning rate `lr`, and returns the trained
parameters and the validation loss after every epoch. The comments mark the
operations of a step; the other lines run the two loops and record the
validation loss.

:begin_tab:`pytorch`
PyTorch adds the gradient of each backward pass to what `.grad` already
holds (:numref:`sec_autograd`), so `train` zeroes `.grad` after each update.
The update runs inside `torch.no_grad()`, since PyTorch refuses to change a
tensor that requires gradients in place while it records.
:end_tab:

:begin_tab:`mxnet`
MXNet records only inside `autograd.record()`, so the update after that
block is not recorded. `backward` overwrites the stored gradient, and
nothing has to be cleared.
:end_tab:

:begin_tab:`tensorflow`
A `GradientTape` records only inside its `with` block, so the update with
`assign_sub` after the block is not recorded. The tape returns the gradients
as new tensors without storing them, and nothing has to be cleared.
:end_tab:

:begin_tab:`jax`
`jax.grad` returns the gradients as new arrays and stores nothing, so
nothing has to be cleared, and the update computes new parameter arrays
from the old ones. `sgd_step` holds the gradient and the update, and
`jax.jit` compiles it once for each shape of minibatch.
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

It is tempting to run the training loop now and see what comes out. But
then we could not tell a correct loss curve from a merely plausible one. So
before training, we predict what a correct implementation must produce. The
label of a new example is $y = \mathbf{x}^\top \mathbf{w}^* + b^* + \epsilon$,
so the error of parameters $(\mathbf{w}, b)$ on it is
$r = \mathbf{x}^\top(\mathbf{w} - \mathbf{w}^*) + (b - b^*) - \epsilon$. The
two features of the synthetic data are independent standard normal
variables, and the noise $\epsilon$ is independent of them. All three have
mean zero, each feature has variance 1, and the product of any two of them
averages to zero. Squaring $r$ and taking the expectation therefore leaves
only the squared terms,

$$\mathbb{E}\left[\tfrac{1}{2} r^2\right] = \tfrac{1}{2}\|\mathbf{w} - \mathbf{w}^*\|^2 + \tfrac{1}{2}(b - b^*)^2 + \tfrac{1}{2}\sigma^2.$$
:eqlabel:`eq_linreg-scratch-expected-loss`

The expected loss is half the squared distance between the parameters and
the true ones, plus a floor of $\sigma^2/2 = 5 \times 10^{-5}$. No choice of
parameters gets below the floor, since even the true parameters leave the noise
$\epsilon$ unexplained. The validation loss, an average over 1,000 examples
that the model does not train on, estimates the expected loss.

The equation also predicts how fast training approaches the floor. We call
$(\mathbf{w} - \mathbf{w}^*, b - b^*)$ the *parameter error*; for the
least-squares solution it is the estimation error of
:numref:`subsec_synthetic-regression-data-recovery`. The gradient of the
expected loss with respect to $(\mathbf{w}, b)$ is the parameter error itself,
so a step along this gradient with learning rate $\eta$ multiplies the
parameter error by $1 - \eta$, and the *excess loss* above the floor, half
its square, by $(1 - \eta)^2$. The minibatch gradient equals this gradient
only on average, so the rule gives an estimate, not an exact value. At
$\eta = 0.03$ an epoch of 32 steps should therefore lower the excess loss by
a factor of about 7. Training starts with weights close to zero and a zero
bias, so the excess loss starts at half the squared length of
$(\mathbf{w}^*, b^*)$, about 16.6, and the rule predicts it for every epoch
from there. We compare the prediction with the measured loss, epoch by
epoch, once the model is trained.

### The Training Run

We train for ten epochs with learning rate 0.03. The loss falls by more than
five orders of magnitude, so we plot the validation loss after each epoch on
a logarithmic axis, on which a constant factor per epoch is a straight line.

```{.python .input #linear-regression-scratch-training-3  n=20}
w, b, val_losses = train(lr=0.03)
d2l.plot(list(range(1, 11)), val_losses, xlabel='epoch',
         ylabel='validation loss', yscale='log')
```

On the plot the loss falls along a straight line, by a factor of about 7
per epoch, and levels off near $5 \times 10^{-5}$ after seven or eight
epochs. To compare the rate in numbers, we print for the first seven epochs
the validation loss above the floor next to the predicted excess loss.

```{.python .input #linear-regression-scratch-the-training-run-1}
floor = noise ** 2 / 2                              # sigma^2 / 2
excess = (float(d2l.reduce_sum(w_star ** 2)) + b_star ** 2) / 2  # 16.6
for epoch, loss in enumerate(val_losses[:7], start=1):
    excess *= (1 - 0.03) ** (2 * 32)                # 32 steps per epoch
    print(f'epoch {epoch}: measured {loss - floor:.1e}, '
          f'predicted {excess:.1e}')
```

Measured and predicted excess agree closely in the first epochs and drift
apart slowly, by a factor of up to about 3 or 4 at the seventh. The
prediction takes the matrix $\mathbf{H}$ of exercise
:exref:`ex_linear-regression-scratch-stability`, the average of
$(\mathbf{x}, 1)(\mathbf{x}, 1)^\top$ over the training set, to be exactly
the identity. For 1,000 examples its eigenvalues lie within about 10% of 1,
and after a few epochs the smallest of them sets the rate, so the run falls
behind the prediction by a little more with every epoch. The prediction
holds, in rate and in floor.

### Recovered Parameters

A loss at the floor is good news, but it is indirect evidence. Because we
generated the data ourselves, we can look at the parameters themselves and
compare them with the true ones. We print the parameter errors
$\hat{\mathbf{w}} - \mathbf{w}^*$ and $\hat{b} - b^*$.

```{.python .input #linear-regression-scratch-training-4  n=21}
w_error = d2l.numpy(d2l.reshape(w, w_star.shape) - w_star)
b_error = d2l.numpy(b - b_star)
print('error in w:', ', '.join(f'{e:.1e}' for e in w_error))
print('error in b:', ', '.join(f'{e:.1e}' for e in b_error))
```

The parameter errors are a few times $10^{-4}$, within the threshold of
$10^{-3}$ that :numref:`subsec_synthetic-regression-data-recovery` set.
Could they be much smaller? No. The noise in the 1,000 training labels
moves even the exact least-squares fit about $3 \times 10^{-4}$ away from
the true value in each coordinate, the standard error $\sigma/\sqrt{n}$
derived there. Minibatch SGD adds two
contributions of similar size. With a constant learning rate its parameters
never settle but keep fluctuating in the noise ball of
:numref:`sec_linear_regression`. And ten epochs leave a remainder of the
initial parameter error: the 320 steps multiply it by
$0.97^{320} \approx 6 \times 10^{-5}$. The bias starts 4.2 below $b^*$, so
about $4.2 \times 0.97^{320} \approx 2.5 \times 10^{-4}$ of that distance
remains.

The loss does not reveal errors of this size. By
:eqref:`eq_linreg-scratch-expected-loss`, a parameter error of
$5 \times 10^{-4}$ in each of the three coordinates raises the expected
loss by $\frac{1}{2} \cdot 3 \cdot (5 \times 10^{-4})^2 \approx 4 \times 10^{-7}$,
less than 1% of the floor and less than the fluctuation of a validation loss
over 1,000 examples. A comparison of parameters detects implementation errors
that a loss curve hides, which is why new code is best tested on synthetic
data whose parameters we know. Such a comparison is possible only when a
single set of parameters minimizes the loss, as here, where no feature is a
combination of the others. Deep networks have many parameter settings that
make the same predictions, so we judge them by their predictions on new
data.

### The Learning Rate

For steps along the gradient of the expected loss, the factor $1 - \eta$
also tells us what other learning rates do. For $0 < \eta < 1$ every step
shrinks the parameter error, the faster the larger $\eta$ is, and a step with
$\eta = 1$ lands on the minimum of the expected loss. For $1 < \eta < 2$ the
steps overshoot the minimum, but the parameter error still shrinks, since
$|1 - \eta| < 1$. For $\eta > 2$ every step multiplies the parameter error by
a number larger than 1 in magnitude, so it grows without bound. Minibatch
noise and the finite training set lower this limit a little (exercise
:exref:`ex_linear-regression-scratch-stability`). We train with four
learning rates, a factor of 10 apart, and print the final validation loss of
each run.

```{.python .input #linear-regression-scratch-the-learning-rate}
for lr in [0.003, 0.03, 0.3, 3]:
    val_loss = train(lr)[2][-1]
    print(f'lr = {lr:<6} validation loss: {val_loss:.1e}')
```

At $\eta = 0.003$ the parameter error shrinks by only $0.997$ per step, and
after ten epochs the loss is still about 2 to 3, as the prediction
$16.6 \times 0.997^{640} \approx 2.4$ says. At $\eta = 0.3$ the loss
reaches the floor within the first epoch. At $\eta = 3$ every step
multiplies the parameter error by about $-2$, and the loss overflows until
it ends as `nan`. Note that the limit of 2 is a property of our data, not of
gradient descent: it holds because every feature has variance 1. A feature
scaled by $s > 1$ multiplies the curvature of the loss along its weight by
$s^2$ and lowers the limit to about $2/s^2$, which is one reason why inputs
are standardized (:numref:`sec_pandas`).

All of these checks relied on knowing the true parameters, a luxury that
real data never provide; :numref:`sec_generalization_basics` asks how to
judge a model by data it has not seen. First, though,
:numref:`sec_oo-design` packages this code into classes, so that later
chapters change only the part under study.

## Summary

Training linear regression takes four components, each a plain function:
parameters with a forward pass, the squared loss, minibatch stochastic
gradient descent, and a training loop of a dozen lines. The gradient of the
squared loss is the error-weighted input, and automatic differentiation
reproduces it up to rounding. A step runs the forward pass and the loss,
the backward pass, and an update that is not recorded, and then clears any
gradient that the framework keeps. On synthetic data the expected loss is
half the squared parameter error plus $\sigma^2/2$, which predicts the
floor of the loss, the rate at which training reaches it, and the learning
rates that diverge. The learned parameters come within a few times
$10^{-4}$ of the true ones, an error that a loss curve cannot show and a
comparison of parameters can.

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
