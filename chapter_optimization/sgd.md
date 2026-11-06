# Stochastic and Minibatch Gradient Descent
:label:`sec_sgd`

Every model since :numref:`sec_linear_regression` has been trained on
minibatches, most of them with minibatch SGD: at each step we drew a few
dozen or a few hundred examples, computed the gradient of their average
loss, and stepped against it. :numref:`sec_gd` studied step size and
curvature with the exact gradient, which requires a pass over the whole
training set for every step. This section asks what is gained and what is
lost when the gradient is estimated from a few examples instead.

The gain is large. A common recipe trains ResNet-50 on the 1.28 million
images of ImageNet for 90 epochs in minibatches of 256 images
:cite:`Goyal.Dollar.Girshick.ea.2017`, which comes to 450,000 parameter
updates. The same number of updates with exact gradients would take
450,000 passes over the data, 5,000 times the work. Yet a few hundred
randomly chosen images already indicate which way is downhill.
*Stochastic gradient descent* (SGD) uses that observation: it estimates the
gradient from a random sample of examples and steps along the estimate.
Each step is cheap, and the estimate is right on average, but any single
estimate is noisy.

We first ask what the noise does. On a quadratic valley we compute exactly
how far from the minimum the iterates of constant-step SGD keep wandering,
and how that distance depends on the learning rate, the curvature, and the
noise. We then ask what averaging over a minibatch of $b$ examples gains.
Averaging divides the variance of the estimate by $b$, which we measure on a
small network. On parallel hardware, moreover, $b$ examples take far less
time than $b$ separate steps, and we measure that too.

A race of full-batch, single-example, and minibatch updates against the
clock closes the section and builds the training function that the next
section uses for momentum. Lowering the learning rate during training is
the subject of :numref:`sec_scheduler`, and the limits of larger batches
that of :numref:`sec_batch_size`.

```{.python .input #sgd-stochastic-gradient-descent}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #sgd-stochastic-gradient-descent}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #sgd-stochastic-and-minibatch-gradient-descent-1}
%%tab pytorch
import time
```

```{.python .input #sgd-stochastic-and-minibatch-gradient-descent-1}
%%tab jax
import random
import time
```

## Stochastic Gradient Descent

### Sampled Gradients

In deep learning the objective is an average of per-example losses. With
$n$ training examples, a parameter vector $\mathbf{x}$, and the loss
$f_i(\mathbf{x})$ on example $i$, the objective and its gradient are

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}), \qquad
\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

One exact gradient therefore costs $n$ per-example gradients. SGD instead
draws an index $i$ uniformly at random from $\{1, \ldots, n\}$ and steps
along that example's gradient,

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

where $\eta$ is the learning rate. A step now costs one per-example gradient
instead of $n$. The sampled gradient is an *unbiased* estimate of the full
gradient: every index has probability $1/n$, so

$$\mathbb{E}_i \big[\nabla f_i(\mathbf{x})\big] = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

On average the stochastic gradient points where the true gradient points.
A single draw, however, can point almost anywhere, since one example's
gradient pulls toward fitting that example alone. It helps to write the
stochastic gradient as the true gradient plus noise,
$\nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}) + \boldsymbol{\xi}$, where
the noise $\boldsymbol{\xi}$ has mean zero.

### SGD on the Valley

To see what the noise does, we return to the valley of
:numref:`sec_optimization-intro`,

$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2,$$

which curves gently along $x_1$ (second derivative $0.2$) and steeply along
$x_2$ (second derivative $4$), a condition number of $20$. Gradient descent
converges on it, slowly along $x_1$. We simulate sampling noise by adding
independent Gaussian noise of unit variance to both coordinates of the
exact gradient. Exercise :exref:`ex_sgd-noise-dataset` shows that this is
SGD on a dataset whose examples are randomly shifted copies of the
valley.

The update rule `sgd_2d` has the form that `d2l.train_2d` of
:numref:`sec_gd` iterates from the starting point $(-5, -2)$, here with
the gradient `d2l.valley_grad` of the valley `d2l.f_valley`. It is the
`gd_2d` of that section with noise added to the gradient. The noise comes
from a NumPy random generator `rng`, which `train_2d` passes on together
with the learning rate.

```{.python .input #sgd-stochastic-gradient-updates-2}
%%tab pytorch, jax
def sgd_2d(x1, x2, s1, s2, f_grad, eta, rng):
    g1, g2 = f_grad(x1, x2)
    n1, n2 = rng.normal(size=2)  # Unit-variance gradient noise
    return x1 - eta * (g1 + n1), x2 - eta * (g2 + n2), 0, 0
```

We run 300 steps at $\eta = 0.1$.

```{.python .input #sgd-stochastic-gradient-updates-3}
%%tab pytorch, jax
rng = np.random.default_rng()
trace = d2l.train_2d(sgd_2d, d2l.valley_grad, steps=300, eta=0.1, rng=rng)
d2l.show_trace_2d(d2l.f_valley, trace)
```

For the first hundred or so steps the iterate heads for the minimum much as
gradient descent would, slowly along $x_1$. Then it stops making progress
and wanders in a cloud around the minimum, a cloud that is longer along the
flat $x_1$ direction than along the steep $x_2$ direction. Running longer
does not shrink the cloud. Near the minimum the true gradient vanishes but
the noise does not, so every step still moves the parameters by $\eta$
times pure noise.

### The Noise Ball

How far from the minimum does the cloud extend? On a quadratic the
question has an exact answer. Each coordinate of the valley evolves on its
own, so take one coordinate with curvature $\lambda$ ($0.2$ or $4$ here)
and gradient noise $\xi_t$ of mean zero and variance $\sigma^2$ ($1$ here),
drawn independently at every step. One SGD step reads

$$x_{t+1} = x_t - \eta\, (\lambda x_t + \xi_t) = (1 - \eta\lambda)\, x_t - \eta\, \xi_t.$$

The first term pulls the iterate toward the minimum at $0$, shrinking it by
the factor $1 - \eta\lambda$ as gradient descent would. The second term
pushes it in a random direction. To see where the two balance, we square
both sides and take expectations. The cross term vanishes because $\xi_t$
has mean zero and is independent of $x_t$, which leaves

$$\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\, \mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2.$$

Far from the minimum the first term dominates, and SGD makes progress like
gradient descent. Near it, the second moment settles at the value $v$ at
which contraction and injected noise cancel,
$v = (1 - \eta\lambda)^2 v + \eta^2 \sigma^2$. Solving for $v$ gives

$$\mathbb{E}\big[x_\infty^2\big] = \frac{\eta\, \sigma^2}{\lambda\, (2 - \eta\lambda)} \approx \frac{\eta\, \sigma^2}{2\lambda} \quad \textrm{for } \eta\lambda \ll 1.$$
:eqlabel:`eq_sgd-noise-ball`

The iterates fluctuate in a *noise ball* around the minimum. Its squared
radius grows in proportion to the learning rate and to the noise variance,
and it shrinks as the curvature grows. The last dependence explains the
shape of the cloud. At $\eta = 0.1$ the formula gives a variance of
$0.1 / (0.2 \cdot 1.98) \approx 0.25$ along $x_1$ and
$0.1 / (4 \cdot 1.6) \approx 0.016$ along $x_2$, so the cloud has standard
deviation $0.5$ along the flat direction and $0.125$ along the steep one,
a quarter of that. The flat directions, where gradient descent is slowest,
are also where the noise carries the iterate farthest, because the pull
back toward the minimum is weakest there.

We check the formula with 10,000 independent SGD runs, the rows of one
array, all started at the minimum so that the noise alone decides where
they go. After 2,000 steps we measure the variance of each coordinate
across the runs, at $\eta = 0.1$ and at half that rate.

```{.python .input #sgd-the-noise-ball}
%%tab pytorch, jax
def noise_ball(eta, lam, num_runs=10000, num_steps=2000):
    """Variance of SGD iterates on sum(lam * x**2 / 2) with unit noise."""
    rng = np.random.default_rng()
    x = np.zeros((num_runs, len(lam)))
    for _ in range(num_steps):
        x = x - eta * (lam * x + rng.normal(size=x.shape))
    return x.var(axis=0)

lam = np.array(d2l.VALLEY_CURVATURES)  # Curvatures of the valley
for eta in (0.1, 0.05):
    print(f'eta={eta}: measured {noise_ball(eta, lam).round(4)}, '
          f'predicted {(eta / (lam * (2 - eta * lam))).round(4)}')
```

Measured and predicted variances agree to within a few percent, the
sampling error of 10,000 runs. Halving the learning rate halves the
variance along $x_1$, as the approximation in
:eqref:`eq_sgd-noise-ball` says. Along $x_2$ it shrinks by a little more
than half, because there $\eta\lambda = 0.4$ is not small and the exact
formula applies.

A constant learning rate therefore forces a choice. Halving $\eta$ halves
the squared radius of the ball, but it also halves the fraction
$\eta\lambda$ by which each step closes the remaining distance, so reaching
the ball takes twice as many steps. Large steps make fast early progress;
small steps finish close. The way out is to change $\eta$ during training,
large at first and small at the end. :citet:`Robbins.Monro.1951` gave the
classical conditions under which such a decreasing sequence $\eta_t$
converges on problems like this one. The rates must sum to infinity,
$\sum_t \eta_t = \infty$, so that the iterate can still travel arbitrarily
far. Their squares must have a finite sum, $\sum_t \eta_t^2 < \infty$, so
that the noise injected over the whole run, whose variances add up to
$\sigma^2 \sum_t \eta_t^2$, stays bounded. The rate $\eta_t \propto 1/t$
satisfies both. :numref:`sec_scheduler` develops the schedules used for
deep networks, where these conditions are a guide rather than a recipe.

## Minibatches
:label:`sec_minibatch_sgd`

A decaying learning rate shrinks the noise ball by taking smaller steps.
Averaging the gradient over several examples shrinks it by reducing the
noise itself, and every training loop in this book does so. We quantify
the reduction, and we measure why averaging adds so little time on modern
hardware.

### Gradient Variance and Batch Size
:label:`subsec_sgd-variance`

A minibatch $\mathcal{B}_t$ of $b$ examples, drawn independently and
uniformly at random, gives the gradient estimate

$$\mathbf{g}_t = \frac{1}{b} \sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{x}_t).$$

It is unbiased because every term is. Its noise is smaller. In
:numref:`subsec_linear-regression-minibatch-sgd` we met this as the
proportionality of the noise ball to $1 / |\mathcal{B}|$; here we derive the
exact constant and use it throughout the chapter. Let
$\boldsymbol{\Sigma}$ be the covariance of a single example's gradient, so
that its trace
$\operatorname{tr} \boldsymbol{\Sigma} = \mathbb{E}_i \|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$
is the expected squared error of a single-example gradient. The errors of
independent draws have mean zero and are uncorrelated, so the expected
squared norm of their sum is the sum of their expected squared norms,
$b \operatorname{tr} \boldsymbol{\Sigma}$. Dividing the sum by $b$ divides
this by $b^2$:

$$\mathbb{E} \big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$

A minibatch of $b$ examples divides the gradient variance by $b$, and it
does so in every coordinate. In the noise ball :eqref:`eq_sgd-noise-ball`
the variance $\sigma^2$ becomes $\sigma^2 / b$, and the squared radius
becomes about $\eta \sigma^2 / (2\lambda b)$. Halving the learning rate
and doubling the batch size shrink the ball by the same factor, at
different costs: the first doubles the number of steps needed to reach the
ball, the second doubles the work per step.

The derivation holds for any model. To see it on a real one, we need data.
The next experiments use a small regression problem, the self-noise of
airfoils that NASA measured in a wind tunnel. Each example has five
features: the frequency, the angle of attack, the chord length, the
free-stream velocity, and the thickness of the boundary layer. The target
is the sound pressure level. `get_airfoil` standardizes every column to
mean zero and variance one and keeps the first 1,500 examples. It returns
a minibatch iterator that reshuffles the examples in each epoch, together
with the number of features. A training run on these data takes seconds,
so we can repeat it many times.

```{.python .input #minibatch-sgd-reading-the-dataset}
%%tab pytorch
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_airfoil(batch_size=10, n=1500):
    """Load the standardized airfoil self-noise data as minibatches."""
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    dataset = torch.utils.data.TensorDataset(
        *(t.to(d2l.try_gpu()) for t in (data[:n, :-1], data[:n, -1])))
    sampler = torch.utils.data.BatchSampler(  # Whole minibatches of indices
        torch.utils.data.RandomSampler(dataset), batch_size, drop_last=False)
    data_iter = torch.utils.data.DataLoader(dataset, sampler=sampler,
                                            batch_size=None)
    return data_iter, data.shape[1]-1
```

```{.python .input #minibatch-sgd-reading-the-dataset}
%%tab jax
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')

#@save
def get_airfoil(batch_size=10, n=1500):
    """Load the standardized airfoil self-noise data as minibatches."""
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (jnp.array(data[:n, :-1]), jnp.array(data[:n, -1])),
        batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

:begin_tab:`pytorch`
The data live on the GPU if there is one. A `DataLoader` given a batch
size would fetch the examples of a minibatch one at a time and stack them,
which on these data takes longer than the update itself. The
`BatchSampler` instead draws the indices of a whole minibatch, and the
dataset gathers them in a single indexing operation.
:end_tab:

:begin_tab:`jax`
`d2l.load_array` slices the arrays one whole minibatch at a time and places
each minibatch on the default device, the GPU if there is one.
:end_tab:

On these data we fix a two-layer MLP with 64 hidden units at a random
initialization. `batch_grads` computes the gradient of the squared loss,
flattened into one vector, for a whole stack of minibatches at once. Given
all 1,500 examples as minibatches of one, it returns every per-example
gradient, and from those we compute the full gradient and
$\operatorname{tr} \boldsymbol{\Sigma}$ exactly.

```{.python .input #sgd-gradient-variance-and-batch-size-1}
%%tab pytorch
data_iter, num_features = get_airfoil(batch_size=1500)
X, y = next(iter(data_iter))  # The whole dataset
device = X.device
params = {'W1': 0.1 * torch.randn(num_features, 64, device=device),
          'b1': torch.zeros(64, device=device),
          'W2': 0.1 * torch.randn(64, 1, device=device),
          'b2': torch.zeros(1, device=device)}

def loss(params, X, y):
    h = torch.relu(X @ params['W1'] + params['b1'])
    y_hat = (h @ params['W2'] + params['b2']).squeeze(-1)
    return (y_hat - y).pow(2).mean() / 2

def flat_grad(params, X, y):
    grads = torch.func.grad(loss)(params, X, y)
    return torch.cat([g.reshape(-1) for g in grads.values()])

batch_grads = torch.func.vmap(flat_grad, in_dims=(None, 0, 0))
G = batch_grads(params, X[:, None], y[:, None])  # (1500, 449)
g_full = G.mean(0)
tr_Sigma = ((G - g_full) ** 2).sum(1).mean()
```

```{.python .input #sgd-gradient-variance-and-batch-size-1}
%%tab jax
data_iter, num_features = get_airfoil(batch_size=1500)
X, y = next(iter(data_iter))  # The whole dataset
key1, key2, key = jax.random.split(d2l.get_key(), 3)
params = {'W1': 0.1 * jax.random.normal(key1, (num_features, 64)),
          'b1': jnp.zeros(64),
          'W2': 0.1 * jax.random.normal(key2, (64, 1)), 'b2': jnp.zeros(1)}

def loss(params, X, y):
    h = jax.nn.relu(X @ params['W1'] + params['b1'])
    y_hat = (h @ params['W2'] + params['b2']).squeeze(-1)
    return jnp.mean((y_hat - y) ** 2) / 2

def flat_grad(params, X, y):
    grads = jax.grad(loss)(params, X, y)
    return jnp.concatenate([g.reshape(-1) for g in jax.tree.leaves(grads)])

batch_grads = jax.jit(jax.vmap(flat_grad, in_axes=(None, 0, 0)))
G = batch_grads(params, X[:, None], y[:, None])  # (1500, 449)
g_full = G.mean(0)
tr_Sigma = ((G - g_full) ** 2).sum(1).mean()
```

For each batch size from 1 to 512 we now draw 200 minibatches, with
replacement as the derivation assumes, and compare their mean squared
gradient error with $\operatorname{tr} \boldsymbol{\Sigma} / b$.

```{.python .input #sgd-gradient-variance-and-batch-size}
%%tab pytorch
batch_sizes = [1, 8, 64, 512]
measured = []
for b in batch_sizes:
    idx = torch.randint(0, len(y), (200, b), device=device)
    errors = ((batch_grads(params, X[idx], y[idx]) - g_full) ** 2).sum(1)
    measured.append(errors.mean().item())
d2l.plot(batch_sizes, [measured, [tr_Sigma.item() / b for b in batch_sizes]],
         'batch size b', 'squared gradient error', xscale='log',
         yscale='log', legend=['measured', r'$\mathrm{tr}\,\Sigma\,/\,b$'])
```

```{.python .input #sgd-gradient-variance-and-batch-size}
%%tab jax
batch_sizes = [1, 8, 64, 512]
measured = []
for b, k in zip(batch_sizes, jax.random.split(key, len(batch_sizes))):
    idx = jax.random.randint(k, (200, b), 0, len(y))
    errors = ((batch_grads(params, X[idx], y[idx]) - g_full) ** 2).sum(1)
    measured.append(float(errors.mean()))
d2l.plot(batch_sizes, [measured, [float(tr_Sigma) / b for b in batch_sizes]],
         'batch size b', 'squared gradient error', xscale='log',
         yscale='log', legend=['measured', r'$\mathrm{tr}\,\Sigma\,/\,b$'])
```

The measured errors lie on the predicted line over a factor of 500 in
batch size, within the sampling error of 200 draws. The logarithmic axes
hide one consequence. Variance falling like $1/b$ means that the size
of the noise falls only like $1/\sqrt{b}$: a batch 100 times larger, at 100
times the computation per step, makes the gradient error only 10 times
smaller. Whether the larger batch is worth its computation depends on how
much the noise still limits the optimizer, which is the question of
:numref:`sec_batch_size`.

### Sampling without Replacement

The derivation drew the examples of a minibatch independently, with
replacement. Training loops do something else: they shuffle the dataset
at the start of every epoch and cut it into consecutive minibatches, so
that each example is used exactly once per epoch. Drawing $n$ times with
replacement would touch only a fraction $1 - (1 - 1/n)^n \approx 1 - e^{-1}
\approx 0.63$ of the examples per pass and use others twice or more.
Within a minibatch, sampling without replacement only helps: the variance
of the average of $b$ distinct examples is

$$\mathbb{E} \big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b} \cdot \frac{n - b}{n - 1},$$

where $\operatorname{tr} \boldsymbol{\Sigma}$ averages over the $n$
examples of the dataset. To see where the factor comes from, write
$\mathbf{e}_i = \nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})$ for the
error of example $i$, so that
$\operatorname{tr} \boldsymbol{\Sigma} = \frac{1}{n} \sum_i \|\mathbf{e}_i\|^2$.
The $n$ errors sum to zero, and so two distinct examples drawn at random
have errors that point, on average, slightly against each other:

$$\mathbb{E} \big[\mathbf{e}_i^\top \mathbf{e}_j\big] = \frac{1}{n(n - 1)} \sum_{i \neq j} \mathbf{e}_i^\top \mathbf{e}_j = \frac{\big\|\sum_i \mathbf{e}_i\big\|^2 - \sum_i \|\mathbf{e}_i\|^2}{n(n - 1)} = -\frac{\operatorname{tr} \boldsymbol{\Sigma}}{n - 1}.$$

The squared norm of the sum of $b$ errors has $b$ terms
$\|\mathbf{e}_i\|^2$, each of expectation
$\operatorname{tr} \boldsymbol{\Sigma}$, and $b(b - 1)$ cross terms, each
of expectation $-\operatorname{tr} \boldsymbol{\Sigma} / (n - 1)$. They
add up to $b \operatorname{tr} \boldsymbol{\Sigma}\, (n - b) / (n - 1)$,
and dividing by $b^2$ gives the formula. The correction factor is close to
1 while the batch is a small part of the dataset. It reaches 0 at $b = n$,
where the minibatch is the whole dataset and its gradient is exact. Across the
minibatches of one epoch, however, the draws are no longer independent:
an example used early in the epoch cannot come back later. The analysis of
this section then no longer applies verbatim. Reshuffling usually
converges faster in practice, and on strongly convex problems with small
learning rates provably so :cite:`Mishchenko.Khaled.Richtarik.2020`.
Exercise :exref:`ex_sgd-reshuffling` measures the difference.

### Batched Computation
:label:`subsec_minibatch-vectorization`

Statistics alone would not make minibatches the default: $b$ examples in
one step cost $b$ per-example gradients, as many as $b$ steps of SGD. What
makes them the default is the hardware, on which one step over $b$
examples takes far less time than $b$ steps over one.
:numref:`sec_linear_regression` measured one part of the reason:
predicting 1,000 examples in one matrix product is hundreds of times
faster than a Python loop over them, because every operation launched
from Python carries a fixed overhead of microseconds.

Memory is the other part. A linear layer with a $d \times d$ weight
matrix, applied to one example, reads all $d^2$ weights from memory and
performs $2d^2$ floating-point operations, a multiplication and an
addition per weight. In single precision, with 4 bytes per weight, that is
half an operation per byte read. Applied to a minibatch of $b$ examples,
the layer reads the weights once and performs $2bd^2$ operations, about
$b/2$ per byte while $b$ is small compared with $d$. This ratio of
operations to bytes moved is the *arithmetic intensity* of a computation.
A current GPU performs on the order of a hundred floating-point operations
in the time it reads one byte from its memory, a CPU on the order of ten.
With one example per call the arithmetic units therefore wait for memory
most of the time. Only a batch of tens to hundreds of examples gives them
enough work per byte.

We measure the combined effect on such a layer: a $1024 \times 1024$
weight matrix applied to 4,096 examples, in blocks of $b$ examples per
call, on a GPU if one is available. `d2l.Benchmark` of
:numref:`subsec_async_dispatch` times a function: a warm-up call absorbs
one-time costs, and the clock stops only once the device has finished all
the timed calls, here five, whose mean time it reports.

```{.python .input #sgd-batched-computation-1}
%%tab pytorch
device = d2l.try_gpu()
W = torch.randn(1024, 1024, device=device)
X = torch.randn(4096, 1024, device=device)
matmul = torch.matmul  # In full float32, PyTorch's default
```

```{.python .input #sgd-batched-computation-1}
%%tab jax
key1, key2 = jax.random.split(d2l.get_key())
W = jax.random.normal(key1, (1024, 1024))
X = jax.random.normal(key2, (4096, 1024))
# In full float32, not in the faster TF32 of recent GPUs
matmul = lambda A, B: jnp.matmul(A, B, precision='highest')
```

```{.python .input #sgd-batched-computation-2}
%%tab pytorch, jax
batch_sizes = [1, 4, 16, 64, 256, 1024, 4096]
per_example = []
for b in batch_sizes:
    blocks = [X[i:i + b] for i in range(0, len(X), b)]
    timing = d2l.Benchmark(lambda: [matmul(Xb, W) for Xb in blocks],
                           warmup=1, repeats=5)
    per_example.append(timing.time / len(X))
d2l.plot(batch_sizes, [[t * 1e6 for t in per_example],
                       [per_example[0] * 1e6 / b for b in batch_sizes]],
         'examples per call b', 'microseconds per example', xscale='log',
         yscale='log', legend=['measured', 'proportional to 1/b'])
```

The time per example first falls steeply, at small $b$ close to the $1/b$
line. A call takes about the same time whether it carries one example or a
few: its time goes to launching it and to reading $\mathbf{W}$, and more
examples share that time. Once a call carries enough examples to keep the
arithmetic units busy, its time grows with $b$, and the time per example
levels off at the time of the arithmetic itself. Between one example per
call and the level part of the curve lies more than an order of magnitude.
A minibatch should therefore be at least large enough to reach the level
part. The weights of this layer take 4 MB, which fits in the on-chip cache
of a current GPU and of many CPUs, so the measurement shows the launch
overhead and the bandwidth of the cache more than the speed of main
memory. The weights of a large network, gigabytes in all, do not fit; a
forward pass streams them from main memory, and the argument above applies
directly.

Memory limits the batch from above, since the activations stored for the
backward pass grow in proportion to $b$. When the desired batch does not
fit, the gradients of several smaller *micro-batches* can be accumulated
before a single update. This reproduces the large batch only if each
micro-batch loss is scaled so that the sum is the mean over the whole
batch. The update and any gradient clipping must also wait for the last
micro-batch. Layers that couple the examples of a batch, batch
normalization above all (:numref:`sec_batch_norm`), still see only the
micro-batch. :numref:`subsec_mp-accumulation` implements this technique.

## Minibatch SGD on the Airfoil Data
:label:`subsec_sgd-airfoil`

We now race the extremes and the middle on the airfoil data: full-batch
gradient descent, single-example SGD, and minibatches of 10 and 100, each
measured against the clock. The race also builds the training function
that the next section uses for momentum.

### The Training Function

The training function takes the optimizer as an update rule with a fixed
interface. The rule receives the parameters, a `states` object that holds
whatever auxiliary variables the algorithm keeps, and a dictionary of
hyperparameters. SGD keeps no state; momentum, in the next section, keeps a
running average of past gradients. The training function averages the loss
over each minibatch, so the update rule never divides by the batch size.

```{.python .input #minibatch-sgd-implementation-from-scratch-1}
%%tab pytorch
def sgd(params, states, hyperparams):
    for p in params:
        with torch.no_grad():
            p.sub_(hyperparams['lr'] * p.grad)
        p.grad.zero_()
```

```{.python .input #minibatch-sgd-implementation-from-scratch-1}
%%tab jax
def sgd(params, grads, states, hyperparams):
    return [p - hyperparams['lr'] * g for p, g in zip(params, grads)], states
```

:begin_tab:`pytorch`
The update reads each parameter's gradient from `p.grad`, changes the
parameter in place, and clears the gradient for the next minibatch.
:end_tab:

:begin_tab:`jax`
JAX arrays are immutable, so the update receives the gradients as an
argument and returns new parameters together with its new states, which
SGD passes back unchanged.
:end_tab:

To record training time we use a small timer that accumulates intervals.
We save it in the `d2l` package, since later chapters use it too.

```{.python .input #minibatch-sgd-vectorization-and-caches-2}
%%tab pytorch, jax
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.perf_counter()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.perf_counter() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

`train_with_rule` trains a linear regression model with a given update rule.
Its inner function `step` is the training step proper: the gradient of the
loss on a minibatch, then the update rule. The rest is bookkeeping, which
`record_run` takes over. It calls `step` on every minibatch, and about
every 200 examples it pauses the timer and evaluates the loss on the whole
dataset, so the recorded times count training alone. It plots the loss
against the epoch unless told not to, and returns a history: the number of
examples processed, the cumulative training time, and the loss at each
evaluation.

```{.python .input #sgd-the-training-function-1}
%%tab pytorch
#@save
def record_run(step, full_loss, data_iter, num_epochs=2, plot=True, *,
               ylim=None):
    """Call step(X, y) on every minibatch; about every 200 examples, record
    full_loss() and the training time so far, and plot the loss in the range
    ylim. Return the history."""
    # Wait until the device has finished (CUDA runs asynchronously)
    wait = torch.cuda.synchronize if torch.cuda.is_available() else (
        lambda: None)
    if plot:
        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                xlim=[0, num_epochs], ylim=ylim)
    history = {'examples': [], 'loss': []}
    n, i, timer = 0, 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            step(X, y)
            n, i = n + len(X), i + 1
            if i % max(1, 200 // len(X)) == 0:  # About every 200 examples
                wait()
                timer.stop()
                with torch.no_grad():
                    l = float(full_loss())
                history['examples'].append(n)
                history['loss'].append(l)
                if plot:
                    animator.add(i / len(data_iter), (l,))
                timer.start()
    history['time'] = timer.cumsum()
    print(f'loss: {l:.3f}, {timer.sum() / num_epochs:.3f} sec/epoch')
    return history
```

```{.python .input #sgd-the-training-function-1}
%%tab jax
#@save
def record_run(step, full_loss, data_iter, num_epochs=2, plot=True, *,
               ylim=None):
    """Call step(X, y) on every minibatch; about every 200 examples, record
    full_loss() and the training time so far, and plot the loss in the range
    ylim. Return the history."""
    if plot:
        animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                                xlim=[0, num_epochs], ylim=ylim)
    history = {'examples': [], 'loss': []}
    n, i, timer = 0, 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            out = step(X, y)
            n, i = n + len(X), i + 1
            if i % max(1, 200 // len(X)) == 0:  # About every 200 examples
                jax.block_until_ready(out)  # Wait for the queued steps
                timer.stop()
                l = float(full_loss())
                history['examples'].append(n)
                history['loss'].append(l)
                if plot:
                    animator.add(i / len(data_iter), (l,))
                timer.start()
    history['time'] = timer.cumsum()
    print(f'loss: {l:.3f}, {timer.sum() / num_epochs:.3f} sec/epoch')
    return history
```

```{.python .input #minibatch-sgd-implementation-from-scratch-2}
%%tab pytorch
#@save
def train_with_rule(trainer_fn, states, hyperparams, data_iter,
                    feature_dim, num_epochs=2, plot=True,
                    ylim=(0.22, 0.35)):
    """Train linear regression with an update rule; return its history. The
    default ylim is the range of the airfoil losses."""
    Xs, ys = (torch.cat(t) for t in zip(*data_iter))  # The whole data
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     device=Xs.device, requires_grad=True)
    b = torch.zeros(1, device=Xs.device, requires_grad=True)
    loss = lambda X, y: d2l.squared_loss(d2l.linreg(X, w, b), y).mean()

    def step(X, y):
        loss(X, y).backward()
        trainer_fn([w, b], states, hyperparams)

    loss(Xs, ys).backward()  # Warm-up, then clear the gradients
    w.grad, b.grad = None, None
    return d2l.record_run(step, lambda: loss(Xs, ys), data_iter, num_epochs,
                          plot, ylim=ylim)
```

```{.python .input #minibatch-sgd-implementation-from-scratch-2}
%%tab jax
#@save
def train_with_rule(trainer_fn, states, hyperparams, data_iter,
                    feature_dim, num_epochs=2, plot=True,
                    ylim=(0.22, 0.35)):
    """Train linear regression with an update rule; return its history. The
    default ylim is the range of the airfoil losses."""
    w = 0.01 * jax.random.normal(d2l.get_key(), (feature_dim, 1))
    b = jnp.zeros(1)
    loss = lambda w, b, X, y: d2l.squared_loss(d2l.linreg(X, w, b), y).mean()
    grad_fn = jax.jit(jax.grad(loss, argnums=(0, 1)))
    Xs, ys = (jnp.concatenate(t) for t in zip(*data_iter))  # The whole data
    full_loss = jax.jit(lambda w, b: loss(w, b, Xs, ys))

    def step(X, y):
        nonlocal w, b, states
        grads = grad_fn(w, b, X, y)
        (w, b), states = trainer_fn([w, b], list(grads), states, hyperparams)
        return w, b

    X, y = next(iter(data_iter))  # One step on the side compiles everything
    trainer_fn([w, b], list(grad_fn(w, b, X, y)), states, hyperparams)
    full_loss(w, b)
    return d2l.record_run(step, lambda: full_loss(w, b), data_iter,
                          num_epochs, plot, ylim=ylim)
```

:begin_tab:`pytorch`
The parameters live on the device of the data. Before the timer starts,
`train_with_rule` runs one forward and backward pass over the whole dataset,
which absorbs one-time costs such as loading the GPU's math libraries.
`record_run` waits for the device before it stops the timer, because CUDA
returns from a call before the device has finished it.
:end_tab:

:begin_tab:`jax`
The gradient and the full-data loss are compiled with `jax.jit`. Before the
timer starts, `train_with_rule` runs one step whose result it discards and
evaluates the loss once, so the recorded times exclude compilation. The
update rule returns new states rather than changing them, so this step
leaves the states of the run untouched.
JAX also returns before the device has finished, so `record_run` waits for
the parameters that `step` returns before it stops the timer.
:end_tab:

### Batch Size and Wall-Clock Time

A smaller batch takes more steps per epoch, and each step is cheaper and
noisier than a step on a larger batch. We now race batch sizes against the
clock to see which lowers the loss fastest.

`train_sgd` trains with plain SGD at a given learning rate and batch size.
We run four configurations. Batch size 1,500 is the whole dataset, which
makes it gradient descent with one update per epoch; it runs for 10 epochs.
Batch size 1 is single-example SGD with 1,500 updates per epoch. Batch
sizes 10 and 100 lie in between. The stochastic runs take 2 epochs each.
Their learning rates keep the ratio $\eta/b$ between $0.004$ and $0.005$,
so by the noise ball their iterates fluctuate about equally. Gradient
descent has no noise and can take larger steps, up to the stability limit
$\eta < 2/L$ of :numref:`sec_gd`, where $L$ is the largest curvature of the
loss. On these data $L \approx 2.1$, so the limit is about $0.95$, and we
run gradient descent at $0.9$.

```{.python .input #minibatch-sgd-implementation-from-scratch-3}
%%tab pytorch, jax
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_airfoil(batch_size)
    return train_with_rule(sgd, None, {'lr': lr}, data_iter, feature_dim,
                           num_epochs, plot=False)

runs = {'gradient descent': train_sgd(0.9, 1500, 10),
        'batch size 1': train_sgd(0.005, 1),
        'batch size 10': train_sgd(0.05, 10),
        'batch size 100': train_sgd(0.4, 100)}
```

All four runs end at about the same loss, so the race is decided by time.
The printed times per epoch already show the hardware effect. Every run
processes the same 1,500 examples per epoch, but single-example SGD does so
in 1,500 steps, against 150 and 15 for the batches of 10 and 100. Each step
launches its computations separately, however little work they hold, so
single-example SGD takes by far the longest per epoch. We plot the loss
twice: against the number of examples processed, which measures how much
each method extracts from the data, and against the training time, which
also reflects how well each method uses the hardware.

```{.python .input #minibatch-sgd-implementation-from-scratch-7}
%%tab pytorch, jax
fig, axes = d2l.plt.subplots(1, 2, figsize=(6.5, 2.8))
for ax, x, xlabel, ylabel, legend in zip(
        axes, ['examples', 'time'], ['examples processed', 'time (sec)'],
        ['loss', None], [list(runs), None]):
    d2l.plot([h[x] for h in runs.values()], [h['loss'] for h in runs.values()],
             xlabel, ylabel, legend, ylim=[0.235, 0.37], xscale='log',
             axes=ax)
```

Measured in examples, the three stochastic runs progress at about the same
rate, since they take about the same step $\eta/b$ per example. Gradient
descent lags far behind: after one pass over the data it has made one
update, the others 15 to 1,500. Measured in seconds, single-example SGD is
the slowest, because it computes its gradients one example at a time, and
the batch of 10 lies between it and the batch of 100. Only the batch of
100 is among the fastest in both views. Gradient descent even pulls ahead
of it on the clock, but only because the dataset is tiny, so that a full
gradient takes no longer than a few minibatch steps. With millions of
examples, one full gradient would take as long as thousands of minibatch
steps, and gradient descent would lag on the clock as well.

### Concise Implementation

Every framework ships these update rules as optimizer objects.
`train_with_optimizer` is the counterpart of `train_with_rule` for them. It
trains the same linear model with the same initialization and the same
squared loss, including the factor $\frac{1}{2}$ of
:numref:`sec_linear_regression`. The built-in mean squared error of most
libraries omits that factor, which would double the gradient and change
the meaning of a learning rate.

```{.python .input #minibatch-sgd-concise-implementation-1}
%%tab pytorch
#@save
def train_with_optimizer(trainer_fn, hyperparams, data_iter, num_epochs=2,
                         plot=True, ylim=(0.22, 0.35)):
    """Train linear regression with a torch.optim optimizer class; return
    its history. The default ylim is the range of the airfoil losses."""
    Xs, ys = (torch.cat(t) for t in zip(*data_iter))  # The whole data
    net = nn.Linear(Xs.shape[1], 1).to(Xs.device)
    nn.init.normal_(net.weight, std=0.01)
    nn.init.zeros_(net.bias)
    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = lambda X, y: d2l.squared_loss(net(X), y).mean()

    def step(X, y):
        optimizer.zero_grad()
        loss(X, y).backward()
        optimizer.step()

    loss(Xs, ys).backward()  # Warm-up; zero_grad clears it
    return d2l.record_run(step, lambda: loss(Xs, ys), data_iter, num_epochs,
                          plot, ylim=ylim)
```

```{.python .input #minibatch-sgd-concise-implementation-1}
%%tab jax
#@save
def train_with_optimizer(trainer_fn, hyperparams, data_iter, num_epochs=2,
                         plot=True, ylim=(0.22, 0.35)):
    """Train linear regression with an optax optimizer factory; return its
    history. The default ylim is the range of the airfoil losses."""
    Xs, ys = (jnp.concatenate(t) for t in zip(*data_iter))  # The whole data
    net = nnx.Linear(Xs.shape[1], 1,
                     kernel_init=nnx.initializers.normal(0.01),
                     rngs=nnx.Rngs(d2l.get_key()))
    optimizer = nnx.Optimizer(net, trainer_fn(**hyperparams), wrt=nnx.Param)
    loss = lambda net, X, y: d2l.squared_loss(net(X), y).mean()

    @nnx.jit
    def step_fn(net, optimizer, X, y):
        l, grads = nnx.value_and_grad(loss)(net, X, y)
        optimizer.update(net, grads)
        return l

    full_loss = nnx.jit(lambda net: loss(net, Xs, ys))
    X, y = next(iter(data_iter))  # One step on copies compiles everything
    step_fn(nnx.clone(net), nnx.clone(optimizer), X, y)
    full_loss(net)
    return d2l.record_run(lambda X, y: step_fn(net, optimizer, X, y),
                          lambda: full_loss(net), data_iter, num_epochs, plot,
                          ylim=ylim)
```

Repeating the run with batch size 10 and learning rate 0.05 through the
framework's optimizer reproduces the from-scratch result.

```{.python .input #minibatch-sgd-concise-implementation-2}
%%tab pytorch
data_iter, _ = get_airfoil(10)
train_with_optimizer(torch.optim.SGD, {'lr': 0.05}, data_iter);
```

```{.python .input #minibatch-sgd-concise-implementation-2}
%%tab jax
data_iter, _ = get_airfoil(10)
train_with_optimizer(optax.sgd, {'learning_rate': 0.05}, data_iter);
```

We can now predict where constant-step SGD stalls and how a minibatch
moves that point, and we have measured why minibatches add so little time
on parallel hardware. Three questions remain. How should the learning rate
fall during training? How large can the batch grow before a larger one
stops reducing the number of steps? And how can SGD move faster along the
flat directions of an ill-conditioned valley, where gradient descent is
slow and the noise ball is widest? Later sections of this chapter take up
the first two questions. The next section answers the third with momentum.

## Summary

Stochastic gradient descent replaces the exact gradient, which requires a
pass over all $n$ examples, with an unbiased estimate from a random sample.
With a constant learning rate $\eta$ the iterates do not converge. They
fluctuate in a noise ball whose squared radius is about
$\eta\sigma^2/(2\lambda)$ in a direction of curvature $\lambda$, so the
ball is widest where the objective is flattest. Decreasing learning rates
converge if their sum diverges and the sum of their squares does not. A
minibatch of $b$ examples divides the gradient variance by $b$, which we
measured over a factor of 500 in batch size. On parallel hardware it also
takes far less time than $b$ single-example steps, because the computation
becomes limited by arithmetic rather than by memory and launch overhead.
Of gradient descent, single-example SGD, and minibatch SGD, only the last
is fast both per example processed and per second.

## Exercises

1. **Noise as a random dataset.** The simulation in `sgd_2d` adds Gaussian
   noise to the gradient of $f(\mathbf{x}) = 0.1 x_1^2 + 2x_2^2$.
   :exlabel:`ex_sgd-noise-dataset`
    1. Show that this is SGD on the per-example losses
       $f(\mathbf{x}, \mathbf{w}) = 0.1 (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$,
       with $\mathbf{w}$ drawn from a normal distribution with mean zero,
       and find the covariance of $\mathbf{w}$ that reproduces unit-variance
       noise in both gradient coordinates.
    1. Why must the examples be spread much more widely along $x_1$ than
       along $x_2$? How does the expected loss
       $\mathbb{E}_{\mathbf{w}} f(\mathbf{x}, \mathbf{w})$ differ from the
       valley?
1. [code] **Reshuffling versus replacement.** ● Draw a dataset of
   $n = 100$ vectors $\mathbf{w}_i$ from the distribution of the previous
   problem and run SGD on the resulting finite sum for 20 epochs from the
   starting point $(-5, -2)$, once sampling $i$ with replacement and once
   reshuffling the dataset before each epoch. Record the distance to the
   minimizer of the finite sum after each epoch, averaged over 20 runs, at
   the constant learning rates
   $\eta = 0.1$ and $\eta = 0.01$. At which learning rate does reshuffling
   make the larger difference, and why?
   :exlabel:`ex_sgd-reshuffling`
1. [code] **Per-coordinate learning rates.** ● The noise ball
   :eqref:`eq_sgd-noise-ball` applies to each coordinate of the valley
   separately, with curvatures $\lambda_1 = 0.2$ and $\lambda_2 = 4$.
    1. Give each coordinate its own learning rate $\eta_j = \eta /
       \lambda_j$. Derive the contraction factor per step and the
       stationary variance of each coordinate, and compare the number of
       steps gradient descent needs to shrink both coordinates by a factor
       of 100 with that of a common rate.
    1. Implement the per-coordinate rates in `sgd_2d` and in `noise_ball`
       with $\eta = 0.1$, and verify both predictions numerically.
    1. Why does this choice need information that a single stochastic
       gradient does not supply?
1. [code] **Gradient variance after training.** ● The measurement in
   :numref:`subsec_sgd-variance` froze the two-layer network at a random
   initialization. Train it on the airfoil data for ten epochs of
   minibatch SGD with batch size 10 and learning rate 0.05, reshuffling
   the examples before each epoch. At initialization and again after
   training, compute `g_full` and $\operatorname{tr}\boldsymbol{\Sigma}$
   from the per-example gradients. Split $\operatorname{tr}\boldsymbol{\Sigma}$
   into the contributions of `W1`, `b1`, `W2`, and `b2`, and explain the
   changes in the `b2` and `W1` contributions from the form of the
   per-example gradient of the squared loss.
1. **Duplicated data.** Suppose every example in the training set is
   silently duplicated, so that the dataset doubles in size while its
   empirical distribution is unchanged. For SGD with $b = 1$, for minibatch
   SGD at a fixed $b$, and for full-batch gradient descent, state whether
   the cost per step, the number of steps to reach a given loss, and the
   variance of the gradient estimate change, and explain each answer.
1. [code] **Where the time per example levels off.** ● The measurement in
   :numref:`subsec_minibatch-vectorization` applied a $1024 \times 1024$
   weight matrix to blocks of $b$ examples.
    1. Repeat it for $d \times d$ weight matrices with
       $d \in \{256, 1024, 4096\}$. For each $d$, find the smallest $b$ at
       which the time per example is within 20% of its minimum.
    1. Including the bytes of the inputs and outputs, the arithmetic
       intensity of the layer in single precision is $b / (2 + 4b/d)$.
       Derive this expression, and use it to predict how the $b$ you found
       should depend on $d$. Does the measurement agree?
    1. On a GPU, repeat the measurement for $d = 1024$ in bfloat16. Which
       way does the bend move, and which two properties of the format
       explain the move?
1. **Memory traffic of matrix multiplication.** Computing
   $\mathbf{A} = \mathbf{B}\mathbf{C}$ for $n \times n$ matrices takes
   $2n^3$ floating-point operations, however they are ordered. Count the
   floats each of three orderings reads from main memory, assuming a cache
   that holds $M$ floats with $n < M \ll n^2$, and keep only the terms of
   order $n^3$.
    1. Computing every entry as a dot product of a row of $\mathbf{B}$ with
       a column of $\mathbf{C}$ reads both for every entry. Computing
       $\mathbf{A}$ one column at a time keeps the column
       $\mathbf{C}_{:,j}$ in cache while it reads all of $\mathbf{B}$. Show
       that the second ordering halves the traffic of the first, and
       compute the arithmetic intensity of both in single precision.
    1. A blocked ordering splits the matrices into $w \times w$ blocks. It
       keeps one block of $\mathbf{A}$ in cache while it accumulates the
       $n/w$ products of a block of $\mathbf{B}$ with a block of
       $\mathbf{C}$ that contribute to it. Show that it reads about
       $2n^3/w$ floats, and find the largest $w$ for which the three blocks
       fit in the cache.
    1. How many bytes of cache does the blocked ordering need to reach an
       arithmetic intensity of 100 operations per byte in single precision?
1. **Unequal micro-batches.** A training loop accumulates the gradients of
   $k = 4$ micro-batches into one update with an effective batch of $100$
   examples, drawn uniformly at random from the training set. The
   micro-batches hold $32$, $32$, $32$, and $4$ examples. The loop averages
   the loss of each micro-batch over its own examples and divides the
   accumulated gradient by $k$. Assume that the per-example gradients are
   independent, with mean $\nabla f(\mathbf{x})$ and covariance
   $\boldsymbol{\Sigma}$.
    1. Find the weight that the accumulated gradient gives to each example,
       and show that it is still an unbiased estimate of
       $\nabla f(\mathbf{x})$.
    1. Express the covariance of the accumulated gradient as a multiple of
       $\boldsymbol{\Sigma}$, and find the batch size of a uniformly
       weighted average with the same covariance.
    1. State the scaling of the micro-batch losses that makes the
       accumulated gradient the average over all $100$ examples. Use
       :eqref:`eq_sgd-noise-ball` to compare the squared radius of the
       noise ball of the two loops at the same learning rate.
1. [code] **Batch size and learning rate together.** ● `train_sgd` trains
   the linear model at a chosen batch size and learning rate and returns
   its history: the examples processed, the cumulative training time, and
   the loss at each evaluation. Fix
   $\eta = 0.05$ at $b = 10$ and derive the learning rate for each
   $b \in \{1, 10, 100, 1000\}$ by two rules, $\eta \propto \sqrt{b}$ and
   $\eta \propto b$.
    1. Train each configuration for two epochs. Report the time per epoch
       and the final loss of each run, and name the runs that diverge.
    1. The training loss is a quadratic in the weights and the bias. Its
       Hessian is

       $$\mathbf{H} = \frac{1}{n} \sum_{i=1}^{n} \tilde{\mathbf{x}}_i \tilde{\mathbf{x}}_i^\top,$$

       where $\tilde{\mathbf{x}}_i$ appends a constant $1$ to the features
       of example $i$ and $n = 1500$. Compute the largest eigenvalue $L$ of
       $\mathbf{H}$ from the standardized data that `get_airfoil` loads,
       and confirm the value quoted for the race. Gradient descent on a
       quadratic is stable only for $\eta < 2/L$ (:numref:`sec_gd`); use
       this to explain which runs diverge.
    1. At $b = 1$, which rule reaches the lower final loss? Explain the
       ordering with :eqref:`eq_sgd-noise-ball`.

:begin_tab:`pytorch`
[Discussions](https://d2l.discourse.group/t/497)
:end_tab:

:begin_tab:`jax`
[Discussions](https://d2l.discourse.group/t/1067)
:end_tab:

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §9.3]{.kicker}

Stochastic and Minibatch Gradient Descent<br>**sampled gradients · the noise ball · gradient variance · batched computation**
:::
:::

::: {.slide title="An Exact Gradient Needs a Full Pass over the Data"}
[The problem]{.kicker}

ResNet-50 on ImageNet, a common recipe:

| quantity | value |
|:--|--:|
| training images | 1.28 million |
| minibatch | 256 images |
| passes over the data with minibatches | 90 |
| updates in 90 passes | 450,000 |
| passes for 450,000 exact-gradient updates | 450,000 |

::: {.d2l-note .warn}
Exact gradients would take 5,000 times the work, yet a few hundred random
images already show which way is downhill.
:::

[Goyal et al., 2017]{.small}
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive how far constant-step SGD wanders from the minimum;
- predict how the learning rate, the curvature, and the batch size change
  that distance;
- compute the arithmetic intensity of a layer applied to a minibatch;
- read a race of batch sizes against examples and against time.

**Running example:** the valley $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$
(condition number 20) with unit gradient noise, then the airfoil
regression data (1,500 examples, 5 features).
:::

::: {.slide title="Noise Is the Second Problem of the Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | adds |
|---|---|
| Landscapes | curvature and noise slow training |
| Gradient Descent | step size against curvature |
| **Stochastic and Minibatch GD** | **sampled gradients, noise ball, batching** |
| Momentum | averaging over time |
| Adam and AdamW | a step size per coordinate, decoupled decay |
| Schedules | warmup and decay |
| Muon | orthogonalized updates for matrices |
| Batch Size | how many examples per step |
| Scaling Up | learning rates across widths |
| Practice | clipping, averaging, fair comparisons |
:::

::: {.col .narrow}
- **Uses:** gradient descent on a quadratic; minibatch SGD for linear
  regression; autograd.
- **Skips:** convergence proofs, variance-reduction methods, schedules
  beyond the Robbins–Monro conditions.
:::
:::
:::

::: {.slide title="A Sampled Gradient Is Right on Average"}
[Sampled gradients]{.kicker}

$$f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n f_i(\mathbf{x}), \qquad
\mathbf{x} \leftarrow \mathbf{x} - \eta\, \nabla f_i(\mathbf{x}),
\quad i \sim \textrm{uniform}$$

- One step costs one per-example gradient instead of $n$.
- Unbiased: $\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$.
- Noisy: $\nabla f_i = \nabla f + \boldsymbol{\xi}$ with
  $\mathbb{E}[\boldsymbol{\xi}] = \mathbf{0}$; one example pulls toward
  fitting itself alone.
:::

::: {.slide title="Constant-Step SGD Stalls in a Cloud"}
[Sampled gradients]{.kicker}

The valley, $\eta = 0.1$, 300 steps, unit-variance noise on both gradient
coordinates:

@!sgd-stochastic-gradient-updates-3

The cloud does not shrink with more steps, and it is longest along the
flat $x_1$.
:::

::: {.slide title="Contraction and Noise Balance at a Fixed Variance"}
[The noise ball]{.kicker}

One coordinate, curvature $\lambda$, noise $\xi_t$ of variance $\sigma^2$:

I. $\;x_{t+1} = (1-\eta\lambda)\,x_t - \eta\,\xi_t$

II. $\;\mathbb{E}[x_{t+1}^2] = (1-\eta\lambda)^2\,\mathbb{E}[x_t^2] + \eta^2\sigma^2$
(the cross term has mean zero)

III. Fixed point $v = (1-\eta\lambda)^2 v + \eta^2\sigma^2$:

$$\boxed{\mathbb{E}[x_\infty^2] = \frac{\eta\sigma^2}{\lambda(2-\eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda}}$$

The squared radius grows with $\eta$ and $\sigma^2$ and shrinks with $\lambda$.
:::

::: {.slide title="The Cloud Is Four Times Wider along the Flat Direction"}
[The noise ball]{.kicker}

The valley at $\eta = 0.1$, $\sigma^2 = 1$:

| direction | $\lambda$ | variance | standard deviation |
|:--|--:|--:|--:|
| $x_1$ (flat) | 0.2 | $0.1/(0.2 \cdot 1.98) \approx 0.25$ | 0.5 |
| $x_2$ (steep) | 4 | $0.1/(4 \cdot 1.6) \approx 0.016$ | 0.125 |

::: {.d2l-note}
The flat directions, slowest for gradient descent, are also where noise
carries the iterate farthest: the pull back is weakest there.
:::
:::

::: {.slide title="10,000 Simulated Runs Match the Formula"}
[The noise ball]{.kicker}

Start 10,000 runs at the minimum, take 2,000 noisy steps, measure the
variance of each coordinate across runs:

@!sgd-the-noise-ball

Measured and predicted agree to within the sampling error.
:::

::: {.slide title="Check: Halve the Learning Rate"}
[Check]{.kicker}

On the valley, $\eta$ drops from 0.1 to 0.05. What happens to the variance
along $x_1$ and along $x_2$, and to the number of steps needed to reach
the cloud?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $x_1$: $0.25 \to 0.126$, half, since $\eta\lambda$ is small.
$x_2$: $0.0156 \to 0.0069$, a little less than half, since
$\eta\lambda = 0.4$ is not small. Each step closes the fraction
$\eta\lambda$ of the distance, so reaching the cloud takes twice as many
steps.
:::
:::
:::

::: {.slide title="Decreasing Rates Converge if Their Sum Diverges and Their Squares Do Not"}
[The noise ball]{.kicker}

- $\sum_t \eta_t = \infty$: the iterate can still travel arbitrarily far.
- $\sum_t \eta_t^2 < \infty$: the injected noise, of total variance
  $\sigma^2 \sum_t \eta_t^2$, stays bounded.
- $\eta_t \propto 1/t$ satisfies both; $\eta_t \propto e^{-ct}$ fails the
  first and stops short of the minimum.

::: {.d2l-note}
Deep networks use schedules (warmup, cosine, WSD) for which these
conditions are a guide, not a recipe.
:::

[Robbins and Monro (1951)]{.small}
:::

::: {.slide title="Averaging b Examples Divides the Variance by b"}
[Minibatches]{.kicker}

$$\mathbf{g}_t = \frac{1}{b}\sum_{i\in\mathcal{B}_t} \nabla f_i(\mathbf{x}_t),
\qquad
\mathbb{E}\,\|\mathbf{g}_t - \nabla f\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$$

- $\operatorname{tr}\boldsymbol{\Sigma}$: expected squared error of one
  example's gradient; independent errors add in squared norm.
- Noise ball: $\eta\sigma^2/(2\lambda) \;\to\; \eta\sigma^2/(2\lambda b)$.
- Halving $\eta$ and doubling $b$ shrink the ball equally: the first
  doubles the steps, the second the work per step.
:::

::: {.slide title="On an MLP, the Error Follows tr Σ / b from b = 1 to 512"}
[Minibatches]{.kicker}

Two-layer MLP (64 hidden units) frozen at a random initialization,
airfoil data, 200 minibatches per batch size:

@!sgd-gradient-variance-and-batch-size
:::

::: {.slide title="A 100 Times Larger Batch Makes the Noise Only 10 Times Smaller"}
[Minibatches]{.kicker}

- Variance $\propto 1/b$, so the size of the noise $\propto 1/\sqrt{b}$.
- Without replacement within a batch:
  $\dfrac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\cdot\dfrac{n-b}{n-1}$,
  exactly 0 at $b = n$.
- Shuffling once per epoch uses every example once; $n$ draws with
  replacement reach only $1 - e^{-1} \approx 63\%$ of them.

::: {.d2l-note}
When a larger batch stops reducing the number of steps is the question of
the Batch Size section.
:::
:::

::: {.slide title="One Example per Call Leaves the Arithmetic Units Idle"}
[Batched computation]{.kicker}

A $d \times d$ linear layer in float32 (4 bytes per weight):

| examples per call | operations | bytes of weights | operations per byte |
|:--|--:|--:|--:|
| 1 | $2d^2$ | $4d^2$ | 0.5 |
| $b \ll d$ | $2bd^2$ | $4d^2$ | $b/2$ |

A GPU does about 100 operations per byte it reads, a CPU about 10.

::: {.d2l-note}
Every call also carries a launch overhead of microseconds. A batch shares
both among its examples.
:::
:::

::: {.slide title="Time per Example Falls until the Device Is Busy"}
[Batched computation]{.kicker}

A $1024 \times 1024$ layer applied to 4,096 examples, $b$ examples per
call:

@!sgd-batched-computation-2

The curve follows $1/b$ at first and levels off at the time of the
arithmetic itself.
:::

::: {.slide title="Check: Which Batch Keeps a Layer Busy?"}
[Check]{.kicker}

A $4096 \times 4096$ layer in bfloat16 (2 bytes per weight) runs on a GPU
that does 300 operations per byte. Ignoring inputs and outputs, from which
$b$ on is the layer limited by arithmetic rather than by memory?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $2bd^2$ operations over $2d^2$ bytes is $b$ operations per
byte, so $b \approx 300$. Counting inputs and outputs, $b$ is somewhat
larger.
:::
:::
:::

::: {.slide title="SGD and Momentum Share One Update-Rule Interface"}
[Minibatch SGD on the airfoil data]{.kicker}

Interface `(params, states, hyperparams)` (JAX also passes the gradients
and returns the new states);
`states` is empty for SGD and holds a running average for momentum:

@minibatch-sgd-implementation-from-scratch-1

`train_with_rule` trains linear regression with any such rule and records the
loss against examples and training time.
:::

::: {.slide title="Only Minibatches Are Fast per Example and per Second"}
[Minibatch SGD on the airfoil data]{.kicker}

@!minibatch-sgd-implementation-from-scratch-7

Per example the stochastic runs tie and gradient descent lags; per second
single examples lag, and on these 1,500 examples gradient descent leads. On
millions of examples gradient descent lags on both.
:::

::: {.slide title="Memory and Batch Normalization Bound the Batch from Above"}
[Minibatch SGD on the airfoil data]{.kicker}

- Stored activations grow in proportion to $b$.
- Too large to fit: accumulate the gradients of several micro-batches,
  scale each loss so the sum is the batch mean, update and clip once.
- Batch normalization still sees only one micro-batch.

::: {.d2l-note}
Choose $b$ large enough to keep the device busy and small enough to fit.
:::
:::

::: {.slide title="Noise Sets the Floor; Batches Lower It Cheaply"}
[Wrap-up]{.kicker}

- SGD: an unbiased gradient from a single example.
- Constant $\eta$: noise ball $\eta\sigma^2/(\lambda(2-\eta\lambda))$; on
  the valley at $\eta = 0.1$, standard deviation 0.5 along $x_1$, 0.125
  along $x_2$.
- Convergence: $\sum_t\eta_t = \infty$, $\sum_t\eta_t^2 < \infty$.
- Minibatch of $b$: variance $\operatorname{tr}\boldsymbol{\Sigma}/b$,
  measured from $b = 1$ to 512.
- Hardware: about $b/2$ operations per byte; time per example falls by
  more than an order of magnitude.

**Next:** momentum, for the slow flat directions where the noise ball is
also widest.
:::
