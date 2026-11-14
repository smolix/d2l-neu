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

<!-- WRITER: this roadmap is out of date after the code pass. The 1/b variance is no longer measured on a small network (the MLP cells are cut, SGD-7), the batched-computation timing is cut (SGD-10/11, to chap_performance), and the race no longer builds a training function for momentum (record_run, train_with_rule and train_with_optimizer are deleted; Timer moved to the build-only legacy-timer-lib.md). The section now measures (i) the noise ball of linear regression on the airfoil data, with the least-squares optimum in closed form, and the exact-fit case in which it vanishes (new subsection "Where the Model Stops"); (ii) batch sizes 1, 10, 100 and 1,500 on the same data, per example and per second ("Batch Size and Wall-Clock Time"). The extras imports cell is gone: nothing uses time or random any more. -->

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

<!-- WRITER: the get_airfoil cells are #@save blocks that must stay byte-identical (structure 5, library fingerprints), so the two tab notes above cannot become code comments. The PyTorch code already carries "# Whole minibatches of indices"; both notes can be cut (SGD-17). One sentence on the data is enough (SGD-6); the feature names are not used again. -->

On these data we fix a two-layer MLP with 64 hidden units at a random
initialization. `batch_grads` computes the gradient of the squared loss,
flattened into one vector, for a whole stack of minibatches at once. Given
all 1,500 examples as minibatches of one, it returns every per-example
gradient, and from those we compute the full gradient and
$\operatorname{tr} \boldsymbol{\Sigma}$ exactly.

<!-- WRITER: cells sgd-gradient-variance-and-batch-size-1 and sgd-gradient-variance-and-batch-size (the MLP frozen at initialization, its 1,500 per-example gradients and the tr(Sigma)/b plot) are cut (SGD-7, structure 3: a check that cannot fail). The paragraph before ("On these data we fix a two-layer MLP ...") and the one after ("For each batch size from 1 to 512 ...") describe cells that no longer exist, and "The measured errors lie on the predicted line over a factor of 500" is no longer measured. Keep the 1/b law with its two-line reason and the 1/sqrt(b) consequence ("a batch 100 times larger ... only 10 times smaller"). -->

For each batch size from 1 to 512 we now draw 200 minibatches, with
replacement as the derivation assumes, and compare their mean squared
gradient error with $\operatorname{tr} \boldsymbol{\Sigma} / b$.

The measured errors lie on the predicted line over a factor of 500 in
batch size, within the sampling error of 200 draws. The logarithmic axes
hide one consequence. Variance falling like $1/b$ means that the size
of the noise falls only like $1/\sqrt{b}$: a batch 100 times larger, at 100
times the computation per step, makes the gradient error only 10 times
smaller. Whether the larger batch is worth its computation depends on how
much the noise still limits the optimizer, which is the question of
:numref:`sec_batch_size`.

### Where the Model Stops

<!-- WRITER: NEW subsection and experiment (SGD-4; structure 3 "Where the model stops"). Linear regression on the 1,500 airfoil examples with the squared loss (1/2)(x_i^T w - y_i)^2, the bias as a sixth weight. The loss is a quadratic, so its Hessian H = X^T X / n and its minimizer w_star (np.linalg.lstsq) are exact, and the excess loss f(w) - f(w_star) = (1/2)(w - w_star)^T H (w - w_star) is computed without sampling error. The first cell prints L, the largest eigenvalue of H (SGD-14: the race's "L = 2.1" is now printed), and tr Sigma, the mean squared norm of the per-example gradients at w_star (their mean is zero there). Measured in every run (the data are fixed): "L = 2.10, tr Sigma = 3.24", so 2/L = 0.95. Other eigenvalues of H, for reference: 0.17, 0.67, 0.92, 1.00 (the bias), 1.13, 2.10. -->

```{.python .input #sgd-where-the-model-stops-1}
%%tab pytorch, jax
data_iter, num_features = get_airfoil(batch_size=1500)
X, y = map(d2l.numpy, next(iter(data_iter)))  # All 1,500 examples
X = np.hstack([X, np.ones((len(X), 1), np.float32)])  # Bias as a 6th weight
H = X.T @ X / len(X)  # Hessian of the loss
w_star = np.linalg.lstsq(X, y)[0]  # The minimizer, in closed form
G = X * (X @ w_star - y)[:, None]  # Per-example gradients at w_star
tr_Sigma = (G ** 2).sum(axis=1).mean()
print(f'L = {np.linalg.eigvalsh(H).max():.2f}, tr Sigma = {tr_Sigma:.2f}')
```

<!-- WRITER: sgd_runs runs 400 independent copies of minibatch SGD at once, one per row of W, drawing each minibatch with replacement as the derivation of the noise ball assumes; excess_loss evaluates the quadratic form above for every stored iterate. Prediction for the prose, from the noise ball: in an eigendirection of H with curvature lambda_k and gradient-noise variance s_k^2 / b, the stationary second moment is about eta s_k^2 / (2 lambda_k b), and the direction contributes (1/2) lambda_k times that to the excess loss, i.e. eta s_k^2 / (4b). The curvature cancels: summed over directions the excess loss is about eta tr(Sigma) / (4b), whatever the conditioning. Halving eta halves it. -->

```{.python .input #sgd-where-the-model-stops-2}
%%tab pytorch, jax
def sgd_runs(X, y, w, eta, b, num_steps, rng, num_runs=400):
    """Run minibatch SGD on the squared loss num_runs times from w, drawing
    examples with replacement; return every iterate, (steps, runs, d)."""
    W, trace = np.tile(w, (num_runs, 1)), []
    for idx in rng.integers(0, len(X), (num_steps, num_runs, b)):
        r = np.einsum('rbd,rd->rb', X[idx], W) - y[idx]  # Residuals
        W = W - eta / b * np.einsum('rbd,rb->rd', X[idx], r)
        trace.append(W)
    return np.array(trace)

def excess_loss(W, w_star, H):
    """f(w) - f(w_star) of the quadratic loss, for every w in the last axis."""
    D = W - w_star
    return 0.5 * np.einsum('...d,de,...e->...', D, H, D)
```

<!-- WRITER: the runs start at w_star, so the cell measures the stationary ball only, with no transient; b = 10, three epochs (450 steps), excess loss averaged over the last epoch and over the 400 runs. Measured over 10 development runs of this cell with 400 runs each, plus the notebook runs: eta = 0.05: 0.0042 to 0.0043 (predicted 0.0041); eta = 0.025: 0.0020 to 0.0021 (predicted 0.0020); the ratio of the two 2.03 to 2.11 (predicted 2). Halving eta halves the excess loss. (With 100 runs per cell the ratio spread was 1.99 to 2.3, hence 400.) The measured value sits up to 6% above the small-step prediction at eta = 0.05 because eta * L = 0.1 is not negligible (the exact factor 2 / (2 - eta lambda) of the noise ball, 1.06 at L = 2.1). Scale for the reader: the excess is under 2% of the minimum loss 0.2417. -->

```{.python .input #sgd-where-the-model-stops-3}
%%tab pytorch, jax
rng = np.random.default_rng()
for eta in (0.05, 0.025):  # b = 10, three epochs of 150 steps from w_star
    trace = sgd_runs(X, y, w_star, eta, 10, 450, rng)
    measured = excess_loss(trace[-150:], w_star, H).mean()  # The last epoch
    print(f'eta={eta}: excess loss {measured:.4f}, '
          f'predicted {eta * tr_Sigma / (4 * 10):.4f}')
```

<!-- WRITER: the exact-fit arm. Redesigned from the worklist's "5 of the examples, b = 1": random sets of 5 airfoil examples are badly conditioned (the features take few distinct values; smallest nonzero curvature 0.02 in the median draw, below 0.001 in one draw in twenty), and constant-step SGD at eta = 0.05 left a median excess loss of 7e-5 after 2,000 steps and 0.04 in the worst 5% of draws; no firm lesson. Instead the cell keeps the same 1,500 inputs, the same Hessian, eta = 0.05 and b = 10, and changes one thing: the targets become y_fit = X w_star, the predictions of the best linear model, so that the model fits every example exactly, as an overparameterized network fits its training set. Every per-example gradient then vanishes at w_star, the noise vanishes with it, and constant-step SGD converges from w = 0 geometrically to float32 rounding. Measured (worst of 100 or 400 runs; 8 development runs plus four notebook runs, the same every time to one digit): after 1 epoch 1e-3; after 5 epochs 3e-8 to 5e-8; after 10 epochs 1e-13 to 2e-13; after 20 epochs 2e-14 (float32 rounding). Compare 0.004 for the noisy targets at the same eta and b. The networks of the book sit between the two cases: the examples still disagree at the end of training, but less than at the start (Exercise 4). -->

```{.python .input #sgd-where-the-model-stops-4}
%%tab pytorch, jax
y_fit = X @ w_star  # Targets that the linear model fits exactly
trace = sgd_runs(X, y_fit, np.zeros_like(w_star), 0.05, 10, 3000, rng)
worst = excess_loss(trace[149::150], w_star, H).max(axis=1)  # After each epoch
print('exact fit, worst of 400 runs after 1, 5, 10, 20 epochs:',
      *(f'{worst[e - 1]:.0e}' for e in (1, 5, 10, 20)))
```


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

<!-- WRITER: cells sgd-batched-computation-1/-2 (the 1024 x 1024 matrix product timed in blocks of b) are cut (SGD-10, SGD-11; structure 3: the arithmetic-intensity argument, the benchmark and the micro-batch paragraph go to the chapter 13 hand-over list). This whole subsection now has no cell: keep one sentence with a pointer to chap_performance on why the device prefers batches, and one sentence on accumulation with subsec_mp-accumulation (SGD-12). The label subsec_minibatch-vectorization is still cited by Exercise 6, which also moves out. -->

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

<!-- WRITER: everything this subsection describes is gone: the update-rule interface and the cell `sgd` (minibatch-sgd-implementation-from-scratch-1), the Timer cell (moved byte for byte to the build-only legacy-timer-lib.md, d2l.Timer unchanged), record_run and train_with_rule (#@save, deleted; their only other consumer, momentum.md, loses its airfoil runs in its own pass), and the four tab notes (SGD-17). In their place is one compact training function per tab, not saved, train_sgd(lr, batch_size, num_epochs): plain minibatch SGD on linear regression from w = 0 (one weight vector, the bias last, as in "Where the Model Stops"); it returns the training loss after every update and the seconds per epoch. The losses are computed after training from the stored iterates, so the clock counts training alone. The time comes from d2l.Benchmark (sec_use_gpu; one clause: it waits for the device before stopping the clock) with one repeat per epoch. PyTorch: a warm-up backward pass before the clock starts. JAX: the step is jitted once at module level, and one call per batch size compiles it before the clock starts. -->

To record training time we use a small timer that accumulates intervals.
We save it in the `d2l` package, since later chapters use it too.

`train_with_rule` trains a linear regression model with a given update rule.
Its inner function `step` is the training step proper: the gradient of the
loss on a minibatch, then the update rule. The rest is bookkeeping, which
`record_run` takes over. It calls `step` on every minibatch, and about
every 200 examples it pauses the timer and evaluates the loss on the whole
dataset, so the recorded times count training alone. It plots the loss
against the epoch unless told not to, and returns a history: the number of
examples processed, the cumulative training time, and the loss at each
evaluation.

```{.python .input #sgd-the-training-function}
%%tab pytorch
def train_sgd(lr, batch_size, num_epochs=2):
    """Minibatch SGD on linear regression from w = 0; return the loss after
    every update and the seconds per epoch."""
    data_iter, num_features = get_airfoil(batch_size)
    X, y = (torch.cat(t) for t in zip(*data_iter))  # The whole dataset
    w = torch.zeros(num_features + 1, device=X.device, requires_grad=True)
    loss = lambda X, y: ((X @ w[:-1] + w[-1] - y) ** 2).mean() / 2
    loss(X, y).backward()  # Warm-up; the first update clears the gradient
    ws = []

    def epoch():
        for X_b, y_b in data_iter:
            w.grad = None
            loss(X_b, y_b).backward()
            with torch.no_grad():
                w.sub_(lr * w.grad)
            ws.append(w.detach().clone())

    seconds = d2l.Benchmark(epoch, warmup=0, repeats=num_epochs).time
    W = torch.stack(ws)  # The weights after every update, (updates, 6)
    losses = ((X @ W[:, :-1].T + W[:, -1] - y[:, None]) ** 2).mean(0) / 2
    return losses, seconds
```

```{.python .input #sgd-the-training-function}
%%tab jax
@jax.jit
def sgd_step(w, X, y, lr):
    """One minibatch SGD step on the squared loss of linear regression."""
    loss = lambda w: jnp.mean((X @ w[:-1] + w[-1] - y) ** 2) / 2
    return w - lr * jax.grad(loss)(w)

def train_sgd(lr, batch_size, num_epochs=2):
    """Minibatch SGD on linear regression from w = 0; return the loss after
    every update and the seconds per epoch."""
    data_iter, num_features = get_airfoil(batch_size)
    X, y = (jnp.concatenate(t) for t in zip(*data_iter))  # The whole dataset
    ws = [jnp.zeros(num_features + 1)]
    sgd_step(ws[0], X[:batch_size], y[:batch_size], lr)  # Compile first

    def epoch():
        for X_b, y_b in data_iter:
            ws.append(sgd_step(ws[-1], X_b, y_b, lr))
        return ws[-1]  # Benchmark waits for it

    seconds = d2l.Benchmark(epoch, warmup=0, repeats=num_epochs).time
    W = jnp.stack(ws[1:])  # The weights after every update, (updates, 6)
    losses = ((X @ W[:, :-1].T + W[:, -1] - y[:, None]) ** 2).mean(0) / 2
    return losses, seconds
```

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

<!-- WRITER: the learning rates now keep eta/b at exactly 0.005 (0.005, 0.05, 0.5), not "between 0.004 and 0.005"; gradient descent stays at 0.9, below 2/L = 0.95 with L = 2.10 printed in "Where the Model Stops" (SGD-14), so "On these data L is about 2.1" can cite that output. -->

```{.python .input #minibatch-sgd-implementation-from-scratch-3}
%%tab pytorch, jax
runs = {}
for b, lr, num_epochs in ((1500, 0.9, 10), (1, 0.005, 2), (10, 0.05, 2),
                          (100, 0.5, 2)):  # lr / b = 0.005 for SGD
    runs[b], seconds = train_sgd(lr, b, num_epochs)
    updates = 1500 // b
    print(f'b={b:4d}: {updates:4d} updates/epoch, {seconds:.3f} s/epoch '
          f'({1000 * seconds / updates:.1f} ms/update), '
          f'last-epoch loss {float(runs[b][-updates:].mean()):.3f}')
```

<!-- WRITER: the printout is now per run: updates per epoch, seconds per epoch, milliseconds per update, and the training loss averaged over the last epoch. Measured on an RTX 4090 (3 development runs and 3 to 4 notebook runs per tab; times depend on the machine and its load). PyTorch: b = 1: 1.31 to 1.84 s/epoch (0.9 to 1.2 ms/update); b = 10: 0.11 to 0.16 s (0.7 to 1.1 ms); b = 100: 0.011 to 0.020 s (0.7 to 1.3 ms); b = 1,500: 0.002 s (2.0 to 2.4 ms for its one update; the loader gathers 1,500 indices). JAX: b = 1: 0.88 to 0.94 s/epoch (0.6 ms/update); b = 10: 0.09 to 0.13 s (0.6 to 0.8 ms); b = 100: 0.010 to 0.021 s (0.7 to 1.4 ms); b = 1,500: 0.002 to 0.004 s (1.8 to 3.6 ms). In the JAX tab about 0.45 ms of every update is the loader copying the minibatch to the device, and a fresh process spends its first epochs at higher cost while the GPU clocks up, which is why b = 100 (15 updates per epoch) reads above b = 1; at steady state its update costs 0.6 to 0.7 ms like the others. Firm lessons, both tabs: an update costs about a millisecond (0.6 to 1.4 ms) whether it carries 1 or 100 examples, so per example b = 100 is 40 to 100 times cheaper than b = 1; a full gradient on 1,500 examples costs no more than two or three minibatch updates, so these data cannot show gradient descent losing on the clock (SGD-13; the ResNet-50 arithmetic of the opening does). Last-epoch losses: 0.242 for gradient descent (deterministic; the minimum is 0.2417) and 0.244 to 0.249 for the three stochastic runs. The eta/b invariance predicts the same ball for all three, 0.2417 + eta tr(Sigma)/(4b) = 0.2417 + 0.005 * 3.24 / 4 = 0.246, which matches; build nothing on their order. -->

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
d2l.plot([b * np.arange(1, len(losses) + 1) for b, losses in runs.items()],
         [d2l.numpy(losses) for losses in runs.values()],
         'examples processed', 'loss', legend=[f'b = {b}' for b in runs],
         xscale='log')
```

<!-- WRITER: the plot is now one panel, loss against examples processed (log axis, every update from the first), so the whole descent from 0.5 at w = 0 is visible; the wall-clock panel is cut (SGD-13). "We plot the loss twice" above and "Measured in seconds, ..." below are false; the clock now lives in the printed seconds per epoch and milliseconds per update. What the plot shows in both tabs: the three stochastic runs fall together from 0.5 and reach the noise floor near 0.245 after about 500 examples, a third of one pass; gradient descent makes one update per 1,500 examples, stands at 0.29 after its first pass, reaches the stochastic runs after about three passes (4,500 examples) and then ends lowest (0.242), since it has no noise ball. Delete "but only because the dataset is tiny" and "even" (SGD-13). -->

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

<!-- WRITER: cells minibatch-sgd-concise-implementation-1/-2 (train_with_optimizer, #@save, deleted; the confirming run with torch.optim.SGD / optax.sgd) are cut (SGD-16; structure 3). Keep at most the remark about the factor 1/2 and the meaning of a learning rate, if anywhere. The "We can now" recap below and its three-question roadmap go (SGD-15); keep the last question as the ending. -->

Repeating the run with batch size 10 and learning rate 0.05 through the
framework's optimizer reproduces the from-scratch result.

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

<!-- WRITER: "which we measured over a factor of 500 in batch size" is false (the variance cells are cut); "the computation becomes limited by arithmetic rather than by memory and launch overhead" is no longer measured (batched computation cut); the new measured facts are the excess loss eta tr(Sigma)/(4b) on the airfoil regression, halved with eta, and its vanishing to rounding error when the model fits every example; about a millisecond per update at b = 1, 10 and 100. -->

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
   <!-- WRITER: the measurement this exercise cites (the MLP at random initialization in subsec_sgd-variance, its batch_grads and tr_Sigma) is cut. The exercise is kept (structure 3) but must now define the two-layer network and the per-example gradients itself, or move to the linear model of "Where the Model Stops" (tr Sigma there is 3.24 at the optimum and 0 for the exact-fit targets). -->
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
   <!-- WRITER: train_sgd now returns (losses after every update, seconds per epoch), not a history dict with examples, cumulative time and loss; its signature train_sgd(lr, batch_size, num_epochs=2) is unchanged. L = 2.10 is now printed in "Where the Model Stops", so part 2 "confirm the value quoted for the race" can say "compute L yourself from H". -->

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

<!-- WRITER: cell sgd-gradient-variance-and-batch-size is cut (SGD-7); this slide needs rebuilding without it, e.g. around the noise ball on data (cells under "Where the Model Stops"). -->
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

<!-- WRITER: cell sgd-batched-computation-2 is cut (SGD-10/11, to chap_performance); this slide and the two around it ("One Example per Call ...", "Check: Which Batch Keeps a Layer Busy?") go with it. -->

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

<!-- WRITER: cell minibatch-sgd-implementation-from-scratch-1 (the update-rule interface) is cut, with train_with_rule; this slide needs rebuilding or removal. -->

`train_with_rule` trains linear regression with any such rule and records the
loss against examples and training time.
:::

::: {.slide title="Only Minibatches Are Fast per Example and per Second"}
[Minibatch SGD on the airfoil data]{.kicker}

@!minibatch-sgd-implementation-from-scratch-7

<!-- WRITER: the plot is now one panel (loss against examples processed); the per-second facts are in the printout of minibatch-sgd-implementation-from-scratch-3 (about 1 ms per update for b = 1, 10, 100). -->

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
