# Stochastic and Minibatch Gradient Descent
:label:`sec_sgd`

Every model since :numref:`sec_linear_regression` has been trained on
minibatches. At each step we drew a few dozen or a few hundred examples,
computed the gradient of their average loss, and stepped against it.
:numref:`sec_gd` did something we have never done in practice: it used the
exact gradient, which takes a pass over the whole training set for every
step.

Nobody trains that way, and a little arithmetic shows why. A common recipe
trains ResNet-50 on the 1.28 million images of ImageNet for 90 epochs in
minibatches of 256 images :cite:`Goyal.Dollar.Girshick.ea.2017`, which
comes to 450,000 parameter updates. The same number of updates with exact
gradients would take 450,000 passes over the data, 5,000 times the work.
Yet a few hundred randomly chosen images already indicate which way is
downhill. *Stochastic gradient descent* (SGD) takes their word for it.

What does the shortcut cost? Linear regression gave a first answer
(:numref:`subsec_linear-regression-minibatch-sgd`): a minibatch gradient
is right on average, and its variance falls in proportion to one over the
batch size. For a single weight we also saw the consequence. At a constant
learning rate the weight never settles but fluctuates in a cloud around
the minimizer, the noise ball of :numref:`fig_linreg_gd_sgd`. A single
weight has a single curvature, however, and the two previous sections
showed that what makes a loss hard is curvature that differs between
directions.

This section puts noise and curvature together. On the valley of
:numref:`sec_optimization-intro` we compute how far the cloud extends in
each direction, and find that it reaches farthest along the flat
direction, where gradient descent was already slowest. We then test the
formula on data, where it predicts the loss at which a training run
stalls, and meet a case in which the cloud disappears altogether. Two
remedies appear along the way, a learning rate that falls during training
and a larger batch. We introduce both here and leave their use on networks
to later sections of the chapter.

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

In deep learning the objective is an average of per-example losses. Let
$f_i(\mathbf{x})$ be the loss on example $i$ of $n$; as everywhere in this
chapter, $\mathbf{x}$ is the vector of parameters, and the data appear only
in the code, as `X` and `y`. The objective and its gradient are

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}), \qquad
\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

One exact gradient costs $n$ per-example gradients. SGD in its simplest
form draws a single index $i$ uniformly at random and steps along that
example's gradient,

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

where $\eta$ is the learning rate. Every index is equally likely, so the
expected value of the sampled gradient is the full gradient. The estimate
is *unbiased*. Being right on average is the easy half of the matter,
though. A single draw can point almost anywhere, since one example's
gradient pulls toward fitting that example alone. We write the sampled
gradient as the true gradient plus noise,
$\nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}) + \boldsymbol{\xi}$, where
$\boldsymbol{\xi}$ has mean zero. The rest of this section is about the
size of $\boldsymbol{\xi}$ and about what it does to the iterates.

### SGD on the Valley

Let us start with the simplest loss that has two different curvatures.
The valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ of
:numref:`sec_optimization-intro` curves gently along $x_1$, with curvature
$0.2$, and steeply along $x_2$, with curvature $4$. It has no examples to
sample from, so we simulate the sampling by adding independent Gaussian
noise of unit variance to both coordinates of the exact gradient.
(Exercise :exref:`ex_sgd-noise-dataset` shows that this is SGD on a dataset
whose examples are randomly shifted copies of the valley.) The update rule
`sgd_2d` is the `gd_2d` of :numref:`sec_gd` with noise added to the
gradient.

```{.python .input #sgd-stochastic-gradient-updates-2}
%%tab pytorch, jax
def sgd_2d(x1, x2, s1, s2, f_grad, eta, rng):
    g1, g2 = f_grad(x1, x2)
    n1, n2 = rng.normal(size=2)  # Unit-variance gradient noise
    return x1 - eta * (g1 + n1), x2 - eta * (g2 + n2), 0, 0
```

We run 300 steps at $\eta = 0.1$ from the starting point $(-5, -2)$ of the
earlier sections.

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
does not shrink the cloud. Near the minimum the true gradient vanishes, but
the noise we add does not, so every step still moves the parameters by
$\eta$ times pure noise.

### The Noise Ball

How far from the minimum does the cloud extend? On a quadratic the
question has an exact answer, and a short one. Each coordinate of the
valley evolves on its own, so take one coordinate with curvature $\lambda$
($0.2$ or $4$ here) and gradient noise $\xi_t$ of mean zero and variance
$\sigma^2$ ($1$ here), drawn independently at every step. That the noise
has the same variance wherever the iterate happens to be is a bold
assumption. Here it holds because we invented the noise ourselves, and on
data we shall have to check it. One SGD step reads

$$x_{t+1} = x_t - \eta\, (\lambda x_t + \xi_t) = (1 - \eta\lambda)\, x_t - \eta\, \xi_t.$$

The first term pulls the iterate toward the minimum at $0$, shrinking it by
the factor $1 - \eta\lambda$ as gradient descent would. The second term
pushes it in a random direction. To see where the two balance, we square
both sides and take expectations. The cross term vanishes because $\xi_t$
has mean zero and is independent of $x_t$, which leaves

$$\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\, \mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2.$$

Far from the minimum the first term dominates, and SGD makes progress like
gradient descent. Near it, the mean square $\mathbb{E}[x_t^2]$ settles at
the value $v$ at which contraction and injected noise cancel,
$v = (1 - \eta\lambda)^2 v + \eta^2 \sigma^2$. Solving for $v$ gives

$$\mathbb{E}\big[x_\infty^2\big] = \frac{\eta\, \sigma^2}{\lambda\, (2 - \eta\lambda)} \approx \frac{\eta\, \sigma^2}{2\lambda} \quad \textrm{for } \eta\lambda \ll 1,$$
:eqlabel:`eq_sgd-noise-ball`

where $x_\infty$ stands for the iterate after many steps. By then its mean
has decayed to zero, so the mean square is the variance of the iterate.
This is the *noise ball*, now with a radius for every direction. Its
squared radius grows in proportion to the learning rate and to the noise
variance, and it shrinks as the curvature grows. A marble in a bowl that
someone keeps shaking behaves the same way: the shallower the bowl, the
farther the marble strays from the bottom.

The dependence on the curvature explains the shape that the plot showed.
At $\eta = 0.1$ the formula gives a variance of
$0.1 / (0.2 \cdot 1.98) \approx 0.25$ along $x_1$ and
$0.1 / (4 \cdot 1.6) \approx 0.016$ along $x_2$, so the ball has standard
deviation $0.5$ along the flat direction and $0.125$ along the steep one,
a quarter of that. The flat directions, where gradient descent is slowest,
are also where the noise carries the iterate farthest, because the pull
back toward the minimum is weakest there.

The formula also answers the question on which :numref:`sec_gd` ended,
what an uncertain gradient leaves of the ceiling on the learning rate and
of the pace below it. It leaves both. As $\eta\lambda$ approaches $2$ the
denominator $2 - \eta\lambda$ vanishes and the ball grows without bound,
so the ceiling $2/\lambda$ stands where it stood. Below it the iterate
still approaches at the pace of gradient descent until it arrives at the
ball. What noise adds is a floor.

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
:eqref:`eq_sgd-noise-ball` says. Along $x_2$ the variance falls to
somewhat less than half, because there $\eta\lambda = 0.4$ is not small
and the exact formula applies. The correction looks like a detail here. It
will return on data, and larger. The appendix solves the recursion step by
step, which adds the time needed to reach the ball
(:numref:`subsec_mdl-stochastic-gradients`).

### Decreasing Learning Rates

A constant learning rate therefore forces a choice. Halving $\eta$ halves
the squared radius of the ball, but it also halves the fraction
$\eta\lambda$ by which each step closes the remaining distance, so reaching
the ball takes twice as many steps. Large steps make fast early progress;
small steps finish close. The way out is to change $\eta$ during training,
large at first and small at the end.

How fast may the rate fall? :citet:`Robbins.Monro.1951` gave the classical
answer for a sequence of rates $\eta_t$, in two conditions:

$$\sum_{t=1}^{\infty} \eta_t = \infty \qquad \textrm{and} \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty.$$
:eqlabel:`eq_sgd-robbins-monro`

The rates must fall, but not too fast. The first condition lets the
iterate travel arbitrarily far, so that no starting point is out of reach.
The second bounds the noise. Step $t$ injects noise of variance
$\eta_t^2 \sigma^2$, and as long as the rates stay below the ceiling
$2/\lambda$, later steps can only shrink it. The noise left at the end of
the run then has variance at most $\sigma^2 \sum_t \eta_t^2$. On a loss
that curves upward in every direction, though nowhere more sharply than
some bound, as the valley does, and with unbiased gradients of bounded
variance, the two conditions guarantee that SGD converges to the
minimizer. The rates $\eta_t = c/t$ satisfy both. A constant rate fails
the second condition, and a geometrically decaying rate the first.

Convergence under noise is slow, though. With $\eta_t = c/t$ and a large
enough constant $c$, the expected squared distance to the minimum falls in
proportion to $1/t$, so the distance itself falls only like $1/\sqrt{t}$.
The appendix derives this rate (:numref:`subsec_mdl-step-size-decay`), and
Exercise :exref:`ex_sgd-one-over-t` finds the threshold on $c$. Compare it
with gradient descent, which at its best rate came a hundred times closer
to the minimum of the valley every 47 steps
(:numref:`sec_optimization-intro`). SGD with the rates $c/t$ has to
multiply its number of steps by 10,000 to do the same. In return, each of
its steps costs one per-example gradient where gradient descent needs $n$,
and a training loss rarely needs to be minimized to many digits. For deep
networks the two conditions are a guide rather than a recipe, since a run
has a fixed budget of steps and cannot wait for a rate of $1/t$.
:numref:`sec_scheduler` develops the schedules that are used there.

## Minibatches
:label:`sec_minibatch_sgd`

A decaying learning rate shrinks the noise ball by taking smaller steps.
Averaging the gradient over several examples shrinks it by reducing the
noise itself, and every training loop in this book does so.

### Gradient Variance and Batch Size
:label:`subsec_sgd-variance`

A minibatch $\mathcal{B}_t$ of $b$ examples, drawn independently and
uniformly at random, gives the gradient estimate

$$\mathbf{g}_t = \frac{1}{b} \sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{x}_t).$$

It is unbiased because every term is, and its noise is smaller by the rule
we used for a single weight: the variance of an average of $b$ independent
draws is the variance of one draw divided by $b$. Nothing new happens when
the gradient is a vector. The rule holds in every coordinate, and we add
the coordinates up. For one example the sum is the expected squared
distance of its gradient from the full gradient,
$\operatorname{tr} \boldsymbol{\Sigma} = \mathbb{E}_i \|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$.
The notation says that this is the trace of the covariance matrix
$\boldsymbol{\Sigma}$ of a single example's gradient. For a minibatch it
is $b$ times smaller,

$$\mathbb{E} \big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$
:eqlabel:`eq_sgd-variance`

This is less than it sounds. The size of the noise falls only like
$1/\sqrt{b}$: a batch 100 times larger, at 100 times the computation per
step, makes the gradient error only 10 times smaller.

In the noise ball :eqref:`eq_sgd-noise-ball` the variance $\sigma^2$
becomes $\sigma^2 / b$, and the squared radius becomes about
$\eta \sigma^2 / (2\lambda b)$. The learning rate and the batch size enter
only through their ratio. We call this the *$\eta/b$ invariance*: as long
as $\eta\lambda$ is small, runs of SGD with the same $\eta/b$ fluctuate in
the same ball. Noise sets a floor, and $\eta/b$ sets its height. Halving
the learning rate and doubling the batch size shrink the ball by the same
factor, at different costs. The first doubles the number of steps needed
to reach the ball, the second doubles the work per step. Which to prefer
depends on how much the noise still limits the optimizer, a question that
we leave to :numref:`sec_batch_size`.

### The Excess Loss on Data
:label:`subsec_sgd-where-stops`

So far the noise was invented: Gaussian, of unit variance, the same at
every point. To test the formula on data we need a problem whose minimum is
known, so that the distance from it can be measured and not merely
estimated. Linear regression is such a problem. As data we take 1,500
wind-tunnel measurements of the sound that airfoils make, recorded by
NASA: five features and one target, the sound pressure level, each
standardized to mean zero and variance one. `get_airfoil` returns them as
a minibatch iterator that reshuffles the examples in every epoch, together
with the number of features.

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

We fit a linear model with the squared loss of
:numref:`sec_linear_regression`, the bias written as a sixth weight on a
constant feature. The loss is a quadratic in six weights, as the valley is
in two, so its Hessian `H` and its minimizer `w_star` have closed forms.
The next cell computes them, and from them the one number that our
prediction will need, $\operatorname{tr} \boldsymbol{\Sigma}$ at the
minimizer. The full gradient vanishes there, so
$\operatorname{tr} \boldsymbol{\Sigma}$ is the mean squared norm of the
per-example gradients: it measures how much the examples still disagree
about the weights when no better compromise exists. The cell also prints
the largest eigenvalue $L$ of the Hessian, which we set aside until the
end of the section.

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

Before running SGD we predict where its loss will stall. Along each
eigenvector of the Hessian the iterate obeys the recursion of the noise
ball on its own, as along the two coordinates of the valley. Take an
eigenvector with curvature $\lambda_k$, along which the gradient of a
single example has variance $\sigma_k^2$. With minibatches of $b$ examples
the iterate fluctuates in that direction with a variance of about
$\eta \sigma_k^2 / (2 \lambda_k b)$. A displacement of mean square $v$
along the direction raises the loss by $\frac{1}{2} \lambda_k v$ on
average, so the direction adds

$$\frac{\eta\, \sigma_k^2}{2 \lambda_k b} \cdot \frac{\lambda_k}{2} = \frac{\eta\, \sigma_k^2}{4b}$$

to the loss. The curvature has cancelled. A flat direction lets the
iterate wander farther, and a displacement there raises the loss less, by
the same factor. Surprisingly, then, the condition number, which decided
everything in the two previous sections, has no say in how far above its
minimum the loss stalls. It only decides where in parameter space the
iterate does its wandering.

It remains to add up the directions. A squared norm is the sum of the
squared components along any set of orthogonal axes, so the variances
$\sigma_k^2$ add up to $\operatorname{tr} \boldsymbol{\Sigma}$. The loss
of SGD should therefore stall at an *excess loss*, the loss above its
minimum, of about

$$\mathbb{E}\big[f(\mathbf{x}_t)\big] - f(\mathbf{x}^*) \approx \frac{\eta \operatorname{tr} \boldsymbol{\Sigma}}{4b},$$
:eqlabel:`eq_sgd-excess-loss`

where $\mathbf{x}^*$ is the minimizer, `w_star` in the code. With
$\eta = 0.05$ and $b = 10$ this is $0.05 \cdot 3.24 / 40 \approx 0.004$,
and half of that at $\eta = 0.025$.

`sgd_runs` runs minibatch SGD 400 times at once, drawing every minibatch
with replacement as the derivation assumes, and `excess_loss` evaluates
the loss above its minimum for every iterate.

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

All runs start at the minimizer, so that they measure the ball alone and
not the approach to it, and we average the excess loss over the last of
three epochs.

```{.python .input #sgd-where-the-model-stops-3}
%%tab pytorch, jax
rng = np.random.default_rng()
for eta in (0.05, 0.025):  # b = 10, three epochs of 150 steps from w_star
    trace = sgd_runs(X, y, w_star, eta, 10, 450, rng)
    measured = excess_loss(trace[-150:], w_star, H).mean()  # The last epoch
    print(f'eta={eta}: excess loss {measured:.4f}, '
          f'predicted {eta * tr_Sigma / (4 * 10):.4f}')
```

The loss stalls where the formula says, at an excess of about $0.004$, and
at half of that for half the learning rate. Note how little is at stake:
the minimum of the loss is about $0.24$, so SGD stalls within 2% of it.

The formula has a hidden assumption, the one we called bold when we made
it. We treated $\sigma^2$, and with it
$\operatorname{tr} \boldsymbol{\Sigma}$, as constants, but they describe
the point at which the iterate sits: the spread of the per-example
gradients there. What if the examples stop disagreeing at the minimum? We
can arrange this without touching the inputs, the Hessian, the learning
rate or the batch size, by replacing the targets with the predictions of
the best linear model, `y_fit = X @ w_star`. The model now fits every
example exactly, as a network with more parameters than examples can fit
its training set. Every per-example gradient vanishes at the minimizer,
and :eqref:`eq_sgd-excess-loss` predicts no ball at all. We start the runs
at zero weights and print the largest excess loss among the 400 runs.

```{.python .input #sgd-where-the-model-stops-4}
%%tab pytorch, jax
y_fit = X @ w_star  # Targets that the linear model fits exactly
trace = sgd_runs(X, y_fit, np.zeros_like(w_star), 0.05, 10, 3000, rng)
worst = excess_loss(trace[149::150], w_star, H).max(axis=1)  # After each epoch
print('exact fit, worst of 400 runs after 1, 5, 10, 20 epochs:',
      *(f'{worst[e - 1]:.0e}' for e in (1, 5, 10, 20)))
```

After one epoch the worst run is already below the $0.004$ at which the
noisy targets held SGD. After five epochs it stands below $10^{-7}$, and
after twenty at about $10^{-14}$. Single-precision arithmetic goes no
further, since it resolves the weights to about seven digits and the
excess loss is quadratic in their error. With the learning rate and the
batch size that stalled a moment ago, SGD now converges geometrically, by
a constant factor per epoch, as gradient descent does.

The noise ball is therefore as large as the disagreement between the
examples at the minimum. Where the examples pull in different directions,
the loss of constant-step SGD stalls at a height set by
$\eta \operatorname{tr} \boldsymbol{\Sigma} / b$. Where one parameter
vector satisfies them all, the noise dies away as the iterate approaches
it, and SGD converges. The networks of this book lie between the two
cases: a classifier trained until it fits its training set approaches the
second, and a language model that reads every sentence once stays in the
first, since no model predicts the next word with certainty. In the
formulas of this section, the noise variances $\sigma^2$ and
$\operatorname{tr} \boldsymbol{\Sigma}$ should therefore be read as those
near the point where the run ends (Exercise
:exref:`ex_sgd-variance-training` follows them through training).

### Sampling without Replacement

Derivation and experiment both drew the examples of a minibatch
independently, with replacement. Training loops do something else, and
so does the iterator that `get_airfoil` returns. They shuffle the dataset
at the start of every epoch and cut it into consecutive minibatches, so
that each example is used once per epoch. Does our analysis survive the
change? It does, and with a correction in our favor.

Within a minibatch the $b$ examples are now distinct, and the variance of
their average is

$$\mathbb{E} \big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b} \cdot \frac{n - b}{n - 1}.$$
:eqlabel:`eq_sgd-finite-population`

The reason takes one line. The errors
$\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})$ of the $n$ examples sum to
zero, so the errors of two distinct examples point, on average, slightly
against each other (Exercise :exref:`ex_sgd-finite-population` fills in
the algebra). The factor is close to 1 while the batch is a small part of
the dataset, $0.994$ for 10 of our 1,500 examples, and it reaches 0 at
$b = n$, where the gradient is exact.

Across the minibatches of an epoch the cancellation is stronger. Evaluated
at a fixed point, the errors of all the minibatches of one epoch sum to
zero, since together they contain every example once. The draws are then
no longer independent, as our derivation assumed, but the dependence works
as it does within a batch. Errors cancel that independent draws would
leave standing. For small learning rates on losses that curve upward in
every direction, SGD with reshuffling provably ends closer to the minimum
than SGD with replacement :cite:`Mishchenko.Khaled.Richtarik.2020`. We
therefore keep the formulas of this section for training loops that
reshuffle, as estimates that err on the high side. Exercise
:exref:`ex_sgd-reshuffling` measures by how much.

### Batch Size and the Cost of an Update

Statistics alone would not make minibatches the default. A step on $b$
examples costs $b$ per-example gradients, as many as $b$ steps on single
examples. If the single steps use a $b$ times smaller learning rate, they
end in the same ball. Do they also get as far? Along a direction of
curvature $\lambda$ they shrink the distance to the minimum by
$(1 - \eta\lambda/b)^b$, the one large step shrinks it by
$1 - \eta\lambda$, and the two factors agree as long as $\eta\lambda$ is
small. Counted in per-example gradients, a minibatch gains nothing.

What it gains is time. Every update has a cost that does not depend on the
number of examples it carries, and a minibatch divides that cost among its
$b$ examples. We measure both, the progress per example and the time, on
the airfoil regression, with batches of 1, 10 and 100 examples and with
all 1,500, which is gradient descent.

`train_sgd` trains the linear model from zero weights with plain minibatch
SGD and returns the training loss after every update and the seconds per
epoch. The time is taken by `d2l.Benchmark`
(:numref:`subsec_async_dispatch`), which waits for the device to finish
before it stops the clock.

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

The three stochastic runs share $\eta/b = 0.005$ and train for two epochs.
The invariance predicts one ball for them, at an excess loss of
$0.005 \cdot 3.24 / 4 \approx 0.004$, provided that their steps are small
against the curvature. One of the three will strain that proviso.
Gradient descent has no noise to fear and runs for ten epochs of one
update each at $\eta = 0.9$, just below its ceiling. The largest curvature
was printed above as $L = 2.10$, so the ceiling $2/L$ is about $0.95$.

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

Consider the clock first. An update takes about a millisecond, and one
that carries 100 examples takes at most about twice as long as one that
carries a single example. The time goes mostly into fixed costs, fetching
a minibatch and launching a handful of operations from Python, as in the
loop that :numref:`sec_linear_regression` timed against a matrix product.
A pass over the data therefore takes roughly as long as its number of
updates. Single-example SGD makes a hundred times as many of them as the
batch of 100, and it needs about fifty to a hundred times as long for the
same 1,500 per-example gradients.

On a large model the arithmetic itself takes time, and batches still win.
:numref:`chap_performance` works out why: a batch reads the weights from
memory once and uses them for every example. When the batch one wants does
not fit in memory, its gradient can be accumulated over smaller pieces
before a single update (:numref:`subsec_mp-accumulation`).

Gradient descent looks better on this clock than it should. A full
gradient over 1,500 examples takes no longer than a few minibatch updates,
so these data cannot show gradient descent losing time. The arithmetic of
the opening can: on ImageNet one exact gradient costs as much as 5,000
minibatch steps.

The clock says how well a method uses the machine. To see how well it
uses the data, we plot the loss against the number of examples processed.

```{.python .input #minibatch-sgd-implementation-from-scratch-7}
%%tab pytorch, jax
d2l.plot([b * np.arange(1, len(losses) + 1) for b, losses in runs.items()],
         [d2l.numpy(losses) for losses in runs.values()],
         'examples processed', 'loss', legend=[f'b = {b}' for b in runs],
         xscale='log')
```

Per example, the three stochastic runs trace one curve. They fall together
from the loss of about $0.5$ at zero weights, which is half the variance
of the standardized target, and reach their floor before the first pass
over the data is complete. One step on $b$ examples has moved the weights
about as far as $b$ steps on one. Gradient descent makes one update per
pass. After its first pass it stands at $0.29$, a loss that the stochastic
runs passed after a few hundred examples. It draws level with them after
about three passes and ends lowest, at $0.242$, because it has no noise
ball.

The stochastic runs end above it, with printed losses between $0.244$ and
about $0.25$. For the batches of 1 and 10 that is the predicted excess of
$0.004$, to within a thousandth or two. The batch of 100 is the run that
strains the proviso. It shares the ratio $\eta/b$, but at $\eta = 0.5$ the
steepest direction of these data has $\eta\lambda = 0.5 \cdot 2.10 = 1.05$,
which is not small. The exact formula :eqref:`eq_sgd-noise-ball` then
divides by $2 - \eta\lambda = 0.95$ where the approximation divides by
$2$, the correction that we met along $x_2$ of the valley. There it
enlarged the ball by a quarter. Here it doubles the ball along the
steepest direction, and summed over the six directions it raises the
floor of the batch of 100 from $0.004$ to about $0.006$. A single run
cannot quite show the difference, because the last epoch of the batch of
100 consists of 15 updates, and their average loss moves by a few
thousandths from one run to the next.

The two extremes each waste something. Single-example SGD learns as much
from every example as any batch here and spends most of its time on fixed
costs; gradient descent uses the machine well and takes one step per
pass. A minibatch gives up neither. The tie per example cannot last for
ever, however, and the batch of 100 has shown where it begins to fray.
Holding $\eta/b$ at $0.005$ drives $\eta$ toward the ceiling $2/L$, which
a batch of about 190 examples reaches on these data. Beyond the ceiling a
larger batch can no longer take a proportionally larger step. Exercise
:exref:`ex_sgd-batch-lr` measures both the floor of the batch of 100 and
the runs beyond the ceiling.

Neither remedy of this section touches the curvature. A smaller $\eta/b$
shrinks the noise ball by the same factor in every direction and leaves it
widest along the flat ones, where the approach was slowest to begin with.
The flat directions of an ill-conditioned loss are thus bad twice over,
slow on the way down and loose on arrival. SGD cannot treat them
differently from the steep ones, because it has one learning rate for all
of them. Momentum is the first method that moves faster along them
(:numref:`sec_momentum`). Before we come to it, we build two small
networks on which such methods can be tested.

## Summary

Stochastic gradient descent replaces the exact gradient, which costs a
pass over all $n$ examples, by an unbiased estimate from a minibatch of
$b$. Noise leaves the ceiling on the learning rate where it was and adds a
floor: at a constant learning rate $\eta$ the iterates fluctuate in a
noise ball of variance about $\eta\sigma^2/(2\lambda b)$ in a direction of
curvature $\lambda$, widest where the loss is flattest. The loss then
stalls about $\eta \operatorname{tr}\boldsymbol{\Sigma}/(4b)$ above its
minimum, whatever the curvatures, and not at all when the model fits every
example, because the ball is as large as the disagreement between the
examples at the minimum. While $\eta$ is small against $2/L$, the learning
rate and the batch size enter only through $\eta/b$; a smaller rate costs
steps, a larger batch costs computation per step, and rates that decay
like $1/t$ remove the ball at a speed of $1/t$. An update on 100 examples
took at most about twice as long as an update on one and made the same
progress per example, which is why training uses minibatches.

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
1. [code] **Per-coordinate learning rates.** ● The noise ball
   :eqref:`eq_sgd-noise-ball` applies to each coordinate of the valley
   separately, with curvatures $\lambda_1 = 0.2$ and $\lambda_2 = 4$.
    1. Give each coordinate its own learning rate $\eta_j = \eta /
       \lambda_j$, the choice that made gradient descent equally fast in
       both coordinates (:numref:`sec_gd`). Derive the contraction factor
       per step and the stationary variance of each coordinate. What has
       the change done to the noise ball along the flat coordinate?
    1. Implement the per-coordinate rates in `sgd_2d` and in `noise_ball`
       with $\eta = 0.1$, and verify both predictions numerically.
    1. Why does this choice need information that a single stochastic
       gradient does not supply?
1. [code] **The constant of a $1/t$ schedule.** ● Take one coordinate with
   curvature $\lambda$ and gradient noise of variance $\sigma^2$, and the
   learning rates $\eta_t = c / (t + t_0)$, where the shift $t_0 \geq 0$
   keeps the first steps small.
   :exlabel:`ex_sgd-one-over-t`
    1. Write down the recursion for the mean square
       $e_t = \mathbb{E}[x_t^2]$, following the derivation
       of :eqref:`eq_sgd-noise-ball`.
    1. For large $t$, substitute $e_t \approx A/t$, expand both sides to
       order $1/t^2$, and show that
       $A = c^2 \sigma^2 / (2 c \lambda - 1)$. Which values of $c$ does
       this solution require, and which $c$ makes $A$ smallest?
    1. Iterate the recursion (no sampling is needed) for the flat
       coordinate of the valley, $\lambda = 0.2$ and $\sigma = 1$, from
       $e_0 = 25$ with $t_0 = 20$ and $c \in \{1, 2.5, 5, 10\}$, up to
       $t = 10^6$. Plot $t\, e_t$ against $t$ on logarithmic axes. Which
       runs approach the constant $A$ of the previous part, what slope
       does the run with $c = 1$ have, and what happens at $c = 2.5$,
       where the formula for $A$ breaks down?
    1. The steep coordinate has $\lambda = 4$. Which $c$ is best for it,
       and why can no single $c$ be best for both coordinates?
1. **From noise ball to sampler.** ● Where the examples disagree at the
   minimum, constant-rate SGD does not converge; its iterates fluctuate in
   the noise ball :eqref:`eq_sgd-noise-ball`, whose size is set by the
   rate. Stochastic gradient Langevin dynamics
   :cite:`Welling.Teh.2011` adds Gaussian noise to every step so that the
   iterates approximately sample the density proportional to $e^{-f}$:
   $$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\, \mathbf{g}_t + \sqrt{2\eta}\,
   \boldsymbol{\zeta}_t, \qquad \boldsymbol{\zeta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$
   where $\mathbf{g}_t$ is a stochastic gradient of $f$.
    1. For $f(x) = \frac{\lambda}{2}x^2$ with the exact gradient, derive
       the stationary variance of $x_t$ and compare it with the variance
       $1/\lambda$ of the density proportional to $e^{-f(x)}$.
    1. Compare the injected noise $\sqrt{2\eta}\, \zeta_t$ with the
       gradient-noise term $\eta\, \xi_t$ of SGD. For which gradient-noise
       variance $\sigma^2$ do the two contributions have equal variance,
       and which of them dominates as $\eta$ decays?
1. [code] **The finite-population factor.** Let
   $\mathbf{e}_i = \nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})$ be the
   error of example $i$, so that
   $\operatorname{tr} \boldsymbol{\Sigma} = \frac{1}{n} \sum_i \|\mathbf{e}_i\|^2$.
   :exlabel:`ex_sgd-finite-population`
    1. Show that the $n$ errors sum to zero, and deduce that two distinct
       examples drawn uniformly at random satisfy
       $\mathbb{E}\big[\mathbf{e}_i^\top \mathbf{e}_j\big] = -\operatorname{tr} \boldsymbol{\Sigma} / (n - 1)$.
    1. The squared norm of the sum of $b$ distinct errors has $b$ terms
       $\|\mathbf{e}_i\|^2$ and $b(b - 1)$ cross terms. Take expectations
       and derive :eqref:`eq_sgd-finite-population`.
    1. With the per-example gradients `G` of the airfoil regression,
       estimate the mean squared error of the minibatch gradient from
       2,000 minibatches each of $b \in \{10, 100, 500, 1000\}$ examples,
       drawn with and without replacement. Compare the eight estimates
       with :eqref:`eq_sgd-variance` and with
       the finite-population formula :eqref:`eq_sgd-finite-population`.
1. [code] **Reshuffling versus replacement.** ● Draw a dataset of
   $n = 100$ vectors $\mathbf{w}_i$ from the distribution of Exercise
   :exref:`ex_sgd-noise-dataset` and run SGD on the resulting finite sum
   for 20 epochs from the starting point $(-5, -2)$, once sampling $i$
   with replacement and once reshuffling the dataset before each epoch.
   Record the distance to the minimizer of the finite sum after each
   epoch, averaged over 20 runs, at the constant learning rates
   $\eta = 0.1$ and $\eta = 0.01$. At which learning rate does reshuffling
   make the larger difference, and why?
   :exlabel:`ex_sgd-reshuffling`
1. [code] **Gradient variance during training.** ● The text evaluated
   $\operatorname{tr}\boldsymbol{\Sigma}$ at the minimizer of a linear
   model. During training it changes.
   :exlabel:`ex_sgd-variance-training`
    1. For the linear model, compute
       $\operatorname{tr}\boldsymbol{\Sigma}$ at zero weights from `X` and
       `y`, and compare it with its value at `w_star`. The per-example
       gradient is the feature vector times the residual of the example;
       use this to explain the direction of the change.
    1. Define a two-layer MLP with 64 ReLU hidden units for the airfoil
       data, with weights `W1` and `W2` drawn from a normal distribution
       with standard deviation $0.1$ and biases `b1` and `b2` at zero.
       Train it with the squared loss for ten epochs of minibatch SGD
       with batch size 10 and learning rate $0.05$. At initialization and
       again after training, compute the 1,500 per-example gradients (a
       loop over the examples is fast enough) and from them
       $\operatorname{tr}\boldsymbol{\Sigma}$.
    1. Split $\operatorname{tr}\boldsymbol{\Sigma}$ into the contributions
       of `W1`, `b1`, `W2`, and `b2`, and explain the changes in the `b2`
       and `W1` contributions from the form of the per-example gradient of
       the squared loss.
1. **Duplicated data.** Suppose every example in the training set is
   silently duplicated, so that the dataset doubles in size while its
   empirical distribution is unchanged. For SGD with $b = 1$, for minibatch
   SGD at a fixed $b$, and for full-batch gradient descent, state whether
   the cost per step, the number of steps to reach a given loss, and the
   variance of the gradient estimate change, and explain each answer.
1. [code] **Batch size and learning rate together.** ● `train_sgd` trains
   the linear model at a chosen learning rate and batch size and returns
   the loss after every update and the seconds per epoch. Fix
   $\eta = 0.05$ at $b = 10$ and derive the learning rate for each
   $b \in \{1, 10, 100, 1500\}$ by two rules, $\eta \propto \sqrt{b}$ and
   $\eta \propto b$.
   :exlabel:`ex_sgd-batch-lr`
    1. Train each configuration for two epochs. Report the time per epoch
       and the mean loss over the last epoch of each run, and name the
       runs that diverge.
    1. Gradient descent on a quadratic is stable only for $\eta < 2/L$
       (:numref:`sec_gd`), and the text printed $L = 2.10$ for these data.
       Use this to explain which runs diverge.
    1. At $b = 1$, which rule reaches the lower loss? Explain the ordering
       with :eqref:`eq_sgd-excess-loss`.
    1. The rule $\eta \propto b$ keeps $\eta/b = 0.005$, yet the text
       claims a higher floor for $b = 100$ than for $b = 1$ and $b = 10$.
       Compute the eigenvectors of `H` and the variance of the per-example
       gradients `G` along each of them, evaluate the exact formula
       :eqref:`eq_sgd-noise-ball` in every direction with $\sigma^2/b$ in
       place of $\sigma^2$, and add up the contributions to the excess
       loss for the three batch sizes. Then measure the three floors with
       `sgd_runs`, started at `w_star`. Which direction contributes most
       of the difference?

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

Stochastic and Minibatch Gradient Descent<br>**sampled gradients · the noise ball · minibatches · where a run stalls**
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

- derive the cloud in which constant-step SGD ends on a quadratic;
- predict how learning rate $\eta$, curvature $\lambda$, noise variance
  $\sigma^2$ and batch size $b$ change it;
- predict the loss at which a training run stalls, or that it will not;
- compare batch sizes per example and per second.

**Idea:** noise adds a floor. Constant-step SGD fluctuates in a *noise
ball* of squared radius $\eta\sigma^2/(2\lambda b)$.

**Running example:** the valley $0.1x_1^2 + 2x_2^2$ with unit gradient
noise, then linear regression on 1,500 wind-tunnel measurements of
airfoil noise (5 features, 6 weights).
:::

::: {.slide title="Noise Is the Second Problem of the Chapter"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| part | decks |
|---|---|
| **1 · Two problems:** uneven curvature, noisy gradients | Landscapes · Gradient Descent · **Stochastic and Minibatch Gradient Descent** |
| **2 · Direction:** which way to step | Two Test Models · Momentum · Adam · AdamW · Muon |
| **3 · The run:** step size over time, noise level, model width | Schedules · Batch Size · Scaling Up · Practice |
:::

::: {.col .narrow}
- **Uses:** gradient descent on a quadratic, its ceiling $\eta < 2/L$
  ($L$: largest curvature).
- **Skips:** convergence proofs, schedules for networks, the
  memory-traffic analysis of batching on large models.
:::
:::
:::

::: {.slide title="A Sampled Gradient Is Right on Average"}
[Sampled gradients]{.kicker}

$$f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n f_i(\mathbf{x}), \qquad
\mathbf{x} \leftarrow \mathbf{x} - \eta\, \nabla f_i(\mathbf{x}),
\quad i \sim \textrm{uniform}$$

- $\mathbf{x}$: parameters; $f_i$: loss on example $i$ of $n$; $\eta$:
  learning rate.
- One step costs one per-example gradient instead of $n$.
- Unbiased: $\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$.
  That is the easy half.
- Noisy: $\nabla f_i = \nabla f + \boldsymbol{\xi}$ with
  $\mathbb{E}[\boldsymbol{\xi}] = \mathbf{0}$; one example pulls toward
  fitting itself alone.
:::

::: {.slide title="On the Valley, SGD Is Gradient Descent plus Unit Noise"}
[Sampled gradients]{.kicker}

The valley $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$: curvature 0.2 along $x_1$
(flat), 4 along $x_2$ (steep). Sampling is simulated by Gaussian noise of
variance 1 on both gradient coordinates:

@-sgd-stochastic-gradient-updates-2

::: {.d2l-note}
Same start $(-5, -2)$ and same update as gradient descent; only the
gradient is noisy. (`s1`, `s2` and the two returned zeros are state slots
that this update does not use.)
:::
:::

::: {.slide title="Constant-Step SGD Stalls in a Cloud"}
[Sampled gradients]{.kicker}

The valley, $\eta = 0.1$, 300 steps:

@!sgd-stochastic-gradient-updates-3

The cloud does not shrink with more steps, and it is longest along the
flat $x_1$.
:::

::: {.slide title="Contraction and Noise Balance at a Fixed Variance"}
[The noise ball]{.kicker}

One coordinate, curvature $\lambda$, noise $\xi_t$ of variance $\sigma^2$,
the same at every point (a bold assumption):

I. $\;x_{t+1} = (1-\eta\lambda)\,x_t - \eta\,\xi_t$

II. $\;\mathbb{E}[x_{t+1}^2] = (1-\eta\lambda)^2\,\mathbb{E}[x_t^2] + \eta^2\sigma^2$
(the cross term has mean zero)

III. Fixed point $v = (1-\eta\lambda)^2 v + \eta^2\sigma^2$, with
$x_\infty$ the iterate after many steps:

$$\boxed{\mathbb{E}[x_\infty^2] = \frac{\eta\sigma^2}{\lambda(2-\eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda}}$$

The squared radius grows with $\eta$ and $\sigma^2$ and shrinks with $\lambda$.
:::

::: {.slide title="Noise Keeps the Ceiling 2/λ and Adds a Floor"}
[The noise ball]{.kicker}

$$\mathbb{E}[x_\infty^2] = \frac{\eta\sigma^2}{\lambda(2-\eta\lambda)}$$

| gradient descent | with noise |
|:--|:--|
| ceiling $\eta < 2/\lambda$ | unchanged: the ball grows without bound as $\eta\lambda \to 2$ |
| factor $1 - \eta\lambda$ per step | unchanged, until the iterate reaches the ball |
| converges to the minimum | stalls at variance $\approx \eta\sigma^2/(2\lambda)$ |

::: {.d2l-note .rule}
What noise adds is a floor.
:::
:::

::: {.slide title="The Ball Is Four Times Wider along the Flat Direction"}
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

::: {.d2l-note}
Agreement within a few percent, the sampling error of 10,000 runs.
:::
:::

::: {.slide title="Check: Halve the Learning Rate"}
[Check]{.kicker}

On the valley, $\eta$ drops from 0.1 to 0.05. What happens to the variance
along $x_1$ and along $x_2$, and to the number of steps needed to reach
the ball?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $x_1$: $0.25 \to 0.126$, half, since $\eta\lambda$ is small.
$x_2$: $0.0156 \to 0.0069$, somewhat less than half, since
$\eta\lambda = 0.4$ is not small and the exact formula applies. Each step
closes the fraction $\eta\lambda$ of the distance, so reaching the ball
takes twice as many steps.
:::
:::
:::

::: {.slide title="Decreasing Rates Converge if Their Sum Diverges and Their Squares Do Not"}
[Decreasing learning rates]{.kicker}

$$\sum_t \eta_t = \infty \qquad \textrm{and} \qquad \sum_t \eta_t^2 < \infty$$

- First: the iterate can travel arbitrarily far.
- Second: with every rate below the ceiling $2/\lambda$, the noise left
  at the end has variance at most $\sigma^2 \sum_t \eta_t^2$.
- $\eta_t = c/t$ satisfies both; a constant rate fails the second,
  geometric decay the first.

[Robbins and Monro, 1951; curvature positive and bounded in every direction, noise of bounded variance]{.small}
:::

::: {.slide title="Noise Turns a Geometric Rate into 1/t"}
[Decreasing learning rates]{.kicker}

Distance to the minimum of the valley after $t$ steps:

| method | squared distance | distance | 100 times closer costs |
|:--|:--|:--|:--|
| gradient descent, best rate $\eta = 0.476$ | $\propto 0.905^{2t}$ | $\propto 0.905^{t}$ | 47 more steps |
| SGD, $\eta_t = c/t$, $c$ large enough | $\propto 1/t$ | $\propto 1/\sqrt{t}$ | 10,000 times the steps |

::: {.d2l-note}
In return an SGD step costs one per-example gradient, a step of gradient
descent $n$, and a training loss rarely needs many digits.
:::
:::

::: {.slide title="Averaging b Examples Divides the Variance by b"}
[Minibatches]{.kicker}

$$\mathbf{g}_t = \frac{1}{b}\sum_{i\in\mathcal{B}_t} \nabla f_i(\mathbf{x}_t),
\qquad
\mathbb{E}\,\|\mathbf{g}_t - \nabla f\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}$$

- $\mathcal{B}_t$: $b$ examples drawn independently.
- $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i - \nabla f\|^2$:
  squared error of one example's gradient, summed over the coordinates.

::: {.d2l-note}
Variance $\propto 1/b$, so noise $\propto 1/\sqrt{b}$: a batch 100 times
larger makes the gradient error only 10 times smaller.
:::
:::

::: {.slide title="The Ball Depends on η and b Only through η/b"}
[Minibatches]{.kicker}

$$\sigma^2 \to \frac{\sigma^2}{b}: \qquad
\mathbb{E}[x_\infty^2] \approx \frac{\eta\,\sigma^2}{2\lambda b}$$

| to halve the squared radius | cost |
|:--|:--|
| halve $\eta$ | twice the steps to reach the ball |
| double $b$ | twice the work per step |

::: {.d2l-note .rule}
**The $\eta/b$ invariance.** For small $\eta\lambda$, runs with the same
$\eta/b$ fluctuate in the same ball.
:::
:::

::: {.slide title="The Loss Stalls η tr Σ / (4b) above Its Minimum"}
[Excess loss on data]{.kicker}

Quadratic loss, eigenvector $k$ of the Hessian: curvature $\lambda_k$,
gradient variance $\sigma_k^2$ per example.

I. Variance of the iterate: $\;\eta \sigma_k^2/(2\lambda_k b)$

II. Loss of a displacement of mean square $v$: $\;\tfrac12\lambda_k v$

III. Product: $\eta \sigma_k^2/(4b)$. The curvature cancels, so the
condition number does not enter. Sum over $k$, with
$\sum_k \sigma_k^2 = \operatorname{tr}\boldsymbol{\Sigma}$:

$$\boxed{\mathbb{E}[f(\mathbf{x}_t)] - f(\mathbf{x}^*) \approx \frac{\eta \operatorname{tr}\boldsymbol{\Sigma}}{4b}}$$

$\mathbf{x}^*$: the minimizer; the left side is the *excess loss*.
:::

::: {.slide title="Linear Regression Stalls 0.004 above Its Minimum of 0.24, as Predicted"}
[Excess loss on data]{.kicker}

Airfoil regression (1,500 examples, 6 weights); Hessian and minimizer in
closed form. $L$: largest curvature;
$\operatorname{tr}\boldsymbol{\Sigma}$ at the minimizer:

@!sgd-where-the-model-stops-1

Prediction at $\eta = 0.05$, $b = 10$:
$0.05 \cdot 3.24/40 \approx 0.004$; half at $\eta = 0.025$. Measured,
400 runs started at the minimizer:

@!sgd-where-the-model-stops-3

SGD stalls within 2% of the minimum.
:::

::: {.slide title="Check: What if the Model Fits Every Example?"}
[Check]{.kicker}

Replace the targets by the predictions of the best linear model. Inputs,
Hessian, $\eta = 0.05$ and $b = 10$ stay the same. Where does the loss of
constant-step SGD stall now?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Every per-example gradient vanishes at the minimizer, so
$\operatorname{tr}\boldsymbol{\Sigma} = 0$ there and the formula predicts
no ball: SGD should converge.
:::
:::
:::

::: {.slide title="With an Exact Fit, Constant-Step SGD Converges"}
[Excess loss on data]{.kicker}

Largest excess loss among 400 runs from zero weights, same $\eta$ and $b$:

@!sgd-where-the-model-stops-4

With noisy targets the same settings stalled at 0.004.

::: {.d2l-note}
The ball is as large as the disagreement between the examples at the
minimum. Networks lie between the two cases.
:::
:::

::: {.slide title="Distinct Examples Shrink the Variance by (n − b)/(n − 1)"}
[Sampling without replacement]{.kicker}

$$\mathbb{E}\,\|\mathbf{g}_t - \nabla f\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\cdot\frac{n-b}{n-1}$$

- The $n$ errors $\nabla f_i - \nabla f$ sum to zero, so two distinct
  examples partly cancel.
- $b = 10$, $n = 1{,}500$: factor 0.994. $b = n$: factor 0, the gradient
  is exact.

::: {.d2l-note}
Reshuffling every epoch cancels more: the draws are no longer
independent, and for small $\eta$ the ball is smaller than the formulas
say.
:::

[Mishchenko, Khaled and Richtárik, 2020]{.small}
:::

::: {.slide title="An Update Costs about a Millisecond with 1 or with 100 Examples"}
[Batch size and cost]{.kicker}

Airfoil regression, $\eta/b = 0.005$ for $b = 1, 10, 100$; gradient
descent ($b = 1{,}500$) at $\eta = 0.9$:

@!minibatch-sgd-implementation-from-scratch-3

::: {.d2l-note}
The time goes mostly into fixed costs per update, so a pass at $b = 1$
takes about 50 to 100 times as long as at $b = 100$. A full gradient on
1,500 examples costs only a few updates.
:::
:::

::: {.slide title="Per Example the Stochastic Runs Tie and Gradient Descent Lags"}
[Batch size and cost]{.kicker}

@!minibatch-sgd-implementation-from-scratch-7

- Equal $\eta/b$: $b$ steps at rate $\eta/b$ shrink a direction by
  $(1 - \eta\lambda/b)^b \approx 1 - \eta\lambda$, as one step at $\eta$
  does.
- Gradient descent: one update per pass, level after about three passes,
  then lowest (0.242, no ball).
:::

::: {.slide title="Check: How Far Does the Tie Extend?"}
[Check]{.kicker}

The stochastic runs keep $\eta/b = 0.005$, and $L = 2.10$. What is
$\eta\lambda$ along the steepest direction at $b = 100$, and what does it
do to the ball there? At which batch size does $\eta$ reach the ceiling
$2/L = 0.95$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\eta\lambda = 0.5 \cdot 2.10 = 1.05$, not small: the exact
formula divides by $2 - \eta\lambda = 0.95$ instead of 2, so the ball is
twice as large in that direction, and the floor rises from 0.004 to about
0.006. The ceiling is reached at $0.95/0.005 = 190$ examples.
:::
:::
:::

::: {.slide title="Noise Sets a Floor, and η/b Sets Its Height"}
[Wrap-up]{.kicker}

- Noise ball $\eta\sigma^2/(2\lambda b)$: on the valley at $\eta = 0.1$,
  standard deviation 0.5 along $x_1$, 0.125 along $x_2$.
- Excess loss $\eta \operatorname{tr}\boldsymbol{\Sigma}/(4b)$, whatever
  the curvatures: 0.004 on the airfoil regression, zero when the model
  fits every example.
- Rates $c/t$ remove the ball at a speed of $1/t$.
- Equal $\eta/b$, small $\eta\lambda$: same progress per example at about
  a millisecond per update, with 1 or with 100 examples.

**Next:** two small networks to test optimizers on, then momentum for the
flat directions.
:::
