# Stochastic and Minibatch Gradient Descent
:label:`sec_sgd`

In :numref:`sec_gd`, every parameter update used the exact gradient of the objective, which requires evaluating the loss across the entire training set on every step. In deep learning, full-batch gradient descent is computationally prohibitive: training ResNet-50 on the $1.28$ million images of ImageNet for $90$ epochs with minibatches of $256$ images takes $450{,}000$ updates :cite:`Goyal.Dollar.Girshick.ea.2017`, whereas $450{,}000$ full-batch steps would require $450{,}000$ complete passes over the dataset—$5{,}000$ times as much computation.

Instead, **stochastic gradient descent** (SGD) estimates the gradient from a single randomly sampled example or a small **minibatch** of $b$ examples (:numref:`subsec_linear-regression-minibatch-sgd`). This estimate is unbiased, and its variance decreases as $1/b$, but the sampling noise fundamentally alters the late-stage behavior of optimization: with a constant learning rate $\eta$, iterates do not converge to a point minimum but fluctuate inside a stationary **noise ball**.

In this section, we combine gradient noise with uneven curvature. Using the two-dimensional valley of :numref:`sec_optimization-intro`, we derive the stationary variance of constant-step SGD along each curvature direction, test the resulting excess-loss prediction on a real regression dataset, examine how sampling without replacement and hardware parallelism shape minibatch selection, and motivate learning-rate decay (:numref:`sec_scheduler`) and momentum (:numref:`sec_momentum`).

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

### Sampled Gradients and the Valley

For an empirical risk $f(\mathbf{x}) = \frac{1}{n} \sum_{i=1}^n f_i(\mathbf{x})$ over $n$ training examples, the exact gradient is the average of the per-example gradients:

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Single-example SGD draws an index $i \in \{1, \ldots, n\}$ uniformly at random at each step and updates

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}).$$

Because $\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$, the stochastic gradient can be decomposed as $\nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}) + \boldsymbol{\xi}$, where the gradient noise $\boldsymbol{\xi}$ has mean zero.

To see how $\boldsymbol{\xi}$ interacts with curvature, we add independent zero-mean, unit-variance Gaussian noise to both gradient coordinates of the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ (:numref:`sec_optimization-intro`) and run $300$ steps at $\eta = 0.1$ from $(-5, -2)$:

```{.python .input #sgd-stochastic-gradient-updates-2}
%%tab pytorch, jax
def sgd_2d(x1, x2, s1, s2, f_grad, eta, rng):
    g1, g2 = f_grad(x1, x2)
    n1, n2 = rng.normal(size=2)  # Unit-variance gradient noise
    return x1 - eta * (g1 + n1), x2 - eta * (g2 + n2), 0, 0
```

```{.python .input #sgd-stochastic-gradient-updates-3}
%%tab pytorch, jax
rng = np.random.default_rng()
trace = d2l.train_2d(sgd_2d, d2l.valley_grad, steps=300, eta=0.1, rng=rng)
d2l.show_trace_2d(d2l.f_valley, trace)
```

During the first $\sim 100$ steps, the deterministic gradient dominates and the iterate approaches the origin. Once near the origin, $\|\nabla f(\mathbf{x})\|$ becomes small while the noise amplitude remains constant, trapping the iterate in an elongated cloud stretched along the flat axis $x_1$.

### The Noise Ball

We can compute the dimensions of this cloud analytically. Along an eigenvector with curvature $\lambda > 0$ and independent zero-mean gradient noise $\xi_t$ of variance $\sigma^2$, a single SGD step evolves according to

$$x_{t+1} = x_t - \eta\, (\lambda x_t + \xi_t) = (1 - \eta\lambda)\, x_t - \eta\, \xi_t.$$

Squaring both sides and taking expectations (using $\mathbb{E}[x_t \xi_t] = 0$) gives the second-moment recursion

$$\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\, \mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2.$$

For any stable step size $0 < \eta < 2/\lambda$, the mean $\mathbb{E}[x_t]$ decays to zero and the second moment converges to the stationary variance where deterministic contraction balances noise injection, $\mathbb{E}[x_\infty^2] = (1 - \eta\lambda)^2 \mathbb{E}[x_\infty^2] + \eta^2 \sigma^2$:

$$\mathbb{E}\big[x_\infty^2\big] = \frac{\eta\, \sigma^2}{\lambda\, (2 - \eta\lambda)} \approx \frac{\eta\, \sigma^2}{2\lambda} \quad \textrm{for } \eta\lambda \ll 1.$$
:eqlabel:`eq_sgd-noise-ball`

Equation :eqref:`eq_sgd-noise-ball` characterizes the **noise ball** (:numref:`subsec_mdl-stochastic-gradients` derives the full finite-time trajectory):
1. **Stability ceiling preserved:** As $\eta \to 2/\lambda$, the denominator $(2 - \eta\lambda) \to 0$ and the variance diverges.
2. **Inverse curvature scaling:** The stationary variance is inversely proportional to curvature $\lambda$. On the valley at $\eta = 0.1$ with $\sigma^2 = 1$, the stationary variance is $0.1 / (0.2 \times 1.98) \approx 0.2525$ along the flat coordinate $x_1$ ($\text{std} \approx 0.50$) and $0.1 / (4 \times 1.6) \approx 0.0156$ along the steep coordinate $x_2$ ($\text{std} \approx 0.125$). Flat directions suffer twice: they converge slowly during the transient phase and fluctuate widely at steady state.
3. **Linear step-size scaling:** For $\eta\lambda \ll 1$, halving the learning rate $\eta$ halves the stationary variance along every direction, as confirmed by simulating $10{,}000$ independent trajectories:

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

### Decreasing Learning Rates

Constant-step SGD faces a fundamental trade-off: a smaller learning rate $\eta$ shrinks the noise ball $\propto \eta$ but slows down initial convergence by reducing the per-step contraction rate $\eta\lambda$. Eliminating the noise floor requires decaying the learning rate $\eta_t$ over time. :citet:`Robbins.Monro.1951` established that for smooth, strongly convex objectives with bounded gradient variance, SGD converges to the exact minimizer if the step sizes satisfy

$$\sum_{t=1}^{\infty} \eta_t = \infty \qquad \textrm{and} \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty.$$
:eqlabel:`eq_sgd-robbins-monro`

The first condition ensures that the cumulative step budget is infinite so the iterate can reach the optimum from any initialization, while the second ensures that the cumulative injected noise variance $\propto \sum_t \eta_t^2$ remains finite. Polynomial schedules such as $\eta_t = c / (t + t_0)$ with $c > 1/(2\lambda_{\min})$ satisfy both conditions and reduce the expected squared error at rate $\mathbb{E}[\|\mathbf{x}_t - \mathbf{x}^\star\|^2] = O(1/t)$ (:numref:`subsec_mdl-step-size-decay`; Exercise :exref:`ex_sgd-one-over-t`). On a fixed training budget of $T$ steps, practical deep learning replaces asymptotic $1/t$ schedules with cosine, linear, or warmup–stable–decay schedules (:numref:`sec_scheduler`).

## Minibatches
:label:`sec_minibatch_sgd`

### Gradient Variance and Batch Size
:label:`subsec_sgd-variance`

Instead of a single example, minibatch SGD averages the gradient over a random subset $\mathcal{B}_t$ of $b$ examples:

$$\mathbf{g}_t = \frac{1}{b} \sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{x}_t).$$

Let $\boldsymbol{\Sigma}(\mathbf{x}) = \mathbb{E}_i\big[(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))^\top\big]$ denote the per-example gradient covariance matrix, whose trace $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$ is the total per-example gradient variance. For $b$ examples drawn independently with replacement, averaging divides the covariance by $b$:

$$\mathbb{E} \big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b}.$$
:eqlabel:`eq_sgd-variance`

Substituting $\sigma^2 \leftarrow \sigma^2 / b$ into :eqref:`eq_sgd-noise-ball` gives the minibatch stationary variance along curvature $\lambda$:

$$\mathbb{E}\big[x_\infty^2\big] = \frac{\eta\, \sigma^2}{b\,\lambda\, (2 - \eta\lambda)} \approx \frac{\eta}{b} \cdot \frac{\sigma^2}{2\lambda} \quad \textrm{for } \eta\lambda \ll 1.$$

In the small-step regime $\eta\lambda \ll 1$, the learning rate and batch size govern the stationary noise ball solely through their ratio **$\eta / b$**: doubling the batch size $b$ has the same effect on the noise floor as halving the learning rate $\eta$.

### The Excess Loss on Data
:label:`subsec_sgd-where-stops`

How much does fluctuation inside the noise ball raise the training loss above its minimum $f(\mathbf{x}^\star)$? Along the $k$-th eigenvector of the Hessian $\mathbf{H}$ (with eigenvalue $\lambda_k$ and per-example gradient variance $\sigma_k^2$), a stationary displacement of variance $v_k \approx \frac{\eta \sigma_k^2}{2 \lambda_k b}$ contributes $\frac{1}{2}\lambda_k v_k$ to the expected quadratic loss:

$$\frac{1}{2} \lambda_k \left(\frac{\eta\, \sigma_k^2}{2 \lambda_k b}\right) = \frac{\eta\, \sigma_k^2}{4b}.$$

Notice that the curvature $\lambda_k$ cancels: flat directions fluctuate across a wider radius ($\propto 1/\lambda_k$), but a unit displacement along a flat direction raises the loss less ($\propto \lambda_k$). Summing across all $d$ orthogonal eigenvectors ($\sum_{k=1}^d \sigma_k^2 = \operatorname{tr}\boldsymbol{\Sigma}$) yields the stationary **excess loss**:

$$\mathbb{E}\big[f(\mathbf{x}_\infty)\big] - f(\mathbf{x}^\star) \approx \frac{\eta \operatorname{tr} \boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}.$$
:eqlabel:`eq_sgd-excess-loss`

We test :eqref:`eq_sgd-excess-loss` on the NASA airfoil self-noise regression dataset ($n = 1{,}500$ wind-tunnel measurements with $5$ standardized features and $1$ standardized sound-pressure target):

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

For linear regression with squared loss (representing the bias as a $6$th weight on a constant feature), we compute the exact Hessian $\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$, the closed-form minimizer $\mathbf{w}^\star$, the largest curvature $L = \lambda_{\max}(\mathbf{H}) \approx 2.10$, and the per-example gradient variance $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) \approx 3.24$:

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

```{.python .input #sgd-where-the-model-stops-3}
%%tab pytorch, jax
rng = np.random.default_rng()
for eta in (0.05, 0.025):  # b = 10, three epochs of 150 steps from w_star
    trace = sgd_runs(X, y, w_star, eta, 10, 450, rng)
    measured = excess_loss(trace[-150:], w_star, H).mean()  # The last epoch
    print(f'eta={eta}: excess loss {measured:.4f}, '
          f'predicted {eta * tr_Sigma / (4 * 10):.4f}')
```

At $b = 10$, the measured excess losses (about $0.004$ at $\eta = 0.05$ and $0.002$ at $\eta = 0.025$) closely match the prediction $\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$. Notice what happens in the **interpolating regime**, where a model has enough capacity to fit every training example exactly so that $\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ for all $i$: in that limit, $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$, the noise ball vanishes as the iterate approaches $\mathbf{x}^\star$, and constant-step SGD converges geometrically to the exact minimizer. Overparameterized classifiers trained for many epochs approach this interpolating regime, whereas large language models trained for a single pass over massive corpora remain in the noise-limited regime where $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) > 0$.

### Sampling without Replacement

In practice, data loaders do not sample minibatches independently with replacement; instead, they randomly permute the dataset at the start of each epoch and partition it into disjoint minibatches of size $b$. Sampling $b$ distinct examples without replacement from a finite population of size $n$ introduces a negative correlation between per-example gradient errors (since $\sum_{i=1}^n (\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})) = \mathbf{0}$), multiplying the minibatch gradient variance :eqref:`eq_sgd-variance` by the finite-population correction factor (Exercise :exref:`ex_sgd-finite-population`):

$$\mathbb{E} \big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr} \boldsymbol{\Sigma}}{b} \cdot \frac{n - b}{n - 1}.$$
:eqlabel:`eq_sgd-finite-population`

When $b \ll n$ ($10$ out of $1{,}500$), this factor is nearly unity ($0.994$); when $b = n$, the variance vanishes as required for full-batch gradient descent. Across a full epoch, random reshuffling also forces the minibatch gradient errors over the $n/b$ steps of the epoch to cancel to first order, yielding a smaller stationary error than sampling with replacement at small learning rates :cite:`Mishchenko.Khaled.Richtarik.2020`.

### Batch Size and Hardware Throughput

When $\eta\lambda \ll 1$ and $\eta / b$ is held constant, a single step on a minibatch of size $b$ with learning rate $\eta$ contracts the expected error by $(1 - \eta\lambda)$ and lands in the same noise ball as $b$ single-example steps of learning rate $\eta / b$, since $(1 - \eta\lambda / b)^b \approx 1 - \eta\lambda$. Counted purely in per-example gradient evaluations, increasing $b$ within this linear regime neither speeds up nor slows down learning.

What makes minibatches essential is **hardware amortization**. Every parameter update incurs fixed Python dispatch and kernel-launch overhead independent of $b$, and vector processors (CPUs and GPUs) evaluate $b$ examples in parallel while reading the model weights from memory only once (:numref:`chap_performance`). We benchmark both wall-clock time per epoch and loss versus examples processed on the airfoil dataset across $b \in \{1, 10, 100\}$ at fixed $\eta / b = 0.005$, alongside full-batch gradient descent ($b = 1{,}500$ at $\eta = 0.9 < 2/L \approx 0.95$):

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

```{.python .input #minibatch-sgd-implementation-from-scratch-7}
%%tab pytorch, jax
d2l.plot([b * np.arange(1, len(losses) + 1) for b, losses in runs.items()],
         [d2l.numpy(losses) for losses in runs.values()],
         'examples processed', 'loss', legend=[f'b = {b}' for b in runs],
         xscale='log')
```

An update on $b = 100$ examples takes roughly the same wall-clock time as an update on $b = 1$, making the $b = 100$ epoch nearly $50\text{--}100\times$ faster while tracking the exact same loss curve per example processed. Notice also where linear scaling $\eta \propto b$ breaks down: holding $\eta / b = 0.005$ pushes the learning rate at $b = 100$ to $\eta = 0.5$, where $\eta L = 0.5 \times 2.10 = 1.05$ is no longer small compared to $2$ (inflating the denominator $(2 - \eta\lambda)^{-1}$ in :eqref:`eq_sgd-noise-ball`), and any batch size beyond $b \approx 190$ violates the stability ceiling $\eta < 2/L \approx 0.95$. We analyze this critical batch size transition systematically in :numref:`sec_batch_size`.

## Summary

* Minibatch SGD estimates the gradient from $b$ examples, reducing the gradient variance to $\operatorname{tr}(\boldsymbol{\Sigma})/b$ (with finite-population factor $(n-b)/(n-1)$ when sampling without replacement).
* With a constant learning rate $\eta$, SGD iterates converge to a stationary **noise ball** whose variance along a direction of curvature $\lambda$ is $\frac{\eta\sigma^2}{b\lambda(2 - \eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda b}$, widest along the flattest directions.
* Summing over all directions, the curvature cancels and the expected training loss plateaus at an **excess loss** of $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$, which vanishes only when the model interpolates every training example at the optimum.
* When $\eta L \ll 1$, trajectories depend on $\eta$ and $b$ through their ratio $\eta / b$. Minibatches amortize per-step dispatch and memory-bandwidth costs across $b$ examples while making the same progress per example processed up to the curvature stability limit $\eta < 2/L$.

## Exercises

1. **Noise as a random dataset.** Show that `sgd_2d` on $f(\mathbf{x}) = 0.1 x_1^2 + 2x_2^2$ with unit-variance Gaussian gradient noise is equivalent to single-example SGD on $f(\mathbf{x}, \mathbf{w}) = 0.1 (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ with $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \mathbf{C})$, and determine the covariance $\mathbf{C}$.
   :exlabel:`ex_sgd-noise-dataset`

1. [code] **The constant of a $1/t$ schedule.** ● For a 1D quadratic of curvature $\lambda$ under gradient noise of variance $\sigma^2$ and learning rate $\eta_t = c / (t + t_0)$:
   :exlabel:`ex_sgd-one-over-t`
    1. Write the exact recursion for $e_t = \mathbb{E}[x_t^2]$ and substitute the asymptotic ansatz $e_t \approx A/t$ to show that $A = c^2 \sigma^2 / (2 c \lambda - 1)$ whenever $c > 1/(2\lambda)$. Which value of $c$ minimizes $A$?
    1. Iterate the deterministic recursion for $e_t$ on $\lambda = 0.2, \sigma = 1$ from $e_0 = 25$ with $t_0 = 20$ and $c \in \{1, 2.5, 5, 10\}$ up to $t = 10^6$, plotting $t\,e_t$ versus $t$ on log–log axes, and explain why no single $c$ is simultaneously optimal for both $\lambda_1 = 0.2$ and $\lambda_2 = 4$.

1. [code] **The finite-population factor and reshuffling.**
    1. Let $\mathbf{e}_i = \nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})$. Using $\sum_{i=1}^n \mathbf{e}_i = \mathbf{0}$, show that $\mathbb{E}_{i \neq j}[\mathbf{e}_i^\top \mathbf{e}_j] = -\operatorname{tr}\boldsymbol{\Sigma}/(n - 1)$ and derive :eqref:`eq_sgd-finite-population`. Verify the formula empirically on the per-example gradients `G` of the airfoil dataset for $b \in \{10, 100, 500, 1000\}$ drawn with and without replacement.
       :exlabel:`ex_sgd-finite-population`
    1. Draw $n = 100$ vectors $\mathbf{w}_i$ from the distribution of Exercise :exref:`ex_sgd-noise-dataset` and compare 20-epoch SGD trajectories sampled with replacement versus per-epoch reshuffling at $\eta \in \{0.1, 0.01\}$.
       :exlabel:`ex_sgd-reshuffling`

1. [code] **Gradient variance during training.** ●
   :exlabel:`ex_sgd-variance-training`
    1. On the airfoil linear model, compare $\operatorname{tr}\boldsymbol{\Sigma}$ at $\mathbf{w} = \mathbf{0}$ versus $\mathbf{w} = \mathbf{w}^\star$.
    1. Train a two-layer ReLU MLP ($64$ hidden units) on the airfoil data for $10$ epochs ($b = 10, \eta = 0.05$) and compare $\operatorname{tr}\boldsymbol{\Sigma}$ (broken down across `W1`, `b1`, `W2`, `b2`) at initialization versus after training.

1. [code] **Batch size and learning rate scaling.** ● Fix $\eta = 0.05$ at $b = 10$ on the airfoil linear regression model and test $b \in \{1, 10, 100, 1500\}$ under both linear scaling ($\eta \propto b$) and square-root scaling ($\eta \propto \sqrt{b}$). Explain which configurations diverge using $L = 2.10$, and compute the exact stationary excess loss across all $6$ eigenvectors of `H` using :eqref:`eq_sgd-noise-ball` to explain why $b = 100, \eta = 0.5$ has a slightly higher floor than $b = 10, \eta = 0.05$.
   :exlabel:`ex_sgd-batch-lr`

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
