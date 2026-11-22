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

To see how $\boldsymbol{\xi}$ interacts with curvature, we first define `sgd_2d` to add independent zero-mean, unit-variance Gaussian noise to both gradient coordinates:

```{.python .input #sgd-stochastic-gradient-updates-2}
%%tab pytorch, jax
def sgd_2d(x1, x2, s1, s2, f_grad, eta, rng):
    g1, g2 = f_grad(x1, x2)
    n1, n2 = rng.normal(size=2)  # Unit-variance gradient noise
    return x1 - eta * (g1 + n1), x2 - eta * (g2 + n2), 0, 0
```

We then run $300$ steps of `sgd_2d` on the valley $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ (:numref:`sec_optimization-intro`) at $\eta = 0.1$ from $(-5, -2)$:

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

To measure the empirical excess loss against this prediction, `sgd_runs` simulates `num_runs=400` vectorized minibatch SGD trajectories sampled with replacement, and `excess_loss` evaluates $\frac{1}{2}(\mathbf{w} - \mathbf{w}^\star)^\top \mathbf{H} (\mathbf{w} - \mathbf{w}^\star)$:

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

Starting at $\mathbf{w}^\star$ with batch size $b = 10$, we average `excess_loss` over the third epoch ($150$ steps) at $\eta \in \{0.05, 0.025\}$ and compare with $\eta\operatorname{tr}\boldsymbol{\Sigma}/(4b)$:

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

What makes minibatches essential is **hardware amortization**. Every parameter update incurs fixed Python dispatch and kernel-launch overhead independent of $b$, and vector processors (CPUs and GPUs) evaluate $b$ examples in parallel while reading the model weights from memory only once (:numref:`chap_performance`). To benchmark both wall-clock time per epoch and loss versus examples processed on the airfoil dataset, `train_sgd` runs minibatch SGD on linear regression from $\mathbf{w} = \mathbf{0}$ and returns the full-dataset loss after every update alongside the seconds per epoch:

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

We now execute `train_sgd` across $b \in \{1, 10, 100\}$ at fixed ratio $\eta / b = 0.005$, alongside full-batch gradient descent ($b = 1{,}500$ at $\eta = 0.9 < 2/L \approx 0.95$):

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

Plotting the training loss against total examples processed ($b \times \text{updates}$) overlays the sample efficiency of all four batch sizes:

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
[Dive into Deep Learning · §9.2]{.kicker}

Stochastic and Minibatch Gradient Descent<br>**sampled gradients · stationary noise ball · Robbins–Monro decay · minibatch variance · excess loss · hardware throughput**
:::
:::

::: {.slide title="Why Full-Batch Gradient Descent Is Prohibitive"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **Full-batch cost explosion:** Evaluating $\nabla f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x})$ over ImageNet ($n = 1.28\text{M}$) for $450{,}000$ updates would require $450{,}000$ full dataset passes ($5{,}000\times$ more compute than $90$ epochs at $b = 256$).
- **Minibatch efficiency ($b = 256$):** Estimates the downhill direction at $1/5{,}000$th the cost per step.
- **The fundamental trade-off:** Sampling $b \ll n$ examples injects **gradient noise** that traps constant-rate iterates in a stationary noise ball.
:::

::: {.col .fig .big}
@fig:ch09-opt-sgd-cost-vs-noise
:::
:::
:::

::: {.slide title="Unbiased Sampled Gradients and the Noise Decomposition"}
[Stochastic Gradients]{.kicker}

::: {.cols .vc}
::: {.col}
- **Unbiased estimator:** Drawing $i \sim \text{Uniform}\{1, \ldots, n\}$ gives $\mathbb{E}_i[\nabla f_i(\mathbf{x})] = \nabla f(\mathbf{x})$.
- **Signal-plus-noise decomposition:**
  $$\nabla f_i(\mathbf{x}_t) = \nabla f(\mathbf{x}_t) + \boldsymbol{\xi}_t, \qquad \mathbb{E}[\boldsymbol{\xi}_t \mid \mathbf{x}_t] = \mathbf{0}$$
- **Non-vanishing noise at $\mathbf{x}^\star$:** Even where $\nabla f(\mathbf{x}^\star) = \mathbf{0}$, individual gradients $\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$ whenever examples disagree, continuously kicking constant-rate iterates off $\mathbf{x}^\star$.
:::

::: {.col .fig}
@fig:ch09-opt-sgd-signal-plus-noise
:::
:::
:::

::: {.slide title="SGD on the Valley: Transient Descent into a Noise Cloud"}
[Valley Simulation]{.kicker}

::: {.cols .vc}
::: {.col}
- Adding independent unit-variance Gaussian noise $\xi_1, \xi_2 \sim \mathcal{N}(0, 1)$ to the valley $f(\mathbf{x}) = 0.1x_1^2 + 2x_2^2$ ($\lambda_1 = 0.2, \lambda_2 = 4.0$) at $\eta = 0.1$ over $300$ steps reveals two phases:
  1. **Transient phase (steps 1–100):** The deterministic gradient $\|\nabla f(\mathbf{x})\|$ dominates $\boldsymbol{\xi}$ and drives the iterate from $(-5, -2)$ toward $(0, 0)$.
  2. **Stationary phase (steps 100+):** Near the origin, $\|\nabla f(\mathbf{x})\|$ shrinks to the noise level, trapping the iterate in an elongated cloud stretched along the flat axis $x_1$.

@-sgd-stochastic-gradient-updates-2
:::

::: {.col .fig}
@!sgd-stochastic-gradient-updates-3
:::
:::
:::

::: {.slide title="Deriving the Stationary Variance of the Noise Ball"}
[Second-Moment Recursion]{.kicker}

::: {.cols .vc}
::: {.col}
- **1D noisy evolution ($x_{t+1} = (1 - \eta\lambda)x_t - \eta\xi_t$):** Squaring and taking expectations ($\mathbb{E}[x_t\xi_t] = 0$) separates deterministic contraction from noise injection:
  $$\mathbb{E}\big[x_{t+1}^2\big] = (1 - \eta\lambda)^2\,\mathbb{E}\big[x_t^2\big] + \eta^2 \sigma^2$$
- **Stationary balance ($0 < \eta < 2/\lambda$):** Setting $\mathbb{E}[x_{t+1}^2] = \mathbb{E}[x_t^2] = \mathbb{E}[x_\infty^2]$ yields the **noise ball**:
  $$\mathbb{E}\big[x_\infty^2\big] = \frac{\eta\,\sigma^2}{\lambda\,(2 - \eta\lambda)} \approx \frac{\eta\,\sigma^2}{2\lambda}$$
:::

::: {.col .fig}
@fig:ch09-opt-sgd-noise-equilibrium
:::
:::
:::

::: {.slide title="Three Laws Governing the Stationary Noise Ball"}
[Geometric Interpretation]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Stability ceiling preserved:** $(2 - \eta\lambda) \to 0$ as $\eta \to 2/\lambda$; variance explodes at the deterministic limit ($\eta < 0.5$ on the valley).
- **2. Inverse curvature scaling ($\propto 1/\lambda$):** Flat directions suffer both slower transient decay ($\eta\lambda$) and wider stationary fluctuations ($16\times$ higher variance along $\lambda_1 = 0.2$ than $\lambda_2 = 4.0$).
- **3. Linear $\eta/b$ scaling:** For $\eta\lambda \ll 1$, stationary variance scales as $\frac{\eta\sigma^2}{2\lambda b}$; halving $\eta$ or doubling $b$ halves the noise ball.
:::

::: {.col .fig .big}
@fig:ch09-opt-sgd-three-laws
:::
:::
:::

::: {.slide title="Simulating 10,000 Runs Confirms the Noise-Ball Formula"}
[Empirical Verification]{.kicker}

::: {.cols .vc}
::: {.col}
- We simulate $10{,}000$ independent SGD trajectories on the valley ($\lambda_1 = 0.2, \lambda_2 = 4.0, \sigma^2 = 1$) for $2{,}000$ steps at $\eta = 0.1$ and $\eta = 0.05$:
  - At $\eta = 0.1$: measured variances $(0.251, 0.0156)$ match $\frac{\eta}{\lambda(2-\eta\lambda)} = (0.2525, 0.0156)$ to within $1\%$.
  - At $\eta = 0.05$: flat variance halves to $0.1256$, while steep variance drops slightly more than half ($0.0156 \to 0.0069$) because $2 - \eta\lambda_2$ increases from $1.6 \to 1.8$.
:::

::: {.col .fig}
@!sgd-the-noise-ball
:::
:::
:::

::: {.slide title="Eliminating the Noise Floor: Robbins–Monro Conditions"}
[Decaying Step Sizes]{.kicker}

::: {.cols .vc}
::: {.col}
- **Robbins–Monro conditions (1951)** for exact convergence $\mathbf{x}_t \to \mathbf{x}^\star$ under noise:
  $$\sum_{t=1}^{\infty} \eta_t = \infty \qquad \text{and} \qquad \sum_{t=1}^{\infty} \eta_t^2 < \infty$$
- **Infinite travel budget ($\sum \eta_t = \infty$):** Reaches $\mathbf{x}^\star$ from any initialization (violated by geometric decay $\eta_0\gamma^t$, which freezes early).
- **Finite cumulative noise ($\sum \eta_t^2 < \infty$):** Contracts tail variance to zero (achieved by $\eta_t = \frac{c}{t+t_0}$ with $O(1/t)$ MSE decay).
:::

::: {.col .fig}
@fig:ch09-opt-sgd-robbins-monro
:::
:::
:::

::: {.slide title="Minibatch Averaging and the 1/b Variance Reduction Law"}
[Minibatch SGD]{.kicker}

Minibatch SGD averages the gradient across $b$ examples $\mathcal{B}_t$ drawn at step $t$:

$$\mathbf{g}_t = \frac{1}{b}\sum_{i \in \mathcal{B}_t} \nabla f_i(\mathbf{x}_t)$$

- **Per-example gradient covariance:** Let $\boldsymbol{\Sigma}(\mathbf{x}) = \mathbb{E}_i\big[(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))(\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x}))^\top\big]$ with total variance $\operatorname{tr}\boldsymbol{\Sigma} = \mathbb{E}_i\|\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})\|^2$.
- **Variance scales as $1/b$:** For $b$ independent draws with replacement, the covariance divides by $b$:
  $$\mathbb{E}\big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}_t)}{b}$$
- **The $\eta / b$ noise-ball invariance:** Substituting $\sigma^2 \leftarrow \sigma^2 / b$ into the noise ball gives $\mathbb{E}[x_\infty^2] \approx \frac{\eta}{b}\cdot\frac{\sigma^2}{2\lambda}$ for $\eta\lambda \ll 1$: **doubling $b$ and halving $\eta$ produce the exact same stationary noise ball**.
:::

::: {.slide title="Stationary Excess Loss: Why Curvature Cancels Out"}
[Where Training Stalls]{.kicker}

::: {.cols .vc}
::: {.col}
- **Coordinate variance vs. loss curvature:** Along eigenvector $k$ (curvature $\lambda_k$, noise variance $\sigma_k^2$), stationary parameter variance is $v_k \approx \frac{\eta\sigma_k^2}{2\lambda_k b}$.
- **Exact $\lambda_k$ cancellation:** Multiplying by $\frac{1}{2}\lambda_k$ in the quadratic loss cancels $\lambda_k$ in every mode: $\frac{1}{2}\lambda_k v_k = \frac{\eta\sigma_k^2}{4b}$.
- **Curvature-free total excess loss:** Summing over all $d$ modes ($\sum_k \sigma_k^2 = \operatorname{tr}\boldsymbol{\Sigma}$) gives:
  $$\mathbb{E}\big[f(\mathbf{x}_\infty)\big] - f(\mathbf{x}^\star) \approx \frac{\eta\,\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$$
:::

::: {.col .fig}
@fig:ch09-opt-sgd-curvature-cancel
:::
:::
:::

::: {.slide title="Testing Excess Loss on the NASA Airfoil Dataset"}
[Exact Data Measurements]{.kicker}

::: {.cols .vc}
::: {.col}
- On the standardized NASA airfoil self-noise dataset ($n = 1{,}500$ wind-tunnel examples, $5$ features $+ 1$ bias $= 6$ weights), linear regression with squared loss has an exact closed-form Hessian and minimizer $\mathbf{w}^\star$:
  $$\mathbf{H} = \frac{1}{n}\mathbf{X}^\top\mathbf{X}$$
- The largest eigenvalue is $L = \lambda_{\max}(\mathbf{H}) \approx 2.10$, and evaluating all $1{,}500$ per-example gradients $\mathbf{g}_i = (\mathbf{x}_i^\top\mathbf{w}^\star - y_i)\mathbf{x}_i$ at $\mathbf{w}^\star$ yields $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) \approx 3.24$.
:::

::: {.col .fig}
@!sgd-where-the-model-stops-1
:::
:::
:::

::: {.slide title="Measured Excess Loss Matches η tr Σ / (4b) on Data"}
[Theory vs. Experiment]{.kicker}

::: {.cols .vc}
::: {.col}
- At batch size $b = 10$ with $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{w}^\star) = 3.24$, our formula predicts:
  - At $\eta = 0.05$: $\frac{0.05 \times 3.24}{4 \times 10} = 0.0040$ above $f(\mathbf{w}^\star) \approx 0.242$.
  - At $\eta = 0.025$: $\frac{0.025 \times 3.24}{4 \times 10} = 0.0020$.
- Averaging across $400$ independent minibatch SGD runs over the third epoch confirms agreement to four decimal places!
:::

::: {.col .fig}
@!sgd-where-the-model-stops-3
:::
:::
:::

::: {.slide title="The Interpolating Regime vs. the Noise-Limited Regime"}
[When Does the Noise Ball Vanish?]{.kicker}

The excess-loss formula $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ reveals two fundamentally different training regimes depending on whether examples agree at $\mathbf{x}^\star$:

| Training Regime | Per-Example Gradients at $\mathbf{x}^\star$ | Noise Trace $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$ | Constant-Rate SGD Behavior | Deep Learning Example |
| :--- | :--- | :---: | :--- | :--- |
| **Underparameterized / Noisy-Label Regime** | Examples disagree ($\nabla f_i(\mathbf{x}^\star) \neq \mathbf{0}$, only $\frac{1}{n}\sum_i \nabla f_i(\mathbf{x}^\star) = \mathbf{0}$) | $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) > 0$ | Stalls at excess loss $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$; **requires LR decay $\eta_t \to 0$** | Airfoil regression; single-pass LLM pretraining on massive corpora |
| **Interpolating Regime** | Model fits every training example exactly ($\nabla f_i(\mathbf{x}^\star) = \mathbf{0}$ for all $i$) | $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star) = 0$ | Noise ball shrinks automatically as $\mathbf{x}_t \to \mathbf{x}^\star$; **converges geometrically** | Overparameterized classifiers trained for many epochs to zero loss |
:::

::: {.slide title="Sampling Without Replacement and Random Reshuffling"}
[Practical Data Loaders]{.kicker}

Real data loaders do not draw minibatches with replacement; they randomly permute the $n$ examples each epoch and slice disjoint minibatches of size $b$:

- **Finite-population variance correction:** Because the $n$ per-example gradient errors sum to zero ($\sum_{i=1}^n (\nabla f_i(\mathbf{x}) - \nabla f(\mathbf{x})) = \mathbf{0}$), distinct examples in a minibatch are slightly negatively correlated ($\mathbb{E}_{i \neq j}[\mathbf{e}_i^\top \mathbf{e}_j] = -\frac{\operatorname{tr}\boldsymbol{\Sigma}}{n-1}$), scaling the variance by:
  $$\mathbb{E}\big\|\mathbf{g}_t - \nabla f(\mathbf{x}_t)\big\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\cdot\frac{n - b}{n - 1}$$
- **Limiting checks:** At $b = 10, n = 1{,}500$, the factor is $0.994 \approx 1$; at $b = n$, the variance vanishes ($0$) as required for full-batch gradient descent.
- **Epoch-level cancellation:** Across one full pass of $n/b$ steps, random reshuffling forces the first-order gradient errors to sum to zero over the epoch, shrinking the stationary noise ball further at small $\eta$ (Mishchenko et al., 2020).
:::

::: {.slide title="Hardware Amortization: 100 Examples Cost the Same as 1"}
[Throughput Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
- Every optimizer update pays fixed Python dispatch, autograd graph, and kernel-launch overhead plus weight memory reads independent of $b$.
- On the $1{,}500$-example airfoil dataset at fixed $\eta / b = 0.005$:
  - $b = 1$ ($1{,}500$ updates/epoch) and $b = 100$ ($15$ updates/epoch) both take comparable sub-millisecond time *per update*, making $b = 100$ **$50\text{--}100\times$ faster per epoch**!
:::

::: {.col .fig}
@!minibatch-sgd-implementation-from-scratch-3
:::
:::
:::

::: {.slide title="Sample Efficiency and Where Linear Scaling η ∝ b Breaks"}
[Progress per Example]{.kicker}

::: {.cols .vc}
::: {.col}
- **Why small-batch curves overlap:** At fixed $\eta / b = 0.005$ and $\eta\lambda \ll 1$, $b$ steps at rate $\eta/b$ contract by $(1 - \eta\lambda/b)^b \approx 1 - \eta\lambda$, matching one step of size $\eta$ per example processed.
- **Breakdown near the curvature ceiling $2/L$:**
  - At $b = 100$, $\eta = 0.5$ gives $\eta L = 0.5 \times 2.10 = 1.05$, inflating the noise-ball denominator $(2 - \eta L)^{-1}$ and raising the last-epoch loss slightly ($0.248$ vs. $0.246$).
  - Beyond $b_{\max} = \frac{2/L}{0.005} \approx 190$, linear scaling violates $\eta < 2/L \approx 0.95$ and diverges!
:::

::: {.col .fig}
@!minibatch-sgd-implementation-from-scratch-7
:::
:::
:::

::: {.slide title="Concept Check: Noise Ball, Excess Loss, and Batch Size"}
[Check Your Understanding]{.kicker}

Calculate the outcome in each minibatch SGD scenario:

1. **Quadrupling the batch size:** In the small-step regime ($\eta\lambda \ll 1$), if you increase the batch size from $b = 16$ to $b = 64$ while holding $\eta$ fixed, by what factor does the stationary excess loss change? What if you also scale $\eta$ by $4\times$?
2. **Anisotropic noise ball vs. isotropic excess loss:** On the valley ($\lambda_1 = 0.2, \lambda_2 = 4.0$) with equal gradient noise variance $\sigma_1^2 = \sigma_2^2 = 1$, what is the ratio of stationary coordinate variances $\mathbb{E}[x_{\infty,1}^2] / \mathbb{E}[x_{\infty,2}^2]$ for small $\eta$, and what is the ratio of their contributions to the excess loss?
3. **Linear scaling ceiling:** If a loss surface has maximum curvature $L = 5.0$ and you tune $\eta = 0.02$ at $b = 8$ ($\eta/b = 0.0025$), at what batch size $b$ does linear scaling $\eta = 0.0025\,b$ hit the deterministic stability limit $2/L$?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Holding $\eta$ fixed reduces the excess loss $\frac{\eta\operatorname{tr}\boldsymbol{\Sigma}}{4b}$ by **$4\times$**; scaling $\eta$ by $4\times$ keeps $\eta / b$ constant, leaving the excess loss **unchanged** (while reaching the floor in $4\times$ fewer updates).
2. Coordinate variance scales as $1/\lambda$, so $\mathbb{E}[x_{\infty,1}^2]/\mathbb{E}[x_{\infty,2}^2] \approx 4.0 / 0.2 =$ **$20\times$ wider** along $x_1$; in the excess loss $\frac{1}{2}\lambda_k \mathbb{E}[x_{\infty,k}^2] \approx \frac{\eta\sigma_k^2}{4b}$, $\lambda_k$ cancels and both axes contribute **equally ($1:1$)**!
3. The stability limit is $2/L = 2/5.0 = 0.4$, which linear scaling reaches at **$b = 0.4 / 0.0025 = 160$ examples**.
:::
:::

::: {.slide title="Summary: Stochastic and Minibatch Gradient Descent"}
[Key Takeaways]{.kicker}

| Quantity / Law | Mathematical Formula | Scaling Dependence | Practical Implication |
| :--- | :--- | :--- | :--- |
| **Minibatch Gradient Variance** | $\mathbb{E}\|\mathbf{g}_t - \nabla f\|^2 = \frac{\operatorname{tr}\boldsymbol{\Sigma}}{b}\frac{n-b}{n-1}$ | Variance $\propto 1/b$; standard deviation $\propto 1/\sqrt{b}$ | $100\times$ larger batch cuts gradient error std by $10\times$ |
| **Stationary Noise Ball** | $\mathbb{E}[x_\infty^2] = \frac{\eta\sigma^2}{b\lambda(2 - \eta\lambda)} \approx \frac{\eta\sigma^2}{2\lambda b}$ | Scales with ratio $\eta/b$ and inversely with curvature $\lambda$ | Flat directions fluctuate over the widest parameter radius |
| **Stationary Excess Loss** | $\mathbb{E}[f(\mathbf{x}_\infty)] - f(\mathbf{x}^\star) \approx \frac{\eta\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)}{4b}$ | Curvature $\lambda_k$ cancels; proportional to $\eta/b$ and $\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)$ | Vanishes in interpolating models ($\operatorname{tr}\boldsymbol{\Sigma}(\mathbf{x}^\star)=0$); requires LR decay otherwise |
| **Robbins–Monro Conditions** | $\sum_t \eta_t = \infty,\; \sum_t \eta_t^2 < \infty$ | $\eta_t = c/(t+t_0)$ gives $\mathbb{E}\|\mathbf{x}_t - \mathbf{x}^\star\|^2 = O(1/t)$ | Infinite travel budget + finite cumulative noise variance |
| **Linear Scaling Regime** | $\eta(b) \propto b$ for $\eta L \ll 2$ | Same progress per example and same noise floor at fixed $\eta/b$ | Amortizes per-step overhead across parallel hardware up to $b_{\text{crit}}$ (§{sec_batch_size}) |
:::

