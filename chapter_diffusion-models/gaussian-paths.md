# Gaussian Paths: Diffusion and Flow Matching
:label:`sec_diffusion-gaussian-paths`

(to be written)

```{.python .input #gaussian-paths-gaussian-paths-diffusion-and-flow-matching}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input #gaussian-paths-gaussian-paths-diffusion-and-flow-matching}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import math
import jax
from jax import numpy as jnp
from flax import nnx
import optax
```

## One Path from Data to Noise

(to be written)

### Signal, Noise and the Log-SNR Clock

(to be written)

`GaussianPath` is the schedule as an object: two functions $\alpha_t$ and $\sigma_t$ on the clock $t \in [0, 1]$, the log-SNR $\lambda_t$, and a draw of $\mathbf{x}_t$; `vp`, `linear` and `cosine` build the three paths of the text.

```{.python .input #gaussian-paths-signal-noise-and-the-log-snr-clock}
%%tab pytorch
#@save
class GaussianPath:
    """x_t = alpha(t) x0 + sigma(t) eps, data at t = 0 and noise at t = 1."""
    def __init__(self, alpha, sigma):
        self.alpha, self.sigma = alpha, sigma

    @staticmethod
    def vp(beta_min=0.1, beta_max=20.0):
        """Variance preserving, Ho et al.'s linear beta in continuous time."""
        log_abar = lambda t: -(beta_min * t + (beta_max - beta_min) * t**2 / 2)
        return GaussianPath(lambda t: torch.exp(log_abar(t) / 2),
                            lambda t: torch.sqrt(-torch.expm1(log_abar(t))))

    @staticmethod
    def linear():
        """The straight line of flow matching and rectified flow."""
        return GaussianPath(lambda t: 1 - t, lambda t: t)

    @staticmethod
    def cosine(s=0.008):
        """Variance preserving, Nichol and Dhariwal's cosine schedule."""
        f = lambda t: (torch.cos(math.pi / 2 * (t + s) / (1 + s))
                       / math.cos(math.pi / 2 * s / (1 + s))).clamp(min=0)
        return GaussianPath(f, lambda t: torch.sqrt(1 - f(t) ** 2))

    def log_snr(self, t):
        """lambda_t = log(alpha_t^2 / sigma_t^2), falling from +inf at t = 0."""
        return 2 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))

    def sample(self, x0, t, eps):
        """x_t for per-example times t of shape (n,)."""
        t = t.reshape(-1, *[1] * (x0.dim() - 1))
        return self.alpha(t) * x0 + self.sigma(t) * eps
```

```{.python .input #gaussian-paths-signal-noise-and-the-log-snr-clock}
%%tab jax
#@save
class GaussianPath:
    """x_t = alpha(t) x0 + sigma(t) eps, data at t = 0 and noise at t = 1."""
    def __init__(self, alpha, sigma):
        self.alpha, self.sigma = alpha, sigma

    @staticmethod
    def vp(beta_min=0.1, beta_max=20.0):
        """Variance preserving, Ho et al.'s linear beta in continuous time."""
        log_abar = lambda t: -(beta_min * t + (beta_max - beta_min) * t**2 / 2)
        return GaussianPath(lambda t: jnp.exp(log_abar(t) / 2),
                            lambda t: jnp.sqrt(-jnp.expm1(log_abar(t))))

    @staticmethod
    def linear():
        """The straight line of flow matching and rectified flow."""
        return GaussianPath(lambda t: 1 - t, lambda t: t)

    @staticmethod
    def cosine(s=0.008):
        """Variance preserving, Nichol and Dhariwal's cosine schedule."""
        f = lambda t: jnp.maximum(jnp.cos(math.pi / 2 * (t + s) / (1 + s))
                                  / math.cos(math.pi / 2 * s / (1 + s)), 0)
        return GaussianPath(f, lambda t: jnp.sqrt(1 - f(t) ** 2))

    def log_snr(self, t):
        """lambda_t = log(alpha_t^2 / sigma_t^2), falling from +inf at t = 0."""
        return 2 * (jnp.log(self.alpha(t)) - jnp.log(self.sigma(t)))

    def sample(self, x0, t, eps):
        """x_t for per-example times t of shape (n,)."""
        t = t.reshape(-1, *[1] * (x0.ndim - 1))
        return self.alpha(t) * x0 + self.sigma(t) * eps
```

### The Chain View

(to be written)

`grid` evaluates the path on the chain's grid $t_k = k/T$, from noise at $t = 1$ down to data at $t = 0$; every sampler steps along it.

```{.python .input #gaussian-paths-the-chain-view}
%%tab pytorch
#@save
@d2l.add_to_class(GaussianPath)
def grid(self, steps):
    """Lists of alpha and sigma at t = 1, 1 - 1/steps, ..., 0."""
    t = torch.linspace(1, 0, steps + 1, dtype=torch.float64)
    return self.alpha(t).tolist(), self.sigma(t).tolist()
```

```{.python .input #gaussian-paths-the-chain-view}
%%tab jax
#@save
@d2l.add_to_class(GaussianPath)
def grid(self, steps):
    """Arrays of alpha and sigma at t = 1, 1 - 1/steps, ..., 0."""
    with jax.ensure_compile_time_eval():         # constants, even under jit
        t = jnp.linspace(1, 0, steps + 1)
        return self.alpha(t), self.sigma(t)
```

### Why Many Levels

(to be written)

## One Regression

(to be written)

### Conditional Targets and Marginal Fields

(to be written)

### Four Targets, One Oracle

(to be written)

The dictionary as code: `target` gives the regression target of each kind (`'eps'`, `'x0'`, `'v'`, `'score'`, `'velocity'`), and `x0_eps` turns a prediction of any kind back into the pair $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$.

```{.python .input #gaussian-paths-four-targets-one-oracle}
%%tab pytorch
#@save
@d2l.add_to_class(GaussianPath)
def rates(self, t):
    """The time derivatives (alpha'(t), sigma'(t)), by forward-mode AD."""
    ones = torch.ones_like(t)
    return (torch.func.jvp(self.alpha, (t,), (ones,))[1],
            torch.func.jvp(self.sigma, (t,), (ones,))[1])

#@save
@d2l.add_to_class(GaussianPath)
def target(self, kind, x0, eps, t):
    """The regression target of a kind: eps, x0, v, score or velocity."""
    t = t.reshape(-1, *[1] * (x0.dim() - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return eps
    if kind == 'x0': return x0
    if kind == 'v': return a * eps - s * x0
    if kind == 'score': return -eps / s
    da, ds = self.rates(t)                      # velocity = d x_t / dt
    return da * x0 + ds * eps

#@save
@d2l.add_to_class(GaussianPath)
def x0_eps(self, kind, out, x, t):
    """The (x0, eps) that a prediction `out` of a kind implies at (x, t)."""
    t = t.reshape(-1, *[1] * (x.dim() - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return (x - s * out) / a, out
    if kind == 'x0': return out, (x - a * out) / s
    if kind == 'v': return ((a * x - s * out) / (a**2 + s**2),
                            (s * x + a * out) / (a**2 + s**2))
    if kind == 'score': return (x + s ** 2 * out) / a, -s * out
    da, ds = self.rates(t)
    det = a * ds - s * da
    return (ds * x - s * out) / det, (a * out - da * x) / det

```

```{.python .input #gaussian-paths-four-targets-one-oracle}
%%tab jax
#@save
@d2l.add_to_class(GaussianPath)
def rates(self, t):
    """The time derivatives (alpha'(t), sigma'(t)), by forward-mode AD."""
    ones = jnp.ones_like(t)
    return (jax.jvp(self.alpha, (t,), (ones,))[1],
            jax.jvp(self.sigma, (t,), (ones,))[1])

#@save
@d2l.add_to_class(GaussianPath)
def target(self, kind, x0, eps, t):
    """The regression target of a kind: eps, x0, v, score or velocity."""
    t = t.reshape(-1, *[1] * (x0.ndim - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return eps
    if kind == 'x0': return x0
    if kind == 'v': return a * eps - s * x0
    if kind == 'score': return -eps / s
    da, ds = self.rates(t)                      # velocity = d x_t / dt
    return da * x0 + ds * eps

#@save
@d2l.add_to_class(GaussianPath)
def x0_eps(self, kind, out, x, t):
    """The (x0, eps) that a prediction `out` of a kind implies at (x, t)."""
    t = t.reshape(-1, *[1] * (x.ndim - 1))
    a, s = self.alpha(t), self.sigma(t)
    if kind == 'eps': return (x - s * out) / a, out
    if kind == 'x0': return out, (x - a * out) / s
    if kind == 'v': return ((a * x - s * out) / (a**2 + s**2),
                            (s * x + a * out) / (a**2 + s**2))
    if kind == 'score': return (x + s ** 2 * out) / a, -s * out
    da, ds = self.rates(t)
    det = a * ds - s * da
    return (ds * x - s * out) / det, (a * out - da * x) / det

```

`make_pred` wraps a trained network as that pair at any $(\alpha, \sigma)$: it queries the network at the time of its own path with the same signal-to-noise ratio, which `time_at` finds by bisection on the log-SNR clock.

```{.python .input #gaussian-paths-four-targets-one-oracle-1}
%%tab pytorch
#@save
@d2l.add_to_class(GaussianPath)
def time_at(self, log_snr):
    """The time with a given log-SNR, by bisection (lambda_t falls in t)."""
    lo, hi = torch.zeros_like(log_snr), torch.ones_like(log_snr)
    for _ in range(50):
        mid = (lo + hi) / 2
        later = self.log_snr(mid) > log_snr      # the answer lies after mid
        lo, hi = torch.where(later, mid, lo), torch.where(later, hi, mid)
    return (lo + hi) / 2

#@save
def make_pred(net, path, kind):
    """pred(x, alpha, sigma) -> (x0, eps) from net(x, t) trained to predict
    `kind` on `path`, for x = alpha x0 + sigma eps on any Gaussian path."""
    def pred(x, alpha, sigma):
        log_snr = 2 * math.log(alpha / sigma) if alpha > 0 else -math.inf
        t = path.time_at(torch.tensor(log_snr, dtype=torch.float64))
        x = x * (path.sigma(t).item() / sigma)  # the same point on our path
        t = torch.full((len(x),), t.item(), device=x.device)
        return path.x0_eps(kind, net(x, t), x, t)
    return pred
```

```{.python .input #gaussian-paths-four-targets-one-oracle-1}
%%tab jax
#@save
@d2l.add_to_class(GaussianPath)
def time_at(self, log_snr):
    """The time with a given log-SNR, by bisection (lambda_t falls in t)."""
    def halve(_, interval):
        lo, hi = interval
        mid = (lo + hi) / 2
        later = self.log_snr(mid) > log_snr      # the answer lies after mid
        return jnp.where(later, mid, lo), jnp.where(later, hi, mid)
    lo, hi = jax.lax.fori_loop(0, 50, halve, (jnp.zeros_like(log_snr),
                                              jnp.ones_like(log_snr)))
    return (lo + hi) / 2

#@save
def make_pred(net, path, kind):
    """pred(x, alpha, sigma) -> (x0, eps) from net(x, t) trained to predict
    `kind` on `path`, for x = alpha x0 + sigma eps on any Gaussian path."""
    def pred(x, alpha, sigma):
        log_snr = jnp.where(alpha > 0, 2 * jnp.log(alpha / sigma), -jnp.inf)
        t = path.time_at(jnp.asarray(log_snr, jnp.float32))
        x = x * (path.sigma(t) / sigma)         # the same point on our path
        t = jnp.full((len(x),), t)
        return path.x0_eps(kind, net(x, t), x, t)
    return pred
```

### The Loss Floor

(to be written)

## Sampling by Integration

(to be written)

### The Probability-Flow ODE and Euler's Method

(to be written)

`ode_sample` integrates the probability-flow ODE $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$ in the coordinates $\bar{\mathbf{x}} = \mathbf{x}/\alpha$, $\bar\sigma = \sigma/\alpha$, with Euler's method or Heun's, on the path's uniform grid in $t$ or on Karras et al.'s grid ($\rho = 7$, $\bar\sigma$ from 80 to 0.002).

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab pytorch
#@save
@torch.no_grad()
def ode_sample(pred, x, path, steps, method='euler', grid='uniform'):
    """Euler or Heun on d x_bar / d sigma_bar = eps from noise x to data."""
    if grid == 'uniform':
        r = [s / a if a > 0 else math.inf for a, s in zip(*path.grid(steps))]
    else:                                     # Karras et al.'s rho = 7 grid
        r = [(80 ** (1 / 7) + i / (steps - 1) * (0.002 ** (1 / 7)
              - 80 ** (1 / 7))) ** 7 for i in range(steps)] + [0.0]
    def at(xb, r):                            # pred at x = alpha x_bar
        if math.isinf(r):                     # here x_bar holds the noise
            return pred(xb, 0.0, 1.0)
        a = 1 / math.sqrt(1 + r ** 2)
        return pred(xb * a, a, r * a)
    xb = x if math.isinf(r[0]) else x * r[0]
    for r_t, r_s in zip(r[:-1], r[1:]):
        x0, eps = at(xb, r_t)
        x_next = x0 + r_s * eps               # the Euler (= DDIM) step
        if method == 'heun' and r_s > 0 and not math.isinf(r_t):
            x_next = xb + (r_s - r_t) * (eps + at(x_next, r_s)[1]) / 2
        xb = x_next
    return xb
```

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab jax
#@save
def ode_sample(pred, x, path, steps, method='euler', grid='uniform'):
    """Euler or Heun on d x_bar / d sigma_bar = eps from noise x to data."""
    with jax.ensure_compile_time_eval():      # the grid is a constant
        if grid == 'uniform':
            a, s = path.grid(steps)
            r = s / a
        else:                                 # Karras et al.'s rho = 7 grid
            i = jnp.arange(steps) / (steps - 1)
            r = jnp.append((80 ** (1 / 7) + i * (0.002 ** (1 / 7)
                            - 80 ** (1 / 7))) ** 7, 0.0)
        from_noise = bool(jnp.isinf(r[0]))
    xb = x if from_noise else x * r[0]
    r_t, r_s = r[:-1], r[1:]
    finite = jnp.isfinite(r_t)                # x_bar holds the noise at inf
    q = jnp.where(finite, 1 / jnp.sqrt(1 + r_t ** 2), 1.0)
    alpha, sigma = jnp.where(finite, q, 0.0), jnp.where(finite, r_t * q, 1.0)
    heun = (method == 'heun') & finite & (r_s > 0)
    a_s = 1 / jnp.sqrt(1 + r_s ** 2)
    def step(xb, inputs):
        q, alpha, sigma, r_s, a_s, heun, dr = inputs
        x0, eps = pred(xb * q, alpha, sigma)
        x_next = x0 + r_s * eps               # the Euler (= DDIM) step
        correct = lambda _: xb + dr * (eps + pred(x_next * a_s, a_s,
                                                  r_s * a_s)[1]) / 2
        return jax.lax.cond(heun, correct, lambda _: x_next, None), None
    dr = jnp.where(heun, r_s - jnp.where(finite, r_t, 0.0), 0.0)
    return jax.lax.scan(step, xb, (q, alpha, sigma, r_s, a_s, heun, dr))[0]
```

### The Reverse SDE and the Ancestral Chain

(to be written)

`ancestral_sample` is Ho et al.'s sampler for any predictor `pred(x, alpha, sigma) -> (x0, eps)`: each step moves to the mean of $q(\mathbf{x}_s \mid \mathbf{x}_t, \hat{\mathbf{x}}_0)$ and adds fresh noise with the variance $\sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ of the forward step from $s$ to $t$ (which is $\beta_t$ on a variance-preserving path).

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain}
%%tab pytorch
#@save
@torch.no_grad()
def ancestral_sample(pred, x, path, steps):
    """Ho et al.'s sampler from noise x at t = 1 to data at t = 0."""
    a, s = path.grid(steps)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        rho = s[i + 1] * a[i] / (a[i + 1] * s[i])
        x = a[i + 1] * x0 + s[i + 1] * rho * eps        # the posterior mean
        if s[i + 1] > 0:                                # no noise at t = 0
            var = s[i] ** 2 - (a[i] / a[i + 1] * s[i + 1]) ** 2  # forward step
            x = x + math.sqrt(var) * torch.randn_like(x)
    return x
```

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain}
%%tab jax
#@save
def ancestral_sample(pred, x, path, steps, key):
    """Ho et al.'s sampler from noise x at t = 1 to data at t = 0."""
    a, s = path.grid(steps)
    a_t, s_t, a_s, s_s = a[:-1], s[:-1], a[1:], s[1:]
    rho = s_s * a_t / (a_s * s_t)
    var = jnp.maximum(s_t ** 2 - (a_t / a_s * s_s) ** 2, 0)  # forward step
    noise = jnp.where(s_s > 0, jnp.sqrt(var), 0)             # none at t = 0
    def step(x, inputs):
        a_t, s_t, a_s, c_eps, c_noise, key = inputs
        x0, eps = pred(x, a_t, s_t)
        x = a_s * x0 + c_eps * eps                      # the posterior mean
        return x + c_noise * jax.random.normal(key, x.shape), None
    return jax.lax.scan(step, x, (a_t, s_t, a_s, s_s * rho, noise,
                                  jax.random.split(key, steps)))[0]
```

## Experiments on the Running Example

(to be written)

`train_regression` is the one training loop of the chapter: Adam on a loss of the model and a fresh batch, constant or cosine-decayed learning rate, and `d2l.EMA` from :numref:`sec_dcgan` on the weights.

```{.python .input #gaussian-paths-experiments-on-the-running-example}
%%tab pytorch
#@save
def train_regression(model, loss_fn, batch_fn, num_steps, *, lr=1e-3,
                cosine=False, ema_half_life=1000):
    """Adam on loss_fn(model, batch_fn()); returns the model, its EMA copy
    and the loss of every step."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_steps)
    ema, losses = d2l.EMA(model, ema_half_life), []
    for _ in range(num_steps):
        loss = loss_fn(model, batch_fn())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if cosine:
            sched.step()
        ema.update(model)
        losses.append(loss.detach())
    return model, ema.model, torch.stack(losses)
```

```{.python .input #gaussian-paths-experiments-on-the-running-example}
%%tab jax
#@save
def train_regression(model, loss_fn, batch_fn, num_steps, key, *, lr=1e-3,
                cosine=False, ema_half_life=1000):
    """Adam on loss_fn(model, batch_fn(key1), key2); returns the model, its
    EMA copy and the loss of every step. Steps run in chunks under scan."""
    schedule = optax.cosine_decay_schedule(lr, num_steps) if cosine else lr
    opt = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)
    ema = d2l.EMA(model, ema_half_life)

    @nnx.jit
    def run(model, opt, avg, batches, keys):
        @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
        def step(carry, batch, key):
            model, opt, avg = carry
            loss, grads = nnx.value_and_grad(loss_fn)(model, batch, key)
            opt.update(model, grads)
            ema._average(avg, model, ema.decay)
            return (model, opt, avg), loss
        return step((model, opt, avg), batches, keys)[1]

    chunk, losses = math.gcd(num_steps, 500), []
    for key in jax.random.split(key, num_steps // chunk):
        k1, k2 = jax.random.split(key)            # batches drawn outside jit
        batches = jax.vmap(batch_fn)(jax.random.split(k1, chunk))
        losses.append(run(model, opt, ema.model, batches,
                          jax.random.split(k2, chunk)))
    return model, ema.model, jnp.concatenate(losses)
```

### The Learned Predictor against the Optimal One

(to be written)

### Samples from Both Samplers

(to be written)

### The Dictionary Checked

(to be written)

## Summary

(to be written)

## Exercises

(to be written)
