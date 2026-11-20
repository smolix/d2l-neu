# Sampling by Integration
:label:`sec_diffusion-integration`

<!-- WRITER: this file is the sampling half of the old gaussian-paths.md, moved verbatim (split, structure 17.3). Write the opening (reader, question: why does following the field produce samples from the data and not just points near the modes; GP-1 continuity equation in 2-4 displayed lines, GP-2 x-bar/sigma-bar in two lines, GP-3 reverse-SDE notation), the setup sentence for the two retrained networks with their cost, the one-step-mean home, the summary and the hand-off. -->

```{.python .input #integration-sampling-by-integration-1}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #integration-sampling-by-integration-1}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #integration-sampling-by-integration-2}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # tf32 matmuls on tensor cores
```

<!-- WRITER: the cell below rebuilds what this section needs from the previous one: the mixture, the two paths, the exact predictor `oracle`, and the two networks of the previous section, retrained with the same recipe but 20,000 steps instead of 40,000 (no checkpoints pass between notebooks). Say so, and why: at 20,000 steps each network already sits on its loss floor of :numref:`sec_diffusion-gaussian-paths`; the cell prints the training loss over the last 1,000 steps (dev runs, two seeds per framework: velocity 4.905-4.919 against the floor 4.917 +- 0.007; noise 0.536-0.541 against 0.539 +- 0.0011; the loss average over 1,000 minibatches carries noise of about 0.01 for the velocity). Development runs measured the excess over the floor directly: 0.0017 (velocity) and 0.0003 (noise) at 20,000 steps against 0.0010 and 0.0002 at 40,000, all far below the floors' standard errors, and every sampling number below unchanged within its standard error; do not quote these dev numbers, only the printed losses. Cost: about 70 s for both trainings in PyTorch on one GPU, about 20 s in JAX (include compilation); quote a combined 'about a minute or two'. -->

```{.python .input #integration-sampling-by-integration-3}
%%tab pytorch
device = d2l.try_gpu()
mix = d2l.GaussianMixture.three_modes(device)
vp, linear = d2l.GaussianPath.vp(), d2l.GaussianPath.linear()

def oracle(x, alpha, sigma):
    """The exact posterior means (x0_hat, eps_hat) of the mixture at x."""
    x0_hat = mix.posterior_mean(x, alpha, sigma)
    a, s = (torch.as_tensor(v, device=x.device).reshape(-1, 1)
            for v in (alpha, sigma))
    return x0_hat, (x - a * x0_hat) / s

num_steps = 20_000
torch.manual_seed(1)
_, ema_v, losses_v = d2l.train_regression(
    d2l.TimeMLP(256, 3).to(device), d2l.regression_loss(linear, 'velocity'),
    lambda: mix.sample(512), num_steps, cosine=True, ema_half_life=700)
torch.manual_seed(2)
_, ema_eps, losses_eps = d2l.train_regression(
    d2l.TimeMLP(256, 3).to(device), d2l.regression_loss(vp, 'eps'),
    lambda: mix.sample(512), num_steps, cosine=True, ema_half_life=700)
print(f'loss over the last 1,000 steps: velocity '
      f'{losses_v[-1000:].mean().item():.3f}, '
      f'noise {losses_eps[-1000:].mean().item():.3f}')
```

```{.python .input #integration-sampling-by-integration-3}
%%tab jax
key = jax.random.key(0)
mix = d2l.GaussianMixture.three_modes()
vp, linear = d2l.GaussianPath.vp(), d2l.GaussianPath.linear()

def oracle(x, alpha, sigma):
    """The exact posterior means (x0_hat, eps_hat) of the mixture at x."""
    x0_hat = mix.posterior_mean(x, alpha, sigma)
    a, s = (jnp.reshape(jnp.asarray(v), (-1, 1)) for v in (alpha, sigma))
    return x0_hat, (x - a * x0_hat) / s

num_steps = 20_000
_, ema_v, losses_v = d2l.train_regression(
    d2l.TimeMLP(256, 3, rngs=nnx.Rngs(1)),
    d2l.regression_loss(linear, 'velocity'), lambda k: mix.sample(k, 512),
    num_steps, jax.random.key(1), cosine=True, ema_half_life=700)
_, ema_eps, losses_eps = d2l.train_regression(
    d2l.TimeMLP(256, 3, rngs=nnx.Rngs(2)), d2l.regression_loss(vp, 'eps'),
    lambda k: mix.sample(k, 512), num_steps, jax.random.key(2), cosine=True,
    ema_half_life=700)
print(f'loss over the last 1,000 steps: velocity '
      f'{float(losses_v[-1000:].mean()):.3f}, '
      f'noise {float(losses_eps[-1000:].mean()):.3f}')
```

The marginal velocity carries the noise distribution to the data distribution, so we sample by following it, along deterministic trajectories or with fresh noise at every step.

## The Probability-Flow ODE and Euler's Method

The pair $(p_t, \mathbf{u}_t)$ satisfies the continuity equation (:numref:`sec_mdl-flow-matching`): the marginal velocity moves the density $p_t$ of $\mathbf{x}_t$ along the path. A point drawn from $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and carried backward in time by the ordinary differential equation

$$
\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x}), \qquad \textrm{from } t = 1 \textrm{ down to } t = 0,
$$
:eqlabel:`eq_diffusion-flow-ode`

is therefore a draw from $p_t$ at every time, and from the data at $t = 0$. This *probability-flow ODE* is the deterministic twin of the diffusion process (:numref:`sec_mdl-probability-flow-ode`). We build it up from the smallest case, as :citet:`Nakkiran.Bradley.Zhou.ea.2025` do.

Let the data be one point $\mathbf{x}_0$. On the linear path a point at $\mathbf{x}$ carries the noise $\boldsymbol\epsilon = (\mathbf{x} - (1 - t)\mathbf{x}_0)/t$ and moves with velocity $\boldsymbol\epsilon - \mathbf{x}_0 = (\mathbf{x} - \mathbf{x}_0)/t$. Every trajectory is a straight line into $\mathbf{x}_0$, and a single step of length one from $t = 1$, $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_1(\mathbf{x}) = \mathbf{x}_0$, is exact. With two data points the field is the posterior-weighted average of two such straight fields, and the trajectories bend where the posterior switches from one point to the other (:numref:`subsec_mdl-fm-two-point`).

Any distribution is a mixture of point masses, with straight conditional paths and a curved marginal flow. At $t = 1$ the point is pure noise, so the posterior equals the prior and $\mathbf{u}_1(\mathbf{x}) = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$: one step of length one sends every start to the mean of the data (:numref:`subsec_mdl-fm-two-point`). For our mixture that mean is $(-0.5, -0.7)$, between the modes, where no data lie.

Euler's method (:numref:`sec_mdl-euler-runge-kutta`) follows the curve in $K$ straight steps, $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_t(\mathbf{x})/K$ at $t = 1, 1 - 1/K, \ldots, 1/K$. We run it with the learned field from $2{,}000$ standard Gaussian starts.

<!-- WRITER: the cell below no longer runs the learned field. It is the small computable case (structure 17.3): Euler on the exact field `mix.velocity` from five fixed starts, (0, 0), (+-1.5, 0), (0, +-1.5), with 1, 10 and 100 steps, printing the end points. The reader can predict the first column (every start on the mean). The 2,000 Gaussian starts and the learned field now begin in the next cell. Measured (both tabs, same numbers): 1 step: all five at (-0.50, -0.70); 10 steps: (-1.91, -1.27), (2.73, -1.34), (-2.92, -1.38), (-0.11, 2.59), (-1.98, -2.06); 100 steps: (-1.98, -1.30), (2.90, -1.36), (-3.06, -1.40), (-0.09, 2.77), (-2.05, -2.17). Ten steps end within about 0.2 of the hundred-step end, a hundred within about 0.02 of a thousand (1,000 steps: (-1.99, -1.30), (2.92, -1.36), (-3.08, -1.40), (-0.09, 2.79), (-2.05, -2.19)); each start ends in the mode on its side, the origin in the heaviest mode at (-2.5, -1.5); the ends are points inside the modes, not their centers: the flow is a deterministic map from noise to data. This is the home of the one-step-mean picture (17.8 cites it). -->

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method}
%%tab pytorch
@torch.no_grad()
def euler(net, x, steps):
    """Euler's method on dx/dt = net(x, t), from t = 1 down to t = 0."""
    for k in range(steps, 0, -1):
        t = torch.full((len(x),), k / steps, device=x.device)
        x = x - net(x, t) / steps
    return x

starts = torch.tensor([[0.0, 0.0], [1.5, 0.0], [-1.5, 0.0], [0.0, 1.5],
                       [0.0, -1.5]], device=device)
ends = torch.stack([starts] + [euler(mix.velocity, starts, steps)
                               for steps in (1, 10, 100)], 1)
print('start          1 step          10 steps        100 steps')
for row in ends.tolist():
    print('  '.join(f'({x:5.2f}, {y:5.2f})' for x, y in row))
```

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method}
%%tab jax
def euler(net, x, steps):
    """Euler's method on dx/dt = net(x, t), from t = 1 down to t = 0."""
    for k in range(steps, 0, -1):
        t = jnp.full((len(x),), k / steps)
        x = x - net(x, t) / steps
    return x

starts = jnp.array([[0.0, 0.0], [1.5, 0.0], [-1.5, 0.0], [0.0, 1.5],
                    [0.0, -1.5]])
ends = jnp.stack([starts] + [euler(mix.velocity, starts, steps)
                             for steps in (1, 10, 100)], 1)
print('start          1 step          10 steps        100 steps')
for row in ends.tolist():
    print('  '.join(f'({x:5.2f}, {y:5.2f})' for x, y in row))
```

<!-- WRITER: "A hundred steps put the samples on the three modes" described the old learned-field run; read the exact-field table instead (note above). -->

A hundred steps put the samples on the three modes. The library's sampler takes the same steps in other coordinates: along every Gaussian path, $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ and the noise-to-signal ratio $\bar\sigma = \sigma_t/\alpha_t$ obey $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$ :cite:`Karras.Aittala.Aila.ea.2022`, a form that serves every path and predictor.

`ode_sample` takes Euler steps of this equation through a list of levels $\bar\sigma$, here the path's grid in $t$ from `sigma_bars`; second-order steps and other grids wait for :numref:`sec_diffusion-samplers`.

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab pytorch
@d2l.add_to_class(d2l.GaussianPath)  #@save
def sigma_bars(self, steps):
    """sigma_bar = sigma / alpha on the grid of steps; inf where alpha = 0."""
    a, s = self.grid(steps)
    return [s_k / a_k if a_k > 0 else math.inf for a_k, s_k in zip(a, s)]

def pred_bar(pred, xb, r):  #@save
    """pred at x = alpha x_bar, sigma_bar = r; at r = inf, x_bar is noise."""
    if math.isinf(r):
        return pred(xb, 0.0, 1.0)
    a = 1 / math.sqrt(1 + r ** 2)
    return pred(xb * a, a, r * a)

@torch.no_grad()  #@save
def ode_sample(pred, x, sigma_bars):
    """Euler (= DDIM) on d x_bar / d sigma_bar = eps, from x to data; x is the
    noisy point at the first level, unit-variance scale (noise if inf)."""
    r0 = sigma_bars[0]
    xb = x if math.isinf(r0) else x * math.sqrt(1 + r0 ** 2)  # x / alpha
    for r_t, r_s in zip(sigma_bars[:-1], sigma_bars[1:]):
        x0, eps = pred_bar(pred, xb, r_t)
        xb = x0 + r_s * eps                    # the Euler (= DDIM) step
    return xb
```

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab jax
@d2l.add_to_class(d2l.GaussianPath)  #@save
def sigma_bars(self, steps):
    """sigma_bar = sigma / alpha on the grid of steps; inf where alpha = 0."""
    a, s = self.grid(steps)
    with jax.ensure_compile_time_eval():         # constants, even under jit
        return s / a

def pred_bar(pred, xb, r):  #@save
    """pred at x = alpha x_bar, sigma_bar = r; at r = inf, x_bar is noise."""
    finite = jnp.isfinite(r)
    a = jnp.where(finite, 1 / jnp.sqrt(1 + r ** 2), 0.0)
    return pred(xb * jnp.where(finite, a, 1.0), a,
                jnp.where(finite, r * a, 1.0))

def ode_sample(pred, x, sigma_bars):  #@save
    """Euler (= DDIM) on d x_bar / d sigma_bar = eps, from x to data; x is the
    noisy point at the first level, unit-variance scale (noise if inf)."""
    with jax.ensure_compile_time_eval():         # the grid is a constant
        r = jnp.asarray(sigma_bars)
        from_inf = bool(jnp.isinf(r[0]))
    def step(xb, r):
        r_t, r_s = r
        x0, eps = pred_bar(pred, xb, r_t)
        return x0 + r_s * eps, None              # the Euler (= DDIM) step
    xb = x if from_inf else x * jnp.sqrt(1 + r[0] ** 2)  # x / alpha
    return jax.lax.scan(step, xb, (r[:-1], r[1:]))[0]
```

`make_pred` wraps a trained network as the pair $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ at any $(\alpha, \sigma)$, by querying it at the time of its own path with the same log-SNR, which `time_at` finds by bisection. An $\boldsymbol\epsilon$-predictor has no $\hat{\mathbf{x}}_0 = (\mathbf{x} - \sigma\hat{\boldsymbol\epsilon})/\alpha$ at pure noise, $\alpha = 0$, so samplers that start there on the linear and cosine paths need another kind; an $\boldsymbol\epsilon$-model samples on the VP path, whose $\alpha$ stays positive.

```{.python .input #gaussian-paths-four-targets-one-oracle-1}
%%tab pytorch
@d2l.add_to_class(d2l.GaussianPath)  #@save
def time_at(self, log_snr):
    """The time with a given log-SNR, by bisection (lambda_t falls in t)."""
    lo, hi = torch.zeros_like(log_snr), torch.ones_like(log_snr)
    for _ in range(50):
        mid = (lo + hi) / 2
        later = self.log_snr(mid) > log_snr      # the answer lies after mid
        lo, hi = torch.where(later, mid, lo), torch.where(later, hi, mid)
    return (lo + hi) / 2

def make_pred(net, path, kind):  #@save
    """pred(x, alpha, sigma) -> (x0, eps) from net(x, t) of a kind on path."""
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
@d2l.add_to_class(d2l.GaussianPath)  #@save
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

def make_pred(net, path, kind):  #@save
    """pred(x, alpha, sigma) -> (x0, eps) from net(x, t) of a kind on path."""
    def pred(x, alpha, sigma):
        log_snr = jnp.where(alpha > 0, 2 * jnp.log(alpha / sigma), -jnp.inf)
        t = path.time_at(jnp.asarray(log_snr, jnp.float32))
        x = x * (path.sigma(t) / sigma)         # the same point on our path
        t = jnp.full((len(x),), t)
        return path.x0_eps(kind, net(x, t), x, t)
    return pred
```

How many steps does the flow need? We integrate the wrapped velocity network from the same starts with one, ten and a hundred steps, and check that a hundred steps of `ode_sample` reproduce the loop above.

<!-- WRITER: the cell now draws its own 2,000 starts (the previous cell has five fixed starts), and the check of `ode_sample` against the `euler` loop runs on the exact field (`oracle` against `mix.velocity`), so it tests the sampler's change of coordinates alone, without the rounding of the network's tf32 matmuls (on the network the difference varied between 1e-4 and 2e-3 from run to run). Printed: about 1e-4 (dev). Rewrite 'and the library's sampler agrees with the loop to within 10^-3' as: on the exact field the two agree to float32 rounding, about 10^-4, as the algebra says (on the linear path the Euler step in x-bar/sigma-bar coordinates is the Euler step in x). Dev numbers with the 20k model: one step mean about (-0.50, -0.71), std 0.03-0.06; 10 steps strays 0.024-0.032, weights about 0.52/0.27-0.29/0.19-0.21; 100 steps strays 0.012-0.014, weights within 0.012 of 0.5/0.3/0.2. -->

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method-1}
%%tab pytorch
torch.manual_seed(3)
x_T = torch.randn(2000, 2, device=device)
pred_v = make_pred(ema_v, linear, 'velocity')
samples = [ode_sample(pred_v, x_T, linear.sigma_bars(steps))
           for steps in (1, 10, 100)]
print('one step: mean', [round(v, 3) for v in samples[0].mean(0).tolist()],
      'std', [round(v, 3) for v in samples[0].std(0).tolist()])
for steps, x in zip((10, 100), samples[1:]):
    print(f'{steps} steps:', mix.summary(x))
exact = (ode_sample(oracle, x_T, linear.sigma_bars(100))
         - euler(mix.velocity, x_T, 100)).abs().max().item()
print(f'ode_sample against the loop on the exact field: {exact:.0e}')
d2l.scatter_panels(samples, ['1 step', '10 steps', '100 steps'],
                   data=mix.sample(2000));
```

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method-1}
%%tab jax
key, k_T = jax.random.split(key)
x_T = jax.random.normal(k_T, (2000, 2))
pred_v = make_pred(ema_v, linear, 'velocity')
samples = [ode_sample(pred_v, x_T, linear.sigma_bars(steps))
           for steps in (1, 10, 100)]
print('one step: mean', [round(v, 3) for v in samples[0].mean(0).tolist()],
      'std', [round(v, 3) for v in samples[0].std(0).tolist()])
for steps, x in zip((10, 100), samples[1:]):
    print(f'{steps} steps:', mix.summary(x))
exact = float(jnp.abs(ode_sample(oracle, x_T, linear.sigma_bars(100))
                      - euler(mix.velocity, x_T, 100)).max())
print(f'ode_sample against the loop on the exact field: {exact:.0e}')
key, k_data = jax.random.split(key)
d2l.scatter_panels(samples, ['1 step', '10 steps', '100 steps'],
                   data=mix.sample(k_data, 2000));
```

One step lands every start on the data mean $(-0.5, -0.7)$, with almost no spread, as the point-mass argument predicts. Ten steps recover the three modes with weights within about $0.03$ of the targets, but about three percent of the samples stray between the modes. A hundred steps bring the strays down to about one percent, and the library's sampler agrees with the loop to within $10^{-3}$. The extra steps follow the curvature of the marginal flow.

## The Reverse SDE and the Ancestral Chain

DDPM runs the reverse chain of :numref:`fig_diffusion-forward-reverse` instead, one stochastic step at a time. The reversal of a small forward step is nearly Gaussian :cite:`sohl2015deep`, and given the clean point it is exactly Gaussian: for $s < t$, $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ has a mean linear in $\mathbf{x}_t$ and $\mathbf{x}_0$. Ho et al.'s *ancestral* step replaces $\mathbf{x}_0$ by $\hat{\mathbf{x}}_0$ in that mean and adds fresh noise with the variance $\beta = 1 - \bar\alpha_t / \bar\alpha_s$ of the forward step. On the VP path, written with the score, it reads

$$
\mathbf{x}_s = \frac{\mathbf{x}_t + \beta\, \nabla \log p_t(\mathbf{x}_t)}{\sqrt{1 - \beta}} + \sqrt{\beta}\, \mathbf{z},
\qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$
:eqlabel:`eq_diffusion-ancestral-score`

With $1/\sqrt{1 - \beta} \approx 1 + \beta/2$, the step splits to first order into two:

$$
\mathbf{x}_s \approx \Big[ \mathbf{x}_t + \tfrac{\beta}{2} \nabla \log p_t(\mathbf{x}_t) + \sqrt{\beta}\, \mathbf{z} \Big] + \tfrac{\beta}{2} \big( \mathbf{x}_t + \nabla \log p_t(\mathbf{x}_t) \big).
$$

The bracket is a Langevin step of size $h = \beta$, which leaves $p_t$ in place and pulls stray points back toward it, and the second term is one Euler step of the probability-flow ODE from $t$ back to $s$, since on the VP path $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$ and $\beta \approx \beta(t)\, (t - s)$.

In continuous time the reversal is exact, by a theorem of :citet:`Anderson.1982`. A forward process $d\mathbf{x} = \mathbf{f}\, dt + g\, d\mathbf{W}$ is reversed, from $t = 1$ down to $0$, by

$$
d\mathbf{x} = \big[ \mathbf{f} - g^2 \nabla \log p_t(\mathbf{x}) \big]\, dt + g\, d\bar{\mathbf{W}},
\qquad\textrm{which has the same marginals as}\qquad
\frac{d\mathbf{x}}{dt} = \mathbf{f} - \tfrac12 g^2 \nabla \log p_t(\mathbf{x}),
$$
:eqlabel:`eq_diffusion-reverse`

where $\bar{\mathbf{W}}$ is a Brownian motion run backward. For the VP path, $\mathbf{f} = -\tfrac12 \beta(t)\mathbf{x}$ and $g^2 = \beta(t)$, the right-hand ODE is the flow we just integrated, and the ancestral step is, to first order, the Euler--Maruyama step (:numref:`sec_mdl-euler-maruyama`) of the SDE. Between the two lies a family of samplers with the same marginals: the more noise a sampler injects, the larger the score correction in its drift (:numref:`sec_mdl-time-reversal`, :numref:`fig_mdl-dyn-lambda-family`).

`ancestral_sample` takes this step on any path with any predictor `pred(x, alpha, sigma) -> (x0, eps)`, with the noise variance $\sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ of the forward step from $s$ to $t$.

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain}
%%tab pytorch
@torch.no_grad()  #@save
def ancestral_sample(pred, x, path, steps, *, t_start=1.0):
    """Ho et al.'s sampler from x at t_start (1 is noise) to data at t = 0."""
    a, s = path.grid(steps, t_start)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        ratio = s[i + 1] * a[i] / (a[i + 1] * s[i])
        x = a[i + 1] * x0 + s[i + 1] * ratio * eps      # the posterior mean
        if s[i + 1] > 0:                                # no noise at t = 0
            var = s[i] ** 2 - (a[i] / a[i + 1] * s[i + 1]) ** 2  # forward step
            x = x + math.sqrt(var) * torch.randn_like(x)
    return x
```

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain}
%%tab jax
def ancestral_sample(pred, x, path, steps, key, *, t_start=1.0):  #@save
    """Ho et al.'s sampler from x at t_start (1 is noise) to data at t = 0."""
    a, s = path.grid(steps, t_start)
    a_t, s_t, a_s, s_s = a[:-1], s[:-1], a[1:], s[1:]
    ratio = s_s * a_t / (a_s * s_t)
    var = jnp.maximum(s_t ** 2 - (a_t / a_s * s_s) ** 2, 0)  # forward step
    noise = jnp.where(s_s > 0, jnp.sqrt(var), 0)             # none at t = 0
    def step(x, inputs):
        a_t, s_t, a_s, c_eps, c_noise, key = inputs
        x0, eps = pred(x, a_t, s_t)
        x = a_s * x0 + c_eps * eps                      # the posterior mean
        return x + c_noise * jax.random.normal(key, x.shape), None
    return jax.lax.scan(step, x, (a_t, s_t, a_s, s_s * ratio, noise,
                                  jax.random.split(key, steps)))[0]
```

We run two hundred steps of the chain with the noise model, from the same $2{,}000$ starts as the flow.

<!-- WRITER: at 2,000 points a stray share of about 1% carries a standard error of about 0.002 and the weights about 0.011; direct runs printed strays 0.005-0.013 and weights within 0.013 of 0.5/0.3/0.2. Keep the reading qualitative ('the right weights up to the sampling error of 2,000 points, about one percent strays'); the paired 20,000-point comparison below is where the numbers are. -->

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain-1}
%%tab pytorch
pred_eps = make_pred(ema_eps, vp, 'eps')
print(mix.summary(ancestral_sample(pred_eps, x_T, vp, 200)))
```

```{.python .input #gaussian-paths-the-reverse-sde-and-the-ancestral-chain-1}
%%tab jax
pred_eps = make_pred(ema_eps, vp, 'eps')
key, k_anc = jax.random.split(key)
print(mix.summary(ancestral_sample(pred_eps, x_T, vp, 200, k_anc)))
```

The chain recovers the three modes with the right weights, up to the sampling error of $2{,}000$ points, and about as few strays as the flow at a hundred steps; :citet:`ho2020denoising` ran a thousand steps, but in two dimensions two hundred suffice. Either sampler can use either network through `make_pred`.

## Samples from Both Samplers

<!-- WRITER: the cell below now gives network and oracle the same starts AND the same injected noise (ancestral), so each network-oracle difference is the network's alone; it prints the standard error of each weight (0.0035, 0.0032, 0.0028) and, per sampler, the share of starts that network and oracle send to different modes. Direct runs of this notebook (both tabs) and dev runs (two seeds per framework) agree: network and oracle end in different modes for 0.2-0.5% of the starts (Euler 0.2-0.5%, ancestral 0.4-0.5%); every network weight lies within 0.004 of the oracle's; strays: network 1.2-1.3%, oracle 1.1-1.2%, data 1.0-1.1%. New lesson the run carries: the network changes the mode of fewer than one sample in two hundred; the remaining differences from 0.5/0.3/0.2 are the sampling error of 20,000 points, which the oracle shares. -->

A sampler accumulates the errors of all its steps. We draw $20{,}000$ samples with two hundred ancestral steps of the noise model and a hundred Euler steps of the velocity model, and repeat both with the oracle in place of the network. At this sample size a mode weight carries a standard error of about $0.0035$.

```{.python .input #gaussian-paths-samples-from-both-samplers}
%%tab pytorch
torch.manual_seed(5)
x_T = torch.randn(20_000, 2, device=device)
runs = {'data': mix.sample(20_000)}
for name, pred in (('network', pred_eps), ('oracle', oracle)):
    torch.manual_seed(6)                       # the same noise for both
    runs[f'ancestral, {name}'] = ancestral_sample(pred, x_T, vp, 200)
for name, pred in (('network', pred_v), ('oracle', oracle)):
    runs[f'Euler, {name}'] = ode_sample(pred, x_T, linear.sigma_bars(100))
for name, x in runs.items():
    print(f'{name:>18}:', mix.summary(x))
print('standard error of a weight:', [round(math.sqrt(w * (1 - w) / 20_000), 4)
                                      for w in mix.weights.tolist()])
mode = lambda x: ((x[:, None] - mix.means) ** 2).sum(-1).argmin(1)
for s in ('ancestral', 'Euler'):
    moved = mode(runs[s + ', network']) != mode(runs[s + ', oracle'])
    print(f'{s}: network and oracle end in different modes for '
          f'{moved.float().mean().item():.2%} of the starts')
d2l.scatter_panels([runs[k][:2000] for k in runs],
                   list(runs), lim=4.5, size=2.4);
```

```{.python .input #gaussian-paths-samples-from-both-samplers}
%%tab jax
key, k_T, k_data, k_noise = jax.random.split(key, 4)
x_T = jax.random.normal(k_T, (20_000, 2))
runs = {'data': mix.sample(k_data, 20_000)}
for name, pred in (('network', pred_eps), ('oracle', oracle)):
    runs[f'ancestral, {name}'] = ancestral_sample(  # the same noise for both
        pred, x_T, vp, 200, k_noise)
for name, pred in (('network', pred_v), ('oracle', oracle)):
    runs[f'Euler, {name}'] = ode_sample(pred, x_T, linear.sigma_bars(100))
for name, x in runs.items():
    print(f'{name:>18}:', mix.summary(x))
print('standard error of a weight:', [round(math.sqrt(w * (1 - w) / 20_000), 4)
                                      for w in mix.weights.tolist()])
mode = lambda x: ((x[:, None] - mix.means) ** 2).sum(-1).argmin(1)
for s in ('ancestral', 'Euler'):
    moved = mode(runs[s + ', network']) != mode(runs[s + ', oracle'])
    print(f'{s}: network and oracle end in different modes for '
          f'{float(moved.mean()):.2%} of the starts')
d2l.scatter_panels([runs[k][:2000] for k in runs],
                   list(runs), lim=4.5, size=2.4);
```

The ancestral chain with the network recovers the weights $0.5/0.3/0.2$ within sampling error, and Euler with the network matches them to within about $0.01$. Every sampler, network or oracle, strays about as often as the data themselves, about one percent, the share of a Gaussian beyond three standard deviations. Both samplers recover the mode weights, which no single noise level could.

## Summary

<!-- WRITER: summary of the sampling half; `tab_diffusion_models` ends this section (structure 17.3); its rows "here" now point to two sections, and `ex_gaussian-paths-ncsn` may move to further reading. -->

:The models of this chapter: the target and path that train the network, and the sampler that produces data.
:label:`tab_diffusion_models`

| model | target and path | sampler | where |
|:--|:--|:--|:--|
| denoising score model | the noise, at one level | Langevin dynamics | :numref:`sec_diffusion-denoising` |
| noise-conditional score network | the noise at each level of a ladder, weighted by $\tilde\sigma^2$ | annealed Langevin dynamics | here, :exref:`ex_gaussian-paths-ncsn` |
| DDPM | $\boldsymbol\epsilon$ on the VP path | the ancestral chain or the probability-flow ODE | here, :numref:`sec_diffusion-images` |
| flow matching, rectified flow | $\boldsymbol\epsilon - \mathbf{x}_0$ on the linear path | Euler steps on the probability-flow ODE | here, :numref:`sec_diffusion-samplers` |
| $\mathbf{v}$-prediction | $\mathbf{v}$ on the cosine path | either, through the dictionary | :numref:`sec_diffusion-design-space` |
| masked diffusion | the clean tokens of a partly masked sequence, by a weighted cross-entropy | iterative unmasking | :numref:`sec_diffusion-discrete` |

Nothing in this section used the fact that $\mathbf{x}$ had two coordinates. An image has thousands, and its pixels are strongly correlated; the next question is what network, and what changes to training and sampling, a path through image space needs.

## Exercises

1. **The ancestral step to first order, for Gaussian data.** ● Let the data be $\mathcal{N}(\mathbf{0}, s^2 \mathbf{I})$, so that on the DDPM chain $q_k = \mathcal{N}(\mathbf{0}, v_k \mathbf{I})$ with $v_k = \bar\alpha_k s^2 + 1 - \bar\alpha_k$.
    1. Show that one ancestral step :eqref:`eq_diffusion-ancestral-score` with the exact score and step variance $\beta_k$ maps $q_k$ to $\mathcal{N}\big(\mathbf{0}, (v_{k-1} + \delta_k)\mathbf{I}\big)$ with $\delta_k = \beta_k^2 (1/v_k - 1)/(1 - \beta_k)$.
    1. For which data is the step exact? Is the variance too large or too small for data narrower than the standard Gaussian, such as one component of the mixture, $s = 0.5$?
    1. How does the error accumulated over the chain change when the schedule is refined to twice as many steps with half the $\beta_k$?
   :exlabel:`ex_gaussian-paths-ancestral-first-order`
1. **Step size and Gaussian reversal.** ● The ancestral chain uses Gaussian transitions, which are justified for small steps. This problem measures how small. Write $\sigma_{t|s}^2 = \sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ for the variance of the forward step from $s$ to $t > s$.
    1. Show that the exact reversal is the mixture $q(\mathbf{x}_s \mid \mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \mid \mathbf{x}_t}\big[q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)\big]$ and that its covariance is $\tau^2 \mathbf{I} + c^2\, \textrm{Cov}[\mathbf{x}_0 \mid \mathbf{x}_t]$, where $\tau^2 = \sigma_{t|s}^2 \sigma_s^2/\sigma_t^2$ is the variance of $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ and $c = \alpha_s \sigma_{t|s}^2/\sigma_t^2$ is the coefficient of $\mathbf{x}_0$ in its mean.
    1. Let the data be $\pm 1$ with equal probability in one dimension, and let $x_t = 0$. Show that the reversal is the equal mixture of $\mathcal{N}(\pm c, \tau^2)$, which is bimodal exactly when $\zeta^2 = c^2/\tau^2$ exceeds one, and that $\zeta^2 = \alpha_s^2 \sigma_{t|s}^2 / (\sigma_t^2 \sigma_s^2)$. How does $\zeta$ shrink as the step $t - s$ is refined?
    1. Evaluate $\zeta$ on the VP path from $t = 0.5$ to $s = 0.495$, $0.45$ and $0.3$, and for one step from $t = 0.3$ to $s = 0$. Which reversals are bimodal? What does a Gaussian transition with the optimal mean produce at $x_t = 0$ in the last case, and what should it produce?
   :exlabel:`ex_gaussian-paths-gaussian-reversal`
1. [code] **Commitment time.** ● When along the path does a sample decide its mode? :citet:`Biroli.Bonnaire.deBortoli.ea.2024` measure this by cloning: start at a point of $q_t$, run two copies of the reverse chain with independent noise, and record whether they end in the same mode.
    1. Draw $2{,}000$ clean points, noise them to $\mathbf{x}_t$ on the VP path for $t = 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1$, and run two copies of `ancestral_sample(pred_eps, x_t, vp, steps, t_start=t)` from each with about $200\, t$ steps. Plot the fraction $\varphi(t)$ of pairs whose ends have the same nearest mean. Which value does $\varphi$ approach as $t \to 1$, and why?
    1. Let $\Lambda$ be the largest eigenvalue of the covariance of the data. Biroli et al. define the speciation time by $\bar\alpha_t \Lambda = 1$, where the signal along the top principal direction stops dominating the noise. Compute it for the mixture and compare it with your curve and with the ladder of this section, whose coarse levels decided the mode weights. Exercise :exref:`ex_mdl-flow-matching-transport-late-decisions` of :numref:`sec_mdl-flow-matching-transport` and Exercise :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` of :numref:`sec_mdl-fokker-planck-probability-flow` work out the same question for two modes in one dimension.
   :exlabel:`ex_gaussian-paths-commitment-time`
1. **Same ratio, same flow.** ● Let two Gaussian paths have coefficients related by $\alpha^B_t = c_t\, \alpha^A_t$ and $\sigma^B_t = c_t\, \sigma^A_t$ for a smooth positive function $c_t$, so that they have the same log-SNR at every time.
    1. Show that $p^B_t(\mathbf{y}) = c_t^{-d}\, p^A_t(\mathbf{y}/c_t)$, and use :eqref:`eq_diffusion-score-velocity` to show that $\mathbf{u}^B_t(\mathbf{y}) = (\dot c_t / c_t)\, \mathbf{y} + c_t\, \mathbf{u}^A_t(\mathbf{y}/c_t)$. Conclude that the trajectories of the probability-flow ODE are related by $\mathbf{y}_t = c_t\, \mathbf{x}_t$.
    1. Let $\tau(t)$ be increasing. Show that the path $p_{\tau(t)}$ is generated by the field $\dot\tau(t)\, \mathbf{u}_{\tau(t)}$, with the same trajectories as $\mathbf{u}$.
    1. Find the change of time $\tau(t)$ for which the cosine path $(\cos\frac{\pi\tau}{2}, \sin\frac{\pi\tau}{2})$ at time $\tau(t)$ has the log-SNR of the linear path at time $t$, and show that $c_t = 1/\sqrt{(1 - t)^2 + t^2}$. Where along the path is the rescaling largest?
    1. What does the probability-flow ODE of the cosine path do to data distributed as $\mathcal{N}(\mathbf{0}, \mathbf{I})$, and why? What does that of the linear path do to the same data?
   :exlabel:`ex_gaussian-paths-same-ratio-same-flow`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17.3]{.kicker}

Sampling by Integration<br>
**the flow ODE and Euler steps · one step lands on the mean · the ancestral chain · network and oracle side by side**
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- sample by Euler steps on the probability-flow ODE;
- say where one Euler step from pure noise lands, and why;
- split an ancestral step into a Langevin step and a flow step;
- run either sampler on any network through a common predictor.

**Running example:** the three-mode mixture with its exact posterior mean,
and the velocity and noise networks of the previous section, retrained.
:::


::: {.slide title="The Flow ODE Carries Noise to Data; Euler Steps Follow It"}
[Sampling]{.kicker}

$$\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x}) \textrm{ from } t = 1 \textrm{ to } 0,
\qquad \textrm{Euler: } \mathbf{x} \leftarrow \mathbf{x} - \tfrac{1}{K}\, \mathbf{u}_t(\mathbf{x})$$

- One data point: $\mathbf{u}_t(\mathbf{x}) = (\mathbf{x} - \mathbf{x}_0)/t$,
  straight lines; one step is exact.
- Many points: straight conditional paths, a curved marginal flow.
- At $t = 1$ the posterior is the prior, $\mathbf{u}_1(\mathbf{x}) = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$.

::: {.d2l-note}
One step sends every start to the data mean $(-0.5, -0.7)$, between the
modes.
:::
:::

::: {.slide title="On the Exact Field, One Step Lands on the Mean and Ten Reach the Modes"}
@!gaussian-paths-the-probability-flow-ode-and-euler-s-method

- Five starts, Euler on the mixture's exact velocity, 1, 10, 100 steps.
- One step: every start at the data mean $(-0.5, -0.7)$, where no data lie.
- Ten steps: within about 0.2 of the hundred-step end, inside the mode on
  the start's side; the origin goes to the heaviest mode.
:::

::: {.slide title="The Learned Field Needs About a Hundred Euler Steps"}
@!gaussian-paths-the-probability-flow-ode-and-euler-s-method-1

One step: the mean $(-0.5, -0.7)$, almost no spread. Ten steps: weights
within about 0.03, about 3% strays. A hundred: about 1% strays.
:::

::: {.slide title="The Ancestral Step Is a Langevin Step Plus an Euler Step"}
DDPM's step on the VP path from $t$ to $s < t$, with
$\beta = 1 - \bar\alpha_t / \bar\alpha_s$ and $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\mathbf{x}_s = \frac{\mathbf{x}_t + \beta\, \nabla \log p_t(\mathbf{x}_t)}{\sqrt{1 - \beta}} + \sqrt{\beta}\, \mathbf{z}$$

To first order in $\beta$:

$$\mathbf{x}_s \approx \underbrace{\mathbf{x}_t + \tfrac{\beta}{2} \nabla \log p_t + \sqrt{\beta}\, \mathbf{z}}_{\textrm{Langevin step: keeps } p_t}
+ \underbrace{\tfrac{\beta}{2} \big(\mathbf{x}_t + \nabla \log p_t\big)}_{\textrm{Euler step of the flow ODE}}$$

::: {.d2l-note}
The Langevin part pulls stray points back toward $p_t$.
:::
:::

::: {.slide title="The Reverse SDE and the Flow ODE Share Their Marginals"}
Forward $d\mathbf{x} = \mathbf{f}\, dt + g\, d\mathbf{W}$; for VP,
$\mathbf{f} = -\tfrac12 \beta(t)\, \mathbf{x}$, $g^2 = \beta(t)$. From
$t = 1$ down to 0 (Anderson, 1982):

$$d\mathbf{x} = \big[\mathbf{f} - g^2 \nabla \log p_t(\mathbf{x})\big]\, dt + g\, d\bar{\mathbf{W}}
\qquad \textrm{and} \qquad
\frac{d\mathbf{x}}{dt} = \mathbf{f} - \tfrac12 g^2 \nabla \log p_t(\mathbf{x})$$

- Ancestral chain: Euler--Maruyama on the left, to first order.
- Flow sampler: Euler on the right.
- Between them, a family: the more noise injected, the larger the score
  correction in the drift.
:::

::: {.slide title="Network and Oracle Sample Alike with Both Samplers"}
[Experiments]{.kicker}

@!gaussian-paths-samples-from-both-samplers

::: {.d2l-note}
200 ancestral steps (noise model), 100 Euler steps (velocity model), and the
same with the exact posterior mean, from the same starts and the same noise.
Network and oracle send fewer than 1 start in 200 to different modes; strays
about 1% everywhere, as in the data.
:::
:::

::: {.slide title="Check: Why Is the Converted Noise Network Worst near the Noise?"}
[Check]{.kicker}

On the linear path $\hat{\mathbf{x}}_0 = (\mathbf{x} - t\,
\hat{\boldsymbol{\epsilon}}) / (1 - t)$. If $\hat{\boldsymbol{\epsilon}}$ is
off by $\boldsymbol{\delta}$, how far off is the velocity
$\hat{\boldsymbol{\epsilon}} - \hat{\mathbf{x}}_0$ at $t = 0.95$?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $\hat{\boldsymbol{\epsilon}} - \hat{\mathbf{x}}_0 =
(\hat{\boldsymbol{\epsilon}} - \mathbf{x}) / (1 - t)$, so the error is
$\boldsymbol{\delta} / (1 - t) = 20\, \boldsymbol{\delta}$. At $\alpha = 0$
it is unbounded: a noise model is never asked for $\hat{\mathbf{x}}_0$ at
pure noise.
:::
:::
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- **Flow ODE:** following the marginal velocity from $t = 1$ to $0$ carries
  noise to data.
- **Euler:** one step lands on the data mean, a hundred recover the modes.
- **Ancestral chain:** a Langevin step plus an Euler step of the flow.
- **Both samplers, network or oracle:** the mode weights that one noise level
  could not find.

**Next:** the same regression on 784-pixel images, with a U-Net, guidance
and a quality score.
:::
