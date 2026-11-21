# Sampling by Integration
:label:`sec_diffusion-integration`

It is tempting to think that we are done once a network knows, at every noise level, where a noisy point should move: a field of arrows that leads from the noise toward the data. Draw a point of noise, follow the arrows, stop when they stop. But arrows that point at the data produce points near the data, and points near the data are not yet samples *of* the data. The Langevin chains of :numref:`sec_diffusion-denoising` showed the difference. Driven by the exact score of the data, they ended near the modes, each with the right spread, and kept the shares of their starts, about a third in each mode, where the data put a half, three tenths and a fifth. The annealed ladder of :numref:`sec_diffusion-gaussian-paths` got the shares right on exact scores, walking down ten noise levels whose spacing and step sizes we chose by hand. It worked, and nothing in it said that it had to. Our network gives a field at every level of a continuous path. Is following it enough, and why should it put the right share of the samples in each mode?

The answer is a conservation law. A velocity field moves probability the way a current moves water, and the field of the previous section moves exactly the mass of the Gaussian path: it carries the noise at $t = 1$ into the data at $t = 0$ through every intermediate density. :citet:`song2021score` built a sampler on this fact, a differential equation that runs the field backward from noise. :citet:`ho2020denoising` had sampled earlier with a chain of noisy steps, and it turns out that their chain follows the same law, with a correction added at every step.

We first show why the field transports the density, then integrate it with Euler's method, then derive the noisy chain and the continuous-time process behind it. Both samplers run on the trained networks and on the exact field of the mixture, so that we can see how much of their error is the network's.

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

Each section runs as a notebook of its own, so the cell below rebuilds what we need from :numref:`sec_diffusion-gaussian-paths`: the three-mode mixture and two of its paths, the variance-preserving (VP) path and the linear path. It also defines the exact posterior means $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ of the mixture at any point and noise level, which the code calls `oracle`, as in the previous section. Then it retrains two networks, one for each sampler of this section: a velocity model on the linear path for the flow, and a noise model on the VP path for DDPM's chain. The recipe is that of the previous section with half as many steps, and twenty thousand steps suffice. To show this, the cell prints the mean training loss over the last thousand steps, to be compared with the floors of the previous section, about $4.9$ for the velocity and $0.54$ for the noise. Both trainings together take about a minute on one GPU.

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

Both losses are at their floors: neither network has much left to learn from the regression. How close they come to the oracle where it matters, in the samples, is the subject of the last part of this section.

## Transport of Probability Mass

Let us start with what a velocity field does to a density. Take a cloud of points distributed as $p_t$ and let every point move for a short time with the velocity $\mathbf{u}_t(\mathbf{x})$ at its position. No point is created or destroyed, so the density at $\mathbf{x}$ can change only by the mass that flows in minus the mass that flows out. The mass crossing a small window per unit time is the density times the velocity, $p_t \mathbf{u}_t$. What this flow carries out of a small box around $\mathbf{x}$, net and per unit volume, is its divergence $\nabla \cdot (p_t \mathbf{u}_t) = \sum_i \partial (p_t u_{t,i}) / \partial x_i$, and the density there falls at exactly that rate. This is the *continuity equation* of fluid mechanics:

$$
\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot \big(p_t(\mathbf{x})\, \mathbf{u}_t(\mathbf{x})\big).
$$

When a density and a field satisfy it, points that follow the field keep the density of the path at every time, forward or backward. Does the marginal velocity of :numref:`sec_diffusion-gaussian-paths` satisfy it together with $p_t$?

Take one clean point $\mathbf{x}_0$ first. Its noisy versions form the Gaussian cloud $\mathcal{N}(\alpha_t \mathbf{x}_0, \sigma_t^2 \mathbf{I})$, and each point of the cloud moves with the conditional velocity $\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon$, where $\boldsymbol\epsilon = (\mathbf{x} - \alpha_t \mathbf{x}_0)/\sigma_t$ is the noise that put it at $\mathbf{x}$. This field, call it $\mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0)$, moves this one cloud correctly: a point that keeps its noise $\boldsymbol\epsilon$ sits at $\alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ at every time, a draw from the cloud, so the cloud and its field satisfy the continuity equation. The marginal density $p_t$ is the average of the clouds over the data, and the equation is linear in the density, so averaging the clouds' equations gives one for $p_t$ with the averaged mass flow on the right. By Bayes' rule, $p(\mathbf{x}_0)\, p_t(\mathbf{x} \mid \mathbf{x}_0) = p_t(\mathbf{x})\, p(\mathbf{x}_0 \mid \mathbf{x}_t = \mathbf{x})$, so this flow is

$$
\mathbb{E}_{\mathbf{x}_0}\big[p_t(\mathbf{x} \mid \mathbf{x}_0)\, \mathbf{u}_t(\mathbf{x} \mid \mathbf{x}_0)\big]
= p_t(\mathbf{x})\, \mathbb{E}\big[\dot\alpha_t \mathbf{x}_0 + \dot\sigma_t \boldsymbol\epsilon \,\big|\, \mathbf{x}_t = \mathbf{x}\big]
= p_t(\mathbf{x})\, \mathbf{u}_t(\mathbf{x}).
$$

The posterior average in the middle is how the marginal velocity was defined. So the mass that all the clouds together carry across $\mathbf{x}$ is the density there times the marginal velocity, and $p_t$ and $\mathbf{u}_t$ satisfy the continuity equation. The samplers of the following sections rest on this result. :numref:`sec_mdl-flow-matching-transport` proves it with the conditions it needs.

A reader may object that averaged velocities do not produce averaged trajectories, and the objection is right. The conditional paths (straight lines, on the linear path) pass through every point at every time and cross one another everywhere, while two trajectories of one field never meet at the same time, because a point has only one velocity at a given time. A sample that follows $\mathbf{u}_t$ follows no conditional path at all. The argument above says nothing about single trajectories; it says that the field reproduces the *density* at every time, and the density is all a sampler needs.

This answers our question. Solved exactly, the flow cannot get the mode weights wrong, because it conserves mass at every instant: half of the noise must end in the mode of weight one half. The Langevin chains would have kept the data distribution in place once they reached it, but nothing carried them there across the valleys. The annealed ladder crossed the valleys at its coarse levels, where the noise fills them in, but how many levels and steps it needed was ours to guess. The flow starts from a density it knows, the standard Gaussian, and moves it into the data with a guarantee. Two things can still go wrong: our steps only approximate the solution, and our network only approximates the field. The rest of the section measures both.

The sampler, then, draws $\mathbf{x}$ from $p_1 = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and solves the ordinary differential equation

$$
\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x}), \qquad \textrm{from } t = 1 \textrm{ down to } t = 0.
$$
:eqlabel:`eq_diffusion-flow-ode`

At every time its solution is a draw from $p_t$, and at $t = 0$ a draw from the data. :citet:`song2021score` called it the *probability-flow ODE* (:numref:`sec_mdl-probability-flow-ode`). Nothing random happens after the start: the whole randomness of a sample lies in its starting point.

## Euler's Method on the Probability-Flow ODE

An ODE is solved numerically by many small straight steps. *Euler's method* (:numref:`sec_mdl-euler-runge-kutta`) cuts the time from $1$ to $0$ into $K$ equal steps and at each one moves the point against the velocity, $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_t(\mathbf{x})/K$ at $t = 1, 1 - 1/K, \ldots, 1/K$. How many steps does the flow need? General data are hard to reason about, so we start from data that make the answer obvious and build up, as :citet:`Nakkiran.Bradley.Zhou.ea.2025` do.

### From One Point to the Data Mean

Let the data be a single point $\mathbf{x}_0$. On the linear path ($\alpha_t = 1 - t$, $\sigma_t = t$) a point $\mathbf{x}$ at time $t$ carries the noise $\boldsymbol\epsilon = (\mathbf{x} - (1 - t)\mathbf{x}_0)/t$ and moves with velocity $\boldsymbol\epsilon - \mathbf{x}_0 = (\mathbf{x} - \mathbf{x}_0)/t$. Every trajectory is a straight line into $\mathbf{x}_0$ at constant speed, and a single Euler step of length one, $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{u}_1(\mathbf{x}) = \mathbf{x}_0$, is exact. With two data points the field is the posterior-weighted average of two such straight fields, and the trajectories bend where the posterior switches from one point to the other (:numref:`subsec_mdl-fm-two-point`).

Any distribution is a mixture of point masses, and its trajectories bend wherever the posterior is undecided. At $t = 1$ it is entirely undecided. A point of pure noise says nothing about the clean point it came from, so the posterior is the prior and

$$
\mathbf{u}_1(\mathbf{x}) = \mathbb{E}\big[\boldsymbol\epsilon - \mathbf{x}_0 \,\big|\, \mathbf{x}_1 = \mathbf{x}\big] = \mathbf{x} - \mathbb{E}[\mathbf{x}_0].
$$

One Euler step of length one therefore sends every start to the mean of the data. For our mixture we can predict the result before running anything: $0.5\,(-2.5, -1.5) + 0.3\,(2.5, -1.5) + 0.2\,(0, 2.5) = (-0.5, -0.7)$, a point between the modes, more than four standard deviations from the nearest one. To see what more steps do, we run Euler's method on the exact field of the mixture on the linear path, `mix.velocity`, from five fixed starts, the origin and four points at distance $1.5$ from it, with one, ten and a hundred steps.

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

The one-step column is the prediction: all five starts land on $(-0.50, -0.70)$, where no data lie. Ten steps already put every start inside a mode, within about $0.2$ of where a hundred steps end, and always on the near side, closer to the data mean; the coarse steps keep a little of the one-step error. Now look at *which* mode each start reaches. The origin, the start below it and the start to its left all end in the heaviest mode, the start on the right in the mode on the right, the one above in the top mode. This is the conservation of mass at work: half of the standard Gaussian must end in the mode of weight one half, so that mode collects the largest region of starts, here including the center of the Gaussian. Nor do the ends sit at the mode centers. The origin ends at $(-1.98, -1.30)$, about one standard deviation of a mode from its center. The flow is a map from noise to data, and every start has a sample of its own.

### Coordinates for Every Path

Euler steps in $t$ depend on the path. On the VP path the noise level is still $\sigma_t = 0.96$ at $t = 0.5$ and has fallen to $0.32$ by $t = 0.1$ (:numref:`sec_diffusion-gaussian-paths`): the first half of the time removes almost no noise, and two thirds of it go in the next two fifths of the time. A step in $t$ moves at the rate it finds at its start, and on such a curve a long step misses badly (Exercise :exref:`ex_integration-coordinates`). The library's sampler therefore takes its steps in coordinates in which every Gaussian path looks the same. Divide the path by $\alpha_t$:

$$
\bar{\mathbf{x}} = \frac{\mathbf{x}_t}{\alpha_t} = \mathbf{x}_0 + \bar\sigma\, \boldsymbol\epsilon,
\qquad \bar\sigma = \frac{\sigma_t}{\alpha_t}.
$$

In these coordinates the data keep their scale and only the noise grows, from $\bar\sigma = 0$ at the data to $\bar\sigma = \infty$ at pure noise. Every path becomes the variance-exploding (VE) path of :numref:`sec_diffusion-gaussian-paths`, the one the annealed ladder walked down, and $\bar\sigma$ is its noise level. :citet:`Karras.Aittala.Aila.ea.2022` sample all paths this way. Here the bar means division by $\alpha_t$, not the square it means in DDPM's $\bar\alpha_t = \alpha_t^2$ (:numref:`tab_diffusion-notation`). Along one conditional path $\bar{\mathbf{x}}$ moves in a straight line in $\bar\sigma$ with slope $\boldsymbol\epsilon$. The marginal field is the posterior average of these slopes, the noise prediction:

$$
\frac{d\bar{\mathbf{x}}}{d\bar\sigma} = \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}).
$$

Taking the posterior mean of the first equation gives $\bar{\mathbf{x}}_t = \hat{\mathbf{x}}_0 + \bar\sigma_t \hat{\boldsymbol\epsilon}$, so an Euler step from $\bar\sigma_t$ down to $\bar\sigma_s$ is $\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\hat{\boldsymbol\epsilon} = \hat{\mathbf{x}}_0 + \bar\sigma_s \hat{\boldsymbol\epsilon}$, and multiplied by $\alpha_s$,

$$
\mathbf{x}_s = \alpha_s\, \hat{\mathbf{x}}_0 + \sigma_s\, \hat{\boldsymbol\epsilon}.
$$

Thus a step predicts the clean point and the noise, and places them at the next level. :numref:`sec_diffusion-samplers` will meet this update again under the name DDIM. On the linear path it is the Euler step in $t$ itself, as substituting $\mathbf{x}_t = (1 - t)\hat{\mathbf{x}}_0 + t\hat{\boldsymbol\epsilon}$ into $\mathbf{x}_t - (t - s)(\hat{\boldsymbol\epsilon} - \hat{\mathbf{x}}_0)$ shows. On other paths it is a different method, and for data made of one point it is exact at any step length, where the Euler step in $t$ on the VP stride above is far off (Exercise :exref:`ex_integration-coordinates`).

`ode_sample` takes these steps through a list of levels $\bar\sigma$, which `sigma_bars` computes from the path's grid in $t$. At pure noise $\bar\sigma$ is infinite and $\bar{\mathbf{x}}$ undefined, but the step needs only $\hat{\mathbf{x}}_0$ and $\hat{\boldsymbol\epsilon}$, which the predictor computes from the noise itself, so the first step starts from $\mathbf{x}$.

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

A sampler needs $\hat{\mathbf{x}}_0$ and $\hat{\boldsymbol\epsilon}$ at any $(\alpha, \sigma)$, and `make_pred` turns a trained network of any kind into such a predictor. In this section each network runs on its own path, and `make_pred` only converts its prediction into the pair. Later sections run a network on another path's grid; then `make_pred` first finds the time on the network's own path with the same log-SNR (by bisection in `time_at`) and rescales the point to that path. A noise model has one blind spot. At pure noise, $\alpha = 0$, the clean point $\hat{\mathbf{x}}_0 = (\mathbf{x} - \sigma\hat{\boldsymbol\epsilon})/\alpha$ is undefined, so a noise model cannot start a sampler on the linear path; on the VP path $\alpha$ stays positive, about $0.007$ at $t = 1$, and that is where we use it.

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

### The Learned Field

Does the trained velocity network behave like the exact field, landing on the data mean after one step and in the modes after ten? We run it from $2{,}000$ standard Gaussian starts with one, ten and a hundred steps of `ode_sample`. Two more rows make the numbers readable: $2{,}000$ draws from the data themselves, the gray points of the plot, and the oracle taking the same ten steps from the same starts. Last, the cell runs `ode_sample` with the oracle against the plain Euler loop with `mix.velocity`. On the linear path the two take identical steps, so they should agree up to rounding.

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method-1}
%%tab pytorch
torch.manual_seed(3)
x_T = torch.randn(2000, 2, device=device)
pred_v = make_pred(ema_v, linear, 'velocity')
samples = [ode_sample(pred_v, x_T, linear.sigma_bars(steps))
           for steps in (1, 10, 100)]
data = mix.sample(2000)
print('one step: mean', [round(v, 3) for v in samples[0].mean(0).tolist()],
      'std', [round(v, 3) for v in samples[0].std(0).tolist()])
print(f'{"data":>18}:', mix.summary(data))
for steps, x in zip((10, 100), samples[1:]):
    print(f'{steps:>3} steps, network:', mix.summary(x))
print('  10 steps, oracle:',
      mix.summary(ode_sample(oracle, x_T, linear.sigma_bars(10))))
exact = (ode_sample(oracle, x_T, linear.sigma_bars(100))
         - euler(mix.velocity, x_T, 100)).abs().max().item()
print(f'ode_sample against the loop on the exact field: {exact:.0e}')
d2l.scatter_panels(samples, ['1 step', '10 steps', '100 steps'], data=data);
```

```{.python .input #gaussian-paths-the-probability-flow-ode-and-euler-s-method-1}
%%tab jax
key, k_T = jax.random.split(key)
x_T = jax.random.normal(k_T, (2000, 2))
pred_v = make_pred(ema_v, linear, 'velocity')
samples = [ode_sample(pred_v, x_T, linear.sigma_bars(steps))
           for steps in (1, 10, 100)]
key, k_data = jax.random.split(key)
data = mix.sample(k_data, 2000)
print('one step: mean', [round(v, 3) for v in samples[0].mean(0).tolist()],
      'std', [round(v, 3) for v in samples[0].std(0).tolist()])
print(f'{"data":>18}:', mix.summary(data))
for steps, x in zip((10, 100), samples[1:]):
    print(f'{steps:>3} steps, network:', mix.summary(x))
print('  10 steps, oracle:',
      mix.summary(ode_sample(oracle, x_T, linear.sigma_bars(10))))
exact = float(jnp.abs(ode_sample(oracle, x_T, linear.sigma_bars(100))
                      - euler(mix.velocity, x_T, 100)).max())
print(f'ode_sample against the loop on the exact field: {exact:.0e}')
d2l.scatter_panels(samples, ['1 step', '10 steps', '100 steps'], data=data);
```

One step puts every sample at the data mean, about $(-0.5, -0.7)$, with a standard deviation of a few hundredths where the exact field has none: the network's field at $t = 1$ is not quite $\mathbf{x} - \mathbb{E}[\mathbf{x}_0]$, and these hundredths are its error there. Ten steps find the three modes, with weights within about $0.03$ of $0.5/0.3/0.2$, but about three percent of the samples stray between the modes. How many should? A Gaussian mode in the plane puts $e^{-9/2} \approx 1.1\%$ of its mass beyond three standard deviations, and the data row strays between $1.1$ and $1.5$ percent. The modes also come out too narrow: their *spread*, the mean squared distance of a sample that does not stray to its mode center (:numref:`sec_diffusion-denoising`), is $0.45$, against $0.48$ to $0.49$ for the data. Is that the network's fault or the steps'? The oracle row settles it. On the exact field, ten steps from the same starts give a spread of $0.46$ and about two to three percent strays. So the narrowing belongs to the steps, with at most about $0.01$ added by the network, and the extra strays are partly the steps' and partly the network's. The oracle's weights match the network's to within $0.003$: whatever separates the ten-step weights from $0.5/0.3/0.2$, it is not the network. Where an exact field exists, run the sampler on it before blaming the network. A hundred steps of the same network bring the strays down to the data's level, about $1.3$ percent, and the spread to within about $0.01$ of the data's. The weights then lie within about two standard errors of $2{,}000$ points, where one standard error is about $0.01$. On the exact field the library's sampler and the loop agree to about $10^{-4}$, single-precision rounding, as the algebra says they should.

## The Ancestral Chain and the Reverse SDE

DDPM samples differently. It runs the reverse chain of :numref:`fig_diffusion-forward-reverse`, one noisy step at a time, from $t = 1$ to $t = 0$. Why should a chain of noisy steps work at all, and how does it relate to the flow?

### The Ancestral Step

The reversal of a small forward step is nearly Gaussian :cite:`sohl2015deep`, and given the clean point it is exactly Gaussian. For $s < t$, Bayes' rule makes $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ proportional to the product of two Gaussians in $\mathbf{x}_s$, the forward step $q(\mathbf{x}_t \mid \mathbf{x}_s)$ and the noising $q(\mathbf{x}_s \mid \mathbf{x}_0)$, so it is Gaussian with a mean linear in $\mathbf{x}_t$ and $\mathbf{x}_0$. Ho et al.'s step puts $\hat{\mathbf{x}}_0$ for $\mathbf{x}_0$ in that mean and adds fresh noise with the variance $\beta = 1 - \alpha_t^2/\alpha_s^2$ that the forward step from $s$ to $t$ adds. It is called *ancestral* because each $\mathbf{x}_s$ is drawn given its parent $\mathbf{x}_t$, as in ancestral sampling of a graphical model. On the VP path, substituting $\hat{\mathbf{x}}_0 = (\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon})/\alpha_t$ and the score $\nabla \log p_t = -\hat{\boldsymbol\epsilon}/\sigma_t$ into the mean and simplifying, the step reads

$$
\mathbf{x}_s = \frac{\mathbf{x}_t + \beta\, \nabla \log p_t(\mathbf{x}_t)}{\sqrt{1 - \beta}} + \sqrt{\beta}\, \mathbf{z},
\qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}).
$$
:eqlabel:`eq_diffusion-ancestral-score`

Is this a new sampler or the old one with noise added? It is instructive to expand $1/\sqrt{1 - \beta} \approx 1 + \beta/2$ and keep the terms of first order in $\beta$, together with the noise $\sqrt\beta\, \mathbf{z}$. The step then splits into two:

$$
\mathbf{x}_s \approx \Big[ \mathbf{x}_t + \tfrac{\beta}{2} \nabla \log p_t(\mathbf{x}_t) + \sqrt{\beta}\, \mathbf{z} \Big] + \tfrac{\beta}{2} \big( \mathbf{x}_t + \nabla \log p_t(\mathbf{x}_t) \big).
$$

The bracket is the Langevin step $\mathbf{x} \leftarrow \mathbf{x} + \frac{h}{2} \nabla \log p + \sqrt{h}\, \boldsymbol\xi$ of :numref:`sec_diffusion-denoising` with step size $h = \beta$ on the density $p_t$: it leaves $p_t$ in place and pulls stray points back toward it. The second term is one Euler step of the probability-flow ODE from $t$ back to $s$. On the VP path the marginal velocity is $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$: this is the score–velocity identity of :numref:`sec_diffusion-gaussian-paths` with $\dot\alpha_t/\alpha_t = -\beta(t)/2$ and $\alpha_t^2 + \sigma_t^2 = 1$. Here $\beta(t) = 0.1 + 19.9\, t$ is the rate of the VP schedule, and a short step has $\beta \approx \beta(t)\, (t - s)$. The Euler step from $t$ back to $s$ therefore moves the point by $-(t - s)\, \mathbf{u}_t = \tfrac12 \beta(t)\, (t - s)\, (\mathbf{x}_t + \nabla \log p_t) \approx \tfrac{\beta}{2} (\mathbf{x}_t + \nabla \log p_t)$, which is the second term. An ancestral step is a flow step plus a Langevin correction at the current noise level. The chain does what the ODE does, and its Langevin half pulls points back toward $p_t$, which can undo part of an earlier error; whether it does so in practice is measured in :numref:`sec_diffusion-samplers`.

### The Reverse SDE

In continuous time the reversal becomes exact. No code in this section needs that limit, but papers write their samplers in its notation, so it is worth one piece of notation. A *stochastic differential equation* (SDE) $d\mathbf{x} = \mathbf{f}(\mathbf{x}, t)\, dt + g(t)\, d\mathbf{W}$ is shorthand for the limit, as $\Delta t \to 0$, of the steps

$$
\mathbf{x} \leftarrow \mathbf{x} + \mathbf{f}\, \Delta t + g \sqrt{\Delta t}\, \mathbf{z}, \qquad \mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}):
$$

a deterministic *drift* $\mathbf{f}$ plus Gaussian noise whose variance grows in proportion to the time step. The forward chain of DDPM is such a step with $\Delta t = 1/T$, $\mathbf{f} = -\tfrac12 \beta(t)\, \mathbf{x}$ and $g^2 = \beta(t)$, because $\sqrt{1 - \beta_k} \approx 1 - \beta_k/2$ and $\beta_k \approx \beta(t)\, \Delta t$. :citet:`Anderson.1982` showed that such a process is reversed, from $t = 1$ down to $t = 0$, by the steps $\mathbf{x} \leftarrow \mathbf{x} - [\mathbf{f} - g^2 \nabla \log p_t(\mathbf{x})]\, \Delta t + g \sqrt{\Delta t}\, \mathbf{z}$ in the limit $\Delta t \to 0$: in that limit they have the same marginals $p_t$ as the forward process. Papers write the reversal as an SDE in which time runs backward,

$$
d\mathbf{x} = \big[ \mathbf{f} - g^2 \nabla \log p_t(\mathbf{x}) \big]\, dt + g\, d\bar{\mathbf{W}},
\qquad\textrm{with the same marginals as}\qquad
\frac{d\mathbf{x}}{dt} = \mathbf{f} - \tfrac12 g^2 \nabla \log p_t(\mathbf{x}),
$$
:eqlabel:`eq_diffusion-reverse`

where $d\bar{\mathbf{W}}$ stands for the fresh noise of each backward step. Why do the two share their marginals? Their backward steps differ by $\tfrac12 g^2 \nabla \log p_t\, \Delta t + g \sqrt{\Delta t}\, \mathbf{z}$, which is a Langevin step of size $h = g^2 \Delta t$ on $p_t$, and a Langevin step leaves $p_t$ in place: the split of the ancestral step, in continuous time. For the VP path the ODE on the right is $\mathbf{u}_t = -\tfrac12 \beta(t)\, (\mathbf{x} + \nabla \log p_t)$, the probability-flow ODE :eqref:`eq_diffusion-flow-ode` of that path. The reversed steps read $\mathbf{x} + \tfrac12 \beta(t) \Delta t\, \mathbf{x} + \beta(t) \Delta t\, \nabla \log p_t + \sqrt{\beta(t) \Delta t}\, \mathbf{z}$, which is the expanded ancestral step with $\beta = \beta(t)\, \Delta t$: the ancestral chain is, to first order, the simplest discretization of the reverse SDE, the Euler--Maruyama method (:numref:`sec_mdl-euler-maruyama`). Between the ODE, which adds no noise, and the SDE lies a whole family of samplers with the same marginals: the more noise a sampler injects, the stronger the score term in its drift that pulls the extra spread back (:numref:`sec_mdl-time-reversal`, :numref:`fig_mdl-dyn-lambda-family`).

`ancestral_sample` takes the exact step, not its first-order form, on any path and with any predictor `pred(x, alpha, sigma) -> (x0, eps)`. It moves to the mean of $\mathbf{x}_s$ given $\mathbf{x}_t$ and $\hat{\mathbf{x}}_0$, and adds noise with the variance $\sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ of the forward step from $s$ to $t$, which is $\beta$ on the VP path.

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

We run two hundred steps of the chain with the noise model, from the same $2{,}000$ starts as the flow. Ho et al. took a thousand steps for images; in the plane two hundred are plenty.

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

The chain recovers the three modes with the right weights, up to the sampling error of $2{,}000$ points, and its strays stay at about one percent or below, like those of the flow at a hundred steps.

## Samples from Both Samplers

So far each sampler ran once on $2{,}000$ points, too few to tell an error of the network from the luck of the draw. We now compare the network with the oracle under identical conditions: $20{,}000$ starts, two hundred ancestral steps of the noise model and a hundred Euler steps of the velocity model, each repeated with `oracle` in place of the network from the same starts and, for the chain, with the same injected noise. Every difference between a network row and its oracle row is then the network's doing. At this sample size a mode weight carries a standard error of about $0.0035$. The cell also counts the starts that network and oracle send to different modes.

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

Network and oracle send only $0.2$ to $0.5$ percent of the starts to different modes, and every weight of a network row lies within $0.004$ of its oracle row. What separates the rows from $0.5/0.3/0.2$ is the sampling error of $20{,}000$ points, which the oracle shares. The spreads lie within about $0.01$ of the data's, and every sampler, network or oracle, strays about one percent of the time, as the data themselves do. These strays are not errors at all: they are the $1.1\%$ of each Gaussian mode that lies beyond three standard deviations. The network leaves one trace. From the same starts, the flow strays a little more often with the network than with the oracle, about $1.3$ against $1.1$ percent, while the two chains stray about equally often. For the mode weights, on this example, the network is as good as the exact posterior mean, and both samplers recover the weights and the spread of the data together, which Langevin dynamics at a single noise level could not.

Which sampler should one use? On the mixture, at these step counts, it hardly matters: the mode weights of both are limited by the number of samples, not by the sampler or the network. They differ in what else they offer. The ODE is a deterministic map, so a sample can be reproduced from its start, interpolated between starts, or run backward from data to noise. The chain gives up that map for its Langevin half, which can pull points that earlier steps moved off back toward $p_t$. When steps are few these properties decide, and :numref:`sec_diffusion-samplers` measures which one matters more.

With a path, a target and a sampler in hand, the models of the literature are choices along three axes (:numref:`tab_diffusion_models`).

:The models of this chapter: the target and path that train the network, and the sampler that produces data.
:label:`tab_diffusion_models`

| model | target and path | sampler | where |
|:--|:--|:--|:--|
| denoising score model | the score $-\boldsymbol\epsilon/\sigma$, at one level | Langevin dynamics | :numref:`sec_diffusion-denoising` |
| noise-conditional score network | the score at each level of a ladder on the VE path, weighted by $\tilde\sigma^2$ | annealed Langevin dynamics | :numref:`sec_diffusion-gaussian-paths` |
| DDPM | $\boldsymbol\epsilon$ on the VP path | the ancestral chain or the probability-flow ODE | :numref:`sec_diffusion-gaussian-paths`, here |
| flow matching, rectified flow | $\boldsymbol\epsilon - \mathbf{x}_0$ on the linear path | Euler steps on the probability-flow ODE | :numref:`sec_diffusion-gaussian-paths`, here |
| $\mathbf{v}$-prediction | $\mathbf{v}$ on the cosine path | either | :numref:`sec_diffusion-gaussian-paths`, :numref:`sec_diffusion-design-space` |
| EDM | the clean point on the VE path, rescaled per noise level | a second-order solver of the probability-flow ODE | :numref:`sec_diffusion-samplers`, :numref:`sec_diffusion-design-space` |

The last row looks ahead. EDM's solver, more accurate than Euler's method, is the subject of :numref:`sec_diffusion-samplers`, and its rescaling of the network's input and output that of :numref:`sec_diffusion-design-space`.

In the plane we could check the network against the oracle, and a hundred evaluations per sample cost nothing. An image model has neither luxury: no oracle tells us how far it is from the posterior mean, and every evaluation is a pass through a large network. The next section trains one on Fashion-MNIST and has to find another way to judge its samples.

## Summary

A velocity field moves probability mass, and the marginal velocity of a Gaussian path moves the noise into the data, because the mass that all conditional clouds carry across a point is the density there times their posterior-average velocity. This conservation of mass is why the probability-flow ODE, solved exactly, yields samples with the right mode weights; what remains is the error of the steps and of the network. On the mixture, one Euler step lands every start on the data mean, and a hundred recover the data. In the coordinates $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ and $\bar\sigma = \sigma_t/\alpha_t$ every Gaussian path becomes $d\bar{\mathbf{x}}/d\bar\sigma = \hat{\boldsymbol\epsilon}$, and a step predicts the clean point and the noise and places them at the next level. DDPM's ancestral step is, to first order, an Euler step of the same flow plus a Langevin step that holds $p_t$ in place, a discretization of Anderson's reverse-time SDE. Run with the trained networks from the same starts as the exact posterior mean, both samplers send only $0.2$ to $0.5$ percent of the starts to a different mode, and their mode weights agree with the oracle's to within $0.004$.

## Exercises

1. **One step in two coordinate systems.**
    1. On the linear path, show that the step $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \hat{\boldsymbol\epsilon}$ equals the Euler step $\mathbf{x}_t - (t - s)\, \mathbf{u}_t(\mathbf{x}_t)$ in $t$.
    1. Let the data be the single point $\mathbf{x}_0 = \mathbf{0}$, on any Gaussian path. Show that $\hat{\boldsymbol\epsilon} = \mathbf{x}/\sigma_t$, that the exact trajectory through $\mathbf{x}_t$ is $\mathbf{x}_s = (\sigma_s/\sigma_t)\, \mathbf{x}_t$, and that the step of the first part follows it exactly for every $s < t$.
    1. On the VP path the marginal velocity of these data is $\mathbf{u}_t(\mathbf{x}) = (\dot\sigma_t/\sigma_t)\, \mathbf{x}$, with $d\sigma_t^2/dt = \beta(t)\, \alpha_t^2$ and $\beta(t) = 0.1 + 19.9\, t$. Take one step from $t = 0.5$, where $\alpha_t \approx 0.28$ and $\sigma_t \approx 0.96$, to $s = 0.1$, where $\sigma_s \approx 0.32$. By what factor does the Euler step in $t$ multiply $\mathbf{x}$, and by what factor should it? What does this suggest about the coordinates in which a sampler with few steps should work?
   :exlabel:`ex_integration-coordinates`
1. **The ancestral step to first order, for Gaussian data.** ● Let the data be $\mathcal{N}(\mathbf{0}, v_0 \mathbf{I})$, so that on the VP path $p_t = \mathcal{N}(\mathbf{0}, v_t \mathbf{I})$ with $v_t = \alpha_t^2 v_0 + \sigma_t^2$.
    1. Show that one ancestral step :eqref:`eq_diffusion-ancestral-score` from $t$ to $s$ with the exact score and $\beta = 1 - \alpha_t^2/\alpha_s^2$ maps $p_t$ to $\mathcal{N}\big(\mathbf{0}, (v_s + \delta)\mathbf{I}\big)$ with $\delta = \beta^2 (1/v_t - 1)/(1 - \beta)$.
    1. For which data is the step exact? Is the variance too large or too small for data narrower than the standard Gaussian, such as one component of the mixture, $v_0 = 0.25$?
    1. How does the error accumulated over the chain change when the grid is refined to twice as many steps, each with about half the $\beta$?
   :exlabel:`ex_gaussian-paths-ancestral-first-order`
1. **Step size and Gaussian reversal.** ● The ancestral chain uses Gaussian transitions, which are justified for small steps. This problem measures how small. Write $\sigma_{t|s}^2 = \sigma_t^2 - (\alpha_t/\alpha_s)^2 \sigma_s^2$ for the variance of the forward step from $s$ to $t > s$.
    1. Show that the exact reversal is the mixture $q(\mathbf{x}_s \mid \mathbf{x}_t) = \mathbb{E}_{\mathbf{x}_0 \mid \mathbf{x}_t}\big[q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)\big]$ and that its covariance is $\tau^2 \mathbf{I} + c^2\, \textrm{Cov}[\mathbf{x}_0 \mid \mathbf{x}_t]$, where $\tau^2 = \sigma_{t|s}^2 \sigma_s^2/\sigma_t^2$ is the variance of $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$ and $c = \alpha_s \sigma_{t|s}^2/\sigma_t^2$ is the coefficient of $\mathbf{x}_0$ in its mean.
    1. Let the data be $\pm 1$ with equal probability in one dimension, and let $x_t = 0$. Show that the reversal is the equal mixture of $\mathcal{N}(\pm c, \tau^2)$, which is bimodal exactly when $\zeta^2 = c^2/\tau^2$ exceeds one, and that $\zeta^2 = \alpha_s^2 \sigma_{t|s}^2 / (\sigma_t^2 \sigma_s^2)$. How does $\zeta$ shrink as the step $t - s$ is refined?
    1. Evaluate $\zeta$ on the VP path from $t = 0.5$ to $s = 0.495$, $0.45$ and $0.3$, and for one step from $t = 0.3$ to $s = 0$. Which reversals are bimodal? What does a Gaussian transition with the optimal mean produce at $x_t = 0$ in the last case, and what should it produce?
   :exlabel:`ex_gaussian-paths-gaussian-reversal`
1. [code] **Where each start ends.** The five starts of this section showed the heaviest mode collecting the center of the noise.
    1. Integrate the exact field `mix.velocity` with `euler` and a hundred steps from a grid of starts with spacing $0.02$ on $[-4, 4]^2$, and label each end by its nearest mode mean. Show the three regions of starts.
    1. Weight each start by the standard Gaussian density and add up the weight of each region. Compare the sums with $0.5$, $0.3$ and $0.2$. Which property of the marginal velocity makes them agree?
    1. Repeat with ten and with three steps. How do the masses of the regions change, and how is the change related to the one-step result?
    1. The Langevin chains of :numref:`sec_diffusion-denoising` with the exact score of the data kept the weights of their starts. What information does the flow use that a chain on the score of the data alone does not have?
   :exlabel:`ex_integration-catchment`
1. [code] **Commitment time.** ● When along the path does a sample decide its mode? :citet:`Biroli.Bonnaire.deBortoli.ea.2024` measure this by cloning: start at a point of $p_t$, run two copies of the reverse chain with independent noise, and record whether they end in the same mode.
    1. Draw $2{,}000$ clean points, noise them to $\mathbf{x}_t$ on the VP path for $t = 0.9, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1$, and run two copies of `ancestral_sample(pred_eps, x_t, vp, steps, t_start=t)` from each with about $200\, t$ steps. Plot the fraction $\varphi(t)$ of pairs whose ends have the same nearest mean. Which value does $\varphi$ approach as $t \to 1$, and why?
    1. Let $\Lambda$ be the largest eigenvalue of the covariance of the data. Biroli et al. define the speciation time by $\alpha_t^2 \Lambda = 1$, where the signal along the top principal direction stops dominating the noise. Compute it for the mixture and compare it with your curve and with the annealing ladder of :numref:`sec_diffusion-gaussian-paths`, whose coarse levels decided the mode weights. Exercise :exref:`ex_mdl-flow-matching-transport-late-decisions` of :numref:`sec_mdl-flow-matching-transport` and Exercise :exref:`ex_mdl-fokker-planck-probability-flow-mode-merging` of :numref:`sec_mdl-fokker-planck-probability-flow` work out the same question for two modes in one dimension.
   :exlabel:`ex_gaussian-paths-commitment-time`
1. **Same ratio, same flow.** ● Let two Gaussian paths have coefficients related by $\alpha^B_t = c_t\, \alpha^A_t$ and $\sigma^B_t = c_t\, \sigma^A_t$ for a smooth positive function $c_t$, so that they have the same log-SNR at every time.
    1. Show that $p^B_t(\mathbf{y}) = c_t^{-d}\, p^A_t(\mathbf{y}/c_t)$. Use the score–velocity identity of :numref:`sec_diffusion-gaussian-paths`, $\mathbf{u}_t(\mathbf{x}) = (\dot\alpha_t/\alpha_t)\, \mathbf{x} - \big(\sigma_t \dot\sigma_t - \sigma_t^2 \dot\alpha_t/\alpha_t\big) \nabla \log p_t(\mathbf{x})$, to show that $\mathbf{u}^B_t(\mathbf{y}) = (\dot c_t / c_t)\, \mathbf{y} + c_t\, \mathbf{u}^A_t(\mathbf{y}/c_t)$. Conclude that the trajectories of the probability-flow ODE are related by $\mathbf{y}_t = c_t\, \mathbf{x}_t$.
    1. Let $\tau(t)$ be increasing. Show that the path $p_{\tau(t)}$ is generated by the field $\dot\tau(t)\, \mathbf{u}_{\tau(t)}$, with the same trajectories as $\mathbf{u}$.
    1. Find the change of time $\tau(t)$ for which the cosine path $(\cos\frac{\pi\tau}{2}, \sin\frac{\pi\tau}{2})$ at time $\tau(t)$ has the log-SNR of the linear path at time $t$, and show that $c_t = 1/\sqrt{(1 - t)^2 + t^2}$. Where along the path is the rescaling largest?
    1. What does the probability-flow ODE of the cosine path do to data distributed as $\mathcal{N}(\mathbf{0}, \mathbf{I})$, and why? What does that of the linear path do to the same data?
   :exlabel:`ex_gaussian-paths-same-ratio-same-flow`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17.3]{.kicker}

Sampling by Integration<br>
**mass moves with the field · Euler steps and the data mean · coordinates for every path · the ancestral chain · network against oracle**
:::
:::

::: {.slide title="Arrows Toward the Data Are Not Yet Samples of the Data"}
[The problem]{.kicker}

- Data: three modes with weights $0.5$, $0.3$, $0.2$.
- Langevin chains on the *exact* score of the data end near the modes,
  with the right spread, but keep their starts' shares: a third each.
- An annealed ladder of ten hand-set levels got the shares right, with no
  reason why it had to.
- A trained network now gives a velocity at every point and noise level.

::: {.d2l-note .warn}
Why should following that field give the right share in each mode?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- explain why the marginal velocity carries noise into data (continuity
  equation);
- predict where one Euler step from noise lands, and what more steps add;
- write a sampler step as $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \hat{\boldsymbol\epsilon}$;
- split an ancestral step into a flow step and a Langevin step;
- measure a network against the exact posterior mean.

**Idea:** a velocity field conserves mass, so the marginal velocity carries
noise into data with the right weights.

**Running example:** the three-mode mixture, its exact field, and the
velocity and noise networks.
:::

::: {.slide title="The Third Section Turns the Trained Field into Samples"}
[You are here]{.kicker}

::: {.cols .vc}
::: {.col}
| section | question |
|---|---|
| Denoising and the Score | what does a denoiser know? |
| Gaussian Paths | one network for every level |
| **Sampling by Integration** | **why does following the field sample?** |
| Diffusion for Images | network, guidance, scores |
| Samplers, Inversion and Editing | fewer steps; run backwards |
| Likelihood, Weighting and the Design Space | is it maximum likelihood? |
| Latent Diffusion and Transformers | why a compressed code? |
| Few-Step Generation | why does one step fail? |
| Discrete Diffusion | noise for a token |
:::

::: {.col .narrow}
- **Uses:** the Gaussian path, the marginal velocity, the noise
  predictor, Langevin steps.
- **Skips:** few-step samplers, images (later decks).
:::
:::
:::

::: {.slide title="Each Network Predicts a Posterior Mean on Its Own Path"}
[From the previous deck]{.kicker}

- Path: $\mathbf{x}_t = \alpha_t\mathbf{x}_0 + \sigma_t\boldsymbol\epsilon$,
  data at $t = 0$, noise at $t = 1$.
- Linear path: $(\alpha_t, \sigma_t) = (1 - t, t)$. VP path:
  $\alpha_t^2 + \sigma_t^2 = 1$, rate $\beta(t) = 0.1 + 19.9\,t$.
- Velocity target $\dot\alpha_t\mathbf{x}_0 + \dot\sigma_t\boldsymbol\epsilon$;
  its posterior mean is the marginal velocity $\mathbf{u}_t(\mathbf{x})$.
- Noise target $\boldsymbol\epsilon$; the score is
  $\nabla\log p_t = -\hat{\boldsymbol\epsilon}/\sigma_t$.

::: {.d2l-note}
Velocity network on the linear path, noise network on the VP path.
:::
:::

::: {.slide title="Both Networks Sit on Their Loss Floors after 20,000 Steps"}
[Running example]{.kicker}

@!integration-sampling-by-integration-3

- Velocity model on the linear path: floor about $4.9$.
- Noise model on the VP path: floor about $0.54$.
- The *oracle*: the mixture's exact $\hat{\mathbf{x}}_0$ and
  $\hat{\boldsymbol\epsilon}$ at any point and level.
:::

::: {.slide title="A Velocity Field Changes a Density Only by Its Mass Flow"}
[Transport]{.kicker}

$$\frac{\partial p_t(\mathbf{x})}{\partial t} = -\nabla \cdot \big(p_t(\mathbf{x})\, \mathbf{u}_t(\mathbf{x})\big)$$

- $p_t\mathbf{u}_t$: mass crossing a point per unit time.
- No mass is created or lost: the density changes by inflow minus outflow.
- If $(p_t, \mathbf{u}_t)$ satisfy this, points that follow
  $\mathbf{u}_t$ keep the density $p_t$.
:::

::: {.slide title="The Clouds' Mass Flows Add Up to the Density Times the Marginal Velocity"}
[Transport]{.kicker}

Each cloud $\mathcal{N}(\alpha_t\mathbf{x}_0, \sigma_t^2\mathbf{I})$ moves
correctly with its own velocity
$\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0) = \dot\alpha_t\mathbf{x}_0 + \dot\sigma_t\boldsymbol\epsilon$.
Average over the data, with Bayes' rule:

$$\mathbb{E}_{\mathbf{x}_0}\big[p_t(\mathbf{x}\mid\mathbf{x}_0)\,\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)\big]
= p_t(\mathbf{x})\,\underbrace{\mathbb{E}\big[\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)\mid\mathbf{x}_t=\mathbf{x}\big]}_{\mathbf{u}_t(\mathbf{x})}$$

::: {.d2l-note}
The marginal velocity moves the whole mixture of clouds, which is $p_t$.
:::
:::

::: {.slide title="Trajectories of One Field Never Meet, So No Sample Follows a Conditional Path"}
[Transport]{.kicker}

- Conditional paths: straight lines (linear path) that cross everywhere.
- Trajectories of one field: never at one point at the same time; one
  velocity per point and time.
- What is reproduced is the *density* at every time, not any path.

::: {.d2l-note}
Solved exactly, the flow conserves mass: half the noise ends in the mode of
weight one half. Errors come only from the steps and the network.
:::
:::

::: {.slide title="The Flow ODE Carries Noise to Data; Euler Steps Follow It"}
[Sampling]{.kicker}

$$\frac{d\mathbf{x}}{dt} = \mathbf{u}_t(\mathbf{x}) \textrm{ from } t = 1 \textrm{ to } 0,
\qquad \textrm{Euler: } \mathbf{x} \leftarrow \mathbf{x} - \tfrac{1}{K}\, \mathbf{u}_t(\mathbf{x})$$

- One data point: $\mathbf{u}_t(\mathbf{x}) = (\mathbf{x} - \mathbf{x}_0)/t$,
  straight lines; one step is exact.
- Many points: the paths bend where the posterior is undecided.
- At $t = 1$ the posterior is the prior:
  $\mathbf{u}_1(\mathbf{x}) = \mathbf{x} - \mathbb{E}[\mathbf{x}_0]$.
:::

::: {.slide title="Check: Where Does One Euler Step from Noise Land?"}
[Check]{.kicker}

Means $(-2.5, -1.5)$, $(2.5, -1.5)$, $(0, 2.5)$ with weights $0.5$,
$0.3$, $0.2$. Where does $\mathbf{x} \leftarrow \mathbf{x} -
\mathbf{u}_1(\mathbf{x})$ send each start?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** Every start goes to $\mathbb{E}[\mathbf{x}_0] = (-0.5, -0.7)$,
between the modes, where no data lie.
:::
:::
:::

::: {.slide title="On the Exact Field, Ten Steps Reach the Modes and the Origin Joins the Heaviest"}
[Sampling]{.kicker}

@!gaussian-paths-the-probability-flow-ode-and-euler-s-method

- One step: all five starts on the mean.
- Ten steps: within about $0.2$ of the hundred-step end, a little short.
- Three of five starts, the origin among them, end in the mode of weight
  $0.5$.

::: {.d2l-note}
Conservation of mass: the heaviest mode collects the largest region of starts.
:::
:::

::: {.slide title="Divided by Alpha, Every Path Becomes One ODE in the Noise Level"}
[Coordinates]{.kicker}

$$\bar{\mathbf{x}} = \frac{\mathbf{x}_t}{\alpha_t} = \mathbf{x}_0 + \bar\sigma\,\boldsymbol\epsilon,
\quad \bar\sigma = \frac{\sigma_t}{\alpha_t}
\quad\Rightarrow\quad \frac{d\bar{\mathbf{x}}}{d\bar\sigma} = \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}})$$

Why: on VP, $\sigma_t = 0.96$ at $t = 0.5$ and $0.32$ at $t = 0.1$, so a long
step in $t$ misses. One Euler step in $\bar\sigma$, with
$\bar{\mathbf{x}}_t = \hat{\mathbf{x}}_0 + \bar\sigma_t\hat{\boldsymbol\epsilon}$:

$$\mathbf{x}_s = \alpha_s\,\hat{\mathbf{x}}_0 + \sigma_s\,\hat{\boldsymbol\epsilon}$$

::: {.d2l-note}
Predict the clean point and the noise; place them at the next level. Exact
for one-point data on every path; on the linear path, Euler in $t$.
:::
:::

::: {.slide title="The Learned Field Needs About a Hundred Euler Steps"}
[Sampling]{.kicker}

@!gaussian-paths-the-probability-flow-ode-and-euler-s-method-1
:::

::: {.slide title="Ten Steps Narrow the Samples, and So Does the Exact Field"}
[Sampling]{.kicker}

- **Strays:** farther than $1.5$ (three mode s.d.) from every mean.
  **Spread:** mean squared distance of the rest to their mean.
- One step lands on the mean $(-0.5, -0.7)$.
- Ten steps: about 3% strays, spread $0.45$ against the data's $0.48$–$0.49$.
- A hundred steps: the data's strays (1.1–1.5%) and spread.

::: {.d2l-note}
The oracle at ten steps is nearly as narrow ($0.46$) and strays about 2–3%: the
steps cause the narrowing and part of the strays.
:::
:::

::: {.slide title="The Ancestral Step Is a Langevin Step Plus an Euler Step"}
[Ancestral chain]{.kicker}

DDPM's step on the VP path from $t$ to $s < t$, with
$\beta = 1 - \alpha_t^2/\alpha_s^2$ and $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$:

$$\mathbf{x}_s = \frac{\mathbf{x}_t + \beta\, \nabla \log p_t(\mathbf{x}_t)}{\sqrt{1 - \beta}} + \sqrt{\beta}\, \mathbf{z}$$

To first order in $\beta$:

$$\mathbf{x}_s \approx \underbrace{\mathbf{x}_t + \tfrac{\beta}{2} \nabla \log p_t + \sqrt{\beta}\, \mathbf{z}}_{\textrm{Langevin step: keeps } p_t}
+ \underbrace{\tfrac{\beta}{2} \big(\mathbf{x}_t + \nabla \log p_t\big)}_{\textrm{Euler step of the flow ODE}}$$

::: {.d2l-note}
The chain is the flow plus a correction that holds $p_t$ in place.
:::
:::

::: {.slide title="The Reverse SDE and the Flow ODE Share Their Marginals"}
[Ancestral chain]{.kicker}

Forward steps $\mathbf{x} \leftarrow \mathbf{x} + \mathbf{f}\,\Delta t + g\sqrt{\Delta t}\,\mathbf{z}$;
for VP, $\mathbf{f} = -\tfrac12 \beta(t)\, \mathbf{x}$, $g^2 = \beta(t)$.
Backward from $t = 1$, as $\Delta t \to 0$:

$$\mathbf{x} \leftarrow \mathbf{x} - \big[\mathbf{f} - g^2 \nabla \log p_t\big]\Delta t + g\sqrt{\Delta t}\,\mathbf{z}
\quad\textrm{or}\quad
\frac{d\mathbf{x}}{dt} = \mathbf{f} - \tfrac12 g^2 \nabla \log p_t$$

Ancestral chain: the left, to first order. Flow sampler: the right.

::: {.d2l-note}
They differ by $\tfrac12 g^2 \nabla\log p_t\,\Delta t + g\sqrt{\Delta t}\,\mathbf{z}$:
a Langevin step of size $g^2\Delta t$, which keeps $p_t$.
:::

[Anderson, 1982]{.small}
:::

::: {.slide title="Network and Oracle Disagree on the Mode of Under 1% of Starts"}
[Experiments]{.kicker}

@!gaussian-paths-samples-from-both-samplers
:::

::: {.slide title="Either Sampler Works on the Mixture"}
[Experiments]{.kicker}

- Shared starts and noise: 0.2 to 0.5% of the starts change mode; weights
  within $0.004$.
- Strays about 1%, as in the data (Euler with the network: 1.3% against
  1.1%).
- The ODE is a reproducible map: the same start always gives the same point.

::: {.d2l-note}
The chain's noise can correct earlier errors; on images, the samplers deck
measures whether it does.
:::
:::

::: {.slide title="Check: How Does a Noise Error Reach the Velocity near Pure Noise?"}
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

::: {.slide title="A Model Is a Path, a Target and a Sampler"}
[Summary]{.kicker}

| model | target and path | sampler |
|:--|:--|:--|
| NCSN | score on a ladder of VE levels | annealed Langevin |
| DDPM | $\boldsymbol\epsilon$, VP path | ancestral chain or flow ODE |
| flow matching | $\boldsymbol\epsilon - \mathbf{x}_0$, linear path | Euler on the flow ODE |
| $\mathbf{v}$-prediction | $\mathbf{v}$, cosine path | either |
| EDM (later decks) | clean point, VE path, rescaled per level | second-order steps |
:::

::: {.slide title="Wrap-up"}
[Wrap-up]{.kicker}

- **Transport:** the marginal velocity moves the mass of $p_t$, so the
  flow ODE, solved exactly, carries noise into data with the right weights.
- **Euler:** one step lands on the mean $(-0.5, -0.7)$; a hundred recover
  the modes, with about as many strays as the data.
- **Coordinates:** $\mathbf{x}_s = \alpha_s\hat{\mathbf{x}}_0 + \sigma_s\hat{\boldsymbol\epsilon}$ on every path.
- **Ancestral chain:** a flow step plus a Langevin step.
- **Network against oracle:** 0.2 to 0.5% of starts change mode.

**Next:** the same regression on 784-pixel images, with a U-Net, guidance
and a quality score.
:::
