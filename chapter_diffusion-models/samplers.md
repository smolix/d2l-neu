# Samplers, Inversion and Editing
:label:`sec_diffusion-samplers`

The last section drew its images with fifty Euler steps of the
probability-flow ODE, and the thousand-step ancestral chain of
:numref:`sec_diffusion-integration` scored only a little lower in FD, at
twenty times the cost. The cost is easy to count: every step of a sampler
evaluates the U-Net once, and everything else the step does is a few
additions. Two questions are therefore worth asking. Which of the thousand
steps do we need, and how few can we take? And once sampling is the
integration of a field, can we run it backwards, from a real image to the
noise that produces it, and edit the image from there?

:citet:`Song.Meng.Ermon.2020` answered the first question with an
observation about training. The regression of
:numref:`sec_diffusion-gaussian-paths` sees single noisy points
$\mathbf{x}_t$ of training examples, never a chain of them, so it determines
the distribution of $\mathbf{x}_t$ at each time and nothing more. Many
samplers produce those distributions, among them a deterministic one that
takes long strides and can be run in reverse, and one trained network serves
all of them.

We first derive this family of samplers and find that its deterministic
member is Euler's method. How few steps it needs then becomes a question a
numerical analyst would recognize: where should the steps go, and how
accurate should each one be? We answer it with numbers, on the mixture whose
exact field we know and on Fashion-MNIST, and turn the answer into a
procedure for choosing a sampler. The last part runs the sampler backwards
to edit images.

```{.python .input #samplers-samplers-inversion-and-editing}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #samplers-samplers-inversion-and-editing}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

```{.python .input #samplers-samplers-inversion-and-editing-1}
%%tab pytorch
torch.set_float32_matmul_precision('high')  # tf32 matmuls on tensor cores
```

## Deterministic and Stochastic Samplers

The two samplers of :numref:`sec_diffusion-integration`, the ancestral
chain and Euler's method, read the network in the same way, although one
adds fresh noise at every step and the other none. At a point
$\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ of a
Gaussian path the network gives an estimate $\hat{\mathbf{x}}_0$ of the
clean point and an estimate $\hat{\boldsymbol\epsilon}$ of the noise, and
each determines the other:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon}}{\alpha_t}.$$
:eqlabel:`eq_diffusion-x0-hat`

Whatever the network was trained to predict, `d2l.make_pred` of
:numref:`sec_diffusion-integration` turns it into a function `pred(x,
alpha, sigma)` returning this pair at any noise level, and the samplers of
this section read nothing else.

### The DDIM Family and Euler's Method

Take a time $t$ and an earlier time $s < t$. A sampler step from $t$ to $s$
must turn a point distributed like $\mathbf{x}_t$ into one distributed like
$\mathbf{x}_s$; the regression constrains nothing else. Write $\mathbf{x}_t
= \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$, choose a noise
scale $\tilde\sigma$ between $0$ and $\sigma_s$, and set

$$\mathbf{x}_s = \alpha_s \mathbf{x}_0 + \sqrt{\sigma_s^2 - \tilde\sigma^2}\, \boldsymbol\epsilon + \tilde\sigma\, \boldsymbol\xi, \qquad \boldsymbol\xi \sim \mathcal{N}(\mathbf{0}, \mathbf{I}),$$
:eqlabel:`eq_diffusion-ddim-family`

with $\boldsymbol\xi$ independent of $\boldsymbol\epsilon$. The two noise
variances add to $\sigma_s^2$, so given $\mathbf{x}_0$ the point
$\mathbf{x}_s$ has its correct distribution for every $\tilde\sigma$, and
since $\boldsymbol\epsilon = (\mathbf{x}_t - \alpha_t \mathbf{x}_0) /
\sigma_t$, each $\tilde\sigma$ defines a step from $\mathbf{x}_t$ to
$\mathbf{x}_s$. A sampler uses the network's estimates in place of
$\mathbf{x}_0$ and $\boldsymbol\epsilon$:

$$\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sqrt{\sigma_s^2 - \tilde\sigma^2}\, \hat{\boldsymbol\epsilon} + \tilde\sigma\, \boldsymbol\xi.$$
:eqlabel:`eq_diffusion-ddim-step`

The step moves the predicted clean point to level $s$ together with the
predicted noise, and replaces part of that noise by fresh noise.
:citet:`Song.Meng.Ermon.2020` measure the fresh part by a strength
$\eta \geq 0$,

$$\tilde\sigma = \eta\, \sigma_s \sqrt{1 - \kappa^2}, \qquad \kappa = \frac{\sigma_s \alpha_t}{\alpha_s \sigma_t}.$$
:eqlabel:`eq_diffusion-ddim-eta`

Here $\kappa^2 = \mathrm{SNR}_t / \mathrm{SNR}_s < 1$, with $\mathrm{SNR} =
\alpha^2 / \sigma^2$. At $\eta = 1$, $\tilde\sigma^2$ is the variance of
$\mathbf{x}_s$ given $\mathbf{x}_t$ and $\mathbf{x}_0$ under the noising
chain of :numref:`sec_diffusion-gaussian-paths` (Exercise
:exref:`ex_samplers-forward-step`). The step then samples that chain's
posterior, with the estimates in place of the truth. At $\eta = 0$ no fresh
noise enters and the step is deterministic. This is the *denoising diffusion
implicit model* (DDIM), called implicit because it defines its samples by a
map from noise rather than by a chain of densities.

`ddim_sample` runs the DDIM family on the grid of `path.grid` with any
predictor `pred(x, alpha, sigma) -> (x0, eps)`, stopping at `t_end` (data by
default) so that a loop can act on the sample between levels.

```{.python .input #samplers-ddim-is-euler-s-method}
%%tab pytorch
@torch.no_grad()  #@save
def ddim_sample(pred, x, path, steps, eta=0.0, *, t_start=1.0, t_end=0.0):
    """DDIM from x at t_start (1 is noise) to t_end; eta = 0 is deterministic.
    eta = 1 draws the posterior variance: not the ancestral sampler's."""
    a, s = path.grid(steps, t_start, t_end)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        ratio = s[i + 1] * a[i] / (a[i + 1] * s[i])     # kappa of the text
        var = (eta * s[i + 1]) ** 2 * (1 - ratio ** 2)   # fresh noise
        x = a[i + 1] * x0 + math.sqrt(s[i + 1] ** 2 - var) * eps
        if var > 0:
            x = x + math.sqrt(var) * torch.randn_like(x)
    return x
```

```{.python .input #samplers-ddim-is-euler-s-method}
%%tab jax
def ddim_sample(pred, x, path, steps, eta=0.0, key=None, *,  #@save
                t_start=1.0, t_end=0.0):
    """DDIM from x at t_start (1 is noise) to t_end; eta > 0 needs a key.
    eta = 1 draws the posterior variance: not the ancestral sampler's."""
    if eta > 0 and key is None:
        raise ValueError('ddim_sample with eta > 0 draws noise: pass a key')
    a, s = path.grid(steps, t_start, t_end)
    a_t, s_t, a_s, s_s = a[:-1], s[:-1], a[1:], s[1:]
    ratio = s_s * a_t / (a_s * s_t)                     # kappa of the text
    var = (eta * s_s) ** 2 * (1 - ratio ** 2)           # fresh noise
    def step(x, inputs):
        a_t, s_t, a_s, c_eps, c_noise, key = inputs
        x0, eps = pred(x, a_t, s_t)
        x = a_s * x0 + c_eps * eps
        if eta > 0:
            x = x + c_noise * jax.random.normal(key, x.shape)
        return x, None
    keys = (jax.random.split(key, steps) if eta > 0
            else jnp.zeros(steps))                      # unused at eta = 0
    return jax.lax.scan(step, x, (a_t, s_t, a_s, jnp.sqrt(s_s ** 2 - var),
                                  jnp.sqrt(var), keys))[0]
```

How large an error does a long deterministic stride make? Take data that are
themselves standard Gaussian, on the variance-preserving path. Then
$\mathbf{x}_t$ is standard Gaussian at every time, so the exact sampler
should move no point. The exact estimates are $\hat{\mathbf{x}}_0 = \alpha_t
\mathbf{x}_t$ and $\hat{\boldsymbol\epsilon} = \sigma_t \mathbf{x}_t$, and
one deterministic stride from $t = 1$ to $t = 0$ returns $\hat{\mathbf{x}}_0
= \alpha_1 \mathbf{x}_1$: with $\alpha_1 \approx 0.0066$ the whole cloud
collapses onto its mean. The step holds the noise estimate fixed across the
stride, while along the exact trajectory it changes.

At $\eta = 0$ the step is $\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 +
\sigma_s \hat{\boldsymbol\epsilon}$, the step that `ode_sample` takes in
:numref:`sec_diffusion-integration`. In the coordinates $\bar{\mathbf{x}} =
\mathbf{x} / \alpha_t$ and $\bar\sigma_t = \sigma_t / \alpha_t$ derived
there it reads

$$\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\, \hat{\boldsymbol\epsilon},$$

one Euler step of the probability-flow ODE $d\bar{\mathbf{x}} / d\bar\sigma =
\hat{\boldsymbol\epsilon}$, and the ratio $\kappa$ above is simply
$\bar\sigma_s / \bar\sigma_t$. On every Gaussian path, then, DDIM is
Euler's method, and with $\eta = 0$ `ddim_sample` and `ode_sample` on the
same levels compute one function in two coordinate systems.
:numref:`sec_mdl-ddim-guidance` proves the equivalence, together with the two extreme cases: a single data point is
integrated without error by strides of any length, because
$\hat{\boldsymbol\epsilon}$ is then constant along each trajectory, and
Gaussian data shrink at every stride (:numref:`fig_mdl-dyn-ddim-strides`).

### Fresh Noise and Long Steps

If one network serves every $\eta$, which strength should we use? With
$f = \tilde\sigma^2 / \sigma_s^2$, the step :eqref:`eq_diffusion-ddim-step`
reads

$$\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \Big(\sqrt{1 - f}\, \hat{\boldsymbol\epsilon} + \sqrt{f}\, \boldsymbol\xi\Big), \qquad f = \eta^2 \Big(1 - \frac{\mathrm{SNR}_t}{\mathrm{SNR}_s}\Big).$$

Every member places the clean estimate at level $s$ in the same way; a
fraction $f$ of the accompanying noise variance is fresh. On a fine grid $f$
is small even at $\eta = 1$; on a grid of ten steps the $\eta = 1$ sampler
replaces most of the noise at every step (Exercise
:exref:`ex_samplers-admissible-noise`).

There is a good argument for fresh noise. The ancestral step of
:numref:`sec_diffusion-integration` splits, to first order, into a Langevin
step, which leaves the current distribution $p_t$ in place and pulls stray
points back toward it, and an Euler step of the probability-flow ODE; every
reverse process that injects noise and keeps the marginals splits the same
way (:numref:`sec_mdl-time-reversal`). A stochastic sampler therefore keeps
pulling its state toward the model's distribution, and an error made at one
step may be partly forgotten at the next. A deterministic sampler carries
every error to the end.

There is an equally good argument against it. The Langevin pull is a
statement about short steps. Over a long step the exact reverse transition
is far from Gaussian (Exercise :exref:`ex_gaussian-paths-gaussian-reversal`
of :numref:`sec_diffusion-integration`), and a long stochastic step adds an
error of its own. The deterministic sampler should win at small budgets;
whether the stochastic one overtakes it once the steps are short, we measure
on images in :numref:`subsec_diffusion-samplers-quality`.
:numref:`fig_diffusion-two-samplers` shows both ends of the family on the
mixture.

![The two ends of the DDIM family on the three-mode mixture with the exact predictor, 200 steps on the variance-preserving path from three starting points at $t = 1$ (circles) to $t = 0$ (crosses); training data in gray. (a) At $\eta = 0$ each start follows a smooth curve into a mode. (b) At $\eta = 1$ the paths are jagged, and here two of the three end in a different mode than in (a); the first half of each path, where noise dominates, is drawn faded. Both produce the same distribution at $t = 0$.](../img/mdl-diffusion-two-samplers.svg)
:label:`fig_diffusion-two-samplers`

Two stochastic samplers must be kept apart. The ancestral chain adds the
variance of the forward step, $\sigma_t^2 - (\alpha_t / \alpha_s)^2
\sigma_s^2$, while $\eta = 1$ adds the posterior variance, smaller by the
factor $\sigma_s^2 / \sigma_t^2$. Both move to the same mean. The ancestral
chain is therefore not a member of the family: the noise it keeps and the
noise it adds have variances that sum to more than $\sigma_s^2$. Which
variance is right? Neither, in general. The posterior variance is exact when
the data are a single point, since then $\mathbf{x}_t$ leaves no doubt about
$\mathbf{x}_0$, and the forward-step variance is exact for standard Gaussian
data on the variance-preserving path. For data narrower than the noise, with
a variance per coordinate below one, the best Gaussian variance lies between
the two; for wider data it can exceed even the forward step's (Exercise
:exref:`ex_samplers-optimal-variance`). On a thousand-step grid the factor
is close to one except in the last few steps, where both variances are tiny,
so the choice hardly matters; on fifty steps it does.

## Fewer Steps

A first-order method makes an error proportional to its step size, so
skipping steps loses accuracy in a predictable way. We measure that loss
first where we can compute it exactly, then ask how much of it better
methods and better placed steps recover. A warning in advance: the numbers
we find on images belong to one small model and will not transfer to the
next one. The procedure that finds them will.

### Skipping Steps on the Running Example

We return to the three-mode mixture `mix` of
:numref:`sec_diffusion-denoising`, whose posterior mean, and with it the
exact pair $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$, we can compute
at every level; `pred_exact` wraps that pair in the interface of the
samplers. We draw 20,000 starting points and as many data points, and print
the data's *spread*, the mean squared distance of a point to the nearest
mode center (:numref:`sec_diffusion-denoising`). Each mode has variance
$0.25$ per coordinate, so the spread is just under $0.5$.

```{.python .input #samplers-skipping-steps-on-the-running-example-1}
%%tab pytorch
mix = d2l.GaussianMixture.three_modes()
vp, lin = d2l.GaussianPath.vp(), d2l.GaussianPath.linear()

def pred_exact(x, alpha, sigma):
    """The mixture's exact (x0, eps) at x = alpha x0 + sigma eps."""
    x0 = mix.posterior_mean(x, alpha, sigma)
    return x0, (x - alpha * x0) / sigma

torch.manual_seed(0)
z, data = torch.randn(20_000, 2), mix.sample(20_000)
print(f'data: spread {mix.summary(data)["spread"]:.3f}')
```

```{.python .input #samplers-skipping-steps-on-the-running-example-1}
%%tab jax
mix = d2l.GaussianMixture.three_modes()
vp, lin = d2l.GaussianPath.vp(), d2l.GaussianPath.linear()

def pred_exact(x, alpha, sigma):
    """The mixture's exact (x0, eps) at x = alpha x0 + sigma eps."""
    x0 = mix.posterior_mean(x, alpha, sigma)
    return x0, (x - alpha * x0) / sigma

k_z, k_data, k_eta = jax.random.split(jax.random.key(0), 3)  # k_eta: per n
z, data = jax.random.normal(k_z, (20_000, 2)), mix.sample(k_data, 20_000)
print(f'data: spread {mix.summary(data)["spread"]:.3f}')
```

With the exact predictor, any difference between a ten-step sample and a
thousand-step sample from the same start is error of the strides, and for a
first-order method it should halve each time the number of steps doubles.
We run DDIM on the variance-preserving grid with 5 to 1,000 steps and print
the mean distance to the thousand-step samples and the spread; beside the
spread of $\eta = 0$ stand those of $\eta = 1$ and of the ancestral chain
from the same starts.

```{.python .input #samplers-skipping-steps-on-the-running-example}
%%tab pytorch
samples = {n: d2l.ddim_sample(pred_exact, z, vp, n)
           for n in (5, 10, 20, 50, 200, 1000)}
spread = lambda pts: f'{mix.summary(pts)["spread"]:.3f}'
for n, x in samples.items():
    gap = (x - samples[1000]).norm(dim=1).mean()
    eta1 = d2l.ddim_sample(pred_exact, z, vp, n, eta=1.0)
    chain = d2l.ancestral_sample(pred_exact, z, vp, n)
    print(f'{n:4d} steps: distance to 1,000 steps {gap:.3f}, spread '
          f'{spread(x)} (eta = 1: {spread(eta1)}, ancestral: {spread(chain)})')
d2l.scatter_panels([samples[10][:2000], samples[200][:2000]],
                   ['10 steps', '200 steps'], data=data[:2000]);
```

```{.python .input #samplers-skipping-steps-on-the-running-example}
%%tab jax
samples = {n: d2l.ddim_sample(pred_exact, z, vp, n)
           for n in (5, 10, 20, 50, 200, 1000)}
spread = lambda pts: f'{mix.summary(pts)["spread"]:.3f}'
for n, x in samples.items():
    gap = jnp.linalg.norm(x - samples[1000], axis=1).mean()
    k1, k2 = jax.random.split(jax.random.fold_in(k_eta, n))
    eta1 = d2l.ddim_sample(pred_exact, z, vp, n, 1.0, k1)
    chain = d2l.ancestral_sample(pred_exact, z, vp, n, k2)
    print(f'{n:4d} steps: distance to 1,000 steps {gap:.3f}, spread '
          f'{spread(x)} (eta = 1: {spread(eta1)}, ancestral: {spread(chain)})')
d2l.scatter_panels([samples[10][:2000], samples[200][:2000]],
                   ['10 steps', '200 steps'], data=data[:2000]);
```

The distance halves each time the number of steps doubles, as predicted:
about 0.4 at five steps, 0.2 at ten and 0.04 at fifty, on a scale where the
modes are about five units apart. The error also has a direction. With ten
steps the spread of the deterministic samples is about 0.38 against the
data's 0.47 to 0.48, and the clouds in the left panel are visibly tighter
than the data behind them. A stride holds the noise estimate at its value at
the start, which for Gaussian data always removes too much noise (Exercise
:exref:`ex_samplers-gaussian-data`), and once the noise is small enough to
tell the modes apart, each mode of the mixture behaves like Gaussian data
with variance $0.25$ per coordinate.

The stochastic samplers miss in both directions. With ten steps $\eta = 1$
contracts the modes further, to a spread of about 0.32, and the ancestral
chain spreads them to about 0.52, beyond the data. It is surprising at first
that $\eta = 1$, which adds noise, ends tighter than $\eta = 0$, which adds
none. Every member splits the same budget $\sigma_s^2$ between kept noise
and fresh noise. For Gaussian data the clean estimate and the noise
estimate are both positive multiples of $\mathbf{x}_t$, so the kept noise
lines up with the clean estimate and adds to its spread through a cross
term. Fresh noise is independent of both and adds its variance and nothing
more. Moving the budget from kept to fresh noise, as $\eta = 1$ does on a
coarse grid, therefore narrows the sample (Exercise
:exref:`ex_samplers-gaussian-data` proves it). The ancestral chain keeps as
much noise as $\eta = 1$ but adds fresh noise beyond the budget, and the
excess suits data as wide as the noise, not a mode narrower than the noise.
On fine grids all three samplers reach the data's spread. A statistic that
can err in either direction cannot rank samplers by its size, so on images
we use a score that penalizes error in either direction.

### Second Order and the Step Grid

Euler's method holds the slope fixed across a step, so its error over the
whole run falls only in proportion to the step size. Heun's method
(:numref:`sec_mdl-euler-runge-kutta`) takes the Euler step as a trial,
evaluates the slope again at the trial point, and moves with the average of
the two slopes:

$$\bar{\mathbf{x}}' = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\, \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_t, \bar\sigma_t), \qquad \bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\, \frac{\hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_t, \bar\sigma_t) + \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}', \bar\sigma_s)}{2}.$$

Its error falls with the square of the step size, at two evaluations per
step. For standard Gaussian data the noise estimate is
$\hat{\boldsymbol\epsilon} = \bar\sigma \bar{\mathbf{x}} / (1 +
\bar\sigma^2)$, and the exact flow multiplies $\bar{\mathbf{x}}$ by
$\sqrt{(1 + \bar\sigma_s^2) / (1 + \bar\sigma_t^2)}$. From $\bar\sigma_t =
1$ to $\bar\sigma_s = 0.5$ the exact factor is $0.791$, Euler's step gives
$0.75$ and Heun's $0.80$: errors of $0.041$ and $0.009$, about a quarter of
the error for twice the evaluations. Two steps cannot take a second slope.
At $\bar\sigma = 0$ there is no noise left to estimate, and a step from
$\bar\sigma = \infty$ is infinitely long and can be written only in the
form $\hat{\mathbf{x}}_0 + \bar\sigma_s \hat{\boldsymbol\epsilon}$. The
last step, to $\bar\sigma = 0$, and a first step from $\bar\sigma = \infty$
are therefore Euler steps, and `heun_sample` spends $2n - 1$ evaluations
on $n$ steps, or $2n - 2$ from pure noise on the linear path.

Whether the second evaluation helps depends on where the steps are. On a
one-dimensional mixture with the exact field, Heun's method on an even grid
loses to Euler's below about forty evaluations, while on a grid crowded
toward the data it wins at every budget, and the same crowding more than
doubles the error of Euler's method
(:numref:`sec_mdl-sampling-learned-dynamics`).
:citet:`Karras.Aittala.Aila.ea.2022` separated two choices that earlier
samplers had made together, the order of the method and the placement of its
steps, and crowd the steps by spacing $\bar\sigma^{1/\rho}$ evenly,

$$\bar\sigma_i = \Big(\bar\sigma_{\max}^{1/\rho} + \tfrac{i}{n - 1}\big(\bar\sigma_{\min}^{1/\rho} - \bar\sigma_{\max}^{1/\rho}\big)\Big)^{\rho}, \qquad i = 0, \ldots, n - 1,$$

followed by a last step to $\bar\sigma = 0$, with $\bar\sigma_{\max} = 80$,
$\bar\sigma_{\min} = 0.002$ and $\rho = 7$ for images in $[-1, 1]$. The grid
is a choice of the sampler, separate from the training path, since
`make_pred` translates any level to the network's own time. `edm_grid`
returns these levels and `heun_sample` takes Heun steps through any list of
levels. We print the ten-step grid of each of three choices: even in time
on the variance-preserving path, even in time on the linear path, and the
grid of Karras et al., which we call EDM after their paper.

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab pytorch
def edm_grid(steps, sigma_max=80.0, sigma_min=0.002, rho=7):  #@save
    """Karras et al.'s levels sigma_bar: even in sigma_bar^(1/rho), then 0."""
    hi, lo = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return [(hi + i / (steps - 1) * (lo - hi)) ** rho
            for i in range(steps)] + [0.0]

@torch.no_grad()  #@save
def heun_sample(pred, x, sigma_bars):
    """Heun on d x_bar / d sigma_bar = eps: an Euler trial, then the average
    of the two slopes; from x at the first level to data, as ode_sample."""
    r0 = sigma_bars[0]
    xb = x if math.isinf(r0) else x * math.sqrt(1 + r0 ** 2)  # x / alpha
    for r_t, r_s in zip(sigma_bars[:-1], sigma_bars[1:]):
        x0, eps = d2l.pred_bar(pred, xb, r_t)
        x_next = x0 + r_s * eps                       # the Euler step
        if r_s > 0 and not math.isinf(r_t):           # no slope at 0 or inf
            x_next = xb + (r_s - r_t) * (
                eps + d2l.pred_bar(pred, x_next, r_s)[1]) / 2
        xb = x_next
    return xb

for name, levels in (('VP', vp.sigma_bars(10)), ('linear', lin.sigma_bars(10)),
                     ('EDM', d2l.edm_grid(10))):
    print(f'{name:7s}', ' '.join(f'{r:7.3g}' for r in levels))
```

```{.python .input #samplers-second-order-and-the-step-grid}
%%tab jax
def edm_grid(steps, sigma_max=80.0, sigma_min=0.002, rho=7):  #@save
    """Karras et al.'s levels sigma_bar: even in sigma_bar^(1/rho), then 0."""
    hi, lo = sigma_max ** (1 / rho), sigma_min ** (1 / rho)
    return [(hi + i / (steps - 1) * (lo - hi)) ** rho
            for i in range(steps)] + [0.0]

def heun_sample(pred, x, sigma_bars):  #@save
    """Heun on d x_bar / d sigma_bar = eps: an Euler trial, then the average
    of the two slopes; from x at the first level to data, as ode_sample."""
    with jax.ensure_compile_time_eval():          # the grid is a constant
        r = jnp.asarray(sigma_bars)
        from_inf = bool(jnp.isinf(r[0]))
    def step(xb, r):
        r_t, r_s = r
        x0, eps = d2l.pred_bar(pred, xb, r_t)
        x_next = x0 + r_s * eps                   # the Euler step
        correct = lambda _: xb + (r_s - r_t) * (
            eps + d2l.pred_bar(pred, x_next, r_s)[1]) / 2
        heun = jnp.isfinite(r_t) & (r_s > 0)      # no slope at inf or 0
        return jax.lax.cond(heun, correct, lambda _: x_next, None), None
    xb = x if from_inf else x * jnp.sqrt(1 + r[0] ** 2)  # x / alpha
    return jax.lax.scan(step, xb, (r[:-1], r[1:]))[0]

for name, levels in (('VP', vp.sigma_bars(10)), ('linear', lin.sigma_bars(10)),
                     ('EDM', d2l.edm_grid(10))):
    print(f'{name:7s}', ' '.join(f'{float(r):7.3g}' for r in levels))
```

Grids that are even in time look nothing alike in $\bar\sigma$. The
variance-preserving grid spends four of its ten steps between $152$ and
$6.1$ and reaches the data from $0.34$. The linear grid covers everything
above $9$ in a single step from infinity and spends seven steps between $4$
and $0.11$. The EDM grid starts at $80$, also spends four steps above $4$,
and crowds its last three levels below $0.12$. Which allocation suits a
model cannot be read off the levels; we measure it on images below.

### Exponential Integrators and Multistep Solvers

Heun's method needs a second evaluation per step for its second order. A
cheaper route starts from the coordinates we have been using, and it turns
out that we have been on it all along. In $\mathbf{x}$, the probability-flow
ODE has a part that is linear in $\mathbf{x}$ and owes nothing to the
network: on the variance-preserving path, for instance, the drift shrinks
$\mathbf{x}$ at the rate $\tfrac12\beta(t)$ whatever the score is, where
$\beta(t)$ is the rate of the noise schedule. The exact solution of this
linear part is an exponential decay, here the factor $\alpha_t$, and
dividing by $\alpha_t$ removes it. What remains, $d\bar{\mathbf{x}} /
d\bar\sigma = \hat{\boldsymbol\epsilon}$, contains only the network.
Integrating the linear part exactly and approximating only the network's
term is the idea of an *exponential integrator*
:cite:`Zhang.Chen.2023,Lu.Zhou.Bao.ea.2022`. DDIM, which holds
$\hat{\boldsymbol\epsilon}$ constant over the step, is its first-order
member, so the change of coordinates that made DDIM look like Euler's method
was the main idea of these solvers. The higher-order members differ only in
how they let $\hat{\boldsymbol\epsilon}$ vary across the step. DPM-Solver
:cite:`Lu.Zhou.Bao.ea.2022` evaluates it a second time, at the midpoint of
the step in $\log \bar\sigma$ (Exercise :exref:`ex_samplers-dpm-solver`).

The second evaluation can be avoided. The previous step has already
evaluated $\hat{\boldsymbol\epsilon}$ at the level we just left, and a
*multistep* method reuses it: it extrapolates $\hat{\boldsymbol\epsilon}$
linearly from the last two estimates to the middle of the current step. With
$\Delta = \bar\sigma_s - \bar\sigma_t$ for the current step, $\Delta'$ for
the previous one, and $\hat{\boldsymbol\epsilon}_t$,
$\hat{\boldsymbol\epsilon}'$ the estimates at their starts, the
second-order Adams--Bashforth step is

$$\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + \Delta \Big(\hat{\boldsymbol\epsilon}_t + \frac{\Delta}{2\Delta'}\big(\hat{\boldsymbol\epsilon}_t - \hat{\boldsymbol\epsilon}'\big)\Big).$$

When the noise estimate does not change from one step to the next, this is
DDIM; otherwise the correction follows the change, at one evaluation per
step. The first step has no previous estimate and is an Euler step, and so
is the step after a start at $\bar\sigma = \infty$, where $\Delta'$ is
infinite. DPM-Solver++(2M) :cite:`Lu.Zhou.Bao.ea.2025` applies the same idea
to the clean-image estimate and extrapolates in $\log \bar\sigma$; it and
its relatives are the usual choice at ten to twenty-five evaluations.
`multistep_sample` takes the step above through any list of levels.

```{.python .input #samplers-second-order-and-the-step-grid-1}
%%tab pytorch
@torch.no_grad()
def multistep_sample(pred, x, sigma_bars):
    """Second-order Adams-Bashforth on d x_bar / d sigma_bar = eps: one
    evaluation per step, the slope extrapolated from the last two."""
    r = sigma_bars
    xb = x if math.isinf(r[0]) else x * math.sqrt(1 + r[0] ** 2)  # x / alpha
    eps_prev = 0
    for k in range(len(r) - 1):
        x0, eps = d2l.pred_bar(pred, xb, r[k])
        c = ((r[k + 1] - r[k]) ** 2 / (2 * (r[k] - r[k - 1]))
             if k > 0 and math.isfinite(r[k - 1]) else 0.0)  # 0: Euler
        xb = x0 + r[k + 1] * eps + c * (eps - eps_prev)
        eps_prev = eps
    return xb
```

```{.python .input #samplers-second-order-and-the-step-grid-1}
%%tab jax
def multistep_sample(pred, x, sigma_bars):
    """Second-order Adams-Bashforth on d x_bar / d sigma_bar = eps: one
    evaluation per step, the slope extrapolated from the last two."""
    with jax.ensure_compile_time_eval():          # the grid is a constant
        r = [float(v) for v in sigma_bars]
    c = [0.0] + [(r[k + 1] - r[k]) ** 2 / (2 * (r[k] - r[k - 1]))
                 if math.isfinite(r[k - 1]) else 0.0     # 0: Euler
                 for k in range(1, len(r) - 1)]
    def step(carry, inputs):
        xb, eps_prev = carry
        r_t, r_s, c = inputs
        x0, eps = d2l.pred_bar(pred, xb, r_t)
        return (x0 + r_s * eps + c * (eps - eps_prev), eps), None
    xb = x if math.isinf(r[0]) else x * math.sqrt(1 + r[0] ** 2)  # x / alpha
    return jax.lax.scan(step, (xb, jnp.zeros_like(xb)),
                        (jnp.array(r[:-1]), jnp.array(r[1:]),
                         jnp.array(c)))[0][0]
```

### Quality against Evaluations on Images
:label:`subsec_diffusion-samplers-quality`

On images the field is learned. We train the class-conditional U-Net of
:numref:`sec_diffusion-images` (about 1.2 million parameters) on the
velocity of the linear path for ten thousand updates, a few minutes on one
GPU. A tenth of its labels are replaced by a null label, and we sample from
the moving average of its weights. This one network serves the rest of the
section. Samplers without guidance ($\gamma = 1$ below) ask for every class
equally often (sample $i$ for class $i$ mod 10), guided ones ($\gamma > 1$)
also evaluate the null label, and inversion and editing give each image its
own label.

```{.python .input #samplers-quality-against-evaluations-on-images-1}
%%tab pytorch
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()
loss = d2l.regression_loss(lin, 'velocity')

def batch_fn():
    i = torch.randint(0, len(X), (128,), device=X.device)
    drop = torch.rand(128, device=X.device) < 0.1   # label dropout
    return X[i], torch.where(drop, 10, y[i])        # 10 is the null label

def loss_fn(net, batch):  # the regression loss of the labeled field
    x0, c = batch
    return loss(d2l.conditional(net, c), x0)

torch.manual_seed(0)
net = d2l.UNet(ch=64, num_classes=10).to(X.device)
net, ema, _ = d2l.train_regression(net, loss_fn, batch_fn, 10_000)
```

```{.python .input #samplers-quality-against-evaluations-on-images-1}
%%tab jax
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()
loss = d2l.regression_loss(lin, 'velocity')

def batch_fn(key):
    k1, k2 = jax.random.split(key)
    i = jax.random.randint(k1, (128,), 0, len(X))
    drop = jax.random.uniform(k2, (128,)) < 0.1     # label dropout
    return X[i], jnp.where(drop, 10, y[i])          # 10 is the null label

def loss_fn(net, batch, key):  # the regression loss of the labeled field
    x0, c = batch
    return loss(d2l.conditional(net, c), x0, key)

net = d2l.UNet(ch=64, num_classes=10, rngs=nnx.Rngs(0))
net, ema, _ = d2l.train_regression(net, loss_fn, batch_fn, 10_000,
                                   jax.random.key(0))
```

We score 1,000 samples by their Fréchet distance (FD) to 1,000 test images
in `d2l.FeatureCNN` features (:numref:`sec_diffusion-images`). At this size
two sets of real images already score a little under 1 against each other,
and the score moves by 0.15 to 0.2 from one draw of images to the next
(:numref:`subsec_diffusion-fd-reference`). Every measurement is therefore
repeated on three draws of 1,000 noise images, shared by all samplers so
that every comparison is paired, and reported as the mean plus or minus half
the range; the floor compares the test images with three other real sets. A
difference within the half-ranges is not a difference. `fd_runs` also takes
a guidance scale $\gamma$, which we use in the next subsection; $\gamma = 1$
is the plain conditional field.

```{.python .input #samplers-quality-against-evaluations-on-images-2}
%%tab pytorch
cnn = d2l.fashion_feature_cnn(X, y)
features = lambda imgs: d2l.fd_features(cnn, imgs)
f_test = features(X_test[:1000])
floors = [d2l.fid_score(features(X_test[k:k + 1000]), f_test)
          for k in (1000, 2000, 3000)]
labels = torch.arange(1000, device=X.device) % 10   # sample i has class i % 10
gen = torch.Generator(X.device).manual_seed(1)
noises = [torch.randn(1000, 1, 28, 28, device=X.device, generator=gen)
          for _ in range(3)]                        # shared by every sampler

def fd_runs(sampler, n, gamma=1.0):
    """FD of sampler(pred, noise, n) on each noise draw, field cfg(gamma)."""
    pred = d2l.make_pred(d2l.cfg(ema, labels, gamma), lin, 'velocity')
    return [d2l.fid_score(features(sampler(pred, z, n).clamp(-1, 1)), f_test)
            for z in noises]

def mean_range(fds):  # the mean, plus or minus half the range of the draws
    return f'{np.mean(fds):5.2f}±{(max(fds) - min(fds)) / 2:.2f}'

print('floor, 1,000 real images against 1,000 others: FD', mean_range(floors))
```

```{.python .input #samplers-quality-against-evaluations-on-images-2}
%%tab jax
cnn = d2l.fashion_feature_cnn(X, y)
features = lambda imgs: d2l.fd_features(cnn, imgs)
f_test = features(X_test[:1000])
floors = [d2l.fid_score(features(X_test[k:k + 1000]), f_test)
          for k in (1000, 2000, 3000)]
labels = jnp.arange(1000) % 10                      # sample i has class i % 10
k_noise, k_run = jax.random.split(jax.random.key(1))
noises = [jax.random.normal(k, (1000, 28, 28, 1))   # shared by every sampler
          for k in jax.random.split(k_noise, 3)]
run_keys = iter(jax.random.split(k_run, 1000))      # fresh noise of the samplers

@nnx.jit(static_argnums=(0, 1, 2))  # one trace per sampler, step count, gamma
def sample(sampler, n, gamma, model, c, z, key):
    pred = d2l.make_pred(d2l.cfg(model, c, gamma), lin, 'velocity')
    return sampler(pred, z, n, key).clip(-1, 1)

def fd_runs(sampler, n, gamma=1.0):
    """FD of sampler(pred, noise, n, key) on each noise draw, field cfg(gamma)."""
    keys = jax.random.split(next(run_keys), len(noises))
    return [d2l.fid_score(features(sample(sampler, n, gamma, ema, labels, z, k)),
                          f_test) for z, k in zip(noises, keys)]

def mean_range(fds):  # the mean, plus or minus half the range of the draws
    return f'{np.mean(fds):5.2f}±{(max(fds) - min(fds)) / 2:.2f}'

print('floor, 1,000 real images against 1,000 others: FD', mean_range(floors))
```

The floor is about 0.9. The first table settles the question left open
above, on the variance-preserving path with even steps in time, so that DDIM
with $\eta = 0$ is Euler's method on the first grid printed above. If noise
corrects errors that a deterministic sampler carries to the end, $\eta = 1$
and the ancestral chain should close on $\eta = 0$ as the steps shorten and
overtake it before a thousand.

```{.python .input #samplers-quality-against-evaluations-on-images-3}
%%tab pytorch
vp_samplers = {  # the DDIM family and the ancestral chain, even steps on VP
    'DDIM, eta = 0': lambda p, z, n: d2l.ddim_sample(p, z, vp, n),
    'DDIM, eta = 1': lambda p, z, n: d2l.ddim_sample(p, z, vp, n, eta=1.0),
    'ancestral': lambda p, z, n: d2l.ancestral_sample(p, z, vp, n)}
steps = (10, 20, 50, 100, 200, 1000)
torch.manual_seed(2)  # the fresh noise of the stochastic samplers
fd_vp = {name: [fd_runs(f, n) for n in steps]
         for name, f in vp_samplers.items()}
print('steps        ', ''.join(f'{n:>11d}' for n in steps))
for name, row in fd_vp.items():
    print(f'{name:13s}', ''.join(f'{mean_range(v):>11s}' for v in row))
```

```{.python .input #samplers-quality-against-evaluations-on-images-3}
%%tab jax
vp_samplers = {  # the DDIM family and the ancestral chain, even steps on VP
    'DDIM, eta = 0': lambda p, z, n, k: d2l.ddim_sample(p, z, vp, n),
    'DDIM, eta = 1': lambda p, z, n, k: d2l.ddim_sample(p, z, vp, n, 1.0, k),
    'ancestral': lambda p, z, n, k: d2l.ancestral_sample(p, z, vp, n, k)}
steps = (10, 20, 50, 100, 200, 1000)
fd_vp = {name: [fd_runs(f, n) for n in steps]
         for name, f in vp_samplers.items()}
print('steps        ', ''.join(f'{n:>11d}' for n in steps))
for name, row in fd_vp.items():
    print(f'{name:13s}', ''.join(f'{mean_range(v):>11s}' for v in row))
```

They close on it but never overtake it. Up to two hundred steps $\eta = 0$
scores between a third and nearly a half lower than $\eta = 1$ at every
budget, far outside the half-ranges; at a thousand the three end within
about 0.2 of each other, between about 1.3 and 1.5. On this model fresh
noise never helped: at no budget did a stochastic sampler score below
$\eta = 0$. The ancestral chain loses most, with about 8 to 11 at
fifty steps, worse than DDIM at ten (a little over 5). Its forward-step
variance suits data as wide as the noise, but a garment is nearly determined
once the noise is small, and the excess variance should show as speckle; the
image rows at the end of the next subsection show it. DDIM itself changes by
about a tenth after a hundred steps, and at fifty it is within about a third
of the thousand-step chain at a twentieth of the evaluations, which is the
central result of :citet:`Song.Meng.Ermon.2020`.

The second table counts network evaluations rather than steps, since that is
what a sample costs. It runs Euler's method, Heun's method and the multistep
sampler at about 10 to 100 evaluations on three grids: the linear grid, the
EDM grid of Karras et al. with $\rho = 7$, and the same with $\rho = 3$,
which moves levels from the data end toward high noise. Heun's method takes
about half as many steps, and on the EDM grids, which start at a finite
level, one evaluation fewer than the budget.

```{.python .input #samplers-quality-against-evaluations-on-images-6}
%%tab pytorch
grids = {'linear': lin.sigma_bars, 'EDM': d2l.edm_grid,
         'EDM, rho = 3': lambda n: d2l.edm_grid(n, rho=3)}
solvers = {'Euler': d2l.ode_sample, 'Heun': d2l.heun_sample,
           'multistep': multistep_sample}
budgets = (10, 20, 50, 100)  # network evaluations per sample

def on_grid(solver, grid):  # a sampler (pred, noise, steps)
    return lambda p, z, n: solver(p, z, grid(n))

def num_steps(s, g, b):  # Heun skips its second slope only at sigma = inf, 0:
    # b // 2 + 1 steps on the linear grid (b evaluations, the first level is
    # infinite), b // 2 on the EDM grids (b - 1)
    return (b // 2 + (g == 'linear')) if s == 'Heun' else b

fd_ode = {(s, g): [fd_runs(on_grid(solver, grid), num_steps(s, g, b))
                   for b in budgets]
          for s, solver in solvers.items() for g, grid in grids.items()}
print('evaluations            ', ''.join(f'{b:>11d}' for b in budgets))
for (s, g), row in fd_ode.items():
    print(f'{s + ", " + g:23s}', ''.join(f'{mean_range(v):>11s}' for v in row))
```

```{.python .input #samplers-quality-against-evaluations-on-images-6}
%%tab jax
grids = {'linear': lin.sigma_bars, 'EDM': d2l.edm_grid,
         'EDM, rho = 3': lambda n: d2l.edm_grid(n, rho=3)}
solvers = {'Euler': d2l.ode_sample, 'Heun': d2l.heun_sample,
           'multistep': multistep_sample}
budgets = (10, 20, 50, 100)  # network evaluations per sample

def on_grid(solver, grid):  # a sampler (pred, noise, steps, key)
    return lambda p, z, n, k: solver(p, z, grid(n))

def num_steps(s, g, b):  # Heun skips its second slope only at sigma = inf, 0:
    # b // 2 + 1 steps on the linear grid (b evaluations, the first level is
    # infinite), b // 2 on the EDM grids (b - 1)
    return (b // 2 + (g == 'linear')) if s == 'Heun' else b

fd_ode = {(s, g): [fd_runs(on_grid(solver, grid), num_steps(s, g, b))
                   for b in budgets]
          for s, solver in solvers.items() for g, grid in grids.items()}
print('evaluations            ', ''.join(f'{b:>11d}' for b in budgets))
for (s, g), row in fd_ode.items():
    print(f'{s + ", " + g:23s}', ''.join(f'{mean_range(v):>11s}' for v in row))
```

Where do the two tables meet? The plot puts the first table's samplers and
the three solvers on the linear grid on one axis of network evaluations.

```{.python .input #samplers-quality-against-evaluations-on-images-4}
%%tab pytorch, jax
curves = {**{name: (steps, row) for name, row in fd_vp.items()},
          **{f'{s}, linear': (budgets, fd_ode[s, 'linear']) for s in solvers}}
d2l.plot([x for x, _ in curves.values()] + [[10, 1000]],
         [[np.mean(v) for v in row] for _, row in curves.values()]
         + [[np.mean(floors)] * 2], 'network evaluations', 'FD',
         legend=list(curves) + ['floor'], xscale='log', yscale='log',
         figsize=(6, 4), fmts=('o-', 's-', '^-', 'v--', 'D--', 'x-.', 'k:'))
```

The second-order methods level off first. Heun's method on the linear grid
scores about 1.2 from twenty evaluations on, the multistep sampler gets
there by fifty, and Euler's method needs fifty to a hundred. The level they
reach, about 1.1 to 1.3, is where DDIM and the ancestral chain end at a
thousand steps, about 0.3 above the floor, and no sampler gets below it. We
call it the *plateau*: what remains there is the model's error, which no
sampler removes.

The grid matters as much as the method. With ten evaluations Euler's method
scores about 3.2 on the linear grid, a little over 5 on the
variance-preserving one and about 6.5 on EDM; Heun's method scores about 1.5
on the linear grid and 12 to 15 on EDM. The EDM grid loses for every method
at every budget, by far at 10 and 20 evaluations. At 50 and 100 it still
loses by 0.2 to 0.4 for Euler's method, well outside the half-ranges; for
the second-order methods the gap there is 0.1 to 0.2, no larger than the
half-ranges, so at those budgets crowding does not help them and we cannot
say that it hurts. Half of the one-dimensional result transfers, then:
crowding the steps toward the data hurts Euler's method, but Heun's gain
from crowding does not appear.

Why does the paper's grid lose here? The grid printout suggests two
culprits, the crowding at the data end and the four steps spent at high
noise. The rows with $\rho = 3$ separate them, since they crowd the data end
less and spend more steps at high noise. If the crowding were at fault they
would do better; at ten and twenty evaluations they do worse (Heun's method
scores 80 to 90 with nine evaluations), so the wasted steps are the ones at
high noise. At 50 and 100 evaluations the two EDM grids differ by about 0.2
or less. Above $\bar\sigma \approx 4$, several times the standard deviation
of the pixels themselves (about 0.7), all noisy images look alike and the
trajectories are nearly straight. The linear grid crosses that range in two
steps, from infinity to 9 and from 9 to 4, while EDM spends four of its ten
steps there and keeps three for the range in which a garment's shape is
decided. Why the grid serves the models of Karras et al., we can only
suggest. Their networks were trained on noise levels drawn log-normally
around $\bar\sigma \approx 0.3$, the low-noise range their grid crowds, ours
with $t$ uniform on the linear path; and they chose $\rho = 7$ at 35 to 79
evaluations. We have tested neither explanation.

### Integration Error, Guidance and the Choice of a Sampler

So far FD alone has judged the samplers, and FD scores a set of samples. It
does not say how far each sample lies from the point that an exact
integration of the field would reach from the same noise image. To see what
a solver loses beyond what FD notices, we integrate every noise image with
200 evaluations of Heun's method as a reference and measure each solver's
root-mean-square distance per pixel to it.

We do this for two fields, since guidance changes the field being
integrated. Classifier-free guidance
(:numref:`subsec_diffusion-images-guided`) extrapolates from the unlabeled
to the labeled prediction, $\tilde{\mathbf{u}} = \mathbf{u}_\varnothing +
\gamma(\mathbf{u}_c - \mathbf{u}_\varnothing)$, so $\gamma = 1$ is the
conditional field and each guided evaluation costs two network passes. A
field that extrapolates beyond both of its parts should also change faster
along a trajectory, and we expect every solver to need more evaluations to
follow it at $\gamma = 2$.

```{.python .input #samplers-quality-against-evaluations-on-images-7}
%%tab pytorch
def endpoints(solver, n, gamma):
    """Samples of a solver with n steps on the linear grid, per noise draw."""
    pred = d2l.make_pred(d2l.cfg(ema, labels, gamma), lin, 'velocity')
    return [solver(pred, z, lin.sigma_bars(n)).clamp(-1, 1) for z in noises]

for gamma in (1.0, 2.0):
    refs = endpoints(d2l.heun_sample, 101, gamma)    # 200 evaluations
    fds = [d2l.fid_score(features(x), f_test) for x in refs]
    print(f'gamma = {gamma:.0f}, reference: FD {mean_range(fds)}; '
          'FD and RMS distance to it')
    print('evaluations', ''.join(f'{b:>18d}' for b in (10, 20, 50)))
    for s, solver in solvers.items():
        cols = []
        for b in (10, 20, 50):
            xs = endpoints(solver, b // 2 + 1 if s == 'Heun' else b, gamma)
            fds = [d2l.fid_score(features(x), f_test) for x in xs]
            rms = np.mean([((x - r) ** 2).mean().sqrt().item()
                           for x, r in zip(xs, refs)])
            cols.append(f'  {mean_range(fds)} {rms:.3f}')
        print(f'  {s:9s}', ''.join(cols))
```

```{.python .input #samplers-quality-against-evaluations-on-images-7}
%%tab jax
def endpoints(solver, n, gamma):
    """Samples of a solver with n steps on the linear grid, per noise draw."""
    keys = jax.random.split(next(run_keys), len(noises))  # unused: no noise
    return [sample(on_grid(solver, lin.sigma_bars), n, gamma, ema, labels,
                   z, k) for z, k in zip(noises, keys)]

for gamma in (1.0, 2.0):
    refs = endpoints(d2l.heun_sample, 101, gamma)    # 200 evaluations
    fds = [d2l.fid_score(features(x), f_test) for x in refs]
    print(f'gamma = {gamma:.0f}, reference: FD {mean_range(fds)}; '
          'FD and RMS distance to it')
    print('evaluations', ''.join(f'{b:>18d}' for b in (10, 20, 50)))
    for s, solver in solvers.items():
        cols = []
        for b in (10, 20, 50):
            xs = endpoints(solver, b // 2 + 1 if s == 'Heun' else b, gamma)
            fds = [d2l.fid_score(features(x), f_test) for x in xs]
            rms = np.mean([float(jnp.sqrt(((x - r) ** 2).mean()))
                           for x, r in zip(xs, refs)])
            cols.append(f'  {mean_range(fds)} {rms:.3f}')
        print(f'  {s:9s}', ''.join(cols))
```

The two measures disagree, and the disagreement teaches something about FD.
At ten evaluations without guidance, Heun's samples lie farther from the
reference than Euler's (a distance of 0.12 to 0.13 against about 0.1), yet
their FD is half of Euler's (about 1.5 against about 3.2). FD scores the
distribution of the samples, not the error of each one: a solver can draw
the wrong garment for a given noise image and still draw a realistic set.

Guidance makes the field harder to follow, as expected. At $\gamma = 2$ the
reference itself scores about 2.1 to 2.3, since guidance trades realism for
agreement with the class. At twenty evaluations every solver is still 0.4
to 1 above it; by fifty the multistep sampler has reached the reference's
range or its edge, while Heun's method is 0.15 to 0.2 above it and Euler's
about 0.3. The distances confirm the prediction only in part. At twenty
evaluations Heun's distance to the reference is a third to a half larger
than without guidance, but Euler's and the multistep sampler's have barely
moved, and at ten and twenty evaluations the multistep sampler is the
closest of the three. Under guidance, then, only the multistep sampler
reaches the plateau by fifty evaluations, a hundred network passes; Heun's
and Euler's methods need more, and the table does not say how many.

The integration error can also be seen. Below are eight images from each of
four samplers, from the same eight noise images and labels: the ancestral
chain with 50 steps, DDIM with 10 and with 50 steps, and Heun's method on
the linear grid with 20 evaluations.

```{.python .input #samplers-quality-against-evaluations-on-images-5}
%%tab pytorch
z8 = noises[0][:8]
pred8 = d2l.make_pred(d2l.conditional(ema, labels[:8]), lin, 'velocity')
rows = [d2l.ancestral_sample(pred8, z8, vp, 50),
        d2l.ddim_sample(pred8, z8, vp, 10), d2l.ddim_sample(pred8, z8, vp, 50),
        d2l.heun_sample(pred8, z8, lin.sigma_bars(11))]
d2l.show_gray(torch.cat(rows).clamp(-1, 1), 4, 8);
```

```{.python .input #samplers-quality-against-evaluations-on-images-5}
%%tab jax
z8 = noises[0][:8]
pred8 = d2l.make_pred(d2l.conditional(ema, labels[:8]), lin, 'velocity')
rows = [d2l.ancestral_sample(pred8, z8, vp, 50, next(run_keys)),
        d2l.ddim_sample(pred8, z8, vp, 10), d2l.ddim_sample(pred8, z8, vp, 50),
        d2l.heun_sample(pred8, z8, lin.sigma_bars(11))]
d2l.show_gray(jnp.concatenate(rows).clip(-1, 1), 4, 8);
```

The three deterministic rows match column by column, since the
deterministic samplers approximate one map from noise to images; ten steps
smooth away details, such as prints and edges, that fifty steps and Heun's
method agree on. The ancestral row is grainy, with speckled backgrounds and
textures. This is the excess variance of the first table, made visible.

How, then, should we choose a sampler for a model we have not measured? Fix
a set of noise images and a score, start with a deterministic solver on the
grid of the model's own training path, and double the number of evaluations
until the score stops moving by more than its run-to-run range. What remains
is the model's error; the best a sampler can do is stop adding its own. On
this model the procedure stops where the curves of the plot level off,
about twenty evaluations for Heun's method, and fresh noise never helps.
Under guidance it stops later, at about fifty evaluations of the multistep
sampler, which with one evaluation per step and the smallest integration
error is the better default there. A grid taken from a paper should pass the
same test before it replaces the training grid. At ten evaluations no solver
has reached the plateau, and reaching it in one or two steps needs a
different kind of model (:numref:`sec_diffusion-few-step`).

## Inversion and Editing

A deterministic sampler integrates a differential equation, and a
differential equation can be run in either direction. Running it backwards
gives each image a noise image, its *code*, from which the sampler
reproduces the image. Codes allow interpolation between images. The same
sampler, started from a partly noised image or constrained at every step,
edits and inpaints, all without retraining.

### The Latent Code

`ddim_invert` runs the deterministic sampler upward, from an image at
$t = 0$ toward noise. Each step goes from a level $t$ to the next level
$t' > t$ by the Euler step

$$\bar{\mathbf{x}}_{t'} = \bar{\mathbf{x}}_t + (\bar\sigma_{t'} - \bar\sigma_t)\, \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_t, \bar\sigma_t),$$

with the noise estimate read at the lower end, the only point the encoder
has. The first step, at $\bar\sigma = 0$ where the predictor is undefined,
reads the image with the noise level of the next level instead.

```{.python .input #samplers-the-latent-code}
%%tab pytorch
@torch.no_grad()  #@save
def ddim_invert(pred, x, path, steps, t_end=1.0):
    """Deterministic DDIM run backwards, from data x at t = 0 to t_end."""
    a, s = path.grid(steps, 0.0, t_end)
    for i in range(steps):
        j = max(i, 1)  # at t = 0 (sigma = 0) x is read at the next level
        eps = pred(x, a[j], s[j])[1]
        x = a[i + 1] * (x - s[i] * eps) / a[i] + s[i + 1] * eps
    return x
```

```{.python .input #samplers-the-latent-code}
%%tab jax
def ddim_invert(pred, x, path, steps, t_end=1.0):  #@save
    """Deterministic DDIM run backwards, from data x at t = 0 to t_end."""
    a, s = path.grid(steps, 0.0, t_end)
    j = np.maximum(np.arange(steps), 1)  # at t = 0 x is read at the next level
    def step(x, inputs):
        a_t, s_t, a_s, s_s, a_j, s_j = inputs
        eps = pred(x, a_j, s_j)[1]
        return a_s * (x - s_t * eps) / a_t + s_s * eps, None
    return jax.lax.scan(step, x, (a[:-1], s[:-1], a[1:], s[1:], a[j], s[j]))[0]
```

The decoder steps back down with
$\hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_{t'}, \bar\sigma_{t'})$, read at
the upper end. The two estimates differ by the change of
$\hat{\boldsymbol\epsilon}$ over a step, so the round trip is inexact even
with an exact field, and its error should shrink with the step size like any
first-order error. The round trip runs one field in both directions, so its
error is the error of the steps on that field, whether or not the field is
right. What the learned field can still do is change faster along its
trajectories than an exact field would; its round trip would then converge
more slowly than one through an exact field. We encode and decode 100 test
images, each with its own label, with 10 to 200 steps on the linear path,
and send the 20,000 mixture points through the same round trip with the
exact field.

```{.python .input #samplers-the-latent-code-1}
%%tab pytorch
x0 = X_test[:100]
pred = d2l.make_pred(d2l.conditional(ema, y_test[:100]), lin, 'velocity')
codes = {n: d2l.ddim_invert(pred, x0, lin, n) for n in (10, 20, 50, 100, 200)}
recon = {n: d2l.ddim_sample(pred, code, lin, n) for n, code in codes.items()}
for n in codes:  # the mixture's points through its exact field, for contrast
    back = d2l.ddim_sample(pred_exact, d2l.ddim_invert(pred_exact, data, lin, n),
                           lin, n)
    print(f'{n:3d} steps: squared error per pixel '
          f'{((recon[n] - x0) ** 2).mean():.4f}, code std {codes[n].std():.2f}; '
          f'mixture {((back - data) ** 2).mean():.4f}')
d2l.show_gray(torch.cat([x0[:8], recon[10][:8], recon[50][:8]]), 3, 8);
```

```{.python .input #samplers-the-latent-code-1}
%%tab jax
x0 = X_test[:100]
pred = d2l.make_pred(d2l.conditional(ema, y_test[:100]), lin, 'velocity')
codes = {n: d2l.ddim_invert(pred, x0, lin, n) for n in (10, 20, 50, 100, 200)}
recon = {n: d2l.ddim_sample(pred, code, lin, n) for n, code in codes.items()}
for n in codes:  # the mixture's points through its exact field, for contrast
    back = d2l.ddim_sample(pred_exact, d2l.ddim_invert(pred_exact, data, lin, n),
                           lin, n)
    print(f'{n:3d} steps: squared error per pixel '
          f'{((recon[n] - x0) ** 2).mean():.4f}, code std {codes[n].std():.2f}; '
          f'mixture {((back - data) ** 2).mean():.4f}')
d2l.show_gray(jnp.concatenate([x0[:8], recon[10][:8], recon[50][:8]]), 3, 8);
```

Both round trips are inexact, and both errors fall steadily: from ten to a
hundred steps the root-mean-square error shrinks about sixfold in both, well
short of the tenfold of a first-order error, and from fifty steps on the
images' error falls at least as fast as the mixture's. The learned field is
therefore no rougher along its trajectories than an exact one, and what
remains is the error of the steps. At fifty steps it is 0.05 to 0.06 per
pixel on a scale from $-1$ to $1$, and the third row is close to the
originals; at ten steps it is about 0.19, and the second row loses detail
such as the check pattern of the last shirt.

The codes are less satisfactory. Their standard deviation is about 0.7 at
ten steps and still below 0.9 at two hundred, so a code that reproduces its
image is narrower than the noise the sampler was built to start from. Codes
narrower than noise are the weak point of DDIM inversion in practice, and
methods that edit real photographs through their codes add machinery to
work around them, for instance by optimizing the null-label embedding so
that guided sampling from the code still reproduces the image.

Because the sampler is a continuous function of its code, we can move
between two images by interpolating between their codes. A straight line is
a poor path. Halfway between two codes of equal norm $r$ at an angle
$\omega$ the norm is $r \cos(\omega/2)$, smaller by a factor of about
$\sqrt{2}$ when the codes are nearly orthogonal, as independent noise images
are, and the decoder has never been started from so short a code. Spherical
interpolation moves along the great circle between the two codes instead,

$$\mathbf{z}_u = \frac{\sin\big((1 - u)\omega\big)\, \mathbf{z}_0 + \sin(u \omega)\, \mathbf{z}_1}{\sin \omega}, \qquad \cos \omega = \frac{\mathbf{z}_0^\top \mathbf{z}_1}{\|\mathbf{z}_0\| \|\mathbf{z}_1\|},$$

which keeps the norm of the endpoints. We decode eight points between the
50-step codes of two ankle boots among the test images; one class for the
whole row lets the decoder use one label.

```{.python .input #samplers-the-latent-code-2}
%%tab pytorch
i, j = torch.nonzero(y_test[:100] == y_test[0])[:2, 0]  # one class: one field
z0, z1 = codes[50][i], codes[50][j]
omega = torch.arccos(((z0 * z1).sum() / (z0.norm() * z1.norm())).clamp(-1, 1))
u = torch.linspace(0, 1, 8, device=z0.device).reshape(-1, 1, 1, 1)
zs = (torch.sin((1 - u) * omega) * z0 + torch.sin(u * omega) * z1
      ) / torch.sin(omega)
pred_i = d2l.make_pred(d2l.conditional(ema, y_test[i].repeat(8)), lin,
                       'velocity')
d2l.show_gray(d2l.ddim_sample(pred_i, zs, lin, 50), 1, 8);
```

```{.python .input #samplers-the-latent-code-2}
%%tab jax
i, j = jnp.nonzero(y_test[:100] == y_test[0])[0][:2]  # one class: one field
z0, z1 = codes[50][i], codes[50][j]
omega = jnp.arccos(jnp.clip(
    (z0 * z1).sum() / (jnp.linalg.norm(z0) * jnp.linalg.norm(z1)), -1, 1))
u = jnp.linspace(0, 1, 8).reshape(-1, 1, 1, 1)
zs = (jnp.sin((1 - u) * omega) * z0 + jnp.sin(u * omega) * z1
      ) / jnp.sin(omega)
pred_i = d2l.make_pred(d2l.conditional(ema, jnp.full(8, y_test[i])), lin,
                       'velocity')
d2l.show_gray(d2l.ddim_sample(pred_i, zs, lin, 50), 1, 8);
```

The shape changes gradually from one boot to the other, but the middle
frames look more like an overlay of the two boots than like a third one.
Spherical interpolation keeps the codes at the right norm; nothing makes the
decoder map the codes between two boots to a third boot.

### Editing by Partial Noising

Inversion finds the noise that reproduces an image exactly. For editing we
want the opposite: a starting point that forgets the details of the input
and keeps its layout. :citet:`Meng.He.Song.ea.2022` therefore noise a rough
input $\mathbf{y}$, such as a sketch or a crude collage, directly with fresh
noise, to an intermediate time $t_0$ with $\mathbf{x}_{t_0} = \alpha_{t_0}
\mathbf{y} + \sigma_{t_0} \boldsymbol\epsilon$, and run the sampler from
$t_0$ back to the data. The noise hides the details of the input that real
images do not have, the sampler replaces them by details that real images
do have, and the coarse layout survives. The time $t_0$ sets the balance
between fidelity to the input and realism. How good can that balance be?

Our rough inputs are the ten class-mean images, blurred silhouettes that no
real garment looks like, each noised a hundred times to $t_0 \in \{0.3,
0.5, 0.7, 1\}$ and integrated back with 25 DDIM steps under its class label.
We print the squared distance per pixel to the input (fidelity) and the FD
of the 1,000 outputs (realism); the row $t_0 = 0$ holds the inputs.

```{.python .input #samplers-editing-by-partial-noising}
%%tab pytorch
means = torch.stack([X[y == c].mean(0) for c in range(10)])
inputs = means.repeat(100, 1, 1, 1)                  # 1,000 inputs, class i % 10
pred = d2l.make_pred(d2l.conditional(ema, labels), lin, 'velocity')
torch.manual_seed(2)
eps = torch.randn_like(inputs)
edits = {0.0: inputs}
for t0 in (0.3, 0.5, 0.7, 1.0):
    x_t0 = lin.sample(inputs, torch.full((1000,), t0, device=X.device), eps)
    edits[t0] = d2l.ddim_sample(pred, x_t0, lin, 25, t_start=t0).clamp(-1, 1)
for t0, x in edits.items():
    print(f't0 = {t0:.1f}: squared distance to the input '
          f'{((x - inputs) ** 2).mean():.3f}, '
          f'FD {d2l.fid_score(features(x), f_test):.2f}')
d2l.show_gray(torch.cat([x[:10] for x in edits.values()]), 5, 10);
```

```{.python .input #samplers-editing-by-partial-noising}
%%tab jax
means = jnp.stack([X[y == c].mean(0) for c in range(10)])
inputs = jnp.tile(means, (100, 1, 1, 1))             # 1,000 inputs, class i % 10
pred = d2l.make_pred(d2l.conditional(ema, labels), lin, 'velocity')
eps = jax.random.normal(jax.random.key(2), inputs.shape)
edits = {0.0: inputs}
for t0 in (0.3, 0.5, 0.7, 1.0):
    x_t0 = lin.sample(inputs, jnp.full((1000,), t0), eps)
    edits[t0] = d2l.ddim_sample(pred, x_t0, lin, 25, t_start=t0).clip(-1, 1)
for t0, x in edits.items():
    print(f't0 = {t0:.1f}: squared distance to the input '
          f'{((x - inputs) ** 2).mean():.3f}, '
          f'FD {d2l.fid_score(features(x), f_test):.2f}')
d2l.show_gray(jnp.concatenate([x[:10] for x in edits.values()]), 5, 10);
```

The inputs score an FD of 85 to 87; as $t_0$ grows, the distance to the
input grows and the FD falls. At $t_0 = 0.3$ the outputs are sharper than
the mean images but blotchy, and their FD of about 30 is that of the
ten-step ancestral chain, far from any deterministic sampler's. At $0.5$
they look like garments with odd textures (11 to 12). Only at $0.7$ does
the FD, about 4, come near that of the ten-step deterministic samplers, and
by then the outline and pose are all that survive of the input. At $t_0 = 1$
the input is gone: each output is a fresh sample of its class, with an FD
below 2, and since the two rows start from the same noise images, the rows
at $0.7$ and $1$ often look alike. For an input as crude as a class mean,
realism arrives only at a $t_0$ that leaves little of the input.

### Inpainting

Suppose that we know part of an image, the upper half, say, and want the
model to fill in the rest. Write $\mathbf{m}$ for the mask that is one on
the known pixels and $\mathbf{y}$ for their values. At every level we know
what the known pixels of $\mathbf{x}_s$ should look like: $\alpha_s
\mathbf{y}$ plus noise of scale $\sigma_s$. Replacement inpainting
:cite:`song2021score,Lugmayr.Danelljan.Romero.ea.2022` therefore takes each
sampler step as usual and then overwrites the known pixels with a fresh
noisy copy of $\mathbf{y}$ at the new level,

$$\mathbf{x}_s \leftarrow \mathbf{m} \odot \big(\alpha_s \mathbf{y} + \sigma_s \boldsymbol\xi\big) + (\mathbf{1} - \mathbf{m}) \odot \mathbf{x}_s.$$

Replacement is an approximation. Each step moves all pixels as if the known
ones were free, then overwrites the known ones with noise that knows nothing
of where the step took them. The unknown pixels learn about the known ones
only through the network's next look at the noisy image, so the sampler
does not draw them exactly from their distribution given the known pixels.
`inpaint` runs the sampler one step at a time to make room for the
overwrite.

```{.python .input #samplers-inpainting-and-posterior-sampling-1}
%%tab pytorch
@torch.no_grad()
def inpaint(pred, known, mask, path, steps):
    """DDIM from noise that resets the known pixels (mask 1) at each level."""
    a, s = path.grid(steps)
    x = torch.randn_like(known)
    for i in range(steps):
        x0, eps = pred(x, a[i], s[i])
        x = a[i + 1] * x0 + s[i + 1] * eps            # one DDIM step, eta = 0
        noised = a[i + 1] * known + s[i + 1] * torch.randn_like(x)
        x = mask * noised + (1 - mask) * x            # keep the known pixels
    return x
```

```{.python .input #samplers-inpainting-and-posterior-sampling-1}
%%tab jax
def inpaint(pred, known, mask, path, steps, key):
    """DDIM from noise that resets the known pixels (mask 1) at each level."""
    a, s = path.grid(steps)
    def step(x, inputs):
        a_t, s_t, a_s, s_s, key = inputs
        x0, eps = pred(x, a_t, s_t)
        x = a_s * x0 + s_s * eps                     # one DDIM step, eta = 0
        noised = a_s * known + s_s * jax.random.normal(key, x.shape)
        return mask * noised + (1 - mask) * x, None   # keep the known pixels
    keys = jax.random.split(key, steps + 1)
    x = jax.random.normal(keys[0], known.shape)
    return jax.lax.scan(step, x, (a[:-1], s[:-1], a[1:], s[1:], keys[1:]))[0]
```

We keep the upper half of eight test images and ask for three completions of
each, from different noise, under each image's label. The rows show the
known halves, the three completions and the originals.

```{.python .input #samplers-inpainting-and-posterior-sampling-2}
%%tab pytorch
mask = torch.zeros(1, 1, 28, 28, device=X.device)
mask[..., :14, :] = 1                                # the upper half is known
pred = d2l.make_pred(d2l.conditional(ema, y_test[:8].repeat(3)), lin,
                     'velocity')
torch.manual_seed(3)
fills = inpaint(pred, X_test[:8].repeat(3, 1, 1, 1), mask, lin, 50)
d2l.show_gray(torch.cat([X_test[:8] * mask, fills, X_test[:8]]), 5, 8);
```

```{.python .input #samplers-inpainting-and-posterior-sampling-2}
%%tab jax
mask = jnp.zeros((1, 28, 28, 1)).at[:, :14].set(1)  # the upper half is known
pred = d2l.make_pred(d2l.conditional(ema, jnp.tile(y_test[:8], 3)), lin,
                     'velocity')
fills = inpaint(pred, jnp.tile(X_test[:8], (3, 1, 1, 1)), mask, lin, 50,
                jax.random.key(3))
d2l.show_gray(jnp.concatenate([X_test[:8] * mask, fills, X_test[:8]]), 5, 8);
```

Most completions continue the known half into a plausible garment of its
class, and the three completions of one image differ. The approximation
shows at the seam: in several completions, the shirts most clearly, the
lower half takes a different texture or shade from the upper half (Exercise
:exref:`ex_samplers-repaint` measures and repairs this).

Every sampler in this section integrated the same trained field and differed
only in how. The field itself we have taken on trust: it came from an
unweighted squared error on a target we picked. The next section asks why
this loss is the right one, and what the choices of target, path and
weighting change.

## Summary

Training fixes only the distribution of the noisy point at each time, so one
network serves a family of samplers. Its deterministic member, DDIM, is
Euler's method for the probability-flow ODE in the coordinates
$(\mathbf{x}/\alpha, \sigma/\alpha)$, and its stochastic members add fresh
noise; for data narrower than the noise the best variance lies between the
posterior's and the forward step's. On the Fashion-MNIST model fresh noise
cost quality at every budget up to two hundred steps and came within about
0.2 of the deterministic sampler only at a thousand. Heun's method on the
model's own training grid reached the plateau of the thousand-step samplers
at about twenty evaluations, and Euler's method at fifty to a hundred. The
grid mattered as much as the method, under guidance only the multistep
sampler reached the plateau by fifty evaluations, and FD did not rank the
solvers by their integration error. To choose a sampler for a new model,
double the evaluations until the score stops moving. Run backwards, the
deterministic sampler gives every image a code, with a first-order
round-trip error; a sampler started from a partly noised image edits it,
and one that overwrites the known pixels at every step inpaints.

## Exercises

1. **Admissible noise scales.** The family :eqref:`eq_diffusion-ddim-family`
   needs $\tilde\sigma \leq \sigma_s$.
    1. Show that a strength $\eta$ is admissible on a step from $t$ to $s$
       exactly when $\eta^2 \leq 1 / (1 - \kappa^2)$, so that $\eta = 1$ is
       admissible on every grid. At the largest admissible $\eta$, what does
       the step keep of $\mathbf{x}_t$ beyond $\hat{\mathbf{x}}_0$?
    1. Evaluate the fraction $f$ of fresh noise at $\eta = 1$ on every step
       of the variance-preserving grids with 10 and with 200 steps. What does
       the $\eta = 1$ sampler carry from one level to the next on the
       ten-step grid?
   :exlabel:`ex_samplers-admissible-noise`
1. **The forward step and the Markov member.** ● Given $\mathbf{x}_0$, draw
   $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t \boldsymbol\epsilon$ and
   $\mathbf{x}_s$ from :eqref:`eq_diffusion-ddim-family` with the same
   $\boldsymbol\epsilon$.
    1. Compute the mean and the covariance of $\mathbf{x}_t$ given
       $\mathbf{x}_s$ and $\mathbf{x}_0$ by conditioning the joint Gaussian.
    1. Show that the mean depends on $\mathbf{x}_0$ unless $\tilde\sigma$
       takes its value at $\eta = 1$, and that the conditional is then the
       forward step $\mathbf{x}_t = (\alpha_t / \alpha_s) \mathbf{x}_s +
       \sqrt{\sigma_t^2 - (\alpha_t / \alpha_s)^2 \sigma_s^2}\,
       \boldsymbol\nu$ of the noising chain of
       :numref:`sec_diffusion-gaussian-paths`, and conclude that this chain
       is the only member of the family whose forward process is Markov.
    1. Show that with $\tilde\sigma = 0$ at every step the forward process
       is $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
       \boldsymbol\epsilon$ with one $\boldsymbol\epsilon$ shared by all
       times. Why can the regression not tell these forward processes apart?
   :exlabel:`ex_samplers-forward-step`
1. **Gaussian data.** Let the data be one-dimensional with distribution
   $\mathcal{N}(0, v)$.
    1. In the coordinates $(\bar x, \bar\sigma)$, show that the exact noise
       estimate is $\hat\epsilon = \bar\sigma \bar x / (v + \bar\sigma^2)$,
       that one deterministic stride from $\bar\sigma$ to $\bar\sigma'$
       multiplies $\bar x$ by $(v + \bar\sigma \bar\sigma') / (v +
       \bar\sigma^2)$, and that the exact flow multiplies it by $\sqrt{(v +
       \bar\sigma'^2) / (v + \bar\sigma^2)}$.
    1. Show that the factor of the stride never exceeds the exact one, with
       equality only for $\bar\sigma' = \bar\sigma$. What does this imply for
       the variance of deterministic samples of Gaussian data on any grid?
    1. Now take any member of the family :eqref:`eq_diffusion-ddim-step`
       with $0 \leq \eta \leq 1$ and exact estimates. Write the variance of
       $x_s$ after one step from $x_t \sim \mathcal{N}(0, \alpha_t^2 v +
       \sigma_t^2)$ as a function of $c = \sqrt{\sigma_s^2 -
       \tilde\sigma^2}$, and show with the Cauchy--Schwarz inequality that it
       stays below the exact $\alpha_s^2 v + \sigma_s^2$ and falls as $\eta$
       grows. Which of the spreads printed in this section does this explain?
    1. Take $v = 0.25$, the variance of one coordinate of a mode of `mix`.
       Compute the standard deviation $s$ of deterministic samples from $x_1
       \sim \mathcal{N}(0, 1)$ on the variance-preserving grids with 200,
       50, 20, 10 and 5 steps. Compare $2s^2$, the spread of a
       two-dimensional cloud with this standard deviation per coordinate,
       with the data's $0.5$ and with the spreads printed in this section.
       Does a Gaussian mode predict the mixture's contraction, more of it,
       or less?
   :exlabel:`ex_samplers-gaussian-data`
1. [code] **The strength $\eta$ on both examples.**
    1. On the mixture with `pred_exact`, run `ddim_sample` with $\eta \in
       \{0, 0.5, 1\}$ and `ancestral_sample` on the variance-preserving grids
       with 200, 50, 20 and 10 steps from 20,000 starting points. Compare all
       three `mix.summary` statistics with those of the data. Where does each
       sampler break first: in the mode weights, in the spread or in the
       stray fraction?
    1. Relate the excess spread of the ancestral chain to the variance error
       of one ancestral step, which Exercise
       :exref:`ex_gaussian-paths-ancestral-first-order` of
       :numref:`sec_diffusion-integration` computes, and to the factor
       $\sigma_s^2 / \sigma_t^2$ between the two samplers' variances.
    1. On the image model, compute the FD of $\eta \in \{0.25, 0.5\}$ at 10,
       20 and 50 steps with `fd_runs`, and set it beside the two ends in the
       first table of this section. Is any intermediate strength better than
       both ends at some budget? Which $\eta$ would you use at each budget,
       and which if the sample must be a function of its code?
   :exlabel:`ex_samplers-eta-strength`
1. [code] **Inversion on both examples.** The reconstruction cell sends
   images and mixture points through `ddim_invert` and `ddim_sample` on the
   linear path.
    1. Repeat the mixture round trip on the variance-preserving grid with 10
       to 200 steps. Does the error fall in proportion to the step size, and
       does the path change the answer?
    1. Compute the mean and the covariance of the 200-step codes of the
       mixture points. Are they those of standard Gaussian noise?
    1. A variant of `ddim_invert` reads the noise estimate of every step at
       the upper level, $\hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_t,
       \bar\sigma_{t'})$, as its first step must. Implement it and compare the
       round-trip errors of the two variants on the mixture and on the images
       at 10 to 200 steps. Which is better at ten steps, and which converges
       faster as the steps shrink?
   :exlabel:`ex_samplers-inversion-steps`
1. [extended] **A second-order sampler in log-SNR.** ●● DPM-Solver-2
   :cite:`Lu.Zhou.Bao.ea.2022` takes a step from $\bar\sigma$ to
   $\bar\sigma'$ in two evaluations, the second at the midpoint
   $\bar\sigma_m = \sqrt{\bar\sigma \bar\sigma'}$ in $\log \bar\sigma$:
   $\bar{\mathbf{x}}_m = \bar{\mathbf{x}} + (\bar\sigma_m - \bar\sigma)\,
   \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}, \bar\sigma)$ and
   $\bar{\mathbf{x}}' = \bar{\mathbf{x}} + (\bar\sigma' - \bar\sigma)\,
   \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_m, \bar\sigma_m)$.
    1. Let $h = \log(\bar\sigma / \bar\sigma')$. Write the exact change of
       $\bar{\mathbf{x}}$ over the step as an integral of
       $\hat{\boldsymbol\epsilon}$ along the trajectory, and show that the
       local error of the update is $O(\bar\sigma h^3)$ while that of the
       Euler step is $O(\bar\sigma h^2)$. Assume that
       $\hat{\boldsymbol\epsilon}$ is a smooth function of $\log \bar\sigma$
       along the trajectory and Lipschitz in $\mathbf{x} =
       \bar{\mathbf{x}} / \sqrt{1 + \bar\sigma^2}$, so with a constant of
       order $1/\bar\sigma$ in $\bar{\mathbf{x}}$ at large $\bar\sigma$.
    1. Implement the update with the interface `pred(x, alpha, sigma)`,
       using $\alpha = 1 / \sqrt{1 + \bar\sigma^2}$ and $\sigma = \bar\sigma
       \alpha$, and take the last step to $\bar\sigma = 0$ with Euler's
       method. Compare it with `ode_sample`, `heun_sample` and
       `multistep_sample` at about 11, 21 and 51 evaluations, on the mixture
       with `pred_exact` by the distance to a 1,000-step reference, and on
       images by the FD.
    1. Repeat with levels evenly spaced in $\log \bar\sigma$ between 80 and
       0.002. Which combination of method and grid is best at each budget?
   :exlabel:`ex_samplers-dpm-solver`
1. **The optimal reverse variance.** ● A stochastic step from $t$ to $s$ moves
   to the posterior mean $\alpha_s \hat{\mathbf{x}}_0 + \sigma_s \kappa
   \hat{\boldsymbol\epsilon}$ and adds Gaussian noise of variance $v$ per
   coordinate. :citet:`ho2020denoising` tried the forward-step variance and
   the posterior variance; :citet:`Bao.Li.Zhu.ea.2022` found the best $v$
   for any data. Assume exact estimates, and let $\bar e_t =
   \mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}\|^2 / d$ be
   the error per coordinate of the exact noise estimate at time $t$ in $d$
   dimensions.
    1. Exercise :exref:`ex_gaussian-paths-gaussian-reversal`a of
       :numref:`sec_diffusion-integration` shows that the true reverse
       transition $q(\mathbf{x}_s \mid \mathbf{x}_t)$ has covariance $\tau^2
       \mathbf{I} + c^2 \textrm{Cov}(\mathbf{x}_0 \mid \mathbf{x}_t)$, where
       $c$ is the coefficient of $\mathbf{x}_0$ in the mean of
       $q(\mathbf{x}_s \mid \mathbf{x}_t, \mathbf{x}_0)$. Write that mean as
       $\alpha_s \mathbf{x}_0 + \sigma_s \kappa \boldsymbol\epsilon$ with
       $\boldsymbol\epsilon = (\mathbf{x}_t - \alpha_t
       \mathbf{x}_0)/\sigma_t$, and conclude that $\tau^2 = \tilde\sigma^2$,
       the posterior variance, and $c = k = \alpha_s - \sigma_s \kappa\,
       \alpha_t / \sigma_t$. What is the mean of the true reverse
       transition?
    1. Show that the $v$ that minimizes the expected KL divergence from
       $q(\mathbf{x}_s \mid \mathbf{x}_t)$ to the Gaussian step is
       $v^\star = \tilde\sigma^2 + k^2 \bar\sigma_t^2 \bar e_t$, and that
       $\bar e_t = 1 - \sigma_t^2\, \mathbb{E}\|\nabla \log
       p_t(\mathbf{x}_t)\|^2 / d$, so that $v^\star$ follows from the score
       alone.
    1. Show that $v^\star$ is the posterior variance when the data are one
       point, and the forward-step variance when the data are standard
       Gaussian on the variance-preserving path.
    1. (Optional, code.) Estimate $\bar e_t$ on the mixture with
       `pred_exact` on the 50-step variance-preserving grid, and compare the
       spread of 20,000 samples of the stochastic step with the three
       variances.
   :exlabel:`ex_samplers-optimal-variance`
1. [code] **Inpainting with harmonization.** ● The completions of `inpaint`
   can disagree with the known half along the boundary.
    1. Measure the seam: the mean absolute difference between rows 13 and 14
       of the completions, against the same quantity for real test images.
    1. RePaint :cite:`Lugmayr.Danelljan.Romero.ea.2022` lets the two halves
       exchange information: after each step from $t$ to $s$, it noises the
       result back to $t$ with the forward step and repeats the step, $r$
       times per level. Implement it with $r \in \{1, 5, 10\}$ at 50 levels
       and measure the seam again. What does each repetition cost?
    1. Draw five completions of one test image for each $r$. How varied are
       they?
   :exlabel:`ex_samplers-repaint`
1. [code] **A rescaled schedule.** ● Let $c_t = 1 + t(1 - t)$ and define a
   path with $\alpha'_t = c_t (1 - t)$ and $\sigma'_t = c_t t$, which has the
   same endpoints as the linear path.
    1. Show that the two paths have the same log-SNR at every $t$, and that
       the deterministic DDIM steps of :eqref:`eq_diffusion-ddim-step` on the
       two paths produce the same $\hat{\mathbf{x}}_0$ at every step and
       states that differ by the factor $c_t$. Check this with `ddim_sample`
       and `pred` on the image model.
    1. Implement Euler's method on the velocity, $\mathbf{x} \leftarrow
       \mathbf{x} + (s - t)\, (\dot\alpha_t \hat{\mathbf{x}}_0 + \dot\sigma_t
       \hat{\boldsymbol\epsilon})$, with the rates from `path.rates`, on both
       paths with 10 steps. Show that the samples differ, and use Exercise
       :exref:`ex_gaussian-paths-same-ratio-same-flow` of
       :numref:`sec_diffusion-integration` to explain why.
    1. :citet:`Gao.Hoogeboom.Heek.ea.2025` conclude that DDIM is unaffected
       by such a rescaling. Which of the two choices, the path or the grid of
       times, can change the samples of DDIM?
   :exlabel:`ex_samplers-rescaled-schedule`
1. [code] **Editing toward another class.** Use the class-conditional model
   of this section and steer with
   `d2l.make_pred(d2l.cfg(ema, c, gamma), lin, 'velocity')`, where `c` holds
   one label per image.
    1. Noise 100 test images of trousers to $t_0 \in \{0.3, 0.5, 0.7, 0.9\}$
       and integrate back with 25 DDIM steps toward the label of dresses, at
       $\gamma = 1$ and $\gamma = 2$. Report the fraction of outputs that
       `cnn` classifies as dresses and the squared distance per pixel to the
       input, as in the editing cell.
    1. From pure noise, sample with the label of trousers and $\gamma = 1$
       from $t = 1$ down to $t^*$, then with the label of dresses and the
       two scales of (a) down to $t = 0$ (`ddim_sample` takes `t_start` and
       `t_end`), for $t^* \in \{0.3, 0.5, 0.7, 0.9\}$. Plot the fraction
       classified as dresses against $t^*$, next to the curve of (a) against
       $t_0$, and compare both with the commitment time of Exercise
       :exref:`ex_gaussian-paths-commitment-time` of
       :numref:`sec_diffusion-integration` on the mixture.
    1. Show that with an exact network the two experiments at the same
       $\gamma$ and $t_0 = t^*$ produce outputs with the same distribution.
       Which differences between the curves remain, and where do they come
       from?

    *Part (a) is adapted from Berkeley CS180/280A,
    [Project 5A](https://cal-cs180.github.io/fa25/hw/proj5/parta.html)
    (Fall 2025), Section 1.7.3.*
   :exlabel:`ex_samplers-class-edit`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17.5]{.kicker}

Samplers, Inversion and Editing<br>
**one network, a family of samplers · DDIM is Euler's method · order and step grid · a procedure for choosing · codes, edits and inpainting**
:::
:::

::: {.slide title="The Thousand-Step Chain Costs Twenty Times What Fifty Steps Cost"}
[The problem]{.kicker}

- Every sampler step evaluates the U-Net once; the ancestral chain takes
  1,000 steps per image.
- On Fashion-MNIST, fifty deterministic steps score a Fréchet distance (FD,
  lower is better) within about a third of the chain's.

::: {.d2l-note .warn}
Which of the thousand steps do we need, and how few can we take? And can we
run sampling backwards, from an image to its noise?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive a family of samplers that one trained network serves;
- show that its deterministic member, DDIM, is Euler's method;
- compute the error of an Euler and a Heun step by hand;
- choose a solver, a grid and a number of evaluations for a new model;
- invert an image to its code, edit it, and inpaint it.

**Running examples:** the three-mode mixture with its exact predictor;
a 1.2M-parameter class-conditional U-Net on Fashion-MNIST.
:::

::: {.slide title="This Deck Turns a Trained Network into Fast Samplers"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Denoising and the Score | what a denoiser knows about the data |
| Gaussian Paths | which regression trains the walk to data |
| Sampling by Integration | how to follow the field |
| Diffusion for Images | what changes for images |
| **Samplers, Inversion and Editing** | **which steps to skip; running backwards** |
| Likelihood, Weighting and the Design Space | is the regression maximum likelihood? |
| Latent Diffusion and Transformers | why diffuse a compressed code |
| Few-Step Generation | why one step fails |
| Discrete Diffusion | what noise is for a token |
:::

::: {.col .narrow}
- **Uses:** the path $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
  \boldsymbol\epsilon$, data at $t = 0$, noise at $t = 1$; the
  probability-flow ODE; FD.
- **Skips:** one- and two-step models; solvers beyond second order.
:::
:::
:::

::: {.slide title="Training Fixes Only the Marginals, so Many Steps Are Valid"}
The network gives $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ at any
noisy point. It was trained on single points $\mathbf{x}_t$, never chains, so
a step from $t$ to $s < t$ only has to produce the law of $\mathbf{x}_s$.
For any $0 \le \tilde\sigma \le \sigma_s$:

$$\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sqrt{\sigma_s^2 - \tilde\sigma^2}\, \hat{\boldsymbol\epsilon} + \tilde\sigma\, \boldsymbol\xi, \qquad \boldsymbol\xi \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

::: {.d2l-note}
The two noise variances add to $\sigma_s^2$: with the true
$\mathbf{x}_0, \boldsymbol\epsilon$ in place of the estimates, every
$\tilde\sigma$ gives $\mathbf{x}_s$ its correct distribution.
:::
:::

::: {.slide title="The Strength η Sets How Much of the Noise Is Fresh"}
$$\tilde\sigma = \eta\, \sigma_s \sqrt{1 - \kappa^2}, \qquad \kappa^2 = \frac{\mathrm{SNR}_t}{\mathrm{SNR}_s} < 1, \qquad \mathrm{SNR} = \frac{\alpha^2}{\sigma^2}$$

- **$\eta = 0$, DDIM:** no fresh noise; the sample is a function of its start.
- **$\eta = 1$:** the posterior variance of the noising chain.
- **Ancestral chain:** same mean, the forward-step variance, larger by
  $\sigma_t^2 / \sigma_s^2$.

::: {.d2l-note}
Posterior variance: exact for one data point. Forward-step variance: exact
for standard Gaussian data. Data narrower than the noise need a variance in
between. The ancestral chain is not a member: its variances sum to more
than $\sigma_s^2$.
:::

[Song, Meng and Ermon, 2020; Bao et al., 2022]{.small}
:::

::: {.slide title="DDIM Is Euler's Method in the Coordinates x/α and σ/α"}
Let $\bar{\mathbf{x}} = \mathbf{x}/\alpha_t$ and $\bar\sigma_t =
\sigma_t/\alpha_t$; every Gaussian path reads $\bar{\mathbf{x}}_t =
\mathbf{x}_0 + \bar\sigma_t \boldsymbol\epsilon$.

$$\bar{\mathbf{x}}_s = \hat{\mathbf{x}}_0 + \bar\sigma_s \hat{\boldsymbol\epsilon} = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\, \hat{\boldsymbol\epsilon}$$

One Euler step of $d\bar{\mathbf{x}} / d\bar\sigma =
\hat{\boldsymbol\epsilon}$: with the exact $\hat{\boldsymbol\epsilon}$, the
probability-flow ODE with the noise level as clock.

::: {.d2l-note}
The step holds $\hat{\boldsymbol\epsilon}$ fixed across the stride; its
error is the change of $\hat{\boldsymbol\epsilon}$ along the trajectory.
:::
:::

::: {.slide title="Check: How Much Noise Does η = 1 Replace on Ten Steps?"}
[Check]{.kicker}

On the ten-step VP grid the first step runs from $\bar\sigma = 152$ to
$59$. With $\mathrm{SNR} = 1/\bar\sigma^2$, what fraction
$f = 1 - \mathrm{SNR}_t/\mathrm{SNR}_s$ of the noise is fresh? On a
1,000-step grid?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** $f = 1 - (59/152)^2 \approx 0.85$: most of the noise is
replaced. On a fine grid neighboring levels have nearly equal SNR, so $f$
is small even at $\eta = 1$.
:::
:::
:::

::: {.slide title="Fresh Noise Pulls toward the Data but Errs on Long Steps"}
::: {.cols .vc}
::: {.col .fig .big}
![](../img/mdl-diffusion-two-samplers.svg)
:::

::: {.col}
200 steps from three starts (circles) to $t = 0$ (crosses). (a) $\eta = 0$:
smooth curves. (b) $\eta = 1$: jagged paths (first half faded) that may
change mode.
:::
:::

- **For noise:** a stochastic step is a Langevin step plus a flow step; the
  Langevin part pulls stray points back.
- **Against:** over a long step the true reverse step is far from Gaussian.
:::

::: {.slide title="On the Mixture, the Stride Error Halves When the Steps Double"}
@!samplers-skipping-steps-on-the-running-example

Exact predictor, VP grid; distance 0.4, 0.2, 0.04 at 5, 10, 50 steps: first order.
:::

::: {.slide title="At Ten Steps, η = 1 Ends Too Tight and the Chain Too Wide"}
Spread: mean squared distance to the nearest mode; the data's is 0.47 to 0.48.

- Ten steps: $\eta = 0$ 0.38, $\eta = 1$ 0.32 (too tight), ancestral 0.52
  (too wide).
- Kept noise lines up with $\hat{\mathbf{x}}_0$; fresh noise does not, so
  $\eta = 1$ ends tighter.
:::

::: {.slide title="On Gaussian Data Euler Shrinks the Sample; Heun Nearly Fixes It"}
Standard Gaussian data: $\hat{\boldsymbol\epsilon} = \bar\sigma
\bar{\mathbf{x}} / (1 + \bar\sigma^2)$. Step from $\bar\sigma_t = 1$ to
$\bar\sigma_s = 0.5$:

| | factor on $\bar{\mathbf{x}}$ | error |
|:--|:--|:--|
| exact flow $\sqrt{(1 + \bar\sigma_s^2)/(1 + \bar\sigma_t^2)}$ | 0.791 | |
| Euler, slope at the start | 0.75 | 0.041 |
| Heun, average of start and trial slopes | 0.80 | 0.009 |

Heun: trial $\bar{\mathbf{x}}' = \bar{\mathbf{x}}_t + \Delta\,
\hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_t, \bar\sigma_t)$, then
$\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + \Delta\,
\tfrac12\big(\hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_t, \bar\sigma_t) +
\hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}', \bar\sigma_s)\big)$, with
$\Delta = \bar\sigma_s - \bar\sigma_t$.

::: {.d2l-note}
Euler always removes too much noise from Gaussian data. Heun: about a
quarter of the error for two evaluations per step.
:::
:::

::: {.slide title="Grids Even in Time Look Nothing Alike in σ̄"}
Levels $\bar\sigma = \sigma/\alpha$ of three ten-step grids; EDM is even in
$\bar\sigma^{1/7}$ from 80 to 0.002:

@!samplers-second-order-and-the-step-grid

- VP and EDM: four steps above $\bar\sigma = 4$.
- Linear: two steps above 4 ($\infty$ to 9 to 4), then seven between 4 and
  0.11.

[Karras et al., 2022]{.small}
:::

::: {.slide title="A Multistep Solver Gets Second Order from One Evaluation per Step"}
Dividing by $\alpha$ removes the linear part of the ODE: DDIM is the
first-order *exponential integrator*. Reuse the last estimate
$\hat{\boldsymbol\epsilon}'$ (step $\Delta'$) to extrapolate over the
current step $\Delta = \bar\sigma_s - \bar\sigma_t$:

$$\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + \Delta \Big(\hat{\boldsymbol\epsilon}_t + \frac{\Delta}{2\Delta'}\big(\hat{\boldsymbol\epsilon}_t - \hat{\boldsymbol\epsilon}'\big)\Big)$$

::: {.d2l-note}
Second-order Adams--Bashforth. Constant $\hat{\boldsymbol\epsilon}$: DDIM.
Heun reaches second order with two evaluations per step.
:::

[Lu et al., 2022, 2025]{.small}
:::

::: {.slide title="Fresh Noise Never Beat η = 0 on This Model"}
VP grid, three paired draws of 1,000 noise images, mean ± half the range;
real against real about 0.9:

@!samplers-quality-against-evaluations-on-images-3

- $\eta = 0$ a third or more below $\eta = 1$ at every budget up to 200
  steps.
- At 1,000 steps all three within about 0.2; noise never overtakes.
- Ancestral at 50 steps (8 to 11) loses to DDIM at 10.
:::

::: {.slide title="Second-Order Solvers Reach the Plateau at About 20 Evaluations"}
@!samplers-quality-against-evaluations-on-images-4

- Heun on the linear grid: about 1.2 at 20 evaluations; Euler needs
  50 to 100.
- Plateau 1.1 to 1.3, where DDIM and the ancestral chain end at 1,000
  steps: the model's error.
:::

::: {.slide title="The Paper's Crowded Grid Loses to the Even Linear Grid"}
FD at equal network evaluations (Heun: one fewer on the EDM grids):

@!samplers-quality-against-evaluations-on-images-6

::: {.d2l-note}
$\rho = 3$ crowds the data end less and spends more steps at high noise; it
loses more at 10 and 20 evaluations, so the wasted steps are above
$\bar\sigma \approx 4$, where all noisy images look alike.
:::
:::

::: {.slide title="Under Guidance the Multistep Sampler Reaches the Plateau First"}
$\tilde{\mathbf{u}} = \mathbf{u}_\varnothing + \gamma(\mathbf{u}_c -
\mathbf{u}_\varnothing)$: $\gamma = 1$ is the conditional field; two network
passes per evaluation. FD and RMS distance per pixel to a 200-evaluation Heun
reference:

@!samplers-quality-against-evaluations-on-images-7

- FD does not rank solvers by integration error: Heun at 10 is farthest
  from the reference, yet scores half of Euler's FD.
- $\gamma = 2$, 50 evaluations: multistep at the reference; Heun and Euler
  still above.
:::

::: {.slide title="Deterministic Samplers Approximate One Map from Noise to Images"}
@!samplers-quality-against-evaluations-on-images-5

Rows: ancestral 50 steps; DDIM 10; DDIM 50; Heun on the linear grid, 20
evaluations. The last three match column by column; ten steps blur. The
ancestral row speckles.
:::

::: {.slide title="Check: Why Does the Ancestral Chain Speckle at Fifty Steps?"}
[Check]{.kicker}

At 50 steps the ancestral chain scores worse than DDIM at 10, while at
1,000 steps it ties DDIM. What changes between the two grids?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** It adds the forward-step variance, larger than the posterior
variance by $\sigma_t^2/\sigma_s^2$. On a fine grid that factor is near one
except in the last steps, where both variances are tiny. A garment is nearly
determined at low noise, so its right variance is close to the posterior's;
the excess shows as speckle.
:::
:::
:::

::: {.slide title="Choose by Doubling the Evaluations until the Score Stops Moving"}
1. Fix the noise images and a score; start deterministic, on the training
   grid.
2. Double the evaluations until the change is within the run-to-run range.
3. What remains is the model's error; no sampler removes it.

| this model | plateau reached at |
|:--|:--|
| Heun, $\gamma = 1$ | about 20 evaluations |
| Euler, $\gamma = 1$ | 50 to 100 |
| multistep, $\gamma = 2$ | about 50; Heun and Euler need more |
| fresh noise | never better |
:::

::: {.slide title="Running the Euler Steps Upward Gives Every Image a Code"}
Encoder: $\bar{\mathbf{x}}_{t'} = \bar{\mathbf{x}}_t + (\bar\sigma_{t'} -
\bar\sigma_t)\, \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}_t,
\bar\sigma_t)$; the decoder reads $\hat{\boldsymbol\epsilon}$ at the upper end.

@!samplers-the-latent-code-1

- Same rate on the exact field: the error is the steps'. Codes are narrower than noise.
:::

::: {.slide title="Spherical Interpolation Keeps Codes at the Norm of Their Endpoints"}
$$\mathbf{z}_u = \frac{\sin\big((1 - u)\omega\big)\, \mathbf{z}_0 + \sin(u \omega)\, \mathbf{z}_1}{\sin \omega}, \qquad \cos \omega = \frac{\mathbf{z}_0^\top \mathbf{z}_1}{\|\mathbf{z}_0\| \|\mathbf{z}_1\|}$$

@!samplers-the-latent-code-2

A straight line's midpoint has norm $r\cos(\omega/2)$: about $r/\sqrt2$ for
orthogonal codes. Two ankle boots; middle frames overlay the two.
:::

::: {.slide title="Partial Noising Trades Fidelity to the Input for Realism"}
::: {.cols .vc}
::: {.col}
@!samplers-editing-by-partial-noising
:::

::: {.col .narrow}
Noise a rough input $\mathbf{y}$ to $\mathbf{x}_{t_0} = \alpha_{t_0}
\mathbf{y} + \sigma_{t_0} \boldsymbol\epsilon$, then sample from $t_0$.
Inputs: the ten class-mean images.

- Rows: $t_0 = 0, 0.3, 0.5, 0.7, 1$.
- FD 85 to 87, about 30, 11 to 12, about 4, under 2: realism only once
  little input is left.

[Meng et al., 2022]{.small}
:::
:::
:::

::: {.slide title="Replacement Inpainting Approximates Conditioning on the Known Pixels"}
::: {.cols .vc}
::: {.col}
@!samplers-inpainting-and-posterior-sampling-2
:::

::: {.col}
Mask $\mathbf{m}$ (one on known pixels), values $\mathbf{y}$; after each
DDIM step:

$$\mathbf{x}_s \leftarrow \mathbf{m} \odot \big(\alpha_s \mathbf{y} + \sigma_s \boldsymbol\xi\big) + (\mathbf{1} - \mathbf{m}) \odot \mathbf{x}_s$$

- Rows: known half, three completions, originals.
- Halves meet only through the network: seams.
:::
:::
:::

::: {.slide title="Check: When Is the DDIM Round Trip Exact?"}
[Check]{.kicker}

The encoder reads $\hat{\boldsymbol\epsilon}$ at the lower end of each
step, the decoder at the upper end. With the exact field, for which data is
the round trip exact at any number of steps? What does its error measure
otherwise?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** When $\hat{\boldsymbol\epsilon}$ is constant along each
trajectory, as for a single data point: both ends then read the same
estimate. Otherwise the error is the change of $\hat{\boldsymbol\epsilon}$
over a step, first order in the step size. One field runs both ways, so the
error measures the steps, not whether the field is right.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **One network, a family:** $\eta = 0$ is DDIM, Euler's method in
  $(\mathbf{x}/\alpha, \sigma/\alpha)$; fresh noise never helped here.
- **Mixture:** stride error 0.4, 0.2, 0.04 at 5, 10, 50 steps.
- **Images:** floor about 0.9, plateau 1.1 to 1.3; Heun reaches it at about
  20 evaluations; under guidance the multistep sampler at about 50.
- **Procedure:** double the evaluations until the score stops moving.
- **Backwards:** codes give interpolation; partial runs give edits and
  inpainting.

**Next:** is the regression maximum likelihood, and which choices matter?
:::
