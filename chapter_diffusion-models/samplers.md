# Samplers, Inversion and Editing
:label:`sec_diffusion-samplers`

Every step of a sampler costs one evaluation of the U-Net, and the ancestral
chain of :numref:`sec_diffusion-gaussian-paths` takes a thousand steps per
image. The last section sampled with fifty Euler steps of the
probability-flow ODE, and on its conditional model the two samplers
scored within about 0.1 of each other in FD. Which of the thousand steps do
we need, and how few can we take? And once sampling is the integration of a
field, can we run it backwards, from a real image to the noise that produces
it, and edit the image from there?

:citet:`Song.Meng.Ermon.2020` answered the first question with an
observation about training. The regression of
:numref:`sec_diffusion-gaussian-paths` sees single noisy points
$\mathbf{x}_t$ of training examples, never a chain of them, so it determines
the distribution of $\mathbf{x}_t$ at each time and nothing more. Many
samplers produce those distributions, among them a deterministic one that
takes long strides and can be run in reverse, and one trained network serves
all of them. :citet:`Karras.Aittala.Aila.ea.2022` read the
deterministic sampler as a numerical method for a differential equation,
separating two choices that earlier samplers had made together: the order of
the method and the placement of its steps. :citet:`Meng.He.Song.ea.2022`
used the reverse direction to edit images.

We derive the family of samplers and its deterministic member, Euler's
method, measure quality against evaluations on the mixture and on
Fashion-MNIST, and end with inversion, editing and inpainting.

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

## One Field, Two Samplers

The ancestral chain and Euler's method of
:numref:`sec_diffusion-gaussian-paths`, one adding fresh noise at every step
and the other none, read the network through the same pair of estimates: at
a point $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
\boldsymbol\epsilon$ of a Gaussian path, the clean point
$\hat{\mathbf{x}}_0$ and the noise $\hat{\boldsymbol\epsilon}$, each of
which determines the other:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon}}{\alpha_t}.$$
:eqlabel:`eq_diffusion-x0-hat`

Whatever the network was trained to predict, `d2l.make_pred` of
:numref:`sec_diffusion-gaussian-paths` turns it into a function `pred(x,
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
chain of :numref:`sec_diffusion-gaussian-paths`, and the step samples that
chain's posterior with the estimates in place of the truth (exercise
:exref:`ex_samplers-forward-step`). At $\eta = 0$ no fresh noise enters and
the step is deterministic: the *denoising diffusion implicit model* (DDIM),
called implicit because it defines its samples by a map from noise rather
than by a chain of densities.

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
stride, while along the exact trajectory it changes. In the coordinates
$\bar{\mathbf{x}} = \mathbf{x} / \alpha_t$ and $\bar\sigma_t = \sigma_t /
\alpha_t$ of :numref:`sec_diffusion-gaussian-paths`, in which every Gaussian
path reads $\bar{\mathbf{x}}_t = \mathbf{x}_0 + \bar\sigma_t
\boldsymbol\epsilon$, the deterministic step is $\bar{\mathbf{x}}_s =
\hat{\mathbf{x}}_0 + \bar\sigma_s \hat{\boldsymbol\epsilon}$, and
substituting :eqref:`eq_diffusion-x0-hat` gives

$$\bar{\mathbf{x}}_s = \bar{\mathbf{x}}_t + (\bar\sigma_s - \bar\sigma_t)\, \hat{\boldsymbol\epsilon},$$

one Euler step of the ODE $d\bar{\mathbf{x}} / d\bar\sigma =
\hat{\boldsymbol\epsilon}$. For the exact noise estimate this is the
probability-flow ODE with the noise level $\bar\sigma$ as the clock, so on
every Gaussian path DDIM is Euler's method. :numref:`sec_mdl-ddim-guidance`
proves this, together with the two extreme cases: a single data point is
integrated without error by strides of any length, because
$\hat{\boldsymbol\epsilon}$ is then constant along each trajectory, and
Gaussian data shrink at every stride (:numref:`fig_mdl-dyn-ddim-strides`).
We check the equivalence with the exact predictor of the mixture `mix` of
:numref:`sec_diffusion-denoising`, running DDIM and `ode_sample` for ten
steps on the linear path from the same 20,000 starting points.

```{.python .input #samplers-the-ddim-family-and-euler-s-method-1}
%%tab pytorch
mix = d2l.GaussianMixture.three_modes()
vp, lin = d2l.GaussianPath.vp(), d2l.GaussianPath.linear()

def pred_exact(x, alpha, sigma):
    """The mixture's exact (x0, eps) at x = alpha x0 + sigma eps."""
    x0 = mix.posterior_mean(x, alpha, sigma)
    return x0, (x - alpha * x0) / sigma

torch.manual_seed(0)
z = torch.randn(20_000, 2)
gap = (d2l.ddim_sample(pred_exact, z, lin, 10)
       - d2l.ode_sample(pred_exact, z, lin.sigma_bars(10)))
print(f'DDIM against Euler: largest difference {gap.abs().max():.1e}')
```

```{.python .input #samplers-the-ddim-family-and-euler-s-method-1}
%%tab jax
mix = d2l.GaussianMixture.three_modes()
vp, lin = d2l.GaussianPath.vp(), d2l.GaussianPath.linear()

def pred_exact(x, alpha, sigma):
    """The mixture's exact (x0, eps) at x = alpha x0 + sigma eps."""
    x0 = mix.posterior_mean(x, alpha, sigma)
    return x0, (x - alpha * x0) / sigma

z = jax.random.normal(jax.random.key(0), (20_000, 2))
gap = (d2l.ddim_sample(pred_exact, z, lin, 10)
       - d2l.ode_sample(pred_exact, z, lin.sigma_bars(10)))
print(f'DDIM against Euler: largest difference {jnp.abs(gap).max():.1e}')
```

The two samplers agree to the rounding error of 32-bit arithmetic: they
compute the same function in two coordinate systems.

### Stochastic and Deterministic Steps

If one network serves every $\eta$, which strength should we use? With
$f = \tilde\sigma^2 / \sigma_s^2$, the step :eqref:`eq_diffusion-ddim-step`
reads

$$\mathbf{x}_s = \alpha_s \hat{\mathbf{x}}_0 + \sigma_s \Big(\sqrt{1 - f}\, \hat{\boldsymbol\epsilon} + \sqrt{f}\, \boldsymbol\xi\Big), \qquad f = \eta^2 \Big(1 - \frac{\mathrm{SNR}_t}{\mathrm{SNR}_s}\Big).$$

Every member places the clean estimate at level $s$ in the same way; a
fraction $f$ of the accompanying noise variance is fresh. On a fine grid $f$
is small even at $\eta = 1$; on a grid of ten steps the $\eta = 1$ sampler
replaces most of the noise at every step (exercise
:exref:`ex_samplers-admissible-noise`).

Fresh noise corrects errors. The probability-flow ODE is one member of a
family of reverse processes with the same marginals
(:numref:`sec_mdl-time-reversal`). A member that adds noise of variance $h$
in a reverse step must also move by $\tfrac{h}{2}\nabla \log p_t$, and the
added pair is the Langevin step of :numref:`sec_diffusion-denoising`, which
moves points toward the density it samples. A stochastic sampler therefore
keeps pulling its state toward the model's distribution at each level, and
an error made at one step is partly forgotten at the next. A deterministic
step has no such pull: an error in $\hat{\mathbf{x}}_0$ moves the state, and
the following steps carry it to the end.

The correction has a cost: a long stochastic step adds error of its own,
because the exact reverse transition over a long step is far from Gaussian
(exercise :exref:`ex_gaussian-paths-gaussian-reversal`). We expect the
deterministic sampler to win at small budgets and the stochastic one to
catch up as the steps shorten. :numref:`fig_diffusion-two-samplers` shows
both on the mixture.

![The two ends of the DDIM family on the three-mode mixture with the exact predictor, 200 steps on the variance-preserving path from six starting points at $t = 1$ (circles) to $t = 0$ (crosses); training data in grey. (a) At $\eta = 0$ each start follows a smooth curve into a mode. (b) At $\eta = 1$ the paths are jagged and may end in a different mode. Both produce the same distribution at $t = 0$.](../img/mdl-diffusion-two-samplers.svg)
:label:`fig_diffusion-two-samplers`

Two stochastic samplers must be kept apart. The ancestral chain of
:numref:`sec_diffusion-gaussian-paths` adds the variance of the forward
step, $\sigma_t^2 - (\alpha_t / \alpha_s)^2 \sigma_s^2$, while $\eta = 1$
adds the posterior variance, smaller by the factor $\sigma_s^2 /
\sigma_t^2$. On a thousand-step grid the factor is close to one and the two
nearly agree; on fifty steps the ancestral sampler adds more noise than the
posterior allows.

## Fewer Steps

We isolate the error of a stride on the mixture, then turn to second-order
solvers and step grids, and measure quality against the number of
evaluations on images.

### Skipping Steps on the Running Example

With the exact predictor, any difference between a ten-step sample and a
thousand-step sample from the same start is error of the stride. The cell
runs DDIM on the variance-preserving grid with 5 to 1,000 steps and prints
the mean distance to the thousand-step samples and the spread (the mean
squared distance to the nearest mode).

```{.python .input #samplers-skipping-steps-on-the-running-example}
%%tab pytorch
samples = {n: d2l.ddim_sample(pred_exact, z, vp, n)
           for n in (5, 10, 20, 50, 200, 1000)}
for n, x in samples.items():
    gap = (x - samples[1000]).norm(dim=1).mean()
    print(f'{n:4d} steps: distance to 1,000 steps {gap:.3f}, '
          f'spread {mix.summary(x)["spread"]:.3f}')
d2l.scatter_panels([samples[10][:2000], samples[200][:2000]],
                   ['10 steps', '200 steps'], data=mix.sample(2000));
```

```{.python .input #samplers-skipping-steps-on-the-running-example}
%%tab jax
samples = {n: d2l.ddim_sample(pred_exact, z, vp, n)
           for n in (5, 10, 20, 50, 200, 1000)}
for n, x in samples.items():
    gap = jnp.linalg.norm(x - samples[1000], axis=1).mean()
    print(f'{n:4d} steps: distance to 1,000 steps {gap:.3f}, '
          f'spread {mix.summary(x)["spread"]:.3f}')
d2l.scatter_panels([samples[10][:2000], samples[200][:2000]],
                   ['10 steps', '200 steps'],
                   data=mix.sample(jax.random.key(1), 2000));
```

The distance halves each time the number of steps doubles, as for a
first-order method: about 0.4 at five steps, 0.2 at ten and 0.04 at fifty,
on a scale where the modes are five units apart. The error has a direction:
the spread falls from about 0.47 at a thousand steps to 0.38 at ten, and the
clouds in the left panel are visibly tighter than the data behind them. A
stride holds the noise estimate at its value at the start, which for
Gaussian data always removes too much noise (exercise
:exref:`ex_samplers-gaussian-data`), and the modes of the mixture behave
like Gaussians. Since the spread rewards contraction, and a stochastic
sample has no thousand-step counterpart, we compare $\eta = 0$ with $\eta =
1$ on images instead.

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
$0.75$ and Heun's $0.80$: errors of $0.041$ and $0.009$, a fifth of the
error for twice the cost. The last step, to $\bar\sigma = 0$, is an Euler
step, and so is a first step from $\bar\sigma = \infty$, so `heun_sample`
spends $2n - 1$ evaluations on $n$ steps, or $2n - 2$ from pure
noise on the linear path.

Whether the second evaluation pays depends on where the steps are, and the
answer depends on the field. On the exact field of a one-dimensional mixture
(:numref:`sec_mdl-sampling-learned-dynamics`), Heun's method loses to
Euler's below about forty evaluations when the steps are equal in $t$, and
wins at every budget from ten evaluations when the steps are crowded toward
the data: step errors of opposite sign cancel, Euler's on equal steps and
Heun's on crowded ones. :citet:`Karras.Aittala.Aila.ea.2022` crowd the steps
by spacing $\bar\sigma^{1/\rho}$ evenly,

$$\bar\sigma_i = \Big(\bar\sigma_{\max}^{1/\rho} + \tfrac{i}{n - 1}\big(\bar\sigma_{\min}^{1/\rho} - \bar\sigma_{\max}^{1/\rho}\big)\Big)^{\rho}, \qquad i = 0, \ldots, n - 1,$$

followed by a last step to $\bar\sigma = 0$, with $\bar\sigma_{\max} = 80$,
$\bar\sigma_{\min} = 0.002$ and $\rho = 7$ for images in $[-1, 1]$; this is
`edm_grid`. The grid is a choice of the sampler, separate
from the training path, since `make_pred` translates any level to the
network's own time. The cell saves `edm_grid` and `heun_sample`, and prints
$\bar\sigma$ on this grid and on the ten-step grids of the variance-preserving and linear paths, even in time.

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

The variance-preserving grid spends four of its ten steps above $\bar\sigma
= 6$, where all noisy points look alike, and reaches the data from $0.34$.
The linear grid's finite levels run from $9$ to $0.11$. The EDM grid starts
at $80$ but crowds its last levels down to $0.002$. Exponential integrators
such as DPM-Solver :cite:`Lu.Zhou.Bao.ea.2022` take steps in $\log
\bar\sigma$ (exercise :exref:`ex_samplers-dpm-solver`).

### Quality against Evaluations on Images

On images the field is learned. We train the unconditional version of the
model of :numref:`sec_diffusion-images`, the U-Net of about 1.2 million
parameters without its label input, on the velocity of the linear path for
ten thousand updates, a few minutes on one GPU, and sample from the moving
average of its weights.

```{.python .input #samplers-quality-against-evaluations-on-images-1}
%%tab pytorch
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()

def batch_fn():
    return X[torch.randint(0, len(X), (128,), device=X.device)]

torch.manual_seed(0)
net = d2l.UNet(ch=64).to(X.device)
net, ema, _ = d2l.train_regression(net, d2l.regression_loss(lin, 'velocity'),
                                   batch_fn, 10_000)
pred = d2l.make_pred(ema, lin, 'velocity')
```

```{.python .input #samplers-quality-against-evaluations-on-images-1}
%%tab jax
X, y, X_test, y_test = d2l.load_fashion_mnist_pm1()

def batch_fn(key):
    return X[jax.random.randint(key, (128,), 0, len(X))]

net = d2l.UNet(ch=64, rngs=nnx.Rngs(0))
net, ema, _ = d2l.train_regression(net, d2l.regression_loss(lin, 'velocity'),
                                   batch_fn, 10_000, jax.random.key(0))
pred = d2l.make_pred(ema, lin, 'velocity')
```

We score samples by the Fréchet distance (FD) of
:numref:`sec_diffusion-images` in `d2l.FeatureCNN` features, 1,000 generated
against 1,000 test images, read against the floor of two disjoint sets of
1,000 real images.

```{.python .input #samplers-quality-against-evaluations-on-images-2}
%%tab pytorch
cnn = d2l.fashion_feature_cnn(X, y)
features = lambda imgs: d2l.fd_features(cnn, imgs)
f_test = features(X_test[:1000])
floor = d2l.fid_score(features(X_test[1000:2000]), f_test)
print(f'floor, 1,000 real images against 1,000 others: FD {floor:.2f}')
```

```{.python .input #samplers-quality-against-evaluations-on-images-2}
%%tab jax
cnn = d2l.fashion_feature_cnn(X, y)
features = lambda imgs: d2l.fd_features(cnn, imgs)
f_test = features(X_test[:1000])
floor = d2l.fid_score(features(X_test[1000:2000]), f_test)
print(f'floor, 1,000 real images against 1,000 others: FD {floor:.2f}')
```

Each sampler draws 1,000 images from the same noise. "VP" and "linear"
denote even steps in time on that path, so DDIM with $\eta = 0$ on "VP" is
Euler's method on the first grid printed above; "EDM" denotes the crowded
grid. We count network evaluations, not steps.

```{.python .input #samplers-quality-against-evaluations-on-images-3}
%%tab pytorch
families = {  # name: (sampler(pred, noise, n), step counts n)
    'ancestral, VP': (lambda p, z, n: d2l.ancestral_sample(p, z, vp, n),
                      (50, 200, 1000)),
    'DDIM eta=1, VP': (lambda p, z, n: d2l.ddim_sample(p, z, vp, n, 1.0),
                       (10, 20, 50, 100)),
    'DDIM eta=0, VP': (lambda p, z, n: d2l.ddim_sample(p, z, vp, n),
                       (10, 20, 50, 100)),
    'Euler, linear': (lambda p, z, n: d2l.ode_sample(p, z, lin.sigma_bars(n)),
                      (10, 20, 50)),
    'Euler, EDM': (lambda p, z, n: d2l.ode_sample(p, z, d2l.edm_grid(n)),
                   (10, 20, 50)),
    'Heun, linear': (lambda p, z, n: d2l.heun_sample(p, z, lin.sigma_bars(n)),
                     (5, 10, 25)),
    'Heun, EDM': (lambda p, z, n: d2l.heun_sample(p, z, d2l.edm_grid(n)),
                  (5, 10, 25))}

def evaluations(name, n):  # Heun's second slope is skipped at inf and at 0
    return {'Heun, linear': 2 * n - 2, 'Heun, EDM': 2 * n - 1}.get(name, n)

torch.manual_seed(1)
noise = torch.randn(1000, 1, 28, 28, device=X.device)
images = {name: [f(pred, noise, n).clamp(-1, 1) for n in steps]
          for name, (f, steps) in families.items()}
fd = {name: [d2l.fid_score(features(x), f_test) for x in xs]
      for name, xs in images.items()}
for name, (_, steps) in families.items():
    print(f'{name:15s}', '  '.join(f'{evaluations(name, n):4d}: {v:5.2f}'
                                   for n, v in zip(steps, fd[name])))
```

```{.python .input #samplers-quality-against-evaluations-on-images-3}
%%tab jax
families = {  # name: (sampler(pred, noise, n, key), step counts n)
    'ancestral, VP': (lambda p, z, n, k: d2l.ancestral_sample(p, z, vp, n, k),
                      (50, 200, 1000)),
    'DDIM eta=1, VP': (lambda p, z, n, k: d2l.ddim_sample(p, z, vp, n, 1.0, k),
                       (10, 20, 50, 100)),
    'DDIM eta=0, VP': (lambda p, z, n, k: d2l.ddim_sample(p, z, vp, n),
                       (10, 20, 50, 100)),
    'Euler, linear': (lambda p, z, n, k: d2l.ode_sample(p, z,
                                                        lin.sigma_bars(n)),
                      (10, 20, 50)),
    'Euler, EDM': (lambda p, z, n, k: d2l.ode_sample(p, z, d2l.edm_grid(n)),
                   (10, 20, 50)),
    'Heun, linear': (lambda p, z, n, k: d2l.heun_sample(p, z,
                                                         lin.sigma_bars(n)),
                     (5, 10, 25)),
    'Heun, EDM': (lambda p, z, n, k: d2l.heun_sample(p, z, d2l.edm_grid(n)),
                  (5, 10, 25))}

def evaluations(name, n):  # Heun's second slope is skipped at inf and at 0
    return {'Heun, linear': 2 * n - 2, 'Heun, EDM': 2 * n - 1}.get(name, n)

k_noise, k_run = jax.random.split(jax.random.key(1))
noise = jax.random.normal(k_noise, (1000, 28, 28, 1))
keys = iter(jax.random.split(k_run, 32))
images = {name: [f(pred, noise, n, next(keys)).clip(-1, 1) for n in steps]
          for name, (f, steps) in families.items()}
fd = {name: [d2l.fid_score(features(x), f_test) for x in xs]
      for name, xs in images.items()}
for name, (_, steps) in families.items():
    print(f'{name:15s}', '  '.join(f'{evaluations(name, n):4d}: {v:5.2f}'
                                   for n, v in zip(steps, fd[name])))
```

The plot shows each FD against the number of evaluations.

```{.python .input #samplers-quality-against-evaluations-on-images-4}
%%tab pytorch, jax
evals = [[evaluations(name, n) for n in steps]
         for name, (_, steps) in families.items()]
d2l.plot(evals + [[5, 1000]], list(fd.values()) + [[floor, floor]],
         'network evaluations', 'FD', legend=list(families) + ['floor'],
         xscale='log', yscale='log', figsize=(6, 4),
         fmts=('o-', 's-', '^-', 'v--', 'D--', 'x-.', '+-.', 'k:'))
```

The thousand-step ancestral chain, the reference, scores two to three times
the floor. Deterministic DDIM at fifty steps scores about 3, within about 1.5
times the chain's FD at a twentieth of the evaluations, which reproduces the
central result of :citet:`Song.Meng.Ermon.2020`. At equal numbers of steps,
$\eta = 0$ beats $\eta = 1$ by about a third at 10, 20 and 50 steps, and the
gap narrows to about a fifth at a hundred, as the trade-off above predicts.
The ancestral sampler at fifty steps scores worse than deterministic DDIM at
ten, because it adds more noise than the posterior allows; its images are
speckled (exercise
:exref:`ex_samplers-ancestral-strides`).

Heun's method on the crowded grid scores about 2 at nineteen evaluations,
better than DDIM at fifty; on the linear grid at forty-eight evaluations it
scores about 2, the quality of the thousand-step chain at a
twentieth of its evaluations. The grid matters as much as the method: with
ten steps, Euler's method scores about 7 on the linear grid, 9 to 10 on the
variance-preserving grid (where it is DDIM), and about 13 on the crowded
grid. Crowding the steps toward the data hurts Euler's method at every
budget, and Heun's at eight and at forty-eight evaluations; only near twenty
do the two grids tie. On this model the even grid of the linear path serves
both methods best.

The one-dimensional result does not transfer, and the printed grids suggest
why. With ten levels the crowded grid spends four steps above $\bar\sigma =
4$, where all noisy images look alike, and three below $0.12$, where a $28
\times 28$ garment barely changes, leaving three steps for the range in which
its shape is decided; the linear grid spends seven of its ten steps there.
:citet:`Karras.Aittala.Aila.ea.2022` tuned $\rho = 7$ for 35 to 79
evaluations on larger images.

Below are eight images from each of four samplers, all from the same eight
noise images: the ancestral chain with 1,000 steps, DDIM with 10 and with 50
steps, and Heun's method on the crowded grid with 19 evaluations.

```{.python .input #samplers-quality-against-evaluations-on-images-5}
%%tab pytorch
rows = [images['ancestral, VP'][2], images['DDIM eta=0, VP'][0],
        images['DDIM eta=0, VP'][2], images['Heun, EDM'][1]]
d2l.show_gray(torch.cat([x[:8] for x in rows]), 4, 8);
```

```{.python .input #samplers-quality-against-evaluations-on-images-5}
%%tab jax
rows = [images['ancestral, VP'][2], images['DDIM eta=0, VP'][0],
        images['DDIM eta=0, VP'][2], images['Heun, EDM'][1]]
d2l.show_gray(jnp.concatenate([x[:8] for x in rows]), 4, 8);
```

The last three rows match column by column: the deterministic samplers
approximate one map from noise to images, and ten steps slightly blur the
image that fifty steps and Heun's method agree on. The first row does not
match, because the ancestral chain's image depends as much on the noise it
adds as on its start.

## Inversion and Editing

A deterministic sampler integrates a differential equation, which can be run
in either direction. Running it backwards assigns each image a noise image,
its *code*, which gives interpolation, editing and inpainting without
retraining.

### The Latent Code

`ddim_invert` runs the deterministic sampler backwards, from an image at $t = 0$ toward noise. Each step evaluates the network at the current point but tells it the noise level of the step's upper end, since the predictor needs $\sigma > 0$ and the first point has none.

```{.python .input #samplers-the-latent-code}
%%tab pytorch
@torch.no_grad()  #@save
def ddim_invert(pred, x, path, steps, t_end=1.0):
    """Deterministic DDIM run backwards, from data x at t = 0 to t_end."""
    a, s = path.grid(steps, 0.0, t_end)
    for i in range(steps):
        eps = pred(x, a[i + 1], s[i + 1])[1]    # x read at the next level
        x = a[i + 1] * (x - s[i] * eps) / a[i] + s[i + 1] * eps
    return x
```

```{.python .input #samplers-the-latent-code}
%%tab jax
def ddim_invert(pred, x, path, steps, t_end=1.0):  #@save
    """Deterministic DDIM run backwards, from data x at t = 0 to t_end."""
    a, s = path.grid(steps, 0.0, t_end)
    def step(x, inputs):
        a_t, s_t, a_s, s_s = inputs
        eps = pred(x, a_s, s_s)[1]              # x read at the next level
        return a_s * (x - s_t * eps) / a_t + s_s * eps, None
    return jax.lax.scan(step, x, (a[:-1], s[:-1], a[1:], s[1:]))[0]
```

In the coordinates $(\bar{\mathbf{x}}, \bar\sigma)$ each inverse step is the
Euler step $\bar{\mathbf{x}}_{t'} = \bar{\mathbf{x}}_t + (\bar\sigma_{t'} -
\bar\sigma_t)\, \hat{\boldsymbol\epsilon}$ taken upward, from the level $t$
to the next level $t' > t$. The decoder evaluates the noise estimate at the
upper point, the encoder at the lower one, the only point it has; the two
differ by the change of $\hat{\boldsymbol\epsilon}$ over one step, so the
round trip is inexact even with an exact predictor. The cell encodes and
decodes 100 test images with 10 to 100 steps on the linear path.

```{.python .input #samplers-the-latent-code-1}
%%tab pytorch
x0 = X_test[:100]
codes = {n: d2l.ddim_invert(pred, x0, lin, n) for n in (10, 20, 50, 100)}
recon = {n: d2l.ddim_sample(pred, code, lin, n) for n, code in codes.items()}
for n in codes:
    print(f'{n:3d} steps: squared error per pixel '
          f'{((recon[n] - x0) ** 2).mean():.4f}, code std {codes[n].std():.2f}')
d2l.show_gray(torch.cat([x0[:8], recon[10][:8], recon[50][:8]]), 3, 8);
```

```{.python .input #samplers-the-latent-code-1}
%%tab jax
x0 = X_test[:100]
codes = {n: d2l.ddim_invert(pred, x0, lin, n) for n in (10, 20, 50, 100)}
recon = {n: d2l.ddim_sample(pred, code, lin, n) for n, code in codes.items()}
for n in codes:
    print(f'{n:3d} steps: squared error per pixel '
          f'{((recon[n] - x0) ** 2).mean():.4f}, code std {codes[n].std():.2f}')
d2l.show_gray(jnp.concatenate([x0[:8], recon[10][:8], recon[50][:8]]), 3, 8);
```

The reconstruction error falls with the number of steps, more slowly than
the step size, since the learned field adds errors of its own: at fifty
steps (third row) the reconstructions are close to the originals, at ten
(second row) they lose fine detail. The codes are not typical noise: their
standard deviation lies below one, approaching one only as the steps get
shorter.

Because the sampler is a continuous function of its code, we can move
between two images by interpolating between their codes. A straight line is
the wrong path: the average of two independent noise images has a norm
smaller by about $\sqrt{2}$, which noise images of this dimension almost
never have. Spherical interpolation moves along the great circle between the
two codes instead,

$$\mathbf{z}_u = \frac{\sin\big((1 - u)\omega\big)\, \mathbf{z}_0 + \sin(u \omega)\, \mathbf{z}_1}{\sin \omega}, \qquad \cos \omega = \frac{\mathbf{z}_0^\top \mathbf{z}_1}{\|\mathbf{z}_0\| \|\mathbf{z}_1\|},$$

which keeps the norm close to that of the endpoints. We decode eight points
between the 50-step codes of the first two test images.

```{.python .input #samplers-the-latent-code-2}
%%tab pytorch
z0, z1 = codes[50][0], codes[50][1]
omega = torch.arccos(((z0 * z1).sum() / (z0.norm() * z1.norm())).clamp(-1, 1))
u = torch.linspace(0, 1, 8, device=z0.device).reshape(-1, 1, 1, 1)
zs = (torch.sin((1 - u) * omega) * z0 + torch.sin(u * omega) * z1
      ) / torch.sin(omega)
d2l.show_gray(d2l.ddim_sample(pred, zs, lin, 50), 1, 8);
```

```{.python .input #samplers-the-latent-code-2}
%%tab jax
z0, z1 = codes[50][0], codes[50][1]
omega = jnp.arccos(jnp.clip(
    (z0 * z1).sum() / (jnp.linalg.norm(z0) * jnp.linalg.norm(z1)), -1, 1))
u = jnp.linspace(0, 1, 8).reshape(-1, 1, 1, 1)
zs = (jnp.sin((1 - u) * omega) * z0 + jnp.sin(u * omega) * z1
      ) / jnp.sin(omega)
d2l.show_gray(d2l.ddim_sample(pred, zs, lin, 50), 1, 8);
```

The row moves from a reconstruction of the first test image to one of the
second, and every intermediate code decodes to a plausible item.

### Editing by Partial Noising

Inversion runs an image all the way to noise; stopping part of the way gives
an editor. :citet:`Meng.He.Song.ea.2022` noise a rough input, such as a
sketch or a crude collage, to an intermediate time $t_0$ with
$\mathbf{x}_{t_0} = \alpha_{t_0} \mathbf{y} + \sigma_{t_0}
\boldsymbol\epsilon$, and run the sampler from $t_0$ back to the data. The
noise hides the details of the input that real images do not have, the
sampler replaces them by details that real images do have, and the coarse
layout survives. The time $t_0$ sets the balance between fidelity to the
input and realism.

Our rough inputs are the ten class-mean images, blurred silhouettes that no
real garment looks like, each noised a hundred times to $t_0 \in \{0.3,
0.5, 0.7, 1\}$ and integrated back with 25 DDIM steps. The cell prints the
squared distance per pixel to the input (fidelity) and the FD of the 1,000
outputs (realism); the row $t_0 = 0$ holds the inputs.

```{.python .input #samplers-editing-by-partial-noising}
%%tab pytorch
means = torch.stack([X[y == c].mean(0) for c in range(10)])
inputs = means.repeat(100, 1, 1, 1)                  # 1,000 inputs
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
inputs = jnp.tile(means, (100, 1, 1, 1))             # 1,000 inputs
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

The distance to the input grows with $t_0$, and the FD falls from far above
that of any sampler toward that of fresh samples. At $t_0 = 0.3$ each output
keeps the silhouette of its mean image and gains the seams, prints and edges
of a real garment; at $0.5$ and $0.7$ class and pose survive while the
details vary; at $t_0 = 1$ the output is an unrelated sample.

### Inpainting and Posterior Sampling

Suppose that we know part of an image, the upper half, say, and want the
model to fill in the rest. Write $\mathbf{m}$ for the mask that is one on
the known pixels and $\mathbf{y}$ for their values. At every level we know
what the known pixels of $\mathbf{x}_s$ should look like, $\alpha_s
\mathbf{y}$ plus noise of scale $\sigma_s$, so replacement inpainting
:cite:`song2021score,Lugmayr.Danelljan.Romero.ea.2022` takes each sampler
step as usual and then overwrites the known pixels with a fresh noisy copy
of $\mathbf{y}$ at the new level,

$$\mathbf{x}_s \leftarrow \mathbf{m} \odot \big(\alpha_s \mathbf{y} + \sigma_s \boldsymbol\xi\big) + (\mathbf{1} - \mathbf{m}) \odot \mathbf{x}_s.$$

`inpaint` therefore runs the sampler one step at a time.

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
each, from different noise. The rows show the known halves, the three
completions and the originals.

```{.python .input #samplers-inpainting-and-posterior-sampling-2}
%%tab pytorch
mask = torch.zeros(1, 1, 28, 28, device=X.device)
mask[..., :14, :] = 1                                # the upper half is known
torch.manual_seed(3)
fills = inpaint(pred, X_test[:8].repeat(3, 1, 1, 1), mask, lin, 50)
d2l.show_gray(torch.cat([X_test[:8] * mask, fills, X_test[:8]]), 5, 8);
```

```{.python .input #samplers-inpainting-and-posterior-sampling-2}
%%tab jax
mask = jnp.zeros((1, 28, 28, 1)).at[:, :14].set(1)  # the upper half is known
fills = inpaint(pred, jnp.tile(X_test[:8], (3, 1, 1, 1)), mask, lin, 50,
                jax.random.key(3))
d2l.show_gray(jnp.concatenate([X_test[:8] * mask, fills, X_test[:8]]), 5, 8);
```

Most completions continue the known half into a plausible garment of the
same kind, some are malformed, and the three completions of one image
differ. The boundary is
not always clean: the network sees the known pixels only through noisy
copies, and the halves can disagree along the seam (exercise
:exref:`ex_samplers-repaint` measures and repairs this).

Most measurements, such as a blurred or low-resolution copy, are not a
subset of the pixels; for these, Bayes' rule adds the gradient of the
measurement's log-likelihood to the score, as guidance did for a class label
in :numref:`sec_diffusion-images`, and :citet:`Chung.Kim.Mccann.ea.2023`
evaluate that likelihood at the network's estimate
$\hat{\mathbf{x}}_0(\mathbf{x}_t)$.

## Summary

Training fixes only the distribution of the noisy point at each time, so one
network serves a family of samplers, from DDIM, which is Euler's method for
the probability-flow ODE in the coordinates $(\mathbf{x}/\alpha,
\sigma/\alpha)$, to a stochastic sampler with the posterior variance. Fresh
noise corrects errors but needs short steps, so the deterministic sampler
wins at small budgets: on Fashion-MNIST fifty DDIM steps came within about
1.5 times the FD of the thousand-step chain, and Heun's method matched the
chain at about fifty evaluations, with the grid mattering as much as the
method. Run backwards, the deterministic sampler gives every image a code,
and with it interpolation, editing and inpainting.

We have trained every model so far by an unweighted squared error on a
target we picked, without asking why this loss is the right one or how the
choices of target, path and weighting compare.
:numref:`sec_diffusion-design-space` derives the loss from the likelihood
and measures the choices.

## Exercises

1. **Admissible noise scales.** The family :eqref:`eq_diffusion-ddim-family`
   needs $\tilde\sigma \leq \sigma_s$.
    1. Show that a strength $\eta$ is admissible on a step from $t$ to $s$
       exactly when $\eta^2 \leq 1 / (1 - \kappa^2)$, so that $\eta = 1$ is
       admissible on every grid. At the largest admissible $\eta$, what does
       the step keep of $\mathbf{x}_t$ beyond $\hat{\mathbf{x}}_0$?
    1. Show that the last step of `ddim_sample`, to $t = 0$, returns
       $\hat{\mathbf{x}}_0$ for every $\eta$.
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
       \boldsymbol\nu$ of the noising chain of the previous section. Conclude
       that this chain is the only member of the family whose forward
       process is Markov.
    1. Show that with $\tilde\sigma = 0$ at every step the forward process
       is $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
       \boldsymbol\epsilon$ with one $\boldsymbol\epsilon$ shared by all
       times. Why can the regression not tell these forward processes apart?
   :exlabel:`ex_samplers-forward-step`
1. **Gaussian data.** Let the data be one-dimensional with distribution
   $\mathcal{N}(0, v)$, and work in the coordinates $(\bar x, \bar\sigma)$.
    1. Show that the exact noise estimate is $\hat\epsilon = \bar\sigma \bar
       x / (v + \bar\sigma^2)$, that one deterministic stride from
       $\bar\sigma$ to $\bar\sigma'$ multiplies $\bar x$ by $(v + \bar\sigma
       \bar\sigma') / (v + \bar\sigma^2)$, and that the exact flow
       multiplies it by $\sqrt{(v + \bar\sigma'^2) / (v + \bar\sigma^2)}$.
    1. Show that the factor of the stride never exceeds the exact one, with
       equality only for $\bar\sigma' = \bar\sigma$. What does this imply for
       the variance of deterministic samples of Gaussian data on any grid?
    1. Take $v = 0.25$, the variance of one coordinate of a mode of `mix`.
       Compute the standard deviation of deterministic samples from $x_1 \sim
       \mathcal{N}(0, 1)$ on the variance-preserving grids with 200, 50, 20,
       10 and 5 steps, and compare it with $0.5$ and with the spreads printed
       in this section.
   :exlabel:`ex_samplers-gaussian-data`
1. [code] **The strength $\eta$ and the code.** Use the image model of this
   section.
    1. Compute the FD of 1,000 samples of `ddim_sample` for
       $\eta \in \{0, 0.25, 0.5, 1\}$ at 10, 20 and 50 steps.
    1. Fix one noise image $\mathbf{x}_1$ and draw 16 samples from it for
       each $\eta$ with different random seeds. Compute the standard
       deviation of each pixel across the 16 samples and average it over the
       pixels. How much does the image depend on $\mathbf{x}_1$ at each
       $\eta$?
    1. Which $\eta$ would you use at each budget, and which if you need the
       sample to be a function of its code?
   :exlabel:`ex_samplers-eta-strength`
1. [code] **Inversion on both examples.** Encode 20,000 draws of `mix` with
   `ddim_invert` and `pred_exact` on the variance-preserving grid with 10,
   20, 50, 100 and 200 steps, and decode each set of codes with
   `ddim_sample` on the same grid.
    1. How does the mean reconstruction error depend on the number of steps?
       Does it fall in proportion to the step size?
    1. Compute the mean and the covariance of the 200-step codes. Are they
       those of standard Gaussian noise?
    1. Repeat part (a) for the 100 test images with `pred` on the linear
       path, adding 200 steps, and compare the trend with Table 2 of
       :citet:`Song.Meng.Ermon.2020`.
   :exlabel:`ex_samplers-inversion-steps`
1. [extended] **A second-order sampler in log-SNR.** ●● DPM-Solver-2
   :cite:`Lu.Zhou.Bao.ea.2022` takes a step from $\bar\sigma$ to
   $\bar\sigma'$ in two evaluations, the second at the midpoint
   $\bar\sigma_m = \sqrt{\bar\sigma \bar\sigma'}$ in $\log \bar\sigma$:
   $\mathbf{u} = \bar{\mathbf{x}} + (\bar\sigma_m - \bar\sigma)\,
   \hat{\boldsymbol\epsilon}(\bar{\mathbf{x}}, \bar\sigma)$ and
   $\bar{\mathbf{x}}' = \bar{\mathbf{x}} + (\bar\sigma' - \bar\sigma)\,
   \hat{\boldsymbol\epsilon}(\mathbf{u}, \bar\sigma_m)$.
    1. Let $h = \log(\bar\sigma / \bar\sigma')$. Write the exact change of
       $\bar{\mathbf{x}}$ over the step as an integral of
       $\hat{\boldsymbol\epsilon}$ along the trajectory, and show that the
       local error of the update is $O(\bar\sigma h^3)$ while that of the
       Euler step is $O(\bar\sigma h^2)$. Assume that
       $\hat{\boldsymbol\epsilon}$ is a smooth function of $\log \bar\sigma$
       along the trajectory and Lipschitz in $\bar{\mathbf{x}}$.
    1. Implement the update with the interface `pred(x, alpha, sigma)`,
       using $\alpha = 1 / \sqrt{1 + \bar\sigma^2}$ and $\sigma = \bar\sigma
       \alpha$, and take the last step to $\bar\sigma = 0$ with Euler's
       method. Compare it with `ode_sample` and `heun_sample` at about 11,
       21 and 51 evaluations, on the mixture with `pred_exact` by the
       distance to a 1,000-step reference, and on images by the FD.
    1. Repeat with levels evenly spaced in $\log \bar\sigma$ between 80 and
       0.002. Which combination of method and grid is best at each budget?
   :exlabel:`ex_samplers-dpm-solver`
1. [code] **Fewer ancestral steps.** Run `ancestral_sample` and
   `ddim_sample` with $\eta = 1$ on the mixture with `pred_exact`, on the
   variance-preserving grids with 200, 50, 20 and 10 steps, from 20,000
   starting points.
    1. Compare the `mix.summary` statistics of the two samplers with those of
       the data. Where does each sampler break first: in the mode weights, in
       the spread or in the stray fraction?
    1. Relate the change of the spread to the variance error of one
       ancestral step (exercise :exref:`ex_gaussian-paths-ancestral-first-order`)
       and to the factor $\sigma_s^2 / \sigma_t^2$ between the two samplers'
       variances.
    1. Show 16 images of each sampler at 50 steps on the image model, and
       explain the speckle of the ancestral ones.
   :exlabel:`ex_samplers-ancestral-strides`
1. **The optimal reverse variance.** ● A stochastic step from $t$ to $s$ moves
   to the posterior mean $\alpha_s \hat{\mathbf{x}}_0 + \sigma_s \kappa
   \hat{\boldsymbol\epsilon}$ and adds Gaussian noise of variance $v$ per
   coordinate. :citet:`ho2020denoising` tried the forward-step variance and
   the posterior variance; :citet:`Bao.Li.Zhu.ea.2022` found the best $v$
   for any data. Assume exact estimates, and let $\bar e_t =
   \mathbb{E}\|\boldsymbol\epsilon - \hat{\boldsymbol\epsilon}\|^2 / d$ be
   the error per coordinate of the exact noise estimate at time $t$ in $d$
   dimensions.
    1. Show that the true reverse transition $q(\mathbf{x}_s \mid
       \mathbf{x}_t)$ has mean $\alpha_s \hat{\mathbf{x}}_0 + \sigma_s \kappa
       \hat{\boldsymbol\epsilon}$ and covariance $\tilde\sigma^2 \mathbf{I} +
       k^2 \mathrm{Cov}(\mathbf{x}_0 \mid \mathbf{x}_t)$, where
       $\tilde\sigma^2$ is the posterior variance and $k = \alpha_s -
       \sigma_s \kappa\, \alpha_t / \sigma_t$.
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
       paths with 10 steps. Show that the samples differ, and explain why
       with exercise :exref:`ex_gaussian-paths-same-ratio-same-flow`.
    1. :citet:`Gao.Hoogeboom.Heek.ea.2025` conclude that DDIM is unaffected
       by such a rescaling. Which of the two choices, the path or the grid of
       times, can change the samples of DDIM?
   :exlabel:`ex_samplers-rescaled-schedule`

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §17.4]{.kicker}

Samplers, Inversion and Editing<br>
**one network, many samplers · DDIM is Euler's method · Heun and the step grid · quality against evaluations · codes, edits and inpainting**
:::
:::

::: {.slide title="The Ancestral Chain Calls the Network a Thousand Times per Image"}
[The problem]{.kicker}

- Each sampler step is one evaluation of the U-Net; the chain takes 1,000.
- On Fashion-MNIST that chain scores a Fréchet distance (FD, lower is
  better) of two to three times the floor of real against real images.

::: {.d2l-note .warn}
Which of the thousand steps do we need? And if sampling integrates a field,
can we run it backwards, from an image to its noise?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive a family of samplers that one trained network serves;
- show that its deterministic member, DDIM, is Euler's method;
- compute the error of an Euler and a Heun step by hand;
- read sample quality against network evaluations, and pick a sampler;
- invert an image to its code, edit it, and inpaint it.

**Running examples:** the three-mode mixture with its exact predictor;
a 1.2M-parameter U-Net on Fashion-MNIST.
:::

::: {.slide title="This Deck Turns a Trained Network into Fast Samplers"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Denoising and the Score | what a denoiser knows about the data |
| Gaussian Paths | how to walk from noise to data |
| Diffusion for Images | what changes for images |
| **Samplers, Inversion and Editing** | **which steps to skip; running backwards** |
| Likelihood and the Design Space | is the regression maximum likelihood? |
| Latent Diffusion and Transformers | why diffuse a compressed code |
| Few-Step Generation | why one step fails |
| Discrete Diffusion | what noise is for a token |
:::

::: {.col .narrow}
- **Uses:** the path $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
  \boldsymbol\epsilon$, data at $t = 0$, noise at $t = 1$; the
  probability-flow ODE; FD.
- **Skips:** distillation and one-step models.
:::
:::
:::

::: {.slide title="Every Sampler Reads the Network as a Pair of Estimates"}
At a noisy point $\mathbf{x}_t = \alpha_t \mathbf{x}_0 + \sigma_t
\boldsymbol\epsilon$ the network gives a clean estimate
$\hat{\mathbf{x}}_0$ and a noise estimate $\hat{\boldsymbol\epsilon}$;
each fixes the other:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sigma_t \hat{\boldsymbol\epsilon}}{\alpha_t}$$

- Noise, velocity or clean-image target: `pred(x, alpha, sigma)` returns
  $(\hat{\mathbf{x}}_0, \hat{\boldsymbol\epsilon})$ at any level.
- The samplers of this deck read nothing else.
:::

::: {.slide title="Training Fixes Only the Marginals, so Many Steps Are Valid"}
The regression sees single noisy points $\mathbf{x}_t$, never chains. A step
from $t$ to an earlier time $s < t$ only has to produce the law of
$\mathbf{x}_s$. For any $0 \le \tilde\sigma \le \sigma_s$:

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
- **$\eta = 1$:** the posterior of the noising chain, a fraction
  $f = 1 - \kappa^2$ of the noise variance fresh.
- **Ancestral chain:** not $\eta = 1$; it adds the forward-step variance,
  larger by the factor $\sigma_t^2 / \sigma_s^2$.

[Song, Meng and Ermon, 2020]{.small}
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

::: {.slide title="DDIM and Euler Agree to 32-Bit Rounding"}
Ten steps on the linear path, the mixture's exact predictor, the same
20,000 starting points:

@!samplers-the-ddim-family-and-euler-s-method-1

Two coordinate systems, one function: differences below $10^{-4}$, on a
scale where the modes are five units apart.
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
replaced. On a fine grid neighbouring levels have nearly equal SNR, so $f$
is small even at $\eta = 1$.
:::
:::
:::

::: {.slide title="Both Ends of the Family Reach the Same Distribution"}
::: {.cols .vc}
::: {.col .fig .big}
![](../img/mdl-diffusion-two-samplers.svg)
:::

::: {.col}
200 steps from six starts (circles) to $t = 0$ (crosses). (a) $\eta = 0$:
smooth curves. (b) $\eta = 1$: jagged paths that may change mode.
:::
:::

- Fresh noise pulls the state toward the model's distribution (a Langevin
  step), so errors are partly forgotten.
- A long stochastic step errs: the true reverse step is far from Gaussian.
:::

::: {.slide title="The Stride Error Halves When the Steps Double"}
::: {.cols .vc}
::: {.col}
@!samplers-skipping-steps-on-the-running-example
:::

::: {.col .narrow}
Exact predictor, VP grid: distance of each sample to its 1,000-step
counterpart, and the spread around the nearest mode.

- About 0.4 at five steps, 0.2 at ten, 0.04 at fifty: first order.
- The spread falls from about 0.47 to 0.38 at ten steps: ten-step clouds
  are too tight.
:::
:::
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

::: {.d2l-note}
Euler always removes too much noise from Gaussian data. Heun: a fifth of
the error for two evaluations per step.
:::
:::

::: {.slide title="The Grid Matters as Much as the Method"}
Levels $\bar\sigma$ of three ten-step grids; EDM is even in
$\bar\sigma^{1/7}$ from 80 to 0.002:

::: {.cols}
::: {.col}
@!samplers-second-order-and-the-step-grid
:::
:::

| Euler, 10 steps (floor 0.8) | linear | VP (DDIM) | EDM |
|:--|:--|:--|:--|
| FD | about 7 | 9 to 10 | about 13 |

::: {.d2l-note}
VP and EDM spend four steps where noisy images look alike; EDM puts three
more below 0.12. The linear grid puts seven where the shape is decided.
:::

[Karras et al., 2022]{.small}
:::

::: {.slide title="Fifty DDIM Steps Score within About 1.5 Times the Thousand-Step Chain"}
@!samplers-quality-against-evaluations-on-images-4

1,000 samples from the same noise against 1,000 test images. Floor about
0.8; DDIM at 50 steps scores about 3, at a twentieth of the chain's
evaluations. Heun on the EDM grid: about 2 at 19 evaluations.
:::

::: {.slide title="At Equal Steps η = 0 Beats η = 1; the Ancestral Chain Needs Many Steps"}
FD at each number of network evaluations:

@!samplers-quality-against-evaluations-on-images-3

- $\eta = 0$ is about a third lower than $\eta = 1$ at 10 to 50 steps,
  about a fifth at 100.
- Ancestral at 50 steps scores worse than DDIM at 10.
:::

::: {.slide title="Deterministic Samplers Approximate One Map from Noise to Images"}
@!samplers-quality-against-evaluations-on-images-5

Rows: ancestral 1,000 steps; DDIM 10; DDIM 50; Heun on the EDM grid, 19
evaluations. The last three match column by column; ten steps blur. The
ancestral image depends on the noise it adds.
:::

::: {.slide title="Check: Why Does the Ancestral Chain Lose at Fifty Steps?"}
[Check]{.kicker}

At 50 steps the ancestral chain scores worse than DDIM at 10, while at
1,000 steps it is the reference. What changes between the two grids?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** It adds the forward-step variance, larger than the posterior
variance by $\sigma_t^2/\sigma_s^2$. On a thousand-step grid the factor is
near one; on fifty steps it is not, and the images come out speckled.
:::
:::
:::

::: {.slide title="Running the Euler Steps Upward Gives Every Image a Code"}
@-samplers-the-latent-code

- Each step reads $\hat{\boldsymbol\epsilon}$ at the lower point, the only
  one it has; the decoder reads it at the upper point.
- The round trip is inexact even with an exact predictor.
:::

::: {.slide title="Reconstruction Improves with Steps; Codes Are Not Typical Noise"}
100 test images, encoded and decoded on the linear path:

@!samplers-the-latent-code-1

Rows: originals, 10 steps, 50 steps. Error per pixel about 0.06 at 10
steps, 0.01 at 100; the code's standard deviation stays below one.
:::

::: {.slide title="Spherical Interpolation Keeps Codes at the Norm of Noise"}
$$\mathbf{z}_u = \frac{\sin\big((1 - u)\omega\big)\, \mathbf{z}_0 + \sin(u \omega)\, \mathbf{z}_1}{\sin \omega}, \qquad \cos \omega = \frac{\mathbf{z}_0^\top \mathbf{z}_1}{\|\mathbf{z}_0\| \|\mathbf{z}_1\|}$$

@!samplers-the-latent-code-2

From the first test image to the second; every intermediate code decodes to
a plausible item.
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
- FD falls from about 90 for the inputs to about 3 at $t_0 = 1$.

[Meng et al., 2022]{.small}
:::
:::
:::

::: {.slide title="Inpainting Resets the Known Pixels at Every Level"}
::: {.cols .vc}
::: {.col}
@!samplers-inpainting-and-posterior-sampling-2
:::

::: {.col}
Mask $\mathbf{m}$ (one on known pixels), values $\mathbf{y}$; after each
DDIM step:

$$\mathbf{x}_s \leftarrow \mathbf{m} \odot \big(\alpha_s \mathbf{y} + \sigma_s \boldsymbol\xi\big) + (\mathbf{1} - \mathbf{m}) \odot \mathbf{x}_s$$

- Rows: known half, three completions, originals.
- Seams can disagree.
:::
:::
:::

::: {.slide title="Check: How Much Does Averaging Two Codes Shrink Them?"}
[Check]{.kicker}

A $28 \times 28$ noise image has norm about $\sqrt{784} = 28$. What is the
norm of the average of two independent ones?

::: {.fragment .fade-up}
::: {.d2l-note}
**Answer.** About $28/\sqrt{2} \approx 20$. Noise norms in 784
dimensions have a standard deviation of about 0.7, so the midpoint of a
straight line is a code that noise almost never produces.
:::
:::
:::

::: {.slide title="Recap"}
[Wrap-up]{.kicker}

- **One network, a family:** $\eta = 0$ is DDIM, Euler's method in
  $(\mathbf{x}/\alpha, \sigma/\alpha)$; $\eta = 1$ uses the posterior variance.
- **Mixture:** stride error about 0.4, 0.2, 0.04 at 5, 10, 50 steps.
- **Images:** floor about 0.8; DDIM-50 about 3; Heun about 2 at 19
  evaluations; the grid matters as much as the method.
- **Backwards:** codes give interpolation, editing, inpainting.

**Next:** is the regression maximum likelihood, and which choices matter?
:::
