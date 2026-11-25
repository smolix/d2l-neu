# Adversarial Losses Beyond GANs
:label:`sec_gan_beyond`

When predicting a single scalar such as a house price, minimizing mean squared error ($L_2$ loss) is standard practice. On conditional image generation problems, such as super-resolution (mapping a low-resolution thumbnail $c$ to a high-resolution image $x$) or sketch-to-photo translation (pix2pix, :citet:`Isola.Zhu.Zhou.ea.2017`), pixelwise $L_2$ and $L_1$ regression losses produce blurry outputs. Many distinct high-resolution textures correspond to the same low-resolution input, and minimizing pixelwise squared error forces the network to output their conditional average $\mathbb{E}[x \mid c]$, a blur that belongs to none of the plausible sharp images. Pairing the regression loss with a discriminator $D(x, c)$ that scores whether the output is a realistic image for condition $c$ restores sharp high-frequency detail.

This mechanism explains why adversarial losses remain widely used even though diffusion and flow-matching models (:numref:`chap_diffusion`) have replaced standalone GANs for general image synthesis. A learned discriminator still trains one-step generators distilled from multi-step diffusion models, the image autoencoders (tokenizers) on which latent diffusion models operate, and neural audio vocoders that reconstruct speech waveforms from spectrograms. Below, we analyze the conditional-mean averaging effect on a minimal bimodal regression problem, demonstrate why one-step distillation from a high-noise denoising teacher collapses to the conditional mean without a discriminator, and summarize where adversarial losses are used in modern generative pipelines.

```{.python .input #adversarial-losses-adversarial-losses-beyond-gans}
%%tab pytorch
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
```

```{.python .input #adversarial-losses-adversarial-losses-beyond-gans}
%%tab jax
import math
import numpy as np
import jax
from jax import numpy as jnp
from flax import nnx
import optax
from d2l import jax as d2l
```

## Conditional Means and Multimodal Targets

### The Conditional-Mean Decomposition

Let $g(c)$ be a deterministic predictor of $x$ given condition $c$, and let $\mu(c) = \mathbb{E}[x \mid c]$ denote the conditional mean. Adding and subtracting $\mu(c)$ inside $\mathbb{E}[\|x - g(c)\|^2]$ eliminates the cross term $2\,\mathbb{E}[\langle x - \mu(c), \mu(c) - g(c) \rangle] = 0$ (by iterated expectation given $c$), leaving the orthogonal decomposition:

$$
\begin{aligned}
\mathbb{E}\big[(x - g(c))^2\big]
&= \mathbb{E}\big[(x - \mu(c))^2\big] + \mathbb{E}\big[(\mu(c) - g(c))^2\big].
\end{aligned}
$$
:eqlabel:`eq_beyond_condmean`

The first term is the irreducible conditional variance $\operatorname{Var}(x \mid c)$, and the second is the squared distance between $g(c)$ and $\mu(c)$. Thus, squared error is uniquely minimized by the conditional mean $g^\star(c) = \mu(c)$. When $p(x \mid c)$ is unimodal and concentrated, $\mu(c)$ is a representative target; when $p(x \mid c)$ is multimodal, $\mu(c)$ falls in the low-density region between modes. Replacing $L_2$ with $L_1$ loss $\mathbb{E}[|x - g(c)|]$ merely replaces the conditional mean with the coordinate-wise conditional median (Exercise :exref:`ex_adversarial-losses-mean-and-median`), which likewise lies outside the support of a symmetric bimodal distribution.

### A Bimodal Two-Band Benchmark
:label:`subsec_gan_beyond_capacity`

Consider a 1D conditional distribution with $c \sim \textrm{Uniform}[-1, 1]$ and two parallel data bands separated by an empty corridor:

$$
\begin{aligned}
x &= c + b + 0.05\,\xi, \\
b &\in \{-1, +1\} \textrm{ with probability } \tfrac12 \textrm{ each},
\quad \xi \sim \mathcal{N}(0, 1).
\end{aligned}
$$
:eqlabel:`eq_beyond_twoband`

The conditional mean $\mu(c) = c$ runs down the empty corridor between the bands $x = c \pm 1$, achieving $\mathbb{E}[(x - \mu(c))^2] = 1 + 0.05^2 = 1.0025$. By contrast, a predictor that commits to the upper band $g(c) = c + 1$ has squared error $\tfrac12(0) + \tfrac12(2^2) + 0.05^2 = 2.0025$—twice as large. Squared error therefore penalizes committing to a valid data mode.

A conditional discriminator $D(x, c)$, however, estimates the log density ratio (:numref:`sec_gan_conditional`):

$$
D^\star(x, c) = \log \frac{p(x, c)}{q(x, c)}.
$$

In the empty corridor where $p(x, c) \approx 0$, $D^\star(x, c)$ is strongly negative, so minimizing the non-saturating adversarial loss $-\log \sigma(D(\hat x, c))$ forces the predictor $\hat x = g(c)$ onto one of the high-density data bands.

We test this by training two identical 2-hidden-layer $\tanh$ MLPs (`g_mse` and `g_adv`) from the same initial weights: `g_mse` minimizes squared error, while `g_adv` trains against a leaky-ReLU discriminator $D(x, c)$ using `d2l.gan_loss_D` and `d2l.gan_loss_G` (:numref:`sec_gan_convergence`).

```{.python .input #adversarial-losses-the-capacity-argument-1}
%%tab pytorch
def make_bands(n):
    """Draw n pairs (c, x) from the two bands x = c + b + 0.05 xi."""
    c = torch.rand(n, 1) * 2 - 1
    b = torch.randint(0, 2, (n, 1)) * 2.0 - 1.0
    return c, c + b + 0.05 * torch.randn(n, 1)

torch.manual_seed(0)
g_mse = d2l.make_mlp([1, 32, 32, 1], nn.Tanh())
g_adv = d2l.make_mlp([1, 32, 32, 1], nn.Tanh())
g_adv.load_state_dict(g_mse.state_dict())  # both students start alike
D = d2l.make_mlp([2, 32, 32, 1], nn.LeakyReLU(0.2))
c_data, x_data = make_bands(200)  # a data sample for the plot
```

```{.python .input #adversarial-losses-the-capacity-argument-1}
%%tab jax
def make_bands(key, n):
    """Draw n pairs (c, x) from the two bands x = c + b + 0.05 xi."""
    k_c, k_b, k_xi = jax.random.split(key, 3)
    c = jax.random.uniform(k_c, (n, 1), minval=-1, maxval=1)
    b = jax.random.rademacher(k_b, (n, 1), dtype=jnp.float32)
    return c, c + b + 0.05 * jax.random.normal(k_xi, (n, 1))

rngs = nnx.Rngs(0)
g_mse = d2l.make_mlp([1, 32, 32, 1], nnx.tanh, rngs=rngs)
g_adv = nnx.clone(g_mse)  # both students start alike
D = d2l.make_mlp([2, 32, 32, 1], lambda h: nnx.leaky_relu(h, 0.2), rngs=rngs)
key_mse, key_adv, key_data = jax.random.split(rngs(), 3)
c_data, x_data = make_bands(key_data, 200)  # a data sample for the plot
```

The first student minimizes squared error with Adam, its learning rate decaying along a cosine from 0.01 to zero over 2,000 steps. We need no experiment to know where it ends: by :eqref:`eq_beyond_condmean` it should follow the line $x = c$, one unit from each band. It is the control for the second student.

```{.python .input #adversarial-losses-two-bands-1}
%%tab pytorch
def train_mse(g, num_steps=2000, batch_size=128, lr=0.01):
    opt = torch.optim.Adam(g.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, num_steps)
    for _ in range(num_steps):
        c, x = make_bands(batch_size)  # a fresh batch at every step
        opt.zero_grad()
        ((g(c) - x) ** 2).mean().backward()
        opt.step()
        sched.step()

train_mse(g_mse)
```

```{.python .input #adversarial-losses-two-bands-1}
%%tab jax
@nnx.jit(static_argnums=3)
def mse_step(g, opt, key, batch_size):
    c, x = make_bands(key, batch_size)  # a fresh batch at every step
    opt.update(g, nnx.grad(lambda g_: ((g_(c) - x) ** 2).mean())(g))

def train_mse(g, key, num_steps=2000, batch_size=128, lr=0.01):
    schedule = optax.cosine_decay_schedule(lr, num_steps)
    opt = nnx.Optimizer(g, optax.adam(schedule), wrt=nnx.Param)
    for k in jax.random.split(key, num_steps):
        mse_step(g, opt, k, batch_size)

train_mse(g_mse, key_mse)
```

The second student plays the log-loss game, with the losses `d2l.gan_loss_D` and `d2l.gan_loss_G` of :numref:`sec_gan_convergence`. At each step the critic takes one logistic-regression step on real pairs $(x, c)$ against predicted pairs $(\hat x, c)$, and then the student takes one step on $-\log \sigma\big(D(\hat x, c)\big)$ against the updated critic. The prediction enters the game with the data's observation noise, $\hat x = g(c) + 0.05\,\xi'$ with a fresh $\xi' \sim \mathcal{N}(0, 1)$, so the two distributions differ in where they put their mass, not in how smooth they are. A game moves more slowly than a regression, so this student gets 8,000 steps; the squared-error student is at its optimum after 2,000, and more steps would not move it.

```{.python .input #adversarial-losses-two-bands-2}
%%tab pytorch
def train_adversarial(g, D, num_steps=8000, batch_size=128):
    opt_G = torch.optim.Adam(g.parameters(), lr=1e-3)
    opt_D = torch.optim.Adam(D.parameters(), lr=2e-3)
    for _ in range(num_steps):
        c, x = make_bands(batch_size)
        x_hat = g(c) + 0.05 * torch.randn(batch_size, 1)  # the data's noise
        real, fake = torch.cat([c, x], 1), torch.cat([c, x_hat], 1)
        opt_D.zero_grad()
        d2l.gan_loss_D(D, real, fake.detach()).backward()
        opt_D.step()
        opt_G.zero_grad()
        d2l.gan_loss_G(D, real, fake).backward()
        opt_G.step()

train_adversarial(g_adv, D)
```

```{.python .input #adversarial-losses-two-bands-2}
%%tab jax
@nnx.jit(static_argnums=5)
def adversarial_step(g, D, opt_G, opt_D, key, batch_size):
    k_data, k_xi = jax.random.split(key)
    c, x = make_bands(k_data, batch_size)
    xi = jax.random.normal(k_xi, (batch_size, 1))  # the data's noise
    real = jnp.concatenate([c, x], 1)
    fake = lambda g_: jnp.concatenate([c, g_(c) + 0.05 * xi], 1)
    loss_D = lambda D_: d2l.gan_loss_D(D_, real, fake(g))
    opt_D.update(D, nnx.grad(loss_D)(D))
    loss_G = lambda g_: d2l.gan_loss_G(D, real, fake(g_))
    opt_G.update(g, nnx.grad(loss_G)(g))

def train_adversarial(g, D, key, num_steps=8000, batch_size=128):
    opt_G = nnx.Optimizer(g, optax.adam(1e-3), wrt=nnx.Param)
    opt_D = nnx.Optimizer(D, optax.adam(2e-3), wrt=nnx.Param)
    for k in jax.random.split(key, num_steps):
        adversarial_step(g, D, opt_G, opt_D, k, batch_size)

train_adversarial(g_adv, D, key_adv)
```

Three numbers summarize each student on a grid of 201 inputs: the mean distance from its curve to the nearer band, the mean distance to the conditional mean $x = c$, and the fraction of inputs at which the curve is more than 0.25 from both bands.

```{.python .input #adversarial-losses-the-capacity-argument-2}
%%tab pytorch
cs = torch.linspace(-1, 1, 201).reshape(-1, 1)
with torch.no_grad():
    curves = {'squared error': g_mse(cs), 'adversarial': g_adv(cs)}
for name, x_hat in curves.items():
    to_band = torch.minimum((x_hat - cs - 1).abs(), (x_hat - cs + 1).abs())
    print(f'{name}: {to_band.mean():.2f} from the nearer band, '
          f'{(x_hat - cs).abs().mean():.2f} from x = c, '
          f'{(to_band > 0.25).float().mean():.0%} off the bands')
```

```{.python .input #adversarial-losses-the-capacity-argument-2}
%%tab jax
cs = jnp.linspace(-1, 1, 201).reshape(-1, 1)
curves = {'squared error': g_mse(cs), 'adversarial': g_adv(cs)}
for name, x_hat in curves.items():
    to_band = jnp.minimum(jnp.abs(x_hat - cs - 1), jnp.abs(x_hat - cs + 1))
    print(f'{name}: {to_band.mean():.2f} from the nearer band, '
          f'{jnp.abs(x_hat - cs).mean():.2f} from x = c, '
          f'{(to_band > 0.25).mean():.0%} off the bands')
```

Plotting both learned curves in `curves` over the training sample
`(c_data, x_data)` alongside the dashed conditional mean $x = c$ makes the
difference between regression and adversarial training immediately visible:

```{.python .input #adversarial-losses-the-capacity-argument-3}
%%tab pytorch, jax
d2l.plot([c_data, cs, cs, cs],
         [x_data, curves['squared error'], curves['adversarial'], cs],
         '$c$', '$x$', ['data', 'squared-error student',
                        'adversarial student', '$x = c$'],
         ylim=(-2.2, 3.6), fmts=('C0.', 'C2-', 'C1-', 'k--'),
         figsize=(5.5, 3.2))
```

As predicted, the squared-error model `g_mse` tracks the conditional mean $\mu(c) = c$ down the empty corridor (about one unit from either band, with $100\%$ of predictions off the bands). The adversarial model `g_adv` commits to valid data bands (a few hundredths from the nearest band on average, with at most $2\%$ of points crossing the corridor).

Notice the fundamental perception–distortion trade-off :cite:`Blau.Michaeli.2017`: even an ideal stochastic sampler $\hat x \sim p(x \mid c)$ has expected squared error $2\,\operatorname{Var}(x \mid c)$—twice the MSE (or $3\text{ dB}$ lower PSNR) of the blurry conditional mean $\mu(c)$ (Exercise :exref:`ex_adversarial-losses-perfect-sampler`). Furthermore, because a deterministic function $g(c)$ can output only one value per input $c$, covering *both* bands simultaneously requires passing a stochastic latent variable $z$ alongside $c$: $\hat x = g(c, z)$ (Exercise :exref:`ex_adversarial-losses-latent-bit`).

## Discriminators in Diffusion Distillation

### Denoisers as Conditional Means

Diffusion and flow-matching models (:numref:`chap_diffusion`) train a denoiser network by squared error to predict clean data $x_0$ from a noisy input $x_\tau = x_0 + \tau\xi$ ($\xi \sim \mathcal{N}(0, I)$). By :eqref:`eq_beyond_condmean`, the optimal denoiser computes the posterior conditional mean $\mathbb{E}[x_0 \mid x_\tau]$ (:numref:`sec_diffusion-denoising`). At large noise levels $\tau$, many clean images are consistent with $x_\tau$, so $\mathbb{E}[x_0 \mid x_\tau]$ is a blurry average; multi-step samplers avoid returning this average by taking many small steps from high noise down to $\tau \to 0$.

When *distilling* a multi-step diffusion teacher into a fast one-step student generator (:numref:`sec_diffusion-few-step`), one common approach adds noise $\tau\xi$ to the student's output $\hat x$ and regresses $\hat x$ toward the frozen teacher's denoised estimate $\mathbb{E}[x_0 \mid \hat x + \tau\xi]$. At high noise $\tau$, however, the teacher's denoised target is itself a conditional mean—pulling the one-step student back toward the blurry average unless an adversarial critic is included.

### Closed-Form Distillation on the Two-Band Benchmark
:label:`subsec_gan_beyond_distillation`

On the two-band problem :eqref:`eq_beyond_twoband`, the exact Bayes-optimal denoising teacher at noise level $\tau$ has a closed form. Given a noisy observation $\tilde x = x + \tau\xi$ of a point on band $b \in \{-1, +1\}$, $\tilde x \mid c, b \sim \mathcal{N}(c + b, \tau^2 + 0.05^2)$. By Bayes' rule, the posterior log-odds of $b = +1$ versus $b = -1$ are $2(\tilde x - c)/(\tau^2 + 0.05^2)$, giving the posterior mean band center:

$$
\mathbb{E}[c + b \mid \tilde x, c] \;=\; c + \tanh\Big(\frac{\tilde x - c}{\tau^2 + 0.05^2}\Big).
$$
:eqlabel:`eq_beyond_teacher`

We now equip each student $g(c, z)$ with a 1D Gaussian latent $z \sim \mathcal{N}(0, 1)$ so it has the capacity to cover both bands, and compare three training objectives at teacher noise level $\tau = 2$:

1. *Regression only*: minimizing $\|\hat x - \operatorname{stop\_grad}(\text{teacher}(\hat x + \tau\xi, c, \tau))\|^2$. Because the slope of :eqref:`eq_beyond_teacher` in $\tilde x$ is at most $(\tau^2 + 0.05^2)^{-1} \approx \tfrac14 < 1$ at $\tau = 2$, the fixed-point equation $u = \mathbb{E}_\xi[\tanh((u + \sqrt{\tau^2 + 0.05^2}\,\xi)/(\tau^2 + 0.05^2))]$ for the offset $u = g(c, z) - c$ has the unique contraction fixed point $u = 0$ (the empty corridor).
2. *Critic only*: training $g(c, z)$ against a conditional discriminator $D(x, c)$ with `d2l.gan_loss_D` and `d2l.gan_loss_G`.
3. *Critic + regression*: summing the adversarial loss and the teacher regression loss with equal weight (as in Adversarial Diffusion Distillation, ADD, :citet:`Sauer.Lorenz.Blattmann.ea.2023`).

```{.python .input #adversarial-losses-distillation-on-two-bands-1}
%%tab pytorch
def teacher(x_tilde, c, tau):
    """Posterior mean of the band center c + b given x_tilde = x + tau xi."""
    return c + torch.tanh((x_tilde - c) / (tau ** 2 + 0.05 ** 2))

def train_student(g, D, lam, tau=2.0, num_steps=4000, batch_size=128):
    """Train g(c, z) on the critic's loss if D is given, plus lam times the
    squared error toward the teacher's estimate at noise level tau."""
    opt_G = torch.optim.Adam(g.parameters(), lr=1e-3)
    if D is not None:
        opt_D = torch.optim.Adam(D.parameters(), lr=2e-3)
    for _ in range(num_steps):
        c, x = make_bands(batch_size)
        z = torch.randn(batch_size, 1)  # the student's latent
        x_hat = g(torch.cat([c, z], 1)) + 0.05 * torch.randn(batch_size, 1)
        real, fake = torch.cat([c, x], 1), torch.cat([c, x_hat], 1)
        loss_G = 0
        if D is not None:
            opt_D.zero_grad()
            d2l.gan_loss_D(D, real, fake.detach()).backward()
            opt_D.step()
            loss_G = d2l.gan_loss_G(D, real, fake)
        with torch.no_grad():  # the teacher's estimate is a fixed target
            target = teacher(x_hat + tau * torch.randn(batch_size, 1), c, tau)
        loss_G = loss_G + lam * ((x_hat - target) ** 2).mean()
        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()
```

```{.python .input #adversarial-losses-distillation-on-two-bands-1}
%%tab jax
def teacher(x_tilde, c, tau):
    """Posterior mean of the band center c + b given x_tilde = x + tau xi."""
    return c + jnp.tanh((x_tilde - c) / (tau ** 2 + 0.05 ** 2))

@nnx.jit(static_argnums=7)
def student_step(g, D, opt_G, opt_D, key, lam, tau, batch_size):
    k_data, k_z, k_xi, k_tau = jax.random.split(key, 4)
    c, x = make_bands(k_data, batch_size)
    z = jax.random.normal(k_z, (batch_size, 1))  # the student's latent
    xi = jax.random.normal(k_xi, (batch_size, 1))  # the data's noise
    sample = lambda g_: g_(jnp.concatenate([c, z], 1)) + 0.05 * xi
    real = jnp.concatenate([c, x], 1)
    fake = lambda g_: jnp.concatenate([c, sample(g_)], 1)
    if D is not None:
        loss_D = lambda D_: d2l.gan_loss_D(D_, real, fake(g))
        opt_D.update(D, nnx.grad(loss_D)(D))
    noise = tau * jax.random.normal(k_tau, (batch_size, 1))
    def loss_G(g_):
        x_hat = sample(g_)  # the teacher's estimate is a fixed target
        target = jax.lax.stop_gradient(teacher(x_hat + noise, c, tau))
        loss = lam * ((x_hat - target) ** 2).mean()
        if D is not None:
            loss += d2l.gan_loss_G(D, real, fake(g_))
        return loss
    opt_G.update(g, nnx.grad(loss_G)(g))

def train_student(g, D, lam, key, tau=2.0, num_steps=4000, batch_size=128):
    """Train g(c, z) on the critic's loss if D is given, plus lam times the
    squared error toward the teacher's estimate at noise level tau."""
    opt_G = nnx.Optimizer(g, optax.adam(1e-3), wrt=nnx.Param)
    opt_D = (None if D is None
             else nnx.Optimizer(D, optax.adam(2e-3), wrt=nnx.Param))
    for k in jax.random.split(key, num_steps):
        student_step(g, D, opt_G, opt_D, k, lam, tau, batch_size)
```

A student is now a distribution of outputs at each $c$. The function `band_shares` evaluates it at the 201 inputs of the first experiment and at 64 evenly spaced quantiles of the latent, and returns the mean distance to the nearer band and the shares of outputs within 0.25 of the upper and of the lower band.

```{.python .input #adversarial-losses-distillation-on-two-bands-2}
%%tab pytorch
def band_shares(g, cs):
    """Mean distance of g(c, z) to the nearer band, and its shares on the
    upper and the lower band, over the inputs cs and 64 quantiles z."""
    zs = torch.special.ndtri((torch.arange(64) + 0.5) / 64)
    c, z = (v.reshape(-1, 1) for v in torch.meshgrid(
        cs.flatten(), zs, indexing='ij'))
    with torch.no_grad():
        u = g(torch.cat([c, z], 1)) - c  # offset from the corridor
    to_band = torch.minimum((u - 1).abs(), (u + 1).abs())
    return (to_band.mean(), ((u - 1).abs() < 0.25).float().mean(),
            ((u + 1).abs() < 0.25).float().mean())
```

```{.python .input #adversarial-losses-distillation-on-two-bands-2}
%%tab jax
def band_shares(g, cs):
    """Mean distance of g(c, z) to the nearer band, and its shares on the
    upper and the lower band, over the inputs cs and 64 quantiles z."""
    zs = jax.scipy.special.ndtri((jnp.arange(64) + 0.5) / 64)
    c, z = (v.reshape(-1, 1) for v in jnp.meshgrid(
        cs.flatten(), zs, indexing='ij'))
    u = g(jnp.concatenate([c, z], 1)) - c  # offset from the corridor
    to_band = jnp.minimum(jnp.abs(u - 1), jnp.abs(u + 1))
    return (to_band.mean(), (jnp.abs(u - 1) < 0.25).mean(),
            (jnp.abs(u + 1) < 0.25).mean())
```

Each student trains from three seeds. Within a seed, the three students start from the same weights and see the same data.

```{.python .input #adversarial-losses-distillation-on-two-bands-3}
%%tab pytorch
students = {'regression': (False, 1.0), 'critic': (True, 0.0),
            'critic + regression': (True, 1.0)}
for name, (has_critic, lam) in students.items():
    for seed in range(3):
        torch.manual_seed(seed)
        g = d2l.make_mlp([2, 32, 32, 1], nn.Tanh())
        D = d2l.make_mlp([2, 32, 32, 1], nn.LeakyReLU(0.2))
        train_student(g, D if has_critic else None, lam)
        dist, upper, lower = band_shares(g, cs)
        print(f'{name}, seed {seed}: {dist:.2f} from the nearer band, '
              f'{upper:.0%} upper, {lower:.0%} lower, '
              f'{1 - upper - lower:.0%} off the bands')
```

```{.python .input #adversarial-losses-distillation-on-two-bands-3}
%%tab jax
students = {'regression': (False, 1.0), 'critic': (True, 0.0),
            'critic + regression': (True, 1.0)}
for name, (has_critic, lam) in students.items():
    for seed in range(3):
        rngs = nnx.Rngs(seed)
        g = d2l.make_mlp([2, 32, 32, 1], nnx.tanh, rngs=rngs)
        D = d2l.make_mlp([2, 32, 32, 1], lambda h: nnx.leaky_relu(h, 0.2),
                         rngs=rngs)
        train_student(g, D if has_critic else None, lam, rngs())
        dist, upper, lower = band_shares(g, cs)
        print(f'{name}, seed {seed}: {dist:.2f} from the nearer band, '
              f'{upper:.0%} upper, {lower:.0%} lower, '
              f'{1 - upper - lower:.0%} off the bands')
```

The three-seed results confirm the fixed-point analysis:

- *Regression onto the high-noise teacher* collapses $100\%$ of outputs into the empty corridor (about one unit from either band), ignoring the latent $z$.
- *Critic only* and *Critic + regression* both place $98\text{--}99\%$ of outputs directly on the two data bands (mean distance of a few hundredths, always below $0.1$), splitting probability mass evenly ($\sim 49\% / 49\%$) between the upper and lower bands via the latent $z$. Because the bands have narrow width $\sigma = 0.05$, the optimal critic's restoring slope ($\delta / 0.05^2 = 400\delta$) easily dominates the gentle inward pull of the high-noise teacher.

### Distillation, Autoencoders, and Vocoders at Scale

This combination of a regression or perceptual anchor with an adversarial discriminator appears across three major classes of modern generative systems:

1. **Adversarial Diffusion Distillation (ADD and DMD2):** ADD :cite:`Sauer.Lorenz.Blattmann.ea.2023` trains a one-step image generator by combining a denoising teacher regression loss with a discriminator built on frozen DINOv2 features, finding the same ordering as our experiment above (teacher regression alone produces blurry images, while adding the discriminator yields sharp one-step samples). Distribution Matching Distillation (DMD, :citet:`Yin.Gharbi.Zhang.ea.2024`) uses the difference between a frozen teacher's score $\nabla_x \log p_\tau$ and a fake-distribution diffusion model's score $\nabla_x \log q_\tau$ to descend $D_{\textrm{KL}}(q_\tau \,\|\, p_\tau)$. We have met this difference before: $\nabla_x \log p_\tau - \nabla_x \log q_\tau = \nabla_x \lambda_\tau$ is the input gradient of the optimal critic $\lambda_\tau = \log(p_\tau/q_\tau)$ between the noised distributions. DMD's generator update is therefore the unweighted ($w = 1$) reverse-KL update of Exercise :exref:`ex_gan-generator-weight-divergences`, with the critic's slope supplied by two score networks instead of a discriminator. Since both distributions are noised, their supports overlap at every noise level $\tau > 0$, which is the instance noise of :numref:`subsec_gan_disjoint_supports`. DMD2 :cite:`Yin.Gharbi.Park.ea.2024` adds an adversarial discriminator against real images so the one-step student can exceed the teacher's quality.
2. **Latent Image Autoencoders (Tokenizers):** Latent diffusion models (:numref:`sec_diffusion-latent`) compress images into a lower-dimensional latent space using an autoencoder (such as VQGAN or the Stable Diffusion VAE, :citet:`Esser.Rombach.Ommer.2021,Rombach.Blattmann.Lorenz.ea.2022`). Because the compressed latent code leaves fine-grained local texture underdetermined, training the decoder with pixelwise $L_1/L_2$ and perceptual (LPIPS) losses is augmented with a patch-level discriminator (PatchGAN) to ensure crisp high-frequency textures.
3. **Neural Speech Vocoders:** Text-to-speech pipelines typically predict a mel-spectrogram (which specifies frame-level frequency magnitudes but discards phase) and use a vocoder such as HiFi-GAN :cite:`Kong.Kim.Bae.2020` to invert the spectrogram into an audio waveform. Because the expected value of a sinusoid $\sin(\omega n + \vartheta)$ over an unknown phase $\vartheta \sim \textrm{Uniform}[-a, a]$ shrinks in amplitude by $\sin(a)/a$ (vanishing completely for $\vartheta \sim \textrm{Uniform}[-\pi, \pi]$, Exercise :exref:`ex_adversarial-losses-phase`), waveform regression collapses to silence; HiFi-GAN instead pairs a mel-spectrogram $L_1$ loss and a feature-matching loss on discriminator activations with multi-period and multi-scale waveform discriminators.

## Comparing the Adversarial Objectives of This Chapter
:label:`subsec_gan_beyond_verdict`

:numref:`tab_gan_games` synthesizes the six adversarial formulations studied across this chapter along three dimensions: what the optimal critic estimates, how it scores inputs, and how the critic function class is regularized.

:Summary of the adversarial objectives analyzed in this chapter.
:label:`tab_gan_games`

| objective | optimal critic estimate | scoring rule | critic constraint / regularization | behavior on disjoint supports (at optimal critic) |
|:--|:--|:--|:--|:--|
| log loss / $f$-GAN (:numref:`sec_basic_gan`, :numref:`sec_gan_objectives`) | $\log(p/q)$, or $f'(p/q)$ | single-sample classification | unconstrained | saturates |
| hinge (:numref:`sec_gan_objectives`) | $\operatorname{sign}\log(p/q)$ | margin classification | unconstrained (or spectral norm) | saturates (linear up to $2/\kappa$ with $\kappa$-Lipschitz critic) |
| MMD (:numref:`sec_gan_objectives`) | kernel-smoothed $p - q$ (witness) | difference of means | RKHS unit ball | nonzero gradient within kernel bandwidth $s$ |
| Wasserstein-1 (:numref:`sec_gan_objectives`) | Kantorovich potential | difference of means | 1-Lipschitz ball (e.g., WGAN-GP) | unit slope at any separation |
| RpGAN (:numref:`sec_gan_relativistic`) | $\log(p/q) + c$ | pairwise Bradley–Terry ranking | unconstrained | saturates |
| R3GAN: RpGAN + $R_1 + R_2$ (:numref:`sec_gan_convergence`) | $\log(p/q)$ low-pass filtered below scale $2\sqrt{\gamma}$ | pairwise Bradley–Terry ranking | zero-centered input-gradient penalty on $p$ and $q$ | smooth interpolation across gaps via penalized network |

Compared to iterative diffusion and flow models (:numref:`chap_diffusion`), adversarial generators offer single-pass inference and sharp outputs on underdetermined inverse problems, balanced against four inherent challenges: (i) no explicit likelihood evaluation, (ii) coupled two-network minimax dynamics requiring tuned gradient penalties, (iii) weak intrinsic penalty for dropped modes (:numref:`subsec_gan_mode_dropping`), and (iv) potential training instabilities at large scale :cite:`Brock.Donahue.Simonyan.2019`. In the next chapter (:numref:`chap_diffusion`), we turn to diffusion and flow-matching models, which replace minimax training with multi-step iterative denoising.

## Summary

Minimizing squared error $\mathbb{E}[\|x - g(c)\|^2]$ drives a predictor toward the conditional mean $\mu(c) = \mathbb{E}[x \mid c]$ (:eqref:`eq_beyond_condmean`), which averages across distinct modes whenever the target $x$ is ambiguous given $c$. Pairing the predictor with a conditional discriminator $D(x, c)$ that estimates $\log(p(x, c)/q(x, c))$ penalizes predictions in low-density corridors between modes and forces outputs onto the data manifold, trading up to a twofold increase in MSE ($3\text{ dB}$ in PSNR) for perceptual realism :cite:`Blau.Michaeli.2017`. Because a diffusion denoiser likewise computes the posterior mean $\mathbb{E}[x_0 \mid x_\tau]$, distilling a one-step generator by regressing directly onto a high-noise denoiser contracts predictions toward the blurry average unless an adversarial critic is included. For the same reason, adversarial discriminators remain essential components of latent image autoencoders, neural speech vocoders, and one-step distilled diffusion generators.

## Exercises

1. **Perception–distortion trade-off of an ideal sampler.** Let $\hat x \sim p(x \mid c)$ be drawn independently of $x$ given $c$, so that the predictor matches the true conditional distribution $p(x \mid c)$ exactly :cite:`Blau.Michaeli.2017`.
    1. Use :eqref:`eq_beyond_condmean` to prove that $\mathbb{E}\big[(x - \hat x)^2\big] = 2\, \mathbb{E}\big[(x - \mu(c))^2\big]$. How many decibels of PSNR ($10 \log_{10}(x_{\max}^2 / \text{MSE})$) does an exact sampler lose relative to the conditional mean $\mu(c)$?
    1. On the two-band distribution :eqref:`eq_beyond_twoband`, compare the expected squared error of the exact sampler against $\mu(c) = c$ ($1.0025$) and the single-band predictor $g(c) = c + 1$ ($2.0025$). Which of the three cannot be distinguished from the true data by an optimal critic $D(x, c)$?
    1. Let $x \sim \mathcal{N}(0, 1)$ and $c = x + \tau\xi$ with independent $\xi \sim \mathcal{N}(0, 1)$. Show that the deterministic linear predictor $\hat x = c / \sqrt{1 + \tau^2}$ matches the marginal distribution $\mathcal{N}(0, 1)$ of $x$, and compute the ratio of its MSE to that of the conditional mean $\mathbb{E}[x \mid c]$ for $\tau \in \{0.1, 1, 10\}$.
   :exlabel:`ex_adversarial-losses-perfect-sampler`

1. [code] **Conditional mean versus conditional median ($L_2$ vs. $L_1$).** Minimizing absolute error $\mathbb{E}[|x - g(c)|]$ yields a conditional median $x_{\textrm{med}}(c)$ satisfying $P(x \leq x_{\textrm{med}}(c) \mid c) = \tfrac12$.
    1. If the offset $b$ in :eqref:`eq_beyond_twoband` takes values $(-1, 0, +1)$ with probabilities $(\tfrac13, \tfrac13, \tfrac13)$ or $(\tfrac14, \tfrac14, \tfrac12)$, find the conditional mean and conditional median in each case and state whether each lies on a data band.
    1. On the two-band problem with asymmetric prior $P(b = +1) = r$, compute the conditional mean as a function of $r$, show that for $r = \tfrac34$ the conditional median lies on the upper band, and characterize the $L_1$ risk across the corridor $g(c) - c \in (-1, 1)$ when $r = \tfrac12$.
    1. Train a copy of `g_mse` using $L_1$ loss on the settings from (a) and (b) and compare the learned curves with your analytical predictions.
   :exlabel:`ex_adversarial-losses-mean-and-median`

1. [code] **Adding a binary latent variable.** ● Augment the student in :numref:`subsec_gan_beyond_capacity` with an independent uniform binary latent $z \in \{-1, +1\}$ so that $\hat x = g(c, z)$, while the critic still scores $(x, c)$ pairs.
    1. Train one student with squared error and one adversarially, and check which bands $z = -1$ and $z = +1$ map to. Why does the squared-error student ignore $z$?
    1. Implicit Maximum Likelihood Estimation (IMLE, :citet:`Li.Malik.2018`) minimizes $\mathbb{E}_{(c, x)}[\min_{z \in \{-1, +1\}} (g(c, z) - x)^2]$ without a critic. Train a student with this loss and explain why taking the minimum over $z$ for each data point forces both latent states to cover distinct bands.
   :exlabel:`ex_adversarial-losses-latent-bit`

1. **Direct Discriminative Optimization (DDO).** DDO :cite:`Zheng.Chen.Chen.ea.2025` fine-tunes a likelihood model $q_\theta$ against a frozen reference copy $q_{\textrm{ref}}$ by parameterizing the critic directly as the log likelihood ratio $D_\theta(x) = \log(q_\theta(x) / q_{\textrm{ref}}(x))$ and minimizing the logistic discriminator loss $J(\theta) = -\mathbb{E}_{x \sim p}[\log \sigma(D_\theta(x))] - \mathbb{E}_{x \sim q_{\textrm{ref}}}[\log \sigma(-D_\theta(x))]$ starting from $\theta = \theta_{\textrm{ref}}$.
    1. Show that if $q_\theta = p$, then $D_\theta(x)$ equals the optimal critic $D^\star(x)$ between $p$ and $q_{\textrm{ref}}$.
    1. Using $\mathbb{E}_{x \sim q_\theta}[\nabla_\theta \log q_\theta(x)] = 0$, show that at initialization ($\theta = \theta_{\textrm{ref}}$, where $D_\theta(x) = 0$), the gradient $\nabla_\theta J(\theta)$ equals $\tfrac12$ times the negative log-likelihood gradient $-\nabla_\theta \mathbb{E}_{x \sim p}[\log q_\theta(x)]$.
   :exlabel:`ex_adversarial-losses-ddo-plays-log-loss`

1. **Phase ambiguity and waveform averaging in speech vocoders.** Model a single audio frame of length $N$ as a sinusoid $x_n = \sin(\omega n + \vartheta)$ for $n = 0, \ldots, N - 1$, where the frequency $\omega$ is known from the spectrogram and the phase $\vartheta \sim \textrm{Uniform}[-a, a]$ ($0 < a \leq \pi$) is unobserved.
    1. Show that the conditional mean waveform $\mathbb{E}_\vartheta[x_n] = \frac{\sin a}{a}\sin(\omega n)$ has amplitude attenuated by $\frac{\sin a}{a}$ (vanishing to silence when $a = \pi$).
    1. For $\omega = 2\pi k / N$ ($0 < k < N/2$), show that the Discrete Fourier Transform magnitudes $|\hat x_m|$ are independent of $\vartheta$.
    1. If two adjacent frames are generated with independent phases $\vartheta_1 \neq \vartheta_2$, frame-level DFT magnitude losses are zero on each frame individually. What artifact occurs at the frame boundary, and why does a time-domain waveform discriminator detect it?
   :exlabel:`ex_adversarial-losses-phase`

[Discussions](https://d2l.discourse.group/t/36814)

<!-- slides -->

::: {.slide}
::: {.cover}
[Dive into Deep Learning · §16.7]{.kicker}

Adversarial Losses Beyond Standalone GANs<br>**conditional-mean blurring · perception–distortion trade-off · diffusion distillation (ADD & DMD2) · VQGAN patch critics · HiFi-GAN speech vocoders**
:::
:::

::: {.slide title="Why Pixelwise Regression Blurs Multimodal Targets"}
[Motivation]{.kicker}

::: {.cols .vc}
::: {.col}
- **The averaging failure of pointwise losses:** On tasks with multiple plausible outputs $x$ per condition $c$ (super-resolution, inpainting, spectrogram inversion), minimizing pixelwise $L_2$ or $L_1$ forces a predictor $g(c)$ to output the **conditional mean** or median—a blur belonging to none of the modes.
- **Where adversarial critics remain essential today:** (1) One-step diffusion distillation (ADD / SDXL-Turbo, DMD2), (2) latent image autoencoders (VQGAN, Stable Diffusion VAE), and (3) neural speech vocoders (HiFi-GAN).
:::

::: {.col .fig .big}
@fig:ch16-gan-multimodal-blur
:::
:::
:::

::: {.slide title="The Conditional-Mean Orthogonal Decomposition"}
[Why L₂ Averages Ambiguity]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}\big[(x - g(c))^2\big] = \underbrace{\mathbb{E}\big[(x - \mu(c))^2\big]}_{\mathrm{Var}(x \mid c)} + \underbrace{\mathbb{E}\big[(\mu(c) - g(c))^2\big]}_{\text{minimized at } g(c) = \mu(c)}$$

- **Unique $L_2$ minimizer:** Adding and subtracting $\mu(c) = \mathbb{E}[x \mid c]$ eliminates the cross term by iterated expectation, proving $g^\star(c) = \mu(c)$ pixel by pixel.
- **Multimodal void:** When $p(x \mid c)$ is bimodal, both the conditional mean $\mu(c)$ ($L_2$) and conditional median ($L_1$) sit in the empty low-density void between modes.
:::

::: {.col .fig}
@fig:ch16-gan-orth-decomp
:::
:::
:::

::: {.slide title="The Bimodal Two-Band Benchmark"}
[Minimal Analytical Benchmark]{.kicker}

::: {.cols .vc}
::: {.col}
$$x = c + b + 0.05\,\xi, \qquad b \in \{-1, +1\}\text{ w.p. }\tfrac{1}{2}$$

- **Conditional mean $\mu(c) = c$ (empty corridor):** Lies $20\sigma$ away from every data point, yet achieves the minimum possible MSE of **$1.0025$**.
- **On-band predictor $g(c) = c + 1$:** Always lands on valid data, yet incurs **$2.0025$** MSE ($2\times$ larger).
- **Adversarial critic $D(x, c)$:** Penalizes the empty corridor ($D^\star = \log(p/q) \ll 0$) and pushes predictions onto a valid band.
:::

::: {.col .fig}
@fig:ch16-gan-two-bands
:::
:::
:::

::: {.slide title="Training Squared-Error vs. Adversarial Predictors" only="pytorch"}
[Controlled Code Comparison]{.kicker}

We train two identical 2-hidden-layer $\tanh$ MLPs (`g_mse` and `g_adv`, $1 \to 32 \to 32 \to 1$) from the same initial weights. `g_adv` adds the data's observation noise $0.05\,\xi'$ and trains against a leaky-ReLU critic $D(x, c)$ via `gan_loss_D` and `gan_loss_G`:

::: {.cols}
::: {.col}
@adversarial-losses-two-bands-1
:::

::: {.col}
@adversarial-losses-two-bands-2
:::
:::
:::

::: {.slide title="Training Squared-Error vs. Adversarial Predictors" only="jax"}
[Controlled Code Comparison]{.kicker}

We train two identical 2-hidden-layer $\tanh$ MLPs (`g_mse` and `g_adv`, $1 \to 32 \to 32 \to 1$) from the same initial weights. `g_adv` adds the data's observation noise $0.05\,\xi'$ and trains against a leaky-ReLU critic $D(x, c)$ via `gan_loss_D` and `gan_loss_G`:

@adversarial-losses-two-bands-1
:::

::: {.slide title="Training Squared-Error vs. Adversarial Predictors (cont.)" only="jax"}
[Controlled Code Comparison]{.kicker}

`g_adv` instead trains against the critic, one `gan_loss_D` and one `gan_loss_G` step per batch:

@adversarial-losses-two-bands-2
:::

::: {.slide title="Two-Band Results: Corridor vs. Mode Commitment"}
[Empirical Verification]{.kicker}

Evaluating both trained predictors across $201$ inputs $c \in [-1, 1]$ confirms our analytical prediction:

::: {.cols .vc}
::: {.col}
@!adversarial-losses-the-capacity-argument-2

- **Squared-error student (`g_mse`):** Tracks the conditional mean $x = c$ down the empty corridor (about one unit from either band; $100\%$ of predictions off the bands).
- **Adversarial student (`g_adv`):** Commits to a valid data band (a few hundredths from the nearer band on average; at most $2\%$ of points off the bands where the continuous curve switches branches).
:::

::: {.col .fig}
@!adversarial-losses-the-capacity-argument-3
:::
:::
:::

::: {.slide title="The Perception–Distortion Trade-Off"}
[Blau and Michaeli Theorem]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}\big[(x - \hat{x})^2\big] = \mathbb{E}\big[(x - \mu(c))^2\big] + \mathbb{E}\big[(\mu(c) - \hat{x})^2\big] = 2\,\mathbb{E}\big[(x - \mu(c))^2\big]$$

- **Exact factor of 2 in MSE:** For an ideal stochastic sampler $\hat{x} \sim p(x \mid c)$ matching the conditional data distribution perfectly ($q = p$), expected squared error is **twice** that of the conditional mean $\mu(c)$ ($2.0050$ vs. $1.0025$).
- **3 dB PSNR penalty (Blau & Michaeli, 2018):** Perfect perceptual realism inherently costs up to $10\log_{10} 2 \approx 3.01\text{ dB}$ in PSNR compared to the blurry conditional mean.
:::

::: {.col .fig}
@fig:ch16-gan-perception-distortion
:::
:::
:::

::: {.slide title="Target Ambiguity vs. Model Capacity"}
[Why a Latent Variable Is Required]{.kicker}

::: {.cols .vc}
::: {.col}
- **1. Target ambiguity (aleatoric uncertainty):** Because $c$ carries zero bits of information about the band sign $b \in \{-1, +1\}$, *no* deterministic function $g(c)$, regardless of width or depth, can cover both bands at the same $c$.
- **2. Model capacity limit:** Even when a deterministic map exists (like a diffusion ODE), a small 1-step network averages across sharp jumps under $L_2$ regression.
- **Stochastic latent routing:** Equipping the student with $z \sim \mathcal{N}(0, 1)$ lets $\hat{x} = g(c, z)$ route $z > 0$ and $z < 0$ to both data bands evenly.
:::

::: {.col .fig .big}
@fig:ch16-gan-latent-routing
:::
:::
:::

::: {.slide title="Diffusion Denoisers and One-Step Distillation"}
[Why High-Noise Teachers Blur]{.kicker}

::: {.cols .vc}
::: {.col}
$$D_\tau^\star(x_\tau) = \arg\min_g \mathbb{E}\big[\|x_0 - g(x_\tau)\|^2\big] = \mathbb{E}\big[x_0 \mid x_\tau\big]$$

- **Low noise ($\tau \ll 1$) vs. high noise ($\tau \gg 1$):** At low noise, $x_\tau = x_0 + \tau\xi$ fixes $x_0$ so $\mathbb{E}[x_0 \mid x_\tau]$ is sharp; at high noise, many clean images match $x_\tau$ so $\mathbb{E}[x_0 \mid x_\tau]$ is a blurry average.
- **One-step distillation hazard:** Regressing a 1-step student $\hat{x} = g(c, z)$ directly onto a frozen high-noise teacher $\mathbb{E}[x_0 \mid \hat{x} + \tau\xi]$ pulls the student straight toward the conditional mean.
:::

::: {.col .fig}
@fig:ch16-gan-high-noise-blur
:::
:::
:::

::: {.slide title="Closed-Form Denoising Teacher on the Two Bands"}
[Analytical Distillation Model]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}\big[c + b \mid \tilde{x}, c\big] = c + \tanh\!\left(\frac{\tilde{x} - c}{\tau^2 + 0.05^2}\right)$$

- **Exact posterior teacher:** By Bayes' rule on $\tilde{x} = x + \tau\xi$, the optimal denoiser applies $\tanh$ with slope $(\tau^2 + 0.05^2)^{-1}$.
- **Contraction to the empty corridor at $\tau = 2$:** Since slope $\le (4.0025)^{-1} \approx 0.25 < 1$ everywhere, the fixed-point equation $u = \mathbb{E}_\xi[\tanh(\frac{u + \sqrt{\tau^2 + 0.05^2}\xi}{\tau^2 + 0.05^2})]$ contracts every latent $z$ to the unique fixed point $u^\star = 0$ (the empty corridor).
:::

::: {.col .fig}
@fig:ch16-gan-tanh-contraction
:::
:::
:::

::: {.slide title="Implementing One-Step Distillation with a Critic" only="pytorch"}
[Code Experiment]{.kicker}

`train_student` trains a stochastic student $g(c, z)$ ($z \sim \mathcal{N}(0, 1)$) using either the frozen denoising teacher regression loss ($\tau = 2.0$), an adversarial critic $D(x, c)$, or both combined (`critic + regression`):

@-adversarial-losses-distillation-on-two-bands-1[1-10]
:::

::: {.slide title="Implementing One-Step Distillation with a Critic (cont.)" only="pytorch"}
[Code Experiment]{.kicker}

Each step updates the critic (if any), then moves the student toward the critic and toward the teacher's fixed estimate:

@adversarial-losses-distillation-on-two-bands-1[11-]
:::

::: {.slide title="Implementing One-Step Distillation with a Critic" only="jax"}
[Code Experiment]{.kicker}

`train_student` trains a stochastic student $g(c, z)$ ($z \sim \mathcal{N}(0, 1)$) using either the frozen denoising teacher regression loss ($\tau = 2.0$), an adversarial critic $D(x, c)$, or both combined (`critic + regression`):

@-adversarial-losses-distillation-on-two-bands-1[1-16]
:::

::: {.slide title="Implementing One-Step Distillation with a Critic (cont.)" only="jax"}
[Code Experiment]{.kicker}

The student's loss adds the critic's term to $\lambda$ times the squared error toward the teacher's fixed estimate:

@adversarial-losses-distillation-on-two-bands-1[17-]
:::

::: {.slide title="Distillation Results: Critic Restores Both Bands"}
[Three-Seed Comparison]{.kicker}

Evaluating all three student configurations across $201$ inputs $c$ and $64$ latent quantiles $z$ over three random seeds confirms our fixed-point analysis:

@!adversarial-losses-distillation-on-two-bands-3

- **1. Teacher regression only:** Collapses $100\%$ of outputs into the empty corridor (about one unit from either band), ignoring $z$.
- **2. Critic only & Critic + regression:** Both place **$98\text{--}99\%$** of outputs directly on the two data bands (a few hundredths away), splitting probability mass evenly ($\sim 49\% / 49\%$) between the upper and lower bands via $z$. Because the band width is $\sigma = 0.05$, the critic's sharp restoring force ($\sim \delta / 0.05^2 = 400\delta$) easily dominates the gentle inward pull of the high-noise teacher ($\delta \approx 0.008$).
:::

::: {.slide title="Adversarial Diffusion Distillation (ADD & DMD2)"}
[One-Step Foundation Model Distillation]{.kicker}

::: {.cols .vc}
::: {.col}
- **ADD / SDXL-Turbo (Sauer et al., 2023):** Pairs squared-error regression toward a frozen diffusion teacher's denoised estimate with a conditional discriminator built on frozen **DINOv2** vision features.
- **DMD & DMD2 (Yin et al., 2024):** DMD uses a frozen teacher score $\nabla_x \log p_\tau$ minus a fake score $\nabla_x \log q_\tau$ to move student samples along $\nabla_x \log(p_\tau/q_\tau)$ (the exact input gradient of an optimal log-loss critic), while DMD2 adds a real-image GAN critic to prevent mode dropping.
:::

::: {.col .fig .big}
@fig:ch16-gan-add-dmd2
:::
:::
:::

::: {.slide title="Latent Image Tokenizers (VQGAN) and PatchGAN"}
[Autoencoders for Latent Diffusion]{.kicker}

::: {.cols .vc}
::: {.col}
- **8× spatial latent compression:** Latent diffusion models (Stable Diffusion, Flux) compress $H \times W \times 3$ images into a compact latent grid via an autoencoder (VQGAN or KL-VAE; Esser et al., 2021; Rombach et al., 2022).
- **Three complementary decoder losses:**
  1. **Pixelwise $L_1$/$L_2$** anchors low-frequency color and layout.
  2. **Perceptual LPIPS** matches mid-level VGG feature statistics.
  3. **PatchGAN critic** scores local $N \times N$ patches to enforce crisp high-frequency textures.
:::

::: {.col .fig .big}
@fig:ch16-gan-vqgan-patchgan
:::
:::
:::

::: {.slide title="Speech Vocoders (HiFi-GAN) and Phase Ambiguity"}
[Why Waveform Regression Collapses to Silence]{.kicker}

::: {.cols .vc}
::: {.col}
$$\mathbb{E}_\vartheta\big[\sin(\omega n + \vartheta)\big] = \frac{\sin a}{a}\,\sin(\omega n) \xrightarrow{a = \pi} 0$$

- **Why time-domain $L_2$ waveform regression fails:** A mel-spectrogram specifies frequency magnitude $\omega$ but discards phase $\vartheta \sim \textrm{Uniform}[-\pi, \pi]$ ($a = \pi$), so the conditional-mean waveform is **complete silence** ($\frac{\sin\pi}{\pi} = 0$).
- **HiFi-GAN (Kong et al., 2020):** Pairs mel-spectrogram $L_1$ loss with multi-period and multi-scale time-domain waveform critics to enforce coherent phases across frames.
:::

::: {.col .fig}
@fig:ch16-gan-hifigan-phase
:::
:::
:::

::: {.slide title="Synthesis: Every Adversarial Game of Chapter §{num:chap_gans}"}
[Comparing the Six Formulations]{.kicker}

| Adversarial Formulation | Optimal Critic Estimate | Scoring Mechanism | Critic Constraint / Penalty | Disjoint-Support Gradient (at $D^\star$) |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic / $f$-GAN (§{sec_basic_gan}–§{num:sec_gan_objectives})** | $\log(p/q)$ or $f'(p/q)$ | Single-sample classification | Unconstrained | Saturates ($\mathrm{JS} = \log 2$) |
| **Hinge (§{sec_gan_objectives})** | $\operatorname{sign}(\log(p/q))$ | Margin classification | Unconstrained or spectral norm | Saturates (linear to $2/\kappa$ with $\kappa$-Lip) |
| **MMD (§{sec_gan_objectives})** | Kernel-smoothed $p - q$ | Difference of means | RKHS unit ball (no critic net) | Nonzero within kernel bandwidth $s$ |
| **Wasserstein-1 (§{sec_gan_objectives})** | Kantorovich potential | Difference of means | 1-Lipschitz ball (WGAN-GP) | Constant unit slope at all distances |
| **RpGAN (§{sec_gan_relativistic})** | $\log(p/q) + c$ | Pairwise Bradley–Terry ranking | Unconstrained | Saturates ($d_{\mathrm{Rp}} = \log 2$) |
| **R3GAN (§{sec_gan_convergence}–§{num:sec_gan_conditional})** | $\log(p/q)$ filtered above $2\sqrt{\gamma}$ | Pairwise Bradley–Terry ranking | Zero-centered $R_1 + R_2$ on $p, q$ | Smooth interpolation via damped critic |
:::

::: {.slide title="Concept Check: Blurring, Distillation, and Phase"}
[Check Your Understanding]{.kicker}

Test your understanding of conditional-mean blurring and adversarial losses in modern generative pipelines:

1. **PSNR cost of exact sampling:** Using Blau and Michaeli's identity $\mathbb{E}[\|x - \hat{x}\|^2] = 2\,\mathbb{E}[\|x - \mu(c)\|^2]$ for an ideal sampler $\hat{x} \sim p(x \mid c)$, how many decibels of Peak Signal-to-Noise Ratio ($\mathrm{PSNR} = 10\log_{10}(x_{\max}^2 / \mathrm{MSE})$) does a perfectly realistic generator lose relative to the blurry conditional mean $\mu(c)$?
2. **Partial phase uncertainty in vocoders:** In the sinusoidal audio frame model $x_n = \sin(\omega n + \vartheta)$ with phase $\vartheta \sim \textrm{Uniform}[-\pi/2, \pi/2]$, to what fraction of the true amplitude does $L_2$ waveform regression shrink the signal? What artifact occurs if a vocoder minimizes only frame-level DFT magnitude error without a time-domain waveform critic?

::: {.d2l-note .fragment .fade-up}
**Answers:**

1. Doubling the MSE reduces $\mathrm{PSNR} = 10\log_{10}(x_{\max}^2 / \mathrm{MSE})$ by $10\log_{10}(2) \approx \mathbf{3.01\text{ dB}}$.
2. At $a = \pi / 2$, the amplitude shrinks by $\frac{\sin(\pi/2)}{\pi/2} = \mathbf{\frac{2}{\pi} \approx 0.637}$. Minimizing only frame-level DFT magnitude error leaves each frame's phase $\vartheta_k$ independent, creating **phase-discontinuity clicks** at every frame boundary that a time-domain waveform discriminator immediately penalizes.
:::
:::

::: {.slide title="Concept Check: Blurring, Distillation, and Phase (cont.)"}
[Check Your Understanding]{.kicker}

3. **Selecting the right training paradigm:** Among (a) pretraining a general text-to-image foundation model from scratch, (b) distilling a 50-step diffusion model into a 1-step generator, and (c) training a latent diffusion image tokenizer decoder, which tasks use an adversarial critic today?

::: {.d2l-note .fragment .fade-up}
**Answer:**

3. **(a)** uses multi-step diffusion/flow regression (for broad coverage and stable scaling), whereas **(b)** one-step diffusion distillation (ADD/DMD2) and **(c)** latent autoencoder decoders (VQGAN/SD-VAE) both pair regression/perceptual anchors with an **adversarial critic** to prevent conditional-mean blurring.
:::
:::

::: {.slide title="Summary: Adversarial Losses Beyond GANs"}
[Key Takeaways]{.kicker}

| Application / Concept | Core Mathematical Identity | Why Regression Alone Fails | Role of the Adversarial Critic |
| :--- | :--- | :--- | :--- |
| **Conditional-Mean Decomposition** | $\mathbb{E}[(x - g(c))^2] = \mathrm{Var}(x \mid c)$ $+ \mathbb{E}[(\mu(c) - g(c))^2]$ | Unique $L_2$ minimizer $\mu(c) = \mathbb{E}[x \mid c]$ averages across distinct modes | Penalizes low-density corridors ($D^\star = \log\frac{p}{q} \ll 0$) to force mode commitment |
| **Perception–Distortion Trade-Off** | $\mathrm{MSE}(\hat{x} \sim p(\cdot \mid c)) = 2\,\mathrm{MSE}(\mu(c))$ | Optimizing PSNR/MSE ($-3\text{ dB}$ for exact samples) rewards blurry averages | Trades pointwise distortion for distributional realism ($q(x \mid c) = p(x \mid c)$) |
| **One-Step Diffusion Distillation** | $D_\tau^\star(x_\tau) = \mathbb{E}[x_0 \mid x_\tau]$ (teacher posterior mean) | High-noise teacher targets contract one-step students into the conditional mean | ADD & DMD2 pair teacher/score regression with a feature critic on real images |
| **Image Tokenizers (VQGAN / VAE)** | $\mathcal{L}_{L_1} + \lambda_{\textrm{perc}}\mathcal{L}_{\textrm{LPIPS}}$ $+ \lambda_{\textrm{adv}}\mathcal{L}_{\textrm{PatchGAN}}$ | $8\times$ latent bottleneck leaves fine local textures ambiguous under $L_1$/LPIPS | PatchGAN critic enforces crisp high-frequency textures across local patches |
| **Speech Vocoders (HiFi-GAN)** | $\mathbb{E}_\vartheta[\sin(\omega n + \vartheta)] = \frac{\sin a}{a}\sin(\omega n)$ | Unknown phase ($a = \pi$) collapses $L_2$ waveform predictions to silence ($0$) | Multi-period & multi-scale waveform critics enforce amplitude and phase continuity |
:::
