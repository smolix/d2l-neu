# Adversarial Losses Beyond GANs
:label:`sec_gan_beyond`

When predicting a single scalar such as a house price, minimizing mean squared error ($L_2$ loss) is standard practice. On conditional image generation problems—such as super-resolution (mapping a low-resolution thumbnail $c$ to a high-resolution image $x$) or sketch-to-photo translation (pix2pix, :citet:`Isola.Zhu.Zhou.ea.2017`)—pixelwise $L_2$ and $L_1$ regression losses produce blurry outputs. Many distinct high-resolution textures correspond to the same low-resolution input, and minimizing pixelwise squared error forces the network to output their conditional average $\mathbb{E}[x \mid c]$—a blur that belongs to none of the plausible sharp images. Pairing the regression loss with a discriminator $D(x, c)$ that scores whether the output is a realistic image for condition $c$ restores sharp high-frequency detail.

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
\mathbb{E}\big[(x - g(c))^2\big]
\;=\;
\mathbb{E}\big[(x - \mu(c))^2\big] \;+\; \mathbb{E}\big[(\mu(c) - g(c))^2\big].
$$
:eqlabel:`eq_beyond_condmean`

The first term is the irreducible conditional variance $\operatorname{Var}(x \mid c)$, and the second is the squared distance between $g(c)$ and $\mu(c)$. Thus, squared error is uniquely minimized by the conditional mean $g^\star(c) = \mu(c)$. When $p(x \mid c)$ is unimodal and concentrated, $\mu(c)$ is a representative target; when $p(x \mid c)$ is multimodal, $\mu(c)$ falls in the low-density region between modes. Replacing $L_2$ with $L_1$ loss $\mathbb{E}[|x - g(c)|]$ merely replaces the conditional mean with the coordinate-wise conditional median (Exercise :exref:`ex_adversarial-losses-mean-and-median`), which likewise lies outside the support of a symmetric bimodal distribution.

### A Bimodal Two-Band Benchmark
:label:`subsec_gan_beyond_capacity`

Consider a 1D conditional distribution with $c \sim \textrm{Uniform}[-1, 1]$ and two parallel data bands separated by an empty corridor:

$$
x \;=\; c + b + 0.05\,\xi,
\qquad
b \in \{-1, +1\} \textrm{ with probability } \tfrac12 \textrm{ each},
\quad \xi \sim \mathcal{N}(0, 1).
$$
:eqlabel:`eq_beyond_twoband`

The conditional mean $\mu(c) = c$ runs down the empty corridor between the bands $x = c \pm 1$, achieving $\mathbb{E}[(x - \mu(c))^2] = 1 + 0.05^2 = 1.0025$. By contrast, a predictor that commits to the upper band $g(c) = c + 1$ has squared error $\tfrac12(0) + \tfrac12(2^2) + 0.05^2 = 2.0025$—twice as large! Squared error therefore penalizes committing to a valid data mode.

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

Three numbers summarize each student on a grid of 201 inputs: the mean distance from its curve to the nearer band, the mean distance to the conditional mean $x = c$, and the fraction of inputs at which the curve is more than 0.25 from both bands. A plot then draws both curves over a sample of the data, with the conditional mean dashed.

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

```{.python .input #adversarial-losses-the-capacity-argument-3}
%%tab pytorch, jax
d2l.plot([c_data, cs, cs, cs],
         [x_data, curves['squared error'], curves['adversarial'], cs],
         '$c$', '$x$', ['data', 'squared-error student',
                        'adversarial student', '$x = c$'],
         ylim=(-2.2, 3.6), fmts=('C0.', 'C2-', 'C1-', 'k--'),
         figsize=(5.5, 3.2))
```

As predicted, the squared-error model `g_mse` tracks the conditional mean $\mu(c) = c$ down the empty corridor ($1.00$ unit from either band, with $100\%$ of predictions off the bands). The adversarial model `g_adv` commits to valid data bands (mean distance $0.03$ from the nearest band, with only $1\text{--}2\%$ of points crossing the corridor).

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

1. *Regression only*: minimizing $\|\hat x - \operatorname{stop\_grad}(\text{teacher}(\hat x + \tau\xi, c, \tau))\|^2$. Because the slope of :eqref:`eq_beyond_teacher` in $\tilde x$ is at most $(\tau^2 + 0.05^2)^{-1} \approx \tfrac14 < 1$ at $\tau = 2$, the fixed-point equation $u = \mathbb{E}_\xi[\tanh((u + \sqrt{\tau^2 + 0.05^2}\,\xi)/(\tau^2 + 0.05^2))]$ for the offset $u = g(c, z) - c$ has the unique contraction fixed point $u = 0$ (the empty corridor!).
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

- *Regression onto the high-noise teacher* collapses $100\%$ of outputs into the empty corridor ($1.00$ unit from either band), ignoring the latent $z$.
- *Critic only* and *Critic + regression* both place $98\text{--}99\%$ of outputs directly on the two data bands (mean distance $0.03\text{--}0.04$), splitting probability mass evenly ($\sim 49\% / 49\%$) between the upper and lower bands via the latent $z$. Because the bands have narrow width $\sigma = 0.05$, the optimal critic's restoring slope ($\delta / 0.05^2 = 400\delta$) easily dominates the gentle inward pull of the high-noise teacher.

### Distillation, Autoencoders, and Vocoders at Scale

This combination of a regression or perceptual anchor with an adversarial discriminator appears across three major classes of modern generative systems:

1. **Adversarial Diffusion Distillation (ADD and DMD2):** ADD :cite:`Sauer.Lorenz.Blattmann.ea.2023` trains a one-step image generator by combining a denoising teacher regression loss with a discriminator built on frozen DINOv2 features, finding the same ordering as our experiment above (teacher regression alone produces blurry images, while adding the discriminator yields sharp one-step samples). Distribution Matching Distillation (DMD, :citet:`Yin.Gharbi.Zhang.ea.2024`) uses the difference between a frozen teacher's score $\nabla_x \log p_\tau$ and a fake-distribution diffusion model's score $\nabla_x \log q_\tau$ to descend $D_{\textrm{KL}}(q_\tau \,\|\, p_\tau)$, and DMD2 :cite:`Yin.Gharbi.Park.ea.2024` adds an adversarial discriminator against real images so the one-step student can exceed the teacher's quality.
2. **Latent Image Autoencoders (Tokenizers):** Latent diffusion models (:numref:`sec_diffusion-latent`) compress images into a lower-dimensional latent space using an autoencoder (such as VQGAN or the Stable Diffusion VAE, :citet:`Esser.Rombach.Ommer.2021,Rombach.Blattmann.Lorenz.ea.2022`). Because the compressed latent code leaves fine-grained local texture underdetermined, training the decoder with pixelwise $L_1/L_2$ and perceptual (LPIPS) losses is augmented with a patch-level discriminator (PatchGAN) to ensure crisp high-frequency textures.
3. **Neural Speech Vocoders:** Text-to-speech pipelines typically predict a mel-spectrogram (which specifies frame-level frequency magnitudes but discards phase) and use a vocoder such as HiFi-GAN :cite:`Kong.Kim.Bae.2020` to invert the spectrogram into an audio waveform. Because the expected value of a sinusoid $\sin(\omega n + \vartheta)$ over an unknown phase $\vartheta \sim \textrm{Uniform}[-a, a]$ shrinks in amplitude by $\sin(a)/a$ (vanishing completely for $\vartheta \sim \textrm{Uniform}[-\pi, \pi]$, Exercise :exref:`ex_adversarial-losses-phase`), waveform regression collapses to silence; HiFi-GAN instead pairs a multi-resolution spectrogram $L_1$ loss with multi-period and multi-scale waveform discriminators.

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

- Minimizing squared error $\mathbb{E}[\|x - g(c)\|^2]$ forces a predictor toward the conditional mean $\mu(c) = \mathbb{E}[x \mid c]$ (:eqref:`eq_beyond_condmean`), which averages across distinct modes whenever the target $x$ is ambiguous given $c$.
- An adversarial discriminator $D(x, c)$ estimates $\log(p(x, c)/q(x, c))$, penalizing predictions in low-density corridors between modes and forcing outputs onto the data manifold (at the cost of up to a $2\times$ increase in MSE / $3\text{ dB}$ drop in PSNR; :citet:`Blau.Michaeli.2017`).
- Because a diffusion denoiser predicts the conditional mean $\mathbb{E}[x_0 \mid x_\tau]$, distilling a one-step generator by regressing directly onto a high-noise denoiser collapses to the conditional average unless paired with an adversarial discriminator.
- For the same reason, adversarial discriminators remain a key component of modern latent image autoencoders (tokenizers), neural audio vocoders, and one-step distilled diffusion models.

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

Adversarial losses beyond GANs<br>
**conditional means · two bands · distillation · tokenizers and vocoders · where GANs stand**
:::
:::

::: {.slide title="Squared Error Turns Many Sharp Answers into One Blur"}
[The problem]{.kicker}

- Enlarge a thumbnail: many sharp photographs shrink to the same thumbnail.
- Squared error rewards the prediction closest to all of them on average,
  their pixelwise mean: a blur that matches none of them.
- Two bands $x = c \pm 1$: the squared-error fit runs between them, about
  **one unit** from every data point.

::: {.d2l-note .warn}
Regression now trains most image and video generators. Why do one-step
students, tokenizers and vocoders still keep a critic?
:::
:::

::: {.slide title="By the End of This Deck You Can"}
[Goals]{.kicker}

- derive the conditional-mean decomposition of squared error;
- predict where a squared-error and an adversarial student land on two bands;
- derive a denoising teacher in closed form and predict where regression
  onto it lands;
- explain why autoencoders and vocoders keep a critic;
- say what GANs still offer, and name their four limits.

**The idea:** a pointwise loss averages over ambiguity, a learned comparison
does not. **Running example:** two bands, one input, two answers.
:::

::: {.slide title="The Last Deck Asks Where the Comparison Survives"}
[Where this fits]{.kicker}

::: {.cols .vc}
::: {.col}
| section | asks |
|---|---|
| Generative Adversarial Networks | what a classifier of samples computes |
| Adversarial Objectives and Divergences | which loss gives which distance |
| Relativistic Objectives | what changes when the critic ranks pairs |
| Gradient Penalties and Convergence | why training circles, and what stops it |
| Adversarial Image Generation | which network, and how to score images |
| Conditional Generation | how to ask for a class |
| **Adversarial Losses Beyond GANs** | **where a learned comparison remains** |
:::

::: {.col .narrow}
- **Uses:** $D^\star = \log(p/q)$; the non-saturating loss; the games of
  the chapter.
- **Skips:** how diffusion models are trained; architectures.
:::
:::
:::

::: {.slide title="Squared Error Grades Distance to the Conditional Mean"}
[Conditional means]{.kicker}

For any predictor $g$ of a sample $x$ from a condition $c$, with
$\mu(c) = \mathbb{E}[x \mid c]$, the cross term vanishes given $c$:

$$\mathbb{E}\big[(x - g(c))^2\big]
= \underbrace{\mathbb{E}\big[(x - \mu(c))^2\big]}_{\textrm{no } g \textrm{ removes it}}
+ \mathbb{E}\big[(\mu(c) - g(c))^2\big]$$

- The best predictor is $g = \mu$; for images, pixel by pixel.
- Several answers: $\mu$ is their average, which need not be an answer.
- Absolute error ($L_1$, pix2pix) gives a median: no sharper.
:::

::: {.slide title="On Two Bands the Mean Lies Where No Data Is"}
[Two bands]{.kicker}

$c \sim U[-1, 1]$, $\;x = c + b + 0.05\,\xi$, $\;b = \pm 1$ with
probability $\tfrac12$ each, independent of $c$.

- Bands $x = c \pm 1$, forty noise widths apart; $\mu(c) = c$, in the corridor.
- Expected squared error of $\mu$: $1 + 0.05^2 \approx 1.0025$.
- Of the band $x = c + 1$: $\tfrac12 \cdot 4 + 0.05^2 \approx 2.0025$.

::: {.d2l-note}
Squared error prefers the corridor, where no data point lies.
:::
:::

::: {.slide title="A Critic Scores Whether a Prediction Lies among the Data"}
[Two bands]{.kicker}

Critic $D(x, c)$, trained by logistic regression on data pairs against
predicted pairs $(\hat x, c)$:

$$D^\star(x, c) = \log \frac{p(x, c)}{q(x, c)}$$

- In the corridor $p \approx 0$: $D^\star$ is strongly negative.
- The student minimizes $-\log \sigma\big(D(\hat x, c)\big)$, the
  non-saturating loss.
- $\hat x = g(c) + 0.05\,\xi'$: the data's noise, so only the location of
  mass differs.
:::

::: {.slide title="One Step of the Game: Critic First, Then the Student" only="pytorch"}
[Two bands]{.kicker}

```
c, x = make_bands(batch_size)                   # a fresh batch
x_hat = g(c) + 0.05 * torch.randn(batch_size, 1)
real, fake = torch.cat([c, x], 1), torch.cat([c, x_hat], 1)
loss_D = softplus(-D(real)) + softplus(D(fake.detach()))
opt_D.zero_grad(); loss_D.mean().backward(); opt_D.step()   # critic
loss_G = softplus(-D(fake)).mean()
opt_G.zero_grad(); loss_G.backward(); opt_G.step()          # student
```

Same 32-unit network for both students, same initial weights;
$\mathrm{softplus}(-t) = -\log \sigma(t)$.
:::

::: {.slide title="One Step of the Game: Critic First, Then the Student" only="jax"}
[Two bands]{.kicker}

```
c, x = make_bands(k_data, batch_size)               # a fresh batch
xi = jax.random.normal(k_xi, (batch_size, 1))
real = jnp.concatenate([c, x], 1)
fake = lambda g_: jnp.concatenate([c, g_(c) + 0.05 * xi], 1)
loss_D = lambda D_: (softplus(-D_(real))
                     + softplus(D_(fake(g)))).mean()
opt_D.update(D, nnx.grad(loss_D)(D))
loss_G = lambda g_: softplus(-D(fake(g_))).mean()
opt_G.update(g, nnx.grad(loss_G)(g))
```

Same 32-unit network for both students, same initial weights;
$\mathrm{softplus}(-t) = -\log \sigma(t)$.
:::

::: {.slide title="The Adversarial Student Commits to a Band"}
[Two bands]{.kicker}

@!adversarial-losses-the-capacity-argument-3

Squared error: on average within about 0.02 of $x = c$, one unit from every band.
Adversarial: a few hundredths from a band; at most 2% of inputs off the bands,
where the curve crosses. Where it crosses is left to the run.
:::

::: {.slide title="Check: Which Predictor Fools the Critic?"}
[Check]{.kicker}

Three predictors on the two bands: the mean $\mu(c) = c$, the band
$x = c + 1$, and an exact sampler $\hat x \sim p(x \mid c)$. Rank them by
squared error. Which one can an optimal critic not tell from the data?

. . .

- Mean $\approx 1.0025$ $<$ band $\approx 2.0025$ $<$ sampler $\approx 2.005$.
- The sampler's error is twice the mean's, $10 \log_{10} 2 \approx 3$ dB less
  peak signal-to-noise ratio.
- Only the sampler matches the data; squared error ranks it last. In
  general, realism costs at most a factor of two in squared error.

[Blau and Michaeli 2018]{.small}
:::

::: {.slide title="A Bigger Network Cannot Pick a Band; a Latent Bit Can"}
[Ambiguity and capacity]{.kicker}

- $c$ carries no information about the band: no deterministic $g(c)$, of any
  size, picks the right band more than half the time.
- Add a random latent bit $z = \pm 1$: $g(c, z)$ can send each value to one
  band.
- **Ambiguity:** the input does not fix the output (thumbnail, bands, phase).
- **Capacity:** it does, but one pass cannot compute the map (a one-step
  network imitating a many-step sampler, whose map jumps).

::: {.d2l-note}
Squared error averages over what the model cannot tell apart. More capacity
fixes only the second case.
:::
:::

::: {.slide title="A Squared-Error Denoiser Returns the Mean of the Clean Image"}
[Distillation]{.kicker}

Train $g(x_\tau) \approx x_0$ by squared error, $x_\tau = x_0$ plus noise of
level $\tau$. With $(c, x) = (x_\tau, x_0)$:

$$g^\star(x_\tau) = \mathbb{E}[x_0 \mid x_\tau]$$

- Low noise: $x_\tau$ nearly fixes $x_0$, the mean is sharp.
- High noise: many images fit $x_\tau$, the mean is a blur, the corridor.
- A sampler steps a little toward the mean and asks again: many evaluations.
- **Distillation:** a one-step student learns from this teacher.
:::

::: {.slide title="On Two Bands the Teacher Is a Tanh"}
[Distillation]{.kicker}

A denoiser trained on data reads every input as noised data,
$\tilde x = x + \tau \xi$, including a noised student output
$\hat x + \tau \xi$. Given $b$, data give
$\tilde x \sim \mathcal{N}(c + b, \tau^2 + 0.05^2)$; odds of $b = +1$:
$\exp\big(2(\tilde x - c)/(\tau^2 + 0.05^2)\big)$.

$$\mathbb{E}[c + b \mid \tilde x, c] = c + \tanh\Big(\frac{\tilde x - c}{\tau^2 + 0.05^2}\Big)$$

- The exact posterior mean of the band center.
- Small $\tau$: a step, snaps to the nearer band.
- $\tau = 2$: slope at most $\tfrac14$, leans only gently.
:::

::: {.slide title="A High-Noise Teacher Pulls Every Output into the Corridor"}
[Distillation]{.kicker}

Students now take a latent $z \sim \mathcal{N}(0, 1)$ besides $c$. Student
(i) regresses onto the teacher at $\tau = 2$, gradient stopped. The offset
$u = g(c, z) - c$ settles where

$$u = \mathbb{E}_\xi\Big[\tanh\Big(\frac{u + \sqrt{\tau^2 + 0.05^2}\,\xi}{\tau^2 + 0.05^2}\Big)\Big]$$

- Right side: slope at most $\tfrac14$, so one solution, $u = 0$.
- Prediction: every output in the corridor, whatever the latent $z$.

::: {.d2l-note}
A teacher that returns averages teaches averages.
:::
:::

::: {.slide title="A Critic, with or without the Teacher, Puts the Student on Both Bands"}
[Distillation]{.kicker}

Students $g(c, z)$, $z \sim \mathcal{N}(0, 1)$; 4,000 steps; three seeds each.

| student | from nearer band | on each band | off the bands |
|:--|--:|--:|--:|
| (i) regression onto the teacher | about 1 | 0% | 100% |
| (ii) critic | a few hundredths | about half | 1--2% |
| (iii) critic + regression | a few hundredths | about half | 1--2% |

Why (iii) stays: the regression pulls with about $2(1 - 0.2) = 1.6$ at a
band; a shift $\delta$ draws a critic push of about
$\tfrac12 \delta / 0.05^2 = 200\,\delta$; they balance at $\delta \approx 0.008$.

::: {.d2l-note}
(ii) and (iii) differ by less than the run-to-run range: with a latent, the
critic covers both bands.
:::
:::

::: {.slide title="At Scale, ADD Is Student (iii)"}
[Distillation]{.kicker}

- One-step text-to-image student, two losses: regression onto a frozen
  teacher's denoised estimate of its own noised output, and a critic against
  real images.
- The critic reads frozen DINOv2 features and trains only small heads.
- The paper's ablation has our ordering: regression alone unusable, critic
  alone good, both about as good, the regression adding prompt agreement.

[Sauer et al. 2023]{.small}
:::

::: {.slide title="DMD Follows the Optimal Critic's Gradient without Training a Classifier"}
[Distillation]{.kicker}

Score: $\nabla_x \log$ density. $p_\tau$, $q_\tau$: data and student
samples, noised to level $\tau$.

- Teacher supplies $\nabla_x \log p_\tau$; a second diffusion model, trained
  on the student's samples, supplies $\nabla_x \log q_\tau$.
- Move each sample along
  $\nabla_x \log p_\tau - \nabla_x \log q_\tau = \nabla_x \log(p_\tau/q_\tau)$,
  the optimal log-loss critic's input gradient.
- It descends $D_{\textrm{KL}}(q_\tau \,\|\, p_\tau)$, cheap on missing
  modes; DMD2 adds a critic on real images.

[Yin et al. 2024, DMD and DMD2]{.small}
:::

::: {.slide title="Texture Is Ambiguous, So Autoencoders Keep a Patch Critic"}
[Tokenizers and vocoders]{.kicker}

The code fixes the layout, not the texture. The decoder trains on three terms:

| decoder loss | what it does |
|:--|:--|
| pixel | layout right; alone a blur |
| perceptual (LPIPS: features of a pretrained classifier) | matches the mean feature; a sharp image fits, a bad texture too |
| patch critic | rejects what still looks smeared |

[Esser et al. 2021; Rombach et al. 2022]{.small}
:::

::: {.slide title="Check: What Does a Squared-Error Vocoder Output?"}
[Check]{.kicker}

A vocoder turns a spectrogram into a waveform. Model a frame as
$x_n = \sin(\omega n + \vartheta)$: the spectrogram fixes $\omega$, not the
phase $\vartheta$. With $\vartheta$ uniform on $[-\pi, \pi]$, what does the
squared-error prediction sound like? And for $[-\pi/2, \pi/2]$?

. . .

- $\mathbb{E}_\vartheta[x_n] = \frac{\sin a}{a}\sin(\omega n)$ for $\vartheta$ uniform on $[-a, a]$.
- $a = \pi$: silence. $a = \pi/2$: $2/\pi \approx 0.64$ of the amplitude.
- HiFi-GAN: a spectrogram loss for the content, critics on the waveform
  against the shrunken frame and broken frame boundaries.

[Kong et al. 2020]{.small}
:::

::: {.slide title="Three Independent Choices Describe Every Game of the Chapter"}
[The games]{.kicker}

| objective | critic estimates | does | constrained by | disjoint supports |
|:--|:--|:--|:--|:--|
| log loss | $\log(p/q)$ | classifies | nothing | saturates |
| hinge | sign of $\log(p/q)$ | classifies | nothing | saturates |
| MMD | smoothed $p - q$ | witnesses | kernel ball | slope, up to bandwidth |
| $W_1$ | potential | witnesses | Lipschitz ball | slope one |
| RpGAN | $\log(p/q)$ + const | ranks | nothing | saturates |
| R3GAN | $\log(p/q)$ above $2\sqrt\gamma$ | ranks | input-gradient penalty | only via critic smoothness |

[Potential, witness: the critics of the $W_1$ and MMD games; $\gamma$: weight
of the $R_1$ and $R_2$ penalties. Last column: best critic of its class.]{.small}
:::

::: {.slide title="GANs Lost Broad Image Synthesis but Keep One-Step Sampling"}
[Where GANs stand]{.kicker}

::: {.cols}
::: {.col}
**Why regression won broad targets**

- covered the distribution
- improved steadily with data and size
- little tuning
:::

::: {.col}
**What the game still offers**

- a sample in one network evaluation
- sharp where the output is ambiguous
- competitive on narrow domains
:::
:::

::: {.d2l-note}
Train one today for a one-pass decoder or vocoder, a one- to four-step
student, or a narrow domain under a real-time budget.
:::
:::

::: {.slide title="Four Limits of the Game Remain after the Repairs"}
[Where GANs stand]{.kicker}

1. **No density:** the generator cannot say how likely an image is.
2. **Two networks and a tuned penalty:** convergence needs a penalty weight
   chosen for the data.
3. **No push toward a missing mode:** a dropped half costs about a quarter
   of a nat; the pairing guarantee holds only for a fixed coupling, which
   none of our image runs used.
4. **Instability at scale:** BigGAN runs collapse late, results come from
   checkpoints before it; GigaGAN and StyleGAN-T needed new architecture.

[Brock et al. 2019; Kang et al. 2023; Sauer et al. 2023]{.small}
:::

::: {.slide title="Check: Which of These Would You Train with a Critic Today?"}
[Check]{.kicker}

A text-to-image model from scratch; a vocoder; a one-step student of a
diffusion model; a sprite generator for a game that must run at 60 frames
per second.

. . .

- Text-to-image from scratch: regression; broad targets favor it.
- Vocoder, one-step student: a critic, beside a regression term.
- Sprite generator: a GAN is a good fit; narrow domain, one evaluation.
:::

::: {.slide title="What Remains after the Repairs"}
[Wrap-up]{.kicker}

- Two bands: squared error within about 0.02 of $x = c$, a unit from the
  data; the critic keeps the student on a band.
- Teacher at $\tau = 2$: regression alone about one unit from a band; with
  a critic a few hundredths, about half on each band, regression term or not.
- Autoencoders and vocoders keep a critic for texture and phase.
- GANs keep one-step sampling; four limits remain: no density, two
  networks and a tuned penalty, no push toward a missing mode, instability
  at scale.

**Next:** diffusion and flow models give up the game and train by regression
on a target computed from each training example.
:::
